#!/usr/bin/env python3
"""
CSSI Research: Systematic investigation of why CSSI fails on real Geneformer attention.
Tests: per-layer, per-head, aggregation methods, normalization, thresholding.
"""
import os, sys, pickle, json, time, traceback
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from sklearn.metrics import roc_auc_score
from collections import Counter

OUT_DIR = "/mnt/d/openclaw/biodyn-nmi-paper/cssi_real"
DATA_PATH = "/mnt/d/openclaw/intelligence-augmentation/data/brain_scrna/DLPFC_11k.h5ad"
TOP_K = 1000
MAX_SEQ = 256
N_CELLS = 500  # Fixed for research

log_file = open(os.path.join(OUT_DIR, "cssi_research.log"), "w", buffering=1)
def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

log("=== CSSI Research Starting ===")

# Load dictionaries
import geneformer
GF_DIR = os.path.dirname(geneformer.__file__)
with open(os.path.join(GF_DIR, "token_dictionary_gc104M.pkl"), "rb") as f:
    token_dict = pickle.load(f)
with open(os.path.join(GF_DIR, "gene_median_dictionary_gc104M.pkl"), "rb") as f:
    gene_median_dict = pickle.load(f)
with open(os.path.join(GF_DIR, "gene_name_id_dict_gc104M.pkl"), "rb") as f:
    gene_name_id_dict = pickle.load(f)
id_to_token = {int(v): k for k, v in token_dict.items()}

def tokenize_cell(expr_vec, gene_ids, max_len=2048):
    if sparse.issparse(expr_vec):
        expr_vec = expr_vec.toarray().flatten()
    nonzero = expr_vec > 0
    genes = gene_ids[nonzero]
    vals = expr_vec[nonzero]
    valid = np.array([g in token_dict and g in gene_median_dict for g in genes])
    if valid.sum() == 0:
        return []
    genes, vals = genes[valid], vals[valid]
    medians = np.array([max(gene_median_dict[g], 1e-6) for g in genes], dtype=np.float32)
    idx = np.argsort(-(vals / medians))[:max_len]
    return [int(token_dict[genes[i]]) for i in idx]


def extract_attention_per_layer_head(model, token_ids_list, device, batch_size=1):
    """Returns per-cell: (token_ids, attn_dict) where attn_dict[(layer,head)] = (seq,seq) matrix."""
    model.eval()
    results = []
    for bs in range(0, len(token_ids_list), batch_size):
        batch = [t[:MAX_SEQ] for t in token_ids_list[bs:bs+batch_size]]
        ml = max(len(t) for t in batch)
        padded = [t + [0]*(ml-len(t)) for t in batch]
        masks = [[1]*len(t) + [0]*(ml-len(t)) for t in batch]
        
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attn_mask = torch.tensor(masks, dtype=torch.long, device=device)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            out = model.bert(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
        
        # out.attentions is tuple of (batch, heads, seq, seq) per layer
        n_layers = len(out.attentions)
        n_heads = out.attentions[0].shape[1]
        
        for i, t in enumerate(batch):
            n = len(t)
            cell_attn = {}
            for layer in range(n_layers):
                for head in range(n_heads):
                    cell_attn[(layer, head)] = out.attentions[layer][i, head, :n, :n].float().cpu().numpy()
            results.append((t, cell_attn))
        
        del out
        torch.cuda.empty_cache()
        
        done = min(bs+batch_size, len(token_ids_list))
        if done % 50 < batch_size:
            log(f"  Attention: {done}/{len(token_ids_list)}")
    
    return results


def build_gene_matrix(attention_results, layer_heads, top_genes=None, aggregation="mean"):
    """
    Build gene-gene matrix from attention, using specified layer/head combos.
    layer_heads: list of (layer, head) tuples, or "all" 
    aggregation: how to combine across layer_heads - "mean" or "max"
    top_genes: pre-computed list of top gene token IDs
    """
    if top_genes is None:
        freq = Counter()
        for tids, _ in attention_results:
            for t in tids:
                if t > 3:
                    freq[t] += 1
        top_genes = [g for g, _ in freq.most_common(TOP_K)]
    
    g2i = {g: i for i, g in enumerate(top_genes)}
    n = len(top_genes)
    
    score_sum = np.zeros((n, n), dtype=np.float64)
    count_mat = np.zeros((n, n), dtype=np.float64)
    
    for tids, cell_attn in attention_results:
        mapped = [(pos, g2i[tid]) for pos, tid in enumerate(tids) if tid in g2i]
        if len(mapped) < 2:
            continue
        
        positions = np.array([m[0] for m in mapped])
        indices = np.array([m[1] for m in mapped])
        
        # Average attention across specified layer/heads
        sub_attns = []
        for lh in layer_heads:
            attn = cell_attn[lh]
            sub_attns.append(attn[np.ix_(positions, positions)])
        
        if aggregation == "mean":
            sub_attn = np.mean(sub_attns, axis=0)
        elif aggregation == "max":
            sub_attn = np.max(sub_attns, axis=0)
        else:
            sub_attn = np.mean(sub_attns, axis=0)
        
        # Accumulate
        for a, i in enumerate(indices):
            for b, j in enumerate(indices):
                if i != j:
                    score_sum[i, j] += sub_attn[a, b]
                    count_mat[i, j] += 1
    
    mask = count_mat > 0
    result = np.zeros_like(score_sum)
    result[mask] = score_sum[mask] / count_mat[mask]
    
    gene_list = [id_to_token.get(g, f"UNK_{g}") for g in top_genes]
    return result, gene_list


def build_cssi_matrices(attention_results, cell_types, layer_heads, top_genes=None):
    """Build CSSI max/mean/median matrices per cell type."""
    if top_genes is None:
        freq = Counter()
        for tids, _ in attention_results:
            for t in tids:
                if t > 3:
                    freq[t] += 1
        top_genes = [g for g, _ in freq.most_common(TOP_K)]
    
    unique_cts = [ct for ct in set(cell_types) if sum(1 for c in cell_types if c == ct) >= 5]
    
    ct_matrices = {}
    ct_counts = {}
    for ct in unique_cts:
        mask = [i for i, c in enumerate(cell_types) if c == ct]
        ct_results = [attention_results[i] for i in mask]
        mat, genes = build_gene_matrix(ct_results, layer_heads, top_genes)
        ct_matrices[ct] = mat
        ct_counts[ct] = len(mask)
    
    if not ct_matrices:
        return None, None, None, None
    
    gene_list = [id_to_token.get(g, f"UNK_{g}") for g in top_genes]
    n = len(top_genes)
    
    all_mats = np.stack(list(ct_matrices.values()))
    cssi_max = np.max(all_mats, axis=0)
    
    weights = np.array([ct_counts[ct] for ct in ct_matrices])
    cssi_mean = np.average(all_mats, axis=0, weights=weights)
    
    cssi_median = np.median(all_mats, axis=0)
    
    # Variance-weighted: weight each cell type by its variance (more variable = more informative)
    ct_vars = np.var(all_mats, axis=(1, 2))
    if ct_vars.sum() > 0:
        cssi_var_weighted = np.average(all_mats, axis=0, weights=ct_vars)
    else:
        cssi_var_weighted = cssi_mean
    
    return cssi_max, cssi_mean, cssi_median, cssi_var_weighted, gene_list


def evaluate_auroc(score_matrix, gene_list, gt_edges):
    """Quick AUROC evaluation."""
    gene_set = set(gene_list)
    g2i = {g: i for i, g in enumerate(gene_list)}
    n = len(gene_list)
    
    y_true = []
    y_score = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            label = 1 if (gene_list[i], gene_list[j]) in gt_edges else 0
            y_true.append(label)
            y_score.append(score_matrix[i, j])
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    n_pos = y_true.sum()
    
    if n_pos < 3:
        return None, n_pos
    
    return float(roc_auc_score(y_true, y_score)), int(n_pos)


def apply_transforms(mat):
    """Return dict of transformed matrices."""
    transforms = {"raw": mat}
    
    # Symmetrize
    transforms["sym"] = (mat + mat.T) / 2
    
    # Row-normalize (subtract row mean)
    row_mean = mat.mean(axis=1, keepdims=True)
    transforms["row_norm"] = mat - row_mean
    
    # Z-score per row
    row_std = mat.std(axis=1, keepdims=True)
    row_std[row_std == 0] = 1
    transforms["zscore"] = (mat - row_mean) / row_std
    
    # Log transform (shift to positive first)
    m = mat.copy()
    m = m - m.min() + 1e-10
    transforms["log"] = np.log(m)
    
    # Top-k sparsify (keep top 10 per row)
    sparse_mat = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        top_idx = np.argsort(mat[i])[-10:]
        sparse_mat[i, top_idx] = mat[i, top_idx]
    transforms["sparse_top10"] = sparse_mat
    
    # Product of forward and reverse (TF->target * target->TF)
    transforms["product"] = mat * mat.T
    
    # Difference from mean (CSSI-like within pooled)
    global_mean = mat.mean()
    transforms["diff_from_mean"] = mat - global_mean
    
    return transforms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    
    from transformers import BertForMaskedLM
    log("Loading model...")
    model = BertForMaskedLM.from_pretrained("ctheodoris/Geneformer", output_attentions=True).to(device).half()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    log(f"Model: {n_layers}L x {n_heads}H")
    
    log("Loading data...")
    adata = sc.read_h5ad(DATA_PATH)
    
    gene_name_to_ensembl = {}
    if 'feature_name' in adata.var.columns:
        for ens_id, row in adata.var.iterrows():
            name = row['feature_name']
            if pd.notna(name) and name != '':
                gene_name_to_ensembl[name] = ens_id
    for gname, ens_id in gene_name_id_dict.items():
        gene_name_to_ensembl.setdefault(gname, ens_id)
    
    # Load TRRUST
    log("Loading TRRUST...")
    try:
        trrust_path = "/tmp/trrust.tsv"
        if not os.path.exists(trrust_path):
            import urllib.request
            urllib.request.urlretrieve(
                "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv", trrust_path)
        df = pd.read_csv(trrust_path, sep='\t', header=None, names=['TF', 'target', 'mode', 'pmid'])
        trrust = set()
        for _, row in df.iterrows():
            src = gene_name_to_ensembl.get(row['TF'])
            tgt = gene_name_to_ensembl.get(row['target'])
            if src and tgt:
                trrust.add((src, tgt))
        log(f"TRRUST: {len(trrust)} edges")
    except Exception as e:
        log(f"TRRUST download failed: {e}, trying local...")
        trrust = set()
    
    # Sample cells
    ct_counts = adata.obs['cell_type'].value_counts()
    valid_cts = ct_counts[ct_counts >= 50].index.tolist()
    adata_filt = adata[adata.obs['cell_type'].isin(valid_cts)].copy()
    gene_ids = np.array(adata_filt.var_names)
    
    np.random.seed(42)
    sampled = []
    for ct in valid_cts:
        idx = np.where(adata_filt.obs['cell_type'] == ct)[0]
        n_ct = max(1, int(N_CELLS * len(idx) / adata_filt.n_obs))
        sampled.extend(np.random.choice(idx, min(n_ct, len(idx)), replace=False))
    if len(sampled) > N_CELLS:
        sampled = list(np.random.choice(sampled, N_CELLS, replace=False))
    
    adata_s = adata_filt[sampled].copy()
    log(f"Sampled: {adata_s.n_obs} cells")
    
    # Tokenize
    log("Tokenizing...")
    tokens = []
    for i in range(adata_s.n_obs):
        expr = adata_s.X[i].toarray().flatten() if sparse.issparse(adata_s.X) else adata_s.X[i].flatten()
        tokens.append(tokenize_cell(expr, gene_ids))
    
    valid_idx = [i for i, t in enumerate(tokens) if len(t) > 10]
    tokens = [tokens[i] for i in valid_idx]
    cell_types = adata_s.obs['cell_type'].values[valid_idx]
    log(f"Valid: {len(tokens)} cells")
    
    # Get top genes (shared across all experiments)
    freq = Counter()
    for t in tokens:
        for tid in t:
            if tid > 3:
                freq[tid] += 1
    top_genes = [g for g, _ in freq.most_common(TOP_K)]
    
    # Extract attention PER LAYER PER HEAD
    log("Extracting per-layer-head attention...")
    t0 = time.time()
    attn_results = extract_attention_per_layer_head(model, tokens, device)
    log(f"Done in {time.time()-t0:.1f}s")
    
    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    
    results = {"metadata": {"n_cells": len(tokens), "n_layers": n_layers, "n_heads": n_heads}}
    
    # ===== EXPERIMENT 1: Per-layer AUROC =====
    log("\n=== EXPERIMENT 1: Per-layer attention ===")
    layer_results = {}
    all_heads_per_layer = {l: [(l, h) for h in range(n_heads)] for l in range(n_layers)}
    
    for layer in range(n_layers):
        mat, genes = build_gene_matrix(attn_results, all_heads_per_layer[layer], top_genes)
        auroc, n_pos = evaluate_auroc(mat, genes, trrust)
        layer_results[f"layer_{layer}"] = {"auroc": auroc, "n_pos": n_pos}
        if auroc:
            log(f"  Layer {layer:2d}: AUROC={auroc:.4f} ({n_pos} pos)")
    
    results["per_layer"] = layer_results
    
    # Find best layer
    best_layer = max(range(n_layers), key=lambda l: layer_results[f"layer_{l}"].get("auroc", 0) or 0)
    best_layer_auroc = layer_results[f"layer_{best_layer}"]["auroc"]
    log(f"  Best layer: {best_layer} (AUROC={best_layer_auroc:.4f})")
    
    # ===== EXPERIMENT 2: Per-head AUROC (for best 3 layers) =====
    log("\n=== EXPERIMENT 2: Per-head attention ===")
    sorted_layers = sorted(range(n_layers), 
                          key=lambda l: layer_results[f"layer_{l}"].get("auroc", 0) or 0, 
                          reverse=True)[:3]
    
    head_results = {}
    for layer in sorted_layers:
        for head in range(n_heads):
            mat, genes = build_gene_matrix(attn_results, [(layer, head)], top_genes)
            auroc, n_pos = evaluate_auroc(mat, genes, trrust)
            key = f"L{layer}_H{head}"
            head_results[key] = {"auroc": auroc, "n_pos": n_pos}
            if auroc and auroc > 0.55:
                log(f"  {key}: AUROC={auroc:.4f}")
    
    results["per_head"] = head_results
    best_head_key = max(head_results, key=lambda k: head_results[k].get("auroc", 0) or 0)
    log(f"  Best head: {best_head_key} (AUROC={head_results[best_head_key]['auroc']:.4f})")
    
    # ===== EXPERIMENT 3: Top-K heads by AUROC =====
    log("\n=== EXPERIMENT 3: Top-K heads combined ===")
    # Evaluate ALL heads
    all_head_aurocs = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            key = f"L{layer}_H{head}"
            if key in head_results:
                all_head_aurocs[(layer, head)] = head_results[key].get("auroc", 0) or 0
            else:
                mat, genes = build_gene_matrix(attn_results, [(layer, head)], top_genes)
                auroc, _ = evaluate_auroc(mat, genes, trrust)
                all_head_aurocs[(layer, head)] = auroc or 0
    
    sorted_heads = sorted(all_head_aurocs.items(), key=lambda x: x[1], reverse=True)
    
    topk_results = {}
    for k in [1, 3, 5, 10, 20, 50]:
        top_k_heads = [h for h, _ in sorted_heads[:k]]
        mat, genes = build_gene_matrix(attn_results, top_k_heads, top_genes)
        auroc, n_pos = evaluate_auroc(mat, genes, trrust)
        topk_results[f"top_{k}"] = {"auroc": auroc, "n_pos": n_pos, 
                                     "heads": [f"L{l}_H{h}" for l,h in top_k_heads]}
        log(f"  Top-{k} heads: AUROC={auroc:.4f}")
    
    results["topk_heads"] = topk_results
    
    # ===== EXPERIMENT 4: Transforms on best config =====
    log("\n=== EXPERIMENT 4: Matrix transforms ===")
    # Use best layer
    best_lh = all_heads_per_layer[best_layer]
    mat_best, genes_best = build_gene_matrix(attn_results, best_lh, top_genes)
    
    transform_results = {}
    transforms = apply_transforms(mat_best)
    for tname, tmat in transforms.items():
        auroc, n_pos = evaluate_auroc(tmat, genes_best, trrust)
        transform_results[tname] = {"auroc": auroc, "n_pos": n_pos}
        log(f"  {tname}: AUROC={auroc:.4f}")
    
    results["transforms_best_layer"] = transform_results
    
    # Also try transforms on top-1 head
    best_lh_single = [sorted_heads[0][0]]
    mat_best_head, _ = build_gene_matrix(attn_results, best_lh_single, top_genes)
    
    transform_results_head = {}
    transforms_h = apply_transforms(mat_best_head)
    for tname, tmat in transforms_h.items():
        auroc, n_pos = evaluate_auroc(tmat, genes_best, trrust)
        transform_results_head[tname] = {"auroc": auroc, "n_pos": n_pos}
        if auroc and auroc > 0.55:
            log(f"  best_head+{tname}: AUROC={auroc:.4f}")
    
    results["transforms_best_head"] = transform_results_head
    
    # ===== EXPERIMENT 5: CSSI with best layer =====
    log("\n=== EXPERIMENT 5: CSSI with best layer ===")
    cssi_max, cssi_mean, cssi_median, cssi_var, cssi_genes = build_cssi_matrices(
        attn_results, cell_types, best_lh, top_genes)
    
    pooled_mat, pooled_genes = build_gene_matrix(attn_results, best_lh, top_genes)
    
    cssi_layer_results = {}
    for name, mat in [("pooled", pooled_mat), ("cssi_max", cssi_max), 
                       ("cssi_mean", cssi_mean), ("cssi_median", cssi_median),
                       ("cssi_var_weighted", cssi_var)]:
        if mat is not None:
            auroc, n_pos = evaluate_auroc(mat, cssi_genes, trrust)
            cssi_layer_results[name] = {"auroc": auroc, "n_pos": n_pos}
            log(f"  {name}: AUROC={auroc:.4f}")
    
    # CSSI deviation: max deviation from pooled mean
    if cssi_max is not None:
        deviation = np.abs(cssi_max - pooled_mat)
        auroc, n_pos = evaluate_auroc(deviation, cssi_genes, trrust)
        cssi_layer_results["cssi_deviation"] = {"auroc": auroc, "n_pos": n_pos}
        log(f"  cssi_deviation: AUROC={auroc:.4f}")
        
        # CSSI range: max - min across cell types
        # Need to rebuild per-ct
        unique_cts = [ct for ct in set(cell_types) if sum(1 for c in cell_types if c == ct) >= 5]
        ct_mats = []
        for ct in unique_cts:
            mask = [i for i, c in enumerate(cell_types) if c == ct]
            ct_res = [attn_results[i] for i in mask]
            m, _ = build_gene_matrix(ct_res, best_lh, top_genes)
            ct_mats.append(m)
        
        if len(ct_mats) > 1:
            ct_stack = np.stack(ct_mats)
            cssi_range = np.max(ct_stack, axis=0) - np.min(ct_stack, axis=0)
            auroc, n_pos = evaluate_auroc(cssi_range, cssi_genes, trrust)
            cssi_layer_results["cssi_range"] = {"auroc": auroc, "n_pos": n_pos}
            log(f"  cssi_range: AUROC={auroc:.4f}")
            
            cssi_std = np.std(ct_stack, axis=0)
            auroc, n_pos = evaluate_auroc(cssi_std, cssi_genes, trrust)
            cssi_layer_results["cssi_std"] = {"auroc": auroc, "n_pos": n_pos}
            log(f"  cssi_std: AUROC={auroc:.4f}")
    
    results["cssi_best_layer"] = cssi_layer_results
    
    # ===== EXPERIMENT 6: CSSI with best head =====
    log("\n=== EXPERIMENT 6: CSSI with best single head ===")
    cssi_max_h, cssi_mean_h, cssi_median_h, cssi_var_h, cssi_genes_h = build_cssi_matrices(
        attn_results, cell_types, best_lh_single, top_genes)
    
    pooled_h, _ = build_gene_matrix(attn_results, best_lh_single, top_genes)
    
    cssi_head_results = {}
    for name, mat in [("pooled", pooled_h), ("cssi_max", cssi_max_h),
                       ("cssi_mean", cssi_mean_h), ("cssi_median", cssi_median_h)]:
        if mat is not None:
            auroc, n_pos = evaluate_auroc(mat, cssi_genes_h, trrust)
            cssi_head_results[name] = {"auroc": auroc, "n_pos": n_pos}
            log(f"  {name}: AUROC={auroc:.4f}")
    
    results["cssi_best_head"] = cssi_head_results
    
    # ===== EXPERIMENT 7: Combined best approaches =====
    log("\n=== EXPERIMENT 7: Combined approaches ===")
    # Best layer + transforms + CSSI
    combined_results = {}
    
    # Try: best layer, row-normalized, then CSSI
    for tname in ["raw", "sym", "row_norm", "zscore", "product", "sparse_top10"]:
        # Build per-ct with transform
        unique_cts = [ct for ct in set(cell_types) if sum(1 for c in cell_types if c == ct) >= 5]
        ct_mats_t = []
        for ct in unique_cts:
            mask = [i for i, c in enumerate(cell_types) if c == ct]
            ct_res = [attn_results[i] for i in mask]
            m, _ = build_gene_matrix(ct_res, best_lh, top_genes)
            ct_mats_t.append(apply_transforms(m)[tname])
        
        if len(ct_mats_t) > 1:
            ct_stack = np.stack(ct_mats_t)
            for cssi_agg, cssi_mat in [("max", np.max(ct_stack, axis=0)),
                                        ("std", np.std(ct_stack, axis=0)),
                                        ("range", np.max(ct_stack, axis=0) - np.min(ct_stack, axis=0))]:
                auroc, n_pos = evaluate_auroc(cssi_mat, genes_best, trrust)
                key = f"{tname}_{cssi_agg}"
                combined_results[key] = {"auroc": auroc, "n_pos": n_pos}
                if auroc and auroc > 0.55:
                    log(f"  {key}: AUROC={auroc:.4f}")
    
    results["combined"] = combined_results
    
    # ===== SUMMARY =====
    log("\n" + "="*60)
    log("SUMMARY: All AUROC values")
    log("="*60)
    
    all_aurocs = []
    
    def collect(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict) and "auroc" in v and v["auroc"] is not None:
                all_aurocs.append((f"{prefix}{k}", v["auroc"], v.get("n_pos", 0)))
            elif isinstance(v, dict) and "auroc" not in v:
                collect(v, f"{prefix}{k}/")
    
    collect(results)
    all_aurocs.sort(key=lambda x: x[1], reverse=True)
    
    log(f"\n{'Method':<50} {'AUROC':<8} {'N_pos'}")
    log("-"*65)
    for name, auroc, npos in all_aurocs[:30]:
        log(f"{name:<50} {auroc:<8.4f} {npos}")
    
    log(f"\nBaseline (v3 pooled, all layers): 0.543")
    log(f"Best found: {all_aurocs[0][0]} = {all_aurocs[0][1]:.4f}")
    
    # Save
    with open(os.path.join(OUT_DIR, "cssi_research_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    log(f"\nResults saved to cssi_research_results.json")
    log("Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=log_file)
        traceback.print_exc()
        sys.exit(1)
