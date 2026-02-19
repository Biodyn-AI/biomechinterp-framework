#!/usr/bin/env python3
"""
CSSI Research v2: Memory-efficient. Process one layer at a time.
"""
import os, sys, pickle, json, time, traceback, gc
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
N_CELLS = 500

log_file = open(os.path.join(OUT_DIR, "cssi_research_v2.log"), "w", buffering=1)
def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

log("=== CSSI Research v2 Starting ===")

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


def build_gene_matrix_from_cells(model, token_ids_list, device, top_genes, 
                                  layer_idx=None, head_idx=None, cell_types=None):
    """
    Extract attention and build gene matrix in one pass.
    layer_idx: which layer (0-17) or None for all
    head_idx: which head (0-17) or None for all heads in layer
    cell_types: if provided, also build per-cell-type matrices
    
    Returns: pooled_matrix, {ct: ct_matrix} or None, gene_list
    """
    model.eval()
    g2i = {g: i for i, g in enumerate(top_genes)}
    n = len(top_genes)
    
    score_sum = np.zeros((n, n), dtype=np.float64)
    count_mat = np.zeros((n, n), dtype=np.float64)
    
    # Per cell-type accumulators
    ct_scores = {}
    ct_counts_mat = {}
    if cell_types is not None:
        unique_cts = list(set(cell_types))
        for ct in unique_cts:
            ct_scores[ct] = np.zeros((n, n), dtype=np.float64)
            ct_counts_mat[ct] = np.zeros((n, n), dtype=np.float64)
    
    for idx in range(0, len(token_ids_list)):
        tids = token_ids_list[idx][:MAX_SEQ]
        ml = len(tids)
        
        input_ids = torch.tensor([tids], dtype=torch.long, device=device)
        attn_mask = torch.ones(1, ml, dtype=torch.long, device=device)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            out = model.bert(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
        
        # Extract attention for specified layer/head
        if layer_idx is not None:
            layer_attn = out.attentions[layer_idx][0]  # (heads, seq, seq)
            if head_idx is not None:
                attn = layer_attn[head_idx].float().cpu().numpy()  # (seq, seq)
            else:
                attn = layer_attn.mean(dim=0).float().cpu().numpy()
        else:
            # All layers, all heads
            all_attn = torch.stack([a[0] for a in out.attentions])  # (layers, heads, seq, seq)
            attn = all_attn.mean(dim=(0, 1)).float().cpu().numpy()
        
        del out
        if idx % 200 == 0:
            torch.cuda.empty_cache()
        
        # Map to gene indices
        mapped = [(pos, g2i[tid]) for pos, tid in enumerate(tids) if tid in g2i]
        if len(mapped) < 2:
            continue
        
        positions = np.array([m[0] for m in mapped])
        indices = np.array([m[1] for m in mapped])
        sub_attn = attn[np.ix_(positions, positions)]
        
        # Accumulate pooled
        for a in range(len(indices)):
            i = indices[a]
            for b in range(len(indices)):
                j = indices[b]
                if i != j:
                    score_sum[i, j] += sub_attn[a, b]
                    count_mat[i, j] += 1
        
        # Accumulate per cell type
        if cell_types is not None:
            ct = cell_types[idx]
            for a in range(len(indices)):
                i = indices[a]
                for b in range(len(indices)):
                    j = indices[b]
                    if i != j:
                        ct_scores[ct][i, j] += sub_attn[a, b]
                        ct_counts_mat[ct][i, j] += 1
        
        if (idx + 1) % 100 == 0:
            log(f"    Cell {idx+1}/{len(token_ids_list)}")
    
    # Average
    mask = count_mat > 0
    pooled = np.zeros_like(score_sum)
    pooled[mask] = score_sum[mask] / count_mat[mask]
    
    ct_matrices = None
    if cell_types is not None:
        ct_matrices = {}
        for ct in ct_scores:
            m = ct_counts_mat[ct] > 0
            ct_mat = np.zeros((n, n), dtype=np.float64)
            ct_mat[m] = ct_scores[ct][m] / ct_counts_mat[ct][m]
            ct_matrices[ct] = ct_mat
    
    gene_list = [id_to_token.get(g, f"UNK_{g}") for g in top_genes]
    return pooled, ct_matrices, gene_list


def evaluate_auroc(score_matrix, gene_list, gt_edges):
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
        return None, int(n_pos)
    return float(roc_auc_score(y_true, y_score)), int(n_pos)


def cssi_aggregations(ct_matrices, pooled):
    """Compute various CSSI aggregations from per-cell-type matrices."""
    if not ct_matrices or len(ct_matrices) < 2:
        return {}
    
    ct_stack = np.stack(list(ct_matrices.values()))
    results = {
        "cssi_max": np.max(ct_stack, axis=0),
        "cssi_mean": np.mean(ct_stack, axis=0),
        "cssi_median": np.median(ct_stack, axis=0),
        "cssi_std": np.std(ct_stack, axis=0),
        "cssi_range": np.max(ct_stack, axis=0) - np.min(ct_stack, axis=0),
        "cssi_deviation": np.max(np.abs(ct_stack - pooled[np.newaxis]), axis=0),
    }
    return results


def apply_transforms(mat):
    results = {"raw": mat}
    results["sym"] = (mat + mat.T) / 2
    row_mean = mat.mean(axis=1, keepdims=True)
    row_std = mat.std(axis=1, keepdims=True)
    row_std[row_std == 0] = 1
    results["zscore"] = (mat - row_mean) / row_std
    results["product"] = mat * mat.T
    # Sparse top-10 per row
    s = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        top = np.argsort(mat[i])[-10:]
        s[i, top] = mat[i, top]
    results["sparse10"] = s
    return results


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
    
    log("Loading TRRUST...")
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
    
    # Tokenize
    log("Tokenizing...")
    tokens = []
    for i in range(adata_s.n_obs):
        expr = adata_s.X[i].toarray().flatten() if sparse.issparse(adata_s.X) else adata_s.X[i].flatten()
        tokens.append(tokenize_cell(expr, gene_ids))
    
    valid_idx = [i for i, t in enumerate(tokens) if len(t) > 10]
    tokens = [tokens[i] for i in valid_idx]
    cell_types_arr = list(adata_s.obs['cell_type'].values[valid_idx])
    log(f"Valid: {len(tokens)} cells, CTs: {Counter(cell_types_arr)}")
    
    # Top genes
    freq = Counter()
    for t in tokens:
        for tid in t:
            if tid > 3:
                freq[tid] += 1
    top_genes = [g for g, _ in freq.most_common(TOP_K)]
    
    results = {"metadata": {"n_cells": len(tokens), "n_layers": n_layers, "n_heads": n_heads}}
    all_aurocs = []  # (name, auroc, n_pos)
    
    # ===== EXPERIMENT 1: Per-layer =====
    log("\n=== EXP 1: Per-layer (all heads averaged) ===")
    layer_results = {}
    for layer in range(n_layers):
        t0 = time.time()
        pooled, ct_mats, genes = build_gene_matrix_from_cells(
            model, tokens, device, top_genes, layer_idx=layer, cell_types=cell_types_arr)
        auroc, n_pos = evaluate_auroc(pooled, genes, trrust)
        
        # Also test CSSI variants
        cssi = cssi_aggregations(ct_mats, pooled)
        best_cssi_name, best_cssi_auroc = "", 0
        for cname, cmat in cssi.items():
            ca, _ = evaluate_auroc(cmat, genes, trrust)
            if ca and ca > best_cssi_auroc:
                best_cssi_auroc = ca
                best_cssi_name = cname
        
        dt = time.time() - t0
        layer_results[layer] = {
            "pooled_auroc": auroc, "n_pos": n_pos,
            "best_cssi": best_cssi_name, "best_cssi_auroc": best_cssi_auroc
        }
        all_aurocs.append((f"L{layer}_pooled", auroc, n_pos))
        if best_cssi_auroc > 0:
            all_aurocs.append((f"L{layer}_{best_cssi_name}", best_cssi_auroc, n_pos))
        
        log(f"  Layer {layer:2d}: pooled={auroc:.4f}, best_cssi={best_cssi_name}={best_cssi_auroc:.4f} ({dt:.0f}s)")
        gc.collect()
    
    results["per_layer"] = {str(k): v for k, v in layer_results.items()}
    
    # Find best layers
    best_layers = sorted(layer_results.keys(), key=lambda l: layer_results[l]["pooled_auroc"] or 0, reverse=True)
    log(f"\n  Top-3 layers: {best_layers[:3]}")
    
    # ===== EXPERIMENT 2: Per-head for best 2 layers =====
    log("\n=== EXP 2: Per-head (top 2 layers) ===")
    head_results = {}
    for layer in best_layers[:2]:
        for head in range(n_heads):
            pooled, _, genes = build_gene_matrix_from_cells(
                model, tokens, device, top_genes, layer_idx=layer, head_idx=head)
            auroc, n_pos = evaluate_auroc(pooled, genes, trrust)
            key = f"L{layer}_H{head}"
            head_results[key] = {"auroc": auroc, "n_pos": n_pos}
            all_aurocs.append((key, auroc, n_pos))
            if auroc and auroc > 0.56:
                log(f"  {key}: AUROC={auroc:.4f}")
        log(f"  Layer {layer} heads done")
        gc.collect()
    
    results["per_head"] = head_results
    best_head = max(head_results, key=lambda k: head_results[k].get("auroc", 0) or 0)
    log(f"  Best head: {best_head} = {head_results[best_head]['auroc']:.4f}")
    
    # ===== EXPERIMENT 3: Transforms on best layer =====
    log("\n=== EXP 3: Transforms on best layer ===")
    best_layer = best_layers[0]
    pooled_best, ct_mats_best, genes_best = build_gene_matrix_from_cells(
        model, tokens, device, top_genes, layer_idx=best_layer, cell_types=cell_types_arr)
    
    transform_results = {}
    for tname, tmat in apply_transforms(pooled_best).items():
        auroc, n_pos = evaluate_auroc(tmat, genes_best, trrust)
        transform_results[tname] = {"auroc": auroc, "n_pos": n_pos}
        all_aurocs.append((f"L{best_layer}_{tname}", auroc, n_pos))
        log(f"  L{best_layer}+{tname}: AUROC={auroc:.4f}")
    
    results["transforms"] = transform_results
    
    # ===== EXPERIMENT 4: CSSI + transforms on best layer =====
    log("\n=== EXP 4: CSSI variants + transforms ===")
    cssi_transform_results = {}
    cssi_mats = cssi_aggregations(ct_mats_best, pooled_best)
    
    for cssi_name, cssi_mat in cssi_mats.items():
        for tname, tmat in apply_transforms(cssi_mat).items():
            auroc, n_pos = evaluate_auroc(tmat, genes_best, trrust)
            key = f"{cssi_name}+{tname}"
            cssi_transform_results[key] = {"auroc": auroc, "n_pos": n_pos}
            all_aurocs.append((f"L{best_layer}_{key}", auroc, n_pos))
            if auroc and auroc > 0.56:
                log(f"  {key}: AUROC={auroc:.4f}")
    
    results["cssi_transforms"] = cssi_transform_results
    
    # ===== EXPERIMENT 5: Multi-layer combinations =====
    log("\n=== EXP 5: Multi-layer combinations ===")
    # Average top-2 and top-3 layers
    multi_results = {}
    for n_top in [2, 3, 5]:
        layers_to_use = best_layers[:n_top]
        # Build matrices for each and average
        mats = []
        for layer in layers_to_use:
            p, _, g = build_gene_matrix_from_cells(model, tokens, device, top_genes, layer_idx=layer)
            mats.append(p)
        
        avg_mat = np.mean(mats, axis=0)
        auroc, n_pos = evaluate_auroc(avg_mat, genes_best, trrust)
        multi_results[f"avg_top{n_top}"] = {"auroc": auroc, "n_pos": n_pos, 
                                             "layers": [int(l) for l in layers_to_use]}
        all_aurocs.append((f"avg_top{n_top}_layers", auroc, n_pos))
        log(f"  Avg top-{n_top} layers {layers_to_use}: AUROC={auroc:.4f}")
        
        # Max across layers
        max_mat = np.max(mats, axis=0)
        auroc2, _ = evaluate_auroc(max_mat, genes_best, trrust)
        multi_results[f"max_top{n_top}"] = {"auroc": auroc2, "n_pos": n_pos}
        all_aurocs.append((f"max_top{n_top}_layers", auroc2, n_pos))
        log(f"  Max top-{n_top} layers: AUROC={auroc2:.4f}")
    
    results["multi_layer"] = multi_results
    
    # ===== EXPERIMENT 6: Last layer (often most task-specific) =====
    log("\n=== EXP 6: Last layer analysis ===")
    last_layer = n_layers - 1
    pooled_last, ct_mats_last, genes_last = build_gene_matrix_from_cells(
        model, tokens, device, top_genes, layer_idx=last_layer, cell_types=cell_types_arr)
    
    auroc_last, n_pos = evaluate_auroc(pooled_last, genes_last, trrust)
    log(f"  Last layer ({last_layer}) pooled: AUROC={auroc_last:.4f}")
    
    cssi_last = cssi_aggregations(ct_mats_last, pooled_last)
    for cname, cmat in cssi_last.items():
        auroc, _ = evaluate_auroc(cmat, genes_last, trrust)
        all_aurocs.append((f"L{last_layer}_{cname}", auroc, n_pos))
        if auroc and auroc > 0.55:
            log(f"  L{last_layer}+{cname}: AUROC={auroc:.4f}")
    
    # ===== FINAL SUMMARY =====
    log("\n" + "="*60)
    log("FINAL RANKING")
    log("="*60)
    
    all_aurocs.sort(key=lambda x: (x[1] or 0), reverse=True)
    log(f"\n{'Method':<50} {'AUROC':<8} {'N_pos'}")
    log("-"*65)
    for name, auroc, npos in all_aurocs[:30]:
        if auroc:
            log(f"{name:<50} {auroc:<8.4f} {npos}")
    
    log(f"\nBaseline (v3 all-layers pooled): 0.543")
    if all_aurocs and all_aurocs[0][1]:
        log(f"Best found: {all_aurocs[0][0]} = {all_aurocs[0][1]:.4f}")
        log(f"Improvement: {all_aurocs[0][1] - 0.543:+.4f}")
    
    results["ranking"] = [{"method": n, "auroc": a, "n_pos": p} for n, a, p in all_aurocs[:50] if a]
    
    # Save
    with open(os.path.join(OUT_DIR, "cssi_research_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log("\nDone! Results saved.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        traceback.print_exc(file=log_file)
        sys.exit(1)
