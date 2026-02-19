#!/usr/bin/env python3
"""
CSSI Final: Head analysis, ensemble, cell-type stratified CSSI.
Memory-optimized: builds gene matrices incrementally (no storing all attentions).
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

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

log_file = open(os.path.join(OUT_DIR, "cssi_final.log"), "w", buffering=1)
log("=== CSSI Final Analysis ===")

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


def build_gene_matrix(model, tokens_list, device, g2i, n_genes, layer_idx, head_idx=None,
                      cell_types_arr=None, ct_indices=None):
    """
    Build gene-gene matrix by processing cells one at a time. Memory efficient.
    Returns: pooled_matrix, {ct: matrix} or None
    """
    model.eval()
    score_sum = np.zeros((n_genes, n_genes), dtype=np.float64)
    count_mat = np.zeros((n_genes, n_genes), dtype=np.float64)
    
    ct_scores = {}
    ct_counts = {}
    if ct_indices is not None:
        ct_of_cell = {}
        for ct, indices in ct_indices.items():
            ct_scores[ct] = np.zeros((n_genes, n_genes), dtype=np.float64)
            ct_counts[ct] = np.zeros((n_genes, n_genes), dtype=np.float64)
            for i in indices:
                ct_of_cell[i] = ct
    
    for idx, tids_full in enumerate(tokens_list):
        tids = tids_full[:MAX_SEQ]
        ml = len(tids)
        if ml < 5:
            continue
        
        input_ids = torch.tensor([tids], dtype=torch.long, device=device)
        attn_mask = torch.ones(1, ml, dtype=torch.long, device=device)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            out = model.bert(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
        
        layer_attn = out.attentions[layer_idx][0]  # (heads, seq, seq)
        if head_idx is not None:
            attn = layer_attn[head_idx].float().cpu().numpy()
        else:
            attn = layer_attn.mean(dim=0).float().cpu().numpy()
        
        del out, layer_attn, input_ids, attn_mask
        if idx % 100 == 0:
            torch.cuda.empty_cache()
        
        # Map tokens to gene indices
        mapped = [(pos, g2i[tid]) for pos, tid in enumerate(tids) if tid in g2i]
        if len(mapped) < 2:
            continue
        
        positions = np.array([m[0] for m in mapped])
        indices = np.array([m[1] for m in mapped])
        sub_attn = attn[np.ix_(positions, positions)]
        
        for a in range(len(indices)):
            i = indices[a]
            for b in range(len(indices)):
                j = indices[b]
                if i != j:
                    v = sub_attn[a, b]
                    score_sum[i, j] += v
                    count_mat[i, j] += 1
        
        if ct_indices is not None and idx in ct_of_cell:
            ct = ct_of_cell[idx]
            for a in range(len(indices)):
                i = indices[a]
                for b in range(len(indices)):
                    j = indices[b]
                    if i != j:
                        ct_scores[ct][i, j] += sub_attn[a, b]
                        ct_counts[ct][i, j] += 1
        
        if (idx + 1) % 100 == 0:
            log(f"    Cell {idx+1}/{len(tokens_list)}")
    
    mask = count_mat > 0
    pooled = np.zeros_like(score_sum)
    pooled[mask] = score_sum[mask] / count_mat[mask]
    
    ct_mats = None
    if ct_indices is not None:
        ct_mats = {}
        for ct in ct_scores:
            m = ct_counts[ct] > 0
            ct_mat = np.zeros((n_genes, n_genes), dtype=np.float64)
            ct_mat[m] = ct_scores[ct][m] / ct_counts[ct][m]
            ct_mats[ct] = ct_mat
    
    return pooled, ct_mats


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
    n_pos = int(y_true.sum())
    if n_pos < 3:
        return None, n_pos
    return float(roc_auc_score(y_true, y_score)), n_pos


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
    
    # Free adata after tokenization
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
    
    log("Tokenizing...")
    tokens = []
    for i in range(adata_s.n_obs):
        expr = adata_s.X[i].toarray().flatten() if sparse.issparse(adata_s.X) else adata_s.X[i].flatten()
        tokens.append(tokenize_cell(expr, gene_ids))
    
    valid_idx = [i for i, t in enumerate(tokens) if len(t) > 10]
    tokens = [tokens[i] for i in valid_idx]
    cell_types_arr = list(adata_s.obs['cell_type'].values[valid_idx])
    log(f"Valid: {len(tokens)} cells, CTs: {Counter(cell_types_arr)}")
    
    # Free large objects
    del adata, adata_filt, adata_s
    gc.collect()
    
    freq = Counter()
    for t in tokens:
        for tid in t:
            if tid > 3:
                freq[tid] += 1
    top_genes = [g for g, _ in freq.most_common(TOP_K)]
    g2i = {g: i for i, g in enumerate(top_genes)}
    n_genes = len(top_genes)
    gene_list = [id_to_token.get(g, f"UNK_{g}") for g in top_genes]
    
    unique_cts = sorted(set(cell_types_arr))
    ct_indices = {ct: [i for i, c in enumerate(cell_types_arr) if c == ct] for ct in unique_cts}
    log(f"Cell types: {unique_cts}")
    
    # Layer results from v2
    layer_data = {
        0: {"pooled": 0.5524, "best_cssi": "cssi_mean", "best_cssi_auroc": 0.5661},
        1: {"pooled": 0.6004, "best_cssi": "cssi_mean", "best_cssi_auroc": 0.6075},
        2: {"pooled": 0.5643, "best_cssi": "cssi_mean", "best_cssi_auroc": 0.5866},
        3: {"pooled": 0.5727, "best_cssi": "cssi_mean", "best_cssi_auroc": 0.5801},
        4: {"pooled": 0.6101, "best_cssi": "cssi_range", "best_cssi_auroc": 0.6090},
        5: {"pooled": 0.5320, "best_cssi": "cssi_mean", "best_cssi_auroc": 0.5511},
        6: {"pooled": 0.5131, "best_cssi": "cssi_mean", "best_cssi_auroc": 0.5384},
        7: {"pooled": 0.5971, "best_cssi": "cssi_deviation", "best_cssi_auroc": 0.6076},
        8: {"pooled": 0.5294, "best_cssi": "cssi_range", "best_cssi_auroc": 0.5897},
        9: {"pooled": 0.5944, "best_cssi": "cssi_range", "best_cssi_auroc": 0.6106},
        10: {"pooled": 0.6152, "best_cssi": "cssi_range", "best_cssi_auroc": 0.6477},
        11: {"pooled": 0.5679, "best_cssi": "cssi_range", "best_cssi_auroc": 0.5887},
        12: {"pooled": 0.6556, "best_cssi": "cssi_range", "best_cssi_auroc": 0.6662},
        13: {"pooled": 0.6940, "best_cssi": "cssi_deviation", "best_cssi_auroc": 0.6937},
        14: {"pooled": 0.6829, "best_cssi": "cssi_deviation", "best_cssi_auroc": 0.6824},
        15: {"pooled": 0.6314, "best_cssi": "cssi_range", "best_cssi_auroc": 0.6405},
        16: {"pooled": 0.6685, "best_cssi": "cssi_range", "best_cssi_auroc": 0.6779},
        17: {"pooled": 0.6732, "best_cssi": "cssi_deviation", "best_cssi_auroc": 0.6727},
    }
    
    results = {
        "metadata": {
            "n_cells": len(tokens), "n_genes": n_genes,
            "cell_types": {ct: len(idx) for ct, idx in ct_indices.items()},
            "trrust_edges": len(trrust),
            "baseline_all_layers_pooled": 0.543
        },
        "per_layer": {str(k): v for k, v in layer_data.items()},
        "per_head": {},
        "ensemble": {},
        "cell_type_stratified": {},
    }
    
    # Checkpoint function
    def save_checkpoint():
        with open(os.path.join(OUT_DIR, "cssi_final_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
    
    # ===== TASK 1: Head-level analysis =====
    target_layers = [13, 14, 12, 10]
    all_head_aurocs = {}
    
    for layer in target_layers:
        log(f"\n=== Head analysis: Layer {layer} ===")
        t0 = time.time()
        
        for head in range(n_heads):
            mat, _ = build_gene_matrix(model, tokens, device, g2i, n_genes,
                                       layer_idx=layer, head_idx=head)
            auroc, n_pos = evaluate_auroc(mat, gene_list, trrust)
            key = f"L{layer}_H{head}"
            all_head_aurocs[key] = auroc
            results["per_head"][key] = {"auroc": auroc, "n_pos": n_pos}
            if auroc and auroc > 0.60:
                log(f"  {key}: AUROC={auroc:.4f}")
            del mat
            gc.collect()
            torch.cuda.empty_cache()
        
        dt = time.time() - t0
        layer_heads = {k: v for k, v in all_head_aurocs.items() if k.startswith(f"L{layer}_")}
        best_h = max(layer_heads, key=lambda k: layer_heads[k] or 0)
        log(f"  Layer {layer} done ({dt:.0f}s). Best: {best_h}={layer_heads[best_h]:.4f}")
        save_checkpoint()
    
    # ===== TASK 2: Ensemble of top heads =====
    log("\n=== Ensemble of top heads ===")
    sorted_heads = sorted(all_head_aurocs.items(), key=lambda x: x[1] or 0, reverse=True)
    log(f"Top 10 heads: {[(k, f'{v:.4f}') for k, v in sorted_heads[:10]]}")
    
    for n_top in [3, 5, 10]:
        top_heads = sorted_heads[:n_top]
        log(f"\n  Ensemble top-{n_top}: {[h[0] for h in top_heads]}")
        
        mats = []
        for hkey, _ in top_heads:
            parts = hkey.split("_")
            layer = int(parts[0][1:])
            head = int(parts[1][1:])
            mat, _ = build_gene_matrix(model, tokens, device, g2i, n_genes,
                                       layer_idx=layer, head_idx=head)
            mats.append(mat)
            gc.collect()
            torch.cuda.empty_cache()
        
        avg_mat = np.mean(mats, axis=0)
        auroc_avg, n_pos = evaluate_auroc(avg_mat, gene_list, trrust)
        results["ensemble"][f"avg_top{n_top}_heads"] = {
            "auroc": auroc_avg, "n_pos": n_pos,
            "heads": [h[0] for h in top_heads]
        }
        log(f"  Avg top-{n_top}: AUROC={auroc_avg:.4f}")
        
        max_mat = np.max(mats, axis=0)
        auroc_max, _ = evaluate_auroc(max_mat, gene_list, trrust)
        results["ensemble"][f"max_top{n_top}_heads"] = {"auroc": auroc_max, "n_pos": n_pos}
        log(f"  Max top-{n_top}: AUROC={auroc_max:.4f}")
        
        del mats, avg_mat, max_mat
        gc.collect()
    
    save_checkpoint()
    
    # ===== TASK 3: Cell-type stratified CSSI on layer 13 =====
    log("\n=== Cell-type stratified CSSI: Layer 13 ===")
    
    pooled_mat, ct_mats = build_gene_matrix(model, tokens, device, g2i, n_genes,
                                             layer_idx=13, ct_indices=ct_indices)
    auroc_pooled, n_pos = evaluate_auroc(pooled_mat, gene_list, trrust)
    log(f"  Pooled: AUROC={auroc_pooled:.4f}")
    
    for ct, ct_mat in ct_mats.items():
        auroc_ct, _ = evaluate_auroc(ct_mat, gene_list, trrust)
        results["cell_type_stratified"][f"L13_{ct}"] = {"auroc": auroc_ct, "n_cells": len(ct_indices[ct])}
        log(f"  {ct} (n={len(ct_indices[ct])}): AUROC={auroc_ct:.4f}" if auroc_ct else f"  {ct}: insufficient")
    
    ct_stack = np.stack(list(ct_mats.values()))
    cssi_variants = {
        "cssi_max": np.max(ct_stack, axis=0),
        "cssi_mean": np.mean(ct_stack, axis=0),
        "cssi_range": np.max(ct_stack, axis=0) - np.min(ct_stack, axis=0),
        "cssi_std": np.std(ct_stack, axis=0),
        "cssi_deviation": np.max(np.abs(ct_stack - pooled_mat[np.newaxis]), axis=0),
    }
    
    log("\n  CSSI variants:")
    for cname, cmat in cssi_variants.items():
        auroc_c, _ = evaluate_auroc(cmat, gene_list, trrust)
        results["cell_type_stratified"][f"L13_{cname}"] = {"auroc": auroc_c}
        log(f"    {cname}: AUROC={auroc_c:.4f}" if auroc_c else f"    {cname}: failed")
    
    # Weighted deviation
    total_cells = sum(len(v) for v in ct_indices.values())
    weights = np.array([len(ct_indices[ct]) / total_cells for ct in ct_mats.keys()])
    weighted_dev = np.sum(np.abs(ct_stack - pooled_mat[np.newaxis]) * weights[:, None, None], axis=0)
    auroc_wd, _ = evaluate_auroc(weighted_dev, gene_list, trrust)
    results["cell_type_stratified"]["L13_cssi_weighted_dev"] = {"auroc": auroc_wd}
    log(f"    cssi_weighted_dev: AUROC={auroc_wd:.4f}" if auroc_wd else "    weighted_dev: failed")
    
    # Combined pooled + cssi
    for alpha in [0.3, 0.5, 0.7]:
        combined = alpha * pooled_mat + (1 - alpha) * cssi_variants["cssi_deviation"]
        auroc_comb, _ = evaluate_auroc(combined, gene_list, trrust)
        results["cell_type_stratified"][f"L13_pooled+dev_a{alpha}"] = {"auroc": auroc_comb}
        log(f"    pooled+dev (a={alpha}): AUROC={auroc_comb:.4f}")
    
    # Combined pooled + cssi_range
    for alpha in [0.3, 0.5, 0.7]:
        combined = alpha * pooled_mat + (1 - alpha) * cssi_variants["cssi_range"]
        auroc_comb, _ = evaluate_auroc(combined, gene_list, trrust)
        results["cell_type_stratified"][f"L13_pooled+range_a{alpha}"] = {"auroc": auroc_comb}
        log(f"    pooled+range (a={alpha}): AUROC={auroc_comb:.4f}")
    
    del ct_mats, ct_stack, pooled_mat
    gc.collect()
    torch.cuda.empty_cache()
    
    save_checkpoint()
    
    # ===== FINAL RANKING =====
    log("\n" + "="*60)
    log("FINAL RANKING")
    log("="*60)
    
    all_methods = []
    for l, d in layer_data.items():
        all_methods.append((f"L{l}_pooled", d["pooled"]))
        all_methods.append((f"L{l}_{d['best_cssi']}", d["best_cssi_auroc"]))
    for k, v in results["per_head"].items():
        if v.get("auroc"):
            all_methods.append((k, v["auroc"]))
    for k, v in results["ensemble"].items():
        if v.get("auroc"):
            all_methods.append((k, v["auroc"]))
    for k, v in results["cell_type_stratified"].items():
        if v.get("auroc"):
            all_methods.append((k, v["auroc"]))
    
    all_methods.sort(key=lambda x: x[1] or 0, reverse=True)
    
    log(f"\n{'Method':<50} {'AUROC':<8}")
    log("-"*58)
    for name, auroc in all_methods[:40]:
        if auroc:
            log(f"{name:<50} {auroc:.4f}")
    
    best = all_methods[0] if all_methods else ("none", 0)
    log(f"\nBaseline (all-layers pooled): 0.543")
    log(f"Best: {best[0]} = {best[1]:.4f} (+{best[1]-0.543:.4f})")
    
    results["final_ranking"] = [{"method": n, "auroc": a} for n, a in all_methods[:50] if a]
    save_checkpoint()
    log("\nDone! Results saved to cssi_final_results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        traceback.print_exc(file=log_file)
        sys.exit(1)
