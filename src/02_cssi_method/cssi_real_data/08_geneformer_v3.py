#!/usr/bin/env python3
"""
CSSI validation on Geneformer attention - v3.
Optimized: use matrix approach with top-K genes, but properly evaluate.
"""

import os, sys, pickle, json, time, traceback
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from collections import Counter

OUT_DIR = "/mnt/d/openclaw/biodyn-nmi-paper/cssi_real"
DATA_PATH = "/mnt/d/openclaw/intelligence-augmentation/data/brain_scrna/DLPFC_11k.h5ad"

log_file = open(os.path.join(OUT_DIR, "cssi_run_v3.log"), "w", buffering=1)
sys.stdout = log_file
sys.stderr = log_file

print("Starting CSSI v3...", flush=True)

import geneformer
GF_DIR = os.path.dirname(geneformer.__file__)

with open(os.path.join(GF_DIR, "token_dictionary_gc104M.pkl"), "rb") as f:
    token_dict = pickle.load(f)
with open(os.path.join(GF_DIR, "gene_median_dictionary_gc104M.pkl"), "rb") as f:
    gene_median_dict = pickle.load(f)
with open(os.path.join(GF_DIR, "gene_name_id_dict_gc104M.pkl"), "rb") as f:
    gene_name_id_dict = pickle.load(f)

id_to_token = {int(v): k for k, v in token_dict.items()}
print(f"Vocab: {len(token_dict)}", flush=True)

TOP_K = 1000  # Use top 1000 genes for matrix (covers more TRRUST genes)
MAX_SEQ = 256


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


def extract_attention(model, token_ids_list, device, batch_size=1):
    """Returns list of (token_ids_truncated, attention_matrix)."""
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
        
        all_attn = torch.stack(out.attentions)  # (18, batch, 18, seq, seq)
        avg_attn = all_attn.mean(dim=(0, 2))  # (batch, seq, seq)
        
        for i, t in enumerate(batch):
            n = len(t)
            results.append((t, avg_attn[i, :n, :n].float().cpu().numpy()))
        
        del all_attn, avg_attn, out
        torch.cuda.empty_cache()
        
        done = min(bs+batch_size, len(token_ids_list))
        if done % 100 < batch_size:
            print(f"  Attention: {done}/{len(token_ids_list)}", flush=True)
    
    return results


def build_gene_matrix_fast(attention_results, top_k=TOP_K):
    """
    Build gene-gene attention matrix using numpy for speed.
    Returns (score_matrix, gene_ensembl_list).
    """
    # Count gene frequencies across all cells
    freq = Counter()
    for tids, _ in attention_results:
        for t in tids:
            if t > 3:  # skip special tokens
                freq[t] += 1
    
    # Take top-k most frequent
    top_genes = [g for g, _ in freq.most_common(top_k)]
    g2i = {g: i for i, g in enumerate(top_genes)}
    n = len(top_genes)
    
    score_sum = np.zeros((n, n), dtype=np.float64)
    count_mat = np.zeros((n, n), dtype=np.float64)
    
    for tids, attn in attention_results:
        # Find which positions map to our top genes
        mapped = []  # (position, gene_index)
        for pos, tid in enumerate(tids):
            if tid in g2i:
                mapped.append((pos, g2i[tid]))
        
        if len(mapped) < 2:
            continue
        
        # Vectorized accumulation
        positions = np.array([m[0] for m in mapped])
        indices = np.array([m[1] for m in mapped])
        
        # Extract submatrix
        sub_attn = attn[np.ix_(positions, positions)]  # (k, k)
        
        # Add to accumulators
        for a, i in enumerate(indices):
            for b, j in enumerate(indices):
                if i != j:
                    score_sum[i, j] += sub_attn[a, b]
                    count_mat[i, j] += 1
    
    # Average
    mask = count_mat > 0
    score_sum[mask] /= count_mat[mask]
    
    gene_list = [id_to_token.get(g, f"UNK_{g}") for g in top_genes]
    return score_sum, gene_list


def evaluate_grn(score_matrix, gene_list, gt_edges, name=""):
    """Evaluate predicted scores against ground truth edges."""
    gene_set = set(gene_list)
    # GT genes that are in our gene list
    gt_genes_in_list = set()
    for g1, g2 in gt_edges:
        if g1 in gene_set:
            gt_genes_in_list.add(g1)
        if g2 in gene_set:
            gt_genes_in_list.add(g2)
    
    g2i = {g: i for i, g in enumerate(gene_list)}
    
    y_true = []
    y_score = []
    
    # Only evaluate pairs where both genes are in gene_list
    n = len(gene_list)
    for i in range(n):
        gi = gene_list[i]
        for j in range(n):
            if i == j:
                continue
            gj = gene_list[j]
            label = 1 if (gi, gj) in gt_edges else 0
            y_true.append(label)
            y_score.append(score_matrix[i, j])
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    n_pos = y_true.sum()
    
    if n_pos < 5:
        print(f"  {name}: only {n_pos} positives, skipping", flush=True)
        return None
    
    auroc = roc_auc_score(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(rec, prec)
    random_auprc = n_pos / len(y_true)
    f1s = 2*prec*rec/(prec+rec+1e-10)
    
    result = {
        "auroc": float(auroc), "auprc": float(auprc),
        "random_auprc": float(random_auprc), "best_f1": float(f1s.max()),
        "n_pos": int(n_pos), "n_total": int(len(y_true))
    }
    print(f"  {name}: AUROC={auroc:.4f} AUPRC={auprc:.4f} (rand={random_auprc:.4f}) F1={f1s.max():.4f} [{n_pos} pos / {len(y_true)} pairs]", flush=True)
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    from transformers import BertForMaskedLM
    print("Loading model...", flush=True)
    model = BertForMaskedLM.from_pretrained("ctheodoris/Geneformer", output_attentions=True).to(device).half()
    print(f"Model: {model.config.num_hidden_layers}L x {model.config.num_attention_heads}H", flush=True)
    
    print("Loading data...", flush=True)
    adata = sc.read_h5ad(DATA_PATH)
    
    # Gene name mapping
    gene_name_to_ensembl = {}
    if 'feature_name' in adata.var.columns:
        for ens_id, row in adata.var.iterrows():
            name = row['feature_name']
            if pd.notna(name) and name != '':
                gene_name_to_ensembl[name] = ens_id
    for gname, ens_id in gene_name_id_dict.items():
        gene_name_to_ensembl.setdefault(gname, ens_id)
    
    # Load TRRUST
    print("Loading TRRUST...", flush=True)
    try:
        import urllib.request
        urllib.request.urlretrieve(
            "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv",
            "/tmp/trrust.tsv"
        )
        df = pd.read_csv("/tmp/trrust.tsv", sep='\t', header=None, names=['TF', 'target', 'mode', 'pmid'])
        trrust = set()
        for _, row in df.iterrows():
            src = gene_name_to_ensembl.get(row['TF'])
            tgt = gene_name_to_ensembl.get(row['target'])
            if src and tgt:
                trrust.add((src, tgt))
        print(f"TRRUST: {len(trrust)} edges from {len(set(g for g,_ in trrust))} TFs", flush=True)
    except Exception as e:
        print(f"TRRUST failed: {e}", flush=True)
        trrust = set()
    
    # Filter cell types
    ct_counts = adata.obs['cell_type'].value_counts()
    valid_cts = ct_counts[ct_counts >= 50].index.tolist()
    adata_filt = adata[adata.obs['cell_type'].isin(valid_cts)].copy()
    gene_ids = np.array(adata_filt.var_names)
    
    results = {}
    
    for n_cells in [200, 500, 1000]:
        print(f"\n{'='*60}", flush=True)
        print(f"n_cells = {n_cells}", flush=True)
        
        np.random.seed(42)
        sampled = []
        for ct in valid_cts:
            idx = np.where(adata_filt.obs['cell_type'] == ct)[0]
            n_ct = max(1, int(n_cells * len(idx) / adata_filt.n_obs))
            sampled.extend(np.random.choice(idx, min(n_ct, len(idx)), replace=False))
        if len(sampled) > n_cells:
            sampled = list(np.random.choice(sampled, n_cells, replace=False))
        
        adata_s = adata_filt[sampled].copy()
        print(f"Sampled: {adata_s.n_obs}", flush=True)
        
        # Tokenize
        print("Tokenizing...", flush=True)
        tokens = []
        for i in range(adata_s.n_obs):
            expr = adata_s.X[i].toarray().flatten() if sparse.issparse(adata_s.X) else adata_s.X[i].flatten()
            tokens.append(tokenize_cell(expr, gene_ids))
        
        valid_idx = [i for i, t in enumerate(tokens) if len(t) > 10]
        tokens = [tokens[i] for i in valid_idx]
        cell_types = adata_s.obs['cell_type'].values[valid_idx]
        print(f"Valid: {len(tokens)}", flush=True)
        
        # Extract attention
        print("Extracting attention...", flush=True)
        t0 = time.time()
        attn_results = extract_attention(model, tokens, device)
        print(f"Attention done in {time.time()-t0:.1f}s", flush=True)
        
        # POOLED matrix
        print("Building POOLED matrix (top-1000 genes)...", flush=True)
        t0 = time.time()
        pooled_mat, pooled_genes = build_gene_matrix_fast(attn_results, top_k=TOP_K)
        print(f"POOLED matrix: {pooled_mat.shape}, {time.time()-t0:.1f}s", flush=True)
        
        # Check TRRUST overlap
        trrust_in_pooled = sum(1 for g1, g2 in trrust if g1 in set(pooled_genes) and g2 in set(pooled_genes))
        print(f"TRRUST edges in pooled genes: {trrust_in_pooled}", flush=True)
        
        # CSSI per cell type - use same gene set as pooled
        print("Building CSSI matrices...", flush=True)
        unique_cts = [ct for ct in set(cell_types) if (cell_types == ct).sum() >= 5]
        
        cssi_max_mat = np.zeros_like(pooled_mat)
        cssi_mean_mat = np.zeros_like(pooled_mat)
        total_weight = 0
        
        # Build per-stratum matrices aligned to pooled gene list
        pooled_gene_set = {g: i for i, g in enumerate(pooled_genes)}
        
        for ct in unique_cts:
            mask = cell_types == ct
            n_ct = mask.sum()
            ct_results = [attn_results[i] for i in range(len(attn_results)) if mask[i]]
            
            ct_mat, ct_genes = build_gene_matrix_fast(ct_results, top_k=TOP_K)
            
            # Align to pooled gene list
            ct_gene_map = {g: i for i, g in enumerate(ct_genes)}
            aligned = np.zeros_like(pooled_mat)
            
            for pi, pg in enumerate(pooled_genes):
                if pg in ct_gene_map:
                    ci = ct_gene_map[pg]
                    for pj, pg2 in enumerate(pooled_genes):
                        if pg2 in ct_gene_map:
                            cj = ct_gene_map[pg2]
                            aligned[pi, pj] = ct_mat[ci, cj]
            
            cssi_max_mat = np.maximum(cssi_max_mat, aligned)
            cssi_mean_mat += n_ct * aligned
            total_weight += n_ct
            print(f"  {ct}: {n_ct} cells", flush=True)
        
        if total_weight > 0:
            cssi_mean_mat /= total_weight
        
        # Evaluate
        print(f"\n--- Results (n_cells={n_cells}) ---", flush=True)
        exp = {"n_cells": n_cells, "n_genes": len(pooled_genes), "trrust_edges_in_genes": trrust_in_pooled}
        
        if trrust:
            exp["TRRUST_pooled"] = evaluate_grn(pooled_mat, pooled_genes, trrust, "POOLED")
            exp["TRRUST_cssi_max"] = evaluate_grn(cssi_max_mat, pooled_genes, trrust, "CSSI-max")
            exp["TRRUST_cssi_mean"] = evaluate_grn(cssi_mean_mat, pooled_genes, trrust, "CSSI-mean")
        
        results[str(n_cells)] = exp
    
    # Save
    with open(os.path.join(OUT_DIR, "cssi_results_v3.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'Method':<12} {'N':<6} {'AUROC':<8} {'AUPRC':<8} {'F1':<8} {'pos':<8}", flush=True)
    print("-"*50, flush=True)
    for n in ["200", "500", "1000"]:
        for method, label in [("TRRUST_pooled","pooled"), ("TRRUST_cssi_max","cssi_max"), ("TRRUST_cssi_mean","cssi_mean")]:
            r = results.get(n, {}).get(method)
            if r:
                print(f"{label:<12} {n:<6} {r['auroc']:<8.4f} {r['auprc']:<8.4f} {r['best_f1']:<8.4f} {r['n_pos']:<8}", flush=True)
    
    print("\nDone!", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
