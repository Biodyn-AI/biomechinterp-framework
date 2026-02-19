#!/usr/bin/env python3
"""
CSSI validation on real Geneformer attention weights.
Brain DLPFC data → Geneformer attention → GRN inference → evaluate vs ground truth.
"""

import os, sys, pickle, json, time, traceback
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

OUT_DIR = "/mnt/d/openclaw/biodyn-nmi-paper/cssi_real"

# Redirect all output to log file
log_file = open(os.path.join(OUT_DIR, "cssi_run.log"), "w", buffering=1)
sys.stdout = log_file
sys.stderr = log_file
DATA_PATH = "/mnt/d/openclaw/intelligence-augmentation/data/brain_scrna/DLPFC_11k.h5ad"

print("Starting CSSI Geneformer validation...", flush=True)

# ─── Load dictionaries ───
import geneformer
GF_DIR = os.path.dirname(geneformer.__file__)

# Use gc104M dicts to match model vocab_size=20275
with open(os.path.join(GF_DIR, "token_dictionary_gc104M.pkl"), "rb") as f:
    token_dict = pickle.load(f)
with open(os.path.join(GF_DIR, "gene_median_dictionary_gc104M.pkl"), "rb") as f:
    gene_median_dict = pickle.load(f)
with open(os.path.join(GF_DIR, "gene_name_id_dict_gc104M.pkl"), "rb") as f:
    gene_name_id_dict = pickle.load(f)

id_to_token = {int(v): k for k, v in token_dict.items()}
print(f"Token dict: {len(token_dict)} genes", flush=True)


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
    rank_vals = vals / medians
    idx = np.argsort(-rank_vals)[:max_len]
    return [int(token_dict[genes[i]]) for i in idx]


def extract_attention(model, token_ids_list, device, batch_size=4, max_seq=512):
    model.eval()
    results = []
    for bs in range(0, len(token_ids_list), batch_size):
        batch = [t[:max_seq] for t in token_ids_list[bs:bs+batch_size]]
        ml = max(len(t) for t in batch)
        padded = [t + [0]*(ml-len(t)) for t in batch]
        masks = [[1]*len(t) + [0]*(ml-len(t)) for t in batch]
        
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attn_mask = torch.tensor(masks, dtype=torch.long, device=device)
        
        with torch.no_grad():
            out = model.bert(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
        
        # Average over layers and heads
        all_attn = torch.stack(out.attentions)  # (layers, batch, heads, seq, seq)
        avg_attn = all_attn.mean(dim=(0, 2))  # (batch, seq, seq)
        
        for i, t in enumerate(batch):
            n = len(t)
            results.append((t, avg_attn[i, :n, :n].cpu().numpy()))
        
        del all_attn, avg_attn, out
        torch.cuda.empty_cache()
        
        done = min(bs+batch_size, len(token_ids_list))
        if done % 50 < batch_size:
            print(f"  Attention: {done}/{len(token_ids_list)} cells", flush=True)
    
    return results


def build_gene_matrix(attention_results, top_k=300):
    from collections import Counter
    freq = Counter()
    for tids, _ in attention_results:
        for t in tids:
            freq[t] += 1
    
    top_genes = [g for g, _ in freq.most_common(top_k)]
    g2i = {g: i for i, g in enumerate(top_genes)}
    n = len(top_genes)
    
    scores = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.float64)
    
    for tids, attn in attention_results:
        pos_map = {}
        for pos, tid in enumerate(tids):
            if tid in g2i:
                pos_map[pos] = g2i[tid]
        
        positions = list(pos_map.keys())
        for p1 in positions:
            i = pos_map[p1]
            for p2 in positions:
                j = pos_map[p2]
                if i != j:
                    scores[i, j] += attn[p1, p2]
                    counts[i, j] += 1
    
    mask = counts > 0
    scores[mask] /= counts[mask]
    
    gene_list = [id_to_token.get(g, f"UNK_{g}") for g in top_genes]
    return scores, gene_list


def load_trrust(gene_name_to_ensembl):
    """Load TRRUST v2 human from web or generate synthetic ground truth."""
    try:
        import urllib.request
        urllib.request.urlretrieve(
            "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv",
            "/tmp/trrust.tsv"
        )
        df = pd.read_csv("/tmp/trrust.tsv", sep='\t', header=None, names=['TF', 'target', 'mode', 'pmid'])
        print(f"TRRUST loaded: {len(df)} edges", flush=True)
        edges = set()
        for _, row in df.iterrows():
            src = gene_name_to_ensembl.get(row['TF'])
            tgt = gene_name_to_ensembl.get(row['target'])
            if src and tgt:
                edges.add((src, tgt))
        print(f"TRRUST mapped: {len(edges)} edges", flush=True)
        return edges
    except Exception as e:
        print(f"TRRUST download failed: {e}", flush=True)
        return set()


def load_dorothea_manual(gene_name_to_ensembl):
    """Load DoRothEA from OmniPath API directly."""
    try:
        url = "https://omnipathdb.org/interactions?datasets=dorothea&dorothea_level=A,B&fields=dorothea_level&genesymbols=1&organisms=9606&license=academic"
        import urllib.request
        urllib.request.urlretrieve(url, "/tmp/dorothea.tsv")
        df = pd.read_csv("/tmp/dorothea.tsv", sep='\t')
        print(f"DoRothEA loaded: {len(df)} edges", flush=True)
        edges = set()
        for _, row in df.iterrows():
            src = gene_name_to_ensembl.get(str(row.get('source_genesymbol', '')))
            tgt = gene_name_to_ensembl.get(str(row.get('target_genesymbol', '')))
            if src and tgt:
                edges.add((src, tgt))
        print(f"DoRothEA mapped: {len(edges)} edges", flush=True)
        return edges
    except Exception as e:
        print(f"DoRothEA download failed: {e}", flush=True)
        return set()


def evaluate_grn(score_matrix, gene_list, gt_edges, name=""):
    n = len(gene_list)
    y_true, y_score = [], []
    for i in range(n):
        for j in range(n):
            if i == j: continue
            y_true.append(1 if (gene_list[i], gene_list[j]) in gt_edges else 0)
            y_score.append(score_matrix[i, j])
    
    y_true, y_score = np.array(y_true), np.array(y_score)
    n_pos = y_true.sum()
    
    if n_pos == 0:
        print(f"  {name}: 0 positive edges in ground truth", flush=True)
        return None
    
    auroc = roc_auc_score(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(rec, prec)
    random_auprc = n_pos / len(y_true)
    f1s = 2*prec*rec/(prec+rec+1e-10)
    
    print(f"  {name}: AUROC={auroc:.4f} AUPRC={auprc:.4f} (rand={random_auprc:.4f}) F1={f1s.max():.4f} pos={n_pos}", flush=True)
    return {"auroc": float(auroc), "auprc": float(auprc), "random_auprc": float(random_auprc), "best_f1": float(f1s.max()), "n_pos": int(n_pos)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    # Load model
    print("Loading Geneformer...", flush=True)
    from transformers import BertForMaskedLM
    model = BertForMaskedLM.from_pretrained("ctheodoris/Geneformer", output_attentions=True, torch_dtype=torch.float16).to(device)
    print(f"Model: {model.config.num_hidden_layers}L x {model.config.num_attention_heads}H", flush=True)
    
    # Load data
    print("Loading brain data...", flush=True)
    adata = sc.read_h5ad(DATA_PATH)
    print(f"Data: {adata.shape}", flush=True)
    
    # Gene name mapping
    gene_name_to_ensembl = {}
    if 'feature_name' in adata.var.columns:
        for ens_id, row in adata.var.iterrows():
            name = row['feature_name']
            if pd.notna(name) and name != '':
                gene_name_to_ensembl[name] = ens_id
    for gname, ens_id in gene_name_id_dict.items():
        gene_name_to_ensembl.setdefault(gname, ens_id)
    print(f"Gene name mapping: {len(gene_name_to_ensembl)}", flush=True)
    
    # Load ground truth
    print("Loading ground truth...", flush=True)
    trrust = load_trrust(gene_name_to_ensembl)
    dorothea = load_dorothea_manual(gene_name_to_ensembl)
    
    # Filter cell types
    ct_counts = adata.obs['cell_type'].value_counts()
    valid_cts = ct_counts[ct_counts >= 50].index.tolist()
    adata_filt = adata[adata.obs['cell_type'].isin(valid_cts)].copy()
    gene_ids = np.array(adata_filt.var_names)
    
    results = {}
    
    for n_cells in [200, 500, 1000]:
        print(f"\n{'='*60}", flush=True)
        print(f"n_cells = {n_cells}", flush=True)
        
        # Stratified sample
        np.random.seed(42)
        sampled = []
        for ct in valid_cts:
            idx = np.where(adata_filt.obs['cell_type'] == ct)[0]
            n_ct = max(1, int(n_cells * len(idx) / adata_filt.n_obs))
            sampled.extend(np.random.choice(idx, min(n_ct, len(idx)), replace=False))
        if len(sampled) > n_cells:
            sampled = list(np.random.choice(sampled, n_cells, replace=False))
        
        adata_s = adata_filt[sampled].copy()
        print(f"Sampled: {adata_s.n_obs} cells", flush=True)
        ct_dist = dict(adata_s.obs['cell_type'].value_counts())
        print(f"Distribution: {ct_dist}", flush=True)
        
        # Tokenize
        print("Tokenizing...", flush=True)
        tokens = []
        for i in range(adata_s.n_obs):
            expr = adata_s.X[i].toarray().flatten() if sparse.issparse(adata_s.X) else adata_s.X[i].flatten()
            tokens.append(tokenize_cell(expr, gene_ids))
        
        valid_idx = [i for i, t in enumerate(tokens) if len(t) > 10]
        tokens = [tokens[i] for i in valid_idx]
        cell_types = adata_s.obs['cell_type'].values[valid_idx]
        print(f"Valid: {len(tokens)} cells, median tokens: {np.median([len(t) for t in tokens]):.0f}", flush=True)
        
        # Extract attention
        print("Extracting attention...", flush=True)
        t0 = time.time()
        attn_results = extract_attention(model, tokens, device, batch_size=1, max_seq=256)
        print(f"Done in {time.time()-t0:.1f}s", flush=True)
        
        # POOLED
        print("Building POOLED matrix...", flush=True)
        pooled_mat, gene_list = build_gene_matrix(attn_results, top_k=300)
        
        # CSSI per cell type
        print("Building CSSI matrices...", flush=True)
        unique_cts = [ct for ct in set(cell_types) if (cell_types == ct).sum() >= 5]
        stratum_data = {}
        for ct in unique_cts:
            mask = cell_types == ct
            ct_results = [attn_results[i] for i in range(len(attn_results)) if mask[i]]
            mat, genes = build_gene_matrix(ct_results, top_k=300)
            stratum_data[ct] = (mat, genes, mask.sum())
            print(f"  {ct}: {mask.sum()} cells", flush=True)
        
        # Align to common genes
        all_gene_sets = [set(genes) for mat, genes, _ in stratum_data.values()]
        all_gene_sets.append(set(gene_list))
        common = sorted(set.intersection(*all_gene_sets))
        nc = len(common)
        print(f"Common genes: {nc}", flush=True)
        
        if nc < 30:
            print("Too few common genes, using top 200 from pooled", flush=True)
            common = gene_list[:200]
            nc = len(common)
        
        def align_matrix(mat, genes, target_genes):
            g2i = {g: i for i, g in enumerate(genes)}
            n = len(target_genes)
            aligned = np.zeros((n, n))
            for i, gi in enumerate(target_genes):
                for j, gj in enumerate(target_genes):
                    if gi in g2i and gj in g2i:
                        aligned[i, j] = mat[g2i[gi], g2i[gj]]
            return aligned
        
        pooled_a = align_matrix(pooled_mat, gene_list, common)
        
        cssi_max = np.zeros((nc, nc))
        cssi_mean = np.zeros((nc, nc))
        total_w = sum(c for _, _, c in stratum_data.values())
        
        for ct, (mat, genes, count) in stratum_data.items():
            aligned = align_matrix(mat, genes, common)
            cssi_max = np.maximum(cssi_max, aligned)
            cssi_mean += (count / total_w) * aligned
        
        # Evaluate
        print(f"\n--- Results (n={n_cells}, genes={nc}) ---", flush=True)
        exp = {"n_cells": n_cells, "n_genes": nc}
        
        for gt_name, gt_edges in [("TRRUST", trrust), ("DoRothEA", dorothea)]:
            if not gt_edges:
                print(f"  {gt_name}: no ground truth available", flush=True)
                continue
            print(f"  vs {gt_name}:", flush=True)
            exp[f"{gt_name}_pooled"] = evaluate_grn(pooled_a, common, gt_edges, "POOLED")
            exp[f"{gt_name}_cssi_max"] = evaluate_grn(cssi_max, common, gt_edges, "CSSI-max")
            exp[f"{gt_name}_cssi_mean"] = evaluate_grn(cssi_mean, common, gt_edges, "CSSI-mean")
        
        results[str(n_cells)] = exp
    
    # Save
    with open(os.path.join(OUT_DIR, "cssi_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary table
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'Method':<12} {'N':<6} {'GT':<10} {'AUROC':<8} {'AUPRC':<8} {'F1':<8}", flush=True)
    print("-"*55, flush=True)
    for n in ["200", "500", "1000"]:
        for gt in ["TRRUST", "DoRothEA"]:
            for method in ["pooled", "cssi_max", "cssi_mean"]:
                r = results.get(n, {}).get(f"{gt}_{method}")
                if r:
                    print(f"{method:<12} {n:<6} {gt:<10} {r['auroc']:<8.4f} {r['auprc']:<8.4f} {r['best_f1']:<8.4f}", flush=True)
    
    print("\nDone!", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
