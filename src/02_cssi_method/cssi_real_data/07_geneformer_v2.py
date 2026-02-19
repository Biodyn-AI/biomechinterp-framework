#!/usr/bin/env python3
"""
CSSI validation on real Geneformer attention weights - v2.
Key fix: Use UNION of genes (not intersection), fill missing with 0.
Evaluate only on gene pairs where both genes appear in ground truth.
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

log_file = open(os.path.join(OUT_DIR, "cssi_run_v2.log"), "w", buffering=1)
sys.stdout = log_file
sys.stderr = log_file

print("Starting CSSI v2...", flush=True)

import geneformer
GF_DIR = os.path.dirname(geneformer.__file__)

with open(os.path.join(GF_DIR, "token_dictionary_gc104M.pkl"), "rb") as f:
    token_dict = pickle.load(f)
with open(os.path.join(GF_DIR, "gene_median_dictionary_gc104M.pkl"), "rb") as f:
    gene_median_dict = pickle.load(f)
with open(os.path.join(GF_DIR, "gene_name_id_dict_gc104M.pkl"), "rb") as f:
    gene_name_id_dict = pickle.load(f)

id_to_token = {int(v): k for k, v in token_dict.items()}
ensembl_to_name = {v: k for k, v in gene_name_id_dict.items()}
print(f"Vocab: {len(token_dict)}", flush=True)


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


def extract_attention(model, token_ids_list, device, batch_size=1, max_seq=256):
    model.eval()
    results = []
    for bs in range(0, len(token_ids_list), batch_size):
        batch = [t[:max_seq] for t in token_ids_list[bs:bs+batch_size]]
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


def build_gene_scores(attention_results):
    """
    Build a dictionary of (gene_i_ensembl, gene_j_ensembl) -> attention_score.
    Uses ALL gene pairs, not just top-k.
    """
    pair_scores = {}
    pair_counts = {}
    
    for tids, attn in attention_results:
        n = len(tids)
        for p1 in range(n):
            ens1 = id_to_token.get(tids[p1])
            if not ens1 or ens1.startswith('<'):
                continue
            for p2 in range(n):
                if p1 == p2:
                    continue
                ens2 = id_to_token.get(tids[p2])
                if not ens2 or ens2.startswith('<'):
                    continue
                pair = (ens1, ens2)
                pair_scores[pair] = pair_scores.get(pair, 0.0) + attn[p1, p2]
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    # Average
    for pair in pair_scores:
        pair_scores[pair] /= pair_counts[pair]
    
    return pair_scores


def evaluate_grn(pair_scores, gt_edges, name=""):
    """Evaluate: for all pairs in pair_scores, check against ground truth."""
    # Get all genes that appear in pair_scores
    all_genes = set()
    for g1, g2 in pair_scores:
        all_genes.add(g1)
        all_genes.add(g2)
    
    # Get genes that appear in ground truth
    gt_genes = set()
    for g1, g2 in gt_edges:
        gt_genes.add(g1)
        gt_genes.add(g2)
    
    # Intersection: genes in both predicted and ground truth
    common_genes = all_genes & gt_genes
    
    if len(common_genes) < 10:
        print(f"  {name}: Only {len(common_genes)} common genes with GT, skipping", flush=True)
        return None
    
    # Evaluate on all pairs where BOTH genes are in common_genes
    y_true = []
    y_score = []
    
    for pair, score in pair_scores.items():
        g1, g2 = pair
        if g1 in common_genes and g2 in common_genes:
            y_true.append(1 if pair in gt_edges else 0)
            y_score.append(score)
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    n_pos = y_true.sum()
    n_total = len(y_true)
    
    if n_pos == 0:
        print(f"  {name}: 0 positives among {n_total} pairs ({len(common_genes)} common genes)", flush=True)
        return None
    
    auroc = roc_auc_score(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(rec, prec)
    random_auprc = n_pos / n_total
    f1s = 2*prec*rec/(prec+rec+1e-10)
    
    result = {
        "auroc": float(auroc), "auprc": float(auprc), 
        "random_auprc": float(random_auprc), "best_f1": float(f1s.max()),
        "n_pos": int(n_pos), "n_pairs": int(n_total), "n_common_genes": int(len(common_genes))
    }
    print(f"  {name}: AUROC={auroc:.4f} AUPRC={auprc:.4f} (rand={random_auprc:.4f}) F1={f1s.max():.4f} pos={n_pos}/{n_total} genes={len(common_genes)}", flush=True)
    return result


def cssi_max_scores(stratum_scores_list):
    """Take max score across strata for each pair."""
    merged = {}
    for scores in stratum_scores_list:
        for pair, score in scores.items():
            merged[pair] = max(merged.get(pair, -np.inf), score)
    return merged


def cssi_mean_scores(stratum_scores_list, stratum_weights):
    """Weighted mean across strata."""
    merged = {}
    weight_sum = {}
    for scores, w in zip(stratum_scores_list, stratum_weights):
        for pair, score in scores.items():
            merged[pair] = merged.get(pair, 0.0) + w * score
            weight_sum[pair] = weight_sum.get(pair, 0.0) + w
    for pair in merged:
        merged[pair] /= weight_sum[pair]
    return merged


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    from transformers import BertForMaskedLM
    print("Loading model...", flush=True)
    model = BertForMaskedLM.from_pretrained("ctheodoris/Geneformer", output_attentions=True).to(device)
    model.half()  # fp16
    print(f"Model: {model.config.num_hidden_layers}L x {model.config.num_attention_heads}H", flush=True)
    
    print("Loading data...", flush=True)
    adata = sc.read_h5ad(DATA_PATH)
    
    # Gene name mapping (for TRRUST)
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
        print(f"TRRUST: {len(trrust)} mapped edges", flush=True)
        
        # Show some TFs
        tf_genes = set(g for g, _ in trrust)
        print(f"TRRUST TFs with Ensembl: {len(tf_genes)}", flush=True)
        # Sample
        sample_tfs = list(tf_genes)[:5]
        for tf in sample_tfs:
            name = ensembl_to_name.get(tf, 'unknown')
            in_vocab = tf in token_dict
            print(f"  TF {name} ({tf}): in_vocab={in_vocab}", flush=True)
    except Exception as e:
        print(f"TRRUST failed: {e}", flush=True)
        traceback.print_exc()
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
        ct_dist = dict(adata_s.obs['cell_type'].value_counts())
        print(f"Sampled: {adata_s.n_obs}, dist: {ct_dist}", flush=True)
        
        # Tokenize
        print("Tokenizing...", flush=True)
        tokens = []
        for i in range(adata_s.n_obs):
            expr = adata_s.X[i].toarray().flatten() if sparse.issparse(adata_s.X) else adata_s.X[i].flatten()
            tokens.append(tokenize_cell(expr, gene_ids))
        
        valid_idx = [i for i, t in enumerate(tokens) if len(t) > 10]
        tokens = [tokens[i] for i in valid_idx]
        cell_types = adata_s.obs['cell_type'].values[valid_idx]
        print(f"Valid: {len(tokens)}, median len: {np.median([len(t) for t in tokens]):.0f}", flush=True)
        
        # Debug: check what genes are in tokens
        all_token_ids = set()
        for t in tokens:
            all_token_ids.update(t[:256])  # only first 256 used
        gene_ensembls = set(id_to_token.get(tid) for tid in all_token_ids if id_to_token.get(tid, '').startswith('ENSG'))
        trrust_genes = set(g for pair in trrust for g in pair)
        overlap = gene_ensembls & trrust_genes
        print(f"Genes in attention: {len(gene_ensembls)}, TRRUST genes: {len(trrust_genes)}, overlap: {len(overlap)}", flush=True)
        
        # Extract attention
        print("Extracting attention...", flush=True)
        t0 = time.time()
        attn_results = extract_attention(model, tokens, device, batch_size=1, max_seq=256)
        print(f"Done in {time.time()-t0:.1f}s", flush=True)
        
        # POOLED scores
        print("Computing POOLED scores...", flush=True)
        t0 = time.time()
        pooled_scores = build_gene_scores(attn_results)
        print(f"POOLED: {len(pooled_scores)} gene pairs in {time.time()-t0:.1f}s", flush=True)
        
        # CSSI per cell type
        print("Computing CSSI scores...", flush=True)
        stratum_scores_list = []
        stratum_weights = []
        for ct in set(cell_types):
            mask = cell_types == ct
            n_ct = mask.sum()
            if n_ct >= 5:
                ct_results = [attn_results[i] for i in range(len(attn_results)) if mask[i]]
                ct_scores = build_gene_scores(ct_results)
                stratum_scores_list.append(ct_scores)
                stratum_weights.append(n_ct)
                print(f"  {ct}: {n_ct} cells, {len(ct_scores)} pairs", flush=True)
        
        cssi_max = cssi_max_scores(stratum_scores_list)
        cssi_mean = cssi_mean_scores(stratum_scores_list, stratum_weights)
        print(f"CSSI-max: {len(cssi_max)} pairs, CSSI-mean: {len(cssi_mean)} pairs", flush=True)
        
        # Evaluate
        print(f"\n--- Evaluation ---", flush=True)
        exp = {"n_cells": n_cells}
        if trrust:
            exp["TRRUST_pooled"] = evaluate_grn(pooled_scores, trrust, "POOLED")
            exp["TRRUST_cssi_max"] = evaluate_grn(cssi_max, trrust, "CSSI-max")
            exp["TRRUST_cssi_mean"] = evaluate_grn(cssi_mean, trrust, "CSSI-mean")
        
        results[str(n_cells)] = exp
    
    # Save
    with open(os.path.join(OUT_DIR, "cssi_results_v2.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'Method':<12} {'N':<6} {'AUROC':<8} {'AUPRC':<8} {'F1':<8} {'pos':<8}", flush=True)
    print("-"*50, flush=True)
    for n in ["200", "500", "1000"]:
        for method in ["TRRUST_pooled", "TRRUST_cssi_max", "TRRUST_cssi_mean"]:
            r = results.get(n, {}).get(method)
            if r:
                mname = method.replace("TRRUST_", "")
                print(f"{mname:<12} {n:<6} {r['auroc']:<8.4f} {r['auprc']:<8.4f} {r['best_f1']:<8.4f} {r['n_pos']:<8}", flush=True)
    
    print("\nDone!", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
