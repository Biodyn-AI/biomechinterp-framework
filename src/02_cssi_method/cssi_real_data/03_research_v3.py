#!/usr/bin/env python3
"""
CSSI Research v3: Layers 5-17 + head-level analysis on best layers.
Incorporates results from v2 (layers 0-4).
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

# Previous results from v2 (layers 0-4)
PREV_RESULTS = {
    0: {"pooled_auroc": 0.5524, "best_cssi": "cssi_mean", "best_cssi_auroc": 0.5661},
    1: {"pooled_auroc": 0.6004, "best_cssi": "cssi_mean", "best_cssi_auroc": 0.6075},
    2: {"pooled_auroc": 0.5643, "best_cssi": "cssi_mean", "best_cssi_auroc": 0.5866},
    3: {"pooled_auroc": 0.5727, "best_cssi": "cssi_mean", "best_cssi_auroc": 0.5801},
    4: {"pooled_auroc": 0.6101, "best_cssi": "cssi_range", "best_cssi_auroc": 0.6090},
}

log_file = open(os.path.join(OUT_DIR, "cssi_research_v3.log"), "w", buffering=1)
def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

log("=== CSSI Research v3: Layers 5-17 + Head Analysis ===")

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


def build_gene_matrix(model, token_ids_list, device, top_genes,
                      layer_idx=None, head_idx=None, cell_types=None):
    model.eval()
    g2i = {g: i for i, g in enumerate(top_genes)}
    n = len(top_genes)
    
    score_sum = np.zeros((n, n), dtype=np.float64)
    count_mat = np.zeros((n, n), dtype=np.float64)
    
    ct_scores = {}
    ct_counts_mat = {}
    if cell_types is not None:
        for ct in set(cell_types):
            ct_scores[ct] = np.zeros((n, n), dtype=np.float64)
            ct_counts_mat[ct] = np.zeros((n, n), dtype=np.float64)
    
    for idx in range(len(token_ids_list)):
        tids = token_ids_list[idx][:MAX_SEQ]
        ml = len(tids)
        input_ids = torch.tensor([tids], dtype=torch.long, device=device)
        attn_mask = torch.ones(1, ml, dtype=torch.long, device=device)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            out = model.bert(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
        
        if layer_idx is not None:
            layer_attn = out.attentions[layer_idx][0]
            if head_idx is not None:
                attn = layer_attn[head_idx].float().cpu().numpy()
            else:
                attn = layer_attn.mean(dim=0).float().cpu().numpy()
        else:
            all_attn = torch.stack([a[0] for a in out.attentions])
            attn = all_attn.mean(dim=(0, 1)).float().cpu().numpy()
        
        del out
        if idx % 200 == 0:
            torch.cuda.empty_cache()
        
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
                    score_sum[i, j] += sub_attn[a, b]
                    count_mat[i, j] += 1
        
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
    if not ct_matrices or len(ct_matrices) < 2:
        return {}
    ct_stack = np.stack(list(ct_matrices.values()))
    return {
        "cssi_max": np.max(ct_stack, axis=0),
        "cssi_mean": np.mean(ct_stack, axis=0),
        "cssi_median": np.median(ct_stack, axis=0),
        "cssi_std": np.std(ct_stack, axis=0),
        "cssi_range": np.max(ct_stack, axis=0) - np.min(ct_stack, axis=0),
        "cssi_deviation": np.max(np.abs(ct_stack - pooled[np.newaxis]), axis=0),
    }


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
    
    # Sample cells (same seed as v2)
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
    
    freq = Counter()
    for t in tokens:
        for tid in t:
            if tid > 3:
                freq[tid] += 1
    top_genes = [g for g, _ in freq.most_common(TOP_K)]
    
    # Start with previous results
    all_layer_results = {}
    for layer, res in PREV_RESULTS.items():
        all_layer_results[layer] = res
    
    all_aurocs = []
    for layer, res in PREV_RESULTS.items():
        all_aurocs.append((f"L{layer}_pooled", res["pooled_auroc"], 0))
        all_aurocs.append((f"L{layer}_{res['best_cssi']}", res["best_cssi_auroc"], 0))
    
    # ===== PHASE 1: Layers 5-17 =====
    log("\n=== PHASE 1: Per-layer (layers 5-17) ===")
    for layer in range(5, n_layers):
        t0 = time.time()
        pooled, ct_mats, genes = build_gene_matrix(
            model, tokens, device, top_genes, layer_idx=layer, cell_types=cell_types_arr)
        auroc, n_pos = evaluate_auroc(pooled, genes, trrust)
        
        cssi = cssi_aggregations(ct_mats, pooled)
        best_cssi_name, best_cssi_auroc = "", 0
        cssi_detail = {}
        for cname, cmat in cssi.items():
            ca, _ = evaluate_auroc(cmat, genes, trrust)
            cssi_detail[cname] = ca
            if ca and ca > best_cssi_auroc:
                best_cssi_auroc = ca
                best_cssi_name = cname
        
        dt = time.time() - t0
        all_layer_results[layer] = {
            "pooled_auroc": auroc, "n_pos": n_pos,
            "best_cssi": best_cssi_name, "best_cssi_auroc": best_cssi_auroc,
            "cssi_detail": cssi_detail
        }
        all_aurocs.append((f"L{layer}_pooled", auroc, n_pos))
        if best_cssi_auroc > 0:
            all_aurocs.append((f"L{layer}_{best_cssi_name}", best_cssi_auroc, n_pos))
        # Also add all CSSI variants to ranking
        for cname, ca in cssi_detail.items():
            if ca and cname != best_cssi_name:
                all_aurocs.append((f"L{layer}_{cname}", ca, n_pos))
        
        log(f"  Layer {layer:2d}: pooled={auroc:.4f}, best_cssi={best_cssi_name}={best_cssi_auroc:.4f} ({dt:.0f}s)")
        gc.collect()
        torch.cuda.empty_cache()
    
    # Find best layers across ALL 0-17
    best_layers = sorted(all_layer_results.keys(), 
                         key=lambda l: max(all_layer_results[l]["pooled_auroc"] or 0,
                                           all_layer_results[l].get("best_cssi_auroc", 0) or 0),
                         reverse=True)
    log(f"\nTop-5 layers (by best score): {best_layers[:5]}")
    for l in best_layers[:5]:
        r = all_layer_results[l]
        log(f"  L{l}: pooled={r['pooled_auroc']:.4f}, {r['best_cssi']}={r.get('best_cssi_auroc',0):.4f}")
    
    # ===== PHASE 2: Head-level analysis on top 3 layers =====
    log("\n=== PHASE 2: Per-head analysis (top 3 layers) ===")
    head_results = {}
    for layer in best_layers[:3]:
        if layer < 5:
            # Need to actually compute these (weren't done in v2)
            pass
        log(f"\n  --- Layer {layer} heads ---")
        for head in range(n_heads):
            t0 = time.time()
            pooled_h, _, genes_h = build_gene_matrix(
                model, tokens, device, top_genes, layer_idx=layer, head_idx=head)
            auroc_h, n_pos = evaluate_auroc(pooled_h, genes_h, trrust)
            key = f"L{layer}_H{head}"
            head_results[key] = {"auroc": auroc_h, "n_pos": n_pos}
            all_aurocs.append((key, auroc_h, n_pos))
            dt = time.time() - t0
            log(f"    {key}: AUROC={auroc_h:.4f} ({dt:.0f}s)")
        gc.collect()
        torch.cuda.empty_cache()
    
    # Find best heads
    best_heads = sorted(head_results, key=lambda k: head_results[k].get("auroc", 0) or 0, reverse=True)
    log(f"\nTop-10 heads:")
    for h in best_heads[:10]:
        log(f"  {h}: {head_results[h]['auroc']:.4f}")
    
    # ===== PHASE 3: CSSI on best heads =====
    log("\n=== PHASE 3: CSSI on top 5 heads ===")
    head_cssi_results = {}
    for hkey in best_heads[:5]:
        parts = hkey.split("_")
        layer = int(parts[0][1:])
        head = int(parts[1][1:])
        
        pooled_h, ct_mats_h, genes_h = build_gene_matrix(
            model, tokens, device, top_genes, layer_idx=layer, head_idx=head,
            cell_types=cell_types_arr)
        
        cssi_h = cssi_aggregations(ct_mats_h, pooled_h)
        for cname, cmat in cssi_h.items():
            ca, n_pos = evaluate_auroc(cmat, genes_h, trrust)
            rkey = f"{hkey}_{cname}"
            head_cssi_results[rkey] = {"auroc": ca, "n_pos": n_pos}
            all_aurocs.append((rkey, ca, n_pos))
            if ca and ca > 0.58:
                log(f"  {rkey}: {ca:.4f}")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # ===== PHASE 4: Multi-layer combos =====
    log("\n=== PHASE 4: Multi-layer combinations ===")
    # Get top layers by pooled AUROC
    top_by_pooled = sorted(all_layer_results.keys(),
                           key=lambda l: all_layer_results[l]["pooled_auroc"] or 0, reverse=True)
    
    multi_results = {}
    for n_top in [2, 3, 5]:
        layers_combo = top_by_pooled[:n_top]
        mats = []
        for layer in layers_combo:
            p, _, g = build_gene_matrix(model, tokens, device, top_genes, layer_idx=layer)
            mats.append(p)
        
        # Average
        avg_mat = np.mean(mats, axis=0)
        auroc, n_pos = evaluate_auroc(avg_mat, g, trrust)
        key = f"avg_top{n_top}"
        multi_results[key] = {"auroc": auroc, "layers": [int(l) for l in layers_combo]}
        all_aurocs.append((key, auroc, n_pos))
        log(f"  {key} {list(layers_combo)}: {auroc:.4f}")
        
        # Max
        max_mat = np.max(mats, axis=0)
        auroc2, _ = evaluate_auroc(max_mat, g, trrust)
        key2 = f"max_top{n_top}"
        multi_results[key2] = {"auroc": auroc2, "layers": [int(l) for l in layers_combo]}
        all_aurocs.append((key2, auroc2, n_pos))
        log(f"  {key2} {list(layers_combo)}: {auroc2:.4f}")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # ===== FINAL RANKING =====
    log("\n" + "=" * 60)
    log("FINAL RANKING (ALL METHODS)")
    log("=" * 60)
    
    all_aurocs.sort(key=lambda x: (x[1] or 0), reverse=True)
    log(f"\n{'Method':<50} {'AUROC':<8}")
    log("-" * 58)
    for name, auroc, npos in all_aurocs[:40]:
        if auroc:
            log(f"{name:<50} {auroc:.4f}")
    
    baseline = 0.543
    best = all_aurocs[0] if all_aurocs else None
    if best and best[1]:
        log(f"\nBaseline (all-layers pooled): {baseline}")
        log(f"Best: {best[0]} = {best[1]:.4f} ({best[1]-baseline:+.4f})")
    
    # ===== SAVE RESULTS =====
    results = {
        "per_layer": {str(k): {kk: vv for kk, vv in v.items() if kk != 'cssi_detail'} 
                      for k, v in all_layer_results.items()},
        "per_layer_detail": {str(k): v.get("cssi_detail", {}) for k, v in all_layer_results.items()},
        "per_head": head_results,
        "head_cssi": head_cssi_results,
        "multi_layer": multi_results,
        "ranking": [{"method": n, "auroc": a} for n, a, _ in all_aurocs[:50] if a],
        "best": {"method": best[0], "auroc": best[1]} if best and best[1] else None,
        "baseline": baseline,
    }
    
    with open(os.path.join(OUT_DIR, "cssi_research_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    log("\nResults saved to cssi_research_results.json")
    
    log_file.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        traceback.print_exc(file=log_file)
        sys.exit(1)
