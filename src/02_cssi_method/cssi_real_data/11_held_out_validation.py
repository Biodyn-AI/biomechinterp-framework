#!/usr/bin/env python3
"""
CSSI Held-Out Validation Experiment
Addresses reviewer concern about layer selection on same data used for evaluation.

Approach:
1. Split 497 cells into two stratified halves 
2. Run layer identification on first half (training), evaluate on second half (test)
3. Reverse: train on second half, test on first half
4. Compare: Do same layers/heads emerge? Does test AUROC match train AUROC?
"""
import os, sys, pickle, json, time, traceback, gc
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict

OUT_DIR = "/mnt/d/openclaw/biodyn-nmi-paper/cssi_real"
DATA_PATH = "/mnt/d/openclaw/intelligence-augmentation/data/brain_scrna/DLPFC_11k.h5ad"
TOP_K = 1000
MAX_SEQ = 256
N_CELLS = 500  # Use same as original

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

log_file = open(os.path.join(OUT_DIR, "held_out_validation.log"), "w", buffering=1)
log("=== CSSI Held-Out Validation Experiment ===")

# Load necessary data/models (same as original script)
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
    """Build gene-gene matrix from tokens list (memory efficient version)"""
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
        if idx % 50 == 0:
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
    
    mask = count_mat > 0
    pooled = np.zeros_like(score_sum)
    pooled[mask] = score_sum[mask] / count_mat[mask]
    
    ct_mats = None
    if ct_indices is not None:
        ct_mats = {}
        for ct in ct_scores:
            m = ct_counts[ct] > 0
            ct_mat = np.zeros((n_genes, n_genes), dtype=np.float64)
            if m.sum() > 0:
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

def analyze_layer_performance(model, tokens_list, cell_types_arr, device, g2i, n_genes, 
                              gene_list, trrust, split_name):
    """Analyze all layers and return ranking"""
    log(f"\n--- Analyzing {split_name} ---")
    layer_results = {}
    
    # Test key layers based on original results
    target_layers = [10, 12, 13, 14, 15, 16, 17]  # Focus on top performers
    
    # Build cell-type indices for CSSI
    unique_cts = sorted(set(cell_types_arr))
    ct_indices = {ct: [i for i, c in enumerate(cell_types_arr) if c == ct] for ct in unique_cts}
    ct_indices = {ct: idx for ct, idx in ct_indices.items() if len(idx) >= 5}  # Require min 5 cells
    
    for layer_idx in target_layers:
        log(f"  Processing Layer {layer_idx}...")
        start_time = time.time()
        
        # Get pooled attention matrix
        pooled_mat, ct_mats = build_gene_matrix(model, tokens_list, device, g2i, n_genes,
                                                 layer_idx, ct_indices=ct_indices)
        
        # Evaluate pooled performance
        pooled_auroc, n_pos = evaluate_auroc(pooled_mat, gene_list, trrust)
        
        # Calculate CSSI variants if we have sufficient cell types
        cssi_results = {}
        if len(ct_mats) >= 3:  # Need at least 3 cell types for meaningful CSSI
            valid_cts = [ct for ct, mat in ct_mats.items() if np.sum(mat > 0) > 100]
            if len(valid_cts) >= 3:
                ct_stack = np.stack([ct_mats[ct] for ct in valid_cts])
                
                cssi_variants = {
                    "cssi_mean": np.mean(ct_stack, axis=0),
                    "cssi_range": np.max(ct_stack, axis=0) - np.min(ct_stack, axis=0),
                    "cssi_deviation": np.max(np.abs(ct_stack - pooled_mat[np.newaxis]), axis=0),
                }
                
                for cssi_name, cssi_mat in cssi_variants.items():
                    cssi_auroc, _ = evaluate_auroc(cssi_mat, gene_list, trrust)
                    cssi_results[cssi_name] = cssi_auroc
        
        # Find best CSSI method for this layer
        best_cssi = None
        best_cssi_auroc = None
        if cssi_results:
            best_cssi = max(cssi_results.keys(), key=lambda k: cssi_results[k] or 0)
            best_cssi_auroc = cssi_results[best_cssi]
        
        layer_results[layer_idx] = {
            "pooled_auroc": pooled_auroc,
            "best_cssi_method": best_cssi,
            "best_cssi_auroc": best_cssi_auroc,
            "cssi_results": cssi_results,
            "n_cell_types": len(ct_indices),
            "n_pos_edges": n_pos
        }
        
        elapsed = time.time() - start_time
        log(f"    Pooled AUROC: {pooled_auroc:.4f}")
        if best_cssi_auroc:
            log(f"    Best CSSI ({best_cssi}): {best_cssi_auroc:.4f}")
        log(f"    Time: {elapsed:.1f}s")
        
        del pooled_mat, ct_mats
        gc.collect()
        torch.cuda.empty_cache()
    
    return layer_results

def analyze_top_heads(model, tokens_list, device, g2i, n_genes, gene_list, trrust, 
                      top_layers, split_name, n_heads=18):
    """Analyze individual heads in top layers"""
    log(f"\n--- Analyzing top heads for {split_name} ---")
    head_results = {}
    
    for layer_idx in top_layers[:2]:  # Analyze heads only for top 2 layers
        log(f"  Layer {layer_idx} heads...")
        for head_idx in range(n_heads):
            head_mat, _ = build_gene_matrix(model, tokens_list, device, g2i, n_genes,
                                           layer_idx, head_idx=head_idx)
            head_auroc, n_pos = evaluate_auroc(head_mat, gene_list, trrust)
            head_key = f"L{layer_idx}_H{head_idx}"
            head_results[head_key] = {
                "auroc": head_auroc,
                "n_pos": n_pos
            }
            
            del head_mat
            gc.collect()
            if head_idx % 6 == 5:  # Clean up every 6 heads
                torch.cuda.empty_cache()
        
        # Report top heads for this layer
        layer_heads = {k: v["auroc"] for k, v in head_results.items() 
                       if k.startswith(f"L{layer_idx}_") and v["auroc"] is not None}
        if layer_heads:
            top_3_heads = sorted(layer_heads.items(), key=lambda x: x[1], reverse=True)[:3]
            log(f"    Top 3 heads: {[(h, f'{a:.4f}') for h, a in top_3_heads]}")
    
    return head_results

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    
    # Load model
    from transformers import BertForMaskedLM
    log("Loading model...")
    model = BertForMaskedLM.from_pretrained("ctheodoris/Geneformer", 
                                           output_attentions=True).to(device).half()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    log(f"Model: {n_layers}L x {n_heads}H")
    
    # Load and process data (same as original script)
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
    
    # Filter and sample data (reproduce original sampling)
    ct_counts = adata.obs['cell_type'].value_counts()
    valid_cts = ct_counts[ct_counts >= 50].index.tolist()
    adata_filt = adata[adata.obs['cell_type'].isin(valid_cts)].copy()
    gene_ids = np.array(adata_filt.var_names)
    
    np.random.seed(42)  # Same seed as original
    sampled = []
    for ct in valid_cts:
        idx = np.where(adata_filt.obs['cell_type'] == ct)[0]
        n_ct = max(1, int(N_CELLS * len(idx) / adata_filt.n_obs))
        sampled.extend(np.random.choice(idx, min(n_ct, len(idx)), replace=False))
    if len(sampled) > N_CELLS:
        sampled = list(np.random.choice(sampled, N_CELLS, replace=False))
    
    adata_s = adata_filt[sampled].copy()
    cell_types_full = list(adata_s.obs['cell_type'].values)
    
    # Tokenize all cells
    log("Tokenizing all cells...")
    tokens_full = []
    for i in range(adata_s.n_obs):
        expr = adata_s.X[i].toarray().flatten() if sparse.issparse(adata_s.X) else adata_s.X[i].flatten()
        tokens_full.append(tokenize_cell(expr, gene_ids))
    
    # Filter valid cells (same as original)
    valid_idx = [i for i, t in enumerate(tokens_full) if len(t) > 10]
    tokens_full = [tokens_full[i] for i in valid_idx]
    cell_types_full = [cell_types_full[i] for i in valid_idx]
    
    log(f"Total valid cells: {len(tokens_full)}")
    log(f"Cell type distribution: {Counter(cell_types_full)}")
    
    # Setup gene mapping (same as original)
    freq = Counter()
    for t in tokens_full:
        for tid in t:
            if tid > 3:
                freq[tid] += 1
    top_genes = [g for g, _ in freq.most_common(TOP_K)]
    g2i = {g: i for i, g in enumerate(top_genes)}
    n_genes = len(top_genes)
    gene_list = [id_to_token.get(g, f"UNK_{g}") for g in top_genes]
    
    # Free large objects
    del adata, adata_filt, adata_s
    gc.collect()
    
    # ===== HELD-OUT VALIDATION EXPERIMENT =====
    log("\n" + "="*60)
    log("HELD-OUT VALIDATION EXPERIMENT")
    log("="*60)
    
    # Create stratified split
    log("Creating stratified train/test split...")
    cell_indices = list(range(len(tokens_full)))
    X_dummy = np.zeros((len(tokens_full), 1))  # Dummy features for sklearn
    
    # Stratified split by cell type
    train_idx, test_idx = train_test_split(
        cell_indices, 
        test_size=0.5, 
        stratify=cell_types_full,
        random_state=42
    )
    
    log(f"Train set: {len(train_idx)} cells")
    log(f"Test set: {len(test_idx)} cells")
    
    # Split data
    tokens_train = [tokens_full[i] for i in train_idx]
    tokens_test = [tokens_full[i] for i in test_idx]
    cell_types_train = [cell_types_full[i] for i in train_idx]
    cell_types_test = [cell_types_full[i] for i in test_idx]
    
    log(f"Train cell types: {Counter(cell_types_train)}")
    log(f"Test cell types: {Counter(cell_types_test)}")
    
    results = {
        "experiment_info": {
            "total_cells": len(tokens_full),
            "train_cells": len(tokens_train),
            "test_cells": len(tokens_test),
            "n_genes": n_genes,
            "trrust_edges": len(trrust),
            "train_cell_types": dict(Counter(cell_types_train)),
            "test_cell_types": dict(Counter(cell_types_test))
        },
        "split_a": {},  # Train on first half, test on second half
        "split_b": {},  # Train on second half, test on first half
        "comparison": {}
    }
    
    # ===== SPLIT A: Train on first half, test on second half =====
    log("\n" + "="*50)
    log("SPLIT A: Train on first half, test on second half")
    log("="*50)
    
    # Training phase - identify best layers/methods
    train_results_a = analyze_layer_performance(
        model, tokens_train, cell_types_train, device, g2i, n_genes, 
        gene_list, trrust, "SPLIT A - TRAIN"
    )
    
    # Find top performing layers from training
    layer_scores = [(layer, res["best_cssi_auroc"] or res["pooled_auroc"] or 0) 
                   for layer, res in train_results_a.items()]
    top_layers_a = [layer for layer, score in sorted(layer_scores, key=lambda x: x[1], reverse=True)[:3]]
    
    log(f"\nSplit A - Top layers identified from training: {top_layers_a}")
    for layer in top_layers_a:
        res = train_results_a[layer]
        log(f"  Layer {layer}: pooled={res['pooled_auroc']:.4f}, "
            f"best_cssi={res['best_cssi_auroc']:.4f if res['best_cssi_auroc'] is not None else 'N/A'}")
    
    # Test phase - evaluate identified layers on test set
    test_results_a = analyze_layer_performance(
        model, tokens_test, cell_types_test, device, g2i, n_genes, 
        gene_list, trrust, "SPLIT A - TEST"
    )
    
    # Analyze heads for top layers
    train_heads_a = analyze_top_heads(
        model, tokens_train, device, g2i, n_genes, gene_list, trrust,
        top_layers_a, "SPLIT A - TRAIN", n_heads
    )
    
    test_heads_a = analyze_top_heads(
        model, tokens_test, device, g2i, n_genes, gene_list, trrust,
        top_layers_a, "SPLIT A - TEST", n_heads
    )
    
    results["split_a"] = {
        "train_layers": train_results_a,
        "test_layers": test_results_a,
        "train_heads": train_heads_a,
        "test_heads": test_heads_a,
        "top_layers_identified": top_layers_a
    }
    
    # ===== SPLIT B: Train on second half, test on first half =====
    log("\n" + "="*50)
    log("SPLIT B: Train on second half, test on first half") 
    log("="*50)
    
    # Training phase
    train_results_b = analyze_layer_performance(
        model, tokens_test, cell_types_test, device, g2i, n_genes,
        gene_list, trrust, "SPLIT B - TRAIN"
    )
    
    # Find top performing layers
    layer_scores_b = [(layer, res["best_cssi_auroc"] or res["pooled_auroc"] or 0) 
                     for layer, res in train_results_b.items()]
    top_layers_b = [layer for layer, score in sorted(layer_scores_b, key=lambda x: x[1], reverse=True)[:3]]
    
    log(f"\nSplit B - Top layers identified from training: {top_layers_b}")
    for layer in top_layers_b:
        res = train_results_b[layer]
        log(f"  Layer {layer}: pooled={res['pooled_auroc']:.4f}, "
            f"best_cssi={res['best_cssi_auroc']:.4f if res['best_cssi_auroc'] is not None else 'N/A'}")
    
    # Test phase
    test_results_b = analyze_layer_performance(
        model, tokens_train, cell_types_train, device, g2i, n_genes,
        gene_list, trrust, "SPLIT B - TEST"
    )
    
    # Analyze heads
    train_heads_b = analyze_top_heads(
        model, tokens_test, device, g2i, n_genes, gene_list, trrust,
        top_layers_b, "SPLIT B - TRAIN", n_heads
    )
    
    test_heads_b = analyze_top_heads(
        model, tokens_train, device, g2i, n_genes, gene_list, trrust,
        top_layers_b, "SPLIT B - TEST", n_heads
    )
    
    results["split_b"] = {
        "train_layers": train_results_b,
        "test_layers": test_results_b,
        "train_heads": train_heads_b,
        "test_heads": test_heads_b,
        "top_layers_identified": top_layers_b
    }
    
    # ===== COMPARISON AND VALIDATION =====
    log("\n" + "="*60)
    log("CROSS-VALIDATION COMPARISON")
    log("="*60)
    
    # Check consistency of layer identification
    layers_consistent = set(top_layers_a) == set(top_layers_b)
    layers_overlap = len(set(top_layers_a) & set(top_layers_b))
    
    log(f"Top layers Split A: {top_layers_a}")
    log(f"Top layers Split B: {top_layers_b}")
    log(f"Layers consistent: {layers_consistent}")
    log(f"Layers overlap: {layers_overlap}/3")
    
    # Performance generalization analysis
    generalization_results = {}
    for split_name, split_data in [("A", results["split_a"]), ("B", results["split_b"])]:
        gen_results = {}
        for layer in split_data["top_layers_identified"]:
            train_res = split_data["train_layers"][layer]
            test_res = split_data["test_layers"][layer]
            
            # Compare train vs test performance
            train_pooled = train_res["pooled_auroc"] or 0
            test_pooled = test_res["pooled_auroc"] or 0
            
            train_cssi = train_res["best_cssi_auroc"] or 0
            test_cssi = test_res["best_cssi_auroc"] or 0
            
            gen_results[f"layer_{layer}"] = {
                "train_pooled": train_pooled,
                "test_pooled": test_pooled,
                "pooled_drop": train_pooled - test_pooled,
                "train_cssi": train_cssi,
                "test_cssi": test_cssi,
                "cssi_drop": train_cssi - test_cssi,
                "train_best_method": train_res["best_cssi_method"],
                "test_best_method": test_res["best_cssi_method"]
            }
        
        generalization_results[f"split_{split_name}"] = gen_results
        
        # Report for this split
        log(f"\nSplit {split_name} - Performance generalization:")
        for layer_key, gen_res in gen_results.items():
            layer_num = layer_key.split("_")[1]
            log(f"  Layer {layer_num}:")
            log(f"    Pooled: {gen_res['train_pooled']:.4f} -> {gen_res['test_pooled']:.4f} "
                f"(drop: {gen_res['pooled_drop']:.4f})")
            if gen_res['train_cssi'] > 0:
                log(f"    CSSI: {gen_res['train_cssi']:.4f} -> {gen_res['test_cssi']:.4f} "
                    f"(drop: {gen_res['cssi_drop']:.4f})")
    
    # Head consistency analysis
    log(f"\nHead consistency analysis:")
    for split_name, split_data in [("A", results["split_a"]), ("B", results["split_b"])]:
        for layer in split_data["top_layers_identified"][:2]:  # Top 2 layers only
            train_heads = {k: v["auroc"] for k, v in split_data["train_heads"].items() 
                          if k.startswith(f"L{layer}_") and v["auroc"] is not None}
            test_heads = {k: v["auroc"] for k, v in split_data["test_heads"].items() 
                         if k.startswith(f"L{layer}_") and v["auroc"] is not None}
            
            if train_heads and test_heads:
                top_train_head = max(train_heads.keys(), key=lambda k: train_heads[k])
                top_test_head = max(test_heads.keys(), key=lambda k: test_heads[k])
                
                log(f"  Split {split_name}, Layer {layer}:")
                log(f"    Top train head: {top_train_head} ({train_heads[top_train_head]:.4f})")
                log(f"    Top test head: {top_test_head} ({test_heads[top_test_head]:.4f})")
                log(f"    Same head: {top_train_head == top_test_head}")
    
    results["comparison"] = {
        "layers_consistent": layers_consistent,
        "layers_overlap": layers_overlap,
        "generalization": generalization_results
    }
    
    # ===== FINAL SUMMARY =====
    log("\n" + "="*60)
    log("HELD-OUT VALIDATION SUMMARY")
    log("="*60)
    
    # Overall conclusions
    avg_pooled_drop_a = np.mean([res["pooled_drop"] for res in generalization_results["split_A"].values()])
    avg_pooled_drop_b = np.mean([res["pooled_drop"] for res in generalization_results["split_B"].values()])
    avg_pooled_drop = (avg_pooled_drop_a + avg_pooled_drop_b) / 2
    
    avg_cssi_drop_a = np.mean([res["cssi_drop"] for res in generalization_results["split_A"].values() 
                               if res["train_cssi"] > 0])
    avg_cssi_drop_b = np.mean([res["cssi_drop"] for res in generalization_results["split_B"].values()
                               if res["train_cssi"] > 0])
    avg_cssi_drop = (avg_cssi_drop_a + avg_cssi_drop_b) / 2
    
    log(f"Layer identification consistency: {layers_overlap}/3 layers overlap")
    log(f"Average performance drop (pooled): {avg_pooled_drop:.4f}")
    log(f"Average performance drop (CSSI): {avg_cssi_drop:.4f}")
    
    # Check if layers 13-14 consistently identified
    original_best = [13, 14]
    layers_13_14_in_a = len(set(original_best) & set(top_layers_a))
    layers_13_14_in_b = len(set(original_best) & set(top_layers_b))
    
    log(f"\nOriginal best layers (13, 14) validation:")
    log(f"  Found in Split A top-3: {layers_13_14_in_a}/2")
    log(f"  Found in Split B top-3: {layers_13_14_in_b}/2")
    
    validation_passed = (
        layers_overlap >= 2 and  # At least 2/3 layers consistent
        avg_pooled_drop < 0.05 and  # Small performance drop
        layers_13_14_in_a >= 1 and layers_13_14_in_b >= 1  # Original layers still in top-3
    )
    
    log(f"\nValidation assessment: {'PASSED' if validation_passed else 'FAILED'}")
    
    results["summary"] = {
        "layers_overlap": layers_overlap,
        "avg_pooled_drop": avg_pooled_drop,
        "avg_cssi_drop": avg_cssi_drop,
        "original_layers_in_split_a": layers_13_14_in_a,
        "original_layers_in_split_b": layers_13_14_in_b,
        "validation_passed": validation_passed
    }
    
    # Save results
    output_file = os.path.join(OUT_DIR, "held_out_validation_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log(f"\nResults saved to: {output_file}")
    log("Held-out validation experiment completed!")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        traceback.print_exc(file=log_file)
        sys.exit(1)