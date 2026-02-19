"""
CSSI Real Data Validation using Geneformer attention weights
Tests whether Cell-State Stratified Interpretability improves GRN recovery
on REAL foundation model attention (not synthetic proxies).
"""
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import os, sys, pickle, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import scanpy as sc
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from datasets import load_from_disk
from transformers import BertForMaskedLM, BertConfig

print("=== CSSI Real Data Validation ===")
print(f"CUDA: {torch.cuda.is_available()}")

# Paths
data_path = "/mnt/d/openclaw/intelligence-augmentation/data/brain_scrna/DLPFC_11k.h5ad"
model_path = "/mnt/d/openclaw/intelligence-augmentation/models/Geneformer/Geneformer-V1-10M"
token_dict_path = os.path.expanduser("~/.local/lib/python3.12/site-packages/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl")
tokenized_path = "/mnt/d/openclaw/intelligence-augmentation/analysis/results/insilico_wsl/brain.dataset"
output_dir = "/mnt/d/openclaw/biodyn-nmi-paper/cssi_real/results"
os.makedirs(output_dir, exist_ok=True)

# Load token dictionary
with open(token_dict_path, "rb") as f:
    token_dict = pickle.load(f)
reverse_dict = {v: k for k, v in token_dict.items()}

# Load TRRUST reference (human TF-target pairs)
print("Loading TRRUST reference...")
trrust_url = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
try:
    import urllib.request
    trrust_path = os.path.join(output_dir, "trrust_human.tsv")
    if not os.path.exists(trrust_path):
        urllib.request.urlretrieve(trrust_url, trrust_path)
    trrust_pairs = set()
    with open(trrust_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                trrust_pairs.add((parts[0], parts[1]))
    print(f"  TRRUST: {len(trrust_pairs)} TF-target pairs")
except Exception as e:
    print(f"  TRRUST download failed: {e}, using empty set")
    trrust_pairs = set()

# Load gene name mapping
gene_name_path = os.path.expanduser("~/.local/lib/python3.12/site-packages/geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl")
with open(gene_name_path, "rb") as f:
    gene_name_dict = pickle.load(f)  # name -> ensembl
gene_id_to_name = {v: k for k, v in gene_name_dict.items()}

# Load original h5ad for cell type labels
print("Loading cell type labels...")
adata = sc.read_h5ad(data_path)
# Use prepared 500-cell version
prep_path = "/mnt/d/openclaw/intelligence-augmentation/analysis/results/insilico_wsl/prepared_data/brain_500.h5ad"
adata_sub = sc.read_h5ad(prep_path)
cell_types = adata_sub.obs['cell_type'].values
print(f"  Cell types: {dict(zip(*np.unique(cell_types, return_counts=True)))}")

# Load tokenized dataset
print("Loading tokenized dataset...")
ds = load_from_disk(tokenized_path)
print(f"  Dataset: {len(ds)} cells")

# Load Geneformer model
print("Loading Geneformer model...")
model = BertForMaskedLM.from_pretrained(model_path, output_attentions=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"  Model on {device}")

# Extract attention weights for each cell
print("Extracting attention weights...")
n_cells = min(len(ds), 500)
all_attention_edges = []  # list of dicts: {(gene_i, gene_j): attention_score}

for cell_idx in range(n_cells):
    if cell_idx % 50 == 0:
        print(f"  Cell {cell_idx}/{n_cells}")
    
    input_ids = torch.tensor([ds[cell_idx]['input_ids'][:2048]]).to(device)  # V1 max 2048
    
    with torch.no_grad():
        outputs = model(input_ids)
        # attentions: tuple of (batch, heads, seq, seq) per layer
        attentions = outputs.attentions  # 6 layers
    
    # Average attention across all layers and heads
    # Shape per layer: (1, 4, seq_len, seq_len)
    avg_attn = torch.zeros(input_ids.shape[1], input_ids.shape[1]).to(device)
    for layer_attn in attentions:
        avg_attn += layer_attn[0].mean(dim=0)  # average over heads
    avg_attn /= len(attentions)  # average over layers
    
    # Convert to gene-gene edges
    tokens = ds[cell_idx]['input_ids'][:2048]
    cell_edges = {}
    # Only compute for top-attended pairs (efficiency)
    attn_np = avg_attn.cpu().numpy()
    for i in range(min(len(tokens), 200)):  # top 200 genes per cell
        for j in range(min(len(tokens), 200)):
            if i != j:
                gene_i = int(tokens[i])
                gene_j = int(tokens[j])
                cell_edges[(gene_i, gene_j)] = float(attn_np[i, j])
    
    all_attention_edges.append(cell_edges)

print(f"  Extracted attention for {n_cells} cells")

# POOLED: average attention across all cells
print("\n=== POOLED approach ===")
pooled_edges = defaultdict(list)
for cell_edges in all_attention_edges:
    for edge, score in cell_edges.items():
        pooled_edges[edge].append(score)

pooled_scores = {edge: np.mean(scores) for edge, scores in pooled_edges.items()}
print(f"  Total edges: {len(pooled_scores)}")

# CSSI: stratify by cell type
print("\n=== CSSI approach ===")
unique_types = np.unique(cell_types[:n_cells])
print(f"  Cell types: {list(unique_types)}")

cssi_max_scores = defaultdict(float)
cssi_mean_scores = defaultdict(float)

type_counts = {}
for ct in unique_types:
    ct_mask = cell_types[:n_cells] == ct
    ct_indices = np.where(ct_mask)[0]
    type_counts[ct] = len(ct_indices)
    
    # Average attention within this cell type
    ct_edges = defaultdict(list)
    for idx in ct_indices:
        for edge, score in all_attention_edges[idx].items():
            ct_edges[edge].append(score)
    
    ct_mean = {edge: np.mean(scores) for edge, scores in ct_edges.items()}
    
    # CSSI-max: take max across cell types
    for edge, score in ct_mean.items():
        cssi_max_scores[edge] = max(cssi_max_scores[edge], score)
    
    # CSSI-mean: weighted average
    weight = len(ct_indices) / n_cells
    for edge, score in ct_mean.items():
        cssi_mean_scores[edge] += weight * score

print(f"  CSSI-max edges: {len(cssi_max_scores)}")
print(f"  CSSI-mean edges: {len(cssi_mean_scores)}")

# Evaluate against TRRUST
print("\n=== Evaluation against TRRUST ===")

def evaluate_against_trrust(edge_scores, label=""):
    """Compute AUROC of edge scores against TRRUST ground truth"""
    y_true = []
    y_score = []
    
    for (gene_i_tok, gene_j_tok), score in edge_scores.items():
        # Convert token IDs to gene names
        gene_i_ens = reverse_dict.get(gene_i_tok, "")
        gene_j_ens = reverse_dict.get(gene_j_tok, "")
        gene_i_name = gene_id_to_name.get(gene_i_ens, "")
        gene_j_name = gene_id_to_name.get(gene_j_ens, "")
        
        if gene_i_name and gene_j_name:
            is_true = 1 if (gene_i_name, gene_j_name) in trrust_pairs else 0
            y_true.append(is_true)
            y_score.append(score)
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    
    if n_pos > 0 and n_neg > 0:
        auroc = roc_auc_score(y_true, y_score)
    else:
        auroc = float('nan')
    
    print(f"  {label}: AUROC={auroc:.4f}, n_pos={n_pos}, n_neg={n_neg}, total={len(y_true)}")
    return auroc, n_pos

auroc_pooled, n_pos = evaluate_against_trrust(pooled_scores, "POOLED")
auroc_cssi_max, _ = evaluate_against_trrust(cssi_max_scores, "CSSI-max")
auroc_cssi_mean, _ = evaluate_against_trrust(cssi_mean_scores, "CSSI-mean")

# Summary
print("\n=== SUMMARY ===")
print(f"POOLED AUROC:    {auroc_pooled:.4f}")
print(f"CSSI-max AUROC:  {auroc_cssi_max:.4f}")
print(f"CSSI-mean AUROC: {auroc_cssi_mean:.4f}")
if not np.isnan(auroc_pooled) and auroc_pooled > 0:
    print(f"CSSI-max improvement: {auroc_cssi_max/auroc_pooled:.2f}x")
    print(f"CSSI-mean improvement: {auroc_cssi_mean/auroc_pooled:.2f}x")

# Save results
results = {
    "pooled_auroc": float(auroc_pooled),
    "cssi_max_auroc": float(auroc_cssi_max),
    "cssi_mean_auroc": float(auroc_cssi_mean),
    "n_cells": n_cells,
    "n_cell_types": len(unique_types),
    "cell_type_counts": {str(k): int(v) for k, v in type_counts.items()},
    "n_trrust_positives": int(n_pos),
    "n_edges_evaluated": len(pooled_scores),
}
with open(os.path.join(output_dir, "cssi_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_dir}/cssi_results.json")
print("=== DONE ===")
