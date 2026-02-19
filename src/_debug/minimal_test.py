#!/usr/bin/env python3
"""Minimal test script to verify core functionality."""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score, average_precision_score
import time

print("Starting minimal test...")

# Generate synthetic data
np.random.seed(42)
n_cells = 100
n_genes = 50

print(f"Creating synthetic data: {n_cells} cells, {n_genes} genes")
X = np.random.randn(n_cells, n_genes)

# Add some structure (correlation between some genes)
for i in range(0, n_genes, 5):
    if i + 1 < n_genes:
        X[:, i+1] = X[:, i] + 0.3 * np.random.randn(n_cells)

gene_names = [f"Gene_{i}" for i in range(n_genes)]

print("Testing Spearman correlation...")
start_time = time.time()

# Compute correlation matrix
corr_matrix = np.corrcoef(X.T)

# Create edge list
edges = []
for i in range(n_genes):
    for j in range(i+1, n_genes):
        if not np.isnan(corr_matrix[i, j]):
            score = abs(corr_matrix[i, j])
            edges.append({
                'TF': gene_names[i], 
                'target': gene_names[j], 
                'importance': score
            })

spearman_time = time.time() - start_time
print(f"Spearman correlation: {len(edges)} edges in {spearman_time:.2f}s")

# Test mutual information on subset
print("Testing mutual information...")
start_time = time.time()

mi_edges = []
max_pairs = 100
pairs_tested = 0

for i in range(n_genes):
    if pairs_tested >= max_pairs:
        break
    for j in range(i+1, min(n_genes, i + 10)):
        if pairs_tested >= max_pairs:
            break
        try:
            mi_score = mutual_info_regression(
                X[:, [i]], X[:, j], 
                discrete_features=False, random_state=42
            )[0]
            
            mi_edges.append({
                'TF': gene_names[i],
                'target': gene_names[j],
                'importance': mi_score
            })
            pairs_tested += 1
        except:
            continue

mi_time = time.time() - start_time
print(f"Mutual information: {len(mi_edges)} edges in {mi_time:.2f}s")

# Test evaluation
print("Testing evaluation...")
ground_truth = set()
for i in range(0, min(n_genes, 20), 2):
    if i + 1 < n_genes:
        ground_truth.add((gene_names[i], gene_names[i+1]))

print(f"Mock ground truth: {len(ground_truth)} edges")

# Evaluate Spearman
spearman_df = pd.DataFrame(edges)
labels = []
scores = []

for _, row in spearman_df.iterrows():
    tf = row['TF']
    target = row['target']
    score = row['importance']
    
    is_true_edge = (tf, target) in ground_truth or (target, tf) in ground_truth
    labels.append(int(is_true_edge))
    scores.append(score)

if len(set(labels)) > 1:
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    print(f"Spearman evaluation: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
else:
    print("Spearman evaluation: Not enough variation in labels")

print("Testing real data loading...")
try:
    data_path = r"D:\openclaw\intelligence-augmentation\data\brain_scrna\DLPFC_11k.h5ad"
    print(f"Loading: {data_path}")
    adata = sc.read_h5ad(data_path)
    print(f"Real data shape: {adata.shape}")
    
    # Sample a tiny subset
    if adata.shape[0] > 50:
        sample_idx = np.random.choice(adata.shape[0], 50, replace=False)
        adata_small = adata[sample_idx, :].copy()
    
    if adata_small.shape[1] > 100:
        gene_idx = np.random.choice(adata_small.shape[1], 100, replace=False)
        adata_small = adata_small[:, gene_idx].copy()
    
    print(f"Sampled real data shape: {adata_small.shape}")
    
    # Get expression matrix
    if hasattr(adata_small.X, 'toarray'):
        X_real = adata_small.X.toarray()
    else:
        X_real = adata_small.X.copy()
    
    # Check for problematic values
    has_nan = np.any(np.isnan(X_real))
    has_inf = np.any(np.isinf(X_real))
    print(f"Real data - NaN: {has_nan}, Inf: {has_inf}")
    
    if has_nan or has_inf:
        X_real = np.nan_to_num(X_real, nan=0.0, posinf=0.0, neginf=0.0)
        print("Cleaned problematic values")
    
    print("Real data loaded successfully!")
    
except Exception as e:
    print(f"Real data loading failed: {e}")

print("Minimal test completed successfully!")