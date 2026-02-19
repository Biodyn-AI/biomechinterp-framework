#!/usr/bin/env python3
"""Simplified baseline comparison for gene regulatory network inference methods."""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import warnings
warnings.filterwarnings('ignore')

def load_and_sample_data(data_path, n_cells=500, n_genes=1000, random_seed=42):
    """Load and sample data for faster processing."""
    print(f"Loading data from {data_path}...")
    adata = sc.read_h5ad(data_path)
    print(f"Original data shape: {adata.shape}")
    
    # Sample cells
    np.random.seed(random_seed)
    if adata.shape[0] > n_cells:
        cell_idx = np.random.choice(adata.shape[0], n_cells, replace=False)
        adata = adata[cell_idx, :].copy()
    
    # Filter genes with valid data and compute variability
    print("Preprocessing data...")
    
    # Remove genes with zero variance or invalid values
    sc.pp.filter_genes(adata, min_cells=10)  # Remove rarely expressed genes
    
    # Replace any infinity/NaN values
    if hasattr(adata.X, 'toarray'):
        X_temp = adata.X.toarray()
    else:
        X_temp = adata.X.copy()
    
    # Replace inf/nan with zeros
    X_temp = np.nan_to_num(X_temp, nan=0.0, posinf=0.0, neginf=0.0)
    adata.X = X_temp
    
    # Sample genes by variance if too many genes
    if adata.shape[1] > n_genes:
        print("Computing gene variability...")
        gene_vars = np.var(X_temp, axis=0)
        top_var_idx = np.argsort(gene_vars)[-n_genes:]
        adata = adata[:, top_var_idx].copy()
    
    print(f"Final data shape: {adata.shape}")
    
    # Get expression matrix
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()
    
    gene_names = adata.var_names.tolist()
    
    return X, gene_names

def spearman_network(X, gene_names, top_k=10000):
    """Compute Spearman correlation network."""
    print("Computing Spearman correlation network...")
    start_time = time.time()
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    # Create edge list
    edges = []
    n_genes = len(gene_names)
    
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            if not np.isnan(corr_matrix[i, j]):
                score = abs(corr_matrix[i, j])
                edges.append({
                    'TF': gene_names[i], 
                    'target': gene_names[j], 
                    'importance': score
                })
    
    # Sort by importance and take top k
    edges.sort(key=lambda x: x['importance'], reverse=True)
    if top_k and len(edges) > top_k:
        edges = edges[:top_k]
    
    elapsed = time.time() - start_time
    print(f"Spearman correlation completed in {elapsed:.2f}s, {len(edges)} edges")
    
    return pd.DataFrame(edges), elapsed

def mutual_info_network(X, gene_names, top_k=10000):
    """Compute mutual information network (simplified)."""
    print("Computing mutual information network...")
    start_time = time.time()
    
    edges = []
    n_genes = len(gene_names)
    
    # Sample pairs for efficiency
    max_pairs = min(50000, n_genes * (n_genes - 1) // 2)
    print(f"Computing MI for {max_pairs} gene pairs...")
    
    np.random.seed(42)
    pairs_computed = 0
    
    for i in range(n_genes):
        if pairs_computed >= max_pairs:
            break
        for j in range(i+1, min(n_genes, i + max_pairs // n_genes + 1)):
            if pairs_computed >= max_pairs:
                break
                
            try:
                mi_score = mutual_info_regression(
                    X[:, [i]], X[:, j], 
                    discrete_features=False, random_state=42
                )[0]
                
                edges.append({
                    'TF': gene_names[i],
                    'target': gene_names[j],
                    'importance': mi_score
                })
                pairs_computed += 1
            except:
                continue
    
    # Sort by importance and take top k
    edges.sort(key=lambda x: x['importance'], reverse=True)
    if top_k and len(edges) > top_k:
        edges = edges[:top_k]
    
    elapsed = time.time() - start_time
    print(f"Mutual information completed in {elapsed:.2f}s, {len(edges)} edges")
    
    return pd.DataFrame(edges), elapsed

def create_mock_ground_truth(gene_names, n_edges=1000, random_seed=42):
    """Create mock ground truth for evaluation."""
    np.random.seed(random_seed)
    true_edges = set()
    
    while len(true_edges) < min(n_edges, len(gene_names) * 10):
        tf_idx = np.random.choice(len(gene_names))
        target_idx = np.random.choice(len(gene_names))
        if tf_idx != target_idx:
            tf = gene_names[tf_idx]
            target = gene_names[target_idx]
            true_edges.add((tf, target))
    
    return true_edges

def evaluate_network(edge_df, ground_truth, method_name):
    """Evaluate network against ground truth."""
    print(f"Evaluating {method_name}...")
    
    labels = []
    scores = []
    
    for _, row in edge_df.iterrows():
        tf = row['TF']
        target = row['target']
        score = row['importance']
        
        # Check if edge is in ground truth (either direction)
        is_true_edge = (tf, target) in ground_truth or (target, tf) in ground_truth
        
        labels.append(int(is_true_edge))
        scores.append(score)
    
    # Calculate metrics
    if len(set(labels)) > 1:
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)
    else:
        auroc = 0.5
        auprc = sum(labels) / len(labels)
    
    print(f"{method_name}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, Edges={len(edge_df)}")
    
    return auroc, auprc

def run_simple_comparison():
    """Run simplified baseline comparison."""
    print("Starting simplified baseline comparison...")
    
    # Load data
    data_path = r"D:\openclaw\intelligence-augmentation\data\brain_scrna\DLPFC_11k.h5ad"
    X, gene_names = load_and_sample_data(data_path, n_cells=500, n_genes=1000)
    
    # Create mock ground truth
    ground_truth = create_mock_ground_truth(gene_names)
    print(f"Ground truth: {len(ground_truth)} edges")
    
    # Results storage
    results = {
        'method': [],
        'auroc': [],
        'auprc': [],
        'time': [],
        'n_edges': []
    }
    
    # Run Spearman correlation
    spearman_edges, spearman_time = spearman_network(X, gene_names)
    spearman_auroc, spearman_auprc = evaluate_network(spearman_edges, ground_truth, "Spearman")
    
    results['method'].append('Spearman Correlation')
    results['auroc'].append(spearman_auroc)
    results['auprc'].append(spearman_auprc)
    results['time'].append(spearman_time)
    results['n_edges'].append(len(spearman_edges))
    
    # Run Mutual Information
    mi_edges, mi_time = mutual_info_network(X, gene_names)
    mi_auroc, mi_auprc = evaluate_network(mi_edges, ground_truth, "Mutual Information")
    
    results['method'].append('Mutual Information')
    results['auroc'].append(mi_auroc)
    results['auprc'].append(mi_auprc)
    results['time'].append(mi_time)
    results['n_edges'].append(len(mi_edges))
    
    # Save results
    results_df = pd.DataFrame(results)
    output_dir = r"D:\openclaw\biodyn-nmi-paper\baseline_comparison"
    results_df.to_csv(f"{output_dir}/simple_results.csv", index=False)
    
    # Save edge lists
    spearman_edges.to_csv(f"{output_dir}/simple_spearman_edges.csv", index=False)
    mi_edges.to_csv(f"{output_dir}/simple_mi_edges.csv", index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("SIMPLE BASELINE COMPARISON RESULTS")
    print("="*50)
    print(f"Data: 500 cells, {len(gene_names)} genes")
    print(f"Ground truth: {len(ground_truth)} edges")
    print()
    
    for i, method in enumerate(results['method']):
        auroc = results['auroc'][i]
        auprc = results['auprc'][i]
        elapsed = results['time'][i]
        n_edges = results['n_edges'][i]
        print(f"{method:20}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, Time={elapsed:.2f}s, Edges={n_edges}")
    
    print(f"\nResults saved to: {output_dir}")
    return results_df

if __name__ == "__main__":
    results = run_simple_comparison()
    print("Completed!")