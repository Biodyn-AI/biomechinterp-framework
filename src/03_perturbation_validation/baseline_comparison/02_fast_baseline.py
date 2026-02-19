#!/usr/bin/env python3
"""
Fast baseline comparison - no scanpy dependency for faster startup.
Uses only essential libraries for gene regulatory network inference comparison.
"""

import numpy as np
import pandas as pd
import h5py
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

def load_h5ad_basic(file_path, n_cells=500, n_genes=500, random_seed=42):
    """Load h5ad file using basic h5py without scanpy."""
    print(f"Loading data from {file_path}...")
    
    with h5py.File(file_path, 'r') as f:
        print("Available keys:", list(f.keys()))
        
        # Read expression matrix - h5ad format typically stores this differently
        X = None
        if 'X' in f:
            X_group = f['X']
            print("X group keys:", list(X_group.keys()) if hasattr(X_group, 'keys') else 'X is dataset')
            
            if 'data' in X_group:
                # Sparse matrix format
                data = X_group['data'][:]
                indices = X_group['indices'][:]
                indptr = X_group['indptr'][:]
                shape = X_group.attrs.get('shape', None)
                
                if shape is not None:
                    from scipy.sparse import csr_matrix
                    X_sparse = csr_matrix((data, indices, indptr), shape=shape)
                    X = X_sparse.toarray()
                else:
                    raise ValueError("Could not determine matrix shape")
            else:
                # Dense matrix
                X = X_group[:]
        
        if X is None:
            raise ValueError("Could not find expression matrix 'X' in file")
        
        # Read gene names
        gene_names = []
        if 'var' in f:
            var_group = f['var']
            if '_index' in var_group:
                gene_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                             for name in var_group['_index'][:]]
            elif 'index' in var_group:
                gene_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                             for name in var_group['index'][:]]
        
        if not gene_names:
            gene_names = [f"Gene_{i}" for i in range(X.shape[1])]
        
        print(f"Original data shape: {X.shape}")
        
        # Handle sparse matrices
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Sample cells
        np.random.seed(random_seed)
        if X.shape[0] > n_cells:
            cell_idx = np.random.choice(X.shape[0], n_cells, replace=False)
            X = X[cell_idx, :]
        
        print(f"After cell sampling: {X.shape}")
        
        # Clean data
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate gene statistics
        gene_means = np.mean(X, axis=0)
        gene_vars = np.var(X, axis=0)
        gene_nonzero = np.sum(X > 0, axis=0)
        
        # Filter genes: expressed in at least 10 cells and have variance > 0
        valid_genes = (gene_nonzero >= 10) & (gene_vars > 0)
        
        print(f"Valid genes: {np.sum(valid_genes)}/{len(valid_genes)}")
        
        X = X[:, valid_genes]
        gene_names = [gene_names[i] for i in range(len(gene_names)) if valid_genes[i]]
        
        # Select top variable genes
        if len(gene_names) > n_genes:
            gene_vars_filtered = gene_vars[valid_genes]
            top_var_idx = np.argsort(gene_vars_filtered)[-n_genes:]
            X = X[:, top_var_idx]
            gene_names = [gene_names[i] for i in top_var_idx]
        
        print(f"Final data shape: {X.shape}")
        
        return X, gene_names

def compute_spearman_correlation(X, gene_names):
    """Compute Spearman correlation network."""
    print("Computing Spearman correlation network...")
    start_time = time.time()
    
    n_genes = len(gene_names)
    edges = []
    
    # Compute pairwise correlations
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            try:
                corr, p_value = spearmanr(X[:, i], X[:, j])
                if not np.isnan(corr) and not np.isinf(corr):
                    edges.append({
                        'TF': gene_names[i],
                        'target': gene_names[j],
                        'importance': abs(corr),
                        'correlation': corr,
                        'p_value': p_value
                    })
            except:
                continue
    
    # Sort by absolute correlation
    edges.sort(key=lambda x: x['importance'], reverse=True)
    
    elapsed = time.time() - start_time
    print(f"Spearman correlation: {len(edges)} edges in {elapsed:.2f}s")
    
    return pd.DataFrame(edges), elapsed

def compute_mutual_information(X, gene_names, max_pairs=20000):
    """Compute mutual information network."""
    print(f"Computing mutual information network (max {max_pairs} pairs)...")
    start_time = time.time()
    
    n_genes = len(gene_names)
    edges = []
    pairs_computed = 0
    
    # Sample pairs for efficiency
    np.random.seed(42)
    all_pairs = [(i, j) for i in range(n_genes) for j in range(i+1, n_genes)]
    if len(all_pairs) > max_pairs:
        selected_pairs = np.random.choice(len(all_pairs), max_pairs, replace=False)
        pairs_to_compute = [all_pairs[idx] for idx in selected_pairs]
    else:
        pairs_to_compute = all_pairs
    
    for pair_idx, (i, j) in enumerate(pairs_to_compute):
        try:
            # Add small noise to break ties
            x_vals = X[:, i] + 1e-6 * np.random.randn(X.shape[0])
            y_vals = X[:, j] + 1e-6 * np.random.randn(X.shape[0])
            
            mi_score = mutual_info_regression(
                x_vals.reshape(-1, 1), y_vals,
                discrete_features=False, random_state=42
            )[0]
            
            edges.append({
                'TF': gene_names[i],
                'target': gene_names[j],
                'importance': mi_score
            })
            
            pairs_computed += 1
            if pairs_computed % 1000 == 0:
                print(f"  Progress: {pairs_computed}/{len(pairs_to_compute)} pairs")
                
        except Exception as e:
            continue
    
    # Sort by MI score
    edges.sort(key=lambda x: x['importance'], reverse=True)
    
    elapsed = time.time() - start_time
    print(f"Mutual information: {len(edges)} edges in {elapsed:.2f}s")
    
    return pd.DataFrame(edges), elapsed

def create_ground_truth(gene_names, random_seed=42):
    """Create synthetic ground truth network."""
    print("Creating synthetic ground truth network...")
    
    np.random.seed(random_seed)
    true_edges = set()
    
    # Strategy: Create structured ground truth
    # 1. Local clusters (genes close in list are connected)
    # 2. Hub genes (some genes connect to many others)
    # 3. Random connections
    
    n_genes = len(gene_names)
    
    # Local clusters
    cluster_size = 5
    for start in range(0, n_genes, cluster_size):
        cluster_genes = gene_names[start:start+cluster_size]
        for i, gene1 in enumerate(cluster_genes):
            for j, gene2 in enumerate(cluster_genes[i+1:], i+1):
                if np.random.random() < 0.3:  # 30% chance within cluster
                    true_edges.add((gene1, gene2))
    
    # Hub genes
    n_hubs = min(10, n_genes // 20)
    hub_genes = np.random.choice(gene_names, n_hubs, replace=False)
    for hub in hub_genes:
        n_connections = np.random.randint(5, 15)
        targets = np.random.choice(gene_names, n_connections, replace=False)
        for target in targets:
            if hub != target:
                true_edges.add((hub, target))
    
    # Random connections
    n_random = min(500, n_genes * 2)
    for _ in range(n_random):
        gene1 = np.random.choice(gene_names)
        gene2 = np.random.choice(gene_names)
        if gene1 != gene2:
            true_edges.add((gene1, gene2))
    
    print(f"Ground truth network: {len(true_edges)} edges")
    return true_edges

def evaluate_network(edge_df, ground_truth, method_name, top_k=10000):
    """Evaluate network performance."""
    print(f"Evaluating {method_name} network...")
    
    # Use top k edges
    eval_edges = edge_df.head(top_k) if len(edge_df) > top_k else edge_df
    
    labels = []
    scores = []
    
    for _, row in eval_edges.iterrows():
        tf = row['TF']
        target = row['target']
        score = row['importance']
        
        # Check if edge exists in ground truth (either direction)
        is_true = (tf, target) in ground_truth or (target, tf) in ground_truth
        
        labels.append(int(is_true))
        scores.append(score)
    
    # Calculate metrics
    if len(set(labels)) > 1:
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)
    else:
        auroc = 0.5
        auprc = np.mean(labels) if labels else 0.0
    
    n_true_found = sum(labels)
    total_evaluated = len(labels)
    precision_at_k = n_true_found / total_evaluated if total_evaluated > 0 else 0.0
    
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")
    print(f"  Precision@{top_k}: {precision_at_k:.4f}")
    print(f"  True edges found: {n_true_found}/{total_evaluated}")
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'precision_at_k': precision_at_k,
        'true_edges_found': n_true_found,
        'total_evaluated': total_evaluated
    }

def save_results_and_plots(results, output_dir):
    """Save results and create comparison plots."""
    from pathlib import Path
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save results as CSV
    pd.DataFrame(results).to_csv(output_path / "baseline_comparison_results.csv", index=False)
    
    # Save as JSON
    with open(output_path / "baseline_comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create comparison plot
    methods = results['method']
    aurocs = results['auroc']
    auprcs = results['auprc']
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # AUROC comparison
    bars1 = ax1.bar(methods, aurocs, color=['skyblue', 'lightcoral'], alpha=0.8)
    ax1.set_ylabel('AUROC')
    ax1.set_title('AUROC Comparison')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(aurocs):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # AUPRC comparison
    bars2 = ax2.bar(methods, auprcs, color=['lightgreen', 'orange'], alpha=0.8)
    ax2.set_ylabel('AUPRC')
    ax2.set_title('AUPRC Comparison')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(auprcs):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Precision@K comparison
    precisions = results['precision_at_k']
    bars3 = ax3.bar(methods, precisions, color=['gold', 'mediumpurple'], alpha=0.8)
    ax3.set_ylabel('Precision@10k')
    ax3.set_title('Precision@10k Comparison')
    ax3.set_ylim(0, max(precisions) * 1.2)
    ax3.tick_params(axis='x', rotation=45)
    for i, v in enumerate(precisions):
        ax3.text(i, v + max(precisions)*0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / "baseline_comparison_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved to: {output_path}")

def run_fast_baseline():
    """Run fast baseline comparison without scanpy."""
    print("="*70)
    print("FAST BASELINE COMPARISON - GENE REGULATORY NETWORK INFERENCE")
    print("="*70)
    
    # Configuration
    data_path = r"D:\openclaw\intelligence-augmentation\data\brain_scrna\DLPFC_11k.h5ad"
    output_dir = r"D:\openclaw\biodyn-nmi-paper\baseline_comparison"
    n_cells = 500
    n_genes = 500
    random_seed = 42
    
    try:
        # Load data
        X, gene_names = load_h5ad_basic(data_path, n_cells, n_genes, random_seed)
        
        # Create ground truth
        ground_truth = create_ground_truth(gene_names, random_seed)
        
        results = {
            'method': [],
            'auroc': [],
            'auprc': [],
            'precision_at_k': [],
            'computation_time': [],
            'n_edges_total': [],
            'true_edges_found': [],
            'total_evaluated': []
        }
        
        # Method 1: Spearman Correlation
        print("\n" + "-"*50)
        spearman_df, spearman_time = compute_spearman_correlation(X, gene_names)
        spearman_metrics = evaluate_network(spearman_df, ground_truth, "Spearman Correlation")
        
        results['method'].append('Spearman Correlation')
        results['auroc'].append(spearman_metrics['auroc'])
        results['auprc'].append(spearman_metrics['auprc'])
        results['precision_at_k'].append(spearman_metrics['precision_at_k'])
        results['computation_time'].append(spearman_time)
        results['n_edges_total'].append(len(spearman_df))
        results['true_edges_found'].append(spearman_metrics['true_edges_found'])
        results['total_evaluated'].append(spearman_metrics['total_evaluated'])
        
        # Save Spearman results
        spearman_df.to_csv(f"{output_dir}/spearman_network.csv", index=False)
        
        # Method 2: Mutual Information
        print("\n" + "-"*50)
        mi_df, mi_time = compute_mutual_information(X, gene_names)
        mi_metrics = evaluate_network(mi_df, ground_truth, "Mutual Information")
        
        results['method'].append('Mutual Information')
        results['auroc'].append(mi_metrics['auroc'])
        results['auprc'].append(mi_metrics['auprc'])
        results['precision_at_k'].append(mi_metrics['precision_at_k'])
        results['computation_time'].append(mi_time)
        results['n_edges_total'].append(len(mi_df))
        results['true_edges_found'].append(mi_metrics['true_edges_found'])
        results['total_evaluated'].append(mi_metrics['total_evaluated'])
        
        # Save MI results
        mi_df.to_csv(f"{output_dir}/mutual_info_network.csv", index=False)
        
        # Save results and create plots
        save_results_and_plots(results, output_dir)
        
        # Print final summary
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY")
        print("="*70)
        print(f"Dataset: DLPFC brain data ({n_cells} cells, {len(gene_names)} genes)")
        print(f"Ground truth: {len(ground_truth)} edges")
        print(f"Random seed: {random_seed}")
        print()
        
        for i, method in enumerate(results['method']):
            print(f"{method}:")
            print(f"  AUROC: {results['auroc'][i]:.4f}")
            print(f"  AUPRC: {results['auprc'][i]:.4f}")
            print(f"  Precision@10k: {results['precision_at_k'][i]:.4f}")
            print(f"  Computation time: {results['computation_time'][i]:.2f}s")
            print(f"  Total edges: {results['n_edges_total'][i]:,}")
            print(f"  True edges found: {results['true_edges_found'][i]:,}")
            print()
        
        print(f"All results and plots saved to: {output_dir}")
        
        # Key insight
        print("KEY FINDINGS:")
        spearman_auroc = results['auroc'][0]
        mi_auroc = results['auroc'][1] if len(results['auroc']) > 1 else 0
        
        if spearman_auroc < 0.6 and mi_auroc < 0.6:
            print("• Both methods show poor performance (~random), suggesting:")
            print("  - Brain tissue may not have strong regulatory signals")
            print("  - Current ground truth may not be appropriate for this tissue")
            print("  - The issue may be with benchmarking approach, not attention methods")
        elif abs(spearman_auroc - mi_auroc) > 0.1:
            print("• Methods show significantly different performance:")
            print(f"  - Spearman: {spearman_auroc:.3f}, MI: {mi_auroc:.3f}")
            print("  - This suggests different methods capture different signals")
        else:
            print("• Both methods perform similarly, suggesting shared limitations")
        
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_fast_baseline()
    if results:
        print("Baseline comparison completed successfully!")
    else:
        print("Baseline comparison failed.")