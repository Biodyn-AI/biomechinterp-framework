#!/usr/bin/env python3
"""
Controlled-Composition Scaling Experiment
==========================================
Addresses Major Concern 2 / Stress Test 1 from peer review:
Disentangle sample size (N) from heterogeneity in scaling behavior.

Three experimental conditions:
(i)   SINGLE cell type, varying N → tests pure sample-size effect
(ii)  MIXED cell types with FIXED composition across N → tests scaling with controlled heterogeneity
(iii) MIXED cell types with INCREASING heterogeneity → tests heterogeneity-driven degradation

Uses correlation-based edge scores as proxy for attention-derived scores
(validated as equivalent in baseline comparison, Section 3.2).
Evaluates against TRRUST reference edges.
"""

import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data paths
DATA_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/data/raw"
REF_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks"
OUT_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/controlled_composition_results"
FIG_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/figures"

os.makedirs(OUT_DIR, exist_ok=True)

def load_trrust():
    """Load TRRUST v2 reference edges."""
    trrust_path = os.path.join(REF_DIR, "trrust_human.tsv")
    df = pd.read_csv(trrust_path, sep='\t', header=None,
                     names=['tf', 'target', 'mode', 'pmid'])
    edges = set(zip(df['tf'].str.upper(), df['target'].str.upper()))
    return edges

def compute_edge_scores(adata, gene_names):
    """Compute Spearman correlation-based edge scores."""
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    n_genes = X.shape[1]
    # Rank-transform for Spearman
    from scipy.stats import rankdata
    ranks = np.apply_along_axis(rankdata, 0, X)
    # Correlation matrix
    corr = np.corrcoef(ranks.T)
    # Replace NaN with 0 (genes with zero variance)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr, gene_names

def evaluate_auroc(corr_matrix, gene_names, trrust_edges):
    """Evaluate AUROC of edge scores against TRRUST."""
    gene_upper = [g.upper() for g in gene_names]
    gene_to_idx = {g: i for i, g in enumerate(gene_upper)}

    scores = []
    labels = []
    for i, g1 in enumerate(gene_upper):
        for j, g2 in enumerate(gene_upper):
            if i == j:
                continue
            score = abs(corr_matrix[i, j])
            label = 1 if (g1, g2) in trrust_edges else 0
            scores.append(score)
            labels.append(label)

    scores = np.array(scores)
    labels = np.array(labels)

    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5, 0, len(labels)

    auroc = roc_auc_score(labels, scores)
    return auroc, int(labels.sum()), len(labels)

def compute_heterogeneity(adata, ct_col='cell_type'):
    """Compute Shannon entropy of cell-type distribution."""
    if ct_col not in adata.obs.columns:
        for col in adata.obs.columns:
            if 'cell' in col.lower() and 'type' in col.lower():
                ct_col = col
                break
    if ct_col not in adata.obs.columns:
        return 0.0, 1

    counts = adata.obs[ct_col].value_counts()
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    n_types = len(counts)
    return entropy, n_types

def run_experiment():
    """Run the three-condition controlled-composition scaling experiment."""
    import scanpy as sc
    import anndata

    print("Loading Tabula Sapiens kidney data...")
    adata = sc.read_h5ad(os.path.join(DATA_DIR, "tabula_sapiens_kidney.h5ad"))
    print(f"  Total cells: {adata.n_obs}, genes: {adata.n_vars}")

    # Find cell type column — prefer readable labels over ontology IDs
    ct_col = None
    for col in ['cell_type', 'free_annotation', 'broad_cell_class', 'celltype']:
        if col in adata.obs.columns:
            ct_col = col
            break
    if ct_col is None:
        for col in adata.obs.columns:
            if 'cell' in col.lower() and 'type' in col.lower() and 'ontology' not in col.lower():
                ct_col = col
                break
    if ct_col is None:
        ct_col = adata.obs.columns[0]
        print(f"  WARNING: Using first obs column '{ct_col}' as cell type")
    else:
        print(f"  Using cell type column: '{ct_col}'")

    cell_types = adata.obs[ct_col].value_counts()
    print(f"  Cell types: {len(cell_types)}")
    print(f"  Top 5: {dict(cell_types.head())}")

    # Map Ensembl IDs to gene symbols via feature_name
    if 'feature_name' in adata.var.columns:
        names = adata.var['feature_name'].astype(str).values.copy()
        adata.var.index = pd.Index(names)
        # Make unique manually
        from collections import Counter
        counts = Counter()
        new_names = []
        for n in names:
            if counts[n] > 0:
                new_names.append(f"{n}-{counts[n]}")
            else:
                new_names.append(n)
            counts[n] += 1
        adata.var.index = pd.Index(new_names)
        print(f"  Mapped to gene symbols: {list(adata.var_names[:5])}")

    # Select HVGs for tractable computation
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=500)
    gene_mask = adata.var['highly_variable']
    gene_names = list(adata.var_names[gene_mask])
    adata_hvg = adata[:, gene_mask].copy()

    trrust_edges = load_trrust()
    print(f"  TRRUST edges loaded: {len(trrust_edges)}")

    # Count how many TRRUST edges are in our gene universe
    gene_upper = set(g.upper() for g in gene_names)
    n_evaluable = sum(1 for (tf, tgt) in trrust_edges
                      if tf in gene_upper and tgt in gene_upper)
    print(f"  Evaluable TRRUST edges in gene universe: {n_evaluable}")

    results = []
    N_values = [100, 200, 500, 1000, 2000, 3000]
    n_seeds = 5

    # ========================================
    # CONDITION 1: Single cell type, varying N
    # ========================================
    print("\n=== CONDITION 1: Single cell type, varying N ===")
    # Find largest cell type
    largest_ct = cell_types.index[0]
    ct_mask = adata.obs[ct_col] == largest_ct
    adata_single = adata_hvg[ct_mask].copy()
    print(f"  Largest cell type: '{largest_ct}' with {adata_single.n_obs} cells")

    for N in N_values:
        if N > adata_single.n_obs:
            print(f"  Skipping N={N} (only {adata_single.n_obs} cells of type '{largest_ct}')")
            continue
        for seed in range(n_seeds):
            np.random.seed(seed * 42 + N)
            idx = np.random.choice(adata_single.n_obs, N, replace=False)
            sub = adata_single[idx]
            corr, gnames = compute_edge_scores(sub, gene_names)
            auroc, n_pos, n_total = evaluate_auroc(corr, gnames, trrust_edges)
            entropy, n_types = compute_heterogeneity(sub, ct_col)
            results.append({
                'condition': 'single_cell_type',
                'N': N,
                'seed': seed,
                'auroc': auroc,
                'n_positive_edges': n_pos,
                'n_total_edges': n_total,
                'entropy': entropy,
                'n_cell_types': n_types,
                'cell_type': largest_ct
            })
            print(f"    N={N}, seed={seed}: AUROC={auroc:.4f}, pos={n_pos}")

    # ========================================
    # CONDITION 2: Fixed composition, varying N
    # ========================================
    print("\n=== CONDITION 2: Fixed composition, varying N ===")
    # Use top cell types with enough cells, fix proportions
    min_cells_per_type = 50
    eligible_types = cell_types[cell_types >= min_cells_per_type].index.tolist()
    n_types_to_use = min(5, len(eligible_types))
    selected_types = eligible_types[:n_types_to_use]
    print(f"  Selected types ({n_types_to_use}): {selected_types}")

    # Fixed proportions (equal)
    proportions = {ct: 1.0/n_types_to_use for ct in selected_types}

    for N in N_values:
        cells_per_type = int(N / n_types_to_use)
        if cells_per_type < 10:
            continue
        # Check feasibility
        feasible = all(cell_types[ct] >= cells_per_type for ct in selected_types)
        if not feasible:
            print(f"  Skipping N={N} (insufficient cells for equal split)")
            continue

        for seed in range(n_seeds):
            np.random.seed(seed * 42 + N + 1000)
            indices = []
            for ct in selected_types:
                ct_indices = np.where(adata.obs[ct_col] == ct)[0]
                chosen = np.random.choice(ct_indices, cells_per_type, replace=False)
                indices.extend(chosen)
            indices = np.array(indices)
            sub = adata_hvg[indices]
            corr, gnames = compute_edge_scores(sub, gene_names)
            auroc, n_pos, n_total = evaluate_auroc(corr, gnames, trrust_edges)
            entropy, n_types = compute_heterogeneity(sub, ct_col)
            results.append({
                'condition': 'fixed_composition',
                'N': N,
                'seed': seed,
                'auroc': auroc,
                'n_positive_edges': n_pos,
                'n_total_edges': n_total,
                'entropy': entropy,
                'n_cell_types': n_types,
                'cell_type': 'mixed_fixed'
            })
            print(f"    N={N}, seed={seed}: AUROC={auroc:.4f}, entropy={entropy:.3f}")

    # ========================================
    # CONDITION 3: Increasing heterogeneity
    # ========================================
    print("\n=== CONDITION 3: Increasing heterogeneity (fixed N=500) ===")
    fixed_N = 500
    # Vary number of cell types included (1, 2, 3, ..., max)
    for n_types_included in range(1, min(len(eligible_types), 8) + 1):
        types_subset = eligible_types[:n_types_included]
        cells_per_type = fixed_N // n_types_included
        if cells_per_type < 10:
            continue
        feasible = all(cell_types[ct] >= cells_per_type for ct in types_subset)
        if not feasible:
            continue

        for seed in range(n_seeds):
            np.random.seed(seed * 42 + n_types_included + 2000)
            indices = []
            for ct in types_subset:
                ct_indices = np.where(adata.obs[ct_col] == ct)[0]
                chosen = np.random.choice(ct_indices, cells_per_type, replace=False)
                indices.extend(chosen)
            indices = np.array(indices)
            sub = adata_hvg[indices]
            corr, gnames = compute_edge_scores(sub, gene_names)
            auroc, n_pos, n_total = evaluate_auroc(corr, gnames, trrust_edges)
            entropy, n_types_actual = compute_heterogeneity(sub, ct_col)
            results.append({
                'condition': 'increasing_heterogeneity',
                'N': len(indices),
                'seed': seed,
                'auroc': auroc,
                'n_positive_edges': n_pos,
                'n_total_edges': n_total,
                'entropy': entropy,
                'n_cell_types': n_types_actual,
                'cell_type': f'mixed_{n_types_included}_types'
            })
            print(f"    types={n_types_included}, seed={seed}: AUROC={auroc:.4f}, entropy={entropy:.3f}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "controlled_composition_results.csv"), index=False)
    print(f"\nSaved {len(df)} results to {OUT_DIR}/controlled_composition_results.csv")

    # ========================================
    # STATISTICAL ANALYSIS
    # ========================================
    print("\n=== STATISTICAL ANALYSIS ===")

    # Test 1: Does AUROC change with N for single cell type?
    cond1 = df[df['condition'] == 'single_cell_type']
    if len(cond1) > 0:
        rho, pval = stats.spearmanr(cond1['N'], cond1['auroc'])
        print(f"Single cell type: Spearman(N, AUROC) = {rho:.3f}, p = {pval:.4f}")
        print(f"  Mean AUROC by N: {dict(cond1.groupby('N')['auroc'].mean().round(4))}")

    # Test 2: Does AUROC change with N for fixed composition?
    cond2 = df[df['condition'] == 'fixed_composition']
    if len(cond2) > 0:
        rho, pval = stats.spearmanr(cond2['N'], cond2['auroc'])
        print(f"Fixed composition: Spearman(N, AUROC) = {rho:.3f}, p = {pval:.4f}")
        print(f"  Mean AUROC by N: {dict(cond2.groupby('N')['auroc'].mean().round(4))}")

    # Test 3: Does AUROC decrease with heterogeneity at fixed N?
    cond3 = df[df['condition'] == 'increasing_heterogeneity']
    if len(cond3) > 0:
        rho, pval = stats.spearmanr(cond3['entropy'], cond3['auroc'])
        print(f"Increasing heterogeneity: Spearman(entropy, AUROC) = {rho:.3f}, p = {pval:.4f}")
        print(f"  Mean AUROC by n_types: {dict(cond3.groupby('n_cell_types')['auroc'].mean().round(4))}")

    # ========================================
    # GENERATE FIGURE
    # ========================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Single cell type
    if len(cond1) > 0:
        ax = axes[0]
        means = cond1.groupby('N')['auroc'].agg(['mean', 'std']).reset_index()
        ax.errorbar(means['N'], means['mean'], yerr=1.96*means['std']/np.sqrt(n_seeds),
                    fmt='o-', color='#2196F3', capsize=5, linewidth=2, markersize=8)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
        rho_val = stats.spearmanr(cond1['N'], cond1['auroc'])[0]
        ax.set_xlabel('Number of cells (N)')
        ax.set_ylabel('AUROC vs TRRUST')
        ax.set_title(f'(A) Single cell type\n(heterogeneity fixed)\n$\\rho$ = {rho_val:.3f}')
        ax.legend()

    # Panel B: Fixed composition
    if len(cond2) > 0:
        ax = axes[1]
        means = cond2.groupby('N')['auroc'].agg(['mean', 'std']).reset_index()
        ax.errorbar(means['N'], means['mean'], yerr=1.96*means['std']/np.sqrt(n_seeds),
                    fmt='s-', color='#4CAF50', capsize=5, linewidth=2, markersize=8)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
        rho_val = stats.spearmanr(cond2['N'], cond2['auroc'])[0]
        ax.set_xlabel('Number of cells (N)')
        ax.set_ylabel('AUROC vs TRRUST')
        ax.set_title(f'(B) Fixed composition\n(heterogeneity controlled)\n$\\rho$ = {rho_val:.3f}')
        ax.legend()

    # Panel C: Increasing heterogeneity
    if len(cond3) > 0:
        ax = axes[2]
        means = cond3.groupby('n_cell_types')['auroc'].agg(['mean', 'std']).reset_index()
        ax.errorbar(means['n_cell_types'], means['mean'], yerr=1.96*means['std']/np.sqrt(n_seeds),
                    fmt='D-', color='#F44336', capsize=5, linewidth=2, markersize=8)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
        rho_val = stats.spearmanr(cond3['n_cell_types'], cond3['auroc'])[0]
        ax.set_xlabel('Number of cell types (heterogeneity)')
        ax.set_ylabel('AUROC vs TRRUST')
        ax.set_title(f'(C) Fixed N, increasing heterogeneity\n$\\rho$ = {rho_val:.3f}')
        ax.legend()

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_controlled_composition_scaling.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to {fig_path}")

    # Summary JSON
    summary = {
        'experiment': 'controlled_composition_scaling',
        'dataset': 'tabula_sapiens_kidney',
        'n_hvgs': 500,
        'n_seeds': n_seeds,
        'conditions': {}
    }
    for cond_name in ['single_cell_type', 'fixed_composition', 'increasing_heterogeneity']:
        cond_df = df[df['condition'] == cond_name]
        if len(cond_df) > 0:
            if cond_name == 'increasing_heterogeneity':
                rho, pval = stats.spearmanr(cond_df['entropy'], cond_df['auroc'])
            else:
                rho, pval = stats.spearmanr(cond_df['N'], cond_df['auroc'])
            summary['conditions'][cond_name] = {
                'n_runs': len(cond_df),
                'spearman_rho': round(float(rho), 4),
                'spearman_p': float(pval),
                'mean_auroc': round(float(cond_df['auroc'].mean()), 4),
                'std_auroc': round(float(cond_df['auroc'].std()), 4),
            }

    with open(os.path.join(OUT_DIR, "controlled_composition_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXPERIMENT COMPLETE ===")
    return df, summary

if __name__ == '__main__':
    df, summary = run_experiment()
    print(json.dumps(summary, indent=2))
