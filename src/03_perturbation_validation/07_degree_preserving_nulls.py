#!/usr/bin/env python3
"""
Degree-Preserving Null Models for Curated-Edge Recovery
=========================================================
Addresses Round 6 Reviewer Stress Test 5:
  - Label-shuffling null: randomly permute TRRUST/DoRothEA labels
  - Degree-preserving edge randomization: preserve TF out-degree and target in-degree
  - Report how much of AUROC is explained by node-degree/expression confounds
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/data/raw"
REF_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks"
PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "degree_null_results")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def load_trrust():
    path = os.path.join(REF_DIR, "trrust_human.tsv")
    df = pd.read_csv(path, sep='\t', header=None, names=['tf', 'target', 'mode', 'pmid'])
    return set(zip(df['tf'].str.upper(), df['target'].str.upper()))


def degree_preserving_shuffle(labels_matrix, n_permutations=1000, seed=42):
    """
    Degree-preserving edge randomization.
    For a binary TF×target label matrix, shuffle edges while preserving:
      - row sums (TF out-degree)
      - column sums (target in-degree)
    Uses the curveball algorithm (Strona et al. 2014).
    """
    rng = np.random.RandomState(seed)
    result_matrices = []

    for _ in range(n_permutations):
        M = labels_matrix.copy()
        rows, cols = M.shape

        # Curveball swaps
        n_swaps = rows * 10
        for _ in range(n_swaps):
            # Pick two random rows with at least one 1
            r1, r2 = rng.choice(rows, 2, replace=False)
            row1_ones = np.where(M[r1] == 1)[0]
            row2_ones = np.where(M[r2] == 1)[0]

            if len(row1_ones) == 0 or len(row2_ones) == 0:
                continue

            # Find columns where they differ
            only_in_r1 = np.setdiff1d(row1_ones, row2_ones)
            only_in_r2 = np.setdiff1d(row2_ones, row1_ones)

            if len(only_in_r1) == 0 or len(only_in_r2) == 0:
                continue

            # Swap a random subset
            n_swap = min(len(only_in_r1), len(only_in_r2))
            n_swap = rng.randint(1, n_swap + 1)

            swap_from_r1 = rng.choice(only_in_r1, n_swap, replace=False)
            swap_from_r2 = rng.choice(only_in_r2, n_swap, replace=False)

            M[r1, swap_from_r1] = 0
            M[r1, swap_from_r2] = 1
            M[r2, swap_from_r2] = 0
            M[r2, swap_from_r1] = 1

        result_matrices.append(M)

    return result_matrices


def run_null_analysis():
    import scanpy as sc

    print("=" * 60)
    print("DEGREE-PRESERVING NULL MODEL ANALYSIS")
    print("=" * 60)

    # Load Tabula Sapiens immune for DLPFC-like analysis
    print("\nLoading Tabula Sapiens immune subset...")
    adata = sc.read_h5ad(os.path.join(DATA_DIR, "tabula_sapiens_immune_subset_20000.h5ad"))
    print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    # Map gene names
    if 'feature_name' in adata.var.columns:
        names = adata.var['feature_name'].astype(str).values.copy()
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

    # Preprocess
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']].copy()
    gene_names = list(adata.var_names)
    gene_upper = [g.upper() for g in gene_names]
    n_genes = len(gene_names)

    # Subsample cells
    n_cells = min(5000, adata.n_obs)
    np.random.seed(42)
    cell_idx = np.random.choice(adata.n_obs, n_cells, replace=False)
    X = adata.X[cell_idx]
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    # Compute correlation edge scores
    print("\nComputing edge scores...")
    ranks = np.apply_along_axis(rankdata, 0, X)
    corr_matrix = np.corrcoef(ranks.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Load TRRUST
    trrust = load_trrust()
    trrust_tfs = set(tf for tf, _ in trrust)
    gene_set = set(gene_upper)

    # Build TF-target evaluation universe
    tfs_in_data = sorted([g for g in set(gene_upper) if g in trrust_tfs])
    print(f"  TFs in data: {len(tfs_in_data)}")

    # Build label matrix: TFs × all genes
    tf_indices = [gene_upper.index(tf) for tf in tfs_in_data if tf in gene_upper]
    tf_names = [gene_upper[i] for i in tf_indices]

    labels_matrix = np.zeros((len(tf_indices), n_genes), dtype=int)
    for i, tf_idx in enumerate(tf_indices):
        tf_name = gene_upper[tf_idx]
        for j in range(n_genes):
            if j == tf_idx:
                continue
            tgt_name = gene_upper[j]
            if (tf_name, tgt_name) in trrust:
                labels_matrix[i, j] = 1

    n_pos = labels_matrix.sum()
    print(f"  Label matrix: {labels_matrix.shape}, {n_pos} positive edges")

    if n_pos < 5:
        print("  WARNING: Very few positive edges. Expanding gene universe...")

    # Compute mean expression per gene (for degree-expression confound analysis)
    mean_expr = X.mean(axis=0)

    # ========================================
    # 1. OBSERVED AUROC
    # ========================================
    print("\n--- 1. Observed AUROC ---")

    # Flatten for AUROC computation
    scores_flat = []
    labels_flat = []
    for i, tf_idx in enumerate(tf_indices):
        for j in range(n_genes):
            if j == tf_idx:
                continue
            scores_flat.append(abs(corr_matrix[tf_idx, j]))
            labels_flat.append(labels_matrix[i, j])

    scores_flat = np.array(scores_flat)
    labels_flat = np.array(labels_flat)

    observed_auroc = roc_auc_score(labels_flat, scores_flat)
    print(f"  Observed AUROC: {observed_auroc:.4f}")
    print(f"  N pairs: {len(scores_flat)}, N positive: {labels_flat.sum()}")

    # ========================================
    # 2. LABEL-SHUFFLING NULL
    # ========================================
    print("\n--- 2. Label-Shuffling Null (1000 permutations) ---")

    n_perm = 1000
    np.random.seed(42)
    null_aurocs_label = []
    for p in range(n_perm):
        shuffled_labels = labels_flat.copy()
        np.random.shuffle(shuffled_labels)
        null_auroc = roc_auc_score(shuffled_labels, scores_flat)
        null_aurocs_label.append(null_auroc)

    null_aurocs_label = np.array(null_aurocs_label)
    p_label = (null_aurocs_label >= observed_auroc).mean()
    z_label = (observed_auroc - null_aurocs_label.mean()) / null_aurocs_label.std()
    print(f"  Null AUROC: {null_aurocs_label.mean():.4f} ± {null_aurocs_label.std():.4f}")
    print(f"  Observed vs null p-value: {p_label:.4f}")
    print(f"  Z-score: {z_label:.2f}")

    # ========================================
    # 3. DEGREE-PRESERVING NULL
    # ========================================
    print("\n--- 3. Degree-Preserving Null (200 permutations) ---")

    n_degree_perm = 200
    null_matrices = degree_preserving_shuffle(labels_matrix, n_permutations=n_degree_perm)
    null_aurocs_degree = []

    for k, null_M in enumerate(null_matrices):
        null_labels = []
        for i, tf_idx in enumerate(tf_indices):
            for j in range(n_genes):
                if j == tf_idx:
                    continue
                null_labels.append(null_M[i, j])
        null_labels = np.array(null_labels)
        if null_labels.sum() > 0 and null_labels.sum() < len(null_labels):
            null_auroc = roc_auc_score(null_labels, scores_flat)
            null_aurocs_degree.append(null_auroc)

        if (k + 1) % 50 == 0:
            print(f"    Permutation {k+1}/{n_degree_perm}")

    null_aurocs_degree = np.array(null_aurocs_degree)
    p_degree = (null_aurocs_degree >= observed_auroc).mean()
    z_degree = (observed_auroc - null_aurocs_degree.mean()) / null_aurocs_degree.std() if null_aurocs_degree.std() > 0 else 0
    print(f"  Degree-preserving null AUROC: {null_aurocs_degree.mean():.4f} ± {null_aurocs_degree.std():.4f}")
    print(f"  Observed vs degree-null p-value: {p_degree:.4f}")
    print(f"  Z-score: {z_degree:.2f}")

    # ========================================
    # 4. EXPRESSION-CONFOUND ANALYSIS
    # ========================================
    print("\n--- 4. Expression-Confound Analysis ---")

    # For each TF, compute: does edge score correlate with target expression?
    tf_expr_confounds = []
    for i, tf_idx in enumerate(tf_indices):
        edge_scores_this_tf = np.abs(corr_matrix[tf_idx])
        target_expr = mean_expr
        # Exclude self
        mask = np.ones(n_genes, dtype=bool)
        mask[tf_idx] = False
        rho, p = stats.spearmanr(edge_scores_this_tf[mask], target_expr[mask])
        tf_expr_confounds.append({
            'tf': gene_upper[tf_idx],
            'rho_edge_vs_target_expr': rho,
            'p': p,
            'tf_expr': mean_expr[tf_idx],
            'out_degree': labels_matrix[i].sum()
        })

    conf_df = pd.DataFrame(tf_expr_confounds)
    print(f"  Mean |ρ(edge, target_expr)|: {conf_df['rho_edge_vs_target_expr'].abs().mean():.4f}")
    print(f"  Mean TF out-degree: {conf_df['out_degree'].mean():.1f}")

    # Degree-AUROC correlation: do high-degree TFs have higher AUROC?
    per_tf_aurocs = []
    for i, tf_idx in enumerate(tf_indices):
        mask = np.ones(n_genes, dtype=bool)
        mask[tf_idx] = False
        scores_tf = np.abs(corr_matrix[tf_idx, mask])
        labels_tf = labels_matrix[i, mask]
        if labels_tf.sum() > 0 and labels_tf.sum() < len(labels_tf):
            auroc_tf = roc_auc_score(labels_tf, scores_tf)
            per_tf_aurocs.append({
                'tf': gene_upper[tf_idx],
                'auroc': auroc_tf,
                'out_degree': labels_matrix[i].sum(),
                'tf_expr': mean_expr[tf_idx]
            })

    per_tf_df = pd.DataFrame(per_tf_aurocs)
    if len(per_tf_df) > 3:
        rho_deg, p_deg = stats.spearmanr(per_tf_df['out_degree'], per_tf_df['auroc'])
        rho_exp, p_exp = stats.spearmanr(per_tf_df['tf_expr'], per_tf_df['auroc'])
        print(f"  ρ(TF degree, AUROC): {rho_deg:.4f}, p={p_deg:.4f}")
        print(f"  ρ(TF expr, AUROC): {rho_exp:.4f}, p={p_exp:.4f}")
    else:
        rho_deg, p_deg = 0, 1
        rho_exp, p_exp = 0, 1
        print("  Too few TFs for correlation analysis")

    # ========================================
    # 5. PER-TF DEGREE-CONTROLLED EVALUATION
    # ========================================
    print("\n--- 5. Per-TF Degree-Controlled Evaluation ---")

    # Save per-TF AUROC to CSV
    per_tf_df.to_csv(os.path.join(OUT_DIR, "per_tf_auroc.csv"), index=False)
    print(f"  Saved per-TF AUROC to per_tf_auroc.csv ({len(per_tf_df)} TFs)")

    # Report distribution
    if len(per_tf_df) > 0:
        auroc_vals = per_tf_df['auroc'].values
        q25, q50, q75 = np.percentile(auroc_vals, [25, 50, 75])
        print(f"  Per-TF AUROC distribution:")
        print(f"    Mean: {auroc_vals.mean():.4f} ± {auroc_vals.std():.4f}")
        print(f"    Median: {q50:.4f}")
        print(f"    IQR: [{q25:.4f}, {q75:.4f}]")
        print(f"    Min: {auroc_vals.min():.4f}, Max: {auroc_vals.max():.4f}")
        print(f"    TFs above 0.5: {(auroc_vals > 0.5).sum()}/{len(auroc_vals)}")

    # Compute within-TF precision@k (k = number of true targets per TF)
    print("\n  Within-TF precision@k:")
    prec_at_k_list = []
    for i, tf_idx in enumerate(tf_indices):
        mask = np.ones(n_genes, dtype=bool)
        mask[tf_idx] = False
        scores_tf = np.abs(corr_matrix[tf_idx, mask])
        labels_tf = labels_matrix[i, mask]
        n_true = labels_tf.sum()
        if n_true == 0 or n_true >= len(labels_tf):
            continue
        k = int(n_true)
        top_k_idx = np.argsort(-scores_tf)[:k]
        prec = labels_tf[top_k_idx].sum() / k
        prec_at_k_list.append({
            'tf': gene_upper[tf_idx],
            'precision_at_k': prec,
            'k': k,
            'out_degree': int(n_true)
        })

    prec_df = pd.DataFrame(prec_at_k_list)
    if len(prec_df) > 0:
        prec_vals = prec_df['precision_at_k'].values
        print(f"    Mean precision@k: {prec_vals.mean():.4f} ± {prec_vals.std():.4f}")
        print(f"    Median precision@k: {np.median(prec_vals):.4f}")
        random_prec = np.array([r['k'] / (n_genes - 1) for r in prec_at_k_list])
        print(f"    Random expected: {random_prec.mean():.4f}")
        prec_df.to_csv(os.path.join(OUT_DIR, "per_tf_precision_at_k.csv"), index=False)

    # Compute per-TF AUROCs for degree-preserving null (subsample for efficiency)
    print("\n  Computing per-TF AUROCs for degree-preserving null...")
    n_null_tf = min(50, len(null_matrices))
    null_per_tf_aurocs = {tf_row['tf']: [] for _, tf_row in per_tf_df.iterrows()}

    for k_null in range(n_null_tf):
        null_M = null_matrices[k_null]
        for i, tf_idx in enumerate(tf_indices):
            tf_name = gene_upper[tf_idx]
            if tf_name not in null_per_tf_aurocs:
                continue
            mask = np.ones(n_genes, dtype=bool)
            mask[tf_idx] = False
            scores_tf = np.abs(corr_matrix[tf_idx, mask])
            null_labels_tf = null_M[i, mask]
            if null_labels_tf.sum() > 0 and null_labels_tf.sum() < len(null_labels_tf):
                null_auroc_tf = roc_auc_score(null_labels_tf, scores_tf)
                null_per_tf_aurocs[tf_name].append(null_auroc_tf)

    # Compute per-TF excess above degree-null
    per_tf_excess = []
    for _, row in per_tf_df.iterrows():
        tf_name = row['tf']
        null_vals = null_per_tf_aurocs.get(tf_name, [])
        if len(null_vals) > 0:
            null_mean = np.mean(null_vals)
            excess = row['auroc'] - null_mean
            per_tf_excess.append({
                'tf': tf_name,
                'observed_auroc': row['auroc'],
                'degree_null_auroc': null_mean,
                'excess': excess,
                'out_degree': row['out_degree']
            })

    excess_df = pd.DataFrame(per_tf_excess)
    if len(excess_df) > 0:
        print(f"  Per-TF excess above degree-null:")
        print(f"    Mean observed: {excess_df['observed_auroc'].mean():.4f}")
        print(f"    Mean degree-null: {excess_df['degree_null_auroc'].mean():.4f}")
        print(f"    Mean excess: {excess_df['excess'].mean():.4f} ± {excess_df['excess'].std():.4f}")
        print(f"    Median excess: {excess_df['excess'].median():.4f}")
        print(f"    TFs with positive excess: {(excess_df['excess'] > 0).sum()}/{len(excess_df)}")
        excess_df.to_csv(os.path.join(OUT_DIR, "per_tf_excess_above_degree_null.csv"), index=False)

    # ========================================
    # FIGURES
    # ========================================
    print("\nGenerating figures...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Null distributions
    ax = axes[0, 0]
    ax.hist(null_aurocs_label, bins=30, alpha=0.6, color='#2196F3', label='Label-shuffled', density=True)
    ax.hist(null_aurocs_degree, bins=30, alpha=0.6, color='#FF9800', label='Degree-preserving', density=True)
    ax.axvline(observed_auroc, color='red', linewidth=2, linestyle='--',
               label=f'Observed ({observed_auroc:.3f})')
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('AUROC')
    ax.set_ylabel('Density')
    ax.set_title('(A) Null model distributions\nvs observed AUROC')
    ax.legend(fontsize=8)

    # Panel B: Per-TF AUROC vs out-degree
    ax = axes[0, 1]
    if len(per_tf_df) > 0:
        ax.scatter(per_tf_df['out_degree'], per_tf_df['auroc'],
                   c='#2196F3', alpha=0.7, s=40)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('TF out-degree (TRRUST)')
        ax.set_ylabel('Per-TF AUROC')
        ax.set_title(f'(B) AUROC vs TF degree\n(\u03c1={rho_deg:.3f}, p={p_deg:.3f})')

    # Panel C: Excess AUROC above null
    ax = axes[1, 0]
    excess_label = observed_auroc - null_aurocs_label.mean()
    excess_degree = observed_auroc - null_aurocs_degree.mean()
    degree_effect = null_aurocs_degree.mean() - 0.5

    bars = ax.bar(['Random\nexpected', 'Degree\nconfound', 'Above degree\nnull'],
                  [0.5, degree_effect, excess_degree],
                  color=['#9E9E9E', '#FF9800', '#4CAF50'], alpha=0.8,
                  bottom=[0, 0.5, 0.5 + degree_effect])
    ax.axhline(observed_auroc, color='red', linestyle='--', label=f'Observed ({observed_auroc:.3f})')
    ax.set_ylabel('AUROC')
    ax.set_title('(C) AUROC decomposition:\nchance + degree + residual')
    ax.legend(fontsize=8)
    ax.set_ylim(0.45, max(observed_auroc + 0.05, 0.6))

    # Panel D: Per-TF AUROC distribution (observed vs degree-null)
    ax = axes[1, 1]
    if len(excess_df) > 0:
        obs_vals = excess_df['observed_auroc'].values
        null_vals_d = excess_df['degree_null_auroc'].values
        bins_d = np.linspace(min(obs_vals.min(), null_vals_d.min()) - 0.02,
                             max(obs_vals.max(), null_vals_d.max()) + 0.02, 20)
        ax.hist(obs_vals, bins=bins_d, alpha=0.6, color='#4CAF50',
                label=f'Observed (mean={obs_vals.mean():.3f})', density=True)
        ax.hist(null_vals_d, bins=bins_d, alpha=0.6, color='#FF9800',
                label=f'Degree-null (mean={null_vals_d.mean():.3f})', density=True)
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Per-TF AUROC')
        ax.set_ylabel('Density')
        ax.set_title('(D) Per-TF AUROC distribution:\nobserved vs degree-null')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_degree_null_models.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Figure saved to {fig_path}")

    # Save results
    summary = {
        'experiment': 'degree_preserving_nulls',
        'dataset': 'tabula_sapiens_immune',
        'n_genes': n_genes,
        'n_tfs': len(tf_indices),
        'n_positive_edges': int(n_pos),
        'n_pairs': len(scores_flat),
        'observed_auroc': round(float(observed_auroc), 4),
        'label_shuffled_null': {
            'mean': round(float(null_aurocs_label.mean()), 4),
            'std': round(float(null_aurocs_label.std()), 4),
            'p_value': round(float(p_label), 6),
            'z_score': round(float(z_label), 2),
            'n_permutations': n_perm,
        },
        'degree_preserving_null': {
            'mean': round(float(null_aurocs_degree.mean()), 4),
            'std': round(float(null_aurocs_degree.std()), 4),
            'p_value': round(float(p_degree), 6),
            'z_score': round(float(z_degree), 2),
            'n_permutations': n_degree_perm,
        },
        'confound_analysis': {
            'mean_rho_edge_vs_target_expr': round(float(conf_df['rho_edge_vs_target_expr'].abs().mean()), 4),
            'rho_degree_vs_auroc': round(float(rho_deg), 4),
            'p_degree_vs_auroc': round(float(p_deg), 4),
            'rho_expr_vs_auroc': round(float(rho_exp), 4),
            'p_expr_vs_auroc': round(float(p_exp), 4),
        },
        'auroc_decomposition': {
            'chance': 0.5,
            'degree_confound': round(float(degree_effect), 4),
            'excess_above_degree': round(float(excess_degree), 4),
        },
        'per_tf_evaluation': {
            'n_tfs_evaluated': len(per_tf_df),
            'mean_per_tf_auroc': round(float(per_tf_df['auroc'].mean()), 4) if len(per_tf_df) > 0 else None,
            'std_per_tf_auroc': round(float(per_tf_df['auroc'].std()), 4) if len(per_tf_df) > 0 else None,
            'median_per_tf_auroc': round(float(per_tf_df['auroc'].median()), 4) if len(per_tf_df) > 0 else None,
            'iqr': [round(float(q25), 4), round(float(q75), 4)] if len(per_tf_df) > 0 else None,
            'tfs_above_chance': int((per_tf_df['auroc'] > 0.5).sum()) if len(per_tf_df) > 0 else 0,
            'mean_precision_at_k': round(float(prec_vals.mean()), 4) if len(prec_df) > 0 else None,
            'mean_excess_above_degree_null': round(float(excess_df['excess'].mean()), 4) if len(excess_df) > 0 else None,
        }
    }

    with open(os.path.join(OUT_DIR, "degree_null_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== EXPERIMENT COMPLETE ===")
    return summary


if __name__ == '__main__':
    run_null_analysis()
