#!/usr/bin/env python3
"""
Round 9 Experiment 3: Attention-Specific Confound Decomposition
================================================================
Applies residualization and degree-preserving null analysis DIRECTLY
to attention-derived edges (not just correlation proxies).

Addresses: MC4 ("core decompositions still use correlation proxies")
           ST2 ("attention-specific confound decomposition")
"""

import os
import sys
import gc
import json
import pickle
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from multiprocessing import Pool, cpu_count

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
EXTRACTION_DIR = os.path.join(PAPER_DIR, "experiments", "round9_unified_extraction")
REF_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks"
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round9_attention_confound_decomposition")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PRIMARY_LAYER = 12
N_RESID_SEEDS = 10
N_NULL_PERMUTATIONS = 200


def load_trrust():
    path = os.path.join(REF_DIR, "trrust_human.tsv")
    df = pd.read_csv(path, sep='\t', header=None, names=['tf', 'target', 'mode', 'pmid'])
    return set(zip(df['tf'].str.upper(), df['target'].str.upper()))


def cross_fitted_residual_auroc(scores, covariates, labels, model_type='ols', n_folds=5, seed=42):
    """5-fold cross-fitted residualization."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    residuals = np.zeros_like(scores)
    r2_folds = []

    for train_idx, test_idx in kf.split(scores):
        X_train = covariates[train_idx]
        y_train = scores[train_idx]
        X_test = covariates[test_idx]
        y_test = scores[test_idx]

        if model_type == 'ols':
            model = LinearRegression()
        else:
            model = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                              random_state=seed, subsample=0.8)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        residuals[test_idx] = y_test - pred

        ss_res = np.sum((y_test - pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2_folds.append(r2)

    abs_residuals = np.abs(residuals)
    try:
        auroc = roc_auc_score(labels, abs_residuals)
    except:
        auroc = 0.5

    return {
        'auroc_abs_residual': float(auroc),
        'mean_r2': float(np.mean(r2_folds)),
        'std_r2': float(np.std(r2_folds)),
    }


def degree_preserving_shuffle(label_matrix, n_permutations=200, seed=42):
    """Curveball algorithm: preserve row/column sums of binary matrix."""
    rng = np.random.RandomState(seed)
    nrows, ncols = label_matrix.shape
    row_sums = label_matrix.sum(axis=1).astype(int)
    col_sums = label_matrix.sum(axis=0).astype(int)

    null_aurocs = []
    for perm in range(n_permutations):
        shuffled = np.zeros_like(label_matrix)
        # Build random matrix with same row/column marginals
        # Using the sequential filling approach
        for i in range(nrows):
            n_ones = row_sums[i]
            if n_ones == 0:
                continue
            # Weight by remaining column capacity
            remaining = col_sums - shuffled.sum(axis=0)
            remaining = np.maximum(remaining, 0)
            if remaining.sum() == 0:
                continue
            probs = remaining / remaining.sum()
            n_available = int((probs > 0).sum())
            n_to_place = min(n_ones, int(remaining.sum()), n_available)
            if n_to_place == 0:
                continue
            chosen = rng.choice(ncols, size=n_to_place, replace=False, p=probs)
            shuffled[i, chosen] = 1

        null_aurocs.append(shuffled)

    return null_aurocs


def _null_auroc_worker(args):
    """Worker for parallelized null AUROC computation."""
    null_labels, score_matrix, n_pos_original = args
    flat_labels = null_labels.ravel()
    flat_scores = score_matrix.ravel()
    if flat_labels.sum() == 0 or flat_labels.sum() == len(flat_labels):
        return 0.5
    try:
        return float(roc_auc_score(flat_labels, flat_scores))
    except:
        return 0.5


def run():
    print("=" * 70)
    print("ATTENTION-SPECIFIC CONFOUND DECOMPOSITION")
    print("=" * 70)

    # Load data
    print("\nLoading extraction data...")
    with open(os.path.join(EXTRACTION_DIR, "hvg_gene_list.json")) as f:
        hvg_genes = json.load(f)
    hvg_upper = [g.upper() for g in hvg_genes]
    hvg_to_local = {g.upper(): i for i, g in enumerate(hvg_genes)}

    corr_matrix = np.load(os.path.join(EXTRACTION_DIR, "correlation_matrix.npy"))
    gene_features_df = pd.read_csv(os.path.join(EXTRACTION_DIR, "gene_features.csv"))

    edge_path = os.path.join(EXTRACTION_DIR, "layer_mean_edges", f"layer_{PRIMARY_LAYER:02d}_edges.pkl")
    with open(edge_path, 'rb') as f:
        attn_edges = pickle.load(f)

    print(f"  L{PRIMARY_LAYER} attention: {len(attn_edges)} edges")

    # Load TRRUST
    trrust = load_trrust()
    trrust_tfs = set(tf for tf, _ in trrust)
    hvg_upper_set = set(hvg_upper)
    tfs_in_data = sorted([g for g in hvg_upper_set if g in trrust_tfs])
    print(f"  TRRUST TFs in HVG: {len(tfs_in_data)}")

    # Build TF × target evaluation universe
    tf_indices = [hvg_to_local[tf] for tf in tfs_in_data]
    n_tfs = len(tfs_in_data)
    n_genes = len(hvg_genes)

    # Build label matrix, attention score matrix, correlation score matrix
    label_matrix = np.zeros((n_tfs, n_genes), dtype=int)
    attn_score_matrix = np.zeros((n_tfs, n_genes), dtype=float)
    corr_score_matrix = np.zeros((n_tfs, n_genes), dtype=float)

    for i, tf_name in enumerate(tfs_in_data):
        tf_idx = hvg_to_local[tf_name]
        tf_orig = hvg_genes[tf_idx]

        for j in range(n_genes):
            if j == tf_idx:
                continue
            tgt_name = hvg_upper[j]
            tgt_orig = hvg_genes[j]

            # Label
            if (tf_name, tgt_name) in trrust:
                label_matrix[i, j] = 1

            # Attention score
            fwd = attn_edges.get((tf_orig, tgt_orig), 0.0)
            rev = attn_edges.get((tgt_orig, tf_orig), 0.0)
            attn_score_matrix[i, j] = max(fwd, rev)

            # Correlation score
            corr_score_matrix[i, j] = abs(corr_matrix[tf_idx, j])

    n_pos = label_matrix.sum()
    print(f"  Evaluation universe: {n_tfs} TFs × {n_genes} genes, {n_pos} positive edges")

    # ============================================================
    # Part A: Residualization
    # ============================================================
    print("\n--- PART A: Residualization ---")

    # Build covariates for each (TF, target) pair
    mean_expr = gene_features_df['mean_expr'].values
    variance = gene_features_df['variance'].values
    dropout = gene_features_df['dropout_rate'].values

    # Flatten for residualization
    flat_labels = label_matrix.ravel()
    flat_attn = attn_score_matrix.ravel()
    flat_corr = corr_score_matrix.ravel()

    # Build covariate matrix
    covariates_list = []
    for i in range(n_tfs):
        tf_idx = hvg_to_local[tfs_in_data[i]]
        for j in range(n_genes):
            covariates_list.append([
                mean_expr[tf_idx],    # TF mean expr
                mean_expr[j],         # target mean expr
                mean_expr[tf_idx] * mean_expr[j],  # interaction
                variance[j],          # target variance
                dropout[j],           # target dropout
            ])
    covariates = np.array(covariates_list)

    # Mask: exclude self-pairs (TF == target)
    mask = flat_labels >= 0  # all True initially
    for i, tf_name in enumerate(tfs_in_data):
        tf_idx = hvg_to_local[tf_name]
        mask[i * n_genes + tf_idx] = False

    flat_labels_m = flat_labels[mask]
    flat_attn_m = flat_attn[mask]
    flat_corr_m = flat_corr[mask]
    covariates_m = covariates[mask]

    # Baseline AUROCs
    attn_baseline_auroc = roc_auc_score(flat_labels_m, flat_attn_m)
    corr_baseline_auroc = roc_auc_score(flat_labels_m, flat_corr_m)
    print(f"  Attention baseline AUROC: {attn_baseline_auroc:.4f}")
    print(f"  Correlation baseline AUROC: {corr_baseline_auroc:.4f}")

    # Cross-fitted residualization for attention
    print("  Residualizing attention edges (OLS)...")
    attn_resid_ols = cross_fitted_residual_auroc(flat_attn_m, covariates_m, flat_labels_m, 'ols')
    print(f"    OLS R² = {attn_resid_ols['mean_r2']:.4f}, residual AUROC = {attn_resid_ols['auroc_abs_residual']:.4f}")

    print("  Residualizing attention edges (GBDT)...")
    attn_resid_gbdt = cross_fitted_residual_auroc(flat_attn_m, covariates_m, flat_labels_m, 'gbdt')
    print(f"    GBDT R² = {attn_resid_gbdt['mean_r2']:.4f}, residual AUROC = {attn_resid_gbdt['auroc_abs_residual']:.4f}")

    # Multi-seed stability for attention OLS
    print(f"  Stability test ({N_RESID_SEEDS} seeds)...")
    stability_aurocs = []
    for seed in range(N_RESID_SEEDS):
        result = cross_fitted_residual_auroc(flat_attn_m, covariates_m, flat_labels_m, 'ols', seed=seed)
        stability_aurocs.append(result['auroc_abs_residual'])
    stability_aurocs = np.array(stability_aurocs)
    print(f"    Stability: {stability_aurocs.mean():.4f} ± {stability_aurocs.std():.4f}")

    # Cross-fitted residualization for correlation (for comparison)
    print("  Residualizing correlation edges (OLS)...")
    corr_resid_ols = cross_fitted_residual_auroc(flat_corr_m, covariates_m, flat_labels_m, 'ols')
    print(f"    OLS R² = {corr_resid_ols['mean_r2']:.4f}, residual AUROC = {corr_resid_ols['auroc_abs_residual']:.4f}")

    print("  Residualizing correlation edges (GBDT)...")
    corr_resid_gbdt = cross_fitted_residual_auroc(flat_corr_m, covariates_m, flat_labels_m, 'gbdt')
    print(f"    GBDT R² = {corr_resid_gbdt['mean_r2']:.4f}, residual AUROC = {corr_resid_gbdt['auroc_abs_residual']:.4f}")

    residualization_results = {
        'attention': {
            'baseline_auroc': round(attn_baseline_auroc, 4),
            'ols': attn_resid_ols,
            'gbdt': attn_resid_gbdt,
            'stability_mean': round(float(stability_aurocs.mean()), 4),
            'stability_std': round(float(stability_aurocs.std()), 4),
        },
        'correlation': {
            'baseline_auroc': round(corr_baseline_auroc, 4),
            'ols': corr_resid_ols,
            'gbdt': corr_resid_gbdt,
        },
    }

    # ============================================================
    # Part B: Degree-Preserving Null (parallelized)
    # ============================================================
    print(f"\n--- PART B: Degree-Preserving Null ({N_NULL_PERMUTATIONS} permutations) ---")

    # Generate null label matrices
    print("  Generating degree-preserving null matrices...")
    null_matrices = degree_preserving_shuffle(label_matrix, N_NULL_PERMUTATIONS)

    # Compute null AUROCs for attention (parallelized)
    print(f"  Computing null AUROCs (parallelized, {min(cpu_count(), 8)} workers)...")
    n_workers = min(cpu_count(), 8)

    null_args_attn = [(nm.ravel()[mask], flat_attn_m, n_pos) for nm in null_matrices]

    with Pool(n_workers) as pool:
        null_aurocs_attn = pool.map(_null_auroc_worker, null_args_attn)
    null_aurocs_attn = np.array(null_aurocs_attn)

    # Also compute for correlation
    null_args_corr = [(nm.ravel()[mask], flat_corr_m, n_pos) for nm in null_matrices]
    with Pool(n_workers) as pool:
        null_aurocs_corr = pool.map(_null_auroc_worker, null_args_corr)
    null_aurocs_corr = np.array(null_aurocs_corr)

    # Statistics
    attn_z = (attn_baseline_auroc - null_aurocs_attn.mean()) / null_aurocs_attn.std() if null_aurocs_attn.std() > 0 else 0
    corr_z = (corr_baseline_auroc - null_aurocs_corr.mean()) / null_aurocs_corr.std() if null_aurocs_corr.std() > 0 else 0
    attn_excess = attn_baseline_auroc - null_aurocs_attn.mean()
    corr_excess = corr_baseline_auroc - null_aurocs_corr.mean()

    print(f"\n  Attention: observed={attn_baseline_auroc:.4f}, degree-null={null_aurocs_attn.mean():.4f}, "
          f"excess={attn_excess:.4f}, Z={attn_z:.2f}")
    print(f"  Correlation: observed={corr_baseline_auroc:.4f}, degree-null={null_aurocs_corr.mean():.4f}, "
          f"excess={corr_excess:.4f}, Z={corr_z:.2f}")

    degree_null_results = {
        'attention': {
            'observed_auroc': round(attn_baseline_auroc, 4),
            'degree_null_mean': round(float(null_aurocs_attn.mean()), 4),
            'degree_null_std': round(float(null_aurocs_attn.std()), 4),
            'excess': round(float(attn_excess), 4),
            'z_score': round(float(attn_z), 2),
        },
        'correlation': {
            'observed_auroc': round(corr_baseline_auroc, 4),
            'degree_null_mean': round(float(null_aurocs_corr.mean()), 4),
            'degree_null_std': round(float(null_aurocs_corr.std()), 4),
            'excess': round(float(corr_excess), 4),
            'z_score': round(float(corr_z), 2),
        },
    }

    # Per-TF evaluation
    print("\n  Per-TF attention AUROC...")
    per_tf_results = []
    for i, tf_name in enumerate(tfs_in_data):
        tf_labels = label_matrix[i, :]
        tf_attn = attn_score_matrix[i, :]
        tf_idx = hvg_to_local[tf_name]

        # Mask self
        mask_tf = np.ones(n_genes, dtype=bool)
        mask_tf[tf_idx] = False

        if tf_labels[mask_tf].sum() == 0 or tf_labels[mask_tf].sum() == mask_tf.sum():
            continue

        try:
            auroc = roc_auc_score(tf_labels[mask_tf], tf_attn[mask_tf])
        except:
            continue

        per_tf_results.append({
            'tf': tf_name,
            'n_targets': int(tf_labels.sum()),
            'auroc': round(float(auroc), 4),
        })

    per_tf_df = pd.DataFrame(per_tf_results)
    if len(per_tf_df) > 0:
        print(f"  Per-TF: {len(per_tf_df)} TFs, mean AUROC = {per_tf_df['auroc'].mean():.4f}")

    # ============================================================
    # Save results
    # ============================================================
    all_results = {
        'experiment': 'round9_attention_confound_decomposition',
        'primary_layer': PRIMARY_LAYER,
        'n_tfs': n_tfs,
        'n_genes': n_genes,
        'n_positive_edges': int(n_pos),
        'residualization': residualization_results,
        'degree_null': degree_null_results,
        'per_tf_mean_auroc': round(float(per_tf_df['auroc'].mean()), 4) if len(per_tf_df) > 0 else None,
    }

    with open(os.path.join(OUT_DIR, "attention_confound_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    if len(per_tf_df) > 0:
        per_tf_df.to_csv(os.path.join(OUT_DIR, "per_tf_attention_auroc.csv"), index=False)

    # Comparison table
    comparison = pd.DataFrame([
        {
            'edge_type': 'Attention (L13)',
            'baseline_auroc': attn_baseline_auroc,
            'ols_r2': attn_resid_ols['mean_r2'],
            'gbdt_r2': attn_resid_gbdt['mean_r2'],
            'residual_auroc_ols': attn_resid_ols['auroc_abs_residual'],
            'residual_auroc_gbdt': attn_resid_gbdt['auroc_abs_residual'],
            'degree_null_auroc': null_aurocs_attn.mean(),
            'excess_above_degree': attn_excess,
        },
        {
            'edge_type': 'Correlation',
            'baseline_auroc': corr_baseline_auroc,
            'ols_r2': corr_resid_ols['mean_r2'],
            'gbdt_r2': corr_resid_gbdt['mean_r2'],
            'residual_auroc_ols': corr_resid_ols['auroc_abs_residual'],
            'residual_auroc_gbdt': corr_resid_gbdt['auroc_abs_residual'],
            'degree_null_auroc': null_aurocs_corr.mean(),
            'excess_above_degree': corr_excess,
        },
    ])
    comparison.to_csv(os.path.join(OUT_DIR, "comparison_attention_vs_correlation.csv"), index=False)
    print(f"\n  Comparison table saved")

    # ============================================================
    # Generate figure
    # ============================================================
    print("\nGenerating figure...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel A: R² comparison
    ax = axes[0]
    categories = ['OLS', 'GBDT']
    attn_r2 = [attn_resid_ols['mean_r2'], attn_resid_gbdt['mean_r2']]
    corr_r2 = [corr_resid_ols['mean_r2'], corr_resid_gbdt['mean_r2']]
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, attn_r2, width, label='Attention', color='#2196F3', alpha=0.8)
    ax.bar(x + width/2, corr_r2, width, label='Correlation', color='#FF9800', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('R² (covariates explaining edge scores)')
    ax.set_title('(A) Covariate explanation')
    ax.legend(fontsize=8)

    # Panel B: Residual AUROC
    ax = axes[1]
    methods = ['Baseline', 'OLS residual', 'GBDT residual']
    attn_vals = [attn_baseline_auroc, attn_resid_ols['auroc_abs_residual'], attn_resid_gbdt['auroc_abs_residual']]
    corr_vals = [corr_baseline_auroc, corr_resid_ols['auroc_abs_residual'], corr_resid_gbdt['auroc_abs_residual']]
    x = np.arange(len(methods))
    ax.bar(x - width/2, attn_vals, width, label='Attention', color='#2196F3', alpha=0.8)
    ax.bar(x + width/2, corr_vals, width, label='Correlation', color='#FF9800', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel('AUROC vs TRRUST')
    ax.set_title('(B) After residualization')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)

    # Panel C: Degree-null stacked bar
    ax = axes[2]
    categories = ['Attention', 'Correlation']
    chance = [0.5, 0.5]
    degree = [null_aurocs_attn.mean() - 0.5, null_aurocs_corr.mean() - 0.5]
    excess = [attn_excess, corr_excess]
    ax.bar(categories, chance, color='lightgray', label='Chance')
    ax.bar(categories, degree, bottom=chance, color='#FFC107', label='Degree confound')
    ax.bar(categories, excess, bottom=[c + d for c, d in zip(chance, degree)],
           color='#4CAF50', label='Signal above degree')
    ax.set_ylabel('AUROC decomposition')
    ax.set_title('(C) Degree-null decomposition')
    ax.legend(fontsize=8)

    # Panel D: Per-TF AUROC
    ax = axes[3]
    if len(per_tf_df) > 0:
        sorted_df = per_tf_df.sort_values('auroc', ascending=True)
        colors = ['#4CAF50' if a > 0.5 else '#F44336' for a in sorted_df['auroc']]
        ax.barh(range(len(sorted_df)), sorted_df['auroc'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['tf'], fontsize=6)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Per-TF Attention AUROC')
        ax.set_title('(D) Per-TF attention AUROC\nvs TRRUST')

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_attention_confound_decomposition.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Figure saved to {fig_path}")

    print(f"\n{'=' * 70}")
    print("ATTENTION CONFOUND DECOMPOSITION COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nKey: Attention baseline AUROC = {attn_baseline_auroc:.4f}")
    print(f"     After OLS residualization: {attn_resid_ols['auroc_abs_residual']:.4f}")
    print(f"     Degree-null: {null_aurocs_attn.mean():.4f}, excess: {attn_excess:.4f}")


if __name__ == '__main__':
    run()
