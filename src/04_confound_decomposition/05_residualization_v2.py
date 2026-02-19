#!/usr/bin/env python3
"""
Robust Attention Residualization Analysis (v2)
===============================================
Addresses Round 6 Reviewer Concern 3:
  - Report R² (not just ρ) for residualization model
  - Cross-fitting: train residualizer on one split, evaluate AUROC on held-out
  - Robustness to signed vs absolute correlations
  - Alternative covariate sets
  - Nonlinear residualizers (GBDT)
  - Report AUROC for both residual and sign-flipped residual
  - Proper interpretation of below-chance AUROC
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
DATA_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/data/raw"
REF_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks"
PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "residualization_results")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def load_trrust():
    """Load TRRUST v2 reference."""
    path = os.path.join(REF_DIR, "trrust_human.tsv")
    df = pd.read_csv(path, sep='\t', header=None,
                     names=['tf', 'target', 'mode', 'pmid'])
    return set(zip(df['tf'].str.upper(), df['target'].str.upper()))


def cross_fitted_residual_auroc(X_cov, y_edge, labels, n_splits=5, model_type='ols', seed=42):
    """
    Cross-fitted residualization: train residualizer on K-1 folds,
    predict on held-out fold, compute AUROC on held-out residuals.

    Returns per-fold AUROCs and concatenated residuals.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_residuals = np.zeros(len(y_edge))
    fold_aurocs = []

    for train_idx, test_idx in kf.split(X_cov):
        X_train, X_test = X_cov[train_idx], X_cov[test_idx]
        y_train, y_test = y_edge[train_idx], y_edge[test_idx]
        labels_test = labels[test_idx]

        if model_type == 'ols':
            model = LinearRegression()
        elif model_type == 'gbdt':
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=seed, subsample=0.8
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        residual = y_test - y_pred
        all_residuals[test_idx] = residual

        # Per-fold AUROC if enough positives
        n_pos = labels_test.sum()
        n_neg = len(labels_test) - n_pos
        if n_pos >= 1 and n_neg >= 1:
            fold_aurocs.append(roc_auc_score(labels_test, np.abs(residual)))

    # Overall AUROC on concatenated cross-fitted residuals
    overall_auroc_abs = roc_auc_score(labels, np.abs(all_residuals))
    overall_auroc_signed = roc_auc_score(labels, all_residuals)
    overall_auroc_neg = roc_auc_score(labels, -all_residuals)  # sign-flipped

    return {
        'overall_auroc_abs_residual': overall_auroc_abs,
        'overall_auroc_signed_residual': overall_auroc_signed,
        'overall_auroc_negated_residual': overall_auroc_neg,
        'per_fold_aurocs': fold_aurocs,
        'mean_fold_auroc': np.mean(fold_aurocs) if fold_aurocs else np.nan,
        'std_fold_auroc': np.std(fold_aurocs) if fold_aurocs else np.nan,
        'residuals': all_residuals
    }


def run_analysis():
    import scanpy as sc

    print("=" * 60)
    print("ROBUST ATTENTION RESIDUALIZATION ANALYSIS (v2)")
    print("=" * 60)

    # Load data
    print("\nLoading Tabula Sapiens immune subset...")
    adata = sc.read_h5ad(os.path.join(DATA_DIR, "tabula_sapiens_immune_subset_20000.h5ad"))
    print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    # Map Ensembl IDs to gene symbols
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
        print(f"  Mapped to gene symbols: {list(adata.var_names[:5])}")

    # Preprocess
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Use MORE HVGs to get more TRRUST positives (reviewer concern about n_positives)
    for n_hvg in [2000, 1500, 1000, 500]:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
        adata_hvg = adata[:, adata.var['highly_variable']].copy()
        gene_upper = [g.upper() for g in adata_hvg.var_names]
        trrust = load_trrust()
        trrust_tfs = set(tf for tf, _ in trrust)
        tfs_in_data = [g for g in gene_upper if g in trrust_tfs]
        # Count evaluable edges
        gene_set = set(gene_upper)
        n_pos = sum(1 for tf, tgt in trrust if tf in gene_set and tgt in gene_set and tf in trrust_tfs)
        print(f"  n_hvg={n_hvg}: {len(tfs_in_data)} TFs, {n_pos} evaluable TRRUST edges")
        if n_pos >= 20:
            print(f"  -> Using n_hvg={n_hvg}")
            break

    gene_names = list(adata_hvg.var_names)
    gene_upper = [g.upper() for g in gene_names]
    gene_to_idx = {g: i for i, g in enumerate(gene_upper)}
    n_genes = len(gene_names)
    print(f"  Final gene universe: {n_genes} genes, {len(tfs_in_data)} TFs, {n_pos} TRRUST edges")

    # Expression matrix
    X = adata_hvg.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    # Subsample cells
    n_cells = min(5000, adata_hvg.n_obs)
    np.random.seed(42)
    cell_idx = np.random.choice(adata_hvg.n_obs, n_cells, replace=False)
    X_sub = X[cell_idx]
    print(f"  Using {n_cells} cells")

    # ========================================
    # COMPUTE FEATURES
    # ========================================
    print("\nComputing edge features...")
    from scipy.stats import rankdata
    ranks = np.apply_along_axis(rankdata, 0, X_sub)
    corr_matrix = np.corrcoef(ranks.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    mean_expr = X_sub.mean(axis=0)
    dropout = (X_sub == 0).mean(axis=0)
    expr_var = X_sub.var(axis=0)

    # Build edge table (all TF-target pairs)
    records = []
    for i, g1 in enumerate(gene_upper):
        if g1 not in trrust_tfs:
            continue
        for j, g2 in enumerate(gene_upper):
            if i == j:
                continue
            records.append({
                'tf': g1, 'target': g2,
                'signed_corr': corr_matrix[i, j],
                'abs_corr': abs(corr_matrix[i, j]),
                'mean_expr_tf': mean_expr[i],
                'mean_expr_tgt': mean_expr[j],
                'expr_product': mean_expr[i] * mean_expr[j],
                'expr_sum': mean_expr[i] + mean_expr[j],
                'dropout_tf': dropout[i],
                'dropout_tgt': dropout[j],
                'dropout_product': dropout[i] * dropout[j],
                'var_tf': expr_var[i],
                'var_tgt': expr_var[j],
                'is_trrust': 1 if (g1, g2) in trrust else 0
            })

    df = pd.DataFrame(records)
    n_pos = df['is_trrust'].sum()
    n_neg = len(df) - n_pos
    print(f"  Total pairs: {len(df)}, positives: {n_pos}, negatives: {n_neg}")
    print(f"  Positive rate: {n_pos/len(df):.6f}")

    if n_pos < 5:
        print("  WARNING: Very few positives. Results will be noisy.")

    # ========================================
    # 1. BASELINE METRICS
    # ========================================
    print("\n--- 1. Baseline Metrics ---")

    # Edge score = |correlation| (proxy for attention)
    baseline_auroc_abs = roc_auc_score(df['is_trrust'], df['abs_corr'])
    baseline_auroc_signed = roc_auc_score(df['is_trrust'], df['signed_corr'])
    print(f"  Baseline AUROC (|corr|): {baseline_auroc_abs:.4f}")
    print(f"  Baseline AUROC (signed corr): {baseline_auroc_signed:.4f}")

    # Spearman correlation between edge score and expression product
    rho_expr, p_expr = stats.spearmanr(df['abs_corr'], df['expr_product'])
    print(f"  Spearman(|corr|, expr_product): rho={rho_expr:.4f}, p={p_expr:.2e}")

    # ========================================
    # 2. R² FROM RESIDUALIZATION MODEL (not just ρ!)
    # ========================================
    print("\n--- 2. R² for Residualization Models ---")

    # Define covariate sets
    cov_sets = {
        'full': ['expr_product', 'expr_sum', 'mean_expr_tf', 'mean_expr_tgt',
                 'dropout_tf', 'dropout_tgt', 'dropout_product', 'var_tf', 'var_tgt'],
        'expr_only': ['expr_product', 'expr_sum', 'mean_expr_tf', 'mean_expr_tgt'],
        'minimal': ['expr_product'],
    }

    r2_results = {}
    for cov_name, cov_cols in cov_sets.items():
        X_cov = df[cov_cols].values
        y = df['abs_corr'].values
        reg = LinearRegression()
        reg.fit(X_cov, y)
        y_pred = reg.predict(X_cov)
        r2 = r2_score(y, y_pred)
        r2_results[cov_name] = r2
        print(f"  R²({cov_name} → |corr|): {r2:.4f}")

    # Also report for nonlinear (GBDT)
    X_cov_full = df[cov_sets['full']].values
    y_abs = df['abs_corr'].values
    gbdt = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    gbdt.fit(X_cov_full, y_abs)
    r2_gbdt = r2_score(y_abs, gbdt.predict(X_cov_full))
    print(f"  R²(GBDT full → |corr|): {r2_gbdt:.4f}")

    # ========================================
    # 3. CROSS-FITTED RESIDUALIZATION (main robustness check)
    # ========================================
    print("\n--- 3. Cross-Fitted Residualization ---")
    labels = df['is_trrust'].values

    cf_results = {}
    for cov_name, cov_cols in cov_sets.items():
        X_cov = df[cov_cols].values

        # OLS cross-fitted
        ols_cf = cross_fitted_residual_auroc(X_cov, y_abs, labels, n_splits=5, model_type='ols')
        print(f"  OLS ({cov_name}): cross-fitted AUROC(|resid|)={ols_cf['overall_auroc_abs_residual']:.4f}, "
              f"AUROC(resid)={ols_cf['overall_auroc_signed_residual']:.4f}, "
              f"AUROC(-resid)={ols_cf['overall_auroc_negated_residual']:.4f}")
        cf_results[f'ols_{cov_name}'] = ols_cf

    # GBDT cross-fitted (full covariates only)
    X_cov_full = df[cov_sets['full']].values
    gbdt_cf = cross_fitted_residual_auroc(X_cov_full, y_abs, labels, n_splits=5, model_type='gbdt')
    print(f"  GBDT (full): cross-fitted AUROC(|resid|)={gbdt_cf['overall_auroc_abs_residual']:.4f}, "
          f"AUROC(resid)={gbdt_cf['overall_auroc_signed_residual']:.4f}, "
          f"AUROC(-resid)={gbdt_cf['overall_auroc_negated_residual']:.4f}")
    cf_results['gbdt_full'] = gbdt_cf

    # ========================================
    # 4. SIGNED VS ABSOLUTE CORRELATION SCORES
    # ========================================
    print("\n--- 4. Signed vs Absolute Edge Scores ---")

    # Residualize signed correlation
    y_signed = df['signed_corr'].values
    for cov_name in ['full', 'minimal']:
        X_cov = df[cov_sets[cov_name]].values
        signed_cf = cross_fitted_residual_auroc(X_cov, y_signed, labels, n_splits=5, model_type='ols')
        print(f"  Signed, OLS ({cov_name}): AUROC(|resid|)={signed_cf['overall_auroc_abs_residual']:.4f}, "
              f"AUROC(resid)={signed_cf['overall_auroc_signed_residual']:.4f}")
        cf_results[f'signed_ols_{cov_name}'] = signed_cf

    # ========================================
    # 5. STABILITY ACROSS RANDOM SEEDS
    # ========================================
    print("\n--- 5. Cross-Fitting Stability (multiple seeds) ---")
    stability_aurocs = []
    for seed in range(10):
        result = cross_fitted_residual_auroc(
            df[cov_sets['full']].values, y_abs, labels,
            n_splits=5, model_type='ols', seed=seed
        )
        stability_aurocs.append(result['overall_auroc_abs_residual'])

    print(f"  AUROC(|resid|) across 10 seeds: {np.mean(stability_aurocs):.4f} ± {np.std(stability_aurocs):.4f}")
    print(f"  Range: [{min(stability_aurocs):.4f}, {max(stability_aurocs):.4f}]")

    # ========================================
    # 6. INTERPRETATION OF BELOW-CHANCE AUROC
    # ========================================
    print("\n--- 6. Below-Chance AUROC Interpretation ---")

    # Non-cross-fitted (for comparison to original report)
    reg_nocf = LinearRegression()
    reg_nocf.fit(df[cov_sets['full']].values, y_abs)
    resid_nocf = y_abs - reg_nocf.predict(df[cov_sets['full']].values)
    auroc_nocf_abs = roc_auc_score(labels, np.abs(resid_nocf))
    auroc_nocf_signed = roc_auc_score(labels, resid_nocf)
    auroc_nocf_neg = roc_auc_score(labels, -resid_nocf)

    print(f"  Non-cross-fitted (full covs):")
    print(f"    AUROC(|resid|) = {auroc_nocf_abs:.4f}")
    print(f"    AUROC(resid)   = {auroc_nocf_signed:.4f}")
    print(f"    AUROC(-resid)  = {auroc_nocf_neg:.4f}")
    print(f"  Cross-fitted (full covs, OLS):")
    ols_full = cf_results['ols_full']
    print(f"    AUROC(|resid|) = {ols_full['overall_auroc_abs_residual']:.4f}")
    print(f"    AUROC(resid)   = {ols_full['overall_auroc_signed_residual']:.4f}")
    print(f"    AUROC(-resid)  = {ols_full['overall_auroc_negated_residual']:.4f}")

    print(f"\n  Interpretation:")
    if ols_full['overall_auroc_abs_residual'] < 0.5:
        print(f"    Below-chance AUROC ({ols_full['overall_auroc_abs_residual']:.3f}) with |residual| indicates")
        print(f"    that residualized scores are *systematically anti-associated* with regulatory labels.")
        print(f"    This means: after removing expression-covariate signal, TRRUST edges have")
        print(f"    LOWER absolute residuals than non-edges — i.e., the linear model over-predicts")
        print(f"    edge scores for true regulatory pairs relative to non-regulatory pairs.")
        print(f"    This is consistent with regulatory edges being well-explained by (not independent of)")
        print(f"    expression covariates.")
    else:
        print(f"    At-or-above-chance AUROC ({ols_full['overall_auroc_abs_residual']:.3f}) suggests")
        print(f"    some residual signal beyond expression covariates.")

    # ========================================
    # FIGURES
    # ========================================
    print("\nGenerating figures...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel A: R² across covariate sets
    ax = axes[0, 0]
    r2_names = list(r2_results.keys()) + ['GBDT_full']
    r2_vals = list(r2_results.values()) + [r2_gbdt]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    bars = ax.bar(range(len(r2_names)), r2_vals, color=colors[:len(r2_names)], alpha=0.8)
    ax.set_xticks(range(len(r2_names)))
    ax.set_xticklabels(r2_names, fontsize=9, rotation=15)
    ax.set_ylabel('R²')
    ax.set_title('(A) Variance explained by\nexpression covariates')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Panel B: Cross-fitted vs non-cross-fitted AUROC
    ax = axes[0, 1]
    comp_labels = ['Baseline\n(raw |corr|)', 'Non-CF\n|resid|', 'CF-OLS\n|resid|', 'CF-GBDT\n|resid|']
    comp_vals = [baseline_auroc_abs, auroc_nocf_abs,
                 ols_full['overall_auroc_abs_residual'],
                 gbdt_cf['overall_auroc_abs_residual']]
    colors2 = ['#2196F3', '#FF9800', '#F44336', '#9C27B0']
    bars2 = ax.bar(range(len(comp_labels)), comp_vals, color=colors2, alpha=0.8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xticks(range(len(comp_labels)))
    ax.set_xticklabels(comp_labels, fontsize=8)
    ax.set_ylabel('AUROC vs TRRUST')
    ax.set_title('(B) Residualization impact\n(cross-fitted vs not)')
    ax.set_ylim(0.0, max(comp_vals) + 0.1)
    for bar, val in zip(bars2, comp_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Panel C: Stability across seeds
    ax = axes[0, 2]
    ax.hist(stability_aurocs, bins=8, color='#FF5722', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(stability_aurocs), color='red', linestyle='--',
               label=f'Mean: {np.mean(stability_aurocs):.3f}')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Cross-fitted AUROC(|resid|)')
    ax.set_ylabel('Count')
    ax.set_title(f'(C) Stability across 10 seeds\n(mean={np.mean(stability_aurocs):.3f}±{np.std(stability_aurocs):.3f})')
    ax.legend(fontsize=8)

    # Panel D: Signed vs absolute across covariate sets
    ax = axes[1, 0]
    methods = ['OLS full\n(abs)', 'OLS full\n(signed)', 'OLS min\n(abs)', 'OLS min\n(signed)',
               'GBDT full\n(abs)']
    vals = [
        cf_results['ols_full']['overall_auroc_abs_residual'],
        cf_results['signed_ols_full']['overall_auroc_abs_residual'],
        cf_results['ols_minimal']['overall_auroc_abs_residual'],
        cf_results['signed_ols_minimal']['overall_auroc_abs_residual'],
        cf_results['gbdt_full']['overall_auroc_abs_residual'],
    ]
    colors3 = ['#2196F3', '#4CAF50', '#FF9800', '#FFC107', '#9C27B0']
    bars3 = ax.bar(range(len(methods)), vals, color=colors3, alpha=0.8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylabel('Cross-fitted AUROC(|resid|)')
    ax.set_title('(D) Robustness: signed vs abs,\ncovariate sets, model types')
    ax.set_ylim(0.0, max(vals) + 0.1)
    for bar, val in zip(bars3, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Panel E: Edge score vs expression product scatter
    ax = axes[1, 1]
    vis_idx = np.random.choice(len(df), min(5000, len(df)), replace=False)
    vis_df = df.iloc[vis_idx]
    ax.scatter(vis_df['expr_product'], vis_df['abs_corr'],
               c=['#F44336' if x else '#BBDEFB' for x in vis_df['is_trrust']],
               alpha=0.15, s=5, rasterized=True)
    # Overlay TRRUST positives
    pos_df = df[df['is_trrust'] == 1]
    ax.scatter(pos_df['expr_product'], pos_df['abs_corr'],
               c='red', s=20, alpha=0.8, zorder=5, label=f'TRRUST (n={n_pos})')
    ax.set_xlabel('Mean expression product (TF × target)')
    ax.set_ylabel('Edge score (|Spearman ρ|)')
    ax.set_title(f'(E) Edge score vs expression\n(ρ={rho_expr:.3f}, R²={r2_results["minimal"]:.3f})')
    ax.legend(fontsize=8)

    # Panel F: Summary table
    ax = axes[1, 2]
    ax.axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['N pairs', f'{len(df):,}'],
        ['N positives (TRRUST)', f'{n_pos}'],
        ['ρ(edge, expr_product)', f'{rho_expr:.3f}'],
        ['R²(full OLS)', f'{r2_results["full"]:.3f}'],
        ['R²(GBDT)', f'{r2_gbdt:.3f}'],
        ['Baseline AUROC', f'{baseline_auroc_abs:.3f}'],
        ['CF AUROC(|resid|) OLS', f'{ols_full["overall_auroc_abs_residual"]:.3f}'],
        ['CF AUROC(|resid|) GBDT', f'{gbdt_cf["overall_auroc_abs_residual"]:.3f}'],
        ['Seed stability (σ)', f'{np.std(stability_aurocs):.4f}'],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                     colWidths=[0.55, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    # Header row bold
    for j in range(2):
        table[0, j].set_text_props(fontweight='bold')
        table[0, j].set_facecolor('#E3F2FD')
    ax.set_title('(F) Summary Statistics')

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_residualization_robust.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Figure saved to {fig_path}")

    # ========================================
    # SAVE COMPREHENSIVE RESULTS
    # ========================================
    summary = {
        'experiment': 'attention_residualization_v2',
        'dataset': 'tabula_sapiens_immune_20k',
        'n_cells': n_cells,
        'n_genes': n_genes,
        'n_pairs': len(df),
        'n_trrust_positives': int(n_pos),
        'positive_rate': float(n_pos / len(df)),
        'baseline': {
            'auroc_abs_corr': round(float(baseline_auroc_abs), 4),
            'auroc_signed_corr': round(float(baseline_auroc_signed), 4),
            'rho_edge_vs_expr': round(float(rho_expr), 4),
        },
        'r_squared': {
            'ols_full': round(float(r2_results['full']), 4),
            'ols_expr_only': round(float(r2_results['expr_only']), 4),
            'ols_minimal': round(float(r2_results['minimal']), 4),
            'gbdt_full': round(float(r2_gbdt), 4),
        },
        'cross_fitted_ols_full': {
            'auroc_abs_residual': round(float(ols_full['overall_auroc_abs_residual']), 4),
            'auroc_signed_residual': round(float(ols_full['overall_auroc_signed_residual']), 4),
            'auroc_negated_residual': round(float(ols_full['overall_auroc_negated_residual']), 4),
        },
        'cross_fitted_gbdt_full': {
            'auroc_abs_residual': round(float(gbdt_cf['overall_auroc_abs_residual']), 4),
            'auroc_signed_residual': round(float(gbdt_cf['overall_auroc_signed_residual']), 4),
            'auroc_negated_residual': round(float(gbdt_cf['overall_auroc_negated_residual']), 4),
        },
        'non_cross_fitted': {
            'auroc_abs_residual': round(float(auroc_nocf_abs), 4),
            'auroc_signed_residual': round(float(auroc_nocf_signed), 4),
            'auroc_negated_residual': round(float(auroc_nocf_neg), 4),
        },
        'stability_10seeds': {
            'mean': round(float(np.mean(stability_aurocs)), 4),
            'std': round(float(np.std(stability_aurocs)), 4),
            'min': round(float(min(stability_aurocs)), 4),
            'max': round(float(max(stability_aurocs)), 4),
        },
        'signed_vs_abs': {
            'signed_ols_full_auroc': round(float(cf_results['signed_ols_full']['overall_auroc_abs_residual']), 4),
            'signed_ols_minimal_auroc': round(float(cf_results['signed_ols_minimal']['overall_auroc_abs_residual']), 4),
        }
    }

    out_path = os.path.join(OUT_DIR, "residualization_v2_summary.json")
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print("\n=== COMPREHENSIVE SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== EXPERIMENT COMPLETE ===")
    return summary


if __name__ == '__main__':
    summary = run_analysis()
