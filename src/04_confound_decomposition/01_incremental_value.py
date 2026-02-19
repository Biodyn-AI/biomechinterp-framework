#!/usr/bin/env python3
"""
Round 9 Experiment 2: Conditional Incremental-Value Test
=========================================================
Tests whether pairwise edge scores (attention/correlation) add predictive
value for DE target identification beyond gene-level features alone.

Addresses: MC3 ("incremental edge-specific value not quantified conditionally")
           ST1 ("conditional incremental-value test")
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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from multiprocessing import Pool, cpu_count

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
EXTRACTION_DIR = os.path.join(PAPER_DIR, "experiments", "round9_unified_extraction")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round9_incremental_value")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PRIMARY_LAYER = 12  # L13 (0-indexed)
N_FOLDS = 5
N_BOOTSTRAP = 100


def build_dataset(attn_edges, corr_matrix, de_labels, gene_features_df,
                  hvg_genes, hvg_to_local):
    """Build the full dataset: one row per (perturbation, target) pair."""
    hvg_upper = [g.upper() for g in hvg_genes]

    # Gene-level feature lookup
    gene_feat_dict = {}
    for _, row in gene_features_df.iterrows():
        gene_feat_dict[row['gene_upper']] = {
            'mean_expr': row['mean_expr'],
            'variance': row['variance'],
            'dropout_rate': row['dropout_rate'],
            'n_perturbations_affected': row.get('n_perturbations_affected', 0),
        }

    rows = []
    pert_genes = sorted(de_labels.keys())

    for pg in pert_genes:
        targets = de_labels[pg]
        pg_local = hvg_to_local.get(pg.upper())
        if pg_local is None:
            continue

        for tgt_gene, is_de in targets.items():
            tgt_local = hvg_to_local.get(tgt_gene)
            if tgt_local is None:
                continue
            tgt_orig = hvg_genes[tgt_local]

            # Attention edge score
            fwd = attn_edges.get((pg, tgt_orig), 0.0)
            rev = attn_edges.get((tgt_orig, pg), 0.0)
            attn_score = max(fwd, rev)

            # Correlation edge score
            corr_score = abs(corr_matrix[pg_local, tgt_local])

            # Gene-level features of target
            tgt_feats = gene_feat_dict.get(tgt_gene, {})

            rows.append({
                'perturbation': pg,
                'target': tgt_gene,
                'is_de': int(is_de),
                'attn_edge': attn_score,
                'corr_edge': corr_score,
                'tgt_mean_expr': tgt_feats.get('mean_expr', 0),
                'tgt_variance': tgt_feats.get('variance', 0),
                'tgt_dropout': tgt_feats.get('dropout_rate', 0),
                'tgt_n_affected': tgt_feats.get('n_perturbations_affected', 0),
            })

    df = pd.DataFrame(rows)
    print(f"  Dataset: {len(df)} rows, {df['perturbation'].nunique()} perturbations, "
          f"{df['is_de'].sum()} positive ({100*df['is_de'].mean():.1f}%)")
    return df


def evaluate_model_cv(df, feature_cols, n_folds=5):
    """5-fold CV over perturbations (GroupKFold), return AUROC and AUPRC."""
    X = df[feature_cols].values
    y = df['is_de'].values
    groups = df['perturbation'].values

    # Unique perturbation groups
    unique_perts = np.unique(groups)
    n_groups = len(unique_perts)

    if n_groups < n_folds:
        n_folds = n_groups

    gkf = GroupKFold(n_splits=n_folds)
    aurocs = []
    auprcs = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            continue
        if y_test.sum() == 0 or y_test.sum() == len(y_test):
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
        model.fit(X_train_s, y_train)

        probs = model.predict_proba(X_test_s)[:, 1]
        aurocs.append(roc_auc_score(y_test, probs))
        auprcs.append(average_precision_score(y_test, probs))

    return np.array(aurocs), np.array(auprcs)


def _bootstrap_worker(args):
    """Worker for bootstrap parallelization."""
    seed, df, model_configs, n_folds = args
    rng = np.random.RandomState(seed)
    unique_perts = df['perturbation'].unique()
    boot_perts = rng.choice(unique_perts, size=len(unique_perts), replace=True)

    # Build bootstrap sample
    boot_dfs = []
    for i, pg in enumerate(boot_perts):
        sub = df[df['perturbation'] == pg].copy()
        sub['perturbation'] = f"{pg}_{i}"  # unique group names
        boot_dfs.append(sub)
    boot_df = pd.concat(boot_dfs, ignore_index=True)

    results = {}
    for name, feature_cols in model_configs.items():
        aurocs, auprcs = evaluate_model_cv(boot_df, feature_cols, n_folds)
        results[name] = {
            'auroc': float(aurocs.mean()) if len(aurocs) > 0 else 0.5,
            'auprc': float(auprcs.mean()) if len(auprcs) > 0 else 0.0,
        }
    return results


def run():
    print("=" * 70)
    print("CONDITIONAL INCREMENTAL-VALUE TEST")
    print("=" * 70)

    # Load data
    print("\nLoading extraction data...")
    with open(os.path.join(EXTRACTION_DIR, "hvg_gene_list.json")) as f:
        hvg_genes = json.load(f)
    hvg_to_local = {g.upper(): i for i, g in enumerate(hvg_genes)}

    with open(os.path.join(EXTRACTION_DIR, "de_labels.pkl"), 'rb') as f:
        de_labels = pickle.load(f)

    corr_matrix = np.load(os.path.join(EXTRACTION_DIR, "correlation_matrix.npy"))
    gene_features_df = pd.read_csv(os.path.join(EXTRACTION_DIR, "gene_features.csv"))

    edge_path = os.path.join(EXTRACTION_DIR, "layer_mean_edges", f"layer_{PRIMARY_LAYER:02d}_edges.pkl")
    with open(edge_path, 'rb') as f:
        attn_edges = pickle.load(f)

    print(f"  {len(de_labels)} evaluable perturbations")
    print(f"  L{PRIMARY_LAYER} attention: {len(attn_edges)} edges")

    # Build dataset
    print("\nBuilding dataset...")
    df = build_dataset(attn_edges, corr_matrix, de_labels, gene_features_df,
                       hvg_genes, hvg_to_local)

    # Define models
    gene_features = ['tgt_mean_expr', 'tgt_variance', 'tgt_dropout', 'tgt_n_affected']
    model_configs = {
        'A_gene_only': gene_features,
        'B_attn_only': ['attn_edge'],
        'C_corr_only': ['corr_edge'],
        'D_gene_plus_attn': gene_features + ['attn_edge'],
        'E_gene_plus_corr': gene_features + ['corr_edge'],
    }

    # ============================================================
    # Cross-validated evaluation
    # ============================================================
    print(f"\n{N_FOLDS}-fold GroupKFold evaluation...")
    cv_results = {}

    for name, feature_cols in model_configs.items():
        aurocs, auprcs = evaluate_model_cv(df, feature_cols, N_FOLDS)
        cv_results[name] = {
            'mean_auroc': round(float(aurocs.mean()), 4),
            'std_auroc': round(float(aurocs.std()), 4),
            'mean_auprc': round(float(auprcs.mean()), 4),
            'std_auprc': round(float(auprcs.std()), 4),
            'per_fold_auroc': [round(float(a), 4) for a in aurocs],
        }
        print(f"  {name:20s}: AUROC = {aurocs.mean():.4f} ± {aurocs.std():.4f}, "
              f"AUPRC = {auprcs.mean():.4f} ± {auprcs.std():.4f}")

    # Compute deltas
    delta_attn = cv_results['D_gene_plus_attn']['mean_auroc'] - cv_results['A_gene_only']['mean_auroc']
    delta_corr = cv_results['E_gene_plus_corr']['mean_auroc'] - cv_results['A_gene_only']['mean_auroc']
    print(f"\n  ΔAUROC (gene+attn vs gene): {delta_attn:+.4f}")
    print(f"  ΔAUROC (gene+corr vs gene): {delta_corr:+.4f}")

    # ============================================================
    # Bootstrap CIs (parallelized)
    # ============================================================
    print(f"\nBootstrap CIs ({N_BOOTSTRAP} iterations, sequential)...")

    bootstrap_results = []
    for seed in range(N_BOOTSTRAP):
        br = _bootstrap_worker((seed, df, model_configs, N_FOLDS))
        bootstrap_results.append(br)
        if (seed + 1) % 20 == 0:
            print(f"  Bootstrap {seed+1}/{N_BOOTSTRAP}")

    # Aggregate bootstrap
    boot_aurocs = {name: [] for name in model_configs}
    for br in bootstrap_results:
        for name in model_configs:
            boot_aurocs[name].append(br[name]['auroc'])

    boot_deltas_attn = np.array(boot_aurocs['D_gene_plus_attn']) - np.array(boot_aurocs['A_gene_only'])
    boot_deltas_corr = np.array(boot_aurocs['E_gene_plus_corr']) - np.array(boot_aurocs['A_gene_only'])

    bootstrap_summary = {}
    for name in model_configs:
        vals = np.array(boot_aurocs[name])
        bootstrap_summary[name] = {
            'mean': round(float(vals.mean()), 4),
            'ci_95': [round(float(np.percentile(vals, 2.5)), 4),
                      round(float(np.percentile(vals, 97.5)), 4)],
        }

    print(f"\n  Bootstrap 95% CIs:")
    for name, bs in bootstrap_summary.items():
        print(f"    {name:20s}: [{bs['ci_95'][0]:.4f}, {bs['ci_95'][1]:.4f}]")

    print(f"\n  ΔAUROC (gene+attn - gene) 95% CI: "
          f"[{np.percentile(boot_deltas_attn, 2.5):.4f}, "
          f"{np.percentile(boot_deltas_attn, 97.5):.4f}]")
    print(f"  ΔAUROC (gene+corr - gene) 95% CI: "
          f"[{np.percentile(boot_deltas_corr, 2.5):.4f}, "
          f"{np.percentile(boot_deltas_corr, 97.5):.4f}]")

    # ============================================================
    # TF vs non-TF stratification
    # ============================================================
    print("\nTF vs non-TF stratification...")
    # Load TRRUST to identify TF perturbations
    trrust_path = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv"
    trrust_df = pd.read_csv(trrust_path, sep='\t', header=None, names=['tf', 'target', 'mode', 'pmid'])
    trrust_tfs = set(trrust_df['tf'].str.upper())

    unique_perts = df['perturbation'].unique()
    tf_perts = [pg for pg in unique_perts if pg.upper() in trrust_tfs]
    nontf_perts = [pg for pg in unique_perts if pg.upper() not in trrust_tfs]
    print(f"  TF perturbations: {len(tf_perts)}, non-TF: {len(nontf_perts)}")

    stratified_results = {}
    for stratum_name, stratum_perts in [('TF', tf_perts), ('non-TF', nontf_perts)]:
        if len(stratum_perts) < N_FOLDS:
            print(f"  Skipping {stratum_name} (only {len(stratum_perts)} perturbations)")
            continue
        sub_df = df[df['perturbation'].isin(stratum_perts)]
        stratum_cv = {}
        for name, feature_cols in model_configs.items():
            aurocs, auprcs = evaluate_model_cv(sub_df, feature_cols, min(N_FOLDS, len(stratum_perts)))
            stratum_cv[name] = round(float(aurocs.mean()), 4) if len(aurocs) > 0 else None
        stratified_results[stratum_name] = stratum_cv
        print(f"  {stratum_name}: gene_only={stratum_cv.get('A_gene_only')}, "
              f"gene+attn={stratum_cv.get('D_gene_plus_attn')}, "
              f"gene+corr={stratum_cv.get('E_gene_plus_corr')}")

    # ============================================================
    # Save results
    # ============================================================
    results = {
        'experiment': 'round9_incremental_value',
        'primary_layer': PRIMARY_LAYER,
        'n_perturbations': int(df['perturbation'].nunique()),
        'n_observations': len(df),
        'positive_rate': round(float(df['is_de'].mean()), 4),
        'cv_results': cv_results,
        'delta_auroc': {
            'gene_plus_attn_minus_gene': round(float(delta_attn), 4),
            'gene_plus_corr_minus_gene': round(float(delta_corr), 4),
        },
        'bootstrap_summary': bootstrap_summary,
        'bootstrap_delta_ci': {
            'attn': [round(float(np.percentile(boot_deltas_attn, 2.5)), 4),
                     round(float(np.percentile(boot_deltas_attn, 97.5)), 4)],
            'corr': [round(float(np.percentile(boot_deltas_corr, 2.5)), 4),
                     round(float(np.percentile(boot_deltas_corr, 97.5)), 4)],
        },
        'stratified_results': stratified_results,
    }

    with open(os.path.join(OUT_DIR, "incremental_value_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # ============================================================
    # Generate figure
    # ============================================================
    print("\nGenerating figure...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Bar chart of CV AUROCs
    ax = axes[0]
    names = list(cv_results.keys())
    means = [cv_results[n]['mean_auroc'] for n in names]
    stds = [cv_results[n]['std_auroc'] for n in names]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#E91E63']
    short_names = ['Gene\nonly', 'Attn\nonly', 'Corr\nonly', 'Gene+\nAttn', 'Gene+\nCorr']
    ax.bar(range(len(names)), means, yerr=stds, color=colors, alpha=0.8, capsize=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel('CV AUROC')
    ax.set_title('(A) Model comparison\n(5-fold GroupKFold)')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

    # Panel B: Bootstrap ΔAUROC distribution
    ax = axes[1]
    ax.hist(boot_deltas_attn, bins=30, alpha=0.6, color='#9C27B0', label='Gene+Attn - Gene')
    ax.hist(boot_deltas_corr, bins=30, alpha=0.6, color='#E91E63', label='Gene+Corr - Gene')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('ΔAUROC')
    ax.set_ylabel('Count')
    ax.set_title(f'(B) Bootstrap ΔAUROC distribution\n(n={N_BOOTSTRAP})')
    ax.legend(fontsize=8)

    # Panel C: Stratified results
    ax = axes[2]
    if stratified_results:
        strata = list(stratified_results.keys())
        x = np.arange(len(strata))
        width = 0.25
        for i, (model_name, color) in enumerate([
            ('A_gene_only', '#4CAF50'), ('D_gene_plus_attn', '#9C27B0'),
            ('E_gene_plus_corr', '#E91E63')
        ]):
            vals = [stratified_results[s].get(model_name, 0.5) or 0.5 for s in strata]
            ax.bar(x + i * width, vals, width, color=color, alpha=0.8,
                   label=model_name.replace('_', ' '))
        ax.set_xticks(x + width)
        ax.set_xticklabels(strata)
        ax.set_ylabel('CV AUROC')
        ax.set_title('(C) TF vs non-TF stratification')
        ax.legend(fontsize=7)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_incremental_value.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Figure saved to {fig_path}")

    print(f"\n{'=' * 70}")
    print("INCREMENTAL-VALUE TEST COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nKey result: ΔAUROC (gene+attn - gene) = {delta_attn:+.4f}")
    print(f"Key result: ΔAUROC (gene+corr - gene) = {delta_corr:+.4f}")


if __name__ == '__main__':
    run()
