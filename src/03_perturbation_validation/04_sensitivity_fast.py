#!/usr/bin/env python3
"""
Perturbation-First Sensitivity Analysis (Optimized)
====================================================
Addresses Round 6 Reviewer Concern 4:
  - Vary control-cell count: N ∈ {500, 2000, 10000}
  - Vary gene universe: HVG ∈ {1000, 2000, 5000}
  - Vary DE threshold
  - Report sensitivity of AUROC to these parameters
  - Clarify TF vs non-TF terminology
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)  # Line-buffered stdout

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata, ttest_ind
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
DATA_DIR = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "perturbation_sensitivity_results")
os.makedirs(OUT_DIR, exist_ok=True)


def compute_correlation_edges(X_ctrl):
    """Compute Spearman correlation matrix from control cells."""
    ranks = np.apply_along_axis(rankdata, 0, X_ctrl)
    corr_matrix = np.corrcoef(ranks.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    return corr_matrix


def call_de_fast(X_ctrl, X_pert, n_genes):
    """Vectorized DE calling: return pvals and LFCs for all genes."""
    pvals = np.ones(n_genes)
    lfcs = np.zeros(n_genes)

    for j in range(n_genes):
        ctrl_vals = X_ctrl[:, j]
        pert_vals = X_pert[:, j]
        lfcs[j] = pert_vals.mean() - ctrl_vals.mean()
        if ctrl_vals.std() > 0 or pert_vals.std() > 0:
            _, pvals[j] = ttest_ind(pert_vals, ctrl_vals, equal_var=False)

    return pvals, lfcs


def run_sensitivity():
    import scanpy as sc

    print("=" * 60)
    print("PERTURBATION-FIRST SENSITIVITY ANALYSIS (optimized)")
    print("=" * 60)

    # Load Replogle data
    print("\nLoading Replogle CRISPRi data...")
    adata = sc.read_h5ad(os.path.join(DATA_DIR, "replogle_concat.h5ad"))
    print(f"  Shape: {adata.shape}")

    # Identify control and perturbed cells
    pert_col = 'gene'
    ctrl_mask = adata.obs[pert_col] == 'non-targeting'
    print(f"  Control cells: {ctrl_mask.sum()}")
    print(f"  Perturbed cells: {(~ctrl_mask).sum()}")

    # Pre-normalize ALL cells once (in-place)
    print("  Normalizing all cells...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("  Done normalizing.")

    # Get perturbed gene list
    perturbed_obs = adata.obs[~ctrl_mask]
    pert_genes = perturbed_obs[pert_col].unique().tolist()
    print(f"  Unique perturbed genes: {len(pert_genes)}")

    # Pre-compute per-perturbation cell indices (ONCE)
    print("  Pre-computing perturbation cell indices...")
    pert_cell_indices = {}
    pert_cell_counts = {}
    for pg in pert_genes:
        idx = np.where(adata.obs[pert_col].values == pg)[0]
        if len(idx) >= 10:
            pert_cell_indices[pg] = idx
            pert_cell_counts[pg] = len(idx)
    print(f"  Perturbations with >= 10 cells: {len(pert_cell_indices)}")

    # Identify TFs
    trrust_path = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv"
    trrust_df = pd.read_csv(trrust_path, sep='\t', header=None, names=['tf', 'target', 'mode', 'pmid'])
    known_tfs = set(trrust_df['tf'].str.upper())
    n_tf_perts = sum(1 for g in pert_cell_indices if g.upper() in known_tfs)
    print(f"  Known TFs in evaluable set: {n_tf_perts}")
    print(f"  Non-TF: {len(pert_cell_indices) - n_tf_perts}")

    # Pre-compute control cell indices
    ctrl_indices = np.where(ctrl_mask.values)[0]
    print(f"  Total control cells: {len(ctrl_indices)}")

    # Sensitivity grid
    n_control_values = [500, 2000, 10000]
    n_hvg_values = [1000, 2000, 5000]
    lfc_thresholds = [0.25, 0.5, 1.0]

    all_results = []

    for n_ctrl in n_control_values:
        if n_ctrl > len(ctrl_indices):
            print(f"\n  Skipping N_ctrl={n_ctrl} (only {len(ctrl_indices)} controls)")
            continue

        print(f"\n{'='*40}")
        print(f"N_control = {n_ctrl}")

        np.random.seed(42)
        ctrl_idx = np.random.choice(ctrl_indices, n_ctrl, replace=False)

        for n_hvg in n_hvg_values:
            print(f"\n  N_HVG = {n_hvg}")

            # Select HVGs from control cells
            ctrl_sub = adata[ctrl_idx].copy()
            sc.pp.highly_variable_genes(ctrl_sub, n_top_genes=min(n_hvg, ctrl_sub.n_vars))
            hvg_genes = list(ctrl_sub.var_names[ctrl_sub.var['highly_variable']])
            hvg_set = set(hvg_genes)
            n_genes = len(hvg_genes)
            gene_upper = [g.upper() for g in hvg_genes]
            gene_upper_set = set(gene_upper)
            gene_to_idx = {g.upper(): i for i, g in enumerate(hvg_genes)}

            # Control expression matrix (HVG only)
            ctrl_hvg = adata[ctrl_idx][:, hvg_genes]
            X_ctrl = ctrl_hvg.X
            if hasattr(X_ctrl, 'toarray'):
                X_ctrl = X_ctrl.toarray()
            X_ctrl = np.asarray(X_ctrl, dtype=np.float64)

            # Correlation edges
            corr_matrix = compute_correlation_edges(X_ctrl)

            # Pre-filter perturbation genes that are in this HVG set
            evaluable_perts = [pg for pg in pert_cell_indices
                               if pg.upper() in gene_upper_set]
            print(f"    Perturbation genes in HVG set: {len(evaluable_perts)}")

            # Pre-extract perturbed expression for evaluable genes
            # and compute DE pvals/LFCs ONCE per perturbation
            # (LFC threshold applied separately)
            de_cache = {}
            for pg in evaluable_perts:
                pidx = pert_cell_indices[pg]
                pert_hvg = adata[pidx][:, hvg_genes]
                X_pert = pert_hvg.X
                if hasattr(X_pert, 'toarray'):
                    X_pert = X_pert.toarray()
                X_pert = np.asarray(X_pert, dtype=np.float64)

                pvals, lfcs = call_de_fast(X_ctrl, X_pert, n_genes)
                reject, adj_pvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
                de_cache[pg] = (reject, adj_pvals, lfcs)

            print(f"    DE cache computed for {len(de_cache)} perturbations")

            for lfc_thresh in lfc_thresholds:
                print(f"    LFC threshold = {lfc_thresh}")

                aurocs = []
                aurocs_tf = []
                aurocs_nontf = []
                n_evaluated = 0
                n_skipped_no_de = 0

                for pg in evaluable_perts:
                    reject, adj_pvals, lfcs = de_cache[pg]

                    # DE targets at this LFC threshold
                    de_targets = set()
                    for j in range(n_genes):
                        if reject[j] and abs(lfcs[j]) >= lfc_thresh:
                            de_targets.add(hvg_genes[j].upper())

                    if len(de_targets) < 5:
                        n_skipped_no_de += 1
                        continue

                    # Evaluate AUROC
                    pg_upper = pg.upper()
                    pert_idx = gene_to_idx.get(pg_upper)
                    if pert_idx is None:
                        continue

                    edge_scores = np.abs(corr_matrix[pert_idx])
                    labels = np.array([1 if gene_upper[j] in de_targets else 0
                                       for j in range(n_genes)])
                    mask = np.ones(n_genes, dtype=bool)
                    mask[pert_idx] = False
                    scores = edge_scores[mask]
                    labs = labels[mask]

                    if labs.sum() == 0 or labs.sum() == len(labs):
                        continue

                    auroc = roc_auc_score(labs, scores)
                    aurocs.append(auroc)
                    n_evaluated += 1
                    is_tf = pg_upper in known_tfs
                    if is_tf:
                        aurocs_tf.append(auroc)
                    else:
                        aurocs_nontf.append(auroc)

                    if n_evaluated <= 5:
                        tf_label = "TF" if is_tf else "non-TF"
                        print(f"      {pg} ({tf_label}): {len(de_targets)} DE, AUROC={auroc:.3f}")

                if len(aurocs) > 0:
                    mean_auroc = np.mean(aurocs)
                    t_stat, t_p = stats.ttest_1samp(aurocs, 0.5) if len(aurocs) > 1 else (0, 1)
                    print(f"    -> Evaluated: {n_evaluated}, Mean AUROC: {mean_auroc:.4f}, "
                          f"p(vs 0.5): {t_p:.4f}")
                    if aurocs_tf:
                        print(f"       TF perts: {len(aurocs_tf)}, mean={np.mean(aurocs_tf):.4f}")
                    if aurocs_nontf:
                        print(f"       Non-TF: {len(aurocs_nontf)}, mean={np.mean(aurocs_nontf):.4f}")

                    result = {
                        'n_control': n_ctrl,
                        'n_hvg': n_hvg,
                        'lfc_threshold': lfc_thresh,
                        'n_evaluated': n_evaluated,
                        'n_tf_evaluated': len(aurocs_tf),
                        'n_nontf_evaluated': len(aurocs_nontf),
                        'mean_auroc': round(float(mean_auroc), 4),
                        'std_auroc': round(float(np.std(aurocs)), 4),
                        'median_auroc': round(float(np.median(aurocs)), 4),
                        'ttest_p': round(float(t_p), 6),
                        'mean_auroc_tf': round(float(np.mean(aurocs_tf)), 4) if aurocs_tf else None,
                        'mean_auroc_nontf': round(float(np.mean(aurocs_nontf)), 4) if aurocs_nontf else None,
                        'n_skipped_no_de': n_skipped_no_de,
                    }
                    all_results.append(result)
                else:
                    print(f"    -> No evaluable perturbations")

    # ========================================
    # FIGURES
    # ========================================
    print("\n\nGenerating figures...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUT_DIR, "sensitivity_results.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: AUROC vs N_control (at fixed HVG=1000, LFC=0.5)
    ax = axes[0]
    subset = results_df[(results_df['n_hvg'] == 1000) & (results_df['lfc_threshold'] == 0.5)]
    if len(subset) > 0:
        ax.plot(subset['n_control'], subset['mean_auroc'], 'o-', color='#2196F3', markersize=8)
        ax.fill_between(subset['n_control'],
                        subset['mean_auroc'] - subset['std_auroc'],
                        subset['mean_auroc'] + subset['std_auroc'],
                        alpha=0.2, color='#2196F3')
        for _, row in subset.iterrows():
            ax.annotate(f"n={row['n_evaluated']}", (row['n_control'], row['mean_auroc']),
                       textcoords="offset points", xytext=(5, 10), fontsize=8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of control cells')
    ax.set_ylabel('Mean AUROC vs DE targets')
    ax.set_title('(A) Sensitivity to N_control\n(HVG=1000, LFC=0.5)')

    # Panel B: AUROC vs N_HVG (at fixed N_ctrl=500, LFC=0.5)
    ax = axes[1]
    subset = results_df[(results_df['n_control'] == 500) & (results_df['lfc_threshold'] == 0.5)]
    if len(subset) > 0:
        ax.plot(subset['n_hvg'], subset['mean_auroc'], 'o-', color='#4CAF50', markersize=8)
        ax.fill_between(subset['n_hvg'],
                        subset['mean_auroc'] - subset['std_auroc'],
                        subset['mean_auroc'] + subset['std_auroc'],
                        alpha=0.2, color='#4CAF50')
        for _, row in subset.iterrows():
            ax.annotate(f"n={row['n_evaluated']}", (row['n_hvg'], row['mean_auroc']),
                       textcoords="offset points", xytext=(5, 10), fontsize=8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of highly variable genes')
    ax.set_ylabel('Mean AUROC vs DE targets')
    ax.set_title('(B) Sensitivity to gene universe\n(N_ctrl=500, LFC=0.5)')

    # Panel C: AUROC vs LFC threshold (at fixed N_ctrl=500, HVG=1000)
    ax = axes[2]
    subset = results_df[(results_df['n_control'] == 500) & (results_df['n_hvg'] == 1000)]
    if len(subset) > 0:
        ax.plot(subset['lfc_threshold'], subset['mean_auroc'], 'o-', color='#FF9800', markersize=8)
        ax.fill_between(subset['lfc_threshold'],
                        subset['mean_auroc'] - subset['std_auroc'],
                        subset['mean_auroc'] + subset['std_auroc'],
                        alpha=0.2, color='#FF9800')
        for _, row in subset.iterrows():
            ax.annotate(f"n={row['n_evaluated']}", (row['lfc_threshold'], row['mean_auroc']),
                       textcoords="offset points", xytext=(5, 10), fontsize=8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('LFC threshold')
    ax.set_ylabel('Mean AUROC vs DE targets')
    ax.set_title('(C) Sensitivity to DE threshold\n(N_ctrl=500, HVG=1000)')

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_perturbation_sensitivity.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Figure saved to {fig_path}")

    # Save summary
    summary = {
        'experiment': 'perturbation_sensitivity',
        'description': 'Sensitivity of perturbation-first AUROC to control cells, gene universe, and DE threshold',
        'results': all_results,
    }

    with open(os.path.join(OUT_DIR, "sensitivity_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== EXPERIMENT COMPLETE ===")
    return summary


if __name__ == '__main__':
    run_sensitivity()
