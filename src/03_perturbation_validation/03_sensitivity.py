#!/usr/bin/env python3
"""
Perturbation-First Sensitivity Analysis
=========================================
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

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
DATA_DIR = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "perturbation_sensitivity_results")
os.makedirs(OUT_DIR, exist_ok=True)


def compute_correlation_edges(X_ctrl, gene_names):
    """Compute |Spearman correlation| edge scores from control cells."""
    ranks = np.apply_along_axis(rankdata, 0, X_ctrl)
    corr_matrix = np.corrcoef(ranks.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    return corr_matrix


def call_de_targets(X_ctrl, X_pert, gene_names, method='ttest', fdr_threshold=0.05, lfc_threshold=0.5):
    """Call DE targets for a perturbation."""
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    n_genes = len(gene_names)
    pvals = []
    lfcs = []

    ctrl_means = X_ctrl.mean(axis=0)

    for j in range(n_genes):
        ctrl_vals = X_ctrl[:, j]
        pert_vals = X_pert[:, j]

        # Log fold change
        ctrl_mean = ctrl_vals.mean()
        pert_mean = pert_vals.mean()
        lfc = pert_mean - ctrl_mean  # already log-space

        if method == 'ttest':
            from scipy.stats import ttest_ind
            t, p = ttest_ind(pert_vals, ctrl_vals, equal_var=False)
        elif method == 'mannwhitney':
            try:
                _, p = mannwhitneyu(pert_vals, ctrl_vals, alternative='two-sided')
            except ValueError:
                p = 1.0
        else:
            raise ValueError(f"Unknown method: {method}")

        pvals.append(p)
        lfcs.append(lfc)

    pvals = np.array(pvals)
    lfcs = np.array(lfcs)

    # FDR correction
    reject, adj_pvals, _, _ = multipletests(pvals, alpha=fdr_threshold, method='fdr_bh')

    # DE targets: significant AND above LFC threshold
    de_targets = set()
    for j in range(n_genes):
        if reject[j] and abs(lfcs[j]) >= lfc_threshold:
            de_targets.add(gene_names[j])

    return de_targets


def evaluate_perturbation(corr_matrix, gene_names, pert_gene, de_targets):
    """Evaluate AUROC for one perturbation."""
    gene_upper = [g.upper() for g in gene_names]
    gene_to_idx = {g: i for i, g in enumerate(gene_upper)}

    pert_upper = pert_gene.upper()
    if pert_upper not in gene_to_idx:
        return None

    pert_idx = gene_to_idx[pert_upper]
    edge_scores = np.abs(corr_matrix[pert_idx])

    # Labels: DE targets
    labels = np.array([1 if gene_names[j].upper() in de_targets else 0
                       for j in range(len(gene_names))])
    # Exclude self
    mask = np.ones(len(gene_names), dtype=bool)
    mask[pert_idx] = False

    scores = edge_scores[mask]
    labs = labels[mask]

    if labs.sum() == 0 or labs.sum() == len(labs):
        return None

    return roc_auc_score(labs, scores)


def run_sensitivity():
    import scanpy as sc

    print("=" * 60)
    print("PERTURBATION-FIRST SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Load Replogle data
    print("\nLoading Replogle CRISPRi data...")
    adata = sc.read_h5ad(os.path.join(DATA_DIR, "replogle_concat.h5ad"))
    print(f"  Shape: {adata.shape}")

    # Identify control cells
    pert_col = 'gene'
    controls = adata[adata.obs[pert_col] == 'non-targeting'].copy()
    perturbed = adata[adata.obs[pert_col] != 'non-targeting']
    print(f"  Control cells: {controls.n_obs}")
    print(f"  Perturbed cells: {perturbed.n_obs}")

    # Get perturbed gene list (unique)
    pert_genes = perturbed.obs[pert_col].unique().tolist()
    print(f"  Unique perturbed genes: {len(pert_genes)}")

    # Identify which perturbed genes are TFs (using TRRUST)
    from collections import defaultdict
    trrust_path = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv"
    trrust_df = pd.read_csv(trrust_path, sep='\t', header=None, names=['tf', 'target', 'mode', 'pmid'])
    known_tfs = set(trrust_df['tf'].str.upper())
    n_tf_perts = sum(1 for g in pert_genes if g.upper() in known_tfs)
    n_non_tf_perts = len(pert_genes) - n_tf_perts
    print(f"  Known TFs in perturbation set: {n_tf_perts}")
    print(f"  Non-TF perturbed genes: {n_non_tf_perts}")

    # Preprocess controls
    sc.pp.normalize_total(controls, target_sum=1e4)
    sc.pp.log1p(controls)

    # Sensitivity grid
    n_control_values = [500, 2000, 10000]
    n_hvg_values = [1000, 2000, 5000]
    de_methods = ['ttest']  # focus on t-test with LFC threshold
    lfc_thresholds = [0.25, 0.5, 1.0]

    all_results = []

    for n_ctrl in n_control_values:
        if n_ctrl > controls.n_obs:
            print(f"\n  Skipping N_ctrl={n_ctrl} (only {controls.n_obs} controls available)")
            continue

        print(f"\n{'='*40}")
        print(f"N_control = {n_ctrl}")

        np.random.seed(42)
        ctrl_idx = np.random.choice(controls.n_obs, n_ctrl, replace=False)
        ctrl_sub = controls[ctrl_idx].copy()

        for n_hvg in n_hvg_values:
            print(f"\n  N_HVG = {n_hvg}")

            # Select HVGs from control subsample
            sc.pp.highly_variable_genes(ctrl_sub, n_top_genes=min(n_hvg, ctrl_sub.n_vars))
            hvg_mask = ctrl_sub.var['highly_variable']
            ctrl_hvg = ctrl_sub[:, hvg_mask].copy()
            gene_names = list(ctrl_hvg.var_names)
            n_genes = len(gene_names)
            gene_upper = [g.upper() for g in gene_names]

            # Control expression matrix
            X_ctrl = ctrl_hvg.X
            if hasattr(X_ctrl, 'toarray'):
                X_ctrl = X_ctrl.toarray()
            X_ctrl = np.asarray(X_ctrl, dtype=np.float64)

            # Correlation edges
            corr_matrix = compute_correlation_edges(X_ctrl, gene_names)

            for lfc_thresh in lfc_thresholds:
                print(f"    LFC threshold = {lfc_thresh}")

                aurocs = []
                aurocs_tf = []     # TF perturbations only
                aurocs_nontf = []  # non-TF perturbations

                n_evaluated = 0
                n_skipped_not_in_genes = 0
                n_skipped_no_de = 0

                for pert_gene in pert_genes:
                    if pert_gene.upper() not in set(gene_upper):
                        n_skipped_not_in_genes += 1
                        continue

                    # Get perturbed cells for this gene
                    pert_cells = adata[adata.obs[pert_col] == pert_gene]
                    if pert_cells.n_obs < 10:
                        continue

                    # Normalize perturbed cells the same way
                    pert_proc = pert_cells.copy()
                    sc.pp.normalize_total(pert_proc, target_sum=1e4)
                    sc.pp.log1p(pert_proc)

                    # Subset to same genes
                    common_genes = [g for g in gene_names if g in pert_proc.var_names]
                    if len(common_genes) < n_genes * 0.9:
                        continue

                    pert_hvg = pert_proc[:, gene_names].copy()
                    X_pert = pert_hvg.X
                    if hasattr(X_pert, 'toarray'):
                        X_pert = X_pert.toarray()
                    X_pert = np.asarray(X_pert, dtype=np.float64)

                    # Call DE targets
                    de_targets = call_de_targets(X_ctrl, X_pert, gene_names,
                                                 method='ttest',
                                                 fdr_threshold=0.05,
                                                 lfc_threshold=lfc_thresh)

                    if len(de_targets) < 5:
                        n_skipped_no_de += 1
                        continue

                    auroc = evaluate_perturbation(corr_matrix, gene_names, pert_gene, de_targets)
                    if auroc is not None:
                        aurocs.append(auroc)
                        n_evaluated += 1
                        is_tf = pert_gene.upper() in known_tfs
                        if is_tf:
                            aurocs_tf.append(auroc)
                        else:
                            aurocs_nontf.append(auroc)

                        if n_evaluated <= 5:
                            tf_label = "TF" if is_tf else "non-TF"
                            print(f"      {pert_gene} ({tf_label}): {len(de_targets)} DE, AUROC={auroc:.3f}")

                if len(aurocs) > 0:
                    mean_auroc = np.mean(aurocs)
                    t_stat, t_p = stats.ttest_1samp(aurocs, 0.5) if len(aurocs) > 1 else (0, 1)
                    print(f"    -> Evaluated: {n_evaluated}, Mean AUROC: {mean_auroc:.4f}, "
                          f"p(vs 0.5): {t_p:.4f}")
                    print(f"       TF perts: {len(aurocs_tf)}, mean={np.mean(aurocs_tf):.4f}" if aurocs_tf else "       No TF perts evaluated")
                    print(f"       Non-TF: {len(aurocs_nontf)}, mean={np.mean(aurocs_nontf):.4f}" if aurocs_nontf else "       No non-TF perts")

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
                        'n_skipped_not_in_genes': n_skipped_not_in_genes,
                        'n_skipped_no_de': n_skipped_no_de,
                    }
                    all_results.append(result)

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
    ax.set_title('(A) Sensitivity to control cell count\n(HVG=1000, LFC≥0.5)')
    ax.set_xscale('log')

    # Panel B: AUROC vs N_HVG (at fixed N_ctrl=2000, LFC=0.5)
    ax = axes[1]
    for n_ctrl in n_control_values:
        subset = results_df[(results_df['n_control'] == n_ctrl) & (results_df['lfc_threshold'] == 0.5)]
        if len(subset) > 0:
            ax.plot(subset['n_hvg'], subset['mean_auroc'], 'o-', label=f'N_ctrl={n_ctrl}', markersize=6)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of HVGs')
    ax.set_ylabel('Mean AUROC vs DE targets')
    ax.set_title('(B) Sensitivity to gene universe\n(LFC≥0.5)')
    ax.legend(fontsize=8)

    # Panel C: AUROC vs LFC threshold (at fixed N_ctrl=2000, HVG=1000)
    ax = axes[2]
    for n_ctrl in n_control_values:
        subset = results_df[(results_df['n_control'] == n_ctrl) & (results_df['n_hvg'] == 1000)]
        if len(subset) > 0:
            ax.plot(subset['lfc_threshold'], subset['mean_auroc'], 'o-', label=f'N_ctrl={n_ctrl}', markersize=6)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('LFC threshold')
    ax.set_ylabel('Mean AUROC vs DE targets')
    ax.set_title('(C) Sensitivity to DE threshold\n(HVG=1000)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_perturbation_sensitivity.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Figure saved to {fig_path}")

    # Save summary
    summary = {
        'experiment': 'perturbation_sensitivity',
        'parameters_tested': {
            'n_control': n_control_values,
            'n_hvg': n_hvg_values,
            'lfc_thresholds': lfc_thresholds,
        },
        'n_conditions': len(all_results),
        'results': all_results,
    }

    with open(os.path.join(OUT_DIR, "sensitivity_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2))
    for r in all_results:
        print(f"  N_ctrl={r['n_control']}, HVG={r['n_hvg']}, LFC={r['lfc_threshold']}: "
              f"AUROC={r['mean_auroc']:.4f} (n={r['n_evaluated']}, p={r['ttest_p']:.4f})")
    print("\n=== EXPERIMENT COMPLETE ===")
    return summary


if __name__ == '__main__':
    run_sensitivity()
