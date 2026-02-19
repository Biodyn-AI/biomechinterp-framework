#!/usr/bin/env python3
"""
Perturbation-First Sensitivity Analysis (Vectorized + Incremental Save)
=======================================================================
Addresses Round 6 Reviewer Concern 4.
Uses fully vectorized Welch's t-test (no Python for-loop over genes).
Resumes from partial results on disk.
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
from scipy.stats import rankdata
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


def call_de_vectorized(X_ctrl, X_pert):
    """Fully vectorized Welch's t-test across all genes at once.

    No Python for-loop — uses numpy broadcasting for ~100x speedup.
    Returns pvals and LFCs arrays.
    """
    n_ctrl = X_ctrl.shape[0]
    n_pert = X_pert.shape[0]

    mean_ctrl = X_ctrl.mean(axis=0)
    mean_pert = X_pert.mean(axis=0)

    lfcs = mean_pert - mean_ctrl

    var_ctrl = X_ctrl.var(axis=0, ddof=1)
    var_pert = X_pert.var(axis=0, ddof=1)

    se_sq = var_ctrl / n_ctrl + var_pert / n_pert
    se = np.sqrt(se_sq)

    # Welch-Satterthwaite degrees of freedom
    with np.errstate(divide='ignore', invalid='ignore'):
        df_num = se_sq ** 2
        df_den = (var_ctrl / n_ctrl) ** 2 / max(n_ctrl - 1, 1) + \
                 (var_pert / n_pert) ** 2 / max(n_pert - 1, 1)
        df = np.where(df_den > 0, df_num / df_den, 1.0)
        t_stats = np.where(se > 0, lfcs / se, 0.0)

    pvals = 2.0 * stats.t.sf(np.abs(t_stats), df)
    pvals = np.where(se > 0, pvals, 1.0)
    pvals = np.nan_to_num(pvals, nan=1.0)

    return pvals, lfcs


def compute_correlation_edges(X_ctrl):
    """Compute Spearman correlation matrix from control cells."""
    ranks = np.apply_along_axis(rankdata, 0, X_ctrl)
    corr_matrix = np.corrcoef(ranks.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    return corr_matrix


def run_sensitivity():
    import scanpy as sc

    print("=" * 60)
    print("PERTURBATION-FIRST SENSITIVITY ANALYSIS (vectorized)")
    print("=" * 60)

    # Check for existing partial results
    partial_path = os.path.join(OUT_DIR, "sensitivity_results_partial.json")
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            saved = json.load(f)
        all_results = saved.get('results', [])
        completed_combos = set()
        for r in all_results:
            completed_combos.add((r['n_control'], r['n_hvg'], r['lfc_threshold']))
        print(f"  Resuming from {len(all_results)} saved results")
    else:
        all_results = []
        completed_combos = set()

    # Load Replogle data
    print("\nLoading Replogle CRISPRi data...")
    adata = sc.read_h5ad(os.path.join(DATA_DIR, "replogle_concat.h5ad"))
    print(f"  Shape: {adata.shape}")

    pert_col = 'gene'
    ctrl_mask = adata.obs[pert_col] == 'non-targeting'
    print(f"  Control cells: {ctrl_mask.sum()}")
    print(f"  Perturbed cells: {(~ctrl_mask).sum()}")

    # Pre-normalize ALL cells once
    print("  Normalizing all cells...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("  Done normalizing.")

    # Get perturbed gene list
    pert_genes = adata.obs[~ctrl_mask][pert_col].unique().tolist()
    print(f"  Unique perturbed genes: {len(pert_genes)}")

    # Pre-compute per-perturbation cell indices
    print("  Pre-computing perturbation cell indices...")
    pert_cell_indices = {}
    for pg in pert_genes:
        idx = np.where(adata.obs[pert_col].values == pg)[0]
        if len(idx) >= 10:
            pert_cell_indices[pg] = idx
    print(f"  Perturbations with >= 10 cells: {len(pert_cell_indices)}")

    # Identify TFs
    trrust_path = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv"
    trrust_df = pd.read_csv(trrust_path, sep='\t', header=None, names=['tf', 'target', 'mode', 'pmid'])
    known_tfs = set(trrust_df['tf'].str.upper())

    ctrl_indices = np.where(ctrl_mask.values)[0]
    print(f"  Total control cells: {len(ctrl_indices)}")

    # Sensitivity grid
    n_control_values = [500, 2000, 10000]
    n_hvg_values = [1000, 2000, 5000]
    lfc_thresholds = [0.25, 0.5, 1.0]

    for n_ctrl in n_control_values:
        if n_ctrl > len(ctrl_indices):
            print(f"\n  Skipping N_ctrl={n_ctrl} (only {len(ctrl_indices)} controls)")
            continue

        print(f"\n{'='*40}")
        print(f"N_control = {n_ctrl}")

        np.random.seed(42)
        ctrl_idx = np.random.choice(ctrl_indices, n_ctrl, replace=False)

        for n_hvg in n_hvg_values:
            # Check if all LFC thresholds for this combo are done
            all_lfc_done = all((n_ctrl, n_hvg, lfc) in completed_combos for lfc in lfc_thresholds)
            if all_lfc_done:
                print(f"\n  N_HVG = {n_hvg} [CACHED, skipping]")
                continue

            print(f"\n  N_HVG = {n_hvg}")

            # Select HVGs
            ctrl_sub = adata[ctrl_idx].copy()
            sc.pp.highly_variable_genes(ctrl_sub, n_top_genes=min(n_hvg, ctrl_sub.n_vars))
            hvg_genes = list(ctrl_sub.var_names[ctrl_sub.var['highly_variable']])
            n_genes = len(hvg_genes)
            gene_upper = [g.upper() for g in hvg_genes]
            gene_upper_set = set(gene_upper)
            gene_to_idx = {g.upper(): i for i, g in enumerate(hvg_genes)}

            # Control expression matrix
            print(f"    Extracting control expression matrix...")
            ctrl_hvg = adata[ctrl_idx][:, hvg_genes]
            X_ctrl = ctrl_hvg.X
            if hasattr(X_ctrl, 'toarray'):
                X_ctrl = X_ctrl.toarray()
            X_ctrl = np.asarray(X_ctrl, dtype=np.float64)

            # Correlation edges
            print(f"    Computing correlation matrix...")
            corr_matrix = compute_correlation_edges(X_ctrl)

            # Filter evaluable perturbation genes
            evaluable_perts = [pg for pg in pert_cell_indices if pg.upper() in gene_upper_set]
            print(f"    Perturbation genes in HVG set: {len(evaluable_perts)}")

            # Compute DE cache one perturbation at a time (vectorized per-gene)
            print(f"    Computing DE cache (vectorized)...")
            de_cache = {}
            for i, pg in enumerate(evaluable_perts):
                pidx = pert_cell_indices[pg]
                pert_hvg = adata[pidx][:, hvg_genes]
                X_pert = pert_hvg.X
                if hasattr(X_pert, 'toarray'):
                    X_pert = X_pert.toarray()
                X_pert = np.asarray(X_pert, dtype=np.float64)

                pvals, lfcs = call_de_vectorized(X_ctrl, X_pert)
                reject, adj_pvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
                de_cache[pg] = (reject, adj_pvals, lfcs)

                if (i + 1) % 200 == 0:
                    print(f"      ... {i+1}/{len(evaluable_perts)} perturbations done")

            print(f"    DE cache computed for {len(de_cache)} perturbations")

            for lfc_thresh in lfc_thresholds:
                if (n_ctrl, n_hvg, lfc_thresh) in completed_combos:
                    print(f"    LFC threshold = {lfc_thresh} [CACHED]")
                    continue

                print(f"    LFC threshold = {lfc_thresh}")

                aurocs = []
                aurocs_tf = []
                aurocs_nontf = []
                n_evaluated = 0
                n_skipped_no_de = 0

                for pg in evaluable_perts:
                    reject, adj_pvals, lfcs = de_cache[pg]
                    de_targets = set()
                    for j in range(n_genes):
                        if reject[j] and abs(lfcs[j]) >= lfc_thresh:
                            de_targets.add(hvg_genes[j].upper())

                    if len(de_targets) < 5:
                        n_skipped_no_de += 1
                        continue

                    pg_upper = pg.upper()
                    pert_idx = gene_to_idx.get(pg_upper)
                    if pert_idx is None:
                        continue

                    edge_scores = np.abs(corr_matrix[pert_idx])
                    labels = np.array([1 if gene_upper[j] in de_targets else 0 for j in range(n_genes)])
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
                    print(f"    -> Evaluated: {n_evaluated}, Mean AUROC: {mean_auroc:.4f}, p(vs 0.5): {t_p:.4f}")
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

            # Save partial results after each (n_ctrl, n_hvg) combo
            partial_save = {'experiment': 'perturbation_sensitivity', 'results': all_results}
            with open(partial_path, 'w') as f:
                json.dump(partial_save, f, indent=2)
            print(f"    [Saved {len(all_results)} results to disk]")

    # Generate figures
    print("\n\nGenerating figures...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUT_DIR, "sensitivity_results.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: AUROC vs N_control (at fixed HVG=2000, LFC=0.5)
    ax = axes[0]
    subset = results_df[(results_df['n_hvg'] == 2000) & (results_df['lfc_threshold'] == 0.5)]
    if len(subset) > 0:
        ax.plot(subset['n_control'], subset['mean_auroc'], 'o-', color='#2196F3', markersize=8)
        ax.fill_between(subset['n_control'],
                        subset['mean_auroc'] - subset['std_auroc'],
                        subset['mean_auroc'] + subset['std_auroc'],
                        alpha=0.2, color='#2196F3')
        for _, row in subset.iterrows():
            ax.annotate(f"n={int(row['n_evaluated'])}", (row['n_control'], row['mean_auroc']),
                       textcoords="offset points", xytext=(5, 10), fontsize=8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of control cells')
    ax.set_ylabel('Mean AUROC vs DE targets')
    ax.set_title('(A) Sensitivity to control cell count\n(HVG=2000, LFC>0.5)')
    ax.set_ylim(0.45, 0.85)

    # Panel B: AUROC vs N_HVG (at fixed N_ctrl=2000, LFC=0.5)
    ax = axes[1]
    subset = results_df[(results_df['n_control'] == 2000) & (results_df['lfc_threshold'] == 0.5)]
    if len(subset) > 0:
        ax.plot(subset['n_hvg'], subset['mean_auroc'], 'o-', color='#4CAF50', markersize=8)
        ax.fill_between(subset['n_hvg'],
                        subset['mean_auroc'] - subset['std_auroc'],
                        subset['mean_auroc'] + subset['std_auroc'],
                        alpha=0.2, color='#4CAF50')
        for _, row in subset.iterrows():
            ax.annotate(f"n={int(row['n_evaluated'])}", (row['n_hvg'], row['mean_auroc']),
                       textcoords="offset points", xytext=(5, 10), fontsize=8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of highly variable genes')
    ax.set_ylabel('Mean AUROC vs DE targets')
    ax.set_title('(B) Sensitivity to gene universe size\n(N_ctrl=2000, LFC>0.5)')
    ax.set_ylim(0.45, 0.85)

    # Panel C: AUROC vs LFC threshold (multiple N_ctrl curves)
    ax = axes[2]
    colors_ctrl = {500: '#FF9800', 2000: '#9C27B0', 10000: '#F44336'}
    for nc in n_control_values:
        subset = results_df[(results_df['n_control'] == nc) & (results_df['n_hvg'] == 2000)]
        if len(subset) > 0:
            ax.plot(subset['lfc_threshold'], subset['mean_auroc'], 'o-',
                    color=colors_ctrl.get(nc, 'gray'), markersize=8, label=f'N_ctrl={nc}')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('LFC threshold')
    ax.set_ylabel('Mean AUROC vs DE targets')
    ax.set_title('(C) Sensitivity to DE threshold\n(HVG=2000)')
    ax.set_ylim(0.45, 0.85)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_perturbation_sensitivity.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Figure saved to {fig_path}")

    # Final summary
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
