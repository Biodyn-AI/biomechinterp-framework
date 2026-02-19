#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals for Per-TF AUROC
=================================================
Extends the degree-preserving null experiment with cell-level bootstrap
uncertainty quantification for per-TF and global AUROC.

Design:
  1. Load Tabula Sapiens immune data, preprocess (normalize, log1p, HVG=2000, 5000 cells)
  2. Load TRRUST, build label matrix
  3. For B bootstrap iterations: resample cells with replacement, recompute
     Spearman correlation, compute per-TF AUROC and global AUROC
  4. Report 95% bootstrap CIs for global AUROC, mean per-TF AUROC, and
     individual per-TF AUROCs
  5. Degree-matched negative sampling control
  6. Generate figure with two panels:
     (A) Bootstrap CI for global AUROC (histogram + 95% CI lines)
     (B) Per-TF AUROC forest plot with bootstrap CIs
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Paths ----
DATA_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/data/raw"
REF_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks"
PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "bootstrap_per_tf_results")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_trrust():
    """Load TRRUST human TF-target pairs."""
    path = os.path.join(REF_DIR, "trrust_human.tsv")
    df = pd.read_csv(path, sep='\t', header=None,
                     names=['tf', 'target', 'mode', 'pmid'])
    return set(zip(df['tf'].str.upper(), df['target'].str.upper()))


def compute_spearman_corr(X):
    """Compute Spearman correlation matrix from data matrix (cells x genes)."""
    ranks = np.apply_along_axis(rankdata, 0, X)
    corr = np.corrcoef(ranks.T)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def compute_per_tf_auroc(corr_matrix, tf_indices, gene_upper, labels_matrix, n_genes):
    """
    Compute per-TF AUROC from correlation matrix and label matrix.
    Returns dict: {tf_name: auroc} for TFs with >0 and <all targets.
    Also returns global AUROC.
    """
    per_tf = {}
    all_scores = []
    all_labels = []

    for i, tf_idx in enumerate(tf_indices):
        mask = np.ones(n_genes, dtype=bool)
        mask[tf_idx] = False
        scores_tf = np.abs(corr_matrix[tf_idx, mask])
        labels_tf = labels_matrix[i, mask]

        all_scores.append(scores_tf)
        all_labels.append(labels_tf)

        n_pos = labels_tf.sum()
        if n_pos > 0 and n_pos < len(labels_tf):
            auroc = roc_auc_score(labels_tf, scores_tf)
            per_tf[gene_upper[tf_idx]] = auroc

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    n_pos_global = all_labels.sum()
    if n_pos_global > 0 and n_pos_global < len(all_labels):
        global_auroc = roc_auc_score(all_labels, all_scores)
    else:
        global_auroc = np.nan

    return per_tf, global_auroc


def degree_matched_negative_auroc(corr_matrix, tf_indices, gene_upper,
                                  labels_matrix, n_genes, trrust, rng):
    """
    For each TF, sample negatives matching the target-degree distribution
    of that TF's true targets.

    For each TF:
      - Find its true targets and their in-degrees (how many TFs regulate them)
      - For each true target, sample a non-target gene with the same in-degree
      - Compute AUROC using true targets as positives and degree-matched
        genes as negatives
    """
    # Compute in-degree for each gene (column sum of labels_matrix)
    in_degrees = labels_matrix.sum(axis=0)  # shape (n_genes,)

    # Group genes by in-degree for fast sampling
    degree_to_genes = {}
    for j in range(n_genes):
        d = int(in_degrees[j])
        if d not in degree_to_genes:
            degree_to_genes[d] = []
        degree_to_genes[d].append(j)

    per_tf_auroc_dm = {}

    for i, tf_idx in enumerate(tf_indices):
        tf_name = gene_upper[tf_idx]
        labels_tf = labels_matrix[i]
        pos_indices = np.where(labels_tf == 1)[0]
        n_pos = len(pos_indices)
        if n_pos == 0:
            continue

        # Get in-degrees of true targets
        target_degrees = [int(in_degrees[j]) for j in pos_indices]

        # For each target degree, sample a non-target gene with same degree
        neg_indices = []
        for d in target_degrees:
            candidates = [g for g in degree_to_genes.get(d, [])
                          if labels_tf[g] == 0 and g != tf_idx]
            if len(candidates) == 0:
                # Fall back to closest available degree
                for delta in range(1, max(in_degrees) + 1):
                    candidates_up = [g for g in degree_to_genes.get(d + delta, [])
                                     if labels_tf[g] == 0 and g != tf_idx]
                    candidates_dn = [g for g in degree_to_genes.get(d - delta, [])
                                     if labels_tf[g] == 0 and g != tf_idx]
                    candidates = candidates_up + candidates_dn
                    if len(candidates) > 0:
                        break
            if len(candidates) > 0:
                neg_indices.append(rng.choice(candidates))

        if len(neg_indices) == 0:
            continue

        # Build evaluation set: positives + degree-matched negatives
        eval_indices = list(pos_indices) + neg_indices
        eval_labels = np.array([1] * len(pos_indices) + [0] * len(neg_indices))
        eval_scores = np.abs(corr_matrix[tf_idx, eval_indices])

        n_pos_eval = eval_labels.sum()
        if n_pos_eval > 0 and n_pos_eval < len(eval_labels):
            auroc = roc_auc_score(eval_labels, eval_scores)
            per_tf_auroc_dm[tf_name] = auroc

    return per_tf_auroc_dm


def run_bootstrap_analysis():
    import scanpy as sc

    print("=" * 60)
    print("BOOTSTRAP CONFIDENCE INTERVALS FOR PER-TF AUROC")
    print("=" * 60)

    # ==============================
    # 1. Load & preprocess data
    # ==============================
    print("\n--- 1. Loading Tabula Sapiens immune subset ---")
    adata = sc.read_h5ad(os.path.join(
        DATA_DIR, "tabula_sapiens_immune_subset_20000.h5ad"))
    print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    # Map gene names (handle duplicates)
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

    # Full data matrix (before subsampling -- we subsample per bootstrap)
    X_full = adata.X
    if hasattr(X_full, 'toarray'):
        X_full = X_full.toarray()
    X_full = np.asarray(X_full, dtype=np.float64)
    n_cells_total = X_full.shape[0]
    print(f"  After HVG filter: {n_genes} genes, {n_cells_total} cells")

    # ==============================
    # 2. Load TRRUST & build labels
    # ==============================
    print("\n--- 2. Loading TRRUST & building label matrix ---")
    trrust = load_trrust()
    trrust_tfs = set(tf for tf, _ in trrust)
    tfs_in_data = sorted([g for g in set(gene_upper) if g in trrust_tfs])
    print(f"  TRRUST TFs in HVG data: {len(tfs_in_data)}")

    tf_indices = [gene_upper.index(tf) for tf in tfs_in_data if tf in gene_upper]
    tf_names_list = [gene_upper[i] for i in tf_indices]

    labels_matrix = np.zeros((len(tf_indices), n_genes), dtype=int)
    for i, tf_idx in enumerate(tf_indices):
        tf_name = gene_upper[tf_idx]
        for j in range(n_genes):
            if j == tf_idx:
                continue
            if (tf_name, gene_upper[j]) in trrust:
                labels_matrix[i, j] = 1

    n_pos = labels_matrix.sum()
    print(f"  Label matrix: {labels_matrix.shape}, {n_pos} positive edges")

    # Identify evaluable TFs (those with >0 and <all positives)
    evaluable_mask = []
    for i in range(len(tf_indices)):
        mask = np.ones(n_genes, dtype=bool)
        mask[tf_indices[i]] = False
        n_pos_tf = labels_matrix[i, mask].sum()
        n_total_tf = mask.sum()
        evaluable_mask.append(n_pos_tf > 0 and n_pos_tf < n_total_tf)
    evaluable_tfs = [tf_names_list[i] for i, e in enumerate(evaluable_mask) if e]
    print(f"  Evaluable TFs: {len(evaluable_tfs)}")

    # ==============================
    # 3. Observed AUROC (baseline)
    # ==============================
    print("\n--- 3. Computing observed AUROC (5000-cell subsample) ---")
    n_subsample = min(5000, n_cells_total)
    rng = np.random.RandomState(42)
    cell_idx_obs = rng.choice(n_cells_total, n_subsample, replace=False)
    X_obs = X_full[cell_idx_obs]
    corr_obs = compute_spearman_corr(X_obs)

    per_tf_obs, global_auroc_obs = compute_per_tf_auroc(
        corr_obs, tf_indices, gene_upper, labels_matrix, n_genes)
    print(f"  Global AUROC: {global_auroc_obs:.4f}")
    print(f"  Mean per-TF AUROC: {np.mean(list(per_tf_obs.values())):.4f}")
    print(f"  TFs evaluated: {len(per_tf_obs)}")

    # ==============================
    # 4. Bootstrap
    # ==============================
    N_BOOT = 200
    print(f"\n--- 4. Running {N_BOOT} bootstrap iterations ---")

    boot_global_aurocs = []
    boot_per_tf_aurocs = {tf: [] for tf in evaluable_tfs}

    t_start = time.time()
    for b in range(N_BOOT):
        # Resample cells WITH replacement
        boot_idx = rng.choice(n_subsample, n_subsample, replace=True)
        X_boot = X_obs[boot_idx]

        # Recompute Spearman correlation
        corr_boot = compute_spearman_corr(X_boot)

        # Compute per-TF and global AUROC
        per_tf_b, global_auroc_b = compute_per_tf_auroc(
            corr_boot, tf_indices, gene_upper, labels_matrix, n_genes)

        boot_global_aurocs.append(global_auroc_b)
        for tf in evaluable_tfs:
            if tf in per_tf_b:
                boot_per_tf_aurocs[tf].append(per_tf_b[tf])

        if (b + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = elapsed / (b + 1)
            eta = rate * (N_BOOT - b - 1)
            print(f"    Bootstrap {b+1}/{N_BOOT} "
                  f"(elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)")

    total_time = time.time() - t_start
    print(f"  Bootstrap complete in {total_time:.0f}s "
          f"({total_time/N_BOOT:.1f}s per iteration)")

    boot_global_aurocs = np.array(boot_global_aurocs)

    # ==============================
    # 5. Compute CIs
    # ==============================
    print("\n--- 5. Bootstrap confidence intervals ---")

    # Global AUROC CI
    global_ci_lo = np.percentile(boot_global_aurocs, 2.5)
    global_ci_hi = np.percentile(boot_global_aurocs, 97.5)
    global_boot_mean = np.mean(boot_global_aurocs)
    global_boot_std = np.std(boot_global_aurocs)
    print(f"  Global AUROC: {global_auroc_obs:.4f} "
          f"[95% CI: {global_ci_lo:.4f} - {global_ci_hi:.4f}]")
    print(f"  Bootstrap mean: {global_boot_mean:.4f} +/- {global_boot_std:.4f}")

    # Mean per-TF AUROC CI
    # For each bootstrap, compute mean across TFs
    boot_mean_per_tf = []
    for b in range(N_BOOT):
        vals = []
        for tf in evaluable_tfs:
            if b < len(boot_per_tf_aurocs[tf]):
                vals.append(boot_per_tf_aurocs[tf][b])
        if len(vals) > 0:
            boot_mean_per_tf.append(np.mean(vals))
    boot_mean_per_tf = np.array(boot_mean_per_tf)

    mean_per_tf_obs = np.mean(list(per_tf_obs.values()))
    mean_per_tf_ci_lo = np.percentile(boot_mean_per_tf, 2.5)
    mean_per_tf_ci_hi = np.percentile(boot_mean_per_tf, 97.5)
    print(f"  Mean per-TF AUROC: {mean_per_tf_obs:.4f} "
          f"[95% CI: {mean_per_tf_ci_lo:.4f} - {mean_per_tf_ci_hi:.4f}]")

    # Individual per-TF CIs
    per_tf_ci_results = []
    for tf in evaluable_tfs:
        vals = boot_per_tf_aurocs[tf]
        if len(vals) >= 10:  # need enough bootstraps for CI
            ci_lo = np.percentile(vals, 2.5)
            ci_hi = np.percentile(vals, 97.5)
            per_tf_ci_results.append({
                'tf': tf,
                'observed_auroc': per_tf_obs.get(tf, np.nan),
                'boot_mean': np.mean(vals),
                'boot_std': np.std(vals),
                'ci_95_lo': ci_lo,
                'ci_95_hi': ci_hi,
                'n_boot_samples': len(vals),
                'out_degree': int(labels_matrix[tf_names_list.index(tf)].sum())
                    if tf in tf_names_list else 0,
            })

    ci_df = pd.DataFrame(per_tf_ci_results)
    ci_df = ci_df.sort_values('observed_auroc', ascending=True).reset_index(drop=True)
    print(f"  Per-TF CIs computed for {len(ci_df)} TFs")

    # How many TFs have CI entirely above 0.5?
    n_sig = (ci_df['ci_95_lo'] > 0.5).sum()
    n_total = len(ci_df)
    print(f"  TFs with 95% CI entirely above 0.5: {n_sig}/{n_total}")

    # ==============================
    # 6. Degree-matched negative sampling
    # ==============================
    print("\n--- 6. Degree-matched negative sampling control ---")

    # Run on observed correlation
    dm_aurocs_obs = degree_matched_negative_auroc(
        corr_obs, tf_indices, gene_upper, labels_matrix, n_genes, trrust,
        rng=np.random.RandomState(42))

    if len(dm_aurocs_obs) > 0:
        dm_mean = np.mean(list(dm_aurocs_obs.values()))
        dm_median = np.median(list(dm_aurocs_obs.values()))
        print(f"  Degree-matched AUROC: mean={dm_mean:.4f}, median={dm_median:.4f}")
        print(f"  TFs evaluated: {len(dm_aurocs_obs)}")

        # Run multiple times to get stability estimate
        dm_runs = []
        for seed in range(20):
            dm_run = degree_matched_negative_auroc(
                corr_obs, tf_indices, gene_upper, labels_matrix, n_genes,
                trrust, rng=np.random.RandomState(seed + 100))
            if len(dm_run) > 0:
                dm_runs.append(np.mean(list(dm_run.values())))
        dm_runs = np.array(dm_runs)
        print(f"  Degree-matched mean AUROC across 20 seeds: "
              f"{dm_runs.mean():.4f} +/- {dm_runs.std():.4f}")
    else:
        dm_mean = np.nan
        dm_runs = np.array([])
        print("  No TFs could be evaluated with degree matching.")

    # Add degree-matched column to CI dataframe
    ci_df['degree_matched_auroc'] = ci_df['tf'].map(dm_aurocs_obs)

    # ==============================
    # 7. Save results
    # ==============================
    print("\n--- 7. Saving results ---")

    # Per-TF CI CSV
    csv_path = os.path.join(OUT_DIR, "per_tf_bootstrap_ci.csv")
    ci_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Global bootstrap samples
    np.save(os.path.join(OUT_DIR, "boot_global_aurocs.npy"), boot_global_aurocs)

    # Summary JSON
    summary = {
        'experiment': 'bootstrap_per_tf_auroc',
        'dataset': 'tabula_sapiens_immune',
        'n_genes': n_genes,
        'n_tfs_evaluable': len(evaluable_tfs),
        'n_cells_subsample': n_subsample,
        'n_bootstrap_iterations': N_BOOT,
        'runtime_seconds': round(total_time, 1),
        'global_auroc': {
            'observed': round(float(global_auroc_obs), 4),
            'boot_mean': round(float(global_boot_mean), 4),
            'boot_std': round(float(global_boot_std), 4),
            'ci_95': [round(float(global_ci_lo), 4),
                      round(float(global_ci_hi), 4)],
        },
        'mean_per_tf_auroc': {
            'observed': round(float(mean_per_tf_obs), 4),
            'boot_mean': round(float(np.mean(boot_mean_per_tf)), 4),
            'boot_std': round(float(np.std(boot_mean_per_tf)), 4),
            'ci_95': [round(float(mean_per_tf_ci_lo), 4),
                      round(float(mean_per_tf_ci_hi), 4)],
        },
        'per_tf_summary': {
            'n_tfs_with_ci': len(ci_df),
            'n_tfs_ci_above_0.5': int(n_sig),
            'fraction_ci_above_0.5': round(float(n_sig / n_total), 4)
                if n_total > 0 else 0,
            'mean_ci_width': round(float(
                (ci_df['ci_95_hi'] - ci_df['ci_95_lo']).mean()), 4)
                if len(ci_df) > 0 else None,
        },
        'degree_matched_control': {
            'mean_auroc': round(float(dm_mean), 4)
                if not np.isnan(dm_mean) else None,
            'mean_across_seeds': round(float(dm_runs.mean()), 4)
                if len(dm_runs) > 0 else None,
            'std_across_seeds': round(float(dm_runs.std()), 4)
                if len(dm_runs) > 0 else None,
            'n_tfs': len(dm_aurocs_obs),
        },
    }

    json_path = os.path.join(OUT_DIR, "bootstrap_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    # ==============================
    # 8. Generate figure
    # ==============================
    print("\n--- 8. Generating figure ---")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # ------ Panel A: Global AUROC bootstrap histogram ------
    ax = axes[0]
    ax.hist(boot_global_aurocs, bins=30, color='#2196F3', alpha=0.7,
            edgecolor='white', linewidth=0.5)
    ax.axvline(global_auroc_obs, color='red', linewidth=2, linestyle='--',
               label=f'Observed: {global_auroc_obs:.4f}')
    ax.axvline(global_ci_lo, color='#FF9800', linewidth=1.5, linestyle=':',
               label=f'95% CI: [{global_ci_lo:.4f}, {global_ci_hi:.4f}]')
    ax.axvline(global_ci_hi, color='#FF9800', linewidth=1.5, linestyle=':')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xlabel('Global AUROC', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('(A) Bootstrap distribution of global AUROC\n'
                 f'({N_BOOT} bootstrap resamples)', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')

    # ------ Panel B: Forest plot of per-TF AUROC with CIs ------
    ax = axes[1]
    if len(ci_df) > 0:
        y_positions = np.arange(len(ci_df))
        observed_vals = ci_df['observed_auroc'].values
        ci_lo_vals = ci_df['ci_95_lo'].values
        ci_hi_vals = ci_df['ci_95_hi'].values
        tf_labels = ci_df['tf'].values

        # Color by whether CI is entirely above 0.5
        colors = ['#4CAF50' if lo > 0.5 else '#2196F3'
                  for lo in ci_lo_vals]

        # Plot CIs
        for idx in range(len(ci_df)):
            ax.plot([ci_lo_vals[idx], ci_hi_vals[idx]],
                    [y_positions[idx], y_positions[idx]],
                    color=colors[idx], linewidth=1.5, alpha=0.7)

        # Plot point estimates
        ax.scatter(observed_vals, y_positions, color=colors, s=20,
                   zorder=5, edgecolors='black', linewidths=0.3)

        # Plot degree-matched values where available
        if 'degree_matched_auroc' in ci_df.columns:
            dm_vals = ci_df['degree_matched_auroc'].values
            valid_dm = ~np.isnan(dm_vals.astype(float))
            if valid_dm.any():
                ax.scatter(dm_vals[valid_dm], y_positions[valid_dm],
                           color='#FF9800', s=15, marker='D', zorder=4,
                           alpha=0.7, label='Degree-matched')

        # Reference line
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Y-axis labels (show TF names if reasonable number)
        if len(ci_df) <= 50:
            ax.set_yticks(y_positions)
            ax.set_yticklabels(tf_labels, fontsize=6)
        else:
            # Show subset of labels
            step = max(1, len(ci_df) // 30)
            show_idx = list(range(0, len(ci_df), step))
            ax.set_yticks([y_positions[i] for i in show_idx])
            ax.set_yticklabels([tf_labels[i] for i in show_idx], fontsize=6)

        ax.set_xlabel('AUROC', fontsize=12)
        ax.set_ylabel('Transcription Factor', fontsize=12)
        ax.set_title(f'(B) Per-TF AUROC with 95% bootstrap CIs\n'
                     f'({n_sig}/{n_total} TFs with CI above 0.5)',
                     fontsize=12)

        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50',
                   markersize=7, label=f'CI > 0.5 (n={n_sig})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
                   markersize=7, label=f'CI includes 0.5 (n={n_total - n_sig})'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#FF9800',
                   markersize=6, label='Degree-matched control'),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

    plt.tight_layout()

    fig_path_png = os.path.join(FIG_DIR, "fig_bootstrap_per_tf.png")
    fig_path_pdf = os.path.join(FIG_DIR, "fig_bootstrap_per_tf.pdf")
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    print(f"  Figure saved: {fig_path_png}")
    print(f"  Figure saved: {fig_path_pdf}")

    # ==============================
    # Final summary
    # ==============================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print("\n=== EXPERIMENT COMPLETE ===")
    return summary


if __name__ == '__main__':
    run_bootstrap_analysis()
