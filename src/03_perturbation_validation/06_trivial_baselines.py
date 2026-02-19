#!/usr/bin/env python3
"""
Perturbation-First Trivial Baselines Experiment
================================================
Nature Machine Intelligence paper supplement.

Compares correlation-edge perturbation-first AUROC against trivial baselines
using only target mean expression, dropout rate, variance, and TF/target
degree priors. Uses Replogle CRISPRi data at primary parameterization
(N_ctrl=2000, HVG=2000, LFC>0.5).

Baselines:
  1. Mean expression: |mean(target)| in control cells
  2. Dropout rate:    1 - fraction_zero(target) in control cells
  3. Variance:        var(target) in control cells
  4. TF degree:       number of TRRUST targets of perturbed gene (constant per TF)
  5. Correlation:     |Spearman corr(perturbation gene, target)| in control cells
"""

import os
import sys
import json
import time
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

# ── Paths ────────────────────────────────────────────────────────────────────
PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
DATA_DIR  = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data")
FIG_DIR   = os.path.join(PAPER_DIR, "figures")
OUT_DIR   = os.path.join(PAPER_DIR, "experiments", "perturbation_trivial_baselines_results")
TRRUST_PATH = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Primary parameterization ────────────────────────────────────────────────
N_CTRL = 2000
N_HVG  = 2000
LFC_THRESHOLD = 0.5
MIN_DE_TARGETS = 5


def call_de_vectorized(X_ctrl, X_pert):
    """Fully vectorized Welch's t-test across all genes at once.

    Identical to perturbation_sensitivity_mp.py implementation.
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


def run_trivial_baselines():
    import scanpy as sc

    t0 = time.time()

    print("=" * 70)
    print("PERTURBATION-FIRST TRIVIAL BASELINES EXPERIMENT")
    print("=" * 70)
    print(f"  Parameterization: N_ctrl={N_CTRL}, N_HVG={N_HVG}, LFC>{LFC_THRESHOLD}")
    print()

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading Replogle CRISPRi data...")
    adata = sc.read_h5ad(os.path.join(DATA_DIR, "replogle_concat.h5ad"))
    print(f"  Shape: {adata.shape}")

    pert_col = 'gene'
    ctrl_mask = adata.obs[pert_col] == 'non-targeting'
    print(f"  Control cells: {ctrl_mask.sum()}")
    print(f"  Perturbed cells: {(~ctrl_mask).sum()}")

    # Normalize
    print("  Normalizing (library-size + log1p)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ── Control cell subsample ───────────────────────────────────────────
    ctrl_indices = np.where(ctrl_mask.values)[0]
    np.random.seed(42)
    ctrl_idx = np.random.choice(ctrl_indices, N_CTRL, replace=False)
    print(f"  Sampled {N_CTRL} control cells")

    # ── Select HVGs ──────────────────────────────────────────────────────
    ctrl_sub = adata[ctrl_idx].copy()
    sc.pp.highly_variable_genes(ctrl_sub, n_top_genes=min(N_HVG, ctrl_sub.n_vars))
    hvg_genes = list(ctrl_sub.var_names[ctrl_sub.var['highly_variable']])
    n_genes = len(hvg_genes)
    gene_upper = [g.upper() for g in hvg_genes]
    gene_upper_set = set(gene_upper)
    gene_to_idx = {g.upper(): i for i, g in enumerate(hvg_genes)}
    print(f"  HVG count: {n_genes}")

    # ── Control expression matrix (HVG) ─────────────────────────────────
    print("  Extracting control expression matrix...")
    ctrl_hvg = adata[ctrl_idx][:, hvg_genes]
    X_ctrl = ctrl_hvg.X
    if hasattr(X_ctrl, 'toarray'):
        X_ctrl = X_ctrl.toarray()
    X_ctrl = np.asarray(X_ctrl, dtype=np.float64)

    # ── Pre-compute trivial features from controls ──────────────────────
    print("  Computing trivial features from control cells...")
    mean_expr = np.abs(X_ctrl.mean(axis=0))                    # Baseline 1
    dropout_rate = (X_ctrl == 0).mean(axis=0)
    anti_dropout = 1.0 - dropout_rate                          # Baseline 2
    variance = X_ctrl.var(axis=0)                              # Baseline 3

    print(f"    Mean expression range: [{mean_expr.min():.4f}, {mean_expr.max():.4f}]")
    print(f"    Dropout rate range:    [{dropout_rate.min():.4f}, {dropout_rate.max():.4f}]")
    print(f"    Variance range:        [{variance.min():.6f}, {variance.max():.4f}]")

    # ── Correlation matrix (Spearman) ────────────────────────────────────
    print("  Computing Spearman correlation matrix...")
    corr_matrix = compute_correlation_edges(X_ctrl)
    print(f"    Correlation matrix shape: {corr_matrix.shape}")

    # ── TRRUST: TF degree prior ──────────────────────────────────────────
    print("  Loading TRRUST network...")
    trrust_df = pd.read_csv(TRRUST_PATH, sep='\t', header=None,
                            names=['tf', 'target', 'mode', 'pmid'])
    known_tfs = set(trrust_df['tf'].str.upper())
    # Compute TF out-degree: number of unique targets per TF
    tf_degree = trrust_df.groupby(trrust_df['tf'].str.upper())['target'].nunique().to_dict()
    print(f"    TRRUST TFs: {len(known_tfs)}")
    print(f"    Median TF degree: {np.median(list(tf_degree.values())):.0f}")

    # ── Per-perturbation cell indices ────────────────────────────────────
    pert_genes = adata.obs[~ctrl_mask][pert_col].unique().tolist()
    pert_cell_indices = {}
    for pg in pert_genes:
        idx = np.where(adata.obs[pert_col].values == pg)[0]
        if len(idx) >= 10:
            pert_cell_indices[pg] = idx
    evaluable_perts = [pg for pg in pert_cell_indices if pg.upper() in gene_upper_set]
    print(f"  Perturbation genes in HVG set (>= 10 cells): {len(evaluable_perts)}")

    # ── Compute DE for each perturbation ─────────────────────────────────
    print("\n  Computing DE for each perturbation (vectorized)...")
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
            print(f"    ... {i+1}/{len(evaluable_perts)} perturbations done")

    print(f"  DE computed for {len(de_cache)} perturbations")

    # ── Evaluate all predictors ──────────────────────────────────────────
    print("\n  Evaluating predictors per perturbation...")
    predictor_names = ['Mean Expression', 'Anti-Dropout', 'Variance',
                       'TF Degree', 'Correlation']

    # Collect per-perturbation AUROCs
    results_per_pert = []
    n_evaluated = 0
    n_skipped_no_de = 0
    n_skipped_tf_degree = 0

    for pg in evaluable_perts:
        reject, adj_pvals, lfcs = de_cache[pg]

        # Identify DE targets
        de_targets = set()
        for j in range(n_genes):
            if reject[j] and abs(lfcs[j]) >= LFC_THRESHOLD:
                de_targets.add(gene_upper[j])

        if len(de_targets) < MIN_DE_TARGETS:
            n_skipped_no_de += 1
            continue

        pg_upper = pg.upper()
        pert_idx = gene_to_idx.get(pg_upper)
        if pert_idx is None:
            continue

        # Labels: 1 for DE target, 0 otherwise (exclude self)
        labels = np.array([1 if gene_upper[j] in de_targets else 0
                           for j in range(n_genes)])
        mask = np.ones(n_genes, dtype=bool)
        mask[pert_idx] = False
        labs = labels[mask]

        if labs.sum() == 0 or labs.sum() == len(labs):
            continue

        is_tf = pg_upper in known_tfs

        row = {
            'perturbation': pg,
            'is_tf': is_tf,
            'n_de_targets': int(labs.sum()),
        }

        # Baseline 1: Mean expression
        scores_mean = mean_expr[mask]
        row['auroc_mean_expr'] = roc_auc_score(labs, scores_mean)

        # Baseline 2: Anti-dropout
        scores_dropout = anti_dropout[mask]
        row['auroc_anti_dropout'] = roc_auc_score(labs, scores_dropout)

        # Baseline 3: Variance
        scores_var = variance[mask]
        row['auroc_variance'] = roc_auc_score(labs, scores_var)

        # Baseline 4: TF degree (only meaningful for known TFs)
        # For a given perturbation gene, the TF degree is a constant scalar,
        # so it cannot discriminate targets -- AUROC is undefined / 0.5.
        # Instead we use it as-is: every target gets the same score, AUROC=0.5.
        # This tests the "TF degree prior" -- it is trivially 0.5.
        degree_val = tf_degree.get(pg_upper, 0)
        row['tf_degree'] = degree_val
        # A constant score gives AUROC = 0.5 by definition.
        # We record it but it is trivially uninformative per-perturbation.
        row['auroc_tf_degree'] = 0.5

        # Baseline 5: Correlation (the "real" predictor)
        scores_corr = np.abs(corr_matrix[pert_idx])[mask]
        row['auroc_correlation'] = roc_auc_score(labs, scores_corr)

        results_per_pert.append(row)
        n_evaluated += 1

        if n_evaluated <= 5:
            tf_label = "TF" if is_tf else "non-TF"
            print(f"    {pg} ({tf_label}): {row['n_de_targets']} DE targets | "
                  f"corr={row['auroc_correlation']:.3f}, "
                  f"mean={row['auroc_mean_expr']:.3f}, "
                  f"drop={row['auroc_anti_dropout']:.3f}, "
                  f"var={row['auroc_variance']:.3f}")

    print(f"\n  Total evaluated: {n_evaluated}")
    print(f"  Skipped (<{MIN_DE_TARGETS} DE targets): {n_skipped_no_de}")

    df = pd.DataFrame(results_per_pert)
    df.to_csv(os.path.join(OUT_DIR, "per_perturbation_aurocs.csv"), index=False)

    # ── Aggregate summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    auroc_cols = ['auroc_mean_expr', 'auroc_anti_dropout', 'auroc_variance',
                  'auroc_tf_degree', 'auroc_correlation']
    labels_nice = ['Mean Expression', '1 - Dropout Rate', 'Variance',
                   'TF Degree (const)', '|Spearman Corr|']

    summary_rows = []
    for col, name in zip(auroc_cols, labels_nice):
        vals = df[col].values
        mean_v = np.mean(vals)
        std_v  = np.std(vals)
        med_v  = np.median(vals)
        se_v   = std_v / np.sqrt(len(vals))
        t_stat, t_p = stats.ttest_1samp(vals, 0.5) if len(vals) > 1 else (0, 1)

        summary_rows.append({
            'predictor': name,
            'n': len(vals),
            'mean_auroc': round(float(mean_v), 4),
            'std_auroc': round(float(std_v), 4),
            'se_auroc': round(float(se_v), 4),
            'median_auroc': round(float(med_v), 4),
            'ci95_lo': round(float(mean_v - 1.96 * se_v), 4),
            'ci95_hi': round(float(mean_v + 1.96 * se_v), 4),
            'ttest_vs_0.5_p': float(t_p),
        })
        sig = "*" if t_p < 0.05 else " "
        print(f"  {name:25s}: mean={mean_v:.4f} +/- {std_v:.4f}  "
              f"(median={med_v:.4f})  p(vs 0.5)={t_p:.2e} {sig}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "summary_aurocs.csv"), index=False)

    # ── Paired comparison: correlation vs each baseline ──────────────────
    print("\n  Paired comparisons (correlation vs each baseline):")
    paired_results = []
    for col, name in zip(auroc_cols[:-1], labels_nice[:-1]):
        diff = df['auroc_correlation'].values - df[col].values
        t_stat, t_p = stats.ttest_rel(df['auroc_correlation'].values, df[col].values)
        mean_diff = np.mean(diff)
        paired_results.append({
            'comparison': f'Correlation vs {name}',
            'mean_diff': round(float(mean_diff), 4),
            'paired_t_p': float(t_p),
        })
        sig = "***" if t_p < 0.001 else ("**" if t_p < 0.01 else ("*" if t_p < 0.05 else "ns"))
        print(f"    Corr vs {name:25s}: delta={mean_diff:+.4f}, p={t_p:.2e} {sig}")

    paired_df = pd.DataFrame(paired_results)
    paired_df.to_csv(os.path.join(OUT_DIR, "paired_comparisons.csv"), index=False)

    # ── TF vs non-TF breakdown ───────────────────────────────────────────
    print("\n  TF vs non-TF breakdown:")
    for col, name in zip(auroc_cols, labels_nice):
        tf_vals = df[df['is_tf']][col].values
        nontf_vals = df[~df['is_tf']][col].values
        tf_str = f"{np.mean(tf_vals):.4f} (n={len(tf_vals)})" if len(tf_vals) > 0 else "N/A"
        nontf_str = f"{np.mean(nontf_vals):.4f} (n={len(nontf_vals)})" if len(nontf_vals) > 0 else "N/A"
        print(f"    {name:25s}: TF={tf_str}, non-TF={nontf_str}")

    # ── Generate figure ──────────────────────────────────────────────────
    print("\n  Generating figure...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel A: Bar chart with error bars ───────────────────────────────
    ax = axes[0]
    pred_order = ['|Spearman Corr|', 'Mean Expression', '1 - Dropout Rate',
                  'Variance', 'TF Degree (const)']
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']

    means = []
    ses = []
    for name in pred_order:
        row = summary_df[summary_df['predictor'] == name].iloc[0]
        means.append(row['mean_auroc'])
        ses.append(row['se_auroc'])

    x = np.arange(len(pred_order))
    bars = ax.bar(x, means, yerr=[1.96 * s for s in ses], capsize=5,
                  color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='Chance (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(pred_order, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Mean AUROC', fontsize=11)
    ax.set_title('(A) Predictor comparison\n(N_ctrl=2000, HVG=2000, LFC>0.5)', fontsize=11)
    ax.set_ylim(0.40, max(means) + 0.08)
    ax.legend(fontsize=8, loc='upper right')

    # Annotate bars with mean values
    for i, (m, s) in enumerate(zip(means, ses)):
        ax.text(i, m + 1.96 * s + 0.005, f'{m:.3f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')

    # ── Panel B: Per-perturbation scatter: correlation vs best baseline ──
    ax = axes[1]
    # Best trivial baseline per perturbation
    trivial_cols = ['auroc_mean_expr', 'auroc_anti_dropout', 'auroc_variance']
    df['best_trivial'] = df[trivial_cols].max(axis=1)

    tf_mask_plot = df['is_tf'].values
    ax.scatter(df.loc[~tf_mask_plot, 'best_trivial'],
               df.loc[~tf_mask_plot, 'auroc_correlation'],
               alpha=0.35, s=15, color='#888888', label='Non-TF', zorder=2)
    ax.scatter(df.loc[tf_mask_plot, 'best_trivial'],
               df.loc[tf_mask_plot, 'auroc_correlation'],
               alpha=0.6, s=25, color='#E53935', marker='D', label='TF', zorder=3)

    lims = [0.3, 1.0]
    ax.plot(lims, lims, 'k--', alpha=0.4, label='y = x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Best trivial baseline AUROC', fontsize=11)
    ax.set_ylabel('|Spearman Corr| AUROC', fontsize=11)
    ax.set_title('(B) Correlation vs best trivial baseline\n(per perturbation)', fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_aspect('equal')

    # How many above diagonal?
    n_above = (df['auroc_correlation'] > df['best_trivial']).sum()
    n_total = len(df)
    frac_above = n_above / n_total
    ax.text(0.05, 0.95, f'{n_above}/{n_total} ({frac_above:.0%}) above diagonal',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()

    fig_path_png = os.path.join(FIG_DIR, "fig_trivial_baselines.png")
    fig_path_pdf = os.path.join(FIG_DIR, "fig_trivial_baselines.pdf")
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    print(f"  Figure saved: {fig_path_png}")
    print(f"  Figure saved: {fig_path_pdf}")
    plt.close()

    # ── Save full JSON summary ───────────────────────────────────────────
    elapsed = time.time() - t0
    full_summary = {
        'experiment': 'perturbation_trivial_baselines',
        'description': (
            'Compares correlation-edge perturbation-first AUROC against trivial '
            'baselines (mean expression, dropout, variance, TF degree) at '
            'primary parameterization.'
        ),
        'parameterization': {
            'n_ctrl': N_CTRL,
            'n_hvg': N_HVG,
            'lfc_threshold': LFC_THRESHOLD,
            'min_de_targets': MIN_DE_TARGETS,
        },
        'n_perturbations_evaluated': n_evaluated,
        'n_skipped_few_de': n_skipped_no_de,
        'summary': summary_rows,
        'paired_comparisons': paired_results,
        'elapsed_seconds': round(elapsed, 1),
    }
    with open(os.path.join(OUT_DIR, "trivial_baselines_summary.json"), 'w') as f:
        json.dump(full_summary, f, indent=2)
    print(f"\n  JSON summary saved to {OUT_DIR}/trivial_baselines_summary.json")

    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print("\n=== EXPERIMENT COMPLETE ===")
    return full_summary


if __name__ == '__main__':
    run_trivial_baselines()
