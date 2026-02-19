"""
Phase 2: Time-series perturbation validation using Dixit 7-day vs 13-day data.

Tests whether known temporal perturbation effects align with pseudotime predictions.
The Dixit dataset provides Perturb-seq data at two time points (7 days and 13 days),
with CRISPR knockdowns of TFs (EGR1, ELK1, IRF1, CREB1, ETS1, GABPA, ELF1, YY1,
E2F4, NR2C2) in K562 cells.

Analysis:
1. For each perturbed TF, identify its downstream targets (genes significantly
   affected by the perturbation).
2. Compare effect sizes at 7d vs 13d -- if pseudotime predictions are valid,
   target effects should be larger at 13d (more time for cascade).
3. Test whether the temporal amplification of perturbation effects matches
   the directionality predicted by pseudotime-based analysis.
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu, ttest_ind, spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

np.random.seed(42)

OUT_DIR = '/Users/ihorkendiukhov/biodyn-work/subproject_29_pseudotime_directionality_audit'
FIG_DIR = os.path.join(OUT_DIR, 'reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("LOADING PERTURBATION DATA")
print("=" * 60)

adata_13d = sc.read_h5ad('/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/perturb/dixit/perturb_processed_symbols.h5ad')
adata_7d = sc.read_h5ad('/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/perturb/dixit_7_days/perturb_processed_symbols.h5ad')

print(f"13-day: {adata_13d.shape}")
print(f"7-day: {adata_7d.shape}")

# Get TF targets (conditions that are actual TF knockdowns, not intergenic controls)
tfs_perturbed = sorted([c for c in adata_13d.obs['condition'].unique()
                        if c != 'control' and not c.startswith('INTERGENIC')])
print(f"\nPerturbed TFs: {tfs_perturbed}")

# ============================================================
# 2. COMPUTE DIFFERENTIAL EXPRESSION FOR EACH TF PERTURBATION
# ============================================================
print("\n" + "=" * 60)
print("DIFFERENTIAL EXPRESSION ANALYSIS")
print("=" * 60)

def compute_perturbation_effects(adata, tf_name):
    """
    Compute effect of TF perturbation vs control.
    Returns DataFrame with gene-level statistics.
    """
    ctrl_mask = adata.obs['condition'] == 'control'
    pert_mask = adata.obs['condition'] == tf_name

    n_ctrl = ctrl_mask.sum()
    n_pert = pert_mask.sum()

    if n_ctrl < 10 or n_pert < 10:
        return None

    # Get expression matrices
    if sparse.issparse(adata.X):
        ctrl_expr = np.asarray(adata.X[ctrl_mask].todense())
        pert_expr = np.asarray(adata.X[pert_mask].todense())
    else:
        ctrl_expr = np.asarray(adata.X[ctrl_mask])
        pert_expr = np.asarray(adata.X[pert_mask])

    genes = adata.var_names

    results = []
    for i, gene in enumerate(genes):
        ctrl_vals = ctrl_expr[:, i]
        pert_vals = pert_expr[:, i]

        # Mean expression
        ctrl_mean = ctrl_vals.mean()
        pert_mean = pert_vals.mean()

        # Skip very lowly expressed genes
        if ctrl_mean < 0.01 and pert_mean < 0.01:
            continue

        # Log fold change
        lfc = pert_mean - ctrl_mean  # Already log-normalized

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n_ctrl - 1) * ctrl_vals.std()**2 + (n_pert - 1) * pert_vals.std()**2) / (n_ctrl + n_pert - 2))
        cohens_d = lfc / pooled_std if pooled_std > 0 else 0

        # Statistical test
        if np.std(ctrl_vals) > 0 or np.std(pert_vals) > 0:
            stat, pval = mannwhitneyu(ctrl_vals, pert_vals, alternative='two-sided')
        else:
            pval = 1.0

        results.append({
            'gene': gene,
            'ctrl_mean': ctrl_mean,
            'pert_mean': pert_mean,
            'lfc': lfc,
            'cohens_d': cohens_d,
            'pval': pval,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        return None

    # FDR correction
    reject, pvals_bh, _, _ = multipletests(df['pval'].values, method='fdr_bh')
    df['pval_bh'] = pvals_bh
    df['significant'] = reject

    return df

# Compute effects at both timepoints
effects_7d = {}
effects_13d = {}

for tf in tfs_perturbed:
    print(f"\nProcessing {tf}...")

    # 7-day
    if tf in adata_7d.obs['condition'].unique():
        n_pert_7 = (adata_7d.obs['condition'] == tf).sum()
        n_ctrl_7 = (adata_7d.obs['condition'] == 'control').sum()
        print(f"  7d: {n_pert_7} perturbed, {n_ctrl_7} control cells")
        de_7 = compute_perturbation_effects(adata_7d, tf)
        if de_7 is not None:
            effects_7d[tf] = de_7
            n_sig_7 = de_7['significant'].sum()
            print(f"  7d: {n_sig_7} significant DE genes (FDR<0.05)")

    # 13-day
    if tf in adata_13d.obs['condition'].unique():
        n_pert_13 = (adata_13d.obs['condition'] == tf).sum()
        n_ctrl_13 = (adata_13d.obs['condition'] == 'control').sum()
        print(f"  13d: {n_pert_13} perturbed, {n_ctrl_13} control cells")
        de_13 = compute_perturbation_effects(adata_13d, tf)
        if de_13 is not None:
            effects_13d[tf] = de_13
            n_sig_13 = de_13['significant'].sum()
            print(f"  13d: {n_sig_13} significant DE genes (FDR<0.05)")

# ============================================================
# 3. TEMPORAL AMPLIFICATION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("TEMPORAL AMPLIFICATION ANALYSIS")
print("=" * 60)

temporal_results = []

for tf in tfs_perturbed:
    if tf not in effects_7d or tf not in effects_13d:
        continue

    de_7 = effects_7d[tf]
    de_13 = effects_13d[tf]

    # Merge on gene
    merged = de_7.merge(de_13, on='gene', suffixes=('_7d', '_13d'))

    if len(merged) == 0:
        continue

    # Genes significant at either timepoint
    sig_either = merged[(merged['significant_7d']) | (merged['significant_13d'])]

    if len(sig_either) < 5:
        # Use top genes by absolute LFC instead
        merged['max_abs_lfc'] = np.maximum(np.abs(merged['lfc_7d']), np.abs(merged['lfc_13d']))
        sig_either = merged.nlargest(min(50, len(merged)), 'max_abs_lfc')

    # Temporal amplification: do effects get stronger from 7d to 13d?
    abs_lfc_7d = np.abs(sig_either['lfc_7d'].values)
    abs_lfc_13d = np.abs(sig_either['lfc_13d'].values)

    # Mean absolute effect size
    mean_abs_7d = abs_lfc_7d.mean()
    mean_abs_13d = abs_lfc_13d.mean()
    amplification_ratio = mean_abs_13d / mean_abs_7d if mean_abs_7d > 0 else np.nan

    # Correlation of effect sizes across timepoints
    if len(sig_either) >= 5:
        r_effects, p_effects = pearsonr(sig_either['lfc_7d'], sig_either['lfc_13d'])
    else:
        r_effects, p_effects = 0, 1

    # Do more genes become significant at 13d?
    n_sig_7d_only = ((merged['significant_7d']) & (~merged['significant_13d'])).sum()
    n_sig_13d_only = ((~merged['significant_7d']) & (merged['significant_13d'])).sum()
    n_sig_both = ((merged['significant_7d']) & (merged['significant_13d'])).sum()
    n_sig_7d = merged['significant_7d'].sum()
    n_sig_13d = merged['significant_13d'].sum()

    # Direction consistency: do effects maintain the same sign?
    sig_both = merged[(merged['significant_7d']) & (merged['significant_13d'])]
    if len(sig_both) > 0:
        same_sign = (np.sign(sig_both['lfc_7d']) == np.sign(sig_both['lfc_13d'])).mean() * 100
    else:
        same_sign = np.nan

    # TF self-effect (does the TF itself show reduced expression?)
    tf_row = merged[merged['gene'] == tf]
    if len(tf_row) > 0:
        tf_lfc_7d = tf_row.iloc[0]['lfc_7d']
        tf_lfc_13d = tf_row.iloc[0]['lfc_13d']
        tf_sig_7d = tf_row.iloc[0]['significant_7d']
        tf_sig_13d = tf_row.iloc[0]['significant_13d']
    else:
        tf_lfc_7d = np.nan
        tf_lfc_13d = np.nan
        tf_sig_7d = False
        tf_sig_13d = False

    temporal_results.append({
        'TF': tf,
        'n_genes_analyzed': len(merged),
        'n_sig_7d': n_sig_7d,
        'n_sig_13d': n_sig_13d,
        'n_sig_both': n_sig_both,
        'n_sig_7d_only': n_sig_7d_only,
        'n_sig_13d_only': n_sig_13d_only,
        'mean_abs_lfc_7d': mean_abs_7d,
        'mean_abs_lfc_13d': mean_abs_13d,
        'amplification_ratio': amplification_ratio,
        'effect_correlation': r_effects,
        'effect_corr_pval': p_effects,
        'direction_consistency': same_sign,
        'tf_lfc_7d': tf_lfc_7d,
        'tf_lfc_13d': tf_lfc_13d,
        'tf_knockdown_7d': tf_sig_7d,
        'tf_knockdown_13d': tf_sig_13d,
    })

    print(f"\n{tf}:")
    print(f"  Sig genes: 7d={n_sig_7d}, 13d={n_sig_13d}, both={n_sig_both}")
    print(f"  Mean |LFC|: 7d={mean_abs_7d:.4f}, 13d={mean_abs_13d:.4f}, ratio={amplification_ratio:.2f}")
    print(f"  Effect corr: r={r_effects:.3f}, p={p_effects:.4f}")
    print(f"  Direction consistency: {same_sign:.1f}%" if not np.isnan(same_sign) else "  Direction consistency: N/A")
    print(f"  TF self-knockdown: 7d={tf_lfc_7d:.3f} (sig={tf_sig_7d}), 13d={tf_lfc_13d:.3f} (sig={tf_sig_13d})" if not np.isnan(tf_lfc_7d) else "  TF not found in data")

temporal_df = pd.DataFrame(temporal_results)

# ============================================================
# 4. PSEUDOTIME ALIGNMENT TEST
# ============================================================
print("\n" + "=" * 60)
print("PSEUDOTIME ALIGNMENT TEST")
print("=" * 60)
print("Testing: do genes affected early in perturbation (7d) correspond")
print("to genes predicted to change early in pseudotime?")

# Compute pseudotime on pooled K562 control cells
# K562 is a leukemia line, so we use control cells as our pseudotime reference
print("\nComputing pseudotime on K562 control cells (13d dataset)...")
ctrl_mask = adata_13d.obs['condition'] == 'control'
adata_ctrl = adata_13d[ctrl_mask].copy()
print(f"  {adata_ctrl.n_obs} control cells")

# Standard pseudotime pipeline
sc.pp.highly_variable_genes(adata_ctrl, n_top_genes=2000, flavor='seurat')
sc.tl.pca(adata_ctrl, n_comps=30)
sc.pp.neighbors(adata_ctrl, n_neighbors=15, n_pcs=30)
sc.tl.diffmap(adata_ctrl, n_comps=10)

# Use cell closest to centroid as root
pca = adata_ctrl.obsm['X_pca']
centroid = pca.mean(axis=0)
dists = np.linalg.norm(pca - centroid, axis=1)
adata_ctrl.uns['iroot'] = int(np.argmin(dists))
sc.tl.dpt(adata_ctrl, n_branchings=0)
sc.tl.umap(adata_ctrl)

pt = adata_ctrl.obs['dpt_pseudotime'].values
print(f"  DPT range: [{np.nanmin(pt):.3f}, {np.nanmax(pt):.3f}]")

# For each TF perturbation, compute expression gradient along pseudotime
# for significantly affected genes
alignment_results = []

for tf in tfs_perturbed:
    if tf not in effects_7d or tf not in effects_13d:
        continue

    de_7 = effects_7d[tf]
    de_13 = effects_13d[tf]
    merged = de_7.merge(de_13, on='gene', suffixes=('_7d', '_13d'))

    # Genes significant at 13d (strongest signal)
    sig_13 = merged[merged['significant_13d']].copy()
    if len(sig_13) < 5:
        sig_13 = merged.nlargest(min(30, len(merged)), 'max_abs_lfc' if 'max_abs_lfc' in merged.columns
                                  else merged.assign(max_abs_lfc=np.maximum(np.abs(merged['lfc_7d']), np.abs(merged['lfc_13d'])))['max_abs_lfc'].name)

    # For each affected gene, compute its peak pseudotime position
    gene_pt_peaks = []
    for _, row in sig_13.iterrows():
        gene = row['gene']
        if gene not in adata_ctrl.var_names:
            continue
        idx = list(adata_ctrl.var_names).index(gene)
        expr = adata_ctrl.X[:, idx]
        if sparse.issparse(expr):
            expr = np.asarray(expr.todense()).flatten()
        else:
            expr = np.asarray(expr).flatten()

        valid = np.isfinite(pt)
        pt_v = pt[valid]
        expr_v = expr[valid]

        # Bin expression
        n_bins = 20
        bin_edges = np.linspace(pt_v.min(), pt_v.max() + 1e-10, n_bins + 1)
        bin_idx = np.clip(np.digitize(pt_v, bin_edges) - 1, 0, n_bins - 1)
        bin_means = np.zeros(n_bins)
        for b in range(n_bins):
            m = bin_idx == b
            if m.sum() > 0:
                bin_means[b] = expr_v[m].mean()

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        peak_pt = bin_centers[np.argmax(bin_means)]

        gene_pt_peaks.append({
            'gene': gene,
            'lfc_7d': row['lfc_7d'],
            'lfc_13d': row['lfc_13d'],
            'peak_pt': peak_pt,
            'temporal_specificity': abs(row['lfc_13d']) - abs(row['lfc_7d']),
        })

    if len(gene_pt_peaks) < 3:
        continue

    gpt_df = pd.DataFrame(gene_pt_peaks)

    # Test: do genes with early pseudotime peaks show earlier perturbation effects?
    # Correlation between peak pseudotime and temporal specificity
    if len(gpt_df) >= 5:
        r_pt_temporal, p_pt_temporal = spearmanr(gpt_df['peak_pt'], gpt_df['temporal_specificity'])
    else:
        r_pt_temporal, p_pt_temporal = 0, 1

    # Also: correlation between peak_pt and lfc magnitude at each timepoint
    r_pt_7d, _ = spearmanr(gpt_df['peak_pt'], np.abs(gpt_df['lfc_7d'])) if len(gpt_df) >= 5 else (0, 1)
    r_pt_13d, _ = spearmanr(gpt_df['peak_pt'], np.abs(gpt_df['lfc_13d'])) if len(gpt_df) >= 5 else (0, 1)

    alignment_results.append({
        'TF': tf,
        'n_genes': len(gpt_df),
        'corr_pt_temporal': r_pt_temporal,
        'p_pt_temporal': p_pt_temporal,
        'corr_pt_lfc7d': r_pt_7d,
        'corr_pt_lfc13d': r_pt_13d,
    })

    print(f"\n{tf} (n={len(gpt_df)} genes):")
    print(f"  Corr(peak_pt, temporal_specificity): r={r_pt_temporal:.3f}, p={p_pt_temporal:.4f}")
    print(f"  Corr(peak_pt, |LFC_7d|): r={r_pt_7d:.3f}")
    print(f"  Corr(peak_pt, |LFC_13d|): r={r_pt_13d:.3f}")

alignment_df = pd.DataFrame(alignment_results)

# ============================================================
# 5. SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")

temporal_df.to_csv(os.path.join(OUT_DIR, 'implementation', 'phase2_perturbation_temporal.csv'), index=False)

if len(alignment_df) > 0:
    alignment_df.to_csv(os.path.join(OUT_DIR, 'implementation', 'phase2_perturbation_alignment.csv'), index=False)

with open(os.path.join(OUT_DIR, 'implementation', 'phase2_perturbation_full.pkl'), 'wb') as f:
    pickle.dump({
        'temporal_df': temporal_df,
        'alignment_df': alignment_df,
        'effects_7d': {tf: de[['gene', 'lfc', 'cohens_d', 'pval_bh', 'significant']].copy()
                       for tf, de in effects_7d.items()},
        'effects_13d': {tf: de[['gene', 'lfc', 'cohens_d', 'pval_bh', 'significant']].copy()
                        for tf, de in effects_13d.items()},
    }, f)

# ============================================================
# 6. FIGURES
# ============================================================
print("\nGenerating figures...")

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# Figure 1: Temporal amplification overview
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# (A) Number of significant genes at 7d vs 13d
ax = axes[0, 0]
if len(temporal_df) > 0:
    x = np.arange(len(temporal_df))
    width = 0.35
    ax.bar(x - width/2, temporal_df['n_sig_7d'], width, label='7-day', color='#4DBBD5', alpha=0.7)
    ax.bar(x + width/2, temporal_df['n_sig_13d'], width, label='13-day', color='#E64B35', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(temporal_df['TF'], rotation=45, ha='right')
    ax.set_ylabel('# Significant DE Genes')
    ax.set_title('(A) DE Genes by Timepoint')
    ax.legend()

# (B) Amplification ratio
ax = axes[0, 1]
if len(temporal_df) > 0:
    valid_amp = temporal_df[np.isfinite(temporal_df['amplification_ratio'])]
    if len(valid_amp) > 0:
        bars = ax.bar(range(len(valid_amp)), valid_amp['amplification_ratio'],
                      color=['#00A087' if r > 1 else '#E64B35' for r in valid_amp['amplification_ratio']],
                      alpha=0.7)
        ax.set_xticks(range(len(valid_amp)))
        ax.set_xticklabels(valid_amp['TF'], rotation=45, ha='right')
        ax.axhline(1, color='gray', linestyle='--', alpha=0.5, label='No amplification')
        ax.set_ylabel('|LFC_13d| / |LFC_7d|')
        ax.set_title('(B) Temporal Amplification Ratio')
        ax.legend()

# (C) Effect size correlation across timepoints
ax = axes[0, 2]
if len(temporal_df) > 0:
    ax.scatter(temporal_df['effect_correlation'], -np.log10(temporal_df['effect_corr_pval']),
               s=50, c='#4DBBD5', alpha=0.7, edgecolors='white')
    for _, row in temporal_df.iterrows():
        ax.annotate(row['TF'], (row['effect_correlation'], -np.log10(row['effect_corr_pval'])),
                    fontsize=7, ha='left')
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.3, label='p=0.05')
    ax.set_xlabel('Effect Correlation (7d vs 13d)')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('(C) Cross-timepoint Effect Correlation')
    ax.legend()

# (D) Example: effect sizes for top TF
ax = axes[1, 0]
if len(temporal_df) > 0:
    # Pick TF with most significant genes
    top_tf = temporal_df.loc[temporal_df['n_sig_13d'].idxmax(), 'TF']
    if top_tf in effects_7d and top_tf in effects_13d:
        de_7 = effects_7d[top_tf]
        de_13 = effects_13d[top_tf]
        merged = de_7.merge(de_13, on='gene', suffixes=('_7d', '_13d'))
        sig = merged[merged['significant_13d']].nlargest(20, 'max_abs_lfc' if 'max_abs_lfc' in merged.columns
                                                          else pd.Series(np.abs(merged['lfc_13d'])).name)
        if 'max_abs_lfc' not in merged.columns:
            sig = merged[merged['significant_13d']].copy()
            sig['abs_lfc_13d'] = np.abs(sig['lfc_13d'])
            sig = sig.nlargest(20, 'abs_lfc_13d')

        if len(sig) > 0:
            y_pos = range(len(sig))
            ax.barh(y_pos, sig['lfc_7d'], height=0.35, label='7-day', color='#4DBBD5', alpha=0.7, align='edge')
            ax.barh([y + 0.35 for y in y_pos], sig['lfc_13d'], height=0.35, label='13-day', color='#E64B35', alpha=0.7, align='edge')
            ax.set_yticks([y + 0.175 for y in y_pos])
            ax.set_yticklabels(sig['gene'], fontsize=6)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_xlabel('Log Fold Change')
            ax.set_title(f'(D) Top DE genes: {top_tf} KD')
            ax.legend(fontsize=7)
            ax.invert_yaxis()

# (E) TF self-knockdown
ax = axes[1, 1]
if len(temporal_df) > 0:
    valid_tf = temporal_df[np.isfinite(temporal_df['tf_lfc_7d']) & np.isfinite(temporal_df['tf_lfc_13d'])]
    if len(valid_tf) > 0:
        x = np.arange(len(valid_tf))
        width = 0.35
        ax.bar(x - width/2, valid_tf['tf_lfc_7d'], width, label='7-day', color='#4DBBD5', alpha=0.7)
        ax.bar(x + width/2, valid_tf['tf_lfc_13d'], width, label='13-day', color='#E64B35', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_tf['TF'], rotation=45, ha='right')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel('TF Expression LFC')
        ax.set_title('(E) TF Self-Knockdown Strength')
        ax.legend()

# (F) Pseudotime alignment
ax = axes[1, 2]
if len(alignment_df) > 0:
    ax.bar(range(len(alignment_df)), alignment_df['corr_pt_temporal'],
           color=['#00A087' if p < 0.05 else '#91D1C2' for p in alignment_df['p_pt_temporal']],
           alpha=0.7)
    ax.set_xticks(range(len(alignment_df)))
    ax.set_xticklabels(alignment_df['TF'], rotation=45, ha='right')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Spearman r')
    ax.set_title('(F) Pseudotime-Temporal Alignment\n(peak_pt vs temporal_specificity)')
else:
    ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'phase2_fig6_perturbation_validation.png'))
fig.savefig(os.path.join(FIG_DIR, 'phase2_fig6_perturbation_validation.pdf'))
plt.close()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

if len(temporal_df) > 0:
    print("\nTemporal Amplification Analysis:")
    print(temporal_df[['TF', 'n_sig_7d', 'n_sig_13d', 'amplification_ratio',
                        'effect_correlation', 'direction_consistency']].to_string(index=False))

    # Aggregate statistics
    mean_amp = temporal_df['amplification_ratio'].mean()
    median_amp = temporal_df['amplification_ratio'].median()
    amp_gt1 = (temporal_df['amplification_ratio'] > 1).mean() * 100
    mean_corr = temporal_df['effect_correlation'].mean()
    mean_dir = temporal_df['direction_consistency'].mean()

    print(f"\n  Mean amplification ratio: {mean_amp:.2f}")
    print(f"  Median amplification ratio: {median_amp:.2f}")
    print(f"  % TFs with amplification > 1: {amp_gt1:.1f}%")
    print(f"  Mean effect correlation (7d vs 13d): {mean_corr:.3f}")
    print(f"  Mean direction consistency: {mean_dir:.1f}%")

if len(alignment_df) > 0:
    print("\n\nPseudotime Alignment:")
    print(alignment_df.to_string(index=False))
    sig_aligned = (alignment_df['p_pt_temporal'] < 0.05).sum()
    print(f"\n  TFs with significant alignment: {sig_aligned}/{len(alignment_df)}")

print("\nDone!")
