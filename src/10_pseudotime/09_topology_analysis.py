"""
Phase 2: Trajectory topology analysis.

Systematically analyzes how trajectory complexity (linear vs branching vs cyclic)
affects directionality scores. Uses all available lineages from Phase 1 and Phase 2
multi-tissue analysis.

Topology metrics:
1. Linearity score: how linear is the trajectory? (geodesic vs Euclidean distance)
2. Branching complexity: eigenvalue gap in diffusion components
3. Pseudotime spread: variance of pseudotime distribution
4. Cell density uniformity: how evenly are cells distributed along pseudotime?
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
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
# TOPOLOGY METRIC FUNCTIONS
# ============================================================

def compute_linearity_score(adata_sub, n_sample=1000):
    """
    Linearity score: ratio of pseudotime distance to geodesic distance.
    High = trajectory is linear; Low = trajectory is branched/cyclic.
    Computed on PCA space for efficiency.
    """
    pt = adata_sub.obs['dpt_pseudotime'].values
    valid = np.isfinite(pt)
    pca = adata_sub.obsm['X_pca'][valid, :10]
    pt_v = pt[valid]

    # Sample for efficiency
    n = len(pt_v)
    if n > n_sample:
        idx = np.random.choice(n, n_sample, replace=False)
        pca = pca[idx]
        pt_v = pt_v[idx]

    # Pairwise pseudotime distances
    pt_dists = np.abs(pt_v[:, None] - pt_v[None, :])

    # Pairwise Euclidean distances in PCA space
    pca_dists = squareform(pdist(pca, 'euclidean'))

    # Linearity: correlation between pseudotime and PCA distance
    # High correlation = linear trajectory
    upper_tri = np.triu_indices(len(pt_v), k=1)
    pt_flat = pt_dists[upper_tri]
    pca_flat = pca_dists[upper_tri]

    if np.std(pt_flat) > 0 and np.std(pca_flat) > 0:
        r, _ = spearmanr(pt_flat, pca_flat)
    else:
        r = 0

    return r

def compute_branching_score(adata_sub):
    """
    Branching score based on eigenvalue gaps in diffusion components.
    A large gap between first and second eigenvalues suggests a linear trajectory;
    small gap suggests branching.
    """
    if 'X_diffmap' not in adata_sub.obsm:
        return np.nan

    dm = adata_sub.obsm['X_diffmap']
    # Variance explained by each component
    var_explained = np.var(dm, axis=0)
    var_explained = var_explained / var_explained.sum()

    # Eigenvalue gap (ratio of first to second component)
    if len(var_explained) >= 2 and var_explained[1] > 0:
        gap = var_explained[0] / var_explained[1]
    else:
        gap = np.inf

    return gap

def compute_pseudotime_uniformity(adata_sub, n_bins=50):
    """
    How uniformly are cells distributed along pseudotime?
    Returns coefficient of variation of bin counts.
    Low CV = uniform; High CV = clustered/bimodal.
    """
    pt = adata_sub.obs['dpt_pseudotime'].values
    valid = np.isfinite(pt)
    pt_v = pt[valid]

    bin_counts, _ = np.histogram(pt_v, bins=n_bins)
    # Ignore empty bins
    nonzero = bin_counts[bin_counts > 0]
    if len(nonzero) < 2 or np.mean(nonzero) == 0:
        return np.nan

    cv = np.std(nonzero) / np.mean(nonzero)
    return cv

def compute_pseudotime_entropy(adata_sub, n_bins=20):
    """
    Entropy of pseudotime distribution. High entropy = uniform spread.
    """
    pt = adata_sub.obs['dpt_pseudotime'].values
    valid = np.isfinite(pt)
    pt_v = pt[valid]

    counts, _ = np.histogram(pt_v, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(n_bins)

    return entropy / max_entropy  # Normalized entropy [0, 1]

def compute_cell_type_separation(adata_sub):
    """
    How well-separated are cell types along pseudotime?
    High separation = distinct stages; Low = intermixed.
    Uses silhouette-like score on pseudotime.
    """
    pt = adata_sub.obs['dpt_pseudotime'].values
    ct = adata_sub.obs['cell_type'].values
    valid = np.isfinite(pt)
    pt_v = pt[valid]
    ct_v = ct[valid]

    unique_ct = np.unique(ct_v)
    if len(unique_ct) < 2:
        return 0.0

    # Compute median pseudotime per cell type
    ct_medians = {}
    for c in unique_ct:
        ct_medians[c] = np.median(pt_v[ct_v == c])

    # Sort by median pseudotime
    sorted_cts = sorted(ct_medians.keys(), key=lambda x: ct_medians[x])

    # Compute overlap between adjacent cell types
    overlaps = []
    for i in range(len(sorted_cts) - 1):
        ct1 = sorted_cts[i]
        ct2 = sorted_cts[i + 1]
        pt1 = pt_v[ct_v == ct1]
        pt2 = pt_v[ct_v == ct2]

        # Overlap: fraction of cells that fall within the other type's range
        range_start = max(pt1.min(), pt2.min())
        range_end = min(pt1.max(), pt2.max())

        if range_end > range_start:
            n_overlap_1 = ((pt1 >= range_start) & (pt1 <= range_end)).sum() / len(pt1)
            n_overlap_2 = ((pt2 >= range_start) & (pt2 <= range_end)).sum() / len(pt2)
            overlaps.append(1 - (n_overlap_1 + n_overlap_2) / 2)
        else:
            overlaps.append(1.0)

    return np.mean(overlaps) if overlaps else 0.0

# ============================================================
# LOAD ALL LINEAGES
# ============================================================
print("=" * 60)
print("LOADING LINEAGE DATA")
print("=" * 60)

lineage_files = {
    'T_cell': 'adata_T_cell.h5ad',
    'B_cell': 'adata_B_cell.h5ad',
    'Myeloid': 'adata_Myeloid.h5ad',
    'Lung_Epithelial': 'adata_Lung_Epithelial.h5ad',
    'Lung_Myeloid': 'adata_Lung_Myeloid.h5ad',
    'Kidney_Epithelial': 'adata_Kidney_Epithelial.h5ad',
    'Kidney_Immune': 'adata_Kidney_Immune.h5ad',
}

lineage_data = {}
for name, fname in lineage_files.items():
    path = os.path.join(OUT_DIR, 'implementation', fname)
    if os.path.exists(path):
        lineage_data[name] = sc.read_h5ad(path)
        print(f"  {name}: {lineage_data[name].n_obs} cells")
    else:
        print(f"  {name}: NOT FOUND ({path})")

# ============================================================
# COMPUTE TOPOLOGY METRICS
# ============================================================
print("\n" + "=" * 60)
print("COMPUTING TOPOLOGY METRICS")
print("=" * 60)

topology_metrics = []

for name, adata_sub in lineage_data.items():
    print(f"\n{name}:")

    # Linearity
    linearity = compute_linearity_score(adata_sub)
    print(f"  Linearity score: {linearity:.3f}")

    # Branching
    branching = compute_branching_score(adata_sub)
    print(f"  Branching score (eigenvalue ratio): {branching:.3f}")

    # Uniformity
    uniformity = compute_pseudotime_uniformity(adata_sub)
    print(f"  Pseudotime CV: {uniformity:.3f}")

    # Entropy
    entropy = compute_pseudotime_entropy(adata_sub)
    print(f"  Pseudotime entropy (normalized): {entropy:.3f}")

    # Cell type separation
    separation = compute_cell_type_separation(adata_sub)
    print(f"  Cell type separation: {separation:.3f}")

    # Number of cell types
    n_cell_types = adata_sub.obs['cell_type'].nunique()

    # Pseudotime range and variance
    pt = adata_sub.obs['dpt_pseudotime'].values
    valid = np.isfinite(pt)
    pt_var = np.var(pt[valid])
    pt_range = np.ptp(pt[valid])

    # Tissue
    if name.startswith('Lung'):
        tissue = 'Lung'
    elif name.startswith('Kidney'):
        tissue = 'Kidney'
    else:
        tissue = 'Immune'

    topology_metrics.append({
        'Lineage': name,
        'Tissue': tissue,
        'n_cells': adata_sub.n_obs,
        'n_cell_types': n_cell_types,
        'linearity': linearity,
        'branching_score': branching,
        'pt_cv': uniformity,
        'pt_entropy': entropy,
        'cell_type_separation': separation,
        'pt_variance': pt_var,
        'pt_range': pt_range,
    })

topo_df = pd.DataFrame(topology_metrics)
print("\n\nTopology Metrics Summary:")
print(topo_df[['Lineage', 'Tissue', 'n_cells', 'n_cell_types',
                'linearity', 'branching_score', 'pt_cv', 'pt_entropy',
                'cell_type_separation']].to_string(index=False))

# ============================================================
# LOAD DIRECTIONALITY RESULTS AND CORRELATE
# ============================================================
print("\n" + "=" * 60)
print("CORRELATING TOPOLOGY WITH DIRECTIONALITY")
print("=" * 60)

# Load Phase 1 results
phase1_df = pd.read_csv(os.path.join(OUT_DIR, 'implementation', 'directionality_results.csv'))

# Load Phase 2 multi-tissue results if available
phase2_path = os.path.join(OUT_DIR, 'implementation', 'phase2_multi_tissue_results.csv')
if os.path.exists(phase2_path):
    phase2_df = pd.read_csv(phase2_path)
    all_results = pd.concat([phase1_df, phase2_df], ignore_index=True)
else:
    all_results = phase1_df

# Compute per-lineage directionality statistics
lineage_stats = []
for lin in all_results['Lineage'].unique():
    sub = all_results[all_results['Lineage'] == lin]
    consistency = sub['Directional_Consistent'].mean() * 100
    mean_ds = sub['Directionality_Score'].mean()
    median_ds = sub['Directionality_Score'].median()
    frac_pos = (sub['Directionality_Score'] > 0).mean() * 100
    n_pairs = len(sub)

    lineage_stats.append({
        'Lineage': lin,
        'n_pairs': n_pairs,
        'consistency_rate': consistency,
        'mean_ds': mean_ds,
        'median_ds': median_ds,
        'frac_positive': frac_pos,
    })

stats_df = pd.DataFrame(lineage_stats)

# Merge topology with directionality stats
merged = topo_df.merge(stats_df, on='Lineage', how='inner')
print(f"\nMerged {len(merged)} lineages with both topology and directionality data")

# Correlate topology metrics with directionality
if len(merged) >= 3:
    topology_cols = ['linearity', 'branching_score', 'pt_cv', 'pt_entropy',
                     'cell_type_separation', 'n_cell_types', 'n_cells']
    dir_cols = ['consistency_rate', 'mean_ds', 'frac_positive']

    print("\nCorrelations (Spearman):")
    corr_results = []
    for tc in topology_cols:
        for dc in dir_cols:
            valid_mask = np.isfinite(merged[tc]) & np.isfinite(merged[dc])
            if valid_mask.sum() >= 3:
                r, p = spearmanr(merged.loc[valid_mask, tc], merged.loc[valid_mask, dc])
                print(f"  {tc} vs {dc}: r={r:.3f}, p={p:.4f}")
                corr_results.append({'topology': tc, 'directionality': dc, 'r': r, 'p': p})

    corr_df = pd.DataFrame(corr_results)
else:
    print("\nNot enough lineages for correlation analysis")
    corr_df = pd.DataFrame()

# ============================================================
# WITHIN-LINEAGE TOPOLOGY ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("WITHIN-LINEAGE TOPOLOGY: LOCAL TRAJECTORY FEATURES")
print("=" * 60)

# For each TF-target pair, compute LOCAL trajectory features:
# - Local density (cells per bin around the TF/target peak)
# - Local curvature (second derivative of pseudotime trajectory)
# - Expression variability along pseudotime

local_results = []
N_BINS = 50

for _, row in all_results.iterrows():
    lin = row['Lineage']
    if lin not in lineage_data:
        continue

    adata_sub = lineage_data[lin]
    pt = adata_sub.obs['dpt_pseudotime'].values
    valid = np.isfinite(pt)
    pt_v = pt[valid]

    # Bin counts (density)
    bin_counts, bin_edges = np.histogram(pt_v, bins=N_BINS)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Local density around TF peak
    tf_peak_pt = row.get('TF_Peak_PT', np.nan)
    tgt_peak_pt = row.get('Target_Peak_PT', np.nan)

    if np.isfinite(tf_peak_pt):
        tf_bin = np.argmin(np.abs(bin_centers - tf_peak_pt))
        local_density_tf = bin_counts[max(0, tf_bin-2):min(N_BINS, tf_bin+3)].mean()
    else:
        local_density_tf = np.nan

    if np.isfinite(tgt_peak_pt):
        tgt_bin = np.argmin(np.abs(bin_centers - tgt_peak_pt))
        local_density_tgt = bin_counts[max(0, tgt_bin-2):min(N_BINS, tgt_bin+3)].mean()
    else:
        local_density_tgt = np.nan

    # Pseudotime span of the TF-target interaction
    if np.isfinite(tf_peak_pt) and np.isfinite(tgt_peak_pt):
        pt_span = abs(tgt_peak_pt - tf_peak_pt) / pt_v.ptp() if pt_v.ptp() > 0 else 0
    else:
        pt_span = np.nan

    local_results.append({
        'TF': row['TF'],
        'Target': row['Target'],
        'Lineage': lin,
        'Directional_Consistent': row['Directional_Consistent'],
        'Directionality_Score': row['Directionality_Score'],
        'local_density_tf': local_density_tf,
        'local_density_tgt': local_density_tgt,
        'pt_span': pt_span,
    })

local_df = pd.DataFrame(local_results)

if len(local_df) > 10:
    # Test: does local density predict consistency?
    valid_ld = local_df[np.isfinite(local_df['local_density_tf']) & np.isfinite(local_df['local_density_tgt'])]
    if len(valid_ld) > 5:
        avg_density = (valid_ld['local_density_tf'] + valid_ld['local_density_tgt']) / 2
        r_density, p_density = spearmanr(avg_density, valid_ld['Directionality_Score'])
        print(f"\nLocal density vs directionality score: r={r_density:.3f}, p={p_density:.4f}")

    # Test: does pseudotime span predict consistency?
    valid_span = local_df[np.isfinite(local_df['pt_span'])]
    if len(valid_span) > 5:
        r_span, p_span = spearmanr(valid_span['pt_span'], valid_span['Directionality_Score'])
        print(f"Pseudotime span vs directionality score: r={r_span:.3f}, p={p_span:.4f}")

        # Compare consistent vs inconsistent
        cons_span = valid_span[valid_span['Directional_Consistent']]['pt_span'].values
        incons_span = valid_span[~valid_span['Directional_Consistent']]['pt_span'].values
        if len(cons_span) > 0 and len(incons_span) > 0:
            print(f"  Consistent pairs: mean span={np.mean(cons_span):.3f}")
            print(f"  Inconsistent pairs: mean span={np.mean(incons_span):.3f}")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")

topo_df.to_csv(os.path.join(OUT_DIR, 'implementation', 'phase2_topology_metrics.csv'), index=False)

if len(merged) > 0:
    merged.to_csv(os.path.join(OUT_DIR, 'implementation', 'phase2_topology_directionality_merged.csv'), index=False)

if len(local_df) > 0:
    local_df.to_csv(os.path.join(OUT_DIR, 'implementation', 'phase2_local_topology.csv'), index=False)

with open(os.path.join(OUT_DIR, 'implementation', 'phase2_topology_full.pkl'), 'wb') as f:
    pickle.dump({
        'topo_df': topo_df,
        'merged': merged,
        'corr_df': corr_df,
        'local_df': local_df,
    }, f)

# ============================================================
# FIGURES
# ============================================================
print("\nGenerating figures...")

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

tissue_colors = {'Immune': '#E64B35', 'Lung': '#4DBBD5', 'Kidney': '#00A087'}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# (A) Linearity vs consistency rate
ax = axes[0, 0]
if len(merged) >= 3:
    for _, row in merged.iterrows():
        color = tissue_colors.get(row['Tissue'], 'gray')
        ax.scatter(row['linearity'], row['consistency_rate'], c=color, s=80,
                   edgecolors='white', linewidths=0.5, zorder=5)
        ax.annotate(row['Lineage'].replace('_', '\n'), (row['linearity'], row['consistency_rate']),
                    fontsize=6, ha='left', va='bottom')
    ax.set_xlabel('Linearity Score')
    ax.set_ylabel('Consistency Rate (%)')
    ax.set_title('(A) Linearity vs Consistency')

# (B) Cell type separation vs consistency
ax = axes[0, 1]
if len(merged) >= 3:
    for _, row in merged.iterrows():
        color = tissue_colors.get(row['Tissue'], 'gray')
        ax.scatter(row['cell_type_separation'], row['consistency_rate'], c=color, s=80,
                   edgecolors='white', linewidths=0.5, zorder=5)
        ax.annotate(row['Lineage'].replace('_', '\n'), (row['cell_type_separation'], row['consistency_rate']),
                    fontsize=6, ha='left', va='bottom')
    ax.set_xlabel('Cell Type Separation')
    ax.set_ylabel('Consistency Rate (%)')
    ax.set_title('(B) Cell Type Separation vs Consistency')

# (C) Number of cell types vs consistency
ax = axes[0, 2]
if len(merged) >= 3:
    for _, row in merged.iterrows():
        color = tissue_colors.get(row['Tissue'], 'gray')
        ax.scatter(row['n_cell_types'], row['consistency_rate'], c=color, s=80,
                   edgecolors='white', linewidths=0.5, zorder=5)
        ax.annotate(row['Lineage'].replace('_', '\n'), (row['n_cell_types'], row['consistency_rate']),
                    fontsize=6, ha='left', va='bottom')
    ax.set_xlabel('Number of Cell Types')
    ax.set_ylabel('Consistency Rate (%)')
    ax.set_title('(C) Cell Types vs Consistency')

# (D) Topology metrics heatmap
ax = axes[1, 0]
if len(topo_df) >= 2:
    metrics_for_heatmap = topo_df.set_index('Lineage')[['linearity', 'pt_entropy',
                                                          'cell_type_separation', 'pt_cv']]
    # Normalize columns
    metrics_norm = (metrics_for_heatmap - metrics_for_heatmap.mean()) / metrics_for_heatmap.std()
    sns.heatmap(metrics_norm, cmap='RdBu_r', center=0, ax=ax, cbar_kws={'shrink': 0.6},
                yticklabels=[l.replace('_', ' ') for l in metrics_norm.index],
                xticklabels=['Linearity', 'PT Entropy', 'CT Sep.', 'PT CV'])
    ax.set_title('(D) Topology Metrics (z-scored)')

# (E) Pseudotime span: consistent vs inconsistent
ax = axes[1, 1]
if len(local_df) > 5:
    valid_span = local_df[np.isfinite(local_df['pt_span'])]
    if len(valid_span) > 0:
        cons_data = valid_span[valid_span['Directional_Consistent']]['pt_span'].values
        incons_data = valid_span[~valid_span['Directional_Consistent']]['pt_span'].values

        parts = ax.violinplot([cons_data, incons_data] if len(cons_data) > 0 and len(incons_data) > 0
                              else [valid_span['pt_span'].values],
                              positions=[1, 2] if len(cons_data) > 0 and len(incons_data) > 0 else [1],
                              showmeans=True, showmedians=True)

        for pc in parts['bodies']:
            pc.set_facecolor('#4DBBD5')
            pc.set_alpha(0.5)

        if len(cons_data) > 0 and len(incons_data) > 0:
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Consistent', 'Inconsistent'])
        ax.set_ylabel('Pseudotime Span')
        ax.set_title('(E) TF-Target PT Span')

# (F) Pseudotime distribution shapes
ax = axes[1, 2]
for name, adata_sub in lineage_data.items():
    pt = adata_sub.obs['dpt_pseudotime'].values
    valid = np.isfinite(pt)
    pt_v = pt[valid]
    tissue = 'Immune' if name in ['T_cell', 'B_cell', 'Myeloid'] else ('Lung' if 'Lung' in name else 'Kidney')
    color = tissue_colors.get(tissue, 'gray')
    ax.hist(pt_v, bins=30, alpha=0.3, color=color, density=True, label=name.replace('_', ' '))
    ax.axvline(np.median(pt_v), color=color, linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel('Pseudotime')
ax.set_ylabel('Density')
ax.set_title('(F) Pseudotime Distributions')
ax.legend(fontsize=5, loc='upper right')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'phase2_fig7_topology_analysis.png'))
fig.savefig(os.path.join(FIG_DIR, 'phase2_fig7_topology_analysis.pdf'))
plt.close()

# ============================================================
# CORRELATION HEATMAP
# ============================================================
if len(corr_df) > 0:
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_pivot = corr_df.pivot(index='topology', columns='directionality', values='r')
    pval_pivot = corr_df.pivot(index='topology', columns='directionality', values='p')

    sns.heatmap(corr_pivot, cmap='RdBu_r', center=0, annot=True, fmt='.2f', ax=ax,
                vmin=-1, vmax=1, cbar_kws={'label': 'Spearman r'})

    # Mark significant correlations
    for i in range(corr_pivot.shape[0]):
        for j in range(corr_pivot.shape[1]):
            p = pval_pivot.iloc[i, j]
            if p < 0.05:
                ax.text(j + 0.5, i + 0.8, '*', ha='center', va='center', fontsize=14, color='gold')

    ax.set_title('Topology-Directionality Correlations (* p<0.05)')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'phase2_fig8_topology_corr_heatmap.png'))
    fig.savefig(os.path.join(FIG_DIR, 'phase2_fig8_topology_corr_heatmap.pdf'))
    plt.close()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("\nTopology Metrics by Lineage:")
print(topo_df[['Lineage', 'Tissue', 'n_cells', 'linearity', 'pt_entropy',
                'cell_type_separation']].to_string(index=False))

if len(merged) >= 3:
    print("\nTopology-Directionality Correlations:")
    if len(corr_df) > 0:
        sig_corrs = corr_df[corr_df['p'] < 0.1]
        if len(sig_corrs) > 0:
            print("Suggestive correlations (p < 0.1):")
            print(sig_corrs.to_string(index=False))
        else:
            print("  No significant correlations found (likely limited by number of lineages)")

    # Key finding
    if 'linearity' in merged.columns and 'consistency_rate' in merged.columns:
        r_lin, p_lin = spearmanr(merged['linearity'], merged['consistency_rate'])
        print(f"\nKey finding: linearity vs consistency: r={r_lin:.3f}, p={p_lin:.4f}")
        if r_lin > 0:
            print("  -> More linear trajectories tend to show higher directional consistency")
        else:
            print("  -> No clear relationship between linearity and consistency")

print("\nDone!")
