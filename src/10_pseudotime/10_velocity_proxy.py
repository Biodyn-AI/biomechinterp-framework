"""
Phase 2: RNA velocity proxy for directionality analysis.

scVelo is not available, so we implement a velocity proxy using expression
gradients along pseudotime. The idea: the local derivative of expression
along pseudotime provides an instantaneous rate-of-change that can be used
as a velocity proxy. If TF velocity (increasing) precedes target velocity
(increasing), this supports TF-leads-target directionality.

Two approaches:
1. Gradient-based velocity: dE/dpt for each gene, then correlate TF velocity
   with downstream target level (and vice versa)
2. Velocity cross-correlation: cross-correlate d(TF)/dpt with d(target)/dpt
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

np.random.seed(42)

OUT_DIR = '/Users/ihorkendiukhov/biodyn-work/subproject_29_pseudotime_directionality_audit'
FIG_DIR = os.path.join(OUT_DIR, 'reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
N_BINS = 50
MAX_LAG = 10

def get_expr(adata_sub, gene_name):
    mask = adata_sub.var['feature_name'] == gene_name
    if mask.sum() == 0:
        return None
    idx = np.where(mask.values)[0][0]
    x = adata_sub.X[:, idx]
    if sparse.issparse(x):
        return np.asarray(x.todense()).flatten()
    return np.asarray(x).flatten()

def bin_expression(adata_sub, gene_name, pt_values=None, n_bins=N_BINS):
    if pt_values is None:
        pt = adata_sub.obs['dpt_pseudotime'].values
    else:
        pt = pt_values
    valid = np.isfinite(pt)
    pt_v = pt[valid]
    expr = get_expr(adata_sub, gene_name)
    if expr is None:
        return None, None, None
    expr_v = expr[valid]
    bin_edges = np.linspace(pt_v.min(), pt_v.max() + 1e-10, n_bins + 1)
    bin_idx = np.clip(np.digitize(pt_v, bin_edges) - 1, 0, n_bins - 1)
    bin_means = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() > 0:
            bin_means[b] = expr_v[m].mean()
            bin_counts[b] = m.sum()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, bin_means, bin_counts

def bin_expression_fast(expr, pt, n_bins=N_BINS):
    valid = np.isfinite(pt)
    pt_v = pt[valid]
    expr_v = expr[valid]
    bin_edges = np.linspace(pt_v.min(), pt_v.max() + 1e-10, n_bins + 1)
    bin_idx = np.clip(np.digitize(pt_v, bin_edges) - 1, 0, n_bins - 1)
    bin_means = np.zeros(n_bins)
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() > 0:
            bin_means[b] = expr_v[m].mean()
    return bin_means

def compute_lagged_correlations(tf_bins, target_bins, max_lag=MAX_LAG):
    n = len(tf_bins)
    lags = list(range(-max_lag, max_lag + 1))
    corrs = []
    for lag in lags:
        if lag == 0:
            tf_s, tgt_s = tf_bins, target_bins
        elif lag > 0:
            tf_s = tf_bins[:n-lag]
            tgt_s = target_bins[lag:]
        else:
            tf_s = tf_bins[-lag:]
            tgt_s = target_bins[:n+lag]
        if len(tf_s) < 5 or np.std(tf_s) == 0 or np.std(tgt_s) == 0:
            corrs.append(0.0)
        else:
            r, _ = spearmanr(tf_s, tgt_s)
            corrs.append(r if not np.isnan(r) else 0.0)
    return lags, corrs

def directionality_score(lags, corrs, direction='activate'):
    lags_arr = np.array(lags)
    corrs_arr = np.array(corrs)
    pos_mask = lags_arr > 0
    neg_mask = lags_arr < 0
    if direction == 'activate':
        pos_s = np.mean(corrs_arr[pos_mask]) if pos_mask.any() else 0
        neg_s = np.mean(corrs_arr[neg_mask]) if neg_mask.any() else 0
    else:
        pos_s = np.mean(-corrs_arr[pos_mask]) if pos_mask.any() else 0
        neg_s = np.mean(-corrs_arr[neg_mask]) if neg_mask.any() else 0
    return pos_s - neg_s

# ============================================================
# VELOCITY PROXY FUNCTIONS
# ============================================================

def compute_velocity_proxy(binned_expr, sigma=2):
    """
    Compute velocity proxy as smoothed gradient of binned expression.
    sigma controls Gaussian smoothing bandwidth.
    Returns velocity array (same length as input, gradient approximation).
    """
    # Smooth to reduce noise
    smoothed = gaussian_filter1d(binned_expr, sigma=sigma)
    # Central difference gradient
    velocity = np.gradient(smoothed)
    return velocity

def velocity_directionality_score(tf_vel, tgt_vel, tf_bins, tgt_bins, direction='activate'):
    """
    Velocity-based directionality score.

    For activation: we expect TF velocity to peak (increase) before target velocity peaks.
    We compute: corr(TF_velocity_t, Target_velocity_{t+lag}) for positive lags.

    Also: cross-correlation between TF velocity and target expression level
    (TF rate of change should predict target level at later times).
    """
    # Method 1: Velocity cross-correlation
    lags_vel, corrs_vel = compute_lagged_correlations(tf_vel, tgt_vel)
    ds_vel = directionality_score(lags_vel, corrs_vel, direction)

    # Method 2: TF velocity -> Target level cross-correlation
    lags_mixed, corrs_mixed = compute_lagged_correlations(tf_vel, tgt_bins)
    ds_mixed = directionality_score(lags_mixed, corrs_mixed, direction)

    return ds_vel, ds_mixed, lags_vel, corrs_vel, lags_mixed, corrs_mixed

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("Loading lineage data...")
lineage_data = {}
for name in ['T_cell', 'B_cell', 'Myeloid']:
    path = os.path.join(OUT_DIR, 'implementation', f'adata_{name}.h5ad')
    if os.path.exists(path):
        lineage_data[name] = sc.read_h5ad(path)
        print(f"  {name}: {lineage_data[name].n_obs} cells")

# TF-target pairs (same as Phase 1)
tf_target_pairs = [
    ('TCF7', 'LEF1', 'T_cell', 'activate'), ('TCF7', 'CCR7', 'T_cell', 'activate'),
    ('TCF7', 'IL7R', 'T_cell', 'activate'), ('TBX21', 'IFNG', 'T_cell', 'activate'),
    ('TBX21', 'CXCR3', 'T_cell', 'activate'), ('TBX21', 'GZMB', 'T_cell', 'activate'),
    ('GATA3', 'IL4', 'T_cell', 'activate'), ('FOXP3', 'IL2RA', 'T_cell', 'activate'),
    ('FOXP3', 'CTLA4', 'T_cell', 'activate'), ('EOMES', 'PRF1', 'T_cell', 'activate'),
    ('EOMES', 'GZMB', 'T_cell', 'activate'), ('RUNX3', 'CD8A', 'T_cell', 'activate'),
    ('RUNX3', 'GZMB', 'T_cell', 'activate'), ('BATF', 'IL10', 'T_cell', 'activate'),
    ('RORC', 'IL17A', 'T_cell', 'activate'), ('RORC', 'CCR6', 'T_cell', 'activate'),
    ('ID2', 'GZMB', 'T_cell', 'activate'), ('MAF', 'IL4', 'T_cell', 'activate'),
    ('STAT4', 'TBX21', 'T_cell', 'activate'), ('STAT6', 'GATA3', 'T_cell', 'activate'),
    ('IRF4', 'IL4', 'T_cell', 'activate'), ('MYC', 'CDK4', 'T_cell', 'activate'),
    ('MYC', 'MCM7', 'T_cell', 'activate'), ('NFKB1', 'BCL2', 'T_cell', 'activate'),
    ('PAX5', 'CD19', 'B_cell', 'activate'), ('PAX5', 'CD79A', 'B_cell', 'activate'),
    ('PAX5', 'MS4A1', 'B_cell', 'activate'), ('EBF1', 'CD79A', 'B_cell', 'activate'),
    ('EBF1', 'PAX5', 'B_cell', 'activate'), ('IRF4', 'PRDM1', 'B_cell', 'activate'),
    ('PRDM1', 'XBP1', 'B_cell', 'activate'), ('PRDM1', 'SDC1', 'B_cell', 'activate'),
    ('PRDM1', 'PAX5', 'B_cell', 'repress'), ('BCL6', 'PRDM1', 'B_cell', 'repress'),
    ('BCL6', 'TP53', 'B_cell', 'repress'), ('IRF8', 'BCL6', 'B_cell', 'activate'),
    ('MYC', 'CDK4', 'B_cell', 'activate'), ('XBP1', 'JCHAIN', 'B_cell', 'activate'),
    ('NFKB1', 'BCL2', 'B_cell', 'activate'),
    ('SPI1', 'CSF1R', 'Myeloid', 'activate'), ('SPI1', 'CD68', 'Myeloid', 'activate'),
    ('SPI1', 'ITGAM', 'Myeloid', 'activate'), ('CEBPA', 'CSF3R', 'Myeloid', 'activate'),
    ('CEBPB', 'IL6', 'Myeloid', 'activate'), ('CEBPB', 'TNF', 'Myeloid', 'activate'),
    ('IRF8', 'IL12B', 'Myeloid', 'activate'), ('IRF4', 'CCL17', 'Myeloid', 'activate'),
    ('STAT1', 'IRF1', 'Myeloid', 'activate'), ('STAT1', 'GBP1', 'Myeloid', 'activate'),
    ('NFKB1', 'IL1B', 'Myeloid', 'activate'), ('NFKB1', 'TNF', 'Myeloid', 'activate'),
    ('RELA', 'CXCL8', 'Myeloid', 'activate'), ('MAF', 'MRC1', 'Myeloid', 'activate'),
    ('KLF4', 'MRC1', 'Myeloid', 'activate'), ('JUN', 'MMP9', 'Myeloid', 'activate'),
    ('FOS', 'MMP9', 'Myeloid', 'activate'),
]

# ============================================================
# ANALYSIS: STANDARD (DPT) vs VELOCITY-BASED DIRECTIONALITY
# ============================================================
print("\n" + "=" * 60)
print("VELOCITY-BASED DIRECTIONALITY ANALYSIS")
print("=" * 60)

results = []
for tf_name, target_name, lineage, direction in tf_target_pairs:
    if lineage not in lineage_data:
        continue
    adata_sub = lineage_data[lineage]

    bc, tf_bins, _ = bin_expression(adata_sub, tf_name)
    _, tgt_bins, _ = bin_expression(adata_sub, target_name)

    if tf_bins is None or tgt_bins is None:
        continue
    if np.max(tf_bins) < 0.01 or np.max(tgt_bins) < 0.01:
        continue

    # Standard DPT-based analysis
    lags_std, corrs_std = compute_lagged_correlations(tf_bins, tgt_bins)
    ds_std = directionality_score(lags_std, corrs_std, direction)

    lags_arr = np.array(lags_std)
    corrs_arr = np.array(corrs_std)
    if direction == 'activate':
        best_idx = np.argmax(corrs_arr)
    else:
        best_idx = np.argmin(corrs_arr)
    peak_lag_std = int(lags_arr[best_idx])
    peak_corr_std = corrs_arr[best_idx]

    if direction == 'activate':
        dc_std = peak_lag_std > 0 and peak_corr_std > 0
    else:
        dc_std = peak_lag_std > 0 and peak_corr_std < 0

    # Velocity-based analysis
    tf_vel = compute_velocity_proxy(tf_bins, sigma=2)
    tgt_vel = compute_velocity_proxy(tgt_bins, sigma=2)

    ds_vel, ds_mixed, lags_vel, corrs_vel, lags_mixed, corrs_mixed = \
        velocity_directionality_score(tf_vel, tgt_vel, tf_bins, tgt_bins, direction)

    # Velocity peak lag
    lags_v_arr = np.array(lags_vel)
    corrs_v_arr = np.array(corrs_vel)
    if direction == 'activate':
        best_v_idx = np.argmax(corrs_v_arr)
    else:
        best_v_idx = np.argmin(corrs_v_arr)
    peak_lag_vel = int(lags_v_arr[best_v_idx])
    peak_corr_vel = corrs_v_arr[best_v_idx]

    if direction == 'activate':
        dc_vel = peak_lag_vel > 0 and peak_corr_vel > 0
    else:
        dc_vel = peak_lag_vel > 0 and peak_corr_vel < 0

    # Mixed (TF velocity -> target expression) peak lag
    lags_m_arr = np.array(lags_mixed)
    corrs_m_arr = np.array(corrs_mixed)
    if direction == 'activate':
        best_m_idx = np.argmax(corrs_m_arr)
    else:
        best_m_idx = np.argmin(corrs_m_arr)
    peak_lag_mixed = int(lags_m_arr[best_m_idx])
    peak_corr_mixed = corrs_m_arr[best_m_idx]

    if direction == 'activate':
        dc_mixed = peak_lag_mixed > 0 and peak_corr_mixed > 0
    else:
        dc_mixed = peak_lag_mixed > 0 and peak_corr_mixed < 0

    results.append({
        'TF': tf_name, 'Target': target_name, 'Lineage': lineage,
        'Direction': direction,
        # Standard DPT
        'Peak_Lag_DPT': peak_lag_std, 'Peak_Corr_DPT': peak_corr_std,
        'DS_DPT': ds_std, 'Consistent_DPT': dc_std,
        # Velocity cross-correlation
        'Peak_Lag_Vel': peak_lag_vel, 'Peak_Corr_Vel': peak_corr_vel,
        'DS_Vel': ds_vel, 'Consistent_Vel': dc_vel,
        # Mixed (TF velocity -> target level)
        'Peak_Lag_Mixed': peak_lag_mixed, 'Peak_Corr_Mixed': peak_corr_mixed,
        'DS_Mixed': ds_mixed, 'Consistent_Mixed': dc_mixed,
        # Raw data for plotting
        'TF_Binned': tf_bins.tolist(), 'Target_Binned': tgt_bins.tolist(),
        'TF_Velocity': tf_vel.tolist(), 'Target_Velocity': tgt_vel.tolist(),
        'Bin_Centers': bc.tolist(),
        'Lags_DPT': lags_std, 'Corrs_DPT': corrs_std,
        'Lags_Vel': lags_vel, 'Corrs_Vel': corrs_vel,
        'Lags_Mixed': lags_mixed, 'Corrs_Mixed': corrs_mixed,
    })

df = pd.DataFrame(results)
print(f"\nAnalyzed {len(df)} pairs")

# ============================================================
# COMPARE METHODS
# ============================================================
print("\n" + "=" * 60)
print("METHOD COMPARISON")
print("=" * 60)

methods = {
    'DPT (standard)': ('Consistent_DPT', 'DS_DPT'),
    'Velocity x-corr': ('Consistent_Vel', 'DS_Vel'),
    'TF velocity -> Target level': ('Consistent_Mixed', 'DS_Mixed'),
}

for method_name, (cons_col, ds_col) in methods.items():
    n_consistent = df[cons_col].sum()
    rate = df[cons_col].mean() * 100
    mean_ds = df[ds_col].mean()
    frac_pos = (df[ds_col] > 0).mean() * 100
    print(f"\n{method_name}:")
    print(f"  Consistent: {n_consistent}/{len(df)} ({rate:.1f}%)")
    print(f"  Mean score: {mean_ds:.4f}")
    print(f"  Fraction positive: {frac_pos:.1f}%")

    for lin in ['T_cell', 'B_cell', 'Myeloid']:
        sub = df[df['Lineage'] == lin]
        nc = sub[cons_col].sum()
        print(f"    {lin}: {nc}/{len(sub)} ({nc/len(sub)*100:.1f}%)")

# ============================================================
# AGREEMENT ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("AGREEMENT BETWEEN METHODS")
print("=" * 60)

# How often do methods agree?
agree_dpt_vel = (df['Consistent_DPT'] == df['Consistent_Vel']).mean() * 100
agree_dpt_mixed = (df['Consistent_DPT'] == df['Consistent_Mixed']).mean() * 100
agree_vel_mixed = (df['Consistent_Vel'] == df['Consistent_Mixed']).mean() * 100
agree_all = ((df['Consistent_DPT'] == df['Consistent_Vel']) & (df['Consistent_Vel'] == df['Consistent_Mixed'])).mean() * 100

print(f"DPT vs Velocity: {agree_dpt_vel:.1f}% agreement")
print(f"DPT vs Mixed: {agree_dpt_mixed:.1f}% agreement")
print(f"Velocity vs Mixed: {agree_vel_mixed:.1f}% agreement")
print(f"All three agree: {agree_all:.1f}%")

# Cases where velocity improves over DPT
improved = (~df['Consistent_DPT'] & df['Consistent_Vel']).sum()
worsened = (df['Consistent_DPT'] & ~df['Consistent_Vel']).sum()
print(f"\nVelocity improves over DPT: {improved} pairs")
print(f"Velocity worsens vs DPT: {worsened} pairs")

if improved > 0:
    print("\nPairs improved by velocity:")
    for _, row in df[~df['Consistent_DPT'] & df['Consistent_Vel']].iterrows():
        print(f"  {row['TF']}->{row['Target']} ({row['Lineage']}): DPT lag={row['Peak_Lag_DPT']}, Vel lag={row['Peak_Lag_Vel']}")

# Score correlation between methods
from scipy.stats import pearsonr
r_dpt_vel, p_dpt_vel = pearsonr(df['DS_DPT'], df['DS_Vel'])
r_dpt_mixed, p_dpt_mixed = pearsonr(df['DS_DPT'], df['DS_Mixed'])
print(f"\nScore correlations:")
print(f"  DPT vs Velocity: r={r_dpt_vel:.3f}, p={p_dpt_vel:.4f}")
print(f"  DPT vs Mixed: r={r_dpt_mixed:.3f}, p={p_dpt_mixed:.4f}")

# ============================================================
# NULL MODEL FOR VELOCITY
# ============================================================
print("\n" + "=" * 60)
print("NULL MODEL: Shuffled pseudotime for velocity method")
print("=" * 60)

N_PERM = 500
pair_exprs = {}
for idx, row in df.iterrows():
    adata_sub = lineage_data[row['Lineage']]
    tf_expr = get_expr(adata_sub, row['TF'])
    tgt_expr = get_expr(adata_sub, row['Target'])
    pt = adata_sub.obs['dpt_pseudotime'].values.copy()
    pair_exprs[idx] = (tf_expr, tgt_expr, pt)

null_scores_vel = []
for perm_i in range(N_PERM):
    if perm_i % 100 == 0:
        print(f"  Permutation {perm_i}/{N_PERM}...")
    perm_scores = []
    for idx, row in df.iterrows():
        tf_expr, tgt_expr, pt = pair_exprs[idx]
        valid = np.isfinite(pt)
        pt_shuf = pt.copy()
        pt_shuf[valid] = np.random.permutation(pt_shuf[valid])
        tf_bins = bin_expression_fast(tf_expr, pt_shuf)
        tgt_bins = bin_expression_fast(tgt_expr, pt_shuf)
        tf_vel = compute_velocity_proxy(tf_bins, sigma=2)
        tgt_vel = compute_velocity_proxy(tgt_bins, sigma=2)
        lags, corrs = compute_lagged_correlations(tf_vel, tgt_vel)
        ds = directionality_score(lags, corrs, row['Direction'])
        perm_scores.append(ds)
    null_scores_vel.append(np.mean(perm_scores))

null_scores_vel = np.array(null_scores_vel)
observed_vel = df['DS_Vel'].mean()
p_vel = (np.sum(null_scores_vel >= observed_vel) + 1) / (N_PERM + 1)
print(f"\nVelocity method:")
print(f"  Observed mean: {observed_vel:.4f}")
print(f"  Null mean: {null_scores_vel.mean():.4f}")
print(f"  P-value: {p_vel:.4f}")
if null_scores_vel.std() > 0:
    print(f"  Cohen's d: {(observed_vel - null_scores_vel.mean()) / null_scores_vel.std():.3f}")

# Compare to DPT null
# Load Phase 1 null for comparison
try:
    with open(os.path.join(OUT_DIR, 'implementation', 'full_results.pkl'), 'rb') as f:
        phase1_data = pickle.load(f)
    null_global_dpt = phase1_data['null_global']
    p_dpt = phase1_data['p_global']
    observed_dpt = df['DS_DPT'].mean()
    print(f"\nDPT method (for comparison):")
    print(f"  Observed mean: {observed_dpt:.4f}")
    print(f"  P-value: {p_dpt:.4f}")
except Exception:
    pass

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")

df_save = df.drop(columns=['TF_Binned', 'Target_Binned', 'TF_Velocity', 'Target_Velocity',
                            'Bin_Centers', 'Lags_DPT', 'Corrs_DPT', 'Lags_Vel', 'Corrs_Vel',
                            'Lags_Mixed', 'Corrs_Mixed'])
df_save.to_csv(os.path.join(OUT_DIR, 'implementation', 'phase2_velocity_results.csv'), index=False)

with open(os.path.join(OUT_DIR, 'implementation', 'phase2_velocity_full.pkl'), 'wb') as f:
    pickle.dump({
        'results_df': df,
        'null_scores_vel': null_scores_vel,
        'p_vel': p_vel,
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

# Figure 1: Method comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# (A) Consistency rates
ax = axes[0, 0]
method_names = ['DPT\n(standard)', 'Velocity\nx-corr', 'TF vel ->\nTarget level']
cons_cols = ['Consistent_DPT', 'Consistent_Vel', 'Consistent_Mixed']
rates = [df[c].mean() * 100 for c in cons_cols]
colors = ['#E64B35', '#4DBBD5', '#00A087']
bars = ax.bar(range(3), rates, color=colors, alpha=0.7)
ax.set_xticks(range(3))
ax.set_xticklabels(method_names)
ax.set_ylabel('Consistency Rate (%)')
ax.set_title('(A) Consistency by Method')
ax.set_ylim(0, 60)
ax.axhline(50, color='gray', linestyle='--', alpha=0.3)
for i, (r, bar) in enumerate(zip(rates, bars)):
    n = df[cons_cols[i]].sum()
    ax.text(bar.get_x() + bar.get_width()/2, r + 1.5, f'{r:.1f}%\n({n}/{len(df)})',
            ha='center', fontsize=8)

# (B) Score distributions
ax = axes[0, 1]
ds_cols = ['DS_DPT', 'DS_Vel', 'DS_Mixed']
ds_data = [df[c].values for c in ds_cols]
bp = ax.boxplot(ds_data, labels=['DPT', 'Velocity', 'Mixed'], patch_artist=True, widths=0.5)
for i, color in enumerate(colors):
    bp['boxes'][i].set_facecolor(color)
    bp['boxes'][i].set_alpha(0.5)
    x = np.random.normal(i+1, 0.05, len(ds_data[i]))
    ax.scatter(x, ds_data[i], c=color, s=10, alpha=0.4, zorder=5)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Directionality Score')
ax.set_title('(B) Score Distributions')

# (C) DPT vs Velocity scatter
ax = axes[0, 2]
ax.scatter(df['DS_DPT'], df['DS_Vel'], c='#4DBBD5', s=30, alpha=0.6, edgecolors='white', linewidths=0.5)
lim = max(abs(df['DS_DPT'].max()), abs(df['DS_Vel'].max()), abs(df['DS_DPT'].min()), abs(df['DS_Vel'].min())) + 0.05
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3)
ax.axhline(0, color='gray', linestyle='--', alpha=0.2)
ax.axvline(0, color='gray', linestyle='--', alpha=0.2)
ax.set_xlabel('DPT Directionality Score')
ax.set_ylabel('Velocity Directionality Score')
ax.set_title(f'(C) DPT vs Velocity (r={r_dpt_vel:.2f})')

# (D-F) Example pairs showing velocity profiles
examples = [
    ('TBX21', 'IFNG', 'T_cell', 'T-bet -> IFN-g'),
    ('BCL6', 'PRDM1', 'B_cell', 'BCL6 -| Blimp1'),
    ('SPI1', 'CSF1R', 'Myeloid', 'PU.1 -> CSF1R'),
]

for i, (tf, tgt, lin, label) in enumerate(examples):
    row = df[(df['TF'] == tf) & (df['Target'] == tgt) & (df['Lineage'] == lin)]
    if len(row) == 0:
        continue
    row = row.iloc[0]

    ax = axes[1, i]
    bc = np.array(row['Bin_Centers'])
    tf_b = np.array(row['TF_Binned'])
    tgt_b = np.array(row['Target_Binned'])
    tf_v = np.array(row['TF_Velocity'])
    tgt_v = np.array(row['Target_Velocity'])

    # Normalize for display
    tf_n = tf_b / np.max(tf_b) if np.max(tf_b) > 0 else tf_b
    tgt_n = tgt_b / np.max(tgt_b) if np.max(tgt_b) > 0 else tgt_b

    ax.plot(bc, tf_n, color='#E64B35', linewidth=2, label=f'{tf} expr', alpha=0.7)
    ax.plot(bc, tgt_n, color='#4DBBD5', linewidth=2, label=f'{tgt} expr', alpha=0.7)

    # Velocity (rescaled)
    ax2 = ax.twinx()
    vel_max = max(np.max(np.abs(tf_v)), np.max(np.abs(tgt_v)))
    if vel_max > 0:
        ax2.plot(bc, tf_v / vel_max, color='#E64B35', linewidth=1.5, linestyle='--',
                 label=f'{tf} vel', alpha=0.6)
        ax2.plot(bc, tgt_v / vel_max, color='#4DBBD5', linewidth=1.5, linestyle='--',
                 label=f'{tgt} vel', alpha=0.6)
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.3)
    ax2.set_ylabel('Velocity (norm)', fontsize=7)
    ax2.tick_params(axis='y', labelsize=6)

    ax.set_xlabel('Pseudotime')
    ax.set_ylabel('Expression (norm)')
    ax.set_title(f'({chr(68+i)}) {label}\nDPT:{row["DS_DPT"]:.3f} Vel:{row["DS_Vel"]:.3f}')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='best')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'phase2_fig4_velocity_comparison.png'))
fig.savefig(os.path.join(FIG_DIR, 'phase2_fig4_velocity_comparison.pdf'))
plt.close()

# Figure 2: Null model comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(null_scores_vel, bins=30, alpha=0.6, color='#4DBBD5', label='Velocity null', edgecolor='white')
ax.axvline(observed_vel, color='red', linewidth=2, label=f'Observed (p={p_vel:.3f})')
ax.set_xlabel('Mean Velocity Directionality Score')
ax.set_ylabel('Count (permutations)')
ax.set_title('(A) Velocity Method vs Shuffled Null')
ax.legend()

# Per-lineage comparison of methods
ax = axes[1]
lineage_order = ['T_cell', 'B_cell', 'Myeloid']
lineage_labels = {'T_cell': 'T cell', 'B_cell': 'B cell', 'Myeloid': 'Myeloid'}
x = np.arange(3)
width = 0.25
for j, (method, (cons_col, ds_col)) in enumerate(methods.items()):
    rates_lin = []
    for lin in lineage_order:
        sub = df[df['Lineage'] == lin]
        rates_lin.append(sub[cons_col].mean() * 100)
    ax.bar(x + j*width, rates_lin, width, label=method, color=colors[j], alpha=0.7)
ax.set_xticks(x + width)
ax.set_xticklabels([lineage_labels[l] for l in lineage_order])
ax.set_ylabel('Consistency Rate (%)')
ax.set_title('(B) Consistency by Lineage and Method')
ax.legend(fontsize=7)
ax.set_ylim(0, 60)
ax.axhline(50, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'phase2_fig5_velocity_null_lineage.png'))
fig.savefig(os.path.join(FIG_DIR, 'phase2_fig5_velocity_null_lineage.pdf'))
plt.close()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

cols = ['TF', 'Target', 'Lineage', 'DS_DPT', 'Consistent_DPT', 'DS_Vel', 'Consistent_Vel', 'DS_Mixed', 'Consistent_Mixed']
print(df[cols].to_string(index=False))

print(f"\n\nMETHOD COMPARISON:")
for method_name, (cons_col, ds_col) in methods.items():
    rate = df[cons_col].mean() * 100
    mean_ds = df[ds_col].mean()
    print(f"  {method_name}: {rate:.1f}% consistent, mean_score={mean_ds:.4f}")

print(f"\nVelocity null model p-value: {p_vel:.4f}")
print(f"Score correlation (DPT vs Vel): r={r_dpt_vel:.3f}")
print("\nDone!")
