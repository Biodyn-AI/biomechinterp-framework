"""
Step 3: Generate all figures for the paper.
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
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

OUT_DIR = '/Users/ihorkendiukhov/biodyn-work/subproject_29_pseudotime_directionality_audit'
FIG_DIR = os.path.join(OUT_DIR, 'reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Load results
with open(os.path.join(OUT_DIR, 'implementation', 'full_results.pkl'), 'rb') as f:
    data = pickle.load(f)

df = data['results_df']
null_global = data['null_global']
null_random = data['null_random']
p_global = data['p_global']
p_random = data['p_random']

# Load lineage data
lineage_data = {}
for name in ['T_cell', 'B_cell', 'Myeloid']:
    lineage_data[name] = sc.read_h5ad(os.path.join(OUT_DIR, 'implementation', f'adata_{name}.h5ad'))

# Plotting style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

lineage_colors = {'T_cell': '#E64B35', 'B_cell': '#4DBBD5', 'Myeloid': '#00A087'}
lineage_labels = {'T_cell': 'T cell', 'B_cell': 'B cell', 'Myeloid': 'Myeloid'}

# ============================================================
# FIGURE 1: UMAP + Pseudotime trajectories
# ============================================================
print("Figure 1: UMAP + Pseudotime...")
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

for i, (name, label) in enumerate(lineage_labels.items()):
    adata_sub = lineage_data[name]

    # Top row: cell type coloring
    ax = axes[0, i]
    ct = adata_sub.obs['cell_type'].astype('category')
    umap = adata_sub.obsm['X_umap']
    cats = ct.cat.categories
    colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))
    for j, c in enumerate(cats):
        mask = ct == c
        ax.scatter(umap[mask, 0], umap[mask, 1], s=1, alpha=0.5, color=colors[j],
                   label=c[:30], rasterized=True)
    ax.set_title(f'{label} lineage')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.legend(fontsize=5, loc='best', markerscale=4, framealpha=0.7)

    # Bottom row: pseudotime coloring
    ax = axes[1, i]
    pt = adata_sub.obs['dpt_pseudotime'].values
    valid = np.isfinite(pt)
    sc_plot = ax.scatter(umap[valid, 0], umap[valid, 1], c=pt[valid], s=1,
                         cmap='viridis', alpha=0.5, rasterized=True)
    ax.set_title(f'{label}: Pseudotime')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    plt.colorbar(sc_plot, ax=ax, label='DPT', shrink=0.7)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig1_umap_pseudotime.png'))
fig.savefig(os.path.join(FIG_DIR, 'fig1_umap_pseudotime.pdf'))
plt.close()

# ============================================================
# FIGURE 2: Example TF-target expression along pseudotime
# ============================================================
print("Figure 2: Example TF-target expression profiles...")

# Select interesting examples (one consistent, one inconsistent per lineage)
examples = [
    # (TF, Target, Lineage, label)
    ('TBX21', 'IFNG', 'T_cell', 'T-bet -> IFN-g (consistent)'),
    ('RORC', 'IL17A', 'T_cell', 'RORgt -> IL-17 (inconsistent)'),
    ('BCL6', 'PRDM1', 'B_cell', 'BCL6 -| Blimp1 (consistent)'),
    ('IRF4', 'PRDM1', 'B_cell', 'IRF4 -> Blimp1 (inconsistent)'),
    ('SPI1', 'CSF1R', 'Myeloid', 'PU.1 -> CSF1R (consistent)'),
    ('FOS', 'MMP9', 'Myeloid', 'Fos -> MMP9 (inconsistent)'),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes_flat = axes.flatten()

for i, (tf, tgt, lin, label) in enumerate(examples):
    row = df[(df['TF'] == tf) & (df['Target'] == tgt) & (df['Lineage'] == lin)]
    if len(row) == 0:
        continue
    row = row.iloc[0]

    ax = axes_flat[i]
    bc = np.array(row['Bin_Centers'])
    tf_b = np.array(row['TF_Binned'])
    tgt_b = np.array(row['Target_Binned'])

    # Normalize for visual comparison
    if np.max(tf_b) > 0:
        tf_n = tf_b / np.max(tf_b)
    else:
        tf_n = tf_b
    if np.max(tgt_b) > 0:
        tgt_n = tgt_b / np.max(tgt_b)
    else:
        tgt_n = tgt_b

    ax.plot(bc, tf_n, color='#E64B35', linewidth=2, label=f'{tf} (TF)')
    ax.plot(bc, tgt_n, color='#4DBBD5', linewidth=2, label=f'{tgt} (target)')

    # Mark peaks
    ax.axvline(bc[np.argmax(tf_b)], color='#E64B35', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(bc[np.argmax(tgt_b)], color='#4DBBD5', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_title(label, fontsize=9)
    ax.set_xlabel('Pseudotime')
    ax.set_ylabel('Normalized expression')
    ax.legend(fontsize=7, loc='best')
    ax.set_xlim(bc.min(), bc.max())

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig2_expression_profiles.png'))
fig.savefig(os.path.join(FIG_DIR, 'fig2_expression_profiles.pdf'))
plt.close()

# ============================================================
# FIGURE 3: Lagged correlation heatmaps
# ============================================================
print("Figure 3: Lagged correlation heatmaps...")

fig, axes = plt.subplots(1, 3, figsize=(16, 7))

for ax_idx, lin in enumerate(['T_cell', 'B_cell', 'Myeloid']):
    sub = df[df['Lineage'] == lin].copy()
    sub = sub.sort_values('Directionality_Score', ascending=False)

    # Build correlation matrix
    n_pairs = len(sub)
    lags = np.array(sub.iloc[0]['Lags'])
    n_lags = len(lags)
    corr_matrix = np.zeros((n_pairs, n_lags))
    labels = []

    for i, (_, row) in enumerate(sub.iterrows()):
        corr_matrix[i] = row['Correlations']
        sym = '-|' if row['Direction'] == 'repress' else '->'
        dc = '*' if row['Directional_Consistent'] else ''
        labels.append(f"{row['TF']} {sym} {row['Target']}{dc}")

    ax = axes[ax_idx]
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1,
                   interpolation='nearest')
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xticks(range(0, n_lags, 5))
    ax.set_xticklabels(lags[::5])
    ax.set_xlabel('Lag (bins)')
    ax.set_title(f'{lineage_labels[lin]} lineage')
    ax.axvline(x=n_lags//2, color='white', linestyle='--', alpha=0.8, linewidth=1)
    plt.colorbar(im, ax=ax, label='Spearman r', shrink=0.5)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig3_lagged_correlation_heatmaps.png'))
fig.savefig(os.path.join(FIG_DIR, 'fig3_lagged_correlation_heatmaps.pdf'))
plt.close()

# ============================================================
# FIGURE 4: Directionality score distribution + null models
# ============================================================
print("Figure 4: Directionality scores and null models...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (A) Distribution of directionality scores by lineage
ax = axes[0]
for lin, color in lineage_colors.items():
    sub = df[df['Lineage'] == lin]
    ax.hist(sub['Directionality_Score'], bins=15, alpha=0.5, color=color,
            label=lineage_labels[lin], edgecolor='white')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.axvline(df['Directionality_Score'].mean(), color='red', linestyle='-', linewidth=2,
           label=f'Mean={df["Directionality_Score"].mean():.3f}')
ax.set_xlabel('Directionality Score')
ax.set_ylabel('Count')
ax.set_title('(A) Directionality Score Distribution')
ax.legend(fontsize=8)

# (B) Observed vs shuffled pseudotime null
ax = axes[1]
ax.hist(null_global, bins=30, alpha=0.6, color='gray', label='Shuffled PT null', edgecolor='white')
ax.axvline(df['Directionality_Score'].mean(), color='red', linewidth=2,
           label=f'Observed (p={p_global:.3f})')
ax.set_xlabel('Mean Directionality Score')
ax.set_ylabel('Count (permutations)')
ax.set_title('(B) vs Shuffled Pseudotime Null')
ax.legend(fontsize=8)

# (C) Observed vs random pairs null
ax = axes[2]
ax.hist(null_random, bins=30, alpha=0.6, color='gray', label='Random pairs null', edgecolor='white')
ax.axvline(df['Directionality_Score'].mean(), color='red', linewidth=2,
           label=f'Observed (p={p_random:.3f})')
ax.set_xlabel('Mean Directionality Score')
ax.set_ylabel('Count (permutations)')
ax.set_title('(C) vs Random Gene Pairs Null')
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig4_directionality_null_models.png'))
fig.savefig(os.path.join(FIG_DIR, 'fig4_directionality_null_models.pdf'))
plt.close()

# ============================================================
# FIGURE 5: Failure mode analysis
# ============================================================
print("Figure 5: Failure mode analysis...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (A) Pie chart of consistency
ax = axes[0]
consistent = df['Directional_Consistent'].sum()
inconsistent = len(df) - consistent
ax.pie([consistent, inconsistent], labels=[f'Consistent\n({consistent})',
       f'Inconsistent\n({inconsistent})'],
       colors=['#00A087', '#E64B35'], autopct='%1.1f%%', startangle=90,
       textprops={'fontsize': 10})
ax.set_title('(A) Lag-based Directionality\nConsistency')

# (B) Failure categories
ax = axes[1]
failed = df[~df['Directional_Consistent']]
cats = []
for _, row in failed.iterrows():
    if abs(row['Peak_Corr']) < 0.1:
        cats.append('Weak signal')
    elif row['Peak_Lag'] < 0:
        cats.append('Target leads')
    elif row['Peak_Lag'] == 0:
        cats.append('Simultaneous')
    elif (row['Direction'] == 'activate' and row['Peak_Corr'] < 0) or \
         (row['Direction'] == 'repress' and row['Peak_Corr'] > 0):
        cats.append('Wrong sign')
    else:
        cats.append('Other')

cat_counts = pd.Series(cats).value_counts()
cat_colors = {'Simultaneous': '#4DBBD5', 'Target leads': '#E64B35',
              'Weak signal': '#F39B7F', 'Wrong sign': '#8491B4', 'Other': '#91D1C2'}
bars = ax.bar(range(len(cat_counts)), cat_counts.values,
              color=[cat_colors.get(c, 'gray') for c in cat_counts.index])
ax.set_xticks(range(len(cat_counts)))
ax.set_xticklabels(cat_counts.index, rotation=30, ha='right')
ax.set_ylabel('Count')
ax.set_title('(B) Failure Mode Categories')

# (C) Peak lag distribution
ax = axes[2]
for lin, color in lineage_colors.items():
    sub = df[df['Lineage'] == lin]
    ax.scatter(sub['Peak_Lag'], sub['Peak_Corr'], c=color, s=50, alpha=0.7,
               label=lineage_labels[lin], edgecolors='white', linewidths=0.5)
ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Peak Lag (bins)')
ax.set_ylabel('Peak Correlation')
ax.set_title('(C) Peak Lag vs Correlation')
ax.legend(fontsize=8)

# Shade the "consistent" quadrant
ax.axvspan(0.5, 11, alpha=0.05, color='green')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig5_failure_analysis.png'))
fig.savefig(os.path.join(FIG_DIR, 'fig5_failure_analysis.pdf'))
plt.close()

# ============================================================
# FIGURE 6: Per-lineage comparison and summary
# ============================================================
print("Figure 6: Lineage comparison...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (A) Directionality score by lineage
ax = axes[0]
lineage_scores = []
for lin in ['T_cell', 'B_cell', 'Myeloid']:
    sub = df[df['Lineage'] == lin]
    lineage_scores.append(sub['Directionality_Score'].values)

bp = ax.boxplot(lineage_scores, labels=[lineage_labels[l] for l in ['T_cell', 'B_cell', 'Myeloid']],
                patch_artist=True, widths=0.5)
for i, (lin, color) in enumerate(lineage_colors.items()):
    bp['boxes'][i].set_facecolor(color)
    bp['boxes'][i].set_alpha(0.5)

# Overlay individual points
for i, (lin, scores) in enumerate(zip(['T_cell', 'B_cell', 'Myeloid'], lineage_scores)):
    x = np.random.normal(i+1, 0.05, len(scores))
    ax.scatter(x, scores, c=lineage_colors[lin], s=20, alpha=0.6, zorder=5)

ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Directionality Score')
ax.set_title('(A) Directionality Score by Lineage')

# (B) Consistency rate by lineage
ax = axes[1]
lin_order = ['T_cell', 'B_cell', 'Myeloid']
rates = []
for lin in lin_order:
    sub = df[df['Lineage'] == lin]
    rates.append(sub['Directional_Consistent'].mean() * 100)

bars = ax.bar(range(3), rates, color=[lineage_colors[l] for l in lin_order], alpha=0.7,
              edgecolor='white')
ax.set_xticks(range(3))
ax.set_xticklabels([lineage_labels[l] for l in lin_order])
ax.set_ylabel('Consistency Rate (%)')
ax.set_title('(B) Lag-based Consistency Rate')
ax.set_ylim(0, 100)
ax.axhline(50, color='gray', linestyle='--', alpha=0.3, label='Chance')
ax.legend()

# Add count annotations
for i, (rate, bar) in enumerate(zip(rates, bars)):
    sub = df[df['Lineage'] == lin_order[i]]
    n = sub['Directional_Consistent'].sum()
    total = len(sub)
    ax.text(bar.get_x() + bar.get_width()/2, rate + 2, f'{n}/{total}',
            ha='center', fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig6_lineage_comparison.png'))
fig.savefig(os.path.join(FIG_DIR, 'fig6_lineage_comparison.pdf'))
plt.close()

# ============================================================
# FIGURE 7 (Supplementary): Full results heatmap
# ============================================================
print("Figure 7: Summary heatmap...")

fig, ax = plt.subplots(figsize=(10, 14))

# Sort by lineage then by directionality score
df_sorted = df.sort_values(['Lineage', 'Directionality_Score'], ascending=[True, False])
labels = []
scores = []
colors_list = []
consistent_markers = []

for _, row in df_sorted.iterrows():
    sym = '-|' if row['Direction'] == 'repress' else '->'
    labels.append(f"{row['TF']} {sym} {row['Target']} ({lineage_labels[row['Lineage']]})")
    scores.append(row['Directionality_Score'])
    colors_list.append(lineage_colors[row['Lineage']])
    consistent_markers.append(row['Directional_Consistent'])

y_pos = range(len(labels))
bars = ax.barh(y_pos, scores, color=colors_list, alpha=0.7, edgecolor='white')

# Mark consistent pairs
for i, (consistent, score) in enumerate(zip(consistent_markers, scores)):
    if consistent:
        marker_x = score + 0.01 if score > 0 else score - 0.01
        ax.plot(marker_x, i, '*', color='gold', markersize=8, zorder=5)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=6)
ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Directionality Score')
ax.set_title('Directionality Score for All TF-Target Pairs\n(* = lag-based consistent)')
ax.invert_yaxis()

# Legend
handles = [Patch(facecolor=c, alpha=0.7, label=lineage_labels[l])
           for l, c in lineage_colors.items()]
handles.append(plt.Line2D([0], [0], marker='*', color='gold', linestyle='None',
               markersize=8, label='Consistent'))
ax.legend(handles=handles, fontsize=8, loc='lower right')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig7_summary_heatmap.png'))
fig.savefig(os.path.join(FIG_DIR, 'fig7_summary_heatmap.pdf'))
plt.close()

print("\nAll figures saved to", FIG_DIR)
print("Files:")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}")
