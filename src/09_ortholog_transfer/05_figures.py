"""
Step 5: Generate publication-quality figures.

Figures:
1. Cross-species scatter plot of edge scores
2. Per-TF conservation bar chart
3. Top-k overlap enrichment curve
4. Null model distribution vs observed
5. Conservation heatmap by TF
6. Venn diagram of top edges
7. Conserved vs fragile edge properties
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import seaborn as sns
from scipy.stats import spearmanr
import json
import os

OUT_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_30_ortholog_transfer_stress_test/implementation"
FIG_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_30_ortholog_transfer_stress_test/reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

# Load data
merged = pd.read_csv(os.path.join(OUT_DIR, "merged_edges.csv"))
per_tf = pd.read_csv(os.path.join(OUT_DIR, "per_tf_conservation.csv"))
human_edges = pd.read_csv(os.path.join(OUT_DIR, "human_edges.csv"))
mouse_edges = pd.read_csv(os.path.join(OUT_DIR, "mouse_edges.csv"))

with open(os.path.join(OUT_DIR, "analysis_results.json")) as f:
    results = json.load(f)

# ============================================================
# FIGURE 1: Main scatter plot of edge scores
# ============================================================
print("Figure 1: Cross-species scatter plot...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel A: All shared edges
ax = axes[0]
# Downsample for density plot
np.random.seed(42)
if len(merged) > 10000:
    plot_df = merged.sample(10000)
else:
    plot_df = merged
ax.scatter(plot_df['rho_human'], plot_df['rho_mouse'], alpha=0.05, s=2, c='steelblue', rasterized=True)
# Add density contours
from scipy.stats import gaussian_kde
try:
    xy = np.vstack([plot_df['rho_human'], plot_df['rho_mouse']])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    ax.scatter(plot_df['rho_human'].values[idx], plot_df['rho_mouse'].values[idx],
               c=z[idx], s=3, alpha=0.3, cmap='viridis', rasterized=True)
except:
    pass
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
ax.plot([-1, 1], [-1, 1], 'r--', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Human edge score (Spearman rho)')
ax.set_ylabel('Mouse edge score (Spearman rho)')
ax.set_title(f'A. Cross-species edge correlation\n(rho={results["spearman_rho"]:.3f}, n={results["n_shared_edges"]:,})')
ax.set_xlim(-0.6, 0.9)
ax.set_ylim(-0.6, 0.9)
ax.set_aspect('equal')

# Panel B: Strong edges only
ax = axes[1]
strong = merged[(merged['rho_human'].abs() > 0.15) | (merged['rho_mouse'].abs() > 0.15)]
colors = []
for _, row in strong.iterrows():
    if row['category'] == 'conserved':
        colors.append('forestgreen')
    elif row['category'] == 'fragile':
        colors.append('darkorange')
    elif row['category'] == 'anti_conserved':
        colors.append('red')
    else:
        colors.append('steelblue')
ax.scatter(strong['rho_human'], strong['rho_mouse'], c=colors, alpha=0.15, s=4, rasterized=True)
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
ax.plot([-1, 1], [-1, 1], 'k--', linewidth=0.5, alpha=0.3)
ax.set_xlabel('Human edge score (Spearman rho)')
ax.set_ylabel('Mouse edge score (Spearman rho)')
ax.set_title(f'B. Edge categories (|rho| > 0.15)')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='forestgreen', markersize=6, label='Conserved'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', markersize=6, label='Fragile'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Anti-conserved'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=6, label='Other'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
ax.set_xlim(-0.6, 0.9)
ax.set_ylim(-0.6, 0.9)
ax.set_aspect('equal')

# Panel C: Sign agreement by edge strength
ax = axes[2]
bins = np.arange(0, 0.65, 0.05)
bin_centers = []
sign_rates = []
counts = []
for i in range(len(bins) - 1):
    mask = (merged['rho_human'].abs() >= bins[i]) & (merged['rho_human'].abs() < bins[i+1])
    subset = merged[mask]
    if len(subset) > 10:
        bin_centers.append((bins[i] + bins[i+1]) / 2)
        sign_rates.append((subset['sign_human'] == subset['sign_mouse']).mean())
        counts.append(len(subset))

ax.bar(bin_centers, sign_rates, width=0.04, color='steelblue', alpha=0.7, edgecolor='navy')
ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Random (50%)')
ax.set_xlabel('|Human rho| bin')
ax.set_ylabel('Sign agreement rate')
ax.set_title('C. Sign agreement by edge strength')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_cross_species_scatter.png"))
plt.savefig(os.path.join(FIG_DIR, "fig1_cross_species_scatter.pdf"))
plt.close()
print("  Saved fig1")

# ============================================================
# FIGURE 2: Per-TF conservation
# ============================================================
print("Figure 2: Per-TF conservation...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Bar chart
ax = axes[0]
per_tf_sorted = per_tf.sort_values('spearman_rho', ascending=True)
colors = ['forestgreen' if r > 0.5 else ('darkorange' if r > 0.2 else 'firebrick')
          for r in per_tf_sorted['spearman_rho']]
ax.barh(range(len(per_tf_sorted)), per_tf_sorted['spearman_rho'], color=colors, alpha=0.8)
ax.set_yticks(range(len(per_tf_sorted)))
ax.set_yticklabels(per_tf_sorted['tf'], fontsize=6)
ax.set_xlabel('Cross-species Spearman rho')
ax.set_title('A. Per-TF conservation of regulatory targets')
ax.axvline(0, color='black', linewidth=0.5)
ax.axvline(0.5, color='gray', linewidth=0.5, linestyle='--')

# Panel B: Conservation vs number of shared edges
ax = axes[1]
ax.scatter(per_tf['n_shared_edges'], per_tf['spearman_rho'],
           s=per_tf['mean_abs_rho_human'] * 200, alpha=0.6, c='steelblue', edgecolors='navy', linewidth=0.5)
for _, row in per_tf.iterrows():
    if row['spearman_rho'] > 0.85 or row['spearman_rho'] < 0.1 or row['n_shared_edges'] > 600:
        ax.annotate(row['tf'], (row['n_shared_edges'], row['spearman_rho']),
                    fontsize=7, ha='left', va='bottom')
ax.set_xlabel('Number of shared edges')
ax.set_ylabel('Cross-species Spearman rho')
ax.set_title('B. Conservation vs edge count\n(size = mean |rho| in human)')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_per_tf_conservation.png"))
plt.savefig(os.path.join(FIG_DIR, "fig2_per_tf_conservation.pdf"))
plt.close()
print("  Saved fig2")

# ============================================================
# FIGURE 3: Top-k overlap and null model
# ============================================================
print("Figure 3: Top-k overlap and null model...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Top-k overlap curve
ax = axes[0]
ks = [50, 100, 200, 500, 1000, 2000, 5000]
overlaps = []
expected = []
human_edges['edge_key'] = human_edges['tf'] + '|' + human_edges['target']
mouse_edges['edge_key'] = mouse_edges['tf'] + '|' + mouse_edges['target']

with open(os.path.join(OUT_DIR, "tf_lists.json")) as f:
    tf_info = json.load(f)
shared_tfs = tf_info['shared_tfs']
h_shared = human_edges[human_edges['tf'].isin(shared_tfs)]
m_shared = mouse_edges[mouse_edges['tf'].isin(shared_tfs)]
n_universe = len(set(h_shared['edge_key']) | set(m_shared['edge_key']))

for k in ks:
    h_top = set(h_shared.nlargest(k, 'abs_rho')['edge_key'])
    m_top = set(m_shared.nlargest(k, 'abs_rho')['edge_key'])
    overlaps.append(len(h_top & m_top))
    expected.append(k * k / n_universe)

ax.plot(ks, overlaps, 'o-', color='steelblue', linewidth=2, markersize=6, label='Observed')
ax.plot(ks, expected, 's--', color='firebrick', linewidth=1.5, markersize=5, label='Expected (random)')
ax.set_xlabel('k (top edges)')
ax.set_ylabel('Overlap count')
ax.set_title('A. Top-k edge overlap')
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')

# Panel B: Null distribution for rank correlation
ax = axes[1]
# Re-run a smaller null for visualization
np.random.seed(42)
null_corrs = []
for _ in range(500):
    perm = m_shared.copy()
    perm['target'] = np.random.permutation(perm['target'].values)
    perm['edge_key'] = perm['tf'] + '|' + perm['target']
    shared_keys = set(h_shared['edge_key']) & set(perm['edge_key'])
    if len(shared_keys) > 10:
        h_sub = h_shared[h_shared['edge_key'].isin(shared_keys)].set_index('edge_key')['rho']
        m_sub = perm[perm['edge_key'].isin(shared_keys)].drop_duplicates('edge_key').set_index('edge_key')['rho']
        common = h_sub.index.intersection(m_sub.index)
        if len(common) > 10:
            r, _ = spearmanr(h_sub[common], m_sub[common])
            null_corrs.append(r)

ax.hist(null_corrs, bins=40, color='gray', alpha=0.7, edgecolor='black', linewidth=0.5, label='Null distribution')
ax.axvline(results['spearman_rho'], color='red', linewidth=2, label=f'Observed (rho={results["spearman_rho"]:.3f})')
ax.set_xlabel('Spearman rho (cross-species)')
ax.set_ylabel('Count')
ax.set_title(f'B. Observed vs null distribution\n(z={results["observed_vs_null_zscore"]:.1f})')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_topk_and_null.png"))
plt.savefig(os.path.join(FIG_DIR, "fig3_topk_and_null.pdf"))
plt.close()
print("  Saved fig3")

# ============================================================
# FIGURE 4: Conservation heatmap
# ============================================================
print("Figure 4: Conservation heatmap...")

# Pick top TFs by number of edges
top_tfs = per_tf.nlargest(20, 'n_shared_edges')['tf'].tolist()

# For each TF pair, compute whether they share conserved targets
# Actually, make a TF x property heatmap
fig, ax = plt.subplots(figsize=(8, 8))

heatmap_data = per_tf.set_index('tf')[['spearman_rho', 'sign_agreement', 'mean_abs_rho_human', 'mean_abs_rho_mouse']]
heatmap_data.columns = ['Cross-species rho', 'Sign agreement', 'Mean |rho| human', 'Mean |rho| mouse']
heatmap_data = heatmap_data.sort_values('Cross-species rho', ascending=False)

sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
            linewidths=0.5, ax=ax, vmin=-0.2, vmax=1.0,
            annot_kws={'size': 6})
ax.set_title('TF Conservation Properties')
ax.set_xlabel('')
ax.tick_params(axis='y', labelsize=7)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_conservation_heatmap.png"))
plt.savefig(os.path.join(FIG_DIR, "fig4_conservation_heatmap.pdf"))
plt.close()
print("  Saved fig4")

# ============================================================
# FIGURE 5: Top conserved and fragile edge networks
# ============================================================
print("Figure 5: Top conserved edge examples...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Top conserved edges - TF-target pairs
conserved_edges = merged.nlargest(30, 'conservation_score')[['tf', 'target', 'rho_human', 'rho_mouse']]
ax = axes[0]
y_pos = range(len(conserved_edges))
ax.barh(y_pos, conserved_edges['rho_human'], height=0.35, color='steelblue', alpha=0.7, label='Human')
ax.barh([y + 0.35 for y in y_pos], conserved_edges['rho_mouse'], height=0.35, color='darkorange', alpha=0.7, label='Mouse')
ax.set_yticks([y + 0.175 for y in y_pos])
labels = [f"{row['tf']} -> {row['target']}" for _, row in conserved_edges.iterrows()]
ax.set_yticklabels(labels, fontsize=6)
ax.set_xlabel('Spearman rho')
ax.set_title('A. Top 30 conserved edges')
ax.legend(fontsize=8)
ax.invert_yaxis()

# Panel B: Top fragile edges
fragile_edges = merged[(merged['max_abs_rho'] > 0.3)].nlargest(30, 'delta_abs_rho')[['tf', 'target', 'rho_human', 'rho_mouse']]
ax = axes[1]
y_pos = range(len(fragile_edges))
ax.barh(y_pos, fragile_edges['rho_human'], height=0.35, color='steelblue', alpha=0.7, label='Human')
ax.barh([y + 0.35 for y in y_pos], fragile_edges['rho_mouse'], height=0.35, color='darkorange', alpha=0.7, label='Mouse')
ax.set_yticks([y + 0.175 for y in y_pos])
labels = [f"{row['tf']} -> {row['target']}" for _, row in fragile_edges.iterrows()]
ax.set_yticklabels(labels, fontsize=6)
ax.set_xlabel('Spearman rho')
ax.set_title('B. Top 30 fragile edges')
ax.legend(fontsize=8)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_conserved_vs_fragile.png"))
plt.savefig(os.path.join(FIG_DIR, "fig5_conserved_vs_fragile.pdf"))
plt.close()
print("  Saved fig5")

# ============================================================
# FIGURE 6: Venn-style diagram
# ============================================================
print("Figure 6: Venn-style overlap...")

fig, ax = plt.subplots(figsize=(7, 5))
k = 1000
h_top = set(h_shared.nlargest(k, 'abs_rho')['edge_key'])
m_top = set(m_shared.nlargest(k, 'abs_rho')['edge_key'])
both = h_top & m_top
h_only = h_top - m_top
m_only = m_top - h_top

# Draw circles
circle1 = Circle((0.35, 0.5), 0.3, fill=True, facecolor='steelblue', alpha=0.3, edgecolor='steelblue', linewidth=2)
circle2 = Circle((0.65, 0.5), 0.3, fill=True, facecolor='darkorange', alpha=0.3, edgecolor='darkorange', linewidth=2)
ax.add_patch(circle1)
ax.add_patch(circle2)

ax.text(0.22, 0.5, f'Human only\n{len(h_only)}', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.5, 0.5, f'Shared\n{len(both)}', ha='center', va='center', fontsize=14, fontweight='bold', color='black')
ax.text(0.78, 0.5, f'Mouse only\n{len(m_only)}', ha='center', va='center', fontsize=14, fontweight='bold')

ax.text(0.22, 0.15, 'Human', ha='center', va='center', fontsize=12, color='steelblue', fontweight='bold')
ax.text(0.78, 0.15, 'Mouse', ha='center', va='center', fontsize=12, color='darkorange', fontweight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(f'Top-{k} Edge Overlap', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig6_venn_diagram.png"))
plt.savefig(os.path.join(FIG_DIR, "fig6_venn_diagram.pdf"))
plt.close()
print("  Saved fig6")

# ============================================================
# FIGURE 7: Summary panel figure
# ============================================================
print("Figure 7: Summary panel...")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# A: Scatter plot
ax = fig.add_subplot(gs[0, 0])
np.random.seed(42)
sample = merged.sample(min(8000, len(merged)))
ax.scatter(sample['rho_human'], sample['rho_mouse'], alpha=0.06, s=2, c='steelblue', rasterized=True)
ax.plot([-1, 1], [-1, 1], 'r--', linewidth=0.5, alpha=0.5)
ax.axhline(0, color='gray', linewidth=0.3)
ax.axvline(0, color='gray', linewidth=0.3)
ax.set_xlabel('Human rho')
ax.set_ylabel('Mouse rho')
ax.set_title(f'A. Edge score correlation\n(Spearman={results["spearman_rho"]:.3f})')
ax.set_xlim(-0.5, 0.85)
ax.set_ylim(-0.5, 0.85)

# B: Top-k overlap
ax = fig.add_subplot(gs[0, 1])
ax.plot(ks, overlaps, 'o-', color='steelblue', linewidth=2, markersize=5)
ax.plot(ks, expected, 's--', color='firebrick', linewidth=1.5, markersize=4)
ax.set_xlabel('k')
ax.set_ylabel('Overlap')
ax.set_title('B. Top-k edge overlap')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(['Observed', 'Random'], fontsize=8)

# C: Null distribution
ax = fig.add_subplot(gs[0, 2])
ax.hist(null_corrs, bins=30, color='gray', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axvline(results['spearman_rho'], color='red', linewidth=2)
ax.set_xlabel('Spearman rho')
ax.set_ylabel('Count')
ax.set_title(f'C. Observed vs null (z={results["observed_vs_null_zscore"]:.1f})')

# D: Per-TF conservation (top 25)
ax = fig.add_subplot(gs[1, 0])
top25 = per_tf.nlargest(25, 'spearman_rho')
colors_tf = ['forestgreen' if r > 0.7 else ('steelblue' if r > 0.4 else 'firebrick')
             for r in top25['spearman_rho']]
ax.barh(range(len(top25)), top25['spearman_rho'], color=colors_tf, alpha=0.8)
ax.set_yticks(range(len(top25)))
ax.set_yticklabels(top25['tf'], fontsize=6)
ax.set_xlabel('Cross-species rho')
ax.set_title('D. Most conserved TFs')
ax.invert_yaxis()

# E: Sign agreement by strength
ax = fig.add_subplot(gs[1, 1])
bins_e = np.arange(0, 0.65, 0.05)
bc, sr = [], []
for i in range(len(bins_e) - 1):
    mask = (merged['rho_human'].abs() >= bins_e[i]) & (merged['rho_human'].abs() < bins_e[i+1])
    subset = merged[mask]
    if len(subset) > 10:
        bc.append((bins_e[i] + bins_e[i+1]) / 2)
        sr.append((subset['sign_human'] == subset['sign_mouse']).mean())
ax.bar(bc, sr, width=0.04, color='steelblue', alpha=0.7, edgecolor='navy')
ax.axhline(0.5, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('|Human rho| bin')
ax.set_ylabel('Sign agreement')
ax.set_title('E. Sign preservation by strength')
ax.set_ylim(0, 1.05)

# F: Category distribution
ax = fig.add_subplot(gs[1, 2])
cat_counts = merged['category'].value_counts()
colors_cat = {'conserved': 'forestgreen', 'fragile': 'darkorange', 'anti_conserved': 'firebrick', 'weak': 'gray'}
ax.bar(range(len(cat_counts)), cat_counts.values,
       color=[colors_cat.get(c, 'gray') for c in cat_counts.index],
       alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(cat_counts)))
ax.set_xticklabels(cat_counts.index, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Number of edges')
ax.set_title('F. Edge categories')

plt.savefig(os.path.join(FIG_DIR, "fig7_summary_panel.png"))
plt.savefig(os.path.join(FIG_DIR, "fig7_summary_panel.pdf"))
plt.close()
print("  Saved fig7")

print("\nAll figures saved to:", FIG_DIR)
print("Done.")
