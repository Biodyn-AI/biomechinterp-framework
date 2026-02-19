"""
Generate all publication figures for Cell-State-Conditional Circuit Discovery.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse, stats
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_28_cell_state_conditional_circuits"
FIG_DIR = os.path.join(OUT_DIR, "reports", "figures")
IMPL_DIR = os.path.join(OUT_DIR, "implementation")
DATA_DIR = "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw"

os.makedirs(FIG_DIR, exist_ok=True)

# Load results
df = pd.read_csv(os.path.join(IMPL_DIR, "revised_contrast_scores.csv"))
edge_agg = pd.read_csv(os.path.join(IMPL_DIR, "edge_summary_revised.csv"))
heatmap_df = pd.read_csv(os.path.join(IMPL_DIR, "regime_edge_heatmap.csv"), index_col=0)

# Fix multi-level columns from CSV
# The columns are tuples as strings like "('NFKB1', 'IL6')"
heatmap_df.columns = [c.replace("('", "").replace("', '", "->").replace("')", "") for c in heatmap_df.columns]

# Shorten regime names for display
regime_short = {
    'B cell': 'B cell',
    'CD4-positive, alpha-beta T cell': 'CD4+ T',
    'CD8-positive, alpha-beta T cell': 'CD8+ T',
    'neutrophil': 'Neutrophil',
    'macrophage': 'Macrophage',
    'monocyte': 'Monocyte',
    'plasma cell': 'Plasma cell',
    'classical monocyte': 'Classical mono',
    'erythrocyte': 'Erythrocyte',
    'natural killer cell': 'NK cell',
    'naive thymus-derived CD4-positive, alpha-beta T cell': 'Naive CD4+ T',
    'T cell': 'T cell',
    'mast cell': 'Mast cell',
    'intermediate monocyte': 'Intermed mono',
    'mature NK T cell': 'NKT cell',
    'mononuclear phagocyte': 'Mono phagocyte',
    'CD4-positive, alpha-beta thymocyte': 'CD4+ thymocyte',
    'regulatory T cell': 'Treg',
}

# Apply short names
df['regime_short'] = df['regime'].map(regime_short)
heatmap_df.index = [regime_short.get(r, r) for r in heatmap_df.index]

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# ============================================================
# FIGURE 1: Global heatmap of regime x edge correlations
# ============================================================
print("Figure 1: Regime x Edge heatmap...")

# Select top edges by regime variability
top_edges = edge_agg.sort_values('n_gated', ascending=False).head(25)
top_edge_cols = [f"{r['tf']}->{r['target']}" for _, r in top_edges.iterrows()]
available_cols = [c for c in top_edge_cols if c in heatmap_df.columns]

if available_cols:
    hm_data = heatmap_df[available_cols]
    # Sort regimes by hierarchical clustering
    fig, ax = plt.subplots(figsize=(14, 8))
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.8)
    g = sns.heatmap(hm_data, cmap='RdBu_r', center=0, norm=norm,
                    linewidths=0.5, linecolor='white',
                    xticklabels=True, yticklabels=True,
                    cbar_kws={'label': 'Spearman rho', 'shrink': 0.8}, ax=ax)
    ax.set_title('Cell-State-Conditional TF-Target Correlations\n(Top 25 Regime-Variable Edges)')
    ax.set_xlabel('TF -> Target')
    ax.set_ylabel('Cell State Regime')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_regime_edge_heatmap.png"))
    plt.savefig(os.path.join(FIG_DIR, "fig1_regime_edge_heatmap.pdf"))
    plt.close()
    print(f"  Saved fig1_regime_edge_heatmap.png ({len(available_cols)} edges)")

# ============================================================
# FIGURE 2: Volcano plot of contrast scores
# ============================================================
print("Figure 2: Volcano plot...")

fig, ax = plt.subplots(figsize=(10, 7))

# Color by edge class
colors = {
    'state-gated-active': '#d62728',
    'state-gated-inactive': '#1f77b4',
    'always-on': '#2ca02c',
    'active-uniform': '#ff7f0e',
    'weak/inactive': '#cccccc',
}

for cls in ['weak/inactive', 'active-uniform', 'always-on', 'state-gated-inactive', 'state-gated-active']:
    mask = df['edge_class'] == cls
    sub = df[mask]
    ax.scatter(sub['z_stat_rest'], -np.log10(sub['p_fisher_rest_adj'].clip(1e-300)),
              c=colors[cls], s=15, alpha=0.6, label=f'{cls} (n={mask.sum()})',
              edgecolors='none')

ax.axhline(-np.log10(0.05), color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Fisher z-statistic (focal regime vs rest)')
ax.set_ylabel('-log10(FDR-adjusted p-value)')
ax.set_title('Volcano Plot: Cell-State-Specific TF-Target Correlations')
ax.legend(loc='upper left', framealpha=0.8)

# Annotate some top points
top_pts = df.sort_values('p_fisher_rest_adj').head(8)
for _, row in top_pts.iterrows():
    label = f"{row['tf']}->{row['target']}\n({row['regime_short']})"
    y_val = min(-np.log10(max(row['p_fisher_rest_adj'], 1e-300)), 300)
    ax.annotate(label, (row['z_stat_rest'], y_val),
               fontsize=6, alpha=0.8,
               xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_volcano_plot.png"))
plt.savefig(os.path.join(FIG_DIR, "fig2_volcano_plot.pdf"))
plt.close()
print("  Saved fig2_volcano_plot.png")

# ============================================================
# FIGURE 3: Top state-gated edges - forest plot with CIs
# ============================================================
print("Figure 3: Forest plot of top state-gated edges...")

gated = df[df['edge_class'].isin(['state-gated-active', 'state-gated-inactive'])].copy()
gated['abs_z'] = gated['z_stat_rest'].abs()
top_gated = gated.sort_values('abs_z', ascending=False).head(30)

fig, ax = plt.subplots(figsize=(10, 10))
y_pos = range(len(top_gated))
labels = [f"{r['tf']}->{r['target']} ({r['regime_short']})"
          for _, r in top_gated.iterrows()]

colors_fg = ['#d62728' if c == 'state-gated-active' else '#1f77b4'
             for c in top_gated['edge_class']]

ax.barh(y_pos, top_gated['rho_focal'].values, height=0.6, color=colors_fg, alpha=0.7)
# Add error bars from bootstrap CIs
for i, (_, row) in enumerate(top_gated.iterrows()):
    if not np.isnan(row['boot_ci_lo']):
        ax.plot([row['boot_ci_lo'], row['boot_ci_hi']], [i, i],
               color='black', linewidth=1.0)

ax.axvline(0, color='grey', linestyle='-', linewidth=0.5)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Spearman rho (focal regime)')
ax.set_title('Top 30 State-Gated Edges with Bootstrap 95% CIs')
ax.invert_yaxis()

# Add global rho markers
for i, (_, row) in enumerate(top_gated.iterrows()):
    ax.plot(row['rho_global'], i, 'kx', markersize=5, markeredgewidth=1)

ax.legend(['Global rho'], loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_forest_plot.png"))
plt.savefig(os.path.join(FIG_DIR, "fig3_forest_plot.pdf"))
plt.close()
print("  Saved fig3_forest_plot.png")

# ============================================================
# FIGURE 4: Edge-level summary barplot
# ============================================================
print("Figure 4: Edge-level gating summary...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# Left: number of gated regimes per edge
top30 = edge_agg.head(30).copy()
top30['label'] = top30['tf'] + '->' + top30['target']

ax1.barh(range(len(top30)), top30['n_gated'], color='#d62728', alpha=0.7, label='State-gated')
ax1.barh(range(len(top30)), top30['n_always_on'], left=top30['n_gated'],
         color='#2ca02c', alpha=0.7, label='Always-on')
ax1.set_yticks(range(len(top30)))
ax1.set_yticklabels(top30['label'], fontsize=8)
ax1.set_xlabel('Number of regimes')
ax1.set_title('Regimes per Edge Category')
ax1.legend(loc='lower right')
ax1.invert_yaxis()

# Right: rho range per edge
ax2.barh(range(len(top30)), top30['rho_range'], color='#9467bd', alpha=0.7)
ax2.set_yticks(range(len(top30)))
ax2.set_yticklabels(top30['label'], fontsize=8)
ax2.set_xlabel('Rho range across regimes')
ax2.set_title('Correlation Variability Across Cell States')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_edge_summary.png"))
plt.savefig(os.path.join(FIG_DIR, "fig4_edge_summary.pdf"))
plt.close()
print("  Saved fig4_edge_summary.png")

# ============================================================
# FIGURE 5: Case study - detailed regime profiles for top edges
# ============================================================
print("Figure 5: Case studies...")

case_study_edges = [
    ('PAX5', 'CD79A', 'B cell identity marker'),
    ('NFKB1', 'IL1B', 'NF-kB inflammatory signaling'),
    ('RUNX3', 'CD8A', 'CD8 T cell identity'),
    ('TCF7', 'LEF1', 'T cell development'),
    ('MYC', 'CDK4', 'Cell cycle regulation'),
    ('SPI1', 'ITGAM', 'Myeloid differentiation'),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (tf, target, desc) in enumerate(case_study_edges):
    ax = axes[idx]
    sub = df[(df['tf'] == tf) & (df['target'] == target)].copy()
    sub = sub.sort_values('rho_focal', ascending=True)

    colors_bar = []
    for _, row in sub.iterrows():
        if row['edge_class'] == 'state-gated-active':
            colors_bar.append('#d62728')
        elif row['edge_class'] == 'state-gated-inactive':
            colors_bar.append('#1f77b4')
        elif row['edge_class'] == 'always-on':
            colors_bar.append('#2ca02c')
        else:
            colors_bar.append('#cccccc')

    ax.barh(range(len(sub)), sub['rho_focal'].values, color=colors_bar, alpha=0.8, height=0.7)

    # Bootstrap CIs
    for i, (_, row) in enumerate(sub.iterrows()):
        if not np.isnan(row['boot_ci_lo']):
            ax.plot([row['boot_ci_lo'], row['boot_ci_hi']], [i, i],
                   color='black', linewidth=0.8)

    ax.axvline(sub['rho_global'].values[0], color='red', linestyle='--',
              linewidth=1.2, alpha=0.7, label='Global')
    ax.axvline(0, color='grey', linewidth=0.5)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub['regime_short'].values, fontsize=7)
    ax.set_xlabel('Spearman rho')
    ax.set_title(f'{tf} -> {target}\n({desc})', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_case_studies.png"))
plt.savefig(os.path.join(FIG_DIR, "fig5_case_studies.pdf"))
plt.close()
print("  Saved fig5_case_studies.png")

# ============================================================
# FIGURE 6: Classification pie chart + bootstrap stability
# ============================================================
print("Figure 6: Classification and stability...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
class_counts = df['edge_class'].value_counts()
pie_colors = [colors.get(c, '#999999') for c in class_counts.index]
ax1.pie(class_counts.values, labels=class_counts.index, colors=pie_colors,
        autopct='%1.1f%%', startangle=140, pctdistance=0.85)
ax1.set_title('Edge-Regime Pair Classification\n(n=1422)')

# Bootstrap stability: CI width vs absolute rho
gated_only = df[df['edge_class'].isin(['state-gated-active', 'state-gated-inactive'])].copy()
non_gated = df[~df['edge_class'].isin(['state-gated-active', 'state-gated-inactive'])].copy()

ax2.scatter(non_gated['rho_focal'].abs(), non_gated['ci_width'],
           c='#cccccc', s=10, alpha=0.3, label='Other')
ax2.scatter(gated_only['rho_focal'].abs(), gated_only['ci_width'],
           c='#d62728', s=20, alpha=0.6, label='State-gated')
ax2.set_xlabel('|Spearman rho| in focal regime')
ax2.set_ylabel('Bootstrap 95% CI width')
ax2.set_title('Score Stability vs Effect Size')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig6_classification_stability.png"))
plt.savefig(os.path.join(FIG_DIR, "fig6_classification_stability.pdf"))
plt.close()
print("  Saved fig6_classification_stability.png")

# ============================================================
# FIGURE 7: Network diagram of state-gated circuits
# ============================================================
print("Figure 7: Network diagram...")

import networkx as nx

# Build a network of top state-gated edges
top_gated_edges = edge_agg[edge_agg['n_gated'] >= 4].copy()

G = nx.DiGraph()
for _, row in top_gated_edges.iterrows():
    G.add_edge(row['tf'], row['target'],
               weight=row['n_gated'],
               rho_range=row['rho_range'])

if len(G.edges) > 0:
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=2.0, seed=42)

    # Node colors: TFs = blue, targets = orange
    tfs = set(top_gated_edges['tf'])
    targets = set(top_gated_edges['target'])
    node_colors = []
    for n in G.nodes:
        if n in tfs and n in targets:
            node_colors.append('#9467bd')
        elif n in tfs:
            node_colors.append('#1f77b4')
        else:
            node_colors.append('#ff7f0e')

    edge_weights = [G[u][v]['weight'] for u, v in G.edges]
    max_w = max(edge_weights) if edge_weights else 1

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                          node_size=600, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos,
                          width=[2 + 3 * w / max_w for w in edge_weights],
                          alpha=0.6, arrows=True, arrowsize=15,
                          edge_color='grey', ax=ax)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
               markersize=10, label='TF only'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e',
               markersize=10, label='Target only'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9467bd',
               markersize=10, label='TF & Target'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_title(f'State-Gated Regulatory Circuit Network\n({len(G.edges)} edges with >= 4 state-gated regimes)')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_network.png"))
    plt.savefig(os.path.join(FIG_DIR, "fig7_network.pdf"))
    plt.close()
    print("  Saved fig7_network.png")

# ============================================================
# FIGURE 8: Pathway-level summary
# ============================================================
print("Figure 8: Pathway-level summary...")

# Group edges by source pathway
pathway_agg = df.groupby('source').agg(
    n_pairs=('tf', 'count'),
    n_gated=('edge_class', lambda x: x.isin(['state-gated-active', 'state-gated-inactive']).sum()),
    n_always_on=('edge_class', lambda x: (x == 'always-on').sum()),
    mean_rho_range=('rho_range', 'mean'),
    mean_abs_contrast=('z_stat_rest', lambda x: np.mean(np.abs(x))),
).reset_index()
pathway_agg['frac_gated'] = pathway_agg['n_gated'] / pathway_agg['n_pairs']
pathway_agg = pathway_agg.sort_values('frac_gated', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
y = range(len(pathway_agg))
ax.barh(y, pathway_agg['frac_gated'], color='#d62728', alpha=0.7,
        label='State-gated fraction')
ax.barh(y, pathway_agg['n_always_on'] / pathway_agg['n_pairs'],
        left=pathway_agg['frac_gated'], color='#2ca02c', alpha=0.7,
        label='Always-on fraction')
ax.set_yticks(list(y))
ax.set_yticklabels(pathway_agg['source'], fontsize=9)
ax.set_xlabel('Fraction of (edge, regime) pairs')
ax.set_title('Pathway-Level Regulatory Gating Summary')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig8_pathway_summary.png"))
plt.savefig(os.path.join(FIG_DIR, "fig8_pathway_summary.pdf"))
plt.close()
print("  Saved fig8_pathway_summary.png")

# ============================================================
# SUPPLEMENTARY TABLE
# ============================================================
print("\nGenerating supplementary tables...")

# Full results table
supp_table = df[['tf', 'target', 'source', 'regime', 'regime_short',
                 'rho_focal', 'rho_global', 'rho_rest',
                 'z_stat_rest', 'p_fisher_rest_adj',
                 'p_within_regime_adj', 'contrast_z',
                 'boot_ci_lo', 'boot_ci_hi', 'ci_width',
                 'n_focal', 'tf_frac', 'tgt_frac', 'edge_class']].copy()
supp_table.to_csv(os.path.join(OUT_DIR, "reports", "supplementary_table_S1.csv"), index=False)

# Edge-level summary
edge_agg.to_csv(os.path.join(OUT_DIR, "reports", "supplementary_table_S2.csv"), index=False)

print("All figures and tables generated!")
print(f"\nFigures saved to: {FIG_DIR}")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}")
