"""Generate Figure 1: Regime x Edge heatmap directly from contrast scores."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import os, warnings
warnings.filterwarnings("ignore")

OUT_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_28_cell_state_conditional_circuits"
FIG_DIR = os.path.join(OUT_DIR, "reports", "figures")
IMPL_DIR = os.path.join(OUT_DIR, "implementation")

df = pd.read_csv(os.path.join(IMPL_DIR, "revised_contrast_scores.csv"))
edge_agg = pd.read_csv(os.path.join(IMPL_DIR, "edge_summary_revised.csv"))

regime_short = {
    'B cell': 'B cell', 'CD4-positive, alpha-beta T cell': 'CD4+ T',
    'CD8-positive, alpha-beta T cell': 'CD8+ T', 'neutrophil': 'Neutrophil',
    'macrophage': 'Macrophage', 'monocyte': 'Monocyte', 'plasma cell': 'Plasma cell',
    'classical monocyte': 'Classical mono', 'erythrocyte': 'Erythrocyte',
    'natural killer cell': 'NK cell',
    'naive thymus-derived CD4-positive, alpha-beta T cell': 'Naive CD4+ T',
    'T cell': 'T cell', 'mast cell': 'Mast cell',
    'intermediate monocyte': 'Intermed mono', 'mature NK T cell': 'NKT cell',
    'mononuclear phagocyte': 'Mono phagocyte',
    'CD4-positive, alpha-beta thymocyte': 'CD4+ thymocyte',
    'regulatory T cell': 'Treg',
}

df['regime_short'] = df['regime'].map(regime_short)
df['edge_label'] = df['tf'] + '->' + df['target']

# Select top 25 edges by n_gated
top_edges = edge_agg.sort_values('n_gated', ascending=False).head(25)
top_labels = [f"{r['tf']}->{r['target']}" for _, r in top_edges.iterrows()]

# Build pivot
sub = df[df['edge_label'].isin(top_labels)]
pivot = sub.pivot_table(index='regime_short', columns='edge_label', values='rho_focal')
# Reorder columns
pivot = pivot[[c for c in top_labels if c in pivot.columns]]

# Sort regimes by mean correlation for better visual
regime_order = pivot.mean(axis=1).sort_values(ascending=False).index
pivot = pivot.loc[regime_order]

fig, ax = plt.subplots(figsize=(16, 9))
norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.8)
sns.heatmap(pivot, cmap='RdBu_r', center=0, norm=norm,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Spearman rho', 'shrink': 0.7},
            xticklabels=True, yticklabels=True, ax=ax)
ax.set_title('Cell-State-Conditional TF-Target Correlations\n(Top 25 Edges by Number of State-Gated Regimes)', fontsize=13)
ax.set_xlabel('TF -> Target')
ax.set_ylabel('Cell State Regime')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_regime_edge_heatmap.png"), dpi=300)
plt.savefig(os.path.join(FIG_DIR, "fig1_regime_edge_heatmap.pdf"))
plt.close()
print("Saved fig1_regime_edge_heatmap.png")

# Also generate a significance overlay heatmap
sig_pivot = sub.pivot_table(index='regime_short', columns='edge_label', values='sig_rest')
sig_pivot = sig_pivot[[c for c in top_labels if c in sig_pivot.columns]].loc[regime_order]

# Create annotation: star for significant
annot = sig_pivot.applymap(lambda x: '*' if x else '')

fig, ax = plt.subplots(figsize=(16, 9))
sns.heatmap(pivot, cmap='RdBu_r', center=0, norm=norm,
            linewidths=0.5, linecolor='white',
            annot=annot, fmt='', annot_kws={'fontsize': 7, 'color': 'black'},
            cbar_kws={'label': 'Spearman rho', 'shrink': 0.7},
            xticklabels=True, yticklabels=True, ax=ax)
ax.set_title('Cell-State-Conditional Correlations with Significance\n(* = FDR < 0.05 vs rest, Fisher z-test)', fontsize=13)
ax.set_xlabel('TF -> Target')
ax.set_ylabel('Cell State Regime')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1b_heatmap_significance.png"), dpi=300)
plt.savefig(os.path.join(FIG_DIR, "fig1b_heatmap_significance.pdf"))
plt.close()
print("Saved fig1b_heatmap_significance.png")
