"""Batch 6 figure fixes:
- Supp Fig 15B: De-overlap scatter labels
- Supp Fig 34B: Fix y-axis label / value annotation overlap
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

BASE = '/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper'
FIG_DIR = os.path.join(BASE, 'figures')
EXP_DIR = os.path.join(BASE, 'experiments')

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
    'figure.titlesize': 20, 'figure.dpi': 150,
})


# ============================================================
# SUPP FIG 15: Per-TF conservation (fix Panel B label overlap)
# ============================================================
print("=== Fixing Supp Fig 15B: Label overlap ===")

ortho_df = pd.read_csv(os.path.join(BASE, 'code/06_ortholog_transfer/per_tf_conservation.csv'))
rho_col = 'spearman_rho'
tf_col = 'tf'
edge_col = 'n_shared_edges'
size_col = 'mean_abs_rho_human'
ortho_sorted = ortho_df.sort_values(rho_col, ascending=False)

fig, axes = plt.subplots(2, 1, figsize=(16, 24))

# Panel A: Horizontal bar chart (unchanged)
ax = axes[0]
n_tfs = len(ortho_sorted)
colors_15 = []
for v in ortho_sorted[rho_col]:
    if v >= 0.5:
        colors_15.append('#009E73')
    elif v >= 0:
        colors_15.append('#E69F00')
    else:
        colors_15.append('#D55E00')

y_pos = np.arange(n_tfs)
ax.barh(y_pos, ortho_sorted[rho_col], color=colors_15, edgecolor='none', height=0.8)
ax.set_yticks(y_pos)
fontsize_15 = max(9, min(14, 600 // n_tfs))
ax.set_yticklabels(ortho_sorted[tf_col], fontsize=fontsize_15)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Cross-species Spearman \u03c1', fontsize=18)
ax.set_title('A. Per-TF conservation of regulatory targets', fontsize=20, fontweight='bold')
ax.invert_yaxis()

# Panel B: Scatter with de-overlapped labels
ax = axes[1]
sizes = ortho_df[size_col].values * 100
ax.scatter(ortho_df[edge_col], ortho_df[rho_col], s=sizes, alpha=0.6,
           color='#0072B2', edgecolors='black', linewidth=0.3)

# Select TFs to label: top 10 by rho + bottom 5
label_tfs_top = ortho_sorted.head(10)[tf_col].values
label_tfs_bot = ortho_sorted.tail(5)[tf_col].values
label_tfs = set(label_tfs_top) | set(label_tfs_bot)

# Collect label positions
label_data = []
for _, row in ortho_df.iterrows():
    if row[tf_col] in label_tfs:
        label_data.append({
            'tf': row[tf_col],
            'x': row[edge_col],
            'y': row[rho_col],
        })

# Sort by y position for greedy de-overlap
label_data.sort(key=lambda d: d['y'])

# Greedy de-overlap: ensure minimum vertical spacing in data coordinates
# Convert minimum spacing to data coordinates
y_range = ortho_df[rho_col].max() - ortho_df[rho_col].min()
min_gap = y_range * 0.035  # minimum ~3.5% of y-range between labels

placed_y = []
for ld in label_data:
    target_y = ld['y']
    # Check if this y overlaps with any already-placed label
    for py in placed_y:
        if abs(target_y - py) < min_gap:
            # Shift away from the conflicting label
            if target_y >= py:
                target_y = py + min_gap
            else:
                target_y = py - min_gap
    ld['label_y'] = target_y
    placed_y.append(target_y)

# Draw labels with arrows
for ld in label_data:
    ax.annotate(
        ld['tf'],
        xy=(ld['x'], ld['y']),
        xytext=(ld['x'] + 20, ld['label_y']),
        fontsize=11, fontweight='bold',
        ha='left', va='center',
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.8, alpha=0.6),
    )

ax.set_xlabel('Number of shared edges', fontsize=18)
ax.set_ylabel('Cross-species Spearman \u03c1', fontsize=18)
ax.set_title('B. Conservation vs edge count\n(size = mean |\u03c1| in human)', fontsize=20, fontweight='bold')

plt.tight_layout(h_pad=6)
outpath = os.path.join(FIG_DIR, 'fig_ortholog_per_tf.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {outpath}")


# ============================================================
# SUPP FIG 34: Orthogonal interventions (fix Panel B overlap)
# ============================================================
print("\n=== Fixing Supp Fig 34B: Y-axis label / annotation overlap ===")

ckpt = json.load(open(os.path.join(EXP_DIR, 'round11_orthogonal_interventions/checkpoint.json')))
conditions = ckpt['results']
baseline_auroc = conditions['baseline']['mean_auroc_attn']
cond_names = [k for k in conditions if k != 'baseline']

aurocs = [conditions[c]['mean_auroc_attn'] for c in cond_names]
short_names = [c.replace('uniform_attn_', 'Unif ').replace('mlp_zero_', 'MLP0 ').replace('_', ' ') for c in cond_names]
colors_34 = ['#009E73', '#4C9BD8', '#E69F00', '#D55E00', '#CC79A7', '#F0E442'][:len(cond_names)]

fig, axes = plt.subplots(2, 2, figsize=(22, 16))

# Panel A: Bar chart (unchanged)
ax = axes[0, 0]
bars = ax.bar(range(len(cond_names)), aurocs, color=colors_34, edgecolor='black', linewidth=0.5, width=0.7)
ax.axhline(baseline_auroc, color='green', linestyle='--', linewidth=2, alpha=0.6, label=f'Baseline={baseline_auroc:.3f}')
ax.set_xticks(range(len(cond_names)))
ax.set_xticklabels(short_names, fontsize=11, rotation=30, ha='right')
ax.set_ylabel('Mean Perturbation-First AUROC', fontsize=16)
ax.legend(fontsize=13)
for i, v in enumerate(aurocs):
    ax.text(i, v + 0.001, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylim(min(aurocs) - 0.015, max(aurocs) + 0.015)
ax.set_title('A. Orthogonal Intervention Results', fontsize=18, fontweight='bold')

# Panel B: Drop from baseline (FIXED: annotations positioned away from y-axis)
ax = axes[0, 1]
drops = [conditions[c]['mean_auroc_attn'] - baseline_auroc for c in cond_names]
y_b = np.arange(len(cond_names))
barcolors = ['#D55E00' if abs(d) > 0.003 else '#E69F00' for d in drops]
ax.barh(y_b, drops, color=barcolors, edgecolor='black', linewidth=0.5, height=0.6)
ax.set_yticks(y_b)
ax.set_yticklabels(short_names, fontsize=12)
ax.axvline(0, color='gray', linestyle='-', alpha=0.5)

# Place value annotations at the END of bars, always to the right with enough clearance
max_abs = max(abs(d) for d in drops)
for i, d in enumerate(drops):
    if abs(d) < 0.0005:
        # Very small bar: place annotation far to the right
        ax.text(max_abs * 0.4, i, f'{d:+.4f}', va='center', fontsize=13, ha='left')
    elif d >= 0:
        ax.text(d + max_abs * 0.08, i, f'{d:+.4f}', va='center', fontsize=13, ha='left')
    else:
        ax.text(d - max_abs * 0.08, i, f'{d:+.4f}', va='center', fontsize=13, ha='right')

ax.set_xlabel('AUROC Change from Baseline', fontsize=16)
ax.set_title('B. Intervention Effects', fontsize=18, fontweight='bold')
# Extend x-axis to give room for annotations
ax.set_xlim(-max_abs * 1.5, max_abs * 1.8)

# Panel C: Cross-intervention comparison
ax = axes[1, 0]
attn_conds = [c for c in cond_names if 'attn' in c]
mlp_conds = [c for c in cond_names if 'mlp' in c]
attn_drops = [conditions[c]['mean_auroc_attn'] - baseline_auroc for c in attn_conds]
mlp_drops = [conditions[c]['mean_auroc_attn'] - baseline_auroc for c in mlp_conds]
attn_names = [c.replace('uniform_attn_', '').replace('_', ' ') for c in attn_conds]
mlp_names = [c.replace('mlp_zero_', '').replace('_', ' ') for c in mlp_conds]
all_names = attn_names + mlp_names
all_drops_c = attn_drops + mlp_drops
all_colors_c = ['#4C9BD8'] * len(attn_conds) + ['#D55E00'] * len(mlp_conds)
y_c = np.arange(len(all_names))
ax.barh(y_c, all_drops_c, color=all_colors_c, edgecolor='black', linewidth=0.5, height=0.6)
ax.set_yticks(y_c)
ax.set_yticklabels(all_names, fontsize=13)
ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
for i, d in enumerate(all_drops_c):
    if abs(d) < 0.0005:
        ax.text(max(abs(min(all_drops_c)), abs(max(all_drops_c))) * 0.4, i, f'{d:+.4f}', va='center', fontsize=13, ha='left')
    elif d >= 0:
        ax.text(d + max(abs(min(all_drops_c)), abs(max(all_drops_c))) * 0.08, i, f'{d:+.4f}', va='center', fontsize=13, ha='left')
    else:
        ax.text(d - max(abs(min(all_drops_c)), abs(max(all_drops_c))) * 0.08, i, f'{d:+.4f}', va='center', fontsize=13, ha='right')
ax.set_xlabel('AUROC Drop from Baseline', fontsize=16)
ax.set_title('C. Cross-Intervention Comparison', fontsize=18, fontweight='bold')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor='#4C9BD8', label='Attention interventions'),
                   Patch(facecolor='#D55E00', label='MLP interventions')], fontsize=13)

# Panel D: Summary
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""Orthogonal Intervention Summary

Baseline AUROC: {baseline_auroc:.4f}
Number of conditions: {len(cond_names)}

Attention interventions:"""
for c in attn_conds:
    d = conditions[c]['mean_auroc_attn'] - baseline_auroc
    summary_text += f"\n  {c}: AUROC={conditions[c]['mean_auroc_attn']:.4f}, \u0394={d:+.4f}"
summary_text += "\n\nMLP interventions:"
for c in mlp_conds:
    d = conditions[c]['mean_auroc_attn'] - baseline_auroc
    summary_text += f"\n  {c}: AUROC={conditions[c]['mean_auroc_attn']:.4f}, \u0394={d:+.4f}"
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('D. Summary Statistics', fontsize=18, fontweight='bold')

plt.tight_layout(h_pad=4, w_pad=3)
outpath = os.path.join(FIG_DIR, 'fig_orthogonal_interventions.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {outpath}")

print("\n=== All batch 6 figures regenerated ===")
