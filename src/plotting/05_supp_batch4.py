"""Batch 4 figure fixes:
- Supp Fig 6: Fix non-additivity computation (add 95% CI subtraction)
- Supp Fig 15: Vertical layout (2 rows x 1 col) with larger text
- Supp Fig 33: Zoom y-axis on bar chart panel
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
MEDIATION_DIR = '/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_09_causal_mediation_circuit_map/implementation/results'

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
    'figure.titlesize': 20, 'figure.dpi': 150,
})


# ============================================================
# SUPP FIG 6: Residual non-additivity (FIXED computation)
# ============================================================
print("=== Regenerating Supp Fig 6: Residual Non-additivity (fixed) ===")

tissues = ['immune', 'kidney', 'lung']
granularities = ['heads', 'mlp']

fig, axes = plt.subplots(3, 2, figsize=(20, 22))

for row_i, tissue in enumerate(tissues):
    for col_j, gran in enumerate(granularities):
        ax = axes[row_i, col_j]

        cs_path = os.path.join(MEDIATION_DIR, tissue, gran, 'causal_scores.tsv')
        cm_path = os.path.join(MEDIATION_DIR, tissue, gran, 'circuit_maps.tsv')

        try:
            cs = pd.read_csv(cs_path, sep='\t')
            cm = pd.read_csv(cm_path, sep='\t')

            pairs = cs[['source', 'target']].drop_duplicates()

            pair_labels = []
            te_abs_vals = []
            resid_abs_vals = []
            nonadd_lb_vals = []

            for _, pair_row in pairs.iterrows():
                src, tgt = pair_row['source'], pair_row['target']
                pair_label = f'{src}\u2192{tgt}'

                mask_cs = (cs['source'] == src) & (cs['target'] == tgt)
                te_mean = cs[mask_cs]['effect_mean'].values[0]
                te_std = cs[mask_cs]['effect_std'].values[0]
                n_cells = cs[mask_cs]['n_cells'].values[0]

                mask_cm = (cm['source'] == src) & (cm['target'] == tgt)
                cm_sub = cm[mask_cm]
                sum_singles = cm_sub['restoration_mean'].sum()

                residual = te_mean - sum_singles

                # Proper SE calculation
                te_se = te_std / np.sqrt(n_cells) if n_cells > 0 else 0
                comp_ses = (cm_sub['restoration_std'] / np.sqrt(cm_sub['n_cells'])).values
                combined_se = np.sqrt(te_se**2 + np.sum(comp_ses**2))

                # Non-additivity LB = max(0, |residual| - 1.96*SE)
                nonadd_lb = max(0, abs(residual) - 1.96 * combined_se)

                pair_labels.append(pair_label)
                te_abs_vals.append(abs(te_mean))
                resid_abs_vals.append(abs(residual))
                nonadd_lb_vals.append(nonadd_lb)

            if pair_labels:
                x = np.arange(len(pair_labels))
                w = 0.25
                ax.bar(x - w, te_abs_vals, width=w, label='|Total effect|', color='#376BB5', edgecolor='black', linewidth=0.5)
                ax.bar(x, resid_abs_vals, width=w, label='|Residual|', color='#D7835E', edgecolor='black', linewidth=0.5)
                ax.bar(x + w, nonadd_lb_vals, width=w, label='Non-additivity LB', color='#4E9E72', edgecolor='black', linewidth=0.5)

                ax.set_xticks(x)
                ax.set_xticklabels(pair_labels, fontsize=12, rotation=20, ha='right')

                # Annotate ALL bars with the non-additivity ratio
                for i in range(len(pair_labels)):
                    ratio = nonadd_lb_vals[i] / max(te_abs_vals[i], 1e-6)
                    ymax = max(te_abs_vals[i], resid_abs_vals[i], nonadd_lb_vals[i])
                    ax.text(i, ymax + 0.03 * ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else ymax + 0.05,
                           f'{ratio:.2f}', ha='center', fontsize=12, fontweight='bold', color='#333333')

                ax.legend(fontsize=13, loc='upper right')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=16)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes, fontsize=12)

        ax.set_ylabel('Effect magnitude', fontsize=14)
        ax.set_title(f'{tissue} / {gran}', fontsize=18, fontweight='bold')
        ax.grid(axis='y', alpha=0.25)

fig.suptitle('Real-data residual non-additivity\n(numbers above bars = non-additivity ratio: LB / |TE|)',
             fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout(h_pad=4, w_pad=3)
outpath = os.path.join(FIG_DIR, 'fig1_real_residual_nonadditivity.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {outpath}")


# ============================================================
# SUPP FIG 15: Ortholog per-TF (VERTICAL layout)
# ============================================================
print("\n=== Regenerating Supp Fig 15: Ortholog per-TF (vertical layout) ===")

ortho_df = pd.read_csv(os.path.join(BASE, 'code/06_ortholog_transfer/per_tf_conservation.csv'))

# Sort by conservation
if 'spearman_rho' in ortho_df.columns:
    rho_col = 'spearman_rho'
elif 'conservation_score' in ortho_df.columns:
    rho_col = 'conservation_score'
else:
    rho_col = [c for c in ortho_df.columns if 'rho' in c.lower() or 'corr' in c.lower() or 'conservation' in c.lower()][0]

tf_col = 'tf' if 'tf' in ortho_df.columns else ortho_df.columns[0]
ortho_sorted = ortho_df.sort_values(rho_col, ascending=False)

# VERTICAL layout: 2 rows x 1 col
fig, axes = plt.subplots(2, 1, figsize=(16, 24))

# Panel A: Horizontal bar chart of per-TF conservation
ax = axes[0]
n_tfs = len(ortho_sorted)
colors_15 = []
for v in ortho_sorted[rho_col]:
    if v >= 0.5:
        colors_15.append('#55A868')
    elif v >= 0:
        colors_15.append('#E8A838')
    else:
        colors_15.append('#C44E52')

y_pos = np.arange(n_tfs)
ax.barh(y_pos, ortho_sorted[rho_col], color=colors_15, edgecolor='none', height=0.8)
ax.set_yticks(y_pos)
fontsize_15 = max(9, min(14, 600 // n_tfs))
ax.set_yticklabels(ortho_sorted[tf_col], fontsize=fontsize_15)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Cross-species Spearman \u03c1', fontsize=18)
ax.set_title('A. Per-TF conservation of regulatory targets', fontsize=20, fontweight='bold')
ax.invert_yaxis()

# Panel B: Scatter of conservation vs edge count
ax = axes[1]
edge_col = None
for c in ortho_df.columns:
    if 'edge' in c.lower() or 'n_shared' in c.lower() or 'count' in c.lower():
        edge_col = c
        break

if edge_col:
    # Mean absolute rho for point size
    size_col = None
    for c in ortho_df.columns:
        if 'mean' in c.lower() and ('rho' in c.lower() or 'abs' in c.lower()):
            size_col = c
            break

    sizes = ortho_df[size_col].values * 100 if size_col else np.full(len(ortho_df), 50)
    ax.scatter(ortho_df[edge_col], ortho_df[rho_col], s=sizes, alpha=0.6, color='#4C72B0', edgecolors='black', linewidth=0.3)

    # Label notable TFs
    for _, row in ortho_sorted.head(10).iterrows():
        ax.annotate(row[tf_col], (row[edge_col], row[rho_col]),
                   fontsize=12, fontweight='bold', ha='left',
                   xytext=(5, 3), textcoords='offset points')
    # Label bottom TFs
    for _, row in ortho_sorted.tail(5).iterrows():
        ax.annotate(row[tf_col], (row[edge_col], row[rho_col]),
                   fontsize=12, fontweight='bold', ha='left',
                   xytext=(5, -8), textcoords='offset points')

    ax.set_xlabel('Number of shared edges', fontsize=18)
    ax.set_ylabel('Cross-species Spearman \u03c1', fontsize=18)
    ax.set_title('B. Conservation vs edge count\n(size = mean |\u03c1| in human)', fontsize=20, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'Edge count column not found', ha='center', va='center', transform=ax.transAxes, fontsize=16)

plt.tight_layout(h_pad=6)
outpath = os.path.join(FIG_DIR, 'fig_ortholog_per_tf.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {outpath}")


# ============================================================
# SUPP FIG 33: Head ablation (ZOOMED y-axis)
# ============================================================
print("\n=== Regenerating Supp Fig 33: Head ablation (zoomed y-axis) ===")

with open(os.path.join(EXP_DIR, 'round9_head_ablation/ablation_results.json')) as f:
    habl = json.load(f)

with open(os.path.join(EXP_DIR, 'round9_head_ablation/head_trrust_aurocs.json')) as f:
    head_aurocs = json.load(f)

conds = habl['conditions']
comparisons = habl['comparisons']

fig, axes = plt.subplots(2, 2, figsize=(22, 18))

# Panel A: Per-head TRRUST AUROC heatmap (18x18)
ax = axes[0, 0]
n_layers = 18
n_heads = 18
heatmap = np.full((n_layers, n_heads), np.nan)
if isinstance(head_aurocs, dict):
    for key, val in head_aurocs.items():
        # key might be "L0_H0" or similar
        parts = key.replace('L', '').replace('H', ' ').split('_')
        if len(parts) == 2:
            try:
                l, h = int(parts[0]), int(parts[1].strip())
                if 0 <= l < n_layers and 0 <= h < n_heads:
                    heatmap[l, h] = val
            except:
                pass
elif isinstance(head_aurocs, list):
    for item in head_aurocs:
        if isinstance(item, dict):
            l = item.get('layer', -1)
            h = item.get('head', -1)
            auroc = item.get('auroc', item.get('trrust_auroc', np.nan))
            if 0 <= l < n_layers and 0 <= h < n_heads:
                heatmap[l, h] = auroc

im = ax.imshow(heatmap, aspect='auto', cmap='RdYlBu_r', vmin=0.35, vmax=0.75)
ax.set_xlabel('Head', fontsize=16)
ax.set_ylabel('Layer', fontsize=16)
cbar = plt.colorbar(im, ax=ax, fraction=0.046)
cbar.set_label('TRRUST AUROC', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Mark top regulatory heads
top_heads = habl.get('top5_regulatory_heads', [])
for th in top_heads:
    l, h = th['layer'], th['head']
    ax.plot(h, l, 'r*', markersize=12)

ax.set_title('Per-Head TRRUST AUROC (18\u00d718)', fontsize=18, fontweight='bold')

# Panel B: Impact on perturbation prediction - ZOOMED Y-AXIS
ax = axes[0, 1]
cond_order = ['baseline', 'ablate_regulatory', 'ablate_random_1', 'ablate_random_2', 'ablate_random_3', 'ablate_entropy_matched']
cond_labels = ['Baseline', 'Ablate\nRegulatory', 'Ablate\nRandom 1', 'Ablate\nRandom 2', 'Ablate\nRandom 3', 'Ablate\nEntropy-matched']
aurocs = [conds[c]['mean_auroc_attn'] for c in cond_order]
stds = [conds[c]['std_auroc_attn'] for c in cond_order]
colors_33 = ['#55A868', '#C44E52', '#4C72B0', '#4C72B0', '#4C72B0', '#E8A838']

ax.bar(range(len(cond_order)), aurocs, yerr=stds, color=colors_33, edgecolor='black',
       linewidth=0.5, width=0.7, capsize=4)

# ZOOM y-axis
y_min = min(aurocs) - max(stds) - 0.01
y_max = max(aurocs) + max(stds) + 0.01
ax.set_ylim(y_min, y_max)

ax.set_xticks(range(len(cond_order)))
ax.set_xticklabels(cond_labels, fontsize=12)
ax.set_ylabel('Mean Perturbation-First AUROC', fontsize=16)

# Add correlation baseline reference
corr_auroc = conds['baseline'].get('mean_auroc_corr', 0.703)
if isinstance(corr_auroc, (int, float)) and corr_auroc > 0:
    ax.axhline(corr_auroc, color='gray', linestyle=':', alpha=0.5, label=f'Correlation ({corr_auroc:.3f})')
    ax.legend(fontsize=13)

for i, (v, s) in enumerate(zip(aurocs, stds)):
    ax.text(i, v + s + 0.002, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')

ax.set_title('Head Ablation: Impact on\nPerturbation Prediction', fontsize=18, fontweight='bold')

# Panel C: Per-perturbation scatter (baseline vs regulatory ablation)
ax = axes[1, 0]
baseline_pg = conds['baseline'].get('per_gene', [])
reg_pg = conds['ablate_regulatory'].get('per_gene', [])
baseline_per = [g['auroc_attn'] for g in baseline_pg] if baseline_pg else None
reg_per = [g['auroc_attn'] for g in reg_pg] if reg_pg else None

if baseline_per and reg_per and len(baseline_per) == len(reg_per):
    ax.scatter(baseline_per, reg_per, alpha=0.3, s=15, color='#4C72B0', rasterized=True)
    lims = [min(min(baseline_per), min(reg_per)), max(max(baseline_per), max(reg_per))]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Baseline AUROC', fontsize=16)
    ax.set_ylabel('Regulatory Ablation AUROC', fontsize=16)
else:
    ax.text(0.5, 0.5, 'Per-perturbation data\nnot available', ha='center', va='center',
            transform=ax.transAxes, fontsize=16)
    ax.set_xlabel('Baseline AUROC', fontsize=16)
    ax.set_ylabel('Regulatory Ablation AUROC', fontsize=16)

ax.set_title('Per-Perturbation: Baseline vs\nRegulatory Ablation', fontsize=18, fontweight='bold')

# Panel D: Delta AUROC distribution
ax = axes[1, 1]
if baseline_per and reg_per:
    deltas_reg = [b - r for b, r in zip(baseline_per, reg_per)]
    rnd_pg = conds['ablate_random_1'].get('per_gene', [])
    rnd_per = [g['auroc_attn'] for g in rnd_pg] if rnd_pg else None
    if rnd_per:
        deltas_rnd = [b - r for b, r in zip(baseline_per, rnd_per)]
        ax.hist(deltas_reg, bins=50, alpha=0.6, color='#C44E52', label='Regulatory ablation', density=True)
        ax.hist(deltas_rnd, bins=50, alpha=0.4, color='#4C72B0', label='Random 1', density=True)
    else:
        ax.hist(deltas_reg, bins=50, alpha=0.7, color='#C44E52', label='Regulatory ablation', density=True)
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('\u0394AUROC (Baseline \u2212 Ablation)', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.legend(fontsize=14)
else:
    # Show comparison summary from comparisons
    ax.text(0.5, 0.5, 'Per-perturbation AUROC\ndistribution not available\n\n' +
            '\n'.join([f'{c}: d={comparisons[c].get("cohens_d", "?"):.3f}'
                      for c in sorted(comparisons.keys()) if isinstance(comparisons[c].get("cohens_d"), (int, float))]),
            ha='center', va='center', transform=ax.transAxes, fontsize=13,
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax.set_title('\u0394AUROC Distribution', fontsize=18, fontweight='bold')

plt.tight_layout(h_pad=4, w_pad=3)
outpath = os.path.join(FIG_DIR, 'fig_head_ablation.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {outpath}")


print("\n=== All batch 4 figures regenerated ===")
