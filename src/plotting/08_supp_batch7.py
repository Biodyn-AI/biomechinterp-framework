"""Batch 7 figure fixes:
- Supp Fig 1: Fix 'large' tier vertical line (F1=0 at N>=1000 on log scale)
- Supp Fig 23: Zoom AUROC y-axis to actual data range
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

BASE = '/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper'
FIG_DIR = os.path.join(BASE, 'figures')

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
    'figure.titlesize': 20, 'figure.dpi': 150,
})


# ============================================================
# SUPP FIG 1: Multi-seed scaling (fix 'large' tier F1=0)
# ============================================================
print("=== Fixing Supp Fig 1: Large tier vertical line ===")

agg_df = pd.read_csv('/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/'
                      'subproject_06_scaling_failure_analysis/reports/data_extended/'
                      'multiseed_aggregate_reference_metrics.csv')
trrust = agg_df[agg_df['reference'] == 'trrust'].copy()

fig, ax = plt.subplots(figsize=(10, 7))

tier_colors = {'small': '#1f77b4', 'medium': '#ff7f0e', 'large': '#2ca02c'}
tier_markers = {'small': 'o', 'medium': 's', 'large': '^'}

for tier in ['small', 'medium', 'large']:
    sub = trrust[trrust['model_tier'] == tier].sort_values('cells')

    # Filter to non-zero F1 for plotting on log scale
    nonzero = sub[sub['f1_mean'] > 0]
    zero = sub[sub['f1_mean'] == 0]

    if len(nonzero) > 0:
        x = nonzero['cells'].values
        y = nonzero['f1_mean'].values
        ci_low = nonzero['f1_ci_low'].values
        ci_high = nonzero['f1_ci_high'].values

        if len(nonzero) > 1:
            # Multiple points: draw line + fill
            ax.plot(x, y, '-o', color=tier_colors[tier], marker=tier_markers[tier],
                    linewidth=2.5, markersize=10, label=tier, zorder=3)
            ax.fill_between(x, ci_low, ci_high, color=tier_colors[tier], alpha=0.15)
        else:
            # Single non-zero point: draw point with error bars
            yerr_low = y - ci_low
            yerr_high = ci_high - y
            ax.errorbar(x, y, yerr=np.array([[yerr_low[0]], [yerr_high[0]]]),
                       fmt=tier_markers[tier], color=tier_colors[tier],
                       markersize=12, linewidth=2.5, capsize=6, capthick=2,
                       label=f'{tier} (F1=0 at N\u22651000)', zorder=3)

    # Annotate zero-F1 points
    if len(zero) > 0:
        for _, row in zero.iterrows():
            ax.annotate(f'F1=0', xy=(row['cells'], ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-6),
                       fontsize=11, color=tier_colors[tier], ha='center', va='bottom',
                       fontweight='bold')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Cells (log scale)', fontsize=18)
ax.set_ylabel('F1 vs TRRUST (log scale)', fontsize=18)
ax.set_title('Multi-seed scaling with 95% CI (TRRUST)', fontsize=20, fontweight='bold')
ax.legend(fontsize=14, loc='upper right')
ax.grid(True, which='both', alpha=0.3)

# Set y-axis limits to exclude the zero-dominated region
all_nonzero = trrust[trrust['f1_mean'] > 0]['f1_mean']
if len(all_nonzero) > 0:
    ymin = all_nonzero.min() * 0.5
    ymax = all_nonzero.max() * 2
    ax.set_ylim(ymin, ymax)

# Now re-annotate zero points at bottom of visible range
for tier in ['small', 'medium', 'large']:
    sub = trrust[trrust['model_tier'] == tier].sort_values('cells')
    zero = sub[sub['f1_mean'] == 0]
    for _, row in zero.iterrows():
        ax.annotate(f'\u2193 F1=0',
                   xy=(row['cells'], ax.get_ylim()[0] * 1.2),
                   fontsize=12, color=tier_colors[tier], ha='center', va='bottom',
                   fontweight='bold')

plt.tight_layout()
outpath = os.path.join(FIG_DIR, 'fig_ext_multiseed_scaling_trrust_ci.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {outpath}")


# ============================================================
# SUPP FIG 23: CSSI null inflation (zoom AUROC y-axis)
# ============================================================
print("\n=== Fixing Supp Fig 23B: AUROC y-axis zoom ===")

null_df = pd.read_csv(os.path.join(BASE, 'novel_method/scaling_v2_null_results.csv'))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

methods_to_plot = [
    ("pooled", "#1f77b4", "o"),
    ("cssi_max_oracle", "#d62728", "s"),
    ("cssi_max_shuffled", "#ff7f0e", "^"),
    ("cssi_max_random", "#9467bd", "x"),
]

for ax, metric, ylabel in [(axes[0], "f1", "F1"), (axes[1], "auc", "AUROC")]:
    all_means = []
    for method, color, marker in methods_to_plot:
        sub = null_df[null_df["method"] == method]
        means = sub.groupby("n_states")[metric].mean()
        stds = sub.groupby("n_states")[metric].std()
        all_means.extend(means.values)
        ax.errorbar(
            means.index,
            means.values,
            yerr=stds.values,
            label=method.replace('_', ' '),
            color=color,
            marker=marker,
            capsize=4,
            linewidth=2.5,
            markersize=8,
        )

    ax.set_xlabel("Number of strata (K)", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.grid(True, alpha=0.25)

    # Zoom y-axis to actual data range (with some padding)
    data_min = min(all_means)
    data_max = max(all_means)
    margin = (data_max - data_min) * 0.15
    ax.set_ylim(max(0, data_min - margin), min(1.05, data_max + margin))

axes[0].legend(fontsize=12, loc="lower left")
fig.suptitle("CSSI-max Null Stress Test (Oracle vs Shuffled/Random Labels)", fontsize=18, fontweight='bold')
fig.tight_layout()
outpath = os.path.join(FIG_DIR, 'fig_cssi_null_inflation.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {outpath}")

print("\n=== All batch 7 figures regenerated ===")
