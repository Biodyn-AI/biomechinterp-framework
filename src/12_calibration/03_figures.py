"""
Step 3: Generate all figures for the paper.

1. Reliability diagrams (pre- and post-calibration)
2. ECE bar chart comparison
3. Conformal set size distributions
4. Threshold policy comparison
5. Score distribution histograms
"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

IMPL_DIR = '/Users/ihorkendiukhov/biodyn-work/subproject_31_uncertainty_calibration_edge_scores/implementation'
FIG_DIR = '/Users/ihorkendiukhov/biodyn-work/subproject_31_uncertainty_calibration_edge_scores/reports/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Load data
with open(os.path.join(IMPL_DIR, 'calibration_results.pkl'), 'rb') as f:
    cal_data = pickle.load(f)

with open(os.path.join(IMPL_DIR, 'edge_scores_and_ground_truth.pkl'), 'rb') as f:
    score_data = pickle.load(f)

pre_cal = cal_data['pre_cal_metrics']
cal_res = cal_data['calibrated_results']
conformal = cal_data['conformal_results']
threshold_res = cal_data['threshold_results']
methods = cal_data['methods']
method_data_raw = cal_data['method_data']

# Nice labels
method_labels = {
    'pearson': 'Pearson',
    'spearman': 'Spearman',
    'mutual_info': 'Mutual Info.',
    'partial_corr': 'Partial Corr.',
    'lasso': 'LASSO',
    'ensemble': 'Ensemble',
}
colors = {
    'pearson': '#1f77b4',
    'spearman': '#ff7f0e',
    'mutual_info': '#2ca02c',
    'partial_corr': '#d62728',
    'lasso': '#9467bd',
    'ensemble': '#8c564b',
}

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 150,
})

# ============================================================
# Figure 1: Reliability diagrams (3 panels: raw, Platt, isotonic)
# ============================================================
print("Figure 1: Reliability diagrams...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))

for ax_idx, (title, key) in enumerate([
    ('Raw Scores', 'raw'),
    ('Platt Scaling', 'platt'),
    ('Isotonic Regression', 'isotonic'),
]):
    ax = axes[ax_idx]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')

    for method in methods:
        if key == 'raw':
            pt = pre_cal[method]['reliability_true']
            pp = pre_cal[method]['reliability_pred']
        else:
            pt = cal_res[method][key]['reliability_true']
            pp = cal_res[method][key]['reliability_pred']

        if len(pt) > 0:
            ax.plot(pp, pt, 'o-', color=colors[method], label=method_labels[method],
                    markersize=3, linewidth=1.2)

    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    if ax_idx == 2:
        ax.legend(loc='lower right', framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_reliability_diagrams.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig1_reliability_diagrams.png'), bbox_inches='tight', dpi=200)
plt.close()

# ============================================================
# Figure 2: ECE and Brier score bar charts
# ============================================================
print("Figure 2: ECE/Brier comparison...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))

# ECE comparison
method_list = [m for m in methods if m != 'ensemble']
x = np.arange(len(method_list))
width = 0.25

ece_raw = [pre_cal[m]['ece'] for m in method_list]
ece_platt = [cal_res[m]['platt']['ece'] for m in method_list]
ece_iso = [cal_res[m]['isotonic']['ece'] for m in method_list]

axes[0].bar(x - width, ece_raw, width, label='Raw', color='#cccccc', edgecolor='black', linewidth=0.5)
axes[0].bar(x, ece_platt, width, label='Platt', color='#4daf4a', edgecolor='black', linewidth=0.5)
axes[0].bar(x + width, ece_iso, width, label='Isotonic', color='#377eb8', edgecolor='black', linewidth=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels([method_labels[m] for m in method_list], rotation=30, ha='right')
axes[0].set_ylabel('Expected Calibration Error')
axes[0].set_title('ECE: Pre- vs Post-Calibration')
axes[0].legend(framealpha=0.9)
axes[0].set_ylim(0, 0.55)

# Brier score comparison
brier_raw = [pre_cal[m]['brier'] for m in method_list]
brier_platt = [cal_res[m]['platt']['brier'] for m in method_list]
brier_iso = [cal_res[m]['isotonic']['brier'] for m in method_list]

axes[1].bar(x - width, brier_raw, width, label='Raw', color='#cccccc', edgecolor='black', linewidth=0.5)
axes[1].bar(x, brier_platt, width, label='Platt', color='#4daf4a', edgecolor='black', linewidth=0.5)
axes[1].bar(x + width, brier_iso, width, label='Isotonic', color='#377eb8', edgecolor='black', linewidth=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels([method_labels[m] for m in method_list], rotation=30, ha='right')
axes[1].set_ylabel('Brier Score')
axes[1].set_title('Brier Score: Pre- vs Post-Calibration')
axes[1].legend(framealpha=0.9)
axes[1].set_ylim(0, 0.55)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_ece_brier_comparison.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig2_ece_brier_comparison.png'), bbox_inches='tight', dpi=200)
plt.close()

# ============================================================
# Figure 3: Conformal prediction set sizes
# ============================================================
print("Figure 3: Conformal prediction sets...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))

for ax_idx, alpha in enumerate([0.05, 0.10, 0.20]):
    ax = axes[ax_idx]
    method_subset = ['pearson', 'mutual_info', 'ensemble']
    x = np.arange(3)  # set sizes: 0, 1, 2

    for i, method in enumerate(method_subset):
        sizes = conformal[method][alpha]['set_sizes']
        counts = [
            (sizes == 0).mean(),
            (sizes == 1).mean(),
            (sizes == 2).mean(),
        ]
        offset = (i - 1) * 0.25
        ax.bar(x + offset, counts, 0.22, label=method_labels[method],
               color=colors[method], edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(['Empty', 'Singleton', 'Both classes'])
    ax.set_ylabel('Fraction of test edges')
    coverage = conformal[method_subset[-1]][alpha]['coverage']
    ax.set_title(f'$\\alpha$={alpha} (target cov.={1-alpha:.0%})')
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_conformal_sets.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig3_conformal_sets.png'), bbox_inches='tight', dpi=200)
plt.close()

# ============================================================
# Figure 4: ROC and PR curves
# ============================================================
print("Figure 4: ROC and PR curves...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

for method in methods:
    d = method_data_raw[method]
    y_test = d['y_test']
    s_test = d['s_test']

    fpr, tpr, _ = roc_curve(y_test, s_test)
    auc_val = pre_cal[method]['auroc']
    axes[0].plot(fpr, tpr, color=colors[method],
                 label=f"{method_labels[method]} ({auc_val:.3f})", linewidth=1.2)

    prec, rec, _ = precision_recall_curve(y_test, s_test)
    axes[1].plot(rec, prec, color=colors[method],
                 label=method_labels[method], linewidth=1.2)

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves')
axes[0].legend(loc='lower right', fontsize=6, framealpha=0.9)

base_rate = sum(d['y_test']) / len(d['y_test'])
axes[1].axhline(base_rate, color='k', linestyle='--', alpha=0.3)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves')
axes[1].legend(loc='upper right', fontsize=6, framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_roc_pr_curves.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig4_roc_pr_curves.png'), bbox_inches='tight', dpi=200)
plt.close()

# ============================================================
# Figure 5: Score distributions (true vs false edges)
# ============================================================
print("Figure 5: Score distributions...", flush=True)
fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))

for i, method in enumerate(methods):
    ax = axes[i // 3][i % 3]
    d = method_data_raw[method]

    s_true = d['s_test'][d['y_test'] == 1]
    s_false = d['s_test'][d['y_test'] == 0]

    bins = np.linspace(min(d['s_test'].min(), 0), d['s_test'].max(), 40)
    ax.hist(s_false, bins=bins, alpha=0.5, color='#377eb8', label='False edges', density=True)
    ax.hist(s_true, bins=bins, alpha=0.5, color='#e41a1c', label='True edges', density=True)
    ax.set_title(method_labels[method])
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    if i == 0:
        ax.legend(fontsize=6, framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_score_distributions.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig5_score_distributions.png'), bbox_inches='tight', dpi=200)
plt.close()

# ============================================================
# Figure 6: Calibration improvement heatmap
# ============================================================
print("Figure 6: Calibration improvement summary...", flush=True)
fig, ax = plt.subplots(figsize=(6, 3.5))

method_list = methods
metric_labels = ['Raw ECE', 'Platt ECE', 'Isotonic ECE', 'Raw Brier', 'Platt Brier', 'Isotonic Brier']
data_matrix = np.zeros((len(method_list), len(metric_labels)))

for i, m in enumerate(method_list):
    data_matrix[i, 0] = pre_cal[m]['ece']
    data_matrix[i, 1] = cal_res[m]['platt']['ece']
    data_matrix[i, 2] = cal_res[m]['isotonic']['ece']
    data_matrix[i, 3] = pre_cal[m]['brier']
    data_matrix[i, 4] = cal_res[m]['platt']['brier']
    data_matrix[i, 5] = cal_res[m]['isotonic']['brier']

im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(metric_labels)))
ax.set_xticklabels(metric_labels, rotation=45, ha='right')
ax.set_yticks(range(len(method_list)))
ax.set_yticklabels([method_labels[m] for m in method_list])

for i in range(len(method_list)):
    for j in range(len(metric_labels)):
        ax.text(j, i, f'{data_matrix[i, j]:.3f}', ha='center', va='center', fontsize=7,
                color='white' if data_matrix[i, j] > 0.3 else 'black')

plt.colorbar(im, ax=ax, label='Score (lower is better)')
ax.set_title('Calibration Quality Summary')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig6_calibration_heatmap.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig6_calibration_heatmap.png'), bbox_inches='tight', dpi=200)
plt.close()

# ============================================================
# Figure 7: Conformal coverage vs set size tradeoff
# ============================================================
print("Figure 7: Conformal coverage-efficiency tradeoff...", flush=True)
fig, ax = plt.subplots(figsize=(5, 3.5))

for method in ['pearson', 'mutual_info', 'partial_corr', 'ensemble']:
    coverages = []
    avg_sizes = []
    for alpha in [0.05, 0.10, 0.20]:
        coverages.append(conformal[method][alpha]['coverage'])
        avg_sizes.append(conformal[method][alpha]['avg_set_size'])
    ax.plot(avg_sizes, coverages, 'o-', color=colors[method],
            label=method_labels[method], markersize=5, linewidth=1.5)

# Target lines
for alpha in [0.05, 0.10, 0.20]:
    ax.axhline(1 - alpha, color='gray', linestyle=':', alpha=0.4)

ax.set_xlabel('Average prediction set size')
ax.set_ylabel('Empirical coverage')
ax.set_title('Conformal Coverage vs. Efficiency')
ax.legend(framealpha=0.9)
ax.set_ylim(0.7, 1.02)
ax.set_xlim(1.3, 2.05)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig7_conformal_tradeoff.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig7_conformal_tradeoff.png'), bbox_inches='tight', dpi=200)
plt.close()

print("\nAll figures saved to:", FIG_DIR, flush=True)
print("Done with Step 3.", flush=True)
