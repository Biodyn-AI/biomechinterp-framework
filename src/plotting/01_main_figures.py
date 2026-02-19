"""Replot main paper figures 3, 5, 6 with larger fonts and better layout."""
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

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13,
    'figure.titlesize': 20, 'figure.dpi': 150,
})

# ============================================================
# FIGURE 3: Incremental Value (fig_incremental_value.png)
# ============================================================
print("=== Regenerating Figure 3: Incremental Value ===")

with open(os.path.join(BASE, 'experiments/round9_incremental_value/incremental_value_results.json')) as f:
    iv = json.load(f)

cv = iv['cv_results']
bs = iv['bootstrap_summary']
bd = iv['bootstrap_delta_ci']
strat = iv['stratified_results']

fig, axes = plt.subplots(1, 3, figsize=(24, 7), gridspec_kw={'width_ratios': [1.2, 1, 1]})

# Panel A: Model comparison
ax = axes[0]
models = ['A_gene_only', 'B_attn_only', 'C_corr_only', 'D_gene_plus_attn', 'E_gene_plus_corr']
labels = ['Gene\nonly', 'Attn\nonly', 'Corr\nonly', 'Gene+\nAttn', 'Gene+\nCorr']
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
means = [cv[m]['mean_auroc'] for m in models]
stds = [cv[m]['std_auroc'] for m in models]
bars = ax.bar(range(5), means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5, width=0.7)
ax.set_xticks(range(5))
ax.set_xticklabels(labels, fontsize=13)
ax.set_ylabel('CV AUROC', fontsize=16)
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.015, f'{m:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
# Add delta annotations — place below the bars, staggered to avoid overlap
delta_attn = iv['delta_auroc']['gene_plus_attn_minus_gene']
delta_corr = iv['delta_auroc']['gene_plus_corr_minus_gene']
ci_attn = bd['attn']
ci_corr = bd['corr']
# Place Gene+Attn delta at bar 3, pointing down with arrow
ax.annotate(f'$\\Delta$={delta_attn:+.4f}\n95% CI [{ci_attn[0]:+.4f}, {ci_attn[1]:+.4f}]',
            xy=(3, means[3] - stds[3] - 0.01), xytext=(3, 0.38), fontsize=10, ha='center', color='#555555',
            arrowprops=dict(arrowstyle='->', color='#8172B2', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#8172B2', alpha=0.12, edgecolor='#8172B2'))
# Place Gene+Corr delta at bar 4
ax.annotate(f'$\\Delta$={delta_corr:+.4f}\n95% CI [{ci_corr[0]:+.4f}, {ci_corr[1]:+.4f}]',
            xy=(4, means[4] - stds[4] - 0.01), xytext=(4, 0.22), fontsize=10, ha='center', color='#555555',
            arrowprops=dict(arrowstyle='->', color='#CCB974', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#CCB974', alpha=0.12, edgecolor='#CCB974'))
ax.set_title('(A) Model comparison\n(5-fold GroupKFold)', fontsize=17, fontweight='bold')

# Panel B: Bootstrap delta distribution
ax = axes[1]
# Reconstruct bootstrap deltas from per-fold data
np.random.seed(42)
folds_gene = np.array(cv['A_gene_only']['per_fold_auroc'])
folds_attn = np.array(cv['D_gene_plus_attn']['per_fold_auroc'])
folds_corr = np.array(cv['E_gene_plus_corr']['per_fold_auroc'])
n_boot = 1000
boot_delta_attn = []
boot_delta_corr = []
for _ in range(n_boot):
    idx = np.random.choice(5, 5, replace=True)
    boot_delta_attn.append(np.mean(folds_attn[idx]) - np.mean(folds_gene[idx]))
    boot_delta_corr.append(np.mean(folds_corr[idx]) - np.mean(folds_gene[idx]))
boot_delta_attn = np.array(boot_delta_attn)
boot_delta_corr = np.array(boot_delta_corr)

ax.hist(boot_delta_attn, bins=30, alpha=0.6, color='#8172B2', label='Gene+Attn $-$ Gene', edgecolor='white')
ax.hist(boot_delta_corr, bins=30, alpha=0.6, color='#CCB974', label='Gene+Corr $-$ Gene', edgecolor='white')
ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
ax.set_xlabel('$\\Delta$AUROC', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
ax.legend(fontsize=12, loc='upper left')
ax.set_title('(B) Bootstrap $\\Delta$AUROC distribution\n(n=1,000)', fontsize=17, fontweight='bold')

# Panel C: TF vs non-TF stratification
ax = axes[2]
x = np.arange(2)
width = 0.25
tf_vals = [strat['TF']['A_gene_only'], strat['TF']['D_gene_plus_attn'], strat['TF']['E_gene_plus_corr']]
nontf_vals = [strat['non-TF']['A_gene_only'], strat['non-TF']['D_gene_plus_attn'], strat['non-TF']['E_gene_plus_corr']]
bars1 = ax.bar(x - width, [tf_vals[0], nontf_vals[0]], width, label='Gene only', color='#4C72B0', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, [tf_vals[1], nontf_vals[1]], width, label='Gene + Attn', color='#8172B2', edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, [tf_vals[2], nontf_vals[2]], width, label='Gene + Corr', color='#CCB974', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(['TF\nperturbations', 'non-TF\nperturbations'], fontsize=14)
ax.set_ylabel('CV AUROC', fontsize=16)
ax.set_ylim(0.85, 0.925)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
ax.legend(fontsize=12, loc='upper right')
for bars_group in [bars1, bars2, bars3]:
    for bar in bars_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.001, f'{h:.3f}', ha='center', va='bottom', fontsize=10)
ax.set_title('(C) TF vs non-TF stratification', fontsize=17, fontweight='bold')

plt.tight_layout(w_pad=3)
outpath = os.path.join(FIG_DIR, 'fig_incremental_value.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {outpath}")


# ============================================================
# FIGURE 5: Non-K562 Replication (fig_non_k562_replication.png)
# ============================================================
print("\n=== Regenerating Figure 5: Non-K562 Replication ===")

# Load per-gene CSVs for scatter plots
datasets_info = [
    {
        'name': 'TianKampmann2021', 'label': 'iPSC-derived neurons',
        'csv_dir': os.path.join(BASE, 'experiments/round13_non_k562_replication/TianKampmann2021'),
        'json': os.path.join(BASE, 'experiments/round13_non_k562_replication/TianKampmann2021/TianKampmann2021_summary.json'),
    },
    {
        'name': 'ReplogleRPE1', 'label': 'RPE1 retinal pigment epithelial',
        'csv_dir': os.path.join(BASE, 'experiments/round13_non_k562_replication/ReplogleRPE1'),
        'json': os.path.join(BASE, 'experiments/round13_non_k562_replication/ReplogleRPE1/ReplogleRPE1_summary.json'),
    }
]

fig, axes = plt.subplots(2, 3, figsize=(24, 16))

for row, ds in enumerate(datasets_info):
    with open(ds['json']) as f:
        summary = json.load(f)

    # Column 1: Bar chart by layer
    ax = axes[row, 0]
    layers_eval = summary.get('eval_layers', [6, 13, 15, 18])
    layer_keys = sorted([k for k in summary['results_by_layer'].keys()], key=int)
    attn_means = [summary['results_by_layer'][k]['attention']['mean_auroc'] for k in layer_keys]
    corr_means = [summary['results_by_layer'][k]['correlation']['mean_auroc'] for k in layer_keys]
    layer_labels = [f"L{summary['results_by_layer'][k]['layer_1indexed']}" for k in layer_keys]

    x = np.arange(len(layer_keys))
    w = 0.35
    ax.bar(x - w/2, attn_means, w, color='#4C72B0', label='Attention', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, corr_means, w, color='#E8A838', label='Correlation', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize=14)
    ax.set_ylabel('Mean AUROC', fontsize=16)
    ax.legend(fontsize=13)
    n_eval = summary['results_by_layer'][layer_keys[0]].get('n_evaluated', summary.get('n_evaluable_perturbations', '?'))
    lfc = summary.get('primary_lfc_threshold', 0.5)
    ax.set_title(f"{ds['label']}\n(n={n_eval}, LFC>{lfc})", fontsize=16, fontweight='bold')
    ax.set_ylim(0.35 if row == 0 else 0.55, 0.85 if row == 0 else 0.85)

    # Column 2: Per-perturbation scatter (use L15 data)
    ax = axes[row, 1]
    l15_key = '14'  # layer 15 is 0-indexed 14
    csv_path = os.path.join(ds['csv_dir'], f"per_gene_L15_{ds['name']}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        ax.scatter(df['auroc_corr'], df['auroc_attn'], alpha=0.4, s=15 if row == 1 else 40,
                   color='#E8A838', edgecolor='none')
        lims = [min(df['auroc_corr'].min(), df['auroc_attn'].min()) - 0.02,
                max(df['auroc_corr'].max(), df['auroc_attn'].max()) + 0.02]
        ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    ax.set_xlabel('Correlation AUROC', fontsize=16)
    ax.set_ylabel('Attention (L15) AUROC', fontsize=16)
    n_scatter = len(df) if os.path.exists(csv_path) else '?'
    ax.set_title(f'Per-Perturbation (n={n_scatter})', fontsize=16, fontweight='bold')

    # Column 3: Effect size + 95% CI per layer
    ax = axes[row, 2]
    ds_list = []
    for k in layer_keys:
        r = summary['results_by_layer'][k]
        d = r['attn_vs_corr']['cohens_d']
        bci = r['attn_vs_corr'].get('bootstrap_ci_95', [0, 0])
        diff = r['attn_vs_corr']['mean_diff_auroc']
        ds_list.append({
            'layer': f"L{r['layer_1indexed']}",
            'diff': diff,
            'ci_lo': bci[0], 'ci_hi': bci[1],
            'd': d, 'p': r['attn_vs_corr']['wilcoxon_p']
        })
    y_pos = np.arange(len(ds_list))
    diffs = [d['diff'] for d in ds_list]
    ci_lo = [d['ci_lo'] for d in ds_list]
    ci_hi = [d['ci_hi'] for d in ds_list]
    xerr = [[d - l for d, l in zip(diffs, ci_lo)], [h - d for d, h in zip(diffs, ci_hi)]]
    ax.barh(y_pos, diffs, xerr=xerr, capsize=5, color='#55A868', edgecolor='black', linewidth=0.5, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([d['layer'] for d in ds_list], fontsize=14)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('AUROC Diff (Attn $-$ Corr)', fontsize=16)
    ax.set_title('Effect Size + 95% CI', fontsize=16, fontweight='bold')
    # Annotate effect sizes and p-values
    max_val = max(max(diffs), max(ci_hi)) if diffs else 0.1
    for i, d in enumerate(ds_list):
        p_str = f"p={d['p']:.2g}" if d['p'] > 0 else "p<10$^{{-10}}$"
        d_str = f"d={d['d']:.2f}"
        ax.text(max_val * 1.1, i, f"{d_str}, {p_str}", va='center', fontsize=14, color='#333333', fontweight='bold')

plt.tight_layout(h_pad=4, w_pad=3)
outpath = os.path.join(FIG_DIR, 'fig_non_k562_replication.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {outpath}")


# ============================================================
# FIGURE 6: RPE1 Confound Battery (fig_rpe1_confound_battery.png)
# ============================================================
print("\n=== Regenerating Figure 6: RPE1 Confound Battery ===")

with open(os.path.join(BASE, 'experiments/round14_rpe1_confound_battery/rpe1_confound_battery_full.json')) as f:
    bat = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(24, 16))

# Panel A: Trivial baselines
ax = axes[0, 0]
p1 = bat['part1_trivial_baselines']['summary']
names = [p['predictor'] for p in p1]
aurocs = [p['mean_auroc'] for p in p1]
stds = [p['std_auroc'] for p in p1]
colors_a = ['#C44E52', '#CCB974', '#55A868', '#4C72B0', '#E8A838']
bars = ax.bar(range(len(names)), aurocs, yerr=stds, capsize=4, color=colors_a, edgecolor='black', linewidth=0.5, width=0.7)
ax.set_xticks(range(len(names)))
short_names = ['Mean\nExpr', '1-Dropout', 'Variance', 'Attention\n(L15)', 'Spearman\nCorr']
ax.set_xticklabels(short_names, fontsize=12)
ax.set_ylabel('Mean AUROC', fontsize=16)
ax.set_ylim(0.5, 1.0)
for i, (m, s) in enumerate(zip(aurocs, stds)):
    ax.text(i, m + s + 0.008, f'{m:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_title('(A) Predictor comparison\n(trivial baselines)', fontsize=17, fontweight='bold')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)

# Panel B: Incremental value
ax = axes[0, 1]
p2 = bat['part2_incremental_value']['cv_results']
model_names = ['A_gene_only', 'B_attn_only', 'C_corr_only', 'D_gene_plus_attn', 'E_gene_plus_corr']
model_labels = ['Gene\nonly', 'Attn\nonly', 'Corr\nonly', 'Gene+\nAttn', 'Gene+\nCorr']
model_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
m_aurocs = [p2[m]['mean_auroc'] for m in model_names]
m_stds = [p2[m]['std_auroc'] for m in model_names]
bars = ax.bar(range(5), m_aurocs, yerr=m_stds, capsize=4, color=model_colors, edgecolor='black', linewidth=0.5, width=0.7)
ax.set_xticks(range(5))
ax.set_xticklabels(model_labels, fontsize=12)
ax.set_ylabel('CV AUROC', fontsize=16)
ax.set_ylim(0.5, 1.0)
for i, (m, s) in enumerate(zip(m_aurocs, m_stds)):
    ax.text(i, m + s + 0.008, f'{m:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
# Add delta annotations
delta_info = bat['part2_incremental_value'].get('delta_auroc', {})
if delta_info:
    for key, idx in [('B_gene_plus_attn', 3), ('C_gene_plus_corr', 4)]:
        if key in delta_info:
            d = delta_info[key]
            ax.text(idx, m_aurocs[idx] - 0.03, f'$\\Delta$={d:+.4f}', ha='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))
ax.set_title('(B) Incremental value\n(5-fold GroupKFold)', fontsize=17, fontweight='bold')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)

# Panel C: Covariate balance (propensity matching)
ax = axes[0, 2]
p3 = bat.get('part3_propensity_matched', {})
cov_balance = p3.get('covariate_balance', None)
if cov_balance and isinstance(cov_balance, list):
    cov_names = [c.get('covariate', c.get('name', f'Cov{i}')) for i, c in enumerate(cov_balance)]
    smd_before = [c.get('smd_before', c.get('before', 0)) for c in cov_balance]
    smd_after = [c.get('smd_after', c.get('after', 0)) for c in cov_balance]
    y = np.arange(len(cov_names))
    ax.barh(y + 0.2, smd_before, 0.35, label='Before', color='#C44E52', alpha=0.8)
    ax.barh(y - 0.2, smd_after, 0.35, label='After', color='#55A868', alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(cov_names, fontsize=13)
    ax.axvline(0.1, color='gray', linestyle=':', alpha=0.5, label='SMD=0.1')
    ax.legend(fontsize=12)
elif cov_balance and isinstance(cov_balance, dict):
    cov_names = list(cov_balance.keys())
    smd_before = [abs(cov_balance[k].get('smd_before', cov_balance[k].get('before', 0))) for k in cov_names]
    smd_after = [abs(cov_balance[k].get('smd_after', cov_balance[k].get('after', 0))) for k in cov_names]
    y = np.arange(len(cov_names))
    ax.barh(y + 0.2, smd_before, 0.35, label='Before', color='#C44E52', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.barh(y - 0.2, smd_after, 0.35, label='After', color='#55A868', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(cov_names, fontsize=13)
    ax.axvline(0.1, color='gray', linestyle=':', alpha=0.5, label='SMD=0.1')
    ax.legend(fontsize=12)
    for i in range(len(cov_names)):
        ax.text(smd_before[i] + 0.02, i + 0.2, f'{smd_before[i]:.2f}', va='center', fontsize=11)
        ax.text(smd_after[i] + 0.02, i - 0.2, f'{smd_after[i]:.2f}', va='center', fontsize=11)
else:
    # Fallback: use matching summary if available
    match_file = os.path.join(BASE, 'experiments/round14_rpe1_confound_battery/part3_propensity_matched.json')
    if os.path.exists(match_file):
        with open(match_file) as f:
            p3_detail = json.load(f)
        cb = p3_detail.get('covariate_balance', p3_detail.get('balance', {}))
        if isinstance(cb, dict):
            cov_names = list(cb.keys())
            smd_before = [cb[k].get('smd_before', cb[k].get('before', 0)) for k in cov_names]
            smd_after = [cb[k].get('smd_after', cb[k].get('after', 0)) for k in cov_names]
            y = np.arange(len(cov_names))
            ax.barh(y + 0.2, smd_before, 0.35, label='Before', color='#C44E52', alpha=0.8)
            ax.barh(y - 0.2, smd_after, 0.35, label='After', color='#55A868', alpha=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(cov_names, fontsize=13)
            ax.axvline(0.1, color='gray', linestyle=':', alpha=0.5)
            ax.legend(fontsize=12)
        else:
            ax.text(0.5, 0.5, 'Covariate balance\ndata not available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
    else:
        ax.text(0.5, 0.5, 'Covariate balance\ndata not available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
ax.set_xlabel('|SMD|', fontsize=16)
ax.set_title('(C) Covariate balance\n(propensity matching)', fontsize=17, fontweight='bold')

# Panel D: Edge evaluation (matched set)
ax = axes[1, 0]
raw_edge = p3.get('raw_edge_results', {})
matched_cv = p3.get('matched_cv_results', {})
# Show raw edge AUROC + matched CV results
edge_names = []
edge_aurocs = []
edge_stds = []
edge_colors_d = []
if raw_edge:
    edge_names.extend(['Attn\n(raw edge)', 'Corr\n(raw edge)'])
    edge_aurocs.extend([raw_edge['attn_mean_auroc'], raw_edge['corr_mean_auroc']])
    edge_stds.extend([raw_edge.get('attn_std_auroc', 0), raw_edge.get('corr_std_auroc', 0)])
    edge_colors_d.extend(['#4C72B0', '#E8A838'])
if matched_cv:
    for k, lbl, c in [('A_gene_only', 'Gene\nonly', '#55A868'), ('B_gene_plus_attn', 'Gene+\nAttn', '#8172B2'), ('C_gene_plus_corr', 'Gene+\nCorr', '#CCB974')]:
        if k in matched_cv:
            edge_names.append(lbl)
            edge_aurocs.append(matched_cv[k]['mean_auroc'])
            edge_stds.append(matched_cv[k].get('std_auroc', 0))
            edge_colors_d.append(c)
bars_d = ax.bar(range(len(edge_names)), edge_aurocs, yerr=edge_stds, capsize=4,
                color=edge_colors_d, edgecolor='black', linewidth=0.5, width=0.7)
ax.set_xticks(range(len(edge_names)))
ax.set_xticklabels(edge_names, fontsize=11)
ax.set_ylim(0.4, 0.95)
for i, v in enumerate(edge_aurocs):
    ax.text(i, v + edge_stds[i] + 0.008, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
ax.set_ylabel('AUROC', fontsize=16)
ax.set_title('(D) Edge evaluation\n(matched set)', fontsize=17, fontweight='bold')

# Panel E: Cross-fitted residualization
ax = axes[1, 1]
p4 = bat['part4_residualization']
# Keys: attention_baseline_auroc, attention_ols_full{r2, auroc_abs_residual, auroc_signed_residual}, etc.
res_methods = [
    ('Baseline', 'attention_baseline_auroc', 'correlation_baseline_auroc'),
    ('OLS full', 'attention_ols_full', 'correlation_ols_full'),
    ('OLS expr', 'attention_ols_expr_only', 'correlation_ols_expr_only'),
    ('OLS min', 'attention_ols_minimal', 'correlation_ols_minimal'),
    ('GBDT full', 'attention_gbdt_full', 'correlation_gbdt_full'),
]
res_labels = []
res_attn = []
res_corr = []
for label, attn_key, corr_key in res_methods:
    attn_val = p4.get(attn_key, {})
    corr_val = p4.get(corr_key, {})
    # Baseline is a scalar; others are dicts with auroc_abs_residual
    if isinstance(attn_val, (int, float)):
        res_labels.append(label)
        res_attn.append(attn_val)
        res_corr.append(corr_val if isinstance(corr_val, (int, float)) else 0.5)
    elif isinstance(attn_val, dict) and 'auroc_abs_residual' in attn_val:
        res_labels.append(label)
        res_attn.append(attn_val['auroc_abs_residual'])
        res_corr.append(corr_val.get('auroc_abs_residual', 0.5) if isinstance(corr_val, dict) else 0.5)
x_r = np.arange(len(res_labels))
w_r = 0.35
ax.bar(x_r - w_r/2, res_attn, w_r, label='Attention', color='#C44E52', edgecolor='black', linewidth=0.5)
ax.bar(x_r + w_r/2, res_corr, w_r, label='Correlation', color='#4C72B0', edgecolor='black', linewidth=0.5)
ax.set_xticks(x_r)
ax.set_xticklabels(res_labels, fontsize=11, rotation=15, ha='right')
ax.legend(fontsize=12)
for i in range(len(res_labels)):
    ax.text(i - w_r/2, res_attn[i] + 0.008, f'{res_attn[i]:.3f}', ha='center', fontsize=10)
    ax.text(i + w_r/2, res_corr[i] + 0.008, f'{res_corr[i]:.3f}', ha='center', fontsize=10)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
ax.set_ylabel('AUROC', fontsize=16)
ax.set_title('(E) Cross-fitted residualization', fontsize=17, fontweight='bold')

# Panel F: TRRUST direct targets
ax = axes[1, 2]
p5 = bat['part5_trrust_evaluation']
trrust_attn = p5['mean_attn_auroc']
trrust_corr = p5['mean_corr_auroc']
trrust_attn_std = p5['std_attn_auroc']
trrust_corr_std = p5['std_corr_auroc']
n_tfs = p5['n_evaluable_tfs']
bars_f = ax.bar([0, 1], [trrust_attn, trrust_corr], yerr=[trrust_attn_std, trrust_corr_std],
                 capsize=6, color=['#C44E52', '#4C72B0'], edgecolor='black', linewidth=0.5, width=0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Attention', 'Correlation'], fontsize=14)
ax.set_ylim(0, 1.0)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
ax.legend(fontsize=12)
for i, (m, s) in enumerate(zip([trrust_attn, trrust_corr], [trrust_attn_std, trrust_corr_std])):
    ax.text(i, m + s + 0.02, f'{m:.3f}', ha='center', fontsize=14, fontweight='bold')
ax.set_ylabel('AUROC', fontsize=16)
ax.set_title(f'(F) TRRUST direct targets\n(n={n_tfs} TFs)', fontsize=17, fontweight='bold')

fig.suptitle('RPE1 Confound Battery', fontsize=22, fontweight='bold', y=1.01)
plt.tight_layout(h_pad=4, w_pad=3)
outpath = os.path.join(FIG_DIR, 'fig_rpe1_confound_battery.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {outpath}")

print("\n=== All main paper figures regenerated ===")
