#!/usr/bin/env python3
"""Quick fixup: re-run TRRUST evaluation and figure generation from checkpoints."""
import os, sys, json, time
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
import numpy as np

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round13_propensity_matched")

# Add the script's parent dir so we can import functions
sys.path.insert(0, os.path.join(PAPER_DIR, "experiments"))

# Load the main script's functions
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pm", os.path.join(PAPER_DIR, "experiments", "round13_propensity_matched.py"))
pm = importlib.util.module_from_spec(spec)

# We need to manually define the functions since importing the whole module would run it.
# Instead, let's just re-do the TRRUST evaluation inline.

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# Load checkpoints
print("Loading checkpoints...")
with open(os.path.join(OUT_DIR, "matching_summary.json")) as f:
    match_summary = json.load(f)
with open(os.path.join(OUT_DIR, "evaluation_results.json")) as f:
    eval_results = json.load(f)
with open(os.path.join(OUT_DIR, "bootstrap_cis.json")) as f:
    boot_cis = json.load(f)

print(f"  Match summary: {match_summary['n_pos_after']} pos, {match_summary['n_pairs_after']} total")

# Re-compute TRRUST evaluation
print("\n[4/5] Direct-target evaluation (TRRUST)...")

# Load data
import pickle
EXTRACTION_DIR = os.path.join(PAPER_DIR, "experiments", "round9_unified_extraction")
with open(os.path.join(EXTRACTION_DIR, "de_labels.pkl"), 'rb') as f:
    de_labels = pickle.load(f)
with open(os.path.join(EXTRACTION_DIR, "layer_mean_edges", "layer_12_edges.pkl"), 'rb') as f:
    attn_edges = pickle.load(f)

import pandas as pd
corr_matrix = np.load(os.path.join(EXTRACTION_DIR, "correlation_matrix.npy"))
gene_features = pd.read_csv(os.path.join(EXTRACTION_DIR, "gene_features.csv"))
hvg_genes = gene_features['gene'].tolist()
hvg_to_idx = {g: i for i, g in enumerate(hvg_genes)}

# Load TRRUST
trrust_path = os.path.join(PAPER_DIR, "cssi_real", "results", "trrust_human.tsv")
trrust = {}
with open(trrust_path) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            tf = parts[0].upper()
            target = parts[1].upper()
            if tf not in trrust:
                trrust[tf] = set()
            trrust[tf].add(target)
print(f"  Loaded TRRUST: {len(trrust)} TFs, {sum(len(v) for v in trrust.values())} relationships")

# Build matched dataset for TRRUST evaluation
from sklearn.metrics import roc_auc_score, average_precision_score

# de_labels structure: {pert_gene: {target_gene: is_de, ...}, ...}
pert_genes_in_data = set(de_labels.keys())

# Filter to TFs in TRRUST with perturbation data
hvg_set = set(g.upper() for g in hvg_genes)
n_eval_tfs = 0
per_tf_attn_auroc = []
per_tf_corr_auroc = []
per_tf_attn_auprc = []
per_tf_corr_auprc = []
tf_details = []
n_direct_total = 0

for pg in pert_genes_in_data:
    pg_upper = pg.upper()
    if pg_upper not in trrust:
        continue
    direct_targets = trrust[pg_upper]

    pg_idx = hvg_to_idx.get(pg)
    if pg_idx is None:
        continue

    targets_dict = de_labels[pg]

    # Get all targets for this perturbation
    labels = []
    scores_attn = []
    scores_corr = []
    n_direct = 0

    for tg, is_de in targets_dict.items():
        tg_upper = tg.upper()
        if tg_upper == pg_upper:
            continue

        is_direct = 1 if tg_upper in direct_targets else 0
        if is_direct:
            n_direct += 1
        labels.append(is_direct)

        # Attention score
        fwd = attn_edges.get((pg, tg), 0.0)
        rev = attn_edges.get((tg, pg), 0.0)
        scores_attn.append(max(fwd, rev))

        # Correlation score
        tg_idx = hvg_to_idx.get(tg)
        if tg_idx is not None:
            scores_corr.append(abs(corr_matrix[pg_idx, tg_idx]))
        else:
            scores_corr.append(0.0)

    labels = np.array(labels)
    if labels.sum() < 2 or labels.sum() == len(labels):
        continue

    try:
        auroc_attn = roc_auc_score(labels, np.array(scores_attn))
        auroc_corr = roc_auc_score(labels, np.array(scores_corr))
        auprc_attn = average_precision_score(labels, np.array(scores_attn))
        auprc_corr = average_precision_score(labels, np.array(scores_corr))
    except ValueError:
        continue

    per_tf_attn_auroc.append(auroc_attn)
    per_tf_corr_auroc.append(auroc_corr)
    per_tf_attn_auprc.append(auprc_attn)
    per_tf_corr_auprc.append(auprc_corr)
    n_direct_total += n_direct
    n_eval_tfs += 1
    tf_details.append({
        'tf': pg,
        'n_direct': int(n_direct),
        'n_total': int(len(labels)),
        'attn_auroc': float(auroc_attn),
        'corr_auroc': float(auroc_corr),
        'attn_auprc': float(auprc_attn),
        'corr_auprc': float(auprc_corr),
    })

direct_results = {
    'n_evaluable_tfs': int(n_eval_tfs),
    'n_direct_targets_total': int(n_direct_total),
    'mean_attn_auroc': float(np.mean(per_tf_attn_auroc)) if per_tf_attn_auroc else None,
    'std_attn_auroc': float(np.std(per_tf_attn_auroc)) if per_tf_attn_auroc else None,
    'mean_corr_auroc': float(np.mean(per_tf_corr_auroc)) if per_tf_corr_auroc else None,
    'std_corr_auroc': float(np.std(per_tf_corr_auroc)) if per_tf_corr_auroc else None,
    'mean_attn_auprc': float(np.mean(per_tf_attn_auprc)) if per_tf_attn_auprc else None,
    'std_attn_auprc': float(np.std(per_tf_attn_auprc)) if per_tf_attn_auprc else None,
    'mean_corr_auprc': float(np.mean(per_tf_corr_auprc)) if per_tf_corr_auprc else None,
    'std_corr_auprc': float(np.std(per_tf_corr_auprc)) if per_tf_corr_auprc else None,
    'per_tf_details': tf_details,
}

print(f"  Evaluable TF perturbations: {n_eval_tfs}")
print(f"  Total direct targets: {n_direct_total}")
if per_tf_attn_auroc:
    print(f"  Attention:   AUROC={np.mean(per_tf_attn_auroc):.4f} "
          f"(+/-{np.std(per_tf_attn_auroc):.4f}), "
          f"AUPRC={np.mean(per_tf_attn_auprc):.4f} "
          f"(+/-{np.std(per_tf_attn_auprc):.4f})")
    print(f"  Correlation: AUROC={np.mean(per_tf_corr_auroc):.4f} "
          f"(+/-{np.std(per_tf_corr_auroc):.4f}), "
          f"AUPRC={np.mean(per_tf_corr_auprc):.4f} "
          f"(+/-{np.std(per_tf_corr_auprc):.4f})")

# Save
with open(os.path.join(OUT_DIR, "direct_target_results.json"), 'w') as f:
    json.dump(direct_results, f, indent=2, default=_json_default)
print("  [checkpoint: direct_target_results.json saved]")

# Generate figure
print("\n[5/5] Generating figure...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel A: Covariate balance
ax = axes[0, 0]
bal_before = match_summary.get('balance_before', {})
bal_after = match_summary.get('balance_after', {})
covariates = list(bal_before.keys())
if covariates:
    before = [abs(bal_before[c]['smd']) for c in covariates]
    after = [abs(bal_after[c]['smd']) for c in covariates]
    x = np.arange(len(covariates))
    ax.barh(x - 0.2, before, 0.35, label='Before', color='#e74c3c', alpha=0.8)
    ax.barh(x + 0.2, after, 0.35, label='After', color='#2ecc71', alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels([c.replace('tgt_', '') for c in covariates], fontsize=9)
    ax.axvline(x=0.1, color='black', linestyle='--', linewidth=0.8, label='SMD=0.1')
    ax.set_xlabel('|Standardized Mean Difference|')
    ax.set_title('A. Covariate Balance')
    ax.legend(fontsize=8)

# Panel B: Raw edge AUROC
ax = axes[0, 1]
raw = eval_results.get('raw_edge_results', {})
if raw:
    methods = ['Attention', 'Correlation']
    aurocs = [
        raw.get('attn_raw', {}).get('mean_auroc', 0.5),
        raw.get('corr_raw', {}).get('mean_auroc', 0.5),
    ]
    stds = [
        raw.get('attn_raw', {}).get('std_auroc', 0),
        raw.get('corr_raw', {}).get('std_auroc', 0),
    ]
    bars = ax.bar(methods, aurocs, yerr=stds, capsize=5,
                  color=['#e74c3c', '#3498db'], alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance')
    ax.set_ylabel('AUROC')
    ax.set_title('B. Raw Edge AUROC (Matched Data)')
    ax.legend(fontsize=8)
    for b, v in zip(bars, aurocs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

# Panel C: Incremental AUROC with bootstrap CIs
ax = axes[1, 0]
if boot_cis:
    models = ['Gene only', 'Gene + Attn', 'Gene + Corr']
    keys = ['A_gene_only', 'B_gene_plus_attn', 'C_gene_plus_corr']
    means = [boot_cis[k]['auroc_mean'] for k in keys]
    lo = [boot_cis[k]['auroc_ci_lo'] for k in keys]
    hi = [boot_cis[k]['auroc_ci_hi'] for k in keys]
    errs = [[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]]
    colors = ['#95a5a6', '#e74c3c', '#3498db']
    bars = ax.bar(models, means, yerr=errs, capsize=5,
                  color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('AUROC')
    ax.set_title('C. Incremental Value (GroupKFold, 95% CI)')
    for b, v, l, h in zip(bars, means, lo, hi):
        ax.text(b.get_x() + b.get_width()/2, h + 0.001,
                f'{v:.4f}\n[{l:.4f}, {h:.4f}]', ha='center', fontsize=7)
    ax.set_ylim(min(lo) - 0.01, max(hi) + 0.02)

# Panel D: Direct-target AUROC (TRRUST)
ax = axes[1, 1]
if direct_results.get('n_evaluable_tfs', 0) > 0:
    methods = ['Attention', 'Correlation']
    aurocs = [direct_results['mean_attn_auroc'], direct_results['mean_corr_auroc']]
    stds = [direct_results['std_attn_auroc'], direct_results['std_corr_auroc']]
    bars = ax.bar(methods, aurocs, yerr=stds, capsize=5,
                  color=['#e74c3c', '#3498db'], alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance')
    ax.set_ylabel('AUROC')
    ax.set_title(f'D. Direct-Target AUROC (TRRUST, n={direct_results["n_evaluable_tfs"]} TFs)')
    ax.legend(fontsize=8)
    for b, v in zip(bars, aurocs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "figures", "fig_propensity_matched.png")
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")

print("\nDone!")
