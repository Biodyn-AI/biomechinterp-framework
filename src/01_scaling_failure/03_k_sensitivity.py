#!/usr/bin/env python3
"""
Scaling K-Sensitivity Analysis
================================
Addresses Round 8 Major Concern 4:
  Show scaling behavior as a function of K (top-K threshold) and continuous-score AUROC.
  Demonstrates that inverse scaling persists across different K values and under
  continuous (unthresholded) AUROC evaluation.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import h5py
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCALING_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/outputs/scaling"
REF_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks"
PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "scaling_k_sensitivity_results")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

K_VALUES = [20, 50, 100, 200, 500]
TIERS = ['small', 'medium', 'large']
SEEDS = [1, 2, 3]
CELL_COUNTS = [200, 1000, 3000]


def load_trrust():
    path = os.path.join(REF_DIR, "trrust_human.tsv")
    df = pd.read_csv(path, sep='\t', header=None, names=['tf', 'target', 'mode', 'pmid'])
    return set(zip(df['tf'].str.upper(), df['target'].str.upper()))


def get_gene_names(run_dir):
    """Load gene names from the processed h5ad via h5py."""
    h5ad_path = os.path.join(run_dir, "tabula_sapiens_processed.h5ad")
    with h5py.File(h5ad_path, 'r') as f:
        names = f['var']['_index'][:]
        return [g.decode() if isinstance(g, bytes) else g for g in names]


def build_label_matrix(gene_names, trrust):
    """Build binary label matrix for TRRUST edges."""
    gene_upper = [g.upper() for g in gene_names]
    n = len(gene_names)
    labels = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and (gene_upper[i], gene_upper[j]) in trrust:
                labels[i, j] = True
    return labels


def compute_topk_f1(scores, labels, K):
    """Compute F1 for top-K edges per source gene."""
    n = scores.shape[0]
    tp, fp, fn = 0, 0, 0
    for i in range(n):
        row_scores = scores[i].copy()
        row_scores[i] = -np.inf  # exclude self
        top_k_idx = np.argsort(-row_scores)[:K]
        predicted = set(top_k_idx)
        true_targets = set(np.where(labels[i])[0])

        tp += len(predicted & true_targets)
        fp += len(predicted - true_targets)
        fn += len(true_targets - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def compute_continuous_auroc(scores, labels):
    """Compute AUROC using continuous (unthresholded) scores."""
    n = scores.shape[0]
    scores_flat = []
    labels_flat = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            scores_flat.append(scores[i, j])
            labels_flat.append(int(labels[i, j]))

    scores_flat = np.array(scores_flat)
    labels_flat = np.array(labels_flat)

    if labels_flat.sum() == 0 or labels_flat.sum() == len(labels_flat):
        return np.nan

    return roc_auc_score(labels_flat, scores_flat)


def run():
    print("=" * 60)
    print("SCALING K-SENSITIVITY ANALYSIS")
    print("=" * 60)

    trrust = load_trrust()
    print(f"TRRUST edges: {len(trrust)}")

    all_results = []

    for tier in TIERS:
        for seed in SEEDS:
            for n_cells in CELL_COUNTS:
                run_name = f"{tier}_kidney_cells{n_cells}_hvg2000_full_hvg_maxgenes1200_seed{seed}_topk100"
                run_dir = os.path.join(SCALING_DIR, run_name)

                if not os.path.exists(run_dir):
                    continue

                scores_path = os.path.join(run_dir, "attention_scores.npy")
                if not os.path.exists(scores_path):
                    continue

                print(f"\n  {run_name}")
                scores = np.load(scores_path)
                gene_names = get_gene_names(run_dir)

                if scores.shape[0] != len(gene_names):
                    print(f"    SKIP: score shape {scores.shape} != {len(gene_names)} genes")
                    continue

                labels = build_label_matrix(gene_names, trrust)
                n_pos = labels.sum()
                print(f"    Genes: {len(gene_names)}, TRRUST edges: {n_pos}")

                # Continuous AUROC
                auroc = compute_continuous_auroc(scores, labels)
                print(f"    Continuous AUROC: {auroc:.4f}")

                result_base = {
                    'tier': tier,
                    'seed': seed,
                    'n_cells': n_cells,
                    'n_genes': len(gene_names),
                    'n_trrust_edges': int(n_pos),
                    'continuous_auroc': round(float(auroc), 4) if not np.isnan(auroc) else None,
                }

                # K-specific metrics
                for K in K_VALUES:
                    prec, rec, f1 = compute_topk_f1(scores, labels, K)
                    result = {**result_base, 'K': K, 'precision': round(prec, 6),
                              'recall': round(rec, 6), 'f1': round(f1, 6)}
                    all_results.append(result)
                    print(f"    K={K:>3d}: P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUT_DIR, "k_sensitivity_results.csv"), index=False)

    # ========================================
    # Summarize 200→1000 degradation across K
    # ========================================
    print("\n" + "=" * 60)
    print("200→1000 DEGRADATION ACROSS K VALUES")
    print("=" * 60)

    degradation_summary = []
    for K in K_VALUES:
        f1_200 = results_df[(results_df['n_cells'] == 200) & (results_df['K'] == K)]['f1'].values
        f1_1000 = results_df[(results_df['n_cells'] == 1000) & (results_df['K'] == K)]['f1'].values

        # Pair by tier+seed
        pairs = []
        for tier in TIERS:
            for seed in SEEDS:
                v200 = results_df[(results_df['tier'] == tier) & (results_df['seed'] == seed) &
                                  (results_df['n_cells'] == 200) & (results_df['K'] == K)]['f1'].values
                v1000 = results_df[(results_df['tier'] == tier) & (results_df['seed'] == seed) &
                                   (results_df['n_cells'] == 1000) & (results_df['K'] == K)]['f1'].values
                if len(v200) == 1 and len(v1000) == 1:
                    pairs.append((v200[0], v1000[0]))

        n_degrade = sum(1 for a, b in pairs if b < a)
        mean_200 = np.mean([a for a, _ in pairs]) if pairs else 0
        mean_1000 = np.mean([b for _, b in pairs]) if pairs else 0
        print(f"  K={K:>3d}: {n_degrade}/{len(pairs)} degrade, "
              f"mean F1 200={mean_200:.4f} → 1000={mean_1000:.4f}")

        degradation_summary.append({
            'K': K,
            'n_pairs': len(pairs),
            'n_degrading': n_degrade,
            'mean_f1_200': round(mean_200, 4),
            'mean_f1_1000': round(mean_1000, 4),
        })

    # Continuous AUROC comparison
    auroc_200 = results_df[results_df['n_cells'] == 200].groupby(['tier', 'seed'])['continuous_auroc'].first()
    auroc_1000 = results_df[results_df['n_cells'] == 1000].groupby(['tier', 'seed'])['continuous_auroc'].first()
    common = auroc_200.index.intersection(auroc_1000.index)
    if len(common) > 0:
        a200 = auroc_200.loc[common].values
        a1000 = auroc_1000.loc[common].values
        n_degrade_auroc = (a1000 < a200).sum()
        print(f"\n  Continuous AUROC: {n_degrade_auroc}/{len(common)} degrade, "
              f"mean 200={a200.mean():.4f} → 1000={a1000.mean():.4f}")

    # ========================================
    # FIGURES
    # ========================================
    print("\nGenerating figures...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: F1 vs K for 200 vs 1000 cells (averaged across tiers/seeds)
    ax = axes[0]
    for n_cells, color, marker in [(200, '#4CAF50', 'o'), (1000, '#F44336', 's'), (3000, '#9E9E9E', '^')]:
        sub = results_df[results_df['n_cells'] == n_cells]
        if len(sub) == 0:
            continue
        mean_f1 = sub.groupby('K')['f1'].mean()
        std_f1 = sub.groupby('K')['f1'].std()
        ax.plot(mean_f1.index, mean_f1.values, f'{marker}-', color=color,
                markersize=8, label=f'N={n_cells}')
        ax.fill_between(mean_f1.index, mean_f1 - std_f1, mean_f1 + std_f1,
                        alpha=0.15, color=color)
    ax.set_xlabel('K (top-K edges per source)')
    ax.set_ylabel('F1 vs TRRUST')
    ax.set_title('(A) F1 vs K threshold\n(3 tiers × 3 seeds)')
    ax.legend(fontsize=9)
    ax.set_xscale('log')
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels(K_VALUES)

    # Panel B: Continuous AUROC by cell count (paired)
    ax = axes[1]
    cont_data = results_df.drop_duplicates(subset=['tier', 'seed', 'n_cells'])[
        ['tier', 'seed', 'n_cells', 'continuous_auroc']].dropna()
    for n_cells, color, pos in [(200, '#4CAF50', 0), (1000, '#F44336', 1), (3000, '#9E9E9E', 2)]:
        vals = cont_data[cont_data['n_cells'] == n_cells]['continuous_auroc'].values
        if len(vals) > 0:
            bp = ax.boxplot([vals], positions=[pos], widths=0.5,
                           patch_artist=True, showmeans=True)
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['N=200', 'N=1000', 'N=3000'])
    ax.set_ylabel('Continuous AUROC vs TRRUST')
    ax.set_title('(B) Continuous AUROC\n(no K thresholding)')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    # Panel C: Degradation rate across K
    ax = axes[2]
    deg_df = pd.DataFrame(degradation_summary)
    bars = ax.bar(range(len(K_VALUES)),
                  [d['n_degrading'] / d['n_pairs'] if d['n_pairs'] > 0 else 0
                   for d in degradation_summary],
                  color='#FF5722', alpha=0.7)
    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels([f'K={k}' for k in K_VALUES])
    ax.set_ylabel('Fraction of runs degrading\n(200→1000)')
    ax.set_title('(C) Degradation rate\nacross K thresholds')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1.1)
    for i, d in enumerate(degradation_summary):
        ax.text(i, d['n_degrading'] / max(d['n_pairs'], 1) + 0.03,
                f"{d['n_degrading']}/{d['n_pairs']}", ha='center', fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_scaling_k_sensitivity.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Figure saved to {fig_path}")

    # Save summary
    summary = {
        'experiment': 'scaling_k_sensitivity',
        'K_values': K_VALUES,
        'degradation_200_to_1000': degradation_summary,
        'continuous_auroc': {
            'mean_200': round(float(a200.mean()), 4) if len(common) > 0 else None,
            'mean_1000': round(float(a1000.mean()), 4) if len(common) > 0 else None,
            'n_degrade': int(n_degrade_auroc) if len(common) > 0 else None,
            'n_pairs': int(len(common)),
        },
    }
    with open(os.path.join(OUT_DIR, "k_sensitivity_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== EXPERIMENT COMPLETE ===")


if __name__ == '__main__':
    run()
