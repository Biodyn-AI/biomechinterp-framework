#!/usr/bin/env python3
"""
Round 10 Experiment 1: Nested Layer/Head Selection Protocol
============================================================
Strict 5-fold nested CV: layer selection INSIDE each fold,
Bonferroni-corrected confirmatory p-value.

Addresses: MC2 ("layer-selection inference remains vulnerable to search effects")
           ST3 ("nested layer/head selection protocol")
"""

import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
DATA_PATH = os.path.join(PAPER_DIR, "experiments", "round9_full_layer_analysis", "per_layer_per_gene.csv")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round10_nested_layer_selection")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS = 5
N_LAYERS = 18
N_BOOTSTRAP = 1000
SEED = 42

def run():
    print("=" * 70)
    print("NESTED LAYER/HEAD SELECTION PROTOCOL")
    print("=" * 70)

    # Load pre-computed per-layer per-gene AUROCs
    print("\nLoading per_layer_per_gene.csv...")
    df = pd.read_csv(DATA_PATH)
    n_perts = len(df)
    print(f"  {n_perts} perturbations, {N_LAYERS} layers + correlation baseline")

    corr_aurocs = df['auroc_corr'].values
    attn_matrix = np.zeros((n_perts, N_LAYERS))
    for l in range(N_LAYERS):
        attn_matrix[:, l] = df[f'auroc_L{l}'].values

    # 5-fold nested CV
    print(f"\nRunning {N_FOLDS}-fold nested CV (seed {SEED})...")
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_perts)
    fold_size = n_perts // N_FOLDS

    fold_results = []
    pooled_diffs = np.zeros(n_perts)  # each perturbation held out exactly once
    pooled_idx = np.zeros(n_perts, dtype=int)  # track which fold each pert went to

    for k in range(N_FOLDS):
        start = k * fold_size
        end = (k + 1) * fold_size if k < N_FOLDS - 1 else n_perts
        test_idx = perm[start:end]
        train_idx = np.concatenate([perm[:start], perm[end:]])

        # Inner: select best layer on training perturbations
        train_means = attn_matrix[train_idx].mean(axis=0)
        best_layer = int(np.argmax(train_means))
        best_train_auroc = float(train_means[best_layer])
        corr_train_auroc = float(corr_aurocs[train_idx].mean())

        # Outer: evaluate on held-out
        test_attn = attn_matrix[test_idx, best_layer]
        test_corr = corr_aurocs[test_idx]
        diffs = test_attn - test_corr

        pooled_diffs[test_idx] = diffs
        pooled_idx[test_idx] = k

        fold_mean_attn = float(test_attn.mean())
        fold_mean_corr = float(test_corr.mean())
        fold_mean_diff = float(diffs.mean())

        fold_w, fold_p = stats.wilcoxon(diffs) if len(diffs) > 10 else (0, 1.0)
        n_attn_better = int((diffs > 0).sum())

        fold_results.append({
            'fold': k,
            'n_test': len(test_idx),
            'selected_layer': best_layer,
            'selected_layer_1indexed': best_layer + 1,
            'train_auroc_selected': best_train_auroc,
            'train_auroc_corr': corr_train_auroc,
            'test_auroc_selected': fold_mean_attn,
            'test_auroc_corr': fold_mean_corr,
            'test_delta': fold_mean_diff,
            'fold_wilcoxon_p': float(fold_p),
            'n_attn_better': n_attn_better,
        })
        print(f"  Fold {k}: selected L{best_layer} (1-idx: L{best_layer+1}), "
              f"test AUROC {fold_mean_attn:.4f} vs corr {fold_mean_corr:.4f}, "
              f"delta = {fold_mean_diff:+.4f}, n_better = {n_attn_better}/{len(test_idx)}")

    # Pooled confirmatory test
    print(f"\nPooled confirmatory test ({n_perts} perturbations)...")
    pooled_mean = float(pooled_diffs.mean())
    pooled_median = float(np.median(pooled_diffs))
    w_stat, w_p = stats.wilcoxon(pooled_diffs)
    bonferroni_p = min(float(w_p) * N_LAYERS, 1.0)  # correct for 18-layer search
    n_positive = int((pooled_diffs > 0).sum())

    print(f"  Pooled delta: mean = {pooled_mean:+.4f}, median = {pooled_median:+.4f}")
    print(f"  Wilcoxon p = {w_p:.6f}")
    print(f"  Bonferroni p (×{N_LAYERS}) = {bonferroni_p:.6f}")
    print(f"  Attn > corr: {n_positive}/{n_perts}")

    # Bootstrap CI on pooled delta
    print(f"\nBootstrap CI ({N_BOOTSTRAP} iterations)...")
    boot_deltas = []
    rng_boot = np.random.RandomState(SEED + 1)
    for b in range(N_BOOTSTRAP):
        idx = rng_boot.choice(n_perts, size=n_perts, replace=True)
        boot_deltas.append(float(pooled_diffs[idx].mean()))
    boot_deltas = np.array(boot_deltas)
    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    # Layer selection stability
    selected_layers = [f['selected_layer'] for f in fold_results]
    unique_selected = list(set(selected_layers))
    stability = len(unique_selected) == 1
    print(f"\nLayer stability: {selected_layers} -> {'STABLE' if stability else 'VARIABLE'}")

    # Cohen's d
    d = pooled_mean / pooled_diffs.std() if pooled_diffs.std() > 0 else 0
    print(f"  Cohen's d = {d:.4f}")

    # Compile results
    results = {
        'experiment': 'round10_nested_layer_selection',
        'protocol': 'strict_5fold_nested_cv',
        'n_perturbations': n_perts,
        'n_layers': N_LAYERS,
        'n_folds': N_FOLDS,
        'seed': SEED,
        'fold_results': fold_results,
        'selected_layers': selected_layers,
        'layer_selection_stable': stability,
        'pooled_delta_mean': pooled_mean,
        'pooled_delta_median': pooled_median,
        'pooled_wilcoxon_p': float(w_p),
        'bonferroni_p': bonferroni_p,
        'n_positive': n_positive,
        'bootstrap_ci_95': [ci_lo, ci_hi],
        'cohens_d': float(d),
    }

    # Generate figure
    print("\nGenerating figure...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Selected layer per fold
    ax = axes[0]
    fold_nums = range(N_FOLDS)
    ax.bar(fold_nums, selected_layers, color='steelblue', alpha=0.8)
    ax.set_xlabel('Outer Fold')
    ax.set_ylabel('Selected Layer (0-indexed)')
    ax.set_title('A. Layer Selected per Fold')
    ax.set_xticks(fold_nums)
    ax.set_ylim(0, N_LAYERS)
    for i, l in enumerate(selected_layers):
        ax.text(i, l + 0.3, f'L{l}', ha='center', fontsize=9)

    # Panel B: Held-out AUROC by fold
    ax = axes[1]
    attn_vals = [f['test_auroc_selected'] for f in fold_results]
    corr_vals = [f['test_auroc_corr'] for f in fold_results]
    x = np.arange(N_FOLDS)
    w = 0.35
    ax.bar(x - w/2, attn_vals, w, label='Selected Layer', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, corr_vals, w, label='Correlation', color='coral', alpha=0.8)
    ax.set_xlabel('Outer Fold')
    ax.set_ylabel('Held-out AUROC')
    ax.set_title('B. Held-out Performance')
    ax.set_xticks(x)
    ax.legend(fontsize=8)

    # Panel C: Pooled delta distribution
    ax = axes[2]
    ax.hist(pooled_diffs, bins=40, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No advantage')
    ax.axvline(pooled_mean, color='darkblue', linestyle='-', linewidth=1.5,
               label=f'Mean = {pooled_mean:+.3f}')
    ax.set_xlabel('AUROC Difference (Selected Layer − Correlation)')
    ax.set_ylabel('Count')
    ax.set_title(f'C. Pooled Δ (Bonf. p = {bonferroni_p:.4f})')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_nested_layer_selection.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Save results
    with open(os.path.join(OUT_DIR, "nested_cv_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}")

    print("\n" + "=" * 70)
    print("NESTED LAYER SELECTION COMPLETE")
    print("=" * 70)
    print(f"  Pooled delta: {pooled_mean:+.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Bonferroni p: {bonferroni_p:.6f}")
    print(f"  Layer stability: {selected_layers}")

if __name__ == "__main__":
    run()
