#!/usr/bin/env python3
"""
Round 9 Experiment 4: Full-Layer Perturbation-First Analysis
=============================================================
Evaluates perturbation-first AUROC for ALL 18 Geneformer V2-316M layers.
Uses pre-extracted edge data from round9_unified_extraction.

Addresses: Stress Test 3 ("evaluate all layers, not just 3")
"""

import os
import sys
import gc
import json
import pickle
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
EXTRACTION_DIR = os.path.join(PAPER_DIR, "experiments", "round9_unified_extraction")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round9_full_layer_analysis")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

N_LAYERS = 18
MIN_DE_TARGETS = 5


def run():
    print("=" * 70)
    print("FULL-LAYER PERTURBATION-FIRST ANALYSIS (18 layers)")
    print("=" * 70)

    # Load shared data
    print("\nLoading extraction data...")
    with open(os.path.join(EXTRACTION_DIR, "hvg_gene_list.json")) as f:
        hvg_genes = json.load(f)
    hvg_upper = [g.upper() for g in hvg_genes]
    hvg_to_local = {g.upper(): i for i, g in enumerate(hvg_genes)}

    with open(os.path.join(EXTRACTION_DIR, "de_labels.pkl"), 'rb') as f:
        de_labels = pickle.load(f)

    corr_matrix = np.load(os.path.join(EXTRACTION_DIR, "correlation_matrix.npy"))
    print(f"  {len(de_labels)} evaluable perturbations, {len(hvg_genes)} HVG genes")

    # Evaluate each layer
    print("\nEvaluating 18 layers...")

    # Store per-layer, per-perturbation AUROCs
    all_aurocs_attn = {}  # layer -> [auroc_per_perturbation]
    all_aurocs_corr = None  # Same for all layers
    pert_genes = sorted(de_labels.keys())
    rng = np.random.RandomState(42)

    # Compute correlation AUROCs once
    print("  Computing correlation baseline...")
    corr_aurocs = []
    valid_perts = []

    for pg in pert_genes:
        targets = de_labels[pg]
        pg_local = hvg_to_local.get(pg.upper())
        if pg_local is None:
            continue

        labels = []
        scores_corr = []
        for tgt_gene, is_de in targets.items():
            tgt_local = hvg_to_local.get(tgt_gene)
            if tgt_local is None:
                continue
            labels.append(1 if is_de else 0)
            scores_corr.append(abs(corr_matrix[pg_local, tgt_local]))

        labels = np.array(labels)
        if labels.sum() < MIN_DE_TARGETS or labels.sum() == len(labels):
            continue

        auroc = roc_auc_score(labels, np.array(scores_corr))
        corr_aurocs.append(auroc)
        valid_perts.append(pg)

    corr_aurocs = np.array(corr_aurocs)
    print(f"  Correlation baseline: mean AUROC = {corr_aurocs.mean():.4f} ± {corr_aurocs.std():.4f} "
          f"(n={len(corr_aurocs)})")

    # Evaluate each layer
    for layer_idx in range(N_LAYERS):
        edge_path = os.path.join(EXTRACTION_DIR, "layer_mean_edges", f"layer_{layer_idx:02d}_edges.pkl")
        if not os.path.exists(edge_path):
            print(f"  Layer {layer_idx}: MISSING")
            all_aurocs_attn[layer_idx] = np.full(len(valid_perts), 0.5)
            continue

        with open(edge_path, 'rb') as f:
            attn_edges = pickle.load(f)

        layer_aurocs = []
        for pg in valid_perts:
            targets = de_labels[pg]
            pg_local = hvg_to_local.get(pg.upper())

            labels = []
            scores_attn = []
            for tgt_gene, is_de in targets.items():
                tgt_local = hvg_to_local.get(tgt_gene)
                if tgt_local is None:
                    continue
                tgt_orig = hvg_genes[tgt_local]

                fwd = attn_edges.get((pg, tgt_orig), 0.0)
                rev = attn_edges.get((tgt_orig, pg), 0.0)
                attn_score = max(fwd, rev)

                labels.append(1 if is_de else 0)
                scores_attn.append(attn_score)

            labels = np.array(labels)
            if labels.sum() < MIN_DE_TARGETS or labels.sum() == len(labels):
                layer_aurocs.append(0.5)
                continue

            auroc = roc_auc_score(labels, np.array(scores_attn))
            layer_aurocs.append(auroc)

        all_aurocs_attn[layer_idx] = np.array(layer_aurocs)
        del attn_edges
        gc.collect()

        mean_auroc = all_aurocs_attn[layer_idx].mean()
        # Paired Wilcoxon vs correlation
        try:
            w_stat, w_p = stats.wilcoxon(all_aurocs_attn[layer_idx] - corr_aurocs)
        except:
            w_p = 1.0
        print(f"  Layer {layer_idx:2d}: AUROC = {mean_auroc:.4f}, "
              f"vs corr diff = {mean_auroc - corr_aurocs.mean():.4f}, "
              f"Wilcoxon p = {w_p:.4f}")

    # ============================================================
    # Split-sample validation
    # ============================================================
    print("\nSplit-sample validation...")
    np.random.seed(42)
    n_perts = len(valid_perts)
    perm = np.random.permutation(n_perts)
    half = n_perts // 2
    discovery_idx = perm[:half]
    validation_idx = perm[half:]

    # Discovery: find best layer
    discovery_means = {}
    for layer_idx in range(N_LAYERS):
        discovery_means[layer_idx] = all_aurocs_attn[layer_idx][discovery_idx].mean()
    best_layer = max(discovery_means, key=discovery_means.get)

    # Validation: evaluate best layer on held-out
    validation_auroc_best = all_aurocs_attn[best_layer][validation_idx].mean()
    validation_auroc_corr = corr_aurocs[validation_idx].mean()
    try:
        w_stat, w_p_val = stats.wilcoxon(
            all_aurocs_attn[best_layer][validation_idx] - corr_aurocs[validation_idx])
    except:
        w_p_val = 1.0

    print(f"  Discovery best layer: L{best_layer} (AUROC = {discovery_means[best_layer]:.4f})")
    print(f"  Validation: best layer AUROC = {validation_auroc_best:.4f}, "
          f"correlation = {validation_auroc_corr:.4f}, Wilcoxon p = {w_p_val:.4f}")

    # ============================================================
    # Save results
    # ============================================================
    results = {
        'n_perturbations': len(valid_perts),
        'correlation_baseline': {
            'mean_auroc': round(float(corr_aurocs.mean()), 4),
            'std_auroc': round(float(corr_aurocs.std()), 4),
        },
        'per_layer': {},
        'split_sample': {
            'discovery_best_layer': int(best_layer),
            'discovery_best_auroc': round(float(discovery_means[best_layer]), 4),
            'validation_best_auroc': round(float(validation_auroc_best), 4),
            'validation_corr_auroc': round(float(validation_auroc_corr), 4),
            'validation_wilcoxon_p': round(float(w_p_val), 6),
        },
    }

    for layer_idx in range(N_LAYERS):
        aurocs = all_aurocs_attn[layer_idx]
        try:
            w_stat, w_p = stats.wilcoxon(aurocs - corr_aurocs)
        except:
            w_p = 1.0
        results['per_layer'][str(layer_idx)] = {
            'layer_1indexed': layer_idx + 1,
            'mean_auroc': round(float(aurocs.mean()), 4),
            'std_auroc': round(float(aurocs.std()), 4),
            'median_auroc': round(float(np.median(aurocs)), 4),
            'diff_vs_corr': round(float(aurocs.mean() - corr_aurocs.mean()), 4),
            'wilcoxon_p_vs_corr': round(float(w_p), 6),
        }

    with open(os.path.join(OUT_DIR, "all_layer_perturbation_first.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Save per-layer per-gene matrix
    matrix_data = {'gene': valid_perts, 'auroc_corr': corr_aurocs.tolist()}
    for layer_idx in range(N_LAYERS):
        matrix_data[f'auroc_L{layer_idx}'] = all_aurocs_attn[layer_idx].tolist()
    pd.DataFrame(matrix_data).to_csv(os.path.join(OUT_DIR, "per_layer_per_gene.csv"), index=False)

    # ============================================================
    # Generate figure
    # ============================================================
    print("\nGenerating figure...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: AUROC by layer
    ax = axes[0]
    layer_means = [all_aurocs_attn[l].mean() for l in range(N_LAYERS)]
    layer_stds = [all_aurocs_attn[l].std() / np.sqrt(len(valid_perts)) for l in range(N_LAYERS)]
    ax.errorbar(range(N_LAYERS), layer_means, yerr=layer_stds, fmt='o-', color='#2196F3',
                capsize=3, label='Attention', markersize=4)
    ax.axhline(corr_aurocs.mean(), color='#FF9800', linestyle='--', linewidth=2,
               label=f'Correlation ({corr_aurocs.mean():.3f})')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Layer (0-indexed)')
    ax.set_ylabel('Mean AUROC vs DE targets')
    ax.set_title(f'(A) Full 18-layer perturbation-first\n(n={len(valid_perts)} perturbations)')
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, 17.5)

    # Panel B: Split-sample validation
    ax = axes[1]
    disc_means = [discovery_means[l] for l in range(N_LAYERS)]
    val_means = [all_aurocs_attn[l][validation_idx].mean() for l in range(N_LAYERS)]
    ax.scatter(disc_means, val_means, c=range(N_LAYERS), cmap='viridis', s=40, zorder=5)
    for l in range(N_LAYERS):
        ax.annotate(f'L{l}', (disc_means[l], val_means[l]), fontsize=6, ha='center', va='bottom')
    ax.plot([min(disc_means + val_means), max(disc_means + val_means)],
            [min(disc_means + val_means), max(disc_means + val_means)], 'k--', alpha=0.3)
    ax.axhline(validation_auroc_corr, color='#FF9800', linestyle='--', alpha=0.5)
    ax.axvline(corr_aurocs[discovery_idx].mean(), color='#FF9800', linestyle='--', alpha=0.5)
    ax.set_xlabel('Discovery AUROC')
    ax.set_ylabel('Validation AUROC')
    ax.set_title(f'(B) Split-sample validation\nBest: L{best_layer}')

    # Panel C: Distribution of per-perturbation differences (best layer vs corr)
    ax = axes[2]
    diff = all_aurocs_attn[best_layer] - corr_aurocs
    ax.hist(diff, bins=30, color='#2196F3', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(diff.mean(), color='#2196F3', linestyle='-', linewidth=2,
               label=f'Mean diff = {diff.mean():.4f}')
    ax.set_xlabel(f'AUROC difference (L{best_layer} attention - correlation)')
    ax.set_ylabel('Count')
    ax.set_title(f'(C) Per-perturbation differences\n(Wilcoxon p = {w_p_val:.4f})')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_full_layer_perturbation_first.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Figure saved to {fig_path}")

    print(f"\n{'=' * 70}")
    print("FULL-LAYER ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nBest layer: L{best_layer} (AUROC = {layer_means[best_layer]:.4f})")
    print(f"Correlation baseline: AUROC = {corr_aurocs.mean():.4f}")
    print(f"Split-sample: L{best_layer} validation AUROC = {validation_auroc_best:.4f}")


if __name__ == '__main__':
    run()
