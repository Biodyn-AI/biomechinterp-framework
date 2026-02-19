#!/usr/bin/env python3
"""
TRRUST circularity sensitivity analysis.

Tests whether key TRRUST-based conclusions hold when restricting to
direction-known entries (Activation/Repression), which require more
direct experimental evidence than "Unknown" entries.

Computes:
1. Global AUROC (full vs direction-known TRRUST)
2. Per-TF AUROC distributions
3. Number of evaluable TFs under each filter
4. Degree-null comparison

Uses Tabula Sapiens immune data (same as degree_preserving_nulls.py
and bootstrap_per_tf.py).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, mannwhitneyu
from sklearn.metrics import roc_auc_score

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)  # line-buffered

# ── Paths ──
DATA_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/data/raw"
REF_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks"
OUT_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/trrust_circularity_results"
os.makedirs(OUT_DIR, exist_ok=True)

TRRUST_PATH = os.path.join(REF_DIR, "trrust_human.tsv")
H5AD_PATH = os.path.join(DATA_DIR, "tabula_sapiens_immune_subset_20000.h5ad")


def load_trrust(direction_known_only=False):
    """Load TRRUST, optionally filtering to Activation/Repression only."""
    df = pd.read_csv(TRRUST_PATH, sep='\t', header=None,
                     names=['tf', 'target', 'mode', 'pmid'])
    if direction_known_only:
        df = df[df['mode'].isin(['Activation', 'Repression'])]
    pairs = set(zip(df['tf'].str.upper(), df['target'].str.upper()))
    print(f"  TRRUST ({'direction-known' if direction_known_only else 'full'}): "
          f"{len(pairs)} unique pairs, {len(set(t for t, _ in pairs))} TFs")
    return pairs


def compute_spearman_corr(X):
    """Spearman correlation matrix."""
    ranks = np.apply_along_axis(rankdata, 0, X)
    corr = np.corrcoef(ranks.T)
    return np.nan_to_num(corr, nan=0.0)


def evaluate_trrust(corr_matrix, gene_upper, trrust_pairs):
    """Compute global and per-TF AUROC against TRRUST."""
    n_genes = len(gene_upper)
    trrust_tfs = set(tf for tf, _ in trrust_pairs)

    # Find TFs in data
    tfs_in_data = sorted([g for g in set(gene_upper) if g in trrust_tfs])
    tf_indices = [gene_upper.index(tf) for tf in tfs_in_data]

    # Build label matrix
    labels_matrix = np.zeros((len(tf_indices), n_genes), dtype=int)
    for i, tf_idx in enumerate(tf_indices):
        tf_name = gene_upper[tf_idx]
        for j in range(n_genes):
            if j == tf_idx:
                continue
            if (tf_name, gene_upper[j]) in trrust_pairs:
                labels_matrix[i, j] = 1

    # Filter TFs with at least 1 positive and 1 negative
    evaluable = []
    for i, tf_idx in enumerate(tf_indices):
        mask = np.ones(n_genes, dtype=bool)
        mask[tf_idx] = False
        n_pos = labels_matrix[i, mask].sum()
        n_neg = mask.sum() - n_pos
        if n_pos > 0 and n_neg > 0:
            evaluable.append(i)

    print(f"  Evaluable TFs: {len(evaluable)} / {len(tf_indices)}")

    # Global AUROC
    scores_flat = []
    labels_flat = []
    for i in evaluable:
        tf_idx = tf_indices[i]
        for j in range(n_genes):
            if j == tf_idx:
                continue
            scores_flat.append(abs(corr_matrix[tf_idx, j]))
            labels_flat.append(labels_matrix[i, j])

    scores_flat = np.array(scores_flat)
    labels_flat = np.array(labels_flat)

    n_pos_total = labels_flat.sum()
    n_neg_total = len(labels_flat) - n_pos_total
    global_auroc = roc_auc_score(labels_flat, scores_flat)
    print(f"  Global AUROC: {global_auroc:.4f} ({n_pos_total} pos, {n_neg_total} neg)")

    # Per-TF AUROC
    per_tf = {}
    for i in evaluable:
        tf_idx = tf_indices[i]
        tf_name = gene_upper[tf_idx]
        mask = np.ones(n_genes, dtype=bool)
        mask[tf_idx] = False
        scores_tf = np.abs(corr_matrix[tf_idx, mask])
        labels_tf = labels_matrix[i, mask]
        n_pos = labels_tf.sum()
        if n_pos > 0 and n_pos < len(labels_tf):
            per_tf[tf_name] = {
                'auroc': float(roc_auc_score(labels_tf, scores_tf)),
                'n_pos': int(n_pos),
                'n_neg': int(len(labels_tf) - n_pos)
            }

    aurocs = [v['auroc'] for v in per_tf.values()]
    print(f"  Per-TF AUROC: mean={np.mean(aurocs):.3f}, "
          f"median={np.median(aurocs):.3f}, "
          f"IQR=[{np.percentile(aurocs,25):.3f}, {np.percentile(aurocs,75):.3f}]")

    # Count TFs with AUROC > 0.5 entirely (simple check)
    above_chance = sum(1 for a in aurocs if a > 0.5)
    print(f"  TFs with AUROC > 0.5: {above_chance}/{len(aurocs)}")

    return {
        'global_auroc': float(global_auroc),
        'n_evaluable_tfs': len(evaluable),
        'n_pos_total': int(n_pos_total),
        'n_neg_total': int(n_neg_total),
        'per_tf': per_tf,
        'per_tf_mean': float(np.mean(aurocs)),
        'per_tf_median': float(np.median(aurocs)),
        'per_tf_iqr': [float(np.percentile(aurocs, 25)), float(np.percentile(aurocs, 75))],
        'n_above_chance': above_chance
    }


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    print("=" * 60)
    print("TRRUST Circularity Sensitivity Analysis")
    print("=" * 60)

    # ── Load data (matching degree_preserving_nulls.py exactly) ──
    print("\n[1] Loading Tabula Sapiens immune data...")
    import scanpy as sc
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    # Map gene names (handle Categorical dtype)
    if 'feature_name' in adata.var.columns:
        from collections import Counter
        names = adata.var['feature_name'].astype(str).values.copy()
        counts = Counter()
        new_names = []
        for n in names:
            if counts[n] > 0:
                new_names.append(f"{n}-{counts[n]}")
            else:
                new_names.append(n)
            counts[n] += 1
        adata.var.index = pd.Index(new_names)

    # Preprocess (matching degree_preserving_nulls.py)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']].copy()
    gene_names = list(adata.var_names)
    gene_upper = [g.upper() for g in gene_names]
    print(f"  HVG genes: {len(gene_upper)}")

    # Subsample cells
    n_cells = min(5000, adata.n_obs)
    np.random.seed(42)
    cell_idx = np.random.choice(adata.n_obs, n_cells, replace=False)
    X = adata.X[cell_idx]
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X_hvg = np.asarray(X, dtype=np.float64)
    print(f"  After subsample: {X_hvg.shape}")

    # ── Compute correlation ──
    print("\n[2] Computing Spearman correlation matrix...")
    corr_matrix = compute_spearman_corr(X_hvg)
    print(f"  Correlation matrix shape: {corr_matrix.shape}")

    # ── Load TRRUST variants ──
    print("\n[3] Loading TRRUST (full)...")
    trrust_full = load_trrust(direction_known_only=False)

    print("\n[4] Loading TRRUST (direction-known only)...")
    trrust_filtered = load_trrust(direction_known_only=True)

    # ── Evaluate ──
    print("\n[5] Evaluating with FULL TRRUST...")
    results_full = evaluate_trrust(corr_matrix, gene_upper, trrust_full)

    print("\n[6] Evaluating with DIRECTION-KNOWN TRRUST...")
    results_filtered = evaluate_trrust(corr_matrix, gene_upper, trrust_filtered)

    # ── Compare ──
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Global AUROC:     Full={results_full['global_auroc']:.4f}  "
          f"Filtered={results_filtered['global_auroc']:.4f}  "
          f"Diff={results_filtered['global_auroc'] - results_full['global_auroc']:+.4f}")
    print(f"  Evaluable TFs:    Full={results_full['n_evaluable_tfs']}  "
          f"Filtered={results_filtered['n_evaluable_tfs']}")
    print(f"  Positive pairs:   Full={results_full['n_pos_total']}  "
          f"Filtered={results_filtered['n_pos_total']}")
    print(f"  Per-TF mean:      Full={results_full['per_tf_mean']:.3f}  "
          f"Filtered={results_filtered['per_tf_mean']:.3f}")
    print(f"  Per-TF median:    Full={results_full['per_tf_median']:.3f}  "
          f"Filtered={results_filtered['per_tf_median']:.3f}")
    print(f"  Above chance:     Full={results_full['n_above_chance']}/{results_full['n_evaluable_tfs']}  "
          f"Filtered={results_filtered['n_above_chance']}/{results_filtered['n_evaluable_tfs']}")

    # Per-TF comparison for shared TFs
    shared_tfs = set(results_full['per_tf'].keys()) & set(results_filtered['per_tf'].keys())
    if shared_tfs:
        diffs = []
        for tf in sorted(shared_tfs):
            a_full = results_full['per_tf'][tf]['auroc']
            a_filt = results_filtered['per_tf'][tf]['auroc']
            diffs.append(a_filt - a_full)
            print(f"    {tf:10s}  Full={a_full:.3f}  Filt={a_filt:.3f}  "
                  f"Diff={a_filt - a_full:+.3f}  "
                  f"(n_pos: {results_full['per_tf'][tf]['n_pos']} -> {results_filtered['per_tf'][tf]['n_pos']})")
        print(f"\n  Mean per-TF AUROC diff: {np.mean(diffs):+.4f}")
        print(f"  Median per-TF AUROC diff: {np.median(diffs):+.4f}")

    # ── Save results ──
    output = {
        'full_trrust': results_full,
        'direction_known_trrust': results_filtered,
        'comparison': {
            'global_auroc_diff': float(results_filtered['global_auroc'] - results_full['global_auroc']),
            'shared_tfs': len(shared_tfs),
            'mean_per_tf_diff': float(np.mean(diffs)) if shared_tfs else None
        }
    }

    out_path = os.path.join(OUT_DIR, "trrust_circularity_sensitivity.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\n  Results saved to {out_path}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    delta = results_filtered['global_auroc'] - results_full['global_auroc']
    if abs(delta) < 0.02:
        print("  Conclusions are ROBUST to TRRUST filtering.")
        print(f"  Global AUROC change ({delta:+.4f}) is negligible.")
    elif delta > 0.02:
        print(f"  Filtering IMPROVES AUROC by {delta:+.4f}.")
        print("  Direction-known entries are more aligned with edge scores.")
    else:
        print(f"  Filtering REDUCES AUROC by {delta:+.4f}.")
        print("  Some signal may come from Unknown-direction entries.")


if __name__ == "__main__":
    main()
