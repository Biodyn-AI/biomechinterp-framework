#!/usr/bin/env python3
"""
Analysis D Hardening: HVG Confound Test + Bootstrap CIs

Tests:
1. HVG confound: Re-run RPE1 restricted to top-2000 HVGs (no forced perturbation
   gene inclusion), matching K562's protocol. If attention advantage vanishes, the
   RPE1 result is an artifact of forced HVG inclusion.
2. Bootstrap CIs: 10,000 bootstrap samples of per-perturbation (attn-corr) deltas
   to get 95% CIs on mean delta for each dataset.
"""
import os, sys, json, pickle, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "context_dependence")
os.makedirs(OUT_DIR, exist_ok=True)

EXPERIMENTS = os.path.join(BASE, "..")
RPE1_DIR = os.path.join(EXPERIMENTS, "round13_non_k562_replication")
RPE1_H5AD = os.path.join(RPE1_DIR, "data", "ReplogleWeissman2022_rpe1.h5ad")
RPE1_ATTN = os.path.join(RPE1_DIR, "ReplogleRPE1", "attention_edges.pkl")
RPE1_DE = os.path.join(RPE1_DIR, "ReplogleRPE1", "de_cache.pkl")
RPE1_CSV = os.path.join(RPE1_DIR, "ReplogleRPE1", "per_gene_L15_ReplogleRPE1.csv")

# K562 data
K562_DIR = os.path.join(EXPERIMENTS, "round9_unified_extraction")
K562_CSV = os.path.join(BASE, "context_dependence", "k562_per_perturbation.csv")

CHECKPOINT = os.path.join(OUT_DIR, "harden_checkpoint.json")

def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(f"Not serializable: {type(o)}")

def save_checkpoint(key, data):
    cp = {}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            cp = json.load(f)
    cp[key] = data
    with open(CHECKPOINT, 'w') as f:
        json.dump(cp, f, indent=2, default=_json_default)

def load_checkpoint(key):
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            cp = json.load(f)
        return cp.get(key)
    return None

# ==============================================================
# PART 1: HVG CONFOUND TEST
# ==============================================================
def hvg_confound_test():
    """Re-run RPE1 with only top-2000 HVGs (no forced perturbation gene inclusion)."""
    import h5py

    cached = load_checkpoint("hvg_confound")
    if cached is not None:
        print("HVG confound test: loaded from checkpoint")
        return cached

    print("=" * 70)
    print("PART 1: HVG CONFOUND TEST")
    print("=" * 70)

    # 1a. Load RPE1 h5ad, compute top-2000 HVGs by variance on control cells
    print("\nLoading RPE1 h5ad...")
    t0 = time.time()
    with h5py.File(RPE1_H5AD, 'r') as f:
        gene_names = [g.decode() if isinstance(g, bytes) else str(g) for g in f['var']['gene_name'][:]]

        # Get perturbation labels to find control cells
        if 'categories' in f['obs']['perturbation']:
            cats = [c.decode() if isinstance(c, bytes) else str(c) for c in f['obs']['perturbation']['categories'][:]]
            codes = f['obs']['perturbation']['codes'][:]
            perturbations = [cats[c] for c in codes]
        else:
            perturbations = [p.decode() if isinstance(p, bytes) else str(p) for p in f['obs']['perturbation'][:]]

        perturbations = np.array(perturbations)

        # Identify control cells
        ctrl_mask = np.isin(perturbations, ['control', 'ctrl', 'non-targeting', 'NT'])
        ctrl_idx = np.where(ctrl_mask)[0]
        print(f"  Total cells: {len(perturbations)}, control cells: {len(ctrl_idx)}")

        # Subsample control cells (max 2000 for efficiency, matching original protocol)
        rng = np.random.RandomState(42)
        if len(ctrl_idx) > 2000:
            ctrl_idx = rng.choice(ctrl_idx, 2000, replace=False)

        # Load expression for control cells
        print(f"  Loading expression matrix for {len(ctrl_idx)} control cells...")
        X = f['X']
        # Read in chunks to avoid memory issues
        expr = np.zeros((len(ctrl_idx), len(gene_names)), dtype=np.float32)
        for i, idx in enumerate(ctrl_idx):
            expr[i] = X[idx, :]
            if (i + 1) % 500 == 0:
                print(f"    Loaded {i+1}/{len(ctrl_idx)} cells...")

    print(f"  h5ad loaded in {time.time()-t0:.1f}s")

    # 1b. Compute gene variance and select top-2000 HVGs
    print("\nComputing gene variances on control cells...")
    gene_var = np.var(expr, axis=0)
    gene_mean = np.mean(expr, axis=0)

    # Filter: only genes with nonzero variance
    nonzero_var = gene_var > 0
    print(f"  Genes with nonzero variance: {nonzero_var.sum()}/{len(gene_var)}")

    # Select top 2000 by variance (simple method matching K562 protocol)
    top_2000_idx = np.argsort(gene_var)[-2000:]
    top_2000_genes = set(np.array(gene_names)[top_2000_idx])
    print(f"  Top 2000 HVGs selected by variance")

    # Original 3309 genes (from per-gene CSV)
    orig_csv = pd.read_csv(RPE1_CSV)
    orig_genes_in_csv = set(orig_csv['gene'].values)
    print(f"  Original HVG set: {len(orig_genes_in_csv)} genes")
    print(f"  Overlap with restricted 2000: {len(top_2000_genes & orig_genes_in_csv)}")

    # 1c. Load DE labels
    print("\nLoading DE labels...")
    with open(RPE1_DE, 'rb') as f:
        de_cache = pickle.load(f)

    de_all_genes = de_cache.get('de_all_genes', de_cache.get('de_targets', {}))
    # Also check structure
    if 'de_results_by_threshold' in de_cache:
        # Use primary threshold
        primary = de_cache.get('primary_lfc', 0.5)
        if primary in de_cache['de_results_by_threshold']:
            entry = de_cache['de_results_by_threshold'][primary]
            if isinstance(entry, tuple) and len(entry) >= 1:
                de_all_genes_thr = entry[0]  # dict of pert -> set of DE genes
            else:
                de_all_genes_thr = entry

    print(f"  DE cache loaded. Available perturbations: {len(de_all_genes)}")

    # 1d. Load attention edges (layer 14 = layer 15 in 1-indexed)
    print("\nLoading attention edges...")
    with open(RPE1_ATTN, 'rb') as f:
        attn_edges = pickle.load(f)

    # Use layer 14 (0-indexed) = layer 15
    layer_key = 14
    if layer_key not in attn_edges:
        # Try other keys
        available_keys = list(attn_edges.keys())
        print(f"  Available layer keys: {available_keys}")
        layer_key = available_keys[-1]  # use last available

    attn_layer = attn_edges[layer_key]
    print(f"  Using layer {layer_key}, {len(attn_layer)} edges")

    # Get all genes that appear in attention edges
    attn_genes = set()
    for (g1, g2) in attn_layer.keys():
        attn_genes.add(g1)
        attn_genes.add(g2)
    print(f"  Genes in attention edges: {len(attn_genes)}")

    # 1e. Compute correlation on restricted 2000 HVGs
    print("\nComputing Spearman correlation on restricted 2000 HVGs...")
    t0 = time.time()

    # Map gene names to indices
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    restricted_genes = sorted(top_2000_genes)
    restricted_idx = [gene_to_idx[g] for g in restricted_genes if g in gene_to_idx]
    restricted_genes = [gene_names[i] for i in restricted_idx]
    restricted_gene_to_pos = {g: i for i, g in enumerate(restricted_genes)}

    # Subset expression
    expr_restricted = expr[:, restricted_idx]

    # Spearman correlation = Pearson on ranks
    ranked = np.apply_along_axis(stats.rankdata, 0, expr_restricted)
    corr_restricted = np.corrcoef(ranked.T)
    corr_restricted = np.nan_to_num(corr_restricted, nan=0.0)
    corr_restricted = np.abs(corr_restricted)

    print(f"  Correlation computed in {time.time()-t0:.1f}s, shape: {corr_restricted.shape}")

    # 1f. Compute per-perturbation AUROC on restricted gene set
    print("\nComputing per-perturbation AUROC (restricted 2000 HVGs)...")

    # Determine which perturbation genes are in the restricted set
    pert_genes_in_restricted = []
    pert_genes_missing = []

    for pert_gene in de_all_genes:
        if pert_gene in restricted_gene_to_pos and pert_gene in attn_genes:
            pert_genes_in_restricted.append(pert_gene)
        else:
            pert_genes_missing.append(pert_gene)

    print(f"  Perturbation genes in restricted HVGs: {len(pert_genes_in_restricted)}/{len(de_all_genes)}")
    print(f"  Perturbation genes dropped: {len(pert_genes_missing)}")

    results_restricted = []
    results_original = []

    for pi, pert_gene in enumerate(pert_genes_in_restricted):
        if (pi + 1) % 200 == 0:
            print(f"    Processing perturbation {pi+1}/{len(pert_genes_in_restricted)}...")

        # Get DE labels for this perturbation
        if isinstance(de_all_genes[pert_gene], set):
            de_targets = de_all_genes[pert_gene]
            # Build binary labels for all genes in restricted set
            labels = []
            attn_scores = []
            corr_scores = []
            for tgt in restricted_genes:
                if tgt == pert_gene:
                    continue
                is_de = tgt in de_targets
                labels.append(int(is_de))

                # Attention score
                a = attn_layer.get((pert_gene, tgt), attn_layer.get((tgt, pert_gene), 0.0))
                attn_scores.append(float(a))

                # Correlation score
                ci = restricted_gene_to_pos[pert_gene]
                cj = restricted_gene_to_pos[tgt]
                corr_scores.append(float(corr_restricted[ci, cj]))
        elif isinstance(de_all_genes[pert_gene], dict):
            labels = []
            attn_scores = []
            corr_scores = []
            for tgt, is_de in de_all_genes[pert_gene].items():
                if tgt not in restricted_gene_to_pos or tgt == pert_gene:
                    continue
                labels.append(int(is_de))
                a = attn_layer.get((pert_gene, tgt), attn_layer.get((tgt, pert_gene), 0.0))
                attn_scores.append(float(a))
                ci = restricted_gene_to_pos[pert_gene]
                cj = restricted_gene_to_pos[tgt]
                corr_scores.append(float(corr_restricted[ci, cj]))
        else:
            continue

        labels = np.array(labels)
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue

        auroc_attn = roc_auc_score(labels, attn_scores)
        auroc_corr = roc_auc_score(labels, corr_scores)

        results_restricted.append({
            'gene': pert_gene,
            'auroc_attn_restricted': auroc_attn,
            'auroc_corr_restricted': auroc_corr,
            'delta_restricted': auroc_attn - auroc_corr,
            'n_de': int(labels.sum()),
            'n_total': int(len(labels))
        })

    print(f"\n  Evaluable perturbations (restricted 2000): {len(results_restricted)}")

    # Load original results for comparison
    orig_csv = pd.read_csv(RPE1_CSV)
    orig_dict = {row['gene']: row for _, row in orig_csv.iterrows()}

    # Merge restricted and original
    both_genes = []
    for r in results_restricted:
        gene = r['gene']
        if gene in orig_dict:
            orig = orig_dict[gene]
            r['auroc_attn_original'] = float(orig['auroc_attn'])
            r['auroc_corr_original'] = float(orig['auroc_corr'])
            r['delta_original'] = float(orig['auroc_attn'] - orig['auroc_corr'])
            both_genes.append(r)

    print(f"  Perturbations with both restricted and original results: {len(both_genes)}")

    # Aggregate
    df = pd.DataFrame(both_genes)

    summary = {
        'n_restricted_hvgs': 2000,
        'n_original_hvgs': 3309,
        'n_pert_in_restricted': len(pert_genes_in_restricted),
        'n_pert_dropped': len(pert_genes_missing),
        'n_evaluable_restricted': len(results_restricted),
        'n_evaluable_both': len(both_genes),
        'restricted': {
            'mean_auroc_attn': float(df['auroc_attn_restricted'].mean()),
            'mean_auroc_corr': float(df['auroc_corr_restricted'].mean()),
            'mean_delta': float(df['delta_restricted'].mean()),
            'median_delta': float(df['delta_restricted'].median()),
            'frac_attn_wins': float((df['delta_restricted'] > 0).mean()),
        },
        'original': {
            'mean_auroc_attn': float(df['auroc_attn_original'].mean()),
            'mean_auroc_corr': float(df['auroc_corr_original'].mean()),
            'mean_delta': float(df['delta_original'].mean()),
            'median_delta': float(df['delta_original'].median()),
            'frac_attn_wins': float((df['delta_original'] > 0).mean()),
        },
    }

    # Paired test: does restricting to 2000 HVGs change the attention advantage?
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(df['delta_restricted'], df['delta_original'])
    summary['paired_test'] = {
        'test': 'Wilcoxon signed-rank',
        'statistic': float(wilcoxon_stat),
        'p_value': float(wilcoxon_p),
        'interpretation': 'Tests whether restricting HVGs changes the attention advantage'
    }

    # Effect of restriction on attention AUROC
    attn_diff = df['auroc_attn_restricted'].mean() - df['auroc_attn_original'].mean()
    corr_diff = df['auroc_corr_restricted'].mean() - df['auroc_corr_original'].mean()
    delta_diff = df['delta_restricted'].mean() - df['delta_original'].mean()

    summary['restriction_effect'] = {
        'attn_auroc_change': float(attn_diff),
        'corr_auroc_change': float(corr_diff),
        'delta_change': float(delta_diff),
        'interpretation': (
            f"Restricting to 2000 HVGs changes attention AUROC by {attn_diff:+.4f}, "
            f"correlation AUROC by {corr_diff:+.4f}, and delta by {delta_diff:+.4f}"
        )
    }

    # Save per-perturbation CSV
    df.to_csv(os.path.join(OUT_DIR, "rpe1_restricted_2000hvg.csv"), index=False)

    print(f"\n  RESULTS (restricted 2000 HVGs):")
    print(f"    Attention AUROC:  {summary['restricted']['mean_auroc_attn']:.4f}")
    print(f"    Correlation AUROC: {summary['restricted']['mean_auroc_corr']:.4f}")
    print(f"    Mean delta:       {summary['restricted']['mean_delta']:+.4f}")
    print(f"    Fraction attn wins: {summary['restricted']['frac_attn_wins']:.3f}")
    print(f"\n  COMPARISON (original 3309 HVGs, same perturbations):")
    print(f"    Attention AUROC:  {summary['original']['mean_auroc_attn']:.4f}")
    print(f"    Correlation AUROC: {summary['original']['mean_auroc_corr']:.4f}")
    print(f"    Mean delta:       {summary['original']['mean_delta']:+.4f}")
    print(f"    Fraction attn wins: {summary['original']['frac_attn_wins']:.3f}")
    print(f"\n  Paired Wilcoxon: p = {wilcoxon_p:.4g}")
    print(f"  Delta change: {delta_diff:+.4f}")

    save_checkpoint("hvg_confound", summary)
    return summary


# ==============================================================
# PART 2: BOOTSTRAP CIs ON PER-DATASET ATTENTION ADVANTAGE
# ==============================================================
def bootstrap_cis():
    """Bootstrap CIs on mean delta (attn - corr) for each dataset."""

    cached = load_checkpoint("bootstrap_cis")
    if cached is not None:
        print("Bootstrap CIs: loaded from checkpoint")
        return cached

    print("\n" + "=" * 70)
    print("PART 2: BOOTSTRAP CIs ON ATTENTION ADVANTAGE")
    print("=" * 70)

    N_BOOT = 10000
    rng = np.random.RandomState(42)

    # Load per-perturbation data for each dataset
    datasets = {}

    # RPE1
    rpe1_csv = os.path.join(OUT_DIR, "rpe1_per_perturbation.csv")
    if os.path.exists(rpe1_csv):
        df = pd.read_csv(rpe1_csv)
        if 'delta' in df.columns:
            datasets['RPE1_CRISPRi'] = df['delta'].values
        elif 'auroc_attn' in df.columns and 'auroc_corr' in df.columns:
            datasets['RPE1_CRISPRi'] = (df['auroc_attn'] - df['auroc_corr']).values

    # K562
    k562_csv = os.path.join(OUT_DIR, "k562_per_perturbation.csv")
    if os.path.exists(k562_csv):
        df = pd.read_csv(k562_csv)
        if 'delta' in df.columns:
            datasets['K562_CRISPRi'] = df['delta'].values
        elif 'auroc_attn' in df.columns and 'auroc_corr' in df.columns:
            datasets['K562_CRISPRi'] = (df['auroc_attn'] - df['auroc_corr']).values

    # Also add restricted RPE1 if available
    restricted_csv = os.path.join(OUT_DIR, "rpe1_restricted_2000hvg.csv")
    if os.path.exists(restricted_csv):
        df = pd.read_csv(restricted_csv)
        if 'delta_restricted' in df.columns:
            datasets['RPE1_restricted_2000'] = df['delta_restricted'].values

    results = {}
    for name, deltas in datasets.items():
        print(f"\n  {name}: n={len(deltas)} perturbations")

        observed_mean = float(np.mean(deltas))
        observed_median = float(np.median(deltas))

        # Bootstrap mean
        boot_means = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            sample = rng.choice(deltas, size=len(deltas), replace=True)
            boot_means[b] = np.mean(sample)

        ci_mean_lo = float(np.percentile(boot_means, 2.5))
        ci_mean_hi = float(np.percentile(boot_means, 97.5))

        # Bootstrap median
        boot_medians = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            sample = rng.choice(deltas, size=len(deltas), replace=True)
            boot_medians[b] = np.median(sample)

        ci_med_lo = float(np.percentile(boot_medians, 2.5))
        ci_med_hi = float(np.percentile(boot_medians, 97.5))

        # One-sample Wilcoxon signed-rank test (delta != 0)
        if len(deltas) > 10:
            w_stat, w_p = stats.wilcoxon(deltas)
        else:
            w_stat, w_p = np.nan, np.nan

        results[name] = {
            'n': int(len(deltas)),
            'mean_delta': observed_mean,
            'median_delta': observed_median,
            'bootstrap_mean_95ci': [ci_mean_lo, ci_mean_hi],
            'bootstrap_median_95ci': [ci_med_lo, ci_med_hi],
            'wilcoxon_signed_rank_p': float(w_p) if not np.isnan(w_p) else None,
            'ci_excludes_zero_mean': bool(ci_mean_lo > 0 or ci_mean_hi < 0),
            'ci_excludes_zero_median': bool(ci_med_lo > 0 or ci_med_hi < 0),
        }

        print(f"    Mean delta: {observed_mean:+.4f} [{ci_mean_lo:+.4f}, {ci_mean_hi:+.4f}]")
        print(f"    Median delta: {observed_median:+.4f} [{ci_med_lo:+.4f}, {ci_med_hi:+.4f}]")
        print(f"    Wilcoxon p: {w_p:.4g}" if not np.isnan(w_p) else "    Wilcoxon: N/A (too few)")
        print(f"    CI excludes zero (mean): {results[name]['ci_excludes_zero_mean']}")

    save_checkpoint("bootstrap_cis", results)
    return results


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ANALYSIS D HARDENING")
    print("=" * 70)

    # Part 1: HVG confound test
    hvg_results = hvg_confound_test()

    # Part 2: Bootstrap CIs
    bootstrap_results = bootstrap_cis()

    # ==============================================================
    # SAVE COMBINED RESULTS
    # ==============================================================
    print("\n" + "=" * 70)
    print("SAVING COMBINED RESULTS")
    print("=" * 70)

    combined = {
        'hvg_confound_test': hvg_results,
        'bootstrap_cis': bootstrap_results,
    }

    with open(os.path.join(OUT_DIR, "hardened_results.json"), 'w') as f:
        json.dump(combined, f, indent=2, default=_json_default)

    # Generate summary markdown
    md = ["# Analysis D: Context-Dependence — Hardened Results\n"]
    md.append("## 1. HVG Confound Test\n")
    md.append("**Question:** Does the RPE1 attention advantage survive when restricted to top-2000 HVGs")
    md.append("(matching K562 protocol, no forced perturbation gene inclusion)?\n")

    h = hvg_results
    md.append(f"| Metric | Original (3309 HVGs) | Restricted (2000 HVGs) |")
    md.append(f"|--------|---------------------|----------------------|")
    md.append(f"| n evaluable perturbations | {h['n_evaluable_both']} | {h['n_evaluable_restricted']} |")
    md.append(f"| Mean attention AUROC | {h['original']['mean_auroc_attn']:.4f} | {h['restricted']['mean_auroc_attn']:.4f} |")
    md.append(f"| Mean correlation AUROC | {h['original']['mean_auroc_corr']:.4f} | {h['restricted']['mean_auroc_corr']:.4f} |")
    md.append(f"| Mean Δ (attn − corr) | {h['original']['mean_delta']:+.4f} | {h['restricted']['mean_delta']:+.4f} |")
    md.append(f"| Fraction attn wins | {h['original']['frac_attn_wins']:.3f} | {h['restricted']['frac_attn_wins']:.3f} |")
    md.append(f"| Perturbations dropped | 0 | {h['n_pert_dropped']} |")
    md.append(f"\nPaired Wilcoxon (delta change): p = {h['paired_test']['p_value']:.4g}")
    md.append(f"\n**{h['restriction_effect']['interpretation']}**\n")

    md.append("\n## 2. Bootstrap Confidence Intervals (10,000 samples)\n")
    md.append("| Dataset | n | Mean Δ | 95% CI (mean) | Median Δ | 95% CI (median) | Wilcoxon p |")
    md.append("|---------|---|--------|--------------|----------|----------------|------------|")
    for name, r in bootstrap_results.items():
        ci_m = r['bootstrap_mean_95ci']
        ci_med = r['bootstrap_median_95ci']
        wp = f"{r['wilcoxon_signed_rank_p']:.4g}" if r['wilcoxon_signed_rank_p'] else "N/A"
        md.append(f"| {name} | {r['n']} | {r['mean_delta']:+.4f} | [{ci_m[0]:+.4f}, {ci_m[1]:+.4f}] | {r['median_delta']:+.4f} | [{ci_med[0]:+.4f}, {ci_med[1]:+.4f}] | {wp} |")

    md.append("\n")

    summary_path = os.path.join(OUT_DIR, "hardened_summary.md")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(md))

    print(f"\nSaved: {summary_path}")
    print("\nAnalysis D hardening COMPLETE")
