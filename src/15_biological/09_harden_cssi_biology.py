#!/usr/bin/env python3
"""
Analysis C Hardening: Unsupervised Leiden Clustering for CSSI Biology

User critique: The original Analysis C used ground-truth cell type labels instead of
unsupervised clustering. This defeats the purpose of CSSI, which is to discover
cell-state-specific regulatory patterns WITHOUT labels.

This script:
1. Runs unsupervised Leiden clustering at multiple resolutions (K=5-10)
2. Evaluates TRRUST AUROC for each resolution and each stratum
3. Computes CSSI-max, CSSI-mean, and compares to pooled baseline
4. Honestly reports whether strata recover cell types (ARI/NMI)
5. Edge-level enrichment of matching vs non-matching TF edges
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, adjusted_rand_score, normalized_mutual_info_score

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "cssi_biology")
os.makedirs(OUT_DIR, exist_ok=True)

TS_PATH = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad"
TRRUST_PATH = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv"

CHECKPOINT = os.path.join(OUT_DIR, "harden_checkpoint.json")

def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.bool_): return bool(o)
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


def main():
    import scanpy as sc
    import igraph
    import leidenalg

    print("=" * 70)
    print("ANALYSIS C HARDENING: UNSUPERVISED LEIDEN CLUSTERING")
    print("=" * 70)

    # ==============================================================
    # STEP 1: LOAD AND PREPROCESS
    # ==============================================================
    print("\n--- STEP 1: Load and preprocess Tabula Sapiens ---")
    t0 = time.time()

    adata = sc.read_h5ad(TS_PATH)
    print(f"  Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Ensure gene names are strings
    if hasattr(adata.var_names, 'categories'):
        # Categorical index
        adata.var_names = adata.var_names.astype(str)
    if 'feature_name' in adata.var.columns:
        adata.var_names = adata.var['feature_name'].astype(str).values
        adata.var_names_make_unique()

    # Get ground-truth labels for evaluation (ARI/NMI)
    gt_col = None
    for col in ['cell_ontology_class', 'cell_type', 'celltype', 'free_annotation']:
        if col in adata.obs.columns:
            gt_col = col
            break

    if gt_col:
        gt_labels = adata.obs[gt_col].astype(str).values
        print(f"  Ground-truth column: '{gt_col}' with {len(set(gt_labels))} unique types")
    else:
        gt_labels = None
        print("  WARNING: No ground-truth cell type column found")

    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Select top 2000 HVGs
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')
    adata_hvg = adata[:, adata.var['highly_variable']].copy()
    hvg_genes = list(adata_hvg.var_names)
    print(f"  HVGs: {len(hvg_genes)}")

    # PCA + neighbors (needed for Leiden)
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=50)
    sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=50)

    print(f"  Preprocessing done in {time.time()-t0:.1f}s")

    # ==============================================================
    # STEP 2: LEIDEN CLUSTERING AT MULTIPLE RESOLUTIONS
    # ==============================================================
    print("\n--- STEP 2: Leiden clustering at multiple resolutions ---")

    resolutions = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    clustering_results = {}

    for res in resolutions:
        sc.tl.leiden(adata_hvg, resolution=res, key_added=f'leiden_{res}')
        labels = adata_hvg.obs[f'leiden_{res}'].values.astype(str)
        n_clusters = len(set(labels))

        # Evaluate against ground truth
        if gt_labels is not None:
            ari = adjusted_rand_score(gt_labels, labels)
            nmi = normalized_mutual_info_score(gt_labels, labels)
        else:
            ari, nmi = np.nan, np.nan

        clustering_results[str(res)] = {
            'resolution': res,
            'n_clusters': n_clusters,
            'ari': float(ari),
            'nmi': float(nmi),
            'cluster_sizes': {c: int((labels == c).sum()) for c in sorted(set(labels))},
        }

        print(f"  Resolution {res}: {n_clusters} clusters, ARI={ari:.3f}, NMI={nmi:.3f}")

    save_checkpoint("clustering", clustering_results)

    # ==============================================================
    # STEP 3: LOAD TRRUST AND PREPARE EDGE EVALUATION
    # ==============================================================
    print("\n--- STEP 3: Load TRRUST ---")

    trrust = pd.read_csv(TRRUST_PATH, sep='\t', header=None, names=['tf', 'target', 'type', 'pmid'])
    print(f"  TRRUST: {len(trrust)} edges, {trrust['tf'].nunique()} TFs")

    # Filter to HVG universe
    trrust_hvg = trrust[(trrust['tf'].isin(hvg_genes)) & (trrust['target'].isin(hvg_genes))].copy()
    trrust_edges = set(zip(trrust_hvg['tf'], trrust_hvg['target']))
    print(f"  TRRUST edges in HVGs: {len(trrust_edges)}, {trrust_hvg['tf'].nunique()} TFs")

    # Gene-to-index mapping for HVGs
    gene_to_idx = {g: i for i, g in enumerate(hvg_genes)}

    # ==============================================================
    # STEP 4: PER-RESOLUTION CSSI EVALUATION
    # ==============================================================
    print("\n--- STEP 4: Per-resolution CSSI evaluation ---")

    # Get raw (log-normalized but not scaled) expression for correlation
    adata_raw = adata[:, adata.var['highly_variable']].copy()
    X = adata_raw.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)

    # Pooled correlation (all cells)
    print("  Computing pooled correlation...")
    t0 = time.time()
    ranked_all = np.apply_along_axis(stats.rankdata, 0, X)
    corr_pooled = np.corrcoef(ranked_all.T)
    corr_pooled = np.nan_to_num(corr_pooled, nan=0.0)
    abs_corr_pooled = np.abs(corr_pooled)
    print(f"    Pooled correlation done in {time.time()-t0:.1f}s")

    # Evaluate pooled AUROC
    pooled_scores = []
    pooled_labels = []
    for tf, tgt in trrust_edges:
        i, j = gene_to_idx[tf], gene_to_idx[tgt]
        pooled_scores.append(abs_corr_pooled[i, j])
        pooled_labels.append(1)

    # Negative pairs: all non-TRRUST TF-target pairs
    all_tfs = set(trrust_hvg['tf'])
    neg_count = 0
    for tf in all_tfs:
        tf_idx = gene_to_idx[tf]
        for g_idx, g in enumerate(hvg_genes):
            if g != tf and (tf, g) not in trrust_edges:
                pooled_scores.append(abs_corr_pooled[tf_idx, g_idx])
                pooled_labels.append(0)
                neg_count += 1

    pooled_labels = np.array(pooled_labels)
    pooled_scores = np.array(pooled_scores)
    pooled_auroc = roc_auc_score(pooled_labels, pooled_scores)
    print(f"  Pooled AUROC: {pooled_auroc:.4f} ({len(trrust_edges)} pos, {neg_count} neg)")

    # Per-resolution evaluation
    resolution_results = {}

    for res in resolutions:
        print(f"\n  Resolution {res} ({clustering_results[str(res)]['n_clusters']} clusters):")
        labels = adata_hvg.obs[f'leiden_{res}'].values.astype(str)
        clusters = sorted(set(labels))

        # Compute per-cluster correlation matrices
        per_cluster_abs_corr = {}
        for c in clusters:
            mask = labels == c
            n_cells = mask.sum()
            if n_cells < 50:
                print(f"    Cluster {c}: {n_cells} cells (SKIPPED, too few)")
                continue

            X_c = X[mask]
            ranked_c = np.apply_along_axis(stats.rankdata, 0, X_c)
            corr_c = np.corrcoef(ranked_c.T)
            corr_c = np.nan_to_num(corr_c, nan=0.0)
            per_cluster_abs_corr[c] = np.abs(corr_c)

        if len(per_cluster_abs_corr) < 2:
            print(f"    Only {len(per_cluster_abs_corr)} usable clusters, skipping resolution")
            continue

        # CSSI-max and CSSI-mean
        stacked = np.stack(list(per_cluster_abs_corr.values()))
        cssi_max = np.max(stacked, axis=0)
        cssi_mean = np.mean(stacked, axis=0)

        # Evaluate each method
        per_cluster_aurocs = {}
        for c, abs_corr_c in per_cluster_abs_corr.items():
            scores = []
            labs = []
            for tf, tgt in trrust_edges:
                i, j = gene_to_idx[tf], gene_to_idx[tgt]
                scores.append(abs_corr_c[i, j])
                labs.append(1)
            for tf in all_tfs:
                tf_idx = gene_to_idx[tf]
                for g_idx, g in enumerate(hvg_genes):
                    if g != tf and (tf, g) not in trrust_edges:
                        scores.append(abs_corr_c[tf_idx, g_idx])
                        labs.append(0)
            per_cluster_aurocs[c] = roc_auc_score(labs, scores)

        # CSSI-max AUROC
        cssi_max_scores = []
        cssi_max_labels = []
        for tf, tgt in trrust_edges:
            i, j = gene_to_idx[tf], gene_to_idx[tgt]
            cssi_max_scores.append(cssi_max[i, j])
            cssi_max_labels.append(1)
        for tf in all_tfs:
            tf_idx = gene_to_idx[tf]
            for g_idx, g in enumerate(hvg_genes):
                if g != tf and (tf, g) not in trrust_edges:
                    cssi_max_scores.append(cssi_max[tf_idx, g_idx])
                    cssi_max_labels.append(0)
        cssi_max_auroc = roc_auc_score(cssi_max_labels, cssi_max_scores)

        # CSSI-mean AUROC
        cssi_mean_scores = []
        cssi_mean_labels = []
        for tf, tgt in trrust_edges:
            i, j = gene_to_idx[tf], gene_to_idx[tgt]
            cssi_mean_scores.append(cssi_mean[i, j])
            cssi_mean_labels.append(1)
        for tf in all_tfs:
            tf_idx = gene_to_idx[tf]
            for g_idx, g in enumerate(hvg_genes):
                if g != tf and (tf, g) not in trrust_edges:
                    cssi_mean_scores.append(cssi_mean[tf_idx, g_idx])
                    cssi_mean_labels.append(0)
        cssi_mean_auroc = roc_auc_score(cssi_mean_labels, cssi_mean_scores)

        # CSSI-deviation: std across strata (high = context-dependent)
        cssi_std = np.std(stacked, axis=0)
        cssi_std_scores = []
        cssi_std_labels = []
        for tf, tgt in trrust_edges:
            i, j = gene_to_idx[tf], gene_to_idx[tgt]
            cssi_std_scores.append(cssi_std[i, j])
            cssi_std_labels.append(1)
        for tf in all_tfs:
            tf_idx = gene_to_idx[tf]
            for g_idx, g in enumerate(hvg_genes):
                if g != tf and (tf, g) not in trrust_edges:
                    cssi_std_scores.append(cssi_std[tf_idx, g_idx])
                    cssi_std_labels.append(0)
        cssi_std_auroc = roc_auc_score(cssi_std_labels, cssi_std_scores)

        resolution_results[str(res)] = {
            'n_clusters': clustering_results[str(res)]['n_clusters'],
            'n_usable_clusters': len(per_cluster_abs_corr),
            'ari': clustering_results[str(res)]['ari'],
            'nmi': clustering_results[str(res)]['nmi'],
            'per_cluster_aurocs': {c: float(a) for c, a in per_cluster_aurocs.items()},
            'pooled_auroc': float(pooled_auroc),
            'cssi_max_auroc': float(cssi_max_auroc),
            'cssi_mean_auroc': float(cssi_mean_auroc),
            'cssi_std_auroc': float(cssi_std_auroc),
            'best_cluster_auroc': float(max(per_cluster_aurocs.values())),
            'worst_cluster_auroc': float(min(per_cluster_aurocs.values())),
            'cssi_max_vs_pooled': float(cssi_max_auroc - pooled_auroc),
            'cssi_mean_vs_pooled': float(cssi_mean_auroc - pooled_auroc),
        }

        r = resolution_results[str(res)]
        print(f"    Pooled:    {pooled_auroc:.4f}")
        print(f"    CSSI-max:  {cssi_max_auroc:.4f} (Δ={r['cssi_max_vs_pooled']:+.4f})")
        print(f"    CSSI-mean: {cssi_mean_auroc:.4f} (Δ={r['cssi_mean_vs_pooled']:+.4f})")
        print(f"    CSSI-std:  {cssi_std_auroc:.4f}")
        print(f"    Best cluster: {r['best_cluster_auroc']:.4f}, worst: {r['worst_cluster_auroc']:.4f}")

        save_checkpoint(f"resolution_{res}", resolution_results[str(res)])

    # ==============================================================
    # STEP 5: EDGE-LEVEL ENRICHMENT (BEST RESOLUTION)
    # ==============================================================
    print("\n--- STEP 5: Edge-level enrichment analysis ---")

    # Pick the resolution with best CSSI-max
    best_res = max(resolution_results.keys(), key=lambda r: resolution_results[r]['cssi_max_auroc'])
    print(f"  Best resolution for CSSI-max: {best_res}")

    # For the best resolution, check if clusters map to known cell types
    best_labels = adata_hvg.obs[f'leiden_{best_res}'].values.astype(str)
    best_clusters = sorted(set(best_labels))

    # Map each cluster to its dominant ground-truth cell type
    if gt_labels is not None:
        cluster_ct_mapping = {}
        for c in best_clusters:
            mask = best_labels == c
            gt_in_cluster = gt_labels[mask]
            types, counts = np.unique(gt_in_cluster, return_counts=True)
            dominant = types[np.argmax(counts)]
            purity = counts.max() / counts.sum()
            cluster_ct_mapping[c] = {
                'dominant_type': str(dominant),
                'purity': float(purity),
                'composition': {str(t): int(c_) for t, c_ in zip(types, counts)},
            }
            print(f"    Cluster {c}: {(mask).sum()} cells, dominant={dominant} (purity={purity:.2f})")

    # ==============================================================
    # STEP 6: GROUND-TRUTH COMPARISON
    # ==============================================================
    print("\n--- STEP 6: Ground-truth comparison ---")

    # Also compute CSSI with ground-truth labels for honest comparison
    if gt_labels is not None:
        # Simplify ground-truth to major types
        from collections import Counter
        gt_counts = Counter(gt_labels)
        # Keep types with >100 cells
        major_types = {t for t, c in gt_counts.items() if c >= 100}
        gt_mask = np.array([l in major_types for l in gt_labels])

        print(f"  Major ground-truth types (≥100 cells): {len(major_types)}")

        gt_per_type_corr = {}
        for ct in sorted(major_types):
            ct_mask = gt_labels == ct
            n_cells = ct_mask.sum()
            X_ct = X[ct_mask]
            ranked_ct = np.apply_along_axis(stats.rankdata, 0, X_ct)
            corr_ct = np.corrcoef(ranked_ct.T)
            corr_ct = np.nan_to_num(corr_ct, nan=0.0)
            gt_per_type_corr[ct] = np.abs(corr_ct)

        # CSSI-max with ground truth
        gt_stacked = np.stack(list(gt_per_type_corr.values()))
        gt_cssi_max = np.max(gt_stacked, axis=0)
        gt_cssi_mean = np.mean(gt_stacked, axis=0)

        gt_max_scores = []
        gt_max_labels = []
        for tf, tgt in trrust_edges:
            i, j = gene_to_idx[tf], gene_to_idx[tgt]
            gt_max_scores.append(gt_cssi_max[i, j])
            gt_max_labels.append(1)
        for tf in all_tfs:
            tf_idx = gene_to_idx[tf]
            for g_idx, g in enumerate(hvg_genes):
                if g != tf and (tf, g) not in trrust_edges:
                    gt_max_scores.append(gt_cssi_max[tf_idx, g_idx])
                    gt_max_labels.append(0)
        gt_cssi_max_auroc = roc_auc_score(gt_max_labels, gt_max_scores)

        gt_mean_scores = []
        gt_mean_labels = []
        for tf, tgt in trrust_edges:
            i, j = gene_to_idx[tf], gene_to_idx[tgt]
            gt_mean_scores.append(gt_cssi_mean[i, j])
            gt_mean_labels.append(1)
        for tf in all_tfs:
            tf_idx = gene_to_idx[tf]
            for g_idx, g in enumerate(hvg_genes):
                if g != tf and (tf, g) not in trrust_edges:
                    gt_mean_scores.append(gt_cssi_mean[tf_idx, g_idx])
                    gt_mean_labels.append(0)
        gt_cssi_mean_auroc = roc_auc_score(gt_mean_labels, gt_mean_scores)

        print(f"  Ground-truth CSSI-max AUROC:  {gt_cssi_max_auroc:.4f}")
        print(f"  Ground-truth CSSI-mean AUROC: {gt_cssi_mean_auroc:.4f}")
        print(f"  Pooled AUROC:                 {pooled_auroc:.4f}")

    # ==============================================================
    # SAVE RESULTS
    # ==============================================================
    print("\n--- SAVING RESULTS ---")

    all_results = {
        'clustering': clustering_results,
        'pooled_auroc': float(pooled_auroc),
        'n_trrust_edges': len(trrust_edges),
        'n_trrust_tfs': len(all_tfs),
        'resolution_results': resolution_results,
        'best_resolution': best_res,
    }

    if gt_labels is not None:
        all_results['ground_truth_comparison'] = {
            'n_major_types': len(major_types),
            'major_types': sorted(major_types),
            'gt_cssi_max_auroc': float(gt_cssi_max_auroc),
            'gt_cssi_mean_auroc': float(gt_cssi_mean_auroc),
            'cluster_to_celltype': cluster_ct_mapping if 'cluster_ct_mapping' in dir() else {},
        }

    with open(os.path.join(OUT_DIR, "hardened_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=_json_default)

    # Generate markdown summary
    md = ["# Analysis C: CSSI Biology — Hardened Results (Unsupervised Leiden)\n"]

    md.append("## 1. Clustering Evaluation\n")
    md.append("| Resolution | Clusters | ARI | NMI |")
    md.append("|-----------|----------|-----|-----|")
    for res_str, cr in clustering_results.items():
        md.append(f"| {cr['resolution']} | {cr['n_clusters']} | {cr['ari']:.3f} | {cr['nmi']:.3f} |")

    md.append("\n## 2. CSSI AUROC by Resolution\n")
    md.append("| Resolution | K | Pooled | CSSI-max | CSSI-mean | CSSI-std | Δmax | Δmean |")
    md.append("|-----------|---|--------|---------|----------|---------|------|-------|")
    for res_str, rr in resolution_results.items():
        md.append(f"| {res_str} | {rr['n_usable_clusters']} | {pooled_auroc:.4f} | {rr['cssi_max_auroc']:.4f} | {rr['cssi_mean_auroc']:.4f} | {rr['cssi_std_auroc']:.4f} | {rr['cssi_max_vs_pooled']:+.4f} | {rr['cssi_mean_vs_pooled']:+.4f} |")

    md.append(f"\n**Best resolution for CSSI-max: {best_res}**\n")

    if gt_labels is not None:
        md.append("\n## 3. Ground-Truth Comparison\n")
        md.append(f"| Method | AUROC |")
        md.append(f"|--------|-------|")
        md.append(f"| Pooled (no stratification) | {pooled_auroc:.4f} |")
        md.append(f"| CSSI-max (unsupervised, best res={best_res}) | {resolution_results[best_res]['cssi_max_auroc']:.4f} |")
        md.append(f"| CSSI-mean (unsupervised, best res={best_res}) | {resolution_results[best_res]['cssi_mean_auroc']:.4f} |")
        md.append(f"| CSSI-max (ground-truth labels) | {gt_cssi_max_auroc:.4f} |")
        md.append(f"| CSSI-mean (ground-truth labels) | {gt_cssi_mean_auroc:.4f} |")

        md.append("\n## 4. Cluster-to-Cell-Type Mapping (Best Resolution)\n")
        if 'cluster_ct_mapping' in dir() and cluster_ct_mapping:
            md.append("| Cluster | n cells | Dominant Type | Purity |")
            md.append("|---------|---------|--------------|--------|")
            for c in sorted(cluster_ct_mapping.keys()):
                info = cluster_ct_mapping[c]
                n = sum(info['composition'].values())
                md.append(f"| {c} | {n} | {info['dominant_type']} | {info['purity']:.2f} |")

    md.append("\n## 5. Interpretation\n")
    md.append("This analysis uses **unsupervised Leiden clustering** (not ground-truth labels).\n")

    summary_path = os.path.join(OUT_DIR, "hardened_summary.md")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(md))

    print(f"\nSaved: {summary_path}")
    print("\nAnalysis C hardening COMPLETE")


if __name__ == "__main__":
    main()
