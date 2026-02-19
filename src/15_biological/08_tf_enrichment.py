#!/usr/bin/env python3
"""
Follow-Up 3: Cell-Type-Specific TF Enrichment (Analysis C)

For each unsupervised Leiden cluster:
1. Identify the top-K TF-target edges by stratum-specific correlation
2. Test if the top edges are enriched for TFs known to be active in the
   cluster's dominant cell type (from CellMarker/literature annotations)
3. Use Fisher's exact test and edge-level Mann-Whitney

If non-significant, recommend dropping Analysis C from the paper.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "cssi_biology")

TRRUST_PATH = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv"
TS_PATH = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad"

# Known cell-type-specific TFs (from literature/CellMarker)
# Maps cell type keywords to TFs known to regulate that lineage
CELLTYPE_TFS = {
    'T cell': ['BCL11B', 'LEF1', 'TCF7', 'GATA3', 'TBX21', 'FOXP3', 'RUNX3', 'EOMES', 'RORC', 'STAT4', 'STAT5A', 'STAT5B'],
    'B cell': ['EBF1', 'PAX5', 'BCL6', 'IRF4', 'PRDM1', 'TCF3', 'POU2F2', 'FOXO1', 'BACH2', 'SPI1'],
    'monocyte': ['SPI1', 'CEBPA', 'CEBPB', 'IRF8', 'KLF4', 'MAFB', 'MAF', 'NR1H3'],
    'macrophage': ['SPI1', 'CEBPA', 'CEBPB', 'IRF8', 'MAFB', 'MAF', 'NR1H3', 'PPARG'],
    'neutrophil': ['SPI1', 'CEBPE', 'CEBPA', 'GFI1', 'HOXA9', 'HOXA10'],
    'erythrocyte': ['GATA1', 'KLF1', 'TAL1', 'NFE2', 'EPOR'],
    'NK cell': ['TBX21', 'EOMES', 'ID2', 'NFIL3', 'ETS1', 'RUNX3'],
    'plasma cell': ['IRF4', 'PRDM1', 'XBP1', 'BCL6'],
    'mast cell': ['GATA2', 'MITF', 'SPI1'],
    'dendritic': ['IRF8', 'BATF3', 'ID2', 'IRF4', 'SPI1', 'ZBTB46'],
}

def match_celltype(dominant_type):
    """Map Tabula Sapiens cell type to our TF annotation keys."""
    dt = dominant_type.lower()
    for key in CELLTYPE_TFS:
        if key.lower() in dt:
            return key
    # Special cases
    if 'cd4' in dt or 'cd8' in dt or 't cell' in dt:
        return 'T cell'
    if 'classical monocyte' in dt or 'non-classical monocyte' in dt:
        return 'monocyte'
    return None


def main():
    import scanpy as sc

    print("=" * 70)
    print("FOLLOW-UP 3: CELL-TYPE-SPECIFIC TF ENRICHMENT")
    print("=" * 70)

    # Load hardened results to get cluster mapping
    with open(os.path.join(OUT_DIR, "hardened_results.json")) as f:
        hardened = json.load(f)

    best_res = hardened['best_resolution']
    print(f"Using best resolution: {best_res}")

    # Load Tabula Sapiens data
    print("\nLoading Tabula Sapiens...")
    adata = sc.read_h5ad(TS_PATH)
    if 'feature_name' in adata.var.columns:
        adata.var_names = adata.var['feature_name'].astype(str).values
        adata.var_names_make_unique()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')
    adata_hvg = adata[:, adata.var['highly_variable']].copy()
    hvg_genes = list(adata_hvg.var_names)
    gene_to_idx = {g: i for i, g in enumerate(hvg_genes)}

    # PCA + neighbors + Leiden at best resolution
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=50)
    sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=50)
    sc.tl.leiden(adata_hvg, resolution=float(best_res), key_added='leiden_best')

    labels = adata_hvg.obs['leiden_best'].values.astype(str)
    clusters = sorted(set(labels))

    # Ground-truth labels
    gt_col = 'cell_type' if 'cell_type' in adata.obs.columns else 'cell_ontology_class'
    gt_labels = adata.obs[gt_col].astype(str).values

    # Load TRRUST
    trrust = pd.read_csv(TRRUST_PATH, sep='\t', header=None, names=['tf', 'target', 'type', 'pmid'])
    trrust_hvg = trrust[(trrust['tf'].isin(hvg_genes)) & (trrust['target'].isin(hvg_genes))]
    trrust_edges = set(zip(trrust_hvg['tf'], trrust_hvg['target']))
    trrust_tfs_in_hvg = set(trrust_hvg['tf'])
    print(f"TRRUST edges in HVG: {len(trrust_edges)}, TFs: {len(trrust_tfs_in_hvg)}")

    # Get raw expression for correlation
    adata_raw = adata[:, adata.var['highly_variable']].copy()
    X = adata_raw.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)

    # Per-cluster analysis
    print("\n" + "=" * 70)
    print("PER-CLUSTER TF ENRICHMENT")
    print("=" * 70)

    enrichment_results = []

    for c in clusters:
        mask = labels == c
        n_cells = mask.sum()
        if n_cells < 50:
            continue

        # Dominant cell type
        gt_in_cluster = gt_labels[mask]
        types, counts = np.unique(gt_in_cluster, return_counts=True)
        dominant = types[np.argmax(counts)]
        purity = counts.max() / counts.sum()

        # Match to known TFs
        ct_key = match_celltype(dominant)
        if ct_key is None:
            print(f"\n  Cluster {c} ({n_cells} cells, {dominant}): no TF annotation, skipping")
            continue

        expected_tfs = set(CELLTYPE_TFS[ct_key]) & trrust_tfs_in_hvg
        if not expected_tfs:
            print(f"\n  Cluster {c} ({n_cells} cells, {dominant} → {ct_key}): no expected TFs in HVG TRRUST")
            continue

        print(f"\n  Cluster {c} ({n_cells} cells, {dominant} → {ct_key}, purity={purity:.2f})")
        print(f"    Expected TFs in HVG TRRUST: {sorted(expected_tfs)}")

        # Compute stratum correlation
        X_c = X[mask]
        ranked_c = np.apply_along_axis(stats.rankdata, 0, X_c)
        corr_c = np.corrcoef(ranked_c.T)
        corr_c = np.nan_to_num(corr_c, nan=0.0)
        abs_corr_c = np.abs(corr_c)

        # Get all TRRUST edge scores in this stratum
        matching_scores = []
        nonmatching_scores = []
        matching_edges = 0
        nonmatching_edges = 0

        for tf, tgt in trrust_edges:
            i, j = gene_to_idx[tf], gene_to_idx[tgt]
            score = abs_corr_c[i, j]
            if tf in expected_tfs:
                matching_scores.append(score)
                matching_edges += 1
            else:
                nonmatching_scores.append(score)
                nonmatching_edges += 1

        matching_scores = np.array(matching_scores)
        nonmatching_scores = np.array(nonmatching_scores)

        if len(matching_scores) < 2 or len(nonmatching_scores) < 2:
            print(f"    Too few edges: matching={len(matching_scores)}, non-matching={len(nonmatching_scores)}")
            continue

        # Mann-Whitney: are matching TF edges ranked higher?
        u_stat, mw_p = stats.mannwhitneyu(matching_scores, nonmatching_scores, alternative='greater')
        # Effect size: rank-biserial correlation
        n1, n2 = len(matching_scores), len(nonmatching_scores)
        r_rb = 1 - (2 * u_stat) / (n1 * n2)

        # Fisher's exact: are matching TFs enriched in top-100 edges?
        all_edge_scores = []
        all_edge_tfs = []
        for tf, tgt in trrust_edges:
            i, j = gene_to_idx[tf], gene_to_idx[tgt]
            all_edge_scores.append(abs_corr_c[i, j])
            all_edge_tfs.append(tf)
        rank_order = np.argsort(all_edge_scores)[::-1]
        top_100_tfs = set(np.array(all_edge_tfs)[rank_order[:100]])

        # Contingency table
        a = len(expected_tfs & top_100_tfs)
        b = len(top_100_tfs - expected_tfs)
        c_val = len(expected_tfs - top_100_tfs)
        d = len(trrust_tfs_in_hvg - expected_tfs - top_100_tfs)
        odds_ratio, fisher_p = stats.fisher_exact([[a, b], [c_val, d]], alternative='greater')

        result = {
            'cluster': c,
            'n_cells': int(n_cells),
            'dominant_type': dominant,
            'ct_key': ct_key,
            'purity': float(purity),
            'expected_tfs': sorted(expected_tfs),
            'n_matching_edges': matching_edges,
            'n_nonmatching_edges': nonmatching_edges,
            'matching_mean_score': float(matching_scores.mean()),
            'nonmatching_mean_score': float(nonmatching_scores.mean()),
            'mannwhitney_p': float(mw_p),
            'rank_biserial_r': float(r_rb),
            'fisher_or': float(odds_ratio) if not np.isinf(odds_ratio) else 'inf',
            'fisher_p': float(fisher_p),
            'matching_tfs_in_top100': sorted(expected_tfs & top_100_tfs),
        }
        enrichment_results.append(result)

        sig = "**" if mw_p < 0.05 else ""
        print(f"    Matching edges: {matching_edges} (mean={matching_scores.mean():.4f})")
        print(f"    Non-matching edges: {nonmatching_edges} (mean={nonmatching_scores.mean():.4f})")
        print(f"    Mann-Whitney (one-sided): p={mw_p:.4g} {sig}, r_rb={r_rb:.3f}")
        print(f"    Fisher top-100: OR={result['fisher_or']}, p={fisher_p:.4f}")
        print(f"    Matching TFs in top-100: {sorted(expected_tfs & top_100_tfs)}")

    # BH correction on Mann-Whitney p-values
    print("\n" + "=" * 70)
    print("MULTIPLE TESTING CORRECTION")
    print("=" * 70)

    from statsmodels.stats.multitest import multipletests
    mw_pvals = [r['mannwhitney_p'] for r in enrichment_results]
    if mw_pvals:
        reject, bh_q, _, _ = multipletests(mw_pvals, alpha=0.05, method='fdr_bh')
        for i, r in enumerate(enrichment_results):
            r['mannwhitney_bh_q'] = float(bh_q[i])
            r['mannwhitney_significant_bh'] = bool(reject[i])
            sig = "**" if reject[i] else ""
            print(f"  Cluster {r['cluster']} ({r['ct_key']}): raw p={r['mannwhitney_p']:.4g}, BH q={bh_q[i]:.4g} {sig}")

        n_sig = sum(reject)
        n_total = len(reject)
        print(f"\n  {n_sig}/{n_total} clusters significant after BH correction")

    # Fisher p BH correction
    fisher_pvals = [r['fisher_p'] for r in enrichment_results]
    if fisher_pvals:
        reject_f, bh_q_f, _, _ = multipletests(fisher_pvals, alpha=0.05, method='fdr_bh')
        for i, r in enumerate(enrichment_results):
            r['fisher_bh_q'] = float(bh_q_f[i])
            r['fisher_significant_bh'] = bool(reject_f[i])

        n_sig_f = sum(reject_f)
        print(f"  Fisher enrichment: {n_sig_f}/{len(reject_f)} significant after BH")

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    def _default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.bool_): return bool(o)
        raise TypeError

    output = {
        'best_resolution': best_res,
        'n_clusters_tested': len(enrichment_results),
        'n_mannwhitney_significant_bh': int(sum(r.get('mannwhitney_significant_bh', False) for r in enrichment_results)),
        'n_fisher_significant_bh': int(sum(r.get('fisher_significant_bh', False) for r in enrichment_results)),
        'results': enrichment_results,
    }

    out_path = os.path.join(OUT_DIR, "tf_enrichment_results.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_default)
    print(f"  Saved: {out_path}")

    # Summary markdown
    md = ["# Follow-Up 3: Cell-Type-Specific TF Enrichment\n"]
    md.append(f"Resolution: {best_res}, {len(enrichment_results)} testable clusters\n")
    md.append("## Edge-Level Mann-Whitney (Matching vs Non-Matching TF Edges)\n")
    md.append("| Cluster | Cell Type | Expected TFs | Match Mean | Non-match Mean | MW p | BH q | Sig |")
    md.append("|---------|-----------|-------------|-----------|---------------|------|------|-----|")
    for r in enrichment_results:
        sig = "Yes" if r.get('mannwhitney_significant_bh') else "No"
        md.append(f"| {r['cluster']} | {r['ct_key']} | {', '.join(r['expected_tfs'])} | {r['matching_mean_score']:.4f} | {r['nonmatching_mean_score']:.4f} | {r['mannwhitney_p']:.4g} | {r.get('mannwhitney_bh_q', 'N/A'):.4g} | {sig} |")

    md.append(f"\n**{output['n_mannwhitney_significant_bh']}/{output['n_clusters_tested']} significant after BH.**")
    md.append(f"\n**{output['n_fisher_significant_bh']}/{output['n_clusters_tested']} Fisher tests significant after BH.**")

    md.append("\n\n## Recommendation\n")
    if output['n_mannwhitney_significant_bh'] < output['n_clusters_tested'] // 2:
        md.append("The cell-type-specific TF enrichment signal is weak. Most clusters do not show")
        md.append("significant enrichment of expected TF edges after multiple testing correction.")
        md.append("**Recommend dropping Analysis C from the paper.** The CSSI contribution is already")
        md.append("supported by existing synthetic/TRRUST validation in Tier 1.\n")
    else:
        md.append("The enrichment signal is present in a majority of clusters. Analysis C could be")
        md.append("included in supplementary materials with appropriate caveats about effect sizes.\n")

    md_path = os.path.join(OUT_DIR, "tf_enrichment_summary.md")
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))
    print(f"  Saved: {md_path}")

    print("\nFollow-Up 3 COMPLETE")


if __name__ == "__main__":
    main()
