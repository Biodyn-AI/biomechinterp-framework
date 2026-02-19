"""
Phase 2, Script 1: Multi-tissue ortholog transfer analysis.

Extends Phase 1 lung-to-lung analysis to:
- Immune tissue (Tabula Sapiens immune)
- Kidney tissue (Tabula Sapiens kidney)

For each human tissue, we compute TF-target edges and compare them to mouse lung
(our only mouse reference). This tests whether the lung-specific conservation
pattern holds for other tissues projected through the same ortholog framework.

We also compute cross-tissue comparisons within human (lung vs immune, lung vs kidney,
immune vs kidney) to set up the cross-tissue vs cross-species analysis.
"""
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr, rankdata
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_30_ortholog_transfer_stress_test/implementation"
PYTHON = "/Users/ihorkendiukhov/biodyn-work/.venv/bin/python3"

# Reuse the TF list from Phase 1
TF_LIST = [
    "NKX2-1", "FOXJ1", "SOX2", "SOX9", "TP63", "HOPX", "FOXA1", "FOXA2",
    "ETV5", "GATA6", "CEBPA", "CEBPB", "CEBPD",
    "SPI1", "IRF1", "IRF4", "IRF8", "STAT1", "STAT3", "STAT6",
    "NFKB1", "RELA", "BATF", "TBX21", "GATA3", "RORC", "BCL6",
    "FOXP3", "RUNX1", "RUNX3", "ETS1", "ETS2",
    "ERG", "FLI1", "SOX17", "SOX18", "KLF2", "KLF4",
    "TWIST1", "SNAI1", "SNAI2", "TCF21", "TBX4",
    "MYC", "MAX", "JUN", "FOS", "JUNB", "JUND", "FOSB",
    "ATF3", "ATF4", "XBP1", "HIF1A", "EPAS1",
    "E2F1", "E2F4", "RB1", "TP53",
    "EGR1", "EGR2", "KLF6", "SP1", "SP3",
    "NFE2L2", "BACH1", "MAFK",
    "REST", "YY1", "CTCF",
    "HES1", "HEY1", "NOTCH1",
    "SMAD2", "SMAD3", "SMAD4",
    "LEF1", "TCF7", "CTNNB1",
    "EZH2", "SUZ12",
    # Additional immune/kidney TFs
    "PAX2", "PAX8", "HNF1A", "HNF1B", "HNF4A",  # kidney
    "IKZF1", "TCF3", "EBF1", "PAX5",  # B cell
    "GATA1", "GATA2",  # hematopoietic
]


def preprocess_tissue(adata, tissue_name, min_cells_pct=0.05, max_cells=15000):
    """Preprocess a tissue dataset: filter genes, map names, subsample."""
    print(f"\n  Processing {tissue_name}...")
    print(f"    Input: {adata.shape}")

    adata = adata.copy()

    # Densify if sparse
    if sp.issparse(adata.X):
        xmax = adata.X.max()
    else:
        xmax = adata.X.max()

    if xmax > 50:
        print(f"    Data appears to be counts (max={xmax:.1f}), normalizing...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print(f"    Data appears log-normalized (max={xmax:.2f})")

    # Subsample cells if too many (before gene filtering for speed)
    if adata.shape[0] > max_cells:
        np.random.seed(42)
        idx = np.random.choice(adata.shape[0], max_cells, replace=False)
        adata = adata[idx].copy()
        print(f"    Subsampled to {max_cells} cells")

    # Filter genes
    min_cells = int(adata.shape[0] * min_cells_pct)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"    After gene filter (>={min_cells} cells): {adata.shape}")

    # Map var_names to feature_name
    if 'feature_name' in adata.var.columns:
        adata.var['original_id'] = adata.var_names
        name_counts = adata.var['feature_name'].value_counts()
        dup_names = name_counts[name_counts > 1].index
        new_names = []
        for idx_name, row in adata.var.iterrows():
            name = row['feature_name']
            if name in dup_names or pd.isna(name) or name == '':
                new_names.append(idx_name)
            else:
                new_names.append(name)
        adata.var_names = new_names
        adata.var_names_make_unique()

    print(f"    Gene name sample: {list(adata.var_names[:10])}")
    return adata


def compute_edges(adata, tf_list, tissue_name, n_targets=3000):
    """Compute TF-target Spearman correlations. Returns edge DataFrame."""
    print(f"\n  Computing edges for {tissue_name}...")

    available_tfs = [tf for tf in tf_list if tf in adata.var_names]
    print(f"    TFs found: {len(available_tfs)} / {len(tf_list)}")

    # HVGs
    sc.pp.highly_variable_genes(adata, n_top_genes=min(n_targets, adata.shape[1]),
                                 flavor='cell_ranger')
    hvg = adata.var_names[adata.var['highly_variable']].tolist()
    target_genes = list(set(hvg))
    print(f"    HVGs: {len(hvg)}, Target genes: {len(target_genes)}")

    tf_idx = [list(adata.var_names).index(tf) for tf in available_tfs]
    target_idx = [list(adata.var_names).index(g) for g in target_genes]

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    tf_expr = X[:, tf_idx]
    target_expr = X[:, target_idx]

    # Subsample for correlation computation
    max_cells = 10000
    if tf_expr.shape[0] > max_cells:
        np.random.seed(42)
        idx = np.random.choice(tf_expr.shape[0], max_cells, replace=False)
        tf_expr = tf_expr[idx]
        target_expr = target_expr[idx]
        print(f"    Subsampled to {max_cells} cells for correlation")

    # Rank-transform for Spearman
    t0 = time.time()
    tf_ranks = np.apply_along_axis(rankdata, 0, tf_expr)
    target_ranks = np.apply_along_axis(rankdata, 0, target_expr)

    tf_centered = tf_ranks - tf_ranks.mean(axis=0)
    target_centered = target_ranks - target_ranks.mean(axis=0)

    tf_norm = np.sqrt((tf_centered ** 2).sum(axis=0))
    target_norm = np.sqrt((target_centered ** 2).sum(axis=0))
    tf_norm[tf_norm == 0] = 1
    target_norm[target_norm == 0] = 1

    tf_normalized = tf_centered / tf_norm
    target_normalized = target_centered / target_norm

    corr_matrix = tf_normalized.T @ target_normalized
    elapsed = time.time() - t0
    print(f"    Correlations computed in {elapsed:.1f}s")

    edges = []
    for i, tf in enumerate(available_tfs):
        for j, target in enumerate(target_genes):
            if tf == target:
                continue
            rho = corr_matrix[i, j]
            if abs(rho) > 0.05:
                edges.append({
                    'tf': tf,
                    'target': target,
                    'rho': rho,
                    'abs_rho': abs(rho),
                })

    edges_df = pd.DataFrame(edges)
    edges_df = edges_df.sort_values('abs_rho', ascending=False).reset_index(drop=True)
    print(f"    Edges with |rho| > 0.05: {len(edges_df)}")
    return edges_df, available_tfs


def compare_edge_sets(edges_a, edges_b, name_a, name_b, shared_tfs):
    """Compare two edge sets. Returns comparison statistics."""
    # Filter to shared TFs
    a = edges_a[edges_a['tf'].isin(shared_tfs)].copy()
    b = edges_b[edges_b['tf'].isin(shared_tfs)].copy()

    a['edge_key'] = a['tf'] + '|' + a['target']
    b['edge_key'] = b['tf'] + '|' + b['target']

    shared_keys = set(a['edge_key']) & set(b['edge_key'])

    if len(shared_keys) < 100:
        print(f"  WARNING: Only {len(shared_keys)} shared edges between {name_a} and {name_b}")
        return None

    a_sub = a[a['edge_key'].isin(shared_keys)].drop_duplicates('edge_key').set_index('edge_key')
    b_sub = b[b['edge_key'].isin(shared_keys)].drop_duplicates('edge_key').set_index('edge_key')

    common = a_sub.index.intersection(b_sub.index)
    rho_spearman, p_spearman = spearmanr(a_sub.loc[common, 'rho'], b_sub.loc[common, 'rho'])

    sign_agree = (np.sign(a_sub.loc[common, 'rho'].values) == np.sign(b_sub.loc[common, 'rho'].values)).mean()

    # Top-k overlap
    topk_results = {}
    for k in [100, 500, 1000]:
        if k > len(a) or k > len(b):
            continue
        a_top = set(a.nlargest(k, 'abs_rho')['edge_key'])
        b_top = set(b.nlargest(k, 'abs_rho')['edge_key'])
        overlap = len(a_top & b_top)
        n_universe = len(set(a['edge_key']) | set(b['edge_key']))
        expected = k * k / n_universe if n_universe > 0 else 0
        fold = overlap / expected if expected > 0 else float('inf')
        topk_results[k] = {'overlap': overlap, 'expected': expected, 'fold': fold}

    # Per-TF conservation
    per_tf = []
    for tf in shared_tfs:
        tf_a = a_sub.loc[a_sub['tf'] == tf] if 'tf' in a_sub.columns else a_sub[a_sub.index.str.startswith(tf + '|')]
        tf_b = b_sub.loc[b_sub['tf'] == tf] if 'tf' in b_sub.columns else b_sub[b_sub.index.str.startswith(tf + '|')]
        # Need to use the merged approach
        pass

    # Build merged for per-TF
    merged = a_sub[['tf', 'rho']].join(b_sub[['rho']], lsuffix='_a', rsuffix='_b')
    merged = merged.dropna()

    per_tf_stats = []
    for tf in shared_tfs:
        tf_merged = merged[merged['tf'] == tf]
        if len(tf_merged) < 10:
            continue
        rho_tf, _ = spearmanr(tf_merged['rho_a'], tf_merged['rho_b'])
        per_tf_stats.append({'tf': tf, 'spearman_rho': rho_tf, 'n_edges': len(tf_merged)})

    result = {
        'comparison': f'{name_a}_vs_{name_b}',
        'n_shared_edges': len(common),
        'spearman_rho': float(rho_spearman),
        'p_value': float(p_spearman),
        'sign_agreement': float(sign_agree),
        'topk': topk_results,
        'per_tf': per_tf_stats,
    }

    return result


# ============================================================
# MAIN ANALYSIS
# ============================================================
print("=" * 60)
print("PHASE 2: Multi-tissue ortholog transfer analysis")
print("=" * 60)

# Load and process all datasets
datasets = {}

# Human lung (reuse from Phase 1 if available, otherwise recompute)
human_lung_edges_path = os.path.join(OUT_DIR, "human_edges.csv")
mouse_lung_edges_path = os.path.join(OUT_DIR, "mouse_edges.csv")

if os.path.exists(human_lung_edges_path):
    print("\nLoading Phase 1 lung edges...")
    datasets['human_lung'] = pd.read_csv(human_lung_edges_path)
    datasets['mouse_lung'] = pd.read_csv(mouse_lung_edges_path)
    with open(os.path.join(OUT_DIR, "tf_lists.json")) as f:
        tf_info = json.load(f)
    lung_tfs = tf_info['shared_tfs']
    print(f"  Human lung edges: {len(datasets['human_lung'])}")
    print(f"  Mouse lung edges: {len(datasets['mouse_lung'])}")
else:
    print("ERROR: Phase 1 results not found. Run scripts 01-04 first.")
    exit(1)

# Human kidney
print("\n\nLoading and processing human kidney...")
kidney = sc.read_h5ad("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad")
kidney_proc = preprocess_tissue(kidney, "Human Kidney", max_cells=11000)
kidney_edges, kidney_tfs = compute_edges(kidney_proc, TF_LIST, "Human Kidney")
datasets['human_kidney'] = kidney_edges
kidney_edges.to_csv(os.path.join(OUT_DIR, "phase2_human_kidney_edges.csv"), index=False)
del kidney, kidney_proc

# Human immune (subsample heavily - it's 592K cells)
print("\n\nLoading and processing human immune...")
immune = sc.read_h5ad("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad")
immune_proc = preprocess_tissue(immune, "Human Immune", max_cells=15000)
immune_edges, immune_tfs = compute_edges(immune_proc, TF_LIST, "Human Immune")
datasets['human_immune'] = immune_edges
immune_edges.to_csv(os.path.join(OUT_DIR, "phase2_human_immune_edges.csv"), index=False)
del immune, immune_proc

# ============================================================
# Cross-species comparisons (each human tissue vs mouse lung)
# ============================================================
print("\n" + "=" * 60)
print("Cross-species comparisons (human tissues vs mouse lung)")
print("=" * 60)

comparisons = {}

for tissue_name, tissue_edges, tissue_tfs in [
    ('human_lung', datasets['human_lung'], lung_tfs),
    ('human_kidney', datasets['human_kidney'], kidney_tfs),
    ('human_immune', datasets['human_immune'], immune_tfs),
]:
    mouse_tfs = set(tf_info['mouse_tfs'])
    shared = sorted(set(tissue_tfs if isinstance(tissue_tfs, list) else list(tissue_tfs)) & mouse_tfs)
    print(f"\n  {tissue_name} vs mouse_lung: {len(shared)} shared TFs")

    result = compare_edge_sets(tissue_edges, datasets['mouse_lung'],
                                tissue_name, 'mouse_lung', shared)
    if result:
        comparisons[f'{tissue_name}_vs_mouse_lung'] = result
        print(f"    Spearman rho: {result['spearman_rho']:.4f}")
        print(f"    Sign agreement: {result['sign_agreement']:.4f}")
        print(f"    Shared edges: {result['n_shared_edges']}")
        for k, v in result['topk'].items():
            print(f"    Top-{k}: overlap={v['overlap']}, fold={v['fold']:.1f}x")

# ============================================================
# Cross-tissue comparisons (within human)
# ============================================================
print("\n" + "=" * 60)
print("Cross-tissue comparisons (within human)")
print("=" * 60)

human_tissues = [
    ('human_lung', datasets['human_lung'], lung_tfs if isinstance(lung_tfs, list) else list(lung_tfs)),
    ('human_kidney', datasets['human_kidney'], kidney_tfs),
    ('human_immune', datasets['human_immune'], immune_tfs),
]

for i in range(len(human_tissues)):
    for j in range(i+1, len(human_tissues)):
        name_a, edges_a, tfs_a = human_tissues[i]
        name_b, edges_b, tfs_b = human_tissues[j]
        shared = sorted(set(tfs_a) & set(tfs_b))
        print(f"\n  {name_a} vs {name_b}: {len(shared)} shared TFs")

        result = compare_edge_sets(edges_a, edges_b, name_a, name_b, shared)
        if result:
            comparisons[f'{name_a}_vs_{name_b}'] = result
            print(f"    Spearman rho: {result['spearman_rho']:.4f}")
            print(f"    Sign agreement: {result['sign_agreement']:.4f}")
            print(f"    Shared edges: {result['n_shared_edges']}")

# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY: All pairwise comparisons")
print("=" * 60)

summary_rows = []
for name, result in comparisons.items():
    topk_100 = result['topk'].get(100, {})
    summary_rows.append({
        'comparison': name,
        'n_shared_edges': result['n_shared_edges'],
        'spearman_rho': result['spearman_rho'],
        'sign_agreement': result['sign_agreement'],
        'topk_100_fold': topk_100.get('fold', None),
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string())
summary_df.to_csv(os.path.join(OUT_DIR, "phase2_multi_tissue_summary.csv"), index=False)

# Save full comparison results
with open(os.path.join(OUT_DIR, "phase2_multi_tissue_results.json"), "w") as f:
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_clean = {}
    for k, v in comparisons.items():
        results_clean[k] = {}
        for kk, vv in v.items():
            if isinstance(vv, dict):
                results_clean[k][kk] = {str(kkk): convert(vvv) if not isinstance(vvv, dict) else {str(kkkk): convert(vvvv) for kkkk, vvvv in vvv.items()} for kkk, vvv in vv.items()}
            elif isinstance(vv, list):
                results_clean[k][kk] = [convert(item) if not isinstance(item, dict) else {str(kkk): convert(vvv) for kkk, vvv in item.items()} for item in vv]
            else:
                results_clean[k][kk] = convert(vv)

    json.dump(results_clean, f, indent=2, default=str)

# Save per-TF data for each comparison
for name, result in comparisons.items():
    if result.get('per_tf'):
        per_tf_df = pd.DataFrame(result['per_tf'])
        per_tf_df.to_csv(os.path.join(OUT_DIR, f"phase2_per_tf_{name}.csv"), index=False)

print("\nPhase 2 multi-tissue analysis complete.")
