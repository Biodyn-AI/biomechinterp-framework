"""
Step 3: Compute TF-target correlation-based edge scores in both species.

We:
1. Identify transcription factors from a curated list
2. Filter to highly variable genes
3. Compute Spearman correlations between TFs and all target genes
4. Save edge score matrices for both species
"""
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
import os
import json
import time

OUT_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_30_ortholog_transfer_stress_test/implementation"

print("=" * 60)
print("STEP 3: Computing TF-target edge scores")
print("=" * 60)

# Known human transcription factors (curated set of lung-relevant TFs + major TFs)
# These are well-characterized TFs with known regulatory roles
TF_LIST = [
    # Core lung TFs
    "NKX2-1", "FOXJ1", "SOX2", "SOX9", "TP63", "HOPX", "FOXA1", "FOXA2",
    "ETV5", "GATA6", "CEBPA", "CEBPB", "CEBPD",
    # Immune TFs
    "SPI1", "IRF1", "IRF4", "IRF8", "STAT1", "STAT3", "STAT6",
    "NFKB1", "RELA", "BATF", "TBX21", "GATA3", "RORC", "BCL6",
    "FOXP3", "RUNX1", "RUNX3", "ETS1", "ETS2",
    # Endothelial TFs
    "ERG", "FLI1", "SOX17", "SOX18", "KLF2", "KLF4",
    # Fibroblast/mesenchymal TFs
    "TWIST1", "SNAI1", "SNAI2", "TCF21", "TBX4",
    # General TFs
    "MYC", "MAX", "JUN", "FOS", "JUNB", "JUND", "FOSB",
    "ATF3", "ATF4", "XBP1", "HIF1A", "EPAS1",
    "E2F1", "E2F4", "RB1", "TP53",
    "EGR1", "EGR2", "KLF6", "SP1", "SP3",
    "NFE2L2", "BACH1", "MAFK",
    "REST", "YY1", "CTCF",
    "HES1", "HEY1", "NOTCH1",
    "SMAD2", "SMAD3", "SMAD4",
    "LEF1", "TCF7", "CTNNB1",
    # Chromatin / epigenetic
    "EZH2", "SUZ12",
]

def preprocess_and_filter(adata, species_name, min_cells_pct=0.05):
    """Preprocess: filter genes, identify HVGs, normalize if needed."""
    print(f"\n  Processing {species_name}...")
    print(f"    Input: {adata.shape}")

    # Work with a copy
    adata = adata.copy()

    # Check if data looks log-normalized already (max < 20 suggests log-normalized)
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

    # Filter genes: require expression in at least min_cells_pct of cells
    min_cells = int(adata.shape[0] * min_cells_pct)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"    After gene filter (>={min_cells} cells): {adata.shape}")

    # Map var_names to feature_name for easier interpretation
    if 'feature_name' in adata.var.columns:
        # Create name mapping
        adata.var['original_id'] = adata.var_names
        # Handle duplicate feature names
        name_counts = adata.var['feature_name'].value_counts()
        dup_names = name_counts[name_counts > 1].index
        new_names = []
        for idx, row in adata.var.iterrows():
            name = row['feature_name']
            if name in dup_names or pd.isna(name) or name == '':
                new_names.append(idx)  # keep ensembl ID
            else:
                new_names.append(name)
        adata.var_names = new_names
        adata.var_names_make_unique()

    print(f"    Gene name sample: {list(adata.var_names[:10])}")

    return adata


def compute_tf_target_correlations(adata, tf_list, species_name, n_targets=3000):
    """
    Compute Spearman correlations between TFs and top variable target genes.
    Returns a DataFrame of edge scores.
    """
    print(f"\n  Computing correlations for {species_name}...")

    # Find which TFs are present
    available_tfs = [tf for tf in tf_list if tf in adata.var_names]
    print(f"    TFs found: {len(available_tfs)} / {len(tf_list)}")
    print(f"    Missing TFs: {[tf for tf in tf_list if tf not in adata.var_names][:10]}...")

    # Select top variable genes as targets
    sc.pp.highly_variable_genes(adata, n_top_genes=min(n_targets, adata.shape[1]),
                                 flavor='seurat_v3' if adata.X.max() > 50 else 'cell_ranger')
    hvg = adata.var_names[adata.var['highly_variable']].tolist()
    print(f"    HVGs selected: {len(hvg)}")

    # Target genes = HVGs (including TFs that are HVGs)
    target_genes = list(set(hvg))
    print(f"    Target genes: {len(target_genes)}")

    # Get expression matrices
    tf_idx = [list(adata.var_names).index(tf) for tf in available_tfs]
    target_idx = [list(adata.var_names).index(g) for g in target_genes]

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    tf_expr = X[:, tf_idx]  # cells x TFs
    target_expr = X[:, target_idx]  # cells x targets

    print(f"    TF expr shape: {tf_expr.shape}")
    print(f"    Target expr shape: {target_expr.shape}")

    # Subsample cells if too many (for speed)
    max_cells = 10000
    if tf_expr.shape[0] > max_cells:
        np.random.seed(42)
        idx = np.random.choice(tf_expr.shape[0], max_cells, replace=False)
        tf_expr = tf_expr[idx]
        target_expr = target_expr[idx]
        print(f"    Subsampled to {max_cells} cells")

    # Compute Spearman correlations: TFs x targets
    print(f"    Computing Spearman correlations ({len(available_tfs)} TFs x {len(target_genes)} targets)...")
    t0 = time.time()

    # Rank-transform for Spearman
    from scipy.stats import rankdata
    tf_ranks = np.apply_along_axis(rankdata, 0, tf_expr)
    target_ranks = np.apply_along_axis(rankdata, 0, target_expr)

    # Compute correlation matrix using Pearson on ranks (= Spearman)
    # Center the ranks
    tf_centered = tf_ranks - tf_ranks.mean(axis=0)
    target_centered = target_ranks - target_ranks.mean(axis=0)

    # Normalize
    tf_norm = np.sqrt((tf_centered ** 2).sum(axis=0))
    target_norm = np.sqrt((target_centered ** 2).sum(axis=0))

    # Avoid division by zero
    tf_norm[tf_norm == 0] = 1
    target_norm[target_norm == 0] = 1

    tf_normalized = tf_centered / tf_norm
    target_normalized = target_centered / target_norm

    # Correlation matrix: TFs x targets
    corr_matrix = tf_normalized.T @ target_normalized
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")

    # Build edge DataFrame
    edges = []
    for i, tf in enumerate(available_tfs):
        for j, target in enumerate(target_genes):
            if tf == target:
                continue  # skip self-edges
            rho = corr_matrix[i, j]
            if abs(rho) > 0.05:  # filter very weak correlations
                edges.append({
                    'tf': tf,
                    'target': target,
                    'rho': rho,
                    'abs_rho': abs(rho),
                })

    edges_df = pd.DataFrame(edges)
    edges_df = edges_df.sort_values('abs_rho', ascending=False).reset_index(drop=True)
    print(f"    Edges with |rho| > 0.05: {len(edges_df)}")
    print(f"    Top edges:")
    print(edges_df.head(20).to_string())

    return edges_df, available_tfs, target_genes


# Load and preprocess
human = sc.read_h5ad("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad")
mouse = sc.read_h5ad("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/krasnow_lung_smartsq2.h5ad")

human_proc = preprocess_and_filter(human, "Human")
mouse_proc = preprocess_and_filter(mouse, "Mouse")

# Compute edges
human_edges, human_tfs, human_targets = compute_tf_target_correlations(human_proc, TF_LIST, "Human")
mouse_edges, mouse_tfs, mouse_targets = compute_tf_target_correlations(mouse_proc, TF_LIST, "Mouse")

# Save
human_edges.to_csv(os.path.join(OUT_DIR, "human_edges.csv"), index=False)
mouse_edges.to_csv(os.path.join(OUT_DIR, "mouse_edges.csv"), index=False)

# Save TF lists
with open(os.path.join(OUT_DIR, "tf_lists.json"), "w") as f:
    json.dump({
        "human_tfs": human_tfs,
        "mouse_tfs": mouse_tfs,
        "shared_tfs": sorted(set(human_tfs) & set(mouse_tfs)),
    }, f, indent=2)

print(f"\nShared TFs: {len(set(human_tfs) & set(mouse_tfs))}")
print(f"Human edges saved: {len(human_edges)}")
print(f"Mouse edges saved: {len(mouse_edges)}")
print("\nDone.")
