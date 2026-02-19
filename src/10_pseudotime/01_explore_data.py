"""Step 1: Explore available data to understand trajectories and cell types."""
import sys
sys.path.insert(0, '/Users/ihorkendiukhov/biodyn-work')
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd

# Load the 20k immune subset first (smaller, faster)
print("=" * 60)
print("Loading immune subset (20k cells)...")
adata = sc.read_h5ad('/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad')
print(f"Shape: {adata.shape}")
print(f"\nobs columns: {list(adata.obs.columns)}")
print(f"\nvar columns: {list(adata.var.columns)}")
print(f"\nobsm keys: {list(adata.obsm.keys())}")
print(f"\nuns keys: {list(adata.uns.keys())}")
print(f"\nobsp keys: {list(adata.obsp.keys())}")

# Check cell types
if 'cell_type' in adata.obs.columns:
    print(f"\nCell types:\n{adata.obs['cell_type'].value_counts().head(20)}")
elif 'cell_ontology_class' in adata.obs.columns:
    print(f"\nCell types:\n{adata.obs['cell_ontology_class'].value_counts().head(20)}")

# Check for any existing pseudotime
for col in adata.obs.columns:
    if 'time' in col.lower() or 'pseudo' in col.lower() or 'dpt' in col.lower():
        print(f"\nPotential pseudotime column: {col}")
        print(adata.obs[col].describe())

# Check if PCA/neighbors already computed
if 'X_pca' in adata.obsm:
    print(f"\nPCA already computed: {adata.obsm['X_pca'].shape}")
if 'X_umap' in adata.obsm:
    print(f"\nUMAP already computed: {adata.obsm['X_umap'].shape}")

# Check raw data
print(f"\nX type: {type(adata.X)}")
if hasattr(adata.X, 'data'):
    print(f"X data range: [{adata.X.data.min():.2f}, {adata.X.data.max():.2f}]")
    print(f"X sparsity: {1 - adata.X.nnz / np.prod(adata.X.shape):.3f}")
else:
    sample = adata.X[:10, :10]
    print(f"X sample:\n{sample}")

# Check var names for known TFs
known_tfs = ['GATA1', 'GATA2', 'GATA3', 'SPI1', 'CEBPA', 'CEBPB', 'IRF4', 'IRF8',
             'PAX5', 'EBF1', 'TCF7', 'LEF1', 'FOXP3', 'TBX21', 'RUNX1', 'RUNX3',
             'BCL6', 'BLIMP1', 'PRDM1', 'STAT1', 'STAT3', 'STAT4', 'STAT5A', 'STAT6',
             'MYC', 'TP53', 'NFKB1', 'RELA', 'JUN', 'FOS', 'ETS1', 'ETV6',
             'KLF2', 'KLF4', 'BATF', 'BATF3', 'EOMES', 'MAF', 'RORC', 'AHR',
             'ID2', 'ID3', 'TCF3', 'IKZF1', 'IKZF3', 'ZEB1', 'ZEB2']
found_tfs = [tf for tf in known_tfs if tf in adata.var_names]
print(f"\nFound {len(found_tfs)}/{len(known_tfs)} known immune TFs: {found_tfs}")

# Also check tissue/compartment
for col in ['tissue', 'organ', 'compartment', 'tissue_in_publication', 'method']:
    if col in adata.obs.columns:
        print(f"\n{col}:\n{adata.obs[col].value_counts().head(10)}")
