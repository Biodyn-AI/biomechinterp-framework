"""
Step 2a: Compute pseudotime for each lineage and save.
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

np.random.seed(42)

OUT_DIR = '/Users/ihorkendiukhov/biodyn-work/subproject_29_pseudotime_directionality_audit'

print("Loading data...")
adata = sc.read_h5ad('/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad')
print(f"Shape: {adata.shape}")

lineage_definitions = {
    'T_cell': {
        'cell_types': [
            'naive thymus-derived CD4-positive, alpha-beta T cell',
            'CD4-positive, alpha-beta T cell',
            'CD8-positive, alpha-beta T cell',
            'regulatory T cell',
            'gamma-delta T cell',
            'mature NK T cell',
            'T cell',
            'CD4-positive, alpha-beta thymocyte',
        ],
        'root_type': 'naive thymus-derived CD4-positive, alpha-beta T cell',
    },
    'B_cell': {
        'cell_types': ['B cell', 'plasma cell'],
        'root_type': 'B cell',
    },
    'Myeloid': {
        'cell_types': [
            'monocyte', 'classical monocyte', 'intermediate monocyte',
            'macrophage', 'mononuclear phagocyte',
        ],
        'root_type': 'classical monocyte',
    },
}

for name, defn in lineage_definitions.items():
    print(f"\n{'='*40}")
    print(f"Processing {name} lineage...")

    mask = adata.obs['cell_type'].isin(defn['cell_types'])
    adata_sub = adata[mask].copy()
    print(f"  {adata_sub.n_obs} cells")
    print(f"  Cell types: {dict(adata_sub.obs['cell_type'].value_counts())}")

    if adata_sub.n_obs < 100:
        print(f"  Skipping: too few cells")
        continue

    # Log-normalize if not already (data appears to be log-normalized based on range)
    # HVG using 'seurat' flavor (dispersion-based, no skmisc needed)
    sc.pp.highly_variable_genes(adata_sub, n_top_genes=2000, flavor='seurat')
    print(f"  HVG computed: {adata_sub.var['highly_variable'].sum()} genes")

    # PCA on subset
    sc.tl.pca(adata_sub, n_comps=30)
    print(f"  PCA done")

    # Neighbors
    sc.pp.neighbors(adata_sub, n_neighbors=15, n_pcs=30)
    print(f"  Neighbors done")

    # Diffusion map
    sc.tl.diffmap(adata_sub, n_comps=10)
    print(f"  Diffmap done")

    # Root cell
    root_mask = adata_sub.obs['cell_type'] == defn['root_type']
    if root_mask.sum() == 0:
        root_mask = adata_sub.obs['cell_type'] == defn['cell_types'][0]

    root_cells_pca = adata_sub[root_mask].obsm['X_pca']
    centroid = root_cells_pca.mean(axis=0)
    dists = np.linalg.norm(root_cells_pca - centroid, axis=1)
    root_idx = np.where(root_mask.values)[0][np.argmin(dists)]
    adata_sub.uns['iroot'] = root_idx

    # DPT
    sc.tl.dpt(adata_sub, n_branchings=0)
    print(f"  DPT done: range [{adata_sub.obs['dpt_pseudotime'].min():.3f}, {adata_sub.obs['dpt_pseudotime'].max():.3f}]")

    # UMAP
    sc.tl.umap(adata_sub)
    print(f"  UMAP done")

    # Print median pseudotime by cell type
    for ct in defn['cell_types']:
        ct_mask = adata_sub.obs['cell_type'] == ct
        if ct_mask.sum() > 0:
            med = adata_sub.obs.loc[ct_mask, 'dpt_pseudotime'].median()
            n = ct_mask.sum()
            print(f"    {ct}: median_pt={med:.3f} (n={n})")

    # Save
    out_path = os.path.join(OUT_DIR, 'implementation', f'adata_{name}.h5ad')
    adata_sub.write(out_path)
    print(f"  Saved to {out_path}")

print("\nDone with pseudotime computation!")
