"""Step 1: Explore metadata in available h5ad files."""
import anndata as ad
import pandas as pd
import sys

datasets = {
    "immune_subset": "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad",
    "kidney": "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad",
    "lung": "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad",
}

for name, path in datasets.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"Path: {path}")
    print(f"{'='*60}")
    try:
        adata = ad.read_h5ad(path, backed='r')
        print(f"Shape: {adata.shape}")
        print(f"\n.obs columns: {list(adata.obs.columns)}")
        print(f"\n.obs dtypes:\n{adata.obs.dtypes}")

        # Check for donor/batch-like columns
        batch_keywords = ['donor', 'batch', 'channel', 'sample', 'patient', 'subject', 'library', 'plate']
        found_cols = []
        for col in adata.obs.columns:
            for kw in batch_keywords:
                if kw in col.lower():
                    found_cols.append(col)
                    break

        print(f"\nPotential batch/donor columns: {found_cols}")
        for col in found_cols:
            vals = adata.obs[col]
            n_unique = vals.nunique()
            print(f"\n  {col}: {n_unique} unique values")
            if n_unique <= 30:
                print(f"    Values: {vals.value_counts().to_dict()}")

        # Also check cell_type column
        for col in ['cell_type', 'cell_ontology_class', 'compartment', 'tissue', 'organ']:
            if col in adata.obs.columns:
                n_unique = adata.obs[col].nunique()
                print(f"\n  {col}: {n_unique} unique values")
                if n_unique <= 30:
                    print(f"    Top values: {adata.obs[col].value_counts().head(10).to_dict()}")

        adata.file.close()
    except Exception as e:
        print(f"Error loading {name}: {e}")
        import traceback; traceback.print_exc()
