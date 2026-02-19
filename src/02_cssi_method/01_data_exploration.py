"""
Step 1: Data Loading & Exploration
Load the 20k immune subset and examine structure.
"""
import scanpy as sc
import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw"
OUT_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_28_cell_state_conditional_circuits"

# Load the 20k immune subset (manageable size)
print("Loading 20k immune subset...")
adata = sc.read_h5ad(os.path.join(DATA_DIR, "tabula_sapiens_immune_subset_20000.h5ad"))

print(f"Shape: {adata.shape}")
print(f"\nobs columns: {list(adata.obs.columns)}")
print(f"\nvar columns: {list(adata.var.columns)}")

# Cell type annotations
for col in adata.obs.columns:
    nuniq = adata.obs[col].nunique()
    if nuniq < 50:
        print(f"\n--- {col} ({nuniq} unique) ---")
        print(adata.obs[col].value_counts().head(20))

# Basic gene info
print(f"\nTotal genes: {adata.n_vars}")
print(f"\nFirst 20 genes: {list(adata.var_names[:20])}")

# Check if raw counts or normalized
x_sample = adata.X[:10, :10]
if hasattr(x_sample, 'toarray'):
    x_sample = x_sample.toarray()
print(f"\nSample values (first 10x10):\n{x_sample}")
print(f"Min: {x_sample.min()}, Max: {x_sample.max()}, dtype: {adata.X.dtype}")

# Check for raw layer
if adata.raw is not None:
    print(f"\nRaw layer shape: {adata.raw.X.shape}")
if adata.layers:
    print(f"\nLayers: {list(adata.layers.keys())}")
