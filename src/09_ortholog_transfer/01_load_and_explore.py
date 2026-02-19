"""
Step 1: Load human and mouse lung data, explore basic properties.
"""
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
import json

OUT_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_30_ortholog_transfer_stress_test/implementation"

print("=" * 60)
print("STEP 1: Loading and exploring data")
print("=" * 60)

# Load human lung
print("\nLoading human lung (Tabula Sapiens)...")
human = sc.read_h5ad("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad")
print(f"  Shape: {human.shape}")
print(f"  Obs columns: {list(human.obs.columns)}")
print(f"  Var columns: {list(human.var.columns)}")
print(f"  First 10 genes: {list(human.var_names[:10])}")
if hasattr(human.obs, 'cell_type') or 'cell_type' in human.obs.columns:
    print(f"  Cell types: {human.obs['cell_type'].nunique()} unique")
    print(f"  Top cell types:\n{human.obs['cell_type'].value_counts().head(10)}")
elif 'cell_ontology_class' in human.obs.columns:
    print(f"  Cell types (cell_ontology_class): {human.obs['cell_ontology_class'].nunique()} unique")
    print(f"  Top cell types:\n{human.obs['cell_ontology_class'].value_counts().head(10)}")

# Load mouse lung
print("\n\nLoading mouse lung (Krasnow)...")
mouse = sc.read_h5ad("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/krasnow_lung_smartsq2.h5ad")
print(f"  Shape: {mouse.shape}")
print(f"  Obs columns: {list(mouse.obs.columns)}")
print(f"  Var columns: {list(mouse.var.columns)}")
print(f"  First 10 genes: {list(mouse.var_names[:10])}")
if 'cell_type' in mouse.obs.columns:
    print(f"  Cell types: {mouse.obs['cell_type'].nunique()} unique")
    print(f"  Top cell types:\n{mouse.obs['cell_type'].value_counts().head(10)}")
elif 'cell_ontology_class' in mouse.obs.columns:
    print(f"  Cell types (cell_ontology_class): {mouse.obs['cell_ontology_class'].nunique()} unique")
    print(f"  Top cell types:\n{mouse.obs['cell_ontology_class'].value_counts().head(10)}")

# Check gene naming conventions
print("\n\nGene naming conventions:")
print(f"  Human genes (sample): {list(human.var_names[:20])}")
print(f"  Mouse genes (sample): {list(mouse.var_names[:20])}")

# Check if data is raw counts or normalized
print("\nData properties:")
print(f"  Human X dtype: {human.X.dtype}, max: {human.X.max():.2f}, min: {human.X.min():.2f}")
try:
    print(f"  Human X mean: {human.X.mean():.4f}")
except:
    import scipy.sparse as sp
    if sp.issparse(human.X):
        print(f"  Human X mean: {human.X.mean():.4f} (sparse)")
print(f"  Mouse X dtype: {mouse.X.dtype}, max: {mouse.X.max():.2f}, min: {mouse.X.min():.2f}")
try:
    print(f"  Mouse X mean: {mouse.X.mean():.4f}")
except:
    import scipy.sparse as sp
    if sp.issparse(mouse.X):
        print(f"  Mouse X mean: {mouse.X.mean():.4f} (sparse)")

# Check layers
print(f"\n  Human layers: {list(human.layers.keys()) if human.layers else 'None'}")
print(f"  Mouse layers: {list(mouse.layers.keys()) if mouse.layers else 'None'}")

# Save summary
summary = {
    "human_cells": int(human.shape[0]),
    "human_genes": int(human.shape[1]),
    "mouse_cells": int(mouse.shape[0]),
    "mouse_genes": int(mouse.shape[1]),
    "human_gene_sample": list(human.var_names[:50]),
    "mouse_gene_sample": list(mouse.var_names[:50]),
}
with open(os.path.join(OUT_DIR, "data_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nSummary saved.")
