"""
Step 2: Build human-mouse ortholog mapping.

Strategy:
1. Extract feature_name from both datasets
2. Use gene name capitalization mapping (mouse gene -> UPPER = human gene convention)
3. Also try Ensembl ID-based mapping if both use ENSG IDs (they might both be mapped to human genome)
4. Build a clean mapping table
"""
import scanpy as sc
import numpy as np
import pandas as pd
import os
import json

OUT_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_30_ortholog_transfer_stress_test/implementation"

print("=" * 60)
print("STEP 2: Building ortholog mapping")
print("=" * 60)

# Load data
human = sc.read_h5ad("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad")
mouse = sc.read_h5ad("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/krasnow_lung_smartsq2.h5ad")

# Check feature_name columns
print("\nHuman var head:")
print(human.var.head(10))
print("\nMouse var head:")
print(mouse.var.head(10))

# Both datasets seem to use ENSG IDs as var_names
# Check if the mouse data is actually mapped to human genome (common in CELLxGENE)
human_ensembl = set(human.var_names)
mouse_ensembl = set(mouse.var_names)
overlap_ensembl = human_ensembl & mouse_ensembl
print(f"\nEnsembl ID overlap: {len(overlap_ensembl)} / human {len(human_ensembl)} / mouse {len(mouse_ensembl)}")

# If there's substantial Ensembl overlap, the mouse data may already be mapped to human genome
# Let's check feature_name
human_names = human.var['feature_name'].dropna()
mouse_names = mouse.var['feature_name'].dropna()

print(f"\nHuman feature names (sample): {list(human_names[:20])}")
print(f"Mouse feature names (sample): {list(mouse_names[:20])}")

# Check naming convention
human_upper = set(human_names.str.upper())
mouse_upper = set(mouse_names.str.upper())
name_overlap = human_upper & mouse_upper
print(f"\nFeature name overlap (case-insensitive): {len(name_overlap)}")

# Strategy: Since both have ENSG IDs, check direct Ensembl overlap first
if len(overlap_ensembl) > 5000:
    print(f"\nUsing direct Ensembl ID mapping ({len(overlap_ensembl)} shared genes)")

    # Get shared genes with their feature names
    shared_genes = sorted(overlap_ensembl)

    # Build mapping table
    h_names = human.var.loc[shared_genes, 'feature_name'].values
    m_names = mouse.var.loc[shared_genes, 'feature_name'].values

    mapping = pd.DataFrame({
        'ensembl_id': shared_genes,
        'human_name': h_names,
        'mouse_name': m_names,
    })

    # Check if names match
    mapping['names_match'] = mapping['human_name'].str.upper() == mapping['mouse_name'].str.upper()
    print(f"  Names match (case-insensitive): {mapping['names_match'].sum()} / {len(mapping)}")
    print(f"  Names don't match: {(~mapping['names_match']).sum()}")
    print("\n  Non-matching examples:")
    non_match = mapping[~mapping['names_match']].head(20)
    print(non_match.to_string())

else:
    print("\nLittle Ensembl overlap - using gene name capitalization mapping")
    # Build name-based mapping
    # Human: typically UPPER (e.g., TP53)
    # Mouse: typically Title case (e.g., Tp53)

    human_name_to_ensembl = dict(zip(human_names.str.upper(), human_names.index))
    mouse_name_to_ensembl = dict(zip(mouse_names.str.upper(), mouse_names.index))

    shared_names = set(human_name_to_ensembl.keys()) & set(mouse_name_to_ensembl.keys())
    print(f"  Shared gene names (case-insensitive): {len(shared_names)}")

    mapping = pd.DataFrame({
        'human_ensembl': [human_name_to_ensembl[n] for n in shared_names],
        'mouse_ensembl': [mouse_name_to_ensembl[n] for n in shared_names],
        'gene_name_upper': list(shared_names),
    })

# Save mapping
mapping.to_csv(os.path.join(OUT_DIR, "ortholog_mapping.csv"), index=False)
print(f"\nSaved ortholog mapping: {len(mapping)} gene pairs")

# Summary statistics
summary = {
    "n_human_genes": int(human.shape[1]),
    "n_mouse_genes": int(mouse.shape[1]),
    "n_ensembl_overlap": int(len(overlap_ensembl)),
    "n_mapped_orthologs": int(len(mapping)),
}
with open(os.path.join(OUT_DIR, "ortholog_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nDone.")
