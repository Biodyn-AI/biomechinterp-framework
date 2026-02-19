"""Check gene naming and find TFs."""
import scanpy as sc
import numpy as np

adata = sc.read_h5ad('/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad')

# Check gene naming convention
print("First 20 gene names:", list(adata.var_names[:20]))
print("\nSample var index names:")
for i in range(0, min(100, len(adata.var_names)), 10):
    print(f"  [{i}]: {adata.var_names[i]}")

# Check feature_name column
if 'feature_name' in adata.var.columns:
    print("\nfeature_name examples:", list(adata.var['feature_name'].head(20)))
    # Search for known TFs in feature_name
    known_tfs = ['GATA1', 'GATA2', 'GATA3', 'SPI1', 'CEBPA', 'CEBPB', 'IRF4', 'IRF8',
                 'PAX5', 'EBF1', 'TCF7', 'LEF1', 'FOXP3', 'TBX21', 'RUNX1', 'RUNX3',
                 'BCL6', 'PRDM1', 'STAT1', 'STAT3', 'STAT4', 'STAT5A', 'STAT6',
                 'MYC', 'TP53', 'NFKB1', 'RELA', 'JUN', 'FOS', 'ETS1', 'ETV6',
                 'KLF2', 'KLF4', 'BATF', 'BATF3', 'EOMES', 'MAF', 'RORC', 'AHR',
                 'ID2', 'ID3', 'TCF3', 'IKZF1', 'IKZF3', 'ZEB1', 'ZEB2']
    found = [tf for tf in known_tfs if tf in adata.var['feature_name'].values]
    print(f"\nFound TFs in feature_name: {found}")

# Check if var_names are ENSEMBL IDs
if adata.var_names[0].startswith('ENSG'):
    print("\nGene names are ENSEMBL IDs - need to use feature_name for lookup")
