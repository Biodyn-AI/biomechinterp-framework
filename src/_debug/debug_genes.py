import scanpy as sc
import json
from pathlib import Path

# Load dataset 
adata = sc.read_h5ad('/mnt/d/openclaw/mechinterp-bio/biodyn-work/single_cell_mechinterp/data/raw/krasnow_lung_smartsq2.h5ad')
print('Dataset gene names (first 10):', list(adata.var_names[:10]))
print('Dataset var columns:', list(adata.var.columns))

# Load vocab using the correct method
import sys
sys.path.insert(0, '/mnt/d/openclaw/mechinterp-bio/biodyn-work/single_cell_mechinterp')
from src.model.vocab import load_vocab

vocab_path = '/mnt/d/openclaw/mechinterp-bio/biodyn-work/single_cell_mechinterp/external/scGPT_checkpoints/brain/vocab.json'
vocab = load_vocab(vocab_path)
gene_to_id = vocab.gene_to_id

print('Vocab gene names (first 10):', list(gene_to_id.keys())[:10])
print('Total vocab genes:', len(gene_to_id))

# Check overlap with Ensembl IDs (var_names)
dataset_genes = set(adata.var_names)
vocab_genes = set(gene_to_id.keys())
overlap = dataset_genes & vocab_genes
print('Ensembl ID overlap:', len(overlap), '/', len(dataset_genes))

# Check overlap with gene symbols from feature_name column
if 'feature_name' in adata.var.columns:
    print('Dataset has feature_name column - checking gene symbol overlap')
    feature_names = adata.var['feature_name'].dropna()
    symbol_genes = set(feature_names.astype(str))
    symbol_overlap = symbol_genes & vocab_genes
    print('Gene symbol overlap:', len(symbol_overlap), '/', len(symbol_genes))
    print('Example feature_names:', list(feature_names[:10]))
    
    # Check common symbols
    common_symbols = list(symbol_overlap)[:10]
    print('Common symbols found:', common_symbols)
    
# Look for any other column that might match
for col in adata.var.columns:
    if col not in ['feature_name'] and ('symbol' in col.lower() or 'gene' in col.lower()):
        print(f'Checking column {col}')
        col_genes = set(adata.var[col].dropna().astype(str))
        col_overlap = col_genes & vocab_genes
        print(f'{col} overlap:', len(col_overlap), '/', len(col_genes))