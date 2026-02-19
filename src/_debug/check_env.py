#!/usr/bin/env python3
"""Check environment: data, model, GPU."""
import scanpy as sc
import torch

# Check GPU
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check brain data
adata = sc.read_h5ad('/mnt/d/openclaw/intelligence-augmentation/data/brain_scrna/DLPFC_11k.h5ad')
print(f"\nData: {adata.shape}")
print(f"\nCell types:\n{adata.obs['cell_type'].value_counts()}")
print(f"\nVar columns: {list(adata.var.columns)}")
print(f"\nFirst 5 genes: {list(adata.var_names[:5])}")

# Check if ensembl IDs available
if 'gene_ids' in adata.var.columns:
    print(f"\nEnsembl IDs available: {list(adata.var['gene_ids'][:3])}")
elif 'ensembl_id' in adata.var.columns:
    print(f"\nEnsembl IDs available: {list(adata.var['ensembl_id'][:3])}")
else:
    print("\nNo ensembl ID column found - will need gene name mapping")

# Check Geneformer model
try:
    from transformers import BertModel, BertConfig
    model = BertModel.from_pretrained('ctheodoris/Geneformer', trust_remote_code=True, output_attentions=True)
    print(f"\nGeneformer loaded: {model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Vocab size: {model.config.vocab_size}")
except Exception as e:
    print(f"\nGeneformer load error: {e}")
    # Try alternative
    try:
        from geneformer import TranscriptomeTokenizer
        print("Geneformer package available, trying different model load...")
    except:
        print("Geneformer package not importable")
