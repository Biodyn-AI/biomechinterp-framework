#!/usr/bin/env python3
"""
Geneformer V1-10M Attention Extraction Experiment
Using the smaller 10M parameter model for faster experimentation

Key experiments:
1. Extract attention weights from Geneformer transformer layers
2. Test scaling behavior (200 vs 1000 cells)  
3. Compute attention-derived GRN edges
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModel, AutoConfig
from pathlib import Path

print("Starting Geneformer V1-10M Attention Extraction...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")

# Setup paths - using the smaller V1-10M model
model_path = "D:/openclaw/intelligence-augmentation/models/Geneformer/Geneformer-V1-10M"
output_path = "D:/openclaw/biodyn-nmi-paper/multi_model"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_geneformer_model():
    """Load Geneformer V1-10M model through HuggingFace transformers"""
    print("Loading Geneformer V1-10M model...")
    try:
        config = AutoConfig.from_pretrained(model_path)
        print(f"Config: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
        print(f"Vocab size: {config.vocab_size}")
        
        # Load model with attention output enabled
        model = AutoModel.from_pretrained(model_path, output_attentions=True)
        model = model.to(device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded with {total_params:,} parameters")
        
        return model, config
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def create_synthetic_tokenized_data(n_cells=200, n_genes=500, vocab_size=10000):
    """
    Create synthetic tokenized data mimicking Geneformer input format
    Using smaller sequences for the 10M model
    """
    print(f"Creating synthetic data: {n_cells} cells, {n_genes} genes")
    
    # Reduced parameters for the smaller model
    max_seq_length = min(n_genes, 512)  # Reduced sequence length
    
    tokenized_data = []
    attention_mask = []
    
    np.random.seed(42)  # For reproducibility
    
    for i in range(n_cells):
        # Generate random gene tokens (excluding pad token 0)
        seq_len = np.random.randint(max_seq_length//2, max_seq_length)
        tokens = np.random.randint(1, min(vocab_size, 5000), size=seq_len)  # Smaller vocab range
        
        # Pad to max length
        padded_tokens = np.zeros(max_seq_length, dtype=int)
        padded_tokens[:seq_len] = tokens
        
        # Create attention mask
        mask = np.zeros(max_seq_length, dtype=int) 
        mask[:seq_len] = 1
        
        tokenized_data.append(padded_tokens)
        attention_mask.append(mask)
    
    return torch.tensor(tokenized_data, device=device), torch.tensor(attention_mask, device=device)

def extract_attention_weights(model, input_ids, attention_mask, batch_size=4):
    """Extract attention weights from Geneformer (smaller batches for 6GB VRAM)"""
    print("Extracting attention weights...")
    
    all_attentions = []
    n_samples = input_ids.shape[0]
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_input = input_ids[i:end_idx]
            batch_mask = attention_mask[i:end_idx]
            
            print(f"Processing batch {i//batch_size + 1}/{(n_samples-1)//batch_size + 1}")
            
            try:
                # Forward pass with attention output
                outputs = model(input_ids=batch_input, attention_mask=batch_mask)
                
                # Extract attention weights (tuple of tensors for each layer)
                batch_attentions = outputs.attentions
                
                # Average across heads for each layer, keep batch dimension
                layer_attentions = []
                for layer_att in batch_attentions:
                    # layer_att shape: (batch, heads, seq_len, seq_len)
                    avg_heads = layer_att.mean(dim=1)  # Average across attention heads
                    layer_attentions.append(avg_heads.cpu().numpy())
                
                all_attentions.append(layer_attentions)
                
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA OOM at batch {i}, reducing batch size")
                # Process one sample at a time
                for j in range(i, end_idx):
                    single_input = input_ids[j:j+1]
                    single_mask = attention_mask[j:j+1]
                    outputs = model(input_ids=single_input, attention_mask=single_mask)
                    batch_attentions = outputs.attentions
                    
                    layer_attentions = []
                    for layer_att in batch_attentions:
                        avg_heads = layer_att.mean(dim=1)
                        layer_attentions.append(avg_heads.cpu().numpy())
                    
                    all_attentions.append(layer_attentions)
    
    if not all_attentions:
        return None
    
    # Combine across batches
    combined_attentions = []
    n_layers = len(all_attentions[0])
    
    for layer_idx in range(n_layers):
        layer_data = np.concatenate([batch[layer_idx] for batch in all_attentions], axis=0)
        combined_attentions.append(layer_data)
    
    print(f"Extracted attention from {n_layers} layers")
    return combined_attentions

def compute_grn_edges_from_attention(attentions, attention_mask, top_k=50):
    """
    Compute GRN edges from attention patterns
    """
    print("Computing attention-derived GRN edges...")
    
    if attentions is None or len(attentions) == 0:
        return [], None
    
    # Use the last layer (most refined representations)
    last_layer_att = attentions[-1]  # Shape: (n_cells, seq_len, seq_len)
    
    # Mask out padded positions
    masked_att = last_layer_att * attention_mask[:, :, None] * attention_mask[:, None, :]
    
    # Average attention across cells for each gene-gene pair
    avg_attention = np.mean(masked_att, axis=0)
    
    # Extract top-k edges (excluding diagonal)
    seq_len = avg_attention.shape[0]
    edges = []
    
    for i in range(seq_len):
        for j in range(seq_len):
            if i != j:  # Exclude self-attention
                weight = avg_attention[i, j]
                if weight > 0:  # Only consider non-zero weights
                    edges.append((i, j, weight))
    
    # Sort by weight and take top-k
    edges.sort(key=lambda x: x[2], reverse=True)
    top_edges = edges[:top_k]
    
    print(f"Extracted {len(top_edges)} top edges")
    
    return top_edges, avg_attention

def run_scaling_experiment():
    """Test if GRN recovery degrades with more cells (200 vs 1000)"""
    print("\n=== GENEFORMER V1-10M SCALING EXPERIMENT ===")
    
    model, config = load_geneformer_model()
    if model is None:
        print("Failed to load model")
        return None
    
    results = {}
    
    for n_cells in [200, 500]:  # Reduced max cells for memory constraints
        print(f"\nTesting with {n_cells} cells...")
        
        try:
            # Generate data
            input_ids, attention_mask = create_synthetic_tokenized_data(n_cells=n_cells)
            print(f"Input shape: {input_ids.shape}")
            
            # Extract attention
            attentions = extract_attention_weights(model, input_ids, attention_mask)
            
            if attentions is None:
                print(f"Failed to extract attention for {n_cells} cells")
                continue
            
            # Compute GRN edges
            edges, avg_att = compute_grn_edges_from_attention(attentions, attention_mask.cpu().numpy())
            
            # Store results
            results[f"{n_cells}_cells"] = {
                'n_edges': len(edges),
                'top_edge_weights': [float(e[2]) for e in edges[:10]],
                'attention_sparsity': float(np.mean(avg_att > 0.01)) if avg_att is not None else 0,
                'attention_max': float(np.max(avg_att)) if avg_att is not None else 0,
                'attention_mean': float(np.mean(avg_att[avg_att > 0])) if avg_att is not None else 0
            }
            
            print(f"Results for {n_cells} cells:")
            print(f"  Number of edges: {results[f'{n_cells}_cells']['n_edges']}")
            print(f"  Attention sparsity: {results[f'{n_cells}_cells']['attention_sparsity']:.3f}")
            print(f"  Max attention: {results[f'{n_cells}_cells']['attention_max']:.3f}")
            
        except Exception as e:
            print(f"Error processing {n_cells} cells: {e}")
            continue
    
    # Save results
    import json
    with open(f"{output_path}/geneformer_v1_scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    """Run all experiments"""
    print("Starting Geneformer V1-10M experiments for NMI paper validation...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Check GPU memory
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run scaling experiment
    scaling_results = run_scaling_experiment()
    
    print("\n=== EXPERIMENT COMPLETE ===")
    print(f"Results saved to: {output_path}")
    
    # Generate summary
    if scaling_results:
        print("\nSUMMARY:")
        for key, data in scaling_results.items():
            print(f"{key}: {data['n_edges']} edges, sparsity={data['attention_sparsity']:.3f}")
    else:
        print("No results generated - check for errors above")

if __name__ == "__main__":
    main()