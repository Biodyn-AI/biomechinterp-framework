#!/usr/bin/env python3
"""
Geneformer Cross-Context Consistency Experiment
Test if attention patterns differ across different cellular contexts (tissues)

This experiment tests whether Geneformer attention patterns are consistent
across different cellular contexts, similar to the NMI paper's analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModel, AutoConfig
from pathlib import Path
import json

print("Starting Geneformer Cross-Context Consistency Experiment...")

# Setup paths 
model_path = "D:/openclaw/intelligence-augmentation/models/Geneformer/Geneformer-V1-10M"
output_path = "D:/openclaw/biodyn-nmi-paper/multi_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_geneformer_model():
    """Load Geneformer V1-10M model"""
    print("Loading Geneformer V1-10M model...")
    try:
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, output_attentions=True)
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
        return model, config
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def create_context_specific_data(context_type="brain", n_cells=200):
    """
    Create context-specific synthetic data
    Simulate different expression patterns for different tissue types
    """
    print(f"Creating {context_type} context data with {n_cells} cells...")
    
    max_seq_length = 512
    vocab_size = 5000
    
    tokenized_data = []
    attention_mask = []
    
    # Set different random seeds for different contexts to create different patterns
    if context_type == "brain":
        np.random.seed(42)
        # Brain-like patterns: higher expression of neural genes (tokens 1000-2000)
        gene_probs = np.ones(vocab_size)
        gene_probs[1000:2000] *= 3  # Bias towards "neural" genes
    elif context_type == "liver":
        np.random.seed(123)
        # Liver-like patterns: higher expression of metabolic genes (tokens 2000-3000)
        gene_probs = np.ones(vocab_size)
        gene_probs[2000:3000] *= 3  # Bias towards "metabolic" genes
    elif context_type == "immune":
        np.random.seed(456)
        # Immune-like patterns: higher expression of immune genes (tokens 3000-4000)
        gene_probs = np.ones(vocab_size)
        gene_probs[3000:4000] *= 3  # Bias towards "immune" genes
    else:
        np.random.seed(789)
        gene_probs = np.ones(vocab_size)
    
    # Normalize probabilities
    gene_probs = gene_probs / gene_probs.sum()
    
    for i in range(n_cells):
        seq_len = np.random.randint(max_seq_length//2, max_seq_length)
        
        # Sample tokens based on context-specific probabilities
        tokens = np.random.choice(range(1, vocab_size), size=seq_len, p=gene_probs[1:]/gene_probs[1:].sum())
        
        # Pad to max length
        padded_tokens = np.zeros(max_seq_length, dtype=int)
        padded_tokens[:seq_len] = tokens
        
        # Create attention mask
        mask = np.zeros(max_seq_length, dtype=int)
        mask[:seq_len] = 1
        
        tokenized_data.append(padded_tokens)
        attention_mask.append(mask)
    
    return torch.tensor(tokenized_data, device=device), torch.tensor(attention_mask, device=device)

def extract_attention_patterns(model, input_ids, attention_mask, context_name):
    """Extract and summarize attention patterns for a specific context"""
    print(f"Extracting attention for {context_name} context...")
    
    batch_size = 4
    n_samples = input_ids.shape[0]
    all_attentions = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_input = input_ids[i:end_idx]
            batch_mask = attention_mask[i:end_idx]
            
            outputs = model(input_ids=batch_input, attention_mask=batch_mask)
            batch_attentions = outputs.attentions
            
            # Average across heads for each layer
            layer_attentions = []
            for layer_att in batch_attentions:
                avg_heads = layer_att.mean(dim=1)
                layer_attentions.append(avg_heads.cpu().numpy())
            
            all_attentions.append(layer_attentions)
    
    # Combine across batches
    combined_attentions = []
    n_layers = len(all_attentions[0])
    
    for layer_idx in range(n_layers):
        layer_data = np.concatenate([batch[layer_idx] for batch in all_attentions], axis=0)
        combined_attentions.append(layer_data)
    
    # Compute attention statistics for this context
    last_layer_att = combined_attentions[-1]  # Use last layer
    attention_mask_np = attention_mask.cpu().numpy()
    
    # Mask and average
    masked_att = last_layer_att * attention_mask_np[:, :, None] * attention_mask_np[:, None, :]
    avg_attention = np.mean(masked_att, axis=0)
    
    # Compute summary statistics
    stats = {
        'context': context_name,
        'avg_attention_strength': float(np.mean(avg_attention[avg_attention > 0])),
        'attention_sparsity': float(np.mean(avg_attention > 0.01)),
        'max_attention': float(np.max(avg_attention)),
        'attention_entropy': float(-np.sum(avg_attention * np.log(avg_attention + 1e-8))),
        'top_attention_positions': avg_attention.flatten().argsort()[-10:].tolist()
    }
    
    print(f"  Average attention strength: {stats['avg_attention_strength']:.4f}")
    print(f"  Attention sparsity: {stats['attention_sparsity']:.4f}")
    
    return stats, avg_attention

def compute_attention_similarity(att1, att2):
    """Compute similarity between attention matrices"""
    # Flatten and normalize
    att1_flat = att1.flatten()
    att2_flat = att2.flatten()
    
    # Remove zeros for meaningful comparison
    nonzero_mask = (att1_flat > 0) & (att2_flat > 0)
    if np.sum(nonzero_mask) < 10:
        return 0.0
    
    att1_nz = att1_flat[nonzero_mask]
    att2_nz = att2_flat[nonzero_mask]
    
    # Pearson correlation
    correlation = np.corrcoef(att1_nz, att2_nz)[0, 1]
    
    return float(correlation) if not np.isnan(correlation) else 0.0

def run_cross_context_experiment():
    """Test attention pattern consistency across different cellular contexts"""
    print("\n=== GENEFORMER CROSS-CONTEXT CONSISTENCY EXPERIMENT ===")
    
    model, config = load_geneformer_model()
    if model is None:
        return None
    
    # Test different contexts
    contexts = ["brain", "liver", "immune", "generic"]
    n_cells_per_context = 150
    
    context_results = {}
    attention_matrices = {}
    
    # Extract attention for each context
    for context in contexts:
        print(f"\nProcessing {context} context...")
        
        # Generate context-specific data
        input_ids, attention_mask = create_context_specific_data(context, n_cells_per_context)
        
        # Extract attention patterns
        stats, avg_attention = extract_attention_patterns(model, input_ids, attention_mask, context)
        
        context_results[context] = stats
        attention_matrices[context] = avg_attention
    
    # Compute pairwise similarities between contexts
    print("\nComputing cross-context attention similarities...")
    similarities = {}
    
    context_list = list(contexts)
    for i, ctx1 in enumerate(context_list):
        for j, ctx2 in enumerate(context_list):
            if i < j:  # Only compute upper triangle
                sim = compute_attention_similarity(attention_matrices[ctx1], attention_matrices[ctx2])
                similarities[f"{ctx1}_vs_{ctx2}"] = sim
                print(f"  {ctx1} vs {ctx2}: {sim:.3f}")
    
    # Compile final results
    results = {
        'context_stats': context_results,
        'cross_context_similarities': similarities,
        'experiment_config': {
            'n_cells_per_context': n_cells_per_context,
            'contexts': contexts,
            'model': 'Geneformer-V1-10M'
        }
    }
    
    # Save results
    with open(f"{output_path}/geneformer_cross_context_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    """Run cross-context experiment"""
    print("Starting Geneformer cross-context consistency analysis...")
    
    os.makedirs(output_path, exist_ok=True)
    
    results = run_cross_context_experiment()
    
    if results:
        print("\n=== EXPERIMENT COMPLETE ===")
        print("\nCross-context similarity summary:")
        for pair, sim in results['cross_context_similarities'].items():
            print(f"{pair}: {sim:.3f}")
        
        # Analyze consistency
        similarities = list(results['cross_context_similarities'].values())
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        
        print(f"\nOverall cross-context consistency:")
        print(f"Average similarity: {avg_similarity:.3f}")
        print(f"Std deviation: {std_similarity:.3f}")
        
        if avg_similarity > 0.5:
            print("HIGH consistency - attention patterns are similar across contexts")
        elif avg_similarity > 0.3:
            print("MODERATE consistency - some context-specific patterns")
        else:
            print("LOW consistency - context-specific attention patterns")
    else:
        print("Experiment failed - check errors above")

if __name__ == "__main__":
    main()