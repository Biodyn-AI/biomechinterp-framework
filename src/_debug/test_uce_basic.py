#!/usr/bin/env python3
"""
Test UCE (Universal Cell Embeddings) basic functionality for NMI paper multi-model validation.
UCE is a transformer-based foundation model for single-cell data.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import logging
logging.set_verbosity_error()
import json
from datetime import datetime
import traceback

def create_synthetic_tokenized_data(n_cells=200, n_genes=2000, vocab_size=30000):
    """Create synthetic tokenized single-cell data for transformer models"""
    print(f"Creating synthetic tokenized dataset: {n_cells} cells x {n_genes} genes")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create synthetic gene expression values and convert to tokens
    # Simulate different cell types
    n_celltypes = 4
    cells_per_type = n_cells // n_celltypes
    
    all_sequences = []
    cell_labels = []
    
    for i in range(n_celltypes):
        for j in range(cells_per_type):
            # Create expression-like pattern for this cell type
            # Use different ranges for different cell types
            base_range = (i + 1) * 100
            
            # Generate random "expression levels" and convert to token IDs
            # This is simplified - real UCE uses sophisticated tokenization
            expression_values = np.random.exponential(scale=base_range, size=n_genes)
            
            # Convert to discrete tokens (simplified binning)
            # In real UCE, this would be much more sophisticated
            token_ids = np.clip(expression_values.astype(int) % vocab_size, 0, vocab_size-1)
            
            # Truncate/pad to reasonable length for transformer
            max_length = 512  # Common transformer limit
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                # Pad with a special token (0)
                padding = np.zeros(max_length - len(token_ids), dtype=int)
                token_ids = np.concatenate([token_ids, padding])
            
            all_sequences.append(token_ids)
            cell_labels.append(f"CellType_{i}")
    
    return np.array(all_sequences), cell_labels

def test_uce_availability():
    """Test if UCE models are available and loadable"""
    print("=" * 50)
    print("Testing UCE Model Availability")
    print("=" * 50)
    
    uce_models = [
        "minwoosun/uce-100m",
        "minwoosun/uce-650m"
    ]
    
    available_models = []
    
    for model_name in uce_models:
        print(f"\nChecking {model_name}...")
        try:
            # Check if we can load the config
            config = AutoConfig.from_pretrained(model_name)
            print(f"  [OK] Config loaded: {config.model_type if hasattr(config, 'model_type') else 'unknown type'}")
            
            # Check model size
            if hasattr(config, 'hidden_size'):
                print(f"  Hidden size: {config.hidden_size}")
            if hasattr(config, 'num_hidden_layers'):
                print(f"  Layers: {config.num_hidden_layers}")
            
            available_models.append({
                "name": model_name,
                "status": "available",
                "config": config.to_dict() if hasattr(config, 'to_dict') else str(config)
            })
            
        except Exception as e:
            print(f"  [FAIL] Failed to load {model_name}: {e}")
            available_models.append({
                "name": model_name,
                "status": "failed",
                "error": str(e)
            })
    
    return available_models

def test_uce_basic(model_name="minwoosun/uce-100m"):
    """Test basic UCE functionality"""
    print(f"\n" + "=" * 50)
    print(f"Testing UCE Basic Functionality: {model_name}")
    print("=" * 50)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "status": "unknown",
        "details": {}
    }
    
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory: {memory_gb:.1f} GB")
        
        # Load model and tokenizer
        print(f"\nLoading UCE model: {model_name}")
        
        # Load config first
        config = AutoConfig.from_pretrained(model_name)
        print(f"Model config loaded")
        
        # Load tokenizer if available
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
            has_tokenizer = True
        except Exception as e:
            print(f"No tokenizer found, will use synthetic data: {e}")
            tokenizer = None
            has_tokenizer = False
        
        # Load model
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded to {device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Create test data
        if has_tokenizer:
            # Use tokenizer if available
            test_texts = ["Test sequence 1", "Test sequence 2"]  # Placeholder
            inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)
        else:
            # Create synthetic tokenized data
            sequences, labels = create_synthetic_tokenized_data(n_cells=20, n_genes=500)  # Small for testing
            input_ids = torch.tensor(sequences, dtype=torch.long).to(device)
        
        print(f"Input shape: {input_ids.shape}")
        
        # Forward pass
        print("\nRunning inference...")
        with torch.no_grad():
            outputs = model(input_ids)
            
        # Extract embeddings
        if hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state
            print(f"Hidden states shape: {embeddings.shape}")
        elif hasattr(outputs, 'pooler_output'):
            embeddings = outputs.pooler_output
            print(f"Pooler output shape: {embeddings.shape}")
        else:
            # Try to get first output
            embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
            print(f"Output shape: {embeddings.shape}")
        
        # Test attention extraction if available
        attention_weights = None
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attention_weights = outputs.attentions
            print(f"Attention weights: {len(attention_weights)} layers")
            print(f"Attention shape per layer: {attention_weights[0].shape}")
        else:
            print("No attention weights in output - may need output_attentions=True")
            
            # Try again with attention outputs
            try:
                with torch.no_grad():
                    outputs_with_attn = model(input_ids, output_attentions=True)
                    if hasattr(outputs_with_attn, 'attentions') and outputs_with_attn.attentions is not None:
                        attention_weights = outputs_with_attn.attentions
                        print(f"Attention weights (retry): {len(attention_weights)} layers")
            except Exception as e:
                print(f"Could not extract attention: {e}")
        
        # Store results
        results["status"] = "success"
        results["details"] = {
            "model_name": model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": device,
            "input_shape": list(input_ids.shape),
            "output_shape": list(embeddings.shape),
            "has_tokenizer": has_tokenizer,
            "has_attention": attention_weights is not None,
            "num_attention_layers": len(attention_weights) if attention_weights else 0,
            "config": config.to_dict() if hasattr(config, 'to_dict') else str(config)
        }
        
        print(f"\n[OK] UCE basic test SUCCESSFUL for {model_name}")
        return True, results, model, tokenizer
        
    except Exception as e:
        print(f"\n[FAIL] UCE basic test FAILED: {e}")
        print("Full traceback:")
        traceback.print_exc()
        results["status"] = "failed"
        results["details"]["error"] = str(e)
        results["details"]["traceback"] = traceback.format_exc()
        return False, results, None, None

def test_uce_scaling(model, tokenizer, model_name):
    """Test UCE scaling behavior"""
    print(f"\n" + "=" * 50)
    print(f"Testing UCE Scaling Behavior: {model_name}")
    print("=" * 50)
    
    scaling_results = []
    cell_counts = [50, 100, 200]  # Conservative for 6GB VRAM and transformer memory usage
    
    device = next(model.parameters()).device
    
    for n_cells in cell_counts:
        print(f"\nTesting with {n_cells} cells...")
        
        try:
            # Create synthetic data
            if tokenizer:
                # If we have a tokenizer, create text-like data
                test_sequences = [f"synthetic cell {i} data" for i in range(n_cells)]
                inputs = tokenizer(test_sequences, return_tensors="pt", padding=True, truncation=True, max_length=256)
                input_ids = inputs["input_ids"].to(device)
            else:
                # Create synthetic tokenized data
                sequences, labels = create_synthetic_tokenized_data(n_cells=n_cells, n_genes=500)
                input_ids = torch.tensor(sequences, dtype=torch.long).to(device)
            
            print(f"  Input shape: {input_ids.shape}")
            
            # Batch processing for memory efficiency
            batch_size = min(8, n_cells)  # Small batches for memory
            all_embeddings = []
            
            with torch.no_grad():
                for i in range(0, input_ids.shape[0], batch_size):
                    batch = input_ids[i:i+batch_size]
                    outputs = model(batch, output_attentions=True if n_cells <= 100 else False)
                    
                    # Extract embeddings
                    if hasattr(outputs, 'last_hidden_state'):
                        batch_emb = outputs.last_hidden_state.mean(dim=1)  # Average over sequence
                    else:
                        batch_emb = outputs[0].mean(dim=1) if len(outputs[0].shape) > 2 else outputs[0]
                    
                    all_embeddings.append(batch_emb)
            
            # Concatenate all embeddings
            embeddings = torch.cat(all_embeddings, dim=0)
            
            # Simple quality metrics
            embedding_mean = embeddings.mean().item()
            embedding_std = embeddings.std().item()
            
            result = {
                "n_cells": n_cells,
                "input_shape": list(input_ids.shape),
                "embedding_shape": list(embeddings.shape),
                "embedding_mean": embedding_mean,
                "embedding_std": embedding_std,
                "batch_size": batch_size,
                "status": "success"
            }
            
            scaling_results.append(result)
            print(f"  [OK] {n_cells} cells: embedding_shape={embeddings.shape}, mean={embedding_mean:.4f}")
            
            # Clear cache
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            
        except Exception as e:
            print(f"  [FAIL] {n_cells} cells: FAILED - {e}")
            scaling_results.append({
                "n_cells": n_cells,
                "status": "failed",
                "error": str(e)
            })
    
    return scaling_results

def main():
    """Main testing routine"""
    print("[MICROSCOPE] UCE Multi-Model Validation Test")
    print("Timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    all_results = {
        "model_name": "UCE",
        "test_timestamp": datetime.now().isoformat(),
        "availability_test": [],
        "basic_test": {},
        "scaling_test": [],
        "summary": {}
    }
    
    # Check availability
    available_models = test_uce_availability()
    all_results["availability_test"] = available_models
    
    # Find a working model
    working_models = [m for m in available_models if m["status"] == "available"]
    
    if not working_models:
        print("\n[FAIL] No UCE models available")
        all_results["summary"]["status"] = "no_models_available"
    else:
        # Test the smallest available model first
        model_to_test = "minwoosun/uce-100m"  # Start with smaller model
        
        print(f"\n[TEST] Testing with {model_to_test}")
        
        # Basic functionality test
        basic_success, basic_results, model, tokenizer = test_uce_basic(model_to_test)
        all_results["basic_test"] = basic_results
        
        if basic_success and model is not None:
            # Scaling test
            scaling_results = test_uce_scaling(model, tokenizer, model_to_test)
            all_results["scaling_test"] = scaling_results
            
            # Analyze results
            successful_scales = [r for r in scaling_results if r.get("status") == "success"]
            if len(successful_scales) >= 2:
                means = [r["embedding_mean"] for r in successful_scales]
                scaling_change = ((means[-1] - means[0]) / abs(means[0])) * 100 if means[0] != 0 else 0
                all_results["summary"]["scaling_behavior"] = f"{scaling_change:.2f}% change"
                print(f"\n[CHART] Scaling analysis: {scaling_change:.2f}% change in embedding mean")
            else:
                all_results["summary"]["scaling_behavior"] = "insufficient_data"
            
            all_results["summary"]["status"] = "success"
        else:
            all_results["summary"]["status"] = "basic_test_failed"
    
    # Save results
    with open("uce_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[SAVE] Results saved to uce_test_results.json")
    
    return all_results

if __name__ == "__main__":
    main()