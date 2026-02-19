#!/usr/bin/env python3
"""
Test C2S-Pythia basic functionality for NMI paper multi-model validation.
C2S-Pythia is a transformer-based model trained on diverse single-cell tasks.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoConfig, GPTNeoXForCausalLM
from transformers import logging
logging.set_verbosity_error()
import json
from datetime import datetime
import traceback

def create_synthetic_text_data(n_cells=200):
    """Create synthetic text data for C2S-Pythia"""
    print(f"Creating synthetic text dataset: {n_cells} cell descriptions")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    cell_types = ["T_cell", "B_cell", "NK_cell", "monocyte", "neutrophil", "platelet"]
    conditions = ["healthy", "diseased", "activated", "naive"]
    tissues = ["blood", "bone_marrow", "spleen", "lymph_node"]
    
    sequences = []
    labels = []
    
    for i in range(n_cells):
        # Generate semi-realistic cell descriptions
        cell_type = np.random.choice(cell_types)
        condition = np.random.choice(conditions)
        tissue = np.random.choice(tissues)
        
        # Create description that might resemble training data
        description = f"single cell {cell_type} from {tissue} tissue in {condition} state expressing genes"
        
        # Add some gene names
        n_genes = np.random.randint(5, 15)
        gene_names = [f"GENE{np.random.randint(1, 1000)}" for _ in range(n_genes)]
        gene_list = " ".join(gene_names)
        
        full_sequence = f"{description} {gene_list}"
        sequences.append(full_sequence)
        labels.append(f"{cell_type}_{condition}")
    
    return sequences, labels

def test_c2s_pythia_availability():
    """Test if C2S-Pythia model is available and loadable"""
    print("=" * 50)
    print("Testing C2S-Pythia Model Availability")
    print("=" * 50)
    
    model_name = "vandijklab/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks"
    
    try:
        print(f"Checking {model_name}...")
        
        # Check if we can load the config
        config = AutoConfig.from_pretrained(model_name)
        print(f"  [OK] Config loaded: {config.model_type if hasattr(config, 'model_type') else 'unknown type'}")
        
        # Check model size
        if hasattr(config, 'hidden_size'):
            print(f"  Hidden size: {config.hidden_size}")
        if hasattr(config, 'num_hidden_layers'):
            print(f"  Layers: {config.num_hidden_layers}")
        if hasattr(config, 'vocab_size'):
            print(f"  Vocab size: {config.vocab_size}")
        
        # Try to load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"  [OK] Tokenizer loaded")
            has_tokenizer = True
        except Exception as e:
            print(f"  [WARN] No tokenizer: {e}")
            has_tokenizer = False
        
        return {
            "name": model_name,
            "status": "available",
            "config": config.to_dict() if hasattr(config, 'to_dict') else str(config),
            "has_tokenizer": has_tokenizer
        }
        
    except Exception as e:
        print(f"  [FAIL] Failed to load {model_name}: {e}")
        return {
            "name": model_name,
            "status": "failed",
            "error": str(e)
        }

def test_c2s_pythia_basic():
    """Test basic C2S-Pythia functionality"""
    print(f"\n" + "=" * 50)
    print("Testing C2S-Pythia Basic Functionality")
    print("=" * 50)
    
    model_name = "vandijklab/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks"
    
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
        print(f"\nLoading C2S-Pythia model...")
        
        # Load config first
        config = AutoConfig.from_pretrained(model_name)
        print(f"Model config loaded: {config.model_type}")
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
            has_tokenizer = True
        except Exception as e:
            print(f"No tokenizer found: {e}")
            tokenizer = None
            has_tokenizer = False
            
        # Load model - try different loading methods
        try:
            # Try as causal LM first (most likely for Pythia)
            model = GPTNeoXForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            model_type = "CausalLM"
        except Exception as e1:
            print(f"Failed to load as CausalLM: {e1}")
            try:
                # Try generic AutoModel
                model = AutoModel.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                model = model.to(device)
                model_type = "AutoModel"
            except Exception as e2:
                print(f"Failed to load as AutoModel: {e2}")
                raise e2
        
        model.eval()
        print(f"Model loaded as {model_type} to {device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Create test data
        if has_tokenizer:
            # Create realistic single-cell text data
            sequences, labels = create_synthetic_text_data(n_cells=20)  # Small for testing
            
            # Tokenize
            inputs = tokenizer(
                sequences, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256  # Keep short for memory
            )
            
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        else:
            # Fallback to random tokens
            vocab_size = getattr(config, 'vocab_size', 50000)
            input_ids = torch.randint(0, vocab_size, (20, 128), dtype=torch.long).to(device)
            attention_mask = torch.ones_like(input_ids)
        
        print(f"Input shape: {input_ids.shape}")
        
        # Forward pass
        print("\nRunning inference...")
        with torch.no_grad():
            if model_type == "CausalLM":
                outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
                
                # For causal LM, we get logits and optionally hidden states
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    print(f"Logits shape: {logits.shape}")
                    
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden_states = outputs.hidden_states
                    print(f"Hidden states: {len(hidden_states)} layers")
                    embeddings = hidden_states[-1]  # Last layer
                else:
                    # Use logits as embeddings if no hidden states
                    embeddings = logits.mean(dim=-1)  # Average over vocab
                    
            else:
                # AutoModel case
                outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
                
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state
                else:
                    embeddings = outputs[0]
        
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Test attention extraction
        attention_weights = None
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attention_weights = outputs.attentions
            print(f"Attention weights: {len(attention_weights)} layers")
            if len(attention_weights) > 0 and attention_weights[0] is not None:
                print(f"Attention shape per layer: {attention_weights[0].shape}")
            else:
                print("Attention weights list contains None values")
        else:
            print("No attention weights extracted")
        
        # Store results
        results["status"] = "success"
        results["details"] = {
            "model_name": model_name,
            "model_type": model_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": device,
            "input_shape": list(input_ids.shape),
            "embedding_shape": list(embeddings.shape),
            "has_tokenizer": has_tokenizer,
            "has_attention": attention_weights is not None and len(attention_weights) > 0,
            "num_attention_layers": len(attention_weights) if attention_weights else 0,
            "config": config.to_dict() if hasattr(config, 'to_dict') else str(config)
        }
        
        print(f"\n[OK] C2S-Pythia basic test SUCCESSFUL")
        return True, results, model, tokenizer
        
    except Exception as e:
        print(f"\n[FAIL] C2S-Pythia basic test FAILED: {e}")
        print("Full traceback:")
        traceback.print_exc()
        results["status"] = "failed"
        results["details"]["error"] = str(e)
        results["details"]["traceback"] = traceback.format_exc()
        return False, results, None, None

def test_c2s_pythia_scaling(model, tokenizer):
    """Test C2S-Pythia scaling behavior"""
    print(f"\n" + "=" * 50)
    print("Testing C2S-Pythia Scaling Behavior")
    print("=" * 50)
    
    scaling_results = []
    cell_counts = [50, 100, 200]  # Conservative for transformer memory
    
    device = next(model.parameters()).device
    model_type = "CausalLM" if hasattr(model, 'generate') else "AutoModel"
    
    for n_cells in cell_counts:
        print(f"\nTesting with {n_cells} cells...")
        
        try:
            # Create text data
            sequences, labels = create_synthetic_text_data(n_cells=n_cells)
            
            if tokenizer:
                # Tokenize with shorter sequences for memory
                inputs = tokenizer(
                    sequences, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=128  # Shorter for larger batches
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
            else:
                # Fallback random data
                input_ids = torch.randint(0, 50000, (n_cells, 128), dtype=torch.long).to(device)
                attention_mask = torch.ones_like(input_ids)
            
            print(f"  Input shape: {input_ids.shape}")
            
            # Batch processing for memory efficiency
            batch_size = min(16, n_cells)  # Small batches for GPU memory
            all_embeddings = []
            
            with torch.no_grad():
                for i in range(0, input_ids.shape[0], batch_size):
                    batch_ids = input_ids[i:i+batch_size]
                    batch_mask = attention_mask[i:i+batch_size] if attention_mask is not None else None
                    
                    # Run model
                    outputs = model(
                        batch_ids, 
                        attention_mask=batch_mask,
                        output_attentions=True if n_cells <= 100 else False  # Skip attention for large batches
                    )
                    
                    # Extract embeddings based on model type
                    if model_type == "CausalLM":
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                            batch_emb = outputs.hidden_states[-1].mean(dim=1)  # Last layer, avg over sequence
                        elif hasattr(outputs, 'logits'):
                            batch_emb = outputs.logits.mean(dim=1)  # Average logits over sequence
                        else:
                            batch_emb = outputs[0].mean(dim=1)
                    else:
                        if hasattr(outputs, 'last_hidden_state'):
                            batch_emb = outputs.last_hidden_state.mean(dim=1)
                        else:
                            batch_emb = outputs[0].mean(dim=1)
                    
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
    print("[MICROSCOPE] C2S-Pythia Multi-Model Validation Test")
    print("Timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    all_results = {
        "model_name": "C2S-Pythia",
        "test_timestamp": datetime.now().isoformat(),
        "availability_test": {},
        "basic_test": {},
        "scaling_test": [],
        "summary": {}
    }
    
    # Check availability
    availability_result = test_c2s_pythia_availability()
    all_results["availability_test"] = availability_result
    
    if availability_result["status"] != "available":
        print("\n[FAIL] C2S-Pythia model not available")
        all_results["summary"]["status"] = "model_not_available"
    else:
        print(f"\n[TEST] Testing C2S-Pythia")
        
        # Basic functionality test
        basic_success, basic_results, model, tokenizer = test_c2s_pythia_basic()
        all_results["basic_test"] = basic_results
        
        if basic_success and model is not None:
            # Scaling test
            scaling_results = test_c2s_pythia_scaling(model, tokenizer)
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
    with open("c2s_pythia_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[SAVE] Results saved to c2s_pythia_test_results.json")
    
    return all_results

if __name__ == "__main__":
    main()