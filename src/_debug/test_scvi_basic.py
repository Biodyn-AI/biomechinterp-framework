#!/usr/bin/env python3
"""
Test scVI basic functionality for NMI paper multi-model validation.
scVI is a variational autoencoder model for single-cell RNA-seq data.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scvi.model import SCVI
import anndata as ad
import json
from datetime import datetime

def create_synthetic_data(n_cells=500, n_genes=2000):
    """Create synthetic single-cell data for testing"""
    print(f"Creating synthetic dataset: {n_cells} cells x {n_genes} genes")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create synthetic count matrix
    # Simulate some structure with different cell types
    n_celltypes = 4
    cells_per_type = n_cells // n_celltypes
    
    count_matrix = []
    cell_labels = []
    
    for i in range(n_celltypes):
        # Each cell type has different expression patterns
        base_expression = np.random.negative_binomial(n=5, p=0.3, size=(cells_per_type, n_genes))
        
        # Add cell-type-specific markers (first 50 genes per type)
        marker_start = i * 50
        marker_end = (i + 1) * 50
        if marker_end <= n_genes:
            base_expression[:, marker_start:marker_end] *= (i + 2)  # Increase expression
        
        count_matrix.append(base_expression)
        cell_labels.extend([f"CellType_{i}"] * cells_per_type)
    
    # Concatenate all cell types
    count_matrix = np.vstack(count_matrix)
    
    # Create gene names
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    cell_names = [f"Cell_{i:04d}" for i in range(count_matrix.shape[0])]
    
    # Create AnnData object
    adata = ad.AnnData(
        X=count_matrix,
        obs=pd.DataFrame({'cell_type': cell_labels}, index=cell_names),
        var=pd.DataFrame({'gene_name': gene_names}, index=gene_names)
    )
    
    # Add basic preprocessing
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # No MT genes in synthetic data
    adata.var['ribo'] = adata.var_names.str.startswith('RPS')  # No ribosomal genes
    
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    return adata

def test_scvi_basic():
    """Test basic scVI functionality"""
    print("=" * 50)
    print("Testing scVI Basic Functionality")
    print("=" * 50)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "scVI",
        "status": "unknown",
        "details": {}
    }
    
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Create test dataset
        adata = create_synthetic_data(n_cells=200, n_genes=1000)
        print(f"Dataset shape: {adata.shape}")
        
        # Basic preprocessing
        adata.layers["counts"] = adata.X.copy()  # Save raw counts
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.raw = adata  # Save normalized data
        
        # Filter genes (keep highly variable)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable]
        
        print(f"After filtering: {adata.shape}")
        
        # Important: copy the AnnData object (scVI requirement)
        adata = adata.copy()
        
        # Setup scVI
        print("\nSetting up scVI model...")
        scvi.settings.seed = 42
        scvi.model.SCVI.setup_anndata(adata, layer="counts")
        
        # Create model
        model = SCVI(adata, n_hidden=128, n_latent=10, n_layers=2)
        print(f"Model created with {sum(p.numel() for p in model.module.parameters())} parameters")
        
        # Train model (quick training for testing)
        print("\nTraining scVI model...")
        model.train(max_epochs=50, batch_size=128, early_stopping=True)
        
        # Get embeddings
        print("\nExtracting embeddings...")
        latent = model.get_latent_representation()
        print(f"Latent representation shape: {latent.shape}")
        
        # Test model capabilities
        print("\nTesting model capabilities...")
        
        # Get normalized expression
        normalized = model.get_normalized_expression(n_samples=1)
        print(f"Normalized expression shape: {normalized.shape}")
        
        # Get reconstruction loss
        test_loss = model.get_reconstruction_error()
        if isinstance(test_loss, dict):
            # Handle case where it returns a dict
            test_loss_value = list(test_loss.values())[0] if test_loss else 0
        else:
            test_loss_value = test_loss
            
        # Convert to scalar if it's a tensor
        if hasattr(test_loss_value, 'mean'):
            test_loss_final = test_loss_value.mean()
        else:
            test_loss_final = test_loss_value
            
        print(f"Reconstruction error: {test_loss_final:.4f}")
        
        # Store results
        results["status"] = "success"
        results["details"] = {
            "dataset_shape": list(adata.shape),
            "model_parameters": sum(p.numel() for p in model.module.parameters()),
            "latent_dim": latent.shape[1],
            "reconstruction_error": float(test_loss_final),
            "device": device,
            "training_epochs": "50 (early stopped)",
        }
        
        print("\n[OK] scVI basic test SUCCESSFUL")
        return True, results
        
    except Exception as e:
        print(f"\n[FAIL] scVI basic test FAILED: {e}")
        results["status"] = "failed"
        results["details"]["error"] = str(e)
        return False, results

def test_scvi_scaling():
    """Test scVI scaling behavior"""
    print("\n" + "=" * 50)
    print("Testing scVI Scaling Behavior")
    print("=" * 50)
    
    scaling_results = []
    cell_counts = [200, 500]  # Start conservative due to 6GB VRAM
    
    for n_cells in cell_counts:
        print(f"\nTesting with {n_cells} cells...")
        
        try:
            # Create dataset
            adata = create_synthetic_data(n_cells=n_cells, n_genes=1000)
            
            # Preprocessing
            adata.layers["counts"] = adata.X.copy()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            
            # Keep more genes for larger datasets to maintain complexity
            sc.pp.highly_variable_genes(adata, n_top_genes=min(800, adata.shape[1]))
            adata = adata[:, adata.var.highly_variable]
            
            # Important: copy the AnnData object (scVI requirement)
            adata = adata.copy()
            
            # Setup and train model
            scvi.model.SCVI.setup_anndata(adata, layer="counts")
            model = SCVI(adata, n_hidden=128, n_latent=10, n_layers=2)
            
            # Train with smaller batch size for larger datasets
            batch_size = max(32, min(128, n_cells // 8))
            model.train(max_epochs=30, batch_size=batch_size, early_stopping=True)
            
            # Extract metrics
            latent = model.get_latent_representation()
            recon_error_raw = model.get_reconstruction_error()
            
            # Handle different return types
            if isinstance(recon_error_raw, dict):
                recon_error = list(recon_error_raw.values())[0] if recon_error_raw else 0
            else:
                recon_error = recon_error_raw
                
            if hasattr(recon_error, 'mean'):
                recon_error = recon_error.mean()
            
            recon_error = float(recon_error)
            
            result = {
                "n_cells": n_cells,
                "n_genes": adata.shape[1],
                "latent_dim": latent.shape[1],
                "reconstruction_error": float(recon_error),
                "batch_size": batch_size,
                "status": "success"
            }
            
            scaling_results.append(result)
            print(f"  [OK] {n_cells} cells: recon_error={recon_error:.4f}")
            
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
    print("[DNA] scVI Multi-Model Validation Test")
    print("Timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    all_results = {
        "model_name": "scVI",
        "test_timestamp": datetime.now().isoformat(),
        "basic_test": {},
        "scaling_test": [],
        "summary": {}
    }
    
    # Basic functionality test
    basic_success, basic_results = test_scvi_basic()
    all_results["basic_test"] = basic_results
    
    if basic_success:
        # Scaling test
        scaling_results = test_scvi_scaling()
        all_results["scaling_test"] = scaling_results
        
        # Analyze scaling
        successful_scales = [r for r in scaling_results if r.get("status") == "success"]
        if len(successful_scales) >= 2:
            errors = [r["reconstruction_error"] for r in successful_scales]
            scaling_change = ((errors[-1] - errors[0]) / errors[0]) * 100
            all_results["summary"]["scaling_behavior"] = f"{scaling_change:.2f}% change"
            print(f"\n[CHART] Scaling analysis: {scaling_change:.2f}% change in reconstruction error")
        else:
            all_results["summary"]["scaling_behavior"] = "insufficient_data"
    
    # Save results
    with open("scvi_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[SAVE] Results saved to scvi_test_results.json")
    
    return all_results

if __name__ == "__main__":
    main()