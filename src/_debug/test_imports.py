#!/usr/bin/env python3
"""Test script to verify all imports work."""

print("Testing imports...")

try:
    import numpy as np
    print("OK numpy")
except Exception as e:
    print(f"FAIL numpy: {e}")

try:
    import pandas as pd
    print("OK pandas")
except Exception as e:
    print(f"FAIL pandas: {e}")

try:
    import scanpy as sc
    print("OK scanpy")
except Exception as e:
    print(f"FAIL scanpy: {e}")

try:
    from arboreto.algo import genie3, grnboost2
    print("OK arboreto")
except Exception as e:
    print(f"FAIL arboreto: {e}")

try:
    import sklearn
    print("OK sklearn")
except Exception as e:
    print(f"FAIL sklearn: {e}")

try:
    import matplotlib.pyplot as plt
    print("OK matplotlib")
except Exception as e:
    print(f"FAIL matplotlib: {e}")

print("All imports tested!")

# Test data loading
print("Testing data loading...")
data_path = r"D:\openclaw\intelligence-augmentation\data\brain_scrna\DLPFC_11k.h5ad"

try:
    print(f"Loading: {data_path}")
    adata = sc.read_h5ad(data_path)
    print(f"Data shape: {adata.shape}")
    print("OK Data loaded successfully")
    
    # Test sampling
    if adata.shape[0] > 10:
        sample_idx = np.random.choice(adata.shape[0], 10, replace=False)
        sampled = adata[sample_idx, :].copy()
        print(f"Sample shape: {sampled.shape}")
        print("OK Sampling works")
    
except Exception as e:
    print(f"FAIL Data loading failed: {e}")

print("Test completed!")