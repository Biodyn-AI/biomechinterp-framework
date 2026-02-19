#!/usr/bin/env python3
"""Test dependencies"""

try:
    import numpy as np
    print("NumPy OK:", np.__version__)
    
    import pandas as pd
    print("Pandas OK:", pd.__version__)
    
    import scanpy as sc
    print("Scanpy OK:", sc.__version__)
    
    import sklearn
    print("Sklearn OK:", sklearn.__version__)
    
    import matplotlib.pyplot as plt
    print("Matplotlib OK")
    
    # Test data path
    import os
    data_path = "/mnt/d/openclaw/mechinterp-bio/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad"
    print(f"Data file exists: {os.path.exists(data_path)}")
    if os.path.exists(data_path):
        print(f"Data file size: {os.path.getsize(data_path) / 1024**3:.1f} GB")
    
    print("All dependencies check out!")
    
except ImportError as e:
    print(f"Missing dependency: {e}")
except Exception as e:
    print(f"Error: {e}")