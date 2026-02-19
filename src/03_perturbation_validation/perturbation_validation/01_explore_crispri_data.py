#!/usr/bin/env python3
"""
Explore the CRISPRi data (Shifrut et al.) to understand structure and design perturbation validation experiment
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data paths
DATA_DIR = Path("/mnt/d/openclaw/mechinterp-bio/biodyn-work/single_cell_mechinterp/data/perturb")
SHIFRUT_PATH = DATA_DIR / "shifrut" / "perturb_processed_symbols.h5ad"
ADAMSON_PATH = DATA_DIR / "adamson" / "perturb_processed_symbols.h5ad"

def load_and_explore_crispri_data():
    """Load and explore the CRISPRi data structure"""
    logger.info("Loading CRISPRi data...")
    
    # Load Shifrut data (CRISPRi)
    if SHIFRUT_PATH.exists():
        logger.info(f"Loading Shifrut CRISPRi data from {SHIFRUT_PATH}")
        adata_crispri = sc.read_h5ad(SHIFRUT_PATH)
        
        logger.info(f"CRISPRi data shape: {adata_crispri.shape}")
        logger.info(f"Observations (cells): {adata_crispri.n_obs}")
        logger.info(f"Variables (genes): {adata_crispri.n_vars}")
        
        # Examine metadata
        logger.info("Observation metadata keys:")
        for key in adata_crispri.obs.columns:
            logger.info(f"  {key}: {adata_crispri.obs[key].dtype}")
            if adata_crispri.obs[key].dtype == 'object' or adata_crispri.obs[key].dtype.name == 'category':
                unique_vals = adata_crispri.obs[key].unique()
                logger.info(f"    Unique values ({len(unique_vals)}): {unique_vals[:10]}")
        
        logger.info("Variable metadata keys:")
        for key in adata_crispri.var.columns:
            logger.info(f"  {key}: {adata_crispri.var[key].dtype}")
        
        # Look for perturbation information
        perturbation_cols = [col for col in adata_crispri.obs.columns 
                           if any(term in col.lower() for term in ['perturb', 'target', 'guide', 'sgRNA', 'condition', 'treatment'])]
        logger.info(f"Potential perturbation columns: {perturbation_cols}")
        
        if perturbation_cols:
            for col in perturbation_cols[:3]:  # Check first 3 perturbation columns
                unique_vals = adata_crispri.obs[col].unique()
                logger.info(f"{col} unique values ({len(unique_vals)}): {unique_vals[:20]}")
        
        return adata_crispri
    else:
        logger.error(f"CRISPRi data not found at {SHIFRUT_PATH}")
        return None

def load_and_explore_adamson_data():
    """Load and explore Adamson data for comparison"""
    if ADAMSON_PATH.exists():
        logger.info(f"Loading Adamson data from {ADAMSON_PATH}")
        adata_adamson = sc.read_h5ad(ADAMSON_PATH)
        
        logger.info(f"Adamson data shape: {adata_adamson.shape}")
        
        # Look for perturbation information
        perturbation_cols = [col for col in adata_adamson.obs.columns 
                           if any(term in col.lower() for term in ['perturb', 'target', 'guide', 'sgRNA', 'condition', 'treatment'])]
        logger.info(f"Adamson perturbation columns: {perturbation_cols}")
        
        if perturbation_cols:
            for col in perturbation_cols[:3]:
                unique_vals = adata_adamson.obs[col].unique()
                logger.info(f"{col} unique values ({len(unique_vals)}): {unique_vals[:20]}")
        
        return adata_adamson
    else:
        logger.error(f"Adamson data not found at {ADAMSON_PATH}")
        return None

def analyze_perturbation_effects(adata):
    """Analyze perturbation effects to understand the experimental design"""
    logger.info("Analyzing perturbation effects...")
    
    # Find the main perturbation column
    perturbation_cols = [col for col in adata.obs.columns 
                        if any(term in col.lower() for term in ['perturb', 'target', 'guide'])]
    
    if not perturbation_cols:
        logger.warning("No perturbation columns found!")
        return None
    
    # Use the first perturbation column
    perturb_col = perturbation_cols[0]
    logger.info(f"Using perturbation column: {perturb_col}")
    
    # Get perturbation targets
    perturb_targets = adata.obs[perturb_col].unique()
    logger.info(f"Found {len(perturb_targets)} perturbation targets: {perturb_targets[:10]}")
    
    # Count cells per perturbation
    perturb_counts = adata.obs[perturb_col].value_counts()
    logger.info(f"Cells per perturbation (top 10):")
    logger.info(perturb_counts.head(10))
    
    # Look for control conditions
    control_terms = ['control', 'ctrl', 'non-targeting', 'nt', 'scramble', 'negative']
    potential_controls = []
    for target in perturb_targets:
        if any(term in str(target).lower() for term in control_terms):
            potential_controls.append(target)
    
    logger.info(f"Potential control conditions: {potential_controls}")
    
    return {
        'perturb_col': perturb_col,
        'perturb_targets': perturb_targets,
        'perturb_counts': perturb_counts,
        'potential_controls': potential_controls
    }

def main():
    logger.info("=== CRISPRi Data Exploration ===")
    
    # Load and explore CRISPRi data
    crispri_data = load_and_explore_crispri_data()
    if crispri_data is not None:
        crispri_info = analyze_perturbation_effects(crispri_data)
        
        # Save basic info for the experiment
        results = {
            'crispri_shape': crispri_data.shape,
            'crispri_info': crispri_info
        }
        
        # Also load Adamson for comparison
        adamson_data = load_and_explore_adamson_data()
        if adamson_data is not None:
            adamson_info = analyze_perturbation_effects(adamson_data)
            results['adamson_shape'] = adamson_data.shape
            results['adamson_info'] = adamson_info
        
        logger.info("=== Exploration Complete ===")
        return results
    
    else:
        logger.error("Could not load CRISPRi data")
        return None

if __name__ == "__main__":
    results = main()