#!/usr/bin/env python3
"""
SERGIO-based validation for single-cell foundation model interpretability.
This script uses the SERGIO simulator for more realistic GRN simulation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sergio import sergio
import os
from datetime import datetime

def run_sergio_validation():
    """Run validation using SERGIO simulator."""
    print("Running SERGIO validation...")
    
    # SERGIO parameters for realistic simulation
    number_genes = 100
    number_bins = 1  # Single steady state
    number_sc = 2000
    noise_params = 1
    decays = 0.8
    sampling_state = 15
    noise_type = 'dpd'  # Poisson-dropout noise
    
    # Initialize SERGIO
    sim = sergio(
        number_genes=number_genes, 
        number_bins=number_bins, 
        number_sc=number_sc,
        noise_params=noise_params, 
        decays=decays, 
        sampling_state=sampling_state, 
        noise_type=noise_type
    )
    
    print("Building gene regulatory network...")
    # Build the gene regulatory network
    # This creates a sparse network with realistic topology
    sim.build_graph(
        input_file_taregts=None,  # Random network
        input_file_regs=None,     # Random regulators
        shared_coop_state=2,      # Cooperative binding
        tf_mrna_production_rate=1,
        tf_mrna_degradation_rate=0.2
    )
    
    print("Running simulation...")
    # Run the simulation
    sim.simulate()
    
    # Get results
    expr = sim.getExpressions()
    count_matrix = sim.getCount_matrix()
    
    print(f"Generated expression matrix: {expr.shape}")
    print(f"Generated count matrix: {count_matrix.shape}")
    
    # Save data
    output_dir = "D:/openclaw/biodyn-nmi-paper/synthetic_validation"
    
    # Save expression data
    np.save(os.path.join(output_dir, "sergio_expression.npy"), expr)
    np.save(os.path.join(output_dir, "sergio_counts.npy"), count_matrix)
    
    # Save network structure if available
    try:
        adj_matrix = sim.adjacency_matrix
        np.save(os.path.join(output_dir, "sergio_network.npy"), adj_matrix)
        print(f"Saved network with shape: {adj_matrix.shape}")
    except AttributeError:
        print("Adjacency matrix not directly accessible")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Expression heatmap
    ax = axes[0, 0]
    # Sample subset for visualization
    subset_cells = np.random.choice(expr.shape[0], min(100, expr.shape[0]), replace=False)
    subset_genes = np.random.choice(expr.shape[1], min(50, expr.shape[1]), replace=False)
    im = ax.imshow(expr[subset_cells][:, subset_genes], aspect='auto', cmap='viridis')
    ax.set_title('SERGIO Expression Matrix\n(subset for visualization)')
    ax.set_xlabel('Genes')
    ax.set_ylabel('Cells')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Expression distribution
    ax = axes[0, 1]
    ax.hist(expr.flatten(), bins=50, alpha=0.7, density=True)
    ax.set_xlabel('Expression Level')
    ax.set_ylabel('Density')
    ax.set_title('Expression Distribution')
    ax.set_yscale('log')
    
    # Gene expression variance
    ax = axes[1, 0]
    gene_vars = np.var(expr, axis=0)
    ax.hist(gene_vars, bins=30, alpha=0.7)
    ax.set_xlabel('Gene Variance')
    ax.set_ylabel('Count')
    ax.set_title('Gene Expression Variability')
    
    # Count vs expression correlation
    ax = axes[1, 1]
    # Sample genes for correlation
    sample_genes = np.random.choice(expr.shape[1], 10, replace=False)
    for i, gene_idx in enumerate(sample_genes):
        if i < 5:  # Only plot first 5 to avoid clutter
            ax.scatter(count_matrix[:100, gene_idx], expr[:100, gene_idx], 
                      alpha=0.6, s=10, label=f'Gene {gene_idx}')
    ax.set_xlabel('Count Matrix')
    ax.set_ylabel('Expression Matrix')
    ax.set_title('Count vs Expression Relationship')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sergio_validation.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create summary statistics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_genes': number_genes,
        'n_cells': number_sc,
        'expression_mean': float(np.mean(expr)),
        'expression_std': float(np.std(expr)),
        'expression_sparsity': float(np.mean(expr == 0)),
        'count_mean': float(np.mean(count_matrix)),
        'count_std': float(np.std(count_matrix)),
        'count_sparsity': float(np.mean(count_matrix == 0))
    }
    
    # Save summary
    with open(os.path.join(output_dir, 'sergio_summary.txt'), 'w') as f:
        f.write("SERGIO Validation Summary\n")
        f.write("========================\n\n")
        f.write(f"Generated: {summary['timestamp']}\n")
        f.write(f"Genes: {summary['n_genes']}\n")
        f.write(f"Cells: {summary['n_cells']}\n\n")
        f.write("Expression Matrix Statistics:\n")
        f.write(f"  Mean: {summary['expression_mean']:.3f}\n")
        f.write(f"  Std: {summary['expression_std']:.3f}\n")
        f.write(f"  Sparsity: {summary['expression_sparsity']:.3f}\n\n")
        f.write("Count Matrix Statistics:\n")
        f.write(f"  Mean: {summary['count_mean']:.3f}\n")
        f.write(f"  Std: {summary['count_std']:.3f}\n")
        f.write(f"  Sparsity: {summary['count_sparsity']:.3f}\n")
    
    print("SERGIO validation completed!")
    print(f"Results saved to {output_dir}")
    
    return summary

if __name__ == "__main__":
    try:
        summary = run_sergio_validation()
        print("SUCCESS: SERGIO validation completed")
        print(f"Generated {summary['n_genes']} genes × {summary['n_cells']} cells")
        print(f"Expression sparsity: {summary['expression_sparsity']:.2%}")
    except Exception as e:
        print(f"ERROR in SERGIO validation: {e}")
        import traceback
        traceback.print_exc()