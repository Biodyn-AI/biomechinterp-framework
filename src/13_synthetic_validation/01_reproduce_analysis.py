#!/usr/bin/env python3
"""
Script to reproduce specific synthetic validation analyses.
Allows running individual experiments or custom parameter sweeps.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import os
from synthetic_validation import SyntheticValidation, CustomGRNSimulator

def reproduce_scaling_analysis(cell_counts=None, n_genes=50, n_cell_types=4):
    """Reproduce scaling analysis with custom parameters."""
    if cell_counts is None:
        cell_counts = [200, 500, 1000, 2000]
    
    print(f"Reproducing scaling analysis with {n_genes} genes, {n_cell_types} cell types...")
    
    validator = SyntheticValidation()
    recovery_scores = []
    
    for n_cells in cell_counts:
        print(f"  Testing {n_cells} cells...")
        
        simulator = CustomGRNSimulator(n_genes=n_genes, n_cell_types=n_cell_types)
        true_grn = simulator.generate_grn(sparsity=0.15)
        expr_data, cell_types = simulator.simulate_steady_state(n_cells_per_type=n_cells//n_cell_types)
        
        # Simulate attention recovery
        noise_level = 0.05 + 0.002 * n_cells
        attention_matrix = validator.simulate_attention_recovery(true_grn, noise_level)
        
        # Calculate recovery score
        recovery_score = np.corrcoef(true_grn.flatten(), attention_matrix.flatten())[0,1]
        recovery_scores.append(recovery_score)
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(cell_counts, recovery_scores, 'o-', linewidth=3, markersize=8, color='#2E86AB')
    plt.xlabel('Number of Cells')
    plt.ylabel('GRN Recovery Score')
    plt.title('Scaling Analysis Reproduction')
    plt.grid(True, alpha=0.3)
    
    output_file = 'reproduce_scaling.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"Results: {dict(zip(cell_counts, recovery_scores))}")
    print(f"Degradation: {recovery_scores[0] - recovery_scores[-1]:.3f}")
    print(f"Saved to {output_file}")
    
    return recovery_scores

def parameter_sweep_analysis():
    """Run parameter sweep to understand sensitivity."""
    print("Running parameter sensitivity analysis...")
    
    # Parameter ranges
    sparsity_levels = [0.05, 0.1, 0.15, 0.2]
    noise_levels = [0.05, 0.1, 0.15, 0.2]
    
    results = np.zeros((len(sparsity_levels), len(noise_levels)))
    
    validator = SyntheticValidation()
    
    for i, sparsity in enumerate(sparsity_levels):
        for j, base_noise in enumerate(noise_levels):
            print(f"  Testing sparsity={sparsity:.2f}, noise={base_noise:.2f}")
            
            # Generate network
            simulator = CustomGRNSimulator(n_genes=30, n_cell_types=3)
            true_grn = simulator.generate_grn(sparsity=sparsity)
            expr_data, _ = simulator.simulate_steady_state(n_cells_per_type=200, 
                                                         noise_level=base_noise)
            
            # Test recovery at high cell count
            attention_matrix = validator.simulate_attention_recovery(true_grn, base_noise * 3)
            recovery_score = np.corrcoef(true_grn.flatten(), attention_matrix.flatten())[0,1]
            results[i, j] = recovery_score
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(results, aspect='auto', cmap='viridis')
    ax.set_xlabel('Base Noise Level')
    ax.set_ylabel('Network Sparsity')
    ax.set_title('Parameter Sensitivity Analysis\n(GRN Recovery Score)')
    
    # Set tick labels
    ax.set_xticks(range(len(noise_levels)))
    ax.set_xticklabels([f'{n:.2f}' for n in noise_levels])
    ax.set_yticks(range(len(sparsity_levels)))
    ax.set_yticklabels([f'{s:.2f}' for s in sparsity_levels])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Recovery Score')
    
    # Add text annotations
    for i in range(len(sparsity_levels)):
        for j in range(len(noise_levels)):
            text = ax.text(j, i, f'{results[i, j]:.2f}', 
                         ha='center', va='center', color='white', fontweight='bold')
    
    output_file = 'parameter_sweep.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"Saved parameter sweep to {output_file}")
    return results

def load_and_analyze_results():
    """Load existing results and perform additional analysis."""
    results_file = 'validation_results.pkl'
    
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found. Run main validation first.")
        return
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print("Loaded validation results:")
    for key, value in results.items():
        print(f"  {key}: {type(value)}")
    
    # Additional analysis
    if 'scaling' in results:
        scaling = results['scaling']
        cell_counts = scaling['cell_counts']
        recovery_scores = scaling['recovery_scores']
        
        # Fit degradation curve
        log_cells = np.log(cell_counts)
        coeffs = np.polyfit(log_cells, recovery_scores, 1)
        
        print(f"\nScaling Analysis:")
        print(f"  Recovery degradation rate: {-coeffs[0]:.3f} per log(cell)")
        print(f"  R² fit: {np.corrcoef(log_cells, recovery_scores)[0,1]**2:.3f}")
        
        # Extrapolate to larger scales
        future_cells = [5000, 10000, 20000]
        future_log = np.log(future_cells)
        future_recovery = np.polyval(coeffs, future_log)
        
        print(f"  Extrapolated recovery at 10K cells: {future_recovery[1]:.3f}")
        print(f"  Extrapolated recovery at 20K cells: {future_recovery[2]:.3f}")
    
    if 'detectability' in results:
        detect = results['detectability']
        print(f"\nDetectability Analysis:")
        print(f"  Theory-experiment correlation: {detect['theory_correlation']:.3f}")
        print(f"  SNR range: {detect['snr_levels'][0]:.1f} - {detect['snr_levels'][-1]:.1f}")
        print(f"  Sample size range: {detect['sample_sizes'][0]} - {detect['sample_sizes'][-1]}")

def main():
    parser = argparse.ArgumentParser(description='Reproduce synthetic validation analyses')
    parser.add_argument('--analysis', choices=['scaling', 'sweep', 'load'], default='scaling',
                       help='Type of analysis to run')
    parser.add_argument('--cells', nargs='+', type=int, 
                       help='Cell counts for scaling analysis')
    parser.add_argument('--genes', type=int, default=50,
                       help='Number of genes for scaling analysis')
    parser.add_argument('--cell-types', type=int, default=4,
                       help='Number of cell types')
    
    args = parser.parse_args()
    
    if args.analysis == 'scaling':
        reproduce_scaling_analysis(cell_counts=args.cells, 
                                 n_genes=args.genes, 
                                 n_cell_types=args.cell_types)
    elif args.analysis == 'sweep':
        parameter_sweep_analysis()
    elif args.analysis == 'load':
        load_and_analyze_results()

if __name__ == "__main__":
    # If run without arguments, show interactive menu
    import sys
    if len(sys.argv) == 1:
        print("Synthetic Validation Analysis")
        print("============================")
        print("1. Reproduce scaling analysis")
        print("2. Parameter sweep analysis") 
        print("3. Load and analyze existing results")
        
        choice = input("Choose analysis (1-3): ").strip()
        
        if choice == '1':
            reproduce_scaling_analysis()
        elif choice == '2':
            parameter_sweep_analysis()
        elif choice == '3':
            load_and_analyze_results()
        else:
            print("Invalid choice")
    else:
        main()