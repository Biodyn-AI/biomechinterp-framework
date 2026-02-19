#!/usr/bin/env python3
"""
Comprehensive Analysis Summary for Multi-Model Validation
Analyze results from Geneformer experiments and create comparison with scGPT findings

This script:
1. Loads results from all Geneformer experiments
2. Compares findings with scGPT results from the NMI paper
3. Creates summary statistics and visualizations
4. Generates LaTeX sections for the paper
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("Starting comprehensive analysis of multi-model validation results...")

# Setup paths
results_path = "D:/openclaw/biodyn-nmi-paper/multi_model"
os.makedirs(results_path, exist_ok=True)

def load_experimental_results():
    """Load all experimental results"""
    results = {}
    
    # Load scaling experiment results
    scaling_file = f"{results_path}/geneformer_v1_scaling_results.json"
    if os.path.exists(scaling_file):
        with open(scaling_file, 'r') as f:
            results['scaling'] = json.load(f)
        print("OK Loaded scaling experiment results")
    else:
        print("MISSING Scaling results not found")
    
    # Load cross-context results (if available)
    context_file = f"{results_path}/geneformer_cross_context_results.json"
    if os.path.exists(context_file):
        with open(context_file, 'r') as f:
            results['cross_context'] = json.load(f)
        print("OK Loaded cross-context experiment results")
    else:
        print("MISSING Cross-context results not yet available")
    
    return results

def analyze_scaling_behavior(scaling_results):
    """Analyze scaling behavior results"""
    print("\n=== SCALING BEHAVIOR ANALYSIS ===")
    
    if not scaling_results:
        print("No scaling results to analyze")
        return None
    
    analysis = {}
    
    # Extract data for different cell counts
    cell_counts = []
    edge_counts = []
    max_attentions = []
    mean_attentions = []
    sparsity_values = []
    
    for condition, data in scaling_results.items():
        if 'cells' in condition:
            cell_count = int(condition.split('_')[0])
            cell_counts.append(cell_count)
            edge_counts.append(data['n_edges'])
            max_attentions.append(data['attention_max'])
            mean_attentions.append(data['attention_mean'])
            sparsity_values.append(data['attention_sparsity'])
    
    # Sort by cell count
    sorted_indices = np.argsort(cell_counts)
    cell_counts = [cell_counts[i] for i in sorted_indices]
    edge_counts = [edge_counts[i] for i in sorted_indices]
    max_attentions = [max_attentions[i] for i in sorted_indices]
    mean_attentions = [mean_attentions[i] for i in sorted_indices]
    sparsity_values = [sparsity_values[i] for i in sorted_indices]
    
    analysis['cell_counts'] = cell_counts
    analysis['edge_counts'] = edge_counts
    analysis['max_attentions'] = max_attentions
    analysis['mean_attentions'] = mean_attentions
    analysis['sparsity_values'] = sparsity_values
    
    # Compute trends
    if len(cell_counts) >= 2:
        edge_change = (edge_counts[-1] - edge_counts[0]) / edge_counts[0] * 100
        attention_change = (mean_attentions[-1] - mean_attentions[0]) / mean_attentions[0] * 100
        
        analysis['edge_change_percent'] = edge_change
        analysis['attention_change_percent'] = attention_change
        
        print(f"Cell count range: {min(cell_counts)} - {max(cell_counts)}")
        print(f"Edge count change: {edge_change:.1f}%")
        print(f"Mean attention change: {attention_change:.1f}%")
        
        # Interpret results
        if abs(edge_change) < 5:
            analysis['scaling_interpretation'] = "STABLE - No significant degradation with scale"
        elif edge_change < -5:
            analysis['scaling_interpretation'] = "DEGRADATION - Performance decreases with more cells"
        else:
            analysis['scaling_interpretation'] = "IMPROVEMENT - Performance increases with more cells"
            
        print(f"Scaling behavior: {analysis['scaling_interpretation']}")
    
    return analysis

def analyze_cross_context_consistency(context_results):
    """Analyze cross-context consistency results"""
    print("\n=== CROSS-CONTEXT CONSISTENCY ANALYSIS ===")
    
    if not context_results:
        print("No cross-context results to analyze")
        return None
    
    analysis = {}
    
    # Extract similarity values
    similarities = context_results.get('cross_context_similarities', {})
    
    if similarities:
        sim_values = list(similarities.values())
        analysis['similarities'] = similarities
        analysis['avg_similarity'] = np.mean(sim_values)
        analysis['std_similarity'] = np.std(sim_values)
        analysis['min_similarity'] = np.min(sim_values)
        analysis['max_similarity'] = np.max(sim_values)
        
        print(f"Cross-context similarities: {len(sim_values)} pairs")
        print(f"Average similarity: {analysis['avg_similarity']:.3f}")
        print(f"Similarity range: {analysis['min_similarity']:.3f} - {analysis['max_similarity']:.3f}")
        
        # Interpret consistency
        avg_sim = analysis['avg_similarity']
        if avg_sim > 0.7:
            analysis['consistency_interpretation'] = "HIGH - Very similar attention patterns across contexts"
        elif avg_sim > 0.5:
            analysis['consistency_interpretation'] = "MODERATE - Some context-specific adaptation"
        elif avg_sim > 0.3:
            analysis['consistency_interpretation'] = "LOW - Significant context-specific patterns"
        else:
            analysis['consistency_interpretation'] = "VERY LOW - Highly context-specific attention"
            
        print(f"Consistency: {analysis['consistency_interpretation']}")
    
    # Extract context-specific statistics
    context_stats = context_results.get('context_stats', {})
    if context_stats:
        analysis['context_stats'] = context_stats
        
        # Compare attention strengths across contexts
        strengths = [stats['avg_attention_strength'] for stats in context_stats.values()]
        analysis['context_variation'] = np.std(strengths) / np.mean(strengths) if np.mean(strengths) > 0 else 0
        
        print(f"Context variation (CV): {analysis['context_variation']:.3f}")
    
    return analysis

def compare_with_scgpt_findings():
    """Compare Geneformer findings with scGPT results from NMI paper"""
    print("\n=== COMPARISON WITH scGPT FINDINGS ===")
    
    # Based on the NMI paper's findings with scGPT
    scgpt_findings = {
        'scaling_degradation': True,  # scGPT showed degradation with more cells
        'attention_sparsity': 'LOW',  # scGPT had sparse attention patterns
        'context_consistency': 'MODERATE',  # scGPT showed some context specificity
        'grn_recovery_quality': 'GOOD'  # scGPT recovered meaningful GRN edges
    }
    
    # This would be filled based on our experimental results
    geneformer_findings = {
        'scaling_degradation': None,  # To be determined from our results
        'attention_sparsity': None,
        'context_consistency': None,
        'grn_recovery_quality': None
    }
    
    comparison = {
        'scgpt': scgpt_findings,
        'geneformer': geneformer_findings,
        'similarities': [],
        'differences': []
    }
    
    print("scGPT findings from NMI paper:")
    for key, value in scgpt_findings.items():
        print(f"  {key}: {value}")
    
    print("\nGeneformer findings (to be updated with actual results):")
    print("  Will be filled in after experiments complete")
    
    return comparison

def create_visualization_plots(analysis_results):
    """Create visualization plots for the results"""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Geneformer Multi-Model Validation Results', fontsize=16)
    
    # Scaling behavior plot
    if 'scaling' in analysis_results and analysis_results['scaling']:
        scaling = analysis_results['scaling']
        if scaling and 'cell_counts' in scaling:
            ax = axes[0, 0]
            ax.plot(scaling['cell_counts'], scaling['edge_counts'], 'b-o', label='Edge Count')
            ax.set_xlabel('Number of Cells')
            ax.set_ylabel('Number of GRN Edges')
            ax.set_title('Scaling Behavior: GRN Edge Recovery')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Attention strength plot
        if scaling and 'cell_counts' in scaling:
            ax = axes[0, 1]
            ax.plot(scaling['cell_counts'], scaling['mean_attentions'], 'r-s', label='Mean Attention')
            ax.set_xlabel('Number of Cells')
            ax.set_ylabel('Mean Attention Weight')
            ax.set_title('Scaling: Attention Strength')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Cross-context similarity heatmap
    if 'cross_context' in analysis_results and analysis_results['cross_context']:
        context = analysis_results['cross_context']
        if context and 'similarities' in context:
            ax = axes[1, 0]
            similarities = context['similarities']
            
            # Create similarity matrix for heatmap
            contexts = set()
            for pair in similarities.keys():
                ctx1, ctx2 = pair.split('_vs_')
                contexts.add(ctx1)
                contexts.add(ctx2)
            
            contexts = sorted(list(contexts))
            n_ctx = len(contexts)
            sim_matrix = np.eye(n_ctx)
            
            for pair, sim in similarities.items():
                ctx1, ctx2 = pair.split('_vs_')
                i1, i2 = contexts.index(ctx1), contexts.index(ctx2)
                sim_matrix[i1, i2] = sim_matrix[i2, i1] = sim
            
            im = ax.imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1)
            ax.set_xticks(range(n_ctx))
            ax.set_yticks(range(n_ctx))
            ax.set_xticklabels(contexts)
            ax.set_yticklabels(contexts)
            ax.set_title('Cross-Context Attention Similarity')
            plt.colorbar(im, ax=ax)
    
    # Summary statistics
    ax = axes[1, 1]
    ax.text(0.1, 0.8, 'Experiment Summary:', fontsize=12, fontweight='bold')
    
    y_pos = 0.7
    if 'scaling' in analysis_results and analysis_results['scaling']:
        scaling = analysis_results['scaling']
        if scaling:
            if 'scaling_interpretation' in scaling:
                ax.text(0.1, y_pos, f"Scaling: {scaling['scaling_interpretation']}", fontsize=10)
                y_pos -= 0.1
    
    if 'cross_context' in analysis_results and analysis_results['cross_context']:
        context = analysis_results['cross_context']
        if context and 'consistency_interpretation' in context:
            ax.text(0.1, y_pos, f"Consistency: {context['consistency_interpretation']}", fontsize=10)
            y_pos -= 0.1
    
    ax.text(0.1, y_pos-0.1, f"Model: Geneformer V1-10M", fontsize=10)
    ax.text(0.1, y_pos-0.2, f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{results_path}/geneformer_analysis_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {results_path}/geneformer_analysis_summary.png")

def generate_latex_sections(analysis_results, comparison):
    """Generate LaTeX sections for the paper"""
    print("\n=== GENERATING LATEX SECTIONS ===")
    
    latex_content = r"""
% Multi-Model Validation Results - Geneformer Analysis

\subsection{Multi-Model Validation with Geneformer}

To validate the robustness of findings from the original scGPT analysis, we conducted parallel experiments using Geneformer~\cite{theodoris2023geneformer}, another state-of-the-art transformer model for single-cell genomics. Geneformer employs a different tokenization strategy based on gene expression ranks rather than raw expression values, providing an important architectural comparison.

\subsubsection{Experimental Setup}

We utilized the Geneformer V1-10M model (10 million parameters) and replicated three key experiments from our scGPT analysis:
\begin{enumerate}
\item \textbf{Scaling behavior analysis}: Testing GRN recovery performance with varying cell numbers (200 vs 500 cells)
\item \textbf{Attention pattern extraction}: Computing gene regulatory networks from transformer attention weights
\item \textbf{Cross-context consistency}: Evaluating attention pattern stability across different cellular contexts
\end{enumerate}

"""
    
    # Add scaling results section
    if 'scaling' in analysis_results and analysis_results['scaling']:
        scaling = analysis_results['scaling']
        latex_content += r"""
\subsubsection{Scaling Behavior Results}

"""
        if 'scaling_interpretation' in scaling:
            if 'STABLE' in scaling['scaling_interpretation']:
                latex_content += r"""Our analysis revealed that Geneformer exhibits stable performance across different cell numbers, with minimal degradation in GRN recovery quality. This contrasts with scGPT's observed performance decline at larger scales, suggesting that Geneformer's rank-based tokenization may provide better scalability for gene network inference tasks.
"""
            elif 'DEGRADATION' in scaling['scaling_interpretation']:
                latex_content += r"""Similar to scGPT, Geneformer showed performance degradation when analyzing larger cell populations. This suggests that the scalability challenges observed in the original study may be inherent to transformer-based approaches for gene regulatory network inference, rather than model-specific limitations.
"""
    
    # Add cross-context results
    if 'cross_context' in analysis_results and analysis_results['cross_context']:
        context = analysis_results['cross_context']
        latex_content += r"""
\subsubsection{Cross-Context Consistency}

"""
        if 'consistency_interpretation' in context:
            if 'HIGH' in context['consistency_interpretation']:
                latex_content += r"""Geneformer demonstrated high consistency in attention patterns across different cellular contexts, with an average cross-context similarity of %.3f. This indicates robust gene relationship detection that generalizes well across tissue types, potentially offering advantages over scGPT for multi-tissue studies.
""" % (context.get('avg_similarity', 0))
            elif 'LOW' in context['consistency_interpretation']:
                latex_content += r"""Our analysis revealed significant context-specific attention patterns in Geneformer, with lower cross-context similarity (%.3f average) compared to expected baseline consistency. This context-sensitivity may reflect biologically meaningful tissue-specific regulatory programs, but could also indicate model limitations in capturing universal gene regulatory principles.
""" % (context.get('avg_similarity', 0))
    
    # Add comparison section
    latex_content += r"""
\subsubsection{Comparison with scGPT Findings}

\begin{table}[h]
\centering
\caption{Comparison of scGPT and Geneformer performance characteristics}
\label{tab:model_comparison}
\begin{tabular}{lcc}
\hline
\textbf{Characteristic} & \textbf{scGPT} & \textbf{Geneformer} \\
\hline
Tokenization Strategy & Raw Expression & Expression Ranks \\
Scaling Behavior & Degradation & [TO BE UPDATED] \\
Context Consistency & Moderate & [TO BE UPDATED] \\
Attention Sparsity & Low & [TO BE UPDATED] \\
GRN Recovery Quality & Good & [TO BE UPDATED] \\
\hline
\end{tabular}
\end{table}

The multi-model validation reveals important insights about the generalizability of transformer-based approaches for gene regulatory network inference. While both models show promise for single-cell analysis, their different architectural choices lead to distinct performance characteristics across various experimental conditions.

\subsubsection{Implications for Single-Cell Foundation Models}

Our multi-model comparison demonstrates that [TO BE COMPLETED BASED ON FINAL RESULTS]:
\begin{itemize}
\item The choice of tokenization strategy significantly impacts model scalability
\item Context-specific attention patterns may reflect biological reality rather than model artifacts
\item Cross-validation across multiple foundation models is essential for robust scientific conclusions
\end{itemize}

"""
    
    # Save LaTeX content
    latex_file = f"{results_path}/multi_model_validation_sections.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX sections saved to {latex_file}")
    
    return latex_content

def main():
    """Main analysis function"""
    print("Multi-Model Validation Analysis")
    print("=" * 50)
    
    # Load results
    results = load_experimental_results()
    
    # Analyze each experiment type
    analysis_results = {}
    
    if 'scaling' in results:
        analysis_results['scaling'] = analyze_scaling_behavior(results['scaling'])
    
    if 'cross_context' in results:
        analysis_results['cross_context'] = analyze_cross_context_consistency(results['cross_context'])
    
    # Compare with scGPT
    comparison = compare_with_scgpt_findings()
    
    # Create visualizations
    if analysis_results:
        create_visualization_plots(analysis_results)
    
    # Generate LaTeX sections
    generate_latex_sections(analysis_results, comparison)
    
    # Save comprehensive analysis
    final_analysis = {
        'analysis_results': analysis_results,
        'comparison_with_scgpt': comparison,
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_used': 'Geneformer-V1-10M',
        'experimental_status': 'Completed' if analysis_results else 'Partial'
    }
    
    with open(f"{results_path}/comprehensive_analysis.json", 'w') as f:
        json.dump(final_analysis, f, indent=2, default=str)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved to: {results_path}")
    print(f"Generated files:")
    print(f"  - comprehensive_analysis.json")
    print(f"  - multi_model_validation_sections.tex")
    if analysis_results:
        print(f"  - geneformer_analysis_summary.png")

if __name__ == "__main__":
    main()