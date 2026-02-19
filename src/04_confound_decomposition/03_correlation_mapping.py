#!/usr/bin/env python3
"""
Attention-Correlation Mapping Analysis
=======================================
Addresses Round 6 Reviewer Concern 1:
  Quantify how well attention-derived scores map to correlation-based scores.
  Compute R² (not just ρ) between attention and correlation edges.
  Test whether the mapping holds across different conditions.
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
DATA_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/data/raw"
FIG_DIR = os.path.join(PAPER_DIR, "figures")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "attention_correlation_mapping")
os.makedirs(OUT_DIR, exist_ok=True)


def run_mapping():
    import scanpy as sc
    from scipy.stats import rankdata

    print("=" * 60)
    print("ATTENTION-CORRELATION MAPPING ANALYSIS")
    print("=" * 60)

    # Load Geneformer edge data (500 cells — DLPFC brain)
    results = {}

    for n_cells in [200, 500, 1000]:
        edge_path = os.path.join(PAPER_DIR, "multi_model_proper",
                                 f"geneformer_edges_{n_cells}cells.csv")
        if not os.path.exists(edge_path):
            print(f"  Skipping {n_cells} cells (file not found)")
            continue

        print(f"\n--- {n_cells} cells ---")
        edges_df = pd.read_csv(edge_path)
        print(f"  Loaded {len(edges_df)} edges")

        # Get unique genes
        all_genes = sorted(set(edges_df['source'].unique()) | set(edges_df['target'].unique()))
        print(f"  Unique genes: {len(all_genes)}")

        # Build attention score matrix (sparse — only keep top edges)
        attn_scores = {}
        for _, row in edges_df.iterrows():
            key = (row['source'], row['target'])
            attn_scores[key] = row['score']

        # Now compute correlation-based scores on the same dataset
        # Load DLPFC brain data (used for Geneformer analysis)
        # We'll compute Spearman correlations from Tabula Sapiens immune
        # (same tissue basis used in multi-model validation)
        print("  Loading expression data for correlation computation...")
        adata = sc.read_h5ad(os.path.join(DATA_DIR, "tabula_sapiens_immune_subset_20000.h5ad"))

        # Map gene names (handle duplicates)
        if 'feature_name' in adata.var.columns:
            names = adata.var['feature_name'].astype(str).values.copy()
            from collections import Counter
            counts = Counter()
            new_names = []
            for n in names:
                if counts[n] > 0:
                    new_names.append(f"{n}-{counts[n]}")
                else:
                    new_names.append(n)
                counts[n] += 1
            adata.var.index = pd.Index(new_names)

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Restrict to genes in the attention edge set (use only unique var names)
        unique_mask = ~adata.var_names.duplicated(keep='first')
        adata_unique = adata[:, unique_mask].copy()
        common_genes = [g for g in all_genes if g in set(adata_unique.var_names)]
        # Limit to manageable size (top 2000 by attention frequency)
        if len(common_genes) > 2000:
            # Keep genes most frequent in attention edges
            gene_freq = {}
            for (src, tgt) in attn_scores.keys():
                gene_freq[src] = gene_freq.get(src, 0) + 1
                gene_freq[tgt] = gene_freq.get(tgt, 0) + 1
            common_genes = sorted(common_genes, key=lambda g: gene_freq.get(g, 0), reverse=True)[:2000]
        print(f"  Common genes with expression data: {len(common_genes)}")

        if len(common_genes) < 50:
            print("  Too few common genes, skipping")
            continue

        adata_sub = adata_unique[:, common_genes].copy()

        # Subsample cells
        np.random.seed(42)
        n_sub = min(n_cells, adata_sub.n_obs)
        idx = np.random.choice(adata_sub.n_obs, n_sub, replace=False)
        X = adata_sub.X[idx]
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

        # Compute Spearman correlation matrix
        gene_names = list(adata_sub.var_names)
        ranks = np.apply_along_axis(rankdata, 0, X)
        corr_matrix = np.corrcoef(ranks.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        # Match attention edges with correlation scores
        matched_attn = []
        matched_corr = []
        matched_abs_corr = []

        for (src, tgt), attn_score in attn_scores.items():
            if src in gene_to_idx and tgt in gene_to_idx:
                i, j = gene_to_idx[src], gene_to_idx[tgt]
                matched_attn.append(attn_score)
                matched_corr.append(corr_matrix[i, j])
                matched_abs_corr.append(abs(corr_matrix[i, j]))

        matched_attn = np.array(matched_attn)
        matched_corr = np.array(matched_corr)
        matched_abs_corr = np.array(matched_abs_corr)
        print(f"  Matched edges: {len(matched_attn)}")

        if len(matched_attn) < 100:
            print("  Too few matched edges, skipping")
            continue

        # Compute statistics
        rho_signed, p_signed = stats.spearmanr(matched_attn, matched_corr)
        rho_abs, p_abs = stats.spearmanr(matched_attn, matched_abs_corr)

        # R² via OLS
        reg = LinearRegression()
        reg.fit(matched_abs_corr.reshape(-1, 1), matched_attn)
        r2_abs = r2_score(matched_attn, reg.predict(matched_abs_corr.reshape(-1, 1)))

        reg_signed = LinearRegression()
        reg_signed.fit(matched_corr.reshape(-1, 1), matched_attn)
        r2_signed = r2_score(matched_attn, reg_signed.predict(matched_corr.reshape(-1, 1)))

        # Multi-covariate R² (corr + mean expression)
        mean_expr = X.mean(axis=0)
        multi_features = []
        for (src, tgt), attn_score in attn_scores.items():
            if src in gene_to_idx and tgt in gene_to_idx:
                i, j = gene_to_idx[src], gene_to_idx[tgt]
                multi_features.append([
                    abs(corr_matrix[i, j]),
                    corr_matrix[i, j],
                    mean_expr[i],
                    mean_expr[j],
                    mean_expr[i] * mean_expr[j]
                ])
        multi_features = np.array(multi_features)
        reg_multi = LinearRegression()
        reg_multi.fit(multi_features, matched_attn)
        r2_multi = r2_score(matched_attn, reg_multi.predict(multi_features))

        print(f"  Spearman(attn, |corr|): rho={rho_abs:.4f}, p={p_abs:.2e}")
        print(f"  Spearman(attn, signed corr): rho={rho_signed:.4f}, p={p_signed:.2e}")
        print(f"  R²(|corr| → attn): {r2_abs:.4f}")
        print(f"  R²(signed corr → attn): {r2_signed:.4f}")
        print(f"  R²(corr + expr → attn): {r2_multi:.4f}")

        results[n_cells] = {
            'n_matched': len(matched_attn),
            'n_common_genes': len(common_genes),
            'spearman_abs': round(float(rho_abs), 4),
            'spearman_abs_p': float(p_abs),
            'spearman_signed': round(float(rho_signed), 4),
            'spearman_signed_p': float(p_signed),
            'r2_abs_corr': round(float(r2_abs), 4),
            'r2_signed_corr': round(float(r2_signed), 4),
            'r2_multi': round(float(r2_multi), 4),
        }

    # Generate figure
    if results:
        print("\nGenerating figure...")
        n_conditions = len(results)
        fig, axes = plt.subplots(1, min(n_conditions, 3), figsize=(5*min(n_conditions, 3), 5))
        if n_conditions == 1:
            axes = [axes]

        for idx, (n_cells, res) in enumerate(sorted(results.items())):
            if idx >= 3:
                break
            ax = axes[idx]

            # Bar chart of R² values
            labels = ['R²(|corr|)', 'R²(signed)', 'R²(corr+expr)']
            vals = [res['r2_abs_corr'], res['r2_signed_corr'], res['r2_multi']]
            colors = ['#2196F3', '#4CAF50', '#FF9800']
            bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.8)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8, rotation=15)
            ax.set_ylabel('R²')
            ax.set_title(f'{n_cells} cells\n(ρ={res["spearman_abs"]:.3f})')
            ax.set_ylim(0, max(vals) + 0.1)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, "fig_attention_correlation_mapping.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"  Figure saved to {fig_path}")

    # Save summary
    summary = {
        'experiment': 'attention_correlation_mapping',
        'description': 'R² between Geneformer attention-derived and correlation-based edge scores',
        'results_by_cell_count': results,
    }

    with open(os.path.join(OUT_DIR, "mapping_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== EXPERIMENT COMPLETE ===")
    return summary


if __name__ == '__main__':
    run_mapping()
