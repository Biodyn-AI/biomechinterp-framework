#!/usr/bin/env python3
"""
Perturbation-First Validation
===============================
Addresses Major Concern 4 / Stress Test 4:
For each perturbed TF in Perturb-seq (Replogle CRISPRi K562), test whether
correlation-based edge scores (proxy for attention) prioritize its true
DE targets. Report enrichment curves and compare to random expectation.

Uses the Replogle CRISPRi dataset available at:
/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/crispri_validation/data/replogle_concat.h5ad
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/crispri_validation/data/replogle_concat.h5ad"
REF_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks"
OUT_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/perturbation_first_results"
FIG_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def run_perturbation_validation():
    import scanpy as sc
    import anndata

    print("=" * 60)
    print("PERTURBATION-FIRST VALIDATION")
    print("=" * 60)

    # Check data availability
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Replogle dataset not found at {DATA_PATH}")
        print("Creating synthetic perturbation validation instead...")
        return run_synthetic_perturbation_proxy()

    print(f"\nLoading Replogle CRISPRi data from {DATA_PATH}...")
    print("  (This is a large file, may take a few minutes...)")

    try:
        adata = sc.read_h5ad(DATA_PATH, backed='r')
        print(f"  Shape: {adata.shape}")
        print(f"  obs columns: {list(adata.obs.columns[:10])}")
        print(f"  var columns: {list(adata.var.columns[:10])}")

        # Find the perturbation column
        pert_col = None
        for col in adata.obs.columns:
            if 'gene' in col.lower() or 'perturbation' in col.lower() or 'target' in col.lower():
                unique_vals = adata.obs[col].nunique()
                print(f"  Candidate pert column '{col}': {unique_vals} unique values")
                if unique_vals > 2 and unique_vals < adata.n_obs:
                    pert_col = col
                    break

        if pert_col is None:
            print("  Could not identify perturbation column, falling back to synthetic")
            return run_synthetic_perturbation_proxy()

        print(f"  Using perturbation column: '{pert_col}'")
        pert_genes = adata.obs[pert_col].value_counts()
        print(f"  Perturbed genes: {len(pert_genes)}")
        print(f"  Top 10: {dict(pert_genes.head(10))}")

        # Find control cells
        control_labels = ['non-targeting', 'control', 'NT', 'NTC', 'ctrl']
        control_mask = adata.obs[pert_col].isin(control_labels)
        if control_mask.sum() == 0:
            # Try partial match
            for label in control_labels:
                control_mask = adata.obs[pert_col].str.contains(label, case=False, na=False)
                if control_mask.sum() > 0:
                    break

        n_control = control_mask.sum()
        print(f"  Control cells: {n_control}")

        if n_control < 50:
            print("  Insufficient control cells, falling back to synthetic")
            return run_synthetic_perturbation_proxy()

        # Load into memory (subsample if needed)
        max_control = min(500, n_control)
        np.random.seed(42)
        control_idx = np.where(control_mask)[0]
        if len(control_idx) > max_control:
            control_idx = np.random.choice(control_idx, max_control, replace=False)

        # Get gene names
        if 'feature_name' in adata.var.columns:
            gene_names = list(adata.var['feature_name'])
        else:
            gene_names = list(adata.var_names)

        # Select top variable genes for tractability
        control_data = adata[control_idx].to_memory()
        X_ctrl = control_data.X
        if hasattr(X_ctrl, 'toarray'):
            X_ctrl = X_ctrl.toarray()
        X_ctrl = np.asarray(X_ctrl, dtype=np.float32)

        # Gene variance
        gene_var = X_ctrl.var(axis=0)
        top_genes_idx = np.argsort(gene_var)[-1000:]
        gene_names_sub = [gene_names[i] for i in top_genes_idx]
        X_ctrl_sub = X_ctrl[:, top_genes_idx]

        print(f"  Using {len(gene_names_sub)} most variable genes from {max_control} control cells")

        # Compute edge scores (Spearman correlation on control cells)
        from scipy.stats import rankdata
        ranks = np.apply_along_axis(rankdata, 0, X_ctrl_sub)
        corr = np.corrcoef(ranks.T)

        # For each perturbed gene, compute DE targets
        perturbed_genes = [g for g in pert_genes.index if g not in control_labels
                          and pert_genes[g] >= 20 and g.upper() in [gn.upper() for gn in gene_names_sub]]

        print(f"  Evaluable perturbed genes: {len(perturbed_genes)}")

        results = []
        for pert_gene in perturbed_genes[:50]:  # Limit to 50 for tractability
            pert_mask = adata.obs[pert_col] == pert_gene
            n_pert = pert_mask.sum()
            if n_pert < 20:
                continue

            pert_idx = np.where(pert_mask)[0]
            if len(pert_idx) > 200:
                pert_idx = np.random.choice(pert_idx, 200, replace=False)

            pert_data = adata[pert_idx].to_memory()
            X_pert = pert_data.X
            if hasattr(X_pert, 'toarray'):
                X_pert = X_pert.toarray()
            X_pert = np.asarray(X_pert, dtype=np.float32)[:, top_genes_idx]

            # Compute DE (simple t-test)
            from scipy.stats import ttest_ind
            pvals = []
            lfc = []
            for g in range(len(gene_names_sub)):
                t, p = ttest_ind(X_pert[:, g], X_ctrl_sub[:, g])
                mean_diff = X_pert[:, g].mean() - X_ctrl_sub[:, g].mean()
                pvals.append(p)
                lfc.append(mean_diff)

            pvals = np.array(pvals)
            lfc = np.array(lfc)

            # Define DE targets (FDR < 0.05, |LFC| > 0.1)
            from statsmodels.stats.multitest import multipletests
            _, adj_pvals, _, _ = multipletests(pvals, method='fdr_bh', alpha=0.05)
            de_mask = (adj_pvals < 0.05) & (np.abs(lfc) > 0.1)
            n_de = de_mask.sum()

            if n_de < 5:
                continue

            # Get edge scores for this TF's targets
            pert_gene_upper = pert_gene.upper()
            gene_names_upper = [g.upper() for g in gene_names_sub]
            if pert_gene_upper not in gene_names_upper:
                continue
            tf_idx = gene_names_upper.index(pert_gene_upper)

            # Edge scores from TF to all other genes
            edge_scores = np.abs(corr[tf_idx, :])
            edge_scores[tf_idx] = 0  # Exclude self

            # AUROC: do high edge scores predict DE targets?
            auroc = roc_auc_score(de_mask.astype(int), edge_scores)

            # Top-K enrichment
            k = n_de
            top_k_idx = np.argsort(edge_scores)[-k:]
            precision_at_k = de_mask[top_k_idx].mean()
            expected_precision = n_de / len(gene_names_sub)
            fold_enrichment = precision_at_k / expected_precision if expected_precision > 0 else 0

            results.append({
                'perturbed_gene': pert_gene,
                'n_pert_cells': n_pert,
                'n_de_targets': n_de,
                'auroc': auroc,
                'precision_at_k': precision_at_k,
                'expected_precision': expected_precision,
                'fold_enrichment': fold_enrichment
            })
            print(f"    {pert_gene}: {n_de} DE targets, AUROC={auroc:.3f}, "
                  f"fold_enrich={fold_enrichment:.2f}")

        if len(results) == 0:
            print("  No evaluable perturbations found, falling back to synthetic")
            return run_synthetic_perturbation_proxy()

        df = pd.DataFrame(results)
        return finalize_results(df, 'replogle_crispri')

    except Exception as e:
        print(f"  Error loading Replogle data: {e}")
        print("  Falling back to synthetic perturbation validation")
        return run_synthetic_perturbation_proxy()


def run_synthetic_perturbation_proxy():
    """
    Synthetic perturbation-first validation as proxy.
    Simulates perturbation outcomes and tests whether edge scores
    (correlation-based, proxy for attention) predict perturbation targets.
    """
    print("\n--- Running synthetic perturbation validation ---")

    np.random.seed(42)
    n_genes = 200
    n_tfs = 20
    n_cells = 500
    n_pert_genes = 15

    # Generate base regulatory network
    A = np.zeros((n_genes, n_genes))
    true_targets = {}  # tf_idx -> set of target indices
    for tf in range(n_tfs):
        n_targets = np.random.randint(5, 20)
        targets = np.random.choice(range(n_tfs, n_genes), n_targets, replace=False)
        for tgt in targets:
            A[tf, tgt] = np.random.choice([-1, 1]) * np.random.uniform(0.3, 0.8)
        true_targets[tf] = set(targets)

    # Generate control expression
    X_ctrl = np.random.randn(n_cells, n_genes) * 0.3
    for tf in range(n_tfs):
        for tgt in true_targets[tf]:
            # Add correlated expression
            shared = np.random.randn(n_cells) * 0.5
            X_ctrl[:, tf] += shared
            X_ctrl[:, tgt] += A[tf, tgt] * shared

    # Compute edge scores from control cells (Spearman correlation)
    from scipy.stats import rankdata
    ranks = np.apply_along_axis(rankdata, 0, X_ctrl)
    corr = np.corrcoef(ranks.T)

    # Simulate perturbations
    results = []
    for tf in range(n_pert_genes):
        # Perturbed expression: knock down TF, observe downstream effects
        X_pert = X_ctrl.copy()
        X_pert[:, tf] *= 0.1  # Knock down TF
        for tgt in true_targets[tf]:
            X_pert[:, tgt] += A[tf, tgt] * np.random.randn(n_cells) * 0.3  # Perturbation effect

        # Compute DE (true targets should show differential expression)
        from scipy.stats import ttest_ind
        pvals = []
        for g in range(n_genes):
            _, p = ttest_ind(X_pert[:, g], X_ctrl[:, g])
            pvals.append(p)
        pvals = np.array(pvals)
        from statsmodels.stats.multitest import multipletests
        _, adj_pvals, _, _ = multipletests(pvals, method='fdr_bh')
        de_mask = adj_pvals < 0.05
        de_mask[tf] = False  # Exclude self

        n_de = de_mask.sum()
        if n_de < 3:
            continue

        # Edge scores for this TF
        edge_scores = np.abs(corr[tf, :])
        edge_scores[tf] = 0

        # AUROC
        auroc = roc_auc_score(de_mask.astype(int), edge_scores)

        # Enrichment
        k = n_de
        top_k_idx = np.argsort(edge_scores)[-k:]
        precision_at_k = de_mask[top_k_idx].mean()
        expected = n_de / n_genes
        fold = precision_at_k / expected if expected > 0 else 0

        # Also compute: do true regulatory targets (from A) get recovered?
        true_mask = np.zeros(n_genes, dtype=bool)
        for tgt in true_targets[tf]:
            true_mask[tgt] = True
        true_mask[tf] = False
        auroc_true = roc_auc_score(true_mask.astype(int), edge_scores) if true_mask.sum() > 0 else 0.5

        results.append({
            'perturbed_gene': f'TF_{tf}',
            'n_true_targets': len(true_targets[tf]),
            'n_de_targets': n_de,
            'auroc_vs_de': auroc,
            'auroc_vs_true_regulatory': auroc_true,
            'precision_at_k': precision_at_k,
            'fold_enrichment': fold
        })
        print(f"  TF_{tf}: {len(true_targets[tf])} true targets, {n_de} DE, "
              f"AUROC(DE)={auroc:.3f}, AUROC(true)={auroc_true:.3f}, fold={fold:.2f}")

    df = pd.DataFrame(results)
    return finalize_results(df, 'synthetic_perturbation')


def finalize_results(df, dataset_name):
    """Generate figures and summary from perturbation validation results."""

    df.to_csv(os.path.join(OUT_DIR, f"perturbation_validation_{dataset_name}.csv"), index=False)

    auroc_col = 'auroc' if 'auroc' in df.columns else 'auroc_vs_de'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: AUROC distribution
    ax = axes[0]
    ax.hist(df[auroc_col], bins=15, color='#2196F3', alpha=0.7, edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Random (0.5)')
    ax.axvline(df[auroc_col].mean(), color='green', linestyle='-', linewidth=2,
               label=f'Mean ({df[auroc_col].mean():.3f})')
    ax.set_xlabel('AUROC (edge score → perturbation targets)')
    ax.set_ylabel('Count')
    ax.set_title(f'(A) Per-TF AUROC distribution\n(n={len(df)} TFs)')
    ax.legend()

    # Panel B: Enrichment
    enrich_col = 'fold_enrichment'
    if enrich_col in df.columns:
        ax = axes[1]
        ax.bar(range(len(df)), sorted(df[enrich_col], reverse=True),
               color='#4CAF50', alpha=0.7)
        ax.axhline(1.0, color='red', linestyle='--', label='No enrichment')
        ax.set_xlabel('Ranked TFs')
        ax.set_ylabel('Fold enrichment (top-K)')
        ax.set_title('(B) Perturbation target enrichment')
        ax.legend()

    # Panel C: AUROC vs number of targets
    ax = axes[2]
    n_col = 'n_de_targets' if 'n_de_targets' in df.columns else 'n_true_targets'
    ax.scatter(df[n_col], df[auroc_col], c='#FF9800', alpha=0.7, s=50)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    rho, pval = stats.spearmanr(df[n_col], df[auroc_col])
    ax.set_xlabel('Number of DE targets')
    ax.set_ylabel('AUROC')
    ax.set_title(f'(C) AUROC vs target count\n(ρ={rho:.3f}, p={pval:.3f})')

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, f"fig_perturbation_first_{dataset_name}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to {fig_path}")

    # One-sample t-test: mean AUROC > 0.5?
    t_stat, t_pval = stats.ttest_1samp(df[auroc_col], 0.5)
    wilcox_stat, wilcox_pval = stats.wilcoxon(df[auroc_col] - 0.5, alternative='greater')

    summary = {
        'experiment': 'perturbation_first_validation',
        'dataset': dataset_name,
        'n_tfs_evaluated': len(df),
        'mean_auroc': round(float(df[auroc_col].mean()), 4),
        'median_auroc': round(float(df[auroc_col].median()), 4),
        'std_auroc': round(float(df[auroc_col].std()), 4),
        'mean_fold_enrichment': round(float(df['fold_enrichment'].mean()), 4) if 'fold_enrichment' in df.columns else None,
        'ttest_vs_0.5': {'t': round(float(t_stat), 4), 'p': float(t_pval)},
        'wilcoxon_vs_0.5': {'stat': float(wilcox_stat), 'p': float(wilcox_pval)},
        'n_tfs_above_0.5': int((df[auroc_col] > 0.5).sum()),
        'n_tfs_below_0.5': int((df[auroc_col] < 0.5).sum()),
    }

    with open(os.path.join(OUT_DIR, f"perturbation_summary_{dataset_name}.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== EXPERIMENT COMPLETE ===")
    return df, summary


if __name__ == '__main__':
    df, summary = run_perturbation_validation()
