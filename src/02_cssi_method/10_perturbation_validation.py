"""
Phase 2: Perturbation Validation of State-Gated Edges
=====================================================
Cross-references state-gated edges with Perturb-seq data from Adamson, Dixit,
and Shifrut datasets to test whether state-gated-active edges are more likely
to show perturbation effects than weak/inactive edges.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse, stats
from scipy.stats import spearmanr, mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings, time

warnings.filterwarnings("ignore")
np.random.seed(42)

PERTURB_DIR = "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/perturb"
OUT_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_28_cell_state_conditional_circuits"
FIG_DIR = os.path.join(OUT_DIR, "reports", "figures")
IMPL_DIR = os.path.join(OUT_DIR, "implementation")
os.makedirs(FIG_DIR, exist_ok=True)

# Load Phase 1 results
df_edges = pd.read_csv(os.path.join(IMPL_DIR, "revised_contrast_scores.csv"))
edge_summary = pd.read_csv(os.path.join(IMPL_DIR, "edge_summary_revised.csv"))

print("Phase 1 edge classes:")
print(df_edges['edge_class'].value_counts())

# Build the set of TFs and targets in our edge list
our_tfs = set(df_edges['tf'].unique())
our_targets = set(df_edges['target'].unique())
our_edges = set(zip(df_edges['tf'], df_edges['target']))
print(f"\nOur TFs: {len(our_tfs)}, Targets: {len(our_targets)}, Edges: {len(our_edges)}")


def load_perturb_data(name, path):
    """Load perturbation dataset and extract relevant information."""
    print(f"\n{'='*50}")
    print(f"Loading {name}...")
    adata = sc.read_h5ad(path)
    print(f"  Shape: {adata.shape}")

    # Identify gene names
    if 'feature_name' in adata.var.columns:
        gene_names = adata.var['feature_name'].values
    elif 'gene_name' in adata.var.columns:
        gene_names = adata.var['gene_name'].values
    else:
        gene_names = adata.var_names.values

    # Check which are symbols vs ensembl IDs
    n_ensg = sum(1 for g in gene_names[:100] if str(g).startswith('ENSG'))
    if n_ensg > 50:
        print("  Gene names appear to be Ensembl IDs, trying var_names or feature_name...")
        # Try to find symbols
        for col in ['feature_name', 'gene_name', 'gene_short_name', 'symbol']:
            if col in adata.var.columns:
                gene_names = adata.var[col].values
                print(f"  Using {col} column")
                break

    # Build symbol index: from the var_names, check if they ARE gene symbols
    # or if we need a mapping column
    symbol_set = set(str(g) for g in gene_names if pd.notna(g))
    matched_tfs = our_tfs & symbol_set
    matched_targets = our_targets & symbol_set
    print(f"  Matched TFs: {matched_tfs}")
    print(f"  Matched targets: {len(matched_targets)}")

    return adata, gene_names, symbol_set


def compute_perturbation_effects(adata, gene_names, name):
    """
    For each perturbed TF in our edge list, compute the differential expression
    of its predicted targets between perturbed and control cells.
    Returns a DataFrame of perturbation effects.
    """
    results = []

    # Build gene name to index mapping
    gene_to_idx = {}
    for i, g in enumerate(gene_names):
        g_str = str(g)
        if pd.notna(g) and g_str != 'nan':
            gene_to_idx[g_str] = i

    # Identify perturbation column and control label
    if name == 'adamson':
        cond_col = 'condition'
        # Perturbed conditions look like "GENENAME+ctrl"
        perturbed_genes = set()
        for c in adata.obs[cond_col].unique():
            if '+ctrl' in str(c) and c != 'ctrl':
                perturbed_genes.add(str(c).split('+')[0])
        control_mask = adata.obs[cond_col] == 'ctrl'
    elif name == 'dixit':
        cond_col = 'condition'
        perturbed_genes = set(str(c) for c in adata.obs[cond_col].unique()
                             if c != 'control' and str(c) != 'nan')
        control_mask = adata.obs[cond_col] == 'control'
    elif name == 'shifrut':
        cond_col = 'condition'
        perturbed_genes = set(str(c) for c in adata.obs[cond_col].unique()
                             if c != 'control' and str(c) != 'nan' and c != 'NonTarget')
        control_mask = adata.obs[cond_col] == 'control'
    else:
        return pd.DataFrame()

    print(f"  Perturbed genes: {perturbed_genes}")
    print(f"  Control cells: {control_mask.sum()}")

    # Get expression matrix
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()

    # Control expression
    X_ctrl = X[control_mask.values]
    n_ctrl = X_ctrl.shape[0]

    for perturbed_tf in perturbed_genes:
        if perturbed_tf not in our_tfs:
            continue

        # Get perturbed cells
        if name == 'adamson':
            pert_mask = adata.obs[cond_col] == f'{perturbed_tf}+ctrl'
        else:
            pert_mask = adata.obs[cond_col] == perturbed_tf
        n_pert = pert_mask.sum()
        if n_pert < 10:
            continue

        X_pert = X[pert_mask.values]

        # For each target of this TF in our edge list
        targets_of_tf = [tgt for (tf, tgt) in our_edges if tf == perturbed_tf]
        for target in targets_of_tf:
            if target not in gene_to_idx:
                continue
            tidx = gene_to_idx[target]

            ctrl_expr = X_ctrl[:, tidx]
            pert_expr = X_pert[:, tidx]

            # Mean log-fold change (pseudocount to avoid log(0))
            mean_ctrl = np.mean(ctrl_expr)
            mean_pert = np.mean(pert_expr)
            lfc = np.log2((mean_pert + 0.01) / (mean_ctrl + 0.01))

            # Mann-Whitney U test
            if np.std(ctrl_expr) == 0 and np.std(pert_expr) == 0:
                mwu_p = 1.0
                mwu_stat = 0.0
            else:
                try:
                    mwu_stat, mwu_p = mannwhitneyu(pert_expr, ctrl_expr, alternative='two-sided')
                except ValueError:
                    mwu_p = 1.0
                    mwu_stat = 0.0

            # Effect size: Cohen's d
            pooled_std = np.sqrt((np.var(ctrl_expr) * (n_ctrl - 1) + np.var(pert_expr) * (n_pert - 1)) /
                                 (n_ctrl + n_pert - 2))
            cohens_d = (mean_pert - mean_ctrl) / max(pooled_std, 1e-8)

            # Also check the TF itself (knockdown efficiency)
            if perturbed_tf in gene_to_idx:
                tf_idx = gene_to_idx[perturbed_tf]
                tf_ctrl = np.mean(X_ctrl[:, tf_idx])
                tf_pert = np.mean(X_pert[:, tf_idx])
                tf_lfc = np.log2((tf_pert + 0.01) / (tf_ctrl + 0.01))
            else:
                tf_lfc = np.nan

            results.append({
                'dataset': name,
                'perturbed_tf': perturbed_tf,
                'target': target,
                'n_pert': n_pert,
                'n_ctrl': n_ctrl,
                'mean_ctrl': mean_ctrl,
                'mean_pert': mean_pert,
                'log2fc': lfc,
                'cohens_d': cohens_d,
                'mwu_stat': mwu_stat,
                'mwu_p': mwu_p,
                'tf_log2fc': tf_lfc,
            })

    return pd.DataFrame(results)


# ============================================================
# Process all perturbation datasets
# ============================================================
all_perturb_results = []

for name, path in [
    ('adamson', os.path.join(PERTURB_DIR, 'adamson', 'perturb_processed_symbols.h5ad')),
    ('dixit', os.path.join(PERTURB_DIR, 'dixit', 'perturb_processed_symbols.h5ad')),
    ('shifrut', os.path.join(PERTURB_DIR, 'shifrut', 'perturb_processed_symbols.h5ad')),
]:
    adata, gene_names, symbol_set = load_perturb_data(name, path)
    perturb_df = compute_perturbation_effects(adata, gene_names, name)
    if len(perturb_df) > 0:
        all_perturb_results.append(perturb_df)
        print(f"\n  {name}: {len(perturb_df)} TF-target perturbation effects computed")
    del adata  # Free memory

if len(all_perturb_results) == 0:
    print("\nNo perturbation effects found. Saving empty results.")
    pd.DataFrame().to_csv(os.path.join(IMPL_DIR, "phase2_perturbation_effects.csv"), index=False)
else:
    df_perturb = pd.concat(all_perturb_results, ignore_index=True)

    # FDR correction across all perturbation tests
    _, df_perturb['mwu_p_adj'], _, _ = multipletests(df_perturb['mwu_p'], method='fdr_bh')
    df_perturb['sig_perturb'] = df_perturb['mwu_p_adj'] < 0.05

    print(f"\n{'='*60}")
    print("PERTURBATION VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"\nTotal TF-target perturbation effects: {len(df_perturb)}")
    print(f"Significant (FDR < 0.05): {df_perturb['sig_perturb'].sum()}")

    # Merge with Phase 1 edge classifications
    # For each (tf, target), get the edge-level summary
    edge_class_map = {}
    for _, row in edge_summary.iterrows():
        edge_class_map[(row['tf'], row['target'])] = {
            'n_gated': row['n_gated'],
            'n_always_on': row['n_always_on'],
            'rho_global': row['rho_global'],
            'rho_range': row['rho_range'],
        }

    # Also get the dominant class from the detailed results
    for (tf, tgt), grp in df_edges.groupby(['tf', 'target']):
        classes = grp['edge_class'].value_counts()
        n_gated = grp['edge_class'].isin(['state-gated-active', 'state-gated-inactive']).sum()
        n_active = (grp['edge_class'] == 'state-gated-active').sum()
        if n_gated > len(grp) / 3:
            dom = 'predominantly-gated'
        elif n_active > 0:
            dom = 'partially-gated'
        elif (grp['edge_class'] == 'always-on').sum() > 0:
            dom = 'always-on'
        else:
            dom = 'weak/inactive'
        if (tf, tgt) in edge_class_map:
            edge_class_map[(tf, tgt)]['dominant_class'] = dom

    df_perturb['n_gated_regimes'] = df_perturb.apply(
        lambda r: edge_class_map.get((r['perturbed_tf'], r['target']), {}).get('n_gated', 0), axis=1)
    df_perturb['dominant_class'] = df_perturb.apply(
        lambda r: edge_class_map.get((r['perturbed_tf'], r['target']), {}).get('dominant_class', 'unknown'), axis=1)
    df_perturb['rho_global'] = df_perturb.apply(
        lambda r: edge_class_map.get((r['perturbed_tf'], r['target']), {}).get('rho_global', np.nan), axis=1)

    # KEY TEST: Are state-gated-active edges more likely to be validated by perturbation?
    print("\n--- Perturbation Validation by Edge Class ---")
    for cls in df_perturb['dominant_class'].unique():
        sub = df_perturb[df_perturb['dominant_class'] == cls]
        n_sig = sub['sig_perturb'].sum()
        print(f"  {cls}: {n_sig}/{len(sub)} significant ({100*n_sig/max(len(sub),1):.1f}%), "
              f"median |log2FC| = {sub['log2fc'].abs().median():.3f}, "
              f"median |Cohen's d| = {sub['cohens_d'].abs().median():.3f}")

    # Statistical test: enrichment of perturbation significance in gated edges
    gated_sig = df_perturb[df_perturb['n_gated_regimes'] > 0]['sig_perturb'].sum()
    gated_total = (df_perturb['n_gated_regimes'] > 0).sum()
    nongated_sig = df_perturb[df_perturb['n_gated_regimes'] == 0]['sig_perturb'].sum()
    nongated_total = (df_perturb['n_gated_regimes'] == 0).sum()

    if gated_total > 0 and nongated_total > 0:
        contingency = [[gated_sig, gated_total - gated_sig],
                       [nongated_sig, nongated_total - nongated_sig]]
        odds, fisher_p = fisher_exact(contingency)
        print(f"\n  Fisher's exact test (gated vs non-gated perturbation validation):")
        print(f"    Gated edges: {gated_sig}/{gated_total} ({100*gated_sig/gated_total:.1f}%) significant")
        print(f"    Non-gated: {nongated_sig}/{nongated_total} ({100*nongated_sig/nongated_total:.1f}%) significant")
        print(f"    OR = {odds:.2f}, p = {fisher_p:.4f}")

    # Correlation between n_gated_regimes and perturbation effect size
    valid_perturb = df_perturb[df_perturb['rho_global'].notna()]
    if len(valid_perturb) >= 5:
        r_gated_effect, p_gated_effect = spearmanr(
            valid_perturb['n_gated_regimes'], valid_perturb['cohens_d'].abs())
        print(f"\n  Spearman(n_gated_regimes, |Cohen's d|): rho={r_gated_effect:.3f}, p={p_gated_effect:.4f}")

        r_rho_effect, p_rho_effect = spearmanr(
            valid_perturb['rho_global'].abs(), valid_perturb['cohens_d'].abs())
        print(f"  Spearman(|rho_global|, |Cohen's d|): rho={r_rho_effect:.3f}, p={p_rho_effect:.4f}")

    # Per-dataset breakdown
    print("\n--- Per-Dataset Breakdown ---")
    for ds in df_perturb['dataset'].unique():
        sub = df_perturb[df_perturb['dataset'] == ds]
        print(f"\n  {ds}:")
        print(f"    Edges tested: {len(sub)}")
        print(f"    Perturbed TFs: {sub['perturbed_tf'].unique().tolist()}")
        print(f"    Significant: {sub['sig_perturb'].sum()}/{len(sub)}")
        if len(sub) > 0:
            print(sub[['perturbed_tf', 'target', 'log2fc', 'cohens_d', 'mwu_p_adj',
                       'sig_perturb', 'n_gated_regimes', 'dominant_class']].to_string(index=False))

    # Save results
    df_perturb.to_csv(os.path.join(IMPL_DIR, "phase2_perturbation_effects.csv"), index=False)

    # ============================================================
    # FIGURES
    # ============================================================
    print("\nGenerating perturbation validation figures...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Effect size by dominant class
    class_order_p = ['predominantly-gated', 'partially-gated', 'always-on', 'weak/inactive']
    colors_p = {'predominantly-gated': '#e74c3c', 'partially-gated': '#f39c12',
                'always-on': '#2ecc71', 'weak/inactive': '#95a5a6', 'unknown': '#bdc3c7'}
    plot_data = df_perturb[df_perturb['dominant_class'].isin(class_order_p)]
    if len(plot_data) > 0:
        sns.boxplot(data=plot_data, x='dominant_class', y='cohens_d',
                    order=[c for c in class_order_p if c in plot_data['dominant_class'].values],
                    palette=colors_p, ax=axes[0], showfliers=False)
        sns.stripplot(data=plot_data, x='dominant_class', y='cohens_d',
                     order=[c for c in class_order_p if c in plot_data['dominant_class'].values],
                     color='black', alpha=0.4, size=4, ax=axes[0])
        axes[0].set_xlabel('Edge Class')
        axes[0].set_ylabel("Cohen's d")
        axes[0].set_title('Perturbation Effect Size\nby Edge Classification')
        axes[0].tick_params(axis='x', rotation=30)
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Panel B: n_gated vs |Cohen's d|
    axes[1].scatter(df_perturb['n_gated_regimes'], df_perturb['cohens_d'].abs(),
                    alpha=0.6, s=40, c='#2c3e50')
    axes[1].set_xlabel('Number of state-gated regimes')
    axes[1].set_ylabel("|Cohen's d|")
    axes[1].set_title('State-Gating vs Perturbation Effect')
    if len(valid_perturb) >= 5:
        axes[1].annotate(f'rho={r_gated_effect:.3f}\np={p_gated_effect:.3f}',
                         xy=(0.05, 0.95), xycoords='axes fraction', va='top', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel C: Volcano-style plot of perturbation effects
    df_perturb['neg_log10_p'] = -np.log10(df_perturb['mwu_p_adj'].clip(1e-50))
    for cls, color in colors_p.items():
        sub = df_perturb[df_perturb['dominant_class'] == cls]
        if len(sub) > 0:
            axes[2].scatter(sub['log2fc'], sub['neg_log10_p'],
                           alpha=0.6, s=40, c=color, label=cls)
    axes[2].axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('log2 fold change')
    axes[2].set_ylabel('-log10(FDR)')
    axes[2].set_title('Perturbation Effects\n(Colored by Edge Class)')
    axes[2].legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "phase2_fig4_perturbation_validation.png"), dpi=150, bbox_inches='tight')
    plt.close()

print("\nPhase 2 perturbation validation complete!")
