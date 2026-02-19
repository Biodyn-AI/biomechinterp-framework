"""
Phase 2: Ceiling Effect Investigation
======================================
Mechanistic investigation of the "ceiling effect" where lineage-defining edges
decorrelate within committed cell types. Tests three hypotheses:
  1. Expression saturation: Both genes are at ceiling, reducing variance
  2. Loss of variance: Coefficient of variation drops within committed types
  3. Genuine regulatory disconnection: The TF truly stops driving the target

Uses quantile-based analysis, variance decomposition, and non-linear measures.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse, stats
from scipy.stats import spearmanr, pearsonr, ks_2samp
from scipy.special import digamma
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings, time
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR = "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw"
OUT_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_28_cell_state_conditional_circuits"
FIG_DIR = os.path.join(OUT_DIR, "reports", "figures")
IMPL_DIR = os.path.join(OUT_DIR, "implementation")
os.makedirs(FIG_DIR, exist_ok=True)

# Load data
print("Loading data...")
adata = sc.read_h5ad(os.path.join(DATA_DIR, "tabula_sapiens_immune_subset_20000.h5ad"))
adata.obs['regime'] = adata.obs['cell_type'].astype(str)
regime_counts = adata.obs['regime'].value_counts()
valid_regimes = regime_counts[regime_counts >= 100].index.tolist()
adata = adata[adata.obs['regime'].isin(valid_regimes)].copy()
regimes = adata.obs['regime'].values
unique_regimes = sorted(set(regimes))
print(f"Data: {adata.shape[0]} cells, {len(unique_regimes)} regimes")

# Gene mapping
symbol_to_ensembl = {}
for ens_id, row in adata.var.iterrows():
    sym = row['feature_name']
    if pd.notna(sym):
        symbol_to_ensembl[sym] = ens_id

# Load Phase 1 results
df_edges = pd.read_csv(os.path.join(IMPL_DIR, "revised_contrast_scores.csv"))
edge_summary = pd.read_csv(os.path.join(IMPL_DIR, "edge_summary_revised.csv"))

# Identify ceiling-effect candidates: edges where rho_global is high but
# rho_focal in the cognate cell type is low/negative
CEILING_CANDIDATES = [
    # (TF, target, cognate_regime) - known lineage-defining edges
    ("PAX5", "CD79A", "B cell"),
    ("PAX5", "CD19", "B cell"),
    ("PAX5", "MS4A1", "B cell"),
    ("EBF1", "CD79A", "B cell"),
    ("EBF1", "CD79B", "B cell"),
    ("RUNX3", "CD8A", "CD8-positive, alpha-beta T cell"),
    ("SPI1", "ITGAM", "monocyte"),
    ("SPI1", "CD68", "macrophage"),
    ("SPI1", "CSF1R", "macrophage"),
    ("TCF7", "LEF1", "CD4-positive, alpha-beta T cell"),
    ("TCF7", "IL7R", "CD4-positive, alpha-beta T cell"),
    ("FOXP3", "CTLA4", "regulatory T cell"),
    ("FOXP3", "IL2RA", "regulatory T cell"),
]

# Also look at non-ceiling edges for comparison
NON_CEILING_EDGES = [
    ("NFKB1", "IL1B", "monocyte"),  # state-gated-active
    ("NFKB1", "TNF", "macrophage"),  # state-gated-active
    ("MYC", "CDK4", "erythrocyte"),  # state-gated-active
    ("HIF1A", "VEGFA", "macrophage"),  # active in cognate type
    ("STAT1", "IRF1", "monocyte"),
]

# Expression matrix
all_genes_needed = set()
for tf, tgt, _ in CEILING_CANDIDATES + NON_CEILING_EDGES:
    if tf in symbol_to_ensembl:
        all_genes_needed.add(symbol_to_ensembl[tf])
    if tgt in symbol_to_ensembl:
        all_genes_needed.add(symbol_to_ensembl[tgt])

gene_idx = [i for i, g in enumerate(adata.var_names) if g in all_genes_needed]
X_sub = adata.X[:, gene_idx]
if sparse.issparse(X_sub):
    X_sub = X_sub.toarray()
expr_df = pd.DataFrame(X_sub, index=adata.obs_names,
                        columns=[adata.var_names[i] for i in gene_idx])


def spearman_safe(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    r, p = spearmanr(x, y)
    return (r, p) if not np.isnan(r) else (0.0, 1.0)


def compute_mutual_information(x, y, n_bins=10):
    """Estimate mutual information using binning."""
    c_xy = np.histogram2d(x, y, bins=n_bins)[0]
    c_xy = c_xy / c_xy.sum()
    c_x = c_xy.sum(axis=1)
    c_y = c_xy.sum(axis=0)
    # Avoid log(0)
    mask = c_xy > 0
    mi = np.sum(c_xy[mask] * np.log(c_xy[mask] / (c_x[:, None] * c_y[None, :])[mask]))
    return mi


# ============================================================
# ANALYSIS 1: Expression saturation signatures
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 1: Expression Saturation Signatures")
print("=" * 60)

ceiling_records = []
regime_masks = {r: (regimes == r) for r in unique_regimes}

for tf_sym, tgt_sym, cognate_regime in CEILING_CANDIDATES + NON_CEILING_EDGES:
    is_ceiling_candidate = (tf_sym, tgt_sym, cognate_regime) in CEILING_CANDIDATES
    tf_ens = symbol_to_ensembl.get(tf_sym)
    tgt_ens = symbol_to_ensembl.get(tgt_sym)
    if tf_ens is None or tgt_ens is None:
        continue
    if tf_ens not in expr_df.columns or tgt_ens not in expr_df.columns:
        continue

    tf_all = expr_df[tf_ens].values
    tgt_all = expr_df[tgt_ens].values

    for regime in unique_regimes:
        mask = regime_masks[regime]
        n = mask.sum()
        if n < 50:
            continue

        tf_r = tf_all[mask]
        tgt_r = tgt_all[mask]
        tf_rest = tf_all[~mask]
        tgt_rest = tgt_all[~mask]

        rho_focal, _ = spearman_safe(tf_r, tgt_r)
        rho_global, _ = spearman_safe(tf_all, tgt_all)

        # Saturation metrics
        tf_mean = np.mean(tf_r)
        tgt_mean = np.mean(tgt_r)
        tf_mean_global = np.mean(tf_all)
        tgt_mean_global = np.mean(tgt_all)

        # Coefficient of variation
        tf_cv = np.std(tf_r) / max(np.mean(tf_r), 1e-8)
        tgt_cv = np.std(tgt_r) / max(np.mean(tgt_r), 1e-8)
        tf_cv_global = np.std(tf_all) / max(np.mean(tf_all), 1e-8)
        tgt_cv_global = np.std(tgt_all) / max(np.mean(tgt_all), 1e-8)

        # Fraction expressing (detection rate)
        tf_frac = np.mean(tf_r > 0)
        tgt_frac = np.mean(tgt_r > 0)

        # Quantile position: what fraction of global distribution is below regime mean?
        tf_quantile = np.mean(tf_all <= tf_mean)
        tgt_quantile = np.mean(tgt_all <= tgt_mean)

        # Variance ratio: within-regime / global
        tf_var_ratio = np.var(tf_r) / max(np.var(tf_all), 1e-8)
        tgt_var_ratio = np.var(tgt_r) / max(np.var(tgt_all), 1e-8)

        # Mutual information (non-linear dependence)
        if np.std(tf_r) > 0 and np.std(tgt_r) > 0:
            mi = compute_mutual_information(tf_r, tgt_r)
        else:
            mi = 0.0

        # Dynamic range: max - min of expressed cells
        tf_expr = tf_r[tf_r > 0]
        tgt_expr = tgt_r[tgt_r > 0]
        tf_dynrange = np.ptp(tf_expr) if len(tf_expr) > 1 else 0.0
        tgt_dynrange = np.ptp(tgt_expr) if len(tgt_expr) > 1 else 0.0

        # Bimodality coefficient (Sarle's BC)
        def bimodality_coefficient(x):
            n = len(x)
            if n < 4 or np.std(x) == 0:
                return 0.0
            skew = stats.skew(x)
            kurt = stats.kurtosis(x, fisher=True)
            bc = (skew**2 + 1) / (kurt + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))
            return bc

        tf_bc = bimodality_coefficient(tf_r)
        tgt_bc = bimodality_coefficient(tgt_r)

        ceiling_records.append({
            'tf': tf_sym, 'target': tgt_sym,
            'cognate_regime': cognate_regime,
            'is_ceiling_candidate': is_ceiling_candidate,
            'regime': regime,
            'is_cognate': regime == cognate_regime,
            'n_cells': n,
            'rho_focal': rho_focal,
            'rho_global': rho_global,
            'tf_mean': tf_mean, 'tgt_mean': tgt_mean,
            'tf_mean_global': tf_mean_global, 'tgt_mean_global': tgt_mean_global,
            'tf_cv': tf_cv, 'tgt_cv': tgt_cv,
            'tf_cv_global': tf_cv_global, 'tgt_cv_global': tgt_cv_global,
            'tf_frac': tf_frac, 'tgt_frac': tgt_frac,
            'tf_quantile': tf_quantile, 'tgt_quantile': tgt_quantile,
            'tf_var_ratio': tf_var_ratio, 'tgt_var_ratio': tgt_var_ratio,
            'mutual_information': mi,
            'tf_dynrange': tf_dynrange, 'tgt_dynrange': tgt_dynrange,
            'tf_bimodality': tf_bc, 'tgt_bimodality': tgt_bc,
        })

df_ceiling = pd.DataFrame(ceiling_records)
print(f"Total ceiling analysis records: {len(df_ceiling)}")

# ============================================================
# ANALYSIS 2: Ceiling vs Non-Ceiling Comparison
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 2: Comparing Ceiling vs Non-Ceiling Edges")
print("=" * 60)

# Focus on cognate regime entries
cognate = df_ceiling[df_ceiling['is_cognate']]
ceiling_cognate = cognate[cognate['is_ceiling_candidate']]
nonceil_cognate = cognate[~cognate['is_ceiling_candidate']]

print("\nCeiling candidates in their cognate regime:")
cols_show = ['tf', 'target', 'regime', 'rho_focal', 'rho_global',
             'tf_mean', 'tgt_mean', 'tf_cv', 'tgt_cv',
             'tf_frac', 'tgt_frac', 'tf_var_ratio', 'tgt_var_ratio', 'mutual_information']
if len(ceiling_cognate) > 0:
    print(ceiling_cognate[cols_show].to_string(index=False))

print("\nNon-ceiling edges in their cognate regime:")
if len(nonceil_cognate) > 0:
    print(nonceil_cognate[cols_show].to_string(index=False))

# Statistical comparison: ceiling vs non-ceiling in cognate regime
if len(ceiling_cognate) >= 3 and len(nonceil_cognate) >= 3:
    for metric in ['tf_cv', 'tgt_cv', 'tf_var_ratio', 'tgt_var_ratio',
                   'tf_frac', 'tgt_frac', 'mutual_information', 'rho_focal']:
        c_vals = ceiling_cognate[metric].values
        nc_vals = nonceil_cognate[metric].values
        u_stat, u_p = stats.mannwhitneyu(c_vals, nc_vals, alternative='two-sided')
        print(f"  {metric}: ceiling median={np.median(c_vals):.3f}, "
              f"non-ceiling median={np.median(nc_vals):.3f}, MWU p={u_p:.4f}")

# ============================================================
# ANALYSIS 3: Variance Decomposition
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 3: Variance Decomposition - Why Does Correlation Drop?")
print("=" * 60)

# For each ceiling candidate edge, decompose the correlation change
decomp_records = []
for tf_sym, tgt_sym, cognate_regime in CEILING_CANDIDATES:
    tf_ens = symbol_to_ensembl.get(tf_sym)
    tgt_ens = symbol_to_ensembl.get(tgt_sym)
    if tf_ens is None or tgt_ens is None:
        continue
    if tf_ens not in expr_df.columns or tgt_ens not in expr_df.columns:
        continue

    tf_all = expr_df[tf_ens].values
    tgt_all = expr_df[tgt_ens].values

    # Global statistics
    rho_global, _ = spearman_safe(tf_all, tgt_all)
    cov_global = np.cov(tf_all, tgt_all)[0, 1]
    var_tf_global = np.var(tf_all)
    var_tgt_global = np.var(tgt_all)

    # Cognate regime
    mask_cog = regime_masks.get(cognate_regime)
    if mask_cog is None or mask_cog.sum() < 50:
        continue

    tf_cog = tf_all[mask_cog]
    tgt_cog = tgt_all[mask_cog]
    rho_cog, _ = spearman_safe(tf_cog, tgt_cog)
    cov_cog = np.cov(tf_cog, tgt_cog)[0, 1]
    var_tf_cog = np.var(tf_cog)
    var_tgt_cog = np.var(tgt_cog)

    # Pearson (for variance decomposition)
    if np.std(tf_cog) > 0 and np.std(tgt_cog) > 0:
        pearson_cog, _ = pearsonr(tf_cog, tgt_cog)
    else:
        pearson_cog = 0
    if np.std(tf_all) > 0 and np.std(tgt_all) > 0:
        pearson_global, _ = pearsonr(tf_all, tgt_all)
    else:
        pearson_global = 0

    # Decompose Pearson r = cov(X,Y) / (sd(X) * sd(Y))
    # r_global vs r_cognate: is the change due to cov or sd?
    # If we fix cov at global level but use cognate sd: "variance-explained" component
    # If we fix sd at global level but use cognate cov: "covariance-explained" component
    sd_tf_global = np.sqrt(var_tf_global) if var_tf_global > 0 else 1e-8
    sd_tgt_global = np.sqrt(var_tgt_global) if var_tgt_global > 0 else 1e-8
    sd_tf_cog = np.sqrt(var_tf_cog) if var_tf_cog > 0 else 1e-8
    sd_tgt_cog = np.sqrt(var_tgt_cog) if var_tgt_cog > 0 else 1e-8

    # Counterfactual: what would r be with cognate covariance but global variance?
    r_if_global_var = cov_cog / (sd_tf_global * sd_tgt_global) if sd_tf_global * sd_tgt_global > 0 else 0
    # Counterfactual: what would r be with global covariance but cognate variance?
    r_if_cog_var = cov_global / (sd_tf_cog * sd_tgt_cog) if sd_tf_cog * sd_tgt_cog > 0 else 0

    decomp_records.append({
        'tf': tf_sym, 'target': tgt_sym, 'cognate_regime': cognate_regime,
        'pearson_global': pearson_global, 'pearson_cognate': pearson_cog,
        'spearman_global': rho_global, 'spearman_cognate': rho_cog,
        'cov_global': cov_global, 'cov_cognate': cov_cog,
        'var_tf_global': var_tf_global, 'var_tf_cognate': var_tf_cog,
        'var_tgt_global': var_tgt_global, 'var_tgt_cognate': var_tgt_cog,
        'sd_tf_ratio': sd_tf_cog / sd_tf_global,
        'sd_tgt_ratio': sd_tgt_cog / sd_tgt_global,
        'cov_ratio': cov_cog / max(abs(cov_global), 1e-8),
        'r_if_global_var': r_if_global_var,
        'r_if_cog_var': r_if_cog_var,
        'delta_r_total': pearson_cog - pearson_global,
        'delta_r_from_cov': r_if_global_var - pearson_global,
        'delta_r_from_var': r_if_cog_var - pearson_global,
    })

df_decomp = pd.DataFrame(decomp_records)
if len(df_decomp) > 0:
    print("\nVariance Decomposition of Correlation Change:")
    print(df_decomp[['tf', 'target', 'cognate_regime',
                     'pearson_global', 'pearson_cognate',
                     'sd_tf_ratio', 'sd_tgt_ratio', 'cov_ratio',
                     'delta_r_total', 'delta_r_from_cov', 'delta_r_from_var']].to_string(index=False))

    # Key question: is the drop in correlation primarily due to:
    # (a) reduced covariance (true regulatory disconnection)?
    # (b) reduced variance (saturation)?
    # (c) both?
    mean_delta_cov = df_decomp['delta_r_from_cov'].mean()
    mean_delta_var = df_decomp['delta_r_from_var'].mean()
    mean_delta_total = df_decomp['delta_r_total'].mean()
    print(f"\n  Mean delta_r (total): {mean_delta_total:.3f}")
    print(f"  Mean delta_r from covariance change: {mean_delta_cov:.3f}")
    print(f"  Mean delta_r from variance change: {mean_delta_var:.3f}")
    if abs(mean_delta_total) > 0:
        pct_cov = 100 * abs(mean_delta_cov) / (abs(mean_delta_cov) + abs(mean_delta_var))
        pct_var = 100 - pct_cov
        print(f"  Fraction attributable to covariance: {pct_cov:.1f}%")
        print(f"  Fraction attributable to variance: {pct_var:.1f}%")

# ============================================================
# ANALYSIS 4: Quantile-Stratified Correlation
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 4: Quantile-Stratified Correlation (Within Cognate)")
print("=" * 60)

# Within the cognate cell type, split cells into quantiles by TF expression
# and compute target-TF correlation within each quantile
quantile_records = []
N_QUANTILES = 4

for tf_sym, tgt_sym, cognate_regime in CEILING_CANDIDATES:
    tf_ens = symbol_to_ensembl.get(tf_sym)
    tgt_ens = symbol_to_ensembl.get(tgt_sym)
    if tf_ens is None or tgt_ens is None:
        continue
    if tf_ens not in expr_df.columns or tgt_ens not in expr_df.columns:
        continue

    mask_cog = regime_masks.get(cognate_regime)
    if mask_cog is None or mask_cog.sum() < 100:
        continue

    tf_cog = expr_df[tf_ens].values[mask_cog]
    tgt_cog = expr_df[tgt_ens].values[mask_cog]

    # Overall correlation
    rho_overall, _ = spearman_safe(tf_cog, tgt_cog)

    # Quantile-based analysis
    try:
        quantile_labels = pd.qcut(tf_cog, N_QUANTILES, labels=False, duplicates='drop')
    except ValueError:
        continue

    for q in range(int(quantile_labels.max()) + 1):
        q_mask = quantile_labels == q
        n_q = q_mask.sum()
        if n_q < 20:
            continue
        tf_q = tf_cog[q_mask]
        tgt_q = tgt_cog[q_mask]
        rho_q, _ = spearman_safe(tf_q, tgt_q)
        quantile_records.append({
            'tf': tf_sym, 'target': tgt_sym, 'regime': cognate_regime,
            'quantile': q, 'n_cells': n_q,
            'tf_mean': np.mean(tf_q), 'tgt_mean': np.mean(tgt_q),
            'tf_cv': np.std(tf_q) / max(np.mean(tf_q), 1e-8),
            'tgt_cv': np.std(tgt_q) / max(np.mean(tgt_q), 1e-8),
            'rho_quantile': rho_q,
            'rho_overall': rho_overall,
        })

df_quantile = pd.DataFrame(quantile_records)
if len(df_quantile) > 0:
    print("\nQuantile-Stratified Correlation (within cognate cell type):")
    print(df_quantile[['tf', 'target', 'regime', 'quantile', 'n_cells',
                       'tf_mean', 'tgt_mean', 'rho_quantile', 'rho_overall']].to_string(index=False))

    # If correlation is lower in high-expression quantiles -> saturation evidence
    # If correlation is uniformly low -> genuine disconnection
    high_q = df_quantile[df_quantile['quantile'] == df_quantile['quantile'].max()]
    low_q = df_quantile[df_quantile['quantile'] == 0]
    if len(high_q) > 0 and len(low_q) > 0:
        print(f"\n  Median rho in lowest TF quantile: {low_q['rho_quantile'].median():.3f}")
        print(f"  Median rho in highest TF quantile: {high_q['rho_quantile'].median():.3f}")

# ============================================================
# ANALYSIS 5: Expression Distribution Comparison
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 5: Expression Distribution Signatures")
print("=" * 60)

# Compare distribution shapes between cognate and non-cognate
for tf_sym, tgt_sym, cognate_regime in CEILING_CANDIDATES[:5]:
    tf_ens = symbol_to_ensembl.get(tf_sym)
    tgt_ens = symbol_to_ensembl.get(tgt_sym)
    if tf_ens is None or tgt_ens is None:
        continue
    if tf_ens not in expr_df.columns or tgt_ens not in expr_df.columns:
        continue

    mask_cog = regime_masks.get(cognate_regime)
    if mask_cog is None:
        continue

    tf_cog = expr_df[tf_ens].values[mask_cog]
    tf_rest = expr_df[tf_ens].values[~mask_cog]
    tgt_cog = expr_df[tgt_ens].values[mask_cog]
    tgt_rest = expr_df[tgt_ens].values[~mask_cog]

    # KS test comparing distributions
    ks_tf, ks_tf_p = ks_2samp(tf_cog, tf_rest)
    ks_tgt, ks_tgt_p = ks_2samp(tgt_cog, tgt_rest)

    print(f"\n  {tf_sym}->{tgt_sym} in {cognate_regime}:")
    print(f"    TF: cognate mean={np.mean(tf_cog):.3f}, rest mean={np.mean(tf_rest):.3f}, "
          f"KS={ks_tf:.3f}, p={ks_tf_p:.2e}")
    print(f"    Target: cognate mean={np.mean(tgt_cog):.3f}, rest mean={np.mean(tgt_rest):.3f}, "
          f"KS={ks_tgt:.3f}, p={ks_tgt_p:.2e}")
    print(f"    Cognate frac>0: TF={np.mean(tf_cog>0):.2f}, Target={np.mean(tgt_cog>0):.2f}")
    print(f"    Rest frac>0: TF={np.mean(tf_rest>0):.2f}, Target={np.mean(tgt_rest>0):.2f}")


# ============================================================
# FIGURES
# ============================================================
print("\nGenerating ceiling effect figures...")

# Figure P2-5: Ceiling Effect Multi-Panel
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Panel A: Scatter - cognate rho vs expression level
cognate_ceil = df_ceiling[(df_ceiling['is_cognate']) & (df_ceiling['is_ceiling_candidate'])]
cognate_nonceil = df_ceiling[(df_ceiling['is_cognate']) & (~df_ceiling['is_ceiling_candidate'])]

if len(cognate_ceil) > 0:
    axes[0, 0].scatter(cognate_ceil['tgt_mean'], cognate_ceil['rho_focal'],
                       c='red', s=80, alpha=0.7, label='Ceiling candidates', zorder=5)
    for _, row in cognate_ceil.iterrows():
        axes[0, 0].annotate(f"{row['tf']}->{row['target']}",
                            (row['tgt_mean'], row['rho_focal']),
                            fontsize=6, alpha=0.7)
if len(cognate_nonceil) > 0:
    axes[0, 0].scatter(cognate_nonceil['tgt_mean'], cognate_nonceil['rho_focal'],
                       c='blue', s=80, alpha=0.7, label='Non-ceiling', zorder=5)
    for _, row in cognate_nonceil.iterrows():
        axes[0, 0].annotate(f"{row['tf']}->{row['target']}",
                            (row['tgt_mean'], row['rho_focal']),
                            fontsize=6, alpha=0.7)
axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Mean target expression (cognate regime)')
axes[0, 0].set_ylabel('Spearman rho (cognate regime)')
axes[0, 0].set_title('Correlation vs Expression Level\nin Cognate Cell Type')
axes[0, 0].legend(fontsize=8)

# Panel B: CV ratio (cognate/global) - ceiling vs non-ceiling
all_cognate = df_ceiling[df_ceiling['is_cognate']]
if len(all_cognate) > 0:
    for grp_name, grp_data, color in [
        ('Ceiling', all_cognate[all_cognate['is_ceiling_candidate']], '#e74c3c'),
        ('Non-ceiling', all_cognate[~all_cognate['is_ceiling_candidate']], '#3498db')
    ]:
        if len(grp_data) > 0:
            cv_ratios = grp_data['tgt_cv'] / grp_data['tgt_cv_global'].clip(lower=1e-8)
            axes[0, 1].hist(cv_ratios.values, bins=15, alpha=0.6, color=color,
                           label=f'{grp_name} (n={len(grp_data)})')
    axes[0, 1].axvline(x=1.0, color='gray', linestyle='--')
    axes[0, 1].set_xlabel('CV ratio (cognate / global)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Target Gene CV Reduction\nin Cognate vs Global')
    axes[0, 1].legend(fontsize=8)

# Panel C: Variance decomposition
if len(df_decomp) > 0:
    x_pos = np.arange(len(df_decomp))
    width = 0.35
    axes[0, 2].bar(x_pos - width/2, df_decomp['delta_r_from_cov'].abs(),
                   width, label='|Delta r from cov|', color='#e74c3c', alpha=0.7)
    axes[0, 2].bar(x_pos + width/2, df_decomp['delta_r_from_var'].abs(),
                   width, label='|Delta r from var|', color='#3498db', alpha=0.7)
    labels = [f"{r['tf']}->{r['target']}" for _, r in df_decomp.iterrows()]
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    axes[0, 2].set_ylabel('|Delta Pearson r|')
    axes[0, 2].set_title('Variance Decomposition\nof Correlation Drop')
    axes[0, 2].legend(fontsize=8)

# Panel D: Quantile-stratified correlation
if len(df_quantile) > 0:
    edges_in_q = df_quantile.groupby(['tf', 'target']).size().reset_index()
    for i, (_, row) in enumerate(edges_in_q.iterrows()):
        sub = df_quantile[(df_quantile['tf'] == row['tf']) & (df_quantile['target'] == row['target'])]
        axes[1, 0].plot(sub['quantile'], sub['rho_quantile'], 'o-',
                       label=f"{row['tf']}->{row['target']}", alpha=0.7)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('TF expression quantile')
    axes[1, 0].set_ylabel('Spearman rho')
    axes[1, 0].set_title('Correlation by TF Expression Quantile\n(within cognate type)')
    axes[1, 0].legend(fontsize=6, loc='best', ncol=2)

# Panel E: Mutual information comparison (cognate vs others)
ceiling_edges = [(tf, tgt) for tf, tgt, _ in CEILING_CANDIDATES]
for tf_sym, tgt_sym, _ in CEILING_CANDIDATES[:5]:
    sub = df_ceiling[(df_ceiling['tf'] == tf_sym) & (df_ceiling['target'] == tgt_sym)]
    if len(sub) > 0:
        sub_sorted = sub.sort_values('mutual_information')
        colors = ['red' if r else 'blue' for r in sub_sorted['is_cognate']]
        axes[1, 1].barh(range(len(sub_sorted)), sub_sorted['mutual_information'],
                       color=colors, alpha=0.5, height=0.8)
break_here = True  # only do one example for MI panel

if len(df_ceiling) > 0:
    # Better approach: compare MI in cognate vs non-cognate
    ceil_only = df_ceiling[df_ceiling['is_ceiling_candidate']]
    mi_cognate = ceil_only[ceil_only['is_cognate']]['mutual_information'].values
    mi_noncognate = ceil_only[~ceil_only['is_cognate']]['mutual_information'].values
    if len(mi_cognate) > 0 and len(mi_noncognate) > 0:
        axes[1, 1].clear()
        axes[1, 1].hist(mi_noncognate, bins=20, alpha=0.6, color='blue', label='Non-cognate regimes')
        axes[1, 1].axvline(np.median(mi_cognate), color='red', linewidth=2,
                          linestyle='--', label=f'Cognate median (n={len(mi_cognate)})')
        axes[1, 1].set_xlabel('Mutual Information')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('MI in Cognate vs Non-Cognate\n(Ceiling Edges)')
        axes[1, 1].legend(fontsize=8)

# Panel F: Example scatter plots for PAX5->CD79A
for tf_sym, tgt_sym, cognate_regime in [("PAX5", "CD79A", "B cell")]:
    tf_ens = symbol_to_ensembl.get(tf_sym)
    tgt_ens = symbol_to_ensembl.get(tgt_sym)
    if tf_ens is None or tgt_ens is None:
        continue
    if tf_ens not in expr_df.columns or tgt_ens not in expr_df.columns:
        continue

    mask_cog = regime_masks.get(cognate_regime, np.zeros(len(regimes), dtype=bool))
    tf_cog = expr_df[tf_ens].values[mask_cog]
    tgt_cog = expr_df[tgt_ens].values[mask_cog]
    tf_rest = expr_df[tf_ens].values[~mask_cog]
    tgt_rest = expr_df[tgt_ens].values[~mask_cog]

    axes[1, 2].scatter(tf_rest, tgt_rest, alpha=0.1, s=5, c='gray', label='Other cells')
    axes[1, 2].scatter(tf_cog, tgt_cog, alpha=0.3, s=10, c='red', label=f'{cognate_regime}')
    rho_cog, _ = spearman_safe(tf_cog, tgt_cog)
    rho_rest, _ = spearman_safe(tf_rest, tgt_rest)
    axes[1, 2].set_xlabel(f'{tf_sym} expression')
    axes[1, 2].set_ylabel(f'{tgt_sym} expression')
    axes[1, 2].set_title(f'{tf_sym}->{tgt_sym} Ceiling Effect\nB cell rho={rho_cog:.2f}, rest rho={rho_rest:.2f}')
    axes[1, 2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "phase2_fig5_ceiling_effect.png"), dpi=150, bbox_inches='tight')
plt.close()

# Save results
df_ceiling.to_csv(os.path.join(IMPL_DIR, "phase2_ceiling_analysis.csv"), index=False)
df_decomp.to_csv(os.path.join(IMPL_DIR, "phase2_variance_decomposition.csv"), index=False)
if len(df_quantile) > 0:
    df_quantile.to_csv(os.path.join(IMPL_DIR, "phase2_quantile_stratified.csv"), index=False)

print("\nPhase 2 ceiling effect analysis complete!")
