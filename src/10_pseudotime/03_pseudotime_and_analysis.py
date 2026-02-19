"""
Step 2: Compute pseudotime, define TF-target pairs, and run directionality audit.

This script:
1. Loads immune subset data
2. Focuses on T-cell and B-cell lineages (clear differentiation trajectories)
3. Computes diffusion pseudotime
4. Defines known TF-target regulatory pairs for immune lineages
5. Performs lagged correlation analysis
6. Computes directionality consistency metric
7. Runs statistical null model
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

np.random.seed(42)

OUT_DIR = '/Users/ihorkendiukhov/biodyn-work/subproject_29_pseudotime_directionality_audit'
FIG_DIR = os.path.join(OUT_DIR, 'reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("1. Loading data...")
adata = sc.read_h5ad('/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad')

# Create gene name lookup
gene_name_to_ensembl = dict(zip(adata.var['feature_name'], adata.var_names))
ensembl_to_gene_name = dict(zip(adata.var_names, adata.var['feature_name']))

def get_expr(adata, gene_name):
    """Get expression vector for a gene by its symbol."""
    if gene_name in gene_name_to_ensembl:
        ens = gene_name_to_ensembl[gene_name]
        if ens in adata.var_names:
            idx = list(adata.var_names).index(ens)
            x = adata.X[:, idx]
            if sparse.issparse(x):
                return np.asarray(x.todense()).flatten()
            return np.asarray(x).flatten()
    return None

# ============================================================
# 2. DEFINE LINEAGE SUBSETS AND COMPUTE PSEUDOTIME
# ============================================================
print("=" * 60)
print("2. Defining lineage subsets...")

# We'll analyze three distinct lineages with known differentiation trajectories:
# (A) T cell differentiation: naive -> effector/memory CD4+ and CD8+ T cells
# (B) B cell differentiation: B cell -> plasma cell
# (C) Monocyte-Macrophage lineage: monocyte -> macrophage

lineage_definitions = {
    'T_cell': {
        'cell_types': [
            'naive thymus-derived CD4-positive, alpha-beta T cell',
            'CD4-positive, alpha-beta T cell',
            'CD8-positive, alpha-beta T cell',
            'regulatory T cell',
            'gamma-delta T cell',
            'mature NK T cell',
            'T cell',
            'CD4-positive, alpha-beta thymocyte',
        ],
        'root_type': 'naive thymus-derived CD4-positive, alpha-beta T cell',
    },
    'B_cell': {
        'cell_types': [
            'B cell',
            'plasma cell',
        ],
        'root_type': 'B cell',
    },
    'Myeloid': {
        'cell_types': [
            'monocyte',
            'classical monocyte',
            'intermediate monocyte',
            'macrophage',
            'mononuclear phagocyte',
        ],
        'root_type': 'classical monocyte',
    },
}

def compute_pseudotime(adata_full, cell_types, root_type, lineage_name):
    """Subset to lineage, recompute neighbors, compute diffusion pseudotime."""
    mask = adata_full.obs['cell_type'].isin(cell_types)
    adata_sub = adata_full[mask].copy()
    print(f"\n  {lineage_name}: {adata_sub.n_obs} cells, {len(cell_types)} cell types")
    print(f"  Cell type distribution:\n{adata_sub.obs['cell_type'].value_counts()}")

    if adata_sub.n_obs < 100:
        print(f"  Skipping {lineage_name}: too few cells")
        return None

    # Use highly variable genes for better signal
    sc.pp.highly_variable_genes(adata_sub, n_top_genes=2000, flavor='seurat')

    # Re-compute PCA on subset
    sc.tl.pca(adata_sub, n_comps=30)

    # Compute neighbors
    sc.pp.neighbors(adata_sub, n_neighbors=15, n_pcs=30)

    # Compute diffusion map
    sc.tl.diffmap(adata_sub, n_comps=10)

    # Find root cell (cell of root_type closest to centroid of root_type cells)
    root_mask = adata_sub.obs['cell_type'] == root_type
    if root_mask.sum() == 0:
        # fallback: use first cell type
        root_mask = adata_sub.obs['cell_type'] == cell_types[0]

    root_cells = adata_sub[root_mask]
    root_pca = root_cells.obsm['X_pca']
    centroid = root_pca.mean(axis=0)
    dists = np.linalg.norm(root_pca - centroid, axis=1)
    root_idx_in_subset = np.where(root_mask.values)[0][np.argmin(dists)]

    adata_sub.uns['iroot'] = root_idx_in_subset

    # Compute DPT
    sc.tl.dpt(adata_sub, n_branchings=0)

    # Recompute UMAP for visualization
    sc.tl.umap(adata_sub)

    print(f"  DPT range: [{adata_sub.obs['dpt_pseudotime'].min():.3f}, {adata_sub.obs['dpt_pseudotime'].max():.3f}]")
    print(f"  Median DPT by cell type:")
    for ct in cell_types:
        ct_mask = adata_sub.obs['cell_type'] == ct
        if ct_mask.sum() > 0:
            med = adata_sub.obs.loc[ct_mask, 'dpt_pseudotime'].median()
            print(f"    {ct}: {med:.3f} (n={ct_mask.sum()})")

    return adata_sub

lineage_data = {}
for name, defn in lineage_definitions.items():
    print(f"\nProcessing {name} lineage...")
    result = compute_pseudotime(adata, defn['cell_types'], defn['root_type'], name)
    if result is not None:
        lineage_data[name] = result

# ============================================================
# 3. DEFINE TF-TARGET REGULATORY PAIRS
# ============================================================
print("\n" + "=" * 60)
print("3. Defining TF-target regulatory pairs...")

# Well-characterized immune TF-target relationships from literature
# Format: (TF, target, lineage, direction, reference)
# direction: 'activate' or 'repress'
tf_target_pairs = [
    # T cell lineage
    ('TCF7', 'LEF1', 'T_cell', 'activate', 'Tcf1 activates Lef1 in early T cells'),
    ('TCF7', 'CCR7', 'T_cell', 'activate', 'Tcf1 maintains naive T cell homing'),
    ('TCF7', 'IL7R', 'T_cell', 'activate', 'Tcf1 drives IL-7R expression'),
    ('TBX21', 'IFNG', 'T_cell', 'activate', 'T-bet drives IFN-gamma in Th1'),
    ('TBX21', 'CXCR3', 'T_cell', 'activate', 'T-bet induces CXCR3'),
    ('TBX21', 'GZMB', 'T_cell', 'activate', 'T-bet drives granzyme B in CD8+'),
    ('GATA3', 'IL4', 'T_cell', 'activate', 'GATA3 drives IL-4 in Th2'),
    ('GATA3', 'IL13', 'T_cell', 'activate', 'GATA3 drives IL-13'),
    ('FOXP3', 'IL2RA', 'T_cell', 'activate', 'Foxp3 drives CD25 in Tregs'),
    ('FOXP3', 'CTLA4', 'T_cell', 'activate', 'Foxp3 drives CTLA-4'),
    ('EOMES', 'PRF1', 'T_cell', 'activate', 'Eomes drives perforin'),
    ('EOMES', 'GZMB', 'T_cell', 'activate', 'Eomes drives granzyme B'),
    ('RUNX3', 'CD8A', 'T_cell', 'activate', 'Runx3 maintains CD8 expression'),
    ('RUNX3', 'GZMB', 'T_cell', 'activate', 'Runx3 drives granzyme B'),
    ('BATF', 'IL10', 'T_cell', 'activate', 'BATF promotes IL-10 in T cells'),
    ('RORC', 'IL17A', 'T_cell', 'activate', 'RORgamma-t drives IL-17'),
    ('RORC', 'CCR6', 'T_cell', 'activate', 'RORgamma-t drives CCR6 in Th17'),
    ('ID2', 'GZMB', 'T_cell', 'activate', 'Id2 in effector CD8+ T cells'),
    ('MAF', 'IL4', 'T_cell', 'activate', 'c-Maf co-regulates IL-4'),
    ('STAT4', 'TBX21', 'T_cell', 'activate', 'STAT4 induces T-bet'),
    ('STAT6', 'GATA3', 'T_cell', 'activate', 'STAT6 induces GATA3'),
    ('IRF4', 'IL4', 'T_cell', 'activate', 'IRF4 co-regulates IL-4'),
    ('MYC', 'CDK4', 'T_cell', 'activate', 'Myc drives cell cycle genes'),
    ('MYC', 'MCM7', 'T_cell', 'activate', 'Myc drives replication genes'),
    ('NFKB1', 'BCL2', 'T_cell', 'activate', 'NF-kB drives anti-apoptotic genes'),

    # B cell lineage
    ('PAX5', 'CD19', 'B_cell', 'activate', 'Pax5 drives CD19 in B cells'),
    ('PAX5', 'CD79A', 'B_cell', 'activate', 'Pax5 drives CD79a/Ig-alpha'),
    ('PAX5', 'MS4A1', 'B_cell', 'activate', 'Pax5 drives CD20'),
    ('EBF1', 'CD79A', 'B_cell', 'activate', 'EBF1 drives Ig-alpha'),
    ('EBF1', 'PAX5', 'B_cell', 'activate', 'EBF1 activates PAX5'),
    ('IRF4', 'PRDM1', 'B_cell', 'activate', 'IRF4 induces Blimp-1'),
    ('PRDM1', 'XBP1', 'B_cell', 'activate', 'Blimp-1 drives XBP1 for plasma cell'),
    ('PRDM1', 'SDC1', 'B_cell', 'activate', 'Blimp-1 drives CD138'),
    ('PRDM1', 'PAX5', 'B_cell', 'repress', 'Blimp-1 represses PAX5'),
    ('BCL6', 'PRDM1', 'B_cell', 'repress', 'BCL6 represses Blimp-1'),
    ('BCL6', 'TP53', 'B_cell', 'repress', 'BCL6 represses p53 in GC'),
    ('IRF8', 'BCL6', 'B_cell', 'activate', 'IRF8 cooperates with BCL6'),
    ('MYC', 'CDK4', 'B_cell', 'activate', 'Myc drives cell cycle in B cells'),
    ('XBP1', 'JCHAIN', 'B_cell', 'activate', 'XBP1 drives Ig secretion'),
    ('NFKB1', 'BCL2', 'B_cell', 'activate', 'NF-kB survival signaling in B cells'),

    # Myeloid lineage
    ('SPI1', 'CSF1R', 'Myeloid', 'activate', 'PU.1 drives CSF1R in macrophages'),
    ('SPI1', 'CD68', 'Myeloid', 'activate', 'PU.1 drives CD68'),
    ('SPI1', 'ITGAM', 'Myeloid', 'activate', 'PU.1 drives CD11b'),
    ('CEBPA', 'CSF3R', 'Myeloid', 'activate', 'C/EBPa drives G-CSF receptor'),
    ('CEBPB', 'IL6', 'Myeloid', 'activate', 'C/EBPb drives IL-6'),
    ('CEBPB', 'TNF', 'Myeloid', 'activate', 'C/EBPb drives TNF'),
    ('IRF8', 'IL12B', 'Myeloid', 'activate', 'IRF8 drives IL-12 in macrophages'),
    ('IRF4', 'CCL17', 'Myeloid', 'activate', 'IRF4 in M2 macrophages'),
    ('STAT1', 'IRF1', 'Myeloid', 'activate', 'STAT1 drives IRF1'),
    ('STAT1', 'GBP1', 'Myeloid', 'activate', 'STAT1 drives GBP1'),
    ('NFKB1', 'IL1B', 'Myeloid', 'activate', 'NF-kB drives IL-1beta'),
    ('NFKB1', 'TNF', 'Myeloid', 'activate', 'NF-kB drives TNF'),
    ('RELA', 'CXCL8', 'Myeloid', 'activate', 'RelA drives IL-8'),
    ('MAF', 'MRC1', 'Myeloid', 'activate', 'c-Maf drives CD206 in M2'),
    ('KLF4', 'MRC1', 'Myeloid', 'activate', 'KLF4 drives M2 polarization'),
    ('JUN', 'MMP9', 'Myeloid', 'activate', 'AP-1/Jun drives MMP9'),
    ('FOS', 'MMP9', 'Myeloid', 'activate', 'AP-1/Fos drives MMP9'),
    ('ETS1', 'CSF2', 'Myeloid', 'activate', 'ETS1 drives GM-CSF'),
]

print(f"Defined {len(tf_target_pairs)} TF-target pairs across lineages")

# ============================================================
# 4. EXPRESSION EXTRACTION AND PSEUDOTIME BINNING
# ============================================================
print("\n" + "=" * 60)
print("4. Extracting expression and binning along pseudotime...")

N_BINS = 50  # Number of pseudotime bins

def bin_expression_along_pseudotime(adata_sub, gene_name, n_bins=N_BINS):
    """Bin cells along pseudotime and compute mean expression per bin."""
    pt = adata_sub.obs['dpt_pseudotime'].values

    # Remove inf values
    valid = np.isfinite(pt)
    pt_valid = pt[valid]

    expr = get_expr_from_subset(adata_sub, gene_name)
    if expr is None:
        return None, None
    expr_valid = expr[valid]

    # Create bins
    bin_edges = np.linspace(pt_valid.min(), pt_valid.max() + 1e-10, n_bins + 1)
    bin_indices = np.digitize(pt_valid, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_means = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_means[b] = expr_valid[mask].mean()
            bin_counts[b] = mask.sum()

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, bin_means, bin_counts

def get_expr_from_subset(adata_sub, gene_name):
    """Get expression for a gene from a subset adata."""
    # Gene names stored in feature_name
    mask = adata_sub.var['feature_name'] == gene_name
    if mask.sum() == 0:
        return None
    idx = np.where(mask.values)[0][0]
    x = adata_sub.X[:, idx]
    if sparse.issparse(x):
        return np.asarray(x.todense()).flatten()
    return np.asarray(x).flatten()

# ============================================================
# 5. LAGGED CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("5. Running lagged correlation analysis...")

MAX_LAG = 10  # Max lag in bins

def compute_lagged_correlations(tf_bins, target_bins, max_lag=MAX_LAG):
    """
    Compute cross-correlation between TF and target at different lags.
    Positive lag = TF leads (TF signal at earlier pseudotime correlates with
    target signal at later pseudotime).
    """
    n = len(tf_bins)
    lags = range(-max_lag, max_lag + 1)
    corrs = []

    for lag in lags:
        if lag >= 0:
            # TF leads: compare TF[:-lag] with target[lag:]
            if lag == 0:
                tf_slice = tf_bins
                tgt_slice = target_bins
            else:
                tf_slice = tf_bins[:n-lag]
                tgt_slice = target_bins[lag:]
        else:
            # Target leads: compare TF[-lag:] with target[:lag]
            tf_slice = tf_bins[-lag:]
            tgt_slice = target_bins[:n+lag]

        if len(tf_slice) < 5:
            corrs.append(np.nan)
            continue

        # Use Spearman correlation
        if np.std(tf_slice) == 0 or np.std(tgt_slice) == 0:
            corrs.append(0.0)
        else:
            r, _ = spearmanr(tf_slice, tgt_slice)
            corrs.append(r)

    return list(lags), corrs

def find_peak_lag(lags, corrs, direction='activate'):
    """Find the lag with peak correlation (or anti-correlation for repression)."""
    lags = np.array(lags)
    corrs = np.array(corrs)
    valid = ~np.isnan(corrs)

    if not valid.any():
        return 0, 0.0

    if direction == 'activate':
        # For activation: peak positive correlation
        best_idx = np.nanargmax(corrs)
    else:
        # For repression: peak negative correlation
        best_idx = np.nanargmin(corrs)

    return lags[best_idx], corrs[best_idx]

# Run analysis for all TF-target pairs
results = []

for tf_name, target_name, lineage, direction, description in tf_target_pairs:
    if lineage not in lineage_data:
        continue

    adata_sub = lineage_data[lineage]

    # Get binned expression
    tf_result = bin_expression_along_pseudotime(adata_sub, tf_name)
    tgt_result = bin_expression_along_pseudotime(adata_sub, target_name)

    if tf_result[0] is None or tgt_result[0] is None:
        continue

    tf_centers, tf_bins_expr, tf_counts = tf_result
    tgt_centers, tgt_bins_expr, tgt_counts = tgt_result

    # Check that genes are expressed (at least some signal)
    if np.max(tf_bins_expr) < 0.01 or np.max(tgt_bins_expr) < 0.01:
        continue

    # Compute lagged correlations
    lags, corrs = compute_lagged_correlations(tf_bins_expr, tgt_bins_expr)

    # Find peak lag
    peak_lag, peak_corr = find_peak_lag(lags, corrs, direction)

    # Determine if directionality is consistent
    # For activation: TF should lead (peak at positive lag) with positive correlation
    # For repression: TF should lead (peak at positive lag) with negative correlation
    if direction == 'activate':
        directional_consistent = peak_lag > 0 and peak_corr > 0
    else:
        directional_consistent = peak_lag > 0 and peak_corr < 0

    # Also compute TF peak pseudotime vs target peak pseudotime
    tf_peak_pt = tf_centers[np.argmax(tf_bins_expr)]
    tgt_peak_pt = tgt_centers[np.argmax(tgt_bins_expr)]
    tf_leads = tf_peak_pt < tgt_peak_pt

    # Zero-lag correlation
    zero_lag_corr = corrs[lags.index(0)]

    results.append({
        'TF': tf_name,
        'Target': target_name,
        'Lineage': lineage,
        'Direction': direction,
        'Description': description,
        'Peak_Lag': peak_lag,
        'Peak_Corr': peak_corr,
        'Zero_Lag_Corr': zero_lag_corr,
        'Directional_Consistent': directional_consistent,
        'TF_Peak_PT': tf_peak_pt,
        'Target_Peak_PT': tgt_peak_pt,
        'TF_Leads_by_Peak': tf_leads,
        'Lags': lags,
        'Correlations': corrs,
        'TF_Binned_Expr': tf_bins_expr.tolist(),
        'Target_Binned_Expr': tgt_bins_expr.tolist(),
        'Bin_Centers': tf_centers.tolist(),
    })

results_df = pd.DataFrame(results)
print(f"\nAnalyzed {len(results_df)} TF-target pairs successfully")
print(f"\nResults by lineage:")
for lineage in results_df['Lineage'].unique():
    sub = results_df[results_df['Lineage'] == lineage]
    n_consistent = sub['Directional_Consistent'].sum()
    print(f"  {lineage}: {n_consistent}/{len(sub)} directionally consistent ({n_consistent/len(sub)*100:.1f}%)")

print(f"\nOverall directionality consistency (lag-based): {results_df['Directional_Consistent'].mean()*100:.1f}%")
print(f"Overall TF-leads-by-peak: {results_df['TF_Leads_by_Peak'].mean()*100:.1f}%")

# ============================================================
# 6. DIRECTIONALITY CONSISTENCY METRIC
# ============================================================
print("\n" + "=" * 60)
print("6. Computing formal directionality metrics...")

def compute_directionality_score(row):
    """
    Directionality score: measures how strongly TF changes precede target changes.
    Range: [-1, 1] where positive = TF leads, negative = target leads.
    """
    lags = np.array(row['Lags'])
    corrs = np.array(row['Correlations'])

    if row['Direction'] == 'activate':
        # Weight by correlation strength
        pos_lag_mask = lags > 0
        neg_lag_mask = lags < 0

        pos_strength = np.nanmean(corrs[pos_lag_mask]) if pos_lag_mask.any() else 0
        neg_strength = np.nanmean(corrs[neg_lag_mask]) if neg_lag_mask.any() else 0

        score = pos_strength - neg_strength
    else:
        # For repression, expect negative correlation with positive lag
        pos_lag_mask = lags > 0
        neg_lag_mask = lags < 0

        pos_strength = np.nanmean(-corrs[pos_lag_mask]) if pos_lag_mask.any() else 0
        neg_strength = np.nanmean(-corrs[neg_lag_mask]) if neg_lag_mask.any() else 0

        score = pos_strength - neg_strength

    return score

results_df['Directionality_Score'] = results_df.apply(compute_directionality_score, axis=1)

print(f"Mean directionality score: {results_df['Directionality_Score'].mean():.4f}")
print(f"Median directionality score: {results_df['Directionality_Score'].median():.4f}")
print(f"Fraction with positive score (TF leads): {(results_df['Directionality_Score'] > 0).mean()*100:.1f}%")

# ============================================================
# 7. NULL MODEL: SHUFFLED PSEUDOTIME & RANDOM PAIRS
# ============================================================
print("\n" + "=" * 60)
print("7. Computing null model statistics...")

N_PERMUTATIONS = 1000

# Null model 1: Shuffled pseudotime
null_scores_shuffled = []
for perm_i in range(N_PERMUTATIONS):
    perm_scores = []
    for idx, row in results_df.iterrows():
        lineage = row['Lineage']
        adata_sub = lineage_data[lineage]

        # Shuffle pseudotime
        pt = adata_sub.obs['dpt_pseudotime'].values.copy()
        valid = np.isfinite(pt)
        pt_shuffled = pt.copy()
        pt_shuffled[valid] = np.random.permutation(pt_shuffled[valid])

        # Re-bin with shuffled pseudotime
        tf_expr = get_expr_from_subset(adata_sub, row['TF'])
        tgt_expr = get_expr_from_subset(adata_sub, row['Target'])

        if tf_expr is None or tgt_expr is None:
            continue

        tf_valid = tf_expr[valid]
        tgt_valid = tgt_expr[valid]
        pt_valid = pt_shuffled[valid]

        n_bins = N_BINS
        bin_edges = np.linspace(pt_valid.min(), pt_valid.max() + 1e-10, n_bins + 1)
        bin_indices = np.digitize(pt_valid, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        tf_bins = np.array([tf_valid[bin_indices == b].mean() if (bin_indices == b).sum() > 0 else 0 for b in range(n_bins)])
        tgt_bins = np.array([tgt_valid[bin_indices == b].mean() if (bin_indices == b).sum() > 0 else 0 for b in range(n_bins)])

        lags, corrs = compute_lagged_correlations(tf_bins, tgt_bins)

        lags_arr = np.array(lags)
        corrs_arr = np.array(corrs)
        pos_mask = lags_arr > 0
        neg_mask = lags_arr < 0

        if row['Direction'] == 'activate':
            pos_s = np.nanmean(corrs_arr[pos_mask]) if pos_mask.any() else 0
            neg_s = np.nanmean(corrs_arr[neg_mask]) if neg_mask.any() else 0
        else:
            pos_s = np.nanmean(-corrs_arr[pos_mask]) if pos_mask.any() else 0
            neg_s = np.nanmean(-corrs_arr[neg_mask]) if neg_mask.any() else 0

        perm_scores.append(pos_s - neg_s)

    if perm_scores:
        null_scores_shuffled.append(np.mean(perm_scores))

null_scores_shuffled = np.array(null_scores_shuffled)
observed_mean = results_df['Directionality_Score'].mean()
p_shuffled = (np.sum(null_scores_shuffled >= observed_mean) + 1) / (N_PERMUTATIONS + 1)

print(f"Observed mean directionality score: {observed_mean:.4f}")
print(f"Null (shuffled PT) mean: {null_scores_shuffled.mean():.4f} +/- {null_scores_shuffled.std():.4f}")
print(f"P-value (shuffled pseudotime): {p_shuffled:.4f}")
print(f"Effect size (Cohen's d): {(observed_mean - null_scores_shuffled.mean()) / null_scores_shuffled.std():.3f}")

# Null model 2: Random gene pairs (non-TF-target)
print("\nNull model 2: Random gene pairs...")
np.random.seed(123)
all_genes = adata.var['feature_name'].values
# Filter to expressed genes
expressed_mask = np.array((adata.X > 0).mean(axis=0)).flatten() > 0.01
expressed_genes = all_genes[expressed_mask]

null_scores_random = []
for perm_i in range(200):  # Fewer perms for random pairs (more expensive)
    pair_scores = []
    for lineage in lineage_data:
        adata_sub = lineage_data[lineage]
        # Pick random gene pairs
        for _ in range(10):
            g1, g2 = np.random.choice(expressed_genes, 2, replace=False)

            r1 = bin_expression_along_pseudotime(adata_sub, g1)
            r2 = bin_expression_along_pseudotime(adata_sub, g2)

            if r1[0] is None or r2[0] is None:
                continue
            if np.max(r1[1]) < 0.01 or np.max(r2[1]) < 0.01:
                continue

            lags, corrs = compute_lagged_correlations(r1[1], r2[1])
            lags_arr = np.array(lags)
            corrs_arr = np.array(corrs)
            pos_mask = lags_arr > 0
            neg_mask = lags_arr < 0
            pos_s = np.nanmean(corrs_arr[pos_mask]) if pos_mask.any() else 0
            neg_s = np.nanmean(corrs_arr[neg_mask]) if neg_mask.any() else 0
            pair_scores.append(pos_s - neg_s)

    if pair_scores:
        null_scores_random.append(np.mean(pair_scores))

null_scores_random = np.array(null_scores_random)
p_random = (np.sum(null_scores_random >= observed_mean) + 1) / (len(null_scores_random) + 1)

print(f"Null (random pairs) mean: {null_scores_random.mean():.4f} +/- {null_scores_random.std():.4f}")
print(f"P-value (random pairs): {p_random:.4f}")

# Per-pair p-values using shuffled pseudotime
print("\nComputing per-pair p-values...")
pair_null_scores = {idx: [] for idx in results_df.index}
for perm_i in range(500):
    for idx, row in results_df.iterrows():
        lineage = row['Lineage']
        adata_sub = lineage_data[lineage]

        pt = adata_sub.obs['dpt_pseudotime'].values.copy()
        valid = np.isfinite(pt)
        pt_shuffled = pt.copy()
        pt_shuffled[valid] = np.random.permutation(pt_shuffled[valid])

        tf_expr = get_expr_from_subset(adata_sub, row['TF'])
        tgt_expr = get_expr_from_subset(adata_sub, row['Target'])

        if tf_expr is None or tgt_expr is None:
            continue

        tf_valid = tf_expr[valid]
        tgt_valid = tgt_expr[valid]
        pt_valid = pt_shuffled[valid]

        n_bins = N_BINS
        bin_edges = np.linspace(pt_valid.min(), pt_valid.max() + 1e-10, n_bins + 1)
        bin_indices = np.digitize(pt_valid, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        tf_bins = np.array([tf_valid[bin_indices == b].mean() if (bin_indices == b).sum() > 0 else 0 for b in range(n_bins)])
        tgt_bins = np.array([tgt_valid[bin_indices == b].mean() if (bin_indices == b).sum() > 0 else 0 for b in range(n_bins)])

        lags, corrs = compute_lagged_correlations(tf_bins, tgt_bins)
        lags_arr = np.array(lags)
        corrs_arr = np.array(corrs)
        pos_mask = lags_arr > 0
        neg_mask = lags_arr < 0

        if row['Direction'] == 'activate':
            pos_s = np.nanmean(corrs_arr[pos_mask]) if pos_mask.any() else 0
            neg_s = np.nanmean(corrs_arr[neg_mask]) if neg_mask.any() else 0
        else:
            pos_s = np.nanmean(-corrs_arr[pos_mask]) if pos_mask.any() else 0
            neg_s = np.nanmean(-corrs_arr[neg_mask]) if neg_mask.any() else 0

        pair_null_scores[idx].append(pos_s - neg_s)

# Compute per-pair p-values
pair_pvalues = {}
for idx, row in results_df.iterrows():
    null = np.array(pair_null_scores[idx])
    if len(null) > 0:
        obs = row['Directionality_Score']
        p = (np.sum(null >= obs) + 1) / (len(null) + 1)
        pair_pvalues[idx] = p
    else:
        pair_pvalues[idx] = 1.0

results_df['P_Value'] = results_df.index.map(pair_pvalues)

# BH correction
from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(results_df['P_Value'].values, method='fdr_bh')
results_df['P_Value_BH'] = pvals_corrected
results_df['Significant_BH'] = reject

print(f"\nSignificant pairs (FDR < 0.05): {reject.sum()}/{len(reject)}")
print(f"Significant and directionally consistent: {(reject & results_df['Directional_Consistent'].values).sum()}/{len(reject)}")

# ============================================================
# 8. FAILURE CASE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("8. Failure case analysis...")

failed = results_df[~results_df['Directional_Consistent']]
print(f"\nDirectionally inconsistent pairs ({len(failed)}):")
for _, row in failed.iterrows():
    print(f"  {row['TF']} -> {row['Target']} ({row['Lineage']}, {row['Direction']})")
    print(f"    Peak lag={row['Peak_Lag']}, Peak corr={row['Peak_Corr']:.3f}")
    print(f"    TF peak PT={row['TF_Peak_PT']:.3f}, Target peak PT={row['Target_Peak_PT']:.3f}")
    print(f"    Dir score={row['Directionality_Score']:.4f}, p={row['P_Value']:.4f}")

# Categorize failure modes
failure_categories = []
for _, row in failed.iterrows():
    if abs(row['Peak_Corr']) < 0.1:
        cat = 'Weak/No correlation'
    elif row['Peak_Lag'] < 0:
        cat = 'Target leads TF'
    elif row['Peak_Lag'] == 0:
        cat = 'Simultaneous'
    elif row['Direction'] == 'activate' and row['Peak_Corr'] < 0:
        cat = 'Wrong sign (anticorrelated)'
    elif row['Direction'] == 'repress' and row['Peak_Corr'] > 0:
        cat = 'Wrong sign (positively correlated)'
    else:
        cat = 'Other'
    failure_categories.append(cat)

if failure_categories:
    failed_cats = pd.Series(failure_categories)
    print(f"\nFailure categories:")
    for cat, count in failed_cats.value_counts().items():
        print(f"  {cat}: {count}")

# ============================================================
# 9. SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("9. Saving results...")

# Save main results
results_save = results_df.drop(columns=['Lags', 'Correlations', 'TF_Binned_Expr', 'Target_Binned_Expr', 'Bin_Centers'])
results_save.to_csv(os.path.join(OUT_DIR, 'implementation', 'directionality_results.csv'), index=False)

# Save full results with lag data for plotting
with open(os.path.join(OUT_DIR, 'implementation', 'full_results.pkl'), 'wb') as f:
    pickle.dump({
        'results_df': results_df,
        'null_scores_shuffled': null_scores_shuffled,
        'null_scores_random': null_scores_random,
        'pair_null_scores': pair_null_scores,
        'lineage_definitions': lineage_definitions,
    }, f)

# Save lineage adata for plotting
for name, adata_sub in lineage_data.items():
    adata_sub.write(os.path.join(OUT_DIR, 'implementation', f'adata_{name}.h5ad'))

print("Results saved.")

# Print summary table
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
summary_cols = ['TF', 'Target', 'Lineage', 'Direction', 'Peak_Lag', 'Peak_Corr',
                'Directionality_Score', 'Directional_Consistent', 'P_Value_BH']
print(results_df[summary_cols].to_string(index=False))

print(f"\n\nOVERALL STATISTICS:")
print(f"Total pairs analyzed: {len(results_df)}")
print(f"Directionally consistent (lag-based): {results_df['Directional_Consistent'].sum()}/{len(results_df)} ({results_df['Directional_Consistent'].mean()*100:.1f}%)")
print(f"TF leads by peak position: {results_df['TF_Leads_by_Peak'].sum()}/{len(results_df)} ({results_df['TF_Leads_by_Peak'].mean()*100:.1f}%)")
print(f"Significant (FDR<0.05): {results_df['Significant_BH'].sum()}/{len(results_df)}")
print(f"Mean directionality score: {results_df['Directionality_Score'].mean():.4f}")
print(f"P-value vs shuffled PT: {p_shuffled:.4f}")
print(f"P-value vs random pairs: {p_random:.4f}")
