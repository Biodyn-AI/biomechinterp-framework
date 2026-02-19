"""
Step 2b: Lagged correlation analysis and directionality audit.
Loads pre-computed pseudotime data from 02a.
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
import pickle

np.random.seed(42)

OUT_DIR = '/Users/ihorkendiukhov/biodyn-work/subproject_29_pseudotime_directionality_audit'

# ============================================================
# LOAD LINEAGE DATA
# ============================================================
print("Loading lineage data...")
lineage_data = {}
for name in ['T_cell', 'B_cell', 'Myeloid']:
    path = os.path.join(OUT_DIR, 'implementation', f'adata_{name}.h5ad')
    lineage_data[name] = sc.read_h5ad(path)
    print(f"  {name}: {lineage_data[name].n_obs} cells")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
N_BINS = 50
MAX_LAG = 10

def get_expr(adata_sub, gene_name):
    """Get expression for a gene by symbol."""
    mask = adata_sub.var['feature_name'] == gene_name
    if mask.sum() == 0:
        return None
    idx = np.where(mask.values)[0][0]
    x = adata_sub.X[:, idx]
    if sparse.issparse(x):
        return np.asarray(x.todense()).flatten()
    return np.asarray(x).flatten()

def bin_expression(adata_sub, gene_name, pt_values=None, n_bins=N_BINS):
    """Bin cells along pseudotime, return (bin_centers, bin_means, bin_counts)."""
    if pt_values is None:
        pt = adata_sub.obs['dpt_pseudotime'].values
    else:
        pt = pt_values

    valid = np.isfinite(pt)
    pt_v = pt[valid]

    expr = get_expr(adata_sub, gene_name)
    if expr is None:
        return None, None, None
    expr_v = expr[valid]

    bin_edges = np.linspace(pt_v.min(), pt_v.max() + 1e-10, n_bins + 1)
    bin_idx = np.clip(np.digitize(pt_v, bin_edges) - 1, 0, n_bins - 1)

    bin_means = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() > 0:
            bin_means[b] = expr_v[m].mean()
            bin_counts[b] = m.sum()

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, bin_means, bin_counts

def bin_expression_fast(expr, pt, n_bins=N_BINS):
    """Fast binning from pre-extracted expression and pseudotime arrays."""
    valid = np.isfinite(pt)
    pt_v = pt[valid]
    expr_v = expr[valid]
    bin_edges = np.linspace(pt_v.min(), pt_v.max() + 1e-10, n_bins + 1)
    bin_idx = np.clip(np.digitize(pt_v, bin_edges) - 1, 0, n_bins - 1)
    bin_means = np.zeros(n_bins)
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() > 0:
            bin_means[b] = expr_v[m].mean()
    return bin_means

def compute_lagged_correlations(tf_bins, target_bins, max_lag=MAX_LAG):
    """Cross-correlation at different lags. Positive lag = TF leads."""
    n = len(tf_bins)
    lags = list(range(-max_lag, max_lag + 1))
    corrs = []
    for lag in lags:
        if lag == 0:
            tf_s, tgt_s = tf_bins, target_bins
        elif lag > 0:
            tf_s = tf_bins[:n-lag]
            tgt_s = target_bins[lag:]
        else:
            tf_s = tf_bins[-lag:]
            tgt_s = target_bins[:n+lag]
        if len(tf_s) < 5 or np.std(tf_s) == 0 or np.std(tgt_s) == 0:
            corrs.append(0.0)
        else:
            r, _ = spearmanr(tf_s, tgt_s)
            corrs.append(r if not np.isnan(r) else 0.0)
    return lags, corrs

def directionality_score(lags, corrs, direction='activate'):
    """Score: positive if TF leads target in expected direction."""
    lags_arr = np.array(lags)
    corrs_arr = np.array(corrs)
    pos_mask = lags_arr > 0
    neg_mask = lags_arr < 0
    if direction == 'activate':
        pos_s = np.mean(corrs_arr[pos_mask]) if pos_mask.any() else 0
        neg_s = np.mean(corrs_arr[neg_mask]) if neg_mask.any() else 0
    else:
        pos_s = np.mean(-corrs_arr[pos_mask]) if pos_mask.any() else 0
        neg_s = np.mean(-corrs_arr[neg_mask]) if neg_mask.any() else 0
    return pos_s - neg_s

# ============================================================
# TF-TARGET PAIRS
# ============================================================
tf_target_pairs = [
    # T cell
    ('TCF7', 'LEF1', 'T_cell', 'activate', 'Tcf1 activates Lef1'),
    ('TCF7', 'CCR7', 'T_cell', 'activate', 'Tcf1 maintains naive homing'),
    ('TCF7', 'IL7R', 'T_cell', 'activate', 'Tcf1 drives IL-7R'),
    ('TBX21', 'IFNG', 'T_cell', 'activate', 'T-bet drives IFN-gamma'),
    ('TBX21', 'CXCR3', 'T_cell', 'activate', 'T-bet induces CXCR3'),
    ('TBX21', 'GZMB', 'T_cell', 'activate', 'T-bet drives granzyme B'),
    ('GATA3', 'IL4', 'T_cell', 'activate', 'GATA3 drives IL-4'),
    ('GATA3', 'IL13', 'T_cell', 'activate', 'GATA3 drives IL-13'),
    ('FOXP3', 'IL2RA', 'T_cell', 'activate', 'Foxp3 drives CD25'),
    ('FOXP3', 'CTLA4', 'T_cell', 'activate', 'Foxp3 drives CTLA-4'),
    ('EOMES', 'PRF1', 'T_cell', 'activate', 'Eomes drives perforin'),
    ('EOMES', 'GZMB', 'T_cell', 'activate', 'Eomes drives granzyme B'),
    ('RUNX3', 'CD8A', 'T_cell', 'activate', 'Runx3 maintains CD8'),
    ('RUNX3', 'GZMB', 'T_cell', 'activate', 'Runx3 drives granzyme B'),
    ('BATF', 'IL10', 'T_cell', 'activate', 'BATF promotes IL-10'),
    ('RORC', 'IL17A', 'T_cell', 'activate', 'RORgt drives IL-17'),
    ('RORC', 'CCR6', 'T_cell', 'activate', 'RORgt drives CCR6'),
    ('ID2', 'GZMB', 'T_cell', 'activate', 'Id2 in effector CD8+'),
    ('MAF', 'IL4', 'T_cell', 'activate', 'c-Maf co-regulates IL-4'),
    ('STAT4', 'TBX21', 'T_cell', 'activate', 'STAT4 induces T-bet'),
    ('STAT6', 'GATA3', 'T_cell', 'activate', 'STAT6 induces GATA3'),
    ('IRF4', 'IL4', 'T_cell', 'activate', 'IRF4 co-regulates IL-4'),
    ('MYC', 'CDK4', 'T_cell', 'activate', 'Myc drives cell cycle'),
    ('MYC', 'MCM7', 'T_cell', 'activate', 'Myc drives replication'),
    ('NFKB1', 'BCL2', 'T_cell', 'activate', 'NF-kB drives anti-apoptotic'),
    # B cell
    ('PAX5', 'CD19', 'B_cell', 'activate', 'Pax5 drives CD19'),
    ('PAX5', 'CD79A', 'B_cell', 'activate', 'Pax5 drives CD79a'),
    ('PAX5', 'MS4A1', 'B_cell', 'activate', 'Pax5 drives CD20'),
    ('EBF1', 'CD79A', 'B_cell', 'activate', 'EBF1 drives Ig-alpha'),
    ('EBF1', 'PAX5', 'B_cell', 'activate', 'EBF1 activates PAX5'),
    ('IRF4', 'PRDM1', 'B_cell', 'activate', 'IRF4 induces Blimp-1'),
    ('PRDM1', 'XBP1', 'B_cell', 'activate', 'Blimp-1 drives XBP1'),
    ('PRDM1', 'SDC1', 'B_cell', 'activate', 'Blimp-1 drives CD138'),
    ('PRDM1', 'PAX5', 'B_cell', 'repress', 'Blimp-1 represses PAX5'),
    ('BCL6', 'PRDM1', 'B_cell', 'repress', 'BCL6 represses Blimp-1'),
    ('BCL6', 'TP53', 'B_cell', 'repress', 'BCL6 represses p53'),
    ('IRF8', 'BCL6', 'B_cell', 'activate', 'IRF8 with BCL6'),
    ('MYC', 'CDK4', 'B_cell', 'activate', 'Myc drives cell cycle'),
    ('XBP1', 'JCHAIN', 'B_cell', 'activate', 'XBP1 drives Ig secretion'),
    ('NFKB1', 'BCL2', 'B_cell', 'activate', 'NF-kB survival'),
    # Myeloid
    ('SPI1', 'CSF1R', 'Myeloid', 'activate', 'PU.1 drives CSF1R'),
    ('SPI1', 'CD68', 'Myeloid', 'activate', 'PU.1 drives CD68'),
    ('SPI1', 'ITGAM', 'Myeloid', 'activate', 'PU.1 drives CD11b'),
    ('CEBPA', 'CSF3R', 'Myeloid', 'activate', 'C/EBPa drives G-CSFR'),
    ('CEBPB', 'IL6', 'Myeloid', 'activate', 'C/EBPb drives IL-6'),
    ('CEBPB', 'TNF', 'Myeloid', 'activate', 'C/EBPb drives TNF'),
    ('IRF8', 'IL12B', 'Myeloid', 'activate', 'IRF8 drives IL-12'),
    ('IRF4', 'CCL17', 'Myeloid', 'activate', 'IRF4 in M2 macrophages'),
    ('STAT1', 'IRF1', 'Myeloid', 'activate', 'STAT1 drives IRF1'),
    ('STAT1', 'GBP1', 'Myeloid', 'activate', 'STAT1 drives GBP1'),
    ('NFKB1', 'IL1B', 'Myeloid', 'activate', 'NF-kB drives IL-1b'),
    ('NFKB1', 'TNF', 'Myeloid', 'activate', 'NF-kB drives TNF'),
    ('RELA', 'CXCL8', 'Myeloid', 'activate', 'RelA drives IL-8'),
    ('MAF', 'MRC1', 'Myeloid', 'activate', 'c-Maf drives CD206'),
    ('KLF4', 'MRC1', 'Myeloid', 'activate', 'KLF4 drives M2'),
    ('JUN', 'MMP9', 'Myeloid', 'activate', 'AP-1/Jun drives MMP9'),
    ('FOS', 'MMP9', 'Myeloid', 'activate', 'AP-1/Fos drives MMP9'),
    ('ETS1', 'CSF2', 'Myeloid', 'activate', 'ETS1 drives GM-CSF'),
]

# ============================================================
# MAIN ANALYSIS
# ============================================================
print(f"\nAnalyzing {len(tf_target_pairs)} TF-target pairs...")

results = []
for tf_name, target_name, lineage, direction, description in tf_target_pairs:
    if lineage not in lineage_data:
        continue
    adata_sub = lineage_data[lineage]

    _, tf_bins, _ = bin_expression(adata_sub, tf_name)
    _, tgt_bins, _ = bin_expression(adata_sub, target_name)

    if tf_bins is None or tgt_bins is None:
        continue
    if np.max(tf_bins) < 0.01 or np.max(tgt_bins) < 0.01:
        continue

    lags, corrs = compute_lagged_correlations(tf_bins, tgt_bins)

    # Peak lag
    lags_arr = np.array(lags)
    corrs_arr = np.array(corrs)
    if direction == 'activate':
        best_idx = np.argmax(corrs_arr)
    else:
        best_idx = np.argmin(corrs_arr)
    peak_lag = lags_arr[best_idx]
    peak_corr = corrs_arr[best_idx]

    # Directionality
    if direction == 'activate':
        dc = peak_lag > 0 and peak_corr > 0
    else:
        dc = peak_lag > 0 and peak_corr < 0

    # Peak pseudotime positions
    bin_centers, _, _ = bin_expression(adata_sub, tf_name)
    tf_peak_pt = bin_centers[np.argmax(tf_bins)]
    tgt_peak_pt = bin_centers[np.argmax(tgt_bins)]

    zero_lag_corr = corrs[lags.index(0)]
    ds = directionality_score(lags, corrs, direction)

    results.append({
        'TF': tf_name, 'Target': target_name, 'Lineage': lineage,
        'Direction': direction, 'Description': description,
        'Peak_Lag': int(peak_lag), 'Peak_Corr': peak_corr,
        'Zero_Lag_Corr': zero_lag_corr,
        'Directional_Consistent': dc,
        'TF_Peak_PT': tf_peak_pt, 'Target_Peak_PT': tgt_peak_pt,
        'TF_Leads_by_Peak': tf_peak_pt < tgt_peak_pt,
        'Directionality_Score': ds,
        'Lags': lags, 'Correlations': corrs,
        'TF_Binned': tf_bins.tolist(), 'Target_Binned': tgt_bins.tolist(),
        'Bin_Centers': bin_centers.tolist(),
    })

df = pd.DataFrame(results)
print(f"\nSuccessfully analyzed {len(df)} pairs")

for lin in df['Lineage'].unique():
    sub = df[df['Lineage'] == lin]
    nc = sub['Directional_Consistent'].sum()
    print(f"  {lin}: {nc}/{len(sub)} consistent ({nc/len(sub)*100:.1f}%)")

print(f"\nOverall lag-based consistency: {df['Directional_Consistent'].mean()*100:.1f}%")
print(f"Overall TF-leads-by-peak: {df['TF_Leads_by_Peak'].mean()*100:.1f}%")
print(f"Mean directionality score: {df['Directionality_Score'].mean():.4f}")
print(f"Fraction with positive score: {(df['Directionality_Score'] > 0).mean()*100:.1f}%")

# ============================================================
# NULL MODEL: SHUFFLED PSEUDOTIME (optimized)
# ============================================================
print("\n" + "=" * 60)
print("NULL MODEL: Shuffled pseudotime (500 permutations)...")

N_PERM = 500

# Pre-extract expression vectors for speed
pair_exprs = {}
for idx, row in df.iterrows():
    adata_sub = lineage_data[row['Lineage']]
    tf_expr = get_expr(adata_sub, row['TF'])
    tgt_expr = get_expr(adata_sub, row['Target'])
    pt = adata_sub.obs['dpt_pseudotime'].values.copy()
    pair_exprs[idx] = (tf_expr, tgt_expr, pt)

pair_null_scores = {idx: [] for idx in df.index}
global_null_scores = []

for perm_i in range(N_PERM):
    if perm_i % 100 == 0:
        print(f"  Permutation {perm_i}/{N_PERM}...")
    perm_scores = []
    for idx, row in df.iterrows():
        tf_expr, tgt_expr, pt = pair_exprs[idx]
        # Shuffle pseudotime
        valid = np.isfinite(pt)
        pt_shuf = pt.copy()
        pt_shuf[valid] = np.random.permutation(pt_shuf[valid])

        tf_bins = bin_expression_fast(tf_expr, pt_shuf)
        tgt_bins = bin_expression_fast(tgt_expr, pt_shuf)

        lags, corrs = compute_lagged_correlations(tf_bins, tgt_bins)
        ds = directionality_score(lags, corrs, row['Direction'])
        pair_null_scores[idx].append(ds)
        perm_scores.append(ds)
    global_null_scores.append(np.mean(perm_scores))

global_null_scores = np.array(global_null_scores)
observed_mean = df['Directionality_Score'].mean()
p_global = (np.sum(global_null_scores >= observed_mean) + 1) / (N_PERM + 1)

print(f"\nObserved mean score: {observed_mean:.4f}")
print(f"Null mean: {global_null_scores.mean():.4f} +/- {global_null_scores.std():.4f}")
print(f"P-value (global): {p_global:.4f}")
if global_null_scores.std() > 0:
    print(f"Cohen's d: {(observed_mean - global_null_scores.mean()) / global_null_scores.std():.3f}")

# Per-pair p-values
pair_pvals = {}
for idx, row in df.iterrows():
    null = np.array(pair_null_scores[idx])
    obs = row['Directionality_Score']
    pair_pvals[idx] = (np.sum(null >= obs) + 1) / (len(null) + 1)

df['P_Value'] = df.index.map(pair_pvals)

from statsmodels.stats.multitest import multipletests
reject, pvals_bh, _, _ = multipletests(df['P_Value'].values, method='fdr_bh')
df['P_Value_BH'] = pvals_bh
df['Significant_BH'] = reject
print(f"\nSignificant (FDR<0.05): {reject.sum()}/{len(reject)}")

# ============================================================
# NULL MODEL 2: RANDOM GENE PAIRS
# ============================================================
print("\n" + "=" * 60)
print("NULL MODEL 2: Random gene pairs (200 permutations)...")

# Load full adata for gene universe
adata_full = sc.read_h5ad('/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad')
all_genes = adata_full.var['feature_name'].values
expr_frac = np.array((adata_full.X > 0).mean(axis=0)).flatten()
expressed_genes = all_genes[expr_frac > 0.01]
print(f"  {len(expressed_genes)} expressed genes in universe")

null_random = []
for perm_i in range(200):
    if perm_i % 50 == 0:
        print(f"  Perm {perm_i}/200...")
    scores = []
    for lineage in lineage_data:
        adata_sub = lineage_data[lineage]
        for _ in range(10):
            g1, g2 = np.random.choice(expressed_genes, 2, replace=False)
            _, b1, _ = bin_expression(adata_sub, g1)
            _, b2, _ = bin_expression(adata_sub, g2)
            if b1 is None or b2 is None:
                continue
            if np.max(b1) < 0.01 or np.max(b2) < 0.01:
                continue
            lags, corrs = compute_lagged_correlations(b1, b2)
            ds = directionality_score(lags, corrs, 'activate')
            scores.append(ds)
    if scores:
        null_random.append(np.mean(scores))

null_random = np.array(null_random)
p_random = (np.sum(null_random >= observed_mean) + 1) / (len(null_random) + 1)
print(f"Null (random) mean: {null_random.mean():.4f} +/- {null_random.std():.4f}")
print(f"P-value: {p_random:.4f}")

# ============================================================
# FAILURE CASE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("FAILURE CASE ANALYSIS")

failed = df[~df['Directional_Consistent']]
print(f"\nDirectionally inconsistent: {len(failed)}/{len(df)} pairs")

failure_cats = []
for _, row in failed.iterrows():
    if abs(row['Peak_Corr']) < 0.1:
        cat = 'Weak/No correlation'
    elif row['Peak_Lag'] < 0:
        cat = 'Target leads TF'
    elif row['Peak_Lag'] == 0:
        cat = 'Simultaneous'
    elif row['Direction'] == 'activate' and row['Peak_Corr'] < 0:
        cat = 'Wrong sign'
    elif row['Direction'] == 'repress' and row['Peak_Corr'] > 0:
        cat = 'Wrong sign'
    else:
        cat = 'Other'
    failure_cats.append(cat)
    print(f"  {row['TF']} -> {row['Target']} ({row['Lineage']}): {cat} | lag={row['Peak_Lag']}, corr={row['Peak_Corr']:.3f}, ds={row['Directionality_Score']:.4f}")

if failure_cats:
    print("\nFailure category summary:")
    for cat, cnt in pd.Series(failure_cats).value_counts().items():
        print(f"  {cat}: {cnt}")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING RESULTS...")

df_save = df.drop(columns=['Lags', 'Correlations', 'TF_Binned', 'Target_Binned', 'Bin_Centers'])
df_save.to_csv(os.path.join(OUT_DIR, 'implementation', 'directionality_results.csv'), index=False)

with open(os.path.join(OUT_DIR, 'implementation', 'full_results.pkl'), 'wb') as f:
    pickle.dump({
        'results_df': df,
        'null_global': global_null_scores,
        'null_random': null_random,
        'pair_null_scores': pair_null_scores,
        'p_global': p_global,
        'p_random': p_random,
    }, f)

# FINAL SUMMARY TABLE
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
cols = ['TF', 'Target', 'Lineage', 'Direction', 'Peak_Lag', 'Peak_Corr',
        'Directionality_Score', 'Directional_Consistent', 'P_Value_BH']
print(df[cols].to_string(index=False))

print(f"\n\nAGGREGATE STATISTICS:")
print(f"  Total pairs: {len(df)}")
print(f"  Directionally consistent: {df['Directional_Consistent'].sum()}/{len(df)} ({df['Directional_Consistent'].mean()*100:.1f}%)")
print(f"  TF leads by peak: {df['TF_Leads_by_Peak'].sum()}/{len(df)} ({df['TF_Leads_by_Peak'].mean()*100:.1f}%)")
print(f"  Significant FDR<0.05: {df['Significant_BH'].sum()}/{len(df)}")
print(f"  Mean directionality score: {df['Directionality_Score'].mean():.4f}")
print(f"  P vs shuffled PT: {p_global:.4f}")
print(f"  P vs random pairs: {p_random:.4f}")
print(f"\nBy lineage:")
for lin in ['T_cell', 'B_cell', 'Myeloid']:
    sub = df[df['Lineage'] == lin]
    if len(sub) > 0:
        print(f"  {lin}: {sub['Directional_Consistent'].sum()}/{len(sub)} consistent, mean_score={sub['Directionality_Score'].mean():.4f}")
