"""
Phase 2: Multi-tissue extension of directionality audit.
Replicates the Phase 1 analysis on lung and kidney tissues from Tabula Sapiens.

Lung: epithelial differentiation (AT2 -> AT1), myeloid (monocyte -> macrophage)
Kidney: epithelial cells (proximal -> distal nephron segments)
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
from statsmodels.stats.multitest import multipletests
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

np.random.seed(42)

OUT_DIR = '/Users/ihorkendiukhov/biodyn-work/subproject_29_pseudotime_directionality_audit'
FIG_DIR = os.path.join(OUT_DIR, 'reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS (same as Phase 1)
# ============================================================
N_BINS = 50
MAX_LAG = 10

def get_expr(adata_sub, gene_name):
    """Get expression for a gene by symbol via feature_name column."""
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

def compute_pseudotime(adata_full, cell_types, root_type, lineage_name):
    """Subset to lineage, compute diffusion pseudotime."""
    mask = adata_full.obs['cell_type'].isin(cell_types)
    adata_sub = adata_full[mask].copy()
    print(f"\n  {lineage_name}: {adata_sub.n_obs} cells")
    ct_counts = adata_sub.obs['cell_type'].value_counts()
    for ct, n in ct_counts.items():
        print(f"    {ct}: {n}")

    if adata_sub.n_obs < 100:
        print(f"  Skipping {lineage_name}: too few cells")
        return None

    sc.pp.highly_variable_genes(adata_sub, n_top_genes=min(2000, adata_sub.n_vars), flavor='seurat')
    sc.tl.pca(adata_sub, n_comps=min(30, adata_sub.n_obs - 1))
    n_pcs = min(30, adata_sub.obsm['X_pca'].shape[1])
    sc.pp.neighbors(adata_sub, n_neighbors=min(15, adata_sub.n_obs - 1), n_pcs=n_pcs)
    sc.tl.diffmap(adata_sub, n_comps=min(10, adata_sub.n_obs - 2))

    root_mask = adata_sub.obs['cell_type'] == root_type
    if root_mask.sum() == 0:
        root_mask = adata_sub.obs['cell_type'] == cell_types[0]
    root_cells_pca = adata_sub[root_mask].obsm['X_pca']
    centroid = root_cells_pca.mean(axis=0)
    dists = np.linalg.norm(root_cells_pca - centroid, axis=1)
    root_idx = np.where(root_mask.values)[0][np.argmin(dists)]
    adata_sub.uns['iroot'] = root_idx

    sc.tl.dpt(adata_sub, n_branchings=0)
    sc.tl.umap(adata_sub)

    pt = adata_sub.obs['dpt_pseudotime']
    print(f"  DPT range: [{pt.min():.3f}, {pt.max():.3f}]")
    for ct in cell_types:
        ct_m = adata_sub.obs['cell_type'] == ct
        if ct_m.sum() > 0:
            print(f"    {ct}: median_pt={pt[ct_m].median():.3f} (n={ct_m.sum()})")
    return adata_sub

# ============================================================
# 1. LUNG DATA
# ============================================================
print("=" * 60)
print("LOADING LUNG DATA")
print("=" * 60)
adata_lung = sc.read_h5ad('/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad')
print(f"Lung: {adata_lung.shape}")

# Lung lineage definitions
lung_lineages = {
    'Lung_Epithelial': {
        'cell_types': [
            'pulmonary alveolar type 2 cell',
            'pulmonary alveolar type 1 cell',
            'club cell',
            'basal cell',
        ],
        'root_type': 'basal cell',
    },
    'Lung_Myeloid': {
        'cell_types': [
            'classical monocyte',
            'intermediate monocyte',
            'non-classical monocyte',
            'monocyte',
            'macrophage',
        ],
        'root_type': 'classical monocyte',
    },
}

# Lung TF-target pairs
lung_tf_target_pairs = [
    # Epithelial differentiation
    ('NKX2-1', 'SFTPC', 'Lung_Epithelial', 'activate', 'NKX2-1 drives surfactant protein C in AT2'),
    ('NKX2-1', 'SFTPB', 'Lung_Epithelial', 'activate', 'NKX2-1 drives surfactant protein B'),
    ('NKX2-1', 'SFTPA1', 'Lung_Epithelial', 'activate', 'NKX2-1 drives surfactant protein A1'),
    ('NKX2-1', 'ABCA3', 'Lung_Epithelial', 'activate', 'NKX2-1 drives ABCA3 lipid transport'),
    ('FOXA2', 'SFTPC', 'Lung_Epithelial', 'activate', 'FOXA2 drives surfactant C'),
    ('FOXA2', 'MUC5AC', 'Lung_Epithelial', 'activate', 'FOXA2 regulates mucin'),
    ('SOX2', 'KRT5', 'Lung_Epithelial', 'activate', 'SOX2 drives KRT5 in basal cells'),
    ('SOX2', 'TP63', 'Lung_Epithelial', 'activate', 'SOX2 drives p63 in basal cells'),
    ('TP63', 'KRT5', 'Lung_Epithelial', 'activate', 'p63 drives KRT5'),
    ('TP63', 'KRT14', 'Lung_Epithelial', 'activate', 'p63 drives KRT14'),
    ('HOPX', 'AGER', 'Lung_Epithelial', 'activate', 'HOPX drives AGER in AT1'),
    ('HOPX', 'PDPN', 'Lung_Epithelial', 'activate', 'HOPX drives podoplanin in AT1'),
    ('ETV5', 'SFTPC', 'Lung_Epithelial', 'activate', 'ETV5 maintains AT2 identity'),
    ('CEBPA', 'SFTPD', 'Lung_Epithelial', 'activate', 'CEBPA drives surfactant D'),
    ('FOXJ1', 'DNAH5', 'Lung_Epithelial', 'activate', 'FOXJ1 drives dynein for cilia'),
    ('SCGB1A1', 'SCGB3A2', 'Lung_Epithelial', 'activate', 'Club cell secretoglobins co-regulated'),
    # Lung myeloid
    ('SPI1', 'CSF1R', 'Lung_Myeloid', 'activate', 'PU.1 drives CSF1R'),
    ('SPI1', 'CD68', 'Lung_Myeloid', 'activate', 'PU.1 drives CD68'),
    ('SPI1', 'ITGAM', 'Lung_Myeloid', 'activate', 'PU.1 drives CD11b'),
    ('CEBPB', 'IL6', 'Lung_Myeloid', 'activate', 'C/EBPb drives IL-6'),
    ('CEBPB', 'TNF', 'Lung_Myeloid', 'activate', 'C/EBPb drives TNF'),
    ('NFKB1', 'IL1B', 'Lung_Myeloid', 'activate', 'NF-kB drives IL-1beta'),
    ('NFKB1', 'TNF', 'Lung_Myeloid', 'activate', 'NF-kB drives TNF'),
    ('STAT1', 'IRF1', 'Lung_Myeloid', 'activate', 'STAT1 drives IRF1'),
    ('STAT1', 'GBP1', 'Lung_Myeloid', 'activate', 'STAT1 drives GBP1'),
    ('IRF4', 'CCL17', 'Lung_Myeloid', 'activate', 'IRF4 in M2 macrophages'),
    ('MAF', 'MRC1', 'Lung_Myeloid', 'activate', 'c-Maf drives CD206'),
    ('KLF4', 'MRC1', 'Lung_Myeloid', 'activate', 'KLF4 drives M2'),
    ('PPARG', 'FABP4', 'Lung_Myeloid', 'activate', 'PPARg drives FABP4 in alveolar macrophages'),
    ('PPARG', 'CD36', 'Lung_Myeloid', 'activate', 'PPARg drives CD36'),
    ('MARCO', 'MSR1', 'Lung_Myeloid', 'activate', 'Scavenger receptors co-expressed'),
]

# ============================================================
# 2. KIDNEY DATA
# ============================================================
print("\n" + "=" * 60)
print("LOADING KIDNEY DATA")
print("=" * 60)
adata_kidney = sc.read_h5ad('/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad')
print(f"Kidney: {adata_kidney.shape}")

# For kidney, nearly all cells are "kidney epithelial cell" - use subclustering
# We'll compute pseudotime on the epithelial population
kidney_lineages = {
    'Kidney_Epithelial': {
        'cell_types': ['kidney epithelial cell'],
        'root_type': 'kidney epithelial cell',
    },
    'Kidney_Immune': {
        'cell_types': [
            'CD4-positive, alpha-beta T cell',
            'CD8-positive, alpha-beta T cell',
            'B cell',
            'macrophage',
            'natural killer cell',
            'T cell',
            'monocyte',
        ],
        'root_type': 'B cell',
    },
}

# Kidney TF-target pairs
# Nephron segment markers and their TFs
kidney_tf_target_pairs = [
    # Proximal tubule
    ('HNF4A', 'SLC34A1', 'Kidney_Epithelial', 'activate', 'HNF4A drives Na-Pi transporter in PT'),
    ('HNF4A', 'SLC22A6', 'Kidney_Epithelial', 'activate', 'HNF4A drives organic anion transporter'),
    ('HNF4A', 'CUBN', 'Kidney_Epithelial', 'activate', 'HNF4A drives cubilin in PT'),
    ('HNF1B', 'SLC12A1', 'Kidney_Epithelial', 'activate', 'HNF1B drives NKCC2 in TAL'),
    ('HNF1B', 'UMOD', 'Kidney_Epithelial', 'activate', 'HNF1B drives uromodulin'),
    ('HNF1A', 'SLC34A1', 'Kidney_Epithelial', 'activate', 'HNF1A drives proximal tubule transporters'),
    # Distal/collecting duct
    ('FOXI1', 'ATP6V1B1', 'Kidney_Epithelial', 'activate', 'FOXI1 drives V-ATPase in intercalated cells'),
    ('FOXI1', 'SLC4A1', 'Kidney_Epithelial', 'activate', 'FOXI1 drives AE1 in intercalated cells'),
    ('TFCP2L1', 'AQP2', 'Kidney_Epithelial', 'activate', 'TFCP2L1 drives AQP2 in CD'),
    ('ELF5', 'AQP2', 'Kidney_Epithelial', 'activate', 'ELF5 in collecting duct principal cells'),
    # General epithelial
    ('PAX2', 'NCAM1', 'Kidney_Epithelial', 'activate', 'PAX2 in nephron development'),
    ('PAX8', 'CDH16', 'Kidney_Epithelial', 'activate', 'PAX8 drives kidney cadherin'),
    ('WT1', 'NPHS1', 'Kidney_Epithelial', 'activate', 'WT1 drives nephrin in podocytes'),
    ('WT1', 'NPHS2', 'Kidney_Epithelial', 'activate', 'WT1 drives podocin'),
    ('SOX9', 'COL2A1', 'Kidney_Epithelial', 'activate', 'SOX9 drives collagen'),
    # Kidney immune (same conserved TF-target pairs)
    ('SPI1', 'CSF1R', 'Kidney_Immune', 'activate', 'PU.1 drives CSF1R'),
    ('SPI1', 'CD68', 'Kidney_Immune', 'activate', 'PU.1 drives CD68'),
    ('NFKB1', 'BCL2', 'Kidney_Immune', 'activate', 'NF-kB survival'),
]

# ============================================================
# 3. COMPUTE PSEUDOTIME FOR ALL LINEAGES
# ============================================================
print("\n" + "=" * 60)
print("COMPUTING PSEUDOTIME")
print("=" * 60)

all_lineage_data = {}

# Lung lineages
for name, defn in lung_lineages.items():
    print(f"\nProcessing {name}...")
    result = compute_pseudotime(adata_lung, defn['cell_types'], defn['root_type'], name)
    if result is not None:
        all_lineage_data[name] = result

# Kidney lineages
for name, defn in kidney_lineages.items():
    print(f"\nProcessing {name}...")
    result = compute_pseudotime(adata_kidney, defn['cell_types'], defn['root_type'], name)
    if result is not None:
        all_lineage_data[name] = result

print(f"\nSuccessfully computed pseudotime for {len(all_lineage_data)} lineages")

# ============================================================
# 4. RUN DIRECTIONALITY ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("DIRECTIONALITY ANALYSIS")
print("=" * 60)

all_pairs = lung_tf_target_pairs + kidney_tf_target_pairs
results = []

for tf_name, target_name, lineage, direction, description in all_pairs:
    if lineage not in all_lineage_data:
        continue
    adata_sub = all_lineage_data[lineage]

    _, tf_bins, _ = bin_expression(adata_sub, tf_name)
    _, tgt_bins, _ = bin_expression(adata_sub, target_name)

    if tf_bins is None or tgt_bins is None:
        continue
    if np.max(tf_bins) < 0.01 or np.max(tgt_bins) < 0.01:
        continue

    lags, corrs = compute_lagged_correlations(tf_bins, tgt_bins)
    lags_arr = np.array(lags)
    corrs_arr = np.array(corrs)

    if direction == 'activate':
        best_idx = np.argmax(corrs_arr)
    else:
        best_idx = np.argmin(corrs_arr)
    peak_lag = int(lags_arr[best_idx])
    peak_corr = corrs_arr[best_idx]

    if direction == 'activate':
        dc = peak_lag > 0 and peak_corr > 0
    else:
        dc = peak_lag > 0 and peak_corr < 0

    bin_centers, _, _ = bin_expression(adata_sub, tf_name)
    tf_peak_pt = bin_centers[np.argmax(tf_bins)]
    tgt_peak_pt = bin_centers[np.argmax(tgt_bins)]
    zero_lag_corr = corrs[lags.index(0)]
    ds = directionality_score(lags, corrs, direction)

    # Determine tissue
    if lineage.startswith('Lung'):
        tissue = 'Lung'
    elif lineage.startswith('Kidney'):
        tissue = 'Kidney'
    else:
        tissue = 'Unknown'

    results.append({
        'TF': tf_name, 'Target': target_name, 'Lineage': lineage,
        'Tissue': tissue,
        'Direction': direction, 'Description': description,
        'Peak_Lag': peak_lag, 'Peak_Corr': peak_corr,
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
print(f"\nSuccessfully analyzed {len(df)} TF-target pairs across tissues")

for tissue in df['Tissue'].unique():
    sub_tissue = df[df['Tissue'] == tissue]
    for lin in sub_tissue['Lineage'].unique():
        sub = sub_tissue[sub_tissue['Lineage'] == lin]
        nc = sub['Directional_Consistent'].sum()
        print(f"  {lin}: {nc}/{len(sub)} consistent ({nc/len(sub)*100:.1f}%)")

print(f"\nOverall multi-tissue consistency: {df['Directional_Consistent'].mean()*100:.1f}%")
print(f"Mean directionality score: {df['Directionality_Score'].mean():.4f}")

# ============================================================
# 5. NULL MODEL: SHUFFLED PSEUDOTIME
# ============================================================
print("\n" + "=" * 60)
print("NULL MODEL: Shuffled pseudotime (500 permutations)")
print("=" * 60)

N_PERM = 500

pair_exprs = {}
for idx, row in df.iterrows():
    adata_sub = all_lineage_data[row['Lineage']]
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

print(f"\nObserved mean: {observed_mean:.4f}")
print(f"Null mean: {global_null_scores.mean():.4f} +/- {global_null_scores.std():.4f}")
print(f"P-value: {p_global:.4f}")
if global_null_scores.std() > 0:
    cohens_d = (observed_mean - global_null_scores.mean()) / global_null_scores.std()
    print(f"Cohen's d: {cohens_d:.3f}")

# Per-pair p-values
pair_pvals = {}
for idx, row in df.iterrows():
    null = np.array(pair_null_scores[idx])
    obs = row['Directionality_Score']
    pair_pvals[idx] = (np.sum(null >= obs) + 1) / (len(null) + 1)

df['P_Value'] = df.index.map(pair_pvals)
reject, pvals_bh, _, _ = multipletests(df['P_Value'].values, method='fdr_bh')
df['P_Value_BH'] = pvals_bh
df['Significant_BH'] = reject

print(f"Significant (FDR<0.05): {reject.sum()}/{len(reject)}")

# ============================================================
# 6. FAILURE MODE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("FAILURE MODE ANALYSIS")
print("=" * 60)

failed = df[~df['Directional_Consistent']]
failure_cats = []
for _, row in failed.iterrows():
    if abs(row['Peak_Corr']) < 0.1:
        cat = 'Weak/No correlation'
    elif row['Peak_Lag'] < 0:
        cat = 'Target leads TF'
    elif row['Peak_Lag'] == 0:
        cat = 'Simultaneous'
    elif (row['Direction'] == 'activate' and row['Peak_Corr'] < 0) or \
         (row['Direction'] == 'repress' and row['Peak_Corr'] > 0):
        cat = 'Wrong sign'
    else:
        cat = 'Other'
    failure_cats.append(cat)

if failure_cats:
    print("Failure categories:")
    for cat, cnt in pd.Series(failure_cats).value_counts().items():
        frac = cnt / len(failure_cats) * 100
        print(f"  {cat}: {cnt} ({frac:.1f}%)")

# ============================================================
# 7. CROSS-TISSUE COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("CROSS-TISSUE COMPARISON")
print("=" * 60)

# Load Phase 1 results for comparison
phase1_df = pd.read_csv(os.path.join(OUT_DIR, 'implementation', 'directionality_results.csv'))
phase1_df['Tissue'] = 'Immune'

# Compare immune vs lung vs kidney
comparison = {}
for tissue in ['Immune', 'Lung', 'Kidney']:
    if tissue == 'Immune':
        sub = phase1_df
    else:
        sub = df[df['Tissue'] == tissue]
    if len(sub) == 0:
        continue
    comparison[tissue] = {
        'n_pairs': len(sub),
        'consistency_rate': sub['Directional_Consistent'].mean() * 100,
        'mean_dir_score': sub['Directionality_Score'].mean(),
        'median_dir_score': sub['Directionality_Score'].median(),
        'frac_positive': (sub['Directionality_Score'] > 0).mean() * 100,
    }
    print(f"\n{tissue}:")
    for k, v in comparison[tissue].items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

# Also compare shared myeloid pairs (same pairs in immune and lung myeloid)
shared_myeloid_tfs = ['SPI1', 'CEBPB', 'NFKB1', 'STAT1', 'IRF4', 'MAF', 'KLF4']
print("\n\nShared myeloid TF pairs across tissues:")
for tf in shared_myeloid_tfs:
    immune_rows = phase1_df[phase1_df['TF'] == tf]
    lung_rows = df[(df['TF'] == tf) & (df['Tissue'] == 'Lung')]
    if len(immune_rows) > 0 and len(lung_rows) > 0:
        for _, ir in immune_rows.iterrows():
            matching_lung = lung_rows[lung_rows['Target'] == ir['Target']]
            if len(matching_lung) > 0:
                lr = matching_lung.iloc[0]
                print(f"  {tf}->{ir['Target']}: Immune ds={ir['Directionality_Score']:.3f}, Lung ds={lr['Directionality_Score']:.3f}")

# ============================================================
# 8. SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")

df_save = df.drop(columns=['Lags', 'Correlations', 'TF_Binned', 'Target_Binned', 'Bin_Centers'])
df_save.to_csv(os.path.join(OUT_DIR, 'implementation', 'phase2_multi_tissue_results.csv'), index=False)

with open(os.path.join(OUT_DIR, 'implementation', 'phase2_multi_tissue_full.pkl'), 'wb') as f:
    pickle.dump({
        'results_df': df,
        'null_global': global_null_scores,
        'pair_null_scores': pair_null_scores,
        'p_global': p_global,
        'comparison': comparison,
    }, f)

# Save lineage adata
for name, adata_sub in all_lineage_data.items():
    adata_sub.write(os.path.join(OUT_DIR, 'implementation', f'adata_{name}.h5ad'))

# ============================================================
# 9. FIGURES
# ============================================================
print("\nGenerating figures...")

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

tissue_colors = {'Immune': '#E64B35', 'Lung': '#4DBBD5', 'Kidney': '#00A087'}

# Figure: Cross-tissue consistency comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (A) Consistency rates across tissues
ax = axes[0]
tissues = list(comparison.keys())
rates = [comparison[t]['consistency_rate'] for t in tissues]
bars = ax.bar(range(len(tissues)), rates,
              color=[tissue_colors.get(t, 'gray') for t in tissues], alpha=0.7)
ax.set_xticks(range(len(tissues)))
ax.set_xticklabels(tissues)
ax.set_ylabel('Consistency Rate (%)')
ax.set_title('(A) Directional Consistency by Tissue')
ax.set_ylim(0, 60)
ax.axhline(50, color='gray', linestyle='--', alpha=0.3, label='Chance')
for i, (rate, bar) in enumerate(zip(rates, bars)):
    n = comparison[tissues[i]]['n_pairs']
    ax.text(bar.get_x() + bar.get_width()/2, rate + 1.5,
            f'{rate:.1f}%\n(n={n})', ha='center', fontsize=8)

# (B) Directionality score distributions
ax = axes[1]
all_scores = []
all_labels = []
all_tissue_cols = []
for tissue in tissues:
    if tissue == 'Immune':
        scores = phase1_df['Directionality_Score'].values
    else:
        scores = df[df['Tissue'] == tissue]['Directionality_Score'].values
    all_scores.append(scores)
    all_labels.append(tissue)

bp = ax.boxplot(all_scores, labels=all_labels, patch_artist=True, widths=0.5)
for i, tissue in enumerate(tissues):
    bp['boxes'][i].set_facecolor(tissue_colors.get(tissue, 'gray'))
    bp['boxes'][i].set_alpha(0.5)
    x = np.random.normal(i+1, 0.05, len(all_scores[i]))
    ax.scatter(x, all_scores[i], c=tissue_colors.get(tissue, 'gray'), s=15, alpha=0.5, zorder=5)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Directionality Score')
ax.set_title('(B) Score Distribution by Tissue')

# (C) Failure mode comparison
ax = axes[2]
failure_data = {}
for tissue in tissues:
    if tissue == 'Immune':
        sub = phase1_df[phase1_df['Directional_Consistent'] == False]
    else:
        sub = df[(df['Tissue'] == tissue) & (~df['Directional_Consistent'])]
    cats_t = []
    for _, row in sub.iterrows():
        if abs(row['Peak_Corr']) < 0.1:
            cats_t.append('Weak')
        elif row['Peak_Lag'] < 0:
            cats_t.append('Target leads')
        elif row['Peak_Lag'] == 0:
            cats_t.append('Simultaneous')
        else:
            cats_t.append('Other')
    if cats_t:
        total = len(cats_t)
        failure_data[tissue] = {c: cats_t.count(c)/total*100 for c in ['Simultaneous', 'Target leads', 'Weak', 'Other']}

if failure_data:
    cats_list = ['Simultaneous', 'Target leads', 'Weak', 'Other']
    cat_colors_map = {'Simultaneous': '#4DBBD5', 'Target leads': '#E64B35', 'Weak': '#F39B7F', 'Other': '#91D1C2'}
    x = np.arange(len(failure_data))
    width = 0.2
    for j, cat in enumerate(cats_list):
        vals = [failure_data[t].get(cat, 0) for t in failure_data]
        ax.bar(x + j*width, vals, width, label=cat, color=cat_colors_map[cat], alpha=0.7)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(list(failure_data.keys()))
    ax.set_ylabel('Fraction of failures (%)')
    ax.set_title('(C) Failure Modes by Tissue')
    ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'phase2_fig1_multi_tissue_comparison.png'))
fig.savefig(os.path.join(FIG_DIR, 'phase2_fig1_multi_tissue_comparison.pdf'))
plt.close()

# Figure: Tissue-specific UMAP + pseudotime
n_lineages = len(all_lineage_data)
if n_lineages > 0:
    fig, axes = plt.subplots(2, n_lineages, figsize=(5*n_lineages, 9))
    if n_lineages == 1:
        axes = axes.reshape(-1, 1)

    for i, (name, adata_sub) in enumerate(all_lineage_data.items()):
        # Cell type
        ax = axes[0, i]
        ct = adata_sub.obs['cell_type'].astype('category')
        umap = adata_sub.obsm['X_umap']
        cats = ct.cat.categories
        colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))
        for j, c in enumerate(cats):
            m = ct == c
            ax.scatter(umap[m, 0], umap[m, 1], s=1, alpha=0.5, color=colors[j],
                       label=c[:25], rasterized=True)
        ax.set_title(f'{name}')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.legend(fontsize=4, loc='best', markerscale=4, framealpha=0.7)

        # Pseudotime
        ax = axes[1, i]
        pt = adata_sub.obs['dpt_pseudotime'].values
        valid = np.isfinite(pt)
        sc_plot = ax.scatter(umap[valid, 0], umap[valid, 1], c=pt[valid], s=1,
                             cmap='viridis', alpha=0.5, rasterized=True)
        ax.set_title(f'{name}: Pseudotime')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        plt.colorbar(sc_plot, ax=ax, label='DPT', shrink=0.7)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'phase2_fig2_tissue_umap_pseudotime.png'))
    fig.savefig(os.path.join(FIG_DIR, 'phase2_fig2_tissue_umap_pseudotime.pdf'))
    plt.close()

# Figure: Per-lineage summary bar chart
fig, ax = plt.subplots(figsize=(10, 5))
lineage_order = sorted(df['Lineage'].unique())
# Add immune lineages
phase1_lineages = ['T_cell', 'B_cell', 'Myeloid']
all_lineage_names = phase1_lineages + lineage_order

rates_all = []
ns_all = []
for lin in all_lineage_names:
    if lin in phase1_lineages:
        sub = phase1_df[phase1_df['Lineage'] == lin]
    else:
        sub = df[df['Lineage'] == lin]
    if len(sub) > 0:
        rates_all.append(sub['Directional_Consistent'].mean() * 100)
        ns_all.append(len(sub))
    else:
        rates_all.append(0)
        ns_all.append(0)

tissue_for_lin = {
    'T_cell': 'Immune', 'B_cell': 'Immune', 'Myeloid': 'Immune',
}
for lin in lineage_order:
    if 'Lung' in lin:
        tissue_for_lin[lin] = 'Lung'
    elif 'Kidney' in lin:
        tissue_for_lin[lin] = 'Kidney'

bar_colors = [tissue_colors.get(tissue_for_lin.get(l, ''), 'gray') for l in all_lineage_names]
bars = ax.bar(range(len(all_lineage_names)), rates_all, color=bar_colors, alpha=0.7)
ax.set_xticks(range(len(all_lineage_names)))
ax.set_xticklabels([l.replace('_', ' ') for l in all_lineage_names], rotation=30, ha='right')
ax.set_ylabel('Consistency Rate (%)')
ax.set_title('Directional Consistency Across All Lineages and Tissues')
ax.axhline(50, color='gray', linestyle='--', alpha=0.3)
ax.set_ylim(0, 70)

for i, (rate, n) in enumerate(zip(rates_all, ns_all)):
    if n > 0:
        ax.text(i, rate + 1, f'{rate:.0f}%\n(n={n})', ha='center', fontsize=7)

handles = [Patch(facecolor=c, alpha=0.7, label=t) for t, c in tissue_colors.items()]
ax.legend(handles=handles, fontsize=8, loc='upper right')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'phase2_fig3_all_lineages_comparison.png'))
fig.savefig(os.path.join(FIG_DIR, 'phase2_fig3_all_lineages_comparison.pdf'))
plt.close()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
cols = ['TF', 'Target', 'Tissue', 'Lineage', 'Peak_Lag', 'Peak_Corr',
        'Directionality_Score', 'Directional_Consistent', 'P_Value_BH']
print(df[cols].to_string(index=False))

print(f"\n\nAGGREGATE:")
print(f"  Total pairs: {len(df)}")
print(f"  Consistent: {df['Directional_Consistent'].sum()}/{len(df)} ({df['Directional_Consistent'].mean()*100:.1f}%)")
print(f"  Mean score: {df['Directionality_Score'].mean():.4f}")
print(f"  P vs shuffled: {p_global:.4f}")

print("\nCross-tissue comparison:")
for tissue, stats in comparison.items():
    print(f"  {tissue}: {stats['consistency_rate']:.1f}% consistent, mean_score={stats['mean_dir_score']:.4f}")

print("\nDone!")
