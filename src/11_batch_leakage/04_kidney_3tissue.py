"""
Phase 2 - Objective 1: Kidney tissue analysis and 3-tissue leakage comparison.

Since kidney has only 1 donor (TSP2), we cannot do donor-based ASI.
Instead we focus on method-based ASI (10X vs smartseq), and compare
leakage metrics across all 3 tissues in a unified table.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, json, time

warnings.filterwarnings('ignore')

OUTDIR = "/Users/ihorkendiukhov/biodyn-work/subproject_33_batch_donor_leakage_audit/reports/figures"
DATADIR = "/Users/ihorkendiukhov/biodyn-work/subproject_33_batch_donor_leakage_audit/implementation"
os.makedirs(OUTDIR, exist_ok=True)

TF_LIST = [
    'STAT1', 'STAT3', 'STAT4', 'STAT5A', 'STAT5B', 'STAT6',
    'IRF1', 'IRF4', 'IRF7', 'IRF8',
    'NFKB1', 'NFKB2', 'RELA', 'REL', 'RELB',
    'TBX21', 'GATA3', 'RORC', 'FOXP3', 'BCL6',
    'SPI1', 'RUNX1', 'RUNX3', 'ETS1', 'ETS2',
    'MYC', 'MAX', 'JUN', 'FOS', 'JUNB', 'JUND', 'FOSL1', 'FOSL2',
    'HIF1A', 'EPAS1', 'ARNT',
    'TP53', 'RB1', 'E2F1', 'E2F4',
    'PAX5', 'EBF1', 'TCF7', 'LEF1',
    'CEBPA', 'CEBPB', 'CEBPD',
    'SOX2', 'SOX4', 'SOX9',
    'KLF2', 'KLF4', 'KLF6',
    'NR4A1', 'NR4A2', 'NR4A3',
    'BATF', 'BATF3', 'MAF', 'MAFB',
    'EOMES', 'PRDM1', 'ID2', 'ID3',
    'NFE2L2', 'BACH1', 'BACH2',
    'ZEB1', 'ZEB2', 'SNAI1', 'TWIST1',
    'FOXO1', 'FOXO3',
    'AHR', 'IKZF1', 'IKZF3',
]

TARGET_POOL = [
    'CD3D', 'CD3E', 'CD4', 'CD8A', 'CD8B', 'CD19', 'MS4A1', 'CD79A', 'CD79B',
    'CD14', 'CD68', 'ITGAM', 'ITGAX', 'FCGR3A', 'CSF1R',
    'IL2', 'IL2RA', 'IL7R', 'IL6', 'IL10', 'IL17A', 'IL21', 'IL4', 'IL13',
    'IFNG', 'TNF', 'TNFRSF9', 'TNFRSF18',
    'GZMB', 'GZMA', 'GZMK', 'PRF1', 'GNLY', 'NKG7',
    'CCL5', 'CCR7', 'CXCR4', 'CXCL13', 'CCL2', 'CXCL10',
    'HLA-DRA', 'HLA-DRB1', 'HLA-A', 'HLA-B', 'HLA-C', 'B2M',
    'LYZ', 'S100A8', 'S100A9', 'S100A12',
    'SELL', 'ITGA4', 'ITGB1', 'ICAM1', 'VCAM1',
    'MKI67', 'TOP2A', 'CDK1', 'PCNA',
    'BCL2', 'BCL2L1', 'MCL1', 'BAX', 'BAK1',
    'CTLA4', 'PDCD1', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX',
    'FN1', 'COL1A1', 'COL3A1', 'VIM', 'ACTA2', 'CDH1', 'EPCAM',
    'SFTPC', 'SFTPB', 'SFTPA1', 'SCGB1A1', 'MUC5AC', 'FOXJ1',
    'PECAM1', 'CDH5', 'VWF', 'KDR',
    'NOTCH1', 'DLL4', 'JAG1', 'WNT5A', 'CTNNB1',
    'TGFB1', 'TGFBR1', 'SMAD3', 'SMAD7',
    'SOD2', 'HMOX1', 'NQO1', 'GSTP1',
    'GAPDH', 'ACTB', 'RPL13A', 'RPS18',
    'PTPRC', 'THY1', 'ENG', 'NT5E',
]


def load_and_prepare(path, max_cells=20000):
    adata = ad.read_h5ad(path)
    if 'feature_name' in adata.var.columns and adata.var_names[0].startswith('ENSG'):
        gene_names = adata.var['feature_name'].values.astype(str)
        seen = {}
        unique_names = []
        for name in gene_names:
            if name in seen:
                seen[name] += 1
                unique_names.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                unique_names.append(name)
        adata.var_names = unique_names
        adata.var_names_make_unique()
    if adata.n_obs > max_cells:
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, max_cells, replace=False)
        adata = adata[idx].copy()
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def get_gene_indices(adata, gene_list):
    var_names = list(adata.var_names)
    return {g: var_names.index(g) for g in gene_list if g in var_names}


def compute_edge_scores(X, tf_idx, target_idx):
    if sparse.issparse(X):
        X = X.toarray()
    edges = []
    for tf, ti in tf_idx.items():
        tf_expr = X[:, ti]
        if np.std(tf_expr) < 1e-10:
            continue
        for tgt, tgi in target_idx.items():
            if tf == tgt:
                continue
            tgt_expr = X[:, tgi]
            if np.std(tgt_expr) < 1e-10:
                continue
            r, _ = pearsonr(tf_expr, tgt_expr)
            if np.isfinite(r):
                edges.append({'tf': tf, 'target': tgt, 'score': r})
    return pd.DataFrame(edges)


def compute_per_cell_edge_features(adata, tf_idx, target_idx, n_edges=200):
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    all_products = []
    all_names = []
    for tf in tf_idx:
        tf_expr = X[:, tf_idx[tf]]
        for tgt in target_idx:
            if tf == tgt:
                continue
            tgt_expr = X[:, target_idx[tgt]]
            all_products.append(tf_expr * tgt_expr)
            all_names.append(f"{tf}->{tgt}")
    all_products = np.column_stack(all_products)
    variances = np.var(all_products, axis=0)
    top_idx = np.argsort(variances)[-n_edges:]
    return all_products[:, top_idx], [all_names[i] for i in top_idx]


def balanced_resample(adata, group_col, seed=42):
    np.random.seed(seed)
    groups = adata.obs[group_col].values
    unique_groups = pd.Series(groups).value_counts()
    min_count = max(unique_groups.min(), 50)
    idx = []
    for grp in unique_groups.index:
        grp_idx = np.where(groups == grp)[0]
        n_sample = min(min_count, len(grp_idx))
        chosen = np.random.choice(grp_idx, n_sample, replace=False)
        idx.extend(chosen)
    return adata[idx].copy()


# ============================================================
# KIDNEY FULL ANALYSIS
# ============================================================
print("=" * 70)
print("PHASE 2: KIDNEY TISSUE ANALYSIS")
print("=" * 70)

adata_kidney = load_and_prepare(
    "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad"
)
print(f"Kidney shape: {adata_kidney.shape}")
print(f"Donors: {adata_kidney.obs['donor_id'].value_counts().to_dict()}")
print(f"Methods: {adata_kidney.obs['method'].value_counts().to_dict()}")
print(f"Cell types: {adata_kidney.obs['cell_type'].nunique()}")

tf_idx = get_gene_indices(adata_kidney, TF_LIST)
target_idx = get_gene_indices(adata_kidney, TARGET_POOL)
print(f"Found {len(tf_idx)} TFs and {len(target_idx)} targets")

# Step 1: Global edge scores
print("\n--- Computing global edge scores ---")
edges_full = compute_edge_scores(adata_kidney.X, tf_idx, target_idx)
print(f"Computed {len(edges_full)} edges")
print(f"Score stats: mean={edges_full['score'].mean():.4f}, std={edges_full['score'].std():.4f}")

# Step 2: Method-balanced ASI (since kidney has only 1 donor, use method as grouping)
print("\n--- Method-balanced resampling ---")
adata_method_bal = balanced_resample(adata_kidney, 'method')
print(f"Balanced shape: {adata_method_bal.shape}")
print(f"Method distribution: {adata_method_bal.obs['method'].value_counts().to_dict()}")

edges_method_bal = compute_edge_scores(adata_method_bal.X, tf_idx, target_idx)

# Compute method-based ASI
merged = edges_full.merge(edges_method_bal, on=['tf', 'target'], suffixes=('_full', '_balanced'))
merged['delta'] = merged['score_full'] - merged['score_balanced']
merged['abs_delta'] = merged['delta'].abs()
eps = 0.01
merged['ASI'] = merged['abs_delta'] / np.maximum(merged['score_full'].abs(), eps)

print(f"\nKidney Method-ASI stats:")
print(f"  Mean: {merged['ASI'].mean():.4f}")
print(f"  Median: {merged['ASI'].median():.4f}")
print(f"  Edges with ASI > 0.5: {(merged['ASI'] > 0.5).sum()} / {len(merged)} "
      f"({100*(merged['ASI'] > 0.5).mean():.1f}%)")

sign_flipped = ((merged['score_full'] * merged['score_balanced']) < 0).sum()
print(f"  Sign-flipped: {sign_flipped} / {len(merged)} ({100*sign_flipped/len(merged):.1f}%)")

r_stability, _ = pearsonr(merged['score_full'], merged['score_balanced'])
print(f"  Full vs balanced correlation: r = {r_stability:.4f}")

# Save kidney sensitivity
merged.to_csv(os.path.join(DATADIR, 'Kidney_sensitivity.csv'), index=False)

kidney_results = {
    'n_cells': int(adata_kidney.n_obs),
    'n_donors': 1,
    'n_batches': int(adata_kidney.obs['_scvi_batch'].nunique()),
    'n_methods': 2,
    'n_cell_types': int(adata_kidney.obs['cell_type'].nunique()),
    'n_edges': len(edges_full),
    'method_ASI_mean': float(merged['ASI'].mean()),
    'method_ASI_median': float(merged['ASI'].median()),
    'n_high_ASI': int((merged['ASI'] > 0.5).sum()),
    'pct_high_ASI': float(100 * (merged['ASI'] > 0.5).mean()),
    'n_sign_flipped': int(sign_flipped),
    'edge_score_correlation': float(r_stability),
}

# Step 3: Kidney ASI figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(merged['ASI'].values, bins=50, color='seagreen', edgecolor='white', alpha=0.8)
axes[0].axvline(0.5, color='red', linestyle='--', label='Blacklist threshold')
axes[0].set_xlabel('Artifact Sensitivity Index (Method)')
axes[0].set_ylabel('Number of edges')
axes[0].set_title('Kidney: Method-Based ASI Distribution')
axes[0].legend()

axes[1].scatter(merged['score_full'], merged['score_balanced'], alpha=0.3, s=5, c='seagreen')
lims = [min(merged['score_full'].min(), merged['score_balanced'].min()) - 0.05,
        max(merged['score_full'].max(), merged['score_balanced'].max()) + 0.05]
axes[1].plot(lims, lims, 'k--', alpha=0.5, lw=1)
bl = merged[merged['ASI'] > 0.5]
axes[1].scatter(bl['score_full'], bl['score_balanced'], alpha=0.5, s=10, c='red',
                label=f'ASI > 0.5 (n={len(bl)})')
axes[1].set_xlabel('Edge Score (Full Dataset)')
axes[1].set_ylabel('Edge Score (Method-Balanced)')
axes[1].set_title(f'Kidney: Edge Score Stability (r = {r_stability:.4f})')
axes[1].legend()

plt.suptitle('Kidney Tissue: Method-Based Artifact Sensitivity', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'phase2_kidney_ASI.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved phase2_kidney_ASI.png")

# Step 4: Cell-type composition check (method vs cell type)
print("\n--- Cell type vs method confound check ---")
ct_method = pd.crosstab(adata_kidney.obs['cell_type'], adata_kidney.obs['method'])
ct_method_norm = ct_method.div(ct_method.sum(axis=1), axis=0)

from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(ct_method.values)
cramers_v = np.sqrt(chi2 / (ct_method.values.sum() * (min(ct_method.shape) - 1)))
print(f"Chi-squared: {chi2:.1f}, p={p:.2e}, Cramer's V={cramers_v:.4f}")
kidney_results['celltype_method_cramersV'] = float(cramers_v)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(ct_method_norm, cmap='YlOrRd', ax=ax, annot=True, fmt='.2f')
ax.set_title(f"Kidney: Cell Type Composition per Method\nCramer's V = {cramers_v:.4f}")
ax.set_xlabel('Method')
ax.set_ylabel('Cell Type')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'phase2_kidney_celltype_method_confound.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved phase2_kidney_celltype_method_confound.png")


# ============================================================
# 3-TISSUE COMPARISON TABLE
# ============================================================
print("\n" + "=" * 70)
print("3-TISSUE COMPARISON")
print("=" * 70)

# Load Phase 1 results
with open(os.path.join(DATADIR, 'all_results.json'), 'r') as f:
    phase1 = json.load(f)

# Build comparison table
comparison = []

# Immune
imm = phase1['Immune']
comparison.append({
    'Tissue': 'Immune',
    'Cells': 20000,
    'Donors': 24,
    'Batches': 42,
    'Cell Types': 43,
    'Methods': '10X + SS2',
    'N Edges': imm['n_edges'],
    'Donor Bal. Acc (LR)': imm.get('donor_leakage', {}).get('LogisticRegression', {}).get('balanced_accuracy', None),
    'Donor AUC (LR)': imm.get('donor_leakage', {}).get('LogisticRegression', {}).get('auc', None),
    'Batch Bal. Acc (LR)': imm.get('batch_leakage', {}).get('LogisticRegression', {}).get('balanced_accuracy', None),
    'Method Bal. Acc (LR)': imm.get('method_leakage', {}).get('LogisticRegression', {}).get('balanced_accuracy', None),
    'Method AUC (LR)': imm.get('method_leakage', {}).get('LogisticRegression', {}).get('auc', None),
    'Donor ASI Mean': imm.get('ASI_mean', None),
    'Donor ASI Median': imm.get('ASI_median', None),
    'Donor Blacklisted %': imm.get('pct_high_ASI', None),
    'Edge Stability r': imm.get('edge_score_correlation', None),
    'Sign-Flipped': imm.get('n_sign_flipped', None),
    'Donor Imbalance': 'Severe (TSP14: 23%)',
})

# Lung
lng = phase1['Lung']
comparison.append({
    'Tissue': 'Lung',
    'Cells': 20000,
    'Donors': 4,
    'Batches': 7,
    'Cell Types': 34,
    'Methods': '10X + SS2',
    'N Edges': lng['n_edges'],
    'Donor Bal. Acc (LR)': lng.get('donor_leakage', {}).get('LogisticRegression', {}).get('balanced_accuracy', None),
    'Donor AUC (LR)': lng.get('donor_leakage', {}).get('LogisticRegression', {}).get('auc', None),
    'Batch Bal. Acc (LR)': lng.get('batch_leakage', {}).get('LogisticRegression', {}).get('balanced_accuracy', None),
    'Method Bal. Acc (LR)': lng.get('method_leakage', {}).get('LogisticRegression', {}).get('balanced_accuracy', None),
    'Method AUC (LR)': lng.get('method_leakage', {}).get('LogisticRegression', {}).get('auc', None),
    'Donor ASI Mean': lng.get('ASI_mean', None),
    'Donor ASI Median': lng.get('ASI_median', None),
    'Donor Blacklisted %': lng.get('pct_high_ASI', None),
    'Edge Stability r': lng.get('edge_score_correlation', None),
    'Sign-Flipped': lng.get('n_sign_flipped', None),
    'Donor Imbalance': 'Moderate (4 donors)',
})

# Kidney
comparison.append({
    'Tissue': 'Kidney',
    'Cells': kidney_results['n_cells'],
    'Donors': 1,
    'Batches': kidney_results['n_batches'],
    'Cell Types': kidney_results['n_cell_types'],
    'Methods': '10X + SS2',
    'N Edges': kidney_results['n_edges'],
    'Donor Bal. Acc (LR)': None,  # only 1 donor
    'Donor AUC (LR)': None,
    'Batch Bal. Acc (LR)': phase1['Kidney'].get('batch_leakage', {}).get('LogisticRegression', {}).get('balanced_accuracy', None),
    'Method Bal. Acc (LR)': phase1['Kidney'].get('method_leakage', {}).get('LogisticRegression', {}).get('balanced_accuracy', None),
    'Method AUC (LR)': phase1['Kidney'].get('method_leakage', {}).get('LogisticRegression', {}).get('auc', None),
    'Donor ASI Mean': None,
    'Donor ASI Median': None,
    'Donor Blacklisted %': None,
    'Edge Stability r': kidney_results['edge_score_correlation'],
    'Sign-Flipped': kidney_results['n_sign_flipped'],
    'Donor Imbalance': 'N/A (single donor)',
})

comp_df = pd.DataFrame(comparison)
comp_df.to_csv(os.path.join(DATADIR, 'phase2_3tissue_comparison.csv'), index=False)
print("\n3-Tissue Comparison Table:")
print(comp_df.to_string(index=False))

# ============================================================
# 3-Tissue Comparison Figure
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

tissues = ['Immune', 'Lung', 'Kidney']
colors = ['steelblue', 'darkorange', 'seagreen']

# Panel A: Method leakage AUC across tissues
method_aucs = [
    imm['method_leakage']['LogisticRegression']['auc'],
    lng['method_leakage']['LogisticRegression']['auc'],
    phase1['Kidney']['method_leakage']['LogisticRegression']['auc'],
]
method_bal_accs = [
    imm['method_leakage']['LogisticRegression']['balanced_accuracy'],
    lng['method_leakage']['LogisticRegression']['balanced_accuracy'],
    phase1['Kidney']['method_leakage']['LogisticRegression']['balanced_accuracy'],
]
x = np.arange(3)
bars = axes[0, 0].bar(x, method_aucs, color=colors, alpha=0.8, width=0.5)
axes[0, 0].axhline(0.5, color='red', linestyle='--', lw=1, label='Chance')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(tissues)
axes[0, 0].set_ylabel('AUC')
axes[0, 0].set_title('Method (10X vs SS2) Leakage AUC')
axes[0, 0].set_ylim(0.5, 1.0)
for i, v in enumerate(method_aucs):
    axes[0, 0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
axes[0, 0].legend()

# Panel B: Edge score correlation (full vs balanced) -- kidney uses method-balanced
stab_corrs = [
    imm['edge_score_correlation'],
    lng['edge_score_correlation'],
    kidney_results['edge_score_correlation'],
]
axes[0, 1].bar(x, stab_corrs, color=colors, alpha=0.8, width=0.5)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(tissues)
axes[0, 1].set_ylabel('Pearson r')
axes[0, 1].set_title('Edge Score Stability\n(Full vs Balanced)')
axes[0, 1].set_ylim(0.85, 1.02)
for i, v in enumerate(stab_corrs):
    axes[0, 1].text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=9)

# Panel C: Number of edges
n_edges = [imm['n_edges'], lng['n_edges'], kidney_results['n_edges']]
axes[1, 0].bar(x, n_edges, color=colors, alpha=0.8, width=0.5)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(tissues)
axes[1, 0].set_ylabel('N Edges')
axes[1, 0].set_title('Number of TF-Target Edges')
for i, v in enumerate(n_edges):
    axes[1, 0].text(i, v + 50, str(v), ha='center', fontsize=9)

# Panel D: Blacklist rate (method-based for kidney)
blacklist_pcts = [
    imm['pct_high_ASI'],
    lng['pct_high_ASI'],
    kidney_results['pct_high_ASI'],
]
labels_bl = ['Donor-balanced', 'Donor-balanced', 'Method-balanced']
bars = axes[1, 1].bar(x, blacklist_pcts, color=colors, alpha=0.8, width=0.5)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels([f'{t}\n({l})' for t, l in zip(tissues, labels_bl)], fontsize=9)
axes[1, 1].set_ylabel('Blacklisted Edges (%)')
axes[1, 1].set_title('Edge Blacklist Rate (ASI > 0.5)')
for i, v in enumerate(blacklist_pcts):
    axes[1, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

plt.suptitle('3-Tissue Leakage Comparison: Immune vs Lung vs Kidney', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'phase2_3tissue_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved phase2_3tissue_comparison.png")

# Save kidney results
kidney_results_file = os.path.join(DATADIR, 'phase2_kidney_results.json')
with open(kidney_results_file, 'w') as f:
    json.dump(kidney_results, f, indent=2)
print(f"Saved {kidney_results_file}")

print("\n--- Kidney characterization summary ---")
print(f"Kidney has {kidney_results['n_cells']} cells, 1 donor, 2 methods (10X: {adata_kidney.obs['method'].value_counts().get('10X', 0)}, SS2: {adata_kidney.obs['method'].value_counts().get('smartseq', 0)})")
print(f"Method-balanced ASI: mean={kidney_results['method_ASI_mean']:.4f}, median={kidney_results['method_ASI_median']:.4f}")
print(f"Blacklisted: {kidney_results['n_high_ASI']} / {kidney_results['n_edges']} ({kidney_results['pct_high_ASI']:.1f}%)")
print(f"Sign-flipped: {kidney_results['n_sign_flipped']}")
print(f"Edge stability r: {kidney_results['edge_score_correlation']:.4f}")
print(f"\nConclusion: Kidney (single donor) is {'more like lung (balanced)' if kidney_results['pct_high_ASI'] < 30 else 'more like immune (imbalanced)'} in terms of edge stability.")

print("\nPhase 2 Kidney + 3-tissue comparison complete.")
