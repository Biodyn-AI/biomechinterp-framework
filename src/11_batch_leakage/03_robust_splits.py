"""
Step 7: Robust held-out evaluation protocol.

Demonstrate leave-one-donor-out edge score stability and
cross-donor generalization tests.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os, time

warnings.filterwarnings('ignore')

OUTDIR = "/Users/ihorkendiukhov/biodyn-work/subproject_33_batch_donor_leakage_audit/reports/figures"
DATADIR = "/Users/ihorkendiukhov/biodyn-work/subproject_33_batch_donor_leakage_audit/implementation"

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


def get_gene_indices(adata, gene_list):
    var_names = list(adata.var_names)
    found = {}
    for g in gene_list:
        if g in var_names:
            found[g] = var_names.index(g)
    return found


def compute_edge_scores_from_X(X, tf_idx, target_idx):
    """Compute edge scores as Pearson correlations."""
    if sparse.issparse(X):
        X = X.toarray()

    scores = {}
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
                scores[f"{tf}→{tgt}"] = r
    return scores


# ============================================================
# Leave-One-Donor-Out (LODO) edge score stability
# ============================================================
print("=" * 60)
print("LEAVE-ONE-DONOR-OUT EDGE SCORE STABILITY")
print("=" * 60)

# Load lung data (4 donors, best for LODO)
print("\nLoading lung data...")
adata = ad.read_h5ad("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad")

# Convert gene names
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

# Subsample to 20k
np.random.seed(42)
if adata.n_obs > 20000:
    idx = np.random.choice(adata.n_obs, 20000, replace=False)
    adata = adata[idx].copy()

sc.pp.filter_genes(adata, min_cells=10)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

tf_idx = get_gene_indices(adata, TF_LIST)
target_idx = get_gene_indices(adata, TARGET_POOL)
print(f"Found {len(tf_idx)} TFs and {len(target_idx)} targets")

donors = adata.obs['donor_id'].unique().tolist()
print(f"Donors: {donors}")

# Global edge scores
print("\nComputing global edge scores...")
global_scores = compute_edge_scores_from_X(adata.X, tf_idx, target_idx)
print(f"  {len(global_scores)} edges")

# LODO: for each donor, compute edges WITHOUT that donor
lodo_scores = {}
for donor in donors:
    mask = adata.obs['donor_id'] != donor
    X_train = adata.X[mask]
    scores = compute_edge_scores_from_X(X_train, tf_idx, target_idx)
    lodo_scores[donor] = scores
    print(f"  LODO {donor}: {len(scores)} edges, {mask.sum()} cells")

# Compare LODO scores to global
common_edges = set(global_scores.keys())
for donor in donors:
    common_edges &= set(lodo_scores[donor].keys())
common_edges = sorted(common_edges)
print(f"\nCommon edges across all LODO folds: {len(common_edges)}")

# Build comparison matrix
global_vec = np.array([global_scores[e] for e in common_edges])
lodo_matrix = np.zeros((len(donors), len(common_edges)))
for i, donor in enumerate(donors):
    lodo_matrix[i] = [lodo_scores[donor][e] for e in common_edges]

# Cross-donor correlation
print("\nLODO vs Global correlation per donor:")
lodo_correlations = []
for i, donor in enumerate(donors):
    r, _ = pearsonr(global_vec, lodo_matrix[i])
    lodo_correlations.append(r)
    print(f"  {donor}: r = {r:.6f}")

# Pairwise donor-excluded correlation
print("\nPairwise LODO correlations:")
pairwise_r = np.zeros((len(donors), len(donors)))
for i in range(len(donors)):
    for j in range(len(donors)):
        r, _ = pearsonr(lodo_matrix[i], lodo_matrix[j])
        pairwise_r[i, j] = r

for i, d1 in enumerate(donors):
    for j, d2 in enumerate(donors):
        if i < j:
            print(f"  {d1} vs {d2}: r = {pairwise_r[i,j]:.6f}")

# Edge-wise LODO variance
edge_lodo_std = np.std(lodo_matrix, axis=0)
print(f"\nEdge LODO std: mean={edge_lodo_std.mean():.6f}, "
      f"median={np.median(edge_lodo_std):.6f}, max={edge_lodo_std.max():.6f}")

# Identify most unstable edges under LODO
unstable_idx = np.argsort(edge_lodo_std)[-20:]
print("\nTop 20 LODO-unstable edges:")
for idx in unstable_idx[::-1]:
    edge = common_edges[idx]
    print(f"  {edge}: std={edge_lodo_std[idx]:.4f}, "
          f"global={global_vec[idx]:.4f}, range=[{lodo_matrix[:,idx].min():.4f}, {lodo_matrix[:,idx].max():.4f}]")

# ============================================================
# Figures
# ============================================================

# Figure: LODO pairwise correlation heatmap
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: Pairwise LODO correlation
im = axes[0].imshow(pairwise_r, cmap='RdYlGn', vmin=0.95, vmax=1.0)
axes[0].set_xticks(range(len(donors)))
axes[0].set_xticklabels(donors, rotation=45, ha='right')
axes[0].set_yticks(range(len(donors)))
axes[0].set_yticklabels(donors)
axes[0].set_title('Pairwise LODO Edge Score\nCorrelation (Lung)')
plt.colorbar(im, ax=axes[0])
# Add text annotations
for i in range(len(donors)):
    for j in range(len(donors)):
        axes[0].text(j, i, f'{pairwise_r[i,j]:.4f}', ha='center', va='center', fontsize=8)

# Panel B: LODO stability distribution
axes[1].hist(edge_lodo_std, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[1].axvline(np.median(edge_lodo_std), color='red', linestyle='--',
               label=f'Median={np.median(edge_lodo_std):.4f}')
axes[1].set_xlabel('LODO Standard Deviation')
axes[1].set_ylabel('Number of Edges')
axes[1].set_title('Edge Score Stability\nAcross LODO Folds')
axes[1].legend()

# Panel C: Global vs LODO-mean scatter
lodo_mean = np.mean(lodo_matrix, axis=0)
r_overall, _ = pearsonr(global_vec, lodo_mean)
axes[2].scatter(global_vec, lodo_mean, s=3, alpha=0.3, c='steelblue')
axes[2].plot([-0.6, 0.85], [-0.6, 0.85], 'k--', lw=1, alpha=0.5)
axes[2].set_xlabel('Global Edge Score')
axes[2].set_ylabel('Mean LODO Edge Score')
axes[2].set_title(f'Global vs LODO-Mean\n(r = {r_overall:.5f})')

plt.suptitle('Leave-One-Donor-Out Stability Analysis (Lung)', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'Lung_LODO_stability.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved Lung_LODO_stability.png")


# ============================================================
# Now do the same LODO for immune subset
# ============================================================
print("\n" + "=" * 60)
print("LODO FOR IMMUNE SUBSET")
print("=" * 60)

adata_imm = ad.read_h5ad("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad")
if 'feature_name' in adata_imm.var.columns and adata_imm.var_names[0].startswith('ENSG'):
    gene_names = adata_imm.var['feature_name'].values.astype(str)
    seen = {}
    unique_names = []
    for name in gene_names:
        if name in seen:
            seen[name] += 1
            unique_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            unique_names.append(name)
    adata_imm.var_names = unique_names
    adata_imm.var_names_make_unique()

sc.pp.filter_genes(adata_imm, min_cells=10)
sc.pp.normalize_total(adata_imm, target_sum=1e4)
sc.pp.log1p(adata_imm)

tf_idx_imm = get_gene_indices(adata_imm, TF_LIST)
target_idx_imm = get_gene_indices(adata_imm, TARGET_POOL)

# Select top donors with enough cells
donor_counts = adata_imm.obs['donor_id'].value_counts()
top_donors = donor_counts[donor_counts >= 200].index.tolist()
print(f"Donors with >= 200 cells: {top_donors}")

# Global edge scores
global_scores_imm = compute_edge_scores_from_X(adata_imm.X, tf_idx_imm, target_idx_imm)

# LODO for top donors
lodo_scores_imm = {}
for donor in top_donors:
    mask = adata_imm.obs['donor_id'] != donor
    scores = compute_edge_scores_from_X(adata_imm.X[mask], tf_idx_imm, target_idx_imm)
    lodo_scores_imm[donor] = scores
    print(f"  LODO {donor}: {len(scores)} edges")

common_edges_imm = set(global_scores_imm.keys())
for donor in top_donors:
    common_edges_imm &= set(lodo_scores_imm[donor].keys())
common_edges_imm = sorted(common_edges_imm)

global_vec_imm = np.array([global_scores_imm[e] for e in common_edges_imm])
lodo_matrix_imm = np.zeros((len(top_donors), len(common_edges_imm)))
for i, donor in enumerate(top_donors):
    lodo_matrix_imm[i] = [lodo_scores_imm[donor][e] for e in common_edges_imm]

print(f"\nLODO correlations (Immune):")
for i, donor in enumerate(top_donors):
    r, _ = pearsonr(global_vec_imm, lodo_matrix_imm[i])
    print(f"  {donor}: r = {r:.6f}")

edge_lodo_std_imm = np.std(lodo_matrix_imm, axis=0)
print(f"\nEdge LODO std (Immune): mean={edge_lodo_std_imm.mean():.6f}, "
      f"median={np.median(edge_lodo_std_imm):.6f}, max={edge_lodo_std_imm.max():.6f}")

# Figure for immune LODO
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(edge_lodo_std_imm, bins=50, color='darkorange', edgecolor='white', alpha=0.8)
axes[0].axvline(np.median(edge_lodo_std_imm), color='red', linestyle='--',
               label=f'Median={np.median(edge_lodo_std_imm):.4f}')
axes[0].set_xlabel('LODO Standard Deviation')
axes[0].set_ylabel('Number of Edges')
axes[0].set_title('Edge Score Stability Across LODO Folds')
axes[0].legend()

lodo_mean_imm = np.mean(lodo_matrix_imm, axis=0)
r_imm, _ = pearsonr(global_vec_imm, lodo_mean_imm)
axes[1].scatter(global_vec_imm, lodo_mean_imm, s=3, alpha=0.3, c='darkorange')
lims = [min(global_vec_imm.min(), lodo_mean_imm.min()) - 0.05,
        max(global_vec_imm.max(), lodo_mean_imm.max()) + 0.05]
axes[1].plot(lims, lims, 'k--', lw=1, alpha=0.5)
axes[1].set_xlabel('Global Edge Score')
axes[1].set_ylabel('Mean LODO Edge Score')
axes[1].set_title(f'Global vs LODO-Mean (r = {r_imm:.5f})')

plt.suptitle('Leave-One-Donor-Out Stability (Immune Subset)', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'Immune_LODO_stability.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved Immune_LODO_stability.png")


# ============================================================
# Cross-donor cell-type classification (robust split test)
# ============================================================
print("\n" + "=" * 60)
print("CROSS-DONOR CELL TYPE CLASSIFICATION")
print("=" * 60)

# Use lung data: train on 3 donors, test on 1
from sklearn.metrics import classification_report

# Compute per-cell features
X_lung = adata.X
if sparse.issparse(X_lung):
    X_lung = X_lung.toarray()

tf_names = list(tf_idx.keys())
target_names = list(target_idx.keys())

# Edge product features (top 200 by variance)
all_products = []
edge_nm = []
for tf in tf_names:
    tf_expr = X_lung[:, tf_idx[tf]]
    for tgt in target_names:
        if tf == tgt:
            continue
        tgt_expr = X_lung[:, target_idx[tgt]]
        all_products.append(tf_expr * tgt_expr)
        edge_nm.append(f"{tf}→{tgt}")

all_products = np.column_stack(all_products)
variances = np.var(all_products, axis=0)
top_200 = np.argsort(variances)[-200:]
X_feat = all_products[:, top_200]

# Cell type labels
le = LabelEncoder()
cell_types = adata.obs['cell_type'].values
y_ct = le.fit_transform(cell_types)

# Leave-one-donor-out classification
results_lodo_clf = []
for test_donor in donors:
    train_mask = adata.obs['donor_id'] != test_donor
    test_mask = adata.obs['donor_id'] == test_donor

    X_train = X_feat[train_mask]
    X_test = X_feat[test_mask]
    y_train = y_ct[train_mask]
    y_test = y_ct[test_mask]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    # Only evaluate on cell types present in both train and test
    common_ct = set(np.unique(y_train)) & set(np.unique(y_test))
    if len(common_ct) < 2:
        print(f"  {test_donor}: too few common cell types, skipping")
        continue

    train_keep = np.isin(y_train, list(common_ct))
    test_keep = np.isin(y_test, list(common_ct))

    clf = LogisticRegression(max_iter=2000, random_state=42, multi_class='ovr', C=1.0)
    clf.fit(X_train[train_keep], y_train[train_keep])
    y_pred = clf.predict(X_test[test_keep])
    bal_acc = balanced_accuracy_score(y_test[test_keep], y_pred)
    results_lodo_clf.append({
        'test_donor': test_donor,
        'balanced_accuracy': bal_acc,
        'n_test': test_keep.sum(),
        'n_common_ct': len(common_ct),
    })
    print(f"  Test donor {test_donor}: bal_acc={bal_acc:.3f}, "
          f"n_test={test_keep.sum()}, n_cell_types={len(common_ct)}")

# Compare to within-donor CV (no donor split)
from sklearn.model_selection import cross_val_score
X_all_scaled = StandardScaler().fit_transform(X_feat)
X_all_scaled = np.nan_to_num(X_all_scaled)
cv_scores = cross_val_score(
    LogisticRegression(max_iter=2000, random_state=42, multi_class='ovr', C=1.0),
    X_all_scaled, y_ct, cv=5, scoring='balanced_accuracy'
)
print(f"\nWithin-study 5-fold CV: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
print(f"Cross-donor mean: {np.mean([r['balanced_accuracy'] for r in results_lodo_clf]):.3f}")

gap = cv_scores.mean() - np.mean([r['balanced_accuracy'] for r in results_lodo_clf])
print(f"Generalization gap: {gap:.3f}")

# Save LODO classification figure
fig, ax = plt.subplots(figsize=(8, 5))
donors_tested = [r['test_donor'] for r in results_lodo_clf]
accs = [r['balanced_accuracy'] for r in results_lodo_clf]
x = np.arange(len(donors_tested))
bars = ax.bar(x, accs, color='steelblue', alpha=0.8, label='Cross-donor')
ax.axhline(cv_scores.mean(), color='red', linestyle='--', lw=2,
          label=f'Within-study CV mean: {cv_scores.mean():.3f}')
ax.axhline(np.mean(accs), color='orange', linestyle='--', lw=2,
          label=f'Cross-donor mean: {np.mean(accs):.3f}')
ax.set_xticks(x)
ax.set_xticklabels(donors_tested, rotation=45, ha='right')
ax.set_ylabel('Balanced Accuracy')
ax.set_title(f'Cell Type Classification: Cross-Donor vs Within-Study\n'
            f'Generalization gap = {gap:.3f}')
ax.legend()
ax.set_ylim(0, 1)

for i, v in enumerate(accs):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'Lung_cross_donor_generalization.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved Lung_cross_donor_generalization.png")

print("\nDone with robust splits analysis.")
