"""
Phase 2 - Objective 2: Composition vs within-cell-type leakage decomposition.

Formally decompose leakage into:
  (a) Leakage due to cell-type proportion differences across donors
  (b) Leakage due to within-cell-type expression differences

Approach:
  1. Compute overall donor leakage AUC (baseline = Phase 1 result)
  2. Create cell-type-balanced subsamples (match cell-type proportions across donors)
     and re-measure donor leakage -> this removes composition effect
  3. Compute within-cell-type donor leakage for major cell types individually
  4. Decompose: composition_leakage = overall - balanced
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, json

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


def compute_per_cell_edge_features(X, tf_idx, target_idx, n_edges=200):
    if sparse.issparse(X):
        X = X.toarray()
    all_products = []
    for tf in tf_idx:
        tf_expr = X[:, tf_idx[tf]]
        for tgt in target_idx:
            if tf == tgt:
                continue
            all_products.append(tf_expr * X[:, target_idx[tgt]])
    all_products = np.column_stack(all_products)
    variances = np.var(all_products, axis=0)
    top_idx = np.argsort(variances)[-n_edges:]
    return all_products[:, top_idx]


def donor_leakage_auc(features, donor_labels, n_splits=5):
    """Return donor classifier AUC (LR)."""
    le = LabelEncoder()
    y = le.fit_transform(donor_labels)
    n_classes = len(le.classes_)
    if n_classes < 2:
        return None, None

    class_counts = pd.Series(y).value_counts()
    valid = class_counts[class_counts >= n_splits].index.tolist()
    if len(valid) < 2:
        return None, None

    mask = np.isin(y, valid)
    X = features[mask]
    y = y[mask]
    le2 = LabelEncoder()
    y = le2.fit_transform(y)
    n_classes = len(le2.classes_)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.nan_to_num(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000, random_state=42, multi_class='ovr', C=1.0)

    y_pred = cross_val_predict(clf, X, y, cv=skf, method='predict')
    y_proba = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')

    bal_acc = balanced_accuracy_score(y, y_pred)
    if n_classes == 2:
        auc_val = roc_auc_score(y, y_proba[:, 1])
    else:
        try:
            auc_val = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')
        except Exception:
            auc_val = np.nan

    return bal_acc, auc_val


def celltype_balanced_subsample(adata, donor_col='donor_id', ct_col='cell_type',
                                 min_cells_per_combo=10, seed=42):
    """
    Create a subsample where cell-type proportions are matched across donors.
    For each (cell_type, donor) combo, sample min_cells_per_combo cells.
    Only keep cell types present in all donors (with enough cells).
    """
    np.random.seed(seed)
    donors = adata.obs[donor_col].unique()

    # Find cell types present in all donors with enough cells
    ct_donor_counts = adata.obs.groupby([ct_col, donor_col]).size().unstack(fill_value=0)
    shared_cts = ct_donor_counts.index[ct_donor_counts.min(axis=1) >= min_cells_per_combo].tolist()

    if len(shared_cts) == 0:
        # Relax: cell types in at least 80% of donors
        n_donors = len(donors)
        threshold = max(2, int(0.8 * n_donors))
        shared_cts = ct_donor_counts.index[(ct_donor_counts > 0).sum(axis=1) >= threshold].tolist()
        # Lower min cells
        min_cells_per_combo = 5

    print(f"  Shared cell types for balancing: {len(shared_cts)} (min {min_cells_per_combo} cells/combo)")

    idx = []
    for ct in shared_cts:
        for donor in donors:
            mask = (adata.obs[ct_col] == ct) & (adata.obs[donor_col] == donor)
            cell_idx = np.where(mask.values)[0]
            if len(cell_idx) >= min_cells_per_combo:
                chosen = np.random.choice(cell_idx, min_cells_per_combo, replace=False)
                idx.extend(chosen)

    return adata[idx].copy(), shared_cts


# ============================================================
# ANALYSIS ON LUNG (4 donors, best for this decomposition)
# ============================================================
print("=" * 70)
print("PHASE 2: COMPOSITION vs WITHIN-CELL-TYPE LEAKAGE DECOMPOSITION")
print("=" * 70)

all_decomp_results = {}

for ds_name, ds_path in [
    ('Lung', '/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad'),
    ('Immune', '/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad'),
]:
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*60}")

    adata = load_and_prepare(ds_path, max_cells=20000)
    print(f"Shape: {adata.shape}")
    print(f"Donors: {adata.obs['donor_id'].nunique()}")
    print(f"Cell types: {adata.obs['cell_type'].nunique()}")

    tf_idx = get_gene_indices(adata, TF_LIST)
    target_idx = get_gene_indices(adata, TARGET_POOL)

    # ----- Step 1: Overall donor leakage (all cells, no composition control) -----
    print("\n--- Step 1: Overall donor leakage ---")
    features_all = compute_per_cell_edge_features(adata.X, tf_idx, target_idx)
    overall_bal_acc, overall_auc = donor_leakage_auc(features_all, adata.obs['donor_id'].values)
    print(f"  Overall: bal_acc={overall_bal_acc:.4f}, AUC={overall_auc:.4f}")

    # ----- Step 2: Cell-type-balanced donor leakage -----
    print("\n--- Step 2: Cell-type-balanced donor leakage ---")
    adata_ct_bal, shared_cts = celltype_balanced_subsample(
        adata, donor_col='donor_id', ct_col='cell_type',
        min_cells_per_combo=10
    )
    print(f"  CT-balanced shape: {adata_ct_bal.shape}")
    print(f"  Donors: {adata_ct_bal.obs['donor_id'].value_counts().to_dict()}")
    print(f"  Cell types used: {len(shared_cts)}")

    # Verify cell-type proportions are equal
    ct_prop = pd.crosstab(adata_ct_bal.obs['cell_type'], adata_ct_bal.obs['donor_id'])
    print(f"  CT x Donor table shape: {ct_prop.shape}")
    print(f"  All combos equal? {ct_prop.min().min() == ct_prop.max().max()}")

    features_bal = compute_per_cell_edge_features(adata_ct_bal.X, tf_idx, target_idx)
    balanced_bal_acc, balanced_auc = donor_leakage_auc(
        features_bal, adata_ct_bal.obs['donor_id'].values
    )
    print(f"  CT-balanced: bal_acc={balanced_bal_acc:.4f}, AUC={balanced_auc:.4f}")

    # ----- Step 3: Within-cell-type donor leakage -----
    print("\n--- Step 3: Within-cell-type donor leakage ---")
    within_ct_results = []
    for ct in shared_cts:
        ct_mask = adata.obs['cell_type'] == ct
        adata_ct = adata[ct_mask]
        n_donors_ct = adata_ct.obs['donor_id'].nunique()
        if n_donors_ct < 2 or adata_ct.n_obs < 50:
            continue

        features_ct = compute_per_cell_edge_features(adata_ct.X, tf_idx, target_idx, n_edges=100)
        ba, au = donor_leakage_auc(features_ct, adata_ct.obs['donor_id'].values)
        if ba is not None:
            within_ct_results.append({
                'cell_type': ct,
                'n_cells': int(adata_ct.n_obs),
                'n_donors': n_donors_ct,
                'balanced_accuracy': ba,
                'auc': au,
            })
            print(f"    {ct}: n={adata_ct.n_obs}, donors={n_donors_ct}, "
                  f"bal_acc={ba:.4f}, AUC={au:.4f}")

    within_ct_df = pd.DataFrame(within_ct_results)
    mean_within_auc = within_ct_df['auc'].mean() if len(within_ct_df) > 0 else np.nan
    mean_within_ba = within_ct_df['balanced_accuracy'].mean() if len(within_ct_df) > 0 else np.nan

    # ----- Decomposition -----
    print(f"\n--- Decomposition ({ds_name}) ---")
    print(f"  Overall donor AUC:                 {overall_auc:.4f}")
    print(f"  CT-balanced donor AUC:             {balanced_auc:.4f}")
    print(f"  Mean within-CT donor AUC:          {mean_within_auc:.4f}")
    print(f"  Composition-attributable AUC drop:  {overall_auc - balanced_auc:.4f}")
    print(f"  Residual within-CT leakage:         {balanced_auc - 0.5:.4f} (above chance)")

    decomp = {
        'overall_bal_acc': overall_bal_acc,
        'overall_auc': overall_auc,
        'ct_balanced_bal_acc': balanced_bal_acc,
        'ct_balanced_auc': balanced_auc,
        'mean_within_ct_bal_acc': mean_within_ba,
        'mean_within_ct_auc': mean_within_auc,
        'composition_auc_drop': overall_auc - balanced_auc,
        'residual_within_ct': balanced_auc - 0.5,
        'n_shared_cts': len(shared_cts),
        'n_cells_balanced': int(adata_ct_bal.n_obs),
        'within_ct_details': within_ct_results,
    }
    all_decomp_results[ds_name] = decomp

    # ----- Step 4: Edge score decomposition -----
    # Compute edge scores globally, on CT-balanced subset, and within each cell type
    print(f"\n--- Step 4: Edge score level decomposition ---")

    def compute_edge_scores_dict(X, tf_idx, target_idx):
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
                    scores[f"{tf}->{tgt}"] = r
        return scores

    global_edges = compute_edge_scores_dict(adata.X, tf_idx, target_idx)
    balanced_edges = compute_edge_scores_dict(adata_ct_bal.X, tf_idx, target_idx)

    common_edges = sorted(set(global_edges) & set(balanced_edges))
    g_vec = np.array([global_edges[e] for e in common_edges])
    b_vec = np.array([balanced_edges[e] for e in common_edges])

    r_gb, _ = pearsonr(g_vec, b_vec)
    print(f"  Global vs CT-balanced edge correlation: r = {r_gb:.4f}")
    decomp['edge_global_vs_ctbal_r'] = float(r_gb)

    # Within-cell-type aggregate: average edge scores across cell types
    ct_edge_scores = {}
    for ct in shared_cts:
        ct_mask = adata.obs['cell_type'] == ct
        if ct_mask.sum() < 30:
            continue
        ct_scores = compute_edge_scores_dict(adata[ct_mask].X, tf_idx, target_idx)
        ct_edge_scores[ct] = ct_scores

    # Compute mean within-CT edge score
    if ct_edge_scores:
        all_ct_edges = set()
        for s in ct_edge_scores.values():
            all_ct_edges.update(s.keys())
        common_ct_edges = sorted(all_ct_edges & set(common_edges))

        within_mean = np.zeros(len(common_ct_edges))
        count = np.zeros(len(common_ct_edges))
        for ct, scores in ct_edge_scores.items():
            for i, e in enumerate(common_ct_edges):
                if e in scores:
                    within_mean[i] += scores[e]
                    count[i] += 1
        valid = count > 0
        within_mean[valid] /= count[valid]

        g_sub = np.array([global_edges[e] for e in common_ct_edges])
        r_within, _ = pearsonr(g_sub[valid], within_mean[valid])
        print(f"  Global vs mean-within-CT edge correlation: r = {r_within:.4f}")
        decomp['edge_global_vs_within_ct_r'] = float(r_within)


# ============================================================
# FIGURES
# ============================================================
print("\n" + "=" * 70)
print("GENERATING DECOMPOSITION FIGURES")
print("=" * 70)

# Figure 1: Decomposition bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, ds_name in enumerate(['Lung', 'Immune']):
    ax = axes[i]
    d = all_decomp_results[ds_name]
    chance = 0.5

    categories = ['Overall\n(all cells)', 'CT-Balanced\n(matched proportions)',
                   'Mean Within-CT\n(per cell type)']
    aucs = [d['overall_auc'], d['ct_balanced_auc'], d['mean_within_ct_auc']]
    colors = ['steelblue', 'darkorange', 'seagreen']

    bars = ax.bar(range(3), aucs, color=colors, alpha=0.8, width=0.6)
    ax.axhline(chance, color='red', linestyle='--', lw=1, label='Chance (0.5)')
    ax.set_xticks(range(3))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Donor Classification AUC')
    ax.set_title(f'{ds_name}\nComposition drop: {d["composition_auc_drop"]:.3f}')
    ax.set_ylim(0.4, 1.0)
    ax.legend()

    for j, v in enumerate(aucs):
        ax.text(j, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    # Annotate decomposition
    ax.annotate('', xy=(0, aucs[0]), xytext=(1, aucs[1]),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    mid_y = (aucs[0] + aucs[1]) / 2
    ax.text(0.5, mid_y, f'Composition\neffect\n({d["composition_auc_drop"]:.3f})',
            ha='center', va='center', fontsize=8, color='gray')

plt.suptitle('Leakage Decomposition: Composition vs Within-Cell-Type',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'phase2_composition_decomposition.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved phase2_composition_decomposition.png")

# Figure 2: Within-cell-type donor leakage by cell type
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for i, ds_name in enumerate(['Lung', 'Immune']):
    ax = axes[i]
    d = all_decomp_results[ds_name]
    wct = pd.DataFrame(d['within_ct_details'])
    if len(wct) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        continue

    wct = wct.sort_values('auc', ascending=True)
    y_pos = range(len(wct))
    ax.barh(y_pos, wct['auc'].values, color='steelblue', alpha=0.8, height=0.7)
    ax.axvline(0.5, color='red', linestyle='--', lw=1, label='Chance')
    ax.set_yticks(y_pos)
    ct_labels = [f"{ct} (n={n})" for ct, n in zip(wct['cell_type'], wct['n_cells'])]
    ax.set_yticklabels(ct_labels, fontsize=7)
    ax.set_xlabel('Donor Classification AUC')
    ax.set_title(f'{ds_name}: Within-Cell-Type Donor Leakage')
    ax.legend(fontsize=8)

plt.suptitle('Within-Cell-Type Donor Leakage per Cell Type', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'phase2_within_ct_leakage.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved phase2_within_ct_leakage.png")

# Save all results
with open(os.path.join(DATADIR, 'phase2_composition_decomposition.json'), 'w') as f:
    json.dump(all_decomp_results, f, indent=2, default=str)
print("Saved phase2_composition_decomposition.json")

print("\n--- Summary ---")
for ds_name, d in all_decomp_results.items():
    print(f"\n{ds_name}:")
    print(f"  Overall AUC:         {d['overall_auc']:.4f}")
    print(f"  CT-balanced AUC:     {d['ct_balanced_auc']:.4f}")
    print(f"  Within-CT mean AUC:  {d['mean_within_ct_auc']:.4f}")
    print(f"  Composition drop:    {d['composition_auc_drop']:.4f} "
          f"({100*d['composition_auc_drop']/max(d['overall_auc']-0.5, 0.01):.1f}% of excess AUC)")

print("\nPhase 2 composition decomposition complete.")
