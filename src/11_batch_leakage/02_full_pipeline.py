"""
Batch and Donor Leakage Audit for Mechanistic Edge Scores.

Full pipeline:
1. Load data, define TF-target gene pairs
2. Compute per-cell edge scores (correlation-based)
3. Leakage classifier: predict donor/batch from edge score vectors
4. Stratified resampling: donor-balanced edge scores
5. Artifact Sensitivity Index per edge
6. Generate all figures and summary statistics
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                             classification_report, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
import time

warnings.filterwarnings('ignore')

OUTDIR = "/Users/ihorkendiukhov/biodyn-work/subproject_33_batch_donor_leakage_audit/reports/figures"
DATADIR = "/Users/ihorkendiukhov/biodyn-work/subproject_33_batch_donor_leakage_audit/implementation"
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# Known human TF list (curated subset)
# ============================================================
# Core transcription factors relevant to immune/lung biology
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

# Known target genes (highly variable, immune/lung relevant)
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
    """Load h5ad, subset to manageable size, ensure dense for correlation."""
    print(f"Loading {path}...")
    adata = ad.read_h5ad(path)
    print(f"  Raw shape: {adata.shape}")

    # Convert var index from Ensembl IDs to gene symbols if needed
    if 'feature_name' in adata.var.columns and adata.var_names[0].startswith('ENSG'):
        print("  Converting Ensembl IDs to gene symbols via feature_name...")
        gene_names = adata.var['feature_name'].values.astype(str)
        # Handle duplicates by appending suffix
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

    # Subsample if too large
    if adata.n_obs > max_cells:
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, max_cells, replace=False)
        adata = adata[idx].copy()
        print(f"  Subsampled to: {adata.shape}")

    # Basic QC filtering already done in Tabula Sapiens, but ensure gene presence
    sc.pp.filter_genes(adata, min_cells=10)
    print(f"  After gene filter: {adata.shape}")

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def get_gene_indices(adata, gene_list):
    """Map gene names to column indices in adata.X."""
    var_names = list(adata.var_names)
    found = {}
    for g in gene_list:
        if g in var_names:
            found[g] = var_names.index(g)
    return found


def compute_edge_scores(X, tf_indices, target_indices, method='pearson'):
    """
    Compute TF-target edge scores as correlations across cells.

    Returns: DataFrame with columns [tf, target, score]
    """
    if sparse.issparse(X):
        X = X.toarray()

    edges = []
    tf_names = list(tf_indices.keys())
    target_names = list(target_indices.keys())

    for tf in tf_names:
        tf_expr = X[:, tf_indices[tf]]
        # Skip TFs with zero variance
        if np.std(tf_expr) < 1e-10:
            continue
        for tgt in target_names:
            if tf == tgt:
                continue
            tgt_expr = X[:, target_indices[tgt]]
            if np.std(tgt_expr) < 1e-10:
                continue
            if method == 'pearson':
                r, _ = pearsonr(tf_expr, tgt_expr)
            else:
                r, _ = spearmanr(tf_expr, tgt_expr)
            edges.append({'tf': tf, 'target': tgt, 'score': r})

    return pd.DataFrame(edges)


def compute_pseudobulk_edge_vectors(adata, tf_idx, target_idx, group_col,
                                     min_cells_per_group=30):
    """
    Compute edge score vectors per group (e.g., per donor, per batch).
    For each group, compute correlations within that group's cells.
    Returns: matrix of shape (n_groups, n_edges), group labels, edge names
    """
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()

    groups = adata.obs[group_col].values
    unique_groups = [g for g in pd.Series(groups).value_counts().index
                     if pd.Series(groups == g).sum() >= min_cells_per_group]

    tf_names = list(tf_idx.keys())
    target_names = list(target_idx.keys())
    edge_names = [f"{tf}→{tgt}" for tf in tf_names for tgt in target_names if tf != tgt]

    vectors = []
    valid_groups = []

    for grp in unique_groups:
        mask = groups == grp
        X_grp = X[mask]
        if X_grp.shape[0] < min_cells_per_group:
            continue

        vec = []
        for tf in tf_names:
            tf_expr = X_grp[:, tf_idx[tf]]
            if np.std(tf_expr) < 1e-10:
                for tgt in target_names:
                    if tf != tgt:
                        vec.append(0.0)
                continue
            for tgt in target_names:
                if tf == tgt:
                    continue
                tgt_expr = X_grp[:, target_idx[tgt]]
                if np.std(tgt_expr) < 1e-10:
                    vec.append(0.0)
                else:
                    r, _ = pearsonr(tf_expr, tgt_expr)
                    vec.append(r if np.isfinite(r) else 0.0)
        vectors.append(vec)
        valid_groups.append(grp)

    return np.array(vectors), valid_groups, edge_names


def compute_per_cell_edge_features(adata, tf_idx, target_idx, n_edges=200):
    """
    Compute per-cell edge features: product of TF and target expression.
    This is a proxy for edge activation at single-cell level.
    Returns: feature matrix (n_cells, n_edges), edge names
    """
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()

    tf_names = list(tf_idx.keys())
    target_names = list(target_idx.keys())

    # Select top edges by variance of product
    all_products = []
    all_names = []
    for tf in tf_names:
        tf_expr = X[:, tf_idx[tf]]
        for tgt in target_names:
            if tf == tgt:
                continue
            tgt_expr = X[:, target_idx[tgt]]
            prod = tf_expr * tgt_expr
            all_products.append(prod)
            all_names.append(f"{tf}→{tgt}")

    all_products = np.column_stack(all_products)
    # Select top N by variance
    variances = np.var(all_products, axis=0)
    top_idx = np.argsort(variances)[-n_edges:]
    return all_products[:, top_idx], [all_names[i] for i in top_idx]


def leakage_classifier(features, labels, label_name, n_splits=5):
    """
    Train leakage classifiers and return performance metrics.
    Uses cross-validation to avoid overfitting.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    if n_classes < 2:
        print(f"  Skipping {label_name}: only {n_classes} class(es)")
        return None

    # Filter classes with too few samples
    class_counts = pd.Series(y).value_counts()
    valid_classes = class_counts[class_counts >= n_splits].index.tolist()
    if len(valid_classes) < 2:
        print(f"  Skipping {label_name}: too few samples per class for CV")
        return None

    mask = np.isin(y, valid_classes)
    X = features[mask]
    y = y[mask]
    # Re-encode
    le2 = LabelEncoder()
    y = le2.fit_transform(y)
    n_classes = len(le2.classes_)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    results = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for clf_name, clf in [
        ('LogisticRegression', LogisticRegression(max_iter=2000, random_state=42,
                                                   multi_class='ovr', C=1.0)),
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42,
                                                 max_depth=10, n_jobs=-1)),
    ]:
        y_pred = cross_val_predict(clf, X, y, cv=skf, method='predict')
        y_proba = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')

        acc = accuracy_score(y, y_pred)
        bal_acc = balanced_accuracy_score(y, y_pred)
        chance = 1.0 / n_classes

        # AUC (macro-averaged for multiclass)
        if n_classes == 2:
            auc_val = roc_auc_score(y, y_proba[:, 1])
        else:
            try:
                auc_val = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')
            except:
                auc_val = np.nan

        results[clf_name] = {
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'auc': auc_val,
            'chance_level': chance,
            'n_classes': n_classes,
            'n_samples': len(y),
            'y_true': y,
            'y_proba': y_proba,
            'y_pred': y_pred,
            'class_names': le2.classes_,
        }

        print(f"  {clf_name} → acc={acc:.3f}, bal_acc={bal_acc:.3f}, "
              f"AUC={auc_val:.3f}, chance={chance:.3f}, n_classes={n_classes}")

    return results


def plot_roc_curves(results_dict, title, filename):
    """Plot ROC curves for leakage classifiers."""
    fig, axes = plt.subplots(1, len(results_dict), figsize=(6*len(results_dict), 5))
    if len(results_dict) == 1:
        axes = [axes]

    for ax, (clf_name, res) in zip(axes, results_dict.items()):
        y_true = res['y_true']
        y_proba = res['y_proba']
        n_classes = res['n_classes']

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            ax.plot(fpr, tpr, lw=2, label=f'AUC = {res["auc"]:.3f}')
        else:
            # One-vs-rest ROC per class
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=range(n_classes))
            for i in range(min(n_classes, 10)):  # Limit to 10 classes for readability
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                roc_auc_i = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=1.5, alpha=0.7,
                        label=f'Class {i} (AUC={roc_auc_i:.2f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{clf_name}\nMacro AUC={res["auc"]:.3f}')
        ax.legend(loc='lower right', fontsize=7)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def balanced_resample(adata, group_col, seed=42):
    """Resample to balance cells across groups."""
    np.random.seed(seed)
    groups = adata.obs[group_col].values
    unique_groups = pd.Series(groups).value_counts()
    min_count = unique_groups.min()
    # Use at least 50 cells per group
    min_count = max(min_count, 50)

    idx = []
    for grp in unique_groups.index:
        grp_idx = np.where(groups == grp)[0]
        n_sample = min(min_count, len(grp_idx))
        chosen = np.random.choice(grp_idx, n_sample, replace=False)
        idx.extend(chosen)

    return adata[idx].copy()


def compute_artifact_sensitivity(edges_full, edges_balanced):
    """
    Compute Artifact Sensitivity Index for each edge.
    ASI = |score_full - score_balanced| / max(|score_full|, epsilon)
    """
    merged = edges_full.merge(edges_balanced, on=['tf', 'target'],
                               suffixes=('_full', '_balanced'))
    merged['delta'] = merged['score_full'] - merged['score_balanced']
    merged['abs_delta'] = merged['delta'].abs()
    eps = 0.01
    merged['ASI'] = merged['abs_delta'] / np.maximum(merged['score_full'].abs(), eps)
    return merged


# ============================================================
# MAIN ANALYSIS
# ============================================================
def run_analysis_on_dataset(data_path, dataset_name, donor_col='donor_id',
                             batch_col='_scvi_batch', max_cells=20000):
    """Run the full leakage audit on one dataset."""
    print(f"\n{'#'*70}")
    print(f"# ANALYSIS: {dataset_name}")
    print(f"{'#'*70}")

    adata = load_and_prepare(data_path, max_cells=max_cells)

    # Find available genes
    tf_idx = get_gene_indices(adata, TF_LIST)
    target_idx = get_gene_indices(adata, TARGET_POOL)
    print(f"\nFound {len(tf_idx)} TFs and {len(target_idx)} targets in data")
    print(f"TFs: {list(tf_idx.keys())[:15]}...")
    print(f"Targets: {list(target_idx.keys())[:15]}...")

    all_results = {'dataset': dataset_name}

    # ----------------------------------------
    # Step 1: Compute global edge scores
    # ----------------------------------------
    print("\n--- Step 1: Global edge scores ---")
    t0 = time.time()
    edges_full = compute_edge_scores(adata.X, tf_idx, target_idx)
    print(f"  Computed {len(edges_full)} edges in {time.time()-t0:.1f}s")
    if len(edges_full) == 0:
        print("  ERROR: No edges computed. Check gene name matching.")
        return {'dataset': dataset_name, 'error': 'no_edges'}, None
    print(f"  Score stats: mean={edges_full['score'].mean():.4f}, "
          f"std={edges_full['score'].std():.4f}, "
          f"min={edges_full['score'].min():.4f}, max={edges_full['score'].max():.4f}")
    all_results['n_edges'] = len(edges_full)

    # ----------------------------------------
    # Step 2: Per-cell edge features for leakage classifiers
    # ----------------------------------------
    print("\n--- Step 2: Per-cell edge features ---")
    cell_features, feature_names = compute_per_cell_edge_features(
        adata, tf_idx, target_idx, n_edges=200)
    print(f"  Feature matrix shape: {cell_features.shape}")

    # ----------------------------------------
    # Step 3: Leakage classifiers
    # ----------------------------------------
    leakage_results = {}

    # Donor leakage
    n_donors = adata.obs[donor_col].nunique()
    print(f"\n--- Step 3a: Donor leakage classifier ({n_donors} donors) ---")
    if n_donors >= 2:
        donor_labels = adata.obs[donor_col].values
        donor_res = leakage_classifier(cell_features, donor_labels, 'Donor')
        if donor_res:
            leakage_results['donor'] = donor_res
            plot_roc_curves(donor_res,
                          f'{dataset_name}: Donor Leakage ROC',
                          f'{dataset_name}_donor_roc.png')
            all_results['donor_leakage'] = {
                clf: {k: v for k, v in r.items() if k not in ['y_true', 'y_proba', 'y_pred', 'class_names']}
                for clf, r in donor_res.items()
            }
    else:
        print("  Only 1 donor, skipping donor leakage test")

    # Batch leakage
    n_batches = adata.obs[batch_col].nunique()
    print(f"\n--- Step 3b: Batch leakage classifier ({n_batches} batches) ---")
    if n_batches >= 2:
        batch_labels = adata.obs[batch_col].values
        batch_res = leakage_classifier(cell_features, batch_labels, 'Batch')
        if batch_res:
            leakage_results['batch'] = batch_res
            plot_roc_curves(batch_res,
                          f'{dataset_name}: Batch Leakage ROC',
                          f'{dataset_name}_batch_roc.png')
            all_results['batch_leakage'] = {
                clf: {k: v for k, v in r.items() if k not in ['y_true', 'y_proba', 'y_pred', 'class_names']}
                for clf, r in batch_res.items()
            }

    # Method leakage (10X vs smartseq)
    if 'method' in adata.obs.columns:
        n_methods = adata.obs['method'].nunique()
        print(f"\n--- Step 3c: Method/assay leakage classifier ({n_methods} methods) ---")
        if n_methods >= 2:
            method_labels = adata.obs['method'].values
            method_res = leakage_classifier(cell_features, method_labels, 'Method')
            if method_res:
                leakage_results['method'] = method_res
                plot_roc_curves(method_res,
                              f'{dataset_name}: Method Leakage ROC',
                              f'{dataset_name}_method_roc.png')
                all_results['method_leakage'] = {
                    clf: {k: v for k, v in r.items() if k not in ['y_true', 'y_proba', 'y_pred', 'class_names']}
                    for clf, r in method_res.items()
                }

    # ----------------------------------------
    # Step 4: Donor-balanced resampling
    # ----------------------------------------
    print(f"\n--- Step 4: Donor-balanced resampling ---")
    if n_donors >= 2:
        adata_balanced = balanced_resample(adata, donor_col)
        print(f"  Balanced shape: {adata_balanced.shape}")
        print(f"  Donor distribution: {adata_balanced.obs[donor_col].value_counts().to_dict()}")

        edges_balanced = compute_edge_scores(adata_balanced.X, tf_idx, target_idx)
        print(f"  Balanced edges computed: {len(edges_balanced)}")

        # ----------------------------------------
        # Step 5: Artifact Sensitivity Index
        # ----------------------------------------
        print("\n--- Step 5: Artifact Sensitivity Index ---")
        sensitivity = compute_artifact_sensitivity(edges_full, edges_balanced)
        print(f"  ASI stats: mean={sensitivity['ASI'].mean():.4f}, "
              f"median={sensitivity['ASI'].median():.4f}, "
              f"max={sensitivity['ASI'].max():.4f}")

        # Top artifact-contaminated edges
        top_artifacts = sensitivity.nlargest(20, 'ASI')
        print(f"\n  Top 20 artifact-sensitive edges:")
        for _, row in top_artifacts.iterrows():
            print(f"    {row['tf']}→{row['target']}: ASI={row['ASI']:.3f}, "
                  f"full={row['score_full']:.4f}, balanced={row['score_balanced']:.4f}")

        all_results['ASI_mean'] = sensitivity['ASI'].mean()
        all_results['ASI_median'] = sensitivity['ASI'].median()
        all_results['n_high_ASI'] = int((sensitivity['ASI'] > 0.5).sum())
        all_results['pct_high_ASI'] = float((sensitivity['ASI'] > 0.5).mean() * 100)

        # ----------------------------------------
        # Step 5b: Batch-balanced resampling
        # ----------------------------------------
        if n_batches >= 2:
            print(f"\n--- Step 5b: Batch-balanced resampling ---")
            adata_batch_bal = balanced_resample(adata, batch_col)
            edges_batch_bal = compute_edge_scores(adata_batch_bal.X, tf_idx, target_idx)
            sensitivity_batch = compute_artifact_sensitivity(edges_full, edges_batch_bal)
            print(f"  Batch ASI stats: mean={sensitivity_batch['ASI'].mean():.4f}, "
                  f"median={sensitivity_batch['ASI'].median():.4f}")
            all_results['batch_ASI_mean'] = sensitivity_batch['ASI'].mean()
            all_results['batch_ASI_median'] = sensitivity_batch['ASI'].median()
        else:
            sensitivity_batch = None

        # ----------------------------------------
        # Step 6: Edge blacklisting
        # ----------------------------------------
        print("\n--- Step 6: Edge blacklisting ---")
        blacklist_threshold = 0.5
        blacklisted = sensitivity[sensitivity['ASI'] > blacklist_threshold]
        print(f"  Blacklisted edges (ASI > {blacklist_threshold}): "
              f"{len(blacklisted)} / {len(sensitivity)} "
              f"({100*len(blacklisted)/len(sensitivity):.1f}%)")

        # Also identify edges that flip sign
        sign_flipped = sensitivity[
            (sensitivity['score_full'] * sensitivity['score_balanced']) < 0
        ]
        print(f"  Sign-flipped edges: {len(sign_flipped)} / {len(sensitivity)} "
              f"({100*len(sign_flipped)/len(sensitivity):.1f}%)")

        all_results['n_blacklisted'] = len(blacklisted)
        all_results['n_sign_flipped'] = len(sign_flipped)

        # ----------------------------------------
        # Figures
        # ----------------------------------------
        print("\n--- Generating figures ---")

        # Figure: ASI distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(sensitivity['ASI'].values, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        axes[0].axvline(0.5, color='red', linestyle='--', label='Blacklist threshold')
        axes[0].set_xlabel('Artifact Sensitivity Index (Donor)')
        axes[0].set_ylabel('Number of edges')
        axes[0].set_title('Donor ASI Distribution')
        axes[0].legend()

        if sensitivity_batch is not None:
            axes[1].hist(sensitivity_batch['ASI'].values, bins=50, color='darkorange', edgecolor='white', alpha=0.8)
            axes[1].axvline(0.5, color='red', linestyle='--', label='Blacklist threshold')
            axes[1].set_xlabel('Artifact Sensitivity Index (Batch)')
            axes[1].set_ylabel('Number of edges')
            axes[1].set_title('Batch ASI Distribution')
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'Single batch\n(N/A)', ha='center', va='center',
                        transform=axes[1].transAxes, fontsize=14)
            axes[1].set_title('Batch ASI Distribution')

        plt.suptitle(f'{dataset_name}: Artifact Sensitivity Index', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f'{dataset_name}_ASI_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Figure: Before/After edge rank comparison
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(sensitivity['score_full'], sensitivity['score_balanced'],
                  alpha=0.3, s=5, c='steelblue')
        lims = [min(sensitivity['score_full'].min(), sensitivity['score_balanced'].min()) - 0.05,
                max(sensitivity['score_full'].max(), sensitivity['score_balanced'].max()) + 0.05]
        ax.plot(lims, lims, 'k--', alpha=0.5, lw=1)
        ax.set_xlabel('Edge Score (Full Dataset)')
        ax.set_ylabel('Edge Score (Donor-Balanced)')
        ax.set_title(f'{dataset_name}: Edge Score Stability Under Donor Balancing')

        # Highlight blacklisted
        bl = sensitivity[sensitivity['ASI'] > blacklist_threshold]
        ax.scatter(bl['score_full'], bl['score_balanced'],
                  alpha=0.5, s=10, c='red', label=f'ASI > {blacklist_threshold} (n={len(bl)})')
        ax.legend()

        # Add correlation
        r, _ = pearsonr(sensitivity['score_full'], sensitivity['score_balanced'])
        ax.text(0.05, 0.95, f'Pearson r = {r:.4f}', transform=ax.transAxes,
               fontsize=11, verticalalignment='top')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f'{dataset_name}_edge_score_stability.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        all_results['edge_score_correlation'] = r

        # Figure: Rank change distribution
        sensitivity['rank_full'] = sensitivity['score_full'].abs().rank(ascending=False)
        sensitivity['rank_balanced'] = sensitivity['score_balanced'].abs().rank(ascending=False)
        sensitivity['rank_change'] = sensitivity['rank_full'] - sensitivity['rank_balanced']

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(sensitivity['rank_change'].values, bins=80, color='steelblue',
               edgecolor='white', alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Rank Change (Full - Balanced)')
        ax.set_ylabel('Number of edges')
        ax.set_title(f'{dataset_name}: Edge Rank Shift Under Donor Balancing')
        ax.text(0.02, 0.95,
               f'Mean |shift|: {sensitivity["rank_change"].abs().mean():.0f}\n'
               f'Median |shift|: {sensitivity["rank_change"].abs().median():.0f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f'{dataset_name}_rank_change.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Save sensitivity data
        sensitivity.to_csv(os.path.join(DATADIR, f'{dataset_name}_sensitivity.csv'), index=False)

    else:
        sensitivity = None

    # ----------------------------------------
    # Step 7: Per-donor edge score heatmap
    # ----------------------------------------
    if n_donors >= 2:
        print("\n--- Step 7: Per-donor edge score profiles ---")
        donor_vectors, donor_labels, edge_names_vec = compute_pseudobulk_edge_vectors(
            adata, tf_idx, target_idx, donor_col, min_cells_per_group=30)
        print(f"  Got vectors for {len(donor_labels)} donors, {len(edge_names_vec)} edges")

        if len(donor_labels) >= 2 and len(edge_names_vec) > 0:
            # Compute inter-donor variance per edge
            edge_donor_var = np.var(donor_vectors, axis=0)
            top_var_idx = np.argsort(edge_donor_var)[-50:]  # Top 50 by variance

            fig, ax = plt.subplots(figsize=(14, max(4, len(donor_labels)*0.6)))
            heatmap_data = donor_vectors[:, top_var_idx]
            edge_labels = [edge_names_vec[i] for i in top_var_idx]

            im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r',
                          vmin=-0.3, vmax=0.3)
            ax.set_yticks(range(len(donor_labels)))
            ax.set_yticklabels(donor_labels, fontsize=8)
            ax.set_xlabel('Edges (top 50 by inter-donor variance)')
            ax.set_title(f'{dataset_name}: Per-Donor Edge Score Profiles')
            plt.colorbar(im, ax=ax, label='Pearson r')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTDIR, f'{dataset_name}_donor_heatmap.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    # ----------------------------------------
    # Cell type confound check
    # ----------------------------------------
    if 'cell_type' in adata.obs.columns and n_donors >= 2:
        print("\n--- Cell type vs donor confound check ---")
        ct_donor = pd.crosstab(adata.obs['cell_type'], adata.obs[donor_col])
        # Normalize per cell type
        ct_donor_norm = ct_donor.div(ct_donor.sum(axis=1), axis=0)
        print(f"  Cell type x Donor contingency table shape: {ct_donor.shape}")

        # Chi-squared test
        from scipy.stats import chi2_contingency
        chi2, p, dof, expected = chi2_contingency(ct_donor.values)
        cramers_v = np.sqrt(chi2 / (ct_donor.values.sum() * (min(ct_donor.shape) - 1)))
        print(f"  Chi-squared: {chi2:.1f}, p={p:.2e}, Cramer's V={cramers_v:.4f}")
        all_results['celltype_donor_cramersV'] = cramers_v
        all_results['celltype_donor_chi2_p'] = p

        fig, ax = plt.subplots(figsize=(12, max(6, ct_donor.shape[0]*0.3)))
        sns.heatmap(ct_donor_norm, cmap='YlOrRd', ax=ax, annot=False)
        ax.set_title(f'{dataset_name}: Cell Type Composition per Donor\n'
                    f"Cramer's V = {cramers_v:.4f}")
        ax.set_xlabel('Donor')
        ax.set_ylabel('Cell Type')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f'{dataset_name}_celltype_donor_confound.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    return all_results, sensitivity


# ============================================================
# RUN ALL ANALYSES
# ============================================================
if __name__ == '__main__':
    all_summaries = {}

    # Analysis 1: Immune subset (multi-donor)
    immune_results, immune_sens = run_analysis_on_dataset(
        "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad",
        "Immune",
        donor_col='donor_id',
        batch_col='_scvi_batch',
        max_cells=20000,
    )
    all_summaries['Immune'] = immune_results

    # Analysis 2: Lung (4 donors, 7 batches)
    lung_results, lung_sens = run_analysis_on_dataset(
        "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad",
        "Lung",
        donor_col='donor_id',
        batch_col='_scvi_batch',
        max_cells=20000,
    )
    all_summaries['Lung'] = lung_results

    # Analysis 3: Kidney (1 donor, but 2 assay types — batch effect test)
    kidney_results, kidney_sens = run_analysis_on_dataset(
        "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad",
        "Kidney",
        donor_col='donor_id',
        batch_col='_scvi_batch',
        max_cells=20000,
    )
    all_summaries['Kidney'] = kidney_results

    # ============================================================
    # SUMMARY FIGURE: Cross-dataset comparison
    # ============================================================
    print("\n--- Cross-dataset summary ---")

    # Compile summary table
    summary_rows = []
    for ds_name, res in all_summaries.items():
        row = {'Dataset': ds_name, 'N_edges': res.get('n_edges', 0)}

        for leakage_type in ['donor_leakage', 'batch_leakage', 'method_leakage']:
            if leakage_type in res:
                for clf_name in ['LogisticRegression', 'RandomForest']:
                    if clf_name in res[leakage_type]:
                        prefix = f'{leakage_type}_{clf_name}'
                        row[f'{prefix}_acc'] = res[leakage_type][clf_name]['accuracy']
                        row[f'{prefix}_bal_acc'] = res[leakage_type][clf_name]['balanced_accuracy']
                        row[f'{prefix}_auc'] = res[leakage_type][clf_name]['auc']
                        row[f'{prefix}_chance'] = res[leakage_type][clf_name]['chance_level']

        row['ASI_mean'] = res.get('ASI_mean', np.nan)
        row['ASI_median'] = res.get('ASI_median', np.nan)
        row['pct_high_ASI'] = res.get('pct_high_ASI', np.nan)
        row['edge_score_correlation'] = res.get('edge_score_correlation', np.nan)
        row['n_sign_flipped'] = res.get('n_sign_flipped', np.nan)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(DATADIR, 'cross_dataset_summary.csv'), index=False)
    print(summary_df.to_string())

    # Summary bar chart: classifier performance
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, leakage_type in enumerate(['donor_leakage', 'batch_leakage', 'method_leakage']):
        ax = axes[idx]
        datasets_with = []
        rf_accs = []
        lr_accs = []
        chances = []

        for ds_name, res in all_summaries.items():
            if leakage_type in res:
                datasets_with.append(ds_name)
                rf_accs.append(res[leakage_type].get('RandomForest', {}).get('balanced_accuracy', 0))
                lr_accs.append(res[leakage_type].get('LogisticRegression', {}).get('balanced_accuracy', 0))
                chances.append(res[leakage_type].get('RandomForest', {}).get('chance_level', 0))

        if datasets_with:
            x = np.arange(len(datasets_with))
            w = 0.3
            ax.bar(x - w/2, lr_accs, w, label='Logistic Regression', color='steelblue', alpha=0.8)
            ax.bar(x + w/2, rf_accs, w, label='Random Forest', color='darkorange', alpha=0.8)
            # Chance level line
            for i, c in enumerate(chances):
                ax.hlines(c, i-0.4, i+0.4, colors='red', linestyles='--', linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(datasets_with)
            ax.set_ylabel('Balanced Accuracy')
            ax.set_title(leakage_type.replace('_', ' ').title())
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(leakage_type.replace('_', ' ').title())

    plt.suptitle('Leakage Classifier Performance Across Datasets', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'cross_dataset_leakage_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save all results as JSON (excluding numpy arrays)
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [clean_for_json(x) for x in obj]
        elif pd.isna(obj):
            return None
        return obj

    with open(os.path.join(DATADIR, 'all_results.json'), 'w') as f:
        json.dump(clean_for_json(all_summaries), f, indent=2, default=str)

    print("\n\nALL ANALYSES COMPLETE.")
    print(f"Figures saved in: {OUTDIR}")
    print(f"Data saved in: {DATADIR}")
