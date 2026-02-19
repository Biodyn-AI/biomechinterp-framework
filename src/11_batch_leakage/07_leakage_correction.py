"""
Phase 2 - Objective 4: Leakage correction methods.

Test correction approaches and measure if they improve edge quality:
1. Regression-based: regress out donor/batch from per-cell expression before
   computing edge scores
2. ComBat-style: use scanpy's combat to remove batch effects, then recompute edges
3. Within-method computation: compute edges separately within 10X and smartseq,
   then average

Evaluate corrections by:
- Re-running donor leakage classifier (should decrease)
- Checking edge-score stability (should improve)
- Checking correlation with perturbation ground truth (should improve or stay same)
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr, mannwhitneyu
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def regress_out_covariates(X, covariates):
    """Regress out covariate effects from expression matrix."""
    X_corrected = X.copy()
    if sparse.issparse(X_corrected):
        X_corrected = X_corrected.toarray()

    # One-hot encode categorical covariates
    design = pd.get_dummies(covariates, drop_first=True).values.astype(float)

    for j in range(X_corrected.shape[1]):
        y = X_corrected[:, j]
        if np.std(y) < 1e-10:
            continue
        reg = LinearRegression()
        reg.fit(design, y)
        residuals = y - reg.predict(design)
        X_corrected[:, j] = residuals

    return X_corrected


# ============================================================
# MAIN ANALYSIS: LUNG DATASET (best for correction tests)
# ============================================================
print("=" * 70)
print("PHASE 2: LEAKAGE CORRECTION METHODS")
print("=" * 70)

all_correction_results = {}

for ds_name, ds_path in [
    ('Lung', '/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad'),
]:
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*60}")

    adata = load_and_prepare(ds_path, max_cells=20000)
    tf_idx = get_gene_indices(adata, TF_LIST)
    target_idx = get_gene_indices(adata, TARGET_POOL)

    # ---- Baseline: uncorrected ----
    print("\n--- Baseline (uncorrected) ---")
    edges_baseline = compute_edge_scores_dict(adata.X, tf_idx, target_idx)
    features_baseline = compute_per_cell_edge_features(adata.X, tf_idx, target_idx)
    ba_base, auc_base = donor_leakage_auc(features_baseline, adata.obs['donor_id'].values)
    print(f"  Donor leakage: bal_acc={ba_base:.4f}, AUC={auc_base:.4f}")
    print(f"  N edges: {len(edges_baseline)}")

    correction_results = {
        'baseline': {
            'donor_bal_acc': float(ba_base),
            'donor_auc': float(auc_base),
            'n_edges': len(edges_baseline),
        }
    }

    # ---- Correction 1: Regress out donor from expression ----
    print("\n--- Correction 1: Regress out donor ---")
    X_regressed_donor = regress_out_covariates(adata.X, adata.obs['donor_id'])
    edges_reg_donor = compute_edge_scores_dict(X_regressed_donor, tf_idx, target_idx)
    features_reg_donor = compute_per_cell_edge_features(X_regressed_donor, tf_idx, target_idx)
    ba_reg_d, auc_reg_d = donor_leakage_auc(features_reg_donor, adata.obs['donor_id'].values)
    print(f"  Donor leakage after regression: bal_acc={ba_reg_d:.4f}, AUC={auc_reg_d:.4f}")

    # Stability: compare regressed edges to baseline
    common = sorted(set(edges_baseline) & set(edges_reg_donor))
    r_reg_d, _ = pearsonr(
        [edges_baseline[e] for e in common],
        [edges_reg_donor[e] for e in common]
    )
    print(f"  Edge correlation (baseline vs donor-regressed): r = {r_reg_d:.4f}")

    correction_results['regress_donor'] = {
        'donor_bal_acc': float(ba_reg_d),
        'donor_auc': float(auc_reg_d),
        'edge_correlation_vs_baseline': float(r_reg_d),
        'n_edges': len(edges_reg_donor),
    }

    # ---- Correction 2: Regress out batch ----
    print("\n--- Correction 2: Regress out batch ---")
    X_regressed_batch = regress_out_covariates(adata.X, adata.obs['_scvi_batch'])
    edges_reg_batch = compute_edge_scores_dict(X_regressed_batch, tf_idx, target_idx)
    features_reg_batch = compute_per_cell_edge_features(X_regressed_batch, tf_idx, target_idx)
    ba_reg_b, auc_reg_b = donor_leakage_auc(features_reg_batch, adata.obs['donor_id'].values)
    print(f"  Donor leakage after batch regression: bal_acc={ba_reg_b:.4f}, AUC={auc_reg_b:.4f}")

    common_b = sorted(set(edges_baseline) & set(edges_reg_batch))
    r_reg_b, _ = pearsonr(
        [edges_baseline[e] for e in common_b],
        [edges_reg_batch[e] for e in common_b]
    )
    print(f"  Edge correlation (baseline vs batch-regressed): r = {r_reg_b:.4f}")

    correction_results['regress_batch'] = {
        'donor_bal_acc': float(ba_reg_b),
        'donor_auc': float(auc_reg_b),
        'edge_correlation_vs_baseline': float(r_reg_b),
        'n_edges': len(edges_reg_batch),
    }

    # ---- Correction 3: ComBat ----
    print("\n--- Correction 3: ComBat (scanpy) ---")
    adata_combat = adata.copy()
    # Make X dense for combat
    if sparse.issparse(adata_combat.X):
        adata_combat.X = adata_combat.X.toarray()
    sc.pp.combat(adata_combat, key='_scvi_batch')
    edges_combat = compute_edge_scores_dict(adata_combat.X, tf_idx, target_idx)
    features_combat = compute_per_cell_edge_features(adata_combat.X, tf_idx, target_idx)
    ba_combat, auc_combat = donor_leakage_auc(features_combat, adata.obs['donor_id'].values)
    print(f"  Donor leakage after ComBat: bal_acc={ba_combat:.4f}, AUC={auc_combat:.4f}")

    common_c = sorted(set(edges_baseline) & set(edges_combat))
    r_combat, _ = pearsonr(
        [edges_baseline[e] for e in common_c],
        [edges_combat[e] for e in common_c]
    )
    print(f"  Edge correlation (baseline vs ComBat): r = {r_combat:.4f}")

    correction_results['combat'] = {
        'donor_bal_acc': float(ba_combat),
        'donor_auc': float(auc_combat),
        'edge_correlation_vs_baseline': float(r_combat),
        'n_edges': len(edges_combat),
    }

    # ---- Correction 4: Within-method computation ----
    print("\n--- Correction 4: Within-method edge scores ---")
    methods = adata.obs['method'].unique()
    method_edges = {}
    for method in methods:
        mask = adata.obs['method'] == method
        if mask.sum() < 100:
            continue
        X_m = adata.X[mask]
        scores_m = compute_edge_scores_dict(X_m, tf_idx, target_idx)
        method_edges[method] = scores_m
        print(f"  {method}: {mask.sum()} cells, {len(scores_m)} edges")

    # Average edge scores across methods
    if len(method_edges) >= 2:
        all_edges_set = set()
        for s in method_edges.values():
            all_edges_set.update(s.keys())
        # Only keep edges present in both methods
        common_method_edges = set.intersection(*[set(s.keys()) for s in method_edges.values()])
        edges_avg_method = {}
        for e in common_method_edges:
            vals = [method_edges[m][e] for m in method_edges]
            edges_avg_method[e] = np.mean(vals)
        print(f"  Averaged {len(edges_avg_method)} edges across {len(method_edges)} methods")

        common_wm = sorted(set(edges_baseline) & set(edges_avg_method))
        r_wm, _ = pearsonr(
            [edges_baseline[e] for e in common_wm],
            [edges_avg_method[e] for e in common_wm]
        )
        print(f"  Edge correlation (baseline vs within-method avg): r = {r_wm:.4f}")

        correction_results['within_method_avg'] = {
            'edge_correlation_vs_baseline': float(r_wm),
            'n_edges': len(edges_avg_method),
            'n_common_edges': len(common_wm),
        }

    # ---- Correction 5: Regress out donor AND method ----
    print("\n--- Correction 5: Regress out donor + method ---")
    covariates_dm = pd.DataFrame({
        'donor': adata.obs['donor_id'].values,
        'method': adata.obs['method'].values,
    })
    X_regressed_dm = regress_out_covariates(adata.X, covariates_dm)
    edges_reg_dm = compute_edge_scores_dict(X_regressed_dm, tf_idx, target_idx)
    features_reg_dm = compute_per_cell_edge_features(X_regressed_dm, tf_idx, target_idx)
    ba_reg_dm, auc_reg_dm = donor_leakage_auc(features_reg_dm, adata.obs['donor_id'].values)
    print(f"  Donor leakage after donor+method regression: bal_acc={ba_reg_dm:.4f}, AUC={auc_reg_dm:.4f}")

    common_dm = sorted(set(edges_baseline) & set(edges_reg_dm))
    r_reg_dm, _ = pearsonr(
        [edges_baseline[e] for e in common_dm],
        [edges_reg_dm[e] for e in common_dm]
    )
    print(f"  Edge correlation (baseline vs donor+method-regressed): r = {r_reg_dm:.4f}")

    correction_results['regress_donor_method'] = {
        'donor_bal_acc': float(ba_reg_dm),
        'donor_auc': float(auc_reg_dm),
        'edge_correlation_vs_baseline': float(r_reg_dm),
        'n_edges': len(edges_reg_dm),
    }

    all_correction_results[ds_name] = correction_results

    # ---- ASI improvement: compare corrected edge scores to donor-balanced ----
    print("\n--- ASI improvement assessment ---")
    # Load Phase 1 sensitivity
    sens_path = os.path.join(DATADIR, f'{ds_name}_sensitivity.csv')
    if os.path.exists(sens_path):
        sens = pd.read_csv(sens_path)
        balanced_edges = dict(zip(sens['tf'] + '->' + sens['target'], sens['score_balanced']))

        for corr_name, corr_edges in [
            ('Baseline', edges_baseline),
            ('Regress donor', edges_reg_donor),
            ('Regress batch', edges_reg_batch),
            ('ComBat', edges_combat),
            ('Regress donor+method', edges_reg_dm),
        ]:
            common_se = sorted(set(corr_edges) & set(balanced_edges))
            if len(common_se) < 100:
                continue
            r_bal, _ = pearsonr(
                [corr_edges[e] for e in common_se],
                [balanced_edges[e] for e in common_se]
            )
            # Compute per-edge ASI
            asi_vals = []
            for e in common_se:
                full = corr_edges[e]
                bal = balanced_edges[e]
                asi = abs(full - bal) / max(abs(full), 0.01)
                asi_vals.append(asi)
            mean_asi = np.mean(asi_vals)
            pct_high = 100 * np.mean(np.array(asi_vals) > 0.5)
            print(f"  {corr_name}: r_vs_balanced={r_bal:.4f}, mean_ASI={mean_asi:.4f}, "
                  f"blacklisted={pct_high:.1f}%")

            correction_results[corr_name.lower().replace(' ', '_').replace('+', '_')] = \
                correction_results.get(corr_name.lower().replace(' ', '_').replace('+', '_'), {})
            key = corr_name.lower().replace(' ', '_').replace('+', '_')
            if key in correction_results:
                correction_results[key]['asi_mean_vs_balanced'] = float(mean_asi)
                correction_results[key]['pct_high_asi_vs_balanced'] = float(pct_high)
                correction_results[key]['r_vs_balanced'] = float(r_bal)


# ============================================================
# FIGURES
# ============================================================
print("\n--- Generating correction comparison figures ---")

for ds_name, results in all_correction_results.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    methods_list = ['baseline', 'regress_donor', 'regress_batch', 'combat', 'regress_donor_method']
    labels = ['Baseline', 'Regress\nDonor', 'Regress\nBatch', 'ComBat', 'Regress\nDonor+Method']
    colors = ['steelblue', 'darkorange', 'seagreen', 'purple', 'crimson']

    # Panel A: Donor leakage AUC
    aucs = [results[m].get('donor_auc', np.nan) for m in methods_list]
    valid_mask = ~np.isnan(aucs)
    x = np.arange(len(methods_list))
    bars = axes[0].bar(x[valid_mask], np.array(aucs)[valid_mask],
                       color=[colors[i] for i in range(len(methods_list)) if valid_mask[i]],
                       alpha=0.8, width=0.6)
    axes[0].axhline(0.5, color='red', linestyle='--', lw=1, label='Chance')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel('Donor Classification AUC')
    axes[0].set_title(f'{ds_name}: Donor Leakage After Correction')
    axes[0].set_ylim(0.4, 1.0)
    axes[0].legend()
    for i, v in enumerate(aucs):
        if not np.isnan(v):
            axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    # Panel B: Donor leakage balanced accuracy
    ba_vals = [results[m].get('donor_bal_acc', np.nan) for m in methods_list]
    chance = 0.25  # 4 donors in lung
    bars = axes[1].bar(x[valid_mask], np.array(ba_vals)[valid_mask],
                       color=[colors[i] for i in range(len(methods_list)) if valid_mask[i]],
                       alpha=0.8, width=0.6)
    axes[1].axhline(chance, color='red', linestyle='--', lw=1, label=f'Chance ({chance})')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel('Balanced Accuracy')
    axes[1].set_title(f'{ds_name}: Donor Bal. Accuracy After Correction')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    for i, v in enumerate(ba_vals):
        if not np.isnan(v):
            axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    # Panel C: Edge correlation vs baseline
    edge_corrs = [results[m].get('edge_correlation_vs_baseline', np.nan) for m in methods_list]
    edge_corrs[0] = 1.0  # baseline vs itself
    bars = axes[2].bar(x[valid_mask], np.array(edge_corrs)[valid_mask],
                       color=[colors[i] for i in range(len(methods_list)) if valid_mask[i]],
                       alpha=0.8, width=0.6)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel('Pearson r')
    axes[2].set_title(f'{ds_name}: Edge Score Preservation\n(vs Baseline)')
    axes[2].set_ylim(0.5, 1.05)
    for i, v in enumerate(edge_corrs):
        if not np.isnan(v):
            axes[2].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

    plt.suptitle(f'{ds_name}: Leakage Correction Method Comparison', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f'phase2_leakage_correction_{ds_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved phase2_leakage_correction_{ds_name}.png")

# Save all results
with open(os.path.join(DATADIR, 'phase2_leakage_correction_results.json'), 'w') as f:
    json.dump(all_correction_results, f, indent=2, default=str)
print("Saved phase2_leakage_correction_results.json")

print("\n--- Correction Summary ---")
for ds_name, results in all_correction_results.items():
    print(f"\n{ds_name}:")
    for method, r in results.items():
        auc = r.get('donor_auc', 'N/A')
        ba = r.get('donor_bal_acc', 'N/A')
        r_corr = r.get('edge_correlation_vs_baseline', 'N/A')
        print(f"  {method}: AUC={auc}, bal_acc={ba}, edge_r={r_corr}")

print("\nPhase 2 leakage correction complete.")
