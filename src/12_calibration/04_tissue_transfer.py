"""
Phase 2, Objective 1: Multi-tissue calibration transfer.

Train calibration models on immune (Dixit/Adamson K562) data and test on
lung and kidney tissue data (Tabula Sapiens). Quantify the calibration gap
when applying immune-trained calibration to different tissues.

Strategy:
- Compute edge scores on Tabula Sapiens lung/kidney control cells using
  the same methods as Phase 1 (Pearson, Spearman, MI, partial corr, ensemble)
- Apply calibration models (Platt + isotonic) trained on K562 data to these
  tissue-specific scores
- Also train tissue-specific calibration models for comparison
- Quantify calibration transfer gap: ECE(cross-tissue) - ECE(within-tissue)
"""

import scanpy as sc
import numpy as np
import pandas as pd
from scipy import stats, sparse
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
import pickle
import os
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

OUT_DIR = '/Users/ihorkendiukhov/biodyn-work/subproject_31_uncertainty_calibration_edge_scores/implementation'

# ============================================================
# Helper functions (reused from Phase 1)
# ============================================================

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def compute_mce(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    mce = 0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        mce = max(mce, abs(bin_acc - bin_conf))
    return mce


def normalize_scores(scores):
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-12:
        return np.full_like(scores, 0.5)
    return (scores - mn) / (mx - mn)


def compute_edge_scores_from_expression(X_ctrl, gene_list, target_genes, n_sub=1500):
    """Compute Pearson, Spearman, MI, partial corr, ensemble scores for target_genes."""
    n_cells, n_genes = X_ctrl.shape
    edge_scores = {}

    # Pearson
    X_c = X_ctrl - X_ctrl.mean(0, keepdims=True)
    norms = np.sqrt((X_c ** 2).sum(0, keepdims=True)) + 1e-12
    X_n = X_c / norms

    # Spearman
    X_ranked = np.zeros_like(X_ctrl)
    for j in range(n_genes):
        X_ranked[:, j] = stats.rankdata(X_ctrl[:, j])
    X_rc = X_ranked - X_ranked.mean(0, keepdims=True)
    norms_r = np.sqrt((X_rc ** 2).sum(0, keepdims=True)) + 1e-12
    X_rn = X_rc / norms_r

    # Partial correlation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ctrl)
    pca = PCA(n_components=min(10, n_genes - 1, n_cells - 1), random_state=42)
    pcs = pca.fit_transform(X_scaled)
    beta = np.linalg.lstsq(pcs, X_scaled, rcond=None)[0]
    X_resid = X_scaled - pcs @ beta
    X_resid_c = X_resid - X_resid.mean(0, keepdims=True)
    norms_res = np.sqrt((X_resid_c ** 2).sum(0, keepdims=True)) + 1e-12
    X_resid_n = X_resid_c / norms_res

    # MI subsample
    sub_idx = np.random.choice(n_cells, min(n_sub, n_cells), replace=False)
    X_sub = X_ctrl[sub_idx]

    for method in ['pearson', 'spearman', 'mutual_info', 'partial_corr']:
        edge_scores[method] = {}

    for tg in target_genes:
        if tg not in gene_list:
            continue
        pi = gene_list.index(tg)

        # Pearson
        corrs_p = X_n[:, pi] @ X_n
        # Spearman
        corrs_s = X_rn[:, pi] @ X_rn
        # Partial corr
        corrs_pc = X_resid_n[:, pi] @ X_resid_n
        # MI
        mi_vals = mutual_info_regression(X_sub, X_sub[:, pi], n_neighbors=5, random_state=42)

        for j, gn in enumerate(gene_list):
            if gn != tg:
                edge_scores['pearson'][(tg, gn)] = float(np.abs(corrs_p[j]))
                edge_scores['spearman'][(tg, gn)] = float(np.abs(corrs_s[j]))
                edge_scores['partial_corr'][(tg, gn)] = float(np.abs(corrs_pc[j]))
                edge_scores['mutual_info'][(tg, gn)] = float(mi_vals[j])

    # Ensemble
    all_edges = set()
    for m in edge_scores:
        all_edges.update(edge_scores[m].keys())
    edge_scores['ensemble'] = {}
    for edge in all_edges:
        vals = [edge_scores[m][edge] for m in edge_scores if edge in edge_scores[m]]
        if vals:
            edge_scores['ensemble'][edge] = float(np.mean(vals))

    return edge_scores


# ============================================================
# Load Phase 1 K562 calibration data (training source)
# ============================================================
print("=" * 70, flush=True)
print("PHASE 2: MULTI-TISSUE CALIBRATION TRANSFER", flush=True)
print("=" * 70, flush=True)

print("\n--- Loading Phase 1 K562 data ---", flush=True)
with open(os.path.join(OUT_DIR, 'edge_scores_and_ground_truth.pkl'), 'rb') as f:
    k562_data = pickle.load(f)
with open(os.path.join(OUT_DIR, 'calibration_results.pkl'), 'rb') as f:
    k562_cal = pickle.load(f)

k562_methods = k562_data['methods']
k562_gt = k562_data['ground_truth']
k562_scores = k562_data['edge_scores']
k562_gene_list = k562_data['gene_list']

print(f"  K562 methods: {k562_methods}", flush=True)
print(f"  K562 ground truth edges: {len(k562_gt)}", flush=True)

# Train K562 calibration models on ALL K562 data (for cross-tissue transfer)
print("\n--- Training K562 calibration models (all data) ---", flush=True)
k562_calibrators = {}
for method in ['pearson', 'spearman', 'mutual_info', 'partial_corr', 'ensemble']:
    scores_dict = k562_scores[method]
    y_all, s_all = [], []
    for (pert, target), score in scores_dict.items():
        if (pert, target) in k562_gt:
            y_all.append(int(k562_gt[(pert, target)]))
            s_all.append(score)
    y_all = np.array(y_all)
    s_all = np.array(s_all)
    s_norm = normalize_scores(s_all)

    # Platt
    platt = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    platt.fit(s_norm.reshape(-1, 1), y_all)

    # Isotonic
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso.fit(s_norm, y_all)

    k562_calibrators[method] = {
        'platt': platt,
        'isotonic': iso,
        'score_min': s_all.min(),
        'score_max': s_all.max(),
    }
    print(f"  {method}: trained on {len(y_all)} edges", flush=True)


# ============================================================
# Process Tabula Sapiens tissues
# ============================================================
def process_tissue(tissue_name, h5ad_path, reference_gene_list):
    """Process a tissue: compute scores using highly variable genes overlapping with reference."""
    print(f"\n--- Processing {tissue_name} ---", flush=True)
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Raw shape: {adata.shape}", flush=True)

    # Basic preprocessing
    sc.pp.filter_genes(adata, min_cells=int(0.01 * adata.shape[0]))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Find HVGs
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
    hvg = set(adata.var_names[adata.var['highly_variable']])

    # Overlap with K562 reference gene list
    shared = hvg & set(reference_gene_list) & set(adata.var_names)
    # Also check non-HVG overlap for a broader pool
    all_shared = set(reference_gene_list) & set(adata.var_names)

    # Use shared HVGs; if too few, expand with all shared genes
    if len(shared) < 50:
        gene_subset = sorted(all_shared)[:300]
    else:
        gene_subset = sorted(shared)

    print(f"  Shared HVGs with K562: {len(shared)}", flush=True)
    print(f"  Using {len(gene_subset)} genes", flush=True)

    # Subset
    adata_sub = adata[:, gene_subset].copy()
    X = adata_sub.X
    if sparse.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    # Pick a subset of genes to act as "pseudo-perturbations"
    # Use genes that are both in K562 perturbation list and in this tissue
    k562_perts = set(k562_data['perturbations'])
    available_perts = k562_perts & set(gene_subset)

    # If not enough perturb genes overlap, pick high-variance genes as targets
    if len(available_perts) < 3:
        variances = X.var(axis=0)
        top_var_idx = np.argsort(variances)[-10:]
        available_perts = set(np.array(gene_subset)[top_var_idx])

    print(f"  Target genes for scoring: {len(available_perts)}", flush=True)

    # Compute edge scores
    tissue_scores = compute_edge_scores_from_expression(
        X, gene_subset, list(available_perts), n_sub=min(1500, X.shape[0]))

    return adata_sub, X, gene_subset, tissue_scores, available_perts


# Process kidney (smaller, faster)
kidney_adata, kidney_X, kidney_genes, kidney_scores, kidney_perts = process_tissue(
    'Kidney',
    '/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad',
    k562_gene_list
)

# Process lung (larger -- subsample for speed)
print("\n--- Loading Lung (subsampling for speed) ---", flush=True)
adata_lung_raw = sc.read_h5ad(
    '/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad')
# Subsample to ~15000 cells for tractability
if adata_lung_raw.shape[0] > 15000:
    sc.pp.subsample(adata_lung_raw, n_obs=15000, random_state=42)
    print(f"  Subsampled lung to {adata_lung_raw.shape[0]} cells", flush=True)

# Save subsampled then process
import tempfile
tmp_path = os.path.join(OUT_DIR, '_tmp_lung_sub.h5ad')
adata_lung_raw.write_h5ad(tmp_path)

lung_adata, lung_X, lung_genes, lung_scores, lung_perts = process_tissue(
    'Lung', tmp_path, k562_gene_list
)
os.remove(tmp_path)

# ============================================================
# Create pseudo-ground-truth for tissue data
# ============================================================
# Without perturbation data for these tissues, we create a proxy ground truth
# based on gene co-expression strength: top 5% strongest co-expressed pairs
# as "positives" (mimicking the rank-based approach from Phase 1).
# This is explicitly a proxy -- we note the limitation.

def make_coexpression_ground_truth(X, gene_list, target_genes, top_pct=5):
    """Create proxy ground truth from co-expression strength."""
    gt = {}
    X_c = X - X.mean(0, keepdims=True)
    norms = np.sqrt((X_c ** 2).sum(0, keepdims=True)) + 1e-12
    X_n = X_c / norms

    for tg in target_genes:
        if tg not in gene_list:
            continue
        pi = gene_list.index(tg)
        corrs = np.abs(X_n[:, pi] @ X_n)
        thresh = np.percentile(corrs, 100 - top_pct)
        for j, gn in enumerate(gene_list):
            if gn != tg:
                gt[(tg, gn)] = bool(corrs[j] >= thresh)
    return gt


print("\n--- Creating proxy ground truth ---", flush=True)
kidney_gt = make_coexpression_ground_truth(kidney_X, kidney_genes, list(kidney_perts))
lung_gt = make_coexpression_ground_truth(lung_X, lung_genes, list(lung_perts))
print(f"  Kidney: {sum(kidney_gt.values())} true / {len(kidney_gt)} total "
      f"({100 * sum(kidney_gt.values()) / max(len(kidney_gt), 1):.1f}%)", flush=True)
print(f"  Lung: {sum(lung_gt.values())} true / {len(lung_gt)} total "
      f"({100 * sum(lung_gt.values()) / max(len(lung_gt), 1):.1f}%)", flush=True)


# ============================================================
# Evaluate calibration transfer
# ============================================================
print("\n" + "=" * 70, flush=True)
print("CALIBRATION TRANSFER ANALYSIS", flush=True)
print("=" * 70, flush=True)

transfer_results = {}

for tissue_name, tissue_scores, tissue_gt in [
    ('kidney', kidney_scores, kidney_gt),
    ('lung', lung_scores, lung_gt)
]:
    print(f"\n--- {tissue_name.upper()} ---", flush=True)
    transfer_results[tissue_name] = {}

    for method in ['pearson', 'spearman', 'mutual_info', 'partial_corr', 'ensemble']:
        if method not in tissue_scores:
            continue
        scores_dict = tissue_scores[method]

        y_all, s_all = [], []
        for (pert, target), score in scores_dict.items():
            if (pert, target) in tissue_gt:
                y_all.append(int(tissue_gt[(pert, target)]))
                s_all.append(score)

        if len(y_all) < 50:
            print(f"  {method}: too few edges ({len(y_all)}), skipping", flush=True)
            continue

        y_all = np.array(y_all)
        s_all = np.array(s_all)
        s_norm = normalize_scores(s_all)

        # 1. Raw ECE (no calibration)
        raw_ece = compute_ece(y_all, s_norm)
        raw_mce = compute_mce(y_all, s_norm)
        raw_brier = brier_score_loss(y_all, s_norm)
        try:
            raw_auc = roc_auc_score(y_all, s_all)
        except:
            raw_auc = float('nan')

        # 2. Cross-tissue calibration (K562 -> tissue)
        cal = k562_calibrators.get(method)
        if cal is not None:
            # Normalize using K562 score range
            k_mn, k_mx = cal['score_min'], cal['score_max']
            if k_mx - k_mn > 1e-12:
                s_k562_norm = np.clip((s_all - k_mn) / (k_mx - k_mn), 0, 1)
            else:
                s_k562_norm = np.full_like(s_all, 0.5)

            # Platt cross-tissue
            cross_platt_probs = cal['platt'].predict_proba(s_k562_norm.reshape(-1, 1))[:, 1]
            cross_platt_ece = compute_ece(y_all, cross_platt_probs)
            cross_platt_brier = brier_score_loss(y_all, cross_platt_probs)

            # Isotonic cross-tissue
            cross_iso_probs = cal['isotonic'].predict(s_k562_norm)
            cross_iso_ece = compute_ece(y_all, cross_iso_probs)
            cross_iso_brier = brier_score_loss(y_all, cross_iso_probs)
        else:
            cross_platt_ece = cross_platt_brier = float('nan')
            cross_iso_ece = cross_iso_brier = float('nan')

        # 3. Within-tissue calibration (train on 70%, test on 30%)
        n_split = int(0.7 * len(y_all))
        idx = np.random.permutation(len(y_all))
        train_idx, test_idx = idx[:n_split], idx[n_split:]

        s_train_n = normalize_scores(s_all[train_idx])
        mn_all, mx_all = s_all.min(), s_all.max()
        s_test_n = (s_all[test_idx] - mn_all) / (mx_all - mn_all + 1e-12)
        s_test_n = np.clip(s_test_n, 0, 1)

        # Platt within-tissue
        platt_local = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        platt_local.fit(s_train_n.reshape(-1, 1), y_all[train_idx])
        local_platt_probs = platt_local.predict_proba(s_test_n.reshape(-1, 1))[:, 1]
        local_platt_ece = compute_ece(y_all[test_idx], local_platt_probs)
        local_platt_brier = brier_score_loss(y_all[test_idx], local_platt_probs)

        # Isotonic within-tissue
        iso_local = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        iso_local.fit(s_train_n, y_all[train_idx])
        local_iso_probs = iso_local.predict(s_test_n)
        local_iso_ece = compute_ece(y_all[test_idx], local_iso_probs)
        local_iso_brier = brier_score_loss(y_all[test_idx], local_iso_probs)

        result = {
            'n_edges': len(y_all),
            'pos_rate': y_all.mean(),
            'raw_ece': raw_ece,
            'raw_mce': raw_mce,
            'raw_brier': raw_brier,
            'raw_auc': raw_auc,
            'cross_platt_ece': cross_platt_ece,
            'cross_platt_brier': cross_platt_brier,
            'cross_iso_ece': cross_iso_ece,
            'cross_iso_brier': cross_iso_brier,
            'local_platt_ece': local_platt_ece,
            'local_platt_brier': local_platt_brier,
            'local_iso_ece': local_iso_ece,
            'local_iso_brier': local_iso_brier,
            'calibration_gap_platt': cross_platt_ece - local_platt_ece,
            'calibration_gap_iso': cross_iso_ece - local_iso_ece,
        }
        transfer_results[tissue_name][method] = result

        print(f"  {method}:", flush=True)
        print(f"    Raw ECE={raw_ece:.4f}, AUROC={raw_auc:.4f}", flush=True)
        print(f"    Cross-tissue (K562->): Platt ECE={cross_platt_ece:.4f}, Iso ECE={cross_iso_ece:.4f}", flush=True)
        print(f"    Within-tissue:         Platt ECE={local_platt_ece:.4f}, Iso ECE={local_iso_ece:.4f}", flush=True)
        print(f"    Calibration gap (Platt): {result['calibration_gap_platt']:.4f}", flush=True)
        print(f"    Calibration gap (Iso):   {result['calibration_gap_iso']:.4f}", flush=True)


# ============================================================
# Save
# ============================================================
print("\n--- Saving ---", flush=True)
output = {
    'transfer_results': transfer_results,
    'tissues': ['kidney', 'lung'],
    'methods': ['pearson', 'spearman', 'mutual_info', 'partial_corr', 'ensemble'],
}

with open(os.path.join(OUT_DIR, 'phase2_tissue_transfer.pkl'), 'wb') as f:
    pickle.dump(output, f)

# Summary table
print("\n" + "=" * 70, flush=True)
print("SUMMARY: CALIBRATION TRANSFER GAP", flush=True)
print("=" * 70, flush=True)

rows = []
for tissue in ['kidney', 'lung']:
    for method in ['pearson', 'spearman', 'mutual_info', 'partial_corr', 'ensemble']:
        if method in transfer_results[tissue]:
            r = transfer_results[tissue][method]
            rows.append({
                'Tissue': tissue,
                'Method': method,
                'N_edges': r['n_edges'],
                'AUROC': f"{r['raw_auc']:.3f}",
                'Raw_ECE': f"{r['raw_ece']:.3f}",
                'Cross_Platt_ECE': f"{r['cross_platt_ece']:.3f}",
                'Cross_Iso_ECE': f"{r['cross_iso_ece']:.3f}",
                'Local_Platt_ECE': f"{r['local_platt_ece']:.3f}",
                'Local_Iso_ECE': f"{r['local_iso_ece']:.3f}",
                'Gap_Platt': f"{r['calibration_gap_platt']:.3f}",
                'Gap_Iso': f"{r['calibration_gap_iso']:.3f}",
            })

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False), flush=True)
summary_df.to_csv(os.path.join(OUT_DIR, 'phase2_tissue_transfer_summary.csv'), index=False)

print("\nDone with Phase 2 tissue transfer.", flush=True)
