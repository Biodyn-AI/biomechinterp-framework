"""
Phase 2, Objective 3: Shifrut dataset validation.

Add Shifrut CRISPR perturbation data as an additional validation source.
Shifrut uses a different perturbation technology (CRISPR knockout in T cells
vs CRISPRi in K562), testing whether calibration quality holds across
perturbation modalities and cell types.

Strategy:
- Load Shifrut data, compute edge scores with same pipeline
- Derive ground truth from Shifrut perturbation effects
- Test K562-trained calibration models on Shifrut data
- Compare within-Shifrut vs cross-dataset calibration
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
from statsmodels.stats.multitest import multipletests
import pickle
import os
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

OUT_DIR = '/Users/ihorkendiukhov/biodyn-work/subproject_31_uncertainty_calibration_edge_scores/implementation'


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


# ============================================================
# Load Phase 1 K562 data and calibrators
# ============================================================
print("=" * 70, flush=True)
print("PHASE 2: SHIFRUT DATASET VALIDATION", flush=True)
print("=" * 70, flush=True)

print("\n--- Loading Phase 1 K562 data ---", flush=True)
with open(os.path.join(OUT_DIR, 'edge_scores_and_ground_truth.pkl'), 'rb') as f:
    k562_data = pickle.load(f)

k562_gt = k562_data['ground_truth']
k562_scores = k562_data['edge_scores']
k562_gene_list = k562_data['gene_list']

# Train K562 calibration models
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

    platt = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    platt.fit(s_norm.reshape(-1, 1), y_all)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso.fit(s_norm, y_all)

    k562_calibrators[method] = {
        'platt': platt,
        'isotonic': iso,
        'score_min': s_all.min(),
        'score_max': s_all.max(),
    }

# ============================================================
# Load and process Shifrut data
# ============================================================
print("\n--- Loading Shifrut data ---", flush=True)
adata = sc.read_h5ad(
    '/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/perturb/shifrut/perturb_processed_symbols.h5ad')
print(f"  Shape: {adata.shape}", flush=True)

# Preprocessing
sc.pp.filter_genes(adata, min_cells=int(0.02 * adata.shape[0]))
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

shifrut_perts = [c for c in adata.obs['condition'].unique() if c != 'control']
print(f"  Perturbations: {shifrut_perts}", flush=True)

# Select HVGs
ctrl_cells = adata[adata.obs['condition'] == 'control'].copy()
sc.pp.highly_variable_genes(ctrl_cells, n_top_genes=1000, flavor='seurat')
hvg = set(ctrl_cells.var_names[ctrl_cells.var['highly_variable']])

# Overlap with K562 genes
shared_genes = hvg & set(k562_gene_list) & set(adata.var_names)
# Add perturbation genes that exist
for p in shifrut_perts:
    if p in adata.var_names:
        shared_genes.add(p)

gene_list = sorted(shared_genes)
print(f"  Shared HVGs with K562: {len(gene_list)}", flush=True)

# Subset
adata_sub = adata[:, [g for g in gene_list if g in adata.var_names]].copy()
gene_list = list(adata_sub.var_names)

# Filter perturbations to those in gene list
perts_in = [p for p in shifrut_perts if p in gene_list]
print(f"  Perturbations in gene list: {len(perts_in)} -> {perts_in}", flush=True)

# ============================================================
# Compute Shifrut ground truth (same hybrid strategy as Phase 1)
# ============================================================
print("\n--- Computing Shifrut ground truth ---", flush=True)

gt_shifrut = {}
de_scores_shifrut = {}

for pert in perts_in:
    mask_p = adata_sub.obs['condition'] == pert
    mask_c = adata_sub.obs['condition'] == 'control'
    n1 = mask_p.sum()
    n2 = mask_c.sum()
    if n1 < 10 or n2 < 10:
        print(f"  {pert}: too few cells (n_pert={n1}, n_ctrl={n2}), skipping", flush=True)
        continue

    X_p = adata_sub[mask_p].X
    X_c = adata_sub[mask_c].X
    if sparse.issparse(X_p):
        X_p = X_p.toarray()
    if sparse.issparse(X_c):
        X_c = X_c.toarray()
    X_p, X_c = X_p.astype(np.float64), X_c.astype(np.float64)

    mean_p, mean_c = X_p.mean(0), X_c.mean(0)
    var_p, var_c = X_p.var(0, ddof=1), X_c.var(0, ddof=1)
    se = np.sqrt(var_p / n1 + var_c / n2 + 1e-15)
    t_stat = (mean_p - mean_c) / se
    df_t = ((var_p / n1 + var_c / n2) ** 2 /
            ((var_p / n1) ** 2 / (n1 - 1) + (var_c / n2) ** 2 / (n2 - 1) + 1e-15))
    pvals = np.nan_to_num(2 * stats.t.sf(np.abs(t_stat), df_t), nan=1.0)
    lfc = mean_p - mean_c

    _, padj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    de_score = -np.log10(pvals + 1e-300) * np.abs(lfc)

    for j, gn in enumerate(gene_list):
        if gn != pert:
            gt_key = (pert, gn)
            de_scores_shifrut[gt_key] = float(de_score[j])
            gt_shifrut[gt_key] = bool(padj[j] < 0.05 and abs(lfc[j]) > 0.1)

# Rank-based augmentation
perts_with_gt = set(k for k, _ in gt_shifrut.keys())
gt_rank = {}
for pert in perts_with_gt:
    pert_edges = {k: v for k, v in de_scores_shifrut.items() if k[0] == pert}
    if len(pert_edges) == 0:
        continue
    scores_arr = np.array(list(pert_edges.values()))
    thresh = np.percentile(scores_arr, 95)
    for k, v in pert_edges.items():
        gt_rank[k] = gt_shifrut.get(k, False) or (v >= thresh)

ground_truth_shifrut = gt_rank
total_true = sum(v for v in ground_truth_shifrut.values())
print(f"  Ground truth: {total_true} true / {len(ground_truth_shifrut)} total "
      f"({100 * total_true / max(len(ground_truth_shifrut), 1):.1f}%)", flush=True)

# ============================================================
# Compute edge scores from Shifrut control cells
# ============================================================
print("\n--- Computing edge scores from Shifrut control cells ---", flush=True)

ctrl_sub = adata_sub[adata_sub.obs['condition'] == 'control']
X_ctrl = ctrl_sub.X
if sparse.issparse(X_ctrl):
    X_ctrl = X_ctrl.toarray()
X_ctrl = X_ctrl.astype(np.float32)
print(f"  Control matrix: {X_ctrl.shape}", flush=True)

shifrut_scores = {}
score_perts = list(set(perts_in) & perts_with_gt)

# Pearson
print("  Pearson...", flush=True)
X_c = X_ctrl - X_ctrl.mean(0, keepdims=True)
norms = np.sqrt((X_c ** 2).sum(0, keepdims=True)) + 1e-12
X_n = X_c / norms
pearson_scores = {}
for pert in score_perts:
    pi = gene_list.index(pert)
    corrs = X_n[:, pi] @ X_n
    for j, gn in enumerate(gene_list):
        if gn != pert and (pert, gn) in ground_truth_shifrut:
            pearson_scores[(pert, gn)] = float(np.abs(corrs[j]))
shifrut_scores['pearson'] = pearson_scores

# Spearman
print("  Spearman...", flush=True)
X_ranked = np.zeros_like(X_ctrl)
for j in range(X_ctrl.shape[1]):
    X_ranked[:, j] = stats.rankdata(X_ctrl[:, j])
X_rc = X_ranked - X_ranked.mean(0, keepdims=True)
norms_r = np.sqrt((X_rc ** 2).sum(0, keepdims=True)) + 1e-12
X_rn = X_rc / norms_r
spearman_scores = {}
for pert in score_perts:
    pi = gene_list.index(pert)
    corrs = X_rn[:, pi] @ X_rn
    for j, gn in enumerate(gene_list):
        if gn != pert and (pert, gn) in ground_truth_shifrut:
            spearman_scores[(pert, gn)] = float(np.abs(corrs[j]))
shifrut_scores['spearman'] = spearman_scores

# Mutual information
print("  Mutual information...", flush=True)
n_sub = min(1500, X_ctrl.shape[0])
sub_idx = np.random.choice(X_ctrl.shape[0], n_sub, replace=False)
X_sub = X_ctrl[sub_idx]
mi_scores = {}
for pert in score_perts:
    pi = gene_list.index(pert)
    mi_vals = mutual_info_regression(X_sub, X_sub[:, pi], n_neighbors=5, random_state=42)
    for j, gn in enumerate(gene_list):
        if gn != pert and (pert, gn) in ground_truth_shifrut:
            mi_scores[(pert, gn)] = float(mi_vals[j])
    print(f"    {pert}: done", flush=True)
shifrut_scores['mutual_info'] = mi_scores

# Partial correlation
print("  Partial correlation...", flush=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ctrl)
pca = PCA(n_components=10, random_state=42)
pcs = pca.fit_transform(X_scaled)
beta_coef = np.linalg.lstsq(pcs, X_scaled, rcond=None)[0]
X_resid = X_scaled - pcs @ beta_coef
X_resid_c = X_resid - X_resid.mean(0, keepdims=True)
norms_res = np.sqrt((X_resid_c ** 2).sum(0, keepdims=True)) + 1e-12
X_resid_n = X_resid_c / norms_res
partial_corr_scores = {}
for pert in score_perts:
    pi = gene_list.index(pert)
    corrs = X_resid_n[:, pi] @ X_resid_n
    for j, gn in enumerate(gene_list):
        if gn != pert and (pert, gn) in ground_truth_shifrut:
            partial_corr_scores[(pert, gn)] = float(np.abs(corrs[j]))
shifrut_scores['partial_corr'] = partial_corr_scores

# Ensemble
print("  Ensemble...", flush=True)
all_edges = set()
for m in shifrut_scores:
    all_edges.update(shifrut_scores[m].keys())
ensemble_scores = {}
for edge in all_edges:
    vals = [shifrut_scores[m][edge] for m in shifrut_scores if edge in shifrut_scores[m]]
    if vals:
        ensemble_scores[edge] = float(np.mean(vals))
shifrut_scores['ensemble'] = ensemble_scores

# ============================================================
# Evaluate calibration: K562-trained vs within-Shifrut
# ============================================================
print("\n" + "=" * 70, flush=True)
print("CALIBRATION EVALUATION ON SHIFRUT", flush=True)
print("=" * 70, flush=True)

shifrut_results = {}

for method in ['pearson', 'spearman', 'mutual_info', 'partial_corr', 'ensemble']:
    scores_dict = shifrut_scores[method]
    y_all, s_all = [], []
    for (pert, target), score in scores_dict.items():
        if (pert, target) in ground_truth_shifrut:
            y_all.append(int(ground_truth_shifrut[(pert, target)]))
            s_all.append(score)

    if len(y_all) < 50:
        print(f"  {method}: too few edges ({len(y_all)}), skipping", flush=True)
        continue

    y_all = np.array(y_all)
    s_all = np.array(s_all)
    s_norm = normalize_scores(s_all)

    # Raw metrics
    raw_ece = compute_ece(y_all, s_norm)
    raw_mce = compute_mce(y_all, s_norm)
    raw_brier = brier_score_loss(y_all, s_norm)
    try:
        raw_auc = roc_auc_score(y_all, s_all)
    except:
        raw_auc = float('nan')

    # Cross-dataset: K562-trained calibration -> Shifrut
    cal = k562_calibrators[method]
    k_mn, k_mx = cal['score_min'], cal['score_max']
    if k_mx - k_mn > 1e-12:
        s_k562_norm = np.clip((s_all - k_mn) / (k_mx - k_mn), 0, 1)
    else:
        s_k562_norm = np.full_like(s_all, 0.5)

    cross_platt_probs = cal['platt'].predict_proba(s_k562_norm.reshape(-1, 1))[:, 1]
    cross_platt_ece = compute_ece(y_all, cross_platt_probs)
    cross_platt_brier = brier_score_loss(y_all, cross_platt_probs)

    cross_iso_probs = cal['isotonic'].predict(s_k562_norm)
    cross_iso_ece = compute_ece(y_all, cross_iso_probs)
    cross_iso_brier = brier_score_loss(y_all, cross_iso_probs)

    # Within-Shifrut: 70/30 split
    n_split = int(0.7 * len(y_all))
    idx = np.random.permutation(len(y_all))
    train_idx, test_idx = idx[:n_split], idx[n_split:]

    s_train_n = normalize_scores(s_all[train_idx])
    mn_all, mx_all = s_all.min(), s_all.max()
    s_test_n = np.clip((s_all[test_idx] - mn_all) / (mx_all - mn_all + 1e-12), 0, 1)

    platt_local = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    platt_local.fit(s_train_n.reshape(-1, 1), y_all[train_idx])
    local_platt_probs = platt_local.predict_proba(s_test_n.reshape(-1, 1))[:, 1]
    local_platt_ece = compute_ece(y_all[test_idx], local_platt_probs)
    local_platt_brier = brier_score_loss(y_all[test_idx], local_platt_probs)

    iso_local = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso_local.fit(s_train_n, y_all[train_idx])
    local_iso_probs = iso_local.predict(s_test_n)
    local_iso_ece = compute_ece(y_all[test_idx], local_iso_probs)
    local_iso_brier = brier_score_loss(y_all[test_idx], local_iso_probs)

    # Reliability diagram data for cross-dataset
    try:
        rel_true_raw, rel_pred_raw = calibration_curve(y_all, s_norm, n_bins=10, strategy='uniform')
    except:
        rel_true_raw, rel_pred_raw = np.array([]), np.array([])
    try:
        rel_true_cross, rel_pred_cross = calibration_curve(y_all, cross_iso_probs,
                                                            n_bins=10, strategy='uniform')
    except:
        rel_true_cross, rel_pred_cross = np.array([]), np.array([])

    shifrut_results[method] = {
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
        'reliability_raw': (rel_true_raw, rel_pred_raw),
        'reliability_cross_iso': (rel_true_cross, rel_pred_cross),
    }

    print(f"\n  {method} ({len(y_all)} edges, {100*y_all.mean():.1f}% pos):", flush=True)
    print(f"    Raw:    ECE={raw_ece:.4f}, AUROC={raw_auc:.4f}, Brier={raw_brier:.4f}", flush=True)
    print(f"    K562->: Platt ECE={cross_platt_ece:.4f}, Iso ECE={cross_iso_ece:.4f}", flush=True)
    print(f"    Local:  Platt ECE={local_platt_ece:.4f}, Iso ECE={local_iso_ece:.4f}", flush=True)
    print(f"    Gap (Platt): {cross_platt_ece - local_platt_ece:.4f}", flush=True)
    print(f"    Gap (Iso):   {cross_iso_ece - local_iso_ece:.4f}", flush=True)

# ============================================================
# Conformal prediction on Shifrut
# ============================================================
print("\n" + "=" * 70, flush=True)
print("CONFORMAL PREDICTION ON SHIFRUT", flush=True)
print("=" * 70, flush=True)

shifrut_conformal = {}

for method in ['mutual_info', 'ensemble', 'pearson']:
    scores_dict = shifrut_scores[method]
    y_all, s_all = [], []
    for (pert, target), score in scores_dict.items():
        if (pert, target) in ground_truth_shifrut:
            y_all.append(int(ground_truth_shifrut[(pert, target)]))
            s_all.append(score)

    if len(y_all) < 100:
        continue

    y_all = np.array(y_all)
    s_all = np.array(s_all)

    # 50/50 cal/test split
    n_half = len(y_all) // 2
    idx = np.random.permutation(len(y_all))
    cal_idx, test_idx = idx[:n_half], idx[n_half:]

    mn, mx = s_all.min(), s_all.max()
    s_cal_n = np.clip((s_all[cal_idx] - mn) / (mx - mn + 1e-12), 0, 1)
    s_test_n = np.clip((s_all[test_idx] - mn) / (mx - mn + 1e-12), 0, 1)

    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso.fit(s_cal_n, y_all[cal_idx])
    cal_probs = iso.predict(s_cal_n)
    test_probs = iso.predict(s_test_n)

    shifrut_conformal[method] = {}

    for alpha in [0.05, 0.10, 0.20]:
        nonconf = np.where(y_all[cal_idx] == 1, 1 - cal_probs, cal_probs)
        n_cal = len(cal_probs)
        q = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
        q = min(q, 1.0)
        threshold = np.quantile(nonconf, q)

        include_1 = (1 - test_probs) <= threshold
        include_0 = test_probs <= threshold
        set_sizes = include_1.astype(int) + include_0.astype(int)

        covered = np.array([
            (y_all[test_idx][i] == 1 and include_1[i]) or
            (y_all[test_idx][i] == 0 and include_0[i])
            for i in range(len(test_idx))
        ])

        coverage = covered.mean()
        avg_size = set_sizes.mean()
        singleton = (set_sizes == 1).mean()

        shifrut_conformal[method][alpha] = {
            'coverage': coverage,
            'avg_set_size': avg_size,
            'singleton_frac': singleton,
        }

        print(f"  {method} (alpha={alpha}): coverage={coverage:.3f} (target={1-alpha:.2f}), "
              f"avg_size={avg_size:.3f}, singletons={singleton:.3f}", flush=True)

# ============================================================
# Save
# ============================================================
print("\n--- Saving ---", flush=True)
output = {
    'shifrut_results': shifrut_results,
    'shifrut_conformal': shifrut_conformal,
    'shifrut_gene_list': gene_list,
    'shifrut_perts': perts_in,
    'shifrut_ground_truth_stats': {
        'n_total': len(ground_truth_shifrut),
        'n_positive': sum(ground_truth_shifrut.values()),
        'pos_rate': sum(ground_truth_shifrut.values()) / max(len(ground_truth_shifrut), 1),
    },
}

with open(os.path.join(OUT_DIR, 'phase2_shifrut_validation.pkl'), 'wb') as f:
    pickle.dump(output, f)

# Summary table
print("\n" + "=" * 70, flush=True)
print("SUMMARY: SHIFRUT VALIDATION", flush=True)
print("=" * 70, flush=True)

rows = []
for method in ['pearson', 'spearman', 'mutual_info', 'partial_corr', 'ensemble']:
    if method in shifrut_results:
        r = shifrut_results[method]
        rows.append({
            'Method': method,
            'N_edges': r['n_edges'],
            'AUROC': f"{r['raw_auc']:.3f}",
            'Raw_ECE': f"{r['raw_ece']:.3f}",
            'K562_Platt_ECE': f"{r['cross_platt_ece']:.3f}",
            'K562_Iso_ECE': f"{r['cross_iso_ece']:.3f}",
            'Local_Platt_ECE': f"{r['local_platt_ece']:.3f}",
            'Local_Iso_ECE': f"{r['local_iso_ece']:.3f}",
            'Gap_Iso': f"{r['calibration_gap_iso']:.3f}",
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False), flush=True)
df.to_csv(os.path.join(OUT_DIR, 'phase2_shifrut_summary.csv'), index=False)

print("\nDone with Phase 2 Shifrut validation.", flush=True)
