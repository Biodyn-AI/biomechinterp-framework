"""
Step 1: Compute edge scores and ground truth for calibration analysis.

Strategy:
- Use BOTH Dixit (TF knockdowns) and Adamson (gene knockdowns) for more ground truth
- Ground truth: rank-based (top N% most affected genes per perturbation) to handle
  the fact that CRISPRi effects are often subtle
- Edge scores: correlation, MI, partial correlation, and a simple L1-penalized logistic model
- Use 500 HVGs for speed with regression methods
"""

import scanpy as sc
import numpy as np
import pandas as pd
from scipy import stats, sparse
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Lasso
from statsmodels.stats.multitest import multipletests
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

OUT_DIR = '/Users/ihorkendiukhov/biodyn-work/subproject_31_uncertainty_calibration_edge_scores/implementation'
np.random.seed(42)


def load_and_process(path, condition_col='condition'):
    """Load, normalize, log-transform."""
    adata = sc.read_h5ad(path)
    sc.pp.filter_genes(adata, min_cells=int(0.02 * adata.shape[0]))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def get_de_ground_truth(adata, perturbations, gene_list, condition_col='condition',
                        padj_thresh=0.05, lfc_thresh=0.1):
    """Compute DE-based ground truth. Also return a continuous DE score for rank-based approach."""
    gt = {}
    de_scores = {}

    for pert in perturbations:
        mask_p = adata.obs[condition_col] == pert
        mask_c = adata.obs[condition_col] == 'control'
        n1 = mask_p.sum()
        n2 = mask_c.sum()
        if n1 < 10 or n2 < 10:
            continue

        X_p = adata[mask_p].X
        X_c = adata[mask_c].X
        if sparse.issparse(X_p):
            X_p = X_p.toarray()
        if sparse.issparse(X_c):
            X_c = X_c.toarray()
        X_p, X_c = X_p.astype(np.float64), X_c.astype(np.float64)

        mean_p, mean_c = X_p.mean(0), X_c.mean(0)
        var_p, var_c = X_p.var(0, ddof=1), X_c.var(0, ddof=1)
        se = np.sqrt(var_p/n1 + var_c/n2 + 1e-15)
        t_stat = (mean_p - mean_c) / se
        df_t = ((var_p/n1 + var_c/n2)**2 /
                ((var_p/n1)**2/(n1-1) + (var_c/n2)**2/(n2-1) + 1e-15))
        pvals = np.nan_to_num(2 * stats.t.sf(np.abs(t_stat), df_t), nan=1.0)
        lfc = mean_p - mean_c

        _, padj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

        # DE score = -log10(pval) * |lfc| (continuous measure of perturbation effect)
        de_score = -np.log10(pvals + 1e-300) * np.abs(lfc)

        for j, gn in enumerate(gene_list):
            if gn != pert:
                gt_key = (pert, gn)
                de_scores[gt_key] = float(de_score[j])
                # Binary: significant DE
                gt[gt_key] = bool(padj[j] < padj_thresh and abs(lfc[j]) > lfc_thresh)

    return gt, de_scores


print("=" * 70, flush=True)
print("STEP 1: EDGE SCORE COMPUTATION AND GROUND TRUTH DERIVATION", flush=True)
print("=" * 70, flush=True)

# ============================================================
# Load Dixit data
# ============================================================
print("\n--- Loading Dixit Perturb-seq ---", flush=True)
adata_dixit = load_and_process(
    '/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/perturb/dixit/perturb_processed_symbols.h5ad')
dixit_perts = [t for t in adata_dixit.obs['condition'].unique() if t != 'control']
print(f"  Shape: {adata_dixit.shape}, perturbations: {dixit_perts}", flush=True)

# ============================================================
# Load Adamson data
# ============================================================
print("\n--- Loading Adamson Perturb-seq ---", flush=True)
adata_adamson = sc.read_h5ad(
    '/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/perturb/adamson/perturb_processed_symbols.h5ad')
sc.pp.filter_genes(adata_adamson, min_cells=int(0.02 * adata_adamson.shape[0]))
sc.pp.normalize_total(adata_adamson, target_sum=1e4)
sc.pp.log1p(adata_adamson)

# Parse perturbation names (format: "GENE+ctrl")
adamson_conds = adata_adamson.obs['condition'].unique()
adamson_perts = [c.replace('+ctrl', '') for c in adamson_conds
                 if c != 'ctrl' and '+ctrl' in c]
# Map condition col for control
adata_adamson.obs['pert_gene'] = adata_adamson.obs['condition'].apply(
    lambda x: 'control' if x == 'ctrl' else x.replace('+ctrl', ''))
print(f"  Shape: {adata_adamson.shape}, perturbations: {len(adamson_perts)}", flush=True)

# ============================================================
# Find shared HVGs across datasets
# ============================================================
print("\n--- Selecting genes ---", flush=True)
# Select HVGs from Dixit control cells
ctrl_dixit = adata_dixit[adata_dixit.obs['condition'] == 'control'].copy()
sc.pp.highly_variable_genes(ctrl_dixit, n_top_genes=1000, flavor='seurat')
hvg_dixit = set(ctrl_dixit.var_names[ctrl_dixit.var['highly_variable']])

# Overlap with Adamson genes
shared_genes = set(adata_dixit.var_names) & set(adata_adamson.var_names)
hvg_shared = hvg_dixit & shared_genes

# Add perturbation genes
for p in dixit_perts:
    if p in shared_genes:
        hvg_shared.add(p)
for p in adamson_perts:
    if p in shared_genes:
        hvg_shared.add(p)

gene_list = sorted(hvg_shared)
print(f"  Shared HVGs: {len(gene_list)}", flush=True)

# Subset datasets
adata_dixit_sub = adata_dixit[:, [g for g in gene_list if g in adata_dixit.var_names]].copy()
adata_adamson_sub = adata_adamson[:, [g for g in gene_list if g in adata_adamson.var_names]].copy()
gene_list_dixit = list(adata_dixit_sub.var_names)
gene_list_adamson = list(adata_adamson_sub.var_names)

# Use Dixit gene list as primary (shared subset)
gene_list = gene_list_dixit

# Filter perturbations to those in gene list
dixit_perts_in = [p for p in dixit_perts if p in gene_list]
adamson_perts_in = [p for p in adamson_perts if p in gene_list]
all_perts = list(set(dixit_perts_in + adamson_perts_in))
print(f"  Dixit TFs in gene list: {len(dixit_perts_in)} -> {dixit_perts_in}", flush=True)
print(f"  Adamson genes in gene list: {len(adamson_perts_in)}", flush=True)

# ============================================================
# Ground truth from BOTH datasets
# ============================================================
print("\n--- Computing ground truth ---", flush=True)

gt_dixit, de_scores_dixit = get_de_ground_truth(
    adata_dixit_sub, dixit_perts_in, gene_list, 'condition')
gt_adamson, de_scores_adamson = get_de_ground_truth(
    adata_adamson_sub, adamson_perts_in, gene_list_adamson, 'pert_gene')

# Merge ground truth
all_gt = {}
all_de_scores = {}
for k, v in gt_dixit.items():
    all_gt[k] = v
    all_de_scores[k] = de_scores_dixit.get(k, 0)
for k, v in gt_adamson.items():
    if k not in all_gt:
        all_gt[k] = v
        all_de_scores[k] = de_scores_adamson.get(k, 0)

total_true_strict = sum(v for v in all_gt.values())
print(f"  Strict DE ground truth: {total_true_strict} true / {len(all_gt)} total "
      f"({100*total_true_strict/len(all_gt):.2f}%)", flush=True)

# If too few strict positives, use rank-based approach:
# Top 5% of DE scores per perturbation = positive
print("\n  Using hybrid approach: strict DE + rank-based (top 5% per perturbation)...", flush=True)
gt_rank = {}
perts_with_gt = set()
for k in all_gt:
    perts_with_gt.add(k[0])

for pert in perts_with_gt:
    pert_edges = {k: v for k, v in all_de_scores.items() if k[0] == pert}
    if len(pert_edges) == 0:
        continue
    scores_arr = np.array(list(pert_edges.values()))
    thresh = np.percentile(scores_arr, 95)
    for k, v in pert_edges.items():
        # True if strict DE OR in top 5% by DE score
        gt_rank[k] = all_gt.get(k, False) or (v >= thresh)

total_true_rank = sum(v for v in gt_rank.values())
print(f"  Rank-augmented ground truth: {total_true_rank} true / {len(gt_rank)} total "
      f"({100*total_true_rank/len(gt_rank):.2f}%)", flush=True)

# Use rank-augmented ground truth
ground_truth = gt_rank

# ============================================================
# Compute edge scores from Dixit control cells
# ============================================================
print("\n--- Computing edge scores from Dixit control cells ---", flush=True)

ctrl_sub = adata_dixit_sub[adata_dixit_sub.obs['condition'] == 'control']
if sparse.issparse(ctrl_sub.X):
    X_ctrl = np.array(ctrl_sub.X.toarray(), dtype=np.float32)
else:
    X_ctrl = np.array(ctrl_sub.X, dtype=np.float32)

print(f"  Control matrix: {X_ctrl.shape}", flush=True)

edge_scores = {}

# All perturbation genes that exist in gene_list
score_perts = list(perts_with_gt & set(gene_list))
print(f"  Scoring edges for {len(score_perts)} perturbation genes", flush=True)

# --- Pearson ---
print("  Pearson...", flush=True)
X_c = X_ctrl - X_ctrl.mean(0, keepdims=True)
norms = np.sqrt((X_c**2).sum(0, keepdims=True)) + 1e-12
X_n = X_c / norms

pearson_scores = {}
for pert in score_perts:
    if pert not in gene_list:
        continue
    pi = gene_list.index(pert)
    corrs = X_n[:, pi] @ X_n
    for j, gn in enumerate(gene_list):
        if gn != pert and (pert, gn) in ground_truth:
            pearson_scores[(pert, gn)] = float(np.abs(corrs[j]))
edge_scores['pearson'] = pearson_scores
print(f"    {len(pearson_scores)} edges", flush=True)

# --- Spearman ---
print("  Spearman...", flush=True)
X_ranked = np.zeros_like(X_ctrl)
for j in range(X_ctrl.shape[1]):
    X_ranked[:, j] = stats.rankdata(X_ctrl[:, j])
X_rc = X_ranked - X_ranked.mean(0, keepdims=True)
norms_r = np.sqrt((X_rc**2).sum(0, keepdims=True)) + 1e-12
X_rn = X_rc / norms_r

spearman_scores = {}
for pert in score_perts:
    if pert not in gene_list:
        continue
    pi = gene_list.index(pert)
    corrs = X_rn[:, pi] @ X_rn
    for j, gn in enumerate(gene_list):
        if gn != pert and (pert, gn) in ground_truth:
            spearman_scores[(pert, gn)] = float(np.abs(corrs[j]))
edge_scores['spearman'] = spearman_scores

# --- Mutual information ---
print("  Mutual information...", flush=True)
n_sub = min(1500, X_ctrl.shape[0])
sub_idx = np.random.choice(X_ctrl.shape[0], n_sub, replace=False)
X_sub = X_ctrl[sub_idx]

mi_scores = {}
for pert in score_perts:
    if pert not in gene_list:
        continue
    pi = gene_list.index(pert)
    mi_vals = mutual_info_regression(X_sub, X_sub[:, pi], n_neighbors=5, random_state=42)
    for j, gn in enumerate(gene_list):
        if gn != pert and (pert, gn) in ground_truth:
            mi_scores[(pert, gn)] = float(mi_vals[j])
    print(f"    {pert}: done", flush=True)
edge_scores['mutual_info'] = mi_scores

# --- Partial correlation ---
print("  Partial correlation...", flush=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ctrl)
pca = PCA(n_components=10, random_state=42)
pcs = pca.fit_transform(X_scaled)
beta = np.linalg.lstsq(pcs, X_scaled, rcond=None)[0]
X_resid = X_scaled - pcs @ beta
X_resid_c = X_resid - X_resid.mean(0, keepdims=True)
norms_res = np.sqrt((X_resid_c**2).sum(0, keepdims=True)) + 1e-12
X_resid_n = X_resid_c / norms_res

partial_corr_scores = {}
for pert in score_perts:
    if pert not in gene_list:
        continue
    pi = gene_list.index(pert)
    corrs = X_resid_n[:, pi] @ X_resid_n
    for j, gn in enumerate(gene_list):
        if gn != pert and (pert, gn) in ground_truth:
            partial_corr_scores[(pert, gn)] = float(np.abs(corrs[j]))
edge_scores['partial_corr'] = partial_corr_scores

# --- LASSO (fixed alpha for speed) ---
print("  LASSO (fixed alpha)...", flush=True)
X_sub_scaled = scaler.transform(X_ctrl[sub_idx])
lasso_scores = {}
for pert in score_perts:
    if pert not in gene_list:
        continue
    pi = gene_list.index(pert)
    y = X_sub_scaled[:, pi]
    other_idx = [i for i in range(len(gene_list)) if i != pi]
    X_feat = X_sub_scaled[:, other_idx]

    lasso = Lasso(alpha=0.05, max_iter=5000, random_state=42)
    lasso.fit(X_feat, y)
    coefs = np.abs(lasso.coef_)
    for k, j_orig in enumerate(other_idx):
        gn = gene_list[j_orig]
        if (pert, gn) in ground_truth:
            lasso_scores[(pert, gn)] = float(coefs[k])
    print(f"    {pert}: nonzero={np.sum(coefs > 0)}", flush=True)
edge_scores['lasso'] = lasso_scores

# --- Combined score (ensemble) ---
print("  Computing ensemble score...", flush=True)
# Min-max normalize each method, then average
all_edges = set()
for m in edge_scores:
    all_edges.update(edge_scores[m].keys())

ensemble_scores = {}
for edge in all_edges:
    vals = []
    for m in edge_scores:
        if edge in edge_scores[m]:
            vals.append(edge_scores[m][edge])
    if len(vals) > 0:
        ensemble_scores[edge] = float(np.mean(vals))
edge_scores['ensemble'] = ensemble_scores

# ============================================================
# Save
# ============================================================
print("\n--- Saving ---", flush=True)
results = {
    'edge_scores': edge_scores,
    'ground_truth': ground_truth,
    'de_scores': all_de_scores,
    'gene_list': gene_list,
    'perturbations': score_perts,
    'methods': list(edge_scores.keys()),
    'dixit_perts': dixit_perts_in,
    'adamson_perts': adamson_perts_in,
}

with open(os.path.join(OUT_DIR, 'edge_scores_and_ground_truth.pkl'), 'wb') as f:
    pickle.dump(results, f)

# Summary dataframe
rows = []
for method, scores in edge_scores.items():
    for (pert, target), score in scores.items():
        if (pert, target) in ground_truth:
            rows.append({
                'method': method,
                'perturbation': pert,
                'target': target,
                'score': score,
                'is_true_edge': ground_truth[(pert, target)]
            })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, 'edge_scores_summary.csv'), index=False)
print(f"  Saved {len(df)} scored edges across {len(edge_scores)} methods", flush=True)

print("\n--- Score distributions ---", flush=True)
for method in edge_scores.keys():
    sub = df[df['method'] == method]
    true_sub = sub[sub['is_true_edge']]
    false_sub = sub[~sub['is_true_edge']]
    n_true = len(true_sub)
    n_false = len(false_sub)
    try:
        auc = roc_auc_score(sub['is_true_edge'].astype(int), sub['score'])
    except:
        auc = float('nan')
    print(f"  {method}: AUROC={auc:.4f}, n_true={n_true}, n_false={n_false}", flush=True)

print("\nDone with Step 1.", flush=True)
