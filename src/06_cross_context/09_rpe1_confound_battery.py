#!/usr/bin/env python3
"""
Round 14: RPE1 Confound Battery
================================
Applies the full K562 confound-decomposition battery to RPE1 data.

Analyses:
  1. Trivial baselines (variance, mean expression, dropout rate)
  2. Incremental-value test (gene-level features ± edge features, GroupKFold)
  3. Propensity-matched decomposition (PSM with bootstrap CIs)
  4. Cross-fitted residualization (OLS + GBDT)
  5. TRRUST direct-target evaluation

Uses cached attention edges and DE labels from round13_non_k562_replication.
Loads RPE1 h5ad only for gene-level features and correlation matrix.
"""

import os
import sys
import json
import pickle
import time
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)  # Line-buffered

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
RPE1_DIR = os.path.join(PAPER_DIR, "experiments", "round13_non_k562_replication")
RPE1_H5AD = os.path.join(RPE1_DIR, "data", "ReplogleWeissman2022_rpe1.h5ad")
RPE1_CACHE = os.path.join(RPE1_DIR, "ReplogleRPE1")
TRRUST_PATH = os.path.join(PAPER_DIR, "cssi_real", "results", "trrust_human.tsv")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round14_rpe1_confound_battery")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Parameters (matching round13) ─────────────────────────────────────────────
PRIMARY_LAYER = 14    # L15, 0-indexed
N_CTRL = 2000
N_HVG = 2000
LFC_THRESHOLD = 0.5
MIN_DE_TARGETS = 5
SEED = 42
N_FOLDS = 5
N_BOOTSTRAP = 200
K_MATCHES = 5
CALIPER_MULTIPLIER = 0.25


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0: Load data and build flat dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_cached_data():
    """Load cached attention edges and DE labels from round13."""
    print("  Loading cached attention edges...")
    with open(os.path.join(RPE1_CACHE, "attention_edges.pkl"), 'rb') as f:
        all_attn_edges = pickle.load(f)
    attn_edges = all_attn_edges[PRIMARY_LAYER]
    print(f"    L{PRIMARY_LAYER+1} edges: {len(attn_edges):,} pairs")

    print("  Loading cached DE labels...")
    with open(os.path.join(RPE1_CACHE, "de_cache.pkl"), 'rb') as f:
        de_cache = pickle.load(f)

    # de_all_genes: {pert_gene: {target_gene: bool}}
    de_labels = de_cache['de_all_genes']
    de_targets = de_cache['de_targets']  # {pert_gene: set(de_targets)}
    print(f"    {len(de_labels)} evaluable perturbations")

    # Extract gene universe from de_labels
    gene_set = set()
    for pg, targets in de_labels.items():
        gene_set.add(pg)
        gene_set.update(targets.keys())
    gene_names = sorted(gene_set)
    print(f"    Gene universe: {len(gene_names)} genes")

    return attn_edges, de_labels, de_targets, gene_names


def compute_gene_features_and_correlation(gene_names):
    """Load RPE1 h5ad, compute gene features and correlation from controls.

    Saves checkpoint to avoid recomputation.
    """
    checkpoint = os.path.join(OUT_DIR, "gene_features_corr_checkpoint.pkl")
    if os.path.exists(checkpoint):
        print("  Loading gene features from checkpoint...")
        with open(checkpoint, 'rb') as f:
            data = pickle.load(f)
        print(f"    Loaded: {len(data['gene_features'])} genes, "
              f"corr matrix {data['corr_matrix'].shape}")
        return data['gene_features'], data['corr_matrix'], data['gene_to_idx']

    import scanpy as sc

    print("  Loading RPE1 h5ad (this takes ~1 min)...")
    t0 = time.time()
    adata = sc.read_h5ad(RPE1_H5AD)
    print(f"    Loaded: {adata.shape} in {time.time()-t0:.1f}s")

    # Identify controls
    if 'perturbation' in adata.obs.columns:
        ctrl_mask = adata.obs['perturbation'] == 'control'
    else:
        ctrl_mask = adata.obs['gene'] == 'non-targeting'
    n_ctrl_total = ctrl_mask.sum()
    print(f"    Control cells: {n_ctrl_total}")

    # Normalize
    print("    Normalizing (library-size + log1p)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Subsample controls (matching round13 seed)
    ctrl_indices = np.where(ctrl_mask.values)[0]
    np.random.seed(SEED)
    ctrl_idx = np.random.choice(ctrl_indices, min(N_CTRL, len(ctrl_indices)),
                                replace=False)
    print(f"    Sampled {len(ctrl_idx)} control cells")

    # Map gene_names to h5ad var_names
    # h5ad gene names are in var['gene_name'] (from h5py inspection)
    h5ad_gene_names = adata.var['gene_name'].values if 'gene_name' in adata.var.columns \
        else adata.var_names.values
    if hasattr(h5ad_gene_names[0], 'decode'):
        h5ad_gene_names = np.array([g.decode() for g in h5ad_gene_names])
    else:
        h5ad_gene_names = np.array([str(g) for g in h5ad_gene_names])

    # Build mapping: gene_name -> h5ad column index
    h5ad_name_to_idx = {}
    for i, gn in enumerate(h5ad_gene_names):
        if gn not in h5ad_name_to_idx:  # first occurrence wins
            h5ad_name_to_idx[gn] = i

    # Find which of our gene_names exist in h5ad
    gene_to_idx = {}  # gene_name -> local index (in our ordered list)
    h5ad_col_indices = []  # h5ad column indices for our genes
    valid_genes = []
    for i, gn in enumerate(gene_names):
        if gn in h5ad_name_to_idx:
            gene_to_idx[gn] = len(valid_genes)
            h5ad_col_indices.append(h5ad_name_to_idx[gn])
            valid_genes.append(gn)

    n_found = len(valid_genes)
    n_missing = len(gene_names) - n_found
    print(f"    Matched {n_found}/{len(gene_names)} genes "
          f"({n_missing} missing from h5ad)")

    # Extract control expression matrix for matched genes
    print("    Extracting control expression matrix...")
    X_ctrl = adata.X[ctrl_idx][:, h5ad_col_indices]
    if hasattr(X_ctrl, 'toarray'):
        X_ctrl = X_ctrl.toarray()
    X_ctrl = np.asarray(X_ctrl, dtype=np.float64)
    print(f"    Control matrix: {X_ctrl.shape}")

    # Compute gene-level features
    print("    Computing gene-level features...")
    mean_expr = X_ctrl.mean(axis=0)
    variance = X_ctrl.var(axis=0)
    dropout_rate = (X_ctrl == 0).mean(axis=0)

    gene_features = {}
    for i, gn in enumerate(valid_genes):
        gene_features[gn] = {
            'mean_expr': float(mean_expr[i]),
            'variance': float(variance[i]),
            'dropout_rate': float(dropout_rate[i]),
        }

    # Compute n_perturbations_affected: how many perturbations cause DE at this gene
    # (This requires de_labels which we don't have here, so we'll add it later)

    # Compute Spearman correlation matrix
    print("    Computing Spearman correlation matrix...")
    t_corr = time.time()
    ranks = np.apply_along_axis(rankdata, 0, X_ctrl)
    corr_matrix = np.corrcoef(ranks.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    print(f"    Correlation matrix: {corr_matrix.shape} in {time.time()-t_corr:.1f}s")

    # Save checkpoint
    data = {
        'gene_features': gene_features,
        'corr_matrix': corr_matrix,
        'gene_to_idx': gene_to_idx,
        'valid_genes': valid_genes,
    }
    with open(checkpoint, 'wb') as f:
        pickle.dump(data, f)
    print(f"    [checkpoint saved: gene_features_corr_checkpoint.pkl]")

    return gene_features, corr_matrix, gene_to_idx


def build_dataset(attn_edges, corr_matrix, de_labels, de_targets,
                  gene_features, gene_to_idx):
    """Build flat dataset: one row per (perturbation, target) pair."""
    checkpoint = os.path.join(OUT_DIR, "flat_dataset.pkl")
    if os.path.exists(checkpoint):
        print("  Loading flat dataset from checkpoint...")
        with open(checkpoint, 'rb') as f:
            df = pickle.load(f)
        print(f"    {len(df)} rows, {df['perturbation'].nunique()} perturbations")
        return df

    # Compute n_perturbations_affected per gene
    n_affected = {}
    for pg, targets in de_targets.items():
        for tgt in targets:
            n_affected[tgt] = n_affected.get(tgt, 0) + 1

    rows = []
    n_skipped_no_idx = 0

    for pg in sorted(de_labels.keys()):
        pg_idx = gene_to_idx.get(pg)
        if pg_idx is None:
            n_skipped_no_idx += 1
            continue

        targets_dict = de_labels[pg]  # {target_gene: bool}
        n_de = sum(1 for v in targets_dict.values() if v)
        if n_de < MIN_DE_TARGETS:
            continue

        for tgt, is_de in targets_dict.items():
            tgt_idx = gene_to_idx.get(tgt)
            if tgt_idx is None:
                continue

            # Attention edge score: max(fwd, rev)
            fwd = attn_edges.get((pg, tgt), 0.0)
            rev = attn_edges.get((tgt, pg), 0.0)
            attn_score = max(fwd, rev)

            # Correlation edge score: |Spearman rho|
            corr_score = abs(corr_matrix[pg_idx, tgt_idx])

            # Gene-level features of target
            tgt_feats = gene_features.get(tgt, {})

            rows.append({
                'perturbation': pg,
                'target': tgt,
                'is_de': int(bool(is_de)),
                'attn_edge': attn_score,
                'corr_edge': corr_score,
                'tgt_mean_expr': tgt_feats.get('mean_expr', 0),
                'tgt_variance': tgt_feats.get('variance', 0),
                'tgt_dropout': tgt_feats.get('dropout_rate', 0),
                'tgt_n_affected': n_affected.get(tgt, 0),
            })

    df = pd.DataFrame(rows)
    print(f"    {len(df):,} rows, {df['perturbation'].nunique()} perturbations, "
          f"{df['is_de'].sum():,} positive ({100*df['is_de'].mean():.1f}%)")
    if n_skipped_no_idx > 0:
        print(f"    Skipped {n_skipped_no_idx} perturbations (gene not in features)")

    with open(checkpoint, 'wb') as f:
        pickle.dump(df, f)
    print(f"    [checkpoint saved: flat_dataset.pkl]")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Trivial Baselines
# ══════════════════════════════════════════════════════════════════════════════

def run_trivial_baselines(df, gene_features, gene_to_idx):
    """Compare attention/correlation AUROCs against trivial gene-level baselines."""
    checkpoint = os.path.join(OUT_DIR, "part1_trivial_baselines.json")
    if os.path.exists(checkpoint):
        print("  Loading from checkpoint...")
        with open(checkpoint) as f:
            return json.load(f)

    # Precompute gene-level feature arrays indexed by gene_to_idx
    n_genes = len(gene_to_idx)
    mean_expr_arr = np.zeros(n_genes)
    variance_arr = np.zeros(n_genes)
    anti_dropout_arr = np.zeros(n_genes)
    for gn, idx in gene_to_idx.items():
        feats = gene_features.get(gn, {})
        mean_expr_arr[idx] = feats.get('mean_expr', 0)
        variance_arr[idx] = feats.get('variance', 0)
        anti_dropout_arr[idx] = 1.0 - feats.get('dropout_rate', 1.0)

    perturbations = df['perturbation'].unique()
    results_per_pert = []

    for pg in perturbations:
        sub = df[df['perturbation'] == pg]
        y = sub['is_de'].values
        if y.sum() == 0 or y.sum() == len(y):
            continue

        # Target indices
        tgt_indices = np.array([gene_to_idx[t] for t in sub['target'].values])

        row = {
            'perturbation': pg,
            'n_de_targets': int(y.sum()),
            'auroc_mean_expr': float(roc_auc_score(y, mean_expr_arr[tgt_indices])),
            'auroc_anti_dropout': float(roc_auc_score(y, anti_dropout_arr[tgt_indices])),
            'auroc_variance': float(roc_auc_score(y, variance_arr[tgt_indices])),
            'auroc_attention': float(roc_auc_score(y, sub['attn_edge'].values)),
            'auroc_correlation': float(roc_auc_score(y, sub['corr_edge'].values)),
        }
        results_per_pert.append(row)

    results_df = pd.DataFrame(results_per_pert)
    n_eval = len(results_df)

    # Aggregate
    predictor_cols = ['auroc_mean_expr', 'auroc_anti_dropout', 'auroc_variance',
                      'auroc_attention', 'auroc_correlation']
    predictor_names = ['Mean Expression', '1 - Dropout Rate', 'Variance',
                       'Attention (L15)', '|Spearman Corr|']

    summary_rows = []
    for col, name in zip(predictor_cols, predictor_names):
        vals = results_df[col].values
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        se_v = std_v / np.sqrt(len(vals))
        t_stat, t_p = stats.ttest_1samp(vals, 0.5)
        summary_rows.append({
            'predictor': name,
            'n': n_eval,
            'mean_auroc': round(mean_v, 4),
            'std_auroc': round(std_v, 4),
            'se_auroc': round(se_v, 4),
            'ci95_lo': round(mean_v - 1.96 * se_v, 4),
            'ci95_hi': round(mean_v + 1.96 * se_v, 4),
            'ttest_vs_0.5_p': float(t_p),
        })

    # Paired comparisons: attention vs each baseline
    paired_results = []
    for col, name in zip(predictor_cols[:3], predictor_names[:3]):
        diff_attn = results_df['auroc_attention'].values - results_df[col].values
        diff_corr = results_df['auroc_correlation'].values - results_df[col].values
        t_attn, p_attn = stats.ttest_rel(results_df['auroc_attention'].values,
                                          results_df[col].values)
        t_corr, p_corr = stats.ttest_rel(results_df['auroc_correlation'].values,
                                          results_df[col].values)
        paired_results.append({
            'baseline': name,
            'attn_minus_baseline': round(float(np.mean(diff_attn)), 4),
            'attn_p': float(p_attn),
            'corr_minus_baseline': round(float(np.mean(diff_corr)), 4),
            'corr_p': float(p_corr),
        })

    # Attention vs correlation paired comparison
    diff_ac = results_df['auroc_attention'].values - results_df['auroc_correlation'].values
    t_ac, p_ac = stats.ttest_rel(results_df['auroc_attention'].values,
                                  results_df['auroc_correlation'].values)

    result = {
        'n_perturbations_evaluated': n_eval,
        'summary': summary_rows,
        'paired_vs_baselines': paired_results,
        'attn_vs_corr': {
            'mean_diff': round(float(np.mean(diff_ac)), 4),
            'paired_t_p': float(p_ac),
            'n_attn_better': int((diff_ac > 0).sum()),
            'n_corr_better': int((diff_ac < 0).sum()),
        },
    }

    # Save per-perturbation data
    results_df.to_csv(os.path.join(OUT_DIR, "part1_per_perturbation.csv"), index=False)

    with open(checkpoint, 'w') as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"    [checkpoint saved: part1_trivial_baselines.json]")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Incremental Value Test
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_cv_lr(df, feature_cols, train_idx, test_idx):
    """Run logistic regression on one CV fold."""
    X = df[feature_cols].values
    y = df['is_de'].values
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        return None
    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        return None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_test_s)[:, 1]

    return {
        'auroc': float(roc_auc_score(y_test, probs)),
        'auprc': float(average_precision_score(y_test, probs)),
    }


def run_incremental_value(df, n_folds=5, n_bootstrap=50, seed=42):
    """Test whether edge scores add value beyond gene-level features.

    Uses perturbation-level bootstrap (resample perturbation-level AUROCs
    rather than resampling + re-fitting CV, which is prohibitively slow
    on 3.8M-row datasets).
    """
    checkpoint = os.path.join(OUT_DIR, "part2_incremental_value.json")
    if os.path.exists(checkpoint):
        print("  Loading from checkpoint...")
        with open(checkpoint) as f:
            return json.load(f)

    gene_features_cols = ['tgt_mean_expr', 'tgt_variance', 'tgt_dropout', 'tgt_n_affected']
    model_specs = {
        'A_gene_only': gene_features_cols,
        'B_attn_only': ['attn_edge'],
        'C_corr_only': ['corr_edge'],
        'D_gene_plus_attn': gene_features_cols + ['attn_edge'],
        'E_gene_plus_corr': gene_features_cols + ['corr_edge'],
    }

    # ── Point estimate (GroupKFold by perturbation) ────────────────────────
    print("    Running CV evaluation...")
    t_cv = time.time()
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(np.arange(len(df)),
                             groups=df['perturbation'].values))

    cv_results = {}
    for name, feat_cols in model_specs.items():
        aurocs, auprcs = [], []
        for train_idx, test_idx in splits:
            res = evaluate_cv_lr(df, feat_cols, train_idx, test_idx)
            if res is not None:
                aurocs.append(res['auroc'])
                auprcs.append(res['auprc'])
        cv_results[name] = {
            'mean_auroc': round(float(np.mean(aurocs)), 4) if aurocs else None,
            'std_auroc': round(float(np.std(aurocs)), 4) if aurocs else None,
            'mean_auprc': round(float(np.mean(auprcs)), 4) if auprcs else None,
            'std_auprc': round(float(np.std(auprcs)), 4) if auprcs else None,
            'n_folds_valid': len(aurocs),
        }
        print(f"      {name}: AUROC={cv_results[name]['mean_auroc']}")
    print(f"    CV complete in {time.time()-t_cv:.1f}s")

    # Deltas
    base = cv_results['A_gene_only']['mean_auroc']
    if base is not None:
        for aug in ['D_gene_plus_attn', 'E_gene_plus_corr']:
            if cv_results[aug]['mean_auroc'] is not None:
                cv_results[f'delta_{aug}'] = round(
                    cv_results[aug]['mean_auroc'] - base, 4)

    # ── Perturbation-level bootstrap CIs ──────────────────────────────────
    # Instead of re-fitting CV on resampled data (slow on 3.8M rows),
    # compute per-perturbation AUROCs for each model, then bootstrap
    # the mean across perturbations.
    print(f"    Computing per-perturbation model AUROCs for bootstrap...")
    t_boot = time.time()

    unique_perts = df['perturbation'].unique()
    pert_to_idx = {pg: df.index[df['perturbation'] == pg].values for pg in unique_perts}

    # For each perturbation, fit model on all OTHER perturbations, predict on this one
    # This is leave-one-perturbation-out (LOPO), but that's too slow for 1167 perts.
    # Instead: compute per-perturbation raw AUROCs for edge-only predictors,
    # and use the CV result for gene-feature models as the point estimate.
    # Bootstrap the per-perturbation AUROCs.

    per_pert_aurocs = {name: {} for name in ['B_attn_only', 'C_corr_only']}
    for pg in unique_perts:
        sub = df[df['perturbation'] == pg]
        y = sub['is_de'].values
        if y.sum() == 0 or y.sum() == len(y):
            continue
        # Raw edge AUROCs
        per_pert_aurocs['B_attn_only'][pg] = float(roc_auc_score(y, sub['attn_edge'].values))
        per_pert_aurocs['C_corr_only'][pg] = float(roc_auc_score(y, sub['corr_edge'].values))

    # For gene-feature models, use permutation-based bootstrap on CV folds
    # Fit once on full data, then bootstrap perturbation-level predictions
    print(f"    Fitting full models for bootstrap prediction...")
    from sklearn.linear_model import LogisticRegression

    # Fit all 5 models on full data
    full_models = {}
    full_scalers = {}
    for name, feat_cols in model_specs.items():
        X = df[feat_cols].values
        y = df['is_de'].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
        lr.fit(X_s, y)
        full_models[name] = lr
        full_scalers[name] = scaler

    # Compute per-perturbation AUROCs using full models
    for name, feat_cols in model_specs.items():
        if name in per_pert_aurocs:
            continue
        per_pert_aurocs[name] = {}
        scaler = full_scalers[name]
        model = full_models[name]
        for pg in unique_perts:
            sub = df[df['perturbation'] == pg]
            y = sub['is_de'].values
            if y.sum() == 0 or y.sum() == len(y):
                continue
            X_s = scaler.transform(sub[feat_cols].values)
            probs = model.predict_proba(X_s)[:, 1]
            per_pert_aurocs[name][pg] = float(roc_auc_score(y, probs))

    # Bootstrap: resample perturbation-level AUROCs
    print(f"    Running perturbation-level bootstrap ({n_bootstrap} iterations)...")
    rng = np.random.RandomState(seed)

    # Get common perturbation set (those with AUROCs for all models)
    common_perts = set(unique_perts)
    for name in model_specs:
        common_perts &= set(per_pert_aurocs[name].keys())
    common_perts = sorted(common_perts)
    n_common = len(common_perts)
    print(f"      {n_common} perturbations with all models evaluated")

    boot_aurocs = {name: [] for name in model_specs}
    delta_attn_aurocs = []
    delta_corr_aurocs = []

    for b in range(n_bootstrap):
        boot_idx = rng.choice(n_common, size=n_common, replace=True)
        boot_perts = [common_perts[i] for i in boot_idx]

        means = {}
        for name in model_specs:
            vals = [per_pert_aurocs[name][pg] for pg in boot_perts]
            m = float(np.mean(vals))
            boot_aurocs[name].append(m)
            means[name] = m

        base_m = means['A_gene_only']
        delta_attn_aurocs.append(means['D_gene_plus_attn'] - base_m)
        delta_corr_aurocs.append(means['E_gene_plus_corr'] - base_m)

    print(f"    Bootstrap complete in {time.time()-t_boot:.1f}s")

    # Build CI summary
    boot_summary = {}
    for name in model_specs:
        vals = np.array(boot_aurocs[name])
        boot_summary[name] = {
            'mean': round(float(vals.mean()), 4),
            'ci_lo': round(float(np.percentile(vals, 2.5)), 4),
            'ci_hi': round(float(np.percentile(vals, 97.5)), 4),
            'n_valid': len(vals),
        }

    delta_attn = np.array(delta_attn_aurocs)
    delta_corr = np.array(delta_corr_aurocs)

    boot_summary['delta_attn'] = {
        'mean': round(float(delta_attn.mean()), 4),
        'ci_lo': round(float(np.percentile(delta_attn, 2.5)), 4),
        'ci_hi': round(float(np.percentile(delta_attn, 97.5)), 4),
    }
    boot_summary['delta_corr'] = {
        'mean': round(float(delta_corr.mean()), 4),
        'ci_lo': round(float(np.percentile(delta_corr, 2.5)), 4),
        'ci_hi': round(float(np.percentile(delta_corr, 97.5)), 4),
    }

    result = {
        'cv_results': cv_results,
        'bootstrap_cis': boot_summary,
        'bootstrap_method': 'perturbation_level_resample',
        'n_common_perts': n_common,
    }

    with open(checkpoint, 'w') as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"    [checkpoint saved: part2_incremental_value.json]")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Propensity-Matched Decomposition
# ══════════════════════════════════════════════════════════════════════════════

def compute_propensity_scores(df, confound_cols):
    """Fit logistic regression P(DE=1 | confounds) and return scores."""
    from sklearn.decomposition import PCA

    X = df[confound_cols].values
    y = df['is_de'].values

    # PCA(2) on confounds for propensity model
    pca = PCA(n_components=min(2, X.shape[1]))
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

    lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    lr.fit(X_pca, y)
    return lr.predict_proba(X_pca)[:, 1]


def propensity_match_perturbation(sub_df, propensity_scores, k=5, caliper_mult=0.25):
    """Match DE+ targets with DE- targets within a perturbation."""
    pos_mask = sub_df['is_de'].values == 1
    neg_mask = ~pos_mask

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return None

    ps_pos = propensity_scores[pos_mask].reshape(-1, 1)
    ps_neg = propensity_scores[neg_mask].reshape(-1, 1)

    caliper = caliper_mult * propensity_scores.std()
    if caliper < 1e-6:
        caliper = 0.05

    nn = NearestNeighbors(n_neighbors=min(k, len(ps_neg)), metric='euclidean')
    nn.fit(ps_neg)
    distances, indices = nn.kneighbors(ps_pos)

    matched_neg_indices = set()
    wider_caliper_count = 0
    for i in range(len(ps_pos)):
        for j in range(distances.shape[1]):
            if distances[i, j] <= caliper:
                matched_neg_indices.add(indices[i, j])
            elif distances[i, j] <= 2 * caliper and j == 0:
                matched_neg_indices.add(indices[i, j])
                wider_caliper_count += 1

    if len(matched_neg_indices) == 0:
        return None

    # Reconstruct matched dataset: all positives + matched negatives
    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]
    matched_neg_orig = neg_indices[list(matched_neg_indices)]
    all_indices = np.concatenate([pos_indices, matched_neg_orig])

    return sub_df.iloc[all_indices].copy(), wider_caliper_count


def run_propensity_matching(df, k=5, caliper_mult=0.25):
    """Run propensity-score matching across all perturbations."""
    checkpoint = os.path.join(OUT_DIR, "part3_propensity_matched.json")
    if os.path.exists(checkpoint):
        print("  Loading from checkpoint...")
        with open(checkpoint) as f:
            return json.load(f)

    confound_cols = ['tgt_mean_expr', 'tgt_variance', 'tgt_dropout', 'tgt_n_affected']

    # Compute propensity scores per perturbation
    print("    Computing propensity scores and matching...")
    matched_dfs = []
    n_matched = 0
    n_skipped = 0
    total_wider = 0

    perturbations = df['perturbation'].unique()
    for pg in perturbations:
        sub = df[df['perturbation'] == pg].copy()
        if sub['is_de'].sum() == 0 or sub['is_de'].sum() == len(sub):
            n_skipped += 1
            continue

        ps = compute_propensity_scores(sub, confound_cols)
        result = propensity_match_perturbation(sub, ps, k=k, caliper_mult=caliper_mult)
        if result is None:
            n_skipped += 1
            continue

        matched_sub, n_wider = result
        matched_dfs.append(matched_sub)
        n_matched += 1
        total_wider += n_wider

    matched_df = pd.concat(matched_dfs, ignore_index=True)
    print(f"    Matched: {n_matched} perturbations, {n_skipped} skipped")
    print(f"    Before: {len(df):,} pairs -> After: {len(matched_df):,} pairs")
    print(f"    Positive rate: {100*df['is_de'].mean():.1f}% -> "
          f"{100*matched_df['is_de'].mean():.1f}%")

    # Covariate balance
    balance = {}
    for c in confound_cols:
        pos_vals = matched_df[matched_df['is_de'] == 1][c].values
        neg_vals = matched_df[matched_df['is_de'] == 0][c].values
        pooled_std = np.sqrt((pos_vals.var() + neg_vals.var()) / 2)
        smd = (pos_vals.mean() - neg_vals.mean()) / pooled_std if pooled_std > 0 else 0
        balance[c] = {
            'smd_after': round(float(smd), 4),
            'pos_mean': round(float(pos_vals.mean()), 4),
            'neg_mean': round(float(neg_vals.mean()), 4),
        }
        # Also compute before-matching balance
        pos_all = df[df['is_de'] == 1][c].values
        neg_all = df[df['is_de'] == 0][c].values
        ps_all = np.sqrt((pos_all.var() + neg_all.var()) / 2)
        smd_before = (pos_all.mean() - neg_all.mean()) / ps_all if ps_all > 0 else 0
        balance[c]['smd_before'] = round(float(smd_before), 4)

    # Evaluate on matched set
    print("    Evaluating on matched set...")

    # Raw edge AUROCs per perturbation
    raw_attn_aurocs, raw_corr_aurocs = [], []
    for pg in matched_df['perturbation'].unique():
        sub = matched_df[matched_df['perturbation'] == pg]
        y = sub['is_de'].values
        if y.sum() == 0 or y.sum() == len(y):
            continue
        raw_attn_aurocs.append(roc_auc_score(y, sub['attn_edge'].values))
        raw_corr_aurocs.append(roc_auc_score(y, sub['corr_edge'].values))

    raw_results = {
        'attn_mean_auroc': round(float(np.mean(raw_attn_aurocs)), 4),
        'attn_std_auroc': round(float(np.std(raw_attn_aurocs)), 4),
        'corr_mean_auroc': round(float(np.mean(raw_corr_aurocs)), 4),
        'corr_std_auroc': round(float(np.std(raw_corr_aurocs)), 4),
        'n_perturbations': len(raw_attn_aurocs),
    }

    # Incremental value on matched set (GroupKFold)
    gene_features_cols = ['tgt_mean_expr', 'tgt_variance', 'tgt_dropout', 'tgt_n_affected']
    matched_model_specs = {
        'A_gene_only': gene_features_cols,
        'B_gene_plus_attn': gene_features_cols + ['attn_edge'],
        'C_gene_plus_corr': gene_features_cols + ['corr_edge'],
    }

    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(np.arange(len(matched_df)),
                             groups=matched_df['perturbation'].values))

    matched_cv = {}
    for name, feat_cols in matched_model_specs.items():
        aurocs = []
        for train_idx, test_idx in splits:
            res = evaluate_cv_lr(matched_df, feat_cols, train_idx, test_idx)
            if res is not None:
                aurocs.append(res['auroc'])
        matched_cv[name] = {
            'mean_auroc': round(float(np.mean(aurocs)), 4) if aurocs else None,
            'std_auroc': round(float(np.std(aurocs)), 4) if aurocs else None,
        }

    # Deltas on matched set
    deltas = {}
    if matched_cv['A_gene_only']['mean_auroc'] is not None:
        base_auroc = matched_cv['A_gene_only']['mean_auroc']
        for aug in ['B_gene_plus_attn', 'C_gene_plus_corr']:
            if matched_cv[aug]['mean_auroc'] is not None:
                deltas[aug] = round(matched_cv[aug]['mean_auroc'] - base_auroc, 4)

    result = {
        'n_perturbations_matched': n_matched,
        'n_perturbations_skipped': n_skipped,
        'n_pairs_before': len(df),
        'n_pairs_after': len(matched_df),
        'positive_rate_before': round(float(df['is_de'].mean()), 4),
        'positive_rate_after': round(float(matched_df['is_de'].mean()), 4),
        'covariate_balance': balance,
        'raw_edge_results': raw_results,
        'matched_cv_results': matched_cv,
        'matched_deltas': deltas,
    }

    with open(checkpoint, 'w') as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"    [checkpoint saved: part3_propensity_matched.json]")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: Cross-Fitted Residualization
# ══════════════════════════════════════════════════════════════════════════════

def cross_fitted_residual_auroc(X_cov, y_edge, labels, n_splits=5,
                                 model_type='ols', seed=42):
    """Cross-fitted residualization: train residualizer on K-1 folds,
    predict on held-out fold, compute AUROC on residuals."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_residuals = np.zeros(len(y_edge))

    for train_idx, test_idx in kf.split(X_cov):
        X_train, X_test = X_cov[train_idx], X_cov[test_idx]
        y_train, y_test = y_edge[train_idx], y_edge[test_idx]

        if model_type == 'ols':
            model = LinearRegression()
        elif model_type == 'gbdt':
            model = HistGradientBoostingRegressor(
                max_iter=100, max_depth=3, learning_rate=0.1,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_residuals[test_idx] = y_test - y_pred

    # R² from residualization (in-sample for reference)
    reg_full = LinearRegression() if model_type == 'ols' \
        else HistGradientBoostingRegressor(max_iter=100, max_depth=3,
                                           learning_rate=0.1, random_state=seed)
    reg_full.fit(X_cov, y_edge)
    r2 = r2_score(y_edge, reg_full.predict(X_cov))

    auroc_abs = roc_auc_score(labels, np.abs(all_residuals))
    auroc_signed = roc_auc_score(labels, all_residuals)

    return {
        'r2': round(float(r2), 4),
        'auroc_abs_residual': round(float(auroc_abs), 4),
        'auroc_signed_residual': round(float(auroc_signed), 4),
    }


def run_residualization(df, n_splits=5):
    """Cross-fitted residualization of edge scores on expression covariates."""
    checkpoint = os.path.join(OUT_DIR, "part4_residualization.json")
    if os.path.exists(checkpoint):
        print("  Loading from checkpoint...")
        with open(checkpoint) as f:
            return json.load(f)

    labels = df['is_de'].values

    cov_sets = {
        'full': ['tgt_mean_expr', 'tgt_variance', 'tgt_dropout', 'tgt_n_affected'],
        'expr_only': ['tgt_mean_expr', 'tgt_variance'],
        'minimal': ['tgt_mean_expr'],
    }

    results = {}

    # For both edge types: attention and correlation
    for edge_name, edge_col in [('attention', 'attn_edge'), ('correlation', 'corr_edge')]:
        y_edge = df[edge_col].values

        # Baseline AUROC (raw edge score)
        baseline_auroc = roc_auc_score(labels, y_edge)
        results[f'{edge_name}_baseline_auroc'] = round(float(baseline_auroc), 4)

        for cov_name, cov_cols in cov_sets.items():
            X_cov = df[cov_cols].values

            # OLS cross-fitted
            ols_result = cross_fitted_residual_auroc(
                X_cov, y_edge, labels, n_splits=n_splits, model_type='ols')
            results[f'{edge_name}_ols_{cov_name}'] = ols_result

        # GBDT cross-fitted (full covariates only)
        X_cov_full = df[cov_sets['full']].values
        gbdt_result = cross_fitted_residual_auroc(
            X_cov_full, y_edge, labels, n_splits=n_splits, model_type='gbdt')
        results[f'{edge_name}_gbdt_full'] = gbdt_result

    # Stability across seeds (OLS, full, attention)
    print("    Checking seed stability...")
    stability_aurocs = []
    X_full = df[cov_sets['full']].values
    y_attn = df['attn_edge'].values
    for s in range(10):
        res = cross_fitted_residual_auroc(
            X_full, y_attn, labels, n_splits=n_splits, model_type='ols', seed=s)
        stability_aurocs.append(res['auroc_abs_residual'])
    results['stability_10seeds'] = {
        'mean': round(float(np.mean(stability_aurocs)), 4),
        'std': round(float(np.std(stability_aurocs)), 4),
    }

    with open(checkpoint, 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"    [checkpoint saved: part4_residualization.json]")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: TRRUST Direct-Target Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def load_trrust():
    """Load TRRUST TF-target database."""
    trrust = {}
    if not os.path.exists(TRRUST_PATH):
        # Try alternate path
        alt_path = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv"
        if os.path.exists(alt_path):
            path = alt_path
        else:
            return {}
    else:
        path = TRRUST_PATH

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            tf = parts[0].upper()
            target = parts[1].upper()
            if tf not in trrust:
                trrust[tf] = set()
            trrust[tf].add(target)
    return trrust


def run_trrust_evaluation(df, gene_to_idx):
    """Evaluate attention/correlation on TRRUST direct-target prediction."""
    checkpoint = os.path.join(OUT_DIR, "part5_trrust_evaluation.json")
    if os.path.exists(checkpoint):
        print("  Loading from checkpoint...")
        with open(checkpoint) as f:
            return json.load(f)

    trrust = load_trrust()
    if not trrust:
        print("    WARNING: TRRUST not found, skipping")
        return {'error': 'TRRUST not found'}

    print(f"    TRRUST: {len(trrust)} TFs, "
          f"{sum(len(v) for v in trrust.values())} relationships")

    gene_upper_set = set(g.upper() for g in gene_to_idx.keys())
    perturbations = df['perturbation'].unique()

    # Find TFs that are perturbation genes AND in TRRUST
    tf_perts = [pg for pg in perturbations if pg.upper() in trrust]
    print(f"    TF perturbations in TRRUST: {len(tf_perts)}")

    results_per_tf = []
    for pg in tf_perts:
        trrust_targets = trrust[pg.upper()] & gene_upper_set
        if len(trrust_targets) == 0:
            continue

        sub = df[df['perturbation'] == pg].copy()
        sub['is_trrust'] = sub['target'].apply(
            lambda t: int(t.upper() in trrust_targets)
        )

        n_pos = sub['is_trrust'].sum()
        n_neg = len(sub) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue

        y = sub['is_trrust'].values
        attn_auroc = roc_auc_score(y, sub['attn_edge'].values)
        corr_auroc = roc_auc_score(y, sub['corr_edge'].values)

        results_per_tf.append({
            'tf': pg,
            'n_trrust_targets': int(n_pos),
            'n_non_targets': int(n_neg),
            'attn_auroc': round(float(attn_auroc), 4),
            'corr_auroc': round(float(corr_auroc), 4),
        })

    if not results_per_tf:
        result = {'n_evaluable_tfs': 0, 'message': 'No evaluable TFs'}
    else:
        rdf = pd.DataFrame(results_per_tf)
        result = {
            'n_evaluable_tfs': len(rdf),
            'n_direct_targets_total': int(rdf['n_trrust_targets'].sum()),
            'mean_attn_auroc': round(float(rdf['attn_auroc'].mean()), 4),
            'std_attn_auroc': round(float(rdf['attn_auroc'].std()), 4),
            'mean_corr_auroc': round(float(rdf['corr_auroc'].mean()), 4),
            'std_corr_auroc': round(float(rdf['corr_auroc'].std()), 4),
            'per_tf_results': results_per_tf,
        }

    with open(checkpoint, 'w') as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"    [checkpoint saved: part5_trrust_evaluation.json]")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_summary_figure(part1, part2, part3, part4, part5):
    """Generate 6-panel summary figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # ── Panel A: Trivial baselines comparison ─────────────────────────────
    ax = axes[0, 0]
    predictors = [s['predictor'] for s in part1['summary']]
    means = [s['mean_auroc'] for s in part1['summary']]
    ses = [s['se_auroc'] for s in part1['summary']]
    colors = ['#FF9800', '#4CAF50', '#9C27B0', '#DD8452', '#2196F3']
    bars = ax.bar(range(len(predictors)), means,
                  yerr=[1.96 * s for s in ses], capsize=3,
                  color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(predictors)))
    ax.set_xticklabels([p.replace(' ', '\n') for p in predictors],
                       fontsize=7, rotation=0)
    ax.set_ylabel('Mean AUROC')
    ax.set_title('(A) Predictor comparison\n(trivial baselines)', fontsize=10)
    for i, (m, s) in enumerate(zip(means, ses)):
        ax.text(i, m + 1.96 * s + 0.005, f'{m:.3f}', ha='center',
                fontsize=7, fontweight='bold')

    # ── Panel B: Incremental value ────────────────────────────────────────
    ax = axes[0, 1]
    cv = part2['cv_results']
    model_names = ['A_gene_only', 'B_attn_only', 'C_corr_only',
                   'D_gene_plus_attn', 'E_gene_plus_corr']
    short_labels = ['Gene\nonly', 'Attn\nonly', 'Corr\nonly',
                    'Gene+\nAttn', 'Gene+\nCorr']
    cv_means = [cv[n]['mean_auroc'] or 0.5 for n in model_names]
    cv_stds = [cv[n]['std_auroc'] or 0 for n in model_names]
    colors_cv = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#E91E63']
    ax.bar(range(len(model_names)), cv_means, yerr=cv_stds,
           color=colors_cv, alpha=0.8, capsize=3, edgecolor='black', linewidth=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel('CV AUROC')
    ax.set_title('(B) Incremental value\n(5-fold GroupKFold)', fontsize=10)
    for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
        ax.text(i, m + s + 0.005, f'{m:.3f}', ha='center', fontsize=7)
    # Annotate deltas
    boot = part2.get('bootstrap_cis', {})
    da = boot.get('delta_attn', {})
    dc = boot.get('delta_corr', {})
    if da.get('mean') is not None:
        ax.text(0.95, 0.95,
                f"$\\Delta$attn: {da['mean']:+.4f} [{da['ci_lo']:+.4f},{da['ci_hi']:+.4f}]\n"
                f"$\\Delta$corr: {dc['mean']:+.4f} [{dc['ci_lo']:+.4f},{dc['ci_hi']:+.4f}]",
                transform=ax.transAxes, fontsize=7, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    # ── Panel C: Propensity matching — covariate balance ──────────────────
    ax = axes[0, 2]
    balance = part3.get('covariate_balance', {})
    if balance:
        cov_names = list(balance.keys())
        smds_before = [abs(balance[c]['smd_before']) for c in cov_names]
        smds_after = [abs(balance[c]['smd_after']) for c in cov_names]
        x = np.arange(len(cov_names))
        width = 0.35
        ax.bar(x - width / 2, smds_before, width, label='Before', color='#E57373',
               alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.bar(x + width / 2, smds_after, width, label='After', color='#81C784',
               alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.axhline(0.1, color='gray', linestyle='--', alpha=0.7, label='SMD=0.1')
        ax.set_xticks(x)
        short_cov = [c.replace('tgt_', '').replace('_', '\n') for c in cov_names]
        ax.set_xticklabels(short_cov, fontsize=7)
        ax.set_ylabel('|SMD|')
        ax.set_title('(C) Covariate balance\n(propensity matching)', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')

    # ── Panel D: Propensity matching — matched AUROC ──────────────────────
    ax = axes[1, 0]
    mcv = part3.get('matched_cv_results', {})
    raw = part3.get('raw_edge_results', {})
    if mcv:
        bar_labels = ['Attn\n(raw)', 'Corr\n(raw)', 'Gene-only\n(LR)',
                      'Gene+Attn\n(LR)', 'Gene+Corr\n(LR)']
        bar_vals = [
            raw.get('attn_mean_auroc', 0.5),
            raw.get('corr_mean_auroc', 0.5),
            mcv.get('A_gene_only', {}).get('mean_auroc', 0.5),
            mcv.get('B_gene_plus_attn', {}).get('mean_auroc', 0.5),
            mcv.get('C_gene_plus_corr', {}).get('mean_auroc', 0.5),
        ]
        bar_colors = ['#DD8452', '#009E73', '#0072B2', '#D55E00', '#8172B3']
        ax.bar(range(len(bar_labels)), bar_vals, color=bar_colors, alpha=0.8,
               edgecolor='black', linewidth=0.5)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xticks(range(len(bar_labels)))
        ax.set_xticklabels(bar_labels, fontsize=8)
        ax.set_ylabel('AUROC')
        ax.set_title('(D) Edge evaluation\n(matched set)', fontsize=10)
        for i, v in enumerate(bar_vals):
            ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=7)

    # ── Panel E: Residualization ──────────────────────────────────────────
    ax = axes[1, 1]
    resid_labels = ['Baseline\n(raw)', 'OLS full\n|resid|', 'OLS min\n|resid|',
                    'GBDT full\n|resid|']
    attn_vals = [
        part4.get('attention_baseline_auroc', 0.5),
        part4.get('attention_ols_full', {}).get('auroc_abs_residual', 0.5),
        part4.get('attention_ols_minimal', {}).get('auroc_abs_residual', 0.5),
        part4.get('attention_gbdt_full', {}).get('auroc_abs_residual', 0.5),
    ]
    corr_vals = [
        part4.get('correlation_baseline_auroc', 0.5),
        part4.get('correlation_ols_full', {}).get('auroc_abs_residual', 0.5),
        part4.get('correlation_ols_minimal', {}).get('auroc_abs_residual', 0.5),
        part4.get('correlation_gbdt_full', {}).get('auroc_abs_residual', 0.5),
    ]
    x = np.arange(len(resid_labels))
    width = 0.35
    ax.bar(x - width / 2, attn_vals, width, label='Attention', color='#DD8452',
           alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, corr_vals, width, label='Correlation', color='#009E73',
           alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(resid_labels, fontsize=8)
    ax.set_ylabel('AUROC')
    ax.set_title('(E) Cross-fitted residualization', fontsize=10)
    ax.legend(fontsize=8)
    for i, (a, c) in enumerate(zip(attn_vals, corr_vals)):
        ax.text(i - width / 2, a + 0.005, f'{a:.3f}', ha='center', fontsize=6)
        ax.text(i + width / 2, c + 0.005, f'{c:.3f}', ha='center', fontsize=6)

    # ── Panel F: TRRUST direct targets ────────────────────────────────────
    ax = axes[1, 2]
    n_tfs = part5.get('n_evaluable_tfs', 0)
    if n_tfs > 0:
        metrics = ['AUROC']
        attn_v = [part5['mean_attn_auroc']]
        corr_v = [part5['mean_corr_auroc']]
        attn_e = [part5['std_attn_auroc']]
        corr_e = [part5['std_corr_auroc']]
        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width / 2, attn_v, width, yerr=attn_e,
               label='Attention', color='#DD8452', alpha=0.8, capsize=3)
        ax.bar(x + width / 2, corr_v, width, yerr=corr_e,
               label='Correlation', color='#009E73', alpha=0.8, capsize=3)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Score')
        ax.set_title(f'(F) TRRUST direct targets\n({n_tfs} TFs)', fontsize=10)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No evaluable\nTF perturbations',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('(F) TRRUST direct targets', fontsize=10)

    fig.suptitle('Round 14: RPE1 Confound Battery\n'
                 '(Full K562 confound decomposition applied to RPE1)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "fig_rpe1_confound_battery.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    return fig_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 70)
    print("ROUND 14: RPE1 CONFOUND BATTERY")
    print("=" * 70)
    print("Applies full K562 confound-decomposition battery to RPE1 data.")
    print(f"  Primary layer: L{PRIMARY_LAYER + 1} (0-indexed: {PRIMARY_LAYER})")
    print(f"  LFC threshold: {LFC_THRESHOLD}")
    print(f"  Bootstrap iterations: {N_BOOTSTRAP}")
    t0 = time.time()

    # ── Step 0: Load data ─────────────────────────────────────────────────
    print("\n[0/6] Loading data...")
    attn_edges, de_labels, de_targets, gene_names = load_cached_data()
    gene_features, corr_matrix, gene_to_idx = \
        compute_gene_features_and_correlation(gene_names)

    print("\n[0/6] Building flat dataset...")
    df = build_dataset(attn_edges, corr_matrix, de_labels, de_targets,
                       gene_features, gene_to_idx)

    # ── Part 1: Trivial Baselines ─────────────────────────────────────────
    print("\n[1/6] TRIVIAL BASELINES")
    print("  Testing: variance, mean expression, dropout rate vs attention/correlation")
    part1 = run_trivial_baselines(df, gene_features, gene_to_idx)

    print(f"\n  Results ({part1['n_perturbations_evaluated']} perturbations):")
    for s in part1['summary']:
        sig = "***" if s['ttest_vs_0.5_p'] < 0.001 else \
              ("**" if s['ttest_vs_0.5_p'] < 0.01 else
               ("*" if s['ttest_vs_0.5_p'] < 0.05 else "ns"))
        print(f"    {s['predictor']:25s}: {s['mean_auroc']:.4f} "
              f"+/- {s['std_auroc']:.4f}  p={s['ttest_vs_0.5_p']:.2e} {sig}")

    ac = part1['attn_vs_corr']
    print(f"\n  Attention vs Correlation: diff={ac['mean_diff']:+.4f}, "
          f"p={ac['paired_t_p']:.2e}")
    print(f"    Attention better: {ac['n_attn_better']}, "
          f"Correlation better: {ac['n_corr_better']}")

    # ── Part 2: Incremental Value ─────────────────────────────────────────
    print("\n[2/6] INCREMENTAL VALUE TEST")
    print("  Testing: gene features ± edge features (GroupKFold)")
    part2 = run_incremental_value(df, n_folds=N_FOLDS,
                                   n_bootstrap=N_BOOTSTRAP, seed=SEED)

    cv = part2['cv_results']
    print(f"\n  CV Results:")
    for name in ['A_gene_only', 'B_attn_only', 'C_corr_only',
                 'D_gene_plus_attn', 'E_gene_plus_corr']:
        r = cv[name]
        print(f"    {name:25s}: AUROC={r['mean_auroc']:.4f} "
              f"+/- {r['std_auroc']:.4f}")

    boot = part2['bootstrap_cis']
    da = boot.get('delta_attn', {})
    dc = boot.get('delta_corr', {})
    if da.get('mean') is not None:
        print(f"\n  Bootstrap Delta CIs:")
        print(f"    ΔAUROC (gene+attn - gene): {da['mean']:+.4f} "
              f"[{da['ci_lo']:+.4f}, {da['ci_hi']:+.4f}]")
        print(f"    ΔAUROC (gene+corr - gene): {dc['mean']:+.4f} "
              f"[{dc['ci_lo']:+.4f}, {dc['ci_hi']:+.4f}]")

    # ── Part 3: Propensity-Matched Decomposition ─────────────────────────
    print("\n[3/6] PROPENSITY-MATCHED DECOMPOSITION")
    print("  Matching DE+ targets with DE- targets by gene-level features")
    part3 = run_propensity_matching(df, k=K_MATCHES,
                                     caliper_mult=CALIPER_MULTIPLIER)

    balance = part3.get('covariate_balance', {})
    print(f"\n  Covariate balance (|SMD|):")
    for c, b in balance.items():
        status = "OK" if abs(b['smd_after']) < 0.1 else "WARN"
        print(f"    {c:25s}: before={abs(b['smd_before']):.4f}  "
              f"after={abs(b['smd_after']):.4f}  [{status}]")

    raw = part3.get('raw_edge_results', {})
    mcv = part3.get('matched_cv_results', {})
    print(f"\n  Raw edges on matched set:")
    print(f"    Attention AUROC:   {raw.get('attn_mean_auroc', 'N/A')}")
    print(f"    Correlation AUROC: {raw.get('corr_mean_auroc', 'N/A')}")
    print(f"  LR on matched set:")
    for name in ['A_gene_only', 'B_gene_plus_attn', 'C_gene_plus_corr']:
        if name in mcv:
            print(f"    {name:25s}: AUROC={mcv[name].get('mean_auroc', 'N/A')}")
    deltas = part3.get('matched_deltas', {})
    for k, v in deltas.items():
        print(f"    Delta({k}): {v:+.4f}")

    # ── Part 4: Cross-Fitted Residualization ──────────────────────────────
    print("\n[4/6] CROSS-FITTED RESIDUALIZATION")
    print("  Testing: OLS and GBDT residualization of edge scores")
    part4 = run_residualization(df, n_splits=N_FOLDS)

    print(f"\n  Attention:")
    print(f"    Baseline AUROC: {part4.get('attention_baseline_auroc', 'N/A')}")
    for key in ['attention_ols_full', 'attention_ols_minimal', 'attention_gbdt_full']:
        if key in part4:
            r = part4[key]
            print(f"    {key:30s}: R²={r['r2']:.4f}, "
                  f"AUROC(|resid|)={r['auroc_abs_residual']:.4f}")
    print(f"  Correlation:")
    print(f"    Baseline AUROC: {part4.get('correlation_baseline_auroc', 'N/A')}")
    for key in ['correlation_ols_full', 'correlation_ols_minimal', 'correlation_gbdt_full']:
        if key in part4:
            r = part4[key]
            print(f"    {key:30s}: R²={r['r2']:.4f}, "
                  f"AUROC(|resid|)={r['auroc_abs_residual']:.4f}")
    stab = part4.get('stability_10seeds', {})
    if stab:
        print(f"  Seed stability: {stab['mean']:.4f} +/- {stab['std']:.4f}")

    # ── Part 5: TRRUST Direct-Target Evaluation ──────────────────────────
    print("\n[5/6] TRRUST DIRECT-TARGET EVALUATION")
    part5 = run_trrust_evaluation(df, gene_to_idx)

    if part5.get('n_evaluable_tfs', 0) > 0:
        print(f"    Evaluable TFs: {part5['n_evaluable_tfs']}")
        print(f"    Total direct targets: {part5['n_direct_targets_total']}")
        print(f"    Attention AUROC:   {part5['mean_attn_auroc']:.4f} "
              f"+/- {part5['std_attn_auroc']:.4f}")
        print(f"    Correlation AUROC: {part5['mean_corr_auroc']:.4f} "
              f"+/- {part5['std_corr_auroc']:.4f}")
    else:
        print(f"    No evaluable TF perturbations found")

    # ── Generate figure ──────────────────────────────────────────────────
    print("\n[6/6] Generating figure...")
    fig_path = generate_summary_figure(part1, part2, part3, part4, part5)
    print(f"  Saved: {fig_path}")

    # ── Final summary ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ROUND 14: RPE1 CONFOUND BATTERY COMPLETE — {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Save combined summary
    combined = {
        'experiment': 'round14_rpe1_confound_battery',
        'dataset': 'ReplogleRPE1',
        'primary_layer': PRIMARY_LAYER,
        'lfc_threshold': LFC_THRESHOLD,
        'n_perturbations': int(df['perturbation'].nunique()),
        'n_observations': len(df),
        'positive_rate': round(float(df['is_de'].mean()), 4),
        'elapsed_seconds': round(elapsed, 1),
        'part1_trivial_baselines': part1,
        'part2_incremental_value': part2,
        'part3_propensity_matched': part3,
        'part4_residualization': part4,
        'part5_trrust_evaluation': part5,
    }

    with open(os.path.join(OUT_DIR, "rpe1_confound_battery_full.json"), 'w') as f:
        json.dump(combined, f, indent=2, default=_json_default)

    # ── Key conclusions ──────────────────────────────────────────────────
    print(f"\n  KEY CONCLUSIONS:")

    # 1. Trivial baselines vs attention/correlation
    attn_auroc = None
    corr_auroc = None
    best_trivial = 0
    for s in part1['summary']:
        if s['predictor'] == 'Attention (L15)':
            attn_auroc = s['mean_auroc']
        elif s['predictor'] == '|Spearman Corr|':
            corr_auroc = s['mean_auroc']
        elif s['predictor'] in ('Mean Expression', '1 - Dropout Rate', 'Variance'):
            best_trivial = max(best_trivial, s['mean_auroc'])

    if attn_auroc is not None:
        if attn_auroc > best_trivial:
            print(f"    1. Attention (L15) AUROC ({attn_auroc:.3f}) EXCEEDS best "
                  f"trivial baseline ({best_trivial:.3f})")
        else:
            print(f"    1. Attention (L15) AUROC ({attn_auroc:.3f}) does NOT exceed "
                  f"best trivial baseline ({best_trivial:.3f})")

    # 2. Incremental value
    if da.get('ci_lo') is not None:
        if da['ci_lo'] > 0:
            print(f"    2. Attention provides SIGNIFICANT incremental value "
                  f"({da['mean']:+.4f}, CI excludes zero)")
        elif da['mean'] > 0:
            print(f"    2. Attention shows positive but NON-SIGNIFICANT "
                  f"incremental value ({da['mean']:+.4f}, "
                  f"CI: [{da['ci_lo']:+.4f}, {da['ci_hi']:+.4f}])")
        else:
            print(f"    2. Attention shows NO incremental value "
                  f"({da['mean']:+.4f})")

    # 3. Propensity matching
    d_attn = deltas.get('B_gene_plus_attn')
    if d_attn is not None:
        print(f"    3. After propensity matching: ΔAUROC(attn) = {d_attn:+.4f}")

    # 4. Residualization
    resid_attn = part4.get('attention_ols_full', {}).get('auroc_abs_residual')
    if resid_attn is not None:
        if resid_attn > 0.5:
            print(f"    4. Residualized attention retains signal "
                  f"(AUROC|resid| = {resid_attn:.3f})")
        else:
            print(f"    4. Residualized attention has NO residual signal "
                  f"(AUROC|resid| = {resid_attn:.3f})")

    print(f"\n  Output: {OUT_DIR}")


if __name__ == '__main__':
    run()
