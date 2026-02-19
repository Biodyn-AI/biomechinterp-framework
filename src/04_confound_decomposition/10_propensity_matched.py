#!/usr/bin/env python3
"""
Round 13 Experiment: Propensity-Matched Perturbation Benchmark
==============================================================
Tests whether attention/correlation edge signal is primarily driven by
gene-level confounds (some genes are just more likely to show DE regardless
of the specific perturbation).

Approach:
  1. Build full dataset (same as round11)
  2. Propensity-score matching: within each perturbation, match each DE+
     target with k=5 DE- targets that have similar gene-level features
  3. Evaluate edge scores on the balanced matched set
  4. Bootstrap CIs on the matched dataset
  5. Direct-target evaluation using TRRUST TF-target database

Addresses reviewer concern: is edge signal confounded by gene propensity?
"""

import os, sys, json, pickle, warnings, time
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
EXTRACTION_DIR = os.path.join(PAPER_DIR, "experiments", "round9_unified_extraction")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round13_propensity_matched")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

PRIMARY_LAYER = 12
N_FOLDS = 5
N_BOOTSTRAP = 200
K_MATCHES = 5
CALIPER_MULTIPLIER = 0.25
SEED = 42


# ── Step 1: Dataset construction (identical to round11) ─────────────────────

def build_dataset(attn_edges, corr_matrix, de_labels, gene_features_df,
                  hvg_genes, hvg_to_local):
    """Build the full dataset: one row per (perturbation, target) pair."""
    hvg_upper = [g.upper() for g in hvg_genes]
    gene_feat_dict = {}
    for _, row in gene_features_df.iterrows():
        gene_feat_dict[row['gene_upper']] = {
            'mean_expr': row['mean_expr'],
            'variance': row['variance'],
            'dropout_rate': row['dropout_rate'],
            'n_perturbations_affected': row.get('n_perturbations_affected', 0),
        }

    rows = []
    for pg in sorted(de_labels.keys()):
        targets = de_labels[pg]
        pg_local = hvg_to_local.get(pg.upper())
        if pg_local is None:
            continue
        for tgt_gene, is_de in targets.items():
            tgt_local = hvg_to_local.get(tgt_gene)
            if tgt_local is None:
                continue
            tgt_orig = hvg_genes[tgt_local]
            fwd = attn_edges.get((pg, tgt_orig), 0.0)
            rev = attn_edges.get((tgt_orig, pg), 0.0)
            attn_score = max(fwd, rev)
            corr_score = abs(corr_matrix[pg_local, tgt_local])
            tgt_feats = gene_feat_dict.get(tgt_gene, {})
            rows.append({
                'perturbation': pg,
                'target': tgt_gene,
                'is_de': int(is_de),
                'attn_edge': attn_score,
                'corr_edge': corr_score,
                'tgt_mean_expr': tgt_feats.get('mean_expr', 0),
                'tgt_variance': tgt_feats.get('variance', 0),
                'tgt_dropout': tgt_feats.get('dropout_rate', 0),
                'tgt_n_affected': tgt_feats.get('n_perturbations_affected', 0),
            })
    return pd.DataFrame(rows)


# ── Step 2: Propensity-score matching ────────────────────────────────────────

def compute_propensity_scores(df_pert):
    """
    Compute propensity scores for a single perturbation's targets.

    Uses PCA(2) on (mean_expr, variance, dropout_rate) to avoid
    multicollinearity (dropout and mean_expr correlated at r=-0.96),
    then fits regularized logistic regression to predict DE status.
    """
    confounds = ['tgt_mean_expr', 'tgt_variance', 'tgt_dropout']
    X = df_pert[confounds].values
    y = df_pert['is_de'].values

    # If only one class, propensity is trivially 0 or 1
    if y.sum() == 0 or y.sum() == len(y):
        return np.full(len(y), y.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA to 2 components to handle multicollinearity
    n_components = min(2, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    lr = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs')
    lr.fit(X_pca, y)
    propensity = lr.predict_proba(X_pca)[:, 1]
    return propensity


def propensity_match_perturbation(df_pert, k=5, caliper_mult=0.25, rng=None):
    """
    For one perturbation, match each positive target with k nearest
    negative targets by propensity score.

    Returns:
        matched_indices: list of indices (from df_pert) in the matched set
        n_wider_caliper: count of positives where caliper was insufficient
    """
    if rng is None:
        rng = np.random.RandomState(42)

    propensity = compute_propensity_scores(df_pert)
    df_pert = df_pert.copy()
    df_pert['propensity'] = propensity

    pos_mask = df_pert['is_de'] == 1
    neg_mask = df_pert['is_de'] == 0

    pos_idx = df_pert.index[pos_mask].tolist()
    neg_idx = df_pert.index[neg_mask].tolist()

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return [], 0

    pos_scores = df_pert.loc[pos_idx, 'propensity'].values
    neg_scores = df_pert.loc[neg_idx, 'propensity'].values

    caliper = caliper_mult * np.std(propensity)

    # Use NearestNeighbors for efficient matching
    nn = NearestNeighbors(n_neighbors=min(k, len(neg_idx)), metric='euclidean')
    nn.fit(neg_scores.reshape(-1, 1))

    matched_neg_indices = set()
    matched_pos_indices = set()
    n_wider_caliper = 0

    for p_idx, p_score in zip(pos_idx, pos_scores):
        distances, neighbors = nn.kneighbors(np.array([[p_score]]))
        distances = distances[0]
        neighbors = neighbors[0]

        # Filter by caliper
        within_caliper = distances <= caliper
        if within_caliper.sum() >= 1:
            selected_neighbors = neighbors[within_caliper][:k]
        else:
            # Use nearest k regardless (wider caliper)
            selected_neighbors = neighbors[:k]
            n_wider_caliper += 1

        matched_pos_indices.add(p_idx)
        for ni in selected_neighbors:
            matched_neg_indices.add(neg_idx[ni])

    all_matched = sorted(matched_pos_indices | matched_neg_indices)
    return all_matched, n_wider_caliper


def run_propensity_matching(df, k=5, caliper_mult=0.25):
    """
    Run propensity matching across all perturbations.

    Returns:
        matched_df: DataFrame with matched subset
        match_summary: dict with summary statistics
    """
    print("\n  Running propensity-score matching...")
    rng = np.random.RandomState(SEED)

    all_matched_indices = []
    total_pos_before = 0
    total_neg_before = 0
    total_pos_after = 0
    total_neg_after = 0
    total_wider_caliper = 0
    n_perts_matched = 0
    n_perts_skipped = 0

    perturbations = sorted(df['perturbation'].unique())
    for i, pg in enumerate(perturbations):
        df_pert = df[df['perturbation'] == pg]
        n_pos = df_pert['is_de'].sum()
        n_neg = len(df_pert) - n_pos

        total_pos_before += n_pos
        total_neg_before += n_neg

        if n_pos == 0 or n_neg == 0:
            n_perts_skipped += 1
            continue

        matched_idx, n_wider = propensity_match_perturbation(
            df_pert, k=k, caliper_mult=caliper_mult, rng=rng
        )

        if len(matched_idx) == 0:
            n_perts_skipped += 1
            continue

        matched_sub = df.loc[matched_idx]
        total_pos_after += matched_sub['is_de'].sum()
        total_neg_after += len(matched_sub) - matched_sub['is_de'].sum()
        total_wider_caliper += n_wider
        n_perts_matched += 1
        all_matched_indices.extend(matched_idx)

        if (i + 1) % 50 == 0:
            print(f"    Matched {i+1}/{len(perturbations)} perturbations...")

    matched_df = df.loc[sorted(set(all_matched_indices))].copy()

    # Compute covariate balance (SMD) before and after matching
    confounds = ['tgt_mean_expr', 'tgt_variance', 'tgt_dropout', 'tgt_n_affected']
    balance_before = {}
    balance_after = {}

    for feat in confounds:
        # Before matching
        pos_vals = df[df['is_de'] == 1][feat].values
        neg_vals = df[df['is_de'] == 0][feat].values
        pooled_std = np.sqrt((np.var(pos_vals) + np.var(neg_vals)) / 2)
        if pooled_std > 0:
            smd_before = (np.mean(pos_vals) - np.mean(neg_vals)) / pooled_std
        else:
            smd_before = 0.0
        balance_before[feat] = {
            'smd': float(smd_before),
            'mean_pos': float(np.mean(pos_vals)),
            'mean_neg': float(np.mean(neg_vals)),
        }

        # After matching
        pos_vals_m = matched_df[matched_df['is_de'] == 1][feat].values
        neg_vals_m = matched_df[matched_df['is_de'] == 0][feat].values
        pooled_std_m = np.sqrt((np.var(pos_vals_m) + np.var(neg_vals_m)) / 2)
        if pooled_std_m > 0:
            smd_after = (np.mean(pos_vals_m) - np.mean(neg_vals_m)) / pooled_std_m
        else:
            smd_after = 0.0
        balance_after[feat] = {
            'smd': float(smd_after),
            'mean_pos': float(np.mean(pos_vals_m)),
            'mean_neg': float(np.mean(neg_vals_m)),
        }

    match_summary = {
        'n_perturbations_total': len(perturbations),
        'n_perturbations_matched': n_perts_matched,
        'n_perturbations_skipped': n_perts_skipped,
        'k_matches': k,
        'caliper_multiplier': caliper_mult,
        'n_pairs_before': len(df),
        'n_pairs_after': len(matched_df),
        'n_pos_before': int(total_pos_before),
        'n_neg_before': int(total_neg_before),
        'n_pos_after': int(total_pos_after),
        'n_neg_after': int(total_neg_after),
        'positive_rate_before': float(total_pos_before / len(df)),
        'positive_rate_after': float(total_pos_after / len(matched_df)) if len(matched_df) > 0 else 0,
        'n_wider_caliper': total_wider_caliper,
        'pct_wider_caliper': float(total_wider_caliper / total_pos_before * 100) if total_pos_before > 0 else 0,
        'balance_before': balance_before,
        'balance_after': balance_after,
    }

    return matched_df, match_summary


# ── Step 3a: Raw edge score evaluation ───────────────────────────────────────

def evaluate_raw_edges_per_perturbation(df):
    """
    For each perturbation, rank targets by attention/correlation edge weight,
    compute AUROC and AUPRC. Return mean across perturbations.
    """
    perturbations = df['perturbation'].unique()
    attn_aurocs, attn_auprcs = [], []
    corr_aurocs, corr_auprcs = [], []

    for pg in perturbations:
        sub = df[df['perturbation'] == pg]
        y = sub['is_de'].values
        if y.sum() == 0 or y.sum() == len(y):
            continue

        # Attention edges
        attn_scores = sub['attn_edge'].values
        attn_aurocs.append(roc_auc_score(y, attn_scores))
        attn_auprcs.append(average_precision_score(y, attn_scores))

        # Correlation edges
        corr_scores = sub['corr_edge'].values
        corr_aurocs.append(roc_auc_score(y, corr_scores))
        corr_auprcs.append(average_precision_score(y, corr_scores))

    return {
        'attn_raw': {
            'mean_auroc': float(np.mean(attn_aurocs)),
            'std_auroc': float(np.std(attn_aurocs)),
            'mean_auprc': float(np.mean(attn_auprcs)),
            'std_auprc': float(np.std(attn_auprcs)),
            'n_perturbations': len(attn_aurocs),
        },
        'corr_raw': {
            'mean_auroc': float(np.mean(corr_aurocs)),
            'std_auroc': float(np.std(corr_aurocs)),
            'mean_auprc': float(np.mean(corr_auprcs)),
            'std_auprc': float(np.std(corr_auprcs)),
            'n_perturbations': len(corr_aurocs),
        },
    }


# ── Step 3b: Incremental value (LR, GroupKFold) ─────────────────────────────

def evaluate_cv_lr(df, feature_cols, train_idx, test_idx):
    """Run logistic regression CV fold, return AUROC and AUPRC."""
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

    auroc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    return {'auroc': auroc, 'auprc': auprc}


def run_incremental_cv(df, n_folds=5, seed=42):
    """
    Run 5-fold GroupKFold (by perturbation) for 3 model specs:
      A: gene features only
      B: gene features + attn_edge
      C: gene features + corr_edge
    """
    gene_features = ['tgt_mean_expr', 'tgt_variance', 'tgt_dropout', 'tgt_n_affected']
    model_specs = {
        'A_gene_only': gene_features,
        'B_gene_plus_attn': gene_features + ['attn_edge'],
        'C_gene_plus_corr': gene_features + ['corr_edge'],
    }

    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(
        np.arange(len(df)), groups=df['perturbation'].values
    ))

    results = {name: {'aurocs': [], 'auprcs': []} for name in model_specs}

    for train_idx, test_idx in splits:
        for name, feat_cols in model_specs.items():
            res = evaluate_cv_lr(df, feat_cols, train_idx, test_idx)
            if res is not None:
                results[name]['aurocs'].append(res['auroc'])
                results[name]['auprcs'].append(res['auprc'])

    summary = {}
    for name, data in results.items():
        aurocs = np.array(data['aurocs'])
        auprcs = np.array(data['auprcs'])
        summary[name] = {
            'mean_auroc': float(aurocs.mean()) if len(aurocs) > 0 else None,
            'std_auroc': float(aurocs.std()) if len(aurocs) > 0 else None,
            'mean_auprc': float(auprcs.mean()) if len(auprcs) > 0 else None,
            'std_auprc': float(auprcs.std()) if len(auprcs) > 0 else None,
            'n_folds_valid': len(aurocs),
        }

    # Compute deltas
    if summary['A_gene_only']['mean_auroc'] is not None:
        base_auroc = summary['A_gene_only']['mean_auroc']
        base_auprc = summary['A_gene_only']['mean_auprc']
        for aug_name in ['B_gene_plus_attn', 'C_gene_plus_corr']:
            if summary[aug_name]['mean_auroc'] is not None:
                summary[f'delta_auroc_{aug_name}'] = float(
                    summary[aug_name]['mean_auroc'] - base_auroc
                )
                summary[f'delta_auprc_{aug_name}'] = float(
                    summary[aug_name]['mean_auprc'] - base_auprc
                )

    return summary


# ── Step 3c: Bootstrap CIs ──────────────────────────────────────────────────

def bootstrap_matched_cv(matched_df, n_bootstrap=200, n_folds=5, seed=42):
    """
    Perturbation-level bootstrap on matched dataset.
    Sequential computation (proven fast enough from R11).
    """
    rng = np.random.RandomState(seed)
    unique_perts = matched_df['perturbation'].unique()

    gene_features = ['tgt_mean_expr', 'tgt_variance', 'tgt_dropout', 'tgt_n_affected']
    model_specs = {
        'A_gene_only': gene_features,
        'B_gene_plus_attn': gene_features + ['attn_edge'],
        'C_gene_plus_corr': gene_features + ['corr_edge'],
    }

    boot_results = {name: {'aurocs': [], 'auprcs': []} for name in model_specs}
    delta_attn_aurocs = []
    delta_attn_auprcs = []
    delta_corr_aurocs = []
    delta_corr_auprcs = []

    for b in range(n_bootstrap):
        if (b + 1) % 20 == 0:
            print(f"    Bootstrap iteration {b+1}/{n_bootstrap}...")

        # Resample perturbations with replacement
        boot_perts = rng.choice(unique_perts, size=len(unique_perts), replace=True)
        boot_dfs = []
        for i, pg in enumerate(boot_perts):
            sub = matched_df[matched_df['perturbation'] == pg].copy()
            sub['perturbation'] = f"{pg}_{i}"  # unique group label
            boot_dfs.append(sub)
        boot_df = pd.concat(boot_dfs, ignore_index=True)

        # Run CV on bootstrap sample
        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(
            np.arange(len(boot_df)), groups=boot_df['perturbation'].values
        ))

        fold_results = {name: {'aurocs': [], 'auprcs': []} for name in model_specs}
        for train_idx, test_idx in splits:
            for name, feat_cols in model_specs.items():
                res = evaluate_cv_lr(boot_df, feat_cols, train_idx, test_idx)
                if res is not None:
                    fold_results[name]['aurocs'].append(res['auroc'])
                    fold_results[name]['auprcs'].append(res['auprc'])

        # Aggregate fold means for this bootstrap iteration
        means = {}
        for name in model_specs:
            aurocs_arr = np.array(fold_results[name]['aurocs'])
            auprcs_arr = np.array(fold_results[name]['auprcs'])
            if len(aurocs_arr) > 0:
                means[name] = {
                    'auroc': float(aurocs_arr.mean()),
                    'auprc': float(auprcs_arr.mean()),
                }
                boot_results[name]['aurocs'].append(float(aurocs_arr.mean()))
                boot_results[name]['auprcs'].append(float(auprcs_arr.mean()))
            else:
                means[name] = None

        # Compute deltas for this iteration
        if means.get('A_gene_only') is not None:
            base_auroc = means['A_gene_only']['auroc']
            base_auprc = means['A_gene_only']['auprc']
            if means.get('B_gene_plus_attn') is not None:
                delta_attn_aurocs.append(means['B_gene_plus_attn']['auroc'] - base_auroc)
                delta_attn_auprcs.append(means['B_gene_plus_attn']['auprc'] - base_auprc)
            if means.get('C_gene_plus_corr') is not None:
                delta_corr_aurocs.append(means['C_gene_plus_corr']['auroc'] - base_auroc)
                delta_corr_auprcs.append(means['C_gene_plus_corr']['auprc'] - base_auprc)

    # Build CI summary
    ci_summary = {}
    for name in model_specs:
        aurocs = np.array(boot_results[name]['aurocs'])
        auprcs = np.array(boot_results[name]['auprcs'])
        ci_summary[name] = {
            'auroc_mean': float(aurocs.mean()) if len(aurocs) > 0 else None,
            'auroc_ci_lo': float(np.percentile(aurocs, 2.5)) if len(aurocs) > 0 else None,
            'auroc_ci_hi': float(np.percentile(aurocs, 97.5)) if len(aurocs) > 0 else None,
            'auprc_mean': float(auprcs.mean()) if len(auprcs) > 0 else None,
            'auprc_ci_lo': float(np.percentile(auprcs, 2.5)) if len(auprcs) > 0 else None,
            'auprc_ci_hi': float(np.percentile(auprcs, 97.5)) if len(auprcs) > 0 else None,
            'n_valid': len(aurocs),
        }

    delta_attn_aurocs = np.array(delta_attn_aurocs)
    delta_attn_auprcs = np.array(delta_attn_auprcs)
    delta_corr_aurocs = np.array(delta_corr_aurocs)
    delta_corr_auprcs = np.array(delta_corr_auprcs)

    ci_summary['delta_attn_auroc'] = {
        'mean': float(delta_attn_aurocs.mean()) if len(delta_attn_aurocs) > 0 else None,
        'ci_lo': float(np.percentile(delta_attn_aurocs, 2.5)) if len(delta_attn_aurocs) > 0 else None,
        'ci_hi': float(np.percentile(delta_attn_aurocs, 97.5)) if len(delta_attn_aurocs) > 0 else None,
        'n_valid': len(delta_attn_aurocs),
    }
    ci_summary['delta_attn_auprc'] = {
        'mean': float(delta_attn_auprcs.mean()) if len(delta_attn_auprcs) > 0 else None,
        'ci_lo': float(np.percentile(delta_attn_auprcs, 2.5)) if len(delta_attn_auprcs) > 0 else None,
        'ci_hi': float(np.percentile(delta_attn_auprcs, 97.5)) if len(delta_attn_auprcs) > 0 else None,
        'n_valid': len(delta_attn_auprcs),
    }
    ci_summary['delta_corr_auroc'] = {
        'mean': float(delta_corr_aurocs.mean()) if len(delta_corr_aurocs) > 0 else None,
        'ci_lo': float(np.percentile(delta_corr_aurocs, 2.5)) if len(delta_corr_aurocs) > 0 else None,
        'ci_hi': float(np.percentile(delta_corr_aurocs, 97.5)) if len(delta_corr_aurocs) > 0 else None,
        'n_valid': len(delta_corr_aurocs),
    }
    ci_summary['delta_corr_auprc'] = {
        'mean': float(delta_corr_auprcs.mean()) if len(delta_corr_auprcs) > 0 else None,
        'ci_lo': float(np.percentile(delta_corr_auprcs, 2.5)) if len(delta_corr_auprcs) > 0 else None,
        'ci_hi': float(np.percentile(delta_corr_auprcs, 97.5)) if len(delta_corr_auprcs) > 0 else None,
        'n_valid': len(delta_corr_auprcs),
    }

    return ci_summary


# ── Step 4: Direct-target evaluation (TRRUST) ───────────────────────────────

def load_trrust(trrust_path):
    """Load TRRUST TF-target database. Returns dict {TF_upper: set(target_upper)}."""
    trrust = {}
    with open(trrust_path, 'r') as f:
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


def evaluate_direct_targets(df, trrust, hvg_genes):
    """
    For TF perturbations in TRRUST that overlap with HVG set:
      - Positive: TRRUST-annotated targets that are in the HVG target set
      - Negative: non-TRRUST targets
      - Evaluate attention and correlation edge AUROC/AUPRC
    """
    hvg_upper = set(g.upper() for g in hvg_genes)
    perturbations = df['perturbation'].unique()

    # Find TFs that are perturbation genes AND in TRRUST
    tf_perts = [pg for pg in perturbations if pg.upper() in trrust]

    results_per_tf = []
    total_direct_targets = 0
    total_evaluable_tfs = 0

    for pg in tf_perts:
        trrust_targets = trrust[pg.upper()] & hvg_upper

        if len(trrust_targets) == 0:
            continue

        # Get this perturbation's data
        df_pert = df[df['perturbation'] == pg].copy()
        df_pert['is_trrust_target'] = df_pert['target'].apply(
            lambda t: int(t.upper() in trrust_targets)
        )

        n_pos = df_pert['is_trrust_target'].sum()
        n_neg = len(df_pert) - n_pos

        if n_pos == 0 or n_neg == 0:
            continue

        y = df_pert['is_trrust_target'].values
        total_direct_targets += n_pos
        total_evaluable_tfs += 1

        # Attention edges
        attn_scores = df_pert['attn_edge'].values
        attn_auroc = roc_auc_score(y, attn_scores)
        attn_auprc = average_precision_score(y, attn_scores)

        # Correlation edges
        corr_scores = df_pert['corr_edge'].values
        corr_auroc = roc_auc_score(y, corr_scores)
        corr_auprc = average_precision_score(y, corr_scores)

        results_per_tf.append({
            'tf': pg,
            'n_trrust_targets': n_pos,
            'n_non_targets': n_neg,
            'attn_auroc': attn_auroc,
            'attn_auprc': attn_auprc,
            'corr_auroc': corr_auroc,
            'corr_auprc': corr_auprc,
        })

    if len(results_per_tf) == 0:
        return {
            'n_evaluable_tfs': 0,
            'n_direct_targets': 0,
            'message': 'No evaluable TF perturbations found',
        }

    results_df = pd.DataFrame(results_per_tf)
    summary = {
        'n_evaluable_tfs': total_evaluable_tfs,
        'n_direct_targets_total': total_direct_targets,
        'mean_attn_auroc': float(results_df['attn_auroc'].mean()),
        'std_attn_auroc': float(results_df['attn_auroc'].std()),
        'mean_attn_auprc': float(results_df['attn_auprc'].mean()),
        'std_attn_auprc': float(results_df['attn_auprc'].std()),
        'mean_corr_auroc': float(results_df['corr_auroc'].mean()),
        'std_corr_auroc': float(results_df['corr_auroc'].std()),
        'mean_corr_auprc': float(results_df['corr_auprc'].mean()),
        'std_corr_auprc': float(results_df['corr_auprc'].std()),
        'per_tf_results': results_per_tf,
    }
    return summary


# ── Step 5: Figure generation ────────────────────────────────────────────────

def generate_figure(match_summary, eval_results, direct_results, boot_cis):
    """
    3-panel figure:
      A: Covariate balance (SMD before/after matching)
      B: Edge evaluation on matched set (AUROC bars)
      C: Direct-target evaluation (AUROC/AUPRC for attention vs correlation)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Panel A: Covariate balance ──────────────────────────────────────
    ax = axes[0]
    confounds = ['tgt_mean_expr', 'tgt_variance', 'tgt_dropout', 'tgt_n_affected']
    labels = ['Mean expr', 'Variance', 'Dropout', 'N affected']
    smds_before = [abs(match_summary['balance_before'][c]['smd']) for c in confounds]
    smds_after = [abs(match_summary['balance_after'][c]['smd']) for c in confounds]

    x = np.arange(len(confounds))
    width = 0.35
    bars_before = ax.bar(x - width/2, smds_before, width, label='Before matching',
                          color='#E57373', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars_after = ax.bar(x + width/2, smds_after, width, label='After matching',
                         color='#81C784', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.axhline(0.1, color='gray', linestyle='--', alpha=0.7, label='SMD=0.1 threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('|Standardized Mean Difference|', fontsize=10)
    ax.set_title('A. Covariate Balance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # Annotate bars
    for bar, val in zip(bars_before, smds_before):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=7)
    for bar, val in zip(bars_after, smds_after):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=7)

    # ── Panel B: Edge evaluation on matched set ─────────────────────────
    ax = axes[1]

    # Collect values for 5 bars:
    # attn_raw, corr_raw, gene_only_LR, gene+attn_LR, gene+corr_LR
    bar_labels = ['Attn\n(raw)', 'Corr\n(raw)', 'Gene-only\n(LR)',
                  'Gene+Attn\n(LR)', 'Gene+Corr\n(LR)']
    bar_colors = ['#DD8452', '#55A868', '#4C72B0', '#C44E52', '#8172B3']

    raw_results = eval_results.get('raw_edge_results', {})
    cv_results = eval_results.get('cv_results', {})

    auroc_vals = [
        raw_results.get('attn_raw', {}).get('mean_auroc', 0.5),
        raw_results.get('corr_raw', {}).get('mean_auroc', 0.5),
        cv_results.get('A_gene_only', {}).get('mean_auroc', 0.5),
        cv_results.get('B_gene_plus_attn', {}).get('mean_auroc', 0.5),
        cv_results.get('C_gene_plus_corr', {}).get('mean_auroc', 0.5),
    ]

    # Bootstrap CIs for LR models (raw edges don't have CIs here)
    ci_lo_vals = [0, 0, 0, 0, 0]
    ci_hi_vals = [0, 0, 0, 0, 0]
    for idx, ci_key in enumerate(['A_gene_only', 'B_gene_plus_attn', 'C_gene_plus_corr']):
        if ci_key in boot_cis and boot_cis[ci_key].get('auroc_ci_lo') is not None:
            ci_lo_vals[idx + 2] = auroc_vals[idx + 2] - boot_cis[ci_key]['auroc_ci_lo']
            ci_hi_vals[idx + 2] = boot_cis[ci_key]['auroc_ci_hi'] - auroc_vals[idx + 2]

    x = np.arange(len(bar_labels))
    bars = ax.bar(x, auroc_vals, color=bar_colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    ax.errorbar(x, auroc_vals, yerr=[ci_lo_vals, ci_hi_vals],
                fmt='none', ecolor='black', capsize=3, linewidth=1)

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Random (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel('AUROC', fontsize=10)
    ax.set_title('B. Edge Evaluation (Matched Set)', fontsize=12, fontweight='bold')
    ax.set_ylim(0.45, max(auroc_vals) * 1.1 if max(auroc_vals) > 0.5 else 0.7)
    ax.legend(fontsize=8, loc='upper left')

    for i, (bar, val) in enumerate(zip(bars, auroc_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ci_hi_vals[i], 0) + 0.005,
                f'{val:.3f}', ha='center', fontsize=8)

    # Add delta annotations
    if 'delta_auroc_B_gene_plus_attn' in cv_results:
        delta_attn = cv_results['delta_auroc_B_gene_plus_attn']
        delta_corr = cv_results.get('delta_auroc_C_gene_plus_corr', 0)
        ax.text(0.95, 0.95,
                f'$\\Delta$AUROC(attn): {delta_attn:+.4f}\n$\\Delta$AUROC(corr): {delta_corr:+.4f}',
                transform=ax.transAxes, fontsize=8, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    # ── Panel C: Direct-target evaluation ───────────────────────────────
    ax = axes[2]

    if direct_results.get('n_evaluable_tfs', 0) > 0:
        metrics = ['AUROC', 'AUPRC']
        attn_vals = [direct_results['mean_attn_auroc'], direct_results['mean_attn_auprc']]
        corr_vals = [direct_results['mean_corr_auroc'], direct_results['mean_corr_auprc']]
        attn_errs = [direct_results['std_attn_auroc'], direct_results['std_attn_auprc']]
        corr_errs = [direct_results['std_corr_auroc'], direct_results['std_corr_auprc']]

        x = np.arange(len(metrics))
        width = 0.35
        bars_attn = ax.bar(x - width/2, attn_vals, width, yerr=attn_errs,
                            label='Attention edges', color='#DD8452', alpha=0.8,
                            edgecolor='black', linewidth=0.5, capsize=3)
        bars_corr = ax.bar(x + width/2, corr_vals, width, yerr=corr_errs,
                            label='Correlation edges', color='#55A868', alpha=0.8,
                            edgecolor='black', linewidth=0.5, capsize=3)

        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(
            f'C. Direct-Target Recovery (TRRUST)\n'
            f'{direct_results["n_evaluable_tfs"]} TFs, '
            f'{direct_results["n_direct_targets_total"]} targets',
            fontsize=12, fontweight='bold'
        )
        ax.legend(fontsize=9, loc='upper right')

        for bar, val in zip(bars_attn, attn_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=8)
        for bar, val in zip(bars_corr, corr_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No evaluable TF\nperturbations found',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('C. Direct-Target Recovery (TRRUST)', fontsize=12, fontweight='bold')

    fig.suptitle('Round 13: Propensity-Matched Perturbation Benchmark\n'
                 '(Controlling for Gene-Level Confounds)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "fig_propensity_matched.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig_path


# ── Main execution ───────────────────────────────────────────────────────────

def run():
    print("=" * 70)
    print("PROPENSITY-MATCHED PERTURBATION BENCHMARK (ROUND 13)")
    print("=" * 70)
    print("Tests whether edge signal survives after controlling for")
    print("gene-level confounds (mean_expr, variance, dropout_rate).")
    t0 = time.time()

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/5] Loading extraction data...")
    with open(os.path.join(EXTRACTION_DIR, "hvg_gene_list.json")) as f:
        hvg_genes = json.load(f)
    hvg_to_local = {g.upper(): i for i, g in enumerate(hvg_genes)}

    with open(os.path.join(EXTRACTION_DIR, "de_labels.pkl"), 'rb') as f:
        de_labels = pickle.load(f)

    corr_matrix = np.load(os.path.join(EXTRACTION_DIR, "correlation_matrix.npy"))
    gene_features_df = pd.read_csv(os.path.join(EXTRACTION_DIR, "gene_features.csv"))

    edge_path = os.path.join(
        EXTRACTION_DIR, "layer_mean_edges",
        f"layer_{PRIMARY_LAYER:02d}_edges.pkl"
    )
    with open(edge_path, 'rb') as f:
        attn_edges = pickle.load(f)

    print(f"  Loaded: {len(hvg_genes)} HVG genes, {len(de_labels)} perturbations")
    print(f"  Correlation matrix: {corr_matrix.shape}")
    print(f"  Attention edges: {len(attn_edges)} pairs (layer {PRIMARY_LAYER})")

    # ── Build dataset ────────────────────────────────────────────────────
    print("\n[1/5] Building dataset...")
    df = build_dataset(attn_edges, corr_matrix, de_labels, gene_features_df,
                       hvg_genes, hvg_to_local)
    n_perts = df['perturbation'].nunique()
    n_targets = df['target'].nunique()
    n_pos = df['is_de'].sum()
    print(f"  {len(df)} rows, {n_perts} perturbations, {n_targets} unique targets, "
          f"{n_pos} positive ({100*n_pos/len(df):.1f}%)")

    # ── Propensity-score matching ────────────────────────────────────────
    print("\n[2/5] Propensity-score matching...")
    matched_df, match_summary = run_propensity_matching(
        df, k=K_MATCHES, caliper_mult=CALIPER_MULTIPLIER
    )

    n_matched = len(matched_df)
    n_matched_pos = matched_df['is_de'].sum()
    print(f"\n  Matching complete:")
    print(f"    Before: {len(df)} pairs ({n_pos} pos, {100*n_pos/len(df):.1f}%)")
    print(f"    After:  {n_matched} pairs ({n_matched_pos} pos, "
          f"{100*n_matched_pos/n_matched:.1f}%)")
    print(f"    Perturbations matched: {match_summary['n_perturbations_matched']}")
    print(f"    Perturbations skipped: {match_summary['n_perturbations_skipped']}")
    print(f"    Wider caliper used: {match_summary['n_wider_caliper']} positives "
          f"({match_summary['pct_wider_caliper']:.1f}%)")

    print(f"\n  Covariate balance (|SMD|):")
    confounds = ['tgt_mean_expr', 'tgt_variance', 'tgt_dropout', 'tgt_n_affected']
    all_smd_below_threshold = True
    for c in confounds:
        before = abs(match_summary['balance_before'][c]['smd'])
        after = abs(match_summary['balance_after'][c]['smd'])
        status = "OK" if after < 0.1 else "WARN"
        if after >= 0.1:
            all_smd_below_threshold = False
        print(f"    {c:25s}  before={before:.4f}  after={after:.4f}  [{status}]")
    if all_smd_below_threshold:
        print(f"    All SMDs < 0.1 after matching -- balance achieved")
    else:
        print(f"    WARNING: Some SMDs >= 0.1 after matching")

    # Save matching checkpoint
    with open(os.path.join(OUT_DIR, "matching_summary.json"), 'w') as f:
        json.dump(match_summary, f, indent=2)
    print(f"  [checkpoint: matching_summary.json saved]")

    # ── Evaluate on matched dataset ──────────────────────────────────────
    print("\n[3/5] Evaluating on matched dataset...")

    # 3a. Raw edge scores
    print("\n  3a. Raw edge scores (per-perturbation ranking)...")
    raw_edge_results = evaluate_raw_edges_per_perturbation(matched_df)
    print(f"    Attention:   AUROC={raw_edge_results['attn_raw']['mean_auroc']:.4f} "
          f"(+/-{raw_edge_results['attn_raw']['std_auroc']:.4f}), "
          f"AUPRC={raw_edge_results['attn_raw']['mean_auprc']:.4f} "
          f"(+/-{raw_edge_results['attn_raw']['std_auprc']:.4f})")
    print(f"    Correlation: AUROC={raw_edge_results['corr_raw']['mean_auroc']:.4f} "
          f"(+/-{raw_edge_results['corr_raw']['std_auroc']:.4f}), "
          f"AUPRC={raw_edge_results['corr_raw']['mean_auprc']:.4f} "
          f"(+/-{raw_edge_results['corr_raw']['std_auprc']:.4f})")

    # 3b. Incremental value (LR, cross-perturbation)
    print("\n  3b. Incremental value (LR, 5-fold GroupKFold)...")
    cv_results = run_incremental_cv(matched_df, n_folds=N_FOLDS, seed=SEED)
    for name in ['A_gene_only', 'B_gene_plus_attn', 'C_gene_plus_corr']:
        r = cv_results[name]
        print(f"    {name:25s}  AUROC={r['mean_auroc']:.4f} (+/-{r['std_auroc']:.4f}), "
              f"AUPRC={r['mean_auprc']:.4f} (+/-{r['std_auprc']:.4f})")
    if 'delta_auroc_B_gene_plus_attn' in cv_results:
        print(f"    DELTA(gene+attn - gene_only):  "
              f"AUROC={cv_results['delta_auroc_B_gene_plus_attn']:+.4f}, "
              f"AUPRC={cv_results['delta_auprc_B_gene_plus_attn']:+.4f}")
    if 'delta_auroc_C_gene_plus_corr' in cv_results:
        print(f"    DELTA(gene+corr - gene_only):  "
              f"AUROC={cv_results['delta_auroc_C_gene_plus_corr']:+.4f}, "
              f"AUPRC={cv_results['delta_auprc_C_gene_plus_corr']:+.4f}")

    # Save evaluation checkpoint
    eval_results = {
        'raw_edge_results': raw_edge_results,
        'cv_results': cv_results,
    }
    with open(os.path.join(OUT_DIR, "evaluation_results.json"), 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"  [checkpoint: evaluation_results.json saved]")

    # 3c. Bootstrap CIs
    print(f"\n  3c. Bootstrap CIs ({N_BOOTSTRAP} iterations, sequential LR)...")
    t_boot = time.time()
    boot_cis = bootstrap_matched_cv(
        matched_df, n_bootstrap=N_BOOTSTRAP, n_folds=N_FOLDS, seed=SEED
    )
    elapsed_boot = time.time() - t_boot
    print(f"    Bootstrap completed in {elapsed_boot:.1f}s "
          f"({elapsed_boot/N_BOOTSTRAP:.1f}s per iteration)")

    for name in ['A_gene_only', 'B_gene_plus_attn', 'C_gene_plus_corr']:
        ci = boot_cis[name]
        print(f"    {name:25s}  AUROC={ci['auroc_mean']:.4f} "
              f"[{ci['auroc_ci_lo']:.4f}, {ci['auroc_ci_hi']:.4f}]")

    da = boot_cis['delta_attn_auroc']
    dc = boot_cis['delta_corr_auroc']
    print(f"    DELTA attn AUROC: {da['mean']:+.4f} [{da['ci_lo']:+.4f}, {da['ci_hi']:+.4f}]")
    print(f"    DELTA corr AUROC: {dc['mean']:+.4f} [{dc['ci_lo']:+.4f}, {dc['ci_hi']:+.4f}]")

    da_pr = boot_cis['delta_attn_auprc']
    dc_pr = boot_cis['delta_corr_auprc']
    print(f"    DELTA attn AUPRC: {da_pr['mean']:+.4f} "
          f"[{da_pr['ci_lo']:+.4f}, {da_pr['ci_hi']:+.4f}]")
    print(f"    DELTA corr AUPRC: {dc_pr['mean']:+.4f} "
          f"[{dc_pr['ci_lo']:+.4f}, {dc_pr['ci_hi']:+.4f}]")

    # Save bootstrap checkpoint
    with open(os.path.join(OUT_DIR, "bootstrap_cis.json"), 'w') as f:
        json.dump(boot_cis, f, indent=2)
    print(f"  [checkpoint: bootstrap_cis.json saved]")

    # ── Direct-target evaluation (TRRUST) ────────────────────────────────
    print("\n[4/5] Direct-target evaluation (TRRUST)...")
    trrust_path = os.path.join(PAPER_DIR, "cssi_real", "results", "trrust_human.tsv")
    trrust = load_trrust(trrust_path)
    print(f"  Loaded TRRUST: {len(trrust)} TFs, "
          f"{sum(len(v) for v in trrust.values())} TF-target relationships")

    direct_results = evaluate_direct_targets(df, trrust, hvg_genes)
    if direct_results.get('n_evaluable_tfs', 0) > 0:
        print(f"  Evaluable TF perturbations: {direct_results['n_evaluable_tfs']}")
        print(f"  Total direct targets: {direct_results['n_direct_targets_total']}")
        print(f"  Attention:   AUROC={direct_results['mean_attn_auroc']:.4f} "
              f"(+/-{direct_results['std_attn_auroc']:.4f}), "
              f"AUPRC={direct_results['mean_attn_auprc']:.4f} "
              f"(+/-{direct_results['std_attn_auprc']:.4f})")
        print(f"  Correlation: AUROC={direct_results['mean_corr_auroc']:.4f} "
              f"(+/-{direct_results['std_corr_auroc']:.4f}), "
              f"AUPRC={direct_results['mean_corr_auprc']:.4f} "
              f"(+/-{direct_results['std_corr_auprc']:.4f})")
    else:
        print(f"  No evaluable TF perturbations found.")

    # Save direct target results
    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    with open(os.path.join(OUT_DIR, "direct_target_results.json"), 'w') as f:
        json.dump(direct_results, f, indent=2, default=_json_default)
    print(f"  [checkpoint: direct_target_results.json saved]")

    # ── Generate figure ──────────────────────────────────────────────────
    print("\n[5/5] Generating figure...")
    fig_path = generate_figure(match_summary, eval_results, direct_results, boot_cis)
    print(f"  Saved: {fig_path}")

    # ── Final summary ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ROUND 13: PROPENSITY-MATCHED BENCHMARK COMPLETE - {elapsed:.1f}s")
    print(f"{'='*70}")

    print(f"\n  Dataset: {len(df)} total pairs -> {len(matched_df)} matched pairs")
    print(f"  Positive rate: {100*n_pos/len(df):.1f}% -> "
          f"{100*n_matched_pos/n_matched:.1f}% after matching")

    print(f"\n  Covariate balance achieved: {'YES' if all_smd_below_threshold else 'NO'}")
    for c in confounds:
        after = abs(match_summary['balance_after'][c]['smd'])
        print(f"    {c}: |SMD|={after:.4f}")

    print(f"\n  Raw edge scores on matched set:")
    print(f"    Attention AUROC: {raw_edge_results['attn_raw']['mean_auroc']:.4f}")
    print(f"    Correlation AUROC: {raw_edge_results['corr_raw']['mean_auroc']:.4f}")

    print(f"\n  Incremental value (LR, matched set):")
    print(f"    Gene-only AUROC:   {cv_results['A_gene_only']['mean_auroc']:.4f}")
    print(f"    Gene+Attn AUROC:   {cv_results['B_gene_plus_attn']['mean_auroc']:.4f}")
    print(f"    Gene+Corr AUROC:   {cv_results['C_gene_plus_corr']['mean_auroc']:.4f}")
    if 'delta_auroc_B_gene_plus_attn' in cv_results:
        print(f"    Delta(attn) AUROC: {cv_results['delta_auroc_B_gene_plus_attn']:+.4f} "
              f"[{da['ci_lo']:+.4f}, {da['ci_hi']:+.4f}]")
    if 'delta_auroc_C_gene_plus_corr' in cv_results:
        print(f"    Delta(corr) AUROC: {cv_results['delta_auroc_C_gene_plus_corr']:+.4f} "
              f"[{dc['ci_lo']:+.4f}, {dc['ci_hi']:+.4f}]")

    if direct_results.get('n_evaluable_tfs', 0) > 0:
        print(f"\n  TRRUST direct-target recovery ({direct_results['n_evaluable_tfs']} TFs):")
        print(f"    Attention AUROC: {direct_results['mean_attn_auroc']:.4f}")
        print(f"    Correlation AUROC: {direct_results['mean_corr_auroc']:.4f}")

    # Interpretation
    print(f"\n  INTERPRETATION:")
    if 'delta_auroc_B_gene_plus_attn' in cv_results:
        delta_attn = cv_results['delta_auroc_B_gene_plus_attn']
        ci_lo = da['ci_lo']
        if ci_lo > 0:
            print(f"    Attention edges provide statistically significant incremental")
            print(f"    value ({delta_attn:+.4f} AUROC) EVEN after propensity matching,")
            print(f"    indicating signal beyond gene-level confounds.")
        elif delta_attn > 0:
            print(f"    Attention edges show positive but not statistically significant")
            print(f"    incremental value ({delta_attn:+.4f} AUROC) after propensity matching.")
            print(f"    CI includes zero: [{ci_lo:+.4f}, {da['ci_hi']:+.4f}]")
        else:
            print(f"    Attention edges show NO incremental value ({delta_attn:+.4f} AUROC)")
            print(f"    after propensity matching, suggesting signal was primarily")
            print(f"    driven by gene-level confounds.")

    print(f"\n  Output directory: {OUT_DIR}")
    print(f"  Files:")
    print(f"    matching_summary.json")
    print(f"    evaluation_results.json")
    print(f"    direct_target_results.json")
    print(f"    bootstrap_cis.json")
    print(f"    figures/fig_propensity_matched.png")


if __name__ == "__main__":
    run()
