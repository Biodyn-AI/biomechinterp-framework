#!/usr/bin/env python3
"""
Round 11 Experiment: Metric-Robust Incremental Value Analysis
==============================================================
Extends the hard-generalization incremental value analysis from round10
with additional metrics: AUPRC bootstrap CIs, top-k recall, and a
nonlinear GBDT model.

Addresses: robustness of incremental-value claims across metric choices
           and model families (logistic regression + gradient boosting).
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
from sklearn.model_selection import GroupKFold, KFold
from sklearn.ensemble import GradientBoostingClassifier
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
EXTRACTION_DIR = os.path.join(PAPER_DIR, "experiments", "round9_unified_extraction")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round11_metric_robust_incremental")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PRIMARY_LAYER = 12
N_FOLDS = 5
N_BOOTSTRAP = 200
SEED = 42


# ── Dataset construction (identical to round10_hard_generalization.py) ───────

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


# ── Evaluation (extended: multiple metrics + model types) ────────────────────

def evaluate_cv(df, feature_cols, train_idx, test_idx, model_type='logistic'):
    """Returns dict with auroc, auprc, recall@10, recall@50, recall@100."""
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

    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    elif model_type == 'gbdt':
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_test_s)[:, 1]

    auroc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)

    # Top-k recall
    recall_at_k = {}
    n_pos = y_test.sum()
    if n_pos > 0:
        ranked_idx = np.argsort(-probs)
        for k in [10, 50, 100]:
            if k <= len(y_test):
                top_k_labels = y_test[ranked_idx[:k]]
                recall_at_k[k] = float(top_k_labels.sum() / n_pos)
            else:
                recall_at_k[k] = float(y_test.sum() / n_pos)  # = 1.0 if k >= n

    return {
        'auroc': auroc,
        'auprc': auprc,
        'recall_at_10': recall_at_k.get(10),
        'recall_at_50': recall_at_k.get(50),
        'recall_at_100': recall_at_k.get(100),
    }


# ── Split design runner (extended for multiple metrics + model types) ────────

def run_split_design(df, model_configs, split_type, n_folds=5, seed=42):
    """Run a specific split design and return per-model metrics dict."""
    n = len(df)
    results_by_model = {
        name: {'aurocs': [], 'auprcs': [],
               'recall_at_10': [], 'recall_at_50': [], 'recall_at_100': []}
        for name in model_configs
    }

    if split_type == 'cross_perturbation':
        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(np.arange(n), groups=df['perturbation'].values))
    elif split_type == 'cross_gene':
        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(np.arange(n), groups=df['target'].values))
    elif split_type == 'joint':
        # Joint: hold out perturbations AND genes simultaneously
        rng = np.random.RandomState(seed)
        unique_perts = df['perturbation'].unique()
        unique_genes = df['target'].unique()
        rng.shuffle(unique_perts)
        rng.shuffle(unique_genes)

        pert_folds = np.array_split(unique_perts, n_folds)
        gene_folds = np.array_split(unique_genes, n_folds)

        splits = []
        for k in range(n_folds):
            test_perts = set(pert_folds[k])
            test_genes = set(gene_folds[k])
            test_mask = df['perturbation'].isin(test_perts) & df['target'].isin(test_genes)
            test_idx = np.where(test_mask.values)[0]
            train_idx = np.where(~test_mask.values)[0]
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            splits.append((train_idx, test_idx))
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    for train_idx, test_idx in splits:
        for name, (model_type, feature_cols) in model_configs.items():
            result = evaluate_cv(df, feature_cols, train_idx, test_idx,
                                 model_type=model_type)
            if result is not None:
                results_by_model[name]['aurocs'].append(result['auroc'])
                results_by_model[name]['auprcs'].append(result['auprc'])
                if result['recall_at_10'] is not None:
                    results_by_model[name]['recall_at_10'].append(result['recall_at_10'])
                if result['recall_at_50'] is not None:
                    results_by_model[name]['recall_at_50'].append(result['recall_at_50'])
                if result['recall_at_100'] is not None:
                    results_by_model[name]['recall_at_100'].append(result['recall_at_100'])

    summary = {}
    for name, data in results_by_model.items():
        aurocs = np.array(data['aurocs'])
        auprcs = np.array(data['auprcs'])
        r10 = np.array(data['recall_at_10'])
        r50 = np.array(data['recall_at_50'])
        r100 = np.array(data['recall_at_100'])
        summary[name] = {
            'mean_auroc': float(aurocs.mean()) if len(aurocs) > 0 else None,
            'std_auroc': float(aurocs.std()) if len(aurocs) > 0 else None,
            'mean_auprc': float(auprcs.mean()) if len(auprcs) > 0 else None,
            'std_auprc': float(auprcs.std()) if len(auprcs) > 0 else None,
            'mean_recall_at_10': float(r10.mean()) if len(r10) > 0 else None,
            'mean_recall_at_50': float(r50.mean()) if len(r50) > 0 else None,
            'mean_recall_at_100': float(r100.mean()) if len(r100) > 0 else None,
            'n_folds_valid': len(aurocs),
        }
    return summary


# ── Bootstrap worker ─────────────────────────────────────────────────────────

def _bootstrap_worker(args):
    """Bootstrap worker for parallel CI computation."""
    seed, df, model_configs, split_type, n_folds = args
    rng = np.random.RandomState(seed)
    unique_perts = df['perturbation'].unique()
    boot_perts = rng.choice(unique_perts, size=len(unique_perts), replace=True)

    boot_dfs = []
    for i, pg in enumerate(boot_perts):
        sub = df[df['perturbation'] == pg].copy()
        sub['perturbation'] = f"{pg}_{i}"
        boot_dfs.append(sub)
    boot_df = pd.concat(boot_dfs, ignore_index=True)

    return run_split_design(boot_df, model_configs, split_type, n_folds, seed)


# ── Main execution ───────────────────────────────────────────────────────────

def run():
    print("=" * 70)
    print("METRIC-ROBUST INCREMENTAL VALUE ANALYSIS (ROUND 11)")
    print("=" * 70)
    t0 = time.time()

    # ── Load data ────────────────────────────────────────────────────────
    print("\nLoading extraction data...")
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

    # ── Build dataset ────────────────────────────────────────────────────
    print("\nBuilding dataset...")
    df = build_dataset(attn_edges, corr_matrix, de_labels, gene_features_df,
                       hvg_genes, hvg_to_local)
    n_perts = df['perturbation'].nunique()
    n_targets = df['target'].nunique()
    n_pos = df['is_de'].sum()
    print(f"  {len(df)} rows, {n_perts} perturbations, {n_targets} unique targets, "
          f"{n_pos} positive ({100*n_pos/len(df):.1f}%)")

    # ── Model configs (8 total: LR + GBDT) ──────────────────────────────
    gene_features = ['tgt_mean_expr', 'tgt_variance', 'tgt_dropout', 'tgt_n_affected']
    model_configs = {
        'A_gene_only_lr':       ('logistic', gene_features),
        'B_attn_only_lr':       ('logistic', ['attn_edge']),
        'C_corr_only_lr':       ('logistic', ['corr_edge']),
        'D_gene_plus_attn_lr':  ('logistic', gene_features + ['attn_edge']),
        'E_gene_plus_corr_lr':  ('logistic', gene_features + ['corr_edge']),
        'F_gene_only_gbdt':     ('gbdt', gene_features),
        'G_gene_plus_attn_gbdt': ('gbdt', gene_features + ['attn_edge']),
        'H_gene_plus_corr_gbdt': ('gbdt', gene_features + ['corr_edge']),
    }

    all_results = {}

    # ── Run each split design ────────────────────────────────────────────
    for split_type in ['cross_perturbation', 'cross_gene', 'joint']:
        print(f"\n{'─'*60}")
        print(f"Split design: {split_type}")
        print(f"{'─'*60}")
        summary = run_split_design(df, model_configs, split_type, N_FOLDS, SEED)

        for name, data in summary.items():
            auroc = data['mean_auroc']
            auprc = data['mean_auprc']
            r10 = data['mean_recall_at_10']
            r50 = data['mean_recall_at_50']
            r100 = data['mean_recall_at_100']
            print(f"  {name:30s}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
                  f"R@10={r10:.4f}  R@50={r50:.4f}  R@100={r100:.4f}  "
                  f"({data['n_folds_valid']} folds)"
                  if auroc is not None else
                  f"  {name:30s}  -- no valid folds --")

        # Compute deltas (LR family)
        gene_lr_auroc = summary['A_gene_only_lr']['mean_auroc'] or 0.5
        gene_lr_auprc = summary['A_gene_only_lr']['mean_auprc'] or 0.0
        for aug in ['D_gene_plus_attn_lr', 'E_gene_plus_corr_lr']:
            aug_auroc = summary[aug]['mean_auroc'] or 0.5
            aug_auprc = summary[aug]['mean_auprc'] or 0.0
            print(f"  LR  Δ({aug} - A_gene_only_lr): "
                  f"AUROC={aug_auroc - gene_lr_auroc:+.4f}  "
                  f"AUPRC={aug_auprc - gene_lr_auprc:+.4f}")

        # Compute deltas (GBDT family)
        gene_gbdt_auroc = summary['F_gene_only_gbdt']['mean_auroc'] or 0.5
        gene_gbdt_auprc = summary['F_gene_only_gbdt']['mean_auprc'] or 0.0
        for aug in ['G_gene_plus_attn_gbdt', 'H_gene_plus_corr_gbdt']:
            aug_auroc = summary[aug]['mean_auroc'] or 0.5
            aug_auprc = summary[aug]['mean_auprc'] or 0.0
            print(f"  GBDT Δ({aug} - F_gene_only_gbdt): "
                  f"AUROC={aug_auroc - gene_gbdt_auroc:+.4f}  "
                  f"AUPRC={aug_auprc - gene_gbdt_auprc:+.4f}")

        all_results[split_type] = summary

        # Save intermediate checkpoint
        with open(os.path.join(OUT_DIR, "metric_robust_results.json"), 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  [checkpoint saved]")

    # ── Bootstrap CIs ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Bootstrap CIs ({N_BOOTSTRAP} iterations, parallel)")
    print(f"{'='*70}")
    boot_results = {}
    n_workers = min(cpu_count(), 4)

    for split_type in ['cross_perturbation', 'cross_gene', 'joint']:
        print(f"\n  Bootstrapping {split_type} ({n_workers} workers)...")
        t_boot = time.time()

        args_list = [
            (SEED + 100 + i, df, model_configs, split_type, N_FOLDS)
            for i in range(N_BOOTSTRAP)
        ]

        with Pool(n_workers) as pool:
            boot_outputs = pool.map(_bootstrap_worker, args_list)

        # Collect per-model bootstrap distributions (AUROC + AUPRC)
        boot_aurocs = {name: [] for name in model_configs}
        boot_auprcs = {name: [] for name in model_configs}
        for bout in boot_outputs:
            for name in model_configs:
                if bout[name]['mean_auroc'] is not None:
                    boot_aurocs[name].append(bout[name]['mean_auroc'])
                if bout[name]['mean_auprc'] is not None:
                    boot_auprcs[name].append(bout[name]['mean_auprc'])

        boot_cis = {}
        for name in model_configs:
            auroc_vals = np.array(boot_aurocs[name])
            auprc_vals = np.array(boot_auprcs[name])
            entry = {}
            if len(auroc_vals) > 0:
                entry['auroc_ci_lo'] = float(np.percentile(auroc_vals, 2.5))
                entry['auroc_ci_hi'] = float(np.percentile(auroc_vals, 97.5))
            else:
                entry['auroc_ci_lo'] = None
                entry['auroc_ci_hi'] = None
            if len(auprc_vals) > 0:
                entry['auprc_ci_lo'] = float(np.percentile(auprc_vals, 2.5))
                entry['auprc_ci_hi'] = float(np.percentile(auprc_vals, 97.5))
            else:
                entry['auprc_ci_lo'] = None
                entry['auprc_ci_hi'] = None
            entry['n_valid'] = len(auroc_vals)
            boot_cis[name] = entry

        # Delta CIs: LR family
        gene_lr_auroc_boots = np.array(boot_aurocs['A_gene_only_lr'])
        gene_lr_auprc_boots = np.array(boot_auprcs['A_gene_only_lr'])
        for aug_name in ['D_gene_plus_attn_lr', 'E_gene_plus_corr_lr']:
            aug_auroc_boots = np.array(boot_aurocs[aug_name])
            aug_auprc_boots = np.array(boot_auprcs[aug_name])
            min_len_auroc = min(len(gene_lr_auroc_boots), len(aug_auroc_boots))
            min_len_auprc = min(len(gene_lr_auprc_boots), len(aug_auprc_boots))
            if min_len_auroc > 0:
                delta_auroc = aug_auroc_boots[:min_len_auroc] - gene_lr_auroc_boots[:min_len_auroc]
                boot_cis[f'delta_auroc_{aug_name}'] = {
                    'mean': float(delta_auroc.mean()),
                    'ci_lo': float(np.percentile(delta_auroc, 2.5)),
                    'ci_hi': float(np.percentile(delta_auroc, 97.5)),
                }
                print(f"    LR ΔAUROC({aug_name}): {delta_auroc.mean():+.4f} "
                      f"[{np.percentile(delta_auroc, 2.5):+.4f}, "
                      f"{np.percentile(delta_auroc, 97.5):+.4f}]")
            if min_len_auprc > 0:
                delta_auprc = aug_auprc_boots[:min_len_auprc] - gene_lr_auprc_boots[:min_len_auprc]
                boot_cis[f'delta_auprc_{aug_name}'] = {
                    'mean': float(delta_auprc.mean()),
                    'ci_lo': float(np.percentile(delta_auprc, 2.5)),
                    'ci_hi': float(np.percentile(delta_auprc, 97.5)),
                }
                print(f"    LR ΔAUPRC({aug_name}): {delta_auprc.mean():+.4f} "
                      f"[{np.percentile(delta_auprc, 2.5):+.4f}, "
                      f"{np.percentile(delta_auprc, 97.5):+.4f}]")

        # Delta CIs: GBDT family
        gene_gbdt_auroc_boots = np.array(boot_aurocs['F_gene_only_gbdt'])
        gene_gbdt_auprc_boots = np.array(boot_auprcs['F_gene_only_gbdt'])
        for aug_name in ['G_gene_plus_attn_gbdt', 'H_gene_plus_corr_gbdt']:
            aug_auroc_boots = np.array(boot_aurocs[aug_name])
            aug_auprc_boots = np.array(boot_auprcs[aug_name])
            min_len_auroc = min(len(gene_gbdt_auroc_boots), len(aug_auroc_boots))
            min_len_auprc = min(len(gene_gbdt_auprc_boots), len(aug_auprc_boots))
            if min_len_auroc > 0:
                delta_auroc = aug_auroc_boots[:min_len_auroc] - gene_gbdt_auroc_boots[:min_len_auroc]
                boot_cis[f'delta_auroc_{aug_name}'] = {
                    'mean': float(delta_auroc.mean()),
                    'ci_lo': float(np.percentile(delta_auroc, 2.5)),
                    'ci_hi': float(np.percentile(delta_auroc, 97.5)),
                }
                print(f"    GBDT ΔAUROC({aug_name}): {delta_auroc.mean():+.4f} "
                      f"[{np.percentile(delta_auroc, 2.5):+.4f}, "
                      f"{np.percentile(delta_auroc, 97.5):+.4f}]")
            if min_len_auprc > 0:
                delta_auprc = aug_auprc_boots[:min_len_auprc] - gene_gbdt_auprc_boots[:min_len_auprc]
                boot_cis[f'delta_auprc_{aug_name}'] = {
                    'mean': float(delta_auprc.mean()),
                    'ci_lo': float(np.percentile(delta_auprc, 2.5)),
                    'ci_hi': float(np.percentile(delta_auprc, 97.5)),
                }
                print(f"    GBDT ΔAUPRC({aug_name}): {delta_auprc.mean():+.4f} "
                      f"[{np.percentile(delta_auprc, 2.5):+.4f}, "
                      f"{np.percentile(delta_auprc, 97.5):+.4f}]")

        boot_results[split_type] = boot_cis
        print(f"  {split_type} bootstrap done in {time.time() - t_boot:.1f}s")

    # ── Combine and save final results ───────────────────────────────────
    final_results = {
        'experiment': 'round11_metric_robust_incremental',
        'n_observations': len(df),
        'n_perturbations': n_perts,
        'n_unique_targets': n_targets,
        'n_positive': int(n_pos),
        'positive_rate': float(n_pos / len(df)),
        'split_results': all_results,
        'bootstrap_cis': boot_results,
        'n_bootstrap': N_BOOTSTRAP,
        'n_folds': N_FOLDS,
        'model_configs': {
            name: {'model_type': mt, 'features': feats}
            for name, (mt, feats) in model_configs.items()
        },
    }

    results_path = os.path.join(OUT_DIR, "metric_robust_results.json")
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # ── Generate 2-row figure (AUROC top, AUPRC bottom, 3 columns) ──────
    print("\nGenerating figure...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    split_labels = {
        'cross_perturbation': 'Cross-Perturbation\n(R9 baseline)',
        'cross_gene': 'Cross-Gene\n(harder)',
        'joint': 'Joint Cross-Gene\n+ Cross-Pert (hardest)',
    }
    model_keys = list(model_configs.keys())
    model_labels = [
        'Gene\nonly\n(LR)',
        'Attn\nonly\n(LR)',
        'Corr\nonly\n(LR)',
        'Gene\n+Attn\n(LR)',
        'Gene\n+Corr\n(LR)',
        'Gene\nonly\n(GBDT)',
        'Gene\n+Attn\n(GBDT)',
        'Gene\n+Corr\n(GBDT)',
    ]
    colors = [
        '#0072B2', '#DD8452', '#009E73', '#D55E00', '#8172B3',
        '#64B5F6', '#E57373', '#81C784',
    ]

    split_types = ['cross_perturbation', 'cross_gene', 'joint']

    for col_idx, split_type in enumerate(split_types):
        summary = all_results[split_type]
        bcis = boot_results[split_type]

        # ── Top row: AUROC ───────────────────────────────────────────
        ax = axes[0, col_idx]
        aurocs = [summary[k]['mean_auroc'] or 0 for k in model_keys]
        ci_los_auroc = [bcis[k].get('auroc_ci_lo') or 0 for k in model_keys]
        ci_his_auroc = [bcis[k].get('auroc_ci_hi') or 0 for k in model_keys]
        errors_lo = [a - c for a, c in zip(aurocs, ci_los_auroc)]
        errors_hi = [c - a for a, c in zip(aurocs, ci_his_auroc)]

        x = np.arange(len(model_keys))
        ax.bar(x, aurocs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.errorbar(x, aurocs, yerr=[errors_lo, errors_hi],
                    fmt='none', ecolor='black', capsize=3, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=7)
        ax.set_ylabel('AUROC' if col_idx == 0 else '')
        ax.set_title(split_labels[split_type], fontsize=10)
        ax.set_ylim(0.45, 1.0)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        for i, v in enumerate(aurocs):
            if v > 0:
                ax.text(i, v + max(errors_hi[i], 0) + 0.01, f'{v:.3f}',
                        ha='center', fontsize=6)

        # ── Bottom row: AUPRC ────────────────────────────────────────
        ax = axes[1, col_idx]
        auprcs = [summary[k]['mean_auprc'] or 0 for k in model_keys]
        ci_los_auprc = [bcis[k].get('auprc_ci_lo') or 0 for k in model_keys]
        ci_his_auprc = [bcis[k].get('auprc_ci_hi') or 0 for k in model_keys]
        errors_lo = [a - c for a, c in zip(auprcs, ci_los_auprc)]
        errors_hi = [c - a for a, c in zip(auprcs, ci_his_auprc)]

        ax.bar(x, auprcs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.errorbar(x, auprcs, yerr=[errors_lo, errors_hi],
                    fmt='none', ecolor='black', capsize=3, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=7)
        ax.set_ylabel('AUPRC' if col_idx == 0 else '')
        ax.set_title('')
        # AUPRC baseline = positive rate
        pos_rate = n_pos / len(df)
        ax.axhline(pos_rate, color='gray', linestyle=':', alpha=0.5,
                    label=f'baseline={pos_rate:.3f}')
        ax.set_ylim(0, max(auprcs) * 1.3 if max(auprcs) > 0 else 1.0)
        if col_idx == 0:
            ax.legend(fontsize=7, loc='upper left')
        for i, v in enumerate(auprcs):
            if v > 0:
                ax.text(i, v + max(errors_hi[i], 0) + 0.005, f'{v:.3f}',
                        ha='center', fontsize=6)

    fig.suptitle('Round 11: Metric-Robust Incremental Value '
                 '(Logistic Regression + GBDT)',
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig_path = os.path.join(FIG_DIR, "fig_metric_robust_incremental.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # ── Summary printout ─────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"METRIC-ROBUST INCREMENTAL VALUE COMPLETE - {elapsed:.1f}s")
    print(f"{'='*70}")

    for split_type in split_types:
        s = all_results[split_type]
        print(f"\n  {split_type}:")
        # LR deltas
        ga_lr = s['A_gene_only_lr']['mean_auroc'] or 0.5
        da_lr = s['D_gene_plus_attn_lr']['mean_auroc'] or 0.5
        gp_lr = s['A_gene_only_lr']['mean_auprc'] or 0.0
        dp_lr = s['D_gene_plus_attn_lr']['mean_auprc'] or 0.0
        print(f"    LR:   gene_only AUROC={ga_lr:.4f}, gene+attn AUROC={da_lr:.4f}, "
              f"ΔAUROC={da_lr-ga_lr:+.4f}")
        print(f"          gene_only AUPRC={gp_lr:.4f}, gene+attn AUPRC={dp_lr:.4f}, "
              f"ΔAUPRC={dp_lr-gp_lr:+.4f}")
        # GBDT deltas
        ga_gb = s['F_gene_only_gbdt']['mean_auroc'] or 0.5
        da_gb = s['G_gene_plus_attn_gbdt']['mean_auroc'] or 0.5
        gp_gb = s['F_gene_only_gbdt']['mean_auprc'] or 0.0
        dp_gb = s['G_gene_plus_attn_gbdt']['mean_auprc'] or 0.0
        print(f"    GBDT: gene_only AUROC={ga_gb:.4f}, gene+attn AUROC={da_gb:.4f}, "
              f"ΔAUROC={da_gb-ga_gb:+.4f}")
        print(f"          gene_only AUPRC={gp_gb:.4f}, gene+attn AUPRC={dp_gb:.4f}, "
              f"ΔAUPRC={dp_gb-gp_gb:+.4f}")


if __name__ == "__main__":
    run()
