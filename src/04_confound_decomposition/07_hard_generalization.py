#!/usr/bin/env python3
"""
Round 10 Experiment 2: Hard-Generalization Incremental Value
=============================================================
Reruns incremental-value models with cross-gene and joint
cross-gene/cross-perturbation splits.

Addresses: MC3 ("incremental-value analysis needs harder splits")
           ST2 ("hard-generalization incremental-value test")
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
from multiprocessing import Pool, cpu_count

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
EXTRACTION_DIR = os.path.join(PAPER_DIR, "experiments", "round9_unified_extraction")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round10_hard_generalization")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PRIMARY_LAYER = 12
N_FOLDS = 5
N_BOOTSTRAP = 200
SEED = 42


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


def evaluate_cv(df, feature_cols, train_idx, test_idx):
    """Evaluate a model on given train/test indices."""
    X = df[feature_cols].values
    y = df['is_de'].values
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        return None, None
    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        return None, None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_test_s)[:, 1]

    return roc_auc_score(y_test, probs), average_precision_score(y_test, probs)


def run_split_design(df, model_configs, split_type, n_folds=5, seed=42):
    """Run a specific split design and return per-model AUROC/AUPRC."""
    n = len(df)
    results_by_model = {name: {'aurocs': [], 'auprcs': []} for name in model_configs}

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
        for name, feature_cols in model_configs.items():
            auroc, auprc = evaluate_cv(df, feature_cols, train_idx, test_idx)
            if auroc is not None:
                results_by_model[name]['aurocs'].append(auroc)
                results_by_model[name]['auprcs'].append(auprc)

    summary = {}
    for name, data in results_by_model.items():
        aurocs = np.array(data['aurocs'])
        auprcs = np.array(data['auprcs'])
        summary[name] = {
            'mean_auroc': float(aurocs.mean()) if len(aurocs) > 0 else None,
            'std_auroc': float(aurocs.std()) if len(aurocs) > 0 else None,
            'mean_auprc': float(auprcs.mean()) if len(auprcs) > 0 else None,
            'n_folds_valid': len(aurocs),
        }
    return summary


def _bootstrap_worker(args):
    """Bootstrap worker for parallel CI computation."""
    seed, df, model_configs, split_type, n_folds = args
    rng = np.random.RandomState(seed)
    unique_perts = df['perturbation'].unique()
    boot_perts = rng.choice(unique_perts, size=len(unique_perts), replace=True)

    boot_dfs = []
    for i, pg in enumerate(boot_perts):
        sub = df[df['perturbation'] == pg].copy()
        sub = sub.copy()
        sub['perturbation'] = f"{pg}_{i}"
        boot_dfs.append(sub)
    boot_df = pd.concat(boot_dfs, ignore_index=True)

    return run_split_design(boot_df, model_configs, split_type, n_folds, seed)


def run():
    print("=" * 70)
    print("HARD-GENERALIZATION INCREMENTAL VALUE TEST")
    print("=" * 70)
    t0 = time.time()

    # Load data
    print("\nLoading extraction data...")
    with open(os.path.join(EXTRACTION_DIR, "hvg_gene_list.json")) as f:
        hvg_genes = json.load(f)
    hvg_to_local = {g.upper(): i for i, g in enumerate(hvg_genes)}

    with open(os.path.join(EXTRACTION_DIR, "de_labels.pkl"), 'rb') as f:
        de_labels = pickle.load(f)

    corr_matrix = np.load(os.path.join(EXTRACTION_DIR, "correlation_matrix.npy"))
    gene_features_df = pd.read_csv(os.path.join(EXTRACTION_DIR, "gene_features.csv"))

    edge_path = os.path.join(EXTRACTION_DIR, "layer_mean_edges", f"layer_{PRIMARY_LAYER:02d}_edges.pkl")
    with open(edge_path, 'rb') as f:
        attn_edges = pickle.load(f)

    # Build dataset
    print("\nBuilding dataset...")
    df = build_dataset(attn_edges, corr_matrix, de_labels, gene_features_df,
                       hvg_genes, hvg_to_local)
    n_perts = df['perturbation'].nunique()
    n_targets = df['target'].nunique()
    n_pos = df['is_de'].sum()
    print(f"  {len(df)} rows, {n_perts} perturbations, {n_targets} unique targets, "
          f"{n_pos} positive ({100*n_pos/len(df):.1f}%)")

    gene_features = ['tgt_mean_expr', 'tgt_variance', 'tgt_dropout', 'tgt_n_affected']
    model_configs = {
        'A_gene_only': gene_features,
        'B_attn_only': ['attn_edge'],
        'C_corr_only': ['corr_edge'],
        'D_gene_plus_attn': gene_features + ['attn_edge'],
        'E_gene_plus_corr': gene_features + ['corr_edge'],
    }

    all_results = {}

    # Run each split design
    for split_type in ['cross_perturbation', 'cross_gene', 'joint']:
        print(f"\n--- Split design: {split_type} ---")
        summary = run_split_design(df, model_configs, split_type, N_FOLDS, SEED)

        for name, data in summary.items():
            auroc = data['mean_auroc']
            print(f"  {name}: AUROC = {auroc:.4f} ({data['n_folds_valid']} valid folds)")

        # Compute deltas
        gene_auroc = summary['A_gene_only']['mean_auroc'] or 0.5
        for aug_name, base_name in [('D_gene_plus_attn', 'A_gene_only'),
                                     ('E_gene_plus_corr', 'A_gene_only')]:
            aug_auroc = summary[aug_name]['mean_auroc'] or 0.5
            delta = aug_auroc - gene_auroc
            print(f"  Δ({aug_name} - {base_name}) = {delta:+.4f}")

        all_results[split_type] = summary

        # Save intermediate
        with open(os.path.join(OUT_DIR, "hard_generalization_results.json"), 'w') as f:
            json.dump(all_results, f, indent=2)

    # Bootstrap CIs for all splits
    print(f"\nBootstrap CIs ({N_BOOTSTRAP} iterations, parallel)...")
    boot_results = {}

    for split_type in ['cross_perturbation', 'cross_gene', 'joint']:
        print(f"  Bootstrapping {split_type}...")
        args_list = [(SEED + 100 + i, df, model_configs, split_type, N_FOLDS)
                     for i in range(N_BOOTSTRAP)]

        n_workers = min(cpu_count(), 4)
        with Pool(n_workers) as pool:
            boot_outputs = pool.map(_bootstrap_worker, args_list)

        boot_aurocs = {name: [] for name in model_configs}
        for bout in boot_outputs:
            for name in model_configs:
                if bout[name]['mean_auroc'] is not None:
                    boot_aurocs[name].append(bout[name]['mean_auroc'])

        boot_cis = {}
        for name in model_configs:
            vals = np.array(boot_aurocs[name])
            if len(vals) > 0:
                boot_cis[name] = {
                    'ci_lo': float(np.percentile(vals, 2.5)),
                    'ci_hi': float(np.percentile(vals, 97.5)),
                    'n_valid': len(vals),
                }
            else:
                boot_cis[name] = {'ci_lo': None, 'ci_hi': None, 'n_valid': 0}

        # Delta CIs
        gene_boots = np.array(boot_aurocs['A_gene_only'])
        for aug_name in ['D_gene_plus_attn', 'E_gene_plus_corr']:
            aug_boots = np.array(boot_aurocs[aug_name])
            min_len = min(len(gene_boots), len(aug_boots))
            if min_len > 0:
                delta_boots = aug_boots[:min_len] - gene_boots[:min_len]
                boot_cis[f'delta_{aug_name}'] = {
                    'mean': float(delta_boots.mean()),
                    'ci_lo': float(np.percentile(delta_boots, 2.5)),
                    'ci_hi': float(np.percentile(delta_boots, 97.5)),
                }
                print(f"    Δ({aug_name}): {delta_boots.mean():+.4f} "
                      f"[{np.percentile(delta_boots, 2.5):+.4f}, "
                      f"{np.percentile(delta_boots, 97.5):+.4f}]")

        boot_results[split_type] = boot_cis

    # Combine all results
    final_results = {
        'experiment': 'round10_hard_generalization',
        'n_observations': len(df),
        'n_perturbations': n_perts,
        'n_unique_targets': n_targets,
        'n_positive': int(n_pos),
        'positive_rate': float(n_pos / len(df)),
        'split_results': all_results,
        'bootstrap_cis': boot_results,
        'n_bootstrap': N_BOOTSTRAP,
        'n_folds': N_FOLDS,
    }

    with open(os.path.join(OUT_DIR, "hard_generalization_results.json"), 'w') as f:
        json.dump(final_results, f, indent=2)

    # Generate figure
    print("\nGenerating figure...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    split_labels = {
        'cross_perturbation': 'A. Cross-Perturbation\n(R9 baseline)',
        'cross_gene': 'B. Cross-Gene\n(harder)',
        'joint': 'C. Joint Cross-Gene\n+ Cross-Pert (hardest)',
    }
    model_labels = ['Gene\nonly', 'Attn\nonly', 'Corr\nonly', 'Gene\n+Attn', 'Gene\n+Corr']
    model_keys = list(model_configs.keys())
    colors = ['#0072B2', '#DD8452', '#009E73', '#D55E00', '#8172B3']

    for ax_idx, split_type in enumerate(['cross_perturbation', 'cross_gene', 'joint']):
        ax = axes[ax_idx]
        summary = all_results[split_type]
        boot_cis = boot_results[split_type]

        aurocs = [summary[k]['mean_auroc'] or 0 for k in model_keys]
        ci_los = [boot_cis[k]['ci_lo'] or 0 for k in model_keys]
        ci_his = [boot_cis[k]['ci_hi'] or 0 for k in model_keys]
        errors_lo = [a - c for a, c in zip(aurocs, ci_los)]
        errors_hi = [c - a for a, c in zip(aurocs, ci_his)]

        x = np.arange(len(model_keys))
        bars = ax.bar(x, aurocs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.errorbar(x, aurocs, yerr=[errors_lo, errors_hi],
                    fmt='none', ecolor='black', capsize=3, linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=8)
        ax.set_ylabel('AUROC' if ax_idx == 0 else '')
        ax.set_title(split_labels[split_type], fontsize=10)
        ax.set_ylim(0.45, 1.0)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

        # Add value labels
        for i, v in enumerate(aurocs):
            if v > 0:
                ax.text(i, v + max(errors_hi[i], 0) + 0.01, f'{v:.3f}',
                        ha='center', fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_hard_generalization.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"HARD GENERALIZATION COMPLETE - {elapsed:.1f}s")
    print(f"{'='*70}")
    for split_type in ['cross_perturbation', 'cross_gene', 'joint']:
        s = all_results[split_type]
        ga = s['A_gene_only']['mean_auroc']
        da = s['D_gene_plus_attn']['mean_auroc']
        print(f"  {split_type}: gene_only={ga:.4f}, gene+attn={da:.4f}, Δ={da-ga:+.4f}")


if __name__ == "__main__":
    run()
