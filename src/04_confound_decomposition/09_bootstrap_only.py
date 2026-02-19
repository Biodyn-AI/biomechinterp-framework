#!/usr/bin/env python3
"""
Bootstrap CIs for metric-robust incremental value (LR only, sequential).
GBDT point estimates already computed; bootstrap CIs use LR for speed.
"""
import os, sys, json, pickle, warnings, time
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
EXTRACTION_DIR = os.path.join(PAPER_DIR, "experiments", "round9_unified_extraction")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round11_metric_robust_incremental")
PRIMARY_LAYER = 12
N_FOLDS = 5
N_BOOTSTRAP = 200
SEED = 42


def evaluate_cv_lr(X, y, train_idx, test_idx):
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
    return roc_auc_score(y_test, probs), average_precision_score(y_test, probs)


def run_one_bootstrap(df, X_gene, X_attn, X_corr, y, split_type, seed):
    """Single bootstrap: resample perturbations, run LR CV, return deltas."""
    rng = np.random.RandomState(seed)
    unique_perts = df['perturbation'].unique()
    boot_perts = rng.choice(unique_perts, size=len(unique_perts), replace=True)

    # Build bootstrap index
    boot_indices = []
    new_pert_labels = []
    for i, pg in enumerate(boot_perts):
        idx = np.where(df['perturbation'].values == pg)[0]
        boot_indices.append(idx)
        new_pert_labels.extend([f"{pg}_{i}"] * len(idx))
    boot_idx = np.concatenate(boot_indices)

    boot_X_gene = X_gene[boot_idx]
    boot_X_attn = X_attn[boot_idx]
    boot_X_corr = X_corr[boot_idx]
    boot_y = y[boot_idx]
    boot_perts_arr = np.array(new_pert_labels)
    boot_targets = df['target'].values[boot_idx]

    n = len(boot_idx)

    # Generate splits
    if split_type == 'cross_perturbation':
        gkf = GroupKFold(n_splits=N_FOLDS)
        splits = list(gkf.split(np.arange(n), groups=boot_perts_arr))
    elif split_type == 'cross_gene':
        gkf = GroupKFold(n_splits=N_FOLDS)
        splits = list(gkf.split(np.arange(n), groups=boot_targets))
    elif split_type == 'joint':
        rng2 = np.random.RandomState(seed)
        up = np.unique(boot_perts_arr)
        ug = np.unique(boot_targets)
        rng2.shuffle(up)
        rng2.shuffle(ug)
        pf = np.array_split(up, N_FOLDS)
        gf = np.array_split(ug, N_FOLDS)
        splits = []
        for k in range(N_FOLDS):
            tp = set(pf[k])
            tg = set(gf[k])
            test_mask = np.array([boot_perts_arr[i] in tp and boot_targets[i] in tg for i in range(n)])
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            if len(test_idx) > 0 and len(train_idx) > 0:
                splits.append((train_idx, test_idx))

    # Run 3 LR models across folds
    results = {'gene': [], 'gene_attn': [], 'gene_corr': []}
    auprc_results = {'gene': [], 'gene_attn': [], 'gene_corr': []}

    for train_idx, test_idx in splits:
        r = evaluate_cv_lr(boot_X_gene, boot_y, train_idx, test_idx)
        if r:
            results['gene'].append(r[0])
            auprc_results['gene'].append(r[1])

        r = evaluate_cv_lr(boot_X_attn, boot_y, train_idx, test_idx)
        if r:
            results['gene_attn'].append(r[0])
            auprc_results['gene_attn'].append(r[1])

        r = evaluate_cv_lr(boot_X_corr, boot_y, train_idx, test_idx)
        if r:
            results['gene_corr'].append(r[0])
            auprc_results['gene_corr'].append(r[1])

    def mean_or_none(lst):
        return float(np.mean(lst)) if lst else None

    return {
        'gene_auroc': mean_or_none(results['gene']),
        'gene_attn_auroc': mean_or_none(results['gene_attn']),
        'gene_corr_auroc': mean_or_none(results['gene_corr']),
        'gene_auprc': mean_or_none(auprc_results['gene']),
        'gene_attn_auprc': mean_or_none(auprc_results['gene_attn']),
        'gene_corr_auprc': mean_or_none(auprc_results['gene_corr']),
    }


def build_dataset(attn_edges, corr_matrix, de_labels, gene_features_df, hvg_genes, hvg_to_local):
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
                'perturbation': pg, 'target': tgt_gene, 'is_de': int(is_de),
                'attn_edge': attn_score, 'corr_edge': corr_score,
                'tgt_mean_expr': tgt_feats.get('mean_expr', 0),
                'tgt_variance': tgt_feats.get('variance', 0),
                'tgt_dropout': tgt_feats.get('dropout_rate', 0),
                'tgt_n_affected': tgt_feats.get('n_perturbations_affected', 0),
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print("=" * 70)
    print("BOOTSTRAP CIs FOR INCREMENTAL VALUE (LR ONLY, SEQUENTIAL)")
    print("=" * 70)
    t0 = time.time()

    print("\nLoading data...")
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

    print("Building dataset...")
    df = build_dataset(attn_edges, corr_matrix, de_labels, gene_features_df, hvg_genes, hvg_to_local)
    print(f"  {len(df)} rows, {df['perturbation'].nunique()} perturbations")

    gene_features = ['tgt_mean_expr', 'tgt_variance', 'tgt_dropout', 'tgt_n_affected']

    # Precompute feature matrices (avoids repeated DataFrame indexing)
    X_gene = df[gene_features].values
    X_attn = df[gene_features + ['attn_edge']].values
    X_corr = df[gene_features + ['corr_edge']].values
    y = df['is_de'].values

    boot_all = {}
    checkpoint_path = os.path.join(OUT_DIR, "bootstrap_cis.json")

    for split_type in ['cross_perturbation', 'cross_gene', 'joint']:
        print(f"\n  Bootstrapping {split_type} ({N_BOOTSTRAP} iters, sequential)...")
        t_split = time.time()

        boot_gene_auroc, boot_attn_auroc, boot_corr_auroc = [], [], []
        boot_gene_auprc, boot_attn_auprc, boot_corr_auprc = [], [], []

        for i in range(N_BOOTSTRAP):
            r = run_one_bootstrap(df, X_gene, X_attn, X_corr, y,
                                  split_type, SEED + 100 + i)
            if r['gene_auroc'] is not None:
                boot_gene_auroc.append(r['gene_auroc'])
            if r['gene_attn_auroc'] is not None:
                boot_attn_auroc.append(r['gene_attn_auroc'])
            if r['gene_corr_auroc'] is not None:
                boot_corr_auroc.append(r['gene_corr_auroc'])
            if r['gene_auprc'] is not None:
                boot_gene_auprc.append(r['gene_auprc'])
            if r['gene_attn_auprc'] is not None:
                boot_attn_auprc.append(r['gene_attn_auprc'])
            if r['gene_corr_auprc'] is not None:
                boot_corr_auprc.append(r['gene_corr_auprc'])

            if (i + 1) % 20 == 0:
                elapsed = time.time() - t_split
                rate = (i + 1) / elapsed
                eta = (N_BOOTSTRAP - i - 1) / rate
                print(f"    {i+1}/{N_BOOTSTRAP}, rate={rate:.2f}/s, ETA={eta:.0f}s")

        # Compute delta CIs
        gene_a = np.array(boot_gene_auroc)
        attn_a = np.array(boot_attn_auroc)
        corr_a = np.array(boot_corr_auroc)
        gene_p = np.array(boot_gene_auprc)
        attn_p = np.array(boot_attn_auprc)
        corr_p = np.array(boot_corr_auprc)

        mn = min(len(gene_a), len(attn_a), len(corr_a))
        da_attn = attn_a[:mn] - gene_a[:mn]
        da_corr = corr_a[:mn] - gene_a[:mn]

        mn2 = min(len(gene_p), len(attn_p), len(corr_p))
        dp_attn = attn_p[:mn2] - gene_p[:mn2]
        dp_corr = corr_p[:mn2] - gene_p[:mn2]

        split_cis = {
            'delta_auroc_attn': {
                'mean': float(da_attn.mean()),
                'ci_lo': float(np.percentile(da_attn, 2.5)),
                'ci_hi': float(np.percentile(da_attn, 97.5)),
            },
            'delta_auroc_corr': {
                'mean': float(da_corr.mean()),
                'ci_lo': float(np.percentile(da_corr, 2.5)),
                'ci_hi': float(np.percentile(da_corr, 97.5)),
            },
            'delta_auprc_attn': {
                'mean': float(dp_attn.mean()),
                'ci_lo': float(np.percentile(dp_attn, 2.5)),
                'ci_hi': float(np.percentile(dp_attn, 97.5)),
            },
            'delta_auprc_corr': {
                'mean': float(dp_corr.mean()),
                'ci_lo': float(np.percentile(dp_corr, 2.5)),
                'ci_hi': float(np.percentile(dp_corr, 97.5)),
            },
            'n_valid': mn,
        }

        print(f"    LR ΔAUROC(attn): {da_attn.mean():+.4f} [{np.percentile(da_attn, 2.5):+.4f}, {np.percentile(da_attn, 97.5):+.4f}]")
        print(f"    LR ΔAUROC(corr): {da_corr.mean():+.4f} [{np.percentile(da_corr, 2.5):+.4f}, {np.percentile(da_corr, 97.5):+.4f}]")
        print(f"    LR ΔAUPRC(attn): {dp_attn.mean():+.4f} [{np.percentile(dp_attn, 2.5):+.4f}, {np.percentile(dp_attn, 97.5):+.4f}]")
        print(f"    LR ΔAUPRC(corr): {dp_corr.mean():+.4f} [{np.percentile(dp_corr, 2.5):+.4f}, {np.percentile(dp_corr, 97.5):+.4f}]")

        elapsed = time.time() - t_split
        print(f"  {split_type} done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

        boot_all[split_type] = split_cis
        with open(checkpoint_path, 'w') as f:
            json.dump(boot_all, f, indent=2)

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"BOOTSTRAP COMPLETE - Total: {total:.0f}s ({total/60:.1f} min)")
    print(f"{'='*70}")
    print(f"Saved to: {checkpoint_path}")
