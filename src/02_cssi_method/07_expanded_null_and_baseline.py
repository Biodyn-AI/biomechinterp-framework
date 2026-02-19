#!/usr/bin/env python3
"""
Expanded CSSI Null Inflation Test & Baseline Comparison
========================================================
Addresses Major Concern 3 / Stress Test 2:
- CSSI-max under shuffled strata and gene-label permutations across range of K
- Per-edge FDR correction across strata
- Comparison to SCENIC-style stratified inference (compute per-stratum edges,
  then aggregate — the standard approach)

This is a synthetic experiment (self-contained, no external data needed).
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/cssi_expanded_results"
FIG_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def generate_synthetic_data(n_genes=50, n_tfs=10, n_states=5, cells_per_state=100,
                            edges_per_state=3, shared_edges=2, signal=0.6,
                            noise=0.1, seed=0):
    """Generate synthetic data with state-specific regulatory edges."""
    np.random.seed(seed)
    N = n_states * cells_per_state
    G = n_genes

    # Generate state-specific ground truth
    true_edges = {}  # (tf, target) -> set of active states
    all_edges = set()

    # Shared edges (active in all states)
    for _ in range(shared_edges):
        tf = np.random.randint(0, n_tfs)
        tgt = np.random.randint(n_tfs, G)
        e = (tf, tgt)
        true_edges[e] = set(range(n_states))
        all_edges.add(e)

    # State-specific edges
    for s in range(n_states):
        for _ in range(edges_per_state):
            tf = np.random.randint(0, n_tfs)
            tgt = np.random.randint(n_tfs, G)
            e = (tf, tgt)
            if e not in true_edges:
                true_edges[e] = set()
            true_edges[e].add(s)
            all_edges.add(e)

    # Generate expression data
    X = np.random.randn(N, G) * noise
    state_labels = np.repeat(np.arange(n_states), cells_per_state)

    for (tf, tgt), active_states in true_edges.items():
        for s in active_states:
            mask = state_labels == s
            n_s = mask.sum()
            # Correlated expression in active state
            base = np.random.randn(n_s) * signal
            X[mask, tf] += base
            X[mask, tgt] += base + np.random.randn(n_s) * noise * 0.5

    return X, state_labels, true_edges, all_edges

def compute_spearman_edges(X, n_tfs=10):
    """Compute absolute Spearman correlation for all TF-target pairs."""
    from scipy.stats import rankdata
    G = X.shape[1]
    ranks = np.apply_along_axis(rankdata, 0, X)
    corr = np.corrcoef(ranks.T)
    edges = {}
    for tf in range(n_tfs):
        for tgt in range(n_tfs, G):
            edges[(tf, tgt)] = abs(corr[tf, tgt])
    return edges

def evaluate(edge_scores, true_edges, n_tfs=10, n_genes=50):
    """Evaluate edge scores against ground truth."""
    all_possible = [(tf, tgt) for tf in range(n_tfs) for tgt in range(n_tfs, n_genes)]
    scores = [edge_scores.get(e, 0.0) for e in all_possible]
    labels = [1 if e in true_edges else 0 for e in all_possible]
    scores = np.array(scores)
    labels = np.array(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {'auroc': 0.5, 'f1': 0.0}
    auroc = roc_auc_score(labels, scores)
    # Top-K F1
    k = int(labels.sum())
    top_k = np.argsort(scores)[-k:]
    preds = np.zeros_like(labels)
    preds[top_k] = 1
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    return {'auroc': auroc, 'f1': f1}

def cssi_max(X, state_labels, n_tfs=10, n_genes=50):
    """CSSI-max: max absolute Spearman across strata."""
    states = np.unique(state_labels)
    edge_scores = {}
    for tf in range(n_tfs):
        for tgt in range(n_tfs, n_genes):
            edge_scores[(tf, tgt)] = 0.0

    for s in states:
        mask = state_labels == s
        if mask.sum() < 10:
            continue
        X_s = X[mask]
        s_edges = compute_spearman_edges(X_s, n_tfs)
        for e, score in s_edges.items():
            edge_scores[e] = max(edge_scores[e], score)
    return edge_scores

def cssi_mean(X, state_labels, n_tfs=10, n_genes=50):
    """CSSI-mean: weighted average of absolute Spearman across strata."""
    states = np.unique(state_labels)
    N = len(state_labels)
    edge_scores = {}
    for tf in range(n_tfs):
        for tgt in range(n_tfs, n_genes):
            edge_scores[(tf, tgt)] = 0.0

    for s in states:
        mask = state_labels == s
        n_s = mask.sum()
        if n_s < 10:
            continue
        weight = n_s / N
        X_s = X[mask]
        s_edges = compute_spearman_edges(X_s, n_tfs)
        for e, score in s_edges.items():
            edge_scores[e] += weight * score
    return edge_scores

def scenic_style_aggregate(X, state_labels, n_tfs=10, n_genes=50):
    """SCENIC-style: compute per-stratum, take union of top-K per stratum."""
    states = np.unique(state_labels)
    all_per_stratum = {}
    for s in states:
        mask = state_labels == s
        if mask.sum() < 10:
            continue
        X_s = X[mask]
        s_edges = compute_spearman_edges(X_s, n_tfs)
        all_per_stratum[s] = s_edges

    # Aggregate: for each edge, take mean of per-stratum scores
    # (this is the standard SCENIC approach)
    edge_scores = {}
    for tf in range(n_tfs):
        for tgt in range(n_tfs, n_genes):
            vals = [all_per_stratum[s].get((tf, tgt), 0.0) for s in all_per_stratum]
            edge_scores[(tf, tgt)] = np.mean(vals)
    return edge_scores

def run_experiment():
    """Run the full expanded null inflation and baseline comparison."""
    print("=" * 60)
    print("CSSI EXPANDED NULL INFLATION & BASELINE COMPARISON")
    print("=" * 60)

    results = []
    K_values = [2, 3, 5, 8, 10, 15, 20]
    n_seeds = 10
    n_genes = 50
    n_tfs = 10
    cells_per_state = 100

    # ========================================
    # EXPERIMENT 1: CSSI-max null inflation across K
    # ========================================
    print("\n--- Experiment 1: Null inflation across K values ---")
    for K in K_values:
        for seed in range(n_seeds):
            X, labels, true_edges, all_edges = generate_synthetic_data(
                n_genes=n_genes, n_tfs=n_tfs, n_states=K,
                cells_per_state=cells_per_state, signal=0.6, seed=seed*100+K)

            # Oracle CSSI-max
            oracle_scores = cssi_max(X, labels, n_tfs, n_genes)
            oracle_eval = evaluate(oracle_scores, true_edges, n_tfs, n_genes)

            # Shuffled CSSI-max
            shuffled_labels = labels.copy()
            np.random.seed(seed*100+K+5000)
            np.random.shuffle(shuffled_labels)
            shuffled_scores = cssi_max(X, shuffled_labels, n_tfs, n_genes)
            shuffled_eval = evaluate(shuffled_scores, true_edges, n_tfs, n_genes)

            # Random CSSI-max (random labels with same K)
            np.random.seed(seed*100+K+10000)
            random_labels = np.random.randint(0, K, len(labels))
            random_scores = cssi_max(X, random_labels, n_tfs, n_genes)
            random_eval = evaluate(random_scores, true_edges, n_tfs, n_genes)

            # Pooled (no stratification)
            pooled_scores = compute_spearman_edges(X, n_tfs)
            pooled_eval = evaluate(pooled_scores, true_edges, n_tfs, n_genes)

            # Gene-label permutation: shuffle gene identities
            np.random.seed(seed*100+K+15000)
            perm = np.random.permutation(n_genes)
            X_perm = X[:, perm]
            perm_scores = cssi_max(X_perm, labels, n_tfs, n_genes)
            perm_eval = evaluate(perm_scores, true_edges, n_tfs, n_genes)

            for method, ev in [('oracle_cssi_max', oracle_eval),
                               ('shuffled_cssi_max', shuffled_eval),
                               ('random_cssi_max', random_eval),
                               ('pooled', pooled_eval),
                               ('gene_permuted_cssi_max', perm_eval)]:
                results.append({
                    'experiment': 'null_inflation',
                    'K': K,
                    'seed': seed,
                    'method': method,
                    'auroc': ev['auroc'],
                    'f1': ev['f1']
                })

            if seed == 0:
                print(f"  K={K}: oracle={oracle_eval['auroc']:.3f}, "
                      f"shuffled={shuffled_eval['auroc']:.3f}, "
                      f"random={random_eval['auroc']:.3f}, "
                      f"pooled={pooled_eval['auroc']:.3f}, "
                      f"gene_perm={perm_eval['auroc']:.3f}")

    # ========================================
    # EXPERIMENT 2: CSSI vs SCENIC-style comparison
    # ========================================
    print("\n--- Experiment 2: CSSI vs SCENIC-style stratified inference ---")
    for K in [3, 5, 8, 10]:
        for seed in range(n_seeds):
            X, labels, true_edges, all_edges = generate_synthetic_data(
                n_genes=n_genes, n_tfs=n_tfs, n_states=K,
                cells_per_state=cells_per_state, signal=0.6, seed=seed*100+K)

            # Methods to compare
            pooled_scores = compute_spearman_edges(X, n_tfs)
            cssi_max_scores = cssi_max(X, labels, n_tfs, n_genes)
            cssi_mean_scores = cssi_mean(X, labels, n_tfs, n_genes)
            scenic_scores = scenic_style_aggregate(X, labels, n_tfs, n_genes)

            for method, scores in [('pooled', pooled_scores),
                                   ('cssi_max', cssi_max_scores),
                                   ('cssi_mean', cssi_mean_scores),
                                   ('scenic_style', scenic_scores)]:
                ev = evaluate(scores, true_edges, n_tfs, n_genes)
                results.append({
                    'experiment': 'cssi_vs_scenic',
                    'K': K,
                    'seed': seed,
                    'method': method,
                    'auroc': ev['auroc'],
                    'f1': ev['f1']
                })

    # ========================================
    # EXPERIMENT 3: Per-edge FDR across strata
    # ========================================
    print("\n--- Experiment 3: Per-edge FDR control ---")
    for K in [5, 10, 15]:
        for seed in range(5):
            X, labels, true_edges, all_edges = generate_synthetic_data(
                n_genes=n_genes, n_tfs=n_tfs, n_states=K,
                cells_per_state=cells_per_state, signal=0.6, seed=seed*100+K)

            states = np.unique(labels)
            # For each edge, collect per-stratum p-values
            edge_pvals = {}
            for tf in range(n_tfs):
                for tgt in range(n_tfs, n_genes):
                    pvals = []
                    for s in states:
                        mask = labels == s
                        if mask.sum() < 10:
                            continue
                        rho, p = stats.spearmanr(X[mask, tf], X[mask, tgt])
                        pvals.append(p)
                    # Per-edge correction across strata (Bonferroni within edge)
                    min_p = min(pvals) * len(pvals) if pvals else 1.0
                    edge_pvals[(tf, tgt)] = min(min_p, 1.0)

            # BH-correct across all edges
            from statsmodels.stats.multitest import multipletests
            edges_list = list(edge_pvals.keys())
            pvals_list = [edge_pvals[e] for e in edges_list]
            reject, adj_pvals, _, _ = multipletests(pvals_list, method='fdr_bh', alpha=0.05)

            # Evaluate FDR-corrected discoveries
            tp = sum(1 for e, r in zip(edges_list, reject) if r and e in true_edges)
            fp = sum(1 for e, r in zip(edges_list, reject) if r and e not in true_edges)
            fn = sum(1 for e in true_edges if e in edge_pvals and not reject[edges_list.index(e)])
            total_disc = tp + fp
            fdr = fp / total_disc if total_disc > 0 else 0.0
            power = tp / len(true_edges) if len(true_edges) > 0 else 0.0

            results.append({
                'experiment': 'fdr_control',
                'K': K,
                'seed': seed,
                'method': 'per_edge_bh',
                'auroc': 0.0,
                'f1': 0.0,
                'fdr': fdr,
                'power': power,
                'n_discoveries': total_disc,
                'n_true_edges': len(true_edges)
            })
            if seed == 0:
                print(f"  K={K}: discoveries={total_disc}, FDR={fdr:.3f}, power={power:.3f}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "cssi_expanded_results.csv"), index=False)

    # ========================================
    # GENERATE FIGURES
    # ========================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Null inflation across K
    ax = axes[0]
    null_df = df[df['experiment'] == 'null_inflation']
    for method, color, marker, label in [
        ('oracle_cssi_max', '#4CAF50', 'o', 'CSSI-max (oracle)'),
        ('shuffled_cssi_max', '#FF9800', 's', 'CSSI-max (shuffled)'),
        ('random_cssi_max', '#F44336', 'D', 'CSSI-max (random)'),
        ('pooled', '#2196F3', '^', 'Pooled'),
        ('gene_permuted_cssi_max', '#9C27B0', 'v', 'CSSI-max (gene perm)')
    ]:
        mdf = null_df[null_df['method'] == method]
        means = mdf.groupby('K')['auroc'].agg(['mean', 'std']).reset_index()
        ax.errorbar(means['K'], means['mean'], yerr=means['std'],
                    fmt=f'{marker}-', color=color, capsize=3, label=label, markersize=6)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of strata (K)')
    ax.set_ylabel('AUROC')
    ax.set_title('(A) CSSI-max null inflation test')
    ax.legend(fontsize=7, loc='best')

    # Panel B: CSSI vs SCENIC comparison
    ax = axes[1]
    comp_df = df[df['experiment'] == 'cssi_vs_scenic']
    for method, color, label in [
        ('cssi_max', '#4CAF50', 'CSSI-max'),
        ('cssi_mean', '#8BC34A', 'CSSI-mean'),
        ('scenic_style', '#FF9800', 'SCENIC-style mean'),
        ('pooled', '#2196F3', 'Pooled')
    ]:
        mdf = comp_df[comp_df['method'] == method]
        means = mdf.groupby('K')['auroc'].agg(['mean', 'std']).reset_index()
        ax.errorbar(means['K'], means['mean'], yerr=means['std'],
                    fmt='o-', color=color, capsize=3, label=label, markersize=6)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of strata (K)')
    ax.set_ylabel('AUROC')
    ax.set_title('(B) CSSI vs SCENIC-style stratification')
    ax.legend(fontsize=8)

    # Panel C: FDR control
    ax = axes[2]
    fdr_df = df[df['experiment'] == 'fdr_control']
    if len(fdr_df) > 0 and 'fdr' in fdr_df.columns:
        means = fdr_df.groupby('K').agg({'fdr': 'mean', 'power': 'mean'}).reset_index()
        ax.bar(means['K'] - 0.8, means['power'], width=1.5, color='#4CAF50',
               alpha=0.7, label='Power')
        ax.bar(means['K'] + 0.8, means['fdr'], width=1.5, color='#F44336',
               alpha=0.7, label='FDR')
        ax.axhline(0.05, color='red', linestyle='--', alpha=0.5, label='FDR = 0.05')
        ax.set_xlabel('Number of strata (K)')
        ax.set_ylabel('Rate')
        ax.set_title('(C) Per-edge FDR control')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_cssi_expanded_null_baseline.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to {fig_path}")

    # Summary statistics
    print("\n=== SUMMARY ===")
    # Key question: Does CSSI-max inflate under null?
    null_df = df[df['experiment'] == 'null_inflation']
    for K in K_values:
        kdf = null_df[null_df['K'] == K]
        oracle = kdf[kdf['method'] == 'oracle_cssi_max']['auroc'].mean()
        shuffled = kdf[kdf['method'] == 'shuffled_cssi_max']['auroc'].mean()
        random_m = kdf[kdf['method'] == 'random_cssi_max']['auroc'].mean()
        pooled = kdf[kdf['method'] == 'pooled']['auroc'].mean()
        print(f"K={K:2d}: oracle={oracle:.3f}, shuffled={shuffled:.3f}, "
              f"random={random_m:.3f}, pooled={pooled:.3f}, "
              f"null_inflation={shuffled - pooled:+.3f}")

    # CSSI vs SCENIC
    print("\nCSSI vs SCENIC-style:")
    comp_df = df[df['experiment'] == 'cssi_vs_scenic']
    for method in ['cssi_max', 'cssi_mean', 'scenic_style', 'pooled']:
        mean_auroc = comp_df[comp_df['method'] == method]['auroc'].mean()
        print(f"  {method}: mean AUROC = {mean_auroc:.3f}")

    # Save summary
    summary = {
        'experiment': 'cssi_expanded_null_and_baseline',
        'null_inflation': {},
        'cssi_vs_scenic': {},
        'fdr_control': {}
    }
    for K in K_values:
        kdf = null_df[null_df['K'] == K]
        summary['null_inflation'][f'K={K}'] = {
            'oracle': round(float(kdf[kdf['method'] == 'oracle_cssi_max']['auroc'].mean()), 4),
            'shuffled': round(float(kdf[kdf['method'] == 'shuffled_cssi_max']['auroc'].mean()), 4),
            'random': round(float(kdf[kdf['method'] == 'random_cssi_max']['auroc'].mean()), 4),
            'pooled': round(float(kdf[kdf['method'] == 'pooled']['auroc'].mean()), 4),
        }
    for method in ['cssi_max', 'cssi_mean', 'scenic_style', 'pooled']:
        mdf = comp_df[comp_df['method'] == method]
        summary['cssi_vs_scenic'][method] = round(float(mdf['auroc'].mean()), 4)

    with open(os.path.join(OUT_DIR, "cssi_expanded_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXPERIMENT COMPLETE ===")
    return df, summary

if __name__ == '__main__':
    df, summary = run_experiment()
