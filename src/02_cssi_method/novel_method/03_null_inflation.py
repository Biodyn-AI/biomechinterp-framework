#!/usr/bin/env python3
"""
CSSI-max Null / Multiple-Comparisons Stress Test
================================================

This experiment quantifies how CSSI-max behaves under:
1) Oracle strata labels (upper bound).
2) Shuffled strata labels (breaks alignment between strata and true states).
3) Random strata labels (purely spurious partitioning).

Motivation:
- CSSI-max is an implicit search over K strata per edge. If K is large,
  some apparent gains can arise from searching over many partitions.

Outputs:
- novel_method/scaling_v2_null_results.csv
- figures/fig_cssi_null_inflation.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = Path(__file__).resolve().parent / "scaling_v2_null_results.csv"
OUT_FIG = ROOT / "figures" / "fig_cssi_null_inflation.png"


def gen(
    n_genes: int,
    n_tfs: int,
    n_states: int,
    cps: int,
    eps: int,
    shared: int,
    sig: float,
    seed: int,
):
    """Generate state-specific GRN signal in expression.

    Each state has its own active TF->target edges plus `shared` edges
    that are active in every state. Pooling across states dilutes
    state-specific edges.
    """
    rng = np.random.RandomState(seed)
    n = cps * n_states

    tfs = list(range(n_tfs))
    tgts = list(range(n_tfs, n_genes))
    all_edges = [(t, g) for t in tfs for g in tgts]
    rng.shuffle(all_edges)

    shared_edges = set(all_edges[:shared])
    rest = all_edges[shared:]
    state_edges = []
    idx = 0
    for _ in range(n_states):
        state_edges.append(set(rest[idx : idx + eps]) | shared_edges)
        idx += eps

    true_edges = set()
    for s in state_edges:
        true_edges |= s

    x = rng.randn(n, n_genes)
    labels = np.repeat(np.arange(n_states), cps)

    start = 0
    for s in range(n_states):
        ns = cps
        for tf, tgt in state_edges[s]:
            x[start : start + ns, tgt] += sig * x[start : start + ns, tf]
        start += ns

    return x, labels.astype(int), true_edges, tfs, tgts


def corr_abs_spearman(x: np.ndarray, tfs: list[int], tgts: list[int]) -> np.ndarray:
    """Fast absolute Spearman correlation matrix between TFs and targets."""
    ranks = np.apply_along_axis(stats.rankdata, 0, x)
    a = ranks[:, tfs]
    b = ranks[:, tgts]
    a = (a - a.mean(0)) / (a.std(0) + 1e-12)
    b = (b - b.mean(0)) / (b.std(0) + 1e-12)
    return np.abs(a.T @ b / len(x))


def to_edges(score: np.ndarray, tfs: list[int], tgts: list[int]):
    edges = [(tfs[i], tgts[j], float(score[i, j])) for i in range(len(tfs)) for j in range(len(tgts))]
    edges.sort(key=lambda x: -x[2])
    return edges


def eval_edges(pred, true_edges: set[tuple[int, int]], tfs: list[int], tgts: list[int]):
    # Top-k uses k = |true_edges| for comparability across configs.
    k = len(true_edges)
    topk = set((e[0], e[1]) for e in pred[:k])
    tp = len(topk & true_edges)
    precision = tp / k if k else 0.0
    recall = tp / len(true_edges) if true_edges else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    all_edges = [(t, g) for t in tfs for g in tgts]
    score_map = {(e[0], e[1]): e[2] for e in pred}
    y_true = np.array([1 if e in true_edges else 0 for e in all_edges], dtype=int)
    y_score = np.array([score_map.get(e, 0.0) for e in all_edges], dtype=float)
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = 0.5
    return f1, auc


def pooled_edges(x: np.ndarray, tfs: list[int], tgts: list[int]):
    return to_edges(corr_abs_spearman(x, tfs, tgts), tfs, tgts)


def cssi_edges(x: np.ndarray, labels: np.ndarray, tfs: list[int], tgts: list[int], mode: str):
    states = np.unique(labels)
    n = len(labels)
    agg = np.zeros((len(tfs), len(tgts)), dtype=float)
    for s in states:
        mask = labels == s
        if mask.sum() < 5:
            continue
        c = corr_abs_spearman(x[mask], tfs, tgts)
        if mode == "max":
            agg = np.maximum(agg, c)
        elif mode == "mean":
            agg += (mask.sum() / n) * c
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return to_edges(agg, tfs, tgts)


def _shuffle_labels(labels: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    return shuffled


def _random_labels_like(labels: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    # Preserve state counts but randomize assignment.
    return _shuffle_labels(labels, rng)


def main() -> None:
    np.random.seed(0)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    # Match the CSSI v2 experiment configs used for the main CSSI scaling plot.
    configs = [
        (2, 100, "N=200, 2 states"),
        (4, 100, "N=400, 4 states"),
        (6, 100, "N=600, 6 states"),
        (8, 125, "N=1000, 8 states"),
        (10, 100, "N=1000, 10 states"),
        (12, 125, "N=1500, 12 states"),
    ]

    rows = []
    for n_states, cps, desc in configs:
        for seed in range(50):
            # Use the same edge-density rule as cssi_experiment_v2.py.
            eps = max(2, 12 // n_states)
            x, labels, true_edges, tfs, tgts = gen(
                n_genes=30,
                n_tfs=5,
                n_states=n_states,
                cps=cps,
                eps=eps,
                shared=2,
                sig=0.5,
                seed=seed * 100 + n_states,
            )

            rng = np.random.RandomState(seed * 1000 + n_states)
            labels_shuf = _shuffle_labels(labels, rng)
            labels_rand = _random_labels_like(labels, rng)

            methods = [
                ("pooled", None, pooled_edges(x, tfs, tgts)),
                ("cssi_mean_oracle", None, cssi_edges(x, labels, tfs, tgts, mode="mean")),
                ("cssi_max_oracle", None, cssi_edges(x, labels, tfs, tgts, mode="max")),
                ("cssi_max_shuffled", None, cssi_edges(x, labels_shuf, tfs, tgts, mode="max")),
                ("cssi_max_random", None, cssi_edges(x, labels_rand, tfs, tgts, mode="max")),
            ]

            for method, _, pred in methods:
                f1, auc = eval_edges(pred, true_edges, tfs, tgts)
                rows.append(
                    dict(
                        n_states=n_states,
                        cps=cps,
                        total=cps * n_states,
                        seed=seed,
                        desc=desc,
                        method=method,
                        f1=f1,
                        auc=auc,
                        n_true_edges=len(true_edges),
                    )
                )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # Plot mean +/- std across seeds.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, metric, ylabel in [(axes[0], "f1", "F1"), (axes[1], "auc", "AUROC")]:
        for method, color, marker in [
            ("pooled", "#1f77b4", "o"),
            ("cssi_max_oracle", "#d62728", "s"),
            ("cssi_max_shuffled", "#ff7f0e", "^"),
            ("cssi_max_random", "#9467bd", "x"),
        ]:
            sub = df[df["method"] == method]
            means = sub.groupby("n_states")[metric].mean()
            stds = sub.groupby("n_states")[metric].std()
            ax.errorbar(
                means.index,
                means.values,
                yerr=stds.values,
                label=method,
                color=color,
                marker=marker,
                capsize=3,
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Number of strata (K)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0.0, 1.05)

    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("CSSI-max Null Stress Test (Oracle vs Shuffled/Random Labels)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
