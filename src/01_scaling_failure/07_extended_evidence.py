#!/usr/bin/env python3
"""Build extended quantitative evidence for scaling failure manuscript.

This script combines:
1) Archived workshop snapshot tables.
2) Fresh rerun metrics parsed from single_cell_mechinterp scaling run logs.

Outputs include analysis tables and additional figures suitable for manuscript use.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SC_ROOT = ROOT / "single_cell_mechinterp"
if str(SC_ROOT) not in sys.path:
    sys.path.insert(0, str(SC_ROOT))

from src.eval.dorothea import load_dorothea
from src.eval.gene_symbols import load_hgnc_alias_map, normalize_edges, normalize_gene_names
from src.eval.metrics import aupr


TIER_ORDER = ["small", "medium", "large"]
CELL_ORDER = [200, 1000, 3000]


def _safe_ratio(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return float(num / den)


def _safe_pct_change(old: float, new: float) -> float:
    if old == 0:
        return float("nan")
    return float((new - old) / old * 100.0)


def _sign_test_one_sided_neg(diffs: Iterable[float]) -> tuple[int, int, float]:
    """Exact one-sided sign test for negative median shift.

    H1: median(diff) < 0
    """
    vals = [x for x in diffs if not math.isnan(x) and x != 0]
    n = len(vals)
    if n == 0:
        return 0, 0, float("nan")
    k_neg = sum(1 for x in vals if x < 0)
    # One-sided p-value for H1: more negatives than expected by chance.
    p = stats.binom.sf(k_neg - 1, n, 0.5)
    return k_neg, n, float(p)


def _jaccard(edges_a: Iterable[tuple[str, str]], edges_b: Iterable[tuple[str, str]]) -> float:
    set_a = set(edges_a)
    set_b = set(edges_b)
    if not set_a and not set_b:
        return float("nan")
    return float(len(set_a & set_b) / len(set_a | set_b))


def _spearman_correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if values_a.size < 2 or values_b.size < 2:
        return float("nan")
    corr, _ = stats.spearmanr(values_a, values_b)
    return float(corr)


def _load_evaluate_rows(run_log: Path) -> pd.DataFrame:
    if not run_log.exists():
        return pd.DataFrame()
    rows = []
    with run_log.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != "success" or row.get("step") != "evaluate":
                continue
            metrics_raw = row.get("metrics", "")
            if not metrics_raw:
                continue
            try:
                metrics = json.loads(metrics_raw)
            except json.JSONDecodeError:
                continue
            rows.append(
                {
                    "timestamp": row.get("timestamp", ""),
                    "experiment_id": row.get("experiment_id", ""),
                    "precision": float(metrics.get("precision", float("nan"))),
                    "recall": float(metrics.get("recall", float("nan"))),
                    "f1": float(metrics.get("f1", float("nan"))),
                }
            )
    return pd.DataFrame(rows)


def _parse_scaling_id(experiment_id: str) -> dict[str, object] | None:
    pattern = re.compile(
        r"^(small|medium|large)_([a-z]+)_cells(\d+)_hvg(\d+)_full_hvg_maxgenes(\d+)_seed(\d+)_topk(\d+)$"
    )
    match = pattern.match(experiment_id)
    if not match:
        return None
    model_tier, tissue, cells, hvg, max_genes, seed, top_k = match.groups()
    return {
        "model_tier": model_tier,
        "tissue": tissue,
        "cells": int(cells),
        "hvg": int(hvg),
        "max_genes": int(max_genes),
        "seed": int(seed),
        "top_k": int(top_k),
    }


def _plot_effect_heatmap(effect_df: pd.DataFrame, out_path: Path) -> None:
    rows = [
        "f1_drop_200_to_1000_pct",
        "f1_drop_1000_to_3000_pct",
        "jaccard_drop_200_to_3000_pct",
        "spearman_delta_200_to_3000",
    ]
    cols = TIER_ORDER
    matrix = np.full((len(rows), len(cols)), np.nan, dtype=float)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            sel = effect_df[(effect_df["metric"] == r) & (effect_df["model_tier"] == c)]
            if not sel.empty:
                matrix[i, j] = float(sel["value"].iloc[0])

    fig, ax = plt.subplots(figsize=(9.8, 5.0), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(
        [
            "F1 % drop (200->1000)",
            "F1 % drop (1000->3000)",
            "Jaccard % drop (200->3000)",
            "Spearman delta (200->3000)",
        ]
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                label = "NA"
            elif abs(v) >= 100 or abs(v) < 0.01:
                label = f"{v:.2e}"
            else:
                label = f"{v:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, shrink=0.9, label="Effect size")
    ax.set_title("Archived workshop effect-size summary")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_repro_vs_snapshot(joined: pd.DataFrame, out_path: Path) -> None:
    if joined.empty:
        return
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for tier, tier_df in joined.groupby("model_tier"):
        ax.scatter(tier_df["f1_snapshot"], tier_df["f1_rerun"], label=tier, s=52)
        for _, row in tier_df.iterrows():
            ax.text(row["f1_snapshot"], row["f1_rerun"], f"{int(row['cells'])}", fontsize=7)
    lo = min(joined["f1_snapshot"].min(), joined["f1_rerun"].min())
    hi = max(joined["f1_snapshot"].max(), joined["f1_rerun"].max())
    lo = max(lo, 1e-8)
    hi = max(hi, lo * 1.1)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Archived snapshot F1")
    ax.set_ylabel("Current rerun F1")
    ax.set_title("Rerun reproducibility vs archived means")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Model tier")
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.15, top=0.9)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_rerun_scaling(rerun_df: pd.DataFrame, out_path: Path) -> None:
    if rerun_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for tier in TIER_ORDER:
        tier_df = rerun_df[rerun_df["model_tier"] == tier].sort_values("cells")
        if tier_df.empty:
            continue
        ax.plot(tier_df["cells"], tier_df["f1"], marker="o", label=tier)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cells (log scale)")
    ax.set_ylabel("Rerun F1 (log scale)")
    ax.set_title("Current-checkout reruns (seed-specific)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Model tier")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_drop_bars(rerun_df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for tier in TIER_ORDER:
        sub = rerun_df[rerun_df["model_tier"] == tier]
        if sub.empty:
            continue
        f200 = sub.loc[sub["cells"] == 200, "f1"]
        f1000 = sub.loc[sub["cells"] == 1000, "f1"]
        if f200.empty or f1000.empty:
            continue
        rows.append(
            {
                "model_tier": tier,
                "pct_drop_200_to_1000": -_safe_pct_change(float(f200.iloc[0]), float(f1000.iloc[0])),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(df["model_tier"], df["pct_drop_200_to_1000"])
    ax.set_ylabel("Percent drop in F1 (200->1000)")
    ax.set_title("Rerun degradation magnitude")
    ax.grid(True, axis="y", alpha=0.25)
    for i, v in enumerate(df["pct_drop_200_to_1000"]):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    if k < 0 or k > n:
        return float("nan"), float("nan")
    lower = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def _normalize_pred_edges_with_score(pred_raw: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    pred = pred_raw.copy()
    pred["source"] = pred["source"].astype(str).str.strip().str.upper().map(lambda value: alias_map.get(value, value))
    pred["target"] = pred["target"].astype(str).str.strip().str.upper().map(lambda value: alias_map.get(value, value))
    pred = pred[(pred["source"] != "") & (pred["target"] != "")]
    pred = pred.sort_values("score", ascending=False)
    # Keep highest-scoring instance when alias normalization collapses multiple raw edges.
    pred = pred.drop_duplicates(subset=["source", "target"], keep="first")
    return pred.reset_index(drop=True)


def _build_rerun_edge_diagnostics(
    root: Path,
    rerun_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rerun_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty

    alias_map = load_hgnc_alias_map(root / "single_cell_mechinterp" / "external" / "hgnc_complete_set.txt")
    truth_edges = load_dorothea(
        root / "single_cell_mechinterp" / "external" / "networks" / "trrust_human.tsv",
        confidence_levels=["A", "B", "C", "D"],
    )
    truth_edges = normalize_edges(truth_edges, alias_map).drop_duplicates()

    rank_k = [100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 150000]

    detail_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    shift_rows: list[dict[str, object]] = []
    true_positive_rows: list[dict[str, object]] = []
    run_edge_sets: dict[str, set[tuple[str, str]]] = {}
    run_score_maps: dict[str, dict[tuple[str, str], float]] = {}

    for _, row in rerun_df.sort_values(["model_tier", "cells"]).iterrows():
        run_id = f"{row['model_tier']}_{int(row['cells'])}"
        exp_id = str(row["experiment_id"])
        run_dir = root / "single_cell_mechinterp" / "outputs" / "scaling" / exp_id
        edge_path = run_dir / "inferred_edges.tsv"
        h5ad_path = run_dir / "tabula_sapiens_processed.h5ad"
        if not edge_path.exists() or not h5ad_path.exists():
            continue

        adata = sc.read_h5ad(h5ad_path)
        gene_names = normalize_gene_names(adata.var_names.values, alias_map)
        gene_set = set(gene_names.tolist())

        pred_raw = pd.read_csv(edge_path, sep="\t")
        n_edges_raw = int(len(pred_raw))
        pred_norm_scored = _normalize_pred_edges_with_score(pred_raw[["source", "target", "score"]], alias_map)
        pred_norm_scored = pred_norm_scored[
            pred_norm_scored["source"].isin(gene_set) & pred_norm_scored["target"].isin(gene_set)
        ].reset_index(drop=True)

        truth_filtered = truth_edges[
            truth_edges["source"].isin(gene_set) & truth_edges["target"].isin(gene_set)
        ].drop_duplicates()
        truth_set = set(map(tuple, truth_filtered[["source", "target"]].itertuples(index=False, name=None)))

        pred_set = set(map(tuple, pred_norm_scored[["source", "target"]].itertuples(index=False, name=None)))
        tp_set = pred_set & truth_set
        tp = len(tp_set)
        fp = len(pred_set - truth_set)
        fn = len(truth_set - pred_set)
        n_pred = len(pred_set)
        n_true = len(truth_set)
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_true if n_true else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        n_genes = len(gene_set)
        n_possible = n_genes * (n_genes - 1)
        prevalence = n_true / n_possible if n_possible > 0 else float("nan")
        expected_tp_random = n_pred * prevalence if not math.isnan(prevalence) else float("nan")
        enrichment = precision / prevalence if prevalence > 0 else float("nan")

        precision_ci_low, precision_ci_high = _clopper_pearson_ci(tp, n_pred)
        recall_ci_low, recall_ci_high = _clopper_pearson_ci(tp, n_true)

        labels = np.array(
            [1 if edge in truth_set else 0 for edge in pred_norm_scored[["source", "target"]].itertuples(index=False, name=None)],
            dtype=int,
        )
        scores = pred_norm_scored["score"].to_numpy(dtype=float)
        run_aupr = float(aupr(scores, labels))
        hit_positions = np.where(labels == 1)[0] + 1
        first_hit_rank = float(hit_positions.min()) if hit_positions.size else float("nan")
        median_hit_rank = float(np.median(hit_positions)) if hit_positions.size else float("nan")

        detail_rows.append(
            {
                "model_tier": row["model_tier"],
                "cells": int(row["cells"]),
                "experiment_id": exp_id,
                "n_genes": n_genes,
                "n_possible_edges": n_possible,
                "n_true_edges": n_true,
                "n_edges_raw": n_edges_raw,
                "n_pred_edges_eval": n_pred,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "precision_ci_low": precision_ci_low,
                "precision_ci_high": precision_ci_high,
                "recall": recall,
                "recall_ci_low": recall_ci_low,
                "recall_ci_high": recall_ci_high,
                "f1": f1,
                "aupr": run_aupr,
                "first_hit_rank": first_hit_rank,
                "median_hit_rank": median_hit_rank,
                "expected_tp_random": expected_tp_random,
                "tp_minus_random_expectation": tp - expected_tp_random if not math.isnan(expected_tp_random) else float("nan"),
                "enrichment_over_random_precision": enrichment,
            }
        )

        for source, target in sorted(tp_set):
            true_positive_rows.append(
                {
                    "model_tier": row["model_tier"],
                    "cells": int(row["cells"]),
                    "source": source,
                    "target": target,
                }
            )

        for k in rank_k:
            if k > n_pred:
                continue
            tp_at_k = int(labels[:k].sum())
            rank_rows.append(
                {
                    "model_tier": row["model_tier"],
                    "cells": int(row["cells"]),
                    "k": k,
                    "tp_at_k": tp_at_k,
                    "precision_at_k": tp_at_k / k if k > 0 else float("nan"),
                    "recall_at_k": tp_at_k / n_true if n_true > 0 else float("nan"),
                }
            )

        run_edge_sets[run_id] = pred_set
        run_score_maps[run_id] = {
            (source, target): float(score)
            for source, target, score in pred_norm_scored[["source", "target", "score"]].itertuples(index=False, name=None)
        }

    detail_df = pd.DataFrame(detail_rows)
    rank_df = pd.DataFrame(rank_rows)
    shift_df = pd.DataFrame(shift_rows)
    true_positive_df = pd.DataFrame(true_positive_rows)

    # Paired cell-count shift analyses by tier.
    for tier in TIER_ORDER:
        id_200 = f"{tier}_200"
        id_1000 = f"{tier}_1000"
        if id_200 not in run_edge_sets or id_1000 not in run_edge_sets:
            continue

        set_200 = run_edge_sets[id_200]
        set_1000 = run_edge_sets[id_1000]
        jaccard_full = _jaccard(set_200, set_1000)

        shift_rows.append(
            {
                "model_tier": tier,
                "analysis": "fullset_jaccard_200_vs_1000",
                "k": np.nan,
                "value": jaccard_full,
                "statistic": np.nan,
                "p_value": np.nan,
            }
        )

        for k in [1000, 5000, 10000, 50000]:
            top_200 = set(list(run_score_maps[id_200].keys())[:k])
            top_1000 = set(list(run_score_maps[id_1000].keys())[:k])
            shift_rows.append(
                {
                    "model_tier": tier,
                    "analysis": "topk_jaccard_200_vs_1000",
                    "k": k,
                    "value": _jaccard(top_200, top_1000),
                    "statistic": np.nan,
                    "p_value": np.nan,
                }
            )

        union_edges = set_200 | set_1000
        s200 = np.array([run_score_maps[id_200].get(edge, 0.0) for edge in union_edges], dtype=float)
        s1000 = np.array([run_score_maps[id_1000].get(edge, 0.0) for edge in union_edges], dtype=float)
        union_spearman = _spearman_correlation(s200, s1000)
        ks = stats.ks_2samp(s200, s1000)
        mw = stats.mannwhitneyu(s200, s1000, alternative="two-sided")
        cliffs_delta = (2.0 * float(mw.statistic) / (len(s200) * len(s1000))) - 1.0 if len(s200) and len(s1000) else float("nan")

        shift_rows.extend(
            [
                {
                    "model_tier": tier,
                    "analysis": "union_spearman_200_vs_1000",
                    "k": np.nan,
                    "value": union_spearman,
                    "statistic": np.nan,
                    "p_value": np.nan,
                },
                {
                    "model_tier": tier,
                    "analysis": "ks_score_shift_200_vs_1000",
                    "k": np.nan,
                    "value": float(np.mean(s1000) - np.mean(s200)),
                    "statistic": float(ks.statistic),
                    "p_value": float(ks.pvalue),
                },
                {
                    "model_tier": tier,
                    "analysis": "cliffs_delta_score_shift_200_vs_1000",
                    "k": np.nan,
                    "value": cliffs_delta,
                    "statistic": float(mw.statistic),
                    "p_value": float(mw.pvalue),
                },
            ]
        )

    shift_df = pd.DataFrame(shift_rows)

    # Pairwise full-set Jaccard matrix across runs.
    run_ids = sorted(run_edge_sets.keys(), key=lambda name: (int(name.split("_")[1]), TIER_ORDER.index(name.split("_")[0])))
    pairwise_rows: list[dict[str, object]] = []
    for run_a in run_ids:
        for run_b in run_ids:
            pairwise_rows.append(
                {
                    "run_a": run_a,
                    "run_b": run_b,
                    "jaccard": _jaccard(run_edge_sets[run_a], run_edge_sets[run_b]),
                }
            )
    pairwise_df = pd.DataFrame(pairwise_rows)

    return detail_df, rank_df, shift_df, pairwise_df, true_positive_df, pd.DataFrame({"run_id": run_ids})


def _plot_tp_vs_random(detail_df: pd.DataFrame, out_path: Path) -> None:
    if detail_df.empty:
        return
    plot_df = detail_df.sort_values(["cells", "model_tier"]).copy()
    labels = [f"{row.model_tier}-{int(row.cells)}" for row in plot_df.itertuples(index=False)]
    x = np.arange(len(plot_df))
    width = 0.37

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar(x - width / 2, plot_df["tp"], width=width, label="Observed TP", color="#1f77b4")
    ax.bar(
        x + width / 2,
        plot_df["expected_tp_random"],
        width=width,
        label="Expected TP (random ranking)",
        color="#ff7f0e",
        alpha=0.8,
    )
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(
            idx - width / 2,
            float(row.tp) + 0.15,
            f"{row.enrichment_over_random_precision:.2f}x",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("True positives in evaluated edge set")
    ax.set_title("Observed vs random-expected TRRUST hits (rerun edge-level analysis)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    # Add top margin so labels don't clip into the title
    ymax = ax.get_ylim()[1]
    ax.set_ylim(top=ymax * 1.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_recall_at_k(rank_df: pd.DataFrame, out_path: Path) -> None:
    if rank_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for tier in TIER_ORDER:
        for cells, style in [(200, "-"), (1000, "--")]:
            sub = rank_df[(rank_df["model_tier"] == tier) & (rank_df["cells"] == cells)].sort_values("k")
            if sub.empty:
                continue
            ax.plot(sub["k"], sub["recall_at_k"], linestyle=style, marker="o", label=f"{tier}-{cells}")
    ax.set_xscale("log")
    ax.set_xlabel("Top-K edges kept (log scale)")
    ax.set_ylabel("Recall@K (TRRUST)")
    ax.set_title("Ranked recovery degrades at higher cell count")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_pairwise_jaccard_heatmap(pairwise_df: pd.DataFrame, run_order_df: pd.DataFrame, out_path: Path) -> None:
    if pairwise_df.empty or run_order_df.empty:
        return
    run_ids = run_order_df["run_id"].tolist()
    matrix = np.full((len(run_ids), len(run_ids)), np.nan, dtype=float)
    for i, run_a in enumerate(run_ids):
        for j, run_b in enumerate(run_ids):
            sel = pairwise_df[(pairwise_df["run_a"] == run_a) & (pairwise_df["run_b"] == run_b)]
            if not sel.empty:
                matrix[i, j] = float(sel["jaccard"].iloc[0])
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(run_ids)))
    ax.set_yticklabels(run_ids, fontsize=8)
    ax.set_title("Pairwise full-edge-set Jaccard across rerun conditions")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Jaccard overlap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    subproj = root / "subproject_06_scaling_failure_analysis"
    reports_dir = subproj / "reports"
    data_dir = reports_dir / "data"
    fig_dir = reports_dir / "figures"
    ext_data_dir = reports_dir / "data_extended"
    ext_fig_dir = reports_dir / "figures_extended"
    ext_data_dir.mkdir(parents=True, exist_ok=True)
    ext_fig_dir.mkdir(parents=True, exist_ok=True)

    snapshot_scaling = pd.read_csv(data_dir / "snapshot_kidney_scaling_f1.csv")
    snapshot_robustness = pd.read_csv(data_dir / "snapshot_kidney_robustness.csv")
    snapshot_policy = pd.read_csv(data_dir / "snapshot_policy_kidney_1000.csv")
    snapshot_baselines = pd.read_csv(data_dir / "snapshot_baselines.csv")
    snapshot_cross = pd.read_csv(data_dir / "snapshot_cross_tissue_large_1000.csv")

    # Archived workshop effects
    effects: list[dict[str, object]] = []
    for tier in TIER_ORDER:
        s = snapshot_scaling[snapshot_scaling["model_tier"] == tier].sort_values("cells")
        r = snapshot_robustness[snapshot_robustness["model_tier"] == tier].sort_values("cells")
        if not s.empty:
            f200 = float(s.loc[s["cells"] == 200, "f1"].iloc[0])
            f1000 = float(s.loc[s["cells"] == 1000, "f1"].iloc[0])
            f3000 = float(s.loc[s["cells"] == 3000, "f1"].iloc[0])
            effects.extend(
                [
                    {"metric": "f1_drop_200_to_1000_pct", "model_tier": tier, "value": -_safe_pct_change(f200, f1000)},
                    {"metric": "f1_drop_1000_to_3000_pct", "model_tier": tier, "value": -_safe_pct_change(f1000, f3000)},
                ]
            )
        if not r.empty:
            j200 = float(r.loc[r["cells"] == 200, "jaccard_mean"].iloc[0])
            j3000 = float(r.loc[r["cells"] == 3000, "jaccard_mean"].iloc[0])
            sp200 = float(r.loc[r["cells"] == 200, "spearman_mean"].iloc[0])
            sp3000 = float(r.loc[r["cells"] == 3000, "spearman_mean"].iloc[0])
            effects.extend(
                [
                    {"metric": "jaccard_drop_200_to_3000_pct", "model_tier": tier, "value": -_safe_pct_change(j200, j3000)},
                    {"metric": "spearman_delta_200_to_3000", "model_tier": tier, "value": sp3000 - sp200},
                ]
            )
    effects_df = pd.DataFrame(effects)
    effects_df.to_csv(ext_data_dir / "archived_effect_sizes.csv", index=False)

    # Archived sign tests
    tests = []
    for metric in [
        "f1_drop_200_to_1000_pct",
        "f1_drop_1000_to_3000_pct",
        "jaccard_drop_200_to_3000_pct",
        "spearman_delta_200_to_3000",
    ]:
        vals = effects_df.loc[effects_df["metric"] == metric, "value"].tolist()
        if metric == "spearman_delta_200_to_3000":
            diffs = vals  # expected negative
        else:
            diffs = [-v for v in vals]  # drop>0 implies negative diff in original quantity
        k_neg, n, p_one_sided = _sign_test_one_sided_neg(diffs)
        tests.append(
            {
                "analysis": "sign_test",
                "metric": metric,
                "n_nonzero": n,
                "k_neg": k_neg,
                "p_one_sided": p_one_sided,
            }
        )

    # Archived policy and baseline derived effects
    for _, row in snapshot_policy.iterrows():
        full = float(row["full_hvg_topk100"])
        best_pct = max(float(row["pct90"]), float(row["pct95"]))
        tests.extend(
            [
                {
                    "analysis": "policy_ratio",
                    "metric": f"{row['model_tier']}_best_percentile_over_full",
                    "n_nonzero": 1,
                    "k_neg": np.nan,
                    "p_one_sided": np.nan,
                    "value": _safe_ratio(best_pct, full),
                },
                {
                    "analysis": "policy_ratio",
                    "metric": f"{row['model_tier']}_tf_sources_targets_over_full",
                    "n_nonzero": 1,
                    "k_neg": np.nan,
                    "p_one_sided": np.nan,
                    "value": _safe_ratio(float(row["tf_sources_targets_topk100"]), full),
                },
            ]
        )

    kidney_attention_mean = float(snapshot_baselines.loc[snapshot_baselines["group"] == "attention-kidney-mean", "f1"].iloc[0])
    kidney_coexpr = float(snapshot_baselines.loc[snapshot_baselines["group"] == "coexpr-kidney", "f1"].iloc[0])
    kidney_grnboost2 = float(snapshot_baselines.loc[snapshot_baselines["group"] == "grnboost2-kidney-seed1", "f1"].iloc[0])
    tests.extend(
        [
            {
                "analysis": "baseline_ratio",
                "metric": "kidney_coexpression_over_attention_mean",
                "n_nonzero": 1,
                "k_neg": np.nan,
                "p_one_sided": np.nan,
                "value": _safe_ratio(kidney_coexpr, kidney_attention_mean),
            },
            {
                "analysis": "baseline_ratio",
                "metric": "kidney_grnboost2_over_attention_mean",
                "n_nonzero": 1,
                "k_neg": np.nan,
                "p_one_sided": np.nan,
                "value": _safe_ratio(kidney_grnboost2, kidney_attention_mean),
            },
        ]
    )

    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(ext_data_dir / "statistical_tests_and_ratios.csv", index=False)

    # Fresh reruns from run log
    run_log = root / "single_cell_mechinterp" / "outputs" / "scaling" / "run_log.csv"
    eval_df = _load_evaluate_rows(run_log)
    parsed = []
    for _, row in eval_df.iterrows():
        parsed_id = _parse_scaling_id(str(row["experiment_id"]))
        if not parsed_id:
            continue
        parsed.append({**parsed_id, **row.to_dict()})
    rerun_df = pd.DataFrame(parsed)
    if not rerun_df.empty:
        rerun_df = rerun_df[
            (rerun_df["tissue"] == "kidney")
            & (rerun_df["hvg"] == 2000)
            & (rerun_df["max_genes"] == 1200)
            & (rerun_df["top_k"] == 100)
            & (rerun_df["seed"] == 1)
            & (rerun_df["cells"].isin([200, 1000]))
        ].copy()
        rerun_df.sort_values(["model_tier", "cells"], inplace=True)
    rerun_df.to_csv(ext_data_dir / "rerun_seed1_kidney_cells200_1000.csv", index=False)

    # Compare rerun seed1 to archived means for same tier/cell.
    join_cols = ["model_tier", "cells"]
    snap_cmp = snapshot_scaling.rename(columns={"f1": "f1_snapshot"})[join_cols + ["f1_snapshot"]]
    rerun_cmp = rerun_df.rename(columns={"f1": "f1_rerun"})[join_cols + ["f1_rerun"]] if not rerun_df.empty else pd.DataFrame(columns=join_cols + ["f1_rerun"])
    joined = snap_cmp.merge(rerun_cmp, on=join_cols, how="inner")
    if not joined.empty:
        joined["abs_diff"] = (joined["f1_rerun"] - joined["f1_snapshot"]).abs()
        joined["rel_diff_pct_vs_snapshot"] = np.where(
            joined["f1_snapshot"] > 0,
            (joined["f1_rerun"] - joined["f1_snapshot"]) / joined["f1_snapshot"] * 100.0,
            np.nan,
        )
    joined.to_csv(ext_data_dir / "rerun_vs_archived_snapshot.csv", index=False)

    # Rerun degradation stats (200->1000 across tiers where available)
    rerun_drop_rows = []
    for tier in TIER_ORDER:
        sub = rerun_df[rerun_df["model_tier"] == tier] if not rerun_df.empty else pd.DataFrame()
        if sub.empty:
            continue
        f200 = sub.loc[sub["cells"] == 200, "f1"]
        f1000 = sub.loc[sub["cells"] == 1000, "f1"]
        if f200.empty or f1000.empty:
            continue
        f200_val = float(f200.iloc[0])
        f1000_val = float(f1000.iloc[0])
        rerun_drop_rows.append(
            {
                "model_tier": tier,
                "f1_200": f200_val,
                "f1_1000": f1000_val,
                "pct_drop_200_to_1000": -_safe_pct_change(f200_val, f1000_val),
                "ratio_1000_over_200": _safe_ratio(f1000_val, f200_val),
            }
        )
    rerun_drop_df = pd.DataFrame(rerun_drop_rows)
    rerun_drop_df.to_csv(ext_data_dir / "rerun_drop_200_to_1000.csv", index=False)

    if not rerun_drop_df.empty:
        diffs = [float(row["f1_1000"] - row["f1_200"]) for _, row in rerun_drop_df.iterrows()]
        k_neg, n, p = _sign_test_one_sided_neg(diffs)
        rerun_test_df = pd.DataFrame(
            [
                {
                    "analysis": "rerun_sign_test",
                    "metric": "f1_1000_minus_200",
                    "n_nonzero": n,
                    "k_neg": k_neg,
                    "p_one_sided": p,
                }
            ]
        )
    else:
        rerun_test_df = pd.DataFrame(columns=["analysis", "metric", "n_nonzero", "k_neg", "p_one_sided"])
    rerun_test_df.to_csv(ext_data_dir / "rerun_tests.csv", index=False)

    # Edge-level rerun diagnostics (counts, ranking behavior, and cell-shift structure).
    (
        rerun_edge_detail_df,
        rerun_rank_df,
        rerun_shift_df,
        rerun_pairwise_jaccard_df,
        rerun_true_positive_df,
        rerun_order_df,
    ) = _build_rerun_edge_diagnostics(root, rerun_df)
    rerun_edge_detail_df.to_csv(ext_data_dir / "rerun_edge_level_counts.csv", index=False)
    rerun_rank_df.to_csv(ext_data_dir / "rerun_rank_recovery.csv", index=False)
    rerun_shift_df.to_csv(ext_data_dir / "rerun_cell_shift_tests.csv", index=False)
    rerun_pairwise_jaccard_df.to_csv(ext_data_dir / "rerun_pairwise_fullset_jaccard.csv", index=False)
    rerun_true_positive_df.to_csv(ext_data_dir / "rerun_true_positive_edges.csv", index=False)

    # Figures
    _plot_effect_heatmap(effects_df, ext_fig_dir / "fig_ext_archived_effect_heatmap.png")
    _plot_repro_vs_snapshot(joined, ext_fig_dir / "fig_ext_rerun_vs_snapshot.png")
    _plot_rerun_scaling(rerun_df, ext_fig_dir / "fig_ext_rerun_scaling_seed1.png")
    _plot_drop_bars(rerun_df, ext_fig_dir / "fig_ext_rerun_drop_200_to_1000.png")
    _plot_tp_vs_random(rerun_edge_detail_df, ext_fig_dir / "fig_ext_rerun_tp_vs_random.png")
    _plot_recall_at_k(rerun_rank_df, ext_fig_dir / "fig_ext_rerun_recall_at_k.png")
    _plot_pairwise_jaccard_heatmap(
        rerun_pairwise_jaccard_df,
        rerun_order_df,
        ext_fig_dir / "fig_ext_rerun_pairwise_jaccard_heatmap.png",
    )

    # Compact narrative table with key numbers for manuscript insertion.
    key_rows = []
    for tier in TIER_ORDER:
        s = snapshot_scaling[snapshot_scaling["model_tier"] == tier].sort_values("cells")
        if s.empty:
            continue
        key_rows.append(
            {
                "source": "archived",
                "model_tier": tier,
                "f1_200": float(s.loc[s["cells"] == 200, "f1"].iloc[0]),
                "f1_1000": float(s.loc[s["cells"] == 1000, "f1"].iloc[0]),
                "f1_3000": float(s.loc[s["cells"] == 3000, "f1"].iloc[0]),
            }
        )
    for _, row in rerun_drop_df.iterrows():
        key_rows.append(
            {
                "source": "rerun_seed1",
                "model_tier": row["model_tier"],
                "f1_200": row["f1_200"],
                "f1_1000": row["f1_1000"],
                "f1_3000": np.nan,
            }
        )
    key_df = pd.DataFrame(key_rows)
    key_df.to_csv(ext_data_dir / "key_f1_table_archived_and_rerun.csv", index=False)

    print(f"Extended evidence written to: {ext_data_dir}")
    print(f"Extended figures written to: {ext_fig_dir}")


if __name__ == "__main__":
    main()
