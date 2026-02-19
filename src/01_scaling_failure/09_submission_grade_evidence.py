#!/usr/bin/env python3
"""Generate submission-grade scaling evidence with multi-seed and external-reference checks.

Outputs:
- Multi-seed TRRUST/DoRothEA run-level and aggregate metrics.
- Paired directional statistics for scaling transitions.
- Seed-robustness summaries (Jaccard and Spearman) from edge score maps.
- Publication figures for scaling, robustness, and transition effects.
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

TIER_ORDER = ["small", "medium", "large"]
CELL_ORDER = [200, 1000, 3000]
REFERENCE_ORDER = ["trrust", "dorothea"]
ID_PATTERN = re.compile(
    r"^(small|medium|large)_([a-z]+)_cells(\d+)_hvg(\d+)_full_hvg_maxgenes(\d+)_seed(\d+)_topk(\d+)$"
)


def _load_success_evaluations(log_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with log_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("step") != "evaluate" or row.get("status") != "success":
                continue
            exp_id = str(row.get("experiment_id", ""))
            match = ID_PATTERN.match(exp_id)
            if not match:
                continue
            tier, tissue, cells, hvg, max_genes, seed, top_k = match.groups()
            metrics_raw = row.get("metrics", "")
            if not metrics_raw:
                continue
            metrics = json.loads(metrics_raw)
            rows.append(
                {
                    "experiment_id": exp_id,
                    "timestamp": row.get("timestamp", ""),
                    "model_tier": tier,
                    "tissue": tissue,
                    "cells": int(cells),
                    "hvg": int(hvg),
                    "max_genes": int(max_genes),
                    "seed": int(seed),
                    "top_k": int(top_k),
                    "precision_log": float(metrics.get("precision", float("nan"))),
                    "recall_log": float(metrics.get("recall", float("nan"))),
                    "f1_log": float(metrics.get("f1", float("nan"))),
                }
            )
    return pd.DataFrame(rows)


def _sign_test_one_sided_neg(values: Iterable[float]) -> tuple[int, int, float]:
    vals = [float(v) for v in values if not math.isnan(v) and float(v) != 0.0]
    n = len(vals)
    if n == 0:
        return 0, 0, float("nan")
    k_neg = sum(1 for v in vals if v < 0)
    p = stats.binom.sf(k_neg - 1, n, 0.5)
    return k_neg, n, float(p)


def _bootstrap_ci(values: np.ndarray, iters: int = 5000, seed: int = 1) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iters, values.size), replace=True)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _safe_pct_drop(a: float, b: float) -> float:
    if a == 0:
        return float("nan")
    return float((a - b) / a * 100.0)


def _jaccard(edges_a: set[tuple[str, str]], edges_b: set[tuple[str, str]]) -> float:
    if not edges_a and not edges_b:
        return float("nan")
    return float(len(edges_a & edges_b) / len(edges_a | edges_b))


def _spearman_union(scores_a: dict[tuple[str, str], float], scores_b: dict[tuple[str, str], float]) -> float:
    union = set(scores_a) | set(scores_b)
    if len(union) < 2:
        return float("nan")
    a = np.array([scores_a.get(edge, 0.0) for edge in union], dtype=float)
    b = np.array([scores_b.get(edge, 0.0) for edge in union], dtype=float)
    corr, _ = stats.spearmanr(a, b)
    return float(corr)


def _normalize_pred_with_score(pred_raw: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    pred = pred_raw.copy()
    pred["source"] = pred["source"].astype(str).str.strip().str.upper().map(lambda v: alias_map.get(v, v))
    pred["target"] = pred["target"].astype(str).str.strip().str.upper().map(lambda v: alias_map.get(v, v))
    pred = pred[(pred["source"] != "") & (pred["target"] != "")]
    pred = pred.sort_values("score", ascending=False)
    pred = pred.drop_duplicates(subset=["source", "target"], keep="first")
    return pred.reset_index(drop=True)


def _evaluate_run_against_reference(
    edge_path: Path,
    h5ad_path: Path,
    truth_edges: pd.DataFrame,
    alias_map: dict[str, str],
) -> dict[str, float]:
    adata = sc.read_h5ad(h5ad_path)
    gene_set = set(normalize_gene_names(adata.var_names.values, alias_map).tolist())

    pred_raw = pd.read_csv(edge_path, sep="\t")[["source", "target", "score"]]
    pred = _normalize_pred_with_score(pred_raw, alias_map)
    pred = pred[pred["source"].isin(gene_set) & pred["target"].isin(gene_set)].reset_index(drop=True)

    truth = truth_edges[truth_edges["source"].isin(gene_set) & truth_edges["target"].isin(gene_set)].drop_duplicates()
    pred_set = set(map(tuple, pred[["source", "target"]].itertuples(index=False, name=None)))
    truth_set = set(map(tuple, truth[["source", "target"]].itertuples(index=False, name=None)))

    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(truth_set) if truth_set else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "n_genes": float(len(gene_set)),
        "n_pred_edges_eval": float(len(pred_set)),
        "n_true_edges_eval": float(len(truth_set)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _aggregate_reference_metrics(run_ref_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (reference, tier, cells), group in run_ref_df.groupby(["reference", "model_tier", "cells"]):
        vals = group["f1"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(vals, iters=4000, seed=11)
        rows.append(
            {
                "reference": reference,
                "model_tier": tier,
                "cells": int(cells),
                "f1_mean": float(np.mean(vals)),
                "f1_std": float(np.std(vals, ddof=0)),
                "f1_ci_low": ci_low,
                "f1_ci_high": ci_high,
                "precision_mean": float(group["precision"].mean()),
                "recall_mean": float(group["recall"].mean()),
                "tp_mean": float(group["tp"].mean()),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _compute_paired_tests(run_ref_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for reference in REFERENCE_ORDER:
        subset = run_ref_df[run_ref_df["reference"] == reference]
        # Tier-level transition summaries and sign tests (n=3 per tier).
        for tier in TIER_ORDER:
            tdf = subset[subset["model_tier"] == tier]
            for a, b, label in [(200, 1000, "200_to_1000"), (1000, 3000, "1000_to_3000")]:
                diffs = []
                for seed in sorted(tdf["seed"].unique()):
                    fa = tdf[(tdf["cells"] == a) & (tdf["seed"] == seed)]["f1"]
                    fb = tdf[(tdf["cells"] == b) & (tdf["seed"] == seed)]["f1"]
                    if fa.empty or fb.empty:
                        continue
                    diffs.append(float(fb.iloc[0] - fa.iloc[0]))
                k_neg, n, p = _sign_test_one_sided_neg(diffs)
                rows.append(
                    {
                        "reference": reference,
                        "scope": "tier",
                        "model_tier": tier,
                        "transition": label,
                        "n_pairs": n,
                        "k_negative": k_neg,
                        "p_sign_one_sided": p,
                        "mean_diff_f1": float(np.mean(diffs)) if diffs else float("nan"),
                    }
                )

        # Pooled across all tiers and seeds (n up to 9).
        for a, b, label in [(200, 1000, "200_to_1000"), (1000, 3000, "1000_to_3000")]:
            diffs = []
            for tier in TIER_ORDER:
                tdf = subset[subset["model_tier"] == tier]
                for seed in sorted(tdf["seed"].unique()):
                    fa = tdf[(tdf["cells"] == a) & (tdf["seed"] == seed)]["f1"]
                    fb = tdf[(tdf["cells"] == b) & (tdf["seed"] == seed)]["f1"]
                    if fa.empty or fb.empty:
                        continue
                    diffs.append(float(fb.iloc[0] - fa.iloc[0]))
            k_neg, n, p = _sign_test_one_sided_neg(diffs)
            wilcoxon_p = float(stats.wilcoxon(diffs, alternative="less").pvalue) if len(diffs) >= 3 else float("nan")
            rows.append(
                {
                    "reference": reference,
                    "scope": "pooled",
                    "model_tier": "all",
                    "transition": label,
                    "n_pairs": n,
                    "k_negative": k_neg,
                    "p_sign_one_sided": p,
                    "p_wilcoxon_one_sided": wilcoxon_p,
                    "mean_diff_f1": float(np.mean(diffs)) if diffs else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _compute_transition_effects(run_ref_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for reference in REFERENCE_ORDER:
        rdf = run_ref_df[run_ref_df["reference"] == reference]
        for tier in TIER_ORDER:
            tdf = rdf[rdf["model_tier"] == tier]
            m200 = float(tdf[tdf["cells"] == 200]["f1"].mean())
            m1000 = float(tdf[tdf["cells"] == 1000]["f1"].mean())
            m3000 = float(tdf[tdf["cells"] == 3000]["f1"].mean())
            rows.extend(
                [
                    {
                        "reference": reference,
                        "model_tier": tier,
                        "metric": "f1_drop_pct_200_to_1000",
                        "value": _safe_pct_drop(m200, m1000),
                    },
                    {
                        "reference": reference,
                        "model_tier": tier,
                        "metric": "f1_drop_pct_1000_to_3000",
                        "value": _safe_pct_drop(m1000, m3000),
                    },
                ]
            )
    return pd.DataFrame(rows)


def _compute_robustness(
    run_df: pd.DataFrame,
    alias_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_maps: dict[str, dict[tuple[str, str], float]] = {}
    edge_sets: dict[str, set[tuple[str, str]]] = {}

    for row in run_df.itertuples(index=False):
        run_dir = ROOT / "single_cell_mechinterp" / "outputs" / "scaling" / row.experiment_id
        pred_raw = pd.read_csv(run_dir / "inferred_edges.tsv", sep="\t")[["source", "target", "score"]]
        pred = _normalize_pred_with_score(pred_raw, alias_map)

        adata = sc.read_h5ad(run_dir / "tabula_sapiens_processed.h5ad")
        gene_set = set(normalize_gene_names(adata.var_names.values, alias_map).tolist())
        pred = pred[pred["source"].isin(gene_set) & pred["target"].isin(gene_set)].reset_index(drop=True)

        key = row.experiment_id
        edge_sets[key] = set(map(tuple, pred[["source", "target"]].itertuples(index=False, name=None)))
        score_maps[key] = {
            (source, target): float(score)
            for source, target, score in pred[["source", "target", "score"]].itertuples(index=False, name=None)
        }

    pair_rows: list[dict[str, object]] = []
    agg_rows: list[dict[str, object]] = []
    for (tier, cells), group in run_df.groupby(["model_tier", "cells"]):
        run_ids = group["experiment_id"].tolist()
        jvals = []
        svals = []
        for i in range(len(run_ids)):
            for j in range(i + 1, len(run_ids)):
                a = run_ids[i]
                b = run_ids[j]
                jac = _jaccard(edge_sets[a], edge_sets[b])
                rho = _spearman_union(score_maps[a], score_maps[b])
                jvals.append(jac)
                svals.append(rho)
                pair_rows.append(
                    {
                        "model_tier": tier,
                        "cells": int(cells),
                        "run_a": a,
                        "run_b": b,
                        "jaccard": jac,
                        "spearman_union": rho,
                    }
                )
        agg_rows.append(
            {
                "model_tier": tier,
                "cells": int(cells),
                "jaccard_mean": float(np.nanmean(jvals)) if jvals else float("nan"),
                "jaccard_std": float(np.nanstd(jvals)) if jvals else float("nan"),
                "spearman_mean": float(np.nanmean(svals)) if svals else float("nan"),
                "spearman_std": float(np.nanstd(svals)) if svals else float("nan"),
                "n_pairs": int(len(jvals)),
            }
        )
    return pd.DataFrame(agg_rows), pd.DataFrame(pair_rows)


def _plot_scaling_with_ci(agg_df: pd.DataFrame, reference: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sub = agg_df[agg_df["reference"] == reference].copy()
    for tier in TIER_ORDER:
        tdf = sub[sub["model_tier"] == tier].sort_values("cells")
        if tdf.empty:
            continue
        ax.plot(tdf["cells"], tdf["f1_mean"], marker="o", linewidth=2, label=tier)
        ax.fill_between(tdf["cells"], tdf["f1_ci_low"], tdf["f1_ci_high"], alpha=0.18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cells (log scale)")
    ax.set_ylabel(f"F1 vs {reference.upper()} (log scale)")
    ax.set_title(f"Multi-seed scaling with 95% CI ({reference.upper()})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Model tier")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_robustness(robust_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharex=True)
    for tier in TIER_ORDER:
        tdf = robust_df[robust_df["model_tier"] == tier].sort_values("cells")
        if tdf.empty:
            continue
        axes[0].plot(tdf["cells"], tdf["jaccard_mean"], marker="o", linewidth=2, label=tier)
        axes[1].plot(tdf["cells"], tdf["spearman_mean"], marker="o", linewidth=2, label=tier)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("Cells (log scale)")
    axes[1].set_xlabel("Cells (log scale)")
    axes[0].set_ylabel("Seed-pair Jaccard (mean)")
    axes[1].set_ylabel("Seed-pair Spearman (mean)")
    axes[0].set_title("Edge-set overlap robustness")
    axes[1].set_title("Score-rank robustness")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    axes[1].legend(title="Model tier")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_transition_drops(run_ref_df: pd.DataFrame, out_path: Path) -> None:
    rows: list[dict[str, object]] = []
    tr = run_ref_df[run_ref_df["reference"] == "trrust"]
    for tier in TIER_ORDER:
        tdf = tr[tr["model_tier"] == tier]
        for seed in sorted(tdf["seed"].unique()):
            fa = tdf[(tdf["cells"] == 200) & (tdf["seed"] == seed)]["f1"]
            fb = tdf[(tdf["cells"] == 1000) & (tdf["seed"] == seed)]["f1"]
            if fa.empty or fb.empty:
                continue
            rows.append(
                {
                    "model_tier": tier,
                    "seed": int(seed),
                    "drop_pct_200_to_1000": _safe_pct_drop(float(fa.iloc[0]), float(fb.iloc[0])),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(TIER_ORDER))
    means = [float(df[df["model_tier"] == tier]["drop_pct_200_to_1000"].mean()) for tier in TIER_ORDER]
    ax.bar(x, means, alpha=0.6, width=0.6, color="#4C78A8")
    # Seed points
    for i, tier in enumerate(TIER_ORDER):
        vals = df[df["model_tier"] == tier]["drop_pct_200_to_1000"].to_numpy(dtype=float)
        jitter = np.linspace(-0.12, 0.12, num=len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full_like(vals, x[i], dtype=float) + jitter, vals, color="black", s=26, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(TIER_ORDER)
    ax.set_ylabel("F1 drop (%) from 200 to 1000 cells")
    ax.set_title("Multi-seed degradation magnitude (TRRUST)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    reports_dir = ROOT / "subproject_06_scaling_failure_analysis" / "reports"
    ext_data_dir = reports_dir / "data_extended"
    ext_fig_dir = reports_dir / "figures_extended"
    ext_data_dir.mkdir(parents=True, exist_ok=True)
    ext_fig_dir.mkdir(parents=True, exist_ok=True)

    run_log_path = ROOT / "single_cell_mechinterp" / "outputs" / "scaling" / "run_log.csv"
    eval_df = _load_success_evaluations(run_log_path)

    run_df = eval_df[
        (eval_df["tissue"] == "kidney")
        & (eval_df["hvg"] == 2000)
        & (eval_df["max_genes"] == 1200)
        & (eval_df["top_k"] == 100)
        & (eval_df["cells"].isin(CELL_ORDER))
        & (eval_df["seed"].isin([1, 2, 3]))
    ].copy()
    run_df = run_df.sort_values(["model_tier", "cells", "seed"]).reset_index(drop=True)
    run_df.to_csv(ext_data_dir / "multiseed_kidney_fullhvg_topk100_runs.csv", index=False)

    expected_n = len(TIER_ORDER) * len(CELL_ORDER) * 3
    if len(run_df) != expected_n:
        print(f"Warning: expected {expected_n} runs but found {len(run_df)}.")

    alias_map = load_hgnc_alias_map(ROOT / "single_cell_mechinterp" / "external" / "hgnc_complete_set.txt")
    trrust = normalize_edges(
        load_dorothea(ROOT / "single_cell_mechinterp" / "external" / "networks" / "trrust_human.tsv"),
        alias_map,
    ).drop_duplicates()
    dorothea = normalize_edges(
        load_dorothea(
            ROOT / "single_cell_mechinterp" / "external" / "networks" / "dorothea_human.tsv",
            confidence_levels=["A", "B", "C", "D"],
        ),
        alias_map,
    ).drop_duplicates()
    references = {"trrust": trrust, "dorothea": dorothea}

    run_ref_rows: list[dict[str, object]] = []
    for row in run_df.itertuples(index=False):
        run_dir = ROOT / "single_cell_mechinterp" / "outputs" / "scaling" / row.experiment_id
        edge_path = run_dir / "inferred_edges.tsv"
        h5ad_path = run_dir / "tabula_sapiens_processed.h5ad"
        for reference, truth_edges in references.items():
            metrics = _evaluate_run_against_reference(edge_path, h5ad_path, truth_edges, alias_map)
            run_ref_rows.append(
                {
                    "experiment_id": row.experiment_id,
                    "model_tier": row.model_tier,
                    "cells": int(row.cells),
                    "seed": int(row.seed),
                    "reference": reference,
                    **metrics,
                }
            )
    run_ref_df = pd.DataFrame(run_ref_rows)
    run_ref_df.to_csv(ext_data_dir / "multiseed_run_reference_metrics.csv", index=False)

    agg_df = _aggregate_reference_metrics(run_ref_df)
    agg_df.to_csv(ext_data_dir / "multiseed_aggregate_reference_metrics.csv", index=False)

    paired_df = _compute_paired_tests(run_ref_df)
    paired_df.to_csv(ext_data_dir / "multiseed_paired_direction_tests.csv", index=False)

    effects_df = _compute_transition_effects(run_ref_df)
    effects_df.to_csv(ext_data_dir / "multiseed_transition_effects.csv", index=False)

    robust_df, robust_pairs_df = _compute_robustness(run_df, alias_map)
    robust_df.to_csv(ext_data_dir / "multiseed_robustness_summary.csv", index=False)
    robust_pairs_df.to_csv(ext_data_dir / "multiseed_robustness_pairs.csv", index=False)

    _plot_scaling_with_ci(agg_df, "trrust", ext_fig_dir / "fig_ext_multiseed_scaling_trrust_ci.png")
    _plot_scaling_with_ci(agg_df, "dorothea", ext_fig_dir / "fig_ext_multiseed_scaling_dorothea_ci.png")
    _plot_robustness(robust_df, ext_fig_dir / "fig_ext_multiseed_robustness.png")
    _plot_transition_drops(run_ref_df, ext_fig_dir / "fig_ext_multiseed_drop_distribution.png")

    print(f"Wrote submission-grade data to: {ext_data_dir}")
    print(f"Wrote submission-grade figures to: {ext_fig_dir}")


if __name__ == "__main__":
    main()

