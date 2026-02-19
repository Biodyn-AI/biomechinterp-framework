#!/usr/bin/env python3
"""
Matched real-data calibration for detectability phase-diagram claims.

This script builds paired edge-level datasets where both signal families are
measured on the same TF-target edges. It reports:
1) Biological alignment metrics (AUROC, AUPRC, top-quantile enrichment).
2) A detectability-proxy ratio from positive-vs-negative score separation.
3) Bootstrap uncertainty (row-wise and cluster-aware variants).
4) Adversarial controls (label permutation, leave-one-seed-out stability,
   and quantile-sensitivity analysis with BH-adjusted p-values).

The goal is to strengthen reviewer-facing robustness checks while keeping the
compute footprint low.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run matched real-data calibration for detectability study."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs/full_study",
        help="Output root where tables/ and figures/ live.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=29,
        help="Random seed for bootstrap and permutation draws.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
        help="Bootstrap draws for confidence intervals.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=5000,
        help="Permutation draws for adversarial null controls.",
    )
    parser.add_argument(
        "--top-quantile",
        type=float,
        default=0.8,
        help="Quantile threshold used for primary enrichment analysis.",
    )
    parser.add_argument(
        "--quantile-sweep",
        type=float,
        nargs="+",
        default=[0.7, 0.8, 0.9],
        help="Quantiles used for enrichment sensitivity checks.",
    )
    return parser.parse_args()


def input_files(repo_root: Path) -> list[tuple[str, Path]]:
    """Return known invariant-causal edge-score files across seeds."""
    files = [
        ("seed_42", repo_root / "single_cell_mechinterp/outputs/invariant_causal_edges/results/edge_scores.tsv"),
        (
            "seed_43",
            repo_root
            / "single_cell_mechinterp/outputs/invariant_causal_edges/multiseed/seed_43/results/edge_scores.tsv",
        ),
        (
            "seed_44",
            repo_root
            / "single_cell_mechinterp/outputs/invariant_causal_edges/multiseed/seed_44/results/edge_scores.tsv",
        ),
    ]
    missing = [str(path) for _, path in files if not path.exists()]
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(f"Missing required edge-score files:\n{missing_text}")
    return files


def load_seed_tables(files: Iterable[tuple[str, Path]]) -> pd.DataFrame:
    """Load and concatenate seed tables with a shared union label."""
    frames: list[pd.DataFrame] = []
    for seed, path in files:
        df = pd.read_csv(path, sep="\t")
        df["seed"] = seed
        df["label_union"] = (
            (df["label_trrust"].fillna(0).astype(int) > 0)
            | (df["label_dorothea"].fillna(0).astype(int) > 0)
        ).astype(int)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(["seed", "source", "target"]).reset_index(drop=True)
    return merged


def build_edge_level_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Build paired edge-level table with one row per (seed, source, target)."""
    edge_df = merged[
        [
            "seed",
            "source",
            "target",
            "causal_mean_abs_score",
            "attention_mean_score",
            "label_union",
            "label_trrust",
            "label_dorothea",
        ]
    ].copy()
    edge_df = edge_df.rename(
        columns={
            "causal_mean_abs_score": "causal_score",
            "attention_mean_score": "attention_score",
        }
    )
    edge_df["causal_score"] = edge_df["causal_score"].astype(float).abs()
    edge_df["attention_score"] = edge_df["attention_score"].astype(float).abs()
    edge_df = edge_df[(edge_df["causal_score"] > 0.0) & (edge_df["attention_score"] > 0.0)].reset_index(drop=True)
    return edge_df


def build_edge_tissue_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Build paired edge-tissue table to increase matched sample support."""
    tissues = ["external_lung", "immune", "kidney", "lung"]
    frames: list[pd.DataFrame] = []
    for tissue in tissues:
        effect_col = f"{tissue}_effect_mean"
        attention_col = f"{tissue}_attention"
        frame = merged[
            [
                "seed",
                "source",
                "target",
                "label_union",
                "label_trrust",
                "label_dorothea",
                effect_col,
                attention_col,
            ]
        ].copy()
        frame["tissue"] = tissue
        frame = frame.rename(columns={effect_col: "causal_score", attention_col: "attention_score"})
        frame["causal_score"] = frame["causal_score"].astype(float).abs()
        frame["attention_score"] = frame["attention_score"].astype(float).abs()
        frame = frame[(frame["causal_score"] > 0.0) & (frame["attention_score"] > 0.0)].reset_index(drop=True)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def separation_stats(score: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """
    Compute positive-negative separation and pooled spread.

    Returns:
      mean difference (positive - negative),
      pooled standard deviation,
      standardized separation d = mean_diff / pooled_std.
    """
    positives = score[labels == 1]
    negatives = score[labels == 0]
    if positives.size < 2 or negatives.size < 2:
        return float("nan"), float("nan"), float("nan")
    mean_diff = float(positives.mean() - negatives.mean())
    pooled_std = float(np.sqrt((positives.var(ddof=1) + negatives.var(ddof=1)) / 2.0))
    if pooled_std <= 0.0:
        return mean_diff, pooled_std, float("nan")
    return mean_diff, pooled_std, mean_diff / pooled_std


def detectability_ratio_from_separation(
    causal_mean_diff: float,
    causal_sigma: float,
    attention_mean_diff: float,
    attention_sigma: float,
) -> float:
    """
    Relative cell budget proxy from matched score separation.

    If n* scales with (sigma / |mu|)^2, then:
      ratio = n*_causal / n*_attention
            = ((sigma_c / |mu_c|) / (sigma_a / |mu_a|))^2.
    """
    if (
        not np.isfinite(causal_mean_diff)
        or not np.isfinite(causal_sigma)
        or not np.isfinite(attention_mean_diff)
        or not np.isfinite(attention_sigma)
    ):
        return float("nan")
    if abs(causal_mean_diff) <= 1e-12 or abs(attention_mean_diff) <= 1e-12:
        return float("nan")
    if causal_sigma <= 0.0 or attention_sigma <= 0.0:
        return float("nan")
    return float(((causal_sigma / abs(causal_mean_diff)) / (attention_sigma / abs(attention_mean_diff))) ** 2)


def safe_roc_auc(labels: np.ndarray, score: np.ndarray) -> float:
    """Return AUROC when both classes are present, else NaN."""
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, score))


def safe_auprc(labels: np.ndarray, score: np.ndarray) -> float:
    """Return AUPRC when both classes are present, else NaN."""
    if np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, score))


def top_quantile_lift(score: np.ndarray, labels: np.ndarray, quantile: float) -> float:
    """Positive-rate lift in top score quantile versus overall prevalence."""
    if score.size == 0:
        return float("nan")
    baseline = float(labels.mean())
    if baseline <= 0.0:
        return float("nan")
    threshold = float(np.quantile(score, quantile))
    top_mask = score >= threshold
    top_rate = float(labels[top_mask].mean())
    return top_rate / baseline


def compute_comparative_stats(
    labels: np.ndarray,
    causal: np.ndarray,
    attention: np.ndarray,
    top_quantile: float,
) -> dict[str, float]:
    """Compute comparative metrics shared by bootstrap and permutation controls."""
    mu_c, sigma_c, d_c = separation_stats(score=causal, labels=labels)
    mu_a, sigma_a, d_a = separation_stats(score=attention, labels=labels)
    return {
        "relative_cells_causal_vs_attention": detectability_ratio_from_separation(
            causal_mean_diff=mu_c,
            causal_sigma=sigma_c,
            attention_mean_diff=mu_a,
            attention_sigma=sigma_a,
        ),
        "causal_d": d_c,
        "attention_d": d_a,
        "d_difference_causal_minus_attention": d_c - d_a,
        "auroc_causal": safe_roc_auc(labels, causal),
        "auroc_attention": safe_roc_auc(labels, attention),
        "auroc_difference_causal_minus_attention": safe_roc_auc(labels, causal)
        - safe_roc_auc(labels, attention),
        "auprc_causal": safe_auprc(labels, causal),
        "auprc_attention": safe_auprc(labels, attention),
        "auprc_difference_causal_minus_attention": safe_auprc(labels, causal)
        - safe_auprc(labels, attention),
        "top_quantile_lift_causal": top_quantile_lift(causal, labels, quantile=top_quantile),
        "top_quantile_lift_attention": top_quantile_lift(attention, labels, quantile=top_quantile),
        "top_quantile_lift_difference_causal_minus_attention": top_quantile_lift(causal, labels, quantile=top_quantile)
        - top_quantile_lift(attention, labels, quantile=top_quantile),
    }


def summarize_dataset(dataset_name: str, df: pd.DataFrame, top_quantile: float) -> pd.DataFrame:
    """Summarize family-level metrics for one matched dataset."""
    labels = df["label_union"].to_numpy(dtype=int)
    rows: list[dict[str, float | int | str]] = []

    for family, score_col in [("causal-like", "causal_score"), ("attention-like", "attention_score")]:
        score = df[score_col].to_numpy(dtype=float)
        mean_diff, pooled_sigma, d_stat = separation_stats(score=score, labels=labels)
        rows.append(
            {
                "dataset": dataset_name,
                "family": family,
                "n_rows": int(df.shape[0]),
                "n_positive_labels": int(labels.sum()),
                "label_prevalence": float(labels.mean()),
                "auroc": safe_roc_auc(labels, score),
                "auprc": safe_auprc(labels, score),
                "top_quantile": top_quantile,
                "top_quantile_lift": top_quantile_lift(score, labels, quantile=top_quantile),
                "positive_minus_negative_mean": mean_diff,
                "pooled_sigma": pooled_sigma,
                "standardized_separation_d": d_stat,
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap_sample(
    df: pd.DataFrame,
    cluster_cols: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Bootstrap rows by resampling edge clusters rather than individual rows."""
    unique_clusters = df[cluster_cols].drop_duplicates().reset_index(drop=True)
    sampled_idx = rng.integers(0, unique_clusters.shape[0], size=unique_clusters.shape[0])
    sampled_clusters = unique_clusters.iloc[sampled_idx].copy()
    counts = sampled_clusters.groupby(cluster_cols, as_index=False).size().rename(columns={"size": "_weight"})
    merged = df.merge(counts, on=cluster_cols, how="inner")
    replicated = merged.loc[merged.index.repeat(merged["_weight"])].drop(columns="_weight").reset_index(drop=True)
    return replicated


def bootstrap_comparison(
    dataset_name: str,
    df: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    top_quantile: float,
    cluster_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap comparative metrics between families."""
    rng = np.random.default_rng(seed)
    draws: list[dict[str, float | int | str]] = []

    for draw_id in range(n_bootstrap):
        if cluster_cols:
            sampled = cluster_bootstrap_sample(df=df, cluster_cols=cluster_cols, rng=rng)
        else:
            idx = rng.integers(0, df.shape[0], size=df.shape[0])
            sampled = df.iloc[idx]

        labels = sampled["label_union"].to_numpy(dtype=int)
        if np.unique(labels).size < 2:
            continue
        causal = sampled["causal_score"].to_numpy(dtype=float)
        attention = sampled["attention_score"].to_numpy(dtype=float)
        stats = compute_comparative_stats(labels=labels, causal=causal, attention=attention, top_quantile=top_quantile)
        draws.append({"dataset": dataset_name, "bootstrap_draw": draw_id, **stats})

    draws_df = pd.DataFrame(draws)
    if draws_df.empty:
        raise RuntimeError(f"Bootstrap produced no valid draws for dataset={dataset_name}")

    ratio = draws_df["relative_cells_causal_vs_attention"].to_numpy(dtype=float)
    auroc_diff = draws_df["auroc_difference_causal_minus_attention"].to_numpy(dtype=float)
    auprc_diff = draws_df["auprc_difference_causal_minus_attention"].to_numpy(dtype=float)
    d_diff = draws_df["d_difference_causal_minus_attention"].to_numpy(dtype=float)
    lift_diff = draws_df["top_quantile_lift_difference_causal_minus_attention"].to_numpy(dtype=float)

    summary = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "n_bootstrap_valid": int(draws_df.shape[0]),
                "relative_cells_point_median": float(np.nanmedian(ratio)),
                "relative_cells_ci_low_95": float(np.nanquantile(ratio, 0.025)),
                "relative_cells_ci_high_95": float(np.nanquantile(ratio, 0.975)),
                "relative_cells_prob_lt_1": float(np.nanmean(ratio < 1.0)),
                "d_diff_point_median": float(np.nanmedian(d_diff)),
                "d_diff_ci_low_95": float(np.nanquantile(d_diff, 0.025)),
                "d_diff_ci_high_95": float(np.nanquantile(d_diff, 0.975)),
                "auroc_diff_point_median": float(np.nanmedian(auroc_diff)),
                "auroc_diff_ci_low_95": float(np.nanquantile(auroc_diff, 0.025)),
                "auroc_diff_ci_high_95": float(np.nanquantile(auroc_diff, 0.975)),
                "auprc_diff_point_median": float(np.nanmedian(auprc_diff)),
                "auprc_diff_ci_low_95": float(np.nanquantile(auprc_diff, 0.025)),
                "auprc_diff_ci_high_95": float(np.nanquantile(auprc_diff, 0.975)),
                "lift_diff_point_median": float(np.nanmedian(lift_diff)),
                "lift_diff_ci_low_95": float(np.nanquantile(lift_diff, 0.025)),
                "lift_diff_ci_high_95": float(np.nanquantile(lift_diff, 0.975)),
            }
        ]
    )
    return draws_df, summary


def permute_labels(
    df: pd.DataFrame,
    rng: np.random.Generator,
    cluster_cols: list[str] | None = None,
) -> np.ndarray:
    """Permute labels row-wise or at the cluster level."""
    if cluster_cols is None:
        return rng.permutation(df["label_union"].to_numpy(dtype=int))

    cluster_label_df = df[cluster_cols + ["label_union"]].drop_duplicates()
    # Defensive check: each cluster must map to exactly one label.
    cluster_counts = cluster_label_df.groupby(cluster_cols, as_index=False)["label_union"].nunique()
    if int(cluster_counts["label_union"].max()) > 1:
        raise ValueError("Cluster columns do not define a unique label per cluster.")

    shuffled = cluster_label_df.copy()
    shuffled["label_perm"] = rng.permutation(cluster_label_df["label_union"].to_numpy(dtype=int))
    merged = df.merge(shuffled[cluster_cols + ["label_perm"]], on=cluster_cols, how="left")
    return merged["label_perm"].to_numpy(dtype=int)


def permutation_controls(
    dataset_name: str,
    df: pd.DataFrame,
    n_permutations: int,
    seed: int,
    top_quantile: float,
    cluster_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run adversarial label-permutation controls for one dataset."""
    rng = np.random.default_rng(seed)
    labels = df["label_union"].to_numpy(dtype=int)
    causal = df["causal_score"].to_numpy(dtype=float)
    attention = df["attention_score"].to_numpy(dtype=float)

    observed = compute_comparative_stats(labels=labels, causal=causal, attention=attention, top_quantile=top_quantile)

    draws: list[dict[str, float | int | str]] = []
    for draw_id in range(n_permutations):
        labels_perm = permute_labels(df=df, rng=rng, cluster_cols=cluster_cols)
        if np.unique(labels_perm).size < 2:
            continue
        stats = compute_comparative_stats(
            labels=labels_perm,
            causal=causal,
            attention=attention,
            top_quantile=top_quantile,
        )
        draws.append({"dataset": dataset_name, "permutation_draw": draw_id, **stats})

    perm_df = pd.DataFrame(draws)
    if perm_df.empty:
        raise RuntimeError(f"No valid permutation draws for dataset={dataset_name}")

    def one_sided_high(column: str) -> float:
        values = perm_df[column].to_numpy(dtype=float)
        return float((np.sum(values >= observed[column]) + 1) / (values.size + 1))

    def one_sided_low(column: str) -> float:
        values = perm_df[column].to_numpy(dtype=float)
        return float((np.sum(values <= observed[column]) + 1) / (values.size + 1))

    def two_sided(column: str) -> float:
        values = perm_df[column].to_numpy(dtype=float)
        return float((np.sum(np.abs(values) >= abs(observed[column])) + 1) / (values.size + 1))

    summary = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "n_permutations_valid": int(perm_df.shape[0]),
                "top_quantile": top_quantile,
                "observed_ratio": observed["relative_cells_causal_vs_attention"],
                "ratio_pvalue_low": one_sided_low("relative_cells_causal_vs_attention"),
                "observed_d_diff": observed["d_difference_causal_minus_attention"],
                "d_diff_pvalue_high": one_sided_high("d_difference_causal_minus_attention"),
                "observed_lift_causal": observed["top_quantile_lift_causal"],
                "lift_causal_pvalue_high": one_sided_high("top_quantile_lift_causal"),
                "observed_lift_attention": observed["top_quantile_lift_attention"],
                "lift_attention_pvalue_high": one_sided_high("top_quantile_lift_attention"),
                "observed_lift_diff": observed["top_quantile_lift_difference_causal_minus_attention"],
                "lift_diff_pvalue_high": one_sided_high("top_quantile_lift_difference_causal_minus_attention"),
                "observed_auroc_diff": observed["auroc_difference_causal_minus_attention"],
                "auroc_diff_pvalue_two_sided": two_sided("auroc_difference_causal_minus_attention"),
                "observed_auprc_diff": observed["auprc_difference_causal_minus_attention"],
                "auprc_diff_pvalue_two_sided": two_sided("auprc_difference_causal_minus_attention"),
            }
        ]
    )
    return perm_df, summary


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.empty(n, dtype=float)
    prev = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        value = min(prev, ranked[idx] * n / rank)
        adjusted_ranked[idx] = value
        prev = value
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def quantile_sensitivity(
    dataset_name: str,
    df: pd.DataFrame,
    quantiles: list[float],
    n_permutations: int,
    seed: int,
    cluster_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Assess top-quantile enrichment sensitivity with permutation p-values."""
    rng = np.random.default_rng(seed)
    labels = df["label_union"].to_numpy(dtype=int)
    causal = df["causal_score"].to_numpy(dtype=float)
    attention = df["attention_score"].to_numpy(dtype=float)

    rows: list[dict[str, float | str]] = []
    for quantile in quantiles:
        obs_causal = top_quantile_lift(causal, labels, quantile=quantile)
        obs_attention = top_quantile_lift(attention, labels, quantile=quantile)

        null_causal: list[float] = []
        null_attention: list[float] = []
        for _ in range(n_permutations):
            labels_perm = permute_labels(df=df, rng=rng, cluster_cols=cluster_cols)
            null_causal.append(top_quantile_lift(causal, labels_perm, quantile=quantile))
            null_attention.append(top_quantile_lift(attention, labels_perm, quantile=quantile))

        null_c = np.array(null_causal, dtype=float)
        null_a = np.array(null_attention, dtype=float)
        p_c = float((np.sum(null_c >= obs_causal) + 1) / (null_c.size + 1))
        p_a = float((np.sum(null_a >= obs_attention) + 1) / (null_a.size + 1))

        rows.append(
            {
                "dataset": dataset_name,
                "quantile": quantile,
                "family": "causal-like",
                "observed_lift": obs_causal,
                "null_lift_mean": float(null_c.mean()),
                "null_lift_ci_low_95": float(np.quantile(null_c, 0.025)),
                "null_lift_ci_high_95": float(np.quantile(null_c, 0.975)),
                "pvalue_high": p_c,
            }
        )
        rows.append(
            {
                "dataset": dataset_name,
                "quantile": quantile,
                "family": "attention-like",
                "observed_lift": obs_attention,
                "null_lift_mean": float(null_a.mean()),
                "null_lift_ci_low_95": float(np.quantile(null_a, 0.025)),
                "null_lift_ci_high_95": float(np.quantile(null_a, 0.975)),
                "pvalue_high": p_a,
            }
        )

    result = pd.DataFrame(rows)
    result["pvalue_bh_within_dataset_family"] = np.nan
    for (_, family), group in result.groupby(["dataset", "family"]):
        adjusted = bh_adjust(group["pvalue_high"].to_numpy(dtype=float))
        result.loc[group.index, "pvalue_bh_within_dataset_family"] = adjusted
    return result.sort_values(["dataset", "family", "quantile"]).reset_index(drop=True)


def leave_one_seed_out(
    dataset_name: str,
    df: pd.DataFrame,
    top_quantile: float,
) -> pd.DataFrame:
    """Run leave-one-seed-out stability summaries."""
    rows: list[dict[str, float | int | str]] = []
    for held_out_seed in sorted(df["seed"].dropna().unique()):
        subset = df[df["seed"] != held_out_seed].reset_index(drop=True)
        labels = subset["label_union"].to_numpy(dtype=int)
        stats = compute_comparative_stats(
            labels=labels,
            causal=subset["causal_score"].to_numpy(dtype=float),
            attention=subset["attention_score"].to_numpy(dtype=float),
            top_quantile=top_quantile,
        )
        rows.append(
            {
                "dataset": dataset_name,
                "held_out_seed": held_out_seed,
                "n_rows": int(subset.shape[0]),
                "label_prevalence": float(labels.mean()),
                **stats,
            }
        )
    return pd.DataFrame(rows).sort_values("held_out_seed")


def seed_breakdown(
    dataset_name: str,
    df: pd.DataFrame,
    top_quantile: float,
) -> pd.DataFrame:
    """Per-seed summaries without seed exclusion."""
    rows: list[dict[str, float | int | str]] = []
    for seed in sorted(df["seed"].dropna().unique()):
        subset = df[df["seed"] == seed].reset_index(drop=True)
        labels = subset["label_union"].to_numpy(dtype=int)
        stats = compute_comparative_stats(
            labels=labels,
            causal=subset["causal_score"].to_numpy(dtype=float),
            attention=subset["attention_score"].to_numpy(dtype=float),
            top_quantile=top_quantile,
        )
        rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "n_rows": int(subset.shape[0]),
                "label_prevalence": float(labels.mean()),
                **stats,
            }
        )
    return pd.DataFrame(rows).sort_values("seed")


def plot_bootstrap(draws_df: pd.DataFrame, output_path: Path) -> None:
    """Plot bootstrap distributions for ratio and comparative metric deltas."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
    colors = {
        "edge_level": "#1f77b4",
        "edge_tissue_level": "#d62728",
        "edge_tissue_cluster_bootstrap": "#2ca02c",
    }

    for dataset, group in draws_df.groupby("dataset"):
        color = colors.get(dataset, "#555555")
        axes[0].hist(
            group["relative_cells_causal_vs_attention"].to_numpy(dtype=float),
            bins=45,
            alpha=0.35,
            label=dataset,
            color=color,
            edgecolor="white",
        )
        axes[1].hist(
            group["d_difference_causal_minus_attention"].to_numpy(dtype=float),
            bins=45,
            alpha=0.35,
            label=dataset,
            color=color,
            edgecolor="white",
        )
        axes[2].hist(
            group["auprc_difference_causal_minus_attention"].to_numpy(dtype=float),
            bins=45,
            alpha=0.35,
            label=dataset,
            color=color,
            edgecolor="white",
        )

    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[2].axvline(0.0, color="black", linestyle="--", linewidth=1.2)

    axes[0].set_title("Relative cell-budget proxy")
    axes[0].set_xlabel("n*_causal / n*_attention")
    axes[0].set_ylabel("Bootstrap count")

    axes[1].set_title("Standardized-separation difference")
    axes[1].set_xlabel("d_causal - d_attention")
    axes[1].set_ylabel("Bootstrap count")

    axes[2].set_title("AUPRC difference")
    axes[2].set_xlabel("causal - attention")
    axes[2].set_ylabel("Bootstrap count")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_quantile_sweep(sensitivity_df: pd.DataFrame, output_path: Path) -> None:
    """Plot enrichment-lift sensitivity across quantiles."""
    datasets = list(sensitivity_df["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.0 * len(datasets), 4.2), constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]

    colors = {"causal-like": "#d62728", "attention-like": "#1f77b4"}
    for axis, dataset in zip(axes, datasets):
        subset = sensitivity_df[sensitivity_df["dataset"] == dataset]
        for family in ["causal-like", "attention-like"]:
            fam = subset[subset["family"] == family].sort_values("quantile")
            axis.plot(
                fam["quantile"],
                fam["observed_lift"],
                marker="o",
                linewidth=2.0,
                color=colors[family],
                label=family,
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.1)
        axis.set_title(dataset)
        axis.set_xlabel("Top-quantile threshold")
        axis.set_ylabel("Observed lift over prevalence")
        axis.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run matched calibration, save tables/figures, and print summary."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = args.output_root
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    quantiles = sorted(set(float(q) for q in args.quantile_sweep))
    if any(q <= 0.0 or q >= 1.0 for q in quantiles):
        raise ValueError("All quantiles must be strictly between 0 and 1.")

    files = input_files(repo_root=repo_root)
    merged = load_seed_tables(files=files)
    edge_df = build_edge_level_table(merged=merged)
    edge_tissue_df = build_edge_tissue_table(merged=merged)

    edge_metrics = summarize_dataset(dataset_name="edge_level", df=edge_df, top_quantile=args.top_quantile)
    edge_tissue_metrics = summarize_dataset(
        dataset_name="edge_tissue_level", df=edge_tissue_df, top_quantile=args.top_quantile
    )
    family_metrics = pd.concat([edge_metrics, edge_tissue_metrics], ignore_index=True)

    edge_draws, edge_summary = bootstrap_comparison(
        dataset_name="edge_level",
        df=edge_df,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        top_quantile=args.top_quantile,
        cluster_cols=None,
    )
    edge_tissue_draws, edge_tissue_summary = bootstrap_comparison(
        dataset_name="edge_tissue_level",
        df=edge_tissue_df,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 1,
        top_quantile=args.top_quantile,
        cluster_cols=None,
    )
    edge_tissue_cluster_draws, edge_tissue_cluster_summary = bootstrap_comparison(
        dataset_name="edge_tissue_cluster_bootstrap",
        df=edge_tissue_df,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 2,
        top_quantile=args.top_quantile,
        cluster_cols=["seed", "source", "target"],
    )

    bootstrap_draws = pd.concat(
        [edge_draws, edge_tissue_draws, edge_tissue_cluster_draws],
        ignore_index=True,
    )
    projection_summary = pd.concat(
        [edge_summary, edge_tissue_summary, edge_tissue_cluster_summary],
        ignore_index=True,
    )

    edge_perm_draws, edge_perm_summary = permutation_controls(
        dataset_name="edge_level_rowwise_permutation",
        df=edge_df,
        n_permutations=args.n_permutations,
        seed=args.seed + 10,
        top_quantile=args.top_quantile,
        cluster_cols=None,
    )
    edge_tissue_perm_draws, edge_tissue_perm_summary = permutation_controls(
        dataset_name="edge_tissue_rowwise_permutation",
        df=edge_tissue_df,
        n_permutations=args.n_permutations,
        seed=args.seed + 11,
        top_quantile=args.top_quantile,
        cluster_cols=None,
    )
    edge_tissue_cluster_perm_draws, edge_tissue_cluster_perm_summary = permutation_controls(
        dataset_name="edge_tissue_cluster_permutation",
        df=edge_tissue_df,
        n_permutations=args.n_permutations,
        seed=args.seed + 12,
        top_quantile=args.top_quantile,
        cluster_cols=["seed", "source", "target"],
    )
    permutation_draws = pd.concat(
        [edge_perm_draws, edge_tissue_perm_draws, edge_tissue_cluster_perm_draws],
        ignore_index=True,
    )
    permutation_summary = pd.concat(
        [edge_perm_summary, edge_tissue_perm_summary, edge_tissue_cluster_perm_summary],
        ignore_index=True,
    )

    quantile_edge = quantile_sensitivity(
        dataset_name="edge_level",
        df=edge_df,
        quantiles=quantiles,
        n_permutations=args.n_permutations,
        seed=args.seed + 20,
        cluster_cols=None,
    )
    quantile_edge_tissue_cluster = quantile_sensitivity(
        dataset_name="edge_tissue_cluster_permutation",
        df=edge_tissue_df,
        quantiles=quantiles,
        n_permutations=args.n_permutations,
        seed=args.seed + 21,
        cluster_cols=["seed", "source", "target"],
    )
    quantile_sensitivity_df = pd.concat([quantile_edge, quantile_edge_tissue_cluster], ignore_index=True)

    loso_edge = leave_one_seed_out(dataset_name="edge_level", df=edge_df, top_quantile=args.top_quantile)
    loso_edge_tissue = leave_one_seed_out(
        dataset_name="edge_tissue_level",
        df=edge_tissue_df,
        top_quantile=args.top_quantile,
    )
    per_seed_edge = seed_breakdown(dataset_name="edge_level", df=edge_df, top_quantile=args.top_quantile)

    edge_df.to_csv(table_dir / "matched_real_data_edge_level.csv", index=False)
    edge_tissue_df.to_csv(table_dir / "matched_real_data_edge_tissue_level.csv", index=False)
    family_metrics.to_csv(table_dir / "matched_real_data_family_metrics.csv", index=False)
    bootstrap_draws.to_csv(table_dir / "matched_real_data_bootstrap_draws.csv", index=False)
    projection_summary.to_csv(table_dir / "matched_real_data_projection_summary.csv", index=False)
    permutation_draws.to_csv(table_dir / "matched_real_data_permutation_draws.csv", index=False)
    permutation_summary.to_csv(table_dir / "matched_real_data_permutation_controls.csv", index=False)
    quantile_sensitivity_df.to_csv(table_dir / "matched_real_data_quantile_sensitivity.csv", index=False)
    loso_edge.to_csv(table_dir / "matched_real_data_loso_edge.csv", index=False)
    loso_edge_tissue.to_csv(table_dir / "matched_real_data_loso_edge_tissue.csv", index=False)
    per_seed_edge.to_csv(table_dir / "matched_real_data_seed_breakdown.csv", index=False)

    plot_bootstrap(draws_df=bootstrap_draws, output_path=figure_dir / "fig_matched_real_data_bootstrap.png")
    plot_quantile_sweep(
        sensitivity_df=quantile_sensitivity_df,
        output_path=figure_dir / "fig_matched_real_data_quantile_sweep.png",
    )

    print("Matched real-data calibration complete.")
    print(f"Edge-level rows: {edge_df.shape[0]}; edge-tissue rows: {edge_tissue_df.shape[0]}")
    print("Bootstrap projection summary:")
    print(projection_summary.to_string(index=False))
    print("Permutation control summary:")
    print(permutation_summary.to_string(index=False))


if __name__ == "__main__":
    main()
