#!/usr/bin/env python3
"""Iterator-pass stress tests for direct interaction measurements.

This script computes uncertainty-aware residual-explanation diagnostics
from already-generated per-cell intervention outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 4000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    n = values.size
    mean = float(np.mean(values))
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = rng.choice(values, size=n, replace=True)
        samples[i] = float(np.mean(draw))
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return mean, lo, hi


def compute_pairwise_ratio_ci(
    tables_dir: Path,
    rng: np.random.Generator,
) -> pd.DataFrame:
    cells = pd.read_csv(tables_dir / "paper_quality_pairwise_measured_cells.csv")
    summary = pd.read_csv(tables_dir / "paper_quality_pairwise_measured_summary.csv")
    residual = pd.read_csv(tables_dir / "paper_quality_pairwise_residual_explanation.csv")

    best = (
        summary.sort_values("interaction_raw_abs_mean", ascending=False)
        .groupby(["run_id", "pair"], as_index=False)
        .first()[["run_id", "pair", "component_i", "component_j"]]
    )

    rows = []
    for row in best.itertuples(index=False):
        subset = cells[
            (cells["run_id"] == row.run_id)
            & (cells["pair"] == row.pair)
            & (cells["component_i"] == row.component_i)
            & (cells["component_j"] == row.component_j)
        ].copy()
        values = subset["interaction_raw_abs"].to_numpy(dtype=float)
        mean_abs, ci_low, ci_high = _bootstrap_mean_ci(values, rng)
        residual_abs = float(
            residual.loc[
                (residual["run_id"] == row.run_id) & (residual["pair"] == row.pair),
                "residual_abs",
            ].iloc[0]
        )
        denom = max(residual_abs, 1e-12)
        rows.append(
            {
                "run_id": row.run_id,
                "pair": row.pair,
                "component_i": row.component_i,
                "component_j": row.component_j,
                "n_cells": int(values.size),
                "best_pair_abs_mean": mean_abs,
                "best_pair_abs_ci_low": ci_low,
                "best_pair_abs_ci_high": ci_high,
                "residual_abs": residual_abs,
                "ratio_mean": mean_abs / denom,
                "ratio_ci_low": ci_low / denom,
                "ratio_ci_high": ci_high / denom,
            }
        )
    return pd.DataFrame(rows).sort_values("ratio_mean", ascending=False).reset_index(drop=True)


def compute_triplet_ratio_ci(
    tables_dir: Path,
    rng: np.random.Generator,
) -> pd.DataFrame:
    cells = pd.read_csv(tables_dir / "paper_quality_triplet_measured_cells.csv")
    residual = pd.read_csv(tables_dir / "paper_quality_triplet_residual_explanation.csv")

    rows = []
    for (run_id, pair), subset in cells.groupby(["run_id", "pair"], sort=False):
        values = subset["triple_interaction_abs"].to_numpy(dtype=float)
        mean_abs, ci_low, ci_high = _bootstrap_mean_ci(values, rng)
        residual_abs = float(
            residual.loc[
                (residual["run_id"] == run_id) & (residual["pair"] == pair),
                "residual_abs",
            ].iloc[0]
        )
        denom = max(residual_abs, 1e-12)
        rows.append(
            {
                "run_id": run_id,
                "pair": pair,
                "n_cells": int(values.size),
                "triplet_abs_mean": mean_abs,
                "triplet_abs_ci_low": ci_low,
                "triplet_abs_ci_high": ci_high,
                "residual_abs": residual_abs,
                "ratio_mean": mean_abs / denom,
                "ratio_ci_low": ci_low / denom,
                "ratio_ci_high": ci_high / denom,
            }
        )
    return pd.DataFrame(rows).sort_values("ratio_mean", ascending=False).reset_index(drop=True)


def compute_pair_plus_triplet_ratio_ci(
    tables_dir: Path,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pair_summary = pd.read_csv(tables_dir / "paper_quality_pairwise_measured_summary.csv")
    pair_cells = pd.read_csv(tables_dir / "paper_quality_pairwise_measured_cells.csv")
    triplet_cells = pd.read_csv(tables_dir / "paper_quality_triplet_measured_cells.csv")
    residual = pd.read_csv(tables_dir / "paper_quality_triplet_residual_explanation.csv")

    triplet_pairs = triplet_cells[["run_id", "pair"]].drop_duplicates()
    best_pairs = (
        pair_summary.merge(triplet_pairs, on=["run_id", "pair"], how="inner")
        .sort_values("interaction_raw_abs_mean", ascending=False)
        .groupby(["run_id", "pair"], as_index=False)
        .first()[["run_id", "pair", "component_i", "component_j"]]
    )

    rows = []
    for row in best_pairs.itertuples(index=False):
        pair_subset = pair_cells[
            (pair_cells["run_id"] == row.run_id)
            & (pair_cells["pair"] == row.pair)
            & (pair_cells["component_i"] == row.component_i)
            & (pair_cells["component_j"] == row.component_j)
        ][["cell_ordinal", "interaction_raw_abs"]]
        triplet_subset = triplet_cells[
            (triplet_cells["run_id"] == row.run_id) & (triplet_cells["pair"] == row.pair)
        ][["cell_ordinal", "triple_interaction_abs"]]
        joined = pair_subset.merge(triplet_subset, on="cell_ordinal", how="inner")
        values = (
            joined["interaction_raw_abs"].to_numpy(dtype=float)
            + joined["triple_interaction_abs"].to_numpy(dtype=float)
        )
        mean_abs, ci_low, ci_high = _bootstrap_mean_ci(values, rng)
        residual_abs = float(
            residual.loc[
                (residual["run_id"] == row.run_id) & (residual["pair"] == row.pair),
                "residual_abs",
            ].iloc[0]
        )
        denom = max(residual_abs, 1e-12)
        rows.append(
            {
                "run_id": row.run_id,
                "pair": row.pair,
                "component_i": row.component_i,
                "component_j": row.component_j,
                "n_cells_matched": int(values.size),
                "pair_plus_triplet_abs_mean": mean_abs,
                "pair_plus_triplet_abs_ci_low": ci_low,
                "pair_plus_triplet_abs_ci_high": ci_high,
                "residual_abs": residual_abs,
                "ratio_mean": mean_abs / denom,
                "ratio_ci_low": ci_low / denom,
                "ratio_ci_high": ci_high / denom,
            }
        )
    return pd.DataFrame(rows).sort_values("ratio_mean", ascending=False).reset_index(drop=True)


def compute_measured_budget_sensitivity(tables_dir: Path, pair_plus_triplet: pd.DataFrame) -> pd.DataFrame:
    measured = pd.read_csv(tables_dir / "paper_quality_measured_budget_top1_certification_measured_pairs.csv")
    pairwise = pd.read_csv(tables_dir / "paper_quality_pairwise_residual_explanation.csv")

    merged = measured.merge(
        pairwise[["run_id", "pair", "mean_abs_interaction_raw"]],
        on=["run_id", "pair"],
        how="left",
    ).merge(
        pair_plus_triplet[["run_id", "pair", "pair_plus_triplet_abs_mean"]],
        on=["run_id", "pair"],
        how="left",
    )
    merged["pair_plus_triplet_abs_mean"] = merged["pair_plus_triplet_abs_mean"].fillna(
        merged["best_abs_interaction_raw"]
    )

    budget_rules = [
        ("best_pair", "best_abs_interaction_raw"),
        ("mean_pair", "mean_abs_interaction_raw"),
        ("pair_plus_triplet_if_available", "pair_plus_triplet_abs_mean"),
    ]

    rows = []
    for rule_name, col in budget_rules:
        certified = (merged["required_structural_for_flip"] > merged[col]).astype(int)
        rows.append(
            {
                "budget_rule": rule_name,
                "n_pairs": int(merged.shape[0]),
                "n_certified": int(certified.sum()),
                "certified_rate": float(certified.mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_headline(
    pairwise_ci: pd.DataFrame,
    triplet_ci: pd.DataFrame,
    pair_plus_triplet_ci: pd.DataFrame,
    budget_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pairwise_pairs": int(pairwise_ci.shape[0]),
                "pairwise_max_ratio_ci_high": float(pairwise_ci["ratio_ci_high"].max()),
                "pairwise_pairs_ratio_ci_high_lt_0p10": int((pairwise_ci["ratio_ci_high"] < 0.10).sum()),
                "pairwise_pairs_ratio_ci_high_lt_0p20": int((pairwise_ci["ratio_ci_high"] < 0.20).sum()),
                "triplet_pairs": int(triplet_ci.shape[0]),
                "triplet_max_ratio_ci_high": float(triplet_ci["ratio_ci_high"].max()),
                "pair_plus_triplet_pairs": int(pair_plus_triplet_ci.shape[0]),
                "pair_plus_triplet_max_ratio_ci_high": float(pair_plus_triplet_ci["ratio_ci_high"].max()),
                "measured_budget_min_cert_rate": float(budget_sensitivity["certified_rate"].min()),
                "measured_budget_max_cert_rate": float(budget_sensitivity["certified_rate"].max()),
            }
        ]
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    tables_dir = repo_root / "subproject_22_patching_mediation_bias_bounds" / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    pairwise_ci = compute_pairwise_ratio_ci(tables_dir, rng)
    triplet_ci = compute_triplet_ratio_ci(tables_dir, rng)
    pair_plus_triplet_ci = compute_pair_plus_triplet_ratio_ci(tables_dir, rng)
    budget_sensitivity = compute_measured_budget_sensitivity(tables_dir, pair_plus_triplet_ci)
    headline = compute_headline(pairwise_ci, triplet_ci, pair_plus_triplet_ci, budget_sensitivity)

    out_pairwise = tables_dir / "paper_quality_pairwise_best_ratio_bootstrap_ci.csv"
    out_triplet = tables_dir / "paper_quality_triplet_ratio_bootstrap_ci.csv"
    out_pair_plus_triplet = tables_dir / "paper_quality_pair_plus_triplet_ratio_bootstrap_ci.csv"
    out_budget = tables_dir / "paper_quality_measured_budget_sensitivity.csv"
    out_headline = tables_dir / "paper_quality_direct_interaction_uncertainty_headline.csv"

    pairwise_ci.to_csv(out_pairwise, index=False)
    triplet_ci.to_csv(out_triplet, index=False)
    pair_plus_triplet_ci.to_csv(out_pair_plus_triplet, index=False)
    budget_sensitivity.to_csv(out_budget, index=False)
    headline.to_csv(out_headline, index=False)

    print(out_pairwise)
    print(out_triplet)
    print(out_pair_plus_triplet)
    print(out_budget)
    print(out_headline)


if __name__ == "__main__":
    main()
