#!/usr/bin/env python3
"""Confound-aware and uncertainty analyses for subproject 04 artifacts.

This script adds two pieces of evidence that were missing in the prior pass:
1. Metadata-derived confound sensitivity (donor/sample composition weighting).
2. Bootstrap confidence intervals for transfer-regret summaries.

It only depends on committed artifact CSVs and raw H5AD metadata.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _safe_index_length(group: h5py.Group, default_name: str) -> int:
    if "_index" in group:
        return int(group["_index"].shape[0])
    if default_name in group:
        return int(group[default_name].shape[0])
    # Fall back to the first key; H5AD schemas can vary across versions.
    for key in group.keys():
        return int(group[key].shape[0])
    raise ValueError("Unable to infer index length from empty group")


def _entropy_stats(counts: np.ndarray) -> Dict[str, float]:
    total = float(counts.sum())
    if total <= 0:
        return {
            "n_categories": 0.0,
            "dominant_fraction": float("nan"),
            "entropy": float("nan"),
            "entropy_norm": float("nan"),
            "effective_n": float("nan"),
        }

    probs = counts[counts > 0] / total
    entropy = float(-(probs * np.log(probs)).sum())
    n_categories = int((counts > 0).sum())
    entropy_norm = float(entropy / np.log(n_categories)) if n_categories > 1 else 0.0
    return {
        "n_categories": float(n_categories),
        "dominant_fraction": float(counts.max() / total),
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "effective_n": float(math.exp(entropy)),
    }


def _categorical_counts(obs_group: h5py.Group, key: str) -> np.ndarray:
    if key not in obs_group:
        return np.array([], dtype=np.int64)

    obj = obs_group[key]
    if isinstance(obj, h5py.Group) and "codes" in obj and "categories" in obj:
        codes = np.asarray(obj["codes"], dtype=np.int64)
        valid = codes[codes >= 0]
        if valid.size == 0:
            return np.array([], dtype=np.int64)
        n_categories = int(obj["categories"].shape[0])
        return np.bincount(valid, minlength=n_categories)

    # Rare fallback for non-categorical obs columns.
    values = np.asarray(obj).astype(str)
    if values.size == 0:
        return np.array([], dtype=np.int64)
    _, counts = np.unique(values, return_counts=True)
    return counts.astype(np.int64)


def tissue_metadata_summary(tissue_to_h5ad: Dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for tissue, h5ad_path in tissue_to_h5ad.items():
        if not h5ad_path.exists():
            raise FileNotFoundError(f"Missing H5AD for tissue '{tissue}': {h5ad_path}")

        with h5py.File(h5ad_path, "r") as handle:
            obs = handle["obs"]
            var = handle["var"]
            n_cells = _safe_index_length(obs, "sample_number")
            n_genes = _safe_index_length(var, "_index")

            donor_stats = _entropy_stats(_categorical_counts(obs, "donor_id"))
            sample_stats = _entropy_stats(_categorical_counts(obs, "sample_id"))
            batch_stats = _entropy_stats(_categorical_counts(obs, "_scvi_batch"))
            cell_type_stats = _entropy_stats(_categorical_counts(obs, "cell_type"))

            # Composite heterogeneity score keeps interpretation simple:
            # higher values mean less dominance by any single donor/sample.
            diversity_components = [
                1.0 - donor_stats["dominant_fraction"]
                if not math.isnan(donor_stats["dominant_fraction"])
                else float("nan"),
                1.0 - sample_stats["dominant_fraction"]
                if not math.isnan(sample_stats["dominant_fraction"])
                else float("nan"),
                cell_type_stats["entropy_norm"],
            ]
            diversity_components = [x for x in diversity_components if not math.isnan(x)]
            confound_diversity_score = (
                float(np.mean(diversity_components)) if diversity_components else float("nan")
            )

            rows.append(
                {
                    "tissue": tissue,
                    "h5ad_path": str(h5ad_path),
                    "n_cells": float(n_cells),
                    "n_genes": float(n_genes),
                    "n_donors": donor_stats["n_categories"],
                    "donor_dominant_fraction": donor_stats["dominant_fraction"],
                    "donor_entropy_norm": donor_stats["entropy_norm"],
                    "donor_effective_n": donor_stats["effective_n"],
                    "n_samples": sample_stats["n_categories"],
                    "sample_dominant_fraction": sample_stats["dominant_fraction"],
                    "sample_entropy_norm": sample_stats["entropy_norm"],
                    "n_batches": batch_stats["n_categories"],
                    "batch_dominant_fraction": batch_stats["dominant_fraction"],
                    "batch_entropy_norm": batch_stats["entropy_norm"],
                    "n_cell_types": cell_type_stats["n_categories"],
                    "cell_type_entropy_norm": cell_type_stats["entropy_norm"],
                    "confound_diversity_score": confound_diversity_score,
                }
            )
    return pd.DataFrame(rows).sort_values("tissue", ignore_index=True)


def _tissue_weights(metadata_df: pd.DataFrame) -> Dict[str, float]:
    df = metadata_df.set_index("tissue")
    raw = df["confound_diversity_score"].astype(float).copy()
    raw = raw.fillna(raw.mean() if raw.notna().any() else 1.0)
    if (raw <= 0).all():
        raw[:] = 1.0
    weights = raw / raw.sum()
    return {tissue: float(weight) for tissue, weight in weights.items()}


def _bootstrap_mean_ci(
    values: np.ndarray,
    weights: np.ndarray | None,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    means = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        sample_values = values[sample_idx]
        if weights is None:
            means[idx] = float(np.mean(sample_values))
        else:
            sample_weights = weights[sample_idx]
            if np.all(sample_weights == 0):
                means[idx] = float(np.mean(sample_values))
            else:
                means[idx] = float(np.average(sample_values, weights=sample_weights))

    point = float(np.mean(values) if weights is None else np.average(values, weights=weights))
    return point, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def transfer_bootstrap_tables(
    artifact_dir: Path,
    tissue_weight_map: Dict[str, float],
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []

    for budget in ("30", "100"):
        transfer_df = _read_csv(artifact_dir / f"transfer_regret_{budget}_cell.csv").copy()
        transfer_df["target_weight"] = transfer_df["target_tissue"].map(tissue_weight_map).astype(float)
        values = transfer_df["relative_regret"].to_numpy(dtype=float)
        weights = transfer_df["target_weight"].to_numpy(dtype=float)

        point_u, lo_u, hi_u = _bootstrap_mean_ci(values, None, n_boot=n_boot, seed=seed)
        point_w, lo_w, hi_w = _bootstrap_mean_ci(values, weights, n_boot=n_boot, seed=seed)

        summary_rows.extend(
            [
                {
                    "cell_budget": budget,
                    "weighting": "unweighted",
                    "mean_relative_regret": point_u,
                    "ci95_low": lo_u,
                    "ci95_high": hi_u,
                    "n_pairs": int(len(transfer_df)),
                },
                {
                    "cell_budget": budget,
                    "weighting": "confound_weighted_target",
                    "mean_relative_regret": point_w,
                    "ci95_low": lo_w,
                    "ci95_high": hi_w,
                    "n_pairs": int(len(transfer_df)),
                },
            ]
        )

        for source_tissue, group in transfer_df.groupby("source_tissue", sort=True):
            group_values = group["relative_regret"].to_numpy(dtype=float)
            group_weights = group["target_weight"].to_numpy(dtype=float)
            p_u, l_u, h_u = _bootstrap_mean_ci(
                group_values, None, n_boot=n_boot, seed=seed + 11
            )
            p_w, l_w, h_w = _bootstrap_mean_ci(
                group_values, group_weights, n_boot=n_boot, seed=seed + 29
            )
            source_rows.extend(
                [
                    {
                        "cell_budget": budget,
                        "source_tissue": source_tissue,
                        "weighting": "unweighted",
                        "mean_relative_regret": p_u,
                        "ci95_low": l_u,
                        "ci95_high": h_u,
                        "n_targets": int(len(group)),
                    },
                    {
                        "cell_budget": budget,
                        "source_tissue": source_tissue,
                        "weighting": "confound_weighted_target",
                        "mean_relative_regret": p_w,
                        "ci95_low": l_w,
                        "ci95_high": h_w,
                        "n_targets": int(len(group)),
                    },
                ]
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["cell_budget", "weighting"], ignore_index=True
    )
    source_df = pd.DataFrame(source_rows).sort_values(
        ["cell_budget", "source_tissue", "weighting"], ignore_index=True
    )
    return summary_df, source_df


def _policy_benchmark_from_table(
    table_df: pd.DataFrame,
    cell_budget: str,
    tissue_weight_map: Dict[str, float],
) -> pd.DataFrame:
    scores = table_df.set_index("tissue").copy()
    probes = list(scores.columns)
    source_tissues = list(scores.index)
    weights = np.array([tissue_weight_map[tissue] for tissue in source_tissues], dtype=float)
    weights = weights / weights.sum()

    oracle_per_tissue = scores.max(axis=1).astype(float).to_numpy()
    oracle_mean_unweighted = float(np.mean(oracle_per_tissue))
    oracle_mean_weighted = float(np.average(oracle_per_tissue, weights=weights))

    rows: list[dict[str, object]] = []

    for source_tissue in source_tissues:
        source_probe = str(scores.loc[source_tissue].idxmax())
        values = scores[source_probe].astype(float).to_numpy()
        rows.append(
            {
                "cell_budget": cell_budget,
                "policy": f"source_best::{source_tissue}",
                "probe": source_probe,
                "mean_aupr_unweighted": float(np.mean(values)),
                "mean_aupr_weighted": float(np.average(values, weights=weights)),
                "mean_regret_unweighted": float(oracle_mean_unweighted - np.mean(values)),
                "mean_regret_weighted": float(
                    oracle_mean_weighted - np.average(values, weights=weights)
                ),
            }
        )

    selected_probes = [probe for probe in ("coexpression", "coexpression_signed") if probe in probes]
    for probe in selected_probes:
        values = scores[probe].astype(float).to_numpy()
        rows.append(
            {
                "cell_budget": cell_budget,
                "policy": f"fixed_probe::{probe}",
                "probe": probe,
                "mean_aupr_unweighted": float(np.mean(values)),
                "mean_aupr_weighted": float(np.average(values, weights=weights)),
                "mean_regret_unweighted": float(oracle_mean_unweighted - np.mean(values)),
                "mean_regret_weighted": float(
                    oracle_mean_weighted - np.average(values, weights=weights)
                ),
            }
        )

    random_values = scores.mean(axis=1).astype(float).to_numpy()
    rows.append(
        {
            "cell_budget": cell_budget,
            "policy": "random_probe_expected",
            "probe": "uniform_over_probes",
            "mean_aupr_unweighted": float(np.mean(random_values)),
            "mean_aupr_weighted": float(np.average(random_values, weights=weights)),
            "mean_regret_unweighted": float(oracle_mean_unweighted - np.mean(random_values)),
            "mean_regret_weighted": float(
                oracle_mean_weighted - np.average(random_values, weights=weights)
            ),
        }
    )

    return pd.DataFrame(rows).sort_values(
        ["cell_budget", "mean_regret_unweighted", "policy"], ignore_index=True
    )


def run(
    artifact_dir: Path,
    output_dir: Path,
    kidney_h5ad: Path,
    lung_h5ad: Path,
    immune_h5ad: Path,
    n_boot: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tissue_paths = {"kidney": kidney_h5ad, "lung": lung_h5ad, "immune": immune_h5ad}
    metadata_df = tissue_metadata_summary(tissue_paths)
    metadata_df.to_csv(output_dir / "tissue_metadata_heterogeneity.csv", index=False)

    tissue_weights = _tissue_weights(metadata_df)
    weights_df = pd.DataFrame(
        [{"tissue": tissue, "confound_weight": weight} for tissue, weight in tissue_weights.items()]
    ).sort_values("tissue", ignore_index=True)
    weights_df.to_csv(output_dir / "tissue_confound_weights.csv", index=False)

    transfer_summary_df, transfer_source_df = transfer_bootstrap_tables(
        artifact_dir=artifact_dir,
        tissue_weight_map=tissue_weights,
        n_boot=n_boot,
        seed=seed,
    )
    transfer_summary_df.to_csv(output_dir / "transfer_regret_bootstrap_ci.csv", index=False)
    transfer_source_df.to_csv(output_dir / "transfer_regret_source_bootstrap_ci.csv", index=False)

    table_30 = _read_csv(artifact_dir / "mvp_mean_aupr_30_cell.csv")
    table_100 = _read_csv(artifact_dir / "mvp_mean_aupr_100_cell.csv")
    policy_30 = _policy_benchmark_from_table(table_30, "30", tissue_weights)
    policy_100 = _policy_benchmark_from_table(table_100, "100", tissue_weights)
    policy_df = pd.concat([policy_30, policy_100], ignore_index=True)
    policy_df.to_csv(output_dir / "policy_transfer_benchmark_weighted.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run confound-aware and uncertainty analyses for subproject 04."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(
            "/Users/ihorkendiukhov/biodyn-work/subproject_04_cross_tissue_probe_threshold_transfer/"
            "reports/artifacts"
        ),
        help="Directory with derived artifact CSVs (transfer regret, AUPR tables).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/Users/ihorkendiukhov/biodyn-work/subproject_04_cross_tissue_probe_threshold_transfer/"
            "reports/artifacts"
        ),
        help="Directory where confound/uncertainty CSVs will be written.",
    )
    parser.add_argument(
        "--kidney-h5ad",
        type=Path,
        default=Path(
            "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad"
        ),
    )
    parser.add_argument(
        "--lung-h5ad",
        type=Path,
        default=Path(
            "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad"
        ),
    )
    parser.add_argument(
        "--immune-h5ad",
        type=Path,
        default=Path(
            "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad"
        ),
    )
    parser.add_argument("--n-boot", type=int, default=10000, help="Bootstrap iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        kidney_h5ad=args.kidney_h5ad,
        lung_h5ad=args.lung_h5ad,
        immune_h5ad=args.immune_h5ad,
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
