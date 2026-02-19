#!/usr/bin/env python3
"""Compute uncertainty and random-overlap controls for bridge benchmark metrics.

This script adds two analysis layers:
1) random-overlap control using a hypergeometric null with matched edge counts
   and BH-adjusted q-values,
2) bootstrap confidence intervals for per-perturbation recall statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path(
            "subproject_07_perturbation_to_network_bridging/reports/artifacts/bridge_benchmark/bridge_eval_metrics.csv"
        ),
        help="Bridge benchmark metrics CSV.",
    )
    parser.add_argument(
        "--reference-summary-csv",
        type=Path,
        default=Path(
            "subproject_07_perturbation_to_network_bridging/reports/artifacts/bridge_benchmark/reference_normalization_summary.csv"
        ),
        help="Reference normalization summary CSV.",
    )
    parser.add_argument(
        "--recall-detail-csv",
        type=Path,
        default=Path(
            "subproject_07_perturbation_to_network_bridging/reports/artifacts/bridge_benchmark/perturbation_recall_concentration_detail.csv"
        ),
        help="Per-perturbation recall detail CSV.",
    )
    parser.add_argument(
        "--processed-h5ad",
        type=Path,
        default=Path(
            "single_cell_mechinterp/outputs/tabula_sapiens_immune_subset_hpn_processed.h5ad"
        ),
        help="Processed immune subset AnnData for candidate-universe size.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_07_perturbation_to_network_bridging/reports/artifacts/bridge_benchmark"
        ),
        help="Output directory.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=2000,
        help="Bootstrap iterations for confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for bootstrap reproducibility.",
    )
    return parser.parse_args()


def resolve_path(repo_root: Path, path_arg: Path) -> Path:
    return path_arg if path_arg.is_absolute() else (repo_root / path_arg)


def infer_true_positives(row: pd.Series, truth_edges: int) -> int:
    """Estimate true positives from precision/recall with stable rounding."""
    tp_from_recall = row["recall"] * truth_edges
    tp_from_precision = row["precision"] * row["pred_edges"]
    tp = int(round((tp_from_recall + tp_from_precision) / 2.0))
    tp = max(0, min(tp, int(truth_edges), int(row["pred_edges"])))
    return tp


def benjamini_hochberg_qvalues(p_values: pd.Series) -> pd.Series:
    """Compute BH-adjusted q-values from a vector of p-values."""
    p = p_values.to_numpy(dtype=float)
    n = p.size
    if n == 0:
        return pd.Series(dtype=float, index=p_values.index)

    order = np.argsort(p)
    ranked_p = p[order]
    q = np.empty(n, dtype=float)

    prev = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        value = ranked_p[idx] * n / rank
        prev = min(prev, value)
        q[idx] = prev

    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return pd.Series(out, index=p_values.index)


def compute_random_overlap_controls(
    metrics: pd.DataFrame, reference_sizes: Dict[str, int], candidate_edges: int
) -> pd.DataFrame:
    rows = []
    for _, row in metrics.iterrows():
        reference = row["reference"]
        truth_edges = int(reference_sizes[reference])
        pred_edges = int(row["pred_edges"])
        tp_obs = infer_true_positives(row, truth_edges)
        expected_tp = pred_edges * truth_edges / candidate_edges
        enrichment = (tp_obs / expected_tp) if expected_tp > 0 else np.nan
        p_ge_tp = float(hypergeom.sf(tp_obs - 1, candidate_edges, truth_edges, pred_edges))
        rows.append(
            {
                "method": row["method"],
                "reference": reference,
                "pred_edges": pred_edges,
                "truth_edges": truth_edges,
                "candidate_edges": candidate_edges,
                "observed_tp": tp_obs,
                "expected_tp_random": expected_tp,
                "tp_enrichment_vs_random": enrichment,
                "p_value_hypergeom_ge_tp": p_ge_tp,
            }
        )
    controls = pd.DataFrame(rows)
    controls["q_value_bh_within_reference"] = (
        controls.groupby("reference", group_keys=False)["p_value_hypergeom_ge_tp"]
        .apply(benjamini_hochberg_qvalues)
        .astype(float)
    )
    return controls


def bootstrap_recall_ci(
    detail: pd.DataFrame, bootstrap_iters: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for method, group in detail.groupby("method"):
        recalls = group["recall"].to_numpy(dtype=float)
        n = recalls.size
        if n == 0:
            continue

        mean_obs = float(recalls.mean())
        nonzero_obs = float((recalls > 0).mean())
        median_obs = float(np.median(recalls))

        mean_samples = np.empty(bootstrap_iters, dtype=float)
        nonzero_samples = np.empty(bootstrap_iters, dtype=float)
        median_samples = np.empty(bootstrap_iters, dtype=float)
        for idx in range(bootstrap_iters):
            sample_idx = rng.integers(0, n, size=n)
            sample = recalls[sample_idx]
            mean_samples[idx] = sample.mean()
            nonzero_samples[idx] = (sample > 0).mean()
            median_samples[idx] = np.median(sample)

        rows.append(
            {
                "method": method,
                "n_perturbations": int(n),
                "mean_recall": mean_obs,
                "mean_recall_ci_low": float(np.quantile(mean_samples, 0.025)),
                "mean_recall_ci_high": float(np.quantile(mean_samples, 0.975)),
                "median_recall": median_obs,
                "median_recall_ci_low": float(np.quantile(median_samples, 0.025)),
                "median_recall_ci_high": float(np.quantile(median_samples, 0.975)),
                "nonzero_fraction": nonzero_obs,
                "nonzero_fraction_ci_low": float(np.quantile(nonzero_samples, 0.025)),
                "nonzero_fraction_ci_high": float(np.quantile(nonzero_samples, 0.975)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_recall", ascending=False)


def render_control_figures(controls: pd.DataFrame, ci_df: pd.DataFrame, out_dir: Path) -> None:
    pert = controls[controls["reference"] == "perturbation_edges"].copy()
    pert = pert.sort_values("tp_enrichment_vs_random", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(
        pert["method"][::-1],
        pert["tp_enrichment_vs_random"][::-1],
        color="#2b8cbe",
    )
    plt.xlabel("True-positive enrichment vs random (matched edge count)")
    plt.ylabel("Method")
    plt.title("Perturbation Reference: Enrichment Above Random Control")
    plt.tight_layout()
    plt.savefig(out_dir / "perturbation_enrichment_top15.png", dpi=220)
    plt.close()

    keep = ci_df.sort_values("mean_recall", ascending=False).head(8).copy()
    keep = keep.sort_values("mean_recall", ascending=True)
    lower = keep["mean_recall"] - keep["mean_recall_ci_low"]
    upper = keep["mean_recall_ci_high"] - keep["mean_recall"]

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x=keep["mean_recall"],
        y=keep["method"],
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#ef6548",
        ecolor="#ef6548",
        capsize=3,
    )
    plt.xlabel("Mean per-perturbation recall (95% bootstrap CI)")
    plt.ylabel("Method")
    plt.title("Uncertainty in Perturbation Coverage (Top Methods)")
    plt.tight_layout()
    plt.savefig(out_dir / "perturbation_recall_bootstrap_ci_top8.png", dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = resolve_path(repo_root, args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = resolve_path(repo_root, args.metrics_csv).resolve()
    reference_summary_path = resolve_path(repo_root, args.reference_summary_csv).resolve()
    recall_detail_path = resolve_path(repo_root, args.recall_detail_csv).resolve()
    processed_h5ad_path = resolve_path(repo_root, args.processed_h5ad).resolve()

    metrics = pd.read_csv(metrics_path)
    reference_summary = pd.read_csv(reference_summary_path)
    recall_detail = pd.read_csv(recall_detail_path)

    reference_sizes = {
        row["reference"]: int(row["normalized_edges"])
        for _, row in reference_summary.iterrows()
    }

    adata = ad.read_h5ad(processed_h5ad_path, backed="r")
    n_genes = int(adata.n_vars)
    adata.file.close()
    candidate_edges = n_genes * (n_genes - 1)

    controls = compute_random_overlap_controls(
        metrics=metrics,
        reference_sizes=reference_sizes,
        candidate_edges=candidate_edges,
    )
    controls.sort_values(
        ["reference", "tp_enrichment_vs_random"],
        ascending=[True, False],
    ).to_csv(out_dir / "random_overlap_controls.csv", index=False)

    ci_df = bootstrap_recall_ci(
        detail=recall_detail,
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
    )
    ci_df.to_csv(out_dir / "perturbation_recall_bootstrap_ci.csv", index=False)

    render_control_figures(controls=controls, ci_df=ci_df, out_dir=out_dir)


if __name__ == "__main__":
    main()
