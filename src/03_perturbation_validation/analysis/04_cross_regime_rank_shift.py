#!/usr/bin/env python3
"""Quantify rank shifts between canonical and independent perturbation regimes.

This analysis addresses reviewer concerns that method conclusions may be tied to a
single reference geometry. It compares method rankings by F1 on:
1) canonical perturbation reference (`perturbation_edges`), and
2) independent perturbation union (`perturbation_independent_union`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from scipy.stats import kendalltau, spearmanr

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
        "--canonical-metrics",
        type=Path,
        default=Path(
            "subproject_07_perturbation_to_network_bridging/reports/artifacts/bridge_benchmark/bridge_eval_metrics.csv"
        ),
        help="Metrics CSV containing canonical perturbation benchmark results.",
    )
    parser.add_argument(
        "--independent-metrics",
        type=Path,
        default=Path(
            "subproject_07_perturbation_to_network_bridging/reports/artifacts/bridge_benchmark/independent_perturbation_metrics.csv"
        ),
        help="Metrics CSV containing independent perturbation benchmark results.",
    )
    parser.add_argument(
        "--canonical-reference",
        type=str,
        default="perturbation_edges",
        help="Reference name in canonical metrics CSV.",
    )
    parser.add_argument(
        "--independent-reference",
        type=str,
        default="perturbation_independent_union",
        help="Reference name in independent metrics CSV.",
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
        "--top-k",
        type=int,
        default=12,
        help="Number of highest-shift methods to plot.",
    )
    return parser.parse_args()


def resolve_path(repo_root: Path, path_arg: Path) -> Path:
    return path_arg if path_arg.is_absolute() else (repo_root / path_arg)


def build_rank_table(
    canonical_df: pd.DataFrame,
    independent_df: pd.DataFrame,
    canonical_reference: str,
    independent_reference: str,
) -> pd.DataFrame:
    canonical = canonical_df[canonical_df["reference"] == canonical_reference].copy()
    independent = independent_df[independent_df["reference"] == independent_reference].copy()

    canonical = canonical[["method", "f1", "precision", "recall", "aupr"]].rename(
        columns={
            "f1": "f1_canonical",
            "precision": "precision_canonical",
            "recall": "recall_canonical",
            "aupr": "aupr_canonical",
        }
    )
    independent = independent[["method", "f1", "precision", "recall", "aupr"]].rename(
        columns={
            "f1": "f1_independent",
            "precision": "precision_independent",
            "recall": "recall_independent",
            "aupr": "aupr_independent",
        }
    )

    merged = canonical.merge(independent, on="method", how="inner")
    merged["rank_canonical"] = (
        merged["f1_canonical"].rank(method="min", ascending=False).astype(int)
    )
    merged["rank_independent"] = (
        merged["f1_independent"].rank(method="min", ascending=False).astype(int)
    )
    merged["delta_rank_independent_minus_canonical"] = (
        merged["rank_independent"] - merged["rank_canonical"]
    )
    merged["abs_delta_rank"] = (
        merged["delta_rank_independent_minus_canonical"].abs().astype(int)
    )
    merged = merged.sort_values(
        ["abs_delta_rank", "rank_independent", "rank_canonical"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return merged


def render_rank_shift_figure(rank_df: pd.DataFrame, out_path: Path, top_k: int) -> None:
    keep = rank_df.head(top_k).copy()
    if keep.empty:
        return
    keep = keep.sort_values("delta_rank_independent_minus_canonical", ascending=True)

    plt.figure(figsize=(11, 6.2))
    colors = ["#2b8cbe" if value < 0 else "#ef6548" for value in keep["delta_rank_independent_minus_canonical"]]
    plt.barh(keep["method"], keep["delta_rank_independent_minus_canonical"], color=colors)
    plt.axvline(0, color="black", linewidth=1, alpha=0.8)
    plt.xlabel("Rank shift: independent - canonical (negative means improved rank on independent panel)")
    plt.ylabel("Method")
    plt.title("Largest Method Rank Shifts Across Perturbation Reference Regimes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = resolve_path(repo_root, args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_path = resolve_path(repo_root, args.canonical_metrics).resolve()
    independent_path = resolve_path(repo_root, args.independent_metrics).resolve()

    canonical_df = pd.read_csv(canonical_path)
    independent_df = pd.read_csv(independent_path)

    rank_df = build_rank_table(
        canonical_df=canonical_df,
        independent_df=independent_df,
        canonical_reference=args.canonical_reference,
        independent_reference=args.independent_reference,
    )
    rank_df.to_csv(out_dir / "cross_regime_rank_shift.csv", index=False)

    if rank_df.shape[0] >= 3:
        rho, rho_p = spearmanr(rank_df["f1_canonical"], rank_df["f1_independent"])
        tau, tau_p = kendalltau(rank_df["f1_canonical"], rank_df["f1_independent"])
    else:
        rho, rho_p, tau, tau_p = float("nan"), float("nan"), float("nan"), float("nan")

    pd.DataFrame(
        [
            {
                "canonical_reference": args.canonical_reference,
                "independent_reference": args.independent_reference,
                "n_methods": int(rank_df.shape[0]),
                "spearman_r_f1": float(rho),
                "spearman_p_value": float(rho_p),
                "kendall_tau_f1": float(tau),
                "kendall_p_value": float(tau_p),
                "max_abs_rank_shift": int(rank_df["abs_delta_rank"].max()) if not rank_df.empty else 0,
            }
        ]
    ).to_csv(out_dir / "cross_regime_rank_shift_summary.csv", index=False)

    render_rank_shift_figure(
        rank_df=rank_df,
        out_path=out_dir / "cross_regime_rank_shift_top12.png",
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
