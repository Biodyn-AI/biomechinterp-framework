#!/usr/bin/env python3
"""Generate derived tables/figures for scaling-failure workshop writing.

This script converts the verified MVP snapshot values into reproducible
subproject-local artifacts. It does not re-run heavy model inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def _fit_loglog_slope(cells: np.ndarray, values: np.ndarray, eps: float = 1e-12) -> float:
    """Fit a log-log slope, clipping near-zero values to keep the fit defined."""
    x = np.asarray(cells, dtype=float)
    y = np.asarray(values, dtype=float)
    y = np.maximum(y, eps)
    lx = np.log10(x)
    ly = np.log10(y)
    slope, _intercept = np.polyfit(lx, ly, 1)
    return float(slope)


def _plot_scaling(kidney_scaling: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for tier, tier_df in kidney_scaling.groupby("model_tier"):
        tier_df = tier_df.sort_values("cells")
        ax.plot(tier_df["cells"], tier_df["f1"], marker="o", label=tier)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cells (log scale)")
    ax.set_ylabel("F1 (log scale)")
    ax.set_title("Kidney Scaling: Attention GRN F1 vs Cell Count")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Model tier")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_robustness(robustness: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for tier, tier_df in robustness.groupby("model_tier"):
        tier_df = tier_df.sort_values("cells")
        axes[0].plot(tier_df["cells"], tier_df["jaccard_mean"], marker="o", label=tier)
        axes[1].plot(tier_df["cells"], tier_df["spearman_mean"], marker="o", label=tier)
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("Cells (log scale)")
    axes[0].set_ylabel("Jaccard mean")
    axes[0].set_title("Edge-set overlap stability")
    axes[1].set_ylabel("Spearman mean")
    axes[1].set_title("Score-rank stability")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].legend(title="Model tier")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_policy(policy: pd.DataFrame, out_path: Path) -> None:
    policy = policy.copy()
    policy["best_percentile"] = policy[["pct90", "pct95"]].max(axis=1)
    policy["topk_full_hvg"] = policy["full_hvg_topk100"]

    plot_cols = ["topk_full_hvg", "pct90", "best_percentile", "tf_sources_topk100", "tf_sources_targets_topk100"]
    labels = ["top-k/full", "pct90/full", "best pct/full", "top-k/tf-src", "top-k/tf-src+tgt"]
    x = np.arange(len(policy["model_tier"]))
    width = 0.14

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    for i, (col, label) in enumerate(zip(plot_cols, labels, strict=True)):
        ax.bar(x + (i - 2) * width, policy[col], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(policy["model_tier"])
    ax.set_yscale("log")
    ax.set_ylabel("F1 (log scale)")
    ax.set_title("Kidney (1000 cells): Inference policy sensitivity")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_baselines(baselines: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(baselines["group"]))
    ax.bar(x, baselines["f1"], width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(baselines["group"], rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("F1 (log scale)")
    ax.set_title("Baseline comparison snapshot (1000-cell context)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    reports_dir = root / "reports"
    data_dir = reports_dir / "data"
    fig_dir = reports_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot values are copied from the verified workshop MVP summary.
    kidney_scaling = pd.DataFrame(
        [
            {"model_tier": "small", "cells": 200, "f1": 9.4e-05},
            {"model_tier": "small", "cells": 1000, "f1": 3.5e-05},
            {"model_tier": "small", "cells": 3000, "f1": 1.4e-05},
            {"model_tier": "medium", "cells": 200, "f1": 9.7e-05},
            {"model_tier": "medium", "cells": 1000, "f1": 4.9e-05},
            {"model_tier": "medium", "cells": 3000, "f1": 7.0e-06},
            {"model_tier": "large", "cells": 200, "f1": 5.2e-05},
            {"model_tier": "large", "cells": 1000, "f1": 0.0},
            {"model_tier": "large", "cells": 3000, "f1": 0.0},
        ]
    )

    robustness = pd.DataFrame(
        [
            {"model_tier": "small", "cells": 200, "jaccard_mean": 0.573, "spearman_mean": 0.212, "n_pairs": 3},
            {"model_tier": "small", "cells": 1000, "jaccard_mean": 0.306, "spearman_mean": -0.063, "n_pairs": 3},
            {"model_tier": "small", "cells": 3000, "jaccard_mean": 0.259, "spearman_mean": -0.085, "n_pairs": 3},
            {"model_tier": "medium", "cells": 200, "jaccard_mean": 0.563, "spearman_mean": 0.184, "n_pairs": 3},
            {"model_tier": "medium", "cells": 1000, "jaccard_mean": 0.293, "spearman_mean": -0.110, "n_pairs": 3},
            {"model_tier": "medium", "cells": 3000, "jaccard_mean": 0.242, "spearman_mean": -0.148, "n_pairs": 3},
            {"model_tier": "large", "cells": 200, "jaccard_mean": 0.537, "spearman_mean": 0.084, "n_pairs": 3},
            {"model_tier": "large", "cells": 1000, "jaccard_mean": 0.285, "spearman_mean": -0.130, "n_pairs": 3},
            {"model_tier": "large", "cells": 3000, "jaccard_mean": 0.249, "spearman_mean": -0.126, "n_pairs": 3},
        ]
    )

    cross_tissue_large = pd.DataFrame(
        [
            {"tissue": "kidney", "f1": 0.0, "jaccard_mean": 0.285, "spearman_mean": -0.130},
            {"tissue": "lung", "f1": 9.0e-05, "jaccard_mean": 0.382, "spearman_mean": -0.035},
            {"tissue": "immune", "f1": 8.6e-05, "jaccard_mean": 0.345, "spearman_mean": -0.057},
        ]
    )

    policy = pd.DataFrame(
        [
            {
                "model_tier": "small",
                "full_hvg_topk100": 3.5e-05,
                "tf_sources_topk100": 1.6e-03,
                "tf_sources_targets_topk100": 1.40e-02,
                "pct90": 2.14e-04,
                "pct95": 1.36e-04,
            },
            {
                "model_tier": "medium",
                "full_hvg_topk100": 4.9e-05,
                "tf_sources_topk100": 2.3e-03,
                "tf_sources_targets_topk100": 1.50e-02,
                "pct90": 2.10e-04,
                "pct95": 2.16e-04,
            },
            {
                "model_tier": "large",
                "full_hvg_topk100": 0.0,
                "tf_sources_topk100": 0.0,
                "tf_sources_targets_topk100": 1.41e-02,
                "pct90": 1.89e-04,
                "pct95": 8.6e-05,
            },
        ]
    )

    baselines = pd.DataFrame(
        [
            {"group": "attention-kidney-mean", "f1": np.mean([3.5e-05, 4.9e-05, 0.0])},
            {"group": "coexpr-kidney", "f1": 1.28e-04},
            {"group": "grnboost2-kidney-seed1", "f1": 2.93e-04},
            {"group": "attention-large-lung", "f1": 9.0e-05},
            {"group": "coexpr-lung", "f1": 3.48e-04},
            {"group": "attention-large-immune", "f1": 8.6e-05},
            {"group": "coexpr-immune", "f1": 2.71e-04},
        ]
    )

    kidney_scaling.to_csv(data_dir / "snapshot_kidney_scaling_f1.csv", index=False)
    robustness.to_csv(data_dir / "snapshot_kidney_robustness.csv", index=False)
    cross_tissue_large.to_csv(data_dir / "snapshot_cross_tissue_large_1000.csv", index=False)
    policy.to_csv(data_dir / "snapshot_policy_kidney_1000.csv", index=False)
    baselines.to_csv(data_dir / "snapshot_baselines.csv", index=False)

    derived_rows: list[dict[str, float | str]] = []

    for tier, tier_df in kidney_scaling.groupby("model_tier"):
        row200 = float(tier_df.loc[tier_df["cells"] == 200, "f1"].iloc[0])
        row1000 = float(tier_df.loc[tier_df["cells"] == 1000, "f1"].iloc[0])
        row3000 = float(tier_df.loc[tier_df["cells"] == 3000, "f1"].iloc[0])
        slope = _fit_loglog_slope(tier_df["cells"].to_numpy(), tier_df["f1"].to_numpy())
        derived_rows.extend(
            [
                {
                    "analysis": "scaling_drop",
                    "group": tier,
                    "comparison": "f1_200_to_1000_pct_change",
                    "value": ((row1000 - row200) / row200) * 100.0 if row200 > 0 else np.nan,
                },
                {
                    "analysis": "scaling_drop",
                    "group": tier,
                    "comparison": "f1_1000_to_3000_pct_change",
                    "value": ((row3000 - row1000) / row1000) * 100.0 if row1000 > 0 else np.nan,
                },
                {
                    "analysis": "scaling_fit",
                    "group": tier,
                    "comparison": "loglog_slope_cells_vs_f1",
                    "value": slope,
                },
            ]
        )

    for tier, tier_df in robustness.groupby("model_tier"):
        j200 = float(tier_df.loc[tier_df["cells"] == 200, "jaccard_mean"].iloc[0])
        j3000 = float(tier_df.loc[tier_df["cells"] == 3000, "jaccard_mean"].iloc[0])
        s200 = float(tier_df.loc[tier_df["cells"] == 200, "spearman_mean"].iloc[0])
        s3000 = float(tier_df.loc[tier_df["cells"] == 3000, "spearman_mean"].iloc[0])
        derived_rows.extend(
            [
                {
                    "analysis": "robustness_drop",
                    "group": tier,
                    "comparison": "jaccard_200_to_3000_pct_change",
                    "value": ((j3000 - j200) / j200) * 100.0,
                },
                {
                    "analysis": "robustness_drop",
                    "group": tier,
                    "comparison": "spearman_200_to_3000_delta",
                    "value": s3000 - s200,
                },
            ]
        )

    for _idx, row in policy.iterrows():
        full = float(row["full_hvg_topk100"])
        best_pct = max(float(row["pct90"]), float(row["pct95"]))
        tf_sources = float(row["tf_sources_topk100"])
        tf_sources_targets = float(row["tf_sources_targets_topk100"])
        tier = str(row["model_tier"])
        derived_rows.extend(
            [
                {
                    "analysis": "policy_effect",
                    "group": tier,
                    "comparison": "best_percentile_minus_full_abs",
                    "value": best_pct - full,
                },
                {
                    "analysis": "policy_effect",
                    "group": tier,
                    "comparison": "best_percentile_over_full_ratio",
                    "value": (best_pct / full) if full > 0 else np.nan,
                },
                {
                    "analysis": "policy_effect",
                    "group": tier,
                    "comparison": "tf_sources_over_full_ratio",
                    "value": (tf_sources / full) if full > 0 else np.nan,
                },
                {
                    "analysis": "policy_effect",
                    "group": tier,
                    "comparison": "tf_sources_targets_over_full_ratio",
                    "value": (tf_sources_targets / full) if full > 0 else np.nan,
                },
            ]
        )

    kidney_attention_mean = float(np.mean([3.5e-05, 4.9e-05, 0.0]))
    derived_rows.extend(
        [
            {
                "analysis": "baseline_gap",
                "group": "kidney",
                "comparison": "coexpression_over_attention_mean_ratio",
                "value": 1.28e-04 / kidney_attention_mean if kidney_attention_mean > 0 else np.nan,
            },
            {
                "analysis": "baseline_gap",
                "group": "kidney",
                "comparison": "grnboost2_over_attention_mean_ratio",
                "value": 2.93e-04 / kidney_attention_mean if kidney_attention_mean > 0 else np.nan,
            },
            {
                "analysis": "baseline_gap",
                "group": "lung_large",
                "comparison": "coexpression_over_attention_ratio",
                "value": 3.48e-04 / 9.0e-05,
            },
            {
                "analysis": "baseline_gap",
                "group": "immune_large",
                "comparison": "coexpression_over_attention_ratio",
                "value": 2.71e-04 / 8.6e-05,
            },
        ]
    )

    derived = pd.DataFrame(derived_rows)
    derived.to_csv(data_dir / "snapshot_derived_effects.csv", index=False)

    _plot_scaling(kidney_scaling, fig_dir / "fig_snapshot_scaling_f1_kidney.png")
    _plot_robustness(robustness, fig_dir / "fig_snapshot_robustness_kidney.png")
    _plot_policy(policy, fig_dir / "fig_snapshot_policy_kidney_1000.png")
    _plot_baselines(baselines, fig_dir / "fig_snapshot_baselines.png")

    print(f"Wrote snapshot artifacts to: {reports_dir}")


if __name__ == "__main__":
    main()
