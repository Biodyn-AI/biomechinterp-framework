#!/usr/bin/env python3
"""Regenerate publication-readable figures for subproject 22.

This script only redraws existing computed results from output tables.
No upstream model inference is rerun.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def regenerate_fig1(pair_diag_path: Path, out_path: Path) -> None:
    """Create a faceted, sorted version of real-data non-additivity diagnostics."""
    df = pd.read_csv(pair_diag_path).copy()
    df["abs_te"] = df["te_mean"].abs()
    df["pair_label"] = df["pair"]

    metrics = [
        ("abs_te", "|Total effect|", "#355C9A"),
        ("residual_abs", "|Residual|", "#D07C54"),
        ("nonadditivity_lb", "Non-additivity lower bound", "#4C9A73"),
    ]

    tissues = sorted(df["tissue"].unique())
    grans = ["heads", "mlp"]

    fig, axes = plt.subplots(len(tissues), len(grans), figsize=(14, 3.8 * len(tissues)), dpi=180, squeeze=False)
    bar_w = 0.26

    for i, tissue in enumerate(tissues):
        for j, gran in enumerate(grans):
            ax = axes[i, j]
            sub = df[(df["tissue"] == tissue) & (df["granularity"] == gran)].copy()
            if sub.empty:
                ax.axis("off")
                continue

            sub = sub.sort_values("nonadditivity_lb_over_te", ascending=False).reset_index(drop=True)
            x = np.arange(sub.shape[0])

            for k, (col, label, color) in enumerate(metrics):
                ax.bar(x + (k - 1) * bar_w, sub[col].to_numpy(dtype=float), width=bar_w, label=label, color=color)

            # Annotate lower-bound ratios to make ranking instability scan-friendly.
            for idx, row in sub.iterrows():
                ratio = float(row["nonadditivity_lb_over_te"])
                ax.text(
                    x[idx] + bar_w,
                    float(row["nonadditivity_lb"]) + 0.03,
                    f"{ratio:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(sub["pair_label"].tolist(), rotation=35, ha="right", fontsize=8)
            ax.set_title(f"{tissue} / {gran}", fontsize=11)
            ax.set_ylabel("Effect magnitude")
            ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Real-data residual non-additivity (sorted within tissue/granularity); numbers are A_lb/|TE|", y=1.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _heatmap(ax: plt.Axes, mat: np.ndarray, xvals: list[int], yvals: list[float], title: str, vmin: float, vmax: float) -> None:
    """Draw annotated heatmap with fixed color scale and readable labels."""
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdBu", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(xvals)))
    ax.set_xticklabels([str(v) for v in xvals], fontsize=9)
    ax.set_yticks(np.arange(len(yvals)))
    ax.set_yticklabels([f"{v:.1f}" for v in yvals], fontsize=9)
    ax.set_xlabel("n_cells")
    ax.set_ylabel("pair_scale")
    ax.set_title(title, fontsize=10)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def regenerate_fig4(sim_summary_path: Path, out_path: Path) -> None:
    """Create annotated synthetic-gain heatmaps with clearer panel labeling."""
    df = pd.read_csv(sim_summary_path)

    scenarios = ["pairwise_only", "mixed_higher_order"]
    n_vals = sorted(df["n_cells"].unique().tolist())
    p_vals = sorted(df["pair_scale"].unique().tolist())

    fig, axes = plt.subplots(len(scenarios), 2, figsize=(13.5, 8.5), dpi=180, squeeze=False)

    for r, scenario in enumerate(scenarios):
        sub = df[df["scenario"] == scenario]
        mat_raw = np.zeros((len(p_vals), len(n_vals)), dtype=float)
        mat_shrink = np.zeros((len(p_vals), len(n_vals)), dtype=float)

        p_to_i = {v: i for i, v in enumerate(p_vals)}
        n_to_j = {v: j for j, v in enumerate(n_vals)}

        for row in sub.itertuples(index=False):
            i = p_to_i[float(row.pair_scale)]
            j = n_to_j[int(row.n_cells)]
            mat_raw[i, j] = float(row.spearman_gain_paircorr)
            mat_shrink[i, j] = float(row.spearman_gain_pairshrink)

        _heatmap(
            axes[r, 0],
            mat_raw,
            xvals=n_vals,
            yvals=p_vals,
            title=f"({chr(65 + 2*r)}) {scenario}: raw pairwise minus single",
            vmin=-0.20,
            vmax=0.20,
        )
        _heatmap(
            axes[r, 1],
            mat_shrink,
            xvals=n_vals,
            yvals=p_vals,
            title=f"({chr(66 + 2*r)}) {scenario}: shrinkage pairwise minus single",
            vmin=-0.20,
            vmax=0.20,
        )

    fig.suptitle("Synthetic estimator gains (annotated Spearman deltas)", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def regenerate_fig7(triplet_summary_path: Path, out_path: Path) -> None:
    """Draw pairwise-vs-triplet incremental explanation bars for targeted contexts."""
    df = pd.read_csv(triplet_summary_path).copy()
    if df.empty:
        return

    df["label"] = df["run_id"] + "\n" + df["pair"]
    df = df.sort_values("pair_plus_triplet_to_residual", ascending=False).reset_index(drop=True)

    x = np.arange(df.shape[0])
    bar_w = 0.22

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=180)
    ax.bar(x - bar_w, df["residual_abs"], width=bar_w, label="|Residual|", color="#D07C54")
    ax.bar(x, df["best_abs_interaction_raw"], width=bar_w, label="Best measured |pair interaction|", color="#4C9A73")
    ax.bar(x + bar_w, df["tau_abs_mean"], width=bar_w, label="Measured |triple interaction|", color="#5A83C2")

    for i, row in df.iterrows():
        ax.text(
            i + bar_w,
            float(row["tau_abs_mean"]) + 0.02,
            f"{100.0 * float(row['triplet_abs_to_residual']):.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
        ax.text(
            i,
            float(row["best_abs_interaction_raw"]) + 0.02,
            f"{100.0 * float(row['best_abs_interaction_to_residual']):.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("Effect magnitude")
    ax.set_title("Measured triplet interactions add limited residual explanation beyond pairwise terms")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    tables = repo / "subproject_22_patching_mediation_bias_bounds" / "outputs" / "tables"
    figs = repo / "subproject_22_patching_mediation_bias_bounds" / "reports" / "figures"

    regenerate_fig1(
        pair_diag_path=tables / "pair_bias_diagnostics.csv",
        out_path=figs / "fig1_real_residual_nonadditivity.png",
    )
    regenerate_fig4(
        sim_summary_path=tables / "synthetic_estimator_summary.csv",
        out_path=figs / "fig4_synthetic_estimator_gains.png",
    )
    regenerate_fig7(
        triplet_summary_path=tables / "paper_quality_triplet_residual_explanation.csv",
        out_path=figs / "fig7_triplet_incremental_explanation.png",
    )

    print(figs / "fig1_real_residual_nonadditivity.png")
    print(figs / "fig4_synthetic_estimator_gains.png")
    print(figs / "fig7_triplet_incremental_explanation.png")


if __name__ == "__main__":
    main()
