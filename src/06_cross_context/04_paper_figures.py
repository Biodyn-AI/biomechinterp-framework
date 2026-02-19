#!/usr/bin/env python3
"""Generate figures for the cross-tissue threshold-transfer paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 300


def _read_csv(artifact_dir: Path, name: str) -> pd.DataFrame:
    path = artifact_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact file: {path}")
    return pd.read_csv(path)


def plot_aupr_heatmaps(artifact_dir: Path, figure_dir: Path) -> None:
    df30 = _read_csv(artifact_dir, "mvp_mean_aupr_30_cell.csv").set_index("tissue")
    df100 = _read_csv(artifact_dir, "mvp_mean_aupr_100_cell.csv").set_index("tissue")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    vmax = max(df30.to_numpy().max(), df100.to_numpy().max())

    sns.heatmap(
        df30,
        ax=axes[0],
        cmap="YlGnBu",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        vmin=0.0,
        vmax=vmax,
        cbar=False,
    )
    axes[0].set_title("AUPR by Tissue/Probe (30 cells)")
    axes[0].set_xlabel("Probe")
    axes[0].set_ylabel("Tissue")

    sns.heatmap(
        df100,
        ax=axes[1],
        cmap="YlGnBu",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        vmin=0.0,
        vmax=vmax,
        cbar_kws={"label": "Mean AUPR"},
    )
    axes[1].set_title("AUPR by Tissue/Probe (100 cells)")
    axes[1].set_xlabel("Probe")
    axes[1].set_ylabel("Tissue")

    out = figure_dir / "fig01_aupr_heatmaps_30_100.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_regret(artifact_dir: Path, figure_dir: Path) -> None:
    summary = _read_csv(artifact_dir, "transfer_regret_summary.csv")
    summary = summary[summary["source_tissue"] != "overall"].copy()
    summary["cell_budget"] = summary["cell_budget"].astype(str)

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    sns.barplot(
        data=summary,
        x="source_tissue",
        y="mean_relative_regret",
        hue="cell_budget",
        palette=["#d95f02", "#1b9e77"],
        ax=ax,
    )
    ax.set_title("Held-Out Transfer Regret by Source Tissue")
    ax.set_xlabel("Source tissue used for probe selection")
    ax.set_ylabel("Mean relative regret")
    ax.legend(title="Cell budget")

    out = figure_dir / "fig02_transfer_regret.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _rank_matrix(rank_df: pd.DataFrame) -> pd.DataFrame:
    tissues = sorted(set(rank_df["tissue_a"]).union(rank_df["tissue_b"]))
    matrix = pd.DataFrame(np.eye(len(tissues)), index=tissues, columns=tissues)
    for row in rank_df.itertuples(index=False):
        matrix.loc[row.tissue_a, row.tissue_b] = row.spearman_rank_corr
        matrix.loc[row.tissue_b, row.tissue_a] = row.spearman_rank_corr
    return matrix


def plot_rank_concordance(artifact_dir: Path, figure_dir: Path) -> None:
    rank30 = _read_csv(artifact_dir, "probe_rank_spearman_30_cell.csv")
    rank100 = _read_csv(artifact_dir, "probe_rank_spearman_100_cell.csv")
    mat30 = _rank_matrix(rank30)
    mat100 = _rank_matrix(rank100)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), constrained_layout=True)
    cmap = sns.diverging_palette(15, 220, as_cmap=True)

    sns.heatmap(
        mat30,
        ax=axes[0],
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        cmap=cmap,
        linewidths=0.5,
        cbar=False,
    )
    axes[0].set_title("Probe-Rank Concordance (30 cells)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    sns.heatmap(
        mat100,
        ax=axes[1],
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={"label": "Spearman rank correlation"},
    )
    axes[1].set_title("Probe-Rank Concordance (100 cells)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    out = figure_dir / "fig03_rank_concordance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_budget_shift_heatmap(artifact_dir: Path, figure_dir: Path) -> None:
    shift = _read_csv(artifact_dir, "cell_budget_shift_30_to_100.csv")
    matrix = shift.pivot(index="tissue", columns="probe", values="delta_100_minus_30")

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    vmax = float(np.abs(matrix.to_numpy()).max())
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="coolwarm",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "AUPR change (100-cell minus 30-cell)"},
    )
    ax.set_title("Cell-Budget Sensitivity by Tissue and Probe")
    ax.set_xlabel("Probe")
    ax.set_ylabel("Tissue")

    out = figure_dir / "fig04_cell_budget_shift_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_best_probe_reference(artifact_dir: Path, figure_dir: Path) -> None:
    best = _read_csv(artifact_dir, "mvp_best_probe_per_reference.csv")
    pivot = best.pivot(index="tissue", columns="reference", values="best_probe")

    probe_order = [
        "attention",
        "consensus",
        "grad_input",
        "integrated_gradients",
        "perturbation",
        "coexpression",
        "coexpression_signed",
    ]
    probe_to_id = {probe: idx for idx, probe in enumerate(probe_order)}
    numeric = pivot.apply(lambda column: column.map(lambda value: probe_to_id.get(value, -1)))

    cmap = sns.color_palette("tab10", n_colors=len(probe_order))
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    hm = sns.heatmap(
        numeric,
        ax=ax,
        cmap=cmap,
        linewidths=0.5,
        cbar=False,
        annot=pivot,
        fmt="",
    )
    ax.set_title("Best Probe per Tissue and Reference (30-cell MVP)")
    ax.set_xlabel("Reference")
    ax.set_ylabel("Tissue")
    hm.collections[0].colorbar = None

    out = figure_dir / "fig05_best_probe_reference_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_robustness(artifact_dir: Path, figure_dir: Path) -> None:
    summary = _read_csv(artifact_dir, "sweep_threshold_summary.csv")
    fixed = _read_csv(artifact_dir, "sweep_threshold_fixed_percentile_regret.csv")
    fixed["fixed_percentile"] = fixed["fixed_percentile"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)

    sns.countplot(
        data=summary,
        x="best_percentile",
        color="#4575b4",
        ax=axes[0],
    )
    axes[0].set_title("Best Percentile Choices Across Sweeps")
    axes[0].set_xlabel("Best percentile")
    axes[0].set_ylabel("Count of sweep files")

    sns.boxplot(
        data=fixed,
        x="fixed_percentile",
        y="delta_best_minus_fixed_f1",
        hue="fixed_percentile",
        palette=["#fee08b", "#66bd63"],
        dodge=False,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Penalty of Fixed Percentiles vs Sweep-Optimal")
    axes[1].set_xlabel("Fixed percentile")
    axes[1].set_ylabel("F1 penalty (best minus fixed)")

    out = figure_dir / "fig06_threshold_robustness.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_stability_vs_aupr(artifact_dir: Path, figure_dir: Path) -> None:
    aupr = _read_csv(artifact_dir, "mvp_mean_aupr_30_cell.csv")
    stability = _read_csv(artifact_dir, "mvp_stability_30_cell.csv")

    merged_rows = []
    probes = [col for col in aupr.columns if col != "tissue"]
    for tissue in aupr["tissue"]:
        row_a = aupr.loc[aupr["tissue"] == tissue].iloc[0]
        row_s = stability.loc[stability["tissue"] == tissue].iloc[0]
        for probe in probes:
            merged_rows.append(
                {
                    "tissue": tissue,
                    "probe": probe,
                    "aupr": float(row_a[probe]),
                    "stability": float(row_s[probe]),
                }
            )
    merged = pd.DataFrame(merged_rows)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    sns.scatterplot(
        data=merged,
        x="stability",
        y="aupr",
        hue="tissue",
        style="probe",
        s=80,
        ax=ax,
    )
    ax.set_title("30-cell AUPR vs Stability Across Tissues and Probes")
    ax.set_xlabel("Mean Jaccard stability")
    ax.set_ylabel("Mean AUPR")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    out = figure_dir / "fig07_stability_vs_aupr.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_policy_benchmark(artifact_dir: Path, figure_dir: Path) -> None:
    summary = _read_csv(artifact_dir, "policy_transfer_benchmark_summary.csv").copy()
    summary["cell_budget"] = summary["cell_budget"].astype(str)
    summary = summary[
        summary["policy"].str.startswith("source_best::")
        | summary["policy"].str.startswith("fixed_probe::coexpression")
        | summary["policy"].str.startswith("fixed_probe::coexpression_signed")
        | (summary["policy"] == "random_probe_expected")
    ].copy()

    summary["policy_label"] = summary["policy"].replace(
        {
            "fixed_probe::coexpression": "fixed::coexpression",
            "fixed_probe::coexpression_signed": "fixed::coexpression_signed",
            "random_probe_expected": "random_expected",
            "source_best::immune": "source_best::immune",
            "source_best::kidney": "source_best::kidney",
            "source_best::lung": "source_best::lung",
        }
    )

    fig, ax = plt.subplots(figsize=(10.2, 5.0), constrained_layout=True)
    sns.barplot(
        data=summary.sort_values(["cell_budget", "mean_regret_vs_oracle"]),
        x="policy_label",
        y="mean_regret_vs_oracle",
        hue="cell_budget",
        palette=["#d95f02", "#1b9e77"],
        ax=ax,
    )
    ax.set_title("Policy Benchmark: Mean Regret vs Tissue-wise Oracle")
    ax.set_xlabel("Transfer policy")
    ax.set_ylabel("Mean regret vs oracle AUPR")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Cell budget")

    out = figure_dir / "fig08_policy_transfer_benchmark.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_confound_sensitivity(artifact_dir: Path, figure_dir: Path) -> None:
    metadata = _read_csv(artifact_dir, "tissue_metadata_heterogeneity.csv").copy()
    transfer_ci = _read_csv(artifact_dir, "transfer_regret_bootstrap_ci.csv").copy()
    tissue_order = sorted(metadata["tissue"].unique())

    # Left panel: directly observable composition dominance and heterogeneity.
    dominance = metadata.melt(
        id_vars=["tissue"],
        value_vars=["donor_dominant_fraction", "sample_dominant_fraction"],
        var_name="metric",
        value_name="value",
    )
    dominance["metric"] = dominance["metric"].replace(
        {
            "donor_dominant_fraction": "donor dominant frac",
            "sample_dominant_fraction": "sample dominant frac",
        }
    )

    # Right panel: unweighted vs confound-weighted transfer regret with CI.
    transfer_ci["cell_budget"] = transfer_ci["cell_budget"].astype(str)
    transfer_ci["weighting"] = transfer_ci["weighting"].replace(
        {
            "unweighted": "unweighted",
            "confound_weighted_target": "confound-weighted",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)

    sns.barplot(
        data=dominance,
        x="tissue",
        y="value",
        hue="metric",
        order=tissue_order,
        palette=["#e08214", "#5aae61"],
        ax=axes[0],
    )
    entropy_series = (
        metadata.set_index("tissue").reindex(tissue_order)["cell_type_entropy_norm"].to_numpy()
    )
    axes[0].plot(
        range(len(tissue_order)),
        entropy_series,
        marker="o",
        linestyle="--",
        color="#2b8cbe",
        label="cell-type entropy (norm.)",
    )
    axes[0].set_title("Composition Imbalance by Tissue")
    axes[0].set_xlabel("Tissue")
    axes[0].set_ylabel("Fraction / normalized entropy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(loc="upper right", frameon=False)

    x_positions = np.arange(len(transfer_ci))
    colors = ["#d95f02" if b == "30" else "#1b9e77" for b in transfer_ci["cell_budget"]]
    axes[1].scatter(
        x_positions,
        transfer_ci["mean_relative_regret"],
        c=colors,
        s=70,
        zorder=3,
    )
    for idx, row in enumerate(transfer_ci.itertuples(index=False)):
        axes[1].plot(
            [idx, idx],
            [row.ci95_low, row.ci95_high],
            color=colors[idx],
            linewidth=2.0,
            zorder=2,
        )
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(
        [
            f"{row.weighting}\n{row.cell_budget} cells"
            for row in transfer_ci.itertuples(index=False)
        ],
        rotation=20,
        ha="right",
    )
    axes[1].set_title("Transfer Regret Sensitivity with 95% Bootstrap CI")
    axes[1].set_xlabel("Estimator")
    axes[1].set_ylabel("Mean relative regret")
    axes[1].grid(axis="y", alpha=0.3)

    out = figure_dir / "fig09_confound_sensitivity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_iterator_stress_tests(artifact_dir: Path, figure_dir: Path) -> None:
    loo = _read_csv(artifact_dir, "iterator_transfer_loo.csv").copy()
    policy = _read_csv(artifact_dir, "iterator_policy_uplift_bootstrap.csv").copy()
    policy["cell_budget"] = policy["cell_budget"].astype(str)

    keep_policies = [
        "fixed_probe::coexpression",
        "fixed_probe::coexpression_signed",
        "source_best::immune",
        "source_best::kidney",
        "source_best::lung",
    ]
    policy = policy[policy["policy"].isin(keep_policies)].copy()
    policy["policy_label"] = policy["policy"].replace(
        {
            "fixed_probe::coexpression": "fixed::coexpr",
            "fixed_probe::coexpression_signed": "fixed::coexpr_signed",
            "source_best::immune": "src_best::immune",
            "source_best::kidney": "src_best::kidney",
            "source_best::lung": "src_best::lung",
        }
    )
    policy = policy.sort_values(["cell_budget", "mean_uplift_vs_random"], ascending=[True, False])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)

    sns.barplot(
        data=loo,
        x="leave_out_target_tissue",
        y="delta_30_minus_100",
        color="#4c78a8",
        ax=axes[0],
    )
    axes[0].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[0].set_title("Leave-One-Target Robustness of Transfer Shift")
    axes[0].set_xlabel("Left-out target tissue")
    axes[0].set_ylabel(r"$\Delta$ mean relative regret (30 minus 100)")

    for idx, row in enumerate(loo.itertuples(index=False)):
        axes[0].text(
            idx,
            row.delta_30_minus_100 + 0.008,
            f"{row.delta_30_minus_100:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    sns.barplot(
        data=policy,
        x="policy_label",
        y="mean_uplift_vs_random",
        hue="cell_budget",
        palette=["#d95f02", "#1b9e77"],
        ax=axes[1],
    )
    # Add CI whiskers manually to ensure consistency across seaborn versions.
    for patch, (_, row) in zip(axes[1].patches, policy.iterrows()):
        x = patch.get_x() + patch.get_width() / 2.0
        axes[1].plot([x, x], [row["ci95_low"], row["ci95_high"]], color="black", linewidth=1.1)
    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_title("Policy Uplift vs Random Baseline (95% CI)")
    axes[1].set_xlabel("Policy")
    axes[1].set_ylabel("Mean AUPR uplift vs random")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(title="Cell budget")

    out = figure_dir / "fig10_iterator_stress_tests.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_iterator_round2_sensitivity(artifact_dir: Path, figure_dir: Path) -> None:
    weight_df = _read_csv(artifact_dir, "iterator_weight_scheme_sensitivity.csv").copy()
    pairwise_df = _read_csv(artifact_dir, "iterator_source_best_vs_random_pairwise.csv").copy()
    threshold_summary = _read_csv(artifact_dir, "iterator_threshold_p95_vs_p90_summary.csv").iloc[0]
    weight_df["cell_budget"] = weight_df["cell_budget"].astype(str)

    weight_pivot = weight_df.pivot(
        index="weight_scheme",
        columns="cell_budget",
        values="weighted_mean_relative_regret",
    ).reset_index()
    weight_pivot["delta_30_minus_100"] = weight_pivot["30"] - weight_pivot["100"]
    weight_pivot = weight_pivot.sort_values("delta_30_minus_100", ascending=False)

    pairwise_df["cell_budget"] = pairwise_df["cell_budget"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)

    sns.barplot(
        data=weight_pivot,
        x="weight_scheme",
        y="delta_30_minus_100",
        color="#4c78a8",
        ax=axes[0],
    )
    axes[0].set_title("Transfer-Shift Robustness Across Weight Schemes")
    axes[0].set_xlabel("Weighting scheme")
    axes[0].set_ylabel(r"$\Delta$ weighted regret (30 minus 100)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].axhline(0.0, color="black", linewidth=1.0, linestyle="--")

    sns.boxplot(
        data=pairwise_df,
        x="cell_budget",
        y="source_best_minus_random",
        hue="cell_budget",
        palette=["#d95f02", "#1b9e77"],
        legend=False,
        ax=axes[1],
    )
    sns.stripplot(
        data=pairwise_df,
        x="cell_budget",
        y="source_best_minus_random",
        color="black",
        size=4.0,
        alpha=0.65,
        ax=axes[1],
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_title("Source-Best vs Random on Held-Out Targets")
    axes[1].set_xlabel("Cell budget")
    axes[1].set_ylabel("AUPR(source-best) minus AUPR(random)")
    axes[1].text(
        0.03,
        0.97,
        f"P95>P90 in {int(threshold_summary['n_positive_improvement'])}/"
        f"{int(threshold_summary['n_paired_sweeps'])} sweeps\n"
        f"sign-test p={threshold_summary['one_sided_sign_test_p_value']:.3f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )

    out = figure_dir / "fig11_iterator_round2_sensitivity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures for subproject 04.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(
            "/Users/ihorkendiukhov/biodyn-work/subproject_04_cross_tissue_probe_threshold_transfer/"
            "reports/artifacts"
        ),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path(
            "/Users/ihorkendiukhov/biodyn-work/subproject_04_cross_tissue_probe_threshold_transfer/"
            "reports/figures"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_style()
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    plot_aupr_heatmaps(args.artifact_dir, args.figure_dir)
    plot_transfer_regret(args.artifact_dir, args.figure_dir)
    plot_rank_concordance(args.artifact_dir, args.figure_dir)
    plot_budget_shift_heatmap(args.artifact_dir, args.figure_dir)
    plot_best_probe_reference(args.artifact_dir, args.figure_dir)
    plot_threshold_robustness(args.artifact_dir, args.figure_dir)
    plot_stability_vs_aupr(args.artifact_dir, args.figure_dir)
    plot_policy_benchmark(args.artifact_dir, args.figure_dir)
    plot_confound_sensitivity(args.artifact_dir, args.figure_dir)
    plot_iterator_stress_tests(args.artifact_dir, args.figure_dir)
    plot_iterator_round2_sensitivity(args.artifact_dir, args.figure_dir)


if __name__ == "__main__":
    main()
