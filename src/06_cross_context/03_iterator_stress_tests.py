#!/usr/bin/env python3
"""Adversarial stress-test analyses for subproject 04 iterator pass."""

from __future__ import annotations

import argparse
from itertools import combinations, permutations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _exact_permutation_delta(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> tuple[float, float]:
    values = np.concatenate([values_a, values_b])
    n_total = len(values)
    n_a = len(values_a)
    observed = float(values_a.mean() - values_b.mean())

    extreme = 0
    total = 0
    for idx_tuple in combinations(range(n_total), n_a):
        idx = set(idx_tuple)
        a = np.array([values[i] for i in range(n_total) if i in idx], dtype=float)
        b = np.array([values[i] for i in range(n_total) if i not in idx], dtype=float)
        delta = float(a.mean() - b.mean())
        total += 1
        if abs(delta) >= abs(observed) - 1e-12:
            extreme += 1
    p_value = float(extreme / total)
    return observed, p_value


def transfer_permutation_and_loo(
    artifact_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transfer_30 = _read_csv(artifact_dir / "transfer_regret_30_cell.csv")
    transfer_100 = _read_csv(artifact_dir / "transfer_regret_100_cell.csv")

    delta, p_value = _exact_permutation_delta(
        transfer_30["relative_regret"].to_numpy(dtype=float),
        transfer_100["relative_regret"].to_numpy(dtype=float),
    )
    perm_df = pd.DataFrame(
        [
            {
                "comparison": "30_cell_vs_100_cell_mean_relative_regret",
                "delta_30_minus_100": delta,
                "exact_two_sided_p_value": p_value,
                "n_pairs_per_group": int(len(transfer_30)),
            }
        ]
    )

    rows: list[dict[str, float | str]] = []
    tissues = sorted(set(transfer_30["target_tissue"]).union(transfer_100["target_tissue"]))
    for tissue in tissues:
        mean_30 = float(
            transfer_30.loc[transfer_30["target_tissue"] != tissue, "relative_regret"].mean()
        )
        mean_100 = float(
            transfer_100.loc[transfer_100["target_tissue"] != tissue, "relative_regret"].mean()
        )
        rows.append(
            {
                "leave_out_target_tissue": tissue,
                "mean_relative_regret_30": mean_30,
                "mean_relative_regret_100": mean_100,
                "delta_30_minus_100": mean_30 - mean_100,
            }
        )
    loo_df = pd.DataFrame(rows).sort_values("leave_out_target_tissue", ignore_index=True)

    source_rows: list[dict[str, float | str]] = []
    sources = sorted(set(transfer_30["source_tissue"]).union(transfer_100["source_tissue"]))
    for source in sources:
        mean_30 = float(
            transfer_30.loc[transfer_30["source_tissue"] != source, "relative_regret"].mean()
        )
        mean_100 = float(
            transfer_100.loc[transfer_100["source_tissue"] != source, "relative_regret"].mean()
        )
        source_rows.append(
            {
                "leave_out_source_tissue": source,
                "mean_relative_regret_30": mean_30,
                "mean_relative_regret_100": mean_100,
                "delta_30_minus_100": mean_30 - mean_100,
            }
        )
    source_loo_df = pd.DataFrame(source_rows).sort_values(
        "leave_out_source_tissue", ignore_index=True
    )
    return perm_df, loo_df, source_loo_df


def confound_weight_scheme_sensitivity(artifact_dir: Path) -> pd.DataFrame:
    meta = _read_csv(artifact_dir / "tissue_metadata_heterogeneity.csv").set_index("tissue")
    transfer_30 = _read_csv(artifact_dir / "transfer_regret_30_cell.csv")
    transfer_100 = _read_csv(artifact_dir / "transfer_regret_100_cell.csv")

    schemes = {
        "uniform": pd.Series(1.0, index=meta.index),
        "donor_diversity": 1.0 - meta["donor_dominant_fraction"],
        "sample_diversity": 1.0 - meta["sample_dominant_fraction"],
        "celltype_entropy": meta["cell_type_entropy_norm"],
        "combined_diversity": meta["confound_diversity_score"],
    }

    rows: list[dict[str, float | str]] = []
    for scheme, raw_weights in schemes.items():
        weights = raw_weights.astype(float).copy()
        fill_value = float(weights.mean()) if np.isfinite(weights.mean()) else 1.0
        weights = weights.fillna(fill_value)
        if np.isclose(weights.sum(), 0.0):
            weights[:] = 1.0
        weights = weights / weights.sum()

        for budget, transfer_df in [("30", transfer_30), ("100", transfer_100)]:
            pair_weights = transfer_df["target_tissue"].map(weights).to_numpy(dtype=float)
            weighted_mean = float(
                np.average(transfer_df["relative_regret"].to_numpy(dtype=float), weights=pair_weights)
            )
            rows.append(
                {
                    "weight_scheme": scheme,
                    "cell_budget": budget,
                    "weighted_mean_relative_regret": weighted_mean,
                }
            )

    result = pd.DataFrame(rows).sort_values(["weight_scheme", "cell_budget"], ignore_index=True)
    # Add explicit shift column per scheme.
    pivot = result.pivot(index="weight_scheme", columns="cell_budget", values="weighted_mean_relative_regret")
    shift = (pivot["30"] - pivot["100"]).rename("delta_30_minus_100")
    out = result.merge(shift, on="weight_scheme", how="left")
    return out


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = float(np.mean(sample))
    point = float(np.mean(values))
    return point, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def policy_uplift_bootstrap(
    artifact_dir: Path,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    table_30 = _read_csv(artifact_dir / "mvp_mean_aupr_30_cell.csv").set_index("tissue")
    table_100 = _read_csv(artifact_dir / "mvp_mean_aupr_100_cell.csv").set_index("tissue")

    rows: list[dict[str, float | str]] = []
    for budget, table in [("30", table_30), ("100", table_100)]:
        random_expected = table.mean(axis=1).to_numpy(dtype=float)

        candidate_policies = {
            "fixed_probe::coexpression": "coexpression",
            "fixed_probe::coexpression_signed": "coexpression_signed",
            "source_best::immune": str(table.loc["immune"].idxmax()),
            "source_best::kidney": str(table.loc["kidney"].idxmax()),
            "source_best::lung": str(table.loc["lung"].idxmax()),
        }
        for policy, probe in candidate_policies.items():
            if probe not in table.columns:
                continue
            uplift = table[probe].to_numpy(dtype=float) - random_expected
            point, lo, hi = _bootstrap_ci(uplift, n_boot=n_boot, seed=seed + len(rows) + 7)
            rows.append(
                {
                    "cell_budget": budget,
                    "policy": policy,
                    "probe_used": probe,
                    "mean_uplift_vs_random": point,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_tissues": int(len(uplift)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["cell_budget", "mean_uplift_vs_random", "policy"], ascending=[True, False, True]
    )


def source_best_vs_random_pairwise(artifact_dir: Path) -> pd.DataFrame:
    table_30 = _read_csv(artifact_dir / "mvp_mean_aupr_30_cell.csv").set_index("tissue")
    table_100 = _read_csv(artifact_dir / "mvp_mean_aupr_100_cell.csv").set_index("tissue")

    rows: list[dict[str, float | str]] = []
    for budget, table in [("30", table_30), ("100", table_100)]:
        probes = list(table.columns)
        for source_tissue in table.index:
            source_best_probe = str(table.loc[source_tissue].idxmax())
            for target_tissue in table.index:
                if target_tissue == source_tissue:
                    continue
                source_best_score = float(table.loc[target_tissue, source_best_probe])
                random_score = float(table.loc[target_tissue, probes].mean())
                rows.append(
                    {
                        "cell_budget": budget,
                        "source_tissue": source_tissue,
                        "target_tissue": target_tissue,
                        "source_best_probe": source_best_probe,
                        "source_best_minus_random": source_best_score - random_score,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["cell_budget", "source_tissue", "target_tissue"], ignore_index=True
    )


def threshold_pairwise_significance(artifact_dir: Path) -> pd.DataFrame:
    fixed = _read_csv(artifact_dir / "sweep_threshold_fixed_percentile_regret.csv")
    p90 = (
        fixed[fixed["fixed_percentile"] == 90][["sweep_file", "delta_best_minus_fixed_f1"]]
        .rename(columns={"delta_best_minus_fixed_f1": "delta_p90"})
        .copy()
    )
    p95 = (
        fixed[fixed["fixed_percentile"] == 95][["sweep_file", "delta_best_minus_fixed_f1"]]
        .rename(columns={"delta_best_minus_fixed_f1": "delta_p95"})
        .copy()
    )
    paired = p90.merge(p95, on="sweep_file", how="inner")
    paired["improvement_p95_vs_p90"] = paired["delta_p90"] - paired["delta_p95"]

    n = int(len(paired))
    n_positive = int((paired["improvement_p95_vs_p90"] > 0).sum())
    one_sided_p = float(sum(comb(n, k) for k in range(n_positive, n + 1)) / (2 ** n))

    summary = pd.DataFrame(
        [
            {
                "n_paired_sweeps": n,
                "n_positive_improvement": n_positive,
                "mean_improvement_p95_vs_p90": float(paired["improvement_p95_vs_p90"].mean()),
                "median_improvement_p95_vs_p90": float(paired["improvement_p95_vs_p90"].median()),
                "one_sided_sign_test_p_value": one_sided_p,
            }
        ]
    )
    return paired, summary


def interpretability_family_and_stability(artifact_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_df = _read_csv(artifact_dir / "mvp_best_probe_per_reference.csv")
    aupr_df = _read_csv(artifact_dir / "mvp_mean_aupr_30_cell.csv")
    stability_df = _read_csv(artifact_dir / "mvp_stability_30_cell.csv")

    attribution_family = {
        "attention",
        "consensus",
        "grad_input",
        "integrated_gradients",
        "perturbation",
    }
    coexpression_family = {"coexpression", "coexpression_signed"}

    def _family(probe: str) -> str:
        if probe in attribution_family:
            return "attribution_family"
        if probe in coexpression_family:
            return "coexpression_family"
        return "other"

    best_df = best_df.copy()
    best_df["family"] = best_df["best_probe"].map(_family)
    family_summary = (
        best_df["family"]
        .value_counts(normalize=False)
        .rename_axis("family")
        .reset_index(name="wins")
    )
    family_summary["win_fraction"] = family_summary["wins"] / family_summary["wins"].sum()

    probes = [col for col in aupr_df.columns if col != "tissue"]
    rows: list[dict[str, float | str]] = []
    for tissue in aupr_df["tissue"]:
        x = aupr_df.loc[aupr_df["tissue"] == tissue, probes].iloc[0].to_numpy(dtype=float)
        y = stability_df.loc[stability_df["tissue"] == tissue, probes].iloc[0].to_numpy(dtype=float)
        obs = float(pd.Series(x).rank().corr(pd.Series(y).rank(), method="pearson"))

        # Exact permutation p-value with n=7 probes (7! = 5040).
        extreme = 0
        total = 0
        for perm_idx in permutations(range(len(y))):
            y_perm = y[list(perm_idx)]
            rho = float(pd.Series(x).rank().corr(pd.Series(y_perm).rank(), method="pearson"))
            total += 1
            if abs(rho) >= abs(obs) - 1e-12:
                extreme += 1
        p_val = float(extreme / total)
        rows.append(
            {
                "tissue": tissue,
                "spearman_rho_aupr_vs_stability": obs,
                "exact_two_sided_p_value": p_val,
                "n_probes": int(len(probes)),
            }
        )
    stability_perm = pd.DataFrame(rows).sort_values("tissue", ignore_index=True)
    return family_summary, stability_perm


def run(artifact_dir: Path, output_dir: Path, n_boot: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    perm_df, loo_df, source_loo_df = transfer_permutation_and_loo(artifact_dir)
    perm_df.to_csv(output_dir / "iterator_transfer_permutation.csv", index=False)
    loo_df.to_csv(output_dir / "iterator_transfer_loo.csv", index=False)
    source_loo_df.to_csv(output_dir / "iterator_transfer_source_loo.csv", index=False)

    weight_scheme_df = confound_weight_scheme_sensitivity(artifact_dir)
    weight_scheme_df.to_csv(output_dir / "iterator_weight_scheme_sensitivity.csv", index=False)

    policy_df = policy_uplift_bootstrap(artifact_dir, n_boot=n_boot, seed=seed)
    policy_df.to_csv(output_dir / "iterator_policy_uplift_bootstrap.csv", index=False)
    pairwise_policy_df = source_best_vs_random_pairwise(artifact_dir)
    pairwise_policy_df.to_csv(output_dir / "iterator_source_best_vs_random_pairwise.csv", index=False)

    threshold_pairwise_df, threshold_summary_df = threshold_pairwise_significance(artifact_dir)
    threshold_pairwise_df.to_csv(output_dir / "iterator_threshold_p95_vs_p90_pairwise.csv", index=False)
    threshold_summary_df.to_csv(output_dir / "iterator_threshold_p95_vs_p90_summary.csv", index=False)

    family_df, stability_df = interpretability_family_and_stability(artifact_dir)
    family_df.to_csv(output_dir / "iterator_interpretability_family_wins.csv", index=False)
    stability_df.to_csv(output_dir / "iterator_stability_permutation.csv", index=False)

    summary_lines = [
        "# Iterator Stress-Test Summary",
        "",
        "## Transfer shift significance",
    ]
    row = perm_df.iloc[0]
    summary_lines.append(
        f"- Exact permutation delta (30 minus 100): {row['delta_30_minus_100']:.4f} "
        f"(p={row['exact_two_sided_p_value']:.4f})."
    )
    summary_lines.append("")
    summary_lines.append("## Leave-one-target-out robustness")
    for item in loo_df.itertuples(index=False):
        summary_lines.append(
            f"- Drop {item.leave_out_target_tissue}: "
            f"delta={item.delta_30_minus_100:.4f} "
            f"(30={item.mean_relative_regret_30:.4f}, 100={item.mean_relative_regret_100:.4f})."
        )
    summary_lines.append("")
    summary_lines.append("## Leave-one-source-out robustness")
    for item in source_loo_df.itertuples(index=False):
        summary_lines.append(
            f"- Drop source {item.leave_out_source_tissue}: "
            f"delta={item.delta_30_minus_100:.4f} "
            f"(30={item.mean_relative_regret_30:.4f}, 100={item.mean_relative_regret_100:.4f})."
        )
    summary_lines.append("")
    summary_lines.append("## Weight-scheme sensitivity")
    for scheme, group in weight_scheme_df.groupby("weight_scheme", sort=True):
        g30 = float(group.loc[group["cell_budget"] == "30", "weighted_mean_relative_regret"].iloc[0])
        g100 = float(group.loc[group["cell_budget"] == "100", "weighted_mean_relative_regret"].iloc[0])
        summary_lines.append(
            f"- {scheme}: 30={g30:.4f}, 100={g100:.4f}, delta={g30-g100:.4f}."
        )
    summary_lines.append("")
    summary_lines.append("## Interpretability validity checks")
    family_map = dict(zip(family_df["family"], family_df["wins"]))
    total_wins = int(family_df["wins"].sum())
    summary_lines.append(
        f"- Best-probe winners: coexpression-family={family_map.get('coexpression_family', 0)}/"
        f"{total_wins}, attribution-family={family_map.get('attribution_family', 0)}/{total_wins}."
    )
    for item in stability_df.itertuples(index=False):
        summary_lines.append(
            f"- {item.tissue}: rho(AUPR, stability)={item.spearman_rho_aupr_vs_stability:.3f}, "
            f"exact p={item.exact_two_sided_p_value:.4f}."
        )
    summary_lines.append("")
    summary_lines.append("## Threshold default stress test")
    threshold_row = threshold_summary_df.iloc[0]
    summary_lines.append(
        f"- P95 improves over P90 in {int(threshold_row['n_positive_improvement'])}/"
        f"{int(threshold_row['n_paired_sweeps'])} paired sweeps; "
        f"mean improvement={threshold_row['mean_improvement_p95_vs_p90']:.6f}; "
        f"one-sided sign-test p={threshold_row['one_sided_sign_test_p_value']:.4f}."
    )
    (output_dir / "iterator_stress_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run iterator stress tests for subproject 04.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(
            "/Users/ihorkendiukhov/biodyn-work/subproject_04_cross_tissue_probe_threshold_transfer/"
            "reports/artifacts"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/Users/ihorkendiukhov/biodyn-work/subproject_04_cross_tissue_probe_threshold_transfer/"
            "reports/artifacts"
        ),
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.artifact_dir, args.output_dir, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
