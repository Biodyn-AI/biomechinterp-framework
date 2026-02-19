#!/usr/bin/env python3
"""Generate transfer-analysis artifacts for subproject 04.

The script is intentionally self-contained and reads only committed artifacts:
- workshop MVP markdown tables (cross-tissue probe benchmark summaries),
- threshold sweep JSON files copied into the subproject implementation area.

It writes flat CSV tables under the reports/artifacts directory so the workshop
paper can cite concrete, reproducible paths for every derived claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def extract_markdown_table(markdown_text: str, heading: str) -> pd.DataFrame:
    """Extract a markdown pipe table that immediately follows a heading line."""

    lines = markdown_text.splitlines()
    try:
        start_idx = lines.index(heading)
    except ValueError as exc:
        raise ValueError(f"Heading not found: {heading}") from exc

    table_lines: list[str] = []
    collecting = False
    for line in lines[start_idx + 1 :]:
        if line.startswith("|"):
            table_lines.append(line)
            collecting = True
            continue
        if collecting:
            break

    if len(table_lines) < 3:
        raise ValueError(f"No markdown table found under heading: {heading}")

    columns = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: list[list[str]] = []
    for row_line in table_lines[2:]:
        rows.append([cell.strip() for cell in row_line.strip("|").split("|")])
    return pd.DataFrame(rows, columns=columns)


def _coerce_probe_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize tissue/probe tables from markdown into numeric dataframe."""

    first_col = df.columns[0]
    out = df.rename(columns={first_col: "tissue"}).copy()
    for column in out.columns:
        if column == "tissue":
            continue
        out[column] = out[column].astype(float)
    return out


def _spearman_rank_correlation(a: Iterable[float], b: Iterable[float]) -> float:
    """Compute Spearman correlation without scipy by rank-transform + Pearson."""

    series_a = pd.Series(list(a), dtype=float).rank(method="average")
    series_b = pd.Series(list(b), dtype=float).rank(method="average")
    corr = series_a.corr(series_b, method="pearson")
    return float(corr) if pd.notna(corr) else float("nan")


def compute_transfer_regret(probe_df: pd.DataFrame, cell_budget: str) -> pd.DataFrame:
    """Evaluate source-tissue best-probe transfer regret on held-out tissues."""

    probes = [col for col in probe_df.columns if col != "tissue"]
    rows: list[dict[str, object]] = []

    for source_row in probe_df.itertuples(index=False):
        source_tissue = source_row.tissue
        source_best_probe = max(probes, key=lambda probe: getattr(source_row, probe))
        source_best_score = float(getattr(source_row, source_best_probe))

        for target_row in probe_df.itertuples(index=False):
            target_tissue = target_row.tissue
            if target_tissue == source_tissue:
                continue

            target_scores = {probe: float(getattr(target_row, probe)) for probe in probes}
            target_best_probe = max(target_scores, key=target_scores.get)
            target_best_score = target_scores[target_best_probe]
            transferred_score = target_scores[source_best_probe]
            abs_regret = target_best_score - transferred_score
            rel_regret = abs_regret / target_best_score if target_best_score > 0 else float("nan")

            rows.append(
                {
                    "cell_budget": cell_budget,
                    "source_tissue": source_tissue,
                    "source_best_probe": source_best_probe,
                    "source_best_score": source_best_score,
                    "target_tissue": target_tissue,
                    "target_best_probe": target_best_probe,
                    "target_best_score": target_best_score,
                    "transferred_score": transferred_score,
                    "absolute_regret": abs_regret,
                    "relative_regret": rel_regret,
                }
            )

    return pd.DataFrame(rows).sort_values(
        by=["cell_budget", "source_tissue", "target_tissue"], ignore_index=True
    )


def summarize_transfer_regret(transfer_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transfer regret by source tissue and overall cell budget."""

    source_summary = (
        transfer_df.groupby(["cell_budget", "source_tissue"], as_index=False)
        .agg(
            mean_relative_regret=("relative_regret", "mean"),
            max_relative_regret=("relative_regret", "max"),
            mean_absolute_regret=("absolute_regret", "mean"),
            max_absolute_regret=("absolute_regret", "max"),
        )
        .sort_values(by=["cell_budget", "source_tissue"], ignore_index=True)
    )
    overall_summary = (
        transfer_df.groupby(["cell_budget"], as_index=False)
        .agg(
            source_tissue=("cell_budget", lambda _: "overall"),
            mean_relative_regret=("relative_regret", "mean"),
            max_relative_regret=("relative_regret", "max"),
            mean_absolute_regret=("absolute_regret", "mean"),
            max_absolute_regret=("absolute_regret", "max"),
        )
        .sort_values(by=["cell_budget"], ignore_index=True)
    )
    return pd.concat([source_summary, overall_summary], ignore_index=True)


def compute_rank_correlations(probe_df: pd.DataFrame, cell_budget: str) -> pd.DataFrame:
    """Pairwise tissue rank-correlation for probe performance vectors."""

    probes = [col for col in probe_df.columns if col != "tissue"]
    rows: list[dict[str, object]] = []
    tissues = probe_df["tissue"].tolist()

    for i, tissue_a in enumerate(tissues):
        vec_a = probe_df.loc[probe_df["tissue"] == tissue_a, probes].iloc[0]
        for tissue_b in tissues[i + 1 :]:
            vec_b = probe_df.loc[probe_df["tissue"] == tissue_b, probes].iloc[0]
            rows.append(
                {
                    "cell_budget": cell_budget,
                    "tissue_a": tissue_a,
                    "tissue_b": tissue_b,
                    "spearman_rank_corr": _spearman_rank_correlation(vec_a, vec_b),
                }
            )

    return pd.DataFrame(rows).sort_values(
        by=["cell_budget", "tissue_a", "tissue_b"], ignore_index=True
    )


def compute_cell_budget_shift(df_30: pd.DataFrame, df_100: pd.DataFrame) -> pd.DataFrame:
    """Per tissue/probe change when moving from 30-cell to 100-cell setting."""

    probes = [col for col in df_30.columns if col != "tissue"]
    rows: list[dict[str, object]] = []

    for tissue in df_30["tissue"]:
        row_30 = df_30.loc[df_30["tissue"] == tissue].iloc[0]
        row_100 = df_100.loc[df_100["tissue"] == tissue].iloc[0]
        for probe in probes:
            value_30 = float(row_30[probe])
            value_100 = float(row_100[probe])
            delta = value_100 - value_30
            pct_change = (delta / value_30) if value_30 != 0 else float("nan")
            rows.append(
                {
                    "tissue": tissue,
                    "probe": probe,
                    "aupr_30_cell": value_30,
                    "aupr_100_cell": value_100,
                    "delta_100_minus_30": delta,
                    "pct_change_100_vs_30": pct_change,
                }
            )

    return pd.DataFrame(rows).sort_values(by=["tissue", "probe"], ignore_index=True)


def compute_stability_association(
    aupr_30_df: pd.DataFrame, stability_df: pd.DataFrame
) -> pd.DataFrame:
    """Tissue-level Spearman correlation between AUPR and stability across probes."""

    probes = [col for col in aupr_30_df.columns if col != "tissue"]
    rows: list[dict[str, object]] = []

    for tissue in aupr_30_df["tissue"]:
        aupr_values = aupr_30_df.loc[aupr_30_df["tissue"] == tissue, probes].iloc[0]
        stability_values = stability_df.loc[stability_df["tissue"] == tissue, probes].iloc[0]
        rows.append(
            {
                "tissue": tissue,
                "spearman_corr_aupr_vs_stability": _spearman_rank_correlation(
                    aupr_values, stability_values
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(by=["tissue"], ignore_index=True)


def compute_best_probe_reference_consistency(best_reference_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize how consistent best probes are per reference across tissues."""

    df = best_reference_df.copy()
    df["aupr_mean"] = df["aupr_mean"].astype(float)
    rows: list[dict[str, object]] = []

    for reference, group in df.groupby("reference", sort=True):
        counts = group["best_probe"].value_counts()
        rows.append(
            {
                "reference": reference,
                "num_tissues": int(group["tissue"].nunique()),
                "num_unique_best_probes": int(group["best_probe"].nunique()),
                "modal_best_probe": counts.index[0],
                "modal_probe_fraction": float(counts.iloc[0] / len(group)),
                "best_aupr_range": float(group["aupr_mean"].max() - group["aupr_mean"].min()),
            }
        )

    return pd.DataFrame(rows).sort_values(by=["reference"], ignore_index=True)


def _best_sweep_row(sweep_df: pd.DataFrame, metric: str) -> pd.Series | None:
    if sweep_df.empty:
        return None
    return sweep_df.loc[sweep_df[metric].idxmax()]


def compute_sweep_threshold_tables(sweep_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build threshold-robustness tables from sweep_results_*.json files."""

    summary_rows: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []

    for sweep_path in sorted(sweep_dir.glob("sweep_results_*.json")):
        payload = json.loads(sweep_path.read_text(encoding="utf-8"))
        percentile_df = pd.DataFrame(payload.get("percentile_sweep", []))
        topk_df = pd.DataFrame(payload.get("top_k_sweep", []))

        best_percentile_row = _best_sweep_row(percentile_df, "f1")
        best_topk_row = _best_sweep_row(topk_df, "f1")

        summary_rows.append(
            {
                "sweep_file": sweep_path.name,
                "candidate_edges": payload.get("candidate_edges"),
                "candidate_aupr": payload.get("aupr"),
                "best_percentile": None
                if best_percentile_row is None
                else float(best_percentile_row["percentile"]),
                "best_percentile_threshold": None
                if best_percentile_row is None
                else float(best_percentile_row["threshold"]),
                "best_percentile_f1": None
                if best_percentile_row is None
                else float(best_percentile_row["f1"]),
                "best_top_k": None
                if best_topk_row is None
                else int(best_topk_row["top_k"]),
                "best_top_k_f1": None
                if best_topk_row is None
                else float(best_topk_row["f1"]),
            }
        )

        if best_percentile_row is None or percentile_df.empty:
            continue

        for fixed_percentile in (90.0, 95.0):
            match = percentile_df[percentile_df["percentile"] == fixed_percentile]
            if match.empty:
                continue
            fixed_row = match.iloc[0]
            fixed_rows.append(
                {
                    "sweep_file": sweep_path.name,
                    "fixed_percentile": int(fixed_percentile),
                    "fixed_threshold": float(fixed_row["threshold"]),
                    "fixed_f1": float(fixed_row["f1"]),
                    "best_percentile": float(best_percentile_row["percentile"]),
                    "best_percentile_f1": float(best_percentile_row["f1"]),
                    "delta_best_minus_fixed_f1": float(best_percentile_row["f1"] - fixed_row["f1"]),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["sweep_file"], ignore_index=True)
    fixed_df = pd.DataFrame(fixed_rows).sort_values(
        by=["sweep_file", "fixed_percentile"], ignore_index=True
    )
    return summary_df, fixed_df


def compute_policy_transfer_benchmark(probe_df: pd.DataFrame, cell_budget: str) -> pd.DataFrame:
    """Benchmark transfer policies against tissue-wise oracle performance."""

    probes = [col for col in probe_df.columns if col != "tissue"]
    tissue_scores = probe_df.set_index("tissue")

    oracle_mean = float(tissue_scores.max(axis=1).mean())
    oracle_min = float(tissue_scores.max(axis=1).min())
    rows: list[dict[str, object]] = []

    # Source-best policy per source tissue.
    for source_tissue, source_row in tissue_scores.iterrows():
        source_best_probe = str(source_row.idxmax())
        values = tissue_scores[source_best_probe].astype(float)
        rows.append(
            {
                "cell_budget": cell_budget,
                "policy": f"source_best::{source_tissue}",
                "probe": source_best_probe,
                "mean_aupr_across_tissues": float(values.mean()),
                "min_aupr_across_tissues": float(values.min()),
                "std_aupr_across_tissues": float(values.std(ddof=0)),
                "oracle_mean_aupr": oracle_mean,
                "oracle_min_aupr": oracle_min,
                "mean_regret_vs_oracle": float(oracle_mean - values.mean()),
            }
        )

    # Single fixed probe policies.
    for probe in probes:
        values = tissue_scores[probe].astype(float)
        rows.append(
            {
                "cell_budget": cell_budget,
                "policy": f"fixed_probe::{probe}",
                "probe": probe,
                "mean_aupr_across_tissues": float(values.mean()),
                "min_aupr_across_tissues": float(values.min()),
                "std_aupr_across_tissues": float(values.std(ddof=0)),
                "oracle_mean_aupr": oracle_mean,
                "oracle_min_aupr": oracle_min,
                "mean_regret_vs_oracle": float(oracle_mean - values.mean()),
            }
        )

    # Uniform random expected policy over probe choices.
    random_expected = tissue_scores.mean(axis=1).astype(float)
    rows.append(
        {
            "cell_budget": cell_budget,
            "policy": "random_probe_expected",
            "probe": "uniform_over_probes",
            "mean_aupr_across_tissues": float(random_expected.mean()),
            "min_aupr_across_tissues": float(random_expected.min()),
            "std_aupr_across_tissues": float(random_expected.std(ddof=0)),
            "oracle_mean_aupr": oracle_mean,
            "oracle_min_aupr": oracle_min,
            "mean_regret_vs_oracle": float(oracle_mean - random_expected.mean()),
        }
    )

    return pd.DataFrame(rows).sort_values(
        by=["cell_budget", "mean_regret_vs_oracle", "policy"], ignore_index=True
    )


def summarize_fixed_percentile_regret(fixed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate best-vs-fixed percentile penalties for portability reporting."""

    if fixed_df.empty:
        return pd.DataFrame(
            columns=[
                "fixed_percentile",
                "mean_delta_best_minus_fixed_f1",
                "median_delta_best_minus_fixed_f1",
                "max_delta_best_minus_fixed_f1",
            ]
        )

    return (
        fixed_df.groupby("fixed_percentile", as_index=False)
        .agg(
            mean_delta_best_minus_fixed_f1=("delta_best_minus_fixed_f1", "mean"),
            median_delta_best_minus_fixed_f1=("delta_best_minus_fixed_f1", "median"),
            max_delta_best_minus_fixed_f1=("delta_best_minus_fixed_f1", "max"),
        )
        .sort_values(by=["fixed_percentile"], ignore_index=True)
    )


def run(mvp_report_path: Path, sweep_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_text = mvp_report_path.read_text(encoding="utf-8")

    table_30 = _coerce_probe_table(
        extract_markdown_table(report_text, "### Mean AUPR Across References (30-cell, MPS)")
    )
    table_100 = _coerce_probe_table(
        extract_markdown_table(report_text, "### Mean AUPR Across References (100-cell, MPS)")
    )
    table_stability = _coerce_probe_table(
        extract_markdown_table(report_text, "### Stability (Mean Jaccard@Top-k)")
    )
    table_best_reference = extract_markdown_table(report_text, "### Best AUPR per Reference")
    table_best_reference = table_best_reference.copy()
    table_best_reference["aupr_mean"] = table_best_reference["aupr_mean"].astype(float)

    transfer_30 = compute_transfer_regret(table_30, "30")
    transfer_100 = compute_transfer_regret(table_100, "100")
    transfer_summary = summarize_transfer_regret(pd.concat([transfer_30, transfer_100], ignore_index=True))
    policy_30 = compute_policy_transfer_benchmark(table_30, "30")
    policy_100 = compute_policy_transfer_benchmark(table_100, "100")
    policy_summary = pd.concat([policy_30, policy_100], ignore_index=True)
    rank_30 = compute_rank_correlations(table_30, "30")
    rank_100 = compute_rank_correlations(table_100, "100")
    budget_shift = compute_cell_budget_shift(table_30, table_100)
    stability_assoc = compute_stability_association(table_30, table_stability)
    best_reference_consistency = compute_best_probe_reference_consistency(table_best_reference)
    sweep_summary, sweep_fixed = compute_sweep_threshold_tables(sweep_dir)
    sweep_fixed_summary = summarize_fixed_percentile_regret(sweep_fixed)

    transfer_30.to_csv(output_dir / "transfer_regret_30_cell.csv", index=False)
    transfer_100.to_csv(output_dir / "transfer_regret_100_cell.csv", index=False)
    transfer_summary.to_csv(output_dir / "transfer_regret_summary.csv", index=False)
    policy_30.to_csv(output_dir / "policy_transfer_benchmark_30_cell.csv", index=False)
    policy_100.to_csv(output_dir / "policy_transfer_benchmark_100_cell.csv", index=False)
    policy_summary.to_csv(output_dir / "policy_transfer_benchmark_summary.csv", index=False)
    table_30.to_csv(output_dir / "mvp_mean_aupr_30_cell.csv", index=False)
    table_100.to_csv(output_dir / "mvp_mean_aupr_100_cell.csv", index=False)
    table_stability.to_csv(output_dir / "mvp_stability_30_cell.csv", index=False)
    table_best_reference.to_csv(output_dir / "mvp_best_probe_per_reference.csv", index=False)
    rank_30.to_csv(output_dir / "probe_rank_spearman_30_cell.csv", index=False)
    rank_100.to_csv(output_dir / "probe_rank_spearman_100_cell.csv", index=False)
    budget_shift.to_csv(output_dir / "cell_budget_shift_30_to_100.csv", index=False)
    stability_assoc.to_csv(output_dir / "stability_aupr_spearman_30_cell.csv", index=False)
    best_reference_consistency.to_csv(
        output_dir / "best_probe_reference_consistency.csv", index=False
    )
    sweep_summary.to_csv(output_dir / "sweep_threshold_summary.csv", index=False)
    sweep_fixed.to_csv(output_dir / "sweep_threshold_fixed_percentile_regret.csv", index=False)
    sweep_fixed_summary.to_csv(output_dir / "sweep_threshold_fixed_percentile_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate transfer-analysis artifacts for subproject 04."
    )
    parser.add_argument(
        "--mvp-report",
        type=Path,
        default=Path(
            "/Users/ihorkendiukhov/biodyn-work/reports/cross_tissue_probe_threshold_transfer/"
            "probe_benchmark_workshop_mvp.md"
        ),
        help="Path to the workshop MVP markdown report.",
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path(
            "/Users/ihorkendiukhov/biodyn-work/subproject_04_cross_tissue_probe_threshold_transfer/"
            "implementation/network_inference_materials/outputs"
        ),
        help="Directory containing sweep_results_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/Users/ihorkendiukhov/biodyn-work/subproject_04_cross_tissue_probe_threshold_transfer/"
            "reports/artifacts"
        ),
        help="Directory where derived CSV artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.mvp_report, args.sweep_dir, args.output_dir)


if __name__ == "__main__":
    main()
