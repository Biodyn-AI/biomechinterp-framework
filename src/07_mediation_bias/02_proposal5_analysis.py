#!/usr/bin/env python3
"""Execute Proposal 5 (Bias bounds for patching-based mediation estimates).

This script is a low-compute, theory-first workflow that:
1. Loads existing single-mediator checkpoint summaries from Subproject 09.
2. Derives residual non-additivity diagnostics from total-effect vs single-effect mismatch.
3. Computes ranking-certification diagnostics under conservative bias budgets.
4. Runs a lightweight synthetic sanity simulation for the certification rule.
5. Writes CSV artifacts, figures, and a report-ready markdown summary.

Theoretical setup used in the report:
- Let F(S) be the expected model output when mediators in S are patched to clean and
  all others are ablated.
- Let mu(S) be the Mobius interaction coefficient of F.
- Single-component patching estimates F({i}) - F(empty), while interaction terms
  mu(S), |S| >= 2 induce structural bias for mediator attribution/ranking.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Z_95 = 1.96
EPS = 1e-12
RNG_SEED = 17


@dataclass(frozen=True)
class WelfordStat:
    """Mean/variance accumulator values stored in checkpoint.json."""

    count: int
    mean: float
    m2: float

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return float("nan")
        return float(self.m2) / float(self.count - 1)

    @property
    def se(self) -> float:
        var = self.variance
        if not np.isfinite(var):
            return float("nan")
        return math.sqrt(var / float(self.count))


def find_repo_root(start: Path) -> Path:
    """Find repository root by looking for known top-level folders."""
    for candidate in [start] + list(start.parents):
        if (candidate / "market_research").exists() and (candidate / "single_cell_mechinterp").exists():
            return candidate
    raise RuntimeError("Could not infer repo root from script location.")


def parse_pair_key(pair_key: str) -> Tuple[str, str]:
    """Parse checkpoint pair key format `SOURCE||TARGET`."""
    if "||" in pair_key:
        source, target = pair_key.split("||", 1)
        return source, target
    return pair_key, pair_key


def load_stat(payload: Dict[str, object]) -> WelfordStat:
    """Convert JSON stat payload into typed structure."""
    return WelfordStat(
        count=int(payload.get("count", 0)),
        mean=float(payload.get("mean", float("nan"))),
        m2=float(payload.get("m2", float("nan"))),
    )


def discover_checkpoint_paths(repo_root: Path) -> List[Path]:
    """Discover mediation checkpoint files produced by Subproject 09."""
    pattern = "subproject_09_causal_mediation_circuit_map/implementation/results/*/*/checkpoint.json"
    paths = sorted(repo_root.glob(pattern))
    if not paths:
        raise FileNotFoundError("No checkpoint.json files found for mediation re-analysis.")
    return paths


def load_checkpoint_tables(checkpoint_paths: Sequence[Path], repo_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pair-level and component-level tables from checkpoint JSON files."""
    pair_rows: List[Dict[str, object]] = []
    component_rows: List[Dict[str, object]] = []

    for path in checkpoint_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pair_stats = payload.get("pair_stats", {})
        component_stats = payload.get("component_stats", {})

        granularity = path.parent.name
        tissue = path.parent.parent.name
        run_id = f"{tissue}:{granularity}"
        rel_path = str(path.relative_to(repo_root))

        for pair_key, stats_obj in pair_stats.items():
            source, target = parse_pair_key(str(pair_key))
            stats_dict = dict(stats_obj)
            ablation_payload = stats_dict.get("ablation")
            if not isinstance(ablation_payload, dict):
                continue
            ablation = load_stat(ablation_payload)

            swap = None
            swap_payload = stats_dict.get("swap")
            if isinstance(swap_payload, dict):
                swap = load_stat(swap_payload)

            pair_rows.append(
                {
                    "run_id": run_id,
                    "tissue": tissue,
                    "granularity": granularity,
                    "pair_key": str(pair_key),
                    "source": source,
                    "target": target,
                    "pair": f"{source}->{target}",
                    "te_mean": ablation.mean,
                    "te_count": ablation.count,
                    "te_se": ablation.se,
                    "swap_mean": swap.mean if swap else float("nan"),
                    "swap_count": swap.count if swap else 0,
                    "swap_se": swap.se if swap else float("nan"),
                    "checkpoint_path": rel_path,
                }
            )

        for pair_key, component_dict in component_stats.items():
            source, target = parse_pair_key(str(pair_key))
            for component_name, stat_payload in dict(component_dict).items():
                if not isinstance(stat_payload, dict):
                    continue
                stat = load_stat(stat_payload)
                component_rows.append(
                    {
                        "run_id": run_id,
                        "tissue": tissue,
                        "granularity": granularity,
                        "pair_key": str(pair_key),
                        "source": source,
                        "target": target,
                        "pair": f"{source}->{target}",
                        "component": str(component_name),
                        "effect_mean": stat.mean,
                        "effect_abs": abs(stat.mean),
                        "effect_count": stat.count,
                        "effect_se": stat.se,
                        "effect_ci95": Z_95 * stat.se if np.isfinite(stat.se) else float("nan"),
                        "checkpoint_path": rel_path,
                    }
                )

    pair_df = pd.DataFrame(pair_rows)
    component_df = pd.DataFrame(component_rows)

    if pair_df.empty:
        raise RuntimeError("Pair table is empty after loading checkpoints.")
    if component_df.empty:
        raise RuntimeError("Component table is empty after loading checkpoints.")

    return pair_df, component_df


def compute_pair_diagnostics(pair_df: pd.DataFrame, component_df: pd.DataFrame) -> pd.DataFrame:
    """Compute residual and non-additivity diagnostics per run/pair."""
    diagnostics: List[Dict[str, object]] = []

    grouped = component_df.groupby(["run_id", "pair"], sort=True)
    pair_index = pair_df.set_index(["run_id", "pair"])

    for (run_id, pair), group in grouped:
        if (run_id, pair) not in pair_index.index:
            continue

        pair_row = pair_index.loc[(run_id, pair)]
        # In case of duplicate rows from repeated loads, keep the first row.
        if isinstance(pair_row, pd.DataFrame):
            pair_row = pair_row.iloc[0]

        te_mean = float(pair_row["te_mean"])
        te_se = float(pair_row["te_se"])

        sum_single = float(group["effect_mean"].sum())
        sum_abs_single = float(group["effect_abs"].sum())

        # Sampling error propagation under independence approximation.
        se_sum_single = math.sqrt(float(np.nansum(np.square(group["effect_se"].to_numpy(dtype=float)))))
        residual = te_mean - sum_single
        residual_se = math.sqrt((te_se ** 2 if np.isfinite(te_se) else 0.0) + se_sum_single ** 2)
        residual_ci95 = Z_95 * residual_se

        # Lower bound on aggregate structural non-additivity (interaction + higher-order terms).
        nonadd_lb = max(0.0, abs(residual) - residual_ci95)
        nonadd_ub = abs(residual) + residual_ci95

        diagnostics.append(
            {
                "run_id": run_id,
                "pair": pair,
                "source": str(pair_row["source"]),
                "target": str(pair_row["target"]),
                "granularity": str(pair_row["granularity"]),
                "components": int(group.shape[0]),
                "te_mean": te_mean,
                "te_se": te_se,
                "sum_single_mean": sum_single,
                "sum_abs_single_mean": sum_abs_single,
                "residual_mean": residual,
                "residual_abs": abs(residual),
                "residual_se": residual_se,
                "residual_ci95": residual_ci95,
                "nonadditivity_lb": nonadd_lb,
                "nonadditivity_ub": nonadd_ub,
                "nonadditivity_lb_over_te": nonadd_lb / max(abs(te_mean), EPS),
                "checkpoint_path": str(pair_row["checkpoint_path"]),
            }
        )

    diag_df = pd.DataFrame(diagnostics)
    if diag_df.empty:
        raise RuntimeError("No pair-level diagnostics computed.")
    return diag_df.sort_values(["run_id", "pair"]).reset_index(drop=True)


def build_ranking_certification(
    component_df: pd.DataFrame,
    pair_diag_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute certifiable ranking comparisons under residual-bias budgets.

    Two structural-bias allocation models are reported:
    - `uniform`: non-additivity lower-bound mass distributed evenly across components.
    - `worst_case`: any one component can absorb the full non-additivity lower-bound mass.
    """
    pair_budget_map = {
        (row.run_id, row.pair): float(row.nonadditivity_lb)
        for row in pair_diag_df.itertuples(index=False)
    }

    pair_rows: List[Dict[str, object]] = []
    compare_rows: List[Dict[str, object]] = []

    for (run_id, pair), group in component_df.groupby(["run_id", "pair"], sort=True):
        budget = pair_budget_map.get((run_id, pair), 0.0)
        ranked = group.sort_values("effect_abs", ascending=False).reset_index(drop=True)
        n_comp = int(ranked.shape[0])
        if n_comp < 2:
            continue

        total_pairs = 0
        cert_uniform = 0
        cert_worst = 0

        for i, j in itertools.combinations(range(n_comp), 2):
            row_i = ranked.iloc[i]
            row_j = ranked.iloc[j]

            gap = float(row_i["effect_abs"] - row_j["effect_abs"])
            sample_budget = float(row_i["effect_ci95"] if np.isfinite(row_i["effect_ci95"]) else 0.0)
            sample_budget += float(row_j["effect_ci95"] if np.isfinite(row_j["effect_ci95"]) else 0.0)

            structural_uniform = 2.0 * (budget / float(n_comp))
            structural_worst = 2.0 * budget

            margin_uniform = gap - (sample_budget + structural_uniform)
            margin_worst = gap - (sample_budget + structural_worst)

            is_cert_uniform = margin_uniform > 0.0
            is_cert_worst = margin_worst > 0.0

            total_pairs += 1
            cert_uniform += int(is_cert_uniform)
            cert_worst += int(is_cert_worst)

            compare_rows.append(
                {
                    "run_id": run_id,
                    "pair": pair,
                    "component_hi": str(row_i["component"]),
                    "component_lo": str(row_j["component"]),
                    "abs_gap": gap,
                    "sample_budget": sample_budget,
                    "nonadditivity_lb": budget,
                    "uniform_margin": margin_uniform,
                    "worst_case_margin": margin_worst,
                    "certified_uniform": bool(is_cert_uniform),
                    "certified_worst_case": bool(is_cert_worst),
                }
            )

        top1 = ranked.iloc[0]
        top2 = ranked.iloc[1]
        top_gap = float(top1["effect_abs"] - top2["effect_abs"])

        pair_rows.append(
            {
                "run_id": run_id,
                "pair": pair,
                "components": n_comp,
                "nonadditivity_lb": budget,
                "top1_component": str(top1["component"]),
                "top2_component": str(top2["component"]),
                "top1_abs_effect": float(top1["effect_abs"]),
                "top2_abs_effect": float(top2["effect_abs"]),
                "top1_top2_gap": top_gap,
                "comparisons": total_pairs,
                "certified_fraction_uniform": cert_uniform / float(total_pairs),
                "certified_fraction_worst_case": cert_worst / float(total_pairs),
            }
        )

    summary_df = pd.DataFrame(pair_rows).sort_values(["run_id", "pair"]).reset_index(drop=True)
    detail_df = pd.DataFrame(compare_rows).sort_values(["run_id", "pair", "abs_gap"], ascending=[True, True, False])

    return summary_df, detail_df


def build_pairwise_patch_candidates(component_df: pd.DataFrame, top_r: int = 4) -> pd.DataFrame:
    """Suggest top pairwise patch candidates to shrink interaction uncertainty."""
    rows: List[Dict[str, object]] = []

    for (run_id, pair), group in component_df.groupby(["run_id", "pair"], sort=True):
        ranked = group.sort_values("effect_abs", ascending=False).head(top_r).reset_index(drop=True)
        for i, j in itertools.combinations(range(ranked.shape[0]), 2):
            ri = ranked.iloc[i]
            rj = ranked.iloc[j]
            rows.append(
                {
                    "run_id": run_id,
                    "pair": pair,
                    "component_i": str(ri["component"]),
                    "component_j": str(rj["component"]),
                    "priority_score": float(ri["effect_abs"] + rj["effect_abs"]),
                    "i_abs_effect": float(ri["effect_abs"]),
                    "j_abs_effect": float(rj["effect_abs"]),
                }
            )

    candidate_df = pd.DataFrame(rows)
    if candidate_df.empty:
        return candidate_df

    return candidate_df.sort_values(["run_id", "pair", "priority_score"], ascending=[True, True, False]).reset_index(drop=True)


def run_synthetic_certification_grid(
    trials_per_cell: int = 2500,
    k: int = 8,
    noise_scale: float = 0.7,
) -> pd.DataFrame:
    """Simulate ranking-certification behavior under interaction + finite-sample noise.

    Synthetic model per trial:
    - true mediator effect: tau_i ~ N(0, 1)
    - omitted interaction bias: b_i = 0.5 * sum_j beta_ij
    - beta_ij ~ N(0, interaction_scale / sqrt(k)) for i < j
    - patching estimator: hat_i = tau_i + b_i + epsilon_i, epsilon_i ~ N(0, noise_scale/sqrt(n))

    Certified ordering condition tested:
        |hat_i - hat_j| > B_i + B_j,
    where B_i = 0.5 * sum_j |beta_ij| + 1.96 * noise_scale/sqrt(n).
    """
    interaction_grid = [0.0, 0.1, 0.2, 0.4, 0.8, 1.2]
    n_grid = [5, 10, 20, 40, 80, 160]

    rng = np.random.default_rng(RNG_SEED)
    rows: List[Dict[str, object]] = []

    for interaction_scale in interaction_grid:
        for n_cells in n_grid:
            total_pairs = 0
            certified_pairs = 0
            certified_errors = 0
            naive_errors = 0

            noise_sd = noise_scale / math.sqrt(float(n_cells))

            for _ in range(trials_per_cell):
                tau = rng.normal(0.0, 1.0, size=k)

                beta = np.zeros((k, k), dtype=float)
                for i in range(k):
                    for j in range(i + 1, k):
                        val = rng.normal(0.0, interaction_scale / math.sqrt(float(k)))
                        beta[i, j] = val
                        beta[j, i] = val

                omitted_bias = 0.5 * beta.sum(axis=1)
                eps = rng.normal(0.0, noise_sd, size=k)
                hat = tau + omitted_bias + eps

                bound = 0.5 * np.sum(np.abs(beta), axis=1) + Z_95 * noise_sd

                for i in range(k):
                    for j in range(i + 1, k):
                        total_pairs += 1
                        hat_diff = float(hat[i] - hat[j])
                        true_diff = float(tau[i] - tau[j])

                        # Baseline ordering error without certification filtering.
                        if hat_diff * true_diff <= 0:
                            naive_errors += 1

                        # Theorem-backed certification filter.
                        if abs(hat_diff) > float(bound[i] + bound[j]):
                            certified_pairs += 1
                            if hat_diff * true_diff <= 0:
                                certified_errors += 1

            rows.append(
                {
                    "interaction_scale": interaction_scale,
                    "n_cells": n_cells,
                    "total_pairs": total_pairs,
                    "certified_pairs": certified_pairs,
                    "certified_fraction": certified_pairs / float(total_pairs),
                    "certified_error_rate": certified_errors / float(max(certified_pairs, 1)),
                    "naive_error_rate": naive_errors / float(total_pairs),
                }
            )

    return pd.DataFrame(rows)


def plot_residual_bars(pair_diag_df: pd.DataFrame, output_path: Path) -> None:
    """Plot absolute total effect vs residual non-additivity diagnostics."""
    data = pair_diag_df.copy()
    data["label"] = data["run_id"] + "\n" + data["pair"]

    x = np.arange(data.shape[0])
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * data.shape[0]), 5.5), dpi=170)
    ax.bar(x - w, data["te_mean"].abs().to_numpy(), width=w, label="|Total effect|", color="#3B6FB6")
    ax.bar(x, data["residual_abs"].to_numpy(), width=w, label="|TE - sum(single)|", color="#D77A61")
    ax.bar(x + w, data["nonadditivity_lb"].to_numpy(), width=w, label="Non-additivity lower bound", color="#4A9F74")

    ax.set_xticks(x)
    ax.set_xticklabels(data["label"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("Effect magnitude")
    ax.set_title("Residual non-additivity diagnostics from existing mediation checkpoints")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_heatmap_panel(ax: plt.Axes, matrix: np.ndarray, xvals: Sequence[int], yvals: Sequence[float], title: str, cmap: str, vmin: float, vmax: float) -> None:
    """Internal helper for consistent heatmap rendering."""
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xticks(np.arange(len(xvals)))
    ax.set_xticklabels([str(x) for x in xvals])
    ax.set_yticks(np.arange(len(yvals)))
    ax.set_yticklabels([f"{y:.1f}" for y in yvals])
    ax.set_xlabel("n_cells")
    ax.set_ylabel("interaction_scale")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_synthetic_heatmap(sim_df: pd.DataFrame, output_path: Path) -> None:
    """Plot certification fraction and certified error rate on the synthetic grid."""
    xvals = sorted(sim_df["n_cells"].unique().tolist())
    yvals = sorted(sim_df["interaction_scale"].unique().tolist())

    cert = np.zeros((len(yvals), len(xvals)), dtype=float)
    err = np.zeros((len(yvals), len(xvals)), dtype=float)

    row_index = {v: i for i, v in enumerate(yvals)}
    col_index = {v: j for j, v in enumerate(xvals)}

    for row in sim_df.itertuples(index=False):
        r = row_index[float(row.interaction_scale)]
        c = col_index[int(row.n_cells)]
        cert[r, c] = float(row.certified_fraction)
        err[r, c] = float(row.certified_error_rate)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=170)
    _plot_heatmap_panel(
        axes[0],
        cert,
        xvals=xvals,
        yvals=yvals,
        title="Certified pair fraction",
        cmap="Blues",
        vmin=0.0,
        vmax=max(0.25, float(np.nanmax(cert))),
    )
    _plot_heatmap_panel(
        axes[1],
        err,
        xvals=xvals,
        yvals=yvals,
        title="Error rate among certified pairs",
        cmap="Reds",
        vmin=0.0,
        vmax=max(0.05, float(np.nanmax(err))),
    )

    fig.suptitle("Synthetic sanity check for ranking-certification bound", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def format_table(df: pd.DataFrame, max_rows: int = 12, float_fmt: str = "{:.4f}") -> str:
    """Render dataframe excerpt as fixed-width text for markdown report."""
    if df.empty:
        return "(empty)"
    excerpt = df.head(max_rows).copy()

    for col in excerpt.select_dtypes(include=["float", "float64", "float32"]).columns:
        excerpt[col] = excerpt[col].map(lambda v: float_fmt.format(v) if np.isfinite(v) else "nan")

    return excerpt.to_string(index=False)


def write_report(
    report_path: Path,
    pair_diag_df: pd.DataFrame,
    ranking_summary_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    checkpoint_paths: Sequence[Path],
    repo_root: Path,
) -> None:
    """Write markdown report with theory and empirical diagnostics."""
    pair_short = pair_diag_df[
        [
            "run_id",
            "pair",
            "te_mean",
            "sum_single_mean",
            "residual_mean",
            "nonadditivity_lb",
            "nonadditivity_lb_over_te",
        ]
    ].copy()

    rank_short = ranking_summary_df[
        [
            "run_id",
            "pair",
            "top1_component",
            "top2_component",
            "top1_top2_gap",
            "certified_fraction_uniform",
            "certified_fraction_worst_case",
        ]
    ].copy()

    sim_short = sim_df[
        [
            "interaction_scale",
            "n_cells",
            "certified_fraction",
            "certified_error_rate",
            "naive_error_rate",
        ]
    ].copy()

    sim_agg = (
        sim_df.groupby("interaction_scale", as_index=False)
        .agg(
            mean_certified_fraction=("certified_fraction", "mean"),
            mean_certified_error_rate=("certified_error_rate", "mean"),
            mean_naive_error_rate=("naive_error_rate", "mean"),
        )
        .sort_values("interaction_scale")
    )

    top_candidates = candidate_df.head(12).copy()

    rel_checkpoints = [str(p.relative_to(repo_root)) for p in checkpoint_paths]

    lines: List[str] = []
    lines.append("# Proposal 5 Execution: Bias Bounds for Patching-Based Mediation Estimates")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Source prompt: `market_research/ambitious_paper_questions/theory_first_low_compute_proposals.md` (Proposal 5).")
    lines.append("- Goal: quantify bias in single-component mediation patching when mediators interact.")
    lines.append("- Empirical footprint: re-analysis of existing checkpoint summaries only (no all-head resweep).")
    lines.append("- Checkpoints used:")
    for rel in rel_checkpoints:
        lines.append(f"  - `{rel}`")
    lines.append("")

    lines.append("## Formal Setup")
    lines.append("Let `F(S)` be the expected model output when mediators in set `S` are patched to clean and all others are ablated.")
    lines.append("Define Mobius interaction coefficients:")
    lines.append("`mu(S) = sum_{T subseteq S} (-1)^{|S|-|T|} F(T)`.")
    lines.append("Then total effect is:")
    lines.append("`TE = F([k]) - F(empty) = sum_{S != empty} mu(S)`.")
    lines.append("Single-mediator patching estimator:")
    lines.append("`mhat_i = F({i}) - F(empty) + eps_i = mu({i}) + eps_i`.")
    lines.append("")

    lines.append("## Theoretical Results")
    lines.append("1. Bias decomposition against Shapley attribution `phi_i = sum_{S contains i} mu(S)/|S|`:")
    lines.append("`b_i = mhat_i - phi_i = -sum_{|S|>=2, i in S} mu(S)/|S| + eps_i`.")
    lines.append("Split terms:")
    lines.append("- omitted pairwise interaction bias: `b_i_pair = -(1/2) * sum_{j!=i} mu({i,j})`")
    lines.append("- higher-order nonlinear composition bias: `b_i_high = -sum_{|S|>=3, i in S} mu(S)/|S|`")
    lines.append("- finite-sample patching error: `eps_i`")
    lines.append("2. Ranking-validity condition: if `mhat_i - mhat_j > B_i + B_j` and `|b_i|<=B_i, |b_j|<=B_j`, then `phi_i > phi_j`.")
    lines.append("3. Observable residual identity:")
    lines.append("`R = TE - sum_i mhat_i = sum_{|S|>=2} mu(S) - sum_i eps_i`.")
    lines.append("So aggregate structural non-additivity `A = sum_{|S|>=2}|mu(S)|` satisfies")
    lines.append("`A >= |R| - |sum_i eps_i|`.")
    lines.append("We estimate a practical lower bound with propagated 95% sampling error:")
    lines.append("`A_lb = max(0, |R| - 1.96 * SE(R))`.")
    lines.append("")

    lines.append("## Empirical Diagnostics (Existing Mediator Set)")
    lines.append("Pair-level decomposition summary:")
    lines.append("```text")
    lines.append(format_table(pair_short, max_rows=20))
    lines.append("```")
    lines.append("")
    lines.append("Ranking-certification summary:")
    lines.append("```text")
    lines.append(format_table(rank_short, max_rows=20))
    lines.append("```")
    lines.append("")
    lines.append("Top pairwise patch candidates (for reducing interaction uncertainty):")
    lines.append("```text")
    lines.append(format_table(top_candidates, max_rows=12))
    lines.append("```")
    lines.append("")

    lines.append("## Synthetic Sanity Check")
    lines.append("Grid-average behavior by interaction strength:")
    lines.append("```text")
    lines.append(format_table(sim_agg, max_rows=20))
    lines.append("```")
    lines.append("Sample of full grid rows:")
    lines.append("```text")
    lines.append(format_table(sim_short, max_rows=18))
    lines.append("```")
    lines.append("")

    lines.append("## Correction Strategies")
    lines.append("1. Pairwise augmentation (recommended first): estimate `mhat_{ij} = F({i,j}) - F(empty)` for top mediator pairs, then")
    lines.append("   `mtilde_i = mhat_i + 0.5 * sum_j (mhat_{ij} - mhat_i - mhat_j)` over measured neighbors.")
    lines.append("2. Residual-bound correction (no new runs): report interval")
    lines.append("   `phi_i in [mhat_i - A_lb - 1.96*SE_i, mhat_i + A_lb + 1.96*SE_i]`.")
    lines.append("   For less conservative reporting, distribute `A_lb` uniformly over components in the pair.")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("- Pair diagnostics: `subproject_22_patching_mediation_bias_bounds/outputs/pair_bias_diagnostics.csv`")
    lines.append("- Component effects: `subproject_22_patching_mediation_bias_bounds/outputs/component_effects.csv`")
    lines.append("- Ranking summary: `subproject_22_patching_mediation_bias_bounds/outputs/ranking_certification_summary.csv`")
    lines.append("- Ranking detail: `subproject_22_patching_mediation_bias_bounds/outputs/ranking_certification_detail.csv`")
    lines.append("- Pairwise candidate list: `subproject_22_patching_mediation_bias_bounds/outputs/pairwise_patch_candidates.csv`")
    lines.append("- Synthetic grid: `subproject_22_patching_mediation_bias_bounds/outputs/synthetic_certification_grid.csv`")
    lines.append("- Residual figure: `subproject_22_patching_mediation_bias_bounds/outputs/figures/residual_nonadditivity_bars.png`")
    lines.append("- Synthetic heatmap: `subproject_22_patching_mediation_bias_bounds/outputs/figures/synthetic_certification_heatmap.png`")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Proposal 5 low-compute analysis")
    parser.add_argument(
        "--output-dir",
        default="subproject_22_patching_mediation_bias_bounds/outputs",
        help="Directory for CSV outputs and figures",
    )
    parser.add_argument(
        "--report-path",
        default="subproject_22_patching_mediation_bias_bounds/reports/proposal5_bias_bounds_report.md",
        help="Markdown report path",
    )
    parser.add_argument(
        "--synthetic-trials",
        type=int,
        default=2500,
        help="Trials per synthetic grid cell",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    output_dir = repo_root / args.output_dir
    report_path = repo_root / args.report_path
    fig_dir = output_dir / "figures"

    checkpoint_paths = discover_checkpoint_paths(repo_root)
    pair_df, component_df = load_checkpoint_tables(checkpoint_paths, repo_root)

    pair_diag_df = compute_pair_diagnostics(pair_df, component_df)
    ranking_summary_df, ranking_detail_df = build_ranking_certification(component_df, pair_diag_df)
    candidate_df = build_pairwise_patch_candidates(component_df, top_r=4)
    sim_df = run_synthetic_certification_grid(trials_per_cell=args.synthetic_trials)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pair_diag_df.to_csv(output_dir / "pair_bias_diagnostics.csv", index=False)
    pair_df.to_csv(output_dir / "pair_effects.csv", index=False)
    component_df.to_csv(output_dir / "component_effects.csv", index=False)
    ranking_summary_df.to_csv(output_dir / "ranking_certification_summary.csv", index=False)
    ranking_detail_df.to_csv(output_dir / "ranking_certification_detail.csv", index=False)
    candidate_df.to_csv(output_dir / "pairwise_patch_candidates.csv", index=False)
    sim_df.to_csv(output_dir / "synthetic_certification_grid.csv", index=False)

    plot_residual_bars(pair_diag_df, fig_dir / "residual_nonadditivity_bars.png")
    plot_synthetic_heatmap(sim_df, fig_dir / "synthetic_certification_heatmap.png")

    write_report(
        report_path=report_path,
        pair_diag_df=pair_diag_df,
        ranking_summary_df=ranking_summary_df,
        candidate_df=candidate_df,
        sim_df=sim_df,
        checkpoint_paths=checkpoint_paths,
        repo_root=repo_root,
    )

    print(f"Loaded checkpoints: {len(checkpoint_paths)}")
    print(f"Wrote outputs to: {output_dir}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
