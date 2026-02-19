#!/usr/bin/env python3
"""
Proposal 3 execution: detectability phase diagram for mechanistic signals.

This script provides a theory-first, low-compute workflow:
1. Closed-form sample-complexity thresholds for edge detectability.
2. Attention-like vs causal-intervention-like phase-diagram comparison.
3. Lightweight synthetic validation under sub-Gaussian and heavy-tail noise.
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalFamily:
    """Signal family with stylized effect-size and noise scale assumptions."""

    name: str
    effect_size: float
    sigma: float


@dataclass(frozen=True)
class SimulationScenario:
    """Synthetic scenario specification for robustness checks."""

    name: str
    noise_model: str
    estimator: str
    student_df: float | None = None


def normal_quantile(probability: float) -> float:
    """Return the standard-normal quantile for a probability in (0, 1)."""
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError(f"Probability must be in (0, 1), got {probability}.")
    return statistics.NormalDist().inv_cdf(probability)


def sample_complexity(
    effect_size: float,
    sigma: float,
    alpha: float,
    power: float,
    m_tests: int,
    tail_inflation: float = 1.0,
) -> int:
    """
    Closed-form detectability threshold for two-sided z-style testing.

    Formula:
      n* = ((z_(1-alpha_eff/2) + z_(power)) * (tail_inflation * sigma) / |mu|)^2
      with alpha_eff = alpha / m_tests (Bonferroni control).
    """
    if effect_size <= 0.0:
        raise ValueError("effect_size must be positive.")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if not (0.0 < power < 1.0):
        raise ValueError("power must be in (0, 1).")
    if m_tests < 1:
        raise ValueError("m_tests must be >= 1.")
    if tail_inflation <= 0.0:
        raise ValueError("tail_inflation must be positive.")

    alpha_eff = alpha / m_tests
    z_alpha = normal_quantile(1.0 - alpha_eff / 2.0)
    z_power = normal_quantile(power)
    required_n = ((z_alpha + z_power) * tail_inflation * sigma / effect_size) ** 2
    return int(math.ceil(required_n))


def benjamini_hochberg_mask(p_values: np.ndarray, q_level: float) -> np.ndarray:
    """Return boolean discoveries selected by BH at target FDR q_level."""
    if p_values.ndim != 1:
        raise ValueError("p_values must be 1D.")
    if not (0.0 < q_level < 1.0):
        raise ValueError("q_level must be in (0, 1).")

    m_tests = p_values.size
    order = np.argsort(p_values)
    ranked = p_values[order]
    thresholds = q_level * (np.arange(1, m_tests + 1) / m_tests)
    passing_indices = np.where(ranked <= thresholds)[0]

    selected = np.zeros(m_tests, dtype=bool)
    if passing_indices.size == 0:
        return selected

    max_rank = passing_indices.max()
    cutoff = ranked[max_rank]
    selected[p_values <= cutoff] = True
    return selected


def z_to_two_sided_p_values(z_scores: np.ndarray) -> np.ndarray:
    """Convert z-scores to two-sided p-values using standard normal CDF."""
    normal = statistics.NormalDist()
    abs_z = np.abs(z_scores)
    return np.array(
        [2.0 * (1.0 - normal.cdf(float(value))) for value in abs_z], dtype=float
    )


def draw_noise(
    rng: np.random.Generator,
    noise_model: str,
    shape: tuple[int, int],
    sigma: float,
    student_df: float | None,
) -> np.ndarray:
    """Draw centered noise with target scale sigma under selected model."""
    if noise_model == "gaussian":
        return rng.normal(loc=0.0, scale=sigma, size=shape)

    if noise_model == "student_t":
        if student_df is None or student_df <= 2.0:
            raise ValueError("student_df must be > 2 for finite variance.")
        # Scale t noise to match the requested variance sigma^2.
        scale = sigma * math.sqrt((student_df - 2.0) / student_df)
        return rng.standard_t(df=student_df, size=shape) * scale

    raise ValueError(f"Unsupported noise_model: {noise_model}")


def estimate_location_and_scale(scores: np.ndarray, estimator: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-edge mean effect and standard deviation.

    The winsorized estimator clips each edge's sample values at [5%, 95%]
    to reduce heavy-tail sensitivity while keeping the implementation simple.
    """
    if estimator == "mean":
        location = scores.mean(axis=1)
        scale = scores.std(axis=1, ddof=1)
        return location, scale
    elif estimator == "winsorized":
        lower = np.quantile(scores, 0.05, axis=1, keepdims=True)
        upper = np.quantile(scores, 0.95, axis=1, keepdims=True)
        clipped = np.clip(scores, lower, upper)
        # Use robust location but raw scale to avoid anti-conservative p-values.
        location = clipped.mean(axis=1)
        scale = scores.std(axis=1, ddof=1)
        return location, scale
    else:
        raise ValueError(f"Unsupported estimator: {estimator}")


def simulate_replicate(
    rng: np.random.Generator,
    family: SignalFamily,
    n_cells: int,
    n_edges: int,
    true_edge_rate: float,
    q_level: float,
    scenario: SimulationScenario,
) -> tuple[float, float]:
    """Simulate one replicate and return (TPR, FDR)."""
    true_mask = rng.random(n_edges) < true_edge_rate
    edge_sign = rng.choice([-1.0, 1.0], size=n_edges)
    mu = np.where(true_mask, edge_sign * family.effect_size, 0.0)

    noise = draw_noise(
        rng=rng,
        noise_model=scenario.noise_model,
        shape=(n_edges, n_cells),
        sigma=family.sigma,
        student_df=scenario.student_df,
    )
    scores = mu[:, None] + noise

    location, scale = estimate_location_and_scale(scores, estimator=scenario.estimator)
    standard_error = np.maximum(scale / math.sqrt(n_cells), 1e-12)
    z_scores = location / standard_error
    p_values = z_to_two_sided_p_values(z_scores)
    discoveries = benjamini_hochberg_mask(p_values, q_level=q_level)

    true_positives = int(np.logical_and(discoveries, true_mask).sum())
    selected = int(discoveries.sum())
    positives = int(true_mask.sum())
    false_positives = selected - true_positives

    tpr = 0.0 if positives == 0 else true_positives / positives
    fdr = 0.0 if selected == 0 else false_positives / selected
    return tpr, fdr


def run_simulation_grid(
    families: Iterable[SignalFamily],
    scenarios: Iterable[SimulationScenario],
    n_grid: list[int],
    replicates: int,
    n_edges: int,
    true_edge_rate: float,
    q_level: float,
    seed: int,
) -> pd.DataFrame:
    """Run all simulation configurations and aggregate summary metrics."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []

    for scenario in scenarios:
        for family in families:
            for n_cells in n_grid:
                replicate_tpr: list[float] = []
                replicate_fdr: list[float] = []

                for _ in range(replicates):
                    tpr, fdr = simulate_replicate(
                        rng=rng,
                        family=family,
                        n_cells=n_cells,
                        n_edges=n_edges,
                        true_edge_rate=true_edge_rate,
                        q_level=q_level,
                        scenario=scenario,
                    )
                    replicate_tpr.append(tpr)
                    replicate_fdr.append(fdr)

                rows.append(
                    {
                        "scenario": scenario.name,
                        "noise_model": scenario.noise_model,
                        "estimator": scenario.estimator,
                        "family": family.name,
                        "n_cells": n_cells,
                        "replicates": replicates,
                        "n_edges": n_edges,
                        "true_edge_rate": true_edge_rate,
                        "q_level": q_level,
                        "tpr_mean": float(np.mean(replicate_tpr)),
                        "tpr_std": float(np.std(replicate_tpr, ddof=1)),
                        "fdr_mean": float(np.mean(replicate_fdr)),
                        "fdr_std": float(np.std(replicate_fdr, ddof=1)),
                    }
                )

    return pd.DataFrame(rows)


def find_detection_boundaries(
    simulation_summary: pd.DataFrame,
    target_tpr: float,
    max_fdr: float,
) -> pd.DataFrame:
    """Compute empirical minimum n that reaches target TPR while controlling FDR."""
    rows: list[dict[str, float | int | str]] = []
    grouped = simulation_summary.groupby(["scenario", "family"], as_index=False)

    for (scenario, family), group in grouped:
        ordered = group.sort_values("n_cells")
        feasible = ordered[(ordered["tpr_mean"] >= target_tpr) & (ordered["fdr_mean"] <= max_fdr)]
        n_boundary = np.nan if feasible.empty else int(feasible["n_cells"].iloc[0])
        rows.append(
            {
                "scenario": scenario,
                "family": family,
                "target_tpr": target_tpr,
                "max_fdr": max_fdr,
                "empirical_n_boundary": n_boundary,
            }
        )

    return pd.DataFrame(rows)


def build_theoretical_table(
    families: Iterable[SignalFamily],
    alpha: float,
    power: float,
    m_tests: int,
) -> pd.DataFrame:
    """Create theoretical sample complexity table under two tail regimes."""
    tail_regimes = {
        "subgaussian": {
            "attention-like": 1.0,
            "causal-intervention-like": 1.0,
        },
        "heavy-tail-stress": {
            "attention-like": 1.2,
            "causal-intervention-like": 1.8,
        },
    }

    rows: list[dict[str, float | int | str]] = []
    for family in families:
        snr = family.effect_size / family.sigma
        for regime_name, inflation_map in tail_regimes.items():
            tail_inflation = inflation_map[family.name]
            n_star = sample_complexity(
                effect_size=family.effect_size,
                sigma=family.sigma,
                alpha=alpha,
                power=power,
                m_tests=m_tests,
                tail_inflation=tail_inflation,
            )
            rows.append(
                {
                    "family": family.name,
                    "regime": regime_name,
                    "effect_size": family.effect_size,
                    "sigma": family.sigma,
                    "snr": snr,
                    "tail_inflation": tail_inflation,
                    "alpha": alpha,
                    "power": power,
                    "m_tests": m_tests,
                    "theoretical_n_star": n_star,
                }
            )

    return pd.DataFrame(rows)


def build_phase_diagram_grid(
    attention_family: SignalFamily,
    alpha: float,
    power: float,
    m_tests: int,
    attention_tail_inflation: float,
    causal_tail_inflation: float,
    effect_ratio_grid: np.ndarray,
    sigma_ratio_grid: np.ndarray,
) -> pd.DataFrame:
    """
    Build a grid comparing n*_causal to n*_attention across effect/noise ratios.

    effect_ratio = mu_causal / mu_attention
    sigma_ratio = sigma_causal / sigma_attention
    """
    n_attention = sample_complexity(
        effect_size=attention_family.effect_size,
        sigma=attention_family.sigma,
        alpha=alpha,
        power=power,
        m_tests=m_tests,
        tail_inflation=attention_tail_inflation,
    )

    rows: list[dict[str, float | int | str]] = []
    for effect_ratio in effect_ratio_grid:
        for sigma_ratio in sigma_ratio_grid:
            causal_effect = attention_family.effect_size * effect_ratio
            causal_sigma = attention_family.sigma * sigma_ratio
            n_causal = sample_complexity(
                effect_size=causal_effect,
                sigma=causal_sigma,
                alpha=alpha,
                power=power,
                m_tests=m_tests,
                tail_inflation=causal_tail_inflation,
            )
            ratio = n_causal / n_attention
            preferred_signal = "causal-intervention-like" if ratio < 1.0 else "attention-like"

            rows.append(
                {
                    "effect_ratio": float(effect_ratio),
                    "sigma_ratio": float(sigma_ratio),
                    "n_attention": int(n_attention),
                    "n_causal": int(n_causal),
                    "ratio_n_causal_to_attention": float(ratio),
                    "log2_ratio": float(np.log2(ratio)),
                    "preferred_signal": preferred_signal,
                    "attention_tail_inflation": attention_tail_inflation,
                    "causal_tail_inflation": causal_tail_inflation,
                }
            )

    return pd.DataFrame(rows)


def plot_phase_diagrams(
    phase_subgaussian: pd.DataFrame,
    phase_heavy_tail: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot sub-Gaussian and heavy-tail detectability phase diagrams side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    plot_items = [
        (axes[0], phase_subgaussian, "Sub-Gaussian regime"),
        (axes[1], phase_heavy_tail, "Heavy-tail stress regime"),
    ]

    for axis, dataframe, title in plot_items:
        pivot = dataframe.pivot(
            index="sigma_ratio",
            columns="effect_ratio",
            values="ratio_n_causal_to_attention",
        ).sort_index()
        pivot = pivot.sort_index(axis=1)

        x_values = pivot.columns.to_numpy(dtype=float)
        y_values = pivot.index.to_numpy(dtype=float)
        z_values = np.log2(pivot.to_numpy(dtype=float))

        image = axis.imshow(
            z_values,
            origin="lower",
            aspect="auto",
            extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()],
            cmap="coolwarm",
            vmin=-2.0,
            vmax=2.0,
        )

        x_mesh, y_mesh = np.meshgrid(x_values, y_values)
        ratio_matrix = pivot.to_numpy(dtype=float)
        axis.contour(
            x_mesh,
            y_mesh,
            ratio_matrix,
            levels=[1.0],
            colors="black",
            linewidths=1.3,
        )

        axis.set_title(title)
        axis.set_xlabel("Causal effect ratio (mu_c / mu_a)")
        axis.set_ylabel("Causal noise ratio (sigma_c / sigma_a)")

    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    colorbar.set_label("log2(n*_causal / n*_attention)")
    fig.suptitle("Detectability phase diagrams: lower is better for causal signals", y=1.02)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_simulation_curves(simulation_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot TPR/FDR curves over sample size for all synthetic scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, constrained_layout=True)

    scenario_style = {
        "subgaussian_mean": "-",
        "heavy_tail_mean": "--",
        "heavy_tail_winsorized": "-.",
    }
    family_color = {
        "attention-like": "#1f77b4",
        "causal-intervention-like": "#d62728",
    }

    for scenario in simulation_summary["scenario"].unique():
        for family in simulation_summary["family"].unique():
            subset = simulation_summary[
                (simulation_summary["scenario"] == scenario)
                & (simulation_summary["family"] == family)
            ].sort_values("n_cells")

            label = f"{family} | {scenario}"
            axes[0].plot(
                subset["n_cells"],
                subset["tpr_mean"],
                linestyle=scenario_style[scenario],
                color=family_color[family],
                linewidth=2.0,
                label=label,
            )
            axes[1].plot(
                subset["n_cells"],
                subset["fdr_mean"],
                linestyle=scenario_style[scenario],
                color=family_color[family],
                linewidth=2.0,
                label=label,
            )

    axes[0].axhline(0.8, color="gray", linewidth=1.0, linestyle=":")
    axes[1].axhline(0.1, color="gray", linewidth=1.0, linestyle=":")
    axes[0].set_title("Empirical TPR vs number of cells")
    axes[1].set_title("Empirical FDR vs number of cells")
    axes[0].set_ylabel("TPR")
    axes[1].set_ylabel("FDR")
    axes[0].set_xlabel("Number of cells")
    axes[1].set_xlabel("Number of cells")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Run Proposal 3 detectability theory + synthetic validation."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
        help="Directory where tables/figures are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for synthetic simulations.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=24,
        help="Replicates per simulation configuration.",
    )
    parser.add_argument(
        "--n-edges",
        type=int,
        default=160,
        help="Number of candidate edges in each synthetic replicate.",
    )
    parser.add_argument(
        "--m-tests",
        type=int,
        default=2000,
        help="Multiple-testing count used by theoretical Bonferroni thresholds.",
    )
    return parser.parse_args()


def main() -> None:
    """Run end-to-end detectability analysis and save artifacts."""
    args = parse_args()

    output_root = args.output_root
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    families = [
        SignalFamily(name="attention-like", effect_size=0.08, sigma=1.00),
        SignalFamily(name="causal-intervention-like", effect_size=0.18, sigma=1.50),
    ]
    attention_family = families[0]

    alpha = 0.05
    power = 0.80
    q_level = 0.10
    target_tpr = 0.80
    max_fdr = 0.10

    theoretical_table = build_theoretical_table(
        families=families,
        alpha=alpha,
        power=power,
        m_tests=args.m_tests,
    )
    theoretical_path = table_dir / "theoretical_sample_complexity.csv"
    theoretical_table.to_csv(theoretical_path, index=False)

    effect_ratio_grid = np.linspace(0.5, 3.0, 101)
    sigma_ratio_grid = np.linspace(0.5, 3.0, 101)

    phase_subgaussian = build_phase_diagram_grid(
        attention_family=attention_family,
        alpha=alpha,
        power=power,
        m_tests=args.m_tests,
        attention_tail_inflation=1.0,
        causal_tail_inflation=1.0,
        effect_ratio_grid=effect_ratio_grid,
        sigma_ratio_grid=sigma_ratio_grid,
    )
    phase_heavy_tail = build_phase_diagram_grid(
        attention_family=attention_family,
        alpha=alpha,
        power=power,
        m_tests=args.m_tests,
        attention_tail_inflation=1.2,
        causal_tail_inflation=1.8,
        effect_ratio_grid=effect_ratio_grid,
        sigma_ratio_grid=sigma_ratio_grid,
    )

    phase_subgaussian.to_csv(table_dir / "phase_diagram_subgaussian.csv", index=False)
    phase_heavy_tail.to_csv(table_dir / "phase_diagram_heavy_tail.csv", index=False)
    plot_phase_diagrams(
        phase_subgaussian=phase_subgaussian,
        phase_heavy_tail=phase_heavy_tail,
        output_path=figure_dir / "phase_diagrams.png",
    )

    scenarios = [
        SimulationScenario(
            name="subgaussian_mean",
            noise_model="gaussian",
            estimator="mean",
            student_df=None,
        ),
        SimulationScenario(
            name="heavy_tail_mean",
            noise_model="student_t",
            estimator="mean",
            student_df=3.5,
        ),
        SimulationScenario(
            name="heavy_tail_winsorized",
            noise_model="student_t",
            estimator="winsorized",
            student_df=3.5,
        ),
    ]

    n_grid = [100, 200, 400, 800, 1200, 1600, 2400, 3200]
    simulation_summary = run_simulation_grid(
        families=families,
        scenarios=scenarios,
        n_grid=n_grid,
        replicates=args.replicates,
        n_edges=args.n_edges,
        true_edge_rate=0.10,
        q_level=q_level,
        seed=args.seed,
    )
    simulation_path = table_dir / "simulation_summary.csv"
    simulation_summary.to_csv(simulation_path, index=False)

    boundaries = find_detection_boundaries(
        simulation_summary=simulation_summary,
        target_tpr=target_tpr,
        max_fdr=max_fdr,
    )
    boundaries_path = table_dir / "empirical_detection_boundaries.csv"
    boundaries.to_csv(boundaries_path, index=False)

    plot_simulation_curves(
        simulation_summary=simulation_summary,
        output_path=figure_dir / "simulation_tpr_fdr_curves.png",
    )

    print("Wrote tables:")
    print(f"- {theoretical_path}")
    print(f"- {table_dir / 'phase_diagram_subgaussian.csv'}")
    print(f"- {table_dir / 'phase_diagram_heavy_tail.csv'}")
    print(f"- {simulation_path}")
    print(f"- {boundaries_path}")
    print("Wrote figures:")
    print(f"- {figure_dir / 'phase_diagrams.png'}")
    print(f"- {figure_dir / 'simulation_tpr_fdr_curves.png'}")
    print("Empirical boundaries:")
    print(boundaries.to_string(index=False))


if __name__ == "__main__":
    main()
