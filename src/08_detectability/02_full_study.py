#!/usr/bin/env python3
"""
Full research-grade execution for Proposal 3:
Detectability phase diagrams for attention-like and intervention-like signals.

The workflow is designed to remain low-compute while expanding rigor:
1) Theory sweep with closed-form sample complexity formulas.
2) Multi-regime phase diagrams and analytic boundary summaries.
3) Adversarial synthetic simulations with estimator/correction stress tests.
4) Bootstrap uncertainty intervals for empirical detectability boundaries.
5) Tiny real-data calibration from existing local artifacts.
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
    """Stylized signal family parameters."""

    name: str
    effect_size: float
    sigma: float


@dataclass(frozen=True)
class TailRegime:
    """Tail inflation assumptions used in theory projections."""

    name: str
    tau_attention: float
    tau_causal: float


@dataclass(frozen=True)
class NoiseScenario:
    """Adversarial simulation setup for synthetic stress testing."""

    name: str
    noise_model: str
    student_df: float | None
    contamination_prob: float
    contamination_scale: float
    heteroskedastic_sigma: float
    correlation_rho: float
    effect_heterogeneity: float


def normal_quantile(probability: float) -> float:
    """Return standard-normal quantile."""
    if not (0.0 < probability < 1.0):
        raise ValueError("probability must be in (0, 1)")
    return statistics.NormalDist().inv_cdf(probability)


def sample_complexity(
    effect_size: float,
    sigma: float,
    alpha: float,
    power: float,
    m_tests: int,
    tail_inflation: float = 1.0,
) -> float:
    """
    Closed-form detectability threshold for two-sided z-style testing.

    n* = ((z_(1-alpha/(2m)) + z_(power)) * (tau * sigma) / |mu|)^2
    """
    if effect_size <= 0.0 or sigma <= 0.0 or tail_inflation <= 0.0:
        raise ValueError("effect_size, sigma, and tail_inflation must be > 0")
    if not (0.0 < alpha < 1.0 and 0.0 < power < 1.0):
        raise ValueError("alpha and power must be in (0, 1)")
    if m_tests < 1:
        raise ValueError("m_tests must be >= 1")

    z_alpha = normal_quantile(1.0 - (alpha / m_tests) / 2.0)
    z_power = normal_quantile(power)
    return ((z_alpha + z_power) * tail_inflation * sigma / effect_size) ** 2


def bh_mask(p_values: np.ndarray, q_level: float) -> np.ndarray:
    """Benjamini-Hochberg selector."""
    if p_values.ndim != 1:
        raise ValueError("p_values must be 1D")
    if not (0.0 < q_level < 1.0):
        raise ValueError("q_level must be in (0, 1)")

    m_tests = p_values.size
    order = np.argsort(p_values)
    ranked = p_values[order]
    threshold = q_level * (np.arange(1, m_tests + 1) / m_tests)
    passed = np.where(ranked <= threshold)[0]
    selected = np.zeros(m_tests, dtype=bool)
    if passed.size == 0:
        return selected

    cutoff = ranked[passed.max()]
    selected[p_values <= cutoff] = True
    return selected


def bonferroni_mask(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Bonferroni selector."""
    cutoff = alpha / p_values.size
    return p_values <= cutoff


def z_to_two_sided_p_values(z_scores: np.ndarray) -> np.ndarray:
    """Convert z-scores to two-sided p-values via normal CDF."""
    normal = statistics.NormalDist()
    abs_z = np.abs(z_scores)
    return np.array([2.0 * (1.0 - normal.cdf(float(v))) for v in abs_z], dtype=float)


def draw_noise(
    rng: np.random.Generator,
    scenario: NoiseScenario,
    n_edges: int,
    n_cells: int,
    sigma: float,
) -> np.ndarray:
    """Draw edge-by-cell noise matrix under adversarial scenario assumptions."""
    if scenario.noise_model == "gaussian":
        base = rng.normal(loc=0.0, scale=sigma, size=(n_edges, n_cells))
    elif scenario.noise_model == "student_t":
        if scenario.student_df is None or scenario.student_df <= 2.0:
            raise ValueError("student_df must be > 2 for student_t noise")
        scaled = sigma * math.sqrt((scenario.student_df - 2.0) / scenario.student_df)
        base = rng.standard_t(df=scenario.student_df, size=(n_edges, n_cells)) * scaled
    else:
        raise ValueError(f"Unsupported noise_model: {scenario.noise_model}")

    if scenario.correlation_rho > 0.0:
        rho = scenario.correlation_rho
        common = rng.normal(loc=0.0, scale=sigma, size=(1, n_cells))
        base = math.sqrt(rho) * common + math.sqrt(1.0 - rho) * base

    if scenario.heteroskedastic_sigma > 0.0:
        multipliers = rng.lognormal(
            mean=-0.5 * scenario.heteroskedastic_sigma**2,
            sigma=scenario.heteroskedastic_sigma,
            size=(n_edges, 1),
        )
        base = base * multipliers

    if scenario.contamination_prob > 0.0:
        outliers = rng.random(size=(n_edges, n_cells)) < scenario.contamination_prob
        contamination = rng.normal(
            loc=0.0,
            scale=scenario.contamination_scale * sigma,
            size=(n_edges, n_cells),
        )
        base = np.where(outliers, contamination, base)

    return base


def estimate_location_and_scale(scores: np.ndarray, estimator: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate location and scale per edge.

    Winsorized estimator uses robust location but raw-score scale to avoid
    anti-conservative standard errors.
    """
    if estimator == "mean":
        location = scores.mean(axis=1)
        scale = scores.std(axis=1, ddof=1)
        return location, scale

    if estimator == "winsorized":
        lower = np.quantile(scores, 0.05, axis=1, keepdims=True)
        upper = np.quantile(scores, 0.95, axis=1, keepdims=True)
        clipped = np.clip(scores, lower, upper)
        location = clipped.mean(axis=1)
        scale = scores.std(axis=1, ddof=1)
        return location, scale

    raise ValueError(f"Unsupported estimator: {estimator}")


def simulate_replicate(
    rng: np.random.Generator,
    family: SignalFamily,
    scenario: NoiseScenario,
    estimator: str,
    correction: str,
    n_cells: int,
    n_edges: int,
    true_edge_rate: float,
    q_level: float,
    alpha_fwer: float,
) -> dict[str, float | int | str]:
    """Run one simulation replicate and return metrics."""
    if not (0.0 < true_edge_rate < 1.0):
        raise ValueError("true_edge_rate must be in (0, 1)")

    true_mask = rng.random(n_edges) < true_edge_rate
    signs = rng.choice([-1.0, 1.0], size=n_edges)

    if scenario.effect_heterogeneity > 0.0:
        multipliers = rng.lognormal(
            mean=-0.5 * scenario.effect_heterogeneity**2,
            sigma=scenario.effect_heterogeneity,
            size=n_edges,
        )
    else:
        multipliers = np.ones(n_edges)

    mu = np.where(true_mask, signs * family.effect_size * multipliers, 0.0)
    noise = draw_noise(rng=rng, scenario=scenario, n_edges=n_edges, n_cells=n_cells, sigma=family.sigma)
    scores = mu[:, None] + noise

    location, scale = estimate_location_and_scale(scores, estimator=estimator)
    standard_error = np.maximum(scale / math.sqrt(n_cells), 1e-12)
    z_scores = location / standard_error
    p_values = z_to_two_sided_p_values(z_scores)

    if correction == "bh":
        selected = bh_mask(p_values, q_level=q_level)
    elif correction == "bonferroni":
        selected = bonferroni_mask(p_values, alpha=alpha_fwer)
    else:
        raise ValueError(f"Unsupported correction: {correction}")

    tp = int(np.logical_and(selected, true_mask).sum())
    selected_count = int(selected.sum())
    positives = int(true_mask.sum())
    fp = selected_count - tp
    negatives = n_edges - positives
    tn = negatives - fp
    fn = positives - tp

    tpr = 0.0 if positives == 0 else tp / positives
    fdr = 0.0 if selected_count == 0 else fp / selected_count
    fpr = 0.0 if negatives == 0 else fp / negatives
    precision = 0.0 if selected_count == 0 else tp / selected_count
    recall = tpr
    f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)

    return {
        "tpr": tpr,
        "fdr": fdr,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def run_simulation_experiments(
    families: Iterable[SignalFamily],
    scenarios: Iterable[NoiseScenario],
    estimators: list[str],
    corrections: list[str],
    n_grid: list[int],
    replicates: int,
    n_edges: int,
    true_edge_rate: float,
    q_level: float,
    alpha_fwer: float,
    seed: int,
) -> pd.DataFrame:
    """Run the full adversarial simulation grid."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []

    for scenario in scenarios:
        for family in families:
            for estimator in estimators:
                for correction in corrections:
                    for n_cells in n_grid:
                        for rep in range(replicates):
                            metrics = simulate_replicate(
                                rng=rng,
                                family=family,
                                scenario=scenario,
                                estimator=estimator,
                                correction=correction,
                                n_cells=n_cells,
                                n_edges=n_edges,
                                true_edge_rate=true_edge_rate,
                                q_level=q_level,
                                alpha_fwer=alpha_fwer,
                            )
                            rows.append(
                                {
                                    "scenario": scenario.name,
                                    "noise_model": scenario.noise_model,
                                    "student_df": scenario.student_df,
                                    "contamination_prob": scenario.contamination_prob,
                                    "contamination_scale": scenario.contamination_scale,
                                    "heteroskedastic_sigma": scenario.heteroskedastic_sigma,
                                    "correlation_rho": scenario.correlation_rho,
                                    "effect_heterogeneity": scenario.effect_heterogeneity,
                                    "family": family.name,
                                    "estimator": estimator,
                                    "correction": correction,
                                    "n_cells": n_cells,
                                    "replicate": rep,
                                    "n_edges": n_edges,
                                    "true_edge_rate": true_edge_rate,
                                    "q_level": q_level,
                                    "alpha_fwer": alpha_fwer,
                                    **metrics,
                                }
                            )
    return pd.DataFrame(rows)


def summarize_simulations(replicates_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate simulation replicates into mean/std summary rows."""
    group_cols = ["scenario", "family", "estimator", "correction", "n_cells"]
    agg = (
        replicates_df.groupby(group_cols, as_index=False)
        .agg(
            tpr_mean=("tpr", "mean"),
            tpr_std=("tpr", "std"),
            fdr_mean=("fdr", "mean"),
            fdr_std=("fdr", "std"),
            fpr_mean=("fpr", "mean"),
            fpr_std=("fpr", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            replicates=("replicate", "count"),
        )
        .sort_values(group_cols)
    )
    return agg


def boundary_from_group(
    group_summary: pd.DataFrame,
    target_tpr: float,
    max_fdr: float,
) -> float:
    """Return minimum n meeting target constraints in a group."""
    ordered = group_summary.sort_values("n_cells")
    feasible = ordered[(ordered["tpr_mean"] >= target_tpr) & (ordered["fdr_mean"] <= max_fdr)]
    if feasible.empty:
        return float("nan")
    return float(feasible["n_cells"].iloc[0])


def estimate_boundaries_with_bootstrap(
    replicates_df: pd.DataFrame,
    target_tpr: float,
    max_fdr: float,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate detectability boundaries with bootstrap confidence intervals."""
    rng = np.random.default_rng(seed)
    group_keys = ["scenario", "family", "estimator", "correction"]
    rows: list[dict[str, float | int | str]] = []

    for key, group in replicates_df.groupby(group_keys):
        scenario, family, estimator, correction = key
        point_summary = summarize_simulations(group)
        point_boundary = boundary_from_group(point_summary, target_tpr=target_tpr, max_fdr=max_fdr)

        n_grid = sorted(group["n_cells"].unique())
        boundary_draws: list[float] = []

        for _ in range(n_bootstrap):
            sampled_rows = []
            for n_cells in n_grid:
                block = group[group["n_cells"] == n_cells]
                sampled_idx = rng.integers(0, block.shape[0], size=block.shape[0])
                sampled = block.iloc[sampled_idx]
                sampled_rows.append(
                    {
                        "n_cells": n_cells,
                        "tpr_mean": float(sampled["tpr"].mean()),
                        "fdr_mean": float(sampled["fdr"].mean()),
                    }
                )
            sampled_summary = pd.DataFrame(sampled_rows)
            boundary_draws.append(boundary_from_group(sampled_summary, target_tpr=target_tpr, max_fdr=max_fdr))

        boundary_array = np.array(boundary_draws, dtype=float)
        defined = boundary_array[np.isfinite(boundary_array)]
        if defined.size == 0:
            ci_low = float("nan")
            ci_high = float("nan")
            median = float("nan")
            defined_prob = 0.0
        else:
            ci_low = float(np.quantile(defined, 0.025))
            median = float(np.quantile(defined, 0.5))
            ci_high = float(np.quantile(defined, 0.975))
            defined_prob = float(defined.size / boundary_array.size)

        rows.append(
            {
                "scenario": scenario,
                "family": family,
                "estimator": estimator,
                "correction": correction,
                "target_tpr": target_tpr,
                "max_fdr": max_fdr,
                "point_boundary_n": point_boundary,
                "bootstrap_boundary_n_median": median,
                "bootstrap_boundary_n_ci_low": ci_low,
                "bootstrap_boundary_n_ci_high": ci_high,
                "bootstrap_boundary_defined_prob": defined_prob,
                "n_bootstrap": n_bootstrap,
            }
        )

    return pd.DataFrame(rows).sort_values(group_keys)


def build_theory_sweep(
    families: Iterable[SignalFamily],
    regimes: Iterable[TailRegime],
    alpha_grid: list[float],
    power_grid: list[float],
    m_grid: list[int],
) -> pd.DataFrame:
    """Build a full sensitivity sweep for closed-form n* values."""
    rows: list[dict[str, float | int | str]] = []
    family_map = {family.name: family for family in families}

    for regime in regimes:
        tau_map = {
            "attention-like": regime.tau_attention,
            "causal-intervention-like": regime.tau_causal,
        }
        for alpha in alpha_grid:
            for power in power_grid:
                for m_tests in m_grid:
                    for family_name, family in family_map.items():
                        n_star = sample_complexity(
                            effect_size=family.effect_size,
                            sigma=family.sigma,
                            alpha=alpha,
                            power=power,
                            m_tests=m_tests,
                            tail_inflation=tau_map[family_name],
                        )
                        rows.append(
                            {
                                "regime": regime.name,
                                "family": family_name,
                                "alpha": alpha,
                                "power": power,
                                "m_tests": m_tests,
                                "effect_size": family.effect_size,
                                "sigma": family.sigma,
                                "tail_inflation": tau_map[family_name],
                                "effective_snr": family.effect_size / (tau_map[family_name] * family.sigma),
                                "theoretical_n_star": float(n_star),
                            }
                        )
    return pd.DataFrame(rows).sort_values(["regime", "alpha", "power", "m_tests", "family"])


def build_phase_grid(
    attention_family: SignalFamily,
    alpha: float,
    power: float,
    m_tests: int,
    regime: TailRegime,
    effect_ratio_grid: np.ndarray,
    sigma_ratio_grid: np.ndarray,
) -> pd.DataFrame:
    """Build phase grid comparing causal and attention sample complexity."""
    n_attention = sample_complexity(
        effect_size=attention_family.effect_size,
        sigma=attention_family.sigma,
        alpha=alpha,
        power=power,
        m_tests=m_tests,
        tail_inflation=regime.tau_attention,
    )

    rows: list[dict[str, float | int | str]] = []
    for effect_ratio in effect_ratio_grid:
        for sigma_ratio in sigma_ratio_grid:
            n_causal = sample_complexity(
                effect_size=attention_family.effect_size * effect_ratio,
                sigma=attention_family.sigma * sigma_ratio,
                alpha=alpha,
                power=power,
                m_tests=m_tests,
                tail_inflation=regime.tau_causal,
            )
            ratio = n_causal / n_attention
            rows.append(
                {
                    "regime": regime.name,
                    "effect_ratio": float(effect_ratio),
                    "sigma_ratio": float(sigma_ratio),
                    "n_attention": float(n_attention),
                    "n_causal": float(n_causal),
                    "ratio_n_causal_to_attention": float(ratio),
                    "log2_ratio": float(np.log2(ratio)),
                    "preferred_signal": "causal-intervention-like" if ratio < 1.0 else "attention-like",
                    "analytic_boundary_sigma_ratio": (regime.tau_attention / regime.tau_causal) * effect_ratio,
                }
            )
    return pd.DataFrame(rows)


def summarize_phase_grid(phase_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize phase occupancy and transition structure by regime."""
    rows = []
    for regime, group in phase_df.groupby("regime"):
        causal_fraction = float((group["ratio_n_causal_to_attention"] < 1.0).mean())
        attention_fraction = 1.0 - causal_fraction
        near_tie_fraction = float((np.abs(np.log2(group["ratio_n_causal_to_attention"])) < 0.1).mean())
        rows.append(
            {
                "regime": regime,
                "causal_preferred_fraction": causal_fraction,
                "attention_preferred_fraction": attention_fraction,
                "near_tie_fraction_abs_log2_lt_0p1": near_tie_fraction,
            }
        )
    return pd.DataFrame(rows).sort_values("regime")


def empirical_tail_inflation(values: np.ndarray) -> float:
    """
    Estimate tail inflation by quantile-width vs Gaussian benchmark.

    For Gaussian data, Q0.99-Q0.01 is approximately 4.6527 * sigma.
    """
    if values.size < 20:
        return float("nan")
    sigma = float(np.std(values, ddof=1))
    if sigma <= 0.0:
        return float("nan")
    width = float(np.quantile(values, 0.99) - np.quantile(values, 0.01))
    return width / (4.6527 * sigma)


def run_real_data_calibration(
    repo_root: Path,
    seed: int,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build tiny real-data calibration from existing artifacts.

    The two signals are not in identical units; this check is intentionally
    framed as a proxy calibration and reported with explicit caveats.
    """
    rng = np.random.default_rng(seed)

    attention_path = repo_root / "network_inference/outputs/head_layer_evidence.tsv"
    causal_pattern = (
        repo_root
        / "subproject_09_causal_mediation_circuit_map/implementation/results"
    )

    attention_df = pd.read_csv(attention_path, sep="\t")
    attention_scores = attention_df["max_head_layer_score"].astype(float).to_numpy()
    attention_scores = attention_scores[attention_scores > 0.0]

    causal_files = sorted(causal_pattern.glob("*/*/causal_scores.tsv"))
    if not causal_files:
        raise FileNotFoundError("No causal_scores.tsv files found for real-data calibration.")
    causal_df = pd.concat([pd.read_csv(path, sep="\t") for path in causal_files], ignore_index=True)
    causal_df = causal_df.drop_duplicates(["source", "target", "intervention", "effect_mean", "effect_std", "n_cells"])
    causal_df = causal_df[causal_df["effect_std"] > 0.0].copy()

    calibration_rows = []

    attention_effect_proxy = float(np.median(attention_scores))
    attention_sigma_proxy = float(np.std(attention_scores, ddof=1))
    attention_standardized = attention_effect_proxy / attention_sigma_proxy
    attention_tail = empirical_tail_inflation(attention_scores)
    calibration_rows.append(
        {
            "family": "attention-like",
            "source_artifact": str(attention_path.relative_to(repo_root)),
            "n_records": int(attention_scores.size),
            "effect_proxy": attention_effect_proxy,
            "sigma_proxy": attention_sigma_proxy,
            "standardized_effect_proxy": attention_standardized,
            "tail_inflation_proxy": attention_tail,
            "notes": "nonzero max_head_layer_score median over full edge evidence",
        }
    )

    causal_effect_values = causal_df["effect_mean"].abs().to_numpy(dtype=float)
    causal_sigma_values = causal_df["effect_std"].to_numpy(dtype=float)
    causal_standardized_values = causal_effect_values / np.maximum(causal_sigma_values, 1e-12)
    causal_effect_proxy = float(np.median(causal_effect_values))
    causal_sigma_proxy = float(np.median(causal_sigma_values))
    causal_standardized = float(np.median(causal_standardized_values))
    causal_tail = empirical_tail_inflation(causal_effect_values)
    calibration_rows.append(
        {
            "family": "causal-intervention-like",
            "source_artifact": "subproject_09_causal_mediation_circuit_map/implementation/results/*/*/causal_scores.tsv",
            "n_records": int(causal_df.shape[0]),
            "effect_proxy": causal_effect_proxy,
            "sigma_proxy": causal_sigma_proxy,
            "standardized_effect_proxy": causal_standardized,
            "tail_inflation_proxy": causal_tail,
            "notes": "median absolute effect_mean and median effect_std for nonzero-std rows",
        }
    )

    calibration_df = pd.DataFrame(calibration_rows)

    # Bootstrap relative detectability factor using standardized proxies only.
    draws = []
    for _ in range(n_bootstrap):
        att_sample = attention_scores[rng.integers(0, attention_scores.size, size=min(5000, attention_scores.size))]
        ca_idx = rng.integers(0, causal_df.shape[0], size=causal_df.shape[0])
        ca_sample = causal_df.iloc[ca_idx]
        att_std = float(np.median(att_sample) / np.maximum(np.std(att_sample, ddof=1), 1e-12))
        ca_std = float(
            np.median(ca_sample["effect_mean"].abs().to_numpy())
            / np.maximum(np.median(ca_sample["effect_std"].to_numpy()), 1e-12)
        )
        ratio = (att_std / ca_std) ** 2
        draws.append(ratio)

    draws_array = np.array(draws, dtype=float)
    projection_summary_df = pd.DataFrame(
        [
            {
                "metric": "relative_cells_causal_vs_attention_from_standardized_proxy",
                "point_estimate": float(np.median(draws_array)),
                "ci_low_95": float(np.quantile(draws_array, 0.025)),
                "ci_high_95": float(np.quantile(draws_array, 0.975)),
                "n_bootstrap": n_bootstrap,
                "interpretation": "values < 1 imply fewer cells needed for causal-like signal",
            }
        ]
    )
    projection_draws_df = pd.DataFrame(
        {"relative_cells_causal_vs_attention": draws_array}
    )

    return calibration_df, projection_summary_df, projection_draws_df


def build_key_results(
    theory_df: pd.DataFrame,
    phase_summary_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    projection_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble compact paper-ready key results table."""
    rows: list[dict[str, float | int | str]] = []

    baseline = theory_df[
        (theory_df["regime"] == "subgaussian")
        & (theory_df["alpha"] == 0.05)
        & (theory_df["power"] == 0.8)
        & (theory_df["m_tests"] == 2000)
    ]
    if not baseline.empty:
        att = float(baseline[baseline["family"] == "attention-like"]["theoretical_n_star"].iloc[0])
        ca = float(
            baseline[baseline["family"] == "causal-intervention-like"]["theoretical_n_star"].iloc[0]
        )
        rows.append(
            {
                "result_id": "theory_baseline_subgaussian",
                "value": ca / att,
                "units": "ratio_n_causal_to_attention",
                "details": "closed-form ratio at alpha=0.05, power=0.8, m=2000",
            }
        )

    for _, row in phase_summary_df.iterrows():
        rows.append(
            {
                "result_id": f"phase_causal_fraction_{row['regime']}",
                "value": row["causal_preferred_fraction"],
                "units": "fraction_grid_points",
                "details": "fraction where causal n* is lower than attention n*",
            }
        )

    boundary_slice = boundary_df[
        (boundary_df["estimator"] == "mean") & (boundary_df["correction"] == "bh")
    ]
    for scenario in sorted(boundary_slice["scenario"].unique()):
        sub = boundary_slice[boundary_slice["scenario"] == scenario]
        if sub.shape[0] != 2:
            continue
        att = sub[sub["family"] == "attention-like"]["point_boundary_n"].iloc[0]
        ca = sub[sub["family"] == "causal-intervention-like"]["point_boundary_n"].iloc[0]
        ratio = float(ca / att) if np.isfinite(att) and np.isfinite(ca) else float("nan")
        rows.append(
            {
                "result_id": f"simulation_boundary_ratio_{scenario}",
                "value": ratio,
                "units": "ratio_n_causal_to_attention",
                "details": "empirical boundary ratio at TPR>=0.8 and FDR<=0.1",
            }
        )

    for _, row in calibration_df.iterrows():
        rows.append(
            {
                "result_id": f"calibration_standardized_effect_{row['family']}",
                "value": row["standardized_effect_proxy"],
                "units": "effect_over_sigma_proxy",
                "details": row["notes"],
            }
        )

    if not projection_summary_df.empty:
        proj = projection_summary_df.iloc[0]
        rows.append(
            {
                "result_id": "real_data_relative_cells_causal_vs_attention",
                "value": proj["point_estimate"],
                "units": "ratio_n_causal_to_attention_proxy",
                "details": (
                    f"bootstrap 95% CI [{proj['ci_low_95']:.3f}, {proj['ci_high_95']:.3f}]"
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_theory_sensitivity(theory_df: pd.DataFrame, output_path: Path) -> None:
    """Plot n* vs m_tests for each regime."""
    regimes = sorted(theory_df["regime"].unique())
    fig, axes = plt.subplots(1, len(regimes), figsize=(5 * len(regimes), 4), constrained_layout=True)
    if len(regimes) == 1:
        axes = [axes]

    family_color = {
        "attention-like": "#1f77b4",
        "causal-intervention-like": "#d62728",
    }

    for axis, regime in zip(axes, regimes):
        subset = theory_df[(theory_df["regime"] == regime) & (theory_df["alpha"] == 0.05) & (theory_df["power"] == 0.8)]
        for family in sorted(subset["family"].unique()):
            fam = subset[subset["family"] == family].sort_values("m_tests")
            axis.plot(
                fam["m_tests"],
                fam["theoretical_n_star"],
                marker="o",
                linewidth=2.0,
                color=family_color[family],
                label=family,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{regime}: n* sensitivity")
        axis.set_xlabel("m (multiple tests)")
        axis.set_ylabel("theoretical n*")
        axis.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_phase_panels(phase_df: pd.DataFrame, output_path: Path) -> None:
    """Plot phase diagrams for each regime."""
    regimes = sorted(phase_df["regime"].unique())
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.5 * len(regimes), 4.8), constrained_layout=True)
    if len(regimes) == 1:
        axes = [axes]

    for axis, regime in zip(axes, regimes):
        subset = phase_df[phase_df["regime"] == regime]
        pivot = subset.pivot(index="sigma_ratio", columns="effect_ratio", values="log2_ratio")
        pivot = pivot.sort_index().sort_index(axis=1)
        x = pivot.columns.to_numpy(dtype=float)
        y = pivot.index.to_numpy(dtype=float)
        z = pivot.to_numpy(dtype=float)
        image = axis.imshow(
            z,
            origin="lower",
            aspect="auto",
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap="coolwarm",
            vmin=-2.0,
            vmax=2.0,
        )
        # Analytic boundary where n*_causal == n*_attention.
        x_line = np.linspace(x.min(), x.max(), 300)
        ratio_values = subset["analytic_boundary_sigma_ratio"] / np.maximum(
            subset["effect_ratio"], 1e-12
        )
        scale = float(np.median(ratio_values))
        y_line = scale * x_line
        axis.plot(x_line, y_line, color="black", linewidth=1.3, linestyle="--", label="analytic tie")
        axis.set_title(regime)
        axis.set_xlabel("effect ratio (mu_c / mu_a)")
        axis.set_ylabel("noise ratio (sigma_c / sigma_a)")
        axis.set_ylim(y.min(), y.max())

    colorbar = fig.colorbar(image, ax=axes, shrink=0.9, pad=0.02)
    colorbar.set_label("log2(n*_causal / n*_attention)")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_boundary_forest(boundary_df: pd.DataFrame, output_path: Path) -> None:
    """Forest-style plot of empirical boundary estimates with confidence intervals."""
    subset = boundary_df[(boundary_df["correction"] == "bh") & (boundary_df["estimator"] == "mean")].copy()
    subset = subset.sort_values(["scenario", "family"])
    if subset.empty:
        return

    labels = [f"{row.scenario} | {row.family}" for row in subset.itertuples()]
    y = np.arange(len(labels))
    point = subset["point_boundary_n"].to_numpy(dtype=float)
    low = subset["bootstrap_boundary_n_ci_low"].to_numpy(dtype=float)
    high = subset["bootstrap_boundary_n_ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([point - low, high - point])

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(labels) + 1.8), constrained_layout=True)
    ax.errorbar(point, y, xerr=xerr, fmt="o", color="#1f77b4", ecolor="#4c78a8", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Boundary n (TPR>=0.8, FDR<=0.1)")
    ax.set_title("Empirical detectability boundaries (BH + mean estimator)")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_real_data_projection(projection_draws: pd.DataFrame, output_path: Path) -> None:
    """Plot distribution of bootstrap relative-cell projection."""
    values = projection_draws["relative_cells_causal_vs_attention"].dropna().astype(float).to_numpy()
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.hist(values, bins=30, color="#2ca02c", alpha=0.85, edgecolor="white")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Projected relative cells (causal / attention)")
    ax.set_ylabel("Bootstrap count")
    ax.set_title("Tiny real-data projection from standardized proxy effects")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Run full Proposal 3 research-grade study.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs/full_study",
        help="Output root for full-study tables and figures.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    parser.add_argument("--replicates", type=int, default=18, help="Simulation replicates per grid point.")
    parser.add_argument("--n-edges", type=int, default=120, help="Edges per simulation replicate.")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=300,
        help="Bootstrap samples for boundary intervals and real-data projection.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the end-to-end full study and save artifacts."""
    args = parse_args()
    output_root = args.output_root
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]

    families = [
        SignalFamily(name="attention-like", effect_size=0.08, sigma=1.00),
        SignalFamily(name="causal-intervention-like", effect_size=0.18, sigma=1.50),
    ]
    regimes = [
        TailRegime(name="subgaussian", tau_attention=1.0, tau_causal=1.0),
        TailRegime(name="heavy_tail_mild", tau_attention=1.1, tau_causal=1.4),
        TailRegime(name="heavy_tail_severe", tau_attention=1.2, tau_causal=1.8),
    ]
    scenarios = [
        NoiseScenario(
            name="gaussian_clean",
            noise_model="gaussian",
            student_df=None,
            contamination_prob=0.0,
            contamination_scale=1.0,
            heteroskedastic_sigma=0.0,
            correlation_rho=0.0,
            effect_heterogeneity=0.20,
        ),
        NoiseScenario(
            name="student_t_df4",
            noise_model="student_t",
            student_df=4.0,
            contamination_prob=0.0,
            contamination_scale=1.0,
            heteroskedastic_sigma=0.0,
            correlation_rho=0.0,
            effect_heterogeneity=0.20,
        ),
        NoiseScenario(
            name="gaussian_contaminated",
            noise_model="gaussian",
            student_df=None,
            contamination_prob=0.06,
            contamination_scale=6.0,
            heteroskedastic_sigma=0.0,
            correlation_rho=0.0,
            effect_heterogeneity=0.20,
        ),
        NoiseScenario(
            name="gaussian_heteroskedastic",
            noise_model="gaussian",
            student_df=None,
            contamination_prob=0.0,
            contamination_scale=1.0,
            heteroskedastic_sigma=0.45,
            correlation_rho=0.0,
            effect_heterogeneity=0.20,
        ),
        NoiseScenario(
            name="gaussian_correlated",
            noise_model="gaussian",
            student_df=None,
            contamination_prob=0.0,
            contamination_scale=1.0,
            heteroskedastic_sigma=0.0,
            correlation_rho=0.35,
            effect_heterogeneity=0.20,
        ),
    ]

    alpha_grid = [0.05, 0.01]
    power_grid = [0.8, 0.9]
    m_grid = [500, 2000, 10000]
    theory_df = build_theory_sweep(families=families, regimes=regimes, alpha_grid=alpha_grid, power_grid=power_grid, m_grid=m_grid)
    theory_df.to_csv(table_dir / "theory_sensitivity_sweep.csv", index=False)

    effect_ratio_grid = np.linspace(0.4, 3.2, 121)
    sigma_ratio_grid = np.linspace(0.4, 3.2, 121)
    phase_frames = []
    for regime in regimes:
        phase_frames.append(
            build_phase_grid(
                attention_family=families[0],
                alpha=0.05,
                power=0.8,
                m_tests=2000,
                regime=regime,
                effect_ratio_grid=effect_ratio_grid,
                sigma_ratio_grid=sigma_ratio_grid,
            )
        )
    phase_df = pd.concat(phase_frames, ignore_index=True)
    phase_df.to_csv(table_dir / "phase_diagram_full_grid.csv", index=False)
    phase_summary_df = summarize_phase_grid(phase_df)
    phase_summary_df.to_csv(table_dir / "phase_diagram_regime_summary.csv", index=False)

    n_grid = [100, 200, 400, 800, 1200, 1600, 2400, 3200]
    estimators = ["mean", "winsorized"]
    corrections = ["bh", "bonferroni"]
    replicates_df = run_simulation_experiments(
        families=families,
        scenarios=scenarios,
        estimators=estimators,
        corrections=corrections,
        n_grid=n_grid,
        replicates=args.replicates,
        n_edges=args.n_edges,
        true_edge_rate=0.10,
        q_level=0.10,
        alpha_fwer=0.05,
        seed=args.seed,
    )
    replicates_df.to_csv(table_dir / "simulation_replicates.csv", index=False)

    simulation_summary_df = summarize_simulations(replicates_df)
    simulation_summary_df.to_csv(table_dir / "simulation_summary.csv", index=False)

    boundaries_df = estimate_boundaries_with_bootstrap(
        replicates_df=replicates_df,
        target_tpr=0.80,
        max_fdr=0.10,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 101,
    )
    boundaries_df.to_csv(table_dir / "simulation_detectability_boundaries.csv", index=False)

    calibration_df, projection_summary_df, projection_draws_df = run_real_data_calibration(
        repo_root=repo_root,
        seed=args.seed + 202,
        n_bootstrap=args.n_bootstrap,
    )
    calibration_df.to_csv(table_dir / "real_data_calibration.csv", index=False)
    projection_summary_df.to_csv(table_dir / "real_data_projection_summary.csv", index=False)
    projection_draws_df.to_csv(table_dir / "real_data_projection_draws.csv", index=False)

    key_results_df = build_key_results(
        theory_df=theory_df,
        phase_summary_df=phase_summary_df,
        boundary_df=boundaries_df,
        calibration_df=calibration_df,
        projection_summary_df=projection_summary_df,
    )
    key_results_df.to_csv(table_dir / "key_results.csv", index=False)

    plot_theory_sensitivity(theory_df=theory_df, output_path=figure_dir / "fig_theory_sensitivity.png")
    plot_phase_panels(phase_df=phase_df, output_path=figure_dir / "fig_phase_diagram_regimes.png")
    plot_boundary_forest(boundary_df=boundaries_df, output_path=figure_dir / "fig_simulation_boundaries.png")
    plot_real_data_projection(
        projection_draws=projection_draws_df,
        output_path=figure_dir / "fig_real_data_projection.png",
    )

    print("Full-study outputs written to:")
    print(f"- tables: {table_dir}")
    print(f"- figures: {figure_dir}")
    print("Key empirical boundaries (BH + mean):")
    print(
        boundaries_df[
            (boundaries_df["estimator"] == "mean")
            & (boundaries_df["correction"] == "bh")
        ][["scenario", "family", "point_boundary_n", "bootstrap_boundary_n_ci_low", "bootstrap_boundary_n_ci_high"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
