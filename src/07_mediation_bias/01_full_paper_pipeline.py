#!/usr/bin/env python3
"""Full paper pipeline for Proposal 5: bias bounds in patching-based mediation.

This script upgrades the initial proposal run into a research-grade artifact pipeline:
- robust real-data diagnostics on existing mediation checkpoints,
- adversarial ranking-certification sensitivity analysis,
- cross-tissue alignment analysis with uncertainty propagation,
- synthetic estimator benchmarks under interacting mediators,
- finalized manuscript draft with traceable artifact references.

The implementation is intentionally low-compute and reproducible from existing outputs.
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
from pandas.errors import EmptyDataError


Z_95 = 1.96
EPS = 1e-12
RNG_SEED = 37


@dataclass(frozen=True)
class RunSpec:
    """Metadata for one mediation run directory."""

    run_id: str
    tissue: str
    granularity: str
    base_dir: Path
    checkpoint_path: Path
    causal_scores_path: Path
    circuit_maps_path: Path


def find_repo_root(start: Path) -> Path:
    """Infer repository root from known top-level directories."""
    for candidate in [start] + list(start.parents):
        if (candidate / "market_research").exists() and (candidate / "subproject_09_causal_mediation_circuit_map").exists():
            return candidate
    raise RuntimeError("Could not infer repository root from script location.")


def discover_runs(repo_root: Path) -> List[RunSpec]:
    """Discover all run directories with checkpoint + causal_scores + circuit_maps."""
    base = repo_root / "subproject_09_causal_mediation_circuit_map" / "implementation" / "results"
    if not base.exists():
        raise FileNotFoundError(f"Missing results directory: {base}")

    runs: List[RunSpec] = []
    for tissue_dir in sorted(base.iterdir()):
        if not tissue_dir.is_dir() or tissue_dir.name == "preprocessed":
            continue
        tissue = tissue_dir.name

        for gran_dir in sorted(tissue_dir.iterdir()):
            if not gran_dir.is_dir():
                continue
            granularity = gran_dir.name
            if granularity not in {"heads", "mlp"}:
                # Restrict to mediator granularities in scope for this paper.
                continue

            checkpoint_path = gran_dir / "checkpoint.json"
            causal_scores_path = gran_dir / "causal_scores.tsv"
            circuit_maps_path = gran_dir / "circuit_maps.tsv"

            if not (checkpoint_path.exists() and causal_scores_path.exists() and circuit_maps_path.exists()):
                continue

            run_id = f"{tissue}:{granularity}"
            runs.append(
                RunSpec(
                    run_id=run_id,
                    tissue=tissue,
                    granularity=granularity,
                    base_dir=gran_dir,
                    checkpoint_path=checkpoint_path,
                    causal_scores_path=causal_scores_path,
                    circuit_maps_path=circuit_maps_path,
                )
            )

    if not runs:
        raise RuntimeError("No mediation runs discovered for analysis.")
    return runs


def build_component_id(component: str, layer: int, head: float) -> str:
    """Create canonical component identifiers shared across run types."""
    component = str(component)
    if component == "head":
        return f"L{int(layer)}:H{int(head)}"
    if component == "mlp":
        return f"L{int(layer)}:MLP"
    return f"L{int(layer)}:{component}"


def _safe_se(std: float, n: int) -> float:
    """Compute standard error safely for finite positive sample size."""
    if n <= 0 or not np.isfinite(std):
        return float("nan")
    return float(std) / math.sqrt(float(n))


def load_real_data(repo_root: Path, runs: Sequence[RunSpec]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pair- and component-level data from causal score/map TSV files.

    Returns:
    - manifest_df: run-level metadata and counts
    - pair_df: pair-level total effect rows (ablation intervention)
    - component_df: single-component restoration rows
    """
    manifest_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []
    component_rows: List[Dict[str, object]] = []

    for run in runs:
        # Parse checkpoint only for coverage diagnostics (which pairs were processed).
        checkpoint_payload = json.loads(run.checkpoint_path.read_text(encoding="utf-8"))
        processed_pairs = sorted(checkpoint_payload.get("processed_pairs", []))

        try:
            scores = pd.read_csv(run.causal_scores_path, sep="\t")
            circuits = pd.read_csv(run.circuit_maps_path, sep="\t")
        except EmptyDataError:
            # Skip incomplete runs that may be mid-write by another process.
            continue

        if scores.empty or circuits.empty:
            continue

        ablation_scores = scores[scores["intervention"].astype(str) == "ablation"].copy()
        ablation_scores["pair"] = ablation_scores["source"].astype(str) + "->" + ablation_scores["target"].astype(str)

        for row in ablation_scores.itertuples(index=False):
            n_cells = int(row.n_cells)
            effect_std = float(row.effect_std)
            pair_rows.append(
                {
                    "run_id": run.run_id,
                    "tissue": run.tissue,
                    "granularity": run.granularity,
                    "pair": str(row.pair),
                    "source": str(row.source),
                    "target": str(row.target),
                    "te_mean": float(row.effect_mean),
                    "te_std": effect_std,
                    "te_n_cells": n_cells,
                    "te_se": _safe_se(effect_std, n_cells),
                    "causal_scores_path": str(run.causal_scores_path.relative_to(repo_root)),
                    "checkpoint_path": str(run.checkpoint_path.relative_to(repo_root)),
                }
            )

        circuits = circuits.copy()
        circuits["pair"] = circuits["source"].astype(str) + "->" + circuits["target"].astype(str)
        circuits["component_id"] = circuits.apply(
            lambda r: build_component_id(r["component"], int(r["layer"]), float(r["head"]) if not pd.isna(r["head"]) else -1.0),
            axis=1,
        )

        for row in circuits.itertuples(index=False):
            n_cells = int(row.n_cells)
            std = float(row.restoration_std)
            component_rows.append(
                {
                    "run_id": run.run_id,
                    "tissue": run.tissue,
                    "granularity": run.granularity,
                    "pair": str(row.pair),
                    "source": str(row.source),
                    "target": str(row.target),
                    "component_id": str(row.component_id),
                    "component": str(row.component),
                    "layer": int(row.layer),
                    "head": (int(row.head) if not pd.isna(row.head) else -1),
                    "effect_mean": float(row.restoration_mean),
                    "effect_abs": abs(float(row.restoration_mean)),
                    "effect_std": std,
                    "effect_n_cells": n_cells,
                    "effect_se": _safe_se(std, n_cells),
                    "effect_ci95": Z_95 * _safe_se(std, n_cells),
                    "circuit_maps_path": str(run.circuit_maps_path.relative_to(repo_root)),
                    "checkpoint_path": str(run.checkpoint_path.relative_to(repo_root)),
                }
            )

        manifest_rows.append(
            {
                "run_id": run.run_id,
                "tissue": run.tissue,
                "granularity": run.granularity,
                "processed_pairs_in_checkpoint": len(processed_pairs),
                "ablation_pairs_in_scores": int(ablation_scores.shape[0]),
                "component_rows_in_maps": int(circuits.shape[0]),
                "checkpoint_path": str(run.checkpoint_path.relative_to(repo_root)),
                "causal_scores_path": str(run.causal_scores_path.relative_to(repo_root)),
                "circuit_maps_path": str(run.circuit_maps_path.relative_to(repo_root)),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["tissue", "granularity"]).reset_index(drop=True)
    pair_df = pd.DataFrame(pair_rows).sort_values(["run_id", "pair"]).reset_index(drop=True)
    component_df = pd.DataFrame(component_rows).sort_values(["run_id", "pair", "component_id"]).reset_index(drop=True)

    if pair_df.empty:
        raise RuntimeError("No ablation pair rows found in discovered runs.")
    if component_df.empty:
        raise RuntimeError("No component rows found in discovered runs.")

    return manifest_df, pair_df, component_df


def _shannon_entropy(weights: np.ndarray) -> float:
    """Shannon entropy for nonnegative weights (natural log)."""
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan")
    probs = np.clip(weights / total, EPS, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def compute_pair_diagnostics(pair_df: pd.DataFrame, component_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pair-level diagnostics including residual non-additivity bounds."""
    pair_lookup = pair_df.set_index(["run_id", "pair"])  # ensures fast joins.

    rows: List[Dict[str, object]] = []
    for (run_id, pair), group in component_df.groupby(["run_id", "pair"], sort=True):
        if (run_id, pair) not in pair_lookup.index:
            continue

        pair_row = pair_lookup.loc[(run_id, pair)]
        if isinstance(pair_row, pd.DataFrame):
            pair_row = pair_row.iloc[0]

        te_mean = float(pair_row["te_mean"])
        te_se = float(pair_row["te_se"])

        # Summation decomposition: TE vs aggregate single-mediator effects.
        sum_single = float(group["effect_mean"].sum())
        sum_abs_single = float(group["effect_abs"].sum())

        se_sum_single = math.sqrt(float(np.nansum(np.square(group["effect_se"].to_numpy(dtype=float)))))
        residual = te_mean - sum_single
        residual_se = math.sqrt((te_se ** 2 if np.isfinite(te_se) else 0.0) + se_sum_single ** 2)
        residual_ci95 = Z_95 * residual_se

        # Observable lower bound on aggregate non-additivity magnitude.
        nonadd_lb = max(0.0, abs(residual) - residual_ci95)
        nonadd_proxy = abs(residual)

        ranked = group.sort_values("effect_abs", ascending=False).reset_index(drop=True)
        top1 = ranked.iloc[0]
        top2 = ranked.iloc[1] if ranked.shape[0] >= 2 else ranked.iloc[0]
        top_gap = float(top1["effect_abs"] - top2["effect_abs"])

        entropy = _shannon_entropy(ranked["effect_abs"].to_numpy(dtype=float))
        eff_components = math.exp(entropy) if np.isfinite(entropy) else float("nan")

        denom_abs = max(sum_abs_single, EPS)
        top5_mass = float(ranked.head(5)["effect_abs"].sum()) / denom_abs

        snr = ranked["effect_abs"].to_numpy(dtype=float) / np.maximum(ranked["effect_se"].to_numpy(dtype=float), EPS)

        rows.append(
            {
                "run_id": run_id,
                "tissue": str(pair_row["tissue"]),
                "granularity": str(pair_row["granularity"]),
                "pair": pair,
                "source": str(pair_row["source"]),
                "target": str(pair_row["target"]),
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
                "nonadditivity_proxy": nonadd_proxy,
                "nonadditivity_lb_over_te": nonadd_lb / max(abs(te_mean), EPS),
                "top1_component": str(top1["component_id"]),
                "top2_component": str(top2["component_id"]),
                "top1_abs_effect": float(top1["effect_abs"]),
                "top2_abs_effect": float(top2["effect_abs"]),
                "top1_top2_gap": top_gap,
                "top5_mass": top5_mass,
                "entropy_abs_effect": entropy,
                "effective_components": eff_components,
                "median_snr": float(np.nanmedian(snr)),
                "min_snr": float(np.nanmin(snr)),
                "max_snr": float(np.nanmax(snr)),
            }
        )

    out = pd.DataFrame(rows).sort_values(["run_id", "pair"]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("Pair diagnostics are empty.")
    return out


def _pairwise_certification_stats(
    ranked_abs: np.ndarray,
    ci95: np.ndarray,
    structural_budget: np.ndarray,
) -> Tuple[float, bool]:
    """Compute certified pair fraction and top1 certification for one budget model."""
    n = ranked_abs.shape[0]
    if n < 2:
        return float("nan"), False

    total = 0
    certified = 0
    top1_certified = False

    for i, j in itertools.combinations(range(n), 2):
        gap = float(ranked_abs[i] - ranked_abs[j])
        margin = gap - (float(ci95[i] + ci95[j]) + float(structural_budget[i] + structural_budget[j]))
        is_cert = margin > 0.0
        total += 1
        certified += int(is_cert)

        if i == 0 and j == 1:
            top1_certified = is_cert

    return certified / float(total), top1_certified


def ranking_certification_sensitivity(
    component_df: pd.DataFrame,
    pair_diag_df: pd.DataFrame,
    lambda_grid: Sequence[float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sensitivity of ranking certification to assumed structural-bias multipliers.

    Since nonadditivity_lb is a lower bound (not an upper bound), we report sensitivity
    across multipliers lambda >= 1 to stress-test ranking claims.
    """
    diag_lookup = pair_diag_df.set_index(["run_id", "pair"])

    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    flip_rows: List[Dict[str, object]] = []

    for (run_id, pair), group in component_df.groupby(["run_id", "pair"], sort=True):
        if (run_id, pair) not in diag_lookup.index:
            continue

        drow = diag_lookup.loc[(run_id, pair)]
        if isinstance(drow, pd.DataFrame):
            drow = drow.iloc[0]

        ranked = group.sort_values("effect_abs", ascending=False).reset_index(drop=True)
        abs_vals = ranked["effect_abs"].to_numpy(dtype=float)
        ci95_vals = ranked["effect_ci95"].fillna(0.0).to_numpy(dtype=float)

        k = ranked.shape[0]
        sum_abs = float(np.sum(abs_vals))
        if sum_abs <= 0:
            sum_abs = 1.0

        base_lb = float(drow["nonadditivity_lb"])
        top_gap = float(drow["top1_top2_gap"])
        sample_top_budget = float(ci95_vals[0] + ci95_vals[1]) if k >= 2 else float("nan")
        required_structural_for_flip = max(0.0, top_gap - sample_top_budget)
        lambda_flip = float("inf") if base_lb <= 0 else required_structural_for_flip / max(base_lb, EPS)

        flip_rows.append(
            {
                "run_id": run_id,
                "pair": pair,
                "granularity": str(drow["granularity"]),
                "top1_component": str(drow["top1_component"]),
                "top2_component": str(drow["top2_component"]),
                "top1_top2_gap": top_gap,
                "sample_top_budget": sample_top_budget,
                "required_structural_for_flip": required_structural_for_flip,
                "nonadditivity_lb": base_lb,
                "lambda_flip_threshold": lambda_flip,
            }
        )

        for lam in lambda_grid:
            assumed_budget = lam * base_lb

            uniform_budget = np.full(k, assumed_budget / float(k), dtype=float)
            proportional_budget = assumed_budget * (abs_vals / sum_abs)

            frac_uniform, top1_uniform = _pairwise_certification_stats(abs_vals, ci95_vals, uniform_budget)
            frac_prop, top1_prop = _pairwise_certification_stats(abs_vals, ci95_vals, proportional_budget)

            # Concentrated adversarial model: a challenger pair can absorb all structural budget.
            total = 0
            cert = 0
            top1_concentrated = False
            for i, j in itertools.combinations(range(k), 2):
                gap = float(abs_vals[i] - abs_vals[j])
                margin = gap - (float(ci95_vals[i] + ci95_vals[j]) + assumed_budget)
                is_cert = margin > 0.0
                total += 1
                cert += int(is_cert)
                if i == 0 and j == 1:
                    top1_concentrated = is_cert
            frac_concentrated = cert / float(total)

            summary_rows.append(
                {
                    "run_id": run_id,
                    "pair": pair,
                    "granularity": str(drow["granularity"]),
                    "lambda_multiplier": float(lam),
                    "assumed_structural_budget": assumed_budget,
                    "certified_fraction_uniform": frac_uniform,
                    "certified_fraction_proportional": frac_prop,
                    "certified_fraction_concentrated": frac_concentrated,
                    "top1_certified_uniform": bool(top1_uniform),
                    "top1_certified_proportional": bool(top1_prop),
                    "top1_certified_concentrated": bool(top1_concentrated),
                }
            )

            # Per-comparison detail for targeted auditing of unstable pairs.
            for i, j in itertools.combinations(range(k), 2):
                gap = float(abs_vals[i] - abs_vals[j])
                sample_budget = float(ci95_vals[i] + ci95_vals[j])

                margin_uniform = gap - (sample_budget + float(uniform_budget[i] + uniform_budget[j]))
                margin_prop = gap - (sample_budget + float(proportional_budget[i] + proportional_budget[j]))
                margin_conc = gap - (sample_budget + assumed_budget)

                detail_rows.append(
                    {
                        "run_id": run_id,
                        "pair": pair,
                        "granularity": str(drow["granularity"]),
                        "lambda_multiplier": float(lam),
                        "component_hi": str(ranked.iloc[i]["component_id"]),
                        "component_lo": str(ranked.iloc[j]["component_id"]),
                        "abs_gap": gap,
                        "sample_budget": sample_budget,
                        "margin_uniform": margin_uniform,
                        "margin_proportional": margin_prop,
                        "margin_concentrated": margin_conc,
                        "certified_uniform": bool(margin_uniform > 0.0),
                        "certified_proportional": bool(margin_prop > 0.0),
                        "certified_concentrated": bool(margin_conc > 0.0),
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(["run_id", "pair", "lambda_multiplier"]).reset_index(drop=True)
    detail_df = pd.DataFrame(detail_rows).sort_values(["run_id", "pair", "lambda_multiplier", "abs_gap"], ascending=[True, True, True, False]).reset_index(drop=True)
    flip_df = pd.DataFrame(flip_rows).sort_values(["run_id", "pair"]).reset_index(drop=True)

    return summary_df, detail_df, flip_df


def _safe_corr(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float:
    """Safe correlation wrapper returning NaN for degenerate vectors."""
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.allclose(np.std(a), 0.0) or np.allclose(np.std(b), 0.0):
        return float("nan")
    sa = pd.Series(a)
    sb = pd.Series(b)
    return float(sa.corr(sb, method=method))


def _topk_ids(values: np.ndarray, labels: Sequence[str], k: int) -> set[str]:
    """Return top-k labels by absolute value."""
    if len(labels) == 0:
        return set()
    k = min(k, len(labels))
    order = np.argsort(-np.abs(values))[:k]
    return {labels[idx] for idx in order}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two finite sets."""
    if not a and not b:
        return float("nan")
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cross_tissue_alignment(
    component_df: pd.DataFrame,
    bootstrap_draws: int,
    seed: int,
) -> pd.DataFrame:
    """Cross-tissue component alignment for shared pairs within each granularity."""
    rng = np.random.default_rng(seed)

    rows: List[Dict[str, object]] = []

    # Restrict to comparisons across tissues for identical pair + granularity.
    for granularity, g_gran in component_df.groupby("granularity", sort=True):
        pair_to_tissues: Dict[str, set[str]] = {}
        for row in g_gran[["pair", "tissue"]].drop_duplicates().itertuples(index=False):
            pair_to_tissues.setdefault(str(row.pair), set()).add(str(row.tissue))

        shared_pairs = sorted([pair for pair, tissues in pair_to_tissues.items() if len(tissues) >= 2])
        for pair in shared_pairs:
            pair_data = g_gran[g_gran["pair"] == pair]
            tissues = sorted(pair_data["tissue"].unique().tolist())

            for ta, tb in itertools.combinations(tissues, 2):
                da = pair_data[pair_data["tissue"] == ta].copy()
                db = pair_data[pair_data["tissue"] == tb].copy()

                merged = da.merge(
                    db,
                    on="component_id",
                    suffixes=("_a", "_b"),
                    how="inner",
                )
                if merged.empty:
                    continue

                a = merged["effect_mean_a"].to_numpy(dtype=float)
                b = merged["effect_mean_b"].to_numpy(dtype=float)
                se_a = merged["effect_se_a"].fillna(0.0).to_numpy(dtype=float)
                se_b = merged["effect_se_b"].fillna(0.0).to_numpy(dtype=float)
                labels = merged["component_id"].astype(str).tolist()

                pearson = _safe_corr(a, b, method="pearson")
                spearman = _safe_corr(a, b, method="spearman")
                sign_agreement = float(np.mean(np.sign(a) == np.sign(b)))
                j_top1 = _jaccard(_topk_ids(a, labels, 1), _topk_ids(b, labels, 1))
                j_top3 = _jaccard(_topk_ids(a, labels, 3), _topk_ids(b, labels, 3))
                j_top5 = _jaccard(_topk_ids(a, labels, 5), _topk_ids(b, labels, 5))

                # Bootstrap uncertainty propagation from summary SE values.
                boots_pearson = []
                boots_spearman = []
                boots_j_top1 = []
                boots_j_top5 = []
                for _ in range(bootstrap_draws):
                    draw_a = rng.normal(loc=a, scale=np.maximum(se_a, EPS))
                    draw_b = rng.normal(loc=b, scale=np.maximum(se_b, EPS))
                    boots_pearson.append(_safe_corr(draw_a, draw_b, method="pearson"))
                    boots_spearman.append(_safe_corr(draw_a, draw_b, method="spearman"))
                    boots_j_top1.append(_jaccard(_topk_ids(draw_a, labels, 1), _topk_ids(draw_b, labels, 1)))
                    boots_j_top5.append(_jaccard(_topk_ids(draw_a, labels, 5), _topk_ids(draw_b, labels, 5)))

                boots_pearson_arr = np.array(boots_pearson, dtype=float)
                boots_spearman_arr = np.array(boots_spearman, dtype=float)
                boots_j_top1_arr = np.array(boots_j_top1, dtype=float)
                boots_j_top5_arr = np.array(boots_j_top5, dtype=float)

                rows.append(
                    {
                        "granularity": granularity,
                        "pair": pair,
                        "tissue_a": ta,
                        "tissue_b": tb,
                        "components_compared": int(merged.shape[0]),
                        "pearson": pearson,
                        "pearson_ci_low": float(np.nanquantile(boots_pearson_arr, 0.025)),
                        "pearson_ci_high": float(np.nanquantile(boots_pearson_arr, 0.975)),
                        "spearman": spearman,
                        "spearman_ci_low": float(np.nanquantile(boots_spearman_arr, 0.025)),
                        "spearman_ci_high": float(np.nanquantile(boots_spearman_arr, 0.975)),
                        "sign_agreement": sign_agreement,
                        "top1_jaccard": j_top1,
                        "top1_jaccard_ci_low": float(np.nanquantile(boots_j_top1_arr, 0.025)),
                        "top1_jaccard_ci_high": float(np.nanquantile(boots_j_top1_arr, 0.975)),
                        "top3_jaccard": j_top3,
                        "top5_jaccard": j_top5,
                        "top5_jaccard_ci_low": float(np.nanquantile(boots_j_top5_arr, 0.025)),
                        "top5_jaccard_ci_high": float(np.nanquantile(boots_j_top5_arr, 0.975)),
                    }
                )

    out = pd.DataFrame(rows).sort_values(["granularity", "pair", "tissue_a", "tissue_b"]).reset_index(drop=True)
    return out


def _spearman_abs(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation between absolute vectors."""
    return _safe_corr(np.abs(x), np.abs(y), method="spearman")


def _pairwise_inversion_rate(est: np.ndarray, truth: np.ndarray) -> float:
    """Pairwise ordering inversion rate using absolute rankings."""
    n = est.size
    if n < 2:
        return float("nan")

    total = 0
    inv = 0
    ae = np.abs(est)
    at = np.abs(truth)

    for i, j in itertools.combinations(range(n), 2):
        de = float(ae[i] - ae[j])
        dt = float(at[i] - at[j])
        if abs(de) < EPS or abs(dt) < EPS:
            continue
        total += 1
        if de * dt < 0:
            inv += 1

    if total == 0:
        return float("nan")
    return inv / float(total)


def _sample_game_coefficients(
    rng: np.random.Generator,
    k: int,
    pair_scale: float,
    triple_scale: float,
    triple_density: float,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, float]]]:
    """Sample additive + interaction coefficients for synthetic cooperative game."""
    alpha = rng.normal(loc=0.0, scale=1.0, size=k)

    beta = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            val = rng.normal(loc=0.0, scale=pair_scale / math.sqrt(float(k)))
            beta[i, j] = val
            beta[j, i] = val

    triples: List[Tuple[int, int, int, float]] = []
    for i in range(k):
        for j in range(i + 1, k):
            for l in range(j + 1, k):
                if rng.random() <= triple_density:
                    val = rng.normal(loc=0.0, scale=triple_scale / float(k))
                    triples.append((i, j, l, float(val)))

    return alpha, beta, triples


def _compute_true_shapley(alpha: np.ndarray, beta: np.ndarray, triples: Sequence[Tuple[int, int, int, float]]) -> np.ndarray:
    """Closed-form Shapley value for additive + pairwise + triple game terms."""
    k = alpha.size
    phi = alpha.astype(float).copy()
    phi += 0.5 * beta.sum(axis=1)
    for i, j, l, val in triples:
        phi[i] += val / 3.0
        phi[j] += val / 3.0
        phi[l] += val / 3.0
    return phi


def _simulate_single_and_pairwise_estimates(
    rng: np.random.Generator,
    alpha: np.ndarray,
    beta: np.ndarray,
    n_cells: int,
    noise_sigma: float,
    pair_noise_multiplier: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate noisy single and raw pairwise-corrected estimators."""
    k = alpha.size
    single_true = alpha
    single_noise_sd = noise_sigma / math.sqrt(float(n_cells))
    pair_noise_sd = pair_noise_multiplier * noise_sigma / math.sqrt(float(n_cells))

    single_hat = single_true + rng.normal(0.0, single_noise_sd, size=k)

    pair_hat = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            true_pair = alpha[i] + alpha[j] + beta[i, j]
            noisy_pair = true_pair + rng.normal(0.0, pair_noise_sd)
            pair_hat[i, j] = noisy_pair
            pair_hat[j, i] = noisy_pair

    # Pairwise corrected estimator using all measured pair patches.
    corrected = single_hat.copy()
    for i in range(k):
        delta_sum = 0.0
        for j in range(k):
            if i == j:
                continue
            delta_ij = pair_hat[i, j] - single_hat[i] - single_hat[j]
            delta_sum += delta_ij
        corrected[i] = single_hat[i] + 0.5 * delta_sum

    return single_hat, corrected


def _shrink_pairwise_correction(
    single_hat: np.ndarray,
    paircorr_hat: np.ndarray,
    n_cells: int,
    k: int,
    noise_sigma: float,
    pair_noise_multiplier: float,
) -> Tuple[np.ndarray, float]:
    """Shrink raw pairwise correction toward single estimates by estimated noise load.

    Raw pairwise correction is unbiased for pairwise interactions in the infinite-sample
    limit, but finite-sample noise can dominate because each component aggregates many
    noisy pair contrasts. This shrinkage estimator stabilizes correction by estimating
    the noise fraction in the correction term and down-weighting accordingly.
    """
    correction = paircorr_hat - single_hat

    single_noise_sd = noise_sigma / math.sqrt(float(n_cells))
    pair_noise_sd = pair_noise_multiplier * noise_sigma / math.sqrt(float(n_cells))

    # Approximate variance of correction noise under independent pair/single noise:
    # c_i = 0.5 * sum_{j!=i}(eta_ij - eps_i - eps_j).
    noise_var = 0.25 * ((k - 1) * (pair_noise_sd ** 2) + k * (k - 1) * (single_noise_sd ** 2))
    observed_var = float(np.var(correction, ddof=1)) if correction.size > 1 else 0.0

    shrink_weight = float(np.clip(1.0 - (noise_var / max(observed_var, EPS)), 0.0, 1.0))
    shrunk = single_hat + shrink_weight * correction
    return shrunk, shrink_weight


def synthetic_estimator_benchmark(
    n_grid: Sequence[int],
    pair_scales: Sequence[float],
    scenarios: Sequence[Tuple[str, float]],
    replicates: int,
    k: int,
    noise_sigma: float,
    pair_noise_multiplier: float,
    triple_density: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark single, raw pairwise, and shrinkage-pairwise estimators."""
    rng = np.random.default_rng(seed)

    trial_rows: List[Dict[str, object]] = []

    for scenario_name, triple_scale in scenarios:
        for pair_scale in pair_scales:
            for n_cells in n_grid:
                for _ in range(replicates):
                    alpha, beta, triples = _sample_game_coefficients(
                        rng=rng,
                        k=k,
                        pair_scale=pair_scale,
                        triple_scale=triple_scale,
                        triple_density=triple_density,
                    )
                    phi_true = _compute_true_shapley(alpha, beta, triples)

                    est_single, est_paircorr = _simulate_single_and_pairwise_estimates(
                        rng=rng,
                        alpha=alpha,
                        beta=beta,
                        n_cells=n_cells,
                        noise_sigma=noise_sigma,
                        pair_noise_multiplier=pair_noise_multiplier,
                    )

                    est_pairshrink, shrink_weight = _shrink_pairwise_correction(
                        single_hat=est_single,
                        paircorr_hat=est_paircorr,
                        n_cells=n_cells,
                        k=k,
                        noise_sigma=noise_sigma,
                        pair_noise_multiplier=pair_noise_multiplier,
                    )

                    top_true = int(np.argmax(np.abs(phi_true)))
                    top_single = int(np.argmax(np.abs(est_single)))
                    top_paircorr = int(np.argmax(np.abs(est_paircorr)))
                    top_pairshrink = int(np.argmax(np.abs(est_pairshrink)))

                    trial_rows.append(
                        {
                            "scenario": scenario_name,
                            "pair_scale": float(pair_scale),
                            "triple_scale": float(triple_scale),
                            "n_cells": int(n_cells),
                            "shrink_weight": float(shrink_weight),
                            "spearman_single": _spearman_abs(est_single, phi_true),
                            "spearman_paircorr": _spearman_abs(est_paircorr, phi_true),
                            "spearman_pairshrink": _spearman_abs(est_pairshrink, phi_true),
                            "mae_single": float(np.mean(np.abs(est_single - phi_true))),
                            "mae_paircorr": float(np.mean(np.abs(est_paircorr - phi_true))),
                            "mae_pairshrink": float(np.mean(np.abs(est_pairshrink - phi_true))),
                            "top1_single": float(top_single == top_true),
                            "top1_paircorr": float(top_paircorr == top_true),
                            "top1_pairshrink": float(top_pairshrink == top_true),
                            "inv_single": _pairwise_inversion_rate(est_single, phi_true),
                            "inv_paircorr": _pairwise_inversion_rate(est_paircorr, phi_true),
                            "inv_pairshrink": _pairwise_inversion_rate(est_pairshrink, phi_true),
                        }
                    )

    trial_df = pd.DataFrame(trial_rows)

    summary_df = (
        trial_df.groupby(["scenario", "pair_scale", "triple_scale", "n_cells"], as_index=False)
        .agg(
            shrink_weight_mean=("shrink_weight", "mean"),
            spearman_single_mean=("spearman_single", "mean"),
            spearman_paircorr_mean=("spearman_paircorr", "mean"),
            spearman_pairshrink_mean=("spearman_pairshrink", "mean"),
            mae_single_mean=("mae_single", "mean"),
            mae_paircorr_mean=("mae_paircorr", "mean"),
            mae_pairshrink_mean=("mae_pairshrink", "mean"),
            top1_single_mean=("top1_single", "mean"),
            top1_paircorr_mean=("top1_paircorr", "mean"),
            top1_pairshrink_mean=("top1_pairshrink", "mean"),
            inv_single_mean=("inv_single", "mean"),
            inv_paircorr_mean=("inv_paircorr", "mean"),
            inv_pairshrink_mean=("inv_pairshrink", "mean"),
        )
        .sort_values(["scenario", "pair_scale", "n_cells"])
        .reset_index(drop=True)
    )

    summary_df["spearman_gain_paircorr"] = summary_df["spearman_paircorr_mean"] - summary_df["spearman_single_mean"]
    summary_df["spearman_gain_pairshrink"] = summary_df["spearman_pairshrink_mean"] - summary_df["spearman_single_mean"]
    summary_df["top1_gain_paircorr"] = summary_df["top1_paircorr_mean"] - summary_df["top1_single_mean"]
    summary_df["top1_gain_pairshrink"] = summary_df["top1_pairshrink_mean"] - summary_df["top1_single_mean"]
    summary_df["inv_reduction_paircorr"] = summary_df["inv_single_mean"] - summary_df["inv_paircorr_mean"]
    summary_df["inv_reduction_pairshrink"] = summary_df["inv_single_mean"] - summary_df["inv_pairshrink_mean"]
    summary_df["mae_reduction_paircorr"] = summary_df["mae_single_mean"] - summary_df["mae_paircorr_mean"]
    summary_df["mae_reduction_pairshrink"] = summary_df["mae_single_mean"] - summary_df["mae_pairshrink_mean"]

    return trial_df, summary_df


def synthetic_certification_benchmark(
    n_grid: Sequence[int],
    pair_scales: Sequence[float],
    scenarios: Sequence[Tuple[str, float]],
    replicates: int,
    k: int,
    noise_sigma: float,
    seed: int,
    triple_density: float,
) -> pd.DataFrame:
    """Validate ranking-certification behavior with explicit per-component bounds.

    The per-component bound uses the true omitted interaction magnitude plus
    a finite-sample noise term to mimic theorem assumptions.
    """
    rng = np.random.default_rng(seed + 123)
    rows: List[Dict[str, object]] = []

    for scenario_name, triple_scale in scenarios:
        for pair_scale in pair_scales:
            for n_cells in n_grid:
                total_pairs = 0
                naive_errors = 0
                certified_pairs = 0
                certified_errors = 0

                for _ in range(replicates):
                    alpha, beta, triples = _sample_game_coefficients(
                        rng=rng,
                        k=k,
                        pair_scale=pair_scale,
                        triple_scale=triple_scale,
                        triple_density=triple_density,
                    )
                    phi_true = _compute_true_shapley(alpha, beta, triples)

                    est_single, _ = _simulate_single_and_pairwise_estimates(
                        rng=rng,
                        alpha=alpha,
                        beta=beta,
                        n_cells=n_cells,
                        noise_sigma=noise_sigma,
                        pair_noise_multiplier=1.0,
                    )

                    # Structural bias upper bound from known coefficient magnitudes.
                    b_pair = 0.5 * np.sum(np.abs(beta), axis=1)
                    b_triple = np.zeros(k, dtype=float)
                    for i, j, l, val in triples:
                        contrib = abs(val) / 3.0
                        b_triple[i] += contrib
                        b_triple[j] += contrib
                        b_triple[l] += contrib

                    noise_bound = Z_95 * noise_sigma / math.sqrt(float(n_cells))
                    b_total = b_pair + b_triple + noise_bound

                    for i, j in itertools.combinations(range(k), 2):
                        total_pairs += 1
                        diff_hat = float(est_single[i] - est_single[j])
                        diff_true = float(phi_true[i] - phi_true[j])
                        mismatch = diff_hat * diff_true <= 0.0
                        naive_errors += int(mismatch)

                        if abs(diff_hat) > float(b_total[i] + b_total[j]):
                            certified_pairs += 1
                            certified_errors += int(mismatch)

                rows.append(
                    {
                        "scenario": scenario_name,
                        "pair_scale": float(pair_scale),
                        "triple_scale": float(triple_scale),
                        "n_cells": int(n_cells),
                        "total_pairs": int(total_pairs),
                        "certified_pairs": int(certified_pairs),
                        "certified_fraction": certified_pairs / float(total_pairs),
                        "naive_error_rate": naive_errors / float(total_pairs),
                        "certified_error_rate": certified_errors / float(max(certified_pairs, 1)),
                    }
                )

    return pd.DataFrame(rows).sort_values(["scenario", "pair_scale", "n_cells"]).reset_index(drop=True)


def plot_real_residuals(pair_diag_df: pd.DataFrame, output_path: Path) -> None:
    """Figure 1: total effect, residual, and non-additivity lower bound."""
    df = pair_diag_df.copy()
    df["label"] = df["run_id"] + "\n" + df["pair"]

    x = np.arange(df.shape[0])
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(10, 1.6 * df.shape[0]), 5.8), dpi=170)
    ax.bar(x - w, df["te_mean"].abs().to_numpy(), width=w, label="|Total effect|", color="#376BB5")
    ax.bar(x, df["residual_abs"].to_numpy(), width=w, label="|Residual|", color="#D7835E")
    ax.bar(x + w, df["nonadditivity_lb"].to_numpy(), width=w, label="Non-additivity lower bound", color="#4E9E72")

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("Effect magnitude")
    ax.set_title("Observed residual non-additivity across run/pair combinations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_ranking_sensitivity(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Figure 2: average certified fraction vs structural-bias multiplier."""
    grouped = (
        summary_df.groupby(["granularity", "lambda_multiplier"], as_index=False)
        .agg(
            uniform=("certified_fraction_uniform", "mean"),
            proportional=("certified_fraction_proportional", "mean"),
            concentrated=("certified_fraction_concentrated", "mean"),
        )
    )

    granularities = sorted(grouped["granularity"].unique().tolist())
    fig, axes = plt.subplots(1, len(granularities), figsize=(6.2 * len(granularities), 4.6), dpi=170, squeeze=False)

    for idx, gran in enumerate(granularities):
        ax = axes[0, idx]
        sub = grouped[grouped["granularity"] == gran].sort_values("lambda_multiplier")
        x = sub["lambda_multiplier"].to_numpy(dtype=float)

        ax.plot(x, sub["uniform"].to_numpy(dtype=float), marker="o", label="Uniform allocation")
        ax.plot(x, sub["proportional"].to_numpy(dtype=float), marker="s", label="Proportional allocation")
        ax.plot(x, sub["concentrated"].to_numpy(dtype=float), marker="^", label="Concentrated adversary")

        ax.set_title(f"{gran} mediators")
        ax.set_xlabel("Assumed structural multiplier (lambda)")
        ax.set_ylabel("Mean certified pair fraction")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    axes[0, -1].legend(frameon=False, loc="upper right")
    fig.suptitle("Ranking certification sensitivity to structural-bias assumptions", y=1.03)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cross_tissue_scatter(component_df: pd.DataFrame, output_path: Path) -> None:
    """Figure 3: kidney vs lung scatter for shared pair components."""
    shared = component_df[component_df["pair"] == "GATA3->GATA3"].copy()
    if shared.empty:
        return

    granularities = sorted(shared["granularity"].unique().tolist())
    fig, axes = plt.subplots(1, len(granularities), figsize=(6.0 * len(granularities), 4.8), dpi=170, squeeze=False)

    for idx, gran in enumerate(granularities):
        ax = axes[0, idx]
        sub = shared[shared["granularity"] == gran]
        if sub.empty:
            continue

        kid = sub[sub["tissue"] == "kidney"][["component_id", "effect_mean"]].rename(columns={"effect_mean": "kidney"})
        lung = sub[sub["tissue"] == "lung"][["component_id", "effect_mean"]].rename(columns={"effect_mean": "lung"})
        merged = kid.merge(lung, on="component_id", how="inner")
        if merged.empty:
            continue

        x = merged["kidney"].to_numpy(dtype=float)
        y = merged["lung"].to_numpy(dtype=float)

        ax.scatter(x, y, s=26, alpha=0.75, edgecolor="black", linewidth=0.25)
        lo = min(float(x.min()), float(y.min()))
        hi = max(float(x.max()), float(y.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.4", linewidth=1.0)
        rho = _safe_corr(x, y, method="spearman")

        ax.set_title(f"{gran} (Spearman={rho:.2f})")
        ax.set_xlabel("Kidney component effect")
        ax.set_ylabel("Lung component effect")
        ax.grid(alpha=0.25)

    fig.suptitle("Cross-tissue component alignment for GATA3->GATA3", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _heatmap_panel(ax: plt.Axes, matrix: np.ndarray, xvals: Sequence[int], yvals: Sequence[float], title: str, cmap: str, vmin: float | None = None, vmax: float | None = None) -> None:
    """Utility for consistent heatmap rendering."""
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(xvals)))
    ax.set_xticklabels([str(x) for x in xvals])
    ax.set_yticks(np.arange(len(yvals)))
    ax.set_yticklabels([f"{y:.1f}" for y in yvals])
    ax.set_xlabel("n_cells")
    ax.set_ylabel("pair_scale")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_simulation_gains(sim_summary_df: pd.DataFrame, output_path: Path) -> None:
    """Figure 4: heatmaps of raw vs shrinkage pairwise-correction gains."""
    scenarios = sorted(sim_summary_df["scenario"].unique().tolist())
    n_vals = sorted(sim_summary_df["n_cells"].unique().tolist())
    p_vals = sorted(sim_summary_df["pair_scale"].unique().tolist())

    fig, axes = plt.subplots(len(scenarios), 2, figsize=(13.2, 4.6 * len(scenarios)), dpi=170, squeeze=False)

    for r, scenario in enumerate(scenarios):
        sub = sim_summary_df[sim_summary_df["scenario"] == scenario]
        mat_raw = np.zeros((len(p_vals), len(n_vals)), dtype=float)
        mat_shrink = np.zeros((len(p_vals), len(n_vals)), dtype=float)

        p_to_i = {v: i for i, v in enumerate(p_vals)}
        n_to_j = {v: j for j, v in enumerate(n_vals)}

        for row in sub.itertuples(index=False):
            i = p_to_i[float(row.pair_scale)]
            j = n_to_j[int(row.n_cells)]
            mat_raw[i, j] = float(row.spearman_gain_paircorr)
            mat_shrink[i, j] = float(row.spearman_gain_pairshrink)

        _heatmap_panel(
            axes[r, 0],
            mat_raw,
            xvals=n_vals,
            yvals=p_vals,
            title=f"{scenario}: Spearman gain (raw pairwise - single)",
            cmap="RdBu",
            vmin=-0.2,
            vmax=0.2,
        )
        _heatmap_panel(
            axes[r, 1],
            mat_shrink,
            xvals=n_vals,
            yvals=p_vals,
            title=f"{scenario}: Spearman gain (shrink pairwise - single)",
            cmap="RdBu",
            vmin=-0.2,
            vmax=0.2,
        )

    fig.suptitle("Synthetic estimator benchmark", y=1.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_synthetic_certification(cert_df: pd.DataFrame, output_path: Path) -> None:
    """Figure 5: certification fraction and certified error heatmaps."""
    scenario = "mixed_higher_order"
    sub = cert_df[cert_df["scenario"] == scenario].copy()
    if sub.empty:
        sub = cert_df.copy()

    n_vals = sorted(sub["n_cells"].unique().tolist())
    p_vals = sorted(sub["pair_scale"].unique().tolist())

    mat_cf = np.zeros((len(p_vals), len(n_vals)), dtype=float)
    mat_ce = np.zeros((len(p_vals), len(n_vals)), dtype=float)
    mat_ne = np.zeros((len(p_vals), len(n_vals)), dtype=float)

    p_to_i = {v: i for i, v in enumerate(p_vals)}
    n_to_j = {v: j for j, v in enumerate(n_vals)}

    for row in sub.itertuples(index=False):
        i = p_to_i[float(row.pair_scale)]
        j = n_to_j[int(row.n_cells)]
        mat_cf[i, j] = float(row.certified_fraction)
        mat_ce[i, j] = float(row.certified_error_rate)
        mat_ne[i, j] = float(row.naive_error_rate)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), dpi=170)
    _heatmap_panel(axes[0], mat_cf, n_vals, p_vals, "Certified fraction", cmap="Blues", vmin=0.0, vmax=max(0.2, float(np.nanmax(mat_cf))))
    _heatmap_panel(axes[1], mat_ce, n_vals, p_vals, "Certified error rate", cmap="Reds", vmin=0.0, vmax=max(0.02, float(np.nanmax(mat_ce))))
    _heatmap_panel(axes[2], mat_ne, n_vals, p_vals, "Naive error rate", cmap="Reds", vmin=0.0, vmax=max(0.25, float(np.nanmax(mat_ne))))

    fig.suptitle(f"Synthetic certification benchmark ({scenario})", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _fmt_float(v: float, digits: int = 4) -> str:
    """Format finite float for report readability."""
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def _table_excerpt(df: pd.DataFrame, max_rows: int = 12) -> str:
    """Render compact table excerpt for markdown report body."""
    if df.empty:
        return "(empty)"
    out = df.head(max_rows).copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: _fmt_float(float(x), digits=4))
    return out.to_string(index=False)


def write_full_paper(
    report_path: Path,
    manifest_df: pd.DataFrame,
    pair_diag_df: pd.DataFrame,
    ranking_summary_df: pd.DataFrame,
    flip_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    sim_summary_df: pd.DataFrame,
    cert_df: pd.DataFrame,
    output_dir_rel: str,
) -> None:
    """Write full-length manuscript-style markdown draft with finalized results."""
    n_runs = int(manifest_df.shape[0])
    n_pairs = int(pair_diag_df.shape[0])
    n_pairs_nonadd = int((pair_diag_df["nonadditivity_lb"] > 0).sum())
    median_nonadd_ratio = float(pair_diag_df["nonadditivity_lb_over_te"].median())
    run_dirs = sorted(set(manifest_df["checkpoint_path"].astype(str).map(lambda p: str(Path(p).parent))))

    worst_pair = pair_diag_df.sort_values("nonadditivity_lb_over_te", ascending=False).iloc[0]

    rank_at_lambda1 = ranking_summary_df[ranking_summary_df["lambda_multiplier"] == 1.0]
    rank_at_lambda4 = ranking_summary_df[ranking_summary_df["lambda_multiplier"] == 4.0]

    rank_l1_uniform = float(rank_at_lambda1["certified_fraction_uniform"].mean()) if not rank_at_lambda1.empty else float("nan")
    rank_l1_conc = float(rank_at_lambda1["certified_fraction_concentrated"].mean()) if not rank_at_lambda1.empty else float("nan")
    rank_l4_uniform = float(rank_at_lambda4["certified_fraction_uniform"].mean()) if not rank_at_lambda4.empty else float("nan")
    rank_l4_conc = float(rank_at_lambda4["certified_fraction_concentrated"].mean()) if not rank_at_lambda4.empty else float("nan")

    if cross_df.empty:
        cross_main_sentence = "Cross-tissue overlap could not be computed because no pair is shared across tissues in this snapshot."
    else:
        cross_range_low = float(cross_df["spearman"].min())
        cross_range_high = float(cross_df["spearman"].max())

        gl = cross_df[(cross_df["pair"] == "GATA3->GATA3") & (cross_df["tissue_a"] == "kidney") & (cross_df["tissue_b"] == "lung")]
        if gl.empty:
            gl = cross_df.head(1)

        gl_summ = []
        for row in gl.itertuples(index=False):
            gl_summ.append(
                f"{row.granularity}: Spearman={_fmt_float(float(row.spearman),3)} "
                f"(95% CI {_fmt_float(float(row.spearman_ci_low),3)} to {_fmt_float(float(row.spearman_ci_high),3)})"
            )
        cross_main_sentence = (
            f"Across {cross_df.shape[0]} cross-tissue comparisons, Spearman ranges from "
            f"{_fmt_float(cross_range_low,3)} to {_fmt_float(cross_range_high,3)}. "
            f"Kidney-vs-lung GATA3 alignment: " + "; ".join(gl_summ) + "."
        )

    sim_focus = sim_summary_df[(sim_summary_df["scenario"] == "mixed_higher_order") & (sim_summary_df["pair_scale"] == 0.8)]
    if sim_focus.empty:
        sim_focus = sim_summary_df.copy()
    sim_focus_best = sim_focus.sort_values("n_cells").iloc[-1]

    sim_focus_strong = sim_summary_df[(sim_summary_df["scenario"] == "mixed_higher_order") & (sim_summary_df["pair_scale"] == 1.2)]
    if sim_focus_strong.empty:
        sim_focus_strong = sim_summary_df.copy()
    sim_focus_strong_best = sim_focus_strong.sort_values("n_cells").iloc[-1]

    mixed = sim_summary_df[sim_summary_df["scenario"] == "mixed_higher_order"].copy()
    if mixed.empty:
        mixed = sim_summary_df.copy()
    share_raw_positive = float(np.mean(mixed["spearman_gain_paircorr"] > 0.0))
    share_shrink_positive = float(np.mean(mixed["spearman_gain_pairshrink"] > 0.0))

    cert_focus = cert_df[(cert_df["scenario"] == "mixed_higher_order") & (cert_df["pair_scale"] == 0.8)]
    if cert_focus.empty:
        cert_focus = cert_df.copy()
    cert_focus_best = cert_focus.sort_values("n_cells").iloc[-1]

    # Evidence tables for manuscript body.
    pair_excerpt = pair_diag_df[
        [
            "run_id",
            "pair",
            "te_mean",
            "sum_single_mean",
            "residual_mean",
            "nonadditivity_lb",
            "nonadditivity_lb_over_te",
            "top1_top2_gap",
        ]
    ]
    rank_excerpt = ranking_summary_df[
        [
            "run_id",
            "pair",
            "lambda_multiplier",
            "certified_fraction_uniform",
            "certified_fraction_proportional",
            "certified_fraction_concentrated",
        ]
    ]
    rank_excerpt = rank_excerpt[rank_excerpt["lambda_multiplier"].isin([1.0, 2.0, 4.0, 8.0])]

    flip_excerpt = flip_df[
        [
            "run_id",
            "pair",
            "top1_component",
            "top2_component",
            "required_structural_for_flip",
            "nonadditivity_lb",
            "lambda_flip_threshold",
        ]
    ]

    sim_excerpt = sim_summary_df[
        [
            "scenario",
            "pair_scale",
            "n_cells",
            "spearman_single_mean",
            "spearman_paircorr_mean",
            "spearman_pairshrink_mean",
            "spearman_gain_paircorr",
            "spearman_gain_pairshrink",
            "top1_single_mean",
            "top1_paircorr_mean",
            "top1_pairshrink_mean",
            "top1_gain_paircorr",
            "top1_gain_pairshrink",
            "shrink_weight_mean",
        ]
    ]

    cert_excerpt = cert_df[
        [
            "scenario",
            "pair_scale",
            "n_cells",
            "certified_fraction",
            "certified_error_rate",
            "naive_error_rate",
        ]
    ]

    lines: List[str] = []
    lines.append("# Quantifying and Stress-Testing Bias in Patching-Based Mediation Estimates")
    lines.append("")
    lines.append("## Abstract")
    lines.append(
        "Single-component patching is widely used for mechanistic mediation maps, but its statistical and causal failure modes under mediator interactions remain under-specified. "
        "We provide a theory-first treatment and low-compute empirical evaluation using existing scGPT mediation traces. "
        "We derive a decomposition of single-component bias into omitted interaction bias, higher-order nonlinear composition bias, and finite-sample error; prove a ranking-validity certificate under componentwise bias bounds; and define an observable residual lower bound on aggregate non-additivity. "
        "On the frozen run baseline ({3} runs), {0}/{1} run-pairs show non-zero lower-bounded non-additivity, with median lower-bound-to-total-effect ratio {2}. "
        "Adversarial sensitivity analysis shows certified ranking coverage drops sharply as structural-bias assumptions tighten. "
        "Synthetic benchmarks show raw pairwise correction is variance-sensitive (positive Spearman gain in {4:.1%} of mixed-regime grid cells), while shrinkage-stabilized correction is positive in {5:.1%} of grid cells and dominates in high-interaction, high-sample settings. "
        "Certification remains conservative with near-zero error among certified pairs. "
        "These results convert mediation maps from heuristic artifacts into estimators with explicit uncertainty and stress-tested interpretability limits.".format(
            n_pairs_nonadd,
            n_pairs,
            _fmt_float(median_nonadd_ratio, 3),
            n_runs,
            share_raw_positive,
            share_shrink_positive,
        )
    )
    lines.append("")

    lines.append("## 1. Introduction")
    lines.append(
        "Patching-based mediation has become a practical tool for probing internal biological foundation model circuits. "
        "In current practice, mediator importance is often estimated component-by-component, even when true effects may be distributed across interacting heads/MLPs. "
        "This creates a bias-risk tradeoff: single-component patching is cheap and scalable, but can mis-rank mediators when interaction terms are non-negligible."
    )
    lines.append(
        "This paper addresses that gap with a deliberately low-compute, theory-first agenda: derive bias bounds, connect them to observable diagnostics, and pressure-test ranking claims with adversarial assumptions."
    )
    lines.append("")

    lines.append("## 2. Setup and Notation")
    lines.append("Let `F(S)` denote expected model output when mediator subset `S` is patched to clean and all others are ablated.")
    lines.append("Mobius coefficients are")
    lines.append("`mu(S) = sum_{T subseteq S} (-1)^{|S|-|T|} F(T)`.")
    lines.append("Total effect:")
    lines.append("`TE = F([k]) - F(empty) = sum_{S != empty} mu(S)`.")
    lines.append("Single-mediator estimator:")
    lines.append("`mhat_i = F({i}) - F(empty) + eps_i`.")
    lines.append("")

    lines.append("## 3. Theory")
    lines.append("### 3.1 Bias Decomposition")
    lines.append("For Shapley attribution `phi_i = sum_{S contains i} mu(S)/|S|`,")
    lines.append("`b_i = mhat_i - phi_i = -sum_{|S|>=2, i in S} mu(S)/|S| + eps_i`. ")
    lines.append("Decompose `b_i = b_i_pair + b_i_high + eps_i` with")
    lines.append("- `b_i_pair = -(1/2) * sum_{j != i} mu({i,j})`,")
    lines.append("- `b_i_high = -sum_{|S|>=3, i in S} mu(S)/|S|`. ")
    lines.append("")

    lines.append("### 3.2 Observable Non-Additivity Lower Bound")
    lines.append("Define residual `R = TE - sum_i mhat_i`. Then")
    lines.append("`R = sum_{|S|>=2} mu(S) - sum_i eps_i`. ")
    lines.append("Let `A = sum_{|S|>=2} |mu(S)|`. By triangle inequality,")
    lines.append("`A >= |R| - |sum_i eps_i|`. ")
    lines.append("Using propagated Gaussian SE approximation:")
    lines.append("`A_lb = max(0, |R| - 1.96 * SE(R))`. ")
    lines.append("")

    lines.append("### 3.3 Ranking Certificate")
    lines.append("If for mediators i,j we have `mhat_i - mhat_j > B_i + B_j` with bounds `|b_i|<=B_i`, `|b_j|<=B_j`, then `phi_i > phi_j`.")
    lines.append("We use this for certified ranking subsets and stress-test via structural multiplier sensitivity because `A_lb` is a lower bound, not an upper bound.")
    lines.append("")

    lines.append("## 4. Data and Protocol")
    lines.append(f"We analyze a frozen baseline of `{n_runs}` mediation runs with `{n_pairs}` run-pairs from:")
    for run_dir in run_dirs:
        lines.append(f"- `{run_dir}`")
    lines.append("All results were generated by:")
    lines.append("`python subproject_22_patching_mediation_bias_bounds/implementation/scripts/run_full_paper_pipeline.py`.")
    lines.append("")

    lines.append("## 5. Real-Data Results")
    lines.append("### 5.1 Residual Non-Additivity")
    lines.append(
        "Evidence: non-additivity lower bounds are positive in {0}/{1} run-pairs; median `A_lb/|TE|={2}`. "
        "Worst-case run-pair by ratio is `{3}:{4}` with `A_lb/|TE|={5}`.".format(
            n_pairs_nonadd,
            n_pairs,
            _fmt_float(median_nonadd_ratio, 3),
            worst_pair["run_id"],
            worst_pair["pair"],
            _fmt_float(float(worst_pair["nonadditivity_lb_over_te"]), 3),
        )
    )
    lines.append("Interpretation: additive single-component explanations are frequently insufficient at observed sample sizes.")
    lines.append("Limitation: `A_lb` is conservative and can be zero under high uncertainty even if true interactions exist.")
    lines.append("![Figure 1: Real-data residual non-additivity diagnostics.](../outputs/figures/fig1_real_residual_nonadditivity.png)")
    lines.append("```text")
    lines.append(_table_excerpt(pair_excerpt, max_rows=20))
    lines.append("```")
    lines.append("")

    lines.append("### 5.2 Adversarial Ranking Sensitivity")
    lines.append(
        "At lambda=1, mean certified fraction is uniform={0}, concentrated={1}; "
        "at lambda=4 it drops to uniform={2}, concentrated={3}.".format(
            _fmt_float(rank_l1_uniform, 3),
            _fmt_float(rank_l1_conc, 3),
            _fmt_float(rank_l4_uniform, 3),
            _fmt_float(rank_l4_conc, 3),
        )
    )
    lines.append("Interpretation: mediator ranking claims are highly assumption-sensitive once structural interaction budgets are allowed to grow beyond observed lower bounds.")
    lines.append("![Figure 2: Ranking-certification sensitivity under adversarial structural-bias assumptions.](../outputs/figures/fig2_ranking_sensitivity.png)")
    lines.append("```text")
    lines.append(_table_excerpt(rank_excerpt, max_rows=24))
    lines.append("```")
    lines.append("Top-1 flip threshold diagnostics:")
    lines.append("```text")
    lines.append(_table_excerpt(flip_excerpt, max_rows=20))
    lines.append("```")
    lines.append("")

    lines.append("### 5.3 Cross-Tissue Alignment")
    lines.append(cross_main_sentence)
    lines.append("![Figure 3: Cross-tissue component effect alignment for shared pairs.](../outputs/figures/fig3_cross_tissue_scatter.png)")
    if not cross_df.empty:
        lines.append("```text")
        lines.append(_table_excerpt(cross_df, max_rows=12))
        lines.append("```")
    lines.append("Implication: transportability of mediator maps across tissue contexts can vary by granularity and should be uncertainty-quantified.")
    lines.append("")

    lines.append("## 6. Synthetic Benchmarks")
    lines.append("We simulate additive + pairwise + higher-order mediator interactions with known ground-truth Shapley attributions.")
    lines.append("### 6.1 Estimator Recovery")
    lines.append(
        "In mixed higher-order setting at pair_scale=0.8 and largest n, raw pairwise correction yields Spearman {0} vs single {1}, "
        "while shrinkage-pairwise yields {2}. Top-1 accuracy is raw={3}, single={4}, shrinkage={5}. "
        "At stronger interaction (pair_scale=1.2), shrinkage reaches Spearman {6} and top-1 accuracy {7}.".format(
            _fmt_float(float(sim_focus_best["spearman_paircorr_mean"]), 3),
            _fmt_float(float(sim_focus_best["spearman_single_mean"]), 3),
            _fmt_float(float(sim_focus_best["spearman_pairshrink_mean"]), 3),
            _fmt_float(float(sim_focus_best["top1_paircorr_mean"]), 3),
            _fmt_float(float(sim_focus_best["top1_single_mean"]), 3),
            _fmt_float(float(sim_focus_best["top1_pairshrink_mean"]), 3),
            _fmt_float(float(sim_focus_strong_best["spearman_pairshrink_mean"]), 3),
            _fmt_float(float(sim_focus_strong_best["top1_pairshrink_mean"]), 3),
        )
    )
    lines.append(
        "Interpretation: naive pairwise correction can overfit noise due contrast accumulation, while shrinkage can recover interaction signal when sample size and interaction magnitude are sufficient."
    )
    lines.append("![Figure 4: Synthetic estimator gains (raw pairwise vs shrinkage-pairwise).](../outputs/figures/fig4_synthetic_estimator_gains.png)")
    lines.append("```text")
    lines.append(_table_excerpt(sim_excerpt, max_rows=24))
    lines.append("```")
    lines.append("")

    lines.append("### 6.2 Certification Behavior")
    lines.append(
        "At mixed higher-order pair_scale=0.8 and largest n, certified fraction={0}, certified error rate={1}, naive error rate={2}.".format(
            _fmt_float(float(cert_focus_best["certified_fraction"]), 3),
            _fmt_float(float(cert_focus_best["certified_error_rate"]), 4),
            _fmt_float(float(cert_focus_best["naive_error_rate"]), 3),
        )
    )
    lines.append("Interpretation: certification is conservative (coverage tradeoff) but effectively filters ranking errors under valid bounds.")
    lines.append("![Figure 5: Synthetic certification coverage vs error tradeoff.](../outputs/figures/fig5_synthetic_certification.png)")
    lines.append("```text")
    lines.append(_table_excerpt(cert_excerpt, max_rows=24))
    lines.append("```")
    lines.append("")

    lines.append("## 7. Practical Correction Strategies")
    lines.append("1. Pairwise augmentation (preferred when additional runs are feasible):")
    lines.append("   `mtilde_i = mhat_i + 0.5 * sum_j (F({i,j}) - F({i}) - F({j}))` for selected high-priority mediator neighbors.")
    lines.append("2. Residual-bound interval reporting (no new runs):")
    lines.append("   report mediator uncertainty intervals inflated by structural budget sensitivity over lambda-grid assumptions.")
    lines.append("3. Claim gating: report only certified rankings at predeclared lambda and allocation model.")
    lines.append("")

    lines.append("## 8. Limitations")
    lines.append("- Real-data baseline remains sparse and unbalanced across tissues/granularities, with missing pair coverage in several runs.")
    lines.append("- Non-additivity diagnostics provide lower bounds; structural upper bounds remain assumption-dependent.")
    lines.append("- Synthetic game family is stylized and does not capture all transformer nonlinearities.")
    lines.append("- Summary-statistic re-analysis cannot replace full raw activation-level mediation reruns.")
    lines.append("")

    lines.append("## 9. Conclusion")
    lines.append(
        "Single-component patching mediation can be useful, but only with explicit bias accounting, adversarial sensitivity analysis, and bounded-claim reporting. "
        "This work provides a concrete framework and executable pipeline to make those safeguards operational at low compute cost."
    )
    lines.append("")

    lines.append("## Reproducibility Artifacts")
    lines.append(f"All artifact paths are under `{output_dir_rel}`:")
    lines.append("- `tables/run_manifest.csv`")
    lines.append("- `tables/pair_bias_diagnostics.csv`")
    lines.append("- `tables/ranking_certification_sensitivity.csv`")
    lines.append("- `tables/ranking_certification_detail.csv`")
    lines.append("- `tables/top1_flip_thresholds.csv`")
    lines.append("- `tables/cross_tissue_alignment.csv`")
    lines.append("- `tables/synthetic_estimator_trial_metrics.csv`")
    lines.append("- `tables/synthetic_estimator_summary.csv`")
    lines.append("- `tables/synthetic_certification_summary.csv`")
    lines.append("- `figures/fig1_real_residual_nonadditivity.png`")
    lines.append("- `figures/fig2_ranking_sensitivity.png`")
    lines.append("- `figures/fig3_cross_tissue_scatter.png`")
    lines.append("- `figures/fig4_synthetic_estimator_gains.png`")
    lines.append("- `figures/fig5_synthetic_certification.png`")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full paper pipeline for Proposal 5")
    parser.add_argument(
        "--output-dir",
        default="subproject_22_patching_mediation_bias_bounds/outputs",
        help="Output directory (tables/, figures/ will be created)",
    )
    parser.add_argument(
        "--report-path",
        default="subproject_22_patching_mediation_bias_bounds/reports/final_paper_bias_bounds_mediation.md",
        help="Final manuscript markdown output path",
    )
    parser.add_argument(
        "--overwrite-report",
        action="store_true",
        help="Overwrite report-path if it already exists",
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=3000,
        help="Bootstrap draws for cross-tissue uncertainty propagation",
    )
    parser.add_argument(
        "--sim-replicates",
        type=int,
        default=600,
        help="Replicates per synthetic benchmark grid point",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    output_dir = repo_root / args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    report_path = repo_root / args.report_path

    runs = discover_runs(repo_root)
    manifest_df, pair_df, component_df = load_real_data(repo_root, runs)

    pair_diag_df = compute_pair_diagnostics(pair_df, component_df)

    lambda_grid = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    rank_summary_df, rank_detail_df, flip_df = ranking_certification_sensitivity(
        component_df=component_df,
        pair_diag_df=pair_diag_df,
        lambda_grid=lambda_grid,
    )

    cross_df = cross_tissue_alignment(
        component_df=component_df,
        bootstrap_draws=args.bootstrap_draws,
        seed=RNG_SEED,
    )

    sim_n_grid = [3, 5, 10, 20, 40, 80]
    sim_pair_scales = [0.0, 0.2, 0.4, 0.8, 1.2]
    sim_scenarios = [("pairwise_only", 0.0), ("mixed_higher_order", 0.35)]

    sim_trial_df, sim_summary_df = synthetic_estimator_benchmark(
        n_grid=sim_n_grid,
        pair_scales=sim_pair_scales,
        scenarios=sim_scenarios,
        replicates=args.sim_replicates,
        k=12,
        noise_sigma=1.0,
        pair_noise_multiplier=1.1,
        triple_density=0.3,
        seed=RNG_SEED,
    )

    cert_df = synthetic_certification_benchmark(
        n_grid=sim_n_grid,
        pair_scales=sim_pair_scales,
        scenarios=sim_scenarios,
        replicates=args.sim_replicates,
        k=12,
        noise_sigma=1.0,
        seed=RNG_SEED,
        triple_density=0.3,
    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest_df.to_csv(tables_dir / "run_manifest.csv", index=False)
    pair_df.to_csv(tables_dir / "pair_effects.csv", index=False)
    component_df.to_csv(tables_dir / "component_effects.csv", index=False)
    pair_diag_df.to_csv(tables_dir / "pair_bias_diagnostics.csv", index=False)
    rank_summary_df.to_csv(tables_dir / "ranking_certification_sensitivity.csv", index=False)
    rank_detail_df.to_csv(tables_dir / "ranking_certification_detail.csv", index=False)
    flip_df.to_csv(tables_dir / "top1_flip_thresholds.csv", index=False)
    cross_df.to_csv(tables_dir / "cross_tissue_alignment.csv", index=False)
    sim_trial_df.to_csv(tables_dir / "synthetic_estimator_trial_metrics.csv", index=False)
    sim_summary_df.to_csv(tables_dir / "synthetic_estimator_summary.csv", index=False)
    cert_df.to_csv(tables_dir / "synthetic_certification_summary.csv", index=False)

    plot_real_residuals(pair_diag_df, figures_dir / "fig1_real_residual_nonadditivity.png")
    plot_ranking_sensitivity(rank_summary_df, figures_dir / "fig2_ranking_sensitivity.png")
    plot_cross_tissue_scatter(component_df, figures_dir / "fig3_cross_tissue_scatter.png")
    plot_simulation_gains(sim_summary_df, figures_dir / "fig4_synthetic_estimator_gains.png")
    plot_synthetic_certification(cert_df, figures_dir / "fig5_synthetic_certification.png")

    if report_path.exists() and not args.overwrite_report:
        print(f"Report exists and was not overwritten (pass --overwrite-report to regenerate): {report_path}")
    else:
        write_full_paper(
            report_path=report_path,
            manifest_df=manifest_df,
            pair_diag_df=pair_diag_df,
            ranking_summary_df=rank_summary_df,
            flip_df=flip_df,
            cross_df=cross_df,
            sim_summary_df=sim_summary_df,
            cert_df=cert_df,
            output_dir_rel=args.output_dir,
        )

    print(f"Runs discovered: {len(runs)}")
    print(f"Pair diagnostics rows: {pair_diag_df.shape[0]}")
    print(f"Synthetic summary rows: {sim_summary_df.shape[0]}")
    print(f"Outputs written to: {output_dir}")
    if report_path.exists():
        print(f"Final paper available at: {report_path}")


if __name__ == "__main__":
    main()
