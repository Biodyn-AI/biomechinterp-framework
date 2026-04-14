#!/usr/bin/env python3
"""
Real-data positive control: per-TF Geneformer attention edge AUROC against
TRRUST direct-target sets.

Motivation (reviewer R1.3): we need a real-data complement to the synthetic
positive control (20_positive_control_pipeline.py) showing that, for TFs
where the attention pipeline *should* recover signal (high-degree, well-
characterised master regulators in TRRUST Activation/Repression entries),
the per-TF AUROC is meaningfully above chance. This complements the
exploratory per-TF analysis in Supp Note 19 by using a stricter "direct
target" definition (TRRUST direction-known entries only) and by reporting
results for both early and late Geneformer layers so the layer-specific
PPI/regulation hierarchy from Supp Note 17 is verified.

Inputs (cached from prior runs, no model re-extraction needed):
- results/round9_unified_extraction/layer_mean_edges/layer_{NN}_edges.pkl
  dict {(src_gene, tgt_gene): float} of per-edge attention scores
- results/round9_unified_extraction/hvg_gene_list.json
  list of 2000 HVG gene symbols (the universe used during extraction)
- src/02_cssi_method/cssi_real_data/results/trrust_human.tsv
  TRRUST human TF-target reference

Output:
- results/positive_control_real/per_tf_auroc.json
- results/positive_control_real/summary.json

Usage:
    python 20_positive_control_tfs.py [--layers 0,6,13,15,17] [--min-targets 8]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 1)

REPO_ROOT = Path(__file__).resolve().parents[3]
EDGES_DIR = REPO_ROOT / "results" / "round9_unified_extraction" / "layer_mean_edges"
HVG_PATH = REPO_ROOT / "results" / "round9_unified_extraction" / "hvg_gene_list.json"
TRRUST_PATH = REPO_ROOT / "src" / "02_cssi_method" / "cssi_real_data" / "results" / "trrust_human.tsv"
OUT_DIR = REPO_ROOT / "results" / "positive_control_real"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_hvg() -> list[str]:
    with open(HVG_PATH) as f:
        return json.load(f)


def load_trrust(direction_known_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(
        TRRUST_PATH, sep="\t", header=None,
        names=["tf", "target", "mode", "pmid"],
    )
    if direction_known_only:
        df = df[df["mode"].isin(["Activation", "Repression"])].copy()
    return df.reset_index(drop=True)


def load_layer_edges(layer: int) -> dict:
    path = EDGES_DIR / f"layer_{layer:02d}_edges.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def per_tf_auroc(edges: dict, hvg: list[str], trrust: pd.DataFrame, min_targets: int) -> dict:
    """For each TF in TRRUST with >=min_targets evaluable targets in HVG,
    compute per-TF AUROC of attention edge magnitude discriminating direct
    targets from non-target HVG genes.
    """
    hvg_set = set(hvg)
    # Build TF -> {target} dict from TRRUST, restricted to HVG genes
    tf_to_targets = {}
    for _, row in trrust.iterrows():
        tf, tgt = row["tf"], row["target"]
        if tf not in hvg_set or tgt not in hvg_set:
            continue
        tf_to_targets.setdefault(tf, set()).add(tgt)
    # Filter to TFs with enough targets
    tf_to_targets = {
        tf: tgts for tf, tgts in tf_to_targets.items() if len(tgts) >= min_targets
    }
    print(f"  {len(tf_to_targets)} TFs with >= {min_targets} HVG targets")
    if not tf_to_targets:
        return {}

    results = {}
    for tf, true_targets in tf_to_targets.items():
        # Score every HVG gene g (g != tf) by attention edge tf -> g
        scores = []
        labels = []
        for g in hvg:
            if g == tf:
                continue
            edge_key = (tf, g)
            edge_val = edges.get(edge_key, np.nan)
            if not np.isfinite(edge_val):
                continue
            scores.append(float(abs(edge_val)))
            labels.append(int(g in true_targets))
        if not scores or sum(labels) < 3 or sum(labels) == len(labels):
            continue
        try:
            aur = roc_auc_score(labels, scores)
        except ValueError:
            continue
        results[tf] = {
            "auroc": float(aur),
            "n_targets": int(sum(labels)),
            "n_evaluable": int(len(labels)),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layers",
        default="0,6,13,15,17",
        help="Comma-separated layer indices to evaluate",
    )
    parser.add_argument(
        "--min-targets",
        type=int,
        default=8,
        help="Minimum HVG-evaluable targets per TF for inclusion",
    )
    args = parser.parse_args()

    hvg = load_hvg()
    print(f"HVG gene universe: {len(hvg)} genes")

    trrust = load_trrust(direction_known_only=True)
    print(f"TRRUST direction-known entries: {len(trrust)}")
    n_in_hvg = trrust[trrust.tf.isin(hvg) & trrust.target.isin(hvg)]
    print(f"TRRUST entries with both TF and target in HVG: {len(n_in_hvg)}")

    layers = [int(l) for l in args.layers.split(",")]
    summary = {"layers": {}, "min_targets": args.min_targets}

    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        try:
            edges = load_layer_edges(layer)
        except FileNotFoundError:
            print(f"  no cached edges for layer {layer}, skipping")
            continue
        per_tf = per_tf_auroc(edges, hvg, trrust, min_targets=args.min_targets)
        if not per_tf:
            print("  no qualifying TFs")
            continue

        aurocs = np.array([v["auroc"] for v in per_tf.values()])
        n_above_chance = int((aurocs > 0.5).sum())
        n_total = len(aurocs)
        # Bootstrap 95% CI on mean per-TF AUROC
        rng = np.random.default_rng(2026)
        boot_means = [
            rng.choice(aurocs, size=len(aurocs), replace=True).mean() for _ in range(2000)
        ]
        ci_low, ci_high = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

        layer_result = {
            "layer": layer,
            "n_tfs": n_total,
            "mean_auroc": float(aurocs.mean()),
            "median_auroc": float(np.median(aurocs)),
            "auroc_ci_95": [ci_low, ci_high],
            "n_above_chance": n_above_chance,
            "frac_above_chance": float(n_above_chance / n_total),
            "max_auroc_tf": max(per_tf.items(), key=lambda x: x[1]["auroc"])[0],
            "max_auroc_value": float(aurocs.max()),
            "per_tf": per_tf,
        }
        summary["layers"][str(layer)] = layer_result
        print(f"  TFs evaluated:        {n_total}")
        print(f"  mean per-TF AUROC:    {aurocs.mean():.4f} [95% CI {ci_low:.3f}, {ci_high:.3f}]")
        print(f"  median per-TF AUROC:  {np.median(aurocs):.4f}")
        print(f"  TFs above chance:     {n_above_chance}/{n_total} ({n_above_chance/n_total:.1%})")
        print(f"  best TF:              {layer_result['max_auroc_tf']} (AUROC = {aurocs.max():.4f})")

    # Save full per-TF results
    with open(OUT_DIR / "per_tf_auroc.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    # Save compact summary
    compact = {
        "min_targets": args.min_targets,
        "n_trrust_entries_in_hvg": int(len(n_in_hvg)),
        "by_layer": {
            str(l): {
                k: v for k, v in summary["layers"][str(l)].items() if k != "per_tf"
            }
            for l in layers if str(l) in summary["layers"]
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(compact, f, indent=2, default=float)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
