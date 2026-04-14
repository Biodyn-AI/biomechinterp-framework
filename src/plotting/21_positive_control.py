#!/usr/bin/env python3
"""Build fig_positive_control.png/pdf.

Four-panel figure for Reviewer R1.3 positive control (synthetic pipeline
sensitivity + real-data per-TF TRRUST control). Reads pre-computed results
from:
  results/positive_control/{variant_A,B,C.json, shuffled_null_A,B,C.json, summary.json}
  results/positive_control_real/{per_tf_auroc.json, summary.json}

Panels:
  A) Synthetic pipeline sensitivity: gene-only vs gene+corr AUROC
     for all three variants under planted ground truth + shuffled null,
     with 95% bootstrap CIs on the Delta.
  B) Per-TF AUROC distribution for variant C (realistic regime): strip plot
     planted vs shuffled demonstrating the pipeline catches pair-level signal
     exactly when it exists.
  C) Real-data positive control: mean per-TF TRRUST AUROC across Geneformer
     layers 0/6/13/15/17 with 95% CI, with a dashed chance line.
  D) Per-TF AUROC at the primary analysis layer (L13) for all 9 TRRUST
     direction-known TFs, sorted.

Output: figures/fig_positive_control.{png,pdf}
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 1)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SYNTH_DIR = REPO_ROOT / "results" / "positive_control"
REAL_DIR = REPO_ROOT / "results" / "positive_control_real"

# Okabe-Ito palette (matches palette.py)
C_PLANTED = "#0072B2"   # blue
C_SHUFFLED = "#D55E00"  # vermillion
C_GENE = "#009E73"      # bluish green
C_CORR = "#E69F00"      # orange
C_CHANCE = "#666666"


def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    # ---- load data ----
    variants = {}
    nulls = {}
    for v in ("A", "B", "C"):
        variants[v] = load_json(SYNTH_DIR / f"variant_{v}.json")
        nulls[v] = load_json(SYNTH_DIR / f"shuffled_null_{v}.json")

    real_per_tf = load_json(REAL_DIR / "per_tf_auroc.json")
    real_summary = load_json(REAL_DIR / "summary.json")

    # ---- figure scaffold ----
    fig = plt.figure(figsize=(11.5, 9.0))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32,
                          left=0.08, right=0.97, top=0.93, bottom=0.08)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # ================= Panel A =================
    # Grouped bars: for each variant, show gene-only vs gene+corr
    # under planted and shuffled, with CI on the planted gene+corr.
    variant_labels = ["A: strong edges", "B: weak + indirect", "C: realistic"]
    x = np.arange(3)
    width = 0.2

    gene_planted = [variants[v]["gene_only_auroc"] for v in ("A", "B", "C")]
    corr_planted = [variants[v]["gene_plus_corr_auroc"] for v in ("A", "B", "C")]
    gene_shuff = [nulls[v]["gene_only_auroc"] for v in ("A", "B", "C")]
    corr_shuff = [nulls[v]["gene_plus_corr_auroc"] for v in ("A", "B", "C")]
    # 95% CI on Delta (planted)
    delta_ci = np.array([variants[v]["delta_auroc_ci"] for v in ("A", "B", "C")])
    delta = np.array([variants[v]["delta_auroc_corr"] for v in ("A", "B", "C")])

    axA.bar(x - 1.5 * width, gene_planted, width, color=C_GENE, label="gene-only, planted", edgecolor="black", lw=0.5)
    axA.bar(x - 0.5 * width, corr_planted, width, color=C_CORR, label="gene+corr, planted", edgecolor="black", lw=0.5)
    axA.bar(x + 0.5 * width, gene_shuff, width, color=C_GENE, alpha=0.35, label="gene-only, shuffled", edgecolor="black", lw=0.5, hatch="//")
    axA.bar(x + 1.5 * width, corr_shuff, width, color=C_CORR, alpha=0.35, label="gene+corr, shuffled", edgecolor="black", lw=0.5, hatch="//")
    axA.axhline(0.5, color=C_CHANCE, ls="--", lw=1.0, alpha=0.7)

    # Delta AUROC annotations with CI, below the bars
    for i, v in enumerate(("A", "B", "C")):
        d = delta[i]
        lo, hi = delta_ci[i]
        axA.annotate(
            f"Δ = {d:+.2f}\n[{lo:+.2f}, {hi:+.2f}]",
            xy=(i, 0.05),
            ha="center", va="bottom", fontsize=7.0, color="black",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85),
        )

    axA.set_xticks(x)
    axA.set_xticklabels(variant_labels, fontsize=8)
    axA.set_ylabel("Perturbation-first AUROC")
    axA.set_ylim(0.0, 1.30)
    axA.set_yticks(np.arange(0.0, 1.01, 0.2))
    axA.set_title("A  Synthetic positive control: pipeline sensitivity",
                  loc="left", fontsize=10, fontweight="bold")
    axA.legend(fontsize=6.5, loc="upper center", ncol=4, frameon=False,
               columnspacing=0.8, handletextpad=0.3,
               bbox_to_anchor=(0.5, 1.00))
    axA.spines[["top", "right"]].set_visible(False)

    # ================= Panel B =================
    # Per-TF AUROC strip plot for variant C: planted vs shuffled
    v = "C"
    planted_vals = sorted(variants[v]["per_tf_corr_auroc"].values())
    shuffled_vals = sorted(nulls[v]["per_tf_corr_auroc"].values())

    rng = np.random.default_rng(42)
    jitter_p = rng.uniform(-0.12, 0.12, size=len(planted_vals))
    jitter_s = rng.uniform(-0.12, 0.12, size=len(shuffled_vals))
    axB.scatter(np.full(len(planted_vals), 0) + jitter_p, planted_vals,
                color=C_PLANTED, s=44, alpha=0.85, edgecolor="black", lw=0.4, label="planted")
    axB.scatter(np.full(len(shuffled_vals), 1) + jitter_s, shuffled_vals,
                color=C_SHUFFLED, s=44, alpha=0.85, edgecolor="black", lw=0.4, label="shuffled")

    # means
    axB.hlines(np.mean(planted_vals), -0.25, 0.25, color="black", lw=2.0)
    axB.hlines(np.mean(shuffled_vals), 0.75, 1.25, color="black", lw=2.0)
    axB.axhline(0.5, color=C_CHANCE, ls="--", lw=1.0, alpha=0.7)

    axB.set_xticks([0, 1])
    axB.set_xticklabels(["planted\nground truth", "shuffled\nnull"], fontsize=9)
    axB.set_ylabel("Per-TF correlation-based AUROC")
    axB.set_ylim(0.20, 1.05)
    axB.set_xlim(-0.55, 1.55)
    axB.set_title(f"B  Per-TF AUROC, variant C realistic regime (n={len(planted_vals)} TFs)",
                  loc="left", fontsize=10, fontweight="bold")
    axB.spines[["top", "right"]].set_visible(False)
    axB.legend(fontsize=8, loc="lower right", frameon=False)

    # ================= Panel C =================
    # Real-data: mean per-TF TRRUST AUROC across layers 0, 6, 13, 15, 17
    by_layer = real_summary["by_layer"]
    layers = sorted([int(k) for k in by_layer.keys()])
    means = [by_layer[str(L)]["mean_auroc"] for L in layers]
    ci_lo = [by_layer[str(L)]["auroc_ci_95"][0] for L in layers]
    ci_hi = [by_layer[str(L)]["auroc_ci_95"][1] for L in layers]
    frac_above = [by_layer[str(L)]["frac_above_chance"] for L in layers]

    x_L = np.arange(len(layers))
    err_lo = np.array(means) - np.array(ci_lo)
    err_hi = np.array(ci_hi) - np.array(means)
    axC.errorbar(x_L, means, yerr=[err_lo, err_hi],
                 fmt="o-", color=C_PLANTED, ms=8, lw=1.6,
                 ecolor="black", elinewidth=1.1, capsize=4,
                 mfc=C_PLANTED, mec="black", mew=0.6, label="mean per-TF AUROC")
    axC.axhline(0.5, color=C_CHANCE, ls="--", lw=1.0, alpha=0.7, label="chance (0.5)")

    for i, L in enumerate(layers):
        axC.annotate(f"{int(frac_above[i]*9)}/9",
                     xy=(x_L[i], ci_hi[i] + 0.02),
                     ha="center", fontsize=7.5, color="black")

    axC.set_xticks(x_L)
    axC.set_xticklabels([f"L{L}" for L in layers])
    axC.set_xlabel("Geneformer V2-316M layer")
    axC.set_ylabel("Mean per-TF TRRUST AUROC (95% CI)")
    axC.set_ylim(0.40, 0.85)
    axC.set_title(f"C  Real-data positive control, {real_summary['n_trrust_entries_in_hvg']} TRRUST direction-known targets",
                  loc="left", fontsize=10, fontweight="bold")
    axC.spines[["top", "right"]].set_visible(False)
    axC.legend(fontsize=8, loc="upper left", frameon=False)

    # ================= Panel D =================
    # Per-TF AUROC at L13, sorted
    L13 = real_per_tf["layers"]["13"]["per_tf"]
    tfs_sorted = sorted(L13.items(), key=lambda kv: kv[1]["auroc"])
    tf_names = [t for t, _ in tfs_sorted]
    tf_aurocs = [d["auroc"] for _, d in tfs_sorted]
    tf_n_targets = [d["n_targets"] for _, d in tfs_sorted]

    y_pos = np.arange(len(tf_names))
    colors_bar = [C_PLANTED if a > 0.5 else C_SHUFFLED for a in tf_aurocs]
    axD.barh(y_pos, tf_aurocs, color=colors_bar, edgecolor="black", lw=0.5, alpha=0.9)
    axD.axvline(0.5, color=C_CHANCE, ls="--", lw=1.0, alpha=0.7)

    for i, (a, n) in enumerate(zip(tf_aurocs, tf_n_targets)):
        axD.annotate(f"n={n}", xy=(a + 0.015, y_pos[i]),
                     va="center", fontsize=7.0, color="black")

    axD.set_yticks(y_pos)
    axD.set_yticklabels(tf_names, fontsize=8)
    axD.set_xlabel("Per-TF TRRUST AUROC at Layer 13")
    axD.set_xlim(0.40, 1.00)
    axD.set_title("D  Per-TF AUROC at primary analysis layer (L13)",
                  loc="left", fontsize=10, fontweight="bold")
    axD.spines[["top", "right"]].set_visible(False)

    # Save
    out_png = FIG_DIR / "fig_positive_control.png"
    out_pdf = FIG_DIR / "fig_positive_control.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
