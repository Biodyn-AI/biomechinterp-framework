#!/usr/bin/env python3
"""Build the new Figure 1: framework schematic (two objectives, five test families).

Reviewer R1.m2 and R2.3 asked for a clear schematic that separates the two
evaluation objectives and shows what each test family can and cannot conclude.
This is a vector-style figure built directly with matplotlib patches; no
underlying data needed.

Output: figures/fig_framework_schematic.png and .pdf
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 1)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Okabe-Ito colorblind-safe palette (R3.3)
C_OBJ_A = "#0072B2"   # blue   — Objective A (GRN recovery)
C_OBJ_B = "#D55E00"   # vermillion — Objective B (perturbation prediction)
C_BOTH  = "#009E73"   # bluish green — applies to both objectives
C_BG    = "#F7F7F7"   # light grey background
C_BORDER = "#222222"
C_TEXT  = "#222222"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.linewidth": 0.0,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

fig = plt.figure(figsize=(12, 9))
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.set_axis_off()


def rounded_box(x, y, w, h, color, edge=C_BORDER, lw=1.2, alpha=1.0, zorder=2):
    return mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        linewidth=lw, edgecolor=edge, facecolor=color, alpha=alpha, zorder=zorder,
    )


def text(x, y, s, fontsize=11, weight="normal", ha="center", va="center",
         color=C_TEXT, zorder=5):
    ax.text(x, y, s, fontsize=fontsize, fontweight=weight, ha=ha, va=va,
            color=color, zorder=zorder)


# Top title
text(6.0, 8.6,
     "Two-objective evaluation framework for single-cell foundation model interpretability",
     fontsize=14, weight="bold")

# ---- Row 1: Two objectives ----
# Objective A (left)
ax.add_patch(rounded_box(0.4, 6.5, 5.4, 1.6, "#E8F0F7", edge=C_OBJ_A, lw=2.0))
text(3.1, 7.8, "Objective A — GRN recovery", fontsize=12, weight="bold", color=C_OBJ_A)
text(3.1, 7.35, "Edge-identification task", fontsize=10, color=C_OBJ_A)
text(3.1, 6.95,
     "Do attention edges identify curated\nregulatory interactions?",
     fontsize=10, color=C_TEXT)
text(3.1, 6.62,
     "Evaluation: AUROC vs. TRRUST/DoRothEA, top-K F1, CSSI",
     fontsize=9, color="#444444")

# Objective B (right)
ax.add_patch(rounded_box(6.2, 6.5, 5.4, 1.6, "#FBEBE0", edge=C_OBJ_B, lw=2.0))
text(8.9, 7.8, "Objective B — Perturbation-target prediction",
     fontsize=12, weight="bold", color=C_OBJ_B)
text(8.9, 7.35, "Effect-magnitude task", fontsize=10, color=C_OBJ_B)
text(8.9, 6.95,
     "Do attention edges predict which genes\nshow DE after a CRISPR perturbation?",
     fontsize=10, color=C_TEXT)
text(8.9, 6.62,
     "Evaluation: per-perturbation AUROC, incremental value",
     fontsize=9, color="#444444")

# Note between objectives
text(6.0, 6.25,
     "Mathematically distinct: an edge can be a true direct regulator yet add little to Objective B (e.g. buffered),\n"
     "and a non-regulatory feature can score on Objective B if it proxies target susceptibility.",
     fontsize=8.5, color="#666666", weight="normal")

# ---- Row 2: Five test families ----
fam_y = 4.1
fam_h = 1.8
fam_w = 2.05
fam_xs = [0.4, 2.65, 4.9, 7.15, 9.4]
families = [
    ("(i) Trivial baseline", "B",
     "variance, mean,\ndropout — no model"),
    ("(ii) Incremental value", "B",
     "edges + gene\nfeatures, hard CV"),
    ("(iii) Residualisation\n   & propensity match", "Both",
     "remove gene-level\nconfounds"),
    ("(iv) Causal ablation", "B",
     "head mask, MLP\nzero, fidelity diag."),
    ("(v) Cross-context", "Both",
     "K562, RPE1, T cells,\nCRISPRi/CRISPRa"),
]
for x, (title, scope, desc) in zip(fam_xs, families):
    if scope == "B":
        scope_color = C_OBJ_B
        bg = "#FFF6F0"
    elif scope == "Both":
        scope_color = C_BOTH
        bg = "#EBF7F2"
    else:
        scope_color = C_OBJ_A
        bg = "#E8F0F7"
    ax.add_patch(rounded_box(x, fam_y, fam_w, fam_h, bg, edge=scope_color, lw=1.5))
    text(x + fam_w / 2, fam_y + fam_h - 0.32, title, fontsize=9.5,
         weight="bold", color=scope_color)
    text(x + fam_w / 2, fam_y + fam_h - 0.85, desc, fontsize=8.5, color=C_TEXT)
    # scope label
    text(x + fam_w / 2, fam_y + 0.28,
         f"→ Objective {scope}" if scope != "Both" else "→ Both objectives",
         fontsize=8.5, color=scope_color, weight="bold")

# Connecting arrows from objectives to families (light)
for x in fam_xs:
    cx = x + fam_w / 2
    # Determine which objective(s) to point from
    family_idx = fam_xs.index(x)
    scope = families[family_idx][1]
    if scope in ("B",):
        ax.annotate("", xy=(cx, fam_y + fam_h + 0.05), xytext=(8.9, 6.5),
                    arrowprops=dict(arrowstyle="-", color=C_OBJ_B, lw=0.8, alpha=0.4),
                    zorder=1)
    elif scope == "Both":
        ax.annotate("", xy=(cx, fam_y + fam_h + 0.05), xytext=(3.1, 6.5),
                    arrowprops=dict(arrowstyle="-", color=C_BOTH, lw=0.8, alpha=0.4),
                    zorder=1)
        ax.annotate("", xy=(cx, fam_y + fam_h + 0.05), xytext=(8.9, 6.5),
                    arrowprops=dict(arrowstyle="-", color=C_BOTH, lw=0.8, alpha=0.4),
                    zorder=1)

# ---- Row 3: Positive control + decision matrix ----
# Positive control box (left half)
ax.add_patch(rounded_box(0.4, 1.8, 5.4, 1.6, "#FFFCEE", edge="#A88600", lw=1.5))
text(3.1, 3.05, "Positive control: pipeline sensitivity",
     fontsize=11, weight="bold", color="#7A6300")
text(3.1, 2.55,
     "Synthetic Perturb-seq with planted ground truth +\n"
     "real-data per-TF AUROC against TRRUST direct targets",
     fontsize=9, color=C_TEXT)
text(3.1, 2.05,
     "Confirms: pipeline detects pairwise signal when it exists",
     fontsize=8.5, color="#666666", weight="bold")

# Decision matrix (right half)
ax.add_patch(rounded_box(6.2, 1.8, 5.4, 1.6, "#F0F0F0", edge=C_BORDER, lw=1.0))
text(8.9, 3.05, "What each family can conclude",
     fontsize=11, weight="bold", color=C_TEXT)
text(8.9, 2.55,
     "(i)–(iv) → no incremental Obj-B value beyond gene features\n"
     "(iii)+CSSI → Obj-A signal is recoverable, layer-specific",
     fontsize=9, color=C_TEXT)
text(8.9, 2.05,
     "(v) → context-dependent on raw scores, but Obj-B null is universal",
     fontsize=8.5, color="#666666", weight="bold")

# ---- Bottom: outputs / publishable claims ----
ax.add_patch(rounded_box(0.4, 0.15, 11.2, 1.4, "#EFEFEF", edge=C_BORDER, lw=1.0))
text(6.0, 1.2,
     "Convergent two-part conclusion",
     fontsize=12, weight="bold")
text(3.5, 0.75,
     "Cautionary (Objective B):\nattention edges add no incremental value\nover gene-level features for predicting DE",
     fontsize=9.5, color=C_OBJ_B, weight="normal")
text(8.5, 0.75,
     "Constructive (Objective A):\nlayer-specific regulatory structure exists;\nCSSI recovers it, up to 1.85× over baseline",
     fontsize=9.5, color=C_OBJ_A, weight="normal")

out_png = FIG_DIR / "fig_framework_schematic.png"
out_pdf = FIG_DIR / "fig_framework_schematic.pdf"
plt.savefig(out_png, dpi=200, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close(fig)
