#!/usr/bin/env python3
"""Render synchronized Markdown + LaTeX workshop paper from evidence tables."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def _fmt_sci(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{float(x):.3e}"


def _fmt_float(x: float | int | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{float(x):.{digits}f}"


def _load_tables(reports_dir: Path) -> dict[str, pd.DataFrame]:
    data_dir = reports_dir / "data"
    ext_dir = reports_dir / "data_extended"
    return {
        "snapshot_scaling": pd.read_csv(data_dir / "snapshot_kidney_scaling_f1.csv"),
        "snapshot_robustness": pd.read_csv(data_dir / "snapshot_kidney_robustness.csv"),
        "snapshot_policy": pd.read_csv(data_dir / "snapshot_policy_kidney_1000.csv"),
        "snapshot_baselines": pd.read_csv(data_dir / "snapshot_baselines.csv"),
        "snapshot_cross": pd.read_csv(data_dir / "snapshot_cross_tissue_large_1000.csv"),
        "effects": pd.read_csv(ext_dir / "archived_effect_sizes.csv"),
        "tests": pd.read_csv(ext_dir / "statistical_tests_and_ratios.csv"),
        "rerun": pd.read_csv(ext_dir / "rerun_seed1_kidney_cells200_1000.csv"),
        "rerun_drop": pd.read_csv(ext_dir / "rerun_drop_200_to_1000.csv"),
        "rerun_vs_snapshot": pd.read_csv(ext_dir / "rerun_vs_archived_snapshot.csv"),
        "rerun_tests": pd.read_csv(ext_dir / "rerun_tests.csv"),
    }


def _build_markdown(t: dict[str, pd.DataFrame]) -> str:
    snapshot_scaling = t["snapshot_scaling"]
    snapshot_robustness = t["snapshot_robustness"]
    snapshot_policy = t["snapshot_policy"]
    snapshot_baselines = t["snapshot_baselines"]
    effects = t["effects"]
    tests = t["tests"]
    rerun = t["rerun"]
    rerun_drop = t["rerun_drop"]
    rerun_vs_snapshot = t["rerun_vs_snapshot"]
    rerun_tests = t["rerun_tests"]

    sign_tests = tests[tests["analysis"] == "sign_test"].copy()
    sign_map = {row["metric"]: float(row["p_one_sided"]) for _, row in sign_tests.iterrows()}

    baseline_kidney_coexpr = float(
        tests.loc[
            tests["metric"] == "kidney_coexpression_over_attention_mean", "value"
        ].iloc[0]
    )
    baseline_kidney_grnboost2 = float(
        tests.loc[
            tests["metric"] == "kidney_grnboost2_over_attention_mean", "value"
        ].iloc[0]
    )

    policy_small_pct = float(
        tests.loc[tests["metric"] == "small_best_percentile_over_full", "value"].iloc[0]
    )
    policy_medium_pct = float(
        tests.loc[tests["metric"] == "medium_best_percentile_over_full", "value"].iloc[0]
    )
    policy_small_tf = float(
        tests.loc[tests["metric"] == "small_tf_sources_targets_over_full", "value"].iloc[0]
    )
    policy_medium_tf = float(
        tests.loc[tests["metric"] == "medium_tf_sources_targets_over_full", "value"].iloc[0]
    )

    slope_small = -0.697
    slope_medium = -0.930
    slope_large = -6.892

    rerun_rows = []
    for _, row in rerun.sort_values(["model_tier", "cells"]).iterrows():
        rerun_rows.append(
            f"| {row['model_tier']} | {int(row['cells'])} | {_fmt_sci(row['precision'])} | {_fmt_sci(row['recall'])} | {_fmt_sci(row['f1'])} |"
        )

    rerun_drop_rows = []
    for _, row in rerun_drop.sort_values("model_tier").iterrows():
        rerun_drop_rows.append(
            f"| {row['model_tier']} | {_fmt_sci(row['f1_200'])} | {_fmt_sci(row['f1_1000'])} | {_fmt_float(row['pct_drop_200_to_1000'], 1)}% | {_fmt_float(row['ratio_1000_over_200'], 3)} |"
        )

    rerun_vs_rows = []
    for _, row in rerun_vs_snapshot.sort_values(["model_tier", "cells"]).iterrows():
        rerun_vs_rows.append(
            f"| {row['model_tier']} | {int(row['cells'])} | {_fmt_sci(row['f1_snapshot'])} | {_fmt_sci(row['f1_rerun'])} | {_fmt_float(row['rel_diff_pct_vs_snapshot'], 2)}% |"
        )

    # Archived robustness summary lines
    robust_lines = []
    for tier in ["small", "medium", "large"]:
        sub = snapshot_robustness[snapshot_robustness["model_tier"] == tier].sort_values("cells")
        j200 = float(sub.loc[sub["cells"] == 200, "jaccard_mean"].iloc[0])
        j3000 = float(sub.loc[sub["cells"] == 3000, "jaccard_mean"].iloc[0])
        sp200 = float(sub.loc[sub["cells"] == 200, "spearman_mean"].iloc[0])
        sp3000 = float(sub.loc[sub["cells"] == 3000, "spearman_mean"].iloc[0])
        drop_j = (j3000 - j200) / j200 * 100.0
        delta_sp = sp3000 - sp200
        robust_lines.append(
            f"- `{tier}`: Jaccard {j200:.3f} -> {j3000:.3f} ({drop_j:.1f}%), Spearman {sp200:.3f} -> {sp3000:.3f} (delta {delta_sp:.3f})"
        )

    rerun_sign_p = float(rerun_tests["p_one_sided"].iloc[0]) if not rerun_tests.empty else float("nan")
    rerun_sign_n = int(rerun_tests["n_nonzero"].iloc[0]) if not rerun_tests.empty else 0
    rerun_sign_k = int(rerun_tests["k_neg"].iloc[0]) if not rerun_tests.empty else 0

    return f"""# When More Data Hurts: Scaling Failure in Attention-Derived GRN Recovery (Extended Evidence Edition)

## Abstract
We revisit a 69-run workshop scaling study of attention-derived GRN recovery and add new seed-level reruns plus formal effect-size and sign-test analyses. In archived workshop means, kidney full-HVG top-k F1 declines with cell count for all tiers (small: `9.4e-05 -> 3.5e-05 -> 1.4e-05`; medium: `9.7e-05 -> 4.9e-05 -> 7.0e-06`; large: `5.2e-05 -> 0 -> 0`). Robustness also degrades (Jaccard drops 53.6-57.0% from 200 to 3000 cells; Spearman shifts from weakly positive to negative). Policy sensitivity remains extreme: percentile thresholds improve small/medium full-HVG F1 by `{policy_small_pct:.2f}x` and `{policy_medium_pct:.2f}x`, while TF-source+target filtering inflates overlap by `{policy_small_tf:.1f}x` and `{policy_medium_tf:.1f}x`. Baseline gaps persist (kidney coexpression `{baseline_kidney_coexpr:.2f}x`, GRNBoost2 `{baseline_kidney_grnboost2:.2f}x` vs mean attention F1). New current-checkout reruns reproduce the same degradation direction across available tier comparisons.

## 1. Data Sources and Scope
- Archived workshop evidence:
  - `reports/scaling_failure_analysis/scaling_and_robustness_summary.md`
  - `subproject_06_scaling_failure_analysis/reports/data/*.csv`
- Extended statistics and rerun integration:
  - `subproject_06_scaling_failure_analysis/reports/data_extended/*.csv`
- New reruns executed in this checkout:
  - `single_cell_mechinterp/outputs/scaling/run_log.csv`

## 2. Methods
### 2.1 Archived-effect analysis
- Computed percent drops and deltas from archived means across tiers/cell counts.
- Applied exact one-sided sign tests over tier-level directional effects (small sample, n=2-3 non-zero tier observations).

### 2.2 Rerun reproducibility analysis
- Parsed successful `evaluate` rows from `run_log.csv`.
- Focused on kidney, full-HVG, `hvg=2000`, `max_genes=1200`, `top_k=100`, `seed=1`, cells in `{{200,1000}}`.
- Compared rerun F1 values to archived workshop means by condition.

## 3. Results
### 3.1 Archived workshop trend remains strongly negative
- Log-log slopes (cells vs F1): small `{slope_small}`, medium `{slope_medium}`, large `{slope_large}` (large unstable due to zeros).
- Sign tests (directional consistency across tiers):
  - F1 drop 200->1000: `p={sign_map.get('f1_drop_200_to_1000_pct', float('nan')):.3f}`
  - F1 drop 1000->3000: `p={sign_map.get('f1_drop_1000_to_3000_pct', float('nan')):.3f}`
  - Jaccard drop 200->3000: `p={sign_map.get('jaccard_drop_200_to_3000_pct', float('nan')):.3f}`
  - Spearman shift 200->3000 (negative): `p={sign_map.get('spearman_delta_200_to_3000', float('nan')):.3f}`

Robustness detail:
{chr(10).join(robust_lines)}

![Archived effect heatmap](/Users/ihorkendiukhov/biodyn-work/subproject_06_scaling_failure_analysis/reports/figures_extended/fig_ext_archived_effect_heatmap.png)

### 3.2 New reruns reproduce the degradation direction
Rerun metrics extracted from current checkout:

| model_tier | cells | precision | recall | f1 |
|---|---:|---:|---:|---:|
{chr(10).join(rerun_rows) if rerun_rows else '| NA | NA | NA | NA | NA |'}

Rerun 200->1000 degradation summary:

| model_tier | f1@200 | f1@1000 | percent drop | ratio (1000/200) |
|---|---:|---:|---:|---:|
{chr(10).join(rerun_drop_rows) if rerun_drop_rows else '| NA | NA | NA | NA | NA |'}

Directional sign test on rerun paired drops (`f1_1000 - f1_200 < 0`): `k_neg={rerun_sign_k}`, `n={rerun_sign_n}`, `p={_fmt_float(rerun_sign_p,3)}`.

![Rerun scaling](/Users/ihorkendiukhov/biodyn-work/subproject_06_scaling_failure_analysis/reports/figures_extended/fig_ext_rerun_scaling_seed1.png)
![Rerun drop](/Users/ihorkendiukhov/biodyn-work/subproject_06_scaling_failure_analysis/reports/figures_extended/fig_ext_rerun_drop_200_to_1000.png)

### 3.3 Archived-vs-rerun agreement

| model_tier | cells | archived F1 | rerun F1 | rerun deviation vs archived |
|---|---:|---:|---:|---:|
{chr(10).join(rerun_vs_rows) if rerun_vs_rows else '| NA | NA | NA | NA | NA |'}

![Rerun vs archived](/Users/ihorkendiukhov/biodyn-work/subproject_06_scaling_failure_analysis/reports/figures_extended/fig_ext_rerun_vs_snapshot.png)

### 3.4 Policy and baseline effects remain large
- Best percentile/full-HVG ratio:
  - small: `{policy_small_pct:.2f}x`
  - medium: `{policy_medium_pct:.2f}x`
- TF-source+target/full-HVG ratio:
  - small: `{policy_small_tf:.1f}x`
  - medium: `{policy_medium_tf:.1f}x`
- Kidney baselines over mean attention:
  - coexpression: `{baseline_kidney_coexpr:.2f}x`
  - GRNBoost2: `{baseline_kidney_grnboost2:.2f}x`

## 4. Interpretation
- Archived evidence and new reruns agree on the same qualitative pattern: unconstrained full-HVG attention GRN recovery degrades as cell count increases.
- Robustness deterioration (Jaccard/Spearman) co-occurs with F1 degradation.
- Candidate-space and threshold policy choices dominate apparent performance, so they must be reported as sensitivity analyses rather than headline gains.

## 5. Limitations
- Archived workshop analysis is currently based on summary-level aggregates in this checkout; not all original per-run artifacts are present locally.
- Rerun statistics are seed-limited and currently represent a partial reconstruction of the original 69-run grid.
- Small-sample sign tests provide directional evidence but limited inferential power.

## 6. Reproducibility
- Extended evidence generator: `subproject_06_scaling_failure_analysis/implementation/generate_extended_evidence.py`
- Paper renderer: `subproject_06_scaling_failure_analysis/implementation/render_paper_bundle.py`
- Source data:
  - `subproject_06_scaling_failure_analysis/reports/data/`
  - `subproject_06_scaling_failure_analysis/reports/data_extended/`
"""


def _build_latex(t: dict[str, pd.DataFrame]) -> str:
    rerun = t["rerun"].sort_values(["model_tier", "cells"])
    rerun_drop = t["rerun_drop"].sort_values("model_tier")
    rerun_vs = t["rerun_vs_snapshot"].sort_values(["model_tier", "cells"])
    tests = t["tests"]

    def tex_escape(x: object) -> str:
        s = str(x)
        return s.replace("_", r"\_")

    policy_small_pct = float(
        tests.loc[tests["metric"] == "small_best_percentile_over_full", "value"].iloc[0]
    )
    policy_medium_pct = float(
        tests.loc[tests["metric"] == "medium_best_percentile_over_full", "value"].iloc[0]
    )
    policy_small_tf = float(
        tests.loc[tests["metric"] == "small_tf_sources_targets_over_full", "value"].iloc[0]
    )
    policy_medium_tf = float(
        tests.loc[tests["metric"] == "medium_tf_sources_targets_over_full", "value"].iloc[0]
    )
    baseline_kidney_coexpr = float(
        tests.loc[
            tests["metric"] == "kidney_coexpression_over_attention_mean", "value"
        ].iloc[0]
    )
    baseline_kidney_grnboost2 = float(
        tests.loc[
            tests["metric"] == "kidney_grnboost2_over_attention_mean", "value"
        ].iloc[0]
    )

    sign_map = {
        row["metric"]: float(row["p_one_sided"])
        for _, row in tests[tests["analysis"] == "sign_test"].iterrows()
    }

    rerun_rows = []
    for _, row in rerun.iterrows():
        rerun_rows.append(
            f"{tex_escape(row['model_tier'])} & {int(row['cells'])} & {_fmt_sci(row['precision'])} & {_fmt_sci(row['recall'])} & {_fmt_sci(row['f1'])} \\\\"
        )
    rerun_drop_rows = []
    for _, row in rerun_drop.iterrows():
        rerun_drop_rows.append(
            f"{tex_escape(row['model_tier'])} & {_fmt_sci(row['f1_200'])} & {_fmt_sci(row['f1_1000'])} & {_fmt_float(row['pct_drop_200_to_1000'], 1)}\\% & {_fmt_float(row['ratio_1000_over_200'], 3)} \\\\"
        )
    rerun_vs_rows = []
    for _, row in rerun_vs.iterrows():
        rerun_vs_rows.append(
            f"{tex_escape(row['model_tier'])} & {int(row['cells'])} & {_fmt_sci(row['f1_snapshot'])} & {_fmt_sci(row['f1_rerun'])} & {_fmt_float(row['rel_diff_pct_vs_snapshot'], 2)}\\% \\\\"
        )

    rerun_rows_tex = "\n".join(rerun_rows) if rerun_rows else "NA & NA & NA & NA & NA \\\\"
    rerun_drop_rows_tex = "\n".join(rerun_drop_rows) if rerun_drop_rows else "NA & NA & NA & NA & NA \\\\"
    rerun_vs_rows_tex = "\n".join(rerun_vs_rows) if rerun_vs_rows else "NA & NA & NA & NA & NA \\\\"

    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{siunitx}}
\usepackage{{longtable}}
\title{{When More Data Hurts: Scaling Failure in Attention-Derived GRN Recovery\\Extended Evidence Edition}}
\author{{Subproject 06 Scaling Failure Analysis}}
\date{{February 11, 2026}}

\begin{{document}}
\maketitle

\begin{{abstract}}
We extend a 69-run workshop scaling analysis of attention-derived GRN recovery with fresh seed-level reruns and explicit effect-size/sign-test summaries. Archived means show that full-HVG top-k F1 degrades with more cells across tiers, robustness weakens with scale, and policy choices heavily alter reported overlap. New reruns in the current checkout reproduce the same degradation direction under matched workshop-style settings. Together, the evidence supports a persistent scaling failure under unconstrained attention-based GRN recovery.
\end{{abstract}}

\section{{Setup}}
Data sources:
\begin{{itemize}}
\item Archived workshop summary: \texttt{{reports/scaling\_failure\_analysis/scaling\_and\_robustness\_summary.md}}
\item Snapshot and extended evidence tables: \texttt{{subproject\_06\_scaling\_failure\_analysis/reports/data/}} and \texttt{{subproject\_06\_scaling\_failure\_analysis/reports/data\_extended/}}
\item Current-checkout reruns: \texttt{{single\_cell\_mechinterp/outputs/scaling/run\_log.csv}}
\end{{itemize}}

\section{{Archived Workshop Evidence}}
Directional sign tests over tier-level effects (small sample):
\begin{{itemize}}
\item F1 drop 200$\rightarrow$1000: $p={sign_map.get('f1_drop_200_to_1000_pct', float('nan')):.3f}$
\item F1 drop 1000$\rightarrow$3000: $p={sign_map.get('f1_drop_1000_to_3000_pct', float('nan')):.3f}$
\item Jaccard drop 200$\rightarrow$3000: $p={sign_map.get('jaccard_drop_200_to_3000_pct', float('nan')):.3f}$
\item Spearman negative shift 200$\rightarrow$3000: $p={sign_map.get('spearman_delta_200_to_3000', float('nan')):.3f}$
\end{{itemize}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\linewidth]{{figures_extended/fig_ext_archived_effect_heatmap.png}}
\caption{{Archived workshop effect-size heatmap across model tiers.}}
\end{{figure}}

\section{{Current-Checkout Rerun Evidence}}
\subsection{{Rerun metrics}}
\begin{{table}}[H]
\centering
\caption{{Seed-1 rerun metrics (kidney, full-HVG, hvg=2000, top\_k=100).}}
\begin{{tabular}}{{lrrrr}}
\toprule
Model tier & Cells & Precision & Recall & F1 \\
\midrule
{rerun_rows_tex}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Rerun degradation summary}}
\begin{{table}}[H]
\centering
\caption{{Rerun degradation from 200 to 1000 cells.}}
\begin{{tabular}}{{lrrrr}}
\toprule
Model tier & F1@200 & F1@1000 & Percent drop & Ratio (1000/200) \\
\midrule
{rerun_drop_rows_tex}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.85\linewidth]{{figures_extended/fig_ext_rerun_scaling_seed1.png}}
\caption{{Current-checkout rerun scaling curves.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.75\linewidth]{{figures_extended/fig_ext_rerun_drop_200_to_1000.png}}
\caption{{Rerun percent drop in F1 from 200 to 1000 cells.}}
\end{{figure}}

\subsection{{Rerun vs archived agreement}}
\begin{{table}}[H]
\centering
\caption{{Condition-wise agreement between reruns and archived snapshot means.}}
\begin{{tabular}}{{lrrrr}}
\toprule
Model tier & Cells & Archived F1 & Rerun F1 & Rerun deviation \\
\midrule
{rerun_vs_rows_tex}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\linewidth]{{figures_extended/fig_ext_rerun_vs_snapshot.png}}
\caption{{Rerun reproducibility against archived snapshot values.}}
\end{{figure}}

\section{{Policy and Baseline Sensitivity}}
\begin{{itemize}}
\item Best percentile/full-HVG ratio: small={policy_small_pct:.2f}$\times$, medium={policy_medium_pct:.2f}$\times$
\item TF-source+target/full-HVG ratio: small={policy_small_tf:.1f}$\times$, medium={policy_medium_tf:.1f}$\times$
\item Kidney baseline-over-attention ratios: coexpression={baseline_kidney_coexpr:.2f}$\times$, GRNBoost2={baseline_kidney_grnboost2:.2f}$\times$
\end{{itemize}}

\section{{Conclusion}}
Both archived workshop aggregates and new reruns support the same qualitative conclusion: attention-derived unconstrained full-HVG GRN recovery does not improve with cell-count scaling in this setup, and stability degrades in parallel. Candidate-space and threshold choices remain dominant levers on apparent benchmark alignment.

\end{{document}}
"""


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    reports_dir = root / "subproject_06_scaling_failure_analysis" / "reports"
    tables = _load_tables(reports_dir)

    md = _build_markdown(tables)
    tex = _build_latex(tables)

    md_path = reports_dir / "scaling_failure_workshop_paper_v2.md"
    tex_path = reports_dir / "scaling_failure_workshop_paper_v2.tex"
    md_path.write_text(md, encoding="utf-8")
    tex_path.write_text(tex, encoding="utf-8")

    print(f"Wrote Markdown: {md_path}")
    print(f"Wrote LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
