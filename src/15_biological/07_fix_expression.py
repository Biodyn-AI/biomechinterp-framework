#!/usr/bin/env python3
"""
Fix Analysis E: Add expression data using correct gene name column (feature_name categorical).
Updates tf_characterization.json and tf_characterization_summary.md.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix, issparse

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

OUT_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/biological_impact/per_tf"
TS_IMMUNE = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad"

def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.bool_): return bool(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

# Load existing results
with open(os.path.join(OUT_DIR, 'tf_characterization.json')) as f:
    results = json.load(f)

print("Loading Tabula Sapiens immune data with correct gene names...")
import h5py

with h5py.File(TS_IMMUNE, 'r') as f:
    # Read feature_name as categorical
    cats = f['var']['feature_name']['categories'][:]
    codes = f['var']['feature_name']['codes'][:]
    gene_names = [cats[c].decode() if isinstance(cats[c], bytes) else str(cats[c]) for c in codes]
    print(f"  {len(gene_names)} genes (feature_name)")

    # Read expression matrix
    data = f['X']['data'][:]
    indices = f['X']['indices'][:]
    indptr = f['X']['indptr'][:]
    shape = (len(indptr) - 1, len(gene_names))
    X = csr_matrix((data, indices, indptr), shape=shape)
    print(f"  {X.shape[0]} cells × {X.shape[1]} genes")

gene_to_idx = {g: i for i, g in enumerate(gene_names)}

# Extract expression stats for all 18 TFs
print("\nComputing expression statistics for 18 TFs:")
tf_expr_data = {}
for tf_entry in results['tf_data']:
    tf = tf_entry['tf']
    if tf in gene_to_idx:
        idx = gene_to_idx[tf]
        col = X[:, idx].toarray().flatten()
        mean_expr = float(np.mean(col))
        var_expr = float(np.var(col))
        median_expr = float(np.median(col))
        nonzero_frac = float(np.mean(col > 0))
        max_expr = float(np.max(col))

        tf_entry['mean_expression'] = mean_expr
        tf_entry['expression_variance'] = var_expr
        tf_entry['median_expression'] = median_expr
        tf_entry['nonzero_fraction'] = nonzero_frac
        tf_entry['max_expression'] = max_expr
        tf_entry['found_in_expr_data'] = True

        tf_expr_data[tf] = {
            'mean_expression': mean_expr,
            'variance': var_expr,
            'nonzero_fraction': nonzero_frac
        }
        print(f"  {tf:8s}: mean={mean_expr:.4f}, var={var_expr:.4f}, nonzero={nonzero_frac:.3f}")
    else:
        tf_entry['found_in_expr_data'] = False
        print(f"  {tf:8s}: NOT FOUND")

# Redo expression-related statistical tests
print("\nRedoing expression-related statistical tests:")

# Build dataframe from results
df_data = []
for tf_entry in results['tf_data']:
    df_data.append({
        'tf': tf_entry['tf'],
        'observed_auroc': tf_entry['observed_auroc'],
        'successful': tf_entry['successful'],
        'mean_expression': tf_entry.get('mean_expression'),
        'expression_variance': tf_entry.get('expression_variance'),
        'nonzero_fraction': tf_entry.get('nonzero_fraction'),
        'found_in_expr_data': tf_entry.get('found_in_expr_data', False)
    })
df = pd.DataFrame(df_data)

expr_valid = df[df['found_in_expr_data'] == True].dropna(subset=['mean_expression'])
print(f"  TFs with expression data: {len(expr_valid)}")

if len(expr_valid) > 3:
    rho3, p3 = stats.spearmanr(expr_valid['observed_auroc'], expr_valid['mean_expression'])
    print(f"  AUROC vs mean expression: ρ={rho3:.3f}, p={p3:.4f}")
    results['statistical_tests']['auroc_vs_mean_expr'] = {'rho': float(rho3), 'p': float(p3), 'n': len(expr_valid)}

    rho4, p4 = stats.spearmanr(expr_valid['observed_auroc'], expr_valid['expression_variance'])
    print(f"  AUROC vs expression variance: ρ={rho4:.3f}, p={p4:.4f}")
    results['statistical_tests']['auroc_vs_variance'] = {'rho': float(rho4), 'p': float(p4), 'n': len(expr_valid)}

    rho5, p5 = stats.spearmanr(expr_valid['observed_auroc'], expr_valid['nonzero_fraction'])
    print(f"  AUROC vs nonzero fraction: ρ={rho5:.3f}, p={p5:.4f}")
    results['statistical_tests']['auroc_vs_nonzero_frac'] = {'rho': float(rho5), 'p': float(p5), 'n': len(expr_valid)}

    # Successful vs unsuccessful: expression
    success_expr = expr_valid[expr_valid['successful']]['mean_expression']
    fail_expr = expr_valid[~expr_valid['successful']]['mean_expression']
    if len(success_expr) >= 2 and len(fail_expr) >= 2:
        u, p = stats.mannwhitneyu(success_expr, fail_expr, alternative='two-sided')
        print(f"  Expression (successful vs unsuccessful): U={u:.0f}, p={p:.4f}")
        print(f"    Successful median: {success_expr.median():.4f}, Unsuccessful median: {fail_expr.median():.4f}")
        results['statistical_tests']['expression_success_vs_fail'] = {
            'U': float(u), 'p': float(p),
            'successful_median': float(success_expr.median()),
            'unsuccessful_median': float(fail_expr.median()),
            'n_successful': len(success_expr),
            'n_unsuccessful': len(fail_expr)
        }

# Fix power analysis (n_success was overwritten by threshold analysis variable)
n_actual_success = sum(1 for t in results['tf_data'] if t['successful'])
results['power_analysis']['n_successful'] = n_actual_success
results['power_analysis']['n_unsuccessful'] = 18 - n_actual_success
print(f"\n  Corrected: {n_actual_success} successful, {18-n_actual_success} unsuccessful")

# Save updated results
with open(os.path.join(OUT_DIR, 'tf_characterization.json'), 'w') as f:
    json.dump(results, f, indent=2, default=_json_default)
print(f"\nSaved updated: {os.path.join(OUT_DIR, 'tf_characterization.json')}")

# ─── Regenerate summary markdown ────────────────────────────────────────────
sorted_tfs = sorted(results['tf_data'], key=lambda x: x['observed_auroc'], reverse=True)

md_lines = [
    "# Analysis E: Per-TF Characterization — Results",
    "",
    "## Overview",
    "",
    f"18 TFs evaluated with bootstrap confidence intervals on Tabula Sapiens immune data (n=5,000 cells, 2,000 HVGs).",
    f"**{n_actual_success} TFs** have 95% CI entirely above 0.5 (successful); **{18-n_actual_success} TFs** have CI including ≤0.5 (unsuccessful).",
    "",
    "## Per-TF Table",
    "",
    "| TF | AUROC | 95% CI | Successful | Family | Category | Full Reg. | Eval. Reg. | Mean Expr | Nonzero % |",
    "|-----|-------|--------|------------|--------|----------|-----------|------------|-----------|-----------|",
]

for tf in sorted_tfs:
    ci = f"[{tf['ci_95_lo']:.3f}, {tf['ci_95_hi']:.3f}]"
    succ = "**Yes**" if tf['successful'] else "No"
    expr = f"{tf['mean_expression']:.4f}" if tf.get('mean_expression') is not None else "N/A"
    nzf = f"{tf['nonzero_fraction']*100:.1f}" if tf.get('nonzero_fraction') is not None else "N/A"
    md_lines.append(
        f"| {tf['tf']} | {tf['observed_auroc']:.3f} | {ci} | {succ} | "
        f"{tf['tf_family']} | {tf['functional_category']} | {tf['full_regulon_size']} | "
        f"{tf['evaluable_regulon_size']} | {expr} | {nzf} |"
    )

md_lines.extend([
    "",
    "## Key Findings",
    "",
    "### 1. Master Regulator Enrichment (Primary Finding)",
    "",
    f"Master regulators have significantly higher AUROC than non-master TFs:",
    f"- Master regulators (n=9): median AUROC = {results['statistical_tests']['master_vs_other_auroc']['master_median']:.3f}",
    f"- Other TFs (n=9): median AUROC = {results['statistical_tests']['master_vs_other_auroc']['other_median']:.3f}",
    f"- Mann-Whitney (one-sided): p = {results['statistical_tests']['master_vs_other_auroc']['p']:.4f}",
    "",
    f"Fisher's exact test for master regulator × success contingency:",
    f"- Master regulators: {results['statistical_tests']['fisher_master_x_success']['master_success_rate']*100:.0f}% successful",
    f"- Others: {results['statistical_tests']['fisher_master_x_success']['other_success_rate']*100:.0f}% successful",
    f"- OR = {results['statistical_tests']['fisher_master_x_success']['odds_ratio']:.2f}, p = {results['statistical_tests']['fisher_master_x_success']['p']:.4f} (not significant — underpowered)",
    "",
    "### 2. Functional Category Breakdown",
    "",
])
for cat, info in results['group_comparisons'].items():
    md_lines.append(f"- **{cat}** (n={info['n']}): mean AUROC = {info['mean_auroc']:.3f}, "
                   f"successful = {info['n_successful']}/{info['n']}, TFs: {', '.join(info['tfs'])}")

md_lines.extend([
    "",
    "### 3. Regulon Size",
    "",
    f"- AUROC vs full regulon size: ρ = {results['statistical_tests']['auroc_vs_full_regulon']['rho']:.3f}, p = {results['statistical_tests']['auroc_vs_full_regulon']['p']:.4f} (n.s.)",
    f"- AUROC vs evaluable regulon size: ρ = {results['statistical_tests']['auroc_vs_evaluable_regulon']['rho']:.3f}, p = {results['statistical_tests']['auroc_vs_evaluable_regulon']['p']:.4f} (marginal)",
    "",
    "| Min Evaluable Targets | n TFs | Mean AUROC | % Successful |",
    "|----------------------|-------|------------|--------------|",
])
for tr in results['threshold_analysis']:
    md_lines.append(f"| ≥{tr['threshold']} | {tr['n_tfs']} | {tr['mean_auroc']:.3f} | {tr['frac_successful']*100:.0f}% |")

md_lines.extend([
    "",
    "### 4. Expression Correlations",
    "",
])
for key in ['auroc_vs_mean_expr', 'auroc_vs_variance', 'auroc_vs_nonzero_frac']:
    if key in results['statistical_tests']:
        t = results['statistical_tests'][key]
        md_lines.append(f"- {key}: ρ = {t['rho']:.3f}, p = {t['p']:.4f}")

if 'expression_success_vs_fail' in results['statistical_tests']:
    t = results['statistical_tests']['expression_success_vs_fail']
    md_lines.append(f"- Expression (successful vs unsuccessful): U = {t['U']:.0f}, p = {t['p']:.4f}")
    md_lines.append(f"  - Successful median = {t['successful_median']:.4f}, Unsuccessful median = {t['unsuccessful_median']:.4f}")

md_lines.extend([
    "",
    "### 5. The 7 Successful TFs",
    "",
])
for tf in sorted_tfs:
    if tf['successful']:
        md_lines.append(f"- **{tf['tf']}** (AUROC {tf['observed_auroc']:.3f}): {tf.get('description', '')}")

md_lines.extend([
    "",
    "### 6. Interpretation",
    "",
    "The successful TFs are predominantly **lineage-defining master regulators** (LEF1, KLF1, GATA1, PPARG, WT1) that:",
    "- Control large, well-characterized regulons",
    "- Have cell-type-restricted expression patterns",
    "- Operate as pioneer/chromatin remodeling factors",
    "",
    "The two exceptions are **EGR1** (immediate-early, signal-dependent) and **MAL** (membrane protein, not a classical TF).",
    "",
    "**Hypothesis**: Correlation-based edge scores work best for TFs with:",
    "1. Strong cell-type-specific expression patterns (which create high-variance correlations in mixed populations)",
    "2. Large, well-characterized regulons (more statistical power for AUROC estimation)",
    "",
    "### 7. Power Analysis Caveat",
    "",
    f"With n=18 total TFs ({n_actual_success} vs {18-n_actual_success} split), statistical power is severely limited.",
    "All results are **hypothesis-generating**, not confirmatory.",
    "",
])

summary_path = os.path.join(OUT_DIR, 'tf_characterization_summary.md')
with open(summary_path, 'w') as f:
    f.write('\n'.join(md_lines))
print(f"Saved updated: {summary_path}")
print("\nDone.")
