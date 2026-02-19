#!/usr/bin/env python3
"""
Analysis E Hardening: Permutation tests, single-target exclusion, BH correction.

Addresses:
1. No multiple testing correction across ~10 tests → BH-FDR
2. No permutation test for main comparison → 10,000 label shuffles
3. Some TFs have AUROC based on a single target → exclude and recompute
"""

import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

OUT_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/biological_impact/per_tf"

def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.bool_): return bool(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

# Load existing results
with open(os.path.join(OUT_DIR, 'tf_characterization.json')) as f:
    results = json.load(f)

tf_data = results['tf_data']
df = pd.DataFrame(tf_data)
print(f"Loaded {len(df)} TFs")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. REPORT EVALUABLE TARGET COUNTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. EVALUABLE TARGET COUNTS")
print("=" * 70)

print("\nTFs by out_degree_in_hvg (number of TRRUST targets scored):")
for _, row in df.sort_values('out_degree_in_hvg', ascending=False).iterrows():
    flag = " ← SINGLE TARGET" if row['out_degree_in_hvg'] <= 1 else ""
    flag2 = " ← ZERO TARGETS" if row['out_degree_in_hvg'] == 0 else ""
    print(f"  {row['tf']:8s}: out_degree={int(row['out_degree_in_hvg']):3d}, "
          f"eval_regulon={int(row['evaluable_regulon_size']):3d}, "
          f"AUROC={row['observed_auroc']:.3f}, "
          f"category={row['functional_category']}{flag}{flag2}")

# Identify TFs with ≥3 evaluable targets
robust_tfs = df[df['out_degree_in_hvg'] >= 3].copy()
fragile_tfs = df[df['out_degree_in_hvg'] < 3].copy()
print(f"\n  TFs with ≥3 evaluable targets: {len(robust_tfs)} / {len(df)}")
print(f"  TFs with <3 evaluable targets (fragile): {len(fragile_tfs)} / {len(df)}")
print(f"  Robust TFs: {list(robust_tfs['tf'])}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PERMUTATION TEST: master vs non-master
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. PERMUTATION TEST: master vs non-master AUROC")
print("=" * 70)

N_PERM = 10000
rng = np.random.RandomState(42)

def permutation_test(aurocs, labels, n_perm, rng, alternative='greater'):
    """Permutation test for difference in means between two groups.

    labels: boolean array, True = group A
    alternative: 'greater' means test if group A > group B
    """
    observed_diff = aurocs[labels].mean() - aurocs[~labels].mean()
    n_a = labels.sum()

    perm_diffs = np.zeros(n_perm)
    for i in range(n_perm):
        perm_labels = rng.permutation(labels)
        perm_diffs[i] = aurocs[perm_labels].mean() - aurocs[~perm_labels].mean()

    if alternative == 'greater':
        p_value = (np.sum(perm_diffs >= observed_diff) + 1) / (n_perm + 1)
    elif alternative == 'two-sided':
        p_value = (np.sum(np.abs(perm_diffs) >= abs(observed_diff)) + 1) / (n_perm + 1)
    else:
        p_value = (np.sum(perm_diffs <= observed_diff) + 1) / (n_perm + 1)

    return observed_diff, p_value, perm_diffs

# --- All 18 TFs ---
aurocs_all = df['observed_auroc'].values
is_master_all = (df['functional_category'] == 'master_regulator').values

obs_diff_all, perm_p_all, perm_dists_all = permutation_test(
    aurocs_all, is_master_all, N_PERM, rng, 'greater')

print(f"\n  ALL 18 TFs:")
print(f"    Master (n={is_master_all.sum()}): mean AUROC = {aurocs_all[is_master_all].mean():.4f}")
print(f"    Other  (n={(~is_master_all).sum()}): mean AUROC = {aurocs_all[~is_master_all].mean():.4f}")
print(f"    Observed Δ = {obs_diff_all:+.4f}")
print(f"    Permutation p (10,000 shuffles, one-sided): {perm_p_all:.4f}")
print(f"    95th percentile of null: {np.percentile(perm_dists_all, 95):.4f}")

# --- Robust TFs only (≥3 targets) ---
if len(robust_tfs) >= 4:
    aurocs_robust = robust_tfs['observed_auroc'].values
    is_master_robust = (robust_tfs['functional_category'] == 'master_regulator').values

    if is_master_robust.sum() >= 1 and (~is_master_robust).sum() >= 1:
        obs_diff_robust, perm_p_robust, perm_dists_robust = permutation_test(
            aurocs_robust, is_master_robust, N_PERM, rng, 'greater')

        print(f"\n  ROBUST TFs ONLY (≥3 evaluable targets, n={len(robust_tfs)}):")
        print(f"    Master (n={is_master_robust.sum()}): mean AUROC = {aurocs_robust[is_master_robust].mean():.4f}")
        print(f"    Other  (n={(~is_master_robust).sum()}): mean AUROC = {aurocs_robust[~is_master_robust].mean():.4f}")
        print(f"    Observed Δ = {obs_diff_robust:+.4f}")
        print(f"    Permutation p (10,000 shuffles, one-sided): {perm_p_robust:.4f}")
    else:
        perm_p_robust = np.nan
        obs_diff_robust = np.nan
        print(f"\n  ROBUST TFs: insufficient group sizes for permutation test")
else:
    perm_p_robust = np.nan
    obs_diff_robust = np.nan
    print(f"\n  ROBUST TFs: too few TFs with ≥3 targets ({len(robust_tfs)})")

# --- Exclude single-target TFs (out_degree ≤ 1) ---
multi_target = df[df['out_degree_in_hvg'] > 1].copy()
aurocs_multi = multi_target['observed_auroc'].values
is_master_multi = (multi_target['functional_category'] == 'master_regulator').values

if is_master_multi.sum() >= 1 and (~is_master_multi).sum() >= 1:
    obs_diff_multi, perm_p_multi, _ = permutation_test(
        aurocs_multi, is_master_multi, N_PERM, rng, 'greater')

    print(f"\n  MULTI-TARGET TFs ONLY (≥2 targets, n={len(multi_target)}):")
    print(f"    Master (n={is_master_multi.sum()}): mean AUROC = {aurocs_multi[is_master_multi].mean():.4f}")
    print(f"    Other  (n={(~is_master_multi).sum()}): mean AUROC = {aurocs_multi[~is_master_multi].mean():.4f}")
    print(f"    Observed Δ = {obs_diff_multi:+.4f}")
    print(f"    Permutation p (10,000 shuffles, one-sided): {perm_p_multi:.4f}")
else:
    perm_p_multi = np.nan
    obs_diff_multi = np.nan

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PERMUTATION TESTS FOR ALL CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. ALL STATISTICAL TESTS (with permutation p-values)")
print("=" * 70)

all_tests = []

# Test 1: Master vs other AUROC (all TFs)
all_tests.append({
    'test': 'master_vs_other_auroc_all18',
    'description': 'Master regulators vs other TFs (all 18)',
    'observed_statistic': float(obs_diff_all),
    'parametric_p': float(results['statistical_tests']['master_vs_other_auroc']['p']),
    'permutation_p': float(perm_p_all),
    'n': 18,
    'alternative': 'greater'
})

# Test 2: Master vs other (≥3 targets)
if not np.isnan(perm_p_robust):
    all_tests.append({
        'test': 'master_vs_other_auroc_robust',
        'description': 'Master vs other (≥3 evaluable targets only)',
        'observed_statistic': float(obs_diff_robust),
        'permutation_p': float(perm_p_robust),
        'n': len(robust_tfs),
        'alternative': 'greater'
    })

# Test 3: Master vs other (≥2 targets)
if not np.isnan(perm_p_multi):
    all_tests.append({
        'test': 'master_vs_other_auroc_multi',
        'description': 'Master vs other (≥2 targets only)',
        'observed_statistic': float(obs_diff_multi),
        'permutation_p': float(perm_p_multi),
        'n': len(multi_target),
        'alternative': 'greater'
    })

# Test 4: Fisher's exact (master × successful)
all_tests.append({
    'test': 'fisher_master_x_success',
    'description': 'Fisher exact: master regulator × successful',
    'observed_statistic': float(results['statistical_tests']['fisher_master_x_success']['odds_ratio']),
    'parametric_p': float(results['statistical_tests']['fisher_master_x_success']['p']),
    'n': 18,
    'alternative': 'two-sided'
})

# Test 5: AUROC vs full regulon size
all_tests.append({
    'test': 'auroc_vs_full_regulon',
    'description': 'Spearman: AUROC vs full regulon size',
    'observed_statistic': float(results['statistical_tests']['auroc_vs_full_regulon']['rho']),
    'parametric_p': float(results['statistical_tests']['auroc_vs_full_regulon']['p']),
    'n': 18,
    'alternative': 'two-sided'
})

# Test 6: AUROC vs evaluable regulon size
all_tests.append({
    'test': 'auroc_vs_evaluable_regulon',
    'description': 'Spearman: AUROC vs evaluable regulon size',
    'observed_statistic': float(results['statistical_tests']['auroc_vs_evaluable_regulon']['rho']),
    'parametric_p': float(results['statistical_tests']['auroc_vs_evaluable_regulon']['p']),
    'n': 18,
    'alternative': 'two-sided'
})

# Test 7-9: Expression correlations
for key in ['auroc_vs_mean_expr', 'auroc_vs_variance', 'auroc_vs_nonzero_frac']:
    if key in results['statistical_tests']:
        t = results['statistical_tests'][key]
        all_tests.append({
            'test': key,
            'description': f'Spearman: {key}',
            'observed_statistic': float(t['rho']),
            'parametric_p': float(t['p']),
            'n': int(t['n']),
            'alternative': 'two-sided'
        })

# Test 10: Expression (successful vs unsuccessful)
if 'expression_success_vs_fail' in results['statistical_tests']:
    t = results['statistical_tests']['expression_success_vs_fail']
    all_tests.append({
        'test': 'expression_success_vs_fail',
        'description': 'Mann-Whitney: expression in successful vs unsuccessful',
        'observed_statistic': float(t['U']),
        'parametric_p': float(t['p']),
        'n': int(t['n_successful']) + int(t['n_unsuccessful']),
        'alternative': 'two-sided'
    })

# Test 11: Regulon size (successful vs unsuccessful)
t = results['statistical_tests']['regulon_size_success_vs_fail']
all_tests.append({
    'test': 'regulon_size_success_vs_fail',
    'description': 'Mann-Whitney: regulon size in successful vs unsuccessful',
    'observed_statistic': float(t['U']),
    'parametric_p': float(t['p']),
    'n': 18,
    'alternative': 'two-sided'
})

# ═══════════════════════════════════════════════════════════════════════════════
# 4. BENJAMINI-HOCHBERG CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. BENJAMINI-HOCHBERG CORRECTION")
print("=" * 70)

# Use permutation p where available, otherwise parametric
for t in all_tests:
    t['raw_p'] = t.get('permutation_p', t.get('parametric_p'))

n_tests = len(all_tests)
raw_ps = np.array([t['raw_p'] for t in all_tests])

# BH correction
sorted_idx = np.argsort(raw_ps)
bh_threshold = np.zeros(n_tests)
for rank, idx in enumerate(sorted_idx, 1):
    bh_threshold[idx] = 0.05 * rank / n_tests

# Adjusted p-values (Benjamini-Hochberg)
bh_adjusted = np.zeros(n_tests)
for i in range(n_tests):
    bh_adjusted[i] = raw_ps[i] * n_tests / (np.searchsorted(np.sort(raw_ps), raw_ps[i]) + 1)
bh_adjusted = np.minimum(bh_adjusted, 1.0)

# Ensure monotonicity (step-up)
from statsmodels.stats.multitest import multipletests
reject, bh_adjusted_proper, _, _ = multipletests(raw_ps, alpha=0.05, method='fdr_bh')

print(f"\n  {n_tests} tests total")
print(f"  Significant after BH correction at α=0.05: {reject.sum()}")

print(f"\n  {'Test':<40s} {'raw_p':>10s} {'BH_q':>10s} {'sig':>5s}")
print("  " + "-" * 70)
for i, t in enumerate(all_tests):
    t['bh_adjusted_p'] = float(bh_adjusted_proper[i])
    t['significant_after_bh'] = bool(reject[i])
    sig_str = "**" if reject[i] else ""
    print(f"  {t['test']:<40s} {t['raw_p']:>10.4f} {t['bh_adjusted_p']:>10.4f} {sig_str:>5s}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SAVE HARDENED RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. SAVING HARDENED RESULTS")
print("=" * 70)

hardened = {
    'all_tests': all_tests,
    'n_tests': n_tests,
    'n_significant_after_bh': int(reject.sum()),
    'permutation_details': {
        'n_permutations': N_PERM,
        'seed': 42,
        'master_vs_other_all18': {
            'observed_diff': float(obs_diff_all),
            'permutation_p': float(perm_p_all),
            'null_95th': float(np.percentile(perm_dists_all, 95)),
            'null_mean': float(np.mean(perm_dists_all)),
            'null_std': float(np.std(perm_dists_all)),
        }
    },
    'robustness_checks': {
        'all_18_tfs': {
            'n': 18,
            'master_mean_auroc': float(aurocs_all[is_master_all].mean()),
            'other_mean_auroc': float(aurocs_all[~is_master_all].mean()),
            'permutation_p': float(perm_p_all)
        },
        'multi_target_only': {
            'n': len(multi_target),
            'excluded': list(df[df['out_degree_in_hvg'] <= 1]['tf']),
            'permutation_p': float(perm_p_multi) if not np.isnan(perm_p_multi) else None,
            'observed_diff': float(obs_diff_multi) if not np.isnan(obs_diff_multi) else None,
        },
        'robust_only_3plus': {
            'n': len(robust_tfs),
            'tfs': list(robust_tfs['tf']),
            'permutation_p': float(perm_p_robust) if not np.isnan(perm_p_robust) else None,
            'observed_diff': float(obs_diff_robust) if not np.isnan(obs_diff_robust) else None,
        }
    },
    'target_count_table': [
        {
            'tf': row['tf'],
            'out_degree_in_hvg': int(row['out_degree_in_hvg']),
            'evaluable_regulon_size': int(row['evaluable_regulon_size']),
            'auroc': float(row['observed_auroc']),
            'functional_category': row['functional_category'],
            'successful': bool(row['successful']),
            'note': 'SINGLE TARGET' if row['out_degree_in_hvg'] <= 1 else ''
        }
        for _, row in df.sort_values('out_degree_in_hvg', ascending=False).iterrows()
    ]
}

with open(os.path.join(OUT_DIR, 'tf_characterization_hardened.json'), 'w') as f:
    json.dump(hardened, f, indent=2, default=_json_default)
print(f"  Saved: tf_characterization_hardened.json")

# ─── Update summary markdown ────────────────────────────────────────────────
md = [
    "# Analysis E: Per-TF Characterization — Hardened Results",
    "",
    "## Methodological Improvements",
    "",
    "1. **Permutation-based p-values** (10,000 label shuffles) replace asymptotic Mann-Whitney",
    "2. **BH-FDR correction** across all tests",
    "3. **Single-target TFs flagged and excluded** in robustness checks",
    "",
    "## Evaluable Target Counts",
    "",
    "| TF | Targets in HVG | Eval. Regulon | AUROC | Category | Note |",
    "|----|---------------|---------------|-------|----------|------|",
]
for row in hardened['target_count_table']:
    md.append(f"| {row['tf']} | {row['out_degree_in_hvg']} | {row['evaluable_regulon_size']} | "
             f"{row['auroc']:.3f} | {row['functional_category']} | {row['note']} |")

md.extend([
    "",
    f"**{len(fragile_tfs)} of 18 TFs have ≤1 evaluable target** — their AUROC is the rank of a single gene",
    "among ~1999, not a regulon-level assessment.",
    "",
    "## Permutation Tests",
    "",
    f"### All 18 TFs",
    f"- Master mean AUROC: {hardened['robustness_checks']['all_18_tfs']['master_mean_auroc']:.4f}",
    f"- Other mean AUROC: {hardened['robustness_checks']['all_18_tfs']['other_mean_auroc']:.4f}",
    f"- Permutation p (one-sided, 10,000 shuffles): **{perm_p_all:.4f}**",
    "",
])

rc = hardened['robustness_checks']['multi_target_only']
if rc['permutation_p'] is not None:
    md.extend([
        f"### Excluding single-target TFs (≥2 targets, n={rc['n']})",
        f"- Excluded: {', '.join(rc['excluded'])}",
        f"- Permutation p: **{rc['permutation_p']:.4f}**",
        f"- Observed Δ: {rc['observed_diff']:+.4f}",
        "",
    ])

rc3 = hardened['robustness_checks']['robust_only_3plus']
if rc3['permutation_p'] is not None:
    md.extend([
        f"### Robust TFs only (≥3 targets, n={rc3['n']})",
        f"- TFs: {', '.join(rc3['tfs'])}",
        f"- Permutation p: **{rc3['permutation_p']:.4f}**",
        f"- Observed Δ: {rc3['observed_diff']:+.4f}",
        "",
    ])

md.extend([
    "## All Tests with BH Correction",
    "",
    f"**{n_tests} tests**, {int(reject.sum())} significant after BH-FDR at α=0.05.",
    "",
    "| Test | raw p | BH q | Significant |",
    "|----|-------|------|------------|",
])
for t in all_tests:
    sig = "Yes" if t['significant_after_bh'] else "No"
    md.append(f"| {t['description']} | {t['raw_p']:.4f} | {t['bh_adjusted_p']:.4f} | {sig} |")

md.extend([
    "",
    "## Interpretation",
    "",
    "The master regulator advantage is evaluated at three stringency levels:",
    "1. All 18 TFs (including single-target): tests the broad pattern",
    "2. ≥2 targets: excludes TFs where AUROC = rank of 1 gene",
    "3. ≥3 targets: most conservative, only regulon-level assessments",
    "",
])

with open(os.path.join(OUT_DIR, 'tf_characterization_hardened_summary.md'), 'w') as f:
    f.write('\n'.join(md))
print(f"  Saved: tf_characterization_hardened_summary.md")

print("\n" + "=" * 70)
print("Analysis E HARDENING COMPLETE")
print("=" * 70)
