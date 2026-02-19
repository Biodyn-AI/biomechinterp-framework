"""
Phase 2, Objective 2: Adaptive conformal prediction.

Implement adaptive conformal prediction that adjusts set sizes based on
local difficulty. Compare to fixed-alpha conformal from Phase 1.

Methods:
1. Mondrian conformal: condition on score bins (local alpha adjustment)
2. Weighted conformal: weight nonconformity scores by local density
3. Comparison metrics: coverage, set size, conditional coverage by subgroup
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

OUT_DIR = '/Users/ihorkendiukhov/biodyn-work/subproject_31_uncertainty_calibration_edge_scores/implementation'


def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


# ============================================================
# Load Phase 1 data
# ============================================================
print("=" * 70, flush=True)
print("PHASE 2: ADAPTIVE CONFORMAL PREDICTION", flush=True)
print("=" * 70, flush=True)

with open(os.path.join(OUT_DIR, 'edge_scores_and_ground_truth.pkl'), 'rb') as f:
    data = pickle.load(f)

edge_scores = data['edge_scores']
ground_truth = data['ground_truth']
methods = data['methods']
perturbations = data['perturbations']

# Reproduce the same split as Phase 1
perts = sorted(set(perturbations))
np.random.shuffle(perts)
n = len(perts)
n_train = int(0.5 * n)
n_cal = int(0.25 * n)
perts_train = set(perts[:n_train])
perts_cal = set(perts[n_train:n_train + n_cal])
perts_test = set(perts[n_train + n_cal:])

print(f"  Perturbation split: train={len(perts_train)}, cal={len(perts_cal)}, test={len(perts_test)}", flush=True)

# Build arrays per method
method_data = {}
for method in methods:
    scores_dict = edge_scores[method]
    y_all, s_all, split_all, pert_all = [], [], [], []
    for (pert, target), score in scores_dict.items():
        if (pert, target) not in ground_truth:
            continue
        y_all.append(int(ground_truth[(pert, target)]))
        s_all.append(score)
        pert_all.append(pert)
        if pert in perts_train:
            split_all.append('train')
        elif pert in perts_cal:
            split_all.append('cal')
        else:
            split_all.append('test')

    y_all = np.array(y_all)
    s_all = np.array(s_all)
    split_all = np.array(split_all)
    pert_all = np.array(pert_all)

    method_data[method] = {
        'y_train': y_all[split_all == 'train'],
        's_train': s_all[split_all == 'train'],
        'y_cal': y_all[split_all == 'cal'],
        's_cal': s_all[split_all == 'cal'],
        'y_test': y_all[split_all == 'test'],
        's_test': s_all[split_all == 'test'],
        'pert_test': pert_all[split_all == 'test'],
        'pert_cal': pert_all[split_all == 'cal'],
    }


def normalize_with_range(scores, mn, mx):
    if mx - mn < 1e-12:
        return np.full_like(scores, 0.5)
    return np.clip((scores - mn) / (mx - mn), 0, 1)


# ============================================================
# Method 1: Standard (fixed-alpha) conformal (baseline from Phase 1)
# ============================================================
def standard_conformal(y_cal, cal_probs, test_probs, alpha):
    """Standard split conformal prediction."""
    n_cal = len(y_cal)
    nonconf = np.where(y_cal == 1, 1 - cal_probs, cal_probs)
    q = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
    q = min(q, 1.0)
    threshold = np.quantile(nonconf, q)

    include_1 = (1 - test_probs) <= threshold
    include_0 = test_probs <= threshold
    set_sizes = include_1.astype(int) + include_0.astype(int)

    return include_0, include_1, set_sizes, threshold


# ============================================================
# Method 2: Mondrian conformal (condition on score bins)
# ============================================================
def mondrian_conformal(y_cal, cal_probs, test_probs, alpha, n_groups=3):
    """Mondrian conformal: compute separate thresholds per score bin."""
    # Bin calibration scores into groups
    group_bounds = np.quantile(cal_probs, np.linspace(0, 1, n_groups + 1))
    group_bounds[0] = -0.01
    group_bounds[-1] = 1.01

    cal_groups = np.digitize(cal_probs, group_bounds) - 1
    cal_groups = np.clip(cal_groups, 0, n_groups - 1)
    test_groups = np.digitize(test_probs, group_bounds) - 1
    test_groups = np.clip(test_groups, 0, n_groups - 1)

    nonconf = np.where(y_cal == 1, 1 - cal_probs, cal_probs)

    include_0 = np.zeros(len(test_probs), dtype=bool)
    include_1 = np.zeros(len(test_probs), dtype=bool)
    group_thresholds = {}

    for g in range(n_groups):
        cal_mask = cal_groups == g
        test_mask = test_groups == g

        if cal_mask.sum() < 5:
            # Too few points; use full calibration set
            group_nonconf = nonconf
        else:
            group_nonconf = nonconf[cal_mask]

        n_g = len(group_nonconf)
        q = np.ceil((1 - alpha) * (n_g + 1)) / n_g
        q = min(q, 1.0)
        thresh = np.quantile(group_nonconf, q)
        group_thresholds[g] = thresh

        include_1[test_mask] = (1 - test_probs[test_mask]) <= thresh
        include_0[test_mask] = test_probs[test_mask] <= thresh

    set_sizes = include_1.astype(int) + include_0.astype(int)
    return include_0, include_1, set_sizes, group_thresholds


# ============================================================
# Method 3: Weighted conformal (CQR-style local adaptation)
# ============================================================
def weighted_conformal(y_cal, cal_probs, test_probs, alpha, bandwidth=0.1):
    """
    Locally-weighted conformal: for each test point, weight calibration
    nonconformity scores by proximity to the test point's predicted probability.
    Uses Gaussian kernel weighting.
    """
    nonconf_cal = np.where(y_cal == 1, 1 - cal_probs, cal_probs)

    include_0 = np.zeros(len(test_probs), dtype=bool)
    include_1 = np.zeros(len(test_probs), dtype=bool)

    for i in range(len(test_probs)):
        # Compute weights based on distance to this test point
        dists = np.abs(cal_probs - test_probs[i])
        weights = np.exp(-0.5 * (dists / bandwidth) ** 2)
        weights = weights / (weights.sum() + 1e-12)

        # Weighted quantile
        sorted_idx = np.argsort(nonconf_cal)
        sorted_scores = nonconf_cal[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cum_weights = np.cumsum(sorted_weights)

        target_q = 1 - alpha
        thresh_idx = np.searchsorted(cum_weights, target_q)
        thresh_idx = min(thresh_idx, len(sorted_scores) - 1)
        thresh = sorted_scores[thresh_idx]

        include_1[i] = (1 - test_probs[i]) <= thresh
        include_0[i] = test_probs[i] <= thresh

    set_sizes = include_1.astype(int) + include_0.astype(int)
    return include_0, include_1, set_sizes


# ============================================================
# Run all methods
# ============================================================
print("\n" + "=" * 70, flush=True)
print("COMPARING CONFORMAL METHODS", flush=True)
print("=" * 70, flush=True)

adaptive_results = {}

for method in methods:
    d = method_data[method]
    all_s = np.concatenate([d['s_train'], d['s_cal'], d['s_test']])
    mn, mx = all_s.min(), all_s.max()

    s_train_n = normalize_with_range(d['s_train'], mn, mx)
    s_cal_n = normalize_with_range(d['s_cal'], mn, mx)
    s_test_n = normalize_with_range(d['s_test'], mn, mx)

    # Fit isotonic on training set
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso.fit(s_train_n, d['y_train'])
    cal_probs = iso.predict(s_cal_n)
    test_probs = iso.predict(s_test_n)

    adaptive_results[method] = {}

    for alpha in [0.05, 0.10, 0.20]:
        print(f"\n  {method}, alpha={alpha}:", flush=True)

        # Standard conformal
        inc0_std, inc1_std, sizes_std, thresh_std = standard_conformal(
            d['y_cal'], cal_probs, test_probs, alpha)
        cov_std = np.mean([
            (d['y_test'][i] == 1 and inc1_std[i]) or (d['y_test'][i] == 0 and inc0_std[i])
            for i in range(len(d['y_test']))
        ])

        # Mondrian conformal (3 groups)
        inc0_mond, inc1_mond, sizes_mond, thresh_mond = mondrian_conformal(
            d['y_cal'], cal_probs, test_probs, alpha, n_groups=3)
        cov_mond = np.mean([
            (d['y_test'][i] == 1 and inc1_mond[i]) or (d['y_test'][i] == 0 and inc0_mond[i])
            for i in range(len(d['y_test']))
        ])

        # Weighted conformal
        inc0_wt, inc1_wt, sizes_wt = weighted_conformal(
            d['y_cal'], cal_probs, test_probs, alpha, bandwidth=0.15)
        cov_wt = np.mean([
            (d['y_test'][i] == 1 and inc1_wt[i]) or (d['y_test'][i] == 0 and inc0_wt[i])
            for i in range(len(d['y_test']))
        ])

        # Conditional coverage by difficulty (split test into easy/medium/hard by predicted prob)
        prob_terciles = np.quantile(test_probs, [1/3, 2/3])
        easy_mask = test_probs >= prob_terciles[1]  # high confidence true
        hard_mask = test_probs <= prob_terciles[0]  # low confidence
        medium_mask = ~easy_mask & ~hard_mask

        def cond_coverage(y, inc0, inc1, mask):
            if mask.sum() == 0:
                return float('nan')
            covered = [(y[i] == 1 and inc1[i]) or (y[i] == 0 and inc0[i])
                       for i in range(len(y)) if mask[i]]
            return np.mean(covered)

        result = {
            'standard': {
                'coverage': cov_std,
                'avg_set_size': sizes_std.mean(),
                'singleton_frac': (sizes_std == 1).mean(),
                'empty_frac': (sizes_std == 0).mean(),
                'cond_cov_easy': cond_coverage(d['y_test'], inc0_std, inc1_std, easy_mask),
                'cond_cov_medium': cond_coverage(d['y_test'], inc0_std, inc1_std, medium_mask),
                'cond_cov_hard': cond_coverage(d['y_test'], inc0_std, inc1_std, hard_mask),
                'set_sizes': sizes_std,
            },
            'mondrian': {
                'coverage': cov_mond,
                'avg_set_size': sizes_mond.mean(),
                'singleton_frac': (sizes_mond == 1).mean(),
                'empty_frac': (sizes_mond == 0).mean(),
                'cond_cov_easy': cond_coverage(d['y_test'], inc0_mond, inc1_mond, easy_mask),
                'cond_cov_medium': cond_coverage(d['y_test'], inc0_mond, inc1_mond, medium_mask),
                'cond_cov_hard': cond_coverage(d['y_test'], inc0_mond, inc1_mond, hard_mask),
                'set_sizes': sizes_mond,
            },
            'weighted': {
                'coverage': cov_wt,
                'avg_set_size': sizes_wt.mean(),
                'singleton_frac': (sizes_wt == 1).mean(),
                'empty_frac': (sizes_wt == 0).mean(),
                'cond_cov_easy': cond_coverage(d['y_test'], inc0_wt, inc1_wt, easy_mask),
                'cond_cov_medium': cond_coverage(d['y_test'], inc0_wt, inc1_wt, medium_mask),
                'cond_cov_hard': cond_coverage(d['y_test'], inc0_wt, inc1_wt, hard_mask),
                'set_sizes': sizes_wt,
            },
        }
        adaptive_results[method][alpha] = result

        for conf_method in ['standard', 'mondrian', 'weighted']:
            r = result[conf_method]
            print(f"    {conf_method:12s}: cov={r['coverage']:.3f} (target={1-alpha:.2f}), "
                  f"avg_size={r['avg_set_size']:.3f}, singletons={r['singleton_frac']:.3f}", flush=True)
            print(f"      Conditional: easy={r['cond_cov_easy']:.3f}, "
                  f"medium={r['cond_cov_medium']:.3f}, hard={r['cond_cov_hard']:.3f}", flush=True)


# ============================================================
# Save
# ============================================================
print("\n--- Saving ---", flush=True)

# Convert set_sizes arrays for storage (keep only summary stats in the main output)
output = {'methods': methods}
for method in methods:
    output[method] = {}
    for alpha in [0.05, 0.10, 0.20]:
        output[method][alpha] = {}
        for conf_method in ['standard', 'mondrian', 'weighted']:
            r = adaptive_results[method][alpha][conf_method]
            output[method][alpha][conf_method] = {
                k: v for k, v in r.items() if k != 'set_sizes'
            }
            output[method][alpha][conf_method]['set_size_dist'] = {
                0: int((r['set_sizes'] == 0).sum()),
                1: int((r['set_sizes'] == 1).sum()),
                2: int((r['set_sizes'] == 2).sum()),
            }

with open(os.path.join(OUT_DIR, 'phase2_adaptive_conformal.pkl'), 'wb') as f:
    pickle.dump(output, f)

# Summary table
print("\n" + "=" * 70, flush=True)
print("SUMMARY: ADAPTIVE vs FIXED CONFORMAL (alpha=0.05)", flush=True)
print("=" * 70, flush=True)

rows = []
for method in ['mutual_info', 'ensemble', 'pearson']:
    for conf_method in ['standard', 'mondrian', 'weighted']:
        r = adaptive_results[method][0.05][conf_method]
        rows.append({
            'Method': method,
            'Conformal': conf_method,
            'Coverage': f"{r['coverage']:.3f}",
            'Avg_Size': f"{r['avg_set_size']:.3f}",
            'Singletons': f"{r['singleton_frac']:.3f}",
            'Cov_Easy': f"{r['cond_cov_easy']:.3f}",
            'Cov_Hard': f"{r['cond_cov_hard']:.3f}",
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False), flush=True)
df.to_csv(os.path.join(OUT_DIR, 'phase2_adaptive_conformal_summary.csv'), index=False)

print("\nDone with Phase 2 adaptive conformal.", flush=True)
