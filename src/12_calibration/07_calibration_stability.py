"""
Phase 2, Objective 4: Calibration stability via bootstrap.

Bootstrap the calibration procedure itself to assess how stable the
calibrated probabilities are. This answers: if we re-sample the calibration
set, how much do the calibrated probabilities vary?

Strategy:
- Bootstrap the calibration set (resample with replacement)
- Refit Platt/isotonic on each bootstrap replicate
- Apply to the same test set
- Report: mean, std, 95% CI of calibrated probabilities
- Report: stability of ECE, Brier across bootstrap replicates
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

OUT_DIR = '/Users/ihorkendiukhov/biodyn-work/subproject_31_uncertainty_calibration_edge_scores/implementation'
N_BOOTSTRAP = 200


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


def normalize_with_range(scores, mn, mx):
    if mx - mn < 1e-12:
        return np.full_like(scores, 0.5)
    return np.clip((scores - mn) / (mx - mn), 0, 1)


# ============================================================
# Load data
# ============================================================
print("=" * 70, flush=True)
print("PHASE 2: CALIBRATION STABILITY (BOOTSTRAP)", flush=True)
print("=" * 70, flush=True)

with open(os.path.join(OUT_DIR, 'edge_scores_and_ground_truth.pkl'), 'rb') as f:
    data = pickle.load(f)

edge_scores = data['edge_scores']
ground_truth = data['ground_truth']
methods = data['methods']
perturbations = data['perturbations']

# Same split as Phase 1
perts = sorted(set(perturbations))
np.random.shuffle(perts)
n = len(perts)
n_train = int(0.5 * n)
n_cal = int(0.25 * n)
perts_train = set(perts[:n_train])
perts_cal = set(perts[n_train:n_train + n_cal])
perts_test = set(perts[n_train + n_cal:])

method_data = {}
for method in methods:
    scores_dict = edge_scores[method]
    y_all, s_all, split_all = [], [], []
    for (pert, target), score in scores_dict.items():
        if (pert, target) not in ground_truth:
            continue
        y_all.append(int(ground_truth[(pert, target)]))
        s_all.append(score)
        if pert in perts_train:
            split_all.append('train')
        elif pert in perts_cal:
            split_all.append('cal')
        else:
            split_all.append('test')

    y_all = np.array(y_all)
    s_all = np.array(s_all)
    split_all = np.array(split_all)

    method_data[method] = {
        'y_train': y_all[split_all == 'train'],
        's_train': s_all[split_all == 'train'],
        'y_cal': y_all[split_all == 'cal'],
        's_cal': s_all[split_all == 'cal'],
        'y_test': y_all[split_all == 'test'],
        's_test': s_all[split_all == 'test'],
    }

print(f"  Methods: {methods}", flush=True)
print(f"  Bootstrap replicates: {N_BOOTSTRAP}", flush=True)

# ============================================================
# Bootstrap calibration
# ============================================================
stability_results = {}

for method in methods:
    print(f"\n--- {method} ---", flush=True)
    d = method_data[method]

    all_s = np.concatenate([d['s_train'], d['s_cal'], d['s_test']])
    mn, mx = all_s.min(), all_s.max()

    s_cal_n = normalize_with_range(d['s_cal'], mn, mx)
    s_test_n = normalize_with_range(d['s_test'], mn, mx)

    n_cal = len(d['y_cal'])
    n_test = len(d['y_test'])

    # Arrays to store bootstrap results
    platt_probs_boot = np.zeros((N_BOOTSTRAP, n_test))
    iso_probs_boot = np.zeros((N_BOOTSTRAP, n_test))
    platt_ece_boot = np.zeros(N_BOOTSTRAP)
    platt_brier_boot = np.zeros(N_BOOTSTRAP)
    iso_ece_boot = np.zeros(N_BOOTSTRAP)
    iso_brier_boot = np.zeros(N_BOOTSTRAP)

    for b in range(N_BOOTSTRAP):
        # Bootstrap resample calibration set
        boot_idx = np.random.choice(n_cal, n_cal, replace=True)
        y_cal_boot = d['y_cal'][boot_idx]
        s_cal_boot = s_cal_n[boot_idx]

        # Skip if only one class in bootstrap sample
        if len(np.unique(y_cal_boot)) < 2:
            platt_probs_boot[b] = np.nan
            iso_probs_boot[b] = np.nan
            platt_ece_boot[b] = np.nan
            platt_brier_boot[b] = np.nan
            iso_ece_boot[b] = np.nan
            iso_brier_boot[b] = np.nan
            continue

        # Platt
        platt = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        platt.fit(s_cal_boot.reshape(-1, 1), y_cal_boot)
        probs_platt = platt.predict_proba(s_test_n.reshape(-1, 1))[:, 1]
        platt_probs_boot[b] = probs_platt
        platt_ece_boot[b] = compute_ece(d['y_test'], probs_platt)
        platt_brier_boot[b] = brier_score_loss(d['y_test'], probs_platt)

        # Isotonic
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        iso.fit(s_cal_boot, y_cal_boot)
        probs_iso = iso.predict(s_test_n)
        iso_probs_boot[b] = probs_iso
        iso_ece_boot[b] = compute_ece(d['y_test'], probs_iso)
        iso_brier_boot[b] = brier_score_loss(d['y_test'], probs_iso)

    # Remove NaN bootstrap replicates
    valid_boot = ~np.isnan(platt_probs_boot[:, 0])
    n_valid = valid_boot.sum()
    print(f"  Valid bootstrap replicates: {n_valid}/{N_BOOTSTRAP}", flush=True)

    platt_probs_valid = platt_probs_boot[valid_boot]
    iso_probs_valid = iso_probs_boot[valid_boot]

    # Statistics of calibrated probabilities
    platt_mean = platt_probs_valid.mean(axis=0)
    platt_std = platt_probs_valid.std(axis=0)
    platt_ci_lo = np.percentile(platt_probs_valid, 2.5, axis=0)
    platt_ci_hi = np.percentile(platt_probs_valid, 97.5, axis=0)

    iso_mean = iso_probs_valid.mean(axis=0)
    iso_std = iso_probs_valid.std(axis=0)
    iso_ci_lo = np.percentile(iso_probs_valid, 2.5, axis=0)
    iso_ci_hi = np.percentile(iso_probs_valid, 97.5, axis=0)

    # ECE stability
    platt_ece_valid = platt_ece_boot[valid_boot]
    iso_ece_valid = iso_ece_boot[valid_boot]
    platt_brier_valid = platt_brier_boot[valid_boot]
    iso_brier_valid = iso_brier_boot[valid_boot]

    stability_results[method] = {
        'platt': {
            'prob_mean': platt_mean,
            'prob_std': platt_std,
            'prob_ci_lo': platt_ci_lo,
            'prob_ci_hi': platt_ci_hi,
            'mean_std': float(platt_std.mean()),
            'median_std': float(np.median(platt_std)),
            'max_std': float(platt_std.max()),
            'mean_ci_width': float((platt_ci_hi - platt_ci_lo).mean()),
            'ece_mean': float(platt_ece_valid.mean()),
            'ece_std': float(platt_ece_valid.std()),
            'ece_ci': (float(np.percentile(platt_ece_valid, 2.5)),
                       float(np.percentile(platt_ece_valid, 97.5))),
            'brier_mean': float(platt_brier_valid.mean()),
            'brier_std': float(platt_brier_valid.std()),
            'brier_ci': (float(np.percentile(platt_brier_valid, 2.5)),
                         float(np.percentile(platt_brier_valid, 97.5))),
        },
        'isotonic': {
            'prob_mean': iso_mean,
            'prob_std': iso_std,
            'prob_ci_lo': iso_ci_lo,
            'prob_ci_hi': iso_ci_hi,
            'mean_std': float(iso_std.mean()),
            'median_std': float(np.median(iso_std)),
            'max_std': float(iso_std.max()),
            'mean_ci_width': float((iso_ci_hi - iso_ci_lo).mean()),
            'ece_mean': float(iso_ece_valid.mean()),
            'ece_std': float(iso_ece_valid.std()),
            'ece_ci': (float(np.percentile(iso_ece_valid, 2.5)),
                       float(np.percentile(iso_ece_valid, 97.5))),
            'brier_mean': float(iso_brier_valid.mean()),
            'brier_std': float(iso_brier_valid.std()),
            'brier_ci': (float(np.percentile(iso_brier_valid, 2.5)),
                         float(np.percentile(iso_brier_valid, 97.5))),
        },
        'n_valid_boot': int(n_valid),
    }

    print(f"  Platt: prob_std mean={platt_std.mean():.4f}, "
          f"ECE={platt_ece_valid.mean():.4f} +/- {platt_ece_valid.std():.4f}", flush=True)
    print(f"  Isotonic: prob_std mean={iso_std.mean():.4f}, "
          f"ECE={iso_ece_valid.mean():.4f} +/- {iso_ece_valid.std():.4f}", flush=True)

# ============================================================
# Per-edge uncertainty categorization
# ============================================================
print("\n" + "=" * 70, flush=True)
print("PER-EDGE CALIBRATION UNCERTAINTY ANALYSIS", flush=True)
print("=" * 70, flush=True)

edge_uncertainty = {}

for method in ['mutual_info', 'ensemble']:
    r = stability_results[method]
    iso_std = r['isotonic']['prob_std']
    iso_mean = r['isotonic']['prob_mean']

    # Categorize edges by calibration uncertainty
    # Low uncertainty: std < 0.05, Medium: 0.05-0.10, High: > 0.10
    low_mask = iso_std < 0.05
    med_mask = (iso_std >= 0.05) & (iso_std < 0.10)
    high_mask = iso_std >= 0.10

    y_test = method_data[method]['y_test']

    # ECE by uncertainty group
    for group_name, mask in [('low', low_mask), ('medium', med_mask), ('high', high_mask)]:
        if mask.sum() > 10:
            group_ece = compute_ece(y_test[mask], iso_mean[mask])
        else:
            group_ece = float('nan')

        print(f"  {method} - {group_name} uncertainty: "
              f"n={mask.sum()}, ECE={group_ece:.4f}, "
              f"mean_std={iso_std[mask].mean():.4f}", flush=True)

    # Correlation between uncertainty and calibrated probability
    # More uncertain near 0.5, less near 0 or 1?
    distance_from_05 = np.abs(iso_mean - 0.5)
    corr_coef = np.corrcoef(distance_from_05, iso_std)[0, 1]
    print(f"  {method}: corr(|prob-0.5|, std) = {corr_coef:.3f}", flush=True)

    edge_uncertainty[method] = {
        'std': iso_std,
        'mean': iso_mean,
        'n_low': int(low_mask.sum()),
        'n_medium': int(med_mask.sum()),
        'n_high': int(high_mask.sum()),
        'corr_prob_std': float(corr_coef),
    }

# ============================================================
# Save
# ============================================================
print("\n--- Saving ---", flush=True)

# Store summary stats (not full bootstrap arrays, to save space)
output = {}
for method in methods:
    r = stability_results[method]
    output[method] = {
        'platt': {k: v for k, v in r['platt'].items()
                  if k not in ['prob_mean', 'prob_std', 'prob_ci_lo', 'prob_ci_hi']},
        'isotonic': {k: v for k, v in r['isotonic'].items()
                     if k not in ['prob_mean', 'prob_std', 'prob_ci_lo', 'prob_ci_hi']},
        'n_valid_boot': r['n_valid_boot'],
    }

# Keep full arrays for key methods (for figures)
for method in ['mutual_info', 'ensemble']:
    r = stability_results[method]
    output[method]['isotonic']['prob_std'] = r['isotonic']['prob_std'].tolist()
    output[method]['isotonic']['prob_mean'] = r['isotonic']['prob_mean'].tolist()
    output[method]['isotonic']['prob_ci_lo'] = r['isotonic']['prob_ci_lo'].tolist()
    output[method]['isotonic']['prob_ci_hi'] = r['isotonic']['prob_ci_hi'].tolist()

output['edge_uncertainty'] = edge_uncertainty
output['methods'] = methods
output['n_bootstrap'] = N_BOOTSTRAP

with open(os.path.join(OUT_DIR, 'phase2_calibration_stability.pkl'), 'wb') as f:
    pickle.dump(output, f)

# Summary table
print("\n" + "=" * 70, flush=True)
print("SUMMARY: CALIBRATION STABILITY", flush=True)
print("=" * 70, flush=True)

rows = []
for method in methods:
    r = stability_results[method]
    rows.append({
        'Method': method,
        'Platt_prob_std': f"{r['platt']['mean_std']:.4f}",
        'Platt_ECE': f"{r['platt']['ece_mean']:.4f} +/- {r['platt']['ece_std']:.4f}",
        'Platt_ECE_95CI': f"[{r['platt']['ece_ci'][0]:.4f}, {r['platt']['ece_ci'][1]:.4f}]",
        'Iso_prob_std': f"{r['isotonic']['mean_std']:.4f}",
        'Iso_ECE': f"{r['isotonic']['ece_mean']:.4f} +/- {r['isotonic']['ece_std']:.4f}",
        'Iso_ECE_95CI': f"[{r['isotonic']['ece_ci'][0]:.4f}, {r['isotonic']['ece_ci'][1]:.4f}]",
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False), flush=True)
df.to_csv(os.path.join(OUT_DIR, 'phase2_calibration_stability_summary.csv'), index=False)

print("\nDone with Phase 2 calibration stability.", flush=True)
