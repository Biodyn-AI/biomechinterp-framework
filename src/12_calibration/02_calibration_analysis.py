"""
Step 2: Calibration analysis of edge scores.

- Split data into train/calibration/test (50/25/25)
- Pre-calibration reliability diagrams
- Apply isotonic regression and Platt scaling calibration
- Compute ECE, MCE, Brier score
- Conformal prediction sets with coverage guarantees
- Threshold policies under different utility functions
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, roc_auc_score, log_loss,
                             precision_recall_curve, average_precision_score)
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = '/Users/ihorkendiukhov/biodyn-work/subproject_31_uncertainty_calibration_edge_scores/implementation'
np.random.seed(42)

# ============================================================
# Load data
# ============================================================
print("Loading edge scores and ground truth...", flush=True)
with open(os.path.join(OUT_DIR, 'edge_scores_and_ground_truth.pkl'), 'rb') as f:
    data = pickle.load(f)

edge_scores = data['edge_scores']
ground_truth = data['ground_truth']
methods = data['methods']
perturbations = data['perturbations']

print(f"  Methods: {methods}", flush=True)
print(f"  Total ground truth edges: {len(ground_truth)}", flush=True)

# ============================================================
# Build per-method arrays and split
# ============================================================
print("\nBuilding arrays and splitting...", flush=True)

# Stratified split by perturbation (so no TF leaks between sets)
perts = sorted(set(perturbations))
np.random.shuffle(perts)
n = len(perts)
n_train = int(0.5 * n)
n_cal = int(0.25 * n)
perts_train = set(perts[:n_train])
perts_cal = set(perts[n_train:n_train + n_cal])
perts_test = set(perts[n_train + n_cal:])
print(f"  Perturbation split: train={len(perts_train)}, cal={len(perts_cal)}, test={len(perts_test)}", flush=True)

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
        'y': y_all,
        'scores': s_all,
        'split': split_all,
        'y_train': y_all[split_all == 'train'],
        's_train': s_all[split_all == 'train'],
        'y_cal': y_all[split_all == 'cal'],
        's_cal': s_all[split_all == 'cal'],
        'y_test': y_all[split_all == 'test'],
        's_test': s_all[split_all == 'test'],
    }
    print(f"  {method}: train={sum(split_all=='train')}, cal={sum(split_all=='cal')}, "
          f"test={sum(split_all=='test')}", flush=True)

# ============================================================
# Pre-calibration metrics
# ============================================================
print("\n" + "=" * 60, flush=True)
print("PRE-CALIBRATION ANALYSIS", flush=True)
print("=" * 60, flush=True)


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
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


def compute_mce(y_true, y_prob, n_bins=10):
    """Maximum Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    mce = 0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        mce = max(mce, abs(bin_acc - bin_conf))
    return mce


def normalize_scores(scores):
    """Min-max normalize to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-12:
        return np.full_like(scores, 0.5)
    return (scores - mn) / (mx - mn)


pre_cal_metrics = {}
for method in methods:
    d = method_data[method]
    # Normalize scores to [0, 1] for calibration analysis
    s_norm_test = normalize_scores(d['s_test'])

    try:
        auc = roc_auc_score(d['y_test'], d['s_test'])
    except:
        auc = float('nan')

    ece = compute_ece(d['y_test'], s_norm_test)
    mce = compute_mce(d['y_test'], s_norm_test)
    brier = brier_score_loss(d['y_test'], s_norm_test)

    # Reliability diagram data
    try:
        prob_true, prob_pred = calibration_curve(d['y_test'], s_norm_test, n_bins=10, strategy='uniform')
    except:
        prob_true, prob_pred = np.array([]), np.array([])

    pre_cal_metrics[method] = {
        'auroc': auc,
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'reliability_true': prob_true,
        'reliability_pred': prob_pred,
    }
    print(f"  {method}: AUROC={auc:.4f}, ECE={ece:.4f}, MCE={mce:.4f}, Brier={brier:.4f}", flush=True)

# ============================================================
# Calibration methods
# ============================================================
print("\n" + "=" * 60, flush=True)
print("APPLYING CALIBRATION", flush=True)
print("=" * 60, flush=True)

calibrated_results = {}

for method in methods:
    d = method_data[method]
    s_train_norm = normalize_scores(d['s_train'])
    s_cal_norm = normalize_scores(d['s_cal'])

    # Use full training range for normalization of test
    all_scores = np.concatenate([d['s_train'], d['s_cal'], d['s_test']])
    mn, mx = all_scores.min(), all_scores.max()
    if mx - mn < 1e-12:
        s_test_norm = np.full_like(d['s_test'], 0.5)
        s_cal_norm2 = np.full_like(d['s_cal'], 0.5)
    else:
        s_test_norm = (d['s_test'] - mn) / (mx - mn)
        s_cal_norm2 = (d['s_cal'] - mn) / (mx - mn)

    calibrated_results[method] = {
        'raw_test': s_test_norm,
        'y_test': d['y_test'],
    }

    # --- Platt scaling (logistic regression) ---
    print(f"\n  {method} - Platt scaling...", flush=True)
    platt = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    platt.fit(s_cal_norm2.reshape(-1, 1), d['y_cal'])
    platt_probs_test = platt.predict_proba(s_test_norm.reshape(-1, 1))[:, 1]

    try:
        platt_auc = roc_auc_score(d['y_test'], platt_probs_test)
    except:
        platt_auc = float('nan')
    platt_ece = compute_ece(d['y_test'], platt_probs_test)
    platt_mce = compute_mce(d['y_test'], platt_probs_test)
    platt_brier = brier_score_loss(d['y_test'], platt_probs_test)

    try:
        pt, pp = calibration_curve(d['y_test'], platt_probs_test, n_bins=10, strategy='uniform')
    except:
        pt, pp = np.array([]), np.array([])

    calibrated_results[method]['platt'] = {
        'probs': platt_probs_test,
        'auroc': platt_auc,
        'ece': platt_ece,
        'mce': platt_mce,
        'brier': platt_brier,
        'reliability_true': pt,
        'reliability_pred': pp,
    }
    print(f"    AUROC={platt_auc:.4f}, ECE={platt_ece:.4f}, MCE={platt_mce:.4f}, "
          f"Brier={platt_brier:.4f}", flush=True)

    # --- Isotonic regression ---
    print(f"  {method} - Isotonic regression...", flush=True)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso.fit(s_cal_norm2, d['y_cal'])
    iso_probs_test = iso.predict(s_test_norm)

    try:
        iso_auc = roc_auc_score(d['y_test'], iso_probs_test)
    except:
        iso_auc = float('nan')
    iso_ece = compute_ece(d['y_test'], iso_probs_test)
    iso_mce = compute_mce(d['y_test'], iso_probs_test)
    iso_brier = brier_score_loss(d['y_test'], iso_probs_test)

    try:
        it, ip = calibration_curve(d['y_test'], iso_probs_test, n_bins=10, strategy='uniform')
    except:
        it, ip = np.array([]), np.array([])

    calibrated_results[method]['isotonic'] = {
        'probs': iso_probs_test,
        'auroc': iso_auc,
        'ece': iso_ece,
        'mce': iso_mce,
        'brier': iso_brier,
        'reliability_true': it,
        'reliability_pred': ip,
    }
    print(f"    AUROC={iso_auc:.4f}, ECE={iso_ece:.4f}, MCE={iso_mce:.4f}, "
          f"Brier={iso_brier:.4f}", flush=True)

# ============================================================
# Conformal prediction sets
# ============================================================
print("\n" + "=" * 60, flush=True)
print("CONFORMAL PREDICTION SETS", flush=True)
print("=" * 60, flush=True)

conformal_results = {}

for method in methods:
    d = method_data[method]
    # Use calibration set for conformal calibration
    all_scores = np.concatenate([d['s_train'], d['s_cal'], d['s_test']])
    mn, mx = all_scores.min(), all_scores.max()
    s_cal_norm = (d['s_cal'] - mn) / (mx - mn) if mx - mn > 1e-12 else np.full_like(d['s_cal'], 0.5)
    s_test_norm = (d['s_test'] - mn) / (mx - mn) if mx - mn > 1e-12 else np.full_like(d['s_test'], 0.5)

    conformal_results[method] = {}

    for alpha in [0.05, 0.10, 0.20]:  # target miscoverage rate
        # Split conformal: compute nonconformity scores on cal set
        # Score = 1 - predicted_prob if y=1, predicted_prob if y=0
        # Use isotonic calibrated probabilities
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        # Fit on training portion
        s_train_norm = (d['s_train'] - mn) / (mx - mn) if mx - mn > 1e-12 else np.full_like(d['s_train'], 0.5)
        iso.fit(s_train_norm, d['y_train'])

        cal_probs = iso.predict(s_cal_norm)
        test_probs = iso.predict(s_test_norm)

        # Nonconformity scores on calibration set
        n_cal = len(d['y_cal'])
        nonconf = np.where(d['y_cal'] == 1, 1 - cal_probs, cal_probs)

        # Compute quantile threshold
        q = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
        q = min(q, 1.0)
        threshold = np.quantile(nonconf, q)

        # Build prediction sets on test
        # For each test point, include class 1 if 1 - prob <= threshold
        # Include class 0 if prob <= threshold
        include_1 = (1 - test_probs) <= threshold
        include_0 = test_probs <= threshold

        # Prediction set sizes
        set_sizes = include_1.astype(int) + include_0.astype(int)

        # Coverage
        covered = np.zeros(len(d['y_test']), dtype=bool)
        for i in range(len(d['y_test'])):
            if d['y_test'][i] == 1 and include_1[i]:
                covered[i] = True
            elif d['y_test'][i] == 0 and include_0[i]:
                covered[i] = True

        coverage = covered.mean()
        avg_set_size = set_sizes.mean()
        empty_sets = (set_sizes == 0).mean()
        singleton_sets = (set_sizes == 1).mean()

        conformal_results[method][alpha] = {
            'coverage': coverage,
            'avg_set_size': avg_set_size,
            'set_sizes': set_sizes,
            'empty_frac': empty_sets,
            'singleton_frac': singleton_sets,
            'threshold': threshold,
        }

        print(f"  {method} (alpha={alpha}): coverage={coverage:.4f} (target={1-alpha:.2f}), "
              f"avg_set_size={avg_set_size:.3f}, singletons={singleton_sets:.3f}", flush=True)

# ============================================================
# Threshold policies
# ============================================================
print("\n" + "=" * 60, flush=True)
print("OPTIMAL THRESHOLD POLICIES", flush=True)
print("=" * 60, flush=True)

threshold_results = {}

for method in methods:
    d = method_data[method]
    all_scores = np.concatenate([d['s_train'], d['s_cal'], d['s_test']])
    mn, mx = all_scores.min(), all_scores.max()
    s_test_norm = (d['s_test'] - mn) / (mx - mn) if mx - mn > 1e-12 else np.full_like(d['s_test'], 0.5)

    # Use isotonic-calibrated probs
    s_train_norm = (d['s_train'] - mn) / (mx - mn) if mx - mn > 1e-12 else np.full_like(d['s_train'], 0.5)
    s_cal_norm = (d['s_cal'] - mn) / (mx - mn) if mx - mn > 1e-12 else np.full_like(d['s_cal'], 0.5)

    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso.fit(np.concatenate([s_train_norm, s_cal_norm]),
            np.concatenate([d['y_train'], d['y_cal']]))
    probs_test = iso.predict(s_test_norm)

    threshold_results[method] = {}

    # Different utility functions
    # 1. Balanced accuracy (Youden's J)
    thresholds = np.linspace(0, 1, 100)
    best_j, best_t_j = -1, 0.5
    for t in thresholds:
        pred = (probs_test >= t).astype(int)
        tp = ((pred == 1) & (d['y_test'] == 1)).sum()
        tn = ((pred == 0) & (d['y_test'] == 0)).sum()
        fn = ((pred == 0) & (d['y_test'] == 1)).sum()
        fp = ((pred == 1) & (d['y_test'] == 0)).sum()
        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_t_j = t

    # 2. High precision (FDR < 0.1)
    best_t_prec = 1.0
    for t in thresholds[::-1]:
        pred = (probs_test >= t).astype(int)
        tp = ((pred == 1) & (d['y_test'] == 1)).sum()
        fp = ((pred == 1) & (d['y_test'] == 0)).sum()
        prec = tp / (tp + fp + 1e-12)
        if prec >= 0.9:
            best_t_prec = t
            break

    # 3. High recall (>0.8)
    best_t_recall = 0.0
    for t in thresholds:
        pred = (probs_test >= t).astype(int)
        tp = ((pred == 1) & (d['y_test'] == 1)).sum()
        fn = ((pred == 0) & (d['y_test'] == 1)).sum()
        recall = tp / (tp + fn + 1e-12)
        if recall >= 0.8:
            best_t_recall = t
        else:
            break

    threshold_results[method] = {
        'youden': {'threshold': best_t_j, 'J': best_j},
        'high_precision': {'threshold': best_t_prec},
        'high_recall': {'threshold': best_t_recall},
    }

    # Evaluate at Youden threshold
    pred_j = (probs_test >= best_t_j).astype(int)
    tp = ((pred_j == 1) & (d['y_test'] == 1)).sum()
    fp = ((pred_j == 1) & (d['y_test'] == 0)).sum()
    fn = ((pred_j == 0) & (d['y_test'] == 1)).sum()
    tn = ((pred_j == 0) & (d['y_test'] == 0)).sum()
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)

    print(f"  {method}:", flush=True)
    print(f"    Youden threshold={best_t_j:.3f}: prec={prec:.3f}, recall={rec:.3f}, F1={f1:.3f}", flush=True)
    print(f"    High-precision threshold={best_t_prec:.3f}", flush=True)
    print(f"    High-recall threshold={best_t_recall:.3f}", flush=True)

# ============================================================
# Save all calibration results
# ============================================================
print("\nSaving calibration results...", flush=True)
results = {
    'pre_cal_metrics': pre_cal_metrics,
    'calibrated_results': calibrated_results,
    'conformal_results': conformal_results,
    'threshold_results': threshold_results,
    'method_data': {m: {k: v for k, v in d.items() if k != 'y' and k != 'scores' and k != 'split'}
                    for m, d in method_data.items()},
    'methods': methods,
    'split_info': {
        'n_train_perts': len(perts_train),
        'n_cal_perts': len(perts_cal),
        'n_test_perts': len(perts_test),
    }
}

with open(os.path.join(OUT_DIR, 'calibration_results.pkl'), 'wb') as f:
    pickle.dump(results, f)

# Summary table
print("\n" + "=" * 60, flush=True)
print("SUMMARY: PRE- vs POST-CALIBRATION", flush=True)
print("=" * 60, flush=True)

rows = []
for method in methods:
    pre = pre_cal_metrics[method]
    cal_res = calibrated_results[method]

    row = {
        'Method': method,
        'AUROC': pre['auroc'],
        'Raw ECE': pre['ece'],
        'Raw Brier': pre['brier'],
        'Platt ECE': cal_res['platt']['ece'],
        'Platt Brier': cal_res['platt']['brier'],
        'Isotonic ECE': cal_res['isotonic']['ece'],
        'Isotonic Brier': cal_res['isotonic']['brier'],
    }
    rows.append(row)

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False), flush=True)
summary_df.to_csv(os.path.join(OUT_DIR, 'calibration_summary.csv'), index=False)

print("\nDone with Step 2.", flush=True)
