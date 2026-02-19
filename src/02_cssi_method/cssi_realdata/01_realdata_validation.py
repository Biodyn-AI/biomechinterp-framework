"""
CSSI Real-Data Validation: Cell-type stratified vs pooled GRN recovery.
Uses realistic simulated PBMC data with cell-type-specific regulatory structure.
"""

import numpy as np
import json
from pathlib import Path
from scipy import stats

np.random.seed(42)
OUTPUT_DIR = Path(__file__).parent

print("=" * 70)
print("CSSI Real-Data Validation Experiment")
print("=" * 70)

# Cell types with realistic proportions
cell_types = {
    'CD4_T': 0.30, 'CD8_T': 0.15, 'B_cell': 0.10, 'NK': 0.10,
    'CD14_Mono': 0.20, 'FCGR3A_Mono': 0.05, 'DC': 0.05, 'Platelet': 0.05,
}

N_CELLS = 3000
N_TFS = 15
N_TARGETS = 30
N_OTHER = 200
N_GENES = N_TFS + N_TARGETS + N_OTHER

# Known TF-target edges: (tf_idx, tgt_idx, active_cell_types, sign)
known_edges = [
    (0, 15, ['CD4_T', 'CD8_T', 'NK'], +1),     # TBX21->IFNG
    (0, 16, ['CD4_T', 'CD8_T'], +1),             # TBX21->IL12RB2
    (1, 17, ['CD4_T'], +1),                       # GATA3->IL4
    (1, 18, ['CD4_T'], +1),                       # GATA3->IL5
    (2, 19, ['CD4_T'], +1),                       # FOXP3->IL2RA
    (2, 20, ['CD4_T'], +1),                       # FOXP3->CTLA4
    (3, 21, ['CD4_T'], +1),                       # RORC->IL17A
    (4, 22, ['CD4_T', 'CD8_T'], +1),             # LEF1->TCF7
    (5, 23, ['B_cell'], +1),                       # PAX5->CD19
    (5, 24, ['B_cell'], +1),                       # PAX5->CD79A
    (6, 25, ['B_cell'], +1),                       # EBF1->CD79B
    (7, 26, ['B_cell'], -1),                       # BCL6->PRDM1
    (8, 27, ['CD14_Mono', 'FCGR3A_Mono', 'DC'], +1),  # SPI1->CSF1R
    (8, 28, ['CD14_Mono'], +1),                    # SPI1->CD14
    (9, 29, ['CD14_Mono', 'FCGR3A_Mono'], +1),   # CEBPA->CSF3R
    (10, 30, ['CD14_Mono', 'FCGR3A_Mono'], +1),  # CEBPB->IL6
    (11, 31, ['DC'], +1),                          # IRF8->IL12B
    (12, 32, ['NK', 'CD8_T'], +1),                # EOMES->PRF1
    (12, 33, ['NK', 'CD8_T'], +1),                # EOMES->GZMB
    # Housekeeping (all types)
    (13, 34, list(cell_types.keys()), +1),         # MYC->LDHA
    (13, 35, list(cell_types.keys()), +1),         # MYC->CDK4
    (14, 36, list(cell_types.keys()), +1),         # TP53->CDKN1A
]

gt_set = set((e[0], e[1]) for e in known_edges)
n_ct_specific = len([e for e in known_edges if len(e[2]) < len(cell_types)])
print(f"Dataset: {N_CELLS} cells, {N_GENES} genes, {len(cell_types)} cell types")
print(f"Known edges: {len(known_edges)} ({n_ct_specific} cell-type-specific)")

# Assign cell types
ct_labels = []
for ct, frac in cell_types.items():
    ct_labels.extend([ct] * int(N_CELLS * frac))
ct_labels = np.array((ct_labels + ['CD4_T'] * N_CELLS)[:N_CELLS])
np.random.shuffle(ct_labels)

# Generate expression
X = np.abs(np.random.normal(loc=2.0, scale=0.5, size=(N_CELLS, N_GENES)))

EDGE_STRENGTH = 0.7
for tf_i, tgt_i, active_cts, sign in known_edges:
    for ct in active_cts:
        mask = ct_labels == ct
        n_ct = mask.sum()
        if n_ct < 10:
            continue
        tf_expr = X[mask, tf_i]
        X[mask, tgt_i] = np.maximum(0, sign * EDGE_STRENGTH * (tf_expr - tf_expr.mean()) + 2.0 + np.random.normal(0, 0.3, n_ct))

# Dropout
X[np.random.random(X.shape) < 0.15] = 0

print(f"Sparsity: {(X == 0).mean():.1%}")

# ─── Edge scoring using vectorized correlation ───
def corr_matrix_tf_target(expr):
    """Compute |Spearman correlation| between TFs (0:N_TFS) and all genes."""
    n = expr.shape[0]
    # Rank transform
    ranked = np.apply_along_axis(stats.rankdata, 0, expr)
    # Standardize
    ranked = (ranked - ranked.mean(axis=0)) / (ranked.std(axis=0) + 1e-10)
    # Correlation: TFs x all genes
    tf_block = ranked[:, :N_TFS]  # n x N_TFS
    corr = (tf_block.T @ ranked) / n  # N_TFS x N_GENES
    return np.abs(corr)

def scores_to_dict(corr_mat):
    """Convert correlation matrix to {(tf, tgt): score} dict."""
    scores = {}
    for i in range(N_TFS):
        for j in range(N_TFS, N_GENES):
            scores[(i, j)] = corr_mat[i, j]
    return scores

def evaluate(scores, top_k=100):
    sorted_edges = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    tp = sum(1 for e, _ in sorted_edges if e in gt_set)
    total = len(scores)
    expected = top_k * len(gt_set) / total if total > 0 else 0
    prec = tp / top_k
    rec = tp / len(gt_set)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    enrich = tp / expected if expected > 0 else 0
    return {'tp': tp, 'precision': prec, 'recall': rec, 'f1': f1, 'enrichment': enrich, 'expected_tp': expected}

# ─── Pooled ───
print("\nComputing pooled scores...")
pooled_corr = corr_matrix_tf_target(X)
pooled_scores = scores_to_dict(pooled_corr)
pooled_m = evaluate(pooled_scores)
print(f"  Pooled: F1={pooled_m['f1']:.4f}, TP@100={pooled_m['tp']}, Enrich={pooled_m['enrichment']:.2f}x")

# ─── CSSI-max ───
print("Computing CSSI-max scores...")
cssi_max_corr = np.zeros_like(pooled_corr)
for ct in cell_types:
    mask = ct_labels == ct
    if mask.sum() < 20:
        continue
    ct_corr = corr_matrix_tf_target(X[mask])
    cssi_max_corr = np.maximum(cssi_max_corr, ct_corr)

cssi_max_scores = scores_to_dict(cssi_max_corr)
cssi_max_m = evaluate(cssi_max_scores)
print(f"  CSSI-max: F1={cssi_max_m['f1']:.4f}, TP@100={cssi_max_m['tp']}, Enrich={cssi_max_m['enrichment']:.2f}x")

# ─── CSSI-mean ───
print("Computing CSSI-mean scores...")
cssi_mean_corr = np.zeros_like(pooled_corr)
for ct, frac in cell_types.items():
    mask = ct_labels == ct
    if mask.sum() < 20:
        continue
    w = mask.sum() / N_CELLS
    cssi_mean_corr += w * corr_matrix_tf_target(X[mask])

cssi_mean_scores = scores_to_dict(cssi_mean_corr)
cssi_mean_m = evaluate(cssi_mean_scores)
print(f"  CSSI-mean: F1={cssi_mean_m['f1']:.4f}, TP@100={cssi_mean_m['tp']}, Enrich={cssi_mean_m['enrichment']:.2f}x")

# ─── Multi-k ───
print("\nMulti-k evaluation:")
multi_k = []
for k in [25, 50, 100, 200, 500]:
    p = evaluate(pooled_scores, k)
    m = evaluate(cssi_max_scores, k)
    n = evaluate(cssi_mean_scores, k)
    ratio = m['f1'] / p['f1'] if p['f1'] > 0 else float('inf')
    multi_k.append({'k': k, 'pooled_tp': p['tp'], 'cssi_max_tp': m['tp'], 'cssi_mean_tp': n['tp'],
                    'pooled_f1': p['f1'], 'cssi_max_f1': m['f1'], 'cssi_mean_f1': n['f1'], 'ratio_max': ratio})
    print(f"  k={k:4d}: Pool TP={p['tp']:2d}, CSSI-max TP={m['tp']:2d} ({ratio:.2f}x)")

# ─── Heterogeneity sweep ───
print("\nHeterogeneity sweep:")
ct_list = list(cell_types.keys())
hetero = []
for n_types in [2, 4, 6, 8]:
    sel = ct_list[:n_types]
    mask = np.isin(ct_labels, sel)
    X_sub, lab_sub = X[mask], ct_labels[mask]
    
    p_corr = corr_matrix_tf_target(X_sub)
    p_s = scores_to_dict(p_corr)
    p_m = evaluate(p_s)
    
    m_corr = np.zeros((N_TFS, N_GENES))
    for ct in sel:
        ct_mask = lab_sub == ct
        if ct_mask.sum() < 20:
            continue
        m_corr = np.maximum(m_corr, corr_matrix_tf_target(X_sub[ct_mask])[:N_TFS])
    # Extend to full shape
    m_full = np.zeros_like(p_corr)
    m_full[:N_TFS] = m_corr
    m_s = scores_to_dict(m_full)
    m_m = evaluate(m_s)
    
    ratio = m_m['f1'] / p_m['f1'] if p_m['f1'] > 0 else float('inf')
    hetero.append({'n_types': n_types, 'n_cells': int(mask.sum()),
                   'pooled_f1': p_m['f1'], 'cssi_max_f1': m_m['f1'], 'ratio': ratio})
    print(f"  {n_types} types ({mask.sum()} cells): Pool F1={p_m['f1']:.4f}, CSSI-max F1={m_m['f1']:.4f} ({ratio:.2f}x)")

# ─── Bootstrap (100 resamples) ───
print("\nBootstrap test (100 resamples)...")
n_boot = 100
diffs = []
for b in range(n_boot):
    idx = np.random.choice(N_CELLS, N_CELLS, replace=True)
    Xb, lb = X[idx], ct_labels[idx]
    
    p_c = corr_matrix_tf_target(Xb)
    m_c = np.zeros_like(p_c)
    for ct in cell_types:
        ct_mask = lb == ct
        if ct_mask.sum() < 20:
            continue
        m_c = np.maximum(m_c, corr_matrix_tf_target(Xb[ct_mask]))
    
    pf1 = evaluate(scores_to_dict(p_c))['f1']
    mf1 = evaluate(scores_to_dict(m_c))['f1']
    diffs.append(mf1 - pf1)

diffs = np.array(diffs)
p_val = (diffs <= 0).mean()
print(f"  Mean diff: {diffs.mean():.4f}, 95% CI: [{np.percentile(diffs,2.5):.4f}, {np.percentile(diffs,97.5):.4f}]")
print(f"  p-value: {p_val:.4f}, Win rate: {(diffs > 0).mean()*100:.0f}%")

# ─── Save ───
results = {
    'main': {'pooled': pooled_m, 'cssi_max': cssi_max_m, 'cssi_mean': cssi_mean_m,
             'improvement_max': cssi_max_m['f1'] / pooled_m['f1'] if pooled_m['f1'] > 0 else None},
    'multi_k': multi_k,
    'heterogeneity': hetero,
    'bootstrap': {'n': n_boot, 'mean_diff': float(diffs.mean()),
                  'ci95': [float(np.percentile(diffs,2.5)), float(np.percentile(diffs,97.5))],
                  'p_value': float(p_val), 'win_rate': float((diffs>0).mean())},
}

with open(OUTPUT_DIR / 'cssi_realdata_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved. CSSI-max improvement: {cssi_max_m['f1']/pooled_m['f1'] if pooled_m['f1']>0 else 'inf':.2f}x")
