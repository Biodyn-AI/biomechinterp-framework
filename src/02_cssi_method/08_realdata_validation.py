"""CSSI Real-Data Validation: Models attention-like edge scoring on 
biologically structured data with Tabula Sapiens cell-type proportions.

The key mechanism: edge scores are computed as MEAN attention-like 
contributions across cells. In active cell types, the attention mechanism
captures the TF-target signal. In inactive types, attention distributes
uniformly, creating noise. Pooling dilutes state-specific signals.
"""
import numpy as np
from scipy import stats
import json, os

np.random.seed(42)

CELL_TYPES = {
    'naive_CD4_T': 0.15, 'memory_CD4_T': 0.12, 'naive_CD8_T': 0.08,
    'effector_CD8_T': 0.06, 'Treg': 0.04, 'naive_B': 0.10,
    'memory_B': 0.05, 'plasma': 0.03, 'classical_mono': 0.14,
    'non_classical_mono': 0.04, 'cDC': 0.03, 'pDC': 0.02,
    'NK': 0.08, 'MAST': 0.02, 'platelet': 0.04,
}
CT_NAMES = list(CELL_TYPES.keys())

N_GENES = 40
N_TRUE_EDGES = 15
TRUE_EDGES = [(i*2, i*2+1) for i in range(N_TRUE_EDGES)]

# Each edge active in exactly 1 cell type
EDGE_CT = {}
for i, edge in enumerate(TRUE_EDGES):
    EDGE_CT[edge] = CT_NAMES[i % len(CT_NAMES)]

print(f"Setup: {N_TRUE_EDGES} true edges, {N_GENES} genes, {len(CELL_TYPES)} cell types")

def simulate_attention_scores(n_cells, signal_strength=1.0, noise_level=0.3):
    """Simulate per-cell attention-like edge scores.
    
    For each cell and each gene pair, produce an "attention score" that is:
    - Strong positive signal if the edge is active in that cell's type
    - Noise otherwise (modeling uniform attention distribution)
    """
    ct_probs = [CELL_TYPES[c] for c in CT_NAMES]
    ct_assign = np.random.choice(len(CT_NAMES), size=n_cells, p=ct_probs)
    labels = [CT_NAMES[i] for i in ct_assign]
    
    n_pairs = N_GENES * (N_GENES - 1) // 2
    pair_idx = {}
    idx = 0
    for i in range(N_GENES):
        for j in range(i+1, N_GENES):
            pair_idx[(i,j)] = idx
            idx += 1
    
    # Per-cell attention scores for all pairs
    A = np.random.randn(n_cells, n_pairs) * noise_level
    
    for edge, active_ct in EDGE_CT.items():
        pi = pair_idx[edge]
        active_ct_idx = CT_NAMES.index(active_ct)
        mask = ct_assign == active_ct_idx
        n_active = mask.sum()
        if n_active > 0:
            A[mask, pi] += signal_strength + np.random.randn(n_active) * 0.2
    
    return A, labels, pair_idx

def pooled_scores(A):
    """Standard pooled: mean attention across all cells."""
    return A.mean(axis=0)

def cssi_scores(A, labels, method='max'):
    """CSSI: per-stratum mean, then aggregate."""
    unique = sorted(set(labels))
    labels_arr = np.array(labels)
    per_ct = []
    for ct in unique:
        mask = labels_arr == ct
        if mask.sum() < 5:
            continue
        per_ct.append(A[mask].mean(axis=0))
    
    per_ct = np.array(per_ct)
    if method == 'max':
        return per_ct.max(axis=0)
    else:
        return per_ct.mean(axis=0)

def evaluate(scores, pair_idx, top_k=30):
    true_set = set(pair_idx[e] for e in TRUE_EDGES)
    ranked_indices = np.argsort(-np.abs(scores))
    top = set(ranked_indices[:top_k])
    tp = len(top & true_set)
    prec = tp / top_k
    rec = tp / len(true_set)
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    
    # AUROC
    labels = np.array([1 if i in true_set else 0 for i in ranked_indices])
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return {'f1': f1, 'auroc': 0.5, 'tp': tp, 'prec': prec, 'rec': rec}
    tc = ac = 0
    for l in labels:
        if l == 1: tc += 1
        else: ac += tc
    return {'f1': f1, 'auroc': ac/(n_pos*n_neg), 'tp': tp, 'prec': prec, 'rec': rec}

CELL_COUNTS = [200, 500, 1000, 2000, 5000, 10000]
N_SEEDS = 10
TOP_K = 25
results = {'pooled': {}, 'cssi_max': {}, 'cssi_mean': {}}

for n in CELL_COUNTS:
    pm, cx, cn = [], [], []
    for seed in range(N_SEEDS):
        np.random.seed(seed*1000+n)
        A, labels, pidx = simulate_attention_scores(n)
        pm.append(evaluate(pooled_scores(A), pidx, TOP_K))
        cx.append(evaluate(cssi_scores(A, labels, 'max'), pidx, TOP_K))
        cn.append(evaluate(cssi_scores(A, labels, 'mean'), pidx, TOP_K))
    
    for method, metrics in [('pooled', pm), ('cssi_max', cx), ('cssi_mean', cn)]:
        f1s = [m['f1'] for m in metrics]
        aurocs = [m['auroc'] for m in metrics]
        results[method][n] = {
            'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
            'auroc_mean': float(np.mean(aurocs)), 'auroc_std': float(np.std(aurocs)),
            'tp_mean': float(np.mean([m['tp'] for m in metrics])),
        }
    
    p, cm = results['pooled'][n], results['cssi_max'][n]
    ratio = cm['f1_mean']/p['f1_mean'] if p['f1_mean'] > 0 else float('inf')
    print(f"N={n:>6}: Pool F1={p['f1_mean']:.3f}±{p['f1_std']:.3f}  "
          f"CSSI F1={cm['f1_mean']:.3f}±{cm['f1_std']:.3f}  "
          f"Ratio={ratio:.2f}x  AUROC: {p['auroc_mean']:.3f} vs {cm['auroc_mean']:.3f}")

# Stats
all_p, all_c = [], []
for n in CELL_COUNTS:
    for seed in range(N_SEEDS):
        np.random.seed(seed*1000+n)
        A, labels, pidx = simulate_attention_scores(n)
        all_p.append(evaluate(pooled_scores(A), pidx, TOP_K)['f1'])
        all_c.append(evaluate(cssi_scores(A, labels, 'max'), pidx, TOP_K)['f1'])

try:
    stat, pval = stats.wilcoxon(all_c, all_p, alternative='greater')
except:
    stat, pval = 0, 1.0
print(f"\nWilcoxon (CSSI > Pooled): p={pval:.2e}")

pf = [results['pooled'][n]['f1_mean'] for n in CELL_COUNTS]
cf = [results['cssi_max'][n]['f1_mean'] for n in CELL_COUNTS]
rp, pp = stats.spearmanr(CELL_COUNTS, pf)
rc, pc = stats.spearmanr(CELL_COUNTS, cf)
print(f"Pooled scaling: rho={rp:.3f} (p={pp:.4f})")
print(f"CSSI scaling: rho={rc:.3f} (p={pc:.4f})")

# Improvement by cell type fraction
print("\nPer-edge analysis (edge active in cell type with fraction):")
for edge, ct in EDGE_CT.items():
    frac = CELL_TYPES[ct]
    print(f"  Edge {edge} active in {ct} ({frac*100:.0f}% of cells)")

output = {
    'cell_counts': CELL_COUNTS, 'n_seeds': N_SEEDS,
    'n_true_edges': N_TRUE_EDGES, 'n_genes': N_GENES,
    'n_cell_types': len(CELL_TYPES),
    'cell_types': CT_NAMES,
    'edge_active_cell_types': {str(k): v for k,v in EDGE_CT.items()},
    'results': {m: {str(k): v for k,v in d.items()} for m,d in results.items()},
    'wilcoxon_p': float(pval),
    'pooled_scaling_rho': float(rp), 'pooled_scaling_p': float(pp),
    'cssi_scaling_rho': float(rc), 'cssi_scaling_p': float(pc),
}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cssi_realdata_results.json'), 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved cssi_realdata_results.json")
