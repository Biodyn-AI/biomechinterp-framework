#!/usr/bin/env python3
"""
CSSI v2 — Models scaling failure: more cells come from more diverse states.
This reproduces the paper's finding that larger N degrades GRN recovery for pooled,
while CSSI mitigates it.
"""
import sys, os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

OUT = os.path.dirname(os.path.abspath(__file__))
P = lambda *a, **kw: (print(*a, **kw), sys.stdout.flush())

def gen(n_genes=30, n_tfs=5, n_states=4, cps=100, eps=6, shared=3, sig=0.5, seed=42):
    """Each state has its own active edges. Signal diluted when pooling across states."""
    rng = np.random.RandomState(seed)
    cpsl = [cps]*n_states if isinstance(cps, int) else cps
    N = sum(cpsl)
    tfs, tgts = list(range(n_tfs)), list(range(n_tfs, n_genes))
    ae = [(t,g) for t in tfs for g in tgts]
    rng.shuffle(ae)
    sh = set(ae[:shared]); rest = ae[shared:]
    se = []; idx = 0
    for s in range(n_states):
        se.append(set(rest[idx:idx+eps]) | sh); idx += eps
    te = set(); [te.update(s) for s in se]
    X = rng.randn(N, n_genes)
    labels = []; ci = 0
    for s in range(n_states):
        ns = cpsl[s]
        for (tf,tgt) in se[s]:
            X[ci:ci+ns, tgt] += sig * X[ci:ci+ns, tf]
        labels.extend([s]*ns); ci += ns
    return X, np.array(labels), te, tfs, tgts, se

def corr(X, tfs, tgts):
    R = np.apply_along_axis(stats.rankdata, 0, X)
    A = R[:, tfs]; B = R[:, tgts]
    A = (A - A.mean(0)) / (A.std(0)+1e-12)
    B = (B - B.mean(0)) / (B.std(0)+1e-12)
    return np.abs(A.T @ B / len(X))

def to_edges(C, tfs, tgts):
    e = [(tfs[i], tgts[j], C[i,j]) for i in range(len(tfs)) for j in range(len(tgts))]
    e.sort(key=lambda x: -x[2]); return e

def ev(pred, true_e, tfs, tgts):
    k = len(true_e)
    topk = set((e[0],e[1]) for e in pred[:k])
    tp = len(topk & true_e)
    p = tp/k if k else 0; r = tp/len(true_e) if true_e else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    ae = [(t,g) for t in tfs for g in tgts]
    sm = {(e[0],e[1]):e[2] for e in pred}
    yt = [1 if e in true_e else 0 for e in ae]
    ys = [sm.get(e,0) for e in ae]
    try: auc = roc_auc_score(yt, ys)
    except: auc = 0.5
    return f1, auc

def pooled(X, tfs, tgts):
    return to_edges(corr(X, tfs, tgts), tfs, tgts)

def cssi_fn(X, lab, tfs, tgts, mode='max'):
    states = np.unique(lab); n = len(lab)
    agg = np.zeros((len(tfs), len(tgts)))
    for s in states:
        m = lab == s
        if m.sum() < 5: continue
        C = corr(X[m], tfs, tgts)
        if mode == 'max': agg = np.maximum(agg, C)
        else: agg += (m.sum()/n) * C
    return to_edges(agg, tfs, tgts)

P("CSSI v2 — Scaling Failure Reproduction\n")

# KEY EXPERIMENT: More cells from more diverse states
# At N=200, we sample from 2 states (homogeneous)
# At N=1000, we sample from 8 states (heterogeneous)
# This models realistic Tabula Sapiens sampling
P("="*60)
P("Scaling with increasing heterogeneity (more cells = more states)")
P("="*60)

configs = [
    (2, 100, "N=200, 2 states"),   # 200 cells, 2 states
    (4, 100, "N=400, 4 states"),   # 400 cells, 4 states
    (6, 100, "N=600, 6 states"),   # 600 cells, 6 states
    (8, 125, "N=1000, 8 states"),  # 1000 cells, 8 states
    (10, 100, "N=1000, 10 states"), # 1000 cells, 10 states
    (12, 125, "N=1500, 12 states"), # 1500 cells, 12 states
]

rows = []
for n_states, cps, desc in configs:
    for seed in range(10):
        X, lab, te, tfs, tgts, se = gen(
            n_genes=30, n_tfs=5, n_states=n_states, cps=cps,
            eps=max(2, 12//n_states), shared=2, sig=0.5, seed=seed*100+n_states)
        total = cps * n_states
        
        pf, pa = ev(pooled(X, tfs, tgts), te, tfs, tgts)
        cf, ca = ev(cssi_fn(X, lab, tfs, tgts, 'max'), te, tfs, tgts)
        mf, ma = ev(cssi_fn(X, lab, tfs, tgts, 'mean'), te, tfs, tgts)
        
        # KMeans with true k
        km = KMeans(n_clusters=n_states, random_state=seed, n_init=3).fit_predict(X)
        ef, ea = ev(cssi_fn(X, km, tfs, tgts, 'max'), te, tfs, tgts)
        
        rows.append(dict(n_states=n_states, cps=cps, total=total, seed=seed, desc=desc,
                        pool_f1=pf, pool_auc=pa, cssi_max_f1=cf, cssi_max_auc=ca,
                        cssi_mean_f1=mf, cssi_mean_auc=ma, est_f1=ef, est_auc=ea,
                        n_true_edges=len(te)))
    P(f"  {desc} done")

df = pd.DataFrame(rows)
P("\nResults — F1 (mean ± std):")
P(f"{'Config':<25s} {'Pooled':>15s} {'CSSI-max':>15s} {'CSSI-mean':>15s} {'CSSI-est':>15s}")
P("-"*85)
for desc in [c[2] for c in configs]:
    s = df[df['desc']==desc]
    P(f"{desc:<25s} {s['pool_f1'].mean():.3f}±{s['pool_f1'].std():.3f}"
      f"   {s['cssi_max_f1'].mean():.3f}±{s['cssi_max_f1'].std():.3f}"
      f"   {s['cssi_mean_f1'].mean():.3f}±{s['cssi_mean_f1'].std():.3f}"
      f"   {s['est_f1'].mean():.3f}±{s['est_f1'].std():.3f}")

P("\nResults — AUROC (mean ± std):")
P(f"{'Config':<25s} {'Pooled':>15s} {'CSSI-max':>15s} {'CSSI-mean':>15s} {'CSSI-est':>15s}")
P("-"*85)
for desc in [c[2] for c in configs]:
    s = df[df['desc']==desc]
    P(f"{desc:<25s} {s['pool_auc'].mean():.3f}±{s['pool_auc'].std():.3f}"
      f"   {s['cssi_max_auc'].mean():.3f}±{s['cssi_max_auc'].std():.3f}"
      f"   {s['cssi_mean_auc'].mean():.3f}±{s['cssi_mean_auc'].std():.3f}"
      f"   {s['est_auc'].mean():.3f}±{s['est_auc'].std():.3f}")

# Improvement ratio
P("\nImprovement ratio (CSSI-max / Pooled):")
for desc in [c[2] for c in configs]:
    s = df[df['desc']==desc]
    ratio_f1 = s['cssi_max_f1'].mean() / max(s['pool_f1'].mean(), 1e-6)
    ratio_auc = s['cssi_max_auc'].mean() / max(s['pool_auc'].mean(), 1e-6)
    P(f"  {desc}: F1 ratio = {ratio_f1:.2f}x, AUROC ratio = {ratio_auc:.2f}x")

# Wilcoxon across all configs
from scipy.stats import wilcoxon
w, p = wilcoxon(df['cssi_max_f1'], df['pool_f1'], alternative='greater')
P(f"\nOverall Wilcoxon (CSSI-max > Pooled): W={w:.0f}, p={p:.2e}")
w2, p2 = wilcoxon(df['cssi_max_auc'], df['pool_auc'], alternative='greater')
P(f"Overall Wilcoxon AUROC: W={w2:.0f}, p={p2:.2e}")

# Scaling trend: does pooled degrade?
P("\n--- Scaling trend (more states = worse pooled?) ---")
from scipy.stats import spearmanr
r_pool, p_pool = spearmanr(df['n_states'], df['pool_f1'])
r_cssi, p_cssi = spearmanr(df['n_states'], df['cssi_max_f1'])
P(f"Spearman(n_states, pool_f1):     r={r_pool:.3f}, p={p_pool:.4f}")
P(f"Spearman(n_states, cssi_max_f1): r={r_cssi:.3f}, p={p_cssi:.4f}")

df.to_csv(os.path.join(OUT, 'scaling_v2_results.csv'), index=False)
P("\nDone.")
