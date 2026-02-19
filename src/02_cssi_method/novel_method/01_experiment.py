#!/usr/bin/env python3
"""CSSI — Cell-State Stratified Interpretability — Synthetic Validation"""
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

def gen(n_genes=30, n_tfs=5, n_states=4, cps=100, eps=6, shared=3, sig=0.6, seed=42):
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
    return X, np.array(labels), te, tfs, tgts

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

def cssi(X, lab, tfs, tgts, mode='max'):
    states = np.unique(lab); n = len(lab)
    agg = np.zeros((len(tfs), len(tgts)))
    for s in states:
        m = lab == s
        if m.sum() < 5: continue
        C = corr(X[m], tfs, tgts)
        if mode == 'max': agg = np.maximum(agg, C)
        else: agg += (m.sum()/n) * C
    return to_edges(agg, tfs, tgts)

P("CSSI Synthetic Validation\n")

# EXP 1: Scaling
P("="*60); P("EXP 1: Scaling")
rows = []
for cps in [20, 50, 100, 200, 500]:
    for seed in range(5):
        X, lab, te, tfs, tgts = gen(cps=cps, seed=seed*100+cps)
        pf, pa = ev(pooled(X, tfs, tgts), te, tfs, tgts)
        cf, ca = ev(cssi(X, lab, tfs, tgts, 'max'), te, tfs, tgts)
        mf, ma = ev(cssi(X, lab, tfs, tgts, 'mean'), te, tfs, tgts)
        km = KMeans(n_clusters=4, random_state=seed, n_init=3).fit_predict(X)
        ef, ea = ev(cssi(X, km, tfs, tgts, 'max'), te, tfs, tgts)
        rows.append(dict(cps=cps, total=cps*4, seed=seed, pool_f1=pf, pool_auc=pa,
                        cssi_f1=cf, cssi_auc=ca, mean_f1=mf, est_f1=ef, est_auc=ea))
    P(f"  cps={cps} done")

df1 = pd.DataFrame(rows)
P("\nScaling F1:")
for tc in sorted(df1['total'].unique()):
    s = df1[df1['total']==tc]
    P(f"  N={tc:5d} | Pool: {s['pool_f1'].mean():.4f}±{s['pool_f1'].std():.4f} | "
      f"CSSI-max: {s['cssi_f1'].mean():.4f}±{s['cssi_f1'].std():.4f} | "
      f"CSSI-est: {s['est_f1'].mean():.4f}±{s['est_f1'].std():.4f}")
P("\nScaling AUROC:")
for tc in sorted(df1['total'].unique()):
    s = df1[df1['total']==tc]
    P(f"  N={tc:5d} | Pool: {s['pool_auc'].mean():.4f}±{s['pool_auc'].std():.4f} | "
      f"CSSI-max: {s['cssi_auc'].mean():.4f}±{s['cssi_auc'].std():.4f} | "
      f"CSSI-est: {s['est_auc'].mean():.4f}±{s['est_auc'].std():.4f}")
df1.to_csv(os.path.join(OUT, 'scaling_results.csv'), index=False)

# EXP 2: Heterogeneity
P("\n"+"="*60); P("EXP 2: Heterogeneity")
rows2 = []
for ns in [1, 2, 4, 6, 8]:
    for seed in range(5):
        cps = 400 // ns
        X, lab, te, tfs, tgts = gen(n_states=ns, cps=cps, eps=max(3,24//ns),
                                     seed=seed*1000+ns)
        pf, pa = ev(pooled(X, tfs, tgts), te, tfs, tgts)
        cf, ca = ev(cssi(X, lab, tfs, tgts), te, tfs, tgts)
        rows2.append(dict(n_states=ns, seed=seed, pool_f1=pf, cssi_f1=cf, pool_auc=pa, cssi_auc=ca))
    P(f"  states={ns} done")

df2 = pd.DataFrame(rows2)
P("\nHeterogeneity F1:")
for ns in sorted(df2['n_states'].unique()):
    s = df2[df2['n_states']==ns]
    P(f"  States={ns} | Pool: {s['pool_f1'].mean():.4f} | CSSI: {s['cssi_f1'].mean():.4f}")
df2.to_csv(os.path.join(OUT, 'heterogeneity_results.csv'), index=False)

# EXP 3: Clustering k
P("\n"+"="*60); P("EXP 3: Clustering k")
rows3 = []
for seed in range(5):
    X, lab, te, tfs, tgts = gen(n_states=4, cps=100, seed=seed*10000)
    pf, _ = ev(pooled(X, tfs, tgts), te, tfs, tgts)
    of1, _ = ev(cssi(X, lab, tfs, tgts), te, tfs, tgts)
    for k in [2, 3, 4, 6, 8, 12]:
        km = KMeans(n_clusters=k, random_state=seed, n_init=3).fit_predict(X)
        kf, _ = ev(cssi(X, km, tfs, tgts), te, tfs, tgts)
        rows3.append(dict(seed=seed, k=k, pool_f1=pf, oracle_f1=of1, est_f1=kf))

df3 = pd.DataFrame(rows3)
P(f"\nPooled:  {df3['pool_f1'].mean():.4f}")
P(f"Oracle:  {df3['oracle_f1'].mean():.4f}")
for k in sorted(df3['k'].unique()):
    s = df3[df3['k']==k]
    P(f"k={k:2d}:    {s['est_f1'].mean():.4f}±{s['est_f1'].std():.4f}")
df3.to_csv(os.path.join(OUT, 'clustering_results.csv'), index=False)

# EXP 4: Stats
P("\n"+"="*60); P("EXP 4: Statistical Test")
rows4 = []
for seed in range(20):
    X, lab, te, tfs, tgts = gen(cps=200, seed=seed*7777)
    pf, pa = ev(pooled(X, tfs, tgts), te, tfs, tgts)
    cf, ca = ev(cssi(X, lab, tfs, tgts), te, tfs, tgts)
    rows4.append(dict(seed=seed, pool_f1=pf, cssi_f1=cf, pool_auc=pa, cssi_auc=ca))

df4 = pd.DataFrame(rows4)
from scipy.stats import wilcoxon
w, p = wilcoxon(df4['cssi_f1'], df4['pool_f1'], alternative='greater')
P(f"\nPool F1: {df4['pool_f1'].mean():.4f} | CSSI F1: {df4['cssi_f1'].mean():.4f}")
P(f"Wilcoxon W={w:.1f}, p={p:.6f}")
w2, p2 = wilcoxon(df4['cssi_auc'], df4['pool_auc'], alternative='greater')
P(f"Pool AUC: {df4['pool_auc'].mean():.4f} | CSSI AUC: {df4['cssi_auc'].mean():.4f}")
P(f"Wilcoxon W={w2:.1f}, p={p2:.6f}")
df4.to_csv(os.path.join(OUT, 'stats_results.csv'), index=False)

P("\nDone.")
