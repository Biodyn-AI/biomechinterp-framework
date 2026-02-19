"""
Revised Statistical Analysis for Cell-State-Conditional Circuits
================================================================
Uses Fisher z-transform test to compare within-regime correlations to global,
plus a within-regime permutation test (shuffle gene labels within each regime).
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse, stats
from scipy.stats import spearmanr, norm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings, time
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR = "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw"
PERTURB_DIR = "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/perturb"
OUT_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_28_cell_state_conditional_circuits"
FIG_DIR = os.path.join(OUT_DIR, "reports", "figures")
IMPL_DIR = os.path.join(OUT_DIR, "implementation")

os.makedirs(FIG_DIR, exist_ok=True)
N_PERMS = 500
N_BOOTS = 200
MIN_CELLS = 50

t0 = time.time()

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
adata = sc.read_h5ad(os.path.join(DATA_DIR, "tabula_sapiens_immune_subset_20000.h5ad"))
adata.obs['regime'] = adata.obs['cell_type'].astype(str)

regime_counts = adata.obs['regime'].value_counts()
valid_regimes = regime_counts[regime_counts >= 100].index.tolist()
adata = adata[adata.obs['regime'].isin(valid_regimes)].copy()
regimes = adata.obs['regime'].values
unique_regimes = sorted(set(regimes))
print(f"Data: {adata.shape[0]} cells, {len(unique_regimes)} regimes")

# Gene symbol mapping
symbol_to_ensembl = {}
for ens_id, row in adata.var.iterrows():
    sym = row['feature_name']
    if pd.notna(sym):
        symbol_to_ensembl[sym] = ens_id

# TF-target pairs
KNOWN_TF_TARGETS = [
    ("NFKB1", "IL6", "NF-kB"), ("NFKB1", "TNF", "NF-kB"),
    ("NFKB1", "IL1B", "NF-kB"), ("NFKB1", "CXCL8", "NF-kB"),
    ("NFKB1", "CCL2", "NF-kB"), ("RELA", "IL6", "NF-kB"),
    ("RELA", "TNF", "NF-kB"), ("RELA", "BCL2", "NF-kB"),
    ("RELA", "ICAM1", "NF-kB"),
    ("STAT1", "IRF1", "JAK-STAT"), ("STAT1", "GBP1", "JAK-STAT"),
    ("STAT1", "CXCL10", "JAK-STAT"), ("STAT1", "IDO1", "JAK-STAT"),
    ("STAT3", "BCL2", "JAK-STAT"), ("STAT3", "MYC", "JAK-STAT"),
    ("STAT3", "SOCS3", "JAK-STAT"), ("STAT3", "VEGFA", "JAK-STAT"),
    ("STAT4", "IFNG", "JAK-STAT"), ("STAT5A", "IL2RA", "JAK-STAT"),
    ("STAT5A", "BCL2L1", "JAK-STAT"), ("STAT6", "GATA3", "JAK-STAT"),
    ("STAT6", "IL4R", "JAK-STAT"),
    ("TBX21", "IFNG", "Th1"), ("TBX21", "CXCR3", "Th1"), ("TBX21", "IL12RB2", "Th1"),
    ("GATA3", "IL5", "Th2"), ("GATA3", "IL13", "Th2"), ("GATA3", "IL4", "Th2"),
    ("RORC", "IL17A", "Th17"), ("RORC", "IL17F", "Th17"), ("RORC", "CCR6", "Th17"),
    ("FOXP3", "IL2RA", "Treg"), ("FOXP3", "CTLA4", "Treg"), ("FOXP3", "TNFRSF18", "Treg"),
    ("BCL6", "CXCR5", "Tfh"),
    ("PAX5", "CD19", "B cell"), ("PAX5", "CD79A", "B cell"), ("PAX5", "MS4A1", "B cell"),
    ("EBF1", "CD79A", "B cell"), ("EBF1", "CD79B", "B cell"),
    ("IRF4", "PRDM1", "Plasma cell"), ("IRF4", "XBP1", "Plasma cell"),
    ("PRDM1", "XBP1", "Plasma cell"), ("PRDM1", "SDC1", "Plasma cell"),
    ("SPI1", "CSF1R", "Myeloid"), ("SPI1", "CD68", "Myeloid"), ("SPI1", "ITGAM", "Myeloid"),
    ("CEBPA", "CSF3R", "Myeloid"), ("CEBPA", "MPO", "Granulocyte"),
    ("CEBPB", "IL6", "Myeloid"), ("CEBPB", "CSF2", "Myeloid"),
    ("FOS", "IL2", "AP-1"), ("FOS", "MMP9", "AP-1"),
    ("JUN", "IL2", "AP-1"), ("JUN", "VEGFA", "AP-1"),
    ("IRF1", "CXCL10", "IFN"), ("IRF1", "GBP1", "IFN"),
    ("IRF7", "IFNB1", "IFN"), ("IRF8", "IL12B", "DC"),
    ("MYC", "CDK4", "Cell cycle"), ("MYC", "LDHA", "MYC-met"),
    ("MYC", "NPM1", "MYC-prolif"), ("MYC", "NCL", "MYC-prolif"),
    ("RUNX3", "CD8A", "CD8 T cell"), ("RUNX3", "GZMB", "Cytotoxic"),
    ("EOMES", "GZMB", "Cytotoxic"), ("EOMES", "PRF1", "Cytotoxic"),
    ("TCF7", "LEF1", "T cell dev"), ("TCF7", "IL7R", "T cell dev"),
    ("ID2", "GATA3", "ILC"), ("BATF", "IL17A", "Th17/Tfh"),
    ("BATF", "IL10", "Exhaustion"), ("TOX", "PDCD1", "Exhaustion"),
    ("TOX", "HAVCR2", "Exhaustion"), ("NFE2L2", "HMOX1", "Antioxidant"),
    ("NFE2L2", "NQO1", "Antioxidant"), ("HIF1A", "VEGFA", "Hypoxia"),
    ("HIF1A", "LDHA", "Hypoxia"), ("HIF1A", "SLC2A1", "Hypoxia"),
]

valid_pairs = []
for tf, target, source in KNOWN_TF_TARGETS:
    if tf in symbol_to_ensembl and target in symbol_to_ensembl:
        valid_pairs.append((tf, target, source, symbol_to_ensembl[tf], symbol_to_ensembl[target]))

print(f"Valid TF-target pairs: {len(valid_pairs)}")

# Extract expression
relevant_ens = set()
for _, _, _, te, tge in valid_pairs:
    relevant_ens.add(te)
    relevant_ens.add(tge)

gene_idx = [i for i, g in enumerate(adata.var_names) if g in relevant_ens]
X_sub = adata.X[:, gene_idx]
if sparse.issparse(X_sub):
    X_sub = X_sub.toarray()
expr_df = pd.DataFrame(X_sub, index=adata.obs_names,
                        columns=[adata.var_names[i] for i in gene_idx])

def spearman_safe(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    r, p = spearmanr(x, y)
    return (r, p) if not np.isnan(r) else (0.0, 1.0)

# ============================================================
# COMPUTE EDGE SCORES + FISHER Z-TRANSFORM TEST
# ============================================================
print("\nComputing edge scores with Fisher z-transform test...")

def fisher_z(r):
    """Fisher z-transform of correlation coefficient."""
    r = np.clip(r, -0.999, 0.999)
    return 0.5 * np.log((1 + r) / (1 - r))

def fisher_z_test(r1, n1, r2, n2):
    """Test if two correlations differ significantly (Fisher z-test)."""
    z1 = fisher_z(r1)
    z2 = fisher_z(r2)
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    if se == 0:
        return 0.0, 1.0
    z_stat = (z1 - z2) / se
    p_val = 2 * norm.sf(abs(z_stat))
    return z_stat, p_val

records = []
regime_masks = {r: (regimes == r) for r in unique_regimes}

for tf_sym, tgt_sym, source, tf_ens, tgt_ens in valid_pairs:
    if tf_ens not in expr_df.columns or tgt_ens not in expr_df.columns:
        continue

    tf_all = expr_df[tf_ens].values
    tgt_all = expr_df[tgt_ens].values
    n_global = len(tf_all)
    rho_global, p_global = spearman_safe(tf_all, tgt_all)

    # Collect per-regime rhos
    regime_rhos = {}
    regime_ns = {}
    for regime in unique_regimes:
        mask = regime_masks[regime]
        n = mask.sum()
        if n < MIN_CELLS:
            continue
        rho_r, p_r = spearman_safe(tf_all[mask], tgt_all[mask])
        regime_rhos[regime] = rho_r
        regime_ns[regime] = n

    if len(regime_rhos) < 3:
        continue

    rho_vals = np.array(list(regime_rhos.values()))

    for regime in regime_rhos:
        rho_focal = regime_rhos[regime]
        n_focal = regime_ns[regime]

        # 1) Fisher z-test: focal regime vs global
        z_stat_global, p_fisher_global = fisher_z_test(
            rho_focal, n_focal, rho_global, n_global)

        # 2) Fisher z-test: focal regime vs rest-of-regimes pooled
        mask_rest = ~regime_masks[regime]
        rho_rest, _ = spearman_safe(tf_all[mask_rest], tgt_all[mask_rest])
        n_rest = mask_rest.sum()
        z_stat_rest, p_fisher_rest = fisher_z_test(
            rho_focal, n_focal, rho_rest, n_rest)

        # 3) One-vs-rest contrast z from distribution of regime rhos
        other_rhos = [regime_rhos[r] for r in regime_rhos if r != regime]
        mean_other = np.mean(other_rhos)
        std_other = np.std(other_rhos) if len(other_rhos) > 1 else 0.01
        contrast_z = (rho_focal - mean_other) / max(std_other, 0.01)

        # Expression fractions
        mask = regime_masks[regime]
        tf_frac = np.mean(tf_all[mask] > 0)
        tgt_frac = np.mean(tgt_all[mask] > 0)

        records.append({
            'tf': tf_sym, 'target': tgt_sym, 'source': source,
            'regime': regime,
            'rho_focal': rho_focal, 'rho_global': rho_global,
            'rho_rest': rho_rest,
            'n_focal': n_focal, 'n_global': n_global,
            'z_stat_global': z_stat_global, 'p_fisher_global': p_fisher_global,
            'z_stat_rest': z_stat_rest, 'p_fisher_rest': p_fisher_rest,
            'contrast_z': contrast_z,
            'rho_range': np.ptp(rho_vals),
            'tf_frac': tf_frac, 'tgt_frac': tgt_frac,
        })

df = pd.DataFrame(records)
print(f"Total (edge, regime) records: {len(df)}")

# FDR correction on Fisher z-test (regime vs rest)
_, df['p_fisher_rest_adj'], _, _ = multipletests(df['p_fisher_rest'], method='fdr_bh')
_, df['p_fisher_global_adj'], _, _ = multipletests(df['p_fisher_global'], method='fdr_bh')

df['sig_rest'] = df['p_fisher_rest_adj'] < 0.05
df['sig_global'] = df['p_fisher_global_adj'] < 0.05

print(f"\nSignificant regime vs rest (FDR<0.05): {df['sig_rest'].sum()}")
print(f"Significant regime vs global (FDR<0.05): {df['sig_global'].sum()}")

# ============================================================
# WITHIN-REGIME PERMUTATION TEST
# ============================================================
print(f"\nRunning within-regime permutation test ({N_PERMS} permutations)...")

# For each (edge, regime), shuffle target expression within that regime
null_exceeds = defaultdict(int)
null_total = defaultdict(int)

for perm_i in range(N_PERMS):
    if (perm_i + 1) % 100 == 0:
        print(f"  Perm {perm_i+1}/{N_PERMS} ({time.time()-t0:.0f}s)")

    for tf_sym, tgt_sym, source, tf_ens, tgt_ens in valid_pairs:
        if tf_ens not in expr_df.columns or tgt_ens not in expr_df.columns:
            continue
        tf_all = expr_df[tf_ens].values
        tgt_all = expr_df[tgt_ens].values

        for regime in unique_regimes:
            mask = regime_masks[regime]
            n = mask.sum()
            if n < MIN_CELLS:
                continue

            idx = np.where(mask)[0]
            tf_r = tf_all[idx]
            # Shuffle target within regime
            tgt_r_perm = tgt_all[idx].copy()
            np.random.shuffle(tgt_r_perm)

            rho_perm, _ = spearman_safe(tf_r, tgt_r_perm)

            key = (tf_sym, tgt_sym, regime)
            null_total[key] += 1
            # Compare absolute correlation: is permuted >= observed?
            obs_rho = df.loc[(df['tf'] == tf_sym) & (df['target'] == tgt_sym) &
                             (df['regime'] == regime), 'rho_focal']
            if len(obs_rho) > 0:
                if abs(rho_perm) >= abs(obs_rho.values[0]):
                    null_exceeds[key] += 1

p_within = []
for _, row in df.iterrows():
    key = (row['tf'], row['target'], row['regime'])
    nt = null_total.get(key, 0)
    ne = null_exceeds.get(key, 0)
    p_within.append((ne + 1) / (nt + 1) if nt > 0 else 1.0)

df['p_within_regime'] = p_within
_, df['p_within_regime_adj'], _, _ = multipletests(df['p_within_regime'], method='fdr_bh')
df['sig_within'] = df['p_within_regime_adj'] < 0.05

print(f"Significant within-regime (FDR<0.05): {df['sig_within'].sum()}")

# ============================================================
# BOOTSTRAP STABILITY
# ============================================================
print(f"\nBootstrap stability ({N_BOOTS} iterations)...")

boot_data = defaultdict(list)
for boot_i in range(N_BOOTS):
    if (boot_i + 1) % 50 == 0:
        print(f"  Boot {boot_i+1}/{N_BOOTS} ({time.time()-t0:.0f}s)")

    for tf_sym, tgt_sym, source, tf_ens, tgt_ens in valid_pairs:
        if tf_ens not in expr_df.columns or tgt_ens not in expr_df.columns:
            continue
        tf_all = expr_df[tf_ens].values
        tgt_all = expr_df[tgt_ens].values

        for regime in unique_regimes:
            mask = regime_masks[regime]
            n = mask.sum()
            if n < MIN_CELLS:
                continue
            idx = np.where(mask)[0]
            bidx = np.random.choice(idx, size=len(idx), replace=True)
            rho_b, _ = spearman_safe(tf_all[bidx], tgt_all[bidx])
            boot_data[(tf_sym, tgt_sym, regime)].append(rho_b)

ci_lo, ci_hi, bstd = [], [], []
for _, row in df.iterrows():
    key = (row['tf'], row['target'], row['regime'])
    vals = boot_data.get(key, [])
    if vals:
        a = np.array(vals)
        ci_lo.append(np.percentile(a, 2.5))
        ci_hi.append(np.percentile(a, 97.5))
        bstd.append(np.std(a))
    else:
        ci_lo.append(np.nan); ci_hi.append(np.nan); bstd.append(np.nan)

df['boot_ci_lo'] = ci_lo
df['boot_ci_hi'] = ci_hi
df['boot_std'] = bstd
df['ci_width'] = df['boot_ci_hi'] - df['boot_ci_lo']

# ============================================================
# CLASSIFY EDGES
# ============================================================
def classify(row):
    """Classify edge-regime pair using multiple criteria."""
    # State-gated: significantly different from rest AND non-trivial correlation
    if row['sig_rest'] and abs(row['rho_focal']) > 0.05:
        if row['rho_focal'] > row['rho_rest'] + 0.05:
            return 'state-gated-active'
        elif row['rho_focal'] < row['rho_rest'] - 0.05:
            return 'state-gated-inactive'
    # Always-on: significant within-regime correlation AND not different from rest
    if row['sig_within'] and not row['sig_rest'] and abs(row['rho_focal']) > 0.1:
        return 'always-on'
    # Significant within but weak
    if row['sig_within'] and abs(row['rho_focal']) > 0.05:
        return 'active-uniform'
    return 'weak/inactive'

df['edge_class'] = df.apply(classify, axis=1)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"\nEdge classification:")
print(df['edge_class'].value_counts())

# Show the key state-gated findings
gated = df[df['edge_class'].isin(['state-gated-active', 'state-gated-inactive'])].copy()
gated = gated.sort_values('z_stat_rest', key=abs, ascending=False)

cols = ['tf', 'target', 'source', 'regime', 'rho_focal', 'rho_rest', 'rho_global',
        'z_stat_rest', 'p_fisher_rest_adj', 'ci_width', 'n_focal', 'edge_class']

print(f"\n--- State-Gated Edges ({len(gated)}) ---")
print(gated[cols].head(40).to_string(index=False))

# Per-edge summary: which edges are most regime-variable?
edge_agg = df.groupby(['tf', 'target', 'source']).agg(
    rho_global=('rho_global', 'first'),
    rho_range=('rho_range', 'first'),
    n_sig_rest=('sig_rest', 'sum'),
    n_gated=('edge_class', lambda x: (x.isin(['state-gated-active', 'state-gated-inactive'])).sum()),
    n_always_on=('edge_class', lambda x: (x == 'always-on').sum()),
    max_rho=('rho_focal', 'max'),
    min_rho=('rho_focal', 'min'),
    mean_ci_width=('ci_width', 'mean'),
).reset_index().sort_values('n_gated', ascending=False)

print(f"\n--- Edge Summary (by number of state-gated regimes) ---")
print(edge_agg.head(30).to_string(index=False))

# Save everything
df.to_csv(os.path.join(IMPL_DIR, "revised_contrast_scores.csv"), index=False)
gated.to_csv(os.path.join(IMPL_DIR, "state_gated_edges.csv"), index=False)
edge_agg.to_csv(os.path.join(IMPL_DIR, "edge_summary_revised.csv"), index=False)

# Also save a regime x edge heatmap matrix
pivot_rho = df.pivot_table(index='regime', columns=['tf', 'target'], values='rho_focal')
pivot_rho.to_csv(os.path.join(IMPL_DIR, "regime_edge_heatmap.csv"))

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed:.0f}s")
print("Revised analysis complete!")
