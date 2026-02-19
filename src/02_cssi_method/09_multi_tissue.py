"""
Phase 2: Multi-Tissue Extension of Cell-State-Conditional Circuits
==================================================================
Replicates the Phase 1 regime-specific analysis on lung and kidney tissues
from the Tabula Sapiens atlas. Compares state-gating rates across tissues
and tests whether the same TF-target pairs show similar gating patterns.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse, stats
from scipy.stats import spearmanr, norm, fisher_exact, chi2_contingency
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
OUT_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_28_cell_state_conditional_circuits"
FIG_DIR = os.path.join(OUT_DIR, "reports", "figures")
IMPL_DIR = os.path.join(OUT_DIR, "implementation")
os.makedirs(FIG_DIR, exist_ok=True)

N_PERMS = 500
N_BOOTS = 200
MIN_CELLS = 50

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


def spearman_safe(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    r, p = spearmanr(x, y)
    return (r, p) if not np.isnan(r) else (0.0, 1.0)


def fisher_z(r):
    r = np.clip(r, -0.999, 0.999)
    return 0.5 * np.log((1 + r) / (1 - r))


def fisher_z_test(r1, n1, r2, n2):
    z1 = fisher_z(r1)
    z2 = fisher_z(r2)
    se = np.sqrt(1.0 / max(n1 - 3, 1) + 1.0 / max(n2 - 3, 1))
    if se == 0:
        return 0.0, 1.0
    z_stat = (z1 - z2) / se
    p_val = 2 * norm.sf(abs(z_stat))
    return z_stat, p_val


def classify(row):
    if row['sig_rest'] and abs(row['rho_focal']) > 0.05:
        if row['rho_focal'] > row['rho_rest'] + 0.05:
            return 'state-gated-active'
        elif row['rho_focal'] < row['rho_rest'] - 0.05:
            return 'state-gated-inactive'
    if row['sig_within'] and not row['sig_rest'] and abs(row['rho_focal']) > 0.1:
        return 'always-on'
    if row['sig_within'] and abs(row['rho_focal']) > 0.05:
        return 'active-uniform'
    return 'weak/inactive'


def analyze_tissue(tissue_name, h5ad_path, min_regime_cells=100):
    """Run the full Phase 1 pipeline on a new tissue."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Analyzing {tissue_name}")
    print(f"{'='*60}")

    print("Loading data...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Raw shape: {adata.shape}")

    # Subsample lung to 20k cells for comparability with immune dataset
    if adata.shape[0] > 25000:
        print(f"  Subsampling to 20000 cells for comparability...")
        sc.pp.subsample(adata, n_obs=20000, random_state=42)

    adata.obs['regime'] = adata.obs['cell_type'].astype(str)
    regime_counts = adata.obs['regime'].value_counts()
    valid_regimes = regime_counts[regime_counts >= min_regime_cells].index.tolist()
    adata = adata[adata.obs['regime'].isin(valid_regimes)].copy()
    regimes = adata.obs['regime'].values
    unique_regimes = sorted(set(regimes))
    print(f"  Filtered: {adata.shape[0]} cells, {len(unique_regimes)} regimes")
    print(f"  Regimes: {', '.join(unique_regimes)}")

    # Gene mapping
    symbol_to_ensembl = {}
    for ens_id, row in adata.var.iterrows():
        sym = row.get('feature_name', None)
        if pd.notna(sym):
            symbol_to_ensembl[sym] = ens_id

    # Valid pairs
    valid_pairs = []
    for tf, target, source in KNOWN_TF_TARGETS:
        if tf in symbol_to_ensembl and target in symbol_to_ensembl:
            valid_pairs.append((tf, target, source, symbol_to_ensembl[tf], symbol_to_ensembl[target]))
    print(f"  Valid TF-target pairs: {len(valid_pairs)}")

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

    # Compute edge scores
    print("  Computing edge scores with Fisher z-transform...")
    records = []
    regime_masks = {r: (regimes == r) for r in unique_regimes}

    for tf_sym, tgt_sym, source, tf_ens, tgt_ens in valid_pairs:
        if tf_ens not in expr_df.columns or tgt_ens not in expr_df.columns:
            continue

        tf_all = expr_df[tf_ens].values
        tgt_all = expr_df[tgt_ens].values
        n_global = len(tf_all)
        rho_global, _ = spearman_safe(tf_all, tgt_all)

        regime_rhos = {}
        regime_ns = {}
        for regime in unique_regimes:
            mask = regime_masks[regime]
            n = mask.sum()
            if n < MIN_CELLS:
                continue
            rho_r, _ = spearman_safe(tf_all[mask], tgt_all[mask])
            regime_rhos[regime] = rho_r
            regime_ns[regime] = n

        if len(regime_rhos) < 3:
            continue

        for regime in regime_rhos:
            rho_focal = regime_rhos[regime]
            n_focal = regime_ns[regime]

            # Fisher z-test: focal vs rest
            mask_rest = ~regime_masks[regime]
            rho_rest, _ = spearman_safe(tf_all[mask_rest], tgt_all[mask_rest])
            n_rest = mask_rest.sum()
            z_stat_rest, p_fisher_rest = fisher_z_test(rho_focal, n_focal, rho_rest, n_rest)

            # Contrast z
            other_rhos = [regime_rhos[r] for r in regime_rhos if r != regime]
            mean_other = np.mean(other_rhos)
            std_other = np.std(other_rhos) if len(other_rhos) > 1 else 0.01
            contrast_z = (rho_focal - mean_other) / max(std_other, 0.01)

            mask = regime_masks[regime]
            tf_frac = np.mean(tf_all[mask] > 0)
            tgt_frac = np.mean(tgt_all[mask] > 0)

            records.append({
                'tf': tf_sym, 'target': tgt_sym, 'source': source,
                'regime': regime, 'tissue': tissue_name,
                'rho_focal': rho_focal, 'rho_global': rho_global,
                'rho_rest': rho_rest,
                'n_focal': n_focal, 'n_global': n_global,
                'z_stat_rest': z_stat_rest, 'p_fisher_rest': p_fisher_rest,
                'contrast_z': contrast_z,
                'rho_range': np.ptp(list(regime_rhos.values())),
                'tf_frac': tf_frac, 'tgt_frac': tgt_frac,
            })

    df = pd.DataFrame(records)
    if len(df) == 0:
        print(f"  WARNING: No valid edge-regime pairs found for {tissue_name}")
        return df
    print(f"  Total edge-regime pairs: {len(df)}")

    # FDR correction
    _, df['p_fisher_rest_adj'], _, _ = multipletests(df['p_fisher_rest'], method='fdr_bh')
    df['sig_rest'] = df['p_fisher_rest_adj'] < 0.05

    # Within-regime permutation test
    print(f"  Running permutation test ({N_PERMS} permutations)...")
    null_exceeds = defaultdict(int)
    null_total = defaultdict(int)

    for perm_i in range(N_PERMS):
        if (perm_i + 1) % 100 == 0:
            print(f"    Perm {perm_i+1}/{N_PERMS} ({time.time()-t0:.0f}s)")
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
                tgt_r_perm = tgt_all[idx].copy()
                np.random.shuffle(tgt_r_perm)
                rho_perm, _ = spearman_safe(tf_r, tgt_r_perm)
                key = (tf_sym, tgt_sym, regime)
                null_total[key] += 1
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

    # Bootstrap stability
    print(f"  Bootstrap stability ({N_BOOTS} iterations)...")
    boot_data = defaultdict(list)
    for boot_i in range(N_BOOTS):
        if (boot_i + 1) % 50 == 0:
            print(f"    Boot {boot_i+1}/{N_BOOTS} ({time.time()-t0:.0f}s)")
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
            ci_lo.append(np.nan)
            ci_hi.append(np.nan)
            bstd.append(np.nan)

    df['boot_ci_lo'] = ci_lo
    df['boot_ci_hi'] = ci_hi
    df['boot_std'] = bstd
    df['ci_width'] = df['boot_ci_hi'] - df['boot_ci_lo']

    # Classify edges
    df['edge_class'] = df.apply(classify, axis=1)

    elapsed = time.time() - t0
    print(f"\n  {tissue_name} Results:")
    print(f"  Classification: {df['edge_class'].value_counts().to_dict()}")
    print(f"  Sig rest: {df['sig_rest'].sum()}/{len(df)} ({100*df['sig_rest'].mean():.1f}%)")
    n_gated = df['edge_class'].isin(['state-gated-active', 'state-gated-inactive']).sum()
    print(f"  State-gated: {n_gated}/{len(df)} ({100*n_gated/len(df):.1f}%)")
    print(f"  Runtime: {elapsed:.0f}s")

    return df


# ============================================================
# Run analysis on all three tissues
# ============================================================
t_total = time.time()

# Load Phase 1 immune results for comparison
df_immune = pd.read_csv(os.path.join(IMPL_DIR, "revised_contrast_scores.csv"))
df_immune['tissue'] = 'immune'

# Run lung analysis
df_lung = analyze_tissue(
    "lung",
    os.path.join(DATA_DIR, "tabula_sapiens_lung.h5ad"),
    min_regime_cells=100
)

# Run kidney analysis
df_kidney = analyze_tissue(
    "kidney",
    os.path.join(DATA_DIR, "tabula_sapiens_kidney.h5ad"),
    min_regime_cells=100
)

# Combine all results
all_results = [df_immune]
if len(df_lung) > 0:
    all_results.append(df_lung)
if len(df_kidney) > 0:
    all_results.append(df_kidney)

df_all = pd.concat(all_results, ignore_index=True)
df_all.to_csv(os.path.join(IMPL_DIR, "phase2_multi_tissue_all.csv"), index=False)

# ============================================================
# CROSS-TISSUE COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("CROSS-TISSUE COMPARISON")
print("=" * 60)

# 1. State-gating rate by tissue
tissue_summary = []
for tissue in df_all['tissue'].unique():
    sub = df_all[df_all['tissue'] == tissue]
    n_total = len(sub)
    n_sig = sub['sig_rest'].sum()
    n_gated = sub['edge_class'].isin(['state-gated-active', 'state-gated-inactive']).sum()
    n_active = (sub['edge_class'] == 'state-gated-active').sum()
    n_inactive = (sub['edge_class'] == 'state-gated-inactive').sum()
    n_always = (sub['edge_class'] == 'always-on').sum()
    tissue_summary.append({
        'tissue': tissue,
        'n_pairs': n_total,
        'n_sig_rest': n_sig,
        'pct_sig_rest': 100 * n_sig / n_total,
        'n_state_gated': n_gated,
        'pct_state_gated': 100 * n_gated / n_total,
        'n_gated_active': n_active,
        'n_gated_inactive': n_inactive,
        'n_always_on': n_always,
    })
tissue_df = pd.DataFrame(tissue_summary)
print("\nTissue-level summary:")
print(tissue_df.to_string(index=False))

# 2. Per-edge concordance across tissues: do the same edges get gated?
# For each edge, check if it's state-gated in multiple tissues
edge_tissue_gating = df_all.groupby(['tf', 'target', 'tissue']).agg(
    n_gated=('edge_class', lambda x: x.isin(['state-gated-active', 'state-gated-inactive']).sum()),
    n_regimes=('edge_class', 'count'),
    max_abs_rho=('rho_focal', lambda x: np.max(np.abs(x))),
).reset_index()
edge_tissue_gating['pct_gated'] = 100 * edge_tissue_gating['n_gated'] / edge_tissue_gating['n_regimes']

# Pivot to see concordance
concordance = edge_tissue_gating.pivot_table(
    index=['tf', 'target'], columns='tissue', values='pct_gated'
).dropna(how='all')

print("\nEdge-level gating concordance (% of regimes state-gated):")
print(concordance.describe().to_string())

# Cross-tissue correlation of gating rates
tissues = [t for t in df_all['tissue'].unique() if t in concordance.columns]
if len(tissues) >= 2:
    print("\nPairwise Spearman correlations of per-edge gating rates:")
    for i in range(len(tissues)):
        for j in range(i+1, len(tissues)):
            t1, t2 = tissues[i], tissues[j]
            shared = concordance[[t1, t2]].dropna()
            if len(shared) >= 5:
                r, p = spearmanr(shared[t1], shared[t2])
                print(f"  {t1} vs {t2}: rho={r:.3f}, p={p:.2e}, n={len(shared)} edges")

# 3. Shared cell types across tissues
if len(df_lung) > 0:
    shared_lung = set(df_immune['regime'].unique()) & set(df_lung['regime'].unique())
    print(f"\nShared cell types (immune & lung): {shared_lung}")
if len(df_kidney) > 0:
    shared_kidney = set(df_immune['regime'].unique()) & set(df_kidney['regime'].unique())
    print(f"Shared cell types (immune & kidney): {shared_kidney}")

# 4. For shared cell types, compare rho_focal directly
print("\nDirect rho comparison for shared cell types & edges:")
if len(df_lung) > 0 and len(shared_lung) > 0:
    merged = df_immune.merge(df_lung, on=['tf', 'target', 'regime'], suffixes=('_immune', '_lung'))
    if len(merged) > 0:
        r, p = spearmanr(merged['rho_focal_immune'], merged['rho_focal_lung'])
        print(f"  Immune vs Lung (shared regimes): rho={r:.3f}, p={p:.2e}, n={len(merged)} pairs")
        # Are state-gated edges in immune also state-gated in lung?
        n_both_gated = ((merged['edge_class_immune'].isin(['state-gated-active', 'state-gated-inactive'])) &
                        (merged['edge_class_lung'].isin(['state-gated-active', 'state-gated-inactive']))).sum()
        n_immune_gated = merged['edge_class_immune'].isin(['state-gated-active', 'state-gated-inactive']).sum()
        print(f"  Of {n_immune_gated} immune-gated pairs, {n_both_gated} also gated in lung ({100*n_both_gated/max(n_immune_gated,1):.1f}%)")

if len(df_kidney) > 0 and len(shared_kidney) > 0:
    merged_k = df_immune.merge(df_kidney, on=['tf', 'target', 'regime'], suffixes=('_immune', '_kidney'))
    if len(merged_k) > 0:
        r, p = spearmanr(merged_k['rho_focal_immune'], merged_k['rho_focal_kidney'])
        print(f"  Immune vs Kidney (shared regimes): rho={r:.3f}, p={p:.2e}, n={len(merged_k)} pairs")

# ============================================================
# FIGURES
# ============================================================
print("\nGenerating figures...")

# Figure P2-1: State-gating rate comparison across tissues
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Classification distribution by tissue
class_order = ['state-gated-active', 'state-gated-inactive', 'always-on', 'active-uniform', 'weak/inactive']
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#95a5a6']
tissue_class_counts = df_all.groupby(['tissue', 'edge_class']).size().unstack(fill_value=0)
for c in class_order:
    if c not in tissue_class_counts.columns:
        tissue_class_counts[c] = 0
tissue_class_pct = tissue_class_counts[class_order].div(tissue_class_counts.sum(axis=1), axis=0) * 100

tissue_class_pct.plot(kind='bar', stacked=True, color=colors, ax=axes[0], edgecolor='white')
axes[0].set_ylabel('Percentage of edge-regime pairs')
axes[0].set_title('Edge Classification by Tissue')
axes[0].set_xlabel('')
axes[0].legend(fontsize=7, loc='upper right')
axes[0].tick_params(axis='x', rotation=0)

# Panel B: Gating rate comparison
for i, tissue in enumerate(tissue_df['tissue']):
    axes[1].bar(i, tissue_df.loc[tissue_df['tissue']==tissue, 'pct_sig_rest'].values[0],
                color='#3498db', alpha=0.7, label='Sig. vs rest' if i==0 else None)
    axes[1].bar(i, tissue_df.loc[tissue_df['tissue']==tissue, 'pct_state_gated'].values[0],
                color='#e74c3c', alpha=0.7, label='State-gated' if i==0 else None)
axes[1].set_xticks(range(len(tissue_df)))
axes[1].set_xticklabels(tissue_df['tissue'])
axes[1].set_ylabel('Percentage')
axes[1].set_title('State-Gating Rates Across Tissues')
axes[1].legend()

# Panel C: Cross-tissue rho concordance scatter (if shared cell types exist)
if len(df_lung) > 0 and 'merged' in dir() and len(merged) > 0:
    axes[2].scatter(merged['rho_focal_immune'], merged['rho_focal_lung'],
                    alpha=0.4, s=20, c='#2c3e50')
    axes[2].plot([-1, 1], [-1, 1], 'r--', alpha=0.5)
    axes[2].set_xlabel('rho_focal (immune)')
    axes[2].set_ylabel('rho_focal (lung)')
    axes[2].set_title('Cross-Tissue Correlation\n(Shared Cell Types)')
    r_val, p_val = spearmanr(merged['rho_focal_immune'], merged['rho_focal_lung'])
    axes[2].annotate(f'rho={r_val:.3f}\np={p_val:.2e}',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
else:
    axes[2].text(0.5, 0.5, 'No shared cell types\nfor direct comparison',
                 transform=axes[2].transAxes, ha='center', va='center')
    axes[2].set_title('Cross-Tissue Correlation')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "phase2_fig1_multi_tissue_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

# Figure P2-2: Heatmap of per-edge gating rates across tissues
if len(concordance) > 0 and len(concordance.columns) > 1:
    fig, ax = plt.subplots(figsize=(8, max(6, len(concordance) * 0.25)))
    # Sort by mean gating rate
    concordance['mean_gating'] = concordance.mean(axis=1)
    concordance_sorted = concordance.sort_values('mean_gating', ascending=True).drop(columns='mean_gating')
    # Show top 40 most variable edges
    concordance_sorted['range'] = concordance_sorted.max(axis=1) - concordance_sorted.min(axis=1)
    top_edges = concordance_sorted.nlargest(40, 'range').drop(columns='range')
    if len(top_edges) > 0:
        sns.heatmap(top_edges, cmap='RdYlBu_r', center=50, vmin=0, vmax=100,
                    annot=True, fmt='.0f', ax=ax, cbar_kws={'label': '% regimes state-gated'})
        ax.set_title('Edge-Level State-Gating Rates Across Tissues')
        ax.set_ylabel('TF -> Target')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "phase2_fig2_cross_tissue_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()

# Figure P2-3: Per-tissue heatmaps (like Phase 1 Fig 1)
for tissue_name_fig, tissue_df_fig in [('lung', df_lung), ('kidney', df_kidney)]:
    if len(tissue_df_fig) == 0:
        continue
    pivot = tissue_df_fig.pivot_table(index='regime', columns='tf', values='rho_focal', aggfunc='mean')
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        continue
    # Select top variable edges
    edge_var = tissue_df_fig.groupby(['tf', 'target'])['rho_focal'].std().nlargest(25)
    top_edges_fig = [(tf, tgt) for (tf, tgt) in edge_var.index]
    pivot2 = tissue_df_fig.pivot_table(
        index='regime', columns=[tissue_df_fig['tf'] + '->' + tissue_df_fig['target']],
        values='rho_focal'
    )
    # Select columns that exist
    fig, ax = plt.subplots(figsize=(14, max(4, pivot.shape[0] * 0.4)))
    sns.heatmap(pivot, cmap='RdBu_r', center=0, vmin=-0.5, vmax=0.5,
                ax=ax, cbar_kws={'label': 'Spearman rho'})
    ax.set_title(f'{tissue_name_fig.capitalize()} - Per-Regime TF Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"phase2_fig3_{tissue_name_fig}_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()

# Save tissue summary
tissue_df.to_csv(os.path.join(IMPL_DIR, "phase2_tissue_summary.csv"), index=False)
concordance.to_csv(os.path.join(IMPL_DIR, "phase2_cross_tissue_concordance.csv"))

elapsed_total = time.time() - t_total
print(f"\nTotal multi-tissue analysis runtime: {elapsed_total:.0f}s")
print("Phase 2 multi-tissue analysis complete!")
