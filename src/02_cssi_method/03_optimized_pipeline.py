"""
Cell-State-Conditional Circuit Discovery: Optimized Pipeline
=============================================================
Identifies TF-target edges that are conditionally active within specific
cell-state regimes rather than globally across all immune cells.

Optimized version with vectorized permutation testing.
"""

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from scipy import sparse, stats
from scipy.stats import spearmanr, rankdata
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os, sys, warnings, time
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw"
PERTURB_DIR = "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/perturb"
OUT_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_28_cell_state_conditional_circuits"
FIG_DIR = os.path.join(OUT_DIR, "reports", "figures")
IMPL_DIR = os.path.join(OUT_DIR, "implementation")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "reports"), exist_ok=True)

N_PERMS = 500
N_BOOTS = 200
MIN_CELLS_REGIME = 30

t0 = time.time()

# ============================================================
# STEP 1: DATA LOADING & PREPROCESSING
# ============================================================
print("=" * 60)
print("STEP 1: Data Loading & Preprocessing")
print("=" * 60)

adata = sc.read_h5ad(os.path.join(DATA_DIR, "tabula_sapiens_immune_subset_20000.h5ad"))
print(f"Loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")

# Map Ensembl to gene symbols
gene_name_map = dict(zip(adata.var.index, adata.var['feature_name']))
gene_name_map = {k: (v if pd.notna(v) else k) for k, v in gene_name_map.items()}

# Use cell_type as regime definition
adata.obs['regime'] = adata.obs['cell_type'].astype(str)

# Filter to regimes with enough cells
regime_counts = adata.obs['regime'].value_counts()
valid_regimes = regime_counts[regime_counts >= 100].index.tolist()
print(f"\nRegimes with >= 100 cells: {len(valid_regimes)}")

adata_filtered = adata[adata.obs['regime'].isin(valid_regimes)].copy()
print(f"Filtered to {adata_filtered.shape[0]} cells in {len(valid_regimes)} regimes")

# ============================================================
# STEP 2: DEFINE TF-TARGET PAIRS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Define TF-Target Pairs")
print("=" * 60)

KNOWN_TF_TARGETS = [
    ("NFKB1", "IL6", "NF-kB pathway"), ("NFKB1", "TNF", "NF-kB pathway"),
    ("NFKB1", "IL1B", "NF-kB pathway"), ("NFKB1", "CXCL8", "NF-kB pathway"),
    ("NFKB1", "CCL2", "NF-kB pathway"), ("RELA", "IL6", "NF-kB pathway"),
    ("RELA", "TNF", "NF-kB pathway"), ("RELA", "BCL2", "NF-kB pathway"),
    ("RELA", "ICAM1", "NF-kB pathway"),
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
    ("CEBPB", "IL6", "Myeloid activation"), ("CEBPB", "CSF2", "Myeloid activation"),
    ("FOS", "IL2", "AP-1"), ("FOS", "MMP9", "AP-1"),
    ("JUN", "IL2", "AP-1"), ("JUN", "VEGFA", "AP-1"),
    ("IRF1", "CXCL10", "IFN response"), ("IRF1", "GBP1", "IFN response"),
    ("IRF7", "IFNB1", "IFN response"), ("IRF8", "IL12B", "DC activation"),
    ("MYC", "CDK4", "Cell cycle"), ("MYC", "LDHA", "MYC-metabolism"),
    ("MYC", "NPM1", "MYC-proliferation"), ("MYC", "NCL", "MYC-proliferation"),
    ("RUNX3", "CD8A", "CD8 T cell"), ("RUNX3", "GZMB", "Cytotoxic"),
    ("EOMES", "GZMB", "Cytotoxic"), ("EOMES", "PRF1", "Cytotoxic"),
    ("TCF7", "LEF1", "T cell dev"), ("TCF7", "IL7R", "T cell dev"),
    ("ID2", "GATA3", "ILC"), ("BATF", "IL17A", "Th17/Tfh"),
    ("BATF", "IL10", "T cell exhaustion"), ("TOX", "PDCD1", "Exhaustion"),
    ("TOX", "HAVCR2", "Exhaustion"), ("NFE2L2", "HMOX1", "Antioxidant"),
    ("NFE2L2", "NQO1", "Antioxidant"), ("HIF1A", "VEGFA", "Hypoxia"),
    ("HIF1A", "LDHA", "Hypoxia"), ("HIF1A", "SLC2A1", "Hypoxia"),
]

symbol_to_ensembl = {}
for ens_id, row in adata.var.iterrows():
    sym = row['feature_name']
    if pd.notna(sym):
        symbol_to_ensembl[sym] = ens_id

valid_pairs = []
all_tf_syms = set()
all_target_syms = set()
for tf, target, source in KNOWN_TF_TARGETS:
    if tf in symbol_to_ensembl and target in symbol_to_ensembl:
        valid_pairs.append((tf, target, source,
                           symbol_to_ensembl[tf], symbol_to_ensembl[target]))
        all_tf_syms.add(tf)
        all_target_syms.add(target)

print(f"Total curated pairs: {len(KNOWN_TF_TARGETS)}")
print(f"Pairs found in dataset: {len(valid_pairs)}")

# ============================================================
# STEP 3: COMPUTE EDGE SCORES
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Compute Global and Per-Regime Edge Scores")
print("=" * 60)

# Extract expression
relevant_ensembl = set()
for _, _, _, tf_ens, tgt_ens in valid_pairs:
    relevant_ensembl.add(tf_ens)
    relevant_ensembl.add(tgt_ens)

gene_idx = [i for i, g in enumerate(adata_filtered.var_names) if g in relevant_ensembl]
gene_names_subset = [adata_filtered.var_names[i] for i in gene_idx]

X_subset = adata_filtered.X[:, gene_idx]
if sparse.issparse(X_subset):
    X_subset = X_subset.toarray()

expr_df = pd.DataFrame(X_subset, index=adata_filtered.obs_names, columns=gene_names_subset)
regimes = adata_filtered.obs['regime'].values
unique_regimes = sorted(set(regimes))

def spearman_safe(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    rho, pval = spearmanr(x, y)
    if np.isnan(rho):
        return 0.0, 1.0
    return rho, pval

# Compute edge scores
edge_results = []
for tf_sym, tgt_sym, source, tf_ens, tgt_ens in valid_pairs:
    if tf_ens not in expr_df.columns or tgt_ens not in expr_df.columns:
        continue
    tf_expr = expr_df[tf_ens].values
    tgt_expr = expr_df[tgt_ens].values
    rho_global, p_global = spearman_safe(tf_expr, tgt_expr)

    regime_scores = {}
    for regime in unique_regimes:
        mask = regimes == regime
        n_cells = mask.sum()
        if n_cells < MIN_CELLS_REGIME:
            continue
        tf_r = tf_expr[mask]
        tgt_r = tgt_expr[mask]
        rho_regime, p_regime = spearman_safe(tf_r, tgt_r)
        regime_scores[regime] = {
            'rho': rho_regime, 'p': p_regime, 'n_cells': n_cells,
            'tf_frac_expressed': np.mean(tf_r > 0),
            'tgt_frac_expressed': np.mean(tgt_r > 0),
        }

    edge_results.append({
        'tf': tf_sym, 'target': tgt_sym, 'source': source,
        'tf_ens': tf_ens, 'tgt_ens': tgt_ens,
        'rho_global': rho_global, 'p_global': p_global,
        'regime_scores': regime_scores,
    })

print(f"Computed scores for {len(edge_results)} edges")

# ============================================================
# STEP 4: CONTRAST SCORING
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Regime-Specific Contrast Scoring")
print("=" * 60)

contrast_records = []
for edge in edge_results:
    regime_rhos = {r: s['rho'] for r, s in edge['regime_scores'].items()}
    if len(regime_rhos) < 3:
        continue
    regimes_list = sorted(regime_rhos.keys())
    rho_values = np.array([regime_rhos[r] for r in regimes_list])

    for i, regime in enumerate(regimes_list):
        rho_focal = rho_values[i]
        rho_rest = np.delete(rho_values, i)
        mean_rest = np.mean(rho_rest)
        std_rest = np.std(rho_rest) if len(rho_rest) > 1 else 0.01
        contrast_z = (rho_focal - mean_rest) / max(std_rest, 0.01)
        rho_range = np.max(rho_values) - np.min(rho_values)

        contrast_records.append({
            'tf': edge['tf'], 'target': edge['target'], 'source': edge['source'],
            'regime': regime, 'rho_focal': rho_focal,
            'rho_global': edge['rho_global'], 'mean_rest': mean_rest,
            'contrast_z': contrast_z, 'rho_range': rho_range,
            'n_cells': edge['regime_scores'][regime]['n_cells'],
            'tf_frac_expressed': edge['regime_scores'][regime]['tf_frac_expressed'],
            'tgt_frac_expressed': edge['regime_scores'][regime]['tgt_frac_expressed'],
        })

contrast_df = pd.DataFrame(contrast_records)
print(f"Contrast records: {len(contrast_df)}")

def classify_edge(row):
    if abs(row['contrast_z']) > 2.0 and abs(row['rho_focal']) > 0.1:
        return 'state-gated-active' if row['contrast_z'] > 0 else 'state-gated-inactive'
    elif abs(row['rho_global']) > 0.1 and abs(row['contrast_z']) < 1.0:
        return 'always-on'
    else:
        return 'weak/inactive'

contrast_df['edge_class'] = contrast_df.apply(classify_edge, axis=1)
print("\nEdge classification:")
print(contrast_df['edge_class'].value_counts())

# ============================================================
# STEP 5: OPTIMIZED PERMUTATION TESTING
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 5: Permutation Testing ({N_PERMS} permutations)")
print("=" * 60)

# Precompute regime masks
regime_masks = {}
for r in unique_regimes:
    regime_masks[r] = (regimes == r)

# Precompute expression arrays for all edges
edge_expr = {}
for tf_sym, tgt_sym, source, tf_ens, tgt_ens in valid_pairs:
    if tf_ens in expr_df.columns and tgt_ens in expr_df.columns:
        edge_expr[(tf_sym, tgt_sym)] = (expr_df[tf_ens].values, expr_df[tgt_ens].values)

# Build observed contrast_z lookup
obs_contrast_z = {}
for _, row in contrast_df.iterrows():
    obs_contrast_z[(row['tf'], row['target'], row['regime'])] = row['contrast_z']

# Vectorized permutation
null_exceeds = defaultdict(int)  # count of null |z| >= observed |z|
n_null_total = defaultdict(int)

for perm_i in range(N_PERMS):
    if (perm_i + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  Permutation {perm_i + 1}/{N_PERMS}  ({elapsed:.0f}s elapsed)")

    perm_regimes = np.random.permutation(regimes)
    perm_masks = {r: (perm_regimes == r) for r in unique_regimes}

    for (tf_sym, tgt_sym), (tf_expr, tgt_expr) in edge_expr.items():
        perm_rhos = {}
        for regime in unique_regimes:
            mask = perm_masks[regime]
            n_cells = mask.sum()
            if n_cells < MIN_CELLS_REGIME:
                continue
            rho_p, _ = spearman_safe(tf_expr[mask], tgt_expr[mask])
            perm_rhos[regime] = rho_p

        if len(perm_rhos) < 3:
            continue

        regimes_list = sorted(perm_rhos.keys())
        rho_values = np.array([perm_rhos[r] for r in regimes_list])

        for i, regime in enumerate(regimes_list):
            rho_focal = rho_values[i]
            rho_rest = np.delete(rho_values, i)
            mean_rest = np.mean(rho_rest)
            std_rest = np.std(rho_rest) if len(rho_rest) > 1 else 0.01
            perm_z = (rho_focal - mean_rest) / max(std_rest, 0.01)

            key = (tf_sym, tgt_sym, regime)
            n_null_total[key] += 1
            if key in obs_contrast_z and abs(perm_z) >= abs(obs_contrast_z[key]):
                null_exceeds[key] += 1

# Compute empirical p-values
pvals = []
for _, row in contrast_df.iterrows():
    key = (row['tf'], row['target'], row['regime'])
    n_total = n_null_total.get(key, 0)
    n_exceed = null_exceeds.get(key, 0)
    if n_total > 0:
        p_emp = (n_exceed + 1) / (n_total + 1)
    else:
        p_emp = 1.0
    pvals.append(p_emp)

contrast_df['p_perm'] = pvals
reject, p_adj, _, _ = multipletests(contrast_df['p_perm'], method='fdr_bh', alpha=0.05)
contrast_df['p_adj'] = p_adj
contrast_df['significant'] = reject

print(f"\nSignificant (FDR < 0.05): {contrast_df['significant'].sum()} / {len(contrast_df)}")

# ============================================================
# STEP 6: BOOTSTRAP STABILITY
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 6: Bootstrap Stability ({N_BOOTS} iterations)")
print("=" * 60)

boot_rhos = defaultdict(list)  # (tf, target, regime) -> list of rho
for boot_i in range(N_BOOTS):
    if (boot_i + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  Bootstrap {boot_i + 1}/{N_BOOTS}  ({elapsed:.0f}s elapsed)")

    for (tf_sym, tgt_sym), (tf_expr, tgt_expr) in edge_expr.items():
        for regime in unique_regimes:
            mask = regime_masks[regime]
            n_cells = mask.sum()
            if n_cells < MIN_CELLS_REGIME:
                continue
            idx = np.where(mask)[0]
            boot_idx = np.random.choice(idx, size=len(idx), replace=True)
            rho_b, _ = spearman_safe(tf_expr[boot_idx], tgt_expr[boot_idx])
            boot_rhos[(tf_sym, tgt_sym, regime)].append(rho_b)

# Add bootstrap stats
boot_ci_low, boot_ci_high, boot_std_list = [], [], []
for _, row in contrast_df.iterrows():
    key = (row['tf'], row['target'], row['regime'])
    vals = boot_rhos.get(key, [])
    if vals:
        arr = np.array(vals)
        boot_ci_low.append(np.percentile(arr, 2.5))
        boot_ci_high.append(np.percentile(arr, 97.5))
        boot_std_list.append(np.std(arr))
    else:
        boot_ci_low.append(np.nan)
        boot_ci_high.append(np.nan)
        boot_std_list.append(np.nan)

contrast_df['boot_ci_low'] = boot_ci_low
contrast_df['boot_ci_high'] = boot_ci_high
contrast_df['boot_std'] = boot_std_list
contrast_df['ci_width'] = contrast_df['boot_ci_high'] - contrast_df['boot_ci_low']

print(f"Mean CI width: {contrast_df['ci_width'].mean():.4f}")
print(f"Median CI width: {contrast_df['ci_width'].median():.4f}")

# ============================================================
# STEP 7: PERTURBATION CROSS-REFERENCE
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Perturbation Data Cross-Reference")
print("=" * 60)

perturb_files = [f for f in os.listdir(PERTURB_DIR) if f.endswith('.h5ad')]
perturb_evidence = []

for pf in perturb_files:
    try:
        pdata = sc.read_h5ad(os.path.join(PERTURB_DIR, pf))
        print(f"\n{pf}: {pdata.shape}")

        # Get gene names from perturbation data
        if 'feature_name' in pdata.var.columns:
            pgenes = set(pdata.var['feature_name'].dropna())
        elif 'gene_name' in pdata.var.columns:
            pgenes = set(pdata.var['gene_name'].dropna())
        else:
            pgenes = set(pdata.var_names)

        # Find perturbation column
        perturb_col = None
        perturbed_genes = set()
        for col in pdata.obs.columns:
            clow = col.lower()
            if any(k in clow for k in ['perturb', 'guide', 'target_gene', 'condition']):
                perturb_col = col
                perturbed_genes = set(pdata.obs[col].unique())
                break

        if perturb_col:
            print(f"  Perturbation column: {perturb_col}")
            tf_perturbed = all_tf_syms.intersection(perturbed_genes)
            print(f"  Our TFs perturbed in dataset: {tf_perturbed}")

        for edge in edge_results:
            tf, target = edge['tf'], edge['target']
            if tf in pgenes and target in pgenes:
                has_tf_perturbed = tf in perturbed_genes if perturb_col else False
                perturb_evidence.append({
                    'tf': tf, 'target': target, 'dataset': pf,
                    'source': edge['source'],
                    'tf_directly_perturbed': has_tf_perturbed,
                })
        del pdata
    except Exception as e:
        print(f"  Error: {e}")

if perturb_evidence:
    perturb_ev_df = pd.DataFrame(perturb_evidence)
    n_direct = perturb_ev_df['tf_directly_perturbed'].sum()
    print(f"\nEdges with gene overlap in perturb data: {len(perturb_ev_df)}")
    print(f"Edges with TF directly perturbed: {n_direct}")
    perturb_ev_df.to_csv(os.path.join(IMPL_DIR, "perturb_evidence.csv"), index=False)
else:
    print("No perturbation overlap found.")
    perturb_ev_df = pd.DataFrame()

# ============================================================
# STEP 8: SAVE ALL RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Save Results")
print("=" * 60)

contrast_df.to_csv(os.path.join(IMPL_DIR, "contrast_scores.csv"), index=False)

# Summary per edge
edge_summary = []
for e in edge_results:
    regime_rhos = {r: s['rho'] for r, s in e['regime_scores'].items()}
    if regime_rhos:
        rho_vals = list(regime_rhos.values())
        max_regime = max(regime_rhos, key=lambda r: regime_rhos[r])
        min_regime = min(regime_rhos, key=lambda r: regime_rhos[r])
        rho_range = max(rho_vals) - min(rho_vals)
        rho_std = np.std(rho_vals)
    else:
        max_regime = min_regime = "N/A"
        rho_range = rho_std = 0
    edge_summary.append({
        'tf': e['tf'], 'target': e['target'], 'source': e['source'],
        'rho_global': e['rho_global'], 'rho_range': rho_range, 'rho_std': rho_std,
        'max_rho_regime': max_regime, 'max_rho': max(rho_vals) if rho_vals else 0,
        'min_rho_regime': min_regime, 'min_rho': min(rho_vals) if rho_vals else 0,
        'n_regimes': len(e['regime_scores']),
    })
edge_summary_df = pd.DataFrame(edge_summary).sort_values('rho_range', ascending=False)
edge_summary_df.to_csv(os.path.join(IMPL_DIR, "edge_summary.csv"), index=False)

# Stable gated edges
stable_gated = contrast_df[
    (contrast_df['significant']) &
    (contrast_df['edge_class'].isin(['state-gated-active', 'state-gated-inactive'])) &
    (contrast_df['ci_width'] < contrast_df['ci_width'].quantile(0.75))
].copy()
stable_gated.to_csv(os.path.join(IMPL_DIR, "stable_gated_edges.csv"), index=False)

print(f"Saved to {IMPL_DIR}")

# Print summary statistics
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"\nTotal edges analyzed: {len(edge_results)}")
print(f"Total (edge, regime) pairs: {len(contrast_df)}")
print(f"Significant at FDR < 0.05: {contrast_df['significant'].sum()}")
print(f"Significant at FDR < 0.10: {(contrast_df['p_adj'] < 0.10).sum()}")
print(f"\nEdge classification (all pairs):")
print(contrast_df['edge_class'].value_counts().to_string())
print(f"\nSignificant state-gated edges:")
sig = contrast_df[contrast_df['significant']]
print(sig['edge_class'].value_counts().to_string())
print(f"\nStable state-gated edges: {len(stable_gated)}")

print(f"\n--- Top 20 Edges by Regime Variability ---")
print(edge_summary_df.head(20).to_string(index=False))

cols_show = ['tf', 'target', 'regime', 'rho_focal', 'rho_global', 'contrast_z', 'p_adj', 'ci_width']
sig_active = contrast_df[
    (contrast_df['significant']) & (contrast_df['edge_class'] == 'state-gated-active')
].sort_values('contrast_z', ascending=False)

print(f"\n--- Top Significant State-Gated-Active ---")
if len(sig_active) > 0:
    print(sig_active[cols_show].head(25).to_string(index=False))
else:
    # Relax threshold
    sig_active2 = contrast_df[contrast_df['p_adj'] < 0.10].sort_values('contrast_z', ascending=False)
    print(f"(relaxing to FDR < 0.10, showing top state-gated)")
    print(sig_active2[cols_show].head(25).to_string(index=False))

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed:.0f}s")
print("Pipeline complete!")
