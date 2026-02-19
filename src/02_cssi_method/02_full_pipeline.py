"""
Cell-State-Conditional Circuit Discovery: Full Pipeline
========================================================
Identifies TF-target edges that are conditionally active within specific
cell-state regimes rather than globally across all immune cells.
"""

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from scipy import sparse, stats
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mutual_info_score
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os, sys, warnings, pickle, json
from collections import defaultdict
from itertools import combinations

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

# ============================================================
# STEP 1: DATA LOADING & PREPROCESSING
# ============================================================
print("=" * 60)
print("STEP 1: Data Loading & Preprocessing")
print("=" * 60)

adata = sc.read_h5ad(os.path.join(DATA_DIR, "tabula_sapiens_immune_subset_20000.h5ad"))
print(f"Loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")

# Gene names are Ensembl IDs - map to symbols using feature_name
gene_name_map = dict(zip(adata.var.index, adata.var['feature_name']))
# Some feature_names may be NaN; keep ensembl ID in those cases
gene_name_map = {k: (v if pd.notna(v) else k) for k, v in gene_name_map.items()}

# Use cell_type as our primary regime definition
adata.obs['regime'] = adata.obs['cell_type'].astype(str)

# Filter to regimes with at least 100 cells for statistical power
regime_counts = adata.obs['regime'].value_counts()
valid_regimes = regime_counts[regime_counts >= 100].index.tolist()
print(f"\nRegimes with >= 100 cells: {len(valid_regimes)}")
for r in valid_regimes:
    print(f"  {r}: {regime_counts[r]} cells")

adata_filtered = adata[adata.obs['regime'].isin(valid_regimes)].copy()
print(f"\nFiltered to {adata_filtered.shape[0]} cells in {len(valid_regimes)} regimes")

# ============================================================
# STEP 2: DEFINE TF-TARGET PAIRS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Define TF-Target Pairs")
print("=" * 60)

# Curated immune-relevant TF-target pairs from literature
# Format: (TF, target, evidence_source)
KNOWN_TF_TARGETS = [
    # NF-kB signaling
    ("NFKB1", "IL6", "NF-kB pathway"),
    ("NFKB1", "TNF", "NF-kB pathway"),
    ("NFKB1", "IL1B", "NF-kB pathway"),
    ("NFKB1", "CXCL8", "NF-kB pathway"),
    ("NFKB1", "CCL2", "NF-kB pathway"),
    ("RELA", "IL6", "NF-kB pathway"),
    ("RELA", "TNF", "NF-kB pathway"),
    ("RELA", "BCL2", "NF-kB pathway"),
    ("RELA", "ICAM1", "NF-kB pathway"),
    # STAT signaling
    ("STAT1", "IRF1", "JAK-STAT pathway"),
    ("STAT1", "GBP1", "JAK-STAT pathway"),
    ("STAT1", "CXCL10", "JAK-STAT pathway"),
    ("STAT1", "IDO1", "JAK-STAT pathway"),
    ("STAT3", "BCL2", "JAK-STAT pathway"),
    ("STAT3", "MYC", "JAK-STAT pathway"),
    ("STAT3", "SOCS3", "JAK-STAT pathway"),
    ("STAT3", "VEGFA", "JAK-STAT pathway"),
    ("STAT4", "IFNG", "JAK-STAT pathway"),
    ("STAT5A", "IL2RA", "JAK-STAT pathway"),
    ("STAT5A", "BCL2L1", "JAK-STAT pathway"),
    ("STAT6", "GATA3", "JAK-STAT pathway"),
    ("STAT6", "IL4R", "JAK-STAT pathway"),
    # T cell master regulators
    ("TBX21", "IFNG", "Th1 differentiation"),
    ("TBX21", "CXCR3", "Th1 differentiation"),
    ("TBX21", "IL12RB2", "Th1 differentiation"),
    ("GATA3", "IL5", "Th2 differentiation"),
    ("GATA3", "IL13", "Th2 differentiation"),
    ("GATA3", "IL4", "Th2 differentiation"),
    ("RORC", "IL17A", "Th17 differentiation"),
    ("RORC", "IL17F", "Th17 differentiation"),
    ("RORC", "CCR6", "Th17 differentiation"),
    ("FOXP3", "IL2RA", "Treg differentiation"),
    ("FOXP3", "CTLA4", "Treg differentiation"),
    ("FOXP3", "TNFRSF18", "Treg differentiation"),
    ("BCL6", "CXCR5", "Tfh differentiation"),
    # B cell regulators
    ("PAX5", "CD19", "B cell identity"),
    ("PAX5", "CD79A", "B cell identity"),
    ("PAX5", "MS4A1", "B cell identity"),
    ("EBF1", "CD79A", "B cell development"),
    ("EBF1", "CD79B", "B cell development"),
    ("IRF4", "PRDM1", "Plasma cell differentiation"),
    ("IRF4", "XBP1", "Plasma cell differentiation"),
    ("PRDM1", "XBP1", "Plasma cell differentiation"),
    ("PRDM1", "SDC1", "Plasma cell differentiation"),
    # Myeloid regulators
    ("SPI1", "CSF1R", "Myeloid differentiation"),
    ("SPI1", "CD68", "Myeloid differentiation"),
    ("SPI1", "ITGAM", "Myeloid differentiation"),
    ("CEBPA", "CSF3R", "Myeloid differentiation"),
    ("CEBPA", "MPO", "Granulocyte differentiation"),
    ("CEBPB", "IL6", "Myeloid activation"),
    ("CEBPB", "CSF2", "Myeloid activation"),
    # AP-1 family
    ("FOS", "IL2", "AP-1 signaling"),
    ("FOS", "MMP9", "AP-1 signaling"),
    ("JUN", "IL2", "AP-1 signaling"),
    ("JUN", "VEGFA", "AP-1 signaling"),
    # IRF family
    ("IRF1", "CXCL10", "Interferon response"),
    ("IRF1", "GBP1", "Interferon response"),
    ("IRF7", "IFNB1", "Interferon response"),
    ("IRF8", "IL12B", "DC activation"),
    # MYC targets
    ("MYC", "CDK4", "Cell cycle"),
    ("MYC", "LDHA", "MYC-metabolism"),
    ("MYC", "NPM1", "MYC-proliferation"),
    ("MYC", "NCL", "MYC-proliferation"),
    # Other immune TFs
    ("RUNX3", "CD8A", "CD8 T cell identity"),
    ("RUNX3", "GZMB", "Cytotoxic function"),
    ("EOMES", "GZMB", "Cytotoxic function"),
    ("EOMES", "PRF1", "Cytotoxic function"),
    ("TCF7", "LEF1", "T cell development"),
    ("TCF7", "IL7R", "T cell development"),
    ("ID2", "GATA3", "ILC development"),
    ("BATF", "IL17A", "Th17/Tfh differentiation"),
    ("BATF", "IL10", "T cell exhaustion"),
    ("TOX", "PDCD1", "T cell exhaustion"),
    ("TOX", "HAVCR2", "T cell exhaustion"),
    ("NFE2L2", "HMOX1", "Antioxidant response"),
    ("NFE2L2", "NQO1", "Antioxidant response"),
    ("HIF1A", "VEGFA", "Hypoxia response"),
    ("HIF1A", "LDHA", "Hypoxia-metabolism"),
    ("HIF1A", "SLC2A1", "Hypoxia-metabolism"),
]

# Map gene symbols to Ensembl IDs
symbol_to_ensembl = {}
for ens_id, row in adata.var.iterrows():
    sym = row['feature_name']
    if pd.notna(sym):
        symbol_to_ensembl[sym] = ens_id

# Filter TF-target pairs to those present in the dataset
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
print(f"Unique TFs: {len(all_tf_syms)}")
print(f"Unique targets: {len(all_target_syms)}")

# ============================================================
# STEP 3: EXTRACT EXPRESSION & COMPUTE EDGE SCORES
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Compute Global and Per-Regime Edge Scores")
print("=" * 60)

# Get dense expression matrix for relevant genes
relevant_ensembl = set()
for _, _, _, tf_ens, tgt_ens in valid_pairs:
    relevant_ensembl.add(tf_ens)
    relevant_ensembl.add(tgt_ens)

# Extract expression for relevant genes
gene_idx = [i for i, g in enumerate(adata_filtered.var_names) if g in relevant_ensembl]
gene_names_subset = [adata_filtered.var_names[i] for i in gene_idx]

X_subset = adata_filtered.X[:, gene_idx]
if sparse.issparse(X_subset):
    X_subset = X_subset.toarray()

expr_df = pd.DataFrame(X_subset,
                        index=adata_filtered.obs_names,
                        columns=gene_names_subset)

regimes = adata_filtered.obs['regime'].values

def compute_spearman_safe(x, y):
    """Compute Spearman correlation, handling constant arrays."""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    rho, pval = spearmanr(x, y)
    if np.isnan(rho):
        return 0.0, 1.0
    return rho, pval

def compute_edge_scores(expr_df, regimes, valid_pairs):
    """Compute global and per-regime Spearman correlations for each TF-target pair."""
    results = []
    unique_regimes = sorted(set(regimes))

    for tf_sym, tgt_sym, source, tf_ens, tgt_ens in valid_pairs:
        if tf_ens not in expr_df.columns or tgt_ens not in expr_df.columns:
            continue

        tf_expr = expr_df[tf_ens].values
        tgt_expr = expr_df[tgt_ens].values

        # Global correlation
        rho_global, p_global = compute_spearman_safe(tf_expr, tgt_expr)

        # Per-regime correlations
        regime_scores = {}
        for regime in unique_regimes:
            mask = regimes == regime
            n_cells = mask.sum()
            if n_cells < 30:  # need minimum cells
                continue
            tf_r = tf_expr[mask]
            tgt_r = tgt_expr[mask]

            # Check if genes are expressed in this regime
            tf_frac = np.mean(tf_r > 0)
            tgt_frac = np.mean(tgt_r > 0)

            rho_regime, p_regime = compute_spearman_safe(tf_r, tgt_r)

            regime_scores[regime] = {
                'rho': rho_regime,
                'p': p_regime,
                'n_cells': n_cells,
                'tf_frac_expressed': tf_frac,
                'tgt_frac_expressed': tgt_frac,
            }

        results.append({
            'tf': tf_sym,
            'target': tgt_sym,
            'source': source,
            'tf_ens': tf_ens,
            'tgt_ens': tgt_ens,
            'rho_global': rho_global,
            'p_global': p_global,
            'regime_scores': regime_scores,
        })

    return results

print("Computing edge scores...")
edge_results = compute_edge_scores(expr_df, regimes, valid_pairs)
print(f"Computed scores for {len(edge_results)} edges across {len(valid_regimes)} regimes")

# ============================================================
# STEP 4: REGIME-SPECIFIC CONTRAST SCORING
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Regime-Specific Contrast Scoring")
print("=" * 60)

def compute_contrast_scores(edge_results):
    """
    For each edge, compute how much each regime's score deviates from the
    background (mean of other regimes). This identifies state-gated edges.
    """
    contrast_records = []

    for edge in edge_results:
        regime_rhos = {}
        for regime, scores in edge.get('regime_scores', {}).items():
            regime_rhos[regime] = scores['rho']

        if len(regime_rhos) < 3:
            continue

        regimes_list = sorted(regime_rhos.keys())
        rho_values = np.array([regime_rhos[r] for r in regimes_list])

        # For each regime: one-vs-rest contrast
        for i, regime in enumerate(regimes_list):
            rho_focal = rho_values[i]
            rho_rest = np.delete(rho_values, i)
            mean_rest = np.mean(rho_rest)
            std_rest = np.std(rho_rest) if len(rho_rest) > 1 else 0.01

            # Contrast: z-score of focal regime relative to rest
            contrast_z = (rho_focal - mean_rest) / max(std_rest, 0.01)

            # Also compute range-based specificity
            rho_range = np.max(rho_values) - np.min(rho_values)

            n_cells = edge['regime_scores'][regime]['n_cells']
            tf_frac = edge['regime_scores'][regime]['tf_frac_expressed']
            tgt_frac = edge['regime_scores'][regime]['tgt_frac_expressed']

            contrast_records.append({
                'tf': edge['tf'],
                'target': edge['target'],
                'source': edge['source'],
                'regime': regime,
                'rho_focal': rho_focal,
                'rho_global': edge['rho_global'],
                'mean_rest': mean_rest,
                'contrast_z': contrast_z,
                'rho_range': rho_range,
                'n_cells': n_cells,
                'tf_frac_expressed': tf_frac,
                'tgt_frac_expressed': tgt_frac,
            })

    return pd.DataFrame(contrast_records)

contrast_df = compute_contrast_scores(edge_results)
print(f"Contrast records: {len(contrast_df)}")
print(f"Unique edges: {contrast_df.groupby(['tf','target']).ngroups}")

# Classify edges
def classify_edge(row):
    """Classify an edge-regime pair as state-gated, always-on, or inactive."""
    if abs(row['contrast_z']) > 2.0 and abs(row['rho_focal']) > 0.1:
        if row['contrast_z'] > 0:
            return 'state-gated-active'
        else:
            return 'state-gated-inactive'
    elif abs(row['rho_global']) > 0.1 and abs(row['contrast_z']) < 1.0:
        return 'always-on'
    else:
        return 'weak/inactive'

contrast_df['edge_class'] = contrast_df.apply(classify_edge, axis=1)
print("\nEdge classification distribution:")
print(contrast_df['edge_class'].value_counts())

# ============================================================
# STEP 5: PERMUTATION-BASED STATISTICAL TESTING
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Permutation Testing for Contrast Scores")
print("=" * 60)

N_PERMS = 1000

def permutation_test_contrast(expr_df, regimes, valid_pairs, n_perms=N_PERMS):
    """
    Permutation test: shuffle regime labels to build null distribution
    of contrast z-scores. Compare observed to null.
    """
    print(f"Running {n_perms} permutations...")

    unique_regimes = sorted(set(regimes))

    # Collect observed contrast z-scores per (edge, regime)
    # We already have these in contrast_df

    # Build null distribution
    null_contrast_z = defaultdict(list)  # key: (tf, target, regime) -> list of null z-scores

    for perm_i in range(n_perms):
        if (perm_i + 1) % 100 == 0:
            print(f"  Permutation {perm_i + 1}/{n_perms}")

        # Shuffle regime labels
        perm_regimes = np.random.permutation(regimes)

        for edge_info in valid_pairs:
            tf_sym, tgt_sym, source, tf_ens, tgt_ens = edge_info
            if tf_ens not in expr_df.columns or tgt_ens not in expr_df.columns:
                continue

            tf_expr = expr_df[tf_ens].values
            tgt_expr = expr_df[tgt_ens].values

            # Compute per-regime rhos under permuted labels
            perm_rhos = {}
            for regime in unique_regimes:
                mask = perm_regimes == regime
                n_cells = mask.sum()
                if n_cells < 30:
                    continue
                rho_p, _ = compute_spearman_safe(tf_expr[mask], tgt_expr[mask])
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
                null_contrast_z[(tf_sym, tgt_sym, regime)].append(perm_z)

    return null_contrast_z

null_z = permutation_test_contrast(expr_df, regimes, valid_pairs, n_perms=N_PERMS)

# Compute empirical p-values
def compute_empirical_pvalues(contrast_df, null_z):
    """Compute two-sided empirical p-values from null distribution."""
    pvals = []
    for _, row in contrast_df.iterrows():
        key = (row['tf'], row['target'], row['regime'])
        if key in null_z and len(null_z[key]) > 0:
            null_vals = np.array(null_z[key])
            obs_z = abs(row['contrast_z'])
            p_emp = (np.sum(np.abs(null_vals) >= obs_z) + 1) / (len(null_vals) + 1)
            pvals.append(p_emp)
        else:
            pvals.append(1.0)
    return pvals

contrast_df['p_perm'] = compute_empirical_pvalues(contrast_df, null_z)

# FDR correction (Benjamini-Hochberg)
reject, p_adj, _, _ = multipletests(contrast_df['p_perm'], method='fdr_bh', alpha=0.05)
contrast_df['p_adj'] = p_adj
contrast_df['significant'] = reject

print(f"\nSignificant (FDR < 0.05): {contrast_df['significant'].sum()} / {len(contrast_df)}")
sig_df = contrast_df[contrast_df['significant']].copy()
print(f"\nSignificant state-gated-active: {(sig_df['edge_class'] == 'state-gated-active').sum()}")
print(f"Significant state-gated-inactive: {(sig_df['edge_class'] == 'state-gated-inactive').sum()}")

# ============================================================
# STEP 6: BOOTSTRAP STABILITY ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Bootstrap Stability Analysis")
print("=" * 60)

N_BOOTS = 500

def bootstrap_stability(expr_df, regimes, valid_pairs, n_boots=N_BOOTS):
    """Bootstrap within each regime to assess score stability."""
    unique_regimes = sorted(set(regimes))

    stability_results = {}  # (tf, target, regime) -> list of bootstrap rhos

    for boot_i in range(n_boots):
        if (boot_i + 1) % 100 == 0:
            print(f"  Bootstrap {boot_i + 1}/{n_boots}")

        for edge_info in valid_pairs:
            tf_sym, tgt_sym, source, tf_ens, tgt_ens = edge_info
            if tf_ens not in expr_df.columns or tgt_ens not in expr_df.columns:
                continue

            for regime in unique_regimes:
                mask = regimes == regime
                n_cells = mask.sum()
                if n_cells < 30:
                    continue

                # Bootstrap sample within regime
                idx = np.where(mask)[0]
                boot_idx = np.random.choice(idx, size=len(idx), replace=True)

                tf_b = expr_df[tf_ens].values[boot_idx]
                tgt_b = expr_df[tgt_ens].values[boot_idx]

                rho_b, _ = compute_spearman_safe(tf_b, tgt_b)

                key = (tf_sym, tgt_sym, regime)
                if key not in stability_results:
                    stability_results[key] = []
                stability_results[key].append(rho_b)

    return stability_results

print(f"Running {N_BOOTS} bootstrap iterations...")
boot_results = bootstrap_stability(expr_df, regimes, valid_pairs, n_boots=N_BOOTS)

# Compute bootstrap CIs and stability metrics
boot_ci_low = []
boot_ci_high = []
boot_std = []
for _, row in contrast_df.iterrows():
    key = (row['tf'], row['target'], row['regime'])
    if key in boot_results and len(boot_results[key]) > 0:
        vals = np.array(boot_results[key])
        boot_ci_low.append(np.percentile(vals, 2.5))
        boot_ci_high.append(np.percentile(vals, 97.5))
        boot_std.append(np.std(vals))
    else:
        boot_ci_low.append(np.nan)
        boot_ci_high.append(np.nan)
        boot_std.append(np.nan)

contrast_df['boot_ci_low'] = boot_ci_low
contrast_df['boot_ci_high'] = boot_ci_high
contrast_df['boot_std'] = boot_std

# Stability: CI width
contrast_df['ci_width'] = contrast_df['boot_ci_high'] - contrast_df['boot_ci_low']

print(f"Mean CI width: {contrast_df['ci_width'].mean():.4f}")
print(f"Median CI width: {contrast_df['ci_width'].median():.4f}")

# A stable state-gated edge: significant, large contrast, narrow CI
stable_gated = contrast_df[
    (contrast_df['significant']) &
    (contrast_df['edge_class'].isin(['state-gated-active', 'state-gated-inactive'])) &
    (contrast_df['ci_width'] < contrast_df['ci_width'].quantile(0.75))
].copy()
print(f"\nStable state-gated edges (significant, narrow CI): {len(stable_gated)}")

# ============================================================
# STEP 7: PERTURBATION VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Perturbation Data Cross-Reference")
print("=" * 60)

perturb_validation = {}
perturb_files = []
for f in os.listdir(PERTURB_DIR):
    if f.endswith('.h5ad'):
        perturb_files.append(f)
        print(f"Found perturbation file: {f}")

# Load perturbation datasets and check for overlapping TFs
for pf in perturb_files:
    try:
        pdata = sc.read_h5ad(os.path.join(PERTURB_DIR, pf))
        print(f"\n--- {pf} ---")
        print(f"  Shape: {pdata.shape}")
        print(f"  obs columns: {list(pdata.obs.columns)[:10]}")

        # Check for perturbation info
        for col in pdata.obs.columns:
            if 'perturb' in col.lower() or 'guide' in col.lower() or 'gene' in col.lower() or 'target' in col.lower():
                print(f"  Perturbation column '{col}': {pdata.obs[col].nunique()} unique values")
                vals = pdata.obs[col].value_counts().head(10)
                print(f"    Top values: {dict(vals)}")

        # Check gene overlap
        perturb_gene_names = set()
        if 'feature_name' in pdata.var.columns:
            perturb_gene_names = set(pdata.var['feature_name'].dropna().values)
        elif 'gene_name' in pdata.var.columns:
            perturb_gene_names = set(pdata.var['gene_name'].dropna().values)
        else:
            perturb_gene_names = set(pdata.var_names)

        tf_overlap = all_tf_syms.intersection(perturb_gene_names)
        target_overlap = all_target_syms.intersection(perturb_gene_names)
        print(f"  TF overlap: {len(tf_overlap)} / {len(all_tf_syms)}")
        print(f"  Target overlap: {len(target_overlap)} / {len(all_target_syms)}")

        perturb_validation[pf] = {
            'shape': pdata.shape,
            'tf_overlap': tf_overlap,
            'target_overlap': target_overlap,
            'gene_names': perturb_gene_names,
        }
        del pdata
    except Exception as e:
        print(f"  Error loading {pf}: {e}")

# For datasets with perturbation info, check if perturbed TFs show expected
# downstream effects consistent with our edge predictions
print("\n--- Cross-referencing perturbation evidence ---")
perturb_evidence = []
for pf, pinfo in perturb_validation.items():
    # Check which of our TF-target edges involve perturbed genes
    for edge in edge_results:
        tf = edge['tf']
        target = edge['target']
        if tf in pinfo.get('tf_overlap', set()) and target in pinfo.get('target_overlap', set()):
            perturb_evidence.append({
                'tf': tf,
                'target': target,
                'perturb_dataset': pf,
                'source': edge['source'],
            })

if perturb_evidence:
    perturb_ev_df = pd.DataFrame(perturb_evidence)
    print(f"\nEdges with perturbation support: {len(perturb_ev_df)}")
    print(perturb_ev_df.drop_duplicates(['tf', 'target'])[['tf', 'target', 'source']].to_string())
else:
    print("No direct perturbation overlap found for our TF-target pairs.")

# ============================================================
# STEP 8: SAVE INTERMEDIATE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Save Results")
print("=" * 60)

contrast_df.to_csv(os.path.join(IMPL_DIR, "contrast_scores.csv"), index=False)
if len(stable_gated) > 0:
    stable_gated.to_csv(os.path.join(IMPL_DIR, "stable_gated_edges.csv"), index=False)

# Save edge results summary
edge_summary = []
for e in edge_results:
    regime_rhos = {r: s['rho'] for r, s in e['regime_scores'].items()}
    if regime_rhos:
        max_regime = max(regime_rhos, key=lambda r: abs(regime_rhos[r]))
        min_regime = min(regime_rhos, key=lambda r: abs(regime_rhos[r]))
        rho_range = max(regime_rhos.values()) - min(regime_rhos.values())
    else:
        max_regime = min_regime = "N/A"
        rho_range = 0
    edge_summary.append({
        'tf': e['tf'],
        'target': e['target'],
        'source': e['source'],
        'rho_global': e['rho_global'],
        'rho_range': rho_range,
        'max_regime': max_regime,
        'min_regime': min_regime,
        'n_regimes': len(e['regime_scores']),
    })
edge_summary_df = pd.DataFrame(edge_summary).sort_values('rho_range', ascending=False)
edge_summary_df.to_csv(os.path.join(IMPL_DIR, "edge_summary.csv"), index=False)
print(f"Saved contrast_scores.csv, stable_gated_edges.csv, edge_summary.csv")

# Print top state-gated edges
print("\n--- Top 20 Edges by Regime Range (most variable across states) ---")
print(edge_summary_df.head(20).to_string(index=False))

# Print top significant state-gated edges
sig_active = contrast_df[
    (contrast_df['significant']) &
    (contrast_df['edge_class'] == 'state-gated-active')
].sort_values('contrast_z', ascending=False)

print(f"\n--- Top 20 Significant State-Gated-Active (by contrast_z) ---")
cols_show = ['tf', 'target', 'regime', 'rho_focal', 'rho_global', 'contrast_z', 'p_adj', 'ci_width']
if len(sig_active) > 0:
    print(sig_active[cols_show].head(20).to_string(index=False))
else:
    print("No significant state-gated-active edges found.")

print("\nDone with main pipeline!")
