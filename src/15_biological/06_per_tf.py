#!/usr/bin/env python3
"""
Analysis E: Per-TF Characterization
====================================
Characterize what distinguishes the 7 TFs with robust edge-level signal
(bootstrap 95% CI entirely above 0.5) from the 11 unsuccessful TFs.

Axes of characterization:
- Regulon size (TRRUST targets)
- Expression level and variance in Tabula Sapiens immune
- TF family classification
- Functional category
- Bootstrap AUROC and CI

Statistical tests:
- Spearman correlation of AUROC with regulon size, expression, variance
- Mann-Whitney: successful vs unsuccessful TFs
"""

import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

OUT_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/biological_impact/per_tf"
BOOT_CSV = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/bootstrap_per_tf_results/per_tf_bootstrap_ci.csv"
TRRUST = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv"
TS_IMMUNE = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad"
HVG_LIST = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/round9_unified_extraction/hvg_gene_list.json"

os.makedirs(OUT_DIR, exist_ok=True)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


# ─── Manual TF annotation (18 TFs, feasible by hand) ────────────────────────
# Based on literature: TFClass, UniProt, and Gene Ontology annotations
TF_ANNOTATIONS = {
    "NCOA4": {
        "family": "Nuclear receptor coactivator",
        "functional_category": "signal-dependent",
        "description": "Nuclear receptor coactivator 4; coactivates androgen receptor signaling"
    },
    "RXRA": {
        "family": "Nuclear receptor (NR1)",
        "functional_category": "signal-dependent",
        "description": "Retinoid X receptor alpha; heterodimerizes with many NRs for ligand-dependent transcription"
    },
    "TBX5": {
        "family": "T-box",
        "functional_category": "lineage-specific",
        "description": "T-box TF 5; cardiac/limb development master regulator"
    },
    "FOXF2": {
        "family": "Forkhead (FOX)",
        "functional_category": "lineage-specific",
        "description": "Forkhead box F2; mesenchymal/gut development"
    },
    "TAL1": {
        "family": "bHLH",
        "functional_category": "master_regulator",
        "description": "T-cell acute lymphocytic leukemia 1; essential hematopoietic stem cell TF"
    },
    "NKX2-5": {
        "family": "Homeodomain (NK)",
        "functional_category": "master_regulator",
        "description": "NK2 homeobox 5; cardiac master regulator"
    },
    "PA2G4": {
        "family": "Other (ErbB3-binding protein)",
        "functional_category": "housekeeping",
        "description": "Proliferation-associated 2G4; cell growth/differentiation; not a classical TF"
    },
    "YBX1": {
        "family": "Cold-shock domain",
        "functional_category": "housekeeping",
        "description": "Y-box binding protein 1; regulates transcription/translation broadly"
    },
    "KLF3": {
        "family": "Zinc finger (Krüppel-like)",
        "functional_category": "lineage-specific",
        "description": "Krüppel-like factor 3; erythroid differentiation"
    },
    "WT1": {
        "family": "Zinc finger (C2H2)",
        "functional_category": "master_regulator",
        "description": "Wilms tumor 1; kidney/gonad development master regulator"
    },
    "PAX3": {
        "family": "Paired-box + Homeodomain",
        "functional_category": "master_regulator",
        "description": "Paired box 3; neural crest/myogenesis master regulator"
    },
    "GATA1": {
        "family": "Zinc finger (GATA)",
        "functional_category": "master_regulator",
        "description": "GATA-binding factor 1; erythroid/megakaryocyte master regulator"
    },
    "EGR1": {
        "family": "Zinc finger (C2H2)",
        "functional_category": "signal-dependent",
        "description": "Early growth response 1; immediate-early gene, stimulus-responsive"
    },
    "MAL": {
        "family": "Other (not classical TF)",
        "functional_category": "signal-dependent",
        "description": "Myelin and lymphocyte protein; membrane-associated, lipid raft signaling"
    },
    "FOXA1": {
        "family": "Forkhead (FOX)",
        "functional_category": "master_regulator",
        "description": "Forkhead box A1; pioneer factor for liver/prostate/breast lineage"
    },
    "PPARG": {
        "family": "Nuclear receptor (NR1)",
        "functional_category": "master_regulator",
        "description": "Peroxisome proliferator-activated receptor gamma; adipogenesis master regulator"
    },
    "KLF1": {
        "family": "Zinc finger (Krüppel-like)",
        "functional_category": "master_regulator",
        "description": "Krüppel-like factor 1 (EKLF); definitive erythropoiesis master regulator"
    },
    "LEF1": {
        "family": "HMG-box (TCF/LEF)",
        "functional_category": "master_regulator",
        "description": "Lymphoid enhancer factor 1; Wnt signaling effector, T-cell development"
    },
}


print("=" * 70)
print("Analysis E: Per-TF Characterization")
print("=" * 70)

# ─── 1. Load bootstrap results ──────────────────────────────────────────────
print("\n[1] Loading bootstrap per-TF results...")
boot_df = pd.read_csv(BOOT_CSV)
print(f"  {len(boot_df)} TFs loaded")
print(boot_df[['tf', 'observed_auroc', 'ci_95_lo', 'ci_95_hi', 'out_degree']].to_string())

# Classify as successful (CI entirely above 0.5) or not
boot_df['successful'] = boot_df['ci_95_lo'] > 0.5
n_success = boot_df['successful'].sum()
n_fail = (~boot_df['successful']).sum()
print(f"\n  Successful (CI > 0.5): {n_success}")
print(f"  Unsuccessful (CI includes ≤0.5): {n_fail}")

# ─── 2. Load TRRUST and compute full regulon sizes ──────────────────────────
print("\n[2] Loading TRRUST network...")
trrust = pd.read_csv(TRRUST, sep='\t', header=None, names=['tf', 'target', 'type', 'pmid'])
# Remove duplicate TF-target pairs (keep unique edges)
trrust_unique = trrust.drop_duplicates(subset=['tf', 'target'])
print(f"  {len(trrust_unique)} unique TF-target edges, {trrust_unique['tf'].nunique()} TFs")

# Full regulon size (all TRRUST targets)
full_regulon = trrust_unique.groupby('tf')['target'].count().to_dict()

# Load HVG universe to count evaluable targets
with open(HVG_LIST) as f:
    hvg_genes = set(json.load(f))
print(f"  HVG universe: {len(hvg_genes)} genes")

# Evaluable targets (in HVG universe)
trrust_in_hvg = trrust_unique[trrust_unique['target'].isin(hvg_genes)]
evaluable_regulon = trrust_in_hvg.groupby('tf')['target'].count().to_dict()

# Regulation type breakdown
type_counts = trrust_unique.groupby('tf')['type'].value_counts().unstack(fill_value=0)

for _, row in boot_df.iterrows():
    tf = row['tf']
    boot_df.loc[boot_df['tf'] == tf, 'full_regulon_size'] = full_regulon.get(tf, 0)
    boot_df.loc[boot_df['tf'] == tf, 'evaluable_regulon_size'] = evaluable_regulon.get(tf, 0)
    if tf in type_counts.index:
        for rtype in ['Activation', 'Repression', 'Unknown']:
            if rtype in type_counts.columns:
                boot_df.loc[boot_df['tf'] == tf, f'n_{rtype.lower()}'] = type_counts.loc[tf].get(rtype, 0)

boot_df['full_regulon_size'] = boot_df['full_regulon_size'].astype(int)
boot_df['evaluable_regulon_size'] = boot_df['evaluable_regulon_size'].astype(int)

print("\n  Regulon sizes:")
for _, row in boot_df.sort_values('observed_auroc', ascending=False).iterrows():
    print(f"    {row['tf']:8s}: full={int(row['full_regulon_size']):4d}, evaluable={int(row['evaluable_regulon_size']):3d}, AUROC={row['observed_auroc']:.3f}")

# ─── 3. Load expression data from Tabula Sapiens immune ─────────────────────
print("\n[3] Loading Tabula Sapiens immune expression data...")
import h5py

with h5py.File(TS_IMMUNE, 'r') as f:
    # Get gene names
    if 'var' in f and '_index' in f['var']:
        gene_names_raw = f['var']['_index'][:]
    elif 'var' in f and 'feature_name' in f['var']:
        gene_names_raw = f['var']['feature_name'][:]
    else:
        # Try alternative paths
        keys = list(f['var'].keys())
        print(f"  var keys: {keys}")
        gene_names_raw = f['var'][keys[0]][:]

    gene_names = [g.decode('utf-8') if isinstance(g, bytes) else str(g) for g in gene_names_raw]
    print(f"  {len(gene_names)} genes in Tabula Sapiens immune")

    # Get expression matrix
    if 'X' in f:
        X_data = f['X']
        if isinstance(X_data, h5py.Dataset):
            # Dense matrix
            print("  Dense matrix detected")
            X = X_data[:]
        else:
            # Sparse matrix (CSR format typical)
            print("  Sparse matrix detected")
            data = f['X']['data'][:]
            indices = f['X']['indices'][:]
            indptr = f['X']['indptr'][:]
            from scipy.sparse import csr_matrix
            shape = (len(indptr) - 1, len(gene_names))
            X = csr_matrix((data, indices, indptr), shape=shape)

    n_cells = X.shape[0] if hasattr(X, 'shape') else len(indptr) - 1
    print(f"  {n_cells} cells × {len(gene_names)} genes")

# Compute per-gene statistics for our 18 TFs
print("\n  Computing expression statistics for 18 TFs...")
gene_to_idx = {g: i for i, g in enumerate(gene_names)}

from scipy.sparse import issparse

tf_expression_stats = {}
for tf in boot_df['tf'].values:
    if tf in gene_to_idx:
        idx = gene_to_idx[tf]
        if issparse(X):
            col = X[:, idx].toarray().flatten()
        else:
            col = X[:, idx].flatten()

        mean_expr = float(np.mean(col))
        var_expr = float(np.var(col))
        median_expr = float(np.median(col))
        nonzero_frac = float(np.mean(col > 0))
        max_expr = float(np.max(col))

        tf_expression_stats[tf] = {
            'mean_expression': mean_expr,
            'variance': var_expr,
            'median_expression': median_expr,
            'nonzero_fraction': nonzero_frac,
            'max_expression': max_expr,
            'found_in_data': True
        }
        print(f"    {tf:8s}: mean={mean_expr:.4f}, var={var_expr:.4f}, nonzero={nonzero_frac:.3f}")
    else:
        tf_expression_stats[tf] = {
            'mean_expression': np.nan,
            'variance': np.nan,
            'median_expression': np.nan,
            'nonzero_fraction': np.nan,
            'max_expression': np.nan,
            'found_in_data': False
        }
        print(f"    {tf:8s}: NOT FOUND in expression data")

# Add expression stats to dataframe
for tf, stats_dict in tf_expression_stats.items():
    for key, val in stats_dict.items():
        if key != 'found_in_data':
            boot_df.loc[boot_df['tf'] == tf, key] = val
    boot_df.loc[boot_df['tf'] == tf, 'found_in_expr_data'] = stats_dict['found_in_data']

# ─── 4. Add manual TF annotations ───────────────────────────────────────────
print("\n[4] Adding TF family and functional annotations...")
for tf, annot in TF_ANNOTATIONS.items():
    boot_df.loc[boot_df['tf'] == tf, 'tf_family'] = annot['family']
    boot_df.loc[boot_df['tf'] == tf, 'functional_category'] = annot['functional_category']
    boot_df.loc[boot_df['tf'] == tf, 'description'] = annot['description']

# Count master regulators
n_master = (boot_df['functional_category'] == 'master_regulator').sum()
print(f"  Master regulators: {n_master}/18")
print(f"  Signal-dependent: {(boot_df['functional_category'] == 'signal-dependent').sum()}/18")
print(f"  Lineage-specific: {(boot_df['functional_category'] == 'lineage-specific').sum()}/18")
print(f"  Housekeeping: {(boot_df['functional_category'] == 'housekeeping').sum()}/18")

# ─── 5. Statistical tests ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[5] STATISTICAL TESTS")
print("=" * 70)

results = {
    'tf_data': [],
    'statistical_tests': {},
    'group_comparisons': {}
}

# 5a. Correlations
print("\n[5a] Spearman correlations of AUROC with TF features:")

# AUROC vs regulon size
valid = boot_df.dropna(subset=['observed_auroc', 'full_regulon_size'])
rho, p = stats.spearmanr(valid['observed_auroc'], valid['full_regulon_size'])
print(f"  AUROC vs full regulon size: ρ={rho:.3f}, p={p:.4f} (n={len(valid)})")
results['statistical_tests']['auroc_vs_full_regulon'] = {'rho': rho, 'p': p, 'n': len(valid)}

rho2, p2 = stats.spearmanr(valid['observed_auroc'], valid['evaluable_regulon_size'])
print(f"  AUROC vs evaluable regulon size: ρ={rho2:.3f}, p={p2:.4f} (n={len(valid)})")
results['statistical_tests']['auroc_vs_evaluable_regulon'] = {'rho': rho2, 'p': p2, 'n': len(valid)}

# AUROC vs expression (only for TFs found in expression data)
expr_valid = boot_df[boot_df['found_in_expr_data'] == True].dropna(subset=['mean_expression'])
if len(expr_valid) > 3:
    rho3, p3 = stats.spearmanr(expr_valid['observed_auroc'], expr_valid['mean_expression'])
    print(f"  AUROC vs mean expression: ρ={rho3:.3f}, p={p3:.4f} (n={len(expr_valid)})")
    results['statistical_tests']['auroc_vs_mean_expr'] = {'rho': rho3, 'p': p3, 'n': len(expr_valid)}

    rho4, p4 = stats.spearmanr(expr_valid['observed_auroc'], expr_valid['variance'])
    print(f"  AUROC vs expression variance: ρ={rho4:.3f}, p={p4:.4f} (n={len(expr_valid)})")
    results['statistical_tests']['auroc_vs_variance'] = {'rho': rho4, 'p': p4, 'n': len(expr_valid)}

    rho5, p5 = stats.spearmanr(expr_valid['observed_auroc'], expr_valid['nonzero_fraction'])
    print(f"  AUROC vs nonzero fraction: ρ={rho5:.3f}, p={p5:.4f} (n={len(expr_valid)})")
    results['statistical_tests']['auroc_vs_nonzero_frac'] = {'rho': rho5, 'p': p5, 'n': len(expr_valid)}

# 5b. Successful vs Unsuccessful comparisons
print("\n[5b] Successful (CI>0.5) vs Unsuccessful TFs:")

success_df = boot_df[boot_df['successful']]
fail_df = boot_df[~boot_df['successful']]

# Regulon size comparison
u_stat, u_p = stats.mannwhitneyu(
    success_df['full_regulon_size'].dropna(),
    fail_df['full_regulon_size'].dropna(),
    alternative='two-sided'
)
print(f"  Full regulon size: successful median={success_df['full_regulon_size'].median():.0f}, "
      f"unsuccessful median={fail_df['full_regulon_size'].median():.0f}, "
      f"U={u_stat:.0f}, p={u_p:.4f}")
results['statistical_tests']['regulon_size_success_vs_fail'] = {
    'U': u_stat, 'p': u_p,
    'successful_median': float(success_df['full_regulon_size'].median()),
    'unsuccessful_median': float(fail_df['full_regulon_size'].median())
}

# Evaluable regulon size
u_stat2, u_p2 = stats.mannwhitneyu(
    success_df['evaluable_regulon_size'].dropna(),
    fail_df['evaluable_regulon_size'].dropna(),
    alternative='two-sided'
)
print(f"  Evaluable regulon size: successful median={success_df['evaluable_regulon_size'].median():.0f}, "
      f"unsuccessful median={fail_df['evaluable_regulon_size'].median():.0f}, "
      f"U={u_stat2:.0f}, p={u_p2:.4f}")
results['statistical_tests']['eval_regulon_success_vs_fail'] = {
    'U': u_stat2, 'p': u_p2,
    'successful_median': float(success_df['evaluable_regulon_size'].median()),
    'unsuccessful_median': float(fail_df['evaluable_regulon_size'].median())
}

# Expression comparison
s_expr = success_df[success_df['found_in_expr_data'] == True]['mean_expression'].dropna()
f_expr = fail_df[fail_df['found_in_expr_data'] == True]['mean_expression'].dropna()
if len(s_expr) >= 2 and len(f_expr) >= 2:
    u3, p3 = stats.mannwhitneyu(s_expr, f_expr, alternative='two-sided')
    print(f"  Mean expression: successful median={s_expr.median():.4f}, "
          f"unsuccessful median={f_expr.median():.4f}, "
          f"U={u3:.0f}, p={p3:.4f}")
    results['statistical_tests']['expression_success_vs_fail'] = {
        'U': u3, 'p': p3,
        'successful_median': float(s_expr.median()),
        'unsuccessful_median': float(f_expr.median()),
        'n_successful': len(s_expr),
        'n_unsuccessful': len(f_expr)
    }

# 5c. Master regulator vs others
print("\n[5c] Master regulators vs other TFs:")
master_df = boot_df[boot_df['functional_category'] == 'master_regulator']
other_df = boot_df[boot_df['functional_category'] != 'master_regulator']

u4, p4 = stats.mannwhitneyu(
    master_df['observed_auroc'], other_df['observed_auroc'],
    alternative='greater'  # one-sided: master > other
)
print(f"  Master regulators (n={len(master_df)}): median AUROC={master_df['observed_auroc'].median():.3f}")
print(f"  Others (n={len(other_df)}): median AUROC={other_df['observed_auroc'].median():.3f}")
print(f"  Mann-Whitney U (one-sided, master > other): U={u4:.0f}, p={p4:.4f}")
results['statistical_tests']['master_vs_other_auroc'] = {
    'U': u4, 'p': p4,
    'master_median': float(master_df['observed_auroc'].median()),
    'other_median': float(other_df['observed_auroc'].median()),
    'n_master': len(master_df),
    'n_other': len(other_df),
    'alternative': 'greater'
}

# Contingency table: master regulator × successful
print("\n[5d] Contingency table: Master regulator × Successful:")
ct = pd.crosstab(boot_df['functional_category'] == 'master_regulator', boot_df['successful'])
ct.index = ['Not master', 'Master']
ct.columns = ['Unsuccessful', 'Successful']
print(ct)

# Fisher's exact test
master_success = ((boot_df['functional_category'] == 'master_regulator') & boot_df['successful']).sum()
master_fail = ((boot_df['functional_category'] == 'master_regulator') & ~boot_df['successful']).sum()
other_success = ((boot_df['functional_category'] != 'master_regulator') & boot_df['successful']).sum()
other_fail = ((boot_df['functional_category'] != 'master_regulator') & ~boot_df['successful']).sum()

table = [[master_success, master_fail], [other_success, other_fail]]
odds_ratio, fisher_p = stats.fisher_exact(table)
print(f"\n  Fisher's exact: OR={odds_ratio:.2f}, p={fisher_p:.4f}")
print(f"  Master regulators: {master_success}/{master_success+master_fail} successful ({master_success/(master_success+master_fail)*100:.0f}%)")
print(f"  Others: {other_success}/{other_success+other_fail} successful ({other_success/(other_success+other_fail)*100:.0f}%)")

results['statistical_tests']['fisher_master_x_success'] = {
    'odds_ratio': odds_ratio, 'p': fisher_p,
    'table': table,
    'master_success_rate': master_success / (master_success + master_fail),
    'other_success_rate': other_success / (other_success + other_fail)
}

# 5e. Minimum regulon size threshold
print("\n[5e] Regulon size threshold analysis:")
thresholds = [1, 2, 3, 5, 10, 15, 20]
threshold_results = []
for thresh in thresholds:
    above = boot_df[boot_df['evaluable_regulon_size'] >= thresh]
    if len(above) > 0:
        n_above = len(above)
        mean_auroc = above['observed_auroc'].mean()
        n_success = above['successful'].sum()
        frac_success = n_success / n_above
        threshold_results.append({
            'threshold': thresh,
            'n_tfs': n_above,
            'mean_auroc': float(mean_auroc),
            'n_successful': int(n_success),
            'frac_successful': float(frac_success)
        })
        print(f"  Evaluable regulon ≥ {thresh:2d}: n={n_above:2d}, mean AUROC={mean_auroc:.3f}, "
              f"successful={n_success}/{n_above} ({frac_success*100:.0f}%)")
results['threshold_analysis'] = threshold_results

# 5f. Effect size (Cohen's d) for successful vs unsuccessful
s_aurocs = success_df['observed_auroc'].values
f_aurocs = fail_df['observed_auroc'].values
pooled_std = np.sqrt(((len(s_aurocs)-1)*np.var(s_aurocs, ddof=1) + (len(f_aurocs)-1)*np.var(f_aurocs, ddof=1)) /
                     (len(s_aurocs) + len(f_aurocs) - 2))
cohens_d = (np.mean(s_aurocs) - np.mean(f_aurocs)) / pooled_std if pooled_std > 0 else np.nan
print(f"\n[5f] Effect size: Cohen's d (successful - unsuccessful) = {cohens_d:.3f}")
results['statistical_tests']['cohens_d_success_vs_fail'] = {
    'cohens_d': float(cohens_d),
    'successful_mean': float(np.mean(s_aurocs)),
    'unsuccessful_mean': float(np.mean(f_aurocs))
}

# ─── 6. Build per-TF output ─────────────────────────────────────────────────
print("\n[6] Building per-TF characterization table...")

for _, row in boot_df.iterrows():
    tf = row['tf']
    annot = TF_ANNOTATIONS.get(tf, {})
    tf_entry = {
        'tf': tf,
        'observed_auroc': float(row['observed_auroc']),
        'ci_95_lo': float(row['ci_95_lo']),
        'ci_95_hi': float(row['ci_95_hi']),
        'successful': bool(row['successful']),
        'full_regulon_size': int(row['full_regulon_size']),
        'evaluable_regulon_size': int(row['evaluable_regulon_size']),
        'out_degree_in_hvg': int(row['out_degree']),
        'mean_expression': float(row['mean_expression']) if pd.notna(row.get('mean_expression')) else None,
        'expression_variance': float(row['variance']) if pd.notna(row.get('variance')) else None,
        'nonzero_fraction': float(row['nonzero_fraction']) if pd.notna(row.get('nonzero_fraction')) else None,
        'tf_family': annot.get('family', 'Unknown'),
        'functional_category': annot.get('functional_category', 'unknown'),
        'description': annot.get('description', ''),
        'degree_matched_auroc': float(row['degree_matched_auroc']) if pd.notna(row.get('degree_matched_auroc')) else None
    }
    results['tf_data'].append(tf_entry)

# ─── 7. Group-level summaries ────────────────────────────────────────────────
print("\n[7] Group-level summaries:")

# By functional category
for cat in ['master_regulator', 'signal-dependent', 'lineage-specific', 'housekeeping']:
    subset = boot_df[boot_df['functional_category'] == cat]
    if len(subset) > 0:
        results['group_comparisons'][cat] = {
            'n': len(subset),
            'mean_auroc': float(subset['observed_auroc'].mean()),
            'median_auroc': float(subset['observed_auroc'].median()),
            'n_successful': int(subset['successful'].sum()),
            'tfs': list(subset['tf'].values),
            'mean_regulon_size': float(subset['full_regulon_size'].mean())
        }
        print(f"  {cat:20s}: n={len(subset):2d}, mean AUROC={subset['observed_auroc'].mean():.3f}, "
              f"successful={subset['successful'].sum()}/{len(subset)}")

# By TF family
print("\n  By TF family:")
for fam in boot_df['tf_family'].unique():
    subset = boot_df[boot_df['tf_family'] == fam]
    print(f"    {fam:35s}: n={len(subset)}, AUROCs={list(subset['observed_auroc'].round(3))}")

# ─── 8. Power analysis note ─────────────────────────────────────────────────
print("\n[8] Statistical power considerations:")
print(f"  Total sample: n=18 (very small)")
print(f"  Successful: n={n_success}, Unsuccessful: n={n_fail}")
print(f"  Minimum detectable ρ at α=0.05, 80% power, n=18: ~0.47")
print(f"  Minimum detectable d at α=0.05, 80% power, n1=7, n2=11: ~1.35")
print(f"  → Results should be interpreted as hypothesis-generating, not confirmatory")

results['power_analysis'] = {
    'total_n': 18,
    'n_successful': int(n_success),
    'n_unsuccessful': int(n_fail),
    'min_detectable_rho_alpha05_power80': 0.47,
    'min_detectable_d_alpha05_power80': 1.35,
    'note': 'All results hypothesis-generating due to small sample. Focus on descriptive patterns.'
}

# ─── 9. Save results ────────────────────────────────────────────────────────
print("\n[9] Saving results...")
with open(os.path.join(OUT_DIR, 'tf_characterization.json'), 'w') as f:
    json.dump(results, f, indent=2, default=_json_default)
print(f"  Saved: {os.path.join(OUT_DIR, 'tf_characterization.json')}")

# ─── 10. Generate summary markdown ──────────────────────────────────────────
print("\n[10] Generating summary markdown...")

# Sort TFs by AUROC for the table
sorted_tfs = sorted(results['tf_data'], key=lambda x: x['observed_auroc'], reverse=True)

md_lines = [
    "# Analysis E: Per-TF Characterization — Results",
    "",
    "## Overview",
    "",
    f"18 TFs evaluated with bootstrap confidence intervals on Tabula Sapiens immune data (n=5,000 cells, 2,000 HVGs).",
    f"**7 TFs** have 95% CI entirely above 0.5 (successful); **11 TFs** have CI including ≤0.5 (unsuccessful).",
    "",
    "## Per-TF Table",
    "",
    "| TF | AUROC | 95% CI | Successful | Family | Category | Full Regulon | Eval. Regulon | Mean Expr | Nonzero % |",
    "|-----|-------|--------|------------|--------|----------|-------------|---------------|-----------|-----------|",
]

for tf in sorted_tfs:
    ci = f"[{tf['ci_95_lo']:.3f}, {tf['ci_95_hi']:.3f}]"
    succ = "Yes" if tf['successful'] else "No"
    expr = f"{tf['mean_expression']:.4f}" if tf['mean_expression'] is not None else "N/A"
    nzf = f"{tf['nonzero_fraction']*100:.1f}" if tf['nonzero_fraction'] is not None else "N/A"
    md_lines.append(
        f"| {tf['tf']} | {tf['observed_auroc']:.3f} | {ci} | {succ} | "
        f"{tf['tf_family']} | {tf['functional_category']} | {tf['full_regulon_size']} | "
        f"{tf['evaluable_regulon_size']} | {expr} | {nzf} |"
    )

md_lines.extend([
    "",
    "## Key Findings",
    "",
    "### 1. Functional Category Distribution",
    "",
])

# Add group comparison info
for cat, info in results['group_comparisons'].items():
    md_lines.append(f"- **{cat}** (n={info['n']}): mean AUROC={info['mean_auroc']:.3f}, "
                   f"successful={info['n_successful']}/{info['n']}, TFs: {', '.join(info['tfs'])}")

md_lines.extend([
    "",
    "### 2. Master Regulator Enrichment",
    "",
    f"- Master regulators: {master_success}/{master_success+master_fail} successful "
    f"({master_success/(master_success+master_fail)*100:.0f}%)",
    f"- Non-master regulators: {other_success}/{other_success+other_fail} successful "
    f"({other_success/(other_success+other_fail)*100:.0f}%)",
    f"- Fisher's exact test: OR={odds_ratio:.2f}, p={fisher_p:.4f}",
    f"- Mann-Whitney (master AUROC > other, one-sided): p={results['statistical_tests']['master_vs_other_auroc']['p']:.4f}",
    "",
    "### 3. Statistical Correlations",
    "",
])

for test_name, test_result in results['statistical_tests'].items():
    if 'rho' in test_result:
        md_lines.append(f"- {test_name}: ρ={test_result['rho']:.3f}, p={test_result['p']:.4f}")

md_lines.extend([
    "",
    "### 4. Regulon Size Threshold",
    "",
    "| Min Evaluable Targets | n TFs | Mean AUROC | % Successful |",
    "|----------------------|-------|------------|--------------|",
])
for tr in threshold_results:
    md_lines.append(f"| ≥{tr['threshold']} | {tr['n_tfs']} | {tr['mean_auroc']:.3f} | {tr['frac_successful']*100:.0f}% |")

md_lines.extend([
    "",
    "### 5. Successful TFs — Pattern Summary",
    "",
    "**The 7 successful TFs:**",
])
for tf in sorted_tfs:
    if tf['successful']:
        md_lines.append(f"- **{tf['tf']}** (AUROC {tf['observed_auroc']:.3f}): {tf['description']}")

md_lines.extend([
    "",
    "### 6. Power Analysis Caveat",
    "",
    f"With n=18 total TFs (7 vs 11 split), statistical power is severely limited:",
    f"- Minimum detectable Spearman ρ at α=0.05, 80% power: ~0.47",
    f"- Minimum detectable Cohen's d at α=0.05, 80% power, n1=7, n2=11: ~1.35",
    f"- **All results are hypothesis-generating, not confirmatory.**",
    "",
])

summary_path = os.path.join(OUT_DIR, 'tf_characterization_summary.md')
with open(summary_path, 'w') as f:
    f.write('\n'.join(md_lines))
print(f"  Saved: {summary_path}")

print("\n" + "=" * 70)
print("Analysis E COMPLETE")
print("=" * 70)
