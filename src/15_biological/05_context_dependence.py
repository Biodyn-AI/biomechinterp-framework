#!/usr/bin/env python3
"""
Analysis D: Context-Dependence Characterization
=================================================
Why does attention outperform correlation in RPE1 but not K562?

Three parts:
1. Dataset-level characterization across all 5 perturbation datasets
2. Per-perturbation analysis within RPE1 and K562 — what predicts attention advantage?
3. Methodological confound test — HVG protocol matching
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

OUT_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/biological_impact/context_dependence"
EXPERIMENTS = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments"
TRRUST_PATH = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv"

os.makedirs(OUT_DIR, exist_ok=True)

def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, pd.Timestamp): return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Dataset-level characterization
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: Dataset-Level Characterization")
print("=" * 70)

# Load TRRUST for TF classification
trrust = pd.read_csv(TRRUST_PATH, sep='\t', header=None, names=['tf', 'target', 'type', 'pmid'])
trrust_tfs = set(trrust['tf'].unique())
print(f"TRRUST: {len(trrust_tfs)} TFs, {len(trrust)} edges")

# Define dataset configurations
datasets = {
    'K562_CRISPRi': {
        'summary_json': f'{EXPERIMENTS}/round9_unified_extraction/extraction_config.json',
        'per_gene_csv': None,  # Need to compute from edge matrices + de_labels
        'cell_type': 'K562 (chronic myelogenous leukemia)',
        'modality': 'CRISPRi',
        'organism': 'human',
        'n_evaluable': 280,
        'n_hvg': 2000,
        'hvg_protocol': 'top-2000 HVG (no perturbation gene forcing)',
        'attn_auroc_L15': 0.704,  # from paper
        'corr_auroc': 0.703,
        'delta': 0.001,
        'p_value': 0.73,
        'cohens_d': 0.01
    },
    'RPE1_CRISPRi': {
        'per_gene_csv': f'{EXPERIMENTS}/round13_non_k562_replication/ReplogleRPE1/per_gene_L15_ReplogleRPE1.csv',
        'summary_json': f'{EXPERIMENTS}/round13_non_k562_replication/ReplogleRPE1/ReplogleRPE1_summary.json',
        'cell_type': 'RPE1 (retinal pigment epithelium)',
        'modality': 'CRISPRi',
        'organism': 'human',
        'n_evaluable': 1251,
        'n_hvg': 3309,
        'hvg_protocol': 'top-2000 HVG + perturbation genes forced in (3,309 total)',
        'attn_auroc_L15': 0.748,
        'corr_auroc': 0.658,
        'delta': 0.090,
        'p_value': 1e-10,
        'cohens_d': 0.47
    },
    'Adamson_CRISPRa': {
        'per_gene_csv': f'{EXPERIMENTS}/round9_adamson_cross_context/per_gene_results_L13_adamson.csv',
        'cell_type': 'K562 (chronic myelogenous leukemia)',
        'modality': 'CRISPRa',
        'organism': 'human',
        'n_evaluable': 77,
        'n_hvg': 2000,
        'hvg_protocol': 'top-2000 HVG',
        'attn_auroc_L15': 0.580,  # L15 from paper; L13=0.547
        'corr_auroc': 0.653,
        'delta': -0.073,
        'p_value': 1e-5,
        'cohens_d': -0.56
    },
    'Shifrut_TCells': {
        'per_gene_csv': f'{EXPERIMENTS}/round10_shifrut_cross_context/per_gene_L15_shifrut.csv',
        'cell_type': 'Primary T cells',
        'modality': 'CRISPRi (CRISPR screen)',
        'organism': 'human',
        'n_evaluable': 7,
        'n_hvg': 2000,
        'hvg_protocol': 'top-2000 HVG',
        'attn_auroc_L15': 0.679,
        'corr_auroc': 0.651,
        'delta': 0.029,
        'p_value': 0.706,
        'cohens_d': 0.16
    },
    'Tian_Neurons': {
        'per_gene_csv': f'{EXPERIMENTS}/round13_non_k562_replication/TianKampmann2021/per_gene_L15_TianKampmann2021.csv',
        'summary_json': f'{EXPERIMENTS}/round13_non_k562_replication/TianKampmann2021/TianKampmann2021_summary.json',
        'cell_type': 'iPSC-derived neurons',
        'modality': 'CRISPRi',
        'organism': 'human',
        'n_evaluable': 7,
        'n_hvg': 2000,
        'hvg_protocol': 'top-2000 HVG',
        'attn_auroc_L15': 0.533,
        'corr_auroc': 0.475,
        'delta': 0.058,
        'p_value': 0.078,
        'cohens_d': 0.80
    }
}

# Load per-gene data where available and compute dataset-level features
print("\nLoading per-gene results and computing dataset features...")

dataset_chars = {}
for ds_name, ds_info in datasets.items():
    print(f"\n  {ds_name}:")
    chars = {
        'cell_type': ds_info['cell_type'],
        'modality': ds_info['modality'],
        'n_evaluable': ds_info['n_evaluable'],
        'n_hvg': ds_info['n_hvg'],
        'hvg_protocol': ds_info['hvg_protocol'],
        'attn_auroc_L15': ds_info['attn_auroc_L15'],
        'corr_auroc': ds_info['corr_auroc'],
        'delta': ds_info['delta'],
        'p_value': ds_info['p_value'],
        'cohens_d': ds_info['cohens_d'],
    }

    # Load per-gene CSV if exists
    if ds_info.get('per_gene_csv') and os.path.exists(ds_info['per_gene_csv']):
        pg = pd.read_csv(ds_info['per_gene_csv'])
        print(f"    Loaded per-gene CSV: {len(pg)} rows, columns: {list(pg.columns)}")

        # Per-perturbation features
        if 'n_de' in pg.columns:
            chars['mean_n_de'] = float(pg['n_de'].mean())
            chars['median_n_de'] = float(pg['n_de'].median())
            chars['std_n_de'] = float(pg['n_de'].std())
        if 'auroc_attn' in pg.columns and 'auroc_corr' in pg.columns:
            pg['delta'] = pg['auroc_attn'] - pg['auroc_corr']
            chars['mean_attn_auroc'] = float(pg['auroc_attn'].mean())
            chars['mean_corr_auroc'] = float(pg['auroc_corr'].mean())
            chars['frac_attn_wins'] = float((pg['delta'] > 0).mean())
            chars['mean_delta'] = float(pg['delta'].mean())
            chars['std_delta'] = float(pg['delta'].std())
            chars['median_delta'] = float(pg['delta'].median())
    else:
        print(f"    No per-gene CSV available")

    # Load summary JSON if available for additional metadata
    if ds_info.get('summary_json') and os.path.exists(ds_info['summary_json']):
        with open(ds_info['summary_json']) as f:
            summary = json.load(f)
        if 'n_control_cells' in summary:
            chars['n_control_cells'] = summary['n_control_cells']
        if 'total_cells' in summary:
            chars['total_cells'] = summary['total_cells']

    # Check if perturbation genes are TFs
    if ds_info.get('per_gene_csv') and os.path.exists(ds_info['per_gene_csv']):
        pg = pd.read_csv(ds_info['per_gene_csv'])
        gene_col = 'gene' if 'gene' in pg.columns else pg.columns[0]
        pert_genes = set(pg[gene_col].values)
        n_tf = len(pert_genes & trrust_tfs)
        chars['n_perturbation_genes_that_are_tfs'] = n_tf
        chars['frac_tfs'] = n_tf / len(pert_genes) if len(pert_genes) > 0 else 0
        print(f"    {n_tf}/{len(pert_genes)} perturbation genes are TRRUST TFs ({chars['frac_tfs']*100:.1f}%)")

    dataset_chars[ds_name] = chars

    # Print summary
    print(f"    Cells: {ds_info['cell_type']}, Modality: {ds_info['modality']}")
    print(f"    n_evaluable: {ds_info['n_evaluable']}, HVGs: {ds_info['n_hvg']}")
    print(f"    Attn AUROC: {ds_info['attn_auroc_L15']:.3f}, Corr AUROC: {ds_info['corr_auroc']:.3f}, Δ: {ds_info['delta']:+.3f}")

# Save dataset characteristics
with open(os.path.join(OUT_DIR, 'dataset_characteristics.json'), 'w') as f:
    json.dump(dataset_chars, f, indent=2, default=_json_default)
print(f"\nSaved: {os.path.join(OUT_DIR, 'dataset_characteristics.json')}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Per-Perturbation Analysis within RPE1 and K562
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: Per-Perturbation Analysis")
print("=" * 70)

perturbation_results = {}

# ── 2a. RPE1 per-perturbation analysis ──────────────────────────────────────
print("\n[2a] RPE1 per-perturbation analysis...")
rpe1_csv = f'{EXPERIMENTS}/round13_non_k562_replication/ReplogleRPE1/per_gene_L15_ReplogleRPE1.csv'
rpe1_pg = pd.read_csv(rpe1_csv)
rpe1_pg['delta'] = rpe1_pg['auroc_attn'] - rpe1_pg['auroc_corr']
print(f"  {len(rpe1_pg)} perturbations")

# Load RPE1 h5ad for expression features
print("  Loading RPE1 expression data for perturbation-level features...")
import h5py
rpe1_h5ad = f'{EXPERIMENTS}/round13_non_k562_replication/data/ReplogleWeissman2022_rpe1.h5ad'

rpe1_gene_features = {}
if os.path.exists(rpe1_h5ad):
    with h5py.File(rpe1_h5ad, 'r') as f:
        # Get gene names - RPE1 uses 'gene_name' as a Dataset
        var_keys = list(f['var'].keys())
        gnames_raw = f['var']['gene_name'][:]
        gnames = [g.decode() if isinstance(g, bytes) else str(g) for g in gnames_raw]
        gene_to_idx = {g: i for i, g in enumerate(gnames)}
        print(f"    {len(gnames)} genes (gene_name column)")

        # We need expression stats per perturbation gene
        gene_col = 'gene' if 'gene' in rpe1_pg.columns else rpe1_pg.columns[0]
        pert_genes = rpe1_pg[gene_col].unique()

        # Use precomputed mean and std from var
        mean_expr = f['var']['mean'][:] if 'mean' in var_keys else None
        std_expr = f['var']['std'][:] if 'std' in var_keys else None
        cv_expr = f['var']['cv'][:] if 'cv' in var_keys else None
        fano_expr = f['var']['fano'][:] if 'fano' in var_keys else None

        for gene in pert_genes:
            if gene in gene_to_idx:
                idx = gene_to_idx[gene]
                features = {'found': True}
                if mean_expr is not None:
                    features['mean_expression'] = float(mean_expr[idx])
                if std_expr is not None:
                    features['std_expression'] = float(std_expr[idx])
                if cv_expr is not None:
                    features['cv'] = float(cv_expr[idx])
                if fano_expr is not None:
                    features['fano'] = float(fano_expr[idx])
                rpe1_gene_features[gene] = features

    print(f"    Got expression features for {len(rpe1_gene_features)}/{len(pert_genes)} perturbation genes")
else:
    print(f"    RPE1 h5ad not found at {rpe1_h5ad}")

# Merge features into per-gene dataframe
gene_col = 'gene' if 'gene' in rpe1_pg.columns else rpe1_pg.columns[0]
rpe1_pg['mean_expression'] = rpe1_pg[gene_col].map(
    lambda g: rpe1_gene_features.get(g, {}).get('mean_expression', np.nan))
rpe1_pg['is_tf'] = rpe1_pg[gene_col].isin(trrust_tfs)
rpe1_pg['trrust_degree'] = rpe1_pg[gene_col].map(
    lambda g: len(trrust[trrust['tf'] == g]) if g in trrust_tfs else 0)

# Correlations of delta with perturbation features
print("\n  Correlations of attention advantage (Δ = attn - corr) with perturbation features:")

rpe1_corrs = {}
for feature, col in [('n_DE_targets', 'n_de'), ('mean_expression', 'mean_expression'),
                      ('TRRUST_degree', 'trrust_degree')]:
    valid = rpe1_pg.dropna(subset=[col, 'delta'])
    if len(valid) > 10:
        rho, p = stats.spearmanr(valid[col], valid['delta'])
        print(f"    Δ vs {feature}: ρ={rho:.4f}, p={p:.4f} (n={len(valid)})")
        rpe1_corrs[feature] = {'rho': float(rho), 'p': float(p), 'n': len(valid)}

# Is delta larger for TFs than non-TFs?
tf_delta = rpe1_pg[rpe1_pg['is_tf']]['delta']
nontf_delta = rpe1_pg[~rpe1_pg['is_tf']]['delta']
if len(tf_delta) >= 2 and len(nontf_delta) >= 2:
    u, p = stats.mannwhitneyu(tf_delta, nontf_delta, alternative='two-sided')
    print(f"\n    TF vs non-TF delta: TF median={tf_delta.median():.4f} (n={len(tf_delta)}), "
          f"non-TF median={nontf_delta.median():.4f} (n={len(nontf_delta)}), U={u:.0f}, p={p:.4f}")
    rpe1_corrs['tf_vs_nontf'] = {
        'tf_median_delta': float(tf_delta.median()),
        'nontf_median_delta': float(nontf_delta.median()),
        'U': float(u), 'p': float(p),
        'n_tf': len(tf_delta), 'n_nontf': len(nontf_delta)
    }

# Quintile analysis: bin by n_de
if 'n_de' in rpe1_pg.columns:
    rpe1_pg['n_de_quintile'] = pd.qcut(rpe1_pg['n_de'], 5, labels=False, duplicates='drop')
    quintile_stats = rpe1_pg.groupby('n_de_quintile').agg({
        'delta': ['mean', 'median', 'std'],
        'n_de': 'mean',
        'auroc_attn': 'mean',
        'auroc_corr': 'mean'
    })
    print(f"\n    Quintile analysis by n_DE:")
    for q in range(quintile_stats.shape[0]):
        row = quintile_stats.iloc[q]
        print(f"      Q{q}: mean_n_de={row[('n_de','mean')]:.0f}, mean_Δ={row[('delta','mean')]:+.4f}, "
              f"attn={row[('auroc_attn','mean')]:.3f}, corr={row[('auroc_corr','mean')]:.3f}")

perturbation_results['RPE1'] = {
    'n_perturbations': len(rpe1_pg),
    'mean_delta': float(rpe1_pg['delta'].mean()),
    'frac_attn_wins': float((rpe1_pg['delta'] > 0).mean()),
    'correlations_with_delta': rpe1_corrs,
}

# ── 2b. K562 per-perturbation analysis ─────────────────────────────────────
print("\n\n[2b] K562 per-perturbation analysis...")
print("  Computing per-perturbation AUROC from edge matrices and DE labels...")

# Load K562 data
k562_base = f'{EXPERIMENTS}/round9_unified_extraction'
with open(f'{k562_base}/de_labels.pkl', 'rb') as f:
    k562_de = pickle.load(f)
with open(f'{k562_base}/hvg_gene_list.json') as f:
    hvg_genes = json.load(f)
with open(f'{k562_base}/layer_mean_edges/layer_14_edges.pkl', 'rb') as f:
    k562_attn_edges = pickle.load(f)

# Load correlation matrix
k562_corr = np.load(f'{k562_base}/correlation_matrix.npy')

# Build gene-to-index mapping
gene_to_idx_k562 = {g: i for i, g in enumerate(hvg_genes)}

# Load gene features
k562_gf = pd.read_csv(f'{k562_base}/gene_features.csv')
gene_feat_map = {row['gene']: row for _, row in k562_gf.iterrows()}

# Compute per-perturbation AUROC for K562
from sklearn.metrics import roc_auc_score

k562_per_pert = []
print(f"  Processing {len(k562_de)} perturbations...")
for pert_gene, de_dict in k562_de.items():
    # Get DE labels for this perturbation
    genes = list(de_dict.keys())
    labels = np.array([int(de_dict[g]) for g in genes])

    if labels.sum() < 3 or labels.sum() == len(labels):
        continue

    # Get attention scores for this perturbation gene
    attn_scores = []
    corr_scores = []
    for target_gene in genes:
        # Attention score
        key1 = (pert_gene, target_gene)
        key2 = (target_gene, pert_gene)
        attn_val = k562_attn_edges.get(key1, k562_attn_edges.get(key2, 0.0))
        attn_scores.append(attn_val)

        # Correlation score
        if pert_gene in gene_to_idx_k562 and target_gene in gene_to_idx_k562:
            i = gene_to_idx_k562[pert_gene]
            j = gene_to_idx_k562[target_gene]
            corr_val = abs(k562_corr[i, j])
        else:
            corr_val = 0.0
        corr_scores.append(corr_val)

    attn_scores = np.array(attn_scores)
    corr_scores = np.array(corr_scores)

    # Compute AUROCs
    try:
        auroc_attn = roc_auc_score(labels, attn_scores)
        auroc_corr = roc_auc_score(labels, corr_scores)
    except ValueError:
        continue

    # Get perturbation gene features
    gf = gene_feat_map.get(pert_gene, {})

    k562_per_pert.append({
        'gene': pert_gene,
        'auroc_attn': auroc_attn,
        'auroc_corr': auroc_corr,
        'delta': auroc_attn - auroc_corr,
        'n_de': int(labels.sum()),
        'n_total': len(labels),
        'mean_expression': float(gf.get('mean_expr', np.nan)),
        'variance': float(gf.get('variance', np.nan)),
        'dropout_rate': float(gf.get('dropout_rate', np.nan)),
        'is_tf': pert_gene in trrust_tfs,
        'trrust_degree': len(trrust[trrust['tf'] == pert_gene]) if pert_gene in trrust_tfs else 0
    })

k562_df = pd.DataFrame(k562_per_pert)
print(f"  {len(k562_df)} perturbations with valid AUROCs")
print(f"  Mean Δ: {k562_df['delta'].mean():+.4f}")
print(f"  Fraction attn > corr: {(k562_df['delta'] > 0).mean():.3f}")

# Correlations for K562
print("\n  Correlations of attention advantage (Δ) with perturbation features:")
k562_corrs = {}
for feature, col in [('n_DE_targets', 'n_de'), ('mean_expression', 'mean_expression'),
                      ('variance', 'variance'), ('dropout_rate', 'dropout_rate'),
                      ('TRRUST_degree', 'trrust_degree')]:
    valid = k562_df.dropna(subset=[col, 'delta'])
    if len(valid) > 10:
        rho, p = stats.spearmanr(valid[col], valid['delta'])
        print(f"    Δ vs {feature}: ρ={rho:.4f}, p={p:.4f} (n={len(valid)})")
        k562_corrs[feature] = {'rho': float(rho), 'p': float(p), 'n': len(valid)}

# TF vs non-TF
tf_delta_k = k562_df[k562_df['is_tf']]['delta']
nontf_delta_k = k562_df[~k562_df['is_tf']]['delta']
if len(tf_delta_k) >= 2 and len(nontf_delta_k) >= 2:
    u, p = stats.mannwhitneyu(tf_delta_k, nontf_delta_k, alternative='two-sided')
    print(f"\n    TF vs non-TF delta: TF median={tf_delta_k.median():.4f} (n={len(tf_delta_k)}), "
          f"non-TF median={nontf_delta_k.median():.4f} (n={len(nontf_delta_k)}), U={u:.0f}, p={p:.4f}")
    k562_corrs['tf_vs_nontf'] = {
        'tf_median_delta': float(tf_delta_k.median()),
        'nontf_median_delta': float(nontf_delta_k.median()),
        'U': float(u), 'p': float(p),
        'n_tf': len(tf_delta_k), 'n_nontf': len(nontf_delta_k)
    }

# Quintile analysis for K562
if 'n_de' in k562_df.columns:
    k562_df['n_de_quintile'] = pd.qcut(k562_df['n_de'], 5, labels=False, duplicates='drop')
    quintile_stats_k = k562_df.groupby('n_de_quintile').agg({
        'delta': ['mean', 'median', 'std'],
        'n_de': 'mean',
        'auroc_attn': 'mean',
        'auroc_corr': 'mean'
    })
    print(f"\n    Quintile analysis by n_DE:")
    for q in range(quintile_stats_k.shape[0]):
        row = quintile_stats_k.iloc[q]
        print(f"      Q{q}: mean_n_de={row[('n_de','mean')]:.0f}, mean_Δ={row[('delta','mean')]:+.4f}, "
              f"attn={row[('auroc_attn','mean')]:.3f}, corr={row[('auroc_corr','mean')]:.3f}")

perturbation_results['K562'] = {
    'n_perturbations': len(k562_df),
    'mean_delta': float(k562_df['delta'].mean()),
    'frac_attn_wins': float((k562_df['delta'] > 0).mean()),
    'correlations_with_delta': k562_corrs,
}

# Save per-perturbation data
k562_df.to_csv(os.path.join(OUT_DIR, 'k562_per_perturbation.csv'), index=False)
rpe1_pg.to_csv(os.path.join(OUT_DIR, 'rpe1_per_perturbation.csv'), index=False)
print(f"\n  Saved per-perturbation CSVs")

# ── 2c. Compare RPE1 and K562 — what differs systematically? ────────────────
print("\n\n[2c] Systematic comparison: RPE1 vs K562")

# Compare distributions of n_de
if 'n_de' in rpe1_pg.columns and 'n_de' in k562_df.columns:
    u, p = stats.mannwhitneyu(rpe1_pg['n_de'], k562_df['n_de'], alternative='two-sided')
    print(f"  n_DE: RPE1 median={rpe1_pg['n_de'].median():.0f}, K562 median={k562_df['n_de'].median():.0f}, "
          f"U={u:.0f}, p={p:.6f}")
    perturbation_results['cross_dataset'] = {
        'n_de_comparison': {
            'rpe1_median': float(rpe1_pg['n_de'].median()),
            'k562_median': float(k562_df['n_de'].median()),
            'U': float(u), 'p': float(p)
        }
    }

# Compare fraction of TFs
n_tf_rpe1 = rpe1_pg['is_tf'].sum()
n_tf_k562 = k562_df['is_tf'].sum()
frac_tf_rpe1 = n_tf_rpe1 / len(rpe1_pg)
frac_tf_k562 = n_tf_k562 / len(k562_df)
print(f"  TF fraction: RPE1={frac_tf_rpe1:.3f} ({n_tf_rpe1}/{len(rpe1_pg)}), "
      f"K562={frac_tf_k562:.3f} ({n_tf_k562}/{len(k562_df)})")

# Check if the attention advantage in RPE1 is driven by n_de
# Partial correlation: delta ~ attn_advantage controlling for n_de
print("\n  Testing if attention advantage is confounded by n_DE...")
# For RPE1: regress delta on n_de, check residual
from sklearn.linear_model import LinearRegression

for name, df in [('RPE1', rpe1_pg), ('K562', k562_df)]:
    if 'n_de' in df.columns:
        valid = df.dropna(subset=['n_de', 'delta'])
        X = valid['n_de'].values.reshape(-1, 1)
        y = valid['delta'].values
        reg = LinearRegression().fit(X, y)
        residuals = y - reg.predict(X)
        # Is the mean residual significantly different from 0?
        t, p = stats.ttest_1samp(residuals, 0)
        print(f"  {name}: mean residual Δ (controlling for n_DE) = {np.mean(residuals):+.4f}, "
              f"t={t:.3f}, p={p:.4f}")
        perturbation_results.setdefault('cross_dataset', {})
        perturbation_results['cross_dataset'][f'{name}_residual_delta_controlling_nde'] = {
            'mean_residual': float(np.mean(residuals)),
            't': float(t), 'p': float(p)
        }

# Save Part 2 results
with open(os.path.join(OUT_DIR, 'perturbation_level_analysis.json'), 'w') as f:
    json.dump(perturbation_results, f, indent=2, default=_json_default)
print(f"\nSaved: {os.path.join(OUT_DIR, 'perturbation_level_analysis.json')}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: HVG Protocol Confound Test
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: HVG Protocol Confound Test")
print("=" * 70)

hvg_results = {}

# The key question: RPE1 uses 3,309 HVGs (with perturbation genes forced in)
# while K562 uses 2,000 HVGs. Does this methodological difference explain the
# attention advantage in RPE1?

# Test 1: Compute K562 with forced HVG inclusion (matching RPE1 protocol)
print("\n[3a] K562 with forced perturbation gene inclusion...")
print("  Loading K562 bulk perturbation data to get perturbation gene list...")

# The perturbation genes in K562 are the 280 genes in de_labels
k562_pert_genes = set(k562_de.keys())
k562_hvg_set = set(hvg_genes)
n_already_in = len(k562_pert_genes & k562_hvg_set)
n_to_force = len(k562_pert_genes - k562_hvg_set)
print(f"  K562 perturbation genes: {len(k562_pert_genes)}")
print(f"  Already in HVG: {n_already_in}")
print(f"  Would need to force: {n_to_force}")
print(f"  Expanded HVG set: {len(k562_hvg_set) + n_to_force}")

# For the correlation-based confound test, we need to recompute correlations
# on the expanded gene set. This requires the raw K562 data.
# Try multiple possible locations for K562 raw data
k562_h5ad_candidates = [
    '/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/replogle/K562_gwps_normalized_bulk_01.h5ad',
    '/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/data/perturb/K562_gwps_normalized_bulk_01.h5ad',
]
k562_h5ad = None
for candidate in k562_h5ad_candidates:
    if os.path.exists(candidate):
        k562_h5ad = candidate
        break

if k562_h5ad is not None:
    print(f"\n  Loading K562 raw data from {k562_h5ad}...")
    from scipy.sparse import csr_matrix, issparse

    with h5py.File(k562_h5ad, 'r') as f:
        var_keys = list(f['var'].keys())
        obs_keys = list(f['obs'].keys())
        print(f"    var keys: {var_keys[:10]}")
        print(f"    obs keys: {obs_keys[:10]}")

        # Get gene names
        if '_index' in var_keys:
            gnames_raw = f['var']['_index'][:]
        else:
            gnames_raw = f['var'][var_keys[0]][:]
        gnames_k562 = [g.decode() if isinstance(g, bytes) else str(g) for g in gnames_raw]
        print(f"    {len(gnames_k562)} genes")

        # Get perturbation condition labels
        if 'gene' in f['obs']:
            obs_gene = f['obs']['gene']
            if isinstance(obs_gene, h5py.Group):
                cats = obs_gene['categories'][:]
                codes = obs_gene['codes'][:]
                conditions = [cats[c].decode() if isinstance(cats[c], bytes) else str(cats[c]) for c in codes]
            else:
                conditions = [g.decode() if isinstance(g, bytes) else str(g) for g in obs_gene[:]]
        elif 'perturbation' in f['obs']:
            obs_pert = f['obs']['perturbation']
            if isinstance(obs_pert, h5py.Group):
                cats = obs_pert['categories'][:]
                codes = obs_pert['codes'][:]
                conditions = [cats[c].decode() if isinstance(cats[c], bytes) else str(cats[c]) for c in codes]
            else:
                conditions = [g.decode() if isinstance(g, bytes) else str(g) for g in obs_pert[:]]
        else:
            conditions = None
            print(f"    WARNING: No condition column found")

        if conditions:
            unique_conditions = set(conditions)
            print(f"    {len(unique_conditions)} unique conditions")
            # Find control cells
            ctrl_names = [c for c in unique_conditions if 'control' in c.lower() or c == 'non-targeting' or c == 'ctrl']
            if not ctrl_names:
                ctrl_names = [c for c in unique_conditions if 'non' in c.lower()]
            print(f"    Control conditions: {ctrl_names[:5]}")

            if ctrl_names:
                ctrl_mask = np.array([c in ctrl_names for c in conditions])
                n_ctrl = ctrl_mask.sum()
                print(f"    {n_ctrl} control cells")

                # Get expression matrix for control cells
                X_data = f['X']
                if isinstance(X_data, h5py.Group):
                    # Sparse
                    data = X_data['data'][:]
                    indices = X_data['indices'][:]
                    indptr = X_data['indptr'][:]
                    X_full = csr_matrix((data, indices, indptr), shape=(len(conditions), len(gnames_k562)))
                    X_ctrl = X_full[ctrl_mask].toarray()
                else:
                    X_ctrl = X_data[ctrl_mask]

                # Subsample control cells to 2000
                if len(X_ctrl) > 2000:
                    np.random.seed(42)
                    idx = np.random.choice(len(X_ctrl), 2000, replace=False)
                    X_ctrl = X_ctrl[idx]
                print(f"    Control expression matrix: {X_ctrl.shape}")

                # Build expanded gene set: top-2000 HVG + forced perturbation genes
                gene_to_idx_full = {g: i for i, g in enumerate(gnames_k562)}

                # Get indices for original HVG + forced perturbation genes
                forced_genes = list(k562_pert_genes - k562_hvg_set)
                forced_in_data = [g for g in forced_genes if g in gene_to_idx_full]
                expanded_hvg = list(k562_hvg_set) + forced_in_data
                print(f"    Expanded HVG set: {len(expanded_hvg)} genes ({len(k562_hvg_set)} original + {len(forced_in_data)} forced)")

                # Extract columns for expanded set
                expanded_idx = [gene_to_idx_full[g] for g in expanded_hvg if g in gene_to_idx_full]
                expanded_names = [g for g in expanded_hvg if g in gene_to_idx_full]
                X_expanded = X_ctrl[:, expanded_idx]
                print(f"    Expanded expression matrix: {X_expanded.shape}")

                # Compute Spearman correlation on expanded set
                print("    Computing Spearman correlations on expanded gene set...")
                from scipy.stats import spearmanr
                corr_expanded, _ = spearmanr(X_expanded)
                corr_expanded = np.nan_to_num(corr_expanded, nan=0.0)
                print(f"    Correlation matrix shape: {corr_expanded.shape}")

                # Also recompute on original 2000 HVG for fair comparison
                orig_idx = [gene_to_idx_full[g] for g in hvg_genes if g in gene_to_idx_full]
                orig_names = [g for g in hvg_genes if g in gene_to_idx_full]
                X_orig = X_ctrl[:, orig_idx]
                corr_orig, _ = spearmanr(X_orig)
                corr_orig = np.nan_to_num(corr_orig, nan=0.0)

                # Compute AUROC for both
                expanded_gene_to_idx = {g: i for i, g in enumerate(expanded_names)}
                orig_gene_to_idx = {g: i for i, g in enumerate(orig_names)}

                aurocs_orig = []
                aurocs_expanded = []
                genes_compared = []

                for pert_gene, de_dict in k562_de.items():
                    # Original HVG AUROC
                    if pert_gene in orig_gene_to_idx:
                        p_idx = orig_gene_to_idx[pert_gene]
                        labels_o = []
                        scores_o = []
                        for target_gene, is_de in de_dict.items():
                            if target_gene in orig_gene_to_idx and target_gene != pert_gene:
                                t_idx = orig_gene_to_idx[target_gene]
                                labels_o.append(int(is_de))
                                scores_o.append(abs(corr_orig[p_idx, t_idx]))
                        if sum(labels_o) >= 3 and sum(labels_o) < len(labels_o):
                            try:
                                auc_o = roc_auc_score(labels_o, scores_o)
                            except:
                                continue

                            # Expanded HVG AUROC
                            if pert_gene in expanded_gene_to_idx:
                                p_idx_e = expanded_gene_to_idx[pert_gene]
                                labels_e = []
                                scores_e = []
                                for target_gene, is_de in de_dict.items():
                                    if target_gene in expanded_gene_to_idx and target_gene != pert_gene:
                                        t_idx_e = expanded_gene_to_idx[target_gene]
                                        labels_e.append(int(is_de))
                                        scores_e.append(abs(corr_expanded[p_idx_e, t_idx_e]))
                                if sum(labels_e) >= 3 and sum(labels_e) < len(labels_e):
                                    try:
                                        auc_e = roc_auc_score(labels_e, scores_e)
                                        aurocs_orig.append(auc_o)
                                        aurocs_expanded.append(auc_e)
                                        genes_compared.append(pert_gene)
                                    except:
                                        pass

                aurocs_orig = np.array(aurocs_orig)
                aurocs_expanded = np.array(aurocs_expanded)
                print(f"\n    Paired comparison: {len(aurocs_orig)} perturbations")
                print(f"    Original (2000 HVG) corr AUROC: {aurocs_orig.mean():.4f} ± {aurocs_orig.std():.4f}")
                print(f"    Expanded (forced HVG) corr AUROC: {aurocs_expanded.mean():.4f} ± {aurocs_expanded.std():.4f}")
                print(f"    Δ = {(aurocs_expanded - aurocs_orig).mean():+.4f}")

                # Paired test
                stat, p = stats.wilcoxon(aurocs_expanded - aurocs_orig)
                print(f"    Wilcoxon signed-rank: stat={stat:.0f}, p={p:.6f}")

                hvg_results['k562_forced_hvg'] = {
                    'n_perturbations': len(aurocs_orig),
                    'original_mean_auroc': float(aurocs_orig.mean()),
                    'expanded_mean_auroc': float(aurocs_expanded.mean()),
                    'delta': float((aurocs_expanded - aurocs_orig).mean()),
                    'wilcoxon_stat': float(stat),
                    'wilcoxon_p': float(p),
                    'original_n_hvg': len(k562_hvg_set),
                    'expanded_n_hvg': len(expanded_hvg),
                    'n_forced_genes': len(forced_in_data)
                }

    # Now the attention side: does attention also change with forced HVG?
    # The attention edges from round9 only cover the original 2000 HVG,
    # so we can't directly compare. But we can note this limitation.
    hvg_results['attention_note'] = (
        "Attention edges are only available for the original 2000 HVG universe. "
        "The forced-HVG test only applies to correlation-based edges. "
        "A full attention re-extraction with expanded HVG would require GPU computation."
    )
else:
    print(f"  K562 raw h5ad not found at any candidate location")
    print(f"  Checked: {k562_h5ad_candidates}")
    print(f"  HVG confound test cannot be performed without raw expression data.")
    print(f"  Note: Existing pre-extracted data (correlation_matrix.npy, de_labels.pkl) only covers 2000 HVGs.")
    hvg_results['k562_forced_hvg'] = {
        'status': 'skipped',
        'reason': 'K562 raw expression data not available locally. Only pre-extracted features exist.',
        'n_pert_genes_in_hvg': n_already_in,
        'n_pert_genes_not_in_hvg': n_to_force,
        'note': ('The round9 extraction already captured all edges within the 2000 HVG universe. '
                 'To test forced HVG inclusion, the raw K562 h5ad would need to be re-downloaded '
                 'from scperturb to recompute the correlation matrix on an expanded gene set.')
    }
    # However, we CAN report the key structural difference
    print(f"\n  KEY STRUCTURAL OBSERVATION:")
    print(f"  K562: {n_already_in}/{len(k562_pert_genes)} perturbation genes are in the 2000-gene HVG set "
          f"({n_already_in/len(k562_pert_genes)*100:.1f}%)")
    print(f"  RPE1: 1251/1251 perturbation genes are in the 3309-gene expanded set (100%)")
    print(f"  This means {n_to_force} K562 perturbation genes have NO correlation-based edge scores,")
    print(f"  while ALL RPE1 perturbation genes have correlation-based edge scores.")

# Save Part 3 results
with open(os.path.join(OUT_DIR, 'hvg_confound_test.json'), 'w') as f:
    json.dump(hvg_results, f, indent=2, default=_json_default)
print(f"\nSaved: {os.path.join(OUT_DIR, 'hvg_confound_test.json')}")

# ═══════════════════════════════════════════════════════════════════════════════
# Combined results and summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("COMBINED RESULTS")
print("=" * 70)

all_results = {
    'dataset_characteristics': dataset_chars,
    'perturbation_level_analysis': perturbation_results,
    'hvg_confound_test': hvg_results
}

with open(os.path.join(OUT_DIR, '..', 'context_dependence_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=_json_default)

# ─── Generate summary markdown ──────────────────────────────────────────────
md_lines = [
    "# Analysis D: Context-Dependence Characterization — Results",
    "",
    "## Overview",
    "",
    "Why does attention outperform correlation in RPE1 (d=0.47) but not K562?",
    "",
    "## 1. Dataset-Level Comparison",
    "",
    "| Dataset | Cell Type | Modality | n_eval | HVGs | Attn AUROC | Corr AUROC | Δ | Cohen's d |",
    "|---------|-----------|----------|--------|------|-----------|------------|------|-----------|",
]

for ds_name, chars in dataset_chars.items():
    md_lines.append(
        f"| {ds_name} | {chars['cell_type'].split('(')[0].strip()} | {chars['modality']} | "
        f"{chars['n_evaluable']} | {chars['n_hvg']} | {chars['attn_auroc_L15']:.3f} | "
        f"{chars['corr_auroc']:.3f} | {chars['delta']:+.3f} | {chars['cohens_d']:.2f} |"
    )

md_lines.extend([
    "",
    "### Key Observations",
    "",
    "1. **Modality matters**: Both CRISPRi datasets on K562 (K562_CRISPRi) and RPE1 show attention ≥ correlation,",
    "   but CRISPRa (Adamson) shows the opposite. CRISPRa activates genes rather than knocking them down,",
    "   fundamentally changing the perturbation signal structure.",
    "2. **Cell type matters**: RPE1 (non-cancerous epithelial) shows strong attention advantage; K562 (cancer) does not.",
    "3. **HVG protocol differs**: RPE1 uses 3,309 HVGs (perturbation genes forced in) vs K562's 2,000.",
    "",
    "## 2. Per-Perturbation Analysis",
    "",
])

for ds_name in ['RPE1', 'K562']:
    if ds_name in perturbation_results:
        pr = perturbation_results[ds_name]
        md_lines.extend([
            f"### {ds_name}",
            f"- {pr['n_perturbations']} perturbations",
            f"- Mean Δ: {pr['mean_delta']:+.4f}",
            f"- Fraction where attention wins: {pr['frac_attn_wins']:.3f}",
            "",
            "Correlations of attention advantage with perturbation features:",
            "",
        ])
        for feat, corr_info in pr['correlations_with_delta'].items():
            if 'rho' in corr_info:
                sig = "**" if corr_info['p'] < 0.05 else ""
                md_lines.append(f"- {feat}: ρ = {corr_info['rho']:.4f}, p = {corr_info['p']:.4f} {sig}")
            elif 'U' in corr_info:
                sig = "**" if corr_info['p'] < 0.05 else ""
                md_lines.append(f"- {feat}: TF median = {corr_info.get('tf_median_delta', 'N/A'):.4f}, "
                              f"non-TF median = {corr_info.get('nontf_median_delta', 'N/A'):.4f}, p = {corr_info['p']:.4f} {sig}")
        md_lines.append("")

md_lines.extend([
    "## 3. HVG Protocol Confound Test",
    "",
])

if 'k562_forced_hvg' in hvg_results and 'delta' in hvg_results.get('k562_forced_hvg', {}):
    h = hvg_results['k562_forced_hvg']
    md_lines.extend([
        f"**K562 with forced perturbation gene inclusion (matching RPE1 protocol):**",
        f"- Original HVG set: {h['original_n_hvg']} genes",
        f"- Expanded HVG set: {h['expanded_n_hvg']} genes (+{h['n_forced_genes']} forced)",
        f"- Original correlation AUROC: {h['original_mean_auroc']:.4f}",
        f"- Expanded correlation AUROC: {h['expanded_mean_auroc']:.4f}",
        f"- Δ = {h['delta']:+.4f}",
        f"- Wilcoxon p = {h['wilcoxon_p']:.6f}",
        "",
    ])
elif 'k562_forced_hvg' in hvg_results:
    h = hvg_results['k562_forced_hvg']
    md_lines.extend([
        f"**K562 forced HVG test: SKIPPED**",
        f"- Reason: {h.get('reason', 'data not available')}",
        f"- K562 perturbation genes in HVG: {h.get('n_pert_genes_in_hvg', 'N/A')}/{h.get('n_pert_genes_in_hvg', 0) + h.get('n_pert_genes_not_in_hvg', 0)}",
        f"- K562 perturbation genes NOT in HVG: {h.get('n_pert_genes_not_in_hvg', 'N/A')}",
        "",
        "**Key structural observation**: RPE1 uses 3,309 HVGs (perturbation genes forced in, 100% coverage)",
        f"while K562 uses 2,000 HVGs ({h.get('n_pert_genes_in_hvg', 'N/A')} of {h.get('n_pert_genes_in_hvg', 0) + h.get('n_pert_genes_not_in_hvg', 0)} perturbation genes covered).",
        "This asymmetry means K562 perturbation genes outside the HVG set have no correlation-based edge scores,",
        "while all RPE1 perturbation genes do. This could contribute to the observed context-dependence.",
        "",
    ])

md_lines.extend([
    "## 4. Interpretation",
    "",
    "### Biological Hypotheses",
    "",
    "1. **Cell-type regulatory architecture**: RPE1 cells may have more structured regulatory programs",
    "   that Geneformer's attention captures beyond simple co-expression. K562 is a cancer cell line",
    "   with dysregulated gene expression that may obscure regulatory structure.",
    "",
    "2. **Modality effect**: CRISPRi (knockdown) creates loss-of-function perturbations that propagate",
    "   through regulatory networks. CRISPRa (activation) creates gain-of-function that may produce",
    "   different downstream signatures not aligned with attention patterns.",
    "",
    "3. **Technical confound**: The HVG protocol difference (3,309 vs 2,000 genes) systematically",
    "   includes perturbation target genes in RPE1 but not K562, potentially biasing the comparison.",
    "",
])

summary_path = os.path.join(OUT_DIR, '..', 'context_dependence_summary.md')
with open(summary_path, 'w') as f:
    f.write('\n'.join(md_lines))
print(f"Saved: {summary_path}")

print("\n" + "=" * 70)
print("Analysis D COMPLETE")
print("=" * 70)
