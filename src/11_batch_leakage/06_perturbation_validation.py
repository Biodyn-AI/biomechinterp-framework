"""
Phase 2 - Objective 3: Perturbation-backed validation of blacklisted edges.

Compare edge scores from blacklisted (artifact-sensitive) vs clean edges
against perturbation ground truth from Dixit (CRISPR perturbation of TFs).

Approach:
1. Load Dixit perturbation data (CRISPR KO of TFs in K562 cells)
2. For each perturbed TF, compute differential expression of targets
   (perturbed vs control). This gives perturbation-validated edges.
3. Load our edge scores + blacklist from lung/immune datasets
4. Compare: do blacklisted edges have weaker perturbation support?
   Metrics: correlation, enrichment of perturbation-validated edges
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr, mannwhitneyu, spearmanr
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, json

warnings.filterwarnings('ignore')

OUTDIR = "/Users/ihorkendiukhov/biodyn-work/subproject_33_batch_donor_leakage_audit/reports/figures"
DATADIR = "/Users/ihorkendiukhov/biodyn-work/subproject_33_batch_donor_leakage_audit/implementation"

# ============================================================
# STEP 1: Build perturbation ground truth from Dixit data
# ============================================================
print("=" * 70)
print("PHASE 2: PERTURBATION-BACKED VALIDATION")
print("=" * 70)

print("\n--- Loading Dixit perturbation data ---")
adata_dixit = ad.read_h5ad(
    "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/perturb/dixit/perturb_processed_symbols.h5ad"
)
print(f"Shape: {adata_dixit.shape}")
print(f"Conditions: {adata_dixit.obs['condition'].value_counts().to_dict()}")

# Normalize
sc.pp.normalize_total(adata_dixit, target_sum=1e4)
sc.pp.log1p(adata_dixit)

# Get perturbed TFs (exclude control and intergenic)
perturbed_tfs = [c for c in adata_dixit.obs['condition'].unique()
                 if c != 'control' and 'INTERGENIC' not in c.upper()]
print(f"Perturbed TFs: {perturbed_tfs}")

# Our target gene list
TARGET_POOL = [
    'CD3D', 'CD3E', 'CD4', 'CD8A', 'CD8B', 'CD19', 'MS4A1', 'CD79A', 'CD79B',
    'CD14', 'CD68', 'ITGAM', 'ITGAX', 'FCGR3A', 'CSF1R',
    'IL2', 'IL2RA', 'IL7R', 'IL6', 'IL10', 'IL17A', 'IL21', 'IL4', 'IL13',
    'IFNG', 'TNF', 'TNFRSF9', 'TNFRSF18',
    'GZMB', 'GZMA', 'GZMK', 'PRF1', 'GNLY', 'NKG7',
    'CCL5', 'CCR7', 'CXCR4', 'CXCL13', 'CCL2', 'CXCL10',
    'HLA-DRA', 'HLA-DRB1', 'HLA-A', 'HLA-B', 'HLA-C', 'B2M',
    'LYZ', 'S100A8', 'S100A9', 'S100A12',
    'SELL', 'ITGA4', 'ITGB1', 'ICAM1', 'VCAM1',
    'MKI67', 'TOP2A', 'CDK1', 'PCNA',
    'BCL2', 'BCL2L1', 'MCL1', 'BAX', 'BAK1',
    'CTLA4', 'PDCD1', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX',
    'FN1', 'COL1A1', 'COL3A1', 'VIM', 'ACTA2', 'CDH1', 'EPCAM',
    'SFTPC', 'SFTPB', 'SFTPA1', 'SCGB1A1', 'MUC5AC', 'FOXJ1',
    'PECAM1', 'CDH5', 'VWF', 'KDR',
    'NOTCH1', 'DLL4', 'JAG1', 'WNT5A', 'CTNNB1',
    'TGFB1', 'TGFBR1', 'SMAD3', 'SMAD7',
    'SOD2', 'HMOX1', 'NQO1', 'GSTP1',
    'GAPDH', 'ACTB', 'RPL13A', 'RPS18',
    'PTPRC', 'THY1', 'ENG', 'NT5E',
]

# Compute perturbation effect sizes for each TF -> target
# Effect = mean(perturbed) - mean(control) for each target gene
print("\n--- Computing perturbation effect sizes ---")
control_mask = adata_dixit.obs['condition'] == 'control'
X_dixit = adata_dixit.X
if sparse.issparse(X_dixit):
    X_dixit = X_dixit.toarray()

var_names_dixit = list(adata_dixit.var_names)

# Find which target genes exist in Dixit data
targets_in_dixit = [t for t in TARGET_POOL if t in var_names_dixit]
target_indices_dixit = {t: var_names_dixit.index(t) for t in targets_in_dixit}
print(f"Target genes found in Dixit data: {len(targets_in_dixit)} / {len(TARGET_POOL)}")

# Also expand to all genes as targets (broader validation)
# Use top variable genes as additional targets
gene_var = np.var(X_dixit, axis=0)
top_var_genes = np.argsort(gene_var)[-500:]
all_target_genes = set(targets_in_dixit)
for gi in top_var_genes:
    all_target_genes.add(var_names_dixit[gi])
all_target_indices = {g: var_names_dixit.index(g) for g in all_target_genes if g in var_names_dixit}

# Control expression
control_expr = X_dixit[control_mask.values]
control_means = np.mean(control_expr, axis=0)

perturbation_effects = {}
for tf in perturbed_tfs:
    pert_mask = adata_dixit.obs['condition'] == tf
    pert_expr = X_dixit[pert_mask.values]
    if pert_expr.shape[0] < 10:
        continue
    pert_means = np.mean(pert_expr, axis=0)

    effects = {}
    for tgt, tgi in all_target_indices.items():
        if tgt == tf:
            continue
        delta = pert_means[tgi] - control_means[tgi]
        # Also compute significance via Mann-Whitney U
        u_stat, p_val = mannwhitneyu(
            pert_expr[:, tgi], control_expr[:, tgi], alternative='two-sided'
        )
        effects[tgt] = {
            'delta': float(delta),
            'abs_delta': float(abs(delta)),
            'p_value': float(p_val),
            'significant': p_val < 0.05,
        }
    perturbation_effects[tf] = effects
    n_sig = sum(1 for e in effects.values() if e['significant'])
    print(f"  {tf}: {len(effects)} targets tested, {n_sig} significant")

print(f"\nTotal TFs with perturbation data: {len(perturbation_effects)}")

# ============================================================
# STEP 2: Match perturbation effects to our edge scores
# ============================================================
print("\n--- Matching perturbation effects to edge scores ---")

# Load sensitivity data from Phase 1
for ds_name in ['Lung', 'Immune']:
    print(f"\n{'='*50}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*50}")

    sens_path = os.path.join(DATADIR, f'{ds_name}_sensitivity.csv')
    if not os.path.exists(sens_path):
        print(f"  Sensitivity file not found: {sens_path}")
        continue

    sens = pd.read_csv(sens_path)
    print(f"  Loaded {len(sens)} edges with ASI scores")

    # Create edge key
    sens['edge_key'] = sens['tf'] + '->' + sens['target']

    # Match to perturbation data
    matched = []
    for _, row in sens.iterrows():
        tf = row['tf']
        tgt = row['target']
        if tf in perturbation_effects and tgt in perturbation_effects[tf]:
            pe = perturbation_effects[tf][tgt]
            matched.append({
                'tf': tf,
                'target': tgt,
                'edge_score_full': row['score_full'],
                'edge_score_balanced': row['score_balanced'],
                'ASI': row['ASI'],
                'blacklisted': row['ASI'] > 0.5,
                'pert_delta': pe['delta'],
                'pert_abs_delta': pe['abs_delta'],
                'pert_significant': pe['significant'],
                'pert_p_value': pe['p_value'],
            })

    if len(matched) == 0:
        print(f"  No matched edges found")
        continue

    matched_df = pd.DataFrame(matched)
    print(f"  Matched edges: {len(matched_df)}")
    print(f"  Blacklisted: {matched_df['blacklisted'].sum()}")
    print(f"  Clean: {(~matched_df['blacklisted']).sum()}")
    print(f"  Perturbation-significant: {matched_df['pert_significant'].sum()}")

    # ---- Comparison: blacklisted vs clean edges ----
    bl = matched_df[matched_df['blacklisted']]
    clean = matched_df[~matched_df['blacklisted']]

    if len(bl) > 0 and len(clean) > 0:
        print(f"\n  --- Blacklisted vs Clean comparison ---")

        # 1. Mean perturbation effect size
        bl_mean_effect = bl['pert_abs_delta'].mean()
        clean_mean_effect = clean['pert_abs_delta'].mean()
        print(f"  Mean |pert effect|: blacklisted={bl_mean_effect:.4f}, clean={clean_mean_effect:.4f}")

        # 2. Fraction perturbation-significant
        bl_sig_frac = bl['pert_significant'].mean()
        clean_sig_frac = clean['pert_significant'].mean()
        print(f"  Perturbation-significant: blacklisted={bl_sig_frac:.3f}, clean={clean_sig_frac:.3f}")

        # 3. Mann-Whitney U test
        u_stat, p_mw = mannwhitneyu(
            bl['pert_abs_delta'].values, clean['pert_abs_delta'].values,
            alternative='two-sided'
        )
        print(f"  Mann-Whitney U (|pert effect|): U={u_stat:.0f}, p={p_mw:.4f}")

        # 4. Correlation between ASI and perturbation effect
        r_asi_pert, p_r = pearsonr(matched_df['ASI'].values, matched_df['pert_abs_delta'].values)
        print(f"  Correlation ASI vs |pert effect|: r={r_asi_pert:.4f}, p={p_r:.4f}")

        # 5. Correlation between edge score and perturbation direction
        r_score_pert, p_sp = spearmanr(
            matched_df['edge_score_full'].values, matched_df['pert_delta'].values
        )
        print(f"  Spearman edge_score vs pert_delta: rho={r_score_pert:.4f}, p={p_sp:.4f}")

        # Save per-dataset results
        perturb_results = {
            'n_matched': len(matched_df),
            'n_blacklisted': int(matched_df['blacklisted'].sum()),
            'n_clean': int((~matched_df['blacklisted']).sum()),
            'bl_mean_abs_pert_effect': float(bl_mean_effect),
            'clean_mean_abs_pert_effect': float(clean_mean_effect),
            'bl_sig_fraction': float(bl_sig_frac),
            'clean_sig_fraction': float(clean_sig_frac),
            'mw_u_stat': float(u_stat),
            'mw_p_value': float(p_mw),
            'asi_pert_correlation': float(r_asi_pert),
            'asi_pert_p_value': float(p_r),
            'score_pert_spearman': float(r_score_pert),
            'score_pert_spearman_p': float(p_sp),
        }

        # Save matched edges
        matched_df.to_csv(os.path.join(DATADIR, f'phase2_perturbation_matched_{ds_name}.csv'),
                          index=False)

    else:
        print(f"  Not enough blacklisted/clean edges for comparison")
        perturb_results = {'n_matched': len(matched_df), 'insufficient_data': True}

    # Save results
    with open(os.path.join(DATADIR, f'phase2_perturbation_results_{ds_name}.json'), 'w') as f:
        json.dump(perturb_results, f, indent=2)

# ============================================================
# STEP 3: Also try Adamson data (broader perturbation coverage)
# ============================================================
print("\n" + "=" * 70)
print("ADAMSON PERTURBATION DATA")
print("=" * 70)

adata_adam = ad.read_h5ad(
    "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/perturb/adamson/perturb_processed.h5ad"
)
print(f"Shape: {adata_adam.shape}")

# Map Ensembl to gene names
if 'gene_name' in adata_adam.var.columns:
    gene_names_adam = adata_adam.var['gene_name'].values.astype(str)
    seen = {}
    unique_names = []
    for name in gene_names_adam:
        if name in seen:
            seen[name] += 1
            unique_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            unique_names.append(name)
    adata_adam.var_names = unique_names
    adata_adam.var_names_make_unique()

sc.pp.normalize_total(adata_adam, target_sum=1e4)
sc.pp.log1p(adata_adam)

# Get conditions (format: GENE+ctrl)
conditions_adam = [c for c in adata_adam.obs['condition'].unique() if c != 'ctrl']
perturbed_genes_adam = [c.split('+')[0] for c in conditions_adam]
print(f"Perturbed genes: {perturbed_genes_adam[:20]}...")

var_names_adam = list(adata_adam.var_names)
X_adam = adata_adam.X
if sparse.issparse(X_adam):
    X_adam = X_adam.toarray()

control_mask_adam = adata_adam.obs['condition'] == 'ctrl'
control_expr_adam = X_adam[control_mask_adam.values]
control_means_adam = np.mean(control_expr_adam, axis=0)

# Compute perturbation effects for Adamson
perturbation_effects_adam = {}
for condition in conditions_adam:
    gene = condition.split('+')[0]
    pert_mask = adata_adam.obs['condition'] == condition
    pert_expr = X_adam[pert_mask.values]
    if pert_expr.shape[0] < 10:
        continue
    pert_means = np.mean(pert_expr, axis=0)

    # For ALL target genes in our list that exist in Adamson
    effects = {}
    for tgt in TARGET_POOL:
        if tgt not in var_names_adam or tgt == gene:
            continue
        tgi = var_names_adam.index(tgt)
        delta = pert_means[tgi] - control_means_adam[tgi]
        u_stat, p_val = mannwhitneyu(
            pert_expr[:, tgi], control_expr_adam[:, tgi], alternative='two-sided'
        )
        effects[tgt] = {
            'delta': float(delta),
            'abs_delta': float(abs(delta)),
            'p_value': float(p_val),
            'significant': p_val < 0.05,
        }
    if effects:
        perturbation_effects_adam[gene] = effects

print(f"Adamson: {len(perturbation_effects_adam)} perturbed genes with target effects")

# Match Adamson to our edge scores
for ds_name in ['Lung', 'Immune']:
    sens_path = os.path.join(DATADIR, f'{ds_name}_sensitivity.csv')
    if not os.path.exists(sens_path):
        continue

    sens = pd.read_csv(sens_path)
    matched = []
    for _, row in sens.iterrows():
        tf = row['tf']
        tgt = row['target']
        if tf in perturbation_effects_adam and tgt in perturbation_effects_adam[tf]:
            pe = perturbation_effects_adam[tf][tgt]
            matched.append({
                'tf': tf, 'target': tgt,
                'edge_score_full': row['score_full'],
                'ASI': row['ASI'],
                'blacklisted': row['ASI'] > 0.5,
                'pert_delta': pe['delta'],
                'pert_abs_delta': pe['abs_delta'],
                'pert_significant': pe['significant'],
                'source': 'Adamson',
            })

    print(f"\n  {ds_name} + Adamson: {len(matched)} matched edges")
    if len(matched) > 0:
        mdf = pd.DataFrame(matched)
        bl = mdf[mdf['blacklisted']]
        clean = mdf[~mdf['blacklisted']]
        if len(bl) > 0 and len(clean) > 0:
            print(f"    Blacklisted mean |pert|: {bl['pert_abs_delta'].mean():.4f}")
            print(f"    Clean mean |pert|:       {clean['pert_abs_delta'].mean():.4f}")
            print(f"    Blacklisted sig frac:    {bl['pert_significant'].mean():.3f}")
            print(f"    Clean sig frac:          {clean['pert_significant'].mean():.3f}")


# ============================================================
# FIGURES
# ============================================================
print("\n--- Generating perturbation validation figures ---")

# Combine Dixit + Adamson matched edges for the best dataset
for ds_name in ['Lung', 'Immune']:
    sens_path = os.path.join(DATADIR, f'{ds_name}_sensitivity.csv')
    if not os.path.exists(sens_path):
        continue
    sens = pd.read_csv(sens_path)

    all_matched = []
    # Dixit matches
    for _, row in sens.iterrows():
        tf, tgt = row['tf'], row['target']
        if tf in perturbation_effects and tgt in perturbation_effects[tf]:
            pe = perturbation_effects[tf][tgt]
            all_matched.append({
                'tf': tf, 'target': tgt,
                'edge_score': row['score_full'], 'ASI': row['ASI'],
                'blacklisted': row['ASI'] > 0.5,
                'pert_abs_delta': pe['abs_delta'],
                'pert_significant': pe['significant'],
                'source': 'Dixit',
            })
    # Adamson matches
    for _, row in sens.iterrows():
        tf, tgt = row['tf'], row['target']
        if tf in perturbation_effects_adam and tgt in perturbation_effects_adam[tf]:
            pe = perturbation_effects_adam[tf][tgt]
            key = f"{tf}->{tgt}"
            # Avoid duplicates with Dixit
            if not any(m['tf'] == tf and m['target'] == tgt for m in all_matched):
                all_matched.append({
                    'tf': tf, 'target': tgt,
                    'edge_score': row['score_full'], 'ASI': row['ASI'],
                    'blacklisted': row['ASI'] > 0.5,
                    'pert_abs_delta': pe['abs_delta'],
                    'pert_significant': pe['significant'],
                    'source': 'Adamson',
                })

    if len(all_matched) < 10:
        continue

    mdf = pd.DataFrame(all_matched)
    bl = mdf[mdf['blacklisted']]
    clean = mdf[~mdf['blacklisted']]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Distribution of perturbation effect sizes
    if len(bl) > 0 and len(clean) > 0:
        axes[0].hist(clean['pert_abs_delta'].values, bins=30, alpha=0.6,
                     color='steelblue', label=f'Clean (n={len(clean)})', density=True)
        axes[0].hist(bl['pert_abs_delta'].values, bins=30, alpha=0.6,
                     color='red', label=f'Blacklisted (n={len(bl)})', density=True)
        u, p = mannwhitneyu(bl['pert_abs_delta'], clean['pert_abs_delta'], alternative='two-sided')
        axes[0].set_xlabel('|Perturbation Effect|')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'{ds_name}: Perturbation Effect Distribution\nMann-Whitney p={p:.4f}')
        axes[0].legend()

    # Panel B: ASI vs perturbation effect scatter
    axes[1].scatter(mdf['ASI'].values, mdf['pert_abs_delta'].values,
                   alpha=0.4, s=15, c=mdf['blacklisted'].map({True: 'red', False: 'steelblue'}))
    r, p = pearsonr(mdf['ASI'].values, mdf['pert_abs_delta'].values)
    axes[1].set_xlabel('Artifact Sensitivity Index')
    axes[1].set_ylabel('|Perturbation Effect|')
    axes[1].set_title(f'{ds_name}: ASI vs Perturbation Support\nr={r:.4f}, p={p:.4f}')
    axes[1].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Blacklist threshold')
    axes[1].legend()

    # Panel C: Fraction perturbation-significant
    if len(bl) > 0 and len(clean) > 0:
        fracs = [clean['pert_significant'].mean(), bl['pert_significant'].mean()]
        bars = axes[2].bar(['Clean', 'Blacklisted'], fracs,
                          color=['steelblue', 'red'], alpha=0.8, width=0.5)
        axes[2].set_ylabel('Fraction Perturbation-Significant')
        axes[2].set_title(f'{ds_name}: Perturbation Validation Rate')
        for i, v in enumerate(fracs):
            axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=11)

    plt.suptitle(f'{ds_name}: Blacklisted vs Clean Edges — Perturbation Ground Truth',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f'phase2_perturbation_validation_{ds_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved phase2_perturbation_validation_{ds_name}.png")

    # Save combined matched data
    mdf.to_csv(os.path.join(DATADIR, f'phase2_perturbation_combined_{ds_name}.csv'), index=False)

print("\nPhase 2 perturbation validation complete.")
