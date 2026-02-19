"""
Step 4: Cross-species edge comparison and null models.

Analysis:
1. Match edges across species via ortholog mapping
2. Compute overlap statistics (top-k overlap, rank correlation, sign agreement)
3. Build null models (random permutation of ortholog mapping)
4. Identify conserved vs species-specific edges
5. Characterize biological properties of conserved vs fragile edges
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, fisher_exact
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_30_ortholog_transfer_stress_test/implementation"

print("=" * 60)
print("STEP 4: Cross-species edge comparison")
print("=" * 60)

# Load edges
human_edges = pd.read_csv(os.path.join(OUT_DIR, "human_edges.csv"))
mouse_edges = pd.read_csv(os.path.join(OUT_DIR, "mouse_edges.csv"))

with open(os.path.join(OUT_DIR, "tf_lists.json")) as f:
    tf_info = json.load(f)

shared_tfs = tf_info['shared_tfs']
print(f"\nShared TFs: {len(shared_tfs)}")
print(f"Human edges: {len(human_edges)}")
print(f"Mouse edges: {len(mouse_edges)}")

# Create edge keys (tf, target) for matching
human_edges['edge_key'] = human_edges['tf'] + '|' + human_edges['target']
mouse_edges['edge_key'] = mouse_edges['tf'] + '|' + mouse_edges['target']

# Filter to shared TFs only
human_shared = human_edges[human_edges['tf'].isin(shared_tfs)].copy()
mouse_shared = mouse_edges[mouse_edges['tf'].isin(shared_tfs)].copy()
print(f"\nEdges with shared TFs: Human={len(human_shared)}, Mouse={len(mouse_shared)}")

# Find shared edges (same TF-target pair in both species)
shared_edge_keys = set(human_shared['edge_key']) & set(mouse_shared['edge_key'])
print(f"Shared edge keys: {len(shared_edge_keys)}")

# Merge shared edges
human_shared_sub = human_shared[human_shared['edge_key'].isin(shared_edge_keys)].set_index('edge_key')
mouse_shared_sub = mouse_shared[mouse_shared['edge_key'].isin(shared_edge_keys)].set_index('edge_key')

merged = human_shared_sub[['tf', 'target', 'rho']].join(
    mouse_shared_sub[['rho']], lsuffix='_human', rsuffix='_mouse'
)
merged.columns = ['tf', 'target', 'rho_human', 'rho_mouse']
merged = merged.reset_index(drop=True)
print(f"\nMerged shared edges: {len(merged)}")

# ============================================================
# ANALYSIS 1: Rank correlation of shared edges
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 1: Rank correlation of shared edges")
print("=" * 60)

rho_rank, p_rank = spearmanr(merged['rho_human'], merged['rho_mouse'])
r_pearson, p_pearson = pearsonr(merged['rho_human'], merged['rho_mouse'])
print(f"  Spearman correlation of edge scores: rho={rho_rank:.4f}, p={p_rank:.2e}")
print(f"  Pearson correlation of edge scores: r={r_pearson:.4f}, p={p_pearson:.2e}")

# ============================================================
# ANALYSIS 2: Sign agreement
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 2: Sign agreement")
print("=" * 60)

merged['sign_human'] = np.sign(merged['rho_human'])
merged['sign_mouse'] = np.sign(merged['rho_mouse'])
merged['sign_agree'] = merged['sign_human'] == merged['sign_mouse']
sign_agreement = merged['sign_agree'].mean()
print(f"  Sign agreement: {sign_agreement:.4f} ({merged['sign_agree'].sum()}/{len(merged)})")

# Sign agreement for strong edges only
strong = merged[(merged['rho_human'].abs() > 0.2) & (merged['rho_mouse'].abs() > 0.2)]
if len(strong) > 0:
    sign_agree_strong = (strong['sign_human'] == strong['sign_mouse']).mean()
    print(f"  Sign agreement (|rho| > 0.2 in both): {sign_agree_strong:.4f} ({len(strong)} edges)")

very_strong = merged[(merged['rho_human'].abs() > 0.4) & (merged['rho_mouse'].abs() > 0.4)]
if len(very_strong) > 0:
    sign_agree_vs = (very_strong['sign_human'] == very_strong['sign_mouse']).mean()
    print(f"  Sign agreement (|rho| > 0.4 in both): {sign_agree_vs:.4f} ({len(very_strong)} edges)")

# ============================================================
# ANALYSIS 3: Top-k overlap
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 3: Top-k edge overlap")
print("=" * 60)

# Use absolute rho for ranking
for k in [100, 500, 1000, 2000, 5000]:
    h_top = set(human_shared.nlargest(k, 'abs_rho')['edge_key'])
    m_top = set(mouse_shared.nlargest(k, 'abs_rho')['edge_key'])
    overlap = len(h_top & m_top)
    # Expected overlap under random
    n_total_h = len(set(human_shared['edge_key']))
    n_total_m = len(set(mouse_shared['edge_key']))
    n_universe = len(set(human_shared['edge_key']) | set(mouse_shared['edge_key']))
    expected = k * k / n_universe
    fold_enrichment = overlap / expected if expected > 0 else float('inf')
    print(f"  Top-{k}: overlap={overlap}, expected={expected:.1f}, fold={fold_enrichment:.2f}")

# ============================================================
# ANALYSIS 4: Per-TF conservation
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 4: Per-TF conservation")
print("=" * 60)

per_tf_stats = []
for tf in shared_tfs:
    tf_merged = merged[merged['tf'] == tf]
    if len(tf_merged) < 10:
        continue
    rho_tf, p_tf = spearmanr(tf_merged['rho_human'], tf_merged['rho_mouse'])
    sign_ag = (tf_merged['sign_human'] == tf_merged['sign_mouse']).mean()
    per_tf_stats.append({
        'tf': tf,
        'n_shared_edges': len(tf_merged),
        'spearman_rho': rho_tf,
        'spearman_p': p_tf,
        'sign_agreement': sign_ag,
        'mean_abs_rho_human': tf_merged['rho_human'].abs().mean(),
        'mean_abs_rho_mouse': tf_merged['rho_mouse'].abs().mean(),
    })

per_tf_df = pd.DataFrame(per_tf_stats).sort_values('spearman_rho', ascending=False)
print("\nTop conserved TFs:")
print(per_tf_df.head(15).to_string())
print("\nLeast conserved TFs:")
print(per_tf_df.tail(15).to_string())

per_tf_df.to_csv(os.path.join(OUT_DIR, "per_tf_conservation.csv"), index=False)

# ============================================================
# ANALYSIS 5: Null models
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 5: Null models (permutation test)")
print("=" * 60)

np.random.seed(42)
n_permutations = 1000

# Null model 1: Random permutation of target gene labels in mouse
null_rho_corrs = []
null_sign_agreements = []
null_topk_overlaps = {k: [] for k in [100, 500, 1000]}

for perm_i in range(n_permutations):
    # Permute target labels in mouse edges
    perm_mouse = mouse_shared.copy()
    perm_mouse['target'] = np.random.permutation(perm_mouse['target'].values)
    perm_mouse['edge_key'] = perm_mouse['tf'] + '|' + perm_mouse['target']

    # Find shared edges under permutation
    perm_shared_keys = set(human_shared['edge_key']) & set(perm_mouse['edge_key'])

    if len(perm_shared_keys) > 10:
        h_sub = human_shared[human_shared['edge_key'].isin(perm_shared_keys)].set_index('edge_key')['rho']
        m_sub = perm_mouse[perm_mouse['edge_key'].isin(perm_shared_keys)].drop_duplicates('edge_key').set_index('edge_key')['rho']
        common = h_sub.index.intersection(m_sub.index)
        if len(common) > 10:
            rho_perm, _ = spearmanr(h_sub[common], m_sub[common])
            null_rho_corrs.append(rho_perm)
            sign_ag_perm = (np.sign(h_sub[common].values) == np.sign(m_sub[common].values)).mean()
            null_sign_agreements.append(sign_ag_perm)

    for k in [100, 500, 1000]:
        h_top = set(human_shared.nlargest(k, 'abs_rho')['edge_key'])
        m_top = set(perm_mouse.nlargest(k, 'abs_rho')['edge_key'])
        null_topk_overlaps[k].append(len(h_top & m_top))

print(f"\nNull correlation distribution ({len(null_rho_corrs)} valid permutations):")
null_rho_corrs = np.array(null_rho_corrs)
print(f"  Mean null rho: {null_rho_corrs.mean():.4f} +/- {null_rho_corrs.std():.4f}")
print(f"  Observed rho: {rho_rank:.4f}")
print(f"  Empirical p-value: {(null_rho_corrs >= rho_rank).mean():.6f}")
print(f"  Z-score: {(rho_rank - null_rho_corrs.mean()) / null_rho_corrs.std():.2f}")

null_sign = np.array(null_sign_agreements)
print(f"\nNull sign agreement: {null_sign.mean():.4f} +/- {null_sign.std():.4f}")
print(f"  Observed: {sign_agreement:.4f}")
print(f"  Empirical p-value: {(null_sign >= sign_agreement).mean():.6f}")

for k in [100, 500, 1000]:
    null_k = np.array(null_topk_overlaps[k])
    obs_h_top = set(human_shared.nlargest(k, 'abs_rho')['edge_key'])
    obs_m_top = set(mouse_shared.nlargest(k, 'abs_rho')['edge_key'])
    obs_overlap = len(obs_h_top & obs_m_top)
    print(f"\nTop-{k} overlap: observed={obs_overlap}, null={null_k.mean():.1f} +/- {null_k.std():.1f}")
    print(f"  Empirical p-value: {(null_k >= obs_overlap).mean():.6f}")

# ============================================================
# ANALYSIS 6: Conserved vs fragile edge characterization
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 6: Conserved vs fragile edges")
print("=" * 60)

# Define conservation score
merged['conservation_score'] = merged['rho_human'] * merged['rho_mouse']
merged['abs_conservation'] = merged['rho_human'].abs() * merged['rho_mouse'].abs()
# Strong in both and same sign = high conservation
# Strong in both but opposite sign = anti-conserved

# Top conserved edges (strong positive conservation)
conserved = merged.nlargest(500, 'conservation_score')
print("\nTop 20 conserved edges (same sign, strong in both):")
print(conserved.head(20)[['tf', 'target', 'rho_human', 'rho_mouse', 'conservation_score']].to_string())

# Anti-conserved edges (opposite sign)
anti_conserved = merged.nsmallest(100, 'conservation_score')
print("\nTop 20 anti-conserved edges (opposite sign):")
print(anti_conserved.head(20)[['tf', 'target', 'rho_human', 'rho_mouse', 'conservation_score']].to_string())

# Fragile edges: strong in one species, weak in other
merged['delta_abs_rho'] = (merged['rho_human'].abs() - merged['rho_mouse'].abs()).abs()
merged['max_abs_rho'] = merged[['rho_human', 'rho_mouse']].abs().max(axis=1)
fragile = merged[(merged['max_abs_rho'] > 0.3) & (merged['delta_abs_rho'] > 0.3)]
fragile = fragile.sort_values('delta_abs_rho', ascending=False)
print(f"\nFragile edges (|delta_rho| > 0.3, max |rho| > 0.3): {len(fragile)}")
print(fragile.head(20)[['tf', 'target', 'rho_human', 'rho_mouse', 'delta_abs_rho']].to_string())

# ============================================================
# ANALYSIS 7: What makes edges conserved?
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 7: Properties of conserved vs fragile edges")
print("=" * 60)

# Label edges
merged['category'] = 'weak'
merged.loc[merged['conservation_score'] > merged['conservation_score'].quantile(0.9), 'category'] = 'conserved'
merged.loc[merged['conservation_score'] < merged['conservation_score'].quantile(0.1), 'category'] = 'anti_conserved'
merged.loc[(merged['max_abs_rho'] > 0.3) & (merged['delta_abs_rho'] > 0.25), 'category'] = 'fragile'

print("\nEdge category distribution:")
print(merged['category'].value_counts())

# Compare mean |rho| across categories
for cat in ['conserved', 'fragile', 'anti_conserved', 'weak']:
    subset = merged[merged['category'] == cat]
    if len(subset) == 0:
        continue
    print(f"\n  {cat} (n={len(subset)}):")
    print(f"    Mean |rho_human|: {subset['rho_human'].abs().mean():.4f}")
    print(f"    Mean |rho_mouse|: {subset['rho_mouse'].abs().mean():.4f}")
    print(f"    Sign agreement: {(subset['sign_human'] == subset['sign_mouse']).mean():.4f}")

# Save all results
merged.to_csv(os.path.join(OUT_DIR, "merged_edges.csv"), index=False)

# Compile results summary
results = {
    "n_shared_edges": len(merged),
    "spearman_rho": float(rho_rank),
    "spearman_p": float(p_rank),
    "pearson_r": float(r_pearson),
    "pearson_p": float(p_pearson),
    "sign_agreement_all": float(sign_agreement),
    "sign_agreement_strong_02": float(sign_agree_strong) if len(strong) > 0 else None,
    "sign_agreement_strong_04": float(sign_agree_vs) if len(very_strong) > 0 else None,
    "null_rho_mean": float(null_rho_corrs.mean()),
    "null_rho_std": float(null_rho_corrs.std()),
    "observed_vs_null_zscore": float((rho_rank - null_rho_corrs.mean()) / null_rho_corrs.std()),
    "n_conserved": int((merged['category'] == 'conserved').sum()),
    "n_fragile": int((merged['category'] == 'fragile').sum()),
    "n_anti_conserved": int((merged['category'] == 'anti_conserved').sum()),
    "per_tf_conservation_median": float(per_tf_df['spearman_rho'].median()),
}
with open(os.path.join(OUT_DIR, "analysis_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n\nAll results saved.")
print("Done.")
