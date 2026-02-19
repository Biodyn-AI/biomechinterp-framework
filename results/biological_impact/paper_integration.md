# Tier 2 Biological Impact Analyses — Paper Integration

## Summary of Findings

Three Tier 2 analyses (C, D, E) were executed and hardened with stress tests.
**The strongest result is the HVG confound test (Analysis D).** The others provide
supplementary context but are not individually paper-worthy without caveats.

---

## Analysis D: Context-Dependence Characterization (HIGH PRIORITY)

### Core Finding: HVG Protocol Asymmetry Does NOT Explain the RPE1 Attention Advantage

The RPE1 and K562 evaluations used different HVG protocols:
- **K562**: Top-2000 HVGs by variance. All 280 perturbation genes are naturally in this set (100% coverage).
- **RPE1**: Top-2000 HVGs + 1309 forced perturbation genes = 3309 total. Only 418/1251 perturbation genes are naturally high-variance.

**Confound test**: Restricting RPE1 to top-2000 HVGs (matching K562 protocol, no forced inclusion):

| Metric | Original (3309 HVGs, n=418) | Restricted (2000 HVGs, n=418) |
|--------|:---------------------------:|:-----------------------------:|
| Mean attention AUROC | 0.699 | 0.762 |
| Mean correlation AUROC | 0.723 | 0.593 |
| Mean Δ (attn − corr) | −0.024 | **+0.168** |
| Fraction attention wins | 0.488 | 0.890 |

The attention advantage **increases** when matching the K562 protocol. This rules out the
hypothesis that forced HVG inclusion drives the RPE1 advantage. Correlation degrades more
than attention when the evaluation universe shrinks (−0.129 vs +0.063), because correlation
benefits from having more co-expressed genes in the scoring universe.

**Reverse confound (K562 expansion) is moot**: All 280 K562 perturbation genes are already
in the top-2000 HVGs by variance. There is nothing to expand. The asymmetry is unidirectional.

### K562 Delta Discrepancy: Aggregation Method Matters

The originally reported K562 Δ=+0.001 (Cohen's d=0.01) was a "from paper" hardcoded value.
Fresh per-perturbation computation from round 9 data (layer 15, layer-mean edges):

| Aggregation | Attn AUROC | Corr AUROC | Δ |
|-------------|:----------:|:----------:|:---:|
| Pooled (edge-level, all 555K edges) | 0.714 | 0.637 | **+0.077** |
| Equal-weight (per-perturbation mean) | 0.764 | 0.703 | **+0.061** |
| Target-weighted (per-perturbation) | 0.716 | 0.668 | **+0.048** |

All three show a **positive** attention advantage on K562, not the zero-difference originally
reported. The discrepancy with the paper's 0.704/0.703 values likely stems from a different
computation path in the original round 9 analysis (e.g., layer-mean across heads vs specific
head selection, or a different evaluation universe).

**Recommendation**: Report equal-weight per-perturbation means consistently across all datasets.

### Layer Confirmation

All evaluations use **Layer 15** (0-indexed: 14), confirmed from:
- `round9_unified_extraction/layer_mean_edges/layer_14_edges.pkl` (K562)
- `round13_non_k562_replication/ReplogleRPE1/attention_edges.pkl[14]` (RPE1)
- This is the best-performing layer from the main paper's layer-wise analysis.

### Bootstrap Confidence Intervals (10,000 samples)

| Dataset | n | Mean Δ | 95% CI | Wilcoxon p |
|---------|---|:------:|:------:|:----------:|
| RPE1 (3309 HVGs) | 1167 | +0.090 | [+0.079, +0.101] | 2.2×10⁻⁵⁴ |
| K562 (2000 HVGs) | 280 | +0.060 | [+0.040, +0.080] | 8.9×10⁻⁸ |
| RPE1 restricted (2000 HVGs) | 418 | +0.168 | [+0.155, +0.182] | 4.9×10⁻⁶⁰ |

All CIs exclude zero. Attention outperforms correlation in both cell lines.

### Suggested Paper Text (Results Section)

> To test whether the HVG protocol asymmetry between K562 (2,000 genes) and RPE1
> (3,309 genes with forced perturbation gene inclusion) explains the stronger attention
> advantage in RPE1, we re-evaluated RPE1 using only the top-2,000 HVGs by variance,
> matching the K562 protocol. Restricting the gene universe to 2,000 did not diminish
> the attention advantage; rather, it increased from Δ=−0.024 to Δ=+0.168 for the 418
> perturbation genes in the restricted set (Wilcoxon p<10⁻⁴⁶), as correlation scores
> degraded more than attention scores when the evaluation universe shrank. Bootstrap
> 95% CIs on the per-perturbation attention advantage exclude zero for both K562
> (Δ=+0.060 [+0.040, +0.080]) and RPE1 (Δ=+0.090 [+0.079, +0.101]).

### Caveats to Document

1. **Selection bias in restricted set**: Only 418 of 1251 RPE1 perturbation genes are
   naturally high-variance. The restricted comparison evaluates a biased subset (high-variance
   genes only). The 833 excluded genes were low-variance genes forced into the original set.

2. **Different evaluation universes**: The "original" AUROCs were computed on 3308 targets,
   the "restricted" on 1999 targets. These are not directly comparable in absolute terms.
   The within-protocol attention vs correlation comparison is the valid test.

3. **Attention scores unchanged**: The attention edges were precomputed on the 3309-gene
   context. Restricting evaluation to 2000 targets changes which edges are scored but not
   the scores themselves. A fully fair test would re-extract attention on only 2000 tokens,
   but this would require GPU re-extraction (~3 hours).

---

## Analysis E: Per-TF Characterization (MEDIUM — Supplementary)

### Result: Suggestive but Not Robust

After hardening with permutation tests and BH-FDR correction:
- Master regulators vs others: raw p=0.011, **BH q=0.122** (not significant)
- 0 of 11 tests significant after BH correction at α=0.05
- 13 of 18 TFs have only 1 evaluable TRRUST target (single-gene AUROC, not regulon-level)
- When restricted to ≥3-target TFs (n=5): p=0.297

### Supplementary Paragraph (Honest Power Caveats)

> **Per-TF characterization (exploratory).** We examined whether the per-TF AUROC
> varies systematically with TF biology. Among 18 evaluable TFs in the Tabula Sapiens
> immune dataset, manually annotated master regulators (n=9) had higher mean AUROC
> (0.80±0.18) than other TF categories (0.58±0.18; permutation p=0.011, 10,000 shuffles).
> However, this comparison did not survive Benjamini-Hochberg correction across all 11
> tests performed (q=0.12). Importantly, 13 of 18 TFs had only a single evaluable
> TRRUST target in the HVG set, meaning their AUROC reflects the rank of one gene among
> ~2,000 rather than a regulon-level assessment. When restricted to the 5 TFs with ≥3
> evaluable targets (GATA1, PPARG, EGR1, WT1, FOXF2), the master regulator advantage
> was not significant (p=0.30). We therefore treat the master regulator association as
> hypothesis-generating; larger TF databases with greater regulon coverage (e.g.,
> DoRothEA, ChIP-Atlas) would be needed to test this rigorously. No tested TF property
> — including mean expression, expression variance, or regulon size — significantly
> predicted AUROC after multiple testing correction.

---

## Analysis C: CSSI Biology (LOW — Likely Drop from Paper)

### Result: Marginal Improvement, Honestly Reported

With unsupervised Leiden clustering (6 resolutions tested):

| Method | AUROC | Δ vs Pooled |
|--------|:-----:|:-----------:|
| Pooled (no stratification) | 0.828 | — |
| CSSI-max (unsupervised, res=0.5, K=21) | 0.835 | +0.007 |
| CSSI-mean (unsupervised, res=0.5) | 0.849 | +0.021 |
| CSSI-max (ground-truth labels, K=18) | 0.833 | +0.005 |
| CSSI-mean (ground-truth labels) | 0.853 | +0.025 |

- Unsupervised clusters recover cell types moderately well (ARI=0.53, NMI=0.68)
- CSSI-max barely outperforms pooled (+0.007); at 4 of 6 resolutions it underperforms
- CSSI-mean provides the best improvement but only +0.02 AUROC
- Unsupervised clusters perform comparably to ground-truth labels

### Cell-Type-Specific TF Enrichment (Follow-Up 3)

For each unsupervised Leiden cluster, we tested whether TRRUST edges from cell-type-matching
TFs (e.g., SPI1 for monocyte clusters, BCL6 for B cell clusters) have higher stratum-specific
correlation than edges from non-matching TFs (Mann-Whitney, one-sided).

**Result: 7 of 20 testable clusters significant after BH correction (q<0.05).**

Significant clusters: B cell clusters (3/3), neutrophil clusters (2/2), macrophage (1/4),
T cell (1/7). The signal is concentrated in high-purity clusters (B cell purity=0.99-1.00,
neutrophil 0.91-0.97) and absent in mixed/impure ones. Fisher's exact tests for TF
enrichment in top-100 edges: 0/20 significant (too few TFs per cell type in TRRUST).

### Recommendation

**Drop Analysis C from the main paper.** The CSSI contribution is already validated by
the synthetic benchmark and TRRUST analysis in Tier 1. The biological impact analysis shows:
- Only marginal CSSI improvement over pooled correlation (Δ<0.03 AUROC)
- CSSI-max underperforms pooled at most resolutions
- TF enrichment is present in only 7/20 clusters (concentrated in high-purity lineages)
- The effect sizes are small (rank-biserial r = −0.26 to −0.59)

If reviewers specifically request cell-type-specific regulatory discovery evidence,
the CSSI-mean result (+0.02) and partial TF enrichment (7/20) can be included in
supplementary with appropriate caveats about the marginal improvement.

---

## Integration with Tier 1 Findings

### Analysis A: Value-Weighted Edges (Negative Result)
Value-weighting attention edges (incorporating hidden state norms) does not improve TRRUST
AUROC over unweighted attention.

### Analysis B: Layer-Specific Biological Specialization (Most Exciting)
Different layers capture different aspects of gene regulation. This is the strongest
Tier 2 finding and should receive prominent treatment.

### Recommended Paper Structure for Tier 2

1. **Main text**: HVG confound test (Analysis D) as a methodological robustness check
   in the perturbation benchmark section. Include bootstrap CIs. One paragraph.

2. **Main text**: Layer-specific specialization (Analysis B) in the mechanistic
   interpretation section. This is the most novel and exciting finding.

3. **Supplementary**: Per-TF characterization (Analysis E) as an exploratory analysis
   with honest power caveats. One paragraph + table.

4. **Omit**: CSSI biology (Analysis C) unless specifically requested by reviewers.

---

## File Locations

| Analysis | Summary | Full Data |
|----------|---------|-----------|
| D (HVG confound) | `context_dependence/hardened_summary.md` | `context_dependence/hardened_results.json` |
| E (Per-TF) | `per_tf/tf_characterization_hardened_summary.md` | `per_tf/tf_characterization_hardened.json` |
| C (CSSI) | `cssi_biology/hardened_summary.md` | `cssi_biology/hardened_results.json` |
| **This document** | `paper_integration.md` | — |
