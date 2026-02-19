# Tier 2 Follow-Up Tasks: Hardened Results → Paper-Ready Outputs

## Context

Three Tier 2 analyses (C, D, E) have been run and hardened. This follow-up task produces paper-ready outputs and addresses remaining issues identified in review.

**IMPORTANT**: Do NOT modify any paper files (main_v3.tex, supplementary_v3.tex). All outputs go to files in this directory (`experiments/biological_impact/`). Results will be reviewed before any paper integration.

## Python Environment

```
~/anaconda3/envs/bio_mech_interp/bin/python
```

NumPy 1.26.4. Use `sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)` for line-buffered output. Save incremental results to JSON.

---

## Follow-Up 1: Analysis D — Paper-Ready HVG Confound Integration

**Priority: HIGH** — This is the strongest Tier 2 finding and should go into the paper.

### Issue identified in review

The K562 bootstrap CI shows per-perturbation mean delta = +0.060 (CI [+0.040, +0.080]), but the paper reports K562 attention ≈ correlation (mean AUROC: 0.704 vs 0.703, delta = 0.001). This discrepancy likely arises because:
- The per-perturbation mean delta **equally weights** each perturbation regardless of number of DE targets
- The paper's overall AUROC effectively **weights by number of targets** (perturbations with more DE targets contribute more)
- The two can legitimately differ

This needs to be verified and documented.

### What to compute

1. **Verify K562 delta discrepancy**: Load `context_dependence/k562_per_perturbation.csv`. Compute both:
   - Simple mean of per-perturbation deltas (should match the +0.060 bootstrap value)
   - Weighted mean where each perturbation is weighted by n_de (number of DE targets)
   - Report both and explain the discrepancy in the output summary

2. **Verify which layer the HVG confound test used**: Read the hardening script `harden_analysis_d.py` lines 152-166. Confirm it uses layer 14 (0-indexed) = L15 (1-indexed). Document this clearly.

3. **Compute the reverse confound test (K562 with forced HVG inclusion)**: The current test restricts RPE1 from 3,309→2,000 HVGs. The complementary test should expand K562 from 2,000 HVGs to include perturbation genes (matching RPE1's protocol). This completes the bidirectional comparison:
   - If restricting RPE1 HVGs changes the delta, does *expanding* K562 HVGs also change it?
   - Load K562 data from `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/replogle/K562_gwps_normalized_bulk_01.h5ad`
   - Load existing K562 attention edges from `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/round9_unified_extraction/layer_mean_edges/` (use L15 = `layer_14_edges.pkl` if that naming convention exists, otherwise check what files are available)
   - Load DE labels from `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/round9_unified_extraction/de_labels.pkl`
   - Load existing HVG list from `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/round9_unified_extraction/hvg_gene_list.json`
   - Expand HVG set to include perturbation gene names (matching RPE1's forced-inclusion protocol)
   - Recompute Spearman correlation on expanded gene set using K562 control cells
   - Re-evaluate per-perturbation AUROC with the expanded gene universe
   - Compare to original K562 results (2,000 HVG)

4. **Write paper-integration summary**: Produce `context_dependence/paper_integration.md` with:
   - A clear narrative suitable for 1-2 paragraphs in supplementary
   - Key numbers for main text (1-2 sentences)
   - Specific finding: "Restricting RPE1 to 2,000 HVGs (matching K562 protocol) *increases* the attention advantage (delta swings from -0.024 to +0.168), driven primarily by correlation AUROC dropping (-0.129) while attention is robust (+0.063). [If K562 forced-inclusion changes delta, report that too.] Gene universe size thus confounds the magnitude but not the direction of the context-dependence finding."

### Data locations

- K562 h5ad: `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/replogle/K562_gwps_normalized_bulk_01.h5ad`
- K562 extraction: `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/round9_unified_extraction/`
- Hardened D results: `context_dependence/hardened_results.json`, `context_dependence/hardened_summary.md`
- Hardening script: `harden_analysis_d.py`
- K562 per-perturbation CSV: `context_dependence/k562_per_perturbation.csv`
- RPE1 per-perturbation CSV: `context_dependence/rpe1_per_perturbation.csv`
- RPE1 restricted CSV: `context_dependence/rpe1_restricted_2000hvg.csv`

### Output files

- `context_dependence/k562_forced_hvg_test.json` — K562 expanded-HVG results
- `context_dependence/delta_discrepancy_analysis.json` — K562 delta discrepancy explanation
- `context_dependence/paper_integration.md` — paper-ready summary with specific numbers and narrative

---

## Follow-Up 2: Analysis E — Supplementary Paragraph with Power Caveats

**Priority: MEDIUM** — Honest supplementary observation, no further computation needed.

### What to produce

Write `per_tf/paper_integration.md` containing:

1. **A supplementary paragraph** (200-300 words) that:
   - States the analysis found a suggestive but non-significant pattern: master regulators tend to have higher edge-level AUROC than other TF categories
   - Reports the permutation test result: p=0.011 (raw), but q=0.122 after BH correction across 11 tests
   - Explicitly notes the severe power limitation: 13/18 TFs have only 1 evaluable target in the HVG universe, making their AUROC the rank of a single gene, not a regulon-level assessment
   - Reports that when restricted to TFs with ≥3 evaluable targets (n=5), the pattern attenuates (permutation p=0.297)
   - Frames as hypothesis-generating: "the pattern is consistent with the interpretation that attention more reliably captures regulatory relationships for master regulators with large, well-characterized regulons, but the sample is too small for definitive conclusions"
   - Notes the top 3 TFs (LEF1 0.997, KLF1 0.995, PPARG 0.900) are all master regulators, but LEF1 and KLF1 each have only 1 evaluable target

2. **A table** (for supplementary) with all 18 TFs showing: TF name, category, evaluable targets, AUROC, 95% CI (if available), and whether classified as "successful" (CI entirely above 0.5)

3. **One sentence for main text** that could be added to the Discussion limitations, e.g.: "A per-TF characterization found a suggestive but non-significant trend for master regulators to yield higher edge-level AUROC (permutation p = 0.011 uncorrected, q = 0.122 after BH correction; Supplementary Note X), limited by the small number of evaluable TFs."

### Data locations

- Hardened per-TF results: `per_tf/tf_characterization_hardened.json`, `per_tf/tf_characterization_hardened_summary.md`
- Original per-TF results: `per_tf/tf_characterization.json`

### Output files

- `per_tf/paper_integration.md` — supplementary paragraph + table + main text sentence

---

## Follow-Up 3: Analysis C — Cell-Type-Specific TF Enrichment or Drop Decision

**Priority: LOW** — The current CSSI biology results are technically correct but biologically thin. The originally planned biological validation (cell-type-specific TF enrichment) was not performed.

### What to compute

1. **Cell-type-specific TF-target enrichment** (the originally planned test from TASK_TIER2.md):
   - The hardened results show 22 Leiden clusters at resolution 0.5, mapping to clear cell types (B cells, T cells, monocytes, neutrophils, macrophages, NK cells, etc.)
   - For each cluster (cell type), extract the top-500 strongest Spearman correlation edges from the per-stratum correlation matrices
   - Load TRRUST and test whether stratum-specific top edges are enriched for known cell-type-specific TF-target pairs:
     - T cell clusters (0, 1, 7, 10, 16, 17, 20, 21): test for TCF7, GATA3, TBX21, RUNX3 targets
     - B cell clusters (3, 5, 19): test for PAX5, EBF1, IRF4 targets
     - Monocyte/macrophage clusters (2, 6, 13, 15, 18): test for SPI1/PU.1, CEBPA, IRF8 targets
     - Neutrophil clusters (4, 11): test for CEBPE, GFI1 targets
   - Use Fisher's exact test for enrichment
   - **Critical cross-validation**: do T-cell-specific TF edges appear *more* in T cell clusters than in monocyte clusters? Compute enrichment ratio across cell types.

2. **If the enrichment tests are all non-significant** (which is possible given the 2,000 HVG universe may not contain enough TF-target pairs), write a brief note in `cssi_biology/paper_integration.md` explaining:
   - CSSI shows modest improvement over pooled correlation (+0.007 to +0.021 AUROC) when applied to unsupervised Leiden clusters on Tabula Sapiens immune data
   - Unsupervised clusters recover known cell types (ARI=0.53 at optimal resolution)
   - Cell-type-specific TF enrichment could not be validated because [reason: too few TF-target pairs in HVG universe / TFs not differentially active / etc.]
   - Recommend: this analysis is not strong enough for the paper in its current form; the CSSI contribution is better supported by the existing synthetic and TRRUST validation already in the paper

3. **If enrichment IS significant for at least 2-3 cell types**, write a supplementary paragraph describing the cell-type-specific regulatory programs recovered by CSSI stratification.

### Data locations

- Hardened CSSI results: `cssi_biology/hardened_results.json`, `cssi_biology/hardened_summary.md`
- Per-stratum edge matrices: `cssi_biology/per_stratum_edges.npz`
- Stratum assignments: `cssi_biology/stratum_assignments.json`
- TRRUST: `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv`
- HVG gene list: check the stratum assignments JSON or the CSSI biology checkpoint for the gene list used

### Output files

- `cssi_biology/tf_enrichment_results.json` — Fisher's exact test results per cell type
- `cssi_biology/paper_integration.md` — paper-ready summary OR drop recommendation

---

## Priority Order

**Follow-Up 1 (Analysis D)** is the most important — the HVG confound finding is novel and paper-worthy. The reverse confound test (K562 forced inclusion) completes the story.

**Follow-Up 2 (Analysis E)** is quick to write — just producing text from existing results.

**Follow-Up 3 (Analysis C)** is lowest priority — run the enrichment test but be prepared to recommend dropping this from the paper if results are weak.

## Success Criteria

- Follow-Up 1 is successful if: the K562 reverse confound test runs and the `paper_integration.md` provides a clear, quotable narrative with all key numbers.
- Follow-Up 2 is successful if: the supplementary paragraph is honest about power limitations while correctly framing the hypothesis-generating finding.
- Follow-Up 3 is successful if: the TF enrichment either finds significant cell-type-specific patterns OR the `paper_integration.md` clearly explains why the analysis should be dropped.
