# Tier 2: CSSI-Derived Biology + Context-Dependence + Per-TF Characterization

## Goal

Three analyses that strengthen the biological contribution of the paper by (a) demonstrating CSSI enables biological discovery, (b) explaining the context-dependence finding biologically, and (c) characterizing which TFs yield reliable attention-derived edges.

**IMPORTANT**: Do NOT modify any paper files (main_v3.tex, supplementary_v3.tex). All results go to files in this directory (`experiments/biological_impact/`). Results will be reviewed before any paper integration.

## Python Environment

```
~/anaconda3/envs/bio_mech_interp/bin/python
```

NumPy 1.26.4. Use `sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)` for line-buffered output. Save incremental results to JSON after each major compute block. Include resume capability (check for existing partial results).

---

## Analysis C: CSSI-Derived Cell-State-Specific Regulatory Programs

### Motivation

CSSI currently demonstrates improved *metric performance* (1.85× TRRUST F1). To elevate this to biological discovery, we need to show that CSSI-derived strata correspond to meaningful cell states and that stratum-specific edges recapitulate known cell-type-specific regulatory programs.

### What to compute

1. **CSSI stratification on Tabula Sapiens immune data** (20,000 cells):
   - Apply Leiden clustering on k-NN graph from Geneformer embeddings (or scanpy's standard pipeline) with K=5-7 strata (the CSSI optimum)
   - Annotate strata by mapping to known immune cell types using marker genes:
     - T cells: CD3D, CD3E, CD4, CD8A
     - B cells: CD19, MS4A1, CD79A
     - NK cells: NKG7, GNLY, KLRD1
     - Monocytes: CD14, LYZ, S100A8
     - Dendritic cells: FCER1A, CLEC10A
   - Confirm strata correspond to distinct cell types (or mixed states)

2. **Per-stratum edge score computation**:
   - Within each stratum, compute Spearman correlation edges (2,000 HVGs)
   - Also compute CSSI-max and CSSI-mean aggregated edges
   - For each stratum, evaluate TRRUST AUROC

3. **Cell-type-specific regulatory program validation**:
   - For each stratum (cell type), extract the top-500 strongest edges
   - Test whether these edges are enriched for known cell-type-specific regulatory programs:
     - T cell strata: expect TCF7→target, GATA3→target, TBX21→target edges
     - Monocyte strata: expect SPI1(PU.1)→target, CEBPA→target edges
     - B cell strata: expect PAX5→target, EBF1→target edges
   - Use Fisher's exact test for enrichment
   - Cross-validate: do T-cell-specific edges disappear in monocyte strata?

4. **Differential edge analysis**:
   - For each TF-target pair, compute the variance of edge scores across strata
   - Identify the most variable edges (those whose regulatory strength changes across cell states)
   - Characterize these: are they known to be cell-type-specific regulators?
   - Compute a "regulatory specificity index" per edge

### Data locations

- Tabula Sapiens immune: `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad`
- TRRUST: `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv`
- CSSI implementation reference: `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/cssi_real/`

### Implementation notes

- The Tabula Sapiens immune data has a `feature_name` column that is Categorical dtype → use `.astype(str)` before `var_names_make_unique()`
- Existing CSSI code in `cssi_real/` can be adapted for stratification
- For cell type annotation, use scanpy's `sc.tl.rank_genes_groups()` on the strata to identify marker genes, then match to known markers
- If strata don't cleanly map to cell types, this is still informative (report as mixed states)
- Save stratum assignments and per-stratum correlation matrices for reuse

### Output files

- `cssi_biology/stratum_assignments.json` — cell-to-stratum mapping with cell type annotations
- `cssi_biology/per_stratum_edges.npz` — per-stratum correlation matrices
- `cssi_biology/regulatory_specificity.json` — differential edge analysis results
- `cssi_biology_results.json` — all validation statistics
- `cssi_biology_summary.md` — human-readable summary

---

## Analysis D: Context-Dependence Characterization

### Motivation

The paper shows attention outperforms correlation in RPE1 (d=0.47) but not in K562. Why? If we can identify a biological or technical predictor of when attention works, this provides actionable guidance and biological insight.

### What to compute

1. **Dataset-level characterization** across available perturbation datasets:
   - K562 CRISPRi (n=280): attention ≈ correlation
   - K562 CRISPRa (n=77): attention < correlation
   - RPE1 CRISPRi (n=1,251): attention > correlation
   - Shifrut T cells (n=7): no difference
   - Tian neurons (n=7): attention trending better

   For each dataset, compute:
   - Mean perturbation effect size (mean absolute LFC across DE genes)
   - Number of DE genes per perturbation (at LFC>0.5)
   - Expression rank stability (variance of gene ranks across cells)
   - Gene universe size and composition
   - Proportion of perturbation genes that are TFs
   - Cell-to-cell expression heterogeneity (mean pairwise distance)

2. **Per-perturbation analysis within RPE1 and K562**:
   - For each perturbation, compute: attention AUROC, correlation AUROC, and the difference
   - Correlate the attention advantage (attention AUROC - correlation AUROC) with:
     - Number of DE targets
     - Mean LFC of DE targets
     - Perturbed gene's expression level
     - Perturbed gene's expression variance
     - Perturbed gene's connectivity in TRRUST
   - Are there perturbation-level predictors of when attention outperforms correlation?

3. **Methodological confound test**:
   - RPE1 uses 3,309 genes (with perturbation genes forced in) vs K562's 2,000
   - Re-run K562 analysis with perturbation genes forced into the HVG set (matching RPE1 protocol)
   - Does this change the attention-correlation relationship?
   - Also try: restrict RPE1 to top 2,000 HVG only (matching K562 protocol)

### Data locations

- Replogle K562 CRISPRi: `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/replogle/K562_gwps_normalized_bulk_01.h5ad`
- RPE1 data: check `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/` for scperturb downloads
- Round9 extraction (K562): `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/round9_unified_extraction/`
- Non-K562 replication results: `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/round13_non_k562_replication*.json`
- Existing RPE1 attention edges: search in `experiments/` for round13 outputs

### Implementation notes

- The K562 vs RPE1 methodological confound (forced HVG inclusion) is mentioned in the paper as a caveat. Testing it directly would either strengthen or qualify the context-dependence finding
- For per-perturbation analysis, the existing `de_labels.pkl` and per-layer attention edges can be reused for K562; RPE1 may need recomputation
- MPS GPU: batch_size=1 for any new attention extraction; ~1.3-1.5 cells/s
- The perturbation-level analysis (correlating attention advantage with perturbation features) can be done without new model runs — just reanalyse existing AUROC distributions

### Output files

- `context_dependence/dataset_characteristics.json` — per-dataset feature comparison
- `context_dependence/perturbation_level_analysis.json` — per-perturbation predictors
- `context_dependence/hvg_confound_test.json` — results of matching HVG protocols
- `context_dependence_results.json` — all analyses combined
- `context_dependence_summary.md` — human-readable summary

---

## Analysis E: Per-TF Characterization

### Motivation

7/18 TFs in the bootstrap analysis have robust edge-level signal (95% CI entirely above 0.5). The top performers are LEF1 (0.997), KLF1 (0.995), PPARG (0.900). What distinguishes successful from unsuccessful TFs? This could reveal when attention-derived edges are reliable.

### What to compute

1. **Characterize all 18 evaluable TFs** along these axes:
   - **Regulon size**: number of known TRRUST targets
   - **Expression level**: mean expression in Tabula Sapiens immune cells
   - **Expression variance**: across cells
   - **TF family**: zinc finger, bHLH, homeodomain, etc. (from TFClass or similar)
   - **Functional category**: master regulator, lineage-specific, housekeeping, signal-dependent
   - **Number of TRRUST targets in HVG universe**: evaluable targets
   - **Bootstrap AUROC and 95% CI**: from existing results

2. **Statistical tests**:
   - Correlate TF AUROC with regulon size (Spearman)
   - Correlate TF AUROC with expression level and variance
   - Mann-Whitney test: is AUROC higher for master regulators vs others?
   - Is there a minimum regulon size threshold for reliable AUROC?

3. **Characterize the 7 successful vs 11 unsuccessful TFs**:
   - Are successful TFs predominantly master regulators?
   - Do they have larger regulons?
   - Are they more highly expressed?
   - Are they in specific functional categories?

### Data locations

- Bootstrap per-TF results: `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/bootstrap_per_tf_results/`
- Tabula Sapiens immune: same as Analysis C
- TRRUST: same as Analysis C
- TF classification: download from TFClass (http://tfclass.bioinf.med.uni-goettingen.de/) or use manual annotation for 18 TFs

### Implementation notes

- Only 18 TFs — this is a small sample. Be honest about statistical power limitations
- Focus on descriptive characterization and hypothesis generation rather than definitive conclusions
- If existing bootstrap results aren't saved, re-run `bootstrap_per_tf.py` (or use the existing analysis values from the paper: LEF1=0.997, KLF1=0.995, PPARG=0.900, etc.)
- Manual annotation of 18 TFs is feasible and may be more reliable than automated classification

### Output files

- `per_tf/tf_characterization.json` — all TF features and statistical tests
- `per_tf/tf_characterization_summary.md` — human-readable summary with table

---

## Success Criteria

**Analysis C is successful if**: CSSI strata map to known immune cell types AND stratum-specific top edges are enriched for known cell-type-specific regulatory programs. Even partial success (2-3 cell types with clear enrichment) is valuable.

**Analysis D is successful if**: At least one biological or technical factor clearly predicts the attention advantage across cell types or perturbations. The methodological confound test (HVG protocol matching) is valuable regardless of outcome.

**Analysis E is successful if**: A clear pattern distinguishes the 7 successful TFs from the 11 unsuccessful ones. Given the small sample, even suggestive patterns (e.g., "all 3 top TFs are lineage master regulators") are informative.

## Timeline

Analyses can run in parallel. Expected: 7-14 days total.
- Analysis C: 7-10 days (stratification + validation against cell-type references)
- Analysis D: 5-7 days (characterization + confound test; most data exists)
- Analysis E: 2-3 days (small sample, mostly characterization)

## Priority Order

If time is limited: E (quickest) → D (medium, high insight potential) → C (longest but highest NMI impact).
