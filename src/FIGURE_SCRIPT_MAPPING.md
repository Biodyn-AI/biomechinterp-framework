# Figure and Table Generation Script Mapping

This document provides a detailed mapping of which scripts generate which figures and tables in the NMI paper.

## Main Paper Figures

### Figure 1: Scaling Failure Across Model Tiers
- **Script**: `01_scaling_analysis/generate_extended_evidence.py`
- **Output**: `figures/fig_ext_multiseed_scaling_trrust_ci.png`
- **Description**: Multi-seed F1 performance with 95% confidence intervals against TRRUST showing monotonic degradation as cell count increases from 200 to 3000 across small, medium, and large model tiers.

### Figure 2: Retrieval Collapse with Scaling  
- **Script**: `01_scaling_analysis/generate_extended_evidence.py`
- **Output**: `figures/fig_ext_rerun_tp_vs_random.png`
- **Description**: Observed versus random-expected true positives showing enrichment collapse from above-chance to below-chance performance.

### Figure 3: Non-additivity in Mediation Analysis
- **Script**: `02_mediation_bias/run_full_paper_pipeline.py`
- **Output**: `figures/fig1_real_residual_nonadditivity.png`
- **Description**: Absolute total effect, residual non-additivity, and lower-bound interaction magnitude per run-pair.

### Figure 4: Ranking Certificate Fragility
- **Script**: `02_mediation_bias/run_targeted_pairwise_patches.py`
- **Output**: `figures/fig2_ranking_sensitivity.png`
- **Description**: Mean certified pair fraction versus structural multiplier λ.

### Figure 5: Detectability Phase Diagrams
- **Script**: `03_detectability/run_detectability_phase_diagram.py`
- **Output**: `figures/fig_phase_diagram_regimes.png`
- **Description**: Different regimes where attention-like versus intervention-like signals become detectable.

### Figure 6: Cross-Tissue Consistency Variability
- **Script**: `04_cross_tissue/make_paper_figures.py`
- **Output**: `figures/fig3_cross_tissue_scatter.png`
- **Description**: Scatter plots of component-level effects between tissue pairs.

### Figure 7: Cross-Species Edge Score Conservation
- **Script**: `06_ortholog_transfer/05_figures.py`
- **Output**: `figures/fig_ortholog_scatter.png`
- **Description**: Scatter plot of Spearman ρ for 25,876 matched TF-target edges between human and mouse lung.

### Figure 8: Per-TF Conservation Variability
- **Script**: `06_ortholog_transfer/04_cross_species_analysis.py`
- **Output**: `figures/fig_ortholog_per_tf.png`
- **Description**: Spearman ρ for each TF's target set, showing lineage-specifying TFs transfer well while signaling-responsive TFs do not.

### Figure 9: Pseudotime Directionality Failure
- **Script**: `07_pseudotime/03_figures.py`
- **Output**: `figures/fig_pseudotime_failure.png`
- **Description**: (A) Overall directional consistency rate (21.4%). (B) Failure mode breakdown. (C) Peak lag vs. peak correlation for all 56 pairs.

### Figure 10: Pseudotime Directionality Null Models
- **Script**: `07_pseudotime/02b_directionality_analysis.py`
- **Output**: `figures/fig_pseudotime_nulls.png`
- **Description**: (A) Directionality score by lineage. (B) Observed mean vs. shuffled-pseudotime null. (C) Observed mean vs. random gene-pair null.

### Figure 11: Cross-Dataset Leakage Summary
- **Script**: `08_batch_leakage/02_full_pipeline.py`
- **Output**: `figures/fig_batch_leakage_summary.png`
- **Description**: Balanced accuracy and AUC for donor, batch, and method classification from edge-product features across tissues.

### Figure 12: Artifact Sensitivity Index Distribution
- **Script**: `08_batch_leakage/phase2_leakage_correction.py`
- **Output**: `figures/fig_batch_asi_distribution.png`
- **Description**: Immune tissue has 54.6% of edges exceeding ASI > 0.5, reflecting extreme donor imbalance.

### Figure 13: Edge Score Calibration
- **Script**: `09_calibration/02_calibration_analysis.py`
- **Output**: `figures/fig_calibration_reliability.png`
- **Description**: Reliability diagrams showing fraction of true positives vs. mean predicted probability. Left: raw scores are severely miscalibrated. Center/Right: Platt scaling and isotonic regression reduce ECE by 4-7×.

### Figure 14: Conformal Prediction Sets
- **Script**: `09_calibration/phase2_adaptive_conformal.py`
- **Output**: `figures/fig_calibration_conformal.png`
- **Description**: Empirical coverage and average set size across methods at α = 0.05, 0.10, 0.20.

### Figure 15: CSSI Mitigates Scaling Failure
- **Script**: `10_cssi/05_figures.py`
- **Output**: `figures/fig_cssi_scaling.png`
- **Description**: F1 score as a function of cell-state heterogeneity. Pooled inference degrades monotonically with increasing heterogeneity, while CSSI-max maintains near-perfect recovery.

### Figure 16: Synthetic Validation Summary
- **Script**: `11_synthetic_validation/synthetic_validation.py`
- **Output**: `figures/synthetic_validation_summary.png`
- **Description**: (A) Attention-based GRN recovery degrades with cell count. (B) Shapley values outperform single-component estimates by 91%. (C-D) Empirical detectability matches theoretical predictions.

### Figure 17: Multi-Model GRN Validation
- **Script**: `12_multi_model/geneformer_grn_validation.py`
- **Output**: `figures/geneformer_analysis_summary.png`
- **Description**: Both scGPT and Geneformer achieve near-random AUROC (≈0.5) for attention-based GRN inference despite different scaling dynamics.

## Main Paper Tables

### Table 1: Seed Robustness Degradation
- **Script**: `01_scaling_analysis/generate_extended_evidence.py`
- **Data**: Generated within script, displayed in paper
- **Description**: Pairwise edge-set Jaccard and union-score Spearman rank correlation for 200→1000 cell transitions across model tiers.

### Table 2: Top-k Overlap Between Human and Mouse
- **Script**: `06_ortholog_transfer/04_cross_species_analysis.py`  
- **Data**: Generated within script, displayed in paper
- **Description**: Observed vs expected overlap for top-100, 500, 1000, 5000 edges with fold enrichment.

### Table 3: CSSI Performance on Biologically Structured Data
- **Script**: `10_cssi/cssi_realdata_validation.py`
- **Data**: Tabula Sapiens immune cell-type proportions (15 types, 15 cell-type-specific edges)
- **Description**: F1 and AUROC for pooled vs CSSI-max across sample sizes from N=200 to N=5,000.

### Table 4: Geneformer vs scGPT Cross-Model AUROC Comparison
- **Script**: `12_multi_model/geneformer_grn_validation.py`
- **Data**: Generated during Geneformer validation
- **Description**: TRRUST and DoRothEA AUROC values at 200, 500, 1000 cells for direct architecture comparison.

## Extended/Supplementary Figures

Many additional figures are generated in the `figures/` directory with the `fig_ext_` prefix. These include:

- `fig_ext_multiseed_robustness.png` - Extended robustness analysis
- `fig_ext_multiseed_scaling_dorothea_ci.png` - DoRothEA scaling validation  
- `fig_batch_lodo_stability.png` - Leave-one-donor-out stability analysis
- `fig_matched_real_data_bootstrap.png` - Bootstrap validation for detectability

## Running Specific Figure Generation

To generate individual figures:

```bash
# Figure 1 (Scaling failure)
cd 01_scaling_analysis/
python generate_extended_evidence.py

# Figure 15 (CSSI)  
cd 10_cssi/
python 05_figures.py

# Figure 17 (Multi-model)
cd 12_multi_model/
python geneformer_grn_validation.py
```

## Figure Dependencies

Most figures require:
- Pre-computed analysis results (JSON/CSV files)
- Reference databases (TRRUST, DoRothEA)
- matplotlib/seaborn for plotting
- Some require model checkpoints for fresh computation

## Runtime Estimates

| Figure | Estimated Runtime | Memory Requirement |
|--------|------------------|-------------------|
| 1-2 | 45 minutes | 8-16GB |
| 3-4 | 3 hours | 16-32GB |
| 5 | 45 minutes | 8GB |
| 6 | 20 minutes | 4GB |
| 7-8 | 55 minutes | 8GB |
| 9-10 | 75 minutes | 8GB |
| 11-12 | 90 minutes | 16GB |
| 13-14 | 75 minutes | 8GB |
| 15 | 20 minutes | 4GB |
| 16 | 60 minutes | 8GB |
| 17 | 2 hours | 16-32GB |

**Total estimated runtime**: 8-10 hours for complete reproduction