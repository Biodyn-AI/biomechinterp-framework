# Systematic Evaluation of Single-Cell Foundation Model Interpretability Reveals Attention Captures Co-Expression Rather Than Unique Regulatory Signal

---

## Abstract

We present a systematic evaluation framework -- thirty-seven analyses, 153
statistical tests, four cell types, two perturbation modalities -- for assessing
mechanistic interpretability in single-cell foundation models. Applying this
framework to scGPT and Geneformer, we find that attention patterns encode
structured biological information with layer-specific organisation -- protein-
protein interactions in early layers, transcriptional regulation in late layers
-- but this structure provides no incremental value for perturbation prediction:
trivial gene-level baselines outperform both attention and correlation edges
(AUROC 0.81-0.88 versus 0.70), pairwise edge scores add zero predictive
contribution, and causal ablation of regulatory heads produces no degradation.
These findings generalise from K562 to RPE1 cells; the attention-correlation
relationship is context-dependent, but gene-level dominance is universal.
Cell-State Stratified Interpretability (CSSI) addresses an attention-specific
scaling failure, improving GRN recovery up to 1.85x. The framework establishes
reusable quality-control standards for the field.

---

## Repository Structure

```
biodyn-nmi-paper/
  paper/              LaTeX source for the manuscript
  src/                All analysis code (191 scripts, 16 topic modules)
  figures/            Paper figures (68 PNG + 18 PDF)
  results/            Experiment output data (32 result directories)
  docs/               Review documentation and reports
  archive/            Previous manuscript versions and original code repository
  prompts/            LLM prompt templates used during review process
  Makefile            Build automation for LaTeX compilation
```


### src/

All analysis code is organised into 16 numbered topic modules that follow the
paper's narrative flow. Each module maps to one or more sections of the main
text and/or Supplementary Notes. See `src/README.md` for a detailed directory
map.

| Module | Topic | Paper Reference |
|--------|-------|-----------------|
| `01_scaling_failure/` | Scaling degradation across model tiers | Main Results section 1; Supp Note 1 |
| `02_cssi_method/` | Cell-State Stratified Interpretability | Main Results section 1; Supp Notes 11-13 |
| `03_perturbation_validation/` | Perturbation-first validation and trivial baselines | Main Results section 2; Supp Notes 2, 6 |
| `04_confound_decomposition/` | Incremental value, residualization, propensity matching | Main Results section 3; Supp Note 15 |
| `05_causal_ablation/` | Head-mask ablation and orthogonal interventions | Main Results section 4; Supp Note 14 |
| `06_cross_context/` | Cross-cell-type replication and RPE1 confound battery | Main Results sections 5-6; Supp Note 14 |
| `07_mediation_bias/` | Bias bounds for patching-based mediation | Supp Note 3 |
| `08_detectability/` | Detectability phase diagrams | Supp Note 4 |
| `09_ortholog_transfer/` | Cross-species ortholog transfer | Supp Note 7 |
| `10_pseudotime/` | Pseudotime directionality audit | Supp Note 8 |
| `11_batch_leakage/` | Batch and donor leakage audit | Supp Note 9 |
| `12_calibration/` | Uncertainty calibration | Supp Note 10 |
| `13_synthetic_validation/` | Synthetic ground-truth validation | Supp Note 12 |
| `14_multi_model/` | Multi-model comparison (scGPT + Geneformer) | Supp Note 13 |
| `15_biological/` | Biological characterization of attention patterns | Supp Notes 17-19 |
| `16_statistical_framework/` | Statistical test registry and multiplicity analysis | Supp Note 16 |
| `shared/` | Shared data extraction utilities | -- |
| `plotting/` | Figure generation and formatting scripts | -- |
| `_debug/` | Test and debug scripts (not part of published analyses) | -- |

### results/

Pre-computed experiment outputs (JSON summaries, CSV tables, NPY arrays,
checkpoint files). Each subdirectory corresponds to one analysis and is named
by experiment type or review round:

- `round9_*` through `round14_*` -- Analyses added during successive review rounds
- `attention_perturbation_first_results/` -- Core perturbation-first AUROC results
- `controlled_composition_results/` -- Controlled-composition scaling experiment
- `biological_impact/` -- Biological characterization data (value-weighted edges, TF enrichment, context dependence)
- `round13_non_k562_replication/` -- RPE1 and Tian-Kampmann replication datasets (~2.8 GB including raw h5ad)

### figures/

All paper figures in PNG format, with PDF vector versions for select figures.
Figures are referenced by both `main_v4.tex` (6 main figures) and
`supplementary_v4.tex` (37 supplementary figures). Figure names follow the
convention `fig_<analysis_name>.png`.

### docs/

Review documentation accumulated during the peer review process:

- `docs/reviews/` -- 13 adversarial review rounds (systematic stress-testing)
- `docs/codex/` -- Review codex files (structured response tracking)
- `docs/rounds/` -- Round-specific review response documents
- `docs/reports/` -- Analysis validation reports (CSSI cross-validation, perturbation validation, scaling characterization, saturation analysis)

### archive/

- `archive/tex_versions/` -- Previous manuscript versions (v0 through v3)
- `archive/tex_artifacts/` -- Build artifacts for archived versions
- `archive/appendix/` -- Statistical test appendix (superseded by Supplementary Table 1)
- `archive/code_original/` -- Original analysis codebase with git history preserved

---

## Key Findings

1. **Attention-specific scaling failure.** Top-K F1 for attention-derived GRN
   recovery degrades unanimously across 9/9 model-seed combinations as cell
   count increases (permutation p = 0.002), while AUROC improves -- a metric-
   dependent dissociation driven by heterogeneity-induced attention dilution.

2. **Cell-State Stratified Interpretability (CSSI).** A formal dilution model
   motivates stratified scoring that recovers lost signal: CSSI-max improves
   GRN recovery up to 1.85x on DLPFC brain data, with no false-positive
   inflation under null stress tests.

3. **Gene-level features dominate.** Trivial baselines (expression variance,
   mean, dropout rate) achieve AUROC 0.81-0.88 for perturbation outcome
   prediction, exceeding both attention (0.70) and correlation (0.70) edges.

4. **Zero incremental pairwise value.** Adding pairwise attention or
   correlation edges to gene-level models produces AUROC change of -0.0004,
   confirmed across three split designs, two model families, and three metrics.

5. **Causal ablation shows distributed redundancy.** Masking up to 50
   TRRUST-ranked attention heads produces no degradation; orthogonal
   interventions (uniform attention replacement, MLP ablation) are also null.
   Intervention-fidelity diagnostics confirm all interventions materially
   perturb internal representations.

6. **Context-dependent attention-correlation relationship.** The direction of
   the attention-vs-correlation comparison varies by cell type (K562 CRISPRi:
   equal; CRISPRa: attention worse; RPE1: attention better), but the gene-level
   dominance pattern generalises universally.

---

## Models and Datasets

### Foundation Models

| Model | Architecture | Parameters | Layers x Heads |
|-------|-------------|------------|----------------|
| scGPT | GPT-style decoder | 3 tiers (small/medium/large) | varies |
| Geneformer V1-10M | BERT encoder | 10M | 6 x 4 |
| Geneformer V2-316M | BERT encoder | 316M | 18 x 18 |

### Perturbation Datasets

| Dataset | Cell Type | Modality | Perturbations | Source |
|---------|-----------|----------|---------------|--------|
| Replogle et al. 2022 | K562 | CRISPRi | 2,024 genes | scPerturb |
| Replogle et al. 2022 | RPE1 | CRISPRi | 1,251 genes | scPerturb |
| Adamson et al. 2016 | K562 | CRISPRa | 77 genes | scPerturb |
| Tian et al. 2021 | iPSC neurons | CRISPRi | 7 genes | scPerturb |
| Shifrut et al. 2018 | T cells | CRISPRi | 7 genes | scPerturb |

### Reference Networks

| Network | Type | Edges | Usage |
|---------|------|-------|-------|
| TRRUST v2 | Curated TF-target | ~9,000 | Primary evaluation reference |
| DoRothEA | Curated TF-target | ~37,000 | Multi-model validation |
| STRING | Protein-protein interaction | ~600,000 | Biological characterization |

### Tissue Transcriptomics

| Dataset | Tissues | Cells | Usage |
|---------|---------|-------|-------|
| Tabula Sapiens | Immune, kidney, lung | ~200,000 | Scaling analysis, cross-tissue |
| DLPFC (brain) | Prefrontal cortex | 497 | CSSI layer pre-specification |

---

## Statistical Framework

The paper comprises 37 distinct analyses generating 153 statistical tests:
- **95 confirmatory tests** with Benjamini-Hochberg correction at alpha = 0.05
- **58 descriptive analyses** (effect sizes, confidence intervals)
- **16 Supplementary Notes** providing full methodological detail

The complete test registry is in Supplementary Table 1 of the paper. The
multiplicity sensitivity analysis (`src/16_statistical_framework/`) confirms
that conclusions are stable across alternative family definitions.

---

## Environment Setup

### Option 1: Conda (recommended)

```bash
conda env create -f src/environment.yml
conda activate nmi-mechanistic-interpretability
```

### Option 2: pip

```bash
pip install -r src/requirements.txt
```

### Key Dependencies

- Python 3.8-3.10
- PyTorch >= 1.12 (with CUDA for GPU acceleration)
- Transformers >= 4.20 (for Geneformer model loading)
- scanpy >= 1.9, anndata >= 0.8 (single-cell data handling)
- scikit-learn >= 1.0, statsmodels >= 0.13 (statistical analysis)
- geneformer (Hugging Face: `ctheodoris/Geneformer`)

### External Data

Perturbation datasets are downloaded via the `scperturb` package. Geneformer
model weights are loaded from Hugging Face (`ctheodoris/Geneformer`, subfolder
`Geneformer-V2-316M`). TRRUST v2 is available at
https://www.grnpedia.org/trrust/.

---

## Building the Paper

Requires a TeX Live installation (tested with TeX Live 2024).

```bash
# Compile both main text and supplementary
make all

# Compile individually
make pdf    # -> paper/main_v4.pdf (13 pages)
make supp   # -> paper/supplementary_v4.pdf (58 pages)

# Clean build artifacts
make clean
```

---

## Reproducing Analyses

Each `src/` module is a self-contained analysis. Scripts are designed to:

1. Load data from external sources or pre-computed checkpoints in `results/`
2. Run the analysis with incremental saving (partial results checkpointed to JSON/CSV after each major computation block)
3. Save outputs to the corresponding `results/` subdirectory

Example -- running the perturbation-first validation:

```bash
cd src/03_perturbation_validation
python perturbation_first_validation.py
```

Example -- running the RPE1 confound battery:

```bash
cd src/06_cross_context
python round14_rpe1_confound_battery.py
```

Example -- regenerating paper figures:

```bash
cd src/plotting
python replot_main_figures.py
python replot_supp_figures.py
```

Most analysis scripts use line-buffered stdout for real-time progress monitoring
and include resume capability (checking for existing partial results before
recomputing).

---

## License

This repository accompanies a manuscript under review. Please contact the
author before reuse.

## Contact

Please open an issue in this repository for questions or correspondence.
