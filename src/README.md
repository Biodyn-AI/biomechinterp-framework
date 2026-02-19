# Source Code

All analysis code for "Attention-Derived Gene Regulatory Networks in Single-Cell
Foundation Models Capture Co-Expression, Not Causal Regulation."

Folders are numbered to follow the paper's narrative flow. Each folder maps to
one or more sections in the main text and/or Supplementary Notes.

## Directory Map

| Folder | Paper Section | Supp Notes | Description |
|--------|--------------|------------|-------------|
| `01_scaling_failure/` | Main Results §1 | Note 1 | Scaling failure analysis across model tiers |
| `02_cssi_method/` | Main Results §1 | Notes 11-13 | Cell-State Stratified Interpretability (CSSI) |
| `03_perturbation_validation/` | Main Results §2 | Notes 2, 6 | Perturbation-first validation + trivial baselines |
| `04_confound_decomposition/` | Main Results §3 | Note 15 | Incremental value, residualization, propensity matching |
| `05_causal_ablation/` | Main Results §4 | Note 14 | Head ablation + orthogonal interventions |
| `06_cross_context/` | Main Results §5-6 | Note 14 | Cross-cell-type replication + RPE1 confound battery |
| `07_mediation_bias/` | — | Note 3 | Bias bounds for patching-based mediation |
| `08_detectability/` | — | Note 4 | Detectability phase diagrams |
| `09_ortholog_transfer/` | — | Note 7 | Cross-species ortholog transfer |
| `10_pseudotime/` | — | Note 8 | Pseudotime directionality audit |
| `11_batch_leakage/` | — | Note 9 | Batch and donor leakage audit |
| `12_calibration/` | — | Note 10 | Uncertainty calibration |
| `13_synthetic_validation/` | — | Note 12 | Synthetic ground-truth validation |
| `14_multi_model/` | — | Note 13 | Multi-model (scGPT + Geneformer) comparison |
| `15_biological/` | — | Notes 17-19 | Biological characterization of attention |
| `16_statistical_framework/` | — | Note 16 | Statistical test registry + multiplicity |
| `shared/` | — | — | Shared data extraction utilities |
| `plotting/` | — | — | Figure generation scripts |
| `_debug/` | — | — | Test and debug scripts (not part of analyses) |

## Experiment Results

Analysis outputs (JSON, CSV, NPY, PKL) are stored in `../results/`, organized
by experiment name and review round (round9 through round14).

## Environment

See `environment.yml` (conda) or `requirements.txt` (pip) for dependencies.
