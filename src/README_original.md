# NMI Paper: Mechanistic Interpretability of Single-Cell Foundation Models

This repository contains all code and analysis scripts for reproducing the results in:

**"A Comprehensive Framework for Mechanistic Interpretability of Single-Cell Foundation Models: Scaling, Bias, Detectability, Cross-Context Validation, and Quality Control"**

## Overview

The paper presents a comprehensive framework addressing 12 critical challenges in mechanistic interpretability of single-cell foundation models, including scaling behavior, mediation bias, detectability theory, cross-context validation, perturbation consistency, cross-species transfer, pseudotime directionality, batch leakage, uncertainty calibration, synthetic ground-truth validation, multi-model generalization, and a novel method (CSSI).

## Repository Structure

```
code/
├── README.md                   # This file
├── requirements.txt           # Python dependencies
├── environment.yml           # Conda environment specification
├── 01_scaling_analysis/      # Scaling behavior analysis (Section 3.1, Figure 1-2)
├── 02_mediation_bias/        # Mediation bias analysis (Section 3.2, Figure 3-4)
├── 03_detectability/         # Detectability theory (Section 3.3, Figure 5)
├── 04_cross_tissue/          # Cross-context consistency (Section 3.4, Figure 6)
├── 05_perturbation/          # Perturbation validation (Section 3.5)
├── 06_ortholog_transfer/     # Cross-species ortholog transfer (Section 3.6, Figure 7-8)
├── 07_pseudotime/            # Pseudotime directionality audit (Section 3.7, Figure 9-10)
├── 08_batch_leakage/         # Batch and donor leakage audit (Section 3.8, Figure 11-12)
├── 09_calibration/           # Uncertainty calibration (Section 3.9, Figure 13-14)
├── 10_cssi/                  # Cell-State Stratified Interpretability (Section 3.10, Figure 15)
├── 11_synthetic_validation/  # Synthetic ground-truth validation (Section 3.11, Figure 16)
├── 12_multi_model/          # Multi-model generalization (Section 3.12, Figure 17)
└── figures/                 # Generated figures and plots
```

## Setup Instructions

### Option 1: Using Conda (Recommended)

```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate nmi-mechanistic-interpretability
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv nmi-env
source nmi-env/bin/activate  # On Windows: nmi-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Additional Setup Notes

1. **Geneformer Installation**: The Geneformer model requires special installation. Follow instructions at: https://github.com/ctheodoris/Geneformer

2. **Data Requirements**: Most scripts expect access to:
   - Tabula Sapiens dataset
   - TRRUST/DoRothEA reference databases
   - Perturb-seq datasets (Dixit, Adamson, Shifrut)
   - Pre-trained scGPT and Geneformer models

3. **GPU Support**: Many analyses benefit from GPU acceleration. Ensure PyTorch is installed with CUDA support if available.

## Reproducing Figures and Tables

### Table of Contents: Figure/Table → Script Mapping

| Figure/Table | Description | Script Location | Runtime |
|--------------|-------------|-----------------|---------|
| Figure 1 | Scaling failure across model tiers | `01_scaling_analysis/generate_extended_evidence.py` | 30 min |
| Figure 2 | Seed robustness degradation | `01_scaling_analysis/generate_submission_graphs.py` | 15 min |
| Figure 3 | Non-additivity in mediation | `02_mediation_bias/run_full_paper_pipeline.py` | 2 hours |
| Figure 4 | Ranking certificate fragility | `02_mediation_bias/run_targeted_pairwise_patches.py` | 1 hour |
| Figure 5 | Detectability phase diagrams | `03_detectability/run_detectability_phase_diagram.py` | 45 min |
| Figure 6 | Cross-tissue consistency | `04_cross_tissue/make_paper_figures.py` | 20 min |
| Figure 7 | Ortholog transfer scatter | `06_ortholog_transfer/05_figures.py` | 30 min |
| Figure 8 | Per-TF conservation | `06_ortholog_transfer/04_cross_species_analysis.py` | 25 min |
| Figure 9 | Pseudotime directionality failure | `07_pseudotime/03_figures.py` | 40 min |
| Figure 10 | Pseudotime null models | `07_pseudotime/02b_directionality_analysis.py` | 35 min |
| Figure 11 | Batch leakage summary | `08_batch_leakage/02_full_pipeline.py` | 1 hour |
| Figure 12 | ASI distribution | `08_batch_leakage/phase2_leakage_correction.py` | 30 min |
| Figure 13 | Calibration reliability | `09_calibration/02_calibration_analysis.py` | 45 min |
| Figure 14 | Conformal prediction | `09_calibration/phase2_adaptive_conformal.py` | 30 min |
| Figure 15 | CSSI scaling mitigation | `10_cssi/05_figures.py` | 20 min |
| Figure 16 | Synthetic validation | `11_synthetic_validation/synthetic_validation.py` | 1 hour |
| Figure 17 | Multi-model comparison | `12_multi_model/geneformer_grn_validation.py` | 2 hours |
| Table 1 | Scaling failure statistics | `01_scaling_analysis/generate_extended_evidence.py` | (included) |
| Table 2 | Cross-species top-k overlap | `06_ortholog_transfer/04_cross_species_analysis.py` | (included) |
| Table 3 | CSSI performance | `10_cssi/cssi_realdata_validation.py` | 25 min |

### Running Individual Analyses

Each analysis directory contains numbered scripts that should generally be run in order:

#### 1. Scaling Analysis (Section 3.1)
```bash
cd 01_scaling_analysis/
python generate_extended_evidence.py
python generate_submission_graphs.py
```

#### 2. Mediation Bias Analysis (Section 3.2)
```bash
cd 02_mediation_bias/
python run_full_paper_pipeline.py
python run_targeted_pairwise_patches.py
```

#### 3. Detectability Theory (Section 3.3)
```bash
cd 03_detectability/
python run_detectability_phase_diagram.py
python run_matched_real_data_calibration.py
```

#### 4. Cross-Context Consistency (Section 3.4)
```bash
cd 04_cross_tissue/
python make_paper_figures.py
```

#### 5. Perturbation Validation (Section 3.5)
```bash
cd 05_perturbation/
python run_bridge_benchmark.py
python run_counterfactual_consistency_study.py
```

#### 6. Cross-Species Ortholog Transfer (Section 3.6)
```bash
cd 06_ortholog_transfer/
python 01_load_and_explore.py
python 02_ortholog_mapping.py
python 03_edge_scoring.py
python 04_cross_species_analysis.py
python 05_figures.py
```

#### 7. Pseudotime Directionality (Section 3.7)
```bash
cd 07_pseudotime/
python 02_pseudotime_and_analysis.py
python 03_figures.py
```

#### 8. Batch/Donor Leakage (Section 3.8)
```bash
cd 08_batch_leakage/
python 01_explore_metadata.py
python 02_full_pipeline.py
python phase2_leakage_correction.py
```

#### 9. Uncertainty Calibration (Section 3.9)
```bash
cd 09_calibration/
python 01_compute_edge_scores.py
python 02_calibration_analysis.py
python phase2_adaptive_conformal.py
```

#### 10. CSSI Method (Section 3.10)
```bash
cd 10_cssi/
python 02_full_pipeline.py
python 05_figures.py
python cssi_realdata_validation.py  # For Table 3
```

#### 11. Synthetic Validation (Section 3.11)
```bash
cd 11_synthetic_validation/
python synthetic_validation.py
```

#### 12. Multi-Model Analysis (Section 3.12)
```bash
cd 12_multi_model/
python geneformer_grn_validation.py
```

### Complete Reproduction Pipeline

To reproduce all results in order:

```bash
# Set up environment
conda env create -f environment.yml
conda activate nmi-mechanistic-interpretability

# Run all analyses (estimated total runtime: 8-10 hours)
for dir in {01..12}_*/; do
    echo "Running analysis in $dir"
    cd "$dir"
    # Run main script(s) - specific commands vary per directory
    cd ..
done
```

## Key Dependencies and Models

### Foundation Models
- **scGPT**: Pre-trained checkpoints for whole-human, brain, kidney tissues
- **Geneformer**: V1-10M model for multi-model validation
- **scVI**: For non-attention baseline comparisons

### Reference Databases
- **TRRUST v2**: Experimentally validated TF-target interactions
- **DoRothEA**: Confidence-scored TF activity estimates (grades A-D)
- **OmniPath**: Integrated regulatory network resource

### Key Datasets
- **Tabula Sapiens**: Multi-tissue human cell atlas (immune, lung, kidney subsets)
- **Perturb-seq Collections**: Dixit (13-day, 7-day), Adamson, Shifrut CRISPR screens
- **Krasnow Mouse Lung**: For cross-species ortholog transfer validation

## Expected Outputs

Each analysis directory will generate:
- **Figures**: PNG/PDF files for manuscript figures
- **Data**: CSV/JSON files with quantitative results
- **Logs**: Runtime information and intermediate outputs
- **Tables**: LaTeX/CSV formatted results tables

Key output files include:
- `figures/fig_*.png`: All manuscript figures
- `*_results.json`: Quantitative metrics for each analysis
- `*_summary.csv`: Tabulated results for statistical reporting

## Quality Control and Validation

The framework includes built-in quality control measures:

1. **Bootstrap Confidence Intervals**: 10,000 resamples for uncertainty quantification
2. **Multi-seed Validation**: Multiple random seeds for robustness testing
3. **Cross-validation**: Donor-stratified and tissue-stratified evaluation
4. **Synthetic Ground Truth**: Controlled validation with known networks
5. **Adversarial Testing**: Stress tests with parameter variations

## Troubleshooting

### Common Issues

1. **Memory Errors**: Some analyses require 16GB+ RAM. Reduce cell counts if needed.
2. **GPU Availability**: Scripts auto-detect GPU. Set `CUDA_VISIBLE_DEVICES=""` to force CPU.
3. **Missing Data**: Scripts will attempt to download public datasets automatically.
4. **Path Issues**: Update hard-coded paths in scripts to match your environment.

### Performance Notes

- **Parallel Processing**: Most scripts support multi-core execution
- **Checkpointing**: Long-running analyses save intermediate results
- **Memory Optimization**: Large datasets use memory-mapped arrays when possible

### Getting Help

For issues with:
- **Setup**: Check environment.yml and requirements.txt versions
- **Data**: Verify dataset paths and download completeness  
- **Models**: Ensure pre-trained model checkpoints are accessible
- **Results**: Compare with expected outputs in figures/ directory

## Citation

If you use this code, please cite:

```
@article{kendiukhov2026nmi,
  title={A Comprehensive Framework for Mechanistic Interpretability of Single-Cell Foundation Models: Scaling, Bias, Detectability, Cross-Context Validation, and Quality Control},
  author={Kendiukhov, Ihor},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This code is released under MIT License. See individual scripts for specific licensing information.

## Contact

For questions about the code or paper:
- Email: kendiukhov@gmail.com
- GitHub Issues: (to be provided upon publication)