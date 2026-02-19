# CSSI Cross-Dataset Validation Experiments

This directory contains scripts to address the **circularity problem** in the NMI paper's CSSI real-data validation, where layer selection and evaluation used the same dataset.

## The Problem

The original NMI paper's CSSI validation had a methodological flaw:
1. Used full dataset to identify "best" layers via CSSI
2. Used same dataset to evaluate those layers
3. **Result:** Circular validation that could overestimate performance

## Our Solution

**Proper cross-dataset validation:**
1. **Split dataset** into non-overlapping subsets A and B (~250 cells each)
2. **Layer selection** on Subset A using CSSI vs TRRUST
3. **Evaluation** on held-out Subset B
4. **Baseline comparison** with random layer selection on Subset B

**Bonus cross-tissue validation:**
- Identify best layers on **immune tissue**
- Evaluate on **kidney tissue** using whole-human checkpoint
- Tests biological generalizability across tissue types

## Quick Start

### Option 1: Run Both Experiments
```bash
# Run both cross-dataset and cross-tissue validation
wsl -u agent -- bash -lc "cd /mnt/d/openclaw/biodyn-nmi-paper/cssi_real && python3 run_cssi_validation.py --both"
```

### Option 2: Run Individual Experiments
```bash
# Cross-dataset validation only
wsl -u agent -- bash -lc "cd /mnt/d/openclaw/biodyn-nmi-paper/cssi_real && python3 cssi_cross_dataset_validation.py"

# Cross-tissue validation only  
wsl -u agent -- bash -lc "cd /mnt/d/openclaw/biodyn-nmi-paper/cssi_real && python3 cssi_cross_tissue_validation.py"
```

## Expected Runtime

- **Cross-dataset validation:** ~15-30 minutes
- **Cross-tissue validation:** ~20-30 minutes
- **Total:** ~45-60 minutes

*Runtime depends on GPU availability and dataset sizes*

## Output Files

### Reports (Main Deliverables)
- `D:/openclaw/biodyn-nmi-paper/CSSI_CROSS_VALIDATION_REPORT.md` - Main validation report
- `D:/openclaw/biodyn-nmi-paper/CSSI_CROSS_TISSUE_REPORT_immune_to_kidney.md` - Cross-tissue report

### Raw Results
- `cssi_validation_results.json` - Cross-dataset results (JSON)
- `cssi_cross_tissue_results_immune_to_kidney.json` - Cross-tissue results (JSON)

## Interpretation

### ✅ SUCCESS CRITERIA
- **Cross-dataset:** Selected layers perform better than random on held-out subset B
- **Cross-tissue:** Layers from immune tissue transfer to kidney tissue
- **Statistical significance:** Improvement > 1-2 standard deviations

### ❌ FAILURE MODES
- **Overfitting:** Layers identified on subset A don't generalize to subset B
- **Tissue specificity:** Immune-selected layers don't work on kidney
- **Insufficient data:** Sample sizes too small for robust layer identification

## Technical Details

### Hardware Requirements
- **GPU:** Recommended (RTX 2060 6GB available)
- **Memory:** ~8GB RAM minimum
- **Storage:** ~1GB for datasets and outputs

### Software Dependencies
- **Python 3.8+** (in WSL)
- **PyTorch** with CUDA support
- **scanpy, pandas, numpy, scikit-learn**
- **scGPT** (with torchtext shim patch)

### Key Configuration
- **Batch size:** 2 (optimized for 6GB GPU)
- **Cell subsets:** 250 cells each (A and B)
- **Top-k layers:** 3 best performing layers
- **Random trials:** 10 for baseline comparison
- **Random seed:** 42 (reproducibility)

## File Structure

```
cssi_real/
├── README.md                           # This file
├── cssi_cross_dataset_validation.py    # Main cross-dataset validation
├── cssi_cross_tissue_validation.py     # Bonus cross-tissue validation  
├── run_cssi_validation.py             # Simple runner script
├── cssi_validation_results.json       # Output: cross-dataset results
└── cssi_cross_tissue_results_*.json   # Output: cross-tissue results
```

## Datasets Used

- **Cross-dataset:** PBMC3K or Tabula Sapiens Immune (as brain substitute)
- **Cross-tissue:** Tabula Sapiens Immune → Kidney
- **Ground truth:** TRRUST human regulatory network
- **Models:** scGPT brain checkpoint (cross-dataset), whole-human (cross-tissue)

## Validation Logic

### Cross-Dataset Validation
1. Load dataset (PBMC3K or immune)
2. Random split into subsets A (250) and B (250)
3. Extract attention matrices from brain checkpoint
4. CSSI on subset A → identify top 3 layers
5. Evaluate same layers on subset B
6. Compare with random layer selection baseline
7. **Pass criterion:** Selected layers > random baseline

### Cross-Tissue Validation
1. Load immune dataset (source) and kidney dataset (target)
2. Extract attention matrices using whole-human checkpoint
3. CSSI on immune → identify top 3 layers
4. Evaluate same layers on kidney
5. Compare with random layer selection on kidney
6. **Pass criterion:** Cross-tissue transfer > random baseline

## Expected Outcomes

### Scenario 1: Both Pass ✅
- **Interpretation:** CSSI identifies robust, biologically meaningful layers
- **Implication:** Original NMI results are validated despite circularity
- **Next steps:** Broader validation across datasets/tissues

### Scenario 2: Cross-Dataset Pass, Cross-Tissue Fail ✅❌  
- **Interpretation:** CSSI works within datasets but not across tissues
- **Implication:** Attention patterns are tissue-specific
- **Next steps:** Tissue-conditional analysis

### Scenario 3: Both Fail ❌❌
- **Interpretation:** CSSI may have been overfitted in original paper
- **Implication:** Need larger datasets or different layer selection methods
- **Next steps:** Investigate sample size requirements

## Troubleshooting

### Common Issues

**GPU Memory Error:**
```bash
# Reduce batch size
python3 cssi_cross_dataset_validation.py --batch-size 1
```

**Dataset Not Found:**
```bash
# Specify custom dataset path
python3 cssi_cross_dataset_validation.py --data-path /path/to/dataset.h5ad
```

**scGPT Import Error:**
```bash
# Verify torchtext patch is working
python3 -c "exec(open('/mnt/c/Users/Agent/.openclaw/workspace/patch_torchtext.py').read()); import scgpt"
```

### Debug Mode
```bash
# Run with verbose logging
python3 cssi_cross_dataset_validation.py --batch-size 1 2>&1 | tee debug.log
```

## Research Impact

This validation experiment directly addresses a **critical methodological concern** in the mechanistic interpretability literature. Proper cross-validation is essential for:

1. **Reproducibility** - Results hold on new data
2. **Generalizability** - Insights transfer across contexts  
3. **Scientific rigor** - Avoid circular reasoning
4. **Clinical translation** - Robust biomarker discovery

The cross-tissue validation represents the **gold standard** for testing biological generalizability of transformer-based gene regulatory network inference.

---

**Authors:** CSSI Validation Team  
**Date:** February 2026  
**Contact:** See main biodyn-nmi-paper repository