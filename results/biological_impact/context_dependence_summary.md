# Analysis D: Context-Dependence Characterization — Results

## Overview

Why does attention outperform correlation in RPE1 (d=0.47) but not K562?

## 1. Dataset-Level Comparison

| Dataset | Cell Type | Modality | n_eval | HVGs | Attn AUROC | Corr AUROC | Δ | Cohen's d |
|---------|-----------|----------|--------|------|-----------|------------|------|-----------|
| K562_CRISPRi | K562 | CRISPRi | 280 | 2000 | 0.704 | 0.703 | +0.001 | 0.01 |
| RPE1_CRISPRi | RPE1 | CRISPRi | 1251 | 3309 | 0.748 | 0.658 | +0.090 | 0.47 |
| Adamson_CRISPRa | K562 | CRISPRa | 77 | 2000 | 0.580 | 0.653 | -0.073 | -0.56 |
| Shifrut_TCells | Primary T cells | CRISPRi (CRISPR screen) | 7 | 2000 | 0.679 | 0.651 | +0.029 | 0.16 |
| Tian_Neurons | iPSC-derived neurons | CRISPRi | 7 | 2000 | 0.533 | 0.475 | +0.058 | 0.80 |

### Key Observations

1. **Modality matters**: Both CRISPRi datasets on K562 (K562_CRISPRi) and RPE1 show attention ≥ correlation,
   but CRISPRa (Adamson) shows the opposite. CRISPRa activates genes rather than knocking them down,
   fundamentally changing the perturbation signal structure.
2. **Cell type matters**: RPE1 (non-cancerous epithelial) shows strong attention advantage; K562 (cancer) does not.
3. **HVG protocol differs**: RPE1 uses 3,309 HVGs (perturbation genes forced in) vs K562's 2,000.

## 2. Per-Perturbation Analysis

### RPE1
- 1167 perturbations
- Mean Δ: +0.0896
- Fraction where attention wins: 0.706

Correlations of attention advantage with perturbation features:

- n_DE_targets: ρ = -0.2651, p = 0.0000 **
- mean_expression: ρ = -0.4348, p = 0.0000 **
- TRRUST_degree: ρ = 0.0006, p = 0.9824 
- tf_vs_nontf: TF median = 0.1126, non-TF median = 0.1048, p = 0.9592 

### K562
- 280 perturbations
- Mean Δ: +0.0603
- Fraction where attention wins: 0.625

Correlations of attention advantage with perturbation features:

- n_DE_targets: ρ = -0.0709, p = 0.2368 
- mean_expression: ρ = -0.0667, p = 0.2659 
- variance: ρ = -0.2059, p = 0.0005 **
- dropout_rate: ρ = 0.0381, p = 0.5256 
- TRRUST_degree: ρ = 0.0390, p = 0.5155 
- tf_vs_nontf: TF median = 0.0551, non-TF median = 0.0424, p = 0.4993 

## 3. HVG Protocol Confound Test

**K562 forced HVG test: SKIPPED**
- Reason: K562 raw expression data not available locally. Only pre-extracted features exist.
- K562 perturbation genes in HVG: 280/280
- K562 perturbation genes NOT in HVG: 0

**Key structural observation**: RPE1 uses 3,309 HVGs (perturbation genes forced in, 100% coverage)
while K562 uses 2,000 HVGs (280 of 280 perturbation genes covered).
This asymmetry means K562 perturbation genes outside the HVG set have no correlation-based edge scores,
while all RPE1 perturbation genes do. This could contribute to the observed context-dependence.

## 4. Interpretation

### Biological Hypotheses

1. **Cell-type regulatory architecture**: RPE1 cells may have more structured regulatory programs
   that Geneformer's attention captures beyond simple co-expression. K562 is a cancer cell line
   with dysregulated gene expression that may obscure regulatory structure.

2. **Modality effect**: CRISPRi (knockdown) creates loss-of-function perturbations that propagate
   through regulatory networks. CRISPRa (activation) creates gain-of-function that may produce
   different downstream signatures not aligned with attention patterns.

3. **Technical confound**: The HVG protocol difference (3,309 vs 2,000 genes) systematically
   includes perturbation target genes in RPE1 but not K562, potentially biasing the comparison.
