# Analysis A: Value-Weighted Edge Extraction — Results Summary

## Overview
Extracted value-weighted context layers (`softmax(QK^T/sqrt(d)) * V`) from Geneformer V2-316M across 18 layers using 2,000 K562 control cells. Computed pairwise cosine similarity (per-cell average and centroid), dot product, and L2-based edges. Evaluated against perturbation-first (Replogle CRISPRi) and TRRUST benchmarks.

## Key Result: Clear Negative — Value-Weighted Edges Underperform

| Metric | VW Cosine (best) | Raw Attention (best) | Correlation | Variance Baseline |
|--------|-----------------|---------------------|-------------|-------------------|
| Pert-first AUROC | 0.587 (L12) | **0.787 (L14)** | 0.706 | **0.887** |
| TRRUST AUROC | 0.606 (L12) | **0.718 (L15)** | 0.638 | — |

### Statistical Tests (Perturbation-First, Paired Wilcoxon)
- **VW cos (L12) vs Raw attn (L14)**: Δ = -0.200, p = 4.94e-39, d = -1.68
- **VW cos (L12) vs Correlation**: Δ = -0.120, p = 2.35e-14, d = -1.01
- **Raw attn (L14) vs Correlation**: Δ = +0.081, p = 5.02e-11

Value-weighted cosine similarity performs **significantly worse** than both raw attention and correlation, with large effect sizes (d > 1.0).

### Incremental Value Test (5-fold GroupKFold)
| Model | AUROC | Δ vs Gene-Only |
|-------|-------|---------------|
| Gene-only | 0.8654 | — |
| Gene + VW cosine | 0.8561 | **-0.0093** |
| Gene + Raw attn | 0.8612 | -0.0042 |
| Gene + Correlation | 0.8617 | -0.0037 |
| VW-only | 0.5866 | — |
| Attn-only | 0.7073 | — |
| Corr-only | 0.6367 | — |

No edge type provides incremental value. VW edges slightly *hurt* performance.

### Layer Profiles (Perturbation-First AUROC)
| Layer | VW Cosine | VW Centroid | VW Dot | Raw Attn |
|-------|-----------|------------|--------|----------|
| L0 | 0.335 | — | — | 0.475 |
| L4 | 0.454 | — | — | 0.746 |
| L8 | 0.373 | — | — | 0.702 |
| L12 | **0.587** | — | — | 0.741 |
| L14 | 0.409 | — | — | **0.787** |
| L15 | 0.355 | — | — | 0.773 |
| L17 | 0.365 | — | — | 0.746 |
| **Corr** | — | — | — | **0.706** |

### TRRUST AUROC by Layer
| Layer | VW Cosine | VW Centroid | Raw Attn |
|-------|-----------|------------|----------|
| L0 | 0.487 | 0.444 | 0.552 |
| L5 | 0.544 | 0.623 | 0.647 |
| L12 | **0.606** | 0.566 | 0.640 |
| L15 | 0.489 | 0.526 | **0.718** |
| **Corr** | — | — | **0.638** |

## Interpretation

**Why do value-weighted edges fail?**

The context layer `attn @ V` mixes information from all attended genes. Cosine similarity between context vectors measures "information-receipt similarity" — whether two genes receive similar blends of information from the attention mechanism. This is fundamentally different from the direct attention pattern (`softmax(QK^T/sqrt(d))`), which captures pairwise gene-to-gene relationships.

For perturbation prediction, what matters is pairwise regulatory coupling (does perturbing gene A affect gene B?). The raw attention pattern, despite capturing co-expression rather than causation, at least preserves this pairwise structure. Value-weighted cosine collapses pairwise information into a global representation similarity metric that loses the edge-specific signal.

**This negative result narrows the search space**: regulatory signal does not reside in value-pathway information routing as measured by representation similarity. Combined with the paper's finding that ablating TRRUST-ranked heads has no effect while random ablation does (d=0.33), this suggests the perturbation-predictive computation is distributed across heads in a way that is not captured by any simple attention-derived edge score.

## Total Runtime
75.4 minutes (extraction: ~73 min, evaluation: ~2.5 min)
