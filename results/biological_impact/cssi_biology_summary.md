# Analysis C: CSSI-Derived Cell-State-Specific Regulatory Programs — Results

## 1. Stratification

Leiden clustering on Tabula Sapiens immune data (20,000 cells, 2000 HVGs):
Resolution ground_truth → 7 strata.

| Stratum | n cells | Assigned Type | Marker Score |
|---------|---------|---------------|-------------|
| B_cell | 4623 | B cell | 1.2412 |
| Erythrocyte | 506 | Erythrocyte | 1.0389 |
| Monocyte | 3848 | Monocyte | 1.4748 |
| NK_cell | 467 | NK cell | 1.6803 |
| Neutrophil | 2455 | Monocyte | 1.5134 |
| Other | 2179 | T cell | 0.4350 |
| T_cell | 5920 | T cell | 0.8657 |

## 2. TRRUST AUROC by Stratum

| Stratum | Cell Type | AUROC | n_cells |
|---------|-----------|-------|---------|
| B_cell | B cell | 0.7342 | 4623 |
| Erythrocyte | Erythrocyte | 0.7751 | 506 |
| Monocyte | Monocyte | 0.7888 | 3848 |
| NK_cell | NK cell | 0.7282 | 467 |
| Neutrophil | Monocyte | 0.7164 | 2455 |
| Other | T cell | 0.7729 | 2179 |
| T_cell | T cell | 0.7054 | 5920 |
| **cssi_max** | — | 0.8352 | — |
| **cssi_mean** | — | 0.8423 | — |
| **pooled** | — | 0.8401 | 19998 |

## 3. Cell-Type-Specific Enrichment

### Stratum B_cell (B cell)

- T cell: 1/2 TFs, OR=0.12, p=0.9842
- B cell **(MATCH)**: 0/2 TFs, OR=0.00, p=1.0000
- Monocyte: 1/1 TFs, OR=inf, p=0.8500
- Dendritic cell: 0/1 TFs, OR=0.00, p=1.0000

### Stratum Erythrocyte (Erythrocyte)

- T cell: 1/2 TFs, OR=0.20, p=0.9684
- B cell: 1/2 TFs, OR=0.20, p=0.9684
- Monocyte: 1/1 TFs, OR=inf, p=0.8000
- Dendritic cell: 0/1 TFs, OR=0.00, p=1.0000

### Stratum Monocyte (Monocyte)

- T cell: 1/2 TFs, OR=0.00, p=1.0000
- B cell: 1/2 TFs, OR=0.00, p=1.0000
- Monocyte **(MATCH)**: 1/1 TFs
- Dendritic cell: 0/1 TFs, OR=0.00, p=1.0000

### Stratum NK_cell (NK cell)

- T cell: 1/2 TFs, OR=0.20, p=0.9684
- B cell: 1/2 TFs, OR=0.20, p=0.9684
- Monocyte: 1/1 TFs, OR=inf, p=0.8000
- Dendritic cell: 0/1 TFs, OR=0.00, p=1.0000

### Stratum Neutrophil (Monocyte)

- T cell: 1/2 TFs, OR=0.38, p=0.9211
- B cell: 1/2 TFs, OR=0.38, p=0.9211
- Monocyte **(MATCH)**: 1/1 TFs, OR=inf, p=0.7000
- Dendritic cell: 0/1 TFs, OR=0.00, p=1.0000

### Stratum Other (T cell)

- T cell **(MATCH)**: 1/2 TFs, OR=0.20, p=0.9684
- B cell: 0/2 TFs, OR=0.00, p=1.0000
- Monocyte: 1/1 TFs, OR=inf, p=0.8000
- Dendritic cell: 0/1 TFs, OR=0.00, p=1.0000

### Stratum T_cell (T cell)

- T cell **(MATCH)**: 1/2 TFs, OR=0.12, p=0.9842
- B cell: 0/2 TFs, OR=0.00, p=1.0000
- Monocyte: 1/1 TFs, OR=inf, p=0.8500
- Dendritic cell: 0/1 TFs, OR=0.00, p=1.0000

## 3b. Edge-Level Enrichment (More Powerful Approach)

For each stratum, test if edges from MATCHING cell-type TFs are ranked higher
than edges from NON-MATCHING cell-type TFs (Mann-Whitney, one-sided).

| Stratum | Cell Type | Matching TFs | Match Mean | Non-match Mean | p-value | r_rb |
|---------|-----------|-------------|-----------|----------------|---------|------|
| B_cell | B cell | EBF1, BCL6 | 0.0227 | 0.0145 | 0.000000 ** | -0.1395 |
| Monocyte | Monocyte | SPI1 | 0.0316 | 0.0259 | 0.000000 ** | -0.1872 |
| Neutrophil | Monocyte | SPI1 | 0.0296 | 0.0168 | 0.000000 ** | -0.2001 |
| Other | T cell | BCL11B, LEF1 | 0.0537 | 0.0404 | 0.000000 ** | -0.1140 |
| T_cell | T cell | BCL11B, LEF1 | 0.0201 | 0.0191 | 0.000000 ** | -0.2017 |

**All testable strata show significant enrichment of matching cell-type TF edges.**

## 4. Specificity (TF-level Fisher's exact)

- Stratum B_cell (B cell): not specific (matching_p=1.0000)
- Stratum Erythrocyte (Erythrocyte): not specific (matching_p=N/A)
- Stratum Monocyte (Monocyte): not specific (matching_p=1.0000)
- Stratum NK_cell (NK cell): not specific (matching_p=N/A)
- Stratum Neutrophil (Monocyte): not specific (matching_p=0.7000)
- Stratum Other (T cell): not specific (matching_p=0.9684)
- Stratum T_cell (T cell): not specific (matching_p=0.9842)

## 5. Differential Edge Analysis

109 TRRUST edges analyzed for cross-stratum variance.
2 edges have their TF expected in the best stratum's cell type.

Top 10 most variable edges:

| TF | Target | Variance | Range | Best Cell Type | Pooled Score |
|-----|--------|----------|-------|----------------|-------------|
| SPI1 | BPI | 0.0825 | 0.909 | NK cell | 0.120 |
| GATA2 | MS4A2 | 0.0631 | 0.640 | T cell | 0.633 |
| GATA2 | IL1RL1 | 0.0586 | 0.737 | T cell | 0.553 |
| SPI1 | HCK | 0.0584 | 0.644 | T cell | 0.703 |
| SPI1 | CD163 | 0.0581 | 0.680 | T cell | 0.516 |
| SPI1 | MNDA | 0.0492 | 0.626 | T cell | 0.677 |
| SPI1 | CSF3R | 0.0484 | 0.585 | T cell | 0.668 |
| KLF1 | HBB | 0.0476 | 0.687 | T cell | 0.482 |
| SPI1 | IL1B | 0.0363 | 0.474 | T cell | 0.362 |
| SPI1 | NCF2 | 0.0339 | 0.574 | T cell | 0.679 |