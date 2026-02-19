# Analysis C: CSSI Biology — Hardened Results (Unsupervised Leiden)

## 1. Clustering Evaluation

| Resolution | Clusters | ARI | NMI |
|-----------|----------|-----|-----|
| 0.3 | 17 | 0.590 | 0.690 |
| 0.5 | 22 | 0.528 | 0.679 |
| 0.7 | 26 | 0.467 | 0.665 |
| 1.0 | 30 | 0.426 | 0.649 |
| 1.5 | 40 | 0.373 | 0.638 |
| 2.0 | 44 | 0.300 | 0.619 |

## 2. CSSI AUROC by Resolution

| Resolution | K | Pooled | CSSI-max | CSSI-mean | CSSI-std | Δmax | Δmean |
|-----------|---|--------|---------|----------|---------|------|-------|
| 0.3 | 16 | 0.8280 | 0.8228 | 0.8374 | 0.8097 | -0.0052 | +0.0094 |
| 0.5 | 21 | 0.8280 | 0.8346 | 0.8485 | 0.8226 | +0.0066 | +0.0205 |
| 0.7 | 25 | 0.8280 | 0.8300 | 0.8442 | 0.8205 | +0.0020 | +0.0162 |
| 1.0 | 29 | 0.8280 | 0.8234 | 0.8442 | 0.8194 | -0.0046 | +0.0162 |
| 1.5 | 36 | 0.8280 | 0.8226 | 0.8420 | 0.8209 | -0.0054 | +0.0140 |
| 2.0 | 42 | 0.8280 | 0.8178 | 0.8316 | 0.8146 | -0.0102 | +0.0036 |

**Best resolution for CSSI-max: 0.5**


## 3. Ground-Truth Comparison

| Method | AUROC |
|--------|-------|
| Pooled (no stratification) | 0.8280 |
| CSSI-max (unsupervised, best res=0.5) | 0.8346 |
| CSSI-mean (unsupervised, best res=0.5) | 0.8485 |
| CSSI-max (ground-truth labels) | 0.8327 |
| CSSI-mean (ground-truth labels) | 0.8526 |

## 4. Cluster-to-Cell-Type Mapping (Best Resolution)

| Cluster | n cells | Dominant Type | Purity |
|---------|---------|--------------|--------|
| 0 | 2974 | CD4-positive, alpha-beta T cell | 0.73 |
| 1 | 2429 | CD8-positive, alpha-beta T cell | 0.68 |
| 10 | 495 | CD4-positive, alpha-beta T cell | 0.18 |
| 11 | 485 | neutrophil | 0.91 |
| 12 | 483 | erythrocyte | 0.93 |
| 13 | 417 | macrophage | 1.00 |
| 14 | 267 | mast cell | 0.76 |
| 15 | 176 | macrophage | 0.56 |
| 16 | 171 | CD8-positive, alpha-beta T cell | 0.26 |
| 17 | 165 | CD8-positive, alpha-beta T cell | 0.54 |
| 18 | 144 | macrophage | 0.71 |
| 19 | 140 | B cell | 0.99 |
| 2 | 2251 | macrophage | 0.68 |
| 20 | 61 | CD4-positive, alpha-beta T cell | 0.33 |
| 21 | 33 | CD8-positive, alpha-beta T cell | 0.52 |
| 3 | 2194 | B cell | 0.99 |
| 4 | 2006 | neutrophil | 0.97 |
| 5 | 1421 | B cell | 1.00 |
| 6 | 1216 | classical monocyte | 0.43 |
| 7 | 967 | CD4-positive, alpha-beta T cell | 0.46 |
| 8 | 826 | plasma cell | 1.00 |
| 9 | 679 | natural killer cell | 0.56 |

## 5. Interpretation

This analysis uses **unsupervised Leiden clustering** (not ground-truth labels).
