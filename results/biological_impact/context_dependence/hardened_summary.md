# Analysis D: Context-Dependence — Hardened Results

## 1. HVG Confound Test

**Question:** Does the RPE1 attention advantage survive when restricted to top-2000 HVGs
(matching K562 protocol, no forced perturbation gene inclusion)?

| Metric | Original (3309 HVGs) | Restricted (2000 HVGs) |
|--------|---------------------|----------------------|
| n evaluable perturbations | 418 | 418 |
| Mean attention AUROC | 0.6989 | 0.7616 |
| Mean correlation AUROC | 0.7225 | 0.5933 |
| Mean Δ (attn − corr) | -0.0236 | +0.1682 |
| Fraction attn wins | 0.488 | 0.890 |
| Perturbations dropped | 0 | 833 |

Paired Wilcoxon (delta change): p = 1.997e-46

**Restricting to 2000 HVGs changes attention AUROC by +0.0627, correlation AUROC by -0.1292, and delta by +0.1918**


## 2. Bootstrap Confidence Intervals (10,000 samples)

| Dataset | n | Mean Δ | 95% CI (mean) | Median Δ | 95% CI (median) | Wilcoxon p |
|---------|---|--------|--------------|----------|----------------|------------|
| RPE1_CRISPRi | 1167 | +0.0896 | [+0.0787, +0.1006] | +0.1054 | [+0.0869, +0.1218] | 2.211e-54 |
| K562_CRISPRi | 280 | +0.0603 | [+0.0403, +0.0804] | +0.0424 | [+0.0310, +0.0685] | 8.864e-08 |
| RPE1_restricted_2000 | 418 | +0.1682 | [+0.1547, +0.1819] | +0.1603 | [+0.1373, +0.1828] | 4.894e-60 |

