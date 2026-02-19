# Analysis E: Per-TF Characterization — Hardened Results

## Methodological Improvements

1. **Permutation-based p-values** (10,000 label shuffles) replace asymptotic Mann-Whitney
2. **BH-FDR correction** across all tests
3. **Single-target TFs flagged and excluded** in robustness checks

## Evaluable Target Counts

| TF | Targets in HVG | Eval. Regulon | AUROC | Category | Note |
|----|---------------|---------------|-------|----------|------|
| GATA1 | 18 | 5 | 0.842 | master_regulator |  |
| PPARG | 10 | 8 | 0.900 | master_regulator |  |
| EGR1 | 9 | 4 | 0.852 | signal-dependent |  |
| WT1 | 5 | 7 | 0.675 | master_regulator |  |
| FOXF2 | 3 | 2 | 0.500 | lineage-specific |  |
| NKX2-5 | 2 | 1 | 0.572 | master_regulator |  |
| FOXA1 | 2 | 2 | 0.874 | master_regulator |  |
| RXRA | 2 | 0 | 0.359 | signal-dependent |  |
| KLF1 | 1 | 0 | 0.995 | master_regulator | SINGLE TARGET |
| MAL | 1 | 0 | 0.860 | signal-dependent | SINGLE TARGET |
| NCOA4 | 1 | 0 | 0.349 | signal-dependent | SINGLE TARGET |
| PAX3 | 1 | 2 | 0.802 | master_regulator | SINGLE TARGET |
| KLF3 | 1 | 0 | 0.644 | lineage-specific | SINGLE TARGET |
| YBX1 | 1 | 8 | 0.640 | housekeeping | SINGLE TARGET |
| PA2G4 | 1 | 0 | 0.630 | housekeeping | SINGLE TARGET |
| TAL1 | 1 | 0 | 0.556 | master_regulator | SINGLE TARGET |
| TBX5 | 1 | 0 | 0.401 | lineage-specific | SINGLE TARGET |
| LEF1 | 1 | 6 | 0.997 | master_regulator | SINGLE TARGET |

**13 of 18 TFs have ≤1 evaluable target** — their AUROC is the rank of a single gene
among ~1999, not a regulon-level assessment.

## Permutation Tests

### All 18 TFs
- Master mean AUROC: 0.8015
- Other mean AUROC: 0.5817
- Permutation p (one-sided, 10,000 shuffles): **0.0111**

### Excluding single-target TFs (≥2 targets, n=8)
- Excluded: NCOA4, TBX5, TAL1, PA2G4, YBX1, KLF3, PAX3, MAL, KLF1, LEF1
- Permutation p: **0.0910**
- Observed Δ: +0.2021

### Robust TFs only (≥3 targets, n=5)
- TFs: FOXF2, WT1, GATA1, EGR1, PPARG
- Permutation p: **0.2968**
- Observed Δ: +0.1296

## All Tests with BH Correction

**11 tests**, 0 significant after BH-FDR at α=0.05.

| Test | raw p | BH q | Significant |
|----|-------|------|------------|
| Master regulators vs other TFs (all 18) | 0.0111 | 0.1221 | No |
| Master vs other (≥3 evaluable targets only) | 0.2968 | 0.5262 | No |
| Master vs other (≥2 targets only) | 0.0910 | 0.3336 | No |
| Fisher exact: master regulator × successful | 0.3348 | 0.5262 | No |
| Spearman: AUROC vs full regulon size | 0.3293 | 0.5262 | No |
| Spearman: AUROC vs evaluable regulon size | 0.0613 | 0.3336 | No |
| Spearman: auroc_vs_mean_expr | 0.5985 | 0.6583 | No |
| Spearman: auroc_vs_variance | 0.4233 | 0.5820 | No |
| Spearman: auroc_vs_nonzero_frac | 0.9708 | 0.9708 | No |
| Mann-Whitney: expression in successful vs unsuccessful | 0.5962 | 0.6583 | No |
| Mann-Whitney: regulon size in successful vs unsuccessful | 0.1736 | 0.4775 | No |

## Interpretation

The master regulator advantage is evaluated at three stringency levels:
1. All 18 TFs (including single-target): tests the broad pattern
2. ≥2 targets: excludes TFs where AUROC = rank of 1 gene
3. ≥3 targets: most conservative, only regulon-level assessments
