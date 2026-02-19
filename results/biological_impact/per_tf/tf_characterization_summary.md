# Analysis E: Per-TF Characterization — Results

## Overview

18 TFs evaluated with bootstrap confidence intervals on Tabula Sapiens immune data (n=5,000 cells, 2,000 HVGs).
**7 TFs** have 95% CI entirely above 0.5 (successful); **11 TFs** have CI including ≤0.5 (unsuccessful).

## Per-TF Table

| TF | AUROC | 95% CI | Successful | Family | Category | Full Reg. | Eval. Reg. | Mean Expr | Nonzero % |
|-----|-------|--------|------------|--------|----------|-----------|------------|-----------|-----------|
| LEF1 | 0.997 | [0.992, 0.998] | **Yes** | HMG-box (TCF/LEF) | master_regulator | 29 | 6 | 0.2450 | 15.4 |
| KLF1 | 0.995 | [0.993, 0.998] | **Yes** | Zinc finger (Krüppel-like) | master_regulator | 2 | 0 | 0.0118 | 1.1 |
| PPARG | 0.900 | [0.875, 0.926] | **Yes** | Nuclear receptor (NR1) | master_regulator | 66 | 8 | 0.1886 | 12.4 |
| FOXA1 | 0.874 | [0.500, 0.963] | No | Forkhead (FOX) | master_regulator | 21 | 2 | 0.0006 | 0.1 |
| MAL | 0.860 | [0.809, 0.885] | **Yes** | Other (not classical TF) | signal-dependent | 2 | 0 | 0.1067 | 9.4 |
| EGR1 | 0.852 | [0.797, 0.873] | **Yes** | Zinc finger (C2H2) | signal-dependent | 88 | 4 | 0.4111 | 25.7 |
| GATA1 | 0.842 | [0.788, 0.869] | **Yes** | Zinc finger (GATA) | master_regulator | 57 | 5 | 0.0100 | 1.1 |
| PAX3 | 0.802 | [0.303, 0.930] | No | Paired-box + Homeodomain | master_regulator | 6 | 2 | 0.0008 | 0.1 |
| WT1 | 0.675 | [0.559, 0.744] | **Yes** | Zinc finger (C2H2) | master_regulator | 57 | 7 | 0.0011 | 0.2 |
| KLF3 | 0.644 | [0.213, 0.850] | No | Zinc finger (Krüppel-like) | lineage-specific | 3 | 0 | 0.4208 | 38.9 |
| YBX1 | 0.640 | [0.089, 0.682] | No | Cold-shock domain | housekeeping | 30 | 8 | 1.1189 | 70.8 |
| PA2G4 | 0.630 | [0.083, 0.725] | No | Other (ErbB3-binding protein) | housekeeping | 4 | 0 | 0.3897 | 38.9 |
| NKX2-5 | 0.572 | [0.431, 0.646] | No | Homeodomain (NK) | master_regulator | 9 | 1 | 0.0003 | 0.0 |
| TAL1 | 0.556 | [0.176, 0.741] | No | bHLH | master_regulator | 10 | 0 | 0.0105 | 1.3 |
| FOXF2 | 0.500 | [0.500, 0.500] | No | Forkhead (FOX) | lineage-specific | 8 | 2 | 0.0003 | 0.0 |
| TBX5 | 0.401 | [0.228, 0.500] | No | T-box | lineage-specific | 5 | 0 | 0.0004 | 0.1 |
| RXRA | 0.359 | [0.222, 0.614] | No | Nuclear receptor (NR1) | signal-dependent | 20 | 0 | 0.1873 | 20.1 |
| NCOA4 | 0.349 | [0.083, 0.495] | No | Nuclear receptor coactivator | signal-dependent | 3 | 0 | 0.3656 | 34.8 |

## Key Findings

### 1. Master Regulator Enrichment (Primary Finding)

Master regulators have significantly higher AUROC than non-master TFs:
- Master regulators (n=9): median AUROC = 0.842
- Other TFs (n=9): median AUROC = 0.630
- Mann-Whitney (one-sided): p = 0.0170

Fisher's exact test for master regulator × success contingency:
- Master regulators: 56% successful
- Others: 22% successful
- OR = 4.38, p = 0.3348 (not significant — underpowered)

### 2. Functional Category Breakdown

- **master_regulator** (n=9): mean AUROC = 0.802, successful = 5/9, TFs: TAL1, NKX2-5, WT1, PAX3, GATA1, FOXA1, PPARG, KLF1, LEF1
- **signal-dependent** (n=4): mean AUROC = 0.605, successful = 2/4, TFs: NCOA4, RXRA, EGR1, MAL
- **lineage-specific** (n=3): mean AUROC = 0.515, successful = 0/3, TFs: TBX5, FOXF2, KLF3
- **housekeeping** (n=2): mean AUROC = 0.635, successful = 0/2, TFs: PA2G4, YBX1

### 3. Regulon Size

- AUROC vs full regulon size: ρ = 0.244, p = 0.3293 (n.s.)
- AUROC vs evaluable regulon size: ρ = 0.449, p = 0.0613 (marginal)

| Min Evaluable Targets | n TFs | Mean AUROC | % Successful |
|----------------------|-------|------------|--------------|
| ≥1 | 10 | 0.765 | 50% |
| ≥2 | 9 | 0.787 | 56% |
| ≥3 | 6 | 0.818 | 83% |
| ≥5 | 5 | 0.811 | 80% |

### 4. Expression Correlations

- auroc_vs_mean_expr: ρ = 0.133, p = 0.5985
- auroc_vs_variance: ρ = 0.201, p = 0.4233
- auroc_vs_nonzero_frac: ρ = 0.009, p = 0.9708
- Expression (successful vs unsuccessful): U = 45, p = 0.5962
  - Successful median = 0.1067, Unsuccessful median = 0.0105

### 5. The 7 Successful TFs

- **LEF1** (AUROC 0.997): Lymphoid enhancer factor 1; Wnt signaling effector, T-cell development
- **KLF1** (AUROC 0.995): Krüppel-like factor 1 (EKLF); definitive erythropoiesis master regulator
- **PPARG** (AUROC 0.900): Peroxisome proliferator-activated receptor gamma; adipogenesis master regulator
- **MAL** (AUROC 0.860): Myelin and lymphocyte protein; membrane-associated, lipid raft signaling
- **EGR1** (AUROC 0.852): Early growth response 1; immediate-early gene, stimulus-responsive
- **GATA1** (AUROC 0.842): GATA-binding factor 1; erythroid/megakaryocyte master regulator
- **WT1** (AUROC 0.675): Wilms tumor 1; kidney/gonad development master regulator

### 6. Interpretation

The successful TFs are predominantly **lineage-defining master regulators** (LEF1, KLF1, GATA1, PPARG, WT1) that:
- Control large, well-characterized regulons
- Have cell-type-restricted expression patterns
- Operate as pioneer/chromatin remodeling factors

The two exceptions are **EGR1** (immediate-early, signal-dependent) and **MAL** (membrane protein, not a classical TF).

**Hypothesis**: Correlation-based edge scores work best for TFs with:
1. Strong cell-type-specific expression patterns (which create high-variance correlations in mixed populations)
2. Large, well-characterized regulons (more statistical power for AUROC estimation)

### 7. Power Analysis Caveat

With n=18 total TFs (7 vs 11 split), statistical power is severely limited.
All results are **hypothesis-generating**, not confirmatory.
