# Analysis B: Biological Characterization of Attention — Corrected Results

## Corrections Applied
1. STRING PPI evaluated at all 18 layers (was 6)
2. BH correction applied to Spearman trend tests (6 databases) and enrichment tests (5 layers x 3 databases x 2 tails = 30 tests)
3. TRRUST removed from enrichment table (175 pairs in ~1.88M possible = 0.009% base rate; 0 overlap at every layer)
4. Reactome acknowledged as null; pathway co-membership signals characterized honestly

## Reference Databases
| Database | Type | Pairs in HVG | Genes in HVG |
|----------|------|-------------|-------------|
| TRRUST | Transcriptional regulation | 175 | 133 |
| STRING >=700 | Protein-protein interaction | 2,747 | 1,185 |
| STRING >=900 | High-confidence PPI | 1,675 | 883 |
| Reactome | Pathway co-membership | 238,640 | 1,302 |
| KEGG | Pathway co-membership | 24,812 | 889 |
| GO_BP | Functional co-annotation | 135,596 | 1,624 |

## Key Finding: Layer-Specific Biological Specialization

| Property | PPI (STRING >=700) | Regulation (TRRUST) |
|----------|-------------------|-------------------|
| Best layer | L0 (AUROC=0.640) | L15 (AUROC=0.750) |
| Trend with depth | **Decreasing** (rho=-0.608, p_raw=7.450e-03, q_BH=1.118e-02) | **Increasing** (rho=+0.511, p_raw=3.028e-02, q_BH=3.028e-02) |

**Cross-correlation (STRING_700 vs TRRUST layer profiles)**: rho=-0.445, p=0.064 (marginal; STRING_900 vs TRRUST: rho=-0.546, p=0.019)

## AUROC Results — Full Layer Profile

| Layer | STRING_700 | STRING_900 | TRRUST | Reactome | KEGG | GO_BP |
|-------|-----------|-----------|--------|----------|------|-------|
| L0 | 0.6397 | 0.6441 | 0.5576 | 0.5047 | 0.5299 | 0.5262 |
| L1 | 0.5170 | 0.5013 | 0.6913 | 0.5008 | 0.4840 | 0.5012 |
| L2 | 0.5456 | 0.5362 | 0.6413 | 0.5068 | 0.4683 | 0.4850 |
| L3 | 0.5536 | 0.5488 | 0.6569 | 0.5016 | 0.4727 | 0.4822 |
| L4 | 0.5458 | 0.5301 | 0.6862 | 0.5164 | 0.5246 | 0.5153 |
| L5 | 0.5482 | 0.5374 | 0.6636 | 0.5037 | 0.4992 | 0.5033 |
| L6 | 0.5403 | 0.5287 | 0.6465 | 0.5078 | 0.5207 | 0.5135 |
| L7 | 0.5311 | 0.5152 | 0.6307 | 0.5109 | 0.5171 | 0.5185 |
| L8 | 0.5303 | 0.5192 | 0.6314 | 0.5088 | 0.5314 | 0.5337 |
| L9 | 0.5370 | 0.5410 | 0.6499 | 0.5182 | 0.5543 | 0.5270 |
| L10 | 0.5475 | 0.5421 | 0.6121 | 0.5123 | 0.5416 | 0.5287 |
| L11 | 0.5183 | 0.5108 | 0.6504 | 0.5130 | 0.5438 | 0.5298 |
| L12 | 0.5155 | 0.5018 | 0.6581 | 0.5061 | 0.5331 | 0.5163 |
| L13 | 0.5742 | 0.5593 | 0.7084 | 0.5234 | 0.5475 | 0.5304 |
| L14 | 0.5119 | 0.4930 | 0.7353 | 0.5186 | 0.5546 | 0.5364 |
| L15 | 0.5249 | 0.4998 | 0.7497 | 0.5206 | 0.5551 | 0.5411 |
| L16 | 0.4746 | 0.4574 | 0.7330 | 0.5156 | 0.5637 | 0.5440 |
| L17 | 0.4847 | 0.4604 | 0.6640 | 0.5164 | 0.5385 | 0.5411 |
| **Correlation** | **0.5624** | **0.5586** | **0.6488** | **0.5139** | **0.5412** | **0.5128** |

## Layer Trend Tests (Spearman rho vs layer depth, BH-corrected)

| Database | rho | p_raw | q_BH | Direction | Significant? |
|----------|-----|-------|------|-----------|-------------|
| KEGG | +0.831 | 1.961e-05 | 5.884e-05 | increasing | Yes |
| GO_BP | +0.846 | 9.868e-06 | 5.884e-05 | increasing | Yes |
| Reactome | +0.731 | 5.677e-04 | 1.135e-03 | increasing | Yes |
| STRING_700 | -0.608 | 7.450e-03 | 1.118e-02 | decreasing | Yes |
| STRING_900 | -0.581 | 1.145e-02 | 1.374e-02 | decreasing | Yes |
| TRRUST | +0.511 | 3.028e-02 | 3.028e-02 | increasing | Yes |

## Attention vs. Correlation by Reference (Best Layer)

| Reference | Best Attn | Corr | Delta | Attn > Corr? |
|-----------|-----------|------|-------|-------------|
| TRRUST | 0.7497 (L15) | 0.6488 | **+0.1009** | Yes |
| STRING_900 | 0.6441 (L0) | 0.5586 | **+0.0855** | Yes |
| STRING_700 | 0.6397 (L0) | 0.5624 | **+0.0773** | Yes |
| GO_BP | 0.5440 (L16) | 0.5128 | +0.0312 | Yes |
| KEGG | 0.5637 (L16) | 0.5412 | +0.0225 | Yes |
| Reactome | 0.5234 (L13) | 0.5139 | +0.0095 | Yes |

## Partial Correlation (Controlling for Expression Similarity)

| Reference | Raw r | Partial r | p (partial) | Signal Retained |
|-----------|-------|-----------|-------------|----------------|
| TRRUST | 0.3628 (p=2.49e-12) | 0.3529 (p=1.06e-11) | 1.06e-11 | 97% |
| Reactome | 0.0205 (p=4.03e-02) | 0.0154 (p=1.24e-01) | 1.24e-01 | 75% |
| KEGG | 0.0782 (p=5.76e-15) | 0.0559 (p=2.38e-08) | 2.38e-08 | 72% |
| GO_BP | 0.0691 (p=5.12e-12) | 0.0616 (p=7.38e-10) | 7.38e-10 | 89% |

## Enrichment Analysis (Top-1000 Attention Edges, BH-corrected)

Fisher's exact test. TRRUST excluded (base rate 0.009%, 0 overlap at all layers).

| Layer | Reactome OR (q_BH) | KEGG OR (q_BH) | GO_BP OR (q_BH) |
|-------|--------------------|----------------|-----------------|
| L0 | 1.25 (2.76e-02)* | 3.38 (2.61e-10)* | 1.47 (1.34e-03)* |
| L5 | 1.21 (6.28e-02) | 0.30 (1.00e+00) | 1.55 (1.83e-04)* |
| L10 | 1.02 (8.57e-01) | 1.15 (7.21e-01) | 0.85 (1.00e+00) |
| L14 | 1.20 (7.08e-02) | 1.85 (1.56e-02)* | 1.23 (9.57e-02) |
| L17 | 1.99 (4.29e-16)* | 3.71 (2.03e-12)* | 2.12 (3.60e-13)* |

## Summary of Findings

1. **TRRUST regulation is the strongest signal** (AUROC=0.75 at L15), and attention substantially outperforms correlation (+0.10). This signal survives expression residualization (97% retained).

2. **STRING PPI shows a clear early-layer signal** (L0: AUROC~0.64), outperforming correlation (+0.08). The decreasing trend with depth is significant after BH correction (q=0.011).

3. **Layer specialization shows a trend toward anti-correlation**: STRING_700 and TRRUST profiles show a negative correlation across layers (rho=-0.445, p=0.064), suggestive of different layers encoding different relationship types, though not reaching conventional significance. STRING_900 vs TRRUST is significant (rho=-0.546, p=0.019).

4. **Reactome is a null result**: AUROC 0.50-0.52 across all layers, partial correlation non-significant (p=0.35). Attention does not meaningfully predict Reactome pathway co-membership.

5. **KEGG and GO_BP show weak but real signals**: AUROC 0.53-0.56 at best layers, both significantly enriched in top attention edges at L0 and L17. KEGG signal is partially confounded by expression (70% retained), while GO_BP is more robust (90% retained). These are above chance but much weaker than TRRUST or STRING.

6. **Top-edge enrichment survives BH correction** for KEGG and GO_BP at L0 and L17 (strongest: KEGG L17 OR=3.71, Reactome L17 OR=1.99). Mid-layers (L10) show no enrichment.
