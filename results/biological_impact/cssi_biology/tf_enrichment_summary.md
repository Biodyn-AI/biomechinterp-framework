# Follow-Up 3: Cell-Type-Specific TF Enrichment

Resolution: 0.5, 20 testable clusters

## Edge-Level Mann-Whitney (Matching vs Non-Matching TF Edges)

| Cluster | Cell Type | Expected TFs | Match Mean | Non-match Mean | MW p | BH q | Sig |
|---------|-----------|-------------|-----------|---------------|------|------|-----|
| 0 | T cell | LEF1, STAT4 | 0.0338 | 0.0437 | 0.269 | 0.3587 | No |
| 1 | T cell | LEF1, STAT4 | 0.0335 | 0.0564 | 0.1809 | 0.2584 | No |
| 10 | T cell | LEF1, STAT4 | 0.0502 | 0.1587 | 0.9108 | 0.9108 | No |
| 11 | neutrophil | SPI1 | 0.0692 | 0.0720 | 0.007458 | 0.02983 | Yes |
| 12 | erythrocyte | GATA1, KLF1, TAL1 | 0.1750 | 0.1153 | 0.03064 | 0.06128 | No |
| 13 | macrophage | SPI1 | 0.0670 | 0.0661 | 0.02492 | 0.06112 | No |
| 14 | T cell | LEF1, STAT4 | 0.0262 | 0.0789 | 0.7969 | 0.8389 | No |
| 15 | macrophage | SPI1 | 0.0480 | 0.1194 | 0.3757 | 0.4174 | No |
| 16 | T cell | LEF1, STAT4 | 0.2073 | 0.1632 | 0.3605 | 0.4174 | No |
| 17 | T cell | LEF1, STAT4 | 0.1264 | 0.1558 | 0.2925 | 0.3656 | No |
| 18 | macrophage | SPI1 | 0.1258 | 0.0910 | 0.002032 | 0.02032 | Yes |
| 19 | B cell | BCL6, SPI1 | 0.0839 | 0.0346 | 0.003201 | 0.02134 | Yes |
| 2 | macrophage | SPI1 | 0.0469 | 0.0730 | 0.05531 | 0.09218 | No |
| 20 | T cell | LEF1, STAT4 | 0.1440 | 0.1278 | 0.05208 | 0.09218 | No |
| 3 | B cell | BCL6, SPI1 | 0.0846 | 0.0412 | 0.0001279 | 0.002558 | Yes |
| 4 | neutrophil | SPI1 | 0.0826 | 0.0735 | 0.004964 | 0.02482 | Yes |
| 5 | B cell | BCL6, SPI1 | 0.0789 | 0.0445 | 0.01247 | 0.03563 | Yes |
| 6 | monocyte | SPI1 | 0.0739 | 0.0833 | 0.0275 | 0.06112 | No |
| 7 | T cell | LEF1, STAT4 | 0.0781 | 0.0336 | 0.01212 | 0.03563 | Yes |
| 8 | plasma cell | BCL6 | 0.5259 | 0.0817 | 0.06914 | 0.1064 | No |

**7/20 significant after BH.**

**0/20 Fisher tests significant after BH.**


## Recommendation

The cell-type-specific TF enrichment signal is weak. Most clusters do not show
significant enrichment of expected TF edges after multiple testing correction.
**Recommend dropping Analysis C from the paper.** The CSSI contribution is already
supported by existing synthetic/TRRUST validation in Tier 1.
