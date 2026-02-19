# Multi-Model Validation (Section 3.12)

This directory validates whether attention-based GRN inference limitations are scGPT-specific or reflect fundamental issues across single-cell foundation models.

## Key Finding

**Convergent Failure Across Architectures**: Both scGPT and Geneformer achieve near-random AUROC (≈0.5) for attention-based GRN inference despite fundamental architectural differences, confirming that the problem is not model-specific but reflects limitations of attention-based regulatory network recovery.

## Models Tested

1. **scGPT** - GPT-style decoder, raw expression tokenization
2. **Geneformer** - BERT encoder, rank-value tokenization  
3. **scVI** - Variational autoencoder baseline (latent distances)
4. **C2S-Pythia** - Causal language model (limited validation)

## Scripts

1. **`geneformer_grn_pipeline.py`** - Geneformer attention extraction pipeline
2. **`geneformer_grn_validation.py`** - Main validation against TRRUST/DoRothEA
3. **Analysis artifacts**: Pre-computed edge scores and results

## Critical Comparison: scGPT vs Geneformer

| Model | Architecture | Tokenization | 200 cells AUROC | 1000 cells AUROC | Scaling Pattern |
|-------|--------------|--------------|------------------|-------------------|-----------------|
| scGPT | GPT Decoder | Raw expression | 0.51 | 0.46 | Monotonic degradation |
| Geneformer | BERT Encoder | Rank-value | 0.444 | 0.522 | Non-monotonic fluctuation |

**Key Insight**: Different scaling dynamics, identical outcome (≈random performance)

## What This Reveals

### scGPT Failure Mode
- **Cause**: Heterogeneity-driven attention dilution
- **Pattern**: Monotonic degradation with cell count
- **Mechanism**: Raw expression magnitudes amplify cell-state differences

### Geneformer Failure Mode  
- **Cause**: Attention encodes co-expression, not regulation
- **Pattern**: Stable but uninformative attention patterns
- **Mechanism**: Rank tokenization prevents dilution but doesn't solve core problem

### Convergent Evidence
Both models show:
- Strong correlation with expression co-occurrence (ρ = 0.31-0.42)
- No correlation with regulatory ground truth (ρ ≈ 0.0)
- **Conclusion**: Attention captures statistical co-expression, not causal regulation

## Generated Files

- `geneformer_edges_200cells.csv` - Attention-derived edges, 200 cells
- `geneformer_edges_500cells.csv` - Attention-derived edges, 500 cells  
- `geneformer_edges_1000cells.csv` - Attention-derived edges, 1000 cells
- `geneformer_grn_results.json` - AUROC/precision/recall metrics
- `geneformer_analysis_summary.png` - Figure 17

## Usage

```bash
# Run Geneformer GRN validation
python geneformer_grn_validation.py

# Alternative: just run pipeline
python geneformer_grn_pipeline.py
```

## Dependencies

- **Geneformer**: Requires special installation from source
- **Transformers**: HuggingFace transformers library
- **PyTorch**: For model inference
- **Standard libraries**: scanpy, pandas, sklearn for evaluation

## Expected Runtime

- Geneformer model loading: ~5 minutes  
- Attention extraction: ~45 minutes per cell count
- GRN evaluation: ~15 minutes
- **Total**: ~2 hours

## Memory Requirements

- **Minimum**: 16GB RAM
- **Recommended**: 32GB RAM + GPU (8GB VRAM)
- **Fallback**: CPU-only mode (slower but functional)

## Validation Protocol

1. **Identical pipeline**: Same attention-to-GRN conversion as scGPT
2. **Same references**: TRRUST and DoRothEA evaluation
3. **Same metrics**: AUROC, precision, recall, F1
4. **Same cell counts**: 200, 500, 1000 cells for direct comparison

## Implications for the Field

### For Practitioners
- Don't assume architectural improvements fix attention-GRN inference
- Rank-based tokenization helps attention stability but not GRN recovery
- Established GRN methods (SCENIC, GENIE3) remain preferred for network reconstruction

### For Method Developers
- The problem is fundamental: attention ≠ regulation
- Future solutions require:
  - Intervention-aware pretraining objectives
  - Causal representation learning
  - Purpose-built GRN inference modules

### For Foundation Model Research
- Attention interpretability limitations extend beyond NLP
- Biology-specific inductive biases needed for causal discovery
- Co-expression ≠ causation: a critical distinction for regulatory inference