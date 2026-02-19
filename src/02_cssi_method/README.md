# Cell-State Stratified Interpretability (CSSI) - Section 3.10

This directory contains the implementation of CSSI, the novel method that eliminates scaling failure by stratifying cells before mechanistic analysis.

## Key Innovation

CSSI resolves the scaling failure problem by computing regulatory signals within homogeneous cell-state strata before aggregation, preventing heterogeneity-driven attention dilution. Achieves up to 1.85× improvement over pooled inference.

## Method Overview

1. **Stratification**: Partition cells into cell-state strata using annotations or clustering
2. **Per-stratum scoring**: Compute edge scores within each homogeneous stratum  
3. **Aggregation**: Combine via CSSI-max (capture any-state activity) or CSSI-mean (weighted consensus)
4. **Standard ranking**: Apply typical top-k or FDR-based selection

## Scripts

### Core Pipeline
1. **`01_data_exploration.py`** - Dataset inspection and quality control
2. **`02_full_pipeline.py`** - Main CSSI implementation and validation
3. **`05_figures.py`** - Generate Figure 15 (CSSI scaling mitigation)

### Validation Studies  
4. **`phase2_01_multi_tissue.py`** - Cross-tissue CSSI validation
5. **`phase2_02_perturbation_validation.py`** - Perturb-seq consistency testing
6. **`cssi_realdata_validation.py`** - Real biological data validation (Table 3)

### Advanced Analysis
7. **`phase2_03_ceiling_effect.py`** - Sample size convergence analysis
8. **`phase2_04_temporal_dynamics.py`** - Pseudotime-aware CSSI

## Key Outputs

- **Figure 15**: CSSI eliminates scaling failure in synthetic experiments
- **Table 3**: CSSI performance on real Tabula Sapiens cell-type compositions  
- `cssi_results.json` - Quantitative improvement metrics
- `stratified_edges.csv` - Cell-state-specific regulatory networks

## Usage Examples

### Basic CSSI Application
```python
from cssi_implementation import apply_cssi

# Stratify by cell type annotations
edges_cssi = apply_cssi(
    adata, 
    stratification_key='cell_type',
    method='cssi_max',  # or 'cssi_mean'
    edge_scoring='spearman'
)
```

### Synthetic Validation
```bash
# Run synthetic ground-truth validation
python 02_full_pipeline.py --mode synthetic --n_states 10

# Real data validation on Tabula Sapiens
python cssi_realdata_validation.py
```

## Validation Results

### Synthetic Experiments
- **Input**: Known ground-truth networks with state-specific edges
- **Result**: CSSI-max maintains F1 ≥ 0.900 across all heterogeneity levels
- **Baseline**: Pooled inference degrades from 0.850 → 0.514 F1 (39.5% drop)

### Real Biological Data
- **Dataset**: Tabula Sapiens immune cells (15 cell types)
- **Edges**: 22 known TF-target pairs with cell-type specificity
- **Improvement**: 1.62× at N=200, 1.33× at N=500 vs pooled analysis

## Algorithm Complexity

- **Time**: O(K × N/K × G²) = O(N × G²) where K = number of strata
- **Space**: O(K × G²) for storing per-stratum edge scores
- **Overhead**: Minimal - adds only cell clustering step

## When to Use CSSI

**Recommended for**:
- Studies with heterogeneous cell populations
- Rare but biologically important cell types (<10% frequency)
- Small to moderate sample sizes (N < 5,000)
- Cell-type-specific regulatory network discovery

**Pooled inference sufficient for**:
- Homogeneous cell populations
- Very large sample sizes (N ≥ 5,000) 
- Global regulatory network inference across all cell states

## Dependencies

- scanpy, anndata - single-cell data handling
- scipy, sklearn - statistical computations  
- networkx - network analysis
- matplotlib, seaborn - visualization

## Expected Runtime

- CSSI implementation: ~20 minutes per dataset
- Synthetic validation: ~1 hour (10 seeds × 6 configurations)  
- Real data validation: ~25 minutes
- Figure generation: ~10 minutes

## Theoretical Foundation

The CSSI advantage stems from signal preservation: for a TF-target edge active in stratum S₁ with correlation ρ₁ > 0, the pooled correlation ρ_pool ≈ (n₁/N)ρ₁ → 0 as heterogeneity grows, while CSSI-max = ρ₁ regardless of stratum count.

This prevents the heterogeneity-driven dilution that causes scaling failure in foundation model attention matrices.