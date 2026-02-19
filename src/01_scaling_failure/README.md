# Scaling Behavior Analysis (Section 3.1)

This directory contains scripts for analyzing the scaling failure phenomenon in attention-based GRN inference from single-cell foundation models.

## Key Finding

Contrary to neural scaling law expectations, increasing cell counts systematically degrades GRN recovery quality in scGPT, with F1 scores dropping 50-100% from 200 to 1000 cells across all model tiers.

## Scripts

1. **`generate_extended_evidence.py`** - Main analysis script
   - Combines archived workshop snapshots with fresh multi-seed reruns
   - Generates Figure 1 (scaling failure curves) and Table 1 (degradation statistics)
   - Runtime: ~30 minutes

2. **`generate_submission_graphs.py`** - Robustness analysis
   - Produces Figure 2 (seed robustness degradation)
   - Cross-validates scaling failure across random seeds
   - Runtime: ~15 minutes

3. **`render_paper_bundle.py`** - Figure compilation
   - Assembles final publication-ready figures
   - Runtime: ~5 minutes

## Generated Outputs

- `fig_ext_multiseed_scaling_trrust_ci.png` - Main scaling curves (Figure 1)
- `fig_ext_multiseed_robustness.png` - Seed stability analysis (Figure 2)
- `scaling_analysis_results.json` - Quantitative metrics
- `robustness_summary.csv` - Cross-seed statistics

## Dependencies

- scanpy, numpy, pandas, scipy, matplotlib
- Pre-trained scGPT models (small, medium, large tiers)
- Tabula Sapiens tissue subsets (immune, kidney, lung)
- TRRUST and DoRothEA reference databases

## Usage

```bash
# Run complete scaling analysis
python generate_extended_evidence.py

# Generate robustness figures
python generate_submission_graphs.py

# Compile final bundle
python render_paper_bundle.py
```

## Expected Runtime

Total: ~50 minutes on standard hardware
- Memory requirement: 8-16GB RAM
- GPU optional but recommended for model inference