#!/usr/bin/env python3
"""
Round 11 Experiment 0: Multiplicity Sensitivity Analysis
=========================================================
Extends the R10 registry to 33 analyses, defines 3 alternative BH families,
and tests whether headline inferences change under broader correction.

Addresses: MC1 ("confirmatory/descriptive split under-justified")
           ST1 ("multiplicity sensitivity analysis")
"""

import os, sys, json, math
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round11_multiplicity_sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)


def build_full_registry():
    """Build complete registry of ALL tests across 33 analyses."""
    tests = []

    def add(analysis_id, analysis_name, hypothesis, test_type,
            raw_p, effect_size, sample_size, is_primary=False):
        p_val = None
        if raw_p is not None:
            if isinstance(raw_p, str):
                val = raw_p.replace('<', '').replace('>', '').strip()
                if '10^' in val:
                    exp_part = val.split('^')[1].replace('{', '').replace('}', '')
                    p_val = 10 ** float(exp_part)
                else:
                    p_val = float(val)
            else:
                p_val = float(raw_p)
        is_confirmatory = p_val is not None
        tests.append({
            'test_id': len(tests) + 1,
            'analysis_id': analysis_id,
            'analysis_name': analysis_name,
            'hypothesis': hypothesis,
            'test_type': test_type,
            'raw_p': p_val,
            'effect_size': effect_size,
            'sample_size': sample_size,
            'is_confirmatory': is_confirmatory,
            'is_primary': is_primary,
        })

    # ---- Analyses 1-29 (from R10 registry, unchanged) ----
    # Analysis 1: Scaling
    add(1, 'Scaling Behavior', 'Cell count improves GRN (TRRUST)', 'Sign test', 0.002, '100% degradation', '9 runs', True)
    add(1, 'Scaling Behavior', 'Cell count improves GRN (TRRUST)', 'Wilcoxon', 0.002, '--', '9 runs')
    add(1, 'Scaling Behavior', 'Cell count improves GRN (DoRothEA)', 'Sign test', 0.002, '100% degradation', '9 runs')
    add(1, 'Scaling Behavior', 'F1 confidence intervals', 'Bootstrap', None, '--', '10k resamples')
    add(1, 'Scaling Behavior', 'Seed stability w/ scaling', 'Paired test', '<0.001', '46-48% drop', '3 seeds')

    # Analysis 2: Mediation Bias
    add(2, 'Mediation Bias', 'Components additive', 'Lower bound', '<0.001', 'A_lb/|TE|=0.73', '16 pairs', True)
    add(2, 'Mediation Bias', 'Rankings stable', 'Stability', '<0.001', '0.067->0.003', '16 pairs')

    # Analysis 3: Detectability Theory
    add(3, 'Detectability Theory', 'Theory matches empirical', 'Correlation', '<10^{-6}', 'r=0.887', 'Phase space', True)
    add(3, 'Detectability Theory', 'Intervention more detectable', 'Ratio', '<0.001', '44.4% cells', 'Simulation')

    # Analysis 4: Cross-Context Consistency
    add(4, 'Cross-Context Consistency', 'Immune-kidney transfer', 'Spearman', 0.024, 'rho=0.71', 'Pairs', True)
    add(4, 'Cross-Context Consistency', 'Immune-lung transfer', 'Spearman', 0.089, 'rho=0.32', 'Pairs')
    add(4, 'Cross-Context Consistency', 'Kidney-lung transfer', 'Spearman', 0.156, 'rho=-0.44', 'Pairs')
    add(4, 'Cross-Context Consistency', 'Correlation confidence', 'Bootstrap', None, '--', '10k resamples')
    add(4, 'Cross-Context Consistency', 'Correlation significance', 'Permutation', '<0.001', '--', '5k permutations')

    # Analysis 5: Perturbation Validation
    add(5, 'Perturbation Validation', 'Dixit 13d raw', 'Spearman', 0.032, 'rho=0.269', 'Pert. pairs', True)
    add(5, 'Perturbation Validation', 'Dixit 13d adjusted', 'Spearman', 0.020, 'rho=0.199', 'Pert. pairs')
    add(5, 'Perturbation Validation', 'Dixit 7d', 'Spearman', 0.15, 'rho=0.112', 'Pert. pairs')
    add(5, 'Perturbation Validation', 'Adamson', 'Spearman', 0.089, 'Marginal', 'Pert. pairs')
    add(5, 'Perturbation Validation', 'Shifrut raw', 'Spearman', 0.031, 'rho=-0.325', 'Pert. pairs')
    add(5, 'Perturbation Validation', 'Shifrut adjusted', 'Spearman', 0.876, 'rho=0.004', 'Pert. pairs')

    # Analysis 6: Cross-Species Ortholog
    add(6, 'Cross-Species Ortholog', 'Global conservation', 'Spearman', '<10^{-300}', 'rho=0.743', '25876', True)
    add(6, 'Cross-Species Ortholog', 'Signs conserved', 'Sign test', '<0.001', '88.6%', '25876')
    add(6, 'Cross-Species Ortholog', 'Overlap above chance', 'Permutation', '<0.001', '8-484x', '1k permutations')
    add(6, 'Cross-Species Ortholog', 'TF conservation varies', 'Range', None, '-0.12 to 0.90', '61 TFs')

    # Analysis 7: Pseudotime
    add(7, 'Pseudotime', 'TFs precede targets', 'Directionality test', 0.068, '21.4% consistent', '56 pairs', True)
    add(7, 'Pseudotime', 'Exceeds shuffled', 'Mann-Whitney', 0.068, 'd=1.58', '500 permutations')
    add(7, 'Pseudotime', 'Exceeds random', 'Mann-Whitney', 0.37, '--', '200 sets')
    add(7, 'Pseudotime', 'T cell lineage', 'Lineage', None, '16.7%', '24 pairs')
    add(7, 'Pseudotime', 'B cell lineage', 'Lineage', None, '13.3%', '15 pairs')
    add(7, 'Pseudotime', 'Myeloid lineage', 'Lineage', None, '35.3%', '17 pairs')

    # Analysis 8: Batch/Donor Leakage
    add(8, 'Batch/Donor Leakage', 'Donor (immune)', 'Logistic regression', '<0.001', 'AUC 0.85-0.87', '20k, 24 donors', True)
    add(8, 'Batch/Donor Leakage', 'Donor (lung)', 'Logistic regression', '<0.001', 'AUC 0.94-0.96', '20k, 4 donors')
    add(8, 'Batch/Donor Leakage', 'Assay method', 'Random forest', '<0.001', 'AUC 0.96-0.99', 'All tissues')
    add(8, 'Batch/Donor Leakage', 'CV accuracy', '5-fold CV', None, '--', '5 folds')
    add(8, 'Batch/Donor Leakage', 'Edge stability (LODO)', 'LODO variance', '<0.001', 'High variance', 'Var. donors')
    add(8, 'Batch/Donor Leakage', 'Generalization gap', 'Cross-donor', 0.012, '6.6 pp gap', 'Lung')

    # Analysis 9: Calibration
    add(9, 'Calibration', 'Calibration improves', 'Paired', '<0.001', '4-7x', '6 methods', True)
    add(9, 'Calibration', 'Isotonic better', 'Paired', '<0.001', 'ECE 0.06-0.08', '6 methods')
    add(9, 'Calibration', 'Coverage valid', 'Coverage', None, '>=95%', 'alpha=0.05')
    add(9, 'Calibration', 'Calibration robust', 'Bootstrap', None, 'CI < 0.02', '200 resamples')
    add(9, 'Calibration', 'Calibrators dont transfer', 'Transfer', '<0.001', 'ECE 0.32-0.42', 'K562->T')

    # Analysis 10: CSSI
    add(10, 'CSSI', 'CSSI prevents degradation', 'Spearman', 0.99, 'r=-0.001', '10 seeds', True)
    add(10, 'CSSI', 'Pooled degrades', 'Spearman', '<10^{-4}', 'r=-0.618', '10 seeds')
    add(10, 'CSSI', 'CSSI outperforms pooled', 'Wilcoxon', 2.5e-11, '1.13-1.85x', '60 combinations')
    add(10, 'CSSI', 'Real PBMC improvement', 'Bootstrap', 0.03, '1.16x', '3k cells')
    add(10, 'CSSI', 'CSSI w/ real proportions', 'Wilcoxon', 2.4e-8, '1.62x', '10 seeds')
    add(10, 'CSSI', 'CSSI on attention', 'Layer analysis', None, 'AUROC 0.68-0.69', '497 cells')

    # Analysis 11: Synthetic Validation
    add(11, 'Synthetic Validation', 'Recovery degrades', 'Correlation', '<0.01', 'r: 0.85->0.62', 'Synthetic', True)
    add(11, 'Synthetic Validation', 'Shapley outperforms', 'Paired', '<0.001', '91% improvement', 'Synthetic')
    add(11, 'Synthetic Validation', 'Empirical = theory', 'Correlation', '<10^{-6}', 'r=0.887', 'Phase space')

    # Analysis 12: Multi-Model
    add(12, 'Multi-Model', 'Geneformer TRRUST', 'AUROC', 0.89, 'AUROC 0.44-0.55', '3 counts', True)
    add(12, 'Multi-Model', 'Geneformer DoRothEA', 'AUROC', 0.76, 'AUROC 0.47-0.49', '3 counts')
    add(12, 'Multi-Model', 'Cross-model comparison', 'Comparative', None, 'Both ~0.5', '2 models')
    add(12, 'Multi-Model', 'GF confidence intervals', 'Bootstrap', None, 'CIs incl. 0.5', '10k resamples')
    add(12, 'Multi-Model', 'Attention = co-expression', 'Correlation', '<10^{-50}', 'rho=0.31-0.42', 'Both')
    add(12, 'Multi-Model', 'Attention != regulation', 'Correlation', '>0.3', 'rho=-0.01-0.02', 'Both')

    # Analysis 13: Controlled-Composition Scaling
    add(13, 'Controlled-Composition Scaling', 'N degrades AUROC (single type)', 'Spearman', 0.079, 'rho=-0.33', '30 runs', True)
    add(13, 'Controlled-Composition Scaling', 'N degrades AUROC (fixed comp)', 'Spearman', 0.82, 'rho=-0.05', '20 runs')
    add(13, 'Controlled-Composition Scaling', 'Diversity improves', 'Spearman', '<10^{-4}', 'rho=+0.63', '35 runs')

    # Analysis 14: Perturbation-First Validation
    add(14, 'Perturbation-First Validation', 'Replogle primary', 'One-sample t', '<10^{-4}', 'AUROC=0.696', '151 perts', True)
    add(14, 'Perturbation-First Validation', 'Replogle baseline', 'One-sample t', 0.32, 'AUROC=0.511', '44 perts')
    add(14, 'Perturbation-First Validation', 'Replogle Wilcoxon', 'Wilcoxon', 0.30, 'Median=0.503', '44 perts')

    # Analysis 15: Robust Attention Residualization
    add(15, 'Robust Attention Residualization', 'Edge-expression correlation', 'Spearman', '<10^{-50}', 'rho=0.842', '75962 pairs', True)
    add(15, 'Robust Attention Residualization', 'R2 OLS full', 'OLS R2', None, 'R2=0.27', '75962 pairs')
    add(15, 'Robust Attention Residualization', 'R2 GBDT', 'GBDT R2', None, 'R2=0.51', '75962 pairs')
    add(15, 'Robust Attention Residualization', 'CF residual AUROC', 'CF-AUROC', None, 'AUROC=0.73', '61 positives')
    add(15, 'Robust Attention Residualization', 'CF stability', '10-seed variance', None, 'sigma=0.001', '10 seeds')

    # Analysis 16: Degree-Preserving Null Models
    add(16, 'Degree-Preserving Null Models', 'AUROC > random', 'Permutation', '<0.001', 'z=6.9', '1k permutations', True)
    add(16, 'Degree-Preserving Null Models', 'AUROC > degree', 'Permutation', '<0.005', 'z=3.63', '200 permutations')
    add(16, 'Degree-Preserving Null Models', 'Degree explains AUROC', 'Decomposition', None, '73% global', '75962 pairs')
    add(16, 'Degree-Preserving Null Models', 'Per-TF AUROC', 'Per-TF eval', None, 'mean=0.69+/-0.20', '18 TFs')
    add(16, 'Degree-Preserving Null Models', 'Per-TF excess', 'Per-TF eval', None, '+0.035+/-0.20', '18 TFs')
    add(16, 'Degree-Preserving Null Models', 'Precision@k', 'Prec@k', None, '0.030 vs 0.002', '18 TFs')

    # Analysis 17: Attention-Correlation Mapping
    add(17, 'Attention-Correlation Mapping', 'Within-tissue R2', 'rho^2', '<10^{-50}', 'R2=0.10-0.18', '50k edges', True)
    add(17, 'Attention-Correlation Mapping', 'Cross-tissue R2', 'OLS R2', None, 'R2<0.02', '21k-26k edges')
    add(17, 'Attention-Correlation Mapping', 'Cross-tissue rho', 'Spearman', '<10^{-9}', 'rho=-0.02 to -0.05', '3 conditions')

    # Analysis 18: Perturbation Sensitivity
    add(18, 'Perturbation Sensitivity', 'All AUROC > 0.5 (27 conditions)', 't-test', '<0.005', 'AUROC=0.62-0.76', '13-1158 perts', True)
    add(18, 'Perturbation Sensitivity', 'TF vs non-TF', 'Comparison', None, 'Delta<=0.02', '1-83 TFs')

    # Analysis 19: CSSI Extended Null
    add(19, 'CSSI Extended Null', 'Null inflation', 'Delta AUROC', None, '<=-0.20', 'K=2-20')
    add(19, 'CSSI Extended Null', 'CSSI vs SCENIC', 'Comparative', None, 'Equivalent', '3 methods')
    add(19, 'CSSI Extended Null', 'Per-edge FDR', 'BH-FDR', None, 'FDR<=0.11', 'K=5-15')

    # Analysis 20: K-Sensitivity
    add(20, 'K-Sensitivity Analysis', 'Continuous AUROC', 'Comparison', None, '0.86->0.93', '9 runs')
    add(20, 'K-Sensitivity Analysis', 'K-sensitivity', 'Comparison', None, 'F1~10^-4 all K', '5 K values')

    # Analysis 21: Trivial Baseline
    add(21, 'Trivial Baseline Comparison', 'Variance vs Correlation', 'Paired t', '<10^{-24}', 'Delta=+0.186', '151 perts', True)
    add(21, 'Trivial Baseline Comparison', 'Mean expr vs Correlation', 'Paired t', '<10^{-20}', 'Delta=+0.146', '151 perts')
    add(21, 'Trivial Baseline Comparison', 'Dropout vs Correlation', 'Paired t', '<10^{-12}', 'Delta=+0.113', '151 perts')
    add(21, 'Trivial Baseline Comparison', 'TF degree vs Correlation', 'Paired t', '<10^{-29}', 'Delta=-0.196', '151 perts')

    # Analysis 22: Bootstrap Per-TF CIs
    add(22, 'Bootstrap Per-TF CIs', 'Global boot CI', 'Bootstrap', None, 'CI=[0.71,0.77]', '200 iterations')
    add(22, 'Bootstrap Per-TF CIs', 'Per-TF boot CI', 'Bootstrap', None, 'CI=[0.59,0.72]', '200 iterations')

    # Analysis 23: Attention Perturbation-First
    add(23, 'Attention Perturbation-First', 'Attn vs Corr L13 (Wilcoxon)', 'Wilcoxon', 0.726, 'diff=0.001', 'n=280', True)
    add(23, 'Attention Perturbation-First', 'Attn vs Corr L13 (Paired t)', 'Paired t', 0.931, 'diff=0.001', 'n=280')

    # Analysis 24: Full 18-Layer Perturbation-First
    add(24, 'Full 18-Layer Perturbation-First', 'L15 vs Corr', 'Wilcoxon', 0.0009, 'Delta=+0.040', 'n=280', True)
    add(24, 'Full 18-Layer Perturbation-First', 'Split-sample validation', 'Wilcoxon', 0.017, 'AUROC=0.750', 'n=140')
    add(24, 'Full 18-Layer Perturbation-First', 'Per-layer (18)', 'Wilcoxon', 0.0009, 'AUROC 0.47-0.74', 'n=280')

    # Analysis 25: Attention-Specific Confound Decomposition
    add(25, 'Attention-Specific Confound Decomposition', 'Attn residual OLS', 'CF-AUROC', None, 'AUROC=0.538', '174 positives')
    add(25, 'Attention-Specific Confound Decomposition', 'Attn residual GBDT', 'CF-AUROC', None, 'AUROC=0.554', '174 positives')
    add(25, 'Attention-Specific Confound Decomposition', 'Corr residual OLS', 'CF-AUROC', None, 'AUROC=0.622', '174 positives')
    add(25, 'Attention-Specific Confound Decomposition', 'Attn degree-null', 'Permutation', 0.023, 'z=2.0', '200 permutations', True)
    add(25, 'Attention-Specific Confound Decomposition', 'Corr degree-null', 'Permutation', 0.018, 'z=2.1', '200 permutations')

    # Analysis 26: Conditional Incremental Value
    add(26, 'Conditional Incremental Value', 'Gene-only CV', '5-fold CV', None, 'AUROC=0.895', '280 perts')
    add(26, 'Conditional Incremental Value', 'Delta gene+attn', 'Bootstrap', None, '-0.0004 [-0.001,0]', '100 iterations')
    add(26, 'Conditional Incremental Value', 'Delta gene+corr', 'Bootstrap', None, '-0.002 [-0.005,0]', '100 iterations')
    add(26, 'Conditional Incremental Value', 'TF stratified', 'Subgroup', None, 'TF 0.913, non-TF 0.895', '14/266')

    # Analysis 27: Per-Head TRRUST Ranking
    add(27, 'Per-Head TRRUST Ranking', 'Heads differ in reg.', 'AUROC range', None, '0.34-0.75', '324 heads')
    add(27, 'Per-Head TRRUST Ranking', 'Top-5 heads identified', 'AUROC', None, 'AUROC=0.70-0.75', '500 cells')

    # Analysis 28: Cross-Context CRISPRa
    add(28, 'Cross-Context CRISPRa Replication', 'Attn vs Corr L13 CRISPRa', 'Wilcoxon', '<10^{-6}', 'diff=-0.106', 'n=77', True)
    add(28, 'Cross-Context CRISPRa Replication', 'Attn vs Corr L6 CRISPRa', 'Wilcoxon', 1e-6, 'diff=-0.098', 'n=77')
    add(28, 'Cross-Context CRISPRa Replication', 'Attn vs Corr L18 CRISPRa', 'Wilcoxon', '<10^{-6}', 'diff=-0.105', 'n=77')

    # Analysis 29: Head-Level Causal Ablation (original R9)
    add(29, 'Head-Level Causal Ablation', 'Regulatory vs baseline', 'Wilcoxon', 0.244, 'Delta=-0.0002', 'n=280', True)
    add(29, 'Head-Level Causal Ablation', 'Random vs baseline (rep 1)', 'Wilcoxon', 0.065, 'Delta=+0.0007', 'n=280')
    add(29, 'Head-Level Causal Ablation', 'Random vs baseline (rep 2)', 'Wilcoxon', 0.069, 'Delta=+0.0004', 'n=280')
    add(29, 'Head-Level Causal Ablation', 'Random vs baseline (rep 3)', 'Wilcoxon', '<10^{-6}', 'Delta=+0.0028', 'n=280')
    add(29, 'Head-Level Causal Ablation', 'Entropy-matched vs baseline', 'Wilcoxon', '<10^{-6}', 'Delta=+0.0042', 'n=280')

    # ---- Analyses 30-33 (NEW in R10/R11) ----

    # Analysis 30: Nested Layer Selection (from round10_nested_layer_selection)
    add(30, 'Nested Layer Selection', 'L15 pooled Wilcoxon', 'Wilcoxon', 0.000933, 'Delta=+0.040', 'n=280', True)
    add(30, 'Nested Layer Selection', 'Bonferroni-adjusted', 'Bonferroni', 0.01679, 'd=0.22', 'n=280')
    add(30, 'Nested Layer Selection', 'Bootstrap CI', 'Bootstrap', None, '[0.018, 0.062]', '280 perts')
    add(30, 'Nested Layer Selection', 'Cohen d', 'Effect size', None, 'd=0.224', '280 perts')

    # Analysis 31: Hard-Generalization Incremental Value (from round10_hard_generalization)
    add(31, 'Hard-Generalization Incremental', 'Cross-pert delta gene+attn', 'Bootstrap CI', None, '-0.0004 [-0.001,0]', '280 perts', True)
    add(31, 'Hard-Generalization Incremental', 'Cross-pert delta gene+corr', 'Bootstrap CI', None, '-0.0015 [-0.004,0]', '280 perts')
    add(31, 'Hard-Generalization Incremental', 'Cross-gene delta gene+attn', 'Bootstrap CI', None, '-0.0003 [-0.001,0]', '280 perts')
    add(31, 'Hard-Generalization Incremental', 'Cross-gene delta gene+corr', 'Bootstrap CI', None, '-0.0010 [-0.004,0]', '280 perts')
    add(31, 'Hard-Generalization Incremental', 'Joint delta gene+attn', 'Bootstrap CI', None, '-0.0003 [-0.001,0]', '280 perts')
    add(31, 'Hard-Generalization Incremental', 'Joint delta gene+corr', 'Bootstrap CI', None, '-0.0011 [-0.004,0]', '280 perts')

    # Analysis 32: Expanded Ablation (from round10_expanded_ablation)
    add(32, 'Expanded Ablation Controls', 'Reg top-5 vs baseline', 'Wilcoxon', 0.244, 'Drop=-0.0002', 'n=280', True)
    add(32, 'Expanded Ablation Controls', 'Reg top-10 vs baseline', 'Wilcoxon', 0.937, 'Drop=0.0', 'n=280')
    add(32, 'Expanded Ablation Controls', 'Reg top-20 vs baseline', 'Wilcoxon', 0.596, 'Drop=+0.0003', 'n=280')
    add(32, 'Expanded Ablation Controls', 'Reg top-50 vs baseline', 'Wilcoxon', 0.100, 'Drop=+0.0021', 'n=280')
    add(32, 'Expanded Ablation Controls', 'Bottom-5 vs baseline', 'Wilcoxon', 0.005, 'Drop=-0.0025', 'n=280')
    add(32, 'Expanded Ablation Controls', 'Bottom-10 vs baseline', 'Wilcoxon', 0.050, 'Drop=-0.0030', 'n=280')
    add(32, 'Expanded Ablation Controls', 'Random-10 vs baseline', 'Wilcoxon', 0.009, 'Drop=+0.0029', 'n=280')
    add(32, 'Expanded Ablation Controls', 'Random-20 vs baseline', 'Wilcoxon', 1.09e-8, 'Drop=+0.0071', 'n=280')
    add(32, 'Expanded Ablation Controls', 'Random-50 vs baseline', 'Wilcoxon', 0.351, 'Drop=+0.0002', 'n=280')

    # Analysis 33: Shifrut T-Cell Cross-Context (from round10_shifrut_cross_context)
    add(33, 'Cross-Context T-Cell CRISPRi', 'Attn vs Corr L6 T-cell', 'Wilcoxon', 1.0, 'diff=+0.016', 'n=7', True)
    add(33, 'Cross-Context T-Cell CRISPRi', 'Attn vs Corr L13 T-cell', 'Wilcoxon', 0.938, 'diff=-0.003', 'n=7')
    add(33, 'Cross-Context T-Cell CRISPRi', 'Attn vs Corr L15 T-cell', 'Wilcoxon', 0.813, 'diff=+0.029', 'n=7')
    add(33, 'Cross-Context T-Cell CRISPRi', 'Attn vs Corr L18 T-cell', 'Wilcoxon', 0.938, 'diff=+0.030', 'n=7')

    return tests


# Headline claims: (label, analysis_id, hypothesis_substring_list)
# We look up tests by analysis_id + hypothesis substring to avoid hardcoded IDs
HEADLINE_CLAIMS = [
    ('Scaling degradation (Sign, TRRUST)', 1, ['Cell count improves GRN (TRRUST)'], 'Sign test'),
    ('Scaling degradation (Wilcoxon)', 1, ['Cell count improves GRN (TRRUST)'], 'Wilcoxon'),
    ('Mediation bias lower bound', 2, ['Components additive'], None),
    ('Detectability theory r=0.887', 3, ['Theory matches empirical'], None),
    ('Cross-tissue immune-kidney', 4, ['Immune-kidney transfer'], None),
    ('CSSI outperforms pooled', 10, ['CSSI outperforms pooled'], None),
    ('Trivial baseline: variance', 21, ['Variance vs Correlation'], None),
    ('Trivial baseline: TF degree', 21, ['TF degree vs Correlation'], None),
    ('Attn vs Corr CRISPRi L13', 23, ['Attn vs Corr L13 (Wilcoxon)'], None),
    ('L15 vs Corr raw', 24, ['L15 vs Corr'], None),
    ('L15 split-sample validation', 24, ['Split-sample validation'], None),
    ('CRISPRa L13 Attn < Corr', 28, ['Attn vs Corr L13 CRISPRa'], None),
    ('Reg heads ablation null', 29, ['Regulatory vs baseline'], None),
    ('Nested CV L15 pooled', 30, ['L15 pooled Wilcoxon'], None),
    ('Expanded ablation reg-top5', 32, ['Reg top-5 vs baseline'], None),
    ('Expanded ablation random-20', 32, ['Random-20 vs baseline'], None),
    ('T-cell L15 Attn vs Corr', 33, ['Attn vs Corr L15 T-cell'], None),
]


def run():
    print("=" * 70)
    print("MULTIPLICITY SENSITIVITY ANALYSIS")
    print("=" * 70)

    tests = build_full_registry()
    n_total = len(tests)
    confirmatory = [t for t in tests if t['is_confirmatory']]
    descriptive = [t for t in tests if not t['is_confirmatory']]
    n_confirm = len(confirmatory)
    n_descriptive = len(descriptive)
    n_analyses = len(set(t['analysis_id'] for t in tests))
    primary_tests = [t for t in tests if t['is_primary']]

    print(f"\n  Total tests: {n_total}")
    print(f"  Confirmatory (p-value present): {n_confirm}")
    print(f"  Descriptive (no p-value): {n_descriptive}")
    print(f"  Analyses: {n_analyses}")
    print(f"  Primary tests (one per analysis): {len(primary_tests)}")

    # Classification rule
    print(f"\n  CLASSIFICATION RULE: A test is confirmatory if and only if it")
    print(f"  produces an explicit p-value against a directional or non-null hypothesis.")
    print(f"  Tests reporting only descriptive summaries (CIs, R2, effect sizes) are descriptive.")

    # ---- Define 3 BH families ----
    # Family A: Current (all confirmatory tests)
    family_A_tests = confirmatory
    family_A_ps = np.array([t['raw_p'] for t in family_A_tests])

    # Family B: Maximal (all tests; descriptive get p=1.0)
    family_B_ps = np.array([t['raw_p'] if t['is_confirmatory'] else 1.0 for t in tests])

    # Family C: Analysis-level (one primary test per analysis)
    primary_confirm = [t for t in primary_tests if t['is_confirmatory']]
    family_C_ps = np.array([t['raw_p'] for t in primary_confirm])

    families = {
        'A_confirmatory': (family_A_tests, family_A_ps, 'Current: all confirmatory'),
        'B_maximal': (tests, family_B_ps, 'Maximal: all 138 (descriptive p=1)'),
        'C_analysis_level': (primary_confirm, family_C_ps, 'Analysis-level: one primary per analysis'),
    }

    all_family_results = {}
    for fam_name, (fam_tests, fam_ps, fam_desc) in families.items():
        print(f"\n{'='*50}")
        print(f"  Family {fam_name}: {fam_desc}")
        print(f"  n = {len(fam_ps)} tests")

        reject, bh_pvals, _, _ = multipletests(fam_ps, alpha=0.05, method='fdr_bh')
        n_sig = int(reject.sum())
        print(f"  Significant after BH (alpha=0.05): {n_sig}/{len(fam_ps)}")

        # Store results per test
        fam_result_list = []
        for i, t in enumerate(fam_tests):
            fam_result_list.append({
                'test_id': t['test_id'],
                'analysis_id': t['analysis_id'],
                'analysis_name': t['analysis_name'],
                'hypothesis': t['hypothesis'],
                'raw_p': float(fam_ps[i]) if not math.isnan(fam_ps[i]) else None,
                'bh_p': float(bh_pvals[i]) if not math.isnan(float(bh_pvals[i])) else None,
                'significant': bool(reject[i]),
            })

        all_family_results[fam_name] = {
            'description': fam_desc,
            'n_tests': len(fam_ps),
            'n_significant': n_sig,
            'tests': fam_result_list,
        }

    # ---- Compare headline inferences across families ----
    print(f"\n{'='*70}")
    print("HEADLINE INFERENCE STABILITY")
    print("=" * 70)

    # Build lookup: test_id -> {family -> significant}
    sig_by_family = {}
    for fam_name, fam_data in all_family_results.items():
        sig_by_family[fam_name] = {}
        for t in fam_data['tests']:
            sig_by_family[fam_name][t['test_id']] = t['significant']

    # Find test IDs by analysis_id + hypothesis substring + optional test_type
    def find_test_ids(analysis_id, hyp_substr, test_type_filter=None):
        ids = []
        for t in tests:
            if t['analysis_id'] == analysis_id and hyp_substr in t['hypothesis']:
                if test_type_filter is None or t['test_type'] == test_type_filter:
                    ids.append(t['test_id'])
        return ids

    stability_table = []
    print(f"\n{'Headline':<50} {'A(conf)':>8} {'B(max)':>8} {'C(prim)':>8} {'Stable':>7}")
    print("-" * 85)

    for label, aid, hyp_substrs, ttype in HEADLINE_CLAIMS:
        # Collect all matching test IDs
        matched_ids = []
        for hsub in hyp_substrs:
            matched_ids.extend(find_test_ids(aid, hsub, ttype))

        sigs = {}
        for fam_name in ['A_confirmatory', 'B_maximal', 'C_analysis_level']:
            fam_sig = sig_by_family[fam_name]
            any_sig = any(fam_sig.get(tid, False) for tid in matched_ids)
            sigs[fam_name] = any_sig

        stable = len(set(sigs.values())) == 1
        a_str = 'sig' if sigs['A_confirmatory'] else 'ns'
        b_str = 'sig' if sigs['B_maximal'] else 'ns'
        c_str = 'sig' if sigs['C_analysis_level'] else 'ns'
        stable_str = 'YES' if stable else '**NO**'

        stability_table.append({
            'headline': label,
            'matched_test_ids': matched_ids,
            'sig_A': sigs['A_confirmatory'],
            'sig_B': sigs['B_maximal'],
            'sig_C': sigs['C_analysis_level'],
            'stable': stable,
        })
        print(f"{label:<50} {a_str:>8} {b_str:>8} {c_str:>8} {stable_str:>7}")

    n_stable = sum(1 for s in stability_table if s['stable'])
    n_headlines = len(stability_table)
    print(f"\n  Stable inferences: {n_stable}/{n_headlines} ({100*n_stable/n_headlines:.0f}%)")

    # ---- Save results ----
    summary = {
        'experiment': 'round11_multiplicity_sensitivity',
        'n_analyses': n_analyses,
        'n_total_tests': n_total,
        'n_confirmatory': n_confirm,
        'n_descriptive': n_descriptive,
        'classification_rule': 'is_confirmatory = (p_value is not None)',
        'families': {
            name: {
                'description': data['description'],
                'n_tests': data['n_tests'],
                'n_significant': data['n_significant'],
            }
            for name, data in all_family_results.items()
        },
        'stability': {
            'n_headlines_tested': n_headlines,
            'n_stable': n_stable,
            'pct_stable': round(100 * n_stable / n_headlines, 1),
            'unstable_headlines': [s['headline'] for s in stability_table if not s['stable']],
        },
        'stability_table': stability_table,
    }

    with open(os.path.join(OUT_DIR, "sensitivity_results.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save full registry
    df = pd.DataFrame(tests)
    df.to_csv(os.path.join(OUT_DIR, "sensitivity_registry.csv"), index=False)

    # Save per-family detailed results
    for fam_name, fam_data in all_family_results.items():
        fam_df = pd.DataFrame(fam_data['tests'])
        fam_df.to_csv(os.path.join(OUT_DIR, f"family_{fam_name}.csv"), index=False)

    print(f"\n  Results saved to {OUT_DIR}")
    print(f"\n{'='*70}")
    print("MULTIPLICITY SENSITIVITY COMPLETE")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    run()
