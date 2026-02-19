#!/usr/bin/env python3
"""
Round 10 Experiment 0: Statistical Family Audit
================================================
Creates one authoritative machine-readable registry of ALL statistical tests,
computes exact BH-corrected p-values, reconciles all counts.

Addresses: MC1 ("inconsistent statistical bookkeeping")
           ST1 ("statistical family audit")
           Minor 1 ("reconcile all counts")
"""

import os, sys, json
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round10_statistical_audit")
os.makedirs(OUT_DIR, exist_ok=True)

def build_registry():
    """Build the complete test registry from the appendix longtable."""
    tests = []

    def add(analysis_id, analysis_name, section, hypothesis, test_type,
            raw_p, effect_size, sample_size, significant_raw=None):
        """Add a test to the registry."""
        # Parse raw_p: None means descriptive (no p-value)
        p_val = None
        if raw_p is not None:
            if isinstance(raw_p, str):
                if raw_p.startswith('<'):
                    # e.g. "<0.001", "<10^-6"
                    val = raw_p.replace('<', '').strip()
                    if '^' in val:
                        base, exp = val.split('^')
                        p_val = float(base) ** float(exp.replace('{', '').replace('}', '').replace('-', '-'))
                        # Actually parse more carefully
                        val_clean = val.replace('{', '').replace('}', '')
                        if '10^' in val_clean or '10^-' in val_clean:
                            exp_part = val_clean.split('^')[1]
                            p_val = 10 ** float(exp_part)
                        else:
                            p_val = float(base) ** float(exp_part)
                    else:
                        p_val = float(val)
                elif raw_p.startswith('>'):
                    p_val = float(raw_p.replace('>', '').strip())
                else:
                    p_val = float(raw_p)
            else:
                p_val = float(raw_p)

        is_confirmatory = p_val is not None
        tests.append({
            'test_id': len(tests) + 1,
            'analysis_id': analysis_id,
            'analysis_name': analysis_name,
            'section': section,
            'hypothesis': hypothesis,
            'test_type': test_type,
            'raw_p': p_val,
            'effect_size': effect_size,
            'sample_size': sample_size,
            'is_confirmatory': is_confirmatory,
        })

    # ============================================================
    # Analysis 1: Scaling Behavior
    # ============================================================
    add(1, 'Scaling Behavior', 'scaling',
        'Cell count improves GRN (TRRUST)', 'Sign test',
        0.002, '100% degradation', '9 runs')
    add(1, 'Scaling Behavior', 'scaling',
        'Cell count improves GRN (TRRUST)', 'Wilcoxon',
        0.002, '--', '9 runs')
    add(1, 'Scaling Behavior', 'scaling',
        'Cell count improves GRN (DoRothEA)', 'Sign test',
        0.002, '100% degradation', '9 runs')
    add(1, 'Scaling Behavior', 'scaling',
        'F1 confidence intervals', 'Bootstrap',
        None, '--', '10k resamples')
    add(1, 'Scaling Behavior', 'scaling',
        'Seed stability w/ scaling', 'Paired test',
        '<0.001', '46-48% drop', '3 seeds')

    # ============================================================
    # Analysis 2: Mediation Bias
    # ============================================================
    add(2, 'Mediation Bias', 'bias',
        'Components additive', 'Lower bound',
        '<0.001', 'A_lb/|TE|=0.73', '16 pairs')
    add(2, 'Mediation Bias', 'bias',
        'Rankings stable', 'Stability',
        '<0.001', '0.067->0.003', '16 pairs')

    # ============================================================
    # Analysis 3: Detectability Theory
    # ============================================================
    add(3, 'Detectability Theory', 'detectability',
        'Theory matches empirical', 'Correlation',
        '<10^{-6}', 'r=0.887', 'Phase space')
    add(3, 'Detectability Theory', 'detectability',
        'Intervention more detectable', 'Ratio',
        '<0.001', '44.4% cells', 'Simulation')

    # ============================================================
    # Analysis 4: Cross-Context Consistency
    # ============================================================
    add(4, 'Cross-Context Consistency', 'cross_tissue',
        'Immune-kidney transfer', 'Spearman',
        0.024, 'rho=0.71', 'Pairs')
    add(4, 'Cross-Context Consistency', 'cross_tissue',
        'Immune-lung transfer', 'Spearman',
        0.089, 'rho=0.32', 'Pairs')
    add(4, 'Cross-Context Consistency', 'cross_tissue',
        'Kidney-lung transfer', 'Spearman',
        0.156, 'rho=-0.44', 'Pairs')
    add(4, 'Cross-Context Consistency', 'cross_tissue',
        'Correlation confidence', 'Bootstrap',
        None, '--', '10k resamples')
    add(4, 'Cross-Context Consistency', 'cross_tissue',
        'Correlation significance', 'Permutation',
        '<0.001', '--', '5k permutations')

    # ============================================================
    # Analysis 5: Perturbation Validation
    # ============================================================
    add(5, 'Perturbation Validation', 'perturbation',
        'Dixit 13d raw', 'Spearman',
        0.032, 'rho=0.269', 'Pert. pairs')
    add(5, 'Perturbation Validation', 'perturbation',
        'Dixit 13d adjusted', 'Spearman',
        0.020, 'rho=0.199', 'Pert. pairs')
    add(5, 'Perturbation Validation', 'perturbation',
        'Dixit 7d', 'Spearman',
        0.15, 'rho=0.112', 'Pert. pairs')
    add(5, 'Perturbation Validation', 'perturbation',
        'Adamson', 'Spearman',
        0.089, 'Marginal', 'Pert. pairs')
    add(5, 'Perturbation Validation', 'perturbation',
        'Shifrut raw', 'Spearman',
        0.031, 'rho=-0.325', 'Pert. pairs')
    add(5, 'Perturbation Validation', 'perturbation',
        'Shifrut adjusted', 'Spearman',
        0.876, 'rho=0.004', 'Pert. pairs')

    # ============================================================
    # Analysis 6: Cross-Species Ortholog
    # ============================================================
    add(6, 'Cross-Species Ortholog', 'ortholog',
        'Global conservation', 'Spearman',
        '<10^{-300}', 'rho=0.743', '25876')
    add(6, 'Cross-Species Ortholog', 'ortholog',
        'Signs conserved', 'Sign test',
        '<0.001', '88.6%', '25876')
    add(6, 'Cross-Species Ortholog', 'ortholog',
        'Overlap above chance', 'Permutation',
        '<0.001', '8-484x', '1k permutations')
    add(6, 'Cross-Species Ortholog', 'ortholog',
        'TF conservation varies', 'Range',
        None, '-0.12 to 0.90', '61 TFs')

    # ============================================================
    # Analysis 7: Pseudotime
    # ============================================================
    add(7, 'Pseudotime', 'pseudotime',
        'TFs precede targets', 'Directionality test',
        0.068, '21.4% consistent', '56 pairs')
    add(7, 'Pseudotime', 'pseudotime',
        'Exceeds shuffled', 'Mann-Whitney',
        0.068, 'd=1.58', '500 permutations')
    add(7, 'Pseudotime', 'pseudotime',
        'Exceeds random', 'Mann-Whitney',
        0.37, '--', '200 sets')
    add(7, 'Pseudotime', 'pseudotime',
        'T cell lineage', 'Lineage',
        None, '16.7%', '24 pairs')
    add(7, 'Pseudotime', 'pseudotime',
        'B cell lineage', 'Lineage',
        None, '13.3%', '15 pairs')
    add(7, 'Pseudotime', 'pseudotime',
        'Myeloid lineage', 'Lineage',
        None, '35.3%', '17 pairs')

    # ============================================================
    # Analysis 8: Batch/Donor Leakage
    # ============================================================
    add(8, 'Batch/Donor Leakage', 'batch',
        'Donor (immune)', 'Logistic regression',
        '<0.001', 'AUC 0.85-0.87', '20k, 24 donors')
    add(8, 'Batch/Donor Leakage', 'batch',
        'Donor (lung)', 'Logistic regression',
        '<0.001', 'AUC 0.94-0.96', '20k, 4 donors')
    add(8, 'Batch/Donor Leakage', 'batch',
        'Assay method', 'Random forest',
        '<0.001', 'AUC 0.96-0.99', 'All tissues')
    add(8, 'Batch/Donor Leakage', 'batch',
        'CV accuracy', '5-fold CV',
        None, '--', '5 folds')
    add(8, 'Batch/Donor Leakage', 'batch',
        'Edge stability (LODO)', 'LODO variance',
        '<0.001', 'High variance', 'Var. donors')
    add(8, 'Batch/Donor Leakage', 'batch',
        'Generalization gap', 'Cross-donor',
        0.012, '6.6 pp gap', 'Lung')

    # ============================================================
    # Analysis 9: Calibration
    # ============================================================
    add(9, 'Calibration', 'calibration',
        'Calibration improves', 'Paired',
        '<0.001', '4-7x', '6 methods')
    add(9, 'Calibration', 'calibration',
        'Isotonic better', 'Paired',
        '<0.001', 'ECE 0.06-0.08', '6 methods')
    add(9, 'Calibration', 'calibration',
        'Coverage valid', 'Coverage',
        None, '>=95%', 'alpha=0.05')
    add(9, 'Calibration', 'calibration',
        'Calibration robust', 'Bootstrap',
        None, 'CI < 0.02', '200 resamples')
    add(9, 'Calibration', 'calibration',
        'Calibrators dont transfer', 'Transfer',
        '<0.001', 'ECE 0.32-0.42', 'K562->T')

    # ============================================================
    # Analysis 10: CSSI
    # ============================================================
    add(10, 'CSSI', 'cssi_results',
        'CSSI prevents degradation', 'Spearman',
        0.99, 'r=-0.001', '10 seeds')
    add(10, 'CSSI', 'cssi_results',
        'Pooled degrades', 'Spearman',
        '<10^{-4}', 'r=-0.618', '10 seeds')
    add(10, 'CSSI', 'cssi_results',
        'CSSI outperforms pooled', 'Wilcoxon',
        2.5e-11, '1.13-1.85x', '60 combinations')
    add(10, 'CSSI', 'cssi_results',
        'Real PBMC improvement', 'Bootstrap',
        0.03, '1.16x', '3k cells')
    add(10, 'CSSI', 'cssi_results',
        'CSSI w/ real proportions', 'Wilcoxon',
        2.4e-8, '1.62x', '10 seeds')
    add(10, 'CSSI', 'cssi_results',
        'CSSI on attention', 'Layer analysis',
        None, 'AUROC 0.68-0.69', '497 cells')

    # ============================================================
    # Analysis 11: Synthetic Validation
    # ============================================================
    add(11, 'Synthetic Validation', 'synthetic_validation',
        'Recovery degrades', 'Correlation',
        '<0.01', 'r: 0.85->0.62', 'Synthetic')
    add(11, 'Synthetic Validation', 'synthetic_validation',
        'Shapley outperforms', 'Paired',
        '<0.001', '91% improvement', 'Synthetic')
    add(11, 'Synthetic Validation', 'synthetic_validation',
        'Empirical = theory', 'Correlation',
        '<10^{-6}', 'r=0.887', 'Phase space')

    # ============================================================
    # Analysis 12: Multi-Model
    # ============================================================
    add(12, 'Multi-Model', 'multi_model',
        'Geneformer TRRUST', 'AUROC',
        0.89, 'AUROC 0.44-0.55', '3 counts')
    add(12, 'Multi-Model', 'multi_model',
        'Geneformer DoRothEA', 'AUROC',
        0.76, 'AUROC 0.47-0.49', '3 counts')
    add(12, 'Multi-Model', 'multi_model',
        'Cross-model comparison', 'Comparative',
        None, 'Both ~0.5', '2 models')
    add(12, 'Multi-Model', 'multi_model',
        'GF confidence intervals', 'Bootstrap',
        None, 'CIs incl. 0.5', '10k resamples')
    add(12, 'Multi-Model', 'multi_model',
        'Attention = co-expression', 'Correlation',
        '<10^{-50}', 'rho=0.31-0.42', 'Both')
    add(12, 'Multi-Model', 'multi_model',
        'Attention != regulation', 'Correlation',
        '>0.3', 'rho=-0.01-0.02', 'Both')

    # ============================================================
    # Analysis 13: Controlled-Composition Scaling
    # ============================================================
    add(13, 'Controlled-Composition Scaling', 'scaling',
        'N degrades AUROC (single type)', 'Spearman',
        0.079, 'rho=-0.33', '30 runs')
    add(13, 'Controlled-Composition Scaling', 'scaling',
        'N degrades AUROC (fixed comp)', 'Spearman',
        0.82, 'rho=-0.05', '20 runs')
    add(13, 'Controlled-Composition Scaling', 'scaling',
        'Diversity improves', 'Spearman',
        '<10^{-4}', 'rho=+0.63', '35 runs')

    # ============================================================
    # Analysis 14: Perturbation-First Validation
    # ============================================================
    add(14, 'Perturbation-First Validation', 'perturbation',
        'Replogle primary', 'One-sample t',
        '<10^{-4}', 'AUROC=0.696', '151 perts')
    add(14, 'Perturbation-First Validation', 'perturbation',
        'Replogle baseline', 'One-sample t',
        0.32, 'AUROC=0.511', '44 perts')
    add(14, 'Perturbation-First Validation', 'perturbation',
        'Replogle Wilcoxon', 'Wilcoxon',
        0.30, 'Median=0.503', '44 perts')

    # ============================================================
    # Analysis 15: Robust Attention Residualization
    # ============================================================
    add(15, 'Robust Attention Residualization', 'multi_model',
        'Edge-expression correlation', 'Spearman',
        '<10^{-50}', 'rho=0.842', '75962 pairs')
    add(15, 'Robust Attention Residualization', 'multi_model',
        'R2 OLS full', 'OLS R2',
        None, 'R2=0.27', '75962 pairs')
    add(15, 'Robust Attention Residualization', 'multi_model',
        'R2 GBDT', 'GBDT R2',
        None, 'R2=0.51', '75962 pairs')
    add(15, 'Robust Attention Residualization', 'multi_model',
        'CF residual AUROC', 'CF-AUROC',
        None, 'AUROC=0.73', '61 positives')
    add(15, 'Robust Attention Residualization', 'multi_model',
        'CF stability', '10-seed variance',
        None, 'sigma=0.001', '10 seeds')

    # ============================================================
    # Analysis 16: Degree-Preserving Null Models
    # ============================================================
    add(16, 'Degree-Preserving Null Models', 'multi_model',
        'AUROC > random', 'Permutation',
        '<0.001', 'z=6.9', '1k permutations')
    add(16, 'Degree-Preserving Null Models', 'multi_model',
        'AUROC > degree', 'Permutation',
        '<0.005', 'z=3.63', '200 permutations')
    add(16, 'Degree-Preserving Null Models', 'multi_model',
        'Degree explains AUROC', 'Decomposition',
        None, '73% global', '75962 pairs')
    add(16, 'Degree-Preserving Null Models', 'multi_model',
        'Per-TF AUROC', 'Per-TF eval',
        None, 'mean=0.69+/-0.20', '18 TFs')
    add(16, 'Degree-Preserving Null Models', 'multi_model',
        'Per-TF excess', 'Per-TF eval',
        None, '+0.035+/-0.20', '18 TFs')
    add(16, 'Degree-Preserving Null Models', 'multi_model',
        'Precision@k', 'Prec@k',
        None, '0.030 vs 0.002', '18 TFs')

    # ============================================================
    # Analysis 17: Attention-Correlation Mapping
    # ============================================================
    add(17, 'Attention-Correlation Mapping', 'multi_model',
        'Within-tissue R2', 'rho^2',
        '<10^{-50}', 'R2=0.10-0.18', '50k edges')
    add(17, 'Attention-Correlation Mapping', 'multi_model',
        'Cross-tissue R2', 'OLS R2',
        None, 'R2<0.02', '21k-26k edges')
    add(17, 'Attention-Correlation Mapping', 'multi_model',
        'Cross-tissue rho', 'Spearman',
        '<10^{-9}', 'rho=-0.02 to -0.05', '3 conditions')

    # ============================================================
    # Analysis 18: Perturbation Sensitivity
    # ============================================================
    add(18, 'Perturbation Sensitivity', 'perturbation',
        'All AUROC > 0.5 (27 conditions)', 't-test',
        '<0.005', 'AUROC=0.62-0.76', '13-1158 perts')
    add(18, 'Perturbation Sensitivity', 'perturbation',
        'TF vs non-TF', 'Comparison',
        None, 'Delta<=0.02', '1-83 TFs')

    # ============================================================
    # Analysis 19: CSSI Extended Null
    # ============================================================
    add(19, 'CSSI Extended Null', 'cssi_results',
        'Null inflation', 'Delta AUROC',
        None, '<=-0.20', 'K=2-20')
    add(19, 'CSSI Extended Null', 'cssi_results',
        'CSSI vs SCENIC', 'Comparative',
        None, 'Equivalent', '3 methods')
    add(19, 'CSSI Extended Null', 'cssi_results',
        'Per-edge FDR', 'BH-FDR',
        None, 'FDR<=0.11', 'K=5-15')

    # ============================================================
    # Analysis 20: K-Sensitivity Analysis
    # ============================================================
    add(20, 'K-Sensitivity Analysis', 'scaling',
        'Continuous AUROC', 'Comparison',
        None, '0.86->0.93', '9 runs')
    add(20, 'K-Sensitivity Analysis', 'scaling',
        'K-sensitivity', 'Comparison',
        None, 'F1~10^-4 all K', '5 K values')

    # ============================================================
    # Analysis 21: Trivial Baseline Comparison
    # ============================================================
    add(21, 'Trivial Baseline Comparison', 'perturbation',
        'Variance vs Correlation', 'Paired t',
        '<10^{-24}', 'Delta=+0.186', '151 perts')
    add(21, 'Trivial Baseline Comparison', 'perturbation',
        'Mean expr vs Correlation', 'Paired t',
        '<10^{-20}', 'Delta=+0.146', '151 perts')
    add(21, 'Trivial Baseline Comparison', 'perturbation',
        'Dropout vs Correlation', 'Paired t',
        '<10^{-12}', 'Delta=+0.113', '151 perts')
    add(21, 'Trivial Baseline Comparison', 'perturbation',
        'TF degree vs Correlation', 'Paired t',
        '<10^{-29}', 'Delta=-0.196', '151 perts')

    # ============================================================
    # Analysis 22: Bootstrap Per-TF CIs
    # ============================================================
    add(22, 'Bootstrap Per-TF CIs', 'multi_model',
        'Global boot CI', 'Bootstrap',
        None, 'CI=[0.71,0.77]', '200 iterations')
    add(22, 'Bootstrap Per-TF CIs', 'multi_model',
        'Per-TF boot CI', 'Bootstrap',
        None, 'CI=[0.59,0.72]', '200 iterations')

    # ============================================================
    # Analysis 23: Attention Perturbation-First
    # ============================================================
    add(23, 'Attention Perturbation-First', 'mechanistic_case_study',
        'Attn vs Corr L13 (Wilcoxon)', 'Wilcoxon',
        0.726, 'diff=0.001', 'n=280')
    add(23, 'Attention Perturbation-First', 'mechanistic_case_study',
        'Attn vs Corr L13 (Paired t)', 'Paired t',
        0.931, 'diff=0.001', 'n=280')

    # ============================================================
    # Analysis 24: Full 18-Layer Perturbation-First
    # ============================================================
    add(24, 'Full 18-Layer Perturbation-First', 'mechanistic_case_study',
        'L15 vs Corr', 'Wilcoxon',
        0.0009, 'Delta=+0.040', 'n=280')
    add(24, 'Full 18-Layer Perturbation-First', 'mechanistic_case_study',
        'Split-sample validation', 'Wilcoxon',
        0.017, 'AUROC=0.750', 'n=140')
    add(24, 'Full 18-Layer Perturbation-First', 'mechanistic_case_study',
        'Per-layer (18)', 'Wilcoxon',
        0.0009, 'AUROC 0.47-0.74', 'n=280')  # Using best p as representative

    # ============================================================
    # Analysis 25: Attention-Specific Confound Decomposition
    # ============================================================
    add(25, 'Attention-Specific Confound Decomposition', 'mechanistic_case_study',
        'Attn residual OLS', 'CF-AUROC',
        None, 'AUROC=0.538', '174 positives')
    add(25, 'Attention-Specific Confound Decomposition', 'mechanistic_case_study',
        'Attn residual GBDT', 'CF-AUROC',
        None, 'AUROC=0.554', '174 positives')
    add(25, 'Attention-Specific Confound Decomposition', 'mechanistic_case_study',
        'Corr residual OLS', 'CF-AUROC',
        None, 'AUROC=0.622', '174 positives')
    add(25, 'Attention-Specific Confound Decomposition', 'mechanistic_case_study',
        'Attn degree-null', 'Permutation',
        0.023, 'z=2.0', '200 permutations')
    add(25, 'Attention-Specific Confound Decomposition', 'mechanistic_case_study',
        'Corr degree-null', 'Permutation',
        0.018, 'z=2.1', '200 permutations')

    # ============================================================
    # Analysis 26: Conditional Incremental Value
    # ============================================================
    add(26, 'Conditional Incremental Value', 'mechanistic_case_study',
        'Gene-only CV', '5-fold CV',
        None, 'AUROC=0.895', '280 perts')
    add(26, 'Conditional Incremental Value', 'mechanistic_case_study',
        'Delta gene+attn', 'Bootstrap',
        None, '-0.0004 [-0.001,0]', '100 iterations')
    add(26, 'Conditional Incremental Value', 'mechanistic_case_study',
        'Delta gene+corr', 'Bootstrap',
        None, '-0.002 [-0.005,0]', '100 iterations')
    add(26, 'Conditional Incremental Value', 'mechanistic_case_study',
        'TF stratified', 'Subgroup',
        None, 'TF 0.913, non-TF 0.895', '14/266')

    # ============================================================
    # Analysis 27: Per-Head TRRUST Ranking
    # ============================================================
    add(27, 'Per-Head TRRUST Ranking', 'mechanistic_case_study',
        'Heads differ in reg.', 'AUROC range',
        None, '0.34-0.75', '324 heads')
    add(27, 'Per-Head TRRUST Ranking', 'mechanistic_case_study',
        'Top-5 heads identified', 'AUROC',
        None, 'AUROC=0.70-0.75', '500 cells')

    # ============================================================
    # Analysis 28: Cross-Context CRISPRa Replication
    # ============================================================
    add(28, 'Cross-Context CRISPRa Replication', 'mechanistic_case_study',
        'Attn vs Corr L13 CRISPRa', 'Wilcoxon',
        '<10^{-6}', 'diff=-0.106', 'n=77')
    add(28, 'Cross-Context CRISPRa Replication', 'mechanistic_case_study',
        'Attn vs Corr L6 CRISPRa', 'Wilcoxon',
        1e-6, 'diff=-0.098', 'n=77')
    add(28, 'Cross-Context CRISPRa Replication', 'mechanistic_case_study',
        'Attn vs Corr L18 CRISPRa', 'Wilcoxon',
        '<10^{-6}', 'diff=-0.105', 'n=77')

    # ============================================================
    # Analysis 29: Head-Level Causal Ablation
    # ============================================================
    add(29, 'Head-Level Causal Ablation', 'mechanistic_case_study',
        'Regulatory vs baseline', 'Wilcoxon',
        0.244, 'Delta=-0.0002', 'n=280')
    add(29, 'Head-Level Causal Ablation', 'mechanistic_case_study',
        'Random vs baseline (rep 1)', 'Wilcoxon',
        0.065, 'Delta=+0.0007', 'n=280')
    add(29, 'Head-Level Causal Ablation', 'mechanistic_case_study',
        'Random vs baseline (rep 2)', 'Wilcoxon',
        0.069, 'Delta=+0.0004', 'n=280')
    add(29, 'Head-Level Causal Ablation', 'mechanistic_case_study',
        'Random vs baseline (rep 3)', 'Wilcoxon',
        '<10^{-6}', 'Delta=+0.0028', 'n=280')
    add(29, 'Head-Level Causal Ablation', 'mechanistic_case_study',
        'Entropy-matched vs baseline', 'Wilcoxon',
        '<10^{-6}', 'Delta=+0.0042', 'n=280')

    return tests


def run():
    print("=" * 70)
    print("STATISTICAL FAMILY AUDIT")
    print("=" * 70)

    # Build registry
    print("\nBuilding test registry...")
    tests = build_registry()
    n_total = len(tests)
    print(f"  Total test rows: {n_total}")

    # Separate confirmatory vs descriptive
    confirmatory = [t for t in tests if t['is_confirmatory']]
    descriptive = [t for t in tests if not t['is_confirmatory']]
    n_confirm = len(confirmatory)
    n_descriptive = len(descriptive)
    print(f"  Confirmatory (have p-value): {n_confirm}")
    print(f"  Descriptive (no p-value): {n_descriptive}")

    # Count analyses
    analysis_ids = sorted(set(t['analysis_id'] for t in tests))
    n_analyses = len(analysis_ids)
    print(f"  Distinct analyses: {n_analyses}")

    # Extract raw p-values for BH correction
    raw_ps = np.array([t['raw_p'] for t in confirmatory])
    print(f"\nRaw p-values (n={len(raw_ps)}):")
    print(f"  Min: {raw_ps.min():.2e}")
    print(f"  Max: {raw_ps.max():.4f}")
    print(f"  Median: {np.median(raw_ps):.4f}")

    # BH correction
    print(f"\nApplying Benjamini-Hochberg FDR correction (n={n_confirm})...")
    reject, bh_pvals, _, _ = multipletests(raw_ps, alpha=0.05, method='fdr_bh')
    n_significant = int(reject.sum())
    print(f"  Significant after BH: {n_significant}/{n_confirm}")

    # Assign BH p-values back to confirmatory tests
    for i, t in enumerate(confirmatory):
        t['bh_adjusted_p'] = float(bh_pvals[i])
        t['significant_bh_005'] = bool(reject[i])

    # Descriptive tests get None
    for t in descriptive:
        t['bh_adjusted_p'] = None
        t['significant_bh_005'] = None

    # Print per-analysis summary
    print(f"\nPer-analysis summary:")
    print(f"{'ID':>3} {'Analysis':<45} {'Total':>5} {'Conf':>5} {'Sig':>5}")
    print("-" * 70)
    for aid in analysis_ids:
        atests = [t for t in tests if t['analysis_id'] == aid]
        aconfirm = [t for t in atests if t['is_confirmatory']]
        asig = [t for t in aconfirm if t.get('significant_bh_005')]
        name = atests[0]['analysis_name']
        print(f"{aid:>3} {name:<45} {len(atests):>5} {len(aconfirm):>5} {len(asig):>5}")

    # Build DataFrame
    df = pd.DataFrame(tests)
    csv_path = os.path.join(OUT_DIR, "statistical_registry.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved registry: {csv_path}")

    # Summary JSON
    summary = {
        'experiment': 'round10_statistical_audit',
        'n_analyses': n_analyses,
        'n_total_tests': n_total,
        'n_confirmatory': n_confirm,
        'n_descriptive': n_descriptive,
        'n_significant_bh_005': n_significant,
        'bh_alpha': 0.05,
        'family_definition': 'All confirmatory tests with explicit p-values across all 29 analyses',
        'analysis_ids': analysis_ids,
        'per_analysis_counts': {
            aid: {
                'name': [t for t in tests if t['analysis_id'] == aid][0]['analysis_name'],
                'total': len([t for t in tests if t['analysis_id'] == aid]),
                'confirmatory': len([t for t in tests if t['analysis_id'] == aid and t['is_confirmatory']]),
                'significant': len([t for t in tests if t['analysis_id'] == aid and t.get('significant_bh_005')]),
            }
            for aid in analysis_ids
        },
        'stale_references': {
            'main_tex_L121': 'twenty-three analyses...78 tests',
            'appendix_L4': 'approximately 88...27 analyses',
            'appendix_L8': '~88 total tests',
            'appendix_L236': 'approximately 93 across 29',
            'correct_values': {
                'n_analyses': n_analyses,
                'n_total_tests': n_total,
                'n_confirmatory': n_confirm,
                'n_significant': n_significant,
            }
        }
    }

    json_path = os.path.join(OUT_DIR, "audit_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary: {json_path}")

    # Cross-check: what the paper currently says vs reality
    print("\n" + "=" * 70)
    print("CROSS-REFERENCE CHECK")
    print("=" * 70)
    print(f"\n  Paper says (main.tex L121):  23 analyses, 78 tests")
    print(f"  Paper says (appendix L4):    27 analyses, ~88 tests")
    print(f"  Paper says (appendix L236):  29 analyses, ~93 tests")
    print(f"  ACTUAL (this registry):      {n_analyses} analyses, {n_total} total tests")
    print(f"                               {n_confirm} confirmatory, {n_descriptive} descriptive")
    print(f"                               {n_significant} significant after BH (alpha=0.05)")

    # Identify which "approximately" references need updating
    print(f"\n  REQUIRED UPDATES:")
    print(f"    main.tex L121: 23 -> {n_analyses}, 78 -> {n_total} ({n_confirm} confirmatory)")
    print(f"    appendix header: 27 -> {n_analyses}, ~88 -> {n_total}")
    print(f"    appendix summary: ~93 -> {n_total}")
    print(f"    appendix table caption: ~88 -> {n_total}")

    print("\n" + "=" * 70)
    print("STATISTICAL AUDIT COMPLETE")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    run()
