# Implementation

Run from repository root:

```bash
python subproject_11_counterfactual_perturbation_consistency/implementation/scripts/run_counterfactual_consistency_study.py
```

Useful options:

```bash
python subproject_11_counterfactual_perturbation_consistency/implementation/scripts/run_counterfactual_consistency_study.py --runs adamson_submission dixit_submission shifrut_submission --shuffle-n 400 --bootstrap-n 1000
python subproject_11_counterfactual_perturbation_consistency/implementation/scripts/run_counterfactual_consistency_study.py --skip-causal --skip-eval
python subproject_11_counterfactual_perturbation_consistency/implementation/scripts/run_iterator_stress_tests.py
python subproject_11_counterfactual_perturbation_consistency/implementation/scripts/generate_multiseed_submission_configs.py --seeds 43 44
python subproject_11_counterfactual_perturbation_consistency/implementation/scripts/analyze_multiseed_consistency.py --seeds 42 43 44
python subproject_11_counterfactual_perturbation_consistency/implementation/scripts/analyze_multiseed_causal_pair_variance.py --seeds 42 43 44
```

Generated artifacts:
- Tables: `subproject_11_counterfactual_perturbation_consistency/outputs/tables/`
- Figures: `subproject_11_counterfactual_perturbation_consistency/outputs/figures/`
- Report: `subproject_11_counterfactual_perturbation_consistency/reports/counterfactual_perturbation_consistency_report.md`
- Timepoint robustness: `subproject_11_counterfactual_perturbation_consistency/outputs/tables/timepoint_consistency.tsv`
- Matched-pair exports: `subproject_11_counterfactual_perturbation_consistency/outputs/tables/matched_pairs_*.tsv`
- Iterator stress tests:
  - `subproject_11_counterfactual_perturbation_consistency/outputs/tables/iterator_null_tail_diagnostics.tsv`
  - `subproject_11_counterfactual_perturbation_consistency/outputs/tables/iterator_effect_decile_tests.tsv`
  - `subproject_11_counterfactual_perturbation_consistency/outputs/tables/iterator_timepoint_topk_overlap.tsv`
  - `subproject_11_counterfactual_perturbation_consistency/outputs/tables/iterator_reference_coverage.tsv`
- Multi-seed summaries:
  - `subproject_11_counterfactual_perturbation_consistency/outputs/tables/multiseed_seed_metrics.tsv`
  - `subproject_11_counterfactual_perturbation_consistency/outputs/tables/multiseed_metric_summary.tsv`
  - `subproject_11_counterfactual_perturbation_consistency/outputs/tables/multiseed_causal_pair_variance.tsv`
  - `subproject_11_counterfactual_perturbation_consistency/outputs/tables/multiseed_causal_pair_variance_summary.tsv`


## Agent Completion Tracking
- Do this for your project: `tracking/prompt.md`.
- When you finish meaningful work, update `tracking/prompt.md` with the project status (progress, decisions, blockers, and next step).
- Skip the update only if nothing meaningful changed.
