# Implementation

Probe sweeps and cross-tissue aggregation are implemented in `single_cell_mechinterp/scripts/`.

Subproject-specific analysis utilities:
- `analyze_transfer_artifacts.py`: derives manuscript CSV artifacts from committed benchmark summaries.
- `analyze_confound_uncertainty.py`: adds donor/sample composition sensitivity and bootstrap uncertainty tables.
- `analyze_iterator_stress_tests.py`: adds adversarial stress tests (exact permutation, leave-one-target robustness, policy-uplift CI, interpretability validity checks).
- `make_paper_figures.py`: regenerates figures (`fig01`-`fig11`) from artifact CSVs.


## Agent Completion Tracking
- Do this for your project: `tracking/prompt.md`.
- When you finish meaningful work, update `tracking/prompt.md` with the project status (progress, decisions, blockers, and next step).
- Skip the update only if nothing meaningful changed.
