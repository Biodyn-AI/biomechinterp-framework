# Implementation

Run from repository root:

```bash
python subproject_21_detectability_phase_diagram/implementation/run_detectability_phase_diagram.py
python subproject_21_detectability_phase_diagram/implementation/run_detectability_full_study.py
python subproject_21_detectability_phase_diagram/implementation/run_matched_real_data_calibration.py
```

Outputs are written to:
- `subproject_21_detectability_phase_diagram/outputs/tables/`
- `subproject_21_detectability_phase_diagram/outputs/figures/`
- `subproject_21_detectability_phase_diagram/outputs/full_study/tables/`
- `subproject_21_detectability_phase_diagram/outputs/full_study/figures/`

The matched calibration script additionally writes adversarial-control artifacts:
- cluster-aware bootstrap summaries,
- row-wise vs cluster-aware permutation controls,
- quantile-sensitivity tables and plots,
- leave-one-seed-out stability tables.


## Agent Completion Tracking
- Do this for your project: `tracking/prompt.md`.
- When you finish meaningful work, update `tracking/prompt.md` with the project status (progress, decisions, blockers, and next step).
- Skip the update only if nothing meaningful changed.
