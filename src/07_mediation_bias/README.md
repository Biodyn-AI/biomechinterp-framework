# Implementation

Entry points:
- Full paper pipeline: `python subproject_22_patching_mediation_bias_bounds/implementation/scripts/run_full_paper_pipeline.py`
- Initial proposal run (kept for compatibility): `python subproject_22_patching_mediation_bias_bounds/implementation/scripts/run_proposal5_analysis.py`
- Compile manuscript PDF: `cd subproject_22_patching_mediation_bias_bounds/reports && pandoc final_paper_bias_bounds_mediation.md -o final_paper_bias_bounds_mediation.pdf --from gfm --pdf-engine=pdflatex`

Note:
- `run_full_paper_pipeline.py` does not overwrite an existing report by default.
- Use `--overwrite-report` if you explicitly want to regenerate the report text.

The full paper pipeline loads existing mediation artifacts from:
- `subproject_09_causal_mediation_circuit_map/implementation/results/*/*/checkpoint.json`
- `subproject_09_causal_mediation_circuit_map/implementation/results/*/*/causal_scores.tsv`
- `subproject_09_causal_mediation_circuit_map/implementation/results/*/*/circuit_maps.tsv`

It writes analysis artifacts to:
- `subproject_22_patching_mediation_bias_bounds/outputs/tables/`
- `subproject_22_patching_mediation_bias_bounds/outputs/figures/`
- `subproject_22_patching_mediation_bias_bounds/reports/final_paper_bias_bounds_mediation.md`


## Agent Completion Tracking
- Do this for your project: `tracking/prompt.md`.
- When you finish meaningful work, update `tracking/prompt.md` with the project status (progress, decisions, blockers, and next step).
- Skip the update only if nothing meaningful changed.
