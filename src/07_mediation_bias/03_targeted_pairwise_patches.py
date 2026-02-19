#!/usr/bin/env python3
"""Run targeted pairwise patching experiments for subproject 22.

This script performs direct two-component patch measurements on selected
run/pair/component combinations. It is designed to close the primary paper-quality
blocker: replacing purely assumption-based structural interaction budgets with
measured interaction terms on the exact unstable real-data pairs.

Outputs:
- per-cell measurements (raw and normalized interaction effects)
- per-component-pair summary statistics
- per-run/pair summary statistics (max/mean interaction magnitude)
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader


EPS = 1e-12
Z95 = 1.96


def find_repo_root(start: Path) -> Path:
    """Infer repository root from stable top-level directories."""
    for candidate in [start] + list(start.parents):
        if (candidate / "market_research").exists() and (candidate / "single_cell_mechinterp").exists():
            return candidate
    raise RuntimeError("Could not infer repository root")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
SC_ROOT = REPO_ROOT / "single_cell_mechinterp"

# Import from the existing scGPT intervention stack.
sys.path.insert(0, str(SC_ROOT))
sys.path.insert(0, str(SC_ROOT / "scripts"))

from run_causal_interventions import (  # type: ignore  # noqa: E402
    _build_pair_list,
    _build_scgpt_model_args,
    _capture_clean_outputs,
    _device,
    _load_scgpt_args,
    _readout_from_outputs,
    _readout_value,
)
from src.data.scgpt_dataset import ScGPTDataset, ScGPTDatasetConfig, collate_scgpt  # type: ignore  # noqa: E402
from src.eval.gene_symbols import load_hgnc_alias_map, normalize_gene_names  # type: ignore  # noqa: E402
from src.interpret.causal_intervention import (  # type: ignore  # noqa: E402
    apply_pad_ablation,
    apply_value_ablation,
    attention_head_slice,
    find_attention_modules,
    find_gene_positions,
    find_mlp_modules,
    find_transformer_layers,
    patch_module_output,
)
from src.model.scgpt_loader import load_scgpt_model  # type: ignore  # noqa: E402
from src.model.vocab import load_vocab  # type: ignore  # noqa: E402
from src.model.wrapper import ScGPTWrapper  # type: ignore  # noqa: E402
from src.utils.config import load_config  # type: ignore  # noqa: E402
from src.utils.torch_utils import move_to_device  # type: ignore  # noqa: E402


@dataclass(frozen=True)
class ComponentSpec:
    """Specification for one patchable mediator component."""

    key: str
    granularity: str
    layer_idx: int
    head_idx: int | None
    module_idx: int
    head_slice: Tuple[int, int] | None


def resolve_input_path(path_value: str | Path, config_dir: Path, repo_root: Path, sc_root: Path) -> Path:
    """Resolve config-provided input path against known execution roots.

    Existing configs were authored for execution from `single_cell_mechinterp/`,
    while this script is run from repository root. We therefore try multiple bases
    in a deterministic order and return the first existing path.
    """
    raw = Path(path_value)
    if raw.is_absolute() and raw.exists():
        return raw

    candidates = [
        raw,
        config_dir / raw,
        repo_root / raw,
        sc_root / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Return config-dir-relative fallback so downstream errors still show a
    # meaningful path even when the target file is genuinely missing.
    return (config_dir / raw).resolve()


def parse_component_key(key: str) -> Tuple[str, int, int | None]:
    """Parse component key in canonical form used by this subproject.

    Supported keys:
    - head:<layer>:<head>
    - mlp:<layer>
    """
    parts = str(key).split(":")
    ctype = parts[0].strip().lower()
    if ctype == "head":
        if len(parts) != 3:
            raise ValueError(f"Invalid head component key: {key}")
        return ctype, int(parts[1]), int(parts[2])
    if ctype == "mlp":
        if len(parts) != 2:
            raise ValueError(f"Invalid mlp component key: {key}")
        return ctype, int(parts[1]), None
    raise ValueError(f"Unsupported component key type: {key}")


def build_component_spec(
    component_key: str,
    granularity: str,
    attn_modules: List[torch.nn.Module],
    mlp_modules: List[torch.nn.Module],
) -> ComponentSpec:
    """Build a patching spec from a serialized component key."""
    ctype, layer_idx, head_idx = parse_component_key(component_key)

    if granularity == "heads":
        if ctype != "head":
            raise ValueError(f"Expected head key for heads run, got {component_key}")
        if layer_idx < 0 or layer_idx >= len(attn_modules):
            raise ValueError(f"Head layer index out of range: {component_key}")
        module = attn_modules[layer_idx]
        assert head_idx is not None
        head_slice = attention_head_slice(module, head_idx)
        return ComponentSpec(
            key=component_key,
            granularity=granularity,
            layer_idx=layer_idx,
            head_idx=head_idx,
            module_idx=layer_idx,
            head_slice=head_slice,
        )

    if granularity == "mlp":
        if ctype != "mlp":
            raise ValueError(f"Expected mlp key for mlp run, got {component_key}")
        if layer_idx < 0 or layer_idx >= len(mlp_modules):
            raise ValueError(f"MLP layer index out of range: {component_key}")
        return ComponentSpec(
            key=component_key,
            granularity=granularity,
            layer_idx=layer_idx,
            head_idx=None,
            module_idx=layer_idx,
            head_slice=None,
        )

    raise ValueError(f"Unsupported granularity: {granularity}")


def forward_with_multi_patch(
    wrapper: ScGPTWrapper,
    sample: Dict[str, torch.Tensor],
    patch_specs: Iterable[Tuple[torch.nn.Module, torch.Tensor, Tuple[int, int] | None]],
    patch_positions: Iterable[int],
    seq_len: int,
):
    """Run one forward pass with multiple module-output patch hooks.

    Each hook copies selected clean activations into the same ablated sample.
    This allows direct measurement of 2-component restoration terms.
    """
    handles = []
    patch_positions = list(patch_positions)

    for module, clean_output, head_slice in patch_specs:

        def _patch_hook(_module, _inputs, layer_output, clean_output=clean_output, head_slice=head_slice):
            return patch_module_output(
                layer_output,
                clean_output,
                patch_positions,
                batch_size=1,
                seq_len=seq_len,
                head_slice=head_slice,
            )

        handles.append(module.register_forward_hook(_patch_hook))

    try:
        return wrapper.forward(sample)
    finally:
        for handle in handles:
            handle.remove()


def _summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-cell interaction rows into inference-ready summary statistics."""
    if df.empty:
        return pd.DataFrame()

    group_cols = [
        "run_id",
        "tissue",
        "granularity",
        "pair",
        "source",
        "target",
        "component_i",
        "component_j",
    ]

    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(
            n_cells=("cell_ordinal", "count"),
            effect_mean=("effect", "mean"),
            effect_std=("effect", "std"),
            rest_i_mean=("rest_i", "mean"),
            rest_j_mean=("rest_j", "mean"),
            rest_ij_mean=("rest_ij", "mean"),
            interaction_raw_mean=("interaction_raw", "mean"),
            interaction_raw_std=("interaction_raw", "std"),
            interaction_norm_mean=("interaction_norm", "mean"),
            interaction_norm_std=("interaction_norm", "std"),
            interaction_raw_abs_mean=("interaction_raw_abs", "mean"),
            interaction_norm_abs_mean=("interaction_norm_abs", "mean"),
            interaction_raw_sign_positive_rate=("interaction_raw_sign_positive", "mean"),
        )
        .sort_values(["run_id", "pair", "interaction_raw_abs_mean"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    for metric in ["interaction_raw", "interaction_norm", "effect"]:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        se_col = f"{metric}_se"
        ci_col = f"{metric}_ci95"
        if mean_col not in summary.columns:
            continue
        summary[se_col] = summary[std_col] / np.sqrt(np.maximum(summary["n_cells"], 1))
        summary[ci_col] = Z95 * summary[se_col]

    return summary


def _pair_rollup(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse component-pair summaries into run/pair-level interaction diagnostics."""
    if summary_df.empty:
        return pd.DataFrame()

    group_cols = ["run_id", "tissue", "granularity", "pair", "source", "target"]
    rollup = (
        summary_df.groupby(group_cols, as_index=False)
        .agg(
            measured_component_pairs=("component_i", "count"),
            best_abs_interaction_raw=("interaction_raw_abs_mean", "max"),
            mean_abs_interaction_raw=("interaction_raw_abs_mean", "mean"),
            best_abs_interaction_norm=("interaction_norm_abs_mean", "max"),
            mean_abs_interaction_norm=("interaction_norm_abs_mean", "mean"),
            max_positive_rate=("interaction_raw_sign_positive_rate", "max"),
        )
        .sort_values(["run_id", "pair"])
        .reset_index(drop=True)
    )
    return rollup


def run_for_config(
    repo_root: Path,
    run_id: str,
    cfg_path: Path,
    run_panel: pd.DataFrame,
    max_cells_override: int | None,
    max_cells_per_pair_override: int | None,
    device: str | None,
) -> List[Dict[str, object]]:
    """Execute pairwise measurements for one run configuration."""
    cfg = load_config(str(cfg_path))
    cfg_dir = cfg_path.parent
    ci_cfg = cfg.get("causal_intervention", {})
    project_seed = int(cfg.get("project", {}).get("seed", 0))

    np.random.seed(project_seed)
    torch.manual_seed(project_seed)

    paths = dict(cfg["paths"])
    for key in [
        "processed_h5ad",
        "scgpt_vocab",
        "scgpt_checkpoint",
        "scgpt_repo",
        "dorothea_tsv",
        "hgnc_alias_tsv",
        "gene_info_csv",
        "tabula_sapiens_h5ad",
    ]:
        if key in paths and paths[key]:
            paths[key] = str(resolve_input_path(paths[key], config_dir=cfg_dir, repo_root=repo_root, sc_root=SC_ROOT))

    # Resolve pair-list and candidate-list paths from causal_intervention block.
    # _build_pair_list reads these directly from cfg, so we rewrite cfg in-place.
    for key in ["pairs_path", "candidate_sources_path", "candidate_targets_path"]:
        raw_val = ci_cfg.get(key)
        if raw_val:
            cfg["causal_intervention"][key] = str(
                resolve_input_path(raw_val, config_dir=cfg_dir, repo_root=repo_root, sc_root=SC_ROOT)
            )

    adata = sc.read_h5ad(paths["processed_h5ad"])
    max_cells = max_cells_override if max_cells_override is not None else ci_cfg.get("max_cells")
    if max_cells:
        max_cells = int(max_cells)
        if adata.n_obs > max_cells:
            adata = adata[:max_cells].copy()

    vocab = load_vocab(paths["scgpt_vocab"])
    if hasattr(adata, "var_names"):
        keep_mask = [name in vocab.gene_to_id for name in adata.var_names]
        if not all(keep_mask):
            adata = adata[:, keep_mask].copy()

    dataset_cfg = ScGPTDatasetConfig(**dict(cfg["scgpt_dataset"]))
    if dataset_cfg.pad_token_id is None:
        if vocab.pad_id is None:
            raise ValueError("pad_token_id missing and vocab has no pad token")
        dataset_cfg.pad_token_id = vocab.pad_id

    dataset = ScGPTDataset(adata, vocab.gene_to_id, dataset_cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_scgpt)

    model_cfg = cfg.get("model", {})
    model_args = model_cfg.get("model_args") or {}
    if not model_args:
        args_path = Path(paths["scgpt_checkpoint"]).with_name("args.json")
        if not args_path.exists():
            raise FileNotFoundError(f"Missing args.json near checkpoint: {args_path}")
        scgpt_args = _load_scgpt_args(args_path)
        model_args = _build_scgpt_model_args(scgpt_args, vocab)

    if ci_cfg.get("disable_fast_transformer", False):
        model_args["use_fast_transformer"] = False

    runtime_device = _device(device)
    model, _, _ = load_scgpt_model(
        entrypoint=model_cfg["entrypoint"],
        repo_path=paths["scgpt_repo"],
        checkpoint_path=paths["scgpt_checkpoint"],
        device=runtime_device,
        model_args=model_args,
        prefix_to_strip=model_cfg.get("prefix_to_strip"),
    )
    model.to(runtime_device)
    for module in model.modules():
        if hasattr(module, "enable_nested_tensor"):
            module.enable_nested_tensor = False
        if hasattr(module, "use_nested_tensor"):
            module.use_nested_tensor = False
    wrapper = ScGPTWrapper(model, model_cfg["forward_key_map"])
    wrapper.eval()

    alias_map = load_hgnc_alias_map(paths.get("hgnc_alias_tsv"))
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)
    gene_to_idx: Dict[str, int] = {}
    for idx, name in enumerate(gene_names_norm):
        if name and name not in gene_to_idx:
            gene_to_idx[name] = idx

    rng = random.Random(int(ci_cfg.get("random_seed", project_seed)))
    pairs = _build_pair_list(cfg, paths, gene_to_idx, alias_map, rng)
    pair_lookup = {(p["source"], p["target"]): p for p in pairs}

    layers = find_transformer_layers(model)
    attn_modules = find_attention_modules(layers)
    mlp_modules = find_mlp_modules(layers)

    output_key = ci_cfg.get("output_key")
    output_reduce = ci_cfg.get("output_reduce")
    ablation_mode = ci_cfg.get("ablation_mode", "value")
    baseline_value = float(ci_cfg.get("baseline_value", 0.0))
    tracing_cfg = ci_cfg.get("tracing", {})
    patch_position_mode = tracing_cfg.get("patch_position", "source")
    normalize_restoration = bool(tracing_cfg.get("normalize_restoration", True))

    max_cells_per_pair = (
        max_cells_per_pair_override if max_cells_per_pair_override is not None else ci_cfg.get("max_cells_per_pair")
    )
    max_cells_per_pair = int(max_cells_per_pair) if max_cells_per_pair else None

    out_rows: List[Dict[str, object]] = []

    # Each pair is processed independently to keep cell matching and normalization coherent.
    for pair_name, pair_panel in run_panel.groupby("pair", sort=False):
        source = str(pair_panel.iloc[0]["source"])
        target = str(pair_panel.iloc[0]["target"])

        pair_info = pair_lookup.get((source, target))
        if pair_info is None:
            print(f"[pairwise-skip] run={run_id} pair={pair_name} reason=pair_not_found")
            continue

        source_idx = int(pair_info["source_idx"])
        target_idx = int(pair_info["target_idx"])
        granularity = str(pair_panel.iloc[0]["granularity"])

        # Build component specs once per pair for speed and consistency.
        component_keys = sorted(set(pair_panel["component_i"].astype(str)).union(set(pair_panel["component_j"].astype(str))))
        component_specs: Dict[str, ComponentSpec] = {}
        for key in component_keys:
            component_specs[key] = build_component_spec(
                component_key=key,
                granularity=granularity,
                attn_modules=attn_modules,
                mlp_modules=mlp_modules,
            )

        trace_modules = attn_modules if granularity == "heads" else mlp_modules

        matched_cells = 0
        for batch in dataloader:
            batch = move_to_device(batch, runtime_device)
            sample = {k: v[0:1] for k, v in batch.items()}

            gene_indices = sample["gene_indices"][0]
            source_positions = find_gene_positions(gene_indices, source_idx)
            target_positions = find_gene_positions(gene_indices, target_idx)
            if not source_positions or not target_positions:
                continue

            source_pos = int(source_positions[0])
            target_pos = int(target_positions[0])

            baseline_val = _readout_value(wrapper, sample, target_pos, output_key, output_reduce)
            if ablation_mode == "pad":
                ablated_sample = apply_pad_ablation(sample, source_positions, dataset_cfg.pad_token_id)
            else:
                ablated_sample = apply_value_ablation(sample, source_positions, baseline_value)
            ablated_val = _readout_value(wrapper, ablated_sample, target_pos, output_key, output_reduce)
            effect = baseline_val - ablated_val

            clean_outputs = _capture_clean_outputs(wrapper, sample, trace_modules)
            patch_positions = source_positions if patch_position_mode == "source" else target_positions
            seq_len = int(sample["gene_ids"].shape[1])

            # Single-component restorations, reused for all component pairs in this cell.
            single_rest: Dict[str, float] = {}
            for ckey, cspec in component_specs.items():
                clean_out = clean_outputs[cspec.module_idx]
                if clean_out is None:
                    single_rest[ckey] = float("nan")
                    continue
                patched_outputs = forward_with_multi_patch(
                    wrapper,
                    ablated_sample,
                    patch_specs=[(trace_modules[cspec.module_idx], clean_out, cspec.head_slice)],
                    patch_positions=patch_positions,
                    seq_len=seq_len,
                )
                patched_val = _readout_from_outputs(
                    patched_outputs,
                    ablated_sample,
                    target_pos,
                    output_key,
                    output_reduce,
                )
                single_rest[ckey] = patched_val - ablated_val

            # Two-component restorations and interaction terms.
            for prow in pair_panel.itertuples(index=False):
                ckey_i = str(prow.component_i)
                ckey_j = str(prow.component_j)
                cspec_i = component_specs[ckey_i]
                cspec_j = component_specs[ckey_j]

                clean_i = clean_outputs[cspec_i.module_idx]
                clean_j = clean_outputs[cspec_j.module_idx]
                if clean_i is None or clean_j is None:
                    continue

                patched_pair_outputs = forward_with_multi_patch(
                    wrapper,
                    ablated_sample,
                    patch_specs=[
                        (trace_modules[cspec_i.module_idx], clean_i, cspec_i.head_slice),
                        (trace_modules[cspec_j.module_idx], clean_j, cspec_j.head_slice),
                    ],
                    patch_positions=patch_positions,
                    seq_len=seq_len,
                )
                patched_pair_val = _readout_from_outputs(
                    patched_pair_outputs,
                    ablated_sample,
                    target_pos,
                    output_key,
                    output_reduce,
                )
                rest_i = float(single_rest.get(ckey_i, np.nan))
                rest_j = float(single_rest.get(ckey_j, np.nan))
                rest_ij = float(patched_pair_val - ablated_val)
                interaction_raw = float(rest_ij - rest_i - rest_j)

                if normalize_restoration and abs(effect) > EPS:
                    rest_i_norm = rest_i / effect
                    rest_j_norm = rest_j / effect
                    rest_ij_norm = rest_ij / effect
                    interaction_norm = interaction_raw / effect
                else:
                    rest_i_norm = float("nan")
                    rest_j_norm = float("nan")
                    rest_ij_norm = float("nan")
                    interaction_norm = float("nan")

                out_rows.append(
                    {
                        "run_id": run_id,
                        "tissue": str(prow.tissue),
                        "granularity": granularity,
                        "pair": pair_name,
                        "source": source,
                        "target": target,
                        "component_i": ckey_i,
                        "component_j": ckey_j,
                        "cell_ordinal": matched_cells,
                        "effect": effect,
                        "rest_i": rest_i,
                        "rest_j": rest_j,
                        "rest_ij": rest_ij,
                        "interaction_raw": interaction_raw,
                        "interaction_norm": interaction_norm,
                        "rest_i_norm": rest_i_norm,
                        "rest_j_norm": rest_j_norm,
                        "rest_ij_norm": rest_ij_norm,
                        "interaction_raw_abs": abs(interaction_raw),
                        "interaction_norm_abs": abs(interaction_norm) if np.isfinite(interaction_norm) else np.nan,
                        "interaction_raw_sign_positive": float(interaction_raw > 0.0),
                        "priority_score": float(getattr(prow, "priority_score", np.nan)),
                    }
                )

            matched_cells += 1
            if max_cells_per_pair is not None and matched_cells >= max_cells_per_pair:
                break

        print(f"[pairwise-run] run={run_id} pair={pair_name} matched_cells={matched_cells}")

    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted pairwise patching experiments")
    parser.add_argument(
        "--panel-csv",
        default="subproject_22_patching_mediation_bias_bounds/implementation/configs/targeted_pairwise_panel.csv",
        help="CSV with run_id/pair/component_i/component_j/config_path",
    )
    parser.add_argument(
        "--cells-out",
        default="subproject_22_patching_mediation_bias_bounds/outputs/tables/paper_quality_pairwise_measured_cells.csv",
    )
    parser.add_argument(
        "--summary-out",
        default="subproject_22_patching_mediation_bias_bounds/outputs/tables/paper_quality_pairwise_measured_summary.csv",
    )
    parser.add_argument(
        "--pair-out",
        default="subproject_22_patching_mediation_bias_bounds/outputs/tables/paper_quality_pairwise_measured_pair_rollup.csv",
    )
    parser.add_argument("--max-cells", type=int, default=None, help="Override max cells loaded per run")
    parser.add_argument(
        "--max-cells-per-pair",
        type=int,
        default=12,
        help="Override matched cells processed per source-target pair",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    panel_path = REPO_ROOT / args.panel_csv
    panel = pd.read_csv(panel_path)
    required = {"run_id", "pair", "source", "target", "component_i", "component_j", "config_path", "granularity", "tissue"}
    missing = required.difference(set(panel.columns))
    if missing:
        raise ValueError(f"Panel missing required columns: {sorted(missing)}")

    all_rows: List[Dict[str, object]] = []

    # Set deterministic low-thread execution once; PyTorch disallows changing
    # interop thread count after parallel work has started.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Group by run to avoid repeated model loads for every pair row.
    for (run_id, config_path), run_panel in panel.groupby(["run_id", "config_path"], sort=False):
        cfg_abs = REPO_ROOT / str(config_path)
        if not cfg_abs.exists():
            raise FileNotFoundError(f"Missing config path in panel: {cfg_abs}")

        run_rows = run_for_config(
            repo_root=REPO_ROOT,
            run_id=str(run_id),
            cfg_path=cfg_abs,
            run_panel=run_panel,
            max_cells_override=args.max_cells,
            max_cells_per_pair_override=args.max_cells_per_pair,
            device=args.device,
        )
        all_rows.extend(run_rows)

    cells_df = pd.DataFrame(all_rows)
    summary_df = _summary_frame(cells_df)
    pair_df = _pair_rollup(summary_df)

    cells_out = REPO_ROOT / args.cells_out
    summary_out = REPO_ROOT / args.summary_out
    pair_out = REPO_ROOT / args.pair_out
    cells_out.parent.mkdir(parents=True, exist_ok=True)

    cells_df.to_csv(cells_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    pair_df.to_csv(pair_out, index=False)

    print(f"[pairwise-done] rows={len(cells_df)}")
    print(f"[pairwise-done] cells={cells_out}")
    print(f"[pairwise-done] summary={summary_out}")
    print(f"[pairwise-done] pair_rollup={pair_out}")


if __name__ == "__main__":
    main()
