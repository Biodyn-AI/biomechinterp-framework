#!/usr/bin/env python3
"""Run targeted triple-component patching experiments.

Goal: directly measure higher-order interaction mass on the most unstable pairs.
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
    for candidate in [start] + list(start.parents):
        if (candidate / "market_research").exists() and (candidate / "single_cell_mechinterp").exists():
            return candidate
    raise RuntimeError("Could not infer repository root")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
SC_ROOT = REPO_ROOT / "single_cell_mechinterp"

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
    key: str
    granularity: str
    layer_idx: int
    head_idx: int | None
    module_idx: int
    head_slice: Tuple[int, int] | None


def resolve_input_path(path_value: str | Path, config_dir: Path, repo_root: Path, sc_root: Path) -> Path:
    raw = Path(path_value)
    if raw.is_absolute() and raw.exists():
        return raw
    candidates = [raw, config_dir / raw, repo_root / raw, sc_root / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (config_dir / raw).resolve()


def parse_component_key(key: str) -> Tuple[str, int, int | None]:
    parts = str(key).split(":")
    ctype = parts[0].strip().lower()
    if ctype == "head":
        if len(parts) != 3:
            raise ValueError(f"Invalid head key: {key}")
        return ctype, int(parts[1]), int(parts[2])
    if ctype == "mlp":
        if len(parts) != 2:
            raise ValueError(f"Invalid mlp key: {key}")
        return ctype, int(parts[1]), None
    raise ValueError(f"Unsupported key type: {key}")


def build_component_spec(
    component_key: str,
    granularity: str,
    attn_modules: List[torch.nn.Module],
    mlp_modules: List[torch.nn.Module],
) -> ComponentSpec:
    ctype, layer_idx, head_idx = parse_component_key(component_key)

    if granularity == "heads":
        if ctype != "head":
            raise ValueError(f"Expected head key for heads run, got {component_key}")
        module = attn_modules[layer_idx]
        assert head_idx is not None
        return ComponentSpec(
            key=component_key,
            granularity=granularity,
            layer_idx=layer_idx,
            head_idx=head_idx,
            module_idx=layer_idx,
            head_slice=attention_head_slice(module, head_idx),
        )

    if granularity == "mlp":
        if ctype != "mlp":
            raise ValueError(f"Expected mlp key for mlp run, got {component_key}")
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


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = [
        "run_id",
        "tissue",
        "granularity",
        "pair",
        "source",
        "target",
        "component_a",
        "component_b",
        "component_c",
    ]
    out = (
        df.groupby(group_cols, as_index=False)
        .agg(
            n_cells=("cell_ordinal", "count"),
            effect_mean=("effect", "mean"),
            tau_raw_mean=("triple_interaction_raw", "mean"),
            tau_raw_std=("triple_interaction_raw", "std"),
            tau_norm_mean=("triple_interaction_norm", "mean"),
            tau_norm_std=("triple_interaction_norm", "std"),
            tau_abs_mean=("triple_interaction_abs", "mean"),
            tau_norm_abs_mean=("triple_interaction_norm_abs", "mean"),
            tau_sign_positive_rate=("triple_interaction_positive", "mean"),
        )
        .sort_values(["run_id", "pair"])
        .reset_index(drop=True)
    )
    out["tau_raw_se"] = out["tau_raw_std"] / np.sqrt(np.maximum(out["n_cells"], 1))
    out["tau_raw_ci95"] = Z95 * out["tau_raw_se"]
    return out


def run_for_config(
    repo_root: Path,
    run_id: str,
    cfg_path: Path,
    run_panel: pd.DataFrame,
    max_cells_override: int | None,
    max_cells_per_pair_override: int | None,
    device: str | None,
) -> List[Dict[str, object]]:
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

    for row in run_panel.itertuples(index=False):
        pair_name = str(row.pair)
        source = str(row.source)
        target = str(row.target)
        granularity = str(row.granularity)

        pair_info = pair_lookup.get((source, target))
        if pair_info is None:
            continue

        source_idx = int(pair_info["source_idx"])
        target_idx = int(pair_info["target_idx"])

        ca = build_component_spec(str(row.component_a), granularity, attn_modules, mlp_modules)
        cb = build_component_spec(str(row.component_b), granularity, attn_modules, mlp_modules)
        cc = build_component_spec(str(row.component_c), granularity, attn_modules, mlp_modules)

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

            def run_patch(specs: List[ComponentSpec]) -> float:
                patch_specs = []
                for s in specs:
                    clean = clean_outputs[s.module_idx]
                    if clean is None:
                        return float("nan")
                    patch_specs.append((trace_modules[s.module_idx], clean, s.head_slice))
                patched_outputs = forward_with_multi_patch(
                    wrapper,
                    ablated_sample,
                    patch_specs=patch_specs,
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
                return float(patched_val - ablated_val)

            r_a = run_patch([ca])
            r_b = run_patch([cb])
            r_c = run_patch([cc])
            r_ab = run_patch([ca, cb])
            r_ac = run_patch([ca, cc])
            r_bc = run_patch([cb, cc])
            r_abc = run_patch([ca, cb, cc])

            d_ab = r_ab - r_a - r_b
            d_ac = r_ac - r_a - r_c
            d_bc = r_bc - r_b - r_c
            tau = r_abc - (r_ab + r_ac + r_bc) + (r_a + r_b + r_c)

            if normalize_restoration and abs(effect) > EPS:
                tau_norm = tau / effect
            else:
                tau_norm = float("nan")

            out_rows.append(
                {
                    "run_id": run_id,
                    "tissue": str(row.tissue),
                    "granularity": granularity,
                    "pair": pair_name,
                    "source": source,
                    "target": target,
                    "component_a": str(row.component_a),
                    "component_b": str(row.component_b),
                    "component_c": str(row.component_c),
                    "cell_ordinal": matched_cells,
                    "effect": effect,
                    "r_a": r_a,
                    "r_b": r_b,
                    "r_c": r_c,
                    "r_ab": r_ab,
                    "r_ac": r_ac,
                    "r_bc": r_bc,
                    "r_abc": r_abc,
                    "delta_ab": d_ab,
                    "delta_ac": d_ac,
                    "delta_bc": d_bc,
                    "triple_interaction_raw": tau,
                    "triple_interaction_norm": tau_norm,
                    "triple_interaction_abs": abs(tau),
                    "triple_interaction_norm_abs": abs(tau_norm) if np.isfinite(tau_norm) else np.nan,
                    "triple_interaction_positive": float(tau > 0.0),
                }
            )

            matched_cells += 1
            if max_cells_per_pair is not None and matched_cells >= max_cells_per_pair:
                break

        print(f"[triplet-run] run={run_id} pair={pair_name} matched_cells={matched_cells}")

    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted triplet patching experiments")
    parser.add_argument(
        "--panel-csv",
        default="subproject_22_patching_mediation_bias_bounds/implementation/configs/targeted_triplet_panel.csv",
    )
    parser.add_argument(
        "--cells-out",
        default="subproject_22_patching_mediation_bias_bounds/outputs/tables/paper_quality_triplet_measured_cells.csv",
    )
    parser.add_argument(
        "--summary-out",
        default="subproject_22_patching_mediation_bias_bounds/outputs/tables/paper_quality_triplet_measured_summary.csv",
    )
    parser.add_argument("--max-cells", type=int, default=400)
    parser.add_argument("--max-cells-per-pair", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    panel = pd.read_csv(REPO_ROOT / args.panel_csv)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    rows: List[Dict[str, object]] = []
    for (run_id, config_path), run_panel in panel.groupby(["run_id", "config_path"], sort=False):
        cfg_abs = REPO_ROOT / str(config_path)
        rows.extend(
            run_for_config(
                repo_root=REPO_ROOT,
                run_id=str(run_id),
                cfg_path=cfg_abs,
                run_panel=run_panel,
                max_cells_override=args.max_cells,
                max_cells_per_pair_override=args.max_cells_per_pair,
                device=args.device,
            )
        )

    cells_df = pd.DataFrame(rows)
    summary_df = summarize(cells_df)

    cells_out = REPO_ROOT / args.cells_out
    summary_out = REPO_ROOT / args.summary_out
    cells_out.parent.mkdir(parents=True, exist_ok=True)
    cells_df.to_csv(cells_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print(f"[triplet-done] rows={len(cells_df)}")
    print(f"[triplet-done] cells={cells_out}")
    print(f"[triplet-done] summary={summary_out}")


if __name__ == "__main__":
    main()
