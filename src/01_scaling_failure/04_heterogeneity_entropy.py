#!/usr/bin/env python3
"""Quantify heterogeneity in scaling runs (cell-type entropy) and relate to F1.

This is a lightweight stress test for the hypothesis that the observed scaling
degradation is driven by increasing heterogeneity/composition diversity rather
than sample size alone.

Inputs (from biodyn-work):
- subproject_06_scaling_failure_analysis/.../multiseed_run_reference_metrics.csv
- single_cell_mechinterp/configs/scaling_workshop_manifest.csv
- single_cell_mechinterp/outputs/scaling/<experiment_id>/tabula_sapiens_processed.h5ad

Outputs (to biodyn-nmi-paper):
- results/scaling_heterogeneity_summary.csv
- figures/fig_scaling_entropy_vs_f1.png
- figures/fig_scaling_n_cell_types_vs_f1.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import h5py


def find_biodyn_work_root(paper_root: Path) -> Path:
    env = os.environ.get("BIODYN_WORK_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"BIODYN_WORK_ROOT does not exist: {p}")
        return p

    candidate = (paper_root.parent / "biodyn-work").resolve()
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not locate biodyn-work. Set BIODYN_WORK_ROOT or place biodyn-work next to biodyn-nmi-paper."
    )


def shannon_entropy(counts: np.ndarray) -> float:
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _load_manifest_row(manifest_path: Path, experiment_id: str) -> dict[str, str]:
    manifest = pd.read_csv(manifest_path, dtype=str)
    sub = manifest[manifest["experiment_id"] == experiment_id]
    if sub.empty:
        raise KeyError(f"experiment_id not found in manifest: {experiment_id}")
    return sub.iloc[0].to_dict()


def _read_cell_types(h5ad_path: Path) -> tuple[int, np.ndarray | None]:
    """Return (n_obs, cell_type array or None) from a processed .h5ad.

    We avoid anndata/scanpy here because some h5ad files contain metadata that
    breaks some readers (`encoding_type='null'`).
    """

    with h5py.File(h5ad_path, "r") as f:
        obs = f.get("obs", None)
        if obs is None or "_index" not in obs:
            return 0, None
        n_obs = int(obs["_index"].shape[0])

        if "cell_type" not in obs:
            return n_obs, None

        obj = obs["cell_type"]
        # Categorical columns are stored as a group with categories/codes.
        if isinstance(obj, h5py.Group) and "categories" in obj and "codes" in obj:
            cats = obj["categories"][...]
            codes = obj["codes"][...].astype(int)
            cats = [c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c) for c in cats]
            out = np.empty(shape=codes.shape[0], dtype=object)
            any_valid = False
            for i, code in enumerate(codes):
                if code >= 0:
                    out[i] = cats[code]
                    any_valid = True
                else:
                    out[i] = None
            if not any_valid:
                return n_obs, None
            return n_obs, out

        if isinstance(obj, h5py.Dataset):
            vals = obj[...]
            return n_obs, np.array(
                [v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v) for v in vals],
                dtype=object,
            )

        return n_obs, None


def main() -> None:
    paper_root = Path(__file__).resolve().parents[2]
    work_root = find_biodyn_work_root(paper_root)

    metrics_path = (
        work_root
        / "subproject_06_scaling_failure_analysis"
        / "reports"
        / "data_extended"
        / "multiseed_run_reference_metrics.csv"
    )
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    metrics = pd.read_csv(metrics_path)
    # Focus on TRRUST F1 (DoRothEA behaves similarly but TRRUST is the main scaling figure).
    metrics = metrics[metrics["reference"] == "trrust"].copy()

    manifest_path = work_root / "single_cell_mechinterp" / "configs" / "scaling_workshop_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    scaling_dir = work_root / "single_cell_mechinterp" / "outputs" / "scaling"
    if not scaling_dir.exists():
        raise FileNotFoundError(scaling_dir)

    rows = []
    for exp_id in sorted(metrics["experiment_id"].unique()):
        try:
            manifest_row = _load_manifest_row(manifest_path, exp_id)
        except Exception:
            manifest_row = {}

        run_dir = scaling_dir / exp_id
        h5ad_path = run_dir / "tabula_sapiens_processed.h5ad"
        if not h5ad_path.exists() and manifest_row.get("processed_h5ad"):
            # Manifest paths are relative to single_cell_mechinterp.
            h5ad_path = (work_root / "single_cell_mechinterp" / manifest_row["processed_h5ad"]).resolve()
        if not h5ad_path.exists():
            continue

        n_obs, ct_all = _read_cell_types(h5ad_path)
        if n_obs <= 0:
            continue

        # Reproduce subsampling logic from single_cell_mechinterp/src/cli.py:cmd_extract_attention.
        max_cells = manifest_row.get("max_cells")
        sample_strategy = manifest_row.get("sample_strategy") or "head"
        sample_seed = manifest_row.get("sample_seed") or manifest_row.get("seed")

        max_cells_int = None
        if max_cells:
            try:
                max_cells_int = int(max_cells)
            except Exception:
                max_cells_int = None

        sample_seed_int = 0
        if sample_seed is not None:
            try:
                sample_seed_int = int(sample_seed)
            except Exception:
                sample_seed_int = 0

        n_selected = n_obs
        selected_ct = ct_all
        if max_cells_int and n_obs > max_cells_int:
            n_selected = max_cells_int
            if ct_all is not None and len(ct_all) == n_obs:
                if sample_strategy == "random":
                    rng = np.random.default_rng(sample_seed_int)
                    indices = np.sort(rng.choice(n_obs, size=max_cells_int, replace=False))
                    selected_ct = ct_all[indices]
                elif sample_strategy == "head":
                    selected_ct = ct_all[:max_cells_int]
                else:
                    raise ValueError(f"Unknown sample_strategy: {sample_strategy}")

        if selected_ct is None or len(selected_ct) == 0:
            n_types = float("nan")
            entropy = float("nan")
        else:
            clean = np.array([x for x in selected_ct if x is not None], dtype=object)
            if clean.size == 0:
                n_types = float("nan")
                entropy = float("nan")
            else:
                unique, counts = np.unique(clean, return_counts=True)
                n_types = int(unique.shape[0])
                entropy = shannon_entropy(counts)

        # Merge in F1 (one row per exp_id for TRRUST in this file).
        sub = metrics[metrics["experiment_id"] == exp_id].iloc[0]
        rows.append(
            dict(
                experiment_id=exp_id,
                model_tier=sub["model_tier"],
                cells=int(sub["cells"]),
                seed=int(sub["seed"]),
                f1=float(sub["f1"]),
                n_cell_types=n_types,
                cell_type_entropy=entropy,
                n_cells_in_run=n_selected,
                n_cells_available=n_obs,
                sample_strategy=sample_strategy,
                sample_seed=sample_seed_int,
            )
        )

    out = pd.DataFrame(rows)
    out_dir = paper_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scaling_heterogeneity_summary.csv"
    out.to_csv(out_path, index=False)

    # Plot entropy vs F1.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    plot_df = out.dropna(subset=["cell_type_entropy", "f1"]).copy()
    if not plot_df.empty:
        # n_cell_types vs F1 (heterogeneity proxy aligned with dilution hypothesis).
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        # Jitter x slightly for readability.
        rng = np.random.default_rng(0)
        x = plot_df["n_cell_types"].to_numpy(dtype=float) + rng.normal(scale=0.05, size=plot_df.shape[0])
        sca = ax.scatter(
            x,
            plot_df["f1"],
            c=plot_df["cells"],
            cmap="viridis",
            s=50,
            alpha=0.85,
        )
        ax.set_xlabel("Number of observed cell types")
        ax.set_ylabel("TRRUST F1")
        ax.set_title("Scaling Runs: Cell-Type Richness vs GRN Recovery")
        ax.grid(True, alpha=0.25)
        cb = fig.colorbar(sca, ax=ax)
        cb.set_label("Cells")
        fig.tight_layout()
        fig.savefig(paper_root / "figures" / "fig_scaling_n_cell_types_vs_f1.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        sca = ax.scatter(
            plot_df["cell_type_entropy"],
            plot_df["f1"],
            c=plot_df["cells"],
            cmap="viridis",
            s=50,
            alpha=0.85,
        )
        ax.set_xlabel("Cell-type Shannon entropy")
        ax.set_ylabel("TRRUST F1")
        ax.set_title("Scaling Runs: Heterogeneity vs GRN Recovery")
        ax.grid(True, alpha=0.25)
        cb = fig.colorbar(sca, ax=ax)
        cb.set_label("Cells")
        fig.tight_layout()
        fig.savefig(paper_root / "figures" / "fig_scaling_entropy_vs_f1.png", dpi=300)
        plt.close(fig)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
