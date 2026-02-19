#!/usr/bin/env python3
"""Run perturbation-to-network bridge benchmark across inferred edge variants.

This script evaluates every `network_inference/outputs/inferred_edges*.tsv` file
against four references:
1) perturbation-derived edges (`timeseries_edges.tsv`)
2) OmniPath proxy edges (`timeseries_edges_proxy.tsv`)
3) HPN-DREAM gold standard (with symbol mapping)
4) BEELINE GSD reference (with symbol mapping)

Outputs are written as CSV tables and diagnostic figures for workshop-paper use.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd

from network_inference.src.evaluation.custom_edges import (
    _load_gene_names,
    load_edge_list,
    normalize_edge_list,
)
from network_inference.src.evaluation.metrics import aupr
from network_inference.src.utils.scm_imports import ensure_mechinterp_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root.",
    )
    parser.add_argument(
        "--processed-h5ad",
        type=Path,
        default=Path(
            "single_cell_mechinterp/outputs/tabula_sapiens_immune_subset_hpn_processed.h5ad"
        ),
        help="Processed AnnData used for normalization/evaluation gene universe.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_07_perturbation_to_network_bridging/reports/artifacts/bridge_benchmark"
        ),
        help="Destination directory for metrics and figures.",
    )
    parser.add_argument(
        "--limit-methods",
        type=int,
        default=0,
        help="Optional cap on number of inferred_edges files for quick checks (0 = all).",
    )
    parser.add_argument(
        "--edge-files",
        type=str,
        default="",
        help="Optional comma-separated inferred edge TSV basenames to evaluate.",
    )
    return parser.parse_args()


def method_from_edge_path(path: Path) -> str:
    stem = path.stem
    if stem == "inferred_edges":
        return "base"
    if stem.startswith("inferred_edges_"):
        return stem[len("inferred_edges_") :]
    return stem


def infer_parent_method(method: str) -> str | None:
    if method.endswith("_calibrated_logistic"):
        return method[: -len("_calibrated_logistic")]
    if method.endswith("_calibrated"):
        return method[: -len("_calibrated")]
    return None


def edge_count(path: Path) -> int:
    # TSVs are small enough to count with pandas once per method.
    return int(pd.read_csv(path, sep="\t").shape[0])


def build_references(repo_root: Path) -> List[Dict[str, object]]:
    data_dir = repo_root / "network_inference" / "data"
    return [
        {
            "reference": "perturbation_edges",
            "truth_edges_path": data_dir / "timeseries_edges.tsv",
            "mapping_cfg": None,
        },
        {
            "reference": "omnipath_proxy",
            "truth_edges_path": data_dir / "timeseries_edges_proxy.tsv",
            "mapping_cfg": None,
        },
        {
            "reference": "hpn_dream",
            "truth_edges_path": data_dir / "hpn_dream_goldstandard.tsv",
            "mapping_cfg": {
                "strip_phospho_suffix": True,
                "symbol_map_tsv": data_dir / "hpn_dream_symbol_map.tsv",
                "hgnc_extra_alias_cols": ["entrez_id", "ensembl_id", "uniprot_ids"],
                "drop_unmapped": True,
            },
        },
        {
            "reference": "beeline_gsd",
            "truth_edges_path": data_dir / "beeline_gsd_refnetwork.tsv",
            "mapping_cfg": {
                "symbol_map_tsv": data_dir / "beeline_gsd_symbol_map.tsv",
                "hgnc_extra_alias_cols": ["entrez_id", "ensembl_id", "uniprot_ids"],
                "drop_unmapped": True,
            },
        },
    ]


def render_figures(metrics: pd.DataFrame, out_dir: Path) -> None:
    # Figure 1: method ranking on perturbation-derived edges.
    pert = metrics[metrics["reference"] == "perturbation_edges"].copy()
    top = pert.sort_values("f1", ascending=False).head(20)
    plt.figure(figsize=(11, 6))
    plt.barh(top["method"], top["f1"], color="#1f77b4")
    plt.gca().invert_yaxis()
    plt.xlabel("F1")
    plt.ylabel("Method")
    plt.title("Top-20 Methods on Perturbation-Derived Edge Benchmark (F1)")
    plt.tight_layout()
    plt.savefig(out_dir / "perturbation_edges_f1_top20.png", dpi=220)
    plt.close()

    # Figure 2: precision-recall landscape on perturbation-derived edges.
    plt.figure(figsize=(7, 6))
    plt.scatter(pert["recall"], pert["precision"], s=38, alpha=0.8, color="#ff7f0e")
    top3 = pert.sort_values("f1", ascending=False).head(3)
    for _, row in top3.iterrows():
        plt.annotate(
            row["method"],
            (row["recall"], row["precision"]),
            textcoords="offset points",
            xytext=(6, 2),
            fontsize=8,
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision vs Recall on Perturbation-Derived Edges")
    plt.tight_layout()
    plt.savefig(out_dir / "perturbation_edges_precision_recall_scatter.png", dpi=220)
    plt.close()

    # Figure 3: AUPR heatmap for the best methods by mean AUPR across references.
    mean_aupr = metrics.groupby("method", as_index=False)["aupr"].mean().sort_values("aupr", ascending=False)
    keep_methods = mean_aupr.head(15)["method"].tolist()
    heat = (
        metrics[metrics["method"].isin(keep_methods)]
        .pivot(index="method", columns="reference", values="aupr")
        .reindex(keep_methods)
        .fillna(0.0)
    )
    plt.figure(figsize=(9, 6))
    plt.imshow(heat.values, aspect="auto", cmap="viridis")
    plt.colorbar(label="AUPR")
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=20, ha="right")
    plt.title("AUPR Heatmap (Top-15 Mean Methods)")
    plt.tight_layout()
    plt.savefig(out_dir / "aupr_heatmap_top15_methods.png", dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    processed_h5ad = (
        args.processed_h5ad if args.processed_h5ad.is_absolute() else (repo_root / args.processed_h5ad)
    ).resolve()
    out_dir = args.output_dir if args.output_dir.is_absolute() else (repo_root / args.output_dir)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    hgnc_alias_tsv = (repo_root / "single_cell_mechinterp" / "external" / "hgnc_complete_set.txt").resolve()
    edges_dir = (repo_root / "network_inference" / "outputs").resolve()
    edge_files = sorted(edges_dir.glob("inferred_edges*.tsv"))
    if args.edge_files.strip():
        requested = [item.strip() for item in args.edge_files.split(",") if item.strip()]
        requested_set = set(requested)
        edge_files = [path for path in edge_files if path.name in requested_set]
        missing = sorted(requested_set - {path.name for path in edge_files})
        if missing:
            raise FileNotFoundError(f"Requested edge files not found: {missing}")
    if args.limit_methods and args.limit_methods > 0:
        edge_files = edge_files[: args.limit_methods]

    if not edge_files:
        raise FileNotFoundError(f"No inferred edge files found under {edges_dir}")
    if not processed_h5ad.exists():
        raise FileNotFoundError(f"Processed h5ad not found: {processed_h5ad}")

    references = build_references(repo_root)
    gene_names_norm, alias_map = _load_gene_names(
        processed_h5ad,
        hgnc_alias_tsv,
        mapping_cfg=None,
    )
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names_norm)}
    n_genes = len(gene_names_norm)
    candidate_mask = np.ones((n_genes, n_genes), dtype=bool)
    candidate_mask &= ~np.eye(n_genes, dtype=bool)

    # Normalize references once to avoid re-loading the AnnData object per run.
    normalized_references: List[Dict[str, object]] = []
    for ref in references:
        truth_edges = load_edge_list(ref["truth_edges_path"], "source", "target")
        truth_edges, _ = normalize_edge_list(
            truth_edges,
            processed_h5ad,
            hgnc_alias_tsv,
            mapping_cfg=ref["mapping_cfg"],
            gene_names_norm=gene_names_norm,
            alias_map=alias_map,
        )
        true_mask = np.zeros((n_genes, n_genes), dtype=bool)
        for _, row in truth_edges.iterrows():
            src_idx = gene_to_idx.get(row["source"])
            tgt_idx = gene_to_idx.get(row["target"])
            if src_idx is None or tgt_idx is None:
                continue
            true_mask[src_idx, tgt_idx] = True
        labels_flat = true_mask[candidate_mask].astype(np.int8)
        normalized_references.append(
            {
                "reference": ref["reference"],
                "truth_edges": truth_edges[["source", "target"]].copy(),
                "labels_flat": labels_flat,
            }
        )

    ensure_mechinterp_path()
    from src.eval.metrics import precision_recall_f1

    rows: List[Dict[str, object]] = []
    total_runs = len(edge_files) * len(normalized_references)
    run_idx = 0
    for edge_path in edge_files:
        method = method_from_edge_path(edge_path)
        n_edges = edge_count(edge_path)

        pred_edges = pd.read_csv(edge_path, sep="\t")
        pred_cols = ["source", "target"] + (["score"] if "score" in pred_edges.columns else [])
        pred_edges = pred_edges[pred_cols].drop_duplicates()
        pred_edges, _ = normalize_edge_list(
            pred_edges,
            processed_h5ad,
            hgnc_alias_tsv,
            mapping_cfg=None,
            gene_names_norm=gene_names_norm,
            alias_map=alias_map,
        )
        if "score" in pred_edges.columns:
            pred_edges = pred_edges.groupby(["source", "target"], as_index=False)["score"].max()
        else:
            pred_edges["score"] = 1.0

        scores = np.zeros((n_genes, n_genes), dtype=np.float32)
        for _, row in pred_edges.iterrows():
            src_idx = gene_to_idx.get(row["source"])
            tgt_idx = gene_to_idx.get(row["target"])
            if src_idx is None or tgt_idx is None:
                continue
            score = float(row["score"])
            if score > scores[src_idx, tgt_idx]:
                scores[src_idx, tgt_idx] = score
        scores_flat = scores[candidate_mask]

        pred_pairs = pred_edges[["source", "target"]]
        for ref in normalized_references:
            run_idx += 1
            print(
                f"[{run_idx}/{total_runs}] evaluating method={method} reference={ref['reference']}",
                flush=True,
            )
            prf = precision_recall_f1(pred_pairs, ref["truth_edges"])
            rows.append(
                {
                    "method": method,
                    "edge_file": edge_path.name,
                    "pred_edges": n_edges,
                    "reference": ref["reference"],
                    "precision": float(prf["precision"]),
                    "recall": float(prf["recall"]),
                    "f1": float(prf["f1"]),
                    "aupr": float(aupr(scores_flat, ref["labels_flat"])),
                }
            )

    metrics_df = pd.DataFrame(rows).sort_values(["reference", "f1", "aupr"], ascending=[True, False, False])
    metrics_df.to_csv(out_dir / "bridge_eval_metrics.csv", index=False)

    best_df = metrics_df.sort_values(["reference", "f1", "aupr"], ascending=[True, False, False]).groupby(
        "reference", as_index=False
    ).head(5)
    best_df.to_csv(out_dir / "bridge_eval_top5_per_reference.csv", index=False)

    mean_rank = metrics_df.groupby("method", as_index=False)[["f1", "aupr"]].mean().sort_values(
        ["f1", "aupr"], ascending=False
    )
    mean_rank.to_csv(out_dir / "bridge_eval_mean_rank.csv", index=False)

    # Calibration delta quantifies what isotonic/logistic do relative to raw edges.
    calibration_rows: List[Dict[str, object]] = []
    keyed = {(row["method"], row["reference"]): row for _, row in metrics_df.iterrows()}
    for method in sorted(metrics_df["method"].unique()):
        parent = infer_parent_method(method)
        if not parent:
            continue
        for ref in sorted(metrics_df["reference"].unique()):
            child = keyed.get((method, ref))
            base = keyed.get((parent, ref))
            if child is None or base is None:
                continue
            calibration_rows.append(
                {
                    "method": method,
                    "parent_method": parent,
                    "reference": ref,
                    "delta_f1": float(child["f1"] - base["f1"]),
                    "delta_aupr": float(child["aupr"] - base["aupr"]),
                }
            )
    calibration_df = pd.DataFrame(calibration_rows)
    if not calibration_df.empty:
        calibration_df.to_csv(out_dir / "bridge_eval_calibration_delta.csv", index=False)

    render_figures(metrics_df, out_dir)

    # Small text summary to make workshop drafting easier.
    pert = metrics_df[metrics_df["reference"] == "perturbation_edges"].copy()
    best_pert = pert.sort_values(["f1", "aupr"], ascending=False).head(1).iloc[0]
    sparse_hits = int((pert["f1"] > 0).sum())
    summary_lines = [
        f"Processed h5ad: {processed_h5ad}",
        f"Methods evaluated: {metrics_df['method'].nunique()}",
        f"References evaluated: {metrics_df['reference'].nunique()}",
        f"Best perturbation_edges method: {best_pert['method']} (F1={best_pert['f1']:.6f}, AUPR={best_pert['aupr']:.6f})",
        f"Methods with non-zero perturbation_edges F1: {sparse_hits}/{pert.shape[0]}",
    ]
    (out_dir / "bridge_eval_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
