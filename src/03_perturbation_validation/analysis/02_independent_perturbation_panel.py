#!/usr/bin/env python3
"""Evaluate bridge methods on independent perturbation-reference panel.

This script extends Subproject 07 with perturbation references derived from
independent perturbation-pair resources (Dixit/Shifrut config tables), and
reports overlap diagnostics against the OmniPath proxy reference, including
BH-adjusted random-overlap controls and edge-level recurrence summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from network_inference.src.evaluation.custom_edges import (
    _load_gene_names,
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
        help="Destination directory for outputs.",
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


def edge_count(path: Path) -> int:
    return int(pd.read_csv(path, sep="\t").shape[0])


def benjamini_hochberg_qvalues(p_values: pd.Series) -> pd.Series:
    """Compute BH-adjusted q-values from a vector of p-values."""
    p = p_values.to_numpy(dtype=float)
    n = p.size
    if n == 0:
        return pd.Series(dtype=float, index=p_values.index)

    order = np.argsort(p)
    ranked_p = p[order]
    q = np.empty(n, dtype=float)

    prev = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        value = ranked_p[idx] * n / rank
        prev = min(prev, value)
        q[idx] = prev

    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return pd.Series(out, index=p_values.index)


def _load_reference_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    if "source" not in table.columns or "target" not in table.columns:
        raise ValueError(f"Reference table missing source/target columns: {path}")
    return table[["source", "target"]].dropna().drop_duplicates()


def build_reference_tables(repo_root: Path) -> Dict[str, pd.DataFrame]:
    data_dir = repo_root / "network_inference" / "data"
    cfg_dir = repo_root / "single_cell_mechinterp" / "configs"

    refs: Dict[str, pd.DataFrame] = {
        "perturbation_edges": _load_reference_table(data_dir / "timeseries_edges.tsv"),
        "omnipath_proxy": _load_reference_table(data_dir / "timeseries_edges_proxy.tsv"),
        "perturbation_dixit_pairs": _load_reference_table(cfg_dir / "perturbation_pairs_dixit.tsv"),
        "perturbation_dixit_targets_pairs": _load_reference_table(
            cfg_dir / "perturbation_pairs_dixit_targets.tsv"
        ),
        "perturbation_shifrut_targets_pairs": _load_reference_table(
            cfg_dir / "perturbation_pairs_shifrut_targets.tsv"
        ),
    }

    independent_union = (
        pd.concat(
            [
                refs["perturbation_dixit_pairs"],
                refs["perturbation_dixit_targets_pairs"],
                refs["perturbation_shifrut_targets_pairs"],
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )
    refs["perturbation_independent_union"] = independent_union
    return refs


def normalize_reference_sets(
    references: Dict[str, pd.DataFrame],
    processed_h5ad: Path,
    hgnc_alias_tsv: Path,
    gene_names_norm: np.ndarray,
    alias_map: Dict[str, str],
) -> Dict[str, Dict[str, object]]:
    ref_norm: Dict[str, Dict[str, object]] = {}
    for ref_name, table in references.items():
        norm, _ = normalize_edge_list(
            table,
            processed_h5ad,
            hgnc_alias_tsv,
            mapping_cfg=None,
            gene_names_norm=gene_names_norm,
            alias_map=alias_map,
        )
        norm = norm[["source", "target"]].drop_duplicates().reset_index(drop=True)
        edge_set = set(map(tuple, norm.itertuples(index=False, name=None)))
        ref_norm[ref_name] = {"table": norm, "set": edge_set}
    return ref_norm


def build_pairwise_jaccard(ref_norm: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    refs = sorted(ref_norm.keys())
    rows: List[Dict[str, object]] = []
    for i, a in enumerate(refs):
        set_a = ref_norm[a]["set"]
        for b in refs[i + 1 :]:
            set_b = ref_norm[b]["set"]
            union = set_a | set_b
            inter = set_a & set_b
            jaccard = (len(inter) / len(union)) if union else 0.0
            rows.append(
                {
                    "reference_a": a,
                    "reference_b": b,
                    "intersection_edges": len(inter),
                    "union_edges": len(union),
                    "jaccard": jaccard,
                }
            )
    return pd.DataFrame(rows).sort_values("jaccard", ascending=False)


def render_figures(metrics: pd.DataFrame, out_dir: Path) -> None:
    union = metrics[metrics["reference"] == "perturbation_independent_union"].copy()
    top = union.sort_values("f1", ascending=False).head(15)

    plt.figure(figsize=(10.5, 5.8))
    plt.barh(top["method"], top["f1"], color="#3182bd")
    plt.gca().invert_yaxis()
    plt.xlabel("F1")
    plt.ylabel("Method")
    plt.title("Independent Perturbation Union: Top-15 Methods by F1")
    plt.tight_layout()
    plt.savefig(out_dir / "independent_union_f1_top15.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(union["recall"], union["precision"], s=40, alpha=0.85, color="#e6550d")
    top3 = union.sort_values("f1", ascending=False).head(3)
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
    plt.title("Independent Perturbation Union: Precision-Recall Landscape")
    plt.tight_layout()
    plt.savefig(out_dir / "independent_union_precision_recall_scatter.png", dpi=220)
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
    if not edge_files:
        raise FileNotFoundError(f"No inferred edge files found under {edges_dir}")

    references = build_reference_tables(repo_root)
    gene_names_norm, alias_map = _load_gene_names(
        processed_h5ad,
        hgnc_alias_tsv,
        mapping_cfg=None,
    )
    n_genes = len(gene_names_norm)
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names_norm)}
    candidate_mask = np.ones((n_genes, n_genes), dtype=bool)
    candidate_mask &= ~np.eye(n_genes, dtype=bool)
    candidate_edges = int(candidate_mask.sum())

    normalized_refs = normalize_reference_sets(
        references=references,
        processed_h5ad=processed_h5ad,
        hgnc_alias_tsv=hgnc_alias_tsv,
        gene_names_norm=gene_names_norm,
        alias_map=alias_map,
    )

    # Export normalization sizes.
    size_rows = []
    for name, payload in normalized_refs.items():
        size_rows.append(
            {
                "reference": name,
                "raw_edges": int(len(references[name])),
                "normalized_edges": int(len(payload["table"])),
            }
        )
    size_df = pd.DataFrame(size_rows).sort_values("reference")
    size_df.to_csv(out_dir / "independent_reference_normalization_summary.csv", index=False)

    jaccard_df = build_pairwise_jaccard(normalized_refs)
    jaccard_df.to_csv(out_dir / "independent_reference_pairwise_jaccard.csv", index=False)

    ensure_mechinterp_path()
    from src.eval.metrics import precision_recall_f1

    # Pre-build flattened labels for AUPR per reference.
    ref_payload = {}
    for name, payload in normalized_refs.items():
        table = payload["table"]
        true_mask = np.zeros((n_genes, n_genes), dtype=bool)
        for _, row in table.iterrows():
            src_idx = gene_to_idx.get(row["source"])
            tgt_idx = gene_to_idx.get(row["target"])
            if src_idx is None or tgt_idx is None:
                continue
            true_mask[src_idx, tgt_idx] = True
        ref_payload[name] = {
            "table": table,
            "labels_flat": true_mask[candidate_mask].astype(np.int8),
            "truth_edges": int(table.shape[0]),
        }

    rows: List[Dict[str, object]] = []
    control_rows: List[Dict[str, object]] = []
    method_predictions: Dict[str, pd.DataFrame] = {}
    total_runs = len(edge_files) * len(ref_payload)
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
        method_predictions[method] = pred_edges[["source", "target", "score"]].copy()
        pred_pairs = pred_edges[["source", "target"]]

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

        for ref_name, payload in ref_payload.items():
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] method={method} reference={ref_name}", flush=True)
            truth_table = payload["table"]
            prf = precision_recall_f1(pred_pairs, truth_table)
            precision = float(prf["precision"])
            recall = float(prf["recall"])
            f1 = float(prf["f1"])
            aupr_val = float(aupr(scores_flat, payload["labels_flat"]))
            rows.append(
                {
                    "method": method,
                    "edge_file": edge_path.name,
                    "pred_edges": int(n_edges),
                    "reference": ref_name,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "aupr": aupr_val,
                }
            )

            truth_edges = payload["truth_edges"]
            tp_from_recall = recall * truth_edges
            tp_from_precision = precision * n_edges
            observed_tp = int(round((tp_from_recall + tp_from_precision) / 2.0))
            observed_tp = max(0, min(observed_tp, truth_edges, n_edges))
            expected_tp = (n_edges * truth_edges / candidate_edges) if candidate_edges else 0.0
            enrichment = (observed_tp / expected_tp) if expected_tp > 0 else np.nan
            p_ge_tp = float(hypergeom.sf(observed_tp - 1, candidate_edges, truth_edges, n_edges))
            control_rows.append(
                {
                    "method": method,
                    "reference": ref_name,
                    "pred_edges": int(n_edges),
                    "truth_edges": int(truth_edges),
                    "candidate_edges": int(candidate_edges),
                    "observed_tp": int(observed_tp),
                    "expected_tp_random": float(expected_tp),
                    "tp_enrichment_vs_random": float(enrichment),
                    "p_value_hypergeom_ge_tp": float(p_ge_tp),
                }
            )

    metrics_df = pd.DataFrame(rows).sort_values(["reference", "f1", "aupr"], ascending=[True, False, False])
    metrics_df.to_csv(out_dir / "independent_perturbation_metrics.csv", index=False)

    top_df = (
        metrics_df.sort_values(["reference", "f1", "aupr"], ascending=[True, False, False])
        .groupby("reference", as_index=False)
        .head(5)
    )
    top_df.to_csv(out_dir / "independent_perturbation_top5_per_reference.csv", index=False)

    controls_df = pd.DataFrame(control_rows).sort_values(
        ["reference", "tp_enrichment_vs_random"], ascending=[True, False]
    )
    controls_df["q_value_bh_within_reference"] = (
        controls_df.groupby("reference", group_keys=False)["p_value_hypergeom_ge_tp"]
        .apply(benjamini_hochberg_qvalues)
        .astype(float)
    )
    controls_df.to_csv(out_dir / "independent_perturbation_random_controls.csv", index=False)

    # Export edge-level overlaps for biological interpretation on the independent union.
    union_name = "perturbation_independent_union"
    if union_name in normalized_refs:
        union_set = normalized_refs[union_name]["set"]
        proxy_set = normalized_refs.get("omnipath_proxy", {}).get("set", set())

        overlap_rows: List[Dict[str, object]] = []
        overlap_count_rows: List[Dict[str, object]] = []
        for method, pred_edges in method_predictions.items():
            edge_tuples = list(
                map(tuple, pred_edges[["source", "target"]].itertuples(index=False, name=None))
            )
            overlap_mask = [edge in union_set for edge in edge_tuples]
            overlap = pred_edges.loc[overlap_mask].copy()
            if overlap.empty:
                overlap_count_rows.append(
                    {
                        "method": method,
                        "overlap_edges": 0,
                        "union_only_overlap_edges": 0,
                        "proxy_overlap_edges": 0,
                    }
                )
                continue

            overlap["edge_in_proxy"] = overlap.apply(
                lambda row: (row["source"], row["target"]) in proxy_set, axis=1
            )
            overlap["edge_union_only"] = ~overlap["edge_in_proxy"]
            overlap_count_rows.append(
                {
                    "method": method,
                    "overlap_edges": int(overlap.shape[0]),
                    "union_only_overlap_edges": int(overlap["edge_union_only"].sum()),
                    "proxy_overlap_edges": int(overlap["edge_in_proxy"].sum()),
                }
            )
            for _, edge_row in overlap.sort_values("score", ascending=False).iterrows():
                overlap_rows.append(
                    {
                        "method": method,
                        "source": edge_row["source"],
                        "target": edge_row["target"],
                        "score": float(edge_row["score"]),
                        "edge_in_proxy": bool(edge_row["edge_in_proxy"]),
                        "edge_union_only": bool(edge_row["edge_union_only"]),
                    }
                )

        overlap_df = pd.DataFrame(
            overlap_rows,
            columns=["method", "source", "target", "score", "edge_in_proxy", "edge_union_only"],
        )
        if not overlap_df.empty:
            overlap_df = overlap_df.sort_values(
                ["method", "score", "source", "target"], ascending=[True, False, True, True]
            )
        overlap_df.to_csv(out_dir / "independent_union_method_overlap_edges.csv", index=False)

        overlap_counts_df = pd.DataFrame(overlap_count_rows).sort_values(
            ["overlap_edges", "union_only_overlap_edges"], ascending=[False, False]
        )
        overlap_counts_df.to_csv(out_dir / "independent_union_method_overlap_counts.csv", index=False)

        union_rank = (
            metrics_df[metrics_df["reference"] == union_name]
            .sort_values(["f1", "aupr"], ascending=False)
            .head(8)
        )
        top_methods = union_rank["method"].tolist()
        if top_methods and not overlap_df.empty:
            top_overlap = overlap_df[overlap_df["method"].isin(top_methods)].copy()
            recurrence_rows: List[Dict[str, object]] = []
            grouped = top_overlap.groupby(["source", "target", "edge_in_proxy", "edge_union_only"])
            for (source, target, edge_in_proxy, edge_union_only), group in grouped:
                methods = sorted(group["method"].unique().tolist())
                recurrence_rows.append(
                    {
                        "source": source,
                        "target": target,
                        "edge_in_proxy": bool(edge_in_proxy),
                        "edge_union_only": bool(edge_union_only),
                        "overlap_top8_methods": len(methods),
                        "methods": "|".join(methods),
                    }
                )
            recurrence_df = pd.DataFrame(recurrence_rows).sort_values(
                ["overlap_top8_methods", "source", "target"], ascending=[False, True, True]
            )
            recurrence_df.to_csv(out_dir / "independent_union_top8_edge_recurrence.csv", index=False)

    # Cross-reference ranking consistency for independent union.
    rank_rows: List[Dict[str, object]] = []
    from scipy.stats import spearmanr

    pivot = metrics_df.pivot(index="method", columns="reference", values="f1")
    if "perturbation_independent_union" in pivot.columns:
        union_scores = pivot["perturbation_independent_union"]
        for ref in pivot.columns:
            if ref == "perturbation_independent_union":
                continue
            pair = pd.concat([union_scores, pivot[ref]], axis=1, join="inner").dropna()
            if pair.shape[0] >= 3:
                rho, pval = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                rank_rows.append(
                    {
                        "reference": ref,
                        "spearman_r_f1_vs_independent_union": float(rho),
                        "spearman_p_value": float(pval),
                        "n_methods": int(pair.shape[0]),
                    }
                )
    pd.DataFrame(rank_rows).sort_values("reference").to_csv(
        out_dir / "independent_union_rank_correlations.csv", index=False
    )

    render_figures(metrics_df, out_dir)


if __name__ == "__main__":
    main()
