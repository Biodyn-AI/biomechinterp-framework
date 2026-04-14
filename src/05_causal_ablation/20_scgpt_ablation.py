#!/usr/bin/env python3
"""
scGPT attention head ablation (edge-level) — cross-architecture replication
of the Geneformer head-ablation null result.

Motivation (reviewer R1.4): the paper's causal ablation is restricted to
Geneformer V2-316M because it has a clean BERT head_mask interface. The
reviewer asked to extend ablation evidence to scGPT to test whether the
distributed-redundancy finding is architecture-specific or generalizes.

Approach: edge-level ablation on cached scGPT per-head attention.
The cached files contain per-(layer, head, source_gene, target_gene)
attention scores from scGPT whole-human checkpoint applied to three
Tabula Sapiens tissue subsets (immune/brain/kidney) over a shared
1200-gene HVG vocabulary. We can simulate "head ablation" by recomputing
the pooled attention edge scores across only the non-ablated heads, then
re-evaluating the resulting GRN against TRRUST direction-known pairs.

This is a first-order approximation to forward-pass head ablation: it
captures the direct loss of pairwise attention contribution from ablated
heads, but does not propagate through the residual stream as a true
forward-pass intervention would. We are explicit about this caveat in
the paper. The test answers a meaningful question: does the per-head
TRRUST recovery signal concentrate in a few heads (such that pooled AUROC
drops sharply when those heads are removed) or is it distributed across
many heads (such that pooled AUROC remains stable)?

Conditions (mirroring the Geneformer 13-condition battery):
  baseline               — all 96 heads (12 x 8) included
  reg_top_k              — exclude top-k TRRUST-AUROC heads (k = 5, 10, 20, 50)
  bottom_k               — exclude bottom-k TRRUST-AUROC heads (k = 5)
  random_k_repeat_r      — exclude k uniformly random heads (3 repeats per k)
  layer_k                — exclude all heads at the best perturbation layer

Output:
- results/scgpt_ablation/per_head_trrust.json   per-(layer, head) AUROC
- results/scgpt_ablation/ablation_results.json  per-condition pooled AUROC
- results/scgpt_ablation/summary.json           top-level summary
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 1)

REPO_ROOT = Path(__file__).resolve().parents[2]
SISTER_REPO = Path("/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp")
ATTN_DIR = SISTER_REPO / "outputs" / "head_layer_checkpoint_replication"
HVG_H5AD = ATTN_DIR / "immune_shared_hvg1200_processed.h5ad"
TRRUST_PATH = REPO_ROOT / "src" / "02_cssi_method" / "cssi_real_data" / "results" / "trrust_human.tsv"
OUT_DIR = REPO_ROOT / "results" / "scgpt_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TISSUES = ["immune", "kidney", "brain"]


def load_genes() -> list[str]:
    with h5py.File(HVG_H5AD, "r") as f:
        idx = f["var"]["_index"][:]
        genes = [x.decode() if isinstance(x, bytes) else x for x in idx]
    return genes


def load_attention(tissue: str) -> np.ndarray:
    """Returns shape (n_layers, n_heads, n_genes, n_genes) per-head attention scores."""
    if tissue == "immune":
        # immune is "whole_human" in the cache
        path = ATTN_DIR / "whole_human" / "attention_scores_head_layer.npy"
    else:
        path = ATTN_DIR / tissue / "attention_scores_head_layer.npy"
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    # Symmetrize (scGPT attention is directed but we treat as undirected for edge scoring)
    # We keep it directional; symmetry is handled at edge scoring step
    return arr


def load_trrust(genes: list[str], direction_known_only: bool = True) -> np.ndarray:
    df = pd.read_csv(
        TRRUST_PATH, sep="\t", header=None,
        names=["tf", "target", "mode", "pmid"],
    )
    if direction_known_only:
        df = df[df["mode"].isin(["Activation", "Repression"])]
    gene_idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    edges = np.zeros((n, n), dtype=bool)
    n_in = 0
    for _, row in df.iterrows():
        tf, tgt = row["tf"], row["target"]
        if tf in gene_idx and tgt in gene_idx:
            edges[gene_idx[tf], gene_idx[tgt]] = True
            n_in += 1
    print(f"  TRRUST direction-known pairs in HVG: {n_in}")
    return edges


def edge_scores_from_heads(
    attn: np.ndarray,
    head_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Pool attention across heads (and layers) excluding masked heads.

    Args:
        attn: (n_layers, n_heads, n_genes, n_genes)
        head_mask: (n_layers, n_heads) bool, True = INCLUDE this head, False = ablate

    Returns:
        (n_genes, n_genes) pooled attention edge score (mean over included heads).
    """
    n_layers, n_heads, n_genes, _ = attn.shape
    if head_mask is None:
        head_mask = np.ones((n_layers, n_heads), dtype=bool)
    incl = head_mask.astype(np.float32)
    # weighted mean: sum over (layer, head) of attn * mask, divided by total included
    incl_count = incl.sum()
    if incl_count == 0:
        return np.zeros((n_genes, n_genes), dtype=np.float32)
    weighted = (attn * incl[:, :, None, None]).sum(axis=(0, 1))
    return weighted / incl_count


def score_edges_against_trrust(edges: np.ndarray, trrust: np.ndarray) -> tuple[float, int, int]:
    """Compute global AUROC of edges against TRRUST positives.

    Returns (auroc, n_positives, n_negatives).
    """
    pos = trrust.flatten()
    scores = edges.flatten()
    # Exclude diagonal (gene -> self) since trrust never has self-loops
    n = edges.shape[0]
    diag_mask = ~np.eye(n, dtype=bool).flatten()
    scores = scores[diag_mask]
    pos = pos[diag_mask]
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos < 5 or n_neg < 5:
        return float("nan"), n_pos, n_neg
    auroc = roc_auc_score(pos.astype(int), scores)
    return float(auroc), n_pos, n_neg


def per_head_trrust_auroc(attn: np.ndarray, trrust: np.ndarray) -> np.ndarray:
    """Return (n_layers, n_heads) AUROC of each individual head against TRRUST."""
    n_layers, n_heads, _, _ = attn.shape
    out = np.zeros((n_layers, n_heads), dtype=np.float32)
    for L in range(n_layers):
        for H in range(n_heads):
            edges = attn[L, H]
            auroc, _, _ = score_edges_against_trrust(edges, trrust)
            out[L, H] = auroc
    return out


def per_layer_pooled_auroc(attn: np.ndarray, trrust: np.ndarray) -> np.ndarray:
    """Return (n_layers,) AUROC of pooled attention at each layer (mean over heads)."""
    n_layers = attn.shape[0]
    out = np.zeros(n_layers, dtype=np.float32)
    for L in range(n_layers):
        edges = attn[L].mean(axis=0)
        auroc, _, _ = score_edges_against_trrust(edges, trrust)
        out[L] = auroc
    return out


def run_ablation_battery(
    attn: np.ndarray,
    trrust: np.ndarray,
    per_head_auroc: np.ndarray,
    rng: np.random.Generator,
):
    """Run the 13-condition ablation battery on the pooled attention.

    Returns a dict of condition -> dict(auroc, delta, k, description).
    """
    n_layers, n_heads = per_head_auroc.shape
    total_heads = n_layers * n_heads

    # Baseline: all heads
    baseline_edges = edge_scores_from_heads(attn)
    baseline_auroc, n_pos, n_neg = score_edges_against_trrust(baseline_edges, trrust)
    print(f"  baseline pooled AUROC: {baseline_auroc:.4f}  (n_pos={n_pos})")

    # Flatten head ranking by descending TRRUST AUROC
    flat_aurocs = per_head_auroc.flatten()
    # Indices sorted from best (highest) to worst
    order_desc = np.argsort(-flat_aurocs)
    order_asc = np.argsort(flat_aurocs)

    def make_mask(exclude_flat_indices):
        mask = np.ones(total_heads, dtype=bool)
        mask[exclude_flat_indices] = False
        return mask.reshape(n_layers, n_heads)

    results = {
        "baseline": {
            "auroc": float(baseline_auroc),
            "delta": 0.0,
            "k": 0,
            "description": f"all {total_heads} heads included",
            "n_positives": n_pos,
            "n_negatives": n_neg,
        }
    }

    # Top-K TRRUST-ranked ablation
    for k in [5, 10, 20, 50]:
        excl = order_desc[:k]
        mask = make_mask(excl)
        edges = edge_scores_from_heads(attn, mask)
        a, _, _ = score_edges_against_trrust(edges, trrust)
        results[f"reg_top_{k}"] = {
            "auroc": float(a),
            "delta": float(a - baseline_auroc),
            "k": k,
            "description": f"exclude top-{k} TRRUST-ranked heads",
        }

    # Bottom-K
    for k in [5]:
        excl = order_asc[:k]
        mask = make_mask(excl)
        edges = edge_scores_from_heads(attn, mask)
        a, _, _ = score_edges_against_trrust(edges, trrust)
        results[f"bottom_{k}"] = {
            "auroc": float(a),
            "delta": float(a - baseline_auroc),
            "k": k,
            "description": f"exclude bottom-{k} TRRUST-ranked heads",
        }

    # Random-K (3 repeats per k for k in {5, 10, 20, 50})
    for k in [5, 10, 20, 50]:
        for rep in range(3):
            excl = rng.choice(total_heads, size=k, replace=False)
            mask = make_mask(excl)
            edges = edge_scores_from_heads(attn, mask)
            a, _, _ = score_edges_against_trrust(edges, trrust)
            results[f"random_{k}_rep{rep}"] = {
                "auroc": float(a),
                "delta": float(a - baseline_auroc),
                "k": k,
                "description": f"exclude {k} random heads (repeat {rep})",
            }

    # Full layer ablation: exclude all heads at the best layer
    pooled_per_layer = np.array(
        [score_edges_against_trrust(attn[L].mean(axis=0), trrust)[0] for L in range(n_layers)]
    )
    best_layer = int(np.nanargmax(pooled_per_layer))
    excl_layer = np.array([best_layer * n_heads + h for h in range(n_heads)])
    mask = make_mask(excl_layer)
    edges = edge_scores_from_heads(attn, mask)
    a, _, _ = score_edges_against_trrust(edges, trrust)
    results[f"full_layer_L{best_layer}"] = {
        "auroc": float(a),
        "delta": float(a - baseline_auroc),
        "k": n_heads,
        "description": f"exclude all heads at best layer L{best_layer} (pooled AUROC {pooled_per_layer[best_layer]:.3f})",
    }

    return results, baseline_auroc, pooled_per_layer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-known", action="store_true",
                        help="Restrict TRRUST to Activation/Repression entries (stricter, fewer pairs)")
    args = parser.parse_args()

    print("Loading shared HVG gene list...")
    genes = load_genes()
    print(f"  {len(genes)} genes")

    print(f"Loading TRRUST (direction_known_only={args.direction_known})...")
    trrust = load_trrust(genes, direction_known_only=args.direction_known)

    summary = {
        "n_genes": len(genes),
        "n_trrust_positives": int(trrust.sum()),
        "tissues": {},
    }

    for tissue in TISSUES:
        print(f"\n=== Tissue: {tissue} ===")
        attn = load_attention(tissue)
        n_layers, n_heads, _, _ = attn.shape
        print(f"  attention shape: {attn.shape}  ({n_layers}x{n_heads} = {n_layers*n_heads} heads)")

        print("  Computing per-head TRRUST AUROC...")
        ph_auroc = per_head_trrust_auroc(attn, trrust)
        print(f"  per-head AUROC range: {np.nanmin(ph_auroc):.3f} -- {np.nanmax(ph_auroc):.3f}")
        print(f"  per-head AUROC mean: {np.nanmean(ph_auroc):.3f} +/- {np.nanstd(ph_auroc):.3f}")

        # Save per-head ranking for later inspection
        ph_path = OUT_DIR / f"per_head_trrust_{tissue}.json"
        with open(ph_path, "w") as f:
            json.dump(ph_auroc.tolist(), f, indent=2)

        rng = np.random.default_rng(2026)
        print("  Running 13-condition ablation battery...")
        cond_results, baseline_auroc, pooled_per_layer = run_ablation_battery(
            attn, trrust, ph_auroc, rng
        )

        # Compute random-baseline summary statistics
        deltas_random_20 = [
            cond_results[f"random_20_rep{r}"]["delta"] for r in range(3)
        ]
        deltas_random_5 = [
            cond_results[f"random_5_rep{r}"]["delta"] for r in range(3)
        ]
        delta_top_5 = cond_results["reg_top_5"]["delta"]
        delta_top_50 = cond_results["reg_top_50"]["delta"]

        print("\n  Condition summary:")
        print(f"    baseline:                     {baseline_auroc:.4f}")
        print(f"    reg_top_5     delta:          {delta_top_5:+.4f}")
        print(f"    reg_top_10    delta:          {cond_results['reg_top_10']['delta']:+.4f}")
        print(f"    reg_top_20    delta:          {cond_results['reg_top_20']['delta']:+.4f}")
        print(f"    reg_top_50    delta:          {delta_top_50:+.4f}")
        print(f"    bottom_5      delta:          {cond_results['bottom_5']['delta']:+.4f}")
        print(f"    random_5      mean delta:     {np.mean(deltas_random_5):+.4f}")
        print(f"    random_20     mean delta:     {np.mean(deltas_random_20):+.4f}")

        summary["tissues"][tissue] = {
            "baseline_auroc": float(baseline_auroc),
            "per_head_auroc_range": [float(np.nanmin(ph_auroc)), float(np.nanmax(ph_auroc))],
            "per_head_auroc_mean": float(np.nanmean(ph_auroc)),
            "per_head_auroc_std": float(np.nanstd(ph_auroc)),
            "per_layer_pooled_auroc": pooled_per_layer.tolist(),
            "best_layer": int(np.nanargmax(pooled_per_layer)),
            "conditions": cond_results,
            "deltas_summary": {
                "reg_top_5": delta_top_5,
                "reg_top_10": cond_results["reg_top_10"]["delta"],
                "reg_top_20": cond_results["reg_top_20"]["delta"],
                "reg_top_50": delta_top_50,
                "random_5_mean": float(np.mean(deltas_random_5)),
                "random_20_mean": float(np.mean(deltas_random_20)),
            },
        }

    out_path = OUT_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
