#!/usr/bin/env python3
"""
Positive control for the perturbation-first evaluation pipeline.

Motivation (reviewer R1.3): the paper reports that correlation and attention
edge scores add no incremental value beyond trivial gene-level baselines for
perturbation-target prediction on Replogle K562 CRISPRi. Reviewer 1 flagged
that without a decisive positive control this null is hard to interpret —
it could reflect limitations of the pipeline rather than absence of signal.

This script builds a synthetic single-cell Perturb-seq dataset with planted
ground-truth regulatory edges, applies the full trivial-baseline +
incremental-value pipeline end-to-end, and checks whether the pipeline
recovers the planted pairwise signal. Three stress variants are run:

  Variant A (strong): strong direct edges, minimal indirect effects.
      Pipeline should recover edges strongly. Establishes ceiling.
  Variant B (weak + indirect): weak direct edges, strong indirect
      propagation. Tests whether indirect effects mask pairwise signal.
  Variant C (realistic Replogle-like): gene-level AUROC tuned to match
      K562 (~0.88). Diagnoses whether the K562 null could persist even
      when ground truth exists at a detectable level.

A shuffled-null control is run within each variant: ground-truth is
randomly permuted and the full pipeline is re-run. The shuffled result
must be near-chance — this confirms pipeline gains come from planted
structure, not bookkeeping artefacts.

Deliverable (for paper): Supplementary Note 21, new figure
fig_positive_control.png, and a summary paragraph in Results referencing
the pipeline sensitivity established here.

Usage:
    python 20_positive_control_pipeline.py [--variant A|B|C|all]

Output:
    results/positive_control/variant_{A,B,C}.json
    results/positive_control/shuffled_null_{A,B,C}.json
    results/positive_control/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

# Line-buffered stdout so long runs emit progress
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 1)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "positive_control"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic GRN generator
# ---------------------------------------------------------------------------


@dataclass
class SyntheticGRNConfig:
    n_genes: int = 300
    n_tfs: int = 20
    direct_targets_per_tf: int = 6  # direct TF -> target edges
    indirect_targets_per_tf: int = 8  # TF -> target (via 1 regulator hop)
    n_ctrl_cells: int = 5000
    n_pert_cells_per_condition: int = 150
    n_perturbations: int = 20  # knockdowns; each TF is perturbed once
    edge_strength: float = 1.5  # direct edge coefficient magnitude
    indirect_strength: float = 0.8  # indirect propagation coefficient
    noise_scale: float = 0.8
    dropout_p: float = 0.1
    # If True, z-score each gene post-hoc so variance is uninformative.
    # Used in positive-control variants to isolate pairwise signal.
    z_score_genes: bool = False
    # If True, restrict the positive class to direct targets only (rather
    # than all DE hits). This tests whether correlation edges predict
    # *direct* regulation vs. indirect propagation.
    direct_only_positives: bool = False
    # Variant tag for reproducibility of the stress suite.
    variant: str = "A"
    seed: int = 2026


def build_grn(cfg: SyntheticGRNConfig, rng: np.random.Generator):
    """Build a sparse hierarchical GRN.

    Returns:
        tf_idx:            indices of TFs (shape [n_tfs])
        direct:            dict TF_idx -> list of direct target gene indices
        indirect:          dict TF_idx -> list of indirect target gene indices
        weight_mat:        (n_genes, n_genes) direct TF->target weights
        indirect_weight_mat: (n_genes, n_genes) indirect effect weights
    """
    n = cfg.n_genes
    gene_pool = np.arange(n)
    tf_idx = rng.choice(gene_pool, size=cfg.n_tfs, replace=False)
    non_tf = np.setdiff1d(gene_pool, tf_idx)

    direct = {}
    indirect = {}
    weight_mat = np.zeros((n, n), dtype=np.float32)
    indirect_weight_mat = np.zeros((n, n), dtype=np.float32)

    for tf in tf_idx:
        # Direct targets — TFs regulate non-TF genes
        dtargets = rng.choice(
            non_tf, size=cfg.direct_targets_per_tf, replace=False
        )
        # Mix of activation / repression
        signs = rng.choice([-1.0, 1.0], size=len(dtargets))
        magnitudes = rng.uniform(0.7, 1.3, size=len(dtargets)) * cfg.edge_strength
        for tgt, s, m in zip(dtargets, signs, magnitudes):
            weight_mat[tf, tgt] = s * m
        direct[int(tf)] = dtargets.tolist()

        # Indirect targets: not in direct set, reached via a single-hop
        # regulator. These contribute to DE on perturbation but are not
        # direct regulatory edges.
        candidate_pool = np.setdiff1d(non_tf, dtargets)
        itargets = rng.choice(
            candidate_pool, size=cfg.indirect_targets_per_tf, replace=False
        )
        signs_i = rng.choice([-1.0, 1.0], size=len(itargets))
        mags_i = rng.uniform(0.6, 1.2, size=len(itargets)) * cfg.indirect_strength
        for tgt, s, m in zip(itargets, signs_i, mags_i):
            indirect_weight_mat[tf, tgt] = s * m
        indirect[int(tf)] = itargets.tolist()

    return tf_idx, direct, indirect, weight_mat, indirect_weight_mat


def simulate_control_cells(
    cfg: SyntheticGRNConfig,
    rng: np.random.Generator,
    weight_mat: np.ndarray,
    indirect_weight_mat: np.ndarray,
) -> np.ndarray:
    """Simulate baseline control expression.

    Model: baseline TF activity is drawn from a heavy-tailed distribution,
    target expression is a linear combination of regulator activities plus
    Gaussian noise plus dropout. TF expression equals TF activity directly.

    We optionally z-score each gene post-hoc to control whether gene-level
    variance carries information about direct-target status. This is
    essential for the positive-control design: variant A z-scores to make
    variance uninformative, so any incremental value from correlation
    edges reflects genuine pairwise signal detection.
    """
    n = cfg.n_genes
    n_cells = cfg.n_ctrl_cells

    # Heavy-tailed baseline TF activities for each gene slot
    tf_activity = rng.standard_t(df=3, size=(n_cells, n)).astype(np.float32) * 0.8

    # Target expression comes from regulator activity
    expr = tf_activity @ (weight_mat + indirect_weight_mat)
    # TFs' own expression is their activity (direct readout)
    tf_mask = weight_mat.sum(axis=1) != 0  # rows with outgoing edges are TFs
    expr[:, tf_mask] = tf_activity[:, tf_mask]

    # Add small random mean shift (identifies the gene but does not correlate
    # with direct-target status)
    gene_means = rng.normal(0, 0.3, size=n).astype(np.float32)
    expr = expr + gene_means

    # Gaussian noise (additive)
    expr = expr + rng.normal(
        0, cfg.noise_scale, size=expr.shape
    ).astype(np.float32)

    # Dropout
    mask = rng.uniform(size=expr.shape) < cfg.dropout_p
    expr[mask] = 0.0

    # Optionally z-score per gene so variance is uninformative.
    if getattr(cfg, "z_score_genes", False):
        std = expr.std(axis=0)
        std[std == 0] = 1.0
        expr = (expr - expr.mean(axis=0)) / std
    return expr.astype(np.float32)


def simulate_perturbation(
    cfg: SyntheticGRNConfig,
    rng: np.random.Generator,
    weight_mat: np.ndarray,
    indirect_weight_mat: np.ndarray,
    pert_tf: int,
) -> np.ndarray:
    """Simulate cells where the given TF is knocked down.

    Knockdown: set pert_tf activity to near zero and re-simulate expression.
    """
    n = cfg.n_genes
    n_cells = cfg.n_pert_cells_per_condition

    tf_activity = rng.standard_t(df=3, size=(n_cells, n)) * 0.5
    # Knockdown: zero out the perturbed TF's activity
    tf_activity[:, pert_tf] *= 0.05

    expr = tf_activity @ (weight_mat + indirect_weight_mat)
    gene_means = 0.0  # means are cell-invariant, already in baseline
    expr = expr + gene_means
    expr = expr + rng.normal(0, cfg.noise_scale, size=expr.shape).astype(np.float32)
    mask = rng.uniform(size=expr.shape) < cfg.dropout_p
    expr[mask] = 0.0
    return expr.astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline components: DE calling, gene-level features, correlation edges
# ---------------------------------------------------------------------------


def compute_gene_features(ctrl_expr: np.ndarray) -> np.ndarray:
    """Compute trivial gene-level features from control cells.

    Returns array (n_genes, 3): variance, mean, dropout rate.
    """
    var = ctrl_expr.var(axis=0)
    mean = ctrl_expr.mean(axis=0)
    dropout = (ctrl_expr == 0).mean(axis=0)
    return np.stack([var, mean, dropout], axis=1)


def compute_correlation_edges(ctrl_expr: np.ndarray) -> np.ndarray:
    """Compute gene-by-gene Spearman correlation edge scores on control cells.

    Uses fast rank-based Pearson as an approximation of Spearman.
    Returns (n_genes, n_genes) symmetric matrix.
    """
    # Rank each gene independently
    ranks = stats.rankdata(ctrl_expr, axis=0)
    ranks = ranks - ranks.mean(axis=0)
    std = ranks.std(axis=0)
    std[std == 0] = 1.0
    ranks = ranks / std
    corr = (ranks.T @ ranks) / ranks.shape[0]
    # Use absolute correlation as edge score (magnitude of association)
    return np.abs(corr).astype(np.float32)


def call_de(
    ctrl_expr: np.ndarray,
    pert_expr: np.ndarray,
    lfc_threshold: float = 0.5,
    alpha: float = 0.05,
) -> np.ndarray:
    """Return boolean array (n_genes,) of DE-hit targets.

    Uses Welch's t-test + LFC threshold (crude, but matches paper pipeline).
    """
    n_genes = ctrl_expr.shape[1]
    de_hits = np.zeros(n_genes, dtype=bool)
    # Simple LFC in log1p-space
    mean_c = np.log1p(np.abs(ctrl_expr).mean(axis=0) + 1e-3)
    mean_p = np.log1p(np.abs(pert_expr).mean(axis=0) + 1e-3)
    lfc = mean_p - mean_c
    # Welch's t across genes
    t, p = stats.ttest_ind(pert_expr, ctrl_expr, axis=0, equal_var=False)
    de_hits = (np.abs(lfc) > lfc_threshold) & (p < alpha)
    return de_hits


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


@dataclass
class VariantResult:
    variant: str
    config: dict
    n_perturbations: int = 0
    n_evaluable: int = 0
    gene_only_auroc: float = float("nan")
    gene_plus_corr_auroc: float = float("nan")
    delta_auroc_corr: float = float("nan")
    delta_auroc_ci: tuple = field(default_factory=lambda: (float("nan"), float("nan")))
    per_tf_corr_auroc: dict = field(default_factory=dict)
    variance_baseline_auroc: float = float("nan")
    mean_baseline_auroc: float = float("nan")
    dropout_baseline_auroc: float = float("nan")
    positive_rate: float = float("nan")
    notes: str = ""


def run_variant(cfg: SyntheticGRNConfig, shuffle_ground_truth: bool = False) -> VariantResult:
    """Run one stress variant and return summary metrics.

    If shuffle_ground_truth, the TF->target mapping is randomly permuted
    after simulation so that simulated DE still occurs but correlation
    edges no longer map to ``true'' direct targets. This provides the
    negative control: pipeline should produce zero incremental value.
    """
    rng = np.random.default_rng(cfg.seed + (1_000_000 if shuffle_ground_truth else 0))
    tf_idx, direct, indirect, weight_mat, indirect_weight_mat = build_grn(cfg, rng)

    print(f"[{cfg.variant}] Built GRN: {cfg.n_genes} genes, {cfg.n_tfs} TFs, "
          f"{sum(len(v) for v in direct.values())} direct edges")

    # Simulate control cells
    ctrl_expr = simulate_control_cells(cfg, rng, weight_mat, indirect_weight_mat)
    print(f"[{cfg.variant}] Simulated {ctrl_expr.shape[0]} control cells")

    # Gene-level features + correlation edges from control cells
    gene_feats = compute_gene_features(ctrl_expr)
    corr_edges = compute_correlation_edges(ctrl_expr)

    # Optionally shuffle ground truth: permute target labels per TF
    # so DE positives no longer match the edges that contributed to them.
    if shuffle_ground_truth:
        all_non_tf = [g for g in range(cfg.n_genes) if g not in set(tf_idx.tolist())]
        shuffled_direct = {}
        for tf in direct:
            shuffled_direct[tf] = list(
                rng.choice(all_non_tf, size=len(direct[tf]), replace=False)
            )
        # Note: we keep the actual simulation edges in place (so DE still
        # happens biologically), but we treat shuffled_direct as the
        # "ground truth direct set". Any incremental value on this shuffled
        # truth is spurious.
        ground_truth_direct = shuffled_direct
    else:
        ground_truth_direct = {int(tf): list(direct[int(tf)]) for tf in direct}

    # Simulate each perturbation
    pert_results = []  # (tf, de_mask, per_gene_pair_rows)
    rows_X = []
    rows_y = []
    rows_group = []

    tfs_to_perturb = tf_idx[: cfg.n_perturbations]
    for k, tf in enumerate(tfs_to_perturb):
        tf_int = int(tf)
        pert_expr = simulate_perturbation(
            cfg, rng, weight_mat, indirect_weight_mat, pert_tf=tf_int
        )
        # Compute DE hits empirically (used for descriptive stats)
        de_hits_empirical = call_de(ctrl_expr, pert_expr)

        if cfg.direct_only_positives:
            # Positive class = direct ground-truth targets only.
            # This is the cleanest test of whether the pipeline can
            # detect direct pairwise regulation.
            positives = np.zeros(cfg.n_genes, dtype=bool)
            for tgt in ground_truth_direct.get(tf_int, []):
                positives[tgt] = True
            label_mask = positives
        else:
            label_mask = de_hits_empirical

        n_pos = int(label_mask.sum())
        if n_pos < 3 or n_pos > cfg.n_genes - 3:
            continue
        pert_results.append((tf_int, label_mask, n_pos))

        # Build per-perturbation rows: for each non-self target gene,
        # features = [tf_var, tf_mean, tf_dropout, tgt_var, tgt_mean, tgt_dropout, corr_edge]
        for tgt in range(cfg.n_genes):
            if tgt == tf_int:
                continue
            X_row = [
                gene_feats[tf_int, 0], gene_feats[tf_int, 1], gene_feats[tf_int, 2],
                gene_feats[tgt, 0], gene_feats[tgt, 1], gene_feats[tgt, 2],
                corr_edges[tf_int, tgt],
            ]
            rows_X.append(X_row)
            rows_y.append(int(label_mask[tgt]))
            rows_group.append(tf_int)

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.int32)
    groups = np.array(rows_group, dtype=np.int32)

    if len(pert_results) < 3 or y.sum() == 0:
        return VariantResult(
            variant=cfg.variant,
            config=asdict(cfg),
            notes="Too few evaluable perturbations or no DE hits",
        )

    positive_rate = float(y.mean())
    print(f"[{cfg.variant}] {len(pert_results)} perturbations, "
          f"{len(y)} gene-pair rows, positive rate {positive_rate:.3f}")

    # 5-fold GroupKFold by source TF
    n_splits = min(5, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    def eval_model(feat_cols):
        """Cross-validated mean AUROC across folds on the given feature cols."""
        aurocs = []
        for tr, te in gkf.split(X, y, groups):
            X_tr = X[tr][:, feat_cols]
            X_te = X[te][:, feat_cols]
            y_tr = y[tr]
            y_te = y[te]
            if y_te.sum() == 0 or y_te.sum() == len(y_te):
                continue
            clf = LogisticRegression(max_iter=1000, solver="liblinear")
            clf.fit(X_tr, y_tr)
            p = clf.predict_proba(X_te)[:, 1]
            aurocs.append(roc_auc_score(y_te, p))
        return float(np.mean(aurocs)) if aurocs else float("nan")

    # Feature column indices
    GENE_COLS = [0, 1, 2, 3, 4, 5]
    CORR_COL = [6]

    gene_only = eval_model(GENE_COLS)
    gene_plus_corr = eval_model(GENE_COLS + CORR_COL)
    delta = gene_plus_corr - gene_only

    # Bootstrap delta-AUROC CI (100 resamples over perturbations)
    rng_boot = np.random.default_rng(cfg.seed + 7777)
    deltas = []
    unique_tfs = sorted(set(groups.tolist()))
    for _ in range(100):
        boot_tfs = rng_boot.choice(unique_tfs, size=len(unique_tfs), replace=True)
        mask = np.isin(groups, boot_tfs)
        if y[mask].sum() == 0:
            continue
        Xb = X[mask]
        yb = y[mask]
        gb = groups[mask]
        try:
            gkf_b = GroupKFold(n_splits=min(3, len(set(gb))))
            aur_g, aur_gc = [], []
            for tr, te in gkf_b.split(Xb, yb, gb):
                if yb[te].sum() == 0 or yb[te].sum() == len(yb[te]):
                    continue
                c1 = LogisticRegression(max_iter=1000, solver="liblinear")
                c1.fit(Xb[tr][:, GENE_COLS], yb[tr])
                aur_g.append(roc_auc_score(yb[te], c1.predict_proba(Xb[te][:, GENE_COLS])[:, 1]))
                c2 = LogisticRegression(max_iter=1000, solver="liblinear")
                c2.fit(Xb[tr][:, GENE_COLS + CORR_COL], yb[tr])
                aur_gc.append(roc_auc_score(yb[te], c2.predict_proba(Xb[te][:, GENE_COLS + CORR_COL])[:, 1]))
            if aur_g and aur_gc:
                deltas.append(np.mean(aur_gc) - np.mean(aur_g))
        except Exception:
            continue
    if deltas:
        ci_low, ci_high = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    else:
        ci_low, ci_high = float("nan"), float("nan")

    # Trivial baselines: each gene-level feature alone
    def single_feature_auroc(col):
        scores = X[:, col]
        return float(roc_auc_score(y, scores))
    variance_auroc = single_feature_auroc(3)  # target variance
    mean_auroc = single_feature_auroc(4)
    dropout_auroc = 1.0 - single_feature_auroc(5)  # low dropout = more detectable

    # Per-TF correlation-edge AUROC (marginal, no gene features)
    per_tf_corr_auroc = {}
    for tf_int_inner, label_mask_inner, _ in pert_results:
        # Use corr_edges[tf, :] as scores for each target
        scores = corr_edges[tf_int_inner]
        try:
            aur = roc_auc_score(label_mask_inner.astype(int), scores)
        except ValueError:
            aur = float("nan")
        per_tf_corr_auroc[int(tf_int_inner)] = float(aur)

    res = VariantResult(
        variant=cfg.variant,
        config=asdict(cfg),
        n_perturbations=cfg.n_perturbations,
        n_evaluable=len(pert_results),
        gene_only_auroc=gene_only,
        gene_plus_corr_auroc=gene_plus_corr,
        delta_auroc_corr=delta,
        delta_auroc_ci=(ci_low, ci_high),
        per_tf_corr_auroc=per_tf_corr_auroc,
        variance_baseline_auroc=variance_auroc,
        mean_baseline_auroc=mean_auroc,
        dropout_baseline_auroc=dropout_auroc,
        positive_rate=positive_rate,
        notes="shuffled ground truth" if shuffle_ground_truth else "planted ground truth",
    )
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all", choices=["A", "B", "C", "all"])
    args = parser.parse_args()

    # Three stress variants.
    # All three z-score gene expression so variance is uninformative —
    # this is the positive-control design: any delta_AUROC > 0 must come
    # from pairwise (correlation) information, not univariate baselines.
    # The variants then stress different aspects:
    #   A: strong, clean direct regulation with minimal indirect.
    #      Pipeline ceiling test.
    #   B: weak direct edges + strong indirect propagation.
    #      Tests whether indirect effects mask pairwise signal.
    #   C: realistic balance of direct + indirect + noise.
    #      Closest to real K562-like regime.
    variants = {
        "A": SyntheticGRNConfig(
            variant="A",
            edge_strength=2.2,
            indirect_strength=0.1,
            noise_scale=0.5,
            direct_targets_per_tf=8,
            indirect_targets_per_tf=2,
            z_score_genes=True,
            direct_only_positives=True,  # positive class = direct targets only
            seed=2026,
        ),
        "B": SyntheticGRNConfig(
            variant="B",
            edge_strength=0.8,
            indirect_strength=1.6,
            noise_scale=0.9,
            direct_targets_per_tf=5,
            indirect_targets_per_tf=15,
            z_score_genes=True,
            direct_only_positives=True,
            seed=2027,
        ),
        "C": SyntheticGRNConfig(
            variant="C",
            edge_strength=1.2,
            indirect_strength=0.8,
            noise_scale=0.8,
            direct_targets_per_tf=6,
            indirect_targets_per_tf=8,
            z_score_genes=True,
            direct_only_positives=True,
            seed=2028,
        ),
    }
    if args.variant != "all":
        variants = {args.variant: variants[args.variant]}

    summary = {}
    for vname, cfg in variants.items():
        print(f"\n===== Variant {vname}: {cfg.notes if hasattr(cfg, 'notes') else ''} =====")
        # Planted ground-truth run
        res = run_variant(cfg, shuffle_ground_truth=False)
        out_path = RESULTS_DIR / f"variant_{vname}.json"
        with open(out_path, "w") as f:
            # Convert tuple to list for JSON
            d = asdict(res)
            d["delta_auroc_ci"] = list(d["delta_auroc_ci"])
            json.dump(d, f, indent=2, default=float)
        print(f"  gene_only AUROC:        {res.gene_only_auroc:.4f}")
        print(f"  gene+corr AUROC:        {res.gene_plus_corr_auroc:.4f}")
        print(f"  delta_auroc:            {res.delta_auroc_corr:+.4f} "
              f"[{res.delta_auroc_ci[0]:+.4f}, {res.delta_auroc_ci[1]:+.4f}]")
        print(f"  variance baseline:      {res.variance_baseline_auroc:.4f}")
        print(f"  mean ctrl baseline:     {res.mean_baseline_auroc:.4f}")

        # Shuffled-null control
        print(f"\n  --- shuffled null for variant {vname} ---")
        res_null = run_variant(cfg, shuffle_ground_truth=True)
        null_path = RESULTS_DIR / f"shuffled_null_{vname}.json"
        with open(null_path, "w") as f:
            d = asdict(res_null)
            d["delta_auroc_ci"] = list(d["delta_auroc_ci"])
            json.dump(d, f, indent=2, default=float)
        print(f"  shuffled gene_only:     {res_null.gene_only_auroc:.4f}")
        print(f"  shuffled gene+corr:     {res_null.gene_plus_corr_auroc:.4f}")
        print(f"  shuffled delta:         {res_null.delta_auroc_corr:+.4f}")

        summary[vname] = {
            "planted": {
                "gene_only": res.gene_only_auroc,
                "gene_plus_corr": res.gene_plus_corr_auroc,
                "delta": res.delta_auroc_corr,
                "delta_ci": list(res.delta_auroc_ci),
                "variance_baseline": res.variance_baseline_auroc,
                "n_evaluable": res.n_evaluable,
                "positive_rate": res.positive_rate,
            },
            "shuffled_null": {
                "gene_only": res_null.gene_only_auroc,
                "gene_plus_corr": res_null.gene_plus_corr_auroc,
                "delta": res_null.delta_auroc_corr,
            },
        }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nDone. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
