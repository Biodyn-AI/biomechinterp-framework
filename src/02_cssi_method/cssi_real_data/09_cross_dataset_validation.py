#!/usr/bin/env python3
"""
CSSI Cross-Dataset Validation Experiment (scGPT brain checkpoint)

Split 497 brain cells into two non-overlapping halves.
Extract per-layer attention from scGPT brain checkpoint on each half.
Identify best layers on half A, evaluate on half B (and vice versa).
Compare against TRRUST ground truth.  Report AUROC per layer.
"""

import sys, os, gc, json, time, traceback

# CRITICAL: scGPT Setup - torchtext shim MUST be loaded first
for _p in [
    r"C:\Users\Agent\.openclaw\workspace\patch_torchtext.py",
    "/mnt/c/Users/Agent/.openclaw/workspace/patch_torchtext.py",
    "/c/Users/Agent/.openclaw/workspace/patch_torchtext.py",
]:
    if os.path.exists(_p):
        exec(open(_p).read())
        break
else:
    raise FileNotFoundError("Cannot find patch_torchtext.py")

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from pathlib import Path
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from collections import Counter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/mnt/d/openclaw/mechinterp-bio/biodyn-work/single_cell_mechinterp")
REPO_PATH = BASE / "external" / "scGPT"
CKPT_DIR  = BASE / "external" / "scGPT_checkpoints" / "brain"
DATA_PATH = Path("/mnt/d/openclaw/intelligence-augmentation/data/brain_scrna/DLPFC_11k.h5ad")
OUT_DIR   = Path("/mnt/d/openclaw/biodyn-nmi-paper/cssi_real")
REPORT_PATH = Path("/mnt/d/openclaw/biodyn-nmi-paper/CSSI_CROSS_VALIDATION_REPORT.md")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Add project root to path so `from src.*` works
sys.path.insert(0, str(BASE))

from src.data.scgpt_dataset import ScGPTDataset, ScGPTDatasetConfig, collate_scgpt
from src.interpret.attention import extract_attention_scores, finalize_attention_scores
from src.model.scgpt_loader import load_scgpt_model
from scgpt.tokenizer import GeneVocab
def load_vocab(path):
    return GeneVocab.from_file(path)
from src.model.wrapper import ScGPTWrapper
from src.utils.torch_utils import move_to_device

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_path = OUT_DIR / "cssi_cross_dataset_validation.log"
log_file = open(log_path, "w", buffering=1)

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

# ---------------------------------------------------------------------------
# Load scGPT model
# ---------------------------------------------------------------------------
def load_model(device):
    """Load scGPT brain checkpoint wrapped for attention extraction."""
    vocab_obj = load_vocab(CKPT_DIR / "vocab.json")
    with open(CKPT_DIR / "args.json") as f:
        args = json.load(f)

    model_args = {
        "ntoken": len(vocab_obj),
        "d_model": args["embsize"],
        "nhead": args["nheads"],
        "d_hid": args["d_hid"],
        "nlayers": args["nlayers"],
        "dropout": args.get("dropout", 0.2),
        "pad_token": args.get("pad_token", "<pad>"),
        "pad_value": args.get("pad_value", -2),
        "do_mvc": False,
        "do_dab": False,
        "use_batch_labels": False,
        "n_input_bins": args.get("n_bins", 51),
        "input_emb_style": args.get("input_emb_style", "continuous"),
        "cell_emb_style": "cls",
        "mvc_decoder_style": "inner product",
        "ecs_threshold": args.get("ecs_thres", 0.0),
        "explicit_zero_prob": not args.get("no_explicit_zero_prob", False),
        "use_fast_transformer": False,  # Must be False for attention capture
        "vocab": vocab_obj,
    }

    model, missing, unexpected = load_scgpt_model(
        entrypoint="scgpt.model.model.TransformerModel",
        repo_path=REPO_PATH,
        checkpoint_path=CKPT_DIR / "best_model.pt",
        device=device,
        model_args=model_args,
    )
    model = model.to(device)

    wrapped = ScGPTWrapper(
        model,
        forward_key_map={
            "gene_ids": "src",
            "gene_values": "values",
            "src_key_padding_mask": "src_key_padding_mask",
        },
    )
    log(f"Model loaded  missing={len(missing)} unexpected={len(unexpected)}")
    return wrapped, vocab_obj

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def prepare_adata(adata):
    """Convert Ensembl IDs -> gene symbols via feature_name column."""
    if "feature_name" not in adata.var.columns:
        return adata.copy()
    feature_names = adata.var["feature_name"].copy()
    seen, keep_idx, symbols = set(), [], []
    for i, sym in enumerate(feature_names):
        s = str(sym).strip() if pd.notna(sym) else ""
        if s == "":
            s = str(adata.var_names[i])
        if s not in seen:
            seen.add(s)
            keep_idx.append(i)
            symbols.append(s)
    out = adata[:, keep_idx].copy()
    out.var_names = symbols
    return out


def build_dataset(adata, vocab_obj, max_genes=1500, max_seq=512):
    """Filter to vocab genes and build ScGPTDataset."""
    adata = prepare_adata(adata)

    # Top HVG by variance
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    variances = np.var(X, axis=0)
    top_idx = np.argsort(variances)[-max_genes:]
    adata = adata[:, top_idx].copy()

    # Keep only vocab genes
    gene2id = vocab_obj.get_stoi()
    vocab_genes = [g for g in adata.var_names if g in gene2id]
    if len(vocab_genes) < 50:
        raise RuntimeError(f"Only {len(vocab_genes)} genes in vocab – too few")
    adata = adata[:, vocab_genes].copy()
    log(f"  Dataset: {adata.n_obs} cells x {adata.n_vars} genes")

    pad_id = vocab_obj["<pad>"]
    cls_id = gene2id.get("<cls>", None)

    cfg = ScGPTDatasetConfig(
        max_genes=max_seq,
        include_zero=False,
        sort_by_expression=True,
        pad_token_id=pad_id,
        cls_token_id=cls_id,
    )
    ds = ScGPTDataset(adata, gene2id, cfg)
    return ds


def extract_per_layer(wrapped_model, ds, device, batch_size=2):
    """Extract per-layer attention matrices (heads averaged)."""
    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_scgpt)
    n_genes = len(ds.gene_names)

    score_sum, score_count = extract_attention_scores(
        model=wrapped_model,
        dataloader=dl,
        n_genes=n_genes,
        device=device,
        reduce_layers=False,
        reduce_heads=True,
        ignore_index=-1,
    )

    n_layers = score_sum.shape[0]
    layer_mats = {}
    for l in range(n_layers):
        layer_mats[l] = finalize_attention_scores(score_sum[l], score_count)
    return layer_mats, list(ds.gene_names)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def load_trrust():
    """Return set of (ensembl_src, ensembl_tgt) and DataFrame."""
    trrust_path = "/tmp/trrust.tsv"
    if not os.path.exists(trrust_path):
        import urllib.request
        urllib.request.urlretrieve(
            "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv",
            trrust_path,
        )
    df = pd.read_csv(trrust_path, sep="\t", header=None,
                      names=["TF", "target", "mode", "pmid"])
    edges = set(zip(df["TF"], df["target"]))
    log(f"TRRUST raw: {len(edges)} unique directed edges")
    return edges, df


def evaluate_auroc(score_matrix, gene_list, gt_edges):
    """AUROC for one layer.  gt_edges is set of (source_symbol, target_symbol)."""
    n = len(gene_list)
    y_true, y_score = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            label = 1 if (gene_list[i], gene_list[j]) in gt_edges else 0
            y_true.append(label)
            y_score.append(float(score_matrix[i, j]))
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    n_pos = int(y_true.sum())
    if n_pos < 3:
        return None, n_pos
    return float(roc_auc_score(y_true, y_score)), n_pos

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load data
    log("Loading brain dataset...")
    adata = sc.read_h5ad(str(DATA_PATH))
    log(f"Raw dataset: {adata.n_obs} cells x {adata.n_vars} genes")

    # Sample ~500 cells (same procedure as held_out_validation)
    N_CELLS = 500
    if "cell_type" in adata.obs.columns:
        ct_counts = adata.obs["cell_type"].value_counts()
        valid_cts = ct_counts[ct_counts >= 50].index.tolist()
        adata = adata[adata.obs["cell_type"].isin(valid_cts)].copy()
        sampled = []
        for ct in valid_cts:
            idx = np.where(adata.obs["cell_type"] == ct)[0]
            n_ct = max(1, int(N_CELLS * len(idx) / adata.n_obs))
            sampled.extend(np.random.choice(idx, min(n_ct, len(idx)), replace=False))
        if len(sampled) > N_CELLS:
            sampled = list(np.random.choice(sampled, N_CELLS, replace=False))
        adata = adata[sampled].copy()
    else:
        if adata.n_obs > N_CELLS:
            idx = np.random.choice(adata.n_obs, N_CELLS, replace=False)
            adata = adata[idx].copy()

    log(f"Sampled dataset: {adata.n_obs} cells")

    # 2. Stratified 50/50 split
    cell_indices = np.arange(adata.n_obs)
    if "cell_type" in adata.obs.columns:
        strat = list(adata.obs["cell_type"].values)
        idx_a, idx_b = train_test_split(cell_indices, test_size=0.5,
                                         stratify=strat, random_state=42)
    else:
        idx_a, idx_b = train_test_split(cell_indices, test_size=0.5, random_state=42)

    adata_a = adata[idx_a].copy()
    adata_b = adata[idx_b].copy()
    log(f"Half A: {adata_a.n_obs} cells | Half B: {adata_b.n_obs} cells")

    # 3. Load model
    log("Loading scGPT brain model...")
    wrapped_model, vocab_obj = load_model(device)

    # 4. Load TRRUST ground truth (symbol-level)
    log("Loading TRRUST...")
    trrust_edges, _ = load_trrust()

    # 5. Extract per-layer attention on both halves
    log("\n=== Extracting attention on Half A ===")
    ds_a = build_dataset(adata_a, vocab_obj, max_genes=1500, max_seq=512)
    layer_mats_a, genes_a = extract_per_layer(wrapped_model, ds_a, device, batch_size=2)
    n_layers = len(layer_mats_a)
    log(f"Extracted {n_layers} layers, {len(genes_a)} genes")

    gc.collect()
    torch.cuda.empty_cache()

    log("\n=== Extracting attention on Half B ===")
    ds_b = build_dataset(adata_b, vocab_obj, max_genes=1500, max_seq=512)
    # Free Half A GPU memory before extracting B
    import gc as _gc; _gc.collect(); torch.cuda.empty_cache()
    layer_mats_b, genes_b = extract_per_layer(wrapped_model, ds_b, device, batch_size=1)
    log(f"Extracted {n_layers} layers, {len(genes_b)} genes")

    gc.collect()
    torch.cuda.empty_cache()

    # 6. Evaluate every layer on both halves
    log("\n=== Evaluating AUROC per layer ===")

    results_a = {}
    for l in range(n_layers):
        auroc, n_pos = evaluate_auroc(layer_mats_a[l], genes_a, trrust_edges)
        results_a[l] = {"auroc": auroc, "n_pos": n_pos}
        log(f"  Half A  Layer {l:2d}  AUROC={auroc if auroc else 'N/A':>8}  pos_edges={n_pos}")

    results_b = {}
    for l in range(n_layers):
        auroc, n_pos = evaluate_auroc(layer_mats_b[l], genes_b, trrust_edges)
        results_b[l] = {"auroc": auroc, "n_pos": n_pos}
        log(f"  Half B  Layer {l:2d}  AUROC={auroc if auroc else 'N/A':>8}  pos_edges={n_pos}")

    # 7. Cross-validation: best on A -> eval on B, and vice versa
    def top_k_layers(res, k=3):
        valid = [(l, r["auroc"]) for l, r in res.items() if r["auroc"] is not None]
        valid.sort(key=lambda x: x[1], reverse=True)
        return [l for l, _ in valid[:k]]

    best_a = top_k_layers(results_a, 3)
    best_b = top_k_layers(results_b, 3)

    log(f"\nBest layers on Half A: {best_a}")
    log(f"Best layers on Half B: {best_b}")
    log(f"Overlap: {len(set(best_a) & set(best_b))}/3")

    # Evaluate best-A layers on half B
    log("\n--- Cross-validation: layers selected on A, evaluated on B ---")
    cross_ab = {}
    for l in best_a:
        auroc_b = results_b[l]["auroc"]
        auroc_a = results_a[l]["auroc"]
        log(f"  Layer {l}: A={auroc_a:.4f}  B={auroc_b if auroc_b else 'N/A'}")
        cross_ab[l] = {"train_auroc": auroc_a, "test_auroc": auroc_b}

    log("\n--- Cross-validation: layers selected on B, evaluated on A ---")
    cross_ba = {}
    for l in best_b:
        auroc_a = results_a[l]["auroc"]
        auroc_b = results_b[l]["auroc"]
        log(f"  Layer {l}: B={auroc_b:.4f}  A={auroc_a if auroc_a else 'N/A'}")
        cross_ba[l] = {"train_auroc": auroc_b, "test_auroc": auroc_a}

    # Random baseline on half B
    import random
    random.seed(42)
    all_layers = list(range(n_layers))
    random_trials = []
    for _ in range(50):
        rand_layers = random.sample(all_layers, 3)
        mean_auroc = np.mean([results_b[l]["auroc"] for l in rand_layers
                              if results_b[l]["auroc"] is not None])
        random_trials.append(mean_auroc)
    rand_mean = np.mean(random_trials)
    rand_std = np.std(random_trials)

    best_a_mean = np.mean([results_b[l]["auroc"] for l in best_a
                           if results_b[l]["auroc"] is not None])
    improvement = best_a_mean - rand_mean
    z_score = improvement / rand_std if rand_std > 0 else 0.0

    log(f"\nBest-A layers mean AUROC on B: {best_a_mean:.4f}")
    log(f"Random-3 baseline on B: {rand_mean:.4f} +/- {rand_std:.4f}")
    log(f"Improvement: {improvement:.4f}  ({z_score:.1f} sigma)")

    elapsed = time.time() - t0

    # 8. Save JSON results
    full_results = {
        "experiment": "cssi_cross_dataset_validation_scgpt",
        "model": "scGPT brain checkpoint",
        "n_cells_total": int(adata.n_obs),
        "n_cells_a": int(adata_a.n_obs),
        "n_cells_b": int(adata_b.n_obs),
        "n_genes_a": len(genes_a),
        "n_genes_b": len(genes_b),
        "n_layers": n_layers,
        "trrust_edges": len(trrust_edges),
        "layer_auroc_half_a": {str(l): r["auroc"] for l, r in results_a.items()},
        "layer_auroc_half_b": {str(l): r["auroc"] for l, r in results_b.items()},
        "best_layers_a": best_a,
        "best_layers_b": best_b,
        "layer_overlap": len(set(best_a) & set(best_b)),
        "cross_ab": {str(l): v for l, v in cross_ab.items()},
        "cross_ba": {str(l): v for l, v in cross_ba.items()},
        "best_a_mean_on_b": best_a_mean,
        "random_baseline_mean": rand_mean,
        "random_baseline_std": rand_std,
        "improvement": improvement,
        "z_score": z_score,
        "elapsed_seconds": elapsed,
    }

    json_path = OUT_DIR / "cssi_cross_dataset_results.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    log(f"\nJSON saved to {json_path}")

    # 9. Write markdown report
    write_report(full_results)
    log(f"Report saved to {REPORT_PATH}")
    log(f"\nDone in {elapsed:.0f}s")


def write_report(r):
    lines = []
    lines.append("# CSSI Cross-Dataset Validation Report (scGPT Brain)")
    lines.append("")
    lines.append("## Experiment")
    lines.append(f"- **Model:** scGPT brain checkpoint (12 layers, 8 heads, d=512)")
    lines.append(f"- **Dataset:** DLPFC 11k brain scRNA-seq")
    lines.append(f"- **Cells:** {r['n_cells_total']} total -> Half A ({r['n_cells_a']}), Half B ({r['n_cells_b']})")
    lines.append(f"- **Genes (Half A):** {r['n_genes_a']}")
    lines.append(f"- **Genes (Half B):** {r['n_genes_b']}")
    lines.append(f"- **Ground truth:** TRRUST ({r['trrust_edges']} directed edges)")
    lines.append(f"- **Metric:** AUROC (attention score vs TRRUST edge label)")
    lines.append("")

    lines.append("## Per-Layer AUROC")
    lines.append("")
    lines.append("| Layer | Half A | Half B |")
    lines.append("|-------|--------|--------|")
    for l in range(r["n_layers"]):
        a = r["layer_auroc_half_a"].get(str(l))
        b = r["layer_auroc_half_b"].get(str(l))
        a_str = f"{a:.4f}" if a is not None else "N/A"
        b_str = f"{b:.4f}" if b is not None else "N/A"
        lines.append(f"| {l:2d} | {a_str} | {b_str} |")
    lines.append("")

    lines.append("## Cross-Validation")
    lines.append("")
    lines.append(f"- **Best layers on Half A:** {r['best_layers_a']}")
    lines.append(f"- **Best layers on Half B:** {r['best_layers_b']}")
    lines.append(f"- **Layer overlap:** {r['layer_overlap']}/3")
    lines.append("")

    lines.append("### Layers selected on A, evaluated on B")
    lines.append("| Layer | AUROC (A, train) | AUROC (B, test) |")
    lines.append("|-------|------------------|-----------------|")
    for l_str, v in r["cross_ab"].items():
        tr = v["train_auroc"]
        te = v["test_auroc"]
        lines.append(f"| {l_str} | {tr:.4f} | {te if te else 'N/A'} |")
    lines.append("")

    lines.append("### Layers selected on B, evaluated on A")
    lines.append("| Layer | AUROC (B, train) | AUROC (A, test) |")
    lines.append("|-------|------------------|-----------------|")
    for l_str, v in r["cross_ba"].items():
        tr = v["train_auroc"]
        te = v["test_auroc"]
        lines.append(f"| {l_str} | {tr:.4f} | {te if te else 'N/A'} |")
    lines.append("")

    lines.append("### Comparison to Random Baseline")
    lines.append(f"- Mean AUROC of best-A layers on B: **{r['best_a_mean_on_b']:.4f}**")
    lines.append(f"- Random 3-layer baseline on B: **{r['random_baseline_mean']:.4f}** +/- {r['random_baseline_std']:.4f}")
    lines.append(f"- Improvement: **{r['improvement']:.4f}** ({r['z_score']:.1f} sigma)")
    lines.append("")

    passed = r["improvement"] > 0 and r["layer_overlap"] >= 1
    if passed:
        lines.append("## Conclusion")
        lines.append(f"Cross-dataset validation **PASSED**. Layers identified on one half of the data")
        lines.append(f"generalize to the held-out half, with {r['layer_overlap']}/3 layer overlap")
        lines.append(f"and {r['improvement']:.4f} AUROC improvement over random layer selection")
        lines.append(f"({r['z_score']:.1f} sigma).")
    else:
        lines.append("## Conclusion")
        lines.append("Cross-dataset validation **did not show clear generalization**.")
        lines.append(f"Layer overlap: {r['layer_overlap']}/3, improvement: {r['improvement']:.4f}.")

    lines.append("")
    lines.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')} | Runtime: {r['elapsed_seconds']:.0f}s*")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        traceback.print_exc(file=log_file)
        sys.exit(1)
