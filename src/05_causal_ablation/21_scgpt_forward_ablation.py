#!/usr/bin/env python3
"""
scGPT forward-pass head ablation — cross-architecture replication upgrade.

This script is the forward-pass counterpart to `20_scgpt_ablation.py` which
performed edge-level (cached-attention) ablation. The forward-pass version
addresses the caveat that edge-level ablation does not propagate through
the residual stream: by running actual forward passes on scGPT with heads
zeroed, we let the ablation affect subsequent layers the same way the
Geneformer head_mask API does.

Ablation mechanism
------------------
scGPT uses `nn.MultiheadAttention`, which internally calls
`F.multi_head_attention_forward` (and therefore bypasses hooks on
`out_proj`). Rather than monkey-patching the attention implementation,
we zero the column slices of each layer's `out_proj.weight` matrix
corresponding to ablated heads:

    out_proj.weight[:, h*head_dim:(h+1)*head_dim] = 0

Mathematically, this is equivalent to zeroing head h's per-token output
immediately before the output projection: the contribution of head h to
the layer's residual update becomes exactly zero, propagating through
LayerNorm, FFN, and all subsequent layers. The per-head attention
weights themselves are unchanged (we still need them for extracting the
pooled edge score over non-ablated heads), but their downstream effect
on the residual stream is eliminated.

Attention extraction uses the same `_sa_block` patch as the sister repo's
wrapper, which enables `need_weights=True, average_attn_weights=False`
and captures the per-head weight tensor at each layer.

Validation
----------
- Zero-mask (no heads ablated): pooled AUROC on 30 immune cells should
  match the cached-attention baseline (~0.706) within ~0.01.
- All-mask (every head at every layer ablated): pooled output over 0
  heads is the zero matrix, AUROC is undefined or 0.5; we report as a
  sanity check that the ablation mechanism is actually doing something.

Conditions (simplified battery; full 13-condition edge-level battery is
in 20_scgpt_ablation.py)
------------------------------------------------------------------------
  baseline     no ablation
  reg_top_5    exclude top-5 TRRUST-ranked heads
  reg_top_20   exclude top-20
  reg_top_50   exclude top-50 (52 percent of all 96 heads)
  random_20    exclude 20 random heads (3 seeds)

The 5-condition set mirrors the main dose-response spine of the edge-
level battery and is what's needed to test whether forward-pass
propagation changes the conclusion. We deliberately scope down from 13
conditions because each condition now requires a full forward pass over
30 cells rather than a cached-attention re-pooling operation.

Output
------
results/scgpt_ablation/forward_pass_immune.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 1)

REPO_ROOT = Path(__file__).resolve().parents[2]
SISTER_REPO = Path("/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp")
NOVAE_REPO = Path("/Volumes/Crucial X6/MacBook/biomechinterp/novae")
SCGPT_REPO = SISTER_REPO / "external" / "scGPT"
CKPT_DIR = SISTER_REPO / "external" / "scGPT_checkpoints" / "whole-human"
CKPT_PATH = CKPT_DIR / "best_model.pt"
VOCAB_PATH = CKPT_DIR / "vocab.json"
ARGS_PATH = CKPT_DIR / "args.json"

# Canonical 1200-gene HVG vocabulary (order matches cached scGPT attention)
HVG_H5AD_IMMUNE = (
    SISTER_REPO
    / "outputs"
    / "head_layer_checkpoint_replication"
    / "immune_shared_hvg1200_processed.h5ad"
)

# Raw tissue h5ads for kidney/brain (no preprocessed shared-HVG copies exist
# locally for these tissues, so we project them onto the immune HVG order
# at load time).
KIDNEY_RAW_H5AD = SISTER_REPO / "data" / "raw" / "tabula_sapiens_kidney.h5ad"
BRAIN_RAW_H5AD = NOVAE_REPO / "datasets" / "mics-lab-novae" / "human" / "brain" / "DLPFC_151675.h5ad"

TRRUST_PATH = (
    REPO_ROOT / "src" / "02_cssi_method" / "cssi_real_data" / "results" / "trrust_human.tsv"
)
PER_HEAD_RANK_DIR = REPO_ROOT / "results" / "scgpt_ablation"
OUT_DIR = REPO_ROOT / "results" / "scgpt_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Constants ----
N_CELLS = int(os.environ.get("NMI_N_CELLS", "30"))  # sample size; overridable for smoke test
TISSUE = os.environ.get("NMI_TISSUE", "immune")     # one of {immune, kidney, brain}
MAX_GENES = 1200      # full HVG sequence
RNG_SEED = 2026
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# =====================================================================
# Model loading
# =====================================================================

def _ensure_scgpt_on_path() -> None:
    repo_str = str(SCGPT_REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def load_scgpt_model() -> Tuple[nn.Module, Dict[str, int]]:
    _ensure_scgpt_on_path()
    from scgpt.model import TransformerModel  # type: ignore
    from scgpt.tokenizer import GeneVocab  # type: ignore
    from scgpt.utils import load_pretrained  # type: ignore

    vocab = GeneVocab.from_file(VOCAB_PATH)
    with open(ARGS_PATH) as fh:
        sa = json.load(fh)

    model_args = dict(
        ntoken=len(vocab),
        d_model=sa["embsize"],
        nhead=sa["nheads"],
        d_hid=sa["d_hid"],
        nlayers=sa["nlayers"],
        nlayers_cls=sa.get("n_layers_cls", 3),
        n_cls=1,
        vocab=vocab,
        dropout=sa.get("dropout", 0.5),
        pad_token="<pad>",
        pad_value=sa.get("pad_value", 0),
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=0,
        domain_spec_batchnorm=False,
        n_input_bins=sa.get("n_bins", 51),
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=sa.get("pre_norm", False),
    )
    model = TransformerModel(**model_args).eval()
    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    load_pretrained(model, state, verbose=False)
    model = model.to(DEVICE)
    return model, vocab.get_stoi()


def get_transformer_layers(model: nn.Module) -> List[nn.Module]:
    for _, mod in model.named_modules():
        if isinstance(mod, nn.TransformerEncoder):
            # Disable the nested-tensor fastpath because it (i) is incompatible
            # with our sa_block patch that requires need_weights=True and
            # (ii) calls aten::_nested_tensor_from_mask_left_aligned which is
            # not implemented on MPS. The attribute is `use_nested_tensor`
            # (not `enable_nested_tensor`) on PyTorch 2.1's TransformerEncoder.
            mod.use_nested_tensor = False
            return list(mod.layers)
    raise RuntimeError("scGPT TransformerEncoder not found")


# =====================================================================
# sa_block patch: capture per-head attention weights
# =====================================================================

def install_sa_block_capture(layers: List[nn.Module]) -> None:
    """Patch _sa_block on each TransformerEncoderLayer so we can read
    per-head attention weights after forward pass. Based on the sister
    repo's `_enable_torch_attention_capture` in wrapper.py.
    """
    import types
    for layer in layers:
        if hasattr(layer, "_scgpt_original_sa_block"):
            continue
        layer._scgpt_original_sa_block = layer._sa_block

        def _sa_block_with_attn(self, x, attn_mask, key_padding_mask, is_causal=False):
            attn_output, attn_weights = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
                is_causal=is_causal,
            )
            self._last_attn_weights = attn_weights  # shape [B, H, L, L]
            return self.dropout1(attn_output)

        layer._sa_block = types.MethodType(_sa_block_with_attn, layer)


def clear_sa_block_capture(layers: List[nn.Module]) -> None:
    for layer in layers:
        if hasattr(layer, "_last_attn_weights"):
            delattr(layer, "_last_attn_weights")


# =====================================================================
# Head ablation via out_proj column zeroing
# =====================================================================

def save_out_proj_weights(layers: List[nn.Module]) -> Dict[int, torch.Tensor]:
    """Snapshot original out_proj weights for each layer so we can restore
    after each ablation condition."""
    snap: Dict[int, torch.Tensor] = {}
    for i, layer in enumerate(layers):
        snap[i] = layer.self_attn.out_proj.weight.data.clone()
    return snap


def restore_out_proj_weights(layers: List[nn.Module], snap: Dict[int, torch.Tensor]) -> None:
    for i, layer in enumerate(layers):
        layer.self_attn.out_proj.weight.data.copy_(snap[i])


def apply_head_mask_to_out_proj(
    layers: List[nn.Module],
    ablated_heads: List[Tuple[int, int]],
) -> None:
    """Zero the column slices of each layer's out_proj.weight that
    correspond to ablated heads.

    nn.MultiheadAttention.out_proj maps concat[h_0, h_1, ..., h_{H-1}]
    -> embed_dim, where each h_i is of dimension head_dim. The columns
    [h*head_dim : (h+1)*head_dim] of out_proj.weight correspond to head
    h's contribution. Zeroing those columns makes head h's output
    invisible to all downstream computation (residual, FFN, subsequent
    layers), which is the operational definition of head ablation.
    """
    by_layer: Dict[int, List[int]] = {}
    for (L, H) in ablated_heads:
        by_layer.setdefault(L, []).append(H)
    for L, heads in by_layer.items():
        attn = layers[L].self_attn
        D = attn.head_dim
        w = attn.out_proj.weight.data
        for H in heads:
            w[:, H * D : (H + 1) * D] = 0.0


# =====================================================================
# Tokenization: prepare input batches from h5ad cells
# =====================================================================

def _read_canonical_genes() -> List[str]:
    """Return the 1200 HVG gene symbols in the canonical order used by
    the cached scGPT per-head attention extraction."""
    with h5py.File(HVG_H5AD_IMMUNE, "r") as f:
        idx = f["var/_index"][:]
        return [g.decode() if isinstance(g, bytes) else g for g in idx]


def _normalise_log1p(counts: np.ndarray) -> np.ndarray:
    """Per-cell library-size normalisation to 10{,}000 then log1p, the
    standard scanpy pipeline used to produce the immune preprocessed
    file. Applied row-wise on a (cells, genes) dense matrix."""
    libsize = counts.sum(axis=1, keepdims=True)
    libsize = np.where(libsize == 0, 1.0, libsize)
    norm = counts * (1e4 / libsize)
    return np.log1p(norm).astype(np.float32)


def _load_immune_cells(n_cells: int, rng: np.random.Generator) -> Tuple[List[str], np.ndarray]:
    with h5py.File(HVG_H5AD_IMMUNE, "r") as f:
        idx = f["var/_index"][:]
        genes = [g.decode() if isinstance(g, bytes) else g for g in idx]
        n_obs = int(f["obs/_index"].shape[0])
        indptr = f["X/indptr"][:]
        indices = f["X/indices"][:]
        data = f["X/data"][:]

    row_idx = rng.choice(n_obs, size=min(n_cells, n_obs), replace=False)
    row_idx.sort()

    n_genes = len(genes)
    out = np.zeros((len(row_idx), n_genes), dtype=np.float32)
    for i, r in enumerate(row_idx):
        s, e = indptr[r], indptr[r + 1]
        out[i, indices[s:e]] = data[s:e]
    # Already normalised + log1p
    return genes, out


def _load_kidney_cells(n_cells: int, rng: np.random.Generator) -> Tuple[List[str], np.ndarray]:
    """Load Tabula Sapiens kidney, project onto the canonical 1200 HVG
    gene order, normalise + log1p."""
    canonical_genes = _read_canonical_genes()
    canon_to_pos = {g: i for i, g in enumerate(canonical_genes)}

    with h5py.File(KIDNEY_RAW_H5AD, "r") as f:
        # Gene symbols are in var/feature_name (Categorical-encoded by scanpy)
        fn_grp = f["var/feature_name"]
        if "categories" in fn_grp and "codes" in fn_grp:
            cats = fn_grp["categories"][:]
            codes = fn_grp["codes"][:]
            symbols = [
                cats[c].decode() if isinstance(cats[c], bytes) else cats[c]
                for c in codes
            ]
        else:
            raw = fn_grp[:]
            symbols = [s.decode() if isinstance(s, bytes) else s for s in raw]

        n_obs = int(f["obs/_index"].shape[0])
        # CSR sparse X
        indptr = f["X/indptr"][:]
        indices = f["X/indices"][:]
        data = f["X/data"][:]

    # Build a column projection: full_gene_idx -> canonical_pos (-1 if absent)
    n_full = len(symbols)
    full_to_canon = np.full(n_full, -1, dtype=np.int64)
    for full_i, sym in enumerate(symbols):
        canon_pos = canon_to_pos.get(sym)
        if canon_pos is not None:
            full_to_canon[full_i] = canon_pos

    # Sample N cells
    row_idx = rng.choice(n_obs, size=min(n_cells, n_obs), replace=False)
    row_idx.sort()

    counts = np.zeros((len(row_idx), len(canonical_genes)), dtype=np.float32)
    for i, r in enumerate(row_idx):
        s, e = indptr[r], indptr[r + 1]
        cols = indices[s:e]
        vals = data[s:e]
        canon_cols = full_to_canon[cols]
        keep = canon_cols >= 0
        counts[i, canon_cols[keep]] += vals[keep]

    expr = _normalise_log1p(counts)
    return canonical_genes, expr


def _load_brain_cells(n_cells: int, rng: np.random.Generator) -> Tuple[List[str], np.ndarray]:
    """Load DLPFC brain (gene symbols already in var/_index)."""
    canonical_genes = _read_canonical_genes()
    canon_to_pos = {g: i for i, g in enumerate(canonical_genes)}

    with h5py.File(BRAIN_RAW_H5AD, "r") as f:
        idx = f["var/_index"][:]
        symbols = [g.decode() if isinstance(g, bytes) else g for g in idx]
        n_obs = int(f["obs/_index"].shape[0])
        indptr = f["X/indptr"][:]
        indices = f["X/indices"][:]
        data = f["X/data"][:]

    n_full = len(symbols)
    full_to_canon = np.full(n_full, -1, dtype=np.int64)
    for full_i, sym in enumerate(symbols):
        canon_pos = canon_to_pos.get(sym)
        if canon_pos is not None:
            full_to_canon[full_i] = canon_pos

    row_idx = rng.choice(n_obs, size=min(n_cells, n_obs), replace=False)
    row_idx.sort()

    counts = np.zeros((len(row_idx), len(canonical_genes)), dtype=np.float32)
    for i, r in enumerate(row_idx):
        s, e = indptr[r], indptr[r + 1]
        cols = indices[s:e]
        vals = data[s:e]
        canon_cols = full_to_canon[cols]
        keep = canon_cols >= 0
        counts[i, canon_cols[keep]] += vals[keep]

    expr = _normalise_log1p(counts)
    return canonical_genes, expr


_TISSUE_LOADERS = {
    "immune": _load_immune_cells,
    "kidney": _load_kidney_cells,
    "brain": _load_brain_cells,
}


def load_tissue_cells(
    tissue: str, n_cells: int, rng: np.random.Generator
) -> Tuple[List[str], np.ndarray]:
    """Dispatch to the appropriate per-tissue loader."""
    if tissue not in _TISSUE_LOADERS:
        raise ValueError(f"unknown tissue {tissue!r}, expected one of {list(_TISSUE_LOADERS)}")
    return _TISSUE_LOADERS[tissue](n_cells, rng)


def build_scgpt_input(
    expr_row: np.ndarray,
    gene_names: List[str],
    vocab_stoi: Dict[str, int],
    cls_token_id: int,
    pad_token_id: int,
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Prepare a single-cell scGPT input with all 1200 HVG genes in a fixed
    order so each token position deterministically maps to a gene index.

    scGPT was trained on dropped-zero, sorted-by-expression sequences, but
    for cross-cell attention pooling we need every cell to contribute to
    every (gene, gene) entry of the shared grid. Including zero-expression
    genes is a mild distributional mismatch that the model handles
    gracefully (attention rows still sum to 1) and gives us dense coverage
    in 30 cells instead of 20{,}000.

    Returns:
        gene_ids:   [1, 1+n_present]
        values:     [1, 1+n_present]
        kpm:        [1, 1+n_present] padding mask (all False here)
        token_to_gene: [1+n_present] gene index per sequence position,
                       or -1 for the CLS prefix.
    """
    n_genes = len(gene_names)
    # Determine which genes are in vocab
    vocab_present_idx: List[int] = []
    vocab_present_tok: List[int] = []
    for g_idx, g in enumerate(gene_names):
        tok = vocab_stoi.get(g)
        if tok is not None:
            vocab_present_idx.append(g_idx)
            vocab_present_tok.append(tok)

    # CLS at position 0, then every in-vocab gene in fixed order
    gene_ids = [cls_token_id] + vocab_present_tok
    values = [0.0] + [float(expr_row[i]) for i in vocab_present_idx]

    L = len(gene_ids)
    if L > max_len:
        gene_ids = gene_ids[:max_len]
        values = values[:max_len]
        vocab_present_idx = vocab_present_idx[: max_len - 1]
        L = max_len

    gene_ids_t = torch.tensor(gene_ids, dtype=torch.long).unsqueeze(0)
    values_t = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
    key_padding_mask = torch.zeros((1, L), dtype=torch.bool)

    token_to_gene = np.full(L, -1, dtype=np.int64)
    token_to_gene[1:] = np.array(vocab_present_idx, dtype=np.int64)
    return gene_ids_t, values_t, key_padding_mask, token_to_gene


# =====================================================================
# Forward pass + attention accumulation
# =====================================================================

@torch.no_grad()
def forward_and_capture_attention(
    model: nn.Module,
    layers: List[nn.Module],
    gene_ids: torch.Tensor,
    values: torch.Tensor,
    key_padding_mask: torch.Tensor,
    token_to_gene: np.ndarray,
    n_genes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a forward pass and collect per-(layer, head) attention weights
    projected into the shared (n_genes, n_genes) grid.

    Returns:
        attn_sum: (n_layers, n_heads, n_genes, n_genes) float32
        attn_cnt: (n_layers, n_heads, n_genes, n_genes) int32
    """
    clear_sa_block_capture(layers)
    _ = model(
        gene_ids.to(DEVICE),
        values.to(DEVICE),
        src_key_padding_mask=key_padding_mask.to(DEVICE),
    )
    n_layers = len(layers)
    n_heads = layers[0].self_attn.num_heads

    attn_sum = np.zeros((n_layers, n_heads, n_genes, n_genes), dtype=np.float32)
    attn_cnt = np.zeros((n_layers, n_heads, n_genes, n_genes), dtype=np.int32)

    # token_to_gene maps each sequence position to a gene index in [0..n_genes),
    # or -1 for special/padding positions to skip.
    valid_pos = np.where(token_to_gene >= 0)[0]
    valid_gene = token_to_gene[valid_pos]

    for L, layer in enumerate(layers):
        w = layer._last_attn_weights  # [1, H, seqlen, seqlen]
        if w is None:
            continue
        w_np = w[0].detach().to("cpu").numpy().astype(np.float32)  # [H, seqlen, seqlen]
        # Slice to valid gene positions on both axes
        w_valid = w_np[:, valid_pos[:, None], valid_pos[None, :]]  # [H, V, V]
        # Scatter into (n_genes, n_genes)
        # Multiple sequence positions can map to the same gene if duplicates;
        # in our setup each gene appears at most once so this is 1-to-1.
        for H in range(n_heads):
            scatter = np.zeros((n_genes, n_genes), dtype=np.float32)
            scatter[np.ix_(valid_gene, valid_gene)] = w_valid[H]
            attn_sum[L, H] += scatter
            cnt = np.zeros((n_genes, n_genes), dtype=np.int32)
            cnt[np.ix_(valid_gene, valid_gene)] = 1
            attn_cnt[L, H] += cnt
    return attn_sum, attn_cnt


# =====================================================================
# TRRUST evaluation (matches 20_scgpt_ablation.py)
# =====================================================================

def load_trrust(genes: List[str], direction_known: bool = False) -> np.ndarray:
    df = pd.read_csv(
        TRRUST_PATH, sep="\t", header=None, names=["tf", "target", "mode", "pmid"]
    )
    if direction_known:
        df = df[df["mode"].isin(["Activation", "Repression"])]
    gene_idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    edges = np.zeros((n, n), dtype=bool)
    for _, row in df.iterrows():
        t, g = row["tf"], row["target"]
        if t in gene_idx and g in gene_idx:
            edges[gene_idx[t], gene_idx[g]] = True
    return edges


def pooled_auroc(
    attn: np.ndarray,
    cnt: np.ndarray,
    trrust: np.ndarray,
    ablated_heads: List[Tuple[int, int]],
) -> Tuple[float, int, int]:
    """Compute pooled TRRUST AUROC across all non-ablated (layer, head)
    pairs. attn is already the sum across cells; we average only over
    non-ablated heads.
    """
    n_layers, n_heads, n_genes, _ = attn.shape
    mask = np.ones((n_layers, n_heads), dtype=bool)
    for (L, H) in ablated_heads:
        mask[L, H] = False
    if mask.sum() == 0:
        return float("nan"), 0, 0

    # Average attention across included heads, normalising by per-cell count
    denom = cnt[mask].sum(axis=0).astype(np.float32)  # [n_genes, n_genes]
    denom = np.where(denom == 0, 1.0, denom)
    num = attn[mask].sum(axis=0)
    pooled = num / denom

    # Flatten and drop diagonal
    diag = np.eye(n_genes, dtype=bool)
    s = pooled[~diag]
    y = trrust[~diag]
    if y.sum() < 5 or (~y).sum() < 5:
        return float("nan"), int(y.sum()), int((~y).sum())
    return float(roc_auc_score(y.astype(int), s)), int(y.sum()), int((~y).sum())


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print("[1/5] Loading scGPT whole-human checkpoint...")
    t0 = time.time()
    model, vocab_stoi = load_scgpt_model()
    layers = get_transformer_layers(model)
    n_layers = len(layers)
    n_heads = layers[0].self_attn.num_heads
    print(f"  layers={n_layers} heads={n_heads} device={DEVICE} in {time.time()-t0:.1f}s")

    install_sa_block_capture(layers)
    snap = save_out_proj_weights(layers)

    print(f"[2/5] Loading {N_CELLS} cells from {TISSUE} tissue...")
    genes, expr = load_tissue_cells(TISSUE, N_CELLS, rng)
    print(f"  n_genes={len(genes)} n_cells={len(expr)}")
    print(f"  expression range: min={expr.min():.3f} max={expr.max():.3f} mean={expr.mean():.4f}")

    # Build vocab mapping for genes (used by scGPT input prep)
    print("[3/5] Loading TRRUST reference and per-head ranking...")
    trrust = load_trrust(genes, direction_known=False)
    print(f"  TRRUST positives in HVG: {int(trrust.sum())}")

    rank_path = PER_HEAD_RANK_DIR / f"per_head_trrust_{TISSUE}.json"
    with open(rank_path) as fh:
        per_head_rank = np.array(json.load(fh))  # (n_layers, n_heads)
    flat_auroc = per_head_rank.flatten()
    order_desc = np.argsort(-flat_auroc)  # best first
    top5 = [(int(i // n_heads), int(i % n_heads)) for i in order_desc[:5]]
    top20 = [(int(i // n_heads), int(i % n_heads)) for i in order_desc[:20]]
    top50 = [(int(i // n_heads), int(i % n_heads)) for i in order_desc[:50]]

    # Random-20 draws (3 seeds)
    total_heads = n_layers * n_heads
    random20_sets: List[List[Tuple[int, int]]] = []
    for seed in (11, 22, 33):
        r = np.random.default_rng(seed)
        idxs = r.choice(total_heads, size=20, replace=False)
        random20_sets.append(
            [(int(i // n_heads), int(i % n_heads)) for i in idxs]
        )

    conditions: List[Tuple[str, List[Tuple[int, int]]]] = [
        ("baseline", []),
        ("reg_top_5", top5),
        ("reg_top_20", top20),
        ("reg_top_50", top50),
        ("random_20_rep0", random20_sets[0]),
        ("random_20_rep1", random20_sets[1]),
        ("random_20_rep2", random20_sets[2]),
    ]

    print(f"[4/5] Running {len(conditions)} ablation conditions x {N_CELLS} cells each...")
    results: Dict[str, Dict] = {}
    cls_token = vocab_stoi.get("<cls>", vocab_stoi.get("[CLS]"))
    pad_token = vocab_stoi.get("<pad>", 0)
    if cls_token is None:
        cls_token = pad_token  # fall back
    # max_len includes the CLS prefix
    max_len = len(genes) + 1

    # Pre-build each cell's input tensor and its token->gene mapping.
    # These are identical across ablation conditions; we reuse them.
    cell_inputs = []
    for c in range(len(expr)):
        gene_ids_t, values_t, kpm, token_to_gene = build_scgpt_input(
            expr[c], genes, vocab_stoi, cls_token, pad_token, max_len
        )
        cell_inputs.append((gene_ids_t, values_t, kpm, token_to_gene))

    n_genes = len(genes)

    for cond_name, ablated in conditions:
        print(f"  -- {cond_name}  (|ablated|={len(ablated)})")
        t0 = time.time()
        restore_out_proj_weights(layers, snap)
        apply_head_mask_to_out_proj(layers, ablated)

        total_sum = np.zeros((n_layers, n_heads, n_genes, n_genes), dtype=np.float32)
        total_cnt = np.zeros((n_layers, n_heads, n_genes, n_genes), dtype=np.int32)

        for gene_ids_t, values_t, kpm, token_to_gene in cell_inputs:
            s, c = forward_and_capture_attention(
                model, layers, gene_ids_t, values_t, kpm, token_to_gene, n_genes
            )
            total_sum += s
            total_cnt += c

        auroc, n_pos, n_neg = pooled_auroc(total_sum, total_cnt, trrust, [])
        # For non-baseline conditions, also re-pool over only non-ablated heads:
        if ablated:
            auroc_abl, _, _ = pooled_auroc(total_sum, total_cnt, trrust, ablated)
        else:
            auroc_abl = auroc

        dt = time.time() - t0
        print(f"     pooled-all AUROC={auroc:.4f}  pooled-non-ablated AUROC={auroc_abl:.4f}  ({dt:.1f}s)")
        results[cond_name] = {
            "auroc_all_heads": float(auroc),
            "auroc_non_ablated": float(auroc_abl),
            "n_ablated": len(ablated),
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "wall_seconds": dt,
        }

    restore_out_proj_weights(layers, snap)

    print("[5/5] Summarising vs edge-level baseline...")
    baseline_fwd = results["baseline"]["auroc_non_ablated"]
    with open(OUT_DIR / "ablation_results.json") as fh:
        edge = json.load(fh)["tissues"][TISSUE]
    baseline_edge = edge["baseline_auroc"]
    print(f"  forward-pass baseline ({TISSUE}): {baseline_fwd:.4f}")
    print(f"  edge-level   baseline ({TISSUE}): {baseline_edge:.4f}")
    print(f"  absolute diff:                     {abs(baseline_fwd - baseline_edge):.4f}")

    payload = {
        "tissue": TISSUE,
        "n_cells": N_CELLS,
        "n_genes": n_genes,
        "n_trrust_positives_pairs": int(trrust.sum()),
        "device": DEVICE,
        "forward_pass_baseline_auroc": baseline_fwd,
        "edge_level_baseline_auroc": baseline_edge,
        "baseline_absolute_difference": float(abs(baseline_fwd - baseline_edge)),
        "conditions": results,
        "top_head_set_top5": top5,
        "top_head_set_top20": top20,
        "top_head_set_top50": top50,
        "random20_sets": random20_sets,
    }
    out_path = OUT_DIR / f"forward_pass_{TISSUE}.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
