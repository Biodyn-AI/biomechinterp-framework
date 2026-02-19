#!/usr/bin/env python3
"""
Round 9 Unified K562 Attention Re-Extraction (v2 - vectorized)
==============================================================
Extracts Geneformer V2-316M attention from 2000 K562 control cells across
ALL 18 layers with per-head resolution. Saves raw edge matrices for
downstream experiments (incremental value, confound decomposition,
full-layer analysis, head ablation).

v2 changes:
- Numpy matrix accumulators instead of dict-based O(n^2) Python loops
- Checkpoint saving after each pass for resume capability
- Vectorized HVG position extraction and accumulation
"""

import os
import sys
import gc
import json
import time
import pickle
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import pandas as pd
import h5py
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
DATA_PATH = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data", "replogle_concat.h5ad")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round9_unified_extraction")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data")
REF_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks"

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

N_CTRL = 2000
N_HVG = 2000
LFC_THRESHOLD = 0.5
PVAL_THRESHOLD = 0.05
MIN_DE_TARGETS = 5

N_LAYERS = 18
N_HEADS = 18
EVAL_LAYERS = list(range(N_LAYERS))
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1
LAYERS_PER_PASS = 6


def load_categorical_column(h5group, col_name):
    col = h5group[col_name]
    if isinstance(col, h5py.Group):
        categories = col['categories'][:]
        codes = col['codes'][:]
        if categories.dtype.kind in ('O', 'S'):
            categories = np.array([x.decode() if isinstance(x, bytes) else x for x in categories])
        return categories[codes]
    else:
        data = col[:]
        if data.dtype.kind in ('O', 'S'):
            return np.array([x.decode() if isinstance(x, bytes) else x for x in data])
        return data


def call_de_vectorized(X_ctrl, X_pert):
    n_ctrl = X_ctrl.shape[0]
    n_pert = X_pert.shape[0]
    mean_ctrl = X_ctrl.mean(axis=0)
    mean_pert = X_pert.mean(axis=0)
    lfcs = mean_pert - mean_ctrl
    var_ctrl = X_ctrl.var(axis=0, ddof=1)
    var_pert = X_pert.var(axis=0, ddof=1)
    se_sq = var_ctrl / n_ctrl + var_pert / n_pert
    se = np.sqrt(se_sq)
    with np.errstate(divide='ignore', invalid='ignore'):
        df_num = se_sq ** 2
        df_den = (var_ctrl / n_ctrl) ** 2 / max(n_ctrl - 1, 1) + \
                 (var_pert / n_pert) ** 2 / max(n_pert - 1, 1)
        df = np.where(df_den > 0, df_num / df_den, 1.0)
        t_stats = np.where(se > 0, lfcs / se, 0.0)
    pvals = 2.0 * stats.t.sf(np.abs(t_stats), df)
    pvals = np.where(se > 0, pvals, 1.0)
    pvals = np.nan_to_num(pvals, nan=1.0)
    return pvals, lfcs


def tokenize_cell(expression_vector, var_indices, token_ids, medians, max_len=2048):
    expr = expression_vector[var_indices]
    nonzero = expr > 0
    if nonzero.sum() == 0:
        return None
    expr_nz = expr[nonzero]
    tokens_nz = token_ids[nonzero]
    medians_nz = medians[nonzero]
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = expr_nz / medians_nz
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0)
    rank_order = np.argsort(-normalized)
    ranked_tokens = tokens_nz[rank_order][:max_len - 2]
    return np.concatenate([[2], ranked_tokens, [3]]).astype(np.int64)


def load_trrust():
    path = os.path.join(REF_DIR, "trrust_human.tsv")
    df = pd.read_csv(path, sep='\t', header=None, names=['tf', 'target', 'mode', 'pmid'])
    return set(zip(df['tf'].str.upper(), df['target'].str.upper()))


def run():
    import torch
    from transformers import BertForMaskedLM

    total_t0 = time.time()

    print("=" * 70)
    print("ROUND 9 UNIFIED K562 ATTENTION EXTRACTION (v2 vectorized)")
    print(f"All {N_LAYERS} layers × {N_HEADS} heads")
    print("=" * 70)

    os.makedirs(os.path.join(OUT_DIR, "layer_mean_edges"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "per_head_trrust"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)

    # ============================================================
    # STEP 1: Load data
    # ============================================================
    print("\nSTEP 1: Loading Replogle CRISPRi data...")
    t0 = time.time()

    with h5py.File(DATA_PATH, 'r') as f:
        cell_genes = load_categorical_column(f['obs'], 'gene')
        var_genes = load_categorical_column(f['var'], 'gene_name_index')
        n_cells_total, n_genes_total = f['X'].shape

    print(f"  Total: {n_cells_total:,} cells × {n_genes_total:,} genes")

    control_mask = np.zeros(n_cells_total, dtype=bool)
    for ctrl_name in ['non-targeting', 'Non-targeting', 'non_targeting']:
        control_mask |= (cell_genes == ctrl_name)
    ctrl_indices = np.where(control_mask)[0]
    print(f"  Control cells: {len(ctrl_indices)}")

    np.random.seed(42)
    if len(ctrl_indices) > N_CTRL:
        ctrl_indices = np.random.choice(ctrl_indices, N_CTRL, replace=False)
        ctrl_indices.sort()
    print(f"  Using {len(ctrl_indices)} control cells")

    print("  Loading control expression matrix...")
    with h5py.File(DATA_PATH, 'r') as f:
        X_ctrl_full = np.empty((len(ctrl_indices), n_genes_total), dtype=np.float32)
        for ci, idx in enumerate(ctrl_indices):
            X_ctrl_full[ci, :] = f['X'][int(idx), :]
            if (ci + 1) % 500 == 0:
                print(f"    ... {ci+1}/{len(ctrl_indices)} cells loaded")

    print("  Normalizing...")
    row_sums = X_ctrl_full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_ctrl_norm = np.log1p(X_ctrl_full / row_sums * 1e4)

    print(f"  Selecting {N_HVG} HVGs...")
    gene_var = X_ctrl_norm.var(axis=0)
    hvg_idx = np.argsort(-gene_var)[:N_HVG]
    hvg_idx.sort()
    hvg_genes = [var_genes[i] for i in hvg_idx]
    hvg_upper = [g.upper() for g in hvg_genes]
    hvg_to_local = {g.upper(): i for i, g in enumerate(hvg_genes)}

    X_ctrl_hvg = X_ctrl_norm[:, hvg_idx]
    print(f"  HVG matrix: {X_ctrl_hvg.shape}")

    # Gene-level features
    print("  Computing gene-level features...")
    gene_mean = X_ctrl_hvg.mean(axis=0)
    gene_var_vals = X_ctrl_hvg.var(axis=0)
    gene_dropout = (X_ctrl_hvg == 0).mean(axis=0)
    gene_features_df = pd.DataFrame({
        'gene': hvg_genes, 'gene_upper': hvg_upper,
        'mean_expr': gene_mean, 'variance': gene_var_vals, 'dropout_rate': gene_dropout,
    })
    gene_features_df.to_csv(os.path.join(OUT_DIR, "gene_features.csv"), index=False)
    with open(os.path.join(OUT_DIR, "hvg_gene_list.json"), 'w') as f:
        json.dump(hvg_genes, f)
    print(f"  Gene features saved ({len(gene_features_df)} genes)")
    print(f"  Step 1 time: {time.time()-t0:.1f}s")

    # ============================================================
    # STEP 2: Load Geneformer tokenization
    # ============================================================
    print("\nSTEP 2: Loading Geneformer gene mappings...")

    with open(os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        token_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl"), 'rb') as f:
        gene_median_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_name_id_dict_gc104M.pkl"), 'rb') as f:
        gene_name_id_dict = pickle.load(f)

    print(f"  Token dict: {len(token_dict)}, Median dict: {len(gene_median_dict)}, Name->ID: {len(gene_name_id_dict)}")

    mapped_var_indices = []
    mapped_token_ids_list = []
    mapped_medians_list = []

    for i in range(n_genes_total):
        gene_name = var_genes[i]
        ensembl = gene_name_id_dict.get(gene_name)
        if ensembl and ensembl in token_dict:
            mapped_var_indices.append(i)
            mapped_token_ids_list.append(token_dict[ensembl])
            mapped_medians_list.append(gene_median_dict.get(ensembl, 1.0))

    mapped_var_indices = np.array(mapped_var_indices)
    mapped_token_ids = np.array(mapped_token_ids_list)
    mapped_medians = np.array(mapped_medians_list)

    # Build token_id -> HVG matrix index mapping for vectorized accumulation
    hvg_to_token = {}
    for g in hvg_genes:
        ens = gene_name_id_dict.get(g)
        if ens and ens in token_dict:
            hvg_to_token[g] = token_dict[ens]
    token_to_hvg = {v: k for k, v in hvg_to_token.items()}

    # Key mapping: token_id -> matrix index (0..n_hvg_tok-1)
    hvg_token_ids = sorted(token_to_hvg.keys())
    n_hvg_tok = len(hvg_token_ids)
    token_to_mat_idx = {tid: i for i, tid in enumerate(hvg_token_ids)}
    mat_idx_to_gene = {i: token_to_hvg[tid] for i, tid in enumerate(hvg_token_ids)}

    print(f"  Mapped var genes: {len(mapped_var_indices)}/{n_genes_total}")
    print(f"  HVG genes with tokens: {n_hvg_tok}/{N_HVG}")

    # ============================================================
    # STEP 2b: Load TRRUST for per-head ranking
    # ============================================================
    print("\n  Loading TRRUST for per-head evaluation...")
    trrust = load_trrust()
    trrust_tfs = set(tf for tf, _ in trrust)

    hvg_upper_set = set(hvg_upper)
    trrust_pairs_in_hvg = []
    for tf, tgt in trrust:
        if tf in hvg_upper_set and tgt in hvg_upper_set:
            trrust_pairs_in_hvg.append((tf, tgt))
    trrust_pairs_set = set(trrust_pairs_in_hvg)

    # Pre-compute TRRUST pair matrix indices for vectorized per-head eval
    trrust_src_mat = []
    trrust_tgt_mat = []
    for tf, tgt in trrust_pairs_in_hvg:
        tf_orig = hvg_genes[hvg_to_local[tf]]
        tgt_orig = hvg_genes[hvg_to_local[tgt]]
        tf_tok = hvg_to_token.get(tf_orig)
        tgt_tok = hvg_to_token.get(tgt_orig)
        if tf_tok is not None and tgt_tok is not None:
            trrust_src_mat.append(token_to_mat_idx[tf_tok])
            trrust_tgt_mat.append(token_to_mat_idx[tgt_tok])
    trrust_src_mat = np.array(trrust_src_mat, dtype=np.int32)
    trrust_tgt_mat = np.array(trrust_tgt_mat, dtype=np.int32)

    tfs_in_data = sorted([g for g in hvg_upper_set if g in trrust_tfs])
    print(f"  TRRUST pairs in HVG: {len(trrust_pairs_in_hvg)}")
    print(f"  TRRUST pairs with token mapping: {len(trrust_src_mat)}")
    print(f"  TFs in HVG: {len(tfs_in_data)}")

    # ============================================================
    # STEP 3: Tokenize control cells
    # ============================================================
    print("\nSTEP 3: Tokenizing control cells...")
    t1 = time.time()

    control_tokens = []
    with h5py.File(DATA_PATH, 'r') as f:
        X = f['X']
        for ci, idx in enumerate(ctrl_indices):
            expr = X[int(idx), :]
            tokens = tokenize_cell(expr, mapped_var_indices, mapped_token_ids,
                                   mapped_medians, max_len=MAX_SEQ_LEN)
            if tokens is not None:
                control_tokens.append(tokens)
            if (ci + 1) % 500 == 0:
                print(f"    Tokenized {ci+1}/{len(ctrl_indices)}")

    print(f"  Tokenized: {len(control_tokens)} cells, avg {np.mean([len(t) for t in control_tokens]):.0f} tokens")
    print(f"  Time: {time.time()-t1:.1f}s")

    # Sort by length for efficient batching
    sorted_indices = sorted(range(len(control_tokens)), key=lambda i: len(control_tokens[i]))

    # ============================================================
    # STEP 4: Extract attention - VECTORIZED with checkpoints
    # ============================================================
    print(f"\nSTEP 4: Extracting Geneformer attention ({N_LAYERS} layers in 3 passes)...")
    print(f"  Matrix accumulator size: {n_hvg_tok} x {n_hvg_tok} = {n_hvg_tok**2/1e6:.1f}M entries per layer")
    t2 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  Device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print(f"  Device: CPU (this will be slow)")

    print(f"  Loading Geneformer V2-316M...")
    model = BertForMaskedLM.from_pretrained(
        MODEL_NAME, subfolder=MODEL_SUBFOLDER,
        output_attentions=False, output_hidden_states=False,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Numpy matrix accumulators: sum and count for each layer
    # shape: (n_hvg_tok, n_hvg_tok) - rows=source, cols=target
    layer_sum_matrices = {}
    layer_count_matrices = {}
    # Per-head: only accumulate for TRRUST pairs (178 pairs, not 1941^2)
    # shape per layer: (N_HEADS, n_trrust_pairs) — tiny
    n_trrust = len(trrust_src_mat)
    per_head_trrust_sum = {}   # layer -> (N_HEADS, n_trrust)
    per_head_trrust_count = {} # layer -> (N_HEADS, n_trrust)
    # Build set for fast lookup: mat_idx pair -> trrust pair index
    trrust_pair_to_idx = {}
    for ti, (si, ti2) in enumerate(zip(trrust_src_mat, trrust_tgt_mat)):
        trrust_pair_to_idx[(int(si), int(ti2))] = ti

    layer_groups = [list(range(i, min(i + LAYERS_PER_PASS, N_LAYERS)))
                    for i in range(0, N_LAYERS, LAYERS_PER_PASS)]

    for pass_num, layer_group in enumerate(layer_groups):
        # Check for checkpoint
        ckpt_path = os.path.join(OUT_DIR, "checkpoints", f"pass_{pass_num}_done.json")
        if os.path.exists(ckpt_path):
            print(f"\n  Pass {pass_num+1}/{len(layer_groups)}: RESUMING (checkpoint found)")
            # Load saved matrices for this pass
            for layer_idx in layer_group:
                sum_path = os.path.join(OUT_DIR, "checkpoints", f"layer_{layer_idx:02d}_sum.npy")
                cnt_path = os.path.join(OUT_DIR, "checkpoints", f"layer_{layer_idx:02d}_count.npy")
                hsum_path = os.path.join(OUT_DIR, "checkpoints", f"layer_{layer_idx:02d}_head_trrust_sum.npy")
                hcnt_path = os.path.join(OUT_DIR, "checkpoints", f"layer_{layer_idx:02d}_head_trrust_count.npy")
                layer_sum_matrices[layer_idx] = np.load(sum_path)
                layer_count_matrices[layer_idx] = np.load(cnt_path)
                per_head_trrust_sum[layer_idx] = np.load(hsum_path)
                per_head_trrust_count[layer_idx] = np.load(hcnt_path)
            print(f"    Loaded checkpoint for layers {layer_group}")
            continue

        print(f"\n  Pass {pass_num+1}/{len(layer_groups)}: layers {layer_group}")

        # Initialize accumulators for this pass
        for layer_idx in layer_group:
            layer_sum_matrices[layer_idx] = np.zeros((n_hvg_tok, n_hvg_tok), dtype=np.float64)
            layer_count_matrices[layer_idx] = np.zeros((n_hvg_tok, n_hvg_tok), dtype=np.int32)
            per_head_trrust_sum[layer_idx] = np.zeros((N_HEADS, n_trrust), dtype=np.float64)
            per_head_trrust_count[layer_idx] = np.zeros((N_HEADS, n_trrust), dtype=np.int32)

        # Set up hooks
        captured_attentions = {}
        orig_forwards = {}

        def make_patched_forward(layer_idx, captured_dict, orig_fn):
            def patched_forward(hidden_states, attention_mask=None, head_mask=None,
                              encoder_hidden_states=None, past_key_values=None,
                              output_attentions=False, cache_position=None, **kwargs):
                result = orig_fn(
                    hidden_states, attention_mask=attention_mask, head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    past_key_values=past_key_values,
                    output_attentions=True, cache_position=cache_position, **kwargs
                )
                if len(result) > 1 and result[1] is not None:
                    captured_dict[layer_idx] = result[1].detach()
                return (result[0],)
            return patched_forward

        for layer_idx in layer_group:
            target_layer = model.bert.encoder.layer[layer_idx]
            orig_forwards[layer_idx] = target_layer.forward
            target_layer.forward = make_patched_forward(
                layer_idx, captured_attentions, orig_forwards[layer_idx])

        pass_t0 = time.time()
        cells_processed = 0

        for batch_start in range(0, len(control_tokens), BATCH_SIZE):
            batch_idx_list = sorted_indices[batch_start:batch_start + BATCH_SIZE]
            batch_tokens = [control_tokens[i] for i in batch_idx_list]

            for i in range(len(batch_tokens)):
                if len(batch_tokens[i]) > MAX_SEQ_LEN:
                    t = batch_tokens[i]
                    batch_tokens[i] = np.concatenate([t[:MAX_SEQ_LEN - 1], [t[-1]]])

            max_len = max(len(t) for t in batch_tokens)
            bs = len(batch_tokens)

            input_ids = np.full((bs, max_len), 0, dtype=np.int64)
            attention_mask = np.zeros((bs, max_len), dtype=np.int64)
            for i, tokens in enumerate(batch_tokens):
                input_ids[i, :len(tokens)] = tokens
                attention_mask[i, :len(tokens)] = 1

            input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
            attn_mask_t = torch.tensor(attention_mask, dtype=torch.long, device=device)

            with torch.no_grad():
                _ = model(input_ids=input_ids_t, attention_mask=attn_mask_t)

            # Vectorized accumulation for each layer
            for layer_idx in layer_group:
                if layer_idx not in captured_attentions:
                    continue

                # full_attn shape: (bs, n_heads, seq_len, seq_len)
                full_attn = captured_attentions[layer_idx].cpu().float().numpy()

                for cell_i in range(bs):
                    seq_len = int(attention_mask[cell_i].sum())
                    cell_tok = input_ids[cell_i, :seq_len]
                    gene_tokens = cell_tok[1:seq_len - 1]

                    # Vectorized: find which positions are HVG tokens
                    mat_indices = np.array([token_to_mat_idx.get(int(tid), -1)
                                           for tid in gene_tokens], dtype=np.int32)
                    hvg_mask = mat_indices >= 0
                    if hvg_mask.sum() < 2:
                        continue

                    hvg_pos = np.where(hvg_mask)[0] + 1  # +1 for CLS token offset
                    hvg_mat = mat_indices[hvg_mask]  # matrix indices for these positions

                    # Extract HVG submatrix from head-averaged attention
                    # head_avg shape: (seq_len, seq_len)
                    head_avg = full_attn[cell_i].mean(axis=0)  # average over heads
                    hvg_attn = head_avg[np.ix_(hvg_pos, hvg_pos)]  # (n_hvg_this, n_hvg_this)

                    # Zero out self-attention (same gene)
                    n_h = len(hvg_mat)
                    # Build same-gene mask
                    same_gene = (hvg_mat[:, None] == hvg_mat[None, :])
                    hvg_attn[same_gene] = 0.0

                    # Accumulate into matrix using np.add.at
                    src_idx = np.repeat(hvg_mat, n_h)
                    tgt_idx = np.tile(hvg_mat, n_h)
                    vals = hvg_attn.ravel()
                    # Filter out zeros (self-gene pairs already zeroed)
                    nonzero = vals != 0.0
                    np.add.at(layer_sum_matrices[layer_idx],
                              (src_idx[nonzero], tgt_idx[nonzero]), vals[nonzero])
                    np.add.at(layer_count_matrices[layer_idx],
                              (src_idx[nonzero], tgt_idx[nonzero]), 1)

                    # Per-head accumulation: TRRUST pairs only (178 pairs, not 1941^2)
                    # Build reverse lookup: mat_idx -> position in hvg_pos
                    hvg_mat_set = set(hvg_mat.tolist())
                    mat_to_pos = {}
                    for pi2, mi in enumerate(hvg_mat.tolist()):
                        if mi not in mat_to_pos:  # first occurrence
                            mat_to_pos[mi] = hvg_pos[pi2]

                    for ti, (src_mi, tgt_mi) in enumerate(zip(
                            trrust_src_mat.tolist(), trrust_tgt_mat.tolist())):
                        if src_mi not in hvg_mat_set or tgt_mi not in hvg_mat_set:
                            continue
                        sp = mat_to_pos[src_mi]
                        tp = mat_to_pos[tgt_mi]
                        # Extract all heads at once: full_attn[cell_i, :, sp, tp] -> (N_HEADS,)
                        head_vals = full_attn[cell_i, :, sp, tp]
                        per_head_trrust_sum[layer_idx][:, ti] += head_vals
                        per_head_trrust_count[layer_idx][:, ti] += 1

                del full_attn

            cells_processed += bs
            captured_attentions.clear()
            if device.type == 'mps':
                torch.mps.empty_cache()

            if cells_processed % 100 == 0 or cells_processed >= len(control_tokens):
                elapsed = time.time() - pass_t0
                rate = cells_processed / elapsed if elapsed > 0 else 0
                eta_pass = (len(control_tokens) - cells_processed) / rate if rate > 0 else 0
                remaining_passes = len(layer_groups) - pass_num - 1
                eta_remaining = remaining_passes * (len(control_tokens) / rate) if rate > 0 else 0
                print(f"    Pass {pass_num+1} cells: {cells_processed}/{len(control_tokens)}, "
                      f"time: {elapsed:.0f}s, rate: {rate:.1f} cells/s, "
                      f"pass ETA: {eta_pass:.0f}s, total ETA: {eta_pass + eta_remaining:.0f}s")

        # Restore original forwards
        for layer_idx in layer_group:
            model.bert.encoder.layer[layer_idx].forward = orig_forwards[layer_idx]
        del captured_attentions, orig_forwards
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()

        # Save checkpoint for this pass
        for layer_idx in layer_group:
            np.save(os.path.join(OUT_DIR, "checkpoints", f"layer_{layer_idx:02d}_sum.npy"),
                    layer_sum_matrices[layer_idx])
            np.save(os.path.join(OUT_DIR, "checkpoints", f"layer_{layer_idx:02d}_count.npy"),
                    layer_count_matrices[layer_idx])
            np.save(os.path.join(OUT_DIR, "checkpoints", f"layer_{layer_idx:02d}_head_trrust_sum.npy"),
                    per_head_trrust_sum[layer_idx])
            np.save(os.path.join(OUT_DIR, "checkpoints", f"layer_{layer_idx:02d}_head_trrust_count.npy"),
                    per_head_trrust_count[layer_idx])

        with open(ckpt_path, 'w') as f:
            json.dump({"pass": pass_num, "layers": layer_group,
                       "cells": cells_processed, "time": time.time() - pass_t0}, f)

        print(f"  Pass {pass_num+1} complete: {time.time()-pass_t0:.0f}s, checkpoint saved")

    print(f"\n  Total attention extraction time: {time.time()-t2:.1f}s")

    # Free model
    del model
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()

    # ============================================================
    # STEP 5: Average and save layer-mean edges as dicts
    # ============================================================
    print("\nSTEP 5: Averaging and saving layer-mean edges...")
    t3 = time.time()

    for layer_idx in EVAL_LAYERS:
        count_mat = layer_count_matrices[layer_idx]
        sum_mat = layer_sum_matrices[layer_idx]

        # Convert to edge dict (matching downstream expected format)
        nonzero_mask = count_mat > 0
        avg_mat = np.zeros_like(sum_mat)
        avg_mat[nonzero_mask] = sum_mat[nonzero_mask] / count_mat[nonzero_mask]

        edges = {}
        rows, cols = np.where(nonzero_mask)
        for r, c in zip(rows, cols):
            src_gene = mat_idx_to_gene[int(r)]
            tgt_gene = mat_idx_to_gene[int(c)]
            edges[(src_gene, tgt_gene)] = float(avg_mat[r, c])

        out_path = os.path.join(OUT_DIR, "layer_mean_edges", f"layer_{layer_idx:02d}_edges.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(edges, f)

        print(f"    Layer {layer_idx}: {len(edges)} edges saved")
        del layer_sum_matrices[layer_idx], layer_count_matrices[layer_idx], edges
        gc.collect()

    print(f"  Layer-mean edges saved: {time.time()-t3:.1f}s")

    # ============================================================
    # STEP 5b: Compute per-head TRRUST AUROCs from sparse accumulators
    # ============================================================
    print("\nSTEP 5b: Computing per-head TRRUST AUROCs...")

    # Build TF evaluation universe: for each TF, which TRRUST pair indices involve it?
    tf_mat_indices = []
    tf_names = []
    for tf_name in tfs_in_data:
        tf_orig = hvg_genes[hvg_to_local[tf_name]]
        tf_tok = hvg_to_token.get(tf_orig)
        if tf_tok is not None and tf_tok in token_to_mat_idx:
            tf_mat_indices.append(token_to_mat_idx[tf_tok])
            tf_names.append(tf_name)

    # Build full evaluation universe labels/scores from per-head TRRUST accumulators
    # For each head: score = per-head average attention for TRRUST pairs
    # AUROC computed against TF->all-targets universe
    head_trrust_aurocs = {}

    for layer_idx in EVAL_LAYERS:
        h_sum = per_head_trrust_sum[layer_idx]   # (N_HEADS, n_trrust)
        h_cnt = per_head_trrust_count[layer_idx]  # (N_HEADS, n_trrust)

        # Average per-head TRRUST attention
        with np.errstate(divide='ignore', invalid='ignore'):
            h_avg = np.where(h_cnt > 0, h_sum / h_cnt, 0.0)  # (N_HEADS, n_trrust)

        # Build per-head score lookup: (src_mat_idx, tgt_mat_idx) -> avg attention
        for head_idx in range(N_HEADS):
            head_scores = h_avg[head_idx]  # (n_trrust,)

            # Build score dict for this head
            pair_score = {}
            for ti in range(n_trrust):
                si, ti2 = int(trrust_src_mat[ti]), int(trrust_tgt_mat[ti])
                pair_score[(si, ti2)] = head_scores[ti]

            # Evaluate: for each TF, score all potential targets
            labels = []
            scores = []
            for tf_mi, tf_name in zip(tf_mat_indices, tf_names):
                for tgt_mi in range(n_hvg_tok):
                    if tgt_mi == tf_mi:
                        continue
                    tgt_gene = mat_idx_to_gene[tgt_mi].upper()
                    if tgt_gene == tf_name:
                        continue
                    is_pos = (tf_name, tgt_gene) in trrust_pairs_set
                    fwd = pair_score.get((tf_mi, tgt_mi), 0.0)
                    rev = pair_score.get((tgt_mi, tf_mi), 0.0)
                    score = max(fwd, rev)
                    labels.append(1 if is_pos else 0)
                    scores.append(score)

            labels = np.array(labels)
            scores = np.array(scores)
            if labels.sum() > 0 and labels.sum() < len(labels):
                try:
                    auroc = roc_auc_score(labels, scores)
                except:
                    auroc = 0.5
            else:
                auroc = 0.5
            head_trrust_aurocs[(layer_idx, head_idx)] = auroc

        del per_head_trrust_sum[layer_idx], per_head_trrust_count[layer_idx]
        gc.collect()

        best = max(head_trrust_aurocs.get((layer_idx, h), 0.5) for h in range(N_HEADS))
        print(f"    Layer {layer_idx}: best head AUROC = {best:.4f}")

    # Save
    head_auroc_data = {f"L{l}_H{h}": round(auroc, 6) for (l, h), auroc in head_trrust_aurocs.items()}
    with open(os.path.join(OUT_DIR, "per_head_trrust", "head_trrust_aurocs.json"), 'w') as f:
        json.dump(head_auroc_data, f, indent=2)

    auroc_matrix = np.zeros((N_LAYERS, N_HEADS))
    for (l, h), auroc in head_trrust_aurocs.items():
        auroc_matrix[l, h] = auroc
    np.save(os.path.join(OUT_DIR, "per_head_trrust", "head_trrust_auroc_matrix.npy"), auroc_matrix)

    head_ranking = sorted(head_trrust_aurocs.items(), key=lambda x: -x[1])
    top5 = head_ranking[:5]
    print(f"\n  Top-5 regulatory heads:")
    for (l, h), auroc in top5:
        print(f"    L{l}_H{h}: AUROC = {auroc:.4f}")
    with open(os.path.join(OUT_DIR, "per_head_trrust", "top5_regulatory_heads.json"), 'w') as f:
        json.dump([{"layer": l, "head": h, "auroc": round(auroc, 6)} for (l, h), auroc in top5], f, indent=2)

    gc.collect()

    # ============================================================
    # STEP 6: Compute and save correlation matrix
    # ============================================================
    print("\nSTEP 6: Computing correlation edges...")
    t4 = time.time()
    ranks = np.apply_along_axis(rankdata, 0, X_ctrl_hvg)
    corr_matrix = np.corrcoef(ranks.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.save(os.path.join(OUT_DIR, "correlation_matrix.npy"), corr_matrix)
    print(f"  Correlation matrix: {corr_matrix.shape}, saved")
    print(f"  Time: {time.time()-t4:.1f}s")

    # ============================================================
    # STEP 7: Compute DE for each perturbation
    # ============================================================
    print("\nSTEP 7: Computing DE labels for evaluable perturbations...")
    t5 = time.time()

    unique_perts = np.unique(cell_genes)
    evaluable_perts = []
    pert_cell_indices = {}

    for pg in unique_perts:
        pg_str = str(pg)
        if pg_str.lower() in ('non-targeting', 'non_targeting', 'control', 'ctrl'):
            continue
        if pg_str.upper() not in hvg_to_local:
            continue
        pert_mask = (cell_genes == pg)
        pert_idx = np.where(pert_mask)[0]
        if len(pert_idx) >= 10:
            evaluable_perts.append(pg_str)
            pert_cell_indices[pg_str] = pert_idx

    print(f"  Evaluable perturbations in HVG: {len(evaluable_perts)}")

    de_labels = {}
    for pi, pg in enumerate(evaluable_perts):
        pidx = pert_cell_indices[pg]
        with h5py.File(DATA_PATH, 'r') as f:
            X_pert_full = np.empty((len(pidx), n_genes_total), dtype=np.float32)
            for ci, idx in enumerate(pidx):
                X_pert_full[ci, :] = f['X'][int(idx), :]

        row_sums_p = X_pert_full.sum(axis=1, keepdims=True)
        row_sums_p[row_sums_p == 0] = 1
        X_pert_norm = np.log1p(X_pert_full / row_sums_p * 1e4)
        X_pert_hvg = X_pert_norm[:, hvg_idx]

        pvals, lfcs = call_de_vectorized(X_ctrl_hvg, X_pert_hvg)
        reject, adj_pvals, _, _ = multipletests(pvals, alpha=PVAL_THRESHOLD, method='fdr_bh')

        targets = {}
        pg_local_idx = hvg_to_local.get(pg.upper())
        for j in range(N_HVG):
            if pg_local_idx is not None and j == pg_local_idx:
                continue
            is_de = bool(reject[j]) and abs(lfcs[j]) >= LFC_THRESHOLD
            targets[hvg_upper[j]] = is_de

        n_de = sum(1 for v in targets.values() if v)
        if n_de >= MIN_DE_TARGETS:
            de_labels[pg] = targets

        del X_pert_full, X_pert_norm, X_pert_hvg
        gc.collect()

        if (pi + 1) % 100 == 0:
            print(f"    ... {pi+1}/{len(evaluable_perts)} perturbations, "
                  f"{len(de_labels)} evaluable so far")

    print(f"  Evaluable perturbations with >= {MIN_DE_TARGETS} DE targets: {len(de_labels)}")

    # Update gene features with n_perturbations_affected
    n_affected = np.zeros(N_HVG)
    for pg, targets in de_labels.items():
        for tgt, is_de in targets.items():
            if is_de:
                local_idx = hvg_to_local.get(tgt)
                if local_idx is not None:
                    n_affected[local_idx] += 1
    gene_features_df = pd.read_csv(os.path.join(OUT_DIR, "gene_features.csv"))
    gene_features_df['n_perturbations_affected'] = n_affected
    gene_features_df.to_csv(os.path.join(OUT_DIR, "gene_features.csv"), index=False)

    with open(os.path.join(OUT_DIR, "de_labels.pkl"), 'wb') as f:
        pickle.dump(de_labels, f)
    print(f"  DE computation time: {time.time()-t5:.1f}s")

    # ============================================================
    # STEP 8: Save config
    # ============================================================
    config = {
        'experiment': 'round9_unified_extraction_v2',
        'model': 'Geneformer-V2-316M',
        'n_control_cells': N_CTRL,
        'n_hvg': N_HVG,
        'n_hvg_with_tokens': n_hvg_tok,
        'lfc_threshold': LFC_THRESHOLD,
        'pval_threshold': PVAL_THRESHOLD,
        'min_de_targets': MIN_DE_TARGETS,
        'n_layers': N_LAYERS,
        'n_heads': N_HEADS,
        'max_seq_len': MAX_SEQ_LEN,
        'n_evaluable_perturbations': len(de_labels),
        'n_trrust_pairs_in_hvg': len(trrust_pairs_in_hvg),
        'n_tfs_in_data': len(tfs_in_data),
        'top5_regulatory_heads': [
            {"layer": l, "head": h, "auroc": round(auroc, 6)}
            for (l, h), auroc in top5
        ],
        'total_time_seconds': round(time.time() - total_t0, 1),
    }
    with open(os.path.join(OUT_DIR, "extraction_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"UNIFIED EXTRACTION COMPLETE - Total: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"\nOutputs saved to: {OUT_DIR}")
    print(f"  layer_mean_edges/: 18 pickle files")
    print(f"  per_head_trrust/: head ranking JSON + matrix")
    print(f"  correlation_matrix.npy")
    print(f"  de_labels.pkl ({len(de_labels)} perturbations)")
    print(f"  gene_features.csv ({N_HVG} genes)")
    print(f"  hvg_gene_list.json")
    print(f"  extraction_config.json")


if __name__ == '__main__':
    run()
