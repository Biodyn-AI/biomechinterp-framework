#!/usr/bin/env python3
"""
Fix per-head TRRUST AUROC ranking.

The unified extraction only accumulated per-head attention for TRRUST pairs,
causing all heads to get AUROC≈1.0 (non-TRRUST pairs score 0). This script
re-extracts per-head attention from 500 cells across all 18 layers,
accumulating ALL HVG pairs, then computes proper TRRUST AUROC.

Uses direct += instead of np.add.at since gene indices are unique per cell.
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
from sklearn.metrics import roc_auc_score

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
DATA_PATH = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data", "replogle_concat.h5ad")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round9_unified_extraction")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data")
REF_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks"

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

N_CELLS = 500  # Subset for per-head ranking
N_LAYERS = 18
N_HEADS = 18
LAYERS_PER_PASS = 6
MAX_SEQ_LEN = 2048


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


def run():
    import torch
    from transformers import BertForMaskedLM

    t0 = time.time()
    print("=" * 70)
    print("FIX PER-HEAD TRRUST RANKING")
    print(f"Re-extracting per-head attention from {N_CELLS} cells")
    print("=" * 70)

    # Load HVG gene list from extraction
    with open(os.path.join(OUT_DIR, "hvg_gene_list.json")) as f:
        hvg_genes = json.load(f)
    hvg_upper = [g.upper() for g in hvg_genes]
    hvg_to_local = {g.upper(): i for i, g in enumerate(hvg_genes)}
    n_hvg = len(hvg_genes)

    # Load token mappings
    with open(os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        token_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl"), 'rb') as f:
        gene_median_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_name_id_dict_gc104M.pkl"), 'rb') as f:
        gene_name_id_dict = pickle.load(f)

    # Build mappings
    with h5py.File(DATA_PATH, 'r') as f:
        var_genes = load_categorical_column(f['var'], 'gene_name_index')
        cell_genes = load_categorical_column(f['obs'], 'gene')
        n_cells_total, n_genes_total = f['X'].shape

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

    hvg_to_token = {}
    for g in hvg_genes:
        ens = gene_name_id_dict.get(g)
        if ens and ens in token_dict:
            hvg_to_token[g] = token_dict[ens]
    token_to_hvg = {v: k for k, v in hvg_to_token.items()}

    hvg_token_ids = sorted(token_to_hvg.keys())
    n_hvg_tok = len(hvg_token_ids)
    token_to_mat_idx = {tid: i for i, tid in enumerate(hvg_token_ids)}
    mat_idx_to_gene = {i: token_to_hvg[tid] for i, tid in enumerate(hvg_token_ids)}

    print(f"  HVG genes with tokens: {n_hvg_tok}")

    # Load TRRUST
    trrust_path = os.path.join(REF_DIR, "trrust_human.tsv")
    df = pd.read_csv(trrust_path, sep='\t', header=None, names=['tf', 'target', 'mode', 'pmid'])
    trrust = set(zip(df['tf'].str.upper(), df['target'].str.upper()))
    trrust_tfs = set(tf for tf, _ in trrust)
    hvg_upper_set = set(hvg_upper)
    trrust_pairs_set = set((tf, tgt) for tf, tgt in trrust
                           if tf in hvg_upper_set and tgt in hvg_upper_set)
    tfs_in_data = sorted([g for g in hvg_upper_set if g in trrust_tfs])
    print(f"  TRRUST pairs in HVG: {len(trrust_pairs_set)}")
    print(f"  TFs in HVG: {len(tfs_in_data)}")

    # TF matrix indices for evaluation
    tf_mat_indices = []
    tf_names = []
    for tf_name in tfs_in_data:
        tf_orig = hvg_genes[hvg_to_local[tf_name]]
        tf_tok = hvg_to_token.get(tf_orig)
        if tf_tok is not None and tf_tok in token_to_mat_idx:
            tf_mat_indices.append(token_to_mat_idx[tf_tok])
            tf_names.append(tf_name)
    print(f"  TFs with matrix index: {len(tf_mat_indices)}")

    # Select and tokenize control cells
    control_mask = np.zeros(n_cells_total, dtype=bool)
    for ctrl_name in ['non-targeting', 'Non-targeting', 'non_targeting']:
        control_mask |= (cell_genes == ctrl_name)
    ctrl_indices = np.where(control_mask)[0]
    np.random.seed(42)
    if len(ctrl_indices) > N_CELLS:
        ctrl_indices = np.random.choice(ctrl_indices, N_CELLS, replace=False)
        ctrl_indices.sort()

    print(f"\n  Tokenizing {N_CELLS} control cells...")
    control_tokens = []
    with h5py.File(DATA_PATH, 'r') as f:
        X = f['X']
        for ci, idx in enumerate(ctrl_indices):
            expr = X[int(idx), :]
            tokens = tokenize_cell(expr, mapped_var_indices, mapped_token_ids,
                                   mapped_medians, max_len=MAX_SEQ_LEN)
            if tokens is not None:
                control_tokens.append(tokens)
    print(f"  Tokenized: {len(control_tokens)} cells")

    sorted_indices = sorted(range(len(control_tokens)), key=lambda i: len(control_tokens[i]))

    # Load model
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    print(f"  Loading Geneformer V2-316M...")
    model = BertForMaskedLM.from_pretrained(
        MODEL_NAME, subfolder=MODEL_SUBFOLDER,
        output_attentions=False, output_hidden_states=False,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()
    print(f"  Model loaded")

    # Per-head accumulators: full (n_hvg_tok, n_hvg_tok) per head per layer
    # Using += since gene indices are unique per cell (rank-value encoding)
    per_head_sum = {}
    per_head_count = {}

    layer_groups = [list(range(i, min(i + LAYERS_PER_PASS, N_LAYERS)))
                    for i in range(0, N_LAYERS, LAYERS_PER_PASS)]

    for pass_num, layer_group in enumerate(layer_groups):
        print(f"\n  Pass {pass_num+1}/{len(layer_groups)}: layers {layer_group}")

        for layer_idx in layer_group:
            per_head_sum[layer_idx] = np.zeros((N_HEADS, n_hvg_tok, n_hvg_tok), dtype=np.float64)
            per_head_count[layer_idx] = np.zeros((N_HEADS, n_hvg_tok, n_hvg_tok), dtype=np.int32)

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

        for batch_start in range(0, len(control_tokens), 1):
            ci = sorted_indices[batch_start]
            tokens = control_tokens[ci]
            if len(tokens) > MAX_SEQ_LEN:
                tokens = np.concatenate([tokens[:MAX_SEQ_LEN - 1], [tokens[-1]]])
            seq_len = len(tokens)

            input_ids = torch.tensor(tokens.reshape(1, -1), dtype=torch.long, device=device)
            attn_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attn_mask)

            gene_tokens = tokens[1:seq_len - 1]
            mat_indices = np.array([token_to_mat_idx.get(int(tid), -1)
                                   for tid in gene_tokens], dtype=np.int32)
            hvg_mask = mat_indices >= 0
            if hvg_mask.sum() < 2:
                cells_processed += 1
                captured_attentions.clear()
                continue

            hvg_pos = np.where(hvg_mask)[0] + 1
            hvg_mat = mat_indices[hvg_mask]

            # Verify unique indices (should always be true for Geneformer)
            assert len(hvg_mat) == len(set(hvg_mat.tolist())), "Duplicate gene indices!"

            same_gene = (hvg_mat[:, None] == hvg_mat[None, :])
            count_patch = np.ones((len(hvg_mat), len(hvg_mat)), dtype=np.int32)
            count_patch[same_gene] = 0

            for layer_idx in layer_group:
                if layer_idx not in captured_attentions:
                    continue

                # full_attn shape: (1, N_HEADS, seq_len, seq_len)
                full_attn = captured_attentions[layer_idx].cpu().float().numpy()[0]

                # Extract HVG submatrix for all heads at once: (N_HEADS, n_hvg, n_hvg)
                hvg_submatrix = full_attn[:, hvg_pos][:, :, hvg_pos]
                hvg_submatrix[:, same_gene] = 0.0

                # Accumulate using direct += (indices are unique)
                ix = np.ix_(hvg_mat, hvg_mat)
                for head in range(N_HEADS):
                    per_head_sum[layer_idx][head][ix] += hvg_submatrix[head]
                per_head_count[layer_idx][:, hvg_mat[:, None], hvg_mat[None, :]] += count_patch

                del full_attn, hvg_submatrix

            cells_processed += 1
            captured_attentions.clear()
            if device.type == 'mps':
                torch.mps.empty_cache()

            if cells_processed % 50 == 0:
                elapsed = time.time() - pass_t0
                rate = cells_processed / elapsed
                eta = (len(control_tokens) - cells_processed) / rate
                print(f"    Cells: {cells_processed}/{len(control_tokens)}, "
                      f"rate: {rate:.1f} cells/s, ETA: {eta:.0f}s")

        for layer_idx in layer_group:
            model.bert.encoder.layer[layer_idx].forward = orig_forwards[layer_idx]
        del captured_attentions, orig_forwards
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()

        print(f"  Pass {pass_num+1} complete: {time.time()-pass_t0:.0f}s")

    del model
    gc.collect()

    # Compute per-head TRRUST AUROC
    print(f"\n  Computing per-head TRRUST AUROCs...")
    head_trrust_aurocs = {}

    for layer_idx in range(N_LAYERS):
        for head_idx in range(N_HEADS):
            cnt = per_head_count[layer_idx][head_idx]
            sm = per_head_sum[layer_idx][head_idx]
            nz = cnt > 0
            avg = np.zeros_like(sm)
            avg[nz] = sm[nz] / cnt[nz]

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
                    fwd = avg[tf_mi, tgt_mi]
                    rev = avg[tgt_mi, tf_mi]
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

        best = max(head_trrust_aurocs.get((layer_idx, h), 0.5) for h in range(N_HEADS))
        print(f"    Layer {layer_idx}: best head AUROC = {best:.4f}")

    # Save (overwrite)
    head_auroc_data = {f"L{l}_H{h}": round(auroc, 6)
                       for (l, h), auroc in head_trrust_aurocs.items()}
    with open(os.path.join(OUT_DIR, "per_head_trrust", "head_trrust_aurocs.json"), 'w') as f:
        json.dump(head_auroc_data, f, indent=2)

    auroc_matrix = np.zeros((N_LAYERS, N_HEADS))
    for (l, h), auroc in head_trrust_aurocs.items():
        auroc_matrix[l, h] = auroc
    np.save(os.path.join(OUT_DIR, "per_head_trrust", "head_trrust_auroc_matrix.npy"), auroc_matrix)

    head_ranking = sorted(head_trrust_aurocs.items(), key=lambda x: -x[1])
    top5 = head_ranking[:5]
    print(f"\n  Top-5 regulatory heads (FIXED):")
    for (l, h), auroc in top5:
        print(f"    L{l}_H{h}: AUROC = {auroc:.4f}")
    with open(os.path.join(OUT_DIR, "per_head_trrust", "top5_regulatory_heads.json"), 'w') as f:
        json.dump([{"layer": l, "head": h, "auroc": round(auroc, 6)}
                   for (l, h), auroc in top5], f, indent=2)

    # Also save bottom-5 for comparison
    bottom5 = head_ranking[-5:]
    print(f"\n  Bottom-5 heads:")
    for (l, h), auroc in bottom5:
        print(f"    L{l}_H{h}: AUROC = {auroc:.4f}")

    print(f"\n  Stats: min={auroc_matrix.min():.4f}, max={auroc_matrix.max():.4f}, "
          f"mean={auroc_matrix.mean():.4f}, std={auroc_matrix.std():.4f}")

    total_time = time.time() - t0
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Per-head TRRUST rankings saved to {OUT_DIR}/per_head_trrust/")


if __name__ == '__main__':
    run()
