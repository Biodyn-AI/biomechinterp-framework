#!/usr/bin/env python3
"""
Analysis A: Value-Weighted Edge Extraction from Geneformer V2-316M
===================================================================
Extracts value-weighted attention representations (context = softmax(QK^T/sqrt(d)) * V)
from Geneformer and computes pairwise cosine similarity as edge scores.

Hypothesis: Raw attention patterns (softmax(QK^T/sqrt(d))) capture co-expression.
The value pathway (V) routes actual information through heads. Value-weighted edges
may capture regulatory signal that raw attention misses.

Evidence supporting this: ablating TRRUST-ranked heads (high raw attention) has no
effect on perturbation prediction, but ablating random heads DOES hurt (d=0.33).
This implies predictive computation resides in the value/FFN pathway.

Output:
  - value_weighted_edges/layer_NN_vw_cos.npy   -- per-layer cosine similarity matrices
  - value_weighted_edges/layer_NN_vw_dot.npy   -- per-layer dot product matrices
  - value_weighted_edges/layer_NN_vw_centroid.npy -- centroid-based cosine matrices
  - value_weighted_results.json                  -- all AUROC comparisons, stats
  - value_weighted_summary.md                    -- human-readable summary
"""

import os
import sys
import gc
import json
import math
import time
import pickle
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# =============================================================================
# PATHS
# =============================================================================
PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
EXTRACTION_DIR = os.path.join(PAPER_DIR, "experiments", "round9_unified_extraction")
DATA_PATH = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data", "replogle_concat.h5ad")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data")
REF_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks"
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "biological_impact")

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

N_LAYERS = 18
N_HEADS = 18
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1
N_CTRL = 2000
N_FOLDS = 5


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def load_categorical_column(h5group, col_name):
    import h5py
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


def perturbation_first_auroc(edge_matrix, de_labels, hvg_genes, hvg_to_local,
                              mat_idx_to_gene, token_to_mat_idx, hvg_to_token,
                              symmetric=True):
    """Compute per-perturbation AUROC using edge_matrix (n_hvg_tok x n_hvg_tok).

    If symmetric=True (e.g., cosine similarity), edge(i,j) = matrix[i,j].
    If symmetric=False, edge(i,j) = max(matrix[i,j], matrix[j,i]).
    """
    hvg_upper = [g.upper() for g in hvg_genes]
    aurocs = []
    pert_names = []

    for pg, targets in de_labels.items():
        pg_local = hvg_to_local.get(pg.upper())
        if pg_local is None:
            continue
        pg_orig = hvg_genes[pg_local]
        pg_tok = hvg_to_token.get(pg_orig)
        if pg_tok is None or pg_tok not in token_to_mat_idx:
            continue
        pg_mat = token_to_mat_idx[pg_tok]

        labels = []
        scores = []
        for tgt_gene, is_de in targets.items():
            tgt_local = hvg_to_local.get(tgt_gene)
            if tgt_local is None:
                continue
            tgt_orig = hvg_genes[tgt_local]
            tgt_tok = hvg_to_token.get(tgt_orig)
            if tgt_tok is None or tgt_tok not in token_to_mat_idx:
                continue
            tgt_mat = token_to_mat_idx[tgt_tok]

            if symmetric:
                score = edge_matrix[pg_mat, tgt_mat]
            else:
                score = max(edge_matrix[pg_mat, tgt_mat], edge_matrix[tgt_mat, pg_mat])
            labels.append(int(is_de))
            scores.append(float(score))

        labels = np.array(labels)
        scores = np.array(scores)
        if labels.sum() > 0 and labels.sum() < len(labels):
            try:
                auroc = roc_auc_score(labels, scores)
                aurocs.append(auroc)
                pert_names.append(pg)
            except ValueError:
                pass

    return np.array(aurocs), pert_names


def trrust_auroc(edge_matrix, trrust_pairs, hvg_genes, hvg_to_local,
                 mat_idx_to_gene, token_to_mat_idx, hvg_to_token, n_hvg_tok,
                 symmetric=True):
    """Compute TRRUST AUROC using the standard TF-evaluation protocol."""
    hvg_upper = [g.upper() for g in hvg_genes]
    hvg_upper_set = set(hvg_upper)
    trrust_tfs = set(tf for tf, _ in trrust_pairs)
    tfs_in_data = sorted([g for g in hvg_upper_set if g in trrust_tfs])

    labels = []
    scores = []
    for tf_name in tfs_in_data:
        tf_local = hvg_to_local.get(tf_name)
        if tf_local is None:
            continue
        tf_orig = hvg_genes[tf_local]
        tf_tok = hvg_to_token.get(tf_orig)
        if tf_tok is None or tf_tok not in token_to_mat_idx:
            continue
        tf_mat = token_to_mat_idx[tf_tok]

        for tgt_mat in range(n_hvg_tok):
            if tgt_mat == tf_mat:
                continue
            tgt_gene = mat_idx_to_gene[tgt_mat].upper()
            if tgt_gene == tf_name:
                continue
            is_pos = (tf_name, tgt_gene) in trrust_pairs
            if symmetric:
                score = edge_matrix[tf_mat, tgt_mat]
            else:
                score = max(edge_matrix[tf_mat, tgt_mat], edge_matrix[tgt_mat, tf_mat])
            labels.append(1 if is_pos else 0)
            scores.append(float(score))

    labels = np.array(labels)
    scores = np.array(scores)
    if labels.sum() > 0 and labels.sum() < len(labels):
        return roc_auc_score(labels, scores)
    return 0.5


def run():
    import torch
    import h5py
    from transformers import BertForMaskedLM
    import torch.nn.functional as F

    total_t0 = time.time()

    print("=" * 70)
    print("ANALYSIS A: VALUE-WEIGHTED EDGE EXTRACTION")
    print("Geneformer V2-316M, K562 CRISPRi, 2000 control cells")
    print("=" * 70)

    # =================================================================
    # STEP 1: Load existing data from round9 extraction
    # =================================================================
    print("\nSTEP 1: Loading existing extraction data...")
    t0 = time.time()

    with open(os.path.join(EXTRACTION_DIR, "hvg_gene_list.json")) as f:
        hvg_genes = json.load(f)
    hvg_upper = [g.upper() for g in hvg_genes]
    hvg_to_local = {g.upper(): i for i, g in enumerate(hvg_genes)}

    with open(os.path.join(EXTRACTION_DIR, "de_labels.pkl"), 'rb') as f:
        de_labels = pickle.load(f)

    corr_matrix = np.load(os.path.join(EXTRACTION_DIR, "correlation_matrix.npy"))
    gene_features_df = pd.read_csv(os.path.join(EXTRACTION_DIR, "gene_features.csv"))

    # Load TRRUST
    trrust_path = os.path.join(REF_DIR, "trrust_human.tsv")
    trrust_df = pd.read_csv(trrust_path, sep='\t', header=None, names=['tf', 'target', 'mode', 'pmid'])
    trrust_pairs = set(zip(trrust_df['tf'].str.upper(), trrust_df['target'].str.upper()))

    print(f"  HVG genes: {len(hvg_genes)}")
    print(f"  DE labels: {len(de_labels)} perturbations")
    print(f"  TRRUST pairs: {len(trrust_pairs)}")
    print(f"  Step 1 time: {time.time()-t0:.1f}s")

    # =================================================================
    # STEP 2: Load tokenization and build mappings
    # =================================================================
    print("\nSTEP 2: Loading tokenization mappings...")

    with open(os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        token_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl"), 'rb') as f:
        gene_median_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_name_id_dict_gc104M.pkl"), 'rb') as f:
        gene_name_id_dict = pickle.load(f)

    # Build HVG -> token mapping
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

    print(f"  HVG genes with tokens: {n_hvg_tok}/{len(hvg_genes)}")

    # Build var_indices mapping for tokenization
    with h5py.File(DATA_PATH, 'r') as f:
        var_genes = load_categorical_column(f['var'], 'gene_name_index')
        n_genes_total = len(var_genes)
        cell_genes = load_categorical_column(f['obs'], 'gene')
        n_cells_total = len(cell_genes)

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

    # =================================================================
    # STEP 3: Tokenize control cells
    # =================================================================
    print("\nSTEP 3: Tokenizing control cells...")
    t1 = time.time()

    # Check for checkpoint
    tokens_ckpt = os.path.join(OUT_DIR, "checkpoints", "control_tokens_vw.pkl")
    if os.path.exists(tokens_ckpt):
        print("  Loading cached tokens...")
        with open(tokens_ckpt, 'rb') as f:
            control_tokens = pickle.load(f)
    else:
        control_mask = np.zeros(n_cells_total, dtype=bool)
        for ctrl_name in ['non-targeting', 'Non-targeting', 'non_targeting']:
            control_mask |= (cell_genes == ctrl_name)
        ctrl_indices = np.where(control_mask)[0]

        np.random.seed(42)
        if len(ctrl_indices) > N_CTRL:
            ctrl_indices = np.random.choice(ctrl_indices, N_CTRL, replace=False)
            ctrl_indices.sort()

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

        with open(tokens_ckpt, 'wb') as f:
            pickle.dump(control_tokens, f)

    print(f"  Tokenized: {len(control_tokens)} cells, avg {np.mean([len(t) for t in control_tokens]):.0f} tokens")
    print(f"  Time: {time.time()-t1:.1f}s")

    sorted_indices = sorted(range(len(control_tokens)), key=lambda i: len(control_tokens[i]))

    # =================================================================
    # STEP 4: Extract value-weighted representations
    # =================================================================
    print(f"\nSTEP 4: Extracting value-weighted context layers ({N_LAYERS} layers)...")
    t2 = time.time()

    # Check for completed extraction checkpoint
    vw_ckpt_path = os.path.join(OUT_DIR, "checkpoints", "vw_extraction_done.json")
    if os.path.exists(vw_ckpt_path):
        print("  Loading cached value-weighted accumulators...")
        # Load all accumulators
        cos_sum = {}
        cos_count = {}
        repr_sum = {}
        repr_count = {}
        for li in range(N_LAYERS):
            cos_sum[li] = np.load(os.path.join(OUT_DIR, "checkpoints", f"vw_cos_sum_L{li:02d}.npy"))
            cos_count[li] = np.load(os.path.join(OUT_DIR, "checkpoints", f"vw_cos_count_L{li:02d}.npy"))
            repr_sum[li] = np.load(os.path.join(OUT_DIR, "checkpoints", f"vw_repr_sum_L{li:02d}.npy"))
            repr_count[li] = np.load(os.path.join(OUT_DIR, "checkpoints", f"vw_repr_count_L{li:02d}.npy"))
        with open(vw_ckpt_path, 'r') as f:
            ckpt_info = json.load(f)
        d_head = ckpt_info['d_head']
        print(f"  Loaded checkpoint: {ckpt_info['cells_processed']} cells, d_head={d_head}")
    else:
        # Set up device and model
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"  Device: MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            print(f"  Device: CPU")

        print(f"  Loading Geneformer V2-316M...")
        model = BertForMaskedLM.from_pretrained(
            MODEL_NAME, subfolder=MODEL_SUBFOLDER,
            output_attentions=False, output_hidden_states=False,
            attn_implementation="eager",
        )
        model = model.to(device)
        model.eval()

        # Get model dimensions
        config = model.config
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        d_head = hidden_size // num_heads
        print(f"  Model: hidden={hidden_size}, heads={num_heads}, d_head={d_head}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

        # Initialize accumulators
        # Strategy 1: Centroid representations (average repr, then cosine at end)
        repr_sum = {li: np.zeros((n_hvg_tok, d_head), dtype=np.float64) for li in range(N_LAYERS)}
        repr_count = {li: np.zeros(n_hvg_tok, dtype=np.int32) for li in range(N_LAYERS)}

        # Strategy 2: Per-cell cosine average
        cos_sum = {li: np.zeros((n_hvg_tok, n_hvg_tok), dtype=np.float64) for li in range(N_LAYERS)}
        cos_count = {li: np.zeros((n_hvg_tok, n_hvg_tok), dtype=np.int32) for li in range(N_LAYERS)}

        # Also accumulate dot product (unnormalized) for alternative metric
        # (We'll compute dot product from centroid representations at the end)

        # Monkey-patch BertSelfAttention to capture context_layer
        captured_contexts = {}
        orig_self_attn_forwards = {}

        def _transpose_for_scores(x, num_heads, head_size):
            """Reshape (bs, seq, all_head) -> (bs, num_heads, seq, head_size)."""
            bs = x.size(0)
            return x.view(bs, -1, num_heads, head_size).transpose(1, 2)

        def make_capturing_self_attn_forward(layer_idx, capture_dict, self_attn_module):
            orig_forward = self_attn_module.forward
            n_heads = self_attn_module.num_attention_heads
            head_sz = self_attn_module.attention_head_size

            def new_forward(hidden_states, attention_mask=None, head_mask=None,
                            encoder_hidden_states=None, past_key_values=None,
                            output_attentions=False, cache_position=None, **kwargs):
                batch_size = hidden_states.size(0)

                # Compute Q, K, V with inline reshape (matches transformers 4.57+)
                query_layer = _transpose_for_scores(
                    self_attn_module.query(hidden_states), n_heads, head_sz)
                key_layer = _transpose_for_scores(
                    self_attn_module.key(hidden_states), n_heads, head_sz)
                value_layer = _transpose_for_scores(
                    self_attn_module.value(hidden_states), n_heads, head_sz)

                # Attention scores
                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                attention_scores = attention_scores / math.sqrt(head_sz)
                if attention_mask is not None:
                    attention_scores = attention_scores + attention_mask

                # Softmax
                attention_probs = F.softmax(attention_scores, dim=-1)
                # Dropout is identity in eval mode
                attention_probs = self_attn_module.dropout(attention_probs)
                if head_mask is not None:
                    attention_probs = attention_probs * head_mask

                # Context layer: softmax(QK^T/sqrt(d)) @ V
                # Shape: (bs, n_heads, seq_len, d_head)
                context_layer = torch.matmul(attention_probs, value_layer)

                # CAPTURE: store pre-reshape context on CPU
                capture_dict[layer_idx] = context_layer.detach().cpu().float().numpy()

                # Reshape for normal output (matches transformers 4.57 return format)
                context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                new_shape = context_layer.size()[:-2] + (self_attn_module.all_head_size,)
                context_layer = context_layer.view(new_shape)

                # Always return (context_layer, attention_probs) to match current API
                return context_layer, attention_probs

            return new_forward

        # Install hooks on ALL 18 layers
        print(f"  Installing hooks on {N_LAYERS} layers...")
        for li in range(N_LAYERS):
            self_attn = model.bert.encoder.layer[li].attention.self
            orig_self_attn_forwards[li] = self_attn.forward
            self_attn.forward = make_capturing_self_attn_forward(li, captured_contexts, self_attn)

        # Process cells
        print(f"\n  Processing {len(control_tokens)} cells (batch_size={BATCH_SIZE})...")
        cells_processed = 0
        pass_t0 = time.time()

        for batch_start in range(0, len(control_tokens), BATCH_SIZE):
            batch_idx_list = sorted_indices[batch_start:batch_start + BATCH_SIZE]
            batch_tokens = [control_tokens[i] for i in batch_idx_list]

            # Truncate if needed
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

            # Process captured context layers for each cell
            for cell_i in range(bs):
                seq_len = int(attention_mask[cell_i].sum())
                cell_tok = input_ids[cell_i, :seq_len]
                gene_tokens = cell_tok[1:seq_len - 1]  # exclude CLS and SEP

                # Map token IDs to HVG matrix indices
                mat_indices = np.array([token_to_mat_idx.get(int(tid), -1)
                                        for tid in gene_tokens], dtype=np.int32)
                hvg_mask = mat_indices >= 0
                if hvg_mask.sum() < 2:
                    continue

                hvg_pos = np.where(hvg_mask)[0] + 1  # +1 for CLS offset
                hvg_mat = mat_indices[hvg_mask]
                n_h = len(hvg_mat)

                for li in range(N_LAYERS):
                    if li not in captured_contexts:
                        continue

                    # context shape: (1, n_heads, seq_len, d_head)
                    context = captured_contexts[li][cell_i]  # (n_heads, seq_len, d_head)

                    # Average across heads -> (seq_len, d_head)
                    head_avg_context = context.mean(axis=0)

                    # Extract HVG representations
                    hvg_repr = head_avg_context[hvg_pos, :]  # (n_hvg, d_head)

                    # Strategy 1: Accumulate representations for centroid
                    np.add.at(repr_sum[li], hvg_mat, hvg_repr)
                    np.add.at(repr_count[li], hvg_mat, 1)

                    # Strategy 2: Compute cosine similarity per cell
                    norms = np.linalg.norm(hvg_repr, axis=1, keepdims=True)
                    norms[norms < 1e-8] = 1.0
                    hvg_normed = hvg_repr / norms
                    cos_sim = hvg_normed @ hvg_normed.T  # (n_hvg, n_hvg)

                    # Zero self-gene pairs
                    same_gene = (hvg_mat[:, None] == hvg_mat[None, :])
                    cos_sim[same_gene] = 0.0

                    # Accumulate
                    src_idx = np.repeat(hvg_mat, n_h)
                    tgt_idx = np.tile(hvg_mat, n_h)
                    vals = cos_sim.ravel()
                    nonzero = vals != 0.0
                    np.add.at(cos_sum[li], (src_idx[nonzero], tgt_idx[nonzero]), vals[nonzero])
                    np.add.at(cos_count[li], (src_idx[nonzero], tgt_idx[nonzero]), 1)

            cells_processed += bs
            captured_contexts.clear()
            if device.type == 'mps':
                torch.mps.empty_cache()

            if cells_processed % 50 == 0 or cells_processed >= len(control_tokens):
                elapsed = time.time() - pass_t0
                rate = cells_processed / elapsed if elapsed > 0 else 0
                eta = (len(control_tokens) - cells_processed) / rate if rate > 0 else 0
                print(f"    Cells: {cells_processed}/{len(control_tokens)}, "
                      f"time: {elapsed:.0f}s, rate: {rate:.2f} cells/s, ETA: {eta:.0f}s")

        # Restore original forwards
        for li in range(N_LAYERS):
            model.bert.encoder.layer[li].attention.self.forward = orig_self_attn_forwards[li]

        # Free model
        del model
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()

        # Save checkpoint
        for li in range(N_LAYERS):
            np.save(os.path.join(OUT_DIR, "checkpoints", f"vw_cos_sum_L{li:02d}.npy"), cos_sum[li])
            np.save(os.path.join(OUT_DIR, "checkpoints", f"vw_cos_count_L{li:02d}.npy"), cos_count[li])
            np.save(os.path.join(OUT_DIR, "checkpoints", f"vw_repr_sum_L{li:02d}.npy"), repr_sum[li])
            np.save(os.path.join(OUT_DIR, "checkpoints", f"vw_repr_count_L{li:02d}.npy"), repr_count[li])

        with open(vw_ckpt_path, 'w') as f:
            json.dump({
                'cells_processed': cells_processed,
                'd_head': d_head,
                'n_hvg_tok': n_hvg_tok,
                'extraction_time_s': time.time() - t2,
            }, f)

        print(f"\n  Extraction complete: {time.time()-t2:.1f}s")

    # =================================================================
    # STEP 5: Compute edge matrices
    # =================================================================
    print("\nSTEP 5: Computing edge matrices...")
    t3 = time.time()

    vw_edges_dir = os.path.join(OUT_DIR, "value_weighted_edges")
    os.makedirs(vw_edges_dir, exist_ok=True)

    for li in range(N_LAYERS):
        # Strategy 2: Per-cell average cosine similarity
        count_mat = cos_count[li]
        sum_mat = cos_sum[li]
        avg_cos = np.zeros_like(sum_mat)
        mask = count_mat > 0
        avg_cos[mask] = sum_mat[mask] / count_mat[mask]
        np.save(os.path.join(vw_edges_dir, f"layer_{li:02d}_vw_cos.npy"), avg_cos.astype(np.float32))

        # Strategy 1: Centroid cosine (average representations, then cosine)
        r_sum = repr_sum[li]
        r_count = repr_count[li]
        avg_repr = np.zeros_like(r_sum)
        valid = r_count > 0
        avg_repr[valid] = r_sum[valid] / r_count[valid, None]

        # Normalize
        norms = np.linalg.norm(avg_repr, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        avg_repr_normed = avg_repr / norms
        centroid_cos = avg_repr_normed @ avg_repr_normed.T
        np.fill_diagonal(centroid_cos, 0.0)
        np.save(os.path.join(vw_edges_dir, f"layer_{li:02d}_vw_centroid.npy"), centroid_cos.astype(np.float32))

        # Dot product (unnormalized centroid similarity)
        dot_product = avg_repr @ avg_repr.T
        np.fill_diagonal(dot_product, 0.0)
        np.save(os.path.join(vw_edges_dir, f"layer_{li:02d}_vw_dot.npy"), dot_product.astype(np.float32))

        print(f"    Layer {li}: cos range [{avg_cos[mask].min():.4f}, {avg_cos[mask].max():.4f}], "
              f"centroid range [{centroid_cos.min():.4f}, {centroid_cos.max():.4f}]")

    print(f"  Edge matrices saved: {time.time()-t3:.1f}s")

    # Free accumulators
    del cos_sum, cos_count, repr_sum, repr_count
    gc.collect()

    # =================================================================
    # STEP 6: Load raw attention edges for comparison
    # =================================================================
    print("\nSTEP 6: Loading raw attention edges for comparison...")
    raw_attn_matrices = {}
    for li in range(N_LAYERS):
        edge_path = os.path.join(EXTRACTION_DIR, "layer_mean_edges", f"layer_{li:02d}_edges.pkl")
        with open(edge_path, 'rb') as f:
            edges_dict = pickle.load(f)

        # Convert to matrix
        mat = np.zeros((n_hvg_tok, n_hvg_tok), dtype=np.float32)
        for (src, tgt), val in edges_dict.items():
            src_tok = hvg_to_token.get(src)
            tgt_tok = hvg_to_token.get(tgt)
            if src_tok is not None and tgt_tok is not None:
                src_mi = token_to_mat_idx.get(src_tok)
                tgt_mi = token_to_mat_idx.get(tgt_tok)
                if src_mi is not None and tgt_mi is not None:
                    mat[src_mi, tgt_mi] = val
        raw_attn_matrices[li] = mat
        del edges_dict
        gc.collect()
    print(f"  Loaded {len(raw_attn_matrices)} raw attention matrices")

    # =================================================================
    # STEP 7: Evaluate - TRRUST AUROC
    # =================================================================
    print("\nSTEP 7: TRRUST AUROC evaluation...")
    t4 = time.time()

    trrust_results = {'vw_cos': {}, 'vw_centroid': {}, 'vw_dot': {}, 'raw_attn': {}}

    for li in range(N_LAYERS):
        # Value-weighted cosine
        vw_cos_mat = np.load(os.path.join(vw_edges_dir, f"layer_{li:02d}_vw_cos.npy"))
        auroc = trrust_auroc(vw_cos_mat, trrust_pairs, hvg_genes, hvg_to_local,
                             mat_idx_to_gene, token_to_mat_idx, hvg_to_token, n_hvg_tok,
                             symmetric=True)
        trrust_results['vw_cos'][li] = round(auroc, 4)

        # Value-weighted centroid cosine
        vw_cent_mat = np.load(os.path.join(vw_edges_dir, f"layer_{li:02d}_vw_centroid.npy"))
        auroc = trrust_auroc(vw_cent_mat, trrust_pairs, hvg_genes, hvg_to_local,
                             mat_idx_to_gene, token_to_mat_idx, hvg_to_token, n_hvg_tok,
                             symmetric=True)
        trrust_results['vw_centroid'][li] = round(auroc, 4)

        # Value-weighted dot product
        vw_dot_mat = np.load(os.path.join(vw_edges_dir, f"layer_{li:02d}_vw_dot.npy"))
        auroc = trrust_auroc(vw_dot_mat, trrust_pairs, hvg_genes, hvg_to_local,
                             mat_idx_to_gene, token_to_mat_idx, hvg_to_token, n_hvg_tok,
                             symmetric=True)
        trrust_results['vw_dot'][li] = round(auroc, 4)

        # Raw attention (directional)
        auroc = trrust_auroc(raw_attn_matrices[li], trrust_pairs, hvg_genes, hvg_to_local,
                             mat_idx_to_gene, token_to_mat_idx, hvg_to_token, n_hvg_tok,
                             symmetric=False)
        trrust_results['raw_attn'][li] = round(auroc, 4)

        print(f"    L{li:2d}: vw_cos={trrust_results['vw_cos'][li]:.4f}, "
              f"vw_cent={trrust_results['vw_centroid'][li]:.4f}, "
              f"vw_dot={trrust_results['vw_dot'][li]:.4f}, "
              f"raw_attn={trrust_results['raw_attn'][li]:.4f}")

    # Correlation baseline (single value)
    abs_corr = np.abs(corr_matrix)
    # Build correlation matrix in mat_idx space for consistent comparison
    corr_mat_idx = np.zeros((n_hvg_tok, n_hvg_tok), dtype=np.float32)
    for mi in range(n_hvg_tok):
        gene_i = mat_idx_to_gene[mi]
        li_i = hvg_to_local.get(gene_i.upper())
        if li_i is None:
            continue
        for mj in range(n_hvg_tok):
            gene_j = mat_idx_to_gene[mj]
            li_j = hvg_to_local.get(gene_j.upper())
            if li_j is None:
                continue
            corr_mat_idx[mi, mj] = abs_corr[li_i, li_j]

    corr_trrust = trrust_auroc(corr_mat_idx, trrust_pairs, hvg_genes, hvg_to_local,
                                mat_idx_to_gene, token_to_mat_idx, hvg_to_token, n_hvg_tok,
                                symmetric=True)
    print(f"\n  Correlation TRRUST AUROC: {corr_trrust:.4f}")
    print(f"  Best vw_cos: L{max(trrust_results['vw_cos'], key=trrust_results['vw_cos'].get)} = "
          f"{max(trrust_results['vw_cos'].values()):.4f}")
    print(f"  Best raw_attn: L{max(trrust_results['raw_attn'], key=trrust_results['raw_attn'].get)} = "
          f"{max(trrust_results['raw_attn'].values()):.4f}")
    print(f"  TRRUST eval time: {time.time()-t4:.1f}s")

    # Save intermediate
    with open(os.path.join(OUT_DIR, "value_weighted_trrust.json"), 'w') as f:
        json.dump({
            'trrust_results': trrust_results,
            'correlation_trrust': round(corr_trrust, 4),
        }, f, indent=2, default=_json_default)

    # =================================================================
    # STEP 8: Evaluate - Perturbation-first AUROC
    # =================================================================
    print("\nSTEP 8: Perturbation-first AUROC evaluation...")
    t5 = time.time()

    pert_results = {'vw_cos': {}, 'vw_centroid': {}, 'vw_dot': {}, 'raw_attn': {}}

    for li in range(N_LAYERS):
        # Value-weighted cosine
        vw_cos_mat = np.load(os.path.join(vw_edges_dir, f"layer_{li:02d}_vw_cos.npy"))
        aurocs, _ = perturbation_first_auroc(
            vw_cos_mat, de_labels, hvg_genes, hvg_to_local,
            mat_idx_to_gene, token_to_mat_idx, hvg_to_token, symmetric=True)
        pert_results['vw_cos'][li] = {
            'mean': round(float(aurocs.mean()), 4),
            'std': round(float(aurocs.std()), 4),
            'n': len(aurocs),
        }

        # Value-weighted centroid
        vw_cent_mat = np.load(os.path.join(vw_edges_dir, f"layer_{li:02d}_vw_centroid.npy"))
        aurocs_cent, _ = perturbation_first_auroc(
            vw_cent_mat, de_labels, hvg_genes, hvg_to_local,
            mat_idx_to_gene, token_to_mat_idx, hvg_to_token, symmetric=True)
        pert_results['vw_centroid'][li] = {
            'mean': round(float(aurocs_cent.mean()), 4),
            'std': round(float(aurocs_cent.std()), 4),
            'n': len(aurocs_cent),
        }

        # Value-weighted dot product
        vw_dot_mat = np.load(os.path.join(vw_edges_dir, f"layer_{li:02d}_vw_dot.npy"))
        aurocs_dot, _ = perturbation_first_auroc(
            vw_dot_mat, de_labels, hvg_genes, hvg_to_local,
            mat_idx_to_gene, token_to_mat_idx, hvg_to_token, symmetric=True)
        pert_results['vw_dot'][li] = {
            'mean': round(float(aurocs_dot.mean()), 4),
            'std': round(float(aurocs_dot.std()), 4),
            'n': len(aurocs_dot),
        }

        # Raw attention
        aurocs_raw, pert_names_raw = perturbation_first_auroc(
            raw_attn_matrices[li], de_labels, hvg_genes, hvg_to_local,
            mat_idx_to_gene, token_to_mat_idx, hvg_to_token, symmetric=False)
        pert_results['raw_attn'][li] = {
            'mean': round(float(aurocs_raw.mean()), 4),
            'std': round(float(aurocs_raw.std()), 4),
            'n': len(aurocs_raw),
        }

        print(f"    L{li:2d}: vw_cos={pert_results['vw_cos'][li]['mean']:.4f} (n={pert_results['vw_cos'][li]['n']}), "
              f"vw_cent={pert_results['vw_centroid'][li]['mean']:.4f}, "
              f"vw_dot={pert_results['vw_dot'][li]['mean']:.4f}, "
              f"raw_attn={pert_results['raw_attn'][li]['mean']:.4f}")

    # Correlation perturbation-first
    corr_aurocs, corr_pert_names = perturbation_first_auroc(
        corr_mat_idx, de_labels, hvg_genes, hvg_to_local,
        mat_idx_to_gene, token_to_mat_idx, hvg_to_token, symmetric=True)
    corr_pert_result = {
        'mean': round(float(corr_aurocs.mean()), 4),
        'std': round(float(corr_aurocs.std()), 4),
        'n': len(corr_aurocs),
    }
    print(f"\n  Correlation pert-first: mean={corr_pert_result['mean']:.4f} (n={corr_pert_result['n']})")

    # Trivial baselines
    print("\n  Trivial baselines (gene-level features)...")
    baseline_results = {}
    gene_feat_lookup = {}
    for _, row in gene_features_df.iterrows():
        gene_feat_lookup[row['gene_upper']] = {
            'variance': row['variance'],
            'mean_expr': row['mean_expr'],
            'dropout_rate': row['dropout_rate'],
        }

    for feat_name in ['variance', 'mean_expr', 'dropout_rate']:
        feat_aurocs = []
        for pg, targets in de_labels.items():
            labels = []
            scores = []
            for tgt_gene, is_de in targets.items():
                feats = gene_feat_lookup.get(tgt_gene)
                if feats is None:
                    continue
                labels.append(int(is_de))
                if feat_name == 'dropout_rate':
                    scores.append(-feats[feat_name])  # lower dropout = more expressed
                else:
                    scores.append(feats[feat_name])
            labels = np.array(labels)
            scores = np.array(scores)
            if labels.sum() > 0 and labels.sum() < len(labels):
                try:
                    feat_aurocs.append(roc_auc_score(labels, scores))
                except ValueError:
                    pass
        feat_aurocs = np.array(feat_aurocs)
        baseline_results[feat_name] = {
            'mean': round(float(feat_aurocs.mean()), 4),
            'std': round(float(feat_aurocs.std()), 4),
            'n': len(feat_aurocs),
        }
        print(f"    {feat_name}: mean={baseline_results[feat_name]['mean']:.4f}")

    print(f"  Pert-first eval time: {time.time()-t5:.1f}s")

    # =================================================================
    # STEP 9: Statistical tests
    # =================================================================
    print("\nSTEP 9: Statistical tests (best layer vs. baselines)...")
    t6 = time.time()

    # Find best layer for each metric
    best_vw_cos_layer = max(pert_results['vw_cos'], key=lambda li: pert_results['vw_cos'][li]['mean'])
    best_raw_layer = max(pert_results['raw_attn'], key=lambda li: pert_results['raw_attn'][li]['mean'])

    # Paired comparison: vw_cos_best vs raw_attn_best (paired over perturbations)
    vw_cos_best_mat = np.load(os.path.join(vw_edges_dir, f"layer_{best_vw_cos_layer:02d}_vw_cos.npy"))
    raw_best_mat = raw_attn_matrices[best_raw_layer]

    # Compute per-perturbation AUROCs for paired tests
    paired_vw = []
    paired_raw = []
    paired_corr = []
    paired_pert_names = []

    for pg, targets in de_labels.items():
        pg_local = hvg_to_local.get(pg.upper())
        if pg_local is None:
            continue
        pg_orig = hvg_genes[pg_local]
        pg_tok = hvg_to_token.get(pg_orig)
        if pg_tok is None or pg_tok not in token_to_mat_idx:
            continue
        pg_mat = token_to_mat_idx[pg_tok]

        labels = []
        vw_scores = []
        raw_scores = []
        c_scores = []

        for tgt_gene, is_de in targets.items():
            tgt_local = hvg_to_local.get(tgt_gene)
            if tgt_local is None:
                continue
            tgt_orig = hvg_genes[tgt_local]
            tgt_tok = hvg_to_token.get(tgt_orig)
            if tgt_tok is None or tgt_tok not in token_to_mat_idx:
                continue
            tgt_mat = token_to_mat_idx[tgt_tok]

            labels.append(int(is_de))
            vw_scores.append(float(vw_cos_best_mat[pg_mat, tgt_mat]))
            raw_scores.append(max(float(raw_best_mat[pg_mat, tgt_mat]),
                                  float(raw_best_mat[tgt_mat, pg_mat])))
            c_scores.append(float(corr_mat_idx[pg_mat, tgt_mat]))

        labels = np.array(labels)
        if labels.sum() > 0 and labels.sum() < len(labels):
            try:
                paired_vw.append(roc_auc_score(labels, vw_scores))
                paired_raw.append(roc_auc_score(labels, raw_scores))
                paired_corr.append(roc_auc_score(labels, c_scores))
                paired_pert_names.append(pg)
            except ValueError:
                pass

    paired_vw = np.array(paired_vw)
    paired_raw = np.array(paired_raw)
    paired_corr = np.array(paired_corr)

    stat_tests = {}

    # VW cos vs Raw attention (paired Wilcoxon)
    if len(paired_vw) > 10:
        stat_w, p_w = stats.wilcoxon(paired_vw, paired_raw, alternative='two-sided')
        d_vw_raw = (paired_vw.mean() - paired_raw.mean()) / paired_vw.std()
        stat_tests['vw_cos_vs_raw_attn'] = {
            'wilcoxon_stat': round(float(stat_w), 2),
            'p_value': float(p_w),
            'mean_diff': round(float(paired_vw.mean() - paired_raw.mean()), 4),
            'cohens_d': round(float(d_vw_raw), 4),
            'n_perturbations': len(paired_vw),
            'vw_cos_mean': round(float(paired_vw.mean()), 4),
            'raw_attn_mean': round(float(paired_raw.mean()), 4),
            'best_vw_layer': int(best_vw_cos_layer),
            'best_raw_layer': int(best_raw_layer),
        }
        print(f"  VW cos (L{best_vw_cos_layer}) vs Raw attn (L{best_raw_layer}): "
              f"Δ={paired_vw.mean()-paired_raw.mean():+.4f}, "
              f"p={p_w:.2e}, d={d_vw_raw:.3f}")

    # VW cos vs Correlation
    if len(paired_vw) > 10:
        stat_w, p_w = stats.wilcoxon(paired_vw, paired_corr, alternative='two-sided')
        d_vw_corr = (paired_vw.mean() - paired_corr.mean()) / paired_vw.std()
        stat_tests['vw_cos_vs_correlation'] = {
            'wilcoxon_stat': round(float(stat_w), 2),
            'p_value': float(p_w),
            'mean_diff': round(float(paired_vw.mean() - paired_corr.mean()), 4),
            'cohens_d': round(float(d_vw_corr), 4),
            'vw_cos_mean': round(float(paired_vw.mean()), 4),
            'corr_mean': round(float(paired_corr.mean()), 4),
        }
        print(f"  VW cos (L{best_vw_cos_layer}) vs Correlation: "
              f"Δ={paired_vw.mean()-paired_corr.mean():+.4f}, "
              f"p={p_w:.2e}, d={d_vw_corr:.3f}")

    # Raw attn vs Correlation (replication of paper result)
    if len(paired_raw) > 10:
        stat_w, p_w = stats.wilcoxon(paired_raw, paired_corr, alternative='two-sided')
        stat_tests['raw_attn_vs_correlation'] = {
            'wilcoxon_stat': round(float(stat_w), 2),
            'p_value': float(p_w),
            'mean_diff': round(float(paired_raw.mean() - paired_corr.mean()), 4),
        }
        print(f"  Raw attn vs Correlation: Δ={paired_raw.mean()-paired_corr.mean():+.4f}, p={p_w:.2e}")

    print(f"  Statistical tests time: {time.time()-t6:.1f}s")

    # =================================================================
    # STEP 10: Incremental value test
    # =================================================================
    print("\nSTEP 10: Incremental value test...")
    t7 = time.time()

    # Build dataset with value-weighted edge features
    rows = []
    for pg, targets in de_labels.items():
        pg_local = hvg_to_local.get(pg.upper())
        if pg_local is None:
            continue
        pg_orig = hvg_genes[pg_local]
        pg_tok = hvg_to_token.get(pg_orig)
        if pg_tok is None or pg_tok not in token_to_mat_idx:
            continue
        pg_mat = token_to_mat_idx[pg_tok]

        for tgt_gene, is_de in targets.items():
            tgt_local = hvg_to_local.get(tgt_gene)
            if tgt_local is None:
                continue
            tgt_orig = hvg_genes[tgt_local]
            tgt_tok = hvg_to_token.get(tgt_orig)
            if tgt_tok is None or tgt_tok not in token_to_mat_idx:
                continue
            tgt_mat = token_to_mat_idx[tgt_tok]

            feats = gene_feat_lookup.get(tgt_gene, {})
            rows.append({
                'perturbation': pg,
                'target': tgt_gene,
                'is_de': int(is_de),
                'vw_cos_edge': float(vw_cos_best_mat[pg_mat, tgt_mat]),
                'raw_attn_edge': max(float(raw_best_mat[pg_mat, tgt_mat]),
                                     float(raw_best_mat[tgt_mat, pg_mat])),
                'corr_edge': float(corr_mat_idx[pg_mat, tgt_mat]),
                'tgt_mean_expr': feats.get('mean_expr', 0),
                'tgt_variance': feats.get('variance', 0),
                'tgt_dropout': feats.get('dropout_rate', 0),
            })

    df = pd.DataFrame(rows)
    print(f"  Dataset: {len(df)} rows, {df['perturbation'].nunique()} perturbations, "
          f"{df['is_de'].sum()} positive ({100*df['is_de'].mean():.1f}%)")

    gene_features_cols = ['tgt_mean_expr', 'tgt_variance', 'tgt_dropout']
    model_configs = {
        'gene_only': gene_features_cols,
        'gene_plus_vw': gene_features_cols + ['vw_cos_edge'],
        'gene_plus_attn': gene_features_cols + ['raw_attn_edge'],
        'gene_plus_corr': gene_features_cols + ['corr_edge'],
        'vw_only': ['vw_cos_edge'],
        'attn_only': ['raw_attn_edge'],
        'corr_only': ['corr_edge'],
    }

    incr_results = {}
    for name, feat_cols in model_configs.items():
        X = df[feat_cols].values
        y = df['is_de'].values
        groups = df['perturbation'].values

        gkf = GroupKFold(n_splits=min(N_FOLDS, len(np.unique(groups))))
        aurocs = []
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                continue
            if y_te.sum() == 0 or y_te.sum() == len(y_te):
                continue
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
            lr.fit(X_tr_s, y_tr)
            probs = lr.predict_proba(X_te_s)[:, 1]
            aurocs.append(roc_auc_score(y_te, probs))

        aurocs = np.array(aurocs)
        incr_results[name] = {
            'mean_auroc': round(float(aurocs.mean()), 4),
            'std_auroc': round(float(aurocs.std()), 4),
        }
        print(f"    {name:20s}: AUROC = {aurocs.mean():.4f} +/- {aurocs.std():.4f}")

    delta_vw = incr_results['gene_plus_vw']['mean_auroc'] - incr_results['gene_only']['mean_auroc']
    delta_attn = incr_results['gene_plus_attn']['mean_auroc'] - incr_results['gene_only']['mean_auroc']
    delta_corr = incr_results['gene_plus_corr']['mean_auroc'] - incr_results['gene_only']['mean_auroc']
    print(f"\n  ΔAUROC (gene+vw - gene): {delta_vw:+.4f}")
    print(f"  ΔAUROC (gene+attn - gene): {delta_attn:+.4f}")
    print(f"  ΔAUROC (gene+corr - gene): {delta_corr:+.4f}")
    print(f"  Incremental value time: {time.time()-t7:.1f}s")

    # =================================================================
    # STEP 11: Save all results
    # =================================================================
    print("\nSTEP 11: Saving results...")

    total_time = time.time() - total_t0
    results = {
        'experiment': 'analysis_a_value_weighted_edges',
        'model': 'Geneformer-V2-316M',
        'n_cells': N_CTRL,
        'n_hvg_tok': n_hvg_tok,
        'trrust_results': trrust_results,
        'correlation_trrust': round(corr_trrust, 4),
        'perturbation_first_results': pert_results,
        'correlation_pert_first': corr_pert_result,
        'trivial_baselines': baseline_results,
        'statistical_tests': stat_tests,
        'incremental_value': incr_results,
        'incremental_deltas': {
            'vw_cos': round(delta_vw, 4),
            'raw_attn': round(delta_attn, 4),
            'correlation': round(delta_corr, 4),
        },
        'best_layers': {
            'vw_cos': int(best_vw_cos_layer),
            'raw_attn': int(best_raw_layer),
        },
        'total_time_seconds': round(total_time, 1),
    }

    with open(os.path.join(OUT_DIR, "value_weighted_results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)

    # Write human-readable summary
    summary_lines = [
        "# Analysis A: Value-Weighted Edge Extraction — Results Summary",
        "",
        "## Overview",
        f"Extracted value-weighted context layers (softmax(QK^T/sqrt(d)) * V) from "
        f"Geneformer V2-316M across {N_LAYERS} layers, {N_CTRL} K562 control cells.",
        f"Computed pairwise cosine similarity, dot product, and centroid-based edges.",
        "",
        "## Key Results",
        "",
        "### Perturbation-First AUROC (primary endpoint)",
        f"Best VW cosine layer: L{best_vw_cos_layer} = "
        f"{pert_results['vw_cos'][best_vw_cos_layer]['mean']:.4f}",
        f"Best raw attention layer: L{best_raw_layer} = "
        f"{pert_results['raw_attn'][best_raw_layer]['mean']:.4f}",
        f"Correlation baseline: {corr_pert_result['mean']:.4f}",
        "",
    ]
    if 'vw_cos_vs_raw_attn' in stat_tests:
        t = stat_tests['vw_cos_vs_raw_attn']
        summary_lines.extend([
            f"VW cos vs Raw attn: Δ={t['mean_diff']:+.4f}, p={t['p_value']:.2e}, d={t['cohens_d']:.3f}",
            f"VW cos vs Correlation: Δ={stat_tests['vw_cos_vs_correlation']['mean_diff']:+.4f}, "
            f"p={stat_tests['vw_cos_vs_correlation']['p_value']:.2e}",
            "",
        ])

    summary_lines.extend([
        "### Trivial Baselines",
        f"Variance: {baseline_results['variance']['mean']:.4f}",
        f"Mean expression: {baseline_results['mean_expr']['mean']:.4f}",
        f"Dropout rate: {baseline_results['dropout_rate']['mean']:.4f}",
        "",
        "### Incremental Value Test",
        f"Gene-only AUROC: {incr_results['gene_only']['mean_auroc']:.4f}",
        f"Gene + VW cos: {incr_results['gene_plus_vw']['mean_auroc']:.4f} "
        f"(Δ={delta_vw:+.4f})",
        f"Gene + Raw attn: {incr_results['gene_plus_attn']['mean_auroc']:.4f} "
        f"(Δ={delta_attn:+.4f})",
        f"Gene + Correlation: {incr_results['gene_plus_corr']['mean_auroc']:.4f} "
        f"(Δ={delta_corr:+.4f})",
        "",
        "### TRRUST AUROC",
    ])
    for li in range(N_LAYERS):
        summary_lines.append(
            f"  L{li:2d}: vw_cos={trrust_results['vw_cos'][li]:.4f}, "
            f"vw_cent={trrust_results['vw_centroid'][li]:.4f}, "
            f"raw_attn={trrust_results['raw_attn'][li]:.4f}"
        )
    summary_lines.extend([
        f"  Correlation: {corr_trrust:.4f}",
        "",
        "### Layer Profiles (Perturbation-First AUROC)",
    ])
    for li in range(N_LAYERS):
        summary_lines.append(
            f"  L{li:2d}: vw_cos={pert_results['vw_cos'][li]['mean']:.4f}, "
            f"raw_attn={pert_results['raw_attn'][li]['mean']:.4f}"
        )
    summary_lines.extend([
        "",
        f"Total runtime: {total_time/60:.1f} minutes",
    ])

    with open(os.path.join(OUT_DIR, "value_weighted_summary.md"), 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f"\n{'='*70}")
    print("ANALYSIS A COMPLETE")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == '__main__':
    run()
