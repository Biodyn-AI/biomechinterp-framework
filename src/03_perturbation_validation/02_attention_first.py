#!/usr/bin/env python3
"""
Attention-on-Perturbation-First Evaluation
=============================================
Addresses Round 8 Major Concern 1 (THE critical experiment):
  Directly evaluate Geneformer attention-derived edge scores on the Replogle
  CRISPRi perturbation-first endpoint, and compare with correlation-based edges
  under identical conditions.

Uses Geneformer V2-316M (18L×18H) on K562 control cells.
Evaluates layers {6, 13, 17} for layer comparison, with L13 as primary.
"""

import os
import sys
import gc
import json
import time
import pickle
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)  # Line-buffered stdout

import numpy as np
import pandas as pd
import h5py
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
DATA_PATH = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data", "replogle_concat.h5ad")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "attention_perturbation_first_results")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Geneformer model and tokenization
MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"
# Local cache for tokenization dicts (downloaded in previous session)
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data")

# Parameters matching the primary perturbation-first parameterization
N_CTRL = 2000
N_HVG = 2000
LFC_THRESHOLD = 0.5
PVAL_THRESHOLD = 0.05
MIN_DE_TARGETS = 5

# Attention extraction parameters
EVAL_LAYERS = [5, 12, 17]  # 0-indexed: L6, L13, L18 in 1-indexed
PRIMARY_LAYER = 12  # L13 (0-indexed)
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1  # Memory-safe on MPS


def load_categorical_column(h5group, col_name):
    """Load a categorical column from h5ad obs/var."""
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
    """Vectorized Welch's t-test."""
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
    """Geneformer rank-value encoding: rank genes by median-normalized expression."""
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
    return np.concatenate([[2], ranked_tokens, [3]]).astype(np.int64)  # <cls> ... <eos>


def run():
    import torch
    from transformers import BertForMaskedLM

    total_t0 = time.time()

    print("=" * 70)
    print("ATTENTION-ON-PERTURBATION-FIRST EVALUATION")
    print("=" * 70)

    # ============================================================
    # STEP 1: Load data via h5py
    # ============================================================
    print("\nSTEP 1: Loading Replogle CRISPRi data...")
    t0 = time.time()

    with h5py.File(DATA_PATH, 'r') as f:
        cell_genes = load_categorical_column(f['obs'], 'gene')
        var_genes = load_categorical_column(f['var'], 'gene_name_index')
        n_cells_total, n_genes_total = f['X'].shape

    print(f"  Total: {n_cells_total:,} cells × {n_genes_total:,} genes")

    # Find control cells
    control_mask = np.zeros(n_cells_total, dtype=bool)
    for ctrl_name in ['non-targeting', 'Non-targeting', 'non_targeting']:
        control_mask |= (cell_genes == ctrl_name)
    ctrl_indices = np.where(control_mask)[0]
    print(f"  Control cells: {len(ctrl_indices)}")

    # Subsample controls
    np.random.seed(42)
    if len(ctrl_indices) > N_CTRL:
        ctrl_indices = np.random.choice(ctrl_indices, N_CTRL, replace=False)
        ctrl_indices.sort()
    print(f"  Using {len(ctrl_indices)} control cells")

    # Load control expression
    print("  Loading control expression matrix...")
    with h5py.File(DATA_PATH, 'r') as f:
        X_ctrl_full = np.empty((len(ctrl_indices), n_genes_total), dtype=np.float32)
        for ci, idx in enumerate(ctrl_indices):
            X_ctrl_full[ci, :] = f['X'][int(idx), :]
            if (ci + 1) % 500 == 0:
                print(f"    ... {ci+1}/{len(ctrl_indices)} cells loaded")

    # Normalize (log1p of counts/10000)
    print("  Normalizing...")
    row_sums = X_ctrl_full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_ctrl_norm = np.log1p(X_ctrl_full / row_sums * 1e4)

    # Select HVGs
    print(f"  Selecting {N_HVG} HVGs...")
    gene_var = X_ctrl_norm.var(axis=0)
    hvg_idx = np.argsort(-gene_var)[:N_HVG]
    hvg_idx.sort()
    hvg_genes = [var_genes[i] for i in hvg_idx]
    hvg_upper = [g.upper() for g in hvg_genes]
    hvg_to_local = {g.upper(): i for i, g in enumerate(hvg_genes)}
    hvg_to_full = {i: fi for i, fi in enumerate(hvg_idx)}

    X_ctrl_hvg = X_ctrl_norm[:, hvg_idx]
    print(f"  HVG matrix: {X_ctrl_hvg.shape}")
    print(f"  Step 1 time: {time.time()-t0:.1f}s")

    # ============================================================
    # STEP 2: Load Geneformer tokenization
    # ============================================================
    print("\nSTEP 2: Loading Geneformer gene mappings...")

    # Try local paths first, then download
    token_dict_path = os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl")
    gene_median_path = os.path.join(TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl")
    gene_name_id_path = os.path.join(TOKEN_DICTS_DIR, "gene_name_id_dict_gc104M.pkl")

    if not os.path.exists(token_dict_path):
        print("  Downloading tokenization dicts from HuggingFace...")
        from huggingface_hub import hf_hub_download
        for fname in ["token_dictionary_gc104M.pkl", "gene_median_dictionary_gc104M.pkl",
                       "gene_name_id_dict_gc104M.pkl"]:
            hf_hub_download(repo_id="ctheodoris/Geneformer", filename=f"geneformer/{fname}",
                          local_dir=TOKEN_DICTS_DIR)
            # Move from subdir
            src = os.path.join(TOKEN_DICTS_DIR, "geneformer", fname)
            dst = os.path.join(TOKEN_DICTS_DIR, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                os.rename(src, dst)

    with open(token_dict_path, 'rb') as f:
        token_dict = pickle.load(f)
    with open(gene_median_path, 'rb') as f:
        gene_median_dict = pickle.load(f)
    with open(gene_name_id_path, 'rb') as f:
        gene_name_id_dict = pickle.load(f)

    print(f"  Token dict: {len(token_dict)}, Median dict: {len(gene_median_dict)}, Name->ID: {len(gene_name_id_dict)}")

    # Map var genes to Ensembl IDs and token IDs
    mapped_var_indices = []
    mapped_token_ids_list = []
    mapped_medians_list = []
    mapped_gene_names_list = []

    for i in range(n_genes_total):
        gene_name = var_genes[i]
        ensembl = gene_name_id_dict.get(gene_name)
        if ensembl and ensembl in token_dict:
            median = gene_median_dict.get(ensembl, 1.0)
            mapped_var_indices.append(i)
            mapped_token_ids_list.append(token_dict[ensembl])
            mapped_medians_list.append(median)
            mapped_gene_names_list.append(gene_name)

    mapped_var_indices = np.array(mapped_var_indices)
    mapped_token_ids = np.array(mapped_token_ids_list)
    mapped_medians = np.array(mapped_medians_list)
    token_to_genename = {tid: gn for tid, gn in zip(mapped_token_ids_list, mapped_gene_names_list)}

    # Map HVG genes to token IDs
    hvg_to_token = {}
    for g in hvg_genes:
        ens = gene_name_id_dict.get(g)
        if ens and ens in token_dict:
            hvg_to_token[g] = token_dict[ens]
    token_to_hvg = {v: k for k, v in hvg_to_token.items()}

    print(f"  Mapped var genes: {len(mapped_var_indices)}/{n_genes_total}")
    print(f"  HVG genes with tokens: {len(hvg_to_token)}/{N_HVG}")

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

    # ============================================================
    # STEP 4: Extract Geneformer attention
    # ============================================================
    print("\nSTEP 4: Extracting Geneformer attention...")
    t2 = time.time()

    # Determine device
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
    )
    model = model.to(device)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Hook ALL evaluation layers simultaneously for a single forward pass per cell
    layer_attention_edges = {}  # layer_idx -> {(gene, gene) -> mean_attn}

    print(f"\n  Hooking {len(EVAL_LAYERS)} layers simultaneously for single-pass extraction...")

    # Set up hooks for all layers at once
    captured_attentions = {}  # layer_idx -> attention tensor
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

    for layer_idx in EVAL_LAYERS:
        target_layer = model.bert.encoder.layer[layer_idx]
        orig_forwards[layer_idx] = target_layer.forward
        target_layer.forward = make_patched_forward(layer_idx, captured_attentions, orig_forwards[layer_idx])

    # Per-layer accumulators
    attn_accumulators = {l: {} for l in EVAL_LAYERS}
    cells_processed = 0

    # Sort by length for efficiency
    sorted_indices = sorted(range(len(control_tokens)), key=lambda i: len(control_tokens[i]))

    for batch_start in range(0, len(control_tokens), BATCH_SIZE):
        batch_idx_list = sorted_indices[batch_start:batch_start + BATCH_SIZE]
        batch_tokens = [control_tokens[i] for i in batch_idx_list]

        # Truncate
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

        # Process captured attention from all layers
        for layer_idx in EVAL_LAYERS:
            if layer_idx not in captured_attentions:
                continue
            layer_attn = captured_attentions[layer_idx].mean(dim=1).cpu().float().numpy()

            for cell_i in range(bs):
                seq_len = int(attention_mask[cell_i].sum())
                cell_tok = input_ids[cell_i, :seq_len]
                cell_attn = layer_attn[cell_i, :seq_len, :seq_len]

                # Gene tokens start after <cls> (position 1) and end before <eos>
                gene_tokens = cell_tok[1:seq_len - 1]

                # Find positions of HVG gene tokens
                hvg_positions = []
                for pos_offset, tok_id in enumerate(gene_tokens):
                    tok_id = int(tok_id)
                    gene = token_to_hvg.get(tok_id)
                    if gene is not None:
                        hvg_positions.append((pos_offset + 1, gene))

                # Extract pairwise attention between HVG genes
                for src_pos, src_gene in hvg_positions:
                    for tgt_pos, tgt_gene in hvg_positions:
                        if src_gene == tgt_gene:
                            continue
                        attn_val = float(cell_attn[src_pos, tgt_pos])
                        key = (src_gene, tgt_gene)
                        entry = attn_accumulators[layer_idx].get(key)
                        if entry:
                            entry[0] += attn_val
                            entry[1] += 1
                        else:
                            attn_accumulators[layer_idx][key] = [attn_val, 1]

            del layer_attn

        cells_processed += bs
        captured_attentions.clear()
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()

        if cells_processed % 200 == 0 or cells_processed >= len(control_tokens):
            elapsed = time.time() - t2
            rate = cells_processed / elapsed if elapsed > 0 else 0
            eta = (len(control_tokens) - cells_processed) / rate if rate > 0 else 0
            print(f"    Cells: {cells_processed}/{len(control_tokens)}, "
                  f"time: {elapsed:.0f}s, rate: {rate:.1f} cells/s, ETA: {eta:.0f}s")

    # Average attention edges for each layer
    for layer_idx in EVAL_LAYERS:
        attention_edges = {}
        for (src, tgt), (asum, cnt) in attn_accumulators[layer_idx].items():
            attention_edges[(src, tgt)] = asum / cnt
        layer_attention_edges[layer_idx] = attention_edges
        print(f"    Layer {layer_idx+1}: {len(attention_edges)} edges from {cells_processed} cells")

    # Restore original forwards
    for layer_idx in EVAL_LAYERS:
        model.bert.encoder.layer[layer_idx].forward = orig_forwards[layer_idx]
    del attn_accumulators
    gc.collect()

    print(f"\n  Total attention extraction time: {time.time()-t2:.1f}s")

    # Free model
    del model
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()

    # ============================================================
    # STEP 5: Compute correlation edges
    # ============================================================
    print("\nSTEP 5: Computing correlation edges from control cells...")
    t3 = time.time()

    ranks = np.apply_along_axis(rankdata, 0, X_ctrl_hvg)
    corr_matrix = np.corrcoef(ranks.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    print(f"  Correlation matrix: {corr_matrix.shape}")
    print(f"  Time: {time.time()-t3:.1f}s")

    # ============================================================
    # STEP 6: Compute DE for each perturbation
    # ============================================================
    print("\nSTEP 6: Computing DE for evaluable perturbations...")
    t4 = time.time()

    # Find all perturbation genes that are in HVG
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

    # Compute DE for each perturbation
    de_targets = {}  # gene -> set of DE target genes (uppercase)
    de_all_genes = {}  # gene -> {target_gene: is_de} for AUROC computation

    for pi, pg in enumerate(evaluable_perts):
        pidx = pert_cell_indices[pg]

        # Load perturbed expression
        with h5py.File(DATA_PATH, 'r') as f:
            X_pert_full = np.empty((len(pidx), n_genes_total), dtype=np.float32)
            for ci, idx in enumerate(pidx):
                X_pert_full[ci, :] = f['X'][int(idx), :]

        # Normalize
        row_sums_p = X_pert_full.sum(axis=1, keepdims=True)
        row_sums_p[row_sums_p == 0] = 1
        X_pert_norm = np.log1p(X_pert_full / row_sums_p * 1e4)
        X_pert_hvg = X_pert_norm[:, hvg_idx]

        # Welch's t-test
        pvals, lfcs = call_de_vectorized(X_ctrl_hvg, X_pert_hvg)
        reject, adj_pvals, _, _ = multipletests(pvals, alpha=PVAL_THRESHOLD, method='fdr_bh')

        # Identify DE targets
        targets = set()
        all_genes_dict = {}
        pg_local_idx = hvg_to_local.get(pg.upper())

        for j in range(N_HVG):
            if pg_local_idx is not None and j == pg_local_idx:
                continue  # skip self
            is_de = bool(reject[j]) and abs(lfcs[j]) >= LFC_THRESHOLD
            gene_name = hvg_upper[j]
            all_genes_dict[gene_name] = is_de
            if is_de:
                targets.add(gene_name)

        if len(targets) >= MIN_DE_TARGETS:
            de_targets[pg] = targets
            de_all_genes[pg] = all_genes_dict

        del X_pert_full, X_pert_norm, X_pert_hvg
        gc.collect()

        if (pi + 1) % 100 == 0:
            print(f"    ... {pi+1}/{len(evaluable_perts)} perturbations, "
                  f"{len(de_targets)} evaluable so far")

    print(f"  Evaluable perturbations with >= {MIN_DE_TARGETS} DE targets: {len(de_targets)}")
    print(f"  DE computation time: {time.time()-t4:.1f}s")

    # ============================================================
    # STEP 7: Evaluate attention vs correlation vs random
    # ============================================================
    print("\nSTEP 7: Evaluating predictors...")
    t5 = time.time()

    results_by_layer = {}

    for layer_idx in EVAL_LAYERS:
        attn_edges = layer_attention_edges[layer_idx]

        layer_aurocs_attn = []
        layer_aurocs_corr = []
        layer_aurocs_rand = []
        per_gene_results = []
        rng = np.random.RandomState(42)

        for pg in de_targets:
            all_genes_dict = de_all_genes[pg]
            pg_local = hvg_to_local.get(pg.upper())
            if pg_local is None:
                continue

            labels = []
            scores_attn = []
            scores_corr = []
            scores_rand = []

            for tgt_gene, is_de in all_genes_dict.items():
                tgt_local = hvg_to_local.get(tgt_gene)
                if tgt_local is None:
                    continue

                # Attention edge: max of (pg→tgt) and (tgt→pg)
                fwd = attn_edges.get((pg, hvg_genes[tgt_local]), 0.0)
                rev = attn_edges.get((hvg_genes[tgt_local], pg), 0.0)
                attn_score = max(fwd, rev)

                # Correlation edge
                corr_score = abs(corr_matrix[pg_local, tgt_local])

                labels.append(1 if is_de else 0)
                scores_attn.append(attn_score)
                scores_corr.append(corr_score)
                scores_rand.append(rng.random())

            labels = np.array(labels)
            if labels.sum() < MIN_DE_TARGETS or labels.sum() == len(labels):
                continue

            auroc_attn = roc_auc_score(labels, np.array(scores_attn))
            auroc_corr = roc_auc_score(labels, np.array(scores_corr))
            auroc_rand = roc_auc_score(labels, np.array(scores_rand))

            layer_aurocs_attn.append(auroc_attn)
            layer_aurocs_corr.append(auroc_corr)
            layer_aurocs_rand.append(auroc_rand)
            per_gene_results.append({
                'gene': pg,
                'auroc_attn': round(auroc_attn, 4),
                'auroc_corr': round(auroc_corr, 4),
                'auroc_rand': round(auroc_rand, 4),
                'n_de': int(labels.sum()),
                'n_total': int(len(labels)),
            })

        aurocs_attn = np.array(layer_aurocs_attn)
        aurocs_corr = np.array(layer_aurocs_corr)
        aurocs_rand = np.array(layer_aurocs_rand)

        results_by_layer[layer_idx] = {
            'layer_1indexed': layer_idx + 1,
            'n_evaluated': len(aurocs_attn),
            'attention': {
                'mean_auroc': round(float(aurocs_attn.mean()), 4) if len(aurocs_attn) > 0 else None,
                'std_auroc': round(float(aurocs_attn.std()), 4) if len(aurocs_attn) > 0 else None,
                'median_auroc': round(float(np.median(aurocs_attn)), 4) if len(aurocs_attn) > 0 else None,
                'iqr': [round(float(np.percentile(aurocs_attn, 25)), 4),
                        round(float(np.percentile(aurocs_attn, 75)), 4)] if len(aurocs_attn) > 0 else None,
            },
            'correlation': {
                'mean_auroc': round(float(aurocs_corr.mean()), 4) if len(aurocs_corr) > 0 else None,
                'std_auroc': round(float(aurocs_corr.std()), 4) if len(aurocs_corr) > 0 else None,
                'median_auroc': round(float(np.median(aurocs_corr)), 4) if len(aurocs_corr) > 0 else None,
            },
            'random': {
                'mean_auroc': round(float(aurocs_rand.mean()), 4) if len(aurocs_rand) > 0 else None,
            },
            'per_gene': per_gene_results,
        }

        # Paired test: attention vs correlation
        if len(aurocs_attn) > 1:
            t_stat, t_p = stats.ttest_rel(aurocs_attn, aurocs_corr)
            w_stat, w_p = stats.wilcoxon(aurocs_attn - aurocs_corr)
            results_by_layer[layer_idx]['attn_vs_corr'] = {
                'paired_t_p': round(float(t_p), 6),
                'wilcoxon_p': round(float(w_p), 6),
                'mean_diff': round(float((aurocs_attn - aurocs_corr).mean()), 4),
                'n_attn_better': int((aurocs_attn > aurocs_corr).sum()),
                'n_corr_better': int((aurocs_corr > aurocs_attn).sum()),
            }

        print(f"\n  Layer {layer_idx+1}:")
        print(f"    Evaluated: {len(aurocs_attn)} perturbations")
        if len(aurocs_attn) > 0:
            print(f"    Attention AUROC:   {aurocs_attn.mean():.4f} ± {aurocs_attn.std():.4f}")
            print(f"    Correlation AUROC: {aurocs_corr.mean():.4f} ± {aurocs_corr.std():.4f}")
            print(f"    Random AUROC:      {aurocs_rand.mean():.4f} ± {aurocs_rand.std():.4f}")
            if 'attn_vs_corr' in results_by_layer[layer_idx]:
                avc = results_by_layer[layer_idx]['attn_vs_corr']
                print(f"    Attn vs Corr: mean diff={avc['mean_diff']:.4f}, "
                      f"paired-t p={avc['paired_t_p']:.4f}, Wilcoxon p={avc['wilcoxon_p']:.4f}")
                print(f"    Attn better: {avc['n_attn_better']}/{len(aurocs_attn)}, "
                      f"Corr better: {avc['n_corr_better']}/{len(aurocs_attn)}")

    print(f"\n  Evaluation time: {time.time()-t5:.1f}s")

    # ============================================================
    # STEP 8: Generate figures
    # ============================================================
    print("\nSTEP 8: Generating figures...")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Layer comparison (mean AUROC)
    ax = axes[0]
    layers_1idx = [l + 1 for l in EVAL_LAYERS]
    attn_means = [results_by_layer[l]['attention']['mean_auroc'] or 0 for l in EVAL_LAYERS]
    attn_stds = [results_by_layer[l]['attention']['std_auroc'] or 0 for l in EVAL_LAYERS]
    corr_means = [results_by_layer[l]['correlation']['mean_auroc'] or 0 for l in EVAL_LAYERS]
    corr_stds = [results_by_layer[l]['correlation']['std_auroc'] or 0 for l in EVAL_LAYERS]
    rand_means = [results_by_layer[l]['random']['mean_auroc'] or 0 for l in EVAL_LAYERS]

    x = np.arange(len(EVAL_LAYERS))
    width = 0.3
    ax.bar(x - width/2, attn_means, width, yerr=attn_stds, label='Attention',
           color='#2196F3', alpha=0.8, capsize=3)
    ax.bar(x + width/2, corr_means, width, yerr=corr_stds, label='Correlation',
           color='#FF9800', alpha=0.8, capsize=3)
    ax.scatter(x, rand_means, color='gray', zorder=5, marker='_', s=200, label='Random')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers_1idx])
    ax.set_ylabel('Mean AUROC vs DE targets')
    ax.set_title('(A) Layer comparison:\nattention vs correlation')
    ax.legend(fontsize=8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

    # Panel B: Per-perturbation scatter (primary layer)
    ax = axes[1]
    primary_results = results_by_layer[PRIMARY_LAYER]
    if primary_results['per_gene']:
        pg_df = pd.DataFrame(primary_results['per_gene'])
        ax.scatter(pg_df['auroc_corr'], pg_df['auroc_attn'], alpha=0.4, s=20, color='#2196F3')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel(f'Correlation AUROC')
        ax.set_ylabel(f'Attention (L{PRIMARY_LAYER+1}) AUROC')
        ax.set_title(f'(B) Per-perturbation AUROC\n(n={len(pg_df)})')
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0.3, 1.0)
        ax.set_aspect('equal')

    # Panel C: AUROC distribution (primary layer)
    ax = axes[2]
    if primary_results['per_gene']:
        pg_df = pd.DataFrame(primary_results['per_gene'])
        bins = np.linspace(0.3, 1.0, 30)
        ax.hist(pg_df['auroc_attn'], bins=bins, alpha=0.6, color='#2196F3',
                label=f'Attention (mean={pg_df["auroc_attn"].mean():.3f})', density=True)
        ax.hist(pg_df['auroc_corr'], bins=bins, alpha=0.6, color='#FF9800',
                label=f'Correlation (mean={pg_df["auroc_corr"].mean():.3f})', density=True)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('AUROC')
        ax.set_ylabel('Density')
        ax.set_title(f'(C) AUROC distribution (L{PRIMARY_LAYER+1})\nvs DE targets')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_attention_perturbation_first.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Figure saved to {fig_path}")

    # ============================================================
    # STEP 9: Save results
    # ============================================================
    print("\nSTEP 9: Saving results...")

    # Convert results for JSON serialization
    save_results = {
        'experiment': 'attention_perturbation_first',
        'config': {
            'model': 'Geneformer-V2-316M',
            'n_control_cells': N_CTRL,
            'n_hvg': N_HVG,
            'lfc_threshold': LFC_THRESHOLD,
            'pval_threshold': PVAL_THRESHOLD,
            'min_de_targets': MIN_DE_TARGETS,
            'eval_layers_0indexed': EVAL_LAYERS,
            'primary_layer_0indexed': PRIMARY_LAYER,
            'max_seq_len': MAX_SEQ_LEN,
            'n_evaluable_perturbations': len(de_targets),
        },
        'results_by_layer': {},
    }

    for layer_idx in EVAL_LAYERS:
        r = results_by_layer[layer_idx]
        save_results['results_by_layer'][str(layer_idx)] = {
            'layer_1indexed': r['layer_1indexed'],
            'n_evaluated': r['n_evaluated'],
            'attention': r['attention'],
            'correlation': r['correlation'],
            'random': r['random'],
            'attn_vs_corr': r.get('attn_vs_corr'),
        }

    with open(os.path.join(OUT_DIR, "attention_perturbation_first_summary.json"), 'w') as f:
        json.dump(save_results, f, indent=2)

    # Save per-gene results for primary layer
    if primary_results['per_gene']:
        pg_df = pd.DataFrame(primary_results['per_gene'])
        pg_df.to_csv(os.path.join(OUT_DIR, "per_gene_results_L13.csv"), index=False)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE - Total: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")

    # Print headline result
    pr = results_by_layer[PRIMARY_LAYER]
    print(f"\nHEADLINE (L{PRIMARY_LAYER+1}, primary):")
    print(f"  Attention mean AUROC: {pr['attention']['mean_auroc']}")
    print(f"  Correlation mean AUROC: {pr['correlation']['mean_auroc']}")
    if pr.get('attn_vs_corr'):
        print(f"  Difference: {pr['attn_vs_corr']['mean_diff']}")
        print(f"  Wilcoxon p: {pr['attn_vs_corr']['wilcoxon_p']}")


if __name__ == '__main__':
    run()
