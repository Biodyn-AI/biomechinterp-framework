#!/usr/bin/env python3
"""
Round 11 Experiment: Adamson CRISPRa Layer 15 Replication
=========================================================
Fork of round9_adamson_cross_context.py, focused on Layer 15 (0-indexed 14)
to replicate the Replogle CRISPRi L15 attention advantage finding.

Adds:
  - Cohen's d for attention vs correlation AUROC difference
  - 95% bootstrap CI for the mean difference
  - Explicit replication verdict logic
  - Cross-dataset comparison with Replogle L15 nested CV results
  - Comparison with Adamson L6/L13/L18 results from round 9
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
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
DATA_PATH = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/data/perturb/adamson/perturb_processed_symbols.h5ad"
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round11_adamson_L15")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

N_CTRL = 2000
N_HVG = 2000
LFC_THRESHOLD = 0.5
PVAL_THRESHOLD = 0.05
MIN_DE_TARGETS = 5
LFC_LENIENT = 0.25

EVAL_LAYERS = [14]
PRIMARY_LAYER = 14
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1

# Replogle L15 nested CV reference values
REPLOGLE_L15_DELTA = 0.040
REPLOGLE_L15_D = 0.22
REPLOGLE_L15_P = 0.000933

CHECKPOINT_PATH = os.path.join(OUT_DIR, "checkpoint_attention.pkl")


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


def load_sparse_row(f, idx):
    """Load a single row from CSR sparse matrix in h5ad."""
    X = f['X']
    indptr = X['indptr']
    start = indptr[idx]
    end = indptr[idx + 1]
    indices = X['indices'][start:end]
    data = X['data'][start:end]
    n_cols = f['X'].attrs.get('shape', [0, 0])[1] if 'shape' in f['X'].attrs else len(f['var']['_index'])
    row = np.zeros(n_cols, dtype=np.float32)
    row[indices] = data
    return row


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


def run():
    import torch
    from transformers import BertForMaskedLM

    total_t0 = time.time()

    print("=" * 70)
    print("ADAMSON CRISPRa LAYER 15 REPLICATION")
    print("=" * 70)
    print(f"  Evaluating layer: L{PRIMARY_LAYER + 1} (0-indexed {PRIMARY_LAYER})")
    print(f"  Output directory: {OUT_DIR}")

    # ============================================================
    # STEP 1: Load Adamson data (sparse CSR format)
    # ============================================================
    print("\nSTEP 1: Loading Adamson CRISPRa data...")
    t0 = time.time()

    with h5py.File(DATA_PATH, 'r') as f:
        conditions = load_categorical_column(f['obs'], 'condition')
        n_cells_total = len(conditions)
        # var gene names
        gene_name_col = f['var']['gene_name']
        if isinstance(gene_name_col, h5py.Dataset):
            var_genes_raw = gene_name_col[:]
            if var_genes_raw.dtype.kind in ('O', 'S'):
                var_genes = np.array([x.decode() if isinstance(x, bytes) else x for x in var_genes_raw])
            else:
                var_genes = var_genes_raw
        n_genes_total = len(var_genes)

    print(f"  Total: {n_cells_total:,} cells x {n_genes_total:,} genes")

    # Find control cells (condition == 'ctrl')
    ctrl_indices = np.where(conditions == 'ctrl')[0]
    print(f"  Control cells: {len(ctrl_indices)}")

    np.random.seed(42)
    if len(ctrl_indices) > N_CTRL:
        ctrl_indices = np.random.choice(ctrl_indices, N_CTRL, replace=False)
        ctrl_indices.sort()
    print(f"  Using {len(ctrl_indices)} control cells")

    # Load control expression (sparse)
    print("  Loading control expression matrix (sparse)...")
    X_ctrl_full = np.zeros((len(ctrl_indices), n_genes_total), dtype=np.float32)
    with h5py.File(DATA_PATH, 'r') as f:
        for ci, idx in enumerate(ctrl_indices):
            X_ctrl_full[ci, :] = load_sparse_row(f, int(idx))
            if (ci + 1) % 500 == 0:
                print(f"    ... {ci+1}/{len(ctrl_indices)} cells loaded")

    # Normalize
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

    X_ctrl_hvg = X_ctrl_norm[:, hvg_idx]
    print(f"  HVG matrix: {X_ctrl_hvg.shape}")
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

    mapped_var_indices = []
    mapped_token_ids_list = []
    mapped_medians_list = []

    for i in range(n_genes_total):
        gene_name = var_genes[i]
        ensembl = gene_name_id_dict.get(gene_name)
        if ensembl and ensembl in token_dict:
            median = gene_median_dict.get(ensembl, 1.0)
            mapped_var_indices.append(i)
            mapped_token_ids_list.append(token_dict[ensembl])
            mapped_medians_list.append(median)

    mapped_var_indices = np.array(mapped_var_indices)
    mapped_token_ids = np.array(mapped_token_ids_list)
    mapped_medians = np.array(mapped_medians_list)

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
        for ci, idx in enumerate(ctrl_indices):
            expr = load_sparse_row(f, int(idx))
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
    # Check for checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print("\nSTEP 4: Loading attention from checkpoint...")
        with open(CHECKPOINT_PATH, 'rb') as f:
            ckpt = pickle.load(f)
        layer_attention_edges = ckpt['layer_attention_edges']
        cells_processed = ckpt['cells_processed']
        print(f"  Loaded checkpoint: {cells_processed} cells, "
              f"layer {PRIMARY_LAYER+1} has {len(layer_attention_edges.get(PRIMARY_LAYER, {}))} edges")
    else:
        print("\nSTEP 4: Extracting Geneformer attention...")
        t2 = time.time()

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"  Device: MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print(f"  Loading Geneformer V2-316M...")
        model = BertForMaskedLM.from_pretrained(
            MODEL_NAME, subfolder=MODEL_SUBFOLDER,
            output_attentions=False, output_hidden_states=False,
        )
        model = model.to(device)
        model.eval()
        print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

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

        print(f"  Hooking {len(EVAL_LAYERS)} layers...")
        for layer_idx in EVAL_LAYERS:
            target_layer = model.bert.encoder.layer[layer_idx]
            orig_forwards[layer_idx] = target_layer.forward
            target_layer.forward = make_patched_forward(layer_idx, captured_attentions, orig_forwards[layer_idx])

        attn_accumulators = {l: {} for l in EVAL_LAYERS}
        cells_processed = 0
        sorted_indices = sorted(range(len(control_tokens)), key=lambda i: len(control_tokens[i]))

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

            for layer_idx in EVAL_LAYERS:
                if layer_idx not in captured_attentions:
                    continue
                layer_attn = captured_attentions[layer_idx].mean(dim=1).cpu().float().numpy()

                for cell_i in range(bs):
                    seq_len = int(attention_mask[cell_i].sum())
                    cell_tok = input_ids[cell_i, :seq_len]
                    cell_attn = layer_attn[cell_i, :seq_len, :seq_len]
                    gene_tokens = cell_tok[1:seq_len - 1]

                    hvg_positions = []
                    for pos_offset, tok_id in enumerate(gene_tokens):
                        tok_id = int(tok_id)
                        gene = token_to_hvg.get(tok_id)
                        if gene is not None:
                            hvg_positions.append((pos_offset + 1, gene))

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

            if cells_processed % 200 == 0 or cells_processed >= len(control_tokens):
                elapsed = time.time() - t2
                rate = cells_processed / elapsed if elapsed > 0 else 0
                eta = (len(control_tokens) - cells_processed) / rate if rate > 0 else 0
                print(f"    Cells: {cells_processed}/{len(control_tokens)}, "
                      f"time: {elapsed:.0f}s, rate: {rate:.1f} cells/s, ETA: {eta:.0f}s")

        layer_attention_edges = {}
        for layer_idx in EVAL_LAYERS:
            edges = {}
            for (src, tgt), (asum, cnt) in attn_accumulators[layer_idx].items():
                edges[(src, tgt)] = asum / cnt
            layer_attention_edges[layer_idx] = edges
            print(f"    Layer {layer_idx+1}: {len(edges)} edges from {cells_processed} cells")

        for layer_idx in EVAL_LAYERS:
            model.bert.encoder.layer[layer_idx].forward = orig_forwards[layer_idx]
        del attn_accumulators, model
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()

        print(f"\n  Attention extraction time: {time.time()-t2:.1f}s")

        # Save checkpoint
        print("  Saving attention checkpoint...")
        with open(CHECKPOINT_PATH, 'wb') as f:
            pickle.dump({
                'layer_attention_edges': layer_attention_edges,
                'cells_processed': cells_processed,
            }, f)
        print(f"  Checkpoint saved to {CHECKPOINT_PATH}")

    # ============================================================
    # STEP 5: Compute correlation edges
    # ============================================================
    print("\nSTEP 5: Computing correlation edges...")
    ranks = np.apply_along_axis(rankdata, 0, X_ctrl_hvg)
    corr_matrix = np.corrcoef(ranks.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # ============================================================
    # STEP 6: Compute DE for Adamson perturbations
    # ============================================================
    print("\nSTEP 6: Computing DE for perturbations...")
    t4 = time.time()

    # Parse perturbation gene names from conditions
    # Format: "GENENAME+ctrl" -> perturbed gene is GENENAME
    unique_conds = np.unique(conditions)
    evaluable_perts = []
    pert_cell_indices = {}

    for cond in unique_conds:
        cond_str = str(cond)
        if cond_str == 'ctrl':
            continue
        # Extract gene name
        if '+' in cond_str:
            pg = cond_str.split('+')[0]
        else:
            pg = cond_str

        if pg.upper() not in hvg_to_local:
            continue
        pert_idx = np.where(conditions == cond)[0]
        if len(pert_idx) >= 10:
            evaluable_perts.append(pg)
            pert_cell_indices[pg] = pert_idx

    print(f"  Evaluable perturbations in HVG: {len(evaluable_perts)}")

    def compute_de(lfc_threshold):
        de_targets = {}
        de_all_genes = {}
        for pg in evaluable_perts:
            pidx = pert_cell_indices[pg]
            X_pert = np.zeros((len(pidx), n_genes_total), dtype=np.float32)
            with h5py.File(DATA_PATH, 'r') as f:
                for ci, idx in enumerate(pidx):
                    X_pert[ci, :] = load_sparse_row(f, int(idx))

            row_sums_p = X_pert.sum(axis=1, keepdims=True)
            row_sums_p[row_sums_p == 0] = 1
            X_pert_norm = np.log1p(X_pert / row_sums_p * 1e4)
            X_pert_hvg = X_pert_norm[:, hvg_idx]

            pvals, lfcs = call_de_vectorized(X_ctrl_hvg, X_pert_hvg)
            reject, _, _, _ = multipletests(pvals, alpha=PVAL_THRESHOLD, method='fdr_bh')

            targets = set()
            all_genes_dict = {}
            pg_local_idx = hvg_to_local.get(pg.upper())
            for j in range(N_HVG):
                if pg_local_idx is not None and j == pg_local_idx:
                    continue
                is_de = bool(reject[j]) and abs(lfcs[j]) >= lfc_threshold
                gene_name = hvg_upper[j]
                all_genes_dict[gene_name] = is_de
                if is_de:
                    targets.add(gene_name)

            if len(targets) >= MIN_DE_TARGETS:
                de_targets[pg] = targets
                de_all_genes[pg] = all_genes_dict

            del X_pert, X_pert_norm, X_pert_hvg
            gc.collect()

        return de_targets, de_all_genes

    # Try primary threshold
    de_targets, de_all_genes = compute_de(LFC_THRESHOLD)
    lfc_used = LFC_THRESHOLD
    print(f"  Evaluable (LFC>{LFC_THRESHOLD}): {len(de_targets)}")

    # Fallback to lenient if too few
    if len(de_targets) < 10:
        print(f"  Too few. Trying lenient LFC>{LFC_LENIENT}...")
        de_targets, de_all_genes = compute_de(LFC_LENIENT)
        lfc_used = LFC_LENIENT
        print(f"  Evaluable (LFC>{LFC_LENIENT}): {len(de_targets)}")

    print(f"  DE computation time: {time.time()-t4:.1f}s")

    # ============================================================
    # STEP 7: Evaluate attention vs correlation (L15 only)
    # ============================================================
    print("\nSTEP 7: Evaluating predictors at Layer 15...")

    results_by_layer = {}
    rng_eval = np.random.RandomState(42)

    for layer_idx in EVAL_LAYERS:
        attn_edges = layer_attention_edges[layer_idx]

        layer_aurocs_attn = []
        layer_aurocs_corr = []
        layer_aurocs_rand = []
        per_gene_results = []

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
                fwd = attn_edges.get((pg, hvg_genes[tgt_local]), 0.0)
                rev = attn_edges.get((hvg_genes[tgt_local], pg), 0.0)
                attn_score = max(fwd, rev)
                corr_score = abs(corr_matrix[pg_local, tgt_local])

                labels.append(1 if is_de else 0)
                scores_attn.append(attn_score)
                scores_corr.append(corr_score)
                scores_rand.append(rng_eval.random())

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
                'gene': pg, 'auroc_attn': round(auroc_attn, 4),
                'auroc_corr': round(auroc_corr, 4), 'auroc_rand': round(auroc_rand, 4),
                'n_de': int(labels.sum()), 'n_total': int(len(labels)),
            })

        aurocs_attn = np.array(layer_aurocs_attn)
        aurocs_corr = np.array(layer_aurocs_corr)

        results_by_layer[layer_idx] = {
            'layer_1indexed': layer_idx + 1,
            'n_evaluated': len(aurocs_attn),
            'attention': {
                'mean_auroc': round(float(aurocs_attn.mean()), 4) if len(aurocs_attn) > 0 else None,
                'std_auroc': round(float(aurocs_attn.std()), 4) if len(aurocs_attn) > 0 else None,
            },
            'correlation': {
                'mean_auroc': round(float(aurocs_corr.mean()), 4) if len(aurocs_corr) > 0 else None,
                'std_auroc': round(float(aurocs_corr.std()), 4) if len(aurocs_corr) > 0 else None,
            },
            'per_gene': per_gene_results,
        }

        # Statistical tests
        w_p = 1.0
        t_p = 1.0
        mean_diff = 0.0
        cohens_d = 0.0
        ci_95 = [0.0, 0.0]
        verdict = "NOT_REPLICATED"

        if len(aurocs_attn) > 1:
            diff = aurocs_attn - aurocs_corr
            mean_diff = float(diff.mean())

            try:
                w_stat, w_p = stats.wilcoxon(diff)
                w_p = float(w_p)
            except:
                w_p = 1.0
            try:
                t_stat, t_p = stats.ttest_rel(aurocs_attn, aurocs_corr)
                t_p = float(t_p)
            except:
                t_p = 1.0

            # Cohen's d
            cohens_d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0

            # Bootstrap CI for mean difference
            n_boot = 2000
            rng = np.random.RandomState(42)
            boot_diffs = []
            for _ in range(n_boot):
                idx = rng.choice(len(diff), size=len(diff), replace=True)
                boot_diffs.append(diff[idx].mean())
            boot_diffs = np.array(boot_diffs)
            ci_95 = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))]

            # Replication verdict
            if w_p < 0.05 and mean_diff > 0:
                verdict = "REPLICATED"
            elif w_p < 0.05 and mean_diff < 0:
                verdict = "REVERSED"
            else:
                verdict = "NOT_REPLICATED"

            results_by_layer[layer_idx]['attn_vs_corr'] = {
                'wilcoxon_p': round(w_p, 6),
                'paired_t_p': round(t_p, 6),
                'mean_diff': round(mean_diff, 4),
                'cohens_d': round(cohens_d, 4),
                'bootstrap_ci_95': [round(ci_95[0], 4), round(ci_95[1], 4)],
                'n_attn_better': int((aurocs_attn > aurocs_corr).sum()),
                'n_corr_better': int((aurocs_corr > aurocs_attn).sum()),
                'replication_verdict': verdict,
            }

        print(f"\n  Layer {layer_idx+1} (L15):")
        print(f"    Evaluated: {len(aurocs_attn)} perturbations")
        if len(aurocs_attn) > 0:
            print(f"    Attention AUROC:   {aurocs_attn.mean():.4f} +/- {aurocs_attn.std():.4f}")
            print(f"    Correlation AUROC: {aurocs_corr.mean():.4f} +/- {aurocs_corr.std():.4f}")
            if 'attn_vs_corr' in results_by_layer[layer_idx]:
                avc = results_by_layer[layer_idx]['attn_vs_corr']
                print(f"    Mean diff (attn - corr): {avc['mean_diff']:.4f}")
                print(f"    Cohen's d: {avc['cohens_d']:.4f}")
                print(f"    95% Bootstrap CI: [{avc['bootstrap_ci_95'][0]:.4f}, {avc['bootstrap_ci_95'][1]:.4f}]")
                print(f"    Wilcoxon p: {avc['wilcoxon_p']}")
                print(f"    Paired t-test p: {avc['paired_t_p']}")
                print(f"    Verdict: {avc['replication_verdict']}")

    # ============================================================
    # Report Replogle L15 reference results
    # ============================================================
    print("\n  --- Replogle CRISPRi L15 reference (nested CV) ---")
    print(f"    Delta (attn - corr): +{REPLOGLE_L15_DELTA:.3f}")
    print(f"    Cohen's d: {REPLOGLE_L15_D:.2f}")
    print(f"    Pooled Wilcoxon p: {REPLOGLE_L15_P:.6f}")

    # ============================================================
    # STEP 8: Save results and generate figure
    # ============================================================
    print("\nSTEP 8: Saving results...")

    # Load existing Adamson round9 results for multi-layer comparison
    round9_path = os.path.join(PAPER_DIR, "experiments", "round9_adamson_cross_context", "adamson_summary.json")
    round9_results = None
    if os.path.exists(round9_path):
        with open(round9_path) as f:
            round9_results = json.load(f)
        print(f"  Loaded round9 Adamson results from {round9_path}")

    # Build primary layer result
    pr = results_by_layer[PRIMARY_LAYER]

    save_results = {
        'experiment': 'round11_adamson_L15',
        'dataset': 'Adamson_K562_CRISPRa',
        'config': {
            'model': 'Geneformer-V2-316M',
            'n_control_cells': N_CTRL, 'n_hvg': N_HVG,
            'lfc_threshold_used': lfc_used,
            'n_evaluable_perturbations': len(de_targets),
            'perturbation_modality': 'CRISPRa',
            'eval_layers': [l + 1 for l in EVAL_LAYERS],
            'primary_layer_0indexed': PRIMARY_LAYER,
        },
        'results_by_layer': {
            str(l): {k: v for k, v in r.items() if k != 'per_gene'}
            for l, r in results_by_layer.items()
        },
        'replogle_L15_reference': {
            'delta': REPLOGLE_L15_DELTA,
            'cohens_d': REPLOGLE_L15_D,
            'pooled_wilcoxon_p': REPLOGLE_L15_P,
            'source': 'nested CV analysis',
        },
        'replication_verdict': verdict,
    }

    # Add Cohen's d and CI to top-level for easy access
    if pr.get('attn_vs_corr'):
        save_results['L15_summary'] = {
            'mean_diff': pr['attn_vs_corr']['mean_diff'],
            'cohens_d': pr['attn_vs_corr']['cohens_d'],
            'bootstrap_ci_95': pr['attn_vs_corr']['bootstrap_ci_95'],
            'wilcoxon_p': pr['attn_vs_corr']['wilcoxon_p'],
            'paired_t_p': pr['attn_vs_corr']['paired_t_p'],
            'n_attn_better': pr['attn_vs_corr']['n_attn_better'],
            'n_corr_better': pr['attn_vs_corr']['n_corr_better'],
            'attn_mean_auroc': pr['attention']['mean_auroc'],
            'corr_mean_auroc': pr['correlation']['mean_auroc'],
            'verdict': verdict,
        }

    with open(os.path.join(OUT_DIR, "adamson_L15_summary.json"), 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"  Saved summary to {os.path.join(OUT_DIR, 'adamson_L15_summary.json')}")

    # Save per-gene CSV
    if pr.get('per_gene'):
        df = pd.DataFrame(pr['per_gene'])
        csv_path = os.path.join(OUT_DIR, "per_gene_results_L15_adamson.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved per-gene results to {csv_path}")

    # ============================================================
    # Generate figure: 2-panel
    # ============================================================
    print("\nGenerating figure...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ----------------------------------------------------------
    # Panel A: Bar chart - Attention vs Correlation AUROC at L15
    # ----------------------------------------------------------
    ax = axes[0]
    attn_mean = pr['attention']['mean_auroc'] or 0
    attn_std = pr['attention']['std_auroc'] or 0
    corr_mean = pr['correlation']['mean_auroc'] or 0
    corr_std = pr['correlation']['std_auroc'] or 0

    x_pos = np.array([0, 1])
    bars = ax.bar(x_pos, [attn_mean, corr_mean], width=0.5,
                  yerr=[attn_std, corr_std], capsize=5,
                  color=['#2196F3', '#FF9800'], alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Attention\n(L15)', 'Correlation'], fontsize=11)
    ax.set_ylabel('Mean AUROC vs DE targets', fontsize=11)
    ax.set_title(f'(A) Adamson CRISPRa - Layer 15\n(n={pr["n_evaluated"]} perturbations)', fontsize=12)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, label='Random baseline')
    ax.legend(fontsize=8, loc='upper right')

    # Annotate with stats
    if pr.get('attn_vs_corr'):
        avc = pr['attn_vs_corr']
        stats_text = (f"$\\Delta$ = {avc['mean_diff']:.3f}\n"
                      f"d = {avc['cohens_d']:.2f}\n"
                      f"p = {avc['wilcoxon_p']:.4f}")
        ax.text(0.5, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    ax.set_ylim(0.4, max(attn_mean + attn_std, corr_mean + corr_std) + 0.08)

    # ----------------------------------------------------------
    # Panel B: Cross-dataset comparison at L15
    # ----------------------------------------------------------
    ax = axes[1]

    # Replogle L15 values
    rep_attn_delta = REPLOGLE_L15_DELTA
    rep_d = REPLOGLE_L15_D

    # Adamson L15 values
    adam_delta = pr['attn_vs_corr']['mean_diff'] if pr.get('attn_vs_corr') else 0
    adam_d = pr['attn_vs_corr']['cohens_d'] if pr.get('attn_vs_corr') else 0

    x_pos = np.array([0, 1])
    width = 0.3

    # Plot delta (mean difference)
    ax_twin = ax.twinx()

    bars1 = ax.bar(x_pos - width/2, [rep_attn_delta, adam_delta], width,
                   label='AUROC diff (attn - corr)', color='#4CAF50', alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    bars2 = ax_twin.bar(x_pos + width/2, [rep_d, adam_d], width,
                        label="Cohen's d", color='#9C27B0', alpha=0.7,
                        edgecolor='black', linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Replogle\nCRISPRi', 'Adamson\nCRISPRa'], fontsize=11)
    ax.set_ylabel('Mean AUROC difference', fontsize=10, color='#4CAF50')
    ax_twin.set_ylabel("Cohen's d", fontsize=10, color='#9C27B0')
    ax.set_title('(B) Cross-dataset L15 comparison', fontsize=12)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.4)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

    # Annotate Replogle
    ax.text(0, rep_attn_delta + 0.003, f"p={REPLOGLE_L15_P:.4f}",
            ha='center', fontsize=8, color='#4CAF50')
    # Annotate Adamson
    adam_p_str = f"p={pr['attn_vs_corr']['wilcoxon_p']:.4f}" if pr.get('attn_vs_corr') else "N/A"
    y_offset = 0.003 if adam_delta >= 0 else -0.008
    ax.text(1, adam_delta + y_offset, adam_p_str,
            ha='center', fontsize=8, color='#4CAF50')

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_adamson_L15_replication.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Figure saved to {fig_path}")
    plt.close()

    # ============================================================
    # Final summary
    # ============================================================
    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"ADAMSON L15 REPLICATION COMPLETE - Total: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")

    print(f"\nHEADLINE (L{PRIMARY_LAYER+1}):")
    print(f"  Attention mean AUROC: {pr['attention']['mean_auroc']}")
    print(f"  Correlation mean AUROC: {pr['correlation']['mean_auroc']}")
    if pr.get('attn_vs_corr'):
        avc = pr['attn_vs_corr']
        print(f"  Mean difference: {avc['mean_diff']}")
        print(f"  Cohen's d: {avc['cohens_d']}")
        print(f"  95% CI: {avc['bootstrap_ci_95']}")
        print(f"  Wilcoxon p: {avc['wilcoxon_p']}")
        print(f"  VERDICT: {avc['replication_verdict']}")

    print(f"\n  Replogle CRISPRi L15 reference:")
    print(f"    Delta: +{REPLOGLE_L15_DELTA}, d={REPLOGLE_L15_D}, p={REPLOGLE_L15_P}")

    # Report round9 multi-layer context if available
    if round9_results:
        print(f"\n  Adamson round9 multi-layer context:")
        for lkey, lres in round9_results.get('results_by_layer', {}).items():
            l1 = lres.get('layer_1indexed', lkey)
            avc = lres.get('attn_vs_corr', {})
            md = avc.get('mean_diff', 'N/A')
            wp = avc.get('wilcoxon_p', 'N/A')
            print(f"    L{l1}: diff={md}, Wilcoxon p={wp}")


if __name__ == '__main__':
    run()
