#!/usr/bin/env python3
"""
Round 9 Experiment 1: Cross-Context Replication on Adamson CRISPRa
===================================================================
Replicates the attention vs correlation perturbation-first comparison
on Adamson K562 CRISPRa data (opposite perturbation modality from Replogle CRISPRi).

Addresses: MC2 ("no advantage claim too broad")
           ST4 ("cross-context perturbation-first replication")
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
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round9_adamson_cross_context")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

N_CTRL = 2000
N_HVG = 2000
LFC_THRESHOLD = 0.5
PVAL_THRESHOLD = 0.05
MIN_DE_TARGETS = 5
# Also try lenient threshold if too few evaluable perturbations
LFC_LENIENT = 0.25

EVAL_LAYERS = [5, 12, 17]  # L6, L13, L18
PRIMARY_LAYER = 12
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1


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
    print("ADAMSON CRISPRa CROSS-CONTEXT REPLICATION")
    print("=" * 70)

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

    print(f"  Total: {n_cells_total:,} cells × {n_genes_total:,} genes")

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
    # STEP 7: Evaluate attention vs correlation
    # ============================================================
    print("\nSTEP 7: Evaluating predictors...")

    results_by_layer = {}
    rng = np.random.RandomState(42)

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

        if len(aurocs_attn) > 1:
            try:
                w_stat, w_p = stats.wilcoxon(aurocs_attn - aurocs_corr)
            except:
                w_p = 1.0
            try:
                t_stat, t_p = stats.ttest_rel(aurocs_attn, aurocs_corr)
            except:
                t_p = 1.0
            results_by_layer[layer_idx]['attn_vs_corr'] = {
                'wilcoxon_p': round(float(w_p), 6),
                'paired_t_p': round(float(t_p), 6),
                'mean_diff': round(float((aurocs_attn - aurocs_corr).mean()), 4),
                'n_attn_better': int((aurocs_attn > aurocs_corr).sum()),
                'n_corr_better': int((aurocs_corr > aurocs_attn).sum()),
            }

        print(f"\n  Layer {layer_idx+1}:")
        print(f"    Evaluated: {len(aurocs_attn)} perturbations")
        if len(aurocs_attn) > 0:
            print(f"    Attention AUROC:   {aurocs_attn.mean():.4f} ± {aurocs_attn.std():.4f}")
            print(f"    Correlation AUROC: {aurocs_corr.mean():.4f} ± {aurocs_corr.std():.4f}")
            if 'attn_vs_corr' in results_by_layer[layer_idx]:
                avc = results_by_layer[layer_idx]['attn_vs_corr']
                print(f"    Diff: {avc['mean_diff']:.4f}, Wilcoxon p={avc['wilcoxon_p']}")

    # ============================================================
    # STEP 8: Save results and generate figure
    # ============================================================
    print("\nSTEP 8: Saving results...")

    save_results = {
        'experiment': 'round9_adamson_cross_context',
        'dataset': 'Adamson_K562_CRISPRa',
        'config': {
            'model': 'Geneformer-V2-316M',
            'n_control_cells': N_CTRL, 'n_hvg': N_HVG,
            'lfc_threshold_used': lfc_used,
            'n_evaluable_perturbations': len(de_targets),
            'perturbation_modality': 'CRISPRa',
        },
        'results_by_layer': {
            str(l): {k: v for k, v in r.items() if k != 'per_gene'}
            for l, r in results_by_layer.items()
        },
    }

    with open(os.path.join(OUT_DIR, "adamson_summary.json"), 'w') as f:
        json.dump(save_results, f, indent=2)

    # Save per-gene CSVs
    for layer_idx in EVAL_LAYERS:
        if results_by_layer[layer_idx].get('per_gene'):
            df = pd.DataFrame(results_by_layer[layer_idx]['per_gene'])
            df.to_csv(os.path.join(OUT_DIR, f"per_gene_results_L{layer_idx+1}_adamson.csv"), index=False)

    # Load Replogle results for comparison
    replogle_path = os.path.join(PAPER_DIR, "experiments", "attention_perturbation_first_results",
                                  "attention_perturbation_first_summary.json")
    replogle_results = None
    if os.path.exists(replogle_path):
        with open(replogle_path) as f:
            replogle_results = json.load(f)

    # Generate figure
    print("\nGenerating figure...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Layer comparison
    ax = axes[0]
    layers_1idx = [l + 1 for l in EVAL_LAYERS]
    attn_means = [results_by_layer[l]['attention']['mean_auroc'] or 0 for l in EVAL_LAYERS]
    corr_means = [results_by_layer[l]['correlation']['mean_auroc'] or 0 for l in EVAL_LAYERS]
    x = np.arange(len(EVAL_LAYERS))
    width = 0.3
    ax.bar(x - width/2, attn_means, width, label='Attention', color='#2196F3', alpha=0.8)
    ax.bar(x + width/2, corr_means, width, label='Correlation', color='#FF9800', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers_1idx])
    ax.set_ylabel('Mean AUROC vs DE targets')
    ax.set_title(f'(A) Adamson CRISPRa\n(n={len(de_targets)} perturbations)')
    ax.legend(fontsize=8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

    # Panel B: Per-perturbation scatter (primary layer)
    ax = axes[1]
    pr = results_by_layer[PRIMARY_LAYER]
    if pr.get('per_gene'):
        pg_df = pd.DataFrame(pr['per_gene'])
        ax.scatter(pg_df['auroc_corr'], pg_df['auroc_attn'], alpha=0.5, s=30, color='#2196F3')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('Correlation AUROC')
        ax.set_ylabel(f'Attention (L{PRIMARY_LAYER+1}) AUROC')
        ax.set_title(f'(B) Per-perturbation scatter\nAdamson CRISPRa')
        ax.set_aspect('equal')

    # Panel C: Cross-dataset comparison
    ax = axes[2]
    if replogle_results:
        rep_attn = replogle_results['results_by_layer']['12']['attention']['mean_auroc']
        rep_corr = replogle_results['results_by_layer']['12']['correlation']['mean_auroc']
        adam_attn = results_by_layer[PRIMARY_LAYER]['attention']['mean_auroc'] or 0
        adam_corr = results_by_layer[PRIMARY_LAYER]['correlation']['mean_auroc'] or 0

        x = np.arange(2)
        ax.bar(x - 0.15, [rep_attn, adam_attn], 0.3, label='Attention', color='#2196F3', alpha=0.8)
        ax.bar(x + 0.15, [rep_corr, adam_corr], 0.3, label='Correlation', color='#FF9800', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(['Replogle\nCRISPRi', 'Adamson\nCRISPRa'])
        ax.set_ylabel('Mean AUROC (L13)')
        ax.set_title('(C) Cross-dataset comparison')
        ax.legend(fontsize=8)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_cross_context_replication.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Figure saved to {fig_path}")

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"ADAMSON CROSS-CONTEXT COMPLETE - Total: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")

    pr = results_by_layer[PRIMARY_LAYER]
    print(f"\nHEADLINE (L{PRIMARY_LAYER+1}):")
    print(f"  Attention mean AUROC: {pr['attention']['mean_auroc']}")
    print(f"  Correlation mean AUROC: {pr['correlation']['mean_auroc']}")
    if pr.get('attn_vs_corr'):
        print(f"  Difference: {pr['attn_vs_corr']['mean_diff']}")
        print(f"  Wilcoxon p: {pr['attn_vs_corr']['wilcoxon_p']}")


if __name__ == '__main__':
    run()
