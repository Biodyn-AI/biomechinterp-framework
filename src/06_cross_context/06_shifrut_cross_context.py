#!/usr/bin/env python3
"""
Round 10 Experiment 4: Shifrut T-Cell Cross-Context Replication
================================================================
Perturbation-first attention vs correlation in primary human T cells
(non-K562). Uses Shifrut CRISPRi data (20 perturbations, ~52K cells).

Addresses: MC5 ("generalizability remains K562-limited")
           ST5 ("cross-context expansion")
"""

import os, sys, gc, json, time, pickle, warnings
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
DATA_PATH = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/data/perturb/shifrut/perturb_processed_symbols.h5ad"
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round10_shifrut_cross_context")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

N_CTRL = 2000
N_HVG = 2000
PVAL_THRESHOLD = 0.05
MIN_DE_TARGETS = 5
LFC_THRESHOLDS = [0.5, 0.25, 0.1]  # Threshold cascade

EVAL_LAYERS = [5, 12, 14, 17]  # L6, L13, L15 (best), L18
PRIMARY_LAYER = 14  # L15 (0-indexed=14), best from nested CV
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1
N_LAYERS = 18
N_HEADS = 18

CHECKPOINT_PATH = os.path.join(OUT_DIR, "checkpoint.json")


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
    X = f['X']
    indptr = X['indptr']
    start = indptr[idx]
    end = indptr[idx + 1]
    indices = X['indices'][start:end]
    data = X['data'][start:end]
    n_cols = X.attrs.get('shape', [0, 0])[1] if 'shape' in X.attrs else len(f['var']['_index'])
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
    print("SHIFRUT T-CELL CROSS-CONTEXT REPLICATION (ROUND 10)")
    print("=" * 70)

    # ============================================================
    # STEP 1: Load Shifrut data
    # ============================================================
    print("\nSTEP 1: Loading Shifrut CRISPRi data...")
    t0 = time.time()

    with h5py.File(DATA_PATH, 'r') as f:
        conditions = load_categorical_column(f['obs'], 'condition')
        n_cells_total = len(conditions)
        var_genes_raw = f['var']['_index'][:]
        if var_genes_raw.dtype.kind in ('O', 'S'):
            var_genes = np.array([x.decode() if isinstance(x, bytes) else x for x in var_genes_raw])
        else:
            var_genes = var_genes_raw
        n_genes_total = len(var_genes)

    print(f"  Total: {n_cells_total:,} cells × {n_genes_total:,} genes")
    print(f"  Cell type: Primary human T cells")
    print(f"  Perturbation type: CRISPRi")

    # Control cells
    ctrl_indices = np.where(conditions == 'control')[0]
    print(f"  Control cells: {len(ctrl_indices)}")

    np.random.seed(42)
    if len(ctrl_indices) > N_CTRL:
        ctrl_indices = np.random.choice(ctrl_indices, N_CTRL, replace=False)
        ctrl_indices.sort()
    print(f"  Using {len(ctrl_indices)} control cells")

    # Perturbation conditions
    unique_conds = np.unique(conditions)
    pert_conds = [c for c in unique_conds if c != 'control']
    print(f"  Perturbation conditions: {len(pert_conds)}")

    # Load control expression
    print("  Loading control expression (sparse)...")
    X_ctrl_full = np.zeros((len(ctrl_indices), n_genes_total), dtype=np.float32)
    with h5py.File(DATA_PATH, 'r') as f:
        for ci, idx in enumerate(ctrl_indices):
            X_ctrl_full[ci, :] = load_sparse_row(f, int(idx))
            if (ci + 1) % 500 == 0:
                print(f"    ... {ci+1}/{len(ctrl_indices)} cells loaded")

    # Normalize
    row_sums = X_ctrl_full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_ctrl_norm = np.log1p(X_ctrl_full / row_sums * 1e4)

    # Select HVGs
    gene_var = X_ctrl_norm.var(axis=0)
    hvg_idx = np.argsort(-gene_var)[:N_HVG]
    hvg_idx.sort()
    hvg_genes = [var_genes[i] for i in hvg_idx]
    hvg_upper = [g.upper() for g in hvg_genes]
    hvg_to_local = {g.upper(): i for i, g in enumerate(hvg_genes)}

    X_ctrl_hvg = X_ctrl_norm[:, hvg_idx]
    print(f"  HVG matrix: {X_ctrl_hvg.shape}")
    print(f"  Step 1 time: {time.time()-t0:.1f}s")

    del X_ctrl_full, X_ctrl_norm
    gc.collect()

    # ============================================================
    # STEP 2: DE calling with threshold cascade
    # ============================================================
    print("\nSTEP 2: Computing DE with threshold cascade...")
    t1 = time.time()

    pert_cell_indices = {}
    for cond in pert_conds:
        idx = np.where(conditions == cond)[0]
        if len(idx) >= 10:
            pert_cell_indices[cond] = idx

    print(f"  Perturbations with >=10 cells: {len(pert_cell_indices)}")

    de_results_by_threshold = {}

    for lfc_thresh in LFC_THRESHOLDS:
        de_targets = {}
        de_all_genes = {}

        for pg, pidx in pert_cell_indices.items():
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
                is_de = bool(reject[j]) and abs(lfcs[j]) >= lfc_thresh
                gene_name = hvg_upper[j]
                all_genes_dict[gene_name] = is_de
                if is_de:
                    targets.add(gene_name)

            if len(targets) >= MIN_DE_TARGETS:
                de_targets[pg] = targets
                de_all_genes[pg] = all_genes_dict

            del X_pert, X_pert_norm, X_pert_hvg

        n_evaluable = len(de_targets)
        de_results_by_threshold[lfc_thresh] = (de_targets, de_all_genes)
        print(f"  LFC > {lfc_thresh}: {n_evaluable} evaluable perturbations")

    # Select primary threshold
    primary_lfc = None
    for lfc_thresh in LFC_THRESHOLDS:
        de_targets, de_all_genes = de_results_by_threshold[lfc_thresh]
        if len(de_targets) >= 5:
            primary_lfc = lfc_thresh
            break

    if primary_lfc is None:
        print("  WARNING: No threshold yields >= 5 evaluable perturbations!")
        primary_lfc = LFC_THRESHOLDS[-1]  # Use most lenient

    de_targets, de_all_genes = de_results_by_threshold[primary_lfc]
    print(f"  Primary threshold: LFC > {primary_lfc} ({len(de_targets)} evaluable)")
    print(f"  DE time: {time.time()-t1:.1f}s")

    gc.collect()

    # ============================================================
    # STEP 3: Load Geneformer and tokenize
    # ============================================================
    print("\nSTEP 3: Loading Geneformer and tokenizing...")

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

    # ============================================================
    # STEP 4: Extract attention (numpy matrix accumulation)
    # ============================================================
    print("\nSTEP 4: Extracting attention...")
    t2 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  Device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = BertForMaskedLM.from_pretrained(
        MODEL_NAME, subfolder=MODEL_SUBFOLDER,
        output_attentions=False, output_hidden_states=False,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Hook layers
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

    for layer_idx in EVAL_LAYERS:
        target_layer = model.bert.encoder.layer[layer_idx]
        orig_forwards[layer_idx] = target_layer.forward
        target_layer.forward = make_patched_forward(layer_idx, captured_attentions, orig_forwards[layer_idx])

    # Numpy matrix accumulators per layer
    n_hvg = len(hvg_genes)
    hvg_name_to_idx = {g: i for i, g in enumerate(hvg_genes)}
    layer_attn_sum = {l: np.zeros((n_hvg, n_hvg), dtype=np.float64) for l in EVAL_LAYERS}
    layer_attn_count = {l: np.zeros((n_hvg, n_hvg), dtype=np.float64) for l in EVAL_LAYERS}

    sorted_indices = sorted(range(len(control_tokens)), key=lambda i: len(control_tokens[i]))
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

        for layer_idx in EVAL_LAYERS:
            if layer_idx not in captured_attentions:
                continue
            layer_attn = captured_attentions[layer_idx].mean(dim=1).cpu().float().numpy()

            for cell_i in range(bs):
                seq_len = int(attention_mask[cell_i].sum())
                cell_tok = input_ids[cell_i, :seq_len]
                cell_attn = layer_attn[cell_i, :seq_len, :seq_len]
                gene_tokens = cell_tok[1:seq_len - 1]

                pos_list = []
                idx_list = []
                for pos_offset, tok_id in enumerate(gene_tokens):
                    gene = token_to_hvg.get(int(tok_id))
                    if gene is not None:
                        hvg_idx_val = hvg_name_to_idx.get(gene)
                        if hvg_idx_val is not None:
                            pos_list.append(pos_offset + 1)
                            idx_list.append(hvg_idx_val)

                if len(pos_list) < 2:
                    continue

                pos_arr = np.array(pos_list)
                idx_arr = np.array(idx_list)
                sub_attn = cell_attn[np.ix_(pos_arr, pos_arr)]
                layer_attn_sum[layer_idx][np.ix_(idx_arr, idx_arr)] += sub_attn
                layer_attn_count[layer_idx][np.ix_(idx_arr, idx_arr)] += 1.0

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
                  f"rate: {rate:.1f}/s, ETA: {eta:.0f}s")

    # Restore
    for layer_idx in EVAL_LAYERS:
        model.bert.encoder.layer[layer_idx].forward = orig_forwards[layer_idx]
    del model
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()

    # Convert to edge dicts
    layer_attention_edges = {}
    for layer_idx in EVAL_LAYERS:
        mask = layer_attn_count[layer_idx] > 0
        avg = np.zeros_like(layer_attn_sum[layer_idx])
        avg[mask] = layer_attn_sum[layer_idx][mask] / layer_attn_count[layer_idx][mask]
        edges = {}
        nz = np.argwhere(mask)
        for r, c in nz:
            if r != c:
                edges[(hvg_genes[r], hvg_genes[c])] = float(avg[r, c])
        layer_attention_edges[layer_idx] = edges
        print(f"    Layer {layer_idx+1}: {len(edges)} edges")

    del layer_attn_sum, layer_attn_count
    gc.collect()

    print(f"  Extraction time: {time.time()-t2:.1f}s")

    # ============================================================
    # STEP 5: Compute correlation
    # ============================================================
    print("\nSTEP 5: Computing Spearman correlation...")
    ranks = np.apply_along_axis(rankdata, 0, X_ctrl_hvg)
    corr_matrix = np.corrcoef(ranks.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # ============================================================
    # STEP 6: Evaluate perturbation-first AUROC
    # ============================================================
    print("\nSTEP 6: Evaluating perturbation-first AUROC...")

    results_by_layer = {}
    rng = np.random.RandomState(42)

    for layer_idx in EVAL_LAYERS:
        attn_edges = layer_attention_edges[layer_idx]
        layer_aurocs_attn = []
        layer_aurocs_corr = []
        per_gene_results = []

        for pg in de_targets:
            all_genes_dict = de_all_genes[pg]
            pg_local = hvg_to_local.get(pg.upper())
            if pg_local is None:
                continue

            labels = []
            scores_attn = []
            scores_corr = []

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

            labels = np.array(labels)
            if labels.sum() < MIN_DE_TARGETS or labels.sum() == len(labels):
                continue

            try:
                auroc_attn = roc_auc_score(labels, np.array(scores_attn))
                auroc_corr = roc_auc_score(labels, np.array(scores_corr))
            except ValueError:
                continue

            layer_aurocs_attn.append(auroc_attn)
            layer_aurocs_corr.append(auroc_corr)
            per_gene_results.append({
                'gene': pg,
                'auroc_attn': round(auroc_attn, 4),
                'auroc_corr': round(auroc_corr, 4),
                'n_de': int(labels.sum()),
                'n_total': int(len(labels)),
            })

        aurocs_attn = np.array(layer_aurocs_attn)
        aurocs_corr = np.array(layer_aurocs_corr)

        layer_result = {
            'layer_1indexed': layer_idx + 1,
            'n_evaluated': len(aurocs_attn),
            'attention': {
                'mean_auroc': round(float(aurocs_attn.mean()), 4) if len(aurocs_attn) > 0 else None,
                'std_auroc': round(float(aurocs_attn.std()), 4) if len(aurocs_attn) > 0 else None,
                'median_auroc': round(float(np.median(aurocs_attn)), 4) if len(aurocs_attn) > 0 else None,
            },
            'correlation': {
                'mean_auroc': round(float(aurocs_corr.mean()), 4) if len(aurocs_corr) > 0 else None,
                'std_auroc': round(float(aurocs_corr.std()), 4) if len(aurocs_corr) > 0 else None,
                'median_auroc': round(float(np.median(aurocs_corr)), 4) if len(aurocs_corr) > 0 else None,
            },
            'per_gene': per_gene_results,
        }

        if len(aurocs_attn) > 1:
            diffs = aurocs_attn - aurocs_corr
            try:
                w_stat, w_p = stats.wilcoxon(diffs)
            except:
                w_p = 1.0
            try:
                t_stat, t_p = stats.ttest_rel(aurocs_attn, aurocs_corr)
            except:
                t_p = 1.0

            cohens_d = float(diffs.mean() / diffs.std()) if diffs.std() > 0 else 0.0

            layer_result['attn_vs_corr'] = {
                'wilcoxon_p': round(float(w_p), 6),
                'paired_t_p': round(float(t_p), 6),
                'mean_diff': round(float(diffs.mean()), 4),
                'cohens_d': round(cohens_d, 4),
                'n_attn_better': int((diffs > 0).sum()),
                'n_corr_better': int((diffs < 0).sum()),
                'ci_95': [round(float(np.percentile(diffs, 2.5)), 4),
                          round(float(np.percentile(diffs, 97.5)), 4)],
            }

        results_by_layer[layer_idx] = layer_result

        print(f"\n  Layer {layer_idx+1}:")
        print(f"    Evaluated: {len(aurocs_attn)} perturbations")
        if len(aurocs_attn) > 0:
            print(f"    Attention AUROC:   {aurocs_attn.mean():.4f} ± {aurocs_attn.std():.4f}")
            print(f"    Correlation AUROC: {aurocs_corr.mean():.4f} ± {aurocs_corr.std():.4f}")
            if 'attn_vs_corr' in layer_result:
                avc = layer_result['attn_vs_corr']
                print(f"    Diff: {avc['mean_diff']:+.4f}, Wilcoxon p={avc['wilcoxon_p']:.6f}")
                print(f"    Cohen's d: {avc['cohens_d']:.4f}")

    # ============================================================
    # STEP 7: Multi-threshold comparison
    # ============================================================
    print("\n\nSTEP 7: Multi-threshold comparison...")

    threshold_comparison = []
    for lfc_thresh in LFC_THRESHOLDS:
        dt, dag = de_results_by_threshold[lfc_thresh]
        if len(dt) < 3:
            continue

        # Quick eval at primary layer
        attn_edges = layer_attention_edges[PRIMARY_LAYER]
        aurocs_a = []
        aurocs_c = []
        for pg in dt:
            agd = dag[pg]
            pg_local = hvg_to_local.get(pg.upper())
            if pg_local is None:
                continue
            labels = []
            sa = []
            sc = []
            for tg, is_de in agd.items():
                tl = hvg_to_local.get(tg)
                if tl is None:
                    continue
                fwd = attn_edges.get((pg, hvg_genes[tl]), 0.0)
                rev = attn_edges.get((hvg_genes[tl], pg), 0.0)
                labels.append(1 if is_de else 0)
                sa.append(max(fwd, rev))
                sc.append(abs(corr_matrix[pg_local, tl]))
            labels = np.array(labels)
            if labels.sum() < MIN_DE_TARGETS or labels.sum() == len(labels):
                continue
            try:
                aurocs_a.append(roc_auc_score(labels, np.array(sa)))
                aurocs_c.append(roc_auc_score(labels, np.array(sc)))
            except:
                continue

        if aurocs_a:
            threshold_comparison.append({
                'lfc_threshold': lfc_thresh,
                'n_evaluable': len(aurocs_a),
                'attn_mean': round(float(np.mean(aurocs_a)), 4),
                'corr_mean': round(float(np.mean(aurocs_c)), 4),
                'diff': round(float(np.mean(aurocs_a) - np.mean(aurocs_c)), 4),
            })
            print(f"  LFC > {lfc_thresh}: n={len(aurocs_a)}, "
                  f"attn={np.mean(aurocs_a):.4f}, corr={np.mean(aurocs_c):.4f}, "
                  f"diff={np.mean(aurocs_a) - np.mean(aurocs_c):+.4f}")

    # ============================================================
    # STEP 8: Generate figure
    # ============================================================
    print("\nSTEP 8: Generating figure...")
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
    ax.set_ylabel('Mean AUROC')
    ax.set_title(f'A. Shifrut T-Cell CRISPRi\n(n={len(de_targets)}, LFC>{primary_lfc})')
    ax.legend(fontsize=8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

    # Panel B: Per-perturbation scatter (primary layer)
    ax = axes[1]
    pr = results_by_layer.get(PRIMARY_LAYER, {})
    if pr.get('per_gene'):
        pg_df = pd.DataFrame(pr['per_gene'])
        ax.scatter(pg_df['auroc_corr'], pg_df['auroc_attn'], alpha=0.6, s=40, color='#2196F3')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('Correlation AUROC')
        ax.set_ylabel(f'Attention (L{PRIMARY_LAYER+1}) AUROC')
        ax.set_title(f'B. Per-Perturbation Scatter')
        ax.set_aspect('equal')
        ax.set_xlim(0.3, 0.9)
        ax.set_ylim(0.3, 0.9)

    # Panel C: Threshold sensitivity
    ax = axes[2]
    if threshold_comparison:
        thresholds = [tc['lfc_threshold'] for tc in threshold_comparison]
        attn_vals = [tc['attn_mean'] for tc in threshold_comparison]
        corr_vals = [tc['corr_mean'] for tc in threshold_comparison]
        x = np.arange(len(thresholds))
        ax.bar(x - 0.15, attn_vals, 0.3, label='Attention', color='#2196F3', alpha=0.8)
        ax.bar(x + 0.15, corr_vals, 0.3, label='Correlation', color='#FF9800', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'LFC>{t}' for t in thresholds])
        ax.set_ylabel('Mean AUROC')
        ax.set_title('C. Threshold Sensitivity')
        ax.legend(fontsize=8)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
        # Annotate n
        for i, tc in enumerate(threshold_comparison):
            ax.text(i, 0.48, f'n={tc["n_evaluable"]}', ha='center', fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_shifrut_cross_context.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")

    # ============================================================
    # STEP 9: Save results
    # ============================================================
    print("\nSTEP 9: Saving results...")

    summary = {
        'experiment': 'round10_shifrut_cross_context',
        'dataset': 'Shifrut_TCells_CRISPRi',
        'cell_type': 'Primary human T cells',
        'perturbation_modality': 'CRISPRi',
        'n_cells_total': int(n_cells_total),
        'n_control_cells_used': len(ctrl_indices),
        'n_hvg': N_HVG,
        'primary_lfc_threshold': primary_lfc,
        'n_evaluable_perturbations': len(de_targets),
        'evaluable_perturbations': list(de_targets.keys()),
        'eval_layers': [l + 1 for l in EVAL_LAYERS],
        'results_by_layer': {
            str(l): {k: v for k, v in r.items() if k != 'per_gene'}
            for l, r in results_by_layer.items()
        },
        'threshold_comparison': threshold_comparison,
        'total_time_seconds': round(time.time() - total_t0, 1),
    }

    with open(os.path.join(OUT_DIR, "shifrut_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save per-gene CSVs
    for layer_idx in EVAL_LAYERS:
        if results_by_layer[layer_idx].get('per_gene'):
            df = pd.DataFrame(results_by_layer[layer_idx]['per_gene'])
            df.to_csv(os.path.join(OUT_DIR, f"per_gene_L{layer_idx+1}_shifrut.csv"), index=False)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"SHIFRUT CROSS-CONTEXT COMPLETE - Total: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")

    pr = results_by_layer.get(PRIMARY_LAYER, {})
    print(f"\nHEADLINE (L{PRIMARY_LAYER+1}, primary):")
    if pr.get('attention', {}).get('mean_auroc') is not None:
        print(f"  Attention mean AUROC: {pr['attention']['mean_auroc']}")
        print(f"  Correlation mean AUROC: {pr['correlation']['mean_auroc']}")
        if pr.get('attn_vs_corr'):
            avc = pr['attn_vs_corr']
            print(f"  Difference: {avc['mean_diff']:+.4f}")
            print(f"  Wilcoxon p: {avc['wilcoxon_p']}")
            print(f"  Cohen's d: {avc['cohens_d']}")

    for layer_idx in EVAL_LAYERS:
        lr = results_by_layer.get(layer_idx, {})
        attn_a = lr.get('attention', {}).get('mean_auroc', 'N/A')
        corr_a = lr.get('correlation', {}).get('mean_auroc', 'N/A')
        diff = lr.get('attn_vs_corr', {}).get('mean_diff', 'N/A')
        print(f"  L{layer_idx+1}: attn={attn_a}, corr={corr_a}, diff={diff}")


if __name__ == '__main__':
    run()
