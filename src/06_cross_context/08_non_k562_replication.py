#!/usr/bin/env python3
"""
Round 13: Non-K562 Perturbation-First Replication
=================================================
Powered, direct attention perturbation-first replication in non-K562 contexts.
Two biologically distinct systems with adequate evaluable perturbations:

  1. TianKampmann2021 — iPSC-derived neurons, CRISPRi, 185 perturbations
  2. ReplogleWeissman2022_rpe1 — RPE1 retinal epithelial cells, CRISPRi, ~2000 perturbations

Both datasets from scPerturb (Zenodo, harmonized format).

Addresses: MC4 ("generalization underpowered beyond K562")
           Stress Test 5 ("attention-specific cross-context replication")
"""

import os, sys, gc, json, time, pickle, warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import pandas as pd
import h5py
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from statsmodels.stats.multitest import multipletests

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
DATA_DIR = os.path.join(PAPER_DIR, "experiments", "round13_non_k562_replication", "data")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round13_non_k562_replication")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

N_CTRL = 2000
N_HVG = 2000
PVAL_THRESHOLD = 0.05
MIN_DE_TARGETS = 5
LFC_THRESHOLDS = [0.5, 0.25, 0.1]
EVAL_LAYERS = [5, 12, 14, 17]  # L6, L13, L15, L18
PRIMARY_LAYER = 14  # L15 (0-indexed)
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1
MIN_CELLS_PER_PERT = 20

CHECKPOINT_PATH = os.path.join(OUT_DIR, "checkpoint.json")

DATASETS = {
    'TianKampmann2021': {
        'path': os.path.join(DATA_DIR, 'TianKampmann2021_CRISPRi.h5ad'),
        'cell_type': 'iPSC-derived neurons',
        'perturbation_type': 'CRISPRi',
        'organism': 'human',
        'description': 'Tian et al. 2021 — iPSC-induced glutamatergic neurons',
    },
    'ReplogleRPE1': {
        'path': os.path.join(DATA_DIR, 'ReplogleWeissman2022_rpe1.h5ad'),
        'cell_type': 'RPE1 retinal pigment epithelial',
        'perturbation_type': 'CRISPRi',
        'organism': 'human',
        'description': 'Replogle et al. 2022 — hTERT-RPE1 retinal pigment epithelial cells',
    },
}


# ==========================================================================
# Utility functions
# ==========================================================================

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


def is_sparse_x(f):
    """Check if X is stored in sparse format."""
    return isinstance(f['X'], h5py.Group) and 'indptr' in f['X']


def load_sparse_row(f, idx):
    X = f['X']
    indptr = X['indptr']
    start = indptr[idx]
    end = indptr[idx + 1]
    indices = X['indices'][start:end]
    data = X['data'][start:end]
    n_cols = X.attrs.get('shape', [0, 0])[1] if 'shape' in X.attrs else _get_n_genes(f)
    row = np.zeros(n_cols, dtype=np.float32)
    row[indices] = data
    return row


def _get_n_genes(f):
    """Get number of genes from var."""
    for key in ['_index', 'gene_name']:
        if key in f['var']:
            return len(f['var'][key])
    return len(list(f['var'].values())[0])


def load_rows_batch(f, indices, n_cols):
    """Load multiple rows efficiently (handles sparse and dense X)."""
    X = f['X']
    if is_sparse_x(f):
        indptr = X['indptr']
        result = np.zeros((len(indices), n_cols), dtype=np.float32)
        for ci, idx in enumerate(indices):
            start = indptr[idx]
            end = indptr[idx + 1]
            col_idx = X['indices'][start:end]
            data = X['data'][start:end]
            result[ci, col_idx] = data
    else:
        # Dense matrix — load row by row to avoid memory issues
        result = np.zeros((len(indices), n_cols), dtype=np.float32)
        for ci, idx in enumerate(indices):
            result[ci, :] = X[idx, :]
    return result


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


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_checkpoint(data):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(data, f, indent=2)


# ==========================================================================
# Per-dataset pipeline
# ==========================================================================

def run_dataset(dataset_name, dataset_info, token_dict, gene_median_dict, gene_name_id_dict, checkpoint):
    """Full perturbation-first pipeline for one dataset."""
    import torch
    from transformers import BertForMaskedLM

    data_path = dataset_info['path']
    if not os.path.exists(data_path):
        print(f"  SKIP: {data_path} not found")
        return None

    ds_out_dir = os.path.join(OUT_DIR, dataset_name)
    os.makedirs(ds_out_dir, exist_ok=True)

    ds_checkpoint = checkpoint.get(dataset_name, {})
    total_t0 = time.time()

    print(f"\n{'=' * 70}")
    print(f"DATASET: {dataset_name}")
    print(f"  {dataset_info['description']}")
    print(f"  Cell type: {dataset_info['cell_type']}")
    print(f"  Perturbation: {dataset_info['perturbation_type']}")
    print(f"{'=' * 70}")

    # ==================================================================
    # STEP 1: Load data from scPerturb h5ad
    # ==================================================================
    print("\n[STEP 1] Loading scPerturb data...")
    t0 = time.time()

    with h5py.File(data_path, 'r') as f:
        # Load obs columns
        perturbations = load_categorical_column(f['obs'], 'perturbation')
        nperts_raw = f['obs']['nperts']
        if isinstance(nperts_raw, h5py.Group):
            cats = nperts_raw['categories'][:]
            codes = nperts_raw['codes'][:]
            nperts = cats[codes]
        else:
            nperts = nperts_raw[:]

        # Gene names (handle different h5ad formats)
        if '_index' in f['var']:
            var_genes_raw = f['var']['_index'][:]
        elif 'gene_name' in f['var']:
            var_genes_raw = f['var']['gene_name'][:]
        else:
            # Fallback: use first string column in var
            for vk in f['var'].keys():
                v = f['var'][vk]
                if isinstance(v, h5py.Dataset) and v.dtype.kind in ('O', 'S'):
                    var_genes_raw = v[:]
                    break
        if var_genes_raw.dtype.kind in ('O', 'S'):
            var_genes = np.array([x.decode() if isinstance(x, bytes) else x for x in var_genes_raw])
        else:
            var_genes = var_genes_raw

        # Ensembl IDs (if available)
        has_ensembl = 'ensembl_id' in f['var']
        if has_ensembl:
            ens_raw = f['var']['ensembl_id'][:]
            if ens_raw.dtype.kind in ('O', 'S'):
                var_ensembl = np.array([x.decode() if isinstance(x, bytes) else x for x in ens_raw])
            else:
                var_ensembl = ens_raw
        else:
            var_ensembl = None

        n_cells_total = len(perturbations)
        n_genes_total = len(var_genes)

    print(f"  Total: {n_cells_total:,} cells x {n_genes_total:,} genes")
    print(f"  Ensembl IDs available: {has_ensembl}")

    # Identify controls
    ctrl_indices = np.where(nperts == 0)[0]
    if len(ctrl_indices) == 0:
        ctrl_indices = np.where(perturbations == 'control')[0]
    print(f"  Control cells: {len(ctrl_indices)}")

    np.random.seed(42)
    if len(ctrl_indices) > N_CTRL:
        ctrl_indices = np.random.choice(ctrl_indices, N_CTRL, replace=False)
        ctrl_indices.sort()
    print(f"  Using {len(ctrl_indices)} control cells")

    # Perturbation conditions
    unique_perts = np.unique(perturbations)
    pert_conds = [p for p in unique_perts if p != 'control' and p not in ('', 'nan')]
    print(f"  Unique perturbation conditions: {len(pert_conds)}")

    # Per-perturbation cell counts
    pert_cell_indices = {}
    for cond in pert_conds:
        idx = np.where(perturbations == cond)[0]
        if len(idx) >= MIN_CELLS_PER_PERT:
            pert_cell_indices[cond] = idx
    print(f"  Perturbations with >= {MIN_CELLS_PER_PERT} cells: {len(pert_cell_indices)}")

    # Load control expression (sparse, batch)
    print("  Loading control expression...")
    with h5py.File(data_path, 'r') as f:
        X_ctrl_full = load_rows_batch(f, ctrl_indices, n_genes_total)
        for ci in range(0, len(ctrl_indices), 500):
            if ci > 0:
                print(f"    ... {ci}/{len(ctrl_indices)} cells loaded")

    # Normalize (log1p CPM)
    row_sums = X_ctrl_full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_ctrl_norm = np.log1p(X_ctrl_full / row_sums * 1e4)

    # Select HVGs + ensure perturbation genes are included
    gene_var = X_ctrl_norm.var(axis=0)
    hvg_idx_initial = set(np.argsort(-gene_var)[:N_HVG].tolist())

    # Add perturbation gene indices if not already in HVG set
    var_genes_upper = [g.upper() for g in var_genes]
    pert_genes_added = 0
    for pg in pert_cell_indices:
        pg_upper = pg.upper()
        for vi, vg in enumerate(var_genes_upper):
            if vg == pg_upper and vi not in hvg_idx_initial:
                hvg_idx_initial.add(vi)
                pert_genes_added += 1
                break
    if pert_genes_added > 0:
        print(f"  Added {pert_genes_added} perturbation genes to gene set")

    hvg_idx = np.sort(np.array(list(hvg_idx_initial)))
    hvg_genes = [var_genes[i] for i in hvg_idx]
    hvg_upper = [g.upper() for g in hvg_genes]
    hvg_to_local = {g.upper(): i for i, g in enumerate(hvg_genes)}

    X_ctrl_hvg = X_ctrl_norm[:, hvg_idx]
    print(f"  HVG matrix: {X_ctrl_hvg.shape}")
    print(f"  Step 1 time: {time.time()-t0:.1f}s")

    del X_ctrl_full, X_ctrl_norm
    gc.collect()

    # ==================================================================
    # STEP 2: DE calling (single pass, then filter by threshold)
    # ==================================================================
    if ds_checkpoint.get('de_complete'):
        print("\n[STEP 2] DE calling — loading from checkpoint...")
        with open(os.path.join(ds_out_dir, 'de_cache.pkl'), 'rb') as f:
            de_cache = pickle.load(f)
        de_results_by_threshold = de_cache['de_results_by_threshold']
        primary_lfc = de_cache['primary_lfc']
        de_targets = de_cache['de_targets']
        de_all_genes = de_cache['de_all_genes']
        print(f"  Primary threshold: LFC > {primary_lfc} ({len(de_targets)} evaluable)")
    else:
        print("\n[STEP 2] Computing DE (single pass, multi-threshold filter)...")
        t1 = time.time()

        # --- Single-pass DE with incremental checkpointing ---
        de_raw_path = os.path.join(ds_out_dir, 'de_raw_checkpoint.pkl')
        raw_de_results = {}  # pg -> (reject_mask, lfcs_array)

        # Resume from incremental checkpoint if available
        if os.path.exists(de_raw_path):
            with open(de_raw_path, 'rb') as f:
                raw_de_results = pickle.load(f)
            print(f"  Resuming DE from checkpoint: {len(raw_de_results)}/{len(pert_cell_indices)} done")

        pert_list = list(pert_cell_indices.items())
        for pi, (pg, pidx) in enumerate(pert_list):
            if pg in raw_de_results:
                continue  # already computed

            with h5py.File(data_path, 'r') as f:
                X_pert = load_rows_batch(f, pidx, n_genes_total)

            row_sums_p = X_pert.sum(axis=1, keepdims=True)
            row_sums_p[row_sums_p == 0] = 1
            X_pert_norm = np.log1p(X_pert / row_sums_p * 1e4)
            X_pert_hvg = X_pert_norm[:, hvg_idx]

            pvals, lfcs = call_de_vectorized(X_ctrl_hvg, X_pert_hvg)
            reject, _, _, _ = multipletests(pvals, alpha=PVAL_THRESHOLD, method='fdr_bh')

            raw_de_results[pg] = (reject, lfcs)

            del X_pert, X_pert_norm, X_pert_hvg

            if (pi + 1) % 50 == 0:
                print(f"    DE: {pi+1}/{len(pert_cell_indices)} perturbations processed")
            # Incremental checkpoint every 200 perturbations
            if (pi + 1) % 200 == 0:
                with open(de_raw_path, 'wb') as f:
                    pickle.dump(raw_de_results, f)

        # Final save of raw DE
        with open(de_raw_path, 'wb') as f:
            pickle.dump(raw_de_results, f)
        print(f"  DE pass complete: {len(raw_de_results)} perturbations, time: {time.time()-t1:.1f}s")

        # --- Apply LFC thresholds (fast, in-memory) ---
        print("  Applying LFC threshold filters...")
        de_results_by_threshold = {}
        for lfc_thresh in LFC_THRESHOLDS:
            de_targets_t = {}
            de_all_genes_t = {}

            for pg, (reject, lfcs) in raw_de_results.items():
                targets = set()
                all_genes_dict = {}
                pg_local_idx = hvg_to_local.get(pg.upper())
                for j in range(len(hvg_genes)):
                    if pg_local_idx is not None and j == pg_local_idx:
                        continue
                    is_de = bool(reject[j]) and abs(lfcs[j]) >= lfc_thresh
                    gene_name = hvg_upper[j]
                    all_genes_dict[gene_name] = is_de
                    if is_de:
                        targets.add(gene_name)

                if len(targets) >= MIN_DE_TARGETS:
                    de_targets_t[pg] = targets
                    de_all_genes_t[pg] = all_genes_dict

            n_evaluable = len(de_targets_t)
            de_results_by_threshold[lfc_thresh] = (de_targets_t, de_all_genes_t)
            print(f"  LFC > {lfc_thresh}: {n_evaluable} evaluable perturbations")

        del raw_de_results
        gc.collect()

        # Select primary threshold
        primary_lfc = None
        for lfc_thresh in LFC_THRESHOLDS:
            dt, dag = de_results_by_threshold[lfc_thresh]
            if len(dt) >= 10:
                primary_lfc = lfc_thresh
                break

        if primary_lfc is None:
            for lfc_thresh in LFC_THRESHOLDS:
                dt, dag = de_results_by_threshold[lfc_thresh]
                if len(dt) >= 5:
                    primary_lfc = lfc_thresh
                    break
            if primary_lfc is None:
                primary_lfc = LFC_THRESHOLDS[-1]

        de_targets, de_all_genes = de_results_by_threshold[primary_lfc]
        print(f"  Primary threshold: LFC > {primary_lfc} ({len(de_targets)} evaluable)")
        print(f"  DE time: {time.time()-t1:.1f}s")

        # Save final DE cache
        with open(os.path.join(ds_out_dir, 'de_cache.pkl'), 'wb') as f:
            pickle.dump({
                'de_results_by_threshold': de_results_by_threshold,
                'primary_lfc': primary_lfc,
                'de_targets': de_targets,
                'de_all_genes': de_all_genes,
            }, f)

        ds_checkpoint['de_complete'] = True
        checkpoint[dataset_name] = ds_checkpoint
        save_checkpoint(checkpoint)

    gc.collect()

    if len(de_targets) < 3:
        print(f"\n  WARNING: Only {len(de_targets)} evaluable perturbations — insufficient for analysis")
        return {'dataset': dataset_name, 'status': 'insufficient_perturbations',
                'n_evaluable': len(de_targets)}

    # ==================================================================
    # STEP 3: Tokenize control cells
    # ==================================================================
    print("\n[STEP 3] Tokenizing control cells for Geneformer...")
    t2 = time.time()

    # Build mapping: var index -> (token_id, median)
    mapped_var_indices = []
    mapped_token_ids_list = []
    mapped_medians_list = []

    for i in range(n_genes_total):
        gene_name = var_genes[i]
        ensembl = None
        # Try direct ensembl if available
        if var_ensembl is not None:
            ensembl = var_ensembl[i]
        # Fallback to gene_name_id_dict
        if ensembl is None or ensembl not in token_dict:
            ensembl = gene_name_id_dict.get(gene_name)
        if ensembl and ensembl in token_dict:
            median = gene_median_dict.get(ensembl, 1.0)
            mapped_var_indices.append(i)
            mapped_token_ids_list.append(token_dict[ensembl])
            mapped_medians_list.append(median)

    mapped_var_indices = np.array(mapped_var_indices)
    mapped_token_ids = np.array(mapped_token_ids_list)
    mapped_medians = np.array(mapped_medians_list)

    # HVG -> token mapping
    hvg_to_token = {}
    for g in hvg_genes:
        ensembl = None
        g_idx_in_var = np.where(var_genes == g)[0]
        if len(g_idx_in_var) > 0 and var_ensembl is not None:
            ensembl = var_ensembl[g_idx_in_var[0]]
        if ensembl is None or ensembl not in token_dict:
            ensembl = gene_name_id_dict.get(g)
        if ensembl and ensembl in token_dict:
            hvg_to_token[g] = token_dict[ensembl]
    token_to_hvg = {v: k for k, v in hvg_to_token.items()}

    print(f"  Mapped var genes: {len(mapped_var_indices)}/{n_genes_total}")
    print(f"  HVG genes with tokens: {len(hvg_to_token)}/{len(hvg_genes)}")

    # Tokenize
    control_tokens = []
    with h5py.File(data_path, 'r') as f:
        sparse = is_sparse_x(f)
        for ci, idx in enumerate(ctrl_indices):
            if sparse:
                expr = load_sparse_row(f, int(idx))
            else:
                expr = f['X'][int(idx), :].astype(np.float32)
            tokens = tokenize_cell(expr, mapped_var_indices, mapped_token_ids,
                                   mapped_medians, max_len=MAX_SEQ_LEN)
            if tokens is not None:
                control_tokens.append(tokens)
            if (ci + 1) % 500 == 0:
                print(f"    Tokenized {ci+1}/{len(ctrl_indices)}")

    print(f"  Tokenized: {len(control_tokens)} cells, avg {np.mean([len(t) for t in control_tokens]):.0f} tokens")
    print(f"  Step 3 time: {time.time()-t2:.1f}s")

    # ==================================================================
    # STEP 4: Extract attention edges
    # ==================================================================
    attn_cache_path = os.path.join(ds_out_dir, 'attention_edges.pkl')
    if ds_checkpoint.get('attention_complete') and os.path.exists(attn_cache_path):
        print("\n[STEP 4] Loading attention edges from cache...")
        with open(attn_cache_path, 'rb') as f:
            layer_attention_edges = pickle.load(f)
        for l in EVAL_LAYERS:
            if l in layer_attention_edges:
                print(f"    Layer {l+1}: {len(layer_attention_edges[l])} edges")
    else:
        print("\n[STEP 4] Extracting attention edges...")
        t3 = time.time()

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
                elapsed = time.time() - t3
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

        # Save attention edges
        with open(attn_cache_path, 'wb') as f:
            pickle.dump(layer_attention_edges, f)

        ds_checkpoint['attention_complete'] = True
        checkpoint[dataset_name] = ds_checkpoint
        save_checkpoint(checkpoint)
        print(f"  Extraction time: {time.time()-t3:.1f}s")

    # ==================================================================
    # STEP 5: Spearman correlation
    # ==================================================================
    print("\n[STEP 5] Computing Spearman correlation...")
    t4 = time.time()
    ranks = np.apply_along_axis(rankdata, 0, X_ctrl_hvg)
    corr_matrix = np.corrcoef(ranks.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    print(f"  Correlation matrix: {corr_matrix.shape}, time: {time.time()-t4:.1f}s")

    # ==================================================================
    # STEP 6: Perturbation-first evaluation (AUROC + AUPRC)
    # ==================================================================
    print("\n[STEP 6] Evaluating perturbation-first (AUROC + AUPRC)...")

    results_by_layer = {}

    for layer_idx in EVAL_LAYERS:
        attn_edges = layer_attention_edges[layer_idx]
        layer_aurocs_attn = []
        layer_aurocs_corr = []
        layer_auprcs_attn = []
        layer_auprcs_corr = []
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
                auprc_attn = average_precision_score(labels, np.array(scores_attn))
                auprc_corr = average_precision_score(labels, np.array(scores_corr))
            except ValueError:
                continue

            layer_aurocs_attn.append(auroc_attn)
            layer_aurocs_corr.append(auroc_corr)
            layer_auprcs_attn.append(auprc_attn)
            layer_auprcs_corr.append(auprc_corr)
            per_gene_results.append({
                'gene': pg,
                'auroc_attn': round(auroc_attn, 4),
                'auroc_corr': round(auroc_corr, 4),
                'auprc_attn': round(auprc_attn, 4),
                'auprc_corr': round(auprc_corr, 4),
                'n_de': int(labels.sum()),
                'n_total': int(len(labels)),
            })

        aurocs_attn = np.array(layer_aurocs_attn)
        aurocs_corr = np.array(layer_aurocs_corr)
        auprcs_attn = np.array(layer_auprcs_attn)
        auprcs_corr = np.array(layer_auprcs_corr)

        layer_result = {
            'layer_1indexed': layer_idx + 1,
            'n_evaluated': len(aurocs_attn),
            'attention': {
                'mean_auroc': round(float(aurocs_attn.mean()), 4) if len(aurocs_attn) > 0 else None,
                'std_auroc': round(float(aurocs_attn.std()), 4) if len(aurocs_attn) > 0 else None,
                'median_auroc': round(float(np.median(aurocs_attn)), 4) if len(aurocs_attn) > 0 else None,
                'mean_auprc': round(float(auprcs_attn.mean()), 4) if len(auprcs_attn) > 0 else None,
                'std_auprc': round(float(auprcs_attn.std()), 4) if len(auprcs_attn) > 0 else None,
            },
            'correlation': {
                'mean_auroc': round(float(aurocs_corr.mean()), 4) if len(aurocs_corr) > 0 else None,
                'std_auroc': round(float(aurocs_corr.std()), 4) if len(aurocs_corr) > 0 else None,
                'median_auroc': round(float(np.median(aurocs_corr)), 4) if len(aurocs_corr) > 0 else None,
                'mean_auprc': round(float(auprcs_corr.mean()), 4) if len(auprcs_corr) > 0 else None,
                'std_auprc': round(float(auprcs_corr.std()), 4) if len(auprcs_corr) > 0 else None,
            },
            'per_gene': per_gene_results,
        }

        if len(aurocs_attn) > 1:
            diffs = aurocs_attn - aurocs_corr
            diffs_prc = auprcs_attn - auprcs_corr
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
                'mean_diff_auroc': round(float(diffs.mean()), 4),
                'mean_diff_auprc': round(float(diffs_prc.mean()), 4),
                'cohens_d': round(cohens_d, 4),
                'n_attn_better': int((diffs > 0).sum()),
                'n_corr_better': int((diffs < 0).sum()),
                'ci_95_auroc': [round(float(np.percentile(diffs, 2.5)), 4),
                                round(float(np.percentile(diffs, 97.5)), 4)],
            }

            # Bootstrap CI for the difference
            boot_diffs = []
            rng = np.random.RandomState(42)
            for _ in range(1000):
                idx_b = rng.choice(len(diffs), len(diffs), replace=True)
                boot_diffs.append(diffs[idx_b].mean())
            boot_diffs = np.array(boot_diffs)
            layer_result['attn_vs_corr']['bootstrap_ci_95'] = [
                round(float(np.percentile(boot_diffs, 2.5)), 4),
                round(float(np.percentile(boot_diffs, 97.5)), 4),
            ]

        results_by_layer[layer_idx] = layer_result

        print(f"\n  Layer {layer_idx+1}:")
        print(f"    Evaluated: {len(aurocs_attn)} perturbations")
        if len(aurocs_attn) > 0:
            print(f"    Attention:   AUROC={aurocs_attn.mean():.4f} +/- {aurocs_attn.std():.4f}, "
                  f"AUPRC={auprcs_attn.mean():.4f}")
            print(f"    Correlation: AUROC={aurocs_corr.mean():.4f} +/- {aurocs_corr.std():.4f}, "
                  f"AUPRC={auprcs_corr.mean():.4f}")
            if 'attn_vs_corr' in layer_result:
                avc = layer_result['attn_vs_corr']
                print(f"    Diff: {avc['mean_diff_auroc']:+.4f}, Wilcoxon p={avc['wilcoxon_p']:.6f}, "
                      f"Cohen's d={avc['cohens_d']:.4f}")
                print(f"    Bootstrap 95% CI: [{avc['bootstrap_ci_95'][0]:+.4f}, {avc['bootstrap_ci_95'][1]:+.4f}]")

    # ==================================================================
    # STEP 7: Multi-threshold comparison
    # ==================================================================
    print("\n\n[STEP 7] Multi-threshold comparison...")
    threshold_comparison = []
    for lfc_thresh in LFC_THRESHOLDS:
        dt, dag = de_results_by_threshold[lfc_thresh]
        if len(dt) < 3:
            continue

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

    # ==================================================================
    # STEP 8: Power analysis
    # ==================================================================
    print("\n[STEP 8] Power analysis...")
    n_eval = len(de_targets)
    # Minimum detectable effect size at alpha=0.05, power=0.80 (two-sided paired t-test)
    # For paired t-test: d = t_crit * sqrt(2/n), simplified
    from scipy.stats import norm
    z_alpha = norm.ppf(0.975)  # 1.96
    z_beta = norm.ppf(0.80)   # 0.84
    min_detectable_d = (z_alpha + z_beta) / np.sqrt(n_eval)
    print(f"  n={n_eval} evaluable perturbations")
    print(f"  Minimum detectable Cohen's d (alpha=0.05, power=0.80): {min_detectable_d:.3f}")

    pr = results_by_layer.get(PRIMARY_LAYER, {})
    if pr.get('attn_vs_corr'):
        observed_d = abs(pr['attn_vs_corr']['cohens_d'])
        print(f"  Observed |d| at L{PRIMARY_LAYER+1}: {observed_d:.4f}")
        if observed_d < min_detectable_d:
            print(f"  NOTE: Observed effect below detection threshold -> null is interpretable")
        else:
            print(f"  NOTE: Observed effect above detection threshold -> result is powered")

    # ==================================================================
    # STEP 9: Save results
    # ==================================================================
    print("\n[STEP 9] Saving results...")

    summary = {
        'experiment': 'round13_non_k562_replication',
        'dataset': dataset_name,
        'cell_type': dataset_info['cell_type'],
        'perturbation_modality': dataset_info['perturbation_type'],
        'description': dataset_info['description'],
        'n_cells_total': int(n_cells_total),
        'n_control_cells_used': int(len(ctrl_indices)),
        'n_hvg': len(hvg_genes),
        'n_genes_with_tokens': len(hvg_to_token),
        'primary_lfc_threshold': primary_lfc,
        'n_evaluable_perturbations': len(de_targets),
        'evaluable_perturbations': sorted(de_targets.keys()),
        'eval_layers': [l + 1 for l in EVAL_LAYERS],
        'results_by_layer': {
            str(l): {k: v for k, v in r.items() if k != 'per_gene'}
            for l, r in results_by_layer.items()
        },
        'threshold_comparison': threshold_comparison,
        'power_analysis': {
            'n_evaluable': n_eval,
            'min_detectable_d': round(min_detectable_d, 4),
            'alpha': 0.05,
            'power': 0.80,
        },
        'total_time_seconds': round(time.time() - total_t0, 1),
    }

    with open(os.path.join(ds_out_dir, f'{dataset_name}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save per-gene CSVs
    for layer_idx in EVAL_LAYERS:
        if results_by_layer.get(layer_idx, {}).get('per_gene'):
            df = pd.DataFrame(results_by_layer[layer_idx]['per_gene'])
            df.to_csv(os.path.join(ds_out_dir, f'per_gene_L{layer_idx+1}_{dataset_name}.csv'), index=False)

    ds_checkpoint['evaluation_complete'] = True
    checkpoint[dataset_name] = ds_checkpoint
    save_checkpoint(checkpoint)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"{dataset_name} COMPLETE - Total: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")

    # Headline
    pr = results_by_layer.get(PRIMARY_LAYER, {})
    print(f"\nHEADLINE (L{PRIMARY_LAYER+1}, primary):")
    if pr.get('attention', {}).get('mean_auroc') is not None:
        print(f"  Attention:   AUROC={pr['attention']['mean_auroc']}, AUPRC={pr['attention']['mean_auprc']}")
        print(f"  Correlation: AUROC={pr['correlation']['mean_auroc']}, AUPRC={pr['correlation']['mean_auprc']}")
        if pr.get('attn_vs_corr'):
            avc = pr['attn_vs_corr']
            print(f"  Diff: {avc['mean_diff_auroc']:+.4f}, d={avc['cohens_d']}, p={avc['wilcoxon_p']}")
            print(f"  Bootstrap CI: {avc['bootstrap_ci_95']}")
            print(f"  Power: min_detectable_d={min_detectable_d:.4f}, observed_d={abs(avc['cohens_d']):.4f}")

    return summary


# ==========================================================================
# Combined figure
# ==========================================================================

def generate_combined_figure(all_summaries):
    """Generate multi-panel figure comparing all datasets."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid_summaries = [s for s in all_summaries if s and s.get('results_by_layer')]
    if not valid_summaries:
        print("  No valid results for figure generation")
        return

    n_datasets = len(valid_summaries)
    fig, axes = plt.subplots(n_datasets, 3, figsize=(16, 5 * n_datasets))
    if n_datasets == 1:
        axes = axes[np.newaxis, :]

    for di, summary in enumerate(valid_summaries):
        ds_name = summary['dataset']
        cell_type = summary['cell_type']
        n_eval = summary['n_evaluable_perturbations']
        lfc = summary['primary_lfc_threshold']
        results = summary['results_by_layer']

        # Panel A: Layer comparison
        ax = axes[di, 0]
        layers_1idx = [l + 1 for l in EVAL_LAYERS]
        attn_means = []
        corr_means = []
        for l in EVAL_LAYERS:
            r = results.get(str(l), {})
            attn_means.append(r.get('attention', {}).get('mean_auroc', 0) or 0)
            corr_means.append(r.get('correlation', {}).get('mean_auroc', 0) or 0)

        x = np.arange(len(EVAL_LAYERS))
        width = 0.3
        ax.bar(x - width/2, attn_means, width, label='Attention', color='#2196F3', alpha=0.8)
        ax.bar(x + width/2, corr_means, width, label='Correlation', color='#FF9800', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{l}' for l in layers_1idx])
        ax.set_ylabel('Mean AUROC')
        ax.set_title(f'A. {cell_type}\n(n={n_eval}, LFC>{lfc})')
        ax.legend(fontsize=8)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
        ax.set_ylim(0.4, 0.8)

        # Panel B: Per-perturbation scatter (primary layer)
        ax = axes[di, 1]
        ds_out_dir = os.path.join(OUT_DIR, ds_name)
        csv_path = os.path.join(ds_out_dir, f'per_gene_L{PRIMARY_LAYER+1}_{ds_name}.csv')
        if os.path.exists(csv_path):
            pg_df = pd.read_csv(csv_path)
            ax.scatter(pg_df['auroc_corr'], pg_df['auroc_attn'], alpha=0.5, s=30, color='#2196F3')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.set_xlabel('Correlation AUROC')
            ax.set_ylabel(f'Attention (L{PRIMARY_LAYER+1}) AUROC')
            ax.set_title(f'B. Per-Perturbation (n={len(pg_df)})')
            ax.set_aspect('equal')
            ax.set_xlim(0.2, 0.9)
            ax.set_ylim(0.2, 0.9)

        # Panel C: Effect size with CI
        ax = axes[di, 2]
        layer_diffs = []
        layer_cis = []
        layer_labels = []
        for l in EVAL_LAYERS:
            r = results.get(str(l), {})
            avc = r.get('attn_vs_corr', {})
            if avc:
                layer_diffs.append(avc.get('mean_diff_auroc', 0))
                ci = avc.get('bootstrap_ci_95', [0, 0])
                layer_cis.append(ci)
                layer_labels.append(f'L{l+1}')

        if layer_diffs:
            y = np.arange(len(layer_diffs))
            errors = np.array([[d - ci[0], ci[1] - d] for d, ci in zip(layer_diffs, layer_cis)]).T
            ax.barh(y, layer_diffs, xerr=errors, color='#4CAF50', alpha=0.7, capsize=4)
            ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_yticks(y)
            ax.set_yticklabels(layer_labels)
            ax.set_xlabel('AUROC Diff (Attn - Corr)')
            ax.set_title(f'C. Effect Size + 95% CI')

        # Add power annotation
        pa = summary.get('power_analysis', {})
        if pa:
            ax.text(0.02, 0.02,
                    f"Min detectable d={pa['min_detectable_d']:.3f}\n({pa['n_evaluable']} perts, "
                    f"alpha={pa['alpha']}, power={pa['power']})",
                    transform=ax.transAxes, fontsize=7, va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_non_k562_replication.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nCombined figure saved: {fig_path}")


# ==========================================================================
# Main
# ==========================================================================

def run():
    total_t0 = time.time()

    print("=" * 70)
    print("ROUND 13: NON-K562 PERTURBATION-FIRST REPLICATION")
    print("=" * 70)
    print(f"Datasets: {list(DATASETS.keys())}")

    # Load token dictionaries
    print("\nLoading Geneformer token dictionaries...")
    with open(os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        token_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl"), 'rb') as f:
        gene_median_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_name_id_dict_gc104M.pkl"), 'rb') as f:
        gene_name_id_dict = pickle.load(f)
    print(f"  token_dict: {len(token_dict)} entries")
    print(f"  gene_median_dict: {len(gene_median_dict)} entries")
    print(f"  gene_name_id_dict: {len(gene_name_id_dict)} entries")

    checkpoint = load_checkpoint()
    all_summaries = []

    for dataset_name, dataset_info in DATASETS.items():
        if checkpoint.get(dataset_name, {}).get('evaluation_complete'):
            print(f"\n{'=' * 70}")
            print(f"SKIP: {dataset_name} — already complete (loading cached results)")
            summary_path = os.path.join(OUT_DIR, dataset_name, f'{dataset_name}_summary.json')
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    summary = json.load(f)
                all_summaries.append(summary)

                # Print headline
                pr = summary['results_by_layer'].get(str(PRIMARY_LAYER), {})
                if pr.get('attention', {}).get('mean_auroc') is not None:
                    print(f"  HEADLINE (L{PRIMARY_LAYER+1}): "
                          f"Attn AUROC={pr['attention']['mean_auroc']}, "
                          f"Corr AUROC={pr['correlation']['mean_auroc']}")
            continue

        try:
            summary = run_dataset(dataset_name, dataset_info, token_dict,
                                  gene_median_dict, gene_name_id_dict, checkpoint)
            if summary:
                all_summaries.append(summary)
        except Exception as e:
            print(f"\n  ERROR in {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate combined figure
    print("\n\nGenerating combined figure...")
    generate_combined_figure(all_summaries)

    # Combined summary
    combined_path = os.path.join(OUT_DIR, 'combined_summary.json')
    combined = {
        'experiment': 'round13_non_k562_replication',
        'n_datasets': len(all_summaries),
        'datasets': [],
    }
    for s in all_summaries:
        ds_summary = {
            'name': s['dataset'],
            'cell_type': s['cell_type'],
            'n_evaluable': s['n_evaluable_perturbations'],
            'primary_layer': PRIMARY_LAYER + 1,
        }
        pr = s['results_by_layer'].get(str(PRIMARY_LAYER), {})
        if pr.get('attention', {}).get('mean_auroc') is not None:
            ds_summary['attn_auroc'] = pr['attention']['mean_auroc']
            ds_summary['corr_auroc'] = pr['correlation']['mean_auroc']
            avc = pr.get('attn_vs_corr', {})
            ds_summary['diff'] = avc.get('mean_diff_auroc', None)
            ds_summary['cohens_d'] = avc.get('cohens_d', None)
            ds_summary['wilcoxon_p'] = avc.get('wilcoxon_p', None)
            ds_summary['bootstrap_ci'] = avc.get('bootstrap_ci_95', None)
        ds_summary['power'] = s.get('power_analysis', {})
        combined['datasets'].append(ds_summary)

    combined['total_time_seconds'] = round(time.time() - total_t0, 1)

    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"ALL DATASETS COMPLETE - Total: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")

    print("\nCOMBINED HEADLINE:")
    for s in all_summaries:
        pr = s['results_by_layer'].get(str(PRIMARY_LAYER), {})
        if pr.get('attention', {}).get('mean_auroc') is not None:
            avc = pr.get('attn_vs_corr', {})
            print(f"  {s['dataset']} ({s['cell_type']}, n={s['n_evaluable_perturbations']}):")
            print(f"    Attn={pr['attention']['mean_auroc']}, Corr={pr['correlation']['mean_auroc']}, "
                  f"d={avc.get('cohens_d', 'N/A')}, p={avc.get('wilcoxon_p', 'N/A')}")


if __name__ == '__main__':
    run()
