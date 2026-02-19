#!/usr/bin/env python3
"""
Round 9 Head-Level Causal Ablation (MC5 + ST5)
================================================
Tests whether Geneformer attention heads that recover regulatory network
structure (TRRUST) are also causally important for perturbation-outcome
prediction. Uses BERT's head_mask parameter to zero out specific heads
during the forward pass, then evaluates perturbation-first AUROC.

Conditions:
  1. Baseline: full model (no ablation)
  2. Ablate top-5 regulatory heads (highest TRRUST AUROC)
  3. Ablate 5 random heads (×3 repeats, matched layer distribution)
  4. Ablate 5 entropy-matched heads (similar entropy, low TRRUST)

Depends on: round9_unified_extraction outputs (per-head TRRUST AUROCs,
DE labels, HVG gene list, correlation matrix).
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
from multiprocessing import Pool

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
DATA_PATH = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data", "replogle_concat.h5ad")
EXTRACTION_DIR = os.path.join(PAPER_DIR, "experiments", "round9_unified_extraction")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round9_head_ablation")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data")

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

N_CTRL = 2000
N_HVG = 2000
LFC_THRESHOLD = 0.5
PVAL_THRESHOLD = 0.05
MIN_DE_TARGETS = 5
N_LAYERS = 18
N_HEADS = 18
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1

# Only extract from L13 (primary evaluation layer) for ablation comparison
EXTRACT_LAYER = 12  # 0-indexed

# Ablation parameters
N_ABLATE = 5
N_RANDOM_REPEATS = 3


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


def evaluate_perturbation_first(attn_edges, corr_matrix, de_labels, hvg_genes,
                                 hvg_to_local, hvg_upper, min_de=5):
    """Evaluate perturbation-first AUROC for attention and correlation edges."""
    aurocs_attn = []
    aurocs_corr = []
    per_gene = []
    rng = np.random.RandomState(42)

    for pg, targets_dict in de_labels.items():
        pg_local = hvg_to_local.get(pg.upper())
        if pg_local is None:
            continue

        labels = []
        scores_attn = []
        scores_corr = []

        for tgt_gene, is_de in targets_dict.items():
            tgt_local = hvg_to_local.get(tgt_gene)
            if tgt_local is None:
                continue

            tgt_orig = hvg_genes[tgt_local]
            fwd = attn_edges.get((pg, tgt_orig), 0.0)
            rev = attn_edges.get((tgt_orig, pg), 0.0)
            attn_score = max(fwd, rev)

            corr_score = abs(corr_matrix[pg_local, tgt_local])

            labels.append(1 if is_de else 0)
            scores_attn.append(attn_score)
            scores_corr.append(corr_score)

        labels = np.array(labels)
        if labels.sum() < min_de or labels.sum() == len(labels):
            continue

        try:
            auroc_attn = roc_auc_score(labels, np.array(scores_attn))
            auroc_corr = roc_auc_score(labels, np.array(scores_corr))
        except ValueError:
            continue

        aurocs_attn.append(auroc_attn)
        aurocs_corr.append(auroc_corr)
        per_gene.append({
            'gene': pg,
            'auroc_attn': round(auroc_attn, 4),
            'auroc_corr': round(auroc_corr, 4),
            'n_de': int(labels.sum()),
            'n_total': int(len(labels)),
        })

    return np.array(aurocs_attn), np.array(aurocs_corr), per_gene


def extract_attention_with_ablation(model, control_tokens, head_mask_tensor,
                                     token_to_hvg, hvg_to_token, device,
                                     extract_layer=12, hvg_genes=None):
    """Run forward pass with head_mask and extract attention from extract_layer.

    Uses numpy matrix accumulation (not dict) for speed.
    When heads are ablated (head_mask=0), we average attention only over
    non-ablated heads in the extract layer.
    """
    import torch

    n_hvg = len(hvg_genes) if hvg_genes is not None else N_HVG
    hvg_name_to_idx = {g: i for i, g in enumerate(hvg_genes)} if hvg_genes is not None else {}

    captured_attentions = {}
    orig_forward = model.bert.encoder.layer[extract_layer].forward

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

    model.bert.encoder.layer[extract_layer].forward = make_patched_forward(
        extract_layer, captured_attentions, orig_forward)

    # Determine which heads in extract_layer are active
    active_heads_mask = head_mask_tensor[extract_layer].cpu().numpy()
    active_head_indices = np.where(active_heads_mask > 0.5)[0]
    n_active = len(active_head_indices)

    # Numpy matrix accumulators instead of dict
    attn_sum = np.zeros((n_hvg, n_hvg), dtype=np.float64)
    attn_count = np.zeros((n_hvg, n_hvg), dtype=np.float64)

    sorted_indices = sorted(range(len(control_tokens)), key=lambda i: len(control_tokens[i]))

    # Per-head entropy (vectorized)
    head_entropy_sums = np.zeros(N_HEADS)
    head_entropy_counts = np.zeros(N_HEADS)

    cells_processed = 0
    t0 = time.time()

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
            _ = model(input_ids=input_ids_t, attention_mask=attn_mask_t,
                      head_mask=head_mask_tensor)

        if extract_layer in captured_attentions:
            full_attn = captured_attentions[extract_layer].cpu().float().numpy()
            # full_attn shape: (bs, n_heads, seq_len, seq_len)

            # Average only over active heads
            if n_active > 0:
                layer_attn = full_attn[:, active_head_indices, :, :].mean(axis=1)
            else:
                layer_attn = full_attn.mean(axis=1)

            for cell_i in range(bs):
                seq_len = int(attention_mask[cell_i].sum())
                cell_tok = input_ids[cell_i, :seq_len]
                cell_attn = layer_attn[cell_i, :seq_len, :seq_len]
                gene_tokens = cell_tok[1:seq_len - 1]

                # Build HVG index arrays for vectorized accumulation
                pos_list = []
                idx_list = []
                for pos_offset, tok_id in enumerate(gene_tokens):
                    gene = token_to_hvg.get(int(tok_id))
                    if gene is not None:
                        hvg_idx = hvg_name_to_idx.get(gene)
                        if hvg_idx is not None:
                            pos_list.append(pos_offset + 1)  # +1 for CLS token
                            idx_list.append(hvg_idx)

                if len(pos_list) < 2:
                    continue

                pos_arr = np.array(pos_list)
                idx_arr = np.array(idx_list)

                # Extract sub-matrix and accumulate with += (indices unique per cell)
                sub_attn = cell_attn[np.ix_(pos_arr, pos_arr)]
                attn_sum[np.ix_(idx_arr, idx_arr)] += sub_attn
                attn_count[np.ix_(idx_arr, idx_arr)] += 1.0

                # Vectorized per-head entropy
                for h in active_head_indices:
                    head_attn = full_attn[cell_i, h, 1:seq_len-1, 1:seq_len-1]
                    # Normalize rows
                    row_sums = head_attn.sum(axis=1, keepdims=True)
                    row_sums = np.maximum(row_sums, 1e-12)
                    probs = head_attn / row_sums
                    # Entropy per row, then average
                    log_probs = np.log(np.maximum(probs, 1e-12))
                    row_entropies = -np.sum(probs * log_probs, axis=1)
                    head_entropy_sums[h] += float(row_entropies.mean())
                    head_entropy_counts[h] += 1

            del full_attn, layer_attn

        cells_processed += bs
        captured_attentions.clear()
        if device.type == 'mps':
            torch.mps.empty_cache()

        if cells_processed % 100 == 0 or cells_processed >= len(control_tokens):
            elapsed = time.time() - t0
            rate = cells_processed / elapsed if elapsed > 0 else 0
            eta = (len(control_tokens) - cells_processed) / rate if rate > 0 else 0
            print(f"      Cells: {cells_processed}/{len(control_tokens)}, "
                  f"rate: {rate:.1f}/s, ETA: {eta:.0f}s")

    # Restore
    model.bert.encoder.layer[extract_layer].forward = orig_forward
    captured_attentions.clear()
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()

    # Convert matrix to edge dict (averaging)
    mask = attn_count > 0
    avg_matrix = np.zeros_like(attn_sum)
    avg_matrix[mask] = attn_sum[mask] / attn_count[mask]

    edges = {}
    nonzero_pairs = np.argwhere(mask)
    for r, c in nonzero_pairs:
        if r != c:
            edges[(hvg_genes[r], hvg_genes[c])] = float(avg_matrix[r, c])

    # Average entropy
    head_entropy = {}
    for h in range(N_HEADS):
        if head_entropy_counts[h] > 0:
            head_entropy[h] = head_entropy_sums[h] / head_entropy_counts[h]
        else:
            head_entropy[h] = 0.0

    return edges, head_entropy


def run():
    import torch
    from transformers import BertForMaskedLM

    total_t0 = time.time()

    print("=" * 70)
    print("ROUND 9 HEAD-LEVEL CAUSAL ABLATION")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ============================================================
    # STEP 1: Load pre-computed data from unified extraction
    # ============================================================
    print("\nSTEP 1: Loading unified extraction outputs...")

    # Per-head TRRUST AUROCs
    auroc_matrix = np.load(os.path.join(EXTRACTION_DIR, "per_head_trrust", "head_trrust_auroc_matrix.npy"))
    print(f"  Head AUROC matrix: {auroc_matrix.shape}")
    print(f"  AUROC range: [{auroc_matrix.min():.4f}, {auroc_matrix.max():.4f}]")

    with open(os.path.join(EXTRACTION_DIR, "per_head_trrust", "top5_regulatory_heads.json")) as f:
        top5_list = json.load(f)
    top5_heads = [(h['layer'], h['head']) for h in top5_list]
    print(f"  Top-5 regulatory heads: {[(l,h) for l,h in top5_heads]}")
    for h in top5_list:
        print(f"    L{h['layer']}_H{h['head']}: AUROC = {h['auroc']:.4f}")

    # DE labels
    with open(os.path.join(EXTRACTION_DIR, "de_labels.pkl"), 'rb') as f:
        de_labels = pickle.load(f)
    print(f"  DE labels: {len(de_labels)} perturbations")

    # HVG genes
    with open(os.path.join(EXTRACTION_DIR, "hvg_gene_list.json")) as f:
        hvg_genes = json.load(f)
    hvg_upper = [g.upper() for g in hvg_genes]
    hvg_to_local = {g.upper(): i for i, g in enumerate(hvg_genes)}

    # Correlation matrix
    corr_matrix = np.load(os.path.join(EXTRACTION_DIR, "correlation_matrix.npy"))
    print(f"  Correlation matrix: {corr_matrix.shape}")

    # ============================================================
    # STEP 2: Load data and tokenize (same as unified extraction)
    # ============================================================
    print("\nSTEP 2: Loading data and tokenizing control cells...")
    t1 = time.time()

    with h5py.File(DATA_PATH, 'r') as f:
        var_genes = load_categorical_column(f['var'], 'gene_name_index')
        n_cells_total, n_genes_total = f['X'].shape

    # Load tokenization dicts
    with open(os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        token_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl"), 'rb') as f:
        gene_median_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_name_id_dict_gc104M.pkl"), 'rb') as f:
        gene_name_id_dict = pickle.load(f)

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

    hvg_to_token = {}
    for g in hvg_genes:
        ens = gene_name_id_dict.get(g)
        if ens and ens in token_dict:
            hvg_to_token[g] = token_dict[ens]
    token_to_hvg = {v: k for k, v in hvg_to_token.items()}

    # Load and tokenize control cells (same seed/indices as unified extraction)
    cell_genes = None
    with h5py.File(DATA_PATH, 'r') as f:
        cell_genes_raw = load_categorical_column(f['obs'], 'gene')

    control_mask = np.zeros(len(cell_genes_raw), dtype=bool)
    for ctrl_name in ['non-targeting', 'Non-targeting', 'non_targeting']:
        control_mask |= (cell_genes_raw == ctrl_name)
    ctrl_indices = np.where(control_mask)[0]

    np.random.seed(42)
    if len(ctrl_indices) > N_CTRL:
        ctrl_indices = np.random.choice(ctrl_indices, N_CTRL, replace=False)
        ctrl_indices.sort()

    print(f"  Tokenizing {len(ctrl_indices)} control cells...")
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
    # STEP 3: Set up model and ablation conditions
    # ============================================================
    print("\nSTEP 3: Setting up model and ablation conditions...")

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
        attn_implementation="eager",  # avoid SDPA fallback warning
    )
    model = model.to(device)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # --- Define ablation conditions ---
    conditions = {}

    # Condition 1: Baseline (all heads active)
    baseline_mask = torch.ones(N_LAYERS, N_HEADS, device=device)
    conditions['baseline'] = {
        'mask': baseline_mask,
        'description': 'Full model (no ablation)',
        'ablated_heads': [],
    }

    # Condition 2: Ablate top-5 regulatory heads
    regulatory_mask = torch.ones(N_LAYERS, N_HEADS, device=device)
    for l, h in top5_heads:
        regulatory_mask[l, h] = 0.0
    conditions['ablate_regulatory'] = {
        'mask': regulatory_mask,
        'description': f'Ablate top-5 regulatory heads ({top5_heads})',
        'ablated_heads': top5_heads,
    }

    # Condition 3: Ablate 5 random heads (×3, matched layer distribution)
    # Match the layer distribution of top-5 regulatory heads
    reg_layer_counts = {}
    for l, h in top5_heads:
        reg_layer_counts[l] = reg_layer_counts.get(l, 0) + 1

    rng = np.random.RandomState(123)
    for rep in range(N_RANDOM_REPEATS):
        random_mask = torch.ones(N_LAYERS, N_HEADS, device=device)
        random_heads = []

        # Match layer distribution
        for layer, count in reg_layer_counts.items():
            available = [h for h in range(N_HEADS) if (layer, h) not in top5_heads]
            chosen = rng.choice(available, size=min(count, len(available)), replace=False)
            for h in chosen:
                random_mask[layer, int(h)] = 0.0
                random_heads.append((layer, int(h)))

        conditions[f'ablate_random_{rep+1}'] = {
            'mask': random_mask,
            'description': f'Ablate 5 random heads (rep {rep+1}), matched layers',
            'ablated_heads': random_heads,
        }

    # Condition 4: Ablate 5 entropy-matched heads (need baseline entropy first)
    # We'll add this after running baseline to get entropy data
    # For now, prepare a placeholder

    print(f"  Defined {len(conditions)} ablation conditions (entropy-matched added after baseline)")

    # ============================================================
    # STEP 4: Run ablation conditions
    # ============================================================
    print("\nSTEP 4: Running ablation conditions...")

    results = {}
    baseline_entropy = None

    # Run baseline first to get head entropy
    condition_order = ['baseline', 'ablate_regulatory'] + \
                      [f'ablate_random_{r+1}' for r in range(N_RANDOM_REPEATS)]

    for cond_name in condition_order:
        cond = conditions[cond_name]
        print(f"\n  --- Condition: {cond_name} ---")
        print(f"  {cond['description']}")
        t_cond = time.time()

        edges, head_entropy = extract_attention_with_ablation(
            model, control_tokens, cond['mask'],
            token_to_hvg, hvg_to_token, device,
            extract_layer=EXTRACT_LAYER,
            hvg_genes=hvg_genes,
        )

        print(f"    Edges extracted: {len(edges)}")
        print(f"    Extraction time: {time.time()-t_cond:.0f}s")

        # Save baseline entropy for entropy-matched condition
        if cond_name == 'baseline':
            baseline_entropy = head_entropy
            print(f"    Head entropy range: [{min(head_entropy.values()):.3f}, {max(head_entropy.values()):.3f}]")

        # Evaluate perturbation-first AUROC
        aurocs_attn, aurocs_corr, per_gene = evaluate_perturbation_first(
            edges, corr_matrix, de_labels, hvg_genes, hvg_to_local, hvg_upper,
            min_de=MIN_DE_TARGETS,
        )

        results[cond_name] = {
            'description': cond['description'],
            'ablated_heads': [(l, h) for l, h in cond['ablated_heads']],
            'n_evaluated': len(aurocs_attn),
            'mean_auroc_attn': round(float(aurocs_attn.mean()), 4) if len(aurocs_attn) > 0 else None,
            'std_auroc_attn': round(float(aurocs_attn.std()), 4) if len(aurocs_attn) > 0 else None,
            'median_auroc_attn': round(float(np.median(aurocs_attn)), 4) if len(aurocs_attn) > 0 else None,
            'mean_auroc_corr': round(float(aurocs_corr.mean()), 4) if len(aurocs_corr) > 0 else None,
            'aurocs_attn': aurocs_attn.tolist(),
            'per_gene': per_gene,
        }

        print(f"    AUROC (attn): {results[cond_name]['mean_auroc_attn']:.4f} ± {results[cond_name]['std_auroc_attn']:.4f}")
        print(f"    AUROC (corr): {results[cond_name]['mean_auroc_corr']:.4f}")
        print(f"    Evaluated: {len(aurocs_attn)} perturbations")
        print(f"    Condition time: {time.time()-t_cond:.0f}s")

    # --- Now define and run entropy-matched condition ---
    if baseline_entropy is not None:
        print("\n  --- Defining entropy-matched condition ---")
        # Get entropy of top-5 regulatory heads
        reg_entropies = [baseline_entropy.get(h, 0.0) for l, h in top5_heads]
        mean_reg_entropy = np.mean(reg_entropies)
        print(f"    Top-5 regulatory head mean entropy: {mean_reg_entropy:.4f}")

        # Find non-regulatory heads and rank by entropy similarity
        all_heads = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]
        non_reg_heads = [lh for lh in all_heads if lh not in top5_heads]

        # For entropy-matching, we need per-layer-head entropy
        # We only have per-head entropy from L13 (EXTRACT_LAYER)
        # Use head-level entropy from baseline as proxy
        head_scores = []
        for l, h in non_reg_heads:
            h_entropy = baseline_entropy.get(h, 0.0)
            h_auroc = auroc_matrix[l, h]
            entropy_diff = abs(h_entropy - mean_reg_entropy)
            head_scores.append((l, h, h_entropy, h_auroc, entropy_diff))

        # Sort by: low TRRUST AUROC first (bottom 50%), then closest entropy
        # Filter to bottom half of TRRUST AUROC
        auroc_threshold = np.median(auroc_matrix)
        low_auroc_heads = [hs for hs in head_scores if hs[3] <= auroc_threshold]
        if len(low_auroc_heads) < N_ABLATE:
            low_auroc_heads = head_scores  # fallback

        # Sort by entropy similarity
        low_auroc_heads.sort(key=lambda x: x[4])
        entropy_matched_heads = [(l, h) for l, h, _, _, _ in low_auroc_heads[:N_ABLATE]]

        entropy_mask = torch.ones(N_LAYERS, N_HEADS, device=device)
        for l, h in entropy_matched_heads:
            entropy_mask[l, h] = 0.0

        print(f"    Entropy-matched heads: {entropy_matched_heads}")
        for l, h in entropy_matched_heads:
            print(f"      L{l}_H{h}: AUROC={auroc_matrix[l,h]:.4f}, entropy={baseline_entropy.get(h, 0):.4f}")

        t_cond = time.time()
        edges, _ = extract_attention_with_ablation(
            model, control_tokens, entropy_mask,
            token_to_hvg, hvg_to_token, device,
            extract_layer=EXTRACT_LAYER,
            hvg_genes=hvg_genes,
        )

        aurocs_attn, aurocs_corr, per_gene = evaluate_perturbation_first(
            edges, corr_matrix, de_labels, hvg_genes, hvg_to_local, hvg_upper,
            min_de=MIN_DE_TARGETS,
        )

        results['ablate_entropy_matched'] = {
            'description': f'Ablate 5 entropy-matched heads (similar entropy, low TRRUST)',
            'ablated_heads': entropy_matched_heads,
            'n_evaluated': len(aurocs_attn),
            'mean_auroc_attn': round(float(aurocs_attn.mean()), 4) if len(aurocs_attn) > 0 else None,
            'std_auroc_attn': round(float(aurocs_attn.std()), 4) if len(aurocs_attn) > 0 else None,
            'median_auroc_attn': round(float(np.median(aurocs_attn)), 4) if len(aurocs_attn) > 0 else None,
            'mean_auroc_corr': round(float(aurocs_corr.mean()), 4) if len(aurocs_corr) > 0 else None,
            'aurocs_attn': aurocs_attn.tolist(),
            'per_gene': per_gene,
        }

        print(f"    AUROC (attn): {results['ablate_entropy_matched']['mean_auroc_attn']:.4f}")
        print(f"    Condition time: {time.time()-t_cond:.0f}s")

    # Free model
    del model
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()

    # ============================================================
    # STEP 5: Statistical comparisons
    # ============================================================
    print("\n\nSTEP 5: Statistical comparisons...")

    baseline_aurocs = np.array(results['baseline']['aurocs_attn'])
    comparisons = {}

    for cond_name in results:
        if cond_name == 'baseline':
            continue

        cond_aurocs = np.array(results[cond_name]['aurocs_attn'])

        # Ensure same length (same perturbations evaluated)
        n = min(len(baseline_aurocs), len(cond_aurocs))
        if n < 10:
            continue

        b = baseline_aurocs[:n]
        c = cond_aurocs[:n]
        diff = b - c  # positive means baseline better

        try:
            w_stat, w_p = stats.wilcoxon(diff)
        except:
            w_p = 1.0
        try:
            t_stat, t_p = stats.ttest_rel(b, c)
        except:
            t_p = 1.0

        comp = {
            'n_compared': int(n),
            'mean_diff': round(float(diff.mean()), 4),
            'std_diff': round(float(diff.std()), 4),
            'wilcoxon_p': round(float(w_p), 6),
            'ttest_p': round(float(t_p), 6),
            'n_baseline_better': int((diff > 0).sum()),
            'n_ablation_better': int((diff < 0).sum()),
            'n_tied': int((diff == 0).sum()),
        }
        comparisons[f'baseline_vs_{cond_name}'] = comp

        print(f"\n  Baseline vs {cond_name}:")
        print(f"    ΔAUROC = {comp['mean_diff']:.4f} ± {comp['std_diff']:.4f}")
        print(f"    Wilcoxon p = {comp['wilcoxon_p']:.6f}")
        print(f"    Baseline better: {comp['n_baseline_better']}/{comp['n_compared']}")

    # Average across random repeats
    random_aurocs = []
    for r in range(N_RANDOM_REPEATS):
        key = f'ablate_random_{r+1}'
        if key in results:
            random_aurocs.append(results[key]['mean_auroc_attn'])
    if random_aurocs:
        mean_random = np.mean(random_aurocs)
        print(f"\n  Average random ablation AUROC: {mean_random:.4f}")

    # Key comparison: regulatory ablation drop vs random ablation drop
    if 'ablate_regulatory' in results and random_aurocs:
        reg_drop = results['baseline']['mean_auroc_attn'] - results['ablate_regulatory']['mean_auroc_attn']
        rand_drops = [results['baseline']['mean_auroc_attn'] - ra for ra in random_aurocs]
        mean_rand_drop = np.mean(rand_drops)

        print(f"\n  Regulatory ablation drop: {reg_drop:.4f}")
        print(f"  Random ablation drop (mean of {N_RANDOM_REPEATS}): {mean_rand_drop:.4f}")
        print(f"  Excess drop (regulatory - random): {reg_drop - mean_rand_drop:.4f}")

        comparisons['regulatory_vs_random_drop'] = {
            'regulatory_drop': round(reg_drop, 4),
            'random_drops': [round(d, 4) for d in rand_drops],
            'mean_random_drop': round(mean_rand_drop, 4),
            'excess_drop': round(reg_drop - mean_rand_drop, 4),
        }

    # ============================================================
    # STEP 6: Generate figure
    # ============================================================
    print("\n\nSTEP 6: Generating figure...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Head AUROC heatmap (18×18)
    ax = axes[0, 0]
    im = ax.imshow(auroc_matrix, aspect='auto', cmap='RdYlBu_r', vmin=0.35, vmax=0.75)
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title('Per-Head TRRUST AUROC (18×18)')
    plt.colorbar(im, ax=ax, shrink=0.8)
    # Mark top-5 regulatory heads
    for l, h in top5_heads:
        ax.plot(h, l, 'k*', markersize=12, markeredgewidth=1.5)

    # Panel B: Ablation bar chart
    ax = axes[0, 1]
    cond_names = ['baseline', 'ablate_regulatory']
    cond_names += [f'ablate_random_{r+1}' for r in range(N_RANDOM_REPEATS)]
    if 'ablate_entropy_matched' in results:
        cond_names.append('ablate_entropy_matched')

    bar_labels = ['Baseline', 'Ablate\nRegulatory']
    bar_labels += [f'Ablate\nRandom {r+1}' for r in range(N_RANDOM_REPEATS)]
    if 'ablate_entropy_matched' in results:
        bar_labels.append('Ablate\nEntropy-matched')

    bar_means = [results[c]['mean_auroc_attn'] for c in cond_names]
    bar_stds = [results[c]['std_auroc_attn'] for c in cond_names]
    colors = ['#2ecc71', '#e74c3c'] + ['#3498db'] * N_RANDOM_REPEATS
    if 'ablate_entropy_matched' in results:
        colors.append('#f39c12')

    x_pos = np.arange(len(cond_names))
    ax.bar(x_pos, bar_means, yerr=bar_stds, color=colors, alpha=0.8, capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylabel('Mean Perturbation-First AUROC')
    ax.set_title('Head Ablation: Impact on Perturbation Prediction')
    # Reference line for correlation
    corr_auroc = results['baseline']['mean_auroc_corr']
    ax.axhline(y=corr_auroc, color='gray', linestyle='--', alpha=0.7, label=f'Correlation ({corr_auroc:.3f})')
    ax.legend(fontsize=8)

    # Panel C: Per-perturbation scatter (baseline vs regulatory ablation)
    ax = axes[1, 0]
    if 'ablate_regulatory' in results and 'baseline' in results:
        base_pg = {g['gene']: g['auroc_attn'] for g in results['baseline']['per_gene']}
        reg_pg = {g['gene']: g['auroc_attn'] for g in results['ablate_regulatory']['per_gene']}
        common_genes = sorted(set(base_pg.keys()) & set(reg_pg.keys()))
        if common_genes:
            x_vals = [base_pg[g] for g in common_genes]
            y_vals = [reg_pg[g] for g in common_genes]
            ax.scatter(x_vals, y_vals, alpha=0.4, s=20, c='steelblue')
            lims = [min(min(x_vals), min(y_vals)) - 0.02, max(max(x_vals), max(y_vals)) + 0.02]
            ax.plot(lims, lims, 'k--', alpha=0.5)
            ax.set_xlabel('Baseline AUROC')
            ax.set_ylabel('Regulatory Ablation AUROC')
            ax.set_title('Per-Perturbation: Baseline vs Regulatory Ablation')

    # Panel D: ΔAUROC distribution
    ax = axes[1, 1]
    if 'ablate_regulatory' in results:
        n = min(len(results['baseline']['aurocs_attn']), len(results['ablate_regulatory']['aurocs_attn']))
        delta_reg = np.array(results['baseline']['aurocs_attn'][:n]) - np.array(results['ablate_regulatory']['aurocs_attn'][:n])
        ax.hist(delta_reg, bins=30, alpha=0.6, color='#e74c3c', label='Regulatory ablation', density=True)

        for r in range(N_RANDOM_REPEATS):
            key = f'ablate_random_{r+1}'
            if key in results:
                nr = min(len(results['baseline']['aurocs_attn']), len(results[key]['aurocs_attn']))
                delta_rand = np.array(results['baseline']['aurocs_attn'][:nr]) - np.array(results[key]['aurocs_attn'][:nr])
                ax.hist(delta_rand, bins=30, alpha=0.3, color='#3498db',
                        label=f'Random {r+1}' if r == 0 else None, density=True)

        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('ΔAUROC (Baseline − Ablation)')
        ax.set_ylabel('Density')
        ax.set_title('ΔAUROC Distribution')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_dir = os.path.join(PAPER_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "fig_head_ablation.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")

    # ============================================================
    # STEP 7: Save results
    # ============================================================
    print("\nSTEP 7: Saving results...")

    # Remove raw aurocs arrays for JSON serialization (keep per_gene)
    results_clean = {}
    for cond_name, cond_results in results.items():
        r = dict(cond_results)
        del r['aurocs_attn']  # large array, not needed in summary
        results_clean[cond_name] = r

    summary = {
        'experiment': 'round9_head_ablation',
        'extract_layer': EXTRACT_LAYER,
        'n_ablate': N_ABLATE,
        'n_random_repeats': N_RANDOM_REPEATS,
        'top5_regulatory_heads': top5_list,
        'conditions': results_clean,
        'comparisons': comparisons,
        'total_time_seconds': round(time.time() - total_t0, 1),
    }

    with open(os.path.join(OUT_DIR, "ablation_results.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save head AUROC data for paper
    with open(os.path.join(OUT_DIR, "head_trrust_aurocs.json"), 'w') as f:
        json.dump({f"L{l}_H{h}": round(float(auroc_matrix[l, h]), 6)
                   for l in range(N_LAYERS) for h in range(N_HEADS)}, f, indent=2)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"HEAD ABLATION COMPLETE - Total: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")

    print(f"\nKey Results:")
    for cond_name in ['baseline', 'ablate_regulatory'] + \
                     [f'ablate_random_{r+1}' for r in range(N_RANDOM_REPEATS)] + \
                     (['ablate_entropy_matched'] if 'ablate_entropy_matched' in results else []):
        r = results[cond_name]
        print(f"  {cond_name}: AUROC = {r['mean_auroc_attn']:.4f}")

    if 'regulatory_vs_random_drop' in comparisons:
        c = comparisons['regulatory_vs_random_drop']
        print(f"\n  Regulatory excess drop: {c['excess_drop']:.4f}")

    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == '__main__':
    run()
