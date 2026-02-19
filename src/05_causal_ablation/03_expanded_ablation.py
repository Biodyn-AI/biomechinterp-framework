#!/usr/bin/env python3
"""
Round 10 Experiment 3: Expanded Causal Ablations
=================================================
Broadens ablation from 6 to ~17 conditions: dose-response (k=5/10/20/50),
alternative ranking (composite TRRUST×layer_AUROC), per-layer ablation,
and matched random controls.

Reuses R9 infrastructure and pre-computed data. Resumes from checkpoint.

Addresses: MC4 ("causal ablation too narrow")
           ST4 ("expanded causal ablations")
"""

import os, sys, gc, json, time, pickle, warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import pandas as pd
import h5py
from scipy import stats
from sklearn.metrics import roc_auc_score

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
DATA_PATH = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data", "replogle_concat.h5ad")
EXTRACTION_DIR = os.path.join(PAPER_DIR, "experiments", "round9_unified_extraction")
R9_DIR = os.path.join(PAPER_DIR, "experiments", "round9_head_ablation")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round10_expanded_ablation")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

N_CTRL = 2000
N_HVG = 2000
LFC_THRESHOLD = 0.5
MIN_DE_TARGETS = 5
N_LAYERS = 18
N_HEADS = 18
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1
EXTRACT_LAYER = 12  # L13, 0-indexed

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
                                 hvg_to_local, min_de=5):
    aurocs_attn = []
    aurocs_corr = []
    per_gene = []
    hvg_name_to_idx = {g.upper(): i for i, g in enumerate(hvg_genes)}

    for pg, targets_dict in de_labels.items():
        pg_local = hvg_name_to_idx.get(pg.upper())
        if pg_local is None:
            continue

        labels = []
        scores_attn = []
        scores_corr = []

        for tgt_gene, is_de in targets_dict.items():
            tgt_local = hvg_name_to_idx.get(tgt_gene)
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
                                     token_to_hvg, device,
                                     extract_layer=12, hvg_genes=None):
    """Run forward pass with head_mask, extract attention from extract_layer.
    Optimized: numpy matrix accumulation, no per-head entropy (not needed for R10).
    """
    import torch

    n_hvg = len(hvg_genes)
    hvg_name_to_idx = {g: i for i, g in enumerate(hvg_genes)}

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

    active_heads_mask = head_mask_tensor[extract_layer].cpu().numpy()
    active_head_indices = np.where(active_heads_mask > 0.5)[0]
    n_active = len(active_head_indices)

    attn_sum = np.zeros((n_hvg, n_hvg), dtype=np.float64)
    attn_count = np.zeros((n_hvg, n_hvg), dtype=np.float64)

    sorted_indices = sorted(range(len(control_tokens)), key=lambda i: len(control_tokens[i]))

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

            if n_active > 0:
                layer_attn = full_attn[:, active_head_indices, :, :].mean(axis=1)
            else:
                layer_attn = full_attn.mean(axis=1)

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
                        hvg_idx = hvg_name_to_idx.get(gene)
                        if hvg_idx is not None:
                            pos_list.append(pos_offset + 1)
                            idx_list.append(hvg_idx)

                if len(pos_list) < 2:
                    continue

                pos_arr = np.array(pos_list)
                idx_arr = np.array(idx_list)
                sub_attn = cell_attn[np.ix_(pos_arr, pos_arr)]
                attn_sum[np.ix_(idx_arr, idx_arr)] += sub_attn
                attn_count[np.ix_(idx_arr, idx_arr)] += 1.0

            del full_attn, layer_attn

        cells_processed += bs
        captured_attentions.clear()
        if device.type == 'mps':
            torch.mps.empty_cache()

        if cells_processed % 200 == 0 or cells_processed >= len(control_tokens):
            elapsed = time.time() - t0
            rate = cells_processed / elapsed if elapsed > 0 else 0
            eta = (len(control_tokens) - cells_processed) / rate if rate > 0 else 0
            print(f"      Cells: {cells_processed}/{len(control_tokens)}, "
                  f"rate: {rate:.1f}/s, ETA: {eta:.0f}s")

    model.bert.encoder.layer[extract_layer].forward = orig_forward
    captured_attentions.clear()
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()

    mask = attn_count > 0
    avg_matrix = np.zeros_like(attn_sum)
    avg_matrix[mask] = attn_sum[mask] / attn_count[mask]

    edges = {}
    nonzero_pairs = np.argwhere(mask)
    for r, c in nonzero_pairs:
        if r != c:
            edges[(hvg_genes[r], hvg_genes[c])] = float(avg_matrix[r, c])

    return edges


def save_checkpoint(results, completed_conditions):
    """Save checkpoint after each condition."""
    checkpoint = {
        'completed_conditions': completed_conditions,
        'results': {},
    }
    for cname in completed_conditions:
        if cname in results:
            r = dict(results[cname])
            # Convert arrays to lists for JSON
            if 'aurocs_attn' in r:
                r['aurocs_attn'] = list(r['aurocs_attn']) if hasattr(r['aurocs_attn'], 'tolist') else r['aurocs_attn']
            checkpoint['results'][cname] = r

    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"    [Checkpoint saved: {len(completed_conditions)} conditions]")


def load_checkpoint():
    """Load checkpoint if exists."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r') as f:
            checkpoint = json.load(f)
        print(f"  Resuming from checkpoint: {len(checkpoint['completed_conditions'])} conditions done")
        return checkpoint
    return None


def run():
    import torch
    from transformers import BertForMaskedLM

    total_t0 = time.time()

    print("=" * 70)
    print("EXPANDED CAUSAL ABLATIONS (ROUND 10)")
    print("=" * 70)

    # ============================================================
    # STEP 1: Load R9 results and pre-computed data
    # ============================================================
    print("\nSTEP 1: Loading pre-computed data...")

    # Per-head TRRUST AUROCs
    auroc_matrix = np.load(os.path.join(EXTRACTION_DIR, "per_head_trrust", "head_trrust_auroc_matrix.npy"))
    print(f"  Head AUROC matrix: {auroc_matrix.shape}")

    with open(os.path.join(EXTRACTION_DIR, "per_head_trrust", "top5_regulatory_heads.json")) as f:
        top5_list = json.load(f)
    top5_heads = [(h['layer'], h['head']) for h in top5_list]

    # Per-layer perturbation-first AUROCs (for composite ranking)
    per_layer_df = pd.read_csv(os.path.join(PAPER_DIR, "experiments", "round9_full_layer_analysis", "per_layer_per_gene.csv"))
    layer_aurocs = np.array([per_layer_df[f'auroc_L{l}'].mean() for l in range(N_LAYERS)])
    print(f"  Layer AUROCs: min={layer_aurocs.min():.4f}, max={layer_aurocs.max():.4f}")
    print(f"  Best layer: L{np.argmax(layer_aurocs)} (AUROC={layer_aurocs.max():.4f})")

    # DE labels
    with open(os.path.join(EXTRACTION_DIR, "de_labels.pkl"), 'rb') as f:
        de_labels = pickle.load(f)

    # HVG genes
    with open(os.path.join(EXTRACTION_DIR, "hvg_gene_list.json")) as f:
        hvg_genes = json.load(f)
    hvg_to_local = {g.upper(): i for i, g in enumerate(hvg_genes)}

    # Correlation matrix
    corr_matrix = np.load(os.path.join(EXTRACTION_DIR, "correlation_matrix.npy"))

    # ============================================================
    # STEP 2: Build comprehensive head rankings
    # ============================================================
    print("\nSTEP 2: Building head rankings...")

    # Flatten head AUROCs with (layer, head) tuples
    all_heads = []
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            all_heads.append({
                'layer': l, 'head': h,
                'trrust_auroc': float(auroc_matrix[l, h]),
                'layer_pertfirst_auroc': float(layer_aurocs[l]),
                'composite_score': float(auroc_matrix[l, h] * layer_aurocs[l]),
            })

    # Sort by TRRUST AUROC (descending) for regulatory ranking
    heads_by_trrust = sorted(all_heads, key=lambda x: x['trrust_auroc'], reverse=True)
    # Sort by composite score (descending)
    heads_by_composite = sorted(all_heads, key=lambda x: x['composite_score'], reverse=True)
    # Sort by TRRUST AUROC (ascending) for bottom ranking
    heads_by_trrust_asc = sorted(all_heads, key=lambda x: x['trrust_auroc'])

    print(f"  Top-5 TRRUST: {[(h['layer'], h['head']) for h in heads_by_trrust[:5]]}")
    print(f"  Top-5 composite: {[(h['layer'], h['head']) for h in heads_by_composite[:5]]}")
    print(f"  Bottom-5 TRRUST: {[(h['layer'], h['head']) for h in heads_by_trrust_asc[:5]]}")

    best_layer = int(np.argmax(layer_aurocs))
    print(f"  Best layer for full-layer ablation: L{best_layer}")

    # ============================================================
    # STEP 3: Define all conditions
    # ============================================================
    print("\nSTEP 3: Defining ablation conditions...")

    conditions = {}

    # -- Baseline (always needed, but reuse from R9 if available) --
    conditions['baseline'] = {
        'heads': [],
        'description': 'Full model (no ablation)',
    }

    # -- TRRUST-ranked dose-response --
    for k in [5, 10, 20, 50]:
        heads = [(h['layer'], h['head']) for h in heads_by_trrust[:k]]
        conditions[f'regulatory_top{k}'] = {
            'heads': heads,
            'description': f'Ablate top-{k} TRRUST-ranked heads',
        }

    # -- Bottom TRRUST (inverse control) --
    for k in [5, 10]:
        heads = [(h['layer'], h['head']) for h in heads_by_trrust_asc[:k]]
        conditions[f'bottom_{k}'] = {
            'heads': heads,
            'description': f'Ablate bottom-{k} TRRUST-ranked heads',
        }

    # -- Composite ranking --
    for k in [5, 10]:
        heads = [(h['layer'], h['head']) for h in heads_by_composite[:k]]
        conditions[f'composite_top{k}'] = {
            'heads': heads,
            'description': f'Ablate top-{k} by TRRUST×layer_AUROC composite',
        }

    # -- Full layer ablation (all 18 heads in best layer) --
    heads = [(best_layer, h) for h in range(N_HEADS)]
    conditions[f'layer_L{best_layer}_all'] = {
        'heads': heads,
        'description': f'Ablate all 18 heads in L{best_layer} (best perturbation-first layer)',
    }

    # -- Matched random controls --
    rng = np.random.RandomState(456)
    for k in [10, 20, 50]:
        heads_all = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]
        chosen = rng.choice(len(heads_all), size=k, replace=False)
        heads = [heads_all[i] for i in chosen]
        conditions[f'random_{k}'] = {
            'heads': heads,
            'description': f'Ablate {k} random heads (control)',
        }

    # Also map R9 conditions to new naming
    r9_mapping = {
        'baseline': 'baseline',
        'ablate_regulatory': 'regulatory_top5',
    }

    print(f"  Total conditions: {len(conditions)}")
    for cname, cond in conditions.items():
        print(f"    {cname}: {len(cond['heads'])} heads")

    # ============================================================
    # STEP 4: Check checkpoint / load R9 results
    # ============================================================
    print("\nSTEP 4: Checking for previous results...")

    results = {}
    completed_conditions = []

    # Try to load R9 results for baseline and regulatory_top5
    r9_results_path = os.path.join(R9_DIR, "ablation_results.json")
    if os.path.exists(r9_results_path):
        print("  Loading R9 ablation results...")
        with open(r9_results_path, 'r') as f:
            r9_data = json.load(f)

        # Map R9 conditions
        if 'conditions' in r9_data:
            for r9_name, new_name in [('baseline', 'baseline'), ('ablate_regulatory', 'regulatory_top5')]:
                if r9_name in r9_data['conditions']:
                    r9_cond = r9_data['conditions'][r9_name]
                    results[new_name] = {
                        'description': conditions[new_name]['description'],
                        'ablated_heads': conditions[new_name]['heads'],
                        'n_ablated': len(conditions[new_name]['heads']),
                        'n_evaluated': r9_cond['n_evaluated'],
                        'mean_auroc_attn': r9_cond['mean_auroc_attn'],
                        'std_auroc_attn': r9_cond['std_auroc_attn'],
                        'median_auroc_attn': r9_cond['median_auroc_attn'],
                        'mean_auroc_corr': r9_cond['mean_auroc_corr'],
                        'per_gene': r9_cond['per_gene'],
                        'source': 'round9',
                    }
                    completed_conditions.append(new_name)
                    print(f"    Reused R9: {new_name} (AUROC={r9_cond['mean_auroc_attn']:.4f})")

    # Try checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        for cname in checkpoint['completed_conditions']:
            if cname not in completed_conditions and cname in checkpoint['results']:
                results[cname] = checkpoint['results'][cname]
                completed_conditions.append(cname)
        print(f"  After checkpoint: {len(completed_conditions)} conditions done")

    remaining = [c for c in conditions if c not in completed_conditions]
    print(f"  Remaining: {len(remaining)} conditions")
    if not remaining:
        print("  All conditions already complete!")
    else:
        print(f"  To run: {remaining}")

    if not remaining:
        # Skip to analysis
        pass
    else:
        # ============================================================
        # STEP 5: Load data and tokenize
        # ============================================================
        print("\nSTEP 5: Loading data and tokenizing...")
        t1 = time.time()

        with h5py.File(DATA_PATH, 'r') as f:
            var_genes = load_categorical_column(f['var'], 'gene_name_index')
            n_cells_total, n_genes_total = f['X'].shape

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

        cell_genes_raw = None
        with h5py.File(DATA_PATH, 'r') as f:
            cell_genes_raw = load_categorical_column(f['obs'], 'gene')

        control_mask = np.zeros(len(cell_genes_raw), dtype=bool)
        for ctrl_name in ['non-targeting', 'Non-targeting', 'non_targeting']:
            control_mask |= (cell_genes_raw == ctrl_name)
        ctrl_indices = np.where(control_mask)[0]
        del cell_genes_raw

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

        print(f"  Tokenized: {len(control_tokens)} cells")
        print(f"  Time: {time.time()-t1:.1f}s")

        # ============================================================
        # STEP 6: Load model and run remaining conditions
        # ============================================================
        print("\nSTEP 6: Loading model...")

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

        for ci, cond_name in enumerate(remaining):
            cond = conditions[cond_name]
            print(f"\n{'='*50}")
            print(f"  Condition {ci+1}/{len(remaining)}: {cond_name}")
            print(f"  {cond['description']}")
            print(f"  Ablating {len(cond['heads'])} heads")
            print(f"{'='*50}")
            t_cond = time.time()

            # Build head mask
            head_mask = torch.ones(N_LAYERS, N_HEADS, device=device)
            for l, h in cond['heads']:
                head_mask[l, h] = 0.0

            edges = extract_attention_with_ablation(
                model, control_tokens, head_mask,
                token_to_hvg, device,
                extract_layer=EXTRACT_LAYER,
                hvg_genes=hvg_genes,
            )

            aurocs_attn, aurocs_corr, per_gene = evaluate_perturbation_first(
                edges, corr_matrix, de_labels, hvg_genes, hvg_to_local,
                min_de=MIN_DE_TARGETS,
            )

            results[cond_name] = {
                'description': cond['description'],
                'ablated_heads': cond['heads'],
                'n_ablated': len(cond['heads']),
                'n_evaluated': len(aurocs_attn),
                'mean_auroc_attn': round(float(aurocs_attn.mean()), 4) if len(aurocs_attn) > 0 else None,
                'std_auroc_attn': round(float(aurocs_attn.std()), 4) if len(aurocs_attn) > 0 else None,
                'median_auroc_attn': round(float(np.median(aurocs_attn)), 4) if len(aurocs_attn) > 0 else None,
                'mean_auroc_corr': round(float(aurocs_corr.mean()), 4) if len(aurocs_corr) > 0 else None,
                'per_gene': per_gene,
                'source': 'round10',
            }

            cond_time = time.time() - t_cond
            print(f"    AUROC: {results[cond_name]['mean_auroc_attn']:.4f} ± {results[cond_name]['std_auroc_attn']:.4f}")
            print(f"    Evaluated: {len(aurocs_attn)} perturbations")
            print(f"    Time: {cond_time:.0f}s ({cond_time/60:.1f} min)")

            completed_conditions.append(cond_name)
            save_checkpoint(results, completed_conditions)

            del edges
            gc.collect()
            if device.type == 'mps':
                torch.mps.empty_cache()

        del model
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()

    # ============================================================
    # STEP 7: Statistical analysis
    # ============================================================
    print("\n\nSTEP 7: Statistical analysis...")

    baseline_pg = {g['gene']: g['auroc_attn'] for g in results['baseline']['per_gene']}
    baseline_genes = sorted(baseline_pg.keys())

    comparisons = {}
    for cond_name in results:
        if cond_name == 'baseline':
            continue

        cond_pg = {g['gene']: g['auroc_attn'] for g in results[cond_name]['per_gene']}
        common = sorted(set(baseline_genes) & set(cond_pg.keys()))
        if len(common) < 10:
            continue

        b_arr = np.array([baseline_pg[g] for g in common])
        c_arr = np.array([cond_pg[g] for g in common])
        diff = b_arr - c_arr  # positive = baseline better

        try:
            w_stat, w_p = stats.wilcoxon(diff)
        except:
            w_p = 1.0
        try:
            t_stat, t_p = stats.ttest_rel(b_arr, c_arr)
        except:
            t_p = 1.0

        cohens_d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0

        comp = {
            'n_compared': len(common),
            'mean_drop': round(float(diff.mean()), 4),
            'std_drop': round(float(diff.std()), 4),
            'median_drop': round(float(np.median(diff)), 4),
            'wilcoxon_p': float(w_p),
            'ttest_p': float(t_p),
            'cohens_d': round(cohens_d, 4),
            'n_baseline_better': int((diff > 0).sum()),
            'n_ablation_better': int((diff < 0).sum()),
            'ci_95': [round(float(np.percentile(diff, 2.5)), 4),
                      round(float(np.percentile(diff, 97.5)), 4)],
        }
        comparisons[cond_name] = comp

        print(f"\n  {cond_name} (k={results[cond_name]['n_ablated']}):")
        print(f"    AUROC: {results[cond_name]['mean_auroc_attn']:.4f}")
        print(f"    Drop: {comp['mean_drop']:+.4f} [{comp['ci_95'][0]:+.4f}, {comp['ci_95'][1]:+.4f}]")
        print(f"    Wilcoxon p={comp['wilcoxon_p']:.4f}, Cohen's d={comp['cohens_d']:.4f}")

    # Dose-response analysis
    print("\n\nDose-Response Analysis:")
    dose_data = []
    for k in [5, 10, 20, 50]:
        reg_key = f'regulatory_top{k}'
        rand_key = f'random_{k}'
        if reg_key in results:
            dose_data.append({
                'k': k,
                'regulatory_auroc': results[reg_key]['mean_auroc_attn'],
                'regulatory_drop': comparisons.get(reg_key, {}).get('mean_drop', None),
                'random_auroc': results.get(rand_key, {}).get('mean_auroc_attn', None),
                'random_drop': comparisons.get(rand_key, {}).get('mean_drop', None),
            })
            reg_drop = comparisons.get(reg_key, {}).get('mean_drop', 0)
            rand_drop = comparisons.get(rand_key, {}).get('mean_drop', 0)
            excess = (reg_drop or 0) - (rand_drop or 0)
            print(f"  k={k}: reg_drop={reg_drop:+.4f}, rand_drop={rand_drop:+.4f}, excess={excess:+.4f}")

    # ============================================================
    # STEP 8: Generate figure
    # ============================================================
    print("\n\nSTEP 8: Generating figure...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Dose-response curve
    ax = axes[0, 0]
    ks = [5, 10, 20, 50]
    reg_aurocs = [results.get(f'regulatory_top{k}', {}).get('mean_auroc_attn', None) for k in ks]
    rand_aurocs = [results.get(f'random_{k}', {}).get('mean_auroc_attn', None) for k in ks]
    baseline_auroc = results['baseline']['mean_auroc_attn']

    ax.axhline(y=baseline_auroc, color='green', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_auroc:.3f})')
    valid_ks_reg = [k for k, a in zip(ks, reg_aurocs) if a is not None]
    valid_aurocs_reg = [a for a in reg_aurocs if a is not None]
    valid_ks_rand = [k for k, a in zip(ks, rand_aurocs) if a is not None]
    valid_aurocs_rand = [a for a in rand_aurocs if a is not None]

    if valid_ks_reg:
        ax.plot(valid_ks_reg, valid_aurocs_reg, 'ro-', label='Regulatory (TRRUST-ranked)', markersize=8)
    if valid_ks_rand:
        ax.plot(valid_ks_rand, valid_aurocs_rand, 'bs-', label='Random controls', markersize=8)

    # Add bottom heads if available
    bottom_ks = [5, 10]
    bottom_aurocs = [results.get(f'bottom_{k}', {}).get('mean_auroc_attn', None) for k in bottom_ks]
    valid_bottom = [(k, a) for k, a in zip(bottom_ks, bottom_aurocs) if a is not None]
    if valid_bottom:
        ax.plot([k for k, a in valid_bottom], [a for k, a in valid_bottom],
                'g^-', label='Bottom (lowest TRRUST)', markersize=8)

    ax.set_xlabel('Number of Heads Ablated (k)')
    ax.set_ylabel('Perturbation-First AUROC')
    ax.set_title('A. Dose-Response: Ablation vs AUROC')
    ax.legend(fontsize=8)
    ax.set_xticks(ks)

    # Panel B: All conditions bar chart
    ax = axes[0, 1]
    cond_order = ['baseline']
    cond_order += [f'regulatory_top{k}' for k in [5, 10, 20, 50] if f'regulatory_top{k}' in results]
    cond_order += [f'bottom_{k}' for k in [5, 10] if f'bottom_{k}' in results]
    cond_order += [f'composite_top{k}' for k in [5, 10] if f'composite_top{k}' in results]
    layer_key = f'layer_L{best_layer}_all'
    if layer_key in results:
        cond_order.append(layer_key)
    cond_order += [f'random_{k}' for k in [10, 20, 50] if f'random_{k}' in results]

    bar_aurocs = [results[c]['mean_auroc_attn'] for c in cond_order if c in results]
    bar_labels_raw = cond_order[:len(bar_aurocs)]
    # Shorten labels
    label_map = {
        'baseline': 'Base',
        'regulatory_top5': 'Reg5', 'regulatory_top10': 'Reg10',
        'regulatory_top20': 'Reg20', 'regulatory_top50': 'Reg50',
        'bottom_5': 'Bot5', 'bottom_10': 'Bot10',
        'composite_top5': 'Comp5', 'composite_top10': 'Comp10',
        f'layer_L{best_layer}_all': f'L{best_layer}',
        'random_10': 'Rnd10', 'random_20': 'Rnd20', 'random_50': 'Rnd50',
    }
    bar_labels = [label_map.get(c, c) for c in bar_labels_raw]
    colors_map = {
        'baseline': '#2ecc71',
        'regulatory_top5': '#e74c3c', 'regulatory_top10': '#c0392b',
        'regulatory_top20': '#a93226', 'regulatory_top50': '#922b21',
        'bottom_5': '#27ae60', 'bottom_10': '#229954',
        'composite_top5': '#e67e22', 'composite_top10': '#d35400',
        f'layer_L{best_layer}_all': '#8e44ad',
        'random_10': '#3498db', 'random_20': '#2980b9', 'random_50': '#2471a3',
    }
    bar_colors = [colors_map.get(c, 'gray') for c in bar_labels_raw]

    ax.bar(range(len(bar_aurocs)), bar_aurocs, color=bar_colors, alpha=0.85)
    ax.set_xticks(range(len(bar_aurocs)))
    ax.set_xticklabels(bar_labels, fontsize=7, rotation=45)
    ax.set_ylabel('Mean AUROC')
    ax.set_title('B. All Ablation Conditions')
    if 'baseline' in results:
        ax.axhline(y=results['baseline']['mean_auroc_attn'], color='green', linestyle='--', alpha=0.5)

    # Panel C: Regulatory vs random drop comparison
    ax = axes[1, 0]
    drops_reg = []
    drops_rand = []
    ks_with_both = []
    for k in [5, 10, 20, 50]:
        reg_key = f'regulatory_top{k}'
        rand_key = f'random_{k}'
        if reg_key in comparisons and rand_key in comparisons:
            drops_reg.append(comparisons[reg_key]['mean_drop'])
            drops_rand.append(comparisons[rand_key]['mean_drop'])
            ks_with_both.append(k)

    if ks_with_both:
        x = np.arange(len(ks_with_both))
        w = 0.35
        ax.bar(x - w/2, drops_reg, w, label='Regulatory', color='#e74c3c', alpha=0.8)
        ax.bar(x + w/2, drops_rand, w, label='Random', color='#3498db', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'k={k}' for k in ks_with_both])
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylabel('AUROC Drop from Baseline')
        ax.set_title('C. Regulatory vs Random Drop')
        ax.legend(fontsize=8)

    # Panel D: Effect sizes (Cohen's d)
    ax = axes[1, 1]
    effect_conds = [c for c in cond_order if c in comparisons and c != 'baseline']
    if effect_conds:
        ds = [comparisons[c]['cohens_d'] for c in effect_conds]
        elabels = [label_map.get(c, c) for c in effect_conds]
        ecolors = [colors_map.get(c, 'gray') for c in effect_conds]
        ax.barh(range(len(ds)), ds, color=ecolors, alpha=0.8)
        ax.set_yticks(range(len(ds)))
        ax.set_yticklabels(elabels, fontsize=8)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.axvline(x=0.2, color='gray', linestyle=':', alpha=0.5, label='Small effect (0.2)')
        ax.set_xlabel("Cohen's d (Baseline − Ablation)")
        ax.set_title('D. Effect Sizes')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_expanded_ablation.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")

    # ============================================================
    # STEP 9: Save final results
    # ============================================================
    print("\nSTEP 9: Saving results...")

    # Strip per_gene for summary (too large)
    results_summary = {}
    for cname, r in results.items():
        rs = {k: v for k, v in r.items() if k != 'per_gene'}
        rs['n_per_gene'] = len(r.get('per_gene', []))
        results_summary[cname] = rs

    final = {
        'experiment': 'round10_expanded_ablation',
        'n_conditions': len(results),
        'n_layers': N_LAYERS,
        'n_heads': N_HEADS,
        'extract_layer': EXTRACT_LAYER,
        'conditions': results_summary,
        'comparisons': comparisons,
        'dose_response': dose_data,
        'head_rankings': {
            'top5_trrust': [(h['layer'], h['head'], h['trrust_auroc']) for h in heads_by_trrust[:5]],
            'top5_composite': [(h['layer'], h['head'], h['composite_score']) for h in heads_by_composite[:5]],
            'bottom5_trrust': [(h['layer'], h['head'], h['trrust_auroc']) for h in heads_by_trrust_asc[:5]],
        },
        'total_time_seconds': round(time.time() - total_t0, 1),
    }

    with open(os.path.join(OUT_DIR, "expanded_ablation_results.json"), 'w') as f:
        json.dump(final, f, indent=2)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"EXPANDED ABLATION COMPLETE - Total: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")

    print(f"\nSummary:")
    for cname in cond_order:
        if cname in results:
            r = results[cname]
            drop_str = ""
            if cname in comparisons:
                c = comparisons[cname]
                drop_str = f"  drop={c['mean_drop']:+.4f}  p={c['wilcoxon_p']:.4f}  d={c['cohens_d']:.4f}"
            print(f"  {cname}: AUROC={r['mean_auroc_attn']:.4f}{drop_str}")


if __name__ == '__main__':
    run()
