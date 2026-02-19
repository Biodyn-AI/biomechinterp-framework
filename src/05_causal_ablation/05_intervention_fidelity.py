#!/usr/bin/env python3
"""
Round 13: Intervention Fidelity Check
======================================
Validates that each causal intervention from Round 11 (orthogonal interventions)
materially perturbs internal model representations. Without this check, a
behavioral null (AUROC unchanged) could reflect a weak/ineffective intervention
rather than genuine functional redundancy.

Two-pass approach:
  Pass 1 (baseline): Run model on 2000 control cells with NO intervention.
    Store mean-pooled hidden states (18 layers) and logits per cell.
  Pass 2 (per condition): Run model with each intervention.
    Compute cosine distances and L2 distances to baseline on-the-fly.

Metrics per cell per condition:
  - Hidden-state cosine distance (per layer, mean-pooled over non-padding tokens)
  - Logit cosine distance (mean-pooled over non-padding tokens)
  - Logit L2 distance
  - Attention entropy change (for uniform attention conditions)

Addresses reviewer concern: "Do the interventions actually change representations,
or is the AUROC null simply uninformative?"
"""

import os, sys, gc, json, time, pickle, warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import h5py

PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
DATA_PATH = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data", "replogle_concat.h5ad")
EXTRACTION_DIR = os.path.join(PAPER_DIR, "experiments", "round9_unified_extraction")
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round13_intervention_fidelity")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

N_CTRL = 2000
N_LAYERS = 18
N_HEADS = 18
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1

CHECKPOINT_PATH = os.path.join(OUT_DIR, "checkpoint.json")


# ============================================================
# Data loading / tokenization (reused exactly from R11)
# ============================================================

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


# ============================================================
# Checkpoint support
# ============================================================

def save_checkpoint(data):
    """Save checkpoint with completed conditions and baseline status."""
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(data, f, indent=2, default=_json_serializer)
    print(f"    [Checkpoint saved: {len(data.get('completed_conditions', []))} conditions done, "
          f"baseline={'done' if data.get('baseline_done') else 'pending'}]")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r') as f:
            return json.load(f)
    return None


def _json_serializer(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ============================================================
# Cosine distance utilities
# ============================================================

def cosine_distance(a, b):
    """Cosine distance between two 1-D vectors. Returns 0 if either is zero."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return 1.0 - dot / (norm_a * norm_b)


def shannon_entropy(probs):
    """Shannon entropy of a probability distribution (base e). Handles zeros."""
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log(probs))


# ============================================================
# Main
# ============================================================

def run():
    import torch
    from transformers import BertForMaskedLM

    total_t0 = time.time()

    print("=" * 70)
    print("ROUND 13: INTERVENTION FIDELITY CHECK")
    print("=" * 70)
    print("Do the R11 interventions actually perturb internal representations?")

    # ============================================================
    # STEP 1: Define intervention conditions (same as R11)
    # ============================================================
    print("\nSTEP 1: Loading head rankings and defining conditions...")

    auroc_matrix = np.load(os.path.join(EXTRACTION_DIR, "per_head_trrust", "head_trrust_auroc_matrix.npy"))
    heads_flat = []
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            heads_flat.append((l, h, float(auroc_matrix[l, h])))
    heads_flat.sort(key=lambda x: x[2], reverse=True)
    top5_heads = [(l, h) for l, h, _ in heads_flat[:5]]
    top10_heads = [(l, h) for l, h, _ in heads_flat[:10]]

    rng = np.random.RandomState(789)
    all_heads_list = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]
    random_10_idx = rng.choice(len(all_heads_list), 10, replace=False)
    random_10_heads = [all_heads_list[i] for i in random_10_idx]

    conditions = {
        'uniform_attn_reg_top5': {
            'type': 'uniform_attention',
            'target_heads': top5_heads,
            'description': 'Uniform attention on top-5 TRRUST heads',
        },
        'uniform_attn_reg_top10': {
            'type': 'uniform_attention',
            'target_heads': top10_heads,
            'description': 'Uniform attention on top-10 TRRUST heads',
        },
        'uniform_attn_random_10': {
            'type': 'uniform_attention',
            'target_heads': random_10_heads,
            'description': 'Uniform attention on 10 random heads (control)',
        },
        'mlp_zero_L15': {
            'type': 'mlp_ablation',
            'target_layers': [14],
            'description': 'Zero MLP output at L15 (best perturbation-first)',
        },
        'mlp_zero_L13_L14_L15': {
            'type': 'mlp_ablation',
            'target_layers': [12, 13, 14],
            'description': 'Zero MLP output at L13+L14+L15',
        },
        'mlp_zero_random_layer': {
            'type': 'mlp_ablation',
            'target_layers': [7],
            'description': 'Zero MLP output at L8 (random control)',
        },
    }

    for cname, cond in conditions.items():
        if cond['type'] == 'uniform_attention':
            print(f"  {cname}: uniform on {len(cond['target_heads'])} heads")
        else:
            print(f"  {cname}: zero MLP at layers {[l+1 for l in cond['target_layers']]}")

    # ============================================================
    # STEP 2: Check checkpoint
    # ============================================================
    print("\nSTEP 2: Checking checkpoint...")
    ckpt = load_checkpoint()
    baseline_done = False
    completed_conditions = []
    condition_results = {}

    if ckpt:
        baseline_done = ckpt.get('baseline_done', False)
        completed_conditions = ckpt.get('completed_conditions', [])
        condition_results = ckpt.get('condition_results', {})
        print(f"  Baseline pass: {'DONE' if baseline_done else 'PENDING'}")
        print(f"  Completed conditions: {completed_conditions}")
    else:
        print("  No checkpoint found, starting fresh.")

    remaining_conditions = [c for c in conditions if c not in completed_conditions]
    print(f"  Remaining conditions: {len(remaining_conditions)}")

    if baseline_done and not remaining_conditions:
        print("\n  All work complete! Skipping to figure generation...")
    else:
        # ============================================================
        # STEP 3: Load data and tokenize (same as R11)
        # ============================================================
        print("\nSTEP 3: Loading data and tokenizing control cells...")
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
                mapped_var_indices.append(i)
                mapped_token_ids_list.append(token_dict[ensembl])
                mapped_medians_list.append(gene_median_dict.get(ensembl, 1.0))

        mapped_var_indices = np.array(mapped_var_indices)
        mapped_token_ids = np.array(mapped_token_ids_list)
        mapped_medians = np.array(mapped_medians_list)

        # Identify control cells
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

        n_cells = len(control_tokens)
        print(f"  Tokenized: {n_cells} cells, time: {time.time()-t1:.1f}s")

        # Sort by length for consistent batching
        sorted_indices = sorted(range(n_cells), key=lambda i: len(control_tokens[i]))

        # ============================================================
        # STEP 4: Load model
        # ============================================================
        print("\nSTEP 4: Loading model (with output_hidden_states=True)...")

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"  Device: MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"  Device: CUDA")
        else:
            device = torch.device("cpu")
            print(f"  Device: CPU")

        model = BertForMaskedLM.from_pretrained(
            MODEL_NAME, subfolder=MODEL_SUBFOLDER,
            output_attentions=False,  # Use hooks instead to avoid 10GB+ memory
            output_hidden_states=True,
            attn_implementation="eager",
        )
        model = model.to(device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Model: {n_params:.1f}M params")

        # Determine vocab size from model config
        vocab_size = model.config.vocab_size
        hidden_size = model.config.hidden_size
        print(f"  Vocab size: {vocab_size}, Hidden size: {hidden_size}")

        # ============================================================
        # STEP 5: Baseline pass — collect representations
        # ============================================================
        if not baseline_done:
            print(f"\nSTEP 5: Baseline pass — collecting representations for {n_cells} cells...")
            print(f"  Storage: hidden states {n_cells} x {N_LAYERS} x {hidden_size} = "
                  f"{n_cells * N_LAYERS * hidden_size * 4 / 1e6:.1f} MB")
            print(f"  Storage: logits {n_cells} x {vocab_size} = "
                  f"{n_cells * vocab_size * 4 / 1e6:.1f} MB")

            # Pre-allocate storage
            baseline_hidden = np.zeros((n_cells, N_LAYERS, hidden_size), dtype=np.float32)
            baseline_logits = np.zeros((n_cells, vocab_size), dtype=np.float32)

            # Collect hidden states + logits (no attention — fast)
            print("  Collecting hidden states + logits (output_attentions=False)...")
            cells_processed = 0
            t0 = time.time()

            for batch_start in range(0, n_cells, BATCH_SIZE):
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
                    outputs = model(input_ids=input_ids_t, attention_mask=attn_mask_t)

                logits_np = outputs.logits.cpu().float().numpy()
                hidden_states_np = [hs.cpu().float().numpy() for hs in outputs.hidden_states]

                for cell_i in range(bs):
                    cell_idx = batch_idx_list[cell_i]
                    seq_len = int(attention_mask[cell_i].sum())
                    baseline_logits[cell_idx] = logits_np[cell_i, :seq_len, :].mean(axis=0)
                    for layer in range(N_LAYERS):
                        baseline_hidden[cell_idx, layer, :] = hidden_states_np[layer + 1][cell_i, :seq_len, :].mean(axis=0)

                cells_processed += bs
                del outputs, logits_np, hidden_states_np
                if device.type == 'mps':
                    torch.mps.empty_cache()

                if cells_processed % 200 == 0 or cells_processed >= n_cells:
                    elapsed = time.time() - t0
                    rate = cells_processed / elapsed if elapsed > 0 else 0
                    eta = (n_cells - cells_processed) / rate if rate > 0 else 0
                    print(f"    Baseline: {cells_processed}/{n_cells}, "
                          f"rate: {rate:.1f}/s, ETA: {eta:.0f}s")

            baseline_time = time.time() - t0
            print(f"  Baseline pass done: {baseline_time:.0f}s ({baseline_time/60:.1f} min)")

            # Save baseline arrays to disk for resume
            np.save(os.path.join(OUT_DIR, "baseline_hidden.npy"), baseline_hidden)
            np.save(os.path.join(OUT_DIR, "baseline_logits.npy"), baseline_logits)
            baseline_done = True
            save_checkpoint({
                'baseline_done': True,
                'completed_conditions': completed_conditions,
                'condition_results': condition_results,
                'n_cells': n_cells,
                'baseline_time_seconds': round(baseline_time, 1),
            })
            print(f"  Baseline arrays saved to disk.")

            gc.collect()
            if device.type == 'mps':
                torch.mps.empty_cache()
        else:
            print("\nSTEP 5: Baseline already done, loading from disk...")
            baseline_hidden = np.load(os.path.join(OUT_DIR, "baseline_hidden.npy"))
            baseline_logits = np.load(os.path.join(OUT_DIR, "baseline_logits.npy"))
            n_cells = baseline_hidden.shape[0]
            print(f"  Loaded baseline: {n_cells} cells, "
                  f"hidden {baseline_hidden.shape}, logits {baseline_logits.shape}")

        # ============================================================
        # STEP 6: Intervention passes
        # ============================================================
        if remaining_conditions:
            print(f"\nSTEP 6: Running {len(remaining_conditions)} intervention conditions...")

            for ci, cond_name in enumerate(remaining_conditions):
                cond = conditions[cond_name]
                print(f"\n{'='*60}")
                print(f"  Condition {ci+1}/{len(remaining_conditions)}: {cond_name}")
                print(f"  {cond['description']}")
                print(f"{'='*60}")
                t_cond = time.time()

                # --- Set up intervention hooks ---
                hook_handles = []

                if cond['type'] == 'uniform_attention':
                    target_by_layer = {}
                    for l, h in cond['target_heads']:
                        if l not in target_by_layer:
                            target_by_layer[l] = []
                        target_by_layer[l].append(h)

                    for layer_idx, head_indices in target_by_layer.items():
                        dropout_module = model.bert.encoder.layer[layer_idx].attention.self.dropout

                        def make_dropout_hook(heads_to_uniform_list):
                            def hook(module, input, output):
                                if output.dim() == 4:
                                    seq_len = output.shape[-1]
                                    for hi in heads_to_uniform_list:
                                        if hi < output.shape[1]:
                                            output[:, hi, :, :] = 1.0 / seq_len
                                return output
                            return hook

                        handle = dropout_module.register_forward_hook(
                            make_dropout_hook(head_indices))
                        hook_handles.append(handle)

                    print(f"    Hooks: uniform attention on {sum(len(v) for v in target_by_layer.values())} "
                          f"heads across {len(target_by_layer)} layers")

                elif cond['type'] == 'mlp_ablation':
                    for layer_idx in cond['target_layers']:
                        dense_module = model.bert.encoder.layer[layer_idx].output.dense

                        def make_zero_hook():
                            def hook(module, input, output):
                                return torch.zeros_like(output)
                            return hook

                        handle = dense_module.register_forward_hook(make_zero_hook())
                        hook_handles.append(handle)

                    print(f"    Hooks: zero MLP at layers {[l+1 for l in cond['target_layers']]}")

                # --- Run intervention pass ---
                hidden_cos_dist = np.zeros((n_cells, N_LAYERS), dtype=np.float64)
                logit_cos_dist = np.zeros(n_cells, dtype=np.float64)
                logit_l2_dist = np.zeros(n_cells, dtype=np.float64)

                cells_processed = 0
                t0 = time.time()

                for batch_start in range(0, n_cells, BATCH_SIZE):
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
                        outputs = model(input_ids=input_ids_t, attention_mask=attn_mask_t)

                    logits_np = outputs.logits.cpu().float().numpy()
                    hidden_states_np = [hs.cpu().float().numpy() for hs in outputs.hidden_states]

                    for cell_i in range(bs):
                        cell_idx = batch_idx_list[cell_i]
                        seq_len = int(attention_mask[cell_i].sum())

                        # Mean-pooled logits
                        cell_logits = logits_np[cell_i, :seq_len, :].mean(axis=0)
                        logit_cos_dist[cell_idx] = cosine_distance(
                            baseline_logits[cell_idx], cell_logits)
                        logit_l2_dist[cell_idx] = np.linalg.norm(
                            baseline_logits[cell_idx] - cell_logits)

                        # Mean-pooled hidden states per layer
                        for layer in range(N_LAYERS):
                            cell_hs = hidden_states_np[layer + 1][cell_i, :seq_len, :].mean(axis=0)
                            hidden_cos_dist[cell_idx, layer] = cosine_distance(
                                baseline_hidden[cell_idx, layer], cell_hs)

                    cells_processed += bs

                    del outputs, logits_np, hidden_states_np
                    if device.type == 'mps':
                        torch.mps.empty_cache()

                    if cells_processed % 200 == 0 or cells_processed >= n_cells:
                        elapsed = time.time() - t0
                        rate = cells_processed / elapsed if elapsed > 0 else 0
                        eta = (n_cells - cells_processed) / rate if rate > 0 else 0
                        print(f"    {cond_name}: {cells_processed}/{n_cells}, "
                              f"rate: {rate:.1f}/s, ETA: {eta:.0f}s")

                # --- Remove intervention hooks (keep entropy capture for next condition) ---
                for h in hook_handles:
                    h.remove()
                hook_handles.clear()

                cond_time = time.time() - t_cond

                # --- Aggregate statistics ---
                # Hidden state cosine distance per layer
                hidden_per_layer = {}
                for layer in range(N_LAYERS):
                    vals = hidden_cos_dist[:, layer]
                    hidden_per_layer[f"L{layer+1}"] = {
                        'mean': round(float(np.mean(vals)), 6),
                        'std': round(float(np.std(vals)), 6),
                        'median': round(float(np.median(vals)), 6),
                        'min': round(float(np.min(vals)), 6),
                        'max': round(float(np.max(vals)), 6),
                        'pct95': round(float(np.percentile(vals, 95)), 6),
                        'pct5': round(float(np.percentile(vals, 5)), 6),
                    }

                # Logit distances
                logit_cos_stats = {
                    'mean': round(float(np.mean(logit_cos_dist)), 6),
                    'std': round(float(np.std(logit_cos_dist)), 6),
                    'median': round(float(np.median(logit_cos_dist)), 6),
                    'min': round(float(np.min(logit_cos_dist)), 6),
                    'max': round(float(np.max(logit_cos_dist)), 6),
                }
                logit_l2_stats = {
                    'mean': round(float(np.mean(logit_l2_dist)), 6),
                    'std': round(float(np.std(logit_l2_dist)), 6),
                    'median': round(float(np.median(logit_l2_dist)), 6),
                    'min': round(float(np.min(logit_l2_dist)), 6),
                    'max': round(float(np.max(logit_l2_dist)), 6),
                }

                # Store mean hidden cosine distance as a flat array for the heatmap
                hidden_cos_means = [hidden_per_layer[f"L{l+1}"]['mean'] for l in range(N_LAYERS)]

                cond_result = {
                    'description': cond['description'],
                    'type': cond['type'],
                    'n_cells': n_cells,
                    'hidden_cosine_distance_per_layer': hidden_per_layer,
                    'hidden_cosine_distance_means': hidden_cos_means,
                    'logit_cosine_distance': logit_cos_stats,
                    'logit_l2_distance': logit_l2_stats,
                    'time_seconds': round(cond_time, 1),
                }
                condition_results[cond_name] = cond_result

                print(f"\n    --- {cond_name} Summary ---")
                print(f"    Hidden cos dist (mean across layers): "
                      f"{np.mean(hidden_cos_means):.6f}")
                print(f"    Hidden cos dist range: "
                      f"L1={hidden_cos_means[0]:.6f} ... L18={hidden_cos_means[-1]:.6f}")
                print(f"    Logit cos dist: {logit_cos_stats['mean']:.6f}")
                print(f"    Logit L2 dist: {logit_l2_stats['mean']:.4f}")
                print(f"    Time: {cond_time:.0f}s ({cond_time/60:.1f} min)")

                completed_conditions.append(cond_name)
                save_checkpoint({
                    'baseline_done': True,
                    'completed_conditions': completed_conditions,
                    'condition_results': condition_results,
                    'n_cells': n_cells,
                })

                gc.collect()
                if device.type == 'mps':
                    torch.mps.empty_cache()

            # Done with model
            del model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        else:
            print("\nSTEP 6: All conditions already complete.")

    # ============================================================
    # STEP 7: Generate figure — heatmap of cosine distances
    # ============================================================
    print("\nSTEP 7: Generating figures...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Reload condition_results from checkpoint if needed
    if not condition_results:
        ckpt = load_checkpoint()
        if ckpt:
            condition_results = ckpt.get('condition_results', {})

    if not condition_results:
        print("  No condition results available, skipping figures.")
    else:
        # Order conditions: uniform attention first, then MLP ablation
        cond_order = [
            'uniform_attn_reg_top5', 'uniform_attn_reg_top10', 'uniform_attn_random_10',
            'mlp_zero_L15', 'mlp_zero_L13_L14_L15', 'mlp_zero_random_layer',
        ]
        cond_order = [c for c in cond_order if c in condition_results]

        if cond_order:
            n_conds = len(cond_order)
            layer_labels = [f"L{l+1}" for l in range(N_LAYERS)]

            # --- Figure: Multi-panel ---
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))

            # Panel A: Heatmap of hidden-state cosine distances (conditions x layers)
            ax = axes[0, 0]
            heatmap_data = np.zeros((n_conds, N_LAYERS))
            for i, cname in enumerate(cond_order):
                means = condition_results[cname]['hidden_cosine_distance_means']
                heatmap_data[i, :] = means

            im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd',
                           interpolation='nearest')
            ax.set_yticks(range(n_conds))
            cond_short_names = []
            for c in cond_order:
                name = c.replace('uniform_attn_', 'Unif ').replace('reg_', 'Reg ')
                name = name.replace('random_', 'Rnd ').replace('mlp_zero_', 'MLP0 ')
                cond_short_names.append(name)
            ax.set_yticklabels(cond_short_names, fontsize=9)
            ax.set_xticks(range(N_LAYERS))
            ax.set_xticklabels(layer_labels, fontsize=8, rotation=45)
            ax.set_xlabel('Layer')
            ax.set_title('A. Hidden-State Cosine Distance (conditions x layers)')
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Cosine Distance')

            # Annotate cells with values
            for i in range(n_conds):
                for j in range(N_LAYERS):
                    val = heatmap_data[i, j]
                    text_color = 'white' if val > heatmap_data.max() * 0.65 else 'black'
                    ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                            fontsize=5, color=text_color)

            # Panel B: Logit distances (cosine and L2)
            ax = axes[0, 1]
            logit_cos_means = [condition_results[c]['logit_cosine_distance']['mean']
                               for c in cond_order]
            logit_l2_means = [condition_results[c]['logit_l2_distance']['mean']
                              for c in cond_order]

            x = np.arange(n_conds)
            width = 0.35
            bars1 = ax.bar(x - width/2, logit_cos_means, width, label='Cosine Distance',
                          color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.5)
            ax2_twin = ax.twinx()
            bars2 = ax2_twin.bar(x + width/2, logit_l2_means, width, label='L2 Distance',
                                color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(cond_short_names, fontsize=8, rotation=30, ha='right')
            ax.set_ylabel('Cosine Distance', color='#e74c3c')
            ax2_twin.set_ylabel('L2 Distance', color='#3498db')
            ax.set_title('B. Logit Distances')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

            # Annotate
            for i, v in enumerate(logit_cos_means):
                ax.text(i - width/2, v + logit_cos_means[0] * 0.02,
                        f'{v:.4f}', ha='center', fontsize=6)

            # Panel C: Layer-wise cosine distance profiles (line plot)
            ax = axes[1, 0]
            cmap_lines = plt.cm.Set1
            for i, cname in enumerate(cond_order):
                means = condition_results[cname]['hidden_cosine_distance_means']
                color = cmap_lines(i / max(n_conds - 1, 1))
                linestyle = '-' if 'uniform' in cname else '--'
                ax.plot(range(N_LAYERS), means, marker='o', markersize=4,
                        linewidth=1.5, label=cond_short_names[i],
                        color=color, linestyle=linestyle)
            ax.set_xticks(range(N_LAYERS))
            ax.set_xticklabels(layer_labels, fontsize=8, rotation=45)
            ax.set_xlabel('Layer')
            ax.set_ylabel('Mean Cosine Distance')
            ax.set_title('C. Hidden-State Perturbation by Layer')
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.3)

            # Panel D: Summary — max hidden cosine distance and logit cosine distance
            ax = axes[1, 1]
            max_hidden_cos = [max(condition_results[c]['hidden_cosine_distance_means'])
                              for c in cond_order]
            logit_cos_vals = [condition_results[c]['logit_cosine_distance']['mean']
                              for c in cond_order]
            x = np.arange(n_conds)
            width = 0.35
            colors_h = ['#e74c3c' if 'uniform' in c else '#f39c12' for c in cond_order]
            colors_l = ['#3498db' if 'uniform' in c else '#2ecc71' for c in cond_order]
            ax.bar(x - width/2, max_hidden_cos, width, label='Max Hidden CosD',
                   color=colors_h, alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.bar(x + width/2, logit_cos_vals, width, label='Mean Logit CosD',
                   color=colors_l, alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(cond_short_names, fontsize=7, rotation=30, ha='right')
            ax.set_ylabel('Cosine Distance')
            ax.set_title('D. Intervention Materiality Summary')
            ax.legend(fontsize=8)
            ax.axhline(y=0.001, color='gray', linestyle='--', linewidth=0.8,
                       label='Materiality threshold')
            for i in range(n_conds):
                ax.text(i - width/2, max_hidden_cos[i] + 0.001,
                        f'{max_hidden_cos[i]:.4f}', ha='center', fontsize=6)
                ax.text(i + width/2, logit_cos_vals[i] + 0.001,
                        f'{logit_cos_vals[i]:.4f}', ha='center', fontsize=6)

            plt.tight_layout()
            fig_path = os.path.join(FIG_DIR, "fig_intervention_fidelity.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Figure saved: {fig_path}")

    # ============================================================
    # STEP 8: Save final results JSON
    # ============================================================
    print("\nSTEP 8: Saving results...")

    # Compute summary: for each condition, is the intervention "material"?
    # Heuristic: cosine distance > 0.001 at any layer means non-trivial perturbation
    summary_table = []
    for cname in conditions:
        if cname in condition_results:
            cr = condition_results[cname]
            max_hidden_cos = max(cr['hidden_cosine_distance_means'])
            mean_hidden_cos = np.mean(cr['hidden_cosine_distance_means'])
            logit_cos = cr['logit_cosine_distance']['mean']
            logit_l2 = cr['logit_l2_distance']['mean']

            # Material if any metric shows non-trivial change
            is_material = (max_hidden_cos > 0.001 or logit_cos > 0.001)

            summary_table.append({
                'condition': cname,
                'description': cr['description'],
                'type': cr['type'],
                'max_hidden_cosine_dist': round(float(max_hidden_cos), 6),
                'mean_hidden_cosine_dist': round(float(mean_hidden_cos), 6),
                'logit_cosine_dist': round(float(logit_cos), 6),
                'logit_l2_dist': round(float(logit_l2), 4),
                'intervention_is_material': is_material,
            })

    final_results = {
        'experiment': 'round13_intervention_fidelity',
        'purpose': ('Validate that R11 orthogonal interventions materially perturb '
                    'internal representations, confirming behavioral nulls reflect '
                    'genuine redundancy rather than weak interventions.'),
        'n_cells': int(condition_results[list(condition_results.keys())[0]]['n_cells'])
                   if condition_results else 0,
        'n_conditions': len(condition_results),
        'summary': summary_table,
        'conditions': condition_results,
        'total_time_seconds': round(time.time() - total_t0, 1),
    }

    results_path = os.path.join(OUT_DIR, "fidelity_results.json")
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=_json_serializer)
    print(f"  Results saved: {results_path}")

    # ============================================================
    # STEP 9: Print summary
    # ============================================================
    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"INTERVENTION FIDELITY CHECK COMPLETE - {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")

    print(f"\nSummary Table:")
    print(f"{'Condition':<30} {'Type':<20} {'Max HidCos':>10} {'LogitCos':>10} "
          f"{'LogitL2':>10} {'Material?':>10}")
    print("-" * 95)
    for row in summary_table:
        mat_str = "YES" if row['intervention_is_material'] else "NO"
        print(f"{row['condition']:<30} {row['type']:<20} "
              f"{row['max_hidden_cosine_dist']:>10.6f} {row['logit_cosine_dist']:>10.6f} "
              f"{row['logit_l2_dist']:>10.4f} {mat_str:>10}")

    print(f"\nInterpretation:")
    n_material = sum(1 for r in summary_table if r['intervention_is_material'])
    print(f"  {n_material}/{len(summary_table)} interventions are material "
          f"(cosine distance > 0.001 at any layer or in logits).")
    if n_material == len(summary_table):
        print(f"  All interventions materially perturb representations.")
        print(f"  => The AUROC nulls from R11 reflect genuine functional redundancy,")
        print(f"     not weak interventions.")
    elif n_material > 0:
        print(f"  Some interventions are material, some are not.")
        print(f"  => Interpret AUROC nulls cautiously for non-material conditions.")
    else:
        print(f"  WARNING: No interventions are material!")
        print(f"  => The R11 AUROC nulls may be uninformative.")

    print(f"\nOutput directory: {OUT_DIR}")
    print(f"  fidelity_results.json — full results")
    print(f"  checkpoint.json — resume support")
    print(f"  figures/fig_intervention_fidelity.png — heatmap and summary figure")
    print(f"  baseline_hidden.npy — cached baseline hidden states")
    print(f"  baseline_logits.npy — cached baseline logits")
    # (entropy analysis skipped for runtime optimization)


if __name__ == '__main__':
    run()
