#!/usr/bin/env python3
"""
Round 11 Experiment 3: Orthogonal Causal Interventions
=======================================================
Two intervention families beyond head masking:
  A. Uniform attention replacement — preserves head output magnitude
     but destroys attention patterning
  B. MLP ablation — zeros FFN contribution at specific layers

Addresses: MC4 ("ablation conclusions outrun what head masking can falsify")
           ST4 ("orthogonal causal interventions")
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
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "round11_orthogonal_interventions")
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

# Reuse baseline AUROC from R10 expanded ablation
BASELINE_AUROC = 0.7035


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
            'gene': pg, 'auroc_attn': round(auroc_attn, 4),
            'auroc_corr': round(auroc_corr, 4),
            'n_de': int(labels.sum()), 'n_total': int(len(labels)),
        })

    return np.array(aurocs_attn), np.array(aurocs_corr), per_gene


def extract_with_hooks(model, control_tokens, hooks_setup_fn, hooks_cleanup_fn,
                        token_to_hvg, device, extract_layer, hvg_genes,
                        head_mask_tensor=None):
    """Generic extraction with arbitrary hooks. Returns attention edges dict."""
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

    # Apply custom hooks
    hook_handles = hooks_setup_fn(model)

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

        hm = head_mask_tensor if head_mask_tensor is not None else None
        with torch.no_grad():
            _ = model(input_ids=input_ids_t, attention_mask=attn_mask_t, head_mask=hm)

        if extract_layer in captured_attentions:
            layer_attn = captured_attentions[extract_layer].mean(dim=1).cpu().float().numpy()

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

            del layer_attn

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

    # Cleanup
    model.bert.encoder.layer[extract_layer].forward = orig_forward
    hooks_cleanup_fn(hook_handles)
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


def save_checkpoint(results, completed):
    checkpoint = {'completed': completed, 'results': {}}
    for cname in completed:
        if cname in results:
            r = dict(results[cname])
            if 'per_gene' in r:
                del r['per_gene']
            checkpoint['results'][cname] = r
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"    [Checkpoint: {len(completed)} conditions done]")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r') as f:
            return json.load(f)
    return None


def run():
    import torch
    from transformers import BertForMaskedLM

    total_t0 = time.time()

    print("=" * 70)
    print("ORTHOGONAL CAUSAL INTERVENTIONS (ROUND 11)")
    print("=" * 70)

    # ============================================================
    # STEP 1: Load pre-computed data
    # ============================================================
    print("\nSTEP 1: Loading data...")

    with open(os.path.join(EXTRACTION_DIR, "de_labels.pkl"), 'rb') as f:
        de_labels = pickle.load(f)
    with open(os.path.join(EXTRACTION_DIR, "hvg_gene_list.json")) as f:
        hvg_genes = json.load(f)
    hvg_to_local = {g.upper(): i for i, g in enumerate(hvg_genes)}
    corr_matrix = np.load(os.path.join(EXTRACTION_DIR, "correlation_matrix.npy"))

    # Head rankings for uniform attention targets
    auroc_matrix = np.load(os.path.join(EXTRACTION_DIR, "per_head_trrust", "head_trrust_auroc_matrix.npy"))
    heads_flat = []
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            heads_flat.append((l, h, float(auroc_matrix[l, h])))
    heads_flat.sort(key=lambda x: x[2], reverse=True)
    top5_heads = [(l, h) for l, h, _ in heads_flat[:5]]
    top10_heads = [(l, h) for l, h, _ in heads_flat[:10]]

    # Random heads (deterministic)
    rng = np.random.RandomState(789)
    all_heads_list = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]
    random_10_idx = rng.choice(len(all_heads_list), 10, replace=False)
    random_10_heads = [all_heads_list[i] for i in random_10_idx]

    print(f"  DE labels: {len(de_labels)} perturbations")
    print(f"  HVG genes: {len(hvg_genes)}")
    print(f"  Top-5 TRRUST heads: {top5_heads}")

    # ============================================================
    # STEP 2: Define conditions
    # ============================================================
    print("\nSTEP 2: Defining intervention conditions...")

    # Format: name -> dict with 'type', 'targets', 'description'
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
            'target_layers': [14],  # 0-indexed
            'description': 'Zero MLP output at L15 (best perturbation-first)',
        },
        'mlp_zero_L13_L14_L15': {
            'type': 'mlp_ablation',
            'target_layers': [12, 13, 14],
            'description': 'Zero MLP output at L13+L14+L15',
        },
        'mlp_zero_random_layer': {
            'type': 'mlp_ablation',
            'target_layers': [7],  # Fixed random layer
            'description': 'Zero MLP output at L8 (random control)',
        },
    }

    for cname, cond in conditions.items():
        if cond['type'] == 'uniform_attention':
            print(f"  {cname}: uniform on {len(cond['target_heads'])} heads")
        else:
            print(f"  {cname}: zero MLP at layers {[l+1 for l in cond['target_layers']]}")

    # ============================================================
    # STEP 3: Check checkpoint
    # ============================================================
    results = {}
    completed = []

    # Load baseline from R10 expanded ablation
    r10_path = os.path.join(PAPER_DIR, "experiments", "round10_expanded_ablation", "expanded_ablation_results.json")
    if os.path.exists(r10_path):
        with open(r10_path) as f:
            r10_data = json.load(f)
        baseline_per_gene = r10_data['conditions']['baseline'].get('per_gene', [])
        # If per_gene not in summary, try loading from checkpoint
        if not baseline_per_gene:
            r10_ckpt = os.path.join(PAPER_DIR, "experiments", "round10_expanded_ablation", "checkpoint.json")
            if os.path.exists(r10_ckpt):
                with open(r10_ckpt) as f:
                    ckpt = json.load(f)
                baseline_per_gene = ckpt.get('results', {}).get('baseline', {}).get('per_gene', [])

        results['baseline'] = {
            'description': 'Full model (no intervention)',
            'mean_auroc_attn': r10_data['conditions']['baseline']['mean_auroc_attn'],
            'n_evaluated': r10_data['conditions']['baseline']['n_evaluated'],
            'per_gene': baseline_per_gene,
            'source': 'round10',
        }
        completed.append('baseline')
        print(f"\n  Baseline reused from R10: AUROC={results['baseline']['mean_auroc_attn']}")

    ckpt = load_checkpoint()
    if ckpt:
        for cname in ckpt.get('completed', []):
            if cname not in completed and cname in ckpt.get('results', {}):
                results[cname] = ckpt['results'][cname]
                completed.append(cname)
        print(f"  Checkpoint: {len(completed)} conditions done")

    remaining = [c for c in conditions if c not in completed]
    print(f"  Remaining: {len(remaining)} conditions")

    if not remaining:
        print("  All conditions complete!")
    else:
        # ============================================================
        # STEP 4: Load data and tokenize
        # ============================================================
        print("\nSTEP 4: Loading data and tokenizing...")
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

        print(f"  Tokenized: {len(control_tokens)} cells, time: {time.time()-t1:.1f}s")

        # ============================================================
        # STEP 5: Load model and run conditions
        # ============================================================
        print("\nSTEP 5: Loading model...")

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
        print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

        for ci, cond_name in enumerate(remaining):
            cond = conditions[cond_name]
            print(f"\n{'='*60}")
            print(f"  Condition {ci+1}/{len(remaining)}: {cond_name}")
            print(f"  {cond['description']}")
            print(f"{'='*60}")
            t_cond = time.time()

            if cond['type'] == 'uniform_attention':
                # Intervention A: replace attention distributions with uniform
                target_heads = set((l, h) for l, h in cond['target_heads'])

                def setup_uniform_hooks(mdl):
                    handles = []
                    for l, h in target_heads:
                        attn_module = mdl.bert.encoder.layer[l].attention.self

                        def make_hook(head_idx):
                            def hook_fn(module, args, output):
                                # output is (context_layer, attention_probs) when output_attentions=True
                                # or just (context_layer,)
                                # We modify the attention module's internal state
                                # Actually, we need a pre-forward hook that modifies
                                # attention_probs before they're used
                                pass
                            return hook_fn

                        # Better approach: use a forward hook on BertSelfAttention
                        # that replaces the attention weights with uniform
                        orig_attn_forward = attn_module.forward

                        def make_uniform_forward(orig_fn, target_head_indices, n_total_heads):
                            def uniform_forward(hidden_states, attention_mask=None,
                                              head_mask=None, encoder_hidden_states=None,
                                              past_key_values=None, output_attentions=False,
                                              cache_position=None, **kwargs):
                                # Call original to get result with attention probs
                                result = orig_fn(
                                    hidden_states, attention_mask=attention_mask,
                                    head_mask=head_mask,
                                    encoder_hidden_states=encoder_hidden_states,
                                    past_key_values=past_key_values,
                                    output_attentions=True,
                                    cache_position=cache_position, **kwargs
                                )
                                if len(result) > 1 and result[1] is not None:
                                    attn_probs = result[1]  # (batch, heads, seq, seq)
                                    seq_len = attn_probs.shape[-1]
                                    for hi in target_head_indices:
                                        attn_probs[:, hi, :, :] = 1.0 / seq_len
                                    # Recompute context with modified attention
                                    # Unfortunately we can't easily recompute context_layer
                                    # without access to value_layer
                                    # So instead, we hook at a different level
                                return result
                            return uniform_forward

                    # Better approach: Hook into the attention computation
                    # by intercepting the softmax output
                    handles = []
                    layers_with_hooks = set()
                    target_by_layer = {}
                    for l, h in target_heads:
                        if l not in target_by_layer:
                            target_by_layer[l] = []
                        target_by_layer[l].append(h)

                    for layer_idx, head_indices in target_by_layer.items():
                        attn_module = mdl.bert.encoder.layer[layer_idx].attention.self
                        orig_fn = attn_module.forward

                        def make_patched(orig, heads_to_uniform):
                            def patched(hidden_states, attention_mask=None,
                                       head_mask=None, encoder_hidden_states=None,
                                       past_key_values=None, output_attentions=False,
                                       cache_position=None, **kwargs):
                                # Get value_layer before attention
                                # We need to intercept and recompute
                                result = orig(
                                    hidden_states, attention_mask=attention_mask,
                                    head_mask=head_mask,
                                    encoder_hidden_states=encoder_hidden_states,
                                    past_key_values=past_key_values,
                                    output_attentions=output_attentions,
                                    cache_position=cache_position, **kwargs
                                )
                                return result
                            return patched

                        # Simplest working approach: register a hook on the dropout
                        # that follows softmax in BERT attention
                        # In BertSelfAttention, after softmax, there's self.dropout
                        # attn_probs = self.dropout(attn_probs)
                        # context_layer = torch.matmul(attn_probs, value_layer)
                        # We can hook the dropout to replace specific heads with uniform

                        dropout_module = attn_module.dropout

                        def make_dropout_hook(heads_to_uniform_list):
                            def hook(module, input, output):
                                # output shape: (batch, num_heads, seq_len, seq_len)
                                if output.dim() == 4:
                                    seq_len = output.shape[-1]
                                    for hi in heads_to_uniform_list:
                                        if hi < output.shape[1]:
                                            output[:, hi, :, :] = 1.0 / seq_len
                                return output
                            return hook

                        handle = dropout_module.register_forward_hook(
                            make_dropout_hook(head_indices))
                        handles.append(handle)

                    return handles

                def cleanup_hooks(handles):
                    for h in handles:
                        h.remove()

                edges = extract_with_hooks(
                    model, control_tokens, setup_uniform_hooks, cleanup_hooks,
                    token_to_hvg, device, EXTRACT_LAYER, hvg_genes,
                )

            elif cond['type'] == 'mlp_ablation':
                # Intervention B: zero the MLP/FFN output at specific layers
                target_layers = cond['target_layers']

                def setup_mlp_hooks(mdl):
                    handles = []
                    for layer_idx in target_layers:
                        # In BERT, the FFN is in bert.encoder.layer[i].output
                        # BertOutput: dense + LayerNorm + dropout
                        # output = self.LayerNorm(self.dropout(self.dense(hidden_states)) + input_tensor)
                        # We want to zero the dense output (before residual)
                        # Hook on the dense layer to zero its output
                        dense_module = mdl.bert.encoder.layer[layer_idx].output.dense

                        def make_zero_hook():
                            def hook(module, input, output):
                                return torch.zeros_like(output)
                            return hook

                        handle = dense_module.register_forward_hook(make_zero_hook())
                        handles.append(handle)
                    return handles

                def cleanup_hooks(handles):
                    for h in handles:
                        h.remove()

                edges = extract_with_hooks(
                    model, control_tokens, setup_mlp_hooks, cleanup_hooks,
                    token_to_hvg, device, EXTRACT_LAYER, hvg_genes,
                )

            # Evaluate
            aurocs_attn, aurocs_corr, per_gene = evaluate_perturbation_first(
                edges, corr_matrix, de_labels, hvg_genes, hvg_to_local,
                min_de=MIN_DE_TARGETS,
            )

            results[cond_name] = {
                'description': cond['description'],
                'type': cond['type'],
                'n_evaluated': len(aurocs_attn),
                'mean_auroc_attn': round(float(aurocs_attn.mean()), 4) if len(aurocs_attn) > 0 else None,
                'std_auroc_attn': round(float(aurocs_attn.std()), 4) if len(aurocs_attn) > 0 else None,
                'mean_auroc_corr': round(float(aurocs_corr.mean()), 4) if len(aurocs_corr) > 0 else None,
                'per_gene': per_gene,
                'source': 'round11',
            }

            cond_time = time.time() - t_cond
            print(f"    AUROC: {results[cond_name]['mean_auroc_attn']:.4f}")
            print(f"    Evaluated: {len(aurocs_attn)} perturbations")
            print(f"    Time: {cond_time:.0f}s ({cond_time/60:.1f} min)")

            completed.append(cond_name)
            save_checkpoint(results, completed)

            del edges
            gc.collect()
            if device.type == 'mps':
                torch.mps.empty_cache()

        del model
        gc.collect()

    # ============================================================
    # STEP 6: Statistical analysis
    # ============================================================
    print("\n\nSTEP 6: Statistical analysis...")

    baseline_pg = {}
    if 'baseline' in results and results['baseline'].get('per_gene'):
        baseline_pg = {g['gene']: g['auroc_attn'] for g in results['baseline']['per_gene']}
    baseline_genes = sorted(baseline_pg.keys())

    comparisons = {}
    for cond_name in results:
        if cond_name == 'baseline':
            continue
        if not results[cond_name].get('per_gene'):
            continue

        cond_pg = {g['gene']: g['auroc_attn'] for g in results[cond_name]['per_gene']}
        common = sorted(set(baseline_genes) & set(cond_pg.keys()))
        if len(common) < 10:
            print(f"  {cond_name}: only {len(common)} common genes, skipping")
            continue

        b_arr = np.array([baseline_pg[g] for g in common])
        c_arr = np.array([cond_pg[g] for g in common])
        diff = b_arr - c_arr

        try:
            w_stat, w_p = stats.wilcoxon(diff)
        except:
            w_p = 1.0
        try:
            t_stat, t_p = stats.ttest_rel(b_arr, c_arr)
        except:
            t_p = 1.0

        cohens_d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0

        comparisons[cond_name] = {
            'n_compared': len(common),
            'mean_drop': round(float(diff.mean()), 4),
            'std_drop': round(float(diff.std()), 4),
            'wilcoxon_p': float(w_p),
            'ttest_p': float(t_p),
            'cohens_d': round(cohens_d, 4),
            'n_baseline_better': int((diff > 0).sum()),
            'n_ablation_better': int((diff < 0).sum()),
            'ci_95': [round(float(np.percentile(diff, 2.5)), 4),
                      round(float(np.percentile(diff, 97.5)), 4)],
        }

        print(f"\n  {cond_name}:")
        print(f"    AUROC: {results[cond_name]['mean_auroc_attn']}")
        print(f"    Drop: {comparisons[cond_name]['mean_drop']:+.4f}")
        print(f"    Wilcoxon p={comparisons[cond_name]['wilcoxon_p']:.4f}")
        print(f"    Cohen's d={comparisons[cond_name]['cohens_d']:.4f}")

    # ============================================================
    # STEP 7: Generate figure
    # ============================================================
    print("\n\nSTEP 7: Generating figure...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: All conditions bar chart
    ax = axes[0]
    cond_order = ['baseline']
    cond_order += ['uniform_attn_reg_top5', 'uniform_attn_reg_top10', 'uniform_attn_random_10']
    cond_order += ['mlp_zero_L15', 'mlp_zero_L13_L14_L15', 'mlp_zero_random_layer']
    cond_order = [c for c in cond_order if c in results]

    aurocs = [results[c]['mean_auroc_attn'] or 0 for c in cond_order]
    labels = ['Baseline', 'Unif\nReg5', 'Unif\nReg10', 'Unif\nRnd10',
              'MLP\nL15', 'MLP\nL13-15', 'MLP\nL8'][:len(cond_order)]
    colors = ['#2ecc71', '#e74c3c', '#c0392b', '#3498db',
              '#9b59b6', '#8e44ad', '#2980b9'][:len(cond_order)]

    ax.bar(range(len(aurocs)), aurocs, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(aurocs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Mean Perturbation-First AUROC')
    ax.set_title('A. Orthogonal Intervention Results')
    if 'baseline' in results:
        ax.axhline(y=results['baseline']['mean_auroc_attn'], color='green',
                   linestyle='--', alpha=0.5, label=f"Baseline={results['baseline']['mean_auroc_attn']:.3f}")
    ax.legend(fontsize=8)
    for i, v in enumerate(aurocs):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=7)

    # Panel B: Drop comparison
    ax = axes[1]
    drop_conds = [c for c in cond_order if c in comparisons]
    if drop_conds:
        drops = [comparisons[c]['mean_drop'] for c in drop_conds]
        drop_labels = [labels[cond_order.index(c)] for c in drop_conds]
        drop_colors = [colors[cond_order.index(c)] for c in drop_conds]
        ax.barh(range(len(drops)), drops, color=drop_colors, alpha=0.8)
        ax.set_yticks(range(len(drops)))
        ax.set_yticklabels(drop_labels, fontsize=8)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('AUROC Drop from Baseline')
        ax.set_title('B. Intervention Effects')
        for i, (d, c) in enumerate(zip(drops, drop_conds)):
            p = comparisons[c]['wilcoxon_p']
            sig = '*' if p < 0.05 else 'ns'
            ax.text(d + 0.001 if d >= 0 else d - 0.001,
                    i, f'{d:+.3f} ({sig})',
                    va='center', fontsize=7,
                    ha='left' if d >= 0 else 'right')

    # Panel C: Intervention family comparison
    ax = axes[2]
    families = {
        'Head masking\n(R10 reg_top5)': {'auroc': 0.7037, 'drop': -0.0002},
        'Head masking\n(R10 reg_top10)': {'auroc': 0.7035, 'drop': 0.0},
    }
    for c in ['uniform_attn_reg_top5', 'uniform_attn_reg_top10',
              'mlp_zero_L15', 'mlp_zero_L13_L14_L15']:
        if c in comparisons:
            short = c.replace('uniform_attn_', 'Uniform\n').replace('mlp_zero_', 'MLP zero\n')
            families[short] = {
                'auroc': results[c]['mean_auroc_attn'],
                'drop': comparisons[c]['mean_drop'],
            }

    if families:
        fam_names = list(families.keys())
        fam_drops = [families[n]['drop'] for n in fam_names]
        fam_colors_list = ['#95a5a6', '#7f8c8d'] + ['#e74c3c', '#c0392b', '#9b59b6', '#8e44ad']
        fam_colors_list = fam_colors_list[:len(fam_names)]

        ax.barh(range(len(fam_names)), fam_drops, color=fam_colors_list, alpha=0.8)
        ax.set_yticks(range(len(fam_names)))
        ax.set_yticklabels(fam_names, fontsize=7)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('AUROC Drop from Baseline')
        ax.set_title('C. Cross-Intervention Comparison')

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_orthogonal_interventions.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")

    # ============================================================
    # STEP 8: Save results
    # ============================================================
    print("\nSTEP 8: Saving results...")

    results_summary = {}
    for cname, r in results.items():
        rs = {k: v for k, v in r.items() if k != 'per_gene'}
        rs['n_per_gene'] = len(r.get('per_gene', []))
        results_summary[cname] = rs

    final = {
        'experiment': 'round11_orthogonal_interventions',
        'n_conditions': len(results),
        'conditions': results_summary,
        'comparisons': comparisons,
        'total_time_seconds': round(time.time() - total_t0, 1),
    }

    with open(os.path.join(OUT_DIR, "orthogonal_results.json"), 'w') as f:
        json.dump(final, f, indent=2)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"ORTHOGONAL INTERVENTIONS COMPLETE - {total_time:.0f}s ({total_time/60:.1f} min)")
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
