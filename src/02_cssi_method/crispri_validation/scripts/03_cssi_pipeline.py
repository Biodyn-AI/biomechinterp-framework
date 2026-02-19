import platform
if platform.system() == 'Windows':
    exec(open('C:/Users/Agent/.openclaw/workspace/patch_torchtext.py').read())
else:
    exec(open('/mnt/c/Users/Agent/.openclaw/workspace/patch_torchtext.py').read())

"""
CSSI (Cell-type Specific Subnetwork Inference) CRISPRi Validation Pipeline
==========================================================================
Tests whether CSSI-weighted attention improves CRISPRi validation over raw attention.

CSSI approach:
  1. For each gene pair, collect per-cell attention values (not just mean)
  2. Compute CV (coefficient of variation) across cells as cell-type specificity index
  3. Weight attention by CV: CSSI_score = mean_attention * CV
  4. Multi-layer CSSI: test layers 10-17, pick best layer per gene pair

Previous result: Layer 13 raw attention AUROC = 0.487 (below random 0.498)
"""

import os, sys, json, time, gc, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import h5py
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# CONFIGURATION
# ============================================================
def wsl_to_win(p):
    """Convert WSL path to Windows path if running on Windows."""
    if platform.system() == 'Windows' and p.startswith('/mnt/'):
        drive = p[5]
        return drive.upper() + ':' + p[6:].replace('/', '\\')
    return p

_DATA_PATH = "/mnt/d/openclaw/biodyn-nmi-paper/experiments/crispri_validation/data/replogle_concat.h5ad"
_MODEL_DIR = "/mnt/d/openclaw/intelligence-augmentation/models/Geneformer/Geneformer-V2-316M"
_TOKEN_DICT_PATH = "/mnt/d/openclaw/intelligence-augmentation/models/Geneformer/geneformer/token_dictionary_gc104M.pkl"
_GENE_MEDIAN_PATH = "/mnt/d/openclaw/intelligence-augmentation/models/Geneformer/geneformer/gene_median_dictionary_gc104M.pkl"
_GENE_NAME_ID_PATH = "/mnt/d/openclaw/intelligence-augmentation/models/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
_RESULTS_DIR = "/mnt/d/openclaw/biodyn-nmi-paper/experiments/crispri_validation/results"

DATA_PATH = wsl_to_win(_DATA_PATH)
MODEL_DIR = wsl_to_win(_MODEL_DIR)
TOKEN_DICT_PATH = wsl_to_win(_TOKEN_DICT_PATH)
GENE_MEDIAN_PATH = wsl_to_win(_GENE_MEDIAN_PATH)
GENE_NAME_ID_PATH = wsl_to_win(_GENE_NAME_ID_PATH)
RESULTS_DIR = wsl_to_win(_RESULTS_DIR)

# CSSI: test layers 10-17 (deep layers most likely to encode regulation)
CSSI_LAYERS = [10, 11, 12, 13, 14, 15, 16, 17]

MAX_CONTROL_CELLS = 500
MAX_PERTURBED_GENES = 100
MIN_CELLS_PER_GENE = 50
BATCH_SIZE = 2  # Reduced for 8-layer simultaneous capture
DE_PVALUE_THRESHOLD = 0.05
DE_LOG2FC_THRESHOLD = 0.25
CELL_LINE = "k562"
MAX_SEQ_LEN = 2048

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# STEP 1: Load data
# ============================================================
print("=" * 70)
print("CSSI PIPELINE - STEP 1: Loading data")
print("=" * 70)
t0 = time.time()

def load_categorical_column(h5group, col_name):
    col = h5group[col_name]
    if isinstance(col, h5py.Group):
        categories = col['categories'][:]
        codes = col['codes'][:]
        if categories.dtype.kind in ('O', 'S'):
            categories = np.array([x.decode() if isinstance(x, bytes) else x for x in categories])
        return categories[codes]
    data = col[:]
    if data.dtype.kind in ('O', 'S'):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in data])
    return data

with h5py.File(DATA_PATH, 'r') as f:
    cell_genes = load_categorical_column(f['obs'], 'gene')
    cell_lines = load_categorical_column(f['obs'], 'cell_line')
    var_genes = load_categorical_column(f['var'], 'gene_name_index')
    n_cells, n_genes = f['X'].shape

k562_mask = (cell_lines == CELL_LINE)
k562_control_mask = k562_mask & (cell_genes == 'non-targeting')
print(f"  K562: {k562_mask.sum():,} cells, Controls: {k562_control_mask.sum():,}")

# Select perturbed genes
k562_genes_arr = cell_genes[k562_mask]
unique_perts, pert_counts = np.unique(k562_genes_arr, return_counts=True)
valid_perts = []
for gn, cnt in zip(unique_perts, pert_counts):
    gl = gn.lower()
    if 'non-targeting' not in gl and gl not in ('control', 'ctrl', 'ntc', 'nt') and cnt >= MIN_CELLS_PER_GENE:
        valid_perts.append((gn, int(cnt)))
valid_perts.sort(key=lambda x: -x[1])
selected_genes = valid_perts[:MAX_PERTURBED_GENES]
print(f"  Selected perturbations: {len(selected_genes)}")
print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# STEP 2: Gene mappings
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Gene mappings")
print("=" * 70)

with open(TOKEN_DICT_PATH, 'rb') as f:
    token_dict = pickle.load(f)
with open(GENE_MEDIAN_PATH, 'rb') as f:
    gene_median_dict = pickle.load(f)
with open(GENE_NAME_ID_PATH, 'rb') as f:
    gene_name_id_dict = pickle.load(f)

var_gene_to_idx = {g: i for i, g in enumerate(var_genes)}
mapped_var_indices, mapped_token_ids_list, mapped_medians_list, mapped_gene_names_list = [], [], [], []

for i, gn in enumerate(var_genes):
    eid = gene_name_id_dict.get(gn)
    if eid and eid in token_dict:
        mapped_var_indices.append(i)
        mapped_token_ids_list.append(token_dict[eid])
        mapped_medians_list.append(gene_median_dict.get(eid, 1.0))
        mapped_gene_names_list.append(gn)

mapped_var_indices = np.array(mapped_var_indices)
mapped_token_ids = np.array(mapped_token_ids_list)
mapped_medians = np.array(mapped_medians_list)
token_to_genename = {t: g for t, g in zip(mapped_token_ids_list, mapped_gene_names_list)}

selected_gene_names_set = set(g[0] for g in selected_genes)
selected_gene_token_ids = set()
for gn in selected_gene_names_set:
    eid = gene_name_id_dict.get(gn)
    if eid and eid in token_dict:
        selected_gene_token_ids.add(token_dict[eid])

print(f"  Mapped: {len(mapped_var_indices)}/{len(var_genes)}, Selected tokens: {len(selected_gene_token_ids)}")

# ============================================================
# STEP 3: Tokenize control cells
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Tokenizing control cells")
print("=" * 70)

def tokenize_cell(expr_vec, var_idx, tok_ids, medians, max_len=4096):
    expr = expr_vec[var_idx]
    nz = expr > 0
    if nz.sum() == 0:
        return None
    e, t, m = expr[nz], tok_ids[nz], medians[nz]
    with np.errstate(divide='ignore', invalid='ignore'):
        norm = e / m
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0)
    order = np.argsort(-norm)
    ranked = t[order][:max_len - 2]
    return np.concatenate([[2], ranked, [3]]).astype(np.int64)

control_indices = np.where(k562_control_mask)[0]
rng = np.random.RandomState(42)
if len(control_indices) > MAX_CONTROL_CELLS:
    control_indices = rng.choice(control_indices, MAX_CONTROL_CELLS, replace=False)
    control_indices.sort()

t1 = time.time()
control_tokens = []
with h5py.File(DATA_PATH, 'r') as f:
    X = f['X']
    for i, ci in enumerate(control_indices):
        t_cell = tokenize_cell(X[int(ci), :], mapped_var_indices, mapped_token_ids, mapped_medians)
        if t_cell is not None:
            control_tokens.append(t_cell)
        if (i+1) % 100 == 0:
            print(f"    {i+1}/{len(control_indices)}...")

# Truncate
for i in range(len(control_tokens)):
    if len(control_tokens[i]) > MAX_SEQ_LEN:
        control_tokens[i] = np.concatenate([control_tokens[i][:MAX_SEQ_LEN-1], [control_tokens[i][-1]]])

# Sort by length for efficient batching
sort_idx = sorted(range(len(control_tokens)), key=lambda i: len(control_tokens[i]))
control_tokens = [control_tokens[i] for i in sort_idx]
print(f"  {len(control_tokens)} cells, avg {np.mean([len(t) for t in control_tokens]):.0f} tokens")
print(f"  Time: {time.time()-t1:.1f}s")

# ============================================================
# STEP 4: CSSI Attention Extraction (ALL LAYERS IN SINGLE PASS)
# ============================================================
# KEY DIFFERENCE FROM V1: Compute CV (coefficient of variation) across cells.
# OPTIMIZATION: Hook ALL 8 layers simultaneously, single forward pass per batch.
# MEMORY: Use online Welford's algorithm (sum, sum_sq, count) instead of storing
# all per-cell values.
print("\n" + "=" * 70)
print("STEP 4: CSSI multi-layer attention extraction (single pass)")
print("=" * 70)

import torch
from transformers import BertForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
if device.type == 'cuda':
    print(f"  GPU free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

print("  Loading Geneformer V2-316M...")
t2 = time.time()
model = BertForMaskedLM.from_pretrained(
    MODEL_DIR, output_attentions=False, output_hidden_states=False,
    torch_dtype=torch.float16, attn_implementation="eager",
)
model = model.to(device)
model.eval()
print(f"  Loaded in {time.time()-t2:.1f}s")
if device.type == 'cuda':
    print(f"  GPU free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

# Hook ALL target layers at once
captured_attentions = {}  # layer_idx -> attention tensor
orig_forwards = {}

for layer_idx in CSSI_LAYERS:
    layer_module = model.bert.encoder.layer[layer_idx]
    orig_forwards[layer_idx] = layer_module.forward

    def make_hook(li, orig):
        def hooked_forward(hidden_states, attention_mask=None, head_mask=None,
                           encoder_hidden_states=None, past_key_values=None,
                           output_attentions=False, cache_position=None, **kwargs):
            result = orig(hidden_states, attention_mask=attention_mask, head_mask=head_mask,
                          encoder_hidden_states=encoder_hidden_states, past_key_values=past_key_values,
                          output_attentions=True, cache_position=cache_position, **kwargs)
            if len(result) > 1 and result[1] is not None:
                # Move to CPU immediately to free GPU memory
                captured_attentions[li] = result[1].detach().cpu()
            return (result[0],)
        return hooked_forward

    layer_module.forward = make_hook(layer_idx, orig_forwards[layer_idx])

print(f"  Hooked {len(CSSI_LAYERS)} layers: {CSSI_LAYERS}")

# NUMPY ARRAY APPROACH: Use compact index arrays instead of dicts
# Map selected gene token IDs to compact indices [0, n_selected)
# Map ALL token IDs seen to compact indices [0, n_all_tokens)
# Then accumulate sum/sumsq/count as numpy arrays: [n_layers, n_selected, n_all_tokens]

# Build token ID -> compact index mapping
all_token_ids = sorted(set(mapped_token_ids_list))
tok_to_compact = {t: i for i, t in enumerate(all_token_ids)}
n_all_toks = len(all_token_ids)

selected_gene_token_list = sorted(selected_gene_token_ids)
sel_tok_to_compact = {t: i for i, t in enumerate(selected_gene_token_list)}
n_sel = len(selected_gene_token_list)

print(f"  Compact indices: {n_sel} selected genes, {n_all_toks} total tokens")

# Accumulators per layer: shape [n_sel, n_all_toks]
# fwd = selected -> target, rev = target -> selected
n_layers_cssi = len(CSSI_LAYERS)
layer_to_idx = {l: i for i, l in enumerate(CSSI_LAYERS)}

# Use float64 for numerical stability with Welford's
fwd_sum = np.zeros((n_layers_cssi, n_sel, n_all_toks), dtype=np.float64)
fwd_ssq = np.zeros((n_layers_cssi, n_sel, n_all_toks), dtype=np.float64)
fwd_cnt = np.zeros((n_layers_cssi, n_sel, n_all_toks), dtype=np.int32)
rev_sum = np.zeros((n_layers_cssi, n_sel, n_all_toks), dtype=np.float64)
rev_ssq = np.zeros((n_layers_cssi, n_sel, n_all_toks), dtype=np.float64)
rev_cnt = np.zeros((n_layers_cssi, n_sel, n_all_toks), dtype=np.int32)

# Memory estimate: 6 arrays * 8 layers * 97 * 6332 * 8 bytes = ~237 MB (ok for 16GB RAM)
mem_mb = 6 * n_layers_cssi * n_sel * n_all_toks * 8 / 1e6
print(f"  Accumulator memory: {mem_mb:.0f} MB")

n_batches = (len(control_tokens) + BATCH_SIZE - 1) // BATCH_SIZE
cells_done = 0
t3 = time.time()

for batch_start in range(0, len(control_tokens), BATCH_SIZE):
    batch_toks = control_tokens[batch_start:batch_start + BATCH_SIZE]
    max_len = max(len(t) for t in batch_toks)
    bs = len(batch_toks)

    input_ids = np.full((bs, max_len), 0, dtype=np.int64)
    attn_mask = np.zeros((bs, max_len), dtype=np.int64)
    for i, toks in enumerate(batch_toks):
        input_ids[i, :len(toks)] = toks
        attn_mask[i, :len(toks)] = 1

    with torch.no_grad():
        _ = model(
            input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
            attention_mask=torch.tensor(attn_mask, dtype=torch.long, device=device),
        )

    for cell_i in range(bs):
        seq_len = int(attn_mask[cell_i].sum())
        cell_tok = input_ids[cell_i, :seq_len]
        gene_start = 1
        gene_tokens = cell_tok[gene_start:seq_len - 1]
        n_gt = len(gene_tokens)

        # Map gene tokens to compact indices, find selected positions
        compact_idx = np.full(n_gt, -1, dtype=np.int32)
        sel_compact_idx = np.full(n_gt, -1, dtype=np.int32)
        for pos in range(n_gt):
            t = int(gene_tokens[pos])
            if t in tok_to_compact:
                compact_idx[pos] = tok_to_compact[t]
            if t in sel_tok_to_compact:
                sel_compact_idx[pos] = sel_tok_to_compact[t]

        # Positions with selected genes
        sel_positions = np.where(sel_compact_idx >= 0)[0]
        if len(sel_positions) == 0:
            continue

        # Valid target positions (token > 3 and has compact index)
        valid_positions = np.where((gene_tokens > 3) & (compact_idx >= 0))[0]
        valid_compact = compact_idx[valid_positions]

        for layer_idx in CSSI_LAYERS:
            if layer_idx not in captured_attentions:
                continue
            li = layer_to_idx[layer_idx]
            # Mean across heads, extract gene submatrix
            gene_attn = captured_attentions[layer_idx][cell_i].mean(dim=0).float().numpy()[
                gene_start:gene_start+n_gt, gene_start:gene_start+n_gt]

            for sp in sel_positions:
                si = sel_compact_idx[sp]  # compact selected index

                # Forward: attention from selected gene to all valid targets
                fwd_vals = gene_attn[sp, valid_positions].astype(np.float64)
                # Reverse: attention from all valid targets to selected gene
                rev_vals = gene_attn[valid_positions, sp].astype(np.float64)

                # Skip self-attention (where valid_position == sp)
                self_mask = valid_positions != sp

                # Vectorized accumulation using fancy indexing
                vc = valid_compact[self_mask]
                fv = fwd_vals[self_mask]
                rv = rev_vals[self_mask]

                # np.add.at handles duplicate indices correctly
                np.add.at(fwd_sum[li, si], vc, fv)
                np.add.at(fwd_ssq[li, si], vc, fv * fv)
                np.add.at(fwd_cnt[li, si], vc, 1)
                np.add.at(rev_sum[li, si], vc, rv)
                np.add.at(rev_ssq[li, si], vc, rv * rv)
                np.add.at(rev_cnt[li, si], vc, 1)

    cells_done += bs
    captured_attentions.clear()
    torch.cuda.empty_cache()

    bi = batch_start // BATCH_SIZE
    if (bi + 1) % 10 == 0 or bi == n_batches - 1:
        elapsed = time.time() - t3
        # Count non-zero entries in L13 forward
        li_13 = layer_to_idx.get(13, 0)
        n_nz = int((fwd_cnt[li_13] > 0).sum())
        print(f"    Batch {bi+1}/{n_batches}, cells: {cells_done}, L13 edges: {n_nz:,}, time: {elapsed:.0f}s")

# Restore original forwards
for layer_idx in CSSI_LAYERS:
    model.bert.encoder.layer[layer_idx].forward = orig_forwards[layer_idx]

print(f"\n  Attention extraction time: {time.time()-t3:.0f}s")

# Compute CSSI scores from numpy arrays
print("  Computing CSSI scores from accumulator arrays...")
cssi_scores_by_layer = {}
raw_mean_by_layer = {}

# Reverse lookup: compact index -> token ID -> gene name
compact_to_genename = {}
for tok, ci in tok_to_compact.items():
    gn = token_to_genename.get(tok)
    if gn:
        compact_to_genename[ci] = gn

sel_compact_to_genename = {}
for tok, si in sel_tok_to_compact.items():
    gn = token_to_genename.get(tok)
    if gn:
        sel_compact_to_genename[si] = gn

for layer_idx in CSSI_LAYERS:
    li = layer_to_idx[layer_idx]
    cssi_edges = {}
    raw_edges = {}

    for si in range(n_sel):
        src_g = sel_compact_to_genename.get(si)
        if not src_g:
            continue

        for direction, s_arr, ssq_arr, c_arr in [
            ('fwd', fwd_sum[li, si], fwd_ssq[li, si], fwd_cnt[li, si]),
            ('rev', rev_sum[li, si], rev_ssq[li, si], rev_cnt[li, si]),
        ]:
            # Find non-zero count entries
            nz = np.where(c_arr > 0)[0]
            for ci in nz:
                tgt_g = compact_to_genename.get(ci)
                if not tgt_g:
                    continue

                cnt = c_arr[ci]
                mean_attn = s_arr[ci] / cnt
                variance = max(0.0, ssq_arr[ci] / cnt - mean_attn * mean_attn)
                std_attn = variance ** 0.5

                if mean_attn > 1e-10:
                    cv = std_attn / mean_attn
                else:
                    cv = 0.0

                cssi_score = mean_attn * cv
                raw_score = mean_attn

                if direction == 'fwd':
                    key = (src_g, tgt_g)
                else:
                    key = (tgt_g, src_g)

                cssi_edges[key] = max(cssi_edges.get(key, 0.0), cssi_score)
                raw_edges[key] = max(raw_edges.get(key, 0.0), raw_score)

    cssi_scores_by_layer[layer_idx] = cssi_edges
    raw_mean_by_layer[layer_idx] = raw_edges
    print(f"    Layer {layer_idx}: {len(cssi_edges):,} gene-pair edges")

del fwd_sum, fwd_ssq, fwd_cnt, rev_sum, rev_ssq, rev_cnt
gc.collect()

# Free GPU
del model
torch.cuda.empty_cache()
gc.collect()

print(f"\n  Total attention extraction time: {time.time()-t2:.0f}s")

# ============================================================
# STEP 5: Build prediction methods
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Building prediction scores")
print("=" * 70)

# Method 1: Raw attention L13 (baseline, reproduces V1)
raw_l13 = raw_mean_by_layer.get(13, {})

# Method 2: CSSI-weighted L13
cssi_l13 = cssi_scores_by_layer.get(13, {})

# Method 3: Raw attention best layer per pair (max across layers 10-17)
raw_best_layer = {}
for layer_idx in CSSI_LAYERS:
    for key, val in raw_mean_by_layer[layer_idx].items():
        raw_best_layer[key] = max(raw_best_layer.get(key, 0.0), val)

# Method 4: CSSI best layer per pair (max CSSI score across layers 10-17)
cssi_best_layer = {}
for layer_idx in CSSI_LAYERS:
    for key, val in cssi_scores_by_layer[layer_idx].items():
        cssi_best_layer[key] = max(cssi_best_layer.get(key, 0.0), val)

# Method 5: CSSI mean across layers
cssi_mean_layers = {}
cssi_layer_counts = {}
for layer_idx in CSSI_LAYERS:
    for key, val in cssi_scores_by_layer[layer_idx].items():
        if key in cssi_mean_layers:
            cssi_mean_layers[key] += val
            cssi_layer_counts[key] += 1
        else:
            cssi_mean_layers[key] = val
            cssi_layer_counts[key] = 1
for key in cssi_mean_layers:
    cssi_mean_layers[key] /= cssi_layer_counts[key]

print(f"  Raw L13 edges: {len(raw_l13):,}")
print(f"  CSSI L13 edges: {len(cssi_l13):,}")
print(f"  Raw best-layer edges: {len(raw_best_layer):,}")
print(f"  CSSI best-layer edges: {len(cssi_best_layer):,}")
print(f"  CSSI mean-layer edges: {len(cssi_mean_layers):,}")

# ============================================================
# STEP 6: DE ground truth
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Differential expression (ground truth)")
print("=" * 70)
t4 = time.time()

print("  Loading control expression...")
with h5py.File(DATA_PATH, 'r') as f:
    X = f['X']
    control_expr = np.empty((len(control_indices), n_genes), dtype=np.float32)
    for ci, idx in enumerate(control_indices):
        control_expr[ci, :] = X[int(idx), :]

de_results = {}
for gene_idx, (pert_gene, cell_cnt) in enumerate(selected_genes):
    pert_mask = (cell_genes == pert_gene) & k562_mask
    pert_indices_g = np.where(pert_mask)[0]
    if len(pert_indices_g) > 200:
        r = np.random.RandomState(hash(pert_gene) % 2**31)
        pert_indices_g = r.choice(pert_indices_g, 200, replace=False)
        pert_indices_g.sort()

    with h5py.File(DATA_PATH, 'r') as f:
        X = f['X']
        pert_expr = np.empty((len(pert_indices_g), n_genes), dtype=np.float32)
        for pi, idx in enumerate(pert_indices_g):
            pert_expr[pi, :] = X[int(idx), :]

    gene_de = {}
    for j in range(n_genes):
        cv = control_expr[:, j]
        pv = pert_expr[:, j]
        if cv.std() < 1e-10 and pv.std() < 1e-10:
            continue
        log2fc = np.log2((pv.mean() + 1e-6) / (cv.mean() + 1e-6))
        try:
            _, pval = stats.mannwhitneyu(pv, cv, alternative='two-sided')
        except ValueError:
            continue
        dg = var_genes[j]
        is_de = (pval < DE_PVALUE_THRESHOLD) and (abs(log2fc) > DE_LOG2FC_THRESHOLD)
        gene_de[dg] = {'pval': float(pval), 'log2fc': float(log2fc), 'is_de': bool(is_de)}

    de_results[pert_gene] = gene_de
    if (gene_idx + 1) % 20 == 0 or gene_idx == 0:
        n_de = sum(1 for v in gene_de.values() if v['is_de'])
        print(f"  [{gene_idx+1}/{len(selected_genes)}] {pert_gene}: {n_de} DE, time: {time.time()-t4:.0f}s")

    del pert_expr
    gc.collect()

total_de = sum(sum(1 for v in d.values() if v['is_de']) for d in de_results.values())
print(f"\n  Total DE pairs: {total_de:,}")
print(f"  DE time: {time.time()-t4:.0f}s")

# ============================================================
# STEP 7: Correlation baseline
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Correlation baseline")
print("=" * 70)
t5 = time.time()

correlation_edges = {}
for pg, _ in selected_genes:
    if pg not in var_gene_to_idx:
        continue
    pi = var_gene_to_idx[pg]
    pe = control_expr[:, pi]
    if pe.std() < 1e-10:
        continue
    for j in range(n_genes):
        dg = var_genes[j]
        if dg == pg:
            continue
        de_expr = control_expr[:, j]
        if de_expr.std() < 1e-10:
            continue
        c, _ = stats.pearsonr(pe, de_expr)
        correlation_edges[(pg, dg)] = abs(float(c))

print(f"  Correlation edges: {len(correlation_edges):,}, time: {time.time()-t5:.0f}s")

# ============================================================
# STEP 8: Evaluation
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Evaluation")
print("=" * 70)
t6 = time.time()

def evaluate(pred_scores, de_res, sel_genes):
    """Evaluate predictions vs DE ground truth."""
    all_labels, all_scores = [], []
    per_gene = {}

    for pg, _ in sel_genes:
        if pg not in de_res:
            continue
        gde = de_res[pg]
        labels, scores = [], []
        for dg, info in gde.items():
            label = 1 if info['is_de'] else 0
            # For attention-based methods: use max(fwd, rev) directionality
            fwd = pred_scores.get((pg, dg), 0.0)
            rev = pred_scores.get((dg, pg), 0.0)
            score = max(fwd, rev)
            labels.append(label)
            scores.append(score)

        if len(labels) < 10 or sum(labels) < 2 or sum(labels) == len(labels):
            continue

        la, sa = np.array(labels), np.array(scores)
        try:
            auroc = roc_auc_score(la, sa)
            auprc = average_precision_score(la, sa)
            per_gene[pg] = {'auroc': float(auroc), 'auprc': float(auprc),
                            'baseline': float(la.mean()), 'n_pos': int(la.sum()), 'n_tot': int(len(la))}
            all_labels.extend(labels)
            all_scores.extend(scores)
        except Exception:
            pass

    la, sa = np.array(all_labels), np.array(all_scores)
    agg = {}
    if len(la) > 0 and len(np.unique(la)) == 2:
        aurocs = [v['auroc'] for v in per_gene.values()]
        auprcs = [v['auprc'] for v in per_gene.values()]
        agg = {
            'agg_auroc': float(roc_auc_score(la, sa)),
            'agg_auprc': float(average_precision_score(la, sa)),
            'baseline_auprc': float(la.mean()),
            'med_auroc': float(np.median(aurocs)),
            'mean_auroc': float(np.mean(aurocs)),
            'std_auroc': float(np.std(aurocs)),
            'med_auprc': float(np.median(auprcs)),
            'mean_auprc': float(np.mean(auprcs)),
            'n_genes': len(per_gene),
            'n_pairs': int(len(la)),
            'n_pos': int(la.sum()),
        }
    return agg, per_gene


# Evaluate all methods
methods = {
    'raw_l13': raw_l13,
    'cssi_l13': cssi_l13,
    'raw_best_layer': raw_best_layer,
    'cssi_best_layer': cssi_best_layer,
    'cssi_mean_layers': cssi_mean_layers,
    'correlation': correlation_edges,
}

# Random baseline
rng_eval = np.random.RandomState(42)
# Build random scores on the same key space as raw_best_layer
random_keys = set()
for pg, _ in selected_genes:
    if pg not in de_results:
        continue
    for dg in de_results[pg]:
        random_keys.add((pg, dg))
random_scores = {k: rng_eval.random() for k in random_keys}
methods['random'] = random_scores

results_all = {}
for name, scores in methods.items():
    print(f"  Evaluating {name}...")
    agg, pg = evaluate(scores, de_results, selected_genes)
    results_all[name] = {'aggregate': agg, 'per_gene': pg}
    auroc = agg.get('agg_auroc', 'N/A')
    auprc = agg.get('agg_auprc', 'N/A')
    med_auroc = agg.get('med_auroc', 'N/A')
    fmt_auroc = f"{auroc:.4f}" if isinstance(auroc, float) else auroc
    fmt_auprc = f"{auprc:.4f}" if isinstance(auprc, float) else auprc
    fmt_med = f"{med_auroc:.4f}" if isinstance(med_auroc, float) else med_auroc
    print(f"    AUROC={fmt_auroc}, AUPRC={fmt_auprc}, Median AUROC={fmt_med}")

# Also evaluate per-layer raw and CSSI
per_layer_results = {}
for layer_idx in CSSI_LAYERS:
    raw_agg, _ = evaluate(raw_mean_by_layer[layer_idx], de_results, selected_genes)
    cssi_agg, _ = evaluate(cssi_scores_by_layer[layer_idx], de_results, selected_genes)
    per_layer_results[layer_idx] = {
        'raw': raw_agg,
        'cssi': cssi_agg,
    }
    print(f"  Layer {layer_idx}: raw AUROC={raw_agg.get('agg_auroc', 0):.4f}, "
          f"CSSI AUROC={cssi_agg.get('agg_auroc', 0):.4f}")

print(f"\n  Evaluation time: {time.time()-t6:.1f}s")

# ============================================================
# STEP 9: Save results JSON
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Saving results")
print("=" * 70)

# Build comparison table
comparison = {}
for name in ['raw_l13', 'cssi_l13', 'raw_best_layer', 'cssi_best_layer', 'cssi_mean_layers', 'correlation', 'random']:
    comparison[name] = results_all[name]['aggregate']

results_data = {
    'config': {
        'cssi_layers': CSSI_LAYERS,
        'model': 'Geneformer-V2-316M',
        'cells_used': len(control_tokens),
        'n_perturbed_genes': len(selected_genes),
        'de_pvalue_threshold': DE_PVALUE_THRESHOLD,
        'de_log2fc_threshold': DE_LOG2FC_THRESHOLD,
        'cell_line': CELL_LINE,
        'batch_size': BATCH_SIZE,
        'max_seq_len': MAX_SEQ_LEN,
    },
    'comparison': comparison,
    'per_layer_results': {str(k): v for k, v in per_layer_results.items()},
    'method_results': {name: results_all[name] for name in ['raw_l13', 'cssi_l13', 'cssi_best_layer']},
    'de_summary': {
        'total_de': total_de,
        'mean_per_gene': total_de / len(selected_genes),
    },
    'selected_genes': [(g, c) for g, c in selected_genes],
    'total_time_seconds': time.time() - t0,
}

results_path = os.path.join(RESULTS_DIR, 'cssi_crispri_results.json')
with open(results_path, 'w') as f:
    json.dump(results_data, f, indent=2)
print(f"  Results: {results_path}")

# ============================================================
# STEP 10: Generate CSSI Report
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: Generating CSSI report")
print("=" * 70)

def fmt(v, d=4):
    return f"{v:.{d}f}" if isinstance(v, float) else str(v) if v is not None else 'N/A'

# Per-layer table
layer_table_rows = ""
for layer_idx in CSSI_LAYERS:
    pr = per_layer_results[layer_idx]
    layer_table_rows += (f"| {layer_idx} | {fmt(pr['raw'].get('agg_auroc'))} | "
                         f"{fmt(pr['cssi'].get('agg_auroc'))} | "
                         f"{fmt(pr['raw'].get('agg_auprc'))} | "
                         f"{fmt(pr['cssi'].get('agg_auprc'))} |\n")

# Determine improvement
raw_l13_auroc = comparison['raw_l13'].get('agg_auroc', 0.5)
cssi_l13_auroc = comparison['cssi_l13'].get('agg_auroc', 0.5)
cssi_bl_auroc = comparison['cssi_best_layer'].get('agg_auroc', 0.5)
random_auroc = comparison['random'].get('agg_auroc', 0.5)
corr_auroc = comparison['correlation'].get('agg_auroc', 0.5)
raw_bl_auroc = comparison['raw_best_layer'].get('agg_auroc', 0.5)

# Top genes for CSSI best layer
cssi_bl_pg = results_all['cssi_best_layer']['per_gene']
top_cssi = sorted(cssi_bl_pg.items(), key=lambda x: -x[1]['auroc'])[:20] if cssi_bl_pg else []

report = f"""# CSSI CRISPRi Validation Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Question:** Does Cell-type Specific Subnetwork Inference (CSSI) improve CRISPRi validation?

---

## Executive Summary

Previous result: Raw attention at Layer 13 achieved **AUROC {fmt(raw_l13_auroc)}** (below random {fmt(random_auroc)}).

**CSSI Results:**
| Method | Aggregate AUROC | vs Random |
|--------|----------------|-----------|
| Raw Attention L13 | {fmt(raw_l13_auroc)} | {fmt(raw_l13_auroc - random_auroc)} |
| **CSSI-weighted L13** | **{fmt(cssi_l13_auroc)}** | {fmt(cssi_l13_auroc - random_auroc)} |
| Raw Best Layer (L10-17) | {fmt(raw_bl_auroc)} | {fmt(raw_bl_auroc - random_auroc)} |
| **CSSI Best Layer (L10-17)** | **{fmt(cssi_bl_auroc)}** | {fmt(cssi_bl_auroc - random_auroc)} |
| Correlation | {fmt(corr_auroc)} | {fmt(corr_auroc - random_auroc)} |
| Random | {fmt(random_auroc)} | 0.0000 |

---

## Method: CSSI-Weighted Attention

### What is CSSI?
Cell-type Specific Subnetwork Inference weights attention scores by their **coefficient of variation (CV)** across cells:

```
CSSI_score(gene_A, gene_B) = mean_attention(A,B) * CV_attention(A,B)
```

Where CV = std / mean. Higher CV means the attention edge varies more across cells,
indicating **cell-type specificity** rather than uniform background attention.

### Rationale
- Raw attention averages may be dominated by constitutive (housekeeping) interactions
- CRISPRi ground truth involves cell-type-specific regulatory effects
- CSSI upweights edges that are active in some cells but not others

### Multi-layer CSSI
Instead of using only Layer 13 (which was selected on Tabula Sapiens immune data),
we test layers 10-17 and for each gene pair take the **maximum CSSI score** across layers.

---

## Experimental Design

| Parameter | Value |
|-----------|-------|
| Dataset | Replogle et al. CRISPRi (K562) |
| Control cells | {len(control_tokens)} |
| Perturbed genes | {len(selected_genes)} |
| Model | Geneformer V2-316M |
| CSSI layers | {CSSI_LAYERS} |
| DE threshold | p<{DE_PVALUE_THRESHOLD}, |log2FC|>{DE_LOG2FC_THRESHOLD} |
| Total DE pairs | {total_de:,} |
| Batch size | {BATCH_SIZE} |

---

## Results

### Full Comparison Table

| Method | Agg AUROC | Agg AUPRC | Baseline AUPRC | Median AUROC | Mean AUROC | Genes |
|--------|----------|----------|---------------|-------------|-----------|-------|
| Raw Attention L13 | {fmt(comparison['raw_l13'].get('agg_auroc'))} | {fmt(comparison['raw_l13'].get('agg_auprc'))} | {fmt(comparison['raw_l13'].get('baseline_auprc'))} | {fmt(comparison['raw_l13'].get('med_auroc'))} | {fmt(comparison['raw_l13'].get('mean_auroc'))} | {comparison['raw_l13'].get('n_genes', 'N/A')} |
| CSSI L13 | {fmt(comparison['cssi_l13'].get('agg_auroc'))} | {fmt(comparison['cssi_l13'].get('agg_auprc'))} | {fmt(comparison['cssi_l13'].get('baseline_auprc'))} | {fmt(comparison['cssi_l13'].get('med_auroc'))} | {fmt(comparison['cssi_l13'].get('mean_auroc'))} | {comparison['cssi_l13'].get('n_genes', 'N/A')} |
| Raw Best Layer | {fmt(comparison['raw_best_layer'].get('agg_auroc'))} | {fmt(comparison['raw_best_layer'].get('agg_auprc'))} | {fmt(comparison['raw_best_layer'].get('baseline_auprc'))} | {fmt(comparison['raw_best_layer'].get('med_auroc'))} | {fmt(comparison['raw_best_layer'].get('mean_auroc'))} | {comparison['raw_best_layer'].get('n_genes', 'N/A')} |
| **CSSI Best Layer** | **{fmt(comparison['cssi_best_layer'].get('agg_auroc'))}** | **{fmt(comparison['cssi_best_layer'].get('agg_auprc'))}** | {fmt(comparison['cssi_best_layer'].get('baseline_auprc'))} | {fmt(comparison['cssi_best_layer'].get('med_auroc'))} | {fmt(comparison['cssi_best_layer'].get('mean_auroc'))} | {comparison['cssi_best_layer'].get('n_genes', 'N/A')} |
| CSSI Mean Layers | {fmt(comparison['cssi_mean_layers'].get('agg_auroc'))} | {fmt(comparison['cssi_mean_layers'].get('agg_auprc'))} | {fmt(comparison['cssi_mean_layers'].get('baseline_auprc'))} | {fmt(comparison['cssi_mean_layers'].get('med_auroc'))} | {fmt(comparison['cssi_mean_layers'].get('mean_auroc'))} | {comparison['cssi_mean_layers'].get('n_genes', 'N/A')} |
| Correlation | {fmt(comparison['correlation'].get('agg_auroc'))} | {fmt(comparison['correlation'].get('agg_auprc'))} | {fmt(comparison['correlation'].get('baseline_auprc'))} | {fmt(comparison['correlation'].get('med_auroc'))} | {fmt(comparison['correlation'].get('mean_auroc'))} | {comparison['correlation'].get('n_genes', 'N/A')} |
| Random | {fmt(comparison['random'].get('agg_auroc'))} | {fmt(comparison['random'].get('agg_auprc'))} | {fmt(comparison['random'].get('baseline_auprc'))} | {fmt(comparison['random'].get('med_auroc'))} | {fmt(comparison['random'].get('mean_auroc'))} | {comparison['random'].get('n_genes', 'N/A')} |

### Per-Layer Comparison (Raw vs CSSI)

| Layer | Raw AUROC | CSSI AUROC | Raw AUPRC | CSSI AUPRC |
|-------|----------|-----------|----------|-----------|
{layer_table_rows}

"""

if top_cssi:
    report += """### Top 20 Genes by CSSI Best-Layer AUROC

| Gene | AUROC | AUPRC | DE genes | Total |
|------|-------|-------|----------|-------|
"""
    for gene, m in top_cssi:
        report += f"| {gene} | {m['auroc']:.4f} | {m['auprc']:.4f} | {m['n_pos']} | {m['n_tot']} |\n"

report += f"""

---

## Interpretation

### Does CSSI help?

"""

# Dynamic interpretation based on results
if cssi_bl_auroc > random_auroc + 0.01:
    report += f"""**YES** - CSSI Best Layer achieves AUROC {fmt(cssi_bl_auroc)}, which is {fmt(cssi_bl_auroc - random_auroc)} above random ({fmt(random_auroc)}).
The CV-weighting successfully upweights cell-type-specific regulatory edges.
"""
elif cssi_bl_auroc > raw_l13_auroc + 0.005:
    report += f"""**PARTIALLY** - CSSI improves over raw Layer 13 attention (AUROC {fmt(cssi_bl_auroc)} vs {fmt(raw_l13_auroc)}),
but the improvement is modest. The fundamental limitation may be that Geneformer's attention
patterns in control cells do not strongly encode CRISPRi-detectable causal edges.
"""
else:
    report += f"""**NO** - CSSI does not meaningfully improve CRISPRi prediction.
- Raw L13: {fmt(raw_l13_auroc)}
- CSSI L13: {fmt(cssi_l13_auroc)}
- CSSI Best Layer: {fmt(cssi_bl_auroc)}
- Random: {fmt(random_auroc)}

This suggests the limitation is not in the aggregation method but in the attention
patterns themselves. Geneformer attention at these layers does not capture the specific
type of regulatory information that CRISPRi perturbation reveals.
"""

report += f"""
### Key Observations

1. **Layer comparison**: The per-layer results show whether deeper/shallower layers
   capture different aspects of gene regulation
2. **CV weighting**: Upweighting variable edges tests whether cell-type-specific attention
   is more informative than constitutive attention
3. **Multi-layer selection**: Allowing different layers per gene pair tests whether
   regulatory information is distributed across layers

### Limitations
- CSSI assumes CV correlates with regulatory importance (may not hold for all interactions)
- K562 is a single cell line; CV across control cells may reflect noise not biology
- Head-averaged attention may dilute head-specific regulatory signals
- 500 control cells may be insufficient for robust CV estimation

---

**Total pipeline time:** {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)

*Generated by CSSI CRISPRi Validation Pipeline*
"""

report_path = os.path.join(RESULTS_DIR, 'CSSI_CRISPRI_REPORT.md')
with open(report_path, 'w') as f:
    f.write(report)
print(f"  Report: {report_path}")

total_time = time.time() - t0
print(f"\n{'=' * 70}")
print(f"CSSI PIPELINE COMPLETE - Total: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"{'=' * 70}")
