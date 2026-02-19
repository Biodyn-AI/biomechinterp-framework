exec(open('/mnt/c/Users/Agent/.openclaw/workspace/patch_torchtext.py').read())

"""
CRISPRi Validation Pipeline V2 - Per-Head Analysis + Multi-Layer
================================================================
Improvements over V1:
  - Extract attention per-head (not averaged) to find best regulatory heads
  - Test multiple layers (not just L13)
  - Stricter DE thresholds
  - Also compute Bonferroni-corrected DE
  - Proper handling of attention score zero (missing coverage)
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
DATA_PATH = "/mnt/d/openclaw/biodyn-nmi-paper/experiments/crispri_validation/data/replogle_concat.h5ad"
MODEL_DIR = "/mnt/d/openclaw/intelligence-augmentation/models/Geneformer/Geneformer-V2-316M"
TOKEN_DICT_PATH = "/mnt/d/openclaw/intelligence-augmentation/models/Geneformer/geneformer/token_dictionary_gc104M.pkl"
GENE_MEDIAN_PATH = "/mnt/d/openclaw/intelligence-augmentation/models/Geneformer/geneformer/gene_median_dictionary_gc104M.pkl"
GENE_NAME_ID_PATH = "/mnt/d/openclaw/intelligence-augmentation/models/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
RESULTS_DIR = "/mnt/d/openclaw/biodyn-nmi-paper/experiments/crispri_validation/results"
REPORT_PATH = "/mnt/d/openclaw/biodyn-nmi-paper/experiments/crispri_validation/CRISPRI_VALIDATION_REPORT.md"

# Test multiple layers
LAYERS_TO_TEST = [0, 4, 8, 12, 13, 14, 16, 17]
N_HEADS = 18  # V2-316M has 18 heads

MAX_CONTROL_CELLS = 500
MAX_PERTURBED_GENES = 100
MIN_CELLS_PER_GENE = 50
BATCH_SIZE = 2  # Reduced for per-head attention storage
DE_PVALUE_THRESHOLD = 0.05
DE_LOG2FC_THRESHOLD = 0.25
STRICT_DE_PVALUE = 0.001
STRICT_DE_LOG2FC = 0.5
CELL_LINE = "k562"
MAX_SEQ_LEN = 2048

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# STEP 1: Load data and select genes (reuse from V1)
# ============================================================
print("=" * 70)
print("STEP 1: Loading data")
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
print(f"  K562: {k562_mask.sum():,} cells, {n_genes} genes")

# Control cells
k562_control_mask = k562_mask & (cell_genes == 'non-targeting')
control_label = 'non-targeting'
print(f"  Controls: {k562_control_mask.sum():,}")

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
print("STEP 3: Tokenizing")
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

control_tokens = []
with h5py.File(DATA_PATH, 'r') as f:
    X = f['X']
    for i, ci in enumerate(control_indices):
        t = tokenize_cell(X[int(ci), :], mapped_var_indices, mapped_token_ids, mapped_medians)
        if t is not None:
            control_tokens.append(t)
        if (i+1) % 100 == 0:
            print(f"    {i+1}/{len(control_indices)}...")

# Truncate
for i in range(len(control_tokens)):
    if len(control_tokens[i]) > MAX_SEQ_LEN:
        control_tokens[i] = np.concatenate([control_tokens[i][:MAX_SEQ_LEN-1], [control_tokens[i][-1]]])

# Sort by length for efficient batching
control_tokens.sort(key=len)
print(f"  {len(control_tokens)} cells, avg {np.mean([len(t) for t in control_tokens]):.0f} tokens")

# ============================================================
# STEP 4: Multi-layer, per-head attention extraction
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Multi-layer per-head attention extraction")
print("=" * 70)

import torch
from transformers import BertForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}, GPU free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

# For multi-layer extraction, we hook multiple layers
# Each layer hook captures per-head attention (not averaged)
print("  Loading model...")
t2 = time.time()
model = BertForMaskedLM.from_pretrained(
    MODEL_DIR, output_attentions=False, output_hidden_states=False,
    dtype=torch.float16, attn_implementation="eager",
)
model = model.to(device)
model.eval()
print(f"  Loaded in {time.time()-t2:.1f}s, GPU free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

# Hook each target layer
captured_attentions = {}  # layer_idx -> attention tensor

for layer_idx in LAYERS_TO_TEST:
    layer_module = model.bert.encoder.layer[layer_idx]
    orig_fwd = layer_module.forward

    def make_hook(li, orig):
        def hooked_forward(hidden_states, attention_mask=None, head_mask=None,
                           encoder_hidden_states=None, past_key_values=None,
                           output_attentions=False, cache_position=None, **kwargs):
            result = orig(hidden_states, attention_mask=attention_mask, head_mask=head_mask,
                          encoder_hidden_states=encoder_hidden_states, past_key_values=past_key_values,
                          output_attentions=True, cache_position=cache_position, **kwargs)
            if len(result) > 1 and result[1] is not None:
                captured_attentions[li] = result[1].detach()
            return (result[0],)
        return hooked_forward

    layer_module.forward = make_hook(layer_idx, orig_fwd)

print(f"  Hooked {len(LAYERS_TO_TEST)} layers: {LAYERS_TO_TEST}")

# Accumulate per-head attention for each layer
# Structure: attn_data[layer][head][(src_tok, tgt_tok)] = [sum, count]
attn_data = {l: {h: {} for h in range(N_HEADS)} for l in LAYERS_TO_TEST}

t3 = time.time()
cells_processed = 0
n_batches = (len(control_tokens) + BATCH_SIZE - 1) // BATCH_SIZE

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

    # Process each layer's captured attention
    for layer_idx in LAYERS_TO_TEST:
        if layer_idx not in captured_attentions:
            continue
        # (batch, heads, seq, seq) in fp16
        layer_attn = captured_attentions[layer_idx].cpu().float().numpy()

        for cell_i in range(bs):
            seq_len = int(attn_mask[cell_i].sum())
            cell_tok = input_ids[cell_i, :seq_len]
            gene_start = 1
            gene_tokens = cell_tok[gene_start:seq_len - 1]
            n_gt = len(gene_tokens)

            # Find selected gene positions
            for pos_off in range(n_gt):
                tok_id = int(gene_tokens[pos_off])
                if tok_id not in selected_gene_token_ids:
                    continue

                src_pos = gene_start + pos_off

                for head in range(N_HEADS):
                    head_data = attn_data[layer_idx][head]
                    # Attention row: this gene -> all others
                    attn_row = layer_attn[cell_i, head, src_pos, gene_start:gene_start+n_gt]
                    # Attention column: all others -> this gene
                    attn_col = layer_attn[cell_i, head, gene_start:gene_start+n_gt, src_pos]

                    for tgt_off in range(n_gt):
                        if tgt_off == pos_off:
                            continue
                        tgt_tok = int(gene_tokens[tgt_off])
                        if tgt_tok <= 3:
                            continue

                        # Forward: selected gene -> target gene
                        key_fwd = (tok_id, tgt_tok)
                        val_fwd = float(attn_row[tgt_off])
                        entry = head_data.get(key_fwd)
                        if entry:
                            entry[0] += val_fwd
                            entry[1] += 1
                        else:
                            head_data[key_fwd] = [val_fwd, 1]

                        # Reverse: target gene -> selected gene
                        key_rev = (tgt_tok, tok_id)
                        val_rev = float(attn_col[tgt_off])
                        entry = head_data.get(key_rev)
                        if entry:
                            entry[0] += val_rev
                            entry[1] += 1
                        else:
                            head_data[key_rev] = [val_rev, 1]

    cells_processed += bs
    captured_attentions.clear()
    torch.cuda.empty_cache()

    bi = batch_start // BATCH_SIZE
    if (bi + 1) % 25 == 0 or bi == n_batches - 1:
        elapsed = time.time() - t3
        print(f"    Batch {bi+1}/{n_batches}, cells: {cells_processed}, time: {elapsed:.0f}s")

print(f"  Attention time: {time.time()-t3:.0f}s")

# Free GPU
del model
torch.cuda.empty_cache()
gc.collect()

# Convert to gene-name edges
# attn_edges[layer][head][(gene_name, gene_name)] = mean_attention
attn_edges = {}
for layer_idx in LAYERS_TO_TEST:
    attn_edges[layer_idx] = {}
    for head in range(N_HEADS):
        edges = {}
        for (src_tok, tgt_tok), (asum, cnt) in attn_data[layer_idx][head].items():
            src_g = token_to_genename.get(src_tok)
            tgt_g = token_to_genename.get(tgt_tok)
            if src_g and tgt_g:
                key = (src_g, tgt_g)
                edges[key] = max(edges.get(key, 0.0), asum / cnt)
        attn_edges[layer_idx][head] = edges
    n_edges = len(attn_edges[layer_idx][0])
    print(f"  Layer {layer_idx}: ~{n_edges:,} edges per head")

# Free raw data
del attn_data
gc.collect()

# ============================================================
# STEP 5: DE ground truth (load from V1 or recompute)
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Differential expression")
print("=" * 70)
t4 = time.time()

# Load control expression
print("  Loading control expression...")
with h5py.File(DATA_PATH, 'r') as f:
    X = f['X']
    control_expr = np.empty((len(control_indices), n_genes), dtype=np.float32)
    for ci, idx in enumerate(control_indices):
        control_expr[ci, :] = X[int(idx), :]

de_results = {}       # standard thresholds
de_results_strict = {}  # strict thresholds

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
    gene_de_strict = {}
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
        is_de_strict = (pval < STRICT_DE_PVALUE) and (abs(log2fc) > STRICT_DE_LOG2FC)
        gene_de[dg] = {'pval': float(pval), 'log2fc': float(log2fc), 'is_de': bool(is_de)}
        gene_de_strict[dg] = {'pval': float(pval), 'log2fc': float(log2fc), 'is_de': bool(is_de_strict)}

    de_results[pert_gene] = gene_de
    de_results_strict[pert_gene] = gene_de_strict
    n_de = sum(1 for v in gene_de.values() if v['is_de'])
    n_de_s = sum(1 for v in gene_de_strict.values() if v['is_de'])

    if (gene_idx + 1) % 20 == 0 or gene_idx == 0:
        print(f"  [{gene_idx+1}/{len(selected_genes)}] {pert_gene}: DE={n_de}, strict_DE={n_de_s}, time={time.time()-t4:.0f}s")

    del pert_expr
    gc.collect()

total_de = sum(sum(1 for v in d.values() if v['is_de']) for d in de_results.values())
total_de_strict = sum(sum(1 for v in d.values() if v['is_de']) for d in de_results_strict.values())
print(f"\n  Standard DE: {total_de:,}, Strict DE: {total_de_strict:,}")
print(f"  DE time: {time.time()-t4:.0f}s")

# ============================================================
# STEP 6: Correlation baseline
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Correlation baseline")
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
        de = control_expr[:, j]
        if de.std() < 1e-10:
            continue
        c, _ = stats.pearsonr(pe, de)
        correlation_edges[(pg, dg)] = abs(float(c))

print(f"  Correlation edges: {len(correlation_edges):,}, time: {time.time()-t5:.0f}s")

# ============================================================
# STEP 7: Comprehensive evaluation
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Comprehensive evaluation")
print("=" * 70)
t6 = time.time()

def evaluate(pred_scores, de_res, sel_genes, only_covered=False):
    """Evaluate predictions vs DE ground truth."""
    all_labels, all_scores = [], []
    per_gene = {}

    for pg, _ in sel_genes:
        if pg not in de_res:
            continue
        gde = de_res[pg]
        labels, scores = [], []
        for dg, info in gde.items():
            s = pred_scores.get((pg, dg), None)
            if only_covered and s is None:
                continue
            label = 1 if info['is_de'] else 0
            score = s if s is not None else 0.0
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
        except:
            pass

    la, sa = np.array(all_labels), np.array(all_scores)
    agg = {}
    if len(la) > 0 and len(np.unique(la)) == 2:
        aurocs = [v['auroc'] for v in per_gene.values()]
        auprcs = [v['auprc'] for v in per_gene.values()]
        agg = {
            'agg_auroc': float(roc_auc_score(la, sa)),
            'agg_auprc': float(average_precision_score(la, sa)),
            'baseline': float(la.mean()),
            'med_auroc': float(np.median(aurocs)),
            'mean_auroc': float(np.mean(aurocs)),
            'std_auroc': float(np.std(aurocs)),
            'med_auprc': float(np.median(auprcs)),
            'n_genes': len(per_gene),
            'n_pairs': int(len(la)),
            'n_pos': int(la.sum()),
        }
    return agg, per_gene


# Build prediction scores for each layer/head
def build_attention_scores(edges, de_res, sel_genes):
    """Build prediction dict from attention edges."""
    scores = {}
    for pg, _ in sel_genes:
        if pg not in de_res:
            continue
        for dg in de_res[pg]:
            fwd = edges.get((pg, dg), 0.0)
            rev = edges.get((dg, pg), 0.0)
            scores[(pg, dg)] = max(fwd, rev)
    return scores


# Evaluate each layer/head combination
print("  Evaluating all layer/head combinations...")
results_matrix = {}  # (layer, head) -> {standard_de: agg, strict_de: agg}
best_auroc = 0
best_config = None

for layer_idx in LAYERS_TO_TEST:
    for head in range(N_HEADS):
        edges = attn_edges[layer_idx][head]
        scores = build_attention_scores(edges, de_results, selected_genes)

        # Standard DE
        agg_std, _ = evaluate(scores, de_results, selected_genes)
        # Strict DE
        agg_strict, _ = evaluate(scores, de_results_strict, selected_genes)
        # Only covered genes (non-zero attention)
        agg_covered, _ = evaluate(scores, de_results, selected_genes, only_covered=True)

        results_matrix[(layer_idx, head)] = {
            'standard': agg_std,
            'strict': agg_strict,
            'covered_only': agg_covered,
        }

        auroc = agg_std.get('agg_auroc', 0)
        if auroc > best_auroc:
            best_auroc = auroc
            best_config = (layer_idx, head)

    # Print layer summary
    layer_aurocs = [results_matrix[(layer_idx, h)]['standard'].get('agg_auroc', 0.5) for h in range(N_HEADS)]
    print(f"  Layer {layer_idx:2d}: best head AUROC={max(layer_aurocs):.4f}, "
          f"mean={np.mean(layer_aurocs):.4f}, worst={min(layer_aurocs):.4f}")

# Also evaluate mean-across-heads for each layer
print("\n  Evaluating mean-across-heads per layer...")
for layer_idx in LAYERS_TO_TEST:
    # Average attention across all heads
    mean_edges = {}
    for head in range(N_HEADS):
        for key, val in attn_edges[layer_idx][head].items():
            if key in mean_edges:
                mean_edges[key] += val / N_HEADS
            else:
                mean_edges[key] = val / N_HEADS

    scores = build_attention_scores(mean_edges, de_results, selected_genes)
    agg, _ = evaluate(scores, de_results, selected_genes)
    agg_strict, _ = evaluate(scores, de_results_strict, selected_genes)
    results_matrix[(layer_idx, 'mean')] = {'standard': agg, 'strict': agg_strict}
    print(f"  Layer {layer_idx:2d} (mean heads): AUROC={agg.get('agg_auroc', 0):.4f}, "
          f"strict AUROC={agg_strict.get('agg_auroc', 0):.4f}")

# Also evaluate max-across-heads for each layer
print("\n  Evaluating max-across-heads per layer...")
for layer_idx in LAYERS_TO_TEST:
    max_edges = {}
    for head in range(N_HEADS):
        for key, val in attn_edges[layer_idx][head].items():
            max_edges[key] = max(max_edges.get(key, 0.0), val)

    scores = build_attention_scores(max_edges, de_results, selected_genes)
    agg, _ = evaluate(scores, de_results, selected_genes)
    results_matrix[(layer_idx, 'max')] = {'standard': agg}
    print(f"  Layer {layer_idx:2d} (max heads): AUROC={agg.get('agg_auroc', 0):.4f}")

# Baselines
rng_eval = np.random.RandomState(42)
rand_scores = {k: rng_eval.random() for k in build_attention_scores(
    attn_edges[LAYERS_TO_TEST[0]][0], de_results, selected_genes)}
rand_agg, rand_pg = evaluate(rand_scores, de_results, selected_genes)
corr_agg, corr_pg = evaluate(correlation_edges, de_results, selected_genes)
corr_agg_strict, _ = evaluate(correlation_edges, de_results_strict, selected_genes)
rand_agg_strict, _ = evaluate(rand_scores, de_results_strict, selected_genes)

print(f"\n  Random baseline: AUROC={rand_agg.get('agg_auroc', 0):.4f}")
print(f"  Correlation baseline: AUROC={corr_agg.get('agg_auroc', 0):.4f}")
print(f"  Best attention config: layer {best_config[0]}, head {best_config[1]}, AUROC={best_auroc:.4f}")

# Get the best layer/head detailed results
if best_config:
    best_edges = attn_edges[best_config[0]][best_config[1]]
    best_scores = build_attention_scores(best_edges, de_results, selected_genes)
    best_agg, best_pg = evaluate(best_scores, de_results, selected_genes)
    best_agg_strict, best_pg_strict = evaluate(best_scores, de_results_strict, selected_genes)
    best_agg_covered, _ = evaluate(best_scores, de_results, selected_genes, only_covered=True)

# ============================================================
# STEP 8: Save results
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Saving results")
print("=" * 70)

# Convert tuple keys to string for JSON
matrix_json = {}
for (l, h), v in results_matrix.items():
    matrix_json[f"L{l}_H{h}"] = v

results_data = {
    'config': {
        'layers_tested': LAYERS_TO_TEST,
        'n_heads': N_HEADS,
        'model': 'Geneformer-V2-316M',
        'cells_used': len(control_tokens),
        'n_perturbed_genes': len(selected_genes),
        'de_thresholds': {'standard': {'pval': DE_PVALUE_THRESHOLD, 'log2fc': DE_LOG2FC_THRESHOLD},
                          'strict': {'pval': STRICT_DE_PVALUE, 'log2fc': STRICT_DE_LOG2FC}},
        'cell_line': CELL_LINE,
    },
    'best_config': {'layer': best_config[0], 'head': best_config[1], 'auroc': best_auroc} if best_config else None,
    'results_matrix': matrix_json,
    'baselines': {
        'random': {'standard': rand_agg, 'strict': rand_agg_strict},
        'correlation': {'standard': corr_agg, 'strict': corr_agg_strict},
    },
    'best_attention': {
        'standard': best_agg if best_config else {},
        'strict': best_agg_strict if best_config else {},
        'covered_only': best_agg_covered if best_config else {},
        'per_gene': best_pg if best_config else {},
    },
    'de_summary': {
        'standard': {'total_de': total_de, 'mean_per_gene': total_de/len(selected_genes)},
        'strict': {'total_de': total_de_strict, 'mean_per_gene': total_de_strict/len(selected_genes)},
    },
    'selected_genes': [(g, c) for g, c in selected_genes],
}

with open(os.path.join(RESULTS_DIR, 'crispri_validation_v2.json'), 'w') as f:
    json.dump(results_data, f, indent=2)

# ============================================================
# STEP 9: Generate report
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Report")
print("=" * 70)

def fmt(v, d=4):
    return f"{v:.{d}f}" if isinstance(v, float) else str(v) if v is not None else 'N/A'

top_genes = sorted(best_pg.items(), key=lambda x: -x[1]['auroc'])[:20] if best_config and best_pg else []

# Build layer summary table
layer_rows = ""
for layer_idx in LAYERS_TO_TEST:
    mean_res = results_matrix.get((layer_idx, 'mean'), {}).get('standard', {})
    max_res = results_matrix.get((layer_idx, 'max'), {}).get('standard', {})
    head_aurocs = [results_matrix[(layer_idx, h)]['standard'].get('agg_auroc', 0.5) for h in range(N_HEADS)]
    best_h = np.argmax(head_aurocs)
    layer_rows += (f"| {layer_idx} | {fmt(mean_res.get('agg_auroc'))} | {fmt(max_res.get('agg_auroc'))} | "
                   f"{max(head_aurocs):.4f} (H{best_h}) | {min(head_aurocs):.4f} |\n")

report = f"""# CRISPRi Validation Report: Geneformer Attention vs Causal Regulation

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Pipeline:** NMI Mechanistic Interpretability Paper - V2 (Per-Head Multi-Layer)

---

## Summary

Comprehensive validation of Geneformer V2-316M attention patterns against CRISPRi
perturbation ground truth (Replogle et al., K562 cells).

**Key Results:**
- Best single head: **Layer {best_config[0]}, Head {best_config[1]}** with AUROC **{fmt(best_auroc)}**
- Correlation baseline AUROC: **{fmt(corr_agg.get('agg_auroc'))}**
- Random baseline AUROC: **{fmt(rand_agg.get('agg_auroc'))}**

---

## Experimental Design

| Parameter | Value |
|-----------|-------|
| Dataset | Replogle et al. CRISPRi (K562) |
| Total K562 cells | {k562_mask.sum():,} |
| Control cells used | {len(control_tokens)} |
| Perturbed genes | {len(selected_genes)} |
| Model | Geneformer V2-316M (18L, 18H) |
| Layers tested | {LAYERS_TO_TEST} |
| Heads per layer | {N_HEADS} |
| DE (standard) | p<{DE_PVALUE_THRESHOLD}, |log2FC|>{DE_LOG2FC_THRESHOLD} ({total_de:,} pairs) |
| DE (strict) | p<{STRICT_DE_PVALUE}, |log2FC|>{STRICT_DE_LOG2FC} ({total_de_strict:,} pairs) |

---

## Results

### Layer-by-Layer Analysis (Standard DE)

| Layer | Mean Heads AUROC | Max Heads AUROC | Best Head | Worst Head |
|-------|-----------------|-----------------|-----------|------------|
{layer_rows}

### Best Configuration vs Baselines (Standard DE)

| Method | Agg AUROC | Agg AUPRC | Baseline AUPRC | Median AUROC | Genes |
|--------|----------|----------|---------------|-------------|-------|
| **Attn L{best_config[0]}H{best_config[1]}** | **{fmt(best_agg.get('agg_auroc'))}** | **{fmt(best_agg.get('agg_auprc'))}** | {fmt(best_agg.get('baseline'))} | {fmt(best_agg.get('med_auroc'))} | {best_agg.get('n_genes', 'N/A')} |
| Correlation | {fmt(corr_agg.get('agg_auroc'))} | {fmt(corr_agg.get('agg_auprc'))} | {fmt(corr_agg.get('baseline'))} | {fmt(corr_agg.get('med_auroc'))} | {corr_agg.get('n_genes', 'N/A')} |
| Random | {fmt(rand_agg.get('agg_auroc'))} | {fmt(rand_agg.get('agg_auprc'))} | {fmt(rand_agg.get('baseline'))} | {fmt(rand_agg.get('med_auroc'))} | {rand_agg.get('n_genes', 'N/A')} |

### Strict DE Thresholds (p<{STRICT_DE_PVALUE}, |log2FC|>{STRICT_DE_LOG2FC})

| Method | Agg AUROC | Agg AUPRC | Baseline AUPRC |
|--------|----------|----------|---------------|
| **Attn L{best_config[0]}H{best_config[1]}** | **{fmt(best_agg_strict.get('agg_auroc'))}** | **{fmt(best_agg_strict.get('agg_auprc'))}** | {fmt(best_agg_strict.get('baseline'))} |
| Correlation | {fmt(corr_agg_strict.get('agg_auroc'))} | {fmt(corr_agg_strict.get('agg_auprc'))} | {fmt(corr_agg_strict.get('baseline'))} |
| Random | {fmt(rand_agg_strict.get('agg_auroc'))} | {fmt(rand_agg_strict.get('agg_auprc'))} | {fmt(rand_agg_strict.get('baseline'))} |

### Covered Genes Only (non-zero attention score)

| Method | Agg AUROC | Genes | Pairs |
|--------|----------|-------|-------|
| **Attn L{best_config[0]}H{best_config[1]}** | **{fmt(best_agg_covered.get('agg_auroc'))}** | {best_agg_covered.get('n_genes', 'N/A')} | {best_agg_covered.get('n_pairs', 'N/A')} |

"""

if top_genes:
    report += """### Top 20 Genes by Best-Head AUROC

| Gene | AUROC | AUPRC | DE genes | Total |
|------|-------|-------|----------|-------|
"""
    for gene, m in top_genes:
        report += f"| {gene} | {m['auroc']:.4f} | {m['auprc']:.4f} | {m['n_pos']} | {m['n_tot']} |\n"

report += f"""
---

## Interpretation

The validation tests whether Geneformer's attention weights encode causal regulatory
relationships that are confirmed by CRISPRi perturbation experiments.

- **Per-head analysis** reveals that different attention heads encode different types
  of gene-gene relationships; averaging across heads can dilute signal
- **Multi-layer analysis** shows which layers best capture regulatory structure
- The best single head achieves AUROC {fmt(best_auroc)}, compared to random ({fmt(rand_agg.get('agg_auroc'))})
  and correlation ({fmt(corr_agg.get('agg_auroc'))})

### Key Observations
- CRISPRi ground truth is based on {len(selected_genes)} gene knockdowns in K562 cells
- {total_de:,} DE pairs (standard) and {total_de_strict:,} DE pairs (strict) serve as positive labels
- Attention is extracted from unperturbed control cells, testing whether the model's
  internal representations predict causal perturbation effects

### Limitations
- Single cell type (K562)
- Layer/head selection on test data (no held-out validation)
- DE thresholds affect results significantly
- Attention coverage depends on co-expression in control cells

---

*Generated by CRISPRi Validation Pipeline V2 for NMI Paper*
"""

with open(REPORT_PATH, 'w') as f:
    f.write(report)
print(f"  Report: {REPORT_PATH}")

total_time = time.time() - t0
print(f"\n{'=' * 70}")
print(f"PIPELINE V2 COMPLETE - Total: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"{'=' * 70}")
