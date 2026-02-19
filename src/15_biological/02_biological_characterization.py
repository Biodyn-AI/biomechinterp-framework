#!/usr/bin/env python3
"""
Analysis B: Biological Characterization of Attention Patterns
==============================================================
Tests attention edges against multiple reference databases to positively
characterize what biological relationships attention patterns encode.

The paper says "attention captures co-expression, not regulation." But
co-expression is vague. If attention preferentially predicts protein-protein
interactions, pathway co-membership, or functional module co-annotation,
this is a specific, positive biological characterization.

Reference databases:
  - TRRUST: transcriptional regulation (already used)
  - STRING: protein-protein interactions
  - Reactome: pathway co-membership
  - KEGG: pathway co-membership
  - GO Biological Process: functional co-annotation

Output:
  - reference_edge_sets/*.json      -- processed reference databases
  - attention_characterization_results.json  -- all AUROC values, stats
  - attention_characterization_summary.md    -- human-readable summary
"""

import os
import sys
import gc
import io
import json
import gzip
import time
import pickle
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import requests

# =============================================================================
# PATHS
# =============================================================================
PAPER_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper"
EXTRACTION_DIR = os.path.join(PAPER_DIR, "experiments", "round9_unified_extraction")
REF_DIR = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks"
OUT_DIR = os.path.join(PAPER_DIR, "experiments", "biological_impact")
REF_EDGE_DIR = os.path.join(OUT_DIR, "reference_edge_sets")

N_LAYERS = 18
# Enrichr API base
ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"
# STRING download base (v12.0)
STRING_BASE = "https://stringdb-downloads.org/download"


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def download_with_retry(url, max_retries=3, timeout=300):
    """Download URL content with retries."""
    for attempt in range(max_retries):
        try:
            print(f"    Downloading (attempt {attempt+1})...")
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            return resp
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return None


def parse_enrichr_gmt(text, hvg_upper_set, min_genes=5, max_genes=500):
    """Parse Enrichr GMT-like text format into gene sets.

    Enrichr format: pathway_name\t\tgene1\tgene2\t...
    Or: pathway_name\tdescription\tgene1\tgene2\t...
    """
    gene_sets = {}
    for line in text.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        name = parts[0]
        # Genes start after the first two fields
        genes = [g.strip().upper() for g in parts[2:] if g.strip()]
        # Filter to HVG universe
        genes_in_hvg = [g for g in genes if g in hvg_upper_set]
        if min_genes <= len(genes_in_hvg) <= max_genes:
            gene_sets[name] = genes_in_hvg
    return gene_sets


def gene_sets_to_pairs(gene_sets, hvg_upper_set):
    """Convert gene sets to pairwise co-membership edges."""
    pairs = set()
    for name, genes in gene_sets.items():
        genes_filt = [g for g in genes if g in hvg_upper_set]
        for i in range(len(genes_filt)):
            for j in range(i + 1, len(genes_filt)):
                g1, g2 = genes_filt[i], genes_filt[j]
                pairs.add((min(g1, g2), max(g1, g2)))
    return pairs


def compute_auroc_for_reference(edge_matrix, ref_pairs, hvg_genes, hvg_to_local,
                                 mat_idx_to_gene, token_to_mat_idx, hvg_to_token,
                                 n_hvg_tok, symmetric=True, sample_negatives=50000):
    """Compute AUROC of edge scores against a reference set.

    Uses all positive pairs and a random sample of negative pairs for efficiency.
    """
    hvg_upper_set = set(g.upper() for g in hvg_genes)
    # Identify gene pairs in the reference that are in our HVG universe
    pos_pairs = []
    for g1, g2 in ref_pairs:
        if g1 not in hvg_upper_set or g2 not in hvg_upper_set:
            continue
        g1_local = hvg_to_local.get(g1)
        g2_local = hvg_to_local.get(g2)
        if g1_local is None or g2_local is None:
            continue
        g1_orig = hvg_genes[g1_local]
        g2_orig = hvg_genes[g2_local]
        g1_tok = hvg_to_token.get(g1_orig)
        g2_tok = hvg_to_token.get(g2_orig)
        if g1_tok is None or g2_tok is None:
            continue
        g1_mat = token_to_mat_idx.get(g1_tok)
        g2_mat = token_to_mat_idx.get(g2_tok)
        if g1_mat is not None and g2_mat is not None and g1_mat != g2_mat:
            pos_pairs.append((g1_mat, g2_mat))

    if len(pos_pairs) < 10:
        return None, len(pos_pairs)

    # Get scores for positive pairs
    pos_scores = []
    for g1_mat, g2_mat in pos_pairs:
        if symmetric:
            score = edge_matrix[g1_mat, g2_mat]
        else:
            score = max(edge_matrix[g1_mat, g2_mat], edge_matrix[g2_mat, g1_mat])
        pos_scores.append(float(score))

    # Sample negative pairs (not in reference)
    pos_set = set(pos_pairs) | set((b, a) for a, b in pos_pairs)
    np.random.seed(42)
    neg_scores = []
    attempts = 0
    max_attempts = sample_negatives * 10
    while len(neg_scores) < sample_negatives and attempts < max_attempts:
        i = np.random.randint(n_hvg_tok)
        j = np.random.randint(n_hvg_tok)
        if i == j:
            attempts += 1
            continue
        if (i, j) in pos_set:
            attempts += 1
            continue
        if symmetric:
            score = edge_matrix[i, j]
        else:
            score = max(edge_matrix[i, j], edge_matrix[j, i])
        neg_scores.append(float(score))
        pos_set.add((i, j))
        pos_set.add((j, i))
        attempts += 1

    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    scores = np.array(pos_scores + neg_scores)

    if labels.sum() > 0 and labels.sum() < len(labels):
        try:
            auroc = roc_auc_score(labels, scores)
            return auroc, len(pos_pairs)
        except ValueError:
            return None, len(pos_pairs)
    return None, len(pos_pairs)


def run():
    total_t0 = time.time()

    print("=" * 70)
    print("ANALYSIS B: BIOLOGICAL CHARACTERIZATION OF ATTENTION PATTERNS")
    print("=" * 70)

    os.makedirs(REF_EDGE_DIR, exist_ok=True)

    # =================================================================
    # STEP 1: Load existing data
    # =================================================================
    print("\nSTEP 1: Loading existing data...")
    t0 = time.time()

    with open(os.path.join(EXTRACTION_DIR, "hvg_gene_list.json")) as f:
        hvg_genes = json.load(f)
    hvg_upper = [g.upper() for g in hvg_genes]
    hvg_upper_set = set(hvg_upper)
    hvg_to_local = {g.upper(): i for i, g in enumerate(hvg_genes)}

    corr_matrix = np.load(os.path.join(EXTRACTION_DIR, "correlation_matrix.npy"))

    # Load tokenization for matrix indexing
    import pickle as pkl
    TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "experiments", "crispri_validation", "data")
    with open(os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        token_dict = pkl.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_name_id_dict_gc104M.pkl"), 'rb') as f:
        gene_name_id_dict = pkl.load(f)

    hvg_to_token = {}
    for g in hvg_genes:
        ens = gene_name_id_dict.get(g)
        if ens and ens in token_dict:
            hvg_to_token[g] = token_dict[ens]
    token_to_hvg = {v: k for k, v in hvg_to_token.items()}
    hvg_token_ids = sorted(token_to_hvg.keys())
    n_hvg_tok = len(hvg_token_ids)
    token_to_mat_idx = {tid: i for i, tid in enumerate(hvg_token_ids)}
    mat_idx_to_gene = {i: token_to_hvg[tid] for i, tid in enumerate(hvg_token_ids)}

    print(f"  HVG genes: {len(hvg_genes)}, with tokens: {n_hvg_tok}")
    print(f"  Step 1 time: {time.time()-t0:.1f}s")

    # =================================================================
    # STEP 2: Load raw attention edge matrices
    # =================================================================
    print("\nSTEP 2: Loading attention edge matrices...")
    t1 = time.time()

    attn_matrices = {}
    for li in range(N_LAYERS):
        edge_path = os.path.join(EXTRACTION_DIR, "layer_mean_edges", f"layer_{li:02d}_edges.pkl")
        with open(edge_path, 'rb') as f:
            edges_dict = pickle.load(f)

        mat = np.zeros((n_hvg_tok, n_hvg_tok), dtype=np.float32)
        for (src, tgt), val in edges_dict.items():
            src_tok = hvg_to_token.get(src)
            tgt_tok = hvg_to_token.get(tgt)
            if src_tok is not None and tgt_tok is not None:
                src_mi = token_to_mat_idx.get(src_tok)
                tgt_mi = token_to_mat_idx.get(tgt_tok)
                if src_mi is not None and tgt_mi is not None:
                    mat[src_mi, tgt_mi] = val
        attn_matrices[li] = mat
        del edges_dict
        gc.collect()

    # Build correlation matrix in token-indexed space
    abs_corr = np.abs(corr_matrix)
    corr_mat_idx = np.zeros((n_hvg_tok, n_hvg_tok), dtype=np.float32)
    for mi in range(n_hvg_tok):
        gene_i = mat_idx_to_gene[mi]
        li_i = hvg_to_local.get(gene_i.upper())
        if li_i is None:
            continue
        for mj in range(n_hvg_tok):
            gene_j = mat_idx_to_gene[mj]
            li_j = hvg_to_local.get(gene_j.upper())
            if li_j is None:
                continue
            corr_mat_idx[mi, mj] = abs_corr[li_i, li_j]

    print(f"  Loaded {len(attn_matrices)} attention matrices, correlation matrix")
    print(f"  Step 2 time: {time.time()-t1:.1f}s")

    # =================================================================
    # STEP 3: Prepare reference databases
    # =================================================================
    print("\nSTEP 3: Preparing reference databases...")
    t2 = time.time()

    reference_edges = {}

    # --- 3a: TRRUST (regulatory) ---
    print("\n  3a: TRRUST (transcriptional regulation)...")
    trrust_path = os.path.join(REF_DIR, "trrust_human.tsv")
    trrust_df = pd.read_csv(trrust_path, sep='\t', header=None,
                             names=['tf', 'target', 'mode', 'pmid'])
    trrust_pairs = set()
    for _, row in trrust_df.iterrows():
        tf = row['tf'].upper()
        tgt = row['target'].upper()
        trrust_pairs.add((min(tf, tgt), max(tf, tgt)))
    trrust_in_hvg = set((a, b) for a, b in trrust_pairs
                         if a in hvg_upper_set and b in hvg_upper_set)
    reference_edges['TRRUST'] = trrust_in_hvg
    print(f"    TRRUST: {len(trrust_pairs)} total, {len(trrust_in_hvg)} in HVG universe")

    # --- 3b: STRING PPI ---
    print("\n  3b: STRING (protein-protein interactions)...")
    string_cache = os.path.join(REF_EDGE_DIR, "string_ppi_edges.json")
    if os.path.exists(string_cache):
        print("    Loading cached STRING edges...")
        with open(string_cache) as f:
            string_data = json.load(f)
        string_pairs_700 = set(tuple(p) for p in string_data['pairs_700'])
        string_pairs_900 = set(tuple(p) for p in string_data['pairs_900'])
    else:
        # Use STRING API to get interactions for our genes
        # Map gene names to STRING IDs first
        print("    Querying STRING API for gene mappings...")
        gene_list = list(hvg_upper_set)

        # STRING API: get_string_ids - map gene names to STRING identifiers
        string_pairs_700 = set()
        string_pairs_900 = set()

        # Process in batches of 500 genes
        batch_size = 500
        all_string_interactions = []

        for batch_start in range(0, len(gene_list), batch_size):
            batch = gene_list[batch_start:batch_start + batch_size]
            identifiers = '%0d'.join(batch)

            # Get network interactions
            url = f"https://string-db.org/api/tsv/network"
            params = {
                'identifiers': '\n'.join(batch),
                'species': 9606,
                'required_score': 700,
                'caller_identity': 'biomech_interp_paper',
                'network_type': 'physical',
            }
            try:
                resp = requests.post(url, data=params, timeout=120)
                resp.raise_for_status()
                lines = resp.text.strip().split('\n')
                if len(lines) > 1:
                    header = lines[0].split('\t')
                    for line in lines[1:]:
                        fields = line.split('\t')
                        if len(fields) >= 6:
                            gene_a = fields[2].upper() if len(fields) > 2 else ''
                            gene_b = fields[3].upper() if len(fields) > 3 else ''
                            score = int(fields[5]) if len(fields) > 5 else 0
                            if gene_a in hvg_upper_set and gene_b in hvg_upper_set:
                                pair = (min(gene_a, gene_b), max(gene_a, gene_b))
                                if score >= 700:
                                    string_pairs_700.add(pair)
                                if score >= 900:
                                    string_pairs_900.add(pair)
                    print(f"    Batch {batch_start//batch_size + 1}: "
                          f"{len(lines)-1} interactions, {len(string_pairs_700)} unique pairs >=700")
            except Exception as e:
                print(f"    STRING API batch failed: {e}")
                # Try with functional network if physical fails
                try:
                    params['network_type'] = 'functional'
                    resp = requests.post(url, data=params, timeout=120)
                    resp.raise_for_status()
                    lines = resp.text.strip().split('\n')
                    if len(lines) > 1:
                        for line in lines[1:]:
                            fields = line.split('\t')
                            if len(fields) >= 6:
                                gene_a = fields[2].upper() if len(fields) > 2 else ''
                                gene_b = fields[3].upper() if len(fields) > 3 else ''
                                score = int(fields[5]) if len(fields) > 5 else 0
                                if gene_a in hvg_upper_set and gene_b in hvg_upper_set:
                                    pair = (min(gene_a, gene_b), max(gene_a, gene_b))
                                    if score >= 700:
                                        string_pairs_700.add(pair)
                                    if score >= 900:
                                        string_pairs_900.add(pair)
                except Exception as e2:
                    print(f"    Functional network also failed: {e2}")

            time.sleep(1)  # Rate limiting

        # Save cache
        with open(string_cache, 'w') as f:
            json.dump({
                'pairs_700': [list(p) for p in string_pairs_700],
                'pairs_900': [list(p) for p in string_pairs_900],
            }, f)

    reference_edges['STRING_700'] = string_pairs_700
    reference_edges['STRING_900'] = string_pairs_900
    print(f"    STRING >=700: {len(string_pairs_700)} pairs in HVG")
    print(f"    STRING >=900: {len(string_pairs_900)} pairs in HVG")

    # --- 3c: Reactome pathways ---
    print("\n  3c: Reactome (pathway co-membership)...")
    reactome_cache = os.path.join(REF_EDGE_DIR, "reactome_gene_sets.json")
    if os.path.exists(reactome_cache):
        print("    Loading cached Reactome gene sets...")
        with open(reactome_cache) as f:
            reactome_sets = json.load(f)
    else:
        print("    Downloading from Enrichr...")
        try:
            url = f"{ENRICHR_BASE}/geneSetLibrary?mode=text&libraryName=Reactome_2022"
            resp = download_with_retry(url, timeout=60)
            reactome_sets = parse_enrichr_gmt(resp.text, hvg_upper_set,
                                               min_genes=5, max_genes=500)
            with open(reactome_cache, 'w') as f:
                json.dump(reactome_sets, f)
        except Exception as e:
            print(f"    Enrichr download failed: {e}")
            # Try alternative: Reactome direct download
            try:
                url2 = "https://reactome.org/download/current/ReactomePathways.gmt.zip"
                resp = download_with_retry(url2, timeout=120)
                import zipfile
                z = zipfile.ZipFile(io.BytesIO(resp.content))
                for name in z.namelist():
                    if name.endswith('.gmt'):
                        text = z.read(name).decode('utf-8')
                        reactome_sets = parse_enrichr_gmt(text, hvg_upper_set,
                                                           min_genes=5, max_genes=500)
                        break
                with open(reactome_cache, 'w') as f:
                    json.dump(reactome_sets, f)
            except Exception as e2:
                print(f"    Direct download also failed: {e2}")
                reactome_sets = {}

    reactome_pairs = gene_sets_to_pairs(reactome_sets, hvg_upper_set)
    reference_edges['Reactome'] = reactome_pairs
    print(f"    Reactome: {len(reactome_sets)} pathways, {len(reactome_pairs)} co-membership pairs")

    # --- 3d: KEGG pathways ---
    print("\n  3d: KEGG (pathway co-membership)...")
    kegg_cache = os.path.join(REF_EDGE_DIR, "kegg_gene_sets.json")
    if os.path.exists(kegg_cache):
        print("    Loading cached KEGG gene sets...")
        with open(kegg_cache) as f:
            kegg_sets = json.load(f)
    else:
        print("    Downloading from Enrichr...")
        try:
            url = f"{ENRICHR_BASE}/geneSetLibrary?mode=text&libraryName=KEGG_2021_Human"
            resp = download_with_retry(url, timeout=60)
            kegg_sets = parse_enrichr_gmt(resp.text, hvg_upper_set,
                                           min_genes=5, max_genes=500)
            with open(kegg_cache, 'w') as f:
                json.dump(kegg_sets, f)
        except Exception as e:
            print(f"    KEGG download failed: {e}")
            kegg_sets = {}

    kegg_pairs = gene_sets_to_pairs(kegg_sets, hvg_upper_set)
    reference_edges['KEGG'] = kegg_pairs
    print(f"    KEGG: {len(kegg_sets)} pathways, {len(kegg_pairs)} co-membership pairs")

    # --- 3e: GO Biological Process ---
    print("\n  3e: GO Biological Process (functional co-annotation)...")
    go_cache = os.path.join(REF_EDGE_DIR, "go_bp_gene_sets.json")
    if os.path.exists(go_cache):
        print("    Loading cached GO BP gene sets...")
        with open(go_cache) as f:
            go_sets = json.load(f)
    else:
        print("    Downloading from Enrichr...")
        try:
            url = f"{ENRICHR_BASE}/geneSetLibrary?mode=text&libraryName=GO_Biological_Process_2023"
            resp = download_with_retry(url, timeout=60)
            go_sets = parse_enrichr_gmt(resp.text, hvg_upper_set,
                                         min_genes=5, max_genes=500)
            with open(go_cache, 'w') as f:
                json.dump(go_sets, f)
        except Exception as e:
            print(f"    GO BP download failed: {e}")
            go_sets = {}

    go_pairs = gene_sets_to_pairs(go_sets, hvg_upper_set)
    reference_edges['GO_BP'] = go_pairs
    print(f"    GO BP: {len(go_sets)} terms, {len(go_pairs)} co-annotation pairs")

    # Save reference edge set summary
    ref_summary = {}
    for name, pairs in reference_edges.items():
        ref_summary[name] = {
            'n_pairs': len(pairs),
            'n_genes': len(set(g for pair in pairs for g in pair)),
        }
    with open(os.path.join(REF_EDGE_DIR, "reference_summary.json"), 'w') as f:
        json.dump(ref_summary, f, indent=2, default=_json_default)

    print(f"\n  Reference databases prepared: {time.time()-t2:.1f}s")

    # =================================================================
    # STEP 4: Compute AUROC against each reference
    # =================================================================
    print("\nSTEP 4: Computing AUROC for each reference x layer...")
    t3 = time.time()

    # Results structure: {reference: {method: {layer: auroc}}}
    all_aurocs = {}
    for ref_name, ref_pairs in reference_edges.items():
        if len(ref_pairs) < 10:
            print(f"  Skipping {ref_name} (only {len(ref_pairs)} pairs)")
            continue

        print(f"\n  {ref_name} ({len(ref_pairs)} pairs):")
        all_aurocs[ref_name] = {'attention': {}, 'correlation': None, 'n_pairs': 0}

        # Attention AUROC per layer
        for li in range(N_LAYERS):
            auroc, n_pairs = compute_auroc_for_reference(
                attn_matrices[li], ref_pairs, hvg_genes, hvg_to_local,
                mat_idx_to_gene, token_to_mat_idx, hvg_to_token, n_hvg_tok,
                symmetric=False, sample_negatives=min(50000, len(ref_pairs) * 10))
            all_aurocs[ref_name]['attention'][li] = round(auroc, 4) if auroc is not None else None
            all_aurocs[ref_name]['n_pairs'] = n_pairs

        # Correlation AUROC (single value)
        auroc_corr, _ = compute_auroc_for_reference(
            corr_mat_idx, ref_pairs, hvg_genes, hvg_to_local,
            mat_idx_to_gene, token_to_mat_idx, hvg_to_token, n_hvg_tok,
            symmetric=True, sample_negatives=min(50000, len(ref_pairs) * 10))
        all_aurocs[ref_name]['correlation'] = round(auroc_corr, 4) if auroc_corr is not None else None

        # Print layer profile
        attn_vals = [all_aurocs[ref_name]['attention'].get(li) for li in range(N_LAYERS)]
        attn_valid = [v for v in attn_vals if v is not None]
        best_layer = max(range(N_LAYERS), key=lambda li: all_aurocs[ref_name]['attention'].get(li, 0))
        best_auroc = all_aurocs[ref_name]['attention'].get(best_layer, 0)
        print(f"    Best attn layer: L{best_layer} = {best_auroc:.4f}")
        print(f"    Correlation: {all_aurocs[ref_name]['correlation']}")
        print(f"    Evaluable pairs: {n_pairs}")

    # Save intermediate
    with open(os.path.join(OUT_DIR, "attention_characterization_aurocs.json"), 'w') as f:
        json.dump(all_aurocs, f, indent=2, default=_json_default)

    print(f"\n  AUROC computation time: {time.time()-t3:.1f}s")

    # =================================================================
    # STEP 5: Statistical comparisons
    # =================================================================
    print("\nSTEP 5: Statistical comparisons...")
    t4 = time.time()

    stat_results = {}

    for ref_name in all_aurocs:
        attn_aurocs = all_aurocs[ref_name]['attention']
        corr_auroc = all_aurocs[ref_name]['correlation']
        if corr_auroc is None:
            continue

        # Best attention layer
        valid_layers = {li: v for li, v in attn_aurocs.items() if v is not None}
        if not valid_layers:
            continue
        best_layer = max(valid_layers, key=valid_layers.get)
        best_attn = valid_layers[best_layer]

        stat_results[ref_name] = {
            'best_attn_layer': int(best_layer),
            'best_attn_auroc': best_attn,
            'correlation_auroc': corr_auroc,
            'attn_minus_corr': round(best_attn - corr_auroc, 4),
            'attn_better': best_attn > corr_auroc,
        }

        # Layer-averaged attention vs correlation
        layer_aurocs = [v for v in valid_layers.values()]
        mean_attn = np.mean(layer_aurocs)
        stat_results[ref_name]['mean_attn_auroc'] = round(float(mean_attn), 4)
        stat_results[ref_name]['mean_attn_minus_corr'] = round(float(mean_attn - corr_auroc), 4)

    # Rank references by attention AUROC
    ref_ranking = sorted(stat_results.items(),
                          key=lambda x: x[1]['best_attn_auroc'], reverse=True)

    print("\n  Reference ranking by best attention AUROC:")
    for rank, (ref_name, res) in enumerate(ref_ranking, 1):
        print(f"    {rank}. {ref_name:15s}: attn L{res['best_attn_layer']}={res['best_attn_auroc']:.4f}, "
              f"corr={res['correlation_auroc']:.4f}, Δ={res['attn_minus_corr']:+.4f}")

    # Cross-reference comparison: does attention predict non-regulatory references
    # better than TRRUST?
    if 'TRRUST' in stat_results:
        trrust_attn = stat_results['TRRUST']['best_attn_auroc']
        print(f"\n  TRRUST attention AUROC: {trrust_attn:.4f}")
        for ref_name, res in stat_results.items():
            if ref_name == 'TRRUST':
                continue
            diff = res['best_attn_auroc'] - trrust_attn
            print(f"    {ref_name} vs TRRUST: {res['best_attn_auroc']:.4f} - {trrust_attn:.4f} = {diff:+.4f}")

    print(f"  Statistical comparison time: {time.time()-t4:.1f}s")

    # =================================================================
    # STEP 6: Enrichment analysis (top/bottom edges)
    # =================================================================
    print("\nSTEP 6: Enrichment analysis of top/bottom attention edges...")
    t5 = time.time()

    enrichment_results = {}

    # Use best overall layer (L15 from paper, or recompute)
    eval_layers = [0, 5, 10, 14, 17]  # representative layers

    for li in eval_layers:
        mat = attn_matrices[li]
        # Get all edge scores (upper triangle, excluding diagonal)
        edge_scores = []
        edge_gene_pairs = []
        for i in range(n_hvg_tok):
            for j in range(i + 1, n_hvg_tok):
                score = max(mat[i, j], mat[j, i])
                edge_scores.append(score)
                g1 = mat_idx_to_gene[i].upper()
                g2 = mat_idx_to_gene[j].upper()
                edge_gene_pairs.append((min(g1, g2), max(g1, g2)))

        edge_scores = np.array(edge_scores)
        n_edges = len(edge_scores)

        # Top and bottom 1000 edges
        top_k = min(1000, n_edges // 10)
        top_idx = np.argsort(-edge_scores)[:top_k]
        bot_idx = np.argsort(edge_scores)[:top_k]

        top_pairs = set(edge_gene_pairs[i] for i in top_idx)
        bot_pairs = set(edge_gene_pairs[i] for i in bot_idx)

        layer_enrichment = {}
        for ref_name, ref_pairs in reference_edges.items():
            if len(ref_pairs) < 10:
                continue

            # Fisher's exact test for top edges
            all_pairs_set = set(edge_gene_pairs)
            a = len(top_pairs & ref_pairs)  # top & ref
            b = len(top_pairs - ref_pairs)  # top & not ref
            c = len(ref_pairs & all_pairs_set) - a  # not top & ref (approximate)
            d = n_edges - a - b - c  # not top & not ref

            # Ensure non-negative
            c = max(c, 0)
            d = max(d, 0)

            if a + b > 0 and a + c > 0:
                odds_ratio_top, p_top = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
            else:
                odds_ratio_top, p_top = 1.0, 1.0

            # Fisher's exact test for bottom edges
            a_bot = len(bot_pairs & ref_pairs)
            b_bot = len(bot_pairs - ref_pairs)
            c_bot = len(ref_pairs & all_pairs_set) - a_bot
            d_bot = n_edges - a_bot - b_bot - c_bot
            c_bot = max(c_bot, 0)
            d_bot = max(d_bot, 0)

            if a_bot + b_bot > 0 and a_bot + c_bot > 0:
                odds_ratio_bot, p_bot = stats.fisher_exact(
                    [[a_bot, b_bot], [c_bot, d_bot]], alternative='greater')
            else:
                odds_ratio_bot, p_bot = 1.0, 1.0

            layer_enrichment[ref_name] = {
                'top_overlap': int(a),
                'top_odds_ratio': round(float(odds_ratio_top), 3),
                'top_p_value': float(p_top),
                'bottom_overlap': int(a_bot),
                'bottom_odds_ratio': round(float(odds_ratio_bot), 3),
                'bottom_p_value': float(p_bot),
            }

        enrichment_results[f'L{li}'] = layer_enrichment

        # Print summary for this layer
        print(f"\n    Layer {li} (top-{top_k} edges):")
        for ref_name, enr in layer_enrichment.items():
            sig_top = "*" if enr['top_p_value'] < 0.05 else ""
            sig_bot = "*" if enr['bottom_p_value'] < 0.05 else ""
            print(f"      {ref_name:15s}: top OR={enr['top_odds_ratio']:6.2f} "
                  f"(p={enr['top_p_value']:.2e}){sig_top}, "
                  f"bot OR={enr['bottom_odds_ratio']:6.2f} "
                  f"(p={enr['bottom_p_value']:.2e}){sig_bot}")

    print(f"\n  Enrichment analysis time: {time.time()-t5:.1f}s")

    # =================================================================
    # STEP 7: Partial correlation analysis
    # =================================================================
    print("\nSTEP 7: Partial correlation (controlling for expression similarity)...")
    t6 = time.time()

    partial_results = {}
    # For each reference, test: does attention still predict membership
    # after controlling for expression correlation?
    for ref_name, ref_pairs in reference_edges.items():
        if len(ref_pairs) < 50:
            continue

        # Sample gene pairs (positive and negative)
        pos_list = []
        neg_list = []
        pos_sampled = list(ref_pairs)[:5000]  # cap for speed

        for g1, g2 in pos_sampled:
            if g1 not in hvg_upper_set or g2 not in hvg_upper_set:
                continue
            g1_local = hvg_to_local.get(g1)
            g2_local = hvg_to_local.get(g2)
            if g1_local is None or g2_local is None:
                continue
            g1_orig = hvg_genes[g1_local]
            g2_orig = hvg_genes[g2_local]
            g1_tok = hvg_to_token.get(g1_orig)
            g2_tok = hvg_to_token.get(g2_orig)
            if g1_tok is None or g2_tok is None:
                continue
            g1_mat = token_to_mat_idx.get(g1_tok)
            g2_mat = token_to_mat_idx.get(g2_tok)
            if g1_mat is not None and g2_mat is not None:
                # Use best layer attention
                best_l = max(range(N_LAYERS),
                             key=lambda l: all_aurocs.get(ref_name, {}).get('attention', {}).get(l, 0))
                attn_score = max(attn_matrices[best_l][g1_mat, g2_mat],
                                 attn_matrices[best_l][g2_mat, g1_mat])
                corr_score = corr_mat_idx[g1_mat, g2_mat]
                pos_list.append((attn_score, corr_score, 1))

        # Sample equal number of negatives
        np.random.seed(42)
        pos_set = set((min(a, b), max(a, b)) for a, b in ref_pairs)
        neg_count = 0
        max_neg = len(pos_list)
        attempts = 0
        while neg_count < max_neg and attempts < max_neg * 20:
            i = np.random.randint(n_hvg_tok)
            j = np.random.randint(n_hvg_tok)
            if i == j:
                attempts += 1
                continue
            g1 = mat_idx_to_gene[i].upper()
            g2 = mat_idx_to_gene[j].upper()
            pair = (min(g1, g2), max(g1, g2))
            if pair in pos_set:
                attempts += 1
                continue
            best_l = max(range(N_LAYERS),
                         key=lambda l: all_aurocs.get(ref_name, {}).get('attention', {}).get(l, 0))
            attn_score = max(attn_matrices[best_l][i, j], attn_matrices[best_l][j, i])
            corr_score = corr_mat_idx[i, j]
            neg_list.append((attn_score, corr_score, 0))
            neg_count += 1
            attempts += 1

        if len(pos_list) < 20 or len(neg_list) < 20:
            continue

        all_data = pos_list + neg_list
        attn_arr = np.array([x[0] for x in all_data])
        corr_arr = np.array([x[1] for x in all_data])
        label_arr = np.array([x[2] for x in all_data])

        # Raw correlation between attention and membership
        rho_attn, p_attn = stats.pointbiserialr(label_arr, attn_arr)
        rho_corr_raw, p_corr_raw = stats.pointbiserialr(label_arr, corr_arr)

        # Partial correlation: regress out expression correlation from attention
        # residualize attention on correlation
        if np.std(corr_arr) > 0 and np.std(attn_arr) > 0:
            slope = np.corrcoef(corr_arr, attn_arr)[0, 1] * np.std(attn_arr) / np.std(corr_arr)
            intercept = np.mean(attn_arr) - slope * np.mean(corr_arr)
            attn_resid = attn_arr - (slope * corr_arr + intercept)
            rho_partial, p_partial = stats.pointbiserialr(label_arr, attn_resid)
        else:
            rho_partial, p_partial = 0.0, 1.0

        partial_results[ref_name] = {
            'n_pairs': len(all_data),
            'n_positive': len(pos_list),
            'attn_corr_with_membership': round(float(rho_attn), 4),
            'attn_p': float(p_attn),
            'expr_corr_with_membership': round(float(rho_corr_raw), 4),
            'expr_p': float(p_corr_raw),
            'partial_attn_corr': round(float(rho_partial), 4),
            'partial_p': float(p_partial),
            'signal_retained': round(abs(rho_partial) / max(abs(rho_attn), 1e-8), 3),
        }

        print(f"    {ref_name:15s}: raw r={rho_attn:.4f} (p={p_attn:.2e}), "
              f"partial r={rho_partial:.4f} (p={p_partial:.2e}), "
              f"retained={partial_results[ref_name]['signal_retained']:.1%}")

    print(f"  Partial correlation time: {time.time()-t6:.1f}s")

    # =================================================================
    # STEP 8: Save all results
    # =================================================================
    print("\nSTEP 8: Saving results...")
    total_time = time.time() - total_t0

    results = {
        'experiment': 'analysis_b_biological_characterization',
        'n_hvg_genes': len(hvg_genes),
        'n_hvg_tok': n_hvg_tok,
        'reference_summary': ref_summary,
        'auroc_results': all_aurocs,
        'statistical_comparisons': stat_results,
        'enrichment_results': enrichment_results,
        'partial_correlation_results': partial_results,
        'total_time_seconds': round(total_time, 1),
    }

    with open(os.path.join(OUT_DIR, "attention_characterization_results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)

    # Write human-readable summary
    summary_lines = [
        "# Analysis B: Biological Characterization of Attention — Results Summary",
        "",
        "## Reference Databases",
    ]
    for name, info in ref_summary.items():
        summary_lines.append(f"- **{name}**: {info['n_pairs']} pairs, {info['n_genes']} genes in HVG")

    summary_lines.extend(["", "## AUROC Results (best layer)", ""])
    summary_lines.append(f"{'Reference':15s} {'Best Attn':>10s} {'Corr':>10s} {'Delta':>10s}")
    summary_lines.append("-" * 50)
    for ref_name, res in sorted(stat_results.items(),
                                  key=lambda x: x[1]['best_attn_auroc'], reverse=True):
        summary_lines.append(
            f"{ref_name:15s} {res['best_attn_auroc']:10.4f} {res['correlation_auroc']:10.4f} "
            f"{res['attn_minus_corr']:+10.4f}")

    summary_lines.extend(["", "## Key Findings", ""])
    # Determine which reference attention predicts best
    if stat_results:
        best_ref = max(stat_results.items(), key=lambda x: x[1]['best_attn_auroc'])
        summary_lines.append(
            f"Attention best predicts: **{best_ref[0]}** (AUROC={best_ref[1]['best_attn_auroc']:.4f})")

        # Which reference shows largest attention advantage over correlation?
        if any(r['attn_minus_corr'] > 0 for r in stat_results.values()):
            best_advantage = max(stat_results.items(), key=lambda x: x[1]['attn_minus_corr'])
            summary_lines.append(
                f"Largest attention advantage: **{best_advantage[0]}** "
                f"(Δ={best_advantage[1]['attn_minus_corr']:+.4f})")

    summary_lines.extend(["", "## Partial Correlation (controlling for expression similarity)", ""])
    for ref_name, res in partial_results.items():
        summary_lines.append(
            f"- **{ref_name}**: raw r={res['attn_corr_with_membership']:.4f}, "
            f"partial r={res['partial_attn_corr']:.4f} "
            f"({res['signal_retained']:.0%} retained)")

    summary_lines.extend(["", "## Enrichment (top-1000 edges)", ""])
    for layer_key, layer_enr in enrichment_results.items():
        summary_lines.append(f"### {layer_key}")
        for ref_name, enr in layer_enr.items():
            sig = " *" if enr['top_p_value'] < 0.05 else ""
            summary_lines.append(
                f"  {ref_name}: OR={enr['top_odds_ratio']:.2f}, "
                f"p={enr['top_p_value']:.2e}{sig}")

    summary_lines.extend(["", f"Total runtime: {total_time/60:.1f} minutes"])

    with open(os.path.join(OUT_DIR, "attention_characterization_summary.md"), 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f"\n{'='*70}")
    print("ANALYSIS B COMPLETE")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == '__main__':
    run()
