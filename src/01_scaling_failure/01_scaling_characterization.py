#!/usr/bin/env python3
"""
Fine-Grained Scaling Characterization for NMI Paper

Addresses reviewer concern: "only four cell-count points for the core scaling claim — 
insufficient to justify confident statements about curve shape."

This script tests GRN recovery using Geneformer at multiple cell count points
to characterize the scaling relationship in detail.
"""
import os, sys, pickle, json, time, traceback, gc
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from scipy.stats import bootstrap
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUT_DIR = "/mnt/d/openclaw/biodyn-nmi-paper/experiments/scaling_characterization"
DATA_PATH = "/mnt/d/openclaw/mechinterp-bio/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad"
CELL_COUNTS = [25, 50, 100, 150, 200, 300, 500, 750, 1000]  # Cell count points to test
N_BOOTSTRAP = 50   # Bootstrap iterations for confidence intervals
N_REPEATS = 2     # Repeat each cell count N times with different random samples
TOP_K = 1000  # Top genes to consider
MAX_SEQ = 256  # Max sequence length
RANDOM_SEED = 42

# Best performing layer/head from previous analyses
BEST_LAYER = 13
BEST_HEAD = None  # Use mean across all heads (pooled attention)

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

log_file = open(os.path.join(OUT_DIR, "scaling_characterization.log"), "w", buffering=1)
log("=== Fine-Grained Scaling Characterization ===")

def load_geneformer_components():
    """Load Geneformer dictionaries and model"""
    import geneformer
    GF_DIR = os.path.dirname(geneformer.__file__)
    
    with open(os.path.join(GF_DIR, "token_dictionary_gc104M.pkl"), "rb") as f:
        token_dict = pickle.load(f)
    with open(os.path.join(GF_DIR, "gene_median_dictionary_gc104M.pkl"), "rb") as f:
        gene_median_dict = pickle.load(f)
    with open(os.path.join(GF_DIR, "gene_name_id_dict_gc104M.pkl"), "rb") as f:
        gene_name_id_dict = pickle.load(f)
    
    id_to_token = {int(v): k for k, v in token_dict.items()}
    
    return token_dict, gene_median_dict, gene_name_id_dict, id_to_token

def tokenize_cell(expr_vec, gene_ids, token_dict, gene_median_dict, max_len=2048):
    """Convert expression vector to token sequence"""
    if sparse.issparse(expr_vec):
        expr_vec = expr_vec.toarray().flatten()
    
    nonzero = expr_vec > 0
    genes = gene_ids[nonzero]
    vals = expr_vec[nonzero]
    
    valid = np.array([g in token_dict and g in gene_median_dict for g in genes])
    if valid.sum() == 0:
        return []
    
    genes, vals = genes[valid], vals[valid]
    medians = np.array([max(gene_median_dict[g], 1e-6) for g in genes], dtype=np.float32)
    idx = np.argsort(-(vals / medians))[:max_len]
    return [int(token_dict[genes[i]]) for i in idx]

def build_gene_matrix(model, tokens_list, device, g2i, n_genes, layer_idx, head_idx=None):
    """Build gene-gene interaction matrix from attention"""
    model.eval()
    score_sum = np.zeros((n_genes, n_genes), dtype=np.float64)
    count_mat = np.zeros((n_genes, n_genes), dtype=np.float64)
    
    for idx, tids_full in enumerate(tokens_list):
        tids = tids_full[:MAX_SEQ]
        ml = len(tids)
        if ml < 5:
            continue
        
        input_ids = torch.tensor([tids], dtype=torch.long, device=device)
        attn_mask = torch.ones(1, ml, dtype=torch.long, device=device)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            out = model.bert(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
        
        layer_attn = out.attentions[layer_idx][0]  # (heads, seq, seq)
        if head_idx is not None:
            attn = layer_attn[head_idx].float().cpu().numpy()
        else:
            attn = layer_attn.mean(dim=0).float().cpu().numpy()
        
        del out, layer_attn, input_ids, attn_mask
        if idx % 50 == 0:
            torch.cuda.empty_cache()
        
        # Map tokens to gene indices
        mapped = [(pos, g2i[tid]) for pos, tid in enumerate(tids) if tid in g2i]
        if len(mapped) < 2:
            continue
        
        positions = np.array([m[0] for m in mapped])
        indices = np.array([m[1] for m in mapped])
        sub_attn = attn[np.ix_(positions, positions)]
        
        for a in range(len(indices)):
            i = indices[a]
            for b in range(len(indices)):
                j = indices[b]
                if i != j:
                    v = sub_attn[a, b]
                    score_sum[i, j] += v
                    count_mat[i, j] += 1
    
    mask = count_mat > 0
    pooled = np.zeros_like(score_sum)
    pooled[mask] = score_sum[mask] / count_mat[mask]
    
    return pooled

def evaluate_auroc(score_matrix, gene_list, gt_edges):
    """Evaluate AUROC against ground truth edges"""
    n = len(gene_list)
    y_true = []
    y_score = []
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            label = 1 if (gene_list[i], gene_list[j]) in gt_edges else 0
            y_true.append(label)
            y_score.append(score_matrix[i, j])
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    n_pos = int(y_true.sum())
    
    if n_pos < 3:
        return None, n_pos
    
    return float(roc_auc_score(y_true, y_score)), n_pos

def bootstrap_auroc(score_matrix, gene_list, gt_edges, n_bootstrap=N_BOOTSTRAP):
    """Calculate AUROC with 95% confidence interval using bootstrap"""
    n = len(gene_list)
    y_true = []
    y_score = []
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            label = 1 if (gene_list[i], gene_list[j]) in gt_edges else 0
            y_true.append(label)
            y_score.append(score_matrix[i, j])
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    n_pos = int(y_true.sum())
    
    if n_pos < 3:
        return None, None, None, n_pos
    
    # Primary AUROC
    primary_auroc = float(roc_auc_score(y_true, y_score))
    
    # Bootstrap for confidence interval
    def auroc_statistic(y_true, y_score, indices):
        return roc_auc_score(y_true[indices], y_score[indices])
    
    try:
        data = (y_true, y_score)
        n_samples = len(y_true)
        rng = np.random.default_rng(RANDOM_SEED)
        
        # Manual bootstrap since scipy bootstrap might be tricky with this setup
        bootstrap_scores = []
        for _ in range(n_bootstrap):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            # Check if we have both positive and negative samples
            if y_true[indices].sum() > 0 and (1 - y_true[indices]).sum() > 0:
                try:
                    score = roc_auc_score(y_true[indices], y_score[indices])
                    bootstrap_scores.append(score)
                except:
                    continue
        
        if len(bootstrap_scores) >= 10:  # Need at least some bootstrap samples
            ci_lower = np.percentile(bootstrap_scores, 2.5)
            ci_upper = np.percentile(bootstrap_scores, 97.5)
        else:
            ci_lower, ci_upper = None, None
            
    except Exception as e:
        log(f"Bootstrap failed: {e}")
        ci_lower, ci_upper = None, None
    
    return primary_auroc, ci_lower, ci_upper, n_pos

def sample_cells_stratified(adata, n_cells, cell_type_col='cell_type', random_state=None):
    """Sample n_cells from dataset with stratified sampling by cell type"""
    if random_state is not None:
        np.random.seed(random_state)
    
    cell_types = adata.obs[cell_type_col].values
    unique_cts = np.unique(cell_types)
    
    # Calculate proportional sampling
    ct_counts = pd.Series(cell_types).value_counts()
    total_cells = len(cell_types)
    
    sampled_indices = []
    for ct in unique_cts:
        ct_indices = np.where(cell_types == ct)[0]
        n_ct_target = max(1, int(n_cells * len(ct_indices) / total_cells))
        n_ct_actual = min(n_ct_target, len(ct_indices))
        
        if n_ct_actual > 0:
            selected = np.random.choice(ct_indices, n_ct_actual, replace=False)
            sampled_indices.extend(selected)
    
    # If we need more cells, add randomly from remaining
    if len(sampled_indices) < n_cells:
        remaining = n_cells - len(sampled_indices)
        all_indices = np.arange(len(cell_types))
        unused = np.setdiff1d(all_indices, sampled_indices)
        if len(unused) > 0:
            additional = np.random.choice(unused, min(remaining, len(unused)), replace=False)
            sampled_indices.extend(additional)
    
    # If we have too many, randomly downsample
    if len(sampled_indices) > n_cells:
        sampled_indices = list(np.random.choice(sampled_indices, n_cells, replace=False))
    
    return sampled_indices

def fit_scaling_curves(cell_counts, aurocs):
    """Fit different scaling curve models"""
    valid_idx = ~np.isnan(aurocs)
    if valid_idx.sum() < 3:
        return {}
    
    x = np.array(cell_counts)[valid_idx]
    y = np.array(aurocs)[valid_idx]
    
    models = {}
    
    # Linear model: y = a * x + b
    try:
        def linear_func(x, a, b):
            return a * x + b
        popt, pcov = curve_fit(linear_func, x, y)
        r2 = 1 - np.sum((y - linear_func(x, *popt))**2) / np.sum((y - np.mean(y))**2)
        models['linear'] = {
            'params': popt.tolist(),
            'param_names': ['slope', 'intercept'],
            'r2': float(r2),
            'equation': f'AUROC = {popt[0]:.6f} * cells + {popt[1]:.6f}'
        }
    except Exception as e:
        log(f"Linear fit failed: {e}")
    
    # Logarithmic model: y = a * log(x) + b
    try:
        def log_func(x, a, b):
            return a * np.log(x) + b
        popt, pcov = curve_fit(log_func, x, y)
        r2 = 1 - np.sum((y - log_func(x, *popt))**2) / np.sum((y - np.mean(y))**2)
        models['logarithmic'] = {
            'params': popt.tolist(),
            'param_names': ['log_coeff', 'intercept'],
            'r2': float(r2),
            'equation': f'AUROC = {popt[0]:.6f} * log(cells) + {popt[1]:.6f}'
        }
    except Exception as e:
        log(f"Logarithmic fit failed: {e}")
    
    # Power law model: y = a * x^b + c
    try:
        def power_func(x, a, b, c):
            return a * np.power(x, b) + c
        popt, pcov = curve_fit(power_func, x, y, maxfev=2000)
        r2 = 1 - np.sum((y - power_func(x, *popt))**2) / np.sum((y - np.mean(y))**2)
        models['power'] = {
            'params': popt.tolist(),
            'param_names': ['coeff', 'exponent', 'offset'],
            'r2': float(r2),
            'equation': f'AUROC = {popt[0]:.6f} * cells^{popt[1]:.6f} + {popt[2]:.6f}'
        }
    except Exception as e:
        log(f"Power fit failed: {e}")
    
    # Exponential saturation: y = a * (1 - exp(-b * x)) + c
    try:
        def exp_sat_func(x, a, b, c):
            return a * (1 - np.exp(-b * x)) + c
        # Initial guess
        p0 = [np.max(y) - np.min(y), 0.001, np.min(y)]
        popt, pcov = curve_fit(exp_sat_func, x, y, p0=p0, maxfev=2000)
        r2 = 1 - np.sum((y - exp_sat_func(x, *popt))**2) / np.sum((y - np.mean(y))**2)
        models['exponential_saturation'] = {
            'params': popt.tolist(),
            'param_names': ['amplitude', 'rate', 'baseline'],
            'r2': float(r2),
            'equation': f'AUROC = {popt[0]:.6f} * (1 - exp(-{popt[1]:.6f} * cells)) + {popt[2]:.6f}'
        }
    except Exception as e:
        log(f"Exponential saturation fit failed: {e}")
    
    return models

def main():
    log("Starting fine-grained scaling characterization...")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    
    # Load Geneformer components
    log("Loading Geneformer components...")
    token_dict, gene_median_dict, gene_name_id_dict, id_to_token = load_geneformer_components()
    
    # Load model
    log("Loading Geneformer model...")
    from transformers import BertForMaskedLM
    model = BertForMaskedLM.from_pretrained("ctheodoris/Geneformer", 
                                           output_attentions=True).to(device).half()
    log(f"Model loaded: {model.config.num_hidden_layers}L x {model.config.num_attention_heads}H")
    
    # Load immune dataset
    log(f"Loading immune dataset: {DATA_PATH}")
    adata = sc.read_h5ad(DATA_PATH)
    log(f"Dataset shape: {adata.shape}")
    log(f"Cell types: {adata.obs['cell_type'].value_counts().head()}")
    
    # Setup gene name mapping
    gene_name_to_ensembl = {}
    if 'feature_name' in adata.var.columns:
        for ens_id, row in adata.var.iterrows():
            name = row['feature_name']
            if pd.notna(name) and name != '':
                gene_name_to_ensembl[name] = ens_id
    for gname, ens_id in gene_name_id_dict.items():
        gene_name_to_ensembl.setdefault(gname, ens_id)
    
    # Load TRRUST
    log("Loading TRRUST...")
    trrust_path = "/tmp/trrust.tsv"
    if not os.path.exists(trrust_path):
        import urllib.request
        urllib.request.urlretrieve(
            "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv", trrust_path)
    df = pd.read_csv(trrust_path, sep='\t', header=None, names=['TF', 'target', 'mode', 'pmid'])
    trrust = set()
    for _, row in df.iterrows():
        src = gene_name_to_ensembl.get(row['TF'])
        tgt = gene_name_to_ensembl.get(row['target'])
        if src and tgt:
            trrust.add((src, tgt))
    log(f"TRRUST: {len(trrust)} edges")
    
    # Pre-filter dataset for efficiency
    log("Pre-filtering dataset...")
    
    # Filter cells with minimum expression
    sc.pp.filter_cells(adata, min_genes=100)  # At least 100 expressed genes
    sc.pp.filter_genes(adata, min_cells=10)   # Gene expressed in at least 10 cells
    log(f"After filtering: {adata.shape}")
    
    # Update gene_ids array after filtering
    gene_ids = np.array(adata.var_names)
    
    # Get maximum cell count we can test
    max_testable = min(max(CELL_COUNTS), adata.n_obs)
    test_cell_counts = [c for c in CELL_COUNTS if c <= max_testable]
    log(f"Will test cell counts: {test_cell_counts}")
    
    # Setup gene vocabulary (same as original)
    log("Setting up gene vocabulary...")
    # First pass: tokenize a sample to build frequency-based gene list
    sample_indices = np.random.choice(adata.n_obs, min(1000, adata.n_obs), replace=False)
    freq = Counter()
    
    for i in sample_indices:
        expr = adata.X[i].toarray().flatten() if sparse.issparse(adata.X) else adata.X[i].flatten()
        tokens = tokenize_cell(expr, gene_ids, token_dict, gene_median_dict)
        for tid in tokens:
            if tid > 3:  # Skip special tokens
                freq[tid] += 1
    
    top_genes = [g for g, _ in freq.most_common(TOP_K)]
    g2i = {g: i for i, g in enumerate(top_genes)}
    n_genes = len(top_genes)
    gene_list = [id_to_token.get(g, f"UNK_{g}") for g in top_genes]
    log(f"Using {n_genes} top genes")
    
    # Main scaling experiment
    results = {
        'experiment_info': {
            'dataset_path': DATA_PATH,
            'dataset_shape': list(adata.shape),
            'test_cell_counts': test_cell_counts,
            'n_repeats': N_REPEATS,
            'n_bootstrap': N_BOOTSTRAP,
            'best_layer': BEST_LAYER,
            'best_head': BEST_HEAD,
            'top_k_genes': TOP_K,
            'n_trrust_edges': len(trrust)
        },
        'scaling_data': {},
        'curve_fits': {}
    }
    
    log(f"\n{'='*60}")
    log("SCALING CHARACTERIZATION EXPERIMENT")
    log(f"{'='*60}")
    
    # Test each cell count
    for cell_count in test_cell_counts:
        log(f"\n--- Testing {cell_count} cells ---")
        
        cell_results = {
            'aurocs': [],
            'ci_lowers': [],
            'ci_uppers': [],
            'n_pos_edges': [],
            'cell_type_distributions': []
        }
        
        # Repeat experiment N_REPEATS times with different random samples
        for repeat in range(N_REPEATS):
            log(f"  Repeat {repeat + 1}/{N_REPEATS}...")
            
            # Sample cells
            sampled_indices = sample_cells_stratified(
                adata, cell_count, 
                random_state=RANDOM_SEED + repeat * 1000 + cell_count
            )
            
            adata_sampled = adata[sampled_indices].copy()
            cell_types = list(adata_sampled.obs['cell_type'].values)
            cell_type_dist = dict(Counter(cell_types))
            cell_results['cell_type_distributions'].append(cell_type_dist)
            
            log(f"    Sampled {len(sampled_indices)} cells")
            log(f"    Cell types: {cell_type_dist}")
            
            # Tokenize cells
            tokens_list = []
            valid_cells = 0
            for i in range(adata_sampled.n_obs):
                expr = adata_sampled.X[i].toarray().flatten() if sparse.issparse(adata_sampled.X) else adata_sampled.X[i].flatten()
                tokens = tokenize_cell(expr, gene_ids, token_dict, gene_median_dict)
                tokens_list.append(tokens)
                if len(tokens) > 10:
                    valid_cells += 1
            
            log(f"    Valid cells with >10 tokens: {valid_cells}")
            
            if valid_cells < 10:
                log(f"    Skipping - insufficient valid cells")
                cell_results['aurocs'].append(np.nan)
                cell_results['ci_lowers'].append(np.nan)
                cell_results['ci_uppers'].append(np.nan)
                cell_results['n_pos_edges'].append(0)
                continue
            
            # Build gene interaction matrix
            log(f"    Extracting attention from layer {BEST_LAYER}...")
            gene_matrix = build_gene_matrix(
                model, tokens_list, device, g2i, n_genes, BEST_LAYER, BEST_HEAD
            )
            
            # Evaluate with bootstrap confidence intervals
            log(f"    Evaluating AUROC with bootstrap...")
            auroc, ci_lower, ci_upper, n_pos = bootstrap_auroc(
                gene_matrix, gene_list, trrust, N_BOOTSTRAP
            )
            
            if auroc is not None:
                log(f"    AUROC: {auroc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] ({n_pos} pos edges)")
                cell_results['aurocs'].append(auroc)
                cell_results['ci_lowers'].append(ci_lower)
                cell_results['ci_uppers'].append(ci_upper)
                cell_results['n_pos_edges'].append(n_pos)
            else:
                log(f"    Failed to compute AUROC")
                cell_results['aurocs'].append(np.nan)
                cell_results['ci_lowers'].append(np.nan)
                cell_results['ci_uppers'].append(np.nan)
                cell_results['n_pos_edges'].append(n_pos)
            
            del adata_sampled, gene_matrix
            gc.collect()
            torch.cuda.empty_cache()
        
        # Aggregate results across repeats
        valid_aurocs = [a for a in cell_results['aurocs'] if not np.isnan(a)]
        if valid_aurocs:
            mean_auroc = np.mean(valid_aurocs)
            std_auroc = np.std(valid_aurocs)
            mean_ci_lower = np.nanmean(cell_results['ci_lowers'])
            mean_ci_upper = np.nanmean(cell_results['ci_uppers'])
            
            log(f"  Summary for {cell_count} cells:")
            log(f"    Mean AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
            log(f"    Mean CI: [{mean_ci_lower:.4f}, {mean_ci_upper:.4f}]")
        else:
            mean_auroc = np.nan
            std_auroc = np.nan
            mean_ci_lower = np.nan
            mean_ci_upper = np.nan
            log(f"  No valid results for {cell_count} cells")
        
        results['scaling_data'][cell_count] = {
            'mean_auroc': mean_auroc,
            'std_auroc': std_auroc,
            'mean_ci_lower': mean_ci_lower,
            'mean_ci_upper': mean_ci_upper,
            'individual_results': cell_results
        }
    
    # Curve fitting analysis
    log(f"\n{'='*50}")
    log("CURVE FITTING ANALYSIS")
    log(f"{'='*50}")
    
    # Extract data for curve fitting
    cell_counts_fit = []
    aurocs_fit = []
    for cell_count in test_cell_counts:
        mean_auroc = results['scaling_data'][cell_count]['mean_auroc']
        if not np.isnan(mean_auroc):
            cell_counts_fit.append(cell_count)
            aurocs_fit.append(mean_auroc)
    
    log(f"Fitting curves to {len(cell_counts_fit)} data points")
    log(f"Cell counts: {cell_counts_fit}")
    log(f"AUROCs: {[f'{a:.4f}' for a in aurocs_fit]}")
    
    curve_models = fit_scaling_curves(cell_counts_fit, aurocs_fit)
    
    for model_name, model_info in curve_models.items():
        log(f"\n{model_name.upper()} MODEL:")
        log(f"  R² = {model_info['r2']:.4f}")
        log(f"  Equation: {model_info['equation']}")
    
    # Find best model
    if curve_models:
        best_model_name = max(curve_models.keys(), key=lambda k: curve_models[k]['r2'])
        best_model = curve_models[best_model_name]
        log(f"\nBest fitting model: {best_model_name} (R² = {best_model['r2']:.4f})")
        
        results['curve_fits'] = {
            'models': curve_models,
            'best_model': best_model_name,
            'best_r2': best_model['r2']
        }
    else:
        log("\nNo successful curve fits")
        results['curve_fits'] = {'models': {}}
    
    # Analysis of scaling pattern
    log(f"\n{'='*50}")
    log("SCALING PATTERN ANALYSIS")
    log(f"{'='*50}")
    
    if len(aurocs_fit) >= 3:
        # Check for different degradation patterns
        differences = np.diff(aurocs_fit)
        relative_changes = differences / np.array(aurocs_fit[:-1])
        
        # Linear degradation test: relatively constant differences
        diff_consistency = np.std(differences) / np.mean(np.abs(differences)) if np.mean(np.abs(differences)) > 0 else np.inf
        
        # Logarithmic pattern: decreasing rate of change
        change_trend = np.polyfit(range(len(relative_changes)), relative_changes, 1)[0]
        
        log(f"Performance differences between consecutive points: {[f'{d:.4f}' for d in differences]}")
        log(f"Relative changes: {[f'{r:.4f}' for r in relative_changes]}")
        log(f"Difference consistency (lower = more linear): {diff_consistency:.4f}")
        log(f"Change trend (negative = decreasing rate): {change_trend:.6f}")
        
        # Determine degradation pattern
        if diff_consistency < 0.3:
            degradation_pattern = "linear"
        elif change_trend < -0.001 and 'logarithmic' in curve_models:
            degradation_pattern = "logarithmic"  
        elif len(set(np.round(differences, 3))) <= 2:
            degradation_pattern = "step_function"
        else:
            degradation_pattern = "complex"
        
        log(f"\nDegradation pattern assessment: {degradation_pattern}")
        
        results['scaling_analysis'] = {
            'degradation_pattern': degradation_pattern,
            'differences': differences.tolist(),
            'relative_changes': relative_changes.tolist(),
            'diff_consistency': diff_consistency,
            'change_trend': change_trend,
            'n_data_points': len(aurocs_fit)
        }
    else:
        log("Insufficient data points for scaling pattern analysis")
        results['scaling_analysis'] = {'insufficient_data': True}
    
    # Save results
    output_file = os.path.join(OUT_DIR, "scaling_characterization_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log(f"\n{'='*60}")
    log("EXPERIMENT COMPLETED")
    log(f"{'='*60}")
    log(f"Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except Exception:
        traceback.print_exc()
        traceback.print_exc(file=log_file)
        sys.exit(1)
    finally:
        log_file.close()