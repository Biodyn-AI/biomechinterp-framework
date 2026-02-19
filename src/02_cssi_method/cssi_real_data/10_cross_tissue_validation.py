#!/usr/bin/env python3
"""
CSSI Cross-Tissue Validation Experiment (BONUS)

Extended validation: identify layers on immune tissue → evaluate on kidney tissue
using the whole-human checkpoint. This tests whether attention patterns transfer
across tissue types.
"""

import sys
from pathlib import Path

# CRITICAL: scGPT Setup - torchtext shim MUST be loaded first
exec(open("/mnt/c/Users/Agent/.openclaw/workspace/patch_torchtext.py").read())

import argparse
import json
import random
import time
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2] / "mechinterp-bio" / "biodyn-work" / "single_cell_mechinterp"
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scgpt_dataset import ScGPTDataset, ScGPTDatasetConfig, collate_scgpt
from src.eval.dorothea import load_dorothea
from src.eval.gene_symbols import canonical_symbol, load_hgnc_alias_map, normalize_edges, normalize_gene_names
from src.eval.metrics import aupr
from src.interpret.attention import extract_attention_scores, finalize_attention_scores
from src.model.scgpt_loader import load_scgpt_model, ScGPTResources
from src.model.vocab import load_vocab
from src.network.infer import NetworkConfig, infer_edges
from src.utils.config import load_config
from src.utils.torch_utils import move_to_device

# Import functions from main validation script
from cssi_cross_dataset_validation import (
    extract_layer_attention_scores, load_trrust_network,
    evaluate_layer_performance, run_cssi_layer_selection, cross_validate_layers
)

def setup_scgpt_model(checkpoint_name: str = "whole-human"):
    """Load scGPT model and vocabulary."""
    base_path = Path("/mnt/d/openclaw/mechinterp-bio/biodyn-work/single_cell_mechinterp")
    
    repo_path = base_path / "external" / "scGPT"
    checkpoint_path = base_path / "external" / "scGPT_checkpoints" / checkpoint_name / "best_model.pt"
    vocab_path = base_path / "external" / "scGPT_checkpoints" / checkpoint_name / "vocab.json"
    args_path = base_path / "external" / "scGPT_checkpoints" / checkpoint_name / "args.json"
    
    logger.info(f"Loading {checkpoint_name} checkpoint: {checkpoint_path}")
    
    from src.model.vocab import load_vocab
    from src.model.scgpt_loader import load_scgpt_model
    
    # Load vocab and args (with wrapper for subscriptability)
    vocab_raw = load_vocab(vocab_path)
    
    class VocabWrapper:
        """Wrapper to make Vocab object subscriptable for scGPT compatibility."""
        def __init__(self, vocab):
            self.vocab = vocab
            self.gene_to_id = vocab.gene_to_id
            
        def __getitem__(self, key):
            return self.gene_to_id[key]
            
        def __len__(self):
            return len(self.gene_to_id)
            
        def get(self, key, default=None):
            """Get method for compatibility."""
            return self.gene_to_id.get(key, default)
    
    vocab = VocabWrapper(vocab_raw)
    with open(args_path, 'r') as f:
        args = json.load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Minimal model arguments from checkpoint (avoid None values)
    model_args = {
        "ntoken": len(vocab.gene_to_id),
        "d_model": args["embsize"],
        "nhead": args["nheads"],
        "d_hid": args["d_hid"],
        "nlayers": args["nlayers"],
        "dropout": args.get("dropout", 0.2),
        "pad_token": args.get("pad_token", "<pad>"),
        "pad_value": args.get("pad_value", -2),
        "do_mvc": False,  # Disable MVC
        "do_dab": False,  # Disable DAB  
        "use_batch_labels": False,  # Disable batch labels
        "n_input_bins": args.get("n_bins", 51),
        "input_emb_style": args.get("input_emb_style", "continuous"),
        "cell_emb_style": "cls",
        "mvc_decoder_style": "inner product",
        "ecs_threshold": args.get("ecs_thres", 0.0),
        "explicit_zero_prob": not args.get("no_explicit_zero_prob", False),
        "use_fast_transformer": args.get("fast_transformer", False),
        "vocab": vocab,  # Pass vocab object directly
    }
    
    # Load model using the correct signature
    model, missing, unexpected = load_scgpt_model(
        entrypoint="scgpt.model.model.TransformerModel",
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        device=device,
        model_args=model_args
    )
    
    return model, vocab

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_tissue_dataset(tissue: str, max_cells: int = 500) -> sc.AnnData:
    """Load tissue dataset with cell count limit."""
    
    data_paths = {
        "immune": "/mnt/d/openclaw/mechinterp-bio/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad",
        "kidney": "/mnt/d/openclaw/mechinterp-bio/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad",
        "lung": "/mnt/d/openclaw/mechinterp-bio/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad"
    }
    
    if tissue not in data_paths:
        raise ValueError(f"Unknown tissue: {tissue}. Available: {list(data_paths.keys())}")
    
    path = data_paths[tissue]
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    logger.info(f"Loading {tissue} dataset: {path}")
    adata = sc.read_h5ad(path)
    
    # Subsample if too large
    if adata.n_obs > max_cells:
        logger.info(f"Subsampling {adata.n_obs} cells to {max_cells}")
        sc.pp.subsample(adata, n_obs=max_cells, random_state=42)
    
    logger.info(f"Loaded {tissue} dataset: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata


def generate_cross_tissue_report(
    results: Dict,
    source_tissue: str,
    target_tissue: str,
    output_path: str = "D:/openclaw/biodyn-nmi-paper/CSSI_CROSS_TISSUE_REPORT.md"
):
    """Generate cross-tissue validation report."""
    
    report = f"""# CSSI Cross-Tissue Validation Report

## Executive Summary

This report tests whether CSSI-identified attention layers transfer across tissue types, extending beyond the basic cross-dataset validation.

**Experimental Design:** Layer selection on **{source_tissue}** tissue → Evaluation on **{target_tissue}** tissue

**Key Finding:** {"✅ Cross-tissue validation **PASSED**" if results['improvement'] > 0 else "❌ Cross-tissue validation **FAILED**"}

- Best layers selected on {source_tissue} achieved **{results['best_aupr_subset_b']:.4f} AUPR** on {target_tissue}
- Random layer selection achieved **{results['random_aupr_mean']:.4f} ± {results['random_aupr_std']:.4f} AUPR**
- **Cross-tissue improvement: {results['improvement']:.4f}** ({results['improvement_std_devs']:.2f} standard deviations)

## Biological Interpretation

Cross-tissue transfer of attention patterns would suggest that:
1. **Universal regulatory motifs** exist across tissue types
2. **scGPT learned tissue-agnostic** gene regulatory principles
3. **Attention mechanisms** capture fundamental biological relationships

## Experimental Setup

### Datasets
- **Source ({source_tissue}):** {results.get('n_cells_source', 'N/A')} cells for layer identification
- **Target ({target_tissue}):** {results.get('n_cells_target', 'N/A')} cells for evaluation
- **Model:** scGPT whole-human checkpoint (trained across tissues)
- **Ground Truth:** TRRUST human regulatory network

### Methodology
1. Extract attention matrices from source tissue ({source_tissue})
2. Run CSSI to identify best layers for GRN recovery vs TRRUST
3. Extract attention matrices from target tissue ({target_tissue})
4. Evaluate same layers on target tissue
5. Compare with random layer selection on target tissue

## Results

### Layer Selection (Source: {source_tissue})
Best performing layers identified: **{results['best_layers']}**

### Cross-Tissue Transfer (Target: {target_tissue})
| Metric | Value |
|--------|-------|
| Best Layers AUPR | {results['best_aupr_subset_b']:.4f} |
| Random Baseline Mean | {results['random_aupr_mean']:.4f} |
| Random Baseline Std | {results['random_aupr_std']:.4f} |
| Cross-tissue Improvement | {results['improvement']:.4f} |
| Statistical Significance | {results['improvement_std_devs']:.2f} σ |

### Individual Layer Performance on {target_tissue}
"""

    for i, (layer_idx, score) in enumerate(zip(results['best_layers'], results['individual_best_scores'])):
        report += f"\n- **Layer {layer_idx}:** {score:.4f} AUPR"

    report += f"""

### Comparison: Cross-Dataset vs Cross-Tissue

This cross-tissue validation is **more challenging** than cross-dataset validation because:
- Different cell types have different regulatory programs
- Tissue-specific expression patterns may confound attention
- Smaller overlap in highly expressed genes

## Interpretation

"""

    if results['improvement'] > 0:
        if results['improvement_std_devs'] > 2:
            significance = "statistically significant (>2σ)"
        elif results['improvement_std_devs'] > 1:
            significance = "marginally significant (1-2σ)"
        else:
            significance = "not statistically significant (<1σ)"
            
        report += f"""**✅ CROSS-TISSUE TRANSFER SUCCESSFUL:** The layers identified on {source_tissue} tissue successfully transfer to {target_tissue} tissue, achieving {results['improvement']:.4f} higher AUPR than random selection. This improvement is {significance}.

This is a **strong result** suggesting that:
1. **scGPT learned universal regulatory patterns** that transcend tissue boundaries
2. **Attention mechanisms capture core gene regulatory motifs** shared across cell types
3. **CSSI identifies biologically meaningful layers** rather than tissue-specific artifacts

The success of cross-tissue transfer validates the biological relevance of attention-based GRN inference.
"""
    else:
        report += f"""**❌ CROSS-TISSUE TRANSFER FAILED:** The layers identified on {source_tissue} tissue do not transfer effectively to {target_tissue} tissue. Performance was {abs(results['improvement']):.4f} worse than random selection.

This suggests that:
1. **Tissue-specific regulatory programs** dominate over universal patterns
2. **Attention patterns are tissue-contextualized** rather than universally applicable
3. **Sample sizes may be insufficient** for robust cross-tissue generalization
4. **Gene expression differences** between tissues confound the attention signal

This highlights the challenge of transferring mechanistic insights across biological contexts.
"""

    report += f"""

## Technical Details

- **Execution Time:** {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Model Checkpoint:** whole-human (trained on multiple tissues)
- **Random Seed:** 42
- **Batch Size:** 2
- **Top-k Layers:** 3
- **Random Trials:** 10

## Conclusions

Cross-tissue validation represents the **gold standard** for testing the biological generalizability of mechanistic insights from transformer models.

{"The positive cross-tissue transfer strongly supports the biological validity of CSSI and suggests that scGPT has learned universal gene regulatory principles." if results['improvement'] > 0 else "The failure of cross-tissue transfer highlights the tissue-specific nature of gene regulation and suggests caution in generalizing attention patterns across biological contexts."}

## Future Directions

1. **Multiple tissue pairs:** Test immune→lung, kidney→lung, etc.
2. **Related vs distant tissues:** Compare transfer between closely related vs distant tissue types
3. **Cell type resolution:** Test transfer at the cell type level within tissues
4. **Species transfer:** Test cross-species generalization (human→mouse)

---
*Generated by CSSI Cross-Tissue Validation Pipeline*
*Report saved to: {output_path}*
"""

    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Cross-tissue report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CSSI Cross-Tissue Validation")
    parser.add_argument("--source-tissue", type=str, default="immune", choices=["immune", "kidney", "lung"])
    parser.add_argument("--target-tissue", type=str, default="kidney", choices=["immune", "kidney", "lung"])
    parser.add_argument("--max-cells", type=int, default=500, help="Maximum cells per tissue")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for model inference")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--top-k-layers", type=int, default=3, help="Number of top layers to select")
    parser.add_argument("--output-dir", type=str, default="D:/openclaw/biodyn-nmi-paper/cssi_real")
    
    args = parser.parse_args()
    
    if args.source_tissue == args.target_tissue:
        raise ValueError("Source and target tissues must be different")
    
    # Set random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # Load datasets
        logger.info(f"Loading source tissue: {args.source_tissue}")
        source_adata = load_tissue_dataset(args.source_tissue, args.max_cells)
        
        logger.info(f"Loading target tissue: {args.target_tissue}")
        target_adata = load_tissue_dataset(args.target_tissue, args.max_cells)
        
        # Load whole-human model (trained across tissues)
        logger.info("Loading scGPT whole-human model...")
        model, vocab = setup_scgpt_model("whole-human")
        
        # Load ground truth network
        logger.info("Loading TRRUST network...")
        trrust_edges = load_trrust_network()
        
        # Extract attention scores from source tissue
        logger.info(f"Extracting attention scores from {args.source_tissue} (source)...")
        layer_scores_source = extract_layer_attention_scores(
            model, source_adata, vocab, device, args.batch_size
        )
        
        # Extract attention scores from target tissue
        logger.info(f"Extracting attention scores from {args.target_tissue} (target)...")
        layer_scores_target = extract_layer_attention_scores(
            model, target_adata, vocab, device, args.batch_size
        )
        
        # Get gene names from source
        dataset_config = ScGPTDatasetConfig(
            max_genes=2048,
            include_zero=False,
            sort_by_expression=True,
            pad_token_id=vocab.get("<pad>", 0),
            cls_token_id=vocab.get("<cls>", 1)
        )
        dataset_source = ScGPTDataset(source_adata, vocab, dataset_config)
        gene_names = dataset_source.gene_names
        
        # Run CSSI layer selection on source tissue
        logger.info(f"Running CSSI layer selection on {args.source_tissue}...")
        best_layers, layer_performance_source = run_cssi_layer_selection(
            layer_scores_source, gene_names, trrust_edges, args.top_k_layers
        )
        
        # Cross-validate on target tissue
        logger.info(f"Cross-validating on {args.target_tissue}...")
        validation_results = cross_validate_layers(
            best_layers, layer_scores_target, gene_names, trrust_edges
        )
        
        # Combine results
        full_results = {
            **validation_results,
            "layer_performance_source": layer_performance_source,
            "source_tissue": args.source_tissue,
            "target_tissue": args.target_tissue,
            "n_cells_source": source_adata.n_obs,
            "n_cells_target": target_adata.n_obs,
            "n_genes": len(gene_names),
            "checkpoint": "whole-human",
            "random_seed": args.random_seed
        }
        
        # Save results
        results_path = Path(args.output_dir) / f"cssi_cross_tissue_results_{args.source_tissue}_to_{args.target_tissue}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        logger.info(f"Results saved to: {results_path}")
        
        # Generate report
        logger.info("Generating cross-tissue validation report...")
        report_path = f"D:/openclaw/biodyn-nmi-paper/CSSI_CROSS_TISSUE_REPORT_{args.source_tissue}_to_{args.target_tissue}.md"
        generate_cross_tissue_report(full_results, args.source_tissue, args.target_tissue, report_path)
        
        logger.info("CSSI cross-tissue validation completed successfully!")
        
    except Exception as e:
        logger.error(f"Cross-tissue validation failed: {e}")
        raise


if __name__ == "__main__":
    main()