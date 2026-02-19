# Tier 1: Value-Weighted Edges + Biological Characterization of Attention

## Goal

Two parallel analyses that together reframe the paper from "attention doesn't work" to "we show what attention captures and how to extract genuine regulatory signal."

**IMPORTANT**: Do NOT modify any paper files (main_v3.tex, supplementary_v3.tex). All results go to files in this directory (`experiments/biological_impact/`). Results will be reviewed before any paper integration.

## Python Environment

```
~/anaconda3/envs/bio_mech_interp/bin/python
```

NumPy 1.26.4 (required by numba/scanpy). Use `sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)` for line-buffered output. Save incremental results to JSON after each major compute block.

---

## Analysis A: Value-Weighted Edge Extraction

### Motivation

The paper shows:
- Raw attention weights don't encode regulation (AUROC ≈ correlation)
- Ablating "regulatory" attention heads has no effect on perturbation prediction
- But ablating *random* heads DOES hurt (d=0.33, p<10⁻⁸)

This implies perturbation-predictive computation resides in the **value/FFN pathway**, not the attention pattern. Standard practice extracts `softmax(QK^T/√d)` — the attention pattern. We should instead extract **value-weighted attention**: `softmax(QK^T/√d) · V`, which captures what information is actually routed through each head.

### What to compute

1. **Extract value-weighted edge scores** from Geneformer V2-316M on K562 control cells:
   - For each layer L and head H, extract `A_vh = softmax(QK^T/√d) · V` (shape: n_genes × d_head)
   - Compute pairwise edge scores as cosine similarity between gene representations in this space: `edge(i,j) = cosine(A_vh[i,:], A_vh[j,:])`
   - Average across heads within each layer, then evaluate per-layer
   - Also try: L2 distance, dot product

2. **Evaluate against existing benchmarks**:
   - TRRUST AUROC (same protocol as existing analyses)
   - Perturbation-first AUROC on Replogle K562 CRISPRi (same protocol: 2,000 control cells, 2,000 HVG, LFC>0.5)
   - Compare to raw attention AUROC (0.704 at L13) and correlation AUROC (0.703)
   - Compare to trivial baselines (variance=0.881, mean=0.841, dropout=0.808)

3. **Incremental value test**: Do value-weighted edges add predictive value beyond gene-level features? (Same logistic regression + GroupKFold protocol as main paper)

### Data locations

- Geneformer V2-316M: `ctheodoris/Geneformer` on HuggingFace, subfolder `Geneformer-V2-316M`
- Replogle K562 CRISPRi: `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/replogle/K562_gwps_normalized_bulk_01.h5ad`
- Existing extraction infrastructure: `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/round9_unified_extraction/`
  - `correlation_matrix.npy` (2000×2000 Spearman correlation)
  - `hvg_gene_list.json` (2000 HVG names)
  - `de_labels.pkl` (DE labels per perturbation)
  - `gene_features.csv` (mean_expr, variance, dropout_rate)
  - `layer_mean_edges/layer_NN_edges.pkl` (raw attention edges per layer)
- TRRUST: `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv`

### Implementation notes

- Use hooks (not `output_attentions=True`) to extract value-weighted representations — `output_attentions` uses ~10.8 GB/cell for all 18 layers
- Register forward hooks on each `BertSelfAttention` module to capture Q, K, V, and attention weights
- Batch size=1 on MPS (Apple Silicon); expect ~1.3-1.5 cells/s
- For 2,000 cells × 18 layers × 18 heads, this is ~25-40 minutes
- Save per-layer value-weighted edge matrices to `.npy` files for reuse
- Use `attn_implementation="eager"` (required for hook access)

### Output files

Save all results to `experiments/biological_impact/`:
- `value_weighted_edges/layer_NN_vw_edges.npy` — per-layer value-weighted edge matrices
- `value_weighted_results.json` — all AUROC comparisons, statistical tests
- `value_weighted_summary.md` — human-readable summary of findings

---

## Analysis B: Biological Characterization of Attention Patterns

### Motivation

The paper says "attention captures co-expression, not regulation." But co-expression is vague. If attention patterns preferentially capture specific biological relationships (pathway co-membership, protein-protein interactions, functional modules), this is a positive biological insight that reframes the narrative.

### What to compute

1. **Prepare reference edge sets** from public databases:
   - **TRRUST** (regulatory): already available, ~8,427 TF-target pairs
   - **STRING** (protein-protein interactions): download human PPI network, filter to high-confidence (score ≥ 700), extract gene pairs
   - **Reactome** (pathway co-membership): for each pathway, all gene pairs within that pathway are "co-pathway" edges
   - **KEGG** (pathway co-membership): same approach
   - **GO Biological Process** (functional co-annotation): gene pairs sharing specific GO BP terms (exclude very broad terms >500 genes)
   - **ENCODE TF-target** (ChIP-seq validated regulation): if available, provides an orthogonal regulatory reference

2. **Compute AUROC of attention edges against each reference**:
   - Use existing Geneformer attention edges from `round9_unified_extraction/layer_mean_edges/`
   - For each layer (L1-L18), compute AUROC against each reference edge set
   - Also compute AUROC of Spearman correlation edges against each reference
   - Restrict to genes present in both the HVG list and the reference database

3. **Statistical comparison**:
   - For each reference: is attention AUROC significantly different from correlation AUROC? (paired Wilcoxon)
   - Which reference does attention predict *best*? (ranking across references)
   - Does the attention advantage/disadvantage vary by layer? (layer profiles per reference)
   - Compute partial correlations: after controlling for expression similarity, does attention still predict STRING/Reactome/GO edges?

4. **Enrichment analysis**:
   - Take the top-1000 attention edges (by weight) and bottom-1000
   - Test enrichment in each reference set (Fisher's exact test)
   - Are high-attention gene pairs enriched for PPIs? Pathway co-membership? Regulatory edges?

### Data sources

- STRING: download from https://string-db.org/cgi/download — file `9606.protein.links.v12.0.txt.gz` (human). Map Ensembl protein IDs to gene symbols via STRING's `9606.protein.info.v12.0.txt.gz`
- Reactome: download from https://reactome.org/download-data — `ReactomePathways.gmt` or use the `Homo_sapiens.gene_association.reactome` file
- KEGG: use `bioservices` or `gseapy` Python package to get human pathway gene sets
- GO: use `goatools` or download from Gene Ontology Consortium, filter to Biological Process
- Existing attention edges: `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/experiments/round9_unified_extraction/layer_mean_edges/`

### Implementation notes

- Some databases are large; restrict to genes in the HVG universe (2,000 genes) to keep evaluation tractable
- For pathway co-membership, the number of co-pathway pairs can be very large — sample or use specific pathways of interest
- STRING has a numerical score (0-1000); use ≥700 as "high confidence" and also test ≥900 for "highest confidence"
- GO terms: exclude terms with >500 annotated genes (too broad) or <5 (too specific)
- Save intermediate reference edge sets as JSON for reproducibility

### Output files

Save all results to `experiments/biological_impact/`:
- `reference_edge_sets/` — processed reference databases (STRING, Reactome, KEGG, GO)
- `attention_characterization_results.json` — all AUROC values, statistical tests
- `attention_characterization_summary.md` — human-readable summary with key findings

---

## Success Criteria

**Analysis A is successful if**: Value-weighted edges show significantly higher AUROC than raw attention for either TRRUST or perturbation-first evaluation, OR provide incremental value beyond gene-level features. Even a null result is informative (narrows where regulatory signal lives).

**Analysis B is successful if**: Attention edges predict at least one non-regulatory reference (STRING, Reactome, GO) significantly better than regulatory references (TRRUST), providing a positive characterization of what attention encodes. Any clear biological pattern is valuable.

## Timeline

Both analyses can run in parallel. Expected: 5-7 days total.
- Analysis A: 3-5 days (extraction + evaluation)
- Analysis B: 3-5 days (database prep + evaluation)
