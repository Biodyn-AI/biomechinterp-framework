#!/usr/bin/env python3
"""
Basic test to see what parts of Geneformer we can import without tdigest
"""

import sys
import os

# Add geneformer path
sys.path.insert(0, 'D:/openclaw/intelligence-augmentation/models/Geneformer')

print("Testing Geneformer imports...")

try:
    import torch
    print(f"OK PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
except ImportError as e:
    print(f"FAIL PyTorch: {e}")

try:
    import transformers
    print(f"OK Transformers {transformers.__version__}")
except ImportError as e:
    print(f"FAIL Transformers: {e}")

try:
    from geneformer.modeling_geneformer import GeneformerForSequenceClassification
    print("OK Geneformer modeling")
except ImportError as e:
    print(f"FAIL Geneformer modeling: {e}")

try:
    from geneformer.tokenizer import TranscriptomeTokenizer
    print("OK Geneformer tokenizer")
except ImportError as e:
    print(f"FAIL Geneformer tokenizer: {e}")

try:
    # This might fail due to TDigest dependency
    from geneformer.emb_extractor import EmbExtractor
    print("OK Geneformer emb_extractor")
except ImportError as e:
    print(f"FAIL Geneformer emb_extractor: {e}")

print("Testing complete!")