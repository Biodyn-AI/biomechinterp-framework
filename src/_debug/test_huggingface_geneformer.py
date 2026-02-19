#!/usr/bin/env python3
"""
Test loading Geneformer through HuggingFace transformers directly
"""

import sys
import os
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

print("Testing HuggingFace Geneformer access...")

# Test loading the local model files
model_path = "D:/openclaw/intelligence-augmentation/models/Geneformer"
print(f"Model path: {model_path}")

try:
    # Try to load config
    config = AutoConfig.from_pretrained(model_path)
    print(f"OK Config loaded: {config}")
    print(f"Model type: {config.model_type}")
    print(f"Vocab size: {config.vocab_size}")
    
    # Try to load model
    model = AutoModel.from_pretrained(model_path)
    print(f"OK Model loaded: {model.__class__.__name__}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("OK Model moved to CUDA")
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
except Exception as e:
    print(f"FAIL: {e}")

print("Testing complete!")