#!/usr/bin/env python3

# CRITICAL: scGPT Setup - torchtext shim MUST be loaded first
exec(open("/mnt/c/Users/Agent/.openclaw/workspace/patch_torchtext.py").read())

import json
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path("/mnt/d/openclaw/mechinterp-bio/biodyn-work/single_cell_mechinterp")
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.vocab import load_vocab

print('Testing vocab loading...')
vocab_path = Path('/mnt/d/openclaw/mechinterp-bio/biodyn-work/single_cell_mechinterp/external/scGPT_checkpoints/brain/vocab.json')
print(f'Vocab path exists: {vocab_path.exists()}')

if vocab_path.exists():
    print('Loading vocab with load_vocab function...')
    try:
        vocab = load_vocab(vocab_path)
        print(f'Vocab loaded successfully: {type(vocab)}')
        if hasattr(vocab, 'gene_to_id'):
            print(f'Vocab has gene_to_id with {len(vocab.gene_to_id)} entries')
        if hasattr(vocab, '__getitem__'):
            try:
                pad_idx = vocab['<pad>']
                print(f'Pad token index: {pad_idx}')
            except Exception as e:
                print(f'Could not access pad token: {e}')
                
        # Try different methods to access pad token
        if hasattr(vocab, 'get_id'):
            try:
                pad_idx = vocab.get_id('<pad>')
                print(f'Pad token index via get_id: {pad_idx}')
            except Exception as e:
                print(f'get_id method failed: {e}')
                
        if hasattr(vocab, 'gene_to_id'):
            try:
                pad_idx = vocab.gene_to_id.get('<pad>')
                print(f'Pad token index via gene_to_id: {pad_idx}')
            except Exception as e:
                print(f'gene_to_id access failed: {e}')
    except Exception as e:
        print(f'Error loading vocab: {e}')
        import traceback
        traceback.print_exc()
        
    print('\nAlso testing direct JSON loading...')
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    print(f'Direct JSON load successful, type: {type(vocab_data)}')
    print(f'Keys: {list(vocab_data.keys())[:5]}...')
else:
    print('Vocab file does not exist!')