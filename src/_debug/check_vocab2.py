import pickle, os, sys
sys.stdout = open('/tmp/vocab_check.txt', 'w')
sys.stderr = sys.stdout
import geneformer
GF_DIR = os.path.dirname(geneformer.__file__)
for name in ['token_dictionary.pkl', 'token_dictionary_gc104M.pkl']:
    path = os.path.join(GF_DIR, name)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        vals = [int(v) for v in d.values()]
        print(f"{name}: {len(d)} entries, max_id={max(vals)}, min_id={min(vals)}")
path_30m = os.path.join(GF_DIR, 'gene_dictionaries_30m', 'token_dictionary_gc30M.pkl')
if os.path.exists(path_30m):
    with open(path_30m, 'rb') as f:
        d30 = pickle.load(f)
    vals = [int(v) for v in d30.values()]
    print(f"30M: {len(d30)} entries, max_id={max(vals)}, min_id={min(vals)}")
