import pickle, os, sys
sys.stdout = open('/tmp/debug_map.txt', 'w')
import geneformer
GF_DIR = os.path.dirname(geneformer.__file__)

with open(os.path.join(GF_DIR, "token_dictionary_gc104M.pkl"), "rb") as f:
    token_dict = pickle.load(f)
with open(os.path.join(GF_DIR, "gene_name_id_dict_gc104M.pkl"), "rb") as f:
    gene_name_id_dict = pickle.load(f)

id_to_token = {int(v): k for k, v in token_dict.items()}

# Show sample mappings
print("Token dict samples (gene->id):")
for i, (k,v) in enumerate(list(token_dict.items())[:10]):
    print(f"  {k} -> {v}")

print("\nReverse (id->gene):")
for i in range(5):
    print(f"  {i} -> {id_to_token.get(i, 'MISSING')}")

print("\nGene name dict samples:")
for i, (k,v) in enumerate(list(gene_name_id_dict.items())[:10]):
    print(f"  {k} -> {v}")

# Check if TRRUST TFs map correctly
# Common TFs: TP53, NFKB1, STAT3, MYC
for tf in ['TP53', 'NFKB1', 'STAT3', 'MYC', 'GATA1']:
    ens = gene_name_id_dict.get(tf, 'NOT_FOUND')
    if ens != 'NOT_FOUND':
        tok = token_dict.get(ens, 'NOT_IN_VOCAB')
        print(f"\n{tf} -> {ens} -> token {tok}")
        if tok != 'NOT_IN_VOCAB':
            reverse = id_to_token.get(int(tok), 'MISSING')
            print(f"  Reverse: token {tok} -> {reverse}")
    else:
        print(f"\n{tf} -> NOT_FOUND in gene_name_id_dict")

# Check: does token dict use Ensembl IDs as keys?
keys = list(token_dict.keys())[:5]
print(f"\nToken dict key format: {keys}")
# Check if they're ENSG or something else
ensg_count = sum(1 for k in token_dict if str(k).startswith('ENSG'))
print(f"ENSG keys: {ensg_count}/{len(token_dict)}")
