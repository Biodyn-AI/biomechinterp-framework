import pickle, os, json, sys
sys.stdout = open("/mnt/d/openclaw/biodyn-nmi-paper/cssi_real/check3_out.txt", "w")
sys.stderr = sys.stdout
import geneformer
gf_dir = os.path.dirname(geneformer.__file__)
print("Geneformer dir:", gf_dir, flush=True)
print("Files:", os.listdir(gf_dir), flush=True)
for root, dirs, files in os.walk(gf_dir):
    for f in files:
        if f.endswith(('.pkl', '.json', '.dict')):
            print(f"Found: {os.path.join(root, f)}", flush=True)
token_dict_path = os.path.join(gf_dir, "token_dictionary.pkl")
if os.path.exists(token_dict_path):
    with open(token_dict_path, "rb") as f:
        token_dict = pickle.load(f)
    print(f"Token dict: {type(token_dict)}, {len(token_dict)} entries", flush=True)
    items = list(token_dict.items())[:5]
    print("Sample:", items, flush=True)
else:
    print("No token_dictionary.pkl found", flush=True)
