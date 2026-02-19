import pickle, os, json

# Find Geneformer's gene mapping files
import geneformer
gf_dir = os.path.dirname(geneformer.__file__)
print("Geneformer dir:", gf_dir)
print("Files:", os.listdir(gf_dir))

# Check for token dictionary
for root, dirs, files in os.walk(gf_dir):
    for f in files:
        if f.endswith(('.pkl', '.json', '.dict')):
            fpath = os.path.join(root, f)
            print(f"Found: {fpath} ({os.path.getsize(fpath)} bytes)")

# Try loading token dictionary
token_dict_path = os.path.join(gf_dir, "token_dictionary.pkl")
if os.path.exists(token_dict_path):
    with open(token_dict_path, "rb") as f:
        token_dict = pickle.load(f)
    print(f"\nToken dict: {type(token_dict)}, {len(token_dict)} entries")
    # Show a few
    items = list(token_dict.items())[:5]
    print("Sample:", items)
