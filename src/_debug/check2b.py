import sys, traceback
try:
    from transformers import BertModel
    model = BertModel.from_pretrained("ctheodoris/Geneformer", trust_remote_code=True, output_attentions=True)
    print(f"Layers: {model.config.num_hidden_layers}, Heads: {model.config.num_attention_heads}")
except Exception as e:
    traceback.print_exc()
    print("\nTrying alternative paths...")
    import os
    # Check if cached
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache):
        print("Cached models:", os.listdir(cache)[:10])
    # Try geneformer package directly
    try:
        import geneformer
        print("Geneformer package at:", geneformer.__file__)
        print("Dir:", dir(geneformer))
    except:
        traceback.print_exc()
