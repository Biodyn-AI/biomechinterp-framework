from transformers import BertModel
model = BertModel.from_pretrained("ctheodoris/Geneformer", trust_remote_code=True, output_attentions=True)
print(f"Layers: {model.config.num_hidden_layers}, Heads: {model.config.num_attention_heads}")
print(f"Hidden: {model.config.hidden_size}, Vocab: {model.config.vocab_size}")
