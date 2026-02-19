import scanpy as sc
adata = sc.read_h5ad("/mnt/d/openclaw/intelligence-augmentation/data/brain_scrna/DLPFC_11k.h5ad")
print(adata.shape)
print(adata.obs["cell_type"].value_counts())
print(list(adata.var.columns))
print(list(adata.var_names[:5]))
