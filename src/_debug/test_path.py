from pathlib import Path
p = Path("/mnt/d/openclaw/mechinterp-bio/biodyn-work/single_cell_mechinterp/external/scGPT_checkpoints/brain/vocab.json")
print(f"Path: {p}")
print(f"Exists: {p.exists()}")
print(f"Is absolute: {p.is_absolute()}")
