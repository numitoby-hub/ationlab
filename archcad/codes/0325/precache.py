import os, torch
from tqdm import tqdm
from config import *
from utils import get_valid_files
from step1_dataset import PanCADDataset

os.makedirs("/content/graph_cache_v2", exist_ok=True)
file_ids, j_dict, p_dict = get_valid_files()

ds = PanCADDataset(file_ids, j_dict, p_dict)
for i in tqdm(range(len(ds)), desc="Caching"):
    fid = file_ids[i]
    out_path = f"/content/graph_cache_v2/{fid}.pt"
    if os.path.exists(out_path): continue
    sample = ds[i]
    torch.save(sample, out_path)

print(f"Done: {len(file_ids)} cached")
