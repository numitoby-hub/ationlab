import os, glob, random, json, numpy as np, cv2
from collections import defaultdict
from config import *

def get_class_id(semantic_id):
    return RAW_TO_NEW_CLASS.get(semantic_id, 30)

def get_valid_files():
    json_paths = glob.glob(os.path.join(JSON_DIR, "*.json"))
    png_paths  = glob.glob(os.path.join(PNG_DIR, "*.png"))
    j_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in json_paths}
    p_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in png_paths}
    valid_ids = [fid for fid in list(set(j_dict.keys()) & set(p_dict.keys())) if os.path.getsize(j_dict[fid]) > 10]
    valid_ids.sort()
    drawing_groups = defaultdict(list)
    for fid in valid_ids:
        drawing_id = fid.rsplit('_', 1)[0] if '_' in fid else fid
        drawing_groups[drawing_id].append(fid)
    unique_drawings = sorted(drawing_groups.keys())
    random.seed(42)
    random.shuffle(unique_drawings)
    final_valid_ids = []
    for drawing_id in unique_drawings:
        final_valid_ids.extend(drawing_groups[drawing_id])
    return final_valid_ids, j_dict, p_dict
