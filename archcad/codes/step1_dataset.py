import os, json, cv2, numpy as np, torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
from collections import defaultdict
from config import *
from utils import get_class_id

class PanCADDataset(Dataset):
    def __init__(self, file_ids, j_dict, p_dict, img_size=IMG_SIZE):
        self.file_ids = file_ids
        self.j_dict = j_dict
        self.p_dict = p_dict
        self.img_size = img_size
    def __len__(self): return len(self.file_ids)
    def __getitem__(self, idx):
        fid = self.file_ids[idx]
        img = cv2.imdecode(np.fromfile(self.p_dict[fid], np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        with open(self.j_dict[fid], 'r', encoding='utf-8') as f:
            data = json.load(f)
        entities_list = data.get('entities', []) if isinstance(data, dict) else data
        entities = [e for e in entities_list if e.get('type') == 'LINE']
        N = len(entities)
        if N < 2: return self._empty(img)
        sem_labels = [get_class_id(e.get('semantic', 0)) for e in entities]
        inst_strings = [e.get('instance', f'none_{i}') for i, e in enumerate(entities)]
        starts = np.array([e['start'] for e in entities])
        ends = np.array([e['end'] for e in entities])
        centers = (starts + ends) / 2.0
        diffs = ends - starts
        lengths = np.linalg.norm(diffs, axis=1, keepdims=True) / SCALE_ORG
        angles = np.arctan2(diffs[:,1], diffs[:,0]).reshape(-1,1) / np.pi
        cx = centers[:,0:1] / SCALE_ORG
        cy = centers[:,1:2] / SCALE_ORG
        dx = diffs[:,0:1] / SCALE_ORG
        dy = diffs[:,1:2] / SCALE_ORG
        tree = cKDTree(centers)
        d, _ = tree.query(centers, k=min(7, N))
        avg_d = np.mean(d[:,1:], axis=1, keepdims=True) / SCALE_ORG
        density = np.sum(d[:,1:] < 50, axis=1, keepdims=True) / 10.0
        geo = np.hstack([lengths, angles, cx, cy, dx, dy, avg_d, density])
        eil, eal = self._build_graphs(starts, ends, centers, angles, N)
        gt_labels, gt_masks = self._build_gt(sem_labels, inst_strings, N)
        return {'image': img, 'geo_features': torch.tensor(geo, dtype=torch.float32),
                'edge_index_list': eil, 'edge_attr_list': eal,
                'starts': torch.tensor(starts, dtype=torch.float32),
                'ends': torch.tensor(ends, dtype=torch.float32),
                'gt_labels': gt_labels, 'gt_masks': gt_masks,
                'sem_labels': torch.tensor(sem_labels, dtype=torch.long), 'num_primitives': N}

    def _build_graphs(self, starts, ends, centers, angles, N):
        tree = cKDTree(centers)
        max_k = max(GRAPH_K_LIST)
        _, indices_all = tree.query(centers, k=min(max_k+1, N))
        eil, eal = [], []
        for K in GRAPH_K_LIST:
            ak = min(K, N-1)
            knn = indices_all[:, 1:ak+1]
            rows = np.repeat(np.arange(N), ak)
            cols = knn.flatten()
            pairs = np.vstack([rows, cols]).T
            _, ui = np.unique(pairs, axis=0, return_index=True)
            r, c = rows[ui], cols[ui]
            cd = np.linalg.norm(centers[r]-centers[c], axis=1) / SCALE_ORG
            dss = np.linalg.norm(starts[r]-starts[c], axis=1)
            dse = np.linalg.norm(starts[r]-ends[c], axis=1)
            des = np.linalg.norm(ends[r]-starts[c], axis=1)
            dee = np.linalg.norm(ends[r]-ends[c], axis=1)
            med = np.min(np.vstack([dss,dse,des,dee]), axis=0) / SCALE_ORG
            ad = np.abs(angles[r].flatten()-angles[c].flatten())
            ad = np.minimum(ad, 2.0-ad)
            par = (ad < 0.1).astype(np.float32)
            eil.append(torch.tensor(np.vstack([r,c]), dtype=torch.long))
            eal.append(torch.tensor(np.vstack([cd,med,ad,par]).T, dtype=torch.float32))
        return eil, eal

    def _build_gt(self, sem_labels, inst_strings, N):
        instances = []
        for sc in STUFF_CLASSES:
            nodes = [i for i in range(N) if sem_labels[i] == sc]
            if nodes: instances.append((sc, nodes))
        tg = defaultdict(list)
        for i in range(N):
            if sem_labels[i] in THING_CLASSES:
                tg[(sem_labels[i], inst_strings[i])].append(i)
        for (c, _), nodes in tg.items():
            instances.append((c, nodes))
        if not instances:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, N, dtype=torch.float32)
        gl = torch.tensor([c for c, _ in instances], dtype=torch.long)
        gm = torch.zeros(len(instances), N, dtype=torch.float32)
        for m, (_, nodes) in enumerate(instances):
            gm[m, nodes] = 1.0
        return gl, gm

    def _empty(self, img):
        ee = torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(0, EDGE_FEAT_DIM)
        return {'image': img, 'geo_features': torch.zeros(1, GEO_FEAT_DIM),
                'edge_index_list': [ee]*len(GRAPH_K_LIST), 'edge_attr_list': [ea]*len(GRAPH_K_LIST),
                'starts': torch.zeros(1,2), 'ends': torch.zeros(1,2),
                'gt_labels': torch.zeros(0, dtype=torch.long), 'gt_masks': torch.zeros(0,1),
                'sem_labels': torch.zeros(1, dtype=torch.long), 'num_primitives': 1}
