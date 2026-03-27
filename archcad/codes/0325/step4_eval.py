import os, torch, numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import f1_score
from config import *
from step2_model import PanCADNetV2
from step1_dataset import PanCADDataset
from utils import get_valid_files


def extract_pred_instances(pc, pm, threshold=0.5):
    probs = pc.softmax(-1)
    scores, labels = probs[:, :-1].max(-1)
    masks = (pm.sigmoid() > 0.5)
    instances = []
    for q in range(pc.shape[0]):
        if scores[q] < threshold: continue
        ents = masks[q].nonzero(as_tuple=True)[0].cpu().tolist()
        if ents:
            instances.append({'label': labels[q].item(),
                               'entities': ents, 'score': scores[q].item()})
    return instances


def extract_gt_instances(gl, gm):
    instances = []
    for m in range(gl.shape[0]):
        ents = gm[m].nonzero(as_tuple=True)[0].cpu().tolist()
        if ents:
            instances.append({'label': gl[m].item(), 'entities': ents})
    return instances


def compute_iou_log(pred_ents, gt_ents, el):
    inter = set(pred_ents) & set(gt_ents)
    union = set(pred_ents) | set(gt_ents)
    if not union: return 0.0
    iw = sum(np.log(1 + el.get(e, 0)) for e in inter)
    uw = sum(np.log(1 + el.get(e, 0)) for e in union)
    return iw / (uw + 1e-8)


def compute_pq(pred_inst, gt_inst, el, nc):
    pq, sq, rq = {}, {}, {}
    for cid in range(nc):
        pc = [p for p in pred_inst if p['label'] == cid]
        gc = [g for g in gt_inst  if g['label'] == cid]
        if not gc and not pc: continue
        TP, FP, FN, iou_sum = 0, 0, 0, 0.0
        matched = set()
        if cid in STUFF_CLASSES:
            if not pc or not gc: FP += len(pc); FN += len(gc)
            else:
                iou = compute_iou_log(pc[0]['entities'], gc[0]['entities'], el)
                if iou > 0.5: TP, iou_sum = 1, iou
                else: FP, FN = 1, 1
        else:
            for p in pc:
                bi, bg = 0.0, -1
                for gi, g in enumerate(gc):
                    iou = compute_iou_log(p['entities'], g['entities'], el)
                    if iou > bi: bi, bg = iou, gi
                if bi > 0.5 and bg not in matched:
                    TP += 1; iou_sum += bi; matched.add(bg)
                else: FP += 1
            FN = len(gc) - len(matched)
        r = TP / (TP + 0.5 * FP + 0.5 * FN + 1e-8)
        s = iou_sum / (TP + 1e-8) if TP > 0 else 0.0
        pq[cid], sq[cid], rq[cid] = r * s, s, r
    return pq, sq, rq


if __name__ == "__main__":
    print("=" * 70)
    print("PanCADNet v2 — PQ / SQ / RQ  +  F1 / wF1 (논문 동일 방식)")
    print("=" * 70)

    file_ids, j_dict, p_dict = get_valid_files()
    s2 = int(len(file_ids) * 0.8)
    test_ds = PanCADDataset(file_ids[s2:], j_dict, p_dict)
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=lambda b: b[0])

    model = PanCADNetV2().to(DEVICE)
    cp = os.path.join(MODEL_OUT, "best_pancadnet_v2.pt")
    model.load_state_dict(torch.load(cp, map_location=DEVICE))
    model.eval()
    print(f"Loaded: {cp}\n")

    cpq = defaultdict(float); csq = defaultdict(float)
    crq = defaultdict(float); cc  = defaultdict(int)
    all_pred, all_gt = [], []

    with torch.no_grad():
        for sample in tqdm(test_dl, desc="Eval"):
            if sample['num_primitives'] < 2: continue
            img  = sample['image'].unsqueeze(0).to(DEVICE)
            geo  = sample['geo_features'].to(DEVICE)
            s    = sample['starts'].to(DEVICE)
            e    = sample['ends'].to(DEVICE)
            eil  = [x.to(DEVICE) for x in sample['edge_index_list']]
            eal  = [x.to(DEVICE) for x in sample['edge_attr_list']]
            bidx = torch.zeros(sample['num_primitives'], dtype=torch.long, device=DEVICE)

            with torch.amp.autocast('cuda'):
                pc_list, pm_list, sem_list = model(img, s, e, geo, eil, eal, bidx)

            pc  = pc_list[0].float().cpu()
            pm  = pm_list[0].float().cpu()
            sem = sem_list[0].float().cpu()     # (N, C)

            # PQ / SQ / RQ
            pi = extract_pred_instances(pc, pm)
            gi = extract_gt_instances(sample['gt_labels'], sample['gt_masks'])
            el = {i: float(sample['geo_features'][i, 0].item() * SCALE_ORG)
                  for i in range(sample['num_primitives'])}
            p, sq_d, r = compute_pq(pi, gi, el, NUM_GNN_CLS)
            for cid in p:
                cpq[cid] += p[cid]; csq[cid] += sq_d[cid]
                crq[cid] += r[cid]; cc[cid]  += 1

            # F1 / wF1 — 논문과 동일: sem_logits argmax
            pred_cls = sem.argmax(-1).numpy()
            gt_cls   = sample['sem_labels'].numpy()
            all_pred.extend(pred_cls.tolist())
            all_gt.extend(gt_cls.tolist())

    # ── 집계 ─────────────────────────────────────────────────────
    def avg(ms, cls_list):
        v = [c for c in cls_list if cc[c] > 0]
        return sum(ms[c] / cc[c] for c in v) / len(v) * 100 if v else 0.0

    y_true  = np.array(all_gt);  y_pred = np.array(all_pred)
    labels  = list(range(NUM_GNN_CLS))
    f1_mac  = f1_score(y_true, y_pred, labels=labels, average='macro',    zero_division=0) * 100
    f1_wgt  = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0) * 100
    f1_cls  = f1_score(y_true, y_pred, labels=labels, average=None,       zero_division=0) * 100

    thing_f1 = np.mean([f1_cls[c] for c in THING_CLASSES if c < len(f1_cls)])
    stuff_f1 = np.mean([f1_cls[c] for c in STUFF_CLASSES if c < len(f1_cls)])

    # ── 출력 ─────────────────────────────────────────────────────
    print("=" * 70)
    print(f"{'':8s} {'PQ':>7} {'SQ':>7} {'RQ':>7} {'F1':>7} {'wF1':>7}")
    print("-" * 70)
    print(f"{'Total':8s}"
          f" {avg(cpq, range(NUM_GNN_CLS)):6.2f}%"
          f" {avg(csq, range(NUM_GNN_CLS)):6.2f}%"
          f" {avg(crq, range(NUM_GNN_CLS)):6.2f}%"
          f" {f1_mac:6.2f}% {f1_wgt:6.2f}%")
    print(f"{'Thing':8s}"
          f" {avg(cpq, THING_CLASSES):6.2f}%"
          f" {avg(csq, THING_CLASSES):6.2f}%"
          f" {avg(crq, THING_CLASSES):6.2f}%"
          f" {thing_f1:6.2f}%")
    print(f"{'Stuff':8s}"
          f" {avg(cpq, STUFF_CLASSES):6.2f}%"
          f" {avg(csq, STUFF_CLASSES):6.2f}%"
          f" {avg(crq, STUFF_CLASSES):6.2f}%"
          f" {stuff_f1:6.2f}%")
    print("=" * 70)

    print(f"\n{'Type':5s}  {'Class':20s} {'PQ':>6} {'SQ':>6} {'RQ':>6} {'F1':>6}  n")
    print("-" * 70)
    for cid in range(NUM_GNN_CLS):
        if cc[cid] == 0 and (cid >= len(f1_cls) or f1_cls[cid] == 0): continue
        t    = "Thing" if cid in THING_CLASSES else "Stuff"
        n    = int(cc[cid])
        pq_v = cpq[cid] / cc[cid] * 100 if cc[cid] > 0 else 0.0
        sq_v = csq[cid] / cc[cid] * 100 if cc[cid] > 0 else 0.0
        rq_v = crq[cid] / cc[cid] * 100 if cc[cid] > 0 else 0.0
        fv   = f1_cls[cid] if cid < len(f1_cls) else 0.0
        print(f"[{t:5s}] {CLASS_NAMES[cid]:20s}"
              f" {pq_v:5.1f}% {sq_v:5.1f}% {rq_v:5.1f}% {fv:5.1f}%  (n={n})")
