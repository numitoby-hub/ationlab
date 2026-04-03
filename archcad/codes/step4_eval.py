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


def match_instances(pred_inst, gt_inst, el, cid):
    """단일 클래스에 대해 TP/FP/FN/iou_sum을 글로벌 누적용으로 반환"""
    pc = [p for p in pred_inst if p['label'] == cid]
    gc = [g for g in gt_inst if g['label'] == cid]
    if not gc and not pc:
        return 0, 0, 0, 0.0

    TP, FP, FN, iou_sum = 0, 0, 0, 0.0
    matched = set()

    if cid in STUFF_CLASSES:
        if not pc or not gc:
            FP += len(pc); FN += len(gc)
        else:
            iou = compute_iou_log(pc[0]['entities'], gc[0]['entities'], el)
            if iou > 0.5:
                TP, iou_sum = 1, iou
            else:
                FP, FN = 1, 1
    else:
        for p in pc:
            bi, bg = 0.0, -1
            for gi, g in enumerate(gc):
                iou = compute_iou_log(p['entities'], g['entities'], el)
                if iou > bi: bi, bg = iou, gi
            if bi > 0.5 and bg not in matched:
                TP += 1; iou_sum += bi; matched.add(bg)
            else:
                FP += 1
        FN = len(gc) - len(matched)

    return TP, FP, FN, iou_sum


# keep for backward compat (step3 import)
def compute_pq(pred_inst, gt_inst, el, nc):
    pq, sq, rq = {}, {}, {}
    for cid in range(nc):
        tp, fp, fn, iou_s = match_instances(pred_inst, gt_inst, el, cid)
        if tp + fp + fn == 0: continue
        r = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-8)
        s = iou_s / (tp + 1e-8) if tp > 0 else 0.0
        pq[cid], sq[cid], rq[cid] = r * s, s, r
    return pq, sq, rq


if __name__ == "__main__":
    print("=" * 70)
    print("PanCADNet v2 — PQ / SQ / RQ  +  F1 / wF1 (글로벌 집계)")
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

    # ── 글로벌 누적 (DPSS 방식) ──
    tp_cls = np.zeros(NUM_GNN_CLS, dtype=np.float64)
    fp_cls = np.zeros(NUM_GNN_CLS, dtype=np.float64)
    fn_cls = np.zeros(NUM_GNN_CLS, dtype=np.float64)
    iou_cls = np.zeros(NUM_GNN_CLS, dtype=np.float64)

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
                pc_list, pm_list, sem_list, _ = model(img, s, e, geo, eil, eal, bidx)

            pc  = pc_list[0].float().cpu()
            pm  = pm_list[0].float().cpu()
            sem = sem_list[0].float().cpu()

            pi = extract_pred_instances(pc, pm)
            gi = extract_gt_instances(sample['gt_labels'], sample['gt_masks'])
            el = {i: float(sample['geo_features'][i, 0].item() * SCALE_ORG)
                  for i in range(sample['num_primitives'])}

            for cid in range(NUM_GNN_CLS):
                tp, fp, fn, iou_s = match_instances(pi, gi, el, cid)
                tp_cls[cid] += tp
                fp_cls[cid] += fp
                fn_cls[cid] += fn
                iou_cls[cid] += iou_s

            pred_cls = sem.argmax(-1).numpy()
            gt_cls   = sample['sem_labels'].numpy()
            all_pred.extend(pred_cls.tolist())
            all_gt.extend(gt_cls.tolist())

    # ── PQ/SQ/RQ 계산 (글로벌) ──
    RQ = tp_cls / (tp_cls + 0.5 * fp_cls + 0.5 * fn_cls + 1e-8)
    SQ = iou_cls / (tp_cls + 1e-8)
    SQ[tp_cls == 0] = 0.0
    PQ = RQ * SQ

    def cls_avg(arr, cls_list):
        valid = [c for c in cls_list if (tp_cls[c] + fp_cls[c] + fn_cls[c]) > 0]
        return np.mean(arr[valid]) * 100 if valid else 0.0

    y_true = np.array(all_gt); y_pred = np.array(all_pred)
    labels = list(range(NUM_GNN_CLS))
    f1_mac = f1_score(y_true, y_pred, labels=labels, average='macro',    zero_division=0) * 100
    f1_wgt = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0) * 100
    f1_cls = f1_score(y_true, y_pred, labels=labels, average=None,       zero_division=0) * 100

    thing_f1 = np.mean([f1_cls[c] for c in THING_CLASSES if c < len(f1_cls)])
    stuff_f1 = np.mean([f1_cls[c] for c in STUFF_CLASSES if c < len(f1_cls)])

    print("=" * 70)
    print(f"{'':8s} {'PQ':>7} {'SQ':>7} {'RQ':>7} {'F1':>7} {'wF1':>7}")
    print("-" * 70)
    print(f"{'Total':8s}"
          f" {cls_avg(PQ, range(NUM_GNN_CLS)):6.2f}%"
          f" {cls_avg(SQ, range(NUM_GNN_CLS)):6.2f}%"
          f" {cls_avg(RQ, range(NUM_GNN_CLS)):6.2f}%"
          f" {f1_mac:6.2f}% {f1_wgt:6.2f}%")
    print(f"{'Thing':8s}"
          f" {cls_avg(PQ, THING_CLASSES):6.2f}%"
          f" {cls_avg(SQ, THING_CLASSES):6.2f}%"
          f" {cls_avg(RQ, THING_CLASSES):6.2f}%"
          f" {thing_f1:6.2f}%")
    print(f"{'Stuff':8s}"
          f" {cls_avg(PQ, STUFF_CLASSES):6.2f}%"
          f" {cls_avg(SQ, STUFF_CLASSES):6.2f}%"
          f" {cls_avg(RQ, STUFF_CLASSES):6.2f}%"
          f" {stuff_f1:6.2f}%")
    print("=" * 70)

    print(f"\n{'Type':5s}  {'Class':20s} {'PQ':>6} {'SQ':>6} {'RQ':>6} {'F1':>6}  TP/FP/FN")
    print("-" * 70)
    for cid in range(NUM_GNN_CLS):
        has_data = (tp_cls[cid] + fp_cls[cid] + fn_cls[cid]) > 0
        has_f1 = cid < len(f1_cls) and f1_cls[cid] > 0
        if not has_data and not has_f1: continue
        t  = "Thing" if cid in THING_CLASSES else "Stuff"
        fv = f1_cls[cid] if cid < len(f1_cls) else 0.0
        print(f"[{t:5s}] {CLASS_NAMES[cid]:20s}"
              f" {PQ[cid]*100:5.1f}% {SQ[cid]*100:5.1f}% {RQ[cid]*100:5.1f}%"
              f" {fv:5.1f}%  ({int(tp_cls[cid])}/{int(fp_cls[cid])}/{int(fn_cls[cid])})")
