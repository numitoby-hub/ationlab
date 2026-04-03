import os, torch, numpy as np, random
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader, Subset
from config import *
from step2_model import PanCADNetV2, PanopticLoss
from step4_eval import extract_pred_instances, extract_gt_instances, compute_iou_log, match_instances
from utils import get_valid_files

TOTAL_EPOCHS      = 30
BATCH_SIZE        = 8
SAMPLES_PER_EPOCH = 20000


# ── Dataset ──────────────────────────────────────────────────────
class CachedDataset(Dataset):
    def __init__(self, file_ids, cache_dir=GRAPH_DIR):
        self.paths = [os.path.join(cache_dir, f"{fid}.pt") for fid in file_ids
                      if os.path.exists(os.path.join(cache_dir, f"{fid}.pt"))]
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return torch.load(self.paths[idx], weights_only=False)


MAX_PRIMITIVES = 3000

def collate_batch(samples):
    samples = [s for s in samples if 2 <= s['num_primitives'] <= MAX_PRIMITIVES]
    if not samples: return None
    images    = torch.stack([s['image'] for s in samples])
    Ns        = [s['num_primitives'] for s in samples]
    batch_idx = torch.cat([torch.full((n,), b, dtype=torch.long) for b, n in enumerate(Ns)])
    geo       = torch.cat([s['geo_features'] for s in samples])
    starts    = torch.cat([s['starts'] for s in samples])
    ends      = torch.cat([s['ends'] for s in samples])
    sem_labels = torch.cat([s['sem_labels'] for s in samples])   # (N_total,)
    num_scales = len(samples[0]['edge_index_list'])
    eil_batched, eal_batched = [], []
    for k in range(num_scales):
        offset, edges_k, attrs_k = 0, [], []
        for s in samples:
            edges_k.append(s['edge_index_list'][k] + offset)
            attrs_k.append(s['edge_attr_list'][k])
            offset += s['num_primitives']
        eil_batched.append(torch.cat(edges_k, dim=1))
        eal_batched.append(torch.cat(attrs_k, dim=0))
    return {
        'image':           images,
        'geo_features':    geo,
        'starts':          starts,
        'ends':            ends,
        'edge_index_list': eil_batched,
        'edge_attr_list':  eal_batched,
        'batch_idx':       batch_idx,
        'gt_labels':       [s['gt_labels'] for s in samples],
        'gt_masks':        [s['gt_masks']  for s in samples],
        'sem_labels':      sem_labels,
        'num_primitives':  Ns,
    }


def make_loader(dataset, n_samples, batch_size, shuffle=True):
    n = min(n_samples, len(dataset))
    indices = random.sample(range(len(dataset)), n)
    return DataLoader(Subset(dataset, indices),
                      batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, collate_fn=collate_batch, pin_memory=True)


# ── Train ────────────────────────────────────────────────────────
def train_one_epoch(model, dl, criterion, optimizer, epoch):
    model.train()
    total_loss, steps = 0, 0
    d = {'loss_cls': 0, 'loss_bce': 0, 'loss_dice': 0, 'loss_sem': 0, 'loss_overlap': 0, 'loss_aux': 0}
    for batch in tqdm(dl, desc=f"Train {epoch+1}/{TOTAL_EPOCHS}", leave=False):
        if batch is None: continue
        img     = batch['image'].to(DEVICE)
        geo     = batch['geo_features'].to(DEVICE)
        s       = batch['starts'].to(DEVICE)
        e       = batch['ends'].to(DEVICE)
        eil     = [x.to(DEVICE) for x in batch['edge_index_list']]
        eal     = [x.to(DEVICE) for x in batch['edge_attr_list']]
        bidx    = batch['batch_idx'].to(DEVICE)
        sem_all = batch['sem_labels'].to(DEVICE)   # (N_total,)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pc_list, pm_list, sem_list, aux_list = model(img, s, e, geo, eil, eal, bidx)
            loss   = torch.tensor(0., device=DEVICE)
            valid  = 0
            offset = 0
            for pc, pm, sem, aux, gl, gm, N in zip(
                    pc_list, pm_list, sem_list, aux_list,
                    batch['gt_labels'], batch['gt_masks'], batch['num_primitives']):
                gl, gm  = gl.to(DEVICE), gm.to(DEVICE)
                sem_lbl = sem_all[offset:offset + N]
                offset += N
                l, ld   = criterion(pc, pm, sem, gl, gm, sem_lbl, aux_outputs=aux)
                loss    = loss + l
                for k in d: d[k] += ld.get(k, 0)
                valid  += 1
            if valid > 0: loss = loss / valid
        if not torch.isfinite(loss):
            optimizer.zero_grad(); continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item(); steps += 1
    if steps == 0: return 0, {}
    return total_loss / steps, {k: v / steps for k, v in d.items()}


# ── Validate: loss + F1(논문 방식) ───────────────────────────────
@torch.no_grad()
def validate(model, dl, criterion):
    model.eval()
    total_loss, steps = 0, 0
    all_pred, all_gt = [], []

    for batch in tqdm(dl, desc="  Val", leave=False):
        if batch is None: continue
        img     = batch['image'].to(DEVICE)
        geo     = batch['geo_features'].to(DEVICE)
        s       = batch['starts'].to(DEVICE)
        e       = batch['ends'].to(DEVICE)
        eil     = [x.to(DEVICE) for x in batch['edge_index_list']]
        eal     = [x.to(DEVICE) for x in batch['edge_attr_list']]
        bidx    = batch['batch_idx'].to(DEVICE)
        sem_all = batch['sem_labels'].to(DEVICE)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pc_list, pm_list, sem_list, aux_list = model(img, s, e, geo, eil, eal, bidx)
            loss   = torch.tensor(0., device=DEVICE)
            valid  = 0
            offset = 0
            for pc, pm, sem, aux, gl, gm, N in zip(
                    pc_list, pm_list, sem_list, aux_list,
                    batch['gt_labels'], batch['gt_masks'], batch['num_primitives']):
                gl_d, gm_d = gl.to(DEVICE), gm.to(DEVICE)
                sem_lbl    = sem_all[offset:offset + N]
                offset    += N
                l, _       = criterion(pc, pm, sem, gl_d, gm_d, sem_lbl, aux_outputs=aux)
                loss = loss + l; valid += 1

                # ── 논문 방식: sem_logits argmax → wF1 ──────────
                pred_cls = sem.float().argmax(-1).cpu().numpy()
                gt_cls   = sem_lbl.cpu().numpy()
                all_pred.extend(pred_cls.tolist())
                all_gt.extend(gt_cls.tolist())
            if valid > 0: loss = loss / valid
        total_loss += loss.item(); steps += 1

    val_loss = total_loss / max(1, steps)
    y_true   = np.array(all_gt);  y_pred = np.array(all_pred)
    labels   = list(range(NUM_GNN_CLS))
    f1_mac   = f1_score(y_true, y_pred, labels=labels, average='macro',    zero_division=0)
    f1_wgt   = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    return val_loss, f1_mac, f1_wgt


# ── 최종 PQ 평가 (학습 끝난 후) ──────────────────────────────────
@torch.no_grad()
def evaluate_pq(model, ds):
    """CachedDataset 전체를 sample 단위로 순회해 PQ/SQ/RQ 계산"""
    model.eval()
    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    num_workers=2, collate_fn=lambda b: b[0])

    tp_cls = np.zeros(NUM_GNN_CLS, dtype=np.float64)
    fp_cls = np.zeros(NUM_GNN_CLS, dtype=np.float64)
    fn_cls = np.zeros(NUM_GNN_CLS, dtype=np.float64)
    iou_cls = np.zeros(NUM_GNN_CLS, dtype=np.float64)

    for sample in tqdm(dl, desc="  PQ eval", leave=False):
        if sample['num_primitives'] < 2: continue
        img  = sample['image'].unsqueeze(0).to(DEVICE)
        geo  = sample['geo_features'].to(DEVICE)
        s    = sample['starts'].to(DEVICE)
        e    = sample['ends'].to(DEVICE)
        eil  = [x.to(DEVICE) for x in sample['edge_index_list']]
        eal  = [x.to(DEVICE) for x in sample['edge_attr_list']]
        bidx = torch.zeros(sample['num_primitives'], dtype=torch.long, device=DEVICE)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pc_list, pm_list, _, _ = model(img, s, e, geo, eil, eal, bidx)
        pc = pc_list[0].float().cpu()
        pm = pm_list[0].float().cpu()

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

    RQ = tp_cls / (tp_cls + 0.5 * fp_cls + 0.5 * fn_cls + 1e-8)
    SQ = iou_cls / (tp_cls + 1e-8)
    SQ[tp_cls == 0] = 0.0
    PQ = RQ * SQ

    def cls_avg(arr, cls_list):
        valid = [c for c in cls_list if (tp_cls[c] + fp_cls[c] + fn_cls[c]) > 0]
        return np.mean(arr[valid]) * 100 if valid else 0.0

    print("\n" + "=" * 60)
    print("Val PQ Results (global)")
    print(f"{'':8s} {'PQ':>7} {'SQ':>7} {'RQ':>7}")
    print("-" * 40)
    print(f"{'Total':8s} {cls_avg(PQ, range(NUM_GNN_CLS)):6.2f}%"
          f" {cls_avg(SQ, range(NUM_GNN_CLS)):6.2f}%"
          f" {cls_avg(RQ, range(NUM_GNN_CLS)):6.2f}%")
    print(f"{'Thing':8s} {cls_avg(PQ, THING_CLASSES):6.2f}%"
          f" {cls_avg(SQ, THING_CLASSES):6.2f}%"
          f" {cls_avg(RQ, THING_CLASSES):6.2f}%")
    print(f"{'Stuff':8s} {cls_avg(PQ, STUFF_CLASSES):6.2f}%"
          f" {cls_avg(SQ, STUFF_CLASSES):6.2f}%"
          f" {cls_avg(RQ, STUFF_CLASSES):6.2f}%")
    print("-" * 60)
    print(f"{'Type':5s}  {'Class':20s} {'PQ':>6} {'SQ':>6} {'RQ':>6}  TP/FP/FN")
    print("-" * 60)
    for cid in range(NUM_GNN_CLS):
        if (tp_cls[cid] + fp_cls[cid] + fn_cls[cid]) == 0: continue
        t = "Thing" if cid in THING_CLASSES else "Stuff"
        print(f"[{t:5s}] {CLASS_NAMES[cid]:20s}"
              f" {PQ[cid]*100:5.1f}% {SQ[cid]*100:5.1f}% {RQ[cid]*100:5.1f}%"
              f"  ({int(tp_cls[cid])}/{int(fp_cls[cid])}/{int(fn_cls[cid])})")
    print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"PanCADNet v2 | batch={BATCH_SIZE} | subset={SAMPLES_PER_EPOCH}/epoch")
    print("=" * 60)
    os.makedirs(MODEL_OUT, exist_ok=True)

    file_ids, _, _ = get_valid_files()
    s1 = int(len(file_ids) * 0.7); s2 = int(len(file_ids) * 0.8)
    train_ds = CachedDataset(file_ids[:s1])
    val_ds   = CachedDataset(file_ids[s1:s2])
    print(f"Train pool: {len(train_ds)} | Val: {len(val_ds)}")

    val_dl = make_loader(val_ds, len(val_ds), BATCH_SIZE, shuffle=False)

    model     = PanCADNetV2().to(DEVICE)
    criterion = PanopticLoss().to(DEVICE)

    bp = [p for n, p in model.named_parameters() if 'backbone' in n]
    op = [p for n, p in model.named_parameters() if 'backbone' not in n]
    BASE_LR = 1e-4  # 2e-4 → 1e-4 (안정성)
    optimizer = torch.optim.AdamW(
        [{'params': bp, 'lr': BASE_LR * 0.1}, {'params': op, 'lr': BASE_LR}],
        weight_decay=WEIGHT_DECAY)
    WARMUP_EPOCHS = 3
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS  # linear warmup
        progress = (epoch - WARMUP_EPOCHS) / (TOTAL_EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * progress))  # cosine decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # bf16 doesn't need GradScaler (same exponent range as fp32)

    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    best_val = float('inf')
    for epoch in range(TOTAL_EPOCHS):
        train_dl = make_loader(train_ds, SAMPLES_PER_EPOCH, BATCH_SIZE)
        tl, td   = train_one_epoch(model, train_dl, criterion, optimizer, epoch)
        vl, f1_mac, f1_wgt = validate(model, val_dl, criterion)
        scheduler.step()
        lr = optimizer.param_groups[1]['lr']

        print(f"Epoch {epoch+1:3d}/{TOTAL_EPOCHS} | "
              f"Train {tl:.4f} "
              f"(cls:{td.get('loss_cls',0):.3f} "
              f"bce:{td.get('loss_bce',0):.3f} "
              f"dice:{td.get('loss_dice',0):.3f} "
              f"sem:{td.get('loss_sem',0):.3f} "
              f"ovlp:{td.get('loss_overlap',0):.3f} "
              f"aux:{td.get('loss_aux',0):.3f}) | "
              f"Val {vl:.4f} | "
              f"F1 {f1_mac*100:.2f}% / wF1 {f1_wgt*100:.2f}% | "
              f"LR {lr:.2e}")

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(),
                       os.path.join(MODEL_OUT, "best_pancadnet_v2.pt"))
            print(f"  ★ Best! val={best_val:.4f}  "
                  f"F1={f1_mac*100:.2f}%  wF1={f1_wgt*100:.2f}%")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),
                       os.path.join(MODEL_OUT, f"ckpt_ep{epoch+1}.pt"))

    print(f"\nDone! Best val loss: {best_val:.4f}")

    # ── 학습 완료 후 best 모델로 val PQ 계산 ──────────────────────
    print("\nLoading best model for PQ evaluation...")
    model.load_state_dict(
        torch.load(os.path.join(MODEL_OUT, "best_pancadnet_v2.pt"), map_location=DEVICE))
    evaluate_pq(model, val_ds)
