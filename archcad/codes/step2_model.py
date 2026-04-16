import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from transformers import SegformerModel
from config import *


class SegFormerBackbone(nn.Module):
    def __init__(self, model_name=SEGFORMER_NAME, out_channels=256):
        super().__init__()
        self.encoder = SegformerModel.from_pretrained(model_name)
        encoder_channels = self.encoder.config.hidden_sizes
        self.proj_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, out_channels, 1), nn.BatchNorm2d(out_channels), nn.GELU())
            for c in encoder_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1), nn.BatchNorm2d(out_channels), nn.GELU())
        self.out_channels = out_channels

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        target_h, target_w = hidden_states[0].shape[2:]
        projected = []
        for feat, proj in zip(hidden_states, self.proj_layers):
            p = proj(feat)
            if p.shape[2:] != (target_h, target_w):
                p = F.interpolate(p, size=(target_h, target_w), mode='bilinear', align_corners=False)
            projected.append(p)
        return self.fuse(torch.cat(projected, dim=1))


class HybridLineSampling(nn.Module):
    def __init__(self, feat_channels=256, n_line_points=LINE_SAMPLE_N, n_groups=LINE_SAMPLE_M):
        super().__init__()
        self.n_points = n_line_points
        self.n_groups = n_groups
        self.points_per_group = n_line_points // n_groups
        n_tokens = n_groups + 9
        self.token_fuse = nn.Sequential(
            nn.Linear(feat_channels * n_tokens, feat_channels * 2), nn.GELU(),
            nn.Linear(feat_channels * 2, feat_channels), nn.LayerNorm(feat_channels))
        self.out_channels = feat_channels

    def forward(self, feature_map, starts, ends, batch_idx, img_size):
        B, C, H, W = feature_map.shape
        device = feature_map.device
        results = []
        for b in range(B):
            mask = (batch_idx == b)
            s_b, e_b = starts[mask], ends[mask]
            N = s_b.shape[0]
            if N == 0:
                results.append(torch.zeros(0, self.out_channels, device=device))
                continue
            fm_b = feature_map[b:b + 1]
            t = torch.linspace(0, 1, self.n_points, device=device).unsqueeze(0)
            line_points = s_b.unsqueeze(1) + t.unsqueeze(-1) * (e_b - s_b).unsqueeze(1)
            gx = (line_points[:, :, 0] / img_size) * 2 - 1
            gy = (line_points[:, :, 1] / img_size) * 2 - 1
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
            lf = F.grid_sample(fm_b, grid, mode='bilinear', padding_mode='border', align_corners=True)
            lf = lf[0].permute(1, 2, 0).view(N, self.n_groups, self.points_per_group, C)
            line_tokens = lf.max(dim=2).values
            centers = (s_b + e_b) / 2.0
            lengths = torch.norm(e_b - s_b, dim=1, keepdim=True).clamp(min=20.0)
            half = lengths / 2.0 / img_size
            offsets = torch.tensor(
                [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]],
                device=device, dtype=torch.float32)
            cn = centers / img_size * 2 - 1
            rp = cn.unsqueeze(1) + offsets.unsqueeze(0) * half.unsqueeze(-1)
            rf = F.grid_sample(fm_b, rp.unsqueeze(0), mode='bilinear', padding_mode='border', align_corners=True)
            region_tokens = rf[0].permute(1, 2, 0)
            all_tokens = torch.cat([line_tokens, region_tokens], dim=1)
            results.append(self.token_fuse(all_tokens.reshape(N, -1)))
        return torch.cat(results, dim=0)


class ProgressiveGATv2Head(nn.Module):
    def __init__(self, in_dim=GEO_FEAT_DIM, edge_dim=EDGE_FEAT_DIM, hidden=GAT_HIDDEN,
                 heads=GAT_HEADS, k_list=GRAPH_K_LIST, dropout=GAT_DROPOUT):
        super().__init__()
        num_layers = len(k_list)
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(num_layers):
            in_c = hidden if i == 0 else hidden * heads
            oh = 1 if i == num_layers - 1 else heads
            cat = i < num_layers - 1
            self.convs.append(GATv2Conv(in_c, hidden, heads=oh, concat=cat, dropout=dropout, edge_dim=edge_dim))
            oc = hidden * oh if cat else hidden
            self.norms.append(nn.LayerNorm(oc))
            self.skips.append(nn.Linear(in_c, oc))
        self.out_dim = hidden

    def forward(self, x, edge_index_list, edge_attr_list):
        x = self.input_proj(x)
        for i, (conv, norm, skip) in enumerate(zip(self.convs, self.norms, self.skips)):
            res = skip(x)
            x = conv(x, edge_index_list[i], edge_attr=edge_attr_list[i])
            x = F.gelu(norm(x)) + res
        return x


class AdaptiveFusion(nn.Module):
    def __init__(self, graph_dim, visual_dim, out_dim=DEC_HIDDEN):
        super().__init__()
        self.pg  = nn.Linear(graph_dim, out_dim)
        self.pv  = nn.Linear(visual_dim, out_dim)
        self.W2  = nn.Linear(out_dim, out_dim)
        self.W3  = nn.Linear(out_dim, out_dim)
        self.W1  = nn.Linear(out_dim, out_dim)
        self.out = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, xg, vi):
        g = self.pg(xg)
        v = self.pv(vi)
        w = torch.sigmoid(self.W1(torch.tanh(self.W2(g) + self.W3(v))))
        return self.out(torch.cat([g, w * v], dim=-1))


class MaskedTransformerDecoderLayer(nn.Module):
    """Mask2Former: masked cross-attention"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Linear(dim_feedforward, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, q, mem, attn_mask=None):
        q2, _ = self.self_attn(q, q, q)
        q = self.norm1(q + self.drop(q2))

        if attn_mask is not None:
            nhead = self.cross_attn.num_heads
            key_mask = attn_mask.unsqueeze(0).expand(nhead, -1, -1)
            key_mask = key_mask.float().masked_fill(key_mask, float('-inf'))
        else:
            key_mask = None

        q2, _ = self.cross_attn(q, mem, mem, attn_mask=key_mask)
        q = self.norm2(q + self.drop(q2))

        q = self.norm3(q + self.drop(self.ffn(q)))
        return q


class PrimitiveMask2FormerDecoder(nn.Module):
    def __init__(self, hidden_dim=DEC_HIDDEN, num_heads=DEC_HEADS, num_layers=DEC_LAYERS,
                 num_queries=NUM_QUERIES, num_classes=NUM_GNN_CLS):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.layers = nn.ModuleList([
            MaskedTransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=0.1)
            for _ in range(num_layers)])
        self.cls_head  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_classes + 1))
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.prim_proj = nn.Linear(hidden_dim, hidden_dim)
        self.sem_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_classes))

    def forward(self, fused):
        pe  = self.prim_proj(fused)
        mem = fused.unsqueeze(0)
        q   = self.query_embed.weight.unsqueeze(0)

        attn_mask = None
        aux_outputs = []

        for layer in self.layers:
            q = layer(q, mem, attn_mask=attn_mask)
            me = self.mask_head(q.squeeze(0))
            curr_mask = torch.mm(me, pe.T)
            aux_outputs.append((self.cls_head(q.squeeze(0)), curr_mask))
            attn_mask = (curr_mask.sigmoid() < 0.5)
            all_masked = attn_mask.all(dim=-1, keepdim=True)
            attn_mask = attn_mask & ~all_masked

        q_final = q.squeeze(0)
        cls   = self.cls_head(q_final)
        me    = self.mask_head(q_final)
        masks = torch.mm(me, pe.T)

        sem_logits = self.sem_head(fused)
        sem_logits = sem_logits.clamp(-30, 30)  # fp16 overflow 방지

        return cls, masks, sem_logits, aux_outputs


class PanopticLoss(nn.Module):
    def __init__(self, num_classes=NUM_GNN_CLS, lc=LAMBDA_CLS, lb=LAMBDA_BCE, ld=LAMBDA_DICE,
                 class_weights=None):
        super().__init__()
        self.nc = num_classes
        self.lc, self.lb, self.ld = lc, lb, ld
        if class_weights is not None:
            self.register_buffer('cw', class_weights)
        else:
            cw = torch.ones(num_classes + 1)
            cw[-1] = 0.1
            self.register_buffer('cw', cw)

    @torch.no_grad()
    def hungarian_match(self, pc, pm, gl, gm):
        from scipy.optimize import linear_sum_assignment
        Q, M = pc.shape[0], gl.shape[0]
        if M == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        pc_safe = pc.float()
        pm_safe = pm.float()
        if not (torch.isfinite(pc_safe).all() and torch.isfinite(pm_safe).all()):
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        pp    = pc_safe.softmax(-1).clamp(1e-6, 1 - 1e-6)
        cc    = -pp[:, gl]
        ps    = pm_safe.sigmoid().clamp(1e-6, 1 - 1e-6)
        inter = torch.mm(ps, gm.T)
        psum  = ps.sum(-1, keepdim=True)
        gsum  = gm.sum(-1, keepdim=True).T
        dc    = 1 - (2 * inter + 1) / (psum + gsum + 1)
        cost  = torch.nan_to_num(cc + 5.0 * dc, nan=1e4, posinf=1e4, neginf=-1e4)
        pi, gi = linear_sum_assignment(cost.cpu().numpy())
        return torch.tensor(pi, dtype=torch.long), torch.tensor(gi, dtype=torch.long)

    def _single_loss(self, pc, pm, gl, gm, dev):
        Q = pc.shape[0]
        pi, gi = self.hungarian_match(pc, pm, gl, gm)
        pi, gi = pi.to(dev), gi.to(dev)

        tgt = torch.full((Q,), self.nc, dtype=torch.long, device=dev)
        if len(pi) > 0:
            tgt[pi] = gl[gi]
        l_cls = F.cross_entropy(pc, tgt, weight=self.cw.to(dev))

        if len(pi) > 0:
            mp, mg = pm[pi], gm[gi].to(dev)
            l_bce  = F.binary_cross_entropy_with_logits(mp, mg, reduction='mean')
            s      = mp.sigmoid()
            n      = 2 * (s * mg).sum(-1)
            d      = s.sum(-1) + mg.sum(-1)
            l_dice = (1 - (n + 1) / (d + 1)).mean()
            overlap = F.relu(s.sum(0) - 1.0)
            l_overlap = overlap.mean()
        else:
            l_bce     = torch.tensor(0., device=dev)
            l_dice    = torch.tensor(0., device=dev)
            l_overlap = torch.tensor(0., device=dev)

        loss = self.lc * l_cls + self.lb * l_bce + self.ld * l_dice + 2.0 * l_overlap
        return loss, l_cls, l_bce, l_dice, l_overlap

    def forward(self, pc, pm, sem, gl, gm, sem_labels, aux_outputs=None):
        dev = pc.device

        if not (torch.isfinite(pc).all() and torch.isfinite(pm).all()):
            dummy = torch.tensor(0.0, device=dev, requires_grad=True)
            return dummy, {'loss_cls': 0., 'loss_bce': 0., 'loss_dice': 0.,
                           'loss_sem': 0., 'loss_overlap': 0., 'loss_aux': 0., 'total': 0.}

        main_loss, l_cls, l_bce, l_dice, l_overlap = self._single_loss(pc, pm, gl, gm, dev)

        # semantic loss (가중치 0.5 + label_smoothing)
        l_sem = 0.5 * F.cross_entropy(sem, sem_labels.to(dev), reduction='mean',
                                       label_smoothing=0.1)

        l_aux = torch.tensor(0., device=dev)
        if aux_outputs is not None:
            for aux_cls, aux_mask in aux_outputs[:-1]:
                if torch.isfinite(aux_cls).all() and torch.isfinite(aux_mask).all():
                    a_loss, _, _, _, _ = self._single_loss(aux_cls, aux_mask, gl, gm, dev)
                    l_aux = l_aux + a_loss
            if len(aux_outputs) > 1:
                l_aux = 0.5 * l_aux / (len(aux_outputs) - 1)

        total = main_loss + l_sem + l_aux
        return total, {
            'loss_cls':     l_cls.item(),
            'loss_bce':     l_bce.item(),
            'loss_dice':    l_dice.item(),
            'loss_sem':     l_sem.item(),
            'loss_overlap': l_overlap.item(),
            'loss_aux':     l_aux.item(),
            'total':        total.item(),
        }


class PanCADNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone      = SegFormerBackbone(out_channels=256)
        self.line_sampling = HybridLineSampling(feat_channels=256)
        self.gatv2         = ProgressiveGATv2Head()
        self.fusion        = AdaptiveFusion(graph_dim=GAT_HIDDEN, visual_dim=256, out_dim=DEC_HIDDEN)
        self.decoder       = PrimitiveMask2FormerDecoder()

    def forward(self, pv, starts, ends, geo, eil, eal, batch_idx):
        fm    = self.backbone(pv)
        vi    = self.line_sampling(fm, starts, ends, batch_idx, SCALE_ORG)
        xg    = self.gatv2(geo, eil, eal)
        fused = self.fusion(xg, vi)
        B     = pv.shape[0]
        pc_list, pm_list, sem_list, aux_list = [], [], [], []
        for b in range(B):
            mask = (batch_idx == b)
            cls_b, mask_b, sem_b, aux_b = self.decoder(fused[mask])
            pc_list.append(cls_b)
            pm_list.append(mask_b)
            sem_list.append(sem_b)
            aux_list.append(aux_b)
        return pc_list, pm_list, sem_list, aux_list
