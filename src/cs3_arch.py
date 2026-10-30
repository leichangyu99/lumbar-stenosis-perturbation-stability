"""CS20261013-MA architectures (results/multiarch_20261013/PLAN.md): winner-style encoders/aggregators
trained end-to-end on raw ROI crops from the frozen engine, plus GPU ShiftScaleRotate/brightness-contrast."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import cs_engine as E

MEAN = torch.tensor([.485, .456, .406]); STD = torch.tensor([.229, .224, .225])
WEIGHT_FILES = {'resnet18': ('resnet18-f37072fd.pth', 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'),
                'convnext_tiny': ('convnext_tiny-983f1562.pth', '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'),
                'efficientnet_v2_s': ('efficientnet_v2_s-dd5fe13b.pth', 'dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74'),
                'swin_t': ('swin_t-704ceda3.pth', '704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b'),
                'resnet18_swsl': ('resnet18_swsl_pytorch_model.bin', 'e03bb9e9b011aa1f6bcb2673e5e26fff5b23a07c222c1130c0796f499469b07a'),
                'resnet50': ('resnet50-11ad3fa6.pth', '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca')}
ARCHS = {'A0': ('resnet18', 'mean'), 'A1': ('convnext_tiny', 'stack25d'), 'A2': ('efficientnet_v2_s', 'lstm_attn'), 'A3': ('swin_t', 'attn'),
         'A4': ('resnet18', 'mean'), 'A5': ('resnet50', 'mean')}
WEIGHT_KEY = {'A0': 'resnet18', 'A1': 'convnext_tiny', 'A2': 'efficientnet_v2_s', 'A3': 'swin_t', 'A4': 'resnet18_swsl', 'A5': 'resnet50'}


def weight_file(arch): return WEIGHT_FILES[WEIGHT_KEY[arch]][0]


def raw_crop(image, x, y, dx, dy, v):
    """Single-channel grid_sample output of E.crop before channel repeat and ImageNet normalisation."""
    from cs_engine import SIZE, WIDTH_MM
    width_mm = WIDTH_MM * v['s'] if v['s'] != 1 else WIDTH_MM
    offsets = (torch.arange(SIZE, dtype=torch.float32) + .5) / SIZE - .5
    if v['theta_deg'] == 0:
        xx = x + offsets * width_mm / dx; yy = y + offsets * width_mm / dy; gy, gx = torch.meshgrid(yy, xx, indexing='ij')
    else:
        t = math.radians(v['theta_deg']); c, s = math.cos(t), math.sin(t)
        mm = offsets * width_mm; vv, uu = torch.meshgrid(mm, mm, indexing='ij')
        gx = x + (c * uu - s * vv) / dx; gy = y + (s * uu + c * vv) / dy
    h, w = image.shape
    grid = torch.stack((2 * (gx + .5) / w - 1, 2 * (gy + .5) / h - 1), -1)[None]
    return F.grid_sample(torch.from_numpy(np.ascontiguousarray(image))[None, None], grid, align_corners=False, mode='bilinear', padding_mode='zeros')[0, 0]


def patient_raw(P, v):
    """([5,3,224,224] float32 raw crops, [5,3] mask) for a non-D view of a cs2_engine.Patient."""
    out = torch.zeros(5, 3, 224, 224); mask = torch.zeros(5, 3, dtype=torch.bool)
    for k, point in P.points:
        for j, ds in enumerate(P.ds):
            c = E.centre(ds, point, v)
            if c is not None: out[k, j] = raw_crop(P.image(j, v), *c, v); mask[k, j] = True
    return out, mask


def patient_views(P, views):
    """{name: (raw, mask)} for evaluation views; D views drop slices from the identity view."""
    res = {}
    for v in views:
        if v['family'] != 'D': res[v['name']] = patient_raw(P, v)
    for v in views:
        if v['family'] == 'D':
            raw, mask = res['I']; dmask, _ = E.drop_slice(mask, v['drop']); r = raw.clone(); r[mask & ~dmask] = 0; res[v['name']] = (r, dmask)
    return res


def augment(raw, gen):
    """ShiftScaleRotate (shift +-6%, scale 0.9-1.1, rotate +-15 deg) + brightness/contrast +-0.2, one draw per
    patient-level (shared by its slices).  raw [B,5,3,H,W] on GPU in [0,1]."""
    B, L = raw.shape[:2]; n = B * L; d = raw.device
    u = lambda lo, hi: torch.rand(n, generator=gen, device=d) * (hi - lo) + lo
    ang = u(-15, 15) * math.pi / 180; sc = u(.9, 1.1); tx = u(-.06, .06) * 2; ty = u(-.06, .06) * 2
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], -1), torch.stack([sin, cos, ty], -1)], 1)
    x = raw.reshape(n, 3, *raw.shape[-2:]).float()
    grid = F.affine_grid(theta, x.shape, align_corners=False); x = F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')
    br = u(-.2, .2)[:, None, None, None]; ct = u(.8, 1.2)[:, None, None, None]; m = x.mean((-1, -2), keepdim=True)
    x = ((x - m) * ct + m + br).clamp(0, 1)
    return x.reshape(raw.shape)


def nearest_fill(raw, mask):
    """For 2.5D: replace each invisible slice of a visible level with the nearest visible slice (lower index on ties)."""
    out = raw.clone()
    for idx in np.ndindex(*mask.shape[:-1]):
        vis = [j for j in range(3) if bool(mask[idx + (j,)])]
        if not vis: continue
        for j in range(3):
            if j not in vis: out[idx + (j,)] = raw[idx + (min(vis, key=lambda q: (abs(q - j), q)),)]
    return out


class Encoder(nn.Module):
    def __init__(self, name, weights_path):
        super().__init__()
        import torchvision.models as tvm
        net = getattr(tvm, name)(weights=None); net.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True)); self.name = name
        if name in ('resnet18', 'resnet50'): net.fc = nn.Identity(); self.dim = 512 if name == 'resnet18' else 2048; self.f = net
        elif name == 'convnext_tiny': self.f = nn.Sequential(net.features, net.avgpool, net.classifier[0], nn.Flatten(1)); self.dim = 768
        elif name == 'efficientnet_v2_s': self.f = nn.Sequential(net.features, net.avgpool, nn.Flatten(1)); self.dim = 1280
        elif name == 'swin_t': self.f = nn.Sequential(net.features, net.norm, net.permute, net.avgpool, nn.Flatten(1)); self.dim = 768
        self.register_buffer('mean', MEAN[:, None, None].clone()); self.register_buffer('std', STD[:, None, None].clone())

    def forward(self, x3):          # x3 [N,3,H,W] in [0,1]
        return self.f((x3 - self.mean) / self.std)


class Grader(nn.Module):
    """encoder -> slice aggregation -> per-level feature -> SagHead-style head with 5-level context."""

    def __init__(self, arch, weights_path):
        super().__init__()
        enc, self.agg = ARCHS[arch]; self.enc = Encoder(enc, weights_path); d = self.enc.dim
        if self.agg == 'lstm_attn':
            self.lstm = nn.LSTM(d, 256, batch_first=True, bidirectional=True); d = 512
        if self.agg in ('lstm_attn', 'attn'):
            self.att = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        self.norm = nn.LayerNorm(d); self.d = d
        self.head = nn.Sequential(nn.Linear(2 * d + 4, 192), nn.ReLU(), nn.Dropout(.1), nn.Linear(192, 3))

    def head_params(self): return [p for n, p in self.named_parameters() if not n.startswith('enc.')]

    def forward(self, raw, mask):    # raw [B,5,3,H,W] in [0,1], mask [B,5,3]
        B = raw.shape[0]; vis = mask.any(-1)
        if self.agg == 'stack25d':
            x = nearest_fill(raw, mask) if (~mask & vis[..., None]).any() else raw
            feat = torch.zeros(B, 5, self.enc.dim, device=raw.device, dtype=torch.float32)
            if vis.any(): feat[vis] = self.enc(x[vis]).float()
        else:
            per = torch.zeros(B, 5, 3, self.enc.dim, device=raw.device, dtype=torch.float32)
            if mask.any(): per[mask] = self.enc(raw[mask][:, None].expand(-1, 3, -1, -1)).float()
            if self.agg == 'mean':
                feat = per.sum(2) / mask.sum(-1).clamp_min(1)[..., None]
            else:
                h = per
                if self.agg == 'lstm_attn':
                    h, _ = self.lstm(per.reshape(B * 5, 3, -1)); h = h.reshape(B, 5, 3, -1)
                a = self.att(h).squeeze(-1).masked_fill(~mask, -1e4); w = a.softmax(-1) * mask
                feat = (w[..., None] * h).sum(2) / w.sum(-1).clamp_min(1e-12)[..., None]
        feat = self.norm(feat) * vis[..., None]
        context = feat.sum(1) / vis.sum(1).clamp_min(1)[:, None]
        levels = torch.linspace(0, 1, 5, device=raw.device)[None, :, None].expand(B, -1, -1)
        pos = torch.cat((levels, levels.square(), vis[..., None].float(), vis.float().mean(1)[:, None, None].expand(-1, 5, -1)), -1)
        return self.head(torch.cat((feat, context[:, None].expand(-1, 5, -1), pos), -1))
