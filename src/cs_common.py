"""CS20261012 shared remote utilities: paths, hashing, frozen grader recipe (EXPERIMENT_PLAN V3 section 0, E2),
consistency/shuffle terms, prediction with Laplace priors, and patient-level statistics."""
import hashlib, json, math, os, time, uuid
from pathlib import Path

PARENT = Path('workspace')
CS = PARENT / 'cs20261012'
R4 = PARENT / 'r4_workspace'
CACHE = PARENT / 'runs/remote/M1_sag_layer4_pair_s20261007/front_cache'
HELPERS = PARENT / 'scripts'
WEIGHTS = PARENT / 'weights/resnet18-f37072fd.pth'
DATA = Path('data/LumbarDISC/dataset')
PRED = PARENT / 'configs/sagittal_predictions.json'
DIAG_LABELS = R4 / 'e0_r5/G1_LABELS_DIAGNOSTIC_ONLY.json'
DIAG_LABELS_SHA = '6c9cb989463c5333c273a1e69c29b7857baff0b186cc5c52b2061f24fb26822b'
SEEDS = (20261012, 20261013, 20261014)
EPOCHS = 12
LEVEL_KEYS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
GRADES = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}


def require(c, m):
    if not c: raise RuntimeError(m)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(p, v):
    p = Path(p); tmp = p.with_name(p.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(v, indent=1, allow_nan=False) + '\n', encoding='utf-8'); os.replace(tmp, p)


def hrank(tag, ids):
    return sorted(ids, key=lambda s: hashlib.sha256(('CS20261012|' + tag + '|' + s).encode()).hexdigest())


def seed_of(text): return int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:16], 16)


class Clock:
    def __init__(self, limit): self.start = time.monotonic(); self.limit = limit

    def check(self):
        if time.monotonic() - self.start >= self.limit: raise TimeoutError('Stage budget exhausted')

    def elapsed(self): return time.monotonic() - self.start


class GpuLock:
    def __init__(self, name):
        self.path = CS / 'GPU_JOB.lock'; self.token = uuid.uuid4().hex; self.name = name; self.owned = False

    def __enter__(self):
        require(not (R4 / 'GPU_JOB.lock').exists(), 'R4 GPU job lock present')
        with self.path.open('x', encoding='utf-8') as f:
            json.dump({'run_id': self.name, 'token': self.token, 'pid': os.getpid(), 'time': time.time()}, f)
        self.owned = True; return self

    def __exit__(self, *a):
        if self.owned and self.path.exists() and load(self.path).get('token') == self.token: self.path.unlink()


def labels_from_train_csv(ids):
    """Grades for the given patients only (role-gated by the caller); -1 = unknown."""
    import csv
    ids = set(ids); out = {}
    for r in csv.DictReader((DATA / 'train.csv').open(encoding='utf-8-sig')):
        if r['study_id'] in ids:
            out[r['study_id']] = [GRADES.get(r['spinal_canal_stenosis_' + k], -1) for k in LEVEL_KEYS]
    require(set(out) == ids, 'Label rows missing')
    return out


def sattolo(n, seed):
    """Uniform cyclic permutation without fixed points (n >= 2)."""
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(seed)); p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i)); p[i], p[j] = p[j], p[i]
    assert all(p[i] != i for i in range(n))
    return p


def js_logits(za, zb):
    """Per-level JS(p_a, p_b)/ln2 from logits, two-sided gradients."""
    import torch
    la = za.log_softmax(-1); lb = zb.log_softmax(-1); lm = torch.logaddexp(la, lb) - math.log(2)
    pa = la.exp(); pb = lb.exp()
    return (0.5 * (pa * (la - lm)).sum(-1) + 0.5 * (pb * (lb - lm)).sum(-1)) / math.log(2)


class Recipe:
    """Frozen grader recipe; one instance per process holding the frozen backbone."""

    def __init__(self, torch, np):
        import sys
        sys.path.insert(1, str(HELPERS))
        from torch import nn
        from torch.nn import functional as F
        from torchvision.models import resnet18
        from train_sag_layer4_pair import Diagnosis, state_hash, bn_state, same_state
        from train_sag_baseline import SagHead
        self.torch, self.np, self.nn, self.F = torch, np, nn, F
        self.Diagnosis, self.SagHead, self.state_hash, self.bn_state, self.same_state = Diagnosis, SagHead, state_hash, bn_state, same_state
        backbone = resnet18(weights=None); backbone.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True)); backbone.eval()
        for p in backbone.parameters(): p.requires_grad_(False)
        self.backbone = backbone; self.l4_hash = state_hash(backbone.layer4.state_dict())

    def supply(self, ids, labels):
        torch = self.torch
        ty = torch.tensor([labels[s] for s in ids], dtype=torch.long)
        counts = torch.bincount(ty[ty >= 0], minlength=3)
        cw = (counts.sum() / (3 * counts.float().clamp_min(1))).sqrt(); cw /= cw.mean()
        lc = torch.stack([torch.bincount(ty[:, j][ty[:, j] >= 0], minlength=3) for j in range(5)])
        prior_counts = torch.stack([lc[j] if lc[j].sum() > 0 else counts for j in range(5)])
        priors = (prior_counts.float() + 1) / (prior_counts.sum(-1).float()[:, None] + 3)
        return ty, counts, cw, priors

    def normalisation(self, ids, gI, mb=4):
        torch, F = self.torch, self.F
        import copy
        fl4 = copy.deepcopy(self.backbone.layer4).cuda().eval(); feats = []
        with torch.no_grad():
            for s0 in range(0, len(ids), mb):
                b = ids[s0:s0 + mb]; maps = torch.stack([gI[s][0] for s in b]); mask = torch.stack([gI[s][1] for s in b])
                per = torch.zeros((*mask.shape, 512), device='cuda')
                if mask.any(): per[mask] = F.adaptive_avg_pool2d(fl4(maps[mask]), 1).flatten(1)
                cnt = mask.sum(-1); pooled = per.sum(2) / cnt.clamp_min(1)[..., None]; feats.append(pooled[cnt > 0].cpu())
        valid = torch.cat(feats); mean, std = valid.mean(0).cuda(), valid.std(0).clamp_min(.05).cuda()
        require(bool(torch.isfinite(mean).all() and torch.isfinite(std).all()), 'Nonfinite normalisation')
        return mean, std

    def fit(self, ids, labels, gI, gV, mode, seed, check, log=None, mb=4, lam=1.0):
        """mode in STD/AUG/CONS/SHUF; returns (net, info).  gI/gV: sid -> (maps, mask) resident on GPU."""
        torch, np, nn, F = self.torch, self.np, self.nn, self.F
        require(mode in ('STD', 'AUG', 'CONS', 'SHUF'), 'mode')
        ty, counts, cw, priors = self.supply(ids, labels)
        cpat = [int((ty == k).any(-1).sum()) for k in range(3)]
        mean, std = self.normalisation(ids, gI, mb)
        torch.manual_seed(seed); np.random.seed(seed)
        net = self.Diagnosis(self.backbone.layer4, self.SagHead().cuda(), mean, std, True).cuda()
        h0, l0, bn0 = self.state_hash(net.head.state_dict()), self.state_hash(net.layer4.state_dict()), self.bn_state(net.layer4)
        require(l0 == self.l4_hash, 'Non-ImageNet layer4 init')
        opt = torch.optim.AdamW([{'params': net.layer4.parameters(), 'lr': 1e-4}, {'params': net.head.parameters(), 'lr': 1e-3}], weight_decay=.01)
        cwc = cw.cuda(); g = torch.Generator().manual_seed(seed); hist = []
        stats = {'cons_pairs': 0, 'cons_pairs_no_common_level': 0, 'shuf_single_patient_microbatches': 0, 'steps': 0}
        n2 = mode != 'STD'
        def ce(z, tgt, known, vp):
            per = F.cross_entropy(z.reshape(-1, 3), tgt.clamp_min(0).reshape(-1), weight=cwc, reduction='none').reshape(-1, 5) * known
            use = known.sum(-1) > 0
            return (per.sum(-1)[use] / known.sum(-1)[use]).sum() / vp if use.any() else None
        for epoch in range(1, EPOCHS + 1):
            check(); net.train(); order = torch.randperm(len(ids), generator=g).tolist(); losses = []
            for e_idx, s0 in enumerate(range(0, len(order), 32)):
                eff = order[s0:s0 + 32]; vp = int((ty[eff] >= 0).any(-1).sum())
                if vp == 0: continue
                micro = [eff[o:o + mb] for o in range(0, len(eff), mb)]
                pairs = []
                for m_idx, ix in enumerate(micro):
                    if mode == 'CONS': pi = list(range(len(ix)))
                    elif mode == 'SHUF':
                        if len(ix) < 2: pi = None; stats['shuf_single_patient_microbatches'] += 1
                        else: pi = sattolo(len(ix), seed_of('CS20261012|shuf|%d|%d|%d|%d' % (seed, epoch, e_idx, m_idx)))
                    else: pi = None
                    pairs.append(pi)
                # contributing pairs in this effective batch (reduction denominator, identical for CONS and SHUF)
                npair = 0
                if mode in ('CONS', 'SHUF'):
                    for ix, pi in zip(micro, pairs):
                        if pi is None: continue
                        for a, b in enumerate(pi):
                            ok = bool((gI[ids[ix[a]]][1].any(-1) & gV[ids[ix[b]]][1].any(-1)).any())
                            npair += ok; stats['cons_pairs'] += 1; stats['cons_pairs_no_common_level'] += (not ok)
                opt.zero_grad(set_to_none=True); bl = 0.
                for ix, pi in zip(micro, pairs):
                    sids = [ids[i] for i in ix]; tgt = ty[ix].cuda(); known = tgt >= 0
                    mI = torch.stack([gI[s][0] for s in sids]); kI = torch.stack([gI[s][1] for s in sids])
                    z0 = net(mI, kI); require(bool(torch.isfinite(z0).all()), 'Nonfinite logits')
                    loss = ce(z0, tgt, known, vp)
                    if n2:
                        mV = torch.stack([gV[s][0] for s in sids]); kV = torch.stack([gV[s][1] for s in sids])
                        zv = net(mV, kV); require(bool(torch.isfinite(zv).all()), 'Nonfinite V logits')
                        lv = ce(zv, tgt, known, vp)
                        loss = None if loss is None else .5 * loss + .5 * lv
                        if mode in ('CONS', 'SHUF') and pi is not None and npair > 0:
                            vis0 = kI.any(-1); visv = kV.any(-1); idx = torch.tensor(pi, device='cuda')
                            both = vis0 & visv[idx]; j = js_logits(z0, zv[idx])
                            cnt = both.sum(-1); okp = cnt > 0
                            if okp.any():
                                term = lam * ((j * both).sum(-1)[okp] / cnt[okp]).sum() / npair
                                loss = term if loss is None else loss + term
                    if loss is not None:
                        loss.backward(); bl += float(loss.detach())
                nn.utils.clip_grad_norm_(net.layer4.parameters(), 5, error_if_nonfinite=True)
                nn.utils.clip_grad_norm_(net.head.parameters(), 5, error_if_nonfinite=True)
                opt.step(); losses.append(bl); stats['steps'] += 1
            require(losses and all(bool(torch.isfinite(x).all()) for x in net.parameters()), 'No steps or nonfinite weights')
            require(self.same_state(bn0, self.bn_state(net.layer4)), 'BN statistics changed')
            hist.append({'epoch': epoch, 'train_loss': sum(losses) / len(losses)})
            if log: log('epoch', mode=mode, seed=seed, **hist[-1])
        require(self.state_hash(net.layer4.state_dict()) != l0 and self.state_hash(net.head.state_dict()) != h0, 'Training did not update layer4/head')
        net.eval()
        info = {'mode': mode, 'seed': seed, 'train_n': len(ids), 'class_counts': counts.tolist(), 'class_patients': cpat, 'class_weights': cw.tolist(),
                'priors': priors.tolist(), 'history': hist, 'stats': stats, 'head_init_sha': h0}
        return net, priors, info

    def predict(self, net, ids, getter, priors, mb=4):
        """{sid: (probs [5,3] with Laplace priors on invisible levels, visible [5])} as numpy float64/bool."""
        torch = self.torch; out = {}
        with torch.no_grad():
            for s0 in range(0, len(ids), mb):
                b = ids[s0:s0 + mb]; items = [getter(s) for s in b]
                z = net(torch.stack([m for m, _ in items]).cuda(), torch.stack([k for _, k in items]).cuda())
                require(bool(torch.isfinite(z).all()), 'Nonfinite prediction')
                pr = z.softmax(-1).cpu().double(); vis = torch.stack([k for _, k in items]).any(-1).cpu()
                pr = torch.where(vis[..., None], pr, priors.double()[None].expand(len(pr), -1, -1))
                for s, p, v in zip(b, pr.numpy(), vis.numpy()): out[s] = (p, v)
        return out


def rps_patient(y, p):
    """Mean SR RPS over known levels; None if no known level."""
    import numpy as np
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64); k = y >= 0
    if not k.any(): return None
    cp = np.cumsum(p, -1)[..., :2]; ind = np.stack([(y <= 0), (y <= 1)], -1).astype(float)
    return float((0.5 * ((cp - ind) ** 2).sum(-1))[k].mean())


def severe_miss(y, p):
    import numpy as np
    return any(y[j] == 2 and int(np.argmax(p[j])) != 2 for j in range(5))


def boot_indices(n, B=10000, seed=20261012):
    import numpy as np
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def ci(values, idx):
    import numpy as np
    m = np.asarray(values, dtype=np.float64)[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
