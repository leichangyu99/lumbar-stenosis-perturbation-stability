"""CS20261013 view construction on top of the frozen CS20261012 engine: AugMix training views and
TTA-jittered evaluation views (results/validated_methods_20261013/PLAN.md section 3.1)."""
import numpy as np
import torch

import cs_engine as E

OPS = ('tcol', 'trow', 'rot', 'scale')


def _rng(text): return np.random.Generator(np.random.PCG64(E.seed_of(text)))


def augmix_spec(sid, epoch, k):
    """w ~ Dir(1,1,1), m ~ Beta(1,1), then 3 chains of depth U{1,2,3} geometric ops (params accumulate)."""
    rng = _rng('CS20261013|augmix|%s|%d|%d' % (sid, epoch, k))
    w = rng.dirichlet([1.0, 1.0, 1.0]); m = float(rng.beta(1.0, 1.0)); chains = []
    for _ in range(3):
        p = {'tx': 0.0, 'ty': 0.0, 'theta_deg': 0.0, 's': 1.0}
        for _ in range(int(rng.integers(1, 4))):
            op = OPS[int(rng.integers(0, 4))]
            if op == 'tcol': p['tx'] += float(rng.uniform(-1, 1))
            elif op == 'trow': p['ty'] += float(rng.uniform(-1, 1))
            elif op == 'rot': p['theta_deg'] += float(rng.uniform(-4, 4))
            else: p['s'] *= float(rng.uniform(.94, 1.06))
        chains.append(E.view('chain', 'X', **p))
    return {'w': [float(x) for x in w], 'm': m, 'chains': chains}


def tta_jitter(sid, m):
    rng = _rng('CS20261013|tta|%s|%d' % (sid, m))
    return {'tx': float(rng.uniform(-.5, .5)), 'ty': float(rng.uniform(-.5, .5)), 'theta_deg': float(rng.uniform(-2, 2)), 's': float(rng.uniform(.97, 1.03))}


def compose(v, j):
    """Evaluation view v composed with geometric jitter j (translations/angles add, scales multiply)."""
    out = dict(v); out.update(tx=v['tx'] + j['tx'], ty=v['ty'] + j['ty'], theta_deg=v['theta_deg'] + j['theta_deg'], s=v['s'] * j['s'])
    out['family'] = 'X'; out['name'] = v['name'] + '|tta'
    return out


class Patient:
    """One decoded patient: shared slice images, level points and per-view crops."""

    def __init__(self, root, row, sid):
        self.sid = sid; self.ds = E.read_series(root, row); self.points = E.level_points(row); self.images = {}

    def image(self, j, v):
        key = (j, tuple(v['window']), v['sigma'])
        if key not in self.images: self.images[key] = E.intensity(self.ds[j], v['window'], v['sigma'], E.noise_seed(self.sid, j))
        return self.images[key]

    def crops(self, v):
        """{(k, j): crop} for slices whose (intervened) centre lies inside the slice."""
        out = {}
        for k, point in self.points:
            for j, ds in enumerate(self.ds):
                c = E.centre(ds, point, v)
                if c is not None: out[(k, j)] = E.crop(self.image(j, v), *c, v)
        return out

    def encode(self, crops, prefix, device='cuda'):
        locs = sorted(crops); return E.encode([crops[l] for l in locs], locs, prefix, device)

    def view(self, v, prefix):
        """maps/mask for a non-D view (any T/R/S/N/W composition)."""
        return self.encode(self.crops(v), prefix)

    def augmix(self, clean, spec, prefix):
        """Mixed crops for visible clean slices; an out-of-slice chain crop falls back to the clean crop."""
        chains = [self.crops(c) for c in spec['chains']]; mixed = {}; fallback = 0
        for loc, c0 in clean.items():
            acc = sum(w * (ch[loc] if loc in ch else c0) for w, ch in zip(spec['w'], chains))
            fallback += sum(loc not in ch for ch in chains)
            mixed[loc] = spec['m'] * c0 + (1 - spec['m']) * acc
        maps, mask = self.encode(mixed, prefix)
        return maps, mask, fallback


def drop(maps, mask, rule):
    dmask, kept = E.drop_slice(mask, rule); dmaps = maps.clone(); dmaps[mask & ~dmask] = 0
    return dmaps, dmask


def eval_views():
    """The 11 evaluation views (subset of the frozen 27): I, T 1px x2, N x3, W x3, D x2."""
    keep = ['I', 'T_1_col', 'T_1_row', 'N_0.02', 'N_0.05', 'N_0.1', 'W_2_98', 'W_5_95', 'W_0.5_99.5', 'D_center', 'D_edge']
    allv = {v['name']: v for v in E.evaluation_views()}
    return [allv[n] for n in keep]


def eval_patient(P, prefix, n_tta=8):
    """{(view name, m or None): (maps, mask)} for 11 views and their 8 TTA jitters (m = 0..7)."""
    out = {}
    for m in [None] + list(range(n_tta)):
        jit = tta_jitter(P.sid, m) if m is not None else None
        base = None
        for v in eval_views():
            if v['family'] == 'D': continue
            vv = compose(v, jit) if jit else v
            out[(v['name'], m)] = P.view(vv, prefix)
            if v['name'] == 'I': base = out[(v['name'], m)]
        for rule in ('center', 'edge'):
            out[('D_' + rule, m)] = drop(*base, rule)
    return out
