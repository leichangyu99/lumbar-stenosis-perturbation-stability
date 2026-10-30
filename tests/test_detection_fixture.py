"""Detection fixture (EXPERIMENT_PLAN V3 section 2): a fixed linear grader on synthetic layer3 maps whose
flip rates and JS values are known analytically; flips must match exactly, JS to float64 rounding."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs_metrics import cohort, js_bits, patient_level  # noqa: E402

PRIOR = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float64)


def linear_grader(maps, mask):
    """p_level = (mean over visible slices of spatial mean of channels 0..2) normalised to sum 1."""
    f = maps[:, :, :3].mean(dim=(-1, -2))  # [5,3 slices,3 channels]
    count = mask.sum(-1)
    pooled = (f * mask[..., None]).sum(1) / count.clamp_min(1)[:, None]
    p = pooled / pooled.sum(-1, keepdim=True).clamp_min(1e-300)
    return torch.where((count > 0)[:, None], p, PRIOR), (count > 0)


def synthetic(levels, visible):
    maps = torch.zeros(5, 3, 256, 14, 14, dtype=torch.float64); mask = torch.zeros(5, 3, dtype=torch.bool)
    for k, (vals, vis) in enumerate(zip(levels, visible)):
        if vis:
            for j in range(3):
                maps[k, j, :3] = torch.tensor(vals, dtype=torch.float64)[:, None, None]; mask[k, j] = True
    return maps, mask


def closed_form_js(p, q):
    total = 0.0
    for a, b in zip(p, q):
        m = (a + b) / 2
        total += (0.5 * a * math.log(a / m) if a > 0 else 0.0) + (0.5 * b * math.log(b / m) if b > 0 else 0.0)
    return total / math.log(2)


def main():
    # Patient P1: A unchanged, B disjoint flip, C tie (lowest grade -> no flip), D unchanged, E invisible.
    p1o = [(1, 0, 0), (1, 0, 0), (1, 0, 0), (0, .5, .5), (1, 0, 0)]
    p1v = [(1, 0, 0), (0, 1, 0), (.5, .5, 0), (0, .5, .5), (0, 0, 1)]
    vis1 = [True, True, True, True, False]
    # Patient P2: no visible level -> excluded from denominator, counted.
    p2o = [(1, 0, 0)] * 5; p2v = [(0, 1, 0)] * 5; vis2 = [False] * 5
    # Patient P3: one visible level flipping 2 -> 0, the rest invisible.
    p3o = [(.2, .3, .5)] + [(1, 0, 0)] * 4; p3v = [(.5, .3, .2)] + [(0, 1, 0)] * 4; vis3 = [True] + [False] * 4
    probs_o, probs_v, visible = {}, {}, {}
    for sid, (po, pv, vis) in {'P1': (p1o, p1v, vis1), 'P2': (p2o, p2v, vis2), 'P3': (p3o, p3v, vis3)}.items():
        mo, ko = synthetic(po, vis); mv, kv = synthetic(pv, vis)
        go, vo = linear_grader(mo, ko); gv, _ = linear_grader(mv, kv)
        probs_o[sid] = go.numpy(); probs_v[sid] = gv.numpy(); visible[sid] = vo.numpy()
    # Exact analytic expectations
    jsC = 0.75 * math.log(4 / 3) / math.log(2)
    jsP3 = closed_form_js((.2, .3, .5), (.5, .3, .2))
    assert js_bits((1, 0, 0), (1, 0, 0)) == 0.0 and js_bits((1, 0, 0), (0, 1, 0)) == 1.0
    assert abs(js_bits((1, 0, 0), (.5, .5, 0)) - jsC) <= 1e-15
    r1 = patient_level(probs_o['P1'], probs_v['P1'], visible['P1'])
    assert r1['flip'] == 0.25 and r1['levels'] == 4, r1
    assert abs(r1['js'] - (0 + 1 + jsC + 0) / 4) <= 1e-15, r1
    assert patient_level(probs_o['P2'], probs_v['P2'], visible['P2']) is None
    r3 = patient_level(probs_o['P3'], probs_v['P3'], visible['P3'])
    assert r3['flip'] == 1.0 and r3['levels'] == 1 and abs(r3['js'] - jsP3) <= 1e-15, (r3, jsP3)
    c = cohort(probs_o, probs_v, visible, ['P1', 'P2', 'P3'])
    assert c['evaluable'] and c['patients'] == 2 and c['excluded_no_visible_level'] == 1
    assert c['flip'] == (0.25 + 1.0) / 2 and abs(c['js'] - (r1['js'] + jsP3) / 2) <= 1e-15, c
    # Identity view: flip and JS exactly 0
    ci = cohort(probs_o, probs_o, visible, ['P1', 'P2', 'P3'])
    assert ci['flip'] == 0.0 and ci['js'] == 0.0
    # No computable patient -> not evaluable (never a pass)
    assert cohort(probs_o, probs_v, visible, ['P2'])['evaluable'] is False
    # Symmetry and bounds
    rng = np.random.default_rng(0); a = rng.dirichlet([1, 1, 1], 1000); b = rng.dirichlet([1, 1, 1], 1000)
    j = js_bits(a, b); assert np.allclose(j, js_bits(b, a), atol=0, rtol=0) and (j >= 0).all() and (j <= 1 + 1e-12).all()
    print('detection fixture passed', {'P1': r1, 'P3': r3, 'cohort_flip': c['flip'], 'cohort_js': c['js']})


if __name__ == '__main__':
    main()
