"""Synthetic check of E3/E4 statistics: identity zero, T flips detected, C3a ordering, structure."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import cs_e34 as M, cs_engine as E
rng = np.random.default_rng(1); vn = [v['name'] for v in E.evaluation_views()]; conf = ['C%02d' % i for i in range(30)]
names = ['%s_%d' % (m, s) for m in M.MODES for s in (20261012, 20261013, 20261014)]
base = rng.dirichlet([2, 1, 1], (30, 5)); probs = np.repeat(np.repeat(base[None, :, None], 27, 2), 12, 0)
for k, n in enumerate(names):
    noise = {'STD': .6, 'AUG': .5, 'CONS': .1, 'SHUF': .45}[n.split('_')[0]]
    for v_i, v in enumerate(vn):
        if v != 'I': probs[k, :, v_i] = 0.5 * base + 0.5 * rng.dirichlet([1, 1, 1], (30, 5)) * noise / max(noise, 1e-9) * (1 if noise > .2 else .1) + (0 if noise > .2 else .45 * base)
probs /= probs.sum(-1, keepdims=True)
vis = np.ones((30, 27, 5), bool); vis[0, :, :] = False
M.labels_from_train_csv = lambda ids: {s: list(rng.integers(-1, 3, 5)) for s in ids}
r = M.phase_b(names, conf, vn, probs, vis, {}, np)
assert r['identity_flip_js_zero'] and r['flip_noncomputable_patients'] == 1
print('C1', r['E3']['C1']); print('E4', r['E4']['conditions'], r['E4']['heldout_flip'])
assert r['E3']['C1']['C1_passed'] and r['E4']['comparisons']['CONS-STD']['mean_delta'] < 0
print('e34 stats synthetic passed')
