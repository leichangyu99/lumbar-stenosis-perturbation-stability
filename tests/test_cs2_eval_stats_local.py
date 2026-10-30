"""Synthetic check of CS2 phase B statistics (structure, AUROC orientation, TTA aggregation)."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import cs2_eval as M, cs2_engine as E2
rng = np.random.default_rng(3); n = 60; vn = [v['name'] for v in E2.eval_views()]; keys = [(v, m) for v in vn for m in [None] + list(range(8))]
names = ['%s_%d' % (s, x) for s in M.SCHEMES for x in (1, 2, 3)]
Y = rng.integers(-1, 3, (n, 5)); Y[:, 0] = 2
base = rng.dirichlet([1, 1, 1], (n, 5)); probs = np.zeros((len(names), n, len(keys), 5, 3))
for k, nm in enumerate(names):
    amp = .02 if nm.startswith('AUGMIX') else .3
    for q, key in enumerate(keys):
        probs[k, :, q] = base if key == ('I', None) else (1 - amp) * base + amp * rng.dirichlet([1, 1, 1], (n, 5))
vis = np.ones((n, len(keys), 5), bool)
M.labels_from_train_csv = lambda ids: {s: list(Y[i]) for i, s in enumerate(ids)}
r = M.phase_b('selection', names, ['S%d' % i for i in range(n)], vn, keys, probs, vis, np)
print(r['R1']['conditions'], r['R1']['AUGMIX-STD_flip'], r['R2']['conditions'], r['R3']['sensitivity'], r['heldout_flip']['TTA-STD'])
assert r['R1']['AUGMIX-STD_flip']['mean_delta'] < 0 and 0 <= r['R3']['sensitivity']['auroc'] <= 1
# AUROC orientation: perfect score -> 1
s = np.array([.1, .2, .9, .8]); e = np.array([0, 0, 1, 1], bool); print(M.auroc_boot(s, e, np.arange(4), 4, np)['auroc']); assert M.auroc_boot(s, e, np.arange(4), 4, np)['auroc'] == 1.0
print('cs2 eval stats synthetic passed')
