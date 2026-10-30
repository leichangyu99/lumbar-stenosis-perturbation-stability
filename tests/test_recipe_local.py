"""Local smoke test of the frozen recipe (all four modes) on synthetic resident tensors, 1 epoch."""
import sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import cs_common as C
C.WEIGHTS = HERE.parent / 'weights/resnet18-f37072fd.pth'; C.HELPERS = HERE.parent; C.EPOCHS = 2
g = torch.Generator().manual_seed(0); ids = ['P%03d' % i for i in range(45)]
gI = {}; gV = {}; labels = {}
for i, s in enumerate(ids):
    m = torch.randn(5, 3, 256, 14, 14, generator=g).abs(); k = torch.rand(5, 3, generator=g) > .2
    if i == 3: k[:] = False
    gI[s] = (m.cuda(), k.cuda()); gV[s] = ((m + .1 * torch.randn(m.shape, generator=g)).cuda(), k.cuda())
    labels[s] = [int(x) for x in torch.randint(-1, 3, (5,), generator=g)]
R = C.Recipe(torch, np); res = {}
for mode in ('STD', 'AUG', 'CONS', 'SHUF'):
    net, pri, info = R.fit(ids, labels, gI, gV, mode, 20261012, lambda: None)
    p = R.predict(net, ids, lambda s: (gI[s][0], gI[s][1]), pri)
    assert all(np.allclose(v[0].sum(-1), 1) for v in p.values())
    res[mode] = (info['history'][-1]['train_loss'], info['stats'])
    print(mode, res[mode])
assert res['CONS'][1]['cons_pairs'] > 0 and res['SHUF'][1]['shuf_single_patient_microbatches'] > 0  # 45 % 32 = 13 -> tail microbatch of 1
assert C.sattolo(4, 5) != list(range(4))
# V == I: CONS adds only the dropout-induced JS on top of the identical AUG forwards
gV2 = {s: gI[s] for s in ids}
a = R.fit(ids[:8], labels, gI, gV2, 'AUG', 1, lambda: None)[2]['history'][0]['train_loss']
c = R.fit(ids[:8], labels, gI, gV2, 'CONS', 1, lambda: None)[2]['history'][0]['train_loss']
print('AUG vs CONS (V==I):', a, c); assert 0 <= c - a < 0.01  # head Dropout(.1) is part of the frozen recipe: JS(V==I) small, positive
print('recipe smoke passed')
