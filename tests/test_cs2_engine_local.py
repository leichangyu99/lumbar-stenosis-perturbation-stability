"""CS20261013 engine checks on real development DICOMs (CPU): identity path == frozen engine, AugMix and TTA run."""
import json, sys, time
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import cs_engine as E, cs2_engine as E2
DATA = Path('data/LumbarDISC/dataset'); PRED = HERE.parent.parent / 'runs/local/M1_sag_export_s20261007/attempt_0/predictions.json'
rows = {r['study_id']: r for r in json.loads(PRED.read_text(encoding='utf-8'))['predictions'] if r['role'] == 'diagnostic_train'}
torch.set_num_threads(4); _, prefix = E.load_prefix(HERE.parent / 'weights/resnet18-f37072fd.pth', 'cpu')
sid = sorted(rows)[0]; ref = E.build_views(DATA, rows[sid], sid, E.evaluation_views(), prefix, 'cpu')
P = E2.Patient(DATA, rows[sid], sid); enc = lambda c: P.encode(c, prefix, 'cpu'); P.encode = lambda c, pf, device='cuda', _e=P.encode: _e(c, pf, 'cpu')
for name in ('I', 'T_1_col', 'N_0.05', 'W_5_95'):
    v = {x['name']: x for x in E.evaluation_views()}[name]; m, k = P.view(v, prefix)
    assert torch.equal(m, ref[name][0]) and torch.equal(k, ref[name][1]), name
clean = P.crops(E.IDENTITY); t = time.time(); spec = E2.augmix_spec(sid, 1, 1); am, ak, fb = P.augmix(clean, spec, prefix)
assert torch.equal(ak, ref['I'][1]) and torch.isfinite(am).all() and not torch.equal(am, ref['I'][0]); print('augmix s', round(time.time() - t, 2), 'fallback', fb, spec['w'], spec['m'])
assert E2.augmix_spec(sid, 1, 1) == spec and E2.augmix_spec(sid, 1, 2) != spec
zero = {'tx': 0., 'ty': 0., 'theta_deg': 0., 's': 1.}
assert torch.equal(P.view(E2.compose(E.IDENTITY | {'name': 'I'}, zero), prefix)[0], ref['I'][0])
t = time.time(); ev = E2.eval_patient(P, prefix); print('eval_patient views', len(ev), 's', round(time.time() - t, 1))
assert len(ev) == 99 and torch.equal(ev[('D_edge', None)][0], ref['D_edge'][0]) and torch.equal(ev[('D_edge', None)][1], ref['D_edge'][1])
print('cs2 engine local passed')
