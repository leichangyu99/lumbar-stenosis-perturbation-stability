"""Local checks: raw_crop == E.crop channel 0 before normalisation; all 4 graders forward/backward under bf16."""
import json, sys, time
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import cs_engine as E, cs2_engine as E2, cs3_arch as A
W = HERE.parent / 'weights_ma'; DATA = Path('data/LumbarDISC/dataset'); PRED = HERE.parent.parent / 'runs/local/M1_sag_export_s20261007/attempt_0/predictions.json'
rows = {r['study_id']: r for r in json.loads(PRED.read_text(encoding='utf-8'))['predictions'] if r['role'] == 'diagnostic_train'}
sid = sorted(rows)[1]; P = E2.Patient(DATA, rows[sid], sid)
views = E.evaluation_views(); pv = A.patient_views(P, views)
for name in ('I', 'T_1_col', 'R_+4', 'S_0.9', 'N_0.05', 'W_5_95'):
    v = {x['name']: x for x in views}[name]; ref = P.crops(v); raw, mask = pv[name]
    for loc, c in ref.items():
        assert torch.allclose(raw[loc], c[0] * .229 + .485, atol=1e-6), (name, loc)
    assert int(mask.sum()) == len(ref)
print('raw crops consistent; D_edge slices', int(pv['D_edge'][1].sum()), 'of', int(pv['I'][1].sum()))
g = torch.Generator(device='cuda').manual_seed(1)
x = torch.stack([pv['I'][0], pv['T_1_col'][0]]).cuda(); m = torch.stack([pv['I'][1], pv['D_edge'][1]]).cuda()
xa = A.augment(x, g); assert xa.shape == x.shape and float(xa.min()) >= 0 and float(xa.max()) <= 1
for arch, (enc, agg) in A.ARCHS.items():
    net = A.Grader(arch, W / A.WEIGHT_FILES[enc][0]).cuda().train(); t = time.time()
    with torch.autocast('cuda', dtype=torch.bfloat16): z = net(xa, m)
    z.float().sum().backward(); torch.cuda.synchronize()
    print(arch, enc, agg, tuple(z.shape), 'params %.1fM' % (sum(p.numel() for p in net.parameters()) / 1e6), 's %.2f' % (time.time() - t), 'maxmem %.2fGB' % (torch.cuda.max_memory_allocated() / 2**30))
    del net; torch.cuda.empty_cache()
print('cs3 arch local passed')
