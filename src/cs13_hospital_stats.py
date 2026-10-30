"""HOSPITAL-257 statistics.  part1 (label-free): 1px flip / dose / families on the merged 256 localised patients
(HOSP-001..100 locked predictions + HOSP-101..257 batch 2).  part2 (labels): written after the prediction lock."""
import hashlib, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs7_external_stats as XS  # noqa: E402

P1 = Path('data/hospital/work/external_dataset3')
P2 = Path('data/hospital/work/external_dataset3_batch2')
OUT = HERE.parent.parent / 'results/external_257_v1cross_20261021'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_merged():
    parts = []
    for w in (P1, P2):
        lock = json.loads((w / 'PREDICTIONS_LOCK.json').read_text()); f = w / 'PRED_dataset3.npz'; assert sha(f) == lock['predictions_sha256'], w
        parts.append(np.load(f))
    assert list(parts[0]['models']) == list(parts[1]['models']) and list(parts[0]['keys']) == list(parts[1]['keys'])
    probs = np.concatenate([p['probs'] for p in parts], 1); vis = np.concatenate([p['visible'] for p in parts], 0)
    ids = [str(s) for p in parts for s in p['patients']]; assert len(ids) == len(set(ids))
    return probs, vis, [str(n) for n in parts[0]['models']], [str(k) for k in parts[0]['keys']], ids


def flips(probs, vis, names, keys, sel, seed=20261021):
    vn = [k.split('|')[0] for k in keys if k.endswith('|None')]; nv = len(vn); vis0 = vis[sel][:, vn.index('I')]; res = {}; idx = None
    for s in XS.SYSTEMS:
        P = probs[[k for k, n in enumerate(names) if n.startswith(s + '_')]][:, sel][:, :, :nv].mean(0)
        pv, fam, comp = XS.family_flips(P, vn, vis0)
        if idx is None: idx = np.random.default_rng(seed).integers(0, int(comp.sum()), (10000, int(comp.sum())))
        def ci(x): m = x[idx].mean(1); return {'mean': float(x.mean()), 'ci95': [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}
        t1 = (pv['T_1_col'][0] + pv['T_1_row'][0]) / 2; dj = (pv['T_1_col'][1] + pv['T_1_row'][1]) / 2 - (pv['T_0.25_col'][1] + pv['T_0.25_row'][1]) / 2
        r = {'T1px_flip': ci(t1), 'dose_JS1_minus_JS025': ci(dj), 'family_flip_means': fam, 'identity_zero': bool(pv['I'][0].max() == 0)}
        r['primary_passed'] = bool(r['identity_zero'] and r['T1px_flip']['ci95'][0] > 0); r['dose_trend'] = bool(r['dose_JS1_minus_JS025']['ci95'][0] > 0)
        res[s] = r
    return res


def part1():
    probs, vis, names, keys, ids = load_merged(); n = len(ids)
    allsel = np.arange(n); newsel = np.array([i for i, s in enumerate(ids) if int(s[4:]) > 100])
    res = {'patients_localised': n, 'patients_total': 257, 'excluded': [{'study_id': 'HOSP-185', 'reason': 'inconsistent_or_unknown_series_frame (frozen locator geometry rule)'}],
           'all': flips(probs, vis, names, keys, allsel), 'new_batch_only': flips(probs, vis, names, keys, newsel), 'labels_read': False}
    res['all_7of7'] = all(v['primary_passed'] for v in res['all'].values()); res['new_7of7'] = all(v['primary_passed'] for v in res['new_batch_only'].values())
    res.update(plan_sha256=sha(OUT / 'PLAN.md'), source_sha256=sha(__file__))
    (OUT / 'PART1_RESULT.json').write_text(json.dumps(res, indent=1) + '\n', encoding='utf-8')
    for k in ('all', 'new_batch_only'):
        print('==', k); [print(s, 'T1 %.4f %s dose %s pass %s' % (r['T1px_flip']['mean'], [round(x, 4) for x in r['T1px_flip']['ci95']], r['dose_trend'], r['primary_passed'])) for s, r in res[k].items()]
    print('7/7 all', res['all_7of7'], 'new', res['new_7of7'])


if __name__ == '__main__':
    part1()
