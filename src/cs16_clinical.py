"""Descriptive add-ons (not pre-registered):
C1 clinical boundary of 1px-shift flips (which grade transitions; severe-status and >=moderate-status changes);
C2 hospital accuracy sensitivity analysis excluding the 7 cases with locator wrong-level errors."""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs15_ensemble as E  # noqa: E402
from cs_common import rps_patient, severe_miss  # noqa: E402
from cs_metrics import grade  # noqa: E402

OUT = HERE.parent.parent / 'results/clinical_addons_20261021'
GT17 = HERE.parent.parent / 'results/locator_validation_20261021/AI_GT17_BREAKDOWN.json'


def transitions(P, vn, v0):
    I = vn.index('I'); g0 = grade(P[:, I]); res = {'flips': 0, 'normal_moderate': 0, 'moderate_severe': 0, 'normal_severe': 0, 'levels': 0}
    pat_sev = np.zeros(len(P), bool); pat_mod = np.zeros(len(P), bool)
    for v in ('T_1_col', 'T_1_row'):
        g1 = grade(P[:, vn.index(v)]); f = (g0 != g1) & v0; res['levels'] += int(v0.sum()); res['flips'] += int(f.sum())
        pair = {frozenset((0, 1)): 'normal_moderate', frozenset((1, 2)): 'moderate_severe', frozenset((0, 2)): 'normal_severe'}
        for a, b in zip(g0[f], g1[f]): res[pair[frozenset((int(a), int(b)))]] += 1
        pat_sev |= (((g0 == 2) != (g1 == 2)) & v0).any(-1); pat_mod |= (((g0 >= 1) != (g1 >= 1)) & v0).any(-1)
    res['flip_rate'] = res['flips'] / res['levels']
    res['severe_status_change_level_rate'] = (res['moderate_severe'] + res['normal_severe']) / res['levels']
    res['ge_moderate_status_change_level_rate'] = (res['normal_moderate'] + res['normal_severe']) / res['levels']
    res['patients_with_severe_status_change'] = float(pat_sev.mean()); res['patients_with_ge_moderate_status_change'] = float(pat_mod.mean())
    return res


def accuracy(P, Y, vn, keep):
    I = vn.index('I'); p = P[:, I][keep]; y = Y[keep]; k = (y >= 0).any(-1)
    rp = np.array([rps_patient(y[i], p[i]) for i in range(len(y)) if k[i]])
    return {'patients': int(keep.sum()), 'RPS': float(rp.mean()), 'level_accuracy': float((p.argmax(-1)[y >= 0] == y[y >= 0]).mean()),
            'severe_recall': float((p.argmax(-1)[y == 2] == 2).mean()), 'severe_miss_patients': int(sum(severe_miss(y[i], p[i]) for i in range(len(y)) if k[i]))}


def main():
    OUT.mkdir(parents=True, exist_ok=True); res = {'note': 'descriptive'}
    for which in ('internal671', 'hospital256', 'spider'):
        P, v0, vn, Y = E.load(which); r = {'single_models_mean': None}
        singles = [transitions(P[k], vn, v0) for k in range(18)]
        r['single_models_mean'] = {k: float(np.mean([s[k] for s in singles])) for k in singles[0] if not isinstance(singles[0][k], int) or True}
        r['full_ensemble'] = transitions(P.mean(0), vn, v0)
        res[which] = r
    # S0 (primary frozen-prefix ResNet18) on hospital
    import cs13_hospital_stats as V
    probs, vis, names, keys, ids = V.load_merged(); vn = [k.split('|')[0] for k in keys if k.endswith('|None')]
    S0 = probs[[k for k, n in enumerate(names) if n.startswith('S0_')]][:, :, :len(vn)].mean(0); v0 = vis[:, vn.index('I')]
    res['hospital256']['S0'] = transitions(S0, vn, v0)
    # C2 accuracy excluding locator wrong-level cases
    bad = sorted({r['case'] for r in json.loads(GT17.read_text(encoding='utf-8')) if r['wrong_level']})
    P, v0h, vnh, Y = E.load('hospital256'); keep_all = np.ones(len(ids), bool); keep = np.array([s not in bad for s in ids])
    full = P.mean(0)
    res['C2_excluding_wrong_level_cases'] = {'excluded_cases': bad,
        'S0_all': accuracy(S0, Y, vn, keep_all), 'S0_excl': accuracy(S0, Y, vn, keep),
        'ensemble_all': accuracy(full, Y, vnh, keep_all), 'ensemble_excl': accuracy(full, Y, vnh, keep)}
    (OUT / 'CLINICAL_ADDONS.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    for which in ('internal671', 'hospital256', 'spider'):
        for tag in ('single_models_mean', 'full_ensemble') + (('S0',) if which == 'hospital256' else ()):
            t = res[which][tag]; print(which, tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()})
    print(json.dumps(res['C2_excluding_wrong_level_cases'], ensure_ascii=False, indent=0))


if __name__ == '__main__':
    main()
