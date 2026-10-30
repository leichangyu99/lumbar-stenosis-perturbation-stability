"""CS20261015-CPU (results/conformal_20261013/PLAN_UNIFIED.md): temperature + ordinal conformal base sets (B) with
severe rule-out add-on (R0 clean, R1 geometric worst case, primary), calibrated once on the validation set
(model_selection 197) and evaluated on the internal test set (671).  Patient bootstrap 10000 for CIs.  Local CPU."""
import hashlib, json
from pathlib import Path

import numpy as np

import cs4_conformal as C
import cs5_severe as S5

HERE = Path(__file__).resolve().parent
VAL = HERE / 'mirror/val'
OUT = C.OUT
B_BOOT, SEED = 10000, 20261021
RULES = ('B', 'R0', 'R1')


def load_val(views):
    lock = json.loads((VAL / 'PREDICTIONS_LOCK.json').read_text()); f = VAL / 'PRED_validation.npz'
    assert C.sha(f) == lock['validation'], 'locked validation prediction hash differs'
    d = np.load(f); names = [str(n) for n in d['models']]; assert [str(v) for v in d['views']] == views
    arrays = {a: d['probs'][[k for k, n in enumerate(names) if n.startswith(a + '_')]].mean(0) for a in C.ARCHS}
    assert all(sum(n.startswith(a + '_') for n in names) == 3 for a in C.ARCHS)
    return arrays, [str(s) for s in d['patients']]


def calibrate(pv, yv, I, geo):
    """Fixed calibration on the validation set: temperature, base threshold, severe thresholds."""
    kv = yv >= 0; T = S5.fit_T(pv[:, I][kv], yv[kv]); PT = S5.temper(pv, T); ST, _ = C.ordinal_scores(PT)
    sy = np.take_along_axis(ST[:, I], np.clip(yv, 0, 2)[..., None], -1)[..., 0]
    sev = 1 - PT[..., 2]; sm = yv == 2
    return {'T': T, 'q_base': S5.qhat(sy[kv]), 'tau': {'R0': S5.qhat(sev[:, I][sm]), 'R1': S5.qhat(sev[:, geo].max(1)[sm])},
            'n_known_levels': int(kv.sum()), 'n_severe_levels': int(sm.sum())}


def build_sets(pt, cal, rule):
    PT = S5.temper(pt, cal['T']); ST, MT = C.ordinal_scores(PT); ins = S5.sets(ST, MT, cal['q_base'])
    if rule != 'B':
        ins[..., 2] |= (1 - PT[..., 2]) <= cal['tau'][rule]; ins[..., 1] |= ins[..., 0] & ins[..., 2]
    return ins, MT


def per_patient(ins, MT, Y, I):
    """Per-patient numerators/denominators so that every metric is a ratio of patient sums (bootstrap-ready)."""
    known = Y >= 0; yc = np.clip(Y, 0, 2); nv = ins.shape[1]
    cov = np.take_along_axis(ins, yc[:, None, :, None].repeat(nv, 1), -1)[..., 0]
    sv = Y == 2; nonsev = known & (Y < 2) & (MT[:, I] < 2); si = ins[:, I].sum(-1) == 1
    return {'cov': ((cov & known[:, None]).sum(-1), known.sum(-1)),                     # [n,27], [n]
            'sev': ((cov & sv[:, None]).sum(-1), sv.sum(-1)),
            'size': (ins.sum(-1).sum(-1), np.full(len(Y), 5)),
            'single': (si.sum(-1), np.full(len(Y), 5)),
            'single_acc': ((si & known & (MT[:, I] == Y)).sum(-1), (si & known).sum(-1)),
            'false_sev': ((ins[:, I, :, 2] & nonsev).sum(-1), nonsev.sum(-1))}


def ratio(num, den, W=None):
    if W is None: return num.sum(0) / max(1, den.sum())
    d = W @ den.astype(float); return (W @ num.astype(float)) / np.maximum(d, 1)[:, None] if num.ndim == 2 else (W @ num.astype(float)) / np.maximum(d, 1)


def ci(boot): return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def main():
    P, ids, views = C.load_preds(); Y = C.labels(ids); PV, vids = load_val(views); YV = C.labels(vids)
    assert not set(ids) & set(vids), 'validation and test overlap'
    vi = {v: i for i, v in enumerate(views)}; I = vi['I']
    geo = [i for i, v in enumerate(views) if v == 'I' or v.startswith(('T_', 'R_', 'S_'))]; assert len(geo) == 19
    other = [i for i in range(len(views)) if i not in geo]
    W = np.random.default_rng(SEED).multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=B_BOOT).astype(np.float32)
    res = {'plan_sha256': C.sha(OUT / 'PLAN_UNIFIED.md'), 'alpha': S5.ALPHA, 'n_validation': len(vids), 'n_test': len(ids), 'bootstrap': B_BOOT,
           'seed': SEED, 'views': views, 'geometric_views': [views[i] for i in geo], 'validation_lock': json.loads((VAL / 'PREDICTIONS_LOCK.json').read_text()), 'arch': {}}
    for a in C.ARCHS:
        cal = calibrate(PV[a], YV, I, geo); A = {'calibration': cal}
        for m in RULES:
            ins, MT = build_sets(P[a], cal, m); pp = per_patient(ins, MT, Y, I)
            cov = ratio(*pp['cov']); sev = ratio(*pp['sev']); sz = ratio(*pp['size'])
            bsev = ratio(*pp['sev'], W); bcov = ratio(*pp['cov'], W); bsz = ratio(*pp['size'], W)
            r = {'overall_by_view': dict(zip(views, cov.round(4).tolist())), 'severe_by_view': dict(zip(views, sev.round(4).tolist())),
                 'overall_min_view': float(cov.min()), 'overall_min_view_ci95': ci(bcov.min(1)),
                 'severe_min_geometric': float(sev[geo].min()), 'severe_min_geometric_view': views[geo[int(sev[geo].argmin())]],
                 'severe_min_geometric_ci95': ci(bsev[:, geo].min(1)), 'severe_identity': float(sev[I]), 'severe_identity_ci95': ci(bsev[:, I]),
                 'severe_min_nongeometric': float(sev[other].min()), 'severe_min_nongeometric_view': views[other[int(sev[other].argmin())]],
                 'size_identity': float(sz[I]), 'size_identity_ci95': ci(bsz[:, I]), 'size_max_view': float(sz.max())}
            for k in ('single', 'single_acc', 'false_sev'):
                num, den = pp[k]; r[k + '_identity'] = float(num.sum() / max(1, den.sum())); r[k + '_identity_ci95'] = ci(ratio(num, den, W))
            A[m] = r
        A['H1'] = bool(A['R1']['severe_min_geometric'] >= .895); A['H3'] = bool(A['R1']['size_identity'] <= 2.0)
        A['safety_overall'] = bool(A['R1']['overall_min_view'] >= .895); A['R0_geo_ok'] = bool(A['R0']['severe_min_geometric'] >= .895)
        res['arch'][a] = A
        print(a, 'T=%.2f' % cal['T'], ' | '.join('%s sevGeo=%.3f sevNon=%.3f covMin=%.3f size=%.2f single=%.2f falseSev=%.3f' % (
            m, A[m]['severe_min_geometric'], A[m]['severe_min_nongeometric'], A[m]['overall_min_view'], A[m]['size_identity'],
            A[m]['single_identity'], A[m]['false_sev_identity']) for m in RULES), flush=True)
    allA = res['arch'].values()
    res.update(H1_passed=all(x['H1'] for x in allA), H3_passed=all(x['H3'] for x in allA), safety_passed=all(x['safety_overall'] for x in allA),
               R0_sufficient=all(x['R0_geo_ok'] for x in allA), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out = OUT / 'UNIFIED_RESULT.json'; assert not out.exists()
    out.write_text(json.dumps(res, indent=1) + '\n', encoding='utf-8')
    print('H1', res['H1_passed'], 'H3', res['H3_passed'], 'safety', res['safety_passed'], 'R0 sufficient', res['R0_sufficient'])


if __name__ == '__main__':
    main()
