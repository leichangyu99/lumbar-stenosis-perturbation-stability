"""CS20261013-SRO (results/conformal_20261013/PLAN_RULEOUT.md): severe rule-out add-on to the temperature-scaled
marginal ordinal conformal set; severe threshold calibrated on clean (R0), geometric-family worst case (R1, primary)
or full 27-view worst case (R2).  Local CPU, 500 splits."""
import hashlib, json
from pathlib import Path

import numpy as np

import cs4_conformal as C
import cs5_severe as S5

OUT = C.OUT
R, SEED = 500, 20261016
GEO_PREFIX = ('I', 'T_', 'R_', 'S_')
RULES = ('B', 'R0', 'R1', 'R2')


def main():
    P, ids, views = C.load_preds(); Y = C.labels(ids); n = len(ids); vi = {v: i for i, v in enumerate(views)}; I = vi['I']
    geo = [i for i, v in enumerate(views) if v == 'I' or v.startswith(GEO_PREFIX[1:])]; assert len(geo) == 19
    known = Y >= 0; yc = np.clip(Y, 0, 2); rng = np.random.default_rng(SEED); splits = [rng.permutation(n) for _ in range(R)]
    res = {'alpha': S5.ALPHA, 'splits': R, 'geometric_views': [views[i] for i in geo], 'views': views, 'arch': {}}
    for a in C.ARCHS:
        acc = {m: {k: [] for k in ('cov', 'sev', 'size', 'single', 'single_acc', 'false_sev', 'tau')} for m in RULES}
        for r, perm in enumerate(splits):
            tf, cal, te = perm[:100], perm[100:335], perm[335:]
            kt = known[tf]; T = S5.fit_T(P[a][tf][:, I][kt], Y[tf][kt]); PT = S5.temper(P[a], T); ST, MT = C.ordinal_scores(PT)
            kc = known[cal]
            sy = np.take_along_axis(ST[cal], yc[cal][:, None, :, None].repeat(27, 1), -1)[..., 0]
            base = S5.sets(ST[te], MT[te], S5.qhat(sy[:, I][kc]))                                   # M2 base sets [t,27,5,3]
            sev_s = 1 - PT[cal][..., 2]                                                               # [c,27,5]
            sevmask = Y[cal] == 2
            taus = {'R0': S5.qhat(sev_s[:, I][sevmask]), 'R1': S5.qhat(sev_s[:, geo].max(1)[sevmask]), 'R2': S5.qhat(sev_s.max(1)[sevmask])}
            for m in RULES:
                ins = base.copy()
                if m != 'B':
                    add = (1 - PT[te][..., 2]) <= taus[m]; ins[..., 2] |= add; ins[..., 1] |= ins[..., 0] & ins[..., 2]; acc[m]['tau'].append(taus[m])
                ke = known[te]; cov = np.take_along_axis(ins, yc[te][:, None, :, None].repeat(27, 1), -1)[..., 0]
                sv = Y[te] == 2; nonsev = ke & (Y[te] < 2)
                acc[m]['cov'].append((cov & ke[:, None]).sum((0, 2)) / ke.sum())
                acc[m]['sev'].append((cov & sv[:, None]).sum((0, 2)) / max(1, sv.sum()))
                acc[m]['size'].append(ins.sum(-1).mean((0, 2)))
                si = ins[:, I].sum(-1) == 1; acc[m]['single'].append(si.mean()); ok = si & ke
                acc[m]['single_acc'].append((MT[te][:, I] == Y[te])[ok].mean() if ok.any() else np.nan)
                acc[m]['false_sev'].append(ins[:, I, :, 2][nonsev & (MT[te][:, I] < 2)].mean())
        def s(x): x = np.asarray(x, dtype=float); return {'mean': float(np.nanmean(x)), 'p2.5': float(np.nanpercentile(x, 2.5)), 'p97.5': float(np.nanpercentile(x, 97.5))} if np.isfinite(x).any() else None
        A = {}
        for m in RULES:
            cov = np.mean(acc[m]['cov'], 0); sev = np.mean(acc[m]['sev'], 0); sz = np.mean(acc[m]['size'], 0)
            other = [i for i in range(27) if i not in geo]
            A[m] = {'overall_min_view': float(cov.min()), 'severe_by_view': dict(zip(views, sev.round(4).tolist())),
                    'severe_min_geometric': float(sev[geo].min()), 'severe_min_geometric_view': views[geo[int(sev[geo].argmin())]],
                    'severe_min_nongeometric': float(sev[other].min()), 'severe_min_nongeometric_view': views[other[int(sev[other].argmin())]],
                    'size_identity': float(sz[I]), 'size_max_view': float(sz.max()), 'singleton_rate_identity': s(acc[m]['single']),
                    'singleton_accuracy_identity': s(acc[m]['single_acc']), 'false_severe_inclusion_identity': s(acc[m]['false_sev']), 'tau_score': s(acc[m]['tau']) if m != 'B' else None}
        A['H1'] = bool(A['R1']['severe_min_geometric'] >= .895); A['R0_geo_ok'] = bool(A['R0']['severe_min_geometric'] >= .895)
        A['H3'] = bool(A['R1']['size_identity'] <= 2.0); A['safety_overall'] = bool(A['R1']['overall_min_view'] >= .895)
        res['arch'][a] = A
        print(a, ' | '.join('%s sevGeo=%.3f sevNon=%.3f size=%.2f single=%.2f falseSev=%.3f' % (m, A[m]['severe_min_geometric'], A[m]['severe_min_nongeometric'], A[m]['size_identity'],
                                                                                              A[m]['singleton_rate_identity']['mean'], A[m]['false_severe_inclusion_identity']['mean']) for m in RULES))
    allA = res['arch'].values()
    res.update(H1_passed=all(x['H1'] for x in allA), H3_passed=all(x['H3'] for x in allA), safety_passed=all(x['safety_overall'] for x in allA),
               R0_sufficient=all(x['R0_geo_ok'] for x in allA))
    res['H2_interpretation'] = ('geometric worst-case calibration necessary' if res['H1_passed'] and not res['R0_sufficient'] else
                                'clean calibration already sufficient' if res['R0_sufficient'] else 'H1 not met')
    res.update(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), plan_sha256=C.sha(OUT / 'PLAN_RULEOUT.md'))
    (OUT / 'SRO_RESULT.json').write_text(json.dumps(res, indent=1) + '\n', encoding='utf-8')
    print('H1', res['H1_passed'], 'H3', res['H3_passed'], 'safety', res['safety_passed'], 'R0 sufficient', res['R0_sufficient'], '->', res['H2_interpretation'])


if __name__ == '__main__':
    main()
