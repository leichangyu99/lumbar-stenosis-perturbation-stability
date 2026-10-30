"""Analysis of three mitigation strategies (not pre-registered) on existing locked predictions:
P1 local recalibration of the severe rule-out on hospital data (repeated 50/50 patient splits);
P2 ensembling (single seed vs 3-seed vs 6-architecture) and shift-TTA against 1px shift flips;
P3 combined risk score (confidence + perturbation sensitivity + model disagreement) fitted on validation."""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs4_conformal as C  # noqa: E402
import cs5_severe as S5  # noqa: E402
import cs12_unified_conformal as U  # noqa: E402
import cs13_hospital_stats as V  # noqa: E402
import cs13_hospital_labels as V2  # noqa: E402
from cs_metrics import grade, js_bits  # noqa: E402

OUT = HERE.parent.parent / 'results/mitigation_analysis_20261021'
ARCHS = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']
RNG = np.random.default_rng(20261021)


def ci(x): return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def flip_rate(P, vn, v0, views=('T_1_col', 'T_1_row')):
    """per-patient mean flip over original-visible levels for the given views."""
    I = vn.index('I'); out = []
    for v in views:
        f = (grade(P[:, I]) != grade(P[:, vn.index(v)])).astype(float); out.append((f * v0).sum(-1) / np.maximum(v0.sum(-1), 1))
    return np.mean(out, 0)


def paired(a, b, B=5000):
    n = len(a); idx = RNG.integers(0, n, (B, n)); d = (a - b)[idx].mean(1)
    return {'a': float(a.mean()), 'b': float(b.mean()), 'diff': float((a - b).mean()), 'ci95': ci(d), 'significant': bool(ci(d)[1] < 0 or ci(d)[0] > 0)}


def load_hospital():
    probs, vis, names, keys, ids = V.load_merged(); Y = V2.read_labels(ids)[0]
    vn = [k.split('|')[0] for k in keys if k.endswith('|None')]; nv = len(vn)
    return probs, vis, names, keys, vn, nv, ids, Y


def p1_local_recalibration(probs, vis, names, vn, nv, Y, R=500):
    """Recalibrate temperature, base threshold and rule-out tau on a hospital half; evaluate on the other half."""
    uni = json.loads((U.OUT / 'UNIFIED_RESULT.json').read_text()); I = vn.index('I'); geo = [vn.index(v) for v in uni['geometric_views']]
    n = len(Y); res = {}
    for a in ARCHS:
        P = probs[[k for k, nm in enumerate(names) if nm.startswith(a + '_')]][:, :, :nv].mean(0); rows = {'transfer': [], 'local_R1': [], 'local_R0': []}
        for r in range(R):
            perm = RNG.permutation(n); cal, te = perm[:n // 2], perm[n // 2:]
            loc = U.calibrate(P[cal], Y[cal], I, geo)
            for tag, cc, rule in (('transfer', uni['arch'][a]['calibration'], 'R1'), ('local_R1', loc, 'R1'), ('local_R0', loc, 'R0')):
                ins, MT = U.build_sets(P[te], cc, rule); pp = U.per_patient(ins, MT, Y[te], I)
                sev = U.ratio(*pp['sev']); sz = U.ratio(*pp['size']); fs = pp['false_sev']; cov = U.ratio(*pp['cov'])
                rows[tag].append([sev[I], sev[geo].min(), sz[I], fs[0].sum() / max(1, fs[1].sum()), cov.min()])
        res[a] = {t: dict(zip(['severe_identity', 'severe_min_geometric', 'size_identity', 'false_severe', 'overall_min_view'],
                              [{'mean': float(np.mean(c)), 'p2.5': float(np.percentile(c, 2.5)), 'p97.5': float(np.percentile(c, 97.5))} for c in np.array(v).T])) for t, v in rows.items()}
    return res


def p2_ensembles(probs, vis, names, vn, nv):
    """Single seed vs 3-seed vs 6-architecture ensemble, 1px flip, hospital 256 (+ internal 671 for A0-A5)."""
    v0 = vis[:, vn.index('I')]; out = {'hospital': {}}
    single = {}; seed3 = {}
    for a in ARCHS:
        ks = [k for k, nm in enumerate(names) if nm.startswith(a + '_')]
        single[a] = np.mean([flip_rate(probs[k][:, :nv], vn, v0) for k in ks], 0); seed3[a] = flip_rate(probs[ks][:, :, :nv].mean(0), vn, v0)
        out['hospital'][a] = {'single_vs_3seed': paired(seed3[a], single[a])}
    all18 = probs[[k for k, nm in enumerate(names) if nm[:2] in ARCHS]][:, :, :nv].mean(0); arch6 = flip_rate(all18, vn, v0)
    best3 = min(ARCHS, key=lambda a: seed3[a].mean())
    out['hospital']['6arch_ensemble'] = {'flip': float(arch6.mean()), 'vs_mean_single': paired(arch6, np.mean([single[a] for a in ARCHS], 0)), 'vs_best_3seed_' + best3: paired(arch6, seed3[best3])}
    # internal 671 (A0-A5)
    P, ids, views = C.load_preds(); vis_i = []
    m = HERE / 'mirror'
    vis_i = np.concatenate([np.load(m / 'maeval' / ('PRED_%s.npz' % t))['visible'] for t in ('confirmation', 'policy')], 0)[:, views.index('I')]
    raw = {}
    for job in ('maeval', 'maevalC'):
        parts = [np.load(m / job / ('PRED_%s.npz' % t)) for t in ('confirmation', 'policy')]; nm = list(parts[0]['models']); pr = np.concatenate([p['probs'] for p in parts], 1)
        for k, n_ in enumerate(nm): raw[str(n_)] = pr[k]
    s_i = {a: np.mean([flip_rate(raw[n_], views, vis_i) for n_ in raw if n_.startswith(a + '_')], 0) for a in ARCHS}
    e_i = {a: flip_rate(np.mean([raw[n_] for n_ in raw if n_.startswith(a + '_')], 0), views, vis_i) for a in ARCHS}
    a6 = flip_rate(np.mean(list(raw.values()), 0), views, vis_i)
    out['internal671'] = {a: {'single_vs_3seed': paired(e_i[a], s_i[a])} for a in ARCHS}
    out['internal671']['6arch_ensemble'] = {'flip': float(a6.mean()), 'vs_mean_single': paired(a6, np.mean([s_i[a] for a in ARCHS], 0))}
    # shift-TTA on policy474 (CS20261013 eval: STD and AUGMIX, 8 geometric jitters per view)
    d = np.load(m / 'v2evalpol/PREDICTIONS.npz'); keys = [str(k) for k in d['keys']]; nmz = [str(x) for x in d['models']]; pz = d['probs']; vz = d['visible']
    ki = {k: i for i, k in enumerate(keys)}; v0p = vz[:, ki['I|None']]
    for sc in ('STD', 'AUGMIX'):
        e = pz[[k for k, x in enumerate(nmz) if x.startswith(sc + '_')]].mean(0)
        plain = np.stack([e[:, ki['%s|None' % v]] for v in ('I', 'T_1_col', 'T_1_row')], 1)
        tta = np.stack([np.mean([e[:, ki['%s|%d' % (v, j)]] for j in range(8)], 0) for v in ('I', 'T_1_col', 'T_1_row')], 1)
        vv = ['I', 'T_1_col', 'T_1_row']; out.setdefault('policy474_shiftTTA', {})[sc] = paired(flip_rate(tta, vv, v0p), flip_rate(plain, vv, v0p))
    return out


def risk_features(Pm, vn, names_arch):
    """level-wise features from per-model probabilities Pm [models, n, views, 5, 3]."""
    I = vn.index('I'); geo = [i for i, v in enumerate(vn) if v.startswith(('T_', 'R_', 'S_'))]
    ens = Pm.mean(0); p = ens[:, I]
    conf = 1 - p.max(-1)
    sens = np.mean([js_bits(p, ens[:, g]) for g in geo], 0)
    disagree = np.mean([js_bits(p, Pm[k][:, I]) for k in range(len(Pm))], 0)
    return np.stack([conf, sens, disagree], -1), p


def p3_risk_score():
    from sklearn.linear_model import LogisticRegression
    vd = np.load(U.VAL / 'PRED_validation.npz'); vnames = [str(x) for x in vd['models']]; views = [str(v) for v in vd['views']]
    Yv = C.labels([str(s) for s in vd['patients']]); Fv, pv = risk_features(vd['probs'], views, vnames)
    kv = (Yv >= 0); errv = (grade(pv) != Yv)
    lr = LogisticRegression().fit(Fv[kv], errv[kv])
    res = {'coef_conf_sens_disagree': lr.coef_[0].tolist(), 'fitted_on': 'validation 197 (A0-A5 18 models, 27 views)'}
    P, ids, _ = C.load_preds(); Yi = C.labels(ids)
    m = HERE / 'mirror'; raw = []
    for job in ('maeval', 'maevalC'):
        parts = [np.load(m / job / ('PRED_%s.npz' % t)) for t in ('confirmation', 'policy')]; raw.append(np.concatenate([p['probs'] for p in parts], 1))
    Pm = np.concatenate(raw, 0); Fi, pi = risk_features(Pm, views, None)
    probs, vis, names, keys, vn, nv, hids, Yh = load_hospital()
    Ph = probs[[k for k, nm in enumerate(names) if nm[:2] in ARCHS]][:, :, :nv]; Fh, ph = risk_features(Ph, vn, None)
    for tag, F, p, Y in (('internal671', Fi, pi, Yi), ('hospital256', Fh, ph, Yh)):
        k = Y >= 0; err = (grade(p) != Y); n = len(Y); pid = np.repeat(np.arange(n)[:, None], 5, 1)
        combo = lr.decision_function(F[k]); r = {}
        for nm_, s in (('confidence', F[k][:, 0]), ('sensitivity', F[k][:, 1]), ('disagreement', F[k][:, 2]), ('combined', combo)):
            r[nm_] = V2.auroc_boot(s, err[k], pid[k], n)
        # paired bootstrap of AUROC difference combined - confidence
        rng = np.random.default_rng(20261021); dd = []
        for _ in range(2000):
            w = np.bincount(rng.integers(0, n, n), minlength=n)[pid[k]].astype(float)
            dd.append(V2.auroc_w(combo, err[k], w) - V2.auroc_w(F[k][:, 0], err[k], w))
        r['combined_minus_confidence'] = {'diff': r['combined']['auroc'] - r['confidence']['auroc'], 'ci95': ci(np.array(dd)), 'significant': bool(np.percentile(dd, 2.5) > 0)}
        # risk-coverage: error rate among the most confident X% (auto-handled) levels
        for nm_, s in (('confidence', F[k][:, 0]), ('combined', combo)):
            o = np.argsort(s); e = err[k][o]; r['risk_at_coverage_' + nm_] = {str(c): float(e[:int(len(e) * c)].mean()) for c in (0.5, 0.7, 0.9, 1.0)}
        res[tag] = r
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    probs, vis, names, keys, vn, nv, ids, Y = load_hospital()
    res = {'note': 'not pre-registered'}
    res['P1_local_recalibration_hospital'] = p1_local_recalibration(probs, vis, names, vn, nv, Y)
    res['P2_ensembles_and_shiftTTA'] = p2_ensembles(probs, vis, names, vn, nv)
    res['P3_combined_risk_score'] = p3_risk_score()
    (OUT / 'MITIGATION_RESULT.json').write_text(json.dumps(res, indent=1) + '\n', encoding='utf-8')
    p1 = res['P1_local_recalibration_hospital']
    for a in ARCHS:
        print('P1', a, ' | '.join('%s sevI %.3f sevGeo %.3f size %.2f falseSev %.3f' % (t, p1[a][t]['severe_identity']['mean'], p1[a][t]['severe_min_geometric']['mean'], p1[a][t]['size_identity']['mean'], p1[a][t]['false_severe']['mean']) for t in p1[a]))
    print('P2', json.dumps(res['P2_ensembles_and_shiftTTA'])[:3000])
    p3 = res['P3_combined_risk_score']; print('P3 coef', p3['coef_conf_sens_disagree'])
    for t in ('internal671', 'hospital256'):
        print('P3', t, {k: (round(v['auroc'], 3), [round(x, 3) for x in v['ci95']]) for k, v in p3[t].items() if 'auroc' in v}, 'diff', p3[t]['combined_minus_confidence'], 'risk', p3[t]['risk_at_coverage_confidence'], p3[t]['risk_at_coverage_combined'])


if __name__ == '__main__':
    main()
