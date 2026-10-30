"""HOSPITAL-257 part 2 (labels; run only after both prediction locks): L1 accuracy, L2 reader agreement, L3 AI
instability vs reader disagreement, L4 frozen unified rule-out transfer."""
import csv, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs12_unified_conformal as U  # noqa: E402
import cs13_hospital_stats as V  # noqa: E402
import cs7_external_stats as XS  # noqa: E402
from cs_common import rps_patient, severe_miss  # noqa: E402
from cs_metrics import grade, js_bits  # noqa: E402

LAB = Path('data/hospital/reference_labels')
LEVELS = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
GR = {'正常/轻度': 0, '中度': 1, '重度': 2}
SEED = 20261021


def read_labels(ids):
    ref, A, B = {}, {}, {}
    for d in ('前100例', '20261020_reader_2'):
        for r in csv.DictReader(open(LAB / d / 'parsed/reference_standard.csv', encoding='utf-8-sig')):
            ref[(r['病例编号'], r['节段'])] = int(r['参考标准代码']) if r['参考标准代码'] != '' else -1
        for f, tgt in (('A', A), ('B', B)):
            for r in csv.DictReader(open(LAB / d / ('parsed/%s.csv' % f), encoding='utf-8-sig')):
                tgt[(r['病例编号'], r['节段'])] = GR.get(r['中央椎管狭窄分级'], -1)
    arr = lambda m: np.array([[m.get((s, l), -1) for l in LEVELS] for s in ids])
    return arr(ref), arr(A), arr(B), ref


def auroc_w(score, y, w):
    o = np.argsort(score, kind='mergesort'); s, y, w = score[o], y[o].astype(float), w[o]
    grp = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])]); wp, wn = w * y, w * (1 - y)
    if wp.sum() == 0 or wn.sum() == 0: return np.nan
    gn = np.bincount(grp, weights=wn); below = np.concatenate([[0], np.cumsum(gn)[:-1]])
    return float((wp * (below[grp] + .5 * gn[grp])).sum() / (wp.sum() * wn.sum()))


def auroc_boot(score, y, pid, n):
    pt = auroc_w(score, y, np.ones(len(y))); rng = np.random.default_rng(SEED); vals = []
    for _ in range(10000):
        m = np.bincount(rng.integers(0, n, n), minlength=n); vals.append(auroc_w(score, y, m[pid].astype(float)))
    v = np.array(vals); v = v[np.isfinite(v)]
    return {'auroc': pt, 'ci95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], 'positives': int(y.sum()), 'n': int(len(y))}


def linear_kappa(a, b):
    k = np.zeros((3, 3))
    np.add.at(k, (np.asarray(a), np.asarray(b)), 1)
    n = k.sum(); w = 1 - np.abs(np.subtract.outer(np.arange(3), np.arange(3))) / 2
    po = (w * k).sum() / n; pe = (w * np.outer(k.sum(1), k.sum(0))).sum() / n ** 2
    return float((po - pe) / (1 - pe))


def main():
    probs, vis, names, keys, ids = V.load_merged(); n = len(ids)
    Y, RA, RB, refmap = read_labels(ids)
    vn = [k.split('|')[0] for k in keys if k.endswith('|None')]; nv = len(vn); I = vn.index('I'); jit = [keys.index('I|%d' % m) for m in range(8)]
    ens = {s: probs[[k for k, nm in enumerate(names) if nm.startswith(s + '_')]].mean(0) for s in XS.SYSTEMS}
    known = Y >= 0; rows = np.random.default_rng(SEED).integers(0, n, (10000, n))
    res = {'patients': n, 'reference_levels_known': int(known.sum()), 'grade_counts': {str(g): int((Y == g).sum()) for g in range(3)},
           'unknown_levels': int((~known).sum()), 'severe_patients': int((Y == 2).any(-1).sum()),
           'HOSP_185_excluded_reference_levels': {str(v): sum(1 for (s, l), x in refmap.items() if s == 'HOSP-185' and x == v) for v in (-1, 0, 1, 2)},
           'labels_opened_after_locks': [json.loads((w / 'PREDICTIONS_LOCK.json').read_text())['predictions_sha256'] for w in (V.P1, V.P2)]}
    L1 = {}
    for s in XS.SYSTEMS:
        P = ens[s][:, I]; rp = np.array([rps_patient(Y[i], P[i]) if known[i].any() else np.nan for i in range(n)])
        bm = np.nanmean(rp[rows], 1)
        L1[s] = {'RPS': float(np.nanmean(rp)), 'RPS_ci95': [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                 'recall': {str(c): float(np.mean(P[Y == c].argmax(-1) == c)) for c in range(3)},
                 'severe_miss_patients': int(sum(severe_miss(Y[i], P[i]) for i in range(n) if known[i].any())),
                 'level_accuracy': float(np.mean(P[known].argmax(-1) == Y[known])), 'weighted_kappa_vs_reference': linear_kappa(P[known].argmax(-1), Y[known])}
    res['L1'] = L1
    both = (RA >= 0) & (RB >= 0); bk = []
    for r in rows[:2000]:
        bb = both[r]; bk.append(linear_kappa(RA[r][bb], RB[r][bb]))
    res['L2'] = {'linear_weighted_kappa_A_vs_B': linear_kappa(RA[both], RB[both]), 'ci95': [float(np.percentile(bk, 2.5)), float(np.percentile(bk, 97.5))], 'boot_B': 2000,
                 'levels_both_read': int(both.sum()), 'level_agreement': float(np.mean(RA[both] == RB[both])), 'disagreement_rate': float(np.mean(RA[both] != RB[both])),
                 'one_grade_disagreements': int(np.sum(np.abs(RA[both] - RB[both]) == 1)), 'two_grade_disagreements': int(np.sum(np.abs(RA[both] - RB[both]) == 2))}
    S0 = ens['S0']; v0 = vis[:, I]
    flip1 = np.mean([(grade(S0[:, I]) != grade(S0[:, vn.index(v)])) for v in ('T_1_col', 'T_1_row')], 0)
    sens = np.mean([js_bits(S0[:, I], S0[:, j]) for j in jit], 0); pid = np.repeat(np.arange(n)[:, None], 5, 1)
    m_dis = both & v0; disc = RA != RB; m_err = known & v0; err = grade(S0[:, I]) != Y
    res['L3'] = {'AI_1px_flip_level_rate_S0': float(flip1[v0].mean()), 'reader_disagreement_level_rate': res['L2']['disagreement_rate'],
                 'AI_flip_rate_on_discordant_levels': float(flip1[m_dis & disc].mean()), 'AI_flip_rate_on_concordant_levels': float(flip1[m_dis & ~disc].mean()),
                 'sensitivity_AUROC_reader_discordance': auroc_boot(sens[m_dis], disc[m_dis], pid[m_dis], n),
                 'sensitivity_AUROC_S0_error': auroc_boot(sens[m_err], err[m_err], pid[m_err], n),
                 'one_minus_maxprob_AUROC_S0_error': auroc_boot(1 - S0[:, I].max(-1)[m_err], err[m_err], pid[m_err], n),
                 'one_minus_maxprob_AUROC_reader_discordance': auroc_boot(1 - S0[:, I].max(-1)[m_dis], disc[m_dis], pid[m_dis], n)}
    res['L3']['b_discordance_significant'] = bool(res['L3']['sensitivity_AUROC_reader_discordance']['ci95'][0] > .5)
    res['L3']['c_error_significant'] = bool(res['L3']['sensitivity_AUROC_S0_error']['ci95'][0] > .5)
    uni = json.loads((U.OUT / 'UNIFIED_RESULT.json').read_text()); assert uni['views'] == vn
    geo = [vn.index(v) for v in uni['geometric_views']]; other = [i for i in range(nv) if i not in geo]
    W = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=10000).astype(np.float32); L4 = {}
    for a in ('A0', 'A1', 'A2', 'A3', 'A4', 'A5'):
        cal = uni['arch'][a]['calibration']; out = {}; bmin = {}
        for m in ('B', 'R0', 'R1'):
            ins, MT = U.build_sets(ens[a][:, :nv], cal, m); pp = U.per_patient(ins, MT, Y, I)
            sev = U.ratio(*pp['sev']); bsev = U.ratio(*pp['sev'], W); cov = U.ratio(*pp['cov']); sz = U.ratio(*pp['size']); bmin[m] = bsev[:, geo].min(1)
            out[m] = {'severe_identity': float(sev[I]), 'severe_min_geometric': float(sev[geo].min()), 'severe_min_geometric_ci95': U.ci(bmin[m]),
                      'severe_min_all27': float(sev.min()), 'severe_min_nongeometric': float(sev[other].min()), 'overall_min_view': float(cov.min()), 'size_identity': float(sz[I])}
            for k in ('single', 'single_acc', 'false_sev'):
                num, den = pp[k]; out[m][k + '_identity'] = float(num.sum() / max(1, den.sum()))
        d = bmin['R1'] - bmin['B']
        out['R1_minus_B_geo_severe'] = {'mean': out['R1']['severe_min_geometric'] - out['B']['severe_min_geometric'], 'ci95': U.ci(d), 'significant': bool(np.percentile(d, 2.5) > 0)}
        out['R1_geo_point_ge_0.895'] = bool(out['R1']['severe_min_geometric'] >= .895); L4[a] = out
    res['L4'] = L4; res.update(plan_sha256=V.sha(V.OUT / 'PLAN.md'), source_sha256=V.sha(__file__))
    (V.OUT / 'PART2_RESULT.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('patients', 'reference_levels_known', 'grade_counts', 'severe_patients', 'HOSP_185_excluded_reference_levels')}))
    for s, r in L1.items(): print('L1', s, 'RPS %.4f %s acc %.3f kappa %.3f recall %s sevmiss %d' % (r['RPS'], [round(x, 4) for x in r['RPS_ci95']], r['level_accuracy'], r['weighted_kappa_vs_reference'], {k: round(v, 2) for k, v in r['recall'].items()}, r['severe_miss_patients']))
    print('L2', json.dumps(res['L2'])); print('L3', json.dumps(res['L3']))
    for a, o in L4.items(): print('L4', a, ' | '.join('%s sevI=%.3f sevGeo=%.3f %s sevAll=%.3f size=%.2f falseSev=%.3f' % (m, o[m]['severe_identity'], o[m]['severe_min_geometric'], [round(x, 3) for x in o[m]['severe_min_geometric_ci95']], o[m]['severe_min_all27'], o[m]['size_identity'], o[m]['false_sev_identity']) for m in ('B', 'R0', 'R1')), 'R1-B', o['R1_minus_B_geo_severe'])


if __name__ == '__main__':
    main()
