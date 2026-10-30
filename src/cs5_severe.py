"""CS20261013-SEV (results/conformal_20261013/PLAN_SEVERE.md): temperature scaling + label-conditional (Mondrian)
ordinal conformal + intervention-worst-case calibration, versus OCP / OCP+T / MCP+T / IWC+T.  Local CPU, 500 splits."""
import hashlib, json, math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import cs4_conformal as C

OUT = C.OUT
R, SEED, ALPHA = 500, 20261015, 0.10
METHODS = {'M1_OCP': (False, False, False), 'M2_OCP_T': (True, False, False), 'M3_MCP_T': (True, True, False),
           'M4_SEV_IWC': (True, True, True), 'M5_IWC_T': (True, False, True)}   # (temperature, class-conditional, worst-case)


def temper(p, T):
    z = np.log(np.clip(p, 1e-12, 1)) / T; z -= z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def fit_T(p, y):
    def nll(T): return -np.log(np.clip(temper(p, T)[np.arange(len(y)), y], 1e-12, 1)).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 50), method='bounded').x)


def ece(p, y, bins=15):
    conf = p.max(-1); acc = (p.argmax(-1) == y).astype(float); e = 0.
    for k in range(bins):
        sel = (conf >= k / bins) & ((conf < (k + 1) / bins) if k < bins - 1 else (conf <= 1))
        if sel.any(): e += sel.mean() * abs(conf[sel].mean() - acc[sel].mean())
    return float(e)


def qhat(s):
    n = len(s); k = math.ceil((n + 1) * (1 - ALPHA)); return np.inf if k > n else float(np.sort(s)[k - 1])


def sets(S, M, q):
    """q scalar (marginal) or [3] per class; returns boolean [...,3] contiguous sets incl. the mode."""
    q = np.broadcast_to(np.asarray(q, dtype=np.float64), (3,))
    ins = (S <= q) | (np.arange(3) == M[..., None])
    ins[..., 1] |= ins[..., 0] & ins[..., 2]
    return ins


def main():
    P, ids, views = C.load_preds(); Y = C.labels(ids); n = len(ids); vi = {v: i for i, v in enumerate(views)}; I = vi['I']
    known = Y >= 0; rng = np.random.default_rng(SEED); splits = [rng.permutation(n) for _ in range(R)]
    res = {'alpha': ALPHA, 'splits': R, 'split_sizes': {'temperature': 100, 'calibration': 235, 'test': n - 335}, 'views': views, 'arch': {}}
    for a in C.ARCHS:
        acc = {m: {k: [] for k in ('cov', 'cov_c', 'size', 'single', 'single_acc', 'chg')} for m in METHODS}; Ts = []; eces = {'before': [], 'after': []}
        S0, M0 = C.ordinal_scores(P[a])
        for r, perm in enumerate(splits):
            tf, cal, te = perm[:100], perm[100:335], perm[335:]
            kt = known[tf]; pt = P[a][tf][:, I][kt]; yt = Y[tf][kt]; T = fit_T(pt, yt); Ts.append(T)
            PT = temper(P[a], T); ST, MT = C.ordinal_scores(PT)
            ke = known[te]; eces['before'].append(ece(P[a][te][:, I][ke], Y[te][ke])); eces['after'].append(ece(PT[te][:, I][ke], Y[te][ke]))
            yc = np.clip(Y, 0, 2)
            for m, (useT, cond, wc) in METHODS.items():
                S, M = (ST, MT) if useT else (S0, M0)
                sy = np.take_along_axis(S[cal], yc[cal][:, None, :, None].repeat(27, 1), -1)[..., 0]      # [c,27,5]
                s_use = sy.max(1) if wc else sy[:, I]                                                     # [c,5]
                kc = known[cal]
                if cond:
                    q = [qhat(s_use[kc & (Y[cal] == c)]) for c in range(3)]
                else:
                    q = qhat(s_use[kc])
                ins = sets(S[te], M[te], q)                                                              # [t,27,5,3]
                cov = np.take_along_axis(ins, yc[te][:, None, :, None].repeat(27, 1), -1)[..., 0]        # [t,27,5]
                acc[m]['cov'].append((cov & ke[:, None]).sum((0, 2)) / ke.sum())
                acc[m]['cov_c'].append(np.stack([(cov & (Y[te] == c)[:, None]).sum((0, 2)) / max(1, (Y[te] == c).sum()) for c in range(3)]))   # [3,27]
                acc[m]['size'].append(ins.sum(-1).mean((0, 2)))
                si = ins[:, I].sum(-1) == 1; acc[m]['single'].append(si.mean()); ok = si & ke
                acc[m]['single_acc'].append((M[te][:, I] == Y[te])[ok].mean() if ok.any() else np.nan)
                acc[m]['chg'].append(np.mean([(ins[:, vi[v]] != ins[:, I]).any(-1).mean() for v in ('T_1_col', 'T_1_row')]))
        def s(x): x = np.asarray(x, dtype=float); return {'mean': float(np.nanmean(x)), 'p2.5': float(np.nanpercentile(x, 2.5)), 'p97.5': float(np.nanpercentile(x, 97.5))}
        A = {'temperature': s(Ts), 'ECE_identity_before': s(eces['before']), 'ECE_identity_after': s(eces['after'])}
        for m in METHODS:
            cov = np.mean(acc[m]['cov'], 0); cc = np.mean(acc[m]['cov_c'], 0); sz = np.mean(acc[m]['size'], 0)
            A[m] = {'coverage_by_view': dict(zip(views, cov.round(4).tolist())), 'min_view_coverage': float(cov.min()),
                    'severe_coverage_by_view': dict(zip(views, cc[2].round(4).tolist())), 'severe_min_view': float(cc[2].min()), 'severe_min_view_name': views[int(cc[2].argmin())],
                    'moderate_min_view': float(cc[1].min()), 'normal_min_view': float(cc[0].min()),
                    'class_coverage_identity': cc[:, I].round(4).tolist(), 'size_identity': float(sz[I]), 'size_max_view': float(sz.max()),
                    'singleton_rate_identity': s(acc[m]['single']), 'singleton_accuracy_identity': s(acc[m]['single_acc']), 'set_change_T1': s(acc[m]['chg'])}
        A['H1_SEV_IWC_severe_all_views_ge_0.895'] = bool(A['M4_SEV_IWC']['severe_min_view'] >= .895)
        A['M3_severe_all_views_ge_0.895'] = bool(A['M3_MCP_T']['severe_min_view'] >= .895)
        A['H3_SEV_IWC_size_identity_le_2'] = bool(A['M4_SEV_IWC']['size_identity'] <= 2.0)
        res['arch'][a] = A
        print(a, 'T=%.2f' % A['temperature']['mean'], ' '.join('%s sevmin=%.3f size=%.2f' % (m[:2], A[m]['severe_min_view'], A[m]['size_identity']) for m in METHODS))
    H1 = all(res['arch'][a]['H1_SEV_IWC_severe_all_views_ge_0.895'] for a in C.ARCHS)
    M3ok = all(res['arch'][a]['M3_severe_all_views_ge_0.895'] for a in C.ARCHS)
    res.update(H1_passed=H1, H3_passed=all(res['arch'][a]['H3_SEV_IWC_size_identity_le_2'] for a in C.ARCHS), M3_already_sufficient=M3ok,
               H2_interpretation=('IWC necessary improvement for severe guarantee' if H1 and not M3ok else 'class-conditional calibration already sufficient; IWC not necessary' if M3ok else 'H1 not met'),
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), plan_sha256=C.sha(OUT / 'PLAN_SEVERE.md'))
    (OUT / 'SEV_RESULT.json').write_text(json.dumps(res, indent=1) + '\n', encoding='utf-8')
    print('H1', res['H1_passed'], 'H3', res['H3_passed'], 'M3 sufficient', M3ok, '->', res['H2_interpretation'])


if __name__ == '__main__':
    main()
