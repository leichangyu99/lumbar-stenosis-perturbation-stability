"""ENSEMBLE (results/ensemble_20261021/PLAN.md): single model vs within-architecture 3-seed ensemble vs best
single-architecture ensemble (selected on validation) vs full 18-model ensemble, on 1px-shift flips; accuracy and
conservativeness checks.  Confirmatory: SPIDER 210 + RSNA validation 197; replication: internal 671 + hospital 256."""
import hashlib, itertools, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs4_conformal as C  # noqa: E402
import cs12_unified_conformal as U  # noqa: E402
import cs13_hospital_stats as V  # noqa: E402
import cs13_hospital_labels as V2  # noqa: E402
from cs_common import rps_patient  # noqa: E402
from cs_metrics import grade  # noqa: E402

OUT = HERE.parent.parent / 'results/ensemble_20261021'
ARCHS = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']; SEEDS = (20261012, 20261013, 20261014)
NAMES = ['%s_%d' % (a, s) for a in ARCHS for s in SEEDS]
FAMS = ('T', 'R', 'S', 'N', 'W', 'D'); B, SEED = 10000, 20261021
SPIDER = Path('data/public/SPIDER_work_20261013')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canon(probs, names):
    ix = [names.index(n) for n in NAMES]; return probs[ix]


def load(which):
    if which == 'spider':
        lock = json.loads((SPIDER / 'PREDICTIONS_LOCK.json').read_text()); f = SPIDER / 'PRED_spider.npz'; assert sha(f) == lock['predictions_sha256']
        d = np.load(f); keys = [str(k) for k in d['keys']]; vn = [k.split('|')[0] for k in keys if k.endswith('|None')]
        return canon(d['probs'][:, :, :len(vn)], [str(n) for n in d['models']]), d['visible'][:, vn.index('I')], vn, None
    if which == 'validation':
        lock = json.loads((U.VAL / 'PREDICTIONS_LOCK.json').read_text()); f = U.VAL / 'PRED_validation.npz'; assert sha(f) == lock['validation']
        d = np.load(f); vn = [str(v) for v in d['views']]
        return canon(d['probs'], [str(n) for n in d['models']]), d['visible'][:, vn.index('I')], vn, C.labels([str(s) for s in d['patients']])
    if which == 'internal671':
        m = HERE / 'mirror'; arr = {}; vis = None; ids = None; vn = None
        for job in ('maeval', 'maevalC'):
            lock = json.loads((m / job / 'PREDICTIONS_LOCK.json').read_text()); parts = []
            for t in ('confirmation', 'policy'):
                f = m / job / ('PRED_%s.npz' % t); assert sha(f) == lock[t]; parts.append(np.load(f))
            nm = [str(n) for n in parts[0]['models']]; pr = np.concatenate([p['probs'] for p in parts], 1)
            for k, n in enumerate(nm): arr[n] = pr[k]
            vn = [str(v) for v in parts[0]['views']]; vis = np.concatenate([p['visible'] for p in parts], 0); ids = [str(s) for p in parts for s in p['patients']]
        return np.stack([arr[n] for n in NAMES]), vis[:, vn.index('I')], vn, C.labels(ids)
    if which == 'hospital256':
        probs, vis, names, keys, ids = V.load_merged(); vn = [k.split('|')[0] for k in keys if k.endswith('|None')]
        return canon(probs[:, :, :len(vn)], names), vis[:, vn.index('I')], vn, V2.read_labels(ids)[0]


def flip(P, vn, v0, views):
    """per-patient flip proportion (original-visible levels) averaged over the given views; P [n,27,5,3]."""
    I = vn.index('I'); c = np.maximum(v0.sum(-1), 1)
    return np.mean([((grade(P[:, I]) != grade(P[:, vn.index(v)])) * v0).sum(-1) / c for v in views], 0)


def boot(diff, idx):
    m = diff[idx].mean(1)
    return {'diff': float(diff.mean()), 'ci95': [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], 'p_one_sided': float((m >= 0).mean())}


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(ps) - r) * ps[i])); adj[i] = run
    return adj.tolist()


def analyse(which, best_arch, extended=True):
    P, v0, vn, Y = load(which); comp = v0.any(-1); P = P[:, comp]; v0 = v0[comp]; Y = Y[comp] if Y is not None else None; n = int(comp.sum())
    T1 = ('T_1_col', 'T_1_row'); idx = np.random.default_rng(SEED).integers(0, n, (B, n))
    single = np.stack([flip(P[k], vn, v0, T1) for k in range(18)])              # [18,n]
    arch_ens = {a: flip(P[3 * i:3 * i + 3].mean(0), vn, v0, T1) for i, a in enumerate(ARCHS)}
    arch_single = {a: single[3 * i:3 * i + 3].mean(0) for i, a in enumerate(ARCHS)}
    full = flip(P.mean(0), vn, v0, T1); mean_single = single.mean(0)
    r = {'patients': n, 'flip': {'mean_single': float(mean_single.mean()), 'full_ensemble': float(full.mean()),
                                  'arch_3seed': {a: float(arch_ens[a].mean()) for a in ARCHS}, 'arch_single_mean': {a: float(arch_single[a].mean()) for a in ARCHS}}}
    h1 = {a: boot(arch_ens[a] - arch_single[a], idx) for a in ARCHS}; adj = holm([h1[a]['p_one_sided'] for a in ARCHS])
    for a, p in zip(ARCHS, adj): h1[a]['p_holm'] = p; h1[a]['significant'] = bool(p < .05)
    r['H1'] = h1; r['H1_n_significant'] = sum(h1[a]['significant'] for a in ARCHS); r['H1_n_direction'] = sum(h1[a]['diff'] < 0 for a in ARCHS)
    r['H2'] = boot(full - mean_single, idx); r['H2']['passed'] = bool(r['H2']['ci95'][1] < 0)
    if best_arch:
        r['H3'] = boot(full - arch_ens[best_arch], idx); r['H3']['best_arch'] = best_arch; r['H3']['passed'] = bool(r['H3']['ci95'][1] < 0)
    I = vn.index('I'); g_full = grade(P.mean(0)[:, I]); g_single = np.stack([grade(P[k][:, I]) for k in range(18)])
    r['H5_normal_fraction'] = {'full': float((g_full == 0).mean()), 'mean_single': float((g_single == 0).mean()), 'diff': float((g_full == 0).mean() - (g_single == 0).mean())}
    r['predicted_distribution'] = {'full': [int((g_full == c).sum()) for c in range(3)], 'mean_single': [float((g_single == c).sum() / 18) for c in range(3)]}
    if Y is not None:
        rp_full = np.array([rps_patient(Y[i], P.mean(0)[i, I]) if (Y[i] >= 0).any() else np.nan for i in range(n)])
        rp_single = np.array([[rps_patient(Y[i], P[k][i, I]) if (Y[i] >= 0).any() else np.nan for i in range(n)] for k in range(18)]).mean(0)
        ok = np.isfinite(rp_full); d = (rp_full - rp_single)[ok]; idr = np.random.default_rng(SEED).integers(0, ok.sum(), (B, ok.sum()))
        r['H4_RPS'] = {'full': float(np.nanmean(rp_full)), 'mean_single': float(np.nanmean(rp_single)), **boot(d, idr)}; r['H4_RPS']['passed'] = bool(r['H4_RPS']['ci95'][1] <= .003)
        sev = Y == 2
        r['severe_recall'] = {'full': float((g_full[sev] == 2).mean()) if sev.any() else None, 'mean_single': float(np.mean([(g_single[k][sev] == 2).mean() for k in range(18)])) if sev.any() else None}
        r['level_accuracy'] = {'full': float((g_full[Y >= 0] == Y[Y >= 0]).mean()), 'mean_single': float(np.mean([(g_single[k][Y >= 0] == Y[Y >= 0]).mean() for k in range(18)]))}
    if extended:
        rng = np.random.default_rng(SEED); curve = {}
        for k in (1, 2, 3):
            curve['seeds_%d' % k] = float(np.mean([flip(P[[3 * i + s for s in c]].mean(0), vn, v0, T1).mean() for i in range(6) for c in itertools.combinations(range(3), k)]))
        for k in range(1, 7):
            combos = list(itertools.combinations(range(6), k)); vals = []
            for c in combos: vals.append(flip(P[[3 * i + s for i in c for s in range(3)]].mean(0), vn, v0, T1).mean())
            curve['archs_%d_x3seeds' % k] = float(np.mean(vals))
        r['size_curve'] = curve
        fam = {}
        for f in FAMS:
            vs = [v for v in vn if v.startswith(f + '_')]
            fam[f] = {'mean_single': float(np.mean([flip(P[k], vn, v0, vs).mean() for k in range(18)])), 'full_ensemble': float(flip(P.mean(0), vn, v0, vs).mean())}
        r['families'] = fam
    return r


def main():
    OUT.mkdir(exist_ok=True); res = {'plan_sha256': sha(OUT / 'PLAN.md'), 'source_sha256': sha(__file__), 'bootstrap': B, 'seed': SEED}
    val = analyse('validation', None)
    best = min(ARCHS, key=lambda a: (val['flip']['arch_3seed'][a], a)); res['best_arch_selected_on_validation'] = best
    val['H3'] = None; res['validation'] = val
    for which in ('spider', 'internal671', 'hospital256'): res[which] = analyse(which, best)
    sp = res['spider']; v = res['validation']
    res['verdict'] = {'H1_spider_significant': '%d/6' % sp['H1_n_significant'], 'H1_validation_direction': '%d/6' % v['H1_n_direction'],
                      'H2': bool(sp['H2']['passed'] and v['H2']['passed']), 'H3_spider': bool(sp['H3']['passed']), 'H4_validation': bool(v['H4_RPS']['passed']),
                      'H5': bool(sp['H5_normal_fraction']['diff'] <= .03 and v['H5_normal_fraction']['diff'] <= .03 and v['severe_recall']['full'] >= v['severe_recall']['mean_single'] - .05)}
    (OUT / 'ENSEMBLE_RESULT.json').write_text(json.dumps(res, indent=1) + '\n', encoding='utf-8')
    print('best arch (validation):', best); print('VERDICT', res['verdict'])
    for which in ('spider', 'validation', 'internal671', 'hospital256'):
        r = res[which]
        print('==', which, 'n', r['patients'], 'single %.4f full %.4f' % (r['flip']['mean_single'], r['flip']['full_ensemble']), 'arch3', {a: round(x, 4) for a, x in r['flip']['arch_3seed'].items()})
        print('   H1', {a: (round(h['diff'], 4), round(h['p_holm'], 4)) for a, h in r['H1'].items()}, 'H2', r['H2'], 'H3', r.get('H3'))
        print('   H5', r['H5_normal_fraction'], 'H4', r.get('H4_RPS'), 'sevrec', r.get('severe_recall'), 'acc', r.get('level_accuracy'))
        print('   curve', r['size_curve']); print('   fam', {f: (round(x['mean_single'], 4), round(x['full_ensemble'], 4)) for f, x in r['families'].items()})


if __name__ == '__main__':
    main()
