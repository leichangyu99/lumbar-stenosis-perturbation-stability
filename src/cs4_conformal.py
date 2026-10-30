"""CS20261013-CP (results/conformal_20261013/PLAN.md): ordinal conformal prediction (Lu et al. 2022 style
ordinal adaptive sets) with clean calibration (OCP) versus intervention-worst-case calibration (IWC-OCP), on the
locked six-architecture predictions for 671 patients x 27 views.  Local CPU; per-level calibration; 500 splits."""
import csv, hashlib, json, math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MIRROR = HERE / 'mirror'
OUT = HERE.parent.parent / 'results/conformal_20261013'
TRAIN_CSV = Path('data/LumbarDISC/dataset/train.csv')
ALPHA, R, SEED = 0.10, 500, 20261014
ARCHS = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']
LEVEL_KEYS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
GRADES = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_preds():
    """{arch: probs [671, 27, 5, 3]} (3-seed mean), patient ids, view names; hashes checked against locks."""
    arrays = {}; ids = None; views = None
    for job in ('maeval', 'maevalC'):
        lock = json.loads((MIRROR / job / 'PREDICTIONS_LOCK.json').read_text())
        parts = []
        for tag in ('confirmation', 'policy'):
            f = MIRROR / job / ('PRED_%s.npz' % tag); assert sha(f) == lock[tag], 'locked prediction hash differs ' + str(f)
            parts.append(np.load(f))
        names = list(parts[0]['models']); pid = list(parts[0]['patients']) + list(parts[1]['patients']); vn = list(parts[0]['views'])
        assert list(parts[1]['views']) == vn and list(parts[1]['models']) == names
        ids = pid if ids is None else ids; views = vn if views is None else views; assert ids == pid and views == vn
        probs = np.concatenate([parts[0]['probs'], parts[1]['probs']], 1)
        for a in sorted({n.split('_')[0] for n in names}):
            arrays[a] = probs[[k for k, n in enumerate(names) if n.startswith(a + '_')]].mean(0)
    return arrays, [str(s) for s in ids], [str(v) for v in views]


def labels(ids):
    want = set(ids); out = {}
    for r in csv.DictReader(TRAIN_CSV.open(encoding='utf-8-sig')):
        if r['study_id'] in want: out[r['study_id']] = [GRADES.get(r['spinal_canal_stenosis_' + k], -1) for k in LEVEL_KEYS]
    assert set(out) == want
    return np.array([out[s] for s in ids])


def ordinal_scores(p):
    """score[..., y]: cumulative mass of the first greedy contiguous interval (grown from the mode, larger neighbour
    first, ties to the lower grade) that contains y."""
    p = np.asarray(p, dtype=np.float64); m = p.argmax(-1); p0, p1, p2 = p[..., 0], p[..., 1], p[..., 2]
    s = np.empty(p.shape); one = np.ones_like(p0)
    # mode 0: {0},{0,1},{0,1,2}; mode 2: {2},{1,2},{0,1,2}
    s0 = np.stack([p0, p0 + p1, one], -1); s2 = np.stack([one, p1 + p2, p2], -1)
    low_first = p0 >= p2   # mode 1: add 0 first if p0 >= p2
    s1 = np.where(low_first[..., None], np.stack([p0 + p1, p1, one], -1), np.stack([one, p1, p1 + p2], -1))
    s = np.where((m == 0)[..., None], s0, np.where((m == 2)[..., None], s2, s1))
    return s, m


def qhat(scores):
    n = len(scores); k = math.ceil((n + 1) * (1 - ALPHA))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def main():
    OUT.mkdir(exist_ok=True)
    P, ids, views = load_preds(); Y = labels(ids); n = len(ids); vi = {v: i for i, v in enumerate(views)}
    rng = np.random.default_rng(SEED); splits = [rng.permutation(n) for _ in range(R)]
    res = {'alpha': ALPHA, 'splits': R, 'patients': n, 'calibration_n': 335, 'test_n': n - 335, 'views': views, 'arch': {}}
    for a in ARCHS:
        S, M = ordinal_scores(P[a])                                  # S [n,27,5,3], M [n,27,5]
        Sy = np.take_along_axis(S, np.clip(Y, 0, 2)[:, None, :, None].repeat(27, 1), -1)[..., 0]   # [n,27,5] score of true y
        known = Y >= 0
        out = {m: {'coverage': np.zeros((R, 27)), 'size': np.zeros((R, 27)), 'severe_cov': np.zeros((R, 27)), 'singleton': np.zeros(R), 'singleton_acc': np.zeros(R),
                   'set_change_T1': np.zeros(R)} for m in ('OCP', 'IWC')}
        flip_T1 = np.zeros(R); qs = {m: np.zeros((R, 5)) for m in out}
        for r, perm in enumerate(splits):
            cal, te = perm[:335], perm[335:]
            for meth in ('OCP', 'IWC'):
                q = np.array([qhat((Sy[cal, vi['I'], l] if meth == 'OCP' else Sy[cal, :, l].max(1))[known[cal, l]]) for l in range(5)]); qs[meth][r] = q
                inset = (S[te] <= q[None, None, :, None]) | (np.arange(3)[None, None, None, :] == M[te][..., None])      # [t,27,5,3]
                kt = known[te]; cov = np.take_along_axis(inset, np.clip(Y[te], 0, 2)[:, None, :, None].repeat(27, 1), -1)[..., 0]
                out[meth]['coverage'][r] = (cov & kt[:, None, :]).sum((0, 2)) / kt.sum()
                out[meth]['size'][r] = inset.sum(-1).mean((0, 2))
                sev = Y[te] == 2; out[meth]['severe_cov'][r] = (cov & sev[:, None, :]).sum((0, 2)) / max(1, sev.sum())
                I = inset[:, vi['I']]; single = I.sum(-1) == 1; out[meth]['singleton'][r] = single.mean()
                ok = single & kt; out[meth]['singleton_acc'][r] = (M[te][:, vi['I']] == Y[te])[ok].mean() if ok.any() else np.nan
                ch = [(inset[:, vi[v]] != I).any(-1).mean() for v in ('T_1_col', 'T_1_row')]; out[meth]['set_change_T1'][r] = float(np.mean(ch))
            flip_T1[r] = np.mean([(M[te][:, vi[v]] != M[te][:, vi['I']]).mean() for v in ('T_1_col', 'T_1_row')])
        def summ(x): return {'mean': float(np.nanmean(x)), 'p2.5': float(np.nanpercentile(x, 2.5)), 'p97.5': float(np.nanpercentile(x, 97.5))}
        A = {'qhat_mean_by_level': {m: qs[m].mean(0).tolist() for m in qs}, 'argmax_flip_T1_all_levels': summ(flip_T1)}
        for m in out:
            cv = out[m]['coverage'].mean(0)
            A[m] = {'coverage_by_view': dict(zip(views, cv.round(4).tolist())), 'min_view_coverage': float(cv.min()), 'min_view': views[int(cv.argmin())],
                    'size_by_view': dict(zip(views, out[m]['size'].mean(0).round(4).tolist())), 'size_identity': summ(out[m]['size'][:, vi['I']]),
                    'severe_coverage_identity': summ(out[m]['severe_cov'][:, vi['I']]), 'severe_coverage_min_view': float(out[m]['severe_cov'].mean(0).min()),
                    'singleton_rate_identity': summ(out[m]['singleton']), 'singleton_accuracy_identity': summ(out[m]['singleton_acc']),
                    'set_change_rate_T1': summ(out[m]['set_change_T1']), 'coverage_identity': summ(out[m]['coverage'][:, vi['I']])}
        A['G1_IWC_all_views_ge_0.895'] = bool(A['IWC']['min_view_coverage'] >= .895)
        A['baseline_all_views_ge_0.895'] = bool(A['OCP']['min_view_coverage'] >= .895)
        A['G2_size_increase_le_0.5'] = bool(A['IWC']['size_identity']['mean'] <= A['OCP']['size_identity']['mean'] + .5)
        res['arch'][a] = A
        print(a, 'OCP min', round(A['OCP']['min_view_coverage'], 4), A['OCP']['min_view'], 'IWC min', round(A['IWC']['min_view_coverage'], 4), A['IWC']['min_view'],
              'size I', round(A['OCP']['size_identity']['mean'], 3), round(A['IWC']['size_identity']['mean'], 3))
    res['G1_passed'] = all(res['arch'][a]['G1_IWC_all_views_ge_0.895'] for a in ARCHS)
    res['G2_passed'] = all(res['arch'][a]['G2_size_increase_le_0.5'] for a in ARCHS)
    res['baseline_robust_everywhere'] = all(res['arch'][a]['baseline_all_views_ge_0.895'] for a in ARCHS)
    res['interpretation'] = ('IWC improvement' if res['G1_passed'] and not res['baseline_robust_everywhere'] else
                             'baseline already robust; IWC unnecessary' if res['baseline_robust_everywhere'] else 'IWC validity not met')
    res['source_sha256'] = sha(__file__); res['plan_sha256'] = sha(OUT / 'PLAN.md'); res['labels'] = 'train.csv rows of the 671 confirmation+policy patients (previously read)'
    (OUT / 'CP_RESULT.json').write_text(json.dumps(res, indent=1) + '\n', encoding='utf-8')
    print('G1', res['G1_passed'], 'G2', res['G2_passed'], 'baseline robust', res['baseline_robust_everywhere'], '->', res['interpretation'])


if __name__ == '__main__':
    main()
