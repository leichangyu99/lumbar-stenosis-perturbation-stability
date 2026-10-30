"""CS20261013-SPX statistics (copy of cs7_external_stats for SPIDER).  Original: CS20261013-EXT part 1 statistics (label-free): per-system T 1px flip CI, dose trend, family flips, Kendall W, and
Spearman correlation of family flip means with RSNA (confirmation + policy474 pooled for A0-A5; confirmation for S0)."""
import hashlib, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cs_metrics import grade, js_bits  # noqa: E402

WORK = Path('data/public/SPIDER_work_20261013')
OUT = HERE.parent.parent / 'results/external_spider_20261013'
SYSTEMS = ['S0', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5']
FAMS = ('T', 'R', 'S', 'N', 'W', 'D')
B, BSEED = 10000, 20261018


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def family_flips(P, vn, vis0):
    """per-view patient flip arrays (original-view-visible levels) and family means."""
    vi = {v: i for i, v in enumerate(vn)}; comp = vis0.any(-1); out = {}
    for v in vn:
        f = (grade(P[:, vi['I']]) != grade(P[:, vi[v]])).astype(float); j = js_bits(P[:, vi['I']], P[:, vi[v]]); c = vis0.sum(-1)
        out[v] = ((f * vis0).sum(-1)[comp] / c[comp], (j * vis0).sum(-1)[comp] / c[comp])
    fam = {f: float(np.mean([out[v][0].mean() for v in vn if v.startswith(f + '_')])) for f in FAMS}
    return out, fam, comp


def rsna_family_means():
    m = HERE / 'mirror'; res = {}
    for job in ('maeval', 'maevalC'):
        parts = [np.load(m / job / ('PRED_%s.npz' % t)) for t in ('confirmation', 'policy')]
        names = list(parts[0]['models']); vn = [str(v) for v in parts[0]['views']]
        probs = np.concatenate([p['probs'] for p in parts], 1); vis = np.concatenate([p['visible'] for p in parts], 0)
        for a in sorted({n.split('_')[0] for n in names}):
            P = probs[[k for k, n in enumerate(names) if n.startswith(a + '_')]].mean(0); res[a] = family_flips(P, vn, vis[:, vn.index('I')])[1]
    e34 = np.load(m / 'e34/PREDICTIONS.npz'); names = list(e34['models']); vn = [str(v) for v in e34['views']]
    P = e34['probs'][[k for k, n in enumerate(names) if n.startswith('STD_')]].mean(0); res['S0'] = family_flips(P, vn, e34['visible'][:, vn.index('I')])[1]
    return res


def main():
    lock = json.loads((WORK / 'PREDICTIONS_LOCK.json').read_text()); f = WORK / 'PRED_spider.npz'; assert sha(f) == lock['predictions_sha256']
    d = np.load(f); names = [str(n) for n in d['models']]; keys = [str(k) for k in d['keys']]; probs = d['probs']; vis = d['visible']
    vn = [k.split('|')[0] for k in keys if k.endswith('|None')]; nv = len(vn); vis0 = vis[:, vn.index('I')]
    idx = None; res = {'patients': int(probs.shape[1]), 'systems': {}, 'labels_read': False}
    rsna = rsna_family_means()
    for s in SYSTEMS:
        P = probs[[k for k, n in enumerate(names) if n.startswith(s + '_')]][:, :, :nv].mean(0)
        pv, fam, comp = family_flips(P, vn, vis0)
        if idx is None: idx = np.random.default_rng(BSEED).integers(0, int(comp.sum()), (B, int(comp.sum())))
        def ci(x): m = x[idx].mean(1); return {'mean': float(x.mean()), 'ci95': [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}
        t1 = (pv['T_1_col'][0] + pv['T_1_row'][0]) / 2; dj = (pv['T_1_col'][1] + pv['T_1_row'][1]) / 2 - (pv['T_0.25_col'][1] + pv['T_0.25_row'][1]) / 2
        r = {'T1px_flip': ci(t1), 'dose_JS1_minus_JS025': ci(dj), 'identity_zero': bool(pv['I'][0].max() == 0 and pv['I'][1].max() == 0),
             'family_flip_means': fam, 'per_view_flip': {v: float(pv[v][0].mean()) for v in vn if v != 'I'},
             'spearman_vs_RSNA_family_means': float(spearmanr([fam[x] for x in FAMS], [rsna[s][x] for x in FAMS]).statistic), 'RSNA_family_flip_means': rsna[s]}
        r['primary_passed'] = bool(r['identity_zero'] and r['T1px_flip']['ci95'][0] > 0); r['dose_trend'] = bool(r['dose_JS1_minus_JS025']['ci95'][0] > 0)
        res['systems'][s] = r
        print(s, 'T1 %.4f %s dose %s pass %s rho %.2f' % (r['T1px_flip']['mean'], [round(x, 4) for x in r['T1px_flip']['ci95']], [round(x, 5) for x in r['dose_JS1_minus_JS025']['ci95']], r['primary_passed'], r['spearman_vs_RSNA_family_means']), {k: round(v, 3) for k, v in fam.items()})
    Rk = np.array([rankdata([res['systems'][s]['family_flip_means'][x] for x in FAMS]) for s in SYSTEMS]); m, n = Rk.shape
    res['kendall_W_family_ranking'] = float(12 * ((Rk.sum(0) - m * (n + 1) / 2) ** 2).sum() / (m ** 2 * (n ** 3 - n)))
    res['generality_7of7'] = all(res['systems'][s]['primary_passed'] for s in SYSTEMS)
    res.update(plan_sha256=sha(OUT / 'PLAN.md'), source_sha256=sha(__file__), predictions_sha256=lock['predictions_sha256'])
    (OUT / 'SPX_RESULT.json').write_text(json.dumps(res, indent=1) + '\n', encoding='utf-8')
    print('generality 7/7', res['generality_7of7'], 'Kendall W', round(res['kendall_W_family_ranking'], 3))


if __name__ == '__main__':
    main()
