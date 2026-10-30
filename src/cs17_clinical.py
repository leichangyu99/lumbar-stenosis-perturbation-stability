"""Post hoc descriptive analyses for the manuscript (not pre-registered).
Replaces the mislabelled cs16 'patient status change' endpoint (cs16 counted patients with >=1 level crossing the
severe boundary in either shift direction) with explicit definitions:
  K1 level flip rate under each 1-px shift (direction-averaged);
  K2 patient-level 'any predicted severe level' status before vs after a 1-px shift (direction-averaged and either-direction);
  K3 patients with >=1 level crossing the severe boundary (direction-averaged; cs16 used the union of directions);
  K4 directed transitions on labelled cohorts (correct->incorrect, incorrect->correct, incorrect->incorrect; severe up/downgrades);
  K5 equal-size ensemble comparison: same-architecture seeds vs different architectures (3 and 6 models);
  K6 hospital: confusion matrices and vendor-stratified accuracy/flip (descriptive).
Patient bootstrap 2000 (seed 20261021); per-patient values are averaged over models before resampling."""
import itertools, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs15_ensemble as E  # noqa: E402
import cs13_hospital_stats as V  # noqa: E402
from cs_metrics import grade  # noqa: E402

OUT = HERE.parent.parent / 'results/clinical_addons_20261021'
VENDOR = Path('data/hospital/hospital_vendor_map.json')
T1 = ('T_1_col', 'T_1_row'); B, SEED = 2000, 20261021


def ci(per_patient, idx):
    m = per_patient[idx].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def per_patient_metrics(P, vn, v0, Y):
    """P [n,27,5,3] one model/ensemble -> dict of per-patient arrays (averaged over the two shift directions)."""
    I = vn.index('I'); g0 = grade(P[:, I]); vis = v0.astype(bool); out = {}
    sev0 = ((g0 == 2) & vis).any(-1); mod0 = ((g0 >= 1) & vis).any(-1)
    acc = {k: [] for k in ('flip', 'sev_status', 'mod_status', 'sev_cross', 'mod_cross')}; either_sev = np.zeros(len(P), bool); either_mod = np.zeros(len(P), bool)
    trans = {k: np.zeros(len(P)) for k in ('c2i', 'i2c', 'i2i', 'sev_down', 'sev_up', 'known_levels')}
    for v in T1:
        g1 = grade(P[:, vn.index(v)]); f = (g0 != g1) & vis; n = np.maximum(vis.sum(-1), 1)
        acc['flip'].append(f.sum(-1) / n)
        s1 = ((g1 == 2) & vis).any(-1); m1 = ((g1 >= 1) & vis).any(-1)
        acc['sev_status'].append(sev0 != s1); acc['mod_status'].append(mod0 != m1); either_sev |= sev0 != s1; either_mod |= mod0 != m1
        acc['sev_cross'].append((((g0 == 2) != (g1 == 2)) & vis).any(-1)); acc['mod_cross'].append((((g0 >= 1) != (g1 >= 1)) & vis).any(-1))
        if Y is not None:
            k = vis & (Y >= 0); c0 = g0 == Y; c1 = g1 == Y
            trans['c2i'] += (f & k & c0 & ~c1).sum(-1) / 2; trans['i2c'] += (f & k & ~c0 & c1).sum(-1) / 2; trans['i2i'] += (f & k & ~c0 & ~c1).sum(-1) / 2
            trans['sev_down'] += (f & (g0 == 2) & (g1 < 2)).sum(-1) / 2; trans['sev_up'] += (f & (g0 < 2) & (g1 == 2)).sum(-1) / 2
            trans['known_levels'] += k.sum(-1) / 2
    for k, v in acc.items():
        out[k] = np.mean(np.stack(v).astype(float), 0)
    out['sev_status_either'] = either_sev.astype(float); out['mod_status_either'] = either_mod.astype(float)
    if Y is not None:
        out.update(trans)
        ysev = ((Y == 2)).any(-1); out['patient_severe_ref'] = ysev.astype(float)
    return out


def summarise(per_list, idx, labelled):
    """per_list: list of per-patient dicts (one per model); average over models then bootstrap."""
    keys = [k for k in per_list[0] if k not in ('patient_severe_ref',)]; avg = {k: np.mean([p[k] for p in per_list], 0) for k in keys}; r = {}
    for k in ('flip', 'sev_status', 'sev_status_either', 'mod_status', 'mod_status_either', 'sev_cross', 'mod_cross'):
        r[k] = {'mean': float(avg[k].mean()), 'ci95': ci(avg[k], idx)}
    if labelled:
        tot = {k: float(avg[k].sum()) for k in ('c2i', 'i2c', 'i2i', 'sev_down', 'sev_up')}; flips = tot['c2i'] + tot['i2c'] + tot['i2i']
        r['transitions_per_direction_mean_counts'] = tot; r['flips_known_levels_per_direction'] = flips
        r['fraction_correct_to_incorrect'] = tot['c2i'] / flips if flips else None; r['fraction_incorrect_to_correct'] = tot['i2c'] / flips if flips else None
    return r


def cohort(which):
    P, v0, vn, Y = E.load(which); comp = v0.any(-1); P = P[:, comp]; v0 = v0[comp]; Y = Y[comp] if Y is not None else None
    n = int(comp.sum()); idx = np.random.default_rng(SEED).integers(0, n, (B, n)); lab = Y is not None
    singles = [per_patient_metrics(P[k], vn, v0, Y) for k in range(18)]
    res = {'patients': n, 'single_models_mean': summarise(singles, idx, lab), 'full_ensemble': summarise([per_patient_metrics(P.mean(0), vn, v0, Y)], idx, lab)}
    if lab:
        res['patients_with_reference_severe'] = int(((Y == 2).any(-1)).sum())
    return res, P, v0, vn, Y, idx


def equal_size(P, vn, v0, idx):
    """K5: mean per-patient 1-px flip of k-model ensembles drawn within one architecture vs across architectures."""
    def fl(models):
        return E.flip(P[list(models)].mean(0), vn, v0, T1)
    same3 = np.mean([fl([3 * a + s for s in range(3)]) for a in range(6)], 0)
    cross3 = np.mean([fl([3 * a + s for a, s in zip(archs, seeds)]) for archs in itertools.combinations(range(6), 3) for seeds in itertools.product(range(3), repeat=3)], 0)
    same6 = np.mean([fl([3 * a + s for a in pair for s in range(3)]) for pair in itertools.combinations(range(6), 2)], 0)
    cross6 = np.mean([fl([3 * a + s for a, s in zip(range(6), seeds)]) for seeds in itertools.product(range(3), repeat=6)], 0)
    out = {}
    for k, a, b in (('k3', same3, cross3), ('k6', same6, cross6)):
        d = b - a; out[k] = {'same_arch': float(a.mean()), 'cross_arch': float(b.mean()), 'diff_cross_minus_same': float(d.mean()), 'diff_ci95': ci(d, idx)}
    out['note'] = 'k3: 6 same-architecture 3-seed ensembles vs 540 three-architecture one-seed ensembles; k6: 15 two-architecture x 3-seed vs 729 six-architecture one-seed ensembles'
    return out


def hospital_extra(P, v0, vn, Y):
    probs, vis, names, keys, ids = V.load_merged(); vn2 = [k.split('|')[0] for k in keys if k.endswith('|None')]
    S0 = probs[[k for k, n in enumerate(names) if n.startswith('S0_')]][:, :, :len(vn2)].mean(0); I = vn.index('I')
    vendor = json.loads(VENDOR.read_text(encoding='utf-8')); ven = np.array([vendor[s]['vendor'] for s in ids])
    def conf(p):
        g = grade(p[:, I]); k = Y >= 0; m = np.zeros((3, 3), int)
        for t, q in zip(Y[k], g[k]): m[t, q] += 1
        return m.tolist()
    res = {'confusion_rows_reference_cols_predicted': {'S0': conf(S0), 'full_ensemble': conf(P.mean(0))}, 'by_vendor': {}}
    for vv in sorted(set(ven)):
        m = ven == vv; r = {'patients': int(m.sum())}
        for tag, p in (('S0', S0), ('full_ensemble', P.mean(0))):
            g = grade(p[m][:, I]); y = Y[m]; k = y >= 0
            r[tag] = {'level_accuracy': float((g[k] == y[k]).mean()), 'severe_recall': float((g[y == 2] == 2).mean()) if (y == 2).any() else None,
                      'flip_1px': float(E.flip(p[m], vn, v0[m], T1).mean())}
        r['reference_grade_counts'] = np.bincount(Y[m][Y[m] >= 0], minlength=3).tolist(); res['by_vendor'][vv] = r
    s0 = per_patient_metrics(S0, vn, v0, Y); idx = np.random.default_rng(SEED).integers(0, len(S0), (B, len(S0)))
    res['S0'] = summarise([s0], idx, True)
    return res


def main():
    res = {'note': 'post hoc descriptive; supersedes cs16 C1 labels', 'definitions': __doc__}
    for which in ('internal671', 'hospital256', 'spider', 'validation'):
        r, P, v0, vn, Y, idx = cohort(which)
        r['equal_size_ensembles'] = equal_size(P, vn, v0, idx)
        if which == 'hospital256':
            r['hospital_extra'] = hospital_extra(P, v0, vn, Y)
        res[which] = r
        print(which, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ('flip', 'sev_status', 'sev_status_either', 'sev_cross', 'mod_status', 'fraction_correct_to_incorrect')}) for k, v in r.items() if k in ('single_models_mean', 'full_ensemble')}))
        print('   equal size', r['equal_size_ensembles'])
    (OUT / 'CLINICAL_V2.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(res['hospital256']['hospital_extra'], indent=0)[:3000])


if __name__ == '__main__':
    main()
