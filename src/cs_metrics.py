"""CS20261012 frozen flip / JS metrics (EXPERIMENT_PLAN V3 section 0 and 10).

argmax ties -> lowest grade; flip and JS only on levels whose ORIGINAL-view ROI is visible;
level -> patient mean -> patient-equal mean; patients without computable levels are excluded
from the denominator and counted.  JS = JS(p, q) / ln 2 with natural logs and 0 log 0 = 0.
"""
import math

import numpy as np

LN2 = math.log(2.0)


def grade(p):
    return np.argmax(np.asarray(p, dtype=np.float64), axis=-1)  # numpy returns the first (lowest) maximum


def _kl(a, m):
    safe_a = np.where(a > 0, a, 1.0); safe_m = np.where(m > 0, m, 1.0)
    return np.sum(np.where(a > 0, a * np.log(safe_a / safe_m), 0.0), axis=-1)


def js_bits(p, q):
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64); m = (p + q) / 2
    return (0.5 * _kl(p, m) + 0.5 * _kl(q, m)) / LN2


def patient_level(p0, pv, visible):
    """p0, pv: [5,3] probabilities (original, intervened); visible: [5] original-view ROI visibility."""
    visible = np.asarray(visible, dtype=bool)
    if not visible.any():
        return None
    flips = (grade(p0) != grade(pv))[visible]; js = js_bits(p0, pv)[visible]
    return {'flip': float(flips.mean()), 'js': float(js.mean()), 'levels': int(visible.sum())}


def cohort(original, intervened, visible, ids):
    """Patient-equal mean flip/JS over ids; returns per-patient values for bootstrap."""
    per = {}; excluded = []
    for sid in ids:
        r = patient_level(original[sid], intervened[sid], visible[sid])
        if r is None:
            excluded.append(sid)
        else:
            per[sid] = r
    if not per:
        return {'evaluable': False, 'patients': 0, 'excluded_no_visible_level': len(excluded), 'per_patient': {}}
    return {'evaluable': True, 'patients': len(per), 'excluded_no_visible_level': len(excluded),
            'flip': float(np.mean([r['flip'] for r in per.values()])), 'js': float(np.mean([r['js'] for r in per.values()])),
            'per_patient': per}
