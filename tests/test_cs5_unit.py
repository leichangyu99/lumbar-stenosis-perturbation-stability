"""Unit checks: contiguity fill, temperature fit direction, label-conditional coverage on synthetic data."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import cs4_conformal as C, cs5_severe as S
ins = S.sets(np.array([[.2, 1., .3]]), np.array([0]), [.5, .5, .5]); assert ins.tolist() == [[True, True, True]]   # gap at 1 filled
rng = np.random.default_rng(1); n = 6000; logit = rng.normal(0, 2, (n, 3)); p = np.exp(logit) / np.exp(logit).sum(1, keepdims=True)
y = np.array([rng.choice(3, p=q) for q in p]); over = S.temper(p, 1 / 3)            # sharpened = overconfident
T = S.fit_T(over[:2000], y[:2000]); print('fitted T on overconfident probs', round(T, 2)); assert 2.5 < T < 3.6
sc, m = C.ordinal_scores(p); sy = sc[np.arange(n), y]; q = [S.qhat(sy[:3000][y[:3000] == c]) for c in range(3)]
ins = S.sets(sc[3000:], m[3000:], q); cov = ins[np.arange(3000), y[3000:]]
for c in range(3): cc = cov[y[3000:] == c].mean(); print('class', c, 'coverage', round(cc, 3)); assert cc >= .88
print('cs5 unit passed')
