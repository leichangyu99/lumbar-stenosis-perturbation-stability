"""Unit checks: ordinal score sequences, contiguous sets, and ~1-alpha coverage on synthetic calibrated data."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import cs4_conformal as C
s, m = C.ordinal_scores(np.array([[.6, .3, .1], [.2, .5, .3], [.35, .5, .15], [.1, .2, .7]]))
assert np.allclose(s, [[.6, .9, 1], [1, .5, .8], [.85, .5, 1], [1, .9, .7]]) and m.tolist() == [0, 1, 1, 2], s
assert C.qhat(np.arange(8) / 10) == np.inf and C.qhat(np.arange(9) / 10) == .8 and C.qhat(np.arange(100) / 100) == .9
rng = np.random.default_rng(0); n = 4000; p = rng.dirichlet([.3, .3, .3], n); y = np.array([rng.choice(3, p=q) for q in p])
S, M = C.ordinal_scores(p); sy = S[np.arange(n), y]; q = C.qhat(sy[:2000])
cov = ((S[2000:] <= q) | (np.arange(3) == M[2000:, None]))[np.arange(2000), y[2000:]].mean(); print('synthetic coverage', round(cov, 3)); assert cov >= .89  # non-randomised discrete scores: valid, possibly conservative
print('cs4 unit passed')
