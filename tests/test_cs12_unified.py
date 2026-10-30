import numpy as np

import cs12_unified_conformal as U


def test_ratio_bootstrap_identity_weights_matches_point_estimate():
    rng = np.random.default_rng(0); num = rng.integers(0, 3, (40, 27)); den = np.full(40, 5)
    W = np.ones((2, 40), dtype=np.float32)
    assert np.allclose(U.ratio(num, den, W)[0], U.ratio(num, den))


def test_rule_out_only_adds_severe_and_keeps_sets_contiguous():
    rng = np.random.default_rng(1); p = rng.dirichlet(np.ones(3), (30, 27, 5)); y = rng.integers(0, 3, (30, 5))
    cal = U.calibrate(p, y, 0, list(range(19)))
    b, _ = U.build_sets(p, cal, 'B'); r1, _ = U.build_sets(p, cal, 'R1')
    assert (r1 >= b).all() and (r1[..., 0] == b[..., 0]).all()
    assert not (r1[..., 0] & r1[..., 2] & ~r1[..., 1]).any()
    assert cal['tau']['R1'] >= cal['tau']['R0']


def test_per_patient_coverage_counts_only_known_levels():
    ins = np.ones((2, 27, 5, 3), dtype=bool); MT = np.zeros((2, 27, 5), dtype=int); Y = np.array([[0, -1, 2, 1, 2], [-1] * 5])
    pp = U.per_patient(ins, MT, Y, 0)
    assert pp['cov'][1].tolist() == [4, 0] and pp['sev'][1].tolist() == [2, 0] and U.ratio(*pp['sev'])[0] == 1.0
