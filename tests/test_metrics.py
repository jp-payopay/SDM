import numpy as np

from sdm_plugin.core.evaluation.metrics import evaluate, max_tss
from sdm_plugin.core.evaluation.threshold import apply_threshold, maxtss_threshold


def test_max_tss_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    tss, thr = max_tss(y, s)
    assert tss == 1.0
    assert 0.3 < thr <= 0.7


def test_evaluate_shapes():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    s = rng.uniform(0, 1, size=200)
    res = evaluate(y, s)
    assert 0.0 <= res.auc <= 1.0
    assert np.isfinite(res.tss)
    assert np.isfinite(res.threshold)


def test_apply_threshold_preserves_nan():
    r = np.array([[0.1, np.nan], [0.6, 0.9]])
    thr = maxtss_threshold(np.array([0, 1, 1, 0]), np.array([0.2, 0.7, 0.8, 0.3]))
    b = apply_threshold(r, thr)
    assert b.dtype == np.uint8
    assert b[0, 1] == 0  # nan → 0
