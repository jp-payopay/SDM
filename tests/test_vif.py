import numpy as np

from sdm_plugin.core.predictors.vif import stepwise_vif


def test_stepwise_vif_drops_collinear():
    rng = np.random.default_rng(0)
    n = 500
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = a + 0.01 * rng.normal(size=n)  # nearly identical to a
    X = np.column_stack([a, b, c])
    Xr, kept, rep = stepwise_vif(X, ["a", "b", "c"], cutoff=10.0)
    assert "a" not in kept or "c" not in kept
    assert "b" in kept
    assert len(rep.steps) >= 1


def test_stepwise_vif_no_drops_when_independent():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    Xr, kept, rep = stepwise_vif(X, ["p", "q", "r"], cutoff=10.0)
    assert kept == ["p", "q", "r"]
    assert rep.dropped == []


def test_stepwise_vif_drops_zero_variance_predictor():
    """Regression test: an infinite VIF (zero-variance/perfectly collinear
    predictor) must be dropped, not treated as a stop condition — otherwise
    it silently rides through to model fitting as "retained" with vif=inf.
    """
    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    const = np.full(n, 5.0)
    X = np.column_stack([a, b, const])
    Xr, kept, rep = stepwise_vif(X, ["a", "b", "const"], cutoff=10.0)
    assert "const" not in kept
    assert "const" in rep.dropped
    assert Xr.shape[1] == len(kept) == 2
