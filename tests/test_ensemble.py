from __future__ import annotations

import numpy as np

from sdm_plugin.core.prediction.ensemble import (
    compute_weights,
    ensemble_permutation_importance,
    ensemble_predictions,
)


def test_compute_weights_mean_is_equal():
    w = compute_weights(["a", "b", "c"], None, "mean")
    assert w == {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}


def test_compute_weights_metric_normalizes():
    w = compute_weights(["a", "b"], {"a": 0.9, "b": 0.3}, "weighted_tss")
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["a"] > w["b"]


def test_compute_weights_falls_back_to_equal_when_all_nonpositive():
    w = compute_weights(["a", "b"], {"a": 0.0, "b": -1.0}, "weighted_auc")
    assert w == {"a": 0.5, "b": 0.5}


def test_compute_weights_requires_metric_for_weighted_methods():
    try:
        compute_weights(["a"], None, "weighted_auc")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ensemble_predictions_mean():
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    mean, sd = ensemble_predictions({"a": a, "b": b}, None, method="mean")
    np.testing.assert_allclose(mean, 0.5 * np.ones((2, 2)))
    np.testing.assert_allclose(sd, 0.5 * np.ones((2, 2)))


def test_ensemble_predictions_weighted_matches_manual_weights():
    a = np.full((2, 2), 1.0, dtype=np.float32)
    b = np.full((2, 2), 0.0, dtype=np.float32)
    mean, _sd = ensemble_predictions({"a": a, "b": b}, {"a": 0.75, "b": 0.25}, method="weighted_tss")
    np.testing.assert_allclose(mean, 0.75 * np.ones((2, 2)), atol=1e-6)


class _LinearStub:
    """Minimal SDMModel-like stub: logistic on a fixed weight vector."""

    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-(np.asarray(X, dtype=float) @ self.w)))


def test_ensemble_permutation_importance_ranks_informative_feature():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (1.0 / (1.0 + np.exp(-(X @ [3.0, 0.0, -1.0]))) > 0.5).astype(int)
    models_by_algo = {
        "a": [_LinearStub([3.0, 0.0, -1.0]), _LinearStub([2.5, 0.1, -1.2])],
        "b": [_LinearStub([1.0, 0.0, 0.0])],
    }
    per_repeat = ensemble_permutation_importance(
        models_by_algo, {"a": 0.7, "b": 0.3}, X, y, ["f0", "f1", "f2"],
        n_repeats=3, rng=np.random.default_rng(1),
    )
    assert len(per_repeat) == 3 and set(per_repeat[0]) == {"f0", "f1", "f2"}
    mean_imp = {k: np.mean([d[k] for d in per_repeat]) for k in per_repeat[0]}
    # The strongly-weighted feature f0 must matter more than the null feature f1.
    assert mean_imp["f0"] > mean_imp["f1"]
