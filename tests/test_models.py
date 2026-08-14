import numpy as np
import pytest

from sdm_plugin.core.models.config_export import build_model_config
from sdm_plugin.core.models.maxent import auto_feature_types
from sdm_plugin.core.models.registry import build_model, default_hyperparameters, list_algorithms

SKIPPABLE = {"gam": "pygam", "xgb": "xgboost", "maxent": "elapid"}


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.uint8)
    return X, y


@pytest.mark.parametrize("algo", list_algorithms())
def test_model_fits_and_predicts(algo, toy_data):
    dep = SKIPPABLE.get(algo)
    if dep:
        pytest.importorskip(dep)
    X, y = toy_data
    model = build_model(algo, random_state=0)
    model.set_feature_names(["a", "b", "c"])
    model.fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (X.shape[0],)
    assert np.all((p >= 0) & (p <= 1))


@pytest.mark.parametrize("algo", list_algorithms())
def test_predict_proba_before_fit_raises_clear_error(algo, toy_data):
    """Regression test: predict_proba() used to guard against an unfitted
    model with `assert self._x is not None` — stripped entirely under
    python -O/PYTHONOPTIMIZE, which would silently skip the check instead of
    failing clearly. The shared SDMModel._check_fitted() must raise a real,
    always-present error instead.
    """
    dep = SKIPPABLE.get(algo)
    if dep:
        pytest.importorskip(dep)
    X, _y = toy_data
    model = build_model(algo, random_state=0)
    with pytest.raises(RuntimeError, match="must be fit"):
        model.predict_proba(X)


def test_permutation_importance_returns_all_features(toy_data):
    X, y = toy_data
    model = build_model("lr", random_state=0)
    model.set_feature_names(["a", "b", "c"])
    model.fit(X, y)
    imp = model.permutation_importance(X, y, n_repeats=2)
    assert set(imp) == {"a", "b", "c"}


def test_partial_dependence_shape(toy_data):
    X, y = toy_data
    model = build_model("rf", random_state=0)
    model.set_feature_names(["a", "b", "c"])
    model.fit(X, y)
    grid, preds = model.partial_dependence(X, feature_idx=0, n_grid=25)
    assert grid.shape == preds.shape == (25,)


@pytest.mark.parametrize("algo", list_algorithms())
def test_default_hyperparameters_returns_dict(algo):
    params = default_hyperparameters(algo)
    assert isinstance(params, dict)
    assert len(params) > 0


@pytest.mark.parametrize(
    "n_presence, expected",
    [
        (5, ("linear",)),
        (12, ("linear", "quadratic")),
        (50, ("linear", "quadratic", "hinge")),
        (200, ("linear", "quadratic", "hinge", "product")),
    ],
)
def test_maxent_auto_feature_types(n_presence, expected):
    assert auto_feature_types(n_presence) == expected


def test_build_model_config_resolves_maxent_feature_types():
    cfg = build_model_config(["lr", "maxent"], n_presence=42)
    assert set(cfg) == {"lr", "maxent"}
    assert cfg["maxent"]["feature_types"] == ["linear", "quadratic", "hinge"]
    assert "feature_types_note" in cfg["maxent"]


def test_build_model_config_without_n_presence_describes_auto_rule():
    cfg = build_model_config(["maxent"])
    assert isinstance(cfg["maxent"]["feature_types"], str)
    assert "auto" in cfg["maxent"]["feature_types"]
