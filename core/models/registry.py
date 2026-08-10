from __future__ import annotations

from .base import SDMModel
from .enfa import ENFAModel
from .gam import GAMModel
from .gbm import GBMModel
from .lr import LRModel
from .maxent import MaxEntModel
from .mlp import MLPModel
from .rf import RFModel
from .svm import SVMModel
from .xgb import XGBModel

_REGISTRY: dict[str, type[SDMModel]] = {
    "lr": LRModel,
    "gam": GAMModel,
    "rf": RFModel,
    "gbm": GBMModel,
    "xgb": XGBModel,
    "svm": SVMModel,
    "mlp": MLPModel,
    "maxent": MaxEntModel,
    "enfa": ENFAModel,
}


def build_model(name: str, random_state: int = 0, **overrides) -> SDMModel:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown algorithm: {name!r}. Options: {list(_REGISTRY)}")
    cls = _REGISTRY[name]
    kwargs: dict = {"random_state": random_state, **overrides}
    return cls(**kwargs)


def algorithm_long_name(name: str) -> str:
    return _REGISTRY[name].long_name if name in _REGISTRY else name


def list_algorithms() -> list[str]:
    return list(_REGISTRY)


def default_hyperparameters(name: str) -> dict:
    """The hyperparameters a fresh model of this type is constructed with —
    i.e. exactly what build_model(name) uses unless overridden. Cheap to
    call (just __init__, no fitting), safe to use for a pre-run "what will
    this actually use" preview. A None value (currently only MaxEnt's
    feature_types) means the model resolves it adaptively at fit() time
    from the training data itself, not a fixed constant.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown algorithm: {name!r}. Options: {list(_REGISTRY)}")
    return dict(_REGISTRY[name]().hyperparams)
