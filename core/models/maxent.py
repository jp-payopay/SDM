from __future__ import annotations

import numpy as np

from .base import SDMModel

# Sample-size-adaptive feature class selection, exactly matching maxnet's
# `maxnet.formula(classes="default")` rule (the R reimplementation of Maxent
# that elapid itself is modeled on) — this is also what dismo::maxent and
# the original Java Maxent's "auto features" use: for n<80 presence records,
# the richer product/threshold feature classes are turned off to keep the
# model from overfitting a small sample; product is re-enabled at n>=80.
# Threshold is excluded at every tier — Phillips et al. (2017) found it
# rarely helps and often hurts, so it's no longer part of any "default" tier.
def auto_feature_types(n_presence: int) -> tuple[str, ...]:
    if n_presence < 10:
        return ("linear",)
    if n_presence < 15:
        return ("linear", "quadratic")
    if n_presence < 80:
        return ("linear", "quadratic", "hinge")
    return ("linear", "quadratic", "hinge", "product")


class MaxEntModel(SDMModel):
    name = "maxent"
    long_name = "MaxEnt (elapid)"

    def __init__(
        self,
        feature_types: tuple | None = None,
        beta_multiplier: float = 1.5,
        n_hinge_features: int = 10,
        n_threads: int = 1,
        **_: object,
    ) -> None:
        # feature_types=None means "auto" — resolved from the training
        # data's presence count at fit() time, per _auto_feature_types above.
        # An explicit tuple overrides auto-selection entirely.
        super().__init__(
            feature_types=feature_types,
            beta_multiplier=beta_multiplier,
            n_hinge_features=n_hinge_features,
            n_threads=n_threads,
        )
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MaxEntModel":
        from elapid import MaxentModel as _Maxent

        feature_types = self.hyperparams["feature_types"]
        if feature_types is None:
            feature_types = auto_feature_types(int(np.sum(y == 1)))
        self._model = _Maxent(
            feature_types=list(feature_types),
            beta_multiplier=self.hyperparams["beta_multiplier"],
            n_hinge_features=self.hyperparams["n_hinge_features"],
            n_cpus=self.hyperparams["n_threads"],
            transform="cloglog",
        )
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # This elapid version fixes the output transform at construction
        # time (transform="cloglog" above) rather than accepting it here.
        self._check_fitted()
        raw = self._model.predict(X)
        return np.asarray(raw, dtype=np.float64).ravel()
