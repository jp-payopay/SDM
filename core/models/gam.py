from __future__ import annotations

import numpy as np

from .base import SDMModel


class GAMModel(SDMModel):
    """n_splines=10 already matches mgcv's own basis-specific default `k`
    for a single-dimension smooth term (the package biomod2 wraps for GAM)
    — no change needed here."""

    name = "gam"
    long_name = "Generalized Additive Model"

    def __init__(self, n_splines: int = 10, **_: object) -> None:
        super().__init__(n_splines=n_splines)
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GAMModel":
        from pygam import LogisticGAM, s, terms

        p = X.shape[1]
        term = s(0, n_splines=self.hyperparams["n_splines"])
        for i in range(1, p):
            term = term + s(i, n_splines=self.hyperparams["n_splines"])
        self._model = LogisticGAM(term).fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        raw = np.asarray(self._model.predict_proba(X), dtype=np.float64)
        # pygam can numerically diverge (overflow/underflow in its link function)
        # on small or separable datasets, producing NaN/inf instead of raising.
        raw = np.nan_to_num(raw, nan=0.5, posinf=1.0, neginf=0.0)
        return np.clip(raw, 0.0, 1.0)
