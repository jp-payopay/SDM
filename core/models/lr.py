from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .base import SDMModel


def _add_quadratic_terms(X: np.ndarray) -> np.ndarray:
    """Append each (already-scaled) predictor's own square, with no
    cross-variable interaction terms. This is the standard "GLM"
    configuration in the SDM literature since Guisan & Zimmermann (2000) /
    Austin (2002). Squaring after scaling (not before) keeps the
    squared terms on a comparable numeric scale regardless of each
    predictor's original units.
    """
    return np.hstack([X, X**2])


class LRModel(SDMModel):
    name = "lr"
    long_name = "Logistic Regression (GLM)"

    def __init__(self, C: float = 1.0, max_iter: int = 500, **_: object) -> None:
        super().__init__(C=C, max_iter=max_iter)
        self._pipe: Pipeline | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LRModel":
        # The classic GLM recipe selects terms by stepwise AIC rather than
        # shrinking them; sklearn has no direct stepwise-AIC equivalent, so
        # L2 regularization (C) plays that same "keep the model from
        # overfitting the added quadratic terms" role here instead.
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("quadratic", FunctionTransformer(_add_quadratic_terms)),
            ("clf", LogisticRegression(
                C=self.hyperparams["C"],
                max_iter=self.hyperparams["max_iter"],
                solver="lbfgs",
            )),
        ])
        self._pipe.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._pipe.predict_proba(X)[:, 1]
