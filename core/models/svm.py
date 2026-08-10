from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .base import SDMModel


class SVMModel(SDMModel):
    """cost=1, RBF kernel matches e1071::svm()'s own defaults (what
    biomod2/sdm wrap for SVM); gamma="scale" is sklearn's own heuristic
    (1 / (n_features * X.var())), the same "one over feature count" spirit
    as e1071's default gamma — no change needed here."""

    name = "svm"
    long_name = "Support Vector Machine"

    def __init__(
        self,
        C: float = 1.0,
        gamma: str = "scale",
        kernel: str = "rbf",
        random_state: int = 0,
        **_: object,
    ) -> None:
        super().__init__(C=C, gamma=gamma, kernel=kernel, random_state=random_state)
        self._pipe: Pipeline | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMModel":
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, **self.hyperparams)),
        ])
        self._pipe.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._pipe is not None
        return self._pipe.predict_proba(X)[:, 1]
