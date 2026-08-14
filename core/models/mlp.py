from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import SDMModel


class MLPModel(SDMModel):
    """biomod2's ANN wraps nnet — a single hidden layer — and picks its
    size/decay via internal 5-fold CV over size ∈ {2,4,6,8} and
    decay ∈ {0.001, 0.01, 0.05, 0.1} (BIOMOD_tuning defaults). We don't run
    that inner CV loop here, so hidden_layer_sizes/alpha are fixed at the
    upper-middle of those same grids (a single 8-unit layer, decay 0.01)
    rather than the much larger, deeper (32, 16) network used previously —
    ecological presence/background datasets are usually far too small to
    support that much network capacity without overfitting. max_iter=500
    matches BIOMOD_tuning's maxit.ANN exactly.
    """

    name = "mlp"
    long_name = "Multi-Layer Perceptron"

    def __init__(
        self,
        hidden_layer_sizes: tuple = (8,),
        max_iter: int = 500,
        alpha: float = 0.01,
        random_state: int = 0,
        **_: object,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            alpha=alpha,
            random_state=random_state,
        )
        self._pipe: Pipeline | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPModel":
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(**self.hyperparams)),
        ])
        self._pipe.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._pipe.predict_proba(X)[:, 1]
