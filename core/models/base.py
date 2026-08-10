from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class SDMModel(ABC):
    """Common interface every SDM algorithm implements.

    Subclasses expose fit / predict_proba on (X, y) where X is (n, p) predictors
    (no NaNs) and y is (n,) 0/1 presence/absence or presence/background.
    """

    name: str = "base"
    long_name: str = "Base SDM model"

    def __init__(self, **hyperparams: Any) -> None:
        self.hyperparams = hyperparams
        self.feature_names: list[str] = []
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SDMModel":
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of presence, shape (n,), values in [0, 1]."""
        ...

    def set_feature_names(self, names: list[str]) -> None:
        self.feature_names = list(names)

    def permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 5,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        from sklearn.metrics import roc_auc_score

        if rng is None:
            rng = np.random.default_rng()
        baseline = roc_auc_score(y, self.predict_proba(X))
        importances: dict[str, float] = {}
        for i, name in enumerate(self.feature_names or [f"x{i}" for i in range(X.shape[1])]):
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[:, i] = rng.permutation(Xp[:, i])
                drops.append(baseline - roc_auc_score(y, self.predict_proba(Xp)))
            importances[name] = float(np.mean(drops))
        return importances

    def partial_dependence(
        self,
        X: np.ndarray,
        feature_idx: int,
        n_grid: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute a 1D partial dependence curve for `feature_idx` by holding
        every other feature at its median and varying the target over its range.
        Returns (grid_values, predicted_probabilities).
        """
        med = np.nanmedian(X, axis=0)
        col = X[:, feature_idx]
        lo, hi = np.nanpercentile(col, 2.5), np.nanpercentile(col, 97.5)
        grid = np.linspace(lo, hi, n_grid)
        Xg = np.tile(med, (n_grid, 1))
        Xg[:, feature_idx] = grid
        preds = self.predict_proba(Xg)
        return grid, preds
