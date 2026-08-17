from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .base import SDMModel


class GBMModel(SDMModel):
    """Boosted regression trees (BRT), the term this algorithm goes by in
    the SDM literature (Elith, Leathwick & Hastie 2008). Defaults follow
    the usual BRT recipe for ecological data: a slow learning rate of 0.01
    with bagging at 0.75, and n_estimators raised to 1000 to compensate for
    that low rate, since the standard guidance is to aim for at least ~1000
    trees. That guidance normally comes with adaptive tree-count selection
    by internal CV, which is not replicated here. Tree depth is kept at a
    fixed 3, within the 2-5 range Elith et al. recommend for typical
    ecological data, rather than the depth-1 stumps some implementations
    default to: without adaptive tree-count selection, stumps would need far
    more trees than we fit to reach comparable complexity.
    """

    name = "gbm"
    long_name = "Gradient Boosting Machine (BRT)"

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.01,
        max_depth: int = 3,
        subsample: float = 0.75,
        random_state: int = 0,
        **_: object,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
        )
        self._clf: GradientBoostingClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBMModel":
        self._clf = GradientBoostingClassifier(**self.hyperparams)
        self._clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._clf.predict_proba(X)[:, 1]
