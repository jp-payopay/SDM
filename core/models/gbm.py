from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .base import SDMModel


class GBMModel(SDMModel):
    """Boosted regression trees (BRT), the term this algorithm goes by in
    the SDM literature (Elith, Leathwick & Hastie 2008). Defaults follow
    dismo::gbm.step's own function defaults (learning_rate=0.01,
    subsample/bag.fraction=0.75), with n_estimators raised to 1000 to
    compensate for the lower learning rate (gbm.step's guidance is to aim
    for at least ~1000 trees; it finds the tree count itself via internal
    CV, which we don't replicate here). tree depth is kept at a fixed 3 —
    within the 2-5 range Elith et al. recommend for typical ecological
    data — rather than gbm.step's own tree.complexity=1 default, since
    without adaptive tree-count selection a depth-1 (stump) ensemble would
    need many more trees than we fit to reach comparable complexity.
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
        assert self._clf is not None
        return self._clf.predict_proba(X)[:, 1]
