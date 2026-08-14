from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import SDMModel


class RFModel(SDMModel):
    """Defaults already match R's randomForest package (what biomod2/sdm/
    ssdm/dismo all wrap for RF): ntree=500, mtry=sqrt(p) for classification,
    and nodesize=1 (== min_samples_leaf=1) — no change needed here."""

    name = "rf"
    long_name = "Random Forest"

    def __init__(
        self,
        n_estimators: int = 500,
        max_features: str = "sqrt",
        min_samples_leaf: int = 1,
        n_jobs: int = -1,
        random_state: int = 0,
        **_: object,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self._clf: RandomForestClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFModel":
        self._clf = RandomForestClassifier(**self.hyperparams)
        self._clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._clf.predict_proba(X)[:, 1]
