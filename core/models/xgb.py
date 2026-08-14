from __future__ import annotations

import numpy as np

from .base import SDMModel


class XGBModel(SDMModel):
    """Defaults match biomod2's own XGBOOST "default" strategy (max.depth=5,
    eta=0.1, nrounds=512) — biomod2 4.x is the only one of the R SDM
    packages surveyed (sdm, biomod2, ssdm, flexsdm, dismo) that wraps
    XGBoost. subsample/colsample_bytree aren't part of that reference
    default; 0.8/0.8 is standard boosted-tree practice and left unchanged.
    """

    name = "xgb"
    long_name = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 0,
        **_: object,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
        )
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBModel":
        from xgboost import XGBClassifier

        self._clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            **self.hyperparams,
        )
        self._clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._clf.predict_proba(X)[:, 1]
