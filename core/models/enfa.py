from __future__ import annotations

import numpy as np

from .base import SDMModel


class ENFAModel(SDMModel):
    """Ecological Niche Factor Analysis (Hirzel et al. 2002), simplified.

    Rather than exposing marginality/specialization axes, we compute habitat
    suitability as a monotone decreasing function of Mahalanobis distance from
    the species centroid in the standardized global environmental space.

    Standardization is done using global mean/std of all provided X rows
    (which include both presences and background/absences), matching the
    'available environment' definition used in the original ENFA.
    """

    name = "enfa"
    long_name = "Ecological Niche Factor Analysis"

    def __init__(self, ridge: float = 1e-6, **_: object) -> None:
        super().__init__(ridge=ridge)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._species_mean: np.ndarray | None = None
        self._species_cov_inv: np.ndarray | None = None
        self._d_scale: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ENFAModel":
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0, ddof=0)
        self._std = np.where(self._std < 1e-12, 1.0, self._std)
        Xs = (X - self._mean) / self._std
        pres = Xs[y.astype(bool)]
        if len(pres) < 2:
            raise ValueError("ENFA requires at least 2 presence points.")
        self._species_mean = pres.mean(axis=0)
        cov = np.cov(pres.T)
        cov = np.atleast_2d(cov)
        cov = cov + np.eye(cov.shape[0]) * self.hyperparams["ridge"]
        self._species_cov_inv = np.linalg.pinv(cov)
        d_pres = self._mahalanobis(pres)
        self._d_scale = float(np.percentile(d_pres, 95))
        if self._d_scale <= 0:
            self._d_scale = 1.0
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._mean is not None
        Xs = (X - self._mean) / self._std
        d = self._mahalanobis(Xs)
        s = np.exp(-d / self._d_scale)
        return np.clip(s, 0.0, 1.0)

    def _mahalanobis(self, Xs: np.ndarray) -> np.ndarray:
        diff = Xs - self._species_mean
        left = diff @ self._species_cov_inv
        return np.sqrt(np.maximum(np.einsum("ij,ij->i", left, diff), 0.0))
