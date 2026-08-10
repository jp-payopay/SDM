from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class EvaluationResult:
    auc: float
    tss: float
    boyce: float
    threshold: float

    def as_dict(self) -> dict:
        return {"auc": self.auc, "tss": self.tss, "boyce": self.boyce, "threshold": self.threshold}


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> EvaluationResult:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()
    ok = np.isfinite(y_score)
    y_true, y_score = y_true[ok], y_score[ok]
    if len(np.unique(y_true)) < 2:
        return EvaluationResult(auc=np.nan, tss=np.nan, boyce=np.nan, threshold=np.nan)
    auc = float(roc_auc_score(y_true, y_score))
    tss, thr = max_tss(y_true, y_score)
    presences = y_score[y_true == 1]
    boyce = continuous_boyce_index(presences, y_score)
    return EvaluationResult(auc=auc, tss=float(tss), boyce=float(boyce), threshold=float(thr))


def max_tss(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Sweep candidate thresholds and return (max_TSS, threshold_at_max)."""
    order = np.argsort(-y_score)
    ys = y_true[order]
    ss = y_score[order]
    n_pos = ys.sum()
    n_neg = len(ys) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    tn = n_neg - fp
    fn = n_pos - tp
    sens = tp / n_pos
    spec = tn / n_neg
    tss = sens + spec - 1.0
    best = int(np.argmax(tss))
    return float(tss[best]), float(ss[best])


def continuous_boyce_index(
    presence_scores: np.ndarray,
    all_scores: np.ndarray,
    n_bins: int = 10,
    window_frac: float = 0.1,
) -> float:
    """Continuous Boyce Index (Hirzel et al. 2006) using a moving window.

    Positive values mean predictions are consistent with presence density;
    values near 0 mean the model performs as well as random; negative values
    indicate counter-predictions.
    """
    all_scores = all_scores[np.isfinite(all_scores)]
    presence_scores = presence_scores[np.isfinite(presence_scores)]
    if len(presence_scores) < 5 or len(all_scores) < 20:
        return float("nan")
    lo = float(np.min(all_scores))
    hi = float(np.max(all_scores))
    if hi <= lo:
        return float("nan")
    window = (hi - lo) * window_frac
    centers = np.linspace(lo + window / 2, hi - window / 2, n_bins)
    pe: list[float] = []
    valid_centers: list[float] = []
    for c in centers:
        a = c - window / 2
        b = c + window / 2
        f_pred = np.mean((all_scores >= a) & (all_scores <= b))
        f_pres = np.mean((presence_scores >= a) & (presence_scores <= b))
        if f_pred <= 0 or f_pres <= 0:
            continue
        pe.append(f_pres / f_pred)
        valid_centers.append(c)
    if len(pe) < 4:
        return float("nan")
    from scipy.stats import spearmanr

    r, _ = spearmanr(valid_centers, pe)
    return float(r) if np.isfinite(r) else float("nan")
