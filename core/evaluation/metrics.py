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


@dataclass
class FoldEvaluation:
    """One cross-validation fold's own held-out scores, plus how much data it
    was scored on.

    The headline metrics a run reports are computed on held-out predictions
    *pooled* across folds (one score per replicate). These per-fold results are
    the same evaluation applied fold by fold, which is what shows whether a
    replicate's pooled score rests on folds that agree or on one good fold and
    one bad one — the distinction that matters most under spatial-block CV,
    where each fold is a different region.

    `fold` is 1-based. `n_background_test` counts held-out background points in
    presence-only mode and true absences in presence-absence mode.
    `n_blocks` is the number of spatial blocks making up the fold, and is None
    for the non-spatial split methods, which have no blocks.
    """

    fold: int
    n_train: int
    n_test: int
    n_presence_test: int
    n_background_test: int
    metrics: EvaluationResult
    n_blocks: int | None = None

    def as_dict(self) -> dict:
        return {
            "fold": self.fold,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_presence_test": self.n_presence_test,
            "n_background_test": self.n_background_test,
            "n_blocks": self.n_blocks,
            **self.metrics.as_dict(),
        }


def evaluate_fold(
    fold: int,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_train: int,
    n_blocks: int | None = None,
) -> FoldEvaluation:
    """Score one fold's held-out predictions and record its size."""
    y_true = np.asarray(y_true).astype(int).ravel()
    return FoldEvaluation(
        fold=fold,
        n_train=int(n_train),
        n_test=int(y_true.size),
        n_presence_test=int((y_true == 1).sum()),
        n_background_test=int((y_true == 0).sum()),
        metrics=evaluate(y_true, y_score),
        n_blocks=n_blocks,
    )


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
