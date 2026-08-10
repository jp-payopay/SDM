from __future__ import annotations

from typing import Literal

import numpy as np

EnsembleMethod = Literal["mean", "weighted_auc", "weighted_tss"]


def compute_weights(
    algo_names: list[str],
    per_algo_metric: dict[str, float] | None,
    method: EnsembleMethod,
) -> dict[str, float]:
    """Per-algorithm ensemble weights, normalized to sum to 1. Shared by the
    raster ensemble below and the ensemble response-curve plot, so both stay
    consistent with each other."""
    if method == "mean":
        raw = {n: 1.0 for n in algo_names}
    else:
        if not per_algo_metric:
            raise ValueError(f"{method} needs per_algo_metric.")
        raw = {n: max(float(per_algo_metric.get(n, 0.0)), 0.0) for n in algo_names}
        if sum(raw.values()) <= 0:
            raw = {n: 1.0 for n in algo_names}
    total = sum(raw.values())
    return {n: v / total for n, v in raw.items()}


def ensemble_predictions(
    per_algo_predictions: dict[str, np.ndarray],
    per_algo_metric: dict[str, float] | None,
    method: EnsembleMethod = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Combine per-algorithm suitability rasters.

    Returns (ensemble, sd) — both shape matching the input rasters. `sd` is the
    across-model standard deviation per pixel (uncertainty map).
    """
    if not per_algo_predictions:
        raise ValueError("No predictions to ensemble.")
    algo_names = list(per_algo_predictions.keys())
    stack = np.stack([per_algo_predictions[n] for n in algo_names], axis=0)  # (m, H, W)

    weights = compute_weights(algo_names, per_algo_metric, method)
    w = np.array([weights[n] for n in algo_names])
    w_expanded = w.reshape(-1, 1, 1)

    # Per-pixel weights are renormalized to the algorithms that actually
    # produced a finite value at that pixel — otherwise a pixel where a
    # (possibly heavily-weighted) algorithm failed would have its remaining
    # weights implicitly sum to < 1, silently understating suitability
    # exactly where a model failed.
    finite = np.isfinite(stack)
    valid_mask = np.any(finite, axis=0)
    w_valid = np.where(finite, w_expanded, 0.0)
    weight_sum = w_valid.sum(axis=0)
    safe_weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    stack_filled = np.where(finite, stack, 0.0)
    mean = (stack_filled * w_valid).sum(axis=0) / safe_weight_sum
    mean = np.where(valid_mask, mean, np.nan)
    sd = np.nanstd(stack, axis=0)
    sd = np.where(valid_mask, sd, np.nan)
    return mean.astype(np.float32), sd.astype(np.float32)


def ensemble_permutation_importance(
    models_by_algo: dict[str, list],
    weights: dict[str, float],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 3,
    rng: np.random.Generator | None = None,
) -> list[dict[str, float]]:
    """Permutation importance of the *ensemble* prediction (biomod2-style):
    permute one predictor at a time, recompute the full weighted ensemble
    prediction from every fitted model, and measure the drop in ensemble AUC.

    The ensemble prediction is the same object the raster ensemble represents:
    per algorithm, the mean over that algorithm's replicate models, then a
    weighted mean across algorithms using `weights`. Returns one dict per
    repeat (feature -> AUC drop) so the caller can show mean +/- SD, matching
    the per-algorithm importance plots.
    """
    from sklearn.metrics import roc_auc_score

    if rng is None:
        rng = np.random.default_rng()
    algos = [a for a in models_by_algo if models_by_algo[a]]
    total_w = sum(max(float(weights.get(a, 0.0)), 0.0) for a in algos)
    if total_w <= 0:
        w = {a: 1.0 / len(algos) for a in algos}
    else:
        w = {a: max(float(weights.get(a, 0.0)), 0.0) / total_w for a in algos}

    def ensemble_predict(x_in: np.ndarray) -> np.ndarray:
        total = np.zeros(len(x_in), dtype=float)
        for a in algos:
            algo_mean = np.mean([m.predict_proba(x_in) for m in models_by_algo[a]], axis=0)
            total += w[a] * algo_mean
        return total

    baseline = roc_auc_score(y, ensemble_predict(X))
    per_repeat: list[dict[str, float]] = []
    for _ in range(n_repeats):
        drops: dict[str, float] = {}
        for i, name in enumerate(feature_names):
            xp = X.copy()
            xp[:, i] = rng.permutation(xp[:, i])
            drops[name] = float(baseline - roc_auc_score(y, ensemble_predict(xp)))
        per_repeat.append(drops)
    return per_repeat
