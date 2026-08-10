from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..io.rasters import RasterStack, iter_windows


def mess(
    stack: RasterStack,
    training_values: np.ndarray,
    *,
    kept_feature_idx: list[int] | None = None,
    block_rows: int = 256,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Multivariate Environmental Similarity Surface (Elith et al. 2010).

    For each pixel, compute the per-predictor similarity to the training range:
      - 100 if value equals training median
      - Interpolated between min and max of training set
      - Negative when outside training range (linearly extrapolated)
    Return the minimum similarity across predictors — negative values flag
    novel (extrapolated) environments.
    """
    if kept_feature_idx is None:
        kept_feature_idx = list(range(training_values.shape[1]))
    # training_values (X_kept) is already reduced to just the kept columns,
    # in kept_feature_idx order — it must NOT be re-indexed by
    # kept_feature_idx a second time (those are positions into the raster
    # stack's full, unreduced band list, used below for `arr`/`sel`).
    tv = training_values
    tmin = np.nanmin(tv, axis=0)
    tmax = np.nanmax(tv, axis=0)
    n = tv.shape[0]

    out = np.full(stack.shape, np.nan, dtype=np.float32)
    total = stack.height
    for row_off, h, arr in iter_windows(stack, block_rows=block_rows):
        sel = arr[kept_feature_idx]  # (p, h, w)
        p, hh, w = sel.shape
        block_min = np.full((hh, w), np.inf, dtype=np.float32)
        valid = np.all(np.isfinite(sel), axis=0)
        for i in range(p):
            v = sel[i]
            col = tv[:, i]
            col_sorted = np.sort(col[np.isfinite(col)])
            if len(col_sorted) == 0:
                continue
            f = np.searchsorted(col_sorted, v.ravel(), side="right").astype(np.float32)
            f_below = 100.0 * f / n
            f_above = 100.0 * (n - f) / n
            sim = np.minimum(f_below, f_above) * 2.0  # in-range: 0..100
            below = v.ravel() < tmin[i]
            above = v.ravel() > tmax[i]
            span = max(tmax[i] - tmin[i], 1e-9)
            sim_below = (v.ravel() - tmin[i]) / span * 100.0
            sim_above = (tmax[i] - v.ravel()) / span * 100.0
            sim = np.where(below, sim_below, sim)
            sim = np.where(above, sim_above, sim)
            sim = sim.reshape(hh, w)
            block_min = np.minimum(block_min, sim)
        block_min = np.where(valid, block_min, np.nan)
        out[row_off : row_off + h] = block_min
        if progress:
            progress(min(1.0, (row_off + h) / total))
    return out


def mop(
    stack: RasterStack,
    training_values: np.ndarray,
    *,
    kept_feature_idx: list[int] | None = None,
    percentile: float = 10.0,
    block_rows: int = 256,
) -> np.ndarray:
    """Mobility-Oriented Parity (Owens et al. 2013), simplified.

    For each pixel, compute the mean Euclidean distance (in standardized space)
    to the closest `percentile` fraction of training points. Larger values
    indicate greater dissimilarity from calibration data.
    """
    if kept_feature_idx is None:
        kept_feature_idx = list(range(training_values.shape[1]))
    # See the matching comment in mess(): training_values is already reduced
    # to the kept columns, so it must not be re-indexed by kept_feature_idx.
    tv = training_values
    mu = np.nanmean(tv, axis=0)
    sd = np.nanstd(tv, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    tv_s = (tv - mu) / sd
    k = max(1, int(round(len(tv_s) * percentile / 100.0)))

    out = np.full(stack.shape, np.nan, dtype=np.float32)
    for row_off, h, arr in iter_windows(stack, block_rows=block_rows):
        sel = arr[kept_feature_idx]  # (p, h, w)
        flat = sel.reshape(sel.shape[0], -1).T  # (h*w, p)
        finite = np.all(np.isfinite(flat), axis=1)
        vals = np.full(flat.shape[0], np.nan, dtype=np.float32)
        if finite.any():
            fs = (flat[finite] - mu) / sd
            chunk = 4096
            m = len(fs)
            distmean = np.empty(m, dtype=np.float32)
            for i in range(0, m, chunk):
                d = np.linalg.norm(fs[i : i + chunk, None, :] - tv_s[None, :, :], axis=2)
                d.sort(axis=1)
                distmean[i : i + chunk] = d[:, :k].mean(axis=1)
            vals[finite] = distmean
        out[row_off : row_off + h] = vals.reshape(h, stack.width)
    return out
