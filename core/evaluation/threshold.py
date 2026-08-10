from __future__ import annotations

import numpy as np

from .metrics import max_tss


def maxtss_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return the threshold that maximizes TSS on the given labels/scores."""
    _, thr = max_tss(np.asarray(y_true, int), np.asarray(y_score, float))
    return thr


def apply_threshold(raster: np.ndarray, threshold: float) -> np.ndarray:
    """Return a uint8 binary raster (1 = suitable, 0 = unsuitable), setting NaN to 0.

    Suitable for in-memory use and for plotting (where the caller supplies its
    own mask). For a written GeoTIFF that should stay clipped to the data mask,
    use apply_threshold_masked instead so nodata cells are not filled with 0.
    """
    out = np.zeros(raster.shape, dtype=np.uint8)
    mask = np.isfinite(raster)
    out[mask] = (raster[mask] >= threshold).astype(np.uint8)
    return out


def apply_threshold_masked(
    raster: np.ndarray, threshold: float, nodata: int = 255
) -> np.ndarray:
    """Like apply_threshold, but non-finite (masked) cells become `nodata`
    rather than 0, so a written binary raster stays clipped to the data mask
    instead of filling the whole extent with 'unsuitable'."""
    out = np.full(raster.shape, nodata, dtype=np.uint8)
    mask = np.isfinite(raster)
    out[mask] = (raster[mask] >= threshold).astype(np.uint8)
    return out
