from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..io.rasters import RasterStack, iter_windows
from ..models.base import SDMModel


def predict_raster(
    model: SDMModel,
    stack: RasterStack,
    *,
    kept_feature_idx: list[int] | None = None,
    block_rows: int = 256,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Apply a fitted model across the raster stack. Returns a (H, W) float32
    suitability array; NaN where inputs are nodata.

    `kept_feature_idx` selects which stack bands map to which trained-feature index
    (post-VIF). If None, all bands are used in order.
    """
    if kept_feature_idx is None:
        kept_feature_idx = list(range(len(stack.paths)))
    out = np.full(stack.shape, np.nan, dtype=np.float32)
    total = stack.height
    for row_off, h, arr in iter_windows(stack, block_rows=block_rows):
        sel = arr[kept_feature_idx]  # (p, h, w)
        flat = sel.reshape(sel.shape[0], -1).T  # (h*w, p)
        valid = np.all(np.isfinite(flat), axis=1)
        preds = np.full(flat.shape[0], np.nan, dtype=np.float32)
        if valid.any():
            preds[valid] = model.predict_proba(flat[valid]).astype(np.float32)
        out[row_off : row_off + h] = preds.reshape(h, stack.width)
        if progress:
            progress(min(1.0, (row_off + h) / total))
    return out
