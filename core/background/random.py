from __future__ import annotations

import numpy as np
import rasterio

from ..io.rasters import RasterStack


def sample_random(
    stack: RasterStack,
    n: int,
    *,
    rng: np.random.Generator,
    max_attempts: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n random background points in valid (non-nodata) cells of the stack.

    Returns (x, y) coordinate arrays. May return fewer than n if the raster is
    dominated by nodata after several rejection rounds.
    """
    minx, miny, maxx, maxy = stack.bounds
    keep_x: list[np.ndarray] = []
    keep_y: list[np.ndarray] = []
    n_kept = 0
    with rasterio.open(stack.paths[0]) as src:
        nodata = src.nodata
        for _ in range(max_attempts):
            if n_kept >= n:
                break
            need = n - n_kept
            batch = max(need * 3, 1000)
            xs = rng.uniform(minx, maxx, size=batch)
            ys = rng.uniform(miny, maxy, size=batch)
            samples = np.asarray(list(src.sample(list(zip(xs, ys)), indexes=1))).ravel()
            if nodata is not None:
                good = (samples != nodata) & np.isfinite(samples)
            else:
                good = np.isfinite(samples)
            xs = xs[good][:need]
            ys = ys[good][:need]
            keep_x.append(xs)
            keep_y.append(ys)
            n_kept += len(xs)
    if not keep_x:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(keep_x), np.concatenate(keep_y)
