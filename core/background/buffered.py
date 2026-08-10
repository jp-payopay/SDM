from __future__ import annotations

import numpy as np
import rasterio

from ..io.rasters import RasterStack


def sample_buffered(
    stack: RasterStack,
    presence_x: np.ndarray,
    presence_y: np.ndarray,
    n: int,
    buffer_distance: float,
    *,
    rng: np.random.Generator,
    max_attempts: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n background points within `buffer_distance` of any presence,
    excluding nodata cells. Distance is in the CRS units of the raster stack.
    """
    minx, miny, maxx, maxy = stack.bounds
    minx = max(minx, presence_x.min() - buffer_distance)
    maxx = min(maxx, presence_x.max() + buffer_distance)
    miny = max(miny, presence_y.min() - buffer_distance)
    maxy = min(maxy, presence_y.max() + buffer_distance)

    keep_x: list[np.ndarray] = []
    keep_y: list[np.ndarray] = []
    n_kept = 0
    buf_sq = buffer_distance ** 2
    with rasterio.open(stack.paths[0]) as src:
        nodata = src.nodata
        for _ in range(max_attempts):
            if n_kept >= n:
                break
            need = n - n_kept
            batch = max(need * 5, 2000)
            xs = rng.uniform(minx, maxx, size=batch)
            ys = rng.uniform(miny, maxy, size=batch)
            near = _within_buffer(xs, ys, presence_x, presence_y, buf_sq)
            xs = xs[near]
            ys = ys[near]
            if len(xs) == 0:
                continue
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


def _within_buffer(
    xs: np.ndarray,
    ys: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    buf_sq: float,
) -> np.ndarray:
    """Return boolean mask of candidates within squared buffer distance of any presence.
    Uses a chunked pairwise-distance calc to bound memory.
    """
    n = len(xs)
    inside = np.zeros(n, dtype=bool)
    chunk = 2048
    for i in range(0, n, chunk):
        dx = xs[i : i + chunk, None] - px[None, :]
        dy = ys[i : i + chunk, None] - py[None, :]
        d2 = dx * dx + dy * dy
        inside[i : i + chunk] = (d2 <= buf_sq).any(axis=1)
    return inside
