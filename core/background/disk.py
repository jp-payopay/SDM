from __future__ import annotations

import numpy as np
import rasterio

from ..io.rasters import RasterStack


def sample_disk(
    stack: RasterStack,
    presence_x: np.ndarray,
    presence_y: np.ndarray,
    n: int,
    min_distance: float,
    max_distance: float,
    *,
    rng: np.random.Generator,
    max_attempts: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n background points whose distance to the *nearest* presence falls
    between `min_distance` and `max_distance`.

    Both distances are in the CRS units of the raster stack (the caller
    converts from metres; see stages.collect_labeled_points_and_extract).
    `max_distance <= 0` means no upper limit, so the whole valid extent
    outside the inner radius is fair game. Nodata cells are excluded.

    The inner radius is the point of the method: points immediately beside a
    presence record are often unrecorded presences rather than genuine
    absences, and treating them as absences teaches the model that suitable
    conditions are unsuitable.
    """
    if min_distance < 0:
        raise ValueError("min_distance cannot be negative.")
    if max_distance > 0 and min_distance >= max_distance:
        raise ValueError(
            f"min_distance ({min_distance:g}) must be smaller than max_distance "
            f"({max_distance:g}); use max_distance = 0 for no upper limit."
        )

    minx, miny, maxx, maxy = stack.bounds
    if max_distance > 0:
        # Nothing beyond the presences' bounding box plus the outer radius can
        # ever qualify, so don't waste candidates out there.
        minx = max(minx, presence_x.min() - max_distance)
        maxx = min(maxx, presence_x.max() + max_distance)
        miny = max(miny, presence_y.min() - max_distance)
        maxy = min(maxy, presence_y.max() + max_distance)
    if maxx <= minx or maxy <= miny:
        raise ValueError(
            "The disk around the presence points falls outside the predictor "
            "rasters. Check that the occurrences and rasters overlap."
        )

    min_sq = min_distance ** 2
    max_sq = np.inf if max_distance <= 0 else max_distance ** 2

    keep_x: list[np.ndarray] = []
    keep_y: list[np.ndarray] = []
    n_kept = 0
    with rasterio.open(stack.paths[0]) as src:
        nodata = src.nodata
        for _ in range(max_attempts):
            if n_kept >= n:
                break
            need = n - n_kept
            batch = max(need * 5, 2000)
            xs = rng.uniform(minx, maxx, size=batch)
            ys = rng.uniform(miny, maxy, size=batch)
            in_band = _within_distance_band(xs, ys, presence_x, presence_y, min_sq, max_sq)
            xs = xs[in_band]
            ys = ys[in_band]
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

    if n_kept == 0:
        raise ValueError(
            "No background points fell between the minimum and maximum distance "
            "from any presence. The ring is probably empty — either the minimum "
            "distance reaches past the edge of the rasters, or the two distances "
            "are too close together to leave room for anything."
        )
    return np.concatenate(keep_x), np.concatenate(keep_y)


def _within_distance_band(
    xs: np.ndarray,
    ys: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    min_sq: float,
    max_sq: float,
) -> np.ndarray:
    """Mask of candidates whose nearest presence lies within [min, max].

    Distance to the *nearest* presence, not to any presence: a candidate 5 km
    from one record but 500 m from another is 500 m from the presences, and
    an inner radius of 1 km must reject it. Chunked to bound memory on large
    presence sets.
    """
    n = len(xs)
    keep = np.zeros(n, dtype=bool)
    chunk = 2048
    for i in range(0, n, chunk):
        dx = xs[i : i + chunk, None] - px[None, :]
        dy = ys[i : i + chunk, None] - py[None, :]
        nearest_sq = (dx * dx + dy * dy).min(axis=1)
        keep[i : i + chunk] = (nearest_sq >= min_sq) & (nearest_sq <= max_sq)
    return keep
