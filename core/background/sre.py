"""Surface Range Envelope (SRE) pseudo-absence selection.

The idea: build a rectilinear envelope around the environmental conditions the
species was actually recorded in — one independent min/max interval per
predictor — and draw pseudo-absences only from places falling *outside* it.
Somewhere is outside the envelope as soon as a single predictor is beyond its
interval, so the envelope is a box in environmental space and the sampling
region is everywhere else.

This is the same envelope BIOCLIM/SRE models use as a predictor, turned
around: instead of calling the inside of the box "suitable", the outside is
treated as confidently unsuitable and used to source absences.

The assumption it rests on is worth stating plainly, because it is easy to
violate: it only holds when the records already cover most of the species'
environmental space. If sampling has missed part of the niche, the envelope
is too small, and genuinely suitable conditions end up supplying the
"absences" — which is worse than random background, not better.
"""

from __future__ import annotations

from contextlib import ExitStack

import numpy as np
import rasterio

from ..io.rasters import RasterStack, extract_values


def presence_envelope(
    stack: RasterStack,
    presence_x: np.ndarray,
    presence_y: np.ndarray,
    quantile: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-predictor (lower, upper) bounds of the presence records' conditions.

    `quantile` trims that fraction off each end before taking the bounds, so a
    single mislocated or atypical record cannot inflate the envelope to cover
    the whole study area. The default of 0.025 discards the outer 2.5% at
    each end; 0 uses the outright min and max.
    """
    if not 0.0 <= quantile < 0.5:
        raise ValueError("SRE quantile must be at least 0 and below 0.5.")
    values = extract_values(stack, presence_x, presence_y)
    usable = values[np.all(np.isfinite(values), axis=1)]
    if len(usable) < 2:
        raise ValueError(
            "SRE needs at least 2 presence points with valid predictor values "
            "to build an environmental envelope. Check that the occurrences "
            "fall inside the rasters and away from nodata."
        )
    lower = np.quantile(usable, quantile, axis=0)
    upper = np.quantile(usable, 1.0 - quantile, axis=0)
    return lower, upper


def sample_sre(
    stack: RasterStack,
    presence_x: np.ndarray,
    presence_y: np.ndarray,
    n: int,
    *,
    quantile: float = 0.025,
    rng: np.random.Generator,
    max_attempts: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n pseudo-absences from outside the presences' environmental
    envelope. Returns (x, y) in the stack's CRS.
    """
    lower, upper = presence_envelope(stack, presence_x, presence_y, quantile)
    minx, miny, maxx, maxy = stack.bounds

    keep_x: list[np.ndarray] = []
    keep_y: list[np.ndarray] = []
    n_kept = 0
    with ExitStack() as open_files:
        # Every band has to be read for each candidate, so the datasets are
        # opened once for the whole rejection loop rather than per batch.
        sources = [open_files.enter_context(rasterio.open(p)) for p in stack.paths]
        for _ in range(max_attempts):
            if n_kept >= n:
                break
            need = n - n_kept
            batch = max(need * 5, 2000)
            xs = rng.uniform(minx, maxx, size=batch)
            ys = rng.uniform(miny, maxy, size=batch)
            values = _sample_all_bands(sources, xs, ys)
            valid = np.all(np.isfinite(values), axis=1)
            # Outside the envelope = beyond its interval on at least one
            # predictor. Requiring *every* predictor to be outside would
            # describe a far smaller and much stranger region.
            outside = np.any((values < lower) | (values > upper), axis=1)
            good = valid & outside
            xs = xs[good][:need]
            ys = ys[good][:need]
            keep_x.append(xs)
            keep_y.append(ys)
            n_kept += len(xs)

    if n_kept == 0:
        raise ValueError(
            "No locations fell outside the presences' environmental envelope, so "
            "SRE has nowhere to draw pseudo-absences from. The records already "
            "span the full range of every predictor across this study area — "
            "try a larger extent, a higher SRE quantile, or the random or disk "
            "method instead."
        )
    return np.concatenate(keep_x), np.concatenate(keep_y)


def _sample_all_bands(sources, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """(n_points, n_bands) values at the given coordinates, with each band's
    own nodata turned into NaN — matching io.rasters.extract_values, which is
    what the modeling stage itself uses."""
    coords = list(zip(xs, ys))
    out = np.full((len(xs), len(sources)), np.nan, dtype=np.float64)
    for i, src in enumerate(sources):
        column = np.asarray([s[0] for s in src.sample(coords, indexes=1)], dtype=np.float64)
        if src.nodata is not None:
            column = np.where(column == src.nodata, np.nan, column)
        out[:, i] = column
    return out
