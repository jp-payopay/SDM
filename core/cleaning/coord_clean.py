from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..io.occurrences import OccurrenceData
from ..io.rasters import RasterStack, extract_values


@dataclass
class CleaningReport:
    n_input: int = 0
    n_output: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    kept_mask: np.ndarray | None = None

    def as_dict(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_output": self.n_output,
            "dropped": self.dropped,
        }


def auto_clean(
    data: OccurrenceData,
    stack: RasterStack,
) -> tuple[OccurrenceData, CleaningReport]:
    """Apply automatic coordinate cleaning against a raster stack extent.

    Rules (applied in order, each rule sees the already-filtered set):
      1. Drop rows with NaN coordinates.
      2. Drop exact (x, y) duplicates.
      3. Drop (0, 0) coordinates.
      4. Drop points outside the raster stack bounds.
      5. Drop points that fall on nodata cells in ANY predictor
         (covers ocean when using terrestrial predictors).
    """
    n = len(data.x)
    keep = np.ones(n, dtype=bool)
    dropped: dict[str, int] = {}

    def _drop(mask: np.ndarray, label: str) -> None:
        removed = int(keep.sum() - (keep & mask).sum())
        if removed:
            dropped[label] = removed
        keep[:] = keep & mask

    valid_coords = ~(np.isnan(data.x) | np.isnan(data.y))
    _drop(valid_coords, "nan_coords")

    seen: dict[tuple[float, float], int] = {}
    dup_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        key = (float(data.x[i]), float(data.y[i]))
        if key in seen:
            dup_mask[i] = False
        else:
            seen[key] = i
    _drop(dup_mask, "duplicate")

    nonzero = ~((data.x == 0) & (data.y == 0))
    _drop(nonzero, "zero_zero")

    minx, miny, maxx, maxy = stack.bounds
    in_bounds = (data.x >= minx) & (data.x <= maxx) & (data.y >= miny) & (data.y <= maxy)
    _drop(in_bounds, "out_of_extent")

    if keep.any():
        vals = extract_values(stack, data.x[keep], data.y[keep])
        finite = np.all(np.isfinite(vals), axis=1)
        finite_full = np.ones(n, dtype=bool)
        finite_full[np.where(keep)[0]] = finite
        _drop(finite_full, "nodata_cell")

    cleaned = OccurrenceData(
        x=data.x[keep],
        y=data.y[keep],
        presence=data.presence[keep],
        crs=data.crs,
    )
    report = CleaningReport(
        n_input=int(n), n_output=int(keep.sum()), dropped=dropped, kept_mask=keep
    )
    return cleaned, report
