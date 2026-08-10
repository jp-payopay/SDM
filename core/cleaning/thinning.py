from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.occurrences import OccurrenceData
from ..io.rasters import RasterStack


@dataclass
class ThinningReport:
    n_input: int
    n_output: int
    n_removed: int

    def as_dict(self) -> dict:
        return {"n_input": self.n_input, "n_output": self.n_output, "n_removed": self.n_removed}


def thin_to_pixel(
    data: OccurrenceData,
    stack: RasterStack,
) -> tuple[OccurrenceData, ThinningReport]:
    """Keep at most one point per raster pixel (per presence class)."""
    inv = ~stack.transform
    cols, rows = inv * (data.x, data.y)
    cols = np.floor(cols).astype(np.int64)
    rows = np.floor(rows).astype(np.int64)
    presence = data.presence.astype(np.int64)
    # Stack rather than pack into a single int key: points outside the raster
    # extent (negative or >= width/height, e.g. when auto-clean is disabled)
    # would otherwise collide with unrelated in-bounds pixels under a packed key.
    keys = np.stack([rows, cols, presence], axis=1)

    _, first_idx = np.unique(keys, axis=0, return_index=True)
    first_idx.sort()
    n_in = len(data.x)
    kept = OccurrenceData(
        x=data.x[first_idx],
        y=data.y[first_idx],
        presence=data.presence[first_idx],
        crs=data.crs,
    )
    return kept, ThinningReport(
        n_input=n_in, n_output=len(first_idx), n_removed=n_in - len(first_idx)
    )
