from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from .rasters import RasterStack


def write_raster(
    path: str | Path,
    data: np.ndarray,
    stack: RasterStack,
    *,
    dtype: str = "float32",
    nodata: float = -9999.0,
) -> None:
    """Write a single-band raster matching the reference stack's grid."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data, dtype=dtype)
    if arr.shape != stack.shape:
        raise ValueError(
            f"Data shape {arr.shape} does not match stack shape {stack.shape}."
        )
    # Map NaN to nodata for floating rasters. Integer rasters (e.g. uint8 binary
    # maps) carry their nodata value in the array already, and np.isnan is not
    # defined for them.
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.where(np.isnan(arr), nodata, arr)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=stack.height,
        width=stack.width,
        count=1,
        dtype=dtype,
        crs=stack.crs,
        transform=stack.transform,
        nodata=nodata,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(arr, 1)


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def save_model(path: str | Path, model) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    """Load a previously joblib-dumped fitted model.

    joblib.load is effectively pickle: loading a file executes arbitrary
    Python objects' __reduce__/__setstate__ code. Only load model files this
    plugin (or another trusted run) produced — never a `.joblib` file
    obtained from an untrusted source.
    """
    import joblib

    return joblib.load(path)
