from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CSV_EXTS = {".csv", ".txt", ".tsv"}
VECTOR_EXTS = {".shp", ".geojson", ".json", ".gpkg", ".kml"}


@dataclass
class OccurrenceData:
    x: np.ndarray  # (n,) longitude / easting
    y: np.ndarray  # (n,) latitude / northing
    presence: np.ndarray  # (n,) uint8 — 1 for presence, 0 for absence
    crs: str


def load_occurrences(
    path: str | Path,
    *,
    x_field: str = "x",
    y_field: str = "y",
    presence_field: str = "",
    crs: str = "EPSG:4326",
    layer_name: str = "",
) -> OccurrenceData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Occurrence file not found: {path}")
    ext = path.suffix.lower()
    if ext in CSV_EXTS:
        df = _load_csv(path, x_field, y_field, presence_field)
        used_crs = crs
    elif ext in VECTOR_EXTS:
        df, used_crs = _load_vector(path, x_field, y_field, presence_field, layer_name, crs)
    else:
        raise ValueError(f"Unsupported occurrence file extension: {ext}")

    x = df["_x"].to_numpy(dtype=np.float64)
    y = df["_y"].to_numpy(dtype=np.float64)
    presence = df["_p"].to_numpy(dtype=np.uint8)
    return OccurrenceData(x=x, y=y, presence=presence, crs=used_crs)


def reproject_occurrences(data: OccurrenceData, target_crs: str) -> OccurrenceData:
    """Reproject occurrence coordinates into `target_crs` (a predictor raster
    stack's CRS). A no-op (returns `data` unchanged) if already in that CRS —
    comparing the CRS *string* rather than resolving+comparing both, since an
    exact string match means there is nothing to do, and callers that already
    reprojected once (setting .crs to target_crs) shouldn't pay for or repeat
    the transform on every subsequent call.
    """
    if data.crs == target_crs:
        return data
    from rasterio.warp import transform as warp_transform

    xs, ys = warp_transform(data.crs, target_crs, data.x.tolist(), data.y.tolist())
    return OccurrenceData(
        x=np.asarray(xs, dtype=np.float64),
        y=np.asarray(ys, dtype=np.float64),
        presence=data.presence,
        crs=target_crs,
    )


def _load_csv(
    path: Path, x_field: str, y_field: str, presence_field: str
) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    if x_field not in df.columns or y_field not in df.columns:
        raise ValueError(
            f"CSV missing coordinate columns '{x_field}' / '{y_field}'. "
            f"Available: {list(df.columns)}"
        )
    out = pd.DataFrame({"_x": df[x_field], "_y": df[y_field]})
    if presence_field:
        if presence_field not in df.columns:
            raise ValueError(f"Presence column '{presence_field}' not found in CSV.")
        out["_p"] = df[presence_field].astype(int).clip(0, 1)
    else:
        out["_p"] = 1
    return out


def _load_vector(
    path: Path,
    x_field: str,
    y_field: str,
    presence_field: str,
    layer_name: str,
    fallback_crs: str,
) -> tuple[pd.DataFrame, str]:
    import fiona
    from fiona.crs import to_string

    open_kwargs = {"layer": layer_name} if layer_name else {}
    with fiona.open(path, **open_kwargs) as src:
        crs = to_string(src.crs) if src.crs else fallback_crs
        xs: list[float] = []
        ys: list[float] = []
        pres: list[int] = []
        for feat in src:
            geom = feat["geometry"]
            if geom is None:
                continue
            gtype = geom["type"]
            coords = geom["coordinates"]
            if gtype == "Point":
                xs.append(float(coords[0]))
                ys.append(float(coords[1]))
            elif gtype == "MultiPoint" and coords:
                xs.append(float(coords[0][0]))
                ys.append(float(coords[0][1]))
            else:
                continue
            if presence_field:
                val = feat["properties"].get(presence_field, 0)
                pres.append(int(val) if val is not None else 0)
            else:
                pres.append(1)
    if not xs:
        raise ValueError(f"No point features found in {path}")
    df = pd.DataFrame({"_x": xs, "_y": ys, "_p": pres})
    return df, crs
