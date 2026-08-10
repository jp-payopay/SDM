from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


@dataclass
class RasterStack:
    names: list[str]
    paths: list[str]
    crs: str
    transform: Affine
    width: int
    height: int
    nodata: float
    shape: tuple[int, int]  # (height, width)

    @property
    def resolution(self) -> tuple[float, float]:
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left = self.transform.c
        top = self.transform.f
        right = left + self.transform.a * self.width
        bottom = top + self.transform.e * self.height
        return (min(left, right), min(top, bottom), max(left, right), max(top, bottom))


class RasterAlignmentError(ValueError):
    pass


def load_stack(paths: list[str | Path]) -> RasterStack:
    if not paths:
        raise ValueError("Empty raster paths list.")
    paths = [Path(p) for p in paths]
    ref = None
    ref_path = None
    names: list[str] = []
    seen_stems: dict[str, Path] = {}
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Raster not found: {p}")
        if p.stem in seen_stems:
            raise RasterAlignmentError(
                f"Duplicate predictor name {p.stem!r}: {seen_stems[p.stem]} and {p} "
                "share the same filename stem. Predictors are matched by name "
                "(VIF selection, prediction columns), so rasters must have unique "
                "stems — rename one of these files."
            )
        seen_stems[p.stem] = p
        with rasterio.open(p) as src:
            info = {
                "crs": src.crs.to_string() if src.crs else "",
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "nodata": src.nodata,
            }
        if ref is None:
            ref = info
            ref_path = p
        else:
            _check_aligned(ref, ref_path, info, p)
        names.append(p.stem)
    return RasterStack(
        names=names,
        paths=[str(p) for p in paths],
        crs=ref["crs"],
        transform=ref["transform"],
        width=ref["width"],
        height=ref["height"],
        nodata=ref["nodata"] if ref["nodata"] is not None else np.nan,
        shape=(ref["height"], ref["width"]),
    )


def _check_aligned(ref: dict, ref_path: Path, other: dict, other_path: Path) -> None:
    problems: list[str] = []
    if ref["crs"] != other["crs"]:
        problems.append(f"CRS mismatch: {ref['crs']!r} vs {other['crs']!r}")
    if (ref["width"], ref["height"]) != (other["width"], other["height"]):
        problems.append(
            f"Shape mismatch: {ref['width']}x{ref['height']} vs "
            f"{other['width']}x{other['height']}"
        )
    if not _transforms_equal(ref["transform"], other["transform"]):
        problems.append(
            f"Transform mismatch: {tuple(ref['transform'])} vs {tuple(other['transform'])}"
        )
    if problems:
        raise RasterAlignmentError(
            f"Rasters must share CRS, shape, and transform.\n"
            f"Reference: {ref_path}\n"
            f"Offender:  {other_path}\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


def _transforms_equal(a: Affine, b: Affine, tol: float = 1e-6) -> bool:
    return all(abs(x - y) < tol for x, y in zip(a[:6], b[:6]))


def extract_values(stack: RasterStack, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample raster stack at (x, y) coordinates → (n_points, n_bands) array."""
    n = len(x)
    out = np.full((n, len(stack.paths)), np.nan, dtype=np.float64)
    coords = list(zip(x, y))
    for i, path in enumerate(stack.paths):
        with rasterio.open(path) as src:
            samples = list(src.sample(coords, indexes=1))
            nodata = src.nodata
        vals = np.asarray([s[0] for s in samples], dtype=np.float64)
        if nodata is not None:
            vals = np.where(vals == nodata, np.nan, vals)
        out[:, i] = vals
    return out


def iter_windows(stack: RasterStack, block_rows: int = 256):
    """Yield (row_off, height, arr) — arr has shape (n_bands, height, width)."""
    with rasterio.open(stack.paths[0]) as ref:
        nodata_ref = ref.nodata
        for row_off in range(0, stack.height, block_rows):
            h = min(block_rows, stack.height - row_off)
            arr = np.empty((len(stack.paths), h, stack.width), dtype=np.float32)
            for i, path in enumerate(stack.paths):
                with rasterio.open(path) as src:
                    band = src.read(1, window=((row_off, row_off + h), (0, stack.width)))
                    nd = src.nodata if src.nodata is not None else nodata_ref
                    band = band.astype(np.float32)
                    if nd is not None:
                        band = np.where(band == nd, np.nan, band)
                    arr[i] = band
            yield row_off, h, arr


def raster_extent_polygon(stack: RasterStack) -> tuple[float, float, float, float]:
    return stack.bounds


@dataclass
class BandEDA:
    """A quick exploratory summary of a single predictor raster: its storage
    type and the spread of its (valid, non-nodata) pixel values. Statistics
    come from a decimated read, so on very large rasters they are close
    estimates rather than exact population values."""

    name: str
    dtype: str
    kind: str  # "integer" | "decimal" | raw dtype for anything unusual
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    nodata: float | None
    valid_fraction: float | None  # share of sampled pixels that were valid
    sampled: bool  # True if computed from a decimated read, not every pixel


def _dtype_kind(dtype: str) -> str:
    d = dtype.lower()
    if d.startswith(("int", "uint")):
        return "integer"
    if d.startswith(("float", "complex")):
        return "decimal"
    return dtype


def describe_band(path: str | Path, sample_cap: int = 1_000_000) -> BandEDA:
    """Read a single raster and summarize band 1. To stay fast on large
    rasters, the band is read at a decimated resolution capped near
    `sample_cap` pixels; nodata and non-finite values are excluded from the
    statistics.
    """
    with rasterio.open(path) as src:
        name = Path(path).stem
        dtype = src.dtypes[0]
        nodata = src.nodata
        width, height = src.width, src.height
        total = width * height
        scale = max(1, int(np.sqrt(total / sample_cap))) if total > sample_cap else 1
        out_w = max(1, width // scale)
        out_h = max(1, height // scale)
        arr = src.read(1, out_shape=(out_h, out_w), masked=True)

    values = np.asarray(arr.compressed(), dtype=np.float64)
    values = values[np.isfinite(values)]
    sampled_total = out_w * out_h
    if values.size:
        minimum = float(values.min())
        maximum = float(values.max())
        mean = float(values.mean())
        std = float(values.std())
    else:
        minimum = maximum = mean = std = None
    return BandEDA(
        name=name,
        dtype=dtype,
        kind=_dtype_kind(dtype),
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        nodata=(None if nodata is None else float(nodata)),
        valid_fraction=(values.size / sampled_total if sampled_total else None),
        sampled=(scale > 1),
    )


def describe_stack(stack: RasterStack, sample_cap: int = 1_000_000) -> list[BandEDA]:
    """Per-raster exploratory summaries for every file in a stack, in order."""
    return [describe_band(path, sample_cap=sample_cap) for path in stack.paths]
