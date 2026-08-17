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


# Absolute tolerance for comparing the six affine-transform coefficients of
# two rasters. Deliberately a single shared constant: the *same* comparison
# decides both whether load_stack() accepts a set of rasters and whether the
# UI calls them misaligned, so the two can never disagree.
GRID_TOL = 1e-6


@dataclass
class RasterProfile:
    """Grid metadata for a single raster file, read without touching any
    pixel data. This is what the alignment check compares, and what the
    wizard shows per layer when a set of rasters doesn't line up.
    """

    name: str
    path: str
    crs: str
    dtype: str
    width: int
    height: int
    transform: Affine
    nodata: float | None

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


def describe_profiles(paths: list[str | Path]) -> list[RasterProfile]:
    """Read every raster's grid metadata (header only — no pixels), in the
    order given. Cheap enough to run before deciding whether a set of
    rasters can be stacked at all."""
    profiles: list[RasterProfile] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Raster not found: {p}")
        with rasterio.open(p) as src:
            profiles.append(
                RasterProfile(
                    name=p.stem,
                    path=str(p),
                    crs=src.crs.to_string() if src.crs else "",
                    dtype=src.dtypes[0],
                    width=src.width,
                    height=src.height,
                    transform=src.transform,
                    nodata=src.nodata,
                )
            )
    return profiles


def check_unique_stems(paths: list[str | Path]) -> None:
    """Predictors are matched by filename stem downstream (VIF selection,
    prediction columns), so two rasters may never share one."""
    seen: dict[str, Path] = {}
    for raw in paths:
        p = Path(raw)
        if p.stem in seen:
            raise RasterAlignmentError(
                f"Duplicate predictor name {p.stem!r}: {seen[p.stem]} and {p} "
                "share the same filename stem. Predictors are matched by name "
                "(VIF selection, prediction columns), so rasters must have unique "
                "stems. Rename one of these files."
            )
        seen[p.stem] = p


@dataclass
class AlignmentIssues:
    """Which grid properties differ across a set of rasters.

    `crs`, `resolution`, `size`, `origin` and `rotation` are the primitive
    comparisons that decide whether the rasters can be stacked at all (see
    `any`). `extent` and `fractional_offset` are derived from them and exist
    only to describe the problem the way a user thinks about it — "the layers
    cover different areas", "the layers don't share a pixel grid".
    """

    crs: bool = False
    resolution: bool = False
    size: bool = False
    origin: bool = False
    rotation: bool = False
    extent: bool = False
    fractional_offset: bool = False

    @property
    def any(self) -> bool:
        """True if the rasters cannot be read as one stack. Covers exactly
        the properties a RasterStack collapses to a single value: one CRS,
        one width/height, one affine transform."""
        return self.crs or self.resolution or self.size or self.origin or self.rotation

    @property
    def labels(self) -> list[str]:
        """Short user-facing names of what's wrong, for a headline like
        'These rasters differ in CRS and resolution.'"""
        out: list[str] = []
        if self.crs:
            out.append("CRS")
        if self.resolution:
            out.append("resolution")
        if self.extent or self.size or self.origin:
            out.append("extent")
        if self.fractional_offset or self.rotation:
            out.append("pixel alignment")
        if not out and self.any:
            # Defensive: never claim a problem without naming one.
            out.append("grid transform")
        return out


def _close(a: float, b: float, tol: float = GRID_TOL) -> bool:
    return abs(a - b) < tol


def diagnose_alignment(profiles: list[RasterProfile]) -> AlignmentIssues:
    """Compare every raster against the first one and report which grid
    properties differ. Unlike raising on the first offender, this looks at
    the whole set so the wizard can show one complete table."""
    issues = AlignmentIssues()
    if len(profiles) < 2:
        return issues
    ref = profiles[0]
    ref_res_x, ref_res_y = ref.resolution
    for other in profiles[1:]:
        if ref.crs != other.crs:
            issues.crs = True
        if (ref.width, ref.height) != (other.width, other.height):
            issues.size = True
        if not (
            _close(ref.transform.a, other.transform.a)
            and _close(ref.transform.e, other.transform.e)
        ):
            issues.resolution = True
        if not (
            _close(ref.transform.c, other.transform.c)
            and _close(ref.transform.f, other.transform.f)
        ):
            issues.origin = True
        if not (
            _close(ref.transform.b, other.transform.b)
            and _close(ref.transform.d, other.transform.d)
        ):
            issues.rotation = True
        if any(not _close(x, y) for x, y in zip(ref.bounds, other.bounds)):
            issues.extent = True
        # A whole-pixel shift is only an extent difference — the two rasters
        # still share a grid and could be stacked by clipping. A fractional
        # shift means the pixel centres genuinely don't line up, which can
        # only be fixed by resampling.
        for ref_o, other_o, res in (
            (ref.transform.c, other.transform.c, ref_res_x),
            (ref.transform.f, other.transform.f, ref_res_y),
        ):
            if res <= 0:
                continue
            steps = (other_o - ref_o) / res
            if abs(steps - round(steps)) > 1e-6:
                issues.fractional_offset = True
    return issues


def _fmt_bounds(bounds: tuple[float, float, float, float]) -> str:
    minx, miny, maxx, maxy = bounds
    return f"x {minx:.6g}…{maxx:.6g}, y {miny:.6g}…{maxy:.6g}"


def format_alignment_error(
    profiles: list[RasterProfile], issues: AlignmentIssues
) -> str:
    """A per-layer breakdown of a failed alignment check, used both as the
    RasterAlignmentError message and as the headline above the wizard's
    layer-properties table."""
    lines = [
        "Rasters must share CRS, extent, resolution, and pixel grid. "
        f"These differ in: {', '.join(issues.labels)}."
    ]
    for i, p in enumerate(profiles):
        res_x, res_y = p.resolution
        lines.append(
            f"  {p.name}{' (reference)' if i == 0 else ''}: "
            f"{p.dtype}, CRS {p.crs or 'none'}, "
            f"{p.width}×{p.height} px, res {res_x:.6g}×{res_y:.6g}, "
            f"{_fmt_bounds(p.bounds)}"
        )
    return "\n".join(lines)


def build_stack(profiles: list[RasterProfile]) -> RasterStack:
    """Assemble a RasterStack from already-read profiles, taking the shared
    grid from the first one. Callers must have run `diagnose_alignment`
    first — this does no checking of its own."""
    ref = profiles[0]
    return RasterStack(
        names=[p.name for p in profiles],
        paths=[p.path for p in profiles],
        crs=ref.crs,
        transform=ref.transform,
        width=ref.width,
        height=ref.height,
        nodata=ref.nodata if ref.nodata is not None else np.nan,
        shape=(ref.height, ref.width),
    )


def load_stack(paths: list[str | Path]) -> RasterStack:
    if not paths:
        raise ValueError("Empty raster paths list.")
    check_unique_stems(paths)
    profiles = describe_profiles(paths)
    issues = diagnose_alignment(profiles)
    if issues.any:
        raise RasterAlignmentError(format_alignment_error(profiles, issues))
    return build_stack(profiles)


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
    for row_off in range(0, stack.height, block_rows):
        h = min(block_rows, stack.height - row_off)
        arr = np.empty((len(stack.paths), h, stack.width), dtype=np.float32)
        for i, path in enumerate(stack.paths):
            with rasterio.open(path) as src:
                band = src.read(1, window=((row_off, row_off + h), (0, stack.width)))
                # Each band's own nodata only — matching extract_values().
                # Falling back to another band's nodata value (e.g. the
                # first raster's) would treat a genuine data value in a band
                # that defines no nodata of its own as missing, whenever it
                # happens to equal some other band's sentinel.
                nd = src.nodata
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
