"""Resample a set of mismatched rasters onto one common grid.

The wizard refuses to stack predictors that don't share a CRS, extent,
resolution and pixel grid (see `core.io.rasters.diagnose_alignment`). This
module is the repair path behind the wizard's "Fix predictor layers" button:
pick one target grid, warp every input onto it, and hand back the new file
paths.

Qt-free, like the rest of core/ — the dialog that collects the user's choices
lives in ui/widgets/fix_rasters_dialog.py.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

from .rasters import RasterProfile

ExtentMode = Literal["intersection", "union", "reference", "custom"]
ResolutionMode = Literal["finest", "coarsest", "reference", "custom"]

# Resampling methods offered in the fix dialog, in the order shown there.
# Deliberately a curated subset of rasterio's enum: the ones an SDM user has
# a reason to pick, described in terms of the data rather than the algorithm.
RESAMPLING_METHODS: tuple[str, ...] = (
    "nearest",
    "bilinear",
    "cubic",
    "average",
    "mode",
    "median",
    "min",
    "max",
)

RESAMPLING_HELP: dict[str, str] = {
    "nearest": "nearest neighbour: keeps original values (use for categories)",
    "bilinear": "bilinear: smooth interpolation (use for continuous data)",
    "cubic": "cubic: smoother interpolation of continuous data",
    "average": "average of the contributing source pixels",
    "mode": "most common value of the contributing source pixels",
    "median": "median of the contributing source pixels",
    "min": "minimum of the contributing source pixels",
    "max": "maximum of the contributing source pixels",
}

_RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "median": Resampling.med,
    "min": Resampling.min,
    "max": Resampling.max,
}


# Output data types offered per layer. Every one of these is writable by the
# GTiff driver and readable by the rest of the pipeline; a layer whose own
# type isn't listed (int8, int64, ...) gets it added so "keep what it already
# is" is always an option — see `dtype_choices`.
DTYPE_CHOICES: tuple[str, ...] = (
    "uint8", "uint16", "int16", "uint32", "int32", "float32", "float64",
)

# What every layer starts on in the fix dialog. Deliberately uniform rather
# than inferred per layer: predictors in a species distribution model are
# overwhelmingly continuous measurements, for which float32 stores fractional
# values without rounding, -9999 is the conventional missing-data sentinel,
# and bilinear interpolates smoothly. A categorical layer (land cover, soil
# class) is the exception and needs its row switched to an integer type and
# nearest-neighbour resampling by hand — hence the guidance in the dialog.
DEFAULT_DTYPE = "float32"
DEFAULT_NODATA = -9999.0
DEFAULT_RESAMPLING = "bilinear"


class AlignmentTargetError(ValueError):
    """The requested target grid can't be built (layers don't overlap, a
    layer has no CRS to reproject from, a nonsensical resolution, ...)."""


def _is_integer_dtype(dtype: str) -> bool:
    return dtype.lower().startswith(("int", "uint", "bool", "byte"))


def _crs_of(profile: RasterProfile, assumed_crs: str) -> str:
    crs = profile.crs or assumed_crs
    if not crs:
        raise AlignmentTargetError(
            f"{profile.name} has no CRS of its own, so there is no way to know "
            "where on Earth it sits. Choose a CRS to assume for layers without "
            "one, or set the layer's CRS outside the plugin first."
        )
    return crs


def _same_crs(a: str, b: str) -> bool:
    if a == b:
        return True
    if not a or not b:
        return False
    try:
        return CRS.from_string(a) == CRS.from_string(b)
    except Exception:
        return False


def bounds_in_crs(
    profile: RasterProfile, dst_crs: str, assumed_crs: str = ""
) -> tuple[float, float, float, float]:
    """This layer's footprint expressed in `dst_crs`."""
    src_crs = _crs_of(profile, assumed_crs)
    if _same_crs(src_crs, dst_crs):
        return profile.bounds
    # densify_pts follows the projected edges rather than just the corners —
    # for a curved projection the corners alone under-report the footprint.
    return tuple(transform_bounds(src_crs, dst_crs, *profile.bounds, densify_pts=21))


def resolution_in_crs(
    profile: RasterProfile, dst_crs: str, assumed_crs: str = ""
) -> tuple[float, float]:
    """This layer's pixel size expressed in `dst_crs` units — the resolution
    a straight reprojection of it would naturally land on, so that "finest"
    and "coarsest" stay meaningful when the target CRS changes the units
    (degrees to metres, say)."""
    src_crs = _crs_of(profile, assumed_crs)
    if _same_crs(src_crs, dst_crs):
        return profile.resolution
    transform, _w, _h = calculate_default_transform(
        src_crs, dst_crs, profile.width, profile.height, *profile.bounds
    )
    return (abs(transform.a), abs(transform.e))


@dataclass
class AlignTarget:
    """The single grid every layer will be resampled onto."""

    crs: str
    bounds: tuple[float, float, float, float]  # requested, in `crs` units
    res_x: float
    res_y: float

    def __post_init__(self) -> None:
        if self.res_x <= 0 or self.res_y <= 0:
            raise AlignmentTargetError("Resolution must be greater than zero.")
        minx, miny, maxx, maxy = self.bounds
        if maxx <= minx or maxy <= miny:
            raise AlignmentTargetError(
                "The target extent is empty. If you chose Intersection, the "
                "layers may not overlap at all. Check the CRS of each one, or "
                "switch to Union."
            )

    @property
    def width(self) -> int:
        minx, _miny, maxx, _maxy = self.bounds
        return _ceil_pixels(maxx - minx, self.res_x)

    @property
    def height(self) -> int:
        _minx, miny, _maxx, maxy = self.bounds
        return _ceil_pixels(maxy - miny, self.res_y)

    @property
    def transform(self) -> Affine:
        minx, _miny, _maxx, maxy = self.bounds
        return from_origin(minx, maxy, self.res_x, self.res_y)

    @property
    def realized_bounds(self) -> tuple[float, float, float, float]:
        """The extent actually covered once the requested one is snapped up
        to a whole number of pixels."""
        minx, _miny, _maxx, maxy = self.bounds
        return (minx, maxy - self.height * self.res_y, minx + self.width * self.res_x, maxy)

    @property
    def cells(self) -> int:
        return self.width * self.height


def _ceil_pixels(span: float, res: float) -> int:
    # The 1e-9 slack keeps an extent that is already an exact pixel multiple
    # from gaining a spurious extra row/column to floating-point noise.
    return max(1, int(np.ceil(span / res - 1e-9)))


def build_target(
    profiles: list[RasterProfile],
    *,
    crs: str,
    extent_mode: ExtentMode = "intersection",
    resolution_mode: ResolutionMode = "coarsest",
    reference: int = 0,
    assumed_crs: str = "",
    custom_bounds: tuple[float, float, float, float] | None = None,
    custom_resolution: tuple[float, float] | None = None,
) -> AlignTarget:
    """Turn the user's choices into a concrete target grid.

    `reference` indexes into `profiles` and backs the "same as reference
    layer" options. Extents and resolutions of every layer are first
    expressed in `crs`, so the modes below stay meaningful even when the
    target CRS differs from the layers' own.
    """
    if not profiles:
        raise AlignmentTargetError("No rasters to align.")
    if not crs:
        raise AlignmentTargetError("Choose a target CRS.")
    ref = profiles[max(0, min(reference, len(profiles) - 1))]

    all_bounds = [bounds_in_crs(p, crs, assumed_crs) for p in profiles]
    if extent_mode == "custom":
        if custom_bounds is None:
            raise AlignmentTargetError("No custom extent given.")
        bounds = tuple(float(v) for v in custom_bounds)
    elif extent_mode == "reference":
        bounds = bounds_in_crs(ref, crs, assumed_crs)
    elif extent_mode == "union":
        bounds = (
            min(b[0] for b in all_bounds),
            min(b[1] for b in all_bounds),
            max(b[2] for b in all_bounds),
            max(b[3] for b in all_bounds),
        )
    else:  # intersection
        bounds = (
            max(b[0] for b in all_bounds),
            max(b[1] for b in all_bounds),
            min(b[2] for b in all_bounds),
            min(b[3] for b in all_bounds),
        )

    if resolution_mode == "custom":
        if custom_resolution is None:
            raise AlignmentTargetError("No custom resolution given.")
        res_x, res_y = (float(v) for v in custom_resolution)
    elif resolution_mode == "reference":
        res_x, res_y = resolution_in_crs(ref, crs, assumed_crs)
    else:
        all_res = [resolution_in_crs(p, crs, assumed_crs) for p in profiles]
        pick = max if resolution_mode == "coarsest" else min
        res_x = pick(r[0] for r in all_res)
        res_y = pick(r[1] for r in all_res)

    return AlignTarget(crs=crs, bounds=bounds, res_x=res_x, res_y=res_y)


def dtype_choices(profile: RasterProfile) -> list[str]:
    """Output types offered for one layer: the standard set, with the layer's
    own type prepended if it isn't already there so keeping it is always
    possible."""
    choices = list(DTYPE_CHOICES)
    if profile.dtype not in choices:
        choices.insert(0, profile.dtype)
    return choices


def dtype_range(dtype: str) -> tuple[float, float] | None:
    """Smallest and largest value `dtype` can store, or None for float types
    (where the practical limits are wide enough not to be worth checking)."""
    if not _is_integer_dtype(dtype):
        return None
    info = np.iinfo(np.dtype(dtype))
    return (float(info.min), float(info.max))


def nodata_fits_dtype(nodata: float, dtype: str) -> bool:
    """Whether `nodata` can actually be stored in `dtype` — -9999 in a uint8
    band would be silently mangled rather than marking anything as missing."""
    limits = dtype_range(dtype)
    if limits is None:
        return True  # float bands accept any finite value, and NaN
    if not np.isfinite(nodata) or nodata != int(nodata):
        return False
    return limits[0] <= nodata <= limits[1]


def default_nodata(dtype: str) -> float:
    """The NoData value a layer starts with in the fix dialog: -9999 wherever
    the type can hold it, and the type's own maximum for unsigned types that
    cannot represent a negative sentinel."""
    if nodata_fits_dtype(DEFAULT_NODATA, dtype):
        return DEFAULT_NODATA
    limits = dtype_range(dtype)
    return limits[1] if limits is not None else float("nan")


@dataclass
class LayerOutput:
    """How one layer is resampled and written. Every field is a choice the
    user makes in the fix dialog; `default_output` seeds them."""

    resampling: str
    dtype: str
    nodata: float

    def validate(self, name: str) -> None:
        if self.resampling not in _RESAMPLING:
            raise AlignmentTargetError(f"Unknown resampling method: {self.resampling!r}")
        if not nodata_fits_dtype(self.nodata, self.dtype):
            limits = dtype_range(self.dtype)
            span = f" ({limits[0]:.0f}…{limits[1]:.0f})" if limits else ""
            raise AlignmentTargetError(
                f"{name}: NoData {self.nodata:g} cannot be stored as "
                f"{self.dtype}{span}. Choose another value or data type."
            )


def default_output(profile: RasterProfile) -> LayerOutput:
    """The settings a layer starts on: float32 / -9999 / bilinear for every
    one of them (see DEFAULT_DTYPE). `profile` is unused today and kept so
    callers need not care whether the defaults stay uniform.

    Note this ignores what the source declared. That is the point: NoData
    especially cannot be inherited, since warping onto a new grid always
    leaves cells no source pixel reaches and those need a sentinel even when
    the source had none.
    """
    return LayerOutput(
        resampling=DEFAULT_RESAMPLING, dtype=DEFAULT_DTYPE, nodata=DEFAULT_NODATA
    )


def align_rasters(
    profiles: list[RasterProfile],
    target: AlignTarget,
    out_dir: str | Path,
    *,
    outputs: dict[str, LayerOutput] | None = None,
    assumed_crs: str = "",
    progress: Callable[[float, str], None] | None = None,
) -> list[str]:
    """Warp every layer onto `target`, writing one GeoTIFF per layer into
    `out_dir`, and return the new paths in the original order.

    `outputs` maps a layer name to its resampling method, output data type
    and NoData value; layers not listed fall back to `default_output`.
    """
    outputs = outputs or {}
    # Resolve and check every layer's settings up front: a bad NoData on the
    # last layer must not leave the earlier ones already written to disk.
    settings = {p.name: outputs.get(p.name) or default_output(p) for p in profiles}
    for name, layer in settings.items():
        layer.validate(name)

    out_dir = Path(out_dir)
    sources = {str(Path(p.path).resolve()) for p in profiles}
    out_paths = [out_dir / f"{p.name}.tif" for p in profiles]
    clashes = [str(o) for o in out_paths if str(o.resolve()) in sources]
    if clashes:
        raise AlignmentTargetError(
            "The output folder would overwrite the rasters being read:\n  "
            + "\n  ".join(clashes)
            + "\nChoose a different output folder."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for i, (profile, out_path) in enumerate(zip(profiles, out_paths)):
        if progress:
            progress(i / len(profiles), profile.name)
        layer = settings[profile.name]
        method = layer.resampling
        dst_dtype, dst_nodata = layer.dtype, layer.nodata
        src_crs = _crs_of(profile, assumed_crs)

        creation = {
            "driver": "GTiff",
            "height": target.height,
            "width": target.width,
            "count": 1,
            "dtype": dst_dtype,
            "crs": target.crs,
            "transform": target.transform,
            "nodata": dst_nodata,
            "compress": "deflate",
            "BIGTIFF": "IF_SAFER",
        }
        if target.width >= 256 and target.height >= 256:
            creation.update(tiled=True, blockxsize=256, blockysize=256)

        with rasterio.open(profile.path) as src, rasterio.open(
            out_path, "w", **creation
        ) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=profile.transform,
                src_crs=src_crs,
                src_nodata=profile.nodata,
                dst_transform=target.transform,
                dst_crs=target.crs,
                dst_nodata=dst_nodata,
                resampling=_RESAMPLING[method],
            )
        written.append(str(out_path))
    if progress:
        progress(1.0, "")
    return written
