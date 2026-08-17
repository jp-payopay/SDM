from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from sdm_plugin.core.io.align import (
    DEFAULT_DTYPE,
    DEFAULT_NODATA,
    DEFAULT_RESAMPLING,
    AlignmentTargetError,
    LayerOutput,
    align_rasters,
    build_target,
    default_nodata,
    default_output,
    dtype_choices,
    nodata_fits_dtype,
)
from sdm_plugin.core.io.rasters import (
    RasterAlignmentError,
    describe_profiles,
    diagnose_alignment,
    load_stack,
)

# A UTM zone 33N tile and the WGS84 patch that actually overlaps it — easting
# 500000 is the zone's central meridian (15°E) and northing 4000000 sits at
# ~36.1447°N, so a geographic raster has to be placed there to intersect.
UTM = "EPSG:32633"
WGS = "EPSG:4326"


def _write(path, arr, crs, transform, nodata=None, dtype="float32"):
    height, width = arr.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype=dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype), 1)
    return str(path)


@pytest.fixture
def mismatched(tmp_path):
    """Three rasters that differ in CRS, resolution, extent and pixel grid —
    every category the wizard reports on, in one set."""
    rng = np.random.default_rng(0)
    a = _write(
        tmp_path / "bio1.tif", rng.normal(size=(40, 40)),
        UTM, from_origin(500000, 4000040, 1, 1), nodata=-9999.0,
    )
    b = _write(  # coarser, and shifted half a pixel off a's grid
        tmp_path / "bio2.tif", rng.normal(size=(20, 20)),
        UTM, from_origin(500000.5, 4000040.0, 2, 2),
    )
    c = _write(  # different CRS, integer categories, no nodata
        tmp_path / "landcover.tif", rng.integers(1, 8, size=(50, 50)),
        WGS, from_origin(14.9995, 36.1500, 0.0002, 0.0002), dtype="int16",
    )
    return [a, b, c]


def test_aligned_rasters_report_no_issues(tiny_stack):
    issues = diagnose_alignment(describe_profiles(tiny_stack))
    assert not issues.any
    assert issues.labels == []


def test_diagnose_reports_every_mismatching_property(mismatched):
    issues = diagnose_alignment(describe_profiles(mismatched))
    assert issues.any
    assert issues.crs and issues.resolution and issues.extent
    assert issues.fractional_offset, "a half-pixel origin shift is a real misalignment"
    assert issues.labels == ["CRS", "resolution", "extent", "pixel alignment"]


def test_whole_pixel_shift_is_an_extent_difference_not_a_misalignment(tmp_path):
    """A raster offset by an exact number of pixels still shares the grid, so
    it must be reported as an extent difference — not as pixels that fail to
    line up. It still can't be stacked as-is."""
    arr = np.ones((10, 10), dtype="float32")
    a = _write(tmp_path / "a.tif", arr, UTM, from_origin(500000, 4000010, 1, 1))
    b = _write(tmp_path / "b.tif", arr, UTM, from_origin(500003, 4000010, 1, 1))

    issues = diagnose_alignment(describe_profiles([a, b]))
    assert issues.any and issues.extent
    assert not issues.fractional_offset
    assert not issues.crs and not issues.resolution
    assert issues.labels == ["extent"]


def test_describe_profiles_reads_grid_metadata(mismatched):
    profiles = describe_profiles(mismatched)
    names = [p.name for p in profiles]
    assert names == ["bio1", "bio2", "landcover"]
    assert profiles[0].crs == UTM and profiles[0].nodata == -9999.0
    assert profiles[1].nodata is None
    assert profiles[1].resolution == (2.0, 2.0)
    assert profiles[2].dtype == "int16" and profiles[2].crs == WGS


def test_load_stack_error_describes_every_layer(mismatched):
    with pytest.raises(RasterAlignmentError) as excinfo:
        load_stack(mismatched)
    message = str(excinfo.value)
    for name in ("bio1", "bio2", "landcover"):
        assert name in message
    assert "CRS, resolution, extent, pixel alignment" in message


def test_every_layer_starts_on_the_same_defaults(mismatched):
    """float32 / -9999 / bilinear for every layer regardless of what it is on
    disk — right for the continuous predictors that dominate an SDM stack.
    Note NoData is never inherited: warping onto a new grid always leaves
    cells no source pixel reaches, so an output needs a sentinel even when
    its source declared none (landcover, here, has none)."""
    defaults = {p.name: default_output(p) for p in describe_profiles(mismatched)}
    assert set(defaults) == {"bio1", "bio2", "landcover"}
    for layer in defaults.values():
        assert layer.dtype == DEFAULT_DTYPE == "float32"
        assert layer.nodata == DEFAULT_NODATA == -9999.0
        assert layer.resampling == DEFAULT_RESAMPLING == "bilinear"


def test_unset_layers_are_written_with_those_defaults(tmp_path, mismatched):
    profiles = describe_profiles(mismatched)
    target = build_target(profiles, crs=UTM, extent_mode="intersection")
    out = align_rasters(profiles, target, tmp_path / "aligned")  # no `outputs`

    for written in describe_profiles(out):
        assert written.dtype == "float32"
        assert written.nodata == -9999.0


def test_unsigned_types_fall_back_from_the_negative_default(tmp_path):
    """-9999 is the plugin's convention, but a uint band cannot hold it — the
    default has to be something that band can actually store."""
    assert default_nodata("int16") == DEFAULT_NODATA
    assert default_nodata("float32") == DEFAULT_NODATA
    assert default_nodata("uint8") == 255.0
    assert default_nodata("uint16") == 65535.0

    assert nodata_fits_dtype(-9999, "int16")
    assert not nodata_fits_dtype(-9999, "uint8")
    assert not nodata_fits_dtype(40000, "int16")  # in range for uint16, not int16
    assert not nodata_fits_dtype(1.5, "int16")  # integer bands hold no fractions
    assert nodata_fits_dtype(float("nan"), "float32")
    assert not nodata_fits_dtype(float("nan"), "int32")


def test_data_type_choices_always_include_the_layers_own(tmp_path):
    arr = np.ones((4, 4), dtype="int8")
    path = _write(tmp_path / "odd.tif", arr, UTM, from_origin(0, 4, 1, 1), dtype="int8")
    (profile,) = describe_profiles([path])
    choices = dtype_choices(profile)
    assert choices[0] == "int8", "keeping the current type must always be offered"
    assert "float32" in choices and "uint8" in choices


def test_a_nodata_the_chosen_type_cannot_hold_is_refused(tmp_path, mismatched):
    profiles = describe_profiles(mismatched)
    target = build_target(profiles, crs=UTM, extent_mode="intersection")
    outputs = {"landcover": LayerOutput("nearest", "uint8", -9999.0)}
    with pytest.raises(AlignmentTargetError, match="cannot be stored as uint8"):
        align_rasters(profiles, target, tmp_path / "aligned", outputs=outputs)
    assert not (tmp_path / "aligned").exists(), (
        "an unusable setting on any layer must be caught before anything is written"
    )


def test_output_type_and_nodata_can_be_overridden(tmp_path, mismatched):
    profiles = describe_profiles(mismatched)
    target = build_target(profiles, crs=UTM, extent_mode="union")
    outputs = {
        "bio1": LayerOutput("bilinear", "float64", -1.0),
        "landcover": LayerOutput("nearest", "int32", -1.0),
    }
    out = align_rasters(profiles, target, tmp_path / "aligned", outputs=outputs)

    written = {p.name: p for p in describe_profiles(out)}
    assert written["bio1"].dtype == "float64" and written["bio1"].nodata == -1.0
    assert written["landcover"].dtype == "int32" and written["landcover"].nodata == -1.0
    # Untouched layers still get the defaults.
    assert written["bio2"].dtype == "float32"
    assert written["bio2"].nodata == DEFAULT_NODATA


def test_extent_and_resolution_modes_relate_as_named(mismatched):
    profiles = describe_profiles(mismatched)
    inter = build_target(profiles, crs=UTM, extent_mode="intersection")
    union = build_target(profiles, crs=UTM, extent_mode="union")
    assert union.bounds[0] <= inter.bounds[0] and union.bounds[1] <= inter.bounds[1]
    assert union.bounds[2] >= inter.bounds[2] and union.bounds[3] >= inter.bounds[3]

    coarse = build_target(profiles, crs=UTM, resolution_mode="coarsest")
    fine = build_target(profiles, crs=UTM, resolution_mode="finest")
    assert coarse.res_x > fine.res_x and coarse.res_y > fine.res_y
    assert coarse.cells < fine.cells

    reference = build_target(profiles, crs=UTM, resolution_mode="reference", reference=1)
    assert reference.res_x == pytest.approx(2.0)


def test_non_overlapping_layers_cannot_intersect(tmp_path):
    arr = np.ones((10, 10), dtype="float32")
    a = _write(tmp_path / "a.tif", arr, UTM, from_origin(500000, 4000010, 1, 1))
    b = _write(tmp_path / "b.tif", arr, UTM, from_origin(900000, 4000010, 1, 1))
    profiles = describe_profiles([a, b])

    with pytest.raises(AlignmentTargetError):
        build_target(profiles, crs=UTM, extent_mode="intersection")
    # ...but their union is perfectly well defined.
    assert build_target(profiles, crs=UTM, extent_mode="union").width > 0


def test_align_rasters_produces_a_loadable_stack(tmp_path, mismatched):
    profiles = describe_profiles(mismatched)
    target = build_target(
        profiles, crs=UTM, extent_mode="intersection", resolution_mode="finest"
    )
    out = align_rasters(profiles, target, tmp_path / "aligned")

    assert [p.name for p in describe_profiles(out)] == ["bio1", "bio2", "landcover"]
    stack = load_stack(out)  # the whole point: it now stacks
    assert stack.width == target.width and stack.height == target.height
    assert stack.crs == UTM
    assert stack.resolution == pytest.approx((target.res_x, target.res_y))


def test_nearest_neighbour_keeps_category_codes_intact(tmp_path, mismatched):
    """The setting a categorical layer has to be switched to by hand, since
    the defaults assume continuous data: nearest-neighbour into an integer
    type must reproduce the source's codes exactly, with no interpolated
    in-between classes."""
    profiles = describe_profiles(mismatched)
    original = set(np.unique(rasterio.open(profiles[2].path).read(1)))
    target = build_target(profiles, crs=UTM, extent_mode="intersection")
    out = align_rasters(
        profiles, target, tmp_path / "aligned",
        outputs={"landcover": LayerOutput("nearest", "int16", -9999.0)},
    )

    with rasterio.open(out[2]) as src:
        values = src.read(1, masked=True).compressed()
    assert values.size, "the intersection should contain some land-cover pixels"
    assert set(np.unique(values)) <= {int(v) for v in original}


def test_align_rasters_reports_nodata_outside_a_layers_coverage(tmp_path):
    """A union target reaches past each layer's own footprint; those cells
    must come back as NoData rather than as a silent zero."""
    arr = np.ones((10, 10), dtype="float32")
    a = _write(tmp_path / "a.tif", arr, UTM, from_origin(500000, 4000010, 1, 1))
    b = _write(tmp_path / "b.tif", arr, UTM, from_origin(500020, 4000010, 1, 1))
    profiles = describe_profiles([a, b])
    target = build_target(profiles, crs=UTM, extent_mode="union")
    out = align_rasters(profiles, target, tmp_path / "aligned")

    with rasterio.open(out[0]) as src:
        masked = src.read(1, masked=True)
    assert masked.mask.any(), "cells beyond layer a must be masked, not filled with data"
    assert masked.compressed().min() == 1.0


def test_align_rasters_refuses_to_overwrite_its_own_sources(tmp_path, mismatched):
    profiles = describe_profiles(mismatched)
    target = build_target(profiles, crs=UTM, extent_mode="intersection")
    with pytest.raises(AlignmentTargetError, match="overwrite"):
        align_rasters(profiles, target, tmp_path)


def test_a_layer_without_a_crs_needs_one_assumed(tmp_path):
    arr = np.ones((10, 10), dtype="float32")
    placed = _write(tmp_path / "placed.tif", arr, UTM, from_origin(500000, 4000010, 1, 1))
    bare = _write(tmp_path / "bare.tif", arr, None, from_origin(500000, 4000010, 1, 1))
    profiles = describe_profiles([placed, bare])
    assert profiles[1].crs == ""

    with pytest.raises(AlignmentTargetError, match="no CRS"):
        build_target(profiles, crs=UTM, extent_mode="union")

    target = build_target(profiles, crs=UTM, extent_mode="union", assumed_crs=UTM)
    out = align_rasters(profiles, target, tmp_path / "aligned", assumed_crs=UTM)
    assert load_stack(out).crs == UTM


def test_align_rasters_reports_progress_per_layer(tmp_path, mismatched):
    profiles = describe_profiles(mismatched)
    target = build_target(profiles, crs=UTM, extent_mode="intersection")
    seen: list[tuple[float, str]] = []
    align_rasters(
        profiles, target, tmp_path / "aligned",
        progress=lambda fraction, name: seen.append((fraction, name)),
    )
    assert [name for _f, name in seen] == ["bio1", "bio2", "landcover", ""]
    assert seen[0][0] == 0.0 and seen[-1][0] == 1.0
