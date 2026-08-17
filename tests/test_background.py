from __future__ import annotations

import numpy as np
import pytest

from sdm_plugin.core.background.disk import sample_disk
from sdm_plugin.core.background.sre import presence_envelope, sample_sre
from sdm_plugin.core.config import SDMConfig
from sdm_plugin.core.io.rasters import extract_values, load_stack


@pytest.fixture
def stack(tiny_stack):
    return load_stack(tiny_stack)


@pytest.fixture
def presences():
    """A tight cluster near the middle of the 40×40 tiny_stack extent, so
    there is room for a ring around it either way."""
    rng = np.random.default_rng(0)
    return rng.uniform(18.0, 22.0, size=30), rng.uniform(18.0, 22.0, size=30)


def _nearest_distances(x, y, px, py):
    return np.min(np.hypot(x[:, None] - px[None, :], y[:, None] - py[None, :]), axis=1)


# ----- disk (min/max distance ring) -----


def test_disk_respects_both_radii(stack, presences):
    px, py = presences
    bx, by = sample_disk(stack, px, py, 400, 5.0, 12.0, rng=np.random.default_rng(1))

    assert len(bx) > 0
    nearest = _nearest_distances(bx, by, px, py)
    assert nearest.min() >= 5.0 - 1e-9, "points inside the hole must be rejected"
    assert nearest.max() <= 12.0 + 1e-9, "points beyond the outer radius must be rejected"


def test_disk_measures_from_the_nearest_presence_not_any(stack):
    """Two presences far apart: a point close to one of them is close to *the
    presences*, and an inner radius has to reject it even though it is far
    from the other one."""
    px = np.array([10.0, 30.0])
    py = np.array([20.0, 20.0])
    bx, by = sample_disk(stack, px, py, 300, 6.0, 0.0, rng=np.random.default_rng(2))

    assert _nearest_distances(bx, by, px, py).min() >= 6.0 - 1e-9


def test_disk_max_of_zero_means_no_upper_limit(stack, presences):
    px, py = presences
    bx, by = sample_disk(stack, px, py, 500, 0.0, 0.0, rng=np.random.default_rng(3))

    nearest = _nearest_distances(bx, by, px, py)
    # The cluster sits mid-extent, so an unlimited ring must reach points
    # further out than any bounded one would.
    assert nearest.max() > 12.0


def test_disk_with_only_a_minimum_still_excludes_the_hole(stack, presences):
    px, py = presences
    bx, by = sample_disk(stack, px, py, 300, 8.0, 0.0, rng=np.random.default_rng(4))

    assert _nearest_distances(bx, by, px, py).min() >= 8.0 - 1e-9


def test_disk_rejects_a_minimum_that_is_not_below_the_maximum(stack, presences):
    px, py = presences
    with pytest.raises(ValueError, match="smaller than max_distance"):
        sample_disk(stack, px, py, 100, 10.0, 10.0, rng=np.random.default_rng(5))


def test_disk_explains_an_empty_ring(stack, presences):
    """A minimum radius past the far corner of the rasters leaves nowhere to
    sample, and saying so beats returning zero points silently."""
    px, py = presences
    with pytest.raises(ValueError, match="minimum and maximum distance"):
        sample_disk(stack, px, py, 100, 500.0, 0.0, rng=np.random.default_rng(6))


# ----- SRE (rectilinear environmental envelope) -----


def test_envelope_brackets_the_presence_conditions(stack, presences):
    px, py = presences
    lower, upper = presence_envelope(stack, px, py, quantile=0.0)

    values = extract_values(stack, px, py)
    values = values[np.all(np.isfinite(values), axis=1)]
    # quantile=0 is the outright min/max, so every record sits inside.
    assert np.all(values >= lower - 1e-9) and np.all(values <= upper + 1e-9)
    assert lower.shape == (len(stack.names),)


def test_envelope_quantile_trims_the_extremes(stack, presences):
    """Trimming has to shrink the envelope — that is the whole point of the
    quantile, so one atypical record cannot stretch it over everything."""
    px, py = presences
    wide_lower, wide_upper = presence_envelope(stack, px, py, quantile=0.0)
    trim_lower, trim_upper = presence_envelope(stack, px, py, quantile=0.1)

    assert np.all(trim_lower >= wide_lower - 1e-9)
    assert np.all(trim_upper <= wide_upper + 1e-9)
    assert np.any(trim_upper - trim_lower < wide_upper - wide_lower)


def test_envelope_rejects_an_out_of_range_quantile(stack, presences):
    px, py = presences
    with pytest.raises(ValueError, match="below 0.5"):
        presence_envelope(stack, px, py, quantile=0.5)


def test_sre_points_fall_outside_the_envelope(stack, presences):
    px, py = presences
    bx, by = sample_sre(stack, px, py, 200, quantile=0.025, rng=np.random.default_rng(7))

    assert len(bx) > 0
    lower, upper = presence_envelope(stack, px, py, quantile=0.025)
    values = extract_values(stack, bx, by)
    assert np.all(np.isfinite(values)), "nodata cells must never be drawn"
    outside = np.any((values < lower) | (values > upper), axis=1)
    assert outside.all(), (
        "every pseudo-absence must be beyond the envelope on at least one predictor"
    )


def test_sre_needs_more_than_one_usable_presence(stack):
    with pytest.raises(ValueError, match="at least 2 presence points"):
        sample_sre(
            stack, np.array([20.0]), np.array([20.0]), 10,
            rng=np.random.default_rng(8),
        )


def test_sre_says_so_when_the_presences_span_everything(stack):
    """A presence on every single cell leaves nothing outside the envelope.
    That is the degenerate end of SRE's assumption failing, and it has to be
    reported rather than silently returning no pseudo-absences."""
    height, width = stack.shape
    minx, miny, _maxx, maxy = stack.bounds
    res_x, res_y = stack.resolution
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    px = (minx + (cols.ravel() + 0.5) * res_x)
    py = (maxy - (rows.ravel() + 0.5) * res_y)
    assert py.min() > miny

    with pytest.raises(ValueError, match="outside the presences"):
        sample_sre(stack, px, py, 50, quantile=0.0, rng=np.random.default_rng(9))


# ----- how many points to draw -----


def test_ratio_method_scales_with_the_presence_count():
    cfg = SDMConfig().background
    cfg.method = "ratio"
    cfg.ratio = 4.0
    assert cfg.resolve_count(50) == 200
    assert cfg.resolve_count(300) == 1200
    # Fractional multipliers round rather than truncate, and never to nothing.
    cfg.ratio = 1.5
    assert cfg.resolve_count(35) == 53
    cfg.ratio = 0.1
    assert cfg.resolve_count(3) == 1


def test_every_other_method_ignores_the_presence_count():
    cfg = SDMConfig().background
    assert cfg.method == "random", "random is the default"
    assert cfg.resolve_count(50) == cfg.count == 10_000
    assert cfg.resolve_count(5000) == 10_000


def test_ratio_method_draws_that_many_points(tiny_stack, po_csv):
    """End to end through the stage, not just the arithmetic: the number of
    background points that come back must match the multiplier."""
    from sdm_plugin.core.io.occurrences import load_occurrences
    from sdm_plugin.core.stages import collect_labeled_points_and_extract

    cfg = SDMConfig()
    cfg.data_mode = "presence_only"
    cfg.background.method = "ratio"
    cfg.background.ratio = 3.0
    occ = load_occurrences(po_csv, crs="EPSG:32633")
    n_presence = len(occ.x)

    _px, _py, presence_flag, _X, _names = collect_labeled_points_and_extract(
        cfg, occ, load_stack(tiny_stack), np.random.default_rng(1)
    )
    assert int((presence_flag == 1).sum()) == n_presence
    assert int((presence_flag == 0).sum()) == 3 * n_presence


def test_the_minimum_count_rule_applies_only_to_a_fixed_count():
    cfg = SDMConfig()
    cfg.output.directory = "out"
    cfg.occurrence.path = "occ.csv"
    cfg.rasters.paths = ["a.tif"]

    cfg.background.count = 20  # below the 100-point floor
    assert any("at least 100" in e for e in cfg.validate())

    # A ratio has no floor of its own; 4 per presence is fine for any dataset.
    cfg.background.method = "ratio"
    assert not any("at least 100" in e for e in cfg.validate())

    cfg.background.ratio = 0.0
    assert any("greater than zero" in e for e in cfg.validate())


# ----- config plumbing -----


def test_config_validates_the_new_background_settings():
    cfg = SDMConfig()
    cfg.output.directory = "out"
    cfg.occurrence.path = "occ.csv"
    cfg.rasters.paths = ["a.tif"]

    cfg.background.method = "disk"
    cfg.background.min_distance = 20_000.0
    cfg.background.max_distance = 10_000.0
    assert any("Minimum distance must be smaller" in e for e in cfg.validate())

    cfg.background.max_distance = 0.0  # no upper limit is allowed
    assert not any("distance" in e for e in cfg.validate())

    cfg.background.method = "sre"
    cfg.background.sre_quantile = 0.6
    assert any("SRE quantile" in e for e in cfg.validate())


def test_ratio_places_points_exactly_as_random_does(tiny_stack, po_csv):
    """The ratio method differs from random only in how many points it asks
    for, so at a matching count the two must produce the same placement."""
    from sdm_plugin.core.io.occurrences import load_occurrences
    from sdm_plugin.core.stages import collect_labeled_points_and_extract

    occ = load_occurrences(po_csv, crs="EPSG:32633")
    stack = load_stack(tiny_stack)
    n_presence = len(occ.x)

    by_ratio = SDMConfig()
    by_ratio.data_mode = "presence_only"
    by_ratio.background.method = "ratio"
    by_ratio.background.ratio = 3.0

    fixed = SDMConfig()
    fixed.data_mode = "presence_only"
    fixed.background.method = "random"
    fixed.background.count = 3 * n_presence

    rx, ry, *_ = collect_labeled_points_and_extract(
        by_ratio, occ, stack, np.random.default_rng(7)
    )
    fx, fy, *_ = collect_labeled_points_and_extract(
        fixed, occ, stack, np.random.default_rng(7)
    )
    np.testing.assert_array_equal(rx, fx)
    np.testing.assert_array_equal(ry, fy)


def test_configs_from_the_count_mode_iteration_still_load():
    """Scaling by the presence count was briefly a separate `count_mode`
    flag before becoming a method in its own right."""
    cfg = SDMConfig.from_dict(
        {"background": {"count_mode": "ratio", "ratio": 6.0, "method": "random"}}
    )
    assert cfg.background.method == "ratio"
    assert cfg.background.ratio == 6.0
    assert cfg.background.resolve_count(20) == 120


def test_old_run_configs_still_load(tmp_path):
    """run_config.json from before the disk rework used a single
    `buffer_distance` with the method named "buffered". That was an outer
    radius with no inner one, so it maps straight onto max_distance and an
    old config reruns unchanged."""
    legacy = {
        "background": {
            "count": 5000,
            "method": "buffered",
            "buffer_distance": 25_000.0,
        }
    }
    cfg = SDMConfig.from_dict(legacy)

    assert cfg.background.method == "disk"
    assert cfg.background.max_distance == 25_000.0
    assert cfg.background.min_distance == 0.0
    assert cfg.background.count == 5000


def test_current_configs_round_trip(tmp_path):
    cfg = SDMConfig()
    cfg.background.method = "disk"
    cfg.background.min_distance = 1_000.0
    cfg.background.max_distance = 30_000.0
    path = tmp_path / "run_config.json"
    cfg.to_json(path)

    loaded = SDMConfig.from_json(path)
    assert loaded.background.method == "disk"
    assert loaded.background.min_distance == 1_000.0
    assert loaded.background.max_distance == 30_000.0
    assert loaded.background.sre_quantile == cfg.background.sre_quantile
