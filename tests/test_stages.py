from __future__ import annotations

import numpy as np

from sdm_plugin.core.config import SDMConfig
from sdm_plugin.core.io.occurrences import load_occurrences
from sdm_plugin.core.io.rasters import load_stack
from sdm_plugin.core.pipeline import Pipeline
from sdm_plugin.core.predictors.vif import stepwise_vif
from sdm_plugin.core.session import PipelineSession
from sdm_plugin.core.stages import (
    collect_labeled_points_and_extract,
    make_folds,
    stage_clean,
    stage_vif,
    validate_matching_bands,
)


def _po_config(tiny_stack, po_csv, tmp_path) -> SDMConfig:
    cfg = SDMConfig()
    cfg.data_mode = "presence_only"
    cfg.occurrence.path = po_csv
    cfg.occurrence.crs = "EPSG:32633"
    cfg.rasters.paths = tiny_stack
    cfg.background.count = 300
    cfg.background.method = "random"
    cfg.vif.cutoff = 10.0
    cfg.split.method = "kfold"
    cfg.split.k = 3
    cfg.modeling.algorithms = ["lr", "rf"]
    cfg.modeling.replicates = 2
    cfg.output.directory = str(tmp_path / "out")
    return cfg


def test_collect_labeled_points_pa(tiny_stack, pa_csv):
    cfg = SDMConfig()
    cfg.data_mode = "presence_absence"
    cfg.occurrence.presence_field = "presence"
    stack = load_stack(tiny_stack)
    occ = load_occurrences(pa_csv, presence_field="presence", crs="EPSG:32633")
    rng = np.random.default_rng(0)

    px, py, presence_flag, X_full, feature_names = collect_labeled_points_and_extract(
        cfg, occ, stack, rng
    )
    assert feature_names == stack.names
    assert X_full.shape[1] == len(stack.names)
    assert len(px) == len(py) == len(presence_flag) == X_full.shape[0]
    assert set(np.unique(presence_flag)).issubset({0, 1})


def test_collect_labeled_points_po_random_deterministic(tiny_stack, po_csv):
    cfg = SDMConfig()
    cfg.data_mode = "presence_only"
    cfg.background.count = 200
    cfg.background.method = "random"
    stack = load_stack(tiny_stack)
    occ = load_occurrences(po_csv, crs="EPSG:32633")

    px1, py1, pf1, X1, _ = collect_labeled_points_and_extract(
        cfg, occ, stack, np.random.default_rng(42)
    )
    px2, py2, pf2, X2, _ = collect_labeled_points_and_extract(
        cfg, occ, stack, np.random.default_rng(42)
    )
    np.testing.assert_array_equal(px1, px2)
    np.testing.assert_array_equal(py1, py2)
    np.testing.assert_array_equal(pf1, pf2)
    np.testing.assert_array_equal(X1, X2)
    assert (pf1 == 1).sum() > 0
    assert (pf1 == 0).sum() > 0


def test_collect_labeled_points_po_disk(tiny_stack, po_csv):
    cfg = SDMConfig()
    cfg.data_mode = "presence_only"
    cfg.background.count = 150
    cfg.background.method = "disk"
    cfg.background.min_distance = 2.0
    cfg.background.max_distance = 10.0
    stack = load_stack(tiny_stack)
    occ = load_occurrences(po_csv, crs="EPSG:32633")

    px, py, presence_flag, X_full, _ = collect_labeled_points_and_extract(
        cfg, occ, stack, np.random.default_rng(1)
    )
    assert (presence_flag == 0).sum() > 0
    # Every drawn absence must respect the inner radius: the stack is in a
    # projected CRS, so config metres pass through as CRS units unchanged.
    bx, by = px[presence_flag == 0], py[presence_flag == 0]
    nearest = np.min(
        np.hypot(bx[:, None] - occ.x[None, :], by[:, None] - occ.y[None, :]), axis=1
    )
    assert nearest.min() >= 2.0 - 1e-9
    assert nearest.max() <= 10.0 + 1e-9


def test_collect_labeled_points_po_sre(tiny_stack, po_csv):
    cfg = SDMConfig()
    cfg.data_mode = "presence_only"
    cfg.background.count = 100
    cfg.background.method = "sre"
    stack = load_stack(tiny_stack)
    occ = load_occurrences(po_csv, crs="EPSG:32633")

    px, py, presence_flag, X_full, _ = collect_labeled_points_and_extract(
        cfg, occ, stack, np.random.default_rng(1)
    )
    assert (presence_flag == 0).sum() > 0


def test_collect_labeled_points_matches_pipeline(tiny_stack, po_csv, tmp_path):
    """The standalone stage call must produce exactly what a from-scratch
    Pipeline.run() uses internally at the same point, for the same config
    and seed — otherwise preview-then-reuse would silently diverge."""
    cfg = _po_config(tiny_stack, po_csv, tmp_path)

    stack = load_stack(cfg.rasters.paths)
    occ_raw = load_occurrences(cfg.occurrence.path, crs=cfg.occurrence.crs)
    occ, _cleaning_rep, _thinning_rep = stage_clean(cfg, occ_raw, stack)
    rng = np.random.default_rng(cfg.random_seed)
    px, py, presence_flag, X_full, feature_names = collect_labeled_points_and_extract(
        cfg, occ, stack, rng
    )

    session = PipelineSession()
    Pipeline(cfg, session=session).run()

    np.testing.assert_array_equal(session.px, px)
    np.testing.assert_array_equal(session.py, py)
    np.testing.assert_array_equal(session.presence_flag, presence_flag)
    np.testing.assert_array_equal(session.X_full, X_full)
    assert session.feature_names == feature_names


def test_stage_clean_reprojects_mismatched_occurrence_crs(tiny_stack):
    """Regression test: occurrence coordinates in a different CRS than the
    predictor stack must be reprojected before extent-checking/extraction —
    otherwise real-world (lon, lat) points get compared directly against a
    projected-CRS (meters) extent and either get silently misattributed or
    dropped with a misleading "out of extent" reason.
    """
    from rasterio.warp import transform as warp_transform

    from sdm_plugin.core.io.occurrences import OccurrenceData

    stack = load_stack(tiny_stack)
    assert stack.crs == "EPSG:32633"

    # Points well inside the tiny_stack's UTM33N extent (x, y both in [0, 40]).
    utm_x = [10.0, 20.0, 30.0]
    utm_y = [10.0, 20.0, 30.0]
    lon, lat = warp_transform("EPSG:32633", "EPSG:4326", utm_x, utm_y)

    occ = OccurrenceData(
        x=np.array(lon), y=np.array(lat), presence=np.ones(3, dtype=np.uint8), crs="EPSG:4326"
    )

    cfg = SDMConfig()
    cfg.cleaning.auto_clean = True
    cfg.cleaning.thin_to_raster_resolution = False
    cleaned, report, _ = stage_clean(cfg, occ, stack)

    assert cleaned.crs == stack.crs
    assert report.dropped == {}, "correctly-reprojected in-extent points must not be dropped"
    np.testing.assert_allclose(sorted(cleaned.x), sorted(utm_x), atol=1e-6)
    np.testing.assert_allclose(sorted(cleaned.y), sorted(utm_y), atol=1e-6)


def test_stage_vif_matches_direct_call(tiny_stack, po_csv, tmp_path):
    cfg = _po_config(tiny_stack, po_csv, tmp_path)
    stack = load_stack(cfg.rasters.paths)
    occ = load_occurrences(cfg.occurrence.path, crs=cfg.occurrence.crs)
    rng = np.random.default_rng(cfg.random_seed)
    _px, _py, _pf, X_full, feature_names = collect_labeled_points_and_extract(
        cfg, occ, stack, rng
    )

    X_kept, kept_names, kept_idx, vif_report = stage_vif(cfg, X_full, feature_names)
    X_kept_direct, kept_names_direct, vif_report_direct = stepwise_vif(
        X_full, feature_names, cutoff=cfg.vif.cutoff
    )
    np.testing.assert_array_equal(X_kept, X_kept_direct)
    assert kept_names == kept_names_direct
    assert kept_idx == [feature_names.index(n) for n in kept_names]
    assert vif_report.as_dict() == vif_report_direct.as_dict()


def test_stage_vif_disabled_keeps_every_predictor(tiny_stack, po_csv, tmp_path):
    """cfg.vif.enabled=False must bypass stepwise elimination entirely, not
    just raise the cutoff — every original predictor survives untouched,
    even ones that would normally be dropped for collinearity."""
    cfg = _po_config(tiny_stack, po_csv, tmp_path)
    cfg.vif.enabled = False
    stack = load_stack(cfg.rasters.paths)
    occ = load_occurrences(cfg.occurrence.path, crs=cfg.occurrence.crs)
    rng = np.random.default_rng(cfg.random_seed)
    _px, _py, _pf, X_full, feature_names = collect_labeled_points_and_extract(
        cfg, occ, stack, rng
    )

    X_kept, kept_names, kept_idx, vif_report = stage_vif(cfg, X_full, feature_names)
    np.testing.assert_array_equal(X_kept, X_full)
    assert kept_names == feature_names
    assert kept_idx == list(range(len(feature_names)))
    assert vif_report.skipped is True
    assert vif_report.retained == feature_names
    assert vif_report.dropped == []
    assert vif_report.steps == []


def test_validate_matching_bands_ok(tiny_stack):
    stack = load_stack(tiny_stack)
    validate_matching_bands(stack, stack)  # no raise


def test_validate_matching_bands_reordered_raises(tiny_stack):
    """Regression test: same predictor set, different order must be rejected
    — kept_idx is computed against the training stack's band order and
    reused as-is against the projection stack, so a silently-accepted
    reordering would feed every model the wrong predictor's values."""
    stack = load_stack(tiny_stack)
    reordered = load_stack(list(reversed(tiny_stack)))
    assert sorted(stack.names) == sorted(reordered.names)
    assert stack.names != reordered.names
    try:
        validate_matching_bands(stack, reordered)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "different order" in str(e)


def test_validate_matching_bands_mismatch(tiny_stack):
    stack = load_stack(tiny_stack)
    other = load_stack(tiny_stack[:2])
    try:
        validate_matching_bands(stack, other)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_make_folds_spatial_block_hexagon(tiny_stack, po_csv):
    """cfg.split.block_shape must actually reach spatial_block_folds (not
    just be accepted and silently ignored) — verified via the plan it
    returns, which only spatial_block_folds itself can populate."""
    cfg = SDMConfig()
    cfg.data_mode = "presence_only"
    cfg.background.count = 200
    cfg.split.method = "spatial_block"
    cfg.split.k = 3
    cfg.split.auto_block_size = True
    cfg.split.block_shape = "hexagon"
    stack = load_stack(tiny_stack)
    occ = load_occurrences(po_csv, crs="EPSG:32633")
    rng = np.random.default_rng(0)

    px, py, presence_flag, X_full, _ = collect_labeled_points_and_extract(cfg, occ, stack, rng)
    folds, plan, fold_id = make_folds(cfg, X_full, presence_flag, px, py, stack, rng)

    assert plan.shape == "hexagon"
    assert plan.block_centers
    assert len(fold_id) == len(px)
    for train, test in folds:
        assert set(train.tolist()).isdisjoint(set(test.tolist()))
