from __future__ import annotations

from pathlib import Path

import numpy as np

from sdm_plugin.core.config import SDMConfig
from sdm_plugin.core.io.occurrences import load_occurrences
from sdm_plugin.core.io.rasters import load_stack
from sdm_plugin.core.pipeline import Pipeline
from sdm_plugin.core.session import PipelineSession
from sdm_plugin.core.stages import collect_labeled_points_and_extract, stage_clean, stage_vif


def _po_config(tiny_stack, po_csv, out_dir) -> SDMConfig:
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
    cfg.output.directory = str(out_dir)
    return cfg


def test_pipeline_with_prepopulated_session_matches_fresh_run(tiny_stack, po_csv, tmp_path):
    cfg_fresh = _po_config(tiny_stack, po_csv, tmp_path / "fresh")
    result_fresh = Pipeline(cfg_fresh).run()

    # Build a session by walking the same stages a wizard's earlier pages
    # would have already run, then hand it to Pipeline.run().
    cfg_cached = _po_config(tiny_stack, po_csv, tmp_path / "cached")
    session = PipelineSession()
    session.stack = load_stack(cfg_cached.rasters.paths)
    session.occ_raw = load_occurrences(cfg_cached.occurrence.path, crs=cfg_cached.occurrence.crs)
    session.occ, session.cleaning_report, session.thinning_report = stage_clean(
        cfg_cached, session.occ_raw, session.stack
    )
    rng = np.random.default_rng(cfg_cached.random_seed)
    (
        session.px,
        session.py,
        session.presence_flag,
        session.X_full,
        session.feature_names,
    ) = collect_labeled_points_and_extract(cfg_cached, session.occ, session.stack, rng)
    (
        session.X_kept,
        session.kept_names,
        session.kept_idx,
        session.vif_report,
    ) = stage_vif(cfg_cached, session.X_full, session.feature_names)

    result_cached = Pipeline(cfg_cached, session=session).run()

    assert len(result_fresh.metrics_summary) == len(result_cached.metrics_summary)
    for fresh, cached in zip(result_fresh.metrics_summary, result_cached.metrics_summary):
        assert fresh["algorithm"] == cached["algorithm"]
        assert np.isclose(fresh["auc_mean"], cached["auc_mean"], equal_nan=True)
        assert np.isclose(fresh["tss_mean"], cached["tss_mean"], equal_nan=True)

    fresh_names = sorted(Path(p).name for p in result_fresh.output_files)
    cached_names = sorted(Path(p).name for p in result_cached.output_files)
    assert fresh_names == cached_names


def test_pipeline_without_session_unchanged(tiny_stack, po_csv, tmp_path):
    """Pipeline(config) with no session argument behaves exactly as before
    the refactor — same call signature, same result shape."""
    cfg = _po_config(tiny_stack, po_csv, tmp_path / "out")
    result = Pipeline(cfg).run()
    assert result.metrics_summary
    assert Path(result.output_dir).exists()
