import json
from pathlib import Path

from sdm_plugin.core.config import SDMConfig
from sdm_plugin.core.pipeline import Pipeline


def test_pipeline_end_to_end_po(tiny_stack, po_csv, tmp_path):
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

    result = Pipeline(cfg).run()
    assert Path(result.output_dir).exists()
    assert any(f.endswith(".tif") for f in result.output_files)
    assert any("run_config.json" in f for f in result.output_files)
    assert result.report_path is not None and Path(result.report_path).exists()

    out = Path(result.output_dir)
    # The ensemble is scored as its own row (2+ algorithms).
    assert any(r["algorithm"] == "Ensemble" for r in result.metrics_summary)
    # Ensemble variable importance (approach B) produces its own plot.
    assert (out / "plots" / "importance_Ensemble.png").exists()
    # Per-replicate metrics include ensemble rows.
    per_rep = json.loads((out / "metrics_per_replicate.json").read_text())
    assert any(r["algorithm"] == "Ensemble" for r in per_rep)
    # The report includes the generalized-interpretation section.
    report_html = Path(result.report_path).read_text(encoding="utf-8")
    assert "Generalized interpretations" in report_html
    # Total runtime is measured and shown in both the result and the report.
    assert result.duration_seconds > 0
    assert "Total runtime:" in report_html
    # The report's footer version must come from cfg.version, not a
    # hardcoded fallback in html_report.py that could silently drift from
    # the actual plugin version.
    assert f"SDM v{cfg.version}" in report_html


def test_pipeline_end_to_end_pa(tiny_stack, pa_csv, tmp_path):
    cfg = SDMConfig()
    cfg.data_mode = "presence_absence"
    cfg.occurrence.path = pa_csv
    cfg.occurrence.presence_field = "presence"
    cfg.occurrence.crs = "EPSG:32633"
    cfg.rasters.paths = tiny_stack
    cfg.vif.cutoff = 10.0
    cfg.split.method = "random"
    cfg.split.test_size = 0.3
    cfg.modeling.algorithms = ["lr"]
    cfg.modeling.replicates = 1
    cfg.output.directory = str(tmp_path / "out")

    result = Pipeline(cfg).run()
    assert result.metrics_summary
    assert not result.failed_runs


def test_pipeline_vif_disabled_keeps_all_predictors_and_report_shows_skipped(tiny_stack, pa_csv, tmp_path):
    cfg = SDMConfig()
    cfg.data_mode = "presence_absence"
    cfg.occurrence.path = pa_csv
    cfg.occurrence.presence_field = "presence"
    cfg.occurrence.crs = "EPSG:32633"
    cfg.rasters.paths = tiny_stack
    cfg.vif.enabled = False
    cfg.split.method = "random"
    cfg.split.test_size = 0.3
    cfg.modeling.algorithms = ["lr"]
    cfg.modeling.replicates = 1
    cfg.output.directory = str(tmp_path / "out")

    result = Pipeline(cfg).run()
    assert result.metrics_summary
    vif_report = json.loads((Path(result.output_dir) / "vif_report.json").read_text())
    assert vif_report["skipped"] is True
    assert len(vif_report["retained"]) == len(tiny_stack)
    assert vif_report["dropped"] == []
    report_html = Path(result.report_path).read_text(encoding="utf-8")
    assert "Skipped by user choice" in report_html


def test_pipeline_report_reflects_spatial_block_plan(tiny_stack, pa_csv, tmp_path):
    """Regression test: the report's Cross-validation section was wired to
    show split.plan.block_size/source/shape, but pipeline.py hardcoded
    split.plan=None so it silently never rendered anything for spatial-block
    runs. The captured plan (and the block-shape choice specifically) must
    actually reach the report.
    """
    cfg = SDMConfig()
    cfg.data_mode = "presence_absence"
    cfg.occurrence.path = pa_csv
    cfg.occurrence.presence_field = "presence"
    cfg.occurrence.crs = "EPSG:32633"
    cfg.rasters.paths = tiny_stack
    cfg.vif.cutoff = 10.0
    cfg.split.method = "spatial_block"
    cfg.split.k = 2
    cfg.split.auto_block_size = True
    cfg.split.block_shape = "hexagon"
    cfg.modeling.algorithms = ["lr"]
    cfg.modeling.replicates = 1
    cfg.output.directory = str(tmp_path / "out")

    result = Pipeline(cfg).run()
    assert result.metrics_summary
    report_html = Path(result.report_path).read_text(encoding="utf-8")
    assert "hexagonal" in report_html
    assert "block size:" in report_html


def test_pipeline_lists_saved_model_files_in_output_files(tiny_stack, pa_csv, tmp_path):
    """Regression test: saved per-replicate .joblib model files were written
    to out_dir/models/ correctly but never appended to result.output_files,
    so report.html's "Output files" section and the Summary page silently
    omitted them even though they exist on disk.
    """
    cfg = SDMConfig()
    cfg.data_mode = "presence_absence"
    cfg.occurrence.path = pa_csv
    cfg.occurrence.presence_field = "presence"
    cfg.occurrence.crs = "EPSG:32633"
    cfg.rasters.paths = tiny_stack
    cfg.vif.cutoff = 10.0
    cfg.split.method = "random"
    cfg.split.test_size = 0.3
    cfg.modeling.algorithms = ["lr"]
    cfg.modeling.replicates = 2
    cfg.output.directory = str(tmp_path / "out")
    assert cfg.output.save_models is True

    result = Pipeline(cfg).run()
    model_files = [f for f in result.output_files if f.endswith(".joblib")]
    on_disk = sorted((Path(cfg.output.directory) / "models").glob("*.joblib"))
    assert on_disk, "expected saved model files on disk"
    assert sorted(Path(f).name for f in model_files) == [p.name for p in on_disk]


def test_pipeline_projection_works_without_saved_models(tiny_stack, po_csv, tmp_path):
    """Regression test: projection must not depend on models being written to
    disk. Reusing the in-memory ReplicateResult.model (set regardless of
    save_models) rather than round-tripping through model_path/load_model —
    previously, save_models=False silently produced zero projection_<algo>.tif
    files (model_path was always None) while still reporting success.
    """
    cfg = SDMConfig()
    cfg.data_mode = "presence_only"
    cfg.occurrence.path = po_csv
    cfg.occurrence.crs = "EPSG:32633"
    cfg.rasters.paths = tiny_stack
    cfg.rasters.projection_paths = tiny_stack
    cfg.background.count = 200
    cfg.background.method = "random"
    cfg.vif.cutoff = 10.0
    cfg.split.method = "kfold"
    cfg.split.k = 3
    cfg.modeling.algorithms = ["lr", "rf"]
    cfg.modeling.replicates = 2
    cfg.output.directory = str(tmp_path / "out")
    cfg.output.save_models = False

    result = Pipeline(cfg).run()
    proj_files = [f for f in result.output_files if "projection_lr" in f or "projection_rf" in f]
    assert proj_files, "expected per-algorithm projection rasters even with save_models=False"
    assert not (Path(cfg.output.directory) / "models").exists()
