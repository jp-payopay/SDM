import csv
import json
from pathlib import Path

from sdm_plugin.core.config import SDMConfig
from sdm_plugin.core.pipeline import Pipeline


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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


def test_pipeline_writes_csv_tables_for_metrics_curves_and_importance(
    tiny_stack, po_csv, tmp_path
):
    """Every number the report and the figures show must also land in a CSV:
    scores per fold, per replicate and summarised, the response-curve values,
    and the variable-importance values."""
    cfg = SDMConfig()
    cfg.data_mode = "presence_only"
    cfg.occurrence.path = po_csv
    cfg.occurrence.crs = "EPSG:32633"
    cfg.rasters.paths = tiny_stack
    cfg.background.count = 200
    cfg.background.method = "random"
    cfg.vif.cutoff = 10.0
    cfg.split.method = "kfold"
    cfg.split.k = 3
    cfg.modeling.algorithms = ["lr", "rf"]
    cfg.modeling.replicates = 2
    cfg.output.directory = str(tmp_path / "out")

    result = Pipeline(cfg).run()
    out = Path(result.output_dir)
    expected = [
        "metrics_detail.csv",
        "metrics_summary.csv",
        "response_curves.csv",
        "response_curves_summary.csv",
        "variable_importance.csv",
        "variable_importance_summary.csv",
    ]
    for name in expected:
        assert (out / name).exists(), f"{name} was not written"
        # Every CSV is also offered to the user as an output file.
        assert any(Path(f).name == name for f in result.output_files)

    # 2 algorithms x 2 replicates, plus the scored ensemble...
    detail = read_csv_rows(out / "metrics_detail.csv")
    per_rep = [r for r in detail if r["level"] == "replicate"]
    assert len(per_rep) == 6
    assert {r["algorithm"] for r in per_rep} == {"lr", "rf", "ensemble"}

    # ...each of which is broken down into its 3 folds in the same file.
    per_fold = [r for r in detail if r["level"] == "fold"]
    assert len(per_fold) == 18
    assert {r["fold"] for r in per_fold} == {"1", "2", "3"}
    assert all(int(r["n_test"]) > 0 for r in per_fold)
    assert all(int(r["n_presence_test"]) > 0 for r in per_fold)
    # A replicate's pooled row covers exactly the points its folds held out.
    for rep in per_rep:
        own_folds = [
            f for f in per_fold
            if f["algorithm"] == rep["algorithm"] and f["replicate"] == rep["replicate"]
        ]
        assert int(rep["n_test"]) == sum(int(f["n_test"]) for f in own_folds)

    # The summary CSV carries the same rows the run returns and the report shows.
    summary = read_csv_rows(out / "metrics_summary.csv")
    assert [r["algorithm"] for r in summary] == [
        r["algorithm"] for r in result.metrics_summary
    ]

    # The curve CSVs describe the same variables the curve PNGs were drawn for.
    curve_rows = read_csv_rows(out / "response_curves_summary.csv")
    plotted = {p.name[len("response_lr_"):-4] for p in (out / "plots").glob("response_lr_*.png")}
    assert {r["variable"] for r in curve_rows if r["algorithm"] == "lr"} == plotted
    assert "ensemble" in {r["algorithm"] for r in curve_rows}

    imp = read_csv_rows(out / "variable_importance_summary.csv")
    assert {r["algorithm"] for r in imp} == {"lr", "rf", "ensemble"}
    # Ranks run 1..n_variables within each algorithm, most important first.
    lr_ranks = [int(r["rank"]) for r in imp if r["algorithm"] == "lr"]
    assert lr_ranks == list(range(1, len(lr_ranks) + 1))


def test_pipeline_per_fold_csv_records_block_counts_for_spatial_blocks(
    tiny_stack, pa_csv, tmp_path
):
    """Under spatial-block CV each fold is a set of whole blocks, so the
    per-fold table says how many blocks the score covers — the one column that
    is only meaningful for this split method."""
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
    cfg.modeling.algorithms = ["lr"]
    cfg.modeling.replicates = 1
    cfg.output.directory = str(tmp_path / "out")

    result = Pipeline(cfg).run()
    rows = read_csv_rows(Path(result.output_dir) / "metrics_detail.csv")
    folds = [r for r in rows if r["level"] == "fold"]
    assert folds
    assert all(int(r["n_blocks"]) > 0 for r in folds)
    # n_blocks describes a fold, so the pooled rows leave it blank.
    assert all(r["n_blocks"] == "" for r in rows if r["level"] == "replicate")


def test_pipeline_csv_tables_cover_a_single_algorithm_run(tiny_stack, pa_csv, tmp_path):
    """With one algorithm there is no across-algorithm ensemble to score, but
    the plugin still draws an ensemble response curve — so the curve CSV must
    still carry it, while the metrics CSVs carry no ensemble row."""
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
    out = Path(result.output_dir)
    assert not result.failed_runs

    detail = read_csv_rows(out / "metrics_detail.csv")
    assert {r["algorithm"] for r in detail} == {"lr"}
    # A random holdout is a single train/test split — one "fold".
    per_fold = [r for r in detail if r["level"] == "fold"]
    assert len(per_fold) == 1
    assert per_fold[0]["fold"] == "1"
    assert per_fold[0]["n_blocks"] == ""

    curves = read_csv_rows(out / "response_curves_summary.csv")
    assert "ensemble" in {r["algorithm"] for r in curves}


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
