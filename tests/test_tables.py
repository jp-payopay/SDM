from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from sdm_plugin.core.evaluation.metrics import EvaluationResult, FoldEvaluation, evaluate_fold
from sdm_plugin.core.io.tables import (
    write_csv,
    write_importance_tables,
    write_metrics_tables,
    write_response_curve_tables,
)
from sdm_plugin.core.viz.response_curves import ensemble_response_values


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class FakeReplicate:
    """The handful of attributes write_metrics_tables reads off a
    ReplicateResult, without running a pipeline to get one."""

    def __init__(self, algorithm, replicate, metrics, fold_metrics=(), error=None):
        self.algorithm = algorithm
        self.replicate = replicate
        self.metrics = metrics
        self.fold_metrics = list(fold_metrics)
        self.error = error


def _metrics(auc=0.8, tss=0.6, boyce=0.5, threshold=0.4):
    return EvaluationResult(auc=auc, tss=tss, boyce=boyce, threshold=threshold)


def _fold(fold, **kw):
    base = {
        "n_train": 80, "n_test": 20, "n_presence_test": 10,
        "n_background_test": 10, "metrics": _metrics(), "n_blocks": None,
    }
    base.update(kw)
    return FoldEvaluation(fold=fold, **base)


def test_write_csv_writes_blank_cells_for_missing_and_non_finite(tmp_path):
    path = write_csv(
        tmp_path / "t.csv",
        ["a", "b", "c"],
        [{"a": float("nan"), "b": None}, {"a": 1.5, "b": "x", "c": 2}],
    )
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["a", "b", "c"]
    # NaN, None and an absent key all read back as an empty cell — which
    # pandas and R both take as missing, unlike the string "nan".
    assert rows[1] == ["", "", ""]
    assert rows[2] == ["1.5", "x", "2"]


def test_write_csv_uses_plain_line_endings(tmp_path):
    """The csv module needs newline="" or it doubles \\r on Windows."""
    path = write_csv(tmp_path / "t.csv", ["a"], [{"a": 1}])
    assert b"\r\r\n" not in path.read_bytes()


def test_metrics_tables_cover_every_fold_replicate_and_algorithm(tmp_path):
    reps = [
        FakeReplicate("lr", 0, _metrics(auc=0.81), [_fold(1), _fold(2)]),
        FakeReplicate("lr", 1, _metrics(auc=0.83), [_fold(1), _fold(2)]),
        FakeReplicate("rf", 0, _metrics(auc=0.90), [_fold(1), _fold(2)]),
        FakeReplicate("rf", 1, _metrics(auc=0.88), [_fold(1), _fold(2)]),
    ]
    ens_reps = [(0, _metrics(auc=0.92)), (1, _metrics(auc=0.91))]
    ens_folds = [(0, _fold(1)), (0, _fold(2)), (1, _fold(1)), (1, _fold(2))]
    summary = [{
        "algorithm": "Random Forest", "auc_mean": 0.89, "auc_sd": 0.01,
        "tss_mean": 0.6, "tss_sd": 0.02, "boyce_mean": 0.5, "boyce_sd": 0.03,
        "n_replicates": 2,
    }]

    write_metrics_tables(
        tmp_path,
        replicate_results=reps,
        ensemble_replicates=ens_reps,
        ensemble_folds=ens_folds,
        metrics_summary=summary,
        label_of={"lr": "Logistic Regression", "rf": "Random Forest"}.get,
    )

    detail = read_rows(tmp_path / "metrics_detail.csv")
    per_rep = [r for r in detail if r["level"] == "replicate"]
    per_fold = [r for r in detail if r["level"] == "fold"]
    assert len(per_rep) == 6  # 2 algorithms x 2 replicates + 2 ensemble rows
    assert len(per_fold) == 12  # each of those broken into its 2 folds
    assert len(detail) == len(per_rep) + len(per_fold)

    # Replicates are 1-based in the CSVs.
    assert {r["replicate"] for r in per_rep} == {"1", "2"}
    assert {r["algorithm"] for r in per_rep} == {"lr", "rf", "ensemble"}
    lr_1 = next(r for r in per_rep if r["algorithm"] == "lr" and r["replicate"] == "1")
    assert lr_1["algorithm_label"] == "Logistic Regression"
    assert float(lr_1["auc"]) == pytest.approx(0.81)
    # A pooled row covers its folds' held-out points together, and has no
    # single training size to report.
    assert int(lr_1["n_test"]) == 40
    assert int(lr_1["n_presence_test"]) == 20
    assert lr_1["n_train"] == ""
    assert lr_1["fold"] == ""

    assert {r["fold"] for r in per_fold} == {"1", "2"}
    assert int(per_fold[0]["n_test"]) == 20
    assert int(per_fold[0]["n_presence_test"]) == 10
    assert int(per_fold[0]["n_train"]) == 80
    # No spatial blocks in this run, so the column is blank rather than 0.
    assert per_fold[0]["n_blocks"] == ""

    # Each replicate's pooled row leads, immediately followed by its own folds.
    assert detail[0]["level"] == "replicate"
    assert [r["level"] for r in detail[:3]] == ["replicate", "fold", "fold"]
    assert {r["algorithm"] for r in detail[:3]} == {"lr"}
    assert {r["replicate"] for r in detail[:3]} == {"1"}

    summary_rows = read_rows(tmp_path / "metrics_summary.csv")
    assert len(summary_rows) == 1
    assert summary_rows[0]["algorithm"] == "Random Forest"
    assert float(summary_rows[0]["auc_mean"]) == pytest.approx(0.89)


def test_metrics_tables_keep_failed_replicates_with_their_error(tmp_path):
    reps = [FakeReplicate("svm", 0, EvaluationResult(*[float("nan")] * 4), error="boom")]
    write_metrics_tables(tmp_path, replicate_results=reps)
    rows = read_rows(tmp_path / "metrics_detail.csv")
    # A failure produces no folds, so it contributes the pooled row alone.
    assert len(rows) == 1
    assert rows[0]["level"] == "replicate"
    assert rows[0]["error"] == "boom"
    assert rows[0]["auc"] == ""
    assert rows[0]["n_test"] == ""


def _curves(n_rep, features, offset=0.0):
    grid = np.linspace(0.0, 1.0, 5)
    return [
        {f: (grid, np.clip(grid * (0.5 + 0.1 * r) + offset, 0, 1)) for f in features}
        for r in range(n_rep)
    ]


def test_response_curve_tables_match_the_plotted_mean_and_ensemble(tmp_path):
    features = ["bio1", "bio12"]
    curves_by_algo = {"lr": _curves(3, features), "rf": _curves(3, features, offset=0.2)}
    per_algo_metric = {"lr": 0.8, "rf": 0.9}

    write_response_curve_tables(
        tmp_path,
        feature_names=features,
        curves_by_algo=curves_by_algo,
        per_algo_metric=per_algo_metric,
        ensemble_method="weighted_auc",
        label_of={"lr": "Logistic Regression", "rf": "Random Forest"}.get,
    )

    raw = read_rows(tmp_path / "response_curves.csv")
    # 2 algorithms x 3 replicates x 2 variables x 5 grid points
    assert len(raw) == 60
    assert {r["variable"] for r in raw} == set(features)

    summary = read_rows(tmp_path / "response_curves_summary.csv")
    # 3 series (lr, rf, ensemble) x 2 variables x 5 grid points
    assert len(summary) == 30
    assert {r["algorithm"] for r in summary} == {"lr", "rf", "ensemble"}

    # The exported values are the ones the figure draws.
    expected = ensemble_response_values(
        feature_names=features,
        curves_by_algo=curves_by_algo,
        per_algo_metric=per_algo_metric,
        ensemble_method="weighted_auc",
    )
    ens_rows = [r for r in summary if r["algorithm"] == "ensemble" and r["variable"] == "bio1"]
    assert [float(r["mean"]) for r in ens_rows] == pytest.approx(
        list(expected["bio1"].ensemble)
    )
    lr_rows = [r for r in summary if r["algorithm"] == "lr" and r["variable"] == "bio1"]
    assert [float(r["mean"]) for r in lr_rows] == pytest.approx(
        list(expected["bio1"].per_algo_mean["lr"])
    )
    # Weights are carried so the ensemble curve can be recomputed from the file.
    assert float(lr_rows[0]["ensemble_weight"]) == pytest.approx(0.8 / 1.7)
    assert ens_rows[0]["ensemble_weight"] == ""
    assert int(lr_rows[0]["n_curves"]) == 3
    assert int(ens_rows[0]["n_curves"]) == 2


def test_response_curve_tables_can_skip_the_ensemble(tmp_path):
    write_response_curve_tables(
        tmp_path,
        feature_names=["bio1"],
        curves_by_algo={"lr": _curves(2, ["bio1"])},
        per_algo_metric=None,
        ensemble_method="mean",
        write_ensemble=False,
    )
    summary = read_rows(tmp_path / "response_curves_summary.csv")
    assert {r["algorithm"] for r in summary} == {"lr"}
    assert all(r["ensemble_weight"] == "" for r in summary)


def test_importance_tables_rank_variables_and_keep_every_replicate(tmp_path):
    importance_by_algo = {
        "rf": [{"bio1": 0.10, "bio12": 0.02}, {"bio1": 0.14, "bio12": 0.04}],
    }
    write_importance_tables(
        tmp_path,
        importance_by_algo=importance_by_algo,
        ensemble_importance=[{"bio1": 0.2, "bio12": 0.01}],
        label_of={"rf": "Random Forest"}.get,
    )

    raw = read_rows(tmp_path / "variable_importance.csv")
    assert len(raw) == 6  # (2 rf replicates + 1 ensemble repeat) x 2 variables
    rf_rows = [r for r in raw if r["algorithm"] == "rf"]
    assert {r["replicate"] for r in rf_rows} == {"1", "2"}

    summary = read_rows(tmp_path / "variable_importance_summary.csv")
    rf_summary = [r for r in summary if r["algorithm"] == "rf"]
    # Rank 1 is the most important variable — the top bar on the plot.
    assert rf_summary[0]["variable"] == "bio1"
    assert rf_summary[0]["rank"] == "1"
    assert float(rf_summary[0]["importance_mean"]) == pytest.approx(0.12)
    assert float(rf_summary[0]["importance_sd"]) == pytest.approx(0.02)
    assert int(rf_summary[0]["n_replicates"]) == 2
    assert rf_summary[1]["variable"] == "bio12"
    assert [r["variable"] for r in summary if r["algorithm"] == "ensemble"] == ["bio1", "bio12"]


def test_evaluate_fold_counts_the_held_out_set(tmp_path):
    y_true = np.array([1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1, 0.3])
    fold = evaluate_fold(2, y_true, y_score, n_train=45, n_blocks=7)
    assert fold.fold == 2
    assert fold.n_train == 45
    assert fold.n_test == 5
    assert fold.n_presence_test == 2
    assert fold.n_background_test == 3
    assert fold.n_blocks == 7
    assert fold.metrics.auc == pytest.approx(1.0)
    assert not math.isnan(fold.metrics.tss)
