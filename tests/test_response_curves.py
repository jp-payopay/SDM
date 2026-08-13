from __future__ import annotations

import numpy as np

from sdm_plugin.core.viz.response_curves import plot_ensemble_response_curves


def test_plot_ensemble_response_curves_writes_one_file_per_feature(tmp_path):
    grid = np.linspace(0.0, 1.0, 10)
    curves_by_algo = {
        "lr": [{"bio1": (grid, np.full(10, 0.2)), "bio2": (grid, np.full(10, 0.4))}],
        "rf": [{"bio1": (grid, np.full(10, 0.8)), "bio2": (grid, np.full(10, 0.6))}],
    }
    out = plot_ensemble_response_curves(
        feature_names=["bio1", "bio2"],
        curves_by_algo=curves_by_algo,
        algo_labels={"lr": "Logistic Regression", "rf": "Random Forest"},
        per_algo_metric=None,
        ensemble_method="mean",
        out_dir=tmp_path,
    )
    assert len(out) == 2
    assert all(p.exists() for p in out)
    assert {p.name for p in out} == {"response_ensemble_bio1.png", "response_ensemble_bio2.png"}


def test_plot_ensemble_response_curves_weighted_uses_compute_weights(tmp_path):
    grid = np.linspace(0.0, 1.0, 5)
    curves_by_algo = {
        "lr": [{"bio1": (grid, np.zeros(5))}],
        "rf": [{"bio1": (grid, np.ones(5))}],
    }
    # weighted_tss with rf given all the weight -> ensemble curve should equal rf's curve
    out = plot_ensemble_response_curves(
        feature_names=["bio1"],
        curves_by_algo=curves_by_algo,
        algo_labels={},
        per_algo_metric={"lr": 0.0, "rf": 1.0},
        ensemble_method="weighted_tss",
        out_dir=tmp_path,
    )
    assert len(out) == 1


def test_plot_ensemble_response_curves_skips_features_with_no_data(tmp_path):
    grid = np.linspace(0.0, 1.0, 5)
    curves_by_algo = {"lr": [{"bio1": (grid, np.zeros(5))}]}
    out = plot_ensemble_response_curves(
        feature_names=["bio1", "bio_missing"],
        curves_by_algo=curves_by_algo,
        algo_labels={},
        per_algo_metric=None,
        ensemble_method="mean",
        out_dir=tmp_path,
    )
    assert len(out) == 1
