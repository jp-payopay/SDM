from __future__ import annotations

from pathlib import Path

import numpy as np

from . import PUBLICATION_DPI
from ..prediction.ensemble import EnsembleMethod, compute_weights


def plot_response_curves(
    *,
    algorithm: str,
    feature_names: list[str],
    curves_per_replicate: list[dict[str, tuple[np.ndarray, np.ndarray]]],
    out_dir: str | Path,
) -> list[Path]:
    """One PNG per feature: overlaid per-replicate lines + mean±SD band.

    `curves_per_replicate` is a list where each item maps feature_name -> (grid, prediction).
    All grids for a given feature are assumed identical (they come from the same
    quantile range of the training set).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for feat in feature_names:
        grids: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        for rep in curves_per_replicate:
            if feat not in rep:
                continue
            g, p = rep[feat]
            grids.append(g)
            preds.append(p)
        if not preds:
            continue
        grid = grids[0]
        P = np.stack(preds, axis=0)  # (n_rep, n_grid)
        mean = P.mean(axis=0)
        sd = P.std(axis=0)

        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=110)
        for row in P:
            ax.plot(grid, row, color="gray", alpha=0.35, linewidth=0.8)
        ax.plot(grid, mean, color="C0", linewidth=2.0, label="mean")
        ax.fill_between(grid, mean - sd, mean + sd, color="C0", alpha=0.25, label="±SD")
        ax.set_xlabel(feat)
        ax.set_ylabel("Predicted suitability")
        ax.set_title(f"{algorithm}: response for {feat}")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fname = out_dir / f"response_{algorithm}_{feat}.png"
        fig.savefig(fname, dpi=PUBLICATION_DPI)
        plt.close(fig)
        saved.append(fname)
    return saved


def plot_ensemble_response_curves(
    *,
    feature_names: list[str],
    curves_by_algo: dict[str, list[dict[str, tuple[np.ndarray, np.ndarray]]]],
    algo_labels: dict[str, str],
    per_algo_metric: dict[str, float] | None,
    ensemble_method: EnsembleMethod,
    out_dir: str | Path,
) -> list[Path]:
    """One PNG per feature: every algorithm's own mean response curve (thin,
    colored) plus the weighted ensemble response (bold black) — the weights
    are computed the same way (`compute_weights`) as the raster ensemble
    itself, so this plot shows exactly what's actually being combined.

    `curves_by_algo` maps algorithm name -> that algorithm's
    `curves_per_replicate` (same shape `plot_response_curves` takes).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for feat in feature_names:
        algo_means: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for algo, curves_per_replicate in curves_by_algo.items():
            grids, preds = [], []
            for rep in curves_per_replicate:
                if feat not in rep:
                    continue
                g, p = rep[feat]
                grids.append(g)
                preds.append(p)
            if not preds:
                continue
            algo_means[algo] = (grids[0], np.stack(preds, axis=0).mean(axis=0))
        if not algo_means:
            continue

        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=110)
        for i, (algo, (grid, mean)) in enumerate(algo_means.items()):
            ax.plot(
                grid, mean,
                color=color_cycle[i % len(color_cycle)], linewidth=1.3, alpha=0.85,
                label=algo_labels.get(algo, algo),
            )

        weights = compute_weights(list(algo_means.keys()), per_algo_metric, ensemble_method)
        common_grid = next(iter(algo_means.values()))[0]
        ensemble_curve = np.zeros_like(common_grid, dtype=float)
        for algo, (grid, mean) in algo_means.items():
            vals = mean if np.array_equal(grid, common_grid) else np.interp(common_grid, grid, mean)
            ensemble_curve += weights[algo] * vals
        ax.plot(common_grid, ensemble_curve, color="black", linewidth=2.5, label="Ensemble")

        ax.set_xlabel(feat)
        ax.set_ylabel("Predicted suitability")
        ax.set_title(f"Ensemble: response for {feat}")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="best", fontsize=7, ncol=2)
        fig.tight_layout()
        fname = out_dir / f"response_ensemble_{feat}.png"
        fig.savefig(fname, dpi=PUBLICATION_DPI)
        plt.close(fig)
        saved.append(fname)
    return saved
