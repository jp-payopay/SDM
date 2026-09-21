from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import PUBLICATION_DPI
from ..prediction.ensemble import EnsembleMethod, compute_weights


@dataclass
class CurveStack:
    """One feature's response curves across a single algorithm's replicates.

    `predictions` is (n_replicate, n_grid) over the shared `grid`, so `mean`
    and `sd` are the line and the band the per-algorithm plot draws.
    """

    grid: np.ndarray
    predictions: np.ndarray

    @property
    def mean(self) -> np.ndarray:
        return self.predictions.mean(axis=0)

    @property
    def sd(self) -> np.ndarray:
        return self.predictions.std(axis=0)

    @property
    def n_replicates(self) -> int:
        return int(self.predictions.shape[0])


@dataclass
class EnsembleResponse:
    """One feature's ensemble response: each algorithm's own mean curve, the
    weights used to combine them, and the resulting weighted curve — the three
    things the ensemble plot draws."""

    grid: np.ndarray
    per_algo_mean: dict[str, np.ndarray]
    per_algo_grid: dict[str, np.ndarray]
    weights: dict[str, float]
    ensemble: np.ndarray


def stack_curves(
    feature: str,
    curves_per_replicate: list[dict[str, tuple[np.ndarray, np.ndarray]]],
) -> CurveStack | None:
    """Gather one feature's curves across replicates, or None if no replicate
    produced one. All grids for a feature are assumed identical (they come
    from the same quantile range of the training set), so the first is kept."""
    grids: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for rep in curves_per_replicate:
        if feature not in rep:
            continue
        g, p = rep[feature]
        grids.append(np.asarray(g, dtype=float))
        preds.append(np.asarray(p, dtype=float))
    if not preds:
        return None
    return CurveStack(grid=grids[0], predictions=np.stack(preds, axis=0))


def ensemble_response_values(
    *,
    feature_names: list[str],
    curves_by_algo: dict[str, list[dict[str, tuple[np.ndarray, np.ndarray]]]],
    per_algo_metric: dict[str, float] | None,
    ensemble_method: EnsembleMethod,
) -> dict[str, EnsembleResponse]:
    """Per feature, the weighted ensemble response and its ingredients.

    Weights come from `compute_weights`, the same function the raster ensemble
    uses, so this curve is the response of exactly what is being combined on
    the map. Features no algorithm produced a curve for are omitted.
    """
    out: dict[str, EnsembleResponse] = {}
    for feat in feature_names:
        per_algo: dict[str, CurveStack] = {}
        for algo, curves_per_replicate in curves_by_algo.items():
            stack = stack_curves(feat, curves_per_replicate)
            if stack is not None:
                per_algo[algo] = stack
        if not per_algo:
            continue
        weights = compute_weights(list(per_algo.keys()), per_algo_metric, ensemble_method)
        common_grid = next(iter(per_algo.values())).grid
        ensemble = np.zeros_like(common_grid, dtype=float)
        for algo, stack in per_algo.items():
            mean = stack.mean
            vals = (
                mean
                if np.array_equal(stack.grid, common_grid)
                else np.interp(common_grid, stack.grid, mean)
            )
            ensemble += weights[algo] * vals
        out[feat] = EnsembleResponse(
            grid=common_grid,
            per_algo_mean={a: s.mean for a, s in per_algo.items()},
            per_algo_grid={a: s.grid for a, s in per_algo.items()},
            weights=weights,
            ensemble=ensemble,
        )
    return out


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
        stack = stack_curves(feat, curves_per_replicate)
        if stack is None:
            continue
        grid = stack.grid
        P = stack.predictions  # (n_rep, n_grid)
        mean = stack.mean
        sd = stack.sd

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

    responses = ensemble_response_values(
        feature_names=feature_names,
        curves_by_algo=curves_by_algo,
        per_algo_metric=per_algo_metric,
        ensemble_method=ensemble_method,
    )
    for feat, response in responses.items():
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=110)
        for i, (algo, mean) in enumerate(response.per_algo_mean.items()):
            ax.plot(
                response.per_algo_grid[algo], mean,
                color=color_cycle[i % len(color_cycle)], linewidth=1.3, alpha=0.85,
                label=algo_labels.get(algo, algo),
            )

        ax.plot(response.grid, response.ensemble, color="black", linewidth=2.5, label="Ensemble")

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
