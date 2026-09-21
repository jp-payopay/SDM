from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.metrics import EvaluationResult, FoldEvaluation
from ..prediction.ensemble import EnsembleMethod
from ..viz.response_curves import ensemble_response_values, stack_curves
from ..viz.var_importance import importance_summary

# Replicates and folds are numbered from 1 in every CSV, the way a methods
# section talks about them ("replicate 3 of 10"). The JSON files keep their
# original 0-based `replicate` field, so a CSV row and its JSON counterpart
# are off by one on that column alone.
ENSEMBLE_CODE = "ensemble"
ENSEMBLE_LABEL = "Ensemble"


def _fmt(value: Any) -> str:
    """One CSV cell. Floats get 10 significant digits — more than any of these
    quantities carries — and anything non-finite or missing is written as an
    empty cell, which pandas, R and Excel all read back as missing rather than
    as the string "nan"."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return "" if not math.isfinite(v) else f"{v:.10g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def write_csv(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Write `rows` as a CSV with `fieldnames` as its header. A field a row
    omits is written as an empty cell."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" is required for the csv module on Windows, which would
    # otherwise write \r\r\n line endings.
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([_fmt(row.get(f)) for f in fieldnames])
    return path


# ----- model performance -----

# Fold rows and pooled replicate rows share one table, told apart by `level`
# and ordered so each replicate's pooled row is followed by the folds it was
# pooled from. Columns that only apply to one level (fold, n_blocks, n_train,
# error) are simply blank on the other.
LEVEL_REPLICATE = "replicate"
LEVEL_FOLD = "fold"

_METRIC_FIELDS = (
    "algorithm",
    "algorithm_label",
    "replicate",
    "level",
    "fold",
    "n_blocks",
    "n_train",
    "n_test",
    "n_presence_test",
    "n_background_test",
    "auc",
    "tss",
    "boyce",
    "threshold",
    "error",
)

_SUMMARY_FIELDS = (
    "algorithm",
    "auc_mean",
    "auc_sd",
    "tss_mean",
    "tss_sd",
    "boyce_mean",
    "boyce_sd",
    "n_replicates",
)


def _pooled_counts(folds: Sequence[FoldEvaluation]) -> dict[str, Any]:
    """How much held-out data a replicate's pooled score covers — the union of
    its folds' test sets, since every point is held out exactly once.

    There is deliberately no `n_train`: each fold trained on a different subset,
    so a pooled row has no single training size to report.
    """
    if not folds:
        return {}
    return {
        "n_test": sum(f.n_test for f in folds),
        "n_presence_test": sum(f.n_presence_test for f in folds),
        "n_background_test": sum(f.n_background_test for f in folds),
    }


def write_metrics_tables(
    out_dir: str | Path,
    *,
    replicate_results: Sequence[Any],
    ensemble_replicates: Sequence[tuple[int, EvaluationResult]] = (),
    ensemble_folds: Sequence[tuple[int, FoldEvaluation]] = (),
    metrics_summary: Sequence[Mapping[str, Any]] = (),
    label_of: Callable[[str], str] = str,
) -> list[Path]:
    """Write the two model-performance tables.

    - `metrics_detail.csv` — every score a run computed, at both grains it
      computes them at, with a `level` column saying which:
      `replicate` rows are scored on the held-out predictions pooled across a
      replicate's folds (the numbers the summary averages, and the row carrying
      the error text if the replicate failed), and `fold` rows are each fold
      scored on its own held-out points alone. A replicate's pooled row comes
      first, followed by its folds, so filtering on `level` gives back either
      of the two tables this used to be.
    - `metrics_summary.csv` — one row per algorithm: mean and SD across
      replicates, the same table the report shows.

    `replicate_results` are the pipeline's per-replicate results; only the
    `.algorithm`, `.replicate`, `.metrics`, `.fold_metrics` and `.error`
    attributes are read. Ensemble rows, when the run scored one, are written
    alongside the individual algorithms.
    """
    out_dir = Path(out_dir)

    rows: list[dict[str, Any]] = []

    def add_replicate(base: dict[str, Any], metrics, folds, error) -> None:
        rows.append({
            **base, "level": LEVEL_REPLICATE, **_pooled_counts(folds),
            **metrics.as_dict(), "error": error,
        })
        for fold in folds:
            rows.append({**base, "level": LEVEL_FOLD, **fold.as_dict()})

    for r in replicate_results:
        add_replicate(
            {
                "algorithm": r.algorithm,
                "algorithm_label": label_of(r.algorithm),
                "replicate": r.replicate + 1,
            },
            r.metrics,
            list(getattr(r, "fold_metrics", ()) or ()),
            r.error,
        )

    ens_folds_by_replicate: dict[int, list[FoldEvaluation]] = {}
    for rep_i, fold in ensemble_folds:
        ens_folds_by_replicate.setdefault(rep_i, []).append(fold)
    for rep_i, metrics in ensemble_replicates:
        add_replicate(
            {
                "algorithm": ENSEMBLE_CODE,
                "algorithm_label": ENSEMBLE_LABEL,
                "replicate": rep_i + 1,
            },
            metrics,
            ens_folds_by_replicate.get(rep_i, []),
            None,
        )

    return [
        write_csv(out_dir / "metrics_detail.csv", _METRIC_FIELDS, rows),
        write_csv(out_dir / "metrics_summary.csv", _SUMMARY_FIELDS, metrics_summary),
    ]


# ----- response curves -----

_CURVE_FIELDS = (
    "algorithm",
    "algorithm_label",
    "replicate",
    "variable",
    "value",
    "suitability",
)

_CURVE_SUMMARY_FIELDS = (
    "algorithm",
    "algorithm_label",
    "variable",
    "value",
    "mean",
    "sd",
    "n_curves",
    "ensemble_weight",
)


def write_response_curve_tables(
    out_dir: str | Path,
    *,
    feature_names: Sequence[str],
    curves_by_algo: Mapping[str, list[dict[str, tuple[np.ndarray, np.ndarray]]]],
    per_algo_metric: Mapping[str, float] | None,
    ensemble_method: EnsembleMethod,
    label_of: Callable[[str], str] = str,
    write_ensemble: bool = True,
) -> list[Path]:
    """Write the numbers behind the response-curve plots.

    - `response_curves.csv` — every individual line: one row per algorithm x
      replicate x variable x grid point. These are the thin grey lines on a
      per-algorithm plot.
    - `response_curves_summary.csv` — the bold lines: each algorithm's mean
      curve with the SD band around it, plus (when the run built an ensemble)
      the weighted ensemble curve, whose SD column is the spread *across
      algorithms* rather than across replicates. `n_curves` counts what went
      into each mean accordingly: replicates for an algorithm row, algorithms
      for an ensemble row. `ensemble_weight` is that algorithm's share of the
      ensemble, and is empty on the ensemble row itself.
    """
    out_dir = Path(out_dir)
    feature_names = list(feature_names)

    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    responses = (
        ensemble_response_values(
            feature_names=feature_names,
            curves_by_algo=dict(curves_by_algo),
            per_algo_metric=dict(per_algo_metric) if per_algo_metric else None,
            ensemble_method=ensemble_method,
        )
        if write_ensemble and curves_by_algo
        else {}
    )

    for algo, curves_per_replicate in curves_by_algo.items():
        label = label_of(algo)
        for feat in feature_names:
            stack = stack_curves(feat, curves_per_replicate)
            if stack is None:
                continue
            for rep_i, preds in enumerate(stack.predictions):
                for x, yv in zip(stack.grid, preds):
                    curve_rows.append({
                        "algorithm": algo,
                        "algorithm_label": label,
                        "replicate": rep_i + 1,
                        "variable": feat,
                        "value": float(x),
                        "suitability": float(yv),
                    })
            weight = responses[feat].weights.get(algo) if feat in responses else None
            mean, sd = stack.mean, stack.sd
            for x, m, s in zip(stack.grid, mean, sd):
                summary_rows.append({
                    "algorithm": algo,
                    "algorithm_label": label,
                    "variable": feat,
                    "value": float(x),
                    "mean": float(m),
                    "sd": float(s),
                    "n_curves": stack.n_replicates,
                    "ensemble_weight": weight,
                })

    for feat, response in responses.items():
        across_algo = np.stack(list(response.per_algo_mean.values()), axis=0)
        sd_across = across_algo.std(axis=0)
        for x, m, s in zip(response.grid, response.ensemble, sd_across):
            summary_rows.append({
                "algorithm": ENSEMBLE_CODE,
                "algorithm_label": ENSEMBLE_LABEL,
                "variable": feat,
                "value": float(x),
                "mean": float(m),
                "sd": float(s),
                "n_curves": len(response.per_algo_mean),
                "ensemble_weight": None,
            })

    return [
        write_csv(out_dir / "response_curves.csv", _CURVE_FIELDS, curve_rows),
        write_csv(
            out_dir / "response_curves_summary.csv", _CURVE_SUMMARY_FIELDS, summary_rows
        ),
    ]


# ----- variable importance -----

_IMPORTANCE_FIELDS = (
    "algorithm",
    "algorithm_label",
    "replicate",
    "variable",
    "importance",
)

_IMPORTANCE_SUMMARY_FIELDS = (
    "algorithm",
    "algorithm_label",
    "variable",
    "importance_mean",
    "importance_sd",
    "n_replicates",
    "rank",
)


def write_importance_tables(
    out_dir: str | Path,
    *,
    importance_by_algo: Mapping[str, Sequence[Mapping[str, float]]],
    ensemble_importance: Sequence[Mapping[str, float]] | None = None,
    label_of: Callable[[str], str] = str,
) -> list[Path]:
    """Write the numbers behind the variable-importance plots.

    - `variable_importance.csv` — the raw permutation-importance scores: one
      row per algorithm x replicate x variable.
    - `variable_importance_summary.csv` — the plotted bars: mean importance per
      variable with the SD across replicates, plus `rank` (1 = most important),
      so the bar order is recoverable without re-sorting.

    Importance is the drop in AUC when that predictor's values are shuffled, so
    larger is more important and a slightly negative value means shuffling the
    predictor did not hurt. For the ensemble, whose importance is measured by
    repeating the permutation rather than by refitting, the `replicate` column
    numbers the permutation repeats.
    """
    out_dir = Path(out_dir)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    sources: list[tuple[str, str, Sequence[Mapping[str, float]]]] = [
        (algo, label_of(algo), per_rep) for algo, per_rep in importance_by_algo.items()
    ]
    if ensemble_importance:
        sources.append((ENSEMBLE_CODE, ENSEMBLE_LABEL, ensemble_importance))

    for algo, label, per_rep in sources:
        for rep_i, imp in enumerate(per_rep):
            for variable, value in imp.items():
                rows.append({
                    "algorithm": algo,
                    "algorithm_label": label,
                    "replicate": rep_i + 1,
                    "variable": variable,
                    "importance": float(value),
                })
        # importance_summary returns ascending by mean (the order the bars are
        # drawn, bottom to top); reverse it so rank 1 leads the file.
        ranked = list(reversed(importance_summary([dict(r) for r in per_rep])))
        for rank, (variable, mean, sd, n) in enumerate(ranked, start=1):
            summary_rows.append({
                "algorithm": algo,
                "algorithm_label": label,
                "variable": variable,
                "importance_mean": mean,
                "importance_sd": sd,
                "n_replicates": n,
                "rank": rank,
            })

    return [
        write_csv(out_dir / "variable_importance.csv", _IMPORTANCE_FIELDS, rows),
        write_csv(
            out_dir / "variable_importance_summary.csv",
            _IMPORTANCE_SUMMARY_FIELDS,
            summary_rows,
        ),
    ]
