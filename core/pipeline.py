from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import SDMConfig
from .evaluation.metrics import EvaluationResult, evaluate
from .evaluation.threshold import apply_threshold, apply_threshold_masked, maxtss_threshold
from .interpret import build_interpretation
from .io.occurrences import load_occurrences
from .io.outputs import save_json, save_model, write_raster
from .io.rasters import RasterStack, load_stack
from .models.config_export import build_model_config
from .models.registry import algorithm_long_name, build_model
from .prediction.ensemble import (
    compute_weights,
    ensemble_permutation_importance,
    ensemble_predictions,
)
from .prediction.mess_mop import mess, mop
from .prediction.predict import predict_raster
from .report.html_report import render_report
from .session import PipelineSession
from .stages import (
    collect_labeled_points_and_extract,
    make_folds,
    stage_clean,
    stage_vif,
    validate_matching_bands,
)
from .viz.raster_maps import plot_raster_map
from .viz.response_curves import plot_ensemble_response_curves, plot_response_curves
from .viz.var_importance import plot_variable_importance

ProgressCallback = Callable[[str, float, str], None]


@dataclass
class ReplicateResult:
    algorithm: str
    replicate: int
    metrics: EvaluationResult
    importances: dict[str, float]
    curves: dict[str, tuple[np.ndarray, np.ndarray]]
    prediction: np.ndarray | None = None
    threshold: float = float("nan")
    model_path: str | None = None
    error: str | None = None
    # Pooled held-out CV predictions (aligned across algorithms within a
    # replicate, since all algorithms share the same folds), used to score the
    # ensemble. `model` is the final all-data fit, kept in memory for ensemble
    # permutation importance. Neither is serialized.
    y_true: np.ndarray | None = None
    y_score: np.ndarray | None = None
    model: object | None = None


@dataclass
class RunResult:
    output_dir: str
    metrics_summary: list[dict]
    failed_runs: list[str] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)
    report_path: str | None = None


class Pipeline:
    def __init__(
        self,
        config: SDMConfig,
        progress: ProgressCallback | None = None,
        session: PipelineSession | None = None,
    ) -> None:
        self.cfg = config
        self._progress_cb = progress or (lambda *_: None)
        self._t0 = time.time()
        self._session = session

    def run(self) -> RunResult:
        errors = self.cfg.validate()
        if errors:
            raise ValueError("Invalid config:\n" + "\n".join(f"- {e}" for e in errors))

        out_dir = Path(self.cfg.output.directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(self.cfg.random_seed)

        # A pre-populated session (e.g. handed off from the wizard's earlier
        # pages) lets each of the stages below be skipped if already cached.
        session = self._session if self._session is not None else PipelineSession()

        # 1. Load
        if session.stack is None:
            self._progress("load", 0.0, "Loading rasters")
            session.stack = load_stack(self.cfg.rasters.paths)
        stack = session.stack

        if session.occ_raw is None:
            self._progress("load", 0.3, "Loading occurrences")
            session.occ_raw = load_occurrences(
                self.cfg.occurrence.path,
                x_field=self.cfg.occurrence.x_field,
                y_field=self.cfg.occurrence.y_field,
                presence_field=self.cfg.occurrence.presence_field,
                crs=self.cfg.occurrence.crs,
                layer_name=self.cfg.occurrence.layer_name,
            )
        self._progress("load", 1.0, f"Loaded {len(session.occ_raw.x)} occurrences")

        # 2. Clean
        if session.occ is None:
            self._progress("clean", 0.3, "Cleaning occurrences")
            occ, cleaning_rep, thinning_rep = stage_clean(self.cfg, session.occ_raw, stack)
            session.occ, session.cleaning_report, session.thinning_report = (
                occ,
                cleaning_rep,
                thinning_rep,
            )
        occ = session.occ
        cleaning_rep = session.cleaning_report
        thinning_rep = session.thinning_report
        self._progress("clean", 1.0, f"After cleaning: {len(occ.x)} points")

        # 3. Build (X, y) — presence and background
        if session.X_full is None:
            self._progress("background", 0.0, "Preparing predictors")
            px, py, presence_flag, X_full, feature_names = collect_labeled_points_and_extract(
                self.cfg, occ, stack, rng
            )
            session.px, session.py, session.presence_flag = px, py, presence_flag
            session.X_full, session.feature_names = X_full, feature_names
        px, py, presence_flag = session.px, session.py, session.presence_flag
        X_full, feature_names = session.X_full, session.feature_names
        y = presence_flag
        self._progress(
            "background",
            1.0,
            f"n_presence={int((y==1).sum())}, n_background={int((y==0).sum())}",
        )

        # 4. Stepwise VIF
        if session.X_kept is None:
            self._progress("vif", 0.0, "Running stepwise VIF")
            X_kept, kept_names, kept_idx, vif_report = stage_vif(self.cfg, X_full, feature_names)
            session.X_kept, session.kept_names = X_kept, kept_names
            session.kept_idx, session.vif_report = kept_idx, vif_report
        X_kept, kept_names = session.X_kept, session.kept_names
        kept_idx, vif_report = session.kept_idx, session.vif_report
        self._progress(
            "vif", 1.0, f"Retained {len(kept_names)}/{len(feature_names)} predictors"
        )

        # 5. Modeling per algorithm × replicate
        replicate_results: list[ReplicateResult] = []
        n_algo = len(self.cfg.modeling.algorithms)
        n_rep = self.cfg.modeling.replicates
        total_units = max(1, n_algo * n_rep)
        failed: list[str] = []

        for a_i, algo in enumerate(self.cfg.modeling.algorithms):
            for r_i in range(n_rep):
                unit = a_i * n_rep + r_i
                self._progress(
                    "modeling",
                    unit / total_units,
                    f"[{algo}] replicate {r_i + 1}/{n_rep}",
                )
                try:
                    res = self._run_one(
                        algo=algo,
                        replicate=r_i,
                        X_kept=X_kept,
                        y=y,
                        px=px,
                        py=py,
                        kept_names=kept_names,
                        stack=stack,
                        kept_idx=kept_idx,
                        rng=rng,
                        out_dir=out_dir,
                    )
                except Exception as exc:
                    tb = traceback.format_exc(limit=2)
                    failed.append(f"{algo} rep {r_i}: {exc}")
                    self._progress("modeling", unit / total_units, f"[{algo}] replicate {r_i} FAILED: {exc}")
                    res = ReplicateResult(
                        algorithm=algo,
                        replicate=r_i,
                        metrics=EvaluationResult(np.nan, np.nan, np.nan, np.nan),
                        importances={},
                        curves={},
                        error=f"{exc}\n{tb}",
                    )
                replicate_results.append(res)

        self._progress("modeling", 1.0, "Modeling complete")

        # 6. Per-algorithm aggregation + rasters
        self._progress("aggregate", 0.0, "Aggregating per-algorithm outputs")
        per_algo_raster: dict[str, np.ndarray] = {}
        per_algo_threshold: dict[str, float] = {}
        per_algo_auc: dict[str, float] = {}
        per_algo_tss: dict[str, float] = {}
        metrics_summary: list[dict] = []
        output_files: list[str] = []
        curves_by_algo: dict[str, list[dict[str, tuple[np.ndarray, np.ndarray]]]] = {}
        map_files: dict[str, dict[str, str]] = {}
        ensemble_rep_metrics: list[tuple[int, EvaluationResult]] = []
        ensemble_importance_name: str | None = None
        uncertainty_hi_frac: float | None = None
        extrap_frac: float | None = None
        suitable_frac: float | None = None
        ensemble_train_auc: float | None = None
        ensemble_cv_auc: float | None = None

        for algo in self.cfg.modeling.algorithms:
            reps = [r for r in replicate_results if r.algorithm == algo and r.error is None]
            if not reps:
                continue
            preds = [r.prediction for r in reps if r.prediction is not None]
            thresholds = [r.threshold for r in reps if np.isfinite(r.threshold)]
            aucs = [r.metrics.auc for r in reps if np.isfinite(r.metrics.auc)]
            tsss = [r.metrics.tss for r in reps if np.isfinite(r.metrics.tss)]
            boyces = [r.metrics.boyce for r in reps if np.isfinite(r.metrics.boyce)]
            if not preds:
                continue
            mean_raster = np.nanmean(np.stack(preds, axis=0), axis=0)
            per_algo_raster[algo] = mean_raster
            per_algo_threshold[algo] = float(np.mean(thresholds)) if thresholds else float("nan")
            per_algo_auc[algo] = float(np.mean(aucs)) if aucs else 0.0
            per_algo_tss[algo] = float(np.mean(tsss)) if tsss else 0.0

            cont_path = out_dir / f"suitability_{algo}.tif"
            bin_path = out_dir / f"suitability_{algo}_binary.tif"
            write_raster(cont_path, mean_raster, stack)
            write_raster(
                bin_path,
                apply_threshold_masked(mean_raster, per_algo_threshold[algo], nodata=255),
                stack,
                dtype="uint8",
                nodata=255,
            )
            output_files += [str(cont_path), str(bin_path)]

            algo_label = algorithm_long_name(algo)
            cont_map = plot_raster_map(
                mean_raster, out_dir / "plots" / f"map_suitability_{algo}.png",
                title=f"{algo_label}: suitability", cmap="Spectral",
            )
            bin_map = plot_raster_map(
                apply_threshold(mean_raster, per_algo_threshold[algo]),
                out_dir / "plots" / f"map_suitability_{algo}_binary.png",
                title=f"{algo_label}: binary (max-TSS)", categorical=True, cmap="Greens",
                mask=~np.isfinite(mean_raster),
            )
            map_files[algo] = {"continuous": cont_map.name, "binary": bin_map.name}

            curves_by_algo[algo] = [r.curves for r in reps]

            metrics_summary.append({
                "algorithm": algorithm_long_name(algo),
                "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
                "auc_sd": float(np.std(aucs)) if aucs else float("nan"),
                "tss_mean": float(np.mean(tsss)) if tsss else float("nan"),
                "tss_sd": float(np.std(tsss)) if tsss else float("nan"),
                "boyce_mean": float(np.mean(boyces)) if boyces else float("nan"),
                "boyce_sd": float(np.std(boyces)) if boyces else float("nan"),
                "n_replicates": len(reps),
            })

            # Plots
            plot_response_curves(
                algorithm=algo,
                feature_names=kept_names,
                curves_per_replicate=[r.curves for r in reps],
                out_dir=out_dir / "plots",
            )
            plot_variable_importance(
                algorithm=algo,
                per_replicate_importance=[r.importances for r in reps],
                out_dir=out_dir / "plots",
            )

        # 7. Ensemble
        ensemble_raster = None
        ensemble_sd = None
        if per_algo_raster:
            self._progress("ensemble", 0.5, "Building ensemble")
            metric_map = per_algo_auc if self.cfg.ensemble.method == "weighted_auc" else per_algo_tss
            ensemble_raster, ensemble_sd = ensemble_predictions(
                per_algo_raster,
                metric_map,
                method=self.cfg.ensemble.method,
            )
            # per_algo_threshold[algo] can itself be NaN (every replicate of
            # that algorithm had a non-finite threshold) — matching the
            # finite-filtering pattern used for aucs/tsss/boyces above, so a
            # single failed algorithm doesn't NaN out the ensemble threshold
            # and silently collapse the binary raster to all-suitable/none.
            finite_thresholds = [t for t in per_algo_threshold.values() if np.isfinite(t)]
            ens_thr = float(np.mean(finite_thresholds)) if finite_thresholds else 0.5
            write_raster(out_dir / "ensemble_suitability.tif", ensemble_raster, stack)
            write_raster(
                out_dir / "ensemble_suitability_binary.tif",
                apply_threshold_masked(ensemble_raster, ens_thr, nodata=255),
                stack,
                dtype="uint8",
                nodata=255,
            )
            write_raster(out_dir / "ensemble_uncertainty_sd.tif", ensemble_sd, stack)
            output_files += [
                str(out_dir / "ensemble_suitability.tif"),
                str(out_dir / "ensemble_suitability_binary.tif"),
                str(out_dir / "ensemble_uncertainty_sd.tif"),
            ]

            ens_cont_map = plot_raster_map(
                ensemble_raster, out_dir / "plots" / "map_ensemble_suitability.png",
                title="Ensemble: suitability", cmap="Spectral",
            )
            ens_bin_map = plot_raster_map(
                apply_threshold(ensemble_raster, ens_thr),
                out_dir / "plots" / "map_ensemble_suitability_binary.png",
                title="Ensemble: binary (mean max-TSS)", categorical=True, cmap="Greens",
                mask=~np.isfinite(ensemble_raster),
            )
            ens_sd_map = plot_raster_map(
                ensemble_sd, out_dir / "plots" / "map_ensemble_uncertainty_sd.png",
                title="Ensemble: across-model uncertainty (SD)", cmap="magma",
            )
            map_files["ensemble"] = {
                "continuous": ens_cont_map.name,
                "binary": ens_bin_map.name,
                "uncertainty": ens_sd_map.name,
            }

            if curves_by_algo:
                plot_ensemble_response_curves(
                    feature_names=kept_names,
                    curves_by_algo=curves_by_algo,
                    algo_labels={a: a.upper() for a in curves_by_algo},
                    per_algo_metric=metric_map,
                    ensemble_method=self.cfg.ensemble.method,
                    out_dir=out_dir / "plots",
                )

            # Fraction of the map where algorithms disagree strongly, and the
            # share classified suitable at the threshold, for interpretation.
            if ensemble_sd is not None:
                sd_valid = ensemble_sd[np.isfinite(ensemble_sd)]
                if sd_valid.size:
                    uncertainty_hi_frac = float(np.mean(sd_valid > 0.20))
            finite_ens = np.isfinite(ensemble_raster)
            if finite_ens.any():
                ens_binary = apply_threshold(ensemble_raster, ens_thr)
                suitable_frac = float(np.mean(ens_binary[finite_ens] == 1))

            # Cross-validated ensemble scores, and (biomod2-style) permutation
            # importance of the ensemble prediction. Only meaningful with 2+
            # algorithms; with one the "ensemble" is just that algorithm.
            ens_weights = compute_weights(
                list(per_algo_raster.keys()), metric_map, self.cfg.ensemble.method
            )
            if len(per_algo_raster) >= 2:
                ensemble_rep_metrics = self._ensemble_cv_metrics(
                    replicate_results, list(per_algo_raster.keys()), ens_weights, n_rep
                )
                if ensemble_rep_metrics:
                    e_auc = [m.auc for _, m in ensemble_rep_metrics if np.isfinite(m.auc)]
                    e_tss = [m.tss for _, m in ensemble_rep_metrics if np.isfinite(m.tss)]
                    e_boyce = [m.boyce for _, m in ensemble_rep_metrics if np.isfinite(m.boyce)]
                    metrics_summary.insert(0, {
                        "algorithm": "Ensemble",
                        "auc_mean": float(np.mean(e_auc)) if e_auc else float("nan"),
                        "auc_sd": float(np.std(e_auc)) if e_auc else float("nan"),
                        "tss_mean": float(np.mean(e_tss)) if e_tss else float("nan"),
                        "tss_sd": float(np.std(e_tss)) if e_tss else float("nan"),
                        "boyce_mean": float(np.mean(e_boyce)) if e_boyce else float("nan"),
                        "boyce_sd": float(np.std(e_boyce)) if e_boyce else float("nan"),
                        "n_replicates": len(ensemble_rep_metrics),
                    })
                    ensemble_cv_auc = float(np.mean(e_auc)) if e_auc else None

                models_by_algo = {
                    a: [r.model for r in replicate_results
                        if r.algorithm == a and r.error is None and r.model is not None]
                    for a in per_algo_raster
                }
                models_by_algo = {a: ms for a, ms in models_by_algo.items() if ms}
                if len(models_by_algo) >= 2:
                    self._progress("ensemble", 0.85, "Ensemble variable importance")
                    ens_imp = ensemble_permutation_importance(
                        models_by_algo, ens_weights, X_kept, y, kept_names,
                        n_repeats=3, rng=np.random.default_rng(self.cfg.random_seed + 777),
                    )
                    plot_variable_importance(
                        algorithm="Ensemble",
                        per_replicate_importance=ens_imp,
                        out_dir=out_dir / "plots",
                    )
                    ensemble_importance_name = "plots/importance_Ensemble.png"

                    # Train-set ensemble AUC, compared to the cross-validated AUC
                    # above, drives the overfitting note.
                    from sklearn.metrics import roc_auc_score

                    algos_ok = list(models_by_algo.keys())
                    w_sum = sum(max(ens_weights.get(a, 0.0), 0.0) for a in algos_ok)
                    ens_train = np.zeros(len(y), dtype=float)
                    for a in algos_ok:
                        wa = (max(ens_weights.get(a, 0.0), 0.0) / w_sum) if w_sum > 0 else 1.0 / len(algos_ok)
                        ens_train += wa * np.mean([m.predict_proba(X_kept) for m in models_by_algo[a]], axis=0)
                    ensemble_train_auc = float(roc_auc_score(y, ens_train))

        # 8. Projection to second raster stack + MESS/MOP
        proj_files: list[str] = []
        proj_map_files: dict[str, str] = {}
        if self.cfg.rasters.projection_paths:
            self._progress("project", 0.0, "Projecting to secondary raster stack")
            if session.proj_stack is None:
                session.proj_stack = load_stack(self.cfg.rasters.projection_paths)
            proj_stack = session.proj_stack
            validate_matching_bands(stack, proj_stack)
            algo_reps = list(_reps_by_algo(replicate_results))
            n_proj_algo = max(1, len(algo_reps))
            for p_i, (algo, reps_data) in enumerate(algo_reps):
                self._progress(
                    "project",
                    0.05 + 0.7 * (p_i / n_proj_algo),
                    f"Projecting [{algo}] ({p_i + 1}/{n_proj_algo})",
                )
                proj_preds = []
                for rep in reps_data:
                    if rep.error is not None or rep.model_path is None:
                        continue
                    from .io.outputs import load_model

                    model = load_model(rep.model_path)
                    proj_preds.append(
                        predict_raster(model, proj_stack, kept_feature_idx=kept_idx)
                    )
                if proj_preds:
                    mean_proj = np.nanmean(np.stack(proj_preds, axis=0), axis=0)
                    p = out_dir / f"projection_{algo}.tif"
                    write_raster(p, mean_proj, proj_stack)
                    proj_files.append(str(p))
                    proj_map = plot_raster_map(
                        mean_proj, out_dir / "plots" / f"map_projection_{algo}.png",
                        title=f"{algorithm_long_name(algo)}: projected suitability", cmap="viridis",
                    )
                    proj_map_files[algo] = proj_map.name
            self._progress("project", 0.78, "Computing MESS (novel environment detection)")
            mess_arr = mess(proj_stack, X_kept, kept_feature_idx=kept_idx)
            mess_valid = mess_arr[np.isfinite(mess_arr)]
            if mess_valid.size:
                extrap_frac = float(np.mean(mess_valid < 0.0))
            self._progress("project", 0.88, "Computing MOP (dissimilarity to training data)")
            mop_arr = mop(proj_stack, X_kept, kept_feature_idx=kept_idx)
            write_raster(out_dir / "projection_mess.tif", mess_arr, proj_stack)
            write_raster(out_dir / "projection_mop.tif", mop_arr, proj_stack)
            proj_files += [
                str(out_dir / "projection_mess.tif"),
                str(out_dir / "projection_mop.tif"),
            ]
            output_files += proj_files

            mess_abs_max = float(np.nanmax(np.abs(mess_arr))) if np.isfinite(mess_arr).any() else 1.0
            mess_map = plot_raster_map(
                mess_arr, out_dir / "plots" / "map_projection_mess.png",
                title="MESS (negative = novel environment)", cmap="RdBu",
                vmin=-mess_abs_max, vmax=mess_abs_max,
            )
            mop_map = plot_raster_map(
                mop_arr, out_dir / "plots" / "map_projection_mop.png",
                title="MOP (dissimilarity to training data)", cmap="viridis",
            )
            proj_map_files["mess"] = mess_map.name
            proj_map_files["mop"] = mop_map.name
            self._progress("project", 1.0, "Projection complete")

        # 9. Save config + reports
        self._progress("report", 0.0, "Writing report")
        cfg_path = out_dir / "run_config.json"
        self.cfg.to_json(cfg_path)
        output_files.append(str(cfg_path))

        # The exact per-algorithm hyperparameters build_model() used for
        # this run — not part of SDMConfig (those are pipeline/data
        # settings, not model internals), so written as its own file and
        # surfaced separately in the report.
        model_config = build_model_config(
            self.cfg.modeling.algorithms,
            n_presence=int((y == 1).sum()),
            overrides=self.cfg.modeling.hyperparameters,
        )
        model_config_path = out_dir / "model_hyperparameters.json"
        save_json(model_config_path, model_config)
        output_files.append(str(model_config_path))

        save_json(out_dir / "vif_report.json", vif_report.as_dict())
        output_files.append(str(out_dir / "vif_report.json"))
        per_replicate_json = [
            {
                "algorithm": r.algorithm,
                "replicate": r.replicate,
                "metrics": r.metrics.as_dict(),
                "error": r.error,
            }
            for r in replicate_results
        ]
        per_replicate_json += [
            {
                "algorithm": "Ensemble",
                "replicate": rep_i,
                "metrics": m.as_dict(),
                "error": None,
            }
            for rep_i, m in ensemble_rep_metrics
        ]
        save_json(out_dir / "metrics_per_replicate.json", per_replicate_json)
        output_files.append(str(out_dir / "metrics_per_replicate.json"))

        report_path = None
        if self.cfg.output.write_html_report:
            plots_dir = out_dir / "plots"
            response_files: dict[str, list[dict]] = {}
            var_import_files: dict[str, str] = {}
            if ensemble_importance_name and (plots_dir / "importance_Ensemble.png").exists():
                var_import_files["Ensemble"] = ensemble_importance_name

            def _curve_entries(pngs: list[str], token: str) -> list[dict]:
                # Filenames are response_{token}_{variable}.png; expose the
                # variable name for the caption (the {variable} part can itself
                # contain underscores, so strip the known prefix rather than
                # splitting) and keep the report-relative path for the image.
                prefix = f"response_{token}_"
                out = []
                for name in pngs:
                    stem = name[:-4] if name.endswith(".png") else name
                    variable = stem[len(prefix):] if stem.startswith(prefix) else stem
                    out.append({"path": f"plots/{name}", "variable": variable})
                return out

            ens_response_pngs = sorted(str(p.name) for p in plots_dir.glob("response_ensemble_*.png"))
            if ens_response_pngs:
                response_files["Ensemble"] = _curve_entries(ens_response_pngs, "ensemble")
            for algo in self.cfg.modeling.algorithms:
                if algo not in per_algo_raster:
                    continue
                pngs = sorted(str(p.name) for p in plots_dir.glob(f"response_{algo}_*.png"))
                if pngs:
                    response_files[algo.upper()] = _curve_entries(pngs, algo)
                imp = plots_dir / f"importance_{algo}.png"
                if imp.exists():
                    var_import_files[algorithm_long_name(algo)] = f"plots/{imp.name}"

            output_maps: list[dict] = []
            for algo in self.cfg.modeling.algorithms:
                if algo not in map_files:
                    continue
                entry = {"label": algorithm_long_name(algo)}
                entry.update({k: f"plots/{v}" for k, v in map_files[algo].items()})
                output_maps.append(entry)
            if "ensemble" in map_files:
                entry = {"label": "Ensemble"}
                entry.update({k: f"plots/{v}" for k, v in map_files["ensemble"].items()})
                output_maps.append(entry)

            projection_maps: list[dict] = []
            for algo in self.cfg.modeling.algorithms:
                if algo in proj_map_files:
                    projection_maps.append({"label": algorithm_long_name(algo), "map": f"plots/{proj_map_files[algo]}"})
            if "mess" in proj_map_files:
                projection_maps.append({"label": "MESS", "map": f"plots/{proj_map_files['mess']}"})
            if "mop" in proj_map_files:
                projection_maps.append({"label": "MOP", "map": f"plots/{proj_map_files['mop']}"})

            # Fraction of input occurrence records lost to cleaning + thinning.
            cleaning_removed_frac = None
            n_raw = cleaning_rep.n_input if cleaning_rep is not None else None
            n_final_occ = cleaning_rep.n_output if cleaning_rep is not None else None
            if thinning_rep is not None:
                if n_raw is None:
                    n_raw = thinning_rep.n_input
                n_final_occ = thinning_rep.n_output
            if n_raw and n_final_occ is not None and n_raw > 0:
                cleaning_removed_frac = 1.0 - (n_final_occ / n_raw)

            interpretation = build_interpretation(
                metrics_summary=metrics_summary,
                n_presence=int((y == 1).sum()),
                split_method=self.cfg.split.method,
                uncertainty_hi_frac=uncertainty_hi_frac,
                extrap_frac=extrap_frac,
                has_maxent="maxent" in self.cfg.modeling.algorithms,
                presence_only=self.cfg.data_mode == "presence_only",
                suitable_frac=suitable_frac,
                ensemble_train_auc=ensemble_train_auc,
                ensemble_cv_auc=ensemble_cv_auc,
                vif=vif_report.as_dict(),
                cleaning_removed_frac=cleaning_removed_frac,
            )

            report_path = render_report(
                out_dir / "report.html",
                {
                    "run_id": Path(self.cfg.output.directory).name,
                    "interpretation": [n.as_dict() for n in interpretation],
                    "config_json": json.dumps(self.cfg.to_dict(), indent=2),
                    "config_dict": self.cfg.to_dict(),
                    "model_config": {
                        algorithm_long_name(algo): params
                        for algo, params in model_config.items()
                    },
                    "cleaning": (cleaning_rep.as_dict() if cleaning_rep else {"n_input": len(occ.x), "n_output": len(occ.x), "dropped": {}}),
                    "thinning": (thinning_rep.as_dict() if thinning_rep else None),
                    "vif": vif_report.as_dict(),
                    "background": {
                        "method": self.cfg.background.method,
                        "n_drawn": int((y == 0).sum()),
                        "buffer_distance": (
                            self.cfg.background.buffer_distance
                            if self.cfg.background.method == "buffered"
                            else None
                        ),
                    } if self.cfg.data_mode == "presence_only" else None,
                    "split": {
                        "method": self.cfg.split.method,
                        "n_folds": self.cfg.split.k,
                        "plan": None,
                    },
                    "metrics_summary": metrics_summary,
                    "failed_runs": failed,
                    "response_curves": response_files,
                    "variable_importance": var_import_files,
                    "output_maps": output_maps,
                    "projection_maps": projection_maps,
                    "output_files": [str(Path(p).name) for p in output_files],
                },
            )
            output_files.append(str(report_path))

        self._progress("report", 1.0, "Done")
        return RunResult(
            output_dir=str(out_dir),
            metrics_summary=metrics_summary,
            failed_runs=failed,
            output_files=output_files,
            report_path=str(report_path) if report_path else None,
        )

    # ----- helpers -----

    def _ensemble_cv_metrics(
        self,
        replicate_results: list[ReplicateResult],
        algos: list[str],
        weights: dict[str, float],
        n_rep: int,
    ) -> list[tuple[int, EvaluationResult]]:
        """Cross-validated ensemble scores per replicate. Within a replicate all
        algorithms share the same folds, so their pooled held-out predictions
        are aligned and can be combined with the ensemble weights and evaluated.
        """
        out: list[tuple[int, EvaluationResult]] = []
        for rep_i in range(n_rep):
            scores: dict[str, np.ndarray] = {}
            y_true_ref: np.ndarray | None = None
            for algo in algos:
                r = next(
                    (rr for rr in replicate_results
                     if rr.algorithm == algo and rr.replicate == rep_i
                     and rr.error is None and rr.y_score is not None),
                    None,
                )
                if r is None:
                    continue
                scores[algo] = np.asarray(r.y_score, dtype=float)
                y_true_ref = r.y_true
            if not scores or y_true_ref is None:
                continue
            w = {a: max(float(weights.get(a, 0.0)), 0.0) for a in scores}
            w_sum = sum(w.values())
            if w_sum <= 0:
                w = {a: 1.0 for a in scores}
                w_sum = float(len(scores))
            ens = np.zeros(len(y_true_ref), dtype=float)
            for a, s in scores.items():
                ens += (w[a] / w_sum) * s
            out.append((rep_i, evaluate(y_true_ref, ens)))
        return out

    def _run_one(
        self,
        *,
        algo: str,
        replicate: int,
        X_kept: np.ndarray,
        y: np.ndarray,
        px: np.ndarray,
        py: np.ndarray,
        kept_names: list[str],
        stack: RasterStack,
        kept_idx: list[int],
        rng: np.random.Generator,
        out_dir: Path,
    ) -> ReplicateResult:
        rep_rng = np.random.default_rng(self.cfg.random_seed + replicate * 101)

        # Fold generation — always freshly generated per replicate (never
        # cached/reused from a wizard-page preview) so each replicate sees an
        # independent split, which is the entire point of replicating.
        folds, _plan, _fold_id = make_folds(self.cfg, X_kept, y, px, py, stack, rep_rng)

        # Pool held-out predictions across folds
        algo_overrides = self.cfg.modeling.hyperparameters.get(algo, {})
        y_true_pool: list[np.ndarray] = []
        y_score_pool: list[np.ndarray] = []
        for train_idx, test_idx in folds:
            model = build_model(
                algo, random_state=self.cfg.random_seed + replicate * 17, **algo_overrides
            )
            model.set_feature_names(kept_names)
            model.fit(X_kept[train_idx], y[train_idx])
            y_true_pool.append(y[test_idx])
            y_score_pool.append(model.predict_proba(X_kept[test_idx]))
        y_true = np.concatenate(y_true_pool)
        y_score = np.concatenate(y_score_pool)
        metrics = evaluate(y_true, y_score)
        threshold = maxtss_threshold(y_true, y_score)

        # Final fit on ALL points for prediction, curves, importance
        final = build_model(
            algo, random_state=self.cfg.random_seed + replicate * 17, **algo_overrides
        )
        final.set_feature_names(kept_names)
        final.fit(X_kept, y)

        importances = final.permutation_importance(X_kept, y, n_repeats=3, rng=rep_rng)
        curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for i, name in enumerate(kept_names):
            grid, preds = final.partial_dependence(X_kept, i)
            curves[name] = (grid, preds)

        prediction = predict_raster(final, stack, kept_feature_idx=kept_idx)

        model_path = None
        if self.cfg.output.save_models:
            model_path = str(out_dir / "models" / f"{algo}_rep{replicate}.joblib")
            save_model(model_path, final)

        return ReplicateResult(
            algorithm=algo,
            replicate=replicate,
            metrics=metrics,
            importances=importances,
            curves=curves,
            prediction=prediction,
            threshold=threshold,
            model_path=model_path,
            y_true=y_true,
            y_score=y_score,
            model=final,
        )

    def _progress(self, stage: str, fraction: float, message: str) -> None:
        try:
            self._progress_cb(stage, float(fraction), message)
        except Exception:
            pass


def _reps_by_algo(reps: list[ReplicateResult]):
    seen: dict[str, list[ReplicateResult]] = {}
    for r in reps:
        seen.setdefault(r.algorithm, []).append(r)
    return seen.items()
