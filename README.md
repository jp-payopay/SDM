# SDM

A guided QGIS 4 plugin for species distribution modeling, walking you through
data loading, cleaning, predictor selection, cross-validation, modeling,
ensembling, and reporting — with a live map preview at every step and no
blind final run.

## Features

- Presence-only and presence/absence workflows
- CSV or vector occurrence input, with a native QGIS CRS picker (the same
  widget used in Processing dialogs) instead of typing an EPSG code by hand
- Automatic coordinate cleaning + optional spatial thinning at raster resolution
- Predictor rasters must share CRS / extent / grid (validated on load)
- Random or buffered background points (user-set count and buffer)
- Stepwise VIF (Variance Inflation Factor) predictor selection for
  multicollinearity, with a configurable cutoff (default 10) — or skip it
  entirely and keep every predictor, if you'd rather handle collinearity
  yourself
- Split strategies: random hold-out, k-fold, or spatial-block CV
  (Cross-Validation), with auto block size (empirical variogram) and a
  choice of **square or hexagonal** block tessellation
- Nine algorithms: LR, GAM, RF, GBM, XGBoost, SVM, MLP, MaxEnt (elapid), ENFA
- Per-algorithm hyperparameters are editable (dropdowns for fixed-choice
  parameters like SVM's kernel, a checkbox picker for MaxEnt's feature
  classes) via "View model configuration…", with a fixed-choice dropdown for
  string parameters so a typo can't produce an invalid value
- Configurable replicated runs
- Skip-and-continue on failed model fits
- Metrics: AUC, TSS, Boyce (CBI) — mean ± SD across replicates
- Binary rasters via max-TSS threshold
- Response curves (per-replicate overlay + mean ± SD band) and permutation importance
- Ensemble: unweighted mean, weighted by AUC, or weighted by TSS + across-model SD uncertainty map
- Optional projection to a second raster stack with MESS and MOP extrapolation flags
- Timestamped run log and total runtime, shown live in the wizard and recorded in the report
- HTML report bundling metrics, plots, settings, and runtime
- `run_config.json` for reproducible reruns; joblib-serialized fitted models
  (joblib/pickle can execute arbitrary code on load — only load `.joblib`
  model files from a run you trust). Note: Random Forest fits in parallel
  (`n_jobs=-1`), so a rerun from the same `run_config.json` can differ from
  the original at the floating-point level for RF — this is usually
  invisible, but the Boyce (CBI) metric's binning can occasionally shift by
  a few hundredths between two runs of an RF-containing config as a result.
  AUC and TSS are unaffected.

## Installation

1. Copy or symlink this folder into your QGIS 4 plugins directory:
   - macOS: `~/Library/Application Support/QGIS/QGIS4/profiles/default/python/plugins/sdm_plugin`
   - Linux: `~/.local/share/QGIS/QGIS4/profiles/default/python/plugins/sdm_plugin`
   - Windows: `%APPDATA%\QGIS\QGIS4\profiles\default\python\plugins\sdm_plugin`
2. Restart QGIS, then enable **SDM** in *Plugins → Manage and Install Plugins*.
3. Launch the plugin. Which packages QGIS already bundles varies by platform/installer, so don't assume — if anything required is missing from QGIS's Python environment, a dialog offers a one-click **Install missing package(s)** button that runs `pip install` against QGIS's own interpreter and streams progress live. You can also install everything ahead of time yourself:

   ```
   pip install -r deps/requirements.txt
   ```

## Usage

Launch from *Plugins → SDM → Run SDM…*, the toolbar, or the SDM dock panel
(*View → Panels → SDM*). The wizard walks you through:

1. Load occurrences
2. Load predictor rasters
3. Cleaning options
4. Background points (presence-only mode)
5. Predictor selection (stepwise VIF, or keep all predictors)
6. Cross-validation strategy (including block shape, for spatial block)
7. Algorithms + replicates
8. Optional projection stack
9. Ensemble method
10. Output directory
11. Run
12. Summary

Outputs land in the chosen directory: continuous + binary suitability rasters per algorithm and per ensemble, response curve plots, variable importance plots, `metrics_per_replicate.json`, `vif_report.json`, `model_hyperparameters.json`, `run_config.json`, and `report.html`.

## Development

The `core/` package is Qt-free and can be driven from plain Python or pytest without QGIS:

```python
from sdm_plugin.core.config import SDMConfig
from sdm_plugin.core.pipeline import Pipeline

cfg = SDMConfig.from_json("run_config.json")
result = Pipeline(cfg).run()
```

Run tests:

```
pytest tests/
```

Build a clean release zip (excludes tests, dev/example data, and caches —
see `scripts/build_zip.py` for the exact include list):

```
python scripts/build_zip.py
```

This writes `dist/sdm_plugin.zip`, ready to upload to the QGIS plugin
repository or attach to a GitHub release.

## License

GPLv3-or-later. See [LICENSE](LICENSE).
