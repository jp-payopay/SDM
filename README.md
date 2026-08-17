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
- Predictor rasters must share CRS / extent / resolution / grid (validated on
  load). When they don't, the page swaps its statistics table for a per-layer
  breakdown — data type, CRS, pixel size, resolution, NoData and extent, with
  the values that differ highlighted — and unlocks a **Fix predictor layers**
  button that resamples every layer onto one grid you choose: target CRS,
  extent (intersection / union / a reference layer / custom), resolution
  (coarsest / finest / a reference layer / custom), plus a per-layer output
  data type, NoData value and resampling method. Those three start at
  float32 / −9999 / bilinear for every layer, which is what continuous
  predictors want; the dialog explains when to reach for int or uint instead,
  and a categorical layer (land cover, soil class) should be switched to an
  integer type and nearest-neighbour resampling. New files are written to a
  folder you pick and the originals are left untouched. The optional
  projection stack gets the same treatment
- Four background / pseudo-absence strategies, each with a live description of
  when it applies: **random** across the raster extent; **ratio to presences**,
  placed at random but scaled to the presence total, so 50 records at 4 per
  presence give 200 pseudo-absences; **disk**, keeping only locations whose
  distance to the *nearest* presence falls between a minimum and a maximum (the
  inner radius keeps points out of the unsurveyed surroundings of a record, the
  outer one inside the accessible region); and **SRE**, drawing pseudo-absences
  from outside a rectilinear envelope of the conditions the species was recorded
  in, which suits data where most of the species' environmental space has already
  been sampled
- Stepwise VIF (Variance Inflation Factor) predictor selection for
  multicollinearity, with a configurable cutoff (default 10) — or skip it
  entirely and keep every predictor, if you'd rather handle collinearity
  yourself
- Split strategies: k-fold, random hold-out, or spatial-block CV
  (Cross-Validation), with auto block size (empirical variogram) and a
  choice of **square or hexagonal** block tessellation. Selecting a strategy
  swaps in a description of what it does, when it is the right choice, and
  the values people normally use (k = 5, an 80/20 hold-out, and so on);
  spatial block remains the default, since species records are almost always
  spatially clustered
- Nine algorithms: LR, GAM, RF, GBM, XGBoost, SVM, MLP, MaxEnt, ENFA.
  Pointing at one describes what it does, when it is the right choice, and what
  to watch out for (RF's weak extrapolation, SVM's cost on large samples, ENFA
  being presence-only, and so on) — so you can read about an algorithm without
  having to select it first
- Per-algorithm hyperparameters are editable (dropdowns for fixed-choice
  parameters like SVM's kernel, a checkbox picker for MaxEnt's feature
  classes) via "View model configuration…", with a fixed-choice dropdown for
  string parameters so a typo can't produce an invalid value
- Configurable replicated runs
- Skip-and-continue on failed model fits
- Metrics: AUC, TSS, Boyce (CBI) — mean ± SD across replicates
- Binary rasters via max-TSS threshold
- Response curves (per-replicate overlay + mean ± SD band) and permutation importance
- Ensemble: unweighted mean, weighted by AUC, or weighted by TSS + across-model
  SD uncertainty map. As on the cross-validation page, selecting a rule swaps in
  a description of what it does, when it is the right choice, and what to expect
  — including why TSS weighting separates algorithms far more sharply than AUC
  weighting does
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
