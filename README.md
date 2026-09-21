<img width="1400" height="350" alt="sdm_plugin_banner" src="https://github.com/user-attachments/assets/f10a156e-5300-4b0c-a41e-e4ac8e33debe" />

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

Requires QGIS 4.0 or newer. Pick whichever of the two options fits you: the
plugin repository if you just want to use SDM, the repository checkout if you
want to follow or edit the code.

### Option 1: from the QGIS Official Plugin Repository (recommended)

The normal route. QGIS tracks the installed version for you and offers upgrades
as they are published.

**Inside QGIS.** Open *Plugins → Manage and Install Plugins…*, go to the **All**
tab, type `SDM` in the search box, select it and click **Install Plugin**. Later
releases show up in the same dialog under **Upgradeable**.

**From the plugins web portal.** Search for `SDM` at
[plugins.qgis.org](https://plugins.qgis.org/plugins/) and download the `.zip`.
Then, in QGIS, open *Plugins → Manage and Install Plugins… → Install from ZIP*,
point it at the file you downloaded and click **Install Plugin**. This is the
one to use on a machine with no internet access inside QGIS itself, since the
download and the install are separate steps.

Either way, enable **SDM** in the **Installed** tab if it is not already ticked.

### Option 2: symlink this repository (development, or unreleased changes)

Use this to run the code as it stands here rather than the last published
release, or if you intend to change it.

1. Clone the repository somewhere you are happy to keep it:

   ```
   git clone https://github.com/jp-payopay/SDM.git
   ```

2. Symlink the checkout into your QGIS 4 plugins directory as `sdm_plugin`:

   - macOS: `~/Library/Application Support/QGIS/QGIS4/profiles/default/python/plugins/sdm_plugin`
   - Linux: `~/.local/share/QGIS/QGIS4/profiles/default/python/plugins/sdm_plugin`
   - Windows: `%APPDATA%\QGIS\QGIS4\profiles\default\python\plugins\sdm_plugin`

   macOS and Linux:

   ```
   ln -s "$PWD/SDM" ~/.local/share/QGIS/QGIS4/profiles/default/python/plugins/sdm_plugin
   ```

   Windows, from an Administrator prompt (or with Developer Mode on):

   ```
   mklink /D "%APPDATA%\QGIS\QGIS4\profiles\default\python\plugins\sdm_plugin" "C:\path\to\SDM"
   ```

   A plain copy works too if symlinks are awkward on your system, but then you
   have to re-copy after every `git pull`. The link name should be `sdm_plugin`,
   matching the folder name inside the released zip, so that installing both
   ways does not leave you with two separately registered copies of the plugin.

3. Restart QGIS, then enable **SDM** in *Plugins → Manage and Install Plugins…*.

With a symlink, a `git pull` updates the installed plugin in place. To pick up
edits without restarting QGIS, install the **Plugin Reloader** plugin and point
it at `sdm_plugin`.

### Python dependencies (either option)

Which Python packages QGIS already bundles varies by platform and installer, so
do not assume. If anything required is missing from QGIS's Python environment,
the plugin opens a dialog with a one-click **Install missing package(s)** button
that runs `pip install` against QGIS's own interpreter and streams the progress
live. You can also install everything ahead of time yourself:

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

### CSV tables

Every number behind the report's tables and the figures is also written as a
CSV, so results can be re-analysed, re-plotted or pasted into a manuscript
without going back through the plugin. All of them are long-format (one row per
observation), and `algorithm` holds the short code (`rf`) next to
`algorithm_label` with the full name (`Random Forest`); the ensemble appears as
`ensemble`. Replicates and folds are numbered from 1.

| File | One row per | Holds |
| --- | --- | --- |
| `metrics_summary.csv` | algorithm | AUC, TSS and Boyce as mean ± SD across replicates — the report's performance table |
| `metrics_detail.csv` | algorithm × replicate, and algorithm × replicate × fold | every score the run computed, at both grains, with a `level` column (`replicate` or `fold`) saying which |
| `response_curves.csv` | algorithm × replicate × variable × grid point | every individual response curve — the thin grey lines on a response-curve plot |
| `response_curves_summary.csv` | series × variable × grid point | the bold lines: each algorithm's mean curve and SD band, plus the weighted ensemble curve and the weight each algorithm carries in it |
| `variable_importance.csv` | algorithm × replicate × variable | raw permutation importance (drop in AUC when that predictor is shuffled) |
| `variable_importance_summary.csv` | algorithm × variable | the plotted bars: mean importance, SD across replicates, and rank (1 = most important) |

`metrics_detail.csv` holds both grains in one file: a `replicate` row is
scored on the held-out predictions pooled across that replicate's folds (the
numbers `metrics_summary.csv` averages, and the row carrying the error text if
a replicate failed), and the `fold` rows following it are each fold scored on
its own held-out points alone — useful for seeing whether a replicate's score
rests on folds that agree, which matters most under spatial-block CV where each
fold is a different region. Filtering on `level` gives back either table on its
own. Columns that apply to one level only are blank on the other: `fold`,
`n_blocks` and `n_train` on a pooled row (its folds each trained on a different
subset, so there is no single training size), and `error` on a fold row.

Two notes on reading them. Empty cells mean missing, not zero — a metric is
blank when it could not be computed (for example Boyce on a fold with too few
presences), which pandas and R both read back as `NA`. And because response
curves and importance are measured on each replicate's final fit over all the
points, a deterministic algorithm such as logistic regression produces the same
curve in every replicate, so its SD column is legitimately 0; the variation you
see for random forest or XGBoost is the model's own stochasticity.

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

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for what changed in each release.

## License

GPLv3-or-later. See [LICENSE](LICENSE).
