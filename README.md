<img width="128" height="128" alt="icon" src="https://github.com/user-attachments/assets/0e2ae2f4-cffe-4584-bf3b-633e2c5621e7" />
# SDM

A guided QGIS 4 plugin for species distribution modeling.

## Features

- Presence-only and presence/absence workflows
- CSV or vector occurrence input
- Automatic coordinate cleaning + optional spatial thinning at raster resolution
- Predictor rasters must share CRS / extent / grid (validated on load)
- Random or buffered background points (user-set count and buffer)
- Stepwise VIF predictor selection (default cutoff 10, configurable)
- Split strategies: random hold-out, k-fold, spatial-block CV (auto block size via empirical variogram)
- Nine algorithms: LR, GAM, RF, GBM, XGBoost, SVM, MLP, MaxEnt (elapid), ENFA
- Configurable replicated runs
- Skip-and-continue on failed model fits
- Metrics: AUC, TSS, Boyce (CBI) — mean ± SD across replicates
- Binary rasters via max-TSS threshold
- Response curves (per-replicate overlay + mean ± SD band) and permutation importance
- Ensemble: unweighted mean, weighted by AUC, or weighted by TSS + across-model SD uncertainty map
- Optional projection to a second raster stack with MESS and MOP extrapolation flags
- HTML report bundling metrics, plots, and settings
- `run_config.json` for reproducible reruns; joblib-serialized fitted models
  (joblib/pickle can execute arbitrary code on load — only load `.joblib`
  model files from a run you trust)

## Installation

1. Copy or symlink this folder into your QGIS 4 plugins directory:
   - macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/sdm_plugin`
   - Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/sdm_plugin`
   - Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\sdm_plugin`
2. Restart QGIS, then enable **SDM** in *Plugins → Manage and Install Plugins*.
3. Launch the plugin. Which packages QGIS already bundles varies by platform/installer, so don't assume — if anything required is missing from QGIS's Python environment, a dialog offers a one-click **Install missing package(s)** button that runs `pip install` against QGIS's own interpreter and streams progress live. You can also install everything ahead of time yourself:

   ```
   pip install -r deps/requirements.txt
   ```

## Usage

Launch from *Plugins → SDM → Run SDM…* or the toolbar. The wizard walks you through:

1. Load occurrences
2. Load predictor rasters
3. Cleaning options
4. Background points (presence-only mode)
5. Stepwise VIF cutoff
6. Cross-validation strategy
7. Algorithms + replicates
8. Optional projection stack
9. Ensemble method
10. Output directory
11. Run

Outputs land in the chosen directory: continuous + binary suitability rasters per algorithm and per ensemble, response curve plots, variable importance plots, `metrics_per_replicate.json`, `vif_report.json`, `run_config.json`, and `report.html`.

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

## License

GPLv3-or-later. See [LICENSE](LICENSE).
