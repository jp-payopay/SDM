# Changelog

## [1.2.1] - 2026-09-21

### Added

**CSV output**

Every number the report and the figures show is now also written as a CSV, so
results can be re-analysed or re-plotted without going back through the plugin.
All are long-format and land in the output directory alongside the rasters.

- `metrics_summary.csv` and `metrics_detail.csv` — the report's performance
  table, and the scores behind it per algorithm, replicate and cross-validation
  fold.
- `response_curves.csv` and `response_curves_summary.csv` — the response-curve
  values: every replicate's curve, each algorithm's mean and SD band, and the
  weighted ensemble curve.
- `variable_importance.csv` and `variable_importance_summary.csv` — the
  per-replicate permutation importances and the plotted bars, ranked.

Scoring each CV fold separately is new rather than a reformat: a replicate's
score can now be checked against the folds it came from, which matters most
under spatial-block CV where each fold is a different region. The ensemble is
scored the same way and appears as a row like any other algorithm.

### Fixed

- The `Plugins -> SDM` submenu now shows the plugin icon. The icon was only
  ever set on the action inside the submenu; QGIS builds the submenu itself and
  gives it no icon, so the entry read as bare text next to plugins that build
  their own submenu.

## [1.1.0] - 2026-08-17

A feature release built around two things that used to stop a run dead:
predictor rasters that do not line up, and having only one useful way to place
pseudo-absences. Both now have a way through, and every page that offers a
choice explains what the options mean.

Existing `run_config.json` files from 1.0.1 still load and rerun unchanged. See
[Upgrading](#upgrading-from-101) for the details.

### Added

**Fixing misaligned predictor rasters**

- When the predictor or projection rasters do not share a CRS, extent,
  resolution and pixel grid, the page now replaces its statistics table with a
  per-layer breakdown: data type, CRS, size in pixels, resolution, NoData and
  extent, with every value that differs from the first raster highlighted. The
  old behaviour reported only the first offending file, which meant working out
  the problem from an error message.
- The alignment check now inspects every raster rather than stopping at the
  first mismatch, and names what differs in the terms you would use yourself:
  CRS, resolution, extent, or pixel alignment. A raster offset by a whole number
  of pixels is reported as an extent difference; only a fractional offset or a
  rotation counts as a misalignment, since that is the one that cannot be fixed
  by clipping.
- A **Fix predictor layers** button (and **Fix projection layers** on the
  projection page) resamples every listed raster onto one common grid. It
  becomes available only after a validation run has actually found a mismatch.
- The fix dialog lets you choose the target grid before anything is written:
  reference layer, target CRS, extent (intersection, union, the reference
  layer, or custom) and resolution (coarsest, finest, the reference layer, or
  custom), with a live preview of the resulting pixel dimensions and extent.
  Extents and resolutions are re-expressed in the target CRS, so "coarsest"
  stays meaningful when the target CRS changes the units.
- Per-layer output settings in the same dialog: output data type, NoData value
  and resampling method, starting at float32 / -9999 / bilinear. The dialog
  explains when to reach for int or uint instead, and refuses to proceed with a
  NoData value the chosen type cannot hold.
- Layers carrying no CRS at all get an "assume this CRS" picker rather than a
  flat refusal, which covers the common bare `.asc` case.
- Resampling runs in the background with a progress bar, writes to a folder you
  pick, confirms before overwriting existing files, and refuses outright to
  write over the rasters it is reading.

**Background and pseudo-absence points**

- **Ratio to presences**: places points at random, as the random method does,
  but scales the count to the presence total. At 4 per presence, 50 records give
  200 pseudo-absences and 300 give 1,200, which keeps the balance between the
  two classes steady across species.
- **SRE (Surface Range Envelope)**: builds a rectilinear envelope around the
  conditions the species was recorded in, one interval per predictor, and draws
  pseudo-absences only from outside it. Suited to data where most of the
  species' environmental space has already been sampled. When the records span
  every predictor and there is nothing left outside the envelope, the run says
  so instead of returning nothing.

**Explanations on the pages that offer a choice**

- Selecting an option on the background points, cross-validation, ensemble and
  algorithms pages now shows what it does, when it is the right choice, and the
  values people normally use. On the algorithms page, which is multi-select, the
  description follows whichever algorithm the pointer or keyboard focus is on,
  so you can read about one without having to select it first.

### Changed

- The buffered background is now a **disk**, taking a minimum and a maximum
  distance from the *nearest* presence instead of a single buffer radius. The
  minimum keeps points out of the immediate surroundings of a record, where an
  apparent absence is usually just somewhere nobody has surveyed. A maximum of 0
  means no upper limit. An empty ring is reported rather than silently yielding
  no points.
- Cross-validation methods are reordered with k-fold first. Spatial block
  remains the selected default, since species records are almost always
  spatially clustered; only the ordering changed.
- Prose explaining block size and block shape is shown only while spatial block
  is selected.
- Every algorithm's fitted defaults are unchanged, but the comments justifying
  them no longer attribute them to other SDM tools. The reasoning and the
  academic citations remain.
- MaxEnt is labelled "MaxEnt" rather than "MaxEnt (elapid)" on the algorithms
  page, in the report and in plot titles.
- Wording pass across every page: no en or em dashes in anything the user reads,
  and no third-party package names in the descriptions.
- README installation instructions are split into two routes, the QGIS Official
  Plugin Repository and a symlinked checkout of this repository.
- `run_config.json` and the HTML report record the new background settings.
- The plugin version is now defined once in `core/config.py` as `SDM_VERSION`
  and read from there by the config default and the report footer.
  `scripts/build_zip.py` refuses to build when it disagrees with
  `metadata.txt`.

### Removed

- `BackgroundConfig.buffer_distance`, replaced by `min_distance` and
  `max_distance`.
- `core.background.buffered.sample_buffered`, replaced by
  `core.background.disk.sample_disk`.

### Upgrading from 1.0.1

Nothing to do. A `run_config.json` written by 1.0.1 loads unchanged: the old
`"buffered"` method maps to `"disk"`, and its `buffer_distance` becomes
`max_distance` with no inner radius, which is exactly what it meant before, so a
rerun draws the same points.

If you drive the `core/` package directly from Python rather than through the
wizard, note the two removals above. The documented entry points, `SDMConfig`
and `Pipeline`, are unchanged.

### Internal

- New modules: `core/io/align.py` (target grids and warping),
  `core/background/disk.py`, `core/background/sre.py`, `ui/raster_stage.py`
  (shared behaviour for the two raster pages) and
  `ui/widgets/fix_rasters_dialog.py`.
- `core/io/rasters.py` gained `RasterProfile`, `describe_profiles`,
  `diagnose_alignment`, `check_unique_stems` and `build_stack`. `load_stack` is
  now assembled from those, so the wizard's verdict and the pipeline's cannot
  drift apart.
- `StagePageMixin.run_stage_async` accepts an optional progress handler,
  delivered to the GUI thread through a signal.
- Test suite grew from 115 to 158, adding `tests/test_align.py`,
  `tests/test_background.py` and `tests/test_algorithm_help.py`.

## [1.0.1]

Initial public release.
