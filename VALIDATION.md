# Real-QGIS 4 validation checklist

Environment: macOS, QGIS at `/Applications/QGIS.app`, plugin already symlinked into `~/Library/Application Support/QGIS/QGIS4/profiles/default/python/plugins/sdm_plugin`.

## 1. Install the missing Python packages

Which packages QGIS already bundles is **not consistent across platforms/installers** — confirmed on the Windows standalone QGIS 4.0.1 installer, only `numpy`, `pandas`, `scipy`, `matplotlib`, `jinja2` came pre-installed; `rasterio`, `fiona`, `scikit-learn`, `joblib`, `pygam`, `xgboost`, and `elapid` were all missing. Don't assume — `_check_dependencies()` in `plugin.py` always checks the real imports and reports whatever's actually missing on the machine you're testing on.

**Preferred path — via the plugin's own dialog** (this is what §1b below exercises): enable the plugin per §2, then click the toolbar action before installing anything. `_check_dependencies()` reports whatever's missing and opens a `DependencyInstallDialog` with an **"Install missing package(s) (...)"** button. Click it and watch the log stream live `pip` output. On success it re-checks imports and launches the wizard directly if everything now imports cleanly; if a native extension still needs a restart, it says so.

Note: this runs `<python> -m pip install ...` against whatever `plugin.py`'s `_python_executable()` resolves to, **not** `sys.executable` directly — inside QGIS's embedded interpreter (confirmed on the Windows standalone installer) `sys.executable` is `qgis-bin.exe` itself, so spawning it directly launches a second QGIS instance instead of running pip. `_python_executable()` instead looks for `python.exe`/`python3.exe` under `sys.exec_prefix` (where QGIS's bundled CPython distribution actually lives, e.g. `...\QGIS 4.0.1\apps\Python312\`).

**Manual fallback** (if you want to install ahead of time, or the dialog's install fails): open the Python Console (**Plugins → Python Console**, or Ctrl+Alt+P) and run:

```python
from sdm_plugin.plugin import _python_executable
import subprocess
subprocess.run([_python_executable(), "-m", "pip", "install", "rasterio", "fiona", "scikit-learn", "joblib", "pygam", "xgboost", "elapid"])
```

(On Windows, do **not** use `sys.executable` here directly — see the note above; it will silently launch another QGIS window instead of installing anything.) Wait for it to finish (can be several minutes; xgboost is the largest download). Restart QGIS afterward so the new packages register cleanly.

**Verify** (Python Console, after restart):

```python
import rasterio, fiona, sklearn, joblib, pygam, xgboost, elapid
print("all imports ok")
```

## 1b. Dependency dialog + install button

With one or more required packages *not yet installed*, click the SDM toolbar action and confirm:
- A dialog titled "SDM — dependency problem" appears (not a plain OK-only message box) listing the missing packages, with an **Install missing package(s) (...)** button and a **Close** button.
- The manual-install command shown in the dialog's message text points at a real Python interpreter path (something like `...\apps\Python312\python.exe`), **not** `...\bin\qgis-bin.exe`.
- Clicking Install disables that button, reveals a log pane, and streams `pip`'s real output live (not just a static "please wait") — you should see download/build progress scroll by, not a frozen dialog, during a multi-minute build. **Clicking Install must not open a new QGIS window** — if it does, `_python_executable()` failed to find the bundled interpreter and fell back to `sys.executable`; report the exact QGIS install layout (output of the Python Console's `import sys; print(sys.exec_prefix, sys.base_exec_prefix, sys.executable)`) so the fallback search can be extended.
- On success, the dialog closes on its own and the wizard opens directly — no manual restart needed for this case, since these were newly-missing (never-imported) packages.
- Closing the dialog mid-install (before it finishes) doesn't crash QGIS.
- Re-running with only some packages missing shows only those in the Install button's label and command.

## 2. Enable the plugin

1. **Plugins → Manage and Install Plugins → Installed** tab.
2. Enable **Show also experimental plugins** (Settings tab).
3. Tick **SDM**.
4. Confirm the toolbar icon appears (or **Plugins → SDM → Run SDM…**).

If the plugin fails to load, the exact error appears in the Python Console log.

## 3. Walk through the wizard — presence/absence run (fastest, 2–5 min)

Test data lives at `~/Documents/sdm_plugin/example_data/`.

Every page listed with a button below now requires clicking it (Load & Preview / Load & Validate / Run Cleaning / Sample Background / Run VIF / Preview Split / Validate) — Next stays disabled until that click succeeds. See §5 for what to check about the embedded preview, and §6 for the real-QGIS layer loading.

| Page | Setting |
|---|---|
| Welcome | Next |
| Occurrence | Data mode: **Presence/absence**. File: `example_data/occurrences_pa.csv`. Presence field: `presence`. CRS: `EPSG:32633`. Click **Load & Preview**. |
| Predictors | Add `temperature.tif`, `precipitation.tif`, `elevation.tif`. Click **Load & Validate**. |
| Cleaning | Both boxes checked. Click **Run Cleaning**. |
| VIF | Leave 10.0. Click **Run VIF**. |
| Split | **k-fold**, k=5. Click **Preview Split**. |
| Algorithms | Nothing is checked by default — check only **LR** and **RF** for this first run. Replicates = 2. |
| Projection | Skip (empty list). |
| Ensemble | Weighted by TSS. |
| Output | Point to a new empty folder, e.g. `~/Documents/sdm_plugin/example_data/run_pa`. |
| Run | Click **Start**. Watch the progress bar + log — loading/cleaning/VIF stages should log "using cached" style messages and the run should feel faster than a from-scratch run, since those stages were already computed on earlier pages. |

**Success criteria for this run:**
- No exceptions in the log.
- Metrics show sensible AUC — expect > 0.85 for both LR and RF (the niche has a clean signal).
- Output folder contains: `suitability_lr.tif`, `suitability_lr_binary.tif`, `suitability_rf.tif`, `suitability_rf_binary.tif`, `ensemble_suitability.tif`, `ensemble_uncertainty_sd.tif`, `plots/`, `report.html`.
- Opening `report.html` in a browser shows all sections filled in, including the new **"7. Output maps"** section (a continuous + binary map image per algorithm, plus Ensemble continuous/binary/uncertainty-SD) and **"9. Response curves"** now leading with an **Ensemble** entry per predictor — each showing every algorithm's mean curve as a thin colored line plus a bold black "Ensemble" line (the actual weighted blend used for the raster ensemble, per `cfg.ensemble.method`).
- Loading the continuous ensemble raster into QGIS produces a map that visually matches `example_data/TRUTH_reference.png` — the new `plots/map_ensemble_suitability.png` in the report should look like a lower-resolution static preview of the same pattern.
- The Summary page's embedded preview shows the same ensemble raster with a blue pseudocolor ramp.

## 4. Presence-only run with projection (longer, 10–20 min)

| Page | Setting |
|---|---|
| Occurrence | **Presence-only**. File: `occurrences_po.csv`. Leave presence field empty. CRS: `EPSG:32633`. Click **Load & Preview**. |
| Predictors | Same 3 rasters. Click **Load & Validate**. |
| Cleaning | Both checked. Click **Run Cleaning**. |
| Background | Count 5000, **Random**. Buffer distance field is unused for Random (only matters for **Buffered**) — try switching to Buffered and entering e.g. `50` km, and check the italic conversion line below it reads sensibly for `EPSG:32633` (a projected CRS, so it should just say "= 50,000 m ..."). Click **Sample Background** — preview should show ~5000 grey background points plus the presence points in green. |
| VIF | 10.0. Click **Run VIF**. |
| Split | **Spatial block**, k=4, auto block size on. Click **Preview Split** — points should be colored by block/fold, and the summary line should show the auto-computed block size in km/m (not a raw small decimal number). |
| Algorithms | Nothing checked by default — check all 9. Replicates = 3. |
| Projection | Add the 3 rasters from `example_data/future_2050/`. Click **Validate**. |
| Ensemble | Weighted by TSS. |
| Output | New folder e.g. `~/Documents/sdm_plugin/example_data/run_po`. |
| Run | Start. |

**Success criteria:**
- MaxEnt should now succeed (previously failed every time with `MaxentModel.__init__() got an unexpected keyword argument 'n_threads'` — fixed by matching the actually-installed `elapid` 1.0.4 API: the parameter is `n_cpus`, and the output transform (`cloglog`) is set on the constructor, not passed to `.predict()`). Some other algorithm runs may still fail on a very small dataset (ENFA can be picky) — that's expected and should be logged without halting the run (`failed_runs` populated, pipeline continues), but MaxEnt specifically should no longer be a guaranteed failure.
- `projection_*.tif`, `projection_mess.tif`, `projection_mop.tif` present in the output folder.
- MESS raster shows negative values (novel environment flags) — the future scenario is warmer/drier than training data.
- Since this run has a projection stack, `report.html` should now also show an **"8. Projection maps"** section (per-algorithm projected suitability, plus MESS with a red/blue diverging colormap centered on zero and MOP) — this section is absent entirely in the §3 run above (no projection stack there), which is expected, not a bug.

**Also worth testing once**: repeat the Background/Split steps above with an `EPSG:4326` (geographic) dataset instead of the `EPSG:32633` example data — that's the path that actually exercises the latitude-corrected degree conversion (`is_geographic_crs`/`distance_to_crs_units` in `core/units.py`), since `EPSG:32633` only ever exercises the projected pass-through branch. Confirm the italic conversion label switches to showing a small degree value (e.g. "= 0.45123° ...") instead of a meters figure, and that the resulting run doesn't error.

## 5. Inline preview behavior (per stage page)

Applies to Occurrence, Predictors, Cleaning, Background, VIF, Split, and Projection.

For each page, confirm:
- **Busy state**: clicking the action button disables it and shows the "Loading…/Running…/etc." label until the result comes back.
- **Correct content**: the table/summary/canvas reflects `example_data/` reality — e.g. Occurrence shows the true point count and presence/absence split; Cleaning's before/after counts and dropped-point table make sense; VIF's table lists the actual predictors dropped at cutoff 10.0 (with 3 predictors, likely 0–1 dropped); Split's fold-size table sums to the full point count.
- **Next gating**: Next stays disabled until the click succeeds; it re-enables the moment the run completes.
- **Edit-after-run**: after a successful run, change any field on the same page (e.g. nudge the VIF cutoff) — Next should immediately disable again without needing another click elsewhere.
- **Cross-page invalidation**: complete Occurrence → Predictors → Cleaning → VIF, then click **Back** repeatedly to Predictors and add/remove a raster, then **Load & Validate** again. Next should now be disabled on Cleaning *and* VIF until each is re-run — confirming a change upstream invalidates cached downstream previews rather than silently reusing stale data.
- **Thread safety**: closing the wizard mid-computation (click a stage button, then immediately close the wizard window before it finishes) should not crash QGIS.

Note: as of the real-QGIS-integration change, each stage's preview now *also* loads into the real project (§6) alongside the small embedded canvas inside the dialog — the two are intentionally kept in sync, not isolated from each other. The embedded canvas widget itself is still a private `QgsMapCanvas` that never touches the project directly; it's `ui/qgis_layers.py`'s separate calls that do the real loading.

## 6. Real QGIS layer loading (new)

Extending `example_data/` walkthroughs §3/§4: after each stage's button click, in addition to the embedded canvas, check the QGIS **Layers panel**:

- A top-level group named **"SDM (preview)"** appears (created on first use), containing one child group per stage you've run: **Occurrence**, **Predictors**, **Cleaning**, **Background**, **Split**, **Projection**.
- **Ordering — most recent on top**: every new group and every new layer is inserted at the *top* of its parent, never appended to the bottom (this applies uniformly: stage groups under "SDM (preview)", subgroups under "SDM Run — ...", and individual raster/point layers within any group). Concretely: if you run Occurrence then Predictors then Cleaning, the stage groups should read Cleaning/Predictors/Occurrence top-to-bottom (most recently run at top), not pipeline order — and within "SDM Run — ...", **Ensemble** (computed after all per-algorithm results) should sit *above* **Per-algorithm**, not below it as originally shipped. (Reported via screenshot: Ensemble was showing below Per-algorithm, and new layers within a group were appending to the bottom instead of the top — fixed in `ui/qgis_layers.py` by inserting at index 0 everywhere instead of appending.)
- **Occurrence**/**Cleaning**/**Background**/**Split** groups contain a real point layer you can open the attribute table on, colored to match the embedded canvas (presence/background, kept/dropped, fold id).
- **Predictors** group contains **every** predictor raster as its own real layer (all of them, e.g. all 32 in a large predictor set) — only the first is checked/visible by default, the rest are present but unchecked so the canvas doesn't try to render dozens of rasters at once. This is the fix for the previously-reported bug (was a nearly-invisible transparent outline; now real rendered pixels, auto-zoomed correctly regardless of the predictor extent's aspect ratio).
- **Projection** group appears the same way once you validate a projection stack — this page had no spatial preview at all before this change.
- **Rerun behavior**: go back to a stage, change its settings, and rerun it (e.g. re-pick predictor rasters and click Load & Validate again) — its QGIS group's contents should be *replaced*, not appended to (no duplicate/stale layers piling up), and the group itself should move back to the top since it was just re-run.
- **Very large background counts**: set the Background page's count above 200,000 and Sample — the status text should note that the QGIS preview is capped to a subsample while the full count is still used for modeling; this should stay responsive (no multi-second freeze) even at 10,000,000.
- **After a run finishes**: a *separate*, persistently-named group **"SDM Run — `<output folder name>`"** appears (not inside "SDM (preview)" and not cleared when you go back and change earlier pages), containing **Ensemble** / **Per-algorithm** subgroups (and **Projection** if you used one) with every output raster loaded and pseudocolor-styled — this should happen automatically, with no confirmation dialog, right when the run completes (before you even reach the Summary page).
- **Rerun into the same output folder**: run once, then go back and rerun into the *identical* output directory — the existing "SDM Run — ..." group should be cleared and repopulated in place, not duplicated into a second identically-named group.
- **Project layer names**: confirm none of this collides with or renames any layers/groups you had in the project before opening the wizard.
- **Symbology legend range**: open Layer Properties → Symbology for a couple of these (a predictor raster and an output like `ensemble_suitability`) — the Min/Max fields and the legend should reflect that layer's real computed data range (e.g. roughly 0.0–1.0 for a suitability output), **not** a flat 0–255 for every layer regardless of content. (Was reported broken: `QgsSingleBandPseudoColorRenderer`'s classification-min/max metadata — what the Symbology panel actually displays — is independent of the color ramp's own stops unless set explicitly; fixed in `ui/render_helpers.py::pseudocolor_renderer` by calling `setClassificationMin`/`setClassificationMax` with the real computed values.)
- **Predictor colors**: with a multi-predictor stack loaded (e.g. all 32 from a real project), each raster in the **Predictors** group should have a visibly *different* color ramp — not all shades of blue as before. Reloading the *same* predictor stack (rerun Load & Validate, or reopen the wizard on the same files) should give each predictor the *same* colors as last time (colors are chosen from the raster's filename, not re-randomized every load).
- **Output colors**: in an "SDM Run — ..." group, `suitability_*`/`ensemble_suitability` (continuous) should render in a **Spectral** (red–yellow–blue) ramp, `*_binary` layers in two shades of **green** (light = unsuitable, dark = suitable, using QGIS's exact/discrete classification rather than an interpolated gradient since only 0 and 1 ever occur), and `ensemble_uncertainty_sd` in **magma** (dark purple → yellow). Projection/MESS/MOP layers are unchanged (still the default ramp) — not part of this request; say the word if you'd like those covered too.
- `report.html`'s own **"7. Output maps"** section (the static PNG previews, not this Layers-panel section) now uses the same Spectral/Greens/magma scheme for suitability/binary/uncertainty, so what you see in QGIS and what's in the report should look consistent with each other.

## 7. Menu, toolbar, and dock panel (new)

- After enabling the plugin, confirm **three** independent ways to launch it all work: the conventional **Plugins → SDM → Run SDM…** entry, a **dedicated "SDM" toolbar** (a named, independently show/hideable toolbar — right-click the toolbar area to confirm "SDM" is listed, distinct from the generic Plugins toolbar), and a **top-level "SDM" menu** in the main menu bar itself (next to Project/Edit/View/.../Help — check it sits before Help, not appended after every other menu).
- **View → Panels → SDM** (or drag the dock out if it's already docked) — confirm a dock panel titled "SDM" appears with a **"Run SDM…"** button and no visible status section yet (before any run).
- Click **Run SDM…** from the dock — the wizard should open (dependency-check dialog first if anything's missing, same as the other two entry points).
- With the wizard already open, trigger any of the three launch paths again — it should **refocus the existing wizard window**, not open a second one.
- **Close the wizard (X button or Cancel), then relaunch it from any entry point** — it must open a fresh wizard, not silently do nothing. (This was reported broken: closing hid the window instead of destroying it, so the reuse-if-already-open logic kept "finding" a hidden window and never called `.show()` on it again. Fixed by setting `WA_DeleteOnClose` on the wizard and checking `isVisible()` before deciding to reuse vs. rebuild in `plugin.py::_open_wizard`.) Repeat close-then-reopen a few times in a row to be sure.
- Complete a run — the dock's status section should become visible, showing the output directory and success/failure summary, with working **Open HTML report** / **Open output folder** buttons.
- Disable then re-enable the plugin (**Plugins → Manage and Install Plugins**) — the toolbar, top-level menu, and dock should all cleanly disappear and reappear, with no duplicate/ghost menu or toolbar left behind.

## 9. Conditional field enabling + small-window layout (new)

- **Background page**: with **Random across raster extent** selected (the default), the buffer distance value/unit fields and the CRS-conversion caption below them should appear grayed out (disabled, not hidden). Switch to **Buffered around presences** — they should become editable immediately.
- **Split page**: with **Random hold-out** selected, only **Test size** is enabled — **k**, **Auto block size**, and **Block size** are grayed out. Switch to **k-fold** — **k** enables, everything else stays disabled. Switch to **Spatial block** (the default) — **k** and **Auto block size** enable; **Block size** stays disabled as long as **Auto block size** is checked, and only becomes editable once you uncheck it.
- **Small window**: manually resize the wizard down as small as it'll go (or drop it well below its default size on a smaller/lower-res screen) — every page's content should become scrollable (mouse wheel / scrollbar) rather than clipping buttons, tables, or the embedded map preview off the bottom of the window. The wizard's default open size is now larger (900×700 minimum, opens at 980×760) to make this a rarer case in normal use.

## 10. Visual styling pass (new)

A field-guide palette (forest green primary #2F5D50, clay #B5622E as a rare
secondary accent, warm paper canvas #F6F5F0) is now applied everywhere via
`ui/theme.py`'s `APP_QSS`, deliberately overriding whatever QGIS's own
active application theme is — the wizard, dependency dialog, and dock should
all look identical regardless of whether QGIS itself is running its default
light theme, "Blend of Gray", or a dark theme. Approved design direction is
mocked up at the artifact referenced in this session; compare against it.

- **Wizard chrome**: open the wizard — the banner/background should be the
  warm paper canvas color, not QGIS's own theme color. The **left sidebar**
  (`WizardSidebar`) should render with a solid forest-dark green background
  listing all 13 steps — if it renders transparent/unstyled instead, the
  `WA_StyledBackground` fix didn't take effect (check `ui/widgets/step_sidebar.py`).
- **Sidebar step states**: the current step's marker should be a filled
  white circle with the step number in forest-dark text, with the row itself
  faintly highlighted. Completed steps (i.e. pages where `isComplete()` is
  true and they're not the current page) should show a clay-colored circle
  with a checkmark. Untouched/future steps show an outlined circle with just
  the number. Step backward and forward through the wizard and confirm the
  sidebar updates live as you go — not just marking one step, but correctly
  recomputing every row each time (e.g. going back from step 5 to step 2
  should make 3 and 4 revert from "done" back to "pending" if you then change
  an earlier setting and invalidate them).
- **Primary buttons**: each page's main action button (Load & Preview, Run
  Cleaning, Sample Background, Run VIF, Preview Split, Validate, Start, Open
  HTML report) should render as a solid forest-green filled button, distinct
  from ordinary white/outlined buttons elsewhere on the same page. The
  native wizard **Next**/**Finish** buttons (bottom-right, not custom
  widgets — built into `QWizard` itself) should also pick up the same forest
  fill; confirm they're tagged and not accidentally left as plain default
  buttons.
- **Inputs, tables, groupboxes**: spot check a page with a `QGroupBox` (if
  any), a `QTableWidget` (VIF page, Split page), and text/spin/combo inputs
  — borders should be a soft warm-gray (`#DAD6C9`), focus rings should turn
  forest green on click, and table headers should show a pale moss-green
  background with forest-dark text rather than the OS-default gray.
- **Dependency dialog + dock panel**: trigger the missing-dependencies
  dialog (or the dock's "Run SDM…" panel) and confirm both also show the
  same palette — canvas background, forest-green "Install missing
  package(s)" / "Run SDM…" / "Open HTML report" buttons — not the
  unstyled OS default.
- **Disabled-state legibility**: on the Background/Split pages, toggle
  through the conditional-field states (§9 above) and confirm disabled
  fields are visibly grayed out (muted text `#9A998F`, canvas-colored
  background) rather than looking identical to enabled fields or unreadable.
- **No dark-QGIS bleed-through**: if you have a dark QGIS theme active,
  double check no native dark-gray/dark-text artifacts leak into the wizard
  through un-styled corners (e.g. tooltips, scrollbars) — the whole surface
  should read as one consistent light theme.

This entire pass was designed and reviewed as QSS/Python source only — there
is no QGIS install in the dev environment, so none of it has been visually
confirmed to render correctly. Please note any places where it looks off
(a widget not picking up the theme, a color that reads wrong on your
monitor, the sidebar not updating) so they can be fixed as follow-ups.

## 11. Styling fixes, round 2 (new)

Follow-up fixes from real-QGIS feedback on the round-1 styling pass (§10).

- **No dark banner**: the wizard switched from `ModernStyle` to `ClassicStyle`
  (`ui/wizard.py`) — `ModernStyle`'s native top banner was picking up a dark
  background from the platform style + our custom `QPalette`, with our QSS
  forcing dark ink text on top of it (unreadable). Open the wizard and
  confirm there is **no colored banner strip** at the top of any page —
  title/subtitle should render as plain bold text directly on the warm paper
  background, clearly readable.
- **Welcome page**: now has grouped sections ("What this does", "Who it's
  for", "What you'll need", "The workflow") instead of a single bullet list —
  confirm it reads as a real landing page, not empty space, and that each
  `QGroupBox` picks up the theme (bold forest-dark titles, white panel
  background).
- **Sidebar no longer marks unvisited steps done**: open the wizard fresh —
  on the Welcome page, every other step in the left sidebar should show as
  **pending** (outlined circle), not a clay checkmark. Step forward through
  a few pages, then check the sidebar only marks a step done once you've
  actually been on it and its `isComplete()` is true — going back to an
  earlier page and changing a setting should be able to un-mark later steps
  if it triggers `invalidate_from`.
- **Radio buttons vs. checkboxes look distinct**: on any page with radio
  buttons (e.g. Occurrence's Presence-only/Presence-absence, Background's
  method choice, Split's CV method) the indicator should be a clearly
  visible **circle** that fills solid forest-green with a white core when
  selected. On the Algorithms page (checkboxes), indicators should be a
  visibly bordered **square** that fills solid forest-green when checked.
  Both should be easy to spot at a glance, not tiny/blended into the
  background like the native OS default.
- **Background points / Cross-validation pages no longer oversized**: open
  each — the embedded map preview should now cap at a reasonable height
  (~300px) rather than stretching to fill all remaining vertical space in a
  tall window. Any leftover space should sit as blank margin below the
  content, not inside the map widget itself.
- **Projection progress actually moves**: with a projection stack configured,
  run the pipeline and watch the Run page's progress bar/log during the
  `[project]` stage — it should visibly step through
  "Projecting [algo] (i/n)" per algorithm, then "Computing MESS…", "Computing
  MOP…", "Projection complete", rather than sitting frozen on "Projecting to
  secondary raster stack" for the whole stage (this was previously reported
  as looking hung, especially with several algorithms/replicates and a large
  projection raster).
- **HTML report now themed**: open a run's `report.html` in a browser —
  it should visually match the wizard's field-guide palette (forest-green
  headings, warm paper background, clay-colored warning callouts, styled
  tables matching the wizard's own table header styling) instead of the
  previous generic blue/white report look.

As before, none of the Qt-side changes in this section have been visually
confirmed in real QGIS (no QGIS install in the dev environment) — the HTML
report change *was* directly rendered and visually inspectable outside QGIS
during development, so that one is lower-risk than the rest.

## 12. Styling fixes, round 3 (new)

Round-2's canvas-height cap didn't fix the Background/CV pages' real problem
— screenshots showed a **horizontal scrollbar** with description text
clipped mid-word ("...used directly for a pr|"). Root cause: the long
explanatory `QLabel` on those pages (and on the VIF and Ensemble pages) was
created without `setWordWrap(True)`, so its preferred width was the entire
sentence on one line — wider than the wizard — which forced the scroll
area's content wider than the window and produced the scrollbar instead of
just wrapping the text downward. Fixed via a new `wrapped_label()` helper in
`ui/page_utils.py`, applied everywhere a long inline label existed.

- **Background points / Cross-validation pages**: open each — there should
  be **no horizontal scrollbar**, and the "Enter the buffer/block distance…"
  explanatory text should wrap onto multiple lines rather than being cut off.
  Resize the wizard narrower and confirm the text keeps wrapping instead of
  the page growing sideways.
- **VIF and Ensemble pages**: same check — their explanatory paragraphs
  should wrap, no horizontal scrollbar.
- **Welcome page — more content**: now includes a small stats row (algorithm
  count / step count / ensemble map count in clay-colored numbers) and a new
  **"Highlights"** section (a 2-column grid of six short feature call-outs:
  live previews, reproducibility, ensemble+uncertainty, spatial CV, smart
  caching, CRS-flexible distances) with a clay left-accent border — confirm
  it renders as a proper 2-column grid, not a single squeezed column, and
  that the accent border is visible.
- **HTML report — VIF section**: open a report with more than 2–3 VIF steps
  (or any report — the layout applies regardless of step count) and confirm
  the step cards lay out **side-by-side in a grid**, filling the row before
  wrapping to a new row, instead of one full-width table stacked per step
  taking up excessive vertical space.

As with round 2, the Qt-side changes are unverified in real QGIS; the HTML
report change was rendered and inspected directly (structurally, not
screenshotted in an actual browser) during development.

## 13. Spatial block CV grid overlay + square/hexagon shape (updated)

The underlying algorithm (`core/split/spatial_block.py::spatial_block_folds`)
partitions the predictor extent into a regular grid (`block_size` estimated
from an empirical variogram range, or user-set), assigns each occupied cell
a block id, shuffles the unique block ids, and deals them round-robin across
k folds. That's the same "random" block-to-fold assignment approach used by
R's `blockCV`/`ENMeval`. The Split page draws this as fold-colored polygons
underneath the colored points (`render_helpers.block_fold_layer()`, wired
into both the embedded canvas via `EmbeddedPreviewCanvas.set_block_polygons`
and the real QGIS project via `qgis_layers.show_spatial_block`) — not thin
unfilled grid lines (an earlier, now-removed `grid_layer()` drew only lines;
it was replaced by the filled-polygon version and is no longer in the code).

**Block shape (new)**: `block_shape` (`core/config.py::SplitConfig`, wired
through `core/stages.py::make_folds` to `spatial_block_folds`) now supports
`"square"` (the original regular grid) or `"hexagon"` (a regular hexagonal
tiling, matching what tools like blockCV default to — every hex has 6
equidistant neighbors, unlike a square's mixed orthogonal/diagonal
distances). Both interpret `block_size` as the same ground area per block,
so switching shape shouldn't change the typical block scale. The actual
hexagon tessellation math (`core/split/spatial_block.py`, axial-coordinate
hex grid with cube-coordinate nearest-cell rounding) is covered by geometry
regression tests in `tests/test_split.py` (regular-hexagon vertex checks,
exact area calibration, axial round-trip) since it can't be visually
verified in this dev environment — real-QGIS confirmation is still needed
for the rendered result.

- **Spatial block preview shows filled blocks**: on the Cross-validation
  page with **Spatial block** selected, click **Preview Split** — both the
  embedded map and the real QGIS "Split" layer group should show
  semi-transparent colored polygons (one per occupied block, colored to
  match the fold its points belong to) underneath the fold-colored points,
  so you can see the folds are contiguous blocks rather than an arbitrary
  per-point label. Points should render on top of the blocks, not hidden
  underneath. Switching to **Random hold-out** or **k-fold** should show
  colored points with *no* blocks (those methods don't use them).
- **Block shape selector (new)**: a **"Block shape"** dropdown (Square /
  Hexagonal) appears once **Spatial block** is selected (grayed out
  otherwise, like the other spatial-block-only fields). Switching it and
  re-running **Preview Split** should visibly change the block tiling —
  square blocks should look like the original rectangular grid; hexagonal
  blocks should look like a honeycomb tiling, each cell roughly the same
  ground area as a square block would be at the same block size. The
  summary line above the table should state which shape was used (e.g.
  "Spatial block: size=..., hexagonal, ...").
- Re-running the preview (e.g. after changing block size or shape) should
  replace both the old points and old blocks in place, not accumulate
  duplicates.

Unverified in real QGIS like the rest of this document's Qt-side changes.

## 14. Report layout consistency + model configuration alignment (new)

**Report layout**: the VIF card-grid pattern from round 3 was generalized
(`.vif-grid`/`.vif-card` → reusable `.card-grid`/`.card`) and applied to the
**Response curves** section too — each algorithm's response-curve images
now sit in their own bordered card, and cards flow side-by-side filling the
row before wrapping, instead of one full-width `<h3>`-headed block per
algorithm stacked all the way down the page. Open a report with 3+
algorithms and confirm §10 "Response curves" reads as a grid of cards, not
a long vertical stack — same visual pattern as §4 "Predictor selection".

**Model hyperparameters researched and aligned**: compared our 9
algorithms' hardcoded defaults (`core/models/*.py`) against how biomod2,
dismo, and the maxnet/elapid Maxent lineage configure the same algorithms
by default. Confirmed already-aligned: RF (randomForest's ntree=500,
mtry=sqrt(p)), GAM (mgcv's basis default k=10), SVM (e1071's cost=1/RBF).
Changed to match a verified reference default:
- **LR/GLM**: now expands each predictor into its own quadratic term (no
  cross-interactions) before fitting — matches biomod2's GLM default
  (`type='quadratic', interaction.level=0`), not a plain linear-terms-only
  logistic regression as before.
- **MaxEnt**: `feature_types` now defaults to `None` ("auto") and is
  resolved per fit from the training data's presence count, following
  maxnet's exact `classes="default"` rule (n<10: linear; n<15: +quadratic;
  n<80: +hinge; n>=80: +product) instead of a fixed
  linear+hinge+product regardless of sample size. `beta_multiplier` default
  corrected from 1.0 to 1.5 (elapid's own actual package default).
- **GBM/BRT**: `learning_rate` 0.05→0.01 and `n_estimators` 500→1000,
  matching dismo's `gbm.step` default learning rate (raising tree count to
  compensate, since we don't do gbm.step's adaptive tree-count search).
- **XGBoost**: `learning_rate` 0.05→0.1, `max_depth` 4→5, matching
  biomod2's own XGBOOST "default" strategy values.
- **MLP/ANN**: `hidden_layer_sizes` (32, 16)→(8,) (a single layer, not two),
  `alpha` 1e-4→0.01 — biomod2's ANN (nnet) tunes a single hidden layer over
  size∈{2,4,6,8} and decay∈{0.001–0.1} via internal CV; we don't run that
  inner search, so defaults now sit within that same grid instead of a much
  larger, unreferenced two-layer network.

This changes what a fresh run actually fits — re-run the presence-only
`test_run2` (or similar) dataset and confirm nothing errors, and that
per-algorithm AUC/TSS in the report's §7 "Model performance" table still
look sane (MaxEnt in particular, given its now-adaptive feature classes).

**Show/download model configuration**: on the **Algorithms** page, a new
**"View model configuration…"** button opens a dialog listing every
selected algorithm's exact hyperparameters (all algorithms' defaults if
none are selected yet) in a tree, with a **"Save as JSON…"** button to
export it. If you're past the Background page when you open it, MaxEnt's
`feature_types` should show already resolved to concrete classes (e.g.
"linear, quadratic, hinge") based on this session's actual presence count,
with a note explaining that; earlier in the wizard it should show the
selection *rule* in words instead. Separately, every run's output directory
now also gets a `model_hyperparameters.json` file, and the HTML report has
a new **§2 "Model configuration"** section (config sections renumbered 1–12
accordingly) showing the same thing per algorithm.

## 15. Input widget polish: model config editors, spinners, CRS picker (new)

None of this has been visually confirmed in real QGIS (no QGIS install in the
dev environment) — same caveat as every prior styling section.

- **Model configuration dialog — no more clipped entry boxes**: open
  **Algorithms → View model configuration…**, double-click a numeric value
  (e.g. RF's `n_estimators`) — the inline edit box should render fully
  inside the row, not vertically squeezed/cut off as it was before.
- **Dropdowns for fixed-choice parameters**: double-click SVM's `gamma` or
  RF's `max_features` — each should open as an editable dropdown listing the
  valid choices (gamma: scale/auto; max_features: sqrt/log2/None). You can
  still type a number into these (e.g. a custom `gamma` float) since the
  dropdown is editable. SVM's `kernel` should instead be a **locked** dropdown
  (linear/poly/rbf/sigmoid only, no typing) — it never accepts a number, so
  there's no reason to allow free text there.
- **Dropdown popup readability (fixed, was broken)**: opening any of the
  dropdowns above (or any other combo box in the wizard) must show every
  option in normal readable text (dark text on a white background, matching
  the theme) — not all-black/unreadable text on a black background with only
  the selected item visible. Also confirm the hovered/selected item in the
  open dropdown shows the pale-green highlight, not a jarring color.
- **MaxEnt's `feature_types` — checkbox popup**: double-click it — instead
  of an inline text box, a small dialog should open with five checkboxes
  (linear/quadratic/product/hinge/threshold), pre-checked to match whatever
  the cell currently shows. OK with nothing checked should set the value
  back to `None` (auto-selection from the presence count); OK with some
  checked should set it to that list, e.g. `['linear', 'hinge']`.
- **Spinner styling (fixed, was broken)**: on any page with a numeric spin
  box (e.g. Background page's count, Split page's k, Algorithms page's
  replicates), the up/down buttons must show actual small triangle glyphs
  (bundled PNG icons under `ui/resources/`) — the first pass only styled the
  button box color/border and left the arrows rendering as plain solid green
  rectangles with no visible triangle shape; confirm that's no longer the
  case and a real ▲/▼ is visible in each button.
- **CRS picker**: on the Occurrence page, the CRS field should now be a
  button showing the current CRS (e.g. "EPSG:4326 - WGS 84") that opens
  QGIS's own CRS selection dialog when clicked — the same widget used in
  native Processing dialogs (searchable by name/EPSG code, recently-used
  list) — instead of a plain text box. Confirm a freshly picked CRS is
  reflected correctly after clicking **Load & Preview** (check the result
  label's "CRS: ..." text matches what was picked).
- **Run log timestamps + total runtime (new)**: on the Run page, start a run
  and confirm every log line is prefixed with a `[HH:MM:SS]` timestamp, and
  that a **"Total runtime: ..."** label appears once the run finishes (and
  the final log line also states it), formatted like "45s", "3m 12s", or
  "1h 02m 05s" depending on length. On a run that fails partway through, a
  **"Failed after: ..."** label/log line should appear instead. Also open
  the resulting `report.html` — its header line (next to "Generated ...")
  should show the same **"Total runtime: ..."** value as the wizard did.

## 16. Beginner-friendly wording, VIF opt-out, CV preview staleness fix (new)

None of this has been visually confirmed in real QGIS (no QGIS install in the
dev environment) — same caveat as every prior styling/behavior section.

- **Spelled-out abbreviations**: read every wizard page's subtitle (the text
  directly under the bold step title) and confirm each technical abbreviation
  is spelled out in parentheses on first use rather than assumed: VIF
  (Predictor selection page, and briefly again on the Run page), CRS
  (Predictor rasters page and the Welcome page), MESS/MOP (Projection page),
  AUC/TSS/SD (Ensemble page), CV (Cross-validation page), NaN (Cleaning
  page), and CSV (Occurrence page). E.g. the Predictor selection page should
  now open with "VIF (Variance Inflation Factor) measures
  multicollinearity...", not just "VIF" cold. Confirm nothing reads
  awkwardly or overly long — these are still one/two sentence subtitles, not
  paragraphs.
- **VIF opt-out ("keep all predictors")**: on the Predictor selection page, a
  new **"Keep all predictors (skip multicollinearity check)"** checkbox
  appears below the cutoff field. Checking it should gray out the cutoff
  spinner (it no longer applies) and change the action button's label from
  **"Run VIF"** to **"Keep All Predictors"** (and back again if unchecked).
  Click it with the checkbox checked — the
  summary should read "Multicollinearity check skipped — all N predictor(s)
  kept: ..." listing every loaded predictor, the step table should stay
  empty (no per-step VIF rows, since nothing ran), and Next should still
  enable normally. Open the resulting run's `report.html` — §4 "Predictor
  selection" should say "Skipped by user choice — all N predictor(s) kept"
  instead of the usual cutoff/step-card breakdown. Unchecking it and
  re-running should go back to the normal stepwise behavior.
- **Cross-validation preview no longer goes stale (fixed, was broken)**: on
  the Cross-validation page, select **Spatial block**, click **Preview
  Split** — confirm the fold-colored block polygons appear as usual. Now
  switch to **Random hold-out** (or **k-fold**) and click **Preview Split**
  again — the old spatial-block polygons must disappear from the embedded
  preview canvas, leaving only the new random/k-fold colored points. (Root
  cause: `EmbeddedPreviewCanvas.set_points()` only ever replaced the
  "points" layer, never the "grid" block-polygon layer a prior spatial-block
  preview had left behind, so switching split methods left the old blocks
  rendering underneath the new points indefinitely. The real QGIS "Split"
  layer group was already unaffected — `show_points()` clears the whole
  stage group before adding anything.) Switch back to **Spatial block** and
  confirm the blocks reappear correctly too (this direction already worked).

## 17. What to report back

If anything breaks, copy from the QGIS Python Console:
- The traceback text.
- The last few `[stage] …` progress messages before the error.

Also useful:
- Which algorithms failed on the presence-only run (expected: 0–2 out of 9).
- Whether `report.html` renders correctly.
- Whether the maxTSS binary threshold produces a sensible-looking suitable-area polygon compared to `TRUTH_reference.png`.

## Troubleshooting shortcuts

- **Plugin doesn't show up in the manager**: check the symlink exists and is not broken:
  ```
  ls -la "/Users/jmpayopay/Library/Application Support/QGIS/QGIS4/profiles/default/python/plugins/sdm_plugin"
  ```
- **Import errors on launch**: run the deps check manually in the Python Console:
  ```python
  for m in ["numpy","pandas","rasterio","fiona","sklearn","matplotlib","jinja2","joblib","pygam","xgboost","elapid"]:
      try: __import__(m); print(m, "ok")
      except Exception as e: print(m, "MISS", e)
  ```
- **Reload plugin after code edits without restarting QGIS**: install the "Plugin Reloader" plugin (`Plugins → Manage and Install → Plugin Reloader`), configure it to reload `sdm_plugin`, then use its toolbar button after each edit.
- **Log fills up but wizard hangs**: the pipeline is CPU-bound; on a small dataset with all 9 algorithms × replicates × k folds it can still take minutes. The Cancel button is not implemented yet; if you need to abort, closing the wizard window will kill the QThread.
