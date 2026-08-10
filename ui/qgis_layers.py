from __future__ import annotations

from pathlib import Path

import numpy as np
from qgis.core import (
    QgsLayerTreeGroup,
    QgsProject,
    QgsRasterLayer,
)

from ..core.io.rasters import RasterStack
from .render_helpers import binary_renderer, block_fold_layer, points_layer, predictor_cmap, pseudocolor_renderer

# Marker properties, not display names, so we never accidentally reuse a
# group the user happens to have created themselves with a matching name.
_ROOT_PROPERTY = "sdm_plugin_root"
_OUTPUT_PROPERTY = "sdm_output_dir"
_PREVIEW_ROOT_NAME = "SDM (preview)"

# Points above this count are rendered as a random subsample only — building
# millions of QgsFeature objects in a Python loop on the GUI thread (this
# code only ever runs from a stage page's success callback, which must stay
# on the GUI thread) would visibly freeze QGIS. This never affects modeling,
# only what's drawn in the real project.
MAX_RENDER_POINTS = 200_000


def _find_sdm_root(project: QgsProject) -> QgsLayerTreeGroup | None:
    root = project.layerTreeRoot()
    for child in root.children():
        if isinstance(child, QgsLayerTreeGroup) and child.customProperty(_ROOT_PROPERTY) == "1":
            return child
    return None


def _sdm_root(project: QgsProject) -> QgsLayerTreeGroup:
    """Find-or-create variant, for callers about to add a layer."""
    existing = _find_sdm_root(project)
    if existing is not None:
        return existing
    group = project.layerTreeRoot().insertGroup(0, _PREVIEW_ROOT_NAME)
    group.setCustomProperty(_ROOT_PROPERTY, "1")
    return group


def _stage_group(stage_name: str) -> QgsLayerTreeGroup:
    """Find-or-create a stage's child group. New groups always land at index
    0 (the top) of their parent, consistent with every other insertion in
    this module: the most recently added group/layer is always on top, not
    appended to the bottom."""
    root = _sdm_root(QgsProject.instance())
    group = root.findGroup(stage_name)
    if group is not None:
        return group
    return root.insertGroup(0, stage_name)


def clear_stage(stage_name: str) -> None:
    """Remove every layer currently under a stage's preview group (the group
    node itself is kept and reused — this is what makes show_* calls
    'replace in place' rather than accumulating layers on every rerun).
    A no-op, creating nothing, if that stage never had a group in the first
    place (e.g. invalidating a downstream stage that hasn't run yet)."""
    root = _find_sdm_root(QgsProject.instance())
    if root is None:
        return
    group = root.findGroup(stage_name)
    if group is None:
        return
    ids = [n.layerId() for n in group.findLayers()]
    if ids:
        QgsProject.instance().removeMapLayers(ids)


def _add_to_group(group: QgsLayerTreeGroup, layer, index: int = 0) -> None:
    QgsProject.instance().addMapLayer(layer, False)
    group.insertLayer(index, layer)


def _zoom(iface, layers) -> None:
    if iface is None or not layers:
        return
    extent = None
    for layer in layers:
        e = layer.extent()
        if e.isEmpty():
            continue
        extent = e if extent is None else extent.combineExtentWith(e)
    if extent is None:
        return
    extent.scale(1.05)
    iface.mapCanvas().setExtent(extent)
    iface.mapCanvas().refresh()


def show_points(
    iface,
    stage_name: str,
    layer_name: str,
    px: np.ndarray,
    py: np.ndarray,
    labels: np.ndarray | None = None,
    colors: dict | None = None,
    crs: str | None = None,
) -> None:
    """Load points as a real, editable memory layer in the user's project,
    replacing whatever this stage previously showed. Caps rendering above
    MAX_RENDER_POINTS (see its docstring) — callers with very large point
    counts (e.g. BackgroundPage allows up to 10,000,000) should mention the
    cap in their own status text when len(px) exceeds it.
    """
    clear_stage(stage_name)
    n = len(px)
    if n == 0:
        return

    idx = np.arange(n)
    if n > MAX_RENDER_POINTS:
        idx = np.random.default_rng(0).choice(n, size=MAX_RENDER_POINTS, replace=False)

    crs_str = crs or "EPSG:4326"
    sub_labels = labels[idx] if labels is not None else None
    layer = points_layer(px[idx], py[idx], sub_labels, colors, crs_str, layer_name)

    group = _stage_group(stage_name)
    _add_to_group(group, layer)
    _zoom(iface, [layer])


def show_spatial_block(
    iface,
    stage_name: str,
    layer_name: str,
    px: np.ndarray,
    py: np.ndarray,
    fold_id: np.ndarray,
    bounds: tuple[float, float, float, float],
    block_size: float,
    n_bx: int,
    fold_of_block: dict[int, int],
    crs: str | None = None,
) -> None:
    """Like show_fold_colors, but also draws the actual block partitioning as
    fold-colored polygons underneath the colored points. A plain scatter of
    colored dots looks identical for any CV method, so without the blocks a
    spatial-block preview gives no visual evidence the folds are spatially
    contiguous blocks rather than an arbitrary label.
    """
    clear_stage(stage_name)
    group = _stage_group(stage_name)
    crs_str = crs or "EPSG:4326"

    minx, miny, _maxx, _maxy = bounds
    blocks = block_fold_layer(minx, miny, block_size, n_bx, fold_of_block, crs_str, name="block_folds")
    _add_to_group(group, blocks)

    n = len(px)
    if n == 0:
        _zoom(iface, [blocks])
        return

    idx = np.arange(n)
    if n > MAX_RENDER_POINTS:
        idx = np.random.default_rng(0).choice(n, size=MAX_RENDER_POINTS, replace=False)
    points = points_layer(px[idx], py[idx], fold_id[idx], None, crs_str, layer_name)
    _add_to_group(group, points)
    _zoom(iface, [points, blocks])


def show_rasters(iface, stage_name: str, stack: RasterStack) -> None:
    """Load every raster file in a RasterStack (one file per predictor, not
    bands of a single file) as its own real layer, replacing this stage's
    previous layers. Only the first file stays visibility-checked so the
    canvas isn't rendering many stacked rasters at once; the rest remain
    available/inspectable in the Layers panel. Each predictor gets a
    different colormap (stably chosen from its name — see
    render_helpers.predictor_cmap) instead of one fixed ramp shared by all,
    so a stack of many predictors is visually distinguishable at a glance.
    """
    clear_stage(stage_name)
    group = _stage_group(stage_name)
    layers = []
    for i, (name, path) in enumerate(zip(stack.names, stack.paths)):
        layer = QgsRasterLayer(path, name)
        if not layer.isValid():
            continue
        layer.setRenderer(pseudocolor_renderer(layer.dataProvider(), cmap=predictor_cmap(name)))
        QgsProject.instance().addMapLayer(layer, False)
        # insertLayer(0, ...) on every iteration would push each new layer
        # above the previous ones, leaving the first (visible/checked)
        # layer at the bottom of the group instead of the top.
        node = group.insertLayer(i, layer)
        if i > 0 and node is not None:
            node.setItemVisibilityChecked(False)
        layers.append(layer)
    _zoom(iface, layers)


def load_run_outputs(iface, result) -> None:
    """Auto-load every output raster from a completed run into a persistent
    group — unlike the ephemeral per-stage preview groups above, this is
    never cleared by stage invalidation. Reruns into the *same* output
    directory reuse (clear + repopulate) the same group instead of creating
    a second, identically-named one, since the on-disk filenames are
    identical each run and a stale duplicate group would otherwise silently
    start showing the new run's pixels under an old-looking label.
    """
    if iface is None:
        return
    project = QgsProject.instance()
    root = project.layerTreeRoot()
    out_dir = result.output_dir

    top = None
    for child in root.children():
        if isinstance(child, QgsLayerTreeGroup) and child.customProperty(_OUTPUT_PROPERTY) == out_dir:
            top = child
            break
    if top is not None:
        ids = [n.layerId() for n in top.findLayers()]
        if ids:
            project.removeMapLayers(ids)
        for sub in list(top.children()):
            if isinstance(sub, QgsLayerTreeGroup):
                top.removeChildNode(sub)
    else:
        run_name = f"SDM Run: {Path(out_dir).name}"
        top = root.insertGroup(0, run_name)
        top.setCustomProperty(_OUTPUT_PROPERTY, out_dir)

    # Subgroups are created lazily, only when a matching file is actually
    # found — so a run with no projection stack simply never gets an empty
    # "Projection" subgroup, no separate cleanup pass needed.
    prefix_titles = {
        "ensemble_": "Ensemble",
        "suitability_": "Per-algorithm",
        "projection_": "Projection",
    }
    created_groups: dict[str, QgsLayerTreeGroup] = {}

    layers = []
    for f in result.output_files:
        p = Path(f)
        if p.suffix.lower() not in (".tif", ".tiff"):
            continue
        title = next((t for prefix, t in prefix_titles.items() if p.stem.startswith(prefix)), None)
        if title is None:
            continue
        layer = QgsRasterLayer(str(p), p.stem)
        if not layer.isValid():
            continue
        stem = p.stem
        if stem.endswith("_binary"):
            layer.setRenderer(binary_renderer(layer.dataProvider(), cmap="Greens"))
        elif stem == "ensemble_uncertainty_sd":
            layer.setRenderer(pseudocolor_renderer(layer.dataProvider(), cmap="magma"))
        elif stem.startswith("suitability_") or stem == "ensemble_suitability":
            layer.setRenderer(pseudocolor_renderer(layer.dataProvider(), cmap="Spectral"))
        else:
            # Projection / MESS / MOP outputs — not covered by the explicit
            # suitability/binary/uncertainty scheme, left on the default ramp.
            layer.setRenderer(pseudocolor_renderer(layer.dataProvider()))
        group = created_groups.get(title)
        if group is None:
            # Insert at 0 (top), not appended — a subgroup created later
            # (e.g. "Ensemble", built from the already-finished per-algorithm
            # rasters) should end up above one created earlier, matching the
            # most-recent-on-top convention used throughout this module.
            group = top.insertGroup(0, title)
            created_groups[title] = group
        QgsProject.instance().addMapLayer(layer, False)
        group.insertLayer(0, layer)
        layers.append(layer)

    _zoom(iface, layers)
