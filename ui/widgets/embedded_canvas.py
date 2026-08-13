from __future__ import annotations

import numpy as np
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsRasterLayer,
    QgsRectangle,
    QgsSymbol,
    QgsVectorLayer,
)
from qgis.gui import QgsMapCanvas
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QPushButton, QSizePolicy, QVBoxLayout, QWidget

from ...core.io.rasters import RasterStack
from ..render_helpers import block_fold_layer, points_layer, pseudocolor_renderer


class EmbeddedPreviewCanvas(QWidget):
    """A small, self-contained QgsMapCanvas for in-dialog previews.

    Deliberately never touches iface.mapCanvas() or QgsProject.instance() —
    all layers live only on this widget's private canvas and are never added
    to the user's real QGIS project.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.canvas = QgsMapCanvas(self)
        self.canvas.setCanvasColor(QColor("white"))
        self.canvas.setMinimumHeight(220)
        # QgsMapCanvas defaults to an Expanding vertical size policy, which
        # left unconstrained lets it swallow all leftover space in a page's
        # QVBoxLayout — a tiny point-cluster preview ends up as several
        # hundred pixels of mostly-blank canvas. Cap it and switch to
        # Preferred so pages read as a fixed-size preview, not a stretched one.
        self.canvas.setMaximumHeight(300)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._layers: dict[str, object] = {}

        zoom_btn = QPushButton("Zoom to extent")
        zoom_btn.clicked.connect(self._zoom_to_layers)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(zoom_btn)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    # ----- public API -----

    def clear(self) -> None:
        self._layers.clear()
        self.canvas.setLayers([])
        self.canvas.refresh()

    def set_points(
        self,
        px: np.ndarray,
        py: np.ndarray,
        labels: np.ndarray | None = None,
        colors: dict | None = None,
        crs: str | None = None,
        max_render: int = 20_000,
    ) -> None:
        """Draw points, optionally colored by category. `labels` may hold any
        hashable value per point (presence flags, kept/dropped booleans, fold
        ids, ...). Rendering subsamples above `max_render` for responsiveness
        — this never affects any count/stat shown elsewhere, only what's
        drawn here.
        """
        if crs:
            self._set_crs(crs)
        n = len(px)
        if n == 0:
            self._replace_layer("points", None)
            return

        idx = np.arange(n)
        if n > max_render:
            idx = np.random.default_rng(0).choice(n, size=max_render, replace=False)

        sub_labels = labels[idx] if labels is not None else None
        layer = points_layer(px[idx], py[idx], sub_labels, colors, crs or "EPSG:4326", "preview_points")
        self._replace_layer("points", layer)
        self._zoom_to_bounds(float(px.min()), float(py.min()), float(px.max()), float(py.max()))

    def set_fold_colors(self, px: np.ndarray, py: np.ndarray, fold_id: np.ndarray, crs: str | None = None) -> None:
        self.set_points(px, py, labels=fold_id, crs=crs)

    def set_block_polygons(self, plan, crs: str | None = None) -> None:
        """Draw the spatial-block CV partitioning (square or hexagon, per
        plan.shape) as fold-colored polygons. Call *after* set_fold_colors so
        the colored points stay on top and the blocks read as the regions
        the points sit inside, not a layer hiding the points they explain.
        """
        if crs:
            self._set_crs(crs)
        layer = block_fold_layer(plan, crs or "EPSG:4326")
        self._replace_layer("grid", layer)

    def clear_block_polygons(self) -> None:
        """Remove the spatial-block polygons drawn by set_block_polygons, if
        any. Callers must call this when previewing a non-spatial-block split
        (random/k-fold) after a spatial-block preview — set_points() only
        ever touches the "points" role, so without an explicit clear here the
        old blocks silently keep rendering underneath the new points."""
        self._replace_layer("grid", None)

    def set_raster_extent(self, stack: RasterStack) -> None:
        """Draw the raster stack's bounding extent as an outline, without
        loading any pixel data — appropriate for predictor rasters of
        unknown/large size, where only the footprint matters.
        """
        self._set_crs(stack.crs)
        minx, miny, maxx, maxy = stack.bounds
        uri = f"Polygon?crs={stack.crs}"
        layer = QgsVectorLayer(uri, "extent", "memory")
        provider = layer.dataProvider()
        ring = [
            QgsPointXY(minx, miny),
            QgsPointXY(maxx, miny),
            QgsPointXY(maxx, maxy),
            QgsPointXY(minx, maxy),
            QgsPointXY(minx, miny),
        ]
        f = QgsFeature()
        f.setGeometry(QgsGeometry.fromPolygonXY([ring]))
        provider.addFeatures([f])
        layer.updateExtents()
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        # A semi-transparent (not fully transparent) fill so the extent still
        # reads as "your data is here" even when letterboxed small by an
        # aspect-ratio mismatch between the data extent and this widget.
        symbol.setColor(QColor(76, 114, 176, 60))
        symbol.symbolLayer(0).setStrokeColor(QColor("#4c72b0"))
        symbol.symbolLayer(0).setStrokeWidth(0.8)
        layer.renderer().setSymbol(symbol)
        self._replace_layer("extent", layer)
        self._zoom_to_bounds(minx, miny, maxx, maxy)

    def set_raster(self, path: str, name: str = "raster") -> None:
        """Load a real raster layer with a default pseudocolor ramp — only
        appropriate for small, plugin-written outputs (e.g. the final
        ensemble suitability raster), not arbitrary user rasters.
        """
        layer = QgsRasterLayer(path, name)
        if not layer.isValid():
            self._replace_layer("raster", None)
            return
        if layer.crs().isValid():
            self.canvas.setDestinationCrs(layer.crs())
        layer.setRenderer(pseudocolor_renderer(layer.dataProvider()))
        self._replace_layer("raster", layer)
        self._zoom_to_layers()

    # ----- internals -----

    def _set_crs(self, crs: str) -> None:
        self.canvas.setDestinationCrs(QgsCoordinateReferenceSystem(crs))

    def _replace_layer(self, role: str, layer) -> None:
        # Remove the old layer from the canvas's layer list *before*
        # dropping the Python reference, to avoid a stale-pointer repaint.
        self._layers.pop(role, None)
        if layer is not None:
            self._layers[role] = layer
        self.canvas.setLayers(list(self._layers.values()))
        self.canvas.refresh()

    def _zoom_to_bounds(self, minx: float, miny: float, maxx: float, maxy: float) -> None:
        pad_x = max((maxx - minx) * 0.05, 1e-9)
        pad_y = max((maxy - miny) * 0.05, 1e-9)
        self.canvas.setExtent(
            QgsRectangle(minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)
        )
        self.canvas.refresh()

    def _zoom_to_layers(self) -> None:
        if not self._layers:
            return
        extent = None
        for layer in self._layers.values():
            e = layer.extent()
            extent = e if extent is None else extent.combineExtentWith(e)
        if extent is not None and not extent.isEmpty():
            self.canvas.setExtent(extent)
            self.canvas.refresh()
