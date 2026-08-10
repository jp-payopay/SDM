from __future__ import annotations

import zlib

import numpy as np
from qgis.core import (
    QgsCategorizedSymbolRenderer,
    QgsColorRampShader,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsRasterShader,
    QgsRendererCategory,
    QgsSingleBandPseudoColorRenderer,
    QgsSymbol,
    QgsVectorLayer,
)
from qgis.PyQt.QtGui import QColor

from .theme import INK

# A small qualitative palette, reused for any categorized preview (folds,
# kept/dropped predictors, presence/background, etc.) whether it ends up on
# the embedded preview canvas or as a real project layer, unless the caller
# supplies explicit colors.
PALETTE = [
    "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
    "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd",
]

# A curated set of matplotlib sequential/perceptually-varied colormaps,
# cycled across loaded predictor rasters so they're visually distinguishable
# in the Layers panel instead of all sharing one fixed blue ramp.
PREDICTOR_CMAPS = [
    "viridis", "plasma", "inferno", "cividis", "Blues", "Greens", "Oranges",
    "Purples", "Reds", "YlOrBr", "YlGnBu", "PuBuGn", "BuPu", "GnBu", "OrRd",
    "RdPu", "YlOrRd", "PuRd", "cool", "copper", "terrain", "ocean", "turbo",
]

# Statistics sample size for pseudocolor min/max — a full-population band
# scan is fine for a single raster, but callers loading many raster files
# back-to-back (e.g. a 32-file predictor stack) need this to stay cheap
# regardless of raster size; this matches what QGIS's own default raster
# loading uses under the hood.
_STATS_SAMPLE_SIZE = 250_000

# Number of interpolated stops sampled from a named colormap — enough for a
# smooth-looking gradient without building an excessively long ramp item list.
_N_STOPS = 12


def predictor_cmap(name: str) -> str:
    """Deterministically pick a colormap for a predictor by name, so the
    same predictor (e.g. "bio1") always gets the same color across sessions
    — "varied", not literally re-randomized on every load, which would make
    a raster's color meaningless from one run to the next. Uses crc32 rather
    than Python's built-in hash(), which is salted per-process and would
    defeat that stability."""
    idx = zlib.crc32(name.encode("utf-8")) % len(PREDICTOR_CMAPS)
    return PREDICTOR_CMAPS[idx]


def _mpl_color(cmap_name: str, frac: float) -> QColor:
    import matplotlib

    cmap = matplotlib.colormaps[cmap_name]
    r, g, b, _a = cmap(frac)
    return QColor(int(r * 255), int(g * 255), int(b * 255))


def categorized_renderer(
    layer: QgsVectorLayer,
    field: str,
    labels: np.ndarray,
    colors: dict | None = None,
) -> QgsCategorizedSymbolRenderer:
    unique = sorted({str(v) for v in labels})
    categories = []
    for i, val in enumerate(unique):
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        hex_color = (colors or {}).get(val, PALETTE[i % len(PALETTE)])
        symbol.setColor(QColor(hex_color))
        categories.append(QgsRendererCategory(val, symbol, val))
    return QgsCategorizedSymbolRenderer(field, categories)


def grid_line_geometries(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    block_size: float,
    n_bx: int,
    n_by: int,
) -> list[QgsGeometry]:
    """Vertical + horizontal line geometries for the literal block grid
    `spatial_block_folds` partitioned points into — the boundaries actually
    used to assign folds, not an approximation of them."""
    lines = []
    for i in range(n_bx + 1):
        x = minx + i * block_size
        lines.append(QgsGeometry.fromPolylineXY([QgsPointXY(x, miny), QgsPointXY(x, maxy)]))
    for j in range(n_by + 1):
        y = miny + j * block_size
        lines.append(QgsGeometry.fromPolylineXY([QgsPointXY(minx, y), QgsPointXY(maxx, y)]))
    return lines


def grid_layer(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    block_size: float,
    n_bx: int,
    n_by: int,
    crs: str,
    name: str = "block_grid",
) -> QgsVectorLayer:
    """A thin, translucent line layer showing spatial-block CV's actual grid
    — without this, a spatial block preview looks identical to a plain
    categorical scatter of points, with no visual evidence the folds are
    spatially contiguous blocks rather than an arbitrary label."""
    layer = QgsVectorLayer(f"LineString?crs={crs}", name, "memory")
    provider = layer.dataProvider()
    feats = []
    for geom in grid_line_geometries(minx, miny, maxx, maxy, block_size, n_bx, n_by):
        f = QgsFeature()
        f.setGeometry(geom)
        feats.append(f)
    provider.addFeatures(feats)
    layer.updateExtents()
    symbol = QgsSymbol.defaultSymbol(layer.geometryType())
    symbol.setColor(QColor(INK))
    symbol.setOpacity(0.4)
    symbol.symbolLayer(0).setWidth(0.4)
    layer.renderer().setSymbol(symbol)
    return layer


def block_fold_geometries(
    minx: float,
    miny: float,
    block_size: float,
    n_bx: int,
    fold_of_block: dict[int, int],
) -> list[tuple[int, QgsGeometry]]:
    """(fold_id, square polygon) for every occupied block, reconstructed from
    the same row/col math spatial_block_folds used (block_id = row * n_bx + col)."""
    out: list[tuple[int, QgsGeometry]] = []
    for block, fold in fold_of_block.items():
        row = block // n_bx
        col = block % n_bx
        x0 = minx + col * block_size
        y0 = miny + row * block_size
        x1 = x0 + block_size
        y1 = y0 + block_size
        ring = [
            QgsPointXY(x0, y0),
            QgsPointXY(x1, y0),
            QgsPointXY(x1, y1),
            QgsPointXY(x0, y1),
            QgsPointXY(x0, y0),
        ]
        out.append((int(fold), QgsGeometry.fromPolygonXY([ring])))
    return out


def block_fold_layer(
    minx: float,
    miny: float,
    block_size: float,
    n_bx: int,
    fold_of_block: dict[int, int],
    crs: str,
    name: str = "fold_blocks",
) -> QgsVectorLayer:
    """A polygon layer of the spatial-block CV blocks, filled semi-transparently
    by the fold each block was assigned to. This is the actual partitioning the
    colored points sit inside, so folds read as contiguous regions rather than
    a scatter of same-colored dots. Fill colors are assigned the same way as
    the point folds (PALETTE indexed by sorted fold value) so a block and the
    points in it share a color."""
    layer = QgsVectorLayer(f"Polygon?crs={crs}&field=fold:string(16)", name, "memory")
    provider = layer.dataProvider()
    feats = []
    labels: list[str] = []
    for fold, geom in block_fold_geometries(minx, miny, block_size, n_bx, fold_of_block):
        f = QgsFeature()
        f.setGeometry(geom)
        f.setAttributes([str(fold)])
        feats.append(f)
        labels.append(str(fold))
    provider.addFeatures(feats)
    layer.updateExtents()
    categories = []
    for i, val in enumerate(sorted(set(labels))):
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        color = QColor(PALETTE[i % len(PALETTE)])
        color.setAlpha(95)
        symbol.setColor(color)
        stroke = symbol.symbolLayer(0)
        stroke.setStrokeColor(QColor(255, 255, 255, 170))
        stroke.setStrokeWidth(0.2)
        categories.append(QgsRendererCategory(val, symbol, val))
    layer.setRenderer(QgsCategorizedSymbolRenderer("fold", categories))
    return layer


def points_layer(
    px: np.ndarray,
    py: np.ndarray,
    labels: np.ndarray | None,
    colors: dict | None,
    crs: str,
    name: str,
) -> QgsVectorLayer:
    """Shared memory-point-layer builder for any categorized point preview
    (presence/background, folds, kept/dropped, ...) — used both for the
    embedded canvas's own layers and for real project layers."""
    layer = QgsVectorLayer(f"Point?crs={crs}&field=label:string(64)", name, "memory")
    provider = layer.dataProvider()
    feats = []
    for i in range(len(px)):
        f = QgsFeature()
        f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(px[i]), float(py[i]))))
        f.setAttributes(["" if labels is None else str(labels[i])])
        feats.append(f)
    provider.addFeatures(feats)
    layer.updateExtents()
    if labels is not None:
        layer.setRenderer(categorized_renderer(layer, "label", labels, colors))
    return layer


def pseudocolor_renderer(
    provider,
    band: int = 1,
    cmap: str = "viridis",
) -> QgsSingleBandPseudoColorRenderer:
    """Continuous single-band renderer, sampling `cmap` (any matplotlib
    colormap name — "Spectral", "magma", "viridis", ...) into an
    interpolated ramp spanning the raster's real computed min/max."""
    stats = provider.bandStatistics(band, sampleSize=_STATS_SAMPLE_SIZE)
    lo, hi = stats.minimumValue, stats.maximumValue
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = 0.0, 1.0
    shader = QgsRasterShader()
    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Type.Interpolated)
    items = []
    for i in range(_N_STOPS):
        frac = i / (_N_STOPS - 1)
        value = lo + frac * (hi - lo)
        items.append(QgsColorRampShader.ColorRampItem(value, _mpl_color(cmap, frac)))
    ramp.setColorRampItemList(items)
    shader.setRasterShaderFunction(ramp)
    renderer = QgsSingleBandPseudoColorRenderer(provider, band, shader)
    # Without this, QGIS's Symbology panel shows its own generic default
    # classification range (commonly 0-255, the classic 8-bit assumption)
    # instead of this layer's real computed min/max — the shader's ramp
    # stops above are used for painting, but the renderer's own
    # classification-min/max metadata is what the properties UI displays
    # and is independent of them unless set explicitly here.
    renderer.setClassificationMin(lo)
    renderer.setClassificationMax(hi)
    return renderer


def binary_renderer(
    provider,
    band: int = 1,
    cmap: str = "Greens",
) -> QgsSingleBandPseudoColorRenderer:
    """Renderer for a strictly 0/1 binary raster — an exact two-value
    classification (not an interpolated gradient, since no value between 0
    and 1 ever occurs), taking its two colors from the low/high end of
    `cmap` rather than a fixed hardcoded pair."""
    shader = QgsRasterShader()
    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Type.Exact)
    ramp.setColorRampItemList([
        QgsColorRampShader.ColorRampItem(0, _mpl_color(cmap, 0.15), "unsuitable"),
        QgsColorRampShader.ColorRampItem(1, _mpl_color(cmap, 0.85), "suitable"),
    ])
    shader.setRasterShaderFunction(ramp)
    renderer = QgsSingleBandPseudoColorRenderer(provider, band, shader)
    renderer.setClassificationMin(0)
    renderer.setClassificationMax(1)
    return renderer
