from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QBrush, QColor
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QLayout,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from .theme import CLAY_SOFT


def wrapped_label(text: str) -> QLabel:
    """A QLabel with word wrap on. Plain `QLabel(text)` keeps its preferred
    width equal to the whole string laid out on one line — for any sentence
    longer than the page, that forces the page's scroll-area content wider
    than the wizard itself, showing a horizontal scrollbar and clipping text
    instead of just growing downward.
    """
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    return lbl


def description_label() -> QLabel:
    """An empty, word-wrapped, rich-text label for copy that changes with the
    user's current selection — the method descriptions on the
    cross-validation and ensemble pages. Rich text so the "Best for:" style
    lead-ins can be bold, which is what makes three paragraphs scannable
    rather than a wall.
    """
    label = QLabel("")
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    return label


def wrap_scrollable(page: QWizardPage, layout: QLayout) -> None:
    """Mount `layout` on `page` via a QScrollArea, so content stays fully
    reachable (not clipped/cramped) when the wizard window is small — used
    in place of a page calling self.setLayout(layout) directly.
    """
    content = QWidget()
    content.setLayout(layout)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setWidget(content)
    outer = QVBoxLayout()
    outer.setContentsMargins(0, 0, 0, 0)
    outer.addWidget(scroll)
    page.setLayout(outer)


# ----- raster exploratory-data-analysis (EDA) views -----

EDA_HEADERS = ["Raster", "Data type", "Min", "Max", "Mean", "Std dev", "NoData", "Valid"]


def _fmt_num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def configure_eda_table(table: QTableWidget) -> None:
    """Set up a QTableWidget to display per-raster EDA rows (read-only)."""
    table.setColumnCount(len(EDA_HEADERS))
    table.setHorizontalHeaderLabels(EDA_HEADERS)
    table.setRowCount(0)
    table.horizontalHeader().setStretchLastSection(True)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)


def fill_eda_table(table: QTableWidget, eda_list) -> bool:
    """Populate an EDA table from a list of BandEDA. Returns True if any of the
    statistics were computed from a sampled (decimated) read rather than every
    pixel, so the caller can mention it."""
    table.setRowCount(len(eda_list))
    any_sampled = False
    for row, band in enumerate(eda_list):
        any_sampled = any_sampled or band.sampled
        dtype_txt = f"{band.kind} ({band.dtype})" if band.kind != band.dtype else band.dtype
        nodata_txt = "none" if band.nodata is None else _fmt_num(band.nodata)
        valid_txt = (
            "n/a" if band.valid_fraction is None else f"{band.valid_fraction * 100:.1f}%"
        )
        cells = [
            band.name,
            dtype_txt,
            _fmt_num(band.minimum),
            _fmt_num(band.maximum),
            _fmt_num(band.mean),
            _fmt_num(band.std),
            nodata_txt,
            valid_txt,
        ]
        for col, text in enumerate(cells):
            table.setItem(row, col, QTableWidgetItem(text))
    table.resizeColumnsToContents()
    return any_sampled


# ----- raster grid-properties view (shown when rasters don't line up) -----

PROFILE_HEADERS = [
    "Raster", "Data type", "CRS", "Size (px)", "Resolution", "NoData", "Extent",
]


def format_resolution(res: tuple[float, float]) -> str:
    res_x, res_y = res
    return f"{res_x:.6g} × {res_y:.6g}"


def format_extent(bounds: tuple[float, float, float, float]) -> str:
    minx, miny, maxx, maxy = bounds
    return f"x {minx:.8g} … {maxx:.8g},  y {miny:.8g} … {maxy:.8g}"


def profile_cells(profile) -> list[str]:
    """One RasterProfile as the row of text shown in a properties table.
    Shared by the wizard pages and the fix dialog so both describe a layer
    identically."""
    return [
        profile.name,
        profile.dtype,
        profile.crs or "none",
        f"{profile.width} × {profile.height}",
        format_resolution(profile.resolution),
        "none" if profile.nodata is None else _fmt_num(profile.nodata),
        format_extent(profile.bounds),
    ]


def configure_profile_table(table: QTableWidget) -> None:
    """Set up a QTableWidget to display per-raster grid properties. Pages
    reuse the same table widget for this and for the EDA view — calling
    either configure_* switches which one it currently shows."""
    table.setColumnCount(len(PROFILE_HEADERS))
    table.setHorizontalHeaderLabels(PROFILE_HEADERS)
    table.setRowCount(0)
    table.horizontalHeader().setStretchLastSection(True)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)


def fill_profile_table(table: QTableWidget, profiles) -> None:
    """Populate a properties table from a list of RasterProfile, tinting
    every cell that differs from the first raster's. The first raster is the
    one load_stack() treats as the reference, so the tinted cells are exactly
    the values that made the stack fail to load."""
    table.setRowCount(len(profiles))
    reference = profile_cells(profiles[0]) if profiles else []
    highlight = QBrush(QColor(CLAY_SOFT))
    # Derived from the table's own font, not a default-constructed one, so a
    # highlighted cell differs from its neighbours in weight only.
    bold = table.font()
    bold.setBold(True)
    for row, profile in enumerate(profiles):
        cells = profile_cells(profile)
        for col, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if row > 0 and text != reference[col]:
                item.setBackground(highlight)
                item.setFont(bold)
            table.setItem(row, col, item)
    table.resizeColumnsToContents()


# Captions for the two views a raster page's single table can show.
EDA_TABLE_CAPTION = "Per-raster summary (exploratory data analysis):"
PROFILE_TABLE_CAPTION = (
    "Layer properties (values that differ from the first raster are highlighted):"
)


def alignment_headline(issues, action: str) -> str:
    """The error line shown when a set of rasters can't be stacked, naming
    both what differs and the button that fixes it."""
    labels = issues.labels
    if len(labels) > 1:
        named = ", ".join(labels[:-1]) + " and " + labels[-1]
    else:
        named = labels[0] if labels else "pixel grid"
    return (
        f"These rasters don't line up. They differ in {named}, and they must "
        f"all share one CRS, extent, resolution and pixel grid. Use “{action}” "
        "to resample them onto a common grid."
    )


def raster_summary_text(stack) -> str:
    """A one-line summary of the shared grid: count, pixel dimensions,
    resolution in both native units and meters, CRS, and extent."""
    from ..core.units import crs_units_to_meters, format_meters, is_geographic_crs

    res_x, res_y = stack.resolution
    minx, miny, maxx, maxy = stack.bounds
    lat = (miny + maxy) / 2.0
    if is_geographic_crs(stack.crs):
        rx_m = crs_units_to_meters(res_x, stack.crs, lat)
        ry_m = crs_units_to_meters(res_y, stack.crs, lat)
        res_text = (
            f"{res_x:.4g}° × {res_y:.4g}° "
            f"(about {format_meters(rx_m)} × {format_meters(ry_m)} "
            f"at latitude {lat:.1f}°)"
        )
    else:
        res_text = (
            f"{format_meters(res_x)} × {format_meters(res_y)} (projected units)"
        )
    return (
        f"{len(stack.names)} raster(s), {stack.width} × {stack.height} pixels. "
        f"Resolution: {res_text}. CRS: {stack.crs}. "
        f"Extent: x from {minx:.4g} to {maxx:.4g}, y from {miny:.4g} to {maxy:.4g}."
    )
