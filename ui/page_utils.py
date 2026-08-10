from __future__ import annotations

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
