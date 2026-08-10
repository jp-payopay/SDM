from __future__ import annotations

import numpy as np
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWizardPage,
)

from ...core.stages import collect_labeled_points_and_extract
from ...core.units import distance_to_crs_units, is_geographic_crs
from ..page_utils import wrap_scrollable, wrapped_label
from ..qgis_layers import MAX_RENDER_POINTS, clear_stage, show_points
from ..widgets.embedded_canvas import EmbeddedPreviewCanvas
from ..workers import StagePageMixin, snapshot_key


class BackgroundPage(StagePageMixin, QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Background / pseudo-absence points")
        self.setSubTitle(
            "Presence-only modeling requires 'background' points sampled from the "
            "landscape. Choose how many and how to place them."
        )
        self.count = QSpinBox()
        self.count.setRange(100, 10_000_000)
        self.count.setValue(10_000)

        self.random = QRadioButton("Random across raster extent")
        self.buffered = QRadioButton("Buffered around presences")
        self.random.setChecked(True)

        self.buffer_value = QDoubleSpinBox()
        self.buffer_value.setRange(0.0, 1e9)
        self.buffer_value.setDecimals(2)
        self.buffer_value.setValue(50.0)
        self.buffer_unit = QComboBox()
        self.buffer_unit.addItems(["km", "m"])
        buffer_row = QHBoxLayout()
        buffer_row.addWidget(self.buffer_value)
        buffer_row.addWidget(self.buffer_unit)

        self.buffer_row_label = QLabel("Buffer distance:")
        form = QFormLayout()
        form.addRow(QLabel("Number of points:"), self.count)
        form.addRow(QLabel("Method:"), self.random)
        form.addRow(QLabel(""), self.buffered)
        form.addRow(self.buffer_row_label, buffer_row)

        self.run_btn = QPushButton("Sample Background")
        self.run_btn.setProperty("cls", "primary")
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.busy_label = QLabel("Sampling…")
        self.busy_label.setVisible(False)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #b00020;")
        self.error_label.setWordWrap(True)
        self.conversion_label = QLabel("")
        self.conversion_label.setStyleSheet("color: #666; font-style: italic;")
        self.conversion_label.setWordWrap(True)
        self.canvas = EmbeddedPreviewCanvas()

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(wrapped_label(
            "Enter the buffer distance in km or m. It is converted automatically to "
            "your predictor rasters' CRS units (used directly for a projected/UTM CRS, "
            "and latitude-corrected for a geographic/degree-based CRS like EPSG:4326)."
        ))
        layout.addWidget(self.conversion_label)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.busy_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.canvas)
        layout.addStretch()
        wrap_scrollable(self, layout)

        self._stage_ok = False
        self._last_snapshot: str | None = None
        self.count.valueChanged.connect(self.completeChanged)
        self.buffer_value.valueChanged.connect(self.completeChanged)
        self.buffer_value.valueChanged.connect(self._update_conversion_label)
        self.buffer_unit.currentIndexChanged.connect(self.completeChanged)
        self.buffer_unit.currentIndexChanged.connect(self._update_conversion_label)
        self.random.toggled.connect(self.completeChanged)
        self.random.toggled.connect(self._update_field_state)
        self.buffered.toggled.connect(self._update_field_state)
        self._update_conversion_label()
        self._update_field_state()

    def initializePage(self) -> None:
        # The predictor stack (needed to know the CRS) is only guaranteed
        # loaded once this page is actually reached.
        self._update_conversion_label()

    def _update_field_state(self) -> None:
        buffered = self.buffered.isChecked()
        for w in (self.buffer_row_label, self.buffer_value, self.buffer_unit, self.conversion_label):
            w.setEnabled(buffered)

    def _buffer_distance_meters(self) -> float:
        factor = 1000.0 if self.buffer_unit.currentText() == "km" else 1.0
        return self.buffer_value.value() * factor

    def _update_conversion_label(self) -> None:
        stack = self.wizard_ref.session.stack
        meters = self._buffer_distance_meters()
        if stack is None:
            self.conversion_label.setText("(predictor CRS not loaded yet)")
            return
        if is_geographic_crs(stack.crs):
            lat = (stack.bounds[1] + stack.bounds[3]) / 2.0
            crs_val = distance_to_crs_units(meters, stack.crs, lat)
            self.conversion_label.setText(
                f"= {crs_val:.5f}° in this raster's geographic CRS "
                f"(computed at centroid latitude {lat:.2f}°)."
            )
        else:
            self.conversion_label.setText(
                f"= {meters:,.0f} m in this raster's projected CRS units."
            )

    def _snapshot(self) -> str:
        wizard = self.wizard_ref
        return snapshot_key(
            self.count.value(),
            self.random.isChecked(),
            self.buffer_value.value(),
            self.buffer_unit.currentText(),
            wizard.session.stage_hashes.get("cleaning"),
        )

    def _on_run_clicked(self) -> None:
        self.error_label.setText("")
        wizard = self.wizard_ref
        self.save_to_config(wizard.config)
        cfg = wizard.config
        occ = wizard.session.occ
        stack = wizard.session.stack
        seed = cfg.random_seed

        def _work():
            rng = np.random.default_rng(seed)
            return collect_labeled_points_and_extract(cfg, occ, stack, rng)

        self.run_stage_async(
            _work, self._on_done, self._on_failed,
            button=self.run_btn, busy_widget=self.busy_label,
        )

    def _on_done(self, result) -> None:
        px, py, presence_flag, X_full, feature_names = result
        wizard = self.wizard_ref
        n_pres = int((presence_flag == 1).sum())
        n_bg = int((presence_flag == 0).sum())
        summary = f"Presence: {n_pres}, Background: {n_bg}."
        if len(px) > MAX_RENDER_POINTS:
            summary += f" (QGIS preview shows a {MAX_RENDER_POINTS:,}-point sample; the full set is used for modeling.)"
        self.summary_label.setText(summary)
        colors = {"1": "#2e7d32", "0": "#9e9e9e"}
        self.canvas.set_points(px, py, labels=presence_flag, colors=colors, crs=wizard.session.stack.crs)
        show_points(
            wizard.iface, "Background", "background_points",
            px, py, labels=presence_flag, colors=colors, crs=wizard.session.stack.crs,
        )

        new_key = self._snapshot()
        old_key = wizard.session.stage_hashes.get("background")
        if old_key is not None and old_key != new_key:
            wizard.invalidate_from(wizard.PAGE_BACKGROUND)
        wizard.session.px, wizard.session.py, wizard.session.presence_flag = px, py, presence_flag
        wizard.session.X_full, wizard.session.feature_names = X_full, feature_names
        wizard.session.stage_hashes["background"] = new_key

        self._stage_ok = True
        self._last_snapshot = new_key

    def _on_failed(self, err: str) -> None:
        self._stage_ok = False
        self.summary_label.setText("")
        self.error_label.setText(err.splitlines()[0] if err else "Background sampling failed.")

    def invalidate(self) -> None:
        self._stage_ok = False
        self._last_snapshot = None
        self.summary_label.setText("")
        self.canvas.clear()
        clear_stage("Background")
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._stage_ok and self._snapshot() == self._last_snapshot

    def save_to_config(self, cfg) -> None:
        cfg.background.count = int(self.count.value())
        cfg.background.method = "random" if self.random.isChecked() else "buffered"
        cfg.background.buffer_distance = self._buffer_distance_meters()

    def validatePage(self) -> bool:
        self.save_to_config(self.wizard_ref.config)
        return True
