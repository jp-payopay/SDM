from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWizardPage,
)

from ...core.stages import stage_clean
from ..page_utils import wrap_scrollable
from ..qgis_layers import clear_stage, show_points
from ..widgets.embedded_canvas import EmbeddedPreviewCanvas
from ..workers import StagePageMixin, snapshot_key


class CleaningPage(StagePageMixin, QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Cleaning options")
        self.setSubTitle(
            "Automatic cleaning drops NaN (Not a Number, i.e. missing coordinates), "
            "duplicate, (0, 0), out-of-extent, and nodata-cell points. Spatial "
            "thinning keeps at most one point per raster pixel."
        )
        self.auto_clean = QCheckBox("Apply automatic coordinate cleaning")
        self.auto_clean.setChecked(True)
        self.thin = QCheckBox("Thin occurrences to raster resolution")
        self.thin.setChecked(True)

        self.run_btn = QPushButton("Run Cleaning")
        self.run_btn.setProperty("cls", "primary")
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.busy_label = QLabel("Cleaning…")
        self.busy_label.setVisible(False)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #b00020;")
        self.error_label.setWordWrap(True)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Rule", "Points dropped"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.canvas = EmbeddedPreviewCanvas()

        layout = QVBoxLayout()
        layout.addWidget(self.auto_clean)
        layout.addWidget(self.thin)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.busy_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.table)
        layout.addWidget(self.canvas)
        layout.addStretch()
        wrap_scrollable(self, layout)

        self._stage_ok = False
        self._last_snapshot: str | None = None
        self.auto_clean.toggled.connect(self.completeChanged)
        self.thin.toggled.connect(self.completeChanged)

    def _snapshot(self) -> str:
        wizard = self.wizard_ref
        return snapshot_key(
            self.auto_clean.isChecked(),
            self.thin.isChecked(),
            wizard.session.stage_hashes.get("occurrence"),
            wizard.session.stage_hashes.get("predictors"),
        )

    def _on_run_clicked(self) -> None:
        self.error_label.setText("")
        wizard = self.wizard_ref
        self.save_to_config(wizard.config)
        cfg = wizard.config
        occ_raw = wizard.session.occ_raw
        stack = wizard.session.stack

        def _work():
            return stage_clean(cfg, occ_raw, stack)

        self.run_stage_async(
            _work, self._on_done, self._on_failed,
            button=self.run_btn, busy_widget=self.busy_label,
        )

    def _on_done(self, result) -> None:
        occ, cleaning_rep, thinning_rep = result
        wizard = self.wizard_ref
        n_before = len(wizard.session.occ_raw.x)
        n_after = len(occ.x)
        self.summary_label.setText(f"Before: {n_before} points. After: {n_after} points.")

        rows: list[tuple[str, int]] = []
        if cleaning_rep:
            rows.extend(cleaning_rep.dropped.items())
        if thinning_rep and thinning_rep.n_removed:
            rows.append(("thinned (duplicate pixel)", thinning_rep.n_removed))
        self.table.setRowCount(len(rows))
        for row, (label, count) in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(label))
            self.table.setItem(row, 1, QTableWidgetItem(str(count)))

        if cleaning_rep is not None:
            px, py = wizard.session.occ_raw.x, wizard.session.occ_raw.y
            labels, colors, crs = cleaning_rep.kept_mask, {"True": "#2e7d32", "False": "#c62828"}, wizard.session.occ_raw.crs
        else:
            px, py = occ.x, occ.y
            labels, colors, crs = None, None, occ.crs
        self.canvas.set_points(px, py, labels=labels, colors=colors, crs=crs)
        show_points(wizard.iface, "Cleaning", "cleaned_points", px, py, labels=labels, colors=colors, crs=crs)

        new_key = self._snapshot()
        old_key = wizard.session.stage_hashes.get("cleaning")
        if old_key is not None and old_key != new_key:
            wizard.invalidate_from(wizard.PAGE_CLEANING)
        wizard.session.occ = occ
        wizard.session.cleaning_report = cleaning_rep
        wizard.session.thinning_report = thinning_rep
        wizard.session.stage_hashes["cleaning"] = new_key

        self._stage_ok = True
        self._last_snapshot = new_key

    def _on_failed(self, err: str) -> None:
        self._stage_ok = False
        self.table.setRowCount(0)
        self.summary_label.setText("")
        self.error_label.setText(err.splitlines()[0] if err else "Cleaning failed.")

    def invalidate(self) -> None:
        self._stage_ok = False
        self._last_snapshot = None
        self.table.setRowCount(0)
        self.summary_label.setText("")
        self.canvas.clear()
        clear_stage("Cleaning")
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._stage_ok and self._snapshot() == self._last_snapshot

    def save_to_config(self, cfg) -> None:
        cfg.cleaning.auto_clean = self.auto_clean.isChecked()
        cfg.cleaning.thin_to_raster_resolution = self.thin.isChecked()

    def validatePage(self) -> bool:
        self.save_to_config(self.wizard_ref.config)
        return True
