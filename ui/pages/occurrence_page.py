from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWizardPage,
)

from ...core.io.occurrences import load_occurrences
from ..page_utils import wrap_scrollable
from ..qgis_layers import clear_stage, show_points
from ..widgets.embedded_canvas import EmbeddedPreviewCanvas
from ..workers import StagePageMixin, snapshot_key


class OccurrencePage(StagePageMixin, QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Occurrence data")
        self.setSubTitle(
            "Select a CSV or vector file with the observation points. "
            "Choose presence-only if the file lists only presences; presence/absence "
            "if it contains a column indicating presence (1) or absence (0)."
        )

        self.mode_po = QRadioButton("Presence-only")
        self.mode_pa = QRadioButton("Presence / absence")
        self.mode_po.setChecked(True)

        self.path_edit = QLineEdit()
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.browse_btn)

        self.x_field = QLineEdit("x")
        self.y_field = QLineEdit("y")
        self.presence_field = QLineEdit("")
        self.presence_field.setPlaceholderText("(only for presence/absence)")
        self.crs_edit = QLineEdit("EPSG:4326")
        self.layer_name = QLineEdit("")
        self.layer_name.setPlaceholderText("(optional: vector layer name)")

        form = QFormLayout()
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.mode_po)
        mode_row.addWidget(self.mode_pa)
        form.addRow(QLabel("Data mode:"), mode_row)
        form.addRow(QLabel("File:"), path_row)
        form.addRow(QLabel("X / longitude field:"), self.x_field)
        form.addRow(QLabel("Y / latitude field:"), self.y_field)
        form.addRow(QLabel("Presence field:"), self.presence_field)
        form.addRow(QLabel("CRS:"), self.crs_edit)
        form.addRow(QLabel("Vector layer name:"), self.layer_name)

        self.load_btn = QPushButton("Load && Preview")
        self.load_btn.setProperty("cls", "primary")
        self.load_btn.clicked.connect(self._on_load_clicked)
        self.busy_label = QLabel("Loading…")
        self.busy_label.setVisible(False)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #b00020;")
        self.error_label.setWordWrap(True)
        self.canvas = EmbeddedPreviewCanvas()

        outer = QVBoxLayout()
        outer.addLayout(form)
        outer.addWidget(self.load_btn)
        outer.addWidget(self.busy_label)
        outer.addWidget(self.result_label)
        outer.addWidget(self.error_label)
        outer.addWidget(self.canvas)
        outer.addStretch()
        wrap_scrollable(self, outer)

        self.registerField("occ_path*", self.path_edit)

        self._stage_ok = False
        self._last_snapshot: str | None = None
        for w in (self.path_edit, self.x_field, self.y_field, self.presence_field, self.crs_edit, self.layer_name):
            w.textChanged.connect(self.completeChanged)
        self.mode_po.toggled.connect(self.completeChanged)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select occurrence file",
            "",
            "Occurrence files (*.csv *.tsv *.txt *.shp *.geojson *.gpkg *.json);;All files (*)",
        )
        if path:
            self.path_edit.setText(path)

    def _snapshot(self) -> str:
        return snapshot_key(
            self.mode_po.isChecked(),
            self.path_edit.text().strip(),
            self.x_field.text().strip(),
            self.y_field.text().strip(),
            self.presence_field.text().strip(),
            self.crs_edit.text().strip(),
            self.layer_name.text().strip(),
        )

    def _on_load_clicked(self) -> None:
        self.error_label.setText("")
        path = self.path_edit.text().strip()
        if not path:
            return
        x_field = self.x_field.text().strip() or "x"
        y_field = self.y_field.text().strip() or "y"
        presence_field = self.presence_field.text().strip()
        crs = self.crs_edit.text().strip() or "EPSG:4326"
        layer_name = self.layer_name.text().strip()

        def _work():
            return load_occurrences(
                path,
                x_field=x_field,
                y_field=y_field,
                presence_field=presence_field,
                crs=crs,
                layer_name=layer_name,
            )

        self.run_stage_async(
            _work, self._on_loaded, self._on_load_failed,
            button=self.load_btn, busy_widget=self.busy_label,
        )

    def _on_loaded(self, occ) -> None:
        is_pa = self.mode_pa.isChecked()
        n = len(occ.x)
        labels = occ.presence if is_pa else None
        colors = {"1": "#2e7d32", "0": "#9e9e9e"} if is_pa else None
        if is_pa:
            n_pres = int((occ.presence == 1).sum())
            n_abs = n - n_pres
            self.result_label.setText(
                f"{n} points loaded: {n_pres} presence, {n_abs} absence. CRS: {occ.crs}."
            )
        else:
            self.result_label.setText(f"{n} points loaded. CRS: {occ.crs}.")
        self.canvas.set_points(occ.x, occ.y, labels=labels, crs=occ.crs)
        show_points(
            self.wizard_ref.iface, "Occurrence", "occurrence_points",
            occ.x, occ.y, labels=labels, colors=colors, crs=occ.crs,
        )

        new_key = self._snapshot()
        wizard = self.wizard_ref
        old_key = wizard.session.stage_hashes.get("occurrence")
        if old_key is not None and old_key != new_key:
            wizard.invalidate_from(wizard.PAGE_OCCURRENCE)
        wizard.session.occ_raw = occ
        wizard.session.stage_hashes["occurrence"] = new_key

        self._stage_ok = True
        self._last_snapshot = new_key

    def _on_load_failed(self, err: str) -> None:
        self._stage_ok = False
        self.error_label.setText(err.splitlines()[0] if err else "Failed to load occurrences.")

    def invalidate(self) -> None:
        self._stage_ok = False
        self._last_snapshot = None
        self.result_label.setText("")
        self.canvas.clear()
        clear_stage("Occurrence")
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return (
            bool(self.path_edit.text().strip())
            and self._stage_ok
            and self._snapshot() == self._last_snapshot
        )

    def save_to_config(self, cfg) -> None:
        cfg.data_mode = "presence_only" if self.mode_po.isChecked() else "presence_absence"
        cfg.occurrence.path = self.path_edit.text().strip()
        cfg.occurrence.x_field = self.x_field.text().strip() or "x"
        cfg.occurrence.y_field = self.y_field.text().strip() or "y"
        cfg.occurrence.presence_field = self.presence_field.text().strip()
        cfg.occurrence.crs = self.crs_edit.text().strip() or "EPSG:4326"
        cfg.occurrence.layer_name = self.layer_name.text().strip()

    def validatePage(self) -> bool:
        self.save_to_config(self.wizard_ref.config)
        return True
