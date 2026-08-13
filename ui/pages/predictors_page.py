from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWizardPage,
)

from ...core.io.rasters import describe_stack, load_stack
from ..page_utils import (
    configure_eda_table,
    fill_eda_table,
    raster_summary_text,
    wrap_scrollable,
)
from ..qgis_layers import clear_stage, show_rasters
from ..workers import StagePageMixin, snapshot_key


class PredictorsPage(StagePageMixin, QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Predictor rasters")
        self.setSubTitle(
            "Add one or more raster predictors. They must share the same CRS "
            "(Coordinate Reference System), extent, resolution, and grid. Load and "
            "Validate will report a clear error otherwise."
        )
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

        add_btn = QPushButton("Add rasters...")
        rem_btn = QPushButton("Remove selected")
        clr_btn = QPushButton("Clear all")
        add_btn.clicked.connect(self._add)
        rem_btn.clicked.connect(self._remove)
        clr_btn.clicked.connect(self._clear)

        btn_row = QHBoxLayout()
        btn_row.addWidget(add_btn)
        btn_row.addWidget(rem_btn)
        btn_row.addWidget(clr_btn)

        self.load_btn = QPushButton("Load && Validate")
        self.load_btn.setProperty("cls", "primary")
        self.load_btn.clicked.connect(self._on_load_clicked)
        self.busy_label = QLabel("Validating...")
        self.busy_label.setVisible(False)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #b00020;")
        self.error_label.setWordWrap(True)
        self.table = QTableWidget()
        configure_eda_table(self.table)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Predictor rasters:"))
        layout.addWidget(self.list)
        layout.addLayout(btn_row)
        layout.addWidget(self.load_btn)
        layout.addWidget(self.busy_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(QLabel("Per-raster summary (exploratory data analysis):"))
        layout.addWidget(self.table)
        layout.addStretch()
        wrap_scrollable(self, layout)

        self._stage_ok = False
        self._last_snapshot: str | None = None

    def _add(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add predictor rasters", "",
            "Rasters (*.tif *.tiff *.vrt *.img *.asc);;All files (*)",
        )
        for p in paths:
            self.list.addItem(p)
        self.completeChanged.emit()

    def _remove(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))
        self.completeChanged.emit()

    def _clear(self) -> None:
        self.list.clear()
        self.completeChanged.emit()

    def _paths(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]

    def _snapshot(self) -> str:
        return snapshot_key(tuple(self._paths()))

    def _on_load_clicked(self) -> None:
        self.error_label.setText("")
        paths = self._paths()
        if not paths:
            return

        def _work():
            stack = load_stack(paths)
            return stack, describe_stack(stack)

        self.run_stage_async(
            _work, self._on_loaded, self._on_load_failed,
            button=self.load_btn, busy_widget=self.busy_label,
        )

    def _on_loaded(self, result) -> None:
        stack, eda = result
        sampled = fill_eda_table(self.table, eda)
        summary = raster_summary_text(stack)
        if sampled:
            summary += " Statistics come from a sampled read of large rasters."
        self.summary_label.setText(summary)
        show_rasters(self.wizard_ref.iface, "Predictors", stack)

        new_key = self._snapshot()
        wizard = self.wizard_ref
        old_key = wizard.session.stage_hashes.get("predictors")
        if old_key is not None and old_key != new_key:
            wizard.invalidate_from(wizard.PAGE_PREDICTORS)
        wizard.session.stack = stack
        wizard.session.stage_hashes["predictors"] = new_key

        self._stage_ok = True
        self._last_snapshot = new_key

    def _on_load_failed(self, err: str) -> None:
        self._stage_ok = False
        self.table.setRowCount(0)
        self.summary_label.setText("")
        self.error_label.setText(err.splitlines()[0] if err else "Failed to load rasters.")

    def invalidate(self) -> None:
        self._stage_ok = False
        self._last_snapshot = None
        self.table.setRowCount(0)
        self.summary_label.setText("")
        clear_stage("Predictors")
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return (
            self.list.count() > 0
            and self._stage_ok
            and self._snapshot() == self._last_snapshot
        )

    def save_to_config(self, cfg) -> None:
        cfg.rasters.paths = self._paths()

    def validatePage(self) -> bool:
        self.save_to_config(self.wizard_ref.config)
        return True
