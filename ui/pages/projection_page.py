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
from ...core.stages import validate_matching_bands
from ..page_utils import (
    configure_eda_table,
    fill_eda_table,
    raster_summary_text,
    wrap_scrollable,
)
from ..qgis_layers import clear_stage, show_rasters
from ..workers import StagePageMixin, snapshot_key


class ProjectionPage(StagePageMixin, QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Projection stack (optional)")
        self.setSubTitle(
            "Optionally supply a second raster stack matching the same predictors "
            "in the same order (e.g. future climate scenario). MESS (Multivariate "
            "Environmental Similarity Surface) and MOP (Mobility-Oriented Parity) "
            "layers will flag extrapolation — places where the projection's "
            "conditions fall outside what the model was actually trained on."
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

        self.run_btn = QPushButton("Validate")
        self.run_btn.setProperty("cls", "primary")
        self.run_btn.clicked.connect(self._on_run_clicked)
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
        layout.addWidget(QLabel("Projection rasters (leave empty to skip):"))
        layout.addWidget(self.list)
        layout.addLayout(btn_row)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.busy_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(QLabel("Per-raster summary (exploratory data analysis):"))
        layout.addWidget(self.table)
        wrap_scrollable(self, layout)

        self._stage_ok = False
        self._last_snapshot: str | None = None
        self.list.model().rowsInserted.connect(self.completeChanged)
        self.list.model().rowsRemoved.connect(self.completeChanged)

    def _add(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add projection rasters", "",
            "Rasters (*.tif *.tiff *.vrt *.img *.asc);;All files (*)",
        )
        for p in paths:
            self.list.addItem(p)

    def _remove(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def _clear(self) -> None:
        self.list.clear()
        # QListWidget.clear() resets the model internally rather than
        # removing rows one at a time, so it doesn't fire rowsRemoved — the
        # only signal wired to completeChanged above — leaving Next stuck
        # disabled even though isComplete() would now correctly return True.
        self.completeChanged.emit()

    def _paths(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]

    def _snapshot(self) -> str:
        wizard = self.wizard_ref
        return snapshot_key(tuple(self._paths()), wizard.session.stage_hashes.get("predictors"))

    def _on_run_clicked(self) -> None:
        self.error_label.setText("")
        paths = self._paths()
        if not paths:
            return
        wizard = self.wizard_ref
        train_stack = wizard.session.stack

        def _work():
            proj_stack = load_stack(paths)
            validate_matching_bands(train_stack, proj_stack)
            return proj_stack, describe_stack(proj_stack)

        self.run_stage_async(
            _work, self._on_done, self._on_failed,
            button=self.run_btn, busy_widget=self.busy_label,
        )

    def _on_done(self, result) -> None:
        proj_stack, eda = result
        sampled = fill_eda_table(self.table, eda)
        wizard = self.wizard_ref
        n_train = len(wizard.session.stack.names)
        summary = raster_summary_text(proj_stack)
        summary += f" Matches the training stack ({n_train} band(s))."
        if sampled:
            summary += " Statistics come from a sampled read of large rasters."
        self.summary_label.setText(summary)
        show_rasters(wizard.iface, "Projection", proj_stack)

        new_key = self._snapshot()
        old_key = wizard.session.stage_hashes.get("projection")
        if old_key is not None and old_key != new_key:
            wizard.invalidate_from(wizard.PAGE_PROJECTION)
        wizard.session.proj_stack = proj_stack
        wizard.session.stage_hashes["projection"] = new_key

        self._stage_ok = True
        self._last_snapshot = new_key

    def _on_failed(self, err: str) -> None:
        self._stage_ok = False
        self.table.setRowCount(0)
        self.summary_label.setText("")
        self.error_label.setText(err.splitlines()[0] if err else "Projection validation failed.")

    def invalidate(self) -> None:
        self._stage_ok = False
        self._last_snapshot = None
        self.table.setRowCount(0)
        self.summary_label.setText("")
        clear_stage("Projection")
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        if self.list.count() == 0:
            return True
        return self._stage_ok and self._snapshot() == self._last_snapshot

    def save_to_config(self, cfg) -> None:
        cfg.rasters.projection_paths = self._paths()

    def validatePage(self) -> bool:
        self.save_to_config(self.wizard_ref.config)
        return True
