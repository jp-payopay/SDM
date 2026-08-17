from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWizardPage,
)

from ...core.stages import validate_matching_bands
from ..page_utils import EDA_TABLE_CAPTION, configure_eda_table, wrap_scrollable
from ..qgis_layers import clear_stage, show_rasters
from ..raster_stage import RasterStackPageMixin, load_outcome
from ..workers import snapshot_key


class ProjectionPage(RasterStackPageMixin, QWizardPage):
    kind = "projection"

    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Projection stack (optional)")
        self.setSubTitle(
            "Optionally supply a second raster stack matching the same predictors "
            "in the same order (e.g. future climate scenario). MESS (Multivariate "
            "Environmental Similarity Surface) and MOP (Mobility-Oriented Parity) "
            "layers will flag extrapolation, meaning places where the projection's "
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

        self.load_btn = QPushButton("Validate")
        self.load_btn.setProperty("cls", "primary")
        self.load_btn.clicked.connect(self._on_run_clicked)
        self.fix_btn = QPushButton("Fix projection layers…")
        self.fix_btn.setProperty("cls", "primary")
        self.fix_btn.setToolTip(
            "Available once Validate finds rasters that don't line up. "
            "Resamples every listed raster onto one common CRS, extent and "
            "resolution, writing new files and leaving the originals untouched."
        )
        self.fix_btn.clicked.connect(self.on_fix_clicked)
        action_row = QHBoxLayout()
        action_row.addWidget(self.load_btn)
        action_row.addWidget(self.fix_btn)

        self.busy_label = QLabel(self.BUSY_TEXT)
        self.busy_label.setVisible(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #b00020;")
        self.error_label.setWordWrap(True)
        self.table_label = QLabel(EDA_TABLE_CAPTION)
        self.table_label.setWordWrap(True)
        self.table = QTableWidget()
        configure_eda_table(self.table)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Projection rasters (leave empty to skip):"))
        layout.addWidget(self.list)
        layout.addLayout(btn_row)
        layout.addLayout(action_row)
        layout.addWidget(self.busy_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.error_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.table_label)
        layout.addWidget(self.table)
        wrap_scrollable(self, layout)

        self._stage_ok = False
        self._last_snapshot: str | None = None
        self.set_fix_available(False)
        self.list.model().rowsInserted.connect(self.completeChanged)
        self.list.model().rowsRemoved.connect(self.completeChanged)

    def _add(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add projection rasters", "",
            "Rasters (*.tif *.tiff *.vrt *.img *.asc);;All files (*)",
        )
        for p in paths:
            self.list.addItem(p)
        self.set_fix_available(False)

    def _remove(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))
        self.set_fix_available(False)

    def _clear(self) -> None:
        self.list.clear()
        self.set_fix_available(False)
        # QListWidget.clear() resets the model internally rather than
        # removing rows one at a time, so it doesn't fire rowsRemoved — the
        # only signal wired to completeChanged above — leaving Next stuck
        # disabled even though isComplete() would now correctly return True.
        self.completeChanged.emit()

    def _snapshot(self) -> str:
        wizard = self.wizard_ref
        return snapshot_key(tuple(self._paths()), wizard.session.stage_hashes.get("predictors"))

    def _on_run_clicked(self) -> None:
        self.error_label.setText("")
        self._start_load()

    def _start_load(self) -> None:
        paths = self._paths()
        if not paths:
            return
        train_stack = self.wizard_ref.session.stack

        def _work():
            return load_outcome(
                paths,
                extra_validate=lambda stack: validate_matching_bands(train_stack, stack),
            )

        self.run_raster_stage(
            _work, self._on_done, self._on_failed, button=self.load_btn,
        )

    def _on_done(self, outcome) -> None:
        if not outcome.aligned:
            self._stage_ok = False
            self._last_snapshot = None
            self.show_alignment_problem(outcome)
            return

        wizard = self.wizard_ref
        n_train = len(wizard.session.stack.names)
        self.show_eda(outcome, f" Matches the training stack ({n_train} band(s)).")
        proj_stack = outcome.stack
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
        self.clear_results()
        self.error_label.setText(err.splitlines()[0] if err else "Projection validation failed.")

    def invalidate(self) -> None:
        self._stage_ok = False
        self._last_snapshot = None
        self.clear_results()
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
