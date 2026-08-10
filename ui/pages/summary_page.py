from __future__ import annotations

from pathlib import Path

from qgis.PyQt.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWizardPage,
)

from ..os_utils import open_path
from ..page_utils import wrap_scrollable
from ..widgets.embedded_canvas import EmbeddedPreviewCanvas


class SummaryPage(QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Done")
        self.setSubTitle("Results are written to your output directory.")
        self.header = QLabel("")
        self.report_btn = QPushButton("Open HTML report")
        self.report_btn.setProperty("cls", "primary")
        self.folder_btn = QPushButton("Open output folder")
        self.report_btn.clicked.connect(self._open_report)
        self.folder_btn.clicked.connect(self._open_folder)
        self.files_view = QPlainTextEdit()
        self.files_view.setReadOnly(True)
        self.canvas = EmbeddedPreviewCanvas()

        layout = QVBoxLayout()
        layout.addWidget(self.header)
        layout.addWidget(self.report_btn)
        layout.addWidget(self.folder_btn)
        layout.addWidget(QLabel("Ensemble suitability preview:"))
        layout.addWidget(self.canvas)
        layout.addWidget(QLabel("Output files:"))
        layout.addWidget(self.files_view)
        layout.addStretch()
        wrap_scrollable(self, layout)

    def initializePage(self) -> None:
        run_page = self.wizard_ref.page(self.wizard_ref.PAGE_RUN)
        result = getattr(run_page, "result", None)
        if result is None:
            self.header.setText("No results available.")
            return
        n_files = len(result.output_files)
        n_failed = len(result.failed_runs)
        self.header.setText(
            f"Run complete. {n_files} output files. "
            + (f"{n_failed} model runs failed (see report)." if n_failed else "All models succeeded.")
        )
        self.files_view.setPlainText("\n".join(result.output_files))
        self.report_btn.setEnabled(bool(result.report_path))

        ensemble_path = next(
            (f for f in result.output_files if Path(f).name == "ensemble_suitability.tif"),
            None,
        )
        if ensemble_path:
            self.canvas.set_raster(ensemble_path, name="ensemble_suitability")
        else:
            self.canvas.clear()

    def _open_report(self) -> None:
        run_page = self.wizard_ref.page(self.wizard_ref.PAGE_RUN)
        result = getattr(run_page, "result", None)
        if result and result.report_path:
            open_path(result.report_path)

    def _open_folder(self) -> None:
        run_page = self.wizard_ref.page(self.wizard_ref.PAGE_RUN)
        result = getattr(run_page, "result", None)
        if result:
            open_path(result.output_dir)
