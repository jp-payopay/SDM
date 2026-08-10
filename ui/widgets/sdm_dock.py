from __future__ import annotations

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..os_utils import open_path
from ..theme import APP_QSS


class SDMDockWidget(QDockWidget):
    """Launcher + status panel for the View -> Panels dock area. Pure UI —
    knows nothing about dependency checking or how the wizard is launched;
    the plugin wires launch_requested to its own launch logic, keeping this
    widget reusable/testable on its own.
    """

    launch_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("SDM", parent)
        self.setObjectName("SDMDockWidget")  # lets QGIS persist dock position across sessions

        self._last_result = None

        self.run_btn = QPushButton("Run SDM…")
        self.run_btn.setProperty("cls", "primary")
        self.run_btn.clicked.connect(self.launch_requested.emit)

        self.status_box = QGroupBox("Last run")
        self.status_box.setVisible(False)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.report_btn = QPushButton("Open HTML report")
        self.report_btn.setProperty("cls", "primary")
        self.folder_btn = QPushButton("Open output folder")
        self.report_btn.clicked.connect(self._open_report)
        self.folder_btn.clicked.connect(self._open_folder)
        status_layout = QVBoxLayout()
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.report_btn)
        status_layout.addWidget(self.folder_btn)
        self.status_box.setLayout(status_layout)

        layout = QVBoxLayout()
        layout.addWidget(self.run_btn)
        layout.addWidget(self.status_box)
        layout.addStretch()

        container = QWidget()
        container.setLayout(layout)
        self.setWidget(container)
        self.setStyleSheet(APP_QSS)

    def set_last_result(self, result) -> None:
        self._last_result = result
        n_files = len(result.output_files)
        n_failed = len(result.failed_runs)
        self.status_label.setText(
            f"Output: {result.output_dir}\n{n_files} file(s). "
            + (f"{n_failed} model run(s) failed." if n_failed else "All models succeeded.")
        )
        self.report_btn.setEnabled(bool(result.report_path))
        self.status_box.setVisible(True)

    def _open_report(self) -> None:
        if self._last_result is not None and self._last_result.report_path:
            open_path(self._last_result.report_path)

    def _open_folder(self) -> None:
        if self._last_result is not None:
            open_path(self._last_result.output_dir)
