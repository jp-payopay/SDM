from __future__ import annotations

import subprocess
import traceback

from qgis.PyQt.QtCore import QObject, Qt, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from ..theme import APP_QSS


class _PipInstallWorker(QObject):
    """Runs `<python_exe> -m pip install <packages>` and streams stdout line
    by line, so a multi-minute install (e.g. xgboost) doesn't look frozen.
    `python_exe` must be a real Python interpreter, not sys.executable —
    inside QGIS's embedded interpreter (notably the Windows standalone
    installer) sys.executable is the QGIS binary itself, and spawning it
    directly launches a second QGIS instance instead of running pip.
    """

    line = pyqtSignal(str)
    finished = pyqtSignal(int)
    failed = pyqtSignal(str)

    def __init__(self, python_exe: str, packages: list[str]) -> None:
        super().__init__()
        self._python_exe = python_exe
        self._packages = packages

    def run(self) -> None:
        try:
            proc = subprocess.Popen(
                [self._python_exe, "-m", "pip", "install", *self._packages],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")
            return
        assert proc.stdout is not None
        for line in proc.stdout:
            self.line.emit(line.rstrip("\n"))
        self.finished.emit(proc.wait())


class DependencyInstallDialog(QDialog):
    """Shown when SDM's Python dependencies are missing. Offers a
    one-click install of the missing packages into QGIS's own Python
    environment. `installed_ok` is set True once pip exits 0 — the caller
    is responsible for re-checking imports afterward, since a freshly
    installed native extension can still need a QGIS restart to load
    cleanly even when pip itself succeeded.
    """

    def __init__(self, parent, message: str, missing_packages: list[str], python_exe: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("SDM dependency problem")
        self.setMinimumWidth(520)
        self.installed_ok = False
        self._missing = missing_packages
        self._python_exe = python_exe
        self._thread: QThread | None = None
        self._worker: _PipInstallWorker | None = None

        self.message_label = QLabel(message)
        self.message_label.setWordWrap(True)
        self.message_label.setTextFormat(Qt.TextFormat.PlainText)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setVisible(False)
        self.log.setMaximumHeight(180)

        self.install_btn = QPushButton(f"Install missing package(s) ({', '.join(missing_packages)})")
        self.install_btn.setProperty("cls", "primary")
        self.install_btn.setVisible(bool(missing_packages))
        self.install_btn.clicked.connect(self._on_install_clicked)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.install_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.close_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.message_label)
        layout.addWidget(self.log)
        layout.addLayout(btn_row)
        self.setLayout(layout)
        self.setStyleSheet(APP_QSS)

    def _on_install_clicked(self) -> None:
        self.install_btn.setEnabled(False)
        self.log.setVisible(True)
        self.log.appendPlainText(f"Running: {self._python_exe} -m pip install {' '.join(self._missing)}\n")

        self._thread = QThread(self)
        self._worker = _PipInstallWorker(self._python_exe, list(self._missing))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.line.connect(self.log.appendPlainText)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _on_finished(self, returncode: int) -> None:
        self._cleanup_thread()
        if returncode == 0:
            self.log.appendPlainText("\nInstall finished successfully.")
            self.installed_ok = True
            self.accept()  # closes the dialog; caller re-checks imports and proceeds
        else:
            self.log.appendPlainText(f"\npip exited with code {returncode}. See the output above.")
            self.install_btn.setEnabled(True)

    def _on_failed(self, err: str) -> None:
        self.log.appendPlainText("\nFailed to run pip:\n" + err)
        self.install_btn.setEnabled(True)
        self._cleanup_thread()

    def _cleanup_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    def closeEvent(self, event) -> None:
        # Let a running install finish rather than killing pip mid-write.
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        super().closeEvent(event)
