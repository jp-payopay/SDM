from __future__ import annotations

import json
import time
import traceback
from datetime import datetime

from qgis.PyQt.QtCore import QObject, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWizardPage,
)

from ...core.config import SDMConfig
from ...core.pipeline import Pipeline, format_duration
from ..page_utils import wrap_scrollable
from ..qgis_layers import load_run_outputs


class _Worker(QObject):
    progress = pyqtSignal(str, float, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config: SDMConfig, session=None) -> None:
        super().__init__()
        self.config = config
        self.session = session

    def run(self) -> None:
        try:
            pipe = Pipeline(
                self.config,
                progress=lambda s, f, m: self.progress.emit(s, f, m),
                session=self.session,
            )
            result = pipe.run()
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


class RunPage(QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Running the pipeline")
        self.setSubTitle(
            "Click 'Start' to begin. Loading, cleaning, background sampling, and VIF "
            "(multicollinearity) selection reuse the results already previewed on "
            "earlier pages, so only modeling, ensembling, and reporting run fresh here."
        )
        self.result = None
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._start_time: float | None = None
        # Snapshot of wizard.config as of the last successful run. Compared
        # against the *current* config in isComplete() — this is the
        # authoritative staleness check (not just a UI nicety): unlike the
        # stage-preview pages, PAGE_ALGORITHMS/PAGE_ENSEMBLE/PAGE_OUTPUT never
        # call wizard.invalidate_from(), so a full-config diff is the only
        # thing that catches "changed replicates/ensemble method/output dir
        # after a run already completed" and blocks reaching Summary stale.
        self._last_run_snapshot: str | None = None

        self.start_btn = QPushButton("Start")
        self.start_btn.setProperty("cls", "primary")
        self.start_btn.clicked.connect(self._start)
        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        self.stage_lbl = QLabel("Idle.")
        self.runtime_lbl = QLabel("")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stage_lbl)
        layout.addWidget(self.pbar)
        layout.addWidget(self.runtime_lbl)
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.log)
        wrap_scrollable(self, layout)

    def initializePage(self) -> None:
        # Re-enable Start whenever the page becomes current and the last
        # result is stale (isComplete()==False) — this is the only thing
        # that unblocks the button for settings pages (Algorithms/Ensemble/
        # Output/Split) that don't go through wizard.invalidate_from(), so
        # without this a completed run followed by any such change leaves
        # both Next (correctly) and Start disabled with no way to recover
        # short of restarting the wizard. Guarded by is_running() so this
        # can't re-enable Start while a run is still actually in flight.
        if not self.isComplete() and not self.is_running():
            self.start_btn.setEnabled(True)

    def _start(self) -> None:
        if self.is_running():
            return
        self.start_btn.setEnabled(False)
        self.runtime_lbl.setText("")
        self._start_time = time.monotonic()
        self._append("Starting pipeline…")
        self._thread = QThread(self)
        self._worker = _Worker(self.wizard_ref.config, session=self.wizard_ref.session)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _on_progress(self, stage: str, fraction: float, message: str) -> None:
        self.pbar.setValue(int(round(fraction * 100)))
        self.stage_lbl.setText(f"[{stage}]  {message}")
        self._append(f"[{stage}] {message}")

    def _snapshot(self) -> str:
        return json.dumps(self.wizard_ref.config.to_dict(), sort_keys=True, default=str)

    def _on_finished(self, result) -> None:
        self.result = result
        self._last_run_snapshot = self._snapshot()
        # result.duration_seconds is core's own measurement (Pipeline._t0),
        # the same number the HTML report shows — used here instead of timing
        # independently in the UI so the two never drift apart.
        runtime = format_duration(result.duration_seconds)
        self.runtime_lbl.setText(f"Total runtime: {runtime}")
        self._append(f"Pipeline complete. Total runtime: {runtime}.")
        load_run_outputs(self.wizard_ref.iface, result)
        self.wizard_ref.run_completed.emit(result)
        self._cleanup()
        self.completeChanged.emit()
        self.wizard_ref.next()

    def _on_failed(self, err: str) -> None:
        # No RunResult on failure, so this is the UI's own wall-clock timer
        # (started in _start()) rather than core's duration_seconds.
        if self._start_time is not None:
            runtime = format_duration(time.monotonic() - self._start_time)
            self.runtime_lbl.setText(f"Failed after: {runtime}")
            self._append(f"FAILED after {runtime}:\n" + err)
        else:
            self._append("FAILED:\n" + err)
        self._cleanup()
        self.start_btn.setEnabled(True)

    def _cleanup(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    def _append(self, msg: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{stamp}] {msg}")

    def isComplete(self) -> bool:
        return (
            self.result is not None
            and self._last_run_snapshot is not None
            and self._snapshot() == self._last_run_snapshot
        )

    def invalidate(self) -> None:
        """Called by wizard.invalidate_from() when an earlier stage page's
        settings actually changed. Resets the displayed run state back to
        idle immediately, instead of leaving a stale progress bar/log/result
        visible until the user notices Next is greyed out."""
        self.result = None
        self._last_run_snapshot = None
        self._start_time = None
        self.pbar.setValue(0)
        self.stage_lbl.setText("Idle.")
        self.runtime_lbl.setText("")
        self.log.clear()
        self.start_btn.setEnabled(True)
        self.completeChanged.emit()

    def is_running(self) -> bool:
        """True while the pipeline worker thread is actively executing.
        Checked by the wizard's closeEvent to avoid tearing down this page
        (and its QThread) while it still owns a running pipeline."""
        return self._thread is not None and self._thread.isRunning()
