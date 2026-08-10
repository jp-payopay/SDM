from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from qgis.PyQt.QtCore import QObject, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import QPushButton, QWidget

# Reusable background-execution pattern for wizard pages that need to run a
# core/ pipeline stage without freezing the GUI thread. Generalizes the
# _Worker/QThread pattern originally written for ui/pages/run_page.py.
#
# HARD RULE: the callable passed to StageWorker/run_stage_async runs on a
# worker thread. It must only touch core/ objects (numpy arrays, dataclasses,
# plain Python) — never a Qt widget and never an EmbeddedPreviewCanvas. All
# widget/canvas updates must happen inside the on_success/on_error callbacks,
# which run back on the GUI thread via the `finished`/`failed` signals.


def snapshot_key(*values: object) -> str:
    """Stable string key for a tuple of widget values, used both to gate
    isComplete() (same-page edit-after-run) and to detect real changes for
    cross-page cache invalidation (PipelineSession.stage_hashes)."""
    return repr(values)


class StageWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")
        else:
            self.finished.emit(result)


class StagePageMixin:
    """Mixin for QWizardPage subclasses that run a stage function off the
    GUI thread from a local action button, à la Wallace's per-module Run
    button. Expects the including class to also inherit QWizardPage (for
    `completeChanged`).
    """

    _stage_thread: QThread | None = None
    _stage_worker: StageWorker | None = None

    def run_stage_async(
        self,
        fn: Callable[[], Any],
        on_success: Callable[[Any], None],
        on_error: Callable[[str], None],
        *,
        button: QPushButton | None = None,
        busy_widget: QWidget | None = None,
    ) -> None:
        if self._stage_thread is not None:
            return  # a run is already in flight
        if button is not None:
            button.setEnabled(False)
        if busy_widget is not None:
            busy_widget.setVisible(True)

        self._stage_thread = QThread(self)
        self._stage_worker = StageWorker(fn)
        self._stage_worker.moveToThread(self._stage_thread)
        self._stage_thread.started.connect(self._stage_worker.run)

        def _finish() -> None:
            if button is not None:
                button.setEnabled(True)
            if busy_widget is not None:
                busy_widget.setVisible(False)
            self._cleanup_stage_thread()

        def _on_finished(result: Any) -> None:
            _finish()
            on_success(result)
            self.completeChanged.emit()

        def _on_failed(err: str) -> None:
            _finish()
            on_error(err)
            self.completeChanged.emit()

        self._stage_worker.finished.connect(_on_finished)
        self._stage_worker.failed.connect(_on_failed)
        self._stage_thread.start()

    def _cleanup_stage_thread(self) -> None:
        if self._stage_thread is not None:
            self._stage_thread.quit()
            self._stage_thread.wait()
            self._stage_thread = None
            self._stage_worker = None

    def is_stage_running(self) -> bool:
        """True while this page's background stage worker is executing.
        Checked by the wizard's closeEvent to avoid tearing down a page (and
        its QThread) while it still owns a running worker."""
        return self._stage_thread is not None and self._stage_thread.isRunning()
