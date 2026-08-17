"""Shared behaviour for the two wizard pages that load a raster stack —
the predictor rasters and the optional projection stack.

Both do the same three things with a list of raster paths: validate that they
form one stack, show either exploratory statistics (when they do) or each
layer's grid properties (when they don't), and offer to resample them onto a
common grid. That middle-and-last part lives here so the two pages can't drift
apart; each page keeps its own widgets and its own wizard-session wiring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from qgis.PyQt.QtWidgets import QDialog

from ..core.io.align import align_rasters
from ..core.io.rasters import (
    AlignmentIssues,
    RasterProfile,
    RasterStack,
    build_stack,
    check_unique_stems,
    describe_profiles,
    describe_stack,
    diagnose_alignment,
)
from .page_utils import (
    EDA_TABLE_CAPTION,
    PROFILE_TABLE_CAPTION,
    alignment_headline,
    configure_eda_table,
    configure_profile_table,
    fill_eda_table,
    fill_profile_table,
    raster_summary_text,
)
from .widgets.fix_rasters_dialog import FixRastersDialog
from .workers import StagePageMixin


@dataclass
class StackOutcome:
    """What a load attempt found. `stack` is None precisely when the rasters
    don't form one grid — the page then shows `profiles` instead of `eda`."""

    profiles: list[RasterProfile]
    issues: AlignmentIssues
    stack: RasterStack | None = None
    eda: list = field(default_factory=list)

    @property
    def aligned(self) -> bool:
        return self.stack is not None


def load_outcome(
    paths: list[str],
    extra_validate: Callable[[RasterStack], None] | None = None,
) -> StackOutcome:
    """Read the rasters and decide whether they stack. Runs on a worker
    thread (see workers.StagePageMixin), so it touches no widgets.

    A misalignment is returned rather than raised: it is the one failure the
    wizard can walk the user out of, and doing so needs every layer's
    properties, not just the first offender's. Everything else (a missing
    file, duplicate predictor names, an unreadable raster) still raises.
    """
    check_unique_stems(paths)
    profiles = describe_profiles(paths)
    issues = diagnose_alignment(profiles)
    if issues.any:
        return StackOutcome(profiles, issues)
    stack = build_stack(profiles)
    if extra_validate is not None:
        extra_validate(stack)
    return StackOutcome(profiles, issues, stack, describe_stack(stack))


class RasterStackPageMixin(StagePageMixin):
    """Table-switching and the fix-layers flow, for a page that provides:

    `list` (QListWidget of paths), `table` (QTableWidget), `table_label`,
    `summary_label`, `error_label`, `busy_label`, `progress_bar`, `load_btn`
    and `fix_btn`, plus a `_start_load()` that kicks off its own validation
    run.
    """

    #: Wording for this page's layers, e.g. "predictor" -> "Fix predictor layers".
    kind = "predictor"
    #: Text of the page's busy label while a validation run is in flight.
    BUSY_TEXT = "Validating…"
    #: Carried from a completed fix into the next result view, which would
    #: otherwise overwrite it — the fix triggers a reload immediately.
    _fix_note = ""
    #: Layer count of the fix currently running, for its progress messages.
    _fix_total = 0

    @property
    def fix_action(self) -> str:
        return f"Fix {self.kind} layers"

    def _paths(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]

    def set_fix_available(self, available: bool) -> None:
        """The fix button is an answer to a specific diagnosis, so it stays
        disabled until a validation run has actually found rasters that don't
        line up — and goes back off the moment they do (or the list changes,
        making the last diagnosis stale)."""
        self.fix_btn.setEnabled(available)

    def _take_fix_note(self) -> str:
        note, self._fix_note = self._fix_note, ""
        return note

    # ----- results -----

    def show_alignment_problem(self, outcome: StackOutcome) -> None:
        """Replace the statistics view with one row per layer, so the offending
        property is visible at a glance instead of having to be inferred from
        an error message."""
        configure_profile_table(self.table)
        fill_profile_table(self.table, outcome.profiles)
        self.table_label.setText(PROFILE_TABLE_CAPTION)
        self.summary_label.setText(self._take_fix_note())
        self.error_label.setText(alignment_headline(outcome.issues, self.fix_action))
        self.set_fix_available(True)

    def show_eda(self, outcome: StackOutcome, extra_summary: str = "") -> None:
        self.set_fix_available(False)
        configure_eda_table(self.table)
        sampled = fill_eda_table(self.table, outcome.eda)
        self.table_label.setText(EDA_TABLE_CAPTION)
        summary = raster_summary_text(outcome.stack) + extra_summary
        if sampled:
            summary += " Statistics come from a sampled read of large rasters."
        self.summary_label.setText(summary + self._take_fix_note())
        self.error_label.setText("")

    def clear_results(self) -> None:
        self.table.setRowCount(0)
        self.summary_label.setText("")
        self.table_label.setText(EDA_TABLE_CAPTION)
        self._fix_note = ""
        # Whatever the last run concluded no longer holds — a load error, an
        # edited path list, or an upstream invalidation all leave the page
        # with nothing diagnosed to fix.
        self.set_fix_available(False)

    # ----- running work -----

    def run_raster_stage(
        self, work, on_success, on_error, *, button, busy_widget=None, on_progress=None
    ) -> None:
        """run_stage_async, with the page's *other* action button disabled for
        the duration. Validating and fixing both rewrite the same state, so
        only one of them may ever be in flight (run_stage_async by itself
        only manages the one button it is handed)."""
        if self._stage_thread is not None:
            # run_stage_async would refuse below anyway; returning first keeps
            # the button changes from being applied with no callback ever
            # coming back to undo them.
            return
        other = self.fix_btn if button is self.load_btn else self.load_btn
        was_enabled = other.isEnabled()
        other.setEnabled(False)

        def _release(callback):
            def wrapped(payload):
                # Restore rather than enable: the fix button's availability is
                # decided by the last validation result, not by this run.
                other.setEnabled(was_enabled)
                callback(payload)

            return wrapped

        self.run_stage_async(
            work, _release(on_success), _release(on_error),
            button=button,
            busy_widget=busy_widget if busy_widget is not None else self.busy_label,
            on_progress=on_progress,
        )

    # ----- fix flow -----

    def on_fix_clicked(self) -> None:
        paths = self._paths()
        if not paths or self.is_stage_running():
            return
        self.error_label.setText("")
        try:
            profiles = describe_profiles(paths)
        except Exception as exc:
            message = str(exc)
            self.error_label.setText(
                message.splitlines()[0] if message else "Could not read the rasters."
            )
            return

        dialog = FixRastersDialog(profiles, kind=self.kind, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        target = dialog.target
        out_dir = dialog.out_dir
        outputs = dialog.outputs
        assumed_crs = dialog.assumed_crs

        def _work(report):
            return align_rasters(
                profiles, target, out_dir,
                outputs=outputs, assumed_crs=assumed_crs, progress=report,
            )

        self._fix_total = len(profiles)
        self.progress_bar.setRange(0, self._fix_total)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Resampling… %v of %m layers done")
        self.run_raster_stage(
            _work, self._on_fixed, self._on_fix_failed,
            button=self.fix_btn,
            busy_widget=self.progress_bar,
            # A bound method of this page (a QObject), never a closure: that is
            # what makes Qt queue the worker thread's reports onto the GUI
            # thread before this touches a widget.
            on_progress=self._on_fix_progress,
        )

    def _on_fix_progress(self, fraction: float, _name: str) -> None:
        """Advance the bar as each layer lands. Warping a full-resolution
        climate stack is minutes of work, so a static indicator would look
        indistinguishable from a hang."""
        self.progress_bar.setValue(round(fraction * self._fix_total))

    def _on_fixed(self, new_paths: list[str]) -> None:
        self.list.clear()
        for path in new_paths:
            self.list.addItem(path)
        self.set_fix_available(False)
        self._fix_note = (
            f" Resampled copies of {len(new_paths)} raster(s) were written to "
            f"{Path(new_paths[0]).parent}; the list now points at those, and your "
            "original files are unchanged."
        )
        self.completeChanged.emit()
        # Validate the new files straight away: the point of the fix is that
        # the page ends up in a loaded, ready-to-continue state.
        self._start_load()

    def _on_fix_failed(self, err: str) -> None:
        self.error_label.setText(
            err.splitlines()[0] if err else "Could not resample the rasters."
        )
