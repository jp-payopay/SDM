from __future__ import annotations

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QPalette
from qgis.PyQt.QtWidgets import QMessageBox, QWizard

from ..core.config import SDMConfig
from ..core.session import PipelineSession
from .pages.algorithms_page import AlgorithmsPage
from .pages.background_page import BackgroundPage
from .pages.cleaning_page import CleaningPage
from .pages.ensemble_page import EnsemblePage
from .pages.occurrence_page import OccurrencePage
from .pages.output_page import OutputPage
from .pages.predictors_page import PredictorsPage
from .pages.projection_page import ProjectionPage
from .pages.run_page import RunPage
from .pages.split_page import SplitPage
from .pages.summary_page import SummaryPage
from .pages.vif_page import VIFPage
from .pages.welcome_page import WelcomePage
from .theme import APP_QSS, CANVAS, FOREST, RAISED
from .widgets.step_sidebar import WizardSidebar


class SDMWizard(QWizard):
    # Emits the core.pipeline.RunResult once a run completes, so UI outside
    # the wizard (e.g. the dock widget's status panel) can react without the
    # wizard needing to know that UI exists.
    run_completed = pyqtSignal(object)

    PAGE_WELCOME = 0
    PAGE_OCCURRENCE = 1
    PAGE_PREDICTORS = 2
    PAGE_CLEANING = 3
    PAGE_BACKGROUND = 4
    PAGE_VIF = 5
    PAGE_SPLIT = 6
    PAGE_ALGORITHMS = 7
    PAGE_PROJECTION = 8
    PAGE_ENSEMBLE = 9
    PAGE_OUTPUT = 10
    PAGE_RUN = 11
    PAGE_SUMMARY = 12

    # Stage pages whose live preview depends on an earlier page's settings.
    # Deliberately a small hardcoded map, not a generic dependency graph.
    # PAGE_RUN is included as a downstream of every stage page — a run's
    # result is stale the moment *any* upstream stage's settings change —
    # but note this is a secondary, immediate-UI-feedback mechanism only:
    # RunPage.isComplete() enforces the invariant itself (see its own
    # config-snapshot check), independent of whether invalidate_from() gets
    # called, since PAGE_ALGORITHMS/PAGE_ENSEMBLE/PAGE_OUTPUT never call it.
    DOWNSTREAM = {
        PAGE_OCCURRENCE: [PAGE_CLEANING, PAGE_BACKGROUND, PAGE_VIF, PAGE_SPLIT, PAGE_RUN],
        PAGE_PREDICTORS: [PAGE_CLEANING, PAGE_BACKGROUND, PAGE_VIF, PAGE_SPLIT, PAGE_PROJECTION, PAGE_RUN],
        PAGE_CLEANING: [PAGE_BACKGROUND, PAGE_VIF, PAGE_SPLIT, PAGE_RUN],
        PAGE_BACKGROUND: [PAGE_VIF, PAGE_SPLIT, PAGE_RUN],
        PAGE_VIF: [PAGE_SPLIT, PAGE_RUN],
        PAGE_SPLIT: [PAGE_RUN],
        PAGE_PROJECTION: [PAGE_RUN],
    }

    # PipelineSession fields owned by each stage page's preview computation.
    SESSION_FIELDS = {
        PAGE_OCCURRENCE: ("occ_raw",),
        PAGE_PREDICTORS: ("stack", "proj_stack"),
        PAGE_CLEANING: ("occ", "cleaning_report", "thinning_report"),
        PAGE_BACKGROUND: ("px", "py", "presence_flag", "X_full", "feature_names"),
        PAGE_VIF: ("X_kept", "kept_names", "kept_idx", "vif_report"),
        PAGE_SPLIT: (),
        PAGE_PROJECTION: ("proj_stack",),
    }

    # session.stage_hashes is keyed by stage name, not by page id or session
    # field name — this maps page id -> that stage-name key, so invalidation
    # can clear the right entry.
    STAGE_NAMES = {
        PAGE_OCCURRENCE: "occurrence",
        PAGE_PREDICTORS: "predictors",
        PAGE_CLEANING: "cleaning",
        PAGE_BACKGROUND: "background",
        PAGE_VIF: "vif",
        PAGE_SPLIT: None,
        PAGE_PROJECTION: "projection",
    }

    def __init__(self, parent=None, iface=None) -> None:
        super().__init__(parent)
        self.iface = iface
        self.config = SDMConfig()
        self.session = PipelineSession()
        self.setWindowTitle("SDM")
        # ClassicStyle deliberately, not ModernStyle: ModernStyle paints a
        # native top banner whose background color is derived from the
        # platform style + palette (not fully controllable via QSS), which
        # combined with our custom palette produced an unreadable dark band
        # behind the title text. ClassicStyle has no such banner — title and
        # subtitle render as plain text directly on the page's own
        # QSS-controlled background, guaranteeing readable contrast. Our step
        # sidebar is already the wizard's real chrome, so no banner is lost.
        self.setWizardStyle(QWizard.WizardStyle.ClassicStyle)
        self.setOption(QWizard.WizardOption.IndependentPages, False)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.setMinimumSize(900, 700)
        self.resize(980, 760)
        # Closing (X or Cancel) actually destroys the wizard instead of just
        # hiding it — otherwise plugin.py's reuse-if-still-open logic can
        # only find a hidden-but-alive window and can never show it again.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.setPage(self.PAGE_WELCOME, WelcomePage(self))
        self.setPage(self.PAGE_OCCURRENCE, OccurrencePage(self))
        self.setPage(self.PAGE_PREDICTORS, PredictorsPage(self))
        self.setPage(self.PAGE_CLEANING, CleaningPage(self))
        self.setPage(self.PAGE_BACKGROUND, BackgroundPage(self))
        self.setPage(self.PAGE_VIF, VIFPage(self))
        self.setPage(self.PAGE_SPLIT, SplitPage(self))
        self.setPage(self.PAGE_ALGORITHMS, AlgorithmsPage(self))
        self.setPage(self.PAGE_PROJECTION, ProjectionPage(self))
        self.setPage(self.PAGE_ENSEMBLE, EnsemblePage(self))
        self.setPage(self.PAGE_OUTPUT, OutputPage(self))
        self.setPage(self.PAGE_RUN, RunPage(self))
        self.setPage(self.PAGE_SUMMARY, SummaryPage(self))
        self.setStartId(self.PAGE_WELCOME)

        self._apply_style()

    def _apply_style(self) -> None:
        # QWizard/QDialog otherwise inherit whatever live application
        # palette QGIS is running (including a dark theme) rather than
        # rendering independently — this pins the wizard to its own
        # deliberate light "field guide" look regardless of QGIS's theme.
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(CANVAS))
        palette.setColor(QPalette.ColorRole.Base, QColor(RAISED))
        # ModernStyle derives its title-banner gradient from the Highlight
        # role, so this tints the native banner toward the same accent the
        # rest of the QSS below uses, without fighting the style engine.
        palette.setColor(QPalette.ColorRole.Highlight, QColor(FOREST))
        self.setPalette(palette)
        self.setStyleSheet(APP_QSS)

        steps = [
            (self.PAGE_WELCOME, "Welcome"),
            (self.PAGE_OCCURRENCE, "Occurrence data"),
            (self.PAGE_PREDICTORS, "Predictor rasters"),
            (self.PAGE_CLEANING, "Cleaning options"),
            (self.PAGE_BACKGROUND, "Background points"),
            (self.PAGE_VIF, "Predictor selection"),
            (self.PAGE_SPLIT, "Cross-validation"),
            (self.PAGE_ALGORITHMS, "Algorithms"),
            (self.PAGE_PROJECTION, "Projection stack"),
            (self.PAGE_ENSEMBLE, "Ensemble"),
            (self.PAGE_OUTPUT, "Output"),
            (self.PAGE_RUN, "Run"),
            (self.PAGE_SUMMARY, "Done"),
        ]
        self.setSideWidget(WizardSidebar(self, steps))

        for role in (QWizard.WizardButton.NextButton, QWizard.WizardButton.FinishButton):
            btn = self.button(role)
            if btn is not None:
                btn.setProperty("cls", "primary")

    def _running_page(self):
        """Return the first wizard page whose background QThread (a stage
        preview worker, or the RunPage's pipeline worker) is still executing,
        or None if nothing is running."""
        for pid in self.pageIds():
            page = self.page(pid)
            if page is None:
                continue
            is_running = getattr(page, "is_running", None) or getattr(page, "is_stage_running", None)
            if is_running is not None and is_running():
                return page
        return None

    def closeEvent(self, event) -> None:
        # WA_DeleteOnClose means a closed wizard is destroyed, along with
        # every page and any QThread it owns. Destroying a QThread while it
        # is still executing (a stage preview, or the pipeline run) is
        # undefined behavior and can crash QGIS mid-write to an output file
        # — so refuse to close until the running work finishes.
        if self._running_page() is not None:
            QMessageBox.warning(
                self,
                "Run in progress",
                "A background computation (stage preview or the pipeline run) is "
                "still in progress. Please wait for it to finish before closing "
                "this window.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def nextId(self) -> int:  # skip background page in PA mode
        cur = self.currentId()
        if cur == self.PAGE_CLEANING and self.config.data_mode == "presence_absence":
            return self.PAGE_VIF
        return super().nextId()

    def invalidate_from(self, page_id: int) -> None:
        """Called by a stage page's on-success handler when its own settings
        actually changed since its last successful run (detected by comparing
        against session.stage_hashes). Clears the PipelineSession fields
        owned by that stage and every downstream stage, and resets the
        preview state of each downstream page (whose instance stays alive
        across Back/Next, so it would otherwise keep reporting
        isComplete()=True from a now-stale run).
        """
        downstream = self.DOWNSTREAM.get(page_id, [])
        fields: list[str] = list(self.SESSION_FIELDS.get(page_id, ()))
        for pid in downstream:
            fields.extend(self.SESSION_FIELDS.get(pid, ()))
        if fields:
            self.session.invalidate(*fields)
        for pid in downstream:
            stage_name = self.STAGE_NAMES.get(pid)
            if stage_name:
                self.session.stage_hashes.pop(stage_name, None)
            page = self.page(pid)
            if page is not None and hasattr(page, "invalidate"):
                page.invalidate()
