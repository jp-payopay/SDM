from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..theme import SIDEBAR_QSS


class WizardSidebar(QWidget):
    """Persistent step list for QWizard.setSideWidget() — the wizard's one
    deliberately bold visual element. This encodes real information (where
    you are in a genuinely sequential 13-step pipeline, and which steps are
    actually complete), not decoration: markers move from an outlined number
    (pending) to a filled white number (current) to a clay checkmark (done,
    i.e. that page's own isComplete() is true and it isn't the current page).
    """

    def __init__(self, wizard, steps: list[tuple[int, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._wizard = wizard
        self._steps = steps
        self.setObjectName("WizardSidebar")
        self.setFixedWidth(200)
        # Plain QWidgets don't paint QSS `background` unless this is set —
        # without it the sidebar (and the current-step row highlight below)
        # would silently render transparent instead of the intended colors.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(SIDEBAR_QSS)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 18, 0, 0)
        outer.setSpacing(0)

        brand = QLabel("SDM WIZARD")
        brand.setObjectName("SidebarBrand")
        outer.addWidget(brand)

        # A page can be isComplete()==True without ever having been opened
        # (e.g. a settings-only page whose fields already have valid
        # defaults, or any QWizardPage subclass that doesn't override
        # isComplete() at all — QWizardPage's own default is True). Gate
        # "done" on having actually visited the page, so the sidebar can't
        # show steps as finished before the user has been anywhere near them.
        self._visited: set[int] = set()
        self._rows: dict[int, tuple[QWidget, QLabel, QLabel]] = {}
        for page_id, label_text in steps:
            row = QWidget()
            row.setObjectName("SidebarStep")
            row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(20, 7, 20, 7)
            row_layout.setSpacing(11)

            marker = QLabel("")
            marker.setObjectName("SidebarMarker")
            marker.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            marker.setFixedSize(20, 20)
            marker.setAlignment(Qt.AlignmentFlag.AlignCenter)

            label = QLabel(label_text)
            label.setObjectName("SidebarLabel")

            row_layout.addWidget(marker)
            row_layout.addWidget(label, 1)
            outer.addWidget(row)
            self._rows[page_id] = (row, marker, label)

        outer.addStretch()

        self._foot = QLabel("")
        self._foot.setObjectName("SidebarFoot")
        outer.addWidget(self._foot)
        outer.addSpacing(12)

        wizard.currentIdChanged.connect(self.refresh)
        self.refresh(wizard.currentId())

    def refresh(self, current_id: int) -> None:
        self._visited.add(current_id)
        total = len(self._steps)
        for i, (page_id, _label_text) in enumerate(self._steps, start=1):
            if page_id not in self._rows:
                continue
            row, marker, label = self._rows[page_id]
            page = self._wizard.page(page_id)
            is_current = page_id == current_id
            is_done = (
                not is_current
                and page_id in self._visited
                and page is not None
                and page.isComplete()
            )
            state = "current" if is_current else ("done" if is_done else "pending")

            marker.setText("✓" if is_done else str(i))
            for w in (row, marker, label):
                w.setProperty("state", state)
                w.style().unpolish(w)
                w.style().polish(w)

            if is_current:
                self._foot.setText(f"Step {i} of {total}")
