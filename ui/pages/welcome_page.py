from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWizardPage,
)

from ...core.config import ALGORITHMS
from ..page_utils import wrap_scrollable, wrapped_label
from ..theme import CLAY, FOREST, FOREST_DARK, MOSS

N_STEPS = 12


def _stat(number: str, caption: str) -> QVBoxLayout:
    col = QVBoxLayout()
    col.setSpacing(0)
    num = QLabel(number)
    num.setAlignment(Qt.AlignmentFlag.AlignCenter)
    num.setStyleSheet(f"font-size: 20pt; font-weight: 700; color: {CLAY};")
    cap = QLabel(caption)
    cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
    cap.setStyleSheet(f"font-size: 8pt; color: {MOSS}; letter-spacing: 0.5px;")
    col.addWidget(num)
    col.addWidget(cap)
    return col


def _highlight(title: str, body: str) -> QVBoxLayout:
    col = QVBoxLayout()
    col.setSpacing(2)
    head = QLabel(title)
    head.setStyleSheet(f"font-weight: 700; color: {FOREST};")
    desc = wrapped_label(body)
    desc.setStyleSheet(f"color: {FOREST_DARK};")
    col.addWidget(head)
    col.addWidget(desc)
    return col


class WelcomePage(QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Welcome to SDM")
        self.setSubTitle(
            "A guided, reproducible species distribution modeling workflow, from raw "
            "occurrence records to a mapped, ensembled, and evaluated prediction."
        )

        hero = wrapped_label(
            "Every step below runs live as you go: load data, preview it on the map, "
            "adjust settings, and see the result before moving on, with no blind final run."
        )
        hero.setStyleSheet(f"font-size: 11pt; font-weight: 600; color: {FOREST};")

        stats_row = QHBoxLayout()
        stats_row.setSpacing(28)
        stats_row.addLayout(_stat(str(len(ALGORITHMS)), "ALGORITHMS"))
        stats_row.addLayout(_stat(str(N_STEPS), "GUIDED STEPS"))
        stats_row.addLayout(_stat("1", "ENSEMBLE MAP"))
        stats_row.addStretch()

        what_box = QGroupBox("What this does")
        what_layout = QVBoxLayout(what_box)
        what_layout.addWidget(wrapped_label(
            "SDM fits and compares up to nine modeling algorithms (logistic "
            "regression, GAM, random forest, gradient boosting, XGBoost, SVM, a "
            "neural net, MaxEnt, and ENFA) against your occurrence records and "
            "environmental predictor rasters, then combines the best-performing "
            "ones into a single ensemble suitability map with an uncertainty layer."
        ))

        who_box = QGroupBox("Who it's for")
        who_layout = QVBoxLayout(who_box)
        who_layout.addWidget(wrapped_label(
            "Anyone working in QGIS who needs a defensible, repeatable distribution "
            "model (ecologists, conservation planners, and researchers) without "
            "writing R or Python. Every setting you choose, and every intermediate "
            "result, is written into the final HTML report."
        ))

        highlight_box = QGroupBox("Highlights")
        highlight_box.setProperty("cls", "accent")
        highlight_grid = QGridLayout(highlight_box)
        highlight_grid.setHorizontalSpacing(24)
        highlight_grid.setVerticalSpacing(14)
        highlights = [
            ("Live previews", "Every load, clean, and sample step draws its result "
                "in the map immediately, both in this wizard and in your real QGIS project."),
            ("Reproducible", "Every setting you choose and every intermediate result "
                "is written into the run's HTML report."),
            ("Ensemble + uncertainty", "Combines your best-performing algorithms into "
                "one map, plus an across-model uncertainty layer, every run."),
            ("Honest cross-validation", "Spatial block CV is available for spatially "
                "autocorrelated data, not just a random split."),
            ("Smart caching", "Going back and changing one earlier setting only "
                "re-runs the steps actually downstream of it."),
            ("CRS-flexible distances", "Enter buffer and block sizes in km or m, and "
                "SDM converts them to your raster's CRS (Coordinate Reference "
                "System) automatically."),
        ]
        for i, (title, body) in enumerate(highlights):
            highlight_grid.addLayout(_highlight(title, body), i // 2, i % 2)
        highlight_grid.setColumnStretch(0, 1)
        highlight_grid.setColumnStretch(1, 1)

        need_box = QGroupBox("What you'll need")
        need_layout = QVBoxLayout(need_box)
        need_layout.addWidget(wrapped_label(
            "• Occurrence records: a CSV or point layer with longitude/latitude.\n"
            "• Predictor rasters: one or more, sharing the same CRS, extent, and "
            "resolution (a projection stack for future/past climate is optional).\n"
            "• A rough idea of your study extent. Everything else has a sensible "
            "default you can change later."
        ))

        steps_box = QGroupBox("The workflow in 12 steps")
        steps_layout = QVBoxLayout(steps_box)
        steps_layout.addWidget(wrapped_label(
            "1. Occurrence data 2. Predictor rasters 3. Cleaning\n"
            "4. Background points 5. Predictor selection (VIF — multicollinearity) "
            "6. Cross-validation (CV)\n"
            "7. Algorithms 8. Projection stack (optional) "
            "9. Ensemble\n"
            "10. Output settings 11. Run 12. Done"
        ))

        footer = QLabel("Click Next to begin.")
        footer.setAlignment(Qt.AlignmentFlag.AlignRight)
        footer.setStyleSheet("font-style: italic;")

        layout = QVBoxLayout()
        layout.addWidget(hero)
        layout.addSpacing(4)
        layout.addLayout(stats_row)
        layout.addSpacing(8)
        layout.addWidget(what_box)
        layout.addWidget(who_box)
        layout.addWidget(highlight_box)
        layout.addWidget(need_box)
        layout.addWidget(steps_box)
        layout.addSpacing(6)
        layout.addWidget(footer)
        layout.addStretch()
        wrap_scrollable(self, layout)
