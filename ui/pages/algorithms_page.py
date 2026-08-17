from __future__ import annotations

from qgis.PyQt.QtCore import QEvent
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWizardPage,
)

from ...core.config import ALGORITHMS
from ...core.models.config_export import build_model_config, changed_overrides
from ...core.models.registry import algorithm_long_name
from ..page_utils import description_label, wrap_scrollable
from ..widgets.model_config_dialog import ModelConfigDialog

# Shown for whichever algorithm the pointer or keyboard focus is on, in the
# same what-it-is / Best for: / Watch out for: shape the cross-validation and
# ensemble pages use. Unlike those two this is a multi-select page, so the
# description deliberately follows what you are *looking at* rather than what
# is ticked — otherwise you would have to select an algorithm to find out
# whether you wanted it.
_ALGORITHM_HELP = {
    "lr": (
        "<p>Fits a linear model of log-odds, with each predictor's own square "
        "added but no cross-predictor interactions, which is the standard GLM "
        "setup in the SDM literature. Predictors are standardised first, and "
        "mild L2 regularisation keeps the quadratic terms from running "
        "away.</p>"
        "<p><b>Best for:</b> a fast, transparent baseline you can reason "
        "about, and small datasets where more flexible learners overfit. "
        "Worth including as a reference point even when you expect it to "
        "lose.</p>"
        "<p><b>Watch out for:</b> it can only express smooth, roughly "
        "unimodal responses. Sharp thresholds and interactions between "
        "predictors are outside its reach, so it usually scores below the "
        "tree ensembles on complex data.</p>"
    ),
    "gam": (
        "<p>Like the GLM, but each predictor gets a flexible spline instead "
        "of a fixed linear and quadratic shape, so each response curve bends "
        "to follow the data.</p>"
        "<p><b>Best for:</b> when the shape of each predictor's effect is "
        "itself the result you want. It produces the most readable response "
        "curves of anything here while still fitting realistic, non-linear "
        "responses.</p>"
        "<p><b>Watch out for:</b> it is still additive, so it cannot capture "
        "interactions between predictors. With few records, splines given too "
        "many knots will wiggle to chase noise, and it is the slowest of the "
        "simple models to fit.</p>"
    ),
    "rf": (
        "<p>Many decision trees, each grown on a bootstrap sample of the data "
        "and a random subset of the predictors. The prediction is the vote "
        "share across the whole forest.</p>"
        "<p><b>Best for:</b> a strong, low-effort default. It picks up "
        "interactions and non-linear responses with no tuning and shrugs off "
        "correlated or uninformative predictors.</p>"
        "<p><b>Watch out for:</b> it extrapolates poorly. Beyond the range it "
        "was trained on, predictions flatten instead of continuing a trend, "
        "which matters when projecting to a future climate, and the MESS and "
        "MOP layers flag where this bites. It also fits in parallel, so two "
        "runs of the same config can differ in the last decimal place.</p>"
    ),
    "gbm": (
        "<p>Trees fitted one after another, each correcting the errors left "
        "by those before it. These are the boosted regression trees (BRT) of "
        "the SDM literature.</p>"
        "<p><b>Best for:</b> accuracy. Given enough trees at a low learning "
        "rate it is often the best single method on tabular ecological "
        "data.</p>"
        "<p><b>Watch out for:</b> it is more sensitive to its settings than "
        "Random Forest, since learning rate and tree count trade off against "
        "each other, and it is slower because the trees are inherently "
        "sequential. Boosted too long on few records, it will fit the "
        "noise.</p>"
    ),
    "xgb": (
        "<p>The same boosting idea as GBM, with a regularised objective and a "
        "considerably faster implementation.</p>"
        "<p><b>Best for:</b> larger datasets, or when GBM is accurate but too "
        "slow to sit through. It usually matches or beats GBM with less "
        "tuning.</p>"
        "<p><b>Watch out for:</b> the same weak extrapolation as the other "
        "tree methods, and the same tendency to fit noise if you boost for "
        "too long. Its extra regularisation settings give you more to tune "
        "than GBM, which helps if you use them and does nothing if you leave "
        "them alone.</p>"
    ),
    "svm": (
        "<p>Separates suitable from unsuitable conditions with a boundary "
        "drawn in a transformed feature space, using an RBF kernel by "
        "default, on standardised predictors, with probabilities calibrated "
        "afterwards.</p>"
        "<p><b>Best for:</b> small to medium datasets whose boundary between "
        "suitable and unsuitable is complex but smooth.</p>"
        "<p><b>Watch out for:</b> fitting time climbs sharply with sample "
        "size, and the probability calibration adds its own internal "
        "cross-validation on top. With a large background sample this is "
        "often the slowest algorithm on the page. Results depend "
        "substantially on C and gamma.</p>"
    ),
    "mlp": (
        "<p>A small feed-forward neural network fitted to standardised "
        "predictors.</p>"
        "<p><b>Best for:</b> large datasets with complex interactions, where "
        "there are enough records to train one properly.</p>"
        "<p><b>Watch out for:</b> this is the most data-hungry and least "
        "stable option here. With a few hundred presences it often lands well "
        "below the tree ensembles and shifts noticeably between replicates, "
        "and it may hit its iteration limit before converging, which puts a "
        "warning in the run log.</p>"
    ),
    "maxent": (
        "<p>The presence-background workhorse of the field. It builds feature "
        "transforms of the predictors (linear, quadratic, product, hinge, "
        "threshold) and finds the distribution closest to uniform that still "
        "matches the conditions observed at presence sites.</p>"
        "<p><b>Best for:</b> presence-only data, particularly with modest "
        "presence counts, and as the natural point of comparison with "
        "published work, since it is the most widely used SDM method there "
        "is.</p>"
        "<p><b>Watch out for:</b> the regularisation (beta multiplier) and "
        "the feature classes drive the outcome, and by default the feature "
        "classes are chosen automatically from your presence count. Both are "
        "editable under <i>Show / edit model configuration</i>.</p>"
    ),
    "enfa": (
        "<p>Compares conditions at presence sites against those available "
        "across the whole sample, scoring suitability from the Mahalanobis "
        "distance to the species' environmental centre.</p>"
        "<p><b>Best for:</b> presence-only data, and as a genuinely different "
        "kind of contrast to everything else here, since it describes where "
        "the niche sits and how wide it is rather than learning a decision "
        "boundary.</p>"
        "<p><b>Watch out for:</b> it is presence-only by design, and "
        "selecting it in presence/absence mode is rejected before the run "
        "starts. It assumes one roughly elliptical niche, so it cannot "
        "represent a species with two separate suitable zones, and it needs "
        "at least 2 presence points.</p>"
    ),
}

_ALGORITHM_HELP_HINT = (
    "<p>Point at or tab to an algorithm above to read what it does, when it "
    "is the right choice, and what to watch out for.</p>"
)


class AlgorithmsPage(QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Algorithms and replicates")
        self.setSubTitle(
            "Select the models to fit. Point at one to read what it does and "
            "when to use it. Each will run for the chosen number of replicates; "
            "the ensemble combines them at the end."
        )
        self.algo_boxes: dict[str, QCheckBox] = {}
        # Reverse lookup for the event filter below, which is handed the
        # checkbox that was entered/focused and needs its algorithm code.
        self._algo_of: dict[QCheckBox, str] = {}
        grid = QGridLayout()
        for i, algo in enumerate(ALGORITHMS):
            cb = QCheckBox(f"{algo.upper()}: {algorithm_long_name(algo)}")
            cb.setChecked(False)
            cb.toggled.connect(self.completeChanged)
            cb.toggled.connect(self._update_info)
            cb.installEventFilter(self)
            self.algo_boxes[algo] = cb
            self._algo_of[cb] = algo
            grid.addWidget(cb, i // 2, i % 2)

        self.algo_help = description_label()
        self.algo_help.setText(_ALGORITHM_HELP_HINT)
        algo_box = QGroupBox("Algorithms")
        algo_layout = QVBoxLayout(algo_box)
        algo_layout.addLayout(grid)
        algo_layout.addWidget(self.algo_help)

        self.replicates = QSpinBox()
        self.replicates.setRange(1, 100)
        self.replicates.setValue(5)
        self.replicates.valueChanged.connect(self._update_info)

        form = QFormLayout()
        form.addRow(QLabel("Replicates per algorithm:"), self.replicates)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)

        self.config_btn = QPushButton("Show / edit model configuration...")
        self.config_btn.clicked.connect(self._show_model_config)

        layout = QVBoxLayout()
        layout.addWidget(algo_box)
        layout.addSpacing(10)
        layout.addLayout(form)
        layout.addWidget(self.info_label)
        layout.addWidget(self.config_btn)
        layout.addStretch()
        wrap_scrollable(self, layout)
        self._update_info()

    def eventFilter(self, obj, event):
        """Point the description at whichever algorithm the user is looking
        at — hovering with the mouse or tabbing to it with the keyboard —
        rather than at whatever happens to be ticked. It stays on the last one
        after the pointer moves away, so it can be read at leisure.
        """
        if event.type() in (QEvent.Type.Enter, QEvent.Type.FocusIn):
            algo = self._algo_of.get(obj)
            if algo is not None:
                # .get, not [] — an algorithm added to ALGORITHMS without a
                # description here should leave the box thin, never crash the
                # page on hover. tests/test_algorithm_help.py catches the gap.
                self.algo_help.setText(
                    f"<p><b>{algo.upper()}: {algorithm_long_name(algo)}</b></p>"
                    + _ALGORITHM_HELP.get(algo, "")
                )
        return super().eventFilter(obj, event)

    def _show_model_config(self) -> None:
        selected = [name for name, cb in self.algo_boxes.items() if cb.isChecked()]
        algos = selected or list(ALGORITHMS)

        presence_flag = self.wizard_ref.session.presence_flag
        n_presence = int((presence_flag == 1).sum()) if presence_flag is not None else None

        overrides = self.wizard_ref.config.modeling.hyperparameters
        config = build_model_config(algos, n_presence=n_presence, overrides=overrides)
        labels = {
            algo: f"{algo.upper()}: {algorithm_long_name(algo)}" for algo in algos
        }

        if not selected:
            note = "No algorithms are selected yet, so these are the defaults for all of them. "
        else:
            note = ""
        if n_presence is not None:
            note += (
                f"MaxEnt's feature classes below are resolved for this session's "
                f"{n_presence} presence points."
            )
        else:
            note += (
                "MaxEnt's feature classes adapt automatically to your final presence "
                "count once background points are sampled. They are shown here as the "
                "selection rule rather than a resolved value."
            )

        dlg = ModelConfigDialog(config, labels=labels, note=note, parent=self)
        if dlg.exec():
            new_overrides = changed_overrides(dlg.edited_config, n_presence=n_presence)
            # Replace the entries for the algorithms just shown (so clearing an
            # edit back to its default drops it), but keep any overrides for
            # algorithms that were not on screen this time.
            merged = dict(self.wizard_ref.config.modeling.hyperparameters)
            for algo in algos:
                merged.pop(algo, None)
            merged.update(new_overrides)
            self.wizard_ref.config.modeling.hyperparameters = merged

    def _update_info(self) -> None:
        n_algo = sum(1 for cb in self.algo_boxes.values() if cb.isChecked())
        n_rep = self.replicates.value()
        self.info_label.setText(
            f"\n{n_algo} algorithm(s) selected times {n_rep} replicate(s) gives "
            f"{n_algo * n_rep} model fits per fold. "
            "More replicates give more stable metrics but a longer runtime."
        )

    def isComplete(self) -> bool:
        return any(cb.isChecked() for cb in self.algo_boxes.values())

    def save_to_config(self, cfg) -> None:
        cfg.modeling.algorithms = [
            name for name, cb in self.algo_boxes.items() if cb.isChecked()
        ]
        cfg.modeling.replicates = int(self.replicates.value())

    def validatePage(self) -> bool:
        self.save_to_config(self.wizard_ref.config)
        return True
