from __future__ import annotations

from qgis.PyQt.QtWidgets import QGroupBox, QRadioButton, QVBoxLayout, QWizardPage

from ..page_utils import description_label, wrap_scrollable

# What each combination rule does, when it is the right one, and what to
# expect from it in practice — swapped in as the selection changes, mirroring
# the cross-validation page. The weighting maths described here is
# core/prediction/ensemble.py::compute_weights: each algorithm's raw weight is
# its mean metric across replicates, clamped at zero, then normalised so the
# weights sum to 1.
_METHOD_HELP = {
    "mean": (
        "<p>Every algorithm counts the same: the ensemble is the plain "
        "average of the per-algorithm suitability maps.</p>"
        "<p><b>Best for:</b> a set of algorithms that all perform comparably, "
        "or when you would rather not let one accuracy score decide how much "
        "each model counts. It is also the steadier choice on a small test "
        "set, where the metrics themselves are noisy enough that weighting by "
        "them mostly amplifies that noise.</p>"
        "<p><b>In practice:</b> the simplest and most transparent option, and "
        "the one to fall back on if a weighted ensemble looks driven by a "
        "single model. Note that a weak algorithm still pulls the result "
        "toward its own errors, so deselect poor performers on the Algorithms "
        "page rather than relying on weights to quiet them.</p>"
    ),
    "weighted_auc": (
        "<p>Each algorithm contributes in proportion to its mean AUC (Area "
        "Under the ROC Curve) across replicates. AUC is an accuracy score "
        "from 0 to 1 where higher is better, measuring how well a model ranks "
        "sites from least to most suitable across every possible cut-off.</p>"
        "<p><b>Best for:</b> weighting on threshold-independent ranking "
        "ability, when what you care about is the order of suitability rather "
        "than performance at any particular cut-off.</p>"
        "<p><b>In practice:</b> a mild weighting. Usable models sit in a "
        "narrow AUC band of roughly 0.7 to 0.95, and random guessing already "
        "scores 0.5, so even a near-useless model keeps more than half the "
        "weight of your best one. Expect results close to the unweighted "
        "mean.</p>"
    ),
    "weighted_tss": (
        "<p>Each algorithm contributes in proportion to its mean TSS (True "
        "Skill Statistic) across replicates. TSS is sensitivity plus "
        "specificity minus 1, scored at each model's own best threshold: 0 "
        "for a model no better than random, 1 for a perfect one.</p>"
        "<p><b>Best for:</b> letting model quality actually change the "
        "outcome, which is why it is the default here. Because TSS starts "
        "from 0 rather than 0.5, it spreads algorithms much further apart "
        "than AUC does. A weak model's weight shrinks toward nothing, and one "
        "scoring below random is clamped to zero weight and drops out of the "
        "ensemble entirely.</p>"
        "<p><b>In practice:</b> the usual choice when your selected "
        "algorithms differ noticeably in quality. Unlike AUC it does depend "
        "on a chosen cut-off, the max-TSS threshold, which is the same one "
        "used for the binary output rasters, so it rewards models that are "
        "decisive at that cut-off rather than merely well-ordered.</p>"
    ),
}


class EnsemblePage(QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Ensemble combination")
        self.setSubTitle(
            "Choose how per-algorithm predictions are combined into the ensemble. "
            "Selecting one updates the description below. An across-model SD "
            "(Standard Deviation, i.e. uncertainty) map is always produced, "
            "whichever you pick."
        )
        self.mean = QRadioButton("Unweighted mean")
        self.wauc = QRadioButton("Weighted by AUC")
        self.wtss = QRadioButton("Weighted by TSS")
        self.wtss.setChecked(True)

        self.method_help = description_label()
        method_box = QGroupBox("Combination")
        method_layout = QVBoxLayout(method_box)
        for radio in (self.mean, self.wauc, self.wtss):
            radio.toggled.connect(self._update_method_help)
            method_layout.addWidget(radio)
        method_layout.addWidget(self.method_help)

        layout = QVBoxLayout()
        layout.addWidget(method_box)
        layout.addStretch()
        wrap_scrollable(self, layout)

        self._update_method_help()

    def _method(self) -> str:
        if self.mean.isChecked():
            return "mean"
        if self.wauc.isChecked():
            return "weighted_auc"
        return "weighted_tss"

    def _update_method_help(self) -> None:
        self.method_help.setText(_METHOD_HELP[self._method()])

    def save_to_config(self, cfg) -> None:
        cfg.ensemble.method = self._method()

    def validatePage(self) -> bool:
        self.save_to_config(self.wizard_ref.config)
        return True
