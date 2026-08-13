from __future__ import annotations

from qgis.PyQt.QtWidgets import QRadioButton, QVBoxLayout, QWizardPage

from ..page_utils import wrap_scrollable, wrapped_label


class EnsemblePage(QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Ensemble combination")
        self.setSubTitle(
            "Choose how per-algorithm predictions are combined into the ensemble. "
            "An across-model SD (Standard Deviation, i.e. uncertainty) map is "
            "always produced."
        )
        self.mean = QRadioButton("Unweighted mean")
        self.wauc = QRadioButton("Weighted by AUC")
        self.wtss = QRadioButton("Weighted by TSS")
        self.wtss.setChecked(True)

        layout = QVBoxLayout()
        layout.addWidget(self.mean)
        layout.addWidget(self.wauc)
        layout.addWidget(self.wtss)
        layout.addSpacing(20)
        layout.addWidget(wrapped_label(
            "Weighted schemes emphasise better-performing algorithms. AUC (Area "
            "Under the ROC Curve) and TSS (True Skill Statistic) are both accuracy "
            "scores, higher is better; metric weights are the mean value across "
            "replicates for each algorithm."
        ))
        layout.addStretch()
        wrap_scrollable(self, layout)

    def save_to_config(self, cfg) -> None:
        if self.mean.isChecked():
            cfg.ensemble.method = "mean"
        elif self.wauc.isChecked():
            cfg.ensemble.method = "weighted_auc"
        else:
            cfg.ensemble.method = "weighted_tss"

    def validatePage(self) -> bool:
        self.save_to_config(self.wizard_ref.config)
        return True
