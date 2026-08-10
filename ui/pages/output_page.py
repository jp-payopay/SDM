from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWizardPage,
)

from ..page_utils import wrap_scrollable


class OutputPage(QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Outputs")
        self.setSubTitle(
            "Pick an output directory. Everything is written there: rasters, plots, "
            "metrics tables, saved models, run configuration, and HTML report."
        )
        self.dir_edit = QLineEdit()
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.dir_edit)
        row.addWidget(self.browse_btn)

        self.report_cb = QCheckBox("Write HTML report")
        self.report_cb.setChecked(True)
        self.models_cb = QCheckBox("Save fitted models (.joblib)")
        self.models_cb.setChecked(True)

        form = QFormLayout()
        form.addRow(QLabel("Output directory:"), row)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.report_cb)
        layout.addWidget(self.models_cb)
        layout.addStretch()
        wrap_scrollable(self, layout)
        self.registerField("out_dir*", self.dir_edit)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self.dir_edit.setText(d)

    def isComplete(self) -> bool:
        return bool(self.dir_edit.text().strip())

    def save_to_config(self, cfg) -> None:
        cfg.output.directory = self.dir_edit.text().strip()
        cfg.output.write_html_report = self.report_cb.isChecked()
        cfg.output.save_models = self.models_cb.isChecked()

    def validatePage(self) -> bool:
        self.save_to_config(self.wizard_ref.config)
        return True
