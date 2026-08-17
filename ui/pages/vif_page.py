from __future__ import annotations

import numpy as np
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWizardPage,
)

from ...core.stages import collect_labeled_points_and_extract, stage_vif
from ..page_utils import wrap_scrollable, wrapped_label
from ..workers import StagePageMixin, snapshot_key


class VIFPage(StagePageMixin, QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Predictor selection (stepwise VIF)")
        self.setSubTitle(
            "VIF (Variance Inflation Factor) measures multicollinearity, meaning how "
            "much "
            "one predictor's information overlaps with the others. This step "
            "iteratively drops the predictor with the highest VIF until all "
            "remaining predictors are below the cutoff, so models aren't fed "
            "redundant, highly correlated variables. Each step is logged to the "
            "report."
        )
        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(1.5, 100.0)
        self.cutoff.setDecimals(1)
        self.cutoff.setSingleStep(0.5)
        self.cutoff.setValue(10.0)

        self.keep_all = QCheckBox("Keep all predictors (skip multicollinearity check)")
        self.keep_all.setChecked(False)

        form = QFormLayout()
        form.addRow(QLabel("VIF cutoff:"), self.cutoff)
        form.addRow(QLabel(""), self.keep_all)

        self.run_btn = QPushButton("Run VIF")
        self.run_btn.setProperty("cls", "primary")
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.busy_label = QLabel("Running…")
        self.busy_label.setVisible(False)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #b00020;")
        self.error_label.setWordWrap(True)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Step", "Predictor", "VIF", "Action"])
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(wrapped_label(
            "Default is 10 (a common threshold in ecological modelling). "
            "Lowering to 5 gives a stricter, more decorrelated set. Checking "
            "'Keep all predictors' bypasses this step entirely, which is useful if "
            "you've already handled multicollinearity yourself, or want every "
            "variable available regardless of overlap."
        ))
        layout.addWidget(self.run_btn)
        layout.addWidget(self.busy_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.table)
        wrap_scrollable(self, layout)

        self._stage_ok = False
        self._last_snapshot: str | None = None
        self.cutoff.valueChanged.connect(self.completeChanged)
        self.keep_all.toggled.connect(self.completeChanged)
        self.keep_all.toggled.connect(self._update_field_state)
        self._update_field_state()

    def _update_field_state(self) -> None:
        checked = self.keep_all.isChecked()
        self.cutoff.setEnabled(not checked)
        self.run_btn.setText("Keep All Predictors" if checked else "Run VIF")

    def _snapshot(self) -> str:
        wizard = self.wizard_ref
        return snapshot_key(
            self.cutoff.value(),
            self.keep_all.isChecked(),
            wizard.session.stage_hashes.get("cleaning"),
            wizard.session.stage_hashes.get("background"),
        )

    def _on_run_clicked(self) -> None:
        self.error_label.setText("")
        wizard = self.wizard_ref
        self.save_to_config(wizard.config)
        cfg = wizard.config
        occ = wizard.session.occ
        stack = wizard.session.stack
        seed = cfg.random_seed
        need_points = wizard.session.X_full is None
        px_c, py_c, pf_c = wizard.session.px, wizard.session.py, wizard.session.presence_flag
        X_full_c, feature_names_c = wizard.session.X_full, wizard.session.feature_names

        def _work():
            if need_points:
                px, py, presence_flag, X_full, feature_names = collect_labeled_points_and_extract(
                    cfg, occ, stack, np.random.default_rng(seed)
                )
            else:
                px, py, presence_flag = px_c, py_c, pf_c
                X_full, feature_names = X_full_c, feature_names_c
            X_kept, kept_names, kept_idx, vif_report = stage_vif(cfg, X_full, feature_names)
            return px, py, presence_flag, X_full, feature_names, X_kept, kept_names, kept_idx, vif_report

        self.run_stage_async(
            _work, self._on_done, self._on_failed,
            button=self.run_btn, busy_widget=self.busy_label,
        )

    def _on_done(self, result) -> None:
        (
            px, py, presence_flag, X_full, feature_names,
            X_kept, kept_names, kept_idx, vif_report,
        ) = result
        wizard = self.wizard_ref

        rows = []
        for s in vif_report.steps:
            if s.dropped:
                rows.append((s.step, s.dropped, s.vifs.get(s.dropped, float("nan")), "Dropped"))
            elif s.vifs:
                worst = max(s.vifs, key=s.vifs.get)
                rows.append((s.step, "none", s.vifs[worst], "Stopped (all at or below cutoff)"))
            else:
                rows.append((s.step, "none", float("nan"), "Stopped"))
        self.table.setRowCount(len(rows))
        for row, (step, name, vif, action) in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(str(step)))
            self.table.setItem(row, 1, QTableWidgetItem(name))
            self.table.setItem(row, 2, QTableWidgetItem(f"{vif:.2f}" if np.isfinite(vif) else "n/a"))
            self.table.setItem(row, 3, QTableWidgetItem(action))

        if vif_report.skipped:
            self.summary_label.setText(
                f"Multicollinearity check skipped. All {len(vif_report.retained)} "
                f"predictor(s) kept: {', '.join(vif_report.retained) or 'none'}."
            )
        else:
            self.summary_label.setText(
                f"Retained ({len(vif_report.retained)}): {', '.join(vif_report.retained) or 'none'}. "
                f"Dropped ({len(vif_report.dropped)}): {', '.join(vif_report.dropped) or 'none'}."
            )

        new_key = self._snapshot()
        old_key = wizard.session.stage_hashes.get("vif")
        if old_key is not None and old_key != new_key:
            wizard.invalidate_from(wizard.PAGE_VIF)
        wizard.session.px, wizard.session.py, wizard.session.presence_flag = px, py, presence_flag
        wizard.session.X_full, wizard.session.feature_names = X_full, feature_names
        wizard.session.X_kept, wizard.session.kept_names = X_kept, kept_names
        wizard.session.kept_idx, wizard.session.vif_report = kept_idx, vif_report
        wizard.session.stage_hashes["vif"] = new_key

        self._stage_ok = True
        self._last_snapshot = new_key

    def _on_failed(self, err: str) -> None:
        self._stage_ok = False
        self.table.setRowCount(0)
        self.summary_label.setText("")
        self.error_label.setText(err.splitlines()[0] if err else "VIF selection failed.")

    def invalidate(self) -> None:
        self._stage_ok = False
        self._last_snapshot = None
        self.table.setRowCount(0)
        self.summary_label.setText("")
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._stage_ok and self._snapshot() == self._last_snapshot

    def save_to_config(self, cfg) -> None:
        cfg.vif.cutoff = float(self.cutoff.value())
        cfg.vif.enabled = not self.keep_all.isChecked()

    def validatePage(self) -> bool:
        self.save_to_config(self.wizard_ref.config)
        return True
