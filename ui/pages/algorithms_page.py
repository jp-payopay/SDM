from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWizardPage,
)

from ...core.config import ALGORITHMS
from ...core.models.config_export import build_model_config, changed_overrides
from ...core.models.registry import algorithm_long_name
from ..page_utils import wrap_scrollable
from ..widgets.model_config_dialog import ModelConfigDialog


class AlgorithmsPage(QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Algorithms and replicates")
        self.setSubTitle(
            "Select the models to fit. Each will run for the chosen number of "
            "replicates; the ensemble combines them at the end."
        )
        self.algo_boxes: dict[str, QCheckBox] = {}
        grid = QGridLayout()
        for i, algo in enumerate(ALGORITHMS):
            cb = QCheckBox(f"{algo.upper()}: {algorithm_long_name(algo)}")
            cb.setChecked(False)
            cb.toggled.connect(self.completeChanged)
            cb.toggled.connect(self._update_info)
            self.algo_boxes[algo] = cb
            grid.addWidget(cb, i // 2, i % 2)

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
        layout.addLayout(grid)
        layout.addSpacing(10)
        layout.addLayout(form)
        layout.addWidget(self.info_label)
        layout.addWidget(self.config_btn)
        layout.addStretch()
        wrap_scrollable(self, layout)
        self._update_info()

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
