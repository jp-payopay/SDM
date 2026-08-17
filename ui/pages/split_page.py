from __future__ import annotations

import numpy as np
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWizardPage,
)

from ...core.stages import make_folds
from ...core.units import crs_units_to_meters, distance_to_crs_units, format_meters, is_geographic_crs
from ..page_utils import description_label, wrap_scrollable, wrapped_label
from ..qgis_layers import clear_stage, show_points, show_spatial_block
from ..widgets.embedded_canvas import EmbeddedPreviewCanvas
from ..workers import StagePageMixin, snapshot_key

# What each strategy does, when to reach for it, and the values people
# actually use — swapped in as the selection changes, so the page explains
# the option in front of you instead of all three at once. Same three-part
# shape for each, so they can be compared by flipping between them.
_METHOD_HELP = {
    "kfold": (
        "<p>Splits the points into k equal parts and fits k times, each fit "
        "holding out a different part as the test set. Every point is tested "
        "exactly once, and the reported score is the average across the k "
        "fits.</p>"
        "<p><b>Best for:</b> a stable estimate when your points are not "
        "strongly clustered in space. Averaging k fits means the result does "
        "not hang on one lucky or unlucky split, which is the main weakness "
        "of a single hold-out.</p>"
        "<p><b>Common values:</b> k = 5 is the usual default and k = 10 is "
        "the other common choice. Below 5, each training set loses a "
        "noticeable share of the data. Above 10 you pay for many more fits "
        "with little gain in return.</p>"
    ),
    "random": (
        "<p>Sets aside one random fraction of the points as a test set and "
        "fits the model once on everything else.</p>"
        "<p><b>Best for:</b> quick exploratory checks and very large "
        "datasets, where a single fit is enough and speed matters. The "
        "estimate rests entirely on that one split, so it varies more between "
        "runs than k-fold's does.</p>"
        "<p><b>Common values:</b> hold out 20% to 30%. An 80/20 split, "
        "meaning a test size of 0.20, is the usual starting point, and this "
        "page defaults to 0.25. Below about 0.10 the test set gets too small "
        "to measure anything reliably.</p>"
    ),
    "spatial_block": (
        "<p>Divides the study area into blocks and holds out whole blocks at "
        "a time, so test points sit some distance from the points the model "
        "was fit on instead of being scattered among them.</p>"
        "<p><b>Best for:</b> species records, which are almost always "
        "spatially clustered, and that is why it is the default here. Under a "
        "random split, test points sit right beside training points, so a "
        "model gets credit for recognising conditions it has effectively "
        "already seen and the score comes out optimistically high.</p>"
        "<p><b>Common values:</b> k = 5 groups of blocks. Blocks should be at "
        "least as wide as the distance over which your predictors stay "
        "similar, and leaving <i>Auto block size</i> ticked estimates that "
        "from an empirical variogram, which is what most users want. Square "
        "is the classic tiling. Hexagonal gives every block six equidistant "
        "neighbours instead of the mixed orthogonal and diagonal distances a "
        "square grid has.</p>"
    ),
}


class SplitPage(StagePageMixin, QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Cross-validation strategy")
        self.setSubTitle(
            "CV (Cross-Validation) tests each model on data it wasn't fit on, to "
            "estimate how well it generalizes. Pick a strategy below and the "
            "description will cover what it does, when it is the right choice, "
            "and the values people normally use."
        )
        # Ordered plainest-first: k-fold is the general-purpose workhorse,
        # random hold-out its one-fit shortcut, spatial block the specialised
        # (and for species data, most defensible) option. Spatial block stays
        # the *checked* default despite being listed last — see _METHOD_HELP.
        self.kfold = QRadioButton("k-fold")
        self.random = QRadioButton("Random hold-out")
        self.block = QRadioButton("Spatial block")
        self.block.setChecked(True)

        self.method_help = description_label()
        method_box = QGroupBox("Method")
        method_layout = QVBoxLayout(method_box)
        for radio in (self.kfold, self.random, self.block):
            method_layout.addWidget(radio)
        method_layout.addWidget(self.method_help)

        self.k = QSpinBox()
        self.k.setRange(2, 20)
        self.k.setValue(5)

        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.05, 0.5)
        self.test_size.setDecimals(2)
        self.test_size.setSingleStep(0.05)
        self.test_size.setValue(0.25)

        self.auto_block = QCheckBox("Auto block size (empirical variogram)")
        self.auto_block.setChecked(True)
        self.block_size_value = QDoubleSpinBox()
        self.block_size_value.setRange(0.0, 1e9)
        self.block_size_value.setDecimals(2)
        self.block_size_value.setValue(0.0)
        self.block_size_unit = QComboBox()
        self.block_size_unit.addItems(["km", "m"])
        block_size_row = QHBoxLayout()
        block_size_row.addWidget(self.block_size_value)
        block_size_row.addWidget(self.block_size_unit)

        # Display labels only; save_to_config()/_block_shape() map these back
        # to the "square"/"hexagon" values core/split/spatial_block.py uses.
        self.block_shape = QComboBox()
        self.block_shape.addItems(["Square", "Hexagonal"])

        self.k_label = QLabel("k (for k-fold / block):")
        self.test_size_label = QLabel("Test size (for random):")
        self.auto_block_label = QLabel("Spatial block:")
        self.block_size_label = QLabel("Block size (if not auto):")
        self.block_shape_label = QLabel("Block shape:")

        form = QFormLayout()
        form.addRow(self.k_label, self.k)
        form.addRow(self.test_size_label, self.test_size)
        form.addRow(self.auto_block_label, self.auto_block)
        form.addRow(self.block_size_label, block_size_row)
        form.addRow(self.block_shape_label, self.block_shape)

        self.run_btn = QPushButton("Preview Split")
        self.run_btn.setProperty("cls", "primary")
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.busy_label = QLabel("Computing…")
        self.busy_label.setVisible(False)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #b00020;")
        self.error_label.setWordWrap(True)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Fold", "Train N", "Test N"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.conversion_label = QLabel("")
        self.conversion_label.setStyleSheet("color: #666; font-style: italic;")
        self.conversion_label.setWordWrap(True)
        self.canvas = EmbeddedPreviewCanvas()

        # Block-specific prose, shown only while Spatial block is selected —
        # leaving it on screen under k-fold would contradict the description
        # box directly above it.
        self.block_size_help = wrapped_label(
            "Enter the block size in km or m. It is converted automatically to your "
            "predictor rasters' CRS units, and only used when 'Auto block size' is "
            "unchecked. The preview reflects one representative split; each replicate's "
            "actual fold assignment may differ slightly (especially the auto-computed "
            "spatial block size, which is itself estimated stochastically)."
        )
        self.block_shape_help = wrapped_label(
            "Block shape: square blocks are the classic grid partition. Hexagonal "
            "blocks give every block six equidistant neighbours instead of the mixed "
            "orthogonal and diagonal distances a square grid has, which is why some "
            "spatial cross-validation tools prefer them. Both use the same block "
            "size, meaning the same ground area per block, so switching shape should "
            "not change the typical block scale, only how they tile."
        )

        layout = QVBoxLayout()
        layout.addWidget(method_box)
        layout.addLayout(form)
        layout.addWidget(self.block_size_help)
        layout.addWidget(self.conversion_label)
        layout.addWidget(self.block_shape_help)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.busy_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.table)
        layout.addWidget(self.canvas)
        layout.addStretch()
        wrap_scrollable(self, layout)

        self._stage_ok = False
        self._last_snapshot: str | None = None
        for w in (self.k, self.test_size):
            w.valueChanged.connect(self.completeChanged)
        for w in (self.random, self.kfold, self.block, self.auto_block):
            w.toggled.connect(self.completeChanged)
        for w in (self.random, self.kfold, self.block, self.auto_block):
            w.toggled.connect(self._update_field_state)
        self.block_size_value.valueChanged.connect(self.completeChanged)
        self.block_size_value.valueChanged.connect(self._update_conversion_label)
        self.block_size_unit.currentIndexChanged.connect(self.completeChanged)
        self.block_size_unit.currentIndexChanged.connect(self._update_conversion_label)
        self.block_shape.currentIndexChanged.connect(self.completeChanged)
        self._update_conversion_label()
        self._update_field_state()

    def initializePage(self) -> None:
        # The predictor stack (needed to know the CRS) is only guaranteed
        # loaded once this page is actually reached.
        self._update_conversion_label()

    def _method(self) -> str:
        if self.kfold.isChecked():
            return "kfold"
        if self.random.isChecked():
            return "random"
        return "spatial_block"

    def _update_field_state(self) -> None:
        is_random = self.random.isChecked()
        is_kfold = self.kfold.isChecked()
        is_block = self.block.isChecked()

        self.method_help.setText(_METHOD_HELP[self._method()])
        for widget in (self.block_size_help, self.block_shape_help, self.conversion_label):
            widget.setVisible(is_block)

        self.test_size_label.setEnabled(is_random)
        self.test_size.setEnabled(is_random)

        self.k_label.setEnabled(is_kfold or is_block)
        self.k.setEnabled(is_kfold or is_block)

        self.auto_block_label.setEnabled(is_block)
        self.auto_block.setEnabled(is_block)

        block_size_relevant = is_block and not self.auto_block.isChecked()
        for w in (self.block_size_label, self.block_size_value, self.block_size_unit, self.conversion_label):
            w.setEnabled(block_size_relevant)

        # Shape applies regardless of auto vs. manual sizing, unlike the
        # size fields above.
        self.block_shape_label.setEnabled(is_block)
        self.block_shape.setEnabled(is_block)

    def _block_size_meters(self) -> float:
        factor = 1000.0 if self.block_size_unit.currentText() == "km" else 1.0
        return self.block_size_value.value() * factor

    def _block_shape_value(self) -> str:
        return "hexagon" if self.block_shape.currentText() == "Hexagonal" else "square"

    def _update_conversion_label(self) -> None:
        stack = self.wizard_ref.session.stack
        meters = self._block_size_meters()
        if stack is None:
            self.conversion_label.setText("(predictor CRS not loaded yet)")
            return
        if is_geographic_crs(stack.crs):
            lat = (stack.bounds[1] + stack.bounds[3]) / 2.0
            crs_val = distance_to_crs_units(meters, stack.crs, lat)
            self.conversion_label.setText(
                f"= {crs_val:.5f}° in this raster's geographic CRS "
                f"(computed at centroid latitude {lat:.2f}°)."
            )
        else:
            self.conversion_label.setText(
                f"= {meters:,.0f} m in this raster's projected CRS units."
            )

    def _snapshot(self) -> str:
        wizard = self.wizard_ref
        return snapshot_key(
            self.random.isChecked(), self.kfold.isChecked(), self.block.isChecked(),
            self.k.value(), self.test_size.value(),
            self.auto_block.isChecked(), self.block_size_value.value(), self.block_size_unit.currentText(),
            self._block_shape_value(),
            wizard.session.stage_hashes.get("vif"),
        )

    def _on_run_clicked(self) -> None:
        self.error_label.setText("")
        wizard = self.wizard_ref
        self.save_to_config(wizard.config)
        cfg = wizard.config
        X_kept = wizard.session.X_kept
        presence_flag = wizard.session.presence_flag
        px, py = wizard.session.px, wizard.session.py
        stack = wizard.session.stack
        seed = cfg.random_seed

        def _work():
            rng = np.random.default_rng(seed)
            return make_folds(cfg, X_kept, presence_flag, px, py, stack, rng)

        self.run_stage_async(
            _work, self._on_done, self._on_failed,
            button=self.run_btn, busy_widget=self.busy_label,
        )

    def _on_done(self, result) -> None:
        folds, plan, fold_id = result
        wizard = self.wizard_ref

        self.table.setRowCount(len(folds))
        for i, (train, test) in enumerate(folds):
            self.table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.table.setItem(i, 1, QTableWidgetItem(str(len(train))))
            self.table.setItem(i, 2, QTableWidgetItem(str(len(test))))

        px, py = wizard.session.px, wizard.session.py
        crs = wizard.session.stack.crs
        if plan is not None and fold_id is not None:
            size_m = crs_units_to_meters(plan.block_size, crs, (wizard.session.stack.bounds[1] + wizard.session.stack.bounds[3]) / 2.0)
            shape_label = "hexagonal" if plan.shape == "hexagon" else "square"
            self.summary_label.setText(
                f"Spatial block: size={format_meters(size_m)} ({plan.source}), "
                f"{shape_label}, {plan.n_blocks_x}×{plan.n_blocks_y} blocks, {len(folds)} fold(s)."
            )
            self.canvas.set_fold_colors(px, py, fold_id, crs=crs)
            self.canvas.set_block_polygons(plan, crs=crs)
            show_spatial_block(wizard.iface, "Split", "fold_preview", px, py, fold_id, plan, crs=crs)
        else:
            self.summary_label.setText(f"{len(folds)} fold(s) generated.")
            # A prior spatial-block preview may have left its fold-colored
            # block polygons on the embedded canvas — set_points() below
            # only ever touches the points layer, so without this the old
            # blocks would keep rendering underneath the new random/k-fold
            # preview. show_points() already clears the real QGIS "Split"
            # group's layers itself (see qgis_layers.clear_stage), so no
            # equivalent call is needed for that side.
            self.canvas.clear_block_polygons()
            if folds:
                _train, test = folds[0]
                labels = np.zeros(len(px), dtype=int)
                labels[test] = 1
                colors = {"0": "#4c72b0", "1": "#dd8452"}
                self.canvas.set_points(px, py, labels=labels, colors=colors, crs=crs)
                show_points(wizard.iface, "Split", "split_preview", px, py, labels=labels, colors=colors, crs=crs)

        new_key = self._snapshot()
        old_key = wizard.session.stage_hashes.get("split")
        if old_key is not None and old_key != new_key:
            wizard.invalidate_from(wizard.PAGE_SPLIT)
        wizard.session.stage_hashes["split"] = new_key

        self._stage_ok = True
        self._last_snapshot = new_key

    def _on_failed(self, err: str) -> None:
        self._stage_ok = False
        self.table.setRowCount(0)
        self.summary_label.setText("")
        self.error_label.setText(err.splitlines()[0] if err else "Split preview failed.")

    def invalidate(self) -> None:
        self._stage_ok = False
        self._last_snapshot = None
        self.table.setRowCount(0)
        self.summary_label.setText("")
        self.canvas.clear()
        clear_stage("Split")
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._stage_ok and self._snapshot() == self._last_snapshot

    def save_to_config(self, cfg) -> None:
        cfg.split.method = self._method()
        cfg.split.k = int(self.k.value())
        cfg.split.test_size = float(self.test_size.value())
        cfg.split.auto_block_size = self.auto_block.isChecked()
        cfg.split.block_size = self._block_size_meters()
        cfg.split.block_shape = self._block_shape_value()

    def validatePage(self) -> bool:
        self.save_to_config(self.wizard_ref.config)
        return True
