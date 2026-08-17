from __future__ import annotations

import numpy as np
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from ...core.config import BackgroundConfig
from ...core.stages import collect_labeled_points_and_extract
from ...core.units import distance_to_crs_units, is_geographic_crs
from ..page_utils import description_label, wrap_scrollable, wrapped_label
from ..qgis_layers import MAX_RENDER_POINTS, clear_stage, show_points
from ..widgets.embedded_canvas import EmbeddedPreviewCanvas
from ..workers import StagePageMixin, snapshot_key

# What each placement rule does, when it is the right one, and the values
# people actually use — same shape as the cross-validation, ensemble and
# algorithm pages.
_METHOD_HELP = {
    "random": (
        "<p>Draws points uniformly from anywhere in the predictor rasters "
        "that is not NoData, ignoring where the presence records are.</p>"
        "<p><b>Best for:</b> the general case. Presence-background modeling "
        "assumes a sample that characterises the environment available across "
        "the whole study area, rather than a set of claimed absences, and "
        "that is exactly what this produces.</p>"
        "<p><b>Common values:</b> 10,000 points is the usual default. Fewer "
        "than about 1,000 starts to describe the available environment "
        "poorly, while far more mostly costs runtime.</p>"
    ),
    "ratio": (
        "<p>Places points at random exactly as the option above does, but "
        "instead of a fixed number it draws a set multiple of however many "
        "presence records you have. At 4 per presence, 50 records give 200 "
        "pseudo-absences and 300 records give 1,200.</p>"
        "<p><b>Best for:</b> keeping the balance between the two classes "
        "steady when you do not know in advance how many records a species "
        "will have, or when you are running several species through the same "
        "settings. It also stops a fixed 10,000 from swamping a species with "
        "only 50 records, which matters most when the pseudo-absences are "
        "meant to stand in for real absences.</p>"
        "<p><b>Common values:</b> multipliers between 1 and 10, with 4 a "
        "reasonable starting point. A multiplier of 1 gives a balanced "
        "dataset with as many pseudo-absences as presences. Note that with "
        "very few records even a high multiplier gives a small sample, so "
        "check the resolved total shown beside the multiplier.</p>"
    ),
    "disk": (
        "<p>Keeps only locations whose distance to the <i>nearest</i> presence "
        "falls between a minimum and a maximum, so the points form a ring "
        "around the records with an empty hole in the middle.</p>"
        "<p><b>Best for:</b> pseudo-absences you intend to treat as real "
        "absences. The minimum distance keeps points out of the immediate "
        "surroundings of a record, where an apparent absence is usually just "
        "somewhere nobody has surveyed. The maximum keeps them inside the "
        "region the species could plausibly have reached, instead of "
        "somewhere it could never have got to.</p>"
        "<p><b>Common values:</b> set the minimum to roughly the species' "
        "dispersal or home range scale, and the maximum to the extent you "
        "believe was accessible to it. Leave the maximum at 0 for no upper "
        "limit. If the minimum is too large the ring comes out empty, and the "
        "run will tell you so.</p>"
    ),
    "sre": (
        "<p>Builds a rectilinear envelope around the conditions the species "
        "was recorded in, one minimum and maximum per predictor, then draws "
        "pseudo-absences only from places that fall outside it. A location "
        "counts as outside as soon as a single predictor is beyond its "
        "range.</p>"
        "<p><b>Best for:</b> picking absences that are environmentally, not "
        "just geographically, distinct from the presences. It is the right "
        "choice when most of the species' environmental space has already "
        "been sampled, because if your records genuinely cover the range of "
        "conditions the species tolerates then anything outside that envelope "
        "is confidently unsuitable.</p>"
        "<p><b>Common values:</b> a quantile of 0.025 trims the outer 2.5% of "
        "records at each end, so one mislocated or unusual record cannot "
        "stretch the envelope over the whole map. A quantile of 0 uses the "
        "outright minimum and maximum instead. Be aware of the assumption "
        "behind the method: if sampling has missed part of the niche, the "
        "envelope comes out too small and genuinely suitable places end up "
        "supplying the absences, which is worse than random background rather "
        "than better.</p>"
    ),
}

def _hint_row(widget, hint: str) -> QWidget:
    """A control and a short trailing hint as one hideable unit."""
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(widget)
    row.addWidget(QLabel(hint))
    row.addStretch()
    return container


class BackgroundPage(StagePageMixin, QWizardPage):
    def __init__(self, wizard) -> None:
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Background / pseudo-absence points")
        self.setSubTitle(
            "Presence-only modeling requires 'background' points sampled from the "
            "landscape. Choose how many and how to place them, and the description "
            "will update as you pick a method."
        )
        self.count = QSpinBox()
        self.count.setRange(100, 10_000_000)
        self.count.setValue(10_000)
        self.ratio = QDoubleSpinBox()
        self.ratio.setRange(0.1, 1000.0)
        self.ratio.setDecimals(1)
        self.ratio.setSingleStep(1.0)
        self.ratio.setValue(4.0)
        self.ratio_result = QLabel("")
        self.ratio_result.setWordWrap(True)

        self.random = QRadioButton("Random across raster extent")
        self.ratio_method = QRadioButton("Ratio to presences: random placement, count scaled by the presence total")
        self.disk = QRadioButton("Disk: between a minimum and maximum distance from presences")
        self.sre = QRadioButton("SRE: outside the presences' environmental envelope")
        self.random.setChecked(True)

        self.method_help = description_label()
        method_box = QGroupBox("Method")
        method_layout = QVBoxLayout(method_box)
        for radio in (self.random, self.ratio_method, self.disk, self.sre):
            method_layout.addWidget(radio)
        method_layout.addWidget(self.method_help)

        self.min_value = QDoubleSpinBox()
        self.min_value.setRange(0.0, 1e9)
        self.min_value.setDecimals(2)
        self.min_value.setValue(0.0)
        self.max_value = QDoubleSpinBox()
        self.max_value.setRange(0.0, 1e9)
        self.max_value.setDecimals(2)
        self.max_value.setValue(50.0)
        # One unit for both radii — a ring with its inner edge in km and its
        # outer in m would be an easy and invisible mistake to make.
        self.distance_unit = QComboBox()
        self.distance_unit.addItems(["km", "m"])

        # Each spin box shares its row with a hint, wrapped in a container
        # widget so hiding the row hides the hint with it — hiding a bare
        # layout's widgets one by one would strand the labels on screen.
        self.min_row = _hint_row(self.min_value, "(0 = no inner hole)")
        self.max_row = _hint_row(self.max_value, "(0 = no upper limit)")

        self.sre_quantile = QDoubleSpinBox()
        self.sre_quantile.setRange(0.0, 0.49)
        self.sre_quantile.setDecimals(3)
        self.sre_quantile.setSingleStep(0.005)
        self.sre_quantile.setValue(0.025)

        self.count_label = QLabel("Number of points:")
        self.ratio_label = QLabel("Per presence point:")
        self.ratio_row = _hint_row(self.ratio, "pseudo-absences per presence")
        self.unit_label = QLabel("Distance unit:")
        self.min_label = QLabel("Minimum distance:")
        self.max_label = QLabel("Maximum distance:")
        self.quantile_label = QLabel("Envelope quantile:")

        form = QFormLayout()
        form.addRow(self.count_label, self.count)
        form.addRow(self.ratio_label, self.ratio_row)
        form.addRow(QLabel(""), self.ratio_result)
        form.addRow(self.unit_label, self.distance_unit)
        form.addRow(self.min_label, self.min_row)
        form.addRow(self.max_label, self.max_row)
        form.addRow(self.quantile_label, self.sre_quantile)

        self.distance_help = wrapped_label(
            "Both distances are measured to the nearest presence and entered in km "
            "or m. They are converted automatically to your predictor rasters' CRS "
            "units (used directly for a projected/UTM CRS, and latitude-corrected "
            "for a geographic/degree-based CRS like EPSG:4326)."
        )
        self.conversion_label = QLabel("")
        self.conversion_label.setStyleSheet("color: #666; font-style: italic;")
        self.conversion_label.setWordWrap(True)

        self.run_btn = QPushButton("Sample Background")
        self.run_btn.setProperty("cls", "primary")
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.busy_label = QLabel("Sampling…")
        self.busy_label.setVisible(False)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #b00020;")
        self.error_label.setWordWrap(True)
        self.canvas = EmbeddedPreviewCanvas()

        layout = QVBoxLayout()
        layout.addWidget(method_box)
        layout.addLayout(form)
        layout.addWidget(self.distance_help)
        layout.addWidget(self.conversion_label)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.busy_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.canvas)
        layout.addStretch()
        wrap_scrollable(self, layout)

        self._stage_ok = False
        self._last_snapshot: str | None = None
        for spin in (self.count, self.ratio, self.min_value, self.max_value, self.sre_quantile):
            spin.valueChanged.connect(self.completeChanged)
        self.ratio.valueChanged.connect(self._update_ratio_result)
        for spin in (self.min_value, self.max_value):
            spin.valueChanged.connect(self._update_conversion_label)
        self.distance_unit.currentIndexChanged.connect(self.completeChanged)
        self.distance_unit.currentIndexChanged.connect(self._update_conversion_label)
        for radio in (self.random, self.ratio_method, self.disk, self.sre):
            radio.toggled.connect(self.completeChanged)
            radio.toggled.connect(self._update_field_state)
        self._update_conversion_label()
        self._update_field_state()

    def initializePage(self) -> None:
        # The predictor stack (for the CRS) and the cleaned occurrences (for
        # the presence count) are only guaranteed loaded once this page is
        # actually reached.
        self._update_conversion_label()
        self._update_ratio_result()

    def _presence_count(self) -> int | None:
        """How many of the loaded records count as presences, mirroring what
        collect_labeled_points_and_extract does: a presence column if one
        carries any positive value, otherwise every record."""
        occ = self.wizard_ref.session.occ
        if occ is None:
            return None
        if occ.presence.size and occ.presence.max() > 0:
            return int((occ.presence == 1).sum())
        return int(len(occ.x))

    def _update_ratio_result(self) -> None:
        n_presence = self._presence_count()
        if n_presence is None:
            self.ratio_result.setText("(load and clean the occurrences to see the total)")
            return
        # Resolved through the config's own method rather than multiplied
        # here, so this preview cannot drift from what the run actually draws.
        total = BackgroundConfig(
            method="ratio", ratio=self.ratio.value()
        ).resolve_count(n_presence)
        self.ratio_result.setText(
            f"= {total:,} pseudo-absences from {n_presence:,} presence points."
        )

    def _method(self) -> str:
        if self.random.isChecked():
            return "random"
        if self.ratio_method.isChecked():
            return "ratio"
        if self.disk.isChecked():
            return "disk"
        return "sre"

    def _update_field_state(self) -> None:
        method = self._method()
        self.method_help.setText(_METHOD_HELP[method])

        # The ratio method computes its own count, so the fixed-count box has
        # nothing to say while it is selected.
        is_ratio = method == "ratio"
        self.count_label.setVisible(not is_ratio)
        self.count.setVisible(not is_ratio)
        for w in (self.ratio_label, self.ratio_row, self.ratio_result):
            w.setVisible(is_ratio)
        if is_ratio:
            self._update_ratio_result()

        is_disk = method == "disk"
        for w in (
            self.unit_label, self.distance_unit,
            self.min_label, self.min_row, self.max_label, self.max_row,
            self.distance_help, self.conversion_label,
        ):
            w.setVisible(is_disk)

        is_sre = method == "sre"
        self.quantile_label.setVisible(is_sre)
        self.sre_quantile.setVisible(is_sre)

    def _to_meters(self, value: float) -> float:
        factor = 1000.0 if self.distance_unit.currentText() == "km" else 1.0
        return value * factor

    def _update_conversion_label(self) -> None:
        stack = self.wizard_ref.session.stack
        low = self._to_meters(self.min_value.value())
        high = self._to_meters(self.max_value.value())
        if stack is None:
            self.conversion_label.setText("(predictor CRS not loaded yet)")
            return
        outer = "no upper limit" if high <= 0 else None
        if is_geographic_crs(stack.crs):
            lat = (stack.bounds[1] + stack.bounds[3]) / 2.0
            low_crs = distance_to_crs_units(low, stack.crs, lat)
            high_crs = distance_to_crs_units(high, stack.crs, lat)
            outer = outer or f"{high_crs:.5f}°"
            self.conversion_label.setText(
                f"= {low_crs:.5f}° to {outer} in this raster's geographic CRS "
                f"(computed at centroid latitude {lat:.2f}°)."
            )
        else:
            outer = outer or f"{high:,.0f} m"
            self.conversion_label.setText(
                f"= {low:,.0f} m to {outer} in this raster's projected CRS units."
            )

    def _snapshot(self) -> str:
        wizard = self.wizard_ref
        return snapshot_key(
            self.count.value(),
            self.ratio.value(),
            self._method(),
            self.min_value.value(),
            self.max_value.value(),
            self.distance_unit.currentText(),
            self.sre_quantile.value(),
            wizard.session.stage_hashes.get("cleaning"),
        )

    def _on_run_clicked(self) -> None:
        self.error_label.setText("")
        wizard = self.wizard_ref
        self.save_to_config(wizard.config)
        cfg = wizard.config
        occ = wizard.session.occ
        stack = wizard.session.stack
        seed = cfg.random_seed

        def _work():
            rng = np.random.default_rng(seed)
            return collect_labeled_points_and_extract(cfg, occ, stack, rng)

        self.run_stage_async(
            _work, self._on_done, self._on_failed,
            button=self.run_btn, busy_widget=self.busy_label,
        )

    def _on_done(self, result) -> None:
        px, py, presence_flag, X_full, feature_names = result
        wizard = self.wizard_ref
        n_pres = int((presence_flag == 1).sum())
        n_bg = int((presence_flag == 0).sum())
        summary = f"Presence: {n_pres}, Background: {n_bg}."
        requested = wizard.config.background.resolve_count(n_pres)
        if n_bg < requested:
            # The disk and SRE methods both sample by rejection, so a tight
            # ring or a near-complete envelope can run out of room before
            # reaching the requested count. Say so rather than let the number
            # quietly differ from what was asked for.
            summary += (
                f" That is fewer than the {requested:,} requested, because the method ran "
                "out of eligible locations. Widen the ring, raise the envelope "
                "quantile, or ask for fewer points."
            )
        if len(px) > MAX_RENDER_POINTS:
            summary += f" (QGIS preview shows a {MAX_RENDER_POINTS:,}-point sample; the full set is used for modeling.)"
        self.summary_label.setText(summary)
        colors = {"1": "#2e7d32", "0": "#9e9e9e"}
        self.canvas.set_points(px, py, labels=presence_flag, colors=colors, crs=wizard.session.stack.crs)
        show_points(
            wizard.iface, "Background", "background_points",
            px, py, labels=presence_flag, colors=colors, crs=wizard.session.stack.crs,
        )

        new_key = self._snapshot()
        old_key = wizard.session.stage_hashes.get("background")
        if old_key is not None and old_key != new_key:
            wizard.invalidate_from(wizard.PAGE_BACKGROUND)
        wizard.session.px, wizard.session.py, wizard.session.presence_flag = px, py, presence_flag
        wizard.session.X_full, wizard.session.feature_names = X_full, feature_names
        wizard.session.stage_hashes["background"] = new_key

        self._stage_ok = True
        self._last_snapshot = new_key

    def _on_failed(self, err: str) -> None:
        self._stage_ok = False
        self.summary_label.setText("")
        self.error_label.setText(err.splitlines()[0] if err else "Background sampling failed.")

    def invalidate(self) -> None:
        self._stage_ok = False
        self._last_snapshot = None
        self.summary_label.setText("")
        self.canvas.clear()
        clear_stage("Background")
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._stage_ok and self._snapshot() == self._last_snapshot

    def save_to_config(self, cfg) -> None:
        cfg.background.count = int(self.count.value())
        cfg.background.ratio = float(self.ratio.value())
        cfg.background.method = self._method()
        cfg.background.min_distance = self._to_meters(self.min_value.value())
        cfg.background.max_distance = self._to_meters(self.max_value.value())
        cfg.background.sre_quantile = float(self.sre_quantile.value())

    def validatePage(self) -> bool:
        self.save_to_config(self.wizard_ref.config)
        return True
