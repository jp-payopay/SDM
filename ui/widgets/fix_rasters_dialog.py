from __future__ import annotations

from pathlib import Path

from qgis.core import QgsCoordinateReferenceSystem
from qgis.gui import QgsProjectionSelectionWidget
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.io.align import (
    RESAMPLING_HELP,
    RESAMPLING_METHODS,
    AlignmentTargetError,
    LayerOutput,
    build_target,
    default_nodata,
    default_output,
    dtype_choices,
    nodata_fits_dtype,
)
from ..page_utils import format_extent, format_resolution
from ..theme import APP_QSS, ERROR, FOREST_DARK

# Above this many cells the target grid is big enough to be worth a second
# look (a float32 band at 100 M cells is ~400 MB on disk) — warned about, not
# blocked: a legitimately large study area is the user's call.
_LARGE_GRID_CELLS = 100_000_000

_EXTENT_MODES = [
    ("intersection", "Intersection: only the area every layer covers (recommended)"),
    ("union", "Union: the whole area any layer covers"),
    ("reference", "Same as the reference layer"),
    ("custom", "Custom…"),
]

_RESOLUTION_MODES = [
    ("coarsest", "Coarsest: the largest pixel size present (recommended)"),
    ("finest", "Finest: the smallest pixel size present"),
    ("reference", "Same as the reference layer"),
    ("custom", "Custom…"),
]

# Every row starts on float32 / -9999 / bilinear, which is right for the
# continuous predictors that make up most SDM stacks. Categorical layers are
# the exception and have to be switched by hand, so the wording leads with
# how to tell which is which rather than just listing the options.
_LAYER_HELP = (
    "Set Data Type controls how each output stores its values. Use float "
    "(float32, float64) for continuous measurements such as temperature, "
    "rainfall, elevation, slope, distance and NDVI. Anything with a decimal "
    "point, or that you would take an average of, is float. Use int (int16, "
    "int32) for whole numbers that can be negative, and uint (uint8, uint16) "
    "for whole numbers that cannot, which usually means class codes in a land "
    "cover, soil or ecoregion map. uint8 holds 0 to 255 and uint16 holds 0 to "
    "65535. Narrower types save disk space, but writing continuous data into "
    "one rounds away everything after the decimal point.\n\n"
    "Set NoData is the value marking cells that no source pixel reaches. Pick "
    "something the layer's real data never uses. -9999 is the usual choice, "
    "and it has to fit inside the data type you picked.\n\n"
    "Resampling: bilinear interpolates smoothly and suits continuous data. "
    "Switch a categorical layer to nearest neighbour, because averaging class "
    "codes invents classes that do not exist. Halfway between forest and "
    "water is not a land cover type."
)


def _fmt(value: float) -> str:
    return f"{value:.10g}"


class FixRastersDialog(QDialog):
    """Choose the one grid a set of mismatched rasters will be resampled onto.

    The dialog only *plans* the fix — it reads nothing but the raster headers
    already in `profiles`, so it stays instant. The actual warping happens in
    the calling page's background worker (see `core.io.align.align_rasters`),
    driven by `target`, `out_dir`, `outputs` and `assumed_crs` once this
    dialog is accepted.
    """

    def __init__(self, profiles, kind: str = "predictor", parent=None) -> None:
        super().__init__(parent)
        self._profiles = list(profiles)
        self._default_ref = self._default_reference()
        self.setWindowTitle(f"Fix {kind} layers")
        self.resize(780, 700)
        self.setStyleSheet(APP_QSS)

        self.target = None
        self.out_dir = ""
        self.outputs: dict[str, LayerOutput] = {}
        self.assumed_crs = ""

        content = QVBoxLayout()
        intro = QLabel(
            "Every layer will be resampled onto one common grid and written as "
            "a new GeoTIFF, leaving your original files untouched. Choose which "
            "CRS, extent and resolution that grid should use."
        )
        intro.setWordWrap(True)
        content.addWidget(intro)
        content.addWidget(self._build_target_box())
        content.addWidget(self._build_layers_box())
        content.addLayout(self._build_output_row())
        content.addStretch()

        # The choices scroll; the running result and the buttons stay pinned,
        # so the dialog still works on a short screen without the user losing
        # sight of what the current settings would produce.
        holder = QWidget()
        holder.setLayout(content)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(holder)

        outer = QVBoxLayout(self)
        outer.addWidget(scroll, 1)

        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet(f"color: {FOREST_DARK}; font-weight: 600;")
        outer.addWidget(self.preview_label)
        self.problem_label = QLabel("")
        self.problem_label.setWordWrap(True)
        self.problem_label.setStyleSheet(f"color: {ERROR};")
        outer.addWidget(self.problem_label)

        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        self.fix_btn = QPushButton("Fix layers")
        self.fix_btn.setProperty("cls", "primary")
        self.fix_btn.clicked.connect(self._accept)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self.fix_btn)
        outer.addLayout(btn_row)

        self._connect_signals()
        self._update_preview()

    # ----- construction -----

    def _build_target_box(self) -> QGroupBox:
        box = QGroupBox("Target grid")
        form = QFormLayout(box)

        self.reference_combo = QComboBox()
        for p in self._profiles:
            self.reference_combo.addItem(p.name)
        self.reference_combo.setCurrentIndex(self._default_ref)
        self.reference_combo.setToolTip(
            "Used by the 'Same as the reference layer' choices below."
        )
        form.addRow(QLabel("Reference layer:"), self.reference_combo)

        self.crs_widget = QgsProjectionSelectionWidget()
        self.crs_widget.setCrs(
            QgsCoordinateReferenceSystem(self._profiles[self._default_ref].crs)
        )
        use_ref_crs = QPushButton("Use reference layer's CRS")
        use_ref_crs.clicked.connect(self._use_reference_crs)
        crs_row = QHBoxLayout()
        crs_row.addWidget(self.crs_widget, 1)
        crs_row.addWidget(use_ref_crs)
        form.addRow(QLabel("CRS:"), crs_row)

        # Only meaningful when at least one layer carries no CRS at all — a
        # bare .asc or a GeoTIFF stripped of its projection. Without one there
        # is no way to place that layer on the target grid, so rather than
        # refusing outright, let the user say what it should be assumed to be.
        self._crsless = [p.name for p in self._profiles if not p.crs]
        self.assumed_crs_widget = QgsProjectionSelectionWidget()
        if self._crsless:
            self.assumed_crs_widget.setCrs(self.crs_widget.crs())
            label = QLabel(
                "Assume CRS for layers with none\n(" + ", ".join(self._crsless) + "):"
            )
            label.setWordWrap(True)
            form.addRow(label, self.assumed_crs_widget)

        self.extent_combo = QComboBox()
        for _key, text in _EXTENT_MODES:
            self.extent_combo.addItem(text)
        form.addRow(QLabel("Extent:"), self.extent_combo)

        self.extent_edits = [QLineEdit() for _ in range(4)]
        self.extent_row_label = QLabel("Custom extent:")
        # The editors live in a container widget rather than a bare layout so
        # that hiding the row hides its "xmin"/"ymin" captions along with the
        # boxes — hiding a layout's widgets one by one would leave those behind.
        self.extent_row_widget = self._labelled_row(
            ("xmin", "ymin", "xmax", "ymax"), self.extent_edits
        )
        form.addRow(self.extent_row_label, self.extent_row_widget)

        self.resolution_combo = QComboBox()
        for _key, text in _RESOLUTION_MODES:
            self.resolution_combo.addItem(text)
        form.addRow(QLabel("Resolution:"), self.resolution_combo)

        self.res_edits = [QLineEdit() for _ in range(2)]
        self.res_row_label = QLabel("Custom resolution:")
        self.res_row_widget = self._labelled_row(("x", "y"), self.res_edits, stretch=True)
        form.addRow(self.res_row_label, self.res_row_widget)

        self._set_custom_rows_visible()
        return box

    @staticmethod
    def _labelled_row(captions, edits, stretch: bool = False) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        for caption, edit in zip(captions, edits):
            row.addWidget(QLabel(caption))
            row.addWidget(edit, 1)
        if stretch:
            row.addStretch()
        return container

    def _build_layers_box(self) -> QGroupBox:
        box = QGroupBox("Per-layer output")
        layout = QVBoxLayout(box)
        note = QLabel(_LAYER_HELP)
        note.setWordWrap(True)
        layout.addWidget(note)

        self.layer_table = QTableWidget()
        self.layer_table.setColumnCount(4)
        self.layer_table.setHorizontalHeaderLabels(
            ["Layer", "Set Data Type", "Set NoData", "Resampling"]
        )
        self.layer_table.setRowCount(len(self._profiles))
        self.layer_table.verticalHeader().setVisible(False)
        self.layer_table.horizontalHeader().setStretchLastSection(True)
        self.layer_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._dtype_combos: list[QComboBox] = []
        self._nodata_edits: list[QLineEdit] = []
        self._method_combos: list[QComboBox] = []
        for row, profile in enumerate(self._profiles):
            defaults = default_output(profile)
            self.layer_table.setItem(row, 0, QTableWidgetItem(profile.name))

            dtype_combo = QComboBox()
            dtype_combo.addItems(dtype_choices(profile))
            dtype_combo.setCurrentText(defaults.dtype)
            dtype_combo.setToolTip(
                f"Currently {profile.dtype} on disk. float for continuous "
                "measurements, int/uint for whole-number class codes."
            )
            self.layer_table.setCellWidget(row, 1, dtype_combo)
            self._dtype_combos.append(dtype_combo)

            nodata_edit = QLineEdit(_fmt(defaults.nodata))
            nodata_edit.setToolTip(
                "Value marking missing cells in the output. Type nan for "
                "floating-point layers."
            )
            self.layer_table.setCellWidget(row, 2, nodata_edit)
            self._nodata_edits.append(nodata_edit)

            method_combo = QComboBox()
            for method in RESAMPLING_METHODS:
                method_combo.addItem(RESAMPLING_HELP[method], method)
            method_combo.setCurrentIndex(RESAMPLING_METHODS.index(defaults.resampling))
            self.layer_table.setCellWidget(row, 3, method_combo)
            self._method_combos.append(method_combo)
        self.layer_table.resizeColumnsToContents()
        # Three of the four columns hold editors rather than text, so the
        # default text-sized row height would clip them.
        self.layer_table.resizeRowsToContents()
        self.layer_table.setMinimumHeight(160)
        layout.addWidget(self.layer_table)
        return box

    def _build_output_row(self) -> QHBoxLayout:
        default_dir = Path(self._profiles[0].path).parent / "sdm_aligned"
        self.out_edit = QLineEdit(str(default_dir))
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_out_dir)
        row = QHBoxLayout()
        row.addWidget(QLabel("Write fixed rasters to:"))
        row.addWidget(self.out_edit, 1)
        row.addWidget(browse_btn)
        return row

    def _connect_signals(self) -> None:
        self.reference_combo.currentIndexChanged.connect(self._update_preview)
        self.crs_widget.crsChanged.connect(self._update_preview)
        self.assumed_crs_widget.crsChanged.connect(self._update_preview)
        self.extent_combo.currentIndexChanged.connect(self._on_extent_mode_changed)
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_mode_changed)
        for edit in (*self.extent_edits, *self.res_edits, self.out_edit):
            edit.textChanged.connect(self._update_preview)
        # The per-layer settings can't change the target grid, but an
        # unstorable NoData/data-type pair still has to block the fix.
        for row, combo in enumerate(self._dtype_combos):
            combo.currentTextChanged.connect(
                lambda _text, r=row: self._on_dtype_changed(r)
            )
        for edit in self._nodata_edits:
            edit.textChanged.connect(self._update_preview)

    # ----- state -----

    def _default_reference(self) -> int:
        """The first layer using the most common CRS — a better starting
        point than simply the first layer, which may be the odd one out."""
        counts: dict[str, int] = {}
        for p in self._profiles:
            counts[p.crs] = counts.get(p.crs, 0) + 1
        best = max(counts.items(), key=lambda kv: (kv[1], kv[0] != ""))[0]
        for i, p in enumerate(self._profiles):
            if p.crs == best:
                return i
        return 0

    def _extent_mode(self) -> str:
        return _EXTENT_MODES[self.extent_combo.currentIndex()][0]

    def _resolution_mode(self) -> str:
        return _RESOLUTION_MODES[self.resolution_combo.currentIndex()][0]

    @staticmethod
    def _crs_string(crs) -> str:
        """A QgsCoordinateReferenceSystem as something rasterio/PROJ accepts.
        Prefer the authority id ("EPSG:32633"); a custom CRS has none, so fall
        back to its WKT."""
        if not crs.isValid():
            return ""
        return crs.authid() or crs.toWkt()

    def _crs_text(self) -> str:
        return self._crs_string(self.crs_widget.crs())

    def _assumed_crs_text(self) -> str:
        if not self._crsless:
            return ""
        return self._crs_string(self.assumed_crs_widget.crs())

    def _use_reference_crs(self) -> None:
        crs = self._profiles[self.reference_combo.currentIndex()].crs
        if crs:
            self.crs_widget.setCrs(QgsCoordinateReferenceSystem(crs))

    def _set_custom_rows_visible(self) -> None:
        extent_custom = self._extent_mode() == "custom"
        self.extent_row_label.setVisible(extent_custom)
        self.extent_row_widget.setVisible(extent_custom)
        res_custom = self._resolution_mode() == "custom"
        self.res_row_label.setVisible(res_custom)
        self.res_row_widget.setVisible(res_custom)

    def _on_extent_mode_changed(self) -> None:
        # Seed the custom boxes from whatever the previous choice produced, so
        # "Custom…" starts from a sensible extent to nudge rather than blank.
        if self._extent_mode() == "custom" and self.target is not None:
            for edit, value in zip(self.extent_edits, self.target.bounds):
                edit.setText(_fmt(value))
        self._set_custom_rows_visible()
        self._update_preview()

    def _on_resolution_mode_changed(self) -> None:
        if self._resolution_mode() == "custom" and self.target is not None:
            self.res_edits[0].setText(_fmt(self.target.res_x))
            self.res_edits[1].setText(_fmt(self.target.res_y))
        self._set_custom_rows_visible()
        self._update_preview()

    def _custom_values(self, edits: list[QLineEdit]) -> tuple[float, ...] | None:
        try:
            return tuple(float(e.text().strip()) for e in edits)
        except ValueError:
            return None

    def _on_dtype_changed(self, row: int) -> None:
        """Carry the row's NoData over to a value the new type can hold.
        Switching a layer to uint8 would otherwise leave the default -9999
        sitting there as an error the user has to decode and clear by hand;
        anything they type themselves is still validated, not overwritten."""
        dtype = self._dtype_combos[row].currentText()
        edit = self._nodata_edits[row]
        try:
            current = float(edit.text().strip())
        except ValueError:
            current = float("nan")
        if not nodata_fits_dtype(current, dtype):
            edit.setText(_fmt(default_nodata(dtype)))
        self._update_preview()

    def _layer_outputs(self) -> dict[str, LayerOutput]:
        """The per-layer table as LayerOutput settings. A NoData box that
        isn't a number at all becomes NaN, which `LayerOutput.validate`
        rejects for every integer type — so a typo is reported the same way
        an out-of-range value is, rather than silently reverting."""
        outputs: dict[str, LayerOutput] = {}
        for profile, dtype_combo, nodata_edit, method_combo in zip(
            self._profiles, self._dtype_combos, self._nodata_edits, self._method_combos
        ):
            try:
                nodata = float(nodata_edit.text().strip())
            except ValueError:
                nodata = float("nan")
            outputs[profile.name] = LayerOutput(
                resampling=method_combo.currentData(),
                dtype=dtype_combo.currentText(),
                nodata=nodata,
            )
        return outputs

    # ----- live preview -----

    def _update_preview(self) -> None:
        self.target = None
        self.preview_label.setText("")
        self.problem_label.setText("")
        try:
            target = build_target(
                self._profiles,
                crs=self._crs_text(),
                extent_mode=self._extent_mode(),
                resolution_mode=self._resolution_mode(),
                reference=self.reference_combo.currentIndex(),
                assumed_crs=self._assumed_crs_text(),
                custom_bounds=self._custom_values(self.extent_edits),
                custom_resolution=self._custom_values(self.res_edits),
            )
        except AlignmentTargetError as exc:
            self.problem_label.setText(str(exc))
            self.fix_btn.setEnabled(False)
            return
        except Exception as exc:  # a CRS pair proj can't transform between
            self.problem_label.setText(f"Cannot build that grid: {exc}")
            self.fix_btn.setEnabled(False)
            return

        self.target = target
        self.preview_label.setText(
            f"Result: {len(self._profiles)} layers at {target.width} × {target.height} "
            f"pixels ({target.cells:,} cells each), resolution "
            f"{format_resolution((target.res_x, target.res_y))}, "
            f"extent {format_extent(target.realized_bounds)}."
        )
        for name, layer in self._layer_outputs().items():
            try:
                layer.validate(name)
            except AlignmentTargetError as exc:
                self.problem_label.setText(str(exc))
                self.fix_btn.setEnabled(False)
                return
        if not self.out_edit.text().strip():
            self.problem_label.setText("Choose a folder to write the fixed rasters to.")
            self.fix_btn.setEnabled(False)
            return
        if target.cells > _LARGE_GRID_CELLS:
            self.problem_label.setText(
                f"That grid is {target.cells:,} cells per layer, so writing it will "
                "take a while and a lot of disk space. A coarser resolution or a "
                "smaller extent would be faster."
            )
        self.fix_btn.setEnabled(True)

    # ----- output -----

    def _browse_out_dir(self) -> None:
        start = self.out_edit.text().strip() or str(Path(self._profiles[0].path).parent)
        path = QFileDialog.getExistingDirectory(self, "Folder for fixed rasters", start)
        if path:
            self.out_edit.setText(path)

    def _accept(self) -> None:
        if self.target is None:
            return
        out_dir = Path(self.out_edit.text().strip())
        existing = [f"{p.name}.tif" for p in self._profiles if (out_dir / f"{p.name}.tif").exists()]
        if existing:
            shown = ", ".join(existing[:6]) + (f" and {len(existing) - 6} more" if len(existing) > 6 else "")
            answer = QMessageBox.question(
                self,
                "Replace existing files?",
                f"{out_dir} already contains {len(existing)} file(s) with the same "
                f"names ({shown}). They will be overwritten. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.out_dir = str(out_dir)
        self.outputs = self._layer_outputs()
        self.assumed_crs = self._assumed_crs_text()
        self.accept()
