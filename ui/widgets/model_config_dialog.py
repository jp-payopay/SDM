from __future__ import annotations

import ast
import json

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyledItemDelegate,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ..theme import APP_QSS

# Keys added by build_model_config for context only (not real constructor
# arguments), shown but never editable.
_NOTE_SUFFIX = "_note"

# Hyperparameters with a small set of valid string choices — shown as a
# dropdown instead of free text so a typo can't produce an invalid/nonexistent
# argument.
_CHOICES: dict[str, list[str]] = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "gamma": ["scale", "auto"],
    "max_features": ["sqrt", "log2", "None"],
}
# kernel only ever accepts these four exact strings, so its dropdown is
# locked; gamma and max_features also legitimately accept a number (a float
# gamma, or an int/float max_features), so theirs stays editable.
_STRICT_CHOICES = {"kernel"}

# MaxEnt's feature_types isn't a single value like the params above — it's a
# combination of these classes — so it gets its own checkbox popup (see
# _FeatureTypesDialog) rather than a dropdown.
_FEATURE_CLASSES = ["linear", "quadratic", "product", "hinge", "threshold"]


def _coerce(text: str):
    """Turn an edited cell back into a real Python value. Numbers, lists,
    tuples, True/False and None parse to their proper types; anything that
    is not a valid literal (a bare word like sqrt, rbf, scale) is kept as a
    plain string.
    """
    text = text.strip()
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


class _ValueColumnDelegate(QStyledItemDelegate):
    """Lets only the Value column (column 1) of parameter rows open an editor,
    so parameter names and the algorithm header rows stay read-only. Rows
    whose parameter name has a known fixed set of choices (_CHOICES) get an
    editable dropdown instead of a plain text box, so a mistyped value can't
    produce an invalid/nonexistent argument. feature_types (MaxEnt) opens its
    own checkbox popup instead (see ModelConfigDialog._on_item_double_clicked),
    so it's excluded from the normal in-place editor entirely.
    """

    def createEditor(self, parent, option, index):
        if index.column() != 1 or not index.parent().isValid():
            return None
        key = index.sibling(index.row(), 0).data(Qt.ItemDataRole.UserRole)
        if key == "feature_types":
            return None
        if key in _CHOICES:
            combo = QComboBox(parent)
            combo.setEditable(key not in _STRICT_CHOICES)
            combo.addItems(_CHOICES[key])
            return combo
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        if isinstance(editor, QComboBox):
            text = str(index.data(Qt.ItemDataRole.EditRole))
            found = editor.findText(text)
            if found >= 0:
                editor.setCurrentIndex(found)
            else:
                editor.setEditText(text)
            return
        super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)
            return
        super().setModelData(editor, model, index)


class _FeatureTypesDialog(QDialog):
    """Popup editor for MaxEnt's feature_types. Unlike every other
    hyperparameter, this one is a combination of classes rather than a single
    value, so it doesn't fit the tree's normal single-editor pattern —
    opened directly from ModelConfigDialog on double-click instead.
    """

    def __init__(self, checked: set[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MaxEnt feature types")
        self.setStyleSheet(APP_QSS)
        layout = QVBoxLayout(self)
        info = QLabel(
            "Choose which feature classes MaxEnt fits. Leave all unchecked to "
            "restore automatic selection based on the presence count."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        self._boxes: dict[str, QCheckBox] = {}
        for name in _FEATURE_CLASSES:
            box = QCheckBox(name)
            box.setChecked(name in checked)
            layout.addWidget(box)
            self._boxes[name] = box

        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("OK")
        ok_btn.setProperty("cls", "primary")
        ok_btn.clicked.connect(self.accept)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

    @property
    def selected(self) -> list[str] | None:
        chosen = [name for name in _FEATURE_CLASSES if self._boxes[name].isChecked()]
        return chosen or None


class ModelConfigDialog(QDialog):
    """View and edit the exact hyperparameters each selected algorithm will be
    built with. Double-click a value to change it; Apply keeps whatever you
    changed for the run, Cancel discards it. The values can also be exported
    to JSON.
    """

    def __init__(
        self,
        config: dict[str, dict],
        labels: dict[str, str] | None = None,
        note: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config = {algo: dict(params) for algo, params in config.items()}
        self._labels = labels or {}
        # Starts as an unchanged copy; replaced with the edited values only if
        # the user clicks Apply.
        self.edited_config = {algo: dict(params) for algo, params in config.items()}

        self.setWindowTitle("Model configuration")
        self.resize(580, 520)
        self.setStyleSheet(APP_QSS)

        layout = QVBoxLayout(self)
        help_label = QLabel(
            "Double-click a value in the Value column to change it. "
            "Click Apply to use your changes for this run, or Cancel to keep "
            "the defaults."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        if note:
            note_label = QLabel(note)
            note_label.setWordWrap(True)
            layout.addWidget(note_label)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Algorithm / parameter", "Value"])
        self.tree.setColumnWidth(0, 270)
        self.tree.setItemDelegate(_ValueColumnDelegate(self.tree))
        self.tree.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        for algo, params in self._config.items():
            top = QTreeWidgetItem([self._labels.get(algo, algo), ""])
            top.setData(0, Qt.ItemDataRole.UserRole, algo)
            bold = top.font(0)
            bold.setBold(True)
            top.setFont(0, bold)
            # Header rows are not parameters, so keep them non-editable.
            top.setFlags(top.flags() & ~Qt.ItemFlag.ItemIsEditable)
            for key, value in params.items():
                child = QTreeWidgetItem(top, [key, str(value)])
                child.setData(0, Qt.ItemDataRole.UserRole, key)
                if key.endswith(_NOTE_SUFFIX):
                    child.setFlags(child.flags() & ~Qt.ItemFlag.ItemIsEditable)
                else:
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsEditable)
            self.tree.addTopLevelItem(top)
            top.setExpanded(True)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save as JSON...")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton("Apply changes")
        apply_btn.setProperty("cls", "primary")
        apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(save_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(apply_btn)
        layout.addLayout(btn_row)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """feature_types has no in-place editor (see _ValueColumnDelegate) —
        double-clicking its Value cell opens the checkbox popup instead."""
        if column != 1 or item.data(0, Qt.ItemDataRole.UserRole) != "feature_types":
            return
        current = _coerce(item.text(1))
        checked = set(current) if isinstance(current, (list, tuple)) else set()
        dlg = _FeatureTypesDialog(checked, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            item.setText(1, str(dlg.selected))

    def _collect(self) -> dict[str, dict]:
        """Read the current tree contents back into an algorithm-to-parameters
        dict, coercing each edited value to a real Python type."""
        result: dict[str, dict] = {}
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            algo = top.data(0, Qt.ItemDataRole.UserRole)
            params: dict = {}
            for j in range(top.childCount()):
                child = top.child(j)
                key = child.data(0, Qt.ItemDataRole.UserRole)
                if key.endswith(_NOTE_SUFFIX):
                    params[key] = self._config.get(algo, {}).get(key)
                else:
                    params[key] = _coerce(child.text(1))
            result[algo] = params
        return result

    def _apply(self) -> None:
        self.edited_config = self._collect()
        self.accept()

    def _save(self) -> None:
        # Persist exactly what is on screen now, edits included.
        current = self._collect()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save model configuration", "model_hyperparameters.json", "JSON files (*.json)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(current, f, indent=2, default=str)
