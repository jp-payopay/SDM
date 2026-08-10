from __future__ import annotations

# Design tokens for the plugin's visual identity — a field-guide/topographic
# palette (forest green primary, clay as a rare secondary accent, warm paper
# ground) grounded in the subject (species distribution modeling), not a
# generic SaaS blue/purple. One committed light theme, applied via QSS
# regardless of whichever theme QGIS itself is running, since QWizard/QDialog
# otherwise inherit QGIS's live application palette (including dark themes)
# rather than rendering independently.
INK = "#20261F"
CANVAS = "#F6F5F0"
RAISED = "#FFFFFF"
FOREST = "#2F5D50"
FOREST_DARK = "#21453B"
FOREST_TINT = "#E4EDE9"
MOSS = "#7C9082"
MOSS_SOFT = "#E4E7DF"
CLAY = "#B5622E"
CLAY_SOFT = "#F1DECE"
ERROR = "#A3342A"
BORDER = "#DAD6C9"
DISABLED_TEXT = "#9A998F"

# Applied to any top-level widget in the plugin (the wizard, the dependency
# dialog, the dock panel) so the whole plugin — not just the wizard — reads
# as one considered piece rather than default-Qt gray.
APP_QSS = f"""
QDialog, QWizard, QDockWidget {{
    background: {CANVAS};
}}
QWidget {{
    color: {INK};
    font-size: 9.5pt;
}}
QWizardPage {{
    background: {CANVAS};
}}
QLabel {{
    background: transparent;
}}
QGroupBox {{
    font-weight: 600;
    color: {FOREST_DARK};
    border: 1px solid {BORDER};
    border-radius: 5px;
    margin-top: 12px;
    padding-top: 14px;
    background: {RAISED};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
QGroupBox[cls="accent"] {{
    border-left: 4px solid {CLAY};
}}
QPushButton {{
    background: {RAISED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 600;
    color: {INK};
}}
QPushButton:hover {{
    border-color: {MOSS};
}}
QPushButton:pressed {{
    background: {MOSS_SOFT};
}}
QPushButton:disabled {{
    color: {DISABLED_TEXT};
    background: {CANVAS};
    border-color: {BORDER};
}}
QPushButton[cls="primary"] {{
    background: {FOREST};
    border-color: {FOREST};
    color: white;
}}
QPushButton[cls="primary"]:hover {{
    background: {FOREST_DARK};
    border-color: {FOREST_DARK};
}}
QPushButton[cls="primary"]:disabled {{
    background: {MOSS_SOFT};
    border-color: {MOSS_SOFT};
    color: #ffffff;
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background: {RAISED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: {FOREST_TINT};
    selection-color: {FOREST_DARK};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
    border-color: {FOREST};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {DISABLED_TEXT};
    background: {CANVAS};
}}
QCheckBox, QRadioButton {{
    spacing: 8px;
    background: transparent;
}}
QCheckBox:disabled, QRadioButton:disabled {{
    color: {DISABLED_TEXT};
}}
QRadioButton::indicator, QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 2px solid {MOSS};
    background: {RAISED};
}}
QRadioButton::indicator {{
    border-radius: 8px;
}}
QCheckBox::indicator {{
    border-radius: 3px;
}}
QRadioButton::indicator:hover, QCheckBox::indicator:hover {{
    border-color: {FOREST};
}}
QRadioButton::indicator:checked {{
    border: 2px solid {FOREST};
    background: qradialgradient(
        cx:0.5, cy:0.5, radius:0.55, fx:0.5, fy:0.5,
        stop:0 {FOREST}, stop:0.45 {FOREST}, stop:0.55 {RAISED}, stop:1 {RAISED}
    );
}}
QCheckBox::indicator:checked {{
    border: 2px solid {FOREST};
    background: {FOREST};
}}
QRadioButton::indicator:disabled, QCheckBox::indicator:disabled {{
    border-color: {BORDER};
    background: {CANVAS};
}}
QListWidget, QTableWidget, QTreeWidget, QTreeView {{
    background: {RAISED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    gridline-color: {BORDER};
    alternate-background-color: {CANVAS};
}}
QTreeWidget::item, QTreeView::item, QTableWidget::item, QListWidget::item {{
    color: {INK};
    padding: 2px 2px;
}}
QHeaderView::section {{
    background: {MOSS_SOFT};
    color: {FOREST_DARK};
    padding: 5px 8px;
    border: none;
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
}}
QTableWidget::item:selected, QListWidget::item:selected,
QTreeWidget::item:selected, QTreeView::item:selected {{
    background: {FOREST_TINT};
    color: {FOREST_DARK};
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: {CANVAS};
}}
QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {RAISED};
    text-align: center;
    color: {INK};
}}
QProgressBar::chunk {{
    background: {FOREST};
    border-radius: 3px;
}}
QScrollBar:vertical {{
    background: {CANVAS};
    width: 13px;
    margin: 0;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {MOSS};
    min-height: 32px;
    border-radius: 4px;
    margin: 2px;
}}
QScrollBar::handle:vertical:hover {{
    background: {FOREST};
}}
QScrollBar::handle:vertical:pressed {{
    background: {FOREST_DARK};
}}
QScrollBar:horizontal {{
    background: {CANVAS};
    height: 13px;
    margin: 0;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {MOSS};
    min-width: 32px;
    border-radius: 4px;
    margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {FOREST};
}}
QScrollBar::handle:horizontal:pressed {{
    background: {FOREST_DARK};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
    height: 0;
    background: none;
    border: none;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}
QToolTip {{
    background: {FOREST_DARK};
    color: white;
    border: none;
    padding: 4px 7px;
}}
"""

# The step sidebar (see ui/widgets/step_sidebar.py) is a separate, darker
# surface (the plugin's one deliberately bold element), styled on its own
# rather than folded into APP_QSS.
SIDEBAR_QSS = f"""
QWidget#WizardSidebar {{
    background: {FOREST_DARK};
}}
QLabel#SidebarBrand {{
    color: #B9CFC4;
    font-size: 8.5pt;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 0 20px 14px 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    margin: 0 0 8px 0;
}}
QLabel#SidebarLabel {{
    color: #B9CFC4;
    font-size: 9pt;
    background: transparent;
}}
QLabel#SidebarLabel[state="current"] {{
    color: #FFFFFF;
    font-weight: 600;
}}
QLabel#SidebarLabel[state="done"] {{
    color: #D9E6DF;
}}
QLabel#SidebarMarker {{
    border-radius: 10px;
    border: 1px solid #5A7B6D;
    color: #9FB9AC;
    font-size: 8pt;
    background: transparent;
}}
QLabel#SidebarMarker[state="current"] {{
    background: #EEF3EE;
    border-color: #EEF3EE;
    color: {FOREST_DARK};
    font-weight: 700;
}}
QLabel#SidebarMarker[state="done"] {{
    background: {CLAY};
    border-color: {CLAY};
    color: white;
}}
QWidget#SidebarStep[state="current"] {{
    background: rgba(255, 255, 255, 0.07);
}}
QLabel#SidebarFoot {{
    color: #7F9D90;
    font-size: 8pt;
    padding: 10px 20px 0 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}}
"""
