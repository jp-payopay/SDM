from __future__ import annotations

import sys
from pathlib import Path

from qgis.PyQt.QtCore import QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMenu, QMessageBox, QToolBar

PLUGIN_DIR = Path(__file__).parent
MENU_LABEL = "&SDM"
ACTION_LABEL = "Run SDM…"
_PKG_MAP = {"sklearn": "scikit-learn", "pygam": "pyGAM"}


def _python_executable() -> str:
    """sys.executable inside QGIS's embedded interpreter is the QGIS binary
    itself (qgis-bin.exe on the Windows standalone installer, confirmed by
    a real user report), not a Python interpreter — spawning it directly
    launches a second QGIS instance instead of running pip. QGIS bundles a
    normal CPython distribution under sys.exec_prefix (e.g.
    .../QGIS 4.0.1/apps/Python312/), which does contain python.exe/python3.exe
    directly at its root, same layout as a standard python.org install."""
    if sys.platform.startswith("win"):
        for prefix in {sys.exec_prefix, sys.base_exec_prefix}:
            if not prefix:
                continue
            for name in ("python3.exe", "python.exe"):
                candidate = Path(prefix) / name
                if candidate.exists():
                    return str(candidate)
    return sys.executable


class SDMWizardPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action: QAction | None = None
        self.menu: QMenu | None = None
        self.toolbar: QToolBar | None = None
        self.dock = None
        self._wizard = None
        self._dep_dialog = None

    def initGui(self) -> None:
        icon_path = PLUGIN_DIR / "icon.png"
        icon = QIcon(str(icon_path)) if icon_path.exists() else QIcon()
        self.action = QAction(icon, ACTION_LABEL, self.iface.mainWindow())
        self.action.triggered.connect(self._launch)

        # Conventional Plugins-menu entry (kept for discoverability alongside
        # the plugin manager's own listing).
        self.iface.addPluginToMenu(MENU_LABEL, self.action)

        # Dedicated named toolbar, instead of the generic Plugins toolbar
        # icon, so SDM isn't shown twice on two different toolbars.
        self.toolbar = self.iface.addToolBar("SDM")
        self.toolbar.setObjectName("SDMToolbar")
        self.toolbar.addAction(self.action)

        # Dedicated top-level menu (next to Project/Edit/.../Help), inserted
        # before the right-aligned standard menu cluster (Help) rather than
        # appended after everything.
        self.menu = QMenu(MENU_LABEL, self.iface.mainWindow())
        self.menu.setObjectName("mSDMMenu")
        self.menu.addAction(self.action)
        menu_bar = self.iface.mainWindow().menuBar()
        menu_bar.insertMenu(self.iface.firstRightStandardMenu().menuAction(), self.menu)

        # Launcher + status dock panel (View -> Panels).
        from .ui.widgets.sdm_dock import SDMDockWidget

        self.dock = SDMDockWidget(self.iface.mainWindow())
        self.dock.launch_requested.connect(self._launch)
        self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)

    def unload(self) -> None:
        if self.action is not None:
            self.iface.removePluginMenu(MENU_LABEL, self.action)
            self.action = None
        if self.toolbar is not None:
            self.iface.mainWindow().removeToolBar(self.toolbar)
            self.toolbar.deleteLater()
            self.toolbar = None
        if self.menu is not None:
            self.iface.mainWindow().menuBar().removeAction(self.menu.menuAction())
            self.menu.deleteLater()
            self.menu = None
        if self.dock is not None:
            self.iface.removeDockWidget(self.dock)
            self.dock.deleteLater()
            self.dock = None
        if self._wizard is not None:
            # Otherwise a still-open wizard survives unload() wired to a
            # dock/iface that a subsequent disable-then-enable cycle has
            # already torn down and replaced, and reacts against stale
            # objects. close() is refused (via SDMWizard.closeEvent, returning
            # False) if a background run is still in flight, same as a
            # user-initiated close — we don't force-destroy a running QThread
            # here either. In that case leave self._wizard set: nulling it
            # unconditionally would orphan the still-open, still-running
            # wizard (wired to the dock we just tore down above) and let the
            # next _open_wizard() call spawn a second, independent one.
            try:
                closed = self._wizard.close()
            except RuntimeError:
                closed = True
            if closed:
                self._wizard = None

    def _launch(self) -> None:
        results = _check_dependencies()
        if any(r[1] != "ok" for r in results):
            self._show_dependency_dialog(results)
            return
        self._open_wizard()

    def _open_wizard(self) -> None:
        if self._wizard is not None:
            # With WA_DeleteOnClose set on the wizard, a closed instance is
            # gone (isVisible() raises RuntimeError on the deleted C++
            # object); a merely-hidden-behind-other-windows one is still
            # visible, so raise/activate is enough.
            try:
                visible = self._wizard.isVisible()
            except RuntimeError:
                visible = False
            if visible:
                self._wizard.raise_()
                self._wizard.activateWindow()
                return
            self._wizard = None

        from .ui.wizard import SDMWizard

        wizard = SDMWizard(parent=self.iface.mainWindow(), iface=self.iface)
        if self.dock is not None:
            wizard.run_completed.connect(self.dock.set_last_result)
        self._wizard = wizard
        wizard.show()

    def _show_dependency_dialog(self, results: list[tuple[str, str, str]]) -> None:
        from .ui.widgets.dependency_dialog import DependencyInstallDialog

        # Reuse a still-installing dialog rather than opening a second one:
        # closing the dialog via the X button lets its background pip
        # install keep running (see DependencyInstallDialog.closeEvent), so
        # without this, clicking the toolbar action again before that
        # install finishes would spawn a second concurrent `pip install` of
        # the same packages against the same QGIS Python environment.
        dialog = None
        if self._dep_dialog is not None:
            try:
                installing = self._dep_dialog.is_installing()
            except RuntimeError:
                installing = False
            if installing:
                dialog = self._dep_dialog
            if dialog is None:
                self._dep_dialog = None

        if dialog is None:
            missing = [_PKG_MAP.get(n, n) for n, s, _ in results if s == "missing"]
            dialog = DependencyInstallDialog(
                self.iface.mainWindow(), _missing_deps_message(results), missing, _python_executable()
            )
            self._dep_dialog = dialog

        dialog.exec()
        if not dialog.is_installing():
            self._dep_dialog = None
        if not dialog.installed_ok:
            return

        results2 = _check_dependencies()
        if all(r[1] == "ok" for r in results2):
            self._open_wizard()
        else:
            QMessageBox.information(
                self.iface.mainWindow(),
                "SDM",
                "Packages installed, but some imports still fail. This can happen "
                "with native extensions (e.g. xgboost). Please restart QGIS and try "
                "launching SDM again.\n\n" + _missing_deps_message(results2),
            )

    @staticmethod
    def tr(message: str) -> str:
        return QCoreApplication.translate("SDMWizardPlugin", message)


def _check_dependencies() -> list[tuple[str, str, str]]:
    """Return (name, status, detail) for each dep. status is one of:
    'ok', 'missing', 'broken'.
    """
    required = [
        "numpy",
        "pandas",
        "scipy",
        "rasterio",
        "fiona",
        "sklearn",
        "matplotlib",
        "jinja2",
        "joblib",
        "pygam",
        "xgboost",
        "elapid",
    ]
    results: list[tuple[str, str, str]] = []
    for name in required:
        try:
            __import__(name)
            results.append((name, "ok", ""))
        except ModuleNotFoundError as exc:
            results.append((name, "missing", str(exc)))
        except Exception as exc:
            results.append((name, "broken", f"{type(exc).__name__}: {exc}"))
    return results


def _missing_deps_message(results: list[tuple[str, str, str]]) -> str:
    missing = [(n, d) for n, s, d in results if s == "missing"]
    broken = [(n, d) for n, s, d in results if s == "broken"]
    lines: list[str] = []
    if missing:
        lines.append("The following packages are not installed:")
        for n, _ in missing:
            lines.append(f"  - {_PKG_MAP.get(n, n)}")
        joined = " ".join(_PKG_MAP.get(n, n) for n, _ in missing)
        lines.append("")
        lines.append("Click 'Install missing package(s)' below, or run manually:")
        lines.append(f"    {_python_executable()} -m pip install {joined}")
    if broken:
        if lines:
            lines.append("")
        lines.append("The following packages are installed but failed to import:")
        for n, d in broken:
            lines.append(f"  - {n}: {d}")
        lines.append("")
        lines.append(
            "This usually means a native dependency is missing. On macOS ARM,\n"
            "xgboost commonly needs OpenMP:  brew install libomp"
        )
    return "\n".join(lines)
