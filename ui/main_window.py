"""Main application window."""

import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget, QTabBar,
    QFileDialog, QStatusBar, QMessageBox, QApplication,
)

from config import get_project_path, save_project_path, DEFAULT_PATHS, get_theme, save_theme
from scanners.dvnr import scan_dvnr
from scanners.dof import scan_dof
from scanners.odt import scan_odt
from scanners.vbp import scan_vbp
from scanners.normnet import scan_normnet
from screens.runs import RunsScreen
from screens.plots import PlotsScreen
from screens.monitor import MonitorScreen
from screens.launcher import LauncherScreen
from screens.vbp_wizard import VBPWizardScreen
from ui import theme
from ui.sidebar import Sidebar

_SCANNERS = {"DVNR": scan_dvnr, "ODT": scan_odt, "VBP": scan_vbp,
             "DOF": scan_dof, "NORMNET": scan_normnet}
_VBP_SUBTYPE_SUFFIX = "_TP"

# Projects that present a sub-type tab bar: one tab per top-level folder.
#   VBP     — folders filtered by the _TP suffix; shows Launcher/Wizard tabs.
#   NORMNET — every top-level folder is an arch family (no suffix filter),
#             scanned recursively for its experiments; no launcher/wizard.
_SUBTYPED = {
    "VBP":     {"suffix": _VBP_SUBTYPE_SUFFIX, "scanner": scan_vbp,     "extra_tabs": True},
    "NORMNET": {"suffix": None,                "scanner": scan_normnet, "extra_tabs": False},
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ExpHandler")
        self.resize(1400, 820)

        # Cache keyed by project name or "<project>/<subtype>" (VBP, NORMNET)
        self._loaded: dict[str, list | None] = {"DVNR": None, "ODT": None, "DOF": None}
        self._active_project: str | None = None
        self._active_subtype: dict[str, str | None] = {}  # per subtyped project

        # ── Central layout ──────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.project_selected.connect(self._on_project_selected)
        self.sidebar.change_path_requested.connect(self._on_change_path)
        self.sidebar.refresh_requested.connect(self._on_refresh)
        self.sidebar.theme_toggle_requested.connect(self._on_theme_toggle)
        h_layout.addWidget(self.sidebar)

        self._sep = QWidget()
        self._sep.setFixedWidth(1)
        self._sep.setStyleSheet(f"background-color: {theme.BORDER};")
        h_layout.addWidget(self._sep)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Sub-type bar (VBP only): one tab per *_TP folder
        self.subtype_bar = QTabBar()
        self.subtype_bar.setShape(QTabBar.RoundedNorth)
        self.subtype_bar.setExpanding(False)
        self.subtype_bar.setDrawBase(False)
        self.subtype_bar.setVisible(False)
        self.subtype_bar.currentChanged.connect(self._on_subtype_changed)
        content_layout.addWidget(self.subtype_bar)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.runs_screen = RunsScreen()
        self.plots_screen = PlotsScreen()
        self.monitor_screen = MonitorScreen()
        self.launcher_screen = LauncherScreen()
        self.wizard_screen = VBPWizardScreen()
        self.tabs.addTab(self.runs_screen, "Runs")
        self.tabs.addTab(self.plots_screen, "Plots")
        self.tabs.addTab(self.monitor_screen, "Monitor")
        self._launcher_tab_idx = self.tabs.addTab(self.launcher_screen, "Launcher")
        self._wizard_tab_idx = self.tabs.addTab(self.wizard_screen, "VBP Wizard")
        self.tabs.setTabVisible(self._launcher_tab_idx, False)
        self.tabs.setTabVisible(self._wizard_tab_idx, False)
        content_layout.addWidget(self.tabs, stretch=1)

        h_layout.addWidget(content, stretch=1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Select a project from the sidebar.")

    # ── Theme handling ───────────────────────────────────────────────
    def _on_theme_toggle(self):
        next_theme = "light" if theme.current_theme == "dark" else "dark"
        theme.set_theme(next_theme)
        save_theme(next_theme)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(theme.QSS)
        self._sep.setStyleSheet(f"background-color: {theme.BORDER};")
        self.sidebar.refresh_styles()

    # ── Project selection ────────────────────────────────────────────
    def _on_project_selected(self, project: str):
        self._active_project = project
        if project in _SUBTYPED:
            self._activate_subtyped(project)
            return

        self._hide_subtype_bar()
        root_path = get_project_path(project)
        if not root_path or not os.path.isdir(root_path):
            root_path = self._ask_for_path(project)
            if not root_path:
                self.sidebar.set_active_silent(project)
                return
            save_project_path(project, root_path)
            self._loaded[project] = None

        if self._loaded.get(project) is None:
            self._scan(project, root_path)
        self._display(project)

    # ── Change path button (📁) ───────────────────────────────────────
    def _on_change_path(self, project: str):
        root_path = self._ask_for_path(project)
        if not root_path:
            return
        save_project_path(project, root_path)
        self._active_project = project
        self.sidebar.set_active_silent(project)
        if project in _SUBTYPED:
            self._invalidate_subtype_cache(project)
            self._activate_subtyped(project)
        else:
            self._hide_subtype_bar()
            self._loaded[project] = None
            self._scan(project, root_path)
            self._display(project)

    # ── Refresh button ───────────────────────────────────────────────
    def _on_refresh(self):
        project = self._active_project
        if not project:
            self.status.showMessage("Select a project first.")
            return
        if project in _SUBTYPED:
            self._invalidate_subtype_cache(project)
            self._activate_subtyped(project)
            return
        root_path = get_project_path(project)
        if not root_path or not os.path.isdir(root_path):
            return
        self._loaded[project] = None
        self._scan(project, root_path)
        self._display(project)

    # ── Sub-typed projects (VBP / NORMNET): one tab per top-level folder ─
    def _activate_subtyped(self, project: str):
        cfg = _SUBTYPED[project]
        root_path = get_project_path(project)

        # Migrate legacy VBP config that saved a specific *_TP folder.
        if (project == "VBP" and root_path
                and root_path.rstrip("/").endswith(_VBP_SUBTYPE_SUFFIX)):
            parent = os.path.dirname(root_path.rstrip("/"))
            if os.path.isdir(parent):
                root_path = parent
                save_project_path("VBP", root_path)

        if not root_path or not os.path.isdir(root_path):
            root_path = self._ask_for_path(project)
            if not root_path:
                self.sidebar.set_active_silent(project)
                return
            save_project_path(project, root_path)

        subtypes = self._discover_subtypes(project, root_path)
        if not subtypes:
            self._hide_subtype_bar()
            what = f"*{cfg['suffix']} folders" if cfg["suffix"] else "sub-folders"
            QMessageBox.warning(
                self, f"No {project} experiments",
                f"No {what} under:\n{root_path}"
            )
            return

        self.subtype_bar.blockSignals(True)
        while self.subtype_bar.count() > 0:
            self.subtype_bar.removeTab(0)
        for s in subtypes:
            self.subtype_bar.addTab(s)
        self.subtype_bar.setVisible(True)

        cur = self._active_subtype.get(project)
        idx = subtypes.index(cur) if cur in subtypes else 0
        self.subtype_bar.setCurrentIndex(idx)
        self.subtype_bar.blockSignals(False)

        self._active_subtype[project] = subtypes[idx]
        self._load_subtype(project, root_path, subtypes[idx])

    def _on_subtype_changed(self, idx: int):
        project = self._active_project
        if idx < 0 or project not in _SUBTYPED:
            return
        subtype = self.subtype_bar.tabText(idx)
        root_path = get_project_path(project)
        if not root_path:
            return
        self._active_subtype[project] = subtype
        self._load_subtype(project, root_path, subtype)

    def _load_subtype(self, project: str, root_path: str, subtype: str):
        cfg = _SUBTYPED[project]
        key = f"{project}/{subtype}"
        full_path = os.path.join(root_path, subtype)
        if self._loaded.get(key) is None:
            self.status.showMessage(f"Scanning {key}…")
            try:
                self._loaded[key] = cfg["scanner"](full_path)
            except Exception as e:
                QMessageBox.warning(self, "Scan error", f"Could not scan {key}:\n{e}")
                self._loaded[key] = []

        data = self._loaded.get(key) or []
        self.runs_screen.load(project, data)
        self.plots_screen.load(project, data)
        self.monitor_screen.load(project, data)
        if cfg["extra_tabs"]:
            self.launcher_screen.load(subtype, root_path)
        self.tabs.setTabVisible(self._launcher_tab_idx, cfg["extra_tabs"])
        self.tabs.setTabVisible(self._wizard_tab_idx, cfg["extra_tabs"])
        n = len(data)
        self.status.showMessage(
            f"{key}  —  {n} experiment{'s' if n != 1 else ''} loaded  ({full_path})"
        )

    def _discover_subtypes(self, project: str, root_path: str) -> list[str]:
        suffix = _SUBTYPED[project]["suffix"]
        try:
            items = os.listdir(root_path)
        except OSError:
            return []
        return sorted(
            d for d in items
            if os.path.isdir(os.path.join(root_path, d))
            and not d.startswith(".")
            and (suffix is None or d.endswith(suffix))
        )

    def _invalidate_subtype_cache(self, project: str):
        prefix = f"{project}/"
        for k in list(self._loaded.keys()):
            if k.startswith(prefix):
                self._loaded[k] = None

    def _hide_subtype_bar(self):
        self.subtype_bar.setVisible(False)
        self.tabs.setTabVisible(self._launcher_tab_idx, False)
        self.tabs.setTabVisible(self._wizard_tab_idx, False)

    # ── Helpers ──────────────────────────────────────────────────────
    def _scan(self, project: str, root_path: str):
        self.status.showMessage(f"Scanning {project}…")
        try:
            self._loaded[project] = _SCANNERS[project](root_path)
        except Exception as e:
            QMessageBox.warning(self, "Scan error", f"Could not scan {project}:\n{e}")
            self._loaded[project] = []

        # If nothing found, offer to pick a different folder
        if self._loaded[project] == []:
            reply = QMessageBox.question(
                self, "No experiments found",
                f"No experiments found in:\n{root_path}\n\nPick a different folder?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                new_path = self._ask_for_path(project)
                if new_path:
                    save_project_path(project, new_path)
                    self._scan(project, new_path)

    def _display(self, project: str):
        data = self._loaded.get(project) or []
        root_path = get_project_path(project) or ""
        self.runs_screen.load(project, data)
        self.plots_screen.load(project, data)
        self.monitor_screen.load(project, data)
        n = len(data)
        self.status.showMessage(
            f"{project}  —  {n} experiment{'s' if n != 1 else ''} loaded  ({root_path})"
        )

    def _ask_for_path(self, project: str) -> str:
        # Start in default path if it exists, else home
        default = DEFAULT_PATHS.get(project, "")
        start_dir = default if os.path.isdir(default) else os.path.expanduser("~")
        return QFileDialog.getExistingDirectory(
            self,
            f"Select root experiments folder for {project}",
            start_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
