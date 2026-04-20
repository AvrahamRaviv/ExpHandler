"""Main application window."""

import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget, QTabBar,
    QFileDialog, QStatusBar, QMessageBox,
)

from config import get_project_path, save_project_path, DEFAULT_PATHS
from scanners.dvnr import scan_dvnr
from scanners.odt import scan_odt
from scanners.vbp import scan_vbp
from screens.runs import RunsScreen
from screens.plots import PlotsScreen
from screens.monitor import MonitorScreen
from ui.sidebar import Sidebar

_SCANNERS = {"DVNR": scan_dvnr, "ODT": scan_odt, "VBP": scan_vbp}
_VBP_SUBTYPE_SUFFIX = "_TP"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ExpHandler")
        self.resize(1400, 820)

        # Cache keyed by project name or "VBP/<subtype>"
        self._loaded: dict[str, list | None] = {"DVNR": None, "ODT": None}
        self._active_project: str | None = None
        self._active_vbp_subtype: str | None = None

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
        h_layout.addWidget(self.sidebar)

        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setStyleSheet("background-color: #2d3748;")
        h_layout.addWidget(sep)

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
        self.tabs.addTab(self.runs_screen, "Runs")
        self.tabs.addTab(self.plots_screen, "Plots")
        self.tabs.addTab(self.monitor_screen, "Monitor")
        content_layout.addWidget(self.tabs, stretch=1)

        h_layout.addWidget(content, stretch=1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Select a project from the sidebar.")

    # ── Project selection ────────────────────────────────────────────
    def _on_project_selected(self, project: str):
        self._active_project = project
        if project == "VBP":
            self._activate_vbp()
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
        if project == "VBP":
            self._invalidate_vbp_cache()
            self._activate_vbp()
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
        if project == "VBP":
            self._invalidate_vbp_cache()
            self._activate_vbp()
            return
        root_path = get_project_path(project)
        if not root_path or not os.path.isdir(root_path):
            return
        self._loaded[project] = None
        self._scan(project, root_path)
        self._display(project)

    # ── VBP multi-subtype handling ───────────────────────────────────
    def _activate_vbp(self):
        root_path = get_project_path("VBP")
        # Migrate legacy config that saved a specific *_TP folder
        if root_path and root_path.rstrip("/").endswith(_VBP_SUBTYPE_SUFFIX):
            parent = os.path.dirname(root_path.rstrip("/"))
            if os.path.isdir(parent):
                root_path = parent
                save_project_path("VBP", root_path)

        if not root_path or not os.path.isdir(root_path):
            root_path = self._ask_for_path("VBP")
            if not root_path:
                self.sidebar.set_active_silent("VBP")
                return
            save_project_path("VBP", root_path)

        subtypes = self._discover_vbp_subtypes(root_path)
        if not subtypes:
            self._hide_subtype_bar()
            QMessageBox.warning(
                self, "No VBP experiments",
                f"No *{_VBP_SUBTYPE_SUFFIX} folders under:\n{root_path}"
            )
            return

        self.subtype_bar.blockSignals(True)
        while self.subtype_bar.count() > 0:
            self.subtype_bar.removeTab(0)
        for s in subtypes:
            self.subtype_bar.addTab(s)
        self.subtype_bar.setVisible(True)

        idx = subtypes.index(self._active_vbp_subtype) \
            if self._active_vbp_subtype in subtypes else 0
        self.subtype_bar.setCurrentIndex(idx)
        self.subtype_bar.blockSignals(False)

        self._active_vbp_subtype = subtypes[idx]
        self._load_vbp_subtype(root_path, subtypes[idx])

    def _on_subtype_changed(self, idx: int):
        if idx < 0 or self._active_project != "VBP":
            return
        subtype = self.subtype_bar.tabText(idx)
        root_path = get_project_path("VBP")
        if not root_path:
            return
        self._active_vbp_subtype = subtype
        self._load_vbp_subtype(root_path, subtype)

    def _load_vbp_subtype(self, root_path: str, subtype: str):
        key = f"VBP/{subtype}"
        full_path = os.path.join(root_path, subtype)
        if self._loaded.get(key) is None:
            self.status.showMessage(f"Scanning {key}…")
            try:
                self._loaded[key] = scan_vbp(full_path)
            except Exception as e:
                QMessageBox.warning(self, "Scan error", f"Could not scan {key}:\n{e}")
                self._loaded[key] = []

        data = self._loaded.get(key) or []
        self.runs_screen.load("VBP", data)
        self.plots_screen.load("VBP", data)
        self.monitor_screen.load("VBP", data)
        n = len(data)
        self.status.showMessage(
            f"{key}  —  {n} experiment{'s' if n != 1 else ''} loaded  ({full_path})"
        )

    def _discover_vbp_subtypes(self, root_path: str) -> list[str]:
        try:
            items = os.listdir(root_path)
        except OSError:
            return []
        return sorted(
            d for d in items
            if d.endswith(_VBP_SUBTYPE_SUFFIX)
            and os.path.isdir(os.path.join(root_path, d))
        )

    def _invalidate_vbp_cache(self):
        for k in list(self._loaded.keys()):
            if k.startswith("VBP/"):
                self._loaded[k] = None

    def _hide_subtype_bar(self):
        self.subtype_bar.setVisible(False)

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
