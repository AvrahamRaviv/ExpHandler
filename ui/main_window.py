"""Main application window."""

import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QTabWidget,
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ExpHandler")
        self.resize(1400, 820)

        self._loaded: dict[str, list | None] = {"DVNR": None, "ODT": None, "VBP": None}
        self._active_project: str | None = None

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
        sep.setStyleSheet("background-color: #d0d0d0;")
        h_layout.addWidget(sep)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.runs_screen = RunsScreen()
        self.plots_screen = PlotsScreen()
        self.monitor_screen = MonitorScreen()
        self.tabs.addTab(self.runs_screen, "Runs")
        self.tabs.addTab(self.plots_screen, "Plots")
        self.tabs.addTab(self.monitor_screen, "Monitor")
        h_layout.addWidget(self.tabs, stretch=1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Select a project from the sidebar.")

    # ── Project selection ────────────────────────────────────────────
    def _on_project_selected(self, project: str):
        self._active_project = project

        root_path = get_project_path(project)
        if not root_path or not os.path.isdir(root_path):
            root_path = self._ask_for_path(project)
            if not root_path:
                self.sidebar.set_active_silent(project)
                return
            save_project_path(project, root_path)
            self._loaded[project] = None  # force re-scan for new path

        if self._loaded[project] is None:
            self._scan(project, root_path)

        self._display(project)

    # ── Change path button (📁) ───────────────────────────────────────
    def _on_change_path(self, project: str):
        root_path = self._ask_for_path(project)
        if not root_path:
            return
        save_project_path(project, root_path)
        self._loaded[project] = None  # invalidate cache
        self._active_project = project
        self.sidebar.set_active_silent(project)
        self._scan(project, root_path)
        self._display(project)

    # ── Refresh button ───────────────────────────────────────────────
    def _on_refresh(self):
        project = self._active_project
        if not project:
            self.status.showMessage("Select a project first.")
            return
        root_path = get_project_path(project)
        if not root_path or not os.path.isdir(root_path):
            return
        self._loaded[project] = None
        self._scan(project, root_path)
        self._display(project)

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
        data = self._loaded[project] or []
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
