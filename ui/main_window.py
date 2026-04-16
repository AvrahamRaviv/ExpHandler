"""Main application window."""

import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QTabWidget,
    QFileDialog, QStatusBar, QMessageBox,
)
from PyQt5.QtCore import Qt

from config import get_project_path, save_project_path
from scanners.dvnr import scan_dvnr
from scanners.odt import scan_odt
from scanners.vbp import scan_vbp
from screens.runs import RunsScreen
from screens.plots import PlotsScreen
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

        # Sidebar
        self.sidebar = Sidebar()
        self.sidebar.project_selected.connect(self._on_project_selected)
        h_layout.addWidget(self.sidebar)

        # Thin separator line
        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setStyleSheet("background-color: #d0d0d0;")
        h_layout.addWidget(sep)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.runs_screen = RunsScreen()
        self.plots_screen = PlotsScreen()
        self.tabs.addTab(self.runs_screen, "Runs")
        self.tabs.addTab(self.plots_screen, "Plots")
        h_layout.addWidget(self.tabs, stretch=1)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Select a project from the sidebar.")

    # ── Project selection ────────────────────────────────────────────
    def _on_project_selected(self, project: str):
        self._active_project = project

        # Ensure we have a root path
        root_path = get_project_path(project)
        if not root_path or not os.path.isdir(root_path):
            root_path = self._ask_for_path(project)
            if not root_path:
                # User cancelled — deselect
                self.sidebar.set_active(self._active_project or "")
                return
            save_project_path(project, root_path)

        # Scan if not yet loaded
        if self._loaded[project] is None:
            self.status.showMessage(f"Scanning {project} experiments…")
            try:
                self._loaded[project] = _SCANNERS[project](root_path)
            except Exception as e:
                QMessageBox.warning(self, "Scan error",
                                    f"Could not scan {project}:\n{e}")
                self._loaded[project] = []

        data = self._loaded[project]
        n = len(data)
        self.runs_screen.load(project, data)
        self.plots_screen.load(project, data)
        self.status.showMessage(
            f"{project}  —  {n} experiment{'s' if n != 1 else ''} loaded  "
            f"({root_path})"
        )

    def _ask_for_path(self, project: str) -> str:
        path = QFileDialog.getExistingDirectory(
            self,
            f"Select root experiments folder for {project}",
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        return path  # empty string if user cancelled
