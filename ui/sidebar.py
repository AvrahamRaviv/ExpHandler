"""Left sidebar: project selector buttons with change-folder icons."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QToolButton,
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont

from config import PROJECTS
from ui.theme import BG_DEEP, BG_ELEVATED, BG_HOVER, BORDER, ACCENT, ACCENT_DIM, TEXT_PRIMARY, TEXT_SECONDARY


class Sidebar(QWidget):
    project_selected = pyqtSignal(str)
    change_path_requested = pyqtSignal(str)
    refresh_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(175)
        self._buttons: dict[str, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 16, 10, 14)
        layout.setSpacing(4)

        # Title
        title = QLabel("ExpHandler")
        title.setFont(QFont("SF Pro Display", 13, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; letter-spacing: 0.5px;")
        layout.addWidget(title)

        version = QLabel("v0.2.0")
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; letter-spacing: 1px;")
        layout.addWidget(version)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {BORDER}; margin-top: 6px; margin-bottom: 2px;")
        layout.addWidget(line)
        layout.addSpacing(6)

        # Project rows: [ProjectButton] [📁]
        for proj in PROJECTS:
            row = QHBoxLayout()
            row.setSpacing(2)

            btn = QPushButton(proj)
            btn.setCheckable(True)
            btn.setFixedHeight(34)
            btn.setStyleSheet(self._btn_style(False))
            btn.clicked.connect(lambda checked, p=proj: self._on_click(p))
            row.addWidget(btn, stretch=1)
            self._buttons[proj] = btn

            folder_btn = QToolButton()
            folder_btn.setText("📁")
            folder_btn.setFixedSize(28, 34)
            folder_btn.setToolTip(f"Change {proj} folder")
            folder_btn.clicked.connect(lambda checked, p=proj: self.change_path_requested.emit(p))
            row.addWidget(folder_btn)

            layout.addLayout(row)

        layout.addStretch()

        refresh_btn = QPushButton("⟳  Refresh")
        refresh_btn.setFixedHeight(32)
        refresh_btn.setToolTip("Re-scan the active project folder")
        refresh_btn.clicked.connect(self.refresh_requested.emit)
        layout.addWidget(refresh_btn)

        self.setStyleSheet(f"background-color: {BG_DEEP};")

    def _on_click(self, project: str):
        self._set_styles(project)
        self.project_selected.emit(project)

    def set_active_silent(self, project: str):
        """Update button styles without emitting any signal."""
        self._set_styles(project)

    def _set_styles(self, active_project: str):
        for p, btn in self._buttons.items():
            btn.setChecked(p == active_project)
            btn.setStyleSheet(self._btn_style(p == active_project))

    def _btn_style(self, active: bool) -> str:
        if active:
            return (f"QPushButton {{ background-color: {ACCENT_DIM}; color: {ACCENT}; "
                    f"border: 1px solid {ACCENT}; border-radius: 6px; font-weight: 600; "
                    f"text-align: left; padding-left: 10px; }}")
        return (f"QPushButton {{ background-color: transparent; color: {TEXT_SECONDARY}; "
                f"border: 1px solid transparent; border-radius: 6px; "
                f"text-align: left; padding-left: 10px; }}"
                f"QPushButton:hover {{ background-color: {BG_ELEVATED}; color: {TEXT_PRIMARY}; "
                f"border-color: {BORDER}; }}")
