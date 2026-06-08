"""Left sidebar: project selector buttons with change-folder icons."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QToolButton,
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont

from config import PROJECTS
from ui import theme


class Sidebar(QWidget):
    project_selected = pyqtSignal(str)
    change_path_requested = pyqtSignal(str)
    refresh_requested = pyqtSignal()
    theme_toggle_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(175)
        self._buttons: dict[str, QPushButton] = {}
        self._active: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 16, 10, 14)
        layout.setSpacing(4)

        # Title
        self.title = QLabel("ExpHandler")
        self.title.setFont(QFont("SF Pro Display", 13, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title)

        self.version = QLabel("v0.2.14")
        self.version.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.version)

        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        layout.addWidget(self.line)
        layout.addSpacing(6)

        # Project rows: [ProjectButton] [📁]
        for proj in PROJECTS:
            row = QHBoxLayout()
            row.setSpacing(2)

            btn = QPushButton(proj)
            btn.setCheckable(True)
            btn.setFixedHeight(34)
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

        self.theme_btn = QPushButton()
        self.theme_btn.setFixedHeight(28)
        self.theme_btn.setToolTip("Toggle theme")
        self.theme_btn.clicked.connect(self.theme_toggle_requested.emit)
        layout.addWidget(self.theme_btn)

        refresh_btn = QPushButton("⟳  Refresh")
        refresh_btn.setFixedHeight(32)
        refresh_btn.setToolTip("Re-scan the active project folder")
        refresh_btn.clicked.connect(self.refresh_requested.emit)
        layout.addWidget(refresh_btn)

        self.refresh_styles()

    # ── Public ────────────────────────────────────────────────────────
    def refresh_styles(self):
        """Re-apply theme-dependent styles. Call after theme switch."""
        self.setStyleSheet(f"background-color: {theme.BG_DEEP};")
        self.title.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; letter-spacing: 0.5px;"
        )
        self.version.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-size: 10px; letter-spacing: 1px;"
        )
        self.line.setStyleSheet(
            f"color: {theme.BORDER}; margin-top: 6px; margin-bottom: 2px;"
        )
        self._set_styles(self._active)
        self.theme_btn.setText(
            "☀  Light" if theme.current_theme == "dark" else "🌙  Dark"
        )

    def set_active_silent(self, project: str):
        """Update button styles without emitting any signal."""
        self._active = project
        self._set_styles(project)

    # ── Internal ──────────────────────────────────────────────────────
    def _on_click(self, project: str):
        self._active = project
        self._set_styles(project)
        self.project_selected.emit(project)

    def _set_styles(self, active_project: str | None):
        for p, btn in self._buttons.items():
            is_active = (p == active_project)
            btn.setChecked(is_active)
            btn.setStyleSheet(self._btn_style(is_active))

    def _btn_style(self, active: bool) -> str:
        if active:
            return (f"QPushButton {{ background-color: {theme.ACCENT_DIM}; color: {theme.ACCENT}; "
                    f"border: 1px solid {theme.ACCENT}; border-radius: 6px; font-weight: 600; "
                    f"text-align: left; padding-left: 10px; }}")
        return (f"QPushButton {{ background-color: transparent; color: {theme.TEXT_SECONDARY}; "
                f"border: 1px solid transparent; border-radius: 6px; "
                f"text-align: left; padding-left: 10px; }}"
                f"QPushButton:hover {{ background-color: {theme.BG_ELEVATED}; color: {theme.TEXT_PRIMARY}; "
                f"border-color: {theme.BORDER}; }}")
