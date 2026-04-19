"""Left sidebar: project selector buttons with change-folder icons."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QToolButton,
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont

from config import PROJECTS


class Sidebar(QWidget):
    project_selected = pyqtSignal(str)
    change_path_requested = pyqtSignal(str)
    refresh_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(170)
        self._buttons: dict[str, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(4)

        # Title
        title = QLabel("ExpHandler")
        title.setFont(QFont("sans-serif", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        version = QLabel("v0.2.0")
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(version)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #ccc;")
        layout.addWidget(line)
        layout.addSpacing(8)

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
            folder_btn.setStyleSheet(
                "QToolButton { border: none; background: transparent; font-size: 14px; }"
                "QToolButton:hover { background: #e0e0e0; border-radius: 4px; }"
            )
            folder_btn.clicked.connect(lambda checked, p=proj: self.change_path_requested.emit(p))
            row.addWidget(folder_btn)

            layout.addLayout(row)

        layout.addStretch()

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setFixedHeight(30)
        refresh_btn.setToolTip("Re-scan the active project folder")
        refresh_btn.setStyleSheet(
            "QPushButton { background-color: transparent; color: #333; "
            "border: 1px solid #ccc; border-radius: 4px; }"
            "QPushButton:hover { background-color: #e0e0e0; }"
        )
        refresh_btn.clicked.connect(self.refresh_requested.emit)
        layout.addWidget(refresh_btn)

        self.setStyleSheet("background-color: #f5f5f5;")

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

    @staticmethod
    def _btn_style(active: bool) -> str:
        if active:
            return ("QPushButton { background-color: #0d6efd; color: white; "
                    "border-radius: 4px; font-weight: bold; }")
        return ("QPushButton { background-color: transparent; color: #333; "
                "border-radius: 4px; text-align: left; padding-left: 8px; }"
                "QPushButton:hover { background-color: #e0e0e0; }")
