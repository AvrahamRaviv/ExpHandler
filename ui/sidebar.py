"""Left sidebar: project selector buttons."""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFrame
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont

from config import PROJECTS


class Sidebar(QWidget):
    project_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(150)
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

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #ccc;")
        layout.addWidget(line)
        layout.addSpacing(8)

        # Project buttons
        for proj in PROJECTS:
            btn = QPushButton(proj)
            btn.setCheckable(True)
            btn.setFixedHeight(36)
            btn.setStyleSheet(self._btn_style(False))
            btn.clicked.connect(lambda checked, p=proj: self._on_click(p))
            layout.addWidget(btn)
            self._buttons[proj] = btn

        layout.addStretch()
        self.setStyleSheet("background-color: #f5f5f5;")

    def _on_click(self, project: str):
        for p, btn in self._buttons.items():
            btn.setChecked(p == project)
            btn.setStyleSheet(self._btn_style(p == project))
        self.project_selected.emit(project)

    def set_active(self, project: str):
        self._on_click(project)

    @staticmethod
    def _btn_style(active: bool) -> str:
        if active:
            return ("QPushButton { background-color: #0d6efd; color: white; "
                    "border-radius: 4px; font-weight: bold; }")
        return ("QPushButton { background-color: transparent; color: #333; "
                "border-radius: 4px; text-align: left; padding-left: 8px; }"
                "QPushButton:hover { background-color: #e0e0e0; }")
