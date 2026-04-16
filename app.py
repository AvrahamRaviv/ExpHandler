"""ExpHandler — Experiment management GUI (PyQt5).

Run:
    python app.py
"""

import sys
import os

# Ensure project root is on path
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ExpHandler")
    app.setStyle("Fusion")  # consistent look on Linux and Mac

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
