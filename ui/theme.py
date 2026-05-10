"""Theming: palette-driven QSS with multiple themes.

Add a new theme by appending to PALETTES below. Status colors are kept
theme-independent so monitor coloring stays consistent.
"""

# ── Status colors (theme-independent) ────────────────────────────────────────
C_SUCCESS = "#10b981"   # green  — DONE
C_RUN     = "#f59e0b"   # amber  — RUN
C_ERROR   = "#f87171"   # red    — EXIT
C_PENDING = "#6b7280"   # gray   — PEND/WAIT
C_UNKNOWN = "#374151"   # muted  — unknown


# ── Palettes ─────────────────────────────────────────────────────────────────
PALETTES: dict[str, dict] = {
    "dark": {
        "BG_DEEP":     "#0a0e1a",
        "BG_SURFACE":  "#111827",
        "BG_ELEVATED": "#1f2937",
        "BG_HOVER":    "#374151",
        "BG_ALT_ROW":  "#131c2e",
        "BORDER":      "#2d3748",
        "BORDER_LIT":  "#4b5563",
        "TEXT_PRIMARY":   "#f0f4f8",
        "TEXT_SECONDARY": "#8b9ab0",
        "TEXT_DIM":       "#4b5563",
        "ACCENT":      "#06b6d4",
        "ACCENT_DARK": "#0891b2",
        "ACCENT_DIM":  "#164e63",
        "TEXTEDIT_FG": "#a7f3d0",
        "DIFF_HIGHLIGHT": "#3b2a1c",
    },
    "light": {
        "BG_DEEP":     "#eef1f5",
        "BG_SURFACE":  "#ffffff",
        "BG_ELEVATED": "#f3f5f8",
        "BG_HOVER":    "#e2e8f0",
        "BG_ALT_ROW":  "#f7f9fc",
        "BORDER":      "#d1d5db",
        "BORDER_LIT":  "#9ca3af",
        "TEXT_PRIMARY":   "#0f172a",
        "TEXT_SECONDARY": "#475569",
        "TEXT_DIM":       "#94a3b8",
        "ACCENT":      "#0e7490",
        "ACCENT_DARK": "#155e75",
        "ACCENT_DIM":  "#cffafe",
        "TEXTEDIT_FG": "#065f46",
        "DIFF_HIGHLIGHT": "#fff4d6",
    },
}

DEFAULT_THEME = "dark"

# Module-level constants (rebound on theme switch). Initialized below.
BG_DEEP = BG_SURFACE = BG_ELEVATED = BG_HOVER = BG_ALT_ROW = ""
BORDER = BORDER_LIT = ""
TEXT_PRIMARY = TEXT_SECONDARY = TEXT_DIM = ""
ACCENT = ACCENT_DARK = ACCENT_DIM = ""
TEXTEDIT_FG = ""
DIFF_HIGHLIGHT = ""

current_theme: str = DEFAULT_THEME
QSS: str = ""


def _build_qss() -> str:
    return f"""
/* ── Base ─────────────────────────────────────────────── */
QWidget {{
    background-color: {BG_SURFACE};
    color: {TEXT_PRIMARY};
    font-family: "SF Pro Text", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 13px;
}}

QMainWindow, QDialog {{
    background-color: {BG_DEEP};
}}

/* ── Tabs ─────────────────────────────────────────────── */
QTabWidget::pane {{
    border: none;
    border-top: 1px solid {BORDER};
    background: {BG_SURFACE};
}}

QTabBar {{
    background: {BG_DEEP};
}}

QTabBar::tab {{
    background: transparent;
    color: {TEXT_SECONDARY};
    padding: 9px 22px;
    border: none;
    border-bottom: 2px solid transparent;
    font-size: 13px;
    font-weight: 500;
    min-width: 70px;
}}

QTabBar::tab:selected {{
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
    background: transparent;
}}

QTabBar::tab:hover:!selected {{
    color: {TEXT_PRIMARY};
}}

/* ── Tables ───────────────────────────────────────────── */
QTableWidget {{
    background-color: {BG_SURFACE};
    alternate-background-color: {BG_ALT_ROW};
    gridline-color: {BORDER};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    selection-background-color: {ACCENT_DIM};
    selection-color: {TEXT_PRIMARY};
    font-family: "JetBrains Mono", "Cascadia Code", "Fira Code", "Menlo", monospace;
    font-size: 12px;
    outline: none;
}}

QTableWidget::item {{
    padding: 3px 6px;
    border: none;
}}

QTableWidget::item:selected {{
    background: {ACCENT_DIM};
}}

QHeaderView {{
    background: {BG_ELEVATED};
}}

QHeaderView::section {{
    background: {BG_ELEVATED};
    color: {TEXT_SECONDARY};
    font-weight: 600;
    font-size: 11px;
    padding: 7px 8px;
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 2px solid {BORDER};
    text-transform: uppercase;
    letter-spacing: 0.4px;
}}

QHeaderView::section:last {{
    border-right: none;
}}

/* ── Scrollbars ───────────────────────────────────────── */
QScrollBar:vertical {{
    background: {BG_ELEVATED};
    width: 7px;
    border-radius: 3px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_LIT};
    border-radius: 3px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_SECONDARY};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

QScrollBar:horizontal {{
    background: {BG_ELEVATED};
    height: 7px;
    border-radius: 3px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_LIT};
    border-radius: 3px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {TEXT_SECONDARY};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none; }}

/* ── Inputs ───────────────────────────────────────────── */
QLineEdit {{
    background: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 10px;
    color: {TEXT_PRIMARY};
    font-size: 13px;
    selection-background-color: {ACCENT_DIM};
}}
QLineEdit:focus {{
    border-color: {ACCENT};
    background: {BG_SURFACE};
}}
QLineEdit:hover {{
    border-color: {BORDER_LIT};
}}

/* ── Buttons ──────────────────────────────────────────── */
QPushButton {{
    background: {BG_ELEVATED};
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 14px;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton:hover {{
    background: {BG_HOVER};
    color: {TEXT_PRIMARY};
    border-color: {BORDER_LIT};
}}
QPushButton:pressed {{
    background: {BG_SURFACE};
}}
QPushButton:disabled {{
    color: {TEXT_DIM};
    border-color: {BORDER};
    background: {BG_SURFACE};
}}

QToolButton {{
    background: transparent;
    border: none;
    border-radius: 4px;
    color: {TEXT_SECONDARY};
    font-size: 14px;
    padding: 2px 4px;
}}
QToolButton:hover {{
    background: {BG_HOVER};
    color: {TEXT_PRIMARY};
}}

/* ── GroupBox ─────────────────────────────────────────── */
QGroupBox {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 10px;
    font-size: 12px;
    font-weight: 600;
    color: {TEXT_SECONDARY};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    background: {BG_SURFACE};
    color: {TEXT_SECONDARY};
}}

/* ── TextEdit ─────────────────────────────────────────── */
QTextEdit, QPlainTextEdit {{
    background: {BG_DEEP};
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXTEDIT_FG};
    font-family: "JetBrains Mono", "Cascadia Code", "Fira Code", "Menlo", monospace;
    font-size: 12px;
    padding: 8px;
    selection-background-color: {ACCENT_DIM};
}}

/* ── Labels ───────────────────────────────────────────── */
QLabel {{
    color: {TEXT_SECONDARY};
    background: transparent;
}}

/* ── Status bar ───────────────────────────────────────── */
QStatusBar {{
    background: {BG_DEEP};
    color: {TEXT_DIM};
    font-size: 12px;
    border-top: 1px solid {BORDER};
    padding: 0 8px;
}}

/* ── Splitter ─────────────────────────────────────────── */
QSplitter::handle {{
    background: {BORDER};
}}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical   {{ height: 1px; }}
QSplitter::handle:hover {{
    background: {ACCENT};
}}

/* ── Dialogs ──────────────────────────────────────────── */
QMessageBox {{
    background: {BG_ELEVATED};
}}
QMessageBox QLabel {{
    color: {TEXT_PRIMARY};
}}
QMessageBox QPushButton {{
    min-width: 80px;
}}
QFileDialog {{
    background: {BG_ELEVATED};
}}

/* ── Lists ────────────────────────────────────────────── */
QListWidget {{
    background: {BG_SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    selection-background-color: {ACCENT_DIM};
    selection-color: {TEXT_PRIMARY};
}}
QListWidget::item:hover {{
    background: {BG_HOVER};
}}

/* ── ComboBox ─────────────────────────────────────────── */
QComboBox {{
    background: {BG_ELEVATED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
}}
QComboBox QAbstractItemView {{
    background: {BG_ELEVATED};
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT_DIM};
    border: 1px solid {BORDER};
}}
"""


def set_theme(name: str):
    """Switch active theme: rebinds module constants and rebuilds QSS."""
    global current_theme, QSS
    global BG_DEEP, BG_SURFACE, BG_ELEVATED, BG_HOVER, BG_ALT_ROW
    global BORDER, BORDER_LIT
    global TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM
    global ACCENT, ACCENT_DARK, ACCENT_DIM
    global TEXTEDIT_FG, DIFF_HIGHLIGHT
    if name not in PALETTES:
        name = DEFAULT_THEME
    current_theme = name
    p = PALETTES[name]
    BG_DEEP, BG_SURFACE = p["BG_DEEP"], p["BG_SURFACE"]
    BG_ELEVATED, BG_HOVER, BG_ALT_ROW = p["BG_ELEVATED"], p["BG_HOVER"], p["BG_ALT_ROW"]
    BORDER, BORDER_LIT = p["BORDER"], p["BORDER_LIT"]
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM = (
        p["TEXT_PRIMARY"], p["TEXT_SECONDARY"], p["TEXT_DIM"]
    )
    ACCENT, ACCENT_DARK, ACCENT_DIM = p["ACCENT"], p["ACCENT_DARK"], p["ACCENT_DIM"]
    TEXTEDIT_FG = p["TEXTEDIT_FG"]
    DIFF_HIGHLIGHT = p["DIFF_HIGHLIGHT"]
    QSS = _build_qss()


def get_themes() -> list[str]:
    return list(PALETTES.keys())


# Initial bind
set_theme(DEFAULT_THEME)
