"""Monitor screen: experiment-folder-driven LSF status + command launcher."""

import os
import shlex
import subprocess
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLabel, QAbstractItemView, QLineEdit,
    QSplitter, QGroupBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QFont

from config import LSF_LOG_DIR

# ── Colours ───────────────────────────────────────────────────────────────────

_C_COMPLETE    = QColor("#28a745")
_C_IN_PROGRESS = QColor("#fd7e14")
_C_CRASH       = QColor("#dc3545")
_C_PENDING     = QColor("#888888")
_C_UNKNOWN     = QColor("#aaaaaa")

_LSF_STAT_MAP = {
    "RUN":   ("In Progress", _C_IN_PROGRESS),
    "DONE":  ("Complete",    _C_COMPLETE),
    "EXIT":  ("Crash",       _C_CRASH),
    "PEND":  ("Pending",     _C_PENDING),
    "WAIT":  ("Pending",     _C_PENDING),
    "SSUSP": ("Suspended",   _C_PENDING),
    "USUSP": ("Suspended",   _C_PENDING),
    "PSUSP": ("Suspended",   _C_PENDING),
}
_UNKNOWN_STATUS = ("Unknown", _C_UNKNOWN)

# ── LSF helpers ───────────────────────────────────────────────────────────────

def _shell(cmd: str) -> str:
    try:
        r = subprocess.run(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=15,
        )
        return r.stdout
    except Exception:
        return ""


def _bjobs_all() -> list[dict]:
    """Parse `bjobs -a` → list of {job_id, stat, queue, exec_host}."""
    out = _shell("bjobs -a 2>/dev/null")
    jobs = []
    for line in out.splitlines()[1:]:
        cols = line.split()
        if len(cols) >= 3:
            jobs.append({
                "job_id":    cols[0],
                "stat":      cols[2],
                "queue":     cols[3] if len(cols) > 3 else "",
                "exec_host": cols[5] if len(cols) > 5 else "",
            })
    return jobs


def _find_log(job_id: str) -> str | None:
    """Search LSF_LOG_DIR/<DD-MM-YY>/<job_id>.out going backward in time."""
    today = datetime.today()
    year, month = today.year, today.month
    for y in (year, year - 1):
        m_range = range(month, 0, -1) if y == year else range(12, 0, -1)
        for m in m_range:
            for d in range(31, 0, -1):
                path = os.path.join(
                    LSF_LOG_DIR,
                    f"{d:02d}-{m:02d}-{str(y)[2:]}",
                    f"{job_id}.out",
                )
                if os.path.isfile(path):
                    return path
    return None


def _read_log(job_id: str) -> str:
    path = _find_log(job_id)
    if path is None:
        return ""
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _extract_name(log: str) -> str:
    """Extract the -D value from the command line embedded in the log."""
    idx = log.find("-D ")
    if idx == -1:
        return ""
    rest = log[idx + 3:].lstrip()
    if rest.startswith(("'", '"')):
        q = rest[0]
        end = rest.find(q, 1)
        return rest[1:end] if end != -1 else rest[1:]
    for ch in (" ", "\t", "\n"):
        pos = rest.find(ch)
        if pos != -1:
            rest = rest[:pos]
    return rest


def _extract_command(log: str) -> str:
    """Return the full launch command line from the log."""
    for prefix in ("/algo/ws", "bsub ", "drun "):
        idx = log.find(prefix)
        if idx != -1:
            end = log.find("\n", idx)
            return (log[idx:end] if end != -1 else log[idx:]).strip()
    return ""


def _build_lsf_index() -> dict[str, dict]:
    """Return {exp_name: {job_id, stat, label, color, command, queue, exec_host}}."""
    index: dict[str, dict] = {}
    for job in _bjobs_all():
        job_id = job["job_id"]
        log = _read_log(job_id)
        name = _extract_name(log)
        if not name:
            continue
        command = _extract_command(log)
        label, color = _LSF_STAT_MAP.get(job["stat"].upper(), _UNKNOWN_STATUS)
        index[name] = {
            "job_id":    job_id,
            "stat":      job["stat"],
            "label":     label,
            "color":     color,
            "command":   command,
            "queue":     job["queue"],
            "exec_host": job["exec_host"],
        }
    return index

# ── Command parser ────────────────────────────────────────────────────────────

def _parse_command(cmd: str) -> list[tuple[str, str]]:
    """Tokenise a shell command into (flag, value) pairs.

    Positional (non-flag) tokens are grouped into a single row with flag="".
    Quoted strings are handled via shlex.split.
    """
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()

    rows: list[tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-"):
            flag = tok
            i += 1
            value = ""
            if i < len(tokens) and not tokens[i].startswith("-"):
                value = tokens[i]
                i += 1
            rows.append((flag, value))
        else:
            pos = [tok]
            i += 1
            while i < len(tokens) and not tokens[i].startswith("-"):
                pos.append(tokens[i])
                i += 1
            rows.append(("", " ".join(pos)))
    return rows


def _reconstruct_command(rows: list[tuple[str, str]]) -> str:
    """Rebuild a shell command string from (flag, value) pairs."""
    parts = []
    for flag, value in rows:
        if flag:
            if value:
                # quote if the value contains shell-special characters
                safe = shlex.quote(value) if any(c in value for c in " \t[](){}\"'") else value
                parts.append(f"{flag} {safe}")
            else:
                parts.append(flag)
        elif value:
            parts.append(value)
    return " ".join(parts)

# ── Experiment name helpers ───────────────────────────────────────────────────

def _display_name(project: str, exp: dict) -> str:
    if project == "VBP":
        return f"{exp['setup']} / {exp['kr_folder']}"
    return exp.get("exp_name", "")


def _lsf_keys(project: str, exp: dict) -> list[str]:
    """Candidate -D values to look up in the LSF index (most specific first)."""
    if project == "VBP":
        return [
            exp["setup"],
            f"{exp['setup']}/{exp['kr_folder']}",
            exp["kr_folder"],
        ]
    name = exp.get("exp_name", "")
    return [name] if name else []

# ── Column headers ────────────────────────────────────────────────────────────

_TABLE_HEADERS = ["Name", "Status", "Command"]
_ARG_HEADERS   = ["Arg", "Value"]

# ── Widget ────────────────────────────────────────────────────────────────────

class MonitorScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: str = ""
        self._data: list = []
        self._exp_infos: list[dict] = []   # parallel to table rows

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Top bar ───────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(QLabel("Filter:"))

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter experiments…")
        self.filter_input.setClearButtonEnabled(True)
        self.filter_input.textChanged.connect(self._apply_filter)
        top.addWidget(self.filter_input, stretch=1)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #555; font-size: 11px;")
        top.addWidget(self.status_label)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setFixedWidth(90)
        self.refresh_btn.clicked.connect(self.refresh)
        top.addWidget(self.refresh_btn)

        layout.addLayout(top)

        # ── Splitter: experiments table / launch panel ────────────────
        splitter = QSplitter(Qt.Vertical)

        # Experiments table
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setStyleSheet(
            "QTableWidget { font-family: monospace; font-size: 12px; }"
            "QHeaderView::section { background: #f0f0f0; font-weight: bold; padding: 4px; }"
        )
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        splitter.addWidget(self.table)

        # Launch panel
        self.launch_box = QGroupBox("Launch")
        self.launch_box.setVisible(False)
        launch_layout = QVBoxLayout(self.launch_box)
        launch_layout.setSpacing(4)

        self.args_table = QTableWidget()
        self.args_table.setColumnCount(2)
        self.args_table.setHorizontalHeaderLabels(_ARG_HEADERS)
        self.args_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.args_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.args_table.verticalHeader().setVisible(False)
        self.args_table.setStyleSheet(
            "QTableWidget { font-family: monospace; font-size: 12px; }"
            "QHeaderView::section { background: #f0f0f0; font-weight: bold; padding: 4px; }"
        )
        self.args_table.cellChanged.connect(self._update_cmd_preview)
        launch_layout.addWidget(self.args_table)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)

        self.cmd_preview = QLabel("")
        self.cmd_preview.setStyleSheet("font-family: monospace; font-size: 10px; color: #555;")
        self.cmd_preview.setWordWrap(True)
        bottom_row.addWidget(self.cmd_preview, stretch=1)

        self.launch_btn = QPushButton("Launch")
        self.launch_btn.setFixedWidth(90)
        self.launch_btn.setStyleSheet(
            "QPushButton { background-color: #0d6efd; color: white; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #0b5ed7; }"
            "QPushButton:disabled { background-color: #aaa; }"
        )
        self.launch_btn.clicked.connect(self._on_launch)
        bottom_row.addWidget(self.launch_btn)

        launch_layout.addLayout(bottom_row)
        splitter.addWidget(self.launch_box)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    # ── Public ────────────────────────────────────────────────────────

    def load(self, project: str, data: list):
        self._project = project
        self._data = data
        self.refresh()

    def refresh(self):
        self.refresh_btn.setEnabled(False)
        self.status_label.setText("Loading…")
        self.launch_box.setVisible(False)
        self._exp_infos.clear()
        self._fill_table()
        self.refresh_btn.setEnabled(True)

    # ── Internal ──────────────────────────────────────────────────────

    def _fill_table(self):
        lsf_index = _build_lsf_index()

        self.table.clearContents()
        self.table.setRowCount(0)
        self.table.setColumnCount(len(_TABLE_HEADERS))
        self.table.setHorizontalHeaderLabels(_TABLE_HEADERS)

        if not self._data:
            self.status_label.setText("No experiments loaded. Select a project first.")
            return

        bold = QFont("monospace", 12)
        bold.setBold(True)

        for exp in self._data:
            dname = _display_name(self._project, exp)
            keys  = _lsf_keys(self._project, exp)

            lsf_info = next(
                (lsf_index[k] for k in keys if k in lsf_index),
                None,
            )
            label, color = (lsf_info["label"], lsf_info["color"]) \
                if lsf_info else _UNKNOWN_STATUS
            command = lsf_info["command"] if lsf_info else ""

            self._exp_infos.append({
                "display_name": dname,
                "label":        label,
                "command":      command,
                "lsf_info":     lsf_info,
            })

            row = self.table.rowCount()
            self.table.insertRow(row)

            for col, text in enumerate([dname, label, command]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                if col == 1:
                    item.setForeground(QBrush(color))
                    item.setFont(bold)
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(1, min(self.table.columnWidth(1), 120))

        n = len(self._data)
        n_run = sum(1 for i in self._exp_infos if i["label"] == "In Progress")
        self.status_label.setText(
            f"{n} experiment{'s' if n != 1 else ''}  •  {n_run} running"
        )

    def _apply_filter(self, text: str):
        text = text.strip().lower()
        for row in range(self.table.rowCount()):
            match = any(
                text in (self.table.item(row, col).text().lower()
                         if self.table.item(row, col) else "")
                for col in range(self.table.columnCount())
            )
            self.table.setRowHidden(row, not match if text else False)

    def _on_row_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.launch_box.setVisible(False)
            return
        row = rows[0].row()
        if row >= len(self._exp_infos):
            return
        self._populate_launch_panel(self._exp_infos[row])

    def _populate_launch_panel(self, info: dict):
        self.launch_box.setTitle(f"Launch — {info['display_name']}")
        arg_rows = _parse_command(info["command"]) if info["command"] else []

        self.args_table.blockSignals(True)
        self.args_table.clearContents()
        self.args_table.setRowCount(len(arg_rows))

        grey = QColor("#f0f0f0")
        for r, (flag, value) in enumerate(arg_rows):
            flag_item = QTableWidgetItem(flag)
            flag_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            flag_item.setBackground(QBrush(grey))
            self.args_table.setItem(r, 0, flag_item)
            self.args_table.setItem(r, 1, QTableWidgetItem(value))

        self.args_table.blockSignals(False)
        self._update_cmd_preview()
        self.launch_box.setVisible(True)

    def _update_cmd_preview(self):
        cmd = _reconstruct_command(self._collect_arg_rows())
        preview = cmd if len(cmd) <= 120 else cmd[:117] + "…"
        self.cmd_preview.setText(preview)

    def _collect_arg_rows(self) -> list[tuple[str, str]]:
        rows = []
        for r in range(self.args_table.rowCount()):
            fi = self.args_table.item(r, 0)
            vi = self.args_table.item(r, 1)
            rows.append((fi.text() if fi else "", vi.text() if vi else ""))
        return rows

    def _on_launch(self):
        cmd = _reconstruct_command(self._collect_arg_rows()).strip()
        if not cmd:
            return
        self.status_label.setText("Launching…")
        try:
            subprocess.Popen(cmd, shell=True)
            self.status_label.setText("Launched.")
        except Exception as e:
            self.status_label.setText(f"Launch failed: {e}")
