"""Monitor screen: LSF job status table."""

import os
import subprocess
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLabel, QAbstractItemView, QLineEdit,
    QSplitter, QTextEdit, QGroupBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QFont

from config import LSF_LOG_DIR

# ── Status colours ────────────────────────────────────────────────────────────

_C_COMPLETE    = QColor("#28a745")   # green
_C_IN_PROGRESS = QColor("#fd7e14")   # orange
_C_CRASH       = QColor("#dc3545")   # red
_C_PENDING     = QColor("#888888")   # grey

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
    """Parse `bjobs -a` into list of {job_id, stat, queue, exec_host}."""
    out = _shell("bjobs -a 2>/dev/null")
    jobs = []
    for line in out.splitlines()[1:]:   # skip header
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
    """Search LOG_DIR/<DD-MM-YY>/<job_id>.out going backward in time."""
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
    """Extract the value of -D from the command line in the log."""
    idx = log.find("-D ")
    if idx == -1:
        return ""
    rest = log[idx + 3:].lstrip()
    if rest.startswith(("'", '"')):
        q = rest[0]
        end = rest.find(q, 1)
        return rest[1:end] if end != -1 else rest[1:]
    # unquoted — up to first space or newline
    for ch in (" ", "\t", "\n"):
        pos = rest.find(ch)
        if pos != -1:
            rest = rest[:pos]
    return rest


def _extract_command(log: str) -> str:
    """Return the training command line from the log (first /algo/ws line or bsub line)."""
    for prefix in ("/algo/ws", "bsub ", "drun "):
        idx = log.find(prefix)
        if idx != -1:
            end = log.find("\n", idx)
            return (log[idx:end] if end != -1 else log[idx:]).strip()
    return ""


def _extract_branch(command: str) -> str:
    """Try to extract a git branch hint from the command string."""
    for flag in ("--branch ", "-branch ", "--br "):
        idx = command.find(flag)
        if idx != -1:
            rest = command[idx + len(flag):].split()[0]
            return rest.strip("'\"")
    return ""


def _stat_label(stat: str) -> tuple[str, QColor]:
    return _LSF_STAT_MAP.get(stat.upper(), (stat, _C_PENDING))


# ── Widget ────────────────────────────────────────────────────────────────────

_HEADERS = ["Job ID", "Name (-D)", "Status", "Command"]


class MonitorScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[dict] = []   # raw job dicts for detail panel

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Top bar ───────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(8)

        filter_label = QLabel("Filter:")
        top.addWidget(filter_label)

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter rows…")
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

        # ── Splitter: table + detail ──────────────────────────────────
        splitter = QSplitter(Qt.Vertical)

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

        self.detail_box = QGroupBox("Detail")
        self.detail_box.setVisible(False)
        detail_layout = QVBoxLayout(self.detail_box)
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setFont(QFont("monospace", 11))
        self.detail_text.setMaximumHeight(180)
        detail_layout.addWidget(self.detail_text)
        splitter.addWidget(self.detail_box)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        layout.addWidget(splitter)

    # ── Public ────────────────────────────────────────────────────────

    def load(self, project: str, _data: list):
        """Called by MainWindow when project changes — auto-refresh."""
        self.refresh()

    def refresh(self):
        self.refresh_btn.setEnabled(False)
        self.status_label.setText("Loading…")
        self.detail_box.setVisible(False)
        self._rows.clear()
        self._fill_table()
        self.refresh_btn.setEnabled(True)

    # ── Internal ──────────────────────────────────────────────────────

    def _fill_table(self):
        jobs = _bjobs_all()

        self.table.clearContents()
        self.table.setRowCount(0)
        self.table.setColumnCount(len(_HEADERS))
        self.table.setHorizontalHeaderLabels(_HEADERS)

        if not jobs:
            self.status_label.setText("No jobs found (bjobs returned nothing).")
            return

        bold = QFont("monospace", 12)
        bold.setBold(True)

        for job in jobs:
            job_id = job["job_id"]
            stat   = job["stat"]
            label, color = _stat_label(stat)

            log      = _read_log(job_id)
            name     = _extract_name(log)
            command  = _extract_command(log)
            branch   = _extract_branch(command)

            self._rows.append({
                "job_id":  job_id,
                "stat":    stat,
                "label":   label,
                "name":    name,
                "command": command,
                "branch":  branch,
                "log":     log,
                **{k: job[k] for k in ("queue", "exec_host")},
            })

            row = self.table.rowCount()
            self.table.insertRow(row)

            for col, text in enumerate([job_id, name, label, command]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                if col == 2:                       # Status — coloured + bold
                    item.setForeground(QBrush(color))
                    item.setFont(bold)
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()
        # cap Job ID and Status columns so command gets space
        self.table.setColumnWidth(0, min(self.table.columnWidth(0), 90))
        self.table.setColumnWidth(2, min(self.table.columnWidth(2), 110))

        n = len(jobs)
        n_run = sum(1 for j in jobs if j["stat"].upper() == "RUN")
        self.status_label.setText(
            f"{n} job{'s' if n != 1 else ''}  •  {n_run} running"
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
            self.detail_box.setVisible(False)
            return
        visual_row = rows[0].row()
        # map visual row back to self._rows (hidden rows shift visual index)
        visible = [r for r in range(self.table.rowCount())
                   if not self.table.isRowHidden(r)]
        if visual_row >= len(visible):
            return
        real_row = visible[visual_row]
        if real_row >= len(self._rows):
            return

        d = self._rows[real_row]
        lines = [
            f"Job ID   : {d['job_id']}",
            f"Status   : {d['label']}  (LSF: {d['stat']})",
            f"Queue    : {d['queue']}",
            f"Host     : {d['exec_host']}",
        ]
        if d["branch"]:
            lines.append(f"Branch   : {d['branch']}")
        lines += ["", "── Command ──", d["command"] or "(not found in log)"]

        self.detail_text.setPlainText("\n".join(lines))
        title = d["name"] or d["job_id"]
        self.detail_box.setTitle(f"Detail — {title}")
        self.detail_box.setVisible(True)
