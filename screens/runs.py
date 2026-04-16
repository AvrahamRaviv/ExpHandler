"""Runs screen: sortable experiment table + collapsible detail panel."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QTextEdit, QLabel, QAbstractItemView,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

ODT_PRIMARY = ["AP", "DR", "mIoU", "total_metric"]


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


class RunsScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: list = []
        self._project: str = ""

        splitter = QSplitter(Qt.Vertical)

        # --- Table ---
        self.table = QTableWidget()
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableWidget { font-family: monospace; font-size: 12px; }"
            "QHeaderView::section { background: #f0f0f0; font-weight: bold; padding: 4px; }"
        )
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        splitter.addWidget(self.table)

        # --- Detail panel ---
        self.detail_box = QGroupBox("Detail")
        self.detail_box.setVisible(False)
        detail_layout = QVBoxLayout(self.detail_box)
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setFont(QFont("monospace", 11))
        self.detail_text.setMaximumHeight(200)
        detail_layout.addWidget(self.detail_text)
        splitter.addWidget(self.detail_box)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(splitter)

    # ------------------------------------------------------------------
    def load(self, project: str, data: list):
        self._project = project
        self._data = data
        self.detail_box.setVisible(False)

        if project == "DVNR":
            self._load_dvnr(data)
        elif project == "ODT":
            self._load_odt(data)
        elif project == "VBP":
            self._load_vbp(data)

    # ------------------------------------------------------------------
    def _set_columns(self, headers: list[str]):
        self.table.clearContents()
        self.table.setRowCount(0)
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

    def _add_row(self, values: list):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, val in enumerate(values):
            item = QTableWidgetItem(_fmt(val))
            item.setTextAlignment(Qt.AlignCenter)
            # Store raw value for sorting
            if isinstance(val, (int, float)) and val is not None:
                item.setData(Qt.UserRole, val)
            self.table.setItem(row, col, item)

    # ------------------------------------------------------------------
    def _load_dvnr(self, data: list):
        loss_keys = list(data[0]["last_losses"].keys()) if data else []
        headers = ["Experiment", "Epochs"] + loss_keys
        self._set_columns(headers)
        for exp in data:
            row = [exp["exp_name"], exp["n_epochs"]] + \
                  [exp["last_losses"].get(k) for k in loss_keys]
            self._add_row(row)
        self.table.resizeColumnsToContents()

    def _load_odt(self, data: list):
        all_keys = list(data[0]["metrics"].keys()) if data else []
        ordered = [k for k in ODT_PRIMARY if k in all_keys] + \
                  [k for k in all_keys if k not in ODT_PRIMARY]
        headers = ["Experiment"] + ordered
        self._set_columns(headers)
        for exp in data:
            row = [exp["exp_name"]] + [exp["metrics"].get(k) for k in ordered]
            self._add_row(row)
        self.table.resizeColumnsToContents()

    def _load_vbp(self, data: list):
        headers = ["Setup", "KR Folder", "Keep Ratio", "Model", "Criterion",
                   "Orig Acc", "Final Acc", "Best Acc",
                   "Base MACs (G)", "Pruned MACs (G)", "Retention %"]
        self._set_columns(headers)
        for exp in data:
            orig = exp.get("original_acc")
            final = exp.get("final_acc") or exp.get("best_acc")
            retention = round(100.0 * final / orig, 2) if orig and final else None
            row = [
                exp["setup"], exp["kr_folder"], exp.get("keep_ratio"),
                exp.get("model", ""), exp.get("criterion", ""),
                orig, exp.get("final_acc"), exp.get("best_acc"),
                exp.get("base_macs_G"), exp.get("pruned_macs_G"), retention,
            ]
            self._add_row(row)
        self.table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    def _on_row_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.detail_box.setVisible(False)
            return
        row_idx = rows[0].row()
        if row_idx >= len(self._data):
            return

        exp = self._data[row_idx]
        lines = []

        if self._project == "DVNR":
            lines.append(f"Experiment : {exp['exp_name']}")
            lines.append(f"Epochs     : {exp['n_epochs']}")
            lines.append("")
            for k, v in exp["last_losses"].items():
                lines.append(f"  {k:<45} {v:.4f}")

        elif self._project == "ODT":
            lines.append(f"Experiment : {exp['exp_name']}")
            lines.append("")
            for k, v in exp["metrics"].items():
                lines.append(f"  {k:<30} {_fmt(v)}")

        elif self._project == "VBP":
            lines.append(f"Setup    : {exp['setup']}  /  {exp['kr_folder']}")
            lines.append("")
            lines.append("── Hyperparams ──")
            for k, v in (exp.get("hyperparams") or {}).items():
                lines.append(f"  {k:<40} {_fmt(v)}")
            lines.append("")
            lines.append("── Summary ──")
            for k, v in (exp.get("summary") or {}).items():
                lines.append(f"  {k:<40} {_fmt(v)}")
            ret = exp.get("step_retentions") or []
            if ret:
                lines.append("")
                lines.append("── Step Retention (last) ──")
                for k, v in ret[-1].items():
                    lines.append(f"  {k:<40} {_fmt(v)}")

        self.detail_text.setPlainText("\n".join(lines))
        self.detail_box.setTitle(f"Detail — {exp.get('exp_name') or exp.get('setup', '')}")
        self.detail_box.setVisible(True)
