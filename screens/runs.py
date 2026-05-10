"""Runs screen: sortable experiment table + collapsible detail panel.

When 2+ rows are selected (VBP), the detail panel switches into a
side-by-side hyperparam comparison table with a "Diff only" toggle and
row highlighting for keys whose values differ across runs.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QTextEdit, QLabel, QAbstractItemView, QLineEdit,
    QStackedWidget, QCheckBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush

from ui import theme

ODT_PRIMARY = ["AP", "DR", "mIoU", "total_metric"]


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _fmt_hp(v) -> str:
    """Compact format for hyperparam comparison cells."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e6:
            return f"{int(v)}"
        return f"{v:g}"
    return str(v)


class RunsScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: list = []
        self._project: str = ""

        splitter = QSplitter(Qt.Vertical)

        # --- Filter bar ---
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 4)
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter rows…")
        self.filter_input.setClearButtonEnabled(True)
        self.filter_input.textChanged.connect(self._apply_filter)
        filter_row.addWidget(QLabel("Filter:"))
        filter_row.addWidget(self.filter_input)

        self.compare_hint = QLabel("Tip: Ctrl/Shift-click rows to compare")
        self.compare_hint.setStyleSheet("color: gray; font-size: 11px;")
        filter_row.addWidget(self.compare_hint)

        # --- Table ---
        self.table = QTableWidget()
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        splitter.addWidget(self.table)

        # --- Detail panel: stacked (text view | compare table) ---
        self.detail_box = QGroupBox("Detail")
        self.detail_box.setVisible(False)
        detail_layout = QVBoxLayout(self.detail_box)

        # Top bar inside detail (only visible in compare mode)
        self.compare_bar = QWidget()
        cb_layout = QHBoxLayout(self.compare_bar)
        cb_layout.setContentsMargins(0, 0, 0, 4)
        self.diff_only = QCheckBox("Diff only")
        self.diff_only.setChecked(True)
        self.diff_only.toggled.connect(self._render_compare)
        cb_layout.addWidget(self.diff_only)
        cb_layout.addStretch()
        self.compare_bar.setVisible(False)
        detail_layout.addWidget(self.compare_bar)

        self.detail_stack = QStackedWidget()
        # Page 0: single-run text
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_stack.addWidget(self.detail_text)
        # Page 1: compare table
        self.compare_table = QTableWidget()
        self.compare_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.compare_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.compare_table.verticalHeader().setVisible(False)
        self.compare_table.setAlternatingRowColors(True)
        self.detail_stack.addWidget(self.compare_table)
        detail_layout.addWidget(self.detail_stack)
        splitter.addWidget(self.detail_box)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        self._splitter = splitter

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(filter_row)
        layout.addWidget(splitter)

    # ------------------------------------------------------------------
    def _apply_filter(self, text: str):
        text = text.strip().lower()
        for row in range(self.table.rowCount()):
            match = any(
                text in (self.table.item(row, col).text().lower() if self.table.item(row, col) else "")
                for col in range(self.table.columnCount())
            )
            self.table.setRowHidden(row, not match if text else False)

    def load(self, project: str, data: list):
        self._project = project
        self._data = data
        self.detail_box.setVisible(False)
        self.compare_bar.setVisible(False)
        self.filter_input.blockSignals(True)
        self.filter_input.clear()
        self.filter_input.blockSignals(False)

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

    def _add_row(self, values: list, data_index: int = -1):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, val in enumerate(values):
            item = QTableWidgetItem(_fmt(val))
            item.setTextAlignment(Qt.AlignCenter)
            if isinstance(val, (int, float)) and val is not None:
                item.setData(Qt.UserRole, val)
            # Store original data index in first column for detail lookup
            if col == 0:
                item.setData(Qt.UserRole + 1, data_index)
            self.table.setItem(row, col, item)

    # ------------------------------------------------------------------
    def _load_dvnr(self, data: list):
        loss_keys = list(data[0]["last_losses"].keys()) if data else []
        headers = ["Experiment", "Epochs"] + loss_keys
        self._set_columns(headers)
        for i, exp in enumerate(data):
            row = [exp["exp_name"], exp["n_epochs"]] + \
                  [exp["last_losses"].get(k) for k in loss_keys]
            self._add_row(row, i)
        self.table.resizeColumnsToContents()

    def _load_odt(self, data: list):
        all_keys = list(data[0]["metrics"].keys()) if data else []
        ordered = [k for k in ODT_PRIMARY if k in all_keys] + \
                  [k for k in all_keys if k not in ODT_PRIMARY]
        headers = ["Experiment"] + ordered
        self._set_columns(headers)
        for i, exp in enumerate(data):
            row = [exp["exp_name"]] + [exp["metrics"].get(k) for k in ordered]
            self._add_row(row, i)
        self.table.resizeColumnsToContents()

    def _load_vbp(self, data: list):
        headers = ["Setup", "KR Folder", "Keep Ratio", "Model", "Criterion",
                   "Orig Acc", "Final Acc", "Best Acc",
                   "Base MACs (G)", "Pruned MACs (G)", "Retention %"]
        self._set_columns(headers)
        for i, exp in enumerate(data):
            orig = exp.get("original_acc")
            final = exp.get("final_acc") or exp.get("best_acc")
            retention = round(100.0 * final / orig, 2) if orig and final else None
            row = [
                exp["setup"], exp["kr_folder"], exp.get("keep_ratio"),
                exp.get("model", ""), exp.get("criterion", ""),
                orig, exp.get("final_acc"), exp.get("best_acc"),
                exp.get("base_macs_G"), exp.get("pruned_macs_G"), retention,
            ]
            self._add_row(row, i)
        self.table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    def _selected_data_indices(self) -> list[int]:
        """Unique data indices from selected rows, in selection order."""
        seen: set[int] = set()
        out: list[int] = []
        for idx in self.table.selectionModel().selectedRows():
            first = self.table.item(idx.row(), 0)
            if first is None:
                continue
            di = first.data(Qt.UserRole + 1)
            if di is None or di in seen or di >= len(self._data):
                continue
            seen.add(di)
            out.append(di)
        return out

    def _on_row_selected(self):
        indices = self._selected_data_indices()
        if not indices:
            self.detail_box.setVisible(False)
            return

        # Compare mode: 2+ runs (VBP only — others lack rich hyperparams)
        if len(indices) >= 2 and self._project == "VBP":
            self._compare_indices = indices
            self.compare_bar.setVisible(True)
            self.detail_stack.setCurrentIndex(1)
            self._render_compare()
            self.detail_box.setTitle(f"Compare — {len(indices)} runs")
            self.detail_box.setVisible(True)
            return

        # Single-run text view
        self.compare_bar.setVisible(False)
        self.detail_stack.setCurrentIndex(0)
        self._render_single(indices[0])

    # ── Single-run text view ──────────────────────────────────────────
    def _render_single(self, data_idx: int):
        exp = self._data[data_idx]
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

    # ── Compare table ────────────────────────────────────────────────
    def _render_compare(self):
        indices = getattr(self, "_compare_indices", [])
        if not indices:
            return
        runs = [self._data[i] for i in indices]
        labels = [f"{r['setup']}/{r['kr_folder']}" for r in runs]

        # Union of hyperparam keys across selected runs
        keys: list[str] = []
        seen_keys: set = set()
        for r in runs:
            hp = r.get("hyperparams") or {}
            for k in hp.keys():
                if k not in seen_keys:
                    seen_keys.add(k)
                    keys.append(k)
        keys.sort()

        diff_only = self.diff_only.isChecked()
        hi_color = QColor(theme.DIFF_HIGHLIGHT)
        rows: list[tuple[str, list, bool]] = []
        for k in keys:
            vals = [(r.get("hyperparams") or {}).get(k) for r in runs]
            differs = any(v != vals[0] for v in vals[1:])
            if diff_only and not differs:
                continue
            rows.append((k, vals, differs))

        # Build table
        self.compare_table.setSortingEnabled(False)
        self.compare_table.clearContents()
        self.compare_table.setColumnCount(1 + len(runs))
        self.compare_table.setHorizontalHeaderLabels(["Flag"] + labels)
        self.compare_table.setRowCount(len(rows))

        diff_brush = QBrush(hi_color)
        for r, (k, vals, differs) in enumerate(rows):
            key_item = QTableWidgetItem(k)
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
            if differs:
                key_item.setBackground(diff_brush)
            self.compare_table.setItem(r, 0, key_item)
            for c, v in enumerate(vals, start=1):
                cell = QTableWidgetItem(_fmt_hp(v))
                cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
                cell.setTextAlignment(Qt.AlignCenter)
                if differs:
                    cell.setBackground(diff_brush)
                self.compare_table.setItem(r, c, cell)

        self.compare_table.resizeColumnsToContents()
        self.compare_table.setSortingEnabled(True)
