"""Launcher screen: per-subtype experiment launcher (VBP).

Two-col table (arg | value), toolbar for sweep params (out_dir_name, e_s, e_e,
step), and a 'Launch sweep' button that mirrors examples/run_ddp.py: per
keep_ratio in the range, render run_ddp_<r>.sh and dispatch via wrapper cmd.
"""

import json
import os
import shlex
import subprocess
from typing import Any

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from config import get_project_path
from launcher_config import load_schema, save_schema


def _coerce(value: str, atype: str) -> Any:
    if atype == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if atype == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def _render_sh(schema: dict, values: dict, kr: float, save_dir: str) -> str:
    parts = [schema["ddp_prefix"], schema["entrypoint"]]
    for arg in schema["args"]:
        name = arg["name"]
        atype = arg.get("type", "str")
        v = values.get(name, arg.get("default"))
        if atype == "bool":
            if v:
                parts.append(f"--{name}")
        else:
            parts.append(f"--{name} {v}")
    parts.append(f"--{schema['kr_arg']} {kr}")
    parts.append(f"--{schema['save_dir_arg']} {save_dir}")
    return " ".join(parts)


def _sweep_values(e_s: float, e_e: float, step: float) -> list[float]:
    if e_e == 0:
        e_e = e_s - 0.01
    return [round(float(r), 2) for r in np.arange(e_s, e_e, -step)]


def _parse_sh(text: str) -> dict:
    """Parse run_ddp.sh-like content; return {flag: value_or_True}."""
    cleaned = text.replace("\\\n", " ")
    try:
        tokens = shlex.split(cleaned, comments=True, posix=True)
    except ValueError:
        tokens = cleaned.split()
    parsed: dict = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--") and len(t) > 2:
            body = t[2:]
            if "=" in body:
                k, v = body.split("=", 1)
                parsed[k] = v
            else:
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                    parsed[body] = tokens[i + 1]
                    i += 1
                else:
                    parsed[body] = True
        i += 1
    return parsed


class _SchemaEditDialog(QDialog):
    """Plain-JSON editor for a subtype schema."""

    def __init__(self, subtype: str, schema: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit defaults — {subtype}")
        self.resize(900, 700)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Schema (JSON). 'args' is the ordered list of CLI flags. Types: "
            "str / int / float / bool. keep_ratio and save_dir are derived per kr."
        ))
        self.editor = QPlainTextEdit()
        self.editor.setFont(QFont("Menlo", 11))
        self.editor.setPlainText(json.dumps(schema, indent=2))
        layout.addWidget(self.editor, stretch=1)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._on_save)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)
        self._result: dict | None = None

    def _on_save(self):
        try:
            parsed = json.loads(self.editor.toPlainText())
        except json.JSONDecodeError as e:
            QMessageBox.warning(self, "Invalid JSON", str(e))
            return
        for required in ("args", "ddp_prefix", "entrypoint", "wrapper_a",
                         "wrapper_b", "desc_template", "kr_arg", "save_dir_arg"):
            if required not in parsed:
                QMessageBox.warning(self, "Missing field", f"Missing key: {required}")
                return
        if not isinstance(parsed["args"], list):
            QMessageBox.warning(self, "Invalid args", "'args' must be a list.")
            return
        self._result = parsed
        self.accept()

    def result_schema(self) -> dict | None:
        return self._result


class LauncherScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._subtype: str = ""
        self._root_path: str = ""
        self._schema: dict = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        # ── Header / status ───────────────────────────────────────────
        self.header = QLabel("Launcher")
        f = self.header.font()
        f.setBold(True)
        f.setPointSize(f.pointSize() + 1)
        self.header.setFont(f)
        outer.addWidget(self.header)

        # ── Sweep toolbar ─────────────────────────────────────────────
        sweep_row = QHBoxLayout()
        sweep_row.addWidget(QLabel("out_dir_name:"))
        self.out_dir_input = QLineEdit()
        self.out_dir_input.setPlaceholderText("e.g. baseA")
        sweep_row.addWidget(self.out_dir_input, stretch=2)

        sweep_row.addWidget(QLabel("e_s:"))
        self.e_s = QDoubleSpinBox()
        self.e_s.setRange(0.01, 1.0)
        self.e_s.setSingleStep(0.05)
        self.e_s.setDecimals(2)
        self.e_s.setValue(0.95)
        sweep_row.addWidget(self.e_s)

        sweep_row.addWidget(QLabel("e_e:"))
        self.e_e = QDoubleSpinBox()
        self.e_e.setRange(0.0, 1.0)
        self.e_e.setSingleStep(0.05)
        self.e_e.setDecimals(2)
        self.e_e.setValue(0.0)
        sweep_row.addWidget(self.e_e)

        sweep_row.addWidget(QLabel("step:"))
        self.step = QDoubleSpinBox()
        self.step.setRange(0.01, 1.0)
        self.step.setSingleStep(0.05)
        self.step.setDecimals(2)
        self.step.setValue(0.1)
        sweep_row.addWidget(self.step)

        sweep_row.addStretch(1)
        outer.addLayout(sweep_row)

        # ── Action buttons ────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_new_run = QPushButton("New run")
        self.btn_new_run.setToolTip("Reset value column to schema defaults")
        self.btn_new_run.clicked.connect(self._on_new_run)
        btn_row.addWidget(self.btn_new_run)

        self.btn_edit_defaults = QPushButton("Edit defaults")
        self.btn_edit_defaults.setToolTip("Edit schema for this subtype")
        self.btn_edit_defaults.clicked.connect(self._on_edit_defaults)
        btn_row.addWidget(self.btn_edit_defaults)

        self.btn_load_sh = QPushButton("Load from .sh")
        self.btn_load_sh.setToolTip("Override values from existing run_ddp_<r>.sh")
        self.btn_load_sh.clicked.connect(self._on_load_sh)
        btn_row.addWidget(self.btn_load_sh)

        self.btn_launch = QPushButton("Launch sweep")
        self.btn_launch.clicked.connect(self._on_launch)
        btn_row.addWidget(self.btn_launch)

        self.btn_clear_log = QPushButton("Clear log")
        self.btn_clear_log.clicked.connect(lambda: self.log.clear())
        btn_row.addWidget(self.btn_clear_log)

        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        # ── Splitter: table on top, log below ─────────────────────────
        splitter = QSplitter(Qt.Vertical)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["arg", "value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        splitter.addWidget(self.table)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Menlo", 10))
        self.log.setPlaceholderText("Launch output appears here…")
        splitter.addWidget(self.log)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, stretch=1)

    # ── Public API ────────────────────────────────────────────────────
    def load(self, subtype: str, root_path: str):
        self._subtype = subtype
        self._root_path = root_path
        self._schema = load_schema(subtype)
        self.header.setText(f"Launcher  —  VBP / {subtype}")
        self._populate_table_from_defaults()

    # ── Table helpers ─────────────────────────────────────────────────
    def _populate_table_from_defaults(self):
        args = self._schema.get("args", [])
        self.table.setRowCount(len(args))
        for i, arg in enumerate(args):
            name_item = QTableWidgetItem(arg["name"])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, name_item)

            atype = arg.get("type", "str")
            default = arg.get("default")
            if atype == "bool":
                cb = QCheckBox()
                cb.setChecked(bool(default))
                # Centered cell widget wrapper
                holder = QWidget()
                lay = QHBoxLayout(holder)
                lay.setContentsMargins(4, 0, 0, 0)
                lay.addWidget(cb)
                lay.addStretch(1)
                self.table.setCellWidget(i, 1, holder)
            else:
                item = QTableWidgetItem("" if default is None else str(default))
                self.table.setItem(i, 1, item)

    def _read_values(self) -> dict:
        values: dict = {}
        for i, arg in enumerate(self._schema.get("args", [])):
            name = arg["name"]
            atype = arg.get("type", "str")
            if atype == "bool":
                holder = self.table.cellWidget(i, 1)
                cb = holder.findChild(QCheckBox) if holder else None
                values[name] = bool(cb.isChecked()) if cb else False
            else:
                item = self.table.item(i, 1)
                raw = item.text() if item else ""
                values[name] = _coerce(raw, atype)
        return values

    # ── Actions ───────────────────────────────────────────────────────
    def _on_new_run(self):
        self._populate_table_from_defaults()

    def _on_load_sh(self):
        if not self._subtype:
            return
        vbp_root = self._root_path or get_project_path("VBP") or ""
        start = (os.path.join(vbp_root, self._subtype)
                 if vbp_root and os.path.isdir(os.path.join(vbp_root, self._subtype))
                 else os.path.expanduser("~"))
        path, _ = QFileDialog.getOpenFileName(
            self, "Select run_ddp.sh", start,
            "Shell scripts (*.sh);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                text = f.read()
        except OSError as e:
            QMessageBox.warning(self, "Read error", str(e))
            return

        parsed = _parse_sh(text)
        kr_arg = self._schema.get("kr_arg", "keep_ratio")
        save_dir_arg = self._schema.get("save_dir_arg", "save_dir")

        # Apply parsed values to table (skip derived args)
        schema_args = self._schema.get("args", [])
        skip = {kr_arg, save_dir_arg}
        matched: list[str] = []
        for i, arg in enumerate(schema_args):
            name = arg["name"]
            if name in skip or name not in parsed:
                continue
            atype = arg.get("type", "str")
            v = parsed[name]
            if atype == "bool":
                holder = self.table.cellWidget(i, 1)
                cb = holder.findChild(QCheckBox) if holder else None
                if cb:
                    cb.setChecked(True if v is True else bool(v))
            else:
                item = self.table.item(i, 1)
                if item is not None:
                    item.setText(str(v))
            matched.append(name)

        # Auto-fill e_s from parsed keep_ratio
        e_s_msg = ""
        if kr_arg in parsed and parsed[kr_arg] is not True:
            try:
                kr_val = float(parsed[kr_arg])
                self.e_s.setValue(kr_val)
                e_s_msg = f", e_s={kr_val}"
            except (TypeError, ValueError):
                pass

        schema_names = {a["name"] for a in schema_args}
        unknown = sorted(k for k in parsed if k not in schema_names and k not in skip)
        unk_msg = f", ignored: {unknown}" if unknown else ""
        self._append_log(
            f"[loaded {os.path.basename(path)}: {len(matched)} args overridden"
            f"{e_s_msg}{unk_msg}]"
        )

    def _on_edit_defaults(self):
        if not self._subtype:
            return
        dlg = _SchemaEditDialog(self._subtype, self._schema, parent=self)
        if dlg.exec_() == QDialog.Accepted and dlg.result_schema() is not None:
            self._schema = dlg.result_schema()
            save_schema(self._subtype, self._schema)
            self._populate_table_from_defaults()
            self._append_log(f"[schema saved for {self._subtype}]")

    def _on_launch(self):
        if not self._subtype:
            QMessageBox.warning(self, "No subtype", "Select a VBP subtype first.")
            return
        out_dir_name = self.out_dir_input.text().strip()
        if not out_dir_name:
            QMessageBox.warning(self, "Missing out_dir_name", "Set out_dir_name first.")
            return

        vbp_root = self._root_path or get_project_path("VBP") or ""
        if not vbp_root or not os.path.isdir(vbp_root):
            QMessageBox.warning(self, "Bad VBP root", f"VBP root invalid: {vbp_root}")
            return
        base_out_dir = os.path.join(vbp_root, self._subtype)

        e_s = float(self.e_s.value())
        e_e = float(self.e_e.value())
        step = float(self.step.value())
        rs = _sweep_values(e_s, e_e, step)
        if not rs:
            QMessageBox.warning(self, "Empty sweep",
                                f"No keep_ratio values from arange({e_s}, {e_e}, -{step}).")
            return

        confirm = QMessageBox.question(
            self, "Confirm launch",
            f"Launch {len(rs)} run(s) for {self._subtype} / {out_dir_name}?\n"
            f"keep_ratio: {rs}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        values = self._read_values()
        self._run_sweep(values, base_out_dir, out_dir_name, rs)

    # ── Sweep execution ───────────────────────────────────────────────
    def _run_sweep(self, values: dict, base_out_dir: str,
                   out_dir_name: str, rs: list[float]):
        schema = self._schema
        out_dir = os.path.join(base_out_dir, out_dir_name)
        self._append_log(f"=== Launching sweep: {self._subtype} / {out_dir_name} ===")
        self._append_log(f"keep_ratio values: {rs}")

        for r in rs:
            kr_dir = os.path.join(out_dir, f"kr_{r}")
            try:
                os.makedirs(kr_dir, exist_ok=True)
                try:
                    os.chmod(kr_dir, 0o777)
                except OSError:
                    pass
            except OSError as e:
                self._append_log(f"[mkdir failed for {kr_dir}: {e}]")
                continue

            sh_text = _render_sh(schema, values, r, kr_dir)
            sh_path = os.path.join(kr_dir, f"run_ddp_{r}.sh")
            try:
                with open(sh_path, "w") as f:
                    f.write("#!/usr/bin/env bash\n" + sh_text + "\n")
                try:
                    os.chmod(sh_path, 0o777)
                except OSError:
                    pass
            except OSError as e:
                self._append_log(f"[write failed for {sh_path}: {e}]")
                continue

            desc = schema["desc_template"].format(
                subtype=self._subtype, out_dir_name=out_dir_name, keep_ratio=r,
            )
            cmd = (schema["wrapper_a"] + sh_path + schema["wrapper_b"]
                   + f" -D '{desc}'")
            self._append_log(f"\n$ {cmd}")
            try:
                res = subprocess.run(cmd, shell=True, capture_output=True,
                                     text=True, timeout=300)
                if res.stdout:
                    self._append_log(res.stdout.rstrip())
                if res.stderr:
                    self._append_log("STDERR: " + res.stderr.rstrip())
                self._append_log(f"-> rc={res.returncode}")
            except subprocess.TimeoutExpired:
                self._append_log("[timeout: 300s]")
            except Exception as e:
                self._append_log(f"[error: {e}]")

        self._append_log("\n=== Sweep done ===")

    def _append_log(self, msg: str):
        self.log.appendPlainText(msg)
        # Force UI refresh between blocking subprocess calls
        self.log.repaint()
