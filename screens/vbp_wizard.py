"""VBP Run Wizard.

4-step state machine that builds a python invocation of vbp_imagenet_pat.py:

    1. Architecture        (model_type, cnn_arch, model_name, ...)
    2. Criterion           (criterion + importance_mode)
    3. Regularization      (sparse_mode + its block)
    4. Fine-tuning / PAT / KD

Final page shows the assembled command, with Copy / Run / Save preset / Load.

Run mode is blocking subprocess.run (UI freezes during training), per spec.
Output mirrored to <repo>/logs/<arch>_<criterion>_<timestamp>.log.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QRadioButton,
    QStackedWidget, QVBoxLayout, QWidget,
)

from config import get_torch_pruning_script, save_torch_pruning_script


# ── Spec tables (declarative) ────────────────────────────────────────────

ARCHS: dict[str, dict] = {
    "MNv2":     {"model_type": "cnn",      "cnn_arch": "mobilenet_v2",
                 "model_name": None,            "interior_default": True},
    "RN50":     {"model_type": "cnn",      "cnn_arch": "resnet50",
                 "model_name": None,            "interior_default": False},
    "ConvNeXt": {"model_type": "convnext", "cnn_arch": None,
                 "model_name": "convnext_tiny", "interior_default": True},
    "DeiT-T":   {"model_type": "vit",      "cnn_arch": None,
                 "model_name": "/path/to/DeiT_tiny", "interior_default": True},
}

CRITERIA: dict[str, dict] = {
    "magnitude": {"criterion": "magnitude", "importance_mode": None,
                  "extras": []},
    "VBP":       {"criterion": "variance",  "importance_mode": "variance",
                  "extras": ["no_compensation", "norm_per_layer"]},
    "tp_var":    {"criterion": "variance",  "importance_mode": "tp_variance",
                  "extras": ["no_compensation", "norm_per_layer", "no_recalib"]},
}

# (name, type, default)
SPARSE_MODES: dict[str, list[tuple[str, str, Any]]] = {
    "l1_group":  [("epochs_sparse", "int",   5),
                  ("lr_sparse",     "float", 1e-4),
                  ("l1_lambda",     "float", 1e-4)],
    "gmp":       [("epochs_sparse", "int",   5),
                  ("lr_sparse",     "float", 1e-4),
                  ("gmp_target_sparsity", "float", 0.5)],
    "reparam":   [("epochs_sparse", "int",   5),
                  ("lr_sparse",     "float", 1e-4),
                  ("reparam_lambda", "float", 0.01),
                  ("reparam_refresh_interval", "int", 1),
                  ("reparam_normalize", "bool", False),
                  ("reparam_target",   "str", "fc1")],
    "group_norm": [("epochs_sparse", "int",   5),
                   ("lr_sparse",     "float", 1e-4),
                   ("reg",           "float", 1e-4)],
}

# Always-on args shown in the "Advanced" panel on Step 1.
# (name, type, default)
ALWAYS_ON: list[tuple[str, str, Any]] = [
    ("data_path",        "str",   "/algo/NetOptimization/outputs/VBP/"),
    ("save_dir",         "str",   ""),
    ("keep_ratio",       "float", 0.95),
    ("max_batches",      "int",   None),
    ("disable_ddp",      "bool",  False),
    ("train_batch_size", "int",   128),
    ("val_batch_size",   "int",   128),
    ("num_workers",      "int",   8),
]

# Order in which build_command emits flags (later groups override earlier
# ones if duplicated, but state holds one value per key so duplicates can't
# happen).
FLAG_ORDER: list[str] = [
    # always-on
    "data_path", "save_dir", "keep_ratio", "max_batches", "disable_ddp",
    "train_batch_size", "val_batch_size", "num_workers",
    # arch
    "model_type", "cnn_arch", "model_name",
    "interior_only", "max_pruning_ratio", "global_pruning", "isomorphic",
    "checkpoint",
    # criterion
    "criterion", "importance_mode",
    "no_compensation", "norm_per_layer", "no_recalib",
    # reg
    "sparse_mode",
    "epochs_sparse", "lr_sparse", "l1_lambda",
    "gmp_target_sparsity",
    "reparam_lambda", "reparam_refresh_interval", "reparam_normalize",
    "reparam_target",
    "reg",
    # FT
    "epochs_ft", "lr_ft", "opt_ft", "momentum_ft", "wd_ft",
    # PAT
    "pat", "pat_steps", "pat_epochs_per_step", "var_loss_weight",
    "reparam_during_pat",
    # KD
    "use_kd", "kd_alpha", "kd_T",
]

BOOL_FLAGS: set[str] = {
    "disable_ddp", "interior_only", "global_pruning", "isomorphic",
    "no_compensation", "norm_per_layer", "no_recalib",
    "reparam_normalize", "pat", "reparam_during_pat", "use_kd",
}


# ── Helpers ──────────────────────────────────────────────────────────────


PRESET_DIR = os.path.expanduser("~/.exphandler/presets")


def _coerce(text: str, atype: str):
    text = text.strip()
    if text == "":
        return None
    if atype == "int":
        try:
            return int(text)
        except ValueError:
            return None
    if atype == "float":
        try:
            return float(text)
        except ValueError:
            return None
    return text


def build_command(state: dict, script_path: str) -> list[str]:
    """Assemble the python invocation as a list[str]."""
    cmd = ["python", script_path]
    for key in FLAG_ORDER:
        if key not in state:
            continue
        v = state[key]
        if v is None or v == "":
            continue
        if key in BOOL_FLAGS:
            if bool(v):
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(v)])
    return cmd


def _repo_root(script_path: str) -> str:
    """Walk up looking for .git; fallback to 3 levels above the script."""
    p = os.path.abspath(script_path)
    cur = os.path.dirname(p)
    seen = set()
    while cur and cur != "/" and cur not in seen:
        if os.path.isdir(os.path.join(cur, ".git")):
            return cur
        seen.add(cur)
        cur = os.path.dirname(cur)
    return os.path.dirname(os.path.dirname(os.path.dirname(p)))


# ── Step pages ───────────────────────────────────────────────────────────


def _hsep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


class _StepArch(QWidget):
    title = "1. Architecture"

    def __init__(self, state: dict, on_change):
        super().__init__()
        self.state = state
        self.on_change = on_change

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel("<b>Architecture</b>"))

        self.arch_group = QButtonGroup(self)
        arch_row = QHBoxLayout()
        self.arch_radios: dict[str, QRadioButton] = {}
        for name in ARCHS:
            rb = QRadioButton(name)
            rb.toggled.connect(self._on_arch_toggled)
            self.arch_group.addButton(rb)
            arch_row.addWidget(rb)
            self.arch_radios[name] = rb
        arch_row.addStretch(1)
        outer.addLayout(arch_row)

        # Arch-extra fields
        form = QFormLayout()
        self.model_name_in = QLineEdit()
        self.model_name_in.setPlaceholderText("(model_name override)")
        form.addRow("model_name:", self.model_name_in)

        self.interior_only_cb = QCheckBox("--interior_only")
        form.addRow(self.interior_only_cb)

        self.max_pr_in = QLineEdit("1.0")
        form.addRow("--max_pruning_ratio:", self.max_pr_in)

        self.global_cb = QCheckBox("--global_pruning")
        form.addRow(self.global_cb)
        self.iso_cb = QCheckBox("--isomorphic")
        form.addRow(self.iso_cb)

        ckpt_row = QHBoxLayout()
        self.checkpoint_in = QLineEdit()
        self.checkpoint_in.setPlaceholderText("(optional .pth)")
        btn = QPushButton("…")
        btn.setFixedWidth(28)
        btn.clicked.connect(self._pick_checkpoint)
        ckpt_row.addWidget(self.checkpoint_in)
        ckpt_row.addWidget(btn)
        form.addRow("--checkpoint:", ckpt_row)
        outer.addLayout(form)

        outer.addWidget(_hsep())

        # Always-on / advanced
        outer.addWidget(QLabel("<b>Always-on (advanced)</b>"))
        adv_form = QFormLayout()
        self.always_on_inputs: dict[str, QWidget] = {}
        for name, atype, default in ALWAYS_ON:
            if atype == "bool":
                cb = QCheckBox(f"--{name}")
                cb.setChecked(bool(default))
                adv_form.addRow(cb)
                self.always_on_inputs[name] = cb
            else:
                line = QLineEdit("" if default is None else str(default))
                adv_form.addRow(f"--{name}:", line)
                self.always_on_inputs[name] = line
        outer.addLayout(adv_form)
        outer.addStretch(1)

        # Default selection
        self.arch_radios["RN50"].setChecked(True)

    def _pick_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint", "", "Checkpoints (*.pth *.pt);;All (*)"
        )
        if path:
            self.checkpoint_in.setText(path)

    def _on_arch_toggled(self, checked: bool):
        if not checked:
            return
        for name, rb in self.arch_radios.items():
            if rb.isChecked():
                self.interior_only_cb.setChecked(ARCHS[name]["interior_default"])
                # Prefill model_name only if the arch ships one
                ship = ARCHS[name]["model_name"]
                if ship is not None and not self.model_name_in.text():
                    self.model_name_in.setText(ship)
                break

    def apply(self) -> None:
        for name, rb in self.arch_radios.items():
            if rb.isChecked():
                cfg = ARCHS[name]
                self.state["_arch_choice"] = name
                self.state["model_type"] = cfg["model_type"]
                if cfg["cnn_arch"] is not None and cfg["model_type"] == "cnn":
                    self.state["model_name"] = None
                    self.state["cnn_arch"] = cfg["cnn_arch"]
                else:
                    self.state["cnn_arch"] = None
                    name_text = self.model_name_in.text().strip()
                    self.state["model_name"] = name_text or cfg["model_name"]
                break

        self.state["interior_only"] = self.interior_only_cb.isChecked()
        v = _coerce(self.max_pr_in.text(), "float")
        self.state["max_pruning_ratio"] = v
        self.state["global_pruning"] = self.global_cb.isChecked()
        self.state["isomorphic"] = self.iso_cb.isChecked()
        ck = self.checkpoint_in.text().strip()
        self.state["checkpoint"] = ck or None

        for name, atype, _ in ALWAYS_ON:
            w = self.always_on_inputs[name]
            if isinstance(w, QCheckBox):
                self.state[name] = w.isChecked()
            else:
                self.state[name] = _coerce(w.text(), atype)

    def populate(self) -> None:
        choice = self.state.get("_arch_choice")
        if choice in self.arch_radios:
            self.arch_radios[choice].setChecked(True)
        if "interior_only" in self.state:
            self.interior_only_cb.setChecked(bool(self.state["interior_only"]))
        if self.state.get("max_pruning_ratio") is not None:
            self.max_pr_in.setText(str(self.state["max_pruning_ratio"]))
        self.global_cb.setChecked(bool(self.state.get("global_pruning")))
        self.iso_cb.setChecked(bool(self.state.get("isomorphic")))
        self.checkpoint_in.setText(self.state.get("checkpoint") or "")
        if self.state.get("model_name"):
            self.model_name_in.setText(self.state["model_name"])
        for name, atype, _ in ALWAYS_ON:
            if name not in self.state:
                continue
            w = self.always_on_inputs[name]
            v = self.state[name]
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            else:
                w.setText("" if v is None else str(v))

    def validate(self) -> list[str]:
        errs: list[str] = []
        kr = self.state.get("keep_ratio")
        if kr is None or not (0 < kr <= 1):
            errs.append("keep_ratio must be in (0, 1].")
        ck = self.state.get("checkpoint")
        if ck and not os.path.isfile(ck):
            errs.append(f"checkpoint file not found: {ck} (warning only)")
        return errs


class _StepCriterion(QWidget):
    title = "2. Criterion"

    def __init__(self, state: dict, on_change):
        super().__init__()
        self.state = state
        self.on_change = on_change

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel("<b>Criterion</b>"))

        self.crit_group = QButtonGroup(self)
        crit_row = QHBoxLayout()
        self.crit_radios: dict[str, QRadioButton] = {}
        for name in CRITERIA:
            rb = QRadioButton(name)
            rb.toggled.connect(self._refresh_extras)
            self.crit_group.addButton(rb)
            crit_row.addWidget(rb)
            self.crit_radios[name] = rb
        crit_row.addStretch(1)
        outer.addLayout(crit_row)

        outer.addWidget(_hsep())
        outer.addWidget(QLabel("<b>Extras</b>"))
        self.extras_form = QFormLayout()
        self.extra_inputs: dict[str, QCheckBox] = {}
        all_extras = sorted({e for c in CRITERIA.values() for e in c["extras"]})
        for ex in all_extras:
            cb = QCheckBox(f"--{ex}")
            self.extras_form.addRow(cb)
            self.extra_inputs[ex] = cb
        outer.addLayout(self.extras_form)
        outer.addStretch(1)

        self.crit_radios["VBP"].setChecked(True)
        self._refresh_extras()

    def _refresh_extras(self):
        choice = self._current_choice()
        if not choice:
            return
        active = set(CRITERIA[choice]["extras"])
        # CNN-only filter for no_recalib
        is_cnn = self.state.get("model_type") == "cnn" \
            or self.state.get("_arch_choice") in {"MNv2", "RN50"}
        for ex, cb in self.extra_inputs.items():
            visible = ex in active
            if ex == "no_recalib" and not is_cnn:
                visible = False
            cb.setVisible(visible)

    def _current_choice(self) -> str | None:
        for name, rb in self.crit_radios.items():
            if rb.isChecked():
                return name
        return None

    def apply(self) -> None:
        choice = self._current_choice()
        if choice is None:
            return
        cfg = CRITERIA[choice]
        self.state["_crit_choice"] = choice
        self.state["criterion"] = cfg["criterion"]
        self.state["importance_mode"] = cfg["importance_mode"]
        # Reset all extras first, then set those visible+checked
        for ex in self.extra_inputs:
            if ex in cfg["extras"]:
                cb = self.extra_inputs[ex]
                self.state[ex] = cb.isChecked() if cb.isVisible() else False
            else:
                self.state[ex] = False

    def populate(self) -> None:
        choice = self.state.get("_crit_choice")
        if choice in self.crit_radios:
            self.crit_radios[choice].setChecked(True)
        for ex, cb in self.extra_inputs.items():
            cb.setChecked(bool(self.state.get(ex, False)))
        self._refresh_extras()

    def validate(self) -> list[str]:
        return []


class _StepReg(QWidget):
    title = "3. Regularization"

    def __init__(self, state: dict, on_change):
        super().__init__()
        self.state = state
        self.on_change = on_change

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel("<b>Use sparse pre-training?</b>"))
        row = QHBoxLayout()
        self.no_rb = QRadioButton("No")
        self.yes_rb = QRadioButton("Yes")
        self.no_rb.setChecked(True)
        self.no_rb.toggled.connect(self._refresh)
        self.yes_rb.toggled.connect(self._refresh)
        row.addWidget(self.no_rb)
        row.addWidget(self.yes_rb)
        row.addStretch(1)
        outer.addLayout(row)

        # sparse_mode dropdown + dynamic block
        self.mode_box = QGroupBox("Sparse mode")
        mode_layout = QVBoxLayout(self.mode_box)
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("--sparse_mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(SPARSE_MODES.keys()))
        self.mode_combo.currentTextChanged.connect(self._rebuild_block)
        sel_row.addWidget(self.mode_combo)
        sel_row.addStretch(1)
        mode_layout.addLayout(sel_row)

        self.block_form = QFormLayout()
        self.block_widget = QWidget()
        self.block_widget.setLayout(self.block_form)
        mode_layout.addWidget(self.block_widget)
        outer.addWidget(self.mode_box)
        outer.addStretch(1)

        self._block_inputs: dict[str, QWidget] = {}
        self._refresh()
        self._rebuild_block(self.mode_combo.currentText())

    def _refresh(self):
        self.mode_box.setVisible(self.yes_rb.isChecked())

    def _rebuild_block(self, mode: str):
        # Clear
        while self.block_form.rowCount() > 0:
            self.block_form.removeRow(0)
        self._block_inputs.clear()
        for name, atype, default in SPARSE_MODES.get(mode, []):
            if atype == "bool":
                cb = QCheckBox(f"--{name}")
                cb.setChecked(bool(default))
                self.block_form.addRow(cb)
                self._block_inputs[name] = cb
            else:
                w = QLineEdit("" if default is None else str(default))
                self.block_form.addRow(f"--{name}:", w)
                self._block_inputs[name] = w

    def apply(self) -> None:
        if self.no_rb.isChecked():
            self.state["sparse_mode"] = "none"
            self.state["_reg_yes"] = False
            for name, _, _ in (
                SPARSE_MODES["l1_group"] + SPARSE_MODES["gmp"]
                + SPARSE_MODES["reparam"] + SPARSE_MODES["group_norm"]
            ):
                self.state.pop(name, None)
            return
        mode = self.mode_combo.currentText()
        self.state["_reg_yes"] = True
        self.state["sparse_mode"] = mode
        # Clear other-mode keys, then write current
        all_keys = {n for items in SPARSE_MODES.values() for n, _, _ in items}
        for k in all_keys:
            self.state.pop(k, None)
        for name, atype, _ in SPARSE_MODES[mode]:
            w = self._block_inputs[name]
            if isinstance(w, QCheckBox):
                self.state[name] = w.isChecked()
            else:
                self.state[name] = _coerce(w.text(), atype)

    def populate(self) -> None:
        if self.state.get("_reg_yes"):
            self.yes_rb.setChecked(True)
            mode = self.state.get("sparse_mode") or "l1_group"
            idx = self.mode_combo.findText(mode)
            if idx >= 0:
                self.mode_combo.setCurrentIndex(idx)
            self._rebuild_block(self.mode_combo.currentText())
            for name, atype, _ in SPARSE_MODES.get(self.mode_combo.currentText(), []):
                if name in self.state:
                    w = self._block_inputs[name]
                    v = self.state[name]
                    if isinstance(w, QCheckBox):
                        w.setChecked(bool(v))
                    else:
                        w.setText("" if v is None else str(v))
        else:
            self.no_rb.setChecked(True)
        self._refresh()

    def validate(self) -> list[str]:
        return []


class _StepFT(QWidget):
    title = "4. Fine-tuning"

    def __init__(self, state: dict, on_change):
        super().__init__()
        self.state = state
        self.on_change = on_change

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel("<b>Fine-tuning</b>"))

        form = QFormLayout()
        self.epochs_ft = QLineEdit("10")
        form.addRow("--epochs_ft:", self.epochs_ft)
        self.lr_ft = QLineEdit("1.5e-5")
        form.addRow("--lr_ft:", self.lr_ft)
        self.opt_ft = QComboBox()
        self.opt_ft.addItems(["adamw", "sgd"])
        self.opt_ft.currentTextChanged.connect(self._refresh_opt)
        form.addRow("--opt_ft:", self.opt_ft)
        self.momentum_ft = QLineEdit("0.9")
        self.momentum_label = QLabel("--momentum_ft:")
        form.addRow(self.momentum_label, self.momentum_ft)
        self.wd_ft = QLineEdit()
        self.wd_ft.setPlaceholderText("(blank = auto)")
        form.addRow("--wd_ft:", self.wd_ft)
        outer.addLayout(form)

        outer.addWidget(_hsep())

        # PAT block
        self.pat_cb = QCheckBox("Iterative PAT?")
        self.pat_cb.toggled.connect(self._refresh_pat)
        outer.addWidget(self.pat_cb)
        self.pat_box = QGroupBox("PAT params")
        pat_form = QFormLayout(self.pat_box)
        self.pat_steps = QLineEdit("5")
        pat_form.addRow("--pat_steps:", self.pat_steps)
        self.pat_eps = QLineEdit("3")
        pat_form.addRow("--pat_epochs_per_step:", self.pat_eps)
        self.var_loss_w = QLineEdit("0.0")
        pat_form.addRow("--var_loss_weight:", self.var_loss_w)
        self.reparam_during_pat_cb = QCheckBox("--reparam_during_pat")
        pat_form.addRow(self.reparam_during_pat_cb)
        outer.addWidget(self.pat_box)

        # KD block
        self.kd_cb = QCheckBox("Use KD?")
        self.kd_cb.toggled.connect(self._refresh_kd)
        outer.addWidget(self.kd_cb)
        self.kd_box = QGroupBox("KD params")
        kd_form = QFormLayout(self.kd_box)
        self.kd_alpha = QLineEdit("0.7")
        kd_form.addRow("--kd_alpha:", self.kd_alpha)
        self.kd_T = QLineEdit("2.0")
        kd_form.addRow("--kd_T:", self.kd_T)
        outer.addWidget(self.kd_box)
        outer.addStretch(1)

        self._refresh_opt(self.opt_ft.currentText())
        self._refresh_pat(False)
        self._refresh_kd(False)

    def _refresh_opt(self, v: str):
        is_sgd = v == "sgd"
        self.momentum_ft.setVisible(is_sgd)
        self.momentum_label.setVisible(is_sgd)

    def _refresh_pat(self, on: bool):
        self.pat_box.setVisible(on)

    def _refresh_kd(self, on: bool):
        self.kd_box.setVisible(on)

    def apply(self) -> None:
        self.state["epochs_ft"] = _coerce(self.epochs_ft.text(), "int")
        self.state["lr_ft"] = _coerce(self.lr_ft.text(), "float")
        self.state["opt_ft"] = self.opt_ft.currentText()
        if self.opt_ft.currentText() == "sgd":
            self.state["momentum_ft"] = _coerce(self.momentum_ft.text(), "float")
        else:
            self.state["momentum_ft"] = None
        self.state["wd_ft"] = _coerce(self.wd_ft.text(), "float")

        if self.pat_cb.isChecked():
            self.state["pat"] = True
            self.state["pat_steps"] = _coerce(self.pat_steps.text(), "int")
            self.state["pat_epochs_per_step"] = _coerce(self.pat_eps.text(), "int")
            self.state["var_loss_weight"] = _coerce(self.var_loss_w.text(), "float")
            self.state["reparam_during_pat"] = self.reparam_during_pat_cb.isChecked()
        else:
            for k in ("pat", "pat_steps", "pat_epochs_per_step",
                      "var_loss_weight", "reparam_during_pat"):
                self.state.pop(k, None)

        if self.kd_cb.isChecked():
            self.state["use_kd"] = True
            self.state["kd_alpha"] = _coerce(self.kd_alpha.text(), "float")
            self.state["kd_T"] = _coerce(self.kd_T.text(), "float")
        else:
            for k in ("use_kd", "kd_alpha", "kd_T"):
                self.state.pop(k, None)

    def populate(self) -> None:
        if self.state.get("epochs_ft") is not None:
            self.epochs_ft.setText(str(self.state["epochs_ft"]))
        if self.state.get("lr_ft") is not None:
            self.lr_ft.setText(str(self.state["lr_ft"]))
        opt = self.state.get("opt_ft", "adamw")
        idx = self.opt_ft.findText(opt)
        if idx >= 0:
            self.opt_ft.setCurrentIndex(idx)
        if self.state.get("momentum_ft") is not None:
            self.momentum_ft.setText(str(self.state["momentum_ft"]))
        wd = self.state.get("wd_ft")
        self.wd_ft.setText("" if wd is None else str(wd))

        self.pat_cb.setChecked(bool(self.state.get("pat")))
        if self.state.get("pat_steps") is not None:
            self.pat_steps.setText(str(self.state["pat_steps"]))
        if self.state.get("pat_epochs_per_step") is not None:
            self.pat_eps.setText(str(self.state["pat_epochs_per_step"]))
        if self.state.get("var_loss_weight") is not None:
            self.var_loss_w.setText(str(self.state["var_loss_weight"]))
        self.reparam_during_pat_cb.setChecked(
            bool(self.state.get("reparam_during_pat")))

        self.kd_cb.setChecked(bool(self.state.get("use_kd")))
        if self.state.get("kd_alpha") is not None:
            self.kd_alpha.setText(str(self.state["kd_alpha"]))
        if self.state.get("kd_T") is not None:
            self.kd_T.setText(str(self.state["kd_T"]))

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.pat_cb.isChecked():
            ps = _coerce(self.pat_steps.text(), "int")
            pe = _coerce(self.pat_eps.text(), "int")
            if ps is None or ps < 1:
                errs.append("pat_steps must be ≥ 1.")
            if pe is None or pe < 0:
                errs.append("pat_epochs_per_step must be ≥ 0.")
        return errs


class _StepFinal(QWidget):
    title = "Review & Run"

    def __init__(self, state: dict, get_script):
        super().__init__()
        self.state = state
        self.get_script = get_script

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel("<b>Final command</b>"))
        self.cmd_view = QPlainTextEdit()
        self.cmd_view.setReadOnly(True)
        self.cmd_view.setFont(QFont("Menlo", 10))
        self.cmd_view.setMaximumHeight(140)
        outer.addWidget(self.cmd_view)

        btn_row = QHBoxLayout()
        self.btn_copy = QPushButton("Copy")
        self.btn_copy.clicked.connect(self._on_copy)
        btn_row.addWidget(self.btn_copy)
        self.btn_run = QPushButton("Run (blocking)")
        btn_row.addWidget(self.btn_run)
        self.btn_save = QPushButton("Save preset")
        btn_row.addWidget(self.btn_save)
        self.btn_load = QPushButton("Load preset")
        btn_row.addWidget(self.btn_load)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        outer.addWidget(QLabel("<b>Run log</b>"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Menlo", 10))
        outer.addWidget(self.log, stretch=1)

    def _on_copy(self):
        from PyQt5.QtWidgets import QApplication
        QApplication.clipboard().setText(self.cmd_view.toPlainText())

    def populate(self) -> None:
        cmd = build_command(self.state, self.get_script())
        self.cmd_view.setPlainText(" ".join(shlex.quote(p) for p in cmd))

    def apply(self) -> None:
        # Final page is read-only
        return

    def validate(self) -> list[str]:
        return []


# ── Main wizard widget ───────────────────────────────────────────────────


class VBPWizardScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state: dict[str, Any] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        # Top: script path + breadcrumb
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Script:"))
        self.script_in = QLineEdit(get_torch_pruning_script())
        self.script_in.editingFinished.connect(
            lambda: save_torch_pruning_script(self.script_in.text().strip()))
        path_row.addWidget(self.script_in, stretch=1)
        btn = QPushButton("…")
        btn.setFixedWidth(28)
        btn.clicked.connect(self._pick_script)
        path_row.addWidget(btn)
        outer.addLayout(path_row)

        self.crumb = QLabel()
        outer.addWidget(self.crumb)

        # Stack
        self.stack = QStackedWidget()
        self.steps: list[QWidget] = [
            _StepArch(self.state, self._on_state_change),
            _StepCriterion(self.state, self._on_state_change),
            _StepReg(self.state, self._on_state_change),
            _StepFT(self.state, self._on_state_change),
            _StepFinal(self.state, lambda: self.script_in.text().strip()),
        ]
        for w in self.steps:
            self.stack.addWidget(w)
        outer.addWidget(self.stack, stretch=1)

        # Nav
        nav = QHBoxLayout()
        self.btn_back = QPushButton("Back")
        self.btn_back.clicked.connect(self._go_back)
        nav.addWidget(self.btn_back)
        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self._go_next)
        nav.addWidget(self.btn_next)
        nav.addStretch(1)
        outer.addLayout(nav)

        # Wire final page buttons (need handlers on this widget)
        final: _StepFinal = self.steps[-1]
        final.btn_run.clicked.connect(self._run_now)
        final.btn_save.clicked.connect(self._save_preset)
        final.btn_load.clicked.connect(self._load_preset)

        self.stack.setCurrentIndex(0)
        self._update_nav()

    # ── Navigation ────────────────────────────────────────────────────
    def _update_nav(self):
        idx = self.stack.currentIndex()
        last = self.stack.count() - 1
        self.btn_back.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < last)
        if idx >= last:
            self.btn_next.setText("Done")
        else:
            self.btn_next.setText("Next")
        crumbs = " → ".join(
            f"<b>{w.title}</b>" if i == idx else w.title
            for i, w in enumerate(self.steps)
        )
        self.crumb.setText(crumbs)

    def _go_next(self):
        idx = self.stack.currentIndex()
        cur: Any = self.steps[idx]
        cur.apply()
        errs = cur.validate()
        if errs:
            QMessageBox.warning(self, "Validation", "\n".join(errs))
        if idx < self.stack.count() - 1:
            nxt: Any = self.steps[idx + 1]
            nxt.populate()
            self.stack.setCurrentIndex(idx + 1)
            self._update_nav()

    def _go_back(self):
        idx = self.stack.currentIndex()
        if idx == 0:
            return
        cur: Any = self.steps[idx]
        cur.apply()
        prev: Any = self.steps[idx - 1]
        prev.populate()
        self.stack.setCurrentIndex(idx - 1)
        self._update_nav()

    def _on_state_change(self):
        # Currently unused; reserved for cross-step reactions.
        pass

    # ── Actions ───────────────────────────────────────────────────────
    def _pick_script(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select vbp_imagenet_pat.py", os.path.dirname(self.script_in.text()),
            "Python (*.py);;All (*)",
        )
        if path:
            self.script_in.setText(path)
            save_torch_pruning_script(path)

    def _run_now(self):
        final: _StepFinal = self.steps[-1]
        cmd = build_command(self.state, self.script_in.text().strip())
        confirm = QMessageBox.question(
            self, "Run?",
            "Running blocks the UI until the process exits.\n\n"
            f"$ {' '.join(shlex.quote(p) for p in cmd)}\n\nProceed?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        repo = _repo_root(self.script_in.text().strip())
        log_dir = os.path.join(repo, "logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        arch = self.state.get("_arch_choice", "arch")
        crit = self.state.get("_crit_choice", "crit")
        log_path = os.path.join(log_dir, f"{arch}_{crit}_{ts}.log")

        final.log.appendPlainText(f"$ {' '.join(shlex.quote(p) for p in cmd)}")
        final.log.appendPlainText(f"[log: {log_path}]")
        final.log.repaint()
        try:
            with open(log_path, "w") as f:
                f.write("$ " + " ".join(shlex.quote(p) for p in cmd) + "\n")
                proc = subprocess.run(cmd, cwd=repo, stdout=f,
                                      stderr=subprocess.STDOUT)
            final.log.appendPlainText(f"-> rc={proc.returncode}")
        except FileNotFoundError as e:
            final.log.appendPlainText(f"[error: {e}]")
        except Exception as e:
            final.log.appendPlainText(f"[error: {e}]")

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Name:")
        if not ok or not name.strip():
            return
        # Snapshot whatever the current step has into state
        idx = self.stack.currentIndex()
        cur: Any = self.steps[idx]
        cur.apply()
        os.makedirs(PRESET_DIR, exist_ok=True)
        path = os.path.join(PRESET_DIR, f"{name.strip()}.json")
        if os.path.exists(path):
            confirm = QMessageBox.question(
                self, "Overwrite?", f"{path} exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return
        with open(path, "w") as f:
            json.dump(self.state, f, indent=2)
        QMessageBox.information(self, "Saved", f"Preset saved: {path}")

    def _load_preset(self):
        os.makedirs(PRESET_DIR, exist_ok=True)
        files = sorted(f for f in os.listdir(PRESET_DIR) if f.endswith(".json"))
        if not files:
            QMessageBox.information(self, "No presets", f"No presets in {PRESET_DIR}")
            return
        choice, ok = QInputDialog.getItem(
            self, "Load preset", "Pick:", files, 0, False,
        )
        if not ok:
            return
        path = os.path.join(PRESET_DIR, choice)
        try:
            with open(path) as f:
                loaded = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        self.state.clear()
        self.state.update(loaded)
        for w in self.steps:
            if hasattr(w, "populate"):
                w.populate()
        # Refresh final preview
        final: _StepFinal = self.steps[-1]
        final.populate()
