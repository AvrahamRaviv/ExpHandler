"""VBP Run Wizard.

4-step state machine that builds a python invocation of vbp_imagenet_pat.py.
Flag names mirror that script's argparse (paste 2026-05-07).

    1. Architecture        (model_type / cnn_arch / model_name + arch extras)
    2. Criterion           (criterion + importance_mode + extras)
    3. Regularization      (sparse_mode block)
    4. Fine-tuning / PAT / KD

Final page shows the assembled command, with Copy / Run / Save preset / Load.
Run mode is blocking subprocess.run; output mirrored to
<repo>/logs/<arch>_<criterion>_<timestamp>.log.
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
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QRadioButton, QStackedWidget, QVBoxLayout,
    QWidget,
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
                 "model_name": "/algo/NetOptimization/outputs/VBP/DeiT_tiny",
                 "interior_default": True},
}

CRITERIA: dict[str, dict] = {
    "magnitude": {"criterion": "magnitude", "importance_mode": None,
                  "extras": []},
    "VBP":       {"criterion": "variance",  "importance_mode": "variance",
                  "extras": ["no_compensation", "norm_per_layer",
                             "similarity_discount"]},
    "tp_var":    {"criterion": "variance",  "importance_mode": "tp_variance",
                  "extras": ["no_compensation", "norm_per_layer",
                             "similarity_discount"]},
}

GROUP_REDUCTIONS = ["mean", "sum", "prod", "max", "first", "dw_proj", "ww"]
IMPORTANCE_MODES = ["variance", "weight_variance", "weight_variance_both",
                    "combined", "rank_fusion", "mag_guided", "tp_variance",
                    "dw_proj_var"]
WV_BASE_MODES = ["variance", "weight_variance", "weight_variance_both"]
PRUNING_SCHEDULES = ["geometric", "linear"]

# (name, type, default)
SPARSE_MODES: dict[str, list[tuple[str, str, Any]]] = {
    "l1_group":   [("epochs_sparse", "int",   5),
                   ("l1_lambda",     "float", 1e-4)],
    "gmp":        [("epochs_sparse", "int",   5),
                   ("gmp_target_sparsity", "float", 0.5)],
    "reparam":    [("epochs_sparse", "int",   5),
                   ("reparam_lambda",     "float", 0.01),
                   ("reparam_refresh_interval", "int", 1),
                   ("reparam_normalize",  "bool",  False),
                   ("reparam_target",     "str",   "fc2")],
    "vnr":        [("epochs_sparse",          "int",   5),
                   ("reparam_lambda",         "float", 0.01),
                   ("reparam_entropy_lambda", "float", 0.0)],
    "group_norm": [("epochs_sparse", "int",   5),
                   ("reg",           "float", 1e-4)],
}

# Always-on args shown in the "Advanced" panel on Step 1.
ALWAYS_ON: list[tuple[str, str, Any]] = [
    ("data_path",        "str",   "/algo/NetOptimization/outputs/VBP/"),
    ("save_dir",         "str",   "./output/vbp_pat"),
    ("keep_ratio",       "float", 0.65),
    ("max_batches",      "int",   200),
    ("disable_ddp",      "bool",  False),
    ("train_batch_size", "int",   64),
    ("val_batch_size",   "int",   128),
    ("num_workers",      "int",   4),
]

FLAG_ORDER: list[str] = [
    # Data + I/O
    "data_path", "save_dir", "keep_ratio", "max_batches", "disable_ddp",
    "train_batch_size", "val_batch_size", "num_workers",
    # Arch
    "model_type", "cnn_arch", "model_name",
    "interior_only", "max_pruning_rate", "global_pruning", "isomorphic",
    "mac_target", "bn_recalibration", "bn_recalib_batches",
    "fold_bn_init", "fold_bn_before_prune", "checkpoint",
    # Criterion
    "criterion", "importance_mode", "group_reduction",
    "no_compensation", "norm_per_layer", "similarity_discount",
    "normalize_importance", "alpha", "wv_base_mode", "mag_guided_delta",
    # Regularization
    "sparse_mode", "epochs_sparse", "l1_lambda",
    "gmp_target_sparsity",
    "reparam_lambda", "reparam_refresh_interval", "reparam_normalize",
    "reparam_target", "reparam_entropy_lambda",
    "reg",
    # FT / PAT schedule
    "pat_steps", "pat_epochs_per_step", "epochs_ft",
    "ft_warmup_epochs", "ft_eta_min",
    "lr", "ft_lr", "opt", "wd",
    "var_loss_weight", "reparam_during_pat",
    "pruning_schedule", "no_mask_only",
    # KD
    "use_kd", "kd_alpha", "kd_T",
]

BOOL_FLAGS: set[str] = {
    "disable_ddp",
    "interior_only", "global_pruning", "isomorphic",
    "mac_target", "bn_recalibration", "fold_bn_init", "fold_bn_before_prune",
    "no_compensation", "norm_per_layer", "similarity_discount",
    "normalize_importance",
    "reparam_normalize", "reparam_during_pat",
    "no_mask_only", "use_kd",
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
    p = os.path.abspath(script_path)
    cur = os.path.dirname(p)
    seen: set = set()
    while cur and cur != "/" and cur not in seen:
        if os.path.isdir(os.path.join(cur, ".git")):
            return cur
        seen.add(cur)
        cur = os.path.dirname(cur)
    return os.path.dirname(os.path.dirname(os.path.dirname(p)))


def _hsep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


# ── Step 1: Architecture ─────────────────────────────────────────────────


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

        # Pruning level radio (3-way: local / global / isomorphic)
        outer.addWidget(QLabel("<b>Pruning level</b>"))
        lvl_row = QHBoxLayout()
        self.lvl_group = QButtonGroup(self)
        self.lvl_local = QRadioButton("local")
        self.lvl_global = QRadioButton("global")
        self.lvl_iso = QRadioButton("isomorphic")
        for rb in (self.lvl_local, self.lvl_global, self.lvl_iso):
            self.lvl_group.addButton(rb)
            lvl_row.addWidget(rb)
        self.lvl_local.setChecked(True)
        lvl_row.addStretch(1)
        outer.addLayout(lvl_row)

        # Arch-extra fields
        form = QFormLayout()
        self.model_name_in = QLineEdit()
        self.model_name_in.setPlaceholderText("(model_name override)")
        form.addRow("--model_name:", self.model_name_in)

        self.interior_only_cb = QCheckBox("--interior_only")
        form.addRow(self.interior_only_cb)

        self.max_pr_in = QLineEdit("0.95")
        form.addRow("--max_pruning_rate:", self.max_pr_in)

        self.mac_target_cb = QCheckBox("--mac_target")
        form.addRow(self.mac_target_cb)

        # CNN-only knobs
        self.bn_recalib_cb = QCheckBox("--bn_recalibration")
        form.addRow(self.bn_recalib_cb)
        self.bn_recalib_batches_in = QLineEdit("100")
        self.bn_recalib_batches_label = QLabel("--bn_recalib_batches:")
        form.addRow(self.bn_recalib_batches_label, self.bn_recalib_batches_in)
        self.fold_bn_init_cb = QCheckBox("--fold_bn_init")
        form.addRow(self.fold_bn_init_cb)
        self.fold_bn_before_prune_cb = QCheckBox("--fold_bn_before_prune")
        form.addRow(self.fold_bn_before_prune_cb)

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

    def _is_cnn(self) -> bool:
        for name, rb in self.arch_radios.items():
            if rb.isChecked():
                return ARCHS[name]["model_type"] == "cnn"
        return False

    def _refresh_cnn_only(self):
        on = self._is_cnn()
        for w in (self.bn_recalib_cb, self.bn_recalib_batches_in,
                  self.bn_recalib_batches_label,
                  self.fold_bn_init_cb, self.fold_bn_before_prune_cb):
            w.setVisible(on)

    def _on_arch_toggled(self, checked: bool):
        if not checked:
            return
        for name, rb in self.arch_radios.items():
            if rb.isChecked():
                self.interior_only_cb.setChecked(ARCHS[name]["interior_default"])
                ship = ARCHS[name]["model_name"]
                if ship is not None and not self.model_name_in.text():
                    self.model_name_in.setText(ship)
                break
        self._refresh_cnn_only()

    def apply(self) -> None:
        for name, rb in self.arch_radios.items():
            if rb.isChecked():
                cfg = ARCHS[name]
                self.state["_arch_choice"] = name
                self.state["model_type"] = cfg["model_type"]
                if cfg["model_type"] == "cnn":
                    self.state["cnn_arch"] = cfg["cnn_arch"]
                    self.state["model_name"] = None
                else:
                    self.state["cnn_arch"] = None
                    name_text = self.model_name_in.text().strip()
                    self.state["model_name"] = name_text or cfg["model_name"]
                break

        # Pruning level
        self.state["global_pruning"] = self.lvl_global.isChecked()
        self.state["isomorphic"] = self.lvl_iso.isChecked()
        self.state["_pruning_level"] = (
            "global" if self.lvl_global.isChecked()
            else ("isomorphic" if self.lvl_iso.isChecked() else "local")
        )

        self.state["interior_only"] = self.interior_only_cb.isChecked()
        self.state["max_pruning_rate"] = _coerce(self.max_pr_in.text(), "float")
        self.state["mac_target"] = self.mac_target_cb.isChecked()

        if self._is_cnn():
            self.state["bn_recalibration"] = self.bn_recalib_cb.isChecked()
            self.state["bn_recalib_batches"] = _coerce(
                self.bn_recalib_batches_in.text(), "int")
            self.state["fold_bn_init"] = self.fold_bn_init_cb.isChecked()
            self.state["fold_bn_before_prune"] = \
                self.fold_bn_before_prune_cb.isChecked()
        else:
            for k in ("bn_recalibration", "bn_recalib_batches",
                      "fold_bn_init", "fold_bn_before_prune"):
                self.state.pop(k, None)

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

        lvl = self.state.get("_pruning_level", "local")
        if lvl == "global":
            self.lvl_global.setChecked(True)
        elif lvl == "isomorphic":
            self.lvl_iso.setChecked(True)
        else:
            self.lvl_local.setChecked(True)

        if "interior_only" in self.state:
            self.interior_only_cb.setChecked(bool(self.state["interior_only"]))
        if self.state.get("max_pruning_rate") is not None:
            self.max_pr_in.setText(str(self.state["max_pruning_rate"]))
        self.mac_target_cb.setChecked(bool(self.state.get("mac_target")))
        self.bn_recalib_cb.setChecked(bool(self.state.get("bn_recalibration")))
        if self.state.get("bn_recalib_batches") is not None:
            self.bn_recalib_batches_in.setText(
                str(self.state["bn_recalib_batches"]))
        self.fold_bn_init_cb.setChecked(bool(self.state.get("fold_bn_init")))
        self.fold_bn_before_prune_cb.setChecked(
            bool(self.state.get("fold_bn_before_prune")))

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

        self._refresh_cnn_only()

    def validate(self) -> list[str]:
        errs: list[str] = []
        kr = self.state.get("keep_ratio")
        if kr is None or not (0 < kr <= 1):
            errs.append("keep_ratio must be in (0, 1].")
        ck = self.state.get("checkpoint")
        if ck and not os.path.isfile(ck):
            errs.append(f"checkpoint file not found: {ck} (warning only)")
        return errs


# ── Step 2: Criterion ────────────────────────────────────────────────────


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
            rb.toggled.connect(self._refresh)
            self.crit_group.addButton(rb)
            crit_row.addWidget(rb)
            self.crit_radios[name] = rb
        crit_row.addStretch(1)
        outer.addLayout(crit_row)

        # group_reduction (magnitude only)
        gr_row = QHBoxLayout()
        self.gr_label = QLabel("--group_reduction:")
        self.gr_combo = QComboBox()
        self.gr_combo.addItems(GROUP_REDUCTIONS)
        gr_row.addWidget(self.gr_label)
        gr_row.addWidget(self.gr_combo)
        gr_row.addStretch(1)
        outer.addLayout(gr_row)

        outer.addWidget(_hsep())
        outer.addWidget(QLabel("<b>Extras</b>"))
        self.extras_form = QFormLayout()
        self.extra_inputs: dict[str, QCheckBox] = {}
        all_extras = sorted(
            {e for c in CRITERIA.values() for e in c["extras"]}
        )
        for ex in all_extras:
            cb = QCheckBox(f"--{ex}")
            self.extras_form.addRow(cb)
            self.extra_inputs[ex] = cb
        outer.addLayout(self.extras_form)

        # Advanced (importance_mode override + ranking blend params)
        outer.addWidget(_hsep())
        self.adv_toggle = QCheckBox("Show advanced (importance_mode / blends)")
        self.adv_toggle.toggled.connect(self._refresh_adv)
        outer.addWidget(self.adv_toggle)

        self.adv_box = QGroupBox("Advanced")
        adv = QFormLayout(self.adv_box)
        self.imp_combo = QComboBox()
        self.imp_combo.addItem("(default for criterion)")
        self.imp_combo.addItems(IMPORTANCE_MODES)
        adv.addRow("--importance_mode:", self.imp_combo)
        self.alpha_in = QLineEdit("0.5")
        adv.addRow("--alpha:", self.alpha_in)
        self.wv_base_combo = QComboBox()
        self.wv_base_combo.addItems(WV_BASE_MODES)
        adv.addRow("--wv_base_mode:", self.wv_base_combo)
        self.mag_delta_in = QLineEdit("0.2")
        adv.addRow("--mag_guided_delta:", self.mag_delta_in)
        self.norm_imp_cb = QCheckBox("--normalize_importance")
        adv.addRow(self.norm_imp_cb)
        outer.addWidget(self.adv_box)
        outer.addStretch(1)

        self.crit_radios["VBP"].setChecked(True)
        self._refresh()
        self._refresh_adv(False)

    def _current_choice(self) -> str | None:
        for name, rb in self.crit_radios.items():
            if rb.isChecked():
                return name
        return None

    def _refresh(self):
        choice = self._current_choice()
        if not choice:
            return
        # group_reduction visible only for magnitude
        is_mag = choice == "magnitude"
        self.gr_label.setVisible(is_mag)
        self.gr_combo.setVisible(is_mag)
        active = set(CRITERIA[choice]["extras"])
        for ex, cb in self.extra_inputs.items():
            cb.setVisible(ex in active)

    def _refresh_adv(self, on: bool):
        self.adv_box.setVisible(bool(on))

    def apply(self) -> None:
        choice = self._current_choice()
        if choice is None:
            return
        cfg = CRITERIA[choice]
        self.state["_crit_choice"] = choice
        self.state["criterion"] = cfg["criterion"]

        # Advanced: importance_mode override (if not "(default…)")
        adv_mode = self.imp_combo.currentText()
        if self.adv_toggle.isChecked() and adv_mode in IMPORTANCE_MODES:
            self.state["importance_mode"] = adv_mode
        else:
            self.state["importance_mode"] = cfg["importance_mode"]

        # group_reduction only for magnitude
        if choice == "magnitude":
            self.state["group_reduction"] = self.gr_combo.currentText()
        else:
            self.state.pop("group_reduction", None)

        # Extras
        for ex in self.extra_inputs:
            if ex in cfg["extras"]:
                cb = self.extra_inputs[ex]
                self.state[ex] = cb.isChecked() if cb.isVisible() else False
            else:
                self.state[ex] = False

        # Advanced numerics
        if self.adv_toggle.isChecked():
            self.state["alpha"] = _coerce(self.alpha_in.text(), "float")
            self.state["wv_base_mode"] = self.wv_base_combo.currentText()
            self.state["mag_guided_delta"] = _coerce(
                self.mag_delta_in.text(), "float")
            self.state["normalize_importance"] = self.norm_imp_cb.isChecked()
        else:
            for k in ("alpha", "wv_base_mode", "mag_guided_delta",
                      "normalize_importance"):
                self.state.pop(k, None)

    def populate(self) -> None:
        choice = self.state.get("_crit_choice")
        if choice in self.crit_radios:
            self.crit_radios[choice].setChecked(True)
        if "group_reduction" in self.state:
            idx = self.gr_combo.findText(self.state["group_reduction"])
            if idx >= 0:
                self.gr_combo.setCurrentIndex(idx)
        for ex, cb in self.extra_inputs.items():
            cb.setChecked(bool(self.state.get(ex, False)))
        if any(k in self.state for k in
               ("alpha", "wv_base_mode", "mag_guided_delta",
                "normalize_importance")):
            self.adv_toggle.setChecked(True)
            if self.state.get("importance_mode") in IMPORTANCE_MODES:
                idx = self.imp_combo.findText(self.state["importance_mode"])
                if idx >= 0:
                    self.imp_combo.setCurrentIndex(idx)
            if self.state.get("alpha") is not None:
                self.alpha_in.setText(str(self.state["alpha"]))
            if self.state.get("wv_base_mode") in WV_BASE_MODES:
                idx = self.wv_base_combo.findText(self.state["wv_base_mode"])
                if idx >= 0:
                    self.wv_base_combo.setCurrentIndex(idx)
            if self.state.get("mag_guided_delta") is not None:
                self.mag_delta_in.setText(str(self.state["mag_guided_delta"]))
            self.norm_imp_cb.setChecked(
                bool(self.state.get("normalize_importance")))
        self._refresh()
        self._refresh_adv(self.adv_toggle.isChecked())

    def validate(self) -> list[str]:
        return []


# ── Step 3: Regularization ───────────────────────────────────────────────


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
        all_keys = {n for items in SPARSE_MODES.values() for n, _, _ in items}
        if self.no_rb.isChecked():
            self.state["sparse_mode"] = "none"
            self.state["_reg_yes"] = False
            for k in all_keys:
                self.state.pop(k, None)
            return
        mode = self.mode_combo.currentText()
        self.state["_reg_yes"] = True
        self.state["sparse_mode"] = mode
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
            for name, atype, _ in SPARSE_MODES.get(
                    self.mode_combo.currentText(), []):
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


# ── Step 4: Fine-tuning / PAT / KD ───────────────────────────────────────


class _StepFT(QWidget):
    title = "4. Fine-tuning"

    def __init__(self, state: dict, on_change):
        super().__init__()
        self.state = state
        self.on_change = on_change

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel("<b>Optimizer / FT loop</b>"))

        form = QFormLayout()
        self.lr_in = QLineEdit("1.5e-5")
        form.addRow("--lr:", self.lr_in)
        self.ft_lr_in = QLineEdit()
        self.ft_lr_in.setPlaceholderText("(blank = use --lr)")
        form.addRow("--ft_lr:", self.ft_lr_in)
        self.opt_combo = QComboBox()
        self.opt_combo.addItems(["adamw", "sgd"])
        form.addRow("--opt:", self.opt_combo)
        self.wd_in = QLineEdit("0.01")
        form.addRow("--wd:", self.wd_in)

        self.epochs_ft = QLineEdit("10")
        form.addRow("--epochs_ft:", self.epochs_ft)
        self.ft_warmup_in = QLineEdit("0")
        form.addRow("--ft_warmup_epochs:", self.ft_warmup_in)
        self.ft_eta_min_in = QLineEdit("1e-8")
        form.addRow("--ft_eta_min:", self.ft_eta_min_in)

        self.sched_combo = QComboBox()
        self.sched_combo.addItems(PRUNING_SCHEDULES)
        form.addRow("--pruning_schedule:", self.sched_combo)
        self.no_mask_only_cb = QCheckBox("--no_mask_only")
        form.addRow(self.no_mask_only_cb)

        outer.addLayout(form)
        outer.addWidget(_hsep())

        # PAT block
        outer.addWidget(QLabel("<b>PAT schedule</b>"))
        pat_form = QFormLayout()
        self.pat_steps = QLineEdit("1")
        pat_form.addRow("--pat_steps:", self.pat_steps)
        self.pat_eps = QLineEdit("0")
        pat_form.addRow("--pat_epochs_per_step:", self.pat_eps)
        self.var_loss_w = QLineEdit("0.0")
        pat_form.addRow("--var_loss_weight:", self.var_loss_w)
        self.reparam_during_pat_cb = QCheckBox("--reparam_during_pat")
        pat_form.addRow(self.reparam_during_pat_cb)
        outer.addLayout(pat_form)
        outer.addWidget(_hsep())

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

        self._refresh_kd(False)

    def _refresh_kd(self, on: bool):
        self.kd_box.setVisible(on)

    def apply(self) -> None:
        self.state["lr"] = _coerce(self.lr_in.text(), "float")
        ft_lr = _coerce(self.ft_lr_in.text(), "float")
        if ft_lr is None:
            self.state.pop("ft_lr", None)
        else:
            self.state["ft_lr"] = ft_lr
        self.state["opt"] = self.opt_combo.currentText()
        self.state["wd"] = _coerce(self.wd_in.text(), "float")

        self.state["epochs_ft"] = _coerce(self.epochs_ft.text(), "int")
        self.state["ft_warmup_epochs"] = _coerce(
            self.ft_warmup_in.text(), "float")
        self.state["ft_eta_min"] = _coerce(self.ft_eta_min_in.text(), "float")
        self.state["pruning_schedule"] = self.sched_combo.currentText()
        self.state["no_mask_only"] = self.no_mask_only_cb.isChecked()

        self.state["pat_steps"] = _coerce(self.pat_steps.text(), "int")
        self.state["pat_epochs_per_step"] = _coerce(
            self.pat_eps.text(), "int")
        self.state["var_loss_weight"] = _coerce(self.var_loss_w.text(), "float")
        self.state["reparam_during_pat"] = \
            self.reparam_during_pat_cb.isChecked()

        if self.kd_cb.isChecked():
            self.state["use_kd"] = True
            self.state["kd_alpha"] = _coerce(self.kd_alpha.text(), "float")
            self.state["kd_T"] = _coerce(self.kd_T.text(), "float")
        else:
            for k in ("use_kd", "kd_alpha", "kd_T"):
                self.state.pop(k, None)

    def populate(self) -> None:
        if self.state.get("lr") is not None:
            self.lr_in.setText(str(self.state["lr"]))
        if self.state.get("ft_lr") is not None:
            self.ft_lr_in.setText(str(self.state["ft_lr"]))
        opt = self.state.get("opt", "adamw")
        idx = self.opt_combo.findText(opt)
        if idx >= 0:
            self.opt_combo.setCurrentIndex(idx)
        if self.state.get("wd") is not None:
            self.wd_in.setText(str(self.state["wd"]))
        if self.state.get("epochs_ft") is not None:
            self.epochs_ft.setText(str(self.state["epochs_ft"]))
        if self.state.get("ft_warmup_epochs") is not None:
            self.ft_warmup_in.setText(str(self.state["ft_warmup_epochs"]))
        if self.state.get("ft_eta_min") is not None:
            self.ft_eta_min_in.setText(str(self.state["ft_eta_min"]))
        sched = self.state.get("pruning_schedule", "geometric")
        idx = self.sched_combo.findText(sched)
        if idx >= 0:
            self.sched_combo.setCurrentIndex(idx)
        self.no_mask_only_cb.setChecked(bool(self.state.get("no_mask_only")))

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
        ps = _coerce(self.pat_steps.text(), "int")
        pe = _coerce(self.pat_eps.text(), "int")
        if ps is None or ps < 1:
            errs.append("pat_steps must be ≥ 1.")
        if pe is None or pe < 0:
            errs.append("pat_epochs_per_step must be ≥ 0.")
        return errs


# ── Final review page ───────────────────────────────────────────────────


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

        nav = QHBoxLayout()
        self.btn_back = QPushButton("Back")
        self.btn_back.clicked.connect(self._go_back)
        nav.addWidget(self.btn_back)
        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self._go_next)
        nav.addWidget(self.btn_next)
        nav.addStretch(1)
        outer.addLayout(nav)

        final: _StepFinal = self.steps[-1]
        final.btn_run.clicked.connect(self._run_now)
        final.btn_save.clicked.connect(self._save_preset)
        final.btn_load.clicked.connect(self._load_preset)

        self.stack.setCurrentIndex(0)
        self._update_nav()

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
        pass

    def _pick_script(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select vbp_imagenet_pat.py",
            os.path.dirname(self.script_in.text()),
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

        final.log.appendPlainText(
            f"$ {' '.join(shlex.quote(p) for p in cmd)}")
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
            QMessageBox.information(
                self, "No presets", f"No presets in {PRESET_DIR}")
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
        final: _StepFinal = self.steps[-1]
        final.populate()
