"""VBP Run Wizard.

Single-page sectioned form (4 QGroupBoxes — Architecture / Criterion /
Regularization / Fine-tuning) for building a vbp_imagenet_pat.py invocation.
Below the form: live command preview + Run (blocking, dispatched via
run_docker_gpu.sh wrapper, same as the per-subtype Launcher) + Save/Load .sh.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QRadioButton, QScrollArea, QVBoxLayout, QWidget,
)

from config import get_torch_pruning_script, save_torch_pruning_script
from screens.launcher import _parse_sh


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
    "data_path", "save_dir", "keep_ratio", "max_batches", "disable_ddp",
    "train_batch_size", "val_batch_size", "num_workers",
    "model_type", "cnn_arch", "model_name",
    "interior_only", "max_pruning_rate", "global_pruning", "isomorphic",
    "mac_target", "bn_recalibration", "bn_recalib_batches",
    "fold_bn_init", "fold_bn_before_prune", "checkpoint",
    "criterion", "importance_mode", "group_reduction",
    "no_compensation", "norm_per_layer", "similarity_discount",
    "normalize_importance", "alpha", "wv_base_mode", "mag_guided_delta",
    "sparse_mode", "epochs_sparse", "l1_lambda",
    "gmp_target_sparsity",
    "reparam_lambda", "reparam_refresh_interval", "reparam_normalize",
    "reparam_target", "reparam_entropy_lambda",
    "reg",
    "pat_steps", "pat_epochs_per_step", "epochs_ft",
    "ft_warmup_epochs", "ft_eta_min",
    "lr", "ft_lr", "opt", "wd",
    "var_loss_weight", "reparam_during_pat",
    "pruning_schedule", "no_mask_only",
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

# Per-flag types for reverse-mapping when loading from .sh.
TYPES: dict[str, str] = {
    "epochs_ft": "int", "pat_steps": "int", "pat_epochs_per_step": "int",
    "epochs_sparse": "int", "max_batches": "int", "train_batch_size": "int",
    "val_batch_size": "int", "num_workers": "int",
    "bn_recalib_batches": "int", "reparam_refresh_interval": "int",
    "lr": "float", "ft_lr": "float", "wd": "float",
    "kd_alpha": "float", "kd_T": "float",
    "ft_warmup_epochs": "float", "ft_eta_min": "float",
    "var_loss_weight": "float", "alpha": "float",
    "mag_guided_delta": "float", "max_pruning_rate": "float",
    "keep_ratio": "float",
    "l1_lambda": "float", "gmp_target_sparsity": "float",
    "reparam_lambda": "float", "reparam_entropy_lambda": "float",
    "reg": "float",
}


# Wrapper for run_docker_gpu.sh dispatch (matches examples/run_ddp.py).
DEFAULT_WRAPPER_A = (
    "/algo/ws/shared/remote-gpu/run_docker_gpu.sh "
    "-d gitlab-srv:4567/od-alg/od_next_gen:v1.7.7_tp2 "
    "-C execute -q gpu_deep_train_low_q -W working_dir -M "
)
DEFAULT_WRAPPER_B = (
    " -s 25gb -n 10 -o 60000 -A '' -p VISION "
    "-v /algo/NetOptimization:/algo/NetOptimization "
    "-R 'select[gpu_hm]' -R 'select[hname != gpusrv11]' "
    "-E force_python_3=yes -x 4"
)


# ── Helpers ──────────────────────────────────────────────────────────────


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


def state_from_parsed(parsed: dict) -> dict:
    """Reverse-map a parsed run_ddp.sh dict to wizard state.

    Resolves the radio-style choices (arch / criterion / pruning level /
    reg yes-no) so each step's populate() can highlight the right option.
    """
    state: dict = {}

    # Architecture
    mt = parsed.get("model_type")
    ca = parsed.get("cnn_arch")
    arch = None
    if mt == "cnn":
        if ca == "mobilenet_v2":
            arch = "MNv2"
        elif ca == "resnet50":
            arch = "RN50"
    elif mt == "convnext":
        arch = "ConvNeXt"
    elif mt == "vit":
        arch = "DeiT-T"
    if arch:
        cfg = ARCHS[arch]
        state["_arch_choice"] = arch
        state["model_type"] = cfg["model_type"]
        state["cnn_arch"] = cfg["cnn_arch"]
        state["model_name"] = parsed.get("model_name") or cfg["model_name"]

    # Pruning level
    iso = parsed.get("isomorphic")
    glb = parsed.get("global_pruning")
    if iso is True or str(iso).lower() in {"true", "1"}:
        state["_pruning_level"] = "isomorphic"
        state["isomorphic"] = True
        state["global_pruning"] = False
    elif glb is True or str(glb).lower() in {"true", "1"}:
        state["_pruning_level"] = "global"
        state["isomorphic"] = False
        state["global_pruning"] = True
    else:
        state["_pruning_level"] = "local"
        state["isomorphic"] = False
        state["global_pruning"] = False

    # Criterion
    crit = parsed.get("criterion", "magnitude")
    imp = parsed.get("importance_mode")
    if crit == "magnitude":
        state["_crit_choice"] = "magnitude"
    elif crit == "variance" and imp == "tp_variance":
        state["_crit_choice"] = "tp_var"
    else:
        state["_crit_choice"] = "VBP"

    # Reg yes/no
    sm = parsed.get("sparse_mode", "none")
    state["_reg_yes"] = (sm != "none")

    # Generic flag passthrough (skip the radios already resolved above).
    skip_keys = {"isomorphic", "global_pruning", "model_type",
                 "cnn_arch", "model_name", "criterion", "importance_mode"}
    for k, v in parsed.items():
        if k in skip_keys:
            continue
        if k in BOOL_FLAGS:
            if v is True:
                state[k] = True
            else:
                state[k] = str(v).lower() in {"true", "1", "yes"}
        else:
            atype = TYPES.get(k, "str")
            raw = "" if v is True else str(v)
            state[k] = _coerce(raw, atype)

    # Allow arch/crit-resolved values to be re-applied if user paths have them
    if "importance_mode" in parsed:
        state["importance_mode"] = parsed["importance_mode"]
    return state


def _hsep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


# ── Step widgets (one per group box) ─────────────────────────────────────


class _StepArch(QWidget):
    title = "1. Architecture"

    def __init__(self, state: dict, on_change):
        super().__init__()
        self.state = state
        self.on_change = on_change

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

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

        # Pruning level radio
        lvl_row = QHBoxLayout()
        lvl_row.addWidget(QLabel("Pruning level:"))
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

        form = QFormLayout()
        form.setSpacing(2)
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
        btn.setFixedWidth(24)
        btn.clicked.connect(self._pick_checkpoint)
        ckpt_row.addWidget(self.checkpoint_in)
        ckpt_row.addWidget(btn)
        form.addRow("--checkpoint:", ckpt_row)
        outer.addLayout(form)

        outer.addWidget(_hsep())
        outer.addWidget(QLabel("<b>Always-on</b>"))
        adv_form = QFormLayout()
        adv_form.setSpacing(2)
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


class _StepCriterion(QWidget):
    title = "2. Criterion"

    def __init__(self, state: dict, on_change):
        super().__init__()
        self.state = state
        self.on_change = on_change

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

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

        gr_row = QHBoxLayout()
        self.gr_label = QLabel("--group_reduction:")
        self.gr_combo = QComboBox()
        self.gr_combo.addItems(GROUP_REDUCTIONS)
        gr_row.addWidget(self.gr_label)
        gr_row.addWidget(self.gr_combo)
        gr_row.addStretch(1)
        outer.addLayout(gr_row)

        self.extras_form = QFormLayout()
        self.extras_form.setSpacing(2)
        self.extra_inputs: dict[str, QCheckBox] = {}
        all_extras = sorted(
            {e for c in CRITERIA.values() for e in c["extras"]}
        )
        for ex in all_extras:
            cb = QCheckBox(f"--{ex}")
            self.extras_form.addRow(cb)
            self.extra_inputs[ex] = cb
        outer.addLayout(self.extras_form)

        outer.addWidget(_hsep())
        self.adv_toggle = QCheckBox(
            "Show advanced (importance_mode / blends)")
        self.adv_toggle.toggled.connect(self._refresh_adv)
        outer.addWidget(self.adv_toggle)

        self.adv_box = QGroupBox("Advanced")
        adv = QFormLayout(self.adv_box)
        adv.setSpacing(2)
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

        adv_mode = self.imp_combo.currentText()
        if self.adv_toggle.isChecked() and adv_mode in IMPORTANCE_MODES:
            self.state["importance_mode"] = adv_mode
        else:
            self.state["importance_mode"] = cfg["importance_mode"]

        if choice == "magnitude":
            self.state["group_reduction"] = self.gr_combo.currentText()
        else:
            self.state.pop("group_reduction", None)

        for ex in self.extra_inputs:
            if ex in cfg["extras"]:
                cb = self.extra_inputs[ex]
                self.state[ex] = cb.isChecked() if cb.isVisible() else False
            else:
                self.state[ex] = False

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
        adv_keys = ("alpha", "wv_base_mode", "mag_guided_delta",
                    "normalize_importance")
        if any(k in self.state for k in adv_keys):
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


class _StepReg(QWidget):
    title = "3. Regularization"

    def __init__(self, state: dict, on_change):
        super().__init__()
        self.state = state
        self.on_change = on_change

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Sparse pre-training:"))
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
        mode_layout.setContentsMargins(6, 6, 6, 6)
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("--sparse_mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(SPARSE_MODES.keys()))
        self.mode_combo.currentTextChanged.connect(self._rebuild_block)
        sel_row.addWidget(self.mode_combo)
        sel_row.addStretch(1)
        mode_layout.addLayout(sel_row)

        self.block_form = QFormLayout()
        self.block_form.setSpacing(2)
        self.block_widget = QWidget()
        self.block_widget.setLayout(self.block_form)
        mode_layout.addWidget(self.block_widget)
        outer.addWidget(self.mode_box)

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


class _StepFT(QWidget):
    title = "4. Fine-tuning"

    def __init__(self, state: dict, on_change):
        super().__init__()
        self.state = state
        self.on_change = on_change

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        form = QFormLayout()
        form.setSpacing(2)
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
        outer.addWidget(QLabel("<b>PAT</b>"))
        pat_form = QFormLayout()
        pat_form.setSpacing(2)
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
        self.kd_cb = QCheckBox("Use KD?")
        self.kd_cb.toggled.connect(self._refresh_kd)
        outer.addWidget(self.kd_cb)
        self.kd_box = QGroupBox("KD params")
        kd_form = QFormLayout(self.kd_box)
        kd_form.setSpacing(2)
        self.kd_alpha = QLineEdit("0.7")
        kd_form.addRow("--kd_alpha:", self.kd_alpha)
        self.kd_T = QLineEdit("2.0")
        kd_form.addRow("--kd_T:", self.kd_T)
        outer.addWidget(self.kd_box)

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
        else:
            self.ft_lr_in.setText("")
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


# ── Main wizard widget ───────────────────────────────────────────────────


class VBPWizardScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state: dict[str, Any] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Script path
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Script:"))
        self.script_in = QLineEdit(get_torch_pruning_script())
        self.script_in.editingFinished.connect(
            lambda: save_torch_pruning_script(self.script_in.text().strip()))
        path_row.addWidget(self.script_in, stretch=1)
        btn = QPushButton("…")
        btn.setFixedWidth(24)
        btn.clicked.connect(self._pick_script)
        path_row.addWidget(btn)
        outer.addLayout(path_row)

        # Scrollable form: 4 group boxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(2, 2, 2, 2)
        body_layout.setSpacing(4)

        self.step_arch = _StepArch(self.state, self._refresh_preview)
        self.step_crit = _StepCriterion(self.state, self._refresh_preview)
        self.step_reg = _StepReg(self.state, self._refresh_preview)
        self.step_ft = _StepFT(self.state, self._refresh_preview)

        for w in (self.step_arch, self.step_crit, self.step_reg, self.step_ft):
            box = QGroupBox(w.title)
            l = QVBoxLayout(box)
            l.setContentsMargins(6, 6, 6, 6)
            l.setSpacing(2)
            l.addWidget(w)
            body_layout.addWidget(box)
        body_layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, stretch=2)

        # Command preview
        outer.addWidget(QLabel("<b>Command preview</b>"))
        self.cmd_view = QPlainTextEdit()
        self.cmd_view.setReadOnly(True)
        self.cmd_view.setFont(QFont("Menlo", 10))
        self.cmd_view.setMaximumHeight(120)
        outer.addWidget(self.cmd_view)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh preview")
        self.btn_refresh.clicked.connect(self._refresh_preview)
        btn_row.addWidget(self.btn_refresh)
        self.btn_copy = QPushButton("Copy")
        self.btn_copy.clicked.connect(self._on_copy)
        btn_row.addWidget(self.btn_copy)
        self.btn_run = QPushButton("Run (blocking, via run_docker_gpu.sh)")
        self.btn_run.clicked.connect(self._run_now)
        btn_row.addWidget(self.btn_run)
        self.btn_save = QPushButton("Save .sh")
        self.btn_save.clicked.connect(self._save_sh)
        btn_row.addWidget(self.btn_save)
        self.btn_load = QPushButton("Load .sh")
        self.btn_load.clicked.connect(self._load_sh)
        btn_row.addWidget(self.btn_load)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        # Log
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Menlo", 10))
        self.log.setMaximumHeight(180)
        outer.addWidget(self.log)

        self._refresh_preview()

    # ── Actions ───────────────────────────────────────────────────────
    def _all_steps(self) -> list[QWidget]:
        return [self.step_arch, self.step_crit, self.step_reg, self.step_ft]

    def _apply_all(self) -> None:
        for s in self._all_steps():
            if hasattr(s, "apply"):
                s.apply()

    def _populate_all(self) -> None:
        for s in self._all_steps():
            if hasattr(s, "populate"):
                s.populate()

    def _refresh_preview(self):
        self._apply_all()
        cmd = build_command(self.state, self.script_in.text().strip())
        self.cmd_view.setPlainText(" ".join(shlex.quote(p) for p in cmd))

    def _on_copy(self):
        from PyQt5.QtWidgets import QApplication
        QApplication.clipboard().setText(self.cmd_view.toPlainText())

    def _pick_script(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select vbp_imagenet_pat.py",
            os.path.dirname(self.script_in.text()),
            "Python (*.py);;All (*)",
        )
        if path:
            self.script_in.setText(path)
            save_torch_pruning_script(path)

    def _build_sh_text(self) -> str:
        cmd = build_command(self.state, self.script_in.text().strip())
        return ("#!/usr/bin/env bash\n"
                + " ".join(shlex.quote(p) for p in cmd) + "\n")

    def _run_now(self):
        self._refresh_preview()
        save_dir = self.state.get("save_dir") or "."
        arch = self.state.get("_arch_choice", "arch")
        crit = self.state.get("_crit_choice", "crit")
        ts = time.strftime("%Y%m%d-%H%M%S")
        try:
            os.makedirs(save_dir, exist_ok=True)
            try:
                os.chmod(save_dir, 0o777)
            except OSError:
                pass
        except OSError as e:
            self.log.appendPlainText(f"[mkdir failed: {e}]")
            return
        sh_path = os.path.join(save_dir, f"wizard_{arch}_{crit}_{ts}.sh")
        try:
            with open(sh_path, "w") as f:
                f.write(self._build_sh_text())
            try:
                os.chmod(sh_path, 0o777)
            except OSError:
                pass
        except OSError as e:
            self.log.appendPlainText(f"[write failed: {e}]")
            return

        desc = f"VBP Wizard {arch} {crit} {ts}"
        full_cmd = (DEFAULT_WRAPPER_A + sh_path + DEFAULT_WRAPPER_B
                    + f" -D '{desc}'")
        confirm = QMessageBox.question(
            self, "Submit?",
            f"$ {full_cmd}\n\nProceed?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        self.log.appendPlainText(f"$ {full_cmd}")
        self.log.repaint()
        try:
            res = subprocess.run(full_cmd, shell=True, capture_output=True,
                                 text=True, timeout=300)
            if res.stdout:
                self.log.appendPlainText(res.stdout.rstrip())
            if res.stderr:
                self.log.appendPlainText("STDERR: " + res.stderr.rstrip())
            self.log.appendPlainText(f"-> rc={res.returncode}")
        except subprocess.TimeoutExpired:
            self.log.appendPlainText("[timeout: 300s]")
        except Exception as e:
            self.log.appendPlainText(f"[error: {e}]")

    def _save_sh(self):
        self._refresh_preview()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save .sh", "wizard_run.sh", "Shell scripts (*.sh)"
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                f.write(self._build_sh_text())
            try:
                os.chmod(path, 0o755)
            except OSError:
                pass
        except OSError as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        self.log.appendPlainText(f"[saved {path}]")

    def _load_sh(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load .sh", "", "Shell scripts (*.sh);;All (*)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                text = f.read()
        except OSError as e:
            QMessageBox.warning(self, "Read failed", str(e))
            return
        parsed = _parse_sh(text)
        new_state = state_from_parsed(parsed)
        # Drop schema-derived flags (script path, etc.) we don't track in state
        unknown = sorted(k for k in parsed
                         if k not in FLAG_ORDER
                         and k not in {"isomorphic", "global_pruning"})
        self.state.clear()
        self.state.update(new_state)
        self._populate_all()
        self._refresh_preview()
        unk_msg = f", ignored: {unknown}" if unknown else ""
        self.log.appendPlainText(
            f"[loaded {os.path.basename(path)}: arch="
            f"{self.state.get('_arch_choice')}, "
            f"crit={self.state.get('_crit_choice')}, "
            f"level={self.state.get('_pruning_level')}, "
            f"reg={'yes' if self.state.get('_reg_yes') else 'no'}{unk_msg}]"
        )
