"""Persist per-subtype launcher schema (arg list + wrapper bits)."""

import json
import os
import copy

LAUNCHER_CONFIG_PATH = os.path.expanduser("~/.exp_handler_launcher.json")


_RESNET50_TP_SCHEMA = {
    "args": [
        {"name": "model_type",          "default": "cnn",         "type": "str"},
        {"name": "cnn_arch",            "default": "resnet50",    "type": "str"},
        {"name": "checkpoint",          "default": "/algo/NetOptimization/outputs/VBP/ResNet50_TP/resnet50_imagenet1k_mag_sparse.pth", "type": "str"},
        {"name": "data_path",           "default": "/algo/NetOptimization/outputs/VBP/", "type": "str"},
        {"name": "global_pruning",      "default": True,          "type": "bool"},
        {"name": "criterion",           "default": "variance",    "type": "str"},
        {"name": "importance_mode",     "default": "tp_variance", "type": "str"},
        {"name": "norm_per_layer",      "default": True,          "type": "bool"},
        {"name": "bn_recalibration",    "default": True,          "type": "bool"},
        {"name": "sparse_mode",         "default": "none",        "type": "str"},
        {"name": "epochs_sparse",       "default": 0,             "type": "int"},
        {"name": "reg",                 "default": "1e-4",        "type": "str"},
        {"name": "train_batch_size",    "default": 128,           "type": "int"},
        {"name": "opt",                 "default": "adamw",       "type": "str"},
        {"name": "lr",                  "default": "2e-4",        "type": "str"},
        {"name": "epochs_ft",           "default": 30,            "type": "int"},
        {"name": "ft_warmup_epochs",    "default": 3,             "type": "int"},
        {"name": "ft_eta_min",          "default": "1e-6",        "type": "str"},
        {"name": "wd",                  "default": "0",           "type": "str"},
        {"name": "use_kd",              "default": True,          "type": "bool"},
        {"name": "kd_alpha",            "default": "0.0",         "type": "str"},
        {"name": "kd_T",                "default": "4.0",         "type": "str"},
        {"name": "pat_steps",           "default": 1,             "type": "int"},
        {"name": "pat_epochs_per_step", "default": 0,             "type": "int"},
    ],
    "ddp_prefix": "TORCH_DISTRIBUTED_TIMEOUT=7200 python3 -m torch.distributed.launch --nproc_per_node=4",
    "entrypoint": "/home/avrahamra/PycharmProjects/sirc-torch-pruning/benchmarks/vbp/vbp_imagenet_pat.py",
    "wrapper_a": "/algo/ws/shared/remote-gpu/run_docker_gpu.sh -d gitlab-srv:4567/od-alg/od_next_gen:v1.7.7_tp2 -C execute -q gpu_deep_train_low_q -W working_dir -M ",
    "wrapper_b": " -s 25gb -n 10 -o 60000 -A '' -p VISION -v /algo/NetOptimization:/algo/NetOptimization -R 'select[gpu_hm]' -R 'select[hname != gpusrv11]' -E force_python_3=yes -x 4",
    "desc_template": "VBP {subtype} {out_dir_name} {keep_ratio}",
    "kr_arg": "keep_ratio",
    "save_dir_arg": "save_dir",
}


# Map subtype → seed schema. Other subtypes start as a copy of ResNet50_TP.
_SEED_SCHEMAS = {
    "ResNet50_TP": _RESNET50_TP_SCHEMA,
}


def _load_all() -> dict:
    if not os.path.exists(LAUNCHER_CONFIG_PATH):
        return {}
    try:
        with open(LAUNCHER_CONFIG_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_all(cfg: dict):
    with open(LAUNCHER_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def load_schema(subtype: str) -> dict:
    """Return persisted schema for subtype, falling back to seed (or ResNet50_TP)."""
    cfg = _load_all()
    if subtype in cfg:
        return cfg[subtype]
    seed = _SEED_SCHEMAS.get(subtype) or _SEED_SCHEMAS["ResNet50_TP"]
    return copy.deepcopy(seed)


def save_schema(subtype: str, schema: dict):
    cfg = _load_all()
    cfg[subtype] = schema
    _save_all(cfg)


def reset_schema(subtype: str):
    cfg = _load_all()
    if subtype in cfg:
        del cfg[subtype]
        _save_all(cfg)
