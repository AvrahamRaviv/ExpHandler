"""Persist per-project root paths in ~/.exp_handler_config.json."""

import json
import os

CONFIG_PATH = os.path.expanduser("~/.exp_handler_config.json")
PROJECTS = ["DVNR", "ODT", "VBP"]

# Directory where LSF writes <job_id>.out files, organised as <DD-MM-YY>/<job_id>.out
LSF_LOG_DIR = "/algo/ws/shared/remote-gpu/log/avrahamra/"

# Default starting directory when opening the folder picker for the first time
DEFAULT_PATHS = {
    "DVNR": "/algo/NetOptimization/outputs/DOF/",
    "ODT":  "/home/avrahamra/PycharmProjects/experiments/ODT_CP/",
    "VBP":  "/algo/NetOptimization/outputs/VBP/",
}

# Default path to vbp_imagenet_pat.py used by the VBP Wizard.
DEFAULT_TORCH_PRUNING_SCRIPT = (
    "/home/avrahamra/PycharmProjects/sirc-torch-pruning/"
    "benchmarks/vbp/vbp_imagenet_pat.py"
)


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_project_path(project: str, path: str):
    cfg = load_config()
    if project not in cfg:
        cfg[project] = {}
    cfg[project]["root_path"] = path
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def get_project_path(project: str):
    cfg = load_config()
    return cfg.get(project, {}).get("root_path", None)


def get_torch_pruning_script() -> str:
    cfg = load_config()
    return cfg.get("torch_pruning_script") or DEFAULT_TORCH_PRUNING_SCRIPT


def save_torch_pruning_script(path: str):
    cfg = load_config()
    cfg["torch_pruning_script"] = path
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
