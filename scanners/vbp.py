"""Scan a VBP experiments root folder.

Expected structure:
    root_dir/
        <setup_name>/
            kr_0.95/
                vbp_imagenet.log
            kr_0.90/
                vbp_imagenet.log

Parsing logic adapted from:
    Torch-Pruning/benchmarks/vbp/parse_logs.py
"""

import os
import re


# ---------------------------------------------------------------------------
# Log parsing (adapted from parse_logs.py)
# ---------------------------------------------------------------------------

def _parse_hyperparams(text: str) -> dict:
    params = {}
    in_args = False
    for line in text.split("\n"):
        if "Unified Pipeline" in line or "VBP Pruning" in line:
            in_args = True
            continue
        if in_args:
            m = re.search(r"INFO \|   (\w+): (.*)", line)
            if m:
                key, val = m.group(1), m.group(2).strip()
                if val in ("True", "False"):
                    val = val == "True"
                elif val == "None":
                    val = None
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                params[key] = val
            elif params:
                break
    return params


def _parse_epochs(text: str) -> list:
    epochs = []
    for m in re.finditer(
        r"\[(\w+)\]\s*Epoch\s+(\d+)/(\d+):\s*train_loss=([\d.]+),\s*val_acc=([\d.]+),\s*MACs=([\d.]+)G"
        r"(?:\s*\|\s*(.*?))?$",
        text, re.MULTILINE
    ):
        entry = {
            "phase": m.group(1),
            "epoch": int(m.group(2)),
            "total_epochs": int(m.group(3)),
            "train_loss": float(m.group(4)),
            "val_acc": float(m.group(5)),
            "macs_G": float(m.group(6)),
        }
        if m.group(7):
            for aux in re.finditer(r"(\w+)=([\d.]+)", m.group(7)):
                entry[aux.group(1)] = float(aux.group(2))
        epochs.append(entry)
    return epochs


def _parse_step_retentions(text: str) -> list:
    retentions = []
    for m in re.finditer(
        r"(?:Step\s+)?[Rr]etention.*?acc=([\d.]+),\s*loss=([\d.]+),\s*MACs=([\d.]+)G",
        text
    ):
        retentions.append({
            "acc": float(m.group(1)),
            "loss": float(m.group(2)),
            "macs_G": float(m.group(3)),
        })
    return retentions


def _parse_summary(text: str, epochs: list) -> dict:
    summary = {}
    m = re.search(r"Original [Aa]cc(?:uracy)?:\s*([\d.]+)", text)
    if m:
        summary["original_acc"] = float(m.group(1))

    m = re.search(r"Final [Aa]cc(?:uracy)?:\s*([\d.]+)", text)
    if m:
        summary["final_acc"] = float(m.group(1))

    ft_accs = [e["val_acc"] for e in epochs if e["phase"].upper() in ("FT", "PAT")]
    if ft_accs:
        summary["best_acc"] = max(ft_accs)

    m = re.search(r"Baseline:\s*([\d.]+)G MACs,\s*([\d.]+)M params", text)
    if m:
        summary["base_macs_G"] = float(m.group(1))
        summary["base_params_M"] = float(m.group(2))

    m = re.search(r"Base MACs:\s*([\d.]+)G\s*->\s*Pruned:\s*([\d.]+)G", text)
    if m:
        summary["base_macs_G"] = float(m.group(1))
        summary["pruned_macs_G"] = float(m.group(2))

    m = re.search(r"Base Params:\s*([\d.]+)M\s*->\s*Pruned:\s*([\d.]+)M", text)
    if m:
        summary["base_params_M"] = float(m.group(1))
        summary["pruned_params_M"] = float(m.group(2))

    return summary


def _parse_log(log_path: str) -> dict:
    with open(log_path) as f:
        text = f.read()

    all_params = _parse_hyperparams(text)
    epochs = _parse_epochs(text)
    summary = _parse_summary(text, epochs)

    # Infer keep_ratio from folder name if not in log
    if "keep_ratio" not in all_params:
        folder = os.path.basename(os.path.dirname(log_path))
        m = re.match(r"kr_([\d.]+)", folder)
        if m:
            all_params["keep_ratio"] = float(m.group(1))

    return {
        "hyperparams": all_params,
        "epochs": epochs,
        "step_retentions": _parse_step_retentions(text),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Public scanner
# ---------------------------------------------------------------------------

def scan_vbp(root_dir: str) -> list:
    results = []
    for setup_name in sorted(os.listdir(root_dir)):
        setup_dir = os.path.join(root_dir, setup_name)
        if not os.path.isdir(setup_dir):
            continue
        for kr_folder in sorted(os.listdir(setup_dir)):
            kr_dir = os.path.join(setup_dir, kr_folder)
            if not os.path.isdir(kr_dir):
                continue
            log_path = os.path.join(kr_dir, "vbp_imagenet.log")
            if not os.path.exists(log_path):
                continue
            try:
                data = _parse_log(log_path)
            except OSError:
                continue

            hp = data["hyperparams"]
            summary = data["summary"]
            results.append({
                "setup": setup_name,
                "kr_folder": kr_folder,
                "keep_ratio": hp.get("keep_ratio"),
                "model": hp.get("cnn_arch") or hp.get("model_name", ""),
                "criterion": hp.get("criterion", ""),
                "final_acc": summary.get("final_acc"),
                "best_acc": summary.get("best_acc"),
                "original_acc": summary.get("original_acc"),
                "base_macs_G": summary.get("base_macs_G"),
                "pruned_macs_G": summary.get("pruned_macs_G"),
                "epochs": data["epochs"],
                "step_retentions": data["step_retentions"],
                "hyperparams": hp,
                "summary": summary,
            })
    return results
