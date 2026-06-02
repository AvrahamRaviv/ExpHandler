"""Scan a normalize-net experiments root folder.

Standalone project for `benchmarks/vbp/normalize_net.py`: every experiment is a
PAIR of arms run at identical hyperparameters —

    normalized arm  (default)     : reparam + train, WD acts on v_tilde
    baseline arm    (--no_reparam) : plain training

The headline metric is Δ best_val_acc = normalized − baseline at matched config.

A run is identified by (save_dir, save_tag); arms may share one save_dir (two
save_tags) or live in separate dirs. We pair on config, not on path.

Canonical machine sources (preferred):
    <save_tag>_run.json      — run summary (config + final results, status)
    <save_tag>_metrics.jsonl — one JSON object per epoch (append-only, tail-able)
Log is used only for the per-layer V-norm health signal (not in the JSONL yet):
    <save_dir>/vbp_imagenet.log
"""

import json
import os
import re


# Config keys that must MATCH for two arms to form a pair. Deliberately
# excludes: arm, no_reparam, save_tag, save_dir (these distinguish the arms).
PAIR_KEYS = [
    "model_type", "epochs", "lr", "wd", "opt",
    "ft_eta_min", "ft_warmup_epochs", "max_batches",
    "exclude_classifier", "exclude_stem",
]

# V-norm aggregate health line (normalized arm only, one per epoch + one at
# reparam time). target_desc itself contains parens, e.g. "fc1 (col-norms)".
_VNORM_RE = re.compile(
    r"V-norm aggregate \((.*?),\s*(\d+)\s*layers,\s*(\d+)\s*channels\):\s*"
    r"mean=([\d.]+)\s+median=([\d.]+)\s+std=([\d.]+)\s+"
    r"<0\.01=([\d.]+)%\s+<0\.1=([\d.]+)%\s+<1\.0=([\d.]+)%"
)
_CMD_RE = re.compile(r"command:\s*(.*)$", re.MULTILINE)

_RUN_SUFFIX = "_run.json"
_METRICS_SUFFIX = "_metrics.jsonl"
_LOG_NAME = "vbp_imagenet.log"


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------

def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: str) -> list:
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


def _parse_vnorm(text: str) -> list:
    """Sequence of V-norm aggregate snapshots; idx 0 = post-reparam init."""
    out = []
    for i, m in enumerate(_VNORM_RE.finditer(text)):
        out.append({
            "idx": i,                       # 0 = init (post-reparam), then per-epoch
            "target": m.group(1),
            "n_layers": int(m.group(2)),
            "n_channels": int(m.group(3)),
            "mean": float(m.group(4)),
            "median": float(m.group(5)),
            "std": float(m.group(6)),
            "frac_below_0_01": float(m.group(7)),
            "frac_below_0_1": float(m.group(8)),
            "frac_below_1_0": float(m.group(9)),
        })
    return out


def _parse_commands(text: str) -> list:
    return [m.group(1).strip() for m in _CMD_RE.finditer(text)]


def _command_for_arm(cmds: list, arm: str) -> str:
    """Pick the logged command line matching this arm (--no_reparam ⇒ baseline)."""
    for c in cmds:
        is_baseline = "--no_reparam" in c
        if (arm == "baseline") == is_baseline:
            return c
    return cmds[0] if cmds else ""


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def _model_name(cfg: dict) -> str:
    return cfg.get("cnn_arch") or cfg.get("model_name") or ""


def _pair_signature(cfg: dict) -> tuple:
    return tuple((k, cfg.get(k)) for k in PAIR_KEYS) + (("model", _model_name(cfg)),)


def pair_runs(records: list) -> list:
    """Group run records into normalized-vs-baseline pairs by matched config.

    Returns a list of {key, label, normalized, baseline, runs, delta_best}.
    Runs without a config (no run.json yet) cannot be paired → solo group.
    """
    groups: dict = {}
    order: list = []
    for r in records:
        cfg = r.get("config")
        key = _pair_signature(cfg) if cfg else ("__solo__", r["run_id"])
        if key not in groups:
            groups[key] = {"normalized": None, "baseline": None, "runs": []}
            order.append(key)
        g = groups[key]
        g["runs"].append(r)
        if r["arm"] in ("normalized", "baseline") and g[r["arm"]] is None:
            g[r["arm"]] = r

    pairs = []
    for i, key in enumerate(order):
        g = groups[key]
        norm, base = g["normalized"], g["baseline"]
        delta_best = None
        if (norm and base
                and norm.get("best_val_acc") is not None
                and base.get("best_val_acc") is not None):
            delta_best = norm["best_val_acc"] - base["best_val_acc"]
        label = _pair_label(g["runs"])
        pairs.append({
            "key": key, "label": label, "index": i,
            "normalized": norm, "baseline": base,
            "runs": g["runs"], "delta_best": delta_best,
        })
    return pairs


def _pair_label(runs: list) -> str:
    """Compact human label for a pair from its matched config."""
    cfg = next((r.get("config") for r in runs if r.get("config")), None)
    if not cfg:
        return runs[0]["name"] if runs else "?"
    return f"{_model_name(cfg)} e{cfg.get('epochs')} lr{cfg.get('lr')} wd{cfg.get('wd')}"


def attach_pairing(records: list):
    """Annotate each record in place with its pair label and Δ best_val_acc."""
    for pair in pair_runs(records):
        for r in pair["runs"]:
            r["pair_index"] = pair["index"]
            r["pair_label"] = pair["label"]
            r["paired_delta_best"] = pair["delta_best"]
            partner = (pair["baseline"] if r["arm"] == "normalized"
                       else pair["normalized"])
            r["partner_name"] = partner["name"] if partner else None


# ---------------------------------------------------------------------------
# Per-run record
# ---------------------------------------------------------------------------

def _best_val_acc(run_json: dict, epochs: list):
    if run_json and run_json.get("best_val_acc") is not None:
        return run_json["best_val_acc"]
    bests = [e.get("best_val_acc") for e in epochs if e.get("best_val_acc") is not None]
    if bests:
        return max(bests)
    accs = [e.get("val_acc") for e in epochs if e.get("val_acc") is not None]
    return max(accs) if accs else None


def _build_record(root: str, save_dir: str, tag: str, log_cache: dict) -> dict | None:
    run_json = _read_json(os.path.join(save_dir, tag + _RUN_SUFFIX))
    epochs = _read_jsonl(os.path.join(save_dir, tag + _METRICS_SUFFIX))
    if run_json is None and not epochs:
        return None
    # Preserve file (append) order and tag each row with a cumulative epoch
    # index. A reset (epoch <= previous epoch) marks a new training phase
    # (e.g. sparse → FT) so the GUI can plot on a continuous x-axis without
    # the two phases overlapping at epochs 1..5. Drop malformed rows lacking
    # an integer "epoch" — guards against partial jsonl writes.
    annotated = []
    last_ep, last_cum, phase_idx, offset = -1, 0, 0, 0
    for e in epochs:
        ep = e.get("epoch")
        if not isinstance(ep, int):
            continue
        if ep <= last_ep:
            phase_idx += 1
            offset = last_cum
        cum = ep + offset
        e["phase_idx"] = phase_idx
        e["cum_epoch"] = cum
        annotated.append(e)
        last_ep, last_cum = ep, cum
    epochs = annotated

    config = (run_json or {}).get("config") or {}
    # Arm: run.json wins, else metrics, else infer from config.
    arm = (run_json or {}).get("arm")
    if not arm and epochs:
        arm = epochs[0].get("arm")
    if not arm:
        arm = "baseline" if config.get("no_reparam") else "normalized"

    status = (run_json or {}).get("status") or ("running" if epochs else "unknown")

    # Log (shared per save_dir): V-norm health + launch command.
    log_path = os.path.join(save_dir, _LOG_NAME)
    if log_path not in log_cache:
        try:
            with open(log_path, errors="replace") as f:
                log_cache[log_path] = f.read()
        except OSError:
            log_cache[log_path] = ""
    log_text = log_cache[log_path]
    cmds = _parse_commands(log_text)
    # V-norm lines exist only for the normalized arm; attribute them to it.
    vnorm = _parse_vnorm(log_text) if arm == "normalized" and log_text else []

    rel = os.path.relpath(save_dir, root)
    name = tag if rel in (".", "") else f"{rel}/{tag}"

    target_epochs = (run_json or {}).get("config", {}).get("epochs")
    if target_epochs is None and epochs:
        target_epochs = epochs[-1].get("epochs")

    return {
        "run_id": os.path.join(save_dir, tag),
        "save_dir": save_dir,
        "save_tag": tag,
        "name": name,
        "arm": arm,
        "status": status,
        "config": config,
        "hyperparams": config,            # alias → reuse Runs compare table
        "pre_train_val_acc": (run_json or {}).get("pre_train_val_acc"),
        "best_val_acc": _best_val_acc(run_json, epochs),
        "macs_g": (run_json or {}).get("macs_g"),
        "params_m": (run_json or {}).get("params_m"),
        "checkpoints": (run_json or {}).get("checkpoints") or {},
        "metrics_file": (run_json or {}).get("metrics_file")
        or os.path.join(save_dir, tag + _METRICS_SUFFIX),
        "epochs": epochs,                 # list of per-epoch metric dicts
        "n_epochs": len(epochs),
        "target_epochs": target_epochs,
        "vnorm": vnorm,
        "command": _command_for_arm(cmds, arm),
        "log_path": log_path if log_text else "",
    }


# ---------------------------------------------------------------------------
# Public scanner
# ---------------------------------------------------------------------------

def scan_normnet(root_dir: str) -> list:
    """Walk root recursively; emit one record per (save_dir, save_tag)."""
    records = []
    log_cache: dict = {}
    for save_dir, _subdirs, fnames in os.walk(root_dir):
        tags: set = set()
        for fn in fnames:
            if fn.endswith(_RUN_SUFFIX):
                tags.add(fn[: -len(_RUN_SUFFIX)])
            elif fn.endswith(_METRICS_SUFFIX):
                tags.add(fn[: -len(_METRICS_SUFFIX)])
        for tag in sorted(tags):
            rec = _build_record(root_dir, save_dir, tag, log_cache)
            if rec is not None:
                records.append(rec)

    records.sort(key=lambda r: (r["name"], r["arm"]))
    attach_pairing(records)
    return records
