"""Loader for per-network channel-score files.

The training project writes one JSON per network at
``<save_dir>/<tag>_channel_scores.json``::

    {
      "schema": "channel_scores/v1",
      "model": "resnet50",
      "scorer": "l1norm",
      "stage": "pre_prune",          # optional: pre_prune | post_ft | init
      "higher_is_better": true,      # optional, default true
      "layers": [
        {"name": "layer1.0.conv1",
         "scores": [0.12, 0.45, ...],   # length = n_channels (required)
         "kept":   [true, false, ...]}  # optional, same length; prune mask
      ]
    }

Only ``layers[].name`` and ``layers[].scores`` are required. Everything else is
used for labels and the architecture-match key (diff mode). This loader is
project-agnostic: discovery is purely path-based and independent of the per
project ``list[dict]`` records produced by the other scanners.
"""

import glob
import json
import os

import numpy as np

SCORE_GLOB = "*_channel_scores.json"


def discover_channel_scores(root_dir: str) -> list:
    """Return sorted abs paths of ``*_channel_scores.json`` under ``root_dir``."""
    if not root_dir or not os.path.isdir(root_dir):
        return []
    paths = glob.glob(os.path.join(root_dir, "**", SCORE_GLOB), recursive=True)
    return sorted(os.path.abspath(p) for p in paths)


def _label(path: str, model, scorer, stage) -> str:
    """Human label: prefer model/scorer/stage, fall back to filename stem."""
    parts = [p for p in (model, scorer, stage) if p]
    if parts:
        return " · ".join(str(p) for p in parts)
    stem = os.path.basename(path)
    for suf in ("_channel_scores.json", ".json"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem or path


def load_channel_scores(path: str) -> dict | None:
    """Parse one channel-score file into a normalized record.

    Returns ``None`` if the file is unreadable or has no usable layers. Layers
    with empty or non-numeric scores are skipped rather than failing the file.
    """
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None

    layers = []
    arch = []
    gmin = np.inf
    gmax = -np.inf
    for ent in raw.get("layers", []) or []:
        if not isinstance(ent, dict):
            continue
        name = ent.get("name")
        try:
            scores = np.asarray(ent.get("scores", []), dtype=float)
        except (TypeError, ValueError):
            continue
        if name is None or scores.size == 0 or not np.isfinite(scores).any():
            continue
        kept = ent.get("kept")
        if kept is not None:
            try:
                kept = np.asarray(kept, dtype=bool)
                if kept.size != scores.size:
                    kept = None
            except (TypeError, ValueError):
                kept = None
        layers.append({"name": str(name), "scores": scores, "kept": kept})
        arch.append((str(name), int(scores.size)))
        finite = scores[np.isfinite(scores)]
        gmin = min(gmin, float(finite.min()))
        gmax = max(gmax, float(finite.max()))

    if not layers:
        return None

    model = raw.get("model")
    scorer = raw.get("scorer")
    stage = raw.get("stage")
    return {
        "path": os.path.abspath(path),
        "label": _label(path, model, scorer, stage),
        "model": model,
        "scorer": scorer,
        "stage": stage,
        "higher_is_better": bool(raw.get("higher_is_better", True)),
        "layers": layers,
        "arch_key": tuple(arch),
        "gmin": gmin,
        "gmax": gmax,
    }
