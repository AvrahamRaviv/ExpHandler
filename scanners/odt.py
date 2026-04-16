"""Scan an ODT experiments root folder.

Expected structure:
    root_dir/
        <exp_name>/
            evaluation_metrices/
                03_total_metrics.json
"""

import json
import os
from pathlib import Path


def _load_metrics_json(path: Path) -> dict:
    """Load metrics JSON, handling bare NaN tokens."""
    text = path.read_text()
    text = text.replace("NaN", "null").replace("nan", "null")
    raw = json.loads(text)
    # Values are lists — take the first element as the scalar
    return {
        k: (v[0] if isinstance(v, list) and len(v) > 0 else v)
        for k, v in raw.items()
    }


def scan_odt(root_dir: str) -> list:
    root = Path(root_dir)
    results = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        metrics_path = exp_dir / "evaluation_metrices" / "03_total_metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = _load_metrics_json(metrics_path)
        except (json.JSONDecodeError, OSError):
            continue

        results.append({
            "exp_name": exp_dir.name,
            "metrics": metrics,
        })
    return results
