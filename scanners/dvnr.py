"""Scan a DVNR experiments root folder.

Expected structure:
    root_dir/
        <exp_name>/
            loss.json
            logger.log   (optional)
"""

import json
import os
from pathlib import Path


def scan_dvnr(root_dir: str) -> list:
    root = Path(root_dir)
    results = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        loss_path = exp_dir / "loss.json"
        if not loss_path.exists():
            continue
        try:
            with open(loss_path) as f:
                losses = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Number of epochs = length of first loss array
        n_epochs = len(next(iter(losses.values()), []))
        last_losses = {k: v[-1] for k, v in losses.items() if v}

        results.append({
            "exp_name": exp_dir.name,
            "n_epochs": n_epochs,
            "losses": losses,
            "last_losses": last_losses,
        })
    return results
