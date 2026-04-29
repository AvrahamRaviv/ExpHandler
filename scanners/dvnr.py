"""Scan a DVNR experiments root folder.

Expected structure:
    root_dir/
        <exp_name>/
            loss.json          — per-component loss curves
            loss.csv           — total loss (epoch, loss_nr columns)
            logger.log         (optional)
"""

import csv
import json
from pathlib import Path


def _load_loss_csv(path: Path) -> list:
    """Return loss_nr values sorted by epoch from loss.csv."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row["epoch"])
                value = float(row["loss_nr"])
                rows.append((epoch, value))
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: r[0])
    return [v for _, v in rows]


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

        # Load total loss from CSV and prepend it
        csv_path = exp_dir / "loss.csv"
        if csv_path.exists():
            try:
                total_values = _load_loss_csv(csv_path)
                if total_values:
                    losses = {"loss_nr": total_values, **losses}
            except OSError:
                pass

        # Number of epochs = length of first loss array
        n_epochs = len(next(iter(losses.values()), []))
        last_losses = {k: v[-1] for k, v in losses.items() if v}

        results.append({
            "exp_name": exp_dir.name.replace("debug_MX_", "").replace("debug_", ""),
            "n_epochs": n_epochs,
            "losses": losses,
            "last_losses": last_losses,
        })
    return results
