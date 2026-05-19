"""Scan a DOF experiments root folder.

Expected structure:
    root_dir/
        <exp_name>/
            loss.csv   — rows of form:
                epoch,loss
                24521,1.8307,epe,0.5959,calc_hard_photo_flow_loss,1.3063,total_loss,1.8307,
            (key/value pairs follow the leading `epoch,loss` columns)
"""

from pathlib import Path


_LOSS_KEYS = ("epe", "calc_hard_photo_flow_loss", "total_loss")


def _load_loss_csv(path: Path) -> dict:
    """Parse DOF loss.csv → {key: [values sorted by epoch]}."""
    rows: list[tuple[int, dict]] = []
    with open(path) as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
            if len(parts) < 2:
                continue
            try:
                epoch = int(parts[0])
            except ValueError:
                continue
            d: dict = {}
            i = 2  # skip [epoch, loss]
            while i + 1 < len(parts):
                k = parts[i]
                try:
                    v = float(parts[i + 1])
                except ValueError:
                    i += 2
                    continue
                d[k] = v
                i += 2
            if d:
                rows.append((epoch, d))

    rows.sort(key=lambda r: r[0])

    keys: list[str] = []
    for _, d in rows:
        for k in d:
            if k not in keys:
                keys.append(k)
    ordered = [k for k in _LOSS_KEYS if k in keys] + \
              [k for k in keys if k not in _LOSS_KEYS]
    return {k: [d.get(k) for _, d in rows if d.get(k) is not None] for k in ordered}


def scan_dof(root_dir: str) -> list:
    root = Path(root_dir)
    results = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        csv_path = exp_dir / "loss.csv"
        if not csv_path.exists():
            continue
        try:
            losses = _load_loss_csv(csv_path)
        except OSError:
            continue
        if not losses:
            continue
        n_epochs = max((len(v) for v in losses.values()), default=0)
        last_losses = {k: v[-1] for k, v in losses.items() if v}

        results.append({
            "exp_name": exp_dir.name,
            "n_epochs": n_epochs,
            "losses": losses,
            "last_losses": last_losses,
        })
    return results
