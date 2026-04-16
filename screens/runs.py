"""Build the Runs table component for a given project and its loaded data."""

from dash import dash_table, html
import dash_mantine_components as dmc


# Columns to always show first in ODT metrics table
ODT_PRIMARY_COLS = ["AP", "DR", "mIoU", "total_metric"]


def build_runs_table(project: str, data: list):
    """Return a DataTable (and optional info text) for the given project data."""
    if not data:
        return dmc.Text("No experiments found in the selected folder.", color="dimmed", mt=20)

    if project == "DVNR":
        return _dvnr_table(data)
    elif project == "ODT":
        return _odt_table(data)
    elif project == "VBP":
        return _vbp_table(data)
    return dmc.Text(f"Unknown project: {project}", color="red")


# ---------------------------------------------------------------------------
# DVNR
# ---------------------------------------------------------------------------

def _dvnr_table(data: list):
    # Collect all loss keys from first experiment
    loss_keys = list(data[0]["last_losses"].keys()) if data else []

    columns = [{"name": "Experiment", "id": "exp_name"}] + \
              [{"name": "Epochs", "id": "n_epochs"}] + \
              [{"name": k, "id": k, "type": "numeric",
                "format": dash_table.Format.Format(precision=3,
                                                   scheme=dash_table.Format.Scheme.fixed)}
               for k in loss_keys]

    rows = []
    for exp in data:
        row = {"exp_name": exp["exp_name"], "n_epochs": exp["n_epochs"]}
        row.update({k: round(v, 4) for k, v in exp["last_losses"].items()})
        rows.append(row)

    return _make_table("dvnr-runs-table", columns, rows)


# ---------------------------------------------------------------------------
# ODT
# ---------------------------------------------------------------------------

def _odt_table(data: list):
    # Collect all metric keys; put primary ones first
    all_keys = list(data[0]["metrics"].keys()) if data else []
    ordered_keys = [k for k in ODT_PRIMARY_COLS if k in all_keys] + \
                   [k for k in all_keys if k not in ODT_PRIMARY_COLS]

    columns = [{"name": "Experiment", "id": "exp_name"}] + [
        {"name": k, "id": k, "type": "numeric",
         "format": dash_table.Format.Format(precision=3,
                                            scheme=dash_table.Format.Scheme.fixed)}
        for k in ordered_keys
    ]

    rows = []
    for exp in data:
        row = {"exp_name": exp["exp_name"]}
        for k in ordered_keys:
            v = exp["metrics"].get(k)
            row[k] = round(v, 4) if v is not None else None
        rows.append(row)

    return _make_table("odt-runs-table", columns, rows)


# ---------------------------------------------------------------------------
# VBP
# ---------------------------------------------------------------------------

def _vbp_table(data: list):
    columns = [
        {"name": "Setup",         "id": "setup"},
        {"name": "KR Folder",     "id": "kr_folder"},
        {"name": "Keep Ratio",    "id": "keep_ratio",   "type": "numeric"},
        {"name": "Model",         "id": "model"},
        {"name": "Criterion",     "id": "criterion"},
        {"name": "Orig Acc",      "id": "original_acc", "type": "numeric",
         "format": dash_table.Format.Format(precision=4, scheme=dash_table.Format.Scheme.fixed)},
        {"name": "Final Acc",     "id": "final_acc",    "type": "numeric",
         "format": dash_table.Format.Format(precision=4, scheme=dash_table.Format.Scheme.fixed)},
        {"name": "Best Acc",      "id": "best_acc",     "type": "numeric",
         "format": dash_table.Format.Format(precision=4, scheme=dash_table.Format.Scheme.fixed)},
        {"name": "Base MACs (G)", "id": "base_macs_G",  "type": "numeric",
         "format": dash_table.Format.Format(precision=3, scheme=dash_table.Format.Scheme.fixed)},
        {"name": "Pruned MACs (G)", "id": "pruned_macs_G", "type": "numeric",
         "format": dash_table.Format.Format(precision=3, scheme=dash_table.Format.Scheme.fixed)},
        {"name": "Retention %",  "id": "retention_pct", "type": "numeric",
         "format": dash_table.Format.Format(precision=1, scheme=dash_table.Format.Scheme.fixed)},
    ]

    rows = []
    for exp in data:
        orig = exp.get("original_acc")
        final = exp.get("final_acc") or exp.get("best_acc")
        retention = round(100.0 * final / orig, 1) if orig and final else None
        row = {
            "setup":        exp["setup"],
            "kr_folder":    exp["kr_folder"],
            "keep_ratio":   exp.get("keep_ratio"),
            "model":        exp.get("model", ""),
            "criterion":    exp.get("criterion", ""),
            "original_acc": round(orig, 4) if orig else None,
            "final_acc":    round(exp["final_acc"], 4) if exp.get("final_acc") else None,
            "best_acc":     round(exp["best_acc"], 4) if exp.get("best_acc") else None,
            "base_macs_G":  exp.get("base_macs_G"),
            "pruned_macs_G": exp.get("pruned_macs_G"),
            "retention_pct": retention,
        }
        rows.append(row)

    return _make_table("vbp-runs-table", columns, rows)


# ---------------------------------------------------------------------------
# Shared table builder
# ---------------------------------------------------------------------------

def _make_table(table_id: str, columns: list, rows: list):
    return html.Div([
        dash_table.DataTable(
            id=table_id,
            columns=columns,
            data=rows,
            sort_action="native",
            filter_action="native",
            page_size=25,
            row_selectable="single",
            selected_rows=[],
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#f8f9fa",
                "fontWeight": "bold",
                "border": "1px solid #dee2e6",
            },
            style_cell={
                "textAlign": "left",
                "padding": "8px 12px",
                "border": "1px solid #dee2e6",
                "fontFamily": "monospace",
                "fontSize": 13,
                "maxWidth": 200,
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
            ],
            tooltip_delay=0,
            tooltip_duration=None,
        ),
        html.Div(id="runs-detail-panel", style={"marginTop": 12}),
    ])
