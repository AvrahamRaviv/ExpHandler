"""Runs screen callbacks: row selection → detail panel."""

import json
from dash import callback, Output, Input, State, no_update as dnu, html
import dash_mantine_components as dmc


def _fmt_val(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, bool):
        return str(v)
    return str(v)


def _detail_panel(title: str, info: dict) -> html.Div:
    rows = [
        html.Tr([html.Td(k, style={"fontWeight": "bold", "paddingRight": 16}),
                 html.Td(_fmt_val(v))])
        for k, v in info.items()
    ]
    return html.Div([
        dmc.Divider(mb=8),
        dmc.Text(title, weight=600, mb=6),
        html.Table(rows, style={"fontSize": 13, "fontFamily": "monospace",
                                 "borderCollapse": "collapse"}),
    ], style={"padding": "8px 4px"})


# ---------------------------------------------------------------------------
# DVNR row select
# ---------------------------------------------------------------------------

@callback(
    Output("runs-detail-panel", "children", allow_duplicate=True),
    Input("dvnr-runs-table", "selected_rows"),
    State("dvnr-runs-table", "data"),
    State("store-app-state", "data"),
    prevent_initial_call=True,
)
def dvnr_row_select(selected_rows, table_data, state):
    if not selected_rows:
        return None
    row = table_data[selected_rows[0]]
    exp_name = row["exp_name"]

    # Get full data from store
    exp_data = next(
        (e for e in (state["data"].get("DVNR") or []) if e["exp_name"] == exp_name),
        None,
    )
    if not exp_data:
        return None

    info = {"exp_name": exp_name, "n_epochs": exp_data["n_epochs"]}
    info.update({f"last_{k}": round(v, 4) for k, v in exp_data["last_losses"].items()})
    return _detail_panel(f"Detail: {exp_name}", info)


# ---------------------------------------------------------------------------
# ODT row select
# ---------------------------------------------------------------------------

@callback(
    Output("runs-detail-panel", "children", allow_duplicate=True),
    Input("odt-runs-table", "selected_rows"),
    State("odt-runs-table", "data"),
    State("store-app-state", "data"),
    prevent_initial_call=True,
)
def odt_row_select(selected_rows, table_data, state):
    if not selected_rows:
        return None
    row = table_data[selected_rows[0]]
    exp_name = row["exp_name"]

    exp_data = next(
        (e for e in (state["data"].get("ODT") or []) if e["exp_name"] == exp_name),
        None,
    )
    if not exp_data:
        return None

    info = {"exp_name": exp_name}
    info.update({k: _fmt_val(v) for k, v in exp_data["metrics"].items()})
    return _detail_panel(f"Detail: {exp_name}", info)


# ---------------------------------------------------------------------------
# VBP row select
# ---------------------------------------------------------------------------

@callback(
    Output("runs-detail-panel", "children", allow_duplicate=True),
    Input("vbp-runs-table", "selected_rows"),
    State("vbp-runs-table", "data"),
    State("store-app-state", "data"),
    prevent_initial_call=True,
)
def vbp_row_select(selected_rows, table_data, state):
    if not selected_rows:
        return None
    row = table_data[selected_rows[0]]
    key_setup = row["setup"]
    key_kr = row["kr_folder"]

    exp_data = next(
        (e for e in (state["data"].get("VBP") or [])
         if e["setup"] == key_setup and e["kr_folder"] == key_kr),
        None,
    )
    if not exp_data:
        return None

    info = {"setup": key_setup, "kr_folder": key_kr}
    hp = exp_data.get("hyperparams", {})
    info.update({k: _fmt_val(v) for k, v in hp.items()})

    summary = exp_data.get("summary", {})
    summary_info = {f"[summary] {k}": _fmt_val(v) for k, v in summary.items()}

    retention = exp_data.get("step_retentions", [])
    if retention:
        last_ret = retention[-1]
        info["[step_retention] acc"] = _fmt_val(last_ret.get("acc"))
        info["[step_retention] macs_G"] = _fmt_val(last_ret.get("macs_G"))

    info.update(summary_info)
    return _detail_panel(f"Detail: {key_setup} / {key_kr}", info)
