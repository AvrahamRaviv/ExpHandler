"""Navigation callbacks: project selection, path modal, data loading, content render."""

from dash import callback, Output, Input, State, ctx, no_update as dnu
import dash_mantine_components as dmc

from config import get_project_path, save_project_path, PROJECTS
from scanners.dvnr import scan_dvnr
from scanners.odt import scan_odt
from scanners.vbp import scan_vbp
from screens.runs import build_runs_table
from screens.plots import build_plots_panel


_SCANNERS = {"DVNR": scan_dvnr, "ODT": scan_odt, "VBP": scan_vbp}


def _run_scanner(project: str, root_path: str):
    try:
        return _SCANNERS[project](root_path)
    except Exception as e:
        print(f"[ExpHandler] scan error for {project}: {e}")
        return []


def _render_content(project: str, data: list):
    runs = build_runs_table(project, data)
    plots = build_plots_panel(project, data)
    return runs, plots


# ---------------------------------------------------------------------------
# 1. NavLink click → maybe open modal, maybe load data, always update active
# ---------------------------------------------------------------------------

@callback(
    Output("store-app-state", "data", allow_duplicate=True),
    Output("modal-path-input", "opened", allow_duplicate=True),
    Output("modal-project-label", "children"),
    Output("input-root-path", "value"),
    Output("content-runs",  "children", allow_duplicate=True),
    Output("content-plots", "children", allow_duplicate=True),
    Output("nav-DVNR", "active"),
    Output("nav-ODT",  "active"),
    Output("nav-VBP",  "active"),
    [Input(f"nav-{p}", "n_clicks") for p in PROJECTS],
    State("store-app-state", "data"),
    prevent_initial_call=True,
)
def on_nav_click(*args):
    state = args[-1]
    triggered = ctx.triggered_id  # e.g. "nav-DVNR"
    if not triggered or not triggered.startswith("nav-"):
        return (dnu,) * 9

    project = triggered.replace("nav-", "")
    state["active_project"] = project

    # Active flags for NavLinks
    active_flags = [project == p for p in PROJECTS]

    root_path = get_project_path(project)
    if not root_path:
        # No path configured → open modal
        state["pending_project"] = project
        return state, True, f"Project: {project}", "", dnu, dnu, *active_flags

    # Path is set — load data if not already loaded
    if state["data"].get(project) is None:
        state["data"][project] = _run_scanner(project, root_path)

    runs_content, plots_content = _render_content(project, state["data"][project])
    return state, False, dnu, dnu, runs_content, plots_content, *active_flags


# ---------------------------------------------------------------------------
# 2. Confirm path → save, scan, close modal, render content
# ---------------------------------------------------------------------------

@callback(
    Output("store-app-state", "data", allow_duplicate=True),
    Output("modal-path-input", "opened", allow_duplicate=True),
    Output("content-runs",  "children", allow_duplicate=True),
    Output("content-plots", "children", allow_duplicate=True),
    Output("notifications-container", "children"),
    Input("btn-path-confirm", "n_clicks"),
    State("input-root-path", "value"),
    State("store-app-state", "data"),
    prevent_initial_call=True,
)
def on_path_confirm(n_clicks, root_path, state):
    if not n_clicks or not root_path:
        return dnu, dnu, dnu, dnu, dnu

    project = state.get("pending_project")
    if not project:
        return dnu, dnu, dnu, dnu, dnu

    root_path = root_path.strip()
    save_project_path(project, root_path)
    state["pending_project"] = None

    data = _run_scanner(project, root_path)
    state["data"][project] = data

    runs_content, plots_content = _render_content(project, data)

    n = len(data)
    notif = dmc.Notification(
        id="notif-loaded",
        title=f"{project} loaded",
        message=f"Found {n} experiment{'s' if n != 1 else ''}.",
        action="show",
        color="green" if n > 0 else "orange",
        autoClose=3000,
    )
    return state, False, runs_content, plots_content, notif


# ---------------------------------------------------------------------------
# 3. Tab switch → re-render active tab content (no re-scan needed)
# ---------------------------------------------------------------------------

@callback(
    Output("content-runs",  "children", allow_duplicate=True),
    Output("content-plots", "children", allow_duplicate=True),
    Input("tabs-screen", "value"),
    State("store-app-state", "data"),
    prevent_initial_call=True,
)
def on_tab_switch(tab, state):
    project = state.get("active_project")
    if not project:
        return dnu, dnu
    data = (state.get("data") or {}).get(project)
    if data is None:
        return dnu, dnu
    runs_content, plots_content = _render_content(project, data)
    return runs_content, plots_content
