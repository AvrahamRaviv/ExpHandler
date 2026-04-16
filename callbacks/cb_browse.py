"""Directory browser callbacks for the path-input modal."""

import os
from dash import callback, Output, Input, State, no_update as dnu


def _list_subdirs(path: str) -> list[dict]:
    """Return sorted list of immediate subdirectories as Select options."""
    try:
        entries = sorted(
            e for e in os.listdir(path)
            if os.path.isdir(os.path.join(path, e)) and not e.startswith(".")
        )
        return [{"value": os.path.join(path, e), "label": e} for e in entries]
    except PermissionError:
        return []


# ---------------------------------------------------------------------------
# 1. Modal opens → initialise browser to home (or existing path in text input)
# ---------------------------------------------------------------------------

@callback(
    Output("store-browse-path", "data"),
    Input("modal-path-input", "opened"),
    State("input-root-path", "value"),
    prevent_initial_call=True,
)
def init_browser(opened, current_text):
    if not opened:
        return dnu
    # Start browser at the typed path if it exists, else home dir
    start = (current_text or "").strip()
    if start and os.path.isdir(start):
        return start
    return os.path.expanduser("~")


# ---------------------------------------------------------------------------
# 2. Browse path changes → update displayed path + repopulate select
# ---------------------------------------------------------------------------

@callback(
    Output("browse-current-path", "children"),
    Output("browse-dir-select", "data"),
    Output("browse-dir-select", "value"),
    Input("store-browse-path", "data"),
    prevent_initial_call=True,
)
def refresh_dir_list(browse_path):
    if not browse_path:
        return "", [], None
    subdirs = _list_subdirs(browse_path)
    return browse_path, subdirs, None


# ---------------------------------------------------------------------------
# 3. Subdir selected → navigate into it
# ---------------------------------------------------------------------------

@callback(
    Output("store-browse-path", "data", allow_duplicate=True),
    Input("browse-dir-select", "value"),
    prevent_initial_call=True,
)
def navigate_into(selected_path):
    if not selected_path:
        return dnu
    return selected_path


# ---------------------------------------------------------------------------
# 4. Up button → go to parent directory
# ---------------------------------------------------------------------------

@callback(
    Output("store-browse-path", "data", allow_duplicate=True),
    Input("btn-browse-up", "n_clicks"),
    State("store-browse-path", "data"),
    prevent_initial_call=True,
)
def navigate_up(n_clicks, browse_path):
    if not n_clicks or not browse_path:
        return dnu
    parent = os.path.dirname(browse_path.rstrip("/"))
    return parent if parent and os.path.isdir(parent) else browse_path


# ---------------------------------------------------------------------------
# 5. "Use This Folder" → fill text input with current browse path
# ---------------------------------------------------------------------------

@callback(
    Output("input-root-path", "value", allow_duplicate=True),
    Input("btn-browse-use", "n_clicks"),
    State("store-browse-path", "data"),
    prevent_initial_call=True,
)
def use_browse_path(n_clicks, browse_path):
    if not n_clicks or not browse_path:
        return dnu
    return browse_path
