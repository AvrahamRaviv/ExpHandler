"""Main layout for ExpHandler.

Structure:
    NotificationsProvider
      root Div
        Store (app state)
        Modal (path input)
        Grid
          Col(2)  ← left sidebar: project nav
          Col(10) ← main area: screen tabs + content
"""

from dash import html, dcc
from dash_iconify import DashIconify
import dash_mantine_components as dmc

from config import PROJECTS

VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

SIDEBAR_STYLE = {
    "height": "100vh",
    "borderRight": "1px solid #dee2e6",
    "padding": "16px 8px",
    "backgroundColor": "#f8f9fa",
}

NAV_ICON = {
    "DVNR": "material-symbols:video-camera-back",
    "ODT":  "material-symbols:person-search",
    "VBP":  "material-symbols:compress",
}

INITIAL_STATE = {
    "active_project": None,
    "pending_project": None,
    "data": {"DVNR": None, "ODT": None, "VBP": None},
    "runs_selected_row": None,
    "plots_selected_exps": [],
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def gen_layout():
    return dmc.NotificationsProvider(
        html.Div([
            # ---- Global state store ----
            dcc.Store(id="store-app-state", data=INITIAL_STATE),

            # ---- Notifications container ----
            html.Div(id="notifications-container"),

            # ---- Browse path store (tracks currently-viewed directory) ----
            dcc.Store(id="store-browse-path", data=""),

            # ---- Path-input modal ----
            dmc.Modal(
                id="modal-path-input",
                opened=False,
                title="Set Root Experiment Folder",
                zIndex=300,
                size="lg",
                children=[
                    dmc.Text("", id="modal-project-label", size="sm", color="dimmed", mb=8),

                    # --- Manual entry row ---
                    dmc.Group([
                        dmc.TextInput(
                            id="input-root-path",
                            label="Type path directly",
                            placeholder="/path/to/experiments",
                            style={"flex": 1},
                        ),
                        dmc.Button(
                            "Confirm",
                            id="btn-path-confirm",
                            color="blue",
                            style={"marginTop": 24},
                        ),
                    ], align="flex-end", spacing=8),

                    dmc.Divider(label="or browse", labelPosition="center", my=14),

                    # --- Browser ---
                    dmc.Group([
                        dmc.Button(
                            "↑ Up",
                            id="btn-browse-up",
                            variant="subtle",
                            compact=True,
                        ),
                        dmc.Text("", id="browse-current-path",
                                 size="xs", color="dimmed",
                                 style={"fontFamily": "monospace", "wordBreak": "break-all"}),
                    ], spacing=8, mb=6),

                    dmc.Select(
                        id="browse-dir-select",
                        placeholder="— subfolders —",
                        data=[],
                        value=None,
                        searchable=True,
                        style={"width": "100%"},
                    ),

                    dmc.Space(h=8),
                    dmc.Button(
                        "Use This Folder",
                        id="btn-browse-use",
                        variant="light",
                        color="teal",
                        fullWidth=True,
                    ),
                ],
            ),

            # ---- Main grid ----
            dmc.Grid([
                # ---- Sidebar ----
                dmc.Col(
                    html.Div([
                        dmc.Group([
                            DashIconify(icon="material-symbols:science", width=28),
                            dmc.Text("ExpHandler", weight=700, size="lg"),
                        ], spacing=6, mb=4),
                        dmc.Text(f"v{VERSION}", size="xs", color="dimmed", mb=12),
                        dmc.Divider(mb=12),

                        dmc.Stack([
                            dmc.NavLink(
                                id=f"nav-{proj}",
                                label=proj,
                                icon=DashIconify(icon=NAV_ICON[proj], width=20),
                                n_clicks=0,
                            )
                            for proj in PROJECTS
                        ], spacing=4),
                    ], style=SIDEBAR_STYLE),
                    span=2,
                ),

                # ---- Main area ----
                dmc.Col(
                    html.Div([
                        dmc.Tabs(
                            id="tabs-screen",
                            value="runs",
                            children=[
                                dmc.TabsList([
                                    dmc.Tab("Runs",  value="runs",
                                            icon=DashIconify(icon="material-symbols:table-rows", width=16)),
                                    dmc.Tab("Plots", value="plots",
                                            icon=DashIconify(icon="material-symbols:show-chart", width=16)),
                                ]),
                                dmc.TabsPanel(
                                    html.Div(
                                        id="content-runs",
                                        children=_splash_message(),
                                        style={"padding": "16px"},
                                    ),
                                    value="runs",
                                ),
                                dmc.TabsPanel(
                                    html.Div(
                                        id="content-plots",
                                        children=_splash_message(),
                                        style={"padding": "16px"},
                                    ),
                                    value="plots",
                                ),
                            ],
                        ),
                    ], style={"padding": "8px"}),
                    span=10,
                ),
            ], gutter=0),
        ]),
        position="top-center",
    )


def _splash_message():
    return dmc.Center(
        dmc.Stack([
            DashIconify(icon="material-symbols:science", width=64, color="#adb5bd"),
            dmc.Text("Select a project from the sidebar", size="xl", color="dimmed"),
            dmc.Text("DVNR · ODT · VBP", size="sm", color="dimmed"),
        ], align="center", spacing=8),
        style={"height": "60vh"},
    )
