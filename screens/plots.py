"""Build the Plots panel component for a given project and its loaded data."""

from dash import dcc, html
import dash_mantine_components as dmc
import plotly.graph_objects as go
import plotly.express as px


def build_plots_panel(project: str, data: list):
    """Return a panel with selectors and an empty graph (populated by cb_plots)."""
    if not data:
        return dmc.Text("No experiments found.", color="dimmed", mt=20)

    if project == "DVNR":
        return _dvnr_panel(data)
    elif project == "ODT":
        return _odt_panel(data)
    elif project == "VBP":
        return _vbp_panel(data)
    return dmc.Text(f"Unknown project: {project}", color="red")


# ---------------------------------------------------------------------------
# DVNR — line chart: loss vs epoch, multi-exp + multi-loss-key selectable
# ---------------------------------------------------------------------------

def _dvnr_panel(data: list):
    exp_options = [{"value": e["exp_name"], "label": e["exp_name"]} for e in data]
    loss_keys = list(data[0]["losses"].keys()) if data else []
    loss_options = [{"value": k, "label": k} for k in loss_keys]
    default_exps = [e["exp_name"] for e in data]
    default_losses = loss_keys[:3] if len(loss_keys) > 3 else loss_keys

    return html.Div([
        dmc.Group([
            dmc.MultiSelect(
                id="plots-exp-selector",
                label="Experiments",
                data=exp_options,
                value=default_exps,
                style={"width": 350},
                searchable=True,
            ),
            dmc.MultiSelect(
                id="plots-loss-selector",
                label="Loss keys",
                data=loss_options,
                value=default_losses,
                style={"width": 400},
                searchable=True,
            ),
        ], spacing=20, align="flex-end"),
        dcc.Graph(id="plot-main", figure=_empty_fig("Select experiments above"),
                  style={"marginTop": 16}),
    ])


# ---------------------------------------------------------------------------
# ODT — bar chart: metrics per experiment
# ---------------------------------------------------------------------------

def _odt_panel(data: list):
    exp_options = [{"value": e["exp_name"], "label": e["exp_name"]} for e in data]
    default_exps = [e["exp_name"] for e in data]

    return html.Div([
        dmc.MultiSelect(
            id="plots-exp-selector",
            label="Experiments",
            data=exp_options,
            value=default_exps,
            style={"width": 450},
            searchable=True,
        ),
        # Hidden placeholder for loss selector (used by shared callback)
        dcc.Store(id="plots-loss-selector-value", data=[]),
        dcc.Graph(id="plot-main", figure=_empty_fig("Select experiments above"),
                  style={"marginTop": 16}),
    ])


# ---------------------------------------------------------------------------
# VBP — val_acc vs epoch (FT phase), multi-exp selectable
# ---------------------------------------------------------------------------

def _vbp_panel(data: list):
    exp_options = [
        {"value": f"{e['setup']}/{e['kr_folder']}", "label": f"{e['setup']} | {e['kr_folder']}"}
        for e in data
    ]
    default_exps = [opt["value"] for opt in exp_options]

    return html.Div([
        dmc.MultiSelect(
            id="plots-exp-selector",
            label="Experiments  (setup / keep-ratio)",
            data=exp_options,
            value=default_exps,
            style={"width": 500},
            searchable=True,
        ),
        dcc.Graph(id="plot-main", figure=_empty_fig("Select experiments above"),
                  style={"marginTop": 16}),
    ])


# ---------------------------------------------------------------------------
# Figure generators (called from cb_plots)
# ---------------------------------------------------------------------------

def make_dvnr_figure(data: list, selected_exps: list, selected_losses: list) -> go.Figure:
    """Line chart: one trace per (exp, loss_key)."""
    exp_map = {e["exp_name"]: e for e in data}
    fig = go.Figure()
    for exp_name in selected_exps:
        exp = exp_map.get(exp_name)
        if not exp:
            continue
        for loss_key in selected_losses:
            values = exp["losses"].get(loss_key, [])
            epochs = list(range(1, len(values) + 1))
            fig.add_trace(go.Scatter(
                x=epochs, y=values,
                mode="lines+markers",
                name=f"{exp_name} / {loss_key}",
                hovertemplate="epoch %{x}<br>%{y:.4f}<extra>%{fullData.name}</extra>",
            ))
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend_title="Exp / Loss key",
        template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=50),
    )
    return fig


def make_odt_figure(data: list, selected_exps: list) -> go.Figure:
    """Grouped bar chart: one group per experiment, one bar per metric."""
    exp_map = {e["exp_name"]: e for e in data}
    fig = go.Figure()
    for exp_name in selected_exps:
        exp = exp_map.get(exp_name)
        if not exp:
            continue
        metrics = exp["metrics"]
        keys = [k for k, v in metrics.items() if v is not None]
        values = [metrics[k] for k in keys]
        fig.add_trace(go.Bar(
            name=exp_name,
            x=keys, y=values,
            hovertemplate="%{x}: %{y:.4f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Value",
        yaxis_range=[0, 1.05],
        legend_title="Experiment",
        template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=100),
        xaxis_tickangle=-45,
    )
    return fig


def make_vbp_figure(data: list, selected_exps: list) -> go.Figure:
    """Line chart: val_acc vs FT epoch, one trace per (setup/kr)."""
    exp_map = {f"{e['setup']}/{e['kr_folder']}": e for e in data}
    fig = go.Figure()
    for key in selected_exps:
        exp = exp_map.get(key)
        if not exp:
            continue
        ft_epochs = [e for e in exp["epochs"] if e["phase"] in ("FT", "PAT")]
        if not ft_epochs:
            continue
        x = [e["epoch"] for e in ft_epochs]
        y = [e["val_acc"] for e in ft_epochs]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines+markers",
            name=key,
            hovertemplate="epoch %{x}<br>val_acc=%{y:.4f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        xaxis_title="FT Epoch",
        yaxis_title="Val Accuracy",
        legend_title="Setup / KR",
        template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=50),
    )
    return fig


def _empty_fig(msg: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        annotations=[{"text": msg, "xref": "paper", "yref": "paper",
                       "x": 0.5, "y": 0.5, "showarrow": False,
                       "font": {"size": 14, "color": "gray"}}],
    )
    return fig
