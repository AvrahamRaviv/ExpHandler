"""Plots screen callbacks: experiment/key selection → Plotly figure."""

from dash import callback, Output, Input, State, no_update as dnu
from screens.plots import make_dvnr_figure, make_odt_figure, make_vbp_figure


@callback(
    Output("plot-main", "figure"),
    Input("plots-exp-selector", "value"),
    State("store-app-state", "data"),
    prevent_initial_call=True,
)
def update_plot(selected_exps, state):
    if not selected_exps:
        from screens.plots import _empty_fig
        return _empty_fig("No experiments selected")

    project = state.get("active_project")
    data = (state.get("data") or {}).get(project) or []

    if project == "ODT":
        return make_odt_figure(data, selected_exps)
    elif project == "VBP":
        return make_vbp_figure(data, selected_exps)
    # DVNR is handled by its own callback below (needs loss-key selector too)
    return dnu


@callback(
    Output("plot-main", "figure", allow_duplicate=True),
    Input("plots-exp-selector", "value"),
    Input("plots-loss-selector", "value"),
    State("store-app-state", "data"),
    prevent_initial_call=True,
)
def update_dvnr_plot(selected_exps, selected_losses, state):
    project = state.get("active_project")
    if project != "DVNR":
        return dnu
    data = (state.get("data") or {}).get("DVNR") or []
    if not selected_exps or not selected_losses:
        from screens.plots import _empty_fig
        return _empty_fig("Select experiments and loss keys")
    return make_dvnr_figure(data, selected_exps, selected_losses)
