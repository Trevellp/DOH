# multi_dashboard.py
from dash import Dash, html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

# 1) Create the ONE app first
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server
app.title = "Substance Use Dashboards"

# 2) Now import submodules (their @callback functions will attach to this app)
import app_alt as alt      # exposes alt.layout and its @callback functions
import polysubstance_dashboard as poly
# Optional co-occurring page (safe import)
try:
    import cooccurring_dashboard_db as co
    HAS_CO = True
except Exception:
    HAS_CO = False

# 3) Build the tabs using the exported layouts
tabs = [
    dcc.Tab(label="Discharges related to substance use", value="alt",
            className="tab", selected_className="tab--selected"),
    dcc.Tab(label="Related to polysubstance use", value="poly",
            className="tab", selected_className="tab--selected"),
]
if HAS_CO:
    tabs.append(
        dcc.Tab(label="Co-occurring: SUD × MH (secondary)", value="co",
                className="tab", selected_className="tab--selected")
    )

app.layout = html.Div(
    [dcc.Tabs(id="view-tabs", value="alt", children=tabs, className="tabs"),
     html.Div(id="view-container", style={"marginTop": "12px"})]
)

@callback(Output("view-container", "children"), Input("view-tabs", "value"))
def render_view(value):
    if value == "poly":
        return poly.layout
    if value == "co" and HAS_CO:
        return co.layout
    return alt.layout

if __name__ == "__main__":
    app.run(debug=True)

