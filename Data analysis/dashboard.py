from dash import Dash, html, dcc
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import pandas as pd

channels_data_scatter_matrix = pio.read_json("plots/json/channels_data_scatter_matrix.json")
channels_subs_growth = pio.read_json("plots/json/channels_subs_growth_per_time.json")
subs_vs_channel_name_len = pio.read_json("plots/json/subs_vs_channel_name_len_line_chart.json")
total_subs_vs_start_date = pio.read_json("plots/json/total_subs_for_channels_per_start_date.json")
channel_subs_per_country = pio.read_json("plots/json/channels_subs_per_country.json")

EXERNAL_STYLESHEETS = ['assets/style.css']

app = Dash(__name__,
           external_stylesheets= EXERNAL_STYLESHEETS)


app.layout = html.Div(children= [
    html.Center(children= html.H2(children= "## Youtube gaming analysis")),
    dcc.markdown("""this is sparta !!""")
    dcc.Graph(
        id='example-graph',
        figure= channel_subs_per_country)])

if __name__ == '__main__':
    app.run_server(debug=True)