from dash import Dash, html, dcc
import plotly.express as px
#include <>
import plotly.io as pio
from datetime import datetime
import pandas as pd

channels_data_scatter_matrix = pio.read_json("plots/json/channels_data_scatter_matrix.json")
channels_subs_growth = pio.read_json("plots/json/channels_subs_growth_per_time.json")
subs_vs_channel_name_len = pio.read_json("plots/json/subs_vs_channel_name_len_line_chart.json")
total_subs_vs_start_date = pio.read_json("plots/json/total_subs_for_channels_per_start_date.json")
youtubers_with_social_accounts = pio.read_json("plots/json/youtubers_count_with_social_accounts.json")
channel_subs_per_country = pio.read_json("plots/json/channels_subs_per_country.json")


EXERNAL_STYLESHEETS = ['assets/style.css']

app = Dash(__name__,
           external_stylesheets= EXERNAL_STYLESHEETS)


app.layout = html.Div(children= [
    
    html.Center(children= html.H1(children= [html.Span(
        "Youtube", style= {"color": "red"}), " Gaming Analysis"])),
    
    dcc.Graph(
        id='channel_subs_per_country',
        figure= channel_subs_per_country.update_layout(width= 610, height= 500),
        style= {'position': 'absolute', 'border': '2px solid #fc0303','left': '10px',
                'display': 'inline-block'}),
    
    dcc.Graph(
        id= 'total_subs_vs_start_date',
        figure= total_subs_vs_start_date.update_layout(width= 1230),
        style= {'position': 'absolute', 'left': '10px', 'top': '600px', 'border': '2px solid #fc0303',
                'display': 'inline-block'}),

    dcc.Graph(
        id= 'youtubers_with_social_accounts',
        figure= youtubers_with_social_accounts.update_layout(width= 610, height= 500),
        style= {'position': 'absolute','right': '10px', 'border': '2px solid #fc0303',
                'display': 'inline-block'}),

    dcc.Graph(
        id= 'channels_subs_growth',
        figure= channels_subs_growth.update_layout(width= 1230,
                                                   xaxis= dict(rangeslider= dict(
                                                   visible= True))),
        style= {'position': 'absolute','left': '10px', 'border': '2px solid #fc0303', 'top': '1020px',
                'display': 'inline-block'})])

@app.callback(
    Output('output-container-range-slider', 'children'),
    [Input('my-range-slider', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)


if __name__ == '__main__':
    app.run_server(debug=True)