# --------------------------Import packeges----------------------
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import numpy as np
import pickle
import plotly.io as pio
from datetime import datetime
import pandas as pd

pio.templates.default = "ggplot2"
# ---------------------------------------------------------------


# -------------Reading data and setting constants----------------
stacked_channels = pd.read_pickle("../Cleaned files/stacked_channels.pickle")

subs_vs_channel_name_len = pio.read_json("plots/json/subs_vs_channel_name_len_line_chart.json")
video_stats_per_tag = pio.read_json("plots/json/video_stats_per_tag.json")
tags_appendings_per_vid = pio.read_json("plots/json/tags_appendings_per_vid_bar_chart.json")
channel_subs_per_country = pio.read_json("plots/json/channels_subs_per_country.json")

with open('functions/z-score.pickle', 'rb') as f:
    z_score = pickle.load(f)

THEME_COLORS = ["#2e2e2e", "#fc0303"]
EXERNAL_STYLESHEETS = ['assets/style.css']
TODAY = datetime.now().strftime("%Y-%m-%d")

outliers = z_score(stacked_channels["subscribers"].to_numpy())
outliers_indexes = np.array(*np.where(np.isin(stacked_channels["subscribers"], outliers)))
stacked_channels = stacked_channels.drop(outliers_indexes)
# ---------------------------------------------------------------


# ------------------Creating the dashboard-----------------------
app = Dash(__name__,
           external_stylesheets= EXERNAL_STYLESHEETS)

app.layout = html.Div(children= [
    
    html.Center(children= html.H1(children= [html.Span(
        "Gaming", style= {"color": THEME_COLORS[1]}), " Channels Analysis"])),
    
    dcc.Graph(
        id='channel_subs_per_country',
        figure= channel_subs_per_country.update_layout(width= 610, height= 500),
        style= {'position': 'absolute', 'border': f'2px solid {THEME_COLORS[0]}','left': '10px',
                'display': 'inline-block'}),
    
    dcc.Graph(
        id= 'total_subs_vs_start_date',
        figure= total_subs_vs_start_date.update_layout(width= 1230),
        style= {'position': 'absolute', 'left': '10px', 'top': '600px', 'border': f'2px solid {THEME_COLORS[0]}',
                'display': 'inline-block'}),

    dcc.Graph(
        id= 'youtubers_with_social_accounts',
        figure= youtubers_with_social_accounts.update_layout(width= 610, height= 500),
        style= {'position': 'absolute','right': '10px', 'border': f'2px solid {THEME_COLORS[0]}',
                'display': 'inline-block'}),
    
    
    dcc.Graph(id='channels_subs_growth',
              style= {'position': 'absolute', 'border': f'2px solid {THEME_COLORS[0]}', 'left': '10px',
                      'display': 'inline-block', 'top': '1050px'}),

    
    html.Div(["Subscribers range selector",
        dcc.RangeSlider(min= stacked_channels["subscribers"].min(),
                        max= stacked_channels["subscribers"].max(),
                        step= 100,
                        value=[10000, 200000], id= 'subs_slider',
                        marks= None, tooltip={"placement": "bottom", "always_visible": False})],
                        style= {'top': '1015px', 'position': 'absolute', 'display': 'inline-block',
                                'width': '1230px'}),

    
    dcc.Graph(
        id= 'subs_vs_channel_name_len',
        figure= subs_vs_channel_name_len.update_layout(width= 1230, height= 400),
        style= {'position': 'absolute','right': '20px', 'border': f'2px solid {THEME_COLORS[0]}',
                'display': 'inline-block', 'top': '1468px'}),

    html.Footer(["By: Muhammed Ahmed Abd-Al-Aleam Elsayegh", html.Br(),
                 f"Last update: {TODAY}"],
                style= {'background-color': '#ededed', 'position': 'absolute', 'top': '1900px',
                        'padding': '3px'})])
# ---------------------------------------------------------------
    

# ---------------Running the app and creating slider-------------
@app.callback(
    Output('channels_subs_growth', 'figure'),
    [Input('subs_slider', 'value')])

def plot_channels_subs_growth(value):
    
    time_df = stacked_channels.copy()


    time_df = time_df[(time_df["subscribers"] > value[0]) &
                      (time_df["subscribers"] < value[1])]

    fig = px.line(time_df, x= "Collecting date", y= "subscribers",
                  color_discrete_sequence= px.colors.sequential.Reds,
                  color= "channel_name",
                  hover_name= "channel_name")


    fig.update_layout(
        font_family= "Franklin Gothic",
        hovermode= "closest",
        width=1000,
        height=400,
        title= "Channels subscribers growth per time.<br>"+\
               "<sub>*Youtube API down cast subscribers count</sub><br>")

    fig.update_xaxes(
        linecolor='black',
        gridcolor='darkgrey')

    fig.update_yaxes(
        linecolor='black',
        gridcolor='darkgrey')
    
    return fig.update_layout(width= 1230)
# ---------------------------------------------------------------

# Run it:
if __name__ == '__main__':
    app.run_server(debug=True)