# --------------------------Import packeges----------------------
from dash             import html, callback, Input, Output, register_page, dcc
import dash
import os
import plotly.express as px
import numpy as np
import pickle
import plotly.io as pio
from datetime import datetime
import pandas as pd

pio.templates.default = "ggplot2"
# os.chdir("Data analysis")
# ---------------------------------------------------------------


# -----------------Creating the function we need-----------------
def tucky_method(array: np.array, indecies= True) -> np.array:
    """
    This function works with any list-like numerical object
    (don't work with pandas series's) and returns the indexes
    of the found outliers in the array.
    
    :Params: Takes only the series.
    :Returns: S list of the outliers indexes.
    """
    
    Q3 = np.quantile(array, 0.75)
    Q1 = np.quantile(array, 0.25)
    IQR = Q3 - Q1
    
    upper_range = Q3 + (IQR * 1.5)
    lower_range = Q1 - (IQR * 1.5)
    
    outliers = [x for x in array if ((x < lower_range) | (x > upper_range))]
    print(f"Found {len(outliers)} outliers from {len(array)} length series!")
    
    return outliers


def z_score(array, indecies= True) -> np.array:
    """
    This function uses Z-score outlier detection method
    to detect outliers and return them into array.

    :Params: array: list-like numerical object
    :Returns: outliers: np.array (array of the outliers)
    """
    
    std: float = array.std()
    mean: float = array.mean()
    
    upper_limit = mean + (3 * std)
    lower_limit = mean - (3 * std)

    outliers = [value for value in array if 
        (value > upper_limit) | (value < lower_limit)]

    print(f"Found {len(outliers)} outliers from {len(array)} length series!")

    return outliers
# ---------------------------------------------------------------


# -------------Reading data and setting constants----------------
stacked_channels = pd.read_pickle("cleaned-files/stacked_channels.pickle")

subs_vs_channel_name_len         = pio.read_json("data-analysis/plots/json/subs_vs_channel_name_len_line_chart.json")
video_stats_per_tag              = pio.read_json("data-analysis/plots/json/video_stats_per_tag.json")
total_subs_vs_start_date         = pio.read_json("data-analysis/plots/json/total_subs_for_channels_per_start_date.json")
tags_appendings_per_vid          = pio.read_json("data-analysis/plots/json/tags_appendings_per_vid_bar_chart.json")
channel_subs_per_country         = pio.read_json("data-analysis/plots/json/channels_subs_per_country.json")
youtubers_with_social_accounts   = pio.read_json("data-analysis/plots/json/youtubers_count_with_social_accounts.json")

# with open('functions/z-score.pickle', 'rb') as f:
#     z_score = pickle.load(f)

THEME_COLORS = ["#2e2e2e", "#fc0303"]
EXERNAL_STYLESHEETS = ['assets/style.css']
TODAY = datetime.now().strftime("%Y-%m-%d")

outliers = z_score(stacked_channels["subscribers"].to_numpy())
outliers_indexes = np.array(*np.where(np.isin(stacked_channels["subscribers"], outliers)))
stacked_channels = stacked_channels.drop(outliers_indexes)

DISTANCE: int  = 5
SPACE_DISTANCE = 0
N_WIDTH: int   = 600 # normal width
L_WIDTH: int   = (N_WIDTH * 2) + DISTANCE # large width

N_HEIGHT: int  = 500
L_HEIGHT: int  = (N_HEIGHT * 2) + DISTANCE
# ---------------------------------------------------------------


# ------------------Creating the dashboard-----------------------
register_page(__name__)

layout = html.Div(children= [
    
    html.Center(children= html.H2(children= [html.Span(
        "Gaming", style= {"color": THEME_COLORS[1]}), " Channels Dashboard"])),

    
    dcc.Graph(
        id='channel_subs_per_country',
        figure= channel_subs_per_country.update_layout(width= N_WIDTH, height= N_HEIGHT),
        
        style= {'position': 'relative',
                'padding':'5px', 'border-radius': '20px', 'box-shadow': '0 3px 5px rgba(0, 0, 0, 0.3)',
                'display': 'inline-block', 'right': f'{DISTANCE}px'},
        
        config= {'displaylogo': False}),


    dcc.Graph(
        id= 'youtubers_with_social_accounts',
        figure= youtubers_with_social_accounts.update_layout(width= N_WIDTH, height= N_HEIGHT),
        
        style= {'position': 'relative',
                'padding':'5px', 'border-radius': '20px', 'box-shadow': '0 3px 5px rgba(0, 0, 0, 0.3)',
                'display': 'inline-block', 'bottom': '0px', 'left': f'{DISTANCE * 2}px'},
        
        config= {'displaylogo': False}),
    
        
    dcc.Graph(
        id= 'total_subs_vs_start_date',
        figure= total_subs_vs_start_date.update_layout(width= L_WIDTH + DISTANCE,
                                                       height= N_HEIGHT),
    
        style= {'position': 'relative',
                'padding':'5px', 'border-radius': '20px', 'box-shadow': '0 3px 5px rgba(0, 0, 0, 0.3)',
                'display': 'inline-block', 'top': f'{DISTANCE * 3}px', 'right': '0px'},
    
         config= {'displaylogo': False}),
    
    
    html.Div(["Subscribers range selector",
        dcc.RangeSlider(min= stacked_channels["subscribers"].min(),
                        max= stacked_channels["subscribers"].max(),
                        step= 100,
                        value=[10000, 200000], id= 'subs_slider',
                        marks= None, tooltip={"placement": "bottom", "always_visible": False})],
                        style= {'position': 'relative', 'display': 'inline-block', 'top': f'{DISTANCE * 4}px',
                                'width': f'{L_WIDTH}px'}),
    
    
    dcc.Graph(id='channels_subs_growth',
              style= {'position': 'relative', 
                      'padding':'5px', 'border-radius': '20px', 'box-shadow': '0 3px 5px rgba(0, 0, 0, 0.3)',
                      'display': 'inline-block', 'bottom': f'{DISTANCE}px'},
    
              config= {'displaylogo': False}),


    dcc.Graph(
        id= 'subs_vs_channel_name_len',
        figure= subs_vs_channel_name_len.update_layout(width= L_WIDTH, height= N_HEIGHT),
        style= {'position': 'relative',
                'padding':'5px', 'border-radius': '20px', 'box-shadow': '0 3px 5px rgba(0, 0, 0, 0.3)',
                'display': 'inline-block', 'top': f'{DISTANCE * 2}px'},
        
        config= {'displaylogo': False}),

    
    html.Footer(["By: Muhammed Ahmed Abd-Al-Aleam Elsayegh", html.Br(),
                 f"Last update: {TODAY}"],
                style= {'background-color': '#ededed', 'position': 'relative',
                        'padding': '3px'})])
# ---------------------------------------------------------------
    

# -----------Running the app and creating extra plot-------------
@callback(Output('channels_subs_growth', 'figure'),
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
    
    return fig.update_layout(width= L_WIDTH, height= N_HEIGHT)
# ---------------------------------------------------------------