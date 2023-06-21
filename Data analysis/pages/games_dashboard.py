# --------------------------Import packeges----------------------
from dash import html, callback, Input, Output, register_page, dcc
import dash
from functions import tucky_method
from functions import z_score
import plotly.graph_objects as go
from scipy import interpolate
import plotly.express as px
import numpy as np
import pytz
import pickle
import plotly.io as pio
from scipy import stats
import os
import ast
from datetime import datetime
import pandas as pd

pio.templates.default = "ggplot2"
# ---------------------------------------------------------------


# -------------Reading data and setting constants----------------

base_games = pd.read_pickle("../Cleaned files/base_games.pickle")
stacked_games = pd.read_pickle("../Cleaned files/stacked_games_df.pickle")


video_stats_per_game = pio.read_json('plots/json/video_stats_per_game.json')

THEME_COLORS = ["#2e2e2e", "#fc0303", "lightgrey"]
EXERNAL_STYLESHEETS = ['assets/style.css']
TODAY = datetime.now().strftime("%Y-%m-%d")
TODAY = pytz.utc.localize(datetime.strptime(TODAY, "%Y-%m-%d"))

GAMES = [*base_games["game"].unique()]

DISTANCE: int = 5
SPACE_DISTANCE = 0
N_WIDTH: int = 600 # normal width
L_WIDTH: int = (N_WIDTH * 2) + DISTANCE # large width

N_HEIGHT: int = 500
L_HEIGHT: int = (N_HEIGHT * 2) + DISTANCE
# ---------------------------------------------------------------


# ------------------Creating the dashboard-----------------------
register_page(__name__)

layout = html.Div(children= [

    html.Center(children= html.H2(children= ["Analysis for specfic", html.Span(
        " Games ", style= {"color": THEME_COLORS[1]}), "Videos"])),

    
    dcc.Graph(id= "video_stats_per_game",
              figure= video_stats_per_game.update_layout(width= L_WIDTH, height= N_HEIGHT),
              style= {'position': 'relative', 'padding':'5px', 'border-radius': '20px',
                      'box-shadow': '0 3px 5px rgba(0, 0, 0, 0.3)',
                      'display': 'inline-block', 'right': f'{DISTANCE * 2}px'},
              
              config= {'displaylogo': False}),

    
    html.Div(["Choose a game:",
        dcc.Dropdown(GAMES, "Minecraft" ,id= "game_dropdown")],
                     style= {'position': 'relative','padding':'5px', 'border-radius': '20px',
                             'box-shadow': '0 3px 5px rgba(0, 0, 0, 0.3)',
                            'width': f'{L_WIDTH}px', 'right': f'{DISTANCE * 2}px'}),

    
    dcc.Graph(id= "duration_vs_view",
              style= {'position': 'relative', 'display': 'inline-block', 'right': f'{DISTANCE * 2}px',
                      'width': f'{L_WIDTH}px', 'top': f'{DISTANCE * 2}px',
                      'padding':'5px', 'border-radius': '20px', 'box-shadow': '0 3px 5px rgba(0, 0, 0, 0.3)',},
              
              config= {'displaylogo': False}),
    
    
    dcc.Graph(id= "stats_growth",
          style= {'top': f'{DISTANCE * 4}px', 'position': 'relative', 'display': 'inline-block',
                  'width': f'{L_WIDTH}px', 'padding':'5px', 'border-radius': '20px',
                  'box-shadow': '0 3px 5px rgba(0, 0, 0, 0.3)',
                  'right': f'{DISTANCE * 2}px'},
              
          config= {'displaylogo': False}),
    
    
    dcc.Graph(id= "top_tags",
          style= {'top': f'{DISTANCE * 6}px', 'position': 'relative', 'display': 'inline-block',
                  'width': f'{L_WIDTH}px', 'padding':'5px', 'border-radius': '20px',
                  'box-shadow': '0 3px 5px rgba(0, 0, 0, 0.3)',
                  'right': f'{DISTANCE * 2}px'},
              
          config= {'displaylogo': False}),
    
    
    html.Div(["By: Muhammed Ahmed Abd-Al-Aleam Elsayegh", html.Br(),
             f"Last update: {TODAY}"],
            style= {'background-color': '#ededed', 'position': 'relative'})])
# ---------------------------------------------------------------


# ----------------------Making it interactive-------------------
@callback(Output('duration_vs_view', 'figure'),
          [Input('game_dropdown', 'value')])

def duration_vs_view(value):
    
    temp_df = base_games.loc[base_games["game"] == value]
    temp_df = temp_df.reset_index()
    
    outliers = tucky_method(temp_df["viewCount"].to_numpy())
    outliers_indices = np.where(np.isin(temp_df["viewCount"], outliers))[0]
    temp_df = temp_df.drop(outliers_indices)

    
    # Dropping outliers again just to make sure that there aren't any outliers
    temp_df.reset_index(inplace= True)
    outliers = tucky_method(temp_df["duration_in_minutes"].astype(float).to_numpy())
    outliers_indices = np.where(np.isin(temp_df["duration_in_minutes"], outliers))[0]
    temp_df = temp_df.drop(outliers_indices)
    
    temp_df.sort_values("duration_in_minutes", ascending = False, inplace= True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x= temp_df["duration_in_minutes"],
                             y= temp_df["viewCount"], name= "Views",
                             line= dict(color= THEME_COLORS[0], width=3, dash= 'dot')))
    
    fig.add_trace(go.Scatter(x= temp_df["duration_in_minutes"],
                         y= temp_df["likeCount"], name= "Views",
                         line= dict(color= THEME_COLORS[1], width=3, dash= 'dash')))

    fig.add_trace(go.Scatter(x= temp_df["duration_in_minutes"],
                     y= temp_df["commentCount"], name= "Likes",
                     line= dict(color= THEME_COLORS[2], width=3, dash= 'dash')))
    
    fig.update_layout(
        font_family= "Franklin Gothic",
        hovermode= "closest",
        xaxis_title= "Duration in minutes",
        yaxis_title= "Stats",
        width= L_WIDTH,
        height= N_HEIGHT,
        title=  "<span style='color: red'>Video </span>" + \
                "stats Vs video duration.")

    fig.update_xaxes(
        linecolor='black',
        gridcolor='darkgrey')

    fig.update_yaxes(
        linecolor='black',
        gridcolor='darkgrey')

    return fig


@callback(Output('stats_growth', 'figure'),
            [Input('game_dropdown', 'value')])

def stats_growth(value):
    
    temp_df = base_games.loc[base_games["game"] == value]
    temp_df = temp_df.reset_index()
    
    outliers = tucky_method(temp_df["viewCount"].to_numpy())
    outliers_indices = np.where(np.isin(temp_df["viewCount"], outliers))[0]
    temp_df = temp_df.drop(outliers_indices)

    # formatting video age into hours
    video_age = TODAY - temp_df["publishedAt"]
    temp_df["video_age"] = video_age.apply(lambda x: x.total_seconds() / 60 ** 2)
    
    # Dropping outliers again just to make sure that there aren't any outliers
    temp_df.reset_index(inplace= True)
    outliers = tucky_method(temp_df["video_age"].astype(float).to_numpy())
    outliers_indices = np.where(np.isin(temp_df["video_age"], outliers))[0]
    temp_df = temp_df.drop(outliers_indices)

    temp_df.sort_values("video_age", ascending = False, inplace= True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x= temp_df["video_age"],
                             y= temp_df["viewCount"], name= "Views",
                             line= dict(color= THEME_COLORS[1], width=3)))

    fig.update_layout(
        font_family= "Franklin Gothic",
        hovermode= "closest",
        xaxis_title= "Video age by day",
        yaxis_title= "Views",
        width= L_WIDTH,
        height= N_HEIGHT,
        title=  "Views <span style='color: red'>Growth </span>" + \
                "per game")

    fig.update_yaxes(
        linecolor='black',
        gridcolor='darkgrey')
    
    temp_df['tick_labels'] = temp_df['video_age'] / 24
    
    fig.update_xaxes(
        linecolor='black',
        gridcolor='darkgrey',
        tickmode = 'array',
        tickvals = temp_df["video_age"][::7],
        ticktext = temp_df['tick_labels'].astype(int))
    
    return fig


@callback(Output('top_tags', 'figure'), 
         [Input('game_dropdown', 'value')])
def top_tags(value):
    
    outliers = tucky_method(base_games["viewCount"].to_numpy())
    outliers_indices = np.where(np.isin(base_games["viewCount"], outliers))[0]
    temp_df = base_games.drop(outliers_indices)
    
    temp_df = temp_df.loc[temp_df["game"] == value]
    temp_df.reset_index(inplace= True)
    
    tags_dict = {"Tags": [], "Views": [], "Comments": [], "Likes": [],
                 "Count": []}
    
    total_views = []
    all_tags_temp = temp_df["tags"].explode().explode().tolist()

    for tags_chunk in all_tags_temp:

        for tag, views, comments, likes in zip(ast.literal_eval(tags_chunk), temp_df["viewCount"],
                                               temp_df["commentCount"], temp_df["likeCount"]):

            tags_dict["Tags"].append(tag)
            tags_dict["Views"].append(views)
            tags_dict["Comments"].append(comments)
            tags_dict["Likes"].append(likes)
            tags_dict["Count"].append(1)

    tags_df = pd.DataFrame({"Tag": tags_dict["Tags"],
                            "Views": tags_dict["Views"],
                            "Comments count": tags_dict["Comments"],
                            "Likes": tags_dict["Likes"],
                            "Count": tags_dict["Count"]}).groupby('Tag').sum().astype({
        "Views": np.uint32,
        "Comments count": np.uint16,
        "Count": np.uint16,
        "Likes": np.uint32})

    # Set the column tag again after we turned it into index
    tags_df["Tag"] = tags_df.index
    top_5_tags = tags_df.sort_values(by= "Count", ascending= False).head(5)

    fig = px.bar(top_5_tags, x= "Tag", y= "Count", 
                 color_discrete_sequence= [THEME_COLORS[1]])

    fig.update_layout(
        font_family= "Franklin Gothic",
        hovermode= "closest",
        title= "Top tags count")

    fig.update_xaxes(
        linecolor='black',
        gridcolor='darkgrey')

    fig.update_yaxes(
        linecolor='black',
        gridcolor='darkgrey')
    
    return  fig
# ---------------------------------------------------------------