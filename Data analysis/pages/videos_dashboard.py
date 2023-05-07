# --------------------------Import packeges----------------------
from dash import html, callback, Input, Output, register_page, dcc
import dash
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import pandas as pd

pio.templates.default = "ggplot2"
# ---------------------------------------------------------------


# -------------Reading data and setting constants----------------
videos_like_per_country = pio.read_json("plots/json/videos_like_per_country.json")
video_stats_growth_per_time = pio.read_json("plots/json/video_stats_growth_per_time.json")
social_accounts_affect_on_vid_stats = pio.read_json("plots/json/social_accounts_affect_on_vid_stats.json")
duration_vs_views = pio.read_json("plots/json/desc_len_vs_views_scatter_plot.json")
likes_vs_views = pio.read_json("plots/json/likes_vs_views_bubble_chart.json")
video_stats_per_tag = pio.read_json("plots/json/video_stats_per_tag.json")

    
THEME_COLORS = ["#2e2e2e", "#fc0303"]
EXERNAL_STYLESHEETS = ['assets/style.css']
TODAY = datetime.now().strftime("%Y-%m-%d")
# ---------------------------------------------------------------


# ------------------Creating the dashboard-----------------------
register_page(__name__)
# app = dash.Dash(__name__)

layout = html.Div(children= [
    
    html.Center(children= html.H2(children= [html.Span(
        "Gaming", style= {"color": THEME_COLORS[1]}), " Videos Dashboard"])),
    
    dcc.Graph(
        id='videos_like_per_country',
        figure= videos_like_per_country.update_layout(width= 610, height= 480),
        style= {'position': 'absolute', 'border': f'2px solid {THEME_COLORS[0]}','left': '10px',
                'display': 'inline-block'}, config= {'displaylogo': False}),
    
    dcc.Graph(
        id= 'social_accounts_affect_on_vid_stats',
        figure= social_accounts_affect_on_vid_stats.update_layout(width= 610, height= 980),
        style= {'position': 'absolute', 'right': '10px', 'border': f'2px solid {THEME_COLORS[0]}',
                'display': 'inline-block'}, config= {'displaylogo': False}),

    dcc.Graph(
        id= 'duration_vs_views',
        figure= duration_vs_views.update_layout(width= 610, height= 480),
        style= {'position': 'absolute', 'top': '640px', 'border': f'2px solid {THEME_COLORS[0]}',
                'display': 'inline-block', 'left': '10px'}, config= {'displaylogo': False}),
    
        html.Div(["*Double click on legend buttons to isolate lines and bars."],
             style= {'position': 'absolute', 'top': '1130px', 'height': '20',
                'display': 'inline-block', 'left': '10px'}),
    
    dcc.Graph(
        id= 'video_stats_growth_per_time',
        figure= video_stats_growth_per_time.update_layout(width= 1240, height= 500),
        style= {'position': 'absolute', 'top': '1150px', 'border': f'2px solid {THEME_COLORS[0]}',
                'display': 'inline-block', 'left': '10px'}, config= {'displaylogo': False}),
    
    dcc.Graph(
        id= 'video_stats_per_tag',
        figure= video_stats_per_tag.update_layout(width= 1240, height= 500),
        style= {'position': 'absolute', 'top': '1670px', 'border': f'2px solid {THEME_COLORS[0]}',
                'display': 'inline-block', 'left': '10px'}, config= {'displaylogo': False}),
    
    dcc.Graph(
        id= 'likes_vs_views',
        figure= likes_vs_views.update_layout(width= 1240, height= 500),
        style= {'position': 'absolute', 'top': '2190px', 'border': f'2px solid {THEME_COLORS[0]}',
                'display': 'inline-block', 'left': '10px'}, config= {'displaylogo': False}),
    
    html.Div(["By: Muhammed Ahmed Abd-Al-Aleam Elsayegh", html.Br(),
                 f"Last update: {TODAY}"],
                style= {'background-color': '#ededed', 'top': '2800px', 'position': 'absolute'})])
# ---------------------------------------------------------------