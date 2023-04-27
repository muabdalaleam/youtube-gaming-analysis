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
videos_like_per_country = pio.read_json("plots/json/videos_like_per_country.json")
video_stats_growth_per_time = pio.read_json("plots/json/video_stats_growth_per_time.json")
social_accounts_affect_on_vid_stats = pio.read_json("plots/json/social_accounts_affect_on_vid_stats.json")
duration_vs_views = pio.read_json("plots/json/desc_len_vs_views_scatter_plot.json")
likes_vs_views = pio.read_json("plots/json/likes_vs_views_bubble_chart.json")
video_stats_per_tag = pio.read_json("plots/json/video_stats_per_tag.json")


with open('functions/z-score.pickle', 'rb') as f:
    z_score = pickle.load(f)

    
THEME_COLORS = ["#2e2e2e", "#fc0303"]
EXERNAL_STYLESHEETS = ['assets/style.css']
TODAY = datetime.now().strftime("%Y-%m-%d")
# ---------------------------------------------------------------


# ------------------Creating the dashboard-----------------------
app = Dash(__name__,
           external_stylesheets= EXERNAL_STYLESHEETS)


app.layout = html.Div(children= [
    
    html.Center(children= html.H1(children= [html.Span(
        "Gaming", style= {"color": THEME_COLORS[1]}), " Videos Analysis"])),
    
    dcc.Graph(
        id='videos_like_per_country',
        figure= videos_like_per_country.update_layout(width= 610, height= 500),
        style= {'position': 'absolute', 'border': f'2px solid {THEME_COLORS[0]}','left': '10px',
                'display': 'inline-block'}),
    
    dcc.Graph(
        id= 'social_accounts_affect_on_vid_stats',
        figure= social_accounts_affect_on_vid_stats.update_layout(width= 610, height= 1020),
        style= {'position': 'absolute', 'right': '10px', 'border': f'2px solid {THEME_COLORS[0]}',
                'display': 'inline-block'}),

    dcc.Graph(
        id= 'duration_vs_views',
        figure= duration_vs_views.update_layout(width= 610, height= 500),
        style= {'position': 'absolute', 'top': '600px', 'border': f'2px solid {THEME_COLORS[0]}',
                'display': 'inline-block', 'left': '10px'}),
    
    
        html.Div(["*Double click on legend buttons to isolate lines and bars."],
             style= {'position': 'absolute', 'top': '1120px', 'height': '20',
                'display': 'inline-block', 'left': '10px'}),
    
    
    dcc.Graph(
        id= 'video_stats_growth_per_time',
        figure= video_stats_growth_per_time.update_layout(width= 1240, height= 500),
        style= {'position': 'absolute', 'top': '1140px', 'border': f'2px solid {THEME_COLORS[0]}',
                'display': 'inline-block', 'left': '10px'}),
    
    
    dcc.Graph(
        id= 'video_stats_per_tag',
        figure= video_stats_per_tag.update_layout(width= 1240, height= 500),
        style= {'position': 'absolute', 'top': '1660px', 'border': f'2px solid {THEME_COLORS[0]}',
                'display': 'inline-block', 'left': '10px'}),

    
    dcc.Graph(
        id= 'likes_vs_views',
        figure= likes_vs_views.update_layout(width= 1240, height= 500),
        style= {'position': 'absolute', 'top': '2180px', 'border': f'2px solid {THEME_COLORS[0]}',
                'display': 'inline-block', 'left': '10px'}),
    

    html.Div(["By: Muhammed Ahmed Abd-Al-Aleam Elsayegh", html.Br(),
                 f"Last update: {TODAY}"],
                style= {'background-color': '#ededed', 'top': '2800px', 'position': 'absolute'})])
# ---------------------------------------------------------------
    


# Run it:
if __name__ == '__main__':
    app.run_server(debug=True)