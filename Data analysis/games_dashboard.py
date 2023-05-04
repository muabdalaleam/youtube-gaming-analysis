# --------------------------Import packeges----------------------
from dash import Dash, html, dcc, Output, Input
import plotly.graph_objects as go
from scipy import interpolate
import plotly.express as px
import numpy as np
import pytz
import pickle
import plotly.io as pio
from scipy import stats
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

with open('functions/tucky.pickle', 'rb') as f:
    tucky_method = pickle.load(f)
    
with open('functions/z-score.pickle', 'rb') as f:
    z_score = pickle.load(f)
    
# base_games["game"] = base_games["game"].astype(str)
    
# tags_dict = {"Tags": [], "Views": [], "Comments": [], "Likes": [],
#              "Count": [], "Games": []}

# total_views = []
# all_tags_temp = base_games["tags"].explode().explode().tolist()

# for tags_chunk in all_tags_temp:
    
#     for tag, views, comments, likes, game in zip(ast.literal_eval(tags_chunk), base_games["viewCount"],
#                                                  base_games["commentCount"], base_games["likeCount"],
#                                                  base_games["game"]):

#         tags_dict["Tags"].append(tag)
#         tags_dict["Views"].append(views)
#         tags_dict["Comments"].append(comments)
#         tags_dict["Likes"].append(likes)
#         tags_dict["Count"].append(1)
#         tags_dict["Games"].append(game)

# tags_df = pd.DataFrame({"Tag": tags_dict["Tags"],
#                         "Views": tags_dict["Views"],
#                         "Comments count": tags_dict["Comments"],
#                         "Likes": tags_dict["Likes"],
#                         "Games": tags_dict["Games"],
#                         "Count": tags_dict["Count"]}).groupby('Tag').sum().astype({
#     "Views": np.uint32,
#     "Comments count": np.uint16,
#     "Count": np.uint16,
#     "Likes": np.uint32})

# tags_df["Tag"] = tags_df.index
# ---------------------------------------------------------------


# ------------------Creating the dashboard-----------------------
app = Dash(__name__,
           external_stylesheets= EXERNAL_STYLESHEETS)


app.layout = html.Div(children= [

    html.Center(children= html.H1(children= ["Analysis for specfic", html.Span(
        " Games ", style= {"color": THEME_COLORS[1]}), "Videos"])),

    dcc.Graph(id= "video_stats_per_game",
              figure= video_stats_per_game.update_layout(width= 1240, height= 500),
              style= {'position': 'absolute', 'border': f'2px solid {THEME_COLORS[0]}','left': '10px',
                      'display': 'inline-block'}),

    html.Div(["Choose a game:",
        dcc.Dropdown(GAMES, "Minecraft" ,id= "game_dropdown")],
                     style= {'top': '600px', 'position': 'absolute', 'display': 'inline-block',
                            'width': '1230px'}),

    dcc.Graph(id= "duration_vs_view",
              style= {'top': '680px', 'position': 'absolute', 'display': 'inline-block',
                      'width': '1240px', 'border': f'2px solid {THEME_COLORS[0]}'}),
    
    dcc.Graph(id= "stats_growth",
          style= {'top': '1220px', 'position': 'absolute', 'display': 'inline-block',
                  'width': '1240px', 'border': f'2px solid {THEME_COLORS[0]}'}),
    
#     dcc.Graph(id= "stats_growth",
#           style= {'top': '1220px', 'position': 'absolute', 'display': 'inline-block',
#               'width': '1240px', 'border': f'2px solid {THEME_COLORS[0]}'}),
    
    html.Div(["By: Muhammed Ahmed Abd-Al-Aleam Elsayegh", html.Br(),
             f"Last update: {TODAY}"],
            style= {'background-color': '#ededed', 'top': '2800px', 'position': 'absolute'})])
# ---------------------------------------------------------------


# ----------------------Making it interactive-------------------
@app.callback(
    Output('duration_vs_view', 'figure'),
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
        width=1240,
        height=500,
        title=  "<span style='color: red'>Video </span>" + \
                "stats Vs video duration.")

    fig.update_xaxes(
        linecolor='black',
        gridcolor='darkgrey')

    fig.update_yaxes(
        linecolor='black',
        gridcolor='darkgrey')

    return fig


@app.callback(
    Output('stats_growth', 'figure'),
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
        width=1240,
        height=500,
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
        tickvals = temp_df["video_age"][::4],
        ticktext = temp_df['tick_labels'].astype(int))
    
    return fig


# ---------------------------------------------------------------


# Run it:
if __name__ == '__main__':
    app.run_server(debug=True)