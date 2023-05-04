# --------------------------Import packeges----------------------
from dash import Dash, html, dcc, Output, Input
import plotly.graph_objects as go
from scipy import interpolate
import plotly.express as px
import numpy as np
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
GAMES = [*base_games["game"].unique()]

with open('functions/tucky.pickle', 'rb') as f:
    tucky_method = pickle.load(f)
    
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
    
    temp_df = temp_df.loc[base_games["game"] == value]
    
    outliers = tucky_method(base_games["viewCount"].to_numpy())
    outliers_indexes = np.where(np.isin(base_games["viewCount"], outliers))[0]
    temp_df = base_games.drop(outliers_indexes)
    
    temp_df.sort_values("duration_in_minutes", ascending = False, inplace= True)
    
    
    # Dropping outliers again just to make sure that there aren't any outliers
    z_scores = stats.zscore(temp_df["duration_in_minutes"])
    temp_df = temp_df[np.abs(z_scores) < 2.5]
    
    
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
    
    outliers = tucky_method(base_games["viewCount"].to_numpy())
    outliers_indexes = np.where(np.isin(base_games["viewCount"], outliers))[0]
    temp_df = base_games.drop(outliers_indexes)
    
    temp_df = temp_df.loc[base_games["game"] == value]
    temp_df["publishedAt"] = temp_df["publishedAt"].astype("datetime64[ns]")
    temp_df.sort_values(by= "publishedAt", inplace= True)
    

    video_age = datetime.strptime(TODAY, "%Y-%m-%d") - temp_df["publishedAt"]
    
    # formatting video age into hours
    temp_df["video_age"] = video_age.apply(lambda x: x.total_seconds() / 3600)
    
    z_scores = stats.zscore(temp_df["video_age"])
    temp_df = temp_df[np.abs(z_scores) < 2.7]
    
    
    
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
    
    fig.update_xaxes(
        linecolor='black',
        gridcolor='darkgrey',
        tickmode = 'array',
        tickvals = [500, 550, 600, 650, 700, 750],
        ticktext = [f'{500/24:.0f}', f'{550/24:.0f}', f'{600/24:.0f}',
                    f'{650/24:.0f}', f'{700/24:.0f}', f'{750/24:.0f}'])
    
    return fig


# ---------------------------------------------------------------


# Run it:
if __name__ == '__main__':
    app.run_server(debug=True)