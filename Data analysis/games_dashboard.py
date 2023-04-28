# --------------------------Import packeges----------------------
from dash import Dash, html, dcc, Output, Input
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pickle
import plotly.io as pio
import ast
from datetime import datetime
import pandas as pd

pio.templates.default = "ggplot2"
# ---------------------------------------------------------------


# -------------Reading data and setting constants----------------
base_games = pd.read_pickle("../Cleaned files/base_games.pickle")
stacked_games = pd.read_pickle("../Cleaned files/stacked_games_df.pickle")


video_stats_per_game = pio.read_json('plots/json/video_stats_per_game.json')

THEME_COLORS = ["#2e2e2e", "#fc0303"]
EXERNAL_STYLESHEETS = ['assets/style.css']
TODAY = datetime.now().strftime("%Y-%m-%d")
GAMES = [*base_games["game"].unique()]

with open('functions/z-score.pickle', 'rb') as f:
    z_score = pickle.load(f)
    
base_games["game"] = base_games["game"].astype(str)
    
tags_dict = {"Tags": [], "Views": [], "Comments": [], "Likes": [],
             "Count": [], "Games": []}

total_views = []
all_tags_temp = base_games["tags"].explode().explode().tolist()

for tags_chunk in all_tags_temp:
    
    for tag, views, comments, likes, game in zip(ast.literal_eval(tags_chunk), base_games["viewCount"],
                                                 base_games["commentCount"], base_games["likeCount"],
                                                 base_games["game"]):

        tags_dict["Tags"].append(tag)
        tags_dict["Views"].append(views)
        tags_dict["Comments"].append(comments)
        tags_dict["Likes"].append(likes)
        tags_dict["Count"].append(1)
        tags_dict["Games"].append(game)

tags_df = pd.DataFrame({"Tag": tags_dict["Tags"],
                        "Views": tags_dict["Views"],
                        "Comments count": tags_dict["Comments"],
                        "Likes": tags_dict["Likes"],
                        "Games": tags_dict["Games"],
                        "Count": tags_dict["Count"]}).groupby('Tag').sum().astype({
    "Views": np.uint32,
    "Comments count": np.uint16,
    "Count": np.uint16,
    "Likes": np.uint32})

# Set the column tag again after we turned it into index
tags_df["Tag"] = tags_df.index
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

    dcc.Graph(id= "tags_per_game",
              style= {'top': '1120px', 'position': 'absolute', 'display': 'inline-block',
                      'width': '1230px', 'border': f'2px solid {THEME_COLORS[0]}'}),
    
    html.Div(["By: Muhammed Ahmed Abd-Al-Aleam Elsayegh", html.Br(),
             f"Last update: {TODAY}"],
            style= {'background-color': '#ededed', 'top': '2800px', 'position': 'absolute'})])
# ---------------------------------------------------------------


# ----------------------Making it interactive-------------------
@app.callback(
    Output('tags_per_game', 'figure'),
    [Input('game_dropdown', 'value')])

def top_tags_per_game(value):
    
    top_20_tags = tags_df.where(tags_df["Games"] == value).sort_values(
        by= "Count", ascending= False).head(20)
    
    fig = go.Figure(go.Bar(
        x= top_20_tags["Count"],
        y= top_20_tags["Tag"],
        name= top_20_tags["Tag"],
        orientation='h',
        marker=dict(
            color= THEME_COLORS[0],
            line=dict(color= THEME_COLORS[1], width=3))))
    
    fig.update_layout(
        font_family= "Franklin Gothic",
        hovermode= "closest",
        width=1240,
        height=500,
        title=  "<span style='color: red'>Tags </span>" + \
                "count")
    
    fig.update_xaxes(
        linecolor='black',
        gridcolor='darkgrey')
    
    fig.update_yaxes(
        linecolor='black',
        gridcolor='darkgrey')
    
    return fig

# ---------------------------------------------------------------


# Run it:
if __name__ == '__main__':
    app.run_server(debug=True)