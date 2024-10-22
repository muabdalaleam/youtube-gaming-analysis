from dash import Dash, html, dcc, register_page
import dash


THEME_COLORS = ["#2e2e2e", "#fc0303"]
app = Dash(__name__, use_pages=True)
server = app.server

app._favicon = ("logo.png")


for i, page in enumerate(dash.page_registry.values()):
    
    if i == 0:
        channels_dashboard = page

    elif i == 1:
        games_dashboard = page

    elif i == 2:
        videos_dashboard = page

# set the default landing page

def create_button(dashboard: str) -> dcc.Link:
    """This function just help me to not repeat my code by
    creating customized buttons without rewriting the code."""
    
    button = dcc.Link(f"{dashboard['name']}",
             href= f"{dashboard['relative_path']}",
             style= {"background-color": THEME_COLORS[0], "padding": "15px 25px",
                     "color": "white", "text-decoration": "none", "position": "relative",
                     'margin': '5px', "top": "30px", "border-radius": "25px"})

    return  button


app.layout = html.Div([
    
    html.Center([html.H1(["Youtube ", html.Span("Gaming", style= {"color": THEME_COLORS[1]}),
                          " Analysis Dashboards"], style= {'font-size': '45px'}),
                 
    dcc.Markdown("""Welcome To My YouTube Gaming Analysis 👋""", style= {'font-size': '25px'}),


    create_button(channels_dashboard),

    create_button(games_dashboard),

    create_button(videos_dashboard),

    html.Hr(style= {"top": "70px", "position": "relative", 'border-top': '1px dashed black',
                    'width': '650px'}),

    html.Div([dash.page_container], style= {"top": "80px", 
                                            "position": "relative"})])])

if __name__ == '__main__':
    app.run_server(debug=True)