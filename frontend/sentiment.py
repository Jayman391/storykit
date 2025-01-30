import dash
from dash import Dash, html, dash_table, dcc
import dash_bootstrap_components as dbc

wordshift = html.Div([
  dbc.Row([
    dbc.Col(id='wordshift-container', width=4),
    dbc.Col([
      dcc.Graph(id='sentiment-plot')
    ], width=8)
  ])
])


def make_img(image_src):
    return html.Img(src=image_src, style={'width': '100%'})