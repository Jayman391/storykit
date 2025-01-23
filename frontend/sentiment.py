import dash
from dash import Dash, html, dash_table, dcc
import dash_bootstrap_components as dbc

wordshift = html.Div([
  html.Img(id='wordshift-graph', style={'width': '100%'}),
])