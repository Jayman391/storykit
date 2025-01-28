import dash
from dash import Dash, html, dash_table, dcc
import dash_bootstrap_components as dbc

topic = html.Div([
  dcc.Graph(id='topic-graph'),
])