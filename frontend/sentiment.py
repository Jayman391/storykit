import dash
from dash import Dash, html, dash_table, dcc
import dash_bootstrap_components as dbc

wordshift = html.Div([
  dbc.Row([
    dbc.Col([
      dbc.Carousel(
        id='wordshift-carousel',
        controls=True,
        items = []
      )
    ], width=8),
    dbc.Col([
      dcc.Graph(id='sentiment-plot')
    ], width=4)
  ])
])