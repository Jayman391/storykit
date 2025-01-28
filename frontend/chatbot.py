import dash
from dash import Dash, html, dash_table, dcc
import dash_bootstrap_components as dbc

chatbot = html.Div([
  #text area and submit button
  #response area
  dcc.Textarea(id='question', placeholder='Ask me anything about the data!', style={'width': '100%'}),
  html.Button('Submit', id='submit-val', n_clicks=0),
  html.Div(id='rag-response-div', children=[
    html.H3('Response'),
    html.P(id='rag-response')
  ]),
])