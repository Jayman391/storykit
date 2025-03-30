import dash
from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd

card_style = {
    'padding': '15px',
    'marginBottom': '20px',
    'boxShadow': '0 0.125rem 0.25rem rgba(0,0,0,0.075)',
    'border': '1px solid #dee2e12',
    'borderRadius': '0.25rem',
    'backgroundColor': '#ffffff',
    'margin': 'auto'
}
stats = dbc.Card(
    dbc.CardBody(
        [
            html.H3('Query Statistics'),
            dash_table.DataTable(
                id="document-stats",
                data=[
                    {'Metric': "Avg Words per Post", 'Value': ""},
                    {'Metric': "Unique Posters", 'Value': ""},
                    {'Metric': "Avg Number of Comments per post", 'Value': ""},
                    {'Metric': "Avg Words per Comment", 'Value': ""},
                    {'Metric': "Unique Commenters", 'Value': ""},
                ],
                columns=[
                    {'name': "Metric", 'id': "Metric"},
                    {'name': "Value", 'id': "Value"},
                ],
                style_cell={
                    'fontSize': '12px',     
                    'textAlign': 'left',
                },
            ),
        ]
    ),
    style=card_style
)