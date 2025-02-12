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
            html.H3('Statistics'),
            dash_table.DataTable(
                id="document-stats",
                columns = [
                    {'name' : "Avg Words per Post", 'id' : "avg_post_length"},
                    {'name' : "Unique Posters", 'id' : "num_unique_posters"},
                    {'name' : "Avg Number of Comments per post", 'id' : "avg_num_comments"},
                    {'name' : "Avg Words per Comment", 'id' : "avg_comment_length"},
                    {'name' : "Unique Commenters", 'id' : "num_unique_commenters"},
                ]
            ),
        ]
    ),
    style=card_style
)