import dash
from dash import Dash, html, dash_table, dcc
import dash_bootstrap_components as dbc

card_style = {
    'padding': '15px',
    'marginBottom': '20px',
    'boxShadow': '0 0.125rem 0.25rem rgba(0,0,0,0.075)',
    'border': '1px solid #dee2e6',
    'borderRadius': '0.25rem',
    'backgroundColor': '#ffffff'
}

wordshift = dbc.Card(
    [
        dbc.CardHeader("Sentiment Analysis"),
        dbc.CardBody([
            dbc.Row([ 
                dbc.Col(
                    id='wordshift-container', width=4,  
                ),
                dbc.Col([
                    dcc.Graph(
                        id='sentiment-plot',
                        config={'displayModeBar': False}
                    ),
                ], width=8),
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Button(
                        "Download Wordshift Plot", 
                        id="download-wordshift-button", 
                        color="secondary",
                        size="sm"
                    ),
                ),
                dbc.Col(
                    dbc.Button(
                        "Download Sentiment Timeseries", 
                        id="download-sentiment-button", 
                         color="secondary",
                        size="sm"
                    ),
                ), 
            ])
        ]),
    ],
    style=card_style,
)
