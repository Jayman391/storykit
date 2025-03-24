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

topic = dbc.Card(
    [
        dbc.CardHeader("Topic Modeling"),
        dbc.CardBody([

            dbc.Row([
                dbc.Button(
                    "Download Topics",
                    id="download-topics-button",
                    color="secondary",
                ),
            ]),
            
            dbc.Row([
                dcc.Graph(
                    id='topic-document-graph',
                    config={'displayModeBar': False}
                ),
                dbc.Button(
                    "Download Documents", 
                    id="download-documents-button", 
                    color="secondary",
                )
            ]),
            
            dbc.Row([
                dcc.Graph(
                    id='topic-hierarchy-graph',
                    config={'displayModeBar': False}
                ),
                dbc.Button(
                    "Download Hierarchy", 
                    id="download-hierarchy-button", 
                    color="secondary",
                )
            ]),
            
            dbc.Row([
                dcc.Graph(
                    id='heatmap-graph',
                    config={'displayModeBar': False}
                ),
                dbc.Button(
                    "Download Heatmap", 
                    id="download-heatmap-button", 
                    color="secondary",
                )
            ]),

            dbc.Row([
                dcc.Graph(
                    id='tot-graph',
                    config={'displayModeBar': False}
                ),
                dbc.Button(
                    "Download Topics over Time", 
                    id="download-tot-button", 
                    color="secondary",
                )
            ]),
        ]),
    ],
    style=card_style,
)