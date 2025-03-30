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

kg = dbc.Card(
    [
        dbc.CardHeader("NER Knowledge Graph"),
        dbc.CardBody([
            
            dbc.Row([
                dcc.Graph(
                    id='knowledge-graph',
                    config={'displayModeBar': False}
                ),
                dbc.Button(
                    "Download Graph", 
                    id="download-graph-button", 
                    color="secondary",
                )
            ]),
            
        ])
    ],
    style=card_style

)