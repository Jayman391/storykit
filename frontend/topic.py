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
            dcc.Graph(
                id='topic-document-graph',
                config={'displayModeBar': False}
            ),
            dcc.Graph(
                id='topic-hierarchy-graph',
                config={'displayModeBar': False}
            ),
            dcc.Graph(
                id='heatmap-graph',
                config={'displayModeBar': False}
            ),
        ])
    ],
    style=card_style,
)