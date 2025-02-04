import dash
from dash import Dash, html, dash_table, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

card_style = {
    'padding': '15px',
    'marginBottom': '20px',
    'boxShadow': '0 0.125rem 0.25rem rgba(0,0,0,0.075)',
    'border': '1px solid #dee2e6',
    'borderRadius': '0.25rem',
    'backgroundColor': '#ffffff'
}

ngram = dbc.Card(
    [
        dbc.CardHeader("Ngram Analysis"),
        dbc.CardBody(
            dbc.Row([
                dbc.Col(
                    dash_table.DataTable(
                        id='ngram-table',
                        columns=[
                            {'name': 'Ngram', 'id': 'ngram'},
                            {'name': 'Counts', 'id': 'counts'},
                            {'name': 'Ranks', 'id': 'ranks'}
                        ],
                        style_table={'height': '300px', 'overflowY': 'auto'},
                        row_selectable='multi',
                        cell_selectable=True,
                        style_cell={'textAlign': 'left'},
                        style_header={
                            'backgroundColor': '#f8f9fa', 
                            'fontWeight': 'bold'
                        },
                    ),
                    width=4
                ),
                dbc.Col(
                    dcc.Graph(
                        id='ngram-plot',
                        config={'displayModeBar': False}
                    ),
                    width=8
                ),
            ])
        ),
    ],
    style=card_style,
)
