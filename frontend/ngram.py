import dash
from dash import Dash, html, dash_table, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

ngram = html.Div([
    html.H3("Ngram Analysis"),
    html.Div([
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id='ngram-table',
                    columns=[
                        {'name': 'Ngram', 'id': 'ngram'},
                        {'name': 'Counts', 'id': 'counts'},
                        {'name': 'Ranks', 'id': 'ranks'}
                    ],
                    style_table={'height': '300px', 'overflowY': 'auto'},
                    # Allow selection or "clicking" on a row:
                    row_selectable='multi',    # or 'multi' if you prefer
                    cell_selectable=True        # so active_cell events can fire
                )
            ], width=4),
            dbc.Col([
                dcc.Graph(id='ngram-plot'),
            ], width=8),
        ]),
    ])
])
