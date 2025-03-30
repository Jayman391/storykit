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

gramradio = dbc.Col(
    [
        dbc.Label('Select N-gram'),
        dbc.RadioItems(
            id='gram-radio',
            options=[
                {'label': '1-gram', 'value': 1},
                {'label': '2-gram', 'value': 2},
                {'label': '3-gram', 'value': 3}
            ],
            value=1,
            inline=True
        ),
    ],
    className="mb-3"
)

smoothingslider = dbc.Col(
    [ 
        dbc.Label('Ngram Smoothing Window (number of days)'),
        dcc.Slider(
            id='smoothing-slider',
            min=1,
            max=20,
            value=1,
            step=1,
            marks={1:'1',20:'20'},
            tooltip={'placement': 'bottom', 'always_visible': True}
        )
    ],
    className="mb-3"
)

ngramform = dbc.Card(
    [
        dbc.CardHeader("Ngram Analysis"),
        dbc.CardBody(
            dbc.Form(
                [
                    dbc.Row([gramradio, smoothingslider]),
                    dbc.Row(
                        dbc.Col(
                            dbc.Button("Submit", id="submit-ngram-config", color="primary"),
                            className="mt-3"
                        )
                    ),
                ]
            )
        ),
    ],
    style=card_style
)


ngram = dbc.Card(
    [
        ngramform,
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
                        export_format="csv",
                        filter_action='native',
                    ),
                    width=4
                ),
                dbc.Col([
                    dcc.Graph(
                        id='ngram-plot',
                        config={'displayModeBar': False}
                    ),
                    dbc.Button(
                        "Download Timeseries", 
                        id="download-timeseries-button", 
                        color="secondary",
                    )
                ], width=8),
               
            ])
        ),
    ],
    style=card_style,
)
