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

kgform = dbc.Card(
    [
        dbc.CardHeader("Knowledge Graph"),
        dbc.CardBody(
            dbc.Form(
                [   
                    dbc.Row(
                        [
                            dbc.Label('Select Entities for NER Extraction'),
                            dcc.Dropdown(
                                id='ner-entities',
                                options=[
                                    {'label': 'PERSON', 'value': 'PERSON'},
                                    {'label': 'NORP', 'value': 'NORP'},
                                    {'label': 'FAC', 'value': 'FAC'},
                                    {'label': 'ORG', 'value': 'ORG'},
                                    {'label': 'GPE', 'value': 'GPE'},
                                    {'label': 'LOC', 'value': 'LOC'},
                                    {'label': 'PRODUCT', 'value': 'PRODUCT'},
                                    {'label': 'EVENT', 'value': 'EVENT'},
                                    {'label': 'WORK_OF_ART', 'value': 'WORK_OF_ART'},
                                    {'label': 'LAW', 'value': 'LAW'},
                                    {'label': 'LANGUAGE', 'value': 'LANGUAGE'},
                                    {'label': 'DATE', 'value': 'DATE'},
                                    {'label': 'TIME', 'value': 'TIME'},
                                    {'label': 'PERCENT', 'value': 'PERCENT'},
                                    {'label': 'MONEY', 'value': 'MONEY'},
                                    {'label': 'QUANTITY', 'value': 'QUANTITY'},
                                    {'label': 'ORDINAL', 'value': 'ORDINAL'},
                                    {'label': 'CARDINAL', 'value': 'CARDINAL'}
                                ],
                                value=['GPE'],
                                multi=True
                            )
                        ]
                    ),
                    dbc.Row(
                        dbc.Col(
                            dbc.Button("Submit", id="submit-kg-config", color="primary"),
                            className="mt-3"
                        )
                    )
                ]
            )
        ),
    ],
    style=card_style
)

kg = dbc.Card(
    [   
        kgform,
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