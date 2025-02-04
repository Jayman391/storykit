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

chatbot = dbc.Card(
    [
        dbc.CardHeader("RAG Chatbot"),
        dbc.CardBody([
            dcc.Textarea(
                id='question',
                placeholder='Ask me anything about the data!',
                style={'width': '100%', 'height': 100, 'resize': 'none'}
            ),
            dbc.Button(
                'Submit',
                id='submit-val',
                n_clicks=0,
                color="primary",
                className="mt-2",
            ),
            html.Div(id='rag-response', className="mt-3")
        ])
    ],
    style=card_style,
)