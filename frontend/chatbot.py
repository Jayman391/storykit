import dash
from dash import Dash, html, dash_table, dcc
import dash_bootstrap_components as dbc

# Your existing chatbot layout
chatbot = html.Div([
    # Text area for user input
    dcc.Textarea(
        id='question',
        placeholder='Ask me anything about the data!',
        style={'width': '100%', 'height': 100}
    ),
    # Submit button
    html.Button('Submit', id='submit-val', n_clicks=0, style={'marginTop': '10px'}),
    # Response area
    html.Div(id='rag-response', style={'marginTop': '20px'})
])