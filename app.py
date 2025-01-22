# Import packages
import numpy as np
import pandas as pd
import dash
from dash import Dash, html, dash_table, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from babycenterdb.results import Results

from frontend.query import form
from backend.query import build_query
from frontend.wordshift import wordshift
from frontend.ngram import ngram
from frontend.chatbot import chatbot
# Initialize the app

app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# create global variable for query data
query_data = pd.DataFrame()

app.layout = html.Div([
    form,
    ngram,      # Ensure 'ngram' is correctly defined/imported
    wordshift,  # Ensure 'wordshift' is correctly defined/imported
    chatbot,    # Ensure 'chatbot' is correctly defined/imported
])

# Callback to control visibility of comment slider and time delta slider
@app.callback(
    [
        Output("time-delta-slider-container", "style"),
        Output("comments-slider-container", "style")
    ],
    [Input("post-or-comment", "value")]
)
def toggle_sliders(post_or_comment_value):
    """
    Toggle visibility of 'time-delta-slider-container' and 'comments-slider-container'
    based on the selected values in 'post-or-comment' checklist.
    """
    # Show the comment slider if 'post' is selected
    comments_style = {"display": "block"} if "post" in post_or_comment_value else {"display": "none"}

    # Show the time delta slider if 'comment' is selected
    time_delta_style = {"display": "block"} if "comment" in post_or_comment_value else {"display": "none"}

    return time_delta_style, comments_style

# Callback to generate query and update the query table
@app.callback(
    Output("query-table", "data"),
    Output("query-table", "columns"),
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("doc-comments-range-slider", "value"),
        Input("time-delta-slider", "value"),
        Input("text-input", "value"),
        Input("group-input", "value"),
        Input("post-or-comment", "value")
    ]
)
def generate_query(start_date, end_date, comments_range, time_delta, ngram_keywords, groups, post_or_comment):
    """
    Generate query results based on form inputs and update the query table.
    """
    # Build the query parameters
    params = {
        'start_date': start_date,
        'end_date': end_date,
        'comments_range': comments_range,
        'time_delta': time_delta,
        'ngram_keywords': ngram_keywords,
        'groups': groups,
        'post_or_comment': post_or_comment
    }

    # Execute the query (assuming build_query returns a pandas DataFrame)
    results = build_query(params)

    query_data = results

    return results.sample(200).to_dict('records'), [{"name": i, "id": i} for i in results.columns]

# Ensure all components are correctly defined/imported before running the server

if __name__ == "__main__":
    app.run_server(debug=True)