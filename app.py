# Import packages
import numpy as np
import pandas as pd
import dash
from dash import Dash, html, dash_table, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from flask_caching import Cache
from babycenterdb.results import Results

from frontend.query import form
from backend.query import build_query

from frontend.ngram import ngram
from backend.ngram import compute_ngrams

from frontend.wordshift import wordshift
from frontend.chatbot import chatbot
# Initialize the app

app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

app.layout = html.Div([
    dcc.Store(id='raw-docs',storage_type='session'),
    dcc.Store(id='ngram-data',storage_type='session'),
    form,
    dbc.Accordion([
        dbc.AccordionItem([
            ngram,      
        ], title="Ngram Analysis",),
        dbc.AccordionItem([
            wordshift,  
        ], title="Sentiment Analysis",),
        dbc.AccordionItem([
            chatbot,  
        ], title="RAG",),
    ]),
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
        Output("raw-docs", "data"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("doc-comments-range-slider", "value"),
        Input("time-delta-slider", "value"),
        Input("text-input", "value"),
        Input("group-input", "value"),
        Input("post-or-comment", "value")
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

    results = build_query(params)

    return results.sample(200).to_dict('records')
    
# read in the data from raw-docs
@app.callback(
    Output("ngram-data", "data"),
    Input("raw-docs", "data")
)
def update_ngram_table(data):
    """
    Update the ngram table with the query results.
    """
    if data is None:
        return []
    else:

        df = pd.DataFrame.from_records(data)[['text', 'date']]
 
        records = df.to_dict('records')

        ngrams = compute_ngrams(records, {'keywords': ['all']})
 
        return ngrams


if __name__ == "__main__":
    app.run_server(debug=True)