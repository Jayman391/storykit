# app.py

import numpy as np
import pandas as pd
import dash
from dash import Dash, html, dash_table, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
from flask_caching import Cache

import matplotlib  # pip install matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import json
from io import BytesIO
import base64

from collections import defaultdict
import shifterator as sh

from babycenterdb.results import Results

from frontend.query import form
from backend.query import build_query

from frontend.ngram import ngram
from backend.ngram import compute_ngrams

from frontend.sentiment import wordshift
from backend.sentiment import make_daily_sentiments_parallel, make_daily_wordshifts_parallel

from frontend.chatbot import chatbot
from backend.chatbot import initialize_global_rag, compute_rag

# Initialize the app
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Initialize raw_docs as an empty DataFrame
raw_docs = pd.DataFrame()

app.layout = html.Div([
    dcc.Store(id='raw-docs', storage_type='session'),
    dcc.Store(id='ngram-data', storage_type='session'),
    dcc.Store(id='sentiments-data', storage_type='session'),
    # Removed 'rag-llm' store as RAG is managed server-side
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
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("doc-comments-range-slider", "value"),
        Input("time-delta-slider", "value"),
        Input("text-input", "value"),
        Input("group-input", "value"),
        Input("post-or-comment", "value"),
        Input("num-documents", "value")
    ]
)
def generate_query(start_date, end_date, comments_range, time_delta, ngram_keywords, groups, post_or_comment, num_documents):
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
        'post_or_comment': post_or_comment,
        'num_documents': num_documents
    }

    # Build the query using backend function
    results = build_query(params)

    # Update raw_docs
    raw_docs_updated = results 

    # Initialize or update the RAG pipeline with new documents
    initialize_global_rag(raw_docs_updated['text'].tolist())

    return results.to_dict('records')


# Callback to update the ngram table
@app.callback(
    Output("ngram-data", "data"),
    Input("raw-docs", "data")
)
def update_ngram_table(data):
    """
    Update the ngram table with the query results.
    """
    if data is None:
        return {}
    else:
        df = pd.DataFrame.from_records(data)[['text', 'date']]
        records = df.to_dict('records')
        ngrams = compute_ngrams(records, {'keywords': ['all']})
        return ngrams


# Callback to store sentiments separately
@app.callback(
    Output("sentiments-data", "data"),
    Input("ngram-data", "data")
)
def update_sentiments(data):
    """
    Compute and store sentiments based on ngram data.
    """
    if not data:
        return {}
    else:
        sentiments = make_daily_sentiments_parallel(data.get('dates', {}))
        return sentiments


# Callback to generate and display the wordshift graph image
@app.callback(
    Output("wordshift-graph", "src"),
    Input("ngram-data", "data")
)
def update_wordshift_graph(data):
    """
    Generate the wordshift graph image and update the 'src' of the image component.
    """
    if not data:
        return ""
    else:
        try:
            shifts = make_daily_wordshifts_parallel(data.get('dates', {}))

            if not shifts:
                return ""

            # For demonstration, we'll display the first shift image
            shift = shifts[0]
            shift.plot()  # Let shift.plot() create its own figure
            fig = plt.gcf()  # Get current figure
            fig.tight_layout()  # Ensure layout is tight to prevent overlaps

            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)  # Close the figure to free memory

            return f"data:image/png;base64,{encoded}"
        except Exception as e:
            print(f"Error generating wordshift graph: {e}")
            return ""


@app.callback(
    Output("rag-response", "children"),
    [Input("submit-val", "n_clicks")],
    [State("question", "value")]
)
def update_rag_response(n_clicks, question):
    if not question:
        return dbc.Alert("Please ask a question.", color="warning")
    
    if n_clicks and n_clicks > 0:
        try:
            rag_response = compute_rag(question)  # This returns a dict with 'answer' and 'context'
            answer = rag_response.get('answer', "No answer found.")
            context = rag_response.get('context', [])

            # Format the answer
            answer_component = html.Div([
                html.H4("Answer"),
                html.P(answer)
            ], style={'marginBottom': '20px'})

            # Format the context
            if context:
                context_components = [
                    html.Li(f"Context {i+1}: {ctx}") for i, ctx in enumerate(context)
                ]
                context_component = html.Div([
                    html.H4("Context"),
                    html.Ul(context_components)
                ])
            else:
                context_component = html.Div([
                    html.H4("Context"),
                    html.P("No context available.")
                ])

            return html.Div([
                answer_component,
                context_component
            ])

        except Exception as e:
            # Log the error as needed
            return dbc.Alert(f"Error processing your request: {str(e)}", color="danger")
    
    return ""

if __name__ == "__main__":
    app.run_server(debug=True)
