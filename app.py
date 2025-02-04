# app.py

import numpy as np
import pandas as pd
import dash
from dash import Dash, html, dash_table, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
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

from frontend.stats import stats
from backend.stats import compute_statistics

from frontend.ngram import ngram
from backend.ngram import compute_ngrams

from frontend.sentiment import wordshift, make_img
from backend.sentiment import make_daily_sentiments_parallel, generate_wordshift_for_date 

from frontend.chatbot import chatbot
from backend.chatbot import initialize_global_rag, compute_rag

from frontend.topic import topic
from backend.topic import fit_topic_model, visualize_documents, visualize_hierarchy, visualize_heatmap

from datetime import datetime


# Initialize the app
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Initialize raw_docs as an empty DataFrame
raw_docs = pd.DataFrame()

navbar = dbc.Navbar(
    dbc.Container(
        dbc.NavbarBrand("BabyCenter Dashboard", className="mx-auto")
    ),
    color="primary",
    dark=True,
    className="mb-4"
)
app.layout = html.Div([
    # Store components for managing session data
    dcc.Store(id='raw-docs', storage_type='session'),
    dcc.Store(id='ngram-data', storage_type='session'),
    dcc.Store(id='sentiments-data', storage_type='session'),
    # Navbar at the top
    navbar,

    dbc.Container([
        dbc.Row([
            dbc.Col(form, width=3),
            dbc.Col(
                dbc.Accordion(
                    [   
                        dbc.AccordionItem(stats, title="Query Statistics"),
                        dbc.AccordionItem(ngram, title="Ngram Analysis"),
                        dbc.AccordionItem(wordshift, title="Sentiment Analysis"),
                        dbc.AccordionItem(topic, title="Topic Modeling"),
                        dbc.AccordionItem(chatbot, title="RAG Chatbot"),
                    ],
                    start_collapsed=True,
                    flush=True
                ),
                width=9
            )
        ], className="mb-4")
    ], fluid=True)
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

@app.callback(
    Output("raw-docs", "data"),
    Input("submit-query-button", "n_clicks"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("doc-comments-range-slider", "value"),
    State("time-delta-slider", "value"),
    State("text-input", "value"),
    State("group-input", "value"),
    State("post-or-comment", "value"),
    State("num-documents", "value"),
)
def generate_query(n_clicks,
                   start_date, end_date,
                   comments_range, time_delta,
                   ngram_keywords, groups,
                   post_or_comment, num_documents):
    """
    Generate query results based on form inputs, but only when the user
    clicks the "Submit Query" button.
    """
    # If the button hasn't been clicked yet, do nothing
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

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

    # Build and run the query
    results = build_query(params)

    # Convert results to a dataframe if needed; or directly
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

# Callback to update the sentiment plot with chronological dates
@app.callback(
    Output("sentiment-plot", "figure"), 
    Input("sentiments-data", "data")
)
def update_sentiment_plot(data):
    if not data:
        return {}
    else:
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame.from_dict(data, orient='index', columns=['sentiment'])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()  # Ensure chronological order
         # Create a line plot with interactive selection
        fig = px.line(df, x=df.index, y='sentiment', title='Daily Sentiment')
        fig.update_layout(clickmode='event+select')
        return fig

# Callback to generate and display the wordshift graph for a selected date
@app.callback(
    Output("wordshift-container", "children"),
    Input("sentiment-plot", "clickData"),
    State("ngram-data", "data")
)
def update_wordshift_graph(clickData, ngram_data):
    """
    Generate the wordshift graph for the selected date and display it.
    """
    if not clickData or not ngram_data:
        return "Click on a date in the sentiment plot to see the wordshift graph."

    try:
        # Extract the selected date
        selected_date = clickData['points'][0]['x']

        selected_date = format_date(str(selected_date))

        # Generate wordshift for the selected date
        shift = generate_wordshift_for_date(selected_date, ngram_data.get('dates', {}))
        if shift is None:
            return f"No wordshift data available for {selected_date}."

        # Plot the wordshift graph
        shift.plot()
        fig = plt.gcf()
        fig.tight_layout()

        # Convert plot to base64 image
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # Create the image component
        image = html.Img(src=f"data:image/png;base64,{encoded}", style={'width': '100%'})

        return image

    except Exception as e:
        print(f"Error generating wordshift graph: {e}")
        return "An error occurred while generating the wordshift graph."

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

@app.callback(
    [
        Output("topic-document-graph", "figure"),
        Output("topic-hierarchy-graph", "figure"),
        Output("heatmap-graph", "figure")
    ],
    Input("raw-docs", "data")
)
def topic_model(data):
    """
    Fit a topic model and return the visualization.
    """
    if not data:
        return {}

    docs = pd.DataFrame.from_records(data)['text'].tolist()
    dates = pd.DataFrame.from_records(data)['date'].tolist()
    dates = [datetime.strptime(x, "%Y-%m-%dT%H:%M:%S") for x in dates]
    print(len(dates))
    print(len(docs))
    topic_model, _, _ = fit_topic_model(docs)
    docs = visualize_documents(topic_model, docs)
    hierarchical_topics = visualize_hierarchy(topic_model)
    topics_over_time = visualize_heatmap(topic_model)

    return docs, hierarchical_topics, topics_over_time

@app.callback(
    Output("ngram-table", "data"),
    Input("ngram-data", "data")
)
def update_ngram_table(data):
    """
    Update the ngram table with the query results.
    """
    if data is None:
        return []
    
    full_corpus = data.get('full_corpus', {})
    
    table_data = []
    # full_corpus has structure: {"1-gram": {counts: {...}, ranks: {...}}, "2-gram": {...}, ...}
    for ngram_size, info_dict in full_corpus.items():
        counts_dict = info_dict.get('counts', {})
        ranks_dict  = info_dict.get('ranks', {})
        
        for ngram_text, count_val in counts_dict.items():
            rank_val = ranks_dict.get(ngram_text, None)
            # Create a row for the DataTable
            row = { 
                'ngram': ngram_text,   
                'counts': count_val,   
                'ranks': rank_val      
            }
            table_data.append(row)
    
    # Sort or manipulate as needed
    # For instance, you might want to sort the table by descending counts:
    table_data = sorted(table_data, key=lambda x: x['counts'], reverse=True)
    
    return table_data   


@app.callback(
    Output("ngram-plot", "figure"),
    [
        Input("ngram-data", "data"),
        Input("ngram-table", "data"),         # The full table data
        Input("ngram-table", "selected_rows") # Which rows are selected
    ]
)
def update_ngram_plot(ngram_data, table_data, selected_rows):
    """
    Update the ngram plot with the time series of RANKS for each selected ngram.
    The user can hide/deselect each trace by clicking it in the legend.
    """
    # If there's no ngram_data yet, return an empty figure
    if not ngram_data:
        return go.Figure()

    dates_dict = ngram_data.get('dates', {})
    
    # Create a blank figure
    fig = go.Figure()

    # If nothing is selected, just show a blank figure
    if not selected_rows:
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Rank",
            yaxis=dict(autorange="reversed")  # Ranks: 1 is at the top
        )
        return fig
    
    # For each selected row index, grab its corresponding row data
    for row_idx in selected_rows:
        row_data = table_data[row_idx]
        ngram_text = row_data['ngram']

        x_vals = []
        y_vals = []

        # Sort dates so lines go from earliest to latest
        sorted_dates = list(dates_dict.keys())
        sorted_dates.sort(key=lambda x: datetime.strptime(x, "%a, %d %b %Y 00:00:00"))

        # For each date, see if that ngram appears and gather its rank
        for date_str in sorted_dates:
            # date_str => something like "Mon, 20 Apr 2020 00:00:00"
            # date_obj => {"1-gram": {...}, "2-gram": {...}, ...}
            date_obj = dates_dict[date_str]

            # We don't know which n-gram size the user clicked, so search them all
            found = False
            for ngram_size, size_info in date_obj.items():
                rank_dict = size_info.get('ranks', {})
                if ngram_text in rank_dict:
                    x_vals.append(date_str)
                    y_vals.append(rank_dict[ngram_text])
                    found = True
                    break
            # If the ngram wasn't found for a given date, it's simply not plotted for that date

        # Add a new Scatter trace for this ngram
        fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=y_vals, 
                mode='lines+markers',
                name=ngram_text
            )
        )

    # Invert y-axis so rank #1 is at the top
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Rank",
        yaxis=dict(autorange="reversed")
    )

    return fig

@app.callback(
    Output("document-stats", "data"),
    Input("raw-docs", "data")
)
def update_document_stats(data):
    """
    Update the document stats table with the query results.
    """
    if data is None:
        return []
    
    df = pd.DataFrame.from_records(data)
    stats = compute_statistics(df)
    
    return stats


def format_date(date_str):
    # Parse the input date string
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Format the date to the desired format
    formatted_date = date_obj.strftime("%a, %d %b %Y 00:00:00")
    
    return formatted_date

if __name__ == "__main__":
    app.run_server(debug=True)
