import dash
from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd

# Range slider for start and end date
date_range_slider = html.Div(
    [
        dbc.Label("Date Range", html_for="date-range"),
        dcc.DatePickerRange(
            id="date-range",
            min_date_allowed="2010-01-01",
            max_date_allowed="2022-12-31",
            start_date="2020-011-01",
            end_date="2021-01-01",
        ),
    ],
    className="mb-3",
    id="date-range-slider-container",  # Optional: Assign if you want to control visibility
)

# Range slider for number of comments in post
comments_slider = html.Div(
    [
        dbc.Label("Number of Comments in Post", html_for="doc-comments-range-slider"),
        dcc.RangeSlider(
            id="doc-comments-range-slider",
            min=0,
            max=500,
            step=10,
            value=[0, 10],  # Initialize with a range
            marks={i: str(i) for i in range(0, 501, 100)},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div(id="comments-slider-output", style={"margin-top": "10px"}),
    ],
    className="mb-3",
    id="comments-slider-container",  # Added ID
)

# Range slider for time delta
time_delta_slider = html.Div(
    [
        dbc.Label("Time Delta", html_for="time-delta-slider"),
        dcc.RangeSlider(
            id="time-delta-slider",
            min=-50,
            max=50,
            step=10,
            value=[0, 10],  # Initialize with a range
            marks={i: str(i) for i in range(-50, 51, 10)},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div(id="time-delta-output", style={"margin-top": "10px"}),
    ],
    className="mb-3",
    id="time-delta-slider-container",  # Added ID
)

# Text input for ngram keywords
ngram_input = html.Div(
    [
        dbc.Label("Enter Ngram Keywords", html_for="text-input"),
        dcc.Input(
            id="text-input",
            type="text",
            placeholder="Enter ngram keywords separated by commas",
            style={"width": "100%"},
        ),
    ],
    className="mb-3",
)

# Text input for groups
group_input = html.Div(
    [
        dbc.Label("Enter Groups", html_for="group-input"),
        dcc.Input(
            id="group-input",
            type="text",
            placeholder="Enter groups separated by commas",
            style={"width": "100%"},
        ),
    ],
    className="mb-3",
)

# Text input for number of documents
num_documents = html.Div(
    [
        dbc.Label("Number of Documents", html_for="num-documents"),
        dcc.Input(
            id="num-documents",
            type="number",
            placeholder=10,
            min=1,
            step=1,
            value=10,  # Default value
            style={"width": "100%"},
        ),
    ],
    className="mb-3",
)

# Checklist for Post or Comment
post_or_comment_checkbox = html.Div(
    [
        dbc.Label("Post or Comment", html_for="post-or-comment"),
        dcc.Checklist(
            id="post-or-comment",
            options=[
                {"label": "Post", "value": "post"},
                {"label": "Comment", "value": "comment"},
            ],
            value=["post", "comment"],  # Default selected values
            labelStyle={'display': 'inline-block', 'margin-right': '10px'},
        ),
    ],
    className="mb-3",
)



# Organize the sliders into rows with appropriate column widths using dbc.Row and dbc.Col
form = dbc.Container(
    [
        dbc.Form(
            [
                dbc.Row(
                    [
                        dbc.Col(date_range_slider, width=6),
                        dbc.Col(comments_slider, width=6),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(time_delta_slider, width=6),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(post_or_comment_checkbox, width=6),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(ngram_input, width=6),
                        dbc.Col(group_input, width=6),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(num_documents, width=6),
                    ],
                    className="mb-3",
                ),
            ]
        ),
    ],
    style={
        'margin-top': '20px',
        'width': '80%',
        'margin-left': 'auto',
        'margin-right': 'auto',
        'text-align': 'left',
    },
)