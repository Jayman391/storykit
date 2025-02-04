import dash
from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd

card_style = {
    'padding': '15px',
    'marginBottom': '20px',
    'boxShadow': '0 0.125rem 0.25rem rgba(0,0,0,0.075)',
    'border': '1px solid #dee2e12',
    'borderRadius': '0.25rem',
    'backgroundColor': '#ffffff',
    'margin': 'auto'
}

date_range_slider = html.Div(
    [
        dbc.Label("Date Range", html_for="date-range"),
        dcc.DatePickerRange(
            id="date-range",
            min_date_allowed="2010-01-01",
            max_date_allowed="2022-12-31",
            start_date="2019-01-01",
            end_date="2021-01-01",
        ),
    ],
    className="mb-3",
    id="date-range-slider-container",
)

comments_slider = html.Div(
    [
        dbc.Label("Number of Comments in Post", html_for="doc-comments-range-slider"),
        dcc.RangeSlider(
            id="doc-comments-range-slider",
            min=0,
            max=500,
            step=10,
            value=[0, 100],
            marks={i: str(i) for i in range(0, 501, 100)},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div(id="comments-slider-output", style={"margin-top": "10px"}),
    ],
    className="mb-3",
    id="comments-slider-container",
)

time_delta_slider = html.Div(
    [
        dbc.Label("Time Delta", html_for="time-delta-slider"),
        dcc.RangeSlider(
            id="time-delta-slider",
            min=-50,
            max=50,
            step=10,
            value=[0, 20],
            marks={i: str(i) for i in range(-50, 51, 10)},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div(id="time-delta-output", style={"margin-top": "10px"}),
    ],
    className="mb-3",
    id="time-delta-slider-container",
)

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

num_documents = html.Div(
    [
        dbc.Label("Number of Documents", html_for="num-documents"),
        dcc.Input(
            id="num-documents",
            type="number",
            placeholder=10,
            min=1,
            step=1,
            value=1000,
            style={"width": "100%"},
        ),
    ],
    className="mb-3",
)

post_or_comment_checkbox = html.Div(
    [
        dbc.Label("Posts or Comments or both", html_for="post-or-comment"),
        dcc.Checklist(
            id="post-or-comment",
            options=[
                {"label": "Post", "value": "post"},
                {"label": "Comment", "value": "comment"},
            ],
            value=["post", "comment"],  # Default selected
            labelStyle={'display': 'inline-block', 'margin-right': '10px'},
        ),
    ],
    className="mb-3",
)

form = dbc.Card(
    dbc.CardBody(
        dbc.Form(
            [
                dbc.Row(
                    dbc.Col(date_range_slider, width=12),
                    className="mb-3",
                ),
                dbc.Row(
                    dbc.Col(comments_slider, width=12),
                    className="mb-3",
                ),
                dbc.Row(
                    dbc.Col(time_delta_slider, width=12),
                    className="mb-3",
                ),
                dbc.Row(
                    dbc.Col(post_or_comment_checkbox, width=12),
                    className="mb-3",
                ),
                dbc.Row(
                    dbc.Col(ngram_input, width=12),
                    className="mb-3",
                ),
                dbc.Row(
                    dbc.Col(group_input, width=12),
                    className="mb-3",
                ),
                dbc.Row(
                    dbc.Col(num_documents, width=12),
                    className="mb-3",
                ),
                dbc.Row(
                    dbc.Col(
                        dbc.Button(
                            "Submit Query", 
                            id="submit-query-button", 
                            color="primary",
                            className="me-2",
                        ), 
                        width=12
                    ),
                    className="mb-3",
                ),
            ]
        )
    ),
    style=card_style
)
