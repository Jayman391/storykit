import dash
from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd

# Use the same card style
card_style = {
    'padding': '15px',
    'marginBottom': '20px',
    'boxShadow': '0 0.125rem 0.25rem rgba(0,0,0,0.075)',
    'border': '1px solid #dee2e12',
    'borderRadius': '0.25rem',
    'backgroundColor': '#ffffff',
    'margin': 'auto'
}

# --- N-gram Analysis Form ---
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

# --- Sentiment Analysis Form ---
windowslider = dbc.Col(
    [
        dbc.Label('Sentiment Smoothing Window (number of days)'),
        dcc.Slider(
            id='window-slider',
            min=1,
            max=20,
            value=1,
            tooltip={'placement': 'bottom', 'always_visible': True}
        )
    ],
    className="mb-3"
)

sentimentform = dbc.Card(
    [
        dbc.CardHeader("Sentiment Analysis"),
        dbc.CardBody(
            dbc.Form(
                [
                    dbc.Row([windowslider]),
                    dbc.Row(
                        dbc.Col(
                            dbc.Button("Submit", id="submit-sentiment-config", color="primary"),
                            className="mt-3"
                        )
                    ),
                ]
            )
        ),
    ],
    style=card_style
)

# --- Topic Modeling Form ---
embeddingmodel = dbc.Col(
    [
        dbc.Label('Select Sentence Transformers Embedding Model'),
        dcc.Link('Sentence Transformers Models', href='https://huggingface.co/models?sort=trending&search=sentence-transformers', target="_blank"),
        dcc.Textarea(
            id='embedding-model',
            placeholder='Enter Sentence Transformers Model Name',
            style={'width': '100%'}
        )
    ],
    className="mb-3"
)

dimredradio = dbc.Col(
    [
        dbc.Label('Select Dimensionality Reduction Technique'),
        dbc.RadioItems(
            id='dimred-radio',
            options=[
                {'label': 'UMAP', 'value': 'UMAP'},
                {'label': 'PCA', 'value': 'PCA'},
            ],
            value='UMAP',
            inline=True
        )
    ],
    className="mb-3"
)

dimreddims = dbc.Col(
    [
        dbc.Label('Select Number of Dimensions'),
        dcc.Input(
            id='dimred-dims',
            type='number',
            placeholder=2,
            min=2,
            step=1,
            value=2,
            style={'width': '100%'}
        )
    ],
    className="mb-3"
)

clusterradio = dbc.Col(
    [
        dbc.Label('Select Clustering Algorithm'),
        dbc.RadioItems(
            id='cluster-radio',
            options=[
                {'label': 'HDBSCAN', 'value': 'HDBSCAN'},
                {'label': 'KMeans', 'value': 'KMeans'},
                {'label': 'Spectral Clustering', 'value': 'Spectral'}
            ],
            value='KMeans',
            inline=True
        )
    ],
    className="mb-3"
)

numclusters = dbc.Col(
    [
        dbc.Label('Select Number of Clusters (Ignore for HDBSCAN)'),
        dcc.Input(
            id='num-clusters',
            type='number',
            placeholder=2,
            min=2,
            step=1,
            value=2,
            style={'width': '100%'}
        )
    ],
    className="mb-3"
)

hdbscanhyperparams = dbc.Col([
    dbc.Label('HDBSCAN Hyperparameters (Ignore for KMeans and Spectral Clustering)'),
    dbc.Row([
        dbc.Col([
            dbc.Label('Minimum Cluster Size'),
            dcc.Input(
                id='min-cluster-size',
                type='number',
                placeholder=2,
                min=2,
                step=1,
                value=2,
                style={'width': '100%'}
            )
        ]),
        dbc.Col([
            dbc.Label('Minimum Samples'),
            dcc.Input(
                id='min-samples',
                type='number',
                placeholder=2,
                min=2,
                step=1,
                value=2,
                style={'width': '100%'}
            )
        ]),
        dbc.Col([
            dbc.Label('Metric'),
            dcc.Dropdown(
                id='metric',
                options=[
                    {'label': 'euclidean', 'value': 'euclidean'},
                    {'label': 'cosine', 'value': 'cosine'},
                    {'label': 'manhattan', 'value': 'manhattan'},
                    {'label': 'l1', 'value': 'l1'},
                    {'label': 'l2', 'value': 'l2'},
                ],
                value='euclidean',
                style={'width': '100%'}
            )
        ])
        ],className="mb-3")
])

topicform = dbc.Card(
    [
        dbc.CardHeader("Topic Modeling"),
        dbc.CardBody(
            dbc.Form(
                [
                    embeddingmodel,
                    dimredradio,
                    dimreddims,
                    clusterradio,
                    numclusters,
                    hdbscanhyperparams,
                    dbc.Row(
                        dbc.Col(
                            dbc.Button("Submit", id="submit-topic-config", color="primary"),
                            className="mt-3"
                        )
                    )
                ]
            )
        ),
    ],
    style=card_style
)

# --- Overall Configuration Card ---
configforms = dbc.Card(
    [
        dbc.CardHeader("Analysis Configuration"),
        dbc.CardBody([
                dbc.Row(dbc.Col(ngramform, width=12), className="mb-3"),
                dbc.Row(dbc.Col(sentimentform, width=12), className="mb-3"),
                dbc.Row(dbc.Col(topicform, width=12), className="mb-3"),
            ])
    ],
    style=card_style
)
