from bertopic import BERTopic
from plotly.graph_objs import Figure
from sentence_transformers import SentenceTransformer

from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from sklearn.decomposition import PCA
from umap import UMAP

def fit_topic_model(docs, modelname, dimredparams, clusterparams):
  reducer = initialize_reducer(dimredparams)
  clusterer = initialize_clusterer(clusterparams)
  topic_model = BERTopic(verbose=True, low_memory=True, calculate_probabilities=False, embedding_model=SentenceTransformer(modelname), nr_topics='auto', umap_model=reducer, hdbscan_model=clusterer)

  topics, probs = topic_model.fit_transform(docs)
  return topic_model, topics, probs

def initialize_clusterer(params : dict):
  if params['cluster_radio'] == 'HDBSCAN':
    clusterer = HDBSCAN(min_cluster_size=params['min_cluster_size'], min_samples=params['min_samples'], metric=params['cluster_metric'])
  elif params['cluster_radio'] == 'KMeans':
    clusterer = KMeans(n_clusters=params['n_clusters'])
  else:
    clusterer = SpectralClustering(n_clusters=params['n_clusters'])
  return clusterer

def initialize_reducer(params : dict):
  if params['dimred_radio'] == 'UMAP':
    reducer = UMAP(n_components=params['dimred_dims'], metric=params['dimred_metric'])
  else:
    reducer = PCA(n_components=params['dimred_dims'])
  return reducer

def visualize_documents(topic_model : BERTopic, docs) -> Figure:
  return topic_model.visualize_documents(docs, sample=0.1)

def visualize_hierarchy(topic_model : BERTopic) -> Figure:
  return topic_model.visualize_hierarchy()

def visualize_heatmap(topic_model : BERTopic) -> Figure:
  return topic_model.visualize_heatmap()

def visualize_topics_over_time(topic_model : BERTopic, docs, dates) -> Figure:
  topics_over_time = topic_model.topics_over_time(docs, dates)
  return topic_model.visualize_topics_over_time(topics_over_time)

