from bertopic import BERTopic
from plotly.graph_objs import Figure
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def fit_topic_model(docs):
  topic_model = BERTopic(verbose=True, low_memory=True, calculate_probabilities=False, embedding_model=model)#, nr_topics='auto'
  topics, probs = topic_model.fit_transform(docs)
  return topic_model, topics, probs

def visualize_documents(topic_model : BERTopic, docs) -> Figure:
  return topic_model.visualize_documents(docs)

def visualize_hierarchy(topic_model : BERTopic) -> Figure:
  return topic_model.visualize_hierarchy()

def visualize_heatmap(topic_model : BERTopic) -> Figure:
  return topic_model.visualize_heatmap()

