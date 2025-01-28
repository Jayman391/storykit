from bertopic import BERTopic
from plotly.graph_objs import Figure

def fit_topic_model(docs):
  topic_model = BERTopic()
  topics, probs = topic_model.fit_transform(docs)
  return topic_model, topics, probs

def make_visualizations(topic_model : BERTopic, docs) -> Figure:
  return topic_model.visualize_documents(docs)