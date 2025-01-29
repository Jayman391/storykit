from bertopic import BERTopic
from plotly.graph_objs import Figure

def fit_topic_model(docs):
  topic_model = BERTopic(verbose=True, low_memory=True, calculate_probabilities=False, nr_topics='auto')
  topics, probs = topic_model.fit_transform(docs)
  return topic_model, topics, probs

def visualize_documents(topic_model : BERTopic, docs) -> Figure:
  return topic_model.visualize_documents(docs)

def visualize_hierarchy(topic_model : BERTopic) -> Figure:
  return topic_model.visualize_hierarchy()