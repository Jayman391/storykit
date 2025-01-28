from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Load from directory
loaded_model = BERTopic.load("BERTopic/TopicModel")

# Load from file
# loaded_model = BERTopic.load("my_model")

# Tests from other file
# print(loaded_model.get_topic_info())

# print('---------------------------------\n')

# print(loaded_model.get_topic(10))

# Topic visualization generates plotly graph there are many visualizations to choose from
# This is just the basic one
# loaded_model.visualize_topics().show()


print(loaded_model.representation_model)