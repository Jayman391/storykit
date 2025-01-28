from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.datasets import fetch_20newsgroups
import safetensors

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

# Choose Topic Representations 
# Im using KeyBERTInspired for this example but there are many options like GPT
representation_model = KeyBERTInspired()

topic_model = BERTopic(representation_model=representation_model).fit(docs)


# Save Model

saveMethod = input("which save method: safetensors, pytorch, pickle")

# Method 1 - safetensors
if saveMethod == "safetensors":
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    topic_model.save("BERTopic/safetensorsModel", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

# Method 2 - pytorch
if saveMethod == "pytorch":
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    topic_model.save("path/to/my/pytorchModel", serialization="pytorch", save_ctfidf=True, save_embedding_model=embedding_model)

# Method 3 - pickle
if saveMethod == "pickle":
    topic_model.save("pickleModel", serialization="pickle")