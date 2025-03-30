import networkx as nx
import spacy
from itertools import combinations
# knowledge graph logic

# pass in list of documents
# for each document, extract entities
# for each entity, create a node
# for each entity-entity pair, create an edge
# add nodes and edges to graph

def create_knowledge_graph(documents : list) -> nx.Graph:
    nlp = spacy.load("en_core_web_sm")


    G = nx.Graph()
    for document in documents:
        doc = nlp(document)
        ents = [e.text for e in doc.ents if e.label_ == "GPE" or e.label_ == "ORG" or e.label_ == "EVENT" or e.label_ == "LOC"]
        for ent in ents: 
            G.add_node(ent, label=ent)
        for pair in combinations(ents, 2):
            G.add_edge(pair[0], pair[1])
    return G