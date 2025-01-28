from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.load import dumps, loads

from typing_extensions import List, TypedDict

import dotenv

import langgraph.store

dotenv.load_dotenv()

# Define state for applications
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def initialize_rag(docs : List[str]):
    
  llm = ChatOpenAI(model="gpt-4o-mini")
  embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
  vector_store = InMemoryVectorStore(embeddings)

  documents = [Document(page_content=doc) for doc in docs]

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  all_splits = text_splitter.split_documents(documents)

  # Index chunks
  _ = vector_store.add_documents(documents=all_splits)

  # Define prompt for question-answering
  prompt = hub.pull("rlm/rag-prompt")

  return llm, vector_store, prompt


# Define application steps
def retrieve(state: State, vector_store: InMemoryVectorStore):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State, llm: ChatOpenAI, prompt):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def compile_rag():
    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    

    return dumps(graph)

def compute_rag(graph, question):
    graph = loads(graph)
    response = graph.invoke({"question": question})
    return response["answer"]
