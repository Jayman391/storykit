# backend/chatbot.py

from langchain import hub
from langchain_openai import ChatOpenAI
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Define state for applications
class StateType(TypedDict):
    question: str
    context: List[Document]
    answer: str

class RAGResponse(TypedDict):
    answer: str
    context: List[str]

class RAGPipeline:
    def __init__(self, docs: List[str]):
        """
        Initialize the RAG pipeline with the provided documents.
        """
        self.llm, self.vector_store, self.prompt = self.initialize_rag(docs)
        self.graph = self.compile_rag()

    def initialize_rag(self, docs: List[str]):
        """
        Initialize RAG components: LLM, embeddings, vector store, and prompt.
        """
        llm = ChatOpenAI(model="gpt-4o-mini", api_key="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")
        embeddings = HuggingFaceEmbeddings()
        self.vector_store = InMemoryVectorStore(embeddings)

        documents = [Document(page_content=doc) for doc in docs]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(documents)

        # Index chunks
        self.vector_store.add_documents(documents=all_splits)

        # Define prompt for question-answering
        prompt = hub.pull("rlm/rag-prompt")

        return llm, self.vector_store, prompt

    def retrieve(self, state: StateType) -> StateType:
        """
        Retrieve relevant documents based on the question.
        """
        retrieved_docs = self.vector_store.similarity_search(state["question"], k=20)
        state["context"] = retrieved_docs
        return state

    def generate(self, state: StateType) -> StateType:
        """
        Generate an answer using the retrieved context.
        """
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        state["answer"] = response.content
        return state

    def compile_rag(self):
        """
        Compile the RAG graph.
        """
        graph_builder = StateGraph(StateType).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        return graph

    def compute_rag_response(self, question: str) -> RAGResponse:
        """
        Compute the RAG response for a given question, including context.
        """
        # retrieve and generate answer
        state = self.graph.invoke({"question": question})
        return {"answer": state["answer"], "context": [doc.page_content for doc in state["context"]]}

# Initialize a global RAGPipeline instance
global_rag_pipeline = None

def initialize_global_rag(docs: List[str]):
    """
    Initialize or reinitialize the global RAG pipeline with new documents.
    """
    global global_rag_pipeline
    global_rag_pipeline = RAGPipeline(docs)

def compute_rag(question: str) -> RAGResponse:
    """
    Compute the RAG response using the global RAG pipeline, returning both answer and context.
    """
    if not global_rag_pipeline:
        raise Exception("RAG pipeline is not initialized.")
    return global_rag_pipeline.compute_rag_response(question)
