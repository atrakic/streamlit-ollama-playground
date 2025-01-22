"""
Adapted from the original code by Lilian Weng
https://python.langchain.com/docs/tutorials/rag/
"""

import streamlit as st
import os
import bs4

from langchain import hub
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class VectorStore:
    def __init__(self, model: str):
        self.embeddings = OllamaEmbeddings(model=model)
        self.vector_store = InMemoryVectorStore(self.embeddings)

    def add_documents(self, documents: List[Document]):
        self.vector_store.add_documents(documents)

    def similarity_search(self, query: str):
        return self.vector_store.similarity_search(query)


class TextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_documents(self, documents: List[Document]):
        return self.text_splitter.split_documents(documents)


class DocumentLoader:
    def __init__(self, web_paths: tuple, bs_kwargs: dict):
        self.loader = WebBaseLoader(web_paths=web_paths, bs_kwargs=bs_kwargs)

    def load(self):
        return self.loader.load()


class GraphBuilder:
    def __init__(self, state_class):
        self.graph_builder = StateGraph(state_class)

    def add_sequence(self, sequence: List):
        self.graph_builder.add_sequence(sequence)

    def add_edge(self, start: str, end: str):
        self.graph_builder.add_edge(start, end)

    def compile(self):
        return self.graph_builder.compile()


def retrieve(state: State, _vector_store: VectorStore):
    retrieved_docs = _vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(_state: State):
    docs_content = "\n\n".join(doc.page_content for doc in _state["context"])
    prompt = hub.pull("rlm/rag-prompt")
    messages = prompt.invoke({"question": _state["question"], "context": docs_content})
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0)
    response = llm.invoke(messages)
    return {"answer": response}


def main():
    st.title("Basic RAG App with Ollama API")
    st.markdown(
        "This demo shows how to create a basic RAG app with "
        "[langchain](https://github.com/langchain-ai/langchain/tree/master) and Ollama API. "
        "Inspired using this [tutorial](https://python.langchain.com/docs/tutorials/rag/)."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat with the data and provide history
    if prompt := st.chat_input("Enter your query:"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Generating response..."):
            vector_store = VectorStore(OLLAMA_MODEL)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )

            # Load and chunk contents of the blog
            loader = WebBaseLoader(
                web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                ),
            )
            docs = loader.load()
            all_splits = text_splitter.split_documents(docs)
            _ = vector_store.add_documents(documents=all_splits)

            graph_builder = GraphBuilder(State)
            graph_builder.add_sequence(
                [
                    ("retrieve", lambda state: retrieve(state, vector_store)),
                    ("generate", lambda state: generate(state)),
                ]
            )
            graph_builder.add_edge(START, "retrieve")
            graph_builder.add_edge("retrieve", "generate")
            graph = graph_builder.compile()
            response = graph.invoke({"question": prompt})

        with st.chat_message("assistant"):
            # st.write(f"Context: {response['context']}")
            st.markdown(response["answer"])

        st.session_state.messages.append(
            {"role": "assistant", "content": response["answer"]}
        )


if __name__ == "__main__":
    main()
