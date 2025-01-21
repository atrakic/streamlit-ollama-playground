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
# from langchain_chroma import Chroma
# from langchain_core.prompts import PromptTemplate

from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0)


## 1. Indexing

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

embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
vector_store = InMemoryVectorStore(embeddings)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response}


# @st.cache_data
# def query_ollama(prompt):
#     data = {"prompt": prompt, "max_tokens": 100}
#     ollama = OllamaLLM(model=OLLAMA_MODEL, temperature=0)
#     response = ollama.invoke(data)
#     return response


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


def main():
    st.title("Basic RAG App with Ollama API")
    st.markdown(
        "This demo shows how to create a basic RAG app with [langchain](https://github.com/langchain-ai/langchain/tree/master) and Ollama API"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # chat with the data and provide history
    user_input = st.chat_input("Enter your query:")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Generating response..."):
            state = graph.invoke({"question": user_input})  ## query_ollama(user_input)
            # st.write(f"Context: {state['context']}")
            st.write(state["answer"])
            st.session_state.messages.append(
                {"role": "assistant", "content": state["answer"]}
            )

    # if st.button("Generate Response"):
    #     if user_input:
    #         with st.spinner("Generating response..."):
    #             state = graph.invoke({"question": user_input})
    #             st.write(state["answer"])
    #             #response = generate( )  # query_ollama(user_input)
    #             #st.success("Response generated!")
    #             #st.write(response.get("text", "No response text found."))
    #     else:
    #         st.error("Please enter a query.")


if __name__ == "__main__":
    main()
