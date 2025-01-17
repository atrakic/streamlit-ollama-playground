"""
pip install llama-index-core

"""
import streamlit as st
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.ollama import Ollama

st.set_page_config(page_title="Mapping Demo", page_icon="🌍")

st.markdown("# Mapping Demo")
st.sidebar.header("Mapping Demo")
st.write(
    """This demo shows how to use
[`llama-index`](https://pypi.org/project/llama-index/) with Ollama API to generate a response to a prompt."""
)

st.title("🦜🔗 LlamaIndex App")

# Load the index
# index = VectorStoreIndex(
#    reader=SimpleDirectoryReader("data"),
#    embedding=HuggingFaceEmbedding("sentence-transformers/all-minilm"),
# )

# Load the LLM
# llm = Ollama()


# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")
# print(response)
