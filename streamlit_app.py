import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

with st.sidebar:
    "[View the source code](https://github.com/atrakic/streamlit-ollama-rag-apps)"

st.title("Ollama RAG Apps")

st.sidebar.success("Select a demo above.")


st.markdown(
    """
    -[View the source code](https://github.com/atrakic/streamlit-ollama-rag-apps)

    **👈 Select a demo from the sidebar** to see some examples
    ### Referances
    - Ollama [embedding moddels](https://ollama.com/blog/embedding-models)
"""
)
