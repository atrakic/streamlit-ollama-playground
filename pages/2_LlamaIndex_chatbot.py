"""
Addapted from: https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
"""

import os
import streamlit as st
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


@st.cache_resource
def get_llm():
    return Ollama(
        model=OLLAMA_MODEL,
        request_timeout=300.0,
    )


@st.cache_data
def generate_response(prompt):
    llm = get_llm()
    return llm.complete(prompt)


def main():
    st.title("LLamaIndex basic LLM chat app demo")
    st.markdown(
        "This demo shows how to use [llama-index](https://github.com/run-llama/llama_index) with Ollama API to create a basic LLM chat app."
    )
    st.markdown(
        "Enter a message to chat with the assistant. Example: *What is the capital of France?*, * 3 ^ 3 *"
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write(generate_response(prompt).text)
            # response = st.write_stream(generate_response(prompt))

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
