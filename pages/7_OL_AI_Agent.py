"""
This app uses the OpenAI API to determine the city and country where the Olympics were held in a given year.
Modified from the original code by adding a Streamlit interface: https://ai.pydantic.dev/models/#ollama
"""

import streamlit as st
import os
import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your_openai_api_key")


class CityLocation(BaseModel):
    city: str
    country: str


def main():
    st.title("Olympic Games City Location AI agent")
    st.write(
        """This app uses the OpenAI API to determine the city and country where the Olympics were held in a given year."""
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input(placeholder="Type a year ...", max_chars=4):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Ensure there is an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        with st.spinner("Generating response..."):
            model = OpenAIModel(
                model_name=OLLAMA_MODEL,
                base_url=OLLAMA_URL,
                api_key=OPENAI_API_KEY,
            )

            agent = Agent(
                model=model,
                retries=3,
                result_type=CityLocation,
            )

            result = agent.run_sync(prompt)
            response = result.data

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
