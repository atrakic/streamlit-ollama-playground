#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "streamlit",
# ]
# ///

import streamlit as st
import streamlit.web.bootstrap


def main():
    st.title("Ollama Streamlit Apps")
    st.sidebar.success("Select a demo above.")
    st.markdown(
        """
    - [View the source code](https://github.com/atrakic/streamlit-ollama-playground.git)
    """
    )


if __name__ == "__main__":
    # See: https://bartbroere.eu/2023/06/17/adding-a-main-to-streamlit/
    if "__streamlitmagic__" not in locals():
        streamlit.web.bootstrap.run(__file__, False, [], {})
    main()
