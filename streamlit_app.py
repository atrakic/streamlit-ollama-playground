import streamlit as st
import asyncio


async def main():
    st.title("Ollama Streamlit Apps")
    st.sidebar.success("Select a demo above.")
    st.markdown(
        """
    - [View the source code](https://github.com/atrakic/streamlit-ollama-playground.git)
    """
    )


if __name__ == "__main__":
    asyncio.run(main())
