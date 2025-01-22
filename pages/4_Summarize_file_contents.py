import streamlit as st
import os
import asyncio

from langchain_ollama import ChatOllama
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


async def main():
    st.title("🦜🔗 Langchain summarize file contents")
    st.markdown(
        "This demo shows how to summarize file contents with [langchain.chains.AnalyzeDocumentChain](https://github.com/langchain-ai/langchain/tree/master) and Ollama API"
    )

    uploaded_file = st.file_uploader("Choose a text file")
    if uploaded_file is not None and uploaded_file.type == "text/plain":
        file_contents = uploaded_file.getvalue().decode("utf-8")

        if st.button("Analyze document"):
            llm = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0,
            )

            qa_chain = load_qa_chain(llm, chain_type="map_reduce")
            qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

            response = qa_document_chain.run(
                question="Summarize file contents and ensure to capture the main points",
                input_document=file_contents,
            )

            st.write(response)


if __name__ == "__main__":
    asyncio.run(main())
