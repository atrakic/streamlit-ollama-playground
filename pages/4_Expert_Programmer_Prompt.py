import streamlit as st
import os
import time

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from prompts.ExpertProgrammer import ExpertProgrammer

load_dotenv()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


@st.cache_resource
def get_llm():
    return OllamaLLM(
        model=OLLAMA_MODEL,
        temperature=0,
    )


@st.cache_data
def generate_response(text):
    llm = get_llm()
    return llm.invoke(text)


def blog_outline(topic):
    template = ExpertProgrammer(topic=topic).INSTRUCTION
    prompt = PromptTemplate(input_variables=["topic"], template=template)
    prompt_query = prompt.format(topic=topic)
    response = generate_response(prompt_query)
    return st.markdown(response)


def main():
    st.title("🦜🔗 Langchain - Expert Programmer Blog Outline Generator App")
    st.write("""Generate an outline for a blog post based on a given topic.""")

    st.markdown(
        """
        ## Examples:
        - C# Dependency Injection
        - Python Decorators
        - JavaScript Promises
        - Java Multithreading
        - SQL Joins
        - Docker Containers
        - AWS Lambda Functions
        - React Hooks
        - GraphQL APIs
        - Flutter State Management
        - Kubernetes Deployment
        - Django Middleware
        - .NET Core Web API
        - Angular Directives
        - Data Structures in C++
        - Algorithms in Python
        - Machine Learning in R
        - Cybersecurity in Go
        - DevOps with Jenkins
        - Web Scraping with Python
        - REST APIs with Node.js
        - Mobile App Development
    """
    )

    with st.form("myform"):
        topic_text = st.text_input("Enter topic", "")
        submitted = st.form_submit_button("Submit")
        if submitted:
            start_time = time.time()
            blog_outline(topic_text)
            st.write(f"Time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
