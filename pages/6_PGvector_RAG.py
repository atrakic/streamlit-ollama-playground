"""
## Streamlit app to demonstrate the usage of pgvector for vector similarity search


## Run the following commands to install the required packages
openai
pgvector
psycopg[binary]

Addapted from: https://www.timescale.com/blog/build-a-fully-local-rag-app-with-postgresql-mistral-and-ollama
"""

import streamlit as st
import os
import ollama
from pgvector.psycopg import register_vector
import psycopg
from dotenv import load_dotenv

load_dotenv()

TEXT_EMBEDDING_MODEL = os.environ.get("TEXT_EMBEDDING_MODEL", "all-minilm")  # 45 MB


def setup_db(size):
    """Setup the database and create the documents table"""
    conn = psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        user=os.environ.get("PGUSER", "postgres"),
        dbname=os.environ.get("POSTGRES_DB", "ollama"),
        autocommit=True,
    )

    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)

    conn.execute("DROP TABLE IF EXISTS documents")
    conn.execute(
        f"""CREATE TABLE documents (
            id bigserial PRIMARY KEY,
            content text,
            embedding vector({size}))
        """
    )
    return conn


def main():
    st.title("PGvector demo")
    st.markdown(
        "Local RAG knowledge base using [pgvector](https://github.com/pgvector/pgvector) for vector similarity search powered by PostgreSQL and Ollama."
    )
    st.image(
        "https://github.com/timescale/private-rag-example/blob/main/Architecture%20%20of%20Private%20RAG%20Application.png?raw=true"
    )

    sample_data = ["The dog is barking", "The cat is purring", "The bear is growling"]

    # Embed the input data and setup the database
    client = ollama.Client()
    response = client.embed(model=TEXT_EMBEDDING_MODEL, input=sample_data)
    embedding_size = len(response["embeddings"][0])
    conn = setup_db(embedding_size)

    ## Insert the data and embeddings into the database
    for content, embedding in zip(sample_data, response["embeddings"]):
        conn.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (content, embedding),
        )

    ## Chat UI

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["type"]):
            st.markdown(msg["content"])

    ## Retrieval and Generation
    user_input = st.chat_input("What questions do you have about our knowledge base?")

    if user_input:
        st.session_state.messages.append({"type": "user", "content": user_input})

        # Embed the user input
        user_embedding = client.embed(model=TEXT_EMBEDDING_MODEL, input=[user_input])[
            "embeddings"
        ][0]

        # Retrieve relevant documents based on cosine distance
        result = conn.execute(
            """
            SELECT content, 1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY similarity DESC
            LIMIT 5;
            """,
            (user_embedding,),
        ).fetchall()

        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            st.write("Top 5 relevant documents:")
            for row in result:
                st.write(f"Content: {row[0]}")
                st.markdown(f"**Similarity: {row[1]:.4f}**")


if __name__ == "__main__":
    main()
