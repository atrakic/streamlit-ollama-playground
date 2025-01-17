import streamlit as st

import os
import ollama
import chromadb

from dotenv import load_dotenv

load_dotenv()

###

TEXT_EMBEDDING_MODEL = os.environ.get("TEXT_EMBEDDING_MODEL", "all-minilm")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def store_document_in_db(documents, collection):
    # store each document in a vector embedding database
    for i, d in enumerate(documents):
        response = ollama.embeddings(model=TEXT_EMBEDDING_MODEL, prompt=d)
        embedding = response["embedding"]
        collection.add(ids=[str(i)], embeddings=[embedding], documents=[d])


def generate_embeddings_and_store_in_db(prompt, documents, collection):
    # generate an embedding for the prompt and retrieve the most relevant doc
    response = ollama.embeddings(prompt=prompt, model=TEXT_EMBEDDING_MODEL)
    results = collection.query(query_embeddings=[response["embedding"]], n_results=1)

    if results["documents"] and results["documents"][0]:
        data = results["documents"][0][0]
        return data
    else:
        return None


def display_results(data, prompt):
    output = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=f"Using this data: {data}. Respond to this prompt: {prompt}",
    )
    return output["response"]


###


def main():
    st.title("ChromaDB embedding with Ollama API")

    client = chromadb.Client()
    collection = client.get_or_create_collection(name="docs")

    ## Step 1: Generate embeddings
    documents = [
        "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
        "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
        "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
        "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
        "Llamas are vegetarians and have very efficient digestive systems",
        "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
    ]
    store_document_in_db(documents, collection)

    ## Step 2: Query embeddings
    prompt = st.text_input(
        "Enter your query...",
        value="What animals are llamas related to?",
        help="Enter a prompt to generate embeddings and retrieve the most relevant document",
    )

    if st.button("Submit"):
        data = generate_embeddings_and_store_in_db(prompt, documents, collection)

        ## Step 3: Display results
        response = display_results(data, prompt)
        st.write(response)


if __name__ == "__main__":
    main()
