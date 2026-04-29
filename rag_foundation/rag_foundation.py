from google import genai
from dotenv import load_dotenv
import os
from rag_foundation.documents import DOCUMENTS
import chromadb

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def embed_text(text): 
    """Embed a string using Gemini and return a list of floats."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values


def build_collection(chromadb_client):
    """Add each document and its embedded text to a ChromaDB collection."""
    collection = chromadb_client.create_collection("rag_foundation_tennis")

    for document in DOCUMENTS:
        embedding = embed_text(document["text"])
        collection.add(
            ids=[document["id"]],
            embeddings=[embedding],
            documents=[document["text"]]
        )
        print(f"Stored: {document['id']}")

    print(f"Stored {len(DOCUMENTS)} documents")
    return collection

def find_relevant_documents(question, collection, n_results=2):
    """For an embedded question query the ChromaDB collection and return the n top results."""
    question_embedding = embed_text(question)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )
    return results['documents'][0]


def main():
    chromadb_client = chromadb.Client()
    collection = build_collection(chromadb_client)

    questions = [
        'Who is the best tennis player on clay?',
        'What does a tiebreak mean?'
    ]

    for question in questions:
        print(f'Question: {question}')
        docs = find_relevant_documents(question, collection)
        for i, doc in enumerate(docs, 1):
            print(f"  Result {i}: {doc[:80]}...")
        print()

if __name__ == "__main__":
    main()