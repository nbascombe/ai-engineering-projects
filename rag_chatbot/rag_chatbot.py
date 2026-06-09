import os
import chromadb
from google import genai
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def chunk_document(document):
    """Split a document into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(document)
    print(f"Total chunks: {len(chunks)}")
    return chunks

def embed_text(text): 
    """Embed a string using Gemini and return a list of floats."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

def embed_and_store(chunks, chromadb_client):
    """Embed each document chunk and store in ChromaDB with metadata"""
    collection = chromadb_client.create_collection("tennis_rules")

    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i + 1}/{len(chunks)}...")
        embedding = embed_text(chunk.page_content)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk.page_content],
            metadatas=[{"page": chunk.metadata.get("page")}]
        )
        time.sleep(0.7)
    print(f'Stored {len(chunks)} chunks')
    return collection

def load_or_build_collection(chromadb_client, chunks):
    """Load existing ChromaDB collection or build a new one if it doesn't exist."""
    try:
        collection = chromadb_client.get_collection("tennis_rules")
        print("Loaded existing collection\n")
        return collection
    except Exception:
        print("No existing collection found — building now...")
        return embed_and_store(chunks, chromadb_client)

def find_relevant_chunks(question, collection, n_results=3):
    """For an embedded question query the ChromaDB collection and return the n top results."""
    question_embedding = embed_text(question)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )
    return results['documents'][0]

def build_prompt(context, question):
    """Format retrieved context and question into a prompt for the LLM."""
    context_str = "\n\n".join(context)
    return f"""You are a helpful assistant. Answer the question using only the context below.
    Context: {context_str}
    
    Question: {question}"""

def main():
    chromadb_client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
    loader = PyPDFLoader(os.path.join(BASE_DIR, "documents", "2026-rules-of-tennis-english.pdf"))
    docs = loader.load()
    docs = docs[6:]
    chunks = chunk_document(docs)
    collection = load_or_build_collection(chromadb_client, chunks)

    chat = client.chats.create(
        model="gemini-2.5-flash",
        config={"system_instruction": "You are TennisRulesBot, a helpful tennis rules assistant. Answer questions using only the context provided. If the answer is not in the context, say so."}
    )

    print("TennisRulesBot - type 'quit' to exit")
    print("--------------------------------------------")

    while True:
        try:
            question = input("You: ")

            if not question.strip():
                print("Please type a question")
                continue

            if question.lower() == "quit":
                print("Goodbye!")
                break

            context = find_relevant_chunks(question, collection, 3)
            response = chat.send_message(build_prompt(context, question))
            print(f"\nAssistant: {response.candidates[0].content.parts[0].text}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Something went wrong: {e}")
            print("Please try again")

if __name__ == "__main__":
    main()
