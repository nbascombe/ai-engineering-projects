import os
import chromadb
from google import genai
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    chromadb_client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
    loader = PyPDFLoader(os.path.join(BASE_DIR, "documents", "2026-rules-of-tennis-english.pdf"))
    docs = loader.load()
    docs = docs[6:]
    chunks = chunk_document(docs)
    app.state.collection = load_or_build_collection(chromadb_client, chunks)
    yield
    # Clean up the collection and release the resources
    app.state.collection = None

app = FastAPI(lifespan=lifespan)

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
    return f"""
    Context: {context_str}
    
    Question: {question}"""

@app.get("/health")
async def health():
    return {"status": "ok"}

class Question(BaseModel):
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def strip_and_check(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("content cannot be empty or whitespace only")
        return v

@app.post("/output")
def content_output(question: Question):
    context = find_relevant_chunks(question.content, app.state.collection, 3)
    response = client.models.generate_content(
    model="gemini-2.5-flash",
    config={"system_instruction": "You are TennisRulesBot, a helpful tennis rules assistant. Answer questions using only the context provided. If the answer is not in the context, say so."},
    contents=build_prompt(context, question.content)
    )
    return {"response": response.text}



