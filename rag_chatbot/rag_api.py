import os
import chromadb
from google import genai
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import redis.asyncio as redis
import asyncio
import hashlib
from langsmith import wrappers, traceable
from rag_chatbot.rag_common import MODEL, SYSTEM_INSTRUCTION, log_llm_call

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

client = wrappers.wrap_gemini(genai.Client(api_key=os.getenv("GOOGLE_API_KEY")))

r = redis.Redis(decode_responses=True)

# Cache keys are namespaced and versioned: rag:<CACHE_VERSION>:<fingerprint>:<question>.
# The fingerprint changes automatically when MODEL or SYSTEM_INSTRUCTION change, so stale
# answers are never served after those edits. CACHE_VERSION covers what the fingerprint
# can't see (retrieval k, chunk size, prompt template, source PDF): bump it by hand.
CACHE_VERSION = "v1"
_CACHE_FINGERPRINT = hashlib.sha256(f"{MODEL}|{SYSTEM_INSTRUCTION}".encode()).hexdigest()[:8]


def make_cache_key(question_text):
    """Normalised, namespaced Redis key for a question."""
    return f"rag:{CACHE_VERSION}:{_CACHE_FINGERPRINT}:{question_text.lower()}"

# asyncio only keeps weak references to tasks, so a fire-and-forget cache write
# can be garbage-collected before it finishes. Hold a strong reference until done.
background_tasks = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    chromadb_client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
    loader = PyPDFLoader(os.path.join(BASE_DIR, "documents", "2026-rules-of-tennis-english.pdf"))
    docs = loader.load()
    docs = docs[6:]
    chunks = chunk_document(docs)
    app.state.collection = load_or_build_collection(chromadb_client, chunks)
    yield
    # Let any in-flight cache writes finish, then release resources
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)
    app.state.collection = None

app = FastAPI(lifespan=lifespan)

# Now reachable via http://localhost:8000/static/websocket_client.html
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

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

@traceable(run_type="embedding", name="embed_text")
def embed_text(text): 
    """Embed a string using Gemini and return a list of floats."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

def embed_and_store(chunks, chromadb_client):
    """Embed each document chunk and store in ChromaDB with metadata.

    If embedding fails part way (quota, network, Ctrl-C), delete the half-built
    collection so the next run rebuilds it instead of silently serving a
    partial index via get_collection().
    """
    collection = chromadb_client.create_collection("tennis_rules")

    try:
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
    except BaseException:
        print("Embedding failed - removing partial collection so the next run rebuilds it")
        chromadb_client.delete_collection("tennis_rules")
        raise

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

@traceable(run_type="retriever", name="retrieve_chunks")
def find_relevant_chunks(question, collection, n_results=3):
    """For an embedded question query the ChromaDB collection and return the n top results.

    Sync version, kept for the CLI and eval_rag.py. Async routes should use
    find_relevant_chunks_async instead.
    """
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

async def embed_text_async(text): 
    """Embed a string using Gemini and return a list of floats."""
    result = await client.aio.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

# Traced like the sync version so the trace tree keeps its retrieve_chunks run.
@traceable(run_type="retriever", name="retrieve_chunks")
async def find_relevant_chunks_async(question, collection, n_results=3):
    """Async retrieval: async embed, then the Chroma query off the event loop."""
    question_embedding = await embed_text_async(question)
    results = await asyncio.to_thread(
        collection.query,
        query_embeddings=[question_embedding],
        n_results=n_results,
    )
    return results['documents'][0]

@traceable(name="rag_request")
async def generate_tokens(question):
    cache_key = make_cache_key(question.content)
    cache_response = await r.get(cache_key)
    if cache_response:
        yield cache_response
    else:
        # Async retrieval and async streaming, so nothing here blocks the event loop.
        context = await find_relevant_chunks_async(question.content, app.state.collection, 3)
        prompt = build_prompt(context, question.content)
        start = time.perf_counter()
        first_token_at = None
        last_chunk = None
        response = await client.aio.models.generate_content_stream(
            model=MODEL,
            config={"system_instruction": SYSTEM_INSTRUCTION},
            contents=prompt
        )
        chunks = []
        async for chunk in response:
            last_chunk = chunk
            if chunk.text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                yield chunk.text
                chunks.append(chunk.text)
        chunks_string = "".join(chunks)

        # Structured log of every real LLM call: prompt, response, latency, tokens.
        # usage_metadata arrives on the final streamed chunk.
        log_llm_call(
            source="api_output",
            model=MODEL,
            question=question.content,
            prompt=prompt,
            response=chunks_string,
            latency_s=time.perf_counter() - start,
            ttft_s=(first_token_at - start) if first_token_at else None,
            usage=getattr(last_chunk, "usage_metadata", None),
        )

        # Don't cache an empty response (blocked or failed generation).
        if chunks_string:
            task = asyncio.create_task(r.set(cache_key, chunks_string, ex=6000))
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

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

# HTTP chunked streaming
@app.post("/output", response_class=StreamingResponse)
async def content_output(question: Question):
    return StreamingResponse(
        generate_tokens(question),
        media_type="text/plain"
    )

# Websocket streaming
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    chat = client.aio.chats.create(
        model=MODEL,
        config={"system_instruction": SYSTEM_INSTRUCTION}
    )

    try:
        while True:
            question = await websocket.receive_text()
            context = await find_relevant_chunks_async(question, app.state.collection, 3)
            prompt = build_prompt(context, question)
            start = time.perf_counter()
            first_token_at = None
            last_chunk = None
            parts = []
            response = await chat.send_message_stream(prompt)
            async for chunk in response:
                last_chunk = chunk
                if chunk.text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    parts.append(chunk.text)
                    await websocket.send_text(chunk.text)
            log_llm_call(
                source="api_ws",
                model=MODEL,
                question=question,
                prompt=prompt,
                response="".join(parts),
                latency_s=time.perf_counter() - start,
                ttft_s=(first_token_at - start) if first_token_at else None,
                usage=getattr(last_chunk, "usage_metadata", None),
            )
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")