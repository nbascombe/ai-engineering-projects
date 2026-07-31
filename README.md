# AI Engineering Projects

A collection of projects built while developing hands-on AI engineering skills.
Each project is intentional - focused on understanding the fundamentals before 
adding complexity.

---

## Projects

### 1. Basic Chatbot (`basic_chatbot.py`)
**Problem:** What does a working LLM API call look like.
**Approach:** SInteractive terminal loop that takes user input, sends it to Gemini 2.5 Flash, and prints the response. Includes basic error handling with a one-time retry on API failure.
**Outcome:** A working baseline confirming API integration, key management, and response structure - now also resilient to a single transient API failure without crashing the session.

**Concepts covered:**
- Gemini API integration
- Secure API key management with environment variables
- Stateless LLM calls
- Terminal input loop
- Basic retry on failure

---

### 2. Tennis Analyst Bot (`tennis_analyst_bot.py`)
**Problem:** LLM responses are free text by default, which breaks any downstream code that needs to parse or act on them.
**Approach:** A conversational AI tennis analyst that maintains context across multiple 
messages, responding within a defined persona. Added a system prompt enforcing a strict JSON schema, typed field validation, and a retry-then-fallback chain for when the model ignores the instruction.
**Outcome:** Every response reliably returns structured JSON with `answer`, `players_mentioned`, `related_topics`, and `is_tennis_related` fields - parseable without special-casing, and graceful when validation fails.

**Concepts covered:**
- Stateful conversation management with chat sessions
- System prompts to control model behaviour and personality
- Structured outputs - forcing the model to return a consistent JSON schema
- JSON validation and schema enforcement with typed field checking
- Retry on parse failure, fall back to raw text
- The difference between stateless calls and conversational memory

**Response schema:**
```json
{
    "answer": "string",
    "players_mentioned": ["array of strings"],
    "related_topics": ["array of strings"],
    "is_tennis_related": true
}
```

**Why structured outputs matter:**
At scale, downstream code reads LLM responses - not humans. Free text breaks 
parsers unpredictably. A consistent schema means reliable parsing, structured 
logging, and code that can make decisions based on response fields.

---

### 3. RAG Foundation (`rag_foundation/`)
**Problem:** LLMs can only answer from what they were trained on - they have no access to private or updated documents.
**Approach:** Embedded 10 tennis documents using Gemini Embedding 001, stored them in ChromaDB, and built a retrieval function that finds the most semantically relevant documents for any question.
**Outcome:** Given "Who is the best tennis player on clay?", the system returns the Nadal document ahead of all others - retrieval by meaning, not keyword matching. This is the component that makes RAG work.

**Concepts covered:**
- Text embeddings - converting meaning into vectors of numbers
- Vector similarity search - finding documents by meaning, not keywords
- ChromaDB - storing and querying embeddings locally
- Separating the embed step from the storage step for model flexibility

**Why this matters:**
This is the retrieval step that makes RAG work. Instead of relying on what the
LLM was trained on, you retrieve relevant context first and pass it in. The next step
is to pass the retrieved documents to an LLM as context to generate an answer.

---

### 4. TennisRulesBot CLI (`rag_chatbot.py`)
**Problem:** LLMs hallucinate confidently when asked about specific rule details they weren't trained on precisely.
**Approach:** Full RAG (Retrieval Augmented Generation) pipeline over the official 2026 ITF Rules of Tennis PDF. Document loaded, chunked at 1200 chars, embedded, stored persistently in ChromaDB, with the LLM instructed to answer only from retrieved context.
**Outcome:** Answers rules questions grounded in the document; correctly refuses out-of-scope questions ("Where can I play baseball?") rather than hallucinating. Loads in under a second on subsequent runs via persistent vector store - no re-embedding needed.
 
**Concepts covered:**
- RAG pipeline end to end - load, chunk, embed, store, retrieve, generate
- LangChain document loaders and text splitters
- Persistent vector storage with ChromaDB
- Retrieval grounding - LLM answers only from retrieved context
- Rate limit handling with controlled embedding throughput
- The limits of RAG - retrieval quality depends on source document terminology

**Why this matters:**
Combines every concept from the previous projects into one working system.
The LLM is grounded in a real document and will say so when it cannot answer,
rather than hallucinating.

---

### 5. FastAPI Chatbot (`fastapi_chatbot.py`)
**Problem:** A Python script calling an LLM directly can only be used by one person 
in one place. And a naive implementation, even wrapped in FastAPI, blocks a thread 
per request, which doesn't scale.
**Approach:** Wrapped the chatbot in a FastAPI POST endpoint with Pydantic request 
validation. First as a sync function (FastAPI offloads to a thread pool), then updated 
to async def with await client.aio.models.generate_content so the event loop isn't 
blocked while Gemini processes.
**Outcome:** A running HTTP service testable via Swagger at /docs, where concurrent 
requests are handled on a single thread without blocking - no thread-per-request overhead.

**Concepts covered:**
- FastAPI endpoint structure and route decoration
- Pydantic models for request validation
- Health check endpoints
- Testing APIs via Swagger UI
- sync vs async endpoints - thread pool vs event loop
- client.aio for non-blocking Gemini calls

---

### 6. TennisRulesBot API (`rag_chatbot/rag_api.py`)
**Problem:** The RAG chatbot from Project 4 only runs as a CLI script usable by one 
person, in one terminal, at a time. It can't be called by a frontend, another service, 
or anything outside that terminal session.
**Approach:** Wrapped the same RAG pipeline as a FastAPI service. The ChromaDB collection 
is built once at startup via a `lifespan` context manager and shared across requests 
through `app.state`, rather than rebuilt per call. Pydantic validation strips and 
rejects empty/whitespace-only input. Unlike the CLI's persistent chat session, each 
request is answered statelessly via a single generate call - no conversation history 
is shared across callers.
**Outcome:** A running HTTP service returning grounded answers to tennis rules questions, 
testable via Swagger UI. Verified against the same questions the CLI answers correctly, 
and confirmed (via a deliberately ambiguous follow-up - "is that the same in doubles?") 
that no history carries over between requests, as expected for a stateless design.

**Concepts covered:**
- Wrapping an existing RAG pipeline as a FastAPI service
- Sharing expensive-to-build state (`app.state`) via `lifespan` startup/shutdown
- Sync route handlers and why blocking SDK calls need FastAPI's thread pool
- Pydantic request validation with a custom field validator
- REST statelessness as a deliberate design tradeoff, not a limitation of AI APIs generally

**Why this matters:**
Same retrieval quality as Project 4, now reachable by any client over HTTP.

---

### 7. TennisRulesBot WebSocket (`rag_chatbot/rag_api.py` - `/ws`)
**Problem:** The stateless `/output` endpoint has no conversational memory, and the client waits for the full response rather than seeing tokens progressively.
**Approach:** Added a `/ws` WebSocket endpoint holding one persistent Gemini chat session per connection, streaming tokens as they generate. Switched to the async Gemini client throughout, since WebSocket handlers in FastAPI must be `async def` and a blocking call inside one stalls every other connected client - not just the caller.
**Outcome:** A stateful, streaming chat endpoint verified against a minimal HTML/JS client. Confirmed empirically (two concurrent connections, timed) that the async fix prevents one client's request from blocking another's. Also surfaced a real finding along the way: identical questions with identical retrieved context can produce contradictory LLM conclusions.

**Concepts covered:**
- WebSocket lifecycle and `async def` requirements in FastAPI
- Async vs sync SDK clients and event-loop blocking
- Connection-scoped persistent state vs stateless REST design
- Defensive handling of streamed chunks that may carry no text
- Distinguishing concurrency bugs from LLM non-determinism through controlled testing

---

## Technical Progression
- `basic_chatbot.py` - stateless, single call, no memory
- `tennis_analyst_bot.py` - stateful, conversational, structured JSON outputs, validated responses
- `rag_foundation/` - embeddings, vector storage, semantic search - retrieval layer of a RAG system
- `rag_chatbot/` - full RAG pipeline with persistent vector store and grounded LLM responses
- `rag_chatbot/rag_api.py`- same RAG pipeline as an HTTP service, stateless per request, Pydantic-validated. Uses `client.models.generate_content` (sync) and `client.models.embed_content` (sync, inside find_relevant_chunks) → correctly paired with plain def, letting FastAPI's thread pool handle it.
- `rag_chatbot/rag_api.py` `/ws` - same pipeline again, now stateful per connection and fully async (`client.aio`), proving why sync calls inside `async def` WebSocket handlers block every other connected client.
- `fastapi_chatbot.py` - LLM wrapped as an HTTP service, Pydantic validation, health endpoint. Uses `client.aio.models.generate_content` → genuinely async → correctly paired with async def.

---

## Tech Stack
- Python 3.13
- Google Gemini 2.5 Flash + Gemini Embedding 001
- google-genai
- python-dotenv
- ChromaDB
- LangChain
- FastAPI
- Uvicorn

---

## Setup

1. Clone the repo
```
git clone https://github.com/nbascombe/ai-engineering-projects.git
```

2. Create and activate a virtual environment
```
python3.13 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Create a Google Gemini API key - get one at aistudio.google.com

5. Create a `.env` file in the root folder
```
GOOGLE_API_KEY=your-key-here
```

6. Run a project
```
python basic_chatbot.py
python tennis_analyst_bot.py
python -m rag_foundation.rag_foundation
python -m rag_chatbot.rag_chatbot
```
FastAPI service (runs on http://localhost:8000):
```
uvicorn fastapi_chatbot:app --reload
uvicorn rag_chatbot.rag_api:app --reload
```
Test via the built-in docs UI at http://localhost:8000/docs

WebSocket test client (with the API running):
```
http://localhost:8000/static/websocket_client.html
```

---

*Nikita - AI Engineering Projects - 2026*