# TennisRulesBot

Three interfaces over the same RAG pipeline: a stateful CLI tool, a stateless HTTP API, and a stateful WebSocket endpoint. All answer questions grounded in the official 2026 ITF Rules of Tennis PDF, and correctly refuse to answer outside that scope rather than hallucinating.

```
PDF → Load → Chunk → Embed → ChromaDB → Query → Retrieve chunks → LLM → Answer
```

On first run the pipeline builds and persists the ChromaDB collection to disk. Every subsequent run - CLI or API - loads the existing collection, so no re-embedding is needed.

---

## 1. CLI - `rag_chatbot.py`

**Problem:** LLMs hallucinate confidently on precise rules questions - citing plausible but incorrect interpretations of tiebreak scoring, service rules, and fault conditions.
**Approach:** Full RAG pipeline over the official 2026 ITF Rules of Tennis PDF. Document loaded, chunked at 1200 characters, embedded with Gemini, stored persistently in ChromaDB. A single long-lived Gemini chat session maintains conversation history across the terminal session, instructed to answer only from retrieved context and explicitly say so when the answer isn't there.
**Outcome:** Answers rules questions grounded in the document; correctly refuses out-of-scope questions ("Where can I play baseball?") in all manual tests. Loads in under a second on subsequent runs - no re-embedding needed. Because it uses one persistent chat session per terminal run, follow-up questions that rely on earlier context ("what about on clay?") are understood correctly.

### Example interactions

```
You: How do I serve?
Assistant: To serve, stand at rest with both feet behind the baseline...

You: How does a tiebreak work?
Assistant: When the score in a set reaches six games all, a tie-break game
is played. The first player to reach seven points with a margin of two wins...

You: What happens on a double fault?
Assistant: The context provided does not contain information about a double
fault. The phrase does not appear in the ITF rules document — the concept is
covered across Rule 19 (Service Fault) and Rule 20 (Second Service) but the
term itself is not used.

You: Where can I play baseball?
Assistant: The context provided does not contain information about where to
play baseball.
```

The baseball example demonstrates the system correctly refusing to answer outside its knowledge base.

### Technical decisions

**Chunk size: 1200 characters** - balances retrieval precision against the Gemini free tier embedding limit of 100 requests per minute. Smaller chunks improve precision but increase the total number of API calls required to embed the document.

**Rate limiting: 0.7s delay per chunk** - keeps embedding requests within the free tier limit on first run. Has no effect on subsequent runs as the collection is loaded from disk.

**Persistent ChromaDB** - the vector store is saved to `chroma_db/` on first run. Subsequent runs skip the embed step entirely and load in under a second.

**Pages skipped: first 6** - the cover, app download page, and contents pages add no retrieval value. Skipping them reduces total chunks from 184 to 86.

**Stateful chat session** - `client.chats.create()` maintains one conversation for the lifetime of the terminal session, so the model has access to prior turns when answering follow-up questions.

### What I'd improve

- Add a confidence score to surface low-similarity retrievals before passing to the LLM
- Experiment with smaller chunk sizes (600–800 chars) to improve precision on specific rule numbers
- Add an eval harness against a set of ground-truth Q&A pairs to measure retrieval accuracy systematically

### Concepts covered

- RAG pipeline end to end - load, chunk, embed, store, retrieve, generate
- LangChain document loaders and text splitters
- Persistent vector storage with ChromaDB
- Retrieval grounding - LLM answers only from retrieved context
- Rate limit handling with controlled embedding throughput
- Stateful conversation management via a persistent chat session
- The limits of RAG - retrieval quality depends on source document terminology

---

## 2. API - `rag_api.py`

**Problem:** A CLI tool can only be used by one person, in one terminal, at a time. Wrapping the same RAG pipeline as an HTTP service makes it usable by any client - a frontend, another service, or multiple simultaneous users - without each of them needing Python installed locally.
**Approach:** FastAPI service built on the same retrieval functions as the CLI (`find_relevant_chunks`, `build_prompt`, `load_or_build_collection`). The ChromaDB collection is built once at startup via FastAPI's `lifespan` context manager and shared across all requests through `app.state`, rather than being rebuilt per request. Each request is answered with a single stateless `generate_content` call rather than a persistent chat session, since REST endpoints don't inherently carry state between calls and the endpoint currently has no mechanism for a client to signal "this is a follow-up." Requests are validated with a Pydantic model that strips whitespace and rejects empty input.
**Outcome:** A running HTTP service, testable via FastAPI's built-in Swagger UI at `/docs`, that returns grounded, accurate answers on in-scope questions and correctly refuses out-of-scope ones - matching the CLI's grounding behaviour. Confirmed via manual testing that the endpoint does **not** retain conversation history across requests (see below).

### Example interactions

```json
POST /output
{ "content": "What happens if you hit the net on your first serve?" }

{ "answer": "If the ball served touches the net, strap, or band and is otherwise
good, the service is a let. In this case, that particular service shall not
count, and the server shall serve again." }
```

```json
POST /output
{ "content": "What is the offside rule in football?" }

{ "answer": "I am sorry, but the provided context does not contain any
information about the offside rule in football." }
```

**No cross-request history** - confirmed by testing a follow-up that only makes sense with prior context:

```json
POST /output
{ "content": "Is that the same in doubles?" }

{ "answer": "The context does not provide information to answer what \"that\"
refers to in relation to doubles." }
```

This is the expected result, not a bug - the CLI's `rag_chatbot.py` keeps a single persistent chat session and would understand this follow-up; the API currently treats every request independently. See "What I'd improve" below.

### Technical decisions

**Shared collection via `app.state`, built once at startup** - rebuilding the ChromaDB collection (or re-loading it) on every request would add unnecessary latency and repeated disk/embedding work. `lifespan` builds it once when the service starts and every request reuses it.

**Sync route handler, not `async def`** - `find_relevant_chunks` and `generate_content` are both blocking calls. A plain `def` route lets FastAPI run the handler in a thread pool automatically; declaring it `async def` while calling blocking code inside it would freeze the event loop for every other in-flight request.

**Stateless per request, not a shared chat session** - the earlier version of this file used one `app.state.chat` object shared by every caller, which meant two different users could end up reading and appending to the same conversation. Rather than solve that with per-session state (see below), the current version drops server-side chat memory entirely and answers each request independently.

**`system_instruction` as the single source of persona/rules** - `build_prompt` previously duplicated the "answer only from context" instruction that also lived in `system_instruction`, left over from before the config option was added. Consolidated to avoid two versions of the same rule drifting apart over time; `build_prompt` now only formats context and question.

**Pydantic validation with a `field_validator`** - `min_length=1` alone accepts whitespace-only strings (`"   "` has length 3). A `field_validator` strips the input first and raises if empty, catching what the length constraint alone would miss.

### What I'd improve

- **Conversation history.** The endpoint currently answers each request independently. To support follow-ups, either (a) have the client send prior turns in the request body and rebuild the prompt with that history each call (fully stateless, no server memory), or (b) key a chat session per client with a session ID and expire unused sessions after a timeout. Went with neither for now since the current schema doesn't ask for it - noted here as the natural next step.
- **Rate limiting** - nothing currently stops one client from hammering the endpoint.
- **Structured request/response logging** - prompt, response, latency, token count per call.
- **Confidence scoring on retrieval**, same as the CLI's improvement list.

### Concepts covered

- Wrapping an existing RAG pipeline as a FastAPI service
- `lifespan` startup/shutdown and sharing expensive-to-build objects via `app.state`
- Sync vs async route handlers and why blocking LLM/embedding calls belong in `def`, not `async def`
- Pydantic request validation, including a custom `field_validator` beyond built-in constraints
- REST statelessness vs. conversational memory - where "state" actually needs to live (client, external store) once you move beyond a single-process CLI
- Diagnosing state bugs empirically (testing with an ambiguous follow-up to prove, not assume, statelessness)

---

## 3. WebSocket - `/ws` endpoint in `rag_api.py`

**Problem:** The stateless `/output` endpoint answers each request independently, matching expected behaviour for REST API but losing conversational memory - a genuine limitation noted in that section's "What I'd improve." A chatbot UX also benefits from tokens streaming in as they're generated rather than the client waiting for a full response.
**Approach:** Added a `/ws` WebSocket endpoint that opens a **persistent Gemini chat session per connection**, unlike `/output`'s single-shot calls. Each connection loops on `receive_text()`/`send_message_stream()`, streaming tokens back to the client as they arrive. Switched all underlying calls (`embed_content`, `send_message_stream`) to the async Gemini client (`client.aio`) rather than the sync versions used elsewhere in this file, since a blocking call inside an `async def` WebSocket handler stalls the event loop for every other connected client - not just the one making the call.
**Outcome:** A working stateful, streaming chat endpoint, tested with a minimal HTML/JS client (`static/websocket_client.html`) served via FastAPI's `StaticFiles` mount at `/static/websocket_client.html`. Verified two concurrent connections stream independently without blocking each other, and that each connection keeps its own conversational memory via its own `chat` object - unlike the shared-session bug documented in `/output`'s design history.

### Example interaction

You: What counts as in during a game?
Assistant: A ball landing in the correct court is considered a good return.

### Technical decisions

**Async Gemini client, not sync** - the sync client (`client.models.*`, used elsewhere in this file for `/output`) blocks the event loop if called inside an `async def` handler. WebSocket routes in FastAPI *must* be `async def` (no sync option, unlike HTTP routes), so blocking calls inside them stall every other open connection, not just the caller. Confirmed this by opening two browser tabs, sending messages in each within the same second, and verifying both streamed concurrently rather than one waiting on the other.

**Persistent `chat` object per connection** - created once when the connection opens (`client.aio.chats.create(...)`), reused across every message on that connection. This gives the WebSocket endpoint the conversational memory the stateless `/output` endpoint explicitly lacks, using the connection's natural lifecycle rather than a separate session-ID scheme.

**Filtering `None`/empty chunks before sending** - Gemini's streamed chunks aren't guaranteed to carry text (e.g. metadata-only chunks). `websocket.send_text(chunk.text)` raises `TypeError: data must be str, bytes-like, or iterable` if `chunk.text` is `None`. Fixed with `if chunk.text: await websocket.send_text(chunk.text)`.

**`WebSocketDisconnect` handling** - without it, a client closing their tab raises an unhandled exception in the server loop. Caught explicitly and logged.

### What I'd improve

- Session cleanup/expiry - currently every open connection holds a `chat` object in memory indefinitely; no timeout or limit on concurrent sessions
- `collection.query()` (ChromaDB) is still a sync call inside the async handler - smaller blocking cost than the network-bound embed/generate calls, but not addressed here. Could wrap with `asyncio.to_thread()`
- Compare against SSE for this exact use case - implement `/output` as proper `text/event-stream` SSE and compare against this WebSocket version directly

### System design: WebSocket vs SSE for this use case

Built as a WebSocket per the exercise, but the data flow here is one-directional per response - the client sends one question, then only listens. Even actions that seem to need bidirectional communication (stopping generation, editing a message) don't actually require sending data into an *open* stream; they're an abort of the current connection followed by a fresh request. SSE (`text/event-stream`) would fit this specific chatbot's needs with less protocol complexity - built-in reconnection via the browser's `EventSource` API, plain HTTP (fewer proxy/firewall issues than a protocol upgrade), no framing to hand-roll. WebSockets earn their complexity when the product needs genuine simultaneous two-way data - e.g. multi-user chat, voice.

### Concepts covered

- WebSocket connection lifecycle in FastAPI - `accept()`, `receive_text()`/`send_text()` loop, `WebSocketDisconnect`
- Why WebSocket handlers must be `async def`, and the consequence of blocking calls inside one
- Async vs sync Gemini client (`client.aio` vs `client.models`) and when each is required, not just preferred
- Persistent, connection-scoped state (`chat` session) vs the stateless-per-request design of `/output`
- Streaming response chunks aren't guaranteed non-empty - defensive handling required
- Empirically distinguishing a concurrency bug from LLM sampling non-determinism by holding retrieval context constant
- FastAPI `StaticFiles` mount for serving a minimal test client

---

## Prerequisites

- A Google Gemini API key - get one at aistudio.google.com
- Create a `.env` file in the repo root:
```
GOOGLE_API_KEY=your-key-here
```

## How to run

From the repo root:

CLI:
```
python -m rag_chatbot.rag_chatbot
```
Type `quit` to exit.

API (runs on http://localhost:8000):
```
uvicorn rag_chatbot.rag_api:app --reload
```
Test via the built-in docs UI at http://localhost:8000/docs

## Files

- `rag_chatbot.py` - CLI pipeline and chat loop, stateful across a terminal session
- `rag_api.py` - FastAPI service exposing the same retrieval pipeline over HTTP, stateless per request
- `documents/2026-ITF-Rules-of-Tennis.pdf` - the ITF Rules of Tennis
- `chroma_db/` - generated on first run, not committed to git
- `static/websocket_client.html` - minimal HTML/JS test client for the `/ws` endpoint