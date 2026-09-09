# TennisRulesBot

Three interfaces over the same RAG pipeline: a stateful CLI tool, a stateless HTTP API (now with Redis cache-aside caching on repeat questions), and a stateful WebSocket endpoint. All answer questions grounded in the official 2026 ITF Rules of Tennis PDF, and correctly refuse to answer outside that scope rather than hallucinating.

```
PDF → Load → Chunk → Embed → ChromaDB → Query → Retrieve chunks → LLM → Answer
```
The HTTP API additionally checks a Redis cache before running this pipeline - on a cache hit, embedding, retrieval, and generation are skipped entirely. See the Caching section below.

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
- **Rate limiting** - nothing currently stops one client from hammering the endpoint. (Caching in section 4 reduces repeat-question load, but doesn't rate-limit distinct questions.)
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

## 4. Caching - cache-aside pattern on the HTTP API (`/output` only, not `/ws`)

**Problem:** Every call to `/output` re-embeds the question, re-queries ChromaDB, and re-generates a full response from Gemini even when the exact same question has already been answered. For a fixed source document (the rules don't change), that's wasted latency and API cost on repeat questions.
**Approach:** Implemented cache-aside caching in `generate_tokens`: on each request, normalise the question (`.strip().lower()`) and check Redis first. On a hit, return the stored answer directly, skipping embedding, retrieval, and generation entirely. On a miss, run the full pipeline as before, but collect the streamed chunks into a single string and write it to Redis with a TTL before returning. Chose a long TTL (6000s) deliberately, since the source PDF is static — a low TTL would suit a system where the underlying documents change regularly, but here it just causes unnecessary cache churn.
**Outcome:** Verified with `time curl` on the identical question: a cache miss took ~4.95s (embedding + retrieval + generation), a cache hit took ~0.034s — roughly a 145x speedup. Confirmed via `redis-cli` and a Python shell that keys and values are stored as expected. Also verified normalisation works — the same question with different casing and leading whitespace (`"      what is a tiebreak?"`) correctly hit the same cache entry rather than creating a duplicate. The bytes/str inconsistency between cache hit and miss branches, found during testing, was resolved with decode_responses=True on the Redis connection.

### Example
```
time curl -X POST http://127.0.0.1:8000/output -d '{"content": "what is a tiebreak?"}'
```
#### Cache miss
real 0m4.954s

```
time curl -X POST http://127.0.0.1:8000/output -d '{"content": "what is a tiebreak?"}'
```
#### Cache hit
real 0m0.034s

### Technical decisions

**Cache key: normalised question text, not the raw string** - the first implementation used the raw `question.content` as the Redis key, which meant `"What is a tiebreak?"` and `"what is a tiebreak?"` were treated as two different entries. Found this by inspecting `redis-cli KEYS *` and seeing a lookup miss on a key I could see was clearly stored. Fixed by lowercasing with `.lower()` before both the `r.get` and `r.set` calls, so casing no longer fragments the cache. Surrounding whitespace is already handled upstream — the Pydantic `field_validator` on `Question.content` strips it before the request reaches this function — so the cache key doesn't need its own `.strip()` call on top of that.

**Consuming the stream before caching, not caching the stream object** - `generate_content_stream` returns a live generator; it can't be handed directly to `r.set()` since Redis only stores primitive types. Instead, each `chunk.text` is yielded to the client *and* appended to a list as it arrives, then joined into a single string (`" ".join(chunks)`) once the stream is exhausted, that string is what gets cached.

**Cache hit yields once, not chunk-by-chunk** - the miss branch streams token-by-token as Gemini generates them; the hit branch yields the whole cached string in a single `yield`. This is a deliberate asymmetry: the client only needs the correct final text delivered as *a* stream, not the exact same chunk boundaries the original generation happened to produce.

**Redis connection with `decode_responses=True`** - `r.get()` returns `bytes` by default, while the cache-miss branch yields `str` (`chunk.text`) - a real type inconsistency between the two code paths, found while testing. Rather than decoding at the single call site (`cache_response.decode("utf-8")`), fixed it at the connection level with `redis.Redis(decode_responses=True)`, so every `r.get()` on this connection returns `str` automatically. A connection-level fix so any future Redis reads added to this file don't need to remember to decode individually - though worth noting it assumes every value on this connection is text, which holds here but wouldn't if binary data were ever cached on the same connection.

### What I'd improve

- **Semantic caching** - exact-match-after-normalisation still misses paraphrases (`"what's a tiebreak?"` vs `"what is a tiebreak?"`). Would need embedding-based similarity lookup against cached questions instead of a literal key match.
- **Cache invalidation strategy** - currently relies purely on TTL expiry. If the source PDF changes, there's no mechanism to invalidate cached answers immediately.

### Update: Async cache writes

**Problem:** The original cache-aside implementation used a synchronous Redis client. `r.set()` ran after the full response had already streamed to the client, meaning the request's worker thread sat idle waiting on a write the caller no longer needed to wait on.

**Approach:** Switched to `redis.asyncio` and converted `generate_tokens` and the `/output` route to `async def`. The cache write is now wrapped in `asyncio.create_task()` rather than awaited directly, so it fires in the background instead of blocking the generator from finishing.

**Outcome:** Verified the write still lands reliably - sent 5 fresh (uncached) questions, waited for each response to fully complete, then confirmed via `redis-cli GET` that every key was present with no failures, despite the fire-and-forget pattern. No garbage-collection issues observed in this testing, though this wasn't stress-tested under concurrent load.

### Benchmark: cache hit vs miss (post async-write change)

Measured with `time curl`, N=10 per case, wall-clock `real` time:

- **Cache hit** (same question repeated): ~0.050s average (first call 0.103s as a warm-up outlier, excluded)
- **Cache miss** (10 distinct, never-asked questions): ~3.18s average, range 1.93s–6.81s

**~64x speedup** on a hit vs a miss. Two things worth noting about these numbers:
- Miss-case variance is wide and expected - it reflects Gemini's response latency for that specific question and answer length, not the caching layer. Hit-case timing is far tighter (Redis + FastAPI overhead is deterministic; LLM generation is not).
- One outlier (6.81s) pulls the mean up noticeably; a median across the same 10 misses would likely tell a firmer story.

### Technical decisions (async writes)

**`asyncio.create_task`, not `await`, for the write** - awaiting `r.set()` would still hold the generator open until the write finished, which defeats the point. `create_task` starts it and lets the generator (and the response) complete independently.

**Converted the route to `async def`** - with `generate_tokens` now an async generator, `content_output` needed to become `async def` too rather than relying on FastAPI's thread-pool fallback for sync routes.

### What I'd improve (async writes)

- `find_relevant_chunks` inside `generate_tokens` is still a fully synchronous call (embedding + Chroma query) now sitting inside an `async def` route - a candidate for `asyncio.to_thread()` so it doesn't block the event loop the way the WebSocket README section warns against.
- Task lifetime isn't explicitly managed - `asyncio.create_task()` results aren't held in a persistent reference, which is a known asyncio footgun (tasks can be garbage-collected before completion). Not observed as a problem in manual testing, but not stress-tested under concurrent rapid-fire requests either.
- Benchmark used mean, not median - a single slow outlier in the miss set skews the average upward; median would be more robust for a small sample size.


### Concepts covered

- Cache-aside pattern - check cache, miss → do the work → populate cache, hit → skip the work
- Redis SET/GET with TTL
- Why LLM calls specifically benefit from caching - I/O-bound latency, not CPU-bound, confirmed by `real` time dropping ~145x while `user`/`sys` stayed flat
- Cache key design - exact-string match vs normalisation, and the fragility of unnormalised keys
- Consuming a generator to produce a single cacheable value while still streaming it to the original caller
- Fixing a type inconsistency at the connection level (`decode_responses=True`) rather than patching each call site individually

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
- `rag_api.py` - FastAPI service exposing the same retrieval pipeline over HTTP, stateless per request, with cache-aside caching on `/output`
- `documents/2026-ITF-Rules-of-Tennis.pdf` - the ITF Rules of Tennis
- `chroma_db/` - generated on first run, not committed to git
- `static/websocket_client.html` - minimal HTML/JS test client for the `/ws` endpoint