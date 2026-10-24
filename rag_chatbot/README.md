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

*Note: this is an early transcript. The rule numbers in the double-fault reply are the model's own and don't match the eval set, which sources this rule to Rule 24a (\"serves two consecutive faults\"). The reply is kept because it shows the wording gap (\"double fault\" vs the PDF's vocabulary) that became eval question Q8 in section 6.*

### Technical decisions

**Chunk size: 1200 characters** - balances retrieval precision against the Gemini free tier embedding limit of 100 requests per minute. Smaller chunks improve precision but increase the total number of API calls required to embed the document.

**Rate limiting: 0.7s delay per chunk** - keeps embedding requests within the free tier limit on first run. Has no effect on subsequent runs as the collection is loaded from disk.

**Persistent ChromaDB** - the vector store is saved to `chroma_db/` on first run. Subsequent runs skip the embed step entirely and load in under a second.

**Pages skipped: first 6** - the cover, app download page, and contents pages add no retrieval value. Skipping them reduces total chunks from 184 to 86.

**Failed builds clean up after themselves** - if embedding fails part way (quota, network, Ctrl-C), the partially built collection is deleted. Without this, the next run's `get_collection()` succeeds and silently serves a half-built index.

**Stateful chat session** - `client.chats.create()` maintains one conversation for the lifetime of the terminal session, so the model has access to prior turns when answering follow-up questions.

**Retrieval k: 3** - the CLI still retrieves 3 chunks. The k=8 change (section 6) was validated on the API's retrieval and prompt path, not on the CLI's persistent chat session, so the CLI was left alone.

### What I'd improve

- Add a confidence score to surface low-similarity retrievals before passing to the LLM
- Experiment with smaller chunk sizes (600–800 chars) to improve precision on specific rule numbers
- Add an eval harness against a set of ground-truth Q&A pairs to measure retrieval accuracy systematically (Done in section 6 below: Evaluation - eval harness for TennisRulesBot)

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

*(Superseded: `/output` is now fully async - async Redis client, async embedding, `asyncio.to_thread` for the Chroma query and `client.aio` streaming. See section 4's async-writes and fully-async updates. The reasoning below explains the original sync design.)*
**Problem:** A CLI tool can only be used by one person, in one terminal, at a time. Wrapping the same RAG pipeline as an HTTP service makes it usable by any client - a frontend, another service, or multiple simultaneous users - without each of them needing Python installed locally.
**Approach:** FastAPI service built on the same retrieval functions as the CLI (`find_relevant_chunks`, `build_prompt`, `load_or_build_collection`). The ChromaDB collection is built once at startup via FastAPI's `lifespan` context manager and shared across all requests through `app.state`, rather than being rebuilt per request. Each request is answered with a single stateless `generate_content` call rather than a persistent chat session, since REST endpoints don't inherently carry state between calls and the endpoint currently has no mechanism for a client to signal "this is a follow-up." Requests are validated with a Pydantic model that strips whitespace and rejects empty input.
**Outcome:** A running HTTP service, testable via FastAPI's built-in Swagger UI at `/docs`, that returns grounded, accurate answers on in-scope questions and correctly refuses out-of-scope ones - matching the CLI's grounding behaviour. Confirmed via manual testing that the endpoint does **not** retain conversation history across requests (see below).

### Example interactions

`/output` streams its answer as `text/plain` chunks. The bodies below are the concatenated stream.

The `-H "Content-Type: application/json"` header is required. Without it, `curl -d` sends a form-encoded body and current FastAPI versions reject it with a 422 (`model_attributes_type`) before the endpoint runs.

```
curl -X POST http://localhost:8000/output -H "Content-Type: application/json" -d '{"content": "What happens if you hit the net on your first serve?"}'

If the ball served touches the net, strap, or band and is otherwise good, the
service is a let. In this case, that particular service shall not count, and
the server shall serve again.
```

```
curl -X POST http://localhost:8000/output -H "Content-Type: application/json" -d '{"content": "What is the offside rule in football?"}'

I am sorry, but the provided context does not contain any information about
the offside rule in football.
```

**No cross-request history** - confirmed by testing a follow-up that only makes sense with prior context:

```
curl -X POST http://localhost:8000/output -H "Content-Type: application/json" -d '{"content": "Is that the same in doubles?"}'

The context does not provide information to answer what "that" refers to in
relation to doubles.
```

This is the expected result, not a bug - the CLI's `rag_chatbot.py` keeps a single persistent chat session and would understand this follow-up; the API currently treats every request independently. See "What I'd improve" below.

### Technical decisions

**Shared collection via `app.state`, built once at startup** - rebuilding the ChromaDB collection (or re-loading it) on every request would add unnecessary latency and repeated disk/embedding work. `lifespan` builds it once when the service starts and every request reuses it.

**Retrieval k: 8, from `K` in `rag_common.py`** - the API (`/output` and `/ws`) retrieves 8 chunks per question. It started at 3; the move to 8 was validated on a dev set and a held-out set (section 6). Defining `K` once keeps the two endpoints and the eval harness in step.

**Sync route handler, not `async def`** - `find_relevant_chunks` and `generate_content` are both blocking calls. A plain `def` route lets FastAPI run the handler in a thread pool automatically; declaring it `async def` while calling blocking code inside it would freeze the event loop for every other in-flight request.

**Stateless per request, not a shared chat session** - the earlier version of this file used one `app.state.chat` object shared by every caller, which meant two different users could end up reading and appending to the same conversation. Rather than solve that with per-session state (see below), the current version drops server-side chat memory entirely and answers each request independently.

**`system_instruction` as the single source of persona/rules** - `build_prompt` previously duplicated the "answer only from context" instruction that also lived in `system_instruction`, left over from before the config option was added. Consolidated to avoid two versions of the same rule drifting apart over time; `build_prompt` now only formats context and question.

**Pydantic validation with a `field_validator`** - `min_length=1` alone accepts whitespace-only strings (`"   "` has length 3). A `field_validator` strips the input first and raises if empty, catching what the length constraint alone would miss.

### What I'd improve

- **Conversation history.** The endpoint currently answers each request independently. To support follow-ups, either (a) have the client send prior turns in the request body and rebuild the prompt with that history each call (fully stateless, no server memory), or (b) key a chat session per client with a session ID and expire unused sessions after a timeout. Went with neither for now since the current schema doesn't ask for it - noted here as the natural next step.
- **Rate limiting** - nothing currently stops one client from hammering the endpoint. (Caching in section 4 reduces repeat-question load, but doesn't rate-limit distinct questions.)
- **Structured request/response logging** - done, see section 7.
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
- ~~`collection.query()` (ChromaDB) is still a sync call inside the async handler~~ - done: `find_relevant_chunks_async` now runs the query via `asyncio.to_thread()` (see section 4's fully-async update)
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
time curl -X POST http://127.0.0.1:8000/output -H "Content-Type: application/json" -d '{"content": "what is a tiebreak?"}'
```
#### Cache miss
real 0m4.954s

```
time curl -X POST http://127.0.0.1:8000/output -H "Content-Type: application/json" -d '{"content": "what is a tiebreak?"}'
```
#### Cache hit
real 0m0.034s

### Technical decisions

**Cache key: normalised question text, not the raw string** - the first implementation used the raw `question.content` as the Redis key, which meant `"What is a tiebreak?"` and `"what is a tiebreak?"` were treated as two different entries. Found this by inspecting `redis-cli KEYS *` and seeing a lookup miss on a key I could see was clearly stored. Fixed by lowercasing with `.lower()` before both the `r.get` and `r.set` calls, so casing no longer fragments the cache. Surrounding whitespace is already handled upstream — the Pydantic `field_validator` on `Question.content` strips it before the request reaches this function — so the cache key doesn't need its own `.strip()` call on top of that.

**Consuming the stream before caching, not caching the stream object** - `generate_content_stream` returns a live generator; it can't be handed directly to `r.set()` since Redis only stores primitive types. Instead, each `chunk.text` is yielded to the client *and* appended to a list as it arrives, then joined with `"".join(chunks)` once the stream is exhausted. Joining with a space was an early bug: Gemini's chunks are slices of one continuous string and can split mid-word ("Tie" / "-Break"), so a space-joined cache entry read "Tie -Break" and differed from the streamed answer. Found by comparing LangSmith's recorded chunk boundaries against `redis-cli GET`.

**Cache hit yields once, not chunk-by-chunk** - the miss branch streams token-by-token as Gemini generates them; the hit branch yields the whole cached string in a single `yield`. This is a deliberate asymmetry: the client only needs the correct final text delivered as *a* stream, not the exact same chunk boundaries the original generation happened to produce.

**Redis connection with `decode_responses=True`** - `r.get()` returns `bytes` by default, while the cache-miss branch yields `str` (`chunk.text`) - a real type inconsistency between the two code paths, found while testing. Rather than decoding at the single call site (`cache_response.decode("utf-8")`), fixed it at the connection level with `redis.Redis(decode_responses=True)`, so every `r.get()` on this connection returns `str` automatically. A connection-level fix so any future Redis reads added to this file don't need to remember to decode individually - though worth noting it assumes every value on this connection is text, which holds here but wouldn't if binary data were ever cached on the same connection.

### Update: namespaced, versioned cache keys

**Problem:** Keys were the bare lowercased question, so a change to the model, system instruction, retrieval k or chunking would keep serving answers generated under the old setup until the TTL expired.
**Approach:** Keys are now `rag:<CACHE_VERSION>:<fingerprint>:<question>`. The fingerprint is a short hash of `MODEL` and `SYSTEM_INSTRUCTION`, so editing either invalidates the cache automatically. `CACHE_VERSION` (a constant in `rag_api.py`) is bumped by hand for changes the fingerprint can't see: retrieval k, chunk size, the prompt template or the source PDF.
**Outcome:** Entries written under the old scheme are never read again and expire on their own within the 6000s TTL, so the only visible effect of the change is one extra miss per question. Nothing else reads these keys (the eval bypasses the cache). The `rag:` prefix also lets `redis-cli --scan --pattern 'rag:*'` find the app's keys.

**Update (retrieval k=8):** `CACHE_VERSION` was bumped from `v1` to `v2` when the API's retrieval k went from 3 to 8. The fingerprint hashes only `MODEL` and `SYSTEM_INSTRUCTION`, so it can't see a retrieval change. Entries written under `v1` are never read again and expire within the 6000s TTL.

### What I'd improve

- **Semantic caching** - exact-match-after-normalisation still misses paraphrases (`"what's a tiebreak?"` vs `"what is a tiebreak?"`). Would need embedding-based similarity lookup against cached questions instead of a literal key match.
- **Cache invalidation strategy** - keys are now versioned (see the update above), so a model or prompt change invalidates automatically, but the source PDF and retrieval k still rely on remembering to bump `CACHE_VERSION`. Hashing the PDF into the fingerprint would remove that step.

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

- ~~`find_relevant_chunks` inside `generate_tokens` is still synchronous~~ - done, see the fully-async update below.
- ~~Task lifetime isn't explicitly managed~~ - done: tasks are held in a `background_tasks` set until complete, and `lifespan` awaits any in-flight writes on shutdown. Still not stress-tested under concurrent rapid-fire requests.
- Benchmark used mean, not median - a single slow outlier in the miss set skews the average upward; median would be more robust for a small sample size.


### Update: fully async request path

**Problem:** `/output` was `async def` but its retrieval (sync embed + Chroma query) and generation (sync streaming iterator) still blocked the event loop on every cache miss, stalling all other in-flight requests.
**Approach:** Retrieval now goes through `find_relevant_chunks_async` (async embed, Chroma query via `asyncio.to_thread`), generation uses `await client.aio.models.generate_content_stream(...)` with `async for`, and cache-write tasks are kept in a `background_tasks` set. Empty responses are not cached. The sync `find_relevant_chunks` remains for the CLI and the eval harness.
**Outcome:** Two simultaneous cache misses on `/output` finished in 1.87s and 2.74s (cache keys cleared first), so one slow request no longer delays another. Single manual run, no sync baseline. Still to check: that LangSmith's `wrap_gemini` still traces the `aio` streaming call (the `retrieve_chunks` run is preserved by decorating the async function too).

### Concepts covered

- Cache-aside pattern - check cache, miss → do the work → populate cache, hit → skip the work
- Redis SET/GET with TTL
- Why LLM calls specifically benefit from caching - I/O-bound latency, not CPU-bound, confirmed by `real` time dropping ~145x while `user`/`sys` stayed flat
- Cache key design - exact-string match vs normalisation, and the fragility of unnormalised keys
- Consuming a generator to produce a single cacheable value while still streaming it to the original caller
- Fixing a type inconsistency at the connection level (`decode_responses=True`) rather than patching each call site individually

---

## 5. Observability - LangSmith tracing on `/output`

**Problem:** The pipeline had several steps (cache check, embed, retrieve, generate) and I was judging it only by the final answer. When an answer was wrong or slow, I couldn't tell which step caused it.
**Approach:** Added LangSmith tracing. The Gemini client is wrapped with `wrappers.wrap_gemini` (beta), and `@traceable` decorators on `generate_tokens` (root run `rag_request`), `find_relevant_chunks` (`retrieve_chunks`, run type retriever) and `embed_text`. Nesting is automatic, so one request becomes one trace.
**Outcome:** Each request now produces a trace tree:

```
rag_request
├── retrieve_chunks
│   └── embed_text
└── ChatGoogleGenerativeAI (the Gemini call)
```

- **Latency split.** On a representative cache miss (7.43s total), embedding took 1.62s and generation 5.42s, so the ChromaDB query itself was only ~0.14s. Other misses ranged from ~3s to ~7s, so generation latency is the most variable cost. A cache hit is a single run with no children (~0.00s).
- **A cache bug found via traces.** Comparing the recorded stream chunks against the stored Redis value showed that `" ".join(chunks)` corrupted cached answers ("Tie -Break"). Fixed with `"".join(chunks)`.
- **A retrieval problem found via traces.** "What is a tiebreak?" kept giving incomplete, inconsistent answers. The trace showed the three retrieved chunks were the match tie-break and short-set variants, not the standard set tie-break rule. An early diagnostic script (`check_retrieval.py`, since replaced by `eval_rag.py --ranks`) showed that rule chunk (`chunk_5`) ranked 4th for "what is a tiebreak?" and "When is a tie-break played?", but 1st for "What happens when a set reaches 6-6?". *These two figures are for different chunks, not conflicting measurements. `chunk_5` is the Rule 6b chunk (when a tie-break starts: "six games all"), which `--ranks` confirms is 1st for the 6-6 question. `chunk_3` is the Rule 5b chunk (the definition: "seven points wins the"), which `--ranks` puts 6th for "What is a tiebreak?". The early diagnostic was tracking the start-condition chunk; the eval set targets the definition chunk, which is the more precise answer to that question. I have not re-measured the original 4th-place figure for `chunk_5`.* The model's answers were faithful to the retrieved context, so this is a context-recall failure at k=3, not a hallucination.

### Technical decisions

**`wrap_gemini` plus `@traceable`, not LangChain chains.** The project calls the `google-genai` SDK directly, so nothing is traced automatically. The wrapper captures the generation call, and decorators cover the steps I wrote myself.
**Free-tier constraints.** The free Developer plan has a monthly trace cap and 14-day retention, so findings worth keeping are recorded here and in my notes rather than relying on the trace history.

### What I'd improve

- Retrieval is wording-sensitive and the PDF says "six games all" while users type "6-6". Candidates: smaller chunks with more overlap, query rewriting, hybrid keyword search.
- Cached answers are served for the full TTL, so a weak answer is frozen for 100 minutes. Consider not caching refusals or low-confidence answers.
- The same question gave different answers across runs. Evals should run each question several times, and `temperature` could be lowered. The eval harness supports `--runs N`; free-tier quota has limited it to 1 run per question so far.
- `/ws` is not traced. Each message needs its own traced function, with a shared thread ID to group a conversation.
- ~~`find_relevant_chunks` (sync embed + Chroma query) still runs inside an `async def` route and blocks the event loop~~ - done in section 4's fully-async update (async embed, `asyncio.to_thread` for the query).

### Concepts covered

- Traces vs runs: one request is one trace, made of nested runs
- Instrumenting code that doesn't use a framework (`@traceable`, `wrap_gemini`)
- Diagnosing retrieval vs generation failures from a trace
- Faithfulness vs context recall: a faithful answer can still be wrong if retrieval missed the key chunk
- Region-specific API endpoints (the EU endpoint fixed a 403 on trace ingestion)
- Why caching can freeze a bad answer, and why checking cache-hit text against miss text matters

---

## 6. Evaluation - eval harness for TennisRulesBot

**Problem:** I was judging the bot by eyeballing answers, so I couldn't say how often it was right, or whether a wrong answer came from retrieval (the right chunk never reached the model) or generation (the model had it and still got it wrong).
**Approach:** Built an eval harness (`eval_rag.py`) over 15 hand-written questions in `eval_set.json`. Each question has keyword checks for the answer and, where applicable, an `expected_phrase` that identifies the source chunk, so retrieval and generation are scored separately. The set mixes wording mismatches, numeric lookups, a question whose answer differs for wheelchair tennis, a multi-part question, a scenario question, and two out-of-scope questions (one clearly unrelated, one a tennis question the rules can't answer). It reuses the API's retrieval and prompt code, bypasses the Redis cache, and records API errors as errors, not failures.
**Outcome:** At the original production setting (k=3) the bot scored 11/15 end to end on two separate runs (9 clean, 2 partial, 4 failed after reading the answers). Retrieval was identical both times (9/13, same four misses: Q1, Q4, Q8, Q13), and every failure traced back to it. The same four questions failed both times, but not identically: Q8 echoed its system instruction on the first run and gave a clean refusal on the second, and Q1 gave a partial refusal and then a grounded but misleading answer (a discretionary variant rule from the retrieved context, stated without its "at the discretion of the sanctioning body" condition). Raising k to 8 recovered all four failures and scored 15/15 on two separate runs, with both out-of-scope refusals intact.

Cost of k=8 vs k=3, measured from the call log on fresh uncached runs (one run each, on different days, so the latency difference is within noise). Mean tokens per query at k=3 / k=8: prompt 735 / 1,985 (2.7x), visible output 39 / 67, thinking 299 / 487, total 1,073 / 2,539 (2.4x). Median latency 2.07s / 2.36s. Thinking tokens outnumber visible output tokens by roughly 7 to 8x at both settings, so the visible output count alone badly understates generation cost. The API now uses k=8 (see "Held-out validation and the decision to adopt k=8" below).

Scoring was tightened after the second k=3 run (see Limitations). I re-scored all four saved reports with `--rescore`: k=3 stayed at 11/15 on the first run and corrected from 12/15 to 11/15 on the second, and k=8 stayed 15/15 on both.

| | Dev k=3 | Dev k=8 | Held-out k=3 | Held-out k=8 |
|---|---|---|---|---|
| Retrieval hit rate | 9/13 | 13/13 | 11/13 | 13/13 |
| End-to-end, keyword scoring | 11/15 (11/15 on rerun) | 15/15 (15/15 on rerun) | 13/15 | 15/15 |
| End-to-end, after reading answers | 9 clean, 2 partial, 4 failed (first run) | the 4 failures fixed, same 2 partials | 13 pass, 2 real failures | 15 pass, 1 softer framing (Q5) |
| Out-of-scope refusals | 2/2 (Q10, Q15) | 2/2 | 2/2 (Q14, Q15) | 2/2 |
| Median latency (non-streamed) | 2.07s | 2.36s | 1.84s | 1.80s |
| Mean prompt tokens | 735 | 1,985 | 740 | 1,938 |
| Mean output tokens (visible) | 39 | 67 | 34 | 47 |
| Mean thinking tokens | 299 | 487 | 281 | 266 |
| Mean total tokens | 1,073 | 2,539 | 1,055 | 2,251 |
| Mean context chars | 2,592 | 7,322 | 2,622 | 7,107 |

Dev question numbers (Q10, Q15) and held-out question numbers (Q14, Q15) refer to different files.

At k=3, when retrieval found the right chunk the answer was right (9/9 passed the check, 7 fully clean). When it didn't, the answer was wrong (0/4). Retrieval is the bottleneck, not generation.

### Which saved report backs which claim

| Claim | File in `eval_results/` | Notes |
|---|---|---|
| k=3 baseline: retrieval 9/13, answers 11/15 | `eval_20261016_135335_baseline_k3.json` | Answers came from the eval cache, so no latency or token data |
| k=3 fresh rerun: 11/15 | `eval_20261017_101346_k3_fresh.json` | **The saved summary says 12/15.** That was scored before Q1's keyword check was tightened (a wrong answer matched "two" and "7"). `--rescore` gives 11/15; the saved file is left unedited as evidence. Cost figures at k=3 (735 prompt / 39 output tokens, 2.07s) come from this file |
| k=8: retrieval 13/13, answers 15/15 | `eval_20261016_133920_k8.json` | Answers from cache |
| k=8 fresh rerun: 15/15, cost figures | `eval_20261016_163457_k8_fresh.json` | 1,985 prompt / 67 output tokens, 2.36s median. Thinking tokens are only in the call log (`logs/sample_llm_calls.jsonl` has a trimmed sample) |
| Held-out k=3: retrieval 11/13, answers 13/15 | `eval_20261023_101240_heldout_k3.json` | Fresh, uncached, `--eval-set eval_set_heldout.json`. Includes thinking and total tokens (mean total 1,055) |
| Held-out k=8: retrieval 13/13, answers 15/15 | `eval_20261024_114824_heldout_k8.json` | Fresh, uncached (mean total 2,251). Run after the decision rule was fixed |

To reproduce the corrected score: `python -m rag_chatbot.eval_rag --rescore rag_chatbot/eval_results/eval_20261017_101346_k3_fresh.json`

### Why retrieval missed

At k=3 the answer chunk ranked 5th to 8th for every miss (Q4 5th, Q1 and Q8 6th, Q13 8th). Two causes:
- **Vocabulary the PDF doesn't use.** Users say "tiebreak" and "double fault"; the rules say "tie-break game" and "serves two consecutive faults".
- **Answers inside example cases.** Q13's answer sits in a Case under Rule 23, not in the rule text.

Q12 (a two-part question) first looked like a third cause, with its chunk ranked 7th, but that came from an `expected_phrase` that was too narrow: the 90 seconds is stated in more than one chunk. With the phrase widened it hits at #1 at k=3, so compound questions were a scoring artefact, not a retrieval cause.

### How the four k=3 failures failed

| Q | Run 1 | Run 2 | Failure type |
|---|---|---|---|
| 1 | Said the context gives no general definition of a tie-break | Described match and short-set tie-breaks only and said a tie-break may be played at "three games all", which is not the standard rule. The phrase was in the retrieved context (checked in the call log), where the rule applies "at the discretion of the sanctioning body"; the answer dropped that qualifier and presented it as a general alternative | Partial refusal, then a grounded answer that dropped a limiting condition |
| 13 | "The answer is not in the context." | Same | Clean refusal |
| 4 | Said a serve touching the net is good and playable | Same: answered from the No-Let rule | Confident wrong answer: used the optional No-Let rule from Appendix VI instead of Rule 22 (service let), presenting it as the answer to a question that never mentioned No-Let |
| 8 | Echoed its own system instruction | "The context does not provide information on what happens on a double fault." | Degenerate output once, then a clean refusal |

Refusals are the safe failure. The Q4 type, a confident answer from the wrong rule, is the one a user would believe. Q4 reproduced on both runs, so the confident wrong answer is a stable failure at k=3, not a one-off.

### Raising k: k=3 vs k=8

Retrieval-only runs on the first nine questions gave 6/9 at k=3, 7/9 at k=5 and 9/9 at k=6, which made k=6 look like a full fix. Six questions written to be different in kind showed it wasn't: answer chunks ranked as deep as 8th, and k=8 (about 10% of the 86 chunks) was the smallest k with 13/13 retrieval. Generation at k=8 (15 fresh calls, 0 errors):
- **Q4** now picked the standard Rule 22 (service let) even with the Appendix VI No-Let chunk in context.
- **Q8 and Q13** gave clean, correct answers. Q8's echo appeared once at k=3 and not on the rerun, so it is intermittent rather than fixed or reproduced.
- **Q1** passes on content but the answer is muddier than the others.
- **Q6 and Q14** were unchanged. A bigger k fixes retrieval failures, not incomplete answers or lenient checks.
- **Both refusals held** with 8 chunks in context.

### Held-out validation and the decision to adopt k=8

**Setup.** k=8 was chosen on the dev set, so I wrote 15 new questions in `eval_set_heldout.json` (13 answerable, 2 out-of-scope) on rules the dev set doesn't touch: medical time-outs, doubles service order, hindrance, coaching, ball changes, net contact, racket size, rest periods, the warm-up and deuce. They are run with `--eval-set`. One edit was made after validation started: Q8's `expected_phrase` straddled a chunk boundary and matched no chunk (caught by `--ranks`), so I replaced it with a shorter phrase from the same sentence. That is a validity fix, not tuning. No question, check or threshold was changed after seeing generation results.

**Decision rule, fixed before the generation runs:** adopt k=8 if retrieval hit rate improves, answer accuracy at k=8 is no worse than at k=3, and both out-of-scope refusals hold.

**Results** (one run each; see the combined table above): retrieval 11/13 to 13/13, answers 13/15 to 15/15, refusals 2/2 at both settings, about 2.1x total tokens.

**Reading the answers.**
- Both k=3 failures were retrieval-driven. Q9 ("Can I volley the serve back?", where the PDF says the receiver "returns the service before it bounces"; answer chunk ranked 5th) said the context didn't cover it. Q11 (chunk ranked 4th) gave only the 120-second set break, which is true but incomplete. The question is also ambiguous, since a 120s break does follow every set. I left its check unchanged.
- k=8 fixed both. Q9 now says the receiver loses the point, and Q11 gives the optional ten-minute rest, announced in advance. The other 13 answers were materially unchanged.
- The refusals (Q14, Q15) held at both settings with more context in the prompt, and Q14 did not name the winner. The deuce trap (Q13, an optional deciding point) did not fire at either k.
- Q5 (coaching) got slightly worse in framing. At k=8 it opens "Yes, your coach can give you advice during a match, subject to certain conditions", where k=3 opened with "may be permitted if allowed by the sanctioning body". The conditions are present in both, so it passed, but the headline is more permissive. I left the check alone and recorded it here.

**Decision.** All three conditions were met, so the API now retrieves 8 chunks (`K` in `rag_common.py`) and `CACHE_VERSION` is `v2`.

**What this does and doesn't show.** Two independent sets agree (retrieval 9/13 to 13/13 on dev, 11/13 to 13/13 on held-out), and every failure fixed was a wording gap, not a generation problem. But these are single runs on 15 questions each, so one question is within noise. k=8 also treats a retrieval weakness (the k needed to hit every question was 8 on dev and 5 on held-out), and both sets have now been inspected, so any further tuning needs a third set.

**Cost.** Prompt tokens scale reliably with k (about 2.6x on both sets), and total tokens were 2.4x on dev and 2.1x on held-out. Thinking tokens did not consistently rise with k: 299 to 487 on dev, but 281 to 266 on held-out. Individual questions swing widely (Q5 used 800+ thinking tokens at both settings), so thinking cost depends on the question more than on k.

### Technical decisions

- **Retrieval and generation scored separately**, since a faithful answer can still be wrong if retrieval missed the key chunk.
- **Errors are not failures.** Quota errors are excluded from accuracy, the run stops after 3 consecutive API errors, and incomplete questions are named in the report.
- **Eval-only answer cache** (`eval_results/answer_cache.json`, git-ignored), keyed on model, system instruction, prompt and run index, so reruns only pay for missing answers. This kept the eval within the free tier's 20 generation calls per day.
- **Expected phrases checked against the stored chunks** before trusting any retrieval number. All nine initial phrases existed, so the misses were real.
- **Re-scoring saved reports.** `--rescore <report.json>` re-applies the current `eval_set.json` checks to a saved report with no API calls. After tightening a check, this showed which past scores changed without spending quota or editing the saved evidence.

### Limitations

- Two sets of 15 questions I wrote. The dev set was partly adjusted after seeing results and k=8 was chosen on it. The held-out set confirmed the choice, but both sets have now been inspected, so neither can validate further tuning. Neither is a statistical estimate of real-world accuracy.
- One run per question per setting, and the model is non-deterministic: failure text varied between the two k=3 runs (Q1, Q8) even though pass/fail didn't.
- Keyword scoring is crude. Q6 and Q14 passed without fully answering, and Q1's first check passed a wrong k=3 answer ("two" matched "two sets all", "7" matched "7-point"). I only caught both by reading the raw answers, then tightened Q1's check and re-scored all saved runs.
- Dev-set token figures were computed from the call log, splitting k=3 and k=8 records by prompt length, because those reports' summaries predate thinking and total tokens. The held-out reports include them directly.
- Held-out Q11 is ambiguous (a 120s set break does follow every set), so its k=3 failure is partly a question-design problem. I left it in rather than edit it after seeing the result.
- The eval harness uses the sync retrieval and generation path, so these numbers predate the async changes to `/output`. They should be unaffected (same chunks, model and prompt), but the API path itself hasn't been re-evaluated.
- A refusal check on "not in the context" would also accept an echo of the system prompt (the harness now rejects the known echo markers, but not every degenerate output).

### What I'd improve

- Write a third, fresh question set before any further tuning, since both existing sets have been inspected. Test a lower thinking budget for this task: thinking is several times the visible output and dominates generation tokens.
- Judge retrieval by whether the context supports the answer, and use an LLM judge for answers. Reject degenerate outputs before scoring them as refusals.
- Fix retrieval itself (query rewriting, smaller chunks, hybrid keyword search or reranking) so a smaller k works, and test it on that fresh set. k=8 treats the symptom.
- Label or restrict optional and discretionary variant rules (No-Let, short sets, "at the discretion of the sanctioning body" alternatives) so the model can't present them as the default rule or drop their conditions. This is a retrieval-side fix that works at any k; Q4 (No-Let) and Q1 ("three games all") both show why it matters. A prompt instruction to preserve conditions such as "at the discretion of…" is a cheaper first step.
- Run each question 3 times to measure variance, starting with Q4 and Q8.

### Concepts covered

- Separating retrieval metrics from generation metrics (context recall vs faithfulness)
- Designing an eval set for coverage and avoiding tuning to the test set
- Treating API errors separately from model failures
- Validating the evaluator itself: checking expected phrases, reading raw answers (which caught a false pass on Q1), and re-scoring saved runs after tightening a check

---

## 7. Structured logging - `rag_common.log_llm_call`

**Problem:** LangSmith's free tier caps traces and expires them after 14 days, so there was no durable record of each call's prompt, response, latency or token cost.
**Approach:** A single `log_llm_call()` in `rag_common.py` appends one JSON line per real LLM call to `logs/llm_calls.jsonl`: timestamp, source (`api_output`, `api_ws` or `eval`), model, question, prompt, response, latency, time to first token, and prompt/output/total/thinking tokens. For streamed calls, usage metadata is taken from the final chunk. It never raises, so logging cannot fail a request.
**Outcome:** Every `/output` miss, every `/ws` message and every eval generation is logged. Cache hits make no LLM call, so they are not. The log gave a measured cost comparison of k=3 vs k=8 (median 2.07s vs 2.36s, mean prompt tokens 735 vs 1,985), and the eval summary now reports thinking and total tokens too. `logs/` is git-ignored apart from a trimmed `sample_llm_calls.jsonl`.

### Technical decisions

**One shared helper, not per-file logging** - the API and the eval use the same function, so the numbers they report are comparable.
**Logging never raises** - the call is wrapped in try/except and prints a warning on failure.
**JSONL, not a database** - append-only, greppable, and easy to analyse with jq or pandas.

### What I'd improve

- Log cache hits too (with a `cache_hit` flag) so hit rate can be measured.
- Log rotation and a size cap; the file grows forever.

---

## Prerequisites

- A Google Gemini API key - get one at aistudio.google.com
- A Redis server running on localhost:6379 (the API's cache): `brew install redis && brew services start redis`, or `docker run -p 6379:6379 redis`
- Create a `.env` file in the repo root:
```
GOOGLE_API_KEY=your-key-here
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=tennis-rules-bot
# Only if your LangSmith account is on the EU region:
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
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

Eval harness (see section 6):
```
python -m rag_chatbot.eval_rag --retrieval-only          # no generation calls
python -m rag_chatbot.eval_rag --retrieval-only --k 8
python -m rag_chatbot.eval_rag --runs 1 --label baseline_k3
python -m rag_chatbot.eval_rag --runs 1 --k 8 --label k8
python -m rag_chatbot.eval_rag --runs 1 --k 3 --no-cache --label k3_fresh
python -m rag_chatbot.eval_rag --runs 1 --k 8 --no-cache --label k8_fresh
python -m rag_chatbot.eval_rag --rescore rag_chatbot/eval_results/<report>.json   # no API calls
python -m rag_chatbot.eval_rag --compare <before.json> <after.json>
```

Held-out validation (`--eval-set` works with every mode, including `--rescore`):
```
python -m rag_chatbot.eval_rag --eval-set rag_chatbot/eval_set_heldout.json --ranks
python -m rag_chatbot.eval_rag --eval-set rag_chatbot/eval_set_heldout.json --retrieval-only --k 8
python -m rag_chatbot.eval_rag --eval-set rag_chatbot/eval_set_heldout.json --runs 1 --k 3 --no-cache --label heldout_k3
python -m rag_chatbot.eval_rag --eval-set rag_chatbot/eval_set_heldout.json --runs 1 --k 8 --no-cache --label heldout_k8
python -m rag_chatbot.eval_rag --eval-set rag_chatbot/eval_set_heldout.json --rescore rag_chatbot/eval_results/<report>.json
```

## Files

- `rag_chatbot.py` - CLI pipeline and chat loop, stateful across a terminal session
- `rag_api.py` - FastAPI service exposing the same retrieval pipeline over HTTP, stateless per request, with cache-aside caching on `/output`
- `documents/2026-rules-of-tennis-english.pdf` - the ITF Rules of Tennis
- `chroma_db/` - generated on first run, not committed to git
- `static/websocket_client.html` - minimal HTML/JS test client for the `/ws` endpoint
- `eval_rag.py` - eval harness: scores retrieval and generation separately, with an answer cache, call budget and `--rescore`
- `eval_set.json` - the 15 dev eval questions with expected phrases and keyword checks
- `eval_set_heldout.json` - 15 held-out questions (13 answerable, 2 out-of-scope), used with `--eval-set` to validate the k choice
- `eval_results/` - saved eval reports (the answer cache in here is git-ignored); re-score any of them with `--rescore`
- `rag_common.py` - code shared by the API and the eval: model, system instruction, retrieval `K`, phrase-matching helper, LLM call logging
- `logs/` - structured logging of every call (token count, response and latency); only `sample_llm_calls.jsonl` is committed