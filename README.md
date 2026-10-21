# AI Engineering Projects

A collection of projects built while developing hands-on AI engineering skills.
Each project is intentional - focused on understanding the fundamentals before 
adding complexity.

---

## Projects

### 1. Basic Chatbot (`basic_chatbot.py`)
**Problem:** What does a working LLM API call look like.
**Approach:** Interactive terminal loop that takes user input, sends it to Gemini 2.5 Flash, and prints the response. Includes basic error handling with a one-time retry on API failure.
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

### 4. TennisRulesBot CLI (`rag_chatbot/rag_chatbot.py`)
**Problem:** LLMs hallucinate confidently when asked about specific rule details they weren't trained on precisely.
**Approach:** Full RAG (Retrieval Augmented Generation) pipeline over the official 2026 ITF Rules of Tennis PDF. Document loaded, chunked at 1200 chars, embedded, stored persistently in ChromaDB, with the LLM instructed to answer only from retrieved context.
**Outcome:** Answers rules questions grounded in the document; correctly refuses out-of-scope questions ("Where can I play baseball?") rather than hallucinating. Loads in under a second on subsequent runs via persistent vector store - no re-embedding needed.
 
**Concepts covered:**
- RAG pipeline end to end - load, chunk, embed, store, retrieve, generate
- LangChain document loaders and text splitters
- Persistent vector storage with ChromaDB
- Retrieval grounding - LLM answers only from retrieved context
- Rate limit handling with controlled embedding throughput
- Failed index builds clean up after themselves: if embedding fails part way, the partial collection is deleted so the next run rebuilds instead of silently serving a half-built index
- The limits of RAG - retrieval quality depends on source document terminology

**Why this matters:**
Combines every concept from the previous projects into one working system.
The LLM is grounded in a real document and will say so when it cannot answer,
rather than hallucinating.

---

### 5. FastAPI Chatbot (`fastapi_chatbot.py`)
**Problem:** A Python script calling an LLM directly can only be used by one person 
in one place. A naive implementation, even wrapped in FastAPI, blocks a thread 
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
Added cache-aside caching on top: repeat questions skip embedding, retrieval, and 
generation entirely via a Redis lookup, cutting response time from ~5s to ~0.03s on a 
cache hit.

**Concepts covered:**
- Wrapping an existing RAG pipeline as a FastAPI service
- Sharing expensive-to-build state (`app.state`) via `lifespan` startup/shutdown
- Why blocking calls inside an `async def` route stall every other request, and how to remove them (async client, `asyncio.to_thread` for the Chroma query)
- Pydantic request validation with a custom field validator
- REST statelessness as a deliberate design tradeoff, not a limitation of AI APIs generally
- Cache-aside pattern with Redis - normalised cache keys, TTL strategy for static source data, consuming a stream into a single cacheable value

**Update: fully async request path**
**Problem:** `/output` was declared `async def`, but still called the sync embedding and streaming APIs, so every cache miss blocked the event loop for all other clients - the same stall documented for the WebSocket endpoint.
**Approach:** Switched `/output` to async embedding (`client.aio`), moved the Chroma query off the event loop with `asyncio.to_thread`, and consumed the stream with `async for` over `client.aio.models.generate_content_stream`. Cache-write tasks are now held in a set until they finish, since asyncio only keeps weak references to tasks, and empty responses are no longer cached.
**Outcome:** Two simultaneous uncached requests to `/output` (cache keys cleared first) finished in 1.87s and 2.74s, against a ~3s average for a single miss, so they overlap rather than queue. One manual run, no sync-version baseline for comparison. Not yet confirmed that LangSmith's `wrap_gemini` traces the `aio` streaming path.

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

### 8. Tennis Tool Agent (`tennis_tool_agent.py`)
**Problem:** RAG grounds answers in static documents, but some questions need live, current data no document can contain. "Will it rain during play in London today?" isn't answerable from a rules PDF or a fixed knowledge base. The model also has to decide *for itself* when a question needs that live data versus when it doesn't.
**Approach:** Gave Gemini a single tool via manual function calling - `types.FunctionDeclaration` + `types.Tool`, passed into `generate_content` via `config`. Used manual (not automatic) function calling deliberately, so the model's decision to call the tool, or not, could be inspected on its own, separate from execution. The tool itself geocodes a city name via Open-Meteo's free geocoding endpoint, then queries current temperature, precipitation, and wind speed for those coordinates.
**Outcome:** Confirmed the model calls the tool only when relevant - a weather-relevant question returns a `FunctionCall` with the correct city argument; a tennis-history question returns no function call and is answered directly from the model's own knowledge. Verified the full round trip: tool result fed back via `types.Part.from_function_response` produces a grounded final answer ("It will not rain in London today, as the precipitation is 0mm."), and an invalid location is handled gracefully - the model reads the tool's own error message and asks a clarifying question rather than guessing.

**System prompt & guardrails:** Found two distinct failure modes from an underspecified system prompt, not from the tool itself. With no system instruction at all, the model narrowed its own scope to just the tool's description and refused an ordinary tennis question ("who won Wimbledon in 2023?") entirely - the tool description became the model's only sense of what it was allowed to do. After adding a system instruction describing tennis knowledge broadly, the model then answered a fully unrelated question ("what's the most popular city to visit?") as if it had no scope restriction at all - describing capabilities isn't the same as restricting them. Fixed with an instruction that explicitly restricts and redirects off-topic questions, rather than only describing what the model knows. This is a soft guardrail only enforced by prompt compliance, not code. `tennis_analyst_bot.py`'s `is_tennis_related` structured-output field already does this properly, since calling code can check a boolean instead of relying on the model to comply.

**Why this matters:**
This is the first project in the repo where the model itself decides whether external code needs to run, rather than every call being explicit. That decision-making step, not the API call, is the actual skill being tested here.


**Update: second tool + multi-tool dispatch**

**Problem:** A single tool only covers questions needing one kind of external help. A real question can need two at once - "Is the match in London going to be delayed, and who's ahead in the set at 5-4?" - which means the agent needs multiple tools declared together, a way to route correctly between them, and (it turned out) more than one round trip.
**Approach:** Added a second, fully deterministic tool - `resolve_set_score` - which resolves a tennis set's status (win, win by tiebreak, in progress, or invalid) purely from two integer game counts, no network call involved. Registered both function declarations on a single `types.Tool` and dispatched execution based on `function_call.name`.
**Outcome:** Confirmed the model correctly selects the right tool for single-need questions (weather-only, scoring-only). Then, using a deliberately double-barreled question, discovered that a single round trip breaks once a question needs two sequential tool calls: the model's second response is itself another function call, not final text, and the existing one-hop code has nothing to execute it against. Confirmed via `final_response.function_calls` that this is a gap in the code's fixed round-trip shape, not a model reasoning failure - the model had already correctly worked out it needed both tools.

**Update: ReAct loop + third tool**

**Problem:** The previous fix (multi-tool dispatch) still only handled a *single* round of tool calls - the code built a `final_response` and printed it, but never fed it back in as the new `response` for a further round. A genuinely multi-hop question (needing tool A's result before deciding whether tool B is needed) had nowhere to go. Also needed a third tool that wasn't just another live API call, to prove tool-calling works for deterministic logic and local retrieval too, not only I/O-bound external services. **Approach:** Added `get_document_lookup` - a third tool that searches the existing `rag_foundation/documents.py` knowledge base for stored facts about a named player, returning a structured `player_not_found` error rather than an empty result when there's no match. Rebuilt the round-trip as a bounded ReAct loop: `for i in range(5): if not response.function_calls: break`, followed by the model handling and re-prompt. The final tool-response-collection line (`tool_response_parts.append(...)`) was moved inside the inner `for function_call in response.function_calls` loop rather than after it - the earlier version only kept the *last* tool's result when multiple tools were called in the same turn, which Gemini needs a response for every one of. **Outcome:** Tested against a deliberately three-tool question ("Is the match in London going to be delayed, who's ahead at 5-4, and is Federer the GOAT?"). Confirmed weather and scoring tools fire in parallel on the first round, the model correctly does *not* call `get_document_lookup` for the Federer question (it isn't in the stored `DOCUMENTS`, and the model answers from its own knowledge instead - proving tool-selection restraint, not just tool-selection success), and the loop cleanly exits via `break` once `function_calls` comes back empty.

### What I'd improve
- Multi-tool sequential questions now work via the ReAct loop, but there's no logging of *why* the model kept calling tools for several rounds - could log iteration count to catch a model that's just being inefficient (not stuck) at scale.
- `resolve_set_score` currently returns bare status strings ('Win', 'Tiebreak') with no player identifier - fine for a single-tool test, but the model needs to know *who* won, not just that someone did, once this feeds into a real conversation.
- `get_document_lookup` does a plain substring match on player name (`player.lower() in document['text'].lower()`) which would break on nicknames or partial names ("Federer" vs "Roger Federer" both work here by luck, but "Fed" wouldn't).
- Still no rate limiting or backoff on the Gemini calls themselves - a `range(5)` cap limits *iterations per question*, not *calls per minute* against the free-tier quota.

**Update: hardening**
**Problem:** Reviewing the loop found three weaknesses. A `for ... else` reported "hit max iterations" even when the model's final text arrived on the last round. A bad argument from the model raised inside the handler and crashed the loop, contradicting the structured-errors approach used inside the tools. And `resolve_set_score` treated negative game counts as "In Progress".
**Approach:** The max-iterations message is now decided after the loop by checking whether the last response still contains function calls. Handler calls are wrapped so any exception comes back to the model as a structured `tool_execution_failed` result. Negative inputs return `Invalid`.
**Outcome:** A final answer on round 5 is reported as a success, and a malformed tool call degrades into something the model can read and recover from instead of a traceback.

**Concepts covered:**
- Manual function calling / tool use with the Gemini SDK
- Testing tool-selection behaviour deliberately - confirming the model calls a tool when relevant, doesn't when it isn't, and can answer from its own knowledge when no tool applies
- Structured, model-readable error responses from a tool function rather than raised exceptions
- Geocoding as a two-step resolution (name → coordinates → data) before an external API call
- The difference between a system prompt describing capabilities vs. restricting scope, and why the two aren't the same instruction
- Building a fully deterministic, network-free tool (set scoring) and testing it against enumerated edge cases, versus an I/O-bound tool (weather) that can only be spot-checked, versus a local-retrieval tool (document lookup) that depends on data coverage
- Translating a written rule into boundary conditions on the inputs, rather than a lookup table of specific cases seen during testing
- The ReAct pattern implemented as a bounded loop (`for i in range(5)`), not just described conceptually
- Diagnosing a multi-hop tool-calling limitation by inspecting a response's `function_calls` rather than assuming the model chose wrong

---

### 9. LangSmith Tracing (`rag_chatbot/rag_api.py`)
**Problem:** The RAG API had several internal steps and I could only judge it by its final answers, so I couldn't tell whether a bad or slow answer came from retrieval, generation or the cache.
**Approach:** Added LangSmith tracing: a wrapped Gemini client plus `@traceable` decorators on the request, retrieval and embedding steps, producing a nested trace per request.
**Outcome:** Found a cache bug (chunks joined mid-word) and diagnosed an incomplete-answer problem as a retrieval ranking failure (the definition chunk sat outside the top 3, later measured at 6th by the eval harness's `--ranks`; see `rag_chatbot/README.md` sections 5 and 6), not a hallucination. Retrieval time was dominated by the embedding call, not the vector search. Details in `rag_chatbot/README.md`, section 5.

**Concepts covered:**
- Traces and runs, and mapping a pipeline onto them
- Instrumenting a framework-free pipeline
- Distinguishing retrieval failures from generation failures
- Context recall vs faithfulness

---

### 10. Eval Harness (`rag_chatbot/eval_rag.py`)
**Problem:** I was judging the RAG bot by eyeballing answers, so I couldn't say how often it was right or whether a failure came from retrieval or generation.
**Approach:** An eval harness over 15 hand-written questions that scores retrieval (does the right chunk reach the model?) and generation (is the answer right?) separately. It reuses the production pipeline code, treats API errors as errors not failures, and caches answers so quota-limited reruns only pay for what is missing. Saved reports can be re-scored with `--rescore` after a check is tightened.
**Outcome:** At the production setting (k=3), retrieval hit 9/13 and answers scored 11/15 on two separate runs. Every failure traced back to retrieval, caused by vocabulary the PDF doesn't use and by answers buried in example cases. One of them (Q4) was a stable confident wrong answer from an optional rule. At k=8, retrieval hit 13/13 and answers scored 15/15 on two separate runs, with both out-of-scope refusals intact. Measured cost of k=8 vs k=3: 2.4x the total tokens per query including thinking (2,539 vs 1,073; prompt alone 2.7x) for about 0.3s more median latency (2.36s vs 2.07s). Reading the raw answers exposed a false pass in my own keyword check (Q1), so I tightened it and re-scored all saved runs, which left the k=8 numbers unchanged and corrected the second k=3 run from 12/15 to 11/15. k=8 was chosen using the same 15 questions, so the API stays at k=3 until it is tested on held-out questions. Details in `rag_chatbot/README.md`, section 6.

**Concepts covered:**
- Separating retrieval metrics from generation metrics
- Designing an eval set for coverage and avoiding tuning to the test set
- Validating the evaluator itself (expected phrases, reading raw answers, re-scoring after tightening a check)
- Treating API errors separately from model failures
- Caching and call budgets for quota-limited evaluation

---

### 11. Structured LLM Call Logging (`rag_chatbot/rag_common.py`)
**Problem:** LangSmith traces are capped on the free tier and expire after 14 days, so I had no durable record of what each LLM call cost or returned, and couldn't measure latency or token cost for a configuration change like k=3 vs k=8.
**Approach:** One shared `log_llm_call()` appends a JSON line per real LLM call (timestamp, source, model, question, prompt, response, latency, time to first token, prompt/output/thinking tokens) to a local log. It is wired into `/output`, `/ws` and the eval harness, and never raises, so a logging failure can't break a request. Cache hits make no LLM call, so they are not logged.
**Outcome:** Every generation call now leaves a record, and the eval harness reads usage from the same helper. This produced a measured cost comparison of k=3 vs k=8 (median 2.07s vs 2.36s; mean total tokens 1,073 vs 2,539). It also showed that thinking tokens (299 / 487 per query) outnumber visible output tokens (39 / 67) by roughly 7x, so the visible output count alone understates generation cost by close to an order of magnitude. A trimmed sample is in `rag_chatbot/logs/sample_llm_calls.jsonl`.

**Concepts covered:**
- Structured (JSONL) logging of LLM calls
- Logging that must never break the request path
- Shared code between the API and the eval harness
- Time to first token vs total latency for streamed responses
- Thinking tokens as a hidden cost on reasoning models

---

## Technical Progression
- `basic_chatbot.py` - stateless, single call, no memory
- `tennis_analyst_bot.py` - stateful, conversational, structured JSON outputs, validated responses
- `rag_foundation/` - embeddings, vector storage, semantic search - retrieval layer of a RAG system
- `rag_chatbot/` - full RAG pipeline with persistent vector store and grounded LLM responses
- `fastapi_chatbot.py` - LLM wrapped as an HTTP service, Pydantic validation, health endpoint. Uses `client.aio.models.generate_content` → genuinely async → correctly paired with async def.
- `rag_chatbot/rag_api.py` - same RAG pipeline as an HTTP service, stateless per request, Pydantic-validated. `/output` is fully async (async embedding, Chroma query via `asyncio.to_thread`, `client.aio` streaming) with Redis cache-aside caching on a namespaced, versioned key (`rag:<version>:<fingerprint>:<question>`): ~0.05s on a hit vs ~3.2s on a miss (~64x, N=10, mean). Two simultaneous misses overlapped (1.87s and 2.74s) rather than queueing. Cache writes are fire-and-forget tasks held in a set until done, and empty responses aren't cached.
- `rag_chatbot/rag_api.py` `/ws` - same pipeline again, now stateful per connection and fully async (`client.aio`), proving why sync calls inside `async def` WebSocket handlers block every other connected client.
- `tennis_tool_agent.py` - tool-use / function-calling project, now with three tools (live weather, deterministic set scoring, local document lookup) and a bounded ReAct loop (`for i in range(5)`) that lets the model chain multiple tool calls before answering. Confirmed via a deliberately three-tool question that the model calls tools in parallel where needed and correctly withholds a tool call when it can answer from its own knowledge instead.
- `rag_chatbot/eval_rag.py` - eval harness for the RAG bot: retrieval and generation scored separately over 15 questions, answer cache and call budget for the free tier, and `--rescore` to re-apply scoring to saved reports. k=3 scored 11/15 on two runs with all failures traced to retrieval; k=8 scored 15/15 on two runs (about 2.4x the total tokens including thinking, ~0.3s more median latency). Not yet validated on held-out questions.

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
- Redis
- Open-Meteo API (weather + geocoding, no key required)
- LangSmith (tracing)

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

4. Install and start Redis (required for the API's cache): `brew install redis && brew services start redis`, or `docker run -p 6379:6379 redis`

5. Create a Google Gemini API key - get one at aistudio.google.com

6. Create a `.env` file in the root folder
```
GOOGLE_API_KEY=your-key-here
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=tennis-rules-bot
# Only if your LangSmith account is on the EU region:
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
```

7. Run a project
```
python basic_chatbot.py
python tennis_analyst_bot.py
python -m rag_foundation.rag_foundation
python -m rag_chatbot.rag_chatbot
python tennis_tool_agent.py
python -m rag_chatbot.eval_rag --retrieval-only
python -m rag_chatbot.eval_rag --runs 1 --label baseline_k3
python -m rag_chatbot.eval_rag --runs 1 --k 3 --no-cache --label k3_fresh
python -m rag_chatbot.eval_rag --runs 1 --k 8 --no-cache --label k8_fresh
python -m rag_chatbot.eval_rag --rescore rag_chatbot/eval_results/<report>.json
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