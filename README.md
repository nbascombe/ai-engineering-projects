# AI Engineering Projects

A collection of projects built while developing hands-on AI engineering skills.
Each project is intentional - focused on understanding the fundamentals before 
adding complexity.

---

## Projects

### 1. Basic Chatbot (`basic_chatbot.py`)
**Problem:** What does a working LLM API call look like.
**Approach:** Single stateless call to Gemini 2.5 Flash with no session, no prompt engineering, no parsing.
**Outcome:** A working baseline that confirms the API integration, key management, and response structure - the foundation every subsequent project builds on.

**Concepts covered:**
- Gemini API integration
- Secure API key management with environment variables
- Stateless LLM calls

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

### 4. TennisRulesBot (`rag_chatbot/`)
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
in one place. Wrapping it as an API makes it usable by any client - a frontend, 
another service, or a mobile app.
**Approach:** Wrapped the basic chatbot in a FastAPI POST endpoint with Pydantic 
request validation. Plain function (not async) so FastAPI runs it in a thread pool 
and the blocking Gemini call doesn't freeze the event loop.
**Outcome:** A running HTTP service that accepts a question and returns an LLM 
response, testable via FastAPI's built-in Swagger UI at /docs.

**Concepts covered:**
- FastAPI endpoint structure and route decoration
- Pydantic models for request validation
- Health check endpoints
- Testing APIs via Swagger UI

---

## Technical Progression
- `basic_chatbot.py` - stateless, single call, no memory
- `tennis_analyst_bot.py` - stateful, conversational, structured JSON outputs, validated responses
- `rag_foundation/` - embeddings, vector storage, semantic search - retrieval layer of a RAG system
- `rag_chatbot/` - full RAG pipeline with persistent vector store and grounded LLM responses
- `fastapi_chatbot.py` - LLM wrapped as an HTTP service, Pydantic validation, health endpoint

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
```
Test via the built-in docs UI at http://localhost:8000/docs

---

*Nikita - AI Engineering Projects - 2026*