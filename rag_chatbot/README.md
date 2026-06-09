# TennisRulesBot

A retrieval augmented generation (RAG) chatbot that answers questions about the
2026 ITF Rules of Tennis. Builds on the embedding and vector storage foundation
in `rag_foundation/`, adding the full pipeline of document loading, chunking,
embedding, vector storage, and LLM response generation.

## What it does

Loads the official ITF Rules of Tennis PDF, chunks and embeds it using Gemini,
stores the vectors in a persistent ChromaDB collection, and answers questions
grounded in the document. If the answer is not in the document, it says so.

## How it works

```
PDF → Load → Chunk → Embed → ChromaDB → Query → Retrieve chunks → LLM → Answer
```

On first run the pipeline builds and persists the ChromaDB collection to disk.
Every subsequent run loads the existing collection meaning no re-embedding needed.

## Example interactions

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

The baseball example demonstrates the system correctly refusing to answer
outside its knowledge base.

## Technical decisions

**Chunk size: 1200 characters** - balances retrieval precision against the
Gemini free tier embedding limit of 100 requests per minute. Smaller chunks
improve precision but increase the total number of API calls required to embed
the document.

**Rate limiting: 0.7s delay per chunk** - keeps embedding requests within the
free tier limit on first run. Has no effect on subsequent runs as the
collection is loaded from disk.

**Persistent ChromaDB** - the vector store is saved to `chroma_db/` on first
run. Subsequent runs skip the embed step entirely and load in under a second.

**Pages skipped: first 6** - the cover, app download page, and contents pages
add no retrieval value. Skipping them reduces total chunks from 184 to 86.

## Concepts covered

- RAG pipeline end to end - load, chunk, embed, store, retrieve, generate
- LangChain document loaders and text splitters
- Persistent vector storage with ChromaDB
- Retrieval grounding - LLM answers only from retrieved context
- Rate limit handling with controlled embedding throughput
- The limits of RAG - retrieval quality depends on source document terminology

## How to run

From the repo root:

```
python -m rag_chatbot.rag_chatbot
```

Type `quit` to exit.

## Files

- `rag_chatbot.py` - full pipeline and chat loop
- `documents/2026-ITF-Rules-of-Tennis.pdf` - the ITF Rules of Tennis
- `chroma_db/` - generated on first run, not committed to git