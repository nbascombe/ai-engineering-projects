# RAG Foundation

The retrieval layer of a RAG (Retrieval Augmented Generation) system. Takes a question, finds the most semantically relevant documents from a collection, returns them and is ready to be passed to an LLM with context.

## What it does

10 tennis documents are embedded using Gemini and stored in a ChromaDB vector store. When a question is asked, it is embedded using the same model and compared against the stored documents by similarity. The closest matches are returned.

## Why this matters

This is the retrieval step that makes RAG work. Instead of relying on what the LLM was trained on, you retrieve relevant context first and pass it in. The quality of the answer depends on the quality of the retrieval.

## Concepts covered

- Text embeddings - converting meaning into vectors of numbers
- Vector similarity search - finding documents by meaning, not keywords
- ChromaDB - storing and querying embeddings locally
- Separating the embed step from the storage step for model flexibility

## How to run

From the repo root:

```
python -m rag_foundation.rag_foundation
```

## Files

- `rag_foundation.py` - embed, store, and query logic
- `documents.py` - the 10 tennis documents used as the knowledge base