# RAG Foundation

**Problem:** Before building a full RAG (Retrieval Augmented Generation) pipeline, the retrieval step needs to work in isolation - finding the right document from a collection by meaning, not keyword matching.
**Approach:** Embedded 10 tennis documents using Gemini Embedding 001, stored them in a ChromaDB in-memory collection, and built a query function that returns the top-n most semantically similar documents for any question.
**Outcome:** Given "Who is the best tennis player on clay?", the system returns the Nadal document first - ahead of Djokovic, Swiatek, and others - purely from vector similarity. Given "What does a tiebreak mean?", it surfaces the tiebreak rules document over all player bios.
 
## How it works
 
```
Documents → Embed (Gemini) → Store (ChromaDB) → Query → Embed question → Similarity search → Top results
```
 
The embed step and the storage step are kept separate so the embedding model can be swapped without changing the retrieval logic.
 
## Sample output
 
```
Question: Who is the best tennis player on clay?
  Result 1: Rafael Nadal is regarded as the greatest clay court player of all time...
  Result 2: Iga Swiatek has dominated tennis in the early 2020s...
 
Question: What does a tiebreak mean?
  Result 1: A tiebreak is played when a set reaches 6-6. The first player to reach 7 points...
  Result 2: Wimbledon is the oldest tennis tournament in the world...
```
 
## What I'd improve
 
- Use a persistent ChromaDB client so the collection survives between runs without re-embedding
- Test retrieval with longer documents to understand how chunk quality affects result ranking
- Add cosine similarity scores to the output to surface low-confidence retrievals


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