import os
import re
import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
col = chromadb.PersistentClient(path="rag_chatbot/chroma_db").get_collection("tennis_rules")

def norm(s):
    """Lowercase and strip ALL whitespace, so PDF extraction quirks like 'tie -break' don't break matching."""
    return re.sub(r"\s+", "", s.lower())

PHRASES = ["six games all", "6-6", "games all", "tie-break game shall be played"]
QUESTIONS = [
    "what is a tiebreak?",
    "When is a tie-break played?",
    "What happens when a set reaches 6-6?",
]

# Step 1: which chunks contain each phrase?
data = col.get(include=["documents", "metadatas"])
print(f"Total chunks in collection: {len(data['ids'])}\n")

phrase_hits = {}
for phrase in PHRASES:
    hits = [
        (cid, meta["page"], doc)
        for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        if norm(phrase) in norm(doc)
    ]
    phrase_hits[phrase] = hits
    print(f'Phrase "{phrase}": {len(hits)} chunk(s)')
    for cid, page, doc in hits:
        # show a little text around the first match
        i = norm(doc).find(norm(phrase))
        print(f"   {cid} (page {page}): {doc[:150].replace(chr(10), ' ')}...")
print()

# Step 2: where do those chunks rank for each question?
watch = {cid for p in ("six games all", "6-6") for cid, _, _ in phrase_hits[p]}
if not watch:
    print("No chunks contain 'six games all' or '6-6'. Check the phrase list above for other candidates.")

for q in QUESTIONS:
    emb = client.models.embed_content(
        model="gemini-embedding-001", contents=q
    ).embeddings[0].values
    res = col.query(query_embeddings=[emb], n_results=col.count(),
                    include=["distances"])
    ids, dists = res["ids"][0], res["distances"][0]
    print(f'Question: "{q}"')
    print("   top 3:", ids[:3])
    for cid in sorted(watch):
        rank = ids.index(cid) + 1
        print(f"   {cid}: rank {rank} of {len(ids)} (distance {dists[rank-1]:.3f})")
    print()
