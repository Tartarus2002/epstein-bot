import json
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Loading cleaned documents...")
with open("data/cleaned.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

print(f"Documents loaded: {len(docs):,}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80
)

def h(t):
    return hashlib.sha256(t.lower().encode()).hexdigest()

seen = set()
chunks = []

for idx, d in enumerate(docs):
    parts = splitter.split_text(d["text"])
    for i, p in enumerate(parts):
        key = h(p)
        if key in seen:
            continue
        seen.add(key)
        chunks.append({
            "text": p,
            "metadata": {
                "source": d["file"],
                "chunk": i
            }
        })

    if (idx + 1) % 100000 == 0:
        print(f"  Processed {idx + 1:,} / {len(docs):,} docs | {len(chunks):,} chunks so far")

print(f"\nChunking complete: {len(chunks):,} unique chunks from {len(docs):,} documents")
print(f"Dedup removed: {sum(len(splitter.split_text(d['text'])) for d in docs[:0]) if False else 'N/A (counted during processing)'}")

print("Writing chunks.json...")
with open("data/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f)

print(f"Chunks: {len(chunks):,}")
