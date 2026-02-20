"""
Chunk the cleaned documents for embedding.
Uses 2000-char chunks (better context per chunk, manageable DB size).
Processes docs in memory (64GB RAM handles it fine).
"""

import json
import hashlib
import time
import os
import sys
import gc

sys.stdout.reconfigure(line_buffering=True)

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
INPUT = "data/cleaned.json"
OUTPUT = "data/chunks.json"

print(f"Chunk config: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}", flush=True)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


def h(t):
    """Return truncated hash (16 bytes) to save memory."""
    return hashlib.sha256(t.lower().encode()).digest()[:16]


# Load all docs into memory (64GB RAM can handle 3.2GB JSON easily)
print("Loading documents...", flush=True)
with open(INPUT, "r", encoding="utf-8") as f:
    docs = json.load(f)
total_docs = len(docs)
print(f"Loaded {total_docs:,} documents", flush=True)

# Process in batches and write incrementally
seen = set()
chunk_count = 0
start = time.time()
WRITE_BATCH = 200000  # write to disk every N chunks to limit memory

with open(OUTPUT, "w", encoding="utf-8") as out:
    out.write("[")
    first = True
    buffer = []

    for idx in range(total_docs):
        d = docs[idx]
        text = d["text"]
        filename = d["file"]

        parts = splitter.split_text(text)
        for i, p in enumerate(parts):
            key = h(p)
            if key in seen:
                continue
            seen.add(key)

            chunk_json = json.dumps({
                "text": p,
                "metadata": {"source": filename, "chunk": i}
            })

            if not first:
                buffer.append(",\n" + chunk_json)
            else:
                buffer.append("\n" + chunk_json)
                first = False
            chunk_count += 1

        # Flush buffer to disk periodically
        if len(buffer) >= WRITE_BATCH:
            out.write("".join(buffer))
            buffer.clear()

        if (idx + 1) % 100000 == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed
            eta = (total_docs - idx - 1) / rate
            print(f"  {idx + 1:,} / {total_docs:,} docs "
                  f"| {chunk_count:,} chunks "
                  f"| {rate:.0f} docs/s "
                  f"| ETA: {eta:.0f}s", flush=True)

    # Flush remaining
    if buffer:
        out.write("".join(buffer))
    out.write("\n]")

elapsed = time.time() - start
print(f"\nChunking complete in {elapsed:.0f}s", flush=True)
print(f"  {chunk_count:,} unique chunks from {total_docs:,} documents", flush=True)

size_gb = os.path.getsize(OUTPUT) / (1024**3)
print(f"  Output: {OUTPUT} ({size_gb:.2f} GB)", flush=True)
