import json
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_db"

# --------------------------------------------------
# Reset Chroma DB
# --------------------------------------------------
if Path(CHROMA_DIR).exists():
    print("Removing existing Chroma DB...")
    shutil.rmtree(CHROMA_DIR)

# --------------------------------------------------
# Load chunks
# --------------------------------------------------
print("Loading chunks...")

with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

total_chunks = len(chunks)
print(f"Loaded {total_chunks} chunks")

# --------------------------------------------------
# Load embedding model
# --------------------------------------------------
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model ready")

# --------------------------------------------------
# Create Chroma DB
# --------------------------------------------------
print("Initializing Chroma vector store...")

db = Chroma(
    collection_name="epstein",
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

# --------------------------------------------------
# Embed in batches
# --------------------------------------------------
BATCH = 1000
print(f"Starting embedding in batches of {BATCH}")

for i in range(0, total_chunks, BATCH):
    end = min(i + BATCH, total_chunks)
    print(f"Embedding chunks {i + 1} -> {end} / {total_chunks}")

    db.add_texts(
        texts=[c["text"] for c in chunks[i:end]],
        metadatas=[c["metadata"] for c in chunks[i:end]]
    )

print("Chroma embedding complete")
print("Vector DB saved at:", CHROMA_DIR)
