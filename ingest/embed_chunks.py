import json
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_db"

# --------------------------------------------------
# Reset Chroma DB (clear contents, keep directory for volume mounts)
# --------------------------------------------------
chroma_path = Path(CHROMA_DIR)
if chroma_path.exists():
    print("Clearing existing Chroma DB contents...")
    for item in chroma_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    print("Chroma DB directory cleared.")
else:
    chroma_path.mkdir(parents=True, exist_ok=True)
    print("Created fresh chroma_db directory.")

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
