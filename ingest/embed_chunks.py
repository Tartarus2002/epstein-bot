"""
Embed chunks into FAISS — crash-safe with incremental saves.
Optimized for: i9-13900K + RTX 5070 Ti (16GB VRAM) + 64GB RAM

Strategy:
- Encode 100K chunks at a time on GPU, save each batch as .npy file
- On crash, restart and skip already-saved batches (resume support)
- After all encoding done, merge .npy files into a single FAISS index
- This way a crash only loses ~2 min of work max

Output: faiss_index/ directory with index.faiss + metadata.pkl + texts.pkl
"""

import json
import time
import os
import gc
import sys
import pickle
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer

# ---- Config ----
INDEX_DIR = "faiss_index"
EMB_CACHE_DIR = "data/emb_cache"  # incremental .npy files saved here
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ENCODE_BATCH = 512       # GPU sub-batch (fits 16GB VRAM with 2000-char chunks)
SAVE_EVERY = 100000      # save to disk every N chunks (~2 min of work)
LOG_FILE = "embed_log.txt"

# ---- Logging ----
log_fh = open(LOG_FILE, "a", encoding="utf-8")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()

# ---- Check GPU ----
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"GPU: {gpu_name} | {vram:.1f} GB VRAM | CUDA {torch.version.cuda}")
else:
    log("WARN: No GPU, using CPU")
log(f"CPU: {os.cpu_count()} cores")

# ---- Load chunks ----
log("Loading chunks.json...")
t0 = time.time()
with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
total_chunks = len(chunks)
log(f"Loaded {total_chunks:,} chunks in {time.time() - t0:.1f}s")

texts = [c["text"] for c in chunks]
metadatas = [c["metadata"] for c in chunks]
del chunks
gc.collect()

# ---- Check for existing progress ----
cache_dir = Path(EMB_CACHE_DIR)
cache_dir.mkdir(parents=True, exist_ok=True)

# Find which batches are already saved
existing_batches = set()
for f in cache_dir.glob("batch_*.npy"):
    try:
        batch_idx = int(f.stem.split("_")[1])
        existing_batches.add(batch_idx)
    except (ValueError, IndexError):
        pass

total_batches = (total_chunks + SAVE_EVERY - 1) // SAVE_EVERY
completed_batches = len(existing_batches)

if completed_batches > 0:
    log(f"RESUME: Found {completed_batches}/{total_batches} saved batches, skipping completed ones")
else:
    log("Starting fresh encoding")

# ---- Load embedding model on GPU ----
log(f"Loading {MODEL_NAME} on {device}...")
model = SentenceTransformer(MODEL_NAME, device=device)
model.max_seq_length = 512
dim = model.get_sentence_embedding_dimension()
log(f"Model ready | dim={dim} | {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

# ---- Encode with incremental saves ----
log(f"{'='*60}")
log(f"Encoding {total_chunks:,} texts -> {total_batches} batches of {SAVE_EVERY:,}")
log(f"{'='*60}")

start = time.time()
encoded = 0
newly_encoded = 0

for batch_idx in range(total_batches):
    batch_start = batch_idx * SAVE_EVERY
    batch_end = min(batch_start + SAVE_EVERY, total_chunks)
    batch_file = cache_dir / f"batch_{batch_idx:04d}.npy"

    # Skip if already saved from a previous run
    if batch_idx in existing_batches:
        encoded = batch_end
        continue

    batch_texts = texts[batch_start:batch_end]

    t0 = time.time()
    with torch.no_grad():
        emb = model.encode(
            batch_texts,
            batch_size=ENCODE_BATCH,
            show_progress_bar=False,
            normalize_embeddings=True,
            device=device,
        )
    t_encode = time.time() - t0

    # Save immediately to disk
    np.save(str(batch_file), emb)
    del emb

    encoded = batch_end
    newly_encoded += (batch_end - batch_start)
    elapsed = time.time() - start
    rate = newly_encoded / elapsed if elapsed > 0 else 0
    remaining = total_chunks - encoded
    eta = remaining / rate if rate > 0 else 0
    pct = encoded / total_chunks * 100

    log(f"  {encoded:,} / {total_chunks:,} ({pct:.1f}%) "
        f"| {t_encode:.1f}s | rate: {rate:,.0f}/s | ETA: {eta/60:.1f} min "
        f"| saved {batch_file.name}")

    # Clean GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

encode_time = time.time() - start
log(f"\nEncoding complete in {encode_time/60:.1f} min")

# ---- Free model and GPU memory ----
del model
if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()

# ---- Merge all .npy files into FAISS index ----
log("Loading all embedding batches from disk...")
t0 = time.time()
batch_files = sorted(cache_dir.glob("batch_*.npy"))
all_emb = [np.load(str(f)) for f in batch_files]
embeddings = np.vstack(all_emb).astype(np.float32)
del all_emb
gc.collect()
log(f"Merged {len(batch_files)} batches: {embeddings.shape} ({embeddings.nbytes/1024**3:.2f} GB) in {time.time()-t0:.1f}s")

assert embeddings.shape[0] == total_chunks, f"Mismatch: {embeddings.shape[0]} != {total_chunks}"

# ---- Build FAISS index ----
log("Building FAISS index (IndexFlatIP -- cosine similarity)...")
t0 = time.time()
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
log(f"FAISS index built in {time.time()-t0:.1f}s | {index.ntotal:,} vectors")
del embeddings
gc.collect()

# ---- Save to disk ----
index_dir = Path(INDEX_DIR)
index_dir.mkdir(parents=True, exist_ok=True)

log("Saving FAISS index...")
faiss.write_index(index, str(index_dir / "index.faiss"))
del index
gc.collect()

# Pickle for our own trusted data (standard FAISS/LangChain pattern)
log("Saving metadata + texts...")
with open(index_dir / "metadata.pkl", "wb") as f:
    pickle.dump(metadatas, f)
with open(index_dir / "texts.pkl", "wb") as f:
    pickle.dump(texts, f)

# ---- Report ----
total_time = time.time() - start
total_size = sum(f.stat().st_size for f in index_dir.rglob("*") if f.is_file())

log(f"\n{'='*60}")
log(f"COMPLETE in {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
log(f"FAISS index: {total_chunks:,} vectors, dim={dim}")
for f in sorted(index_dir.rglob("*")):
    if f.is_file():
        log(f"  {f.name}: {f.stat().st_size/1024**2:.1f} MB")
log(f"Total index size: {total_size/1024**3:.2f} GB")
log(f"{'='*60}")

log_fh.close()
