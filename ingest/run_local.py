"""
Local ingestion orchestrator — run this on your PC to build ChromaDB.

Steps:
  1. Download the rhowardstone full-text SQLite corpus (2.3GB download → 6.3GB)
  2. Extract documents with EFTA IDs from SQLite → cleaned.json
  3. Chunk documents → chunks.json
  4. Embed into ChromaDB (local directory)

After this completes, upload the chroma_db/ folder to Railway.

Usage:
  python ingest/run_local.py
"""

import subprocess
import sys
import os
import time
from pathlib import Path

SCRIPTS = [
    ("Downloading full-text corpus (2.3GB)...", "ingest/download_corpus.py"),
    ("Extracting documents with EFTA IDs...", "ingest/extract_from_sqlite.py"),
    ("Chunking documents...",                  "ingest/chunk_dataset.py"),
    ("Embedding into ChromaDB...",             "ingest/embed_chunks.py"),
]


def main():
    start = time.time()
    print("=" * 60)
    print("LOCAL INGESTION — Full EFTA Corpus (1.38M documents)")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print()

    for desc, script in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"  Script: {script}")
        print(f"{'='*60}")

        step_start = time.time()
        result = subprocess.run([sys.executable, script], cwd=os.getcwd())

        if result.returncode != 0:
            print(f"\nFATAL: {script} failed with exit code {result.returncode}")
            sys.exit(1)

        elapsed = time.time() - step_start
        print(f"  Completed in {elapsed/60:.1f} minutes")

    total = time.time() - start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total/3600:.1f} hours")
    print(f"{'='*60}")

    chroma_path = Path("chroma_db")
    if chroma_path.exists():
        size_gb = sum(f.stat().st_size for f in chroma_path.rglob("*") if f.is_file()) / (1024**3)
        print(f"\nChromaDB location: {chroma_path.absolute()}")
        print(f"ChromaDB size: {size_gb:.2f} GB")
        print(f"\nNext step: Upload chroma_db/ to Railway volume.")
    else:
        print("\nWARNING: chroma_db/ not found. Check for errors above.")


if __name__ == "__main__":
    main()
