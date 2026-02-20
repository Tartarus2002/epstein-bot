"""
Railway entrypoint — checks if ChromaDB is populated on the volume,
runs the ingestion pipeline if needed, then launches the bot.
"""

import os
import subprocess
import sys
from pathlib import Path

CHROMA_DIR = Path("chroma_db")
CHROMA_MARKER = CHROMA_DIR / "chroma.sqlite3"
DATA_DIR = Path("data")


def run_step(description: str, script: str) -> None:
    """Run an ingestion script, abort on failure."""
    print(f"\n{'='*50}")
    print(f"[Ingestion] {description}")
    print(f"{'='*50}")
    result = subprocess.run(
        [sys.executable, script],
        cwd=os.getcwd(),
    )
    if result.returncode != 0:
        print(f"[FATAL] {script} failed with exit code {result.returncode}")
        sys.exit(1)


def run_ingestion() -> None:
    """Run the full ingestion pipeline: download -> clean -> chunk -> embed."""
    print("\n" + "#" * 60)
    print("# ChromaDB not found on volume — running ingestion pipeline")
    print("# This only happens on the FIRST deploy (~20-30 min)")
    print("#" * 60)

    run_step("Downloading EFTA dataset from HuggingFace...", "ingest/download_dataset.py")
    run_step("Cleaning documents...", "ingest/clean_dataset.py")
    run_step("Chunking documents...", "ingest/chunk_dataset.py")
    run_step("Embedding chunks into ChromaDB (this takes a while)...", "ingest/embed_chunks.py")

    # Clean up raw data files to save volume space
    if DATA_DIR.exists():
        import shutil
        print("\n[Cleanup] Removing raw data files to save space...")
        shutil.rmtree(DATA_DIR)

    print("\n[Ingestion] Pipeline complete! ChromaDB is ready.")


def main() -> None:
    # Check if ChromaDB already exists on the volume
    if CHROMA_MARKER.exists():
        print("[Start] ChromaDB found on volume — skipping ingestion.")
    else:
        print("[Start] ChromaDB NOT found — first-time setup required.")
        run_ingestion()

    # Launch the bot
    print("\n[Start] Launching x_bot.py...")
    os.execv(sys.executable, [sys.executable, "x_bot.py"])


if __name__ == "__main__":
    main()
