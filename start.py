"""
Railway entrypoint — FAISS index is pre-built locally and uploaded to the volume.
This script just verifies it exists and launches the bot.
"""

import os
import sys
from pathlib import Path

FAISS_DIR = Path(os.getenv("FAISS_DIR", "faiss_index"))
FAISS_MARKER = FAISS_DIR / "index.faiss"


def main() -> None:
    if FAISS_MARKER.exists():
        size_gb = FAISS_MARKER.stat().st_size / (1024**3)
        print(f"[Start] FAISS index found ({size_gb:.2f} GB).")
    else:
        print(f"[FATAL] FAISS index NOT found at {FAISS_DIR}/index.faiss")
        print("[FATAL] You must upload the pre-built FAISS index to the Railway volume.")
        print("[FATAL] Run `python ingest/embed_runner.py` locally, then upload faiss_index/")
        sys.exit(1)

    # Launch the bot
    print("[Start] Launching x_bot.py...")
    os.execv(sys.executable, [sys.executable, "x_bot.py"])


if __name__ == "__main__":
    main()
