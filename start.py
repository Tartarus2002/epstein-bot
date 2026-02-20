"""
Railway entrypoint — ChromaDB is pre-built locally and uploaded to the volume.
This script just verifies it exists and launches the bot.
"""

import os
import sys
from pathlib import Path

CHROMA_DIR = Path("chroma_db")
CHROMA_MARKER = CHROMA_DIR / "chroma.sqlite3"


def main() -> None:
    if CHROMA_MARKER.exists():
        print("[Start] ChromaDB found on volume.")
    else:
        print("[FATAL] ChromaDB NOT found on volume at /app/chroma_db")
        print("[FATAL] You must upload the pre-built ChromaDB to the Railway volume.")
        print("[FATAL] Run `python ingest/run_local.py` locally, then upload chroma_db/")
        sys.exit(1)

    # Launch the bot
    print("[Start] Launching x_bot.py...")
    os.execv(sys.executable, [sys.executable, "x_bot.py"])


if __name__ == "__main__":
    main()
