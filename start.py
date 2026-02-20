"""
Railway entrypoint — FAISS index is pre-built locally and uploaded to the volume.
If FAISS_URL is set and no index exists, auto-downloads and extracts it (one-time).
"""

import os
import sys
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

FAISS_DIR = Path(os.getenv("FAISS_DIR", "faiss_index"))
FAISS_MARKER = FAISS_DIR / "index.faiss"
FAISS_URL = os.getenv("FAISS_URL", "")


def download_faiss() -> None:
    """Download and extract the FAISS index archive to the volume."""
    archive = Path("/tmp/faiss_index.tar.zst")

    print(f"[Download] FAISS index not found. Downloading from FAISS_URL...")
    print(f"[Download] URL: {FAISS_URL[:80]}...")
    print(f"[Download] Target: {FAISS_DIR}")

    # Clear any old data on the volume
    if FAISS_DIR.exists():
        for item in FAISS_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        print("[Download] Cleared existing volume data.")
    else:
        FAISS_DIR.mkdir(parents=True)

    # Download archive with retry
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Download] Downloading archive (attempt {attempt}/{max_retries})...")
            urllib.request.urlretrieve(FAISS_URL, str(archive))
            size_gb = archive.stat().st_size / (1024**3)
            print(f"[Download] Downloaded: {size_gb:.2f} GB")
            break
        except Exception as e:
            print(f"[Download] Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait = 30 * attempt
                print(f"[Download] Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print("[FATAL] All download attempts failed.")
                sys.exit(1)

    # Extract — tar expects faiss_index/ directory inside the archive,
    # but we want files directly in FAISS_DIR (which might be /app/chroma_db)
    print("[Download] Extracting...")
    subprocess.run(
        ["tar", "--zstd", "-xf", str(archive),
         "-C", str(FAISS_DIR), "--strip-components=1"],
        check=True,
    )

    # Clean up archive
    archive.unlink(missing_ok=True)

    # Verify
    if FAISS_MARKER.exists():
        total = sum(f.stat().st_size for f in FAISS_DIR.rglob("*") if f.is_file())
        print(f"[Download] Success! FAISS index: {total / (1024**3):.2f} GB")
    else:
        print("[FATAL] Extraction failed -- index.faiss not found after extract.")
        sys.exit(1)


def main() -> None:
    if FAISS_MARKER.exists():
        size_gb = FAISS_MARKER.stat().st_size / (1024**3)
        print(f"[Start] FAISS index found ({size_gb:.2f} GB).")
    elif FAISS_URL:
        download_faiss()
    else:
        print(f"[FATAL] FAISS index NOT found at {FAISS_DIR}/index.faiss")
        print("[FATAL] Set FAISS_URL env var to auto-download, or upload manually.")
        sys.exit(1)

    # Launch the bot
    print("[Start] Launching x_bot.py...")
    os.execv(sys.executable, [sys.executable, "x_bot.py"])


if __name__ == "__main__":
    main()
