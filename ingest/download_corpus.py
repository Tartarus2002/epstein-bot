"""
Download the rhowardstone/Epstein-research-data full-text SQLite corpus.
All 12 DOJ datasets + House Oversight + FBI Vault = 1.38M documents.

The DB is split into two gzip parts on GitHub Releases (v4.0).
This script downloads, concatenates, and decompresses them.
"""

import os
import gzip
import shutil
import urllib.request
import sys

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://github.com/rhowardstone/Epstein-research-data/releases/download/v4.0"
PARTS = [
    f"{BASE_URL}/full_text_corpus.db.gz.part_aa",
    f"{BASE_URL}/full_text_corpus.db.gz.part_ab",
]
GZ_FILE = os.path.join(DATA_DIR, "full_text_corpus.db.gz")
DB_FILE = os.path.join(DATA_DIR, "full_text_corpus.db")


def download_with_progress(url: str, dest: str) -> None:
    """Download a file with progress bar."""
    filename = url.split("/")[-1]
    print(f"Downloading {filename}...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req)
    total = int(resp.headers.get("Content-Length", 0))

    downloaded = 0
    block_size = 1024 * 1024  # 1 MB
    with open(dest, "ab") as f:
        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)", end="", flush=True)
    print()


def main():
    # Skip if DB already exists
    if os.path.exists(DB_FILE):
        size_gb = os.path.getsize(DB_FILE) / (1024**3)
        print(f"SQLite corpus already exists: {DB_FILE} ({size_gb:.1f} GB) — skipping download.")
        return

    # Step 1: Download both parts and concatenate into a single .gz file
    if not os.path.exists(GZ_FILE):
        # Remove partial download if present
        if os.path.exists(GZ_FILE + ".partial"):
            os.remove(GZ_FILE + ".partial")

        for url in PARTS:
            download_with_progress(url, GZ_FILE + ".partial")

        os.rename(GZ_FILE + ".partial", GZ_FILE)
        gz_size = os.path.getsize(GZ_FILE) / (1024**3)
        print(f"Downloaded and concatenated: {gz_size:.2f} GB")
    else:
        print(f"Compressed file already exists: {GZ_FILE}")

    # Step 2: Decompress
    print("Decompressing (this takes a few minutes)...")
    with gzip.open(GZ_FILE, "rb") as f_in, open(DB_FILE, "wb") as f_out:
        while True:
            chunk = f_in.read(1024 * 1024 * 10)  # 10 MB blocks
            if not chunk:
                break
            f_out.write(chunk)

    db_size = os.path.getsize(DB_FILE) / (1024**3)
    print(f"Decompressed: {DB_FILE} ({db_size:.1f} GB)")

    # Clean up compressed file to save disk space
    os.remove(GZ_FILE)
    print("Removed compressed file to save space.")


if __name__ == "__main__":
    main()
