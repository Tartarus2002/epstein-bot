"""
One-shot Railway service: downloads and extracts pre-built FAISS index to the volume.
Deploy this temporarily, then delete the service after it completes.

Set FAISS_URL env var to the direct download URL of the compressed archive.
"""

import os
import sys
import subprocess
import urllib.request
import shutil
from pathlib import Path

VOLUME_DIR = Path(os.getenv("FAISS_DIR", "/app/faiss_index"))
ARCHIVE = Path("/tmp/faiss_index.tar.zst")


def main():
    url = os.environ.get("FAISS_URL")
    if not url:
        print("[FATAL] FAISS_URL environment variable not set.")
        print("[FATAL] Set it to the direct download URL of faiss_index.tar.zst")
        sys.exit(1)

    print(f"[Download] Target volume: {VOLUME_DIR}")
    print(f"[Download] Source URL: {url[:80]}...")

    # Clear existing data on volume
    if VOLUME_DIR.exists():
        for item in VOLUME_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        print("[Download] Cleared existing volume data.")
    else:
        VOLUME_DIR.mkdir(parents=True)

    # Download
    print("[Download] Downloading archive...")
    urllib.request.urlretrieve(url, str(ARCHIVE))
    size_gb = ARCHIVE.stat().st_size / (1024**3)
    print(f"[Download] Downloaded: {size_gb:.2f} GB")

    # Extract directly to parent of volume (tar contains faiss_index/ directory)
    print("[Download] Extracting to volume...")
    subprocess.run(
        ["tar", "--zstd", "-xf", str(ARCHIVE), "-C", str(VOLUME_DIR.parent), "--strip-components=0"],
        check=True,
    )

    # Verify
    marker = VOLUME_DIR / "index.faiss"
    if marker.exists():
        total = sum(f.stat().st_size for f in VOLUME_DIR.rglob("*") if f.is_file())
        print(f"[Download] Success! FAISS index on volume: {total / (1024**3):.2f} GB")
    else:
        print("[FATAL] Extraction failed -- index.faiss not found on volume.")
        sys.exit(1)

    # Clean up
    ARCHIVE.unlink(missing_ok=True)
    print("[Download] Done. You can now delete this service and redeploy the bot.")

    # Keep alive briefly so Railway shows success in logs
    import time
    time.sleep(30)


if __name__ == "__main__":
    main()
