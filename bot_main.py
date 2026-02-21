"""
Orchestrator: runs Twitter polling + Instagram webhook server in one process.

The FAISS index (~10 GB) is loaded once when x_bot is imported as a module,
then shared by both the Twitter polling loop and Instagram webhook handlers
via Python module globals.  No duplicate loading.

Thread layout:
  - Main thread: Twitter polling loop (x_bot.run_bot) — handles KeyboardInterrupt
  - Daemon thread: FastAPI/uvicorn webhook server (Instagram comments)
"""

import os
import sys
import threading
import logging

import uvicorn

# ---- Step 1: Import x_bot to trigger FAISS + LLM loading ----
# This executes x_bot.py's module-level code (lines 1-89):
#   - Loads FAISS index (3.6 GB) + metadata.pkl + texts.pkl into RAM
#   - Initializes HuggingFace embeddings model
#   - Initializes ChatNVIDIA LLM client
#   - Sets up tweepy Twitter client
from x_bot import run_bot as run_twitter_bot  # noqa: E402

# ---- Step 2: Import the Instagram webhook FastAPI app ----
from instagram_webhook import app as webhook_app  # noqa: E402

# Configure logging for Instagram modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def run_webhook_server() -> None:
    """Run FastAPI/uvicorn webhook server in a background daemon thread."""
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(
        webhook_app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        # Disable uvicorn's signal handlers — the main thread handles SIGTERM/SIGINT
        # (Railway sends SIGTERM on shutdown; we want the Twitter loop to catch it)
    )


def main() -> None:
    port = os.getenv("PORT", "8080")

    # Check if Instagram is configured
    ig_configured = bool(os.getenv("IG_ACCESS_TOKEN"))

    if ig_configured:
        # Start the Instagram webhook server in a daemon thread
        webhook_thread = threading.Thread(
            target=run_webhook_server,
            daemon=True,
            name="instagram-webhook",
        )
        webhook_thread.start()
        print(f"[Main] Instagram webhook server started on port {port}")
    else:
        print("[Main] Instagram not configured (IG_ACCESS_TOKEN missing) -- skipping webhook server")

    # Run Twitter polling on the main thread (blocking)
    poll_interval = int(os.getenv("POLL_INTERVAL", "180"))
    print(f"[Main] Starting Twitter polling (interval={poll_interval}s)...")
    run_twitter_bot(poll_interval=poll_interval)


if __name__ == "__main__":
    main()
