"""
Auto-restart wrapper for embed_chunks.py.
If the embedding crashes (ChromaDB HNSW segfault bug), this script
automatically restarts it. The embed script has resume support, so
it picks up from the last completed batch each time.
"""

import subprocess
import sys
import time
import os

SCRIPT = os.path.join(os.path.dirname(__file__), "embed_chunks.py")
MAX_RESTARTS = 200  # more than enough for 2.47M chunks in 5K batches
RESTART_DELAY = 5   # seconds between restarts

print(f"[Runner] Auto-restart wrapper for {SCRIPT}", flush=True)
print(f"[Runner] Max restarts: {MAX_RESTARTS}", flush=True)
print(f"[Runner] Will auto-resume on crash", flush=True)

for attempt in range(1, MAX_RESTARTS + 1):
    print(f"\n[Runner] === Attempt {attempt} ===", flush=True)

    result = subprocess.run(
        [sys.executable, SCRIPT],
        cwd=os.path.dirname(os.path.dirname(__file__)),  # EpsteinFiles-RAG/
    )

    if result.returncode == 0:
        print(f"\n[Runner] Embedding completed successfully!", flush=True)
        break
    else:
        print(f"\n[Runner] Process exited with code {result.returncode}", flush=True)

        # Check if we're actually done by reading the log
        log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embed_log.txt")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                last_lines = f.readlines()[-3:]
            for line in last_lines:
                print(f"[Runner] Last log: {line.strip()}", flush=True)
                if "COMPLETE" in line:
                    print(f"[Runner] Embedding is complete despite exit code!", flush=True)
                    sys.exit(0)

        print(f"[Runner] Restarting in {RESTART_DELAY}s...", flush=True)
        time.sleep(RESTART_DELAY)

else:
    print(f"\n[Runner] FAILED after {MAX_RESTARTS} attempts!", flush=True)
    sys.exit(1)
