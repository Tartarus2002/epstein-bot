# ---- Build stage: install Python deps ----
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for building wheels (faiss-cpu, sentence-transformers, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Runtime stage ----
FROM python:3.11-slim

WORKDIR /app

# zstd needed for FAISS index extraction on first deploy
RUN apt-get update && apt-get install -y --no-install-recommends zstd && rm -rf /var/lib/apt/lists/*

# Force Python to flush stdout/stderr immediately (critical for Railway log visibility)
ENV PYTHONUNBUFFERED=1

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code (no ingest/ needed — FAISS index is pre-built on volume)
COPY app.py .
COPY x_bot.py .
COPY bot_main.py .
COPY instagram_bot.py .
COPY instagram_webhook.py .
COPY vision_utils.py .
COPY security.py .
COPY start.py .
COPY api/ ./api/

# FAISS index is pre-built locally and uploaded to Railway Volume at /app/faiss_index.
# The bot reads directly from the volume — no ingestion happens at deploy time.

CMD ["python", "start.py"]
