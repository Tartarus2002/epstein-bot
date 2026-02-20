# ---- Build stage: install Python deps ----
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for building wheels (chromadb, torch, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Runtime stage ----
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app.py .
COPY x_bot.py .
COPY vision_utils.py .
COPY security.py .
COPY start.py .
COPY api/ ./api/
COPY ingest/ ./ingest/

# chroma_db is NOT included in the image (1.3GB, exceeds upload limit).
# It gets built on first boot via start.py -> ingest pipeline,
# then persisted on a Railway Volume mounted at /app/chroma_db.

CMD ["python", "start.py"]
