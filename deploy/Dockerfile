# One-shot service to download pre-built FAISS index to Railway volume
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends zstd && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY download_faiss.py .

CMD ["python", "download_faiss.py"]
