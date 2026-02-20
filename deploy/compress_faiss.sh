#!/bin/bash
# Compress the local FAISS index for upload to Railway
# Run from EpsteinFiles-RAG directory after embedding completes

set -e

FAISS_DIR="faiss_index"
ARCHIVE="faiss_index.tar.zst"

if [ ! -f "$FAISS_DIR/index.faiss" ]; then
    echo "ERROR: $FAISS_DIR/index.faiss not found. Embedding may still be running."
    exit 1
fi

echo "Contents of $FAISS_DIR:"
ls -lh "$FAISS_DIR/"

echo ""
echo "Compressing $FAISS_DIR..."

# Use zstd compression (fast + good ratio)
tar -cf - "$FAISS_DIR" | zstd -3 -T0 -o "$ARCHIVE"

SIZE=$(du -h "$ARCHIVE" | cut -f1)
echo "Done! Archive: $ARCHIVE ($SIZE)"
echo ""
echo "Next steps:"
echo "  1. Upload $ARCHIVE to a cloud host (Google Drive, Backblaze, etc.)"
echo "  2. Get a direct download URL"
echo "  3. Deploy the Railway downloader: cd deploy && railway up"
