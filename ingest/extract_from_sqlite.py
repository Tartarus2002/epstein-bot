"""
Extract text from the rhowardstone full_text_corpus.db SQLite database.
Groups pages by EFTA number and concatenates text into per-document records.

Input:  data/full_text_corpus.db  (SQLite with 'pages' table)
Output: data/cleaned.json         (same format chunk_dataset.py expects)

Schema of pages table:
  - efta_number  (e.g. "EFTA02730265")
  - page_number  (int)
  - text_content (extracted text)
"""

import json
import os
import re
import sqlite3

DATA_DIR = "data"
DB_FILE = os.path.join(DATA_DIR, "full_text_corpus.db")
OUTPUT = os.path.join(DATA_DIR, "cleaned.json")

MIN_TEXT_LENGTH = 100  # Skip docs with barely any text (mostly blank/image-only pages)


def clean_text(text: str) -> str:
    """Collapse excessive whitespace."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def main():
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"SQLite corpus not found at {DB_FILE}. Run download_corpus.py first.")

    print(f"Opening {DB_FILE}...")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Check what tables/columns exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Tables found: {tables}")

    # Figure out the correct table and column names
    target_table = None
    for candidate in ["pages", "documents", "texts", "corpus"]:
        if candidate in tables:
            target_table = candidate
            break

    if not target_table:
        # Use the first non-FTS table
        for t in tables:
            if not t.endswith("_fts") and not t.startswith("sqlite_"):
                target_table = t
                break

    if not target_table:
        print(f"ERROR: Could not find a suitable table. Available: {tables}")
        conn.close()
        return

    print(f"Using table: {target_table}")

    # Get column names
    cursor.execute(f"PRAGMA table_info({target_table})")
    columns = [(row[1], row[2]) for row in cursor.fetchall()]
    col_names = [c[0] for c in columns]
    print(f"Columns: {col_names}")

    # Identify the EFTA number column and text column
    efta_col = None
    text_col = None
    page_col = None

    for name in col_names:
        lower = name.lower()
        if "efta" in lower or "bates" in lower or "document" in lower or "filename" in lower:
            efta_col = name
        if "text" in lower or "content" in lower or "body" in lower:
            text_col = name
        if "page" in lower:
            page_col = name

    if not efta_col or not text_col:
        print(f"ERROR: Could not identify EFTA column ({efta_col}) or text column ({text_col})")
        print(f"Columns available: {col_names}")
        conn.close()
        return

    print(f"EFTA column: {efta_col}")
    print(f"Text column: {text_col}")
    print(f"Page column: {page_col}")

    # Count total rows
    cursor.execute(f"SELECT COUNT(*) FROM {target_table}")
    total_rows = cursor.fetchone()[0]
    print(f"Total rows: {total_rows:,}")

    # Count unique documents
    cursor.execute(f"SELECT COUNT(DISTINCT {efta_col}) FROM {target_table}")
    total_docs = cursor.fetchone()[0]
    print(f"Unique documents: {total_docs:,}")

    # Stream through documents, grouping pages by EFTA number
    order_clause = f"ORDER BY {efta_col}, {page_col}" if page_col else f"ORDER BY {efta_col}"
    query = f"SELECT {efta_col}, {text_col} FROM {target_table} {order_clause}"

    print(f"Extracting documents...")
    docs = []
    current_efta = None
    current_pages = []
    processed = 0
    skipped_short = 0

    for row in cursor.execute(query):
        efta_num = str(row[0] or "").strip()
        text = str(row[1] or "").strip()

        if not efta_num:
            continue

        if efta_num != current_efta:
            # Flush previous document
            if current_efta and current_pages:
                full_text = clean_text("\n".join(current_pages))
                if len(full_text) >= MIN_TEXT_LENGTH:
                    # Format filename as EFTA########.pdf
                    filename = current_efta
                    if not filename.lower().endswith(".pdf"):
                        filename = f"{filename}.pdf"
                    docs.append({"file": filename, "text": full_text})
                else:
                    skipped_short += 1

            current_efta = efta_num
            current_pages = []
            processed += 1

            if processed % 50000 == 0:
                print(f"  Processed {processed:,} / {total_docs:,} documents ({len(docs):,} kept)")

        if text:
            current_pages.append(text)

    # Flush last document
    if current_efta and current_pages:
        full_text = clean_text("\n".join(current_pages))
        if len(full_text) >= MIN_TEXT_LENGTH:
            filename = current_efta
            if not filename.lower().endswith(".pdf"):
                filename = f"{filename}.pdf"
            docs.append({"file": filename, "text": full_text})
        else:
            skipped_short += 1

    conn.close()

    print(f"\nExtraction complete:")
    print(f"  Documents with text: {len(docs):,}")
    print(f"  Skipped (< {MIN_TEXT_LENGTH} chars): {skipped_short:,}")

    # Write output
    print(f"Writing {OUTPUT}...")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    output_size = os.path.getsize(OUTPUT) / (1024**3)
    print(f"Saved: {OUTPUT} ({output_size:.2f} GB)")


if __name__ == "__main__":
    main()
