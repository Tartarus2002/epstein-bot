# epstein-bot

A bot that searches the public Epstein files (a full-text corpus of 1.38 million documents) and replies to a claim with the names of the source files to read.

Built on [EpsteinFiles-RAG](https://github.com/AnkitNayak-dev/EpsteinFiles-RAG) by Ankit Nayak (MIT license). That project is a RAG pipeline over the [Epstein Files 20K](https://huggingface.co/datasets/teyler/epstein-files-20k) dataset on Hugging Face (House Oversight estate documents), with ChromaDB, a FastAPI endpoint and a Streamlit UI. The first seven commits in this repo (Feb 10-14, 2026) are the original author's. The ten commits after them are mine.

## What I added

- **Full corpus:** replaced the 20K dataset with a public full-text corpus of 1.38M documents, covering all 12 DOJ datasets plus House Oversight and FBI Vault material. DOJ documents keep their EFTA numbers as file names (`EFTA########.pdf`), and the bot replies with those names.
- **FAISS instead of ChromaDB:** about 2.47M chunk vectors in one flat FAISS index. ChromaDB's HNSW index kept getting corrupted when the embedding job crashed and restarted.
- **Resumable embedding:** the GPU job saves its work every 100K chunks and picks up from the last saved batch. A small wrapper restarts it after a crash.
- **Self-provisioning index:** on first boot, `start.py` downloads the prebuilt index from `FAISS_URL` (up to 5 attempts, with a longer wait after each failure) and extracts it.
- **Models:** moved the LLM from Groq to NVIDIA NIM, then upgraded to `deepseek-ai/deepseek-v3.2` for choosing the relevant files and `nvidia/nemotron-nano-12b-v2-vl` for OCR on images in posts.
- **X bot:** when someone tags the bot in a reply under a post, it reads that post, including text in attached images, and replies with the names of the most relevant files.
- **Instagram comments (experimental):** the same pipeline can also answer Instagram comments that tag the bot's account, through a Meta webhook. It only starts when `IG_ACCESS_TOKEN` is set.
- **Guardrails:** redacts common prompt-injection phrases in post text and OCR output, posts only lines that look like file names, and limits each user to 5 requests an hour.
- **Railway deployment:** Dockerfile and `railway.toml`, the index on a mounted volume, a fix for a crash when clearing that volume (`rmtree` on a mount point fails with `EBUSY`), and unbuffered output so logs show up.

## How it works

### Building the index (once, on a local machine)

1. `ingest/download_corpus.py` downloads the full-text SQLite corpus and decompresses it.
2. `ingest/extract_from_sqlite.py` joins the pages of each document (grouped by EFTA number) into one record and skips documents with under 100 characters of text. About 1.3M remain.
3. `ingest/chunk_dataset.py` splits each document into chunks of up to 2,000 characters with 200 characters of overlap and drops duplicate chunks.
4. `ingest/embed_chunks.py` embeds the chunks with `sentence-transformers/all-MiniLM-L6-v2` and writes a FAISS `IndexFlatIP` over normalized vectors (cosine similarity) to `faiss_index/`, with `metadata.pkl` and `texts.pkl` beside it.
5. `deploy/compress_faiss.sh` packs `faiss_index/` into `faiss_index.tar.zst` so it can be hosted for `FAISS_URL`.

### Answering a post

```text
X: mention in a reply under a post        Instagram: comment that tags the bot
               |                                        |
     parent post text + photos              comment + post caption + photos
               +-------------------+--------------------+
                                   |
              OCR on the post's images (Nemotron Nano VL)
                                   |
         redact injection phrases, cap the query at 1,500 chars
                                   |
       embed, FAISS top 60, best chunk per file, keep 12 files
                                   |
              DeepSeek V3.2 picks the relevant file names
                                   |
      drop lines that do not look like file names, post the reply
```

On X the bot checks mentions every 180 seconds by default and skips mentions that are not replies. A reply starts with "Related documents found in the Epstein files:" and lists as many file names as fit in 280 characters (2,200 on Instagram). With no match, the X reply is "Not found in files."

Code map: `start.py` (entry point and index download), `x_bot.py` (X and retrieval), `instagram_bot.py` and `instagram_webhook.py` (Instagram), `vision_utils.py` (OCR), `security.py` (guardrails), `bot_main.py` (runs both bots in one process).

## Status

The hosted bot ran on X and is offline now. The code and the steps to run your own copy are below.

## Run it yourself

### Requirements

- Python 3.11 (the Docker image uses `python:3.11-slim`), plus `tar` and `zstd`.
- An NVIDIA NIM API key and X API keys for the bot account. Instagram also needs a Meta app with webhooks.
- The FAISS index, about 6.8 GB, which the bot loads into memory at startup. Build it with the steps below. If you host the archive, set `FAISS_URL` and `start.py` downloads it on first boot.

### Environment variables

Copy `.env.example` to `.env` and fill in your own values. Add the other names as you need them.

| Variable | Used for |
| --- | --- |
| `NVIDIA_API_KEY` | LLM and OCR calls to NVIDIA NIM |
| `TWITTER_BEARER_TOKEN`, `TWITTER_API_KEY`, `TWITTER_API_SECRET`, `TWITTER_ACCESS_TOKEN`, `TWITTER_ACCESS_SECRET` | Reading mentions and posting replies on X |
| `FAISS_DIR` | Index folder (default `faiss_index`) |
| `FAISS_URL` | Direct link to `faiss_index.tar.zst` for the first-boot download |
| `POLL_INTERVAL` | Seconds between mention checks (default 180) |
| `IG_ACCESS_TOKEN`, `IG_ACCOUNT_ID`, `IG_USERNAME` | Instagram replies (off unless `IG_ACCESS_TOKEN` is set) |
| `IG_WEBHOOK_VERIFY_TOKEN`, `IG_APP_SECRET` | Meta webhook verification and payload signatures (signatures are checked only when `IG_APP_SECRET` is set) |
| `PORT` | Webhook server port (default 8080) |

The first two rows are in `.env.example`. The rest are read in `start.py`, `x_bot.py`, `bot_main.py` and the Instagram modules.

### Start the bot

```bash
pip install -r requirements.txt
python start.py
```

`start.py` looks for `faiss_index/index.faiss`, downloads the archive from `FAISS_URL` if it is missing, then runs `bot_main.py`. For the download, `FAISS_URL` must be a real environment variable, because `start.py` does not read `.env`. The archive goes to `/tmp` and is unpacked with `tar --zstd`. Docker covers all of that:

```bash
docker build -t epstein-bot .
docker run --env-file .env -v "$PWD/faiss_index:/app/faiss_index" epstein-bot
```

Add `-p 8080:8080` for the Instagram webhook. On Railway, `railway.toml` builds the same Dockerfile and runs `python start.py`, restarting it up to 5 times on failure. Mount a volume at `/app/faiss_index` and set `FAISS_URL` so the first boot fills it.

### Build the index

```bash
pip install -r requirements.txt langchain-text-splitters
python ingest/download_corpus.py
python ingest/extract_from_sqlite.py
python ingest/chunk_dataset.py
python ingest/embed_runner.py    # runs embed_chunks.py and restarts it after a crash
bash deploy/compress_faiss.sh    # optional: pack the index for FAISS_URL
```

- The v4.0 download links in `ingest/download_corpus.py` return 404. As of November 2026 the corpus project publishes only a v5.2 release, with different file names, so that script needs updating before it will run. The counts in this README (1.38M documents, about 2.47M vectors) come from the v4.0 build.
- `langchain-text-splitters` is not in `requirements.txt` because the Docker image does not run ingestion.
- `chunk_dataset.py` and `embed_chunks.py` hold the whole corpus in memory. They were written for a machine with 64 GB of RAM.
- Embedding uses a CUDA GPU if PyTorch finds one and falls back to the CPU.
- `api/main.py` (Q&A API) and `app.py` (Streamlit UI) come from the original project and are not used by the bot. The API still reads a ChromaDB store through `langchain-chroma`, which is no longer in `requirements.txt`.

## Responsible use

The bot searches public records and replies with document names, so people can look the documents up and read them. It does not decide what is true, and it does not summarize or quote documents. Any line from the model that does not look like a file name is dropped before posting. The documents themselves can contain unverified claims. A listed file is one the model judged related to the post. It does not mean the post is true.

## Credits

- [EpsteinFiles-RAG](https://github.com/AnkitNayak-dev/EpsteinFiles-RAG) by Ankit Nayak (MIT): the original RAG pipeline, API and UI.
- [Epstein-research-data](https://github.com/rhowardstone/Epstein-research-data) by Rye Howard-Stone: the full-text corpus (`full_text_corpus.db`), licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The ingest scripts download it. It is not stored in this repo.
- Source documents: the U.S. Department of Justice release under the Epstein Files Transparency Act (Public Law 119-38), at [justice.gov/epstein](https://www.justice.gov/epstein), plus the House Oversight and FBI Vault material in the corpus.
- Models: `sentence-transformers/all-MiniLM-L6-v2` for embeddings, plus `deepseek-ai/deepseek-v3.2` and `nvidia/nemotron-nano-12b-v2-vl` through NVIDIA NIM.

## License

MIT, same as the original project. [LICENSE](LICENSE) keeps the original copyright line. The MIT license covers the code in this repo. The corpus has its own license (CC BY-NC-SA 4.0). A prebuilt index includes corpus text (`texts.pkl`), so share one only under that license.
