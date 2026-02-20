import os
import time
import tweepy
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from vision_utils import extract_text_from_image
from security import sanitize_input, validate_output, rate_limiter

load_dotenv()

# =============================================================
# Twitter / X Authentication (API v2 + v1.1 for media lookup)
# =============================================================
bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
api_key = os.getenv("TWITTER_API_KEY")
api_secret = os.getenv("TWITTER_API_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_secret = os.getenv("TWITTER_ACCESS_SECRET")

# v2 client — for reading mentions and posting replies
client_v2 = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=api_key,
    consumer_secret=api_secret,
    access_token=access_token,
    access_token_secret=access_secret,
    wait_on_rate_limit=True,
)

# v1.1 auth — needed to resolve media URLs from tweets
auth_v1 = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
api_v1 = tweepy.API(auth_v1, wait_on_rate_limit=True)

# =============================================================
# RAG Stack — matches existing repo architecture exactly
# =============================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    collection_name="epstein",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 12, "fetch_k": 60},
)

# =============================================================
# LLM — NVIDIA NIM (replaces Groq)
# =============================================================
llm = ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0,
    max_completion_tokens=500,
)

# =============================================================
# Prompt — strict file-name-only retriever
# =============================================================
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict database retriever for the Epstein Files Transparency Act (EFTA). "
        "You are given context chunks from EFTA documents and a user query. "
        "Your ONLY job is to return the exact source file name(s) from the metadata "
        "(e.g. 'EFTA0005421.pdf') that are relevant to the query. "
        "Rules:\n"
        "1. Return ONLY the file name(s), one per line. Nothing else.\n"
        "2. Do NOT summarize, explain, or add any commentary.\n"
        "3. If no documents are relevant, respond with exactly: not found in files"
    ),
    (
        "human",
        "Context:\n{context}\n\nQuery:\n{question}"
    ),
])

# =============================================================
# Persistence — track last processed mention
# =============================================================
LAST_SEEN_FILE = "last_seen_id.txt"


def get_last_seen_id() -> int | None:
    """Read the last processed mention ID from disk."""
    if os.path.exists(LAST_SEEN_FILE):
        with open(LAST_SEEN_FILE, "r") as f:
            content = f.read().strip()
            return int(content) if content else None
    return None


def save_last_seen_id(mention_id: int) -> None:
    """Persist the latest mention ID so we never double-reply."""
    with open(LAST_SEEN_FILE, "w") as f:
        f.write(str(mention_id))


# =============================================================
# Core Logic
# =============================================================
def get_parent_tweet(parent_id: str) -> dict:
    """
    Fetch the parent tweet (the tweet being replied to).
    Returns tweet text, and media URLs if any images are attached.
    """
    response = client_v2.get_tweet(
        parent_id,
        expansions=["attachments.media_keys"],
        media_fields=["url", "preview_image_url", "type"],
        tweet_fields=["text"],
    )

    tweet_text = response.data.text if response.data else ""
    image_urls = []

    if response.includes and "media" in response.includes:
        for media in response.includes["media"]:
            if media.type == "photo" and media.url:
                image_urls.append(media.url)

    return {"text": tweet_text, "image_urls": image_urls}


def search_rag(query: str) -> str:
    """
    Run the combined query against ChromaDB + NVIDIA NIM LLM.
    Returns file name(s) or 'not found in files'.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "not found in files"

    # Build context with source metadata visible to the LLM
    context_parts = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{d.page_content}")

    context = "\n\n".join(context_parts)

    messages = prompt.format_messages(context=context, question=query)
    response = llm.invoke(messages)

    result = response.content.strip()
    return result if result else "not found in files"


def process_mention(mention, bot_user_id: str = None) -> None:
    """Handle a single mention: security checks, fetch parent, OCR, RAG, reply."""

    # ----- SECURITY: Ignore DMs and non-reply mentions -----
    # The mention must be a reply to another tweet (not a standalone tag or DM)
    if not mention.in_reply_to_user_id or not mention.referenced_tweets:
        print(f"[Skip] Mention {mention.id} is not a reply -- ignoring.")
        return

    # Ignore mentions from our own bot account (prevents self-reply loops)
    mention_author_id = str(getattr(mention, "author_id", ""))
    if bot_user_id and mention_author_id == str(bot_user_id):
        print(f"[Skip] Mention {mention.id} is from our own account -- ignoring.")
        return

    # ----- SECURITY: Per-user rate limiting -----
    if mention_author_id:
        if not rate_limiter.is_allowed(mention_author_id):
            remaining = rate_limiter.remaining(mention_author_id)
            print(f"[Rate Limit] User {mention_author_id} exceeded limit "
                  f"({remaining} remaining) -- skipping mention {mention.id}.")
            return

    # Find the parent tweet ID
    parent_id = None
    for ref in mention.referenced_tweets:
        if ref.type == "replied_to":
            parent_id = ref.id
            break

    if not parent_id:
        print(f"[Skip] Mention {mention.id} has no replied_to reference.")
        return

    print(f"[Processing] Mention {mention.id} -> Parent tweet {parent_id}")

    # 1. Fetch parent tweet (text + images)
    parent = get_parent_tweet(parent_id)
    caption_text = parent["text"]
    image_urls = parent["image_urls"]

    print(f"  Caption: {caption_text[:100]}...")
    print(f"  Images found: {len(image_urls)}")

    # ----- SECURITY: Sanitize caption text -----
    caption_text, caption_flags = sanitize_input(caption_text)
    if caption_flags:
        print(f"  [Security] Caption sanitized -- flags: {caption_flags}")

    # 2. OCR any images via NVIDIA Vision
    image_text = ""
    for url in image_urls:
        print(f"  [Vision] Extracting text from: {url}")
        extracted = extract_text_from_image(url)
        if extracted and extracted.lower() != "no text":
            # ----- SECURITY: Sanitize OCR text too (images can contain injections) -----
            sanitized_ocr, ocr_flags = sanitize_input(extracted)
            if ocr_flags:
                print(f"  [Security] OCR text sanitized -- flags: {ocr_flags}")
            image_text += " " + sanitized_ocr

    # 3. Combine into search query
    search_query = f"{caption_text} {image_text}".strip()

    # ----- SECURITY: Final length cap on combined query -----
    if len(search_query) > 1500:
        search_query = search_query[:1500]
        print(f"  [Security] Combined query truncated to 1500 chars.")

    print(f"  [Query] {search_query[:150]}...")

    if not search_query:
        reply_text = "Not found in files."
    else:
        # 4. RAG search
        rag_result = search_rag(search_query)
        print(f"  [RAG Result (raw)] {rag_result[:200]}")

        # ----- SECURITY: Validate LLM output before posting -----
        rag_result = validate_output(rag_result)
        print(f"  [RAG Result (validated)] {rag_result}")

        # 5. Format reply
        if rag_result.lower() == "not found in files":
            reply_text = "Not found in files."
        else:
            # Build reply filename-by-filename to avoid cutting mid-name
            TWITTER_LIMIT = 280
            PREFIX = "Related documents found in the Epstein files:\n"
            filenames = [f.strip() for f in rag_result.splitlines() if f.strip()]
            included = []
            omitted = 0

            for fname in filenames:
                candidate = PREFIX + "\n".join(included + [fname])
                if len(candidate) <= TWITTER_LIMIT:
                    included.append(fname)
                else:
                    omitted += 1

            if not included:
                # Extremely unlikely -- single filename exceeds limit
                reply_text = PREFIX + filenames[0][:TWITTER_LIMIT - len(PREFIX)]
            else:
                reply_text = PREFIX + "\n".join(included)
                if omitted:
                    suffix = f"\n(+{omitted} more)"
                    if len(reply_text) + len(suffix) <= TWITTER_LIMIT:
                        reply_text += suffix

    print(f"  [Reply] {reply_text}")

    try:
        client_v2.create_tweet(text=reply_text, in_reply_to_tweet_id=mention.id)
        print(f"  [OK] Replied to mention {mention.id}")
    except Exception as e:
        print(f"  [Error] Failed to reply: {e}")


# =============================================================
# Main Polling Loop
# =============================================================
def run_bot(poll_interval: int = 60) -> None:
    """Poll for new mentions and process them."""

    # Get our own user ID
    me = client_v2.get_me()
    if not me or not me.data:
        print("[Fatal] Could not authenticate. Check your Twitter API keys.")
        return

    bot_user_id = me.data.id
    bot_username = me.data.username
    print(f"[Bot] Authenticated as @{bot_username} (ID: {bot_user_id})")
    print(f"[Bot] Polling every {poll_interval}s for mentions...")

    last_seen_id = get_last_seen_id()
    if last_seen_id:
        print(f"[Bot] Resuming from last seen mention ID: {last_seen_id}")

    print("[Bot] Running. Press Ctrl+C to stop cleanly.")

    try:
        while True:
            try:
                # Fetch mentions since last seen
                # Note: author_id is requested for rate limiting and self-reply prevention
                mentions = client_v2.get_users_mentions(
                    id=bot_user_id,
                    since_id=last_seen_id,
                    expansions=["referenced_tweets.id", "author_id"],
                    tweet_fields=["in_reply_to_user_id", "referenced_tweets", "text", "author_id"],
                    max_results=10,
                )

                if mentions.data:
                    # Process oldest first so we save progress correctly
                    for mention in reversed(mentions.data):
                        print(f"\n{'='*50}")
                        process_mention(mention, bot_user_id=bot_user_id)
                        last_seen_id = mention.id
                        save_last_seen_id(last_seen_id)
                else:
                    print(".", end="", flush=True)

            except tweepy.TooManyRequests:
                print("\n[Rate Limit] Waiting 60s...")
                time.sleep(60)
            except Exception as e:
                print(f"\n[Error] {e}")
                time.sleep(10)

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[Bot] Shutdown requested. Last processed mention ID saved. Goodbye.")


# =============================================================
# Entry Point
# =============================================================
if __name__ == "__main__":
    print("Loading RAG components...")
    print(f"ChromaDB collection: epstein | Dir: ./chroma_db")
    print(f"LLM: meta/llama-3.1-70b-instruct via NVIDIA NIM")
    print(f"Vision: meta/llama-3.2-11b-vision-instruct via NVIDIA NIM")
    print(f"Security: anti-injection sanitization | output validation | rate limiting (5/hr/user)")
    print("-" * 50)
    # 180s = ~14,400 reads/month (fits within X API Basic tier 10K limit with margin)
    # Lower to 60 if you upgrade your X API plan
    poll_interval = int(os.getenv("POLL_INTERVAL", "180"))
    run_bot(poll_interval=poll_interval)
