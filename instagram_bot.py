"""
Instagram comment handler for the EFTA Epstein Bot.

Receives webhook events from instagram_webhook.py and processes them
using the same FAISS index and RAG pipeline as the Twitter bot.
"""

import os
import time
import threading
import logging
import requests

from security import sanitize_input, validate_output, rate_limiter
from vision_utils import extract_text_from_image

logger = logging.getLogger("instagram_bot")

# Instagram Graph API config
GRAPH_API_VERSION = "v21.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"
IG_ACCESS_TOKEN = os.getenv("IG_ACCESS_TOKEN", "")
IG_ACCOUNT_ID = os.getenv("IG_ACCOUNT_ID", "")
IG_USERNAME = os.getenv("IG_USERNAME", "").lower().strip().lstrip("@")
IG_CHAR_LIMIT = 2200

# Deduplication: Meta sends at-least-once, so track seen comment IDs
_seen_lock = threading.Lock()
_seen_comments: dict[str, float] = {}  # comment_id -> timestamp
_SEEN_TTL = 3600  # 1 hour


def _prune_seen() -> None:
    """Remove expired entries from the seen-comments cache."""
    cutoff = time.time() - _SEEN_TTL
    expired = [cid for cid, ts in _seen_comments.items() if ts < cutoff]
    for cid in expired:
        del _seen_comments[cid]


def _is_duplicate(comment_id: str) -> bool:
    """Check if we've already processed this comment (Meta at-least-once delivery)."""
    with _seen_lock:
        _prune_seen()
        if comment_id in _seen_comments:
            return True
        _seen_comments[comment_id] = time.time()
        return False


# -----------------------------------------------------------------
# Instagram Graph API helpers
# -----------------------------------------------------------------

def get_media_details(media_id: str) -> dict:
    """Fetch post caption, media type, and media URL."""
    try:
        resp = requests.get(
            f"{GRAPH_API_BASE}/{media_id}",
            params={
                "fields": "id,caption,media_type,media_url,children{media_url,media_type}",
                "access_token": IG_ACCESS_TOKEN,
            },
            timeout=30,
        )
        data = resp.json()
        if "error" in data:
            logger.error(f"[IG] Failed to fetch media {media_id}: {data['error']}")
            return {}
        return data
    except Exception as e:
        logger.error(f"[IG] Error fetching media {media_id}: {e}")
        return {}


def get_image_urls_from_media(media: dict) -> list[str]:
    """Extract image URLs from a media object (single or carousel)."""
    urls = []
    media_type = media.get("media_type", "")

    if media_type == "IMAGE" and media.get("media_url"):
        urls.append(media["media_url"])
    elif media_type == "CAROUSEL_ALBUM":
        children = media.get("children", {}).get("data", [])
        for child in children:
            if child.get("media_type") == "IMAGE" and child.get("media_url"):
                urls.append(child["media_url"])

    return urls


def post_reply(comment_id: str, message: str) -> bool:
    """Reply to an Instagram comment."""
    try:
        resp = requests.post(
            f"{GRAPH_API_BASE}/{comment_id}/replies",
            params={
                "message": message,
                "access_token": IG_ACCESS_TOKEN,
            },
            timeout=30,
        )
        data = resp.json()
        if "error" in data:
            logger.error(f"[IG] Failed to reply to {comment_id}: {data['error']}")
            return False
        logger.info(f"[IG] Reply posted: {data.get('id', 'unknown')}")
        return True
    except Exception as e:
        logger.error(f"[IG] Error posting reply to {comment_id}: {e}")
        return False


def format_reply_text(rag_result: str) -> str:
    """
    Format the RAG result for Instagram's 2,200 char limit.
    Can show ~100 EFTA filenames vs Twitter's ~11.
    """
    if rag_result.lower() == "not found in files":
        return "Not found in the Epstein files."

    PREFIX = "Related documents found in the Epstein files:\n\n"
    filenames = [f.strip() for f in rag_result.splitlines() if f.strip()]
    included = []

    for fname in filenames:
        candidate = PREFIX + "\n".join(included + [fname])
        if len(candidate) <= IG_CHAR_LIMIT:
            included.append(fname)
        else:
            break

    if not included:
        return PREFIX + filenames[0][:IG_CHAR_LIMIT - len(PREFIX)]

    reply = PREFIX + "\n".join(included)

    omitted = len(filenames) - len(included)
    if omitted > 0:
        suffix = f"\n\n(+{omitted} more documents)"
        if len(reply) + len(suffix) <= IG_CHAR_LIMIT:
            reply += suffix

    return reply


# -----------------------------------------------------------------
# Main comment handler (called from webhook in a background thread)
# -----------------------------------------------------------------

def handle_comment_event(value: dict) -> None:
    """
    Process an incoming Instagram comment webhook event.

    Webhook value structure for 'comments' field:
    {
        "id": "comment_id",
        "text": "comment text with @mention",
        "from": {"id": "user_ig_scoped_id", "username": "username"},
        "media": {"id": "media_id", "media_product_type": "FEED"},
        "parent_id": "parent_comment_id"  (only if reply to another comment)
    }
    """
    # Import search_rag here (lazy) to avoid circular import at module load time.
    # x_bot is already imported and FAISS is loaded by bot_main.py before any
    # webhook events arrive.
    from x_bot import search_rag

    comment_id = value.get("id", "")
    comment_text = value.get("text", "")
    from_user = value.get("from", {})
    user_id = from_user.get("id", "")
    username = from_user.get("username", "").lower()
    media_info = value.get("media", {})
    media_id = media_info.get("id", "")

    logger.info(f"[IG] Comment {comment_id} from @{username}: {comment_text[:100]}...")

    # --- Deduplication (Meta at-least-once delivery) ---
    if _is_duplicate(comment_id):
        logger.info(f"[IG] Duplicate comment {comment_id} -- skipping")
        return

    # --- Skip self-replies ---
    if user_id == IG_ACCOUNT_ID or username == IG_USERNAME:
        logger.info(f"[IG] Skipping self-comment from @{username}")
        return

    # --- Check if bot is @mentioned in comment ---
    if IG_USERNAME and f"@{IG_USERNAME}" not in comment_text.lower():
        logger.info(f"[IG] Comment does not mention @{IG_USERNAME}, skipping")
        return

    # --- Rate limiting (reuses security.py, prefixed to avoid collision with Twitter) ---
    if user_id and not rate_limiter.is_allowed(f"ig_{user_id}"):
        logger.info(f"[IG] Rate limited user {user_id} (@{username})")
        return

    # --- Build search query: comment text + parent post caption + OCR ---
    # Remove @mention from query
    query_text = comment_text
    if IG_USERNAME:
        query_text = query_text.lower().replace(f"@{IG_USERNAME}", "").strip()

    # Sanitize comment text
    query_text, query_flags = sanitize_input(query_text)
    if query_flags:
        logger.info(f"[IG] Comment sanitized -- flags: {query_flags}")

    # Fetch parent post caption for context
    caption_text = ""
    image_urls = []
    if media_id:
        media_details = get_media_details(media_id)
        caption_text = media_details.get("caption", "") or ""
        image_urls = get_image_urls_from_media(media_details)

    if caption_text:
        caption_text, caption_flags = sanitize_input(caption_text)
        if caption_flags:
            logger.info(f"[IG] Caption sanitized -- flags: {caption_flags}")

    # OCR any images from the post
    image_text = ""
    for url in image_urls[:5]:  # Max 5 images per carousel
        logger.info(f"[IG] [Vision] Extracting text from: {url}")
        try:
            extracted = extract_text_from_image(url)
            if extracted and extracted.lower() != "no text":
                sanitized_ocr, ocr_flags = sanitize_input(extracted)
                if ocr_flags:
                    logger.info(f"[IG] OCR sanitized -- flags: {ocr_flags}")
                image_text += " " + sanitized_ocr
        except Exception as e:
            logger.error(f"[IG] Vision error: {e}")

    # Combine all context
    search_query = f"{query_text} {caption_text} {image_text}".strip()
    if len(search_query) > 1500:
        search_query = search_query[:1500]

    if not search_query:
        post_reply(comment_id, "Not found in the Epstein files.")
        return

    logger.info(f"[IG] Query: {search_query[:150]}...")

    # --- RAG search (shared FAISS globals from x_bot.py) ---
    try:
        rag_result = search_rag(search_query)
    except Exception as e:
        logger.error(f"[IG] RAG search error: {e}")
        return

    logger.info(f"[IG] RAG result (raw): {rag_result[:200]}")

    # --- Validate output ---
    rag_result = validate_output(rag_result)
    logger.info(f"[IG] RAG result (validated): {rag_result}")

    # --- Format reply for Instagram (2,200 char limit) ---
    reply_text = format_reply_text(rag_result)
    logger.info(f"[IG] Reply ({len(reply_text)} chars): {reply_text[:200]}...")

    # --- Post reply ---
    success = post_reply(comment_id, reply_text)
    if success:
        logger.info(f"[IG] Successfully replied to comment {comment_id}")
    else:
        logger.error(f"[IG] Failed to reply to comment {comment_id}")
