"""
FastAPI app for Instagram webhook endpoints.

Handles Meta webhook verification (GET) and incoming comment
event dispatch (POST). Runs in a daemon thread alongside the
Twitter polling loop (started by bot_main.py).
"""

import os
import hmac
import hashlib
import logging

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import PlainTextResponse

from instagram_bot import handle_comment_event

logger = logging.getLogger("instagram_webhook")

app = FastAPI(title="EFTA Epstein Bot Webhooks")

# Meta webhook config
VERIFY_TOKEN = os.getenv("IG_WEBHOOK_VERIFY_TOKEN", "")
APP_SECRET = os.getenv("IG_APP_SECRET", "")


# -----------------------------------------------------------------
# Health check
# -----------------------------------------------------------------
@app.get("/")
async def health():
    """Health check for Railway."""
    return {"status": "ok", "service": "epstein-bot"}


# -----------------------------------------------------------------
# Webhook verification (one-time, called by Meta when subscribing)
# -----------------------------------------------------------------
@app.get("/webhook/instagram")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    """
    Meta webhook verification challenge-response.

    When you configure a webhook in the Facebook App Dashboard, Meta sends
    a GET request with hub.mode=subscribe, hub.verify_token (your secret),
    and hub.challenge (a random string). You must return hub.challenge as
    plain text to confirm the subscription.
    """
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        logger.info("[Webhook] Verification successful")
        return PlainTextResponse(content=hub_challenge)

    logger.warning(f"[Webhook] Verification failed: mode={hub_mode}")
    raise HTTPException(status_code=403, detail="Verification failed")


# -----------------------------------------------------------------
# Webhook event receiver
# -----------------------------------------------------------------
@app.post("/webhook/instagram")
async def receive_webhook(request: Request):
    """
    Receive Instagram webhook events.

    Meta signs every payload with X-Hub-Signature-256 using your app secret.
    We verify the signature, then dispatch comment events to background
    threads for processing (must return 200 within 20 seconds or Meta retries).
    """
    body = await request.body()

    # --- HMAC signature verification ---
    if APP_SECRET:
        signature_header = request.headers.get("x-hub-signature-256", "")
        expected = "sha256=" + hmac.new(
            APP_SECRET.encode("utf-8"),
            msg=body,
            digestmod=hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(expected, signature_header):
            logger.warning("[Webhook] Invalid signature -- rejecting payload")
            raise HTTPException(status_code=403, detail="Invalid signature")

    # --- Parse and dispatch ---
    payload = await request.json()

    # Only process Instagram events
    if payload.get("object") != "instagram":
        return {"status": "ignored"}

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            field = change.get("field")
            value = change.get("value", {})

            if field == "comments":
                # Process in a background thread so we return 200 fast
                import threading
                threading.Thread(
                    target=handle_comment_event,
                    args=(value,),
                    daemon=True,
                    name=f"ig-comment-{value.get('id', 'unknown')}",
                ).start()

    # Meta requires 200 within 20 seconds or it retries
    return {"status": "ok"}
