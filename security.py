"""
Security module for the EFTA Twitter Bot.

Provides anti-prompt-injection sanitization, output validation,
and per-user rate limiting to harden the bot against abuse.
"""

import re
import time
from collections import defaultdict


# =============================================================
# 1. Input Sanitization — strip prompt injection patterns
# =============================================================

# Maximum combined query length (caption + OCR text) before truncation.
# Prevents adversarial payloads from flooding the LLM context window.
MAX_QUERY_LENGTH = 1500

# Patterns that attackers commonly use to hijack LLM behaviour.
# Each tuple: (compiled regex, human-readable label for logging).
INJECTION_PATTERNS = [
    # Direct instruction overrides
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|context)",
                re.IGNORECASE), "instruction override"),
    (re.compile(r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
                re.IGNORECASE), "disregard instructions"),
    (re.compile(r"forget\s+(everything|all|your)\s+(you|instructions?|rules?|prompts?)",
                re.IGNORECASE), "forget instructions"),

    # Role / persona hijacking
    (re.compile(r"you\s+are\s+now\s+", re.IGNORECASE), "role hijack"),
    (re.compile(r"act\s+as\s+(if\s+you\s+are|a|an)\s+", re.IGNORECASE), "persona hijack"),
    (re.compile(r"pretend\s+(to\s+be|you\s+are)\s+", re.IGNORECASE), "pretend hijack"),
    (re.compile(r"switch\s+to\s+.{0,30}mode", re.IGNORECASE), "mode switch"),

    # System prompt extraction
    (re.compile(r"(reveal|show|print|output|repeat|echo)\s+(your|the|system)\s+(system\s+)?(prompt|instructions?|rules?)",
                re.IGNORECASE), "prompt extraction"),
    (re.compile(r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?|rules?)",
                re.IGNORECASE), "prompt extraction question"),

    # Delimiter / context escape attempts
    (re.compile(r"```\s*system", re.IGNORECASE), "code-block system escape"),
    (re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", re.IGNORECASE), "raw template tags"),
    (re.compile(r"<\|im_start\|>|<\|im_end\|>", re.IGNORECASE), "ChatML tags"),

    # Output manipulation
    (re.compile(r"(respond|reply|answer|output)\s+with\s+(only|exactly|just)\s*[:\"]",
                re.IGNORECASE), "output manipulation"),
    (re.compile(r"do\s+not\s+(mention|say|include|reference)\s+(that|the|any)\s+(you|context|document)",
                re.IGNORECASE), "output suppression"),

    # Data exfiltration
    (re.compile(r"(send|post|transmit|upload|exfiltrate)\s+.{0,40}(to|at|via)\s+https?://",
                re.IGNORECASE), "data exfiltration URL"),

    # Jailbreak keywords
    (re.compile(r"(DAN|do\s+anything\s+now|developer\s+mode|evil\s+mode|jailbreak)",
                re.IGNORECASE), "jailbreak keyword"),
]


def sanitize_input(text: str) -> tuple[str, list[str]]:
    """
    Sanitise raw tweet/OCR text before it reaches the LLM.

    Returns:
        (cleaned_text, list_of_flagged_patterns)

    The cleaned text has injection patterns redacted and is
    truncated to MAX_QUERY_LENGTH characters.
    """
    if not text:
        return "", []

    flagged: list[str] = []
    cleaned = text

    for pattern, label in INJECTION_PATTERNS:
        if pattern.search(cleaned):
            flagged.append(label)
            cleaned = pattern.sub("[REDACTED]", cleaned)

    # Hard length cap — truncate, don't reject (we still want partial matches)
    if len(cleaned) > MAX_QUERY_LENGTH:
        cleaned = cleaned[:MAX_QUERY_LENGTH]
        flagged.append(f"truncated from {len(text)} to {MAX_QUERY_LENGTH} chars")

    return cleaned.strip(), flagged


# =============================================================
# 2. Output Validation — only post verified EFTA filenames
# =============================================================

# Valid EFTA filename pattern: letters/digits, ends with .pdf/.txt/.jpg etc.
# Examples: EFTA0005421.pdf, AGR-007.pdf, epstein_doc_123.pdf
VALID_FILENAME_RE = re.compile(
    r"^[A-Za-z0-9_\-]{2,60}\.(pdf|txt|jpg|jpeg|png|doc|docx)$"
)

# Known safe response for "no match"
SAFE_NOT_FOUND = "not found in files"


def validate_output(raw_llm_response: str) -> str:
    """
    Validate the LLM response before posting to Twitter.

    Rules:
    - If the response is "not found in files", pass through.
    - Otherwise, every non-empty line must match a valid filename pattern.
    - Lines that don't match are stripped out (the LLM sometimes adds
      commentary despite the strict prompt).
    - If ALL lines are stripped, fall back to "not found in files".

    Returns:
        Validated response string safe to post.
    """
    if not raw_llm_response:
        return SAFE_NOT_FOUND

    stripped = raw_llm_response.strip()

    # "not found in files" — safe
    if stripped.lower() == SAFE_NOT_FOUND:
        return SAFE_NOT_FOUND

    # Validate each line
    valid_lines = []
    rejected_lines = []

    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        if VALID_FILENAME_RE.match(line):
            valid_lines.append(line)
        else:
            rejected_lines.append(line)

    if rejected_lines:
        print(f"  [Security] Rejected {len(rejected_lines)} invalid output line(s):")
        for rl in rejected_lines[:5]:  # Log at most 5 to avoid spam
            print(f"    - {rl[:120]}")

    if not valid_lines:
        return SAFE_NOT_FOUND

    return "\n".join(valid_lines)


# =============================================================
# 3. Per-User Rate Limiting
# =============================================================

class RateLimiter:
    """
    Simple in-memory sliding-window rate limiter.

    Tracks how many times a given user has triggered the bot within
    `window_seconds`. If they exceed `max_requests`, further mentions
    are silently skipped (no reply) until the window expires.
    """

    def __init__(self, max_requests: int = 5, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window = window_seconds
        # user_id -> list of timestamps
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, user_id: str) -> bool:
        """Return True if the user is under the rate limit."""
        now = time.time()
        cutoff = now - self.window

        # Prune old entries
        self._requests[user_id] = [
            ts for ts in self._requests[user_id] if ts > cutoff
        ]

        if len(self._requests[user_id]) >= self.max_requests:
            return False

        self._requests[user_id].append(now)
        return True

    def remaining(self, user_id: str) -> int:
        """Return how many requests the user has left in the current window."""
        now = time.time()
        cutoff = now - self.window
        recent = [ts for ts in self._requests[user_id] if ts > cutoff]
        return max(0, self.max_requests - len(recent))


# Global rate limiter instance — 5 requests per user per hour
rate_limiter = RateLimiter(max_requests=5, window_seconds=3600)
