"""Input/output guardrails: speech-safe text, length limits and PII redaction for logs."""
from __future__ import annotations

import re

MAX_USER_CHARS = 600
MAX_REPLY_CHARS = 420

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_LONG_NUMBER = re.compile(r"\b\d{9,}\b")


def redact(text: str) -> str:
    """Mask e-mails and phone/long numbers so transcripts can be logged without storing personal data."""
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub("[phone]", text)
    return _LONG_NUMBER.sub("[number]", text)


def clean_user_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:MAX_USER_CHARS]


def to_speech(text: str, limit: int = MAX_REPLY_CHARS) -> str:
    """Make model output safe and pleasant to read aloud: no markdown, URLs, emoji or runaway length."""
    text = re.sub(r"https?://\S+", "", text or "")
    text = re.sub(r"[*_`#>|~]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return (cut[: end + 1] if end > 80 else cut.rstrip()) .strip()
