"""Error sanitization helpers for realtime voice events."""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

from agent.redact import redact_sensitive_text


_BEARER_RE = re.compile(r"\b(Bearer\s+)[A-Za-z0-9._~+/\-=]+", re.IGNORECASE)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b("
    r"token"
    r"|api[_-]?key"
    r"|access[_-]?token"
    r"|refresh[_-]?token"
    r"|auth[_-]?token"
    r"|provider[_-]?token"
    r"|password"
    r"|secret"
    r")=([^&\s]+)",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"\b([a-z][a-z0-9+.-]*://[^\s<>'\"]+)", re.IGNORECASE)


def sanitize_realtime_voice_error(error: Any) -> str:
    """Return a user-visible realtime voice error without credentials."""

    text = str(error or "").strip()
    if not text:
        return "unknown realtime voice error"

    text = redact_sensitive_text(text, force=True)
    text = _BEARER_RE.sub(r"\1***", text)
    text = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=***", text)
    return _URL_RE.sub(_redact_url, text)


def _redact_url(match: re.Match[str]) -> str:
    raw = match.group(1)
    trailing = ""
    while raw and raw[-1] in ".,);]}>":
        trailing = raw[-1] + trailing
        raw = raw[:-1]

    try:
        parsed = urllib.parse.urlparse(raw)
    except Exception:
        return f"***{trailing}"

    if parsed.scheme.lower() not in {"http", "https", "ws", "wss"}:
        return f"***{trailing}"

    netloc = parsed.netloc
    if "@" in netloc:
        _, host_part = netloc.rsplit("@", 1)
        netloc = f"***@{host_part}"

    redacted = urllib.parse.urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))
    return f"{redacted}{trailing}"
