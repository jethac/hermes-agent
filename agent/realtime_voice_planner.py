"""Speech planning helpers for realtime Hermes voice sessions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

_TAG_RE = re.compile(r"<(?:thinking|reasoning|tool|memory-context)\b.*?</(?:thinking|reasoning|tool|memory-context)>", re.I | re.S)
_MEDIA_RE = re.compile(r"MEDIA:\S+")


@dataclass(frozen=True)
class PlannedSpeech:
    """Assistant text that is safe to surface as spoken output."""

    chunks: List[str]
    committed_text: str


class RealtimeSpeechPlanner:
    """Small conservative planner for text-oracle voice output.

    The first implementation only speaks text that has already returned from
    the Hermes oracle. It still centralizes the rules that matter for realtime
    voice: hide internal/tool markup, split text into stable chunks, and mark
    the exact committed text that may be written to durable transcript.
    """

    def plan(self, text: str) -> PlannedSpeech:
        cleaned = self.clean(text)
        return PlannedSpeech(chunks=list(self.chunk(cleaned)), committed_text=cleaned)

    def clean(self, text: str) -> str:
        value = _TAG_RE.sub("", text or "")
        value = _MEDIA_RE.sub("", value)
        value = value.replace("[[audio_as_voice]]", "")
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def chunk(self, text: str) -> Iterable[str]:
        if not text:
            return []

        chunks: List[str] = []
        remaining = text.strip()
        while remaining:
            match = re.match(r"^(.{24,220}?[.!?。！？])(?:\s+|$)", remaining)
            if match:
                chunk = match.group(1).strip()
                chunks.append(chunk)
                remaining = remaining[len(match.group(1)):].strip()
                continue

            if len(remaining) <= 220:
                chunks.append(remaining)
                break

            split_at = max(
                remaining.rfind(", ", 0, 220),
                remaining.rfind("; ", 0, 220),
                remaining.rfind(": ", 0, 220),
                remaining.rfind(" ", 0, 220),
            )
            if split_at < 80:
                split_at = 220
            chunks.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()

        return chunks
