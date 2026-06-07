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
            has_whitespace = any(character.isspace() for character in remaining)
            sentence_at = _find_delimiter(remaining, _NON_ASCII_SENTENCE_BOUNDARY_CHARS, start=8, end=220)
            if sentence_at < 0:
                sentence_min = 24 if has_whitespace else 8
                sentence_at = _find_delimiter(remaining, _ASCII_SENTENCE_BOUNDARY_CHARS, start=sentence_min, end=220)
            if sentence_at >= 0:
                chunks.append(remaining[: sentence_at + 1].strip())
                remaining = remaining[sentence_at + 1 :].strip()
                continue

            phrase_min = 80 if has_whitespace else 16
            phrase_trigger = 220 if has_whitespace else 32
            if len(remaining) >= phrase_trigger:
                phrase_at = _find_delimiter(remaining, _PHRASE_BOUNDARY_CHARS, start=phrase_min, end=220)
                if phrase_at >= phrase_min:
                    chunks.append(remaining[: phrase_at + 1].strip())
                    remaining = remaining[phrase_at + 1 :].strip()
                    continue

            if len(remaining) <= 220:
                chunks.append(remaining)
                break

            split_at = max(
                _find_delimiter(remaining, _PHRASE_BOUNDARY_CHARS, start=phrase_min, end=220),
                remaining.rfind(" ", 0, 220),
            )
            if split_at < 80:
                split_at = 220
            suffix_start = split_at + 1 if remaining[split_at] in _PHRASE_BOUNDARY_CHARS else split_at
            chunks.append(remaining[:suffix_start].strip())
            remaining = remaining[suffix_start:].strip()

        return chunks


_ASCII_SENTENCE_BOUNDARY_CHARS = frozenset(".!?")
_NON_ASCII_SENTENCE_BOUNDARY_CHARS = frozenset("。！？؟।")
_PHRASE_BOUNDARY_CHARS = frozenset(",;:，、；：،؛")


def _find_delimiter(text: str, delimiters: frozenset[str], *, start: int, end: int) -> int:
    upper = min(len(text), end)
    for index in range(upper - 1, max(-1, start - 1), -1):
        if text[index] in delimiters:
            return index
    return -1
