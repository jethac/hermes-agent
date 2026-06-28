"""KAME-style realtime voice request contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Optional


class KameRoute(StrEnum):
    """Routing decisions made by the realtime reflex."""

    LOCAL = "local"
    DEFER = "defer"
    ORACLE_DIRECT = "oracle_direct"
    REJECT_OR_CLARIFY = "reject_or_clarify"


@dataclass(frozen=True)
class KameOracleRequest:
    """Structured request from the realtime reflex to the Hermes oracle."""

    session_id: str
    turn_id: str
    source: str
    user_id: Optional[str]
    intent: str
    intent_source: str = "reflex_audio"
    transcript: str = ""
    transcript_source: str = "none"
    transcript_confidence: Optional[float] = None
    mode: str = "voice"
    urgency: str = "interactive"
    interface_already_said: str = ""
    conversation_summary: str = ""
    max_spoken_sentences: int = 2
    cancellation_token: str = ""

    @property
    def oracle_text(self) -> str:
        """Return the best text to persist as the user's oracle-facing message."""

        return (self.transcript or self.intent).strip()

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "voice_architecture": "kame_frontend_oracle",
            "kame_turn_id": self.turn_id,
            "kame_source": self.source,
            "kame_intent": self.intent,
            "kame_intent_source": self.intent_source,
            "kame_transcript": self.transcript,
            "kame_transcript_source": self.transcript_source,
            "kame_mode": self.mode,
            "kame_urgency": self.urgency,
            "max_spoken_sentences": self.max_spoken_sentences,
        }
        if self.user_id:
            metadata["kame_user_id"] = self.user_id
        if self.transcript_confidence is not None:
            metadata["kame_transcript_confidence"] = self.transcript_confidence
        if self.interface_already_said:
            metadata["kame_interface_already_said"] = self.interface_already_said
        if self.conversation_summary:
            metadata["kame_conversation_summary"] = self.conversation_summary
        if self.cancellation_token:
            metadata["kame_cancellation_token"] = self.cancellation_token
        return metadata

    @classmethod
    def from_turn(
        cls,
        *,
        session_id: str,
        turn_id: str,
        source: str,
        user_id: Optional[str],
        payload: Mapping[str, Any],
        fallback_text: str,
    ) -> "KameOracleRequest":
        """Build a KAME oracle request from a reflex/ASR event payload."""

        intent = _optional_text(payload.get("intent")) or _optional_text(payload.get("text")) or fallback_text
        transcript = _optional_text(payload.get("transcript")) or _optional_text(payload.get("asr_transcript")) or ""
        transcript_source = _optional_text(payload.get("transcript_source"))
        if not transcript_source:
            transcript_source = "asr" if transcript else "none"
        return cls(
            session_id=session_id,
            turn_id=_optional_text(payload.get("turn_id")) or turn_id,
            source=_optional_text(payload.get("source")) or source,
            user_id=_optional_text(payload.get("user_id")) or user_id,
            intent=intent.strip(),
            intent_source=_optional_text(payload.get("intent_source")) or "reflex_audio",
            transcript=transcript.strip(),
            transcript_source=transcript_source,
            transcript_confidence=_confidence(payload.get("transcript_confidence")),
            mode=_optional_text(payload.get("mode")) or "voice",
            urgency=_optional_text(payload.get("urgency")) or "interactive",
            interface_already_said=_optional_text(payload.get("interface_already_said")) or "",
            conversation_summary=_optional_text(payload.get("conversation_summary")) or "",
            max_spoken_sentences=_positive_int(payload.get("max_spoken_sentences"), default=2),
            cancellation_token=_optional_text(payload.get("cancellation_token")) or "",
        )


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _confidence(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _positive_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
