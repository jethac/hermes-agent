"""KAME-style realtime voice request contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Optional


class KameRoute(StrEnum):
    """Routing decisions made by the realtime reflex."""

    LOCAL = "local"
    DEFER = "defer"
    ORACLE_DIRECT = "oracle_direct"
    REJECT_OR_CLARIFY = "reject_or_clarify"


KAME_REFLEX_ROUTES = frozenset(route.value for route in KameRoute)
KAME_VOICE_DENIAL_PATTERNS = (
    "cannot hear",
    "can't hear",
    "can not hear",
    "unable to hear",
    "cannot listen",
    "can't listen",
    "cannot speak",
    "can't speak",
    "cannot join",
    "can't join",
    "only process text",
    "no ability to listen",
    "no ability to join",
    "no ability to speak",
)


@dataclass(frozen=True)
class KameReflexDecision:
    """Validated interface-model decision for one voice turn."""

    text: str
    intent: str
    intent_source: str = "reflex_audio"
    route: KameRoute = KameRoute.ORACLE_DIRECT
    route_confidence: Optional[float] = None
    local_reply: str = ""
    transcript: str = ""
    transcript_source: str = "none"
    transcript_confidence: Optional[float] = None
    validation_errors: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, fallback_text: str = "") -> "KameReflexDecision":
        """Build and validate a reflex decision from model JSON or sidecar payload."""

        transcript = _optional_text(payload.get("transcript"))
        intent = _optional_text(payload.get("intent")) or _optional_text(payload.get("text")) or fallback_text
        text = _optional_text(payload.get("text")) or transcript or intent
        transcript_source = _optional_text(payload.get("transcript_source")) or ("reflex_audio" if transcript else "none")
        local_reply = (
            _optional_text(payload.get("local_reply"))
            or _optional_text(payload.get("reply"))
            or _optional_text(payload.get("clarification"))
            or _optional_text(payload.get("interface_reply"))
        )
        validation_errors: list[str] = []
        raw_route = _optional_text(payload.get("route")).lower()
        if raw_route and raw_route not in KAME_REFLEX_ROUTES:
            validation_errors.append("invalid_route")
        route = _route(raw_route)
        if route in {KameRoute.LOCAL, KameRoute.REJECT_OR_CLARIFY} and not local_reply:
            validation_errors.append("missing_local_reply")
            route = KameRoute.ORACLE_DIRECT
        if local_reply and kame_local_reply_denies_voice_capability(local_reply):
            validation_errors.append("voice_capability_denial")
            route = KameRoute.ORACLE_DIRECT
            local_reply = ""
        return cls(
            text=text.strip(),
            intent=(intent or text).strip(),
            intent_source=_optional_text(payload.get("intent_source")) or "reflex_audio",
            route=route,
            route_confidence=_confidence(
                payload.get("route_confidence") if payload.get("route_confidence") is not None else payload.get("confidence")
            ),
            local_reply=local_reply,
            transcript=transcript.strip(),
            transcript_source=transcript_source,
            transcript_confidence=_confidence(payload.get("transcript_confidence")),
            validation_errors=tuple(validation_errors),
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": self.text,
            "intent": self.intent,
            "intent_source": self.intent_source,
            "transcript_source": self.transcript_source,
            "route": self.route.value,
        }
        if self.route_confidence is not None:
            payload["route_confidence"] = self.route_confidence
        if self.local_reply:
            payload["local_reply"] = self.local_reply
        if self.validation_errors:
            payload["reflex_validation_error"] = ",".join(self.validation_errors)
        if self.transcript:
            payload["transcript"] = self.transcript
        if self.transcript_confidence is not None:
            payload["transcript_confidence"] = self.transcript_confidence
        return payload


@dataclass(frozen=True)
class KameOracleRequest:
    """Structured request from the realtime reflex to the Hermes oracle."""

    session_id: str
    turn_id: str
    source: str
    user_id: Optional[str]
    intent: str
    intent_source: str = "reflex_audio"
    route: KameRoute = KameRoute.ORACLE_DIRECT
    local_reply: str = ""
    transcript: str = ""
    transcript_source: str = "none"
    transcript_confidence: Optional[float] = None
    asr_transcript: str = ""
    asr_transcript_source: str = ""
    asr_transcript_confidence: Optional[float] = None
    mode: str = "voice"
    urgency: str = "interactive"
    interface_already_said: str = ""
    conversation_summary: str = ""
    max_spoken_sentences: int = 2
    requested_response_style: Mapping[str, Any] = field(default_factory=dict)
    cancellation_token: str = ""

    @property
    def oracle_text(self) -> str:
        """Return the best text to persist as the user's oracle-facing message."""

        return (self.asr_transcript or self.transcript or self.intent).strip()

    def to_metadata(self) -> dict[str, Any]:
        response_style = _response_style(
            self.requested_response_style,
            max_sentences=self.max_spoken_sentences,
        )
        metadata: dict[str, Any] = {
            "voice_architecture": "kame_frontend_oracle",
            "kame_turn_id": self.turn_id,
            "kame_source": self.source,
            "kame_intent": self.intent,
            "kame_intent_source": self.intent_source,
            "kame_route": self.route.value,
            "kame_local_reply": self.local_reply,
            "kame_transcript": self.transcript,
            "kame_transcript_source": self.transcript_source,
            "kame_mode": self.mode,
            "kame_urgency": self.urgency,
            "max_spoken_sentences": self.max_spoken_sentences,
            "kame_requested_response_style": response_style,
        }
        if self.user_id:
            metadata["kame_user_id"] = self.user_id
        if self.transcript_confidence is not None:
            metadata["kame_transcript_confidence"] = self.transcript_confidence
        if self.asr_transcript:
            metadata["kame_asr_transcript"] = self.asr_transcript
            metadata["kame_asr_transcript_source"] = self.asr_transcript_source or "asr"
        if self.asr_transcript_confidence is not None:
            metadata["kame_asr_transcript_confidence"] = self.asr_transcript_confidence
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
        default_max_spoken_sentences: int = 2,
    ) -> "KameOracleRequest":
        """Build a KAME oracle request from a reflex/ASR event payload."""

        intent = _optional_text(payload.get("intent")) or _optional_text(payload.get("text")) or fallback_text
        transcript = _optional_text(payload.get("transcript")) or ""
        transcript_source = _optional_text(payload.get("transcript_source"))
        if not transcript_source:
            transcript_source = "reflex_audio" if transcript else "none"
        asr_transcript = (
            _optional_text(payload.get("asr_transcript"))
            or _optional_text(payload.get("oracle_verbatim_transcript"))
            or ""
        )
        asr_transcript_source = (
            _optional_text(payload.get("asr_transcript_source"))
            or _optional_text(payload.get("oracle_verbatim_transcript_source"))
            or ""
        )
        asr_transcript_confidence = _confidence(
            payload.get("asr_transcript_confidence")
            if payload.get("asr_transcript_confidence") is not None
            else payload.get("oracle_verbatim_transcript_confidence")
        )
        if not asr_transcript and transcript and transcript_source.lower().startswith("asr"):
            asr_transcript = transcript
            asr_transcript_source = transcript_source
            asr_transcript_confidence = _confidence(payload.get("transcript_confidence"))
        local_reply = (
            _optional_text(payload.get("local_reply"))
            or _optional_text(payload.get("reply"))
            or _optional_text(payload.get("clarification"))
            or _optional_text(payload.get("interface_reply"))
        )
        requested_response_style = _response_style(
            payload.get("requested_response_style"),
            max_sentences=_positive_int(
                payload.get("max_spoken_sentences"),
                default=default_max_spoken_sentences,
            ),
        )
        return cls(
            session_id=session_id,
            turn_id=_optional_text(payload.get("turn_id")) or turn_id,
            source=_optional_text(payload.get("source")) or source,
            user_id=_optional_text(payload.get("user_id")) or user_id,
            intent=intent.strip(),
            intent_source=_optional_text(payload.get("intent_source")) or "reflex_audio",
            route=_route(payload.get("route")),
            local_reply=local_reply,
            transcript=transcript.strip(),
            transcript_source=transcript_source,
            transcript_confidence=_confidence(payload.get("transcript_confidence")),
            asr_transcript=asr_transcript.strip(),
            asr_transcript_source=asr_transcript_source or ("asr" if asr_transcript else ""),
            asr_transcript_confidence=asr_transcript_confidence,
            mode=_optional_text(payload.get("mode")) or "voice",
            urgency=_optional_text(payload.get("urgency")) or "interactive",
            interface_already_said=_optional_text(payload.get("interface_already_said")) or "",
            conversation_summary=_optional_text(payload.get("conversation_summary")) or "",
            max_spoken_sentences=_positive_int(
                payload.get("max_spoken_sentences")
                if payload.get("max_spoken_sentences") is not None
                else requested_response_style.get("max_sentences"),
                default=default_max_spoken_sentences,
            ),
            requested_response_style=requested_response_style,
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


def _route(value: Any) -> KameRoute:
    text = _optional_text(value).lower()
    if not text:
        return KameRoute.ORACLE_DIRECT
    try:
        return KameRoute(text)
    except ValueError:
        return KameRoute.ORACLE_DIRECT


def kame_local_reply_denies_voice_capability(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    return any(pattern in normalized for pattern in KAME_VOICE_DENIAL_PATTERNS)


def _positive_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _response_style(value: Any, *, max_sentences: int) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, Mapping) else {}
    return {
        "spoken": _bool(raw.get("spoken"), default=True),
        "max_sentences": _positive_int(raw.get("max_sentences"), default=max_sentences),
        "allow_followup_offer": _bool(raw.get("allow_followup_offer"), default=False),
    }


def _bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default
