"""KAME-style realtime voice request contracts."""

from __future__ import annotations

import copy
import contextlib
from dataclasses import dataclass, field
from enum import StrEnum
import json
import re
from typing import Any, Mapping, Optional, Sequence


class KameRoute(StrEnum):
    """Routing decisions made by the realtime reflex."""

    LOCAL = "local"
    DEFER = "defer"
    ORACLE_DIRECT = "oracle_direct"
    REJECT_OR_CLARIFY = "reject_or_clarify"


KAME_REFLEX_ROUTES = frozenset(route.value for route in KameRoute)
VOICE_RESPONSE_POLICIES = frozenset({"sentence_cap", "brief_summary", "full"})
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
KAME_TOOL_OR_TASK_TERMS = frozenset(
    {
        "call",
        "command",
        "commit",
        "deploy",
        "execute",
        "grep",
        "install",
        "pytest",
        "push",
        "rebase",
        "restart",
        "run",
        "search",
        "shell",
        "tool",
    }
)
KAME_FILE_OR_PROJECT_TERMS = frozenset(
    {
        "branch",
        "code",
        "config",
        "directory",
        "diff",
        "file",
        "folder",
        "function",
        "github",
        "issue",
        "log",
        "pr",
        "project",
        "repo",
        "repository",
        "source",
        "workspace",
    }
)
KAME_MEMORY_TERMS = frozenset({"forget", "memory", "recall", "remember", "save"})
KAME_GREETING_OR_HEAR_ME_TERMS = frozenset(
    {
        "hello",
        "hi",
        "hey",
        "hear",
        "hearing",
        "listening",
        "speak",
        "speaking",
        "test",
        "testing",
        "voice",
    }
)
KAME_PROMOTED_TRANSCRIPT_SOURCES = frozenset(
    {
        "gemma_interpreter",
        "human_verified",
        "interpreter",
        "interpreter_audio",
        "oracle",
        "oracle_verified",
        "user_verified",
        "verified",
    }
)

KAME_REFLEX_DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["route", "intent", "text", "route_confidence", "transcript", "transcript_confidence"],
    "properties": {
        "route": {
            "type": "string",
            "enum": sorted(KAME_REFLEX_ROUTES),
            "description": "Routing path for this spoken turn.",
        },
        "intent": {
            "type": "string",
            "description": "Normalized user intent inferred from the audio segment.",
        },
        "text": {
            "type": "string",
            "description": "Best oracle-facing user wording for this turn.",
        },
        "route_confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence in the selected route.",
        },
        "local_reply": {
            "type": "string",
            "description": "Exact short phrase to speak for local or clarification routes.",
        },
        "transcript": {
            "type": "string",
            "description": "Literal reflex audio hypothesis. Use an empty string if no intelligible speech is present.",
        },
        "transcript_confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "interface_already_said": {
            "type": "string",
            "description": (
                "For defer routes, exact short first-person narration the reflex "
                "already spoke to the user about what it is asking the oracle to do."
            ),
        },
    },
}
KAME_REFLEX_ALLOWED_KEYS = frozenset(KAME_REFLEX_DECISION_JSON_SCHEMA["properties"])
KAME_DIRECT_TOOL_AUTHORITY_KEYS = frozenset(
    {
        "arguments",
        "function",
        "function_call",
        "function_calls",
        "function_name",
        "call_payload",
        "email_payload",
        "file_content",
        "file_write",
        "mcp",
        "mcp_call",
        "mcp_server",
        "mcp_tool",
        "message_payload",
        "tool",
        "tool_call",
        "tool_call_id",
        "tool_calls",
        "tool_name",
        "payment_payload",
        "phone_call_payload",
        "provisioning_payload",
        "purchase_payload",
        "sms_payload",
        "spend_payload",
        "spend_reason",
        "stripe_payload",
        "whatsapp_payload",
        "write_file",
        "write_memory",
        "memory_write",
        "memory_update",
    }
)
KAME_FRONTEND_BRAIN_BRIDGE_NAMES = frozenset(
    {
        "agent_consult",
        "ask_brain",
        "ask_hermes_oracle",
        "openclaw_agent_consult",
    }
)


def kame_reflex_decision_json_schema() -> dict[str, Any]:
    """Return the JSON contract expected from the KAME interface model."""

    return copy.deepcopy(KAME_REFLEX_DECISION_JSON_SCHEMA)


def kame_reflex_instruction_text(
    *,
    routing_policy: str = "",
    asr_mode: str = "on_escalation",
    preflight: bool = False,
) -> str:
    """Prompt text shared by runtime and preflight KAME reflex probes."""

    schema_json = json.dumps(KAME_REFLEX_DECISION_JSON_SCHEMA, sort_keys=True, separators=(",", ":"))
    preflight_text = (
        "This is a preflight probe. The audio may be silence. "
        "Use route=reject_or_clarify and intent='preflight audio probe' if no speech is present. "
        if preflight
        else ""
    )
    routing_text = f" Configured routing policy: {routing_policy}" if routing_policy else ""
    return (
        "You are the low-latency KAME reflex for a Hermes realtime voice session. "
        "Listen to the audio segment and return only a compact JSON object. "
        "Required keys: route, intent, text, route_confidence, transcript, transcript_confidence. "
        "route must be one of local, defer, oracle_direct, or reject_or_clarify. Include route_confidence "
        "and transcript_confidence from 0 to 1. "
        "text should equal the best oracle-facing user wording. For local or reject_or_clarify, "
        "include local_reply with the exact short phrase to speak. "
        "Use intent for what the user wants. Use transcript for the literal words you heard; if the audio "
        "is silence, echo, background sound, or not intelligible, set transcript to an empty string, "
        "transcript_confidence to 0, route to reject_or_clarify, and do not invent a command. "
        "dedicated ASR evidence is attached separately when configured. "
        "For defer routes, include interface_already_said with one concise spoken fragment "
        "that transforms the oracle-facing request into what you are doing now, such as "
        "\"I'm checking the deployment status.\" or \"I'm looking at the logs.\" "
        "This voice session is already connected; never claim Hermes cannot hear, listen, join, "
        "or speak through the live voice interface. For can-you-hear-me checks, use route=local "
        "and a brief affirmative local_reply. Only use local for greetings, repeats, "
        "can-you-hear-me checks, or low-risk conversational glue. Use oracle_direct for tools, "
        "files, memory, projects, or any nontrivial answer. The reflex has no direct tool, MCP, "
        "filesystem, payment, provisioning, message, call, or memory-write authority; "
        "do not include tool/function call fields or action payload fields."
        f"{routing_text} ASR evidence mode is {asr_mode}; do not rely on external tools. "
        f"{preflight_text}JSON schema: {schema_json}. Do not add markdown or commentary."
    )


def kame_reflex_schema_issues(payload: Mapping[str, Any]) -> list[str]:
    """Validate the portable subset of the KAME reflex JSON schema."""

    issues: list[str] = []
    route = str(payload.get("route") or "").strip()
    if route not in KAME_REFLEX_ROUTES:
        issues.append("route must be local, defer, oracle_direct, or reject_or_clarify")
    for key in ("intent", "text"):
        if key not in payload:
            issues.append(f"missing {key}")
    confidence = payload.get("route_confidence")
    if isinstance(confidence, bool):
        issues.append("route_confidence must be numeric")
    else:
        try:
            parsed_confidence = float(confidence)
        except (TypeError, ValueError):
            issues.append("route_confidence must be numeric")
        else:
            if parsed_confidence < 0 or parsed_confidence > 1:
                issues.append("route_confidence must be between 0 and 1")
    if route in {KameRoute.LOCAL.value, KameRoute.REJECT_OR_CLARIFY.value} and "local_reply" not in payload:
        issues.append("local_reply is required for local or reject_or_clarify")
    if route == KameRoute.DEFER.value and not _optional_text(payload.get("interface_already_said")):
        issues.append("interface_already_said is required for defer")
    for key in payload:
        if str(key) not in KAME_REFLEX_ALLOWED_KEYS:
            issues.append(f"unexpected key {key}")
    if kame_payload_requests_direct_tool_authority(payload):
        issues.append("direct tool authority is not allowed for the reflex")
    transcript_confidence = payload.get("transcript_confidence")
    if transcript_confidence is not None:
        if isinstance(transcript_confidence, bool):
            issues.append("transcript_confidence must be numeric")
        else:
            try:
                parsed_transcript_confidence = float(transcript_confidence)
            except (TypeError, ValueError):
                issues.append("transcript_confidence must be numeric")
            else:
                if parsed_transcript_confidence < 0 or parsed_transcript_confidence > 1:
                    issues.append("transcript_confidence must be between 0 and 1")
    return issues


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
        if kame_payload_requests_direct_tool_authority(payload):
            validation_errors.append("direct_tool_authority_not_allowed")
            route = KameRoute.ORACLE_DIRECT
            local_reply = ""
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
    route_confidence: Optional[float] = None
    local_reply: str = ""
    transcript: str = ""
    transcript_source: str = "none"
    transcript_confidence: Optional[float] = None
    asr_transcript: str = ""
    asr_transcript_source: str = ""
    asr_transcript_confidence: Optional[float] = None
    reflex_transcript_hypothesis: str = ""
    reflex_transcript_source: str = ""
    reflex_transcript_confidence: Optional[float] = None
    audio_segment_ref: str = ""
    audio_time_range_ms: tuple[int, int] | tuple[()] = field(default_factory=tuple)
    audio_metadata: Mapping[str, Any] = field(default_factory=dict)
    speaker_metadata: Mapping[str, Any] = field(default_factory=dict)
    channel_metadata: Mapping[str, Any] = field(default_factory=dict)
    auxiliary_transcript_hypotheses: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    mode: str = "voice"
    urgency: str = "interactive"
    interface_already_said: str = ""
    conversation_summary: str = ""
    priority: str = ""
    job_updates: Sequence[str] = field(default_factory=tuple)
    max_spoken_sentences: int = 2
    requested_response_style: Mapping[str, Any] = field(default_factory=dict)
    cancellation_token: str = ""
    reflex_validation_error: str = ""
    interface_input_source: str = ""
    interface_tool_call_id: str = ""
    interface_audio_input_fallback: bool = False
    reflex_provider: str = ""

    @property
    def raw_audio_available(self) -> bool:
        """Return whether this request carries primary raw-audio evidence."""

        return bool(self.audio_segment_ref)

    @property
    def evidence_bundle_status(self) -> str:
        """Return the request's interpreter-evidence completeness class."""

        if self.raw_audio_available:
            return "primary_audio"
        if self.audio_metadata:
            return "degraded_audio_metadata_only"
        if self.interface_audio_input_fallback:
            return "fallback_text_only"
        if _optional_text(self.interface_input_source) in KAME_FRONTEND_BRAIN_BRIDGE_NAMES:
            return "degraded_text_only"
        if _optional_text(self.source) in {"external_kame_frontend", "voiceclaw", "openclaw_talk"}:
            return "degraded_text_only"
        return "degraded_no_raw_audio"

    @property
    def oracle_text(self) -> str:
        """Return promoted text for the user's oracle-facing message."""

        if self.interface_audio_input_fallback and self.asr_transcript:
            return self.asr_transcript.strip()
        if self.transcript and _transcript_source_is_promoted(self.transcript_source):
            return self.transcript.strip()
        return self.intent.strip()

    @property
    def oracle_text_source(self) -> str:
        """Return the source label for ``oracle_text`` so ASR evidence is explicit."""

        if self.interface_audio_input_fallback and self.asr_transcript:
            return self.asr_transcript_source or "asr"
        if self.transcript and _transcript_source_is_promoted(self.transcript_source):
            return self.transcript_source or "reflex_audio"
        return self.intent_source or "reflex_audio"

    @property
    def evidence_authority(self) -> Mapping[str, str]:
        """Return explicit authority labels for evidence fields in this request."""

        authority: dict[str, str] = {
            "intent": _kame_source_authority(self.intent_source, default="reflex_hypothesis"),
            "oracle_text": _kame_source_authority(self.oracle_text_source, default="reflex_hypothesis"),
        }
        if self.audio_segment_ref:
            authority["raw_audio"] = "primary_audio"
        if self.transcript:
            authority["transcript"] = _kame_source_authority(self.transcript_source, default="reflex_hypothesis")
            if self.transcript_source == "reflex_audio":
                authority["reflex_transcript_hypothesis"] = "reflex_hypothesis"
        if self.asr_transcript:
            authority["classic_asr_hypothesis"] = "auxiliary_hypothesis"
        if self.reflex_transcript_hypothesis:
            authority["reflex_transcript_hypothesis"] = "reflex_hypothesis"
        if self.auxiliary_transcript_hypotheses:
            authority["auxiliary_transcript_hypotheses"] = "auxiliary_hypothesis"
        return authority

    @property
    def transcript_hypotheses(self) -> tuple[dict[str, Any], ...]:
        """Return the canonical evidence-bundle transcript hypothesis view."""

        items: list[dict[str, Any]] = []
        reflex_text = self.reflex_transcript_hypothesis.strip()
        reflex_source = self.reflex_transcript_source or "reflex_audio"
        reflex_confidence = self.reflex_transcript_confidence
        if not reflex_text and self.transcript and self.transcript_source == "reflex_audio":
            reflex_text = self.transcript.strip()
            reflex_source = self.transcript_source or "reflex_audio"
            reflex_confidence = self.transcript_confidence
        if reflex_text:
            item: dict[str, Any] = {
                "kind": "reflex_transcript_hypothesis",
                "source": reflex_source,
                "text": reflex_text,
                "authority": "reflex_hypothesis",
            }
            if reflex_confidence is not None:
                item["confidence"] = reflex_confidence
            items.append(item)

        if self.asr_transcript:
            item = {
                "kind": "classic_asr_hypothesis",
                "source": self.asr_transcript_source or "asr",
                "text": self.asr_transcript.strip(),
                "authority": "auxiliary_hypothesis",
            }
            if self.asr_transcript_confidence is not None:
                item["confidence"] = self.asr_transcript_confidence
            items.append(item)

        for hypothesis in self.auxiliary_transcript_hypotheses:
            if not isinstance(hypothesis, Mapping):
                continue
            text = _optional_text(
                hypothesis.get("text")
                or hypothesis.get("transcript")
                or hypothesis.get("hypothesis")
            )
            if not text:
                continue
            source = _optional_text(hypothesis.get("source")) or _optional_text(hypothesis.get("provider")) or "unknown"
            item = {
                "kind": _optional_text(hypothesis.get("kind"))
                or _transcript_hypothesis_kind(source, default="s2s_transcript_hypothesis"),
                "source": source,
                "text": text,
                "authority": "auxiliary_hypothesis",
            }
            confidence = _confidence(hypothesis.get("confidence"))
            if confidence is not None:
                item["confidence"] = confidence
            latency_ms = _non_negative_int(hypothesis.get("latency_ms"))
            if latency_ms is not None:
                item["latency_ms"] = latency_ms
            language = _optional_text(hypothesis.get("language"))
            if language:
                item["language"] = language
            items.append(item)

        compact: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in items:
            key = (str(item.get("kind") or ""), str(item.get("source") or ""), str(item.get("text") or ""))
            if key in seen:
                continue
            seen.add(key)
            compact.append(item)
            if len(compact) >= 8:
                break
        return tuple(compact)

    def to_metadata(self) -> dict[str, Any]:
        response_style = _response_style(
            self.requested_response_style,
            max_sentences=self.max_spoken_sentences,
        )
        metadata: dict[str, Any] = {
            "voice_architecture": "kame_frontend_oracle",
            "session_id": self.session_id,
            "mode": self.mode,
            "urgency": self.urgency,
            "kame_session_id": self.session_id,
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
            "kame_oracle_text_source": self.oracle_text_source,
            "kame_raw_audio_available": self.raw_audio_available,
            "kame_evidence_bundle_status": self.evidence_bundle_status,
            "kame_evidence_authority": dict(self.evidence_authority),
            "max_spoken_sentences": self.max_spoken_sentences,
            "voice_response_policy": response_style.get("policy") or "sentence_cap",
            "kame_requested_response_style": response_style,
        }
        if self.route_confidence is not None:
            metadata["kame_route_confidence"] = self.route_confidence
        if self.user_id:
            metadata["kame_user_id"] = self.user_id
        if self.speaker_metadata:
            metadata["kame_speaker"] = dict(self.speaker_metadata)
        if self.channel_metadata:
            metadata["kame_channel"] = dict(self.channel_metadata)
        if self.transcript_confidence is not None:
            metadata["kame_transcript_confidence"] = self.transcript_confidence
        if self.asr_transcript:
            metadata["kame_asr_transcript"] = self.asr_transcript
            metadata["kame_asr_transcript_source"] = self.asr_transcript_source or "asr"
        if self.asr_transcript_confidence is not None:
            metadata["kame_asr_transcript_confidence"] = self.asr_transcript_confidence
        if self.reflex_transcript_hypothesis:
            metadata["kame_reflex_transcript_hypothesis"] = self.reflex_transcript_hypothesis
            metadata["kame_reflex_transcript_source"] = self.reflex_transcript_source or "reflex_audio"
        if self.reflex_transcript_confidence is not None:
            metadata["kame_reflex_transcript_confidence"] = self.reflex_transcript_confidence
        transcript_hypotheses = self.transcript_hypotheses
        if transcript_hypotheses:
            metadata["kame_transcript_hypotheses"] = transcript_hypotheses
        if self.audio_segment_ref:
            metadata["kame_audio_segment_ref"] = self.audio_segment_ref
        if self.audio_time_range_ms:
            metadata["kame_audio_time_range_ms"] = tuple(self.audio_time_range_ms)
        if self.audio_metadata:
            metadata["kame_audio"] = dict(self.audio_metadata)
        if self.auxiliary_transcript_hypotheses:
            metadata["kame_auxiliary_transcript_hypotheses"] = tuple(
                dict(item) for item in self.auxiliary_transcript_hypotheses
            )
        if self.interface_already_said:
            metadata["kame_interface_already_said"] = self.interface_already_said
        if self.conversation_summary:
            metadata["kame_conversation_summary"] = self.conversation_summary
        if self.priority:
            metadata["kame_priority"] = self.priority
        if self.job_updates:
            metadata["kame_job_updates"] = tuple(_optional_text(update) for update in self.job_updates if _optional_text(update))
        if self.cancellation_token:
            metadata["kame_cancellation_token"] = self.cancellation_token
        if self.reflex_validation_error:
            metadata["kame_reflex_validation_error"] = self.reflex_validation_error
        if self.interface_input_source:
            metadata["kame_interface_input_source"] = self.interface_input_source
        if self.interface_tool_call_id:
            metadata["kame_interface_tool_call_id"] = self.interface_tool_call_id
        if self.interface_audio_input_fallback:
            metadata["kame_interface_audio_input_fallback"] = True
        if self.reflex_provider:
            metadata["kame_reflex_provider"] = self.reflex_provider
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
        routing_policy: Optional[Mapping[str, Any]] = None,
    ) -> "KameOracleRequest":
        """Build a KAME oracle request from a reflex/ASR event payload."""

        payload = apply_kame_routing_policy(payload, routing_policy)
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
        canonical_hypotheses = _canonical_transcript_hypotheses(payload)
        canonical_reflex_hypothesis = _canonical_reflex_transcript_hypothesis(canonical_hypotheses)
        reflex_transcript_hypothesis = (
            _optional_text(payload.get("reflex_transcript_hypothesis"))
            or _optional_text(payload.get("reflex_transcript"))
            or _optional_text(payload.get("reflex_hypothesis"))
            or _optional_text(canonical_reflex_hypothesis.get("text"))
            or (transcript if transcript and transcript_source == "reflex_audio" else "")
        )
        reflex_transcript_source = (
            _optional_text(payload.get("reflex_transcript_source"))
            or _optional_text(canonical_reflex_hypothesis.get("source"))
            or ("reflex_audio" if reflex_transcript_hypothesis else "")
        )
        reflex_transcript_confidence = _confidence(
            payload.get("reflex_transcript_confidence")
            if payload.get("reflex_transcript_confidence") is not None
            else (
                payload.get("reflex_hypothesis_confidence")
                if payload.get("reflex_hypothesis_confidence") is not None
                else canonical_reflex_hypothesis.get("confidence")
            )
        )
        local_reply = (
            _optional_text(payload.get("local_reply"))
            or _optional_text(payload.get("reply"))
            or _optional_text(payload.get("clarification"))
            or _optional_text(payload.get("interface_reply"))
        )
        requested_response_style_payload = (
            dict(payload.get("requested_response_style"))
            if isinstance(payload.get("requested_response_style"), Mapping)
            else {}
        )
        if "policy" not in requested_response_style_payload and payload.get("voice_response_policy") is not None:
            requested_response_style_payload["policy"] = payload.get("voice_response_policy")
        requested_response_style = _response_style(
            requested_response_style_payload,
            max_sentences=_positive_int(
                payload.get("max_spoken_sentences"),
                default=default_max_spoken_sentences,
            ),
        )
        canonical_audio = _canonical_audio_payload(payload)
        canonical_speaker = _canonical_speaker_payload(payload)
        canonical_channel = _canonical_channel_payload(payload)
        resolved_user_id = (
            _optional_text(payload.get("user_id"))
            or _optional_text(payload.get("speaker_id"))
            or _canonical_speaker_user_id(canonical_speaker)
            or user_id
        )
        audio_segment_ref = _optional_text(
            payload.get("audio_segment_ref")
            or payload.get("audio_ref")
            or payload.get("clipped_audio_ref")
            or payload.get("audio_artifact_ref")
            or _canonical_audio_segment_ref(canonical_audio)
        )
        audio_time_range_ms = _audio_time_range_ms(
            payload.get("audio_time_range_ms")
            if payload.get("audio_time_range_ms") is not None
            else _canonical_audio_time_range_ms(canonical_audio)
        )
        return cls(
            session_id=session_id,
            turn_id=_optional_text(payload.get("turn_id")) or turn_id,
            source=_optional_text(payload.get("source")) or _optional_text(payload.get("transport")) or source,
            user_id=resolved_user_id,
            intent=intent.strip(),
            intent_source=_optional_text(payload.get("intent_source")) or "reflex_audio",
            route=_route(payload.get("route")),
            route_confidence=_confidence(
                payload.get("route_confidence") if payload.get("route_confidence") is not None else payload.get("confidence")
            ),
            local_reply=local_reply,
            transcript=transcript.strip(),
            transcript_source=transcript_source,
            transcript_confidence=_confidence(payload.get("transcript_confidence")),
            asr_transcript=asr_transcript.strip(),
            asr_transcript_source=asr_transcript_source or ("asr" if asr_transcript else ""),
            asr_transcript_confidence=asr_transcript_confidence,
            reflex_transcript_hypothesis=reflex_transcript_hypothesis.strip(),
            reflex_transcript_source=reflex_transcript_source,
            reflex_transcript_confidence=reflex_transcript_confidence,
            audio_segment_ref=audio_segment_ref,
            audio_time_range_ms=audio_time_range_ms,
            audio_metadata=_canonical_audio_metadata(
                canonical_audio,
                audio_segment_ref=audio_segment_ref,
                audio_time_range_ms=audio_time_range_ms,
            ),
            speaker_metadata=_canonical_speaker_metadata(
                canonical_speaker,
                user_id=resolved_user_id,
            ),
            channel_metadata=_canonical_channel_metadata(canonical_channel),
            auxiliary_transcript_hypotheses=_auxiliary_transcript_hypotheses(payload),
            mode=_optional_text(payload.get("mode")) or "voice",
            urgency=_optional_text(payload.get("urgency")) or "interactive",
            interface_already_said=_optional_text(payload.get("interface_already_said")) or "",
            conversation_summary=_optional_text(payload.get("conversation_summary")) or "",
            priority=_optional_text(payload.get("priority")) or "",
            job_updates=_job_updates_from_payload(payload.get("job_updates")),
            max_spoken_sentences=_positive_int(
                payload.get("max_spoken_sentences")
                if payload.get("max_spoken_sentences") is not None
                else requested_response_style.get("max_sentences"),
                default=default_max_spoken_sentences,
            ),
            requested_response_style=requested_response_style,
            cancellation_token=_optional_text(payload.get("cancellation_token")) or "",
            reflex_validation_error=_optional_text(payload.get("reflex_validation_error")) or "",
            interface_input_source=_optional_text(payload.get("interface_input_source")) or "",
            interface_tool_call_id=_correlation_id(
                payload.get("interface_tool_call_id") or payload.get("external_tool_call_id")
            ),
            interface_audio_input_fallback=_bool(payload.get("interface_audio_input_fallback"), default=False),
            reflex_provider=_optional_text(payload.get("reflex_provider")) or "",
        )


def kame_external_brain_request_to_oracle_request(
    payload: Mapping[str, Any],
    *,
    session_id: str,
    turn_id: str,
    source: str = "external_kame_frontend",
    user_id: Optional[str] = None,
    default_max_spoken_sentences: int = 2,
) -> KameOracleRequest:
    """Normalize VoiceClaw/OpenClaw-style brain calls into Hermes oracle jobs.

    External realtime frontends may expose an ``ask_brain``-style bridge tool.
    That bridge is allowed only as a request-normalization surface: it unwraps
    into a normal ``KameOracleRequest`` and never grants the frontend direct
    Hermes tool, file, memory, payment, or provisioning authority.
    """

    raw = dict(payload)
    bridge_name = _frontend_bridge_name(raw)
    bridge_arguments = _frontend_bridge_arguments(raw) if bridge_name in KAME_FRONTEND_BRAIN_BRIDGE_NAMES else raw
    normalized = dict(bridge_arguments)
    if "transcript_hypotheses" not in normalized and isinstance(
        raw.get("transcript_hypotheses"),
        (list, tuple),
    ):
        normalized["transcript_hypotheses"] = raw["transcript_hypotheses"]
    if "audio" not in normalized and isinstance(raw.get("audio"), Mapping):
        normalized["audio"] = raw["audio"]
    if "speaker" not in normalized and isinstance(raw.get("speaker"), Mapping):
        normalized["speaker"] = raw["speaker"]
    if "channel" not in normalized and isinstance(raw.get("channel"), Mapping):
        normalized["channel"] = raw["channel"]
    bridge_call_id = _frontend_bridge_call_id(raw)
    if bridge_call_id:
        normalized["interface_tool_call_id"] = bridge_call_id
    if bridge_name:
        normalized["interface_input_source"] = bridge_name
    else:
        normalized.setdefault("interface_input_source", "external_kame_frontend")

    if kame_payload_requests_direct_tool_authority(normalized):
        normalized = _without_direct_tool_authority_fields(normalized)
        normalized["reflex_validation_error"] = ",".join(
            part
            for part in (
                _optional_text(normalized.get("reflex_validation_error")),
                "direct_tool_authority_not_allowed",
            )
            if part
        )

    text = (
        _optional_text(normalized.get("text"))
        or _optional_text(normalized.get("query"))
        or _optional_text(normalized.get("question"))
        or _optional_text(normalized.get("prompt"))
        or _optional_text(normalized.get("message"))
        or _optional_text(normalized.get("request"))
        or _optional_text(normalized.get("intent"))
    )
    intent = _optional_text(normalized.get("intent")) or text
    transcript = _optional_text(normalized.get("transcript"))
    if transcript:
        transcript_source = _optional_text(normalized.get("transcript_source"))
        if _external_transcript_source_is_promoted(transcript_source):
            normalized["transcript"] = transcript
            normalized["transcript_source"] = transcript_source
        else:
            normalized.pop("transcript", None)
            normalized.pop("transcript_source", None)
            _append_auxiliary_transcript_hypothesis(
                normalized,
                {
                    "source": transcript_source
                    or _optional_text(normalized.get("provider"))
                    or source
                    or "external_frontend",
                    "text": transcript,
                    "confidence": normalized.get("transcript_confidence"),
                    "authority": "hypothesis",
                },
            )
    if text:
        normalized["text"] = text
    if intent:
        normalized["intent"] = intent
    normalized["route"] = _external_brain_route(normalized.get("route")).value
    normalized.setdefault("source", source)
    normalized.setdefault(
        "user_id",
        _canonical_speaker_user_id(_canonical_speaker_payload(normalized)) or user_id or "",
    )
    normalized.setdefault("interface_already_said", _frontend_already_said(normalized))
    normalized.setdefault("requested_response_style", {"spoken": True, "max_sentences": default_max_spoken_sentences})
    if not _optional_text(normalized.get("priority")):
        normalized["priority"] = "normal"
    return KameOracleRequest.from_turn(
        session_id=session_id,
        turn_id=turn_id,
        source=source,
        user_id=user_id,
        payload=normalized,
        fallback_text=text or intent,
        default_max_spoken_sentences=default_max_spoken_sentences,
    )


def _external_transcript_source_is_promoted(source: str) -> bool:
    return _transcript_source_is_promoted(source)


def _transcript_source_is_promoted(source: str) -> bool:
    text = _optional_text(source).lower()
    if not text:
        return False
    return text in KAME_PROMOTED_TRANSCRIPT_SOURCES


def _kame_source_authority(source: str, *, default: str) -> str:
    text = _optional_text(source).lower()
    if not text:
        return default
    if text in {"gemma_interpreter", "interpreter", "interpreter_audio"}:
        return "interpreter_promoted"
    if text in {"oracle", "oracle_verified"}:
        return "oracle_promoted"
    if text in {"human_verified", "user_verified", "verified"}:
        return "oracle_promoted"
    if text in {"asr", "asr_fallback", "classic_asr", "stt"} or text.startswith("asr"):
        return "auxiliary_hypothesis"
    if "reflex" in text or "s2s" in text or "moshi" in text or "voiceclaw" in text or "openclaw" in text:
        return "reflex_hypothesis" if "reflex" in text else "auxiliary_hypothesis"
    return default


def _transcript_hypothesis_kind(source: str, *, default: str) -> str:
    text = _optional_text(source).lower()
    if "classic_asr" in text or text in {"asr", "stt"} or text.startswith("asr"):
        return "classic_asr_hypothesis"
    if "reflex" in text:
        return "reflex_transcript_hypothesis"
    if "moshi" in text or "s2s" in text or "voiceclaw" in text or "openclaw" in text:
        return "s2s_transcript_hypothesis"
    return default


def _append_auxiliary_transcript_hypothesis(payload: dict[str, Any], hypothesis: Mapping[str, Any]) -> None:
    raw = payload.get("auxiliary_transcript_hypotheses")
    if isinstance(raw, list):
        items = list(raw)
    elif isinstance(raw, tuple):
        items = list(raw)
    elif isinstance(raw, Mapping):
        items = [dict(raw)]
    elif raw:
        items = [raw]
    else:
        items = []
    items.append(dict(hypothesis))
    payload["auxiliary_transcript_hypotheses"] = items


def _canonical_transcript_hypotheses(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = payload.get("transcript_hypotheses")
    if isinstance(raw, Mapping):
        raw_items: list[Any] = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray, str)):
        raw_items = list(raw)
    else:
        raw_items = []

    hypotheses: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw_items:
        hypothesis = _canonical_transcript_hypothesis(item)
        if not hypothesis:
            continue
        key = (
            str(hypothesis.get("kind") or ""),
            str(hypothesis.get("source") or ""),
            str(hypothesis.get("text") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        hypotheses.append(hypothesis)
        if len(hypotheses) >= 8:
            break
    return tuple(hypotheses)


def _canonical_transcript_hypothesis(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    text = (
        _optional_text(value.get("text"))
        or _optional_text(value.get("transcript"))
        or _optional_text(value.get("hypothesis"))
    )
    if not text:
        return {}
    source = _optional_text(value.get("source")) or _optional_text(value.get("provider")) or "unknown"
    kind = _optional_text(value.get("kind")) or _transcript_hypothesis_kind(
        source,
        default="s2s_transcript_hypothesis",
    )
    authority = _canonical_transcript_hypothesis_authority(value, kind=kind, source=source)
    hypothesis: dict[str, Any] = {
        "kind": kind,
        "source": source,
        "text": text,
        "authority": authority,
    }
    confidence = _confidence(value.get("confidence"))
    if confidence is not None:
        hypothesis["confidence"] = confidence
    latency_ms = _non_negative_int(value.get("latency_ms"))
    if latency_ms is not None:
        hypothesis["latency_ms"] = latency_ms
    language = _optional_text(value.get("language"))
    if language:
        hypothesis["language"] = language
    partial = value.get("partial")
    if isinstance(partial, bool):
        hypothesis["partial"] = partial
    return hypothesis


def _canonical_transcript_hypothesis_authority(
    value: Mapping[str, Any],
    *,
    kind: str,
    source: str,
) -> str:
    authority = _optional_text(value.get("authority"))
    if authority in {"reflex_hypothesis", "auxiliary_hypothesis"}:
        return authority
    if _optional_text(kind) == "reflex_transcript_hypothesis" or "reflex" in _optional_text(source).lower():
        return "reflex_hypothesis"
    return "auxiliary_hypothesis"


def _canonical_reflex_transcript_hypothesis(
    hypotheses: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    for hypothesis in hypotheses:
        kind = _optional_text(hypothesis.get("kind"))
        authority = _optional_text(hypothesis.get("authority"))
        if kind == "reflex_transcript_hypothesis" or authority == "reflex_hypothesis":
            return hypothesis
    return {}


def _canonical_audio_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    audio = payload.get("audio")
    return audio if isinstance(audio, Mapping) else {}


def _canonical_audio_segment_ref(audio: Mapping[str, Any]) -> str:
    return (
        _optional_text(audio.get("audio_segment_ref"))
        or _optional_text(audio.get("segment_ref"))
        or _optional_text(audio.get("artifact_ref"))
        or _optional_text(audio.get("ref"))
        or _optional_text(audio.get("uri"))
    )


def _canonical_audio_time_range_ms(audio: Mapping[str, Any]) -> object:
    if audio.get("time_range_ms") is not None:
        return audio.get("time_range_ms")
    if audio.get("audio_time_range_ms") is not None:
        return audio.get("audio_time_range_ms")
    vad = audio.get("vad")
    if isinstance(vad, Mapping):
        start = vad.get("speech_start_ms")
        end = vad.get("speech_end_ms")
        if start is not None and end is not None:
            return (start, end)
    return ()


def _canonical_audio_metadata(
    audio: Mapping[str, Any],
    *,
    audio_segment_ref: str,
    audio_time_range_ms: tuple[int, int] | tuple[()],
) -> dict[str, Any]:
    if not audio:
        return {}
    metadata: dict[str, Any] = {}
    if audio_segment_ref:
        metadata["audio_segment_ref"] = audio_segment_ref
    if audio_time_range_ms:
        metadata["time_range_ms"] = tuple(audio_time_range_ms)
    for key in ("codec", "authority"):
        value = _optional_text(audio.get(key))
        if value:
            metadata[key] = value[:80]
    for key in ("sample_rate_hz", "channels"):
        value = _non_negative_int(audio.get(key))
        if value is not None:
            metadata[key] = value
    vad = audio.get("vad")
    if isinstance(vad, Mapping):
        compact_vad: dict[str, int] = {}
        for key in ("speech_start_ms", "speech_end_ms"):
            value = _non_negative_int(vad.get(key))
            if value is not None:
                compact_vad[key] = value
        if compact_vad:
            metadata["vad"] = compact_vad
    return metadata


def _canonical_speaker_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    speaker = payload.get("speaker")
    return speaker if isinstance(speaker, Mapping) else {}


def _canonical_speaker_user_id(speaker: Mapping[str, Any]) -> str:
    return (
        _optional_text(speaker.get("channel_user_id"))
        or _optional_text(speaker.get("platform_user_id"))
        or _optional_text(speaker.get("user_id"))
        or _optional_text(speaker.get("speaker_id"))
        or _optional_text(speaker.get("id"))
    )


def _canonical_speaker_metadata(
    speaker: Mapping[str, Any],
    *,
    user_id: Optional[str],
) -> dict[str, Any]:
    if not speaker:
        return {}
    metadata: dict[str, Any] = {}
    for key in (
        "platform",
        "channel_user_id",
        "platform_user_id",
        "user_id",
        "speaker_id",
        "display_name",
        "channel_id",
        "guild_id",
    ):
        value = _optional_text(speaker.get(key))
        if value:
            metadata[key] = " ".join(value.split())[:160]
    is_bot = speaker.get("is_bot")
    if isinstance(is_bot, bool):
        metadata["is_bot"] = is_bot
    if user_id and "user_id" not in metadata:
        metadata["user_id"] = " ".join(str(user_id).split())[:160]
    return metadata


def _canonical_channel_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    channel = payload.get("channel")
    return channel if isinstance(channel, Mapping) else {}


def _canonical_channel_metadata(channel: Mapping[str, Any]) -> dict[str, Any]:
    if not channel:
        return {}
    metadata: dict[str, Any] = {}
    for key in ("transport", "guild_id", "channel_id", "surface", "name"):
        value = _optional_text(channel.get(key))
        if value:
            metadata[key] = " ".join(value.split())[:160]
    return metadata


def _job_updates_from_payload(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        text = _optional_text(value)
        return (text,) if text else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    updates: list[str] = []
    for item in value:
        text = _optional_text(item)
        if text:
            updates.append(text)
    return tuple(updates)


def _audio_time_range_ms(value: object) -> tuple[int, int] | tuple[()]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        return ()
    if len(value) < 2:
        return ()
    start = _non_negative_int(value[0])
    end = _non_negative_int(value[1])
    if start is None or end is None or end < start:
        return ()
    return (start, end)


def _non_negative_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _auxiliary_transcript_hypotheses(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw_items: list[object] = []
    raw = payload.get("auxiliary_transcript_hypotheses")
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray, str)):
        raw_items.extend(raw)
    elif isinstance(raw, Mapping):
        raw_items.append(raw)

    for hypothesis in _canonical_transcript_hypotheses(payload):
        if _canonical_transcript_hypothesis_is_reflex(hypothesis):
            continue
        raw_items.append(hypothesis)

    for source, key in (
        ("moshi", "moshi_transcript_hypothesis"),
        ("s2s", "s2s_transcript_hypothesis"),
        ("asr", "asr_transcript"),
        ("asr", "oracle_verbatim_transcript"),
    ):
        text = _optional_text(payload.get(key))
        if text:
            raw_items.append(
                {
                    "source": source,
                    "text": text,
                    "confidence": payload.get(f"{key}_confidence"),
                }
            )

    hypotheses: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_items:
        hypothesis = _auxiliary_transcript_hypothesis(item)
        if not hypothesis:
            continue
        key = (str(hypothesis.get("source") or ""), str(hypothesis.get("text") or ""))
        if key in seen:
            continue
        seen.add(key)
        hypotheses.append(hypothesis)
        if len(hypotheses) >= 5:
            break
    return tuple(hypotheses)


def _auxiliary_transcript_hypothesis(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        text = _optional_text(value)
        return {"source": "unknown", "text": text, "authority": "hypothesis"} if text else {}
    if not isinstance(value, Mapping):
        return {}
    text = (
        _optional_text(value.get("text"))
        or _optional_text(value.get("transcript"))
        or _optional_text(value.get("hypothesis"))
    )
    if not text:
        return {}
    source = _optional_text(value.get("source")) or _optional_text(value.get("provider")) or "unknown"
    hypothesis: dict[str, Any] = {
        "source": source,
        "text": text,
        "authority": "hypothesis",
    }
    kind = _optional_text(value.get("kind"))
    if kind:
        hypothesis["kind"] = kind
    confidence = _confidence(value.get("confidence"))
    if confidence is not None:
        hypothesis["confidence"] = confidence
    latency_ms = _non_negative_int(value.get("latency_ms"))
    if latency_ms is not None:
        hypothesis["latency_ms"] = latency_ms
    language = _optional_text(value.get("language"))
    if language:
        hypothesis["language"] = language
    return hypothesis


def _canonical_transcript_hypothesis_is_reflex(hypothesis: Mapping[str, Any]) -> bool:
    kind = _optional_text(hypothesis.get("kind"))
    authority = _optional_text(hypothesis.get("authority"))
    return kind == "reflex_transcript_hypothesis" or authority == "reflex_hypothesis"


def apply_kame_routing_policy(
    payload: Mapping[str, Any],
    routing_policy: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Apply Hermes authority gates to a reflex routing payload.

    This is intentionally independent of ``RealtimeVoiceSessionConfig`` so both
    the reference sidecar and alternate sidecars feeding the Hermes KAME engine
    get the same local-route safety behavior.
    """

    routed = dict(payload)
    if kame_payload_requests_direct_tool_authority(routed):
        return downgrade_kame_local_route(
            _without_direct_tool_authority_fields(routed),
            reason="direct_tool_authority_not_allowed",
        )
    route = _optional_text(routed.get("route")).lower()
    if route not in {KameRoute.LOCAL.value, KameRoute.REJECT_OR_CLARIFY.value}:
        return routed
    local_reply = (
        _optional_text(routed.get("local_reply"))
        or _optional_text(routed.get("reply"))
        or _optional_text(routed.get("clarification"))
        or _optional_text(routed.get("interface_reply"))
    )
    if local_reply and kame_local_reply_denies_voice_capability(local_reply):
        return downgrade_kame_local_route(routed, reason="voice_capability_denial")
    routing = routing_policy if isinstance(routing_policy, Mapping) else {}
    if route == KameRoute.LOCAL.value:
        required_reason = kame_oracle_required_reason(routed, routing)
        if required_reason:
            return downgrade_kame_local_route(routed, reason=required_reason)
        if not _bool(routing.get("allow_local_greetings"), default=True) and kame_is_greeting_or_hear_me_check(routed):
            return downgrade_kame_local_route(routed, reason="local_greetings_disabled")
    if route == KameRoute.REJECT_OR_CLARIFY.value and not _bool(
        routing.get("allow_local_clarifications"),
        default=True,
    ):
        return downgrade_kame_local_route(routed, reason="local_clarifications_disabled")
    confidence = _confidence(
        routed.get("route_confidence") if routed.get("route_confidence") is not None else routed.get("confidence")
    )
    if confidence is None:
        return routed
    threshold = _bounded_float(routing.get("local_confidence_threshold"), default=0.75)
    if confidence >= threshold:
        return routed
    return downgrade_kame_local_route(routed, reason="local_confidence_below_threshold")


def downgrade_kame_local_route(payload: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    routed = dict(payload)
    routed["route"] = KameRoute.ORACLE_DIRECT.value
    routed.pop("local_reply", None)
    existing_error = _optional_text(routed.get("reflex_validation_error"))
    routed["reflex_validation_error"] = ",".join(
        part for part in (existing_error, reason) if part
    )
    return routed


def kame_oracle_required_reason(
    payload: Mapping[str, Any],
    routing_policy: Optional[Mapping[str, Any]] = None,
) -> str:
    routing = routing_policy if isinstance(routing_policy, Mapping) else {}
    terms = _policy_terms(payload)
    if (
        _bool(routing.get("require_oracle_for_tools"), default=True)
        and terms.intersection(KAME_TOOL_OR_TASK_TERMS)
    ):
        return "oracle_required_for_tools"
    if (
        _bool(routing.get("require_oracle_for_files"), default=True)
        and terms.intersection(KAME_FILE_OR_PROJECT_TERMS)
    ):
        return "oracle_required_for_files"
    if (
        _bool(routing.get("require_oracle_for_memory"), default=True)
        and terms.intersection(KAME_MEMORY_TERMS)
    ):
        return "oracle_required_for_memory"
    return ""


def kame_is_greeting_or_hear_me_check(payload: Mapping[str, Any]) -> bool:
    return bool(_policy_terms(payload).intersection(KAME_GREETING_OR_HEAR_ME_TERMS))


def kame_payload_requests_direct_tool_authority(payload: Mapping[str, Any]) -> bool:
    """Return whether a reflex payload tries to carry direct tool authority.

    The interface model can route to Hermes's oracle, but the design keeps all
    tool, MCP, filesystem, and memory authority in the oracle layer. These keys
    are therefore treated as an explicit contract violation even if the model
    selected ``defer`` or ``oracle_direct``.
    """

    return any(_is_direct_tool_authority_key(key) for key in payload)


def _without_direct_tool_authority_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if not _is_direct_tool_authority_key(key)
    }


def _is_direct_tool_authority_key(key: Any) -> bool:
    normalized = str(key or "").strip().lower().replace("-", "_")
    if normalized in KAME_DIRECT_TOOL_AUTHORITY_KEYS:
        return True
    return (
        normalized.endswith("_tool_call")
        or normalized.endswith("_tool_calls")
        or normalized.endswith("_function_call")
        or normalized.endswith("_function_calls")
    )


def _policy_terms(payload: Mapping[str, Any]) -> set[str]:
    text = " ".join(
        str(payload.get(key) or "")
        for key in ("text", "transcript", "intent")
    ).lower()
    return set(re.findall(r"[a-z][a-z0-9_-]*", text))


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _correlation_id(value: Any) -> str:
    return " ".join(_optional_text(value).split())[:160]


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


def _external_brain_route(value: Any) -> KameRoute:
    route = _route(value)
    if route in {KameRoute.DEFER, KameRoute.ORACLE_DIRECT}:
        return route
    return KameRoute.ORACLE_DIRECT


def _frontend_bridge_name(payload: Mapping[str, Any]) -> str:
    for key in ("tool_name", "name", "function_name"):
        text = _optional_text(payload.get(key)).lower()
        if text:
            return text
    function = payload.get("function")
    if isinstance(function, Mapping):
        return _optional_text(function.get("name")).lower()
    return ""


def _frontend_bridge_arguments(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("arguments", "args", "input", "parameters"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
        if isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError):
                decoded = json.loads(value)
                if isinstance(decoded, Mapping):
                    return decoded
    function = payload.get("function")
    if isinstance(function, Mapping):
        value = function.get("arguments")
        if isinstance(value, Mapping):
            return value
        if isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError):
                decoded = json.loads(value)
                if isinstance(decoded, Mapping):
                    return decoded
    return {}


def _frontend_bridge_call_id(payload: Mapping[str, Any]) -> str:
    for key in ("tool_call_id", "call_id", "id"):
        text = _correlation_id(payload.get(key))
        if text:
            return text
    function = payload.get("function")
    if isinstance(function, Mapping):
        for key in ("tool_call_id", "call_id", "id"):
            text = _correlation_id(function.get(key))
            if text:
                return text
    return ""


def _frontend_already_said(payload: Mapping[str, Any]) -> str:
    return (
        _optional_text(payload.get("interface_already_said"))
        or _optional_text(payload.get("already_said"))
        or _optional_text(payload.get("spoken_ack"))
        or _optional_text(payload.get("ack_text"))
        or _optional_text(payload.get("placeholder"))
    )


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


def _bounded_float(value: Any, *, default: float) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def _response_style(value: Any, *, max_sentences: int) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, Mapping) else {}
    return {
        "spoken": _bool(raw.get("spoken"), default=True),
        "max_sentences": _positive_int(raw.get("max_sentences"), default=max_sentences),
        "policy": _voice_response_policy(raw.get("policy") or raw.get("voice_response_policy")),
        "allow_followup_offer": _bool(raw.get("allow_followup_offer"), default=False),
    }


def _voice_response_policy(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return text if text in VOICE_RESPONSE_POLICIES else "sentence_cap"


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
