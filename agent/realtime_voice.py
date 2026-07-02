"""Realtime voice protocol primitives for KAME-inspired Hermes sessions.

This module is intentionally transport- and model-agnostic. The first live
implementation is expected to sit behind a websocket endpoint and can choose a
text pipeline (streaming STT -> Hermes oracle -> streaming TTS) or a native
speech-to-speech front-end. Both engines should expose the same event contract
so the desktop app does not know which model stack is active.
"""

from __future__ import annotations

import abc
import asyncio
import base64
import json
import math
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, AsyncIterator, Dict, Mapping, Optional


class RealtimeVoiceEngineKind(StrEnum):
    """Engine families supported by the realtime voice session."""

    TEXT_ORACLE_TTS = "text_oracle_tts"
    NATIVE_S2S_ORACLE = "native_s2s_oracle"
    KAME_INTERFACE_ORACLE = "kame_interface_oracle"


class RealtimeVoiceASRMode(StrEnum):
    """ASR role in a KAME-style realtime voice session."""

    DISABLED = "disabled"
    ON_ESCALATION = "on_escalation"
    SPECULATIVE = "speculative"
    DEBUG = "debug"
    FALLBACK = "fallback"


class VoiceAudioCodec(StrEnum):
    """Wire codecs the browser and backend may exchange."""

    PCM16 = "pcm16"
    OPUS = "opus"
    WEBM_OPUS = "webm_opus"


class VoiceEventType(StrEnum):
    """Typed event names shared by browser, Hermes, and model sidecars."""

    SESSION_STARTED = "session.started"
    SESSION_STOP = "session.stop"
    SESSION_CLOSED = "session.closed"
    SESSION_ERROR = "session.error"
    SESSION_METRICS = "session.metrics"
    AUDIO_INPUT_CHUNK = "audio.input.chunk"
    SPEECH_START = "speech.start"
    SPEECH_ENERGY = "speech.energy"
    SPEECH_END = "speech.end"
    AUDIO_OUTPUT_CHUNK = "audio.output.chunk"
    ASSISTANT_AUDIO_CHUNK = "assistant.audio.chunk"
    ASSISTANT_AUDIO_END = "assistant.audio.end"
    PLAYBACK_STARTED = "playback.started"
    PLAYBACK_STOPPED = "playback.stopped"
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"
    FRONTEND_STATE = "frontend.state"
    INTERFACE_INTENT_PARTIAL = "interface.intent.partial"
    INTERFACE_INTENT_FINAL = "interface.intent.final"
    INTERFACE_REPLY_LOCAL = "interface.reply.local"
    INTERFACE_REPLY_DEFER = "interface.reply.defer"
    INTERFACE_ORACLE_REQUEST = "interface.oracle.request"
    INTERFACE_ORACLE_CANCEL = "interface.oracle.cancel"
    INTERFACE_ORACLE_UPDATE = "interface.oracle.update"
    INTERFACE_COMMIT = "interface.commit"
    ORACLE_JOB_ACCEPTED = "oracle.job.accepted"
    ORACLE_JOB_QUEUED = "oracle.job.queued"
    ORACLE_JOB_STARTED = "oracle.job.started"
    ORACLE_JOB_PROGRESS = "oracle.job.progress"
    ORACLE_JOB_WAITING_FOR_APPROVAL = "oracle.job.waiting_for_approval"
    ORACLE_JOB_COMPLETED = "oracle.job.completed"
    ORACLE_JOB_FAILED = "oracle.job.failed"
    ORACLE_JOB_CANCEL_REQUESTED = "oracle.job.cancel_requested"
    ORACLE_JOB_CANCELLED = "oracle.job.cancelled"
    ORACLE_ACCEPTED = "oracle.accepted"
    ORACLE_TOOL_CALL = "oracle.tool_call"
    ORACLE_TOOL_RESULT = "oracle.tool_result"
    ORACLE_RESPONSE_PARTIAL = "oracle.response.partial"
    ORACLE_RESPONSE_FINAL = "oracle.response.final"
    ORACLE_ERROR = "oracle.error"
    ORACLE_HINT = "oracle.hint"
    ASSISTANT_CAPTION_PARTIAL = "assistant.caption.partial"
    ASSISTANT_CAPTION_FINAL = "assistant.caption.final"
    ASSISTANT_TEXT_PARTIAL = "assistant.text.partial"
    ASSISTANT_COMMIT = "assistant.commit"
    BARGE_IN = "barge_in.detected"
    TOOL_PENDING = "tool.pending"
    TOOL_RESULT = "tool.result"


CLIENT_EVENT_TYPES = frozenset(
    {
        VoiceEventType.AUDIO_INPUT_CHUNK,
        VoiceEventType.SPEECH_START,
        VoiceEventType.SPEECH_ENERGY,
        VoiceEventType.SPEECH_END,
        VoiceEventType.PLAYBACK_STARTED,
        VoiceEventType.PLAYBACK_STOPPED,
        VoiceEventType.BARGE_IN,
        VoiceEventType.INTERFACE_ORACLE_CANCEL,
        VoiceEventType.INTERFACE_ORACLE_UPDATE,
        VoiceEventType.SESSION_STOP,
        VoiceEventType.SESSION_CLOSED,
    }
)

SERVER_EVENT_TYPES = frozenset(
    {
        VoiceEventType.SESSION_STARTED,
        VoiceEventType.SESSION_CLOSED,
        VoiceEventType.SESSION_ERROR,
        VoiceEventType.SESSION_METRICS,
        VoiceEventType.AUDIO_OUTPUT_CHUNK,
        VoiceEventType.ASSISTANT_AUDIO_CHUNK,
        VoiceEventType.ASSISTANT_AUDIO_END,
        VoiceEventType.PLAYBACK_STARTED,
        VoiceEventType.PLAYBACK_STOPPED,
        VoiceEventType.TRANSCRIPT_PARTIAL,
        VoiceEventType.TRANSCRIPT_FINAL,
        VoiceEventType.FRONTEND_STATE,
        VoiceEventType.INTERFACE_INTENT_PARTIAL,
        VoiceEventType.INTERFACE_INTENT_FINAL,
        VoiceEventType.INTERFACE_REPLY_LOCAL,
        VoiceEventType.INTERFACE_REPLY_DEFER,
        VoiceEventType.INTERFACE_ORACLE_REQUEST,
        VoiceEventType.INTERFACE_ORACLE_CANCEL,
        VoiceEventType.INTERFACE_ORACLE_UPDATE,
        VoiceEventType.INTERFACE_COMMIT,
        VoiceEventType.ORACLE_JOB_ACCEPTED,
        VoiceEventType.ORACLE_JOB_QUEUED,
        VoiceEventType.ORACLE_JOB_STARTED,
        VoiceEventType.ORACLE_JOB_PROGRESS,
        VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL,
        VoiceEventType.ORACLE_JOB_COMPLETED,
        VoiceEventType.ORACLE_JOB_FAILED,
        VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED,
        VoiceEventType.ORACLE_JOB_CANCELLED,
        VoiceEventType.ORACLE_ACCEPTED,
        VoiceEventType.ORACLE_TOOL_CALL,
        VoiceEventType.ORACLE_TOOL_RESULT,
        VoiceEventType.ORACLE_RESPONSE_PARTIAL,
        VoiceEventType.ORACLE_RESPONSE_FINAL,
        VoiceEventType.ORACLE_ERROR,
        VoiceEventType.ORACLE_HINT,
        VoiceEventType.ASSISTANT_CAPTION_PARTIAL,
        VoiceEventType.ASSISTANT_CAPTION_FINAL,
        VoiceEventType.ASSISTANT_TEXT_PARTIAL,
        VoiceEventType.ASSISTANT_COMMIT,
        VoiceEventType.BARGE_IN,
        VoiceEventType.TOOL_PENDING,
        VoiceEventType.TOOL_RESULT,
    }
)

OUTPUT_AUDIO_EVENT_TYPES = frozenset(
    {
        VoiceEventType.AUDIO_OUTPUT_CHUNK,
        VoiceEventType.ASSISTANT_AUDIO_CHUNK,
    }
)


def is_output_audio_event_type(value: Any) -> bool:
    try:
        event_type = value if isinstance(value, VoiceEventType) else VoiceEventType(str(value))
    except (TypeError, ValueError):
        return False
    return event_type in OUTPUT_AUDIO_EVENT_TYPES

BINARY_AUDIO_FRAME_HEADER_BYTES = 4
BINARY_AUDIO_FRAME_HEADER_LIMIT = 64 * 1024
REALTIME_VOICE_EVENT_QUEUE_LIMIT = 256
REALTIME_VOICE_SIDECAR_SEND_TIMEOUT_SECONDS = 2.0
TRANSCRIPT_METADATA_KEYS = ("language", "locale", "script")
TRANSCRIPT_METADATA_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
TRANSCRIPT_EVENT_NUMERIC_KEYS = ("confidence", "stability")
TRANSCRIPT_EVENT_GENERATION_KEYS = ("input_generation", "playback_generation")
REALTIME_VOICE_INTERFACE_AUDIO_INPUTS = frozenset({"auto", "native_audio", "text_fallback"})
REALTIME_VOICE_RESPONSE_POLICIES = frozenset({"sentence_cap", "brief_summary", "full"})


@dataclass(frozen=True)
class RealtimeVoiceSessionConfig:
    """Configuration for one realtime voice session."""

    session_id: str
    engine: RealtimeVoiceEngineKind = RealtimeVoiceEngineKind.TEXT_ORACLE_TTS
    input_codec: VoiceAudioCodec = VoiceAudioCodec.OPUS
    output_codec: VoiceAudioCodec = VoiceAudioCodec.OPUS
    sample_rate_hz: int = 16000
    channels: int = 1
    input_buffer_limit_bytes: int = 8 * 1024 * 1024
    frontend_provider: Optional[str] = None
    frontend_model: Optional[str] = None
    interface_temperature: float = 0.2
    interface_max_output_tokens: int = 160
    interface_timeout_seconds: float = 0.8
    interface_max_audio_seconds: float = 30.0
    interface_audio_input: Optional[str] = None
    interface_base_url: Optional[str] = None
    asr_mode: RealtimeVoiceASRMode = RealtimeVoiceASRMode.ON_ESCALATION
    asr_provider: Optional[str] = None
    asr_model: Optional[str] = None
    asr_base_url: Optional[str] = None
    preferred_local_oracle_model: Optional[str] = None
    oracle_timeout_seconds: float = 60.0
    max_spoken_sentences: int = 2
    voice_response_policy: str = "sentence_cap"
    tts_provider: Optional[str] = None
    tts_model: Optional[str] = None
    tts_voice: Optional[str] = None
    tts_base_url: Optional[str] = None
    fallback_policy: Optional[str] = None
    sidecar_base_url: Optional[str] = None
    sidecar_token: Optional[str] = None
    sidecar_connect_timeout_seconds: float = 10.0
    spark_base_url: Optional[str] = None
    spark_token: Optional[str] = None
    turn_acknowledgement: Mapping[str, Any] = field(default_factory=dict)
    routing_policy: Mapping[str, Any] = field(default_factory=dict)
    metrics_policy: Mapping[str, Any] = field(default_factory=dict)
    output_events: Mapping[str, Any] = field(default_factory=dict)
    quality_targets_ms: Mapping[str, Any] = field(default_factory=dict)
    oracle_jobs: Mapping[str, Any] = field(default_factory=dict)
    oracle_tool_router: Mapping[str, Any] = field(default_factory=dict)
    barge_in_policy: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def effective_sidecar_base_url(self) -> Optional[str]:
        return self.sidecar_base_url or self.spark_base_url

    @property
    def effective_sidecar_token(self) -> Optional[str]:
        return self.sidecar_token or self.spark_token

    def to_wire(self) -> Dict[str, Any]:
        sidecar_base_url = self.effective_sidecar_base_url
        sidecar_token = self.effective_sidecar_token
        return {
            "session_id": self.session_id,
            "engine": self.engine.value,
            "input_codec": self.input_codec.value,
            "output_codec": self.output_codec.value,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "input_buffer_limit_bytes": self.input_buffer_limit_bytes,
            "frontend_provider": self.frontend_provider,
            "frontend_model": self.frontend_model,
            "interface_temperature": self.interface_temperature,
            "interface_max_output_tokens": self.interface_max_output_tokens,
            "interface_timeout_seconds": self.interface_timeout_seconds,
            "interface_max_audio_seconds": self.interface_max_audio_seconds,
            "interface_audio_input": normalize_realtime_voice_interface_audio_input(self.interface_audio_input),
            "interface_base_url": self.interface_base_url,
            "asr_mode": self.asr_mode.value,
            "asr_provider": self.asr_provider,
            "asr_model": self.asr_model,
            "asr_base_url": self.asr_base_url,
            "preferred_local_oracle_model": self.preferred_local_oracle_model,
            "oracle_timeout_seconds": self.oracle_timeout_seconds,
            "max_spoken_sentences": self.max_spoken_sentences,
            "voice_response_policy": normalize_realtime_voice_response_policy(self.voice_response_policy),
            "tts_provider": self.tts_provider,
            "tts_model": self.tts_model,
            "tts_voice": self.tts_voice,
            "tts_base_url": self.tts_base_url,
            "fallback_policy": self.fallback_policy,
            "sidecar_base_url": sidecar_base_url,
            "sidecar_token": sidecar_token,
            "sidecar_connect_timeout_seconds": self.sidecar_connect_timeout_seconds,
            "spark_base_url": sidecar_base_url,
            "spark_token": sidecar_token,
            "turn_acknowledgement": dict(self.turn_acknowledgement),
            "routing_policy": dict(self.routing_policy),
            "metrics_policy": dict(self.metrics_policy),
            "output_events": dict(self.output_events),
            "quality_targets_ms": dict(self.quality_targets_ms),
            "oracle_jobs": dict(self.oracle_jobs),
            "oracle_tool_router": dict(self.oracle_tool_router),
            "barge_in_policy": dict(self.barge_in_policy),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "RealtimeVoiceSessionConfig":
        return cls(
            session_id=str(payload["session_id"]),
            engine=RealtimeVoiceEngineKind(str(payload.get("engine") or RealtimeVoiceEngineKind.TEXT_ORACLE_TTS.value)),
            input_codec=VoiceAudioCodec(str(payload.get("input_codec") or VoiceAudioCodec.OPUS.value)),
            output_codec=VoiceAudioCodec(str(payload.get("output_codec") or VoiceAudioCodec.OPUS.value)),
            sample_rate_hz=int(payload.get("sample_rate_hz") or 16000),
            channels=int(payload.get("channels") or 1),
            input_buffer_limit_bytes=_positive_int(
                payload.get("input_buffer_limit_bytes"),
                default=8 * 1024 * 1024,
            ),
            frontend_provider=_optional_str(payload.get("frontend_provider")),
            frontend_model=_optional_str(payload.get("frontend_model")),
            interface_temperature=_bounded_float(
                payload.get("interface_temperature"),
                default=0.2,
                minimum=0.0,
                maximum=2.0,
            ),
            interface_max_output_tokens=_positive_int(
                payload.get("interface_max_output_tokens"),
                default=160,
            ),
            interface_timeout_seconds=_positive_float(
                payload.get("interface_timeout_seconds"),
                default=0.8,
            ),
            interface_max_audio_seconds=_bounded_float(
                payload.get("interface_max_audio_seconds"),
                default=30.0,
                minimum=1.0,
                maximum=30.0,
            ),
            interface_audio_input=normalize_realtime_voice_interface_audio_input(payload.get("interface_audio_input")),
            interface_base_url=_optional_str(payload.get("interface_base_url")),
            asr_mode=_asr_mode(payload.get("asr_mode")),
            asr_provider=_optional_str(payload.get("asr_provider")),
            asr_model=_optional_str(payload.get("asr_model")),
            asr_base_url=_optional_str(payload.get("asr_base_url")),
            preferred_local_oracle_model=_optional_str(payload.get("preferred_local_oracle_model")),
            oracle_timeout_seconds=_positive_float(
                payload.get("oracle_timeout_seconds"),
                default=60.0,
            ),
            max_spoken_sentences=_positive_int(
                payload.get("max_spoken_sentences"),
                default=2,
            ),
            voice_response_policy=normalize_realtime_voice_response_policy(payload.get("voice_response_policy")),
            tts_provider=_optional_str(payload.get("tts_provider")),
            tts_model=_optional_str(payload.get("tts_model")),
            tts_voice=_optional_str(payload.get("tts_voice")),
            tts_base_url=_optional_str(payload.get("tts_base_url")),
            fallback_policy=_optional_str(payload.get("fallback_policy")),
            sidecar_base_url=_optional_str(payload.get("sidecar_base_url")),
            sidecar_token=_optional_str(payload.get("sidecar_token")),
            sidecar_connect_timeout_seconds=_positive_float(
                payload.get("sidecar_connect_timeout_seconds"),
                default=10.0,
            ),
            spark_base_url=_optional_str(payload.get("spark_base_url")),
            spark_token=_optional_str(payload.get("spark_token")),
            turn_acknowledgement=_mapping(payload.get("turn_acknowledgement")),
            routing_policy=_mapping(payload.get("routing_policy")),
            metrics_policy=_mapping(payload.get("metrics_policy")),
            output_events=_mapping(payload.get("output_events")),
            quality_targets_ms=_mapping(payload.get("quality_targets_ms")),
            oracle_jobs=_mapping(payload.get("oracle_jobs")),
            oracle_tool_router=_mapping(payload.get("oracle_tool_router")),
            barge_in_policy=_mapping(payload.get("barge_in_policy")),
            metadata=_mapping(payload.get("metadata")),
        )


@dataclass(frozen=True)
class VoiceEvent:
    """JSON-serializable realtime voice event."""

    type: VoiceEventType
    session_id: str
    sequence: int
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_wire(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "timestamp_ms": self.timestamp_ms,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "VoiceEvent":
        return cls(
            type=voice_event_type_from_wire(data["type"]),
            session_id=str(data["session_id"]),
            sequence=int(data["sequence"]),
            timestamp_ms=int(data.get("timestamp_ms") or int(time.time() * 1000)),
            payload=_mapping(data.get("payload")),
        )


@dataclass(frozen=True)
class AudioChunk:
    """Base64 audio payload for websocket JSON transport."""

    codec: VoiceAudioCodec
    data: bytes
    sample_rate_hz: int = 16000
    channels: int = 1

    def to_payload(self) -> Dict[str, Any]:
        return {
            "codec": self.codec.value,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "data_b64": base64.b64encode(self.data).decode("ascii"),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AudioChunk":
        raw = payload.get("data_b64")
        if not isinstance(raw, str):
            raise ValueError("audio chunk payload requires data_b64")
        return cls(
            codec=VoiceAudioCodec(str(payload["codec"])),
            sample_rate_hz=int(payload.get("sample_rate_hz") or 16000),
            channels=int(payload.get("channels") or 1),
            data=base64.b64decode(raw.encode("ascii"), validate=True),
        )


class RealtimeVoiceEngine(abc.ABC):
    """Abstract base class for realtime Hermes voice engines."""

    @property
    @abc.abstractmethod
    def kind(self) -> RealtimeVoiceEngineKind:
        """Return the engine family."""

    @abc.abstractmethod
    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        """Start resources for a voice session."""

    @abc.abstractmethod
    async def receive_event(self, event: VoiceEvent) -> None:
        """Receive a client control event or audio chunk."""

    @abc.abstractmethod
    async def events(self) -> AsyncIterator[VoiceEvent]:
        """Yield server events until the session closes."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Release resources and stop background work."""


def realtime_voice_session_contract_payload(config: RealtimeVoiceSessionConfig) -> Dict[str, Any]:
    """Return sanitized, non-secret session contract metadata for client events."""

    metadata = config.metadata if isinstance(config.metadata, Mapping) else {}
    payload: Dict[str, Any] = {}
    first_class_mappings: tuple[tuple[str, Mapping[str, Any]], ...] = (
        ("quality_targets_ms", config.quality_targets_ms),
        ("routing", config.routing_policy),
        ("metrics", config.metrics_policy),
        ("oracle_jobs", config.oracle_jobs),
        ("barge_in", config.barge_in_policy),
    )
    for key, value in first_class_mappings:
        if isinstance(value, Mapping) and value:
            payload[key] = dict(value)
    for key in ("language_support", "quality_targets_ms", "conversation_quality", "routing", "metrics", "oracle_jobs", "barge_in"):
        if key in payload:
            continue
        value = metadata.get(key)
        if isinstance(value, Mapping):
            payload[key] = dict(value)
    return payload


def binary_audio_frame_from_event(event: VoiceEvent) -> Optional[bytes]:
    payload = dict(event.payload)
    raw = payload.pop("data_b64", None)
    if not isinstance(raw, str) or not raw:
        return None
    try:
        audio = base64.b64decode(raw.encode("ascii"), validate=True)
    except Exception:
        return None

    wire = event.to_wire()
    wire["payload"] = payload
    header = json.dumps(wire, separators=(",", ":")).encode("utf-8")
    if not header or len(header) > BINARY_AUDIO_FRAME_HEADER_LIMIT:
        return None

    return len(header).to_bytes(BINARY_AUDIO_FRAME_HEADER_BYTES, "big") + header + audio


def event_from_binary_audio_frame(frame: bytes, *, expected_type: Optional[VoiceEventType] = None) -> VoiceEvent:
    if len(frame) < BINARY_AUDIO_FRAME_HEADER_BYTES:
        raise ValueError("binary audio frame missing header length")

    header_length = int.from_bytes(frame[:BINARY_AUDIO_FRAME_HEADER_BYTES], "big", signed=False)
    if header_length <= 0 or header_length > BINARY_AUDIO_FRAME_HEADER_LIMIT:
        raise ValueError("binary audio frame header length is invalid")

    header_start = BINARY_AUDIO_FRAME_HEADER_BYTES
    header_end = header_start + header_length
    if len(frame) < header_end:
        raise ValueError("binary audio frame header is truncated")

    header = json.loads(frame[header_start:header_end].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("binary audio frame header must be a JSON object")

    payload = header.get("payload")
    payload = dict(payload) if isinstance(payload, dict) else {}
    header["payload"] = payload
    payload["data_b64"] = base64.b64encode(frame[header_end:]).decode("ascii")

    event = VoiceEvent.from_wire(header)
    if expected_type is not None and event.type != expected_type:
        raise ValueError(f"binary audio frame must carry {expected_type.value}")
    return event


def event_from_binary_output_audio_frame(frame: bytes) -> VoiceEvent:
    event = event_from_binary_audio_frame(frame)
    if not is_output_audio_event_type(event.type):
        raise ValueError("binary audio frame must carry an output audio event")
    return event


def create_realtime_voice_event_queue() -> asyncio.Queue[VoiceEvent | None]:
    return asyncio.Queue(maxsize=REALTIME_VOICE_EVENT_QUEUE_LIMIT)


async def put_realtime_voice_event(queue: asyncio.Queue[VoiceEvent | None], event: VoiceEvent | None) -> bool:
    """Queue a realtime voice event without letting audio backlog grow forever.

    Audio chunks are expendable under pressure; control, transcript, and error
    events are not. When the queue is full, drop the oldest queued audio first.
    If a non-audio event still cannot fit, evict the oldest queued item so the
    latest control state can reach the desktop.
    """

    if queue.maxsize <= 0:
        await queue.put(event)
        return True

    try:
        queue.put_nowait(event)
        return True
    except asyncio.QueueFull:
        pass

    if _drop_one_queued_audio_event(queue):
        try:
            queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            pass

    if isinstance(event, VoiceEvent) and is_output_audio_event_type(event.type):
        return False

    _drop_oldest_queued_event(queue)
    try:
        queue.put_nowait(event)
        return True
    except asyncio.QueueFull:
        return False


def _drop_one_queued_audio_event(queue: asyncio.Queue[VoiceEvent | None]) -> bool:
    kept: list[VoiceEvent | None] = []
    dropped = False
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if not dropped and isinstance(item, VoiceEvent) and is_output_audio_event_type(item.type):
            dropped = True
            continue
        kept.append(item)
    for item in kept:
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            break
    return dropped


def _drop_oldest_queued_event(queue: asyncio.Queue[VoiceEvent | None]) -> bool:
    try:
        queue.get_nowait()
        return True
    except asyncio.QueueEmpty:
        return False


def validate_client_event(event: VoiceEvent) -> None:
    if event.type not in CLIENT_EVENT_TYPES:
        raise ValueError(f"{event.type.value!r} is not accepted from clients")


def validate_server_event(event: VoiceEvent) -> None:
    if event.type not in SERVER_EVENT_TYPES:
        raise ValueError(f"{event.type.value!r} is not a server event")


def voice_event_type_from_wire(value: Any) -> VoiceEventType:
    """Parse event names, accepting legacy wire aliases."""

    text = str(value)
    if text == "barge_in":
        return VoiceEventType.BARGE_IN
    return VoiceEventType(text)


def transcript_metadata_from_payload(payload: Mapping[str, Any]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for key in TRANSCRIPT_METADATA_KEYS:
        value = payload.get(key)
        if not isinstance(value, str):
            continue
        token = value.strip()
        if TRANSCRIPT_METADATA_VALUE_RE.fullmatch(token):
            metadata[key] = token
    return metadata


def transcript_event_payload_from_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a sanitized transcript payload safe to forward to clients."""

    sanitized: Dict[str, Any] = {}
    text = payload.get("text")
    if isinstance(text, str):
        sanitized["text"] = text

    for key in TRANSCRIPT_EVENT_NUMERIC_KEYS:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            sanitized[key] = max(0.0, min(1.0, float(value)))

    for key in TRANSCRIPT_EVENT_GENERATION_KEYS:
        value = _wire_non_negative_int(payload.get(key))
        if value is not None:
            sanitized[key] = value

    sanitized.update(transcript_metadata_from_payload(payload))
    return sanitized


def _wire_non_negative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_realtime_voice_interface_audio_input(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_")
    if not text:
        return None
    return text if text in REALTIME_VOICE_INTERFACE_AUDIO_INPUTS else "auto"


def normalize_realtime_voice_response_policy(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return text if text in REALTIME_VOICE_RESPONSE_POLICIES else "sentence_cap"


def _asr_mode(value: Any) -> RealtimeVoiceASRMode:
    if value is None:
        return RealtimeVoiceASRMode.ON_ESCALATION
    try:
        return RealtimeVoiceASRMode(str(value).strip().lower())
    except ValueError:
        return RealtimeVoiceASRMode.ON_ESCALATION


def _mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("expected an object mapping")
    return dict(value)


def _positive_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _bounded_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _positive_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
