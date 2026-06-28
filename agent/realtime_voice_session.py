"""Realtime voice session state machine."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional

from agent.realtime_voice import (
    RealtimeVoiceEngine,
    RealtimeVoiceEngineKind,
    RealtimeVoiceSessionConfig,
    VoiceEvent,
    VoiceEventType,
    validate_client_event,
    validate_server_event,
)
from agent.realtime_voice_s2s_engine import NativeS2SSidecarEngine
from agent.realtime_voice_text_engine import KameInterfaceOracleEngine, TextOracleTTSEngine


class RealtimeVoiceSessionState(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    LISTENING = "listening"
    ASSISTANT_PENDING = "assistant_pending"
    SPEAKING = "speaking"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class RealtimeVoiceTranscript:
    partial_user_text: str = ""
    final_user_segments: List[str] = field(default_factory=list)
    assistant_draft: str = ""
    active_playback_generation: int = 0
    committed_assistant_segments: List[str] = field(default_factory=list)
    interrupted_assistant_segments: List[str] = field(default_factory=list)


STALE_GENERATION_EVENT_TYPES = frozenset(
    {
        VoiceEventType.AUDIO_OUTPUT_CHUNK,
        VoiceEventType.ASSISTANT_AUDIO_CHUNK,
        VoiceEventType.ASSISTANT_AUDIO_END,
        VoiceEventType.PLAYBACK_STARTED,
        VoiceEventType.PLAYBACK_STOPPED,
        VoiceEventType.INTERFACE_INTENT_FINAL,
        VoiceEventType.INTERFACE_REPLY_LOCAL,
        VoiceEventType.INTERFACE_REPLY_DEFER,
        VoiceEventType.INTERFACE_ORACLE_REQUEST,
        VoiceEventType.INTERFACE_ORACLE_CANCEL,
        VoiceEventType.INTERFACE_COMMIT,
        VoiceEventType.ORACLE_ACCEPTED,
        VoiceEventType.ORACLE_TOOL_CALL,
        VoiceEventType.ORACLE_TOOL_RESULT,
        VoiceEventType.ORACLE_RESPONSE_PARTIAL,
        VoiceEventType.ORACLE_RESPONSE_FINAL,
        VoiceEventType.ORACLE_ERROR,
        VoiceEventType.SESSION_METRICS,
        VoiceEventType.ASSISTANT_COMMIT,
        VoiceEventType.ASSISTANT_CAPTION_FINAL,
        VoiceEventType.ASSISTANT_CAPTION_PARTIAL,
        VoiceEventType.ASSISTANT_TEXT_PARTIAL,
        VoiceEventType.TRANSCRIPT_FINAL,
    }
)

QUALITY_TARGET_METRIC_KEYS = frozenset(
    {
        "audio_to_partial_transcript_ms",
        "final_transcript_to_first_text_ms",
        "final_transcript_to_first_audio_ms",
        "barge_in_ack_ms",
    }
)


class RealtimeVoiceSession:
    """Owns state and persistence boundaries for one realtime voice session."""

    def __init__(
        self,
        config: RealtimeVoiceSessionConfig,
        *,
        engine: Optional[RealtimeVoiceEngine] = None,
    ):
        self.config = config
        self.engine = engine or create_realtime_voice_engine(config)
        self.state = RealtimeVoiceSessionState.IDLE
        self.transcript = RealtimeVoiceTranscript()
        self._last_client_sequence = 0
        self._closed = False
        self._started_at_monotonic: Optional[float] = None
        self._turn_audio_started_at: Optional[float] = None
        self._turn_eou_at: Optional[float] = None
        self._response_speech_boundary_at: Optional[float] = None
        self._last_transcript_final_at: Optional[float] = None
        self._last_barge_in_at: Optional[float] = None
        self._turn_first_assistant_text = False
        self._turn_first_audio_output = False
        self._quality_target_miss_count = 0
        self._last_quality_target_miss: Optional[dict] = None

    async def start(self) -> None:
        self._started_at_monotonic = time.monotonic()
        self.state = RealtimeVoiceSessionState.STARTING
        await self.engine.start(self.config)
        self.state = RealtimeVoiceSessionState.LISTENING

    async def receive_client_event(self, event: VoiceEvent) -> None:
        validate_client_event(event)
        if event.sequence <= self._last_client_sequence:
            raise ValueError("client event sequence must increase monotonically")
        self._last_client_sequence = event.sequence
        now = time.monotonic()
        if event.type == VoiceEventType.SPEECH_START:
            if self._turn_audio_started_at is None:
                self._turn_audio_started_at = now
            self.state = RealtimeVoiceSessionState.LISTENING
        elif event.type == VoiceEventType.SPEECH_END:
            self._turn_eou_at = now
        if event.type == VoiceEventType.AUDIO_INPUT_CHUNK:
            if self._turn_audio_started_at is None:
                self._turn_audio_started_at = now
            if event.payload.get("end_of_utterance") is True:
                self._turn_eou_at = now
        if event.type == VoiceEventType.BARGE_IN:
            self._last_barge_in_at = now
            self.transcript.active_playback_generation += 1
            self.state = RealtimeVoiceSessionState.LISTENING
            if self.transcript.assistant_draft:
                self.transcript.interrupted_assistant_segments.append(self.transcript.assistant_draft)
                self.transcript.assistant_draft = ""
        elif event.type == VoiceEventType.SESSION_CLOSED:
            self.state = RealtimeVoiceSessionState.CLOSING
        await self.engine.receive_event(event)

    async def events(self) -> AsyncIterator[VoiceEvent]:
        async for event in self.engine.events():
            validate_server_event(event)
            if self._should_drop_stale_server_event(event):
                continue
            annotated = self._annotate_server_event(event)
            self._apply_server_event(annotated)
            yield annotated

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.state = RealtimeVoiceSessionState.CLOSING
        await self.engine.close()
        self.state = RealtimeVoiceSessionState.CLOSED

    def durable_messages(self) -> List[dict]:
        messages: List[dict] = []
        for text in self.transcript.final_user_segments:
            if text.strip():
                messages.append({"role": "user", "content": text.strip()})
        for text in self.transcript.committed_assistant_segments:
            if text.strip():
                messages.append({"role": "assistant", "content": text.strip()})
        return messages

    def _apply_server_event(self, event: VoiceEvent) -> None:
        if event.type == VoiceEventType.TRANSCRIPT_PARTIAL:
            self.transcript.partial_user_text = str(event.payload.get("text") or "")
        elif event.type == VoiceEventType.TRANSCRIPT_FINAL:
            text = str(event.payload.get("text") or "").strip()
            generation = _payload_generation(event.payload)
            if generation is not None:
                self.transcript.active_playback_generation = max(
                    self.transcript.active_playback_generation,
                    generation,
                )
            self.transcript.partial_user_text = ""
            if text:
                self.transcript.final_user_segments.append(text)
            self.state = RealtimeVoiceSessionState.ASSISTANT_PENDING
        elif event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
            generation = _payload_generation(event.payload)
            if generation is not None:
                if generation < self.transcript.active_playback_generation:
                    return
                self.transcript.active_playback_generation = generation
            self.transcript.assistant_draft = _assistant_draft_after_partial(
                self.transcript.assistant_draft,
                event.payload,
            )
            self.state = RealtimeVoiceSessionState.SPEAKING
        elif event.type == VoiceEventType.ASSISTANT_CAPTION_PARTIAL:
            generation = _payload_generation(event.payload)
            if generation is not None:
                if generation < self.transcript.active_playback_generation:
                    return
                self.transcript.active_playback_generation = generation
            self.state = RealtimeVoiceSessionState.SPEAKING
        elif event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
            generation = _payload_generation(event.payload)
            if generation is not None:
                if generation < self.transcript.active_playback_generation:
                    return
                self.transcript.active_playback_generation = generation
            self.state = RealtimeVoiceSessionState.SPEAKING
        elif event.type == VoiceEventType.ASSISTANT_AUDIO_END:
            generation = _payload_generation(event.payload)
            if generation is not None:
                if generation < self.transcript.active_playback_generation:
                    return
                self.transcript.active_playback_generation = generation
            self.state = RealtimeVoiceSessionState.SPEAKING
        elif event.type == VoiceEventType.PLAYBACK_STARTED:
            generation = _payload_generation(event.payload)
            if generation is not None:
                if generation < self.transcript.active_playback_generation:
                    return
                self.transcript.active_playback_generation = generation
            self.state = RealtimeVoiceSessionState.SPEAKING
        elif event.type == VoiceEventType.PLAYBACK_STOPPED:
            generation = _payload_generation(event.payload)
            if generation is not None and generation < self.transcript.active_playback_generation:
                return
            self.state = RealtimeVoiceSessionState.LISTENING
        elif event.type == VoiceEventType.ASSISTANT_COMMIT:
            generation = _payload_generation(event.payload)
            if generation is not None and generation < self.transcript.active_playback_generation:
                return
            if event.payload.get("interrupted") is True:
                if self.transcript.assistant_draft:
                    self.transcript.interrupted_assistant_segments.append(self.transcript.assistant_draft)
                self.transcript.assistant_draft = ""
            else:
                text = str(event.payload.get("text") or self.transcript.assistant_draft or "").strip()
                if text:
                    self.transcript.committed_assistant_segments.append(text)
                self.transcript.assistant_draft = ""
            self.state = RealtimeVoiceSessionState.LISTENING
        elif event.type == VoiceEventType.ASSISTANT_CAPTION_FINAL:
            generation = _payload_generation(event.payload)
            if generation is not None and generation < self.transcript.active_playback_generation:
                return
            self.state = RealtimeVoiceSessionState.LISTENING
        elif event.type == VoiceEventType.BARGE_IN:
            generation = _payload_generation(event.payload)
            if generation is not None:
                self.transcript.active_playback_generation = max(
                    self.transcript.active_playback_generation,
                    generation,
                )
            self.state = RealtimeVoiceSessionState.LISTENING
        elif event.type == VoiceEventType.SESSION_ERROR:
            self.state = RealtimeVoiceSessionState.CLOSING
        elif event.type == VoiceEventType.SESSION_CLOSED:
            self.state = RealtimeVoiceSessionState.CLOSED

    def _should_drop_stale_server_event(self, event: VoiceEvent) -> bool:
        if event.type not in STALE_GENERATION_EVENT_TYPES:
            return False
        generation = _payload_generation(event.payload)
        return generation is not None and generation < self.transcript.active_playback_generation

    def _annotate_server_event(self, event: VoiceEvent) -> VoiceEvent:
        session_state = self._state_after_event(event)
        metrics = self._event_metrics(event)
        if not metrics and session_state is None:
            return event
        payload = dict(event.payload)
        if session_state is not None:
            payload["session_state"] = session_state.value
        existing = payload.get("metrics")
        if metrics:
            if isinstance(existing, dict):
                payload["metrics"] = {**existing, **metrics}
            else:
                payload["metrics"] = metrics
            misses = self._quality_target_misses(payload["metrics"])
            if misses:
                payload["quality_target_misses"] = misses
                self._record_quality_target_misses(misses)
            if self._quality_target_miss_count > 0:
                payload["quality_summary"] = self._quality_summary_payload()
        return VoiceEvent(
            type=event.type,
            session_id=event.session_id,
            sequence=event.sequence,
            timestamp_ms=event.timestamp_ms,
            payload=payload,
        )

    def _state_after_event(self, event: VoiceEvent) -> Optional[RealtimeVoiceSessionState]:
        if event.type == VoiceEventType.SESSION_STARTED:
            return RealtimeVoiceSessionState.LISTENING
        if event.type == VoiceEventType.TRANSCRIPT_PARTIAL:
            return self.state
        if event.type == VoiceEventType.TRANSCRIPT_FINAL:
            return RealtimeVoiceSessionState.ASSISTANT_PENDING
        if event.type in {
            VoiceEventType.ASSISTANT_CAPTION_PARTIAL,
            VoiceEventType.ASSISTANT_TEXT_PARTIAL,
            VoiceEventType.AUDIO_OUTPUT_CHUNK,
            VoiceEventType.ASSISTANT_AUDIO_END,
            VoiceEventType.PLAYBACK_STARTED,
        }:
            return RealtimeVoiceSessionState.SPEAKING
        if event.type in {
            VoiceEventType.ASSISTANT_CAPTION_FINAL,
            VoiceEventType.ASSISTANT_COMMIT,
            VoiceEventType.BARGE_IN,
            VoiceEventType.PLAYBACK_STOPPED,
        }:
            return RealtimeVoiceSessionState.LISTENING
        if event.type == VoiceEventType.SESSION_ERROR:
            return RealtimeVoiceSessionState.CLOSING
        if event.type == VoiceEventType.SESSION_CLOSED:
            return RealtimeVoiceSessionState.CLOSED
        return None

    def _event_metrics(self, event: VoiceEvent) -> Dict[str, int]:
        now = time.monotonic()
        metrics: Dict[str, int] = {}
        if self._started_at_monotonic is not None:
            metrics["session_elapsed_ms"] = _elapsed_ms(self._started_at_monotonic, now)

        if event.type == VoiceEventType.TRANSCRIPT_PARTIAL and self._turn_audio_started_at is not None:
            metrics["audio_to_partial_transcript_ms"] = _elapsed_ms(self._turn_audio_started_at, now)
        elif event.type == VoiceEventType.TRANSCRIPT_FINAL:
            if self._turn_audio_started_at is not None:
                metrics["audio_to_final_transcript_ms"] = _elapsed_ms(self._turn_audio_started_at, now)
            if self._turn_eou_at is not None:
                metrics["eou_to_final_transcript_ms"] = _elapsed_ms(self._turn_eou_at, now)
            self._response_speech_boundary_at = self._turn_eou_at
            self._last_transcript_final_at = now
            self._turn_audio_started_at = None
            self._turn_eou_at = None
            self._turn_first_assistant_text = False
            self._turn_first_audio_output = False
        elif event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
            if self._last_transcript_final_at is not None and not self._turn_first_assistant_text:
                metrics["final_transcript_to_first_text_ms"] = _elapsed_ms(self._last_transcript_final_at, now)
                self._turn_first_assistant_text = True
        elif event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
            if self._last_transcript_final_at is not None and not self._turn_first_audio_output:
                metrics["final_transcript_to_first_audio_ms"] = _elapsed_ms(self._last_transcript_final_at, now)
                if self._response_speech_boundary_at is not None:
                    metrics["speech_boundary_to_first_audio_ms"] = _elapsed_ms(
                        self._response_speech_boundary_at,
                        now,
                    )
                    self._response_speech_boundary_at = None
                self._turn_first_audio_output = True
        elif event.type == VoiceEventType.BARGE_IN and self._last_barge_in_at is not None:
            metrics["barge_in_ack_ms"] = _elapsed_ms(self._last_barge_in_at, now)
        return metrics

    def _quality_target_misses(self, metrics: Mapping[str, Any]) -> List[dict]:
        targets = self.config.metadata.get("quality_targets_ms") if isinstance(self.config.metadata, Mapping) else {}
        if not isinstance(targets, Mapping):
            return []

        misses: List[dict] = []
        for key in QUALITY_TARGET_METRIC_KEYS:
            actual = _positive_int(metrics.get(key))
            target = _positive_int(targets.get(key))
            if actual is None or target is None or actual <= target:
                continue
            misses.append({"metric": key, "actual_ms": actual, "target_ms": target})
        return sorted(misses, key=lambda item: item["metric"])

    def _record_quality_target_misses(self, misses: List[dict]) -> None:
        self._quality_target_miss_count += len(misses)
        self._last_quality_target_miss = dict(misses[-1])

    def _quality_summary_payload(self) -> dict:
        payload = {
            "target_miss_count": self._quality_target_miss_count,
        }
        if self._last_quality_target_miss is not None:
            payload["last_target_miss"] = dict(self._last_quality_target_miss)
        return payload


def _payload_generation(payload: dict) -> Optional[int]:
    value = payload.get("playback_generation")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _assistant_draft_after_partial(current: str, payload: dict) -> str:
    delta = payload.get("delta")
    if isinstance(delta, str) and delta:
        return f"{current}{delta}" if current else delta.lstrip()

    text = payload.get("text")
    if isinstance(text, str) and text:
        return text.strip()

    return current


def _elapsed_ms(start: float, end: float) -> int:
    return max(0, int((end - start) * 1000))


def create_realtime_voice_engine(config: RealtimeVoiceSessionConfig) -> RealtimeVoiceEngine:
    if config.engine == RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE:
        return NativeS2SSidecarEngine()
    if config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
        return KameInterfaceOracleEngine()
    return TextOracleTTSEngine()
