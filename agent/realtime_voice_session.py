"""Realtime voice session state machine."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import AsyncIterator, Dict, List, Optional

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
from agent.realtime_voice_text_engine import TextOracleTTSEngine


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
        self._last_transcript_final_at: Optional[float] = None
        self._last_barge_in_at: Optional[float] = None
        self._turn_first_assistant_text = False
        self._turn_first_audio_output = False

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
        if event.type == VoiceEventType.AUDIO_INPUT_CHUNK:
            if self._turn_audio_started_at is None:
                self._turn_audio_started_at = now
            if event.payload.get("end_of_utterance") is True:
                self._turn_eou_at = now
        if event.type == VoiceEventType.BARGE_IN:
            self._last_barge_in_at = now
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
            self.transcript.assistant_draft = (
                self.transcript.assistant_draft + " " + str(event.payload.get("text") or "")
            ).strip()
            self.state = RealtimeVoiceSessionState.SPEAKING
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
        elif event.type == VoiceEventType.SESSION_CLOSED:
            self.state = RealtimeVoiceSessionState.CLOSED

    def _annotate_server_event(self, event: VoiceEvent) -> VoiceEvent:
        metrics = self._event_metrics(event)
        if not metrics:
            return event
        payload = dict(event.payload)
        existing = payload.get("metrics")
        if isinstance(existing, dict):
            payload["metrics"] = {**existing, **metrics}
        else:
            payload["metrics"] = metrics
        return VoiceEvent(
            type=event.type,
            session_id=event.session_id,
            sequence=event.sequence,
            timestamp_ms=event.timestamp_ms,
            payload=payload,
        )

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
                self._turn_first_audio_output = True
        elif event.type == VoiceEventType.BARGE_IN and self._last_barge_in_at is not None:
            metrics["barge_in_ack_ms"] = _elapsed_ms(self._last_barge_in_at, now)
        return metrics


def _payload_generation(payload: dict) -> Optional[int]:
    value = payload.get("playback_generation")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _elapsed_ms(start: float, end: float) -> int:
    return max(0, int((end - start) * 1000))


def create_realtime_voice_engine(config: RealtimeVoiceSessionConfig) -> RealtimeVoiceEngine:
    if config.engine == RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE:
        return NativeS2SSidecarEngine()
    return TextOracleTTSEngine()
