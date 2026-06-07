"""Realtime voice session state machine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import AsyncIterator, List, Optional

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

    async def start(self) -> None:
        self.state = RealtimeVoiceSessionState.STARTING
        await self.engine.start(self.config)
        self.state = RealtimeVoiceSessionState.LISTENING

    async def receive_client_event(self, event: VoiceEvent) -> None:
        validate_client_event(event)
        if event.sequence <= self._last_client_sequence:
            raise ValueError("client event sequence must increase monotonically")
        self._last_client_sequence = event.sequence
        if event.type == VoiceEventType.BARGE_IN:
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
            self._apply_server_event(event)
            yield event

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


def _payload_generation(payload: dict) -> Optional[int]:
    value = payload.get("playback_generation")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def create_realtime_voice_engine(config: RealtimeVoiceSessionConfig) -> RealtimeVoiceEngine:
    if config.engine == RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE:
        return NativeS2SSidecarEngine()
    return TextOracleTTSEngine()
