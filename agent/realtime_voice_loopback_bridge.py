"""Local loopback streaming voice bridge for Hermes realtime validation."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping, Optional

from starlette.requests import Request
from starlette.websockets import WebSocket, WebSocketDisconnect

from agent.realtime_voice import (
    AudioChunk,
    RealtimeVoiceSessionConfig,
    VoiceAudioCodec,
    VoiceEvent,
    VoiceEventType,
    binary_audio_frame_from_event,
    create_realtime_voice_event_queue,
    event_from_binary_audio_frame,
    put_realtime_voice_event,
    transcript_metadata_from_payload,
)


@dataclass(frozen=True)
class LoopbackStreamingBridgeConfig:
    """Runtime settings for the deterministic local streaming bridge."""

    auth_token: Optional[str] = None
    transcript: str = "loopback transcript"
    partial_transcript: str = "loopback"
    sample_rate_hz: int = 16000
    channels: int = 1
    output_languages: tuple[str, ...] = ("en", "ja")
    input_languages: tuple[str, ...] = ("en", "ja")


class LoopbackStreamingSTTBridgeSession:
    def __init__(self, runtime: LoopbackStreamingBridgeConfig):
        self.runtime = runtime
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._events: asyncio.Queue[VoiceEvent | None] = create_realtime_voice_event_queue()
        self._sequence = 0
        self._closed = False
        self._partial_emitted = False

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        self.config = config
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "ready",
                "provider": "loopback",
                "model": "loopback-streaming-stt",
                "streaming_stt": True,
            },
        )

    async def receive_event(self, event: VoiceEvent) -> None:
        if self._closed:
            return
        if event.type == VoiceEventType.SESSION_CLOSED:
            await self.close()
            return
        if event.type == VoiceEventType.BARGE_IN:
            payload = {"reason": event.payload.get("reason") or "client"}
            if "playback_generation" in event.payload:
                payload["playback_generation"] = event.payload["playback_generation"]
            await self._emit(VoiceEventType.BARGE_IN, payload)
            return
        if event.type != VoiceEventType.AUDIO_INPUT_CHUNK:
            return
        try:
            AudioChunk.from_payload(event.payload)
        except Exception:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid audio chunk"})
            return
        metadata = transcript_metadata_from_payload(event.payload)
        generation = event.payload.get("input_generation")
        if not self._partial_emitted:
            payload: dict[str, Any] = {
                "text": self.runtime.partial_transcript,
                "stability": 0.6,
                **metadata,
            }
            if generation is not None:
                payload["input_generation"] = generation
            await self._emit(VoiceEventType.TRANSCRIPT_PARTIAL, payload)
            self._partial_emitted = True
        if event.payload.get("end_of_utterance") is True:
            payload = {
                "text": self.runtime.transcript,
                "confidence": 1.0,
                **metadata,
            }
            if generation is not None:
                payload["input_generation"] = generation
            await self._emit(VoiceEventType.TRANSCRIPT_FINAL, payload)

    async def events(self) -> AsyncIterator[VoiceEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._emit(VoiceEventType.SESSION_CLOSED, {"reason": "closed"})
        await put_realtime_voice_event(self._events, None)

    async def _emit(self, event_type: VoiceEventType, payload: Mapping[str, Any]) -> None:
        if self.config is None:
            return
        if self._closed and event_type != VoiceEventType.SESSION_CLOSED:
            return
        self._sequence += 1
        await put_realtime_voice_event(
            self._events,
            VoiceEvent(
                type=event_type,
                session_id=self.config.session_id,
                sequence=self._sequence,
                payload=dict(payload),
            ),
        )


class LoopbackStreamingTTSBridgeSession:
    def __init__(self, runtime: LoopbackStreamingBridgeConfig):
        self.runtime = runtime
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._events: asyncio.Queue[VoiceEvent | None] = create_realtime_voice_event_queue()
        self._sequence = 0
        self._closed = False

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        self.config = config
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "ready",
                "provider": "loopback",
                "model": "loopback-streaming-tts",
                "streaming_tts": True,
            },
        )

    async def receive_event(self, event: VoiceEvent) -> None:
        if self._closed:
            return
        if event.type == VoiceEventType.SESSION_CLOSED:
            await self.close()
            return
        if event.type == VoiceEventType.BARGE_IN:
            payload = {"reason": event.payload.get("reason") or "client"}
            if "playback_generation" in event.payload:
                payload["playback_generation"] = event.payload["playback_generation"]
            await self._emit(VoiceEventType.BARGE_IN, payload)
            return
        if event.type != VoiceEventType.ASSISTANT_TEXT_PARTIAL or event.payload.get("speak") is not True:
            return
        text = str(event.payload.get("text") or "").strip()
        if not text:
            return
        frame = b"\x00" * int(self.runtime.sample_rate_hz * self.runtime.channels * 2 * 0.02)
        payload = AudioChunk(
            codec=VoiceAudioCodec.PCM16,
            data=frame,
            sample_rate_hz=self.runtime.sample_rate_hz,
            channels=self.runtime.channels,
        ).to_payload()
        if "playback_generation" in event.payload:
            payload["playback_generation"] = event.payload["playback_generation"]
        payload["metrics"] = {"streaming_tts_ms": 0, "loopback": True}
        await self._emit(
            VoiceEventType.PLAYBACK_STARTED,
            {"playback_generation": payload.get("playback_generation")} if "playback_generation" in payload else {},
        )
        await self._emit(VoiceEventType.AUDIO_OUTPUT_CHUNK, payload)
        await self._emit(
            VoiceEventType.PLAYBACK_STOPPED,
            {"playback_generation": payload.get("playback_generation")} if "playback_generation" in payload else {},
        )
        await self._emit(
            VoiceEventType.ASSISTANT_AUDIO_END,
            {"playback_generation": payload.get("playback_generation")} if "playback_generation" in payload else {},
        )

    async def events(self) -> AsyncIterator[VoiceEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._emit(VoiceEventType.SESSION_CLOSED, {"reason": "closed"})
        await put_realtime_voice_event(self._events, None)

    async def _emit(self, event_type: VoiceEventType, payload: Mapping[str, Any]) -> None:
        if self.config is None:
            return
        if self._closed and event_type != VoiceEventType.SESSION_CLOSED:
            return
        self._sequence += 1
        await put_realtime_voice_event(
            self._events,
            VoiceEvent(
                type=event_type,
                session_id=self.config.session_id,
                sequence=self._sequence,
                payload=dict(payload),
            ),
        )


def create_loopback_streaming_bridge_app(runtime: Optional[LoopbackStreamingBridgeConfig] = None):
    """Create a local streaming STT/TTS bridge app for protocol validation."""

    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Hermes loopback streaming voice bridge")
    runtime = runtime or loopback_bridge_config_from_env()

    @app.get("/health")
    async def health(request: Request):
        if not _authorized(request.headers, runtime.auth_token):
            raise HTTPException(status_code=401, detail="unauthorized")
        return {
            "ok": True,
            "kind": "streaming_loopback_bridge",
            "frontend": {
                "provider": "loopback",
                "model": "loopback-streaming-stt",
                "tts_model": "loopback-streaming-tts",
                "languages": list(runtime.input_languages),
                "tts_model_languages": list(runtime.output_languages),
            },
            "capabilities": {
                "streaming_stt": True,
                "tts": True,
                "streaming_tts": True,
                "native_s2s": False,
                "input_languages": list(runtime.input_languages),
                "output_languages": list(runtime.output_languages),
            },
        }

    @app.websocket("/v1/streaming-stt/session")
    async def streaming_stt_session(ws: WebSocket):
        if not _authorized(ws.headers, runtime.auth_token):
            await ws.close(code=1008, reason="unauthorized")
            return
        await _run_bridge_session(ws, LoopbackStreamingSTTBridgeSession(runtime), binary_input=True)

    @app.websocket("/v1/streaming-tts/session")
    async def streaming_tts_session(ws: WebSocket):
        if not _authorized(ws.headers, runtime.auth_token):
            await ws.close(code=1008, reason="unauthorized")
            return
        await _run_bridge_session(ws, LoopbackStreamingTTSBridgeSession(runtime), binary_input=False)

    return app


async def _run_bridge_session(ws: WebSocket, session: Any, *, binary_input: bool) -> None:
    await ws.accept()
    pump_task: Optional[asyncio.Task[None]] = None

    async def pump_events() -> None:
        async for event in session.events():
            frame = binary_audio_frame_from_event(event)
            if frame is not None and event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
                await ws.send_bytes(frame)
                continue
            await ws.send_json(event.to_wire())

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(code=int(message.get("code") or 1000))
            frame = message.get("bytes")
            if isinstance(frame, bytes):
                if not binary_input:
                    await ws.send_json({"type": "session.error", "payload": {"error": "invalid binary audio frame"}})
                    continue
                try:
                    event = event_from_binary_audio_frame(frame, expected_type=VoiceEventType.AUDIO_INPUT_CHUNK)
                except Exception:
                    await ws.send_json({"type": "session.error", "payload": {"error": "invalid binary audio frame"}})
                    continue
                await session.receive_event(event)
                continue
            raw = message.get("text")
            if not isinstance(raw, str):
                await ws.send_json({"type": "session.error", "payload": {"error": "invalid websocket frame"}})
                continue
            data = json.loads(raw)
            if data.get("type") == "session.config":
                config = RealtimeVoiceSessionConfig.from_wire(data.get("payload") or {})
                await session.start(config)
                pump_task = asyncio.create_task(pump_events())
                continue
            if session.config is None:
                await ws.send_json({"type": "session.error", "payload": {"error": "missing session.config"}})
                continue
            await session.receive_event(VoiceEvent.from_wire(data))
    except WebSocketDisconnect:
        pass
    finally:
        if pump_task:
            pump_task.cancel()
        await session.close()


def loopback_bridge_config_from_env() -> LoopbackStreamingBridgeConfig:
    token_env = os.environ.get("HERMES_LOOPBACK_BRIDGE_TOKEN_ENV") or "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    return LoopbackStreamingBridgeConfig(
        auth_token=os.environ.get(token_env) or None,
        transcript=os.environ.get("HERMES_LOOPBACK_TRANSCRIPT") or "loopback transcript",
        partial_transcript=os.environ.get("HERMES_LOOPBACK_PARTIAL_TRANSCRIPT") or "loopback",
        output_languages=tuple(_parse_languages(os.environ.get("HERMES_LOOPBACK_OUTPUT_LANGUAGES") or "en,ja")),
        input_languages=tuple(_parse_languages(os.environ.get("HERMES_LOOPBACK_INPUT_LANGUAGES") or "en,ja")),
    )


def _authorized(headers: Mapping[str, str], token: Optional[str]) -> bool:
    if not token:
        return True
    return headers.get("authorization") == f"Bearer {token}"


def _parse_languages(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").replace(" ", ",").split(",") if part.strip()]
