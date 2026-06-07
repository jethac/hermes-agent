"""Deepgram-compatible streaming STT bridge for Hermes realtime voice."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import urllib.parse
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
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error


@dataclass(frozen=True)
class DeepgramStreamingSTTBridgeConfig:
    """Runtime settings for a Hermes-compatible Deepgram streaming STT bridge."""

    api_key: Optional[str] = None
    auth_token: Optional[str] = None
    deepgram_url: str = "wss://api.deepgram.com/v1/listen"
    deepgram_tts_url: str = "wss://api.deepgram.com/v1/speak"
    model: str = "nova-3"
    tts_model: str = "aura-2-thalia-en"
    language: Optional[str] = None
    tts_sample_rate_hz: int = 24000
    endpointing_ms: int = 80
    connect_timeout_seconds: float = 10.0


class DeepgramStreamingSTTBridgeSession:
    """One Hermes bridge session backed by Deepgram live transcription."""

    def __init__(self, runtime: DeepgramStreamingSTTBridgeConfig):
        self.runtime = runtime
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._events: asyncio.Queue[VoiceEvent | None] = create_realtime_voice_event_queue()
        self._deepgram_ws: Any = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._sequence = 0
        self._closed = False
        self._last_input_generation: Optional[int] = None

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        if not self.runtime.api_key:
            raise RuntimeError("Deepgram streaming STT bridge requires DEEPGRAM_API_KEY")
        self.config = config
        self._deepgram_ws = await self._connect_deepgram(config)
        self._reader_task = asyncio.create_task(self._consume_deepgram_events())
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "ready",
                "provider": "deepgram",
                "model": self.runtime.model,
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
            playback_generation = _payload_int(event.payload.get("playback_generation"))
            if playback_generation is not None:
                payload["playback_generation"] = playback_generation
            await self._emit(VoiceEventType.BARGE_IN, payload)
            return
        if event.type != VoiceEventType.AUDIO_INPUT_CHUNK:
            return
        try:
            chunk = AudioChunk.from_payload(event.payload)
        except Exception:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid audio chunk"})
            return
        input_generation = _payload_int(event.payload.get("input_generation"))
        if input_generation is not None:
            self._last_input_generation = input_generation
        await self._deepgram_ws.send(chunk.data)
        if event.payload.get("end_of_utterance") is True:
            await self._deepgram_ws.send(json.dumps({"type": "Finalize"}))

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
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._deepgram_ws is not None:
            try:
                await self._deepgram_ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            await self._deepgram_ws.close()
        await self._emit(VoiceEventType.SESSION_CLOSED, {"reason": "closed"})
        await put_realtime_voice_event(self._events, None)

    async def _connect_deepgram(self, config: RealtimeVoiceSessionConfig) -> Any:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("Deepgram streaming STT bridge requires the websockets package") from exc

        url = deepgram_listen_url(self.runtime, config)
        headers = {"Authorization": f"Token {self.runtime.api_key}"}
        try:
            connect = websockets.connect(url, additional_headers=headers)
        except TypeError:
            connect = websockets.connect(url, extra_headers=headers)
        return await asyncio.wait_for(connect, timeout=max(0.1, self.runtime.connect_timeout_seconds))

    async def _consume_deepgram_events(self) -> None:
        try:
            async for raw in self._deepgram_ws:
                if not isinstance(raw, str):
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                if data.get("type") != "Results":
                    continue
                payload = deepgram_result_to_transcript_payload(
                    data,
                    input_generation=self._last_input_generation,
                )
                if not payload:
                    continue
                event_type = (
                    VoiceEventType.TRANSCRIPT_FINAL
                    if data.get("speech_final") is True or data.get("from_finalize") is True
                    else VoiceEventType.TRANSCRIPT_PARTIAL
                )
                await self._emit(event_type, payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"deepgram stream failed: {sanitize_realtime_voice_error(exc)}"},
            )

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


class DeepgramStreamingTTSBridgeSession:
    """One Hermes bridge session backed by Deepgram streaming TTS."""

    def __init__(self, runtime: DeepgramStreamingSTTBridgeConfig):
        self.runtime = runtime
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._deepgram_ws: Any = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._events: asyncio.Queue[VoiceEvent | None] = create_realtime_voice_event_queue()
        self._sequence = 0
        self._closed = False
        self._playback_generation: Optional[int] = None

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        if not self.runtime.api_key:
            raise RuntimeError("Deepgram streaming TTS bridge requires DEEPGRAM_API_KEY")
        self.config = config
        self._deepgram_ws = await self._connect_deepgram_tts()
        self._reader_task = asyncio.create_task(self._consume_deepgram_audio())
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "ready",
                "provider": "deepgram",
                "model": self.runtime.tts_model,
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
            generation = _payload_int(event.payload.get("playback_generation"))
            if generation is not None:
                self._playback_generation = generation
            with contextlib.suppress(Exception):
                await self._deepgram_ws.send(json.dumps({"type": "Clear"}))
            payload = {"reason": event.payload.get("reason") or "client"}
            if generation is not None:
                payload["playback_generation"] = generation
            await self._emit(VoiceEventType.BARGE_IN, payload)
            return
        if event.type != VoiceEventType.ASSISTANT_TEXT_PARTIAL or event.payload.get("speak") is not True:
            return
        text = str(event.payload.get("text") or "").strip()
        if not text:
            return
        generation = _payload_int(event.payload.get("playback_generation"))
        if generation is not None:
            self._playback_generation = generation
        await self._deepgram_ws.send(json.dumps({"type": "Speak", "text": text}))
        await self._deepgram_ws.send(json.dumps({"type": "Flush"}))

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
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._deepgram_ws is not None:
            with contextlib.suppress(Exception):
                await self._deepgram_ws.send(json.dumps({"type": "Close"}))
            await self._deepgram_ws.close()
        await self._emit(VoiceEventType.SESSION_CLOSED, {"reason": "closed"})
        await put_realtime_voice_event(self._events, None)

    async def _connect_deepgram_tts(self) -> Any:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("Deepgram streaming TTS bridge requires the websockets package") from exc

        url = deepgram_tts_url(self.runtime)
        headers = {"Authorization": f"Token {self.runtime.api_key}"}
        try:
            connect = websockets.connect(url, additional_headers=headers)
        except TypeError:
            connect = websockets.connect(url, extra_headers=headers)
        return await asyncio.wait_for(connect, timeout=max(0.1, self.runtime.connect_timeout_seconds))

    async def _consume_deepgram_audio(self) -> None:
        try:
            async for raw in self._deepgram_ws:
                if isinstance(raw, bytes):
                    payload = AudioChunk(
                        codec=VoiceAudioCodec.PCM16,
                        data=raw,
                        sample_rate_hz=self.runtime.tts_sample_rate_hz,
                        channels=1,
                    ).to_payload()
                    payload["mime_type"] = "audio/L16"
                    if self._playback_generation is not None:
                        payload["playback_generation"] = self._playback_generation
                    await self._emit(VoiceEventType.AUDIO_OUTPUT_CHUNK, payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"deepgram tts stream failed: {sanitize_realtime_voice_error(exc)}"},
            )

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


def deepgram_listen_url(
    runtime: DeepgramStreamingSTTBridgeConfig,
    config: RealtimeVoiceSessionConfig,
) -> str:
    query = {
        "model": runtime.model,
        "interim_results": "true",
        "punctuate": "true",
        "smart_format": "true",
        "endpointing": str(max(1, int(runtime.endpointing_ms))),
        "channels": str(max(1, int(config.channels or 1))),
    }
    if runtime.language:
        query["language"] = runtime.language
    if config.input_codec == VoiceAudioCodec.PCM16:
        query["encoding"] = "linear16"
        query["sample_rate"] = str(max(1, int(config.sample_rate_hz or 16000)))
    separator = "&" if urllib.parse.urlparse(runtime.deepgram_url).query else "?"
    return f"{runtime.deepgram_url}{separator}{urllib.parse.urlencode(query)}"


def deepgram_tts_url(runtime: DeepgramStreamingSTTBridgeConfig) -> str:
    query = {
        "model": runtime.tts_model,
        "encoding": "linear16",
        "sample_rate": str(max(1, int(runtime.tts_sample_rate_hz or 24000))),
    }
    separator = "&" if urllib.parse.urlparse(runtime.deepgram_tts_url).query else "?"
    return f"{runtime.deepgram_tts_url}{separator}{urllib.parse.urlencode(query)}"


def deepgram_result_to_transcript_payload(
    data: Mapping[str, Any],
    *,
    input_generation: Optional[int] = None,
) -> dict[str, Any]:
    channel = data.get("channel")
    alternatives = channel.get("alternatives") if isinstance(channel, Mapping) else None
    if not isinstance(alternatives, list) or not alternatives:
        return {}
    first = alternatives[0]
    if not isinstance(first, Mapping):
        return {}
    transcript = str(first.get("transcript") or "").strip()
    if not transcript:
        return {}
    payload: dict[str, Any] = {"text": transcript}
    confidence = first.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        payload["confidence"] = confidence
    language = _deepgram_language(first)
    if language:
        payload["language"] = language
    if input_generation is not None:
        payload["input_generation"] = input_generation
    return payload


def create_deepgram_streaming_stt_bridge_app(runtime: Optional[DeepgramStreamingSTTBridgeConfig] = None):
    """Create the FastAPI app for the Deepgram streaming STT bridge."""

    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Hermes Deepgram streaming STT bridge")
    runtime = runtime or deepgram_bridge_config_from_env()

    @app.get("/health")
    async def health(request: Request):
        if not _authorized(request.headers, runtime.auth_token):
            raise HTTPException(status_code=401, detail="unauthorized")
        return {
            "ok": bool(runtime.api_key),
            "kind": "streaming_stt_bridge",
            "frontend": {
                "provider": "deepgram",
                "model": runtime.model,
            },
            "capabilities": {
                "streaming_stt": bool(runtime.api_key),
                "tts": bool(runtime.api_key),
                "streaming_tts": bool(runtime.api_key),
                "native_s2s": False,
            },
        }

    @app.websocket("/v1/streaming-stt/session")
    async def streaming_stt_session(ws: WebSocket):
        if not _authorized(ws.headers, runtime.auth_token):
            await ws.close(code=1008, reason="unauthorized")
            return
        await ws.accept()
        session: Optional[DeepgramStreamingSTTBridgeSession] = None
        pump_task: Optional[asyncio.Task[None]] = None

        async def pump_events() -> None:
            assert session is not None
            async for event in session.events():
                frame = binary_audio_frame_from_event(event)
                if frame is not None:
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
                    if session is None:
                        await ws.send_json({"type": "session.error", "payload": {"error": "missing session.config"}})
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
                    session = DeepgramStreamingSTTBridgeSession(runtime)
                    try:
                        await session.start(config)
                    except Exception as exc:
                        await ws.send_json(
                            {
                                "type": "session.error",
                                "payload": {"error": sanitize_realtime_voice_error(exc)},
                            }
                        )
                        await ws.close(code=1011, reason="streaming stt unavailable")
                        return
                    pump_task = asyncio.create_task(pump_events())
                    continue
                if session is None:
                    await ws.send_json({"type": "session.error", "payload": {"error": "missing session.config"}})
                    continue
                await session.receive_event(VoiceEvent.from_wire(data))
        except WebSocketDisconnect:
            pass
        finally:
            if pump_task:
                pump_task.cancel()
            if session is not None:
                await session.close()

    @app.websocket("/v1/streaming-tts/session")
    async def streaming_tts_session(ws: WebSocket):
        if not _authorized(ws.headers, runtime.auth_token):
            await ws.close(code=1008, reason="unauthorized")
            return
        await ws.accept()
        session: Optional[DeepgramStreamingTTSBridgeSession] = None
        pump_task: Optional[asyncio.Task[None]] = None

        async def pump_events() -> None:
            assert session is not None
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
                raw = message.get("text")
                if not isinstance(raw, str):
                    await ws.send_json({"type": "session.error", "payload": {"error": "invalid websocket frame"}})
                    continue
                data = json.loads(raw)
                if data.get("type") == "session.config":
                    config = RealtimeVoiceSessionConfig.from_wire(data.get("payload") or {})
                    session = DeepgramStreamingTTSBridgeSession(runtime)
                    try:
                        await session.start(config)
                    except Exception as exc:
                        await ws.send_json(
                            {
                                "type": "session.error",
                                "payload": {"error": sanitize_realtime_voice_error(exc)},
                            }
                        )
                        await ws.close(code=1011, reason="streaming tts unavailable")
                        return
                    pump_task = asyncio.create_task(pump_events())
                    continue
                if session is None:
                    await ws.send_json({"type": "session.error", "payload": {"error": "missing session.config"}})
                    continue
                await session.receive_event(VoiceEvent.from_wire(data))
        except WebSocketDisconnect:
            pass
        finally:
            if pump_task:
                pump_task.cancel()
            if session is not None:
                await session.close()

    return app


def deepgram_bridge_config_from_env() -> DeepgramStreamingSTTBridgeConfig:
    auth_token_env = os.environ.get("HERMES_DEEPGRAM_BRIDGE_TOKEN_ENV") or "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    return DeepgramStreamingSTTBridgeConfig(
        api_key=os.environ.get("DEEPGRAM_API_KEY") or os.environ.get("HERMES_DEEPGRAM_API_KEY") or None,
        auth_token=os.environ.get(auth_token_env) or None,
        deepgram_url=os.environ.get("HERMES_DEEPGRAM_LISTEN_URL") or "wss://api.deepgram.com/v1/listen",
        deepgram_tts_url=os.environ.get("HERMES_DEEPGRAM_TTS_URL") or "wss://api.deepgram.com/v1/speak",
        model=os.environ.get("HERMES_DEEPGRAM_MODEL") or "nova-3",
        tts_model=os.environ.get("HERMES_DEEPGRAM_TTS_MODEL") or "aura-2-thalia-en",
        language=os.environ.get("HERMES_DEEPGRAM_LANGUAGE") or None,
        tts_sample_rate_hz=int(os.environ.get("HERMES_DEEPGRAM_TTS_SAMPLE_RATE_HZ") or 24000),
        endpointing_ms=int(os.environ.get("HERMES_DEEPGRAM_ENDPOINTING_MS") or 80),
        connect_timeout_seconds=float(os.environ.get("HERMES_DEEPGRAM_CONNECT_TIMEOUT_SECONDS") or 10),
    )


def deepgram_bridge_prerequisite_issues(
    runtime: Optional[DeepgramStreamingSTTBridgeConfig] = None,
    *,
    require_auth_token: bool = False,
    module_available: Optional[Any] = None,
) -> list[str]:
    runtime = runtime or deepgram_bridge_config_from_env()
    available = module_available or _module_available
    issues: list[str] = []
    if not runtime.api_key:
        issues.append("DEEPGRAM_API_KEY or HERMES_DEEPGRAM_API_KEY is required")
    if not available("websockets"):
        issues.append("Python package 'websockets' is required")
    if require_auth_token and not runtime.auth_token:
        issues.append("HERMES_STREAMING_STT_BRIDGE_TOKEN is required in strict mode")
    return issues


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _authorized(headers: Mapping[str, str], token: Optional[str]) -> bool:
    if not token:
        return True
    return headers.get("authorization") == f"Bearer {token}"


def _deepgram_language(alternative: Mapping[str, Any]) -> str:
    languages = alternative.get("languages")
    if isinstance(languages, list) and languages:
        return _clean_language(languages[0])
    words = alternative.get("words")
    if isinstance(words, list):
        for word in words:
            if isinstance(word, Mapping):
                language = _clean_language(word.get("language"))
                if language:
                    return language
    return ""


def _clean_language(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not value or len(value) > 64:
        return ""
    if not all(ch.isalnum() or ch in {"-", "_", "."} for ch in value):
        return ""
    return value


def _payload_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
