"""Cartesia streaming STT/TTS bridge for Hermes realtime voice."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Optional, Sequence

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
class CartesiaRealtimeBridgeConfig:
    """Runtime settings for a Hermes-compatible Cartesia realtime bridge."""

    api_key: Optional[str] = None
    auth_token: Optional[str] = None
    stt_url: str = "wss://api.cartesia.ai/stt/websocket"
    tts_url: str = "wss://api.cartesia.ai/tts/websocket"
    api_version: str = "2026-03-01"
    model: str = "ink-2"
    tts_model: str = "sonic-3.5"
    voice_id: str = ""
    language: Optional[str] = "en"
    output_languages: tuple[str, ...] = ()
    tts_model_by_language: Mapping[str, str] = field(default_factory=dict)
    tts_voice_by_language: Mapping[str, str] = field(default_factory=dict)
    stt_sample_rate_hz: int = 16000
    tts_sample_rate_hz: int = 24000
    connect_timeout_seconds: float = 10.0


class CartesiaStreamingSTTBridgeSession:
    """One Hermes bridge session backed by Cartesia realtime STT."""

    def __init__(self, runtime: CartesiaRealtimeBridgeConfig):
        self.runtime = runtime
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._events: asyncio.Queue[VoiceEvent | None] = create_realtime_voice_event_queue()
        self._cartesia_ws: Any = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._sequence = 0
        self._closed = False
        self._last_input_generation: Optional[int] = None

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        if not self.runtime.api_key:
            raise RuntimeError("Cartesia realtime bridge requires CARTESIA_API_KEY")
        self.config = config
        self._cartesia_ws = await self._connect_cartesia(config)
        self._reader_task = asyncio.create_task(self._consume_cartesia_events())
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "ready",
                "provider": "cartesia",
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
        try:
            audio = cartesia_stt_audio_bytes(
                chunk,
                self.config,
                target_sample_rate_hz=self.runtime.stt_sample_rate_hz,
                target_channels=1,
            )
        except Exception as exc:
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"cartesia stt audio conversion failed: {sanitize_realtime_voice_error(exc)}"},
            )
            return
        if audio:
            await self._cartesia_ws.send(audio)
        if event.payload.get("end_of_utterance") is True:
            await self._cartesia_ws.send("finalize")

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
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        if self._cartesia_ws is not None:
            with contextlib.suppress(Exception):
                await self._cartesia_ws.send("close")
            with contextlib.suppress(Exception):
                await self._cartesia_ws.close()
        await self._emit(VoiceEventType.SESSION_CLOSED, {"reason": "closed"})
        await put_realtime_voice_event(self._events, None)

    async def _connect_cartesia(self, config: RealtimeVoiceSessionConfig) -> Any:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("Cartesia realtime bridge requires the websockets package") from exc

        url = cartesia_stt_url(self.runtime, config)
        headers = {"X-API-Key": self.runtime.api_key}
        try:
            connect = websockets.connect(url, additional_headers=headers)
        except TypeError:
            connect = websockets.connect(url, extra_headers=headers)
        return await asyncio.wait_for(connect, timeout=max(0.1, self.runtime.connect_timeout_seconds))

    async def _consume_cartesia_events(self) -> None:
        try:
            async for raw in self._cartesia_ws:
                if not isinstance(raw, str):
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                event_type, payload = cartesia_stt_message_to_transcript_payload(
                    data,
                    input_generation=self._last_input_generation,
                )
                if event_type is not None:
                    await self._emit(event_type, payload)
                    continue
                error = cartesia_error_from_message(data)
                if error:
                    await self._emit(VoiceEventType.SESSION_ERROR, {"error": error})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"cartesia stt stream failed: {sanitize_realtime_voice_error(exc)}"},
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


class CartesiaStreamingTTSBridgeSession:
    """One Hermes bridge session backed by Cartesia realtime TTS."""

    def __init__(self, runtime: CartesiaRealtimeBridgeConfig):
        self.runtime = runtime
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._cartesia_ws: Any = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._events: asyncio.Queue[VoiceEvent | None] = create_realtime_voice_event_queue()
        self._sequence = 0
        self._closed = False
        self._playback_generation: Optional[int] = None
        self._playback_active = False
        self._playback_started_generation: Optional[int] = None
        self._context_id = ""

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        if not self.runtime.api_key:
            raise RuntimeError("Cartesia realtime bridge requires CARTESIA_API_KEY")
        if not self.runtime.voice_id:
            raise RuntimeError("Cartesia realtime TTS bridge requires CARTESIA_VOICE_ID")
        self.config = config
        self._context_id = f"{config.session_id}-tts-0"
        self._cartesia_ws = await self._connect_cartesia_tts()
        self._reader_task = asyncio.create_task(self._consume_cartesia_audio())
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "ready",
                "provider": "cartesia",
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
                await self._cartesia_ws.send(json.dumps({"context_id": self._context_id, "cancel": True}))
            self._context_id = self._next_context_id()
            payload = {"reason": event.payload.get("reason") or "client"}
            if generation is not None:
                payload["playback_generation"] = generation
            await self._emit_playback_stopped(generation)
            await self._emit(VoiceEventType.BARGE_IN, payload)
            return
        if event.type != VoiceEventType.ASSISTANT_TEXT_PARTIAL or event.payload.get("speak") is not True:
            return
        text = str(event.payload.get("text") or "").strip()
        if not text:
            return
        generation = _payload_int(event.payload.get("playback_generation"))
        if generation is not None and generation != self._playback_generation:
            if self._playback_started_generation not in (None, generation):
                await self._emit_playback_stopped(self._playback_started_generation)
            self._playback_generation = generation
            self._context_id = self._next_context_id()
        await self._cartesia_ws.send(
            json.dumps(cartesia_tts_generation_message(self.runtime, event.payload, text, self._context_id))
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
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        if self._cartesia_ws is not None:
            with contextlib.suppress(Exception):
                await self._cartesia_ws.close()
        await self._emit(VoiceEventType.SESSION_CLOSED, {"reason": "closed"})
        await put_realtime_voice_event(self._events, None)

    async def _connect_cartesia_tts(self) -> Any:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("Cartesia realtime bridge requires the websockets package") from exc

        url = cartesia_tts_url(self.runtime)
        headers = {"X-API-Key": self.runtime.api_key}
        try:
            connect = websockets.connect(url, additional_headers=headers)
        except TypeError:
            connect = websockets.connect(url, extra_headers=headers)
        return await asyncio.wait_for(connect, timeout=max(0.1, self.runtime.connect_timeout_seconds))

    async def _consume_cartesia_audio(self) -> None:
        try:
            async for raw in self._cartesia_ws:
                if not isinstance(raw, str):
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                if data.get("type") == "error":
                    await self._emit(VoiceEventType.SESSION_ERROR, {"error": cartesia_error_from_message(data)})
                    continue
                if str(data.get("type") or "").lower() in {"done", "complete", "finished"}:
                    await self._emit_playback_stopped(self._playback_generation)
                    continue
                audio = data.get("data")
                if data.get("type") != "chunk" or not isinstance(audio, str) or not audio:
                    continue
                try:
                    audio_bytes = base64.b64decode(audio)
                except Exception:
                    continue
                await self._emit_playback_started(self._playback_generation)
                payload = AudioChunk(
                    codec=VoiceAudioCodec.PCM16,
                    data=audio_bytes,
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
                {"error": f"cartesia tts stream failed: {sanitize_realtime_voice_error(exc)}"},
            )

    async def _emit_playback_started(self, generation: Optional[int]) -> None:
        if self._playback_active and self._playback_started_generation == generation:
            return
        if self._playback_active:
            await self._emit_playback_stopped(self._playback_started_generation)
        self._playback_active = True
        self._playback_started_generation = generation
        payload = {"playback_generation": generation} if generation is not None else {}
        await self._emit(VoiceEventType.PLAYBACK_STARTED, payload)

    async def _emit_playback_stopped(self, generation: Optional[int]) -> None:
        if not self._playback_active:
            return
        payload_generation = generation if generation is not None else self._playback_started_generation
        payload = {"playback_generation": payload_generation} if payload_generation is not None else {}
        self._playback_active = False
        self._playback_started_generation = None
        await self._emit(VoiceEventType.PLAYBACK_STOPPED, payload)
        await self._emit(VoiceEventType.ASSISTANT_AUDIO_END, payload)

    def _next_context_id(self) -> str:
        if self.config is None:
            return "cartesia-tts"
        generation = self._playback_generation if self._playback_generation is not None else self._sequence
        return f"{self.config.session_id}-tts-{generation}"

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


def cartesia_stt_url(runtime: CartesiaRealtimeBridgeConfig, config: RealtimeVoiceSessionConfig) -> str:
    sample_rate = int(runtime.stt_sample_rate_hz or config.sample_rate_hz or 16000)
    query = {
        "model": runtime.model,
        "encoding": "pcm_s16le",
        "sample_rate": str(max(1, sample_rate)),
        "cartesia_version": runtime.api_version,
    }
    separator = "&" if urllib.parse.urlparse(runtime.stt_url).query else "?"
    return f"{runtime.stt_url}{separator}{urllib.parse.urlencode(query)}"


def cartesia_tts_url(runtime: CartesiaRealtimeBridgeConfig) -> str:
    query = {"cartesia_version": runtime.api_version}
    separator = "&" if urllib.parse.urlparse(runtime.tts_url).query else "?"
    return f"{runtime.tts_url}{separator}{urllib.parse.urlencode(query)}"


def cartesia_tts_generation_message(
    runtime: CartesiaRealtimeBridgeConfig,
    payload: Mapping[str, Any],
    text: str,
    context_id: str,
) -> dict[str, Any]:
    language = _cartesia_tts_language(payload, runtime)
    return {
        "model_id": cartesia_tts_model_for_payload(runtime, payload),
        "transcript": text,
        "voice": {"mode": "id", "id": cartesia_tts_voice_for_payload(runtime, payload)},
        "language": language,
        "context_id": context_id,
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": max(1, int(runtime.tts_sample_rate_hz or 24000)),
        },
        "continue": False,
    }


def cartesia_stt_message_to_transcript_payload(
    data: Mapping[str, Any],
    *,
    input_generation: Optional[int] = None,
) -> tuple[Optional[VoiceEventType], dict[str, Any]]:
    message_type = str(data.get("type") or "")
    text = str(data.get("text") or data.get("transcript") or "").strip()
    if not text:
        return None, {}
    payload: dict[str, Any] = {"text": text}
    language = _clean_language(data.get("language"))
    if language:
        payload["language"] = language
    if input_generation is not None:
        payload["input_generation"] = input_generation
    if message_type in {"turn.update", "transcript"} and data.get("is_final") is not True:
        return VoiceEventType.TRANSCRIPT_PARTIAL, payload
    if message_type in {"turn.end", "turn.eager_end", "transcript"} or data.get("is_final") is True:
        return VoiceEventType.TRANSCRIPT_FINAL, payload
    return None, {}


def cartesia_stt_audio_bytes(
    chunk: AudioChunk,
    config: Optional[RealtimeVoiceSessionConfig] = None,
    *,
    target_sample_rate_hz: Optional[int] = None,
    target_channels: int = 1,
) -> bytes:
    source_sample_rate_hz = max(
        1,
        int(chunk.sample_rate_hz or (config.sample_rate_hz if config is not None else 16000) or 16000),
    )
    source_channels = max(1, int(chunk.channels or (config.channels if config is not None else 1) or 1))
    output_sample_rate_hz = max(1, int(target_sample_rate_hz or source_sample_rate_hz or 16000))
    output_channels = max(1, int(target_channels or 1))
    if chunk.codec == VoiceAudioCodec.PCM16:
        if source_sample_rate_hz == output_sample_rate_hz and source_channels == output_channels:
            return chunk.data
        return _ffmpeg_raw_pcm16le(
            chunk.data,
            input_sample_rate_hz=source_sample_rate_hz,
            input_channels=source_channels,
            output_sample_rate_hz=output_sample_rate_hz,
            output_channels=output_channels,
        )
    return _ffmpeg_to_pcm16le(
        chunk.data,
        codec=chunk.codec,
        sample_rate_hz=output_sample_rate_hz,
        channels=output_channels,
    )


def _ffmpeg_raw_pcm16le(
    audio: bytes,
    *,
    input_sample_rate_hz: int,
    input_channels: int,
    output_sample_rate_hz: int,
    output_channels: int,
) -> bytes:
    if not audio:
        return b""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to resample PCM16 audio for Cartesia STT")
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(input_sample_rate_hz),
            "-ac",
            str(input_channels),
            "-i",
            "pipe:0",
            "-ar",
            str(output_sample_rate_hz),
            "-ac",
            str(output_channels),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ],
        input=audio,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or "ffmpeg PCM resampling failed")
    if not completed.stdout:
        raise RuntimeError("ffmpeg produced no PCM audio")
    return completed.stdout


def _ffmpeg_to_pcm16le(
    audio: bytes,
    *,
    codec: VoiceAudioCodec,
    sample_rate_hz: int,
    channels: int,
) -> bytes:
    if not audio:
        return b""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to convert compressed audio to PCM16 for Cartesia STT")
    suffix = ".webm" if codec == VoiceAudioCodec.WEBM_OPUS else ".opus"
    with tempfile.TemporaryDirectory(prefix="hermes-cartesia-stt-") as tmp:
        src = Path(tmp) / f"input{suffix}"
        src.write_bytes(audio)
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(src),
                "-ac",
                str(channels),
                "-ar",
                str(sample_rate_hz),
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or "ffmpeg audio conversion failed")
    if not completed.stdout:
        raise RuntimeError("ffmpeg produced no PCM audio")
    return completed.stdout


def create_cartesia_realtime_bridge_app(runtime: Optional[CartesiaRealtimeBridgeConfig] = None):
    """Create the FastAPI app for the Cartesia realtime bridge."""

    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Hermes Cartesia realtime bridge")
    runtime = runtime or cartesia_bridge_config_from_env()

    @app.get("/health")
    async def health(request: Request):
        if not _authorized(request.headers, runtime.auth_token):
            raise HTTPException(status_code=401, detail="unauthorized")
        ready = bool(runtime.api_key) and _module_available("websockets")
        tts_ready = ready and bool(runtime.voice_id)
        return {
            "ok": ready and tts_ready,
            "kind": "streaming_stt_bridge",
            "frontend": {
                "provider": "cartesia",
                "model": runtime.model,
                "language": runtime.language or "en",
                "tts_model": runtime.tts_model,
                "voice_id": "configured" if runtime.voice_id else "",
                "tts_model_languages": sorted(runtime.tts_model_by_language.keys()),
                "tts_voice_languages": sorted(runtime.tts_voice_by_language.keys()),
            },
            "capabilities": {
                "streaming_stt": ready,
                "tts": tts_ready,
                "streaming_tts": tts_ready,
                "native_s2s": False,
                "input_languages": cartesia_input_languages(runtime),
                "output_languages": cartesia_tts_output_languages(runtime),
            },
        }

    @app.websocket("/v1/streaming-stt/session")
    async def streaming_stt_session(ws: WebSocket):
        if not _authorized(ws.headers, runtime.auth_token):
            await ws.close(code=1008, reason="unauthorized")
            return
        await ws.accept()
        session: Optional[CartesiaStreamingSTTBridgeSession] = None
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
                    session = CartesiaStreamingSTTBridgeSession(runtime)
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
        session: Optional[CartesiaStreamingTTSBridgeSession] = None
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
                    session = CartesiaStreamingTTSBridgeSession(runtime)
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


def cartesia_bridge_config_from_env() -> CartesiaRealtimeBridgeConfig:
    auth_token_env = os.environ.get("HERMES_CARTESIA_BRIDGE_TOKEN_ENV") or "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    return CartesiaRealtimeBridgeConfig(
        api_key=os.environ.get("CARTESIA_API_KEY") or os.environ.get("HERMES_CARTESIA_API_KEY") or None,
        auth_token=os.environ.get(auth_token_env) or None,
        stt_url=os.environ.get("HERMES_CARTESIA_STT_URL") or "wss://api.cartesia.ai/stt/websocket",
        tts_url=os.environ.get("HERMES_CARTESIA_TTS_URL") or "wss://api.cartesia.ai/tts/websocket",
        api_version=os.environ.get("HERMES_CARTESIA_API_VERSION") or "2026-03-01",
        model=os.environ.get("HERMES_CARTESIA_STT_MODEL") or "ink-2",
        tts_model=os.environ.get("HERMES_CARTESIA_TTS_MODEL") or "sonic-3.5",
        voice_id=os.environ.get("CARTESIA_VOICE_ID") or os.environ.get("HERMES_CARTESIA_VOICE_ID") or "",
        language=os.environ.get("HERMES_CARTESIA_LANGUAGE") or "en",
        output_languages=tuple(_parse_language_list(os.environ.get("HERMES_CARTESIA_OUTPUT_LANGUAGES") or "")),
        tts_model_by_language=_parse_mapping(os.environ.get("HERMES_CARTESIA_TTS_MODEL_BY_LANGUAGE") or ""),
        tts_voice_by_language=_parse_mapping(os.environ.get("HERMES_CARTESIA_TTS_VOICE_BY_LANGUAGE") or ""),
        stt_sample_rate_hz=int(os.environ.get("HERMES_CARTESIA_STT_SAMPLE_RATE_HZ") or 16000),
        tts_sample_rate_hz=int(os.environ.get("HERMES_CARTESIA_TTS_SAMPLE_RATE_HZ") or 24000),
        connect_timeout_seconds=float(os.environ.get("HERMES_CARTESIA_CONNECT_TIMEOUT_SECONDS") or 10),
    )


def cartesia_bridge_prerequisite_issues(
    runtime: Optional[CartesiaRealtimeBridgeConfig] = None,
    *,
    require_auth_token: bool = False,
    required_input_languages: Sequence[str] = (),
    required_output_languages: Sequence[str] = (),
    module_available: Optional[Any] = None,
) -> list[str]:
    runtime = runtime or cartesia_bridge_config_from_env()
    available = module_available or _module_available
    issues: list[str] = []
    if not runtime.api_key:
        issues.append("CARTESIA_API_KEY or HERMES_CARTESIA_API_KEY is required")
    if not runtime.voice_id:
        issues.append("CARTESIA_VOICE_ID or HERMES_CARTESIA_VOICE_ID is required for streaming TTS")
    if not available("websockets"):
        issues.append(
            "Python package 'websockets==15.0.1' is required; "
            "install with `python -m pip install 'hermes-agent[voice]'`"
        )
    if require_auth_token and not runtime.auth_token:
        issues.append("HERMES_STREAMING_STT_BRIDGE_TOKEN is required in strict mode")
    configured_input_languages = set(cartesia_input_languages(runtime))
    missing_input_languages = [
        language
        for language in (_primary_language(value) for value in required_input_languages)
        if language and language not in configured_input_languages
    ]
    if missing_input_languages:
        configured = ",".join(sorted(configured_input_languages)) or "none"
        required = ",".join(missing_input_languages)
        issues.append(
            "Cartesia realtime STT input routing is missing required language(s) "
            f"{required}; configured input language(s): {configured}"
        )
    configured_output_languages = set(cartesia_tts_output_languages(runtime))
    missing_output_languages = [
        language
        for language in (_primary_language(value) for value in required_output_languages)
        if language and language not in configured_output_languages
    ]
    if missing_output_languages:
        configured = ",".join(sorted(configured_output_languages)) or "none"
        required = ",".join(missing_output_languages)
        issues.append(
            "Cartesia realtime TTS output routing is missing required language(s) "
            f"{required}; configured output language(s): {configured}"
        )
    return issues


def cartesia_input_languages(runtime: CartesiaRealtimeBridgeConfig) -> list[str]:
    language = _primary_language(runtime.language)
    return [language or "en"]


def cartesia_tts_output_languages(runtime: CartesiaRealtimeBridgeConfig) -> list[str]:
    languages = {_primary_language(language) for language in runtime.output_languages}
    languages.update(_primary_language(language) for language in dict(runtime.tts_model_by_language or {}).keys())
    languages.update(_primary_language(language) for language in dict(runtime.tts_voice_by_language or {}).keys())
    languages.discard("")
    return sorted(languages) if languages else ["en", "ja"]


def cartesia_tts_model_for_payload(runtime: CartesiaRealtimeBridgeConfig, payload: Mapping[str, Any]) -> str:
    return _mapped_value(runtime.tts_model_by_language, payload) or str(runtime.tts_model or "").strip() or "sonic-3.5"


def cartesia_tts_voice_for_payload(runtime: CartesiaRealtimeBridgeConfig, payload: Mapping[str, Any]) -> str:
    return _mapped_value(runtime.tts_voice_by_language, payload) or str(runtime.voice_id or "").strip()


def _cartesia_tts_language(payload: Mapping[str, Any], runtime: CartesiaRealtimeBridgeConfig) -> str:
    for raw in (payload.get("locale"), payload.get("language")):
        language = _primary_language(raw)
        if language:
            return language
    return _primary_language(runtime.language) or "en"


def _mapped_value(mapping: Mapping[str, str], payload: Mapping[str, Any]) -> str:
    normalized = {
        _clean_language(key).lower(): str(value).strip()
        for key, value in dict(mapping or {}).items()
        if _clean_language(key) and str(value).strip()
    }
    for raw in (payload.get("locale"), payload.get("language")):
        language = _clean_language(raw).lower()
        if not language:
            continue
        if language in normalized:
            return normalized[language]
        primary = language.split("-", 1)[0]
        if primary in normalized:
            return normalized[primary]
    return ""


def cartesia_error_from_message(data: Mapping[str, Any]) -> str:
    if data.get("type") != "error" and not data.get("error_code"):
        return ""
    message = data.get("message") or data.get("title") or data.get("error_code") or "cartesia error"
    return sanitize_realtime_voice_error(message)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _authorized(headers: Mapping[str, str], token: Optional[str]) -> bool:
    if not token:
        return True
    return headers.get("authorization") == f"Bearer {token}"


def _clean_language(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip().replace("_", "-")
    if not value or len(value) > 64:
        return ""
    if not all(ch.isalnum() or ch in {"-", "_", "."} for ch in value):
        return ""
    return value


def _primary_language(value: Any) -> str:
    language = _clean_language(value).lower()
    return language.split("-", 1)[0] if language else ""


def _parse_language_list(value: str) -> list[str]:
    result: list[str] = []
    for item in str(value or "").split(","):
        language = _primary_language(item)
        if language and language not in result:
            result.append(language)
    return result


def _parse_mapping(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in str(value or "").split(","):
        text = item.strip()
        if not text:
            continue
        if "=" in text:
            raw_key, raw_value = text.split("=", 1)
        elif ":" in text:
            raw_key, raw_value = text.split(":", 1)
        else:
            continue
        key = _clean_language(raw_key).lower()
        mapped = str(raw_value or "").strip()
        if key and mapped:
            result[key] = mapped[:160]
    return result


def _payload_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None
