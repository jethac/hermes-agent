"""ElevenLabs streaming STT/TTS bridge for Hermes realtime voice."""

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
class ElevenLabsRealtimeBridgeConfig:
    """Runtime settings for a Hermes-compatible ElevenLabs realtime bridge."""

    api_key: Optional[str] = None
    auth_token: Optional[str] = None
    stt_url: str = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    tts_url: str = "wss://api.elevenlabs.io/v1/text-to-speech"
    model: str = "scribe_v2_realtime"
    tts_model: str = "eleven_flash_v2_5"
    voice_id: str = ""
    language: Optional[str] = None
    output_format: str = "pcm_24000"
    tts_sample_rate_hz: int = 24000
    output_languages: tuple[str, ...] = ()
    voice_settings: Mapping[str, Any] = field(default_factory=dict)
    chunk_length_schedule: tuple[int, ...] = (80, 120, 180, 240)
    stt_chunk_bytes: int = 6400
    stt_chunk_sleep_seconds: float = 0.02
    connect_timeout_seconds: float = 10.0


class ElevenLabsStreamingSTTBridgeSession:
    """One Hermes bridge session backed by ElevenLabs realtime STT."""

    def __init__(self, runtime: ElevenLabsRealtimeBridgeConfig):
        self.runtime = runtime
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._events: asyncio.Queue[VoiceEvent | None] = create_realtime_voice_event_queue()
        self._elevenlabs_ws: Any = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._sequence = 0
        self._closed = False
        self._last_input_generation: Optional[int] = None

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        if not self.runtime.api_key:
            raise RuntimeError("ElevenLabs realtime bridge requires ELEVENLABS_API_KEY")
        self.config = config
        self._elevenlabs_ws = await self._connect_elevenlabs(config)
        self._reader_task = asyncio.create_task(self._consume_elevenlabs_events())
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "ready",
                "provider": "elevenlabs",
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
            audio = elevenlabs_stt_audio_bytes(chunk, self.config)
        except Exception as exc:
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"elevenlabs stt audio conversion failed: {sanitize_realtime_voice_error(exc)}"},
            )
            return
        config = self.config
        await self._send_elevenlabs_audio(
            audio,
            commit=event.payload.get("end_of_utterance") is True,
            sample_rate=max(1, int(chunk.sample_rate_hz or (config.sample_rate_hz if config is not None else 16000) or 16000)),
        )

    async def _send_elevenlabs_audio(self, audio: bytes, *, commit: bool, sample_rate: int) -> None:
        if not audio and not commit:
            return
        chunk_bytes = max(1, int(self.runtime.stt_chunk_bytes or len(audio) or 1))
        sleep_seconds = max(0.0, float(self.runtime.stt_chunk_sleep_seconds or 0.0))
        if not audio:
            await self._send_elevenlabs_audio_message(b"", commit=True, sample_rate=sample_rate)
            return
        chunks = [audio[index:index + chunk_bytes] for index in range(0, len(audio), chunk_bytes)]
        for index, part in enumerate(chunks):
            is_last = index == len(chunks) - 1
            await self._send_elevenlabs_audio_message(part, commit=commit and is_last, sample_rate=sample_rate)
            if sleep_seconds and not is_last:
                await asyncio.sleep(sleep_seconds)

    async def _send_elevenlabs_audio_message(self, audio: bytes, *, commit: bool, sample_rate: int) -> None:
        await self._elevenlabs_ws.send(
            json.dumps(
                {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": base64.b64encode(audio).decode("ascii"),
                    "commit": commit,
                    "sample_rate": sample_rate,
                }
            )
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
        if self._elevenlabs_ws is not None:
            with contextlib.suppress(Exception):
                await self._elevenlabs_ws.close()
        await self._emit(VoiceEventType.SESSION_CLOSED, {"reason": "closed"})
        await put_realtime_voice_event(self._events, None)

    async def _connect_elevenlabs(self, config: RealtimeVoiceSessionConfig) -> Any:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("ElevenLabs realtime bridge requires the websockets package") from exc

        url = elevenlabs_stt_url(self.runtime, config)
        headers = {"xi-api-key": self.runtime.api_key}
        try:
            connect = websockets.connect(url, additional_headers=headers)
        except TypeError:
            connect = websockets.connect(url, extra_headers=headers)
        return await asyncio.wait_for(connect, timeout=max(0.1, self.runtime.connect_timeout_seconds))

    async def _consume_elevenlabs_events(self) -> None:
        try:
            async for raw in self._elevenlabs_ws:
                if not isinstance(raw, str):
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                event_type, payload = elevenlabs_stt_message_to_transcript_payload(
                    data,
                    input_generation=self._last_input_generation,
                )
                if event_type is None:
                    error = elevenlabs_error_from_message(data)
                    if error:
                        await self._emit(VoiceEventType.SESSION_ERROR, {"error": error})
                    continue
                await self._emit(event_type, payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"elevenlabs stt stream failed: {sanitize_realtime_voice_error(exc)}"},
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


class ElevenLabsStreamingTTSBridgeSession:
    """One Hermes bridge session backed by ElevenLabs realtime TTS."""

    def __init__(self, runtime: ElevenLabsRealtimeBridgeConfig):
        self.runtime = runtime
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._elevenlabs_ws: Any = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._events: asyncio.Queue[VoiceEvent | None] = create_realtime_voice_event_queue()
        self._sequence = 0
        self._closed = False
        self._playback_generation: Optional[int] = None

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        if not self.runtime.api_key:
            raise RuntimeError("ElevenLabs realtime bridge requires ELEVENLABS_API_KEY")
        if not self.runtime.voice_id:
            raise RuntimeError("ElevenLabs realtime TTS bridge requires ELEVENLABS_VOICE_ID")
        self.config = config
        self._elevenlabs_ws = await self._connect_elevenlabs_tts()
        self._reader_task = asyncio.create_task(self._consume_elevenlabs_audio())
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "ready",
                "provider": "elevenlabs",
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
            await self._reset_tts_socket()
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
        await self._elevenlabs_ws.send(json.dumps({"text": text, "try_trigger_generation": True}))
        await self._elevenlabs_ws.send(json.dumps({"text": "", "flush": True}))

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
        if self._elevenlabs_ws is not None:
            with contextlib.suppress(Exception):
                await self._elevenlabs_ws.send(json.dumps({"text": ""}))
            with contextlib.suppress(Exception):
                await self._elevenlabs_ws.close()
        await self._emit(VoiceEventType.SESSION_CLOSED, {"reason": "closed"})
        await put_realtime_voice_event(self._events, None)

    async def _connect_elevenlabs_tts(self) -> Any:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("ElevenLabs realtime bridge requires the websockets package") from exc

        url = elevenlabs_tts_url(self.runtime)
        try:
            connect = websockets.connect(url)
        except TypeError:
            connect = websockets.connect(url)
        ws = await asyncio.wait_for(connect, timeout=max(0.1, self.runtime.connect_timeout_seconds))
        await ws.send(json.dumps(elevenlabs_tts_start_message(self.runtime)))
        return ws

    async def _reset_tts_socket(self) -> None:
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        if self._elevenlabs_ws is not None:
            with contextlib.suppress(Exception):
                await self._elevenlabs_ws.send(json.dumps({"text": ""}))
            with contextlib.suppress(Exception):
                await self._elevenlabs_ws.close()
        self._elevenlabs_ws = await self._connect_elevenlabs_tts()
        self._reader_task = asyncio.create_task(self._consume_elevenlabs_audio())

    async def _consume_elevenlabs_audio(self) -> None:
        try:
            async for raw in self._elevenlabs_ws:
                if not isinstance(raw, str):
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                audio = data.get("audio")
                if not isinstance(audio, str) or not audio:
                    continue
                try:
                    audio_bytes = base64.b64decode(audio)
                except Exception:
                    continue
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
                {"error": f"elevenlabs tts stream failed: {sanitize_realtime_voice_error(exc)}"},
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


def elevenlabs_stt_url(
    runtime: ElevenLabsRealtimeBridgeConfig,
    config: RealtimeVoiceSessionConfig,
) -> str:
    query = {
        "model_id": runtime.model,
        "audio_format": _elevenlabs_audio_format(config.sample_rate_hz),
        "commit_strategy": "manual",
        "include_language_detection": "true",
    }
    language = _clean_language(runtime.language)
    if language and language.lower() not in {"auto", "multi"}:
        query["language_code"] = language
    separator = "&" if urllib.parse.urlparse(runtime.stt_url).query else "?"
    return f"{runtime.stt_url}{separator}{urllib.parse.urlencode(query)}"


def elevenlabs_tts_url(runtime: ElevenLabsRealtimeBridgeConfig) -> str:
    base = str(runtime.tts_url or "").rstrip("/")
    path = urllib.parse.quote(str(runtime.voice_id or "").strip(), safe="")
    query = {
        "model_id": runtime.tts_model,
        "output_format": runtime.output_format,
    }
    separator = "&" if urllib.parse.urlparse(base).query else "?"
    return f"{base}/{path}/stream-input{separator}{urllib.parse.urlencode(query)}"


def elevenlabs_tts_start_message(runtime: ElevenLabsRealtimeBridgeConfig) -> dict[str, Any]:
    message: dict[str, Any] = {
        "text": " ",
        "xi_api_key": runtime.api_key or "",
        "generation_config": {"chunk_length_schedule": list(runtime.chunk_length_schedule)},
    }
    if runtime.voice_settings:
        message["voice_settings"] = dict(runtime.voice_settings)
    return message


def elevenlabs_stt_message_to_transcript_payload(
    data: Mapping[str, Any],
    *,
    input_generation: Optional[int] = None,
) -> tuple[Optional[VoiceEventType], dict[str, Any]]:
    message_type = str(data.get("message_type") or "")
    if message_type not in {
        "partial_transcript",
        "committed_transcript",
        "committed_transcript_with_timestamps",
    }:
        return None, {}
    text = str(data.get("text") or "").strip()
    if not text:
        return None, {}
    payload: dict[str, Any] = {"text": text}
    language = _clean_language(data.get("language_code"))
    if language:
        payload["language"] = language
    if input_generation is not None:
        payload["input_generation"] = input_generation
    event_type = (
        VoiceEventType.TRANSCRIPT_PARTIAL
        if message_type == "partial_transcript"
        else VoiceEventType.TRANSCRIPT_FINAL
    )
    return event_type, payload


def elevenlabs_stt_audio_bytes(
    chunk: AudioChunk,
    config: Optional[RealtimeVoiceSessionConfig] = None,
) -> bytes:
    """Return raw PCM16LE audio bytes for ElevenLabs realtime STT.

    Hermes accepts browser-friendly audio such as WebM/Opus at the provider-
    neutral realtime boundary. ElevenLabs realtime STT is configured with an
    audio_format=pcm_* query parameter, so compressed container bytes must be
    normalized at this provider bridge before upload.
    """

    if chunk.codec == VoiceAudioCodec.PCM16:
        return chunk.data
    sample_rate_hz = max(
        1,
        int(chunk.sample_rate_hz or (config.sample_rate_hz if config is not None else 16000) or 16000),
    )
    channels = max(1, int(chunk.channels or (config.channels if config is not None else 1) or 1))
    return _ffmpeg_to_pcm16le(
        chunk.data,
        codec=chunk.codec,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
    )


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
        raise RuntimeError("ffmpeg is required to convert compressed audio to PCM16 for ElevenLabs STT")
    suffix = ".webm" if codec == VoiceAudioCodec.WEBM_OPUS else ".opus"
    with tempfile.TemporaryDirectory(prefix="hermes-elevenlabs-stt-") as tmp:
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


def elevenlabs_error_from_message(data: Mapping[str, Any]) -> str:
    message_type = str(data.get("message_type") or "")
    if not message_type.endswith("_error") and "error" not in message_type:
        return ""
    message = data.get("message") or data.get("error") or data.get("detail") or message_type
    return sanitize_realtime_voice_error(message)


def create_elevenlabs_realtime_bridge_app(runtime: Optional[ElevenLabsRealtimeBridgeConfig] = None):
    """Create the FastAPI app for the ElevenLabs realtime bridge."""

    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Hermes ElevenLabs realtime bridge")
    runtime = runtime or elevenlabs_bridge_config_from_env()

    @app.get("/health")
    async def health(request: Request):
        if not _authorized(request.headers, runtime.auth_token):
            raise HTTPException(status_code=401, detail="unauthorized")
        has_websockets = _module_available("websockets")
        stt_ready = bool(runtime.api_key) and has_websockets
        tts_ready = stt_ready and bool(runtime.voice_id)
        return {
            "ok": stt_ready and tts_ready,
            "kind": "streaming_stt_bridge",
            "frontend": {
                "provider": "elevenlabs",
                "model": runtime.model,
                "language": runtime.language or "auto",
                "tts_model": runtime.tts_model,
                "voice_id": "configured" if runtime.voice_id else "",
            },
            "capabilities": {
                "streaming_stt": stt_ready,
                "tts": tts_ready,
                "streaming_tts": tts_ready,
                "native_s2s": False,
                "input_languages": elevenlabs_input_languages(runtime),
                "output_languages": elevenlabs_tts_output_languages(runtime),
            },
        }

    @app.websocket("/v1/streaming-stt/session")
    async def streaming_stt_session(ws: WebSocket):
        if not _authorized(ws.headers, runtime.auth_token):
            await ws.close(code=1008, reason="unauthorized")
            return
        await ws.accept()
        session: Optional[ElevenLabsStreamingSTTBridgeSession] = None
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
                    session = ElevenLabsStreamingSTTBridgeSession(runtime)
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
        session: Optional[ElevenLabsStreamingTTSBridgeSession] = None
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
                    session = ElevenLabsStreamingTTSBridgeSession(runtime)
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


def elevenlabs_bridge_config_from_env() -> ElevenLabsRealtimeBridgeConfig:
    auth_token_env = os.environ.get("HERMES_ELEVENLABS_BRIDGE_TOKEN_ENV") or "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    output_format = os.environ.get("HERMES_ELEVENLABS_OUTPUT_FORMAT") or "pcm_24000"
    return ElevenLabsRealtimeBridgeConfig(
        api_key=os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("HERMES_ELEVENLABS_API_KEY") or None,
        auth_token=os.environ.get(auth_token_env) or None,
        stt_url=os.environ.get("HERMES_ELEVENLABS_STT_URL")
        or "wss://api.elevenlabs.io/v1/speech-to-text/realtime",
        tts_url=os.environ.get("HERMES_ELEVENLABS_TTS_URL") or "wss://api.elevenlabs.io/v1/text-to-speech",
        model=os.environ.get("HERMES_ELEVENLABS_STT_MODEL") or "scribe_v2_realtime",
        tts_model=os.environ.get("HERMES_ELEVENLABS_TTS_MODEL") or "eleven_flash_v2_5",
        voice_id=os.environ.get("ELEVENLABS_VOICE_ID") or os.environ.get("HERMES_ELEVENLABS_VOICE_ID") or "",
        language=os.environ.get("HERMES_ELEVENLABS_LANGUAGE") or None,
        output_format=output_format,
        tts_sample_rate_hz=_sample_rate_from_output_format(output_format),
        output_languages=tuple(_parse_language_list(os.environ.get("HERMES_ELEVENLABS_OUTPUT_LANGUAGES") or "")),
        voice_settings=_parse_json_object(os.environ.get("HERMES_ELEVENLABS_VOICE_SETTINGS") or ""),
        chunk_length_schedule=tuple(
            _parse_int_list(os.environ.get("HERMES_ELEVENLABS_CHUNK_LENGTH_SCHEDULE") or "80,120,180,240")
        ),
        stt_chunk_bytes=int(os.environ.get("HERMES_ELEVENLABS_STT_CHUNK_BYTES") or 6400),
        stt_chunk_sleep_seconds=float(os.environ.get("HERMES_ELEVENLABS_STT_CHUNK_SLEEP_SECONDS") or 0.02),
        connect_timeout_seconds=float(os.environ.get("HERMES_ELEVENLABS_CONNECT_TIMEOUT_SECONDS") or 10),
    )


def elevenlabs_bridge_prerequisite_issues(
    runtime: Optional[ElevenLabsRealtimeBridgeConfig] = None,
    *,
    require_auth_token: bool = False,
    required_input_languages: Sequence[str] = (),
    required_output_languages: Sequence[str] = (),
    module_available: Optional[Any] = None,
) -> list[str]:
    runtime = runtime or elevenlabs_bridge_config_from_env()
    available = module_available or _module_available
    issues: list[str] = []
    if not runtime.api_key:
        issues.append("ELEVENLABS_API_KEY or HERMES_ELEVENLABS_API_KEY is required")
    if not runtime.voice_id:
        issues.append("ELEVENLABS_VOICE_ID or HERMES_ELEVENLABS_VOICE_ID is required for streaming TTS")
    if not available("websockets"):
        issues.append(
            "Python package 'websockets==15.0.1' is required; "
            "install with `python -m pip install 'hermes-agent[voice]'`"
        )
    if require_auth_token and not runtime.auth_token:
        issues.append("HERMES_STREAMING_STT_BRIDGE_TOKEN is required in strict mode")
    configured_input_languages = set(elevenlabs_input_languages(runtime))
    missing_input_languages = [
        language
        for language in (_primary_language(value) for value in required_input_languages)
        if language and language not in configured_input_languages
    ]
    if missing_input_languages:
        configured = ",".join(sorted(configured_input_languages)) or "none"
        required = ",".join(missing_input_languages)
        issues.append(
            "ElevenLabs realtime STT input routing is missing required language(s) "
            f"{required}; configured input language(s): {configured}"
        )
    configured_output_languages = set(elevenlabs_tts_output_languages(runtime))
    missing_output_languages = [
        language
        for language in (_primary_language(value) for value in required_output_languages)
        if language and language not in configured_output_languages
    ]
    if missing_output_languages:
        configured = ",".join(sorted(configured_output_languages)) or "none"
        required = ",".join(missing_output_languages)
        issues.append(
            "ElevenLabs realtime TTS output routing is missing required language(s) "
            f"{required}; configured output language(s): {configured}"
        )
    return issues


def elevenlabs_input_languages(runtime: ElevenLabsRealtimeBridgeConfig) -> list[str]:
    language = _primary_language(runtime.language)
    if language and language not in {"auto", "multi"}:
        return [language]
    return ["en", "ja"]


def elevenlabs_tts_output_languages(runtime: ElevenLabsRealtimeBridgeConfig) -> list[str]:
    languages = {_primary_language(language) for language in runtime.output_languages}
    languages.discard("")
    if languages:
        return sorted(languages)
    model = str(runtime.tts_model or "").lower()
    if "multilingual" in model or model.startswith("eleven_flash"):
        return ["en", "ja"]
    return ["en"]


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _authorized(headers: Mapping[str, str], token: Optional[str]) -> bool:
    if not token:
        return True
    return headers.get("authorization") == f"Bearer {token}"


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


def _elevenlabs_audio_format(sample_rate_hz: int) -> str:
    sample_rate = max(1, int(sample_rate_hz or 16000))
    if sample_rate <= 8000:
        return "pcm_8000"
    if sample_rate <= 16000:
        return "pcm_16000"
    if sample_rate <= 22050:
        return "pcm_22050"
    if sample_rate <= 24000:
        return "pcm_24000"
    if sample_rate <= 44100:
        return "pcm_44100"
    return "pcm_48000"


def _sample_rate_from_output_format(value: str) -> int:
    text = str(value or "")
    for rate in (8000, 16000, 22050, 24000, 44100, 48000):
        if str(rate) in text:
            return rate
    return 24000


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
    if not language:
        return ""
    return language.split("-", 1)[0]


def _parse_language_list(value: str) -> list[str]:
    result: list[str] = []
    for item in str(value or "").split(","):
        language = _primary_language(item)
        if language and language not in result:
            result.append(language)
    return result


def _parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for item in str(value or "").split(","):
        try:
            parsed = int(item.strip())
        except ValueError:
            continue
        if parsed > 0:
            result.append(parsed)
    return result or [80, 120, 180, 240]


def _parse_json_object(value: str) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except Exception:
        return {}
    return dict(data) if isinstance(data, Mapping) else {}
