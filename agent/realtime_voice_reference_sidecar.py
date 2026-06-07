"""Reference realtime voice sidecar for Hermes.

This is the server-side counterpart to ``RealtimeVoiceSidecarClient``. It is
designed to run locally on ordinary developer machines, while optionally using
a LAN vLLM/Gemma audio endpoint when one is configured.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Mapping, Optional

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


TranscribeFn = Callable[[str], Mapping[str, Any]]
SynthesizeFn = Callable[[str], Any]


@dataclass(frozen=True)
class ReferenceSidecarRuntimeConfig:
    """Runtime knobs for the reference sidecar process."""

    vllm_base_url: Optional[str] = None
    vllm_model: Optional[str] = None
    vllm_timeout_seconds: float = 60.0
    local_stt_enabled: bool = True
    local_tts_enabled: bool = True
    auth_token: Optional[str] = None


def reference_sidecar_health_payload(runtime: ReferenceSidecarRuntimeConfig) -> dict[str, Any]:
    vllm_enabled = bool(runtime.vllm_base_url and runtime.vllm_model)

    return {
        "ok": True,
        "kind": "reference",
        "frontend": {
            "provider": "vllm" if vllm_enabled else "local",
            "model": runtime.vllm_model or "",
        },
        "capabilities": {
            "utterance_stt": vllm_enabled or runtime.local_stt_enabled,
            "streaming_stt": False,
            "tts": runtime.local_tts_enabled,
            "native_s2s": False,
            "vllm_audio_frontend": vllm_enabled,
        },
        "local": {
            "stt": runtime.local_stt_enabled,
            "tts": runtime.local_tts_enabled,
        },
    }


class ReferenceRealtimeVoiceSidecarSession:
    """One realtime sidecar session.

    Audio chunks are accumulated until the client marks an utterance boundary.
    This makes the baseline path usable without specialized streaming ASR. The
    same event contract is kept for higher-grade frontends: sidecars can emit
    partial transcript events as soon as they have them.
    """

    def __init__(
        self,
        runtime: ReferenceSidecarRuntimeConfig,
        *,
        transcribe_audio_func: Optional[TranscribeFn] = None,
        synthesize_func: Optional[SynthesizeFn] = None,
    ):
        self.runtime = runtime
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._transcribe_audio_func = transcribe_audio_func
        self._synthesize_func = synthesize_func
        self._events: asyncio.Queue[VoiceEvent | None] = create_realtime_voice_event_queue()
        self._audio: list[bytes] = []
        self._sequence = 0
        self._closed = False
        self._active_tasks: set[asyncio.Task[None]] = set()

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        self.config = config
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "ready",
                "provider": config.frontend_provider or "local",
                "model": config.frontend_model or "",
                "vllm": bool(self.runtime.vllm_base_url and self.runtime.vllm_model),
                "local_stt": self.runtime.local_stt_enabled,
                "local_tts": self.runtime.local_tts_enabled,
            },
        )

    async def receive_event(self, event: VoiceEvent) -> None:
        if self._closed:
            return
        if event.type == VoiceEventType.SESSION_CLOSED:
            await self.close()
            return
        if event.type == VoiceEventType.BARGE_IN:
            self._audio.clear()
            self._cancel_active_tasks()
            await self._emit(VoiceEventType.BARGE_IN, {"reason": event.payload.get("reason") or "client"})
            return
        if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL and event.payload.get("speak") is True:
            text = str(event.payload.get("text") or "").strip()
            if text:
                self._track_task(asyncio.create_task(self._speak(text, _payload_generation(event.payload))))
            return
        if event.type != VoiceEventType.AUDIO_INPUT_CHUNK:
            return

        transcript = str(event.payload.get("transcript") or "").strip()
        if transcript:
            if event.payload.get("end_of_utterance") is True:
                await self._emit(VoiceEventType.TRANSCRIPT_FINAL, {"text": transcript})
            else:
                await self._emit(VoiceEventType.TRANSCRIPT_PARTIAL, {"text": transcript, "stability": 0.8})
            return

        try:
            chunk = AudioChunk.from_payload(event.payload)
        except Exception:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid audio chunk"})
            return

        self._audio.append(chunk.data)
        if event.payload.get("end_of_utterance") is True:
            audio = b"".join(self._audio)
            self._audio.clear()
            if audio:
                await self._emit(VoiceEventType.TRANSCRIPT_PARTIAL, {"text": "", "stability": 0.1})
                self._track_task(asyncio.create_task(self._transcribe(audio, chunk.codec)))

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
        self._cancel_active_tasks()
        await self._emit(VoiceEventType.SESSION_CLOSED, {"reason": "closed"})
        await put_realtime_voice_event(self._events, None)

    def _track_task(self, task: asyncio.Task[None]) -> None:
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    def _cancel_active_tasks(self) -> None:
        for task in list(self._active_tasks):
            if not task.done():
                task.cancel()

    async def _transcribe(self, audio: bytes, codec: VoiceAudioCodec) -> None:
        try:
            transcript = await asyncio.to_thread(self._transcribe_sync, audio, codec)
            if transcript:
                await self._emit(VoiceEventType.TRANSCRIPT_FINAL, {"text": transcript})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": f"transcription failed: {exc}"})

    def _transcribe_sync(self, audio: bytes, codec: VoiceAudioCodec) -> str:
        if self.runtime.vllm_base_url and self.runtime.vllm_model:
            return self._transcribe_with_vllm(audio, codec)
        if not self.runtime.local_stt_enabled:
            raise RuntimeError("local STT is disabled and no vLLM audio frontend is configured")

        transcribe_audio = self._transcribe_audio_func
        if transcribe_audio is None:
            from tools.transcription_tools import transcribe_audio as transcribe_audio

        path = _write_temp_audio(audio, codec)
        try:
            result = transcribe_audio(path)
            if not result.get("success"):
                raise RuntimeError(str(result.get("error") or "transcription failed"))
            return str(result.get("transcript") or "").strip()
        finally:
            _unlink(path)

    def _transcribe_with_vllm(self, audio: bytes, codec: VoiceAudioCodec) -> str:
        mime_type = _mime_type_for_codec(codec)
        audio_b64 = base64.b64encode(audio).decode("ascii")
        payload = {
            "model": self.runtime.vllm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": f"data:{mime_type};base64,{audio_b64}"}},
                        {
                            "type": "text",
                            "text": "Transcribe the speech in this audio. Return only the spoken words.",
                        },
                    ],
                }
            ],
            "max_tokens": 256,
            "temperature": 0,
        }
        url = f"{self.runtime.vllm_base_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.runtime.vllm_timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data["choices"][0]["message"].get("content") or "").strip()

    async def _speak(self, text: str, playback_generation: Optional[int] = None) -> None:
        if not self.runtime.local_tts_enabled:
            return
        try:
            file_path = await asyncio.to_thread(self._speak_sync, text)
            if not file_path:
                return
            try:
                with open(file_path, "rb") as fh:
                    data = fh.read()
                if data:
                    payload = AudioChunk(codec=VoiceAudioCodec.OPUS, data=data).to_payload()
                    payload["mime_type"] = _mime_type_for_path(file_path)
                    if playback_generation is not None:
                        payload["playback_generation"] = playback_generation
                    await self._emit(VoiceEventType.AUDIO_OUTPUT_CHUNK, payload)
            finally:
                _unlink(file_path)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": f"tts failed: {exc}"})

    def _speak_sync(self, text: str) -> str:
        synthesize = self._synthesize_func
        if synthesize is None:
            from tools.tts_tool import text_to_speech_tool as synthesize

        raw = synthesize(text)
        result = json.loads(raw) if isinstance(raw, str) else raw
        if not result.get("success"):
            raise RuntimeError(str(result.get("error") or "speech synthesis failed"))
        return str(result.get("file_path") or "")

    async def _emit(self, event_type: VoiceEventType, payload: Mapping[str, Any]) -> None:
        if self.config is None:
            return
        self._sequence += 1
        await put_realtime_voice_event(
            self._events,
            VoiceEvent(
                type=event_type,
                session_id=self.config.session_id,
                sequence=self._sequence,
                payload=dict(payload),
            )
        )


def create_reference_sidecar_app(runtime: Optional[ReferenceSidecarRuntimeConfig] = None):
    """Create the FastAPI app for the reference sidecar."""

    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Hermes realtime voice reference sidecar")
    runtime = runtime or runtime_config_from_env()

    @app.get("/health")
    async def health(request: Request):
        if not _authorized(request.headers, runtime.auth_token):
            raise HTTPException(status_code=401, detail="unauthorized")
        return reference_sidecar_health_payload(runtime)

    @app.websocket("/v1/realtime-text/session")
    async def realtime_text_session(ws: WebSocket):
        if not _authorized(ws.headers, runtime.auth_token):
            await ws.close(code=1008, reason="unauthorized")
            return
        await ws.accept()
        session: Optional[ReferenceRealtimeVoiceSidecarSession] = None
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
                    session = ReferenceRealtimeVoiceSidecarSession(runtime)
                    await session.start(config)
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


def runtime_config_from_env() -> ReferenceSidecarRuntimeConfig:
    return ReferenceSidecarRuntimeConfig(
        vllm_base_url=os.environ.get("HERMES_VOICE_VLLM_BASE_URL") or None,
        vllm_model=os.environ.get("HERMES_VOICE_VLLM_MODEL") or None,
        vllm_timeout_seconds=float(os.environ.get("HERMES_VOICE_VLLM_TIMEOUT_SECONDS") or 60),
        local_stt_enabled=_env_bool("HERMES_VOICE_LOCAL_STT_ENABLED", True),
        local_tts_enabled=_env_bool("HERMES_VOICE_LOCAL_TTS_ENABLED", True),
        auth_token=os.environ.get("HERMES_VOICE_SIDECAR_TOKEN")
        or os.environ.get("HERMES_SPARK_VOICE_TOKEN")
        or None,
    )


def _authorized(headers: Mapping[str, str], token: Optional[str]) -> bool:
    if not token:
        return True
    return headers.get("authorization") == f"Bearer {token}"


def _write_temp_audio(audio: bytes, codec: VoiceAudioCodec) -> str:
    suffix = {
        VoiceAudioCodec.PCM16: ".wav",
        VoiceAudioCodec.OPUS: ".ogg",
        VoiceAudioCodec.WEBM_OPUS: ".webm",
    }.get(codec, ".webm")
    with tempfile.NamedTemporaryFile(prefix="hermes-voice-sidecar-", suffix=suffix, delete=False) as tmp:
        tmp.write(audio)
        return tmp.name


def _mime_type_for_codec(codec: VoiceAudioCodec) -> str:
    return {
        VoiceAudioCodec.PCM16: "audio/wav",
        VoiceAudioCodec.OPUS: "audio/ogg",
        VoiceAudioCodec.WEBM_OPUS: "audio/webm",
    }.get(codec, "audio/webm")


def _mime_type_for_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
    }.get(ext, "audio/mpeg")


def _payload_generation(payload: Mapping[str, Any]) -> Optional[int]:
    value = payload.get("playback_generation")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
