"""OpenAI Realtime frontend provider for Hermes realtime voice."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, Optional
from urllib.parse import quote

from agent.realtime_voice import (
    AudioChunk,
    RealtimeVoiceSessionConfig,
    VoiceAudioCodec,
    VoiceEvent,
    VoiceEventType,
    create_realtime_voice_event_queue,
    put_realtime_voice_event,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error


OPENAI_REALTIME_SAMPLE_RATE_HZ = 24000
OPENAI_REALTIME_DEFAULT_MODEL = "gpt-realtime-2"
OPENAI_REALTIME_DEFAULT_TRANSCRIPTION_MODEL = "gpt-realtime-whisper"
OPENAI_REALTIME_DEFAULT_VOICE = "marin"


WebSocketConnector = Callable[[str, Mapping[str, str], float], Awaitable[Any]]


@dataclass(frozen=True)
class OpenAIRealtimeFrontendConfig:
    """Runtime config for the OpenAI Realtime frontend provider."""

    api_key: Optional[str] = None
    base_url: str = "wss://api.openai.com/v1/realtime"
    model: str = OPENAI_REALTIME_DEFAULT_MODEL
    voice: str = OPENAI_REALTIME_DEFAULT_VOICE
    transcription_model: str = OPENAI_REALTIME_DEFAULT_TRANSCRIPTION_MODEL
    connect_timeout_seconds: float = 10.0
    safety_identifier: Optional[str] = None
    instructions: str = (
        "You are Hermes' low-latency realtime voice interface. Keep spoken output short. "
        "Do not claim to be a separate bot; Hermes' backend oracle owns reasoning, tools, "
        "memory, and durable work."
    )


class OpenAIRealtimeFrontendSession:
    """Bridge Hermes realtime voice events to OpenAI Realtime WebSocket events."""

    def __init__(
        self,
        runtime: OpenAIRealtimeFrontendConfig,
        *,
        connector: Optional[WebSocketConnector] = None,
    ):
        self.runtime = runtime
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._connector = connector or _connect_websocket
        self._ws: Any = None
        self._events: asyncio.Queue[VoiceEvent | None] = create_realtime_voice_event_queue()
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._sequence = 0
        self._closed = False
        self._active_playback_generation: Optional[int] = None
        self._playback_active = False
        self._playback_started_generation: Optional[int] = None
        self._active_response_id: Optional[str] = None

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        api_key = (self.runtime.api_key or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or HERMES_OPENAI_REALTIME_API_KEY is required")
        self.config = config
        url = _openai_realtime_url(self.runtime.base_url, config.frontend_model or self.runtime.model)
        headers = {"Authorization": f"Bearer {api_key}"}
        if self.runtime.safety_identifier:
            headers["OpenAI-Safety-Identifier"] = self.runtime.safety_identifier
        self._ws = await self._connector(url, headers, max(self.runtime.connect_timeout_seconds, 0.1))
        await self._send_openai(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": self.runtime.instructions,
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": OPENAI_REALTIME_SAMPLE_RATE_HZ},
                            "transcription": {"model": self.runtime.transcription_model},
                            "turn_detection": None,
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": OPENAI_REALTIME_SAMPLE_RATE_HZ},
                            "voice": self.runtime.voice,
                        },
                    },
                },
            }
        )
        self._reader_task = asyncio.create_task(self._consume_openai_events())
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "ready",
                "provider": "openai_realtime",
                "model": config.frontend_model or self.runtime.model,
                "voice": self.runtime.voice,
                "streaming_stt": True,
                "streaming_tts": True,
                "native_s2s": True,
                "server_vad": False,
                "response_cancel": True,
                "input_sample_rate_hz": OPENAI_REALTIME_SAMPLE_RATE_HZ,
                "output_sample_rate_hz": OPENAI_REALTIME_SAMPLE_RATE_HZ,
            },
        )

    async def receive_event(self, event: VoiceEvent) -> None:
        if self._closed:
            return
        if event.type == VoiceEventType.SESSION_CLOSED:
            await self.close()
            return
        if event.type == VoiceEventType.BARGE_IN:
            await self._handle_barge_in(event)
            return
        if event.type == VoiceEventType.AUDIO_INPUT_CHUNK:
            await self._handle_audio_input(event)
            return
        if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL and event.payload.get("speak") is True:
            await self._handle_speak(event)

    async def events(self) -> AsyncIterator[VoiceEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def close(self) -> None:
        if self._closed:
            return
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._send_openai({"type": "response.cancel"})
            with contextlib.suppress(Exception):
                await self._send_openai({"type": "input_audio_buffer.clear"})
        await self._emit_playback_stopped(self._playback_started_generation)
        self._closed = True
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        await self._emit(VoiceEventType.SESSION_CLOSED, {"reason": "closed"})
        await put_realtime_voice_event(self._events, None)

    async def _handle_audio_input(self, event: VoiceEvent) -> None:
        try:
            chunk = AudioChunk.from_payload(event.payload)
        except Exception:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid audio chunk"})
            return
        if chunk.codec != VoiceAudioCodec.PCM16:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": "OpenAI Realtime provider requires pcm16 input"})
            return
        pcm24 = resample_pcm16_mono(
            chunk.data,
            from_rate_hz=int(event.payload.get("sample_rate_hz") or chunk.sample_rate_hz),
            to_rate_hz=OPENAI_REALTIME_SAMPLE_RATE_HZ,
            channels=int(event.payload.get("channels") or chunk.channels),
        )
        await self._send_openai(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm24).decode("ascii"),
            }
        )
        if event.payload.get("end_of_utterance") is True:
            await self._send_openai({"type": "input_audio_buffer.commit"})

    async def _handle_speak(self, event: VoiceEvent) -> None:
        text = str(event.payload.get("text") or "").strip()
        if not text:
            return
        playback_generation = _payload_int(event.payload.get("playback_generation"))
        self._active_playback_generation = playback_generation
        await self._send_openai(
            {
                "type": "response.create",
                "response": {
                    "modalities": ["audio", "text"],
                    "instructions": (
                        "Speak the following Hermes response verbatim and do not add extra words:\n\n"
                        f"{text}"
                    ),
                    "audio": {
                        "output": {
                            "format": {"type": "audio/pcm", "rate": OPENAI_REALTIME_SAMPLE_RATE_HZ},
                            "voice": self.runtime.voice,
                        }
                    },
                },
            }
        )

    async def _handle_barge_in(self, event: VoiceEvent) -> None:
        payload = {"reason": event.payload.get("reason") or "client"}
        playback_generation = _payload_int(event.payload.get("playback_generation"))
        if playback_generation is not None:
            payload["playback_generation"] = playback_generation
        with contextlib.suppress(Exception):
            await self._send_openai({"type": "response.cancel"})
        with contextlib.suppress(Exception):
            await self._send_openai({"type": "input_audio_buffer.clear"})
        self._active_playback_generation = None
        self._active_response_id = None
        await self._emit_playback_stopped(playback_generation)
        await self._emit(VoiceEventType.BARGE_IN, payload)

    async def _consume_openai_events(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    event = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
                except Exception:
                    await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid OpenAI realtime event"})
                    continue
                await self._handle_openai_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                await self._emit(
                    VoiceEventType.SESSION_ERROR,
                    {"error": f"OpenAI realtime event stream failed: {sanitize_realtime_voice_error(exc)}"},
                )

    async def _handle_openai_event(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type == "error":
            error = event.get("error")
            if isinstance(error, Mapping):
                message = error.get("message") or error.get("code") or "OpenAI realtime error"
            else:
                message = "OpenAI realtime error"
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": sanitize_realtime_voice_error(message)})
            return
        if event_type == "response.created":
            response = event.get("response")
            if isinstance(response, Mapping):
                response_id = response.get("id")
                if isinstance(response_id, str):
                    self._active_response_id = response_id
            return
        if event_type == "conversation.item.input_audio_transcription.delta":
            text = str(event.get("delta") or "")
            if text:
                await self._emit(VoiceEventType.TRANSCRIPT_PARTIAL, {"text": text, "stability": 0.7})
            return
        if event_type in {"response.output_text.delta", "response.text.delta"}:
            text = str(event.get("delta") or "")
            if text:
                await self._emit(VoiceEventType.ASSISTANT_TEXT_PARTIAL, {"text": text})
            return
        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = str(event.get("transcript") or "").strip()
            if transcript:
                await self._emit(VoiceEventType.TRANSCRIPT_FINAL, {"text": transcript})
            return
        if event_type in {"response.output_audio.delta", "response.audio.delta"}:
            delta = event.get("delta")
            if isinstance(delta, str):
                await self._emit_audio_delta(delta)
            return
        if event_type in {"response.done", "response.output_item.done"}:
            await self._emit_playback_stopped(self._active_playback_generation)
            await self._emit(VoiceEventType.ASSISTANT_COMMIT, {"response_id": self._active_response_id})
            self._active_response_id = None
            return
        if event_type == "input_audio_buffer.speech_started":
            await self._emit(VoiceEventType.BARGE_IN, {"reason": "speech_started"})

    async def _emit_audio_delta(self, data_b64: str) -> None:
        config = self.config
        if config is None:
            return
        try:
            pcm24 = base64.b64decode(data_b64.encode("ascii"), validate=True)
        except Exception:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid OpenAI output audio delta"})
            return
        pcm = resample_pcm16_mono(
            pcm24,
            from_rate_hz=OPENAI_REALTIME_SAMPLE_RATE_HZ,
            to_rate_hz=int(config.sample_rate_hz or 16000),
            channels=1,
        )
        payload = AudioChunk(
            codec=VoiceAudioCodec.PCM16,
            data=pcm,
            sample_rate_hz=int(config.sample_rate_hz or 16000),
            channels=1,
        ).to_payload()
        if self._active_playback_generation is not None:
            payload["playback_generation"] = self._active_playback_generation
        payload["metrics"] = {"openai_realtime": True}
        await self._emit_playback_started(self._active_playback_generation)
        await self._emit(VoiceEventType.AUDIO_OUTPUT_CHUNK, payload)

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

    async def _send_openai(self, payload: Mapping[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("OpenAI realtime websocket is not connected")
        await self._ws.send(json.dumps(dict(payload)))

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


async def _connect_websocket(url: str, headers: Mapping[str, str], timeout: float) -> Any:
    import websockets

    return await websockets.connect(
        url,
        additional_headers=dict(headers),
        open_timeout=timeout,
    )


def _openai_realtime_url(base_url: str, model: str) -> str:
    base = (base_url or "wss://api.openai.com/v1/realtime").rstrip("/")
    return f"{base}?model={quote(model or OPENAI_REALTIME_DEFAULT_MODEL, safe='')}"


def resample_pcm16_mono(data: bytes, *, from_rate_hz: int, to_rate_hz: int, channels: int = 1) -> bytes:
    """Convert signed 16-bit PCM to mono and resample with linear interpolation."""

    if not data:
        return b""
    channels = max(1, int(channels or 1))
    from_rate_hz = max(1, int(from_rate_hz or to_rate_hz))
    to_rate_hz = max(1, int(to_rate_hz or from_rate_hz))
    usable = len(data) - (len(data) % (2 * channels))
    if usable <= 0:
        return b""
    samples = []
    for offset in range(0, usable, 2 * channels):
        total = 0
        for channel in range(channels):
            sample_offset = offset + channel * 2
            total += int.from_bytes(data[sample_offset:sample_offset + 2], "little", signed=True)
        samples.append(int(round(total / channels)))
    if from_rate_hz == to_rate_hz:
        return b"".join(_pcm16(sample) for sample in samples)
    if len(samples) == 1:
        return b"".join(_pcm16(samples[0]) for _ in range(max(1, int(round(to_rate_hz / from_rate_hz)))))
    output_len = max(1, int(round(len(samples) * to_rate_hz / from_rate_hz)))
    ratio = from_rate_hz / to_rate_hz
    output = bytearray()
    for index in range(output_len):
        source_pos = index * ratio
        left = min(int(math.floor(source_pos)), len(samples) - 1)
        right = min(left + 1, len(samples) - 1)
        frac = source_pos - left
        value = int(round(samples[left] * (1.0 - frac) + samples[right] * frac))
        output.extend(_pcm16(value))
    return bytes(output)


def _pcm16(value: int) -> bytes:
    return max(-32768, min(32767, int(value))).to_bytes(2, "little", signed=True)


def _payload_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
