"""Realtime voice model-sidecar client."""

from __future__ import annotations

import asyncio
import base64
import json
import urllib.parse
from typing import Any, AsyncIterator, Optional

from agent.realtime_voice import (
    REALTIME_VOICE_SIDECAR_SEND_TIMEOUT_SECONDS,
    RealtimeVoiceSessionConfig,
    VoiceAudioCodec,
    VoiceEvent,
    VoiceEventType,
    binary_audio_frame_from_event,
    create_realtime_voice_event_queue,
    event_from_binary_output_audio_frame,
    put_realtime_voice_event,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error


class RealtimeVoiceSidecarClient:
    """Websocket client for a Gemma/STT/TTS realtime sidecar.

    Hermes owns authorization, session state, the backend oracle, and durable
    history. The sidecar owns realtime audio front-end work such as streaming
    STT/audio understanding and optional streaming TTS.
    """

    def __init__(self, *, path: str = "/v1/realtime-text/session"):
        self.path = path
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._ws: Any = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._events: asyncio.Queue[VoiceEvent | None] = create_realtime_voice_event_queue()
        self._closed = False

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._closed

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        sidecar_base_url = config.effective_sidecar_base_url
        if not sidecar_base_url:
            raise RuntimeError("realtime voice sidecar requires voice.realtime.sidecar_base_url")
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("realtime voice sidecar requires the websockets package") from exc

        self.config = config
        url = sidecar_ws_url(sidecar_base_url, self.path)
        headers = {}
        if config.effective_sidecar_token:
            headers["Authorization"] = f"Bearer {config.effective_sidecar_token}"
        timeout = max(0.1, float(config.sidecar_connect_timeout_seconds or 10.0))
        try:
            connect = websockets.connect(url, additional_headers=headers or None)
        except TypeError:
            connect = websockets.connect(url, extra_headers=headers or None)
        try:
            self._ws = await asyncio.wait_for(connect, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"realtime voice sidecar connect timed out after {timeout:g}s") from exc
        await self._send_with_timeout(json.dumps({"type": "session.config", "payload": config.to_wire()}))
        self._reader_task = asyncio.create_task(self._read_events())

    async def send_event(self, event: VoiceEvent) -> None:
        if not self.connected:
            raise RuntimeError("realtime voice sidecar is not connected")
        frame = binary_audio_frame_from_event(event)
        if frame is not None and event.type == VoiceEventType.AUDIO_INPUT_CHUNK:
            await self._send_with_timeout(frame)
            return
        await self._send_with_timeout(json.dumps(event.to_wire()))

    async def speak(self, event: VoiceEvent) -> None:
        await self.send_event(event)

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
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._ws is not None:
            await self._ws.close()
        await put_realtime_voice_event(self._events, None)

    async def _send_with_timeout(self, payload: Any) -> None:
        try:
            await asyncio.wait_for(
                self._ws.send(payload),
                timeout=REALTIME_VOICE_SIDECAR_SEND_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "realtime voice sidecar send timed out after "
                f"{REALTIME_VOICE_SIDECAR_SEND_TIMEOUT_SECONDS:g}s"
            ) from exc

    async def _read_events(self) -> None:
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    try:
                        await put_realtime_voice_event(
                            self._events,
                            event_from_binary_output_audio_frame(raw),
                        )
                        continue
                    except Exception:
                        pass
                    await put_realtime_voice_event(
                        self._events,
                        VoiceEvent(
                            type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
                            session_id=self.config.session_id if self.config else "",
                            sequence=0,
                            payload={
                                "codec": VoiceAudioCodec.OPUS.value,
                                "sample_rate_hz": 16000,
                                "channels": 1,
                                "data_b64": base64.b64encode(raw).decode("ascii"),
                            },
                        )
                    )
                    continue
                try:
                    event = VoiceEvent.from_wire(json.loads(raw))
                except Exception as exc:
                    await put_realtime_voice_event(
                        self._events,
                        VoiceEvent(
                            type=VoiceEventType.SESSION_ERROR,
                            session_id=self.config.session_id if self.config else "",
                            sequence=0,
                            payload={"error": f"invalid sidecar event: {sanitize_realtime_voice_error(exc)}"},
                        )
                    )
                    continue
                await put_realtime_voice_event(self._events, event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await put_realtime_voice_event(
                self._events,
                VoiceEvent(
                    type=VoiceEventType.SESSION_ERROR,
                    session_id=self.config.session_id if self.config else "",
                    sequence=0,
                    payload={"error": f"sidecar closed: {sanitize_realtime_voice_error(exc)}"},
                )
            )


def wants_realtime_sidecar(config: RealtimeVoiceSessionConfig) -> bool:
    provider = (config.frontend_provider or "").strip().lower()
    if not config.effective_sidecar_base_url:
        return False
    return provider in {
        "sidecar",
        "reference",
        "local",
        "provider",
        "gemma",
        "gemma4",
        "vllm",
        "lmstudio",
    } or bool(config.frontend_model)


def sidecar_ws_url(base_url: str, path: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    root = parsed.path if parsed.netloc else ""
    return urllib.parse.urlunparse((scheme, netloc, f"{root.rstrip('/')}{path}", "", "", ""))
