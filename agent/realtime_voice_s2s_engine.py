"""Native speech-to-speech sidecar engine for realtime Hermes voice."""

from __future__ import annotations

import asyncio
import contextlib
import json
import urllib.parse
from typing import Any, AsyncIterator, Optional

from agent.realtime_voice import (
    RealtimeVoiceEngine,
    RealtimeVoiceEngineKind,
    RealtimeVoiceSessionConfig,
    VoiceEvent,
    VoiceEventType,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_oracle import HermesRealtimeOracle


class NativeS2SSidecarEngine(RealtimeVoiceEngine):
    """Bridge browser voice events to a native S2S inference sidecar."""

    def __init__(self):
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._events: asyncio.Queue[VoiceEvent | None] = asyncio.Queue()
        self._sequence = 0
        self._closed = False
        self._ws: Any = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._oracle: Optional[HermesRealtimeOracle] = None
        self._oracle_hint_task: Optional[asyncio.Task[None]] = None
        self._playback_generation = 0

    @property
    def kind(self) -> RealtimeVoiceEngineKind:
        return RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        if not config.effective_sidecar_base_url:
            raise RuntimeError("native S2S engine requires voice.realtime.sidecar_base_url")
        self.config = config
        self._oracle = HermesRealtimeOracle(config)
        await self._connect_sidecar(config)
        await self._emit(VoiceEventType.SESSION_STARTED, {"engine": self.kind.value})

    async def receive_event(self, event: VoiceEvent) -> None:
        if self._closed:
            return
        if event.type == VoiceEventType.SESSION_CLOSED:
            await self.close()
            return
        if event.type == VoiceEventType.BARGE_IN:
            self._playback_generation += 1
            self._cancel_oracle_hint("Realtime voice native S2S turn interrupted")
            event = VoiceEvent(
                type=event.type,
                session_id=event.session_id,
                sequence=event.sequence,
                timestamp_ms=event.timestamp_ms,
                payload={
                    **dict(event.payload),
                    "reason": event.payload.get("reason") or "client",
                    "playback_generation": self._playback_generation,
                },
            )
            await self._emit(
                VoiceEventType.BARGE_IN,
                {
                    "reason": event.payload.get("reason") or "client",
                    "playback_generation": self._playback_generation,
                },
            )
        if self._ws is not None:
            await self._ws.send(json.dumps(event.to_wire()))

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
        self._cancel_oracle_hint()
        if self._reader_task:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        if self._ws is not None:
            await self._ws.close()
        if self._oracle_hint_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._oracle_hint_task
        await self._emit(VoiceEventType.SESSION_CLOSED, {"reason": "closed"})
        await self._events.put(None)

    async def _connect_sidecar(self, config: RealtimeVoiceSessionConfig) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("native S2S sidecar requires the websockets package") from exc

        url = _sidecar_ws_url(config.effective_sidecar_base_url or "", "/v1/s2s/session")
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
            raise RuntimeError(f"native S2S sidecar connect timed out after {timeout:g}s") from exc
        await self._ws.send(json.dumps({"type": "session.config", "payload": config.to_wire()}))
        self._reader_task = asyncio.create_task(self._read_sidecar())

    async def _read_sidecar(self) -> None:
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    payload = {
                        "codec": "opus",
                        "sample_rate_hz": 16000,
                        "channels": 1,
                        "data_b64": _b64(raw),
                    }
                    if self._playback_generation:
                        payload["playback_generation"] = self._playback_generation
                    await self._emit(
                        VoiceEventType.AUDIO_OUTPUT_CHUNK,
                        payload,
                    )
                    continue
                try:
                    event = VoiceEvent.from_wire(json.loads(raw))
                except Exception:
                    await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid sidecar event"})
                    continue
                if event.type == VoiceEventType.TRANSCRIPT_FINAL:
                    event = self._normalize_sidecar_event(event, new_generation=True)
                    self._start_oracle_hint(
                        str(event.payload.get("text") or ""),
                        _payload_generation(event.payload),
                    )
                else:
                    event = self._normalize_sidecar_event(event)
                await self._queue_sidecar_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"sidecar closed: {sanitize_realtime_voice_error(exc)}"},
            )

    async def _emit(self, event_type: VoiceEventType, payload: dict) -> Optional[VoiceEvent]:
        if self.config is None:
            return None
        self._sequence += 1
        event = VoiceEvent(
            type=event_type,
            session_id=self.config.session_id,
            sequence=self._sequence,
            payload=payload,
        )
        await self._events.put(event)
        return event

    async def _queue_sidecar_event(self, event: VoiceEvent) -> None:
        await self._emit(event.type, dict(event.payload))

    def _normalize_sidecar_event(self, event: VoiceEvent, *, new_generation: bool = False) -> VoiceEvent:
        payload = dict(event.payload)
        generation = _payload_generation(payload)

        if new_generation and generation is None:
            self._playback_generation += 1
            generation = self._playback_generation
            payload["playback_generation"] = generation
        elif generation is not None:
            self._playback_generation = max(self._playback_generation, generation)
        elif event.type in {
            VoiceEventType.AUDIO_OUTPUT_CHUNK,
            VoiceEventType.ASSISTANT_TEXT_PARTIAL,
            VoiceEventType.ASSISTANT_COMMIT,
        } and self._playback_generation:
            payload["playback_generation"] = self._playback_generation

        return VoiceEvent(
            type=event.type,
            session_id=self.config.session_id if self.config else event.session_id,
            sequence=event.sequence,
            timestamp_ms=event.timestamp_ms,
            payload=payload,
        )

    def _start_oracle_hint(self, transcript: str, playback_generation: Optional[int]) -> None:
        self._cancel_oracle_hint("Realtime voice native S2S turn superseded")
        task = asyncio.create_task(self._send_oracle_hint(transcript, playback_generation))
        self._oracle_hint_task = task
        task.add_done_callback(self._clear_oracle_hint_task)

    def _clear_oracle_hint_task(self, task: asyncio.Task[None]) -> None:
        if self._oracle_hint_task is task:
            self._oracle_hint_task = None

    def _cancel_oracle_hint(self, message: str = "Realtime voice turn interrupted") -> None:
        if self._oracle_hint_task and not self._oracle_hint_task.done():
            self._oracle_hint_task.cancel()
        if self._oracle is not None:
            with contextlib.suppress(Exception):
                self._oracle.interrupt(message)

    async def _send_oracle_hint(self, transcript: str, playback_generation: Optional[int] = None) -> None:
        if not transcript.strip() or self._oracle is None or self._ws is None:
            return
        try:
            full_text = ""
            sent_any = False
            async for delta in self._oracle.stream_answer(transcript):
                if not delta:
                    continue
                full_text += str(delta)
                sent_any = True
                await self._send_oracle_hint_event(
                    text=full_text,
                    delta=str(delta),
                    final=False,
                    playback_generation=playback_generation,
                )
            if not sent_any:
                return
            await self._send_oracle_hint_event(
                text=full_text,
                delta="",
                final=True,
                playback_generation=playback_generation,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"oracle hint failed: {sanitize_realtime_voice_error(exc)}"},
            )

    async def _send_oracle_hint_event(
        self,
        *,
        text: str,
        delta: str,
        final: bool,
        playback_generation: Optional[int],
    ) -> None:
        payload = {
            "text": text,
            "delta": delta,
            "final": final,
            "source": "hermes",
        }
        if playback_generation is not None:
            payload["playback_generation"] = playback_generation
        event = await self._emit(VoiceEventType.ORACLE_HINT, payload)
        if event is not None and self._ws is not None:
            await self._ws.send(json.dumps(event.to_wire()))


def _sidecar_ws_url(base_url: str, path: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    root = parsed.path if parsed.netloc else ""
    return urllib.parse.urlunparse((scheme, netloc, f"{root.rstrip('/')}{path}", "", "", ""))


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


def _payload_generation(payload: dict) -> Optional[int]:
    value = payload.get("playback_generation")
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
