"""Native speech-to-speech sidecar engine for realtime Hermes voice."""

from __future__ import annotations

import asyncio
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
            await self._emit(VoiceEventType.BARGE_IN, {"reason": event.payload.get("reason") or "client"})
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
        if self._reader_task:
            self._reader_task.cancel()
        if self._ws is not None:
            await self._ws.close()
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
        try:
            self._ws = await websockets.connect(url, additional_headers=headers or None)
        except TypeError:
            self._ws = await websockets.connect(url, extra_headers=headers or None)
        await self._ws.send(json.dumps({"type": "session.config", "payload": config.to_wire()}))
        self._reader_task = asyncio.create_task(self._read_sidecar())

    async def _read_sidecar(self) -> None:
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    await self._emit(
                        VoiceEventType.AUDIO_OUTPUT_CHUNK,
                        {"codec": "opus", "sample_rate_hz": 16000, "channels": 1, "data_b64": _b64(raw)},
                    )
                    continue
                try:
                    event = VoiceEvent.from_wire(json.loads(raw))
                except Exception:
                    await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid sidecar event"})
                    continue
                if event.type == VoiceEventType.TRANSCRIPT_FINAL:
                    asyncio.create_task(self._send_oracle_hint(str(event.payload.get("text") or "")))
                await self._events.put(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": f"sidecar closed: {exc}"})

    async def _emit(self, event_type: VoiceEventType, payload: dict) -> None:
        if self.config is None:
            return
        self._sequence += 1
        await self._events.put(
            VoiceEvent(
                type=event_type,
                session_id=self.config.session_id,
                sequence=self._sequence,
                payload=payload,
            )
        )

    async def _send_oracle_hint(self, transcript: str) -> None:
        if not transcript.strip() or self._oracle is None or self._ws is None:
            return
        try:
            answer = await self._oracle.answer(transcript)
            if not answer:
                return
            event = VoiceEvent(
                type=VoiceEventType.ORACLE_HINT,
                session_id=self.config.session_id if self.config else "",
                sequence=0,
                payload={"text": answer, "source": "hermes"},
            )
            await self._ws.send(json.dumps(event.to_wire()))
            await self._events.put(event)
        except Exception as exc:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": f"oracle hint failed: {exc}"})


def _sidecar_ws_url(base_url: str, path: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    root = parsed.path if parsed.netloc else ""
    return urllib.parse.urlunparse((scheme, netloc, f"{root.rstrip('/')}{path}", "", "", ""))


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")
