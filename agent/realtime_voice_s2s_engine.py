"""Native speech-to-speech sidecar engine for realtime Hermes voice."""

from __future__ import annotations

import asyncio
import contextlib
import json
import urllib.parse
from typing import Any, AsyncIterator, Optional

from agent.realtime_voice import (
    AudioChunk,
    REALTIME_VOICE_SIDECAR_SEND_TIMEOUT_SECONDS,
    RealtimeVoiceEngine,
    RealtimeVoiceEngineKind,
    RealtimeVoiceSessionConfig,
    VoiceAudioCodec,
    VoiceEvent,
    VoiceEventType,
    binary_audio_frame_from_event,
    create_realtime_voice_event_queue,
    event_from_binary_audio_frame,
    put_realtime_voice_event,
    realtime_voice_session_contract_payload,
    transcript_event_payload_from_payload,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_oracle import HermesRealtimeOracle


STALE_SIDECAR_GENERATION_EVENT_TYPES = frozenset(
    {
        VoiceEventType.AUDIO_OUTPUT_CHUNK,
        VoiceEventType.ASSISTANT_COMMIT,
        VoiceEventType.ASSISTANT_TEXT_PARTIAL,
        VoiceEventType.TRANSCRIPT_FINAL,
    }
)


class NativeS2SSidecarEngine(RealtimeVoiceEngine):
    """Bridge browser voice events to a native S2S inference sidecar."""

    def __init__(self):
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._events: asyncio.Queue[VoiceEvent | None] = create_realtime_voice_event_queue()
        self._sequence = 0
        self._closed = False
        self._ws: Any = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._oracle: Optional[HermesRealtimeOracle] = None
        self._oracle_hint_task: Optional[asyncio.Task[None]] = None
        self._playback_generation = 0
        self._assistant_output_active = False
        self._auto_barge_in_input_active = False

    @property
    def kind(self) -> RealtimeVoiceEngineKind:
        return RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        if not config.effective_sidecar_base_url:
            raise RuntimeError("native S2S engine requires voice.realtime.sidecar_base_url")
        self.config = config
        self._oracle = HermesRealtimeOracle(config)
        await self._connect_sidecar(config)
        await self._emit(
            VoiceEventType.SESSION_STARTED,
            {
                "engine": self.kind.value,
                "input_codec": config.input_codec.value,
                "output_codec": config.output_codec.value,
                "frontend_provider": config.frontend_provider or "",
                "frontend_model": config.frontend_model or "",
                "sidecar": True,
                **realtime_voice_session_contract_payload(config),
            },
        )

    async def receive_event(self, event: VoiceEvent) -> None:
        if self._closed:
            return
        if event.type == VoiceEventType.SESSION_CLOSED:
            await self.close()
            return
        if event.type == VoiceEventType.AUDIO_INPUT_CHUNK:
            await self._auto_barge_in_for_speech(event)
        if event.type == VoiceEventType.BARGE_IN:
            event = await self._interrupt_active_turn(event, reason=str(event.payload.get("reason") or "client"))
        await self._send_sidecar_event(event)
        if event.type == VoiceEventType.AUDIO_INPUT_CHUNK and event.payload.get("end_of_utterance") is True:
            self._auto_barge_in_input_active = False

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
        await put_realtime_voice_event(self._events, None)

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
        await self._send_ws_with_timeout(json.dumps({"type": "session.config", "payload": config.to_wire()}))
        self._reader_task = asyncio.create_task(self._read_sidecar())

    async def _auto_barge_in_for_speech(self, event: VoiceEvent) -> None:
        if self._auto_barge_in_input_active:
            return
        if not _payload_has_audio(event.payload):
            return
        if not self._has_active_generation_work():
            return

        self._auto_barge_in_input_active = True
        barge_in = await self._interrupt_active_turn(event, reason="user_speech")
        await self._send_sidecar_event(barge_in)

    async def _interrupt_active_turn(self, event: VoiceEvent, *, reason: str) -> VoiceEvent:
        self._playback_generation += 1
        self._assistant_output_active = False
        self._cancel_oracle_hint("Realtime voice native S2S turn interrupted")
        payload = {
            **dict(event.payload),
            "reason": reason or "client",
            "playback_generation": self._playback_generation,
        }
        forwarded = VoiceEvent(
            type=VoiceEventType.BARGE_IN,
            session_id=event.session_id,
            sequence=event.sequence,
            timestamp_ms=event.timestamp_ms,
            payload=payload,
        )
        await self._emit(
            VoiceEventType.BARGE_IN,
            {
                "reason": payload["reason"],
                "playback_generation": self._playback_generation,
            },
        )
        return forwarded

    def _has_active_generation_work(self) -> bool:
        return self._assistant_output_active or bool(self._oracle_hint_task and not self._oracle_hint_task.done())

    async def _send_sidecar_event(self, event: VoiceEvent) -> None:
        if self._ws is None:
            return
        frame = binary_audio_frame_from_event(event)
        if frame is not None and event.type == VoiceEventType.AUDIO_INPUT_CHUNK:
            await self._send_ws_with_timeout(frame)
            return
        await self._send_ws_with_timeout(json.dumps(event.to_wire()))

    async def _read_sidecar(self) -> None:
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    try:
                        event = event_from_binary_audio_frame(raw, expected_type=VoiceEventType.AUDIO_OUTPUT_CHUNK)
                    except Exception:
                        payload = AudioChunk(codec=VoiceAudioCodec.OPUS, data=raw).to_payload()
                        if self._playback_generation:
                            payload["playback_generation"] = self._playback_generation
                        await self._emit(VoiceEventType.AUDIO_OUTPUT_CHUNK, payload)
                        continue
                    if self._is_stale_sidecar_event(event):
                        continue
                    event = self._normalize_sidecar_event(event)
                    await self._queue_sidecar_event(event)
                    continue
                try:
                    event = VoiceEvent.from_wire(json.loads(raw))
                except Exception:
                    await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid sidecar event"})
                    continue
                if self._is_stale_sidecar_event(event):
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
            if self._closed:
                return
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
        await put_realtime_voice_event(self._events, event)
        return event

    async def _queue_sidecar_event(self, event: VoiceEvent) -> None:
        self._track_sidecar_output_state(event)
        await self._emit(event.type, dict(event.payload))

    def _track_sidecar_output_state(self, event: VoiceEvent) -> None:
        if event.type in {VoiceEventType.AUDIO_OUTPUT_CHUNK, VoiceEventType.ASSISTANT_TEXT_PARTIAL}:
            self._assistant_output_active = True
        elif event.type == VoiceEventType.TRANSCRIPT_FINAL:
            self._auto_barge_in_input_active = False
        elif event.type in {VoiceEventType.ASSISTANT_COMMIT, VoiceEventType.BARGE_IN, VoiceEventType.SESSION_ERROR}:
            self._assistant_output_active = False

    def _is_stale_sidecar_event(self, event: VoiceEvent) -> bool:
        if event.type not in STALE_SIDECAR_GENERATION_EVENT_TYPES:
            return False
        generation = _payload_generation(dict(event.payload))
        return generation is not None and generation < self._playback_generation

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

        if event.type in {VoiceEventType.TRANSCRIPT_PARTIAL, VoiceEventType.TRANSCRIPT_FINAL}:
            payload = transcript_event_payload_from_payload(payload)

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
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "oracle_hint_failed",
                    "error": sanitize_realtime_voice_error(exc),
                    "sidecar": True,
                },
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
            await self._send_ws_with_timeout(json.dumps(event.to_wire()))

    async def _send_ws_with_timeout(self, payload: Any) -> None:
        try:
            await asyncio.wait_for(
                self._ws.send(payload),
                timeout=REALTIME_VOICE_SIDECAR_SEND_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "native S2S sidecar send timed out after "
                f"{REALTIME_VOICE_SIDECAR_SEND_TIMEOUT_SECONDS:g}s"
            ) from exc


def _sidecar_ws_url(base_url: str, path: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    root = parsed.path if parsed.netloc else ""
    return urllib.parse.urlunparse((scheme, netloc, f"{root.rstrip('/')}{path}", "", "", ""))


def _payload_generation(payload: dict) -> Optional[int]:
    value = payload.get("playback_generation")
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _payload_has_audio(payload: dict) -> bool:
    data = payload.get("data_b64")
    if isinstance(data, str) and data:
        return True
    data_bytes = payload.get("data_bytes")
    return isinstance(data_bytes, (bytes, bytearray, memoryview)) and len(data_bytes) > 0
