"""Text-oracle + TTS realtime voice engine."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
from typing import AsyncIterator, List, Optional

from agent.realtime_voice import (
    AudioChunk,
    RealtimeVoiceEngine,
    RealtimeVoiceEngineKind,
    RealtimeVoiceSessionConfig,
    VoiceAudioCodec,
    VoiceEvent,
    VoiceEventType,
    create_realtime_voice_event_queue,
    put_realtime_voice_event,
    realtime_voice_session_contract_payload,
    transcript_event_payload_from_payload,
    transcript_metadata_from_payload,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_oracle import HermesRealtimeOracle, NullRealtimeOracle
from agent.realtime_voice_planner import RealtimeSpeechPlanner
from agent.realtime_voice_sidecar import RealtimeVoiceSidecarClient, wants_realtime_sidecar


class TextOracleTTSEngine(RealtimeVoiceEngine):
    """Realtime engine backed by STT, the Hermes oracle, and TTS.

    The initial audio path buffers client audio frames until an
    ``end_of_utterance`` marker, then reuses Hermes' existing STT provider
    chain. Browser clients may also send a trusted ``transcript`` in the audio
    event payload for tests or Web Speech API experiments.
    """

    def __init__(self, *, oracle: Optional[object] = None, sidecar: Optional[object] = None):
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._events: asyncio.Queue[VoiceEvent | None] = create_realtime_voice_event_queue()
        self._inbound_audio: List[bytes] = []
        self._inbound_audio_bytes = 0
        self._sequence = 0
        self._closed = False
        self._active_task: Optional[asyncio.Task[None]] = None
        self._planner = RealtimeSpeechPlanner()
        self._oracle = oracle
        self._sidecar = sidecar
        self._sidecar_task: Optional[asyncio.Task[None]] = None
        self._playback_generation = 0
        self._pending_turn_generation: Optional[int] = None
        self._input_generation = 0
        self._input_generation_active = False

    @property
    def kind(self) -> RealtimeVoiceEngineKind:
        return RealtimeVoiceEngineKind.TEXT_ORACLE_TTS

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        self.config = config
        if self._oracle is None:
            self._oracle = HermesRealtimeOracle(config)
        if self._sidecar is None and wants_realtime_sidecar(config):
            self._sidecar = RealtimeVoiceSidecarClient()
        if self._sidecar is not None:
            try:
                await self._sidecar.start(config)  # type: ignore[attr-defined]
                self._sidecar_task = asyncio.create_task(self._consume_sidecar_events())
            except Exception as exc:
                await self._disable_sidecar()
                await self._emit(
                    VoiceEventType.FRONTEND_STATE,
                    {
                        "status": "fallback",
                        "reason": "sidecar_unavailable",
                        "error": sanitize_realtime_voice_error(exc),
                        "sidecar": False,
                    },
                )
        await self._emit(
            VoiceEventType.SESSION_STARTED,
            {
                "engine": self.kind.value,
                "input_codec": config.input_codec.value,
                "output_codec": config.output_codec.value,
                "frontend_provider": config.frontend_provider or "",
                "frontend_model": config.frontend_model or "",
                "sidecar": self._sidecar is not None,
                **realtime_voice_session_contract_payload(config),
            },
        )

    async def receive_event(self, event: VoiceEvent) -> None:
        if self._closed:
            return
        if event.type == VoiceEventType.BARGE_IN:
            await self._interrupt_active_turn(event, reason=str(event.payload.get("reason") or "client"))
            return
        if event.type == VoiceEventType.SESSION_CLOSED:
            await self.close()
            return
        if event.type != VoiceEventType.AUDIO_INPUT_CHUNK:
            return

        transcript = str(event.payload.get("transcript") or "").strip()
        if transcript:
            await self._auto_barge_in_for_speech(event)
            if not _payload_marks_final_transcript(event.payload):
                await self._emit(
                    VoiceEventType.TRANSCRIPT_PARTIAL,
                    {
                        "text": transcript,
                        "stability": 0.8,
                        **transcript_metadata_from_payload(event.payload),
                    },
                )
                return
            await self._start_turn(transcript, metadata=transcript_metadata_from_payload(event.payload))
            return

        try:
            chunk = AudioChunk.from_payload(event.payload)
            if chunk.data:
                await self._auto_barge_in_for_speech(event)
            if self._sidecar is not None:
                sidecar_event = self._sidecar_input_event(event)
                if await self._send_sidecar_event(sidecar_event):
                    self._finish_input_generation_if_needed(sidecar_event)
                    return
            if not await self._append_inbound_audio(chunk.data):
                return
        except Exception:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid audio chunk"})
            return

        if event.payload.get("end_of_utterance") is True:
            self._finish_input_generation_if_needed(event)
            audio = b"".join(self._inbound_audio)
            self._clear_inbound_audio()
            if audio:
                await self._emit(VoiceEventType.TRANSCRIPT_PARTIAL, {"text": "", "stability": 0.1})
                self._active_task = asyncio.create_task(self._transcribe_and_answer(audio, chunk.codec))

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
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._active_task
        if self._sidecar_task and not self._sidecar_task.done():
            self._sidecar_task.cancel()
        if self._sidecar is not None:
            try:
                await self._sidecar.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        if self._sidecar_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._sidecar_task
        await self._emit(VoiceEventType.SESSION_CLOSED, {"reason": "closed"})
        await put_realtime_voice_event(self._events, None)

    async def _auto_barge_in_for_speech(self, event: VoiceEvent) -> None:
        if self._pending_turn_generation is not None:
            return
        if self._active_task is None or self._active_task.done():
            return
        await self._interrupt_active_turn(event, reason="user_speech")

    async def _interrupt_active_turn(self, event: VoiceEvent, *, reason: str) -> None:
        self._playback_generation += 1
        self._pending_turn_generation = self._playback_generation
        self._input_generation += 1
        self._input_generation_active = False
        payload = {
            "reason": reason or "client",
            "playback_generation": self._playback_generation,
        }
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
        oracle = self._oracle
        if hasattr(oracle, "interrupt"):
            oracle.interrupt("Realtime voice barge-in")  # type: ignore[attr-defined]
        self._clear_inbound_audio()
        await self._emit(VoiceEventType.BARGE_IN, payload)
        if self._sidecar is not None:
            await self._send_sidecar_event(
                VoiceEvent(
                    type=VoiceEventType.BARGE_IN,
                    session_id=event.session_id,
                    sequence=event.sequence,
                    timestamp_ms=event.timestamp_ms,
                    payload=payload,
                )
            )

    async def _consume_sidecar_events(self) -> None:
        if self._sidecar is None:
            return
        try:
            async for event in self._sidecar.events():  # type: ignore[attr-defined]
                if event.type == VoiceEventType.TRANSCRIPT_PARTIAL:
                    payload = dict(event.payload)
                    if self._is_stale_sidecar_input(payload):
                        continue
                    await self._emit(VoiceEventType.TRANSCRIPT_PARTIAL, transcript_event_payload_from_payload(payload))
                elif event.type == VoiceEventType.TRANSCRIPT_FINAL:
                    payload = dict(event.payload)
                    if self._is_stale_sidecar_input(payload):
                        continue
                    text = str(payload.get("text") or "").strip()
                    if text:
                        await self._start_turn(
                            text,
                            input_generation=_payload_input_generation(payload),
                            metadata=transcript_metadata_from_payload(payload),
                        )
                elif event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
                    payload = dict(event.payload)
                    generation = _payload_generation(payload)
                    if generation is not None and generation < self._playback_generation:
                        continue
                    payload.setdefault("playback_generation", self._playback_generation)
                    await self._emit(VoiceEventType.AUDIO_OUTPUT_CHUNK, payload)
                elif event.type == VoiceEventType.FRONTEND_STATE:
                    await self._emit(VoiceEventType.FRONTEND_STATE, dict(event.payload))
                elif event.type == VoiceEventType.SESSION_ERROR:
                    await self._disable_sidecar()
                    await self._emit(
                        VoiceEventType.FRONTEND_STATE,
                        {
                            "status": "fallback",
                            "reason": "sidecar_session_error",
                            "error": sanitize_realtime_voice_error(event.payload.get("error") or ""),
                            "sidecar": False,
                        },
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._disable_sidecar()
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "sidecar_event_stream_failed",
                    "error": sanitize_realtime_voice_error(exc),
                    "sidecar": False,
                },
            )

    async def _send_sidecar_event(self, event: VoiceEvent) -> bool:
        if self._sidecar is None:
            return False
        try:
            await self._sidecar.send_event(event)  # type: ignore[attr-defined]
            return True
        except Exception as exc:
            await self._disable_sidecar()
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "fallback",
                    "reason": "sidecar_send_failed",
                    "error": sanitize_realtime_voice_error(exc),
                    "sidecar": False,
                },
            )
            return False

    async def _disable_sidecar(self) -> None:
        sidecar = self._sidecar
        self._sidecar = None
        if sidecar is None:
            return
        close = getattr(sidecar, "close", None)
        if close is None:
            return
        with contextlib.suppress(Exception):
            result = close()
            if asyncio.iscoroutine(result):
                await result

    def _sidecar_input_event(self, event: VoiceEvent) -> VoiceEvent:
        if not self._input_generation_active:
            self._input_generation += 1
            self._input_generation_active = True
        payload = dict(event.payload)
        payload["input_generation"] = self._input_generation
        return VoiceEvent(
            type=event.type,
            session_id=event.session_id,
            sequence=event.sequence,
            timestamp_ms=event.timestamp_ms,
            payload=payload,
        )

    def _finish_input_generation_if_needed(self, event: VoiceEvent) -> None:
        if event.payload.get("end_of_utterance") is True:
            self._input_generation_active = False

    def _is_stale_sidecar_input(self, payload: dict) -> bool:
        generation = _payload_input_generation(payload)
        return generation is not None and generation < self._input_generation

    async def _append_inbound_audio(self, data: bytes) -> bool:
        config = self.config
        limit = int(config.input_buffer_limit_bytes if config is not None else 8 * 1024 * 1024)
        if self._inbound_audio_bytes + len(data) > max(1, limit):
            self._clear_inbound_audio()
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "input_buffer_limit_exceeded",
                    "sidecar": False,
                    "limit_bytes": limit,
                },
            )
            return False
        self._inbound_audio.append(data)
        self._inbound_audio_bytes += len(data)
        return True

    def _clear_inbound_audio(self) -> None:
        self._inbound_audio.clear()
        self._inbound_audio_bytes = 0

    async def _transcribe_and_answer(self, audio: bytes, codec: VoiceAudioCodec) -> None:
        try:
            transcript = await asyncio.to_thread(self._transcribe_sync, audio, codec)
            if transcript:
                await self._start_turn(transcript)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"transcription failed: {sanitize_realtime_voice_error(exc)}"},
            )

    async def _start_turn(
        self,
        transcript: str,
        *,
        input_generation: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
        if self._pending_turn_generation is not None:
            generation = self._pending_turn_generation
            self._pending_turn_generation = None
        else:
            self._playback_generation += 1
            generation = self._playback_generation
        payload = {"text": transcript, "playback_generation": generation}
        if input_generation is not None:
            payload["input_generation"] = input_generation
        if metadata:
            payload.update(metadata)
        await self._emit(VoiceEventType.TRANSCRIPT_FINAL, payload)
        self._active_task = asyncio.create_task(self._answer_and_speak(transcript, generation, metadata or {}))

    async def _answer_and_speak(self, transcript: str, playback_generation: int, metadata: dict) -> None:
        speak_tasks: List[asyncio.Task[None]] = []
        speak_chain: Optional[asyncio.Task[None]] = None
        assistant_metadata = dict(metadata)
        tts_error_reported = False

        def queue_speak(text: str) -> None:
            nonlocal speak_chain, tts_error_reported
            previous = speak_chain

            async def report_tts_failure(exc: Exception) -> None:
                nonlocal tts_error_reported
                if tts_error_reported:
                    return
                tts_error_reported = True
                await self._emit(
                    VoiceEventType.FRONTEND_STATE,
                    {
                        "status": "degraded",
                        "reason": "tts_failed",
                        "error": sanitize_realtime_voice_error(exc),
                        "sidecar": False,
                    },
                )

            async def speak_after_previous() -> None:
                if previous is not None:
                    await previous
                if tts_error_reported:
                    return
                if playback_generation == self._playback_generation:
                    try:
                        await self._speak_chunk(text, playback_generation)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        await report_tts_failure(exc)

            speak_chain = asyncio.create_task(speak_after_previous())
            speak_tasks.append(speak_chain)

        async def cancel_speak_tasks() -> None:
            for task in speak_tasks:
                task.cancel()
            for task in speak_tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

        try:
            oracle = self._oracle or NullRealtimeOracle()
            answer = ""
            buffer = ""
            async for delta in _stream_oracle_answer(oracle, transcript, assistant_metadata):
                if playback_generation != self._playback_generation:
                    return
                answer += delta
                buffer += delta
                chunk, buffer = _take_speakable_chunk(buffer)
                if chunk:
                    planned_chunk = self._planner.clean(chunk)
                    if planned_chunk:
                        await self._emit(
                            VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                            {
                                "text": planned_chunk,
                                "playback_generation": playback_generation,
                                **assistant_metadata,
                            },
                        )
                        queue_speak(planned_chunk)

            if buffer.strip():
                planned_chunk = self._planner.clean(buffer)
                if planned_chunk:
                    await self._emit(
                        VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                        {
                            "text": planned_chunk,
                            "playback_generation": playback_generation,
                            **assistant_metadata,
                        },
                    )
                    queue_speak(planned_chunk)

            plan = self._planner.plan(answer)
            if speak_chain is not None:
                await speak_chain
            if not plan.committed_text:
                return
            if playback_generation == self._playback_generation:
                await self._emit(
                    VoiceEventType.ASSISTANT_COMMIT,
                    {
                        "text": plan.committed_text,
                        "playback_generation": playback_generation,
                        **assistant_metadata,
                    },
                )
        except asyncio.CancelledError:
            await cancel_speak_tasks()
            if not self._closed and playback_generation == self._playback_generation:
                await self._emit(
                    VoiceEventType.ASSISTANT_COMMIT,
                    {"interrupted": True, "text": "", "playback_generation": playback_generation},
                )
            raise
        except Exception as exc:
            await cancel_speak_tasks()
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"oracle/tts failed: {sanitize_realtime_voice_error(exc)}"},
            )

    def _transcribe_sync(self, audio: bytes, codec: VoiceAudioCodec) -> str:
        from tools.transcription_tools import transcribe_audio

        suffix = {
            VoiceAudioCodec.PCM16: ".wav",
            VoiceAudioCodec.OPUS: ".ogg",
            VoiceAudioCodec.WEBM_OPUS: ".webm",
        }.get(codec, ".webm")
        path = ""
        try:
            with tempfile.NamedTemporaryFile(prefix="hermes-realtime-voice-", suffix=suffix, delete=False) as tmp:
                tmp.write(audio)
                path = tmp.name
            result = transcribe_audio(path)
            if not result.get("success"):
                raise RuntimeError(result.get("error") or "transcription failed")
            return str(result.get("transcript") or "").strip()
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    async def _speak_chunk(self, text: str, playback_generation: int) -> None:
        if playback_generation != self._playback_generation:
            return
        if self._sidecar is not None and self.config is not None:
            event = VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id=self.config.session_id,
                sequence=self._sequence + 1,
                payload={"text": text, "speak": True, "playback_generation": playback_generation},
            )
            try:
                await self._sidecar.speak(event)  # type: ignore[attr-defined]
                return
            except Exception as exc:
                await self._disable_sidecar()
                await self._emit(
                    VoiceEventType.FRONTEND_STATE,
                    {
                        "status": "fallback",
                        "reason": "sidecar_tts_failed",
                        "error": sanitize_realtime_voice_error(exc),
                        "sidecar": False,
                    },
                )

        file_path = await asyncio.to_thread(self._tts_sync, text)
        if playback_generation != self._playback_generation:
            if file_path:
                with contextlib.suppress(OSError):
                    os.unlink(file_path)
            return
        if not file_path:
            return
        try:
            with open(file_path, "rb") as fh:
                data = fh.read()
            if data:
                payload = AudioChunk(codec=VoiceAudioCodec.OPUS, data=data).to_payload()
                payload["mime_type"] = _mime_type_for_path(file_path)
                payload["playback_generation"] = playback_generation
                await self._emit(
                    VoiceEventType.AUDIO_OUTPUT_CHUNK,
                    payload,
                )
        finally:
            try:
                os.unlink(file_path)
            except OSError:
                pass

    def _tts_sync(self, text: str) -> str:
        from tools.tts_tool import text_to_speech_tool

        raw = text_to_speech_tool(text)
        result = json.loads(raw) if isinstance(raw, str) else raw
        if not result.get("success"):
            raise RuntimeError(result.get("error") or "speech synthesis failed")
        return str(result.get("file_path") or "")

    async def _emit(self, event_type: VoiceEventType, payload: dict) -> None:
        if self.config is None:
            return
        self._sequence += 1
        await put_realtime_voice_event(
            self._events,
            VoiceEvent(
                type=event_type,
                session_id=self.config.session_id,
                sequence=self._sequence,
                payload=payload,
            )
        )


def _mime_type_for_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
    }.get(ext, "audio/mpeg")


def _payload_generation(payload: dict) -> Optional[int]:
    value = payload.get("playback_generation")
    return _payload_int(value)


def _payload_input_generation(payload: dict) -> Optional[int]:
    value = payload.get("input_generation")
    return _payload_int(value)


async def _stream_oracle_answer(oracle: object, transcript: str, metadata: dict) -> AsyncIterator[str]:
    metadata_stream = getattr(oracle, "stream_answer_with_metadata", None)
    if callable(metadata_stream):
        async for delta in metadata_stream(transcript, metadata):  # type: ignore[misc]
            yield delta
        return

    async for delta in oracle.stream_answer(transcript):  # type: ignore[attr-defined]
        yield delta


def _payload_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _payload_marks_final_transcript(payload: dict) -> bool:
    if "end_of_utterance" in payload:
        return payload.get("end_of_utterance") is True
    if "final" in payload:
        return payload.get("final") is True
    if "is_final" in payload:
        return payload.get("is_final") is True
    return True


_SENTENCE_BOUNDARY_CHARS = frozenset(".!?。！？؟।")
_PHRASE_BOUNDARY_CHARS = frozenset(",;:，、；：،؛")


def _take_speakable_chunk(buffer: str) -> tuple[Optional[str], str]:
    normalized = " ".join((buffer or "").split())
    if not normalized:
        return None, ""

    sentence_at = _find_delimiter(normalized, _SENTENCE_BOUNDARY_CHARS, start=8, end=260)
    if sentence_at >= 0:
        return normalized[: sentence_at + 1].strip(), normalized[sentence_at + 1 :].strip()

    has_whitespace = any(character.isspace() for character in normalized)
    phrase_min = 48 if has_whitespace else 16
    phrase_trigger = 96 if has_whitespace else 32
    phrase_end = 160 if has_whitespace else 96

    if len(normalized) >= phrase_trigger:
        split_at = _find_delimiter(normalized, _PHRASE_BOUNDARY_CHARS, start=phrase_min, end=phrase_end)
        if split_at >= phrase_min:
            return normalized[: split_at + 1].strip(), normalized[split_at + 1 :].strip()

        split_at = normalized.rfind(" ", 96, 160) if has_whitespace else -1
        if split_at >= 96:
            return normalized[:split_at].strip(), normalized[split_at:].strip()

    if len(normalized) > 220:
        split_at = _find_delimiter(normalized, _PHRASE_BOUNDARY_CHARS, start=0, end=220)
        split_at = max(split_at, normalized.rfind(" ", 0, 220))
        if split_at >= 80:
            suffix_start = split_at + 1 if normalized[split_at] in _PHRASE_BOUNDARY_CHARS else split_at
            return normalized[:suffix_start].strip(), normalized[suffix_start:].strip()

    return None, normalized


def _find_delimiter(text: str, delimiters: frozenset[str], *, start: int, end: int) -> int:
    upper = min(len(text), end)
    for index in range(upper - 1, max(-1, start - 1), -1):
        if text[index] in delimiters:
            return index
    return -1
