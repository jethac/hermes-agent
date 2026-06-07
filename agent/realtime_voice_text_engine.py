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
                await self._emit(
                    VoiceEventType.FRONTEND_STATE,
                    {
                        "status": "fallback",
                        "reason": "sidecar_unavailable",
                        "error": sanitize_realtime_voice_error(exc),
                        "sidecar": False,
                    },
                )
                self._sidecar = None
        await self._emit(
            VoiceEventType.SESSION_STARTED,
            {
                "engine": self.kind.value,
                "input_codec": config.input_codec.value,
                "output_codec": config.output_codec.value,
                "frontend_provider": config.frontend_provider or "",
                "frontend_model": config.frontend_model or "",
                "sidecar": self._sidecar is not None,
            },
        )

    async def receive_event(self, event: VoiceEvent) -> None:
        if self._closed:
            return
        if event.type == VoiceEventType.BARGE_IN:
            self._playback_generation += 1
            if self._active_task and not self._active_task.done():
                self._active_task.cancel()
            oracle = self._oracle
            if hasattr(oracle, "interrupt"):
                oracle.interrupt("Realtime voice barge-in")  # type: ignore[attr-defined]
            self._clear_inbound_audio()
            if self._sidecar is not None:
                await self._send_sidecar_event(event)
            await self._emit(
                VoiceEventType.BARGE_IN,
                {
                    "reason": event.payload.get("reason") or "client",
                    "playback_generation": self._playback_generation,
                },
            )
            return
        if event.type == VoiceEventType.SESSION_CLOSED:
            await self.close()
            return
        if event.type != VoiceEventType.AUDIO_INPUT_CHUNK:
            return

        transcript = str(event.payload.get("transcript") or "").strip()
        if transcript:
            if not _payload_marks_final_transcript(event.payload):
                await self._emit(VoiceEventType.TRANSCRIPT_PARTIAL, {"text": transcript, "stability": 0.8})
                return
            await self._start_turn(transcript)
            return

        try:
            chunk = AudioChunk.from_payload(event.payload)
            if self._sidecar is not None:
                if await self._send_sidecar_event(event):
                    return
            if not await self._append_inbound_audio(chunk.data):
                return
        except Exception:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid audio chunk"})
            return

        if event.payload.get("end_of_utterance") is True:
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

    async def _consume_sidecar_events(self) -> None:
        if self._sidecar is None:
            return
        try:
            async for event in self._sidecar.events():  # type: ignore[attr-defined]
                if event.type == VoiceEventType.TRANSCRIPT_PARTIAL:
                    await self._emit(VoiceEventType.TRANSCRIPT_PARTIAL, dict(event.payload))
                elif event.type == VoiceEventType.TRANSCRIPT_FINAL:
                    text = str(event.payload.get("text") or "").strip()
                    if text:
                        await self._start_turn(text)
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
                    await self._emit(VoiceEventType.SESSION_ERROR, dict(event.payload))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._sidecar = None
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
            self._sidecar = None
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

    async def _start_turn(self, transcript: str) -> None:
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
        self._playback_generation += 1
        generation = self._playback_generation
        await self._emit(VoiceEventType.TRANSCRIPT_FINAL, {"text": transcript, "playback_generation": generation})
        self._active_task = asyncio.create_task(self._answer_and_speak(transcript, generation))

    async def _answer_and_speak(self, transcript: str, playback_generation: int) -> None:
        try:
            oracle = self._oracle or NullRealtimeOracle()
            answer = ""
            buffer = ""
            async for delta in oracle.stream_answer(transcript):  # type: ignore[attr-defined]
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
                            {"text": planned_chunk, "playback_generation": playback_generation},
                        )
                        await self._speak_chunk(planned_chunk, playback_generation)

            if buffer.strip():
                planned_chunk = self._planner.clean(buffer)
                if planned_chunk:
                    await self._emit(
                        VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                        {"text": planned_chunk, "playback_generation": playback_generation},
                    )
                    await self._speak_chunk(planned_chunk, playback_generation)

            plan = self._planner.plan(answer)
            if not plan.committed_text:
                return
            if playback_generation == self._playback_generation:
                await self._emit(
                    VoiceEventType.ASSISTANT_COMMIT,
                    {"text": plan.committed_text, "playback_generation": playback_generation},
                )
        except asyncio.CancelledError:
            if playback_generation == self._playback_generation:
                await self._emit(
                    VoiceEventType.ASSISTANT_COMMIT,
                    {"interrupted": True, "text": "", "playback_generation": playback_generation},
                )
            raise
        except Exception as exc:
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
                self._sidecar = None
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


def _take_speakable_chunk(buffer: str) -> tuple[Optional[str], str]:
    normalized = " ".join((buffer or "").split())
    if not normalized:
        return None, ""

    import re

    match = re.match(r"^(.{8,260}?[.!?。！？])(?:\s+|$)", normalized)
    if match:
        chunk = match.group(1).strip()
        return chunk, normalized[len(match.group(1)):].strip()

    if len(normalized) > 260:
        split_at = max(
            normalized.rfind(", ", 0, 220),
            normalized.rfind("; ", 0, 220),
            normalized.rfind(": ", 0, 220),
            normalized.rfind(" ", 0, 220),
        )
        if split_at >= 80:
            return normalized[:split_at].strip(), normalized[split_at:].strip()

    return None, normalized
