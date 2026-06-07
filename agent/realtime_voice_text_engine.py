"""Text-oracle + TTS realtime voice engine."""

from __future__ import annotations

import asyncio
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
)
from agent.realtime_voice_oracle import HermesRealtimeOracle, NullRealtimeOracle
from agent.realtime_voice_planner import RealtimeSpeechPlanner


class TextOracleTTSEngine(RealtimeVoiceEngine):
    """Realtime engine backed by STT, the Hermes oracle, and TTS.

    The initial audio path buffers client audio frames until an
    ``end_of_utterance`` marker, then reuses Hermes' existing STT provider
    chain. Browser clients may also send a trusted ``transcript`` in the audio
    event payload for tests or Web Speech API experiments.
    """

    def __init__(self, *, oracle: Optional[object] = None):
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._events: asyncio.Queue[VoiceEvent | None] = asyncio.Queue()
        self._inbound_audio: List[bytes] = []
        self._sequence = 0
        self._closed = False
        self._active_task: Optional[asyncio.Task[None]] = None
        self._planner = RealtimeSpeechPlanner()
        self._oracle = oracle

    @property
    def kind(self) -> RealtimeVoiceEngineKind:
        return RealtimeVoiceEngineKind.TEXT_ORACLE_TTS

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        self.config = config
        if self._oracle is None:
            self._oracle = HermesRealtimeOracle(config)
        await self._emit(
            VoiceEventType.SESSION_STARTED,
            {
                "engine": self.kind.value,
                "input_codec": config.input_codec.value,
                "output_codec": config.output_codec.value,
            },
        )

    async def receive_event(self, event: VoiceEvent) -> None:
        if self._closed:
            return
        if event.type == VoiceEventType.BARGE_IN:
            if self._active_task and not self._active_task.done():
                self._active_task.cancel()
            self._inbound_audio.clear()
            await self._emit(VoiceEventType.BARGE_IN, {"reason": event.payload.get("reason") or "client"})
            return
        if event.type == VoiceEventType.SESSION_CLOSED:
            await self.close()
            return
        if event.type != VoiceEventType.AUDIO_INPUT_CHUNK:
            return

        transcript = str(event.payload.get("transcript") or "").strip()
        if transcript:
            await self._start_turn(transcript)
            return

        try:
            chunk = AudioChunk.from_payload(event.payload)
            self._inbound_audio.append(chunk.data)
        except Exception:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid audio chunk"})
            return

        if event.payload.get("end_of_utterance") is True:
            audio = b"".join(self._inbound_audio)
            self._inbound_audio.clear()
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
        await self._emit(VoiceEventType.SESSION_CLOSED, {"reason": "closed"})
        await self._events.put(None)

    async def _transcribe_and_answer(self, audio: bytes, codec: VoiceAudioCodec) -> None:
        try:
            transcript = await asyncio.to_thread(self._transcribe_sync, audio, codec)
            if transcript:
                await self._start_turn(transcript)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": f"transcription failed: {exc}"})

    async def _start_turn(self, transcript: str) -> None:
        await self._emit(VoiceEventType.TRANSCRIPT_FINAL, {"text": transcript})
        self._active_task = asyncio.create_task(self._answer_and_speak(transcript))

    async def _answer_and_speak(self, transcript: str) -> None:
        try:
            oracle = self._oracle or NullRealtimeOracle()
            answer = ""
            buffer = ""
            async for delta in oracle.stream_answer(transcript):  # type: ignore[attr-defined]
                answer += delta
                buffer += delta
                chunk, buffer = _take_speakable_chunk(buffer)
                if chunk:
                    planned_chunk = self._planner.clean(chunk)
                    if planned_chunk:
                        await self._emit(VoiceEventType.ASSISTANT_TEXT_PARTIAL, {"text": planned_chunk})
                        await self._speak_chunk(planned_chunk)

            if buffer.strip():
                planned_chunk = self._planner.clean(buffer)
                if planned_chunk:
                    await self._emit(VoiceEventType.ASSISTANT_TEXT_PARTIAL, {"text": planned_chunk})
                    await self._speak_chunk(planned_chunk)

            plan = self._planner.plan(answer)
            if not plan.committed_text:
                return
            await self._emit(VoiceEventType.ASSISTANT_COMMIT, {"text": plan.committed_text})
        except asyncio.CancelledError:
            await self._emit(VoiceEventType.ASSISTANT_COMMIT, {"interrupted": True, "text": ""})
            raise
        except Exception as exc:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": f"oracle/tts failed: {exc}"})

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

    async def _speak_chunk(self, text: str) -> None:
        file_path = await asyncio.to_thread(self._tts_sync, text)
        if not file_path:
            return
        try:
            with open(file_path, "rb") as fh:
                data = fh.read()
            if data:
                payload = AudioChunk(codec=VoiceAudioCodec.OPUS, data=data).to_payload()
                payload["mime_type"] = _mime_type_for_path(file_path)
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
        await self._events.put(
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
