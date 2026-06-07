"""Portable realtime voice smoke checks.

These checks exercise the sidecar websocket/session protocol without opening a
microphone or requiring audio-model hardware. They are not an acoustic quality
benchmark; they prove that the configured sidecar can accept a session, emit a
ready state, and round-trip a transcript turn through the realtime event stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from agent.realtime_voice import AudioChunk, RealtimeVoiceSessionConfig, VoiceAudioCodec, VoiceEvent, VoiceEventType
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_session import RealtimeVoiceSession
from agent.realtime_voice_text_engine import TextOracleTTSEngine
from agent.realtime_voice_sidecar import RealtimeVoiceSidecarClient


@dataclass(frozen=True)
class RealtimeVoiceSidecarSmokeResult:
    ok: bool
    ready_ms: Optional[int] = None
    transcript_partial_ms: Optional[int] = None
    transcript_final_ms: Optional[int] = None
    first_text_ms: Optional[int] = None
    first_audio_ms: Optional[int] = None
    barge_in_ack_ms: Optional[int] = None
    final_text: str = ""
    audio_bytes: int = 0
    output_audio_bytes: int = 0
    audio_after_barge_in_bytes: int = 0
    events: Tuple[str, ...] = ()
    error: str = ""


def realtime_voice_smoke_result_payload(
    result: RealtimeVoiceSidecarSmokeResult,
    *,
    kind: str,
) -> dict[str, Any]:
    """Return a JSON-safe realtime voice smoke result payload."""
    return {
        "kind": kind,
        "ok": bool(result.ok),
        "ready_ms": result.ready_ms,
        "transcript_partial_ms": result.transcript_partial_ms,
        "transcript_final_ms": result.transcript_final_ms,
        "first_text_ms": result.first_text_ms,
        "first_audio_ms": result.first_audio_ms,
        "barge_in_ack_ms": result.barge_in_ack_ms,
        "final_text": result.final_text,
        "audio_bytes": result.audio_bytes,
        "output_audio_bytes": result.output_audio_bytes,
        "audio_after_barge_in_bytes": result.audio_after_barge_in_bytes,
        "events": list(result.events),
        "error": result.error or None,
    }


async def run_realtime_voice_sidecar_smoke(
    config: RealtimeVoiceSessionConfig,
    *,
    transcript: str = "hello Hermes",
    audio: Optional[bytes] = None,
    audio_codec: VoiceAudioCodec = VoiceAudioCodec.WEBM_OPUS,
    timeout_seconds: float = 5.0,
) -> RealtimeVoiceSidecarSmokeResult:
    """Run a protocol-level realtime sidecar smoke check."""
    timeout = max(0.1, float(timeout_seconds or 5.0))
    started_at = time.perf_counter()
    ready_ms: Optional[int] = None
    transcript_partial_ms: Optional[int] = None
    events: list[str] = []
    client = RealtimeVoiceSidecarClient()

    try:
        await client.start(config)
        if audio is not None:
            payload = AudioChunk(
                codec=audio_codec,
                data=audio,
                sample_rate_hz=config.sample_rate_hz,
                channels=config.channels,
            ).to_payload()
            payload["end_of_utterance"] = True
        else:
            payload = {
                "transcript": transcript,
                "end_of_utterance": True,
            }
        await client.send_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id=config.session_id,
                sequence=1,
                payload=payload,
            )
        )

        stream = client.events()
        deadline = started_at + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    ready_ms=ready_ms,
                    transcript_partial_ms=transcript_partial_ms,
                    audio_bytes=len(audio or b""),
                    events=tuple(events),
                    error=f"timed out after {timeout:g}s waiting for transcript.final",
                )
            try:
                event = await asyncio.wait_for(anext(stream), timeout=remaining)
            except StopAsyncIteration:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    ready_ms=ready_ms,
                    transcript_partial_ms=transcript_partial_ms,
                    audio_bytes=len(audio or b""),
                    events=tuple(events),
                    error="sidecar event stream ended before transcript.final",
                )

            elapsed_ms = int(round((time.perf_counter() - started_at) * 1000))
            events.append(event.type.value)

            if event.type == VoiceEventType.FRONTEND_STATE and ready_ms is None:
                ready_ms = elapsed_ms
            if event.type == VoiceEventType.TRANSCRIPT_PARTIAL and transcript_partial_ms is None:
                transcript_partial_ms = elapsed_ms
            if event.type == VoiceEventType.SESSION_ERROR:
                error = str(event.payload.get("error") or "sidecar session error")
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    ready_ms=ready_ms,
                    transcript_partial_ms=transcript_partial_ms,
                    audio_bytes=len(audio or b""),
                    events=tuple(events),
                    error=error,
                )
            if event.type == VoiceEventType.TRANSCRIPT_FINAL:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=True,
                    ready_ms=ready_ms,
                    transcript_partial_ms=transcript_partial_ms,
                    transcript_final_ms=elapsed_ms,
                    final_text=str(event.payload.get("text") or ""),
                    audio_bytes=len(audio or b""),
                    events=tuple(events),
                )
    except Exception as exc:
        return RealtimeVoiceSidecarSmokeResult(
            ok=False,
            ready_ms=ready_ms,
            transcript_partial_ms=transcript_partial_ms,
            audio_bytes=len(audio or b""),
            events=tuple(events),
            error=sanitize_realtime_voice_error(exc),
        )
    finally:
        await client.close()


async def run_realtime_voice_sidecar_tts_smoke(
    config: RealtimeVoiceSessionConfig,
    *,
    text: str = "Hello from Hermes.",
    metadata: Optional[Mapping[str, str]] = None,
    timeout_seconds: float = 5.0,
) -> RealtimeVoiceSidecarSmokeResult:
    """Run a sidecar TTS/output smoke check."""
    timeout = max(0.1, float(timeout_seconds or 5.0))
    started_at = time.perf_counter()
    ready_ms: Optional[int] = None
    events: list[str] = []
    client = RealtimeVoiceSidecarClient()

    try:
        await client.start(config)
        await client.send_event(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id=config.session_id,
                sequence=1,
                payload={
                    "text": text,
                    "speak": True,
                    "playback_generation": 1,
                    **dict(metadata or {}),
                },
            )
        )

        stream = client.events()
        deadline = started_at + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    ready_ms=ready_ms,
                    events=tuple(events),
                    error=f"timed out after {timeout:g}s waiting for audio.output.chunk",
                )
            try:
                event = await asyncio.wait_for(anext(stream), timeout=remaining)
            except StopAsyncIteration:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    ready_ms=ready_ms,
                    events=tuple(events),
                    error="sidecar event stream ended before audio.output.chunk",
                )

            elapsed_ms = int(round((time.perf_counter() - started_at) * 1000))
            events.append(event.type.value)

            if event.type == VoiceEventType.FRONTEND_STATE and ready_ms is None:
                ready_ms = elapsed_ms
            if event.type == VoiceEventType.SESSION_ERROR:
                error = str(event.payload.get("error") or "sidecar session error")
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    ready_ms=ready_ms,
                    events=tuple(events),
                    error=error,
                )
            if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
                try:
                    output_audio_bytes = len(AudioChunk.from_payload(event.payload).data)
                except Exception:
                    output_audio_bytes = 0
                return RealtimeVoiceSidecarSmokeResult(
                    ok=output_audio_bytes > 0,
                    ready_ms=ready_ms,
                    first_audio_ms=elapsed_ms,
                    output_audio_bytes=output_audio_bytes,
                    events=tuple(events),
                    error="" if output_audio_bytes > 0 else "audio.output.chunk contained no audio bytes",
                )
    except Exception as exc:
        return RealtimeVoiceSidecarSmokeResult(
            ok=False,
            ready_ms=ready_ms,
            events=tuple(events),
            error=sanitize_realtime_voice_error(exc),
        )
    finally:
        await client.close()


class _StaticRealtimeOracle:
    def __init__(self, answer: str):
        self.answer = answer

    async def stream_answer(self, _transcript: str):
        yield self.answer


async def run_realtime_voice_session_turn_smoke(
    config: RealtimeVoiceSessionConfig,
    *,
    answer: str = "Hello from Hermes.",
    metadata: Optional[Mapping[str, str]] = None,
    transcript: str = "hello Hermes",
    timeout_seconds: float = 5.0,
) -> RealtimeVoiceSidecarSmokeResult:
    """Run a Hermes session turn smoke through transcript, oracle text, and TTS."""

    timeout = max(0.1, float(timeout_seconds or 5.0))
    started_at = time.perf_counter()
    transcript_final_elapsed_ms: Optional[int] = None
    transcript_final_ms: Optional[int] = None
    first_text_ms: Optional[int] = None
    first_audio_ms: Optional[int] = None
    output_audio_bytes = 0
    final_text = ""
    events: list[str] = []
    engine = TextOracleTTSEngine(oracle=_StaticRealtimeOracle(answer))
    session = RealtimeVoiceSession(config, engine=engine)

    try:
        await session.start()
        await session.receive_client_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id=config.session_id,
                sequence=1,
                payload={
                    "transcript": transcript,
                    "end_of_utterance": True,
                    **dict(metadata or {}),
                },
            )
        )

        stream = session.events()
        deadline = started_at + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    transcript_final_ms=transcript_final_ms,
                    first_text_ms=first_text_ms,
                    first_audio_ms=first_audio_ms,
                    final_text=final_text,
                    output_audio_bytes=output_audio_bytes,
                    events=tuple(events),
                    error=f"timed out after {timeout:g}s waiting for session assistant text/audio",
                )
            try:
                event = await asyncio.wait_for(anext(stream), timeout=remaining)
            except StopAsyncIteration:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    transcript_final_ms=transcript_final_ms,
                    first_text_ms=first_text_ms,
                    first_audio_ms=first_audio_ms,
                    final_text=final_text,
                    output_audio_bytes=output_audio_bytes,
                    events=tuple(events),
                    error="session event stream ended before assistant text/audio",
                )

            elapsed_ms = int(round((time.perf_counter() - started_at) * 1000))
            events.append(event.type.value)

            if event.type == VoiceEventType.SESSION_ERROR:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    transcript_final_ms=transcript_final_ms,
                    first_text_ms=first_text_ms,
                    first_audio_ms=first_audio_ms,
                    final_text=final_text,
                    output_audio_bytes=output_audio_bytes,
                    events=tuple(events),
                    error=str(event.payload.get("error") or "session error"),
                )
            if event.type == VoiceEventType.TRANSCRIPT_FINAL:
                transcript_final_elapsed_ms = elapsed_ms
                transcript_final_ms = _metric_ms(
                    event.payload,
                    "audio_to_final_transcript_ms",
                    fallback=elapsed_ms,
                )
            elif event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL and first_text_ms is None:
                first_text_ms = _metric_ms(
                    event.payload,
                    "final_transcript_to_first_text_ms",
                    fallback=_elapsed_from(transcript_final_elapsed_ms, elapsed_ms),
                )
                final_text = _assistant_text_from_payload(event.payload) or final_text
            elif event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK and first_audio_ms is None:
                first_audio_ms = _metric_ms(
                    event.payload,
                    "final_transcript_to_first_audio_ms",
                    fallback=_elapsed_from(transcript_final_elapsed_ms, elapsed_ms),
                )
                try:
                    output_audio_bytes = len(AudioChunk.from_payload(event.payload).data)
                except Exception:
                    output_audio_bytes = 0
            elif event.type == VoiceEventType.ASSISTANT_COMMIT:
                final_text = str(event.payload.get("text") or final_text)

            if first_text_ms is not None and first_audio_ms is not None:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=output_audio_bytes > 0,
                    transcript_final_ms=transcript_final_ms,
                    first_text_ms=first_text_ms,
                    first_audio_ms=first_audio_ms,
                    final_text=final_text,
                    output_audio_bytes=output_audio_bytes,
                    events=tuple(events),
                    error="" if output_audio_bytes > 0 else "audio.output.chunk contained no audio bytes",
                )
    except Exception as exc:
        return RealtimeVoiceSidecarSmokeResult(
            ok=False,
            transcript_final_ms=transcript_final_ms,
            first_text_ms=first_text_ms,
            first_audio_ms=first_audio_ms,
            final_text=final_text,
            output_audio_bytes=output_audio_bytes,
            events=tuple(events),
            error=sanitize_realtime_voice_error(exc),
        )
    finally:
        await session.close()


async def run_realtime_voice_session_audio_smoke(
    config: RealtimeVoiceSessionConfig,
    *,
    audio: bytes,
    audio_codec: VoiceAudioCodec = VoiceAudioCodec.WEBM_OPUS,
    answer: str = "Hello from Hermes.",
    timeout_seconds: float = 5.0,
    sidecar: Optional[object] = None,
) -> RealtimeVoiceSidecarSmokeResult:
    """Run audio -> STT -> Hermes oracle text -> TTS in one session."""

    timeout = max(0.1, float(timeout_seconds or 5.0))
    audio_bytes = len(audio or b"")
    started_at = time.perf_counter()
    transcript_final_elapsed_ms: Optional[int] = None
    transcript_partial_ms: Optional[int] = None
    transcript_final_ms: Optional[int] = None
    first_text_ms: Optional[int] = None
    first_audio_ms: Optional[int] = None
    output_audio_bytes = 0
    final_text = ""
    events: list[str] = []
    engine = TextOracleTTSEngine(oracle=_StaticRealtimeOracle(answer), sidecar=sidecar)
    session = RealtimeVoiceSession(config, engine=engine)

    try:
        await session.start()
        payload = AudioChunk(
            codec=audio_codec,
            data=audio,
            sample_rate_hz=config.sample_rate_hz,
            channels=config.channels,
        ).to_payload()
        payload["end_of_utterance"] = True
        await session.receive_client_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id=config.session_id,
                sequence=1,
                payload=payload,
            )
        )

        stream = session.events()
        deadline = started_at + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    transcript_partial_ms=transcript_partial_ms,
                    transcript_final_ms=transcript_final_ms,
                    first_text_ms=first_text_ms,
                    first_audio_ms=first_audio_ms,
                    final_text=final_text,
                    audio_bytes=audio_bytes,
                    output_audio_bytes=output_audio_bytes,
                    events=tuple(events),
                    error=f"timed out after {timeout:g}s waiting for audio session transcript/text/audio",
                )
            try:
                event = await asyncio.wait_for(anext(stream), timeout=remaining)
            except StopAsyncIteration:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    transcript_partial_ms=transcript_partial_ms,
                    transcript_final_ms=transcript_final_ms,
                    first_text_ms=first_text_ms,
                    first_audio_ms=first_audio_ms,
                    final_text=final_text,
                    audio_bytes=audio_bytes,
                    output_audio_bytes=output_audio_bytes,
                    events=tuple(events),
                    error="session event stream ended before audio session transcript/text/audio",
                )

            elapsed_ms = int(round((time.perf_counter() - started_at) * 1000))
            events.append(event.type.value)

            if event.type == VoiceEventType.SESSION_ERROR:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    transcript_partial_ms=transcript_partial_ms,
                    transcript_final_ms=transcript_final_ms,
                    first_text_ms=first_text_ms,
                    first_audio_ms=first_audio_ms,
                    final_text=final_text,
                    audio_bytes=audio_bytes,
                    output_audio_bytes=output_audio_bytes,
                    events=tuple(events),
                    error=str(event.payload.get("error") or "session error"),
                )
            if event.type == VoiceEventType.TRANSCRIPT_PARTIAL and transcript_partial_ms is None:
                transcript_partial_ms = _metric_ms(
                    event.payload,
                    "audio_to_partial_transcript_ms",
                    fallback=elapsed_ms,
                )
            elif event.type == VoiceEventType.TRANSCRIPT_FINAL:
                transcript_final_elapsed_ms = elapsed_ms
                transcript_final_ms = _metric_ms(
                    event.payload,
                    "audio_to_final_transcript_ms",
                    fallback=elapsed_ms,
                )
                final_text = str(event.payload.get("text") or final_text).strip()
            elif event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL and first_text_ms is None:
                first_text_ms = _metric_ms(
                    event.payload,
                    "final_transcript_to_first_text_ms",
                    fallback=_elapsed_from(transcript_final_elapsed_ms, elapsed_ms),
                )
            elif event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK and first_audio_ms is None:
                first_audio_ms = _metric_ms(
                    event.payload,
                    "final_transcript_to_first_audio_ms",
                    fallback=_elapsed_from(transcript_final_elapsed_ms, elapsed_ms),
                )
                try:
                    output_audio_bytes = len(AudioChunk.from_payload(event.payload).data)
                except Exception:
                    output_audio_bytes = 0

            if (
                transcript_partial_ms is not None
                and final_text
                and first_text_ms is not None
                and first_audio_ms is not None
            ):
                return RealtimeVoiceSidecarSmokeResult(
                    ok=output_audio_bytes > 0,
                    transcript_partial_ms=transcript_partial_ms,
                    transcript_final_ms=transcript_final_ms,
                    first_text_ms=first_text_ms,
                    first_audio_ms=first_audio_ms,
                    final_text=final_text,
                    audio_bytes=audio_bytes,
                    output_audio_bytes=output_audio_bytes,
                    events=tuple(events),
                    error="" if output_audio_bytes > 0 else "audio.output.chunk contained no audio bytes",
                )
    except Exception as exc:
        return RealtimeVoiceSidecarSmokeResult(
            ok=False,
            transcript_partial_ms=transcript_partial_ms,
            transcript_final_ms=transcript_final_ms,
            first_text_ms=first_text_ms,
            first_audio_ms=first_audio_ms,
            final_text=final_text,
            audio_bytes=audio_bytes,
            output_audio_bytes=output_audio_bytes,
            events=tuple(events),
            error=sanitize_realtime_voice_error(exc),
        )
    finally:
        await session.close()


def realtime_voice_smoke_text_metadata(text: str) -> dict[str, str]:
    value = str(text or "")
    if _contains_japanese_script(value):
        return {"language": "ja", "locale": "ja-JP", "script": "Jpan"}
    if any(("A" <= char <= "Z") or ("a" <= char <= "z") for char in value):
        return {"language": "en", "locale": "en-US", "script": "Latn"}
    return {}


def _contains_japanese_script(text: str) -> bool:
    for char in text:
        codepoint = ord(char)
        if (
            0x3040 <= codepoint <= 0x30FF
            or 0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        ):
            return True
    return False


def _metric_ms(payload: Mapping[str, Any], key: str, *, fallback: Optional[int] = None) -> Optional[int]:
    metrics = payload.get("metrics") if isinstance(payload, Mapping) else None
    if isinstance(metrics, Mapping):
        value = _nonnegative_int(metrics.get(key))
        if value is not None:
            return value
    return fallback


def _elapsed_from(start_ms: Optional[int], end_ms: int) -> Optional[int]:
    if start_ms is None:
        return None
    return max(0, end_ms - start_ms)


def _assistant_text_from_payload(payload: Mapping[str, Any]) -> str:
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    delta = payload.get("delta")
    return delta.strip() if isinstance(delta, str) else ""


def _nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


async def run_realtime_voice_sidecar_barge_in_smoke(
    config: RealtimeVoiceSessionConfig,
    *,
    post_barge_in_quiet_seconds: float = 0.25,
    text: str = "Hello from Hermes.",
    timeout_seconds: float = 5.0,
) -> RealtimeVoiceSidecarSmokeResult:
    """Run a sidecar barge-in acknowledgement smoke check."""
    timeout = max(0.1, float(timeout_seconds or 5.0))
    started_at = time.perf_counter()
    barge_sent_at: Optional[float] = None
    ready_ms: Optional[int] = None
    events: list[str] = []
    client = RealtimeVoiceSidecarClient()
    event_queue: asyncio.Queue[VoiceEvent | None] = asyncio.Queue()
    reader_task: Optional[asyncio.Task[None]] = None

    async def read_events() -> None:
        try:
            async for event in client.events():
                await event_queue.put(event)
        finally:
            await event_queue.put(None)

    try:
        await client.start(config)
        reader_task = asyncio.create_task(read_events())
        ready_ms, startup_events, startup_error = await _drain_barge_in_startup_events(
            event_queue,
            started_at=started_at,
            timeout=timeout,
        )
        events.extend(startup_events)
        if startup_error:
            return RealtimeVoiceSidecarSmokeResult(
                ok=False,
                ready_ms=ready_ms,
                events=tuple(events),
                error=startup_error,
            )
        await client.send_event(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id=config.session_id,
                sequence=1,
                payload={
                    "text": text,
                    "speak": True,
                    "playback_generation": 1,
                },
            )
        )
        barge_sent_at = time.perf_counter()
        await client.send_event(
            VoiceEvent(
                type=VoiceEventType.BARGE_IN,
                session_id=config.session_id,
                sequence=2,
                payload={
                    "reason": "doctor_smoke",
                    "playback_generation": 2,
                },
            )
        )

        deadline = started_at + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    ready_ms=ready_ms,
                    events=tuple(events),
                    error=f"timed out after {timeout:g}s waiting for barge_in",
                )
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    ready_ms=ready_ms,
                    events=tuple(events),
                    error=f"timed out after {timeout:g}s waiting for barge_in",
                )
            if event is None:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    ready_ms=ready_ms,
                    events=tuple(events),
                    error="sidecar event stream ended before barge_in",
                )

            elapsed_ms = int(round((time.perf_counter() - started_at) * 1000))
            events.append(event.type.value)

            if event.type == VoiceEventType.FRONTEND_STATE and ready_ms is None:
                ready_ms = elapsed_ms
            if event.type == VoiceEventType.SESSION_ERROR:
                error = str(event.payload.get("error") or "sidecar session error")
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    ready_ms=ready_ms,
                    events=tuple(events),
                    error=error,
                )
            if event.type == VoiceEventType.BARGE_IN:
                ack_ms = int(round((time.perf_counter() - barge_sent_at) * 1000)) if barge_sent_at else elapsed_ms
                generation = event.payload.get("playback_generation")
                ok = generation in (None, 2)
                if not ok:
                    return RealtimeVoiceSidecarSmokeResult(
                        ok=False,
                        ready_ms=ready_ms,
                        barge_in_ack_ms=ack_ms,
                        events=tuple(events),
                        error=f"barge_in ack used stale playback_generation={generation}",
                    )
                quiet = await _drain_post_barge_in_audio(
                    event_queue,
                    started_at=started_at,
                    timeout=timeout,
                    quiet_seconds=post_barge_in_quiet_seconds,
                    events=events,
                )
                if quiet.audio_after_barge_in_bytes > 0:
                    return RealtimeVoiceSidecarSmokeResult(
                        ok=False,
                        ready_ms=ready_ms,
                        barge_in_ack_ms=ack_ms,
                        audio_after_barge_in_bytes=quiet.audio_after_barge_in_bytes,
                        events=tuple(events),
                        error=(
                            "audio.output.chunk arrived after barge_in "
                            f"({quiet.audio_after_barge_in_bytes} byte(s))"
                        ),
                    )
                if quiet.error:
                    return RealtimeVoiceSidecarSmokeResult(
                        ok=False,
                        ready_ms=ready_ms,
                        barge_in_ack_ms=ack_ms,
                        events=tuple(events),
                        error=quiet.error,
                    )
                return RealtimeVoiceSidecarSmokeResult(
                    ok=ok,
                    ready_ms=ready_ms,
                    barge_in_ack_ms=ack_ms,
                    events=tuple(events),
                    error="",
                )
    except Exception as exc:
        return RealtimeVoiceSidecarSmokeResult(
            ok=False,
            ready_ms=ready_ms,
            events=tuple(events),
            error=sanitize_realtime_voice_error(exc),
        )
    finally:
        if reader_task is not None:
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader_task
        await client.close()


@dataclass(frozen=True)
class _PostBargeInQuietResult:
    audio_after_barge_in_bytes: int = 0
    error: str = ""


async def _drain_post_barge_in_audio(
    event_queue: asyncio.Queue[VoiceEvent | None],
    *,
    events: list[str],
    quiet_seconds: float,
    started_at: float,
    timeout: float,
) -> _PostBargeInQuietResult:
    """Verify an interrupted sidecar does not keep emitting audio."""

    quiet_window = max(0.0, min(float(quiet_seconds or 0.0), max(0.0, timeout)))
    if quiet_window <= 0:
        return _PostBargeInQuietResult()

    deadline = min(time.perf_counter() + quiet_window, started_at + timeout)
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return _PostBargeInQuietResult()
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=remaining)
        except asyncio.TimeoutError:
            return _PostBargeInQuietResult()
        if event is None:
            return _PostBargeInQuietResult()
        events.append(event.type.value)
        if event.type == VoiceEventType.SESSION_ERROR:
            return _PostBargeInQuietResult(
                error=str(event.payload.get("error") or "sidecar session error after barge_in")
            )
        if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
            try:
                audio_bytes = len(AudioChunk.from_payload(event.payload).data)
            except Exception:
                audio_bytes = 0
            return _PostBargeInQuietResult(audio_after_barge_in_bytes=max(1, audio_bytes))


async def _drain_barge_in_startup_events(
    event_queue: asyncio.Queue[VoiceEvent | None],
    *,
    started_at: float,
    timeout: float,
) -> tuple[Optional[int], list[str], str]:
    """Drain startup/degraded state so barge-in latency measures the ack path."""

    ready_ms: Optional[int] = None
    events: list[str] = []
    deadline = time.perf_counter() + max(0.0, timeout)

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return ready_ms, events, "timed out waiting for sidecar ready before barge_in"
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=remaining)
        except asyncio.TimeoutError:
            return ready_ms, events, "timed out waiting for sidecar ready before barge_in"
        if event is None:
            return ready_ms, events, "sidecar event stream ended before barge_in"

        elapsed_ms = int(round((time.perf_counter() - started_at) * 1000))
        events.append(event.type.value)
        if event.type == VoiceEventType.FRONTEND_STATE:
            if ready_ms is None:
                ready_ms = elapsed_ms
            if str(event.payload.get("status") or "").lower() == "ready":
                return ready_ms, events, ""
            continue
        if event.type == VoiceEventType.SESSION_ERROR:
            return ready_ms, events, str(event.payload.get("error") or "sidecar session error")
        return ready_ms, events, ""
