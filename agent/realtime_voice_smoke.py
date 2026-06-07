"""Portable realtime voice smoke checks.

These checks exercise the sidecar websocket/session protocol without opening a
microphone or requiring audio-model hardware. They are not an acoustic quality
benchmark; they prove that the configured sidecar can accept a session, emit a
ready state, and round-trip a transcript turn through the realtime event stream.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from agent.realtime_voice import AudioChunk, RealtimeVoiceSessionConfig, VoiceAudioCodec, VoiceEvent, VoiceEventType
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_sidecar import RealtimeVoiceSidecarClient


@dataclass(frozen=True)
class RealtimeVoiceSidecarSmokeResult:
    ok: bool
    ready_ms: Optional[int] = None
    transcript_partial_ms: Optional[int] = None
    transcript_final_ms: Optional[int] = None
    first_audio_ms: Optional[int] = None
    barge_in_ack_ms: Optional[int] = None
    final_text: str = ""
    audio_bytes: int = 0
    output_audio_bytes: int = 0
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
        "first_audio_ms": result.first_audio_ms,
        "barge_in_ack_ms": result.barge_in_ack_ms,
        "final_text": result.final_text,
        "audio_bytes": result.audio_bytes,
        "output_audio_bytes": result.output_audio_bytes,
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


async def run_realtime_voice_sidecar_barge_in_smoke(
    config: RealtimeVoiceSessionConfig,
    *,
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

        stream = client.events()
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
                event = await asyncio.wait_for(anext(stream), timeout=remaining)
            except StopAsyncIteration:
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
                return RealtimeVoiceSidecarSmokeResult(
                    ok=ok,
                    ready_ms=ready_ms,
                    barge_in_ack_ms=ack_ms,
                    events=tuple(events),
                    error="" if ok else f"barge_in ack used stale playback_generation={generation}",
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
