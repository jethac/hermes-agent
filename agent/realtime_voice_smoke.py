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
from typing import Optional, Tuple

from agent.realtime_voice import AudioChunk, RealtimeVoiceSessionConfig, VoiceAudioCodec, VoiceEvent, VoiceEventType
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_sidecar import RealtimeVoiceSidecarClient


@dataclass(frozen=True)
class RealtimeVoiceSidecarSmokeResult:
    ok: bool
    ready_ms: Optional[int] = None
    transcript_partial_ms: Optional[int] = None
    transcript_final_ms: Optional[int] = None
    final_text: str = ""
    audio_bytes: int = 0
    events: Tuple[str, ...] = ()
    error: str = ""


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
