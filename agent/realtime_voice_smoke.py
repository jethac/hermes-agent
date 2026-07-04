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

from agent.realtime_voice import (
    AudioChunk,
    RealtimeVoiceEngineKind,
    RealtimeVoiceSessionConfig,
    VoiceAudioCodec,
    VoiceEvent,
    VoiceEventType,
    is_output_audio_event_type,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_session import RealtimeVoiceSession
from agent.realtime_voice_text_engine import KameInterfaceOracleEngine, TextOracleTTSEngine
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
    assistant_final_text: str = ""
    audio_bytes: int = 0
    output_audio_bytes: int = 0
    audio_after_barge_in_bytes: int = 0
    events: Tuple[str, ...] = ()
    first_audio_metrics: Optional[Mapping[str, Any]] = None
    route: str = ""
    interface_input_source: str = ""
    reflex_provider: str = ""
    reflex_validation_error: str = ""
    turn_id: str = ""
    audio_segment_ref: str = ""
    evidence_bundle_id: str = ""
    evidence_merge_key: str = ""
    audio_segment_ref_observed: bool = False
    interpreter_evidence_observed: bool = False
    transcript_hypotheses_labeled: bool = False
    witness_arrival_phases: Tuple[str, ...] = ()
    interpreter_input_order: Tuple[str, ...] = ()
    transcript_hypotheses: Tuple[Mapping[str, Any], ...] = ()
    interpreter_adjudication_outcomes: Tuple[str, ...] = ()
    promoted_evidence_authority: Mapping[str, Any] | None = None
    transport: str = ""
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
        "assistant_final_text": result.assistant_final_text or None,
        "audio_bytes": result.audio_bytes,
        "output_audio_bytes": result.output_audio_bytes,
        "audio_after_barge_in_bytes": result.audio_after_barge_in_bytes,
        "events": list(result.events),
        "first_audio_metrics": dict(result.first_audio_metrics or {}) or None,
        "metrics": dict(result.first_audio_metrics or {}) or None,
        "route": result.route or None,
        "interface_input_source": result.interface_input_source or None,
        "reflex_provider": result.reflex_provider or None,
        "reflex_validation_error": result.reflex_validation_error or None,
        "turn_id": result.turn_id or None,
        "audio_segment_ref": result.audio_segment_ref or None,
        "evidence_bundle_id": result.evidence_bundle_id or None,
        "evidence_merge_key": result.evidence_merge_key or None,
        "audio_segment_ref_observed": result.audio_segment_ref_observed,
        "interpreter_evidence_observed": result.interpreter_evidence_observed,
        "transcript_hypotheses_labeled": result.transcript_hypotheses_labeled,
        "witness_arrival_phases": list(result.witness_arrival_phases),
        "interpreter_input_order": list(result.interpreter_input_order),
        "transcript_hypotheses": [dict(item) for item in result.transcript_hypotheses],
        "interpreter_adjudication_outcomes": list(result.interpreter_adjudication_outcomes),
        "promoted_evidence_authority": dict(result.promoted_evidence_authority or {}),
        "transport": result.transport or None,
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
                    error=f"timed out after {timeout:g}s waiting for final user turn event",
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
                    error="sidecar event stream ended before final user turn event",
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
            if _is_final_user_turn_event(event):
                return RealtimeVoiceSidecarSmokeResult(
                    ok=True,
                    ready_ms=ready_ms,
                    transcript_partial_ms=transcript_partial_ms,
                    transcript_final_ms=elapsed_ms,
                    final_text=_final_user_turn_text(event),
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
    first_audio_metrics: Optional[dict[str, Any]] = None
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
                    error=f"timed out after {timeout:g}s waiting for output audio chunk",
                )
            try:
                event = await asyncio.wait_for(anext(stream), timeout=remaining)
            except StopAsyncIteration:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    ready_ms=ready_ms,
                    events=tuple(events),
                    error="sidecar event stream ended before output audio chunk",
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
            if is_output_audio_event_type(event.type):
                first_audio_metrics = _safe_metrics(event.payload)
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
                    first_audio_metrics=first_audio_metrics,
                    error="" if output_audio_bytes > 0 else f"{event.type.value} contained no audio bytes",
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
    first_audio_metrics: Optional[dict[str, Any]] = None
    output_audio_bytes = 0
    final_text = ""
    assistant_final_text = ""
    assistant_committed = False
    events: list[str] = []
    kame_evidence: dict[str, str] = {}
    engine = _smoke_engine(config, answer=answer)
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
                    **kame_evidence,
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
                    **kame_evidence,
                    error="session event stream ended before assistant text/audio",
                )

            elapsed_ms = int(round((time.perf_counter() - started_at) * 1000))
            events.append(event.type.value)
            _capture_kame_evidence(kame_evidence, event.payload)

            if event.type == VoiceEventType.SESSION_ERROR:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    transcript_final_ms=transcript_final_ms,
                    first_text_ms=first_text_ms,
                    first_audio_ms=first_audio_ms,
                    final_text=final_text,
                    output_audio_bytes=output_audio_bytes,
                    events=tuple(events),
                    **kame_evidence,
                    error=str(event.payload.get("error") or "session error"),
                )
            if _is_final_user_turn_event(event):
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
            elif is_output_audio_event_type(event.type) and first_audio_ms is None:
                first_audio_metrics = _safe_metrics(event.payload)
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
                assistant_committed = True

            if first_text_ms is not None and first_audio_ms is not None and assistant_committed:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=output_audio_bytes > 0,
                    transcript_final_ms=transcript_final_ms,
                    first_text_ms=first_text_ms,
                    first_audio_ms=first_audio_ms,
                    final_text=final_text,
                    output_audio_bytes=output_audio_bytes,
                    events=tuple(events),
                    first_audio_metrics=first_audio_metrics,
                    **kame_evidence,
                    error="" if output_audio_bytes > 0 else f"{event.type.value} contained no audio bytes",
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
            first_audio_metrics=first_audio_metrics,
            **kame_evidence,
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
    first_audio_metrics: Optional[dict[str, Any]] = None
    output_audio_bytes = 0
    final_text = ""
    assistant_final_text = ""
    assistant_committed = False
    events: list[str] = []
    kame_evidence: dict[str, str] = {}
    engine = _smoke_engine(config, answer=answer, sidecar=sidecar)
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
                    assistant_final_text=assistant_final_text,
                    audio_bytes=audio_bytes,
                    output_audio_bytes=output_audio_bytes,
                    events=tuple(events),
                    **kame_evidence,
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
                    assistant_final_text=assistant_final_text,
                    audio_bytes=audio_bytes,
                    output_audio_bytes=output_audio_bytes,
                    events=tuple(events),
                    **kame_evidence,
                    error="session event stream ended before audio session transcript/text/audio",
                )

            elapsed_ms = int(round((time.perf_counter() - started_at) * 1000))
            events.append(event.type.value)
            _capture_kame_evidence(kame_evidence, event.payload)

            if event.type == VoiceEventType.SESSION_ERROR:
                return RealtimeVoiceSidecarSmokeResult(
                    ok=False,
                    transcript_partial_ms=transcript_partial_ms,
                    transcript_final_ms=transcript_final_ms,
                    first_text_ms=first_text_ms,
                    first_audio_ms=first_audio_ms,
                    final_text=final_text,
                    assistant_final_text=assistant_final_text,
                    audio_bytes=audio_bytes,
                    output_audio_bytes=output_audio_bytes,
                    events=tuple(events),
                    **kame_evidence,
                    error=str(event.payload.get("error") or "session error"),
                )
            if event.type == VoiceEventType.TRANSCRIPT_PARTIAL and transcript_partial_ms is None:
                transcript_partial_ms = _metric_ms(
                    event.payload,
                    "audio_to_partial_transcript_ms",
                    fallback=elapsed_ms,
                )
            elif _is_final_user_turn_event(event):
                transcript_final_elapsed_ms = elapsed_ms
                transcript_final_ms = _metric_ms(
                    event.payload,
                    "audio_to_final_transcript_ms",
                    fallback=elapsed_ms,
                )
                if event.type == VoiceEventType.TRANSCRIPT_FINAL or not final_text:
                    final_text = _final_user_turn_text(event) or final_text
            elif event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL and first_text_ms is None:
                first_text_ms = _metric_ms(
                    event.payload,
                    "final_transcript_to_first_text_ms",
                    fallback=_elapsed_from(transcript_final_elapsed_ms, elapsed_ms),
                )
            elif is_output_audio_event_type(event.type) and first_audio_ms is None:
                first_audio_metrics = _safe_metrics(event.payload)
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
                assistant_final_text = str(event.payload.get("text") or assistant_final_text).strip()
                assistant_committed = True

            partial_or_kame_ready = (
                transcript_partial_ms is not None
                or config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE
            )
            assistant_ready = config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE or assistant_committed
            if (
                partial_or_kame_ready
                and assistant_ready
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
                    assistant_final_text=assistant_final_text,
                    audio_bytes=audio_bytes,
                    output_audio_bytes=output_audio_bytes,
                    events=tuple(events),
                    first_audio_metrics=first_audio_metrics,
                    **kame_evidence,
                    error="" if output_audio_bytes > 0 else f"{event.type.value} contained no audio bytes",
                )
    except Exception as exc:
        return RealtimeVoiceSidecarSmokeResult(
            ok=False,
            transcript_partial_ms=transcript_partial_ms,
            transcript_final_ms=transcript_final_ms,
            first_text_ms=first_text_ms,
            first_audio_ms=first_audio_ms,
            final_text=final_text,
            assistant_final_text=assistant_final_text,
            audio_bytes=audio_bytes,
            output_audio_bytes=output_audio_bytes,
            events=tuple(events),
            first_audio_metrics=first_audio_metrics,
            **kame_evidence,
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


def _smoke_engine(
    config: RealtimeVoiceSessionConfig,
    *,
    answer: str,
    sidecar: Optional[object] = None,
):
    oracle = _StaticRealtimeOracle(answer)
    if config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
        return KameInterfaceOracleEngine(oracle=oracle, sidecar=sidecar)
    return TextOracleTTSEngine(oracle=oracle, sidecar=sidecar)


def _capture_kame_evidence(target: dict[str, Any], payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        return
    route = str(payload.get("route") or payload.get("kame_route") or "").strip()
    if route and not target.get("route"):
        target["route"] = route
    input_source = str(
        payload.get("interface_input_source")
        or payload.get("kame_interface_input_source")
        or ""
    ).strip()
    if input_source and not target.get("interface_input_source"):
        target["interface_input_source"] = input_source
    reflex_provider = str(payload.get("reflex_provider") or payload.get("kame_reflex_provider") or "").strip()
    if reflex_provider and not target.get("reflex_provider"):
        target["reflex_provider"] = reflex_provider
    validation_error = str(
        payload.get("reflex_validation_error")
        or payload.get("kame_reflex_validation_error")
        or ""
    ).strip()
    if validation_error and not target.get("reflex_validation_error"):
        target["reflex_validation_error"] = validation_error
    for source_key, target_key in (
        ("turn_id", "turn_id"),
        ("kame_turn_id", "turn_id"),
        ("audio_segment_ref", "audio_segment_ref"),
        ("kame_audio_segment_ref", "audio_segment_ref"),
        ("evidence_bundle_id", "evidence_bundle_id"),
        ("kame_evidence_bundle_id", "evidence_bundle_id"),
        ("evidence_merge_key", "evidence_merge_key"),
        ("kame_evidence_merge_key", "evidence_merge_key"),
    ):
        value = str(payload.get(source_key) or "").strip()
        if value and not target.get(target_key):
            target[target_key] = value
    if target.get("audio_segment_ref"):
        target["audio_segment_ref_observed"] = True
    if payload.get("audio_segment_ref_observed") is True:
        target["audio_segment_ref_observed"] = True
    prompt_order = payload.get("interpreter_input_order") or payload.get("kame_interpreter_prompt_input_order")
    if isinstance(prompt_order, (list, tuple)) and not target.get("interpreter_input_order"):
        target["interpreter_input_order"] = tuple(str(item) for item in prompt_order if str(item or "").strip())
    hypotheses = payload.get("transcript_hypotheses") or payload.get("kame_transcript_hypotheses")
    if isinstance(hypotheses, (list, tuple)) and hypotheses and not target.get("transcript_hypotheses"):
        compact: list[dict[str, Any]] = []
        outcomes: list[str] = []
        for item in hypotheses:
            if not isinstance(item, Mapping):
                continue
            hypothesis = {
                str(key): value
                for key, value in item.items()
                if key
                in {
                    "kind",
                    "source",
                    "text",
                    "arrival_phase",
                    "authority",
                    "tool_authority",
                    "partial",
                    "confidence",
                    "latency_ms",
                    "adjudication",
                    "outcome",
                    "interpreter_adjudication",
                }
            }
            if "text" in hypothesis:
                hypothesis["text"] = str(hypothesis["text"] or "")[:240]
            if hypothesis:
                compact.append(hypothesis)
            outcome = str(
                item.get("adjudication")
                or item.get("outcome")
                or item.get("interpreter_adjudication")
                or ""
            ).strip()
            if outcome and outcome not in outcomes:
                outcomes.append(outcome)
        if compact:
            target["transcript_hypotheses"] = tuple(compact)
            target["transcript_hypotheses_labeled"] = True
        if outcomes and not target.get("interpreter_adjudication_outcomes"):
            target["interpreter_adjudication_outcomes"] = tuple(outcomes)
    phases = payload.get("witness_arrival_phases") or payload.get("kame_witness_arrival_phases")
    if isinstance(phases, (list, tuple)) and phases and not target.get("witness_arrival_phases"):
        target["witness_arrival_phases"] = tuple(str(item) for item in phases if str(item or "").strip())
    outcomes = payload.get("interpreter_adjudication_outcomes") or payload.get("witness_adjudication_outcomes")
    if isinstance(outcomes, (list, tuple)) and outcomes and not target.get("interpreter_adjudication_outcomes"):
        target["interpreter_adjudication_outcomes"] = tuple(str(item) for item in outcomes if str(item or "").strip())
    promoted = payload.get("promoted_evidence_authority") or payload.get("promoted_fields_authority")
    if isinstance(promoted, Mapping) and promoted and not target.get("promoted_evidence_authority"):
        target["promoted_evidence_authority"] = {
            str(key): str(value)
            for key, value in promoted.items()
            if str(key or "").strip() and str(value or "").strip()
        }
    if payload.get("interpreter_evidence_observed") is True:
        target["interpreter_evidence_observed"] = True
    elif (
        target.get("interpreter_input_order")
        and target.get("interpreter_adjudication_outcomes")
        and target.get("promoted_evidence_authority")
    ):
        target["interpreter_evidence_observed"] = True


def _is_final_user_turn_event(event: VoiceEvent) -> bool:
    return event.type in {VoiceEventType.TRANSCRIPT_FINAL, VoiceEventType.INTERFACE_INTENT_FINAL}


def _final_user_turn_text(event: VoiceEvent) -> str:
    if event.type == VoiceEventType.INTERFACE_INTENT_FINAL:
        keys = ("kame_intent", "intent", "text", "transcript")
    else:
        keys = ("text", "transcript", "kame_intent", "intent")
    for key in keys:
        text = str(event.payload.get(key) or "").strip()
        if text:
            return text
    return ""


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


def _safe_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload, Mapping) else None
    if not isinstance(metrics, Mapping):
        return {}
    safe: dict[str, Any] = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, bool):
            safe[key] = value
        elif isinstance(value, (int, float, str)) or value is None:
            safe[key] = value
    return safe


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
                            "output audio chunk arrived after barge_in "
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
        if is_output_audio_event_type(event.type):
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
