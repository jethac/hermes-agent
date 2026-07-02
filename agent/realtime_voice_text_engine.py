"""Text-oracle + TTS realtime voice engine."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import tempfile
import time
from dataclasses import replace
from typing import Any, AsyncIterator, List, Mapping, Optional

from agent.realtime_voice import (
    AudioChunk,
    RealtimeVoiceASRMode,
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
from agent.realtime_voice_kame import KameOracleRequest, KameRoute, kame_local_reply_denies_voice_capability
from agent.realtime_voice_oracle import HermesRealtimeOracle, NullRealtimeOracle
from agent.realtime_voice_oracle_jobs import (
    OracleJob,
    OracleJobEvent,
    OracleJobEventType,
    OracleJobManager,
    OracleJobNotFoundError,
    OracleJobQueueFullError,
    OracleJobReprioritizationRequiredError,
    OracleJobState,
)
from agent.realtime_voice_planner import RealtimeSpeechPlanner
from agent.realtime_voice_sidecar import RealtimeVoiceSidecarClient, wants_realtime_sidecar
from agent.think_scrubber import StreamingThinkScrubber, strip_leading_reasoning_trace


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
        self._completed_input_generations: set[int] = set()
        self._frontend_output_active = False
        self._active_task_interrupts_oracle = True
        self._assistant_metadata_by_generation: dict[int, dict] = {}
        self._cancellation_token_by_generation: dict[int, str] = {}
        self._interface_decision_at_by_generation: dict[int, float] = {}
        self._oracle_first_token_at_by_generation: dict[int, float] = {}
        self._first_audio_metric_generations: set[int] = set()
        self._speech_energy_ms_by_user: dict[str, int] = {}
        self._kame_committed_turns: list[tuple[str, str]] = []
        self._oracle_job_manager: Optional[OracleJobManager] = None
        self._oracle_job_context_by_turn_id: dict[str, tuple[int, dict, KameOracleRequest]] = {}
        self._running_oracle_request_by_job_id: dict[str, KameOracleRequest] = {}

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
                if _realtime_voice_fail_closed(config):
                    raise RuntimeError(
                        "realtime voice sidecar unavailable and fallback_policy=fail_closed: "
                        f"{sanitize_realtime_voice_error(exc)}"
                    ) from exc
                await self._emit(
                    VoiceEventType.FRONTEND_STATE,
                    {
                        "status": "fallback",
                        "reason": "sidecar_unavailable",
                        "error": sanitize_realtime_voice_error(exc),
                        "sidecar": False,
                    },
                )
        if _async_oracle_jobs_enabled(config):
            self._oracle_job_manager = OracleJobManager(
                max_concurrent=_oracle_jobs_config_int(config, "max_concurrent", default=1),
                queue_limit=_oracle_jobs_config_int(config, "queue_limit", default=16),
                default_priority=_oracle_jobs_config_str(config, "default_priority", default="normal"),
                overflow_policy=_oracle_jobs_config_str(config, "overflow_policy", default="queue"),
                runner=self._run_oracle_job,
                event_callback=self._emit_oracle_job_event,
                interrupt_callback=self._interrupt_oracle_job,
                audit_ledger_path=_oracle_jobs_config_optional_str(config, "audit_ledger_path"),
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
        if event.type in {VoiceEventType.SESSION_STOP, VoiceEventType.SESSION_CLOSED}:
            await self._close(sidecar_stop_event=event)
            return
        if event.type in {VoiceEventType.SPEECH_START, VoiceEventType.SPEECH_ENERGY, VoiceEventType.SPEECH_END}:
            await self._handle_speech_lifecycle_event(event)
            return
        if event.type in {VoiceEventType.PLAYBACK_STARTED, VoiceEventType.PLAYBACK_STOPPED}:
            await self._handle_playback_lifecycle_event(event)
            return
        if event.type == VoiceEventType.INTERFACE_INTENT_FINAL:
            await self._handle_interface_intent_final(event)
            return
        if event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL:
            await self._handle_oracle_job_cancel_event(event)
            return
        if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE:
            await self._handle_oracle_job_update_event(event)
            return
        if event.type != VoiceEventType.AUDIO_INPUT_CHUNK:
            return

        transcript = str(event.payload.get("transcript") or "").strip()
        if transcript:
            await self._auto_barge_in_for_speech(event)
            if not _payload_marks_final_transcript(event.payload):
                if self.config is not None and self.config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
                    partial_intent_payload = _kame_interface_partial_payload_from_payload(event.payload)
                    if partial_intent_payload:
                        await self._emit(VoiceEventType.INTERFACE_INTENT_PARTIAL, partial_intent_payload)
                    if not _allow_kame_transcript_events(self.config):
                        return
                await self._emit(
                    VoiceEventType.TRANSCRIPT_PARTIAL,
                    {
                        "text": transcript,
                        "stability": 0.8,
                        **transcript_metadata_from_payload(event.payload),
                    },
                )
                return
            await self._start_turn(
                transcript,
                metadata=transcript_metadata_from_payload(event.payload),
                oracle_payload=event.payload,
            )
            return

        try:
            chunk = AudioChunk.from_payload(event.payload)
            if chunk.data and _payload_confirms_speech_for_barge_in(event.payload):
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
                if _allow_kame_transcript_events(self.config):
                    await self._emit(VoiceEventType.TRANSCRIPT_PARTIAL, {"text": "", "stability": 0.1})
                self._active_task_interrupts_oracle = True
                self._active_task = asyncio.create_task(self._transcribe_and_answer(audio, chunk.codec))

    async def events(self) -> AsyncIterator[VoiceEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def get_oracle_job_status(self) -> dict[str, Any]:
        manager = self._oracle_job_manager
        if manager is None:
            return {}
        status = await manager.status_view()
        return {"enabled": True, **status}

    async def close(self) -> None:
        await self._close()

    async def _close(self, *, sidecar_stop_event: Optional[VoiceEvent] = None) -> None:
        if self._closed:
            return
        self._closed = True
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._active_task
        if self._oracle_job_manager is not None:
            with contextlib.suppress(Exception):
                await self._oracle_job_manager.shutdown(
                    reason="session closed",
                    timeout_seconds=_oracle_jobs_config_float(
                        self.config,
                        "shutdown_timeout_seconds",
                        default=2.0,
                    ),
                )
        await self._notify_sidecar_session_stop(sidecar_stop_event)
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

    async def _notify_sidecar_session_stop(self, event: Optional[VoiceEvent]) -> None:
        if self._sidecar is None or self.config is None:
            return
        stop_event = event or VoiceEvent(
            type=VoiceEventType.SESSION_CLOSED,
            session_id=self.config.session_id,
            sequence=self._sequence + 1,
            payload={"reason": "closed"},
        )
        with contextlib.suppress(Exception):
            await self._sidecar.send_event(stop_event)  # type: ignore[attr-defined]

    async def _auto_barge_in_for_speech(self, event: VoiceEvent) -> None:
        if self._pending_turn_generation is not None:
            return
        backend_active = self._active_task is not None and not self._active_task.done()
        if not backend_active and not self._frontend_output_active:
            return
        await self._interrupt_active_turn(event, reason="user_speech")

    async def _handle_speech_lifecycle_event(self, event: VoiceEvent) -> None:
        if event.type == VoiceEventType.SPEECH_START:
            self._speech_energy_ms_by_user[_speech_energy_user_key(event.payload)] = 0
        elif event.type == VoiceEventType.SPEECH_END:
            self._speech_energy_ms_by_user.pop(_speech_energy_user_key(event.payload), None)
        elif event.type == VoiceEventType.SPEECH_ENERGY and self._speech_energy_confirms_barge_in(event.payload):
            await self._auto_barge_in_for_speech(event)
        if self._sidecar is not None:
            await self._send_sidecar_event(event)

    async def _handle_playback_lifecycle_event(self, event: VoiceEvent) -> None:
        generation = _payload_generation(event.payload)
        if generation is not None and generation < self._playback_generation:
            return
        if generation is not None:
            self._playback_generation = max(self._playback_generation, generation)
        self._frontend_output_active = event.type == VoiceEventType.PLAYBACK_STARTED
        if self._sidecar is not None:
            await self._send_sidecar_event(event)

    def _speech_energy_confirms_barge_in(self, payload: Mapping[str, Any]) -> bool:
        if _payload_confirms_speech_for_barge_in(payload):
            return True
        rms = _payload_nonnegative_float(payload.get("rms"))
        if rms is None:
            return False
        min_rms = _barge_in_min_rms(self.config)
        key = _speech_energy_user_key(payload)
        if rms < min_rms:
            self._speech_energy_ms_by_user[key] = 0
            return False
        duration_ms = _speech_energy_duration_ms(payload)
        if duration_ms <= 0:
            duration_ms = 20
        accumulated = self._speech_energy_ms_by_user.get(key, 0) + duration_ms
        self._speech_energy_ms_by_user[key] = accumulated
        return accumulated >= _barge_in_min_speech_ms(self.config)

    async def _interrupt_active_turn(self, event: VoiceEvent, *, reason: str) -> None:
        cancelled_generation = self._playback_generation
        cancelled_metadata = self._assistant_metadata_by_generation.get(cancelled_generation) or {}
        cancellation_token = self._cancellation_token_by_generation.pop(cancelled_generation, "")
        self._playback_generation += 1
        self._pending_turn_generation = self._playback_generation
        self._input_generation += 1
        self._input_generation_active = False
        self._frontend_output_active = False
        backend_interrupt_requested = bool(self._active_task and not self._active_task.done())
        frontend_cancel_requested = self._sidecar is not None
        payload = {
            "reason": reason or "client",
            "playback_generation": self._playback_generation,
            "cancelled_playback_generation": cancelled_generation,
            "frontend_cancel_requested": frontend_cancel_requested,
            "backend_interrupt_requested": backend_interrupt_requested,
        }
        if cancellation_token:
            payload["cancellation_token"] = cancellation_token
        if backend_interrupt_requested and self._active_task is not None:
            self._active_task.cancel()
        oracle = self._oracle
        if (not backend_interrupt_requested or self._active_task_interrupts_oracle) and hasattr(oracle, "interrupt"):
            oracle.interrupt("Realtime voice barge-in")  # type: ignore[attr-defined]
        self._clear_inbound_audio()
        cancelled_kame_oracle = bool(
            cancellation_token
            and cancelled_metadata.get("kame_route") in {KameRoute.DEFER.value, KameRoute.ORACLE_DIRECT.value}
        )
        if cancelled_kame_oracle:
            cancel_event = await self._emit(
                VoiceEventType.INTERFACE_ORACLE_CANCEL,
                {
                    "reason": payload["reason"],
                    "playback_generation": self._playback_generation,
                    "cancelled_playback_generation": cancelled_generation,
                    "cancellation_token": cancellation_token,
                    **_kame_interface_payload_from_metadata(cancelled_metadata),
                },
            )
            if cancel_event is not None and self._sidecar is not None:
                await self._send_sidecar_event(cancel_event)
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
        if cancelled_kame_oracle:
            await self._emit_oracle_cancelled(
                playback_generation=self._playback_generation,
                cancelled_playback_generation=cancelled_generation,
                metadata=cancelled_metadata,
                reason=payload["reason"],
                cancellation_token=cancellation_token,
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
                    if self.config is not None and self.config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
                        partial_intent_payload = _kame_interface_partial_payload_from_payload(payload)
                        if partial_intent_payload:
                            await self._emit(VoiceEventType.INTERFACE_INTENT_PARTIAL, partial_intent_payload)
                        if not _allow_kame_transcript_events(self.config):
                            continue
                    await self._emit(VoiceEventType.TRANSCRIPT_PARTIAL, transcript_event_payload_from_payload(payload))
                elif event.type == VoiceEventType.INTERFACE_INTENT_PARTIAL:
                    payload = dict(event.payload)
                    if self._is_stale_sidecar_input(payload):
                        continue
                    await self._emit(VoiceEventType.INTERFACE_INTENT_PARTIAL, payload)
                elif event.type == VoiceEventType.INTERFACE_INTENT_FINAL:
                    if await self._handle_interface_intent_final(event, from_sidecar=True):
                        continue
                elif event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL:
                    await self._handle_oracle_job_cancel_event(event)
                    continue
                elif event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE:
                    await self._handle_oracle_job_update_event(event)
                    continue
                elif event.type == VoiceEventType.TRANSCRIPT_FINAL:
                    payload = dict(event.payload)
                    if self._is_stale_sidecar_input(payload):
                        continue
                    if not _kame_transcript_final_can_start_turn(self.config, payload):
                        continue
                    text = str(payload.get("text") or "").strip()
                    if text:
                        self._mark_sidecar_input_completed(payload)
                        await self._start_turn(
                            text,
                            input_generation=_payload_input_generation(payload),
                            metadata=transcript_metadata_from_payload(payload),
                            oracle_payload=payload,
                        )
                elif event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
                    payload = dict(event.payload)
                    generation = _payload_generation(payload)
                    if generation is not None and generation < self._playback_generation:
                        continue
                    if generation is not None:
                        self._playback_generation = max(self._playback_generation, generation)
                    payload.setdefault("playback_generation", self._playback_generation)
                    payload = self._kame_sidecar_audio_payload_with_metrics(payload)
                    self._frontend_output_active = True
                    await self._emit(VoiceEventType.AUDIO_OUTPUT_CHUNK, payload)
                elif event.type == VoiceEventType.ASSISTANT_AUDIO_END:
                    payload = dict(event.payload)
                    generation = _payload_generation(payload)
                    if generation is not None and generation < self._playback_generation:
                        continue
                    if generation is not None:
                        self._playback_generation = max(self._playback_generation, generation)
                    payload.setdefault("playback_generation", self._playback_generation)
                    await self._emit(VoiceEventType.ASSISTANT_AUDIO_END, payload)
                elif event.type == VoiceEventType.PLAYBACK_STARTED:
                    payload = dict(event.payload)
                    generation = _payload_generation(payload)
                    if generation is not None and generation < self._playback_generation:
                        continue
                    if generation is not None:
                        self._playback_generation = max(self._playback_generation, generation)
                    payload.setdefault("playback_generation", self._playback_generation)
                    self._frontend_output_active = True
                    await self._emit(VoiceEventType.PLAYBACK_STARTED, payload)
                elif event.type == VoiceEventType.PLAYBACK_STOPPED:
                    payload = dict(event.payload)
                    generation = _payload_generation(payload)
                    if generation is not None and generation < self._playback_generation:
                        continue
                    if generation is not None:
                        self._playback_generation = max(self._playback_generation, generation)
                    payload.setdefault("playback_generation", self._playback_generation)
                    self._frontend_output_active = False
                    await self._emit(VoiceEventType.PLAYBACK_STOPPED, payload)
                elif event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
                    payload = dict(event.payload)
                    generation = _payload_generation(payload)
                    if generation is not None and generation < self._playback_generation:
                        continue
                    if generation is not None:
                        self._playback_generation = max(self._playback_generation, generation)
                    payload.setdefault("playback_generation", self._playback_generation)
                    self._frontend_output_active = True
                    await self._emit(VoiceEventType.ASSISTANT_TEXT_PARTIAL, payload)
                elif event.type == VoiceEventType.ASSISTANT_COMMIT:
                    payload = dict(event.payload)
                    generation = _payload_generation(payload)
                    if generation is not None and generation < self._playback_generation:
                        continue
                    if generation is not None:
                        self._playback_generation = max(self._playback_generation, generation)
                    payload.setdefault("playback_generation", self._playback_generation)
                    self._frontend_output_active = False
                    await self._emit(VoiceEventType.ASSISTANT_COMMIT, payload)
                elif event.type == VoiceEventType.BARGE_IN:
                    self._frontend_output_active = False
                    await self._emit(VoiceEventType.BARGE_IN, dict(event.payload))
                elif event.type == VoiceEventType.FRONTEND_STATE:
                    await self._emit(VoiceEventType.FRONTEND_STATE, dict(event.payload))
                elif event.type == VoiceEventType.SESSION_ERROR:
                    self._frontend_output_active = False
                    error = sanitize_realtime_voice_error(event.payload.get("error") or "")
                    await self._disable_sidecar()
                    if _realtime_voice_fail_closed(self.config):
                        await self._emit(
                            VoiceEventType.SESSION_ERROR,
                            {
                                "reason": "sidecar_session_error",
                                "error": (
                                    "realtime voice sidecar session error and "
                                    f"fallback_policy=fail_closed: {error}"
                                ),
                                "sidecar": False,
                            },
                        )
                        return
                    await self._emit(
                        VoiceEventType.FRONTEND_STATE,
                        {
                            "status": "fallback",
                            "reason": "sidecar_session_error",
                            "error": error,
                            "sidecar": False,
                        },
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._disable_sidecar()
            error = sanitize_realtime_voice_error(exc)
            if _realtime_voice_fail_closed(self.config):
                await self._emit(
                    VoiceEventType.SESSION_ERROR,
                    {
                        "reason": "sidecar_event_stream_failed",
                        "error": (
                            "realtime voice sidecar event stream failed and "
                            f"fallback_policy=fail_closed: {error}"
                        ),
                        "sidecar": False,
                    },
                )
                return
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "sidecar_event_stream_failed",
                    "error": error,
                    "sidecar": False,
                },
            )

    async def _handle_interface_intent_final(self, event: VoiceEvent, *, from_sidecar: bool = False) -> bool:
        if self.config is None or self.config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
            return False
        payload = dict(event.payload)
        if from_sidecar and self._is_stale_sidecar_input(payload):
            return True
        text = _kame_final_turn_text_from_payload(payload)
        if not text:
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "empty_interface_intent_final",
                    "sidecar": from_sidecar,
                },
            )
            return True
        input_generation = _payload_input_generation(payload)
        if input_generation is not None:
            self._mark_sidecar_input_completed(payload)
        await self._start_turn(
            text,
            input_generation=input_generation,
            metadata=transcript_metadata_from_payload(payload),
            oracle_payload=payload,
        )
        return True

    async def _handle_oracle_job_cancel_event(self, event: VoiceEvent) -> None:
        manager = self._oracle_job_manager
        reason = str(event.payload.get("reason") or "user requested cancellation").strip()
        job_id = str(event.payload.get("job_id") or "").strip()
        cancel_all = _metadata_bool(event.payload.get("all"), default=False) or job_id.lower() == "all"
        if manager is None:
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "oracle_jobs_unavailable",
                    "sidecar": self._sidecar is not None,
                },
            )
            return
        if cancel_all:
            cancelled = await manager.cancel_all(reason=reason)
            await self._emit_interface_event(
                VoiceEventType.INTERFACE_ORACLE_CANCEL,
                {
                    "all": True,
                    "reason": reason,
                    "cancelled_jobs": [job.job_id for job in cancelled],
                },
            )
            return
        if not job_id:
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "oracle_job_cancel_missing_job_id",
                    "sidecar": self._sidecar is not None,
                },
            )
            return
        try:
            await manager.cancel(job_id, reason=reason)
        except OracleJobNotFoundError:
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "oracle_job_not_found",
                    "job_id": job_id,
                    "sidecar": self._sidecar is not None,
                },
            )
            return
        await self._emit_interface_event(
            VoiceEventType.INTERFACE_ORACLE_CANCEL,
            {
                "job_id": job_id,
                "reason": reason,
            },
        )

    async def _handle_oracle_job_update_event(self, event: VoiceEvent) -> None:
        manager = self._oracle_job_manager
        job_id = str(event.payload.get("job_id") or "").strip()
        priority = str(event.payload.get("priority") or "").strip()
        update_text = (
            str(event.payload.get("update_text") or "").strip()
            or str(event.payload.get("text") or "").strip()
            or str(event.payload.get("clarification") or "").strip()
        )
        reason = str(event.payload.get("reason") or "user requested update").strip()
        if manager is None:
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "oracle_jobs_unavailable",
                    "sidecar": self._sidecar is not None,
                },
            )
            return
        if not job_id or (not priority and not update_text):
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "oracle_job_update_missing_fields",
                    "job_id": job_id,
                    "sidecar": self._sidecar is not None,
                },
            )
            return
        try:
            if priority:
                job = await manager.update_priority(job_id, priority=priority)
            else:
                job = await manager.get(job_id)
            if update_text:
                job = await manager.add_update(
                    job_id,
                    text=update_text,
                    source=str(event.payload.get("source") or event.payload.get("transport") or "user"),
                    update_type=str(event.payload.get("update_type") or "clarification"),
                )
                await self._notify_running_oracle_job_update(job, update_text=update_text, reason=reason)
        except OracleJobNotFoundError:
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "oracle_job_not_found",
                    "job_id": job_id,
                    "sidecar": self._sidecar is not None,
                },
            )
            return
        await self._emit_interface_event(
            VoiceEventType.INTERFACE_ORACLE_UPDATE,
            _oracle_job_update_event_payload(job, reason=reason),
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
        return generation is not None and (
            generation < self._input_generation or generation in self._completed_input_generations
        )

    def _mark_sidecar_input_completed(self, payload: Mapping[str, Any]) -> None:
        generation = _payload_input_generation(dict(payload))
        if generation is None:
            return
        self._completed_input_generations.add(generation)
        if len(self._completed_input_generations) <= 256:
            return
        stale_count = len(self._completed_input_generations) - 256
        for old_generation in sorted(self._completed_input_generations)[:stale_count]:
            self._completed_input_generations.discard(old_generation)

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
                fallback_payload = _kame_local_stt_fallback_payload(self.config, transcript)
                if fallback_payload:
                    await self._emit(VoiceEventType.FRONTEND_STATE, _kame_local_stt_fallback_state_payload(self.config))
                await self._start_turn(
                    transcript,
                    metadata=fallback_payload,
                    oracle_payload=fallback_payload,
                )
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
        oracle_payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        turn_started_at = time.perf_counter()
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
        if self._pending_turn_generation is not None:
            generation = self._pending_turn_generation
            self._pending_turn_generation = None
        else:
            self._playback_generation += 1
            generation = self._playback_generation
        assistant_metadata = dict(metadata or {})
        if input_generation is not None:
            assistant_metadata["input_generation"] = input_generation
        if oracle_payload is not None:
            payload_metrics = oracle_payload.get("metrics")
            if isinstance(payload_metrics, Mapping):
                assistant_metadata["metrics"] = _merge_metrics(
                    assistant_metadata.get("metrics"),
                    payload_metrics,
                )
        cancellation_token = _kame_cancellation_token(self.config, generation)
        oracle_request = self._kame_oracle_request(
            transcript,
            generation,
            oracle_payload=oracle_payload,
            metadata=assistant_metadata,
            cancellation_token=cancellation_token,
        )
        interface_decision_at = time.perf_counter()
        if oracle_request is not None:
            assistant_metadata.update(oracle_request.to_metadata())
            if _kame_metrics_policy_turn_spans_enabled(self.config):
                assistant_metadata["metrics"] = _merge_metrics(
                    assistant_metadata.get("metrics"),
                    {
                        "kame_final_transcript_to_interface_decision_ms": _elapsed_perf_ms(
                            turn_started_at,
                            interface_decision_at,
                        )
                    },
                )
            if oracle_request.cancellation_token:
                self._cancellation_token_by_generation[generation] = oracle_request.cancellation_token
            self._interface_decision_at_by_generation[generation] = interface_decision_at
        payload = {"text": transcript, "playback_generation": generation}
        if input_generation is not None:
            payload["input_generation"] = input_generation
        if assistant_metadata:
            payload.update(assistant_metadata)
        await self._emit(VoiceEventType.TRANSCRIPT_FINAL, payload)
        self._assistant_metadata_by_generation[generation] = assistant_metadata
        if oracle_request is not None:
            interface_payload = _kame_interface_payload_with_metrics(
                oracle_request,
                generation,
                assistant_metadata,
            )
            await self._emit_interface_event(VoiceEventType.INTERFACE_INTENT_FINAL, interface_payload)
        local_reply = await self._kame_oracle_job_control_reply(oracle_request)
        if local_reply:
            control_metadata = {**assistant_metadata, "oracle_job_control": True}
            if oracle_request is not None:
                await self._emit_interface_event(
                    VoiceEventType.INTERFACE_REPLY_LOCAL,
                    {
                        **_kame_interface_payload_with_metrics(oracle_request, generation, control_metadata),
                        "text": local_reply,
                        "oracle_job_control": True,
                    },
                )
            self._active_task_interrupts_oracle = True
            self._active_task = asyncio.create_task(
                self._speak_kame_local_reply(local_reply, generation, control_metadata)
            )
            return
        local_reply = await self._kame_oracle_job_status_reply(oracle_request)
        if not local_reply:
            local_reply = _kame_local_reply(oracle_request)
        if local_reply:
            if oracle_request is not None:
                await self._emit_interface_event(
                    VoiceEventType.INTERFACE_REPLY_LOCAL,
                    {
                        **_kame_interface_payload_with_metrics(oracle_request, generation, assistant_metadata),
                        "text": local_reply,
                    },
                )
            self._active_task_interrupts_oracle = True
            self._active_task = asyncio.create_task(
                self._speak_kame_local_reply(local_reply, generation, assistant_metadata)
            )
            return
        if oracle_request is not None:
            interface_payload = _kame_interface_payload_with_metrics(oracle_request, generation, assistant_metadata)
            if oracle_request.route == KameRoute.DEFER:
                await self._emit_interface_event(
                    VoiceEventType.INTERFACE_REPLY_DEFER,
                    _kame_defer_reply_payload_with_metrics(oracle_request, generation, assistant_metadata),
                )
            await self._emit_interface_event(VoiceEventType.INTERFACE_ORACLE_REQUEST, interface_payload)
            if await self._submit_async_oracle_job_if_enabled(
                oracle_request,
                generation,
                assistant_metadata,
            ):
                return
        self._active_task_interrupts_oracle = True
        self._active_task = asyncio.create_task(
            self._answer_and_speak(
                transcript,
                generation,
                assistant_metadata,
                oracle_request=oracle_request,
            )
        )

    async def _kame_oracle_job_status_reply(self, oracle_request: Optional[KameOracleRequest]) -> str:
        manager = self._oracle_job_manager
        if manager is None or oracle_request is None:
            return ""
        if oracle_request.route not in {KameRoute.LOCAL, KameRoute.REJECT_OR_CLARIFY}:
            return ""
        if not _kame_oracle_job_status_requested(oracle_request):
            return ""
        return _kame_oracle_job_status_text(await manager.status_view())

    async def _kame_oracle_job_control_reply(self, oracle_request: Optional[KameOracleRequest]) -> str:
        manager = self._oracle_job_manager
        if manager is None or oracle_request is None:
            return ""
        operation = _kame_oracle_job_control_operation(oracle_request, await manager.status_view())
        if not operation:
            return ""
        kind = operation["kind"]
        reason = operation.get("reason") or "spoken oracle job control"
        if kind == "cancel_all":
            await self._stop_playback_for_spoken_cancel_all(reason=reason)
            cancelled = await manager.cancel_all(reason=reason)
            await self._emit_interface_event(
                VoiceEventType.INTERFACE_ORACLE_CANCEL,
                {
                    "all": True,
                    "reason": reason,
                    "cancelled_jobs": [job.job_id for job in cancelled],
                    "spoken_control": True,
                },
            )
            if cancelled:
                return "I cancelled all current oracle jobs."
            return "There were no active oracle jobs to cancel."
        job_id = str(operation.get("job_id") or "").strip()
        if not job_id:
            return ""
        try:
            if kind == "cancel":
                job = await manager.cancel(job_id, reason=reason)
                await self._emit_interface_event(
                    VoiceEventType.INTERFACE_ORACLE_CANCEL,
                    {
                        "job_id": job.job_id,
                        "reason": reason,
                        "spoken_control": True,
                    },
                )
                label = _oracle_job_control_label(job.to_status())
                return f"I cancelled {label}." if label else "I cancelled that oracle job."
            if kind == "priority":
                priority = str(operation.get("priority") or "").strip()
                job = await manager.update_priority(job_id, priority=priority)
                await self._emit_interface_event(
                    VoiceEventType.INTERFACE_ORACLE_UPDATE,
                    {
                        **_oracle_job_update_event_payload(job, reason=reason),
                        "spoken_control": True,
                    },
                )
                label = _oracle_job_control_label(job.to_status())
                target = f" for {label}" if label else ""
                return f"I set {job.priority} priority{target}."
            if kind == "update":
                update_text = str(operation.get("update_text") or "").strip()
                job = await manager.add_update(
                    job_id,
                    text=update_text,
                    source=oracle_request.source or "voice",
                    update_type="clarification",
                )
                await self._notify_running_oracle_job_update(job, update_text=update_text, reason=reason)
                await self._emit_interface_event(
                    VoiceEventType.INTERFACE_ORACLE_UPDATE,
                    {
                        **_oracle_job_update_event_payload(job, reason=reason),
                        "spoken_control": True,
                    },
                )
                label = _oracle_job_control_label(job.to_status())
                return f"I added that to {label}." if label else "I added that to the oracle job."
        except OracleJobNotFoundError:
            return "I could not find that oracle job."
        return ""

    async def _stop_playback_for_spoken_cancel_all(self, *, reason: str) -> None:
        cancelled_generation = max(0, self._playback_generation - 1)
        self._frontend_output_active = False
        payload = {
            "reason": reason or "spoken request to stop everything",
            "playback_generation": self._playback_generation,
            "cancelled_playback_generation": cancelled_generation,
            "frontend_cancel_requested": self._sidecar is not None,
            "backend_interrupt_requested": False,
            "oracle_job_control": True,
        }
        await self._emit(VoiceEventType.BARGE_IN, payload)
        if self._sidecar is not None and self.config is not None:
            await self._send_sidecar_event(
                VoiceEvent(
                    type=VoiceEventType.BARGE_IN,
                    session_id=self.config.session_id,
                    sequence=self._sequence,
                    payload=payload,
                )
            )

    async def _submit_async_oracle_job_if_enabled(
        self,
        oracle_request: KameOracleRequest,
        playback_generation: int,
        metadata: dict,
    ) -> bool:
        manager = self._oracle_job_manager
        if manager is None or oracle_request.route not in {KameRoute.DEFER, KameRoute.ORACLE_DIRECT}:
            return False
        self._cancellation_token_by_generation.pop(playback_generation, None)
        self._oracle_job_context_by_turn_id[oracle_request.turn_id] = (
            playback_generation,
            dict(metadata),
            oracle_request,
        )
        try:
            job = await manager.submit(oracle_request, priority=oracle_request.priority)
        except OracleJobReprioritizationRequiredError:
            self._oracle_job_context_by_turn_id.pop(oracle_request.turn_id, None)
            await self._emit_oracle_error(
                playback_generation,
                metadata,
                reason="oracle_job_reprioritization_required",
                error="oracle job reprioritization required",
            )
            self._active_task_interrupts_oracle = False
            self._active_task = asyncio.create_task(
                self._speak_oracle_job_reprioritization_required_status(playback_generation, metadata)
            )
            return True
        except OracleJobQueueFullError:
            self._oracle_job_context_by_turn_id.pop(oracle_request.turn_id, None)
            await self._emit_oracle_error(
                playback_generation,
                metadata,
                reason="oracle_job_queue_full",
                error="oracle job queue is full",
            )
            self._active_task_interrupts_oracle = False
            self._active_task = asyncio.create_task(
                self._speak_oracle_job_capacity_status(playback_generation, metadata)
            )
            return True
        except Exception as exc:
            self._oracle_job_context_by_turn_id.pop(oracle_request.turn_id, None)
            await self._emit_oracle_error(
                playback_generation,
                metadata,
                reason="oracle_job_submit_failed",
                error=sanitize_realtime_voice_error(exc),
            )
            return False

        metadata["oracle_job_id"] = job.job_id
        self._assistant_metadata_by_generation[playback_generation] = metadata
        self._active_task_interrupts_oracle = False
        self._active_task = asyncio.create_task(
            self._speak_kame_oracle_job_ack(oracle_request, playback_generation, metadata)
        )
        return True

    async def _speak_kame_oracle_job_ack(
        self,
        oracle_request: KameOracleRequest,
        playback_generation: int,
        metadata: Mapping[str, Any],
    ) -> None:
        try:
            text = self._planner.clean(_oracle_reflex_narration_text(self.config, oracle_request))
            if not text:
                return
            await self._emit(
                VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                {
                    "text": text,
                    "playback_generation": playback_generation,
                    "reflex_narration": oracle_request.route == KameRoute.DEFER,
                    "acknowledgement": oracle_request.route != KameRoute.DEFER,
                    "oracle_job_ack": True,
                    **metadata,
                    **_kame_route_metrics_payload(metadata, oracle_called=True),
                },
            )
            await self._speak_chunk(text, playback_generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "oracle_job_ack_tts_failed",
                    "error": sanitize_realtime_voice_error(exc),
                    "sidecar": self._sidecar is not None,
                },
            )
        finally:
            self._assistant_metadata_by_generation.pop(playback_generation, None)
            self._interface_decision_at_by_generation.pop(playback_generation, None)
            self._oracle_first_token_at_by_generation.pop(playback_generation, None)
            self._first_audio_metric_generations.discard(playback_generation)

    async def _speak_oracle_job_capacity_status(
        self,
        playback_generation: int,
        metadata: Mapping[str, Any],
    ) -> None:
        status_text = self._planner.clean("I am at oracle job capacity right now.")
        if not status_text:
            return
        await self._emit(
            VoiceEventType.ASSISTANT_TEXT_PARTIAL,
            {
                "text": status_text,
                "playback_generation": playback_generation,
                "oracle_job_queue_full": True,
                **metadata,
                **_kame_route_metrics_payload(metadata, oracle_called=True),
            },
        )
        with contextlib.suppress(Exception):
            await self._speak_chunk(status_text, playback_generation)

    async def _speak_oracle_job_reprioritization_required_status(
        self,
        playback_generation: int,
        metadata: Mapping[str, Any],
    ) -> None:
        status_text = self._planner.clean(
            "I am at oracle job capacity. Tell me which job to prioritize or cancel."
        )
        if not status_text:
            return
        await self._emit(
            VoiceEventType.ASSISTANT_TEXT_PARTIAL,
            {
                "text": status_text,
                "playback_generation": playback_generation,
                "oracle_job_reprioritization_required": True,
                **metadata,
                **_kame_route_metrics_payload(metadata, oracle_called=True),
            },
        )
        with contextlib.suppress(Exception):
            await self._speak_chunk(status_text, playback_generation)

    async def _run_oracle_job(self, job: OracleJob) -> Mapping[str, Any]:
        request = job.request
        if request is None:
            raise RuntimeError("oracle job is missing its KAME request")
        request = _oracle_request_for_job(job, request)
        metadata = dict(job.metadata)
        metadata.update(request.to_metadata())
        oracle = self._oracle or NullRealtimeOracle()
        answer = ""
        scrubber = StreamingThinkScrubber()
        accepted_at = time.perf_counter()
        self._running_oracle_request_by_job_id[job.job_id] = request
        try:
            await self._emit_oracle_job_progress(job, phase="accepted", delta="", text="")
            async for item in _stream_oracle_answer(
                oracle,
                request.oracle_text or request.intent,
                metadata,
                oracle_request=request,
                timeout_seconds=_oracle_timeout_seconds(self.config),
            ):
                if job.state in {OracleJobState.CANCEL_REQUESTED, OracleJobState.CANCELLED}:
                    raise asyncio.CancelledError
                if isinstance(item, Mapping):
                    oracle_tool_event_type = _oracle_tool_event_type(item)
                    if oracle_tool_event_type is not None:
                        if oracle_tool_event_type == VoiceEventType.ORACLE_TOOL_CALL and _oracle_tool_event_waits_for_approval(item):
                            manager = self._oracle_job_manager
                            if manager is not None:
                                with contextlib.suppress(OracleJobNotFoundError):
                                    await manager.mark_waiting_for_approval(
                                        job.job_id,
                                        reason=_oracle_tool_approval_reason(item),
                                        approval=_oracle_tool_event_payload(item),
                                    )
                        elif oracle_tool_event_type == VoiceEventType.ORACLE_TOOL_RESULT:
                            manager = self._oracle_job_manager
                            if manager is not None and job.state == OracleJobState.WAITING_FOR_APPROVAL:
                                with contextlib.suppress(OracleJobNotFoundError):
                                    await manager.mark_running(job.job_id)
                        await self._emit_oracle_job_progress(
                            job,
                            phase="tool",
                            delta="",
                            text=answer,
                            tool_event=_oracle_tool_event_payload(
                                item,
                                redact_sensitive=_oracle_tool_event_waits_for_approval(item),
                            ),
                        )
                        continue
                delta = _oracle_stream_text_delta(item)
                if not delta:
                    continue
                delta = scrubber.feed(delta)
                if not delta:
                    continue
                answer += delta
                await self._emit_oracle_job_progress(job, phase="stream", delta=delta, text=answer)
            tail = scrubber.flush()
            if tail:
                answer += tail
                await self._emit_oracle_job_progress(job, phase="stream", delta=tail, text=answer)
            answer = strip_leading_reasoning_trace(answer)
            await self._emit_oracle_job_progress(
                job,
                phase="final",
                delta="",
                text=answer,
                metrics={"oracle_job_total_stream_ms": _elapsed_perf_ms(accepted_at, time.perf_counter())},
            )
            return {"result_summary": answer}
        finally:
            self._running_oracle_request_by_job_id.pop(job.job_id, None)

    async def _notify_running_oracle_job_update(
        self,
        job: OracleJob,
        *,
        update_text: str,
        reason: str,
    ) -> None:
        if job.state not in {OracleJobState.RUNNING, OracleJobState.WAITING_FOR_APPROVAL}:
            return
        update_text = " ".join(str(update_text or "").split())
        if not update_text:
            return
        request = self._running_oracle_request_by_job_id.get(job.job_id)
        if request is None:
            return
        updated_request = _oracle_request_for_job(job, request)
        self._running_oracle_request_by_job_id[job.job_id] = updated_request
        updater = getattr(self._oracle, "update_request", None)
        if not callable(updater):
            return
        metadata = {
            "job_id": job.job_id,
            "state": job.state.value,
            "reason": reason,
            "update_count": len(job.updates),
        }
        latest_update = str(job.to_status().get("latest_update") or "").strip()
        if latest_update:
            metadata["latest_update"] = latest_update
        try:
            result = updater(updated_request, update_text, metadata)
            if hasattr(result, "__await__"):
                await result
        except Exception as exc:
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "oracle_job_update_delivery_failed",
                    "job_id": job.job_id,
                    "error": sanitize_realtime_voice_error(exc),
                    "sidecar": self._sidecar is not None,
                },
            )

    async def _emit_oracle_job_progress(
        self,
        job: OracleJob,
        *,
        phase: str,
        delta: str,
        text: str,
        metrics: Optional[Mapping[str, int]] = None,
        tool_event: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload: dict[str, Any] = {
            **_oracle_job_payload(job),
            "phase": phase,
            "delta": delta,
            "text": text if phase == "final" else "",
            "final": phase == "final",
        }
        if metrics:
            payload["metrics"] = _nonnegative_int_metrics(metrics)
        if tool_event:
            payload["tool_event"] = dict(tool_event)
        await self._emit_oracle_job_voice_event(VoiceEventType.ORACLE_JOB_PROGRESS, payload)

    async def _emit_oracle_job_event(self, event: OracleJobEvent) -> None:
        voice_event_type = _voice_event_type_for_oracle_job_event(event.type)
        if voice_event_type is None:
            return
        payload = {
            **dict(event.payload),
            "job_id": event.job_id,
            "session_id": event.session_id,
            "state": event.state.value,
            "timestamp_ms": event.timestamp_ms,
        }
        turn_id = str(payload.get("turn_id") or "").strip()
        context = self._oracle_job_context_by_turn_id.get(turn_id)
        if context is not None:
            source_playback_generation, _, _ = context
            payload.setdefault("source_playback_generation", source_playback_generation)
            payload.setdefault("playback_generation", self._playback_generation)
        await self._emit_oracle_job_voice_event(
            voice_event_type,
            payload,
        )
        if event.type in {
            OracleJobEventType.COMPLETED,
            OracleJobEventType.FAILED,
            OracleJobEventType.CANCELLED,
        }:
            self._schedule_oracle_job_terminal_speech(event)

    async def _emit_oracle_job_voice_event(self, event_type: VoiceEventType, payload: dict[str, Any]) -> None:
        event = await self._emit(event_type, payload)
        if event is not None and self._sidecar is not None:
            await self._send_sidecar_event(event)

    def _schedule_oracle_job_terminal_speech(self, event: OracleJobEvent) -> None:
        payload = dict(event.payload)
        turn_id = str(payload.get("turn_id") or "").strip()
        context = self._oracle_job_context_by_turn_id.get(turn_id)
        if context is None:
            return
        playback_generation, metadata, oracle_request = context
        if event.type in {OracleJobEventType.CANCELLED, OracleJobEventType.FAILED}:
            self._oracle_job_context_by_turn_id.pop(turn_id, None)
        if event.type == OracleJobEventType.CANCELLED:
            return
        if not _oracle_job_terminal_speech_enabled(self.config):
            self._oracle_job_context_by_turn_id.pop(turn_id, None)
            return
        if self._closed:
            self._oracle_job_context_by_turn_id.pop(turn_id, None)
            return
        if playback_generation != self._playback_generation:
            self._oracle_job_context_by_turn_id.pop(turn_id, None)
            return
        if event.type == OracleJobEventType.COMPLETED and not str(payload.get("result_summary") or "").strip():
            self._oracle_job_context_by_turn_id.pop(turn_id, None)
            return
        previous_task = self._active_task if self._active_task and not self._active_task.done() else None
        task = asyncio.create_task(
            self._speak_oracle_job_terminal_event(
                event,
                playback_generation,
                metadata,
                oracle_request,
                previous_task=previous_task,
            )
        )
        self._active_task_interrupts_oracle = False
        self._active_task = task
        if event.type == OracleJobEventType.COMPLETED:
            task.add_done_callback(lambda _task, key=turn_id: self._oracle_job_context_by_turn_id.pop(key, None))

    async def _speak_oracle_job_terminal_event(
        self,
        event: OracleJobEvent,
        playback_generation: int,
        metadata: Mapping[str, Any],
        oracle_request: KameOracleRequest,
        *,
        previous_task: Optional[asyncio.Task[None]] = None,
    ) -> None:
        if previous_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await previous_task
        if self._closed or playback_generation != self._playback_generation:
            return
        payload = dict(event.payload)
        metadata = {
            **dict(metadata),
            "oracle_job_id": event.job_id,
        }
        if event.type == OracleJobEventType.FAILED:
            raw_text = _oracle_job_failed_status_text(payload)
            outcome = "oracle_job_failed"
        else:
            raw_text = str(payload.get("result_summary") or "").strip()
            outcome = "oracle_job_completed"
        spoken_text = self._planner.clean(strip_leading_reasoning_trace(raw_text))
        if not spoken_text:
            return
        spoken_text, spoken_truncated = _limit_spoken_text(
            spoken_text,
            max_sentences=_effective_max_spoken_sentences(self.config, oracle_request=oracle_request),
        )
        if not spoken_text:
            return
        metadata = {
            **metadata,
            **_voice_response_policy_payload(
                policy=_voice_response_policy(self.config, oracle_request=oracle_request),
                max_sentences=_effective_max_spoken_sentences(self.config, oracle_request=oracle_request),
                truncated=spoken_truncated,
            ),
        }
        self._assistant_metadata_by_generation[playback_generation] = metadata
        await self._emit(
            VoiceEventType.ASSISTANT_TEXT_PARTIAL,
            {
                "text": spoken_text,
                "playback_generation": playback_generation,
                "oracle_job_id": event.job_id,
                "oracle_job_result": event.type == OracleJobEventType.COMPLETED,
                "oracle_job_failed": event.type == OracleJobEventType.FAILED,
                **metadata,
                **_kame_route_metrics_payload(metadata, oracle_called=True),
            },
        )
        try:
            await self._speak_chunk(spoken_text, playback_generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "oracle_job_result_tts_failed",
                    "error": sanitize_realtime_voice_error(exc),
                    "sidecar": self._sidecar is not None,
                },
            )
            return
        try:
            if playback_generation != self._playback_generation:
                return
            plan = self._planner.plan(spoken_text)
            if not plan.committed_text:
                return
            await self._emit_interface_commit(
                playback_generation,
                metadata,
                text=plan.committed_text,
            )
            await self._emit_session_metrics(
                playback_generation,
                metadata,
                oracle_called=True,
                outcome=outcome,
            )
            await self._emit(
                VoiceEventType.ASSISTANT_COMMIT,
                {
                    "text": plan.committed_text,
                    "playback_generation": playback_generation,
                    "oracle_job_id": event.job_id,
                    "oracle_job_result": event.type == OracleJobEventType.COMPLETED,
                    "oracle_job_failed": event.type == OracleJobEventType.FAILED,
                    **metadata,
                    **_kame_route_metrics_payload(metadata, oracle_called=True),
                },
            )
        finally:
            self._assistant_metadata_by_generation.pop(playback_generation, None)
            self._interface_decision_at_by_generation.pop(playback_generation, None)
            self._oracle_first_token_at_by_generation.pop(playback_generation, None)
            self._first_audio_metric_generations.discard(playback_generation)

    async def _interrupt_oracle_job(self, job: OracleJob, reason: str) -> None:
        oracle = self._oracle
        interrupt_request = getattr(oracle, "interrupt_request", None)
        if callable(interrupt_request) and job.request is not None:
            interrupt_request(job.request, f"Realtime voice oracle job {job.job_id} cancelled: {reason}")
            return
        if hasattr(oracle, "interrupt"):
            oracle.interrupt(f"Realtime voice oracle job {job.job_id} cancelled: {reason}")  # type: ignore[attr-defined]

    def _kame_oracle_request(
        self,
        transcript: str,
        playback_generation: int,
        *,
        oracle_payload: Optional[Mapping[str, Any]],
        metadata: Mapping[str, Any],
        cancellation_token: str,
    ) -> Optional[KameOracleRequest]:
        config = self.config
        if config is None or config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
            return None
        config_metadata = config.metadata if isinstance(config.metadata, Mapping) else {}
        source = str(config_metadata.get("transport") or config_metadata.get("source") or "voice").strip() or "voice"
        user_id = str(config_metadata.get("user_id") or "").strip() or None
        payload: dict[str, Any] = {}
        payload.update(metadata)
        if oracle_payload is not None:
            payload.update(dict(oracle_payload))
        if config.asr_mode == RealtimeVoiceASRMode.DISABLED:
            for key in (
                "asr_transcript",
                "asr_transcript_source",
                "asr_transcript_confidence",
                "oracle_verbatim_transcript",
                "oracle_verbatim_transcript_source",
                "oracle_verbatim_transcript_confidence",
            ):
                payload.pop(key, None)
        if cancellation_token:
            payload.setdefault("cancellation_token", cancellation_token)
        payload.setdefault("voice_response_policy", _voice_response_policy(config))
        if "conversation_summary" not in payload:
            summary = self._kame_conversation_summary()
            if summary:
                payload["conversation_summary"] = summary
        request = KameOracleRequest.from_turn(
            session_id=config.session_id,
            turn_id=f"{config.session_id}:{playback_generation}",
            source=source,
            user_id=user_id,
            payload=payload,
            fallback_text=transcript,
            default_max_spoken_sentences=_effective_max_spoken_sentences(config),
            routing_policy=_kame_routing_policy(config),
        )
        if request.route == KameRoute.DEFER and not request.interface_already_said:
            narration = _kame_oracle_handoff_narration(config, request)
            if narration:
                request = replace(request, interface_already_said=narration)
        return request

    async def _speak_kame_local_reply(
        self,
        reply: str,
        playback_generation: int,
        metadata: dict,
    ) -> None:
        try:
            planned_reply = self._planner.clean(reply)
            if not planned_reply:
                return
            planned_reply, truncated = _limit_spoken_text(
                planned_reply,
                max_sentences=_effective_max_spoken_sentences(self.config),
            )
            if not planned_reply:
                return
            metadata = {
                **metadata,
                **_voice_response_policy_payload(
                    policy=_voice_response_policy(self.config),
                    max_sentences=_effective_max_spoken_sentences(self.config),
                    truncated=truncated,
                ),
            }
            self._assistant_metadata_by_generation[playback_generation] = metadata
            await self._emit(
                VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                {
                    "text": planned_reply,
                    "playback_generation": playback_generation,
                    "local_reply": True,
                    **metadata,
                    **_kame_route_metrics_payload(metadata, oracle_called=False),
                },
            )
            await self._speak_chunk(planned_reply, playback_generation)
            plan = self._planner.plan(planned_reply)
            if not plan.committed_text:
                return
            if playback_generation == self._playback_generation:
                await self._emit_interface_commit(
                    playback_generation,
                    metadata,
                    text=plan.committed_text,
                    local_reply=True,
                )
                await self._emit_session_metrics(
                    playback_generation,
                    metadata,
                    oracle_called=False,
                    outcome="local_commit",
                    local_reply=True,
                )
                await self._emit(
                    VoiceEventType.ASSISTANT_COMMIT,
                    {
                        "text": plan.committed_text,
                        "playback_generation": playback_generation,
                        "local_reply": True,
                        **metadata,
                        **_kame_route_metrics_payload(metadata, oracle_called=False),
                    },
                )
        except asyncio.CancelledError:
            if not self._closed and playback_generation == self._playback_generation:
                await self._emit(
                    VoiceEventType.ASSISTANT_COMMIT,
                    {"interrupted": True, "text": "", "playback_generation": playback_generation},
                )
            raise
        except Exception as exc:
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"local reply TTS failed: {sanitize_realtime_voice_error(exc)}"},
            )
        finally:
            self._assistant_metadata_by_generation.pop(playback_generation, None)
            self._cancellation_token_by_generation.pop(playback_generation, None)
            self._interface_decision_at_by_generation.pop(playback_generation, None)
            self._first_audio_metric_generations.discard(playback_generation)

    async def _answer_and_speak(
        self,
        transcript: str,
        playback_generation: int,
        metadata: dict,
        *,
        oracle_request: Optional[KameOracleRequest] = None,
    ) -> None:
        speak_tasks: List[asyncio.Task[None]] = []
        speak_chain: Optional[asyncio.Task[None]] = None
        assistant_metadata = dict(metadata)
        tts_error_reported = False
        voice_response_policy = _voice_response_policy(self.config, oracle_request=oracle_request)
        max_spoken_sentences = _effective_max_spoken_sentences(
            self.config,
            oracle_request=oracle_request,
        )
        spoken_answer = ""
        spoken_truncated = False
        turn_started_at = time.perf_counter()
        oracle_accepted_at: Optional[float] = None
        oracle_first_token_at: Optional[float] = None
        first_spoken_text_at: Optional[float] = None
        kame_timing_metrics: dict[str, int] = {}
        voice_denial_corrected = False
        kame_provider_metrics_enabled = _kame_metrics_policy_provider_spans_enabled(self.config)

        def sync_kame_timing_metrics() -> None:
            if not kame_timing_metrics:
                return
            metrics = _kame_route_metrics(
                assistant_metadata,
                oracle_called=True,
                extra_metrics=kame_timing_metrics,
            )
            if metrics:
                assistant_metadata["metrics"] = metrics
                self._assistant_metadata_by_generation[playback_generation] = assistant_metadata

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

        def correct_voice_denial(text: str) -> tuple[str, bool]:
            nonlocal voice_denial_corrected
            if oracle_request is None or not kame_local_reply_denies_voice_capability(text):
                return text, False
            if voice_denial_corrected:
                return "", True
            voice_denial_corrected = True
            return _kame_voice_capability_correction_text(self.config), True

        async def emit_planned_speech_chunk(raw_chunk: str) -> bool:
            nonlocal first_spoken_text_at, spoken_answer, spoken_truncated
            planned_chunk = self._planner.clean(raw_chunk)
            if not planned_chunk:
                return True
            planned_chunk, denial_corrected = correct_voice_denial(planned_chunk)
            if denial_corrected:
                spoken_truncated = True
            if not planned_chunk:
                return True
            planned_chunk, chunk_truncated = _limit_spoken_text(
                planned_chunk,
                max_sentences=max_spoken_sentences,
                already_spoken=spoken_answer,
            )
            spoken_truncated = spoken_truncated or chunk_truncated
            if not planned_chunk:
                if _spoken_sentence_count(spoken_answer) >= max_spoken_sentences > 0:
                    spoken_truncated = True
                    return False
                return True
            spoken_answer = _join_spoken_text(spoken_answer, planned_chunk)
            if oracle_request is not None and first_spoken_text_at is None:
                first_spoken_text_at = time.perf_counter()
                if kame_provider_metrics_enabled and oracle_first_token_at is not None:
                    kame_timing_metrics["kame_oracle_first_token_to_first_spoken_text_ms"] = _elapsed_perf_ms(
                        oracle_first_token_at,
                        first_spoken_text_at,
                    )
                sync_kame_timing_metrics()
            await self._emit(
                VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                {
                    "text": planned_chunk,
                    "playback_generation": playback_generation,
                    **assistant_metadata,
                    **_voice_response_policy_payload(
                        policy=voice_response_policy,
                        max_sentences=max_spoken_sentences,
                        truncated=spoken_truncated,
                    ),
                    **_kame_route_metrics_payload(
                        assistant_metadata,
                        oracle_called=True,
                        extra_metrics=kame_timing_metrics,
                    ),
                },
            )
            queue_speak(planned_chunk)
            if _spoken_sentence_count(spoken_answer) >= max_spoken_sentences > 0:
                spoken_truncated = True
                return False
            return True

        try:
            reflex_narration = _oracle_reflex_narration_text(self.config, oracle_request)
            if reflex_narration:
                planned_reflex_narration = self._planner.clean(reflex_narration)
                if planned_reflex_narration:
                    is_kame_reflex_narration = (
                        oracle_request is not None
                        and oracle_request.route == KameRoute.DEFER
                    )
                    await self._emit(
                        VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                        {
                            "text": planned_reflex_narration,
                            "playback_generation": playback_generation,
                            **({"reflex_narration": True} if is_kame_reflex_narration else {"acknowledgement": True}),
                            **assistant_metadata,
                            **_kame_route_metrics_payload(assistant_metadata, oracle_called=True),
                        },
                    )
                    queue_speak(planned_reflex_narration)

            oracle = self._oracle or NullRealtimeOracle()
            answer = ""
            buffer = ""
            speech_limit_reached = False
            oracle_text_scrubber = StreamingThinkScrubber()
            if oracle_request is not None:
                await self._emit_oracle_hint(
                    text="",
                    delta="",
                    final=False,
                    playback_generation=playback_generation,
                    metadata=assistant_metadata,
                    accepted=True,
                )
                oracle_accepted_at = time.perf_counter()
                if kame_provider_metrics_enabled:
                    kame_timing_metrics["kame_interface_decision_to_oracle_accepted_ms"] = _elapsed_perf_ms(
                        self._interface_decision_metric_start(playback_generation, fallback=turn_started_at),
                        oracle_accepted_at,
                    )
                    sync_kame_timing_metrics()
            async for item in _stream_oracle_answer(
                oracle,
                transcript,
                assistant_metadata,
                oracle_request=oracle_request,
                timeout_seconds=_oracle_timeout_seconds(self.config),
            ):
                if playback_generation != self._playback_generation:
                    return
                if isinstance(item, Mapping):
                    oracle_tool_event_type = _oracle_tool_event_type(item)
                    if oracle_tool_event_type is not None:
                        await self._emit_oracle_tool_event(
                            event_type=oracle_tool_event_type,
                            payload=item,
                            playback_generation=playback_generation,
                            metadata=assistant_metadata,
                            metrics=kame_timing_metrics,
                        )
                        continue
                delta = _oracle_stream_text_delta(item)
                if not delta:
                    continue
                delta = oracle_text_scrubber.feed(delta)
                if not delta:
                    continue
                now = time.perf_counter()
                if oracle_request is not None and oracle_first_token_at is None:
                    oracle_first_token_at = now
                    self._oracle_first_token_at_by_generation[playback_generation] = now
                    if kame_provider_metrics_enabled and oracle_accepted_at is not None:
                        kame_timing_metrics["kame_oracle_accepted_to_first_token_ms"] = _elapsed_perf_ms(
                            oracle_accepted_at,
                            oracle_first_token_at,
                        )
                        sync_kame_timing_metrics()
                answer += delta
                buffer += delta
                if oracle_request is not None:
                    await self._emit_oracle_hint(
                        text=answer,
                        delta=str(delta),
                        final=False,
                        playback_generation=playback_generation,
                        metadata=assistant_metadata,
                    )
                while True:
                    chunk, buffer = _take_speakable_chunk(buffer)
                    if not chunk:
                        break
                    if not await emit_planned_speech_chunk(chunk):
                        speech_limit_reached = True
                        break
                if speech_limit_reached:
                    break

            if not speech_limit_reached:
                tail = oracle_text_scrubber.flush()
                if tail:
                    answer += tail
                    buffer += tail
            if buffer.strip() and not speech_limit_reached:
                while buffer.strip():
                    chunk, buffer = _take_speakable_chunk(buffer)
                    if not chunk:
                        chunk, buffer = buffer, ""
                    if not await emit_planned_speech_chunk(chunk):
                        break

            if oracle_request is not None and answer:
                answer = strip_leading_reasoning_trace(answer)
                if kame_provider_metrics_enabled and oracle_accepted_at is not None:
                    kame_timing_metrics["kame_oracle_total_stream_ms"] = _elapsed_perf_ms(
                        oracle_accepted_at,
                        time.perf_counter(),
                    )
                    sync_kame_timing_metrics()
                await self._emit_oracle_hint(
                    text=answer,
                    delta="",
                    final=True,
                    playback_generation=playback_generation,
                    metadata=assistant_metadata,
                    metrics=kame_timing_metrics,
                )

            commit_text = spoken_answer if max_spoken_sentences > 0 and spoken_answer else answer
            if max_spoken_sentences > 0 and spoken_answer and answer and self._planner.clean(answer) != self._planner.clean(spoken_answer):
                spoken_truncated = True
            plan = self._planner.plan(commit_text)
            if speak_chain is not None:
                await speak_chain
            if not plan.committed_text:
                return
            if playback_generation == self._playback_generation:
                await self._emit_interface_commit(
                    playback_generation,
                    assistant_metadata,
                    text=plan.committed_text,
                )
                await self._emit_session_metrics(
                    playback_generation,
                    assistant_metadata,
                    oracle_called=True,
                    outcome="oracle_commit",
                    extra_metrics=kame_timing_metrics,
                )
                await self._emit(
                    VoiceEventType.ASSISTANT_COMMIT,
                    {
                        "text": plan.committed_text,
                        "playback_generation": playback_generation,
                        **assistant_metadata,
                        **_voice_response_policy_payload(
                            policy=voice_response_policy,
                            max_sentences=max_spoken_sentences,
                            truncated=spoken_truncated,
                        ),
                        **_kame_route_metrics_payload(
                            assistant_metadata,
                            oracle_called=True,
                            extra_metrics=kame_timing_metrics,
                        ),
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
        except asyncio.TimeoutError:
            await cancel_speak_tasks()
            if playback_generation == self._playback_generation:
                await self._emit_oracle_error(
                    playback_generation,
                    assistant_metadata,
                    reason="oracle_timeout",
                    error="oracle response timed out",
                )
                await self._speak_oracle_timeout_status(playback_generation, assistant_metadata)
        except Exception as exc:
            await cancel_speak_tasks()
            await self._emit_oracle_error(
                playback_generation,
                assistant_metadata,
                reason="oracle_or_tts_failed",
                error=sanitize_realtime_voice_error(exc),
            )
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"oracle/tts failed: {sanitize_realtime_voice_error(exc)}"},
            )
        finally:
            self._assistant_metadata_by_generation.pop(playback_generation, None)
            self._cancellation_token_by_generation.pop(playback_generation, None)
            self._interface_decision_at_by_generation.pop(playback_generation, None)
            self._oracle_first_token_at_by_generation.pop(playback_generation, None)
            self._first_audio_metric_generations.discard(playback_generation)

    async def _speak_oracle_timeout_status(
        self,
        playback_generation: int,
        metadata: Mapping[str, Any],
    ) -> None:
        status_text = self._planner.clean(_oracle_timeout_status_text(self.config))
        if not status_text:
            return
        payload = {
            "text": status_text,
            "playback_generation": playback_generation,
            "oracle_timeout": True,
            **metadata,
            **_kame_route_metrics_payload(metadata, oracle_called=True),
        }
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "degraded",
                "reason": "oracle_timeout",
                "oracle_timeout_seconds": _oracle_timeout_seconds(self.config),
                "sidecar": self._sidecar is not None,
            },
        )
        await self._emit(VoiceEventType.ASSISTANT_TEXT_PARTIAL, payload)
        try:
            await self._speak_chunk(status_text, playback_generation)
        except Exception as exc:
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "tts_failed",
                    "error": sanitize_realtime_voice_error(exc),
                    "sidecar": False,
                },
            )
        if playback_generation == self._playback_generation:
            await self._emit_interface_commit(playback_generation, metadata, text=status_text)
            await self._emit_session_metrics(
                playback_generation,
                metadata,
                oracle_called=True,
                outcome="oracle_timeout_status",
            )
            await self._emit(VoiceEventType.ASSISTANT_COMMIT, payload)

    async def _emit_interface_commit(
        self,
        playback_generation: int,
        metadata: Mapping[str, Any],
        *,
        text: str,
        local_reply: bool = False,
    ) -> None:
        if not _is_kame_metadata(metadata):
            return
        await self._emit_interface_event(
            VoiceEventType.INTERFACE_COMMIT,
            {
                **_kame_interface_payload_from_metadata(metadata),
                "playback_generation": playback_generation,
                "local_reply": local_reply,
                "text": text,
            },
        )
        self._record_kame_committed_turn(metadata, text)

    async def _emit_interface_event(self, event_type: VoiceEventType, payload: dict[str, Any]) -> Optional[VoiceEvent]:
        event = await self._emit(event_type, payload)
        if event is not None and self._sidecar is not None:
            await self._send_sidecar_event(event)
        return event

    def _record_kame_committed_turn(self, metadata: Mapping[str, Any], assistant_text: str) -> None:
        if not _is_kame_metadata(metadata):
            return
        user_text = _compact_kame_summary_text(
            str(metadata.get("kame_intent") or metadata.get("kame_transcript") or "")
        )
        assistant = _compact_kame_summary_text(assistant_text)
        if not user_text and not assistant:
            return
        self._kame_committed_turns.append((user_text, assistant))
        del self._kame_committed_turns[:-4]

    def _kame_conversation_summary(self) -> str:
        if not self._kame_committed_turns:
            return ""
        parts = []
        for user_text, assistant_text in self._kame_committed_turns[-4:]:
            if user_text and assistant_text:
                parts.append(f"User: {user_text} / Hermes: {assistant_text}")
            elif user_text:
                parts.append(f"User: {user_text}")
            elif assistant_text:
                parts.append(f"Hermes: {assistant_text}")
        return "Recent committed voice turns: " + " | ".join(parts) if parts else ""

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
        metadata = self._assistant_metadata_by_generation.get(playback_generation) or {}
        if self._sidecar is not None and self.config is not None:
            event = VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id=self.config.session_id,
                sequence=self._sequence + 1,
                payload={
                    "text": text,
                    "speak": True,
                    "playback_generation": playback_generation,
                    **metadata,
                },
            )
            try:
                await self._sidecar.speak(event)  # type: ignore[attr-defined]
                return
            except Exception as exc:
                await self._disable_sidecar()
                await self._emit(
                    VoiceEventType.FRONTEND_STATE,
                    {
                        "status": "degraded",
                        "reason": "sidecar_tts_failed",
                        "error": sanitize_realtime_voice_error(exc),
                        "sidecar": False,
                    },
                )

        tts_started_at = time.perf_counter()
        file_path = await asyncio.to_thread(self._tts_sync, text)
        tts_synthesis_ms = int(round((time.perf_counter() - tts_started_at) * 1000))
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
                first_tts_audio_at = time.perf_counter()
                playback_started_at = time.perf_counter()
                playback_start_metrics: dict[str, int] = {}
                include_provider_metrics = not _is_kame_metadata(metadata) or _kame_metrics_policy_provider_spans_enabled(
                    self.config
                )
                if _is_kame_metadata(metadata) and include_provider_metrics:
                    playback_start_metrics["kame_first_tts_audio_to_playback_start_ms"] = _elapsed_perf_ms(
                        first_tts_audio_at,
                        playback_started_at,
                    )
                    metadata["metrics"] = _merge_metrics(metadata.get("metrics"), playback_start_metrics)
                    self._assistant_metadata_by_generation[playback_generation] = metadata
                await self._emit(
                    VoiceEventType.PLAYBACK_STARTED,
                    {"playback_generation": playback_generation, **metadata},
                )
                payload = AudioChunk(codec=VoiceAudioCodec.OPUS, data=data).to_payload()
                payload["mime_type"] = _mime_type_for_path(file_path)
                payload["playback_generation"] = playback_generation
                payload.update(metadata)
                existing_metrics = payload.get("metrics")
                metrics = dict(existing_metrics) if isinstance(existing_metrics, dict) else {}
                if include_provider_metrics:
                    metrics["tts_synthesis_ms"] = tts_synthesis_ms
                if playback_start_metrics:
                    metrics.update(playback_start_metrics)
                first_audio_metrics = self._kame_first_audio_metrics(playback_generation, metadata)
                if first_audio_metrics:
                    metrics.update(first_audio_metrics)
                    metadata["metrics"] = _merge_metrics(metadata.get("metrics"), metrics)
                    self._assistant_metadata_by_generation[playback_generation] = metadata
                payload["metrics"] = metrics
                await self._emit(
                    VoiceEventType.AUDIO_OUTPUT_CHUNK,
                    payload,
                )
                await self._emit(
                    VoiceEventType.PLAYBACK_STOPPED,
                    {"playback_generation": playback_generation, **metadata},
                )
                await self._emit(
                    VoiceEventType.ASSISTANT_AUDIO_END,
                    {"playback_generation": playback_generation, **metadata},
                )
        finally:
            try:
                os.unlink(file_path)
            except OSError:
                pass

    def _kame_first_audio_metrics(self, playback_generation: int, metadata: dict) -> dict[str, int]:
        if playback_generation in self._first_audio_metric_generations:
            return {}
        if not _is_kame_metadata(metadata):
            return {}
        if not _kame_metrics_policy_provider_spans_enabled(self.config):
            return {}
        decision_at = self._interface_decision_at_by_generation.get(playback_generation)
        if decision_at is None:
            return {}
        first_audio_at = time.perf_counter()
        elapsed = _elapsed_perf_ms(decision_at, first_audio_at)
        self._first_audio_metric_generations.add(playback_generation)
        metrics = {"kame_interface_decision_to_first_audio_ms": elapsed}
        existing_metrics = metadata.get("metrics")
        if isinstance(existing_metrics, Mapping):
            speech_end_to_decision = _nonnegative_int_metrics(existing_metrics).get(
                "kame_speech_end_to_interface_decision_ms"
            )
            if speech_end_to_decision is not None:
                metrics["kame_speech_end_to_first_audio_ms"] = speech_end_to_decision + elapsed
        oracle_first_token_at = self._oracle_first_token_at_by_generation.get(playback_generation)
        if oracle_first_token_at is not None:
            metrics["kame_oracle_first_token_to_first_tts_audio_ms"] = _elapsed_perf_ms(
                oracle_first_token_at,
                first_audio_at,
            )
        route = str(metadata.get("kame_route") or "")
        if route in {KameRoute.LOCAL.value, KameRoute.REJECT_OR_CLARIFY.value}:
            metrics["kame_interface_decision_to_local_first_audio_ms"] = elapsed
            if "kame_speech_end_to_first_audio_ms" in metrics:
                metrics["kame_speech_end_to_local_first_audio_ms"] = metrics[
                    "kame_speech_end_to_first_audio_ms"
                ]
        elif route == KameRoute.DEFER.value and metadata.get("kame_interface_already_said"):
            metrics["kame_interface_decision_to_defer_first_audio_ms"] = elapsed
            if "kame_speech_end_to_first_audio_ms" in metrics:
                metrics["kame_speech_end_to_defer_first_audio_ms"] = metrics[
                    "kame_speech_end_to_first_audio_ms"
                ]
        return metrics

    def _kame_sidecar_audio_payload_with_metrics(self, payload: dict) -> dict:
        generation = _payload_generation(payload)
        if generation is None:
            return payload
        metadata = self._assistant_metadata_by_generation.get(generation)
        if not isinstance(metadata, dict) or not _is_kame_metadata(metadata):
            return payload
        for key, value in metadata.items():
            if key == "metrics" or key in payload:
                continue
            payload[key] = value
        existing_metrics = payload.get("metrics")
        metrics = _merge_metrics(metadata.get("metrics"), existing_metrics)
        first_audio_metrics = self._kame_first_audio_metrics(generation, metadata)
        if first_audio_metrics:
            metrics.update(first_audio_metrics)
        if metrics:
            payload["metrics"] = metrics
            metadata["metrics"] = _merge_metrics(metadata.get("metrics"), metrics)
            self._assistant_metadata_by_generation[generation] = metadata
        return payload

    def _interface_decision_metric_start(self, playback_generation: int, *, fallback: float) -> float:
        return self._interface_decision_at_by_generation.get(playback_generation, fallback)

    def _tts_sync(self, text: str) -> str:
        from tools.tts_tool import text_to_speech_tool

        raw = text_to_speech_tool(text)
        result = json.loads(raw) if isinstance(raw, str) else raw
        if not result.get("success"):
            raise RuntimeError(result.get("error") or "speech synthesis failed")
        return str(result.get("file_path") or "")

    async def _emit_oracle_hint(
        self,
        *,
        text: str,
        delta: str,
        final: bool,
        playback_generation: int,
        metadata: Mapping[str, Any],
        accepted: bool = False,
        metrics: Optional[Mapping[str, int]] = None,
    ) -> None:
        if playback_generation != self._playback_generation:
            return
        voice_capability_corrected = False
        if _is_kame_metadata(metadata) and kame_local_reply_denies_voice_capability(text or delta):
            text = _kame_voice_capability_correction_text(self.config)
            delta = text if delta else ""
            voice_capability_corrected = True
        payload: dict[str, Any] = {
            "text": text,
            "delta": delta,
            "final": final,
            "source": "hermes",
            "playback_generation": playback_generation,
            **_kame_route_metrics_payload(metadata, oracle_called=True, extra_metrics=metrics),
        }
        if voice_capability_corrected:
            payload["voice_capability_corrected"] = True
        if accepted:
            payload["accepted"] = True
        if _is_kame_metadata(metadata):
            if accepted:
                oracle_event_type = VoiceEventType.ORACLE_ACCEPTED
            elif final:
                oracle_event_type = VoiceEventType.ORACLE_RESPONSE_FINAL
            else:
                oracle_event_type = VoiceEventType.ORACLE_RESPONSE_PARTIAL
            oracle_payload = {
                **_kame_interface_payload_from_metadata(metadata),
                "text": text,
                "delta": delta,
                "final": final,
                "source": "hermes",
                "playback_generation": playback_generation,
                **_kame_route_metrics_payload(metadata, oracle_called=True, extra_metrics=metrics),
            }
            if voice_capability_corrected:
                oracle_payload["voice_capability_corrected"] = True
            if accepted:
                oracle_payload["accepted"] = True
            oracle_event = await self._emit(oracle_event_type, oracle_payload)
            if oracle_event is not None and self._sidecar is not None:
                await self._send_sidecar_event(oracle_event)
        event = await self._emit(VoiceEventType.ORACLE_HINT, payload)
        if event is not None and self._sidecar is not None:
            await self._send_sidecar_event(event)

    async def _emit_oracle_tool_event(
        self,
        *,
        event_type: VoiceEventType,
        payload: Mapping[str, Any],
        playback_generation: int,
        metadata: Mapping[str, Any],
        metrics: Optional[Mapping[str, int]] = None,
    ) -> None:
        if playback_generation != self._playback_generation or not _is_kame_metadata(metadata):
            return
        event_payload = {
            **_kame_interface_payload_from_metadata(metadata),
            **_oracle_tool_event_payload(payload),
            "source": "hermes",
            "playback_generation": playback_generation,
            **_kame_route_metrics_payload(metadata, oracle_called=True, extra_metrics=metrics),
        }
        event = await self._emit(event_type, event_payload)
        if event is not None and self._sidecar is not None:
            await self._send_sidecar_event(event)

    async def _emit_oracle_error(
        self,
        playback_generation: int,
        metadata: Mapping[str, Any],
        *,
        reason: str,
        error: str,
    ) -> None:
        if playback_generation != self._playback_generation or not _is_kame_metadata(metadata):
            return
        payload = {
            **_kame_interface_payload_from_metadata(metadata),
            "reason": reason,
            "error": sanitize_realtime_voice_error(error),
            "source": "hermes",
            "playback_generation": playback_generation,
            **_kame_route_metrics_payload(metadata, oracle_called=True),
        }
        event = await self._emit(VoiceEventType.ORACLE_ERROR, payload)
        if event is not None and self._sidecar is not None:
            await self._send_sidecar_event(event)
        await self._emit_session_metrics(
            playback_generation,
            metadata,
            oracle_called=True,
            outcome=f"oracle_error:{reason}",
        )

    async def _emit_oracle_cancelled(
        self,
        *,
        playback_generation: int,
        cancelled_playback_generation: int,
        metadata: Mapping[str, Any],
        reason: str,
        cancellation_token: str,
    ) -> None:
        if not _is_kame_metadata(metadata):
            return
        payload = {
            **_kame_interface_payload_from_metadata(metadata),
            "reason": "oracle_cancelled",
            "cancel_reason": reason or "client",
            "error": "oracle request cancelled by realtime voice interruption",
            "source": "hermes",
            "playback_generation": playback_generation,
            "cancelled_playback_generation": cancelled_playback_generation,
            "cancellation_token": cancellation_token,
            **_kame_route_metrics_payload(metadata, oracle_called=True),
        }
        event = await self._emit(VoiceEventType.ORACLE_ERROR, payload)
        if event is not None and self._sidecar is not None:
            await self._send_sidecar_event(event)
        await self._emit_session_metrics(
            playback_generation,
            metadata,
            oracle_called=True,
            outcome="oracle_cancelled",
        )

    async def _emit_session_metrics(
        self,
        playback_generation: int,
        metadata: Mapping[str, Any],
        *,
        oracle_called: bool,
        outcome: str,
        extra_metrics: Optional[Mapping[str, int]] = None,
        local_reply: bool = False,
    ) -> None:
        if playback_generation != self._playback_generation or not _is_kame_metadata(metadata):
            return
        if not _kame_metrics_policy_turn_spans_enabled(self.config):
            return
        metrics = _kame_route_metrics(
            metadata,
            oracle_called=oracle_called,
            extra_metrics=extra_metrics,
        )
        if not metrics:
            return
        payload = {
            **_kame_interface_payload_from_metadata(metadata),
            "playback_generation": playback_generation,
            "outcome": outcome,
            "oracle_called": bool(oracle_called),
            "local_reply": bool(local_reply),
            "metrics": metrics,
        }
        event = await self._emit(VoiceEventType.SESSION_METRICS, payload)
        if event is not None and self._sidecar is not None:
            await self._send_sidecar_event(event)

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
        await self._emit_caption_alias_if_needed(event)
        await self._emit_audio_alias_if_needed(event)
        return event

    async def _emit_caption_alias_if_needed(self, event: VoiceEvent) -> None:
        if not _caption_alias_events_enabled(self.config):
            return
        if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
            alias_type = VoiceEventType.ASSISTANT_CAPTION_PARTIAL
        elif event.type == VoiceEventType.ASSISTANT_COMMIT and event.payload.get("interrupted") is not True:
            alias_type = VoiceEventType.ASSISTANT_CAPTION_FINAL
        else:
            return
        text = str(event.payload.get("text") or "").strip()
        if not text:
            return
        self._sequence += 1
        alias = VoiceEvent(
            type=alias_type,
            session_id=event.session_id,
            sequence=self._sequence,
            payload={
                **dict(event.payload),
                "text": text,
                "caption_alias_for": event.type.value,
            },
        )
        await put_realtime_voice_event(self._events, alias)

    async def _emit_audio_alias_if_needed(self, event: VoiceEvent) -> None:
        if not _audio_alias_events_enabled(self.config):
            return
        if event.type != VoiceEventType.AUDIO_OUTPUT_CHUNK:
            return
        self._sequence += 1
        alias = VoiceEvent(
            type=VoiceEventType.ASSISTANT_AUDIO_CHUNK,
            session_id=event.session_id,
            sequence=self._sequence,
            payload={
                **dict(event.payload),
                "audio_alias_for": event.type.value,
            },
        )
        await put_realtime_voice_event(self._events, alias)


class KameInterfaceOracleEngine(TextOracleTTSEngine):
    """KAME reflex + Hermes oracle engine.

    This first implementation reuses the hardened text-oracle/TTS lifecycle while
    changing the engine contract and oracle request shape. A Gemma E2B reflex
    sidecar can now target this engine by emitting audio-derived intent fields
    and optional oracle-verbatim ASR evidence in transcript final payloads.
    """

    @property
    def kind(self) -> RealtimeVoiceEngineKind:
        return RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE


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


def _realtime_voice_fail_closed(config: Optional[RealtimeVoiceSessionConfig]) -> bool:
    return str(getattr(config, "fallback_policy", "") or "").strip().lower() == "fail_closed"


def _kame_cancellation_token(config: Optional[RealtimeVoiceSessionConfig], playback_generation: int) -> str:
    if config is None or config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
        return ""
    session_id = str(config.session_id or "voice").strip() or "voice"
    return f"{session_id}:{playback_generation}:cancel"


def _allow_kame_transcript_events(config: Optional[RealtimeVoiceSessionConfig]) -> bool:
    if config is None or config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
        return True
    if config.asr_mode.value in {"debug", "fallback"}:
        return True
    return str(config.interface_audio_input or "").strip().lower() == "text_fallback"


def _kame_transcript_final_can_start_turn(
    config: Optional[RealtimeVoiceSessionConfig],
    payload: Mapping[str, Any],
) -> bool:
    if _allow_kame_transcript_events(config):
        return True
    source = str(payload.get("source") or "").strip().lower()
    if source in {"gemini_live_tool", "openai_realtime_tool", "kame_interface_tool"}:
        return True
    return False


def _turn_acknowledgement_text(config: Optional[RealtimeVoiceSessionConfig]) -> str:
    if config is None:
        return ""
    acknowledgement: Any = config.turn_acknowledgement
    if not acknowledgement:
        metadata = config.metadata if isinstance(config.metadata, Mapping) else {}
        acknowledgement = metadata.get("turn_acknowledgement")
    if not isinstance(acknowledgement, Mapping):
        return ""
    if not _metadata_bool(acknowledgement.get("enabled"), default=False):
        return ""
    text = str(acknowledgement.get("text") or "One moment.").strip()
    if not text:
        return ""
    return text[:120]


def _oracle_reflex_narration_text(
    config: Optional[RealtimeVoiceSessionConfig],
    oracle_request: Optional[KameOracleRequest],
) -> str:
    if (
        oracle_request is not None
        and oracle_request.route == KameRoute.DEFER
        and oracle_request.interface_already_said
    ):
        return oracle_request.interface_already_said.strip()[:120]
    if oracle_request is not None and oracle_request.route == KameRoute.DEFER:
        return _kame_oracle_handoff_narration(config, oracle_request)
    return _turn_acknowledgement_text(config)


def _kame_oracle_handoff_narration(
    config: Optional[RealtimeVoiceSessionConfig],
    request: KameOracleRequest,
) -> str:
    task = _compact_kame_summary_text(request.oracle_text or request.intent)
    if not task:
        return ""
    task = task.rstrip(".!?。！？")
    prefix = "I'm"
    if config is not None and isinstance(config.metadata, Mapping):
        configured = str(config.metadata.get("kame_oracle_handoff_prefix") or "").strip()
        if configured:
            prefix = configured.rstrip()
    lower_task = task[:1].lower() + task[1:]
    verb_map = {
        "check ": "checking ",
        "look ": "looking ",
        "look up ": "looking up ",
        "find ": "finding ",
        "run ": "running ",
        "call ": "calling ",
        "open ": "opening ",
        "search ": "searching ",
        "verify ": "verifying ",
        "review ": "reviewing ",
        "diagnose ": "diagnosing ",
        "fix ": "fixing ",
        "provision ": "provisioning ",
        "create ": "creating ",
        "buy ": "buying ",
    }
    action = ""
    for source, replacement in verb_map.items():
        if lower_task.startswith(source):
            action = f"{replacement}{lower_task[len(source):]}".strip()
            break
    if action:
        if prefix.lower().endswith((" to", " to:")):
            text = f"{prefix} {lower_task}."
        else:
            text = f"{prefix} {action}."
    else:
        text = f"{prefix} checking on {task}."
    return text[:160]


def _compact_kame_summary_text(text: str) -> str:
    compacted = " ".join(str(text or "").split())
    return compacted[:220]


def _kame_voice_capability_correction_text(config: Optional[RealtimeVoiceSessionConfig]) -> str:
    if config is not None and isinstance(config.metadata, Mapping):
        text = str(config.metadata.get("voice_capability_correction_text") or "").strip()
        if text:
            return text[:160]
    return "Voice is active here; I can hear you and speak in this channel."


def _kame_local_reply(request: Optional[KameOracleRequest]) -> str:
    if request is None:
        return ""
    if request.route not in {KameRoute.LOCAL, KameRoute.REJECT_OR_CLARIFY}:
        return ""
    return request.local_reply.strip()


def _kame_oracle_job_status_requested(request: KameOracleRequest) -> bool:
    haystack = " ".join(
        str(value or "")
        for value in (
            request.intent,
            request.transcript,
            request.local_reply,
            request.oracle_text,
        )
    ).lower()
    if not haystack.strip():
        return False
    status_phrases = (
        "what are you working on",
        "what are you doing",
        "what finished",
        "what completed",
        "what happened with the last job",
        "what happened with my last job",
        "what happened with that job",
        "what happened with that task",
        "what did you finish",
        "what did you complete",
        "last job",
        "last task",
        "job status",
        "jobs status",
        "oracle jobs",
        "background jobs",
        "active jobs",
        "running jobs",
        "what's running",
        "what is running",
    )
    if any(phrase in haystack for phrase in status_phrases):
        return True
    return "status" in haystack and any(token in haystack for token in ("job", "jobs", "task", "tasks", "oracle"))


def _kame_oracle_job_status_text(status: Mapping[str, Any]) -> str:
    capacity = status.get("capacity") if isinstance(status.get("capacity"), Mapping) else {}
    jobs = [dict(job) for job in status.get("jobs", []) if isinstance(job, Mapping)]
    active = int(capacity.get("active") or 0)
    running = int(capacity.get("running") or 0)
    queued = int(capacity.get("queued") or 0)
    waiting_for_approval = int(capacity.get("waiting_for_approval") or 0)
    cancel_requested = int(capacity.get("cancel_requested") or 0)
    max_concurrent = capacity.get("max_concurrent") or "?"
    active_jobs = [
        job
        for job in jobs
        if str(job.get("state") or "") in {"running", "queued", "waiting_for_approval", "cancel_requested"}
    ]
    if not active_jobs:
        recent_labels = _kame_oracle_job_recent_terminal_labels(jobs)
        if recent_labels:
            return "No oracle jobs are running or queued right now. Recent: " + " | ".join(recent_labels)
        if jobs:
            return "No oracle jobs are running or queued right now."
        return "I don't have any oracle jobs yet."
    if active and active != running:
        fragments = [f"{active} active out of {max_concurrent}", f"{running} running"]
    else:
        fragments = [f"{running} running out of {max_concurrent}"]
    if queued:
        fragments.append(f"{queued} queued")
    if waiting_for_approval:
        fragments.append(f"{waiting_for_approval} waiting for approval")
    if cancel_requested:
        fragments.append(f"{cancel_requested} cancelling")
    headline = "Oracle jobs: " + ", ".join(fragments) + "."
    labels = []
    for job in active_jobs[:3]:
        label = str(job.get("spoken_status") or job.get("intent") or "").strip()
        state = str(job.get("state") or "").strip()
        if label and state:
            labels.append(f"{state}: {label[:90]}")
        elif label:
            labels.append(label[:90])
    if labels:
        return headline + " " + " | ".join(labels)
    return headline


def _kame_oracle_job_recent_terminal_labels(jobs: list[dict[str, Any]]) -> list[str]:
    labels = []
    for job in reversed(jobs):
        state = str(job.get("state") or "").strip()
        if state not in {"completed", "failed", "cancelled"}:
            continue
        label = _kame_oracle_job_terminal_label(job, state)
        if label:
            labels.append(label)
        if len(labels) >= 3:
            break
    return labels


def _kame_oracle_job_terminal_label(job: Mapping[str, Any], state: str) -> str:
    if state == "completed":
        text = str(job.get("result_summary") or job.get("spoken_status") or job.get("intent") or "").strip()
    elif state == "failed":
        text = str(job.get("error") or job.get("spoken_status") or job.get("intent") or "").strip()
    else:
        text = str(job.get("cancel_reason") or job.get("spoken_status") or job.get("intent") or "").strip()
    text = " ".join(text.split())
    if not text:
        return state
    return f"{state}: {text[:120]}"


def _kame_oracle_job_control_operation(
    request: KameOracleRequest,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    haystack = _oracle_job_control_text(request)
    if not haystack:
        return {}
    jobs = _oracle_job_control_active_jobs(status)
    if _oracle_job_control_cancel_all_requested(haystack):
        return {
            "kind": "cancel_all",
            "reason": "spoken request to stop everything",
        }
    if not jobs:
        return {}
    priority = _oracle_job_control_priority(haystack)
    if priority:
        job = _oracle_job_control_match_job(haystack, jobs)
        if job:
            return {
                "kind": "priority",
                "job_id": str(job.get("job_id") or ""),
                "priority": priority,
                "reason": f"spoken request to set {priority} priority",
            }
    update_text = _oracle_job_control_update_text(haystack)
    if update_text:
        job = _oracle_job_control_match_job(haystack, jobs)
        if job:
            return {
                "kind": "update",
                "job_id": str(job.get("job_id") or ""),
                "update_text": update_text,
                "reason": "spoken update to oracle job",
            }
    if _oracle_job_control_cancel_one_requested(haystack):
        job = _oracle_job_control_match_job(haystack, jobs)
        if job:
            return {
                "kind": "cancel",
                "job_id": str(job.get("job_id") or ""),
                "reason": "spoken request to cancel oracle job",
            }
    return {}


def _oracle_job_control_text(request: KameOracleRequest) -> str:
    for value in (request.intent, request.transcript, request.oracle_text, request.local_reply):
        text = " ".join(str(value or "").split()).lower()
        if text:
            return text
    return ""


def _oracle_job_control_active_jobs(status: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = status.get("jobs") if isinstance(status.get("jobs"), list) else []
    active_states = {"running", "queued", "waiting_for_approval", "cancel_requested"}
    return [
        dict(job)
        for job in jobs
        if isinstance(job, Mapping) and str(job.get("state") or "") in active_states
    ]


def _oracle_job_control_cancel_all_requested(text: str) -> bool:
    if _oracle_job_control_is_playback_only_stop(text):
        return False
    phrases = (
        "stop everything",
        "cancel everything",
        "cancel all",
        "stop all jobs",
        "stop all tasks",
        "cancel all jobs",
        "cancel all tasks",
        "kill all jobs",
        "kill all tasks",
    )
    return any(phrase in text for phrase in phrases)


def _oracle_job_control_cancel_one_requested(text: str) -> bool:
    if _oracle_job_control_is_playback_only_stop(text):
        return False
    if "cancel" in text:
        return True
    if "stop that" in text or "stop this" in text:
        return True
    return "stop" in text and any(token in text for token in (" job", " jobs", " task", " tasks"))


def _oracle_job_control_is_playback_only_stop(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "stop talking",
            "stop speaking",
            "be quiet",
            "shut up",
        )
    )


def _oracle_job_control_priority(text: str) -> str:
    if "priority" not in text and not any(token in text for token in ("urgent", "important", "background")):
        return ""
    if any(token in text for token in ("highest", "urgent", "important", "high priority")):
        return "high"
    if any(token in text for token in ("low priority", "background", "later")):
        return "low"
    if any(token in text for token in ("normal priority", "default priority", "medium priority")):
        return "normal"
    return "high" if "priority" in text and "make" in text else ""


def _oracle_job_control_update_text(text: str) -> str:
    update_prefixes = (
        "also ",
        "and also ",
        "add that ",
        "add this ",
        "add ",
        "include ",
        "update that ",
        "update this ",
        "tell it to ",
        "ask it to ",
    )
    for prefix in update_prefixes:
        if text.startswith(prefix):
            return text[len(prefix):].strip(" .")
    if " also " in text:
        return text.split(" also ", 1)[1].strip(" .")
    return ""


def _oracle_job_control_match_job(text: str, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    if not jobs:
        return {}
    for job in jobs:
        job_id = str(job.get("job_id") or "").strip().lower()
        if job_id and job_id in text:
            return job
    ordinal = _oracle_job_control_ordinal(text)
    if ordinal is not None and 0 <= ordinal < len(jobs):
        return jobs[ordinal]
    text_terms = _oracle_job_control_terms(text)
    best: tuple[int, dict[str, Any]] = (0, {})
    for job in jobs:
        label = _oracle_job_control_label(job)
        score = len(text_terms.intersection(_oracle_job_control_terms(label)))
        if score > best[0]:
            best = (score, job)
    if best[0] > 0:
        return best[1]
    if len(jobs) == 1:
        return jobs[0]
    update_fallback = text.startswith(("also ", "and also ", "add ", "include ", "tell it ", "ask it "))
    if text_terms and not update_fallback:
        return {}
    if _oracle_job_control_has_fallback_reference(text) or update_fallback:
        return jobs[-1]
    return {}


def _oracle_job_control_ordinal(text: str) -> Optional[int]:
    patterns = (
        (0, r"\b(?:first|1st|task one|job one|task 1|job 1)\b"),
        (1, r"\b(?:second|2nd|task two|job two|task 2|job 2)\b"),
        (2, r"\b(?:third|3rd|task three|job three|task 3|job 3)\b"),
        (3, r"\b(?:fourth|4th|task four|job four|task 4|job 4)\b"),
        (4, r"\b(?:fifth|5th|task five|job five|task 5|job 5)\b"),
    )
    for index, pattern in patterns:
        if re.search(pattern, text):
            return index
    return None


def _oracle_job_control_has_fallback_reference(text: str) -> bool:
    return re.search(r"\b(?:that|this|current|latest|last|it)\b", text) is not None


def _oracle_job_control_terms(text: str) -> set[str]:
    stop = {
        "a",
        "all",
        "also",
        "and",
        "cancel",
        "check",
        "current",
        "do",
        "for",
        "high",
        "highest",
        "it",
        "job",
        "jobs",
        "low",
        "make",
        "normal",
        "priority",
        "run",
        "running",
        "set",
        "stop",
        "task",
        "tasks",
        "that",
        "the",
        "this",
        "to",
        "urgent",
    }
    terms: set[str] = set()
    for token in re.findall(r"[a-z][a-z0-9_-]*", str(text or "").lower()):
        if token in stop or len(token) <= 2:
            continue
        terms.add(token)
        normalized = _oracle_job_control_normalized_term(token)
        if normalized and normalized not in stop and len(normalized) > 2:
            terms.add(normalized)
    return terms


def _oracle_job_control_normalized_term(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _oracle_job_control_label(job: Mapping[str, Any]) -> str:
    return " ".join(
        str(value or "").strip()
        for value in (
            job.get("spoken_status"),
            job.get("intent"),
        )
        if str(value or "").strip()
    )[:120]


def _async_oracle_jobs_enabled(config: Optional[RealtimeVoiceSessionConfig]) -> bool:
    if config is None or config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
        return False
    if not isinstance(config.oracle_jobs, Mapping) or not config.oracle_jobs:
        return False
    return _metadata_bool(config.oracle_jobs.get("enabled"), default=True)


def _oracle_job_terminal_speech_enabled(config: Optional[RealtimeVoiceSessionConfig]) -> bool:
    if config is None or not isinstance(config.oracle_jobs, Mapping):
        return False
    return _metadata_bool(config.oracle_jobs.get("speak_terminal_results"), default=True)


def _oracle_job_failed_status_text(payload: Mapping[str, Any]) -> str:
    intent = str(payload.get("intent") or payload.get("spoken_status") or "").strip()
    intent = intent.rstrip(".!?。！？")
    error = str(payload.get("error") or "").strip()
    if intent and error:
        return f"I couldn't finish {intent}: {error}"
    if intent:
        return f"I couldn't finish {intent}."
    if error:
        return f"That oracle job failed: {error}"
    return "That oracle job failed."


def _oracle_jobs_config_int(config: RealtimeVoiceSessionConfig, key: str, *, default: int) -> int:
    if not isinstance(config.oracle_jobs, Mapping):
        return default
    value = config.oracle_jobs.get(key)
    if isinstance(value, bool) or value is None:
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _oracle_jobs_config_str(config: RealtimeVoiceSessionConfig, key: str, *, default: str) -> str:
    if not isinstance(config.oracle_jobs, Mapping):
        return default
    value = str(config.oracle_jobs.get(key) or "").strip()
    return value or default


def _oracle_jobs_config_optional_str(config: RealtimeVoiceSessionConfig, key: str) -> Optional[str]:
    if not isinstance(config.oracle_jobs, Mapping):
        return None
    value = str(config.oracle_jobs.get(key) or "").strip()
    return value or None


def _oracle_jobs_config_float(config: Optional[RealtimeVoiceSessionConfig], key: str, *, default: float) -> float:
    if config is None or not isinstance(config.oracle_jobs, Mapping):
        return default
    value = config.oracle_jobs.get(key)
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return parsed


_ORACLE_JOB_EVENT_VOICE_TYPES: Mapping[OracleJobEventType, VoiceEventType] = {
    OracleJobEventType.ACCEPTED: VoiceEventType.ORACLE_JOB_ACCEPTED,
    OracleJobEventType.QUEUED: VoiceEventType.ORACLE_JOB_QUEUED,
    OracleJobEventType.STARTED: VoiceEventType.ORACLE_JOB_STARTED,
    OracleJobEventType.PROGRESS: VoiceEventType.ORACLE_JOB_PROGRESS,
    OracleJobEventType.WAITING_FOR_APPROVAL: VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL,
    OracleJobEventType.COMPLETED: VoiceEventType.ORACLE_JOB_COMPLETED,
    OracleJobEventType.FAILED: VoiceEventType.ORACLE_JOB_FAILED,
    OracleJobEventType.CANCEL_REQUESTED: VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED,
    OracleJobEventType.CANCELLED: VoiceEventType.ORACLE_JOB_CANCELLED,
}


def _voice_event_type_for_oracle_job_event(event_type: OracleJobEventType) -> Optional[VoiceEventType]:
    return _ORACLE_JOB_EVENT_VOICE_TYPES.get(event_type)


def _oracle_job_payload(job: OracleJob) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": job.job_id,
        "session_id": job.session_id,
        "state": job.state.value,
        "priority": job.priority,
        "route": job.route,
        "intent": job.reflex_intent,
        "oracle_text": job.oracle_text,
        "interface_already_said": job.interface_already_said,
        "requested_response_style": dict(job.requested_response_style or {}),
    }
    request = job.request
    if request is not None:
        playback_generation = _playback_generation_from_turn_id(request.turn_id)
        payload.update(
            {
                "turn_id": request.turn_id,
                "source": request.source,
                "mode": request.mode,
                "urgency": request.urgency,
                "playback_generation": playback_generation,
                "intent_source": request.intent_source,
                "oracle_text_source": request.oracle_text_source,
            }
        )
        if request.user_id:
            payload["user_id"] = request.user_id
        if request.cancellation_token:
            payload["cancellation_token"] = request.cancellation_token
    return payload


def _oracle_job_update_event_payload(job: OracleJob, *, reason: str) -> dict[str, Any]:
    status = job.to_status()
    payload: dict[str, Any] = {
        "job_id": job.job_id,
        "priority": job.priority,
        "state": job.state.value,
        "reason": str(reason or "oracle job updated")[:240],
        "update_count": len(job.updates),
    }
    latest_update = str(status.get("latest_update") or "").strip()
    if latest_update:
        payload["latest_update"] = latest_update
    return payload


def _playback_generation_from_turn_id(turn_id: str) -> int:
    try:
        return int(str(turn_id).rsplit(":", 1)[-1])
    except (TypeError, ValueError):
        return 0


def _kame_interface_payload(request: KameOracleRequest, playback_generation: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": request.session_id,
        "turn_id": request.turn_id,
        "route": request.route.value,
        "intent": request.intent,
        "intent_source": request.intent_source,
        "text": request.oracle_text,
        "oracle_text_source": request.oracle_text_source,
        "playback_generation": playback_generation,
        "source": request.source,
        "mode": request.mode,
        "urgency": request.urgency,
        "transcript_source": request.transcript_source,
        "requested_response_style": dict(request.requested_response_style),
    }
    if request.route_confidence is not None:
        payload["route_confidence"] = request.route_confidence
    if request.user_id:
        payload["user_id"] = request.user_id
    if request.local_reply:
        payload["local_reply"] = request.local_reply
    if request.transcript:
        payload["transcript"] = request.transcript
    if request.transcript_confidence is not None:
        payload["transcript_confidence"] = request.transcript_confidence
    if request.asr_transcript:
        payload["asr_transcript"] = request.asr_transcript
        payload["asr_transcript_source"] = request.asr_transcript_source or "asr"
    if request.asr_transcript_confidence is not None:
        payload["asr_transcript_confidence"] = request.asr_transcript_confidence
    if request.interface_already_said:
        payload["interface_already_said"] = request.interface_already_said
    if request.conversation_summary:
        payload["conversation_summary"] = request.conversation_summary
    if request.cancellation_token:
        payload["cancellation_token"] = request.cancellation_token
    if request.reflex_validation_error:
        payload["reflex_validation_error"] = request.reflex_validation_error
    if request.interface_input_source:
        payload["interface_input_source"] = request.interface_input_source
    if request.interface_audio_input_fallback:
        payload["interface_audio_input_fallback"] = True
    if request.reflex_provider:
        payload["reflex_provider"] = request.reflex_provider
    return payload


def _oracle_request_for_job(job: OracleJob, request: KameOracleRequest) -> KameOracleRequest:
    updates = tuple(
        dict.fromkeys(
            (
                *(
                    str(update or "").strip()
                    for update in request.job_updates
                    if str(update or "").strip()
                ),
                *_oracle_job_update_texts(job),
            )
        )
    )
    if not updates and request.priority == job.priority:
        return request
    return replace(
        request,
        priority=job.priority,
        job_updates=updates,
    )


def _oracle_job_update_texts(job: OracleJob) -> tuple[str, ...]:
    return tuple(
        str(update.get("text") or "").strip()
        for update in job.updates
        if str(update.get("text") or "").strip()
    )


def _kame_interface_payload_with_metrics(
    request: KameOracleRequest,
    playback_generation: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _kame_interface_payload(request, playback_generation)
    input_generation = _payload_input_generation(dict(metadata))
    if input_generation is not None:
        payload["input_generation"] = input_generation
    metrics = metadata.get("metrics") if isinstance(metadata, Mapping) else None
    if isinstance(metrics, Mapping):
        sanitized = _nonnegative_int_metrics(metrics)
        if sanitized:
            payload["metrics"] = sanitized
    return payload


def _kame_defer_reply_payload_with_metrics(
    request: KameOracleRequest,
    playback_generation: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _kame_interface_payload_with_metrics(request, playback_generation, metadata)
    narration = request.interface_already_said.strip()
    if narration:
        payload["text"] = narration
        payload["reflex_narration_text"] = narration
        payload["oracle_text"] = request.oracle_text
        payload["oracle_text_source"] = request.oracle_text_source
    return payload


def _kame_interface_payload_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    oracle_text_source = str(metadata.get("kame_oracle_text_source") or "").strip()
    transcript = str(metadata.get("kame_transcript") or "")
    intent = str(metadata.get("kame_intent") or "")
    asr_transcript = str(metadata.get("kame_asr_transcript") or "")
    text = transcript or intent
    if oracle_text_source.lower().startswith("asr") and asr_transcript:
        text = asr_transcript
    payload: dict[str, Any] = {
        "session_id": str(metadata.get("kame_session_id") or ""),
        "turn_id": str(metadata.get("kame_turn_id") or ""),
        "route": str(metadata.get("kame_route") or ""),
        "intent": intent,
        "intent_source": str(metadata.get("kame_intent_source") or ""),
        "text": text,
        "source": str(metadata.get("kame_source") or ""),
        "mode": str(metadata.get("kame_mode") or ""),
        "urgency": str(metadata.get("kame_urgency") or ""),
        "transcript_source": str(metadata.get("kame_transcript_source") or ""),
    }
    if oracle_text_source:
        payload["oracle_text_source"] = oracle_text_source
    if metadata.get("kame_user_id"):
        payload["user_id"] = str(metadata.get("kame_user_id"))
    if metadata.get("kame_route_confidence") is not None:
        payload["route_confidence"] = metadata.get("kame_route_confidence")
    if metadata.get("kame_local_reply"):
        payload["local_reply"] = str(metadata.get("kame_local_reply"))
    if metadata.get("kame_transcript"):
        payload["transcript"] = str(metadata.get("kame_transcript"))
    if metadata.get("kame_transcript_confidence") is not None:
        payload["transcript_confidence"] = metadata.get("kame_transcript_confidence")
    if metadata.get("kame_asr_transcript"):
        payload["asr_transcript"] = str(metadata.get("kame_asr_transcript"))
        payload["asr_transcript_source"] = str(metadata.get("kame_asr_transcript_source") or "asr")
    if metadata.get("kame_asr_transcript_confidence") is not None:
        payload["asr_transcript_confidence"] = metadata.get("kame_asr_transcript_confidence")
    if metadata.get("kame_interface_already_said"):
        payload["interface_already_said"] = str(metadata.get("kame_interface_already_said"))
    if metadata.get("kame_conversation_summary"):
        payload["conversation_summary"] = str(metadata.get("kame_conversation_summary"))
    if metadata.get("kame_cancellation_token"):
        payload["cancellation_token"] = str(metadata.get("kame_cancellation_token"))
    if metadata.get("kame_reflex_validation_error"):
        payload["reflex_validation_error"] = str(metadata.get("kame_reflex_validation_error"))
    if metadata.get("kame_interface_input_source"):
        payload["interface_input_source"] = str(metadata.get("kame_interface_input_source"))
    if metadata.get("kame_interface_audio_input_fallback") is True:
        payload["interface_audio_input_fallback"] = True
    if metadata.get("kame_reflex_provider"):
        payload["reflex_provider"] = str(metadata.get("kame_reflex_provider"))
    input_generation = _payload_input_generation(dict(metadata))
    if input_generation is not None:
        payload["input_generation"] = input_generation
    if isinstance(metadata.get("kame_requested_response_style"), Mapping):
        payload["requested_response_style"] = dict(metadata.get("kame_requested_response_style") or {})
    return {key: value for key, value in payload.items() if value != ""}


def _kame_routing_policy(config: Optional[RealtimeVoiceSessionConfig]) -> Mapping[str, Any]:
    if config is not None and isinstance(config.routing_policy, Mapping) and config.routing_policy:
        return config.routing_policy
    metadata = config.metadata if config is not None and isinstance(config.metadata, Mapping) else {}
    routing = metadata.get("routing") if isinstance(metadata, Mapping) else {}
    return routing if isinstance(routing, Mapping) else {}


def _kame_interface_partial_payload_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    intent = str(payload.get("intent") or "").strip()
    if not intent:
        return {}
    partial: dict[str, Any] = {
        "intent": intent,
        "intent_source": str(payload.get("intent_source") or "reflex_audio").strip() or "reflex_audio",
    }
    text = str(payload.get("text") or payload.get("transcript") or "").strip()
    if text:
        partial["text"] = text
    route = str(payload.get("route") or "").strip().lower()
    if route:
        partial["route"] = route
    source = str(payload.get("source") or "").strip()
    if source:
        partial["source"] = source
    user_id = str(payload.get("user_id") or "").strip()
    if user_id:
        partial["user_id"] = user_id
    input_generation = _payload_input_generation(dict(payload))
    if input_generation is not None:
        partial["input_generation"] = input_generation
    return partial


def _kame_final_turn_text_from_payload(payload: Mapping[str, Any]) -> str:
    for key in ("text", "transcript", "intent", "local_reply"):
        text = str(payload.get(key) or "").strip()
        if text:
            return text
    return ""


def _kame_local_stt_fallback_payload(
    config: Optional[RealtimeVoiceSessionConfig],
    transcript: str,
) -> dict[str, Any]:
    if config is None or config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
        return {}
    text = str(transcript or "").strip()
    if not text:
        return {}
    return {
        "text": text,
        "intent": text,
        "intent_source": "asr_fallback",
        "route": KameRoute.ORACLE_DIRECT.value,
        "transcript": text,
        "transcript_source": "asr",
        "asr_transcript": text,
        "asr_transcript_source": "asr",
        "interface_audio_input_fallback": True,
        "interface_input_source": "local_stt",
        "reflex_provider": "local_stt",
        "reflex_validation_error": "audio_reflex_unavailable_local_stt_fallback",
    }


def _kame_local_stt_fallback_state_payload(config: Optional[RealtimeVoiceSessionConfig]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "fallback",
        "reason": "kame_audio_reflex_unavailable",
        "provider": "local_stt",
        "fallback_provider": "local_stt",
        "intent_source": "asr_fallback",
        "transcript_source": "asr",
        "interface_audio_input_fallback": True,
        "interface_input_source": "local_stt",
        "reflex_provider": "local_stt",
        "sidecar": False,
    }
    if config is not None:
        interface_audio_input = str(config.interface_audio_input or "").strip()
        if interface_audio_input:
            payload["interface_audio_input"] = interface_audio_input
        payload["asr_mode"] = config.asr_mode.value
    return payload


def _is_kame_metadata(metadata: Mapping[str, Any]) -> bool:
    return str(metadata.get("voice_architecture") or "") == "kame_frontend_oracle"


def _merge_metrics(*values: Any) -> dict[str, int]:
    merged: dict[str, int] = {}
    for value in values:
        if isinstance(value, Mapping):
            merged.update(_nonnegative_int_metrics(value))
    return merged


def _kame_route_metrics_payload(
    metadata: Mapping[str, Any],
    *,
    oracle_called: bool,
    extra_metrics: Optional[Mapping[str, int]] = None,
) -> dict:
    metrics = _kame_route_metrics(metadata, oracle_called=oracle_called, extra_metrics=extra_metrics)
    return {"metrics": metrics} if metrics else {}


def _kame_route_metrics(
    metadata: Mapping[str, Any],
    *,
    oracle_called: bool,
    extra_metrics: Optional[Mapping[str, int]] = None,
) -> dict[str, int]:
    route = str(metadata.get("kame_route") or "").strip()
    if not route:
        return {}
    oracle_called_int = 1 if oracle_called else 0
    existing_metrics = metadata.get("metrics") if isinstance(metadata, Mapping) else None
    metrics = _nonnegative_int_metrics(existing_metrics) if isinstance(existing_metrics, Mapping) else {}
    metrics.update(
        {
            "kame_oracle_called": oracle_called_int,
            "kame_oracle_bypassed": 0 if oracle_called else 1,
        }
    )
    if extra_metrics:
        metrics.update(_nonnegative_int_metrics(extra_metrics))
    return metrics


def _kame_metrics_policy_enabled(config: Optional[RealtimeVoiceSessionConfig]) -> bool:
    if config is None or config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
        return False
    if isinstance(config.metrics_policy, Mapping) and config.metrics_policy:
        return _metadata_bool(config.metrics_policy.get("enabled"), default=True)
    metadata = config.metadata if isinstance(config.metadata, Mapping) else {}
    metrics = metadata.get("metrics") if isinstance(metadata, Mapping) else {}
    if not isinstance(metrics, Mapping):
        return True
    return _metadata_bool(metrics.get("enabled"), default=True)


def _kame_metrics_policy_turn_spans_enabled(config: Optional[RealtimeVoiceSessionConfig]) -> bool:
    if not _kame_metrics_policy_enabled(config):
        return False
    if config is not None and isinstance(config.metrics_policy, Mapping) and config.metrics_policy:
        return _metadata_bool(config.metrics_policy.get("log_turn_spans"), default=True)
    metadata = config.metadata if config is not None and isinstance(config.metadata, Mapping) else {}
    metrics = metadata.get("metrics") if isinstance(metadata, Mapping) else {}
    if not isinstance(metrics, Mapping):
        return True
    return _metadata_bool(metrics.get("log_turn_spans"), default=True)


def _kame_metrics_policy_provider_spans_enabled(config: Optional[RealtimeVoiceSessionConfig]) -> bool:
    if not _kame_metrics_policy_enabled(config):
        return False
    if config is not None and isinstance(config.metrics_policy, Mapping) and config.metrics_policy:
        return _metadata_bool(config.metrics_policy.get("log_provider_spans"), default=True)
    metadata = config.metadata if config is not None and isinstance(config.metadata, Mapping) else {}
    metrics = metadata.get("metrics") if isinstance(metadata, Mapping) else {}
    if not isinstance(metrics, Mapping):
        return True
    return _metadata_bool(metrics.get("log_provider_spans"), default=True)


def _caption_alias_events_enabled(config: Optional[RealtimeVoiceSessionConfig]) -> bool:
    if config is not None and isinstance(config.output_events, Mapping) and config.output_events:
        if _metadata_bool(config.output_events.get("caption_aliases"), default=False):
            return True
    metadata = config.metadata if config is not None and isinstance(config.metadata, Mapping) else {}
    output_events = metadata.get("output_events") if isinstance(metadata, Mapping) else {}
    if isinstance(output_events, Mapping) and _metadata_bool(output_events.get("caption_aliases"), default=False):
        return True
    caption_events = metadata.get("caption_events") if isinstance(metadata, Mapping) else {}
    return isinstance(caption_events, Mapping) and _metadata_bool(caption_events.get("enabled"), default=False)


def _audio_alias_events_enabled(config: Optional[RealtimeVoiceSessionConfig]) -> bool:
    if config is not None and isinstance(config.output_events, Mapping) and config.output_events:
        return _metadata_bool(config.output_events.get("audio_aliases"), default=False)
    metadata = config.metadata if config is not None and isinstance(config.metadata, Mapping) else {}
    output_events = metadata.get("output_events") if isinstance(metadata, Mapping) else {}
    return isinstance(output_events, Mapping) and _metadata_bool(output_events.get("audio_aliases"), default=False)


def _nonnegative_int_metrics(metrics: Mapping[str, Any]) -> dict[str, int]:
    sanitized: dict[str, int] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            sanitized[str(key)] = parsed
    return sanitized


def _elapsed_perf_ms(start: float, end: float) -> int:
    return max(0, int(round((end - start) * 1000)))


def _metadata_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _payload_confirms_speech_for_barge_in(payload: Mapping[str, Any]) -> bool:
    for key in (
        "speech_confirmed",
        "barge_in_confirmed",
        "vad_speech",
        "speech_detected",
    ):
        if key in payload:
            return _metadata_bool(payload.get(key), default=False)
    return False


def _speech_energy_user_key(payload: Mapping[str, Any]) -> str:
    user_id = str(payload.get("user_id") or payload.get("speaker_id") or "").strip()
    return user_id or "default"


def _speech_energy_duration_ms(payload: Mapping[str, Any]) -> int:
    duration_ms = _payload_nonnegative_float(payload.get("duration_ms"))
    if duration_ms is not None:
        return int(round(duration_ms))
    duration_seconds = _payload_nonnegative_float(payload.get("duration_seconds"))
    if duration_seconds is not None:
        return int(round(duration_seconds * 1000))
    return 0


def _payload_nonnegative_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _barge_in_min_rms(config: Optional[RealtimeVoiceSessionConfig]) -> float:
    if config is not None and isinstance(config.barge_in_policy, Mapping) and config.barge_in_policy:
        value = _payload_nonnegative_float(config.barge_in_policy.get("min_rms"))
        if value is not None:
            return value
    metadata = config.metadata if config is not None and isinstance(config.metadata, Mapping) else {}
    barge_in = metadata.get("barge_in")
    if isinstance(barge_in, Mapping):
        value = _payload_nonnegative_float(barge_in.get("min_rms"))
        if value is not None:
            return value
    value = _payload_nonnegative_float(metadata.get("barge_in_min_rms"))
    return value if value is not None else 350.0


def _barge_in_min_speech_ms(config: Optional[RealtimeVoiceSessionConfig]) -> int:
    if config is not None and isinstance(config.barge_in_policy, Mapping) and config.barge_in_policy:
        value = _payload_nonnegative_float(config.barge_in_policy.get("min_speech_ms"))
        if value is not None:
            return int(round(value))
    metadata = config.metadata if config is not None and isinstance(config.metadata, Mapping) else {}
    barge_in = metadata.get("barge_in")
    if isinstance(barge_in, Mapping):
        value = _payload_nonnegative_float(barge_in.get("min_speech_ms"))
        if value is not None:
            return int(round(value))
    value = _payload_nonnegative_float(metadata.get("barge_in_min_speech_ms"))
    return int(round(value)) if value is not None else 120


def _payload_input_generation(payload: dict) -> Optional[int]:
    value = payload.get("input_generation")
    return _payload_int(value)


async def _stream_oracle_answer(
    oracle: object,
    transcript: str,
    metadata: dict,
    *,
    oracle_request: Optional[KameOracleRequest] = None,
    timeout_seconds: float = 60.0,
) -> AsyncIterator[Any]:
    request_stream = getattr(oracle, "stream_answer_for_request", None)
    if oracle_request is not None and callable(request_stream):
        async for delta in _with_next_timeout(
            request_stream(oracle_request),  # type: ignore[misc]
            timeout_seconds=timeout_seconds,
        ):
            yield delta
        return

    metadata_stream = getattr(oracle, "stream_answer_with_metadata", None)
    if callable(metadata_stream):
        async for delta in _with_next_timeout(
            metadata_stream(transcript, metadata),  # type: ignore[misc]
            timeout_seconds=timeout_seconds,
        ):
            yield delta
        return

    async for delta in _with_next_timeout(
        oracle.stream_answer(transcript),  # type: ignore[attr-defined]
        timeout_seconds=timeout_seconds,
    ):
        yield delta


async def _with_next_timeout(stream: AsyncIterator[Any], *, timeout_seconds: float) -> AsyncIterator[Any]:
    timeout = max(0.001, float(timeout_seconds or 60.0))
    iterator = stream.__aiter__()
    try:
        while True:
            try:
                yield await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
            except StopAsyncIteration:
                return
    finally:
        aclose = getattr(iterator, "aclose", None)
        if callable(aclose):
            await aclose()


_ORACLE_TOOL_CALL_EVENT_NAMES = frozenset({"oracle.tool_call", "tool_call", "tool.call"})
_ORACLE_TOOL_RESULT_EVENT_NAMES = frozenset({"oracle.tool_result", "tool_result", "tool.result"})
_ORACLE_TOOL_APPROVAL_TRUE_KEYS = frozenset(
    {
        "approval_required",
        "requires_approval",
        "needs_approval",
        "awaiting_approval",
        "waiting_for_approval",
        "requires_user_approval",
        "human_approval_required",
        "pending_approval",
    }
)
_ORACLE_TOOL_APPROVAL_STATUS_VALUES = frozenset(
    {
        "approval_required",
        "awaiting_approval",
        "waiting_for_approval",
        "pending_approval",
        "requires_approval",
    }
)


def _oracle_tool_event_type(item: Mapping[str, Any]) -> Optional[VoiceEventType]:
    raw_type = str(item.get("type") or item.get("event") or "").strip().lower()
    if raw_type in _ORACLE_TOOL_CALL_EVENT_NAMES:
        return VoiceEventType.ORACLE_TOOL_CALL
    if raw_type in _ORACLE_TOOL_RESULT_EVENT_NAMES:
        return VoiceEventType.ORACLE_TOOL_RESULT
    return None


def _oracle_tool_event_waits_for_approval(item: Mapping[str, Any]) -> bool:
    for key in _ORACLE_TOOL_APPROVAL_TRUE_KEYS:
        if _metadata_bool(item.get(key), default=False):
            return True
    status = str(item.get("approval_status") or item.get("status") or item.get("state") or "").strip().lower()
    if status in _ORACLE_TOOL_APPROVAL_STATUS_VALUES:
        return True
    approval = item.get("approval")
    if isinstance(approval, Mapping):
        for key in _ORACLE_TOOL_APPROVAL_TRUE_KEYS:
            if _metadata_bool(approval.get(key), default=False):
                return True
        nested_status = str(
            approval.get("status") or approval.get("state") or approval.get("approval_status") or ""
        ).strip().lower()
        if nested_status in _ORACLE_TOOL_APPROVAL_STATUS_VALUES:
            return True
    return False


def _oracle_tool_approval_reason(item: Mapping[str, Any]) -> str:
    for key in ("approval_reason", "reason", "message", "summary"):
        text = str(item.get(key) or "").strip()
        if text:
            return " ".join(text.split())[:240]
    approval = item.get("approval")
    if isinstance(approval, Mapping):
        for key in ("approval_reason", "reason", "message", "summary"):
            text = str(approval.get(key) or "").strip()
            if text:
                return " ".join(text.split())[:240]
    tool_name = str(item.get("tool_name") or item.get("name") or "").strip()
    if tool_name:
        return f"{tool_name} is waiting for approval"
    return "waiting for approval"


def _oracle_stream_text_delta(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    if isinstance(item, Mapping):
        for key in ("delta", "text", "content"):
            value = item.get(key)
            if value is not None:
                return str(value)
        return ""
    return str(item)


def _oracle_tool_event_payload(
    item: Mapping[str, Any],
    *,
    redact_sensitive: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in item.items():
        normalized_key = str(key)
        if normalized_key in {"type", "event"}:
            continue
        if redact_sensitive and normalized_key in {"arguments", "args", "input", "parameters", "function"}:
            continue
        payload[normalized_key] = _realtime_json_safe(value)

    function = item.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments")
        if name is not None and not payload.get("tool_name"):
            payload["tool_name"] = str(name)
        if arguments is not None and "arguments" not in payload and not redact_sensitive:
            payload["arguments"] = _realtime_json_safe(arguments)

    if item.get("name") is not None and not payload.get("tool_name"):
        payload["tool_name"] = str(item.get("name"))
    for call_id_key in ("tool_call_id", "call_id", "id"):
        if item.get(call_id_key) is not None and not payload.get("tool_call_id"):
            payload["tool_call_id"] = str(item.get(call_id_key))
    return payload


def _realtime_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): _realtime_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_realtime_json_safe(item) for item in value]
    return str(value)


def _max_spoken_sentences(
    config: Optional[RealtimeVoiceSessionConfig],
    *,
    oracle_request: Optional[KameOracleRequest] = None,
) -> int:
    if oracle_request is not None:
        value = oracle_request.max_spoken_sentences
    elif config is not None and config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
        value = config.max_spoken_sentences
    else:
        return 0
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _voice_response_policy(
    config: Optional[RealtimeVoiceSessionConfig],
    *,
    oracle_request: Optional[KameOracleRequest] = None,
) -> str:
    value: Any = None
    if oracle_request is not None and isinstance(oracle_request.requested_response_style, Mapping):
        value = (
            oracle_request.requested_response_style.get("policy")
            or oracle_request.requested_response_style.get("voice_response_policy")
        )
    if value is None and config is not None:
        value = getattr(config, "voice_response_policy", None)
    text = str(value or "").strip().lower().replace("-", "_")
    return text if text in {"sentence_cap", "brief_summary", "full"} else "sentence_cap"


def _effective_max_spoken_sentences(
    config: Optional[RealtimeVoiceSessionConfig],
    *,
    oracle_request: Optional[KameOracleRequest] = None,
) -> int:
    policy = _voice_response_policy(config, oracle_request=oracle_request)
    if policy == "full":
        return 0
    max_sentences = _max_spoken_sentences(config, oracle_request=oracle_request)
    if policy == "brief_summary":
        return 1 if max_sentences <= 0 else min(max_sentences, 1)
    return max_sentences


def _voice_response_policy_payload(*, policy: str, max_sentences: int, truncated: bool) -> dict[str, Any]:
    if max_sentences <= 0 and (policy or "sentence_cap") == "sentence_cap":
        return {}
    payload: dict[str, Any] = {"voice_response_policy": policy or "sentence_cap"}
    if max_sentences <= 0:
        payload["voice_response_truncated"] = False
        return payload
    return {
        **payload,
        "max_spoken_sentences": max_sentences,
        "voice_response_truncated": bool(truncated),
    }


def _limit_spoken_text(
    text: str,
    *,
    max_sentences: int,
    already_spoken: str = "",
) -> tuple[str, bool]:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned or max_sentences <= 0:
        return cleaned, False

    remaining = max_sentences - _spoken_sentence_count(already_spoken)
    if remaining <= 0:
        return "", True

    sentence_count = 0
    truncate_at: Optional[int] = None
    for index, character in enumerate(cleaned):
        if character not in _SENTENCE_BOUNDARY_CHARS:
            continue
        sentence_count += 1
        if sentence_count >= remaining:
            truncate_at = index + 1
            break

    if truncate_at is None:
        return cleaned, False

    limited = cleaned[:truncate_at].strip()
    truncated = bool(cleaned[truncate_at:].strip())
    return limited, truncated


def _spoken_sentence_count(text: str) -> int:
    return sum(1 for character in text or "" if character in _SENTENCE_BOUNDARY_CHARS)


def _join_spoken_text(left: str, right: str) -> str:
    left = (left or "").strip()
    right = (right or "").strip()
    if not left:
        return right
    if not right:
        return left
    separator = " " if _needs_spoken_separator(left[-1], right[0]) else ""
    return f"{left}{separator}{right}"


def _needs_spoken_separator(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left.isspace() or right.isspace():
        return False
    if ord(left) > 127 or ord(right) > 127:
        return False
    return True


def _oracle_timeout_seconds(config: Optional[RealtimeVoiceSessionConfig]) -> float:
    if config is None:
        return 60.0
    timeout = config.oracle_timeout_seconds
    if isinstance(timeout, bool):
        return 60.0
    try:
        parsed = float(timeout)
    except (TypeError, ValueError):
        return 60.0
    if parsed <= 0:
        return 60.0
    return parsed


def _oracle_timeout_status_text(config: Optional[RealtimeVoiceSessionConfig]) -> str:
    if config is not None and isinstance(config.metadata, Mapping):
        text = str(config.metadata.get("oracle_timeout_status_text") or "").strip()
        if text:
            return text[:160]
    return "Hermes is taking too long to answer. Please try that again in a moment."


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
    raw = (buffer or "").replace("\r\n", "\n").replace("\r", "\n")
    paragraph_boundary = re.search(r"\n\s*\n+", raw)
    if paragraph_boundary is not None:
        paragraph = raw[: paragraph_boundary.start()].strip()
        remaining = raw[paragraph_boundary.end() :].strip()
        if paragraph:
            return " ".join(paragraph.split()), remaining

    normalized = " ".join((buffer or "").split())
    if not normalized:
        return None, ""

    sentence_at = _find_delimiter(normalized, _SENTENCE_BOUNDARY_CHARS, start=6, end=180)
    if sentence_at >= 0:
        return normalized[: sentence_at + 1].strip(), normalized[sentence_at + 1 :].strip()

    has_whitespace = any(character.isspace() for character in normalized)
    phrase_min = 28 if has_whitespace else 12
    phrase_trigger = 56 if has_whitespace else 24
    phrase_end = 104 if has_whitespace else 72

    if len(normalized) >= phrase_trigger:
        split_at = _find_delimiter(normalized, _PHRASE_BOUNDARY_CHARS, start=phrase_min, end=phrase_end)
        if split_at >= phrase_min:
            return normalized[: split_at + 1].strip(), normalized[split_at + 1 :].strip()

        split_at = normalized.rfind(" ", 56, 104) if has_whitespace else -1
        if split_at >= 56:
            return normalized[:split_at].strip(), normalized[split_at:].strip()

    if len(normalized) > 140:
        split_at = _find_delimiter(normalized, _PHRASE_BOUNDARY_CHARS, start=0, end=140)
        split_at = max(split_at, normalized.rfind(" ", 0, 140))
        if split_at >= 48:
            suffix_start = split_at + 1 if normalized[split_at] in _PHRASE_BOUNDARY_CHARS else split_at
            return normalized[:suffix_start].strip(), normalized[suffix_start:].strip()

    return None, buffer if buffer.strip() else normalized


def _find_delimiter(text: str, delimiters: frozenset[str], *, start: int, end: int) -> int:
    upper = min(len(text), end)
    for index in range(upper - 1, max(-1, start - 1), -1):
        if text[index] in delimiters:
            return index
    return -1
