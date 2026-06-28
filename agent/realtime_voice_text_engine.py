"""Text-oracle + TTS realtime voice engine."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import time
from dataclasses import replace
from typing import Any, AsyncIterator, List, Mapping, Optional

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
from agent.realtime_voice_kame import KameOracleRequest, KameRoute, kame_local_reply_denies_voice_capability
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
        self._frontend_output_active = False
        self._assistant_metadata_by_generation: dict[int, dict] = {}
        self._cancellation_token_by_generation: dict[int, str] = {}
        self._interface_decision_at_by_generation: dict[int, float] = {}
        self._oracle_first_token_at_by_generation: dict[int, float] = {}
        self._first_audio_metric_generations: set[int] = set()

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
        backend_active = self._active_task is not None and not self._active_task.done()
        if not backend_active and not self._frontend_output_active:
            return
        await self._interrupt_active_turn(event, reason="user_speech")

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
        if hasattr(oracle, "interrupt"):
            oracle.interrupt("Realtime voice barge-in")  # type: ignore[attr-defined]
        self._clear_inbound_audio()
        cancelled_kame_oracle = bool(
            cancellation_token
            and cancelled_metadata.get("kame_route") in {KameRoute.DEFER.value, KameRoute.ORACLE_DIRECT.value}
        )
        if cancelled_kame_oracle:
            await self._emit(
                VoiceEventType.INTERFACE_ORACLE_CANCEL,
                {
                    "reason": payload["reason"],
                    "playback_generation": self._playback_generation,
                    "cancelled_playback_generation": cancelled_generation,
                    "cancellation_token": cancellation_token,
                    **_kame_interface_payload_from_metadata(cancelled_metadata),
                },
            )
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
            await self._emit(VoiceEventType.INTERFACE_INTENT_FINAL, interface_payload)
        local_reply = _kame_local_reply(oracle_request)
        if local_reply:
            if oracle_request is not None:
                await self._emit(
                    VoiceEventType.INTERFACE_REPLY_LOCAL,
                    {
                        **_kame_interface_payload_with_metrics(oracle_request, generation, assistant_metadata),
                        "text": local_reply,
                    },
                )
            self._active_task = asyncio.create_task(
                self._speak_kame_local_reply(local_reply, generation, assistant_metadata)
            )
            return
        if oracle_request is not None:
            interface_payload = _kame_interface_payload_with_metrics(oracle_request, generation, assistant_metadata)
            if oracle_request.route == KameRoute.DEFER:
                await self._emit(VoiceEventType.INTERFACE_REPLY_DEFER, interface_payload)
            await self._emit(VoiceEventType.INTERFACE_ORACLE_REQUEST, interface_payload)
        self._active_task = asyncio.create_task(
            self._answer_and_speak(
                transcript,
                generation,
                assistant_metadata,
                oracle_request=oracle_request,
            )
        )

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
        source = str(config_metadata.get("transport") or "voice").strip() or "voice"
        user_id = str(config_metadata.get("user_id") or "").strip() or None
        payload: dict[str, Any] = {}
        payload.update(metadata)
        if oracle_payload is not None:
            payload.update(dict(oracle_payload))
        if cancellation_token:
            payload.setdefault("cancellation_token", cancellation_token)
        request = KameOracleRequest.from_turn(
            session_id=config.session_id,
            turn_id=f"{config.session_id}:{playback_generation}",
            source=source,
            user_id=user_id,
            payload=payload,
            fallback_text=transcript,
            default_max_spoken_sentences=_max_spoken_sentences(config),
            routing_policy=_kame_routing_policy(config),
        )
        if request.route == KameRoute.DEFER and not request.interface_already_said:
            acknowledgement = _turn_acknowledgement_text(config)
            if acknowledgement:
                request = replace(request, interface_already_said=acknowledgement)
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
                max_sentences=_max_spoken_sentences(self.config),
            )
            if not planned_reply:
                return
            metadata = {
                **metadata,
                **_voice_response_policy_payload(
                    max_sentences=_max_spoken_sentences(self.config),
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
        max_spoken_sentences = _max_spoken_sentences(self.config, oracle_request=oracle_request)
        spoken_answer = ""
        spoken_truncated = False
        turn_started_at = time.perf_counter()
        oracle_accepted_at: Optional[float] = None
        oracle_first_token_at: Optional[float] = None
        first_spoken_text_at: Optional[float] = None
        kame_timing_metrics: dict[str, int] = {}
        voice_denial_corrected = False

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

        try:
            acknowledgement = _turn_acknowledgement_text(self.config)
            if acknowledgement:
                planned_acknowledgement = self._planner.clean(acknowledgement)
                if planned_acknowledgement:
                    await self._emit(
                        VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                        {
                            "text": planned_acknowledgement,
                            "playback_generation": playback_generation,
                            "acknowledgement": True,
                            **assistant_metadata,
                            **_kame_route_metrics_payload(assistant_metadata, oracle_called=True),
                        },
                    )
                    queue_speak(planned_acknowledgement)

            oracle = self._oracle or NullRealtimeOracle()
            answer = ""
            buffer = ""
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
                kame_timing_metrics["kame_interface_decision_to_oracle_accepted_ms"] = _elapsed_perf_ms(
                    turn_started_at,
                    oracle_accepted_at,
                )
                sync_kame_timing_metrics()
            async for delta in _stream_oracle_answer(
                oracle,
                transcript,
                assistant_metadata,
                oracle_request=oracle_request,
                timeout_seconds=_oracle_timeout_seconds(self.config),
            ):
                if playback_generation != self._playback_generation:
                    return
                now = time.perf_counter()
                if oracle_request is not None and oracle_first_token_at is None:
                    oracle_first_token_at = now
                    self._oracle_first_token_at_by_generation[playback_generation] = now
                    if oracle_accepted_at is not None:
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
                chunk, buffer = _take_speakable_chunk(buffer)
                if chunk:
                    planned_chunk = self._planner.clean(chunk)
                    if planned_chunk:
                        planned_chunk, denial_corrected = correct_voice_denial(planned_chunk)
                        if denial_corrected:
                            spoken_truncated = True
                        if not planned_chunk:
                            continue
                        planned_chunk, chunk_truncated = _limit_spoken_text(
                            planned_chunk,
                            max_sentences=max_spoken_sentences,
                            already_spoken=spoken_answer,
                        )
                        spoken_truncated = spoken_truncated or chunk_truncated
                        if not planned_chunk:
                            if _spoken_sentence_count(spoken_answer) >= max_spoken_sentences > 0:
                                break
                            continue
                        spoken_answer = _join_spoken_text(spoken_answer, planned_chunk)
                        if oracle_request is not None and first_spoken_text_at is None:
                            first_spoken_text_at = time.perf_counter()
                            if oracle_first_token_at is not None:
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
                            break

            if buffer.strip():
                planned_chunk = self._planner.clean(buffer)
                if planned_chunk:
                    planned_chunk, denial_corrected = correct_voice_denial(planned_chunk)
                    if denial_corrected:
                        spoken_truncated = True
                if planned_chunk:
                    planned_chunk, chunk_truncated = _limit_spoken_text(
                        planned_chunk,
                        max_sentences=max_spoken_sentences,
                        already_spoken=spoken_answer,
                    )
                    spoken_truncated = spoken_truncated or chunk_truncated
                if planned_chunk:
                    spoken_answer = _join_spoken_text(spoken_answer, planned_chunk)
                    if oracle_request is not None and first_spoken_text_at is None:
                        first_spoken_text_at = time.perf_counter()
                        if oracle_first_token_at is not None:
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

            if oracle_request is not None and answer:
                if oracle_accepted_at is not None:
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
        await self._emit(
            VoiceEventType.INTERFACE_COMMIT,
            {
                **_kame_interface_payload_from_metadata(metadata),
                "playback_generation": playback_generation,
                "local_reply": local_reply,
                "text": text,
            },
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
                metrics["tts_synthesis_ms"] = tts_synthesis_ms
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
        decision_at = self._interface_decision_at_by_generation.get(playback_generation)
        if decision_at is None:
            return {}
        first_audio_at = time.perf_counter()
        elapsed = _elapsed_perf_ms(decision_at, first_audio_at)
        self._first_audio_metric_generations.add(playback_generation)
        metrics = {"kame_interface_decision_to_first_audio_ms": elapsed}
        oracle_first_token_at = self._oracle_first_token_at_by_generation.get(playback_generation)
        if oracle_first_token_at is not None:
            metrics["kame_oracle_first_token_to_first_tts_audio_ms"] = _elapsed_perf_ms(
                oracle_first_token_at,
                first_audio_at,
            )
        route = str(metadata.get("kame_route") or "")
        if route in {KameRoute.LOCAL.value, KameRoute.REJECT_OR_CLARIFY.value}:
            metrics["kame_interface_decision_to_local_first_audio_ms"] = elapsed
        return metrics

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
        if not _kame_metrics_policy_enabled(self.config):
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


def _turn_acknowledgement_text(config: Optional[RealtimeVoiceSessionConfig]) -> str:
    if config is None or not isinstance(config.metadata, Mapping):
        return ""
    acknowledgement = config.metadata.get("turn_acknowledgement")
    if not isinstance(acknowledgement, Mapping):
        return ""
    if not _metadata_bool(acknowledgement.get("enabled"), default=False):
        return ""
    text = str(acknowledgement.get("text") or "One moment.").strip()
    if not text:
        return ""
    return text[:120]


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


def _kame_interface_payload(request: KameOracleRequest, playback_generation: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "turn_id": request.turn_id,
        "route": request.route.value,
        "intent": request.intent,
        "intent_source": request.intent_source,
        "text": request.oracle_text,
        "playback_generation": playback_generation,
        "source": request.source,
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
    if request.cancellation_token:
        payload["cancellation_token"] = request.cancellation_token
    if request.reflex_validation_error:
        payload["reflex_validation_error"] = request.reflex_validation_error
    return payload


def _kame_interface_payload_with_metrics(
    request: KameOracleRequest,
    playback_generation: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _kame_interface_payload(request, playback_generation)
    metrics = metadata.get("metrics") if isinstance(metadata, Mapping) else None
    if isinstance(metrics, Mapping):
        sanitized = _nonnegative_int_metrics(metrics)
        if sanitized:
            payload["metrics"] = sanitized
    return payload


def _kame_interface_payload_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "turn_id": str(metadata.get("kame_turn_id") or ""),
        "route": str(metadata.get("kame_route") or ""),
        "intent": str(metadata.get("kame_intent") or ""),
        "intent_source": str(metadata.get("kame_intent_source") or ""),
        "text": str(metadata.get("kame_transcript") or metadata.get("kame_intent") or ""),
        "source": str(metadata.get("kame_source") or ""),
        "transcript_source": str(metadata.get("kame_transcript_source") or ""),
    }
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
    if metadata.get("kame_cancellation_token"):
        payload["cancellation_token"] = str(metadata.get("kame_cancellation_token"))
    if metadata.get("kame_reflex_validation_error"):
        payload["reflex_validation_error"] = str(metadata.get("kame_reflex_validation_error"))
    if isinstance(metadata.get("kame_requested_response_style"), Mapping):
        payload["requested_response_style"] = dict(metadata.get("kame_requested_response_style") or {})
    return {key: value for key, value in payload.items() if value != ""}


def _kame_routing_policy(config: Optional[RealtimeVoiceSessionConfig]) -> Mapping[str, Any]:
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
    metadata = config.metadata if isinstance(config.metadata, Mapping) else {}
    metrics = metadata.get("metrics") if isinstance(metadata, Mapping) else {}
    if not isinstance(metrics, Mapping):
        return True
    return _metadata_bool(metrics.get("enabled"), default=True)


def _caption_alias_events_enabled(config: Optional[RealtimeVoiceSessionConfig]) -> bool:
    metadata = config.metadata if config is not None and isinstance(config.metadata, Mapping) else {}
    output_events = metadata.get("output_events") if isinstance(metadata, Mapping) else {}
    if isinstance(output_events, Mapping) and _metadata_bool(output_events.get("caption_aliases"), default=False):
        return True
    caption_events = metadata.get("caption_events") if isinstance(metadata, Mapping) else {}
    return isinstance(caption_events, Mapping) and _metadata_bool(caption_events.get("enabled"), default=False)


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
) -> AsyncIterator[str]:
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


async def _with_next_timeout(stream: AsyncIterator[str], *, timeout_seconds: float) -> AsyncIterator[str]:
    timeout = max(0.001, float(timeout_seconds or 60.0))
    iterator = stream.__aiter__()
    while True:
        try:
            yield await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return


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


def _voice_response_policy_payload(*, max_sentences: int, truncated: bool) -> dict[str, Any]:
    if max_sentences <= 0:
        return {}
    return {
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

    return None, buffer if buffer.strip() else normalized


def _find_delimiter(text: str, delimiters: frozenset[str], *, start: int, end: int) -> int:
    upper = min(len(text), end)
    for index in range(upper - 1, max(-1, start - 1), -1):
        if text[index] in delimiters:
            return index
    return -1
