"""Reference realtime voice sidecar for Hermes.

This is the server-side counterpart to ``RealtimeVoiceSidecarClient``. It is
designed to run locally on ordinary developer machines, while optionally using
a LAN vLLM/Gemma audio endpoint when one is configured.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, Callable, Mapping, Optional

from starlette.requests import Request
from starlette.websockets import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

from agent.realtime_voice import (
    AudioChunk,
    RealtimeVoiceEngineKind,
    RealtimeVoiceSessionConfig,
    VoiceAudioCodec,
    VoiceEvent,
    VoiceEventType,
    binary_audio_frame_from_event,
    create_realtime_voice_event_queue,
    event_from_binary_audio_frame,
    normalize_realtime_voice_interface_audio_input,
    put_realtime_voice_event,
    realtime_voice_session_contract_payload,
    transcript_metadata_from_payload,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_kame import (
    KameReflexDecision,
    KameRoute,
    apply_kame_routing_policy,
    kame_reflex_decision_json_schema,
    kame_reflex_instruction_text,
)
from agent.realtime_voice_sidecar import RealtimeVoiceSidecarClient


TranscribeFn = Callable[[str], Mapping[str, Any]]
SynthesizeFn = Callable[..., Any]
REFERENCE_SIDECAR_CLOSE_DRAIN_TIMEOUT_SECONDS = 1.0
REFERENCE_SIDECAR_PROVIDER_CLOSE_TIMEOUT_SECONDS = 1.0


class KameAudioSegmentTooLongError(RuntimeError):
    """Raised when a buffered audio segment exceeds the interface model limit."""


ORACLE_JOB_EVENT_TYPES = frozenset(
    {
        VoiceEventType.ORACLE_JOB_ACCEPTED,
        VoiceEventType.ORACLE_JOB_QUEUED,
        VoiceEventType.ORACLE_JOB_STARTED,
        VoiceEventType.ORACLE_JOB_INTERPRETER_EVIDENCE_ATTACHED,
        VoiceEventType.ORACLE_JOB_INTERPRETER_EVIDENCE_LATE,
        VoiceEventType.ORACLE_JOB_PROGRESS,
        VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL,
        VoiceEventType.ORACLE_JOB_COMPLETED,
        VoiceEventType.ORACLE_JOB_FAILED,
        VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED,
        VoiceEventType.ORACLE_JOB_CANCELLED,
        VoiceEventType.ORACLE_JOB_RESULT_SUPPRESSED,
    }
)

PROVIDER_FORWARDED_EVENT_TYPES = frozenset(
    {
        VoiceEventType.TRANSCRIPT_PARTIAL,
        VoiceEventType.TRANSCRIPT_FINAL,
        VoiceEventType.AUDIO_OUTPUT_CHUNK,
        VoiceEventType.ASSISTANT_AUDIO_END,
        VoiceEventType.PLAYBACK_STARTED,
        VoiceEventType.PLAYBACK_STOPPED,
        VoiceEventType.FRONTEND_STATE,
        VoiceEventType.INTERFACE_INTENT_PARTIAL,
        VoiceEventType.INTERFACE_INTENT_FINAL,
        VoiceEventType.INTERFACE_REPLY_LOCAL,
        VoiceEventType.INTERFACE_REPLY_DEFER,
        VoiceEventType.INTERFACE_ORACLE_REQUEST,
        VoiceEventType.INTERFACE_ORACLE_CANCEL,
        VoiceEventType.INTERFACE_ORACLE_UPDATE,
        VoiceEventType.INTERFACE_COMMIT,
        VoiceEventType.ORACLE_ACCEPTED,
        VoiceEventType.ORACLE_HINT,
        VoiceEventType.ORACLE_TOOL_CALL,
        VoiceEventType.ORACLE_TOOL_RESULT,
        VoiceEventType.ORACLE_RESPONSE_PARTIAL,
        VoiceEventType.ORACLE_RESPONSE_FINAL,
        VoiceEventType.ORACLE_ERROR,
        VoiceEventType.ASSISTANT_CAPTION_PARTIAL,
        VoiceEventType.ASSISTANT_CAPTION_FINAL,
        VoiceEventType.ASSISTANT_AUDIO_CHUNK,
        VoiceEventType.ASSISTANT_TEXT_PARTIAL,
        VoiceEventType.ASSISTANT_COMMIT,
        VoiceEventType.BARGE_IN,
        VoiceEventType.SESSION_METRICS,
        VoiceEventType.TOOL_PENDING,
        VoiceEventType.TOOL_RESULT,
    }
) | ORACLE_JOB_EVENT_TYPES

KAME_FEEDBACK_EVENT_TYPES = frozenset(
    {
        VoiceEventType.INTERFACE_INTENT_FINAL,
        VoiceEventType.INTERFACE_REPLY_LOCAL,
        VoiceEventType.INTERFACE_REPLY_DEFER,
        VoiceEventType.INTERFACE_ORACLE_REQUEST,
        VoiceEventType.INTERFACE_ORACLE_CANCEL,
        VoiceEventType.INTERFACE_ORACLE_UPDATE,
        VoiceEventType.INTERFACE_COMMIT,
        VoiceEventType.ORACLE_ACCEPTED,
        VoiceEventType.ORACLE_HINT,
        VoiceEventType.ORACLE_TOOL_CALL,
        VoiceEventType.ORACLE_TOOL_RESULT,
        VoiceEventType.ORACLE_RESPONSE_PARTIAL,
        VoiceEventType.ORACLE_RESPONSE_FINAL,
        VoiceEventType.ORACLE_ERROR,
        VoiceEventType.SESSION_METRICS,
    }
) | ORACLE_JOB_EVENT_TYPES
KAME_LIVE_FRONTEND_ORACLE_CONTEXT_EVENT_TYPES = frozenset(
    {
        VoiceEventType.ORACLE_ACCEPTED,
        VoiceEventType.ORACLE_HINT,
        VoiceEventType.ORACLE_TOOL_CALL,
        VoiceEventType.ORACLE_TOOL_RESULT,
        VoiceEventType.ORACLE_RESPONSE_PARTIAL,
        VoiceEventType.ORACLE_RESPONSE_FINAL,
        VoiceEventType.ORACLE_ERROR,
    }
) | ORACLE_JOB_EVENT_TYPES
KAME_EXTERNAL_CONTROL_EVENT_TYPES = frozenset(
    {
        VoiceEventType.INTERFACE_ORACLE_CANCEL,
        VoiceEventType.INTERFACE_ORACLE_UPDATE,
    }
)


@dataclass(frozen=True)
class ReferenceSidecarRuntimeConfig:
    """Runtime knobs for the reference sidecar process."""

    interface_provider: Optional[str] = None
    vllm_base_url: Optional[str] = None
    vllm_model: Optional[str] = None
    vllm_token: Optional[str] = None
    vllm_timeout_seconds: float = 60.0
    streaming_stt_provider: Optional[str] = None
    streaming_stt_base_url: Optional[str] = None
    streaming_stt_model: Optional[str] = None
    streaming_stt_token: Optional[str] = None
    streaming_stt_timeout_seconds: float = 10.0
    streaming_bridge_health_timeout_seconds: float = 0.2
    streaming_tts_provider: Optional[str] = None
    streaming_tts_base_url: Optional[str] = None
    streaming_tts_model: Optional[str] = None
    streaming_tts_token: Optional[str] = None
    streaming_tts_timeout_seconds: float = 10.0
    openai_realtime_api_key: Optional[str] = None
    openai_realtime_base_url: str = "wss://api.openai.com/v1/realtime"
    openai_realtime_model: str = "gpt-realtime-2"
    openai_realtime_voice: str = "marin"
    openai_realtime_transcription_model: str = "gpt-realtime-whisper"
    openai_realtime_safety_identifier: Optional[str] = None
    gemini_live_api_key: Optional[str] = None
    gemini_live_base_url: str = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    )
    gemini_live_model: str = "gemini-3.1-flash-live-preview"
    gemini_live_voice: str = "Puck"
    gemini_live_google_search: bool = False
    gemini_live_oracle_tool: bool = True
    local_stt_enabled: bool = True
    local_tts_enabled: bool = True
    auth_token: Optional[str] = None
    input_languages: tuple[str, ...] = ()
    output_languages: tuple[str, ...] = ()
    scripts: tuple[str, ...] = ()


def _runtime_with_session_config(
    runtime: ReferenceSidecarRuntimeConfig,
    config: RealtimeVoiceSessionConfig,
) -> ReferenceSidecarRuntimeConfig:
    """Apply per-session endpoint/model choices to the reference sidecar runtime."""

    return replace(
        runtime,
        interface_provider=config.frontend_provider or runtime.interface_provider,
        vllm_base_url=config.interface_base_url or runtime.vllm_base_url,
        vllm_model=config.frontend_model or runtime.vllm_model,
        streaming_stt_provider=config.asr_provider or runtime.streaming_stt_provider,
        streaming_stt_base_url=config.asr_base_url or runtime.streaming_stt_base_url,
        streaming_stt_model=config.asr_model or runtime.streaming_stt_model,
        streaming_tts_provider=config.tts_provider or runtime.streaming_tts_provider,
        streaming_tts_base_url=config.tts_base_url or runtime.streaming_tts_base_url,
        streaming_tts_model=config.tts_model or runtime.streaming_tts_model,
    )


def reference_sidecar_health_payload(
    runtime: ReferenceSidecarRuntimeConfig,
    *,
    vllm_health: Optional[Mapping[str, Any]] = None,
    vllm_health_checked: bool = False,
    streaming_stt_health: Optional[Mapping[str, Any]] = None,
    streaming_tts_health: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    vllm_configured = bool(runtime.vllm_base_url and runtime.vllm_model)
    vllm_enabled = vllm_configured and (
        not vllm_health_checked or _health_supports_vllm_model(vllm_health, runtime.vllm_model)
    )
    streaming_stt_configured = bool(runtime.streaming_stt_base_url)
    streaming_stt_ready = _health_supports_streaming_stt(streaming_stt_health)
    streaming_tts_configured = bool(runtime.streaming_tts_base_url)
    streaming_tts_ready = _health_supports_tts(streaming_tts_health)
    interface_provider_label = _provider_label(runtime.interface_provider, default="vllm")
    streaming_stt_provider_label = _provider_label(runtime.streaming_stt_provider, default="streaming_stt")
    streaming_tts_provider_label = _provider_label(runtime.streaming_tts_provider, default="streaming_tts")
    openai_realtime_configured = bool(runtime.openai_realtime_api_key)
    gemini_live_configured = bool(runtime.gemini_live_api_key)
    input_languages = _sanitize_metadata_list(runtime.input_languages)
    configured_output_languages = _sanitize_metadata_list(runtime.output_languages)
    bridge_output_languages = _streaming_tts_health_output_languages(streaming_tts_health)
    tts_model_languages = _streaming_tts_health_model_languages(streaming_tts_health)
    output_languages = (
        bridge_output_languages
        if streaming_tts_configured
        else configured_output_languages
    )
    scripts = _sanitize_metadata_list(runtime.scripts)
    frontend_languages = _dedupe_metadata([*input_languages, *output_languages])
    frontend_provider = (
        "openai_realtime"
        if openai_realtime_configured
        else "gemini_live"
        if gemini_live_configured
        else interface_provider_label
        if vllm_enabled
        else streaming_stt_provider_label
        if streaming_stt_ready
        else "local"
    )
    frontend_model = (
        runtime.openai_realtime_model
        if openai_realtime_configured
        else runtime.gemini_live_model
        if gemini_live_configured
        else (runtime.vllm_model or "")
        if vllm_enabled
        else (runtime.streaming_stt_model or "")
    )

    payload = {
        "ok": True,
        "kind": "reference",
        "frontend": {
            "provider": frontend_provider,
            "model": frontend_model,
        },
        "capabilities": {
            "utterance_stt": streaming_stt_ready or vllm_enabled or runtime.local_stt_enabled,
            "streaming_stt": streaming_stt_ready or openai_realtime_configured or gemini_live_configured,
            "tts": streaming_tts_ready or runtime.local_tts_enabled or openai_realtime_configured or gemini_live_configured,
            "native_s2s": openai_realtime_configured or gemini_live_configured,
            "vllm_audio_frontend": vllm_enabled,
        },
        "local": {
            "stt": runtime.local_stt_enabled,
            "tts": runtime.local_tts_enabled,
        },
    }
    if streaming_stt_configured:
        payload["capabilities"]["streaming_stt_bridge"] = True
        payload["frontend"]["streaming_stt_bridge"] = {
            "configured": True,
            "healthy": streaming_stt_ready,
        }
        if streaming_stt_provider_label != "streaming_stt":
            payload["frontend"]["streaming_stt_bridge"]["provider"] = streaming_stt_provider_label
            payload["frontend"]["streaming_stt_bridge"]["implementation_provider"] = "streaming_stt"
    if streaming_tts_configured:
        payload["capabilities"]["streaming_tts_bridge"] = True
        payload["frontend"]["streaming_tts_bridge"] = {
            "configured": True,
            "healthy": streaming_tts_ready,
            "model": runtime.streaming_tts_model or "",
        }
        if streaming_tts_provider_label != "streaming_tts":
            payload["frontend"]["streaming_tts_bridge"]["provider"] = streaming_tts_provider_label
            payload["frontend"]["streaming_tts_bridge"]["implementation_provider"] = "streaming_tts"
    if vllm_configured:
        payload["capabilities"]["vllm_audio_frontend_configured"] = True
        payload["frontend"]["vllm_audio_frontend"] = {
            "configured": True,
            "healthy": vllm_enabled,
            "model": runtime.vllm_model or "",
            "token_configured": bool(runtime.vllm_token),
        }
        if interface_provider_label != "vllm":
            payload["frontend"]["vllm_audio_frontend"]["provider"] = interface_provider_label
            payload["frontend"]["vllm_audio_frontend"]["implementation_provider"] = "vllm"
    if tts_model_languages:
        payload["frontend"]["tts_model_languages"] = tts_model_languages
    if openai_realtime_configured:
        payload["capabilities"]["openai_realtime"] = True
        payload["capabilities"]["response_cancel"] = True
        payload["capabilities"]["server_vad"] = False
        payload["frontend"]["openai_realtime"] = {
            "configured": True,
            "model": runtime.openai_realtime_model,
            "voice": runtime.openai_realtime_voice,
            "transcription_model": runtime.openai_realtime_transcription_model,
        }
    if gemini_live_configured:
        payload["capabilities"]["gemini_live"] = True
        payload["capabilities"]["response_cancel"] = True
        payload["capabilities"]["server_vad"] = True
        payload["capabilities"]["tool_calls"] = runtime.gemini_live_oracle_tool
        payload["capabilities"]["google_search"] = runtime.gemini_live_google_search
        payload["frontend"]["gemini_live"] = {
            "configured": True,
            "model": runtime.gemini_live_model,
            "voice": runtime.gemini_live_voice,
            "oracle_tool": runtime.gemini_live_oracle_tool,
            "google_search": runtime.gemini_live_google_search,
        }
    if frontend_languages:
        payload["frontend"]["languages"] = frontend_languages
    if scripts:
        payload["frontend"]["scripts"] = scripts
        payload["capabilities"]["scripts"] = scripts
    if input_languages:
        payload["capabilities"]["input_languages"] = input_languages
    if output_languages:
        payload["capabilities"]["output_languages"] = output_languages
    return payload


class ReferenceRealtimeVoiceSidecarSession:
    """One realtime sidecar session.

    Audio chunks are accumulated until the client marks an utterance boundary.
    This makes the baseline path usable without specialized streaming ASR. The
    same event contract is kept for higher-grade frontends: sidecars can emit
    partial transcript events as soon as they have them.
    """

    def __init__(
        self,
        runtime: ReferenceSidecarRuntimeConfig,
        *,
        transcribe_audio_func: Optional[TranscribeFn] = None,
        synthesize_func: Optional[SynthesizeFn] = None,
    ):
        self.runtime = runtime
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._transcribe_audio_func = transcribe_audio_func
        self._synthesize_func = synthesize_func
        self._events: asyncio.Queue[VoiceEvent | None] = create_realtime_voice_event_queue()
        self._audio: list[bytes] = []
        self._audio_bytes = 0
        self._audio_input_generation: Optional[int] = None
        self._audio_started_at: Optional[float] = None
        self._sequence = 0
        self._closed = False
        self._active_tasks: set[asyncio.Task[None]] = set()
        self._speak_lock = asyncio.Lock()
        self._streaming_stt: Optional[RealtimeVoiceSidecarClient] = None
        self._streaming_stt_task: Optional[asyncio.Task[None]] = None
        self._asr_hypotheses_by_generation: dict[int, dict[str, Any]] = {}
        self._streaming_tts: Optional[RealtimeVoiceSidecarClient] = None
        self._streaming_tts_task: Optional[asyncio.Task[None]] = None
        self._last_streaming_tts_failure: Optional[dict[str, str]] = None
        self._openai_realtime: Any = None
        self._openai_realtime_task: Optional[asyncio.Task[None]] = None
        self._gemini_live: Any = None
        self._gemini_live_task: Optional[asyncio.Task[None]] = None
        self._active_playback_generations: set[Optional[int]] = set()
        self._last_speech_lifecycle_event: Optional[dict[str, Any]] = None
        self._kame_feedback_events: list[dict[str, Any]] = []
        self._kame_feedback_events_by_generation: dict[int, list[dict[str, Any]]] = {}
        self._kame_last_interface_event: Optional[dict[str, Any]] = None
        self._kame_last_oracle_event: Optional[dict[str, Any]] = None

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        self.runtime = _runtime_with_session_config(self.runtime, config)
        self.config = config
        await self._emit(
            VoiceEventType.SESSION_STARTED,
            {
                "engine": config.engine.value,
                "input_codec": config.input_codec.value,
                "output_codec": config.output_codec.value,
                "frontend_provider": config.frontend_provider or "",
                "frontend_model": config.frontend_model or "",
                "sidecar": True,
                **realtime_voice_session_contract_payload(config),
            },
        )
        requested_provider = str(config.frontend_provider or "").strip().lower()
        if requested_provider in {"openai_realtime", "openai"}:
            await self._start_openai_realtime(config)
        if requested_provider in {"gemini_live", "gemini"}:
            await self._start_gemini_live(config)
        if (
            self._openai_realtime is None
            and self._gemini_live is None
            and self.runtime.streaming_stt_base_url
            and self._should_start_streaming_stt(config)
        ):
            await self._start_streaming_stt(config)
        if self._openai_realtime is None and self._gemini_live is None and self.runtime.streaming_tts_base_url:
            await self._start_streaming_tts(config)
        fallback_reason = self._kame_audio_reflex_fallback_reason(config)
        text_fallback_requested = _interface_audio_input(config) == "text_fallback"
        vllm_drives_reflex = bool(
            self._openai_realtime is None
            and self._gemini_live is None
            and self._wants_kame_vllm_reflex()
            and not fallback_reason
            and not text_fallback_requested
        )
        streaming_stt_drives_reflex = self._streaming_stt is not None and self._streaming_stt_drives_reflex()
        local_stt_drives_reflex = bool(
            (fallback_reason or text_fallback_requested)
            and self._streaming_stt is None
            and self.runtime.local_stt_enabled
            and self._kame_stt_reflex_fallback_allowed(config)
        )
        kame_text_fallback_drives_reflex = bool(
            config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE
            and (local_stt_drives_reflex or streaming_stt_drives_reflex)
        )
        implementation_provider = (
            "openai_realtime"
            if self._openai_realtime is not None
            else "gemini_live" if self._gemini_live is not None
            else "vllm" if vllm_drives_reflex
            else "streaming_stt" if streaming_stt_drives_reflex
            else "local_stt" if local_stt_drives_reflex
            else "unavailable" if fallback_reason
            else "local"
        )
        provider = (
            implementation_provider
            if implementation_provider in {"openai_realtime", "gemini_live", "local_stt", "unavailable", "local"}
            else _provider_label(config.frontend_provider or self.runtime.interface_provider, default="vllm")
            if implementation_provider == "vllm"
            else _provider_label(config.asr_provider or self.runtime.streaming_stt_provider, default="streaming_stt")
            if implementation_provider == "streaming_stt"
            else implementation_provider
        )
        payload = {
            "status": (
                "fallback"
                if kame_text_fallback_drives_reflex
                else "degraded" if fallback_reason else "ready"
            ),
            "provider": provider,
            "model": _reported_frontend_model(
                provider,
                implementation_provider=implementation_provider,
                runtime=self.runtime,
                config=config,
            ),
            "streaming_stt": self._streaming_stt is not None,
            "streaming_tts": self._streaming_tts is not None,
            "vllm": bool(self.runtime.vllm_base_url and self.runtime.vllm_model),
            "local_stt": self.runtime.local_stt_enabled,
            "local_tts": self.runtime.local_tts_enabled,
            "asr_mode": config.asr_mode.value,
            "interface_audio_input": _interface_audio_input(config),
        }
        if implementation_provider != provider:
            payload["implementation_provider"] = implementation_provider
        if fallback_reason:
            payload.update(
                {
                    "reason": fallback_reason,
                    "requested_provider": config.frontend_provider or "",
                    "fallback_provider": provider,
                }
            )
            if kame_text_fallback_drives_reflex:
                payload.update(
                    {
                        "intent_source": "asr_fallback",
                        "transcript_source": "asr",
                }
            )
        elif kame_text_fallback_drives_reflex:
            payload.update(
                {
                    "reason": "kame_text_fallback_requested",
                    "requested_provider": config.frontend_provider or "",
                    "fallback_provider": provider,
                    "intent_source": "asr_fallback",
                    "transcript_source": "asr",
                }
            )
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            payload,
        )

    async def receive_event(self, event: VoiceEvent) -> None:
        if self._closed:
            return
        if event.type in {VoiceEventType.SESSION_STOP, VoiceEventType.SESSION_CLOSED}:
            await self.close()
            return
        if event.type == VoiceEventType.BARGE_IN:
            self._clear_kame_feedback_state()
            self._clear_audio_buffer()
            cancelled_tasks = self._cancel_active_tasks()
            payload = {"reason": event.payload.get("reason") or "client"}
            playback_generation = _payload_generation(event.payload)
            if playback_generation is not None:
                payload["playback_generation"] = playback_generation
            await self._emit(VoiceEventType.BARGE_IN, payload)
            await self._emit_interrupted_playback_finalizers(payload)
            if self._streaming_stt is not None:
                await self._send_streaming_stt_event(
                    VoiceEvent(
                        type=VoiceEventType.BARGE_IN,
                        session_id=event.session_id,
                        sequence=event.sequence,
                        timestamp_ms=event.timestamp_ms,
                        payload=payload,
                    )
                )
            if self._streaming_tts is not None:
                await self._send_streaming_tts_event(
                    VoiceEvent(
                        type=VoiceEventType.BARGE_IN,
                        session_id=event.session_id,
                        sequence=event.sequence,
                        timestamp_ms=event.timestamp_ms,
                        payload=payload,
                    )
                )
            if self._openai_realtime is not None:
                await self._send_openai_realtime_event(
                    VoiceEvent(
                        type=VoiceEventType.BARGE_IN,
                        session_id=event.session_id,
                        sequence=event.sequence,
                        timestamp_ms=event.timestamp_ms,
                        payload=payload,
                    )
                )
            if self._gemini_live is not None:
                await self._send_gemini_live_event(
                    VoiceEvent(
                        type=VoiceEventType.BARGE_IN,
                        session_id=event.session_id,
                        sequence=event.sequence,
                        timestamp_ms=event.timestamp_ms,
                        payload=payload,
                    )
                )
            await self._drain_cancelled_tasks(cancelled_tasks)
            return
        if event.type in {VoiceEventType.PLAYBACK_STARTED, VoiceEventType.PLAYBACK_STOPPED}:
            self._track_playback_lifecycle_event(event.type, event.payload)
            await self._forward_playback_lifecycle_event(event)
            return
        if event.type in KAME_FEEDBACK_EVENT_TYPES:
            self._record_kame_feedback_event(event)
            if event.type in KAME_EXTERNAL_CONTROL_EVENT_TYPES and event.payload.get("transport"):
                await self._emit(event.type, {**dict(event.payload), "sidecar_control": True})
            if event.type in KAME_LIVE_FRONTEND_ORACLE_CONTEXT_EVENT_TYPES:
                await self._forward_live_frontend_oracle_context_event(event)
            return
        if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL and event.payload.get("speak") is True:
            text = str(event.payload.get("text") or "").strip()
            if text:
                if self._openai_realtime is not None:
                    await self._send_openai_realtime_event(event)
                    return
                if self._gemini_live is not None:
                    await self._send_gemini_live_event(event)
                    return
                self._track_task(
                    asyncio.create_task(
                        self._speak_ordered(
                            text,
                            _payload_generation(event.payload),
                            _assistant_speak_metadata_from_payload(event.payload),
                        )
                    )
                )
            return
        if event.type in {VoiceEventType.SPEECH_START, VoiceEventType.SPEECH_ENERGY, VoiceEventType.SPEECH_END}:
            self._record_speech_lifecycle_event(event)
            await self._forward_speech_lifecycle_event(event)
            return
        if event.type != VoiceEventType.AUDIO_INPUT_CHUNK:
            return

        transcript = str(event.payload.get("transcript") or "").strip()
        if transcript:
            payload = {"text": transcript, **transcript_metadata_from_payload(event.payload)}
            input_generation = _payload_input_generation(event.payload)
            if input_generation is not None:
                payload["input_generation"] = input_generation
            if event.payload.get("end_of_utterance") is True:
                await self._emit(VoiceEventType.TRANSCRIPT_FINAL, payload)
            else:
                if self.config is not None and self.config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
                    partial_intent_payload = _kame_interface_partial_payload_from_payload(event.payload)
                    if partial_intent_payload:
                        await self._emit(VoiceEventType.INTERFACE_INTENT_PARTIAL, partial_intent_payload)
                    if not _allow_kame_transcript_events(self.config):
                        return
                payload["stability"] = 0.8
                await self._emit(VoiceEventType.TRANSCRIPT_PARTIAL, payload)
            return

        try:
            chunk = AudioChunk.from_payload(event.payload)
        except Exception:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid audio chunk"})
            return

        input_generation = _payload_input_generation(event.payload)
        if input_generation is not None:
            if self._audio_input_generation is not None and input_generation != self._audio_input_generation:
                self._clear_audio_buffer()
            self._audio_input_generation = input_generation
            self._clear_stale_speech_lifecycle_event(input_generation)
        if self._openai_realtime is not None:
            await self._send_openai_realtime_event(event)
            return
        if self._gemini_live is not None:
            await self._send_gemini_live_event(event)
            return
        if self._streaming_stt is not None:
            if await self._send_streaming_stt_event(event) and self._streaming_stt_drives_reflex():
                return
        if not await self._append_audio_chunk(chunk.data):
            return
        if event.payload.get("end_of_utterance") is True:
            audio = b"".join(self._audio)
            audio_input_generation = self._audio_input_generation
            audio_started_at = self._audio_started_at
            speech_boundary_at = time.perf_counter()
            self._clear_audio_buffer()
            if audio:
                if _allow_kame_transcript_events(self.config):
                    payload = {"text": "", "stability": 0.1}
                    if audio_input_generation is not None:
                        payload["input_generation"] = audio_input_generation
                    await self._emit(VoiceEventType.TRANSCRIPT_PARTIAL, payload)
                self._track_task(
                    asyncio.create_task(
                        self._transcribe(
                            audio,
                            chunk.codec,
                            audio_input_generation,
                            audio_started_at=audio_started_at,
                            speech_boundary_at=speech_boundary_at,
                        )
                    )
                )

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
        await self._drain_cancelled_tasks(self._cancel_active_tasks())
        providers = (
            (self._streaming_stt, "streaming_stt"),
            (self._streaming_tts, "streaming_tts"),
            (self._openai_realtime, "openai_realtime"),
            (self._gemini_live, "gemini_live"),
        )
        await self._send_provider_session_stops(providers)
        streaming_stt_task = self._streaming_stt_task
        streaming_tts_task = self._streaming_tts_task
        openai_realtime_task = self._openai_realtime_task
        gemini_live_task = self._gemini_live_task
        self._streaming_stt_task = None
        self._streaming_tts_task = None
        self._openai_realtime_task = None
        self._gemini_live_task = None
        for task in (streaming_stt_task, streaming_tts_task, openai_realtime_task, gemini_live_task):
            if task is not None and not task.done():
                task.cancel()

        streaming_stt = self._streaming_stt
        streaming_tts = self._streaming_tts
        openai_realtime = self._openai_realtime
        gemini_live = self._gemini_live
        self._streaming_stt = None
        self._streaming_tts = None
        self._openai_realtime = None
        self._gemini_live = None
        self._asr_hypotheses_by_generation.clear()
        self._clear_kame_feedback_state()
        self._last_speech_lifecycle_event = None
        self._last_streaming_tts_failure = None
        self._clear_audio_buffer()

        await self._close_provider(streaming_stt, "streaming_stt")
        await self._close_provider(streaming_tts, "streaming_tts")
        await self._close_provider(openai_realtime, "openai_realtime")
        await self._close_provider(gemini_live, "gemini_live")
        if streaming_stt_task:
            with contextlib.suppress(asyncio.CancelledError):
                await streaming_stt_task
        if streaming_tts_task:
            with contextlib.suppress(asyncio.CancelledError):
                await streaming_tts_task
        if openai_realtime_task:
            with contextlib.suppress(asyncio.CancelledError):
                await openai_realtime_task
        if gemini_live_task:
            with contextlib.suppress(asyncio.CancelledError):
                await gemini_live_task
        await self._emit_shutdown_playback_finalizers()
        await self._emit(VoiceEventType.SESSION_CLOSED, {"reason": "closed"})
        await put_realtime_voice_event(self._events, None)

    async def _start_openai_realtime(self, config: RealtimeVoiceSessionConfig) -> None:
        from agent.realtime_voice_openai import OpenAIRealtimeFrontendConfig, OpenAIRealtimeFrontendSession

        provider = OpenAIRealtimeFrontendSession(
            OpenAIRealtimeFrontendConfig(
                api_key=self.runtime.openai_realtime_api_key,
                base_url=self.runtime.openai_realtime_base_url,
                model=self.runtime.openai_realtime_model,
                voice=self.runtime.openai_realtime_voice,
                transcription_model=self.runtime.openai_realtime_transcription_model,
                connect_timeout_seconds=config.sidecar_connect_timeout_seconds,
                safety_identifier=self.runtime.openai_realtime_safety_identifier,
            )
        )
        try:
            await provider.start(config)
        except Exception as exc:
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "openai_realtime_unavailable",
                    "error": sanitize_realtime_voice_error(exc),
                    "openai_realtime": False,
                },
            )
            return
        self._openai_realtime = provider
        self._openai_realtime_task = asyncio.create_task(self._consume_openai_realtime_events())

    async def _start_gemini_live(self, config: RealtimeVoiceSessionConfig) -> None:
        from agent.realtime_voice_gemini import GeminiLiveFrontendConfig, GeminiLiveFrontendSession

        provider = GeminiLiveFrontendSession(
            GeminiLiveFrontendConfig(
                api_key=self.runtime.gemini_live_api_key,
                base_url=self.runtime.gemini_live_base_url,
                model=self.runtime.gemini_live_model,
                voice=self.runtime.gemini_live_voice,
                connect_timeout_seconds=config.sidecar_connect_timeout_seconds,
                enable_google_search=self.runtime.gemini_live_google_search,
                enable_oracle_tool=self.runtime.gemini_live_oracle_tool,
            )
        )
        try:
            await provider.start(config)
        except Exception as exc:
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "gemini_live_unavailable",
                    "error": sanitize_realtime_voice_error(exc),
                    "gemini_live": False,
                },
            )
            return
        self._gemini_live = provider
        self._gemini_live_task = asyncio.create_task(self._consume_gemini_live_events())

    async def _start_streaming_stt(self, config: RealtimeVoiceSessionConfig) -> None:
        client = RealtimeVoiceSidecarClient(path="/v1/streaming-stt/session")
        downstream_config = RealtimeVoiceSessionConfig(
            session_id=config.session_id,
            engine=config.engine,
            input_codec=config.input_codec,
            output_codec=config.output_codec,
            sample_rate_hz=config.sample_rate_hz,
            channels=config.channels,
            input_buffer_limit_bytes=config.input_buffer_limit_bytes,
            frontend_provider="streaming_stt",
            frontend_model=self.runtime.streaming_stt_model or config.frontend_model,
            interface_audio_input=config.interface_audio_input,
            asr_mode=config.asr_mode,
            asr_provider=config.asr_provider,
            asr_model=config.asr_model,
            oracle_timeout_seconds=config.oracle_timeout_seconds,
            max_spoken_sentences=config.max_spoken_sentences,
            tts_provider=config.tts_provider,
            tts_model=config.tts_model,
            tts_voice=config.tts_voice,
            fallback_policy=config.fallback_policy,
            sidecar_base_url=self.runtime.streaming_stt_base_url,
            sidecar_token=self.runtime.streaming_stt_token,
            sidecar_connect_timeout_seconds=self.runtime.streaming_stt_timeout_seconds,
            metadata=config.metadata,
        )
        try:
            await client.start(downstream_config)
        except Exception as exc:
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "streaming_stt_unavailable",
                    "error": sanitize_realtime_voice_error(exc),
                    "streaming_stt": False,
                },
            )
            return
        self._streaming_stt = client
        self._streaming_stt_task = asyncio.create_task(self._consume_streaming_stt_events())

    async def _start_streaming_tts(self, config: RealtimeVoiceSessionConfig) -> None:
        client = RealtimeVoiceSidecarClient(path="/v1/streaming-tts/session")
        downstream_config = RealtimeVoiceSessionConfig(
            session_id=config.session_id,
            engine=config.engine,
            input_codec=config.input_codec,
            output_codec=config.output_codec,
            sample_rate_hz=config.sample_rate_hz,
            channels=config.channels,
            input_buffer_limit_bytes=config.input_buffer_limit_bytes,
            frontend_provider="streaming_tts",
            frontend_model=self.runtime.streaming_tts_model or config.frontend_model,
            interface_audio_input=config.interface_audio_input,
            asr_mode=config.asr_mode,
            asr_provider=config.asr_provider,
            asr_model=config.asr_model,
            oracle_timeout_seconds=config.oracle_timeout_seconds,
            max_spoken_sentences=config.max_spoken_sentences,
            tts_provider=config.tts_provider,
            tts_model=config.tts_model,
            tts_voice=config.tts_voice,
            fallback_policy=config.fallback_policy,
            sidecar_base_url=self.runtime.streaming_tts_base_url,
            sidecar_token=self.runtime.streaming_tts_token,
            sidecar_connect_timeout_seconds=self.runtime.streaming_tts_timeout_seconds,
            metadata=config.metadata,
        )
        try:
            await client.start(downstream_config)
        except Exception as exc:
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "streaming_tts_unavailable",
                    "error": sanitize_realtime_voice_error(exc),
                    "streaming_tts": False,
                },
            )
            return
        self._streaming_tts = client
        self._last_streaming_tts_failure = None
        self._streaming_tts_task = asyncio.create_task(self._consume_streaming_tts_events())

    async def _send_streaming_stt_event(self, event: VoiceEvent) -> bool:
        if self._streaming_stt is None:
            return False
        try:
            await self._streaming_stt.send_event(event)
            return True
        except Exception as exc:
            await self._disable_streaming_stt("streaming_stt_send_failed", exc)
            return False

    async def _send_streaming_tts_event(self, event: VoiceEvent) -> bool:
        if self._streaming_tts is None:
            return False
        try:
            await self._streaming_tts.send_event(event)
            return True
        except Exception as exc:
            await self._disable_streaming_tts("streaming_tts_send_failed", exc)
            return False

    async def _send_openai_realtime_event(self, event: VoiceEvent) -> bool:
        if self._openai_realtime is None:
            return False
        try:
            await self._openai_realtime.receive_event(event)
            return True
        except Exception as exc:
            await self._disable_openai_realtime("openai_realtime_send_failed", exc)
            return False

    async def _send_gemini_live_event(self, event: VoiceEvent) -> bool:
        if self._gemini_live is None:
            return False
        try:
            await self._gemini_live.receive_event(event)
            return True
        except Exception as exc:
            await self._disable_gemini_live("gemini_live_send_failed", exc)
            return False

    async def _forward_speech_lifecycle_event(self, event: VoiceEvent) -> None:
        if self._streaming_stt is not None:
            await self._send_streaming_stt_event(event)
        if self._openai_realtime is not None:
            await self._send_openai_realtime_event(event)
        if self._gemini_live is not None:
            await self._send_gemini_live_event(event)

    async def _forward_playback_lifecycle_event(self, event: VoiceEvent) -> None:
        if self._openai_realtime is not None:
            await self._send_openai_realtime_event(event)
        if self._gemini_live is not None:
            await self._send_gemini_live_event(event)

    async def _forward_live_frontend_oracle_context_event(self, event: VoiceEvent) -> None:
        if self._openai_realtime is not None:
            await self._send_openai_realtime_event(event)
        if self._gemini_live is not None:
            await self._send_gemini_live_event(event)

    async def _consume_streaming_stt_events(self) -> None:
        if self._streaming_stt is None:
            return
        try:
            async for event in self._streaming_stt.events():
                if event.type in {VoiceEventType.TRANSCRIPT_PARTIAL, VoiceEventType.TRANSCRIPT_FINAL}:
                    if self._suppress_streaming_stt_transcript_events():
                        self._record_asr_hypothesis(event)
                        continue
                    text = str(event.payload.get("text") or "")
                    payload = transcript_metadata_from_payload(event.payload) | {
                        "text": text,
                        **_numeric_transcript_fields(event.payload),
                        **_generation_transcript_fields(event.payload),
                    }
                    if (
                        event.type == VoiceEventType.TRANSCRIPT_FINAL
                        and self.config is not None
                        and self.config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE
                        and self._streaming_stt_drives_reflex()
                    ):
                        payload.update(
                            {
                                "intent": text,
                                "intent_source": "asr_fallback",
                                "route": KameRoute.ORACLE_DIRECT.value,
                                "transcript": text,
                                "transcript_source": "asr",
                                "asr_transcript": text,
                                "asr_transcript_source": "asr",
                                "interface_audio_input_fallback": True,
                                "interface_input_source": "streaming_stt",
                                "reflex_provider": "streaming_stt",
                            }
                        )
                    await self._emit(event.type, payload)
                elif event.type == VoiceEventType.FRONTEND_STATE:
                    payload = dict(event.payload)
                    payload.setdefault("streaming_stt", True)
                    await self._emit(VoiceEventType.FRONTEND_STATE, payload)
                elif event.type == VoiceEventType.SESSION_ERROR:
                    await self._disable_streaming_stt(
                        "streaming_stt_session_error",
                        event.payload.get("error") or "",
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._disable_streaming_stt("streaming_stt_event_stream_failed", exc)

    async def _consume_streaming_tts_events(self) -> None:
        if self._streaming_tts is None:
            return
        try:
            async for event in self._streaming_tts.events():
                if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
                    await self._emit(VoiceEventType.AUDIO_OUTPUT_CHUNK, dict(event.payload))
                elif event.type in {
                    VoiceEventType.ASSISTANT_AUDIO_END,
                    VoiceEventType.PLAYBACK_STARTED,
                    VoiceEventType.PLAYBACK_STOPPED,
                }:
                    await self._emit(event.type, dict(event.payload))
                elif event.type == VoiceEventType.FRONTEND_STATE:
                    payload = dict(event.payload)
                    payload.setdefault("streaming_tts", True)
                    await self._emit(VoiceEventType.FRONTEND_STATE, payload)
                elif event.type == VoiceEventType.SESSION_ERROR:
                    await self._disable_streaming_tts(
                        "streaming_tts_session_error",
                        event.payload.get("error") or "",
                    )
                    return
                elif event.type == VoiceEventType.BARGE_IN:
                    await self._emit(VoiceEventType.BARGE_IN, dict(event.payload))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._disable_streaming_tts("streaming_tts_event_stream_failed", exc)

    async def _consume_openai_realtime_events(self) -> None:
        if self._openai_realtime is None:
            return
        try:
            async for event in self._openai_realtime.events():
                if event.type in PROVIDER_FORWARDED_EVENT_TYPES:
                    await self._emit(event.type, dict(event.payload))
                elif event.type == VoiceEventType.SESSION_ERROR:
                    await self._disable_openai_realtime(
                        "openai_realtime_session_error",
                        event.payload.get("error") or "",
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._disable_openai_realtime("openai_realtime_event_stream_failed", exc)

    async def _consume_gemini_live_events(self) -> None:
        if self._gemini_live is None:
            return
        try:
            async for event in self._gemini_live.events():
                if event.type in PROVIDER_FORWARDED_EVENT_TYPES:
                    await self._emit(event.type, dict(event.payload))
                elif event.type == VoiceEventType.SESSION_ERROR:
                    await self._disable_gemini_live(
                        "gemini_live_session_error",
                        event.payload.get("error") or "",
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._disable_gemini_live("gemini_live_event_stream_failed", exc)

    async def _disable_streaming_stt(self, reason: str, error: Any) -> None:
        client = self._streaming_stt
        self._streaming_stt = None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "degraded",
                "reason": reason,
                "error": sanitize_realtime_voice_error(error),
                "streaming_stt": False,
            },
        )

    async def _disable_streaming_tts(self, reason: str, error: Any) -> None:
        client = self._streaming_tts
        self._streaming_tts = None
        sanitized_error = sanitize_realtime_voice_error(error)
        self._last_streaming_tts_failure = {
            "reason": reason,
            "error": sanitized_error,
        }
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "degraded",
                "reason": reason,
                "error": sanitized_error,
                "streaming_tts": False,
            },
        )

    async def _disable_openai_realtime(self, reason: str, error: Any) -> None:
        provider = self._openai_realtime
        self._openai_realtime = None
        await self._close_provider(provider, "openai_realtime")
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "degraded",
                "reason": reason,
                "error": sanitize_realtime_voice_error(error),
                "openai_realtime": False,
            },
        )

    async def _disable_gemini_live(self, reason: str, error: Any) -> None:
        provider = self._gemini_live
        self._gemini_live = None
        await self._close_provider(provider, "gemini_live")
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "degraded",
                "reason": reason,
                "error": sanitize_realtime_voice_error(error),
                "gemini_live": False,
            },
        )

    def _track_task(self, task: asyncio.Task[None]) -> None:
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    def _cancel_active_tasks(self) -> list[asyncio.Task[None]]:
        tasks = list(self._active_tasks)
        for task in list(self._active_tasks):
            if not task.done():
                task.cancel()
        return tasks

    async def _drain_cancelled_tasks(self, tasks: list[asyncio.Task[None]]) -> None:
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=REFERENCE_SIDECAR_CLOSE_DRAIN_TIMEOUT_SECONDS)
        self._active_tasks.difference_update(pending)
        for task in done:
            try:
                task.result()
            except (asyncio.CancelledError, Exception):
                pass

    async def _close_provider(self, provider: Any, label: str) -> None:
        if provider is None:
            return
        close = getattr(provider, "close", None)
        if close is None:
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await asyncio.wait_for(result, timeout=REFERENCE_SIDECAR_PROVIDER_CLOSE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("Realtime voice sidecar provider close timed out: %s", label)
        except Exception as exc:
            logger.debug("Realtime voice sidecar provider close failed (%s): %s", label, exc)

    async def _send_provider_session_stops(self, providers: tuple[tuple[Any, str], ...]) -> None:
        for provider, label in providers:
            await self._send_provider_session_stop(provider, label)

    async def _send_provider_session_stop(self, provider: Any, label: str) -> None:
        if provider is None or self.config is None:
            return
        send = getattr(provider, "receive_event", None) or getattr(provider, "send_event", None)
        if send is None:
            return
        self._sequence += 1
        event = VoiceEvent(
            type=VoiceEventType.SESSION_CLOSED,
            session_id=self.config.session_id,
            sequence=self._sequence,
            payload={"reason": "sidecar_session_closed"},
        )
        try:
            result = send(event)
            if inspect.isawaitable(result):
                await asyncio.wait_for(result, timeout=REFERENCE_SIDECAR_PROVIDER_CLOSE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("Realtime voice sidecar provider stop timed out: %s", label)
        except Exception as exc:
            logger.debug("Realtime voice sidecar provider stop failed (%s): %s", label, exc)

    async def _append_audio_chunk(self, data: bytes) -> bool:
        config = self.config
        limit = int(config.input_buffer_limit_bytes if config is not None else 8 * 1024 * 1024)
        if self._audio_bytes + len(data) > max(1, limit):
            self._clear_audio_buffer()
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "input_buffer_limit_exceeded",
                    "sidecar": True,
                    "limit_bytes": limit,
                },
            )
            return False
        self._audio.append(data)
        self._audio_bytes += len(data)
        if data and self._audio_started_at is None:
            self._audio_started_at = time.perf_counter()
        return True

    def _clear_audio_buffer(self) -> None:
        self._audio.clear()
        self._audio_bytes = 0
        self._audio_input_generation = None
        self._audio_started_at = None

    def _should_start_streaming_stt(self, config: RealtimeVoiceSessionConfig) -> bool:
        if config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
            return True
        interface_audio_input = _interface_audio_input(config)
        return (
            config.asr_mode.value in {"fallback", "debug", "speculative"}
            or interface_audio_input == "text_fallback"
        )

    def _streaming_stt_drives_reflex(self) -> bool:
        config = self.config
        if config is None or config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
            return True
        interface_audio_input = _interface_audio_input(config)
        return (
            config.asr_mode.value == "fallback"
            or interface_audio_input == "text_fallback"
        )

    def _suppress_streaming_stt_transcript_events(self) -> bool:
        config = self.config
        if config is None or config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
            return False
        return not _allow_kame_transcript_events(config)

    def _record_asr_hypothesis(self, event: VoiceEvent) -> None:
        if event.type != VoiceEventType.TRANSCRIPT_FINAL:
            return
        generation = _payload_input_generation(event.payload)
        if generation is None:
            return
        text = str(event.payload.get("text") or "").strip()
        if not text:
            return
        hypothesis: dict[str, Any] = {
            "asr_transcript": text,
            "asr_transcript_source": "asr",
        }
        confidence = _bounded_confidence(event.payload.get("confidence"))
        if confidence is not None:
            hypothesis["asr_transcript_confidence"] = confidence
        self._asr_hypotheses_by_generation[generation] = hypothesis

    async def _transcribe(
        self,
        audio: bytes,
        codec: VoiceAudioCodec,
        input_generation: Optional[int] = None,
        *,
        audio_started_at: Optional[float] = None,
        speech_boundary_at: Optional[float] = None,
    ) -> None:
        try:
            understand_started_at = time.perf_counter()
            payload = await asyncio.to_thread(self._understand_audio_sync, audio, codec)
            if self.config is not None and self.config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
                interface_decision_at = time.perf_counter()
                metrics = dict(payload.get("metrics") or {}) if isinstance(payload.get("metrics"), Mapping) else {}
                metrics["kame_speech_end_to_interface_decision_ms"] = int(
                    round((interface_decision_at - (speech_boundary_at or understand_started_at)) * 1000)
                )
                if audio_started_at is not None:
                    metrics["kame_first_audio_to_interface_decision_ms"] = int(
                        round((interface_decision_at - audio_started_at) * 1000)
                    )
                if speech_boundary_at is not None:
                    metrics["kame_speech_boundary_to_final_intent_ms"] = int(
                        round((interface_decision_at - speech_boundary_at) * 1000)
                    )
                payload["metrics"] = metrics
            fallback_reason = str(payload.get("fallback_reason") or "").strip()
            if fallback_reason:
                await self._emit(
                    VoiceEventType.FRONTEND_STATE,
                    {
                        "status": "fallback",
                        "reason": fallback_reason,
                        "provider": "local_stt",
                        "requested_provider": self.config.frontend_provider if self.config is not None else "",
                        "fallback_provider": "local_stt",
                        "intent_source": "asr_fallback",
                        "transcript_source": "asr",
                        **({"error": str(payload.get("fallback_error") or "")} if payload.get("fallback_error") else {}),
                    },
                )
            text = str(payload.get("text") or "").strip()
            if text:
                if input_generation is not None:
                    asr_hypothesis = self._asr_hypotheses_by_generation.pop(input_generation, None)
                    if (
                        asr_hypothesis
                        and self.config is not None
                        and self.config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE
                        and _kame_payload_accepts_oracle_asr_evidence(payload)
                    ):
                        payload.setdefault("asr_transcript", asr_hypothesis.get("asr_transcript"))
                        payload.setdefault("asr_transcript_source", asr_hypothesis.get("asr_transcript_source"))
                        if "asr_transcript_confidence" in asr_hypothesis:
                            payload.setdefault("asr_transcript_confidence", asr_hypothesis["asr_transcript_confidence"])
                if self._should_run_oracle_verbatim_asr(payload):
                    started_at = time.perf_counter()
                    asr_hypothesis = await self._run_oracle_verbatim_asr_once(
                        audio,
                        codec,
                        input_generation,
                    )
                    if asr_hypothesis:
                        payload["asr_transcript"] = asr_hypothesis["asr_transcript"]
                        payload["asr_transcript_source"] = asr_hypothesis["asr_transcript_source"]
                        if "asr_transcript_confidence" in asr_hypothesis:
                            payload["asr_transcript_confidence"] = asr_hypothesis["asr_transcript_confidence"]
                        existing_metrics = payload.get("metrics")
                        metrics = dict(existing_metrics) if isinstance(existing_metrics, Mapping) else {}
                        metrics["oracle_verbatim_asr_ms"] = int(round((time.perf_counter() - started_at) * 1000))
                        payload["metrics"] = metrics
                if input_generation is not None:
                    payload["input_generation"] = input_generation
                await self._emit(_final_understanding_event_type(self.config), payload)
        except asyncio.CancelledError:
            raise
        except KameAudioSegmentTooLongError as exc:
            error = sanitize_realtime_voice_error(exc)
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "kame_audio_segment_too_long",
                    "error": error,
                    "interface_audio_input": _interface_audio_input(self.config),
                    "interface_max_audio_seconds": _interface_max_audio_seconds(self.config),
                },
            )
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"kame audio segment too long: {error}"},
            )
        except Exception as exc:
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"transcription failed: {sanitize_realtime_voice_error(exc)}"},
            )

    def _should_run_oracle_verbatim_asr(self, payload: Mapping[str, Any]) -> bool:
        config = self.config
        if config is None or config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
            return False
        # Full KAME keeps ASR out of the hot path. Transcript providers may
        # attach hypothesis evidence through fallback/debug/speculative lanes,
        # but final reflex intent must not wait for a one-shot ASR request.
        return False

    async def _run_oracle_verbatim_asr_once(
        self,
        audio: bytes,
        codec: VoiceAudioCodec,
        input_generation: Optional[int],
    ) -> dict[str, Any]:
        config = self.config
        if config is None or not audio:
            return {}
        client = RealtimeVoiceSidecarClient(path="/v1/streaming-stt/session")
        downstream_config = RealtimeVoiceSessionConfig(
            session_id=config.session_id,
            engine=config.engine,
            input_codec=config.input_codec,
            output_codec=config.output_codec,
            sample_rate_hz=config.sample_rate_hz,
            channels=config.channels,
            input_buffer_limit_bytes=config.input_buffer_limit_bytes,
            frontend_provider="streaming_stt",
            frontend_model=self.runtime.streaming_stt_model or config.frontend_model,
            interface_audio_input=config.interface_audio_input,
            asr_mode=config.asr_mode,
            asr_provider=config.asr_provider,
            asr_model=config.asr_model,
            oracle_timeout_seconds=config.oracle_timeout_seconds,
            max_spoken_sentences=config.max_spoken_sentences,
            tts_provider=config.tts_provider,
            tts_model=config.tts_model,
            tts_voice=config.tts_voice,
            fallback_policy=config.fallback_policy,
            sidecar_base_url=self.runtime.streaming_stt_base_url,
            sidecar_token=self.runtime.streaming_stt_token,
            sidecar_connect_timeout_seconds=self.runtime.streaming_stt_timeout_seconds,
            metadata=config.metadata,
        )
        try:
            await client.start(downstream_config)
            payload = AudioChunk(codec=codec, data=audio).to_payload()
            payload["end_of_utterance"] = True
            if input_generation is not None:
                payload["input_generation"] = input_generation
            await client.send_event(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_INPUT_CHUNK,
                    session_id=config.session_id,
                    sequence=self._sequence + 1,
                    payload=payload,
                )
            )
            event_stream = client.events()
            while True:
                event = await asyncio.wait_for(
                    event_stream.__anext__(),
                    timeout=max(0.1, float(self.runtime.streaming_stt_timeout_seconds)),
                )
                if event.type == VoiceEventType.TRANSCRIPT_FINAL:
                    text = str(event.payload.get("text") or "").strip()
                    if not text:
                        return {}
                    hypothesis: dict[str, Any] = {
                        "asr_transcript": text,
                        "asr_transcript_source": "asr",
                    }
                    confidence = _bounded_confidence(event.payload.get("confidence"))
                    if confidence is not None:
                        hypothesis["asr_transcript_confidence"] = confidence
                    return hypothesis
                if event.type == VoiceEventType.SESSION_ERROR:
                    await self._emit(
                        VoiceEventType.FRONTEND_STATE,
                        {
                            "status": "degraded",
                            "reason": "oracle_verbatim_asr_failed",
                            "error": sanitize_realtime_voice_error(event.payload.get("error") or ""),
                            "streaming_stt": False,
                        },
                    )
                    return {}
        except StopAsyncIteration:
            return {}
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                VoiceEventType.FRONTEND_STATE,
                {
                    "status": "degraded",
                    "reason": "oracle_verbatim_asr_failed",
                    "error": sanitize_realtime_voice_error(exc),
                    "streaming_stt": False,
                },
            )
        finally:
            with contextlib.suppress(Exception):
                await client.close()
        return {}

    def _transcribe_sync(self, audio: bytes, codec: VoiceAudioCodec) -> str:
        return str(self._understand_audio_sync(audio, codec).get("text") or "").strip()

    def _understand_audio_sync(self, audio: bytes, codec: VoiceAudioCodec) -> dict[str, Any]:
        if self._wants_kame_vllm_reflex():
            try:
                return self._understand_kame_with_vllm(audio, codec)
            except KameAudioSegmentTooLongError:
                raise
            except Exception as exc:
                if not (self.runtime.local_stt_enabled and self._kame_stt_reflex_fallback_allowed()):
                    raise RuntimeError(
                        "KAME audio reflex failed and ASR reflex fallback is disabled: "
                        f"{sanitize_realtime_voice_error(exc)}"
                    ) from exc
                logger.warning("KAME audio reflex failed; falling back to local STT: %s", sanitize_realtime_voice_error(exc))
                return self._understand_with_local_stt(
                    audio,
                    codec,
                    force_kame_fallback=True,
                    fallback_reason="kame_audio_reflex_failed",
                    fallback_error=sanitize_realtime_voice_error(exc),
                )
        if self.config is not None and self.config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
            if not self._kame_stt_reflex_fallback_allowed():
                raise RuntimeError("KAME audio reflex unavailable and ASR reflex fallback is disabled")
            if not self.runtime.local_stt_enabled:
                raise RuntimeError("local STT is disabled and no KAME audio reflex is configured")
            return self._understand_with_local_stt(audio, codec)
        if self.runtime.vllm_base_url and self.runtime.vllm_model:
            return {"text": self._transcribe_with_vllm(audio, codec)}
        if not self.runtime.local_stt_enabled:
            raise RuntimeError("local STT is disabled and no vLLM audio frontend is configured")

        return self._understand_with_local_stt(audio, codec)

    def _understand_with_local_stt(
        self,
        audio: bytes,
        codec: VoiceAudioCodec,
        *,
        force_kame_fallback: bool = False,
        fallback_reason: str = "",
        fallback_error: str = "",
    ) -> dict[str, Any]:
        transcribe_audio = self._transcribe_audio_func
        if transcribe_audio is None:
            from tools.transcription_tools import transcribe_audio as transcribe_audio

        path = _write_temp_audio(audio, codec, self.config)
        try:
            result = transcribe_audio(path)
            if not result.get("success"):
                raise RuntimeError(str(result.get("error") or "transcription failed"))
            text = str(result.get("transcript") or "").strip()
            if force_kame_fallback or self._uses_kame_local_stt_fallback():
                payload = {
                    "text": text,
                    "intent": text,
                    "intent_source": "asr_fallback",
                    "route": "oracle_direct",
                    "transcript": text,
                    "transcript_source": "asr",
                    "asr_transcript": text,
                    "asr_transcript_source": "asr",
                    "interface_audio_input_fallback": True,
                    "interface_input_source": "local_stt",
                    "reflex_provider": "local_stt",
                }
                if not fallback_reason and self.config is not None:
                    fallback_reason = self._kame_audio_reflex_fallback_reason(self.config)
                if fallback_reason:
                    payload["fallback_reason"] = fallback_reason
                if fallback_error:
                    payload["fallback_error"] = fallback_error
                return payload
            return {"text": text}
        finally:
            _unlink(path)

    def _wants_kame_vllm_reflex(self) -> bool:
        config = self.config
        return (
            config is not None
            and config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE
            and bool(self.runtime.vllm_base_url and self.runtime.vllm_model)
            and _interface_audio_input(config) != "text_fallback"
        )

    def _uses_kame_local_stt_fallback(self) -> bool:
        config = self.config
        return (
            config is not None
            and config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE
            and self._openai_realtime is None
            and self._gemini_live is None
            and (
                not bool(self.runtime.vllm_base_url and self.runtime.vllm_model)
                or _interface_audio_input(config) == "text_fallback"
            )
            and self._kame_stt_reflex_fallback_allowed(config)
        )

    def _kame_stt_reflex_fallback_allowed(self, config: Optional[RealtimeVoiceSessionConfig] = None) -> bool:
        config = config or self.config
        if config is None or config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
            return False
        interface_audio_input = _interface_audio_input(config)
        return config.asr_mode.value == "fallback" or interface_audio_input == "text_fallback"

    def _kame_audio_reflex_fallback_reason(self, config: RealtimeVoiceSessionConfig) -> str:
        if config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
            return ""
        if self._openai_realtime is not None or self._gemini_live is not None:
            return ""
        if _interface_audio_input(config) == "text_fallback":
            return ""
        if _interface_audio_input(config) == "auto" and self._kame_stt_reflex_fallback_allowed(config):
            return "kame_auto_text_fallback_selected"
        if self.runtime.vllm_base_url and self.runtime.vllm_model:
            return ""
        if self._streaming_stt is not None and self._kame_stt_reflex_fallback_allowed(config):
            return ""
        return "kame_audio_reflex_unavailable"

    def _transcribe_with_vllm(self, audio: bytes, codec: VoiceAudioCodec) -> str:
        mime_type = _mime_type_for_codec(codec)
        audio_b64 = base64.b64encode(_audio_bytes_for_codec(audio, codec, self.config)).decode("ascii")
        payload = {
            "model": self.runtime.vllm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": f"data:{mime_type};base64,{audio_b64}"}},
                        {
                            "type": "text",
                            "text": (
                                "Transcribe the speech in this audio. Preserve the speaker's language "
                                "and script. Return only the spoken words; do not translate or add commentary."
                            ),
                        },
                    ],
                }
            ],
            "max_tokens": 256,
            "temperature": 0,
        }
        url = f"{self.runtime.vllm_base_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=_vllm_request_headers(self.runtime, content_json=True),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.runtime.vllm_timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(_format_http_error(exc)) from exc
        return str(data["choices"][0]["message"].get("content") or "").strip()

    def _understand_kame_with_vllm(self, audio: bytes, codec: VoiceAudioCodec) -> dict[str, Any]:
        mime_type = _mime_type_for_codec(codec)
        self._raise_if_kame_audio_segment_too_long(audio, codec)
        audio_b64 = base64.b64encode(_audio_bytes_for_codec(audio, codec, self.config)).decode("ascii")
        config = self.config
        asr_mode = str(config.asr_mode.value if config is not None else "on_escalation")
        routing_policy = _kame_routing_policy_text(config)
        payload = {
            "model": self.runtime.vllm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": f"data:{mime_type};base64,{audio_b64}"}},
                        {
                            "type": "text",
                            "text": kame_reflex_instruction_text(
                                routing_policy=routing_policy,
                                asr_mode=asr_mode,
                            )
                            + "\n\n"
                            + self._kame_live_session_context_text(),
                        },
                    ],
                }
            ],
            "max_tokens": 256,
            "temperature": _interface_temperature(config),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "kame_reflex_decision",
                    "strict": True,
                    "schema": kame_reflex_decision_json_schema(),
                },
            },
        }
        payload["max_tokens"] = _interface_max_output_tokens(config)
        url = f"{self.runtime.vllm_base_url.rstrip('/')}/chat/completions"
        request_started_at = time.perf_counter()
        response_format_fallback = ""
        timeout = _interface_timeout_seconds(config, self.runtime.vllm_timeout_seconds)
        try:
            data = _post_vllm_chat_completion(self.runtime, url, payload, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if not _vllm_rejected_json_schema_response_format(exc):
                raise RuntimeError(_format_http_error(exc)) from exc
            fallback_payload = dict(payload)
            fallback_payload["response_format"] = {"type": "json_object"}
            try:
                data = _post_vllm_chat_completion(self.runtime, url, fallback_payload, timeout=timeout)
            except urllib.error.HTTPError as fallback_exc:
                raise RuntimeError(_format_http_error(fallback_exc)) from fallback_exc
            response_format_fallback = "json_object"
        request_ms = int(round((time.perf_counter() - request_started_at) * 1000))
        content = str(data["choices"][0]["message"].get("content") or "").strip()
        payload = _kame_reflex_payload_from_content(content, config=config)
        payload.setdefault("interface_input_source", "native_audio")
        payload.setdefault("reflex_provider", "vllm")
        if response_format_fallback:
            payload["reflex_response_format_fallback"] = response_format_fallback
        if _kame_provider_metrics_enabled(config):
            metrics = dict(payload.get("metrics")) if isinstance(payload.get("metrics"), Mapping) else {}
            metrics["kame_interface_model_request_ms"] = max(0, request_ms)
            if response_format_fallback:
                metrics["kame_interface_response_format_fallback"] = 1
            payload["metrics"] = metrics
        return payload

    def _raise_if_kame_audio_segment_too_long(self, audio: bytes, codec: VoiceAudioCodec) -> None:
        config = self.config
        max_seconds = _interface_max_audio_seconds(config)
        duration_seconds = _audio_duration_seconds(audio, codec, config)
        if duration_seconds is None:
            return
        if duration_seconds > max_seconds:
            raise KameAudioSegmentTooLongError(
                "KAME audio segment exceeds interface_max_audio_seconds "
                f"({duration_seconds:.2f}s > {max_seconds:.2f}s)"
            )

    async def _speak(
        self,
        text: str,
        playback_generation: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        clean_metadata = dict(metadata or {})
        if self._streaming_tts is not None and self.config is not None:
            sent = await self._send_streaming_tts_event(
                VoiceEvent(
                    type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                    session_id=self.config.session_id,
                    sequence=self._sequence + 1,
                    payload={
                        "text": text,
                        "speak": True,
                        **({"playback_generation": playback_generation} if playback_generation is not None else {}),
                        **clean_metadata,
                    },
                )
            )
            if sent:
                return
        if not self.runtime.local_tts_enabled:
            await self._emit_tts_unavailable_error(playback_generation)
            return
        try:
            tts_started_at = time.perf_counter()
            file_path = await asyncio.to_thread(self._speak_sync, text, clean_metadata)
            tts_synthesis_ms = int(round((time.perf_counter() - tts_started_at) * 1000))
            if not file_path:
                return
            try:
                with open(file_path, "rb") as fh:
                    data = fh.read()
                if data:
                    first_tts_audio_at = time.perf_counter()
                    playback_started_at = time.perf_counter()
                    playback_start_metrics = _kame_playback_start_metrics(
                        clean_metadata,
                        first_tts_audio_at,
                        playback_started_at,
                    )
                    await self._emit(
                        VoiceEventType.PLAYBACK_STARTED,
                        _playback_lifecycle_payload(
                            playback_generation,
                            clean_metadata,
                            playback_start_metrics,
                        ),
                    )
                    payload = _audio_file_to_pcm16_chunk(file_path, data).to_payload()
                    payload["mime_type"] = _mime_type_for_path(file_path)
                    if playback_generation is not None:
                        payload["playback_generation"] = playback_generation
                    payload.update(clean_metadata)
                    existing_metrics = payload.get("metrics")
                    metrics = dict(existing_metrics) if isinstance(existing_metrics, dict) else {}
                    metrics["tts_synthesis_ms"] = tts_synthesis_ms
                    metrics.update(playback_start_metrics)
                    payload["metrics"] = metrics
                    await self._emit(VoiceEventType.AUDIO_OUTPUT_CHUNK, payload)
                    await self._emit(
                        VoiceEventType.PLAYBACK_STOPPED,
                        {"playback_generation": playback_generation} if playback_generation is not None else {},
                    )
                    await self._emit(
                        VoiceEventType.ASSISTANT_AUDIO_END,
                        {"playback_generation": playback_generation} if playback_generation is not None else {},
                    )
            finally:
                _unlink(file_path)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"tts failed: {sanitize_realtime_voice_error(exc)}"},
            )

    async def _speak_ordered(
        self,
        text: str,
        playback_generation: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        async with self._speak_lock:
            await self._speak(text, playback_generation, metadata)

    async def _emit_tts_unavailable_error(self, playback_generation: Optional[int]) -> None:
        failure = self._last_streaming_tts_failure or {}
        reason = str(failure.get("reason") or "tts_unavailable")
        error = str(failure.get("error") or "local TTS disabled and no streaming TTS bridge is available")
        payload: dict[str, Any] = {
            "reason": "tts_unavailable",
            "error": f"{reason}: {error}" if reason != "tts_unavailable" else error,
            "streaming_tts": False,
            "local_tts": False,
        }
        if playback_generation is not None:
            payload["playback_generation"] = playback_generation
        config = self.config
        if config is not None:
            if config.tts_provider:
                payload["tts_provider"] = config.tts_provider
            if config.tts_model:
                payload["tts_model"] = config.tts_model
            if config.tts_voice:
                payload["tts_voice"] = config.tts_voice
        await self._emit(VoiceEventType.SESSION_ERROR, payload)

    def _speak_sync(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> str:
        synthesize = self._synthesize_func
        if synthesize is None:
            from tools.tts_tool import text_to_speech_tool as synthesize

        raw = _call_synthesize(synthesize, text, _tts_synthesis_metadata(metadata or {}))
        result = json.loads(raw) if isinstance(raw, str) else raw
        if not result.get("success"):
            raise RuntimeError(str(result.get("error") or "speech synthesis failed"))
        return str(result.get("file_path") or "")

    async def _emit(self, event_type: VoiceEventType, payload: Mapping[str, Any]) -> None:
        if self.config is None:
            return
        if self._closed and event_type != VoiceEventType.SESSION_CLOSED:
            return
        self._track_playback_lifecycle_event(event_type, payload)
        self._sequence += 1
        await put_realtime_voice_event(
            self._events,
            VoiceEvent(
                type=event_type,
                session_id=self.config.session_id,
                sequence=self._sequence,
                payload=dict(payload),
            )
        )
        await self._emit_caption_alias_if_needed(event_type, payload)
        await self._emit_audio_alias_if_needed(event_type, payload)

    async def _emit_caption_alias_if_needed(self, event_type: VoiceEventType, payload: Mapping[str, Any]) -> None:
        if not _caption_alias_events_enabled(self.config):
            return
        if event_type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
            alias_type = VoiceEventType.ASSISTANT_CAPTION_PARTIAL
        elif event_type == VoiceEventType.ASSISTANT_COMMIT and payload.get("interrupted") is not True:
            alias_type = VoiceEventType.ASSISTANT_CAPTION_FINAL
        else:
            return
        text = str(payload.get("text") or "").strip()
        if not text:
            return
        self._sequence += 1
        await put_realtime_voice_event(
            self._events,
            VoiceEvent(
                type=alias_type,
                session_id=self.config.session_id,
                sequence=self._sequence,
                payload={
                    "text": text,
                    "caption_alias_for": event_type.value,
                    **({"playback_generation": payload["playback_generation"]} if "playback_generation" in payload else {}),
                },
            ),
        )

    async def _emit_audio_alias_if_needed(self, event_type: VoiceEventType, payload: Mapping[str, Any]) -> None:
        if event_type != VoiceEventType.AUDIO_OUTPUT_CHUNK:
            return
        if not _audio_alias_events_enabled(self.config):
            return
        self._sequence += 1
        await put_realtime_voice_event(
            self._events,
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_AUDIO_CHUNK,
                session_id=self.config.session_id,
                sequence=self._sequence,
                payload={
                    **dict(payload),
                    "audio_alias_for": event_type.value,
                },
            ),
        )

    def _record_kame_feedback_event(self, event: VoiceEvent) -> None:
        record = {
            "type": event.type.value,
            "payload": dict(event.payload),
        }
        self._kame_feedback_events.append(record)
        if len(self._kame_feedback_events) > 64:
            self._kame_feedback_events = self._kame_feedback_events[-64:]
        generation = _payload_generation(event.payload)
        if generation is not None:
            generation_records = self._kame_feedback_events_by_generation.setdefault(generation, [])
            generation_records.append(record)
            if len(generation_records) > 32:
                self._kame_feedback_events_by_generation[generation] = generation_records[-32:]
        if event.type in {
            VoiceEventType.INTERFACE_INTENT_FINAL,
            VoiceEventType.INTERFACE_REPLY_LOCAL,
            VoiceEventType.INTERFACE_REPLY_DEFER,
            VoiceEventType.INTERFACE_ORACLE_REQUEST,
            VoiceEventType.INTERFACE_ORACLE_CANCEL,
            VoiceEventType.INTERFACE_ORACLE_UPDATE,
            VoiceEventType.INTERFACE_COMMIT,
        }:
            self._kame_last_interface_event = record
        elif event.type in {
            VoiceEventType.ORACLE_ACCEPTED,
            VoiceEventType.ORACLE_HINT,
            VoiceEventType.ORACLE_TOOL_CALL,
            VoiceEventType.ORACLE_TOOL_RESULT,
            VoiceEventType.ORACLE_RESPONSE_PARTIAL,
            VoiceEventType.ORACLE_RESPONSE_FINAL,
            VoiceEventType.ORACLE_ERROR,
        } or event.type in ORACLE_JOB_EVENT_TYPES:
            self._kame_last_oracle_event = record

    def _clear_kame_feedback_state(self) -> None:
        self._kame_feedback_events.clear()
        self._kame_feedback_events_by_generation.clear()
        self._kame_last_interface_event = None
        self._kame_last_oracle_event = None

    def _track_playback_lifecycle_event(self, event_type: VoiceEventType, payload: Mapping[str, Any]) -> None:
        generation = _payload_generation(payload)
        if event_type in {VoiceEventType.PLAYBACK_STARTED, VoiceEventType.AUDIO_OUTPUT_CHUNK}:
            self._active_playback_generations.add(generation)
            return
        if event_type in {VoiceEventType.PLAYBACK_STOPPED, VoiceEventType.ASSISTANT_AUDIO_END}:
            if generation is None:
                self._active_playback_generations.clear()
            else:
                self._active_playback_generations.discard(generation)

    def _record_speech_lifecycle_event(self, event: VoiceEvent) -> None:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        record: dict[str, Any] = {"event": event.type.value}
        for key in ("user_id", "input_generation", "rms", "duration_ms"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                record[key] = value.strip()
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                record[key] = int(value) if float(value).is_integer() else float(value)
        self._last_speech_lifecycle_event = record

    def _clear_stale_speech_lifecycle_event(self, input_generation: int) -> None:
        if not self._last_speech_lifecycle_event:
            return
        event_generation = _payload_int(self._last_speech_lifecycle_event.get("input_generation"))
        if event_generation is not None and event_generation != input_generation:
            self._last_speech_lifecycle_event = None

    def _kame_live_session_context_text(self) -> str:
        active_generations = [
            str(generation)
            for generation in sorted(
                self._active_playback_generations,
                key=lambda item: -1 if item is None else item,
            )
        ]
        parts = [
            "Live session context:",
            f"playback_active={str(bool(self._active_playback_generations)).lower()}",
            f"active_playback_generations={','.join(active_generations) if active_generations else 'none'}",
        ]
        if self._last_speech_lifecycle_event:
            speech = self._last_speech_lifecycle_event
            parts.append(f"last_speech_event={speech.get('event')}")
            for key in ("user_id", "input_generation", "rms", "duration_ms"):
                if key in speech:
                    parts.append(f"last_speech_{key}={speech[key]}")
        else:
            parts.append("last_speech_event=none")
        parts.extend(self._kame_feedback_context_parts())
        parts.append(
            "If playback_active=true, treat the new user segment as interruption-sensitive and keep any local reply brief."
        )
        return " ".join(parts)

    def _kame_feedback_context_parts(self) -> list[str]:
        parts: list[str] = []
        if self._kame_last_interface_event:
            parts.extend(_kame_feedback_record_context_parts("last_interface", self._kame_last_interface_event))
        else:
            parts.append("last_interface_event=none")
        if self._kame_last_oracle_event:
            parts.extend(_kame_feedback_record_context_parts("last_oracle", self._kame_last_oracle_event))
        else:
            parts.append("last_oracle_event=none")
        return parts

    async def _emit_shutdown_playback_finalizers(self) -> None:
        if self.config is None or not self._active_playback_generations:
            return
        active_generations = list(self._active_playback_generations)
        self._active_playback_generations.clear()
        for generation in active_generations:
            payload = {"reason": "session_closed"}
            if generation is not None:
                payload["playback_generation"] = generation
            for event_type in (VoiceEventType.PLAYBACK_STOPPED, VoiceEventType.ASSISTANT_AUDIO_END):
                self._sequence += 1
                await put_realtime_voice_event(
                    self._events,
                    VoiceEvent(
                        type=event_type,
                        session_id=self.config.session_id,
                        sequence=self._sequence,
                        payload=dict(payload),
                    )
                )

    async def _emit_interrupted_playback_finalizers(self, barge_in_payload: Mapping[str, Any]) -> None:
        if self.config is None or not self._active_playback_generations:
            return
        active_generations = list(self._active_playback_generations)
        for generation in active_generations:
            payload: dict[str, Any] = {
                "reason": "barge_in",
                "interrupted": True,
                "barge_in_reason": str(barge_in_payload.get("reason") or "client"),
            }
            if generation is not None:
                payload["playback_generation"] = generation
            await self._emit(VoiceEventType.PLAYBACK_STOPPED, payload)
            await self._emit(VoiceEventType.ASSISTANT_AUDIO_END, payload)


def create_reference_sidecar_app(runtime: Optional[ReferenceSidecarRuntimeConfig] = None):
    """Create the FastAPI app for the reference sidecar."""

    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Hermes realtime voice reference sidecar")
    runtime = runtime or runtime_config_from_env()

    @app.get("/health")
    async def health(request: Request):
        if not _authorized(request.headers, runtime.auth_token):
            raise HTTPException(status_code=401, detail="unauthorized")
        vllm_health = await _probe_vllm_health(runtime)
        streaming_stt_health = await _probe_streaming_stt_health(runtime)
        streaming_tts_health = await _probe_streaming_tts_health(runtime)
        return reference_sidecar_health_payload(
            runtime,
            vllm_health=vllm_health,
            vllm_health_checked=bool(runtime.vllm_base_url and runtime.vllm_model),
            streaming_stt_health=streaming_stt_health,
            streaming_tts_health=streaming_tts_health,
        )

    @app.websocket("/v1/realtime-text/session")
    async def realtime_text_session(ws: WebSocket):
        if not _authorized(ws.headers, runtime.auth_token):
            await ws.close(code=1008, reason="unauthorized")
            return
        await ws.accept()
        session: Optional[ReferenceRealtimeVoiceSidecarSession] = None
        pump_task: Optional[asyncio.Task[None]] = None

        async def pump_events() -> None:
            assert session is not None
            async for event in session.events():
                frame = binary_audio_frame_from_event(event)
                if frame is not None and event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
                    await ws.send_bytes(frame)
                    continue
                await ws.send_json(event.to_wire())

        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect(code=int(message.get("code") or 1000))
                frame = message.get("bytes")
                if isinstance(frame, bytes):
                    if session is None:
                        await ws.send_json({"type": "session.error", "payload": {"error": "missing session.config"}})
                        continue
                    try:
                        event = event_from_binary_audio_frame(frame, expected_type=VoiceEventType.AUDIO_INPUT_CHUNK)
                    except Exception:
                        await ws.send_json({"type": "session.error", "payload": {"error": "invalid binary audio frame"}})
                        continue
                    await session.receive_event(event)
                    continue
                raw = message.get("text")
                if not isinstance(raw, str):
                    await ws.send_json({"type": "session.error", "payload": {"error": "invalid websocket frame"}})
                    continue
                data = json.loads(raw)
                if data.get("type") == "session.config":
                    config = RealtimeVoiceSessionConfig.from_wire(data.get("payload") or {})
                    session = ReferenceRealtimeVoiceSidecarSession(runtime)
                    await session.start(config)
                    pump_task = asyncio.create_task(pump_events())
                    continue
                if session is None:
                    await ws.send_json({"type": "session.error", "payload": {"error": "missing session.config"}})
                    continue
                await session.receive_event(VoiceEvent.from_wire(data))
        except WebSocketDisconnect:
            pass
        finally:
            if pump_task:
                pump_task.cancel()
            if session is not None:
                await session.close()

    return app


def runtime_config_from_env() -> ReferenceSidecarRuntimeConfig:
    language_fallback = os.environ.get("HERMES_VOICE_LANGUAGES") or ""
    return ReferenceSidecarRuntimeConfig(
        interface_provider=os.environ.get("HERMES_KAME_INTERFACE_PROVIDER") or None,
        vllm_base_url=(
            os.environ.get("HERMES_KAME_INTERFACE_BASE_URL")
            or os.environ.get("HERMES_VOICE_VLLM_BASE_URL")
            or None
        ),
        vllm_model=(
            os.environ.get("HERMES_KAME_INTERFACE_MODEL")
            or os.environ.get("HERMES_VOICE_VLLM_MODEL")
            or None
        ),
        vllm_token=(
            os.environ.get("HERMES_KAME_INTERFACE_API_KEY")
            or os.environ.get("HERMES_KAME_INTERFACE_TOKEN")
            or os.environ.get("HERMES_VOICE_VLLM_API_KEY")
            or os.environ.get("HERMES_VOICE_VLLM_TOKEN")
            or None
        ),
        vllm_timeout_seconds=float(os.environ.get("HERMES_VOICE_VLLM_TIMEOUT_SECONDS") or 60),
        streaming_stt_provider=(
            os.environ.get("HERMES_DGX_SPARK_ASR_PROVIDER")
            or os.environ.get("HERMES_KAME_ASR_PROVIDER")
            or None
        ),
        streaming_stt_base_url=os.environ.get("HERMES_VOICE_STREAMING_STT_BASE_URL") or None,
        streaming_stt_model=os.environ.get("HERMES_VOICE_STREAMING_STT_MODEL") or None,
        streaming_stt_token=os.environ.get("HERMES_VOICE_STREAMING_STT_TOKEN") or None,
        streaming_stt_timeout_seconds=float(os.environ.get("HERMES_VOICE_STREAMING_STT_TIMEOUT_SECONDS") or 10),
        streaming_bridge_health_timeout_seconds=float(
            os.environ.get("HERMES_VOICE_STREAMING_BRIDGE_HEALTH_TIMEOUT_SECONDS") or 0.2
        ),
        streaming_tts_provider=(
            os.environ.get("HERMES_DGX_SPARK_TTS_PROVIDER")
            or os.environ.get("HERMES_KAME_TTS_PROVIDER")
            or None
        ),
        streaming_tts_base_url=os.environ.get("HERMES_VOICE_STREAMING_TTS_BASE_URL") or None,
        streaming_tts_model=os.environ.get("HERMES_VOICE_STREAMING_TTS_MODEL") or None,
        streaming_tts_token=os.environ.get("HERMES_VOICE_STREAMING_TTS_TOKEN") or None,
        streaming_tts_timeout_seconds=float(os.environ.get("HERMES_VOICE_STREAMING_TTS_TIMEOUT_SECONDS") or 10),
        openai_realtime_api_key=os.environ.get("HERMES_OPENAI_REALTIME_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or None,
        openai_realtime_base_url=os.environ.get("HERMES_OPENAI_REALTIME_BASE_URL")
        or "wss://api.openai.com/v1/realtime",
        openai_realtime_model=os.environ.get("HERMES_OPENAI_REALTIME_MODEL") or "gpt-realtime-2",
        openai_realtime_voice=os.environ.get("HERMES_OPENAI_REALTIME_VOICE") or "marin",
        openai_realtime_transcription_model=os.environ.get("HERMES_OPENAI_REALTIME_TRANSCRIPTION_MODEL")
        or "gpt-realtime-whisper",
        openai_realtime_safety_identifier=os.environ.get("HERMES_OPENAI_REALTIME_SAFETY_IDENTIFIER")
        or None,
        gemini_live_api_key=os.environ.get("HERMES_GEMINI_LIVE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or None,
        gemini_live_base_url=os.environ.get("HERMES_GEMINI_LIVE_BASE_URL")
        or (
            "wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        ),
        gemini_live_model=os.environ.get("HERMES_GEMINI_LIVE_MODEL")
        or "gemini-3.1-flash-live-preview",
        gemini_live_voice=os.environ.get("HERMES_GEMINI_LIVE_VOICE") or "Puck",
        gemini_live_google_search=_env_bool("HERMES_GEMINI_LIVE_GOOGLE_SEARCH", False),
        gemini_live_oracle_tool=_env_bool("HERMES_GEMINI_LIVE_ORACLE_TOOL", True),
        local_stt_enabled=_env_bool("HERMES_VOICE_LOCAL_STT_ENABLED", True),
        local_tts_enabled=_env_bool("HERMES_VOICE_LOCAL_TTS_ENABLED", True),
        auth_token=os.environ.get("HERMES_VOICE_SIDECAR_TOKEN")
        or os.environ.get("HERMES_SPARK_VOICE_TOKEN")
        or None,
        input_languages=tuple(
            _sanitize_metadata_list(os.environ.get("HERMES_VOICE_INPUT_LANGUAGES") or language_fallback)
        ),
        output_languages=tuple(
            _sanitize_metadata_list(os.environ.get("HERMES_VOICE_OUTPUT_LANGUAGES") or language_fallback)
        ),
        scripts=tuple(_sanitize_metadata_list(os.environ.get("HERMES_VOICE_SCRIPTS") or "")),
    )


def _authorized(headers: Mapping[str, str], token: Optional[str]) -> bool:
    if not token:
        return True
    return headers.get("authorization") == f"Bearer {token}"


def _vllm_request_headers(
    runtime: ReferenceSidecarRuntimeConfig,
    *,
    accept_json: bool = False,
    content_json: bool = False,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if accept_json:
        headers["Accept"] = "application/json"
    if content_json:
        headers["Content-Type"] = "application/json"
    if runtime.vllm_token:
        headers["Authorization"] = f"Bearer {runtime.vllm_token}"
    return headers


def _post_vllm_chat_completion(
    runtime: ReferenceSidecarRuntimeConfig,
    url: str,
    payload: Mapping[str, Any],
    *,
    timeout: float,
) -> Mapping[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=_vllm_request_headers(runtime, content_json=True),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data if isinstance(data, Mapping) else {}


def _vllm_rejected_json_schema_response_format(exc: urllib.error.HTTPError) -> bool:
    if exc.code not in {400, 422}:
        return False
    detail = _http_error_text(exc).lower()
    return "json_schema" in detail and (
        "response_format" in detail
        or "unsupported" in detail
        or "not supported" in detail
        or "schema" in detail
    )


def _http_error_text(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read()
    except Exception:
        body = b""
    try:
        body_text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)
    except Exception:
        body_text = ""
    return f"{exc.reason or ''} {body_text}".strip()


def _format_http_error(exc: urllib.error.HTTPError) -> str:
    detail = _http_error_text(exc)
    if detail:
        return f"HTTP {exc.code}: {detail}"
    return f"HTTP {exc.code}: {exc.reason or 'request failed'}"


async def _probe_vllm_health(runtime: ReferenceSidecarRuntimeConfig) -> Optional[Mapping[str, Any]]:
    if not runtime.vllm_base_url or not runtime.vllm_model:
        return None
    return await asyncio.to_thread(_probe_vllm_health_sync, runtime)


async def _probe_streaming_stt_health(runtime: ReferenceSidecarRuntimeConfig) -> Optional[Mapping[str, Any]]:
    if not runtime.streaming_stt_base_url:
        return None
    return await asyncio.to_thread(_probe_streaming_stt_health_sync, runtime)


async def _probe_streaming_tts_health(runtime: ReferenceSidecarRuntimeConfig) -> Optional[Mapping[str, Any]]:
    if not runtime.streaming_tts_base_url:
        return None
    return await asyncio.to_thread(_probe_streaming_tts_health_sync, runtime)


def _probe_streaming_stt_health_sync(runtime: ReferenceSidecarRuntimeConfig) -> Optional[Mapping[str, Any]]:
    if not runtime.streaming_stt_base_url:
        return None
    url = f"{runtime.streaming_stt_base_url.rstrip('/')}/health"
    headers = {}
    if runtime.streaming_stt_token:
        headers["Authorization"] = f"Bearer {runtime.streaming_stt_token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(
            request,
            timeout=runtime.streaming_bridge_health_timeout_seconds,
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, Mapping) else None


def _probe_streaming_tts_health_sync(runtime: ReferenceSidecarRuntimeConfig) -> Optional[Mapping[str, Any]]:
    if not runtime.streaming_tts_base_url:
        return None
    url = f"{runtime.streaming_tts_base_url.rstrip('/')}/health"
    headers = {}
    if runtime.streaming_tts_token:
        headers["Authorization"] = f"Bearer {runtime.streaming_tts_token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(
            request,
            timeout=runtime.streaming_bridge_health_timeout_seconds,
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, Mapping) else None


def _probe_vllm_health_sync(runtime: ReferenceSidecarRuntimeConfig) -> Optional[Mapping[str, Any]]:
    if not runtime.vllm_base_url or not runtime.vllm_model:
        return None
    url = f"{runtime.vllm_base_url.rstrip('/')}/models"
    request = urllib.request.Request(url, headers=_vllm_request_headers(runtime, accept_json=True), method="GET")
    try:
        with urllib.request.urlopen(
            request,
            timeout=min(max(0.1, float(runtime.vllm_timeout_seconds or 1.0)), 2.0),
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, Mapping) else None


def _health_supports_vllm_model(health: Optional[Mapping[str, Any]], model: Optional[str]) -> bool:
    if not isinstance(health, Mapping):
        return False
    expected = str(model or "").strip()
    data = health.get("data")
    if not isinstance(data, list):
        return False
    for item in data:
        if not isinstance(item, Mapping):
            continue
        model_id = str(item.get("id") or item.get("root") or "").strip()
        if expected and model_id == expected:
            return True
    return False


def _health_supports_streaming_stt(health: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(health, Mapping) or health.get("ok") is not True:
        return False
    capabilities = health.get("capabilities")
    if not isinstance(capabilities, Mapping):
        return False
    return capabilities.get("streaming_stt") is True


def _health_supports_tts(health: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(health, Mapping) or health.get("ok") is not True:
        return False
    capabilities = health.get("capabilities")
    if not isinstance(capabilities, Mapping):
        return False
    return capabilities.get("tts") is True


def _streaming_tts_health_output_languages(health: Optional[Mapping[str, Any]]) -> list[str]:
    if not isinstance(health, Mapping) or health.get("ok") is not True:
        return []
    capabilities = health.get("capabilities")
    if not isinstance(capabilities, Mapping):
        return []
    return _sanitize_metadata_list(
        capabilities.get("output_languages", capabilities.get("tts_languages"))
    )


def _streaming_tts_health_model_languages(health: Optional[Mapping[str, Any]]) -> list[str]:
    if not isinstance(health, Mapping) or health.get("ok") is not True:
        return []
    frontend = health.get("frontend")
    if not isinstance(frontend, Mapping):
        return []
    return _sanitize_metadata_list(frontend.get("tts_model_languages"))


def _numeric_transcript_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = {}
    for key in ("confidence", "stability"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            fields[key] = value
    return fields


def _generation_transcript_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = {}
    for key in ("input_generation", "playback_generation"):
        value = _payload_int(payload.get(key))
        if value is not None:
            fields[key] = value
    return fields


def _write_temp_audio(
    audio: bytes,
    codec: VoiceAudioCodec,
    config: Optional[RealtimeVoiceSessionConfig] = None,
) -> str:
    suffix = {
        VoiceAudioCodec.PCM16: ".wav",
        VoiceAudioCodec.OPUS: ".ogg",
        VoiceAudioCodec.WEBM_OPUS: ".webm",
    }.get(codec, ".webm")
    with tempfile.NamedTemporaryFile(prefix="hermes-voice-sidecar-", suffix=suffix, delete=False) as tmp:
        tmp.write(_audio_bytes_for_codec(audio, codec, config))
        return tmp.name


def _audio_bytes_for_codec(
    audio: bytes,
    codec: VoiceAudioCodec,
    config: Optional[RealtimeVoiceSessionConfig] = None,
) -> bytes:
    if codec == VoiceAudioCodec.PCM16:
        return _pcm16_wav_bytes(audio, config)
    return audio


def _pcm16_wav_bytes(audio: bytes, config: Optional[RealtimeVoiceSessionConfig]) -> bytes:
    sample_rate_hz = getattr(config, "sample_rate_hz", 16000) if config is not None else 16000
    channels = getattr(config, "channels", 1) if config is not None else 1
    try:
        sample_rate = max(1, int(sample_rate_hz))
    except (TypeError, ValueError):
        sample_rate = 16000
    try:
        channel_count = max(1, int(channels))
    except (TypeError, ValueError):
        channel_count = 1
    bits_per_sample = 16
    block_align = channel_count * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    data_size = len(audio)
    riff_size = 36 + data_size
    return b"".join(
        (
            b"RIFF",
            riff_size.to_bytes(4, "little", signed=False),
            b"WAVE",
            b"fmt ",
            (16).to_bytes(4, "little", signed=False),
            (1).to_bytes(2, "little", signed=False),
            channel_count.to_bytes(2, "little", signed=False),
            sample_rate.to_bytes(4, "little", signed=False),
            byte_rate.to_bytes(4, "little", signed=False),
            block_align.to_bytes(2, "little", signed=False),
            bits_per_sample.to_bytes(2, "little", signed=False),
            b"data",
            data_size.to_bytes(4, "little", signed=False),
            audio,
        )
    )


def _mime_type_for_codec(codec: VoiceAudioCodec) -> str:
    return {
        VoiceAudioCodec.PCM16: "audio/wav",
        VoiceAudioCodec.OPUS: "audio/ogg",
        VoiceAudioCodec.WEBM_OPUS: "audio/webm",
    }.get(codec, "audio/webm")


def _mime_type_for_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
    }.get(ext, "audio/mpeg")


def _audio_file_to_pcm16_chunk(path: str, data: Optional[bytes] = None) -> AudioChunk:
    """Decode a synthesized audio file into raw PCM16 for realtime playback."""

    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        try:
            with wave.open(path, "rb") as wav:
                if wav.getcomptype() != "NONE":
                    raise RuntimeError(f"unsupported WAV compression: {wav.getcomptype()}")
                if wav.getsampwidth() != 2:
                    raise RuntimeError(f"unsupported WAV sample width: {wav.getsampwidth()}")
                channels = int(wav.getnchannels())
                sample_rate = int(wav.getframerate())
                pcm = wav.readframes(wav.getnframes())
            if channels <= 0 or sample_rate <= 0:
                raise RuntimeError("invalid WAV audio geometry")
            return AudioChunk(
                codec=VoiceAudioCodec.PCM16,
                data=pcm,
                sample_rate_hz=sample_rate,
                channels=channels,
            )
        except wave.Error as exc:
            raise RuntimeError(f"invalid WAV audio: {exc}") from exc

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to decode non-WAV TTS audio to PCM16")
    source = path if path else "pipe:0"
    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            source,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "24000",
            "pipe:1",
        ],
        input=data if source == "pipe:0" else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        error = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(error or "ffmpeg failed to decode TTS audio to PCM16")
    if not proc.stdout:
        raise RuntimeError("ffmpeg produced no PCM audio")
    return AudioChunk(
        codec=VoiceAudioCodec.PCM16,
        data=proc.stdout,
        sample_rate_hz=24000,
        channels=1,
    )


def _kame_reflex_payload_from_content(
    content: str,
    *,
    config: Optional[RealtimeVoiceSessionConfig] = None,
) -> dict[str, Any]:
    text = str(content or "").strip()
    if not text:
        return {"text": ""}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {
            "text": text,
            "intent": text,
            "intent_source": "reflex_audio",
            "transcript_source": "none",
            "route": KameRoute.ORACLE_DIRECT.value,
            "reflex_validation_error": "invalid_json",
        }
    if not isinstance(parsed, Mapping):
        return {
            "text": text,
            "intent": text,
            "intent_source": "reflex_audio",
            "transcript_source": "none",
            "route": KameRoute.ORACLE_DIRECT.value,
            "reflex_validation_error": "invalid_json_shape",
        }
    payload = KameReflexDecision.from_payload(parsed, fallback_text=text).to_payload()
    return _apply_kame_routing_policy(payload, config)


def _kame_payload_accepts_oracle_asr_evidence(payload: Mapping[str, Any]) -> bool:
    route = str(payload.get("route") or KameRoute.ORACLE_DIRECT.value).strip().lower()
    return route in {KameRoute.DEFER.value, KameRoute.ORACLE_DIRECT.value}


def _final_understanding_event_type(config: Optional[RealtimeVoiceSessionConfig]) -> VoiceEventType:
    if config is not None and config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
        return VoiceEventType.INTERFACE_INTENT_FINAL
    return VoiceEventType.TRANSCRIPT_FINAL


def _apply_kame_routing_policy(
    payload: Mapping[str, Any],
    config: Optional[RealtimeVoiceSessionConfig],
) -> dict[str, Any]:
    return apply_kame_routing_policy(payload, _kame_routing_policy(config))


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
    input_generation = _payload_input_generation(payload)
    if input_generation is not None:
        partial["input_generation"] = input_generation
    return partial


def _bounded_confidence(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, parsed))


def _allow_kame_transcript_events(config: Optional[RealtimeVoiceSessionConfig]) -> bool:
    if config is None or config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
        return True
    if config.asr_mode.value in {"debug", "fallback", "from_reflex"}:
        return True
    return str(config.interface_audio_input or "").strip().lower() == "text_fallback"


def _caption_alias_events_enabled(config: Optional[RealtimeVoiceSessionConfig]) -> bool:
    if config is None:
        return False
    if isinstance(config.output_events, Mapping) and _metadata_bool(config.output_events.get("caption_aliases"), default=False):
        return True
    metadata = config.metadata if isinstance(config.metadata, Mapping) else {}
    output_events = metadata.get("output_events") if isinstance(metadata, Mapping) else {}
    return isinstance(output_events, Mapping) and _metadata_bool(output_events.get("caption_aliases"), default=False)


def _audio_alias_events_enabled(config: Optional[RealtimeVoiceSessionConfig]) -> bool:
    if config is None:
        return False
    if isinstance(config.output_events, Mapping):
        return _metadata_bool(config.output_events.get("audio_aliases"), default=False)
    metadata = config.metadata if isinstance(config.metadata, Mapping) else {}
    output_events = metadata.get("output_events") if isinstance(metadata, Mapping) else {}
    return isinstance(output_events, Mapping) and _metadata_bool(output_events.get("audio_aliases"), default=False)


def _assistant_speak_metadata_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = dict(transcript_metadata_from_payload(payload))
    for key in (
        "voice_architecture",
        "kame_route",
        "kame_interface_already_said",
        "intent_source",
        "transcript_source",
    ):
        value = payload.get(key)
        if not isinstance(value, str):
            continue
        token = value.strip()
        if token:
            metadata[key] = token
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        clean_metrics: dict[str, int] = {}
        for key, value in metrics.items():
            if isinstance(value, bool):
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed >= 0:
                clean_metrics[str(key)] = parsed
        if clean_metrics:
            metadata["metrics"] = clean_metrics
    return metadata


def _tts_synthesis_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key in ("language", "locale", "script"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            clean[key] = value
    return clean


def _call_synthesize(synthesize: SynthesizeFn, text: str, metadata: Mapping[str, Any]) -> Any:
    if metadata and _synthesize_accepts_metadata(synthesize):
        return synthesize(text, metadata=metadata)
    return synthesize(text)


def _synthesize_accepts_metadata(synthesize: SynthesizeFn) -> bool:
    try:
        signature = inspect.signature(synthesize)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == "metadata"
        for parameter in signature.parameters.values()
    )


def _payload_generation(payload: Mapping[str, Any]) -> Optional[int]:
    value = payload.get("playback_generation")
    return _payload_int(value)


def _kame_playback_start_metrics(
    metadata: Mapping[str, Any],
    first_tts_audio_at: float,
    playback_started_at: float,
) -> dict[str, int]:
    if str(metadata.get("voice_architecture") or "") != "kame_frontend_oracle":
        return {}
    playback_start_ms = max(
        0,
        int(round((playback_started_at - first_tts_audio_at) * 1000)),
    )
    metrics = {"kame_first_tts_audio_to_playback_start_ms": playback_start_ms}
    existing_metrics = metadata.get("metrics")
    if isinstance(existing_metrics, Mapping):
        speech_to_first_audio_ms = existing_metrics.get("kame_speech_end_to_first_audio_ms")
        if isinstance(speech_to_first_audio_ms, int) and not isinstance(speech_to_first_audio_ms, bool):
            metrics["kame_speech_end_to_playback_start_ms"] = max(
                0,
                speech_to_first_audio_ms + playback_start_ms,
            )
    return metrics


def _playback_lifecycle_payload(
    playback_generation: Optional[int],
    metadata: Mapping[str, Any],
    metrics: Mapping[str, int],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if playback_generation is not None:
        payload["playback_generation"] = playback_generation
    if not metrics:
        return payload
    metadata_metrics = metadata.get("metrics")
    merged_metrics = dict(metadata_metrics) if isinstance(metadata_metrics, Mapping) else {}
    merged_metrics.update(metrics)
    payload["metrics"] = merged_metrics
    return payload


def _payload_input_generation(payload: Mapping[str, Any]) -> Optional[int]:
    value = payload.get("input_generation")
    return _payload_int(value)


def _interface_audio_input(config: Optional[RealtimeVoiceSessionConfig]) -> str:
    if config is None:
        return "auto"
    return normalize_realtime_voice_interface_audio_input(config.interface_audio_input) or "auto"


def _reported_frontend_model(
    provider: str,
    *,
    implementation_provider: Optional[str] = None,
    runtime: ReferenceSidecarRuntimeConfig,
    config: RealtimeVoiceSessionConfig,
) -> str:
    effective_provider = implementation_provider or provider
    if effective_provider == "openai_realtime":
        return runtime.openai_realtime_model
    if effective_provider == "gemini_live":
        return runtime.gemini_live_model
    if effective_provider == "vllm":
        return runtime.vllm_model or config.frontend_model or ""
    if effective_provider == "streaming_stt":
        return runtime.streaming_stt_model or config.frontend_model or ""
    if effective_provider == "local_stt":
        return config.frontend_model or runtime.streaming_stt_model or ""
    return config.frontend_model or runtime.vllm_model or runtime.streaming_stt_model or ""


def _kame_routing_policy_text(config: Optional[RealtimeVoiceSessionConfig]) -> str:
    routing: Any = config.routing_policy if config is not None else {}
    if not routing:
        metadata = config.metadata if config is not None and isinstance(config.metadata, Mapping) else {}
        routing = metadata.get("routing") if isinstance(metadata, Mapping) else {}
    if not isinstance(routing, Mapping):
        routing = {}
    return (
        f"allow_local_greetings={_metadata_bool(routing.get('allow_local_greetings'), default=True)}, "
        f"allow_local_clarifications={_metadata_bool(routing.get('allow_local_clarifications'), default=True)}, "
        f"require_oracle_for_tools={_metadata_bool(routing.get('require_oracle_for_tools'), default=True)}, "
        f"require_oracle_for_memory={_metadata_bool(routing.get('require_oracle_for_memory'), default=True)}, "
        f"require_oracle_for_files={_metadata_bool(routing.get('require_oracle_for_files'), default=True)}, "
        f"local_confidence_threshold={_metadata_float(routing.get('local_confidence_threshold'), default=0.75):.2f}."
    )


def _interface_temperature(config: Optional[RealtimeVoiceSessionConfig]) -> float:
    value = getattr(config, "interface_temperature", 0.2) if config is not None else 0.2
    if isinstance(value, bool):
        return 0.2
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.2
    return max(0.0, min(2.0, parsed))


def _interface_max_output_tokens(config: Optional[RealtimeVoiceSessionConfig]) -> int:
    value = getattr(config, "interface_max_output_tokens", 160) if config is not None else 160
    if isinstance(value, bool):
        return 160
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 160
    return max(1, min(4096, parsed))


def _interface_timeout_seconds(
    config: Optional[RealtimeVoiceSessionConfig],
    runtime_timeout_seconds: float,
) -> float:
    value = getattr(config, "interface_timeout_seconds", 0.8) if config is not None else 0.8
    if isinstance(value, bool):
        value = 0.8
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.8
    if parsed <= 0:
        parsed = 0.8
    try:
        runtime_timeout = float(runtime_timeout_seconds)
    except (TypeError, ValueError):
        runtime_timeout = parsed
    if runtime_timeout > 0:
        return min(parsed, runtime_timeout)
    return parsed


def _interface_max_audio_seconds(config: Optional[RealtimeVoiceSessionConfig]) -> float:
    value = getattr(config, "interface_max_audio_seconds", 30.0) if config is not None else 30.0
    if isinstance(value, bool):
        return 30.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 30.0
    if parsed <= 0:
        return 30.0
    return min(parsed, 30.0)


def _audio_duration_seconds(
    audio: bytes,
    codec: VoiceAudioCodec,
    config: Optional[RealtimeVoiceSessionConfig],
) -> Optional[float]:
    if codec != VoiceAudioCodec.PCM16:
        return None
    if not audio:
        return 0.0
    sample_rate_hz = getattr(config, "sample_rate_hz", 16000) if config is not None else 16000
    channels = getattr(config, "channels", 1) if config is not None else 1
    try:
        sample_rate = int(sample_rate_hz)
        channel_count = int(channels)
    except (TypeError, ValueError):
        return None
    bytes_per_second = sample_rate * max(1, channel_count) * 2
    if bytes_per_second <= 0:
        return None
    return len(audio) / bytes_per_second


def _kame_provider_metrics_enabled(config: Optional[RealtimeVoiceSessionConfig]) -> bool:
    if config is None or config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
        return False
    if isinstance(config.metrics_policy, Mapping) and config.metrics_policy:
        if not _metadata_bool(config.metrics_policy.get("enabled"), default=True):
            return False
        return _metadata_bool(config.metrics_policy.get("log_provider_spans"), default=True)
    metadata = config.metadata if isinstance(config.metadata, Mapping) else {}
    metrics = metadata.get("metrics") if isinstance(metadata, Mapping) else {}
    if not isinstance(metrics, Mapping):
        return True
    if not _metadata_bool(metrics.get("enabled"), default=True):
        return False
    return _metadata_bool(metrics.get("log_provider_spans"), default=True)


def _metadata_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _metadata_float(value: Any, *, default: float) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def _payload_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _kame_feedback_record_context_parts(prefix: str, record: Mapping[str, Any]) -> list[str]:
    event_type = _kame_feedback_prompt_value(record.get("type"), limit=96)
    parts = [f"{prefix}_event={event_type}" if event_type else f"{prefix}_event=unknown"]
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        return parts
    for key in (
        "turn_id",
        "route",
        "text",
        "delta",
        "final",
        "accepted",
        "tool_name",
        "tool_call_id",
        "reason",
        "error",
        "outcome",
    ):
        if key not in payload:
            continue
        value = _kame_feedback_prompt_value(
            payload.get(key),
            limit=240 if key in {"text", "delta", "error", "outcome"} else 96,
        )
        if value:
            parts.append(f"{prefix}_{key}={value}")
    return parts


def _kame_feedback_prompt_value(value: Any, *, limit: int) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        return str(int(value)) if float(value).is_integer() else str(value)
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    if len(text) > limit:
        text = text[: max(0, limit - 1)].rstrip() + "..."
    return json.dumps(text, ensure_ascii=True)


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_HEALTH_METADATA_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _provider_label(value: Any, *, default: str) -> str:
    text = str(value or "").strip()
    if not text or not _HEALTH_METADATA_RE.fullmatch(text):
        return default
    return text


def _sanitize_metadata_list(value: Any, *, limit: int = 32) -> list[str]:
    if isinstance(value, str):
        candidates = re.split(r"[\s,]+", value)
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        return []
    return _dedupe_metadata(str(candidate).strip() for candidate in candidates if isinstance(candidate, str))[:limit]


def _dedupe_metadata(values) -> list[str]:
    result: list[str] = []
    seen = set()
    for value in values:
        if not value or not _HEALTH_METADATA_RE.fullmatch(value):
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
