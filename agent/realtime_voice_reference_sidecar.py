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
import tempfile
import time
import urllib.request
from dataclasses import dataclass
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
    put_realtime_voice_event,
    transcript_metadata_from_payload,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_sidecar import RealtimeVoiceSidecarClient


TranscribeFn = Callable[[str], Mapping[str, Any]]
SynthesizeFn = Callable[..., Any]
REFERENCE_SIDECAR_CLOSE_DRAIN_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True)
class ReferenceSidecarRuntimeConfig:
    """Runtime knobs for the reference sidecar process."""

    vllm_base_url: Optional[str] = None
    vllm_model: Optional[str] = None
    vllm_timeout_seconds: float = 60.0
    streaming_stt_base_url: Optional[str] = None
    streaming_stt_model: Optional[str] = None
    streaming_stt_token: Optional[str] = None
    streaming_stt_timeout_seconds: float = 10.0
    streaming_bridge_health_timeout_seconds: float = 0.2
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


def reference_sidecar_health_payload(
    runtime: ReferenceSidecarRuntimeConfig,
    *,
    streaming_stt_health: Optional[Mapping[str, Any]] = None,
    streaming_tts_health: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    vllm_enabled = bool(runtime.vllm_base_url and runtime.vllm_model)
    streaming_stt_configured = bool(runtime.streaming_stt_base_url)
    streaming_stt_ready = _health_supports_streaming_stt(streaming_stt_health)
    streaming_tts_configured = bool(runtime.streaming_tts_base_url)
    streaming_tts_ready = _health_supports_tts(streaming_tts_health)
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

    payload = {
        "ok": True,
        "kind": "reference",
        "frontend": {
            "provider": (
                "openai_realtime"
                if openai_realtime_configured
                else "gemini_live"
                if gemini_live_configured
                else "streaming_stt" if streaming_stt_ready else "vllm" if vllm_enabled else "local"
            ),
            "model": (
                runtime.openai_realtime_model
                if openai_realtime_configured
                else runtime.gemini_live_model
                if gemini_live_configured
                else (runtime.streaming_stt_model or "") if streaming_stt_ready else runtime.vllm_model or ""
            ),
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
    if streaming_tts_configured:
        payload["capabilities"]["streaming_tts_bridge"] = True
        payload["frontend"]["streaming_tts_bridge"] = {
            "configured": True,
            "healthy": streaming_tts_ready,
            "model": runtime.streaming_tts_model or "",
        }
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
        self._sequence = 0
        self._closed = False
        self._active_tasks: set[asyncio.Task[None]] = set()
        self._streaming_stt: Optional[RealtimeVoiceSidecarClient] = None
        self._streaming_stt_task: Optional[asyncio.Task[None]] = None
        self._asr_hypotheses_by_generation: dict[int, dict[str, Any]] = {}
        self._streaming_tts: Optional[RealtimeVoiceSidecarClient] = None
        self._streaming_tts_task: Optional[asyncio.Task[None]] = None
        self._openai_realtime: Any = None
        self._openai_realtime_task: Optional[asyncio.Task[None]] = None
        self._gemini_live: Any = None
        self._gemini_live_task: Optional[asyncio.Task[None]] = None
        self._cached_acknowledgement_audio: Optional[dict[str, Any]] = None

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        self.config = config
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
        await self._prepare_acknowledgement_audio(config)
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "ready",
                "provider": (
                    "openai_realtime"
                    if self._openai_realtime is not None
                    else "gemini_live" if self._gemini_live is not None
                    else "streaming_stt" if self._streaming_stt is not None
                    else config.frontend_provider if requested_provider not in {"openai_realtime", "openai", "gemini_live", "gemini"} and config.frontend_provider
                    else "local"
                ),
                "model": self.runtime.streaming_stt_model or config.frontend_model or "",
                "streaming_stt": self._streaming_stt is not None,
                "streaming_tts": self._streaming_tts is not None,
                "vllm": bool(self.runtime.vllm_base_url and self.runtime.vllm_model),
                "local_stt": self.runtime.local_stt_enabled,
                "local_tts": self.runtime.local_tts_enabled,
                "asr_mode": config.asr_mode.value,
                "interface_audio_input": config.interface_audio_input or "",
            },
        )

    async def receive_event(self, event: VoiceEvent) -> None:
        if self._closed:
            return
        if event.type == VoiceEventType.SESSION_CLOSED:
            await self.close()
            return
        if event.type == VoiceEventType.BARGE_IN:
            self._clear_audio_buffer()
            cancelled_tasks = self._cancel_active_tasks()
            payload = {"reason": event.payload.get("reason") or "client"}
            playback_generation = _payload_generation(event.payload)
            if playback_generation is not None:
                payload["playback_generation"] = playback_generation
            await self._emit(VoiceEventType.BARGE_IN, payload)
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
                        self._speak(
                            text,
                            _payload_generation(event.payload),
                            transcript_metadata_from_payload(event.payload),
                        )
                    )
                )
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
            self._clear_audio_buffer()
            if audio:
                payload = {"text": "", "stability": 0.1}
                if audio_input_generation is not None:
                    payload["input_generation"] = audio_input_generation
                await self._emit(VoiceEventType.TRANSCRIPT_PARTIAL, payload)
                self._track_task(asyncio.create_task(self._transcribe(audio, chunk.codec, audio_input_generation)))

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
        if self._streaming_stt_task and not self._streaming_stt_task.done():
            self._streaming_stt_task.cancel()
        if self._streaming_tts_task and not self._streaming_tts_task.done():
            self._streaming_tts_task.cancel()
        if self._openai_realtime_task and not self._openai_realtime_task.done():
            self._openai_realtime_task.cancel()
        if self._gemini_live_task and not self._gemini_live_task.done():
            self._gemini_live_task.cancel()
        if self._streaming_stt is not None:
            with contextlib.suppress(Exception):
                await self._streaming_stt.close()
        if self._streaming_tts is not None:
            with contextlib.suppress(Exception):
                await self._streaming_tts.close()
        if self._openai_realtime is not None:
            with contextlib.suppress(Exception):
                await self._openai_realtime.close()
        if self._gemini_live is not None:
            with contextlib.suppress(Exception):
                await self._gemini_live.close()
        if self._streaming_stt_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._streaming_stt_task
        if self._streaming_tts_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._streaming_tts_task
        if self._openai_realtime_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._openai_realtime_task
        if self._gemini_live_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._gemini_live_task
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
            preferred_local_oracle_model=config.preferred_local_oracle_model,
            oracle_model=config.oracle_model,
            tts_provider=config.tts_provider,
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
            preferred_local_oracle_model=config.preferred_local_oracle_model,
            oracle_model=config.oracle_model,
            tts_provider=config.tts_provider,
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

    async def _consume_streaming_stt_events(self) -> None:
        if self._streaming_stt is None:
            return
        try:
            async for event in self._streaming_stt.events():
                if event.type in {VoiceEventType.TRANSCRIPT_PARTIAL, VoiceEventType.TRANSCRIPT_FINAL}:
                    if self._suppress_streaming_stt_transcript_events():
                        self._record_asr_hypothesis(event)
                        continue
                    await self._emit(event.type, transcript_metadata_from_payload(event.payload) | {
                        "text": str(event.payload.get("text") or ""),
                        **_numeric_transcript_fields(event.payload),
                        **_generation_transcript_fields(event.payload),
                    })
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
                if event.type in {
                    VoiceEventType.TRANSCRIPT_PARTIAL,
                    VoiceEventType.TRANSCRIPT_FINAL,
                    VoiceEventType.AUDIO_OUTPUT_CHUNK,
                    VoiceEventType.FRONTEND_STATE,
                    VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                    VoiceEventType.ASSISTANT_COMMIT,
                    VoiceEventType.BARGE_IN,
                }:
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
                if event.type in {
                    VoiceEventType.TRANSCRIPT_PARTIAL,
                    VoiceEventType.TRANSCRIPT_FINAL,
                    VoiceEventType.AUDIO_OUTPUT_CHUNK,
                    VoiceEventType.FRONTEND_STATE,
                    VoiceEventType.ORACLE_HINT,
                    VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                    VoiceEventType.ASSISTANT_COMMIT,
                    VoiceEventType.BARGE_IN,
                    VoiceEventType.TOOL_PENDING,
                    VoiceEventType.TOOL_RESULT,
                }:
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
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "degraded",
                "reason": reason,
                "error": sanitize_realtime_voice_error(error),
                "streaming_tts": False,
            },
        )

    async def _disable_openai_realtime(self, reason: str, error: Any) -> None:
        provider = self._openai_realtime
        self._openai_realtime = None
        if provider is not None:
            with contextlib.suppress(Exception):
                await provider.close()
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
        if provider is not None:
            with contextlib.suppress(Exception):
                await provider.close()
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
        return True

    def _clear_audio_buffer(self) -> None:
        self._audio.clear()
        self._audio_bytes = 0
        self._audio_input_generation = None

    def _should_start_streaming_stt(self, config: RealtimeVoiceSessionConfig) -> bool:
        if config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
            return True
        return config.asr_mode.value in {"fallback", "debug", "speculative"} or (
            str(config.interface_audio_input or "") == "text_fallback"
        )

    def _streaming_stt_drives_reflex(self) -> bool:
        config = self.config
        if config is None or config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
            return True
        return config.asr_mode.value == "fallback" or str(config.interface_audio_input or "") == "text_fallback"

    def _suppress_streaming_stt_transcript_events(self) -> bool:
        config = self.config
        if config is None or config.engine != RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
            return False
        return not self._streaming_stt_drives_reflex()

    def _record_asr_hypothesis(self, event: VoiceEvent) -> None:
        if event.type != VoiceEventType.TRANSCRIPT_FINAL:
            return
        generation = _payload_input_generation(event.payload)
        if generation is None:
            return
        text = str(event.payload.get("text") or "").strip()
        if not text:
            return
        hypothesis: dict[str, Any] = {"transcript": text, "transcript_source": "asr"}
        confidence = _bounded_confidence(event.payload.get("confidence"))
        if confidence is not None:
            hypothesis["transcript_confidence"] = confidence
        self._asr_hypotheses_by_generation[generation] = hypothesis

    async def _transcribe(
        self,
        audio: bytes,
        codec: VoiceAudioCodec,
        input_generation: Optional[int] = None,
    ) -> None:
        try:
            payload = await asyncio.to_thread(self._understand_audio_sync, audio, codec)
            text = str(payload.get("text") or "").strip()
            if text:
                if input_generation is not None:
                    asr_hypothesis = self._asr_hypotheses_by_generation.pop(input_generation, None)
                    if asr_hypothesis and self.config is not None and self.config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE:
                        payload.setdefault("transcript", asr_hypothesis.get("transcript"))
                        payload.setdefault("transcript_source", asr_hypothesis.get("transcript_source"))
                        if "transcript_confidence" in asr_hypothesis:
                            payload.setdefault("transcript_confidence", asr_hypothesis["transcript_confidence"])
                    payload["input_generation"] = input_generation
                await self._emit(VoiceEventType.TRANSCRIPT_FINAL, payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"transcription failed: {sanitize_realtime_voice_error(exc)}"},
            )

    def _transcribe_sync(self, audio: bytes, codec: VoiceAudioCodec) -> str:
        return str(self._understand_audio_sync(audio, codec).get("text") or "").strip()

    def _understand_audio_sync(self, audio: bytes, codec: VoiceAudioCodec) -> dict[str, Any]:
        if self._wants_kame_vllm_reflex():
            return self._understand_kame_with_vllm(audio, codec)
        if self.runtime.vllm_base_url and self.runtime.vllm_model:
            return {"text": self._transcribe_with_vllm(audio, codec)}
        if not self.runtime.local_stt_enabled:
            raise RuntimeError("local STT is disabled and no vLLM audio frontend is configured")

        transcribe_audio = self._transcribe_audio_func
        if transcribe_audio is None:
            from tools.transcription_tools import transcribe_audio as transcribe_audio

        path = _write_temp_audio(audio, codec)
        try:
            result = transcribe_audio(path)
            if not result.get("success"):
                raise RuntimeError(str(result.get("error") or "transcription failed"))
            return {"text": str(result.get("transcript") or "").strip()}
        finally:
            _unlink(path)

    def _wants_kame_vllm_reflex(self) -> bool:
        config = self.config
        return (
            config is not None
            and config.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE
            and bool(self.runtime.vllm_base_url and self.runtime.vllm_model)
            and str(config.interface_audio_input or "auto") != "text_fallback"
        )

    def _transcribe_with_vllm(self, audio: bytes, codec: VoiceAudioCodec) -> str:
        mime_type = _mime_type_for_codec(codec)
        audio_b64 = base64.b64encode(audio).decode("ascii")
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
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.runtime.vllm_timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data["choices"][0]["message"].get("content") or "").strip()

    def _understand_kame_with_vllm(self, audio: bytes, codec: VoiceAudioCodec) -> dict[str, Any]:
        mime_type = _mime_type_for_codec(codec)
        audio_b64 = base64.b64encode(audio).decode("ascii")
        config = self.config
        asr_mode = str(config.asr_mode.value if config is not None else "on_escalation")
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
                                "You are the low-latency KAME reflex for a Hermes realtime voice session. "
                                "Listen to the audio segment and return only a compact JSON object. "
                                "Required keys: route, intent, text. route must be one of local, defer, "
                                "oracle_direct, or reject_or_clarify. text should equal the best "
                                "oracle-facing user wording. For local or reject_or_clarify, include "
                                "local_reply with the exact short phrase to speak. Optional keys: "
                                "transcript, transcript_confidence. "
                                "Use intent for what the user wants. Use transcript only as a verbatim "
                                "hypothesis for names, numbers, code identifiers, and tool arguments. "
                                "Only use local for greetings, repeats, can-you-hear-me checks, or low-risk "
                                "conversational glue. Use oracle_direct for tools, files, memory, projects, "
                                "or any nontrivial answer. "
                                f"ASR evidence mode is {asr_mode}; do not rely on external tools. "
                                "Do not add markdown or commentary."
                            ),
                        },
                    ],
                }
            ],
            "max_tokens": 256,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        url = f"{self.runtime.vllm_base_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.runtime.vllm_timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = str(data["choices"][0]["message"].get("content") or "").strip()
        return _kame_reflex_payload_from_content(content)

    async def _speak(
        self,
        text: str,
        playback_generation: Optional[int] = None,
        metadata: Optional[Mapping[str, str]] = None,
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
            return
        if await self._emit_cached_acknowledgement_audio(text, playback_generation, clean_metadata):
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
                    payload = AudioChunk(codec=VoiceAudioCodec.OPUS, data=data).to_payload()
                    payload["mime_type"] = _mime_type_for_path(file_path)
                    if playback_generation is not None:
                        payload["playback_generation"] = playback_generation
                    payload.update(clean_metadata)
                    existing_metrics = payload.get("metrics")
                    metrics = dict(existing_metrics) if isinstance(existing_metrics, dict) else {}
                    metrics["tts_synthesis_ms"] = tts_synthesis_ms
                    payload["metrics"] = metrics
                    await self._emit(VoiceEventType.AUDIO_OUTPUT_CHUNK, payload)
            finally:
                _unlink(file_path)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                VoiceEventType.SESSION_ERROR,
                {"error": f"tts failed: {sanitize_realtime_voice_error(exc)}"},
            )

    async def _prepare_acknowledgement_audio(self, config: RealtimeVoiceSessionConfig) -> None:
        if self._streaming_tts is not None or not self.runtime.local_tts_enabled:
            return
        text = _turn_acknowledgement_text(config)
        if not text:
            return
        try:
            tts_started_at = time.perf_counter()
            file_path = await asyncio.to_thread(self._speak_sync, text, {})
            tts_synthesis_ms = int(round((time.perf_counter() - tts_started_at) * 1000))
            if not file_path:
                return
            try:
                with open(file_path, "rb") as fh:
                    data = fh.read()
            finally:
                _unlink(file_path)
            if not data:
                return
            self._cached_acknowledgement_audio = {
                "text": text,
                "data": data,
                "mime_type": _mime_type_for_path(file_path),
                "metrics": {"tts_synthesis_ms": tts_synthesis_ms, "tts_cache": "prewarmed"},
            }
        except Exception as exc:
            logger.debug("Reference realtime voice acknowledgement prewarm failed: %s", exc)

    async def _emit_cached_acknowledgement_audio(
        self,
        text: str,
        playback_generation: Optional[int],
        metadata: Mapping[str, str],
    ) -> bool:
        cached = self._cached_acknowledgement_audio
        if not cached or str(cached.get("text") or "") != text:
            return False
        data = cached.get("data")
        if not isinstance(data, bytes) or not data:
            return False
        payload = AudioChunk(codec=VoiceAudioCodec.OPUS, data=data).to_payload()
        payload["mime_type"] = str(cached.get("mime_type") or "audio/mpeg")
        if playback_generation is not None:
            payload["playback_generation"] = playback_generation
        payload.update(dict(metadata))
        payload["metrics"] = dict(cached.get("metrics") or {})
        await self._emit(VoiceEventType.AUDIO_OUTPUT_CHUNK, payload)
        return True

    def _speak_sync(self, text: str, metadata: Optional[Mapping[str, str]] = None) -> str:
        synthesize = self._synthesize_func
        if synthesize is None:
            from tools.tts_tool import text_to_speech_tool as synthesize

        raw = _call_synthesize(synthesize, text, metadata or {})
        result = json.loads(raw) if isinstance(raw, str) else raw
        if not result.get("success"):
            raise RuntimeError(str(result.get("error") or "speech synthesis failed"))
        return str(result.get("file_path") or "")

    async def _emit(self, event_type: VoiceEventType, payload: Mapping[str, Any]) -> None:
        if self.config is None:
            return
        if self._closed and event_type != VoiceEventType.SESSION_CLOSED:
            return
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


def create_reference_sidecar_app(runtime: Optional[ReferenceSidecarRuntimeConfig] = None):
    """Create the FastAPI app for the reference sidecar."""

    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Hermes realtime voice reference sidecar")
    runtime = runtime or runtime_config_from_env()

    @app.get("/health")
    async def health(request: Request):
        if not _authorized(request.headers, runtime.auth_token):
            raise HTTPException(status_code=401, detail="unauthorized")
        streaming_stt_health = await _probe_streaming_stt_health(runtime)
        streaming_tts_health = await _probe_streaming_tts_health(runtime)
        return reference_sidecar_health_payload(
            runtime,
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
        vllm_base_url=os.environ.get("HERMES_VOICE_VLLM_BASE_URL") or None,
        vllm_model=os.environ.get("HERMES_VOICE_VLLM_MODEL") or None,
        vllm_timeout_seconds=float(os.environ.get("HERMES_VOICE_VLLM_TIMEOUT_SECONDS") or 60),
        streaming_stt_base_url=os.environ.get("HERMES_VOICE_STREAMING_STT_BASE_URL") or None,
        streaming_stt_model=os.environ.get("HERMES_VOICE_STREAMING_STT_MODEL") or None,
        streaming_stt_token=os.environ.get("HERMES_VOICE_STREAMING_STT_TOKEN") or None,
        streaming_stt_timeout_seconds=float(os.environ.get("HERMES_VOICE_STREAMING_STT_TIMEOUT_SECONDS") or 10),
        streaming_bridge_health_timeout_seconds=float(
            os.environ.get("HERMES_VOICE_STREAMING_BRIDGE_HEALTH_TIMEOUT_SECONDS") or 0.2
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


def _write_temp_audio(audio: bytes, codec: VoiceAudioCodec) -> str:
    suffix = {
        VoiceAudioCodec.PCM16: ".wav",
        VoiceAudioCodec.OPUS: ".ogg",
        VoiceAudioCodec.WEBM_OPUS: ".webm",
    }.get(codec, ".webm")
    with tempfile.NamedTemporaryFile(prefix="hermes-voice-sidecar-", suffix=suffix, delete=False) as tmp:
        tmp.write(audio)
        return tmp.name


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


def _kame_reflex_payload_from_content(content: str) -> dict[str, Any]:
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
        }
    if not isinstance(parsed, Mapping):
        return {"text": text, "intent": text, "intent_source": "reflex_audio", "transcript_source": "none"}

    intent = str(parsed.get("intent") or parsed.get("text") or "").strip()
    transcript = str(parsed.get("transcript") or parsed.get("asr_transcript") or "").strip()
    oracle_text = str(parsed.get("text") or transcript or intent).strip()
    payload: dict[str, Any] = {
        "text": oracle_text,
        "intent": intent or oracle_text,
        "intent_source": str(parsed.get("intent_source") or "reflex_audio"),
        "transcript_source": str(parsed.get("transcript_source") or ("asr" if transcript else "none")),
    }
    route = str(parsed.get("route") or "").strip()
    if route:
        payload["route"] = route
    local_reply = (
        str(
            parsed.get("local_reply")
            or parsed.get("reply")
            or parsed.get("clarification")
            or parsed.get("interface_reply")
            or ""
        )
        .strip()
    )
    if local_reply:
        payload["local_reply"] = local_reply
    if transcript:
        payload["transcript"] = transcript
    confidence = _bounded_confidence(parsed.get("transcript_confidence"))
    if confidence is not None:
        payload["transcript_confidence"] = confidence
    return payload


def _bounded_confidence(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, parsed))


def _call_synthesize(synthesize: SynthesizeFn, text: str, metadata: Mapping[str, str]) -> Any:
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


def _payload_input_generation(payload: Mapping[str, Any]) -> Optional[int]:
    value = payload.get("input_generation")
    return _payload_int(value)


def _turn_acknowledgement_text(config: RealtimeVoiceSessionConfig) -> str:
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


def _metadata_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _payload_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


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
