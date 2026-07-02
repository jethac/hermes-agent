"""Realtime voice bridge for Discord voice channels.

This module keeps Discord-specific audio transport concerns at the plugin edge
and speaks the provider-neutral Hermes realtime voice sidecar protocol.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import logging
import sys
import time
from array import array
from typing import Any, Callable, Optional

from agent.realtime_voice import (
    AudioChunk,
    RealtimeVoiceASRMode,
    RealtimeVoiceEngineKind,
    RealtimeVoiceSessionConfig,
    VoiceAudioCodec,
    VoiceEvent,
    VoiceEventType,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_sidecar import RealtimeVoiceSidecarClient

logger = logging.getLogger(__name__)

DISCORD_SAMPLE_RATE = 48000
DISCORD_CHANNELS = 2
DISCORD_FRAME_MS = 20
DISCORD_FRAME_SAMPLES = DISCORD_SAMPLE_RATE * DISCORD_FRAME_MS // 1000
DISCORD_FRAME_BYTES = DISCORD_FRAME_SAMPLES * DISCORD_CHANNELS * 2
SIDECAR_SAMPLE_RATE = 16000
SIDECAR_CHANNELS = 1
SIDECAR_FRAME_SAMPLES = SIDECAR_SAMPLE_RATE * DISCORD_FRAME_MS // 1000
SIDECAR_FRAME_BYTES = SIDECAR_FRAME_SAMPLES * SIDECAR_CHANNELS * 2
DEFAULT_BARGE_IN_STOP_PLAYBACK_DEADLINE_MS = 150


def _is_legacy_audio_output_alias(event: VoiceEvent) -> bool:
    return (
        event.type == VoiceEventType.ASSISTANT_AUDIO_CHUNK
        and event.payload.get("audio_alias_for") == VoiceEventType.AUDIO_OUTPUT_CHUNK.value
    )


def _int16_array(pcm: bytes) -> array:
    samples = array("h")
    usable = len(pcm) - (len(pcm) % 2)
    samples.frombytes(pcm[:usable])
    if sys.byteorder != "little":  # PCM is always little-endian s16le.
        samples.byteswap()
    return samples


def _bytes_from_int16(samples: array) -> bytes:
    out = array("h", samples)
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def discord_pcm48_stereo_to_pcm16_mono(
    pcm48_stereo: bytes,
    *,
    target_rate_hz: int = SIDECAR_SAMPLE_RATE,
) -> bytes:
    """Downmix Discord-native 48 kHz stereo PCM16 to mono PCM16.

    The common realtime-provider path is 16 kHz mono.  The Discord packet cadence
    is 20 ms, so 3840 input bytes become 640 output bytes.  This helper is kept
    in-process to avoid spawning ffmpeg for every voice packet.
    """
    if target_rate_hz <= 0:
        raise ValueError("target_rate_hz must be positive")
    if DISCORD_SAMPLE_RATE % target_rate_hz != 0:
        raise ValueError(
            f"unsupported Discord realtime downsample rate: {DISCORD_SAMPLE_RATE}->{target_rate_hz}"
        )
    ratio = DISCORD_SAMPLE_RATE // target_rate_hz
    samples = _int16_array(pcm48_stereo)
    mono = array("h")
    # Interleaved L/R stereo -> mono average over each output sample window.
    # Discord gives 48 kHz packets; for the common 16 kHz sidecar path each
    # output sample is the average of three stereo frames, not just every third
    # frame. That keeps levels stable and avoids dropping one channel's timing
    # information during the integer-rate conversion.
    window_samples = DISCORD_CHANNELS * ratio
    for stereo_index in range(0, len(samples) - window_samples + 1, window_samples):
        total = 0
        for offset in range(0, window_samples, DISCORD_CHANNELS):
            total += int(samples[stereo_index + offset])
            total += int(samples[stereo_index + offset + 1])
        mono.append(round(total / window_samples))
    return _bytes_from_int16(mono)


def pcm16_mono_to_discord_pcm48_stereo(
    pcm16_mono: bytes,
    *,
    source_rate_hz: int = SIDECAR_SAMPLE_RATE,
) -> bytes:
    """Upsample mono PCM16 to Discord-native 48 kHz stereo PCM16."""
    if source_rate_hz <= 0:
        raise ValueError("source_rate_hz must be positive")
    if DISCORD_SAMPLE_RATE % source_rate_hz != 0:
        raise ValueError(
            f"unsupported Discord realtime upsample rate: {source_rate_hz}->{DISCORD_SAMPLE_RATE}"
        )
    ratio = DISCORD_SAMPLE_RATE // source_rate_hz
    samples = _int16_array(pcm16_mono)
    stereo = array("h")
    for sample in samples:
        value = int(sample)
        for _ in range(ratio):
            stereo.append(value)
            stereo.append(value)
    return _bytes_from_int16(stereo)


class DiscordRealtimeVoiceSession:
    """One realtime Hermes sidecar session bound to a Discord voice channel."""

    def __init__(
        self,
        *,
        guild_id: int,
        voice_channel_id: int,
        text_channel_id: Optional[int],
        sidecar_base_url: str,
        sidecar_token: Optional[str] = None,
        engine: str = RealtimeVoiceEngineKind.TEXT_ORACLE_TTS.value,
        frontend_provider: Optional[str] = None,
        frontend_model: Optional[str] = None,
        interface_base_url: Optional[str] = None,
        interface_temperature: float = 0.2,
        interface_max_output_tokens: int = 160,
        interface_timeout_seconds: float = 0.8,
        interface_max_audio_seconds: float = 30.0,
        interface_audio_input: Optional[str] = None,
        asr_mode: str = RealtimeVoiceASRMode.ON_ESCALATION.value,
        asr_provider: Optional[str] = None,
        asr_model: Optional[str] = None,
        asr_base_url: Optional[str] = None,
        preferred_local_oracle_model: Optional[str] = None,
        oracle_timeout_seconds: float = 60.0,
        max_spoken_sentences: int = 2,
        voice_response_policy: str = "sentence_cap",
        tts_provider: Optional[str] = None,
        tts_model: Optional[str] = None,
        tts_voice: Optional[str] = None,
        tts_base_url: Optional[str] = None,
        fallback_policy: Optional[str] = None,
        sidecar_connect_timeout_seconds: float = 10.0,
        turn_acknowledgement: Optional[dict] = None,
        routing_policy: Optional[dict] = None,
        metrics_policy: Optional[dict] = None,
        output_events: Optional[dict] = None,
        quality_targets_ms: Optional[dict] = None,
        oracle_jobs: Optional[dict] = None,
        barge_in_stop_playback_deadline_ms: int = DEFAULT_BARGE_IN_STOP_PLAYBACK_DEADLINE_MS,
        sidecar: Any = None,
        mixer: Any = None,
        degraded_callback: Optional[Callable[[str, str], Any]] = None,
        event_callback: Optional[Callable[[str, dict[str, Any]], Any]] = None,
    ) -> None:
        self.guild_id = int(guild_id)
        self.voice_channel_id = int(voice_channel_id)
        self.text_channel_id = int(text_channel_id) if text_channel_id is not None else None
        self.sidecar_base_url = sidecar_base_url
        self.sidecar_token = sidecar_token
        self.engine = engine
        self.frontend_provider = frontend_provider
        self.frontend_model = frontend_model
        self.interface_base_url = interface_base_url
        self.interface_temperature = interface_temperature
        self.interface_max_output_tokens = interface_max_output_tokens
        self.interface_timeout_seconds = interface_timeout_seconds
        self.interface_max_audio_seconds = interface_max_audio_seconds
        self.interface_audio_input = interface_audio_input
        self.asr_mode = asr_mode
        self.asr_provider = asr_provider
        self.asr_model = asr_model
        self.asr_base_url = asr_base_url
        self.preferred_local_oracle_model = preferred_local_oracle_model
        self.oracle_timeout_seconds = oracle_timeout_seconds
        self.max_spoken_sentences = max_spoken_sentences
        self.voice_response_policy = str(voice_response_policy or "sentence_cap")
        self.tts_provider = tts_provider
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.tts_base_url = tts_base_url
        self.fallback_policy = fallback_policy
        self.sidecar_connect_timeout_seconds = sidecar_connect_timeout_seconds
        self.turn_acknowledgement = dict(turn_acknowledgement or {})
        self.routing_policy = dict(routing_policy or {})
        self.metrics_policy = dict(metrics_policy or {})
        self.output_events = dict(output_events or {})
        self.quality_targets_ms = dict(quality_targets_ms or {})
        self.oracle_jobs = dict(oracle_jobs or {})
        try:
            self.barge_in_stop_playback_deadline_ms = max(0, int(barge_in_stop_playback_deadline_ms))
        except (TypeError, ValueError):
            self.barge_in_stop_playback_deadline_ms = DEFAULT_BARGE_IN_STOP_PLAYBACK_DEADLINE_MS
        self.sidecar = sidecar if sidecar is not None else RealtimeVoiceSidecarClient()
        self.mixer = mixer
        self.degraded_callback = degraded_callback
        self.event_callback = event_callback
        self.session_id = f"discord:{self.guild_id}:{self.voice_channel_id}"
        self._sequence = 0
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._started = False
        self._activity = asyncio.Event()
        self._playback_pcm48_stereo_buffer = bytearray()
        self._active_playback_generation = 0
        self._first_tts_audio_received_at_by_generation: dict[int, float] = {}
        self._playback_start_metric_generations: set[int] = set()
        self._last_input_user_id: Optional[str] = None
        self._input_user_ids_since_result: set[str] = set()

    async def start(self) -> None:
        config = RealtimeVoiceSessionConfig(
            session_id=self.session_id,
            engine=RealtimeVoiceEngineKind(str(self.engine or RealtimeVoiceEngineKind.TEXT_ORACLE_TTS.value)),
            input_codec=VoiceAudioCodec.PCM16,
            output_codec=VoiceAudioCodec.PCM16,
            sample_rate_hz=SIDECAR_SAMPLE_RATE,
            channels=SIDECAR_CHANNELS,
            frontend_provider=self.frontend_provider,
            frontend_model=self.frontend_model,
            interface_temperature=self.interface_temperature,
            interface_max_output_tokens=self.interface_max_output_tokens,
            interface_timeout_seconds=self.interface_timeout_seconds,
            interface_max_audio_seconds=self.interface_max_audio_seconds,
            interface_audio_input=self.interface_audio_input,
            interface_base_url=self.interface_base_url,
            asr_mode=RealtimeVoiceASRMode(str(self.asr_mode or RealtimeVoiceASRMode.ON_ESCALATION.value)),
            asr_provider=self.asr_provider,
            asr_model=self.asr_model,
            asr_base_url=self.asr_base_url,
            preferred_local_oracle_model=self.preferred_local_oracle_model,
            oracle_timeout_seconds=self.oracle_timeout_seconds,
            max_spoken_sentences=self.max_spoken_sentences,
            voice_response_policy=self.voice_response_policy,
            tts_provider=self.tts_provider,
            tts_model=self.tts_model,
            tts_voice=self.tts_voice,
            tts_base_url=self.tts_base_url,
            fallback_policy=self.fallback_policy,
            sidecar_base_url=self.sidecar_base_url,
            sidecar_token=self.sidecar_token,
            sidecar_connect_timeout_seconds=self.sidecar_connect_timeout_seconds,
            turn_acknowledgement=dict(self.turn_acknowledgement),
            routing_policy=dict(self.routing_policy),
            metrics_policy=dict(self.metrics_policy),
            output_events=dict(self.output_events),
            quality_targets_ms=dict(self.quality_targets_ms),
            oracle_jobs=dict(self.oracle_jobs),
            barge_in_policy={
                "stop_playback_deadline_ms": self.barge_in_stop_playback_deadline_ms,
            },
            metadata={
                "transport": "discord_voice",
                "voice_architecture": "kame_frontend_oracle",
                "frontend_role": "low_latency_voice_interface",
                "oracle_role": "hermes_backend_oracle",
                "frontend_provider": self.frontend_provider,
                "frontend_model": self.frontend_model,
                "interface_base_url": self.interface_base_url,
                "interface_temperature": self.interface_temperature,
                "interface_max_output_tokens": self.interface_max_output_tokens,
                "interface_timeout_seconds": self.interface_timeout_seconds,
                "interface_max_audio_seconds": self.interface_max_audio_seconds,
                "interface_audio_input": self.interface_audio_input,
                "asr_mode": self.asr_mode,
                "asr_provider": self.asr_provider,
                "asr_model": self.asr_model,
                "asr_base_url": self.asr_base_url,
                "preferred_local_oracle_model": self.preferred_local_oracle_model,
                "oracle_timeout_seconds": self.oracle_timeout_seconds,
                "max_spoken_sentences": self.max_spoken_sentences,
                "voice_response_policy": self.voice_response_policy,
                "tts_provider": self.tts_provider,
                "tts_model": self.tts_model,
                "tts_voice": self.tts_voice,
                "tts_base_url": self.tts_base_url,
                "fallback_policy": self.fallback_policy,
                "guild_id": str(self.guild_id),
                "voice_channel_id": str(self.voice_channel_id),
                "text_channel_id": str(self.text_channel_id) if self.text_channel_id is not None else None,
                "turn_acknowledgement": dict(self.turn_acknowledgement),
                "routing": dict(self.routing_policy),
                "metrics": dict(self.metrics_policy),
                "output_events": dict(self.output_events),
                "quality_targets_ms": dict(self.quality_targets_ms),
                "oracle_jobs": dict(self.oracle_jobs),
                "barge_in": {
                    "stop_playback_deadline_ms": self.barge_in_stop_playback_deadline_ms,
                },
                "barge_in_stop_playback_deadline_ms": self.barge_in_stop_playback_deadline_ms,
            },
        )
        await self.sidecar.start(config)
        self._started = True
        self._reader_task = asyncio.create_task(self._consume_sidecar_events())

    async def close(self) -> None:
        if self._closed:
            return
        if self._started:
            await self._stop_mixer_speech_for_shutdown()
            self._flush_playback_buffer()
            try:
                await self._send_event(
                    VoiceEventType.SESSION_CLOSED,
                    {"transport": "discord_voice"},
                )
            except Exception as exc:
                logger.debug("Discord realtime voice session close event failed: %s", exc)
        self._closed = True
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        await self.sidecar.close()

    async def _stop_mixer_speech_for_shutdown(self) -> None:
        with contextlib.suppress(Exception):
            await self._stop_mixer_speech_for_barge_in(require_active=True)

    async def handle_speech_start(self, *, user_id: int | str) -> None:
        if self._closed or not self._started:
            return
        await self._send_event(
            VoiceEventType.SPEECH_START,
            {
                "user_id": str(user_id),
                "transport": "discord_voice",
            },
        )
        stop_result = await self._stop_mixer_speech_for_barge_in(require_active=True)
        self._active_playback_generation += 1
        await self._send_event(
            VoiceEventType.BARGE_IN,
            {
                "user_id": str(user_id),
                "transport": "discord_voice",
                "playback_active": stop_result["playback_active"],
                "playback_generation": self._active_playback_generation,
                **stop_result,
            },
        )

    async def handle_speech_energy(
        self,
        *,
        user_id: int | str,
        rms: int | float,
        duration_seconds: float,
    ) -> None:
        if self._closed or not self._started:
            return
        try:
            rms_value = max(0, int(rms))
        except (TypeError, ValueError):
            rms_value = 0
        try:
            duration_ms = max(0, int(round(float(duration_seconds) * 1000)))
        except (TypeError, ValueError):
            duration_ms = 0
        await self._send_event(
            VoiceEventType.SPEECH_ENERGY,
            {
                "user_id": str(user_id),
                "transport": "discord_voice",
                "rms": rms_value,
                "duration_ms": duration_ms,
            },
        )

    async def handle_speech_end(self, *, user_id: int | str) -> None:
        if self._closed or not self._started:
            return
        await self._send_event(
            VoiceEventType.SPEECH_END,
            {
                "user_id": str(user_id),
                "transport": "discord_voice",
            },
        )
        await self._send_event(
            VoiceEventType.AUDIO_INPUT_CHUNK,
            {
                "codec": VoiceAudioCodec.PCM16.value,
                "sample_rate_hz": SIDECAR_SAMPLE_RATE,
                "channels": SIDECAR_CHANNELS,
                "data_b64": "",
                "user_id": str(user_id),
                "transport": "discord_voice",
                "end_of_utterance": True,
            },
        )

    async def handle_pcm_frame(self, *, user_id: int | str, pcm48_stereo: bytes) -> None:
        if self._closed or not self._started or not pcm48_stereo:
            return
        self._last_input_user_id = str(user_id)
        self._input_user_ids_since_result.add(str(user_id))
        pcm16_mono = discord_pcm48_stereo_to_pcm16_mono(pcm48_stereo)
        await self._send_event(
            VoiceEventType.AUDIO_INPUT_CHUNK,
            {
                "codec": VoiceAudioCodec.PCM16.value,
                "sample_rate_hz": SIDECAR_SAMPLE_RATE,
                "channels": SIDECAR_CHANNELS,
                "data_b64": base64.b64encode(pcm16_mono).decode("ascii"),
                "user_id": str(user_id),
                "transport": "discord_voice",
            },
        )

    async def wait_until_idle(self) -> None:
        try:
            await asyncio.wait_for(self._activity.wait(), timeout=0.25)
        except asyncio.TimeoutError:
            return
        self._activity.clear()

    async def _send_event(self, event_type: VoiceEventType, payload: dict[str, Any]) -> None:
        self._sequence += 1
        await self.sidecar.send_event(
            VoiceEvent(
                type=event_type,
                session_id=self.session_id,
                sequence=self._sequence,
                payload=payload,
            )
        )

    async def _consume_sidecar_events(self) -> None:
        try:
            async for event in self.sidecar.events():
                if event.type in {VoiceEventType.AUDIO_OUTPUT_CHUNK, VoiceEventType.ASSISTANT_AUDIO_CHUNK}:
                    if self._drop_stale_playback_event(event):
                        continue
                    if not _is_legacy_audio_output_alias(event):
                        self._handle_audio_output(event)
                elif event.type == VoiceEventType.ASSISTANT_AUDIO_END:
                    if self._drop_stale_playback_event(event):
                        continue
                    self._flush_playback_buffer()
                elif event.type == VoiceEventType.PLAYBACK_STARTED:
                    if self._drop_stale_playback_event(event):
                        continue
                elif event.type == VoiceEventType.PLAYBACK_STOPPED:
                    if self._drop_stale_playback_event(event):
                        continue
                    self._flush_playback_buffer()
                elif event.type == VoiceEventType.ASSISTANT_COMMIT:
                    if self._drop_stale_playback_event(event):
                        continue
                    self._flush_playback_buffer()
                elif event.type == VoiceEventType.SESSION_CLOSED:
                    self._flush_playback_buffer()
                elif event.type == VoiceEventType.BARGE_IN:
                    self._advance_playback_generation(event)
                    event.payload.update(
                        await self._stop_mixer_speech_for_barge_in(require_active=False)
                    )
                elif event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
                    self._advance_playback_generation(event)
                elif event.type == VoiceEventType.SESSION_ERROR:
                    error = sanitize_realtime_voice_error(event.payload.get("error") or "sidecar session error")
                    logger.warning("Discord realtime voice sidecar reported session error: %s", error)
                    await self._notify_event_observed(event)
                    await self._notify_degraded("sidecar_session_error", error)
                    self._activity.set()
                    return
                if event.type in {
                    VoiceEventType.TRANSCRIPT_PARTIAL,
                    VoiceEventType.TRANSCRIPT_FINAL,
                    VoiceEventType.INTERFACE_INTENT_PARTIAL,
                    VoiceEventType.INTERFACE_INTENT_FINAL,
                }:
                    if "user_id" not in event.payload:
                        if len(self._input_user_ids_since_result) == 1:
                            event.payload["user_id"] = next(iter(self._input_user_ids_since_result))
                        elif len(self._input_user_ids_since_result) > 1:
                            logger.info(
                                "Discord realtime voice left untagged provider event without user_id "
                                "(type=%s, users=%s)",
                                event.type.value,
                                sorted(self._input_user_ids_since_result),
                            )
                    if event.type == VoiceEventType.TRANSCRIPT_FINAL:
                        self._input_user_ids_since_result.clear()
                await self._notify_event_observed(event)
                self._activity.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive gateway logging
            error = sanitize_realtime_voice_error(exc)
            logger.warning("Discord realtime voice sidecar event loop failed: %s", error, exc_info=True)
            await self._notify_degraded("sidecar_event_stream_failed", error)
            self._activity.set()

    async def _notify_degraded(self, reason: str, error: str) -> None:
        callback = self.degraded_callback
        if callback is None:
            return
        try:
            result = callback(reason, error)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # pragma: no cover - defensive callback isolation
            logger.debug("Discord realtime degradation callback failed: %s", exc)

    async def _notify_event_observed(self, event: VoiceEvent) -> None:
        callback = self.event_callback
        if callback is None:
            return
        try:
            result = callback(event.type.value, dict(event.payload))
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # pragma: no cover - defensive callback isolation
            logger.debug("Discord realtime event callback failed: %s", exc)

    async def _stop_mixer_speech_for_barge_in(self, *, require_active: bool) -> dict[str, Any]:
        result: dict[str, Any] = {
            "playback_active": False,
            "playback_stop_attempted": False,
            "playback_stop_timed_out": False,
            "playback_stop_deadline_ms": self.barge_in_stop_playback_deadline_ms,
        }
        self._playback_pcm48_stereo_buffer.clear()
        mixer = self.mixer
        if mixer is None:
            return result
        try:
            result["playback_active"] = bool(getattr(mixer, "speech_active", False))
        except Exception:
            result["playback_active"] = False
        if require_active and not result["playback_active"]:
            return result
        stop = getattr(mixer, "stop_speech", None)
        if not callable(stop):
            return result

        result["playback_stop_attempted"] = True
        timeout_seconds = max(self.barge_in_stop_playback_deadline_ms / 1000.0, 0.001)
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            await asyncio.wait_for(self._call_mixer_stop(stop), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            result["playback_stop_timed_out"] = True
            logger.warning(
                "Discord realtime mixer stop exceeded barge-in deadline: deadline_ms=%d",
                self.barge_in_stop_playback_deadline_ms,
            )
        finally:
            playback_stop_ms = int(round((loop.time() - started) * 1000))
            result["playback_stop_ms"] = playback_stop_ms
            result["barge_in_confirmed_to_playback_stopped_ms"] = playback_stop_ms
            result["metrics"] = {
                "barge_in_confirmed_to_playback_stopped_ms": playback_stop_ms,
            }
        return result

    async def _call_mixer_stop(self, stop: Callable[[], Any]) -> None:
        if inspect.iscoroutinefunction(stop):
            await stop()
            return
        value = await asyncio.to_thread(stop)
        if inspect.isawaitable(value):
            await value

    def _drop_stale_playback_event(self, event: VoiceEvent) -> bool:
        generation = _payload_generation(event.payload)
        if generation is None:
            return False
        if generation < self._active_playback_generation:
            logger.debug(
                "Dropping stale Discord realtime playback event: generation=%s active=%s type=%s",
                generation,
                self._active_playback_generation,
                event.type.value,
            )
            return True
        self._active_playback_generation = generation
        return False

    def _advance_playback_generation(self, event: VoiceEvent) -> None:
        generation = _payload_generation(event.payload)
        if generation is None:
            return
        self._active_playback_generation = max(self._active_playback_generation, generation)
        self._first_tts_audio_received_at_by_generation = {
            key: value
            for key, value in self._first_tts_audio_received_at_by_generation.items()
            if key >= self._active_playback_generation
        }
        self._playback_start_metric_generations = {
            key for key in self._playback_start_metric_generations if key >= self._active_playback_generation
        }

    def _handle_audio_output(self, event: VoiceEvent) -> None:
        if self.mixer is None:
            return
        generation = _payload_generation(event.payload)
        if generation is not None:
            self._first_tts_audio_received_at_by_generation.setdefault(generation, time.perf_counter())
        try:
            chunk = AudioChunk.from_payload(event.payload)
        except Exception as exc:
            logger.warning("Invalid Discord realtime audio output chunk: %s", exc)
            return
        if chunk.codec != VoiceAudioCodec.PCM16:
            logger.warning("Unsupported Discord realtime output codec: %s", chunk.codec)
            return
        try:
            if chunk.sample_rate_hz == DISCORD_SAMPLE_RATE and chunk.channels == DISCORD_CHANNELS:
                pcm48_stereo = chunk.data
            elif chunk.channels == 1:
                pcm48_stereo = pcm16_mono_to_discord_pcm48_stereo(
                    chunk.data,
                    source_rate_hz=chunk.sample_rate_hz,
                )
            else:
                logger.warning(
                    "Unsupported Discord realtime output geometry: %s Hz, %s channel(s)",
                    chunk.sample_rate_hz,
                    chunk.channels,
                )
                return
        except Exception as exc:
            logger.warning("Could not convert Discord realtime output audio: %s", exc)
            return

        if self._enqueue_mixer_pcm(pcm48_stereo):
            self._add_playback_start_metric(event, generation)

    def _enqueue_mixer_pcm(self, pcm48_stereo: bytes) -> bool:
        enqueued = False
        self._playback_pcm48_stereo_buffer.extend(pcm48_stereo)
        enqueue = getattr(self.mixer, "enqueue_speech_frame", None)
        while len(self._playback_pcm48_stereo_buffer) >= DISCORD_FRAME_BYTES:
            frame = bytes(self._playback_pcm48_stereo_buffer[:DISCORD_FRAME_BYTES])
            del self._playback_pcm48_stereo_buffer[:DISCORD_FRAME_BYTES]
            if callable(enqueue):
                enqueue(frame, fade_in_ms=0)
            else:
                self.mixer.play_speech(frame, fade_in_ms=0)
            enqueued = True
        return enqueued

    def _add_playback_start_metric(self, event: VoiceEvent, generation: Optional[int]) -> None:
        if generation is None or generation in self._playback_start_metric_generations:
            return
        first_tts_audio_at = self._first_tts_audio_received_at_by_generation.get(generation)
        if first_tts_audio_at is None:
            return
        existing_metrics = event.payload.get("metrics")
        if not isinstance(existing_metrics, dict) or not any(
            str(key).startswith("kame_") for key in existing_metrics
        ):
            return
        metrics = dict(existing_metrics)
        playback_start_ms = max(
            0,
            int(round((time.perf_counter() - first_tts_audio_at) * 1000)),
        )
        metrics["kame_first_tts_audio_to_playback_start_ms"] = playback_start_ms
        speech_to_first_audio_ms = metrics.get("kame_speech_end_to_first_audio_ms")
        if isinstance(speech_to_first_audio_ms, int) and not isinstance(speech_to_first_audio_ms, bool):
            metrics["kame_speech_end_to_playback_start_ms"] = max(
                0,
                speech_to_first_audio_ms + playback_start_ms,
            )
        event.payload["metrics"] = metrics
        self._playback_start_metric_generations.add(generation)

    def _flush_playback_buffer(self) -> None:
        if self.mixer is None:
            self._playback_pcm48_stereo_buffer.clear()
            return
        if not self._playback_pcm48_stereo_buffer:
            finish = getattr(self.mixer, "finish_speech_stream", None)
            if callable(finish):
                finish()
            return
        remainder = len(self._playback_pcm48_stereo_buffer) % DISCORD_FRAME_BYTES
        if remainder:
            self._playback_pcm48_stereo_buffer.extend(b"\x00" * (DISCORD_FRAME_BYTES - remainder))
        self._enqueue_mixer_pcm(b"")
        finish = getattr(self.mixer, "finish_speech_stream", None)
        if callable(finish):
            finish()


def _payload_generation(payload: dict[str, Any]) -> Optional[int]:
    value = payload.get("playback_generation")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
