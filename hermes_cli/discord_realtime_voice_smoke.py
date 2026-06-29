"""Local Discord realtime voice bridge smoke check."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import struct
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.realtime_voice import AudioChunk, VoiceAudioCodec, VoiceEvent, VoiceEventType
from plugins.platforms.discord.realtime_voice import (
    DISCORD_FRAME_BYTES,
    SIDECAR_FRAME_BYTES,
    DiscordRealtimeVoiceSession,
)


@dataclass(frozen=True)
class DiscordRealtimeVoiceSmokeResult:
    ok: bool
    mode: str
    transport: str
    input_pcm48_bytes: int
    sidecar_pcm16_bytes: int
    sidecar_pcm16_first_sample: int | None
    sidecar_pcm16_checksum: int
    mixer_frames: int
    mixer_frame_bytes: int
    barge_in_sent: bool
    speech_energy_sent: bool
    mixer_stop_calls: int
    sidecar_closed: bool
    shutdown_elapsed_ms: int | None
    shutdown_bounded: bool
    shutdown_timed_out: bool
    events: list[str]
    evidence_context: dict[str, Any] = field(default_factory=dict)
    latency_metrics_ms: dict[str, int] = field(default_factory=dict)
    error: str = ""


class _SmokeSidecar:
    def __init__(self):
        self.started_with = None
        self.sent: list[VoiceEvent] = []
        self._events: asyncio.Queue[VoiceEvent | None] = asyncio.Queue()
        self.closed = False

    async def start(self, config):
        self.started_with = config

    async def send_event(self, event: VoiceEvent):
        self.sent.append(event)
        if event.type == VoiceEventType.AUDIO_INPUT_CHUNK:
            generation = 1
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.TRANSCRIPT_PARTIAL,
                    session_id=event.session_id,
                    sequence=1,
                    payload={
                        "text": "loopback",
                        "stability": 0.8,
                        "playback_generation": generation,
                    },
                )
            )
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.TRANSCRIPT_FINAL,
                    session_id=event.session_id,
                    sequence=2,
                    payload={
                        "text": "loopback discord transcript",
                        "playback_generation": generation,
                    },
                )
            )
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                    session_id=event.session_id,
                    sequence=3,
                    payload={
                        "text": "One moment.",
                        "playback_generation": generation,
                    },
                )
            )
            payload = AudioChunk(
                codec=VoiceAudioCodec.PCM16,
                data=b"\x00" * SIDECAR_FRAME_BYTES,
                sample_rate_hz=16000,
                channels=1,
            ).to_payload()
            payload["playback_generation"] = generation
            payload["metrics"] = {"discord_loopback_first_audio_ms": 0}
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
                    session_id=event.session_id,
                    sequence=4,
                    payload=payload,
                )
            )
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.ASSISTANT_COMMIT,
                    session_id=event.session_id,
                    sequence=5,
                    payload={"text": "One moment.", "playback_generation": generation},
                )
            )
        elif event.type == VoiceEventType.BARGE_IN:
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.BARGE_IN,
                    session_id=event.session_id,
                    sequence=6,
                    payload=dict(event.payload),
                )
            )

    async def events(self):
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def close(self):
        self.closed = True
        await self._events.put(None)


class _SmokeMixer:
    def __init__(self):
        self.frames: list[bytes] = []
        self.stop_calls = 0
        self.speech_active = False

    def enqueue_speech_frame(self, frame: bytes, *, fade_in_ms: int = 0):
        self.frames.append(frame)
        self.speech_active = True

    def stop_speech(self):
        self.stop_calls += 1
        self.speech_active = False


async def run_discord_realtime_voice_smoke() -> DiscordRealtimeVoiceSmokeResult:
    sidecar = _SmokeSidecar()
    mixer = _SmokeMixer()
    observed: list[str] = []
    latency_metrics_ms: dict[str, int] = {}
    evidence_context = _build_evidence_context()
    shutdown_elapsed_ms: int | None = None
    shutdown_bounded = False
    shutdown_timed_out = False
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar_base_url="memory://discord-loopback",
        sidecar=sidecar,
        mixer=mixer,
        turn_acknowledgement={"enabled": True, "text": "One moment."},
        event_callback=lambda event_type, _payload: observed.append(event_type),
    )
    try:
        start_ms = time.monotonic()
        await session.start()
        latency_metrics_ms["session_start_ms"] = _elapsed_ms(start_ms)
        audio_started = time.monotonic()
        await session.handle_pcm_frame(user_id=42, pcm48_stereo=_sentinel_pcm48_frame())
        for _ in range(8):
            await session.wait_until_idle()
            if mixer.frames:
                latency_metrics_ms["input_to_first_mixer_frame_ms"] = _elapsed_ms(audio_started)
                break
        mixer.speech_active = True
        barge_started = time.monotonic()
        await session.handle_speech_energy(user_id=42, rms=512, duration_seconds=0.02)
        await session.handle_speech_start(user_id=42)
        for _ in range(4):
            await session.wait_until_idle()
            if any(event.type == VoiceEventType.BARGE_IN for event in sidecar.sent):
                latency_metrics_ms["barge_in_ack_ms"] = _elapsed_ms(barge_started)
                break
        shutdown_started = time.monotonic()
        try:
            await asyncio.wait_for(session.close(), timeout=2.0)
            shutdown_bounded = True
        except asyncio.TimeoutError:
            shutdown_timed_out = True
            raise RuntimeError("discord realtime loopback shutdown timed out after 2.0s") from None
        finally:
            shutdown_elapsed_ms = _elapsed_ms(shutdown_started)
            latency_metrics_ms["shutdown_ms"] = shutdown_elapsed_ms
    except Exception as exc:
        return DiscordRealtimeVoiceSmokeResult(
            ok=False,
            mode="discord_loopback",
            transport="discord_voice",
            evidence_context=evidence_context,
            latency_metrics_ms=latency_metrics_ms,
            input_pcm48_bytes=DISCORD_FRAME_BYTES,
            sidecar_pcm16_bytes=0,
            sidecar_pcm16_first_sample=None,
            sidecar_pcm16_checksum=0,
            mixer_frames=len(mixer.frames),
            mixer_frame_bytes=len(mixer.frames[0]) if mixer.frames else 0,
            barge_in_sent=False,
            speech_energy_sent=any(event.type == VoiceEventType.SPEECH_ENERGY for event in sidecar.sent),
            mixer_stop_calls=mixer.stop_calls,
            sidecar_closed=sidecar.closed,
            shutdown_elapsed_ms=shutdown_elapsed_ms,
            shutdown_bounded=shutdown_bounded,
            shutdown_timed_out=shutdown_timed_out,
            events=observed,
            error=str(exc),
        )

    audio_events = [event for event in sidecar.sent if event.type == VoiceEventType.AUDIO_INPUT_CHUNK]
    sidecar_pcm16_bytes = 0
    sidecar_pcm16_first_sample: int | None = None
    sidecar_pcm16_checksum = 0
    if audio_events:
        pcm16 = base64.b64decode(audio_events[0].payload.get("data_b64") or "")
        sidecar_pcm16_bytes = len(pcm16)
        sidecar_pcm16_checksum = sum(pcm16)
        if len(pcm16) >= 2:
            sidecar_pcm16_first_sample = struct.unpack("<h", pcm16[:2])[0]
    barge_in_sent = any(event.type == VoiceEventType.BARGE_IN for event in sidecar.sent)
    speech_energy_sent = any(event.type == VoiceEventType.SPEECH_ENERGY for event in sidecar.sent)
    ok = (
        sidecar.started_with is not None
        and sidecar.started_with.metadata.get("transport") == "discord_voice"
        and sidecar_pcm16_bytes == SIDECAR_FRAME_BYTES
        and sidecar_pcm16_first_sample == 450
        and sidecar_pcm16_checksum > 0
        and len(mixer.frames) >= 1
        and len(mixer.frames[0]) == DISCORD_FRAME_BYTES
        and speech_energy_sent
        and barge_in_sent
        and mixer.stop_calls >= 1
        and shutdown_bounded
        and not shutdown_timed_out
    )
    return DiscordRealtimeVoiceSmokeResult(
        ok=ok,
        mode="discord_loopback",
        transport="discord_voice",
        evidence_context=evidence_context,
        latency_metrics_ms=latency_metrics_ms,
        input_pcm48_bytes=DISCORD_FRAME_BYTES,
        sidecar_pcm16_bytes=sidecar_pcm16_bytes,
        sidecar_pcm16_first_sample=sidecar_pcm16_first_sample,
        sidecar_pcm16_checksum=sidecar_pcm16_checksum,
        mixer_frames=len(mixer.frames),
        mixer_frame_bytes=len(mixer.frames[0]) if mixer.frames else 0,
        barge_in_sent=barge_in_sent,
        speech_energy_sent=speech_energy_sent,
        mixer_stop_calls=mixer.stop_calls,
        sidecar_closed=sidecar.closed,
        shutdown_elapsed_ms=shutdown_elapsed_ms,
        shutdown_bounded=shutdown_bounded,
        shutdown_timed_out=shutdown_timed_out,
        events=observed,
        error="" if ok else "discord realtime loopback smoke did not satisfy invariants",
    )


def _sentinel_pcm48_frame() -> bytes:
    first_window = struct.pack(
        "<hhhhhh",
        300,
        600,
        300,
        600,
        300,
        600,
    )
    return first_window + (b"\x00" * (DISCORD_FRAME_BYTES - len(first_window)))


def _build_evidence_context() -> dict[str, str]:
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
    }


def _git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ("git", *args),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _elapsed_ms(start: float) -> int:
    return max(0, int(round((time.monotonic() - start) * 1000)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local Discord realtime voice bridge smoke check")
    parser.add_argument("--report", default="", help="Optional JSON report path")
    args = parser.parse_args(argv)
    result = asyncio.run(run_discord_realtime_voice_smoke())
    payload = asdict(result)
    if args.report:
        path = Path(args.report).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
