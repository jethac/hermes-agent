"""Bounded live Discord voice-channel probe for realtime voice validation."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ctypes.util
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DiscordVoiceLiveProbeResult:
    ok: bool
    evidence_context: dict[str, Any] = field(default_factory=dict)
    latency_metrics_ms: dict[str, int] = field(default_factory=dict)
    guild_name: str = ""
    voice_channel_name: str = ""
    connect_perm: bool = False
    speak_perm: bool = False
    members_before: int = 0
    connected: bool = False
    opus_loaded: bool = False
    accepted_audio_source: bool = False
    played: bool = False
    playing_during_probe: bool = False
    receiver_started: bool = False
    receiver_frames: int = 0
    receiver_speech_start: int = 0
    inbound_observed: bool = False
    members_after: int = 0
    disconnected: bool = False
    require_inbound: bool = False
    wait_seconds: float = 0.0
    failure_reason: str = ""
    error: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded live Discord voice-channel probe")
    parser.add_argument("--guild-id", default=os.environ.get("DISCORD_GUILD_ID", ""))
    parser.add_argument("--text-channel-id", default=os.environ.get("DISCORD_HOME_CHANNEL", ""))
    parser.add_argument("--voice-channel-id", default=os.environ.get("DISCORD_VOICE_CHANNEL_ID", ""))
    parser.add_argument("--voice-channel-name", default=os.environ.get("DISCORD_VOICE_CHANNEL_NAME", "General"))
    parser.add_argument("--wait-seconds", type=float, default=2.0)
    parser.add_argument(
        "--require-inbound",
        action="store_true",
        help="Fail unless inbound live speech frames or speech-start callbacks are observed",
    )
    parser.add_argument("--report", default="", help="Optional JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = asyncio.run(run_discord_voice_live_probe(args))
    payload = asdict(result)
    if args.report:
        path = Path(args.report).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


async def run_discord_voice_live_probe(args: argparse.Namespace) -> DiscordVoiceLiveProbeResult:
    evidence_context = _build_evidence_context(args)
    try:
        import discord
    except Exception as exc:
        return DiscordVoiceLiveProbeResult(
            ok=False,
            evidence_context=evidence_context,
            error=f"discord.py unavailable: {exc}",
        )

    token = str(os.environ.get("DISCORD_BOT_TOKEN") or "").strip()
    if not token:
        return DiscordVoiceLiveProbeResult(
            ok=False,
            evidence_context=evidence_context,
            error="DISCORD_BOT_TOKEN is required",
        )

    opus_loaded = _load_opus(discord)
    if not opus_loaded:
        return DiscordVoiceLiveProbeResult(
            ok=False,
            evidence_context=evidence_context,
            error="opus codec is not loaded",
        )

    intents = discord.Intents.default()
    intents.guilds = True
    intents.voice_states = True
    client = discord.Client(intents=intents)
    result_holder: dict[str, Any] = {"result": None}

    @client.event
    async def on_ready():
        try:
            result_holder["result"] = await _run_ready_probe(client, discord, args, opus_loaded=opus_loaded)
        except Exception as exc:
            result_holder["result"] = DiscordVoiceLiveProbeResult(
                ok=False,
                evidence_context=evidence_context,
                opus_loaded=opus_loaded,
                require_inbound=bool(args.require_inbound),
                wait_seconds=max(0.0, float(args.wait_seconds or 0.0)),
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            await client.close()

    await client.start(token)
    result = result_holder.get("result")
    if isinstance(result, DiscordVoiceLiveProbeResult):
        return result
    return DiscordVoiceLiveProbeResult(
        ok=False,
        evidence_context=evidence_context,
        opus_loaded=opus_loaded,
        error="probe did not complete",
    )


async def _run_ready_probe(
    client: Any,
    discord: Any,
    args: argparse.Namespace,
    *,
    opus_loaded: bool,
) -> DiscordVoiceLiveProbeResult:
    from plugins.platforms.discord.adapter import VoiceReceiver
    from plugins.platforms.discord.voice_mixer import FRAME_SIZE, VoiceMixer

    guild = await _resolve_guild(client, args)
    if guild is None:
        raise RuntimeError("target guild not found")
    channel = _resolve_voice_channel(guild, args)
    if channel is None:
        raise RuntimeError("target voice channel not found")
    member = getattr(guild, "me", None)
    perms = channel.permissions_for(member) if member is not None else None
    connect_perm = bool(getattr(perms, "connect", False))
    speak_perm = bool(getattr(perms, "speak", False))
    wait_seconds = max(0.0, float(args.wait_seconds or 0.0))
    members_before = len(getattr(channel, "members", []) or [])
    evidence_context = _build_evidence_context(args)

    vc = None
    receiver = None
    receiver_events = {"frames": 0, "speech_start": 0}
    latency_metrics_ms: dict[str, int] = {}
    try:
        connect_started = time.monotonic()
        vc = await channel.connect(timeout=15, reconnect=False)
        latency_metrics_ms["connect_ms"] = _elapsed_ms(connect_started)
        receiver = VoiceReceiver(
            vc,
            realtime_frame_callback=lambda _user_id, _pcm: receiver_events.__setitem__(
                "frames",
                receiver_events["frames"] + 1,
            ),
            realtime_speech_start_callback=lambda _user_id: receiver_events.__setitem__(
                "speech_start",
                receiver_events["speech_start"] + 1,
            ),
        )
        receiver.start()
        mixer = VoiceMixer()
        mixer.play_speech(b"\x00" * FRAME_SIZE * 12)
        accepted_audio_source = isinstance(mixer, discord.AudioSource)
        playback_started = time.monotonic()
        vc.play(mixer)
        deadline = time.monotonic() + wait_seconds
        playing_during_probe = False
        while time.monotonic() < deadline:
            playing_during_probe = playing_during_probe or bool(vc.is_playing())
            if playing_during_probe and "playback_observed_ms" not in latency_metrics_ms:
                latency_metrics_ms["playback_observed_ms"] = _elapsed_ms(playback_started)
            if args.require_inbound and (receiver_events["frames"] or receiver_events["speech_start"]):
                latency_metrics_ms["inbound_observed_ms"] = _elapsed_ms(playback_started)
                break
            await asyncio.sleep(0.05)
        if wait_seconds == 0:
            playing_during_probe = bool(vc.is_playing())
        receiver.stop()
        receiver = None
        disconnect_started = time.monotonic()
        await vc.disconnect(force=True)
        latency_metrics_ms["disconnect_ms"] = _elapsed_ms(disconnect_started)
        members_after = len(getattr(channel, "members", []) or [])
        disconnected = not vc.is_connected()
        inbound_ok = receiver_events["frames"] > 0 or receiver_events["speech_start"] > 0
        failure_reason = _probe_failure_reason(
            connect_perm=connect_perm,
            speak_perm=speak_perm,
            accepted_audio_source=accepted_audio_source,
            playing_during_probe=playing_during_probe,
            disconnected=disconnected,
            require_inbound=bool(args.require_inbound),
            inbound_observed=inbound_ok,
            members_before=members_before,
            members_after=members_after,
        )
        ok = not failure_reason
        return DiscordVoiceLiveProbeResult(
            ok=ok,
            evidence_context=evidence_context,
            latency_metrics_ms=latency_metrics_ms,
            guild_name=str(getattr(guild, "name", "") or ""),
            voice_channel_name=str(getattr(channel, "name", "") or ""),
            connect_perm=connect_perm,
            speak_perm=speak_perm,
            members_before=members_before,
            connected=True,
            opus_loaded=opus_loaded,
            accepted_audio_source=accepted_audio_source,
            played=True,
            playing_during_probe=playing_during_probe,
            receiver_started=True,
            receiver_frames=receiver_events["frames"],
            receiver_speech_start=receiver_events["speech_start"],
            inbound_observed=inbound_ok,
            members_after=members_after,
            disconnected=disconnected,
            require_inbound=bool(args.require_inbound),
            wait_seconds=wait_seconds,
            failure_reason=failure_reason,
            error="" if ok else f"live Discord voice probe did not satisfy invariants: {failure_reason}",
        )
    finally:
        if receiver is not None:
            with contextlib.suppress(Exception):
                receiver.stop()
        if vc is not None and vc.is_connected():
            with contextlib.suppress(Exception):
                await vc.disconnect(force=True)


def _probe_failure_reason(
    *,
    connect_perm: bool,
    speak_perm: bool,
    accepted_audio_source: bool,
    playing_during_probe: bool,
    disconnected: bool,
    require_inbound: bool,
    inbound_observed: bool,
    members_before: int,
    members_after: int,
) -> str:
    failures: list[str] = []
    if not connect_perm:
        failures.append("missing_connect_permission")
    if not speak_perm:
        failures.append("missing_speak_permission")
    if not accepted_audio_source:
        failures.append("mixer_not_audio_source")
    if not playing_during_probe:
        failures.append("playback_not_observed")
    if not disconnected:
        failures.append("disconnect_failed")
    if require_inbound and not inbound_observed:
        if max(members_before, members_after) <= 1:
            failures.append("inbound_required_but_no_other_members")
        else:
            failures.append("inbound_required_but_no_frames")
    return ",".join(failures)


def _build_evidence_context(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "config_snapshot": {
            "guild_id_configured": _positive_int(getattr(args, "guild_id", "")) is not None,
            "text_channel_id_configured": _positive_int(getattr(args, "text_channel_id", "")) is not None,
            "voice_channel_id_configured": _positive_int(getattr(args, "voice_channel_id", "")) is not None,
            "voice_channel_name": str(getattr(args, "voice_channel_name", "") or ""),
            "wait_seconds": max(0.0, float(getattr(args, "wait_seconds", 0.0) or 0.0)),
            "require_inbound": bool(getattr(args, "require_inbound", False)),
        },
        "env_presence": {
            "DISCORD_BOT_TOKEN": bool(os.environ.get("DISCORD_BOT_TOKEN")),
            "DISCORD_GUILD_ID": bool(os.environ.get("DISCORD_GUILD_ID")),
            "DISCORD_HOME_CHANNEL": bool(os.environ.get("DISCORD_HOME_CHANNEL")),
            "DISCORD_VOICE_CHANNEL_ID": bool(os.environ.get("DISCORD_VOICE_CHANNEL_ID")),
            "DISCORD_VOICE_CHANNEL_NAME": bool(os.environ.get("DISCORD_VOICE_CHANNEL_NAME")),
            "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
            "HERMES_OPENAI_REALTIME_API_KEY": bool(os.environ.get("HERMES_OPENAI_REALTIME_API_KEY")),
        },
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


async def _resolve_guild(client: Any, args: argparse.Namespace) -> Any:
    guild_id = _positive_int(getattr(args, "guild_id", ""))
    if guild_id is not None:
        guild = client.get_guild(guild_id)
        if guild is not None:
            return guild
    text_channel_id = _positive_int(getattr(args, "text_channel_id", ""))
    if text_channel_id is not None:
        channel = client.get_channel(text_channel_id) or await client.fetch_channel(text_channel_id)
        guild = getattr(channel, "guild", None)
        if guild is not None:
            return guild
    if len(client.guilds) == 1:
        return client.guilds[0]
    return None


def _resolve_voice_channel(guild: Any, args: argparse.Namespace) -> Any:
    voice_channel_id = _positive_int(getattr(args, "voice_channel_id", ""))
    if voice_channel_id is not None:
        channel = guild.get_channel(voice_channel_id)
        if channel is not None:
            return channel
    voice_channel_name = str(getattr(args, "voice_channel_name", "") or "").strip()
    for channel in getattr(guild, "voice_channels", []) or []:
        if voice_channel_name and str(getattr(channel, "name", "") or "") != voice_channel_name:
            continue
        return channel
    return None


def _positive_int(value: Any) -> int | None:
    try:
        result = int(str(value or "").strip())
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _load_opus(discord: Any) -> bool:
    try:
        if discord.opus.is_loaded():
            return True
        opus_path = ctypes.util.find_library("opus")
        if not opus_path:
            for path in ("/opt/homebrew/lib/libopus.dylib", "/usr/local/lib/libopus.dylib"):
                if os.path.isfile(path):
                    opus_path = path
                    break
        if opus_path:
            discord.opus.load_opus(opus_path)
        return bool(discord.opus.is_loaded())
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
