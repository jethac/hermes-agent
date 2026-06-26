"""Collect realtime voice live-evidence artifacts in one command."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeVoiceLiveEvidenceResult:
    ok: bool
    output_dir: str
    require_live_discord: bool = False
    require_openai_realtime: bool = False
    reports: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    evidence_context: dict[str, Any] = field(default_factory=dict)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Hermes realtime voice live-evidence artifacts")
    parser.add_argument(
        "--output-dir",
        default="./artifacts/realtime-voice-evidence/live-current",
        help="Directory where evidence JSON files will be written",
    )
    parser.add_argument(
        "--require-live-discord",
        action="store_true",
        help="Fail if the Discord live voice probe does not pass",
    )
    parser.add_argument(
        "--require-openai-realtime",
        action="store_true",
        help="Fail unless an OpenAI Realtime API key env var is present",
    )
    parser.add_argument("--guild-id", default=os.environ.get("DISCORD_GUILD_ID", ""))
    parser.add_argument("--text-channel-id", default=os.environ.get("DISCORD_HOME_CHANNEL", ""))
    parser.add_argument("--voice-channel-id", default=os.environ.get("DISCORD_VOICE_CHANNEL_ID", ""))
    parser.add_argument("--voice-channel-name", default=os.environ.get("DISCORD_VOICE_CHANNEL_NAME", "General"))
    parser.add_argument("--wait-seconds", type=float, default=2.0)
    parser.add_argument(
        "--require-inbound",
        action="store_true",
        help="Require inbound speech frames during the Discord live probe",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = asyncio.run(collect_realtime_voice_live_evidence(args))
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


async def collect_realtime_voice_live_evidence(args: argparse.Namespace) -> RealtimeVoiceLiveEvidenceResult:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, str] = {}
    issues: list[str] = []
    context = _evidence_context(args)

    loopback_report = output_dir / "discord-loopback.json"
    loopback_result = await _run_discord_loopback_smoke()
    _write_json(loopback_report, asdict(loopback_result))
    reports["discord_loopback"] = str(loopback_report)
    if not getattr(loopback_result, "ok", False):
        issues.append(f"discord_loopback: {getattr(loopback_result, 'error', '') or 'failed'}")

    live_report = output_dir / "discord-live-probe.json"
    live_result = await _run_discord_live_probe(args)
    _write_json(live_report, asdict(live_result))
    reports["discord_live_probe"] = str(live_report)
    if args.require_live_discord and not getattr(live_result, "ok", False):
        issues.append(f"discord_live_probe: {getattr(live_result, 'error', '') or 'failed'}")

    if args.require_openai_realtime and not _openai_realtime_key_present():
        issues.append("openai_realtime: OPENAI_API_KEY or HERMES_OPENAI_REALTIME_API_KEY is required")

    result = RealtimeVoiceLiveEvidenceResult(
        ok=not issues,
        output_dir=str(output_dir),
        require_live_discord=bool(args.require_live_discord),
        require_openai_realtime=bool(args.require_openai_realtime),
        reports=reports,
        issues=issues,
        evidence_context=context,
    )
    _write_json(output_dir / "manifest.json", asdict(result))
    return result


async def _run_discord_loopback_smoke() -> Any:
    from hermes_cli.discord_realtime_voice_smoke import run_discord_realtime_voice_smoke

    return await run_discord_realtime_voice_smoke()


async def _run_discord_live_probe(args: argparse.Namespace) -> Any:
    from hermes_cli.discord_voice_live_probe import run_discord_voice_live_probe

    probe_args = argparse.Namespace(
        guild_id=str(getattr(args, "guild_id", "") or ""),
        text_channel_id=str(getattr(args, "text_channel_id", "") or ""),
        voice_channel_id=str(getattr(args, "voice_channel_id", "") or ""),
        voice_channel_name=str(getattr(args, "voice_channel_name", "") or "General"),
        wait_seconds=float(getattr(args, "wait_seconds", 2.0) or 0.0),
        require_inbound=bool(getattr(args, "require_inbound", False)),
        report="",
    )
    return await run_discord_voice_live_probe(probe_args)


def _evidence_context(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "config_snapshot": {
            "guild_id_configured": _configured(getattr(args, "guild_id", "")),
            "text_channel_id_configured": _configured(getattr(args, "text_channel_id", "")),
            "voice_channel_id_configured": _configured(getattr(args, "voice_channel_id", "")),
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


def _openai_realtime_key_present() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("HERMES_OPENAI_REALTIME_API_KEY"))


def _configured(value: Any) -> bool:
    return bool(str(value or "").strip())


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
