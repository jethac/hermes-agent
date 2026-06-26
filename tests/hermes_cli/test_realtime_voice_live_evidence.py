import asyncio
import json
from dataclasses import dataclass

from hermes_cli import realtime_voice_live_evidence


@dataclass(frozen=True)
class _FakeProbeResult:
    ok: bool
    error: str = ""


def test_live_evidence_collects_loopback_and_readiness_reports(monkeypatch, tmp_path):
    async def fake_loopback():
        return _FakeProbeResult(ok=True)

    async def fake_live(_args):
        return _FakeProbeResult(ok=False, error="DISCORD_BOT_TOKEN is required")

    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", fake_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", fake_live)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_OPENAI_REALTIME_API_KEY", raising=False)

    args = realtime_voice_live_evidence.build_parser().parse_args(["--output-dir", str(tmp_path)])
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    assert result.ok is True
    assert result.issues == []
    assert (tmp_path / "discord-loopback.json").is_file()
    assert (tmp_path / "discord-live-probe.json").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ok"] is True
    assert manifest["reports"]["discord_loopback"].endswith("discord-loopback.json")
    assert manifest["evidence_context"]["env_presence"]["OPENAI_API_KEY"] is False


def test_live_evidence_strict_mode_requires_live_discord_and_openai(monkeypatch, tmp_path):
    async def fake_loopback():
        return _FakeProbeResult(ok=True)

    async def fake_live(_args):
        return _FakeProbeResult(ok=False, error="DISCORD_BOT_TOKEN is required")

    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", fake_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", fake_live)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_OPENAI_REALTIME_API_KEY", raising=False)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--require-live-discord",
            "--require-openai-realtime",
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    assert result.ok is False
    assert "discord_live_probe: DISCORD_BOT_TOKEN is required" in result.issues
    assert "openai_realtime: OPENAI_API_KEY or HERMES_OPENAI_REALTIME_API_KEY is required" in result.issues
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ok"] is False
    assert manifest["require_live_discord"] is True
    assert manifest["require_openai_realtime"] is True


def test_live_evidence_main_returns_nonzero_when_strict_requirements_fail(monkeypatch, tmp_path, capsys):
    async def fake_loopback():
        return _FakeProbeResult(ok=True)

    async def fake_live(_args):
        return _FakeProbeResult(ok=False, error="DISCORD_BOT_TOKEN is required")

    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", fake_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", fake_live)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_OPENAI_REALTIME_API_KEY", raising=False)

    exit_code = realtime_voice_live_evidence.main(
        [
            "--output-dir",
            str(tmp_path),
            "--require-live-discord",
        ]
    )

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
