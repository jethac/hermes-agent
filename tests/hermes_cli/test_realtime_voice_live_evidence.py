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
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_GEMINI_LIVE_API_KEY", raising=False)

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
    assert manifest["evidence_context"]["env_presence"]["GEMINI_API_KEY"] is False


def test_live_evidence_manifest_references_optional_sidecar_and_turn_evidence(monkeypatch, tmp_path):
    async def fake_loopback():
        return _FakeProbeResult(ok=True)

    async def fake_live(_args):
        return _FakeProbeResult(ok=True)

    sidecar_path = tmp_path / "sidecar-session.json"
    live_turn_path = tmp_path / "live-turn.json"
    sidecar_path.write_text(json.dumps({"sidecar_running": True}), encoding="utf-8")
    live_turn_path.write_text(json.dumps({"transcript_observed": True}), encoding="utf-8")
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", fake_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", fake_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--sidecar-session-evidence",
            str(sidecar_path),
            "--live-turn-evidence",
            str(live_turn_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    assert result.ok is True
    assert result.reports["sidecar_session"] == str(sidecar_path)
    assert result.reports["live_turn"] == str(live_turn_path)
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reports"]["sidecar_session"] == str(sidecar_path)
    assert manifest["reports"]["live_turn"] == str(live_turn_path)


def test_live_evidence_manifest_rejects_invalid_optional_evidence(monkeypatch, tmp_path):
    async def fake_loopback():
        return _FakeProbeResult(ok=True)

    async def fake_live(_args):
        return _FakeProbeResult(ok=True)

    bad_turn_path = tmp_path / "live-turn.json"
    bad_turn_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", fake_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", fake_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--sidecar-session-evidence",
            str(tmp_path / "missing-sidecar.json"),
            "--live-turn-evidence",
            str(bad_turn_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    assert result.ok is False
    assert "sidecar_session: evidence file not found" in result.issues
    assert "live_turn: evidence root must be an object" in result.issues
    assert "sidecar_session" not in result.reports
    assert "live_turn" not in result.reports


def test_live_evidence_strict_mode_requires_live_discord_and_openai(monkeypatch, tmp_path):
    async def fake_loopback():
        return _FakeProbeResult(ok=True)

    async def fake_live(_args):
        return _FakeProbeResult(ok=False, error="DISCORD_BOT_TOKEN is required")

    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", fake_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", fake_live)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_OPENAI_REALTIME_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_GEMINI_LIVE_API_KEY", raising=False)

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


def test_live_evidence_strict_mode_requires_gemini_live(monkeypatch, tmp_path):
    async def fake_loopback():
        return _FakeProbeResult(ok=True)

    async def fake_live(_args):
        return _FakeProbeResult(ok=True)

    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", fake_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", fake_live)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_GEMINI_LIVE_API_KEY", raising=False)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--require-gemini-live",
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    assert result.ok is False
    assert "gemini_live: GEMINI_API_KEY or HERMES_GEMINI_LIVE_API_KEY is required" in result.issues
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ok"] is False
    assert manifest["require_gemini_live"] is True


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
