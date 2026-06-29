import asyncio
import json
from dataclasses import dataclass

from hermes_cli import realtime_voice_live_evidence


@dataclass(frozen=True)
class _FakeProbeResult:
    ok: bool
    error: str = ""


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _complete_discord_probe():
    return {
        "kind": "discord_live_probe",
        "ok": True,
        "connect_perm": True,
        "speak_perm": True,
        "connected": True,
        "opus_loaded": True,
        "accepted_audio_source": True,
        "played": True,
        "playing_during_probe": True,
        "receiver_started": True,
        "receiver_frames": 18,
        "receiver_speech_start": 1,
        "inbound_observed": True,
        "disconnected": True,
        "require_inbound": True,
    }


def _complete_sidecar_session():
    return {
        "kind": "sidecar_session",
        "sidecar_running": True,
        "sidecar_healthy": True,
        "session_started": True,
        "session_closed": True,
        "fallback_mode_visible": True,
        "shutdown_bounded": True,
        "shutdown_timed_out": False,
        "latency_metrics_ms": {"shutdown_ms": 80},
    }


def _complete_live_turn():
    return {
        "kind": "live_turn",
        "transcript_observed": True,
        "assistant_audio_observed": True,
        "barge_in_observed": True,
        "spoken_reply_short": True,
        "no_voice_denial_observed": True,
        "speech_end_to_first_audio_ms": 950,
        "barge_in_stop_ms": 80,
    }


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
    assert result.live_probe_ok is False
    assert result.live_probe_status == "failed"
    assert result.warnings == ["discord_live_probe: DISCORD_BOT_TOKEN is required"]
    assert (tmp_path / "discord-loopback.json").is_file()
    assert (tmp_path / "discord-live-probe.json").is_file()
    loopback = json.loads((tmp_path / "discord-loopback.json").read_text(encoding="utf-8"))
    live_probe = json.loads((tmp_path / "discord-live-probe.json").read_text(encoding="utf-8"))
    assert loopback["kind"] == "discord_loopback"
    assert loopback["source_artifact"] == str(tmp_path / "discord-loopback.json")
    assert live_probe["kind"] == "discord_live_probe"
    assert live_probe["source_artifact"] == str(tmp_path / "discord-live-probe.json")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "voiceops.realtime_voice_live_evidence_manifest.v1"
    assert manifest["ok"] is True
    assert manifest["live_probe_ok"] is False
    assert manifest["live_probe_status"] == "failed"
    assert manifest["warnings"] == ["discord_live_probe: DISCORD_BOT_TOKEN is required"]
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
    _write_json(sidecar_path, _complete_sidecar_session())
    _write_json(live_turn_path, _complete_live_turn())
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
    assert result.live_probe_ok is True
    assert result.live_probe_status == "passed"
    assert result.reports["sidecar_session"] == str(sidecar_path)
    assert result.reports["live_turn"] == str(live_turn_path)
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reports"]["sidecar_session"] == str(sidecar_path)
    assert manifest["reports"]["live_turn"] == str(live_turn_path)


def test_live_evidence_validate_mode_does_not_call_discord_probes(monkeypatch, tmp_path):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run in validation mode")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run in validation mode")

    discord_path = _write_json(tmp_path / "discord-live-probe.json", _complete_discord_probe())
    sidecar_path = _write_json(tmp_path / "sidecar-session.json", _complete_sidecar_session())
    live_turn_path = _write_json(tmp_path / "live-turn.json", _complete_live_turn())
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--validate-live-evidence",
            "--discord-live-probe-evidence",
            str(discord_path),
            "--sidecar-session-evidence",
            str(sidecar_path),
            "--live-turn-evidence",
            str(live_turn_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    assert result.ok is True
    assert result.validate_live_evidence is True
    assert result.live_probe_ok is None
    assert result.live_probe_status == "not_run"
    assert result.issues == []
    assert result.reports == {
        "discord_live_probe": str(discord_path),
        "sidecar_session": str(sidecar_path),
        "live_turn": str(live_turn_path),
    }
    assert result.strict_validation["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert result.strict_validation["missing_gates"] == []
    assert not (tmp_path / "bundle" / "discord-loopback.json").exists()
    assert not (tmp_path / "bundle" / "discord-live-probe.json").exists()
    validation = json.loads((tmp_path / "bundle" / "live-evidence-validation.json").read_text(encoding="utf-8"))
    assert validation["schema_version"] == "voiceops.realtime_voice_live_evidence_validation.v1"


def test_live_evidence_validate_mode_surfaces_strict_ingester_issues(monkeypatch, tmp_path):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run in validation mode")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run in validation mode")

    bad_turn = _complete_live_turn()
    bad_turn["barge_in_stop_ms"] = 999
    bad_turn["assistant_text"] = "I cannot hear you in Discord voice."
    discord_path = _write_json(tmp_path / "discord-live-probe.json", _complete_discord_probe())
    sidecar_path = _write_json(tmp_path / "sidecar-session.json", _complete_sidecar_session())
    live_turn_path = _write_json(tmp_path / "live-turn.json", bad_turn)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--validate-live-evidence",
            "--discord-live-probe-evidence",
            str(discord_path),
            "--sidecar-session-evidence",
            str(sidecar_path),
            "--live-turn-evidence",
            str(live_turn_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    assert result.ok is False
    assert result.strict_validation["overall_status"] == "partial_live_evidence"
    assert "live_turn" in result.strict_validation["missing_gates"]
    assert "live_evidence_validation:live_turn:barge_in_stop_ms_over_target" in result.issues
    assert "live_evidence_validation:live_turn.assistant_text:forbidden_evidence_field" in result.issues
    assert "live_evidence_validation:live_turn.assistant_text:voice_capability_denial_text" in result.issues


def test_live_evidence_manifest_rejects_anonymous_optional_evidence(monkeypatch, tmp_path):
    async def fake_loopback():
        return _FakeProbeResult(ok=True)

    async def fake_live(_args):
        return _FakeProbeResult(ok=True)

    sidecar_path = tmp_path / "sidecar-session.json"
    sidecar_path.write_text(json.dumps({"sidecar_running": True}), encoding="utf-8")
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", fake_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", fake_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--sidecar-session-evidence",
            str(sidecar_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    assert result.ok is False
    assert "sidecar_session: evidence file must include kind, evidence_type, or live evidence schema" in result.issues
    assert "sidecar_session" not in result.reports


def test_live_evidence_manifest_rejects_example_only_optional_evidence(monkeypatch, tmp_path):
    async def fake_loopback():
        return _FakeProbeResult(ok=True)

    async def fake_live(_args):
        return _FakeProbeResult(ok=True)

    sidecar_path = tmp_path / "sidecar-session.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "kind": "sidecar_session",
                "example_only": True,
                "sidecar_running": True,
                "sidecar_healthy": True,
                "session_started": True,
                "session_closed": True,
                "fallback_mode_visible": True,
                "shutdown_bounded": True,
                "shutdown_timed_out": False,
                "latency_metrics_ms": {"shutdown_ms": 80},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", fake_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", fake_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--sidecar-session-evidence",
            str(sidecar_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    assert result.ok is False
    assert "sidecar_session: example_only evidence is not accepted" in result.issues
    assert "sidecar_session" not in result.reports


def test_live_evidence_manifest_rejects_incomplete_optional_evidence(monkeypatch, tmp_path):
    async def fake_loopback():
        return _FakeProbeResult(ok=True)

    async def fake_live(_args):
        return _FakeProbeResult(ok=True)

    sidecar_path = tmp_path / "sidecar-session.json"
    sidecar_path.write_text(json.dumps({"kind": "sidecar_session", "sidecar_running": True}), encoding="utf-8")
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", fake_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", fake_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--sidecar-session-evidence",
            str(sidecar_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    assert result.ok is False
    assert "sidecar_session: sidecar_healthy must be true" in result.issues
    assert "sidecar_session: latency_metrics_ms.shutdown_ms must be a non-negative number" in result.issues
    assert "sidecar_session" not in result.reports


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
