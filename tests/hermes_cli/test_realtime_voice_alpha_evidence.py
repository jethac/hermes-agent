import json
import sys
import types

from agent.realtime_voice_smoke_report import (
    ALPHA_REQUIRED_AUDIO_FIXTURES,
    ALPHA_REQUIRED_BARGE_IN_TEXTS,
    ALPHA_REQUIRED_TTS_METADATA,
    ALPHA_REQUIRED_TTS_TEXTS,
)
from hermes_cli import realtime_voice_alpha_evidence


def _write_required_audio_fixtures(root):
    for fixture in ALPHA_REQUIRED_AUDIO_FIXTURES:
        fixture_path = root / fixture
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_bytes(b"fixture")


def _fake_doctor_module(run_doctor):
    return types.SimpleNamespace(
        _realtime_voice_smoke_config=lambda: object(),
        run_doctor=run_doctor,
    )


class _FakeSidecarProcess:
    def __init__(self):
        self.alive = True
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminated = True
        self.alive = False

    def wait(self, timeout=None):
        self.waited = True
        return 0

    def kill(self):
        self.killed = True
        self.alive = False


def _fake_web_server_module(
    *,
    autostart=False,
    healthy=True,
    proc=None,
    ensure_error=None,
    status_payload=None,
):
    module = types.SimpleNamespace()
    module._VOICE_SIDECAR_PROC = proc
    module.load_env = lambda: {}
    module._realtime_voice_config_dict = lambda: {"frontend_provider": "reference"}
    module._realtime_voice_sidecar_base_url = lambda realtime: "http://127.0.0.1:8765"
    module._realtime_voice_should_autostart_sidecar = lambda realtime, base_url: autostart
    module._realtime_voice_sidecar_token = lambda realtime, env_on_disk: ""
    module.ensure_calls = []
    module._healthy_values = list(healthy if isinstance(healthy, list) else [healthy])

    def fake_healthy(base_url, token=""):
        if module._healthy_values:
            return module._healthy_values.pop(0)
        return bool(healthy)

    def fake_ensure(realtime):
        module.ensure_calls.append(realtime)
        if ensure_error is not None:
            raise ensure_error
        if module._VOICE_SIDECAR_PROC is None:
            module._VOICE_SIDECAR_PROC = _FakeSidecarProcess()

    module._realtime_voice_sidecar_healthy = fake_healthy
    module._ensure_realtime_voice_sidecar = fake_ensure
    module._realtime_voice_status_payload = lambda: status_payload or {
        "enabled": True,
        "available": True,
        "unavailable_reason": None,
        "conversation_quality": {
            "live_like": True,
            "mode": "streaming_text",
            "reason": "streaming_stt_tts",
        },
    }
    return module


def _install_fake_web_server(monkeypatch, module=None):
    import hermes_cli

    fake = module or _fake_web_server_module()
    monkeypatch.setitem(sys.modules, "hermes_cli.web_server", fake)
    monkeypatch.setattr(hermes_cli, "web_server", fake, raising=False)
    return fake


def _valid_alpha_report():
    entries = [
        {
            "kind": "manifest",
            "ok": True,
            "available": True,
            "conversation_quality": {
                "live_like": True,
                "mode": "streaming_text",
                "reason": "streaming_stt_tts",
                "sidecar_verified": True,
            },
            "quality_targets_ms": {
                "audio_to_partial_transcript_ms": 300,
                "final_transcript_to_first_text_ms": 500,
                "final_transcript_to_first_audio_ms": 900,
                "barge_in_ack_ms": 150,
            },
            "sidecar": {
                "healthy": True,
                "health": {
                    "capabilities": {
                        "streaming_stt": True,
                        "streaming_tts": True,
                        "tts": True,
                        "native_s2s": False,
                        "output_languages": ["en", "ja"],
                    },
                }
            },
        },
        {
            "kind": "protocol",
            "ok": True,
            "ready_ms": 12,
            "transcript_final_ms": 25,
            "events": ["frontend.state", "transcript.final"],
        },
    ]
    for fixture in ALPHA_REQUIRED_AUDIO_FIXTURES:
        entries.append(
            {
                "kind": "audio_fixture",
                "ok": True,
                "fixture": fixture,
                "codec": "webm_opus",
                "audio_bytes": 1234,
                "transcript_partial_ms": 90,
                "transcript_final_ms": 180,
                "target_ms": 300,
                "events": ["frontend.state", "transcript.partial", "transcript.final"],
            }
        )
    for text in ALPHA_REQUIRED_TTS_TEXTS:
        metadata = ALPHA_REQUIRED_TTS_METADATA[text]
        entries.append(
            {
                "kind": "tts",
                "ok": True,
                "text": text,
                **metadata,
                "first_audio_ms": 250,
                "target_ms": 900,
                "output_audio_bytes": 4321,
                "events": ["frontend.state", "audio.output.chunk"],
            }
        )
    for text in ALPHA_REQUIRED_BARGE_IN_TEXTS:
        entries.append(
            {
                "kind": "barge_in",
                "ok": True,
                "text": text,
                "barge_in_ack_ms": 45,
                "audio_after_barge_in_bytes": 0,
                "target_ms": 150,
                "events": ["frontend.state", "barge_in"],
            }
        )
    return entries


def test_alpha_evidence_runner_collects_and_validates_runs(monkeypatch, tmp_path, capsys):
    calls = []
    reports_dir = tmp_path / "reports"
    _write_required_audio_fixtures(tmp_path)
    monkeypatch.chdir(tmp_path)
    _install_fake_web_server(monkeypatch)

    def fake_run_doctor(args):
        calls.append(args)
        with open(args.realtime_voice_report, "w", encoding="utf-8") as handle:
            json.dump(_valid_alpha_report(), handle, ensure_ascii=False)

    monkeypatch.setitem(
        __import__("sys").modules,
        "hermes_cli.doctor",
        _fake_doctor_module(fake_run_doctor),
    )

    result = realtime_voice_alpha_evidence.main(
        [
            "--output-dir",
            str(reports_dir),
            "--runs",
            "3",
            "--prefix",
            "alpha",
        ]
    )

    assert result == 0
    assert [path.name for path in sorted(reports_dir.glob("*.json"))] == [
        "alpha-001.json",
        "alpha-002.json",
        "alpha-003.json",
    ]
    assert len(calls) == 3
    assert all(call.realtime_voice is True for call in calls)
    assert all(call.realtime_voice_alpha is True for call in calls)
    assert all(call.realtime_voice_audio_codec == "webm_opus" for call in calls)
    output = capsys.readouterr().out
    assert "Realtime voice alpha evidence OK" in output
    assert "audio_to_partial_transcript" in output


def test_alpha_evidence_runner_apply_updates_production_evidence_report(
    monkeypatch,
    tmp_path,
    capsys,
):
    calls = []
    saved = {}
    reports_dir = tmp_path / "reports"
    _write_required_audio_fixtures(tmp_path)
    monkeypatch.chdir(tmp_path)
    _install_fake_web_server(monkeypatch)
    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {"voice": {"realtime": {"enabled": True}}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")

    def fake_run_doctor(args):
        calls.append(args)
        with open(args.realtime_voice_report, "w", encoding="utf-8") as handle:
            json.dump(_valid_alpha_report(), handle, ensure_ascii=False)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.doctor",
        _fake_doctor_module(fake_run_doctor),
    )

    result = realtime_voice_alpha_evidence.main(
        [
            "--output-dir",
            str(reports_dir),
            "--runs",
            "3",
            "--prefix",
            "alpha",
            "--apply",
        ]
    )

    assert result == 0
    assert len(calls) == 3
    assert saved["config"]["voice"]["realtime"]["enabled"] is True
    assert saved["config"]["voice"]["realtime"]["production_evidence_report"] == str(reports_dir)
    assert "Updated realtime voice production_evidence_report" in capsys.readouterr().out


def test_alpha_evidence_runner_refuses_to_overwrite_existing_report(tmp_path, capsys):
    (tmp_path / "alpha-001.json").write_text("[]", encoding="utf-8")

    result = realtime_voice_alpha_evidence.main(
        [
            "--output-dir",
            str(tmp_path),
            "--runs",
            "1",
            "--prefix",
            "alpha",
        ]
    )

    assert result == 1
    assert "already exists" in capsys.readouterr().err


def test_alpha_evidence_runner_reports_missing_realtime_smoke_config(
    monkeypatch,
    tmp_path,
    capsys,
):
    calls = []
    reports_dir = tmp_path / "reports"
    _write_required_audio_fixtures(tmp_path)
    monkeypatch.chdir(tmp_path)

    def fake_config():
        raise RuntimeError("realtime voice smoke requires voice.realtime.sidecar_base_url")

    def fake_run_doctor(args):
        calls.append(args)

    monkeypatch.setitem(
        __import__("sys").modules,
        "hermes_cli.doctor",
        types.SimpleNamespace(
            _realtime_voice_smoke_config=fake_config,
            run_doctor=fake_run_doctor,
        ),
    )

    result = realtime_voice_alpha_evidence.main(
        [
            "--output-dir",
            str(reports_dir),
            "--runs",
            "1",
            "--prefix",
            "alpha",
        ]
    )

    assert result == 1
    assert calls == []
    assert not reports_dir.exists()
    error = capsys.readouterr().err
    assert "realtime voice smoke is not configured" in error
    assert "sidecar_base_url" in error
    assert "realtime_voice_profile --preset deepgram --apply --generate-bridge-token" in error
    assert "realtime_voice_deepgram_bridge --check --strict --production-en-ja" in error


def test_alpha_evidence_runner_reports_missing_fixtures(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.chdir(tmp_path)

    def fake_run_doctor(args):
        calls.append(args)

    monkeypatch.setitem(
        __import__("sys").modules,
        "hermes_cli.doctor",
        types.SimpleNamespace(run_doctor=fake_run_doctor),
    )

    result = realtime_voice_alpha_evidence.main(
        [
            "--output-dir",
            str(tmp_path / "reports"),
            "--runs",
            "1",
            "--prefix",
            "alpha",
        ]
    )

    assert result == 1
    assert calls == []
    error = capsys.readouterr().err
    assert "missing required audio fixture" in error
    for fixture in ALPHA_REQUIRED_AUDIO_FIXTURES:
        assert fixture in error
    assert "realtime_voice_fixture_pack --output-dir ./fixtures/realtime-voice" in error


def test_alpha_evidence_runner_returns_validation_failure(monkeypatch, tmp_path, capsys):
    _write_required_audio_fixtures(tmp_path)
    monkeypatch.chdir(tmp_path)
    _install_fake_web_server(monkeypatch)
    saved = {}
    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))

    def fake_run_doctor(args):
        report = _valid_alpha_report()
        report = [entry for entry in report if entry.get("kind") != "barge_in"]
        with open(args.realtime_voice_report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False)

    monkeypatch.setitem(
        __import__("sys").modules,
        "hermes_cli.doctor",
        _fake_doctor_module(fake_run_doctor),
    )

    result = realtime_voice_alpha_evidence.main(
        [
            "--output-dir",
            str(tmp_path),
            "--runs",
            "1",
            "--prefix",
            "alpha",
            "--apply",
        ]
    )

    assert result == 1
    assert saved == {}
    assert "missing required text" in capsys.readouterr().err


def test_alpha_evidence_runner_starts_and_stops_managed_sidecar(monkeypatch, tmp_path):
    _write_required_audio_fixtures(tmp_path)
    monkeypatch.chdir(tmp_path)
    proc = _FakeSidecarProcess()
    fake_web_server = _install_fake_web_server(
        monkeypatch,
        _fake_web_server_module(autostart=True, healthy=False, proc=proc),
    )

    def fake_run_doctor(args):
        with open(args.realtime_voice_report, "w", encoding="utf-8") as handle:
            json.dump(_valid_alpha_report(), handle, ensure_ascii=False)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.doctor",
        _fake_doctor_module(fake_run_doctor),
    )

    result = realtime_voice_alpha_evidence.main(
        [
            "--output-dir",
            str(tmp_path / "reports"),
            "--runs",
            "1",
            "--prefix",
            "alpha",
        ]
    )

    assert result == 0
    assert len(fake_web_server.ensure_calls) == 1
    assert proc.terminated is True
    assert proc.waited is True
    assert proc.killed is False


def test_alpha_evidence_runner_leaves_existing_healthy_sidecar_running(monkeypatch, tmp_path):
    _write_required_audio_fixtures(tmp_path)
    monkeypatch.chdir(tmp_path)
    proc = _FakeSidecarProcess()
    fake_web_server = _install_fake_web_server(
        monkeypatch,
        _fake_web_server_module(autostart=True, healthy=True, proc=proc),
    )

    def fake_run_doctor(args):
        with open(args.realtime_voice_report, "w", encoding="utf-8") as handle:
            json.dump(_valid_alpha_report(), handle, ensure_ascii=False)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.doctor",
        _fake_doctor_module(fake_run_doctor),
    )

    result = realtime_voice_alpha_evidence.main(
        [
            "--output-dir",
            str(tmp_path / "reports"),
            "--runs",
            "1",
            "--prefix",
            "alpha",
        ]
    )

    assert result == 0
    assert fake_web_server.ensure_calls == []
    assert proc.terminated is False


def test_alpha_evidence_runner_reports_managed_sidecar_start_failure(
    monkeypatch,
    tmp_path,
    capsys,
):
    _write_required_audio_fixtures(tmp_path)
    monkeypatch.chdir(tmp_path)
    _install_fake_web_server(
        monkeypatch,
        _fake_web_server_module(
            autostart=True,
            healthy=False,
            ensure_error=RuntimeError("sidecar exited with code 1"),
        ),
    )

    calls = []

    def fake_run_doctor(args):
        calls.append(args)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.doctor",
        _fake_doctor_module(fake_run_doctor),
    )

    result = realtime_voice_alpha_evidence.main(
        [
            "--output-dir",
            str(tmp_path / "reports"),
            "--runs",
            "1",
            "--prefix",
            "alpha",
        ]
    )

    assert result == 1
    assert calls == []
    assert "managed sidecar is not ready" in capsys.readouterr().err


def test_alpha_evidence_runner_requires_live_like_status_before_collecting(
    monkeypatch,
    tmp_path,
    capsys,
):
    _write_required_audio_fixtures(tmp_path)
    monkeypatch.chdir(tmp_path)
    _install_fake_web_server(
        monkeypatch,
        _fake_web_server_module(
            status_payload={
                "enabled": True,
                "available": False,
                "unavailable_reason": "live_like_required",
                "conversation_quality": {
                    "live_like": False,
                    "mode": "turn_based_text",
                    "reason": "utterance_stt_tts",
                },
            },
        ),
    )

    calls = []

    def fake_run_doctor(args):
        calls.append(args)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.doctor",
        _fake_doctor_module(fake_run_doctor),
    )

    result = realtime_voice_alpha_evidence.main(
        [
            "--output-dir",
            str(tmp_path / "reports"),
            "--runs",
            "1",
            "--prefix",
            "alpha",
        ]
    )

    assert result == 1
    assert calls == []
    error = capsys.readouterr().err
    assert "live-like realtime voice is not ready" in error
    assert "realtime_voice_profile --preset deepgram --apply --generate-bridge-token" in error
    assert "realtime_voice_deepgram_bridge --check --strict --production-en-ja" in error
    assert "realtime_voice_fixture_pack --output-dir ./fixtures/realtime-voice" in error
    assert "realtime_voice_alpha_evidence --runs 3 --apply" in error
