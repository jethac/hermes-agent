import asyncio
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from agent.realtime_voice_smoke_report import (
    ALPHA_REQUIRED_AUDIO_FIXTURE_TEXTS,
    ALPHA_REQUIRED_AUDIO_FIXTURES,
    ALPHA_REQUIRED_AUDIO_SESSION_FIXTURES,
    ALPHA_REQUIRED_BARGE_IN_TEXTS,
    ALPHA_REQUIRED_SESSION_TURN_METADATA,
    ALPHA_REQUIRED_SESSION_TURN_TEXTS,
    ALPHA_REQUIRED_TTS_METADATA,
    ALPHA_REQUIRED_TTS_TEXTS,
)
from hermes_cli import realtime_voice_live_evidence


@dataclass(frozen=True)
class _FakeProbeResult:
    ok: bool
    error: str = ""


@dataclass(frozen=True)
class _FakeDiscordLiveProbeResult:
    ok: bool = True
    error: str = ""
    connect_perm: bool = True
    speak_perm: bool = True
    connected: bool = True
    opus_loaded: bool = True
    accepted_audio_source: bool = True
    played: bool = True
    playing_during_probe: bool = True
    receiver_started: bool = True
    receiver_frames: int = 18
    receiver_speech_start: int = 1
    inbound_observed: bool = True
    disconnected: bool = True
    require_inbound: bool = True
    latency_metrics_ms: dict[str, int] = field(
        default_factory=lambda: {
            "connect_ms": 420,
            "playback_observed_ms": 180,
            "inbound_observed_ms": 900,
            "disconnect_ms": 120,
        }
    )


def _write_json(path, payload):
    if isinstance(payload, dict):
        attestation = payload.get("collector_attestation") or payload.get("collector_provenance")
        if isinstance(attestation, dict):
            attested_payload = dict(payload)
            attested_payload.pop("collector_attestation", None)
            attested_payload.pop("collector_provenance", None)
            raw = json.dumps(attested_payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
            payload_sha256 = hashlib.sha256(raw).hexdigest()
            attestation["raw_artifact_sha256"] = payload_sha256
            attestation["redacted_artifact_sha256"] = payload_sha256
            attestation["parent_manifest_sha256"] = payload_sha256
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _collector_attestation(section_name):
    return {
        "collector_name": "pytest.realtime_voice_live_evidence",
        "collector_version": "voiceops-live-fixture-v1",
        "run_id": f"pytest-{section_name}",
        "command_argv": ["pytest", section_name],
        "git_commit": "abc123def456",
        "started_at": "2026-06-29T00:00:00Z",
        "finished_at": "2026-06-29T00:00:01Z",
        "raw_artifact_sha256": "a" * 64,
        "redacted_artifact_sha256": "b" * 64,
        "parent_manifest_sha256": "c" * 64,
    }


def _complete_discord_probe():
    return {
        "kind": "discord_live_probe",
        "collector_attestation": _collector_attestation("discord_live_probe"),
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
        "latency_metrics_ms": {
            "connect_ms": 420,
            "playback_observed_ms": 180,
            "inbound_observed_ms": 900,
            "disconnect_ms": 120,
        },
    }


def _complete_sidecar_session():
    return {
        "kind": "sidecar_session",
        "collector_attestation": _collector_attestation("sidecar_session"),
        "sidecar_running": True,
        "sidecar_healthy": True,
        "session_started": True,
        "session_closed": True,
        "fallback_mode_visible": True,
        "fallback_reason": "none",
        "sidecar_mode": "production",
        "healthcheck_observed": True,
        "provider_transport_observed": True,
        "session_id_redacted": True,
        "shutdown_bounded": True,
        "shutdown_timed_out": False,
        "latency_metrics_ms": {"session_start_ms": 110, "shutdown_ms": 80},
    }


def _complete_live_turn():
    return {
        "kind": "live_turn",
        "collector_attestation": _collector_attestation("live_turn"),
        "turn_id": "voiceops-live-turn-budget",
        "audio_segment_ref": "artifact://redacted/voiceops-live-turn-budget.wav",
        "evidence_bundle_id": "kame-evidence-live-turn-budget",
        "evidence_merge_key": "kame-merge-live-turn-budget",
        "transcript_observed": True,
        "audio_segment_ref_observed": True,
        "interpreter_evidence_observed": True,
        "transcript_hypotheses_labeled": True,
        "witness_arrival_phases": ["before_raw_audio", "with_raw_audio", "after_interpreter_start"],
        "interpreter_input_order": ["raw_audio", "metadata", "reflex", "transcript_hypotheses"],
        "transcript_hypotheses": [
            {
                "kind": "frontend_witness_hypothesis",
                "source": "moshi",
                "text": "[redacted witness hypothesis]",
                "arrival_phase": "with_raw_audio",
                "authority": "hypothesis",
                "tool_authority": False,
            }
        ],
        "interpreter_adjudication_outcomes": ["corrected_by_audio"],
        "promoted_evidence_authority": {
            "interpreter_corrected_transcript": "interpreter_promoted",
            "interpreter_normalized_intent": "interpreter_promoted",
        },
        "assistant_audio_observed": True,
        "barge_in_observed": True,
        "spoken_reply_short": True,
        "no_voice_denial_observed": True,
        "speech_end_to_first_audio_ms": 950,
        "barge_in_stop_ms": 80,
    }


def _alpha_realtime_voice_report(
    *,
    include_discord: bool = False,
    invalid: bool = False,
    sidecar_mode: str = "production",
    bridge_closed: bool = True,
    denial_text: bool = False,
):
    entries = [
        {
            "kind": "manifest",
            "ok": True,
            "run_id": "voiceops-test-run-0001",
            "collected_at": "2026-06-29T00:00:00Z",
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
                "mode": sidecar_mode,
                "healthy": True,
                "health": {
                    "ok": True,
                    "capabilities": {
                        "streaming_stt": True,
                        "streaming_tts": True,
                        "tts": True,
                        "native_s2s": False,
                        "output_languages": ["en", "ja"],
                    },
                },
            },
        },
        {
            "kind": "protocol",
            "ok": True,
            "ready_ms": 12,
            "transcript_final_ms": 25,
            "events": ["frontend.state", "transcript.final"],
        },
        {
            "kind": "discord_bridge",
            "ok": True,
            "mode": "production",
            "transport": "websocket",
            "sidecar_closed": bridge_closed,
            "shutdown_elapsed_ms": 80,
            "shutdown_bounded": True,
            "shutdown_timed_out": False,
            "events": ["session.started", "session.closed", "barge_in"],
        },
    ]
    for text in ALPHA_REQUIRED_SESSION_TURN_TEXTS:
        assistant_text = "I cannot hear you in Discord voice." if denial_text else text
        entries.append(
            {
                "kind": "session_turn",
                "ok": True,
                "turn_id": "voiceops-report-turn-001",
                "audio_segment_ref": "artifact://redacted/voiceops-report-turn-001.wav",
                "evidence_bundle_id": "kame-evidence-voiceops-report-turn-001",
                "evidence_merge_key": "kame-merge-voiceops-report-turn-001",
                "speaker": {
                    "platform": "discord",
                    "channel_user_id": "jetha-redacted",
                },
                "channel": {
                    "transport": "discord_voice",
                    "channel_id": "general-redacted",
                },
                "audio_segment_ref_observed": True,
                "interpreter_evidence_observed": True,
                "transcript_hypotheses_labeled": True,
                "witness_arrival_phases": ["before_raw_audio", "with_raw_audio"],
                "transcript_hypotheses": [
                    {
                        "kind": "frontend_witness_hypothesis",
                        "source": "moshi",
                        "text": "redacted hypothesis",
                        "arrival_phase": "after_interpreter_start",
                        "authority": "hypothesis",
                        "tool_authority": False,
                        "speaker_guess": {
                            "platform": "discord",
                            "channel_user_id": "jetha-redacted",
                        },
                        "channel_guess": {
                            "transport": "discord_voice",
                            "channel_id": "general-redacted",
                        },
                    }
                ],
                "interpreter_input_order": ["raw_audio", "metadata", "reflex", "transcript_hypotheses"],
                "interpreter_adjudication_outcomes": ["corrected_by_audio"],
                "promoted_evidence_authority": {
                    "interpreter_corrected_transcript": "interpreter_promoted",
                    "interpreter_normalized_intent": "interpreter_promoted",
                },
                "text": assistant_text,
                **ALPHA_REQUIRED_SESSION_TURN_METADATA[text],
                "transcript_final_ms": 10,
                "first_text_ms": 90,
                "first_text_target_ms": 500,
                "first_audio_ms": 250,
                "first_audio_target_ms": 900,
                "output_audio_bytes": 4321,
                "events": ["session.started", "transcript.final", "assistant.text.partial", "audio.output.chunk"],
            }
        )
    for fixture in ALPHA_REQUIRED_AUDIO_FIXTURES:
        entries.append(
            {
                "kind": "audio_fixture",
                "ok": True,
                "fixture": fixture,
                "codec": "webm_opus",
                "audio_bytes": 1234,
                "final_text": ALPHA_REQUIRED_AUDIO_FIXTURE_TEXTS[fixture],
                "transcript_partial_ms": 90,
                "transcript_final_ms": 180,
                "target_ms": 300,
                "events": ["frontend.state", "transcript.partial", "transcript.final"],
            }
        )
    for fixture in ALPHA_REQUIRED_AUDIO_SESSION_FIXTURES:
        entries.append(
            {
                "kind": "audio_session",
                "ok": True,
                "turn_id": "voiceops-report-turn-001",
                "audio_segment_ref": "artifact://redacted/voiceops-report-turn-001.wav",
                "evidence_bundle_id": "kame-evidence-voiceops-report-turn-001",
                "evidence_merge_key": "kame-merge-voiceops-report-turn-001",
                "audio_segment_ref_observed": True,
                "interpreter_evidence_observed": True,
                "transcript_hypotheses_labeled": True,
                "fixture": fixture,
                "codec": "webm_opus",
                "audio_bytes": 1234,
                "final_text": ALPHA_REQUIRED_AUDIO_FIXTURE_TEXTS[fixture],
                "transcript_partial_ms": 90,
                "transcript_final_ms": 180,
                "target_ms": 300,
                "first_text_ms": 90,
                "first_text_target_ms": 500,
                "first_audio_ms": 250,
                "first_audio_target_ms": 900,
                "output_audio_bytes": 4321,
                "events": [
                    "session.started",
                    "frontend.state",
                    "transcript.partial",
                    "transcript.final",
                    "assistant.text.partial",
                    "audio.output.chunk",
                ],
            }
        )
    for text in ALPHA_REQUIRED_TTS_TEXTS:
        entries.append(
            {
                "kind": "tts",
                "ok": True,
                "text": text,
                **ALPHA_REQUIRED_TTS_METADATA[text],
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
    if include_discord:
        entries.append(_complete_discord_probe())
    if invalid:
        entries = [entry for entry in entries if entry.get("kind") != "barge_in"]
    return entries


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


def test_live_evidence_parser_accepts_realtime_voice_report_source(tmp_path):
    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--from-realtime-voice-report",
            str(tmp_path / "report.json"),
        ]
    )

    assert args.from_realtime_voice_report == tmp_path / "report.json"


def test_live_evidence_parser_accepts_realtime_voice_doctor_report_runner(tmp_path):
    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--run-realtime-voice-doctor-report",
            "--require-inbound",
            "--wait-seconds",
            "5",
            "--voice-channel-id",
            "voice-channel-ref-demo",
        ]
    )

    assert args.run_realtime_voice_doctor_report is True
    assert args.require_inbound is True
    assert args.wait_seconds == 5
    assert args.voice_channel_id == "voice-channel-ref-demo"


def test_live_evidence_closure_runs_doctor_report_derives_and_validates(monkeypatch, tmp_path):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run when closure mode runs doctor")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run directly when closure mode runs doctor")

    commands = []

    def fake_run(command, **_kwargs):
        if isinstance(command, list) and command[:8] == [
            "uv",
            "run",
            "--extra",
            "dev",
            "--extra",
            "voice",
            "hermes",
            "doctor",
        ]:
            commands.append(command)
            report_path = Path(command[-1])
            _write_json(report_path, _alpha_realtime_voice_report(include_discord=True))
            return subprocess.CompletedProcess(command, 0, stdout="doctor output redacted", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="git-ref", stderr="")

    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)
    monkeypatch.setattr(realtime_voice_live_evidence.subprocess, "run", fake_run)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--run-realtime-voice-doctor-report",
            "--require-inbound",
            "--wait-seconds",
            "5",
            "--voice-channel-id",
            "voice-channel-ref-demo",
            "--voice-channel-name",
            "General",
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    assert result.ok is True
    assert result.doctor_report["ok"] is True
    assert result.doctor_report["stdout_present"] is True
    assert result.doctor_report["stderr_present"] is False
    assert result.doctor_report["report_exists"] is True
    assert result.strict_validation["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert result.strict_validation["missing_gates"] == []
    assert result.reports["discord_live_probe"].endswith("discord-live-probe.from-realtime-report.json")
    assert result.reports["sidecar_session"].endswith("sidecar-session.from-realtime-report.json")
    assert result.reports["live_turn"].endswith("live-turn.from-realtime-report.json")
    assert len(commands) == 1
    command = commands[0]
    assert command[:8] == ["uv", "run", "--extra", "dev", "--extra", "voice", "hermes", "doctor"]
    assert "--discord-voice-live-probe-require-inbound" in command
    assert command[command.index("--discord-voice-live-probe-wait-seconds") + 1] == "5.0"
    assert command[command.index("--discord-voice-live-probe-channel-id") + 1] == "voice-channel-ref-demo"
    assert command[command.index("--discord-voice-live-probe-channel-name") + 1] == "General"
    assert command[-2:] == [
        "--realtime-voice-report",
        str(tmp_path / "bundle" / "realtime-voice-doctor-report.json"),
    ]


def test_live_evidence_closure_reports_doctor_failure_without_derivation(monkeypatch, tmp_path):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run when closure mode runs doctor")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run directly when closure mode runs doctor")

    def fake_run(command, **_kwargs):
        if isinstance(command, list) and command[:8] == [
            "uv",
            "run",
            "--extra",
            "dev",
            "--extra",
            "voice",
            "hermes",
            "doctor",
        ]:
            return subprocess.CompletedProcess(command, 17, stdout="", stderr="doctor error redacted")
        return subprocess.CompletedProcess(command, 0, stdout="git-ref", stderr="")

    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)
    monkeypatch.setattr(realtime_voice_live_evidence.subprocess, "run", fake_run)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--run-realtime-voice-doctor-report",
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    assert result.ok is False
    assert result.reports == {}
    assert result.doctor_report["returncode"] == 17
    assert result.doctor_report["stderr_present"] is True
    assert "realtime_voice_doctor_report: command exited 17" in result.issues
    assert "realtime_voice_doctor_report: report file was not written" in result.issues
    assert not (tmp_path / "bundle" / "live-evidence-validation.json").exists()


def test_live_evidence_manifest_references_optional_sidecar_and_turn_evidence(monkeypatch, tmp_path):
    async def fake_loopback():
        return _FakeProbeResult(ok=True)

    async def fake_live(_args):
        return _FakeDiscordLiveProbeResult()

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
    assert result.reports["sidecar_session"] == "sidecar-session.json"
    assert result.reports["live_turn"] == "live-turn.json"
    assert (tmp_path / "bundle" / "sidecar-session.json").is_file()
    assert (tmp_path / "bundle" / "live-turn.json").is_file()
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reports"]["sidecar_session"] == "sidecar-session.json"
    assert manifest["reports"]["live_turn"] == "live-turn.json"


def test_live_evidence_manifest_with_relative_output_dir_is_reingestable(monkeypatch, tmp_path):
    from scripts.voiceops_voice_operator import _load_live_evidence

    async def fake_loopback():
        return _FakeProbeResult(ok=True)

    async def fake_live(_args):
        return _FakeDiscordLiveProbeResult()

    monkeypatch.chdir(tmp_path)
    output_dir = "artifacts/realtime-voice-evidence/live-current"
    sidecar_path = _write_json(Path(output_dir) / "sidecar-session.json", _complete_sidecar_session())
    live_turn_path = _write_json(Path(output_dir) / "live-turn.json", _complete_live_turn())
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", fake_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", fake_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            output_dir,
            "--require-inbound",
            "--sidecar-session-evidence",
            str(sidecar_path),
            "--live-turn-evidence",
            str(live_turn_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    manifest_path = Path(output_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result.ok is True
    assert manifest["reports"]["discord_live_probe"] == "discord-live-probe.json"
    assert manifest["reports"]["discord_loopback"] == "discord-loopback.json"
    live_evidence = _load_live_evidence([manifest_path])
    assert live_evidence["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert "live_evidence_manifest:discord_live_probe:live_evidence_file_not_found" not in live_evidence["issues"]
    assert live_evidence["issues"] == []


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
        "discord_live_probe": "discord-live-probe.json",
        "sidecar_session": "sidecar-session.json",
        "live_turn": "live-turn.json",
    }
    assert (tmp_path / "bundle" / "discord-live-probe.json").is_file()
    assert (tmp_path / "bundle" / "sidecar-session.json").is_file()
    assert (tmp_path / "bundle" / "live-turn.json").is_file()
    assert result.strict_validation["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert result.strict_validation["missing_gates"] == []
    assert not (tmp_path / "bundle" / "discord-loopback.json").exists()
    validation = json.loads((tmp_path / "bundle" / "live-evidence-validation.json").read_text(encoding="utf-8"))
    assert validation["schema_version"] == "voiceops.realtime_voice_live_evidence_validation.v1"


def test_live_evidence_parser_accepts_audit_only(tmp_path):
    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--audit-only",
        ]
    )

    assert args.audit_only is True


def test_live_evidence_audit_only_passes_without_writing_or_calling_probes(monkeypatch, tmp_path, capsys):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run in audit-only mode")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run in audit-only mode")

    discord_path = _write_json(tmp_path / "discord-live-probe.json", _complete_discord_probe())
    sidecar_path = _write_json(tmp_path / "sidecar-session.json", _complete_sidecar_session())
    live_turn_path = _write_json(tmp_path / "live-turn.json", _complete_live_turn())
    output_dir = tmp_path / "bundle"
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    exit_code = realtime_voice_live_evidence.main(
        [
            "--output-dir",
            str(output_dir),
            "--audit-only",
            "--discord-live-probe-evidence",
            str(discord_path),
            "--sidecar-session-evidence",
            str(sidecar_path),
            "--live-turn-evidence",
            str(live_turn_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema_version"] == "voiceops.realtime_voice_live_evidence_audit.v1"
    assert payload["ok"] is True
    assert payload["artifact_writes"] is False
    assert payload["discord_probe_run"] is False
    assert payload["report_derivation_run"] is False
    assert payload["issues"] == []
    assert payload["strict_validation"]["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert payload["strict_validation"]["missing_gates"] == []
    assert not output_dir.exists()


def test_live_evidence_audit_only_accepts_manifest_without_writing_or_calling_probes(monkeypatch, tmp_path, capsys):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run in audit-only mode")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run in audit-only mode")

    evidence_dir = tmp_path / "evidence"
    _write_json(evidence_dir / "discord-live-probe.json", _complete_discord_probe())
    _write_json(evidence_dir / "sidecar-session.json", _complete_sidecar_session())
    _write_json(evidence_dir / "live-turn.json", _complete_live_turn())
    manifest_path = _write_json(
        evidence_dir / "manifest.json",
        {
            "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
            "reports": {
                "discord_live_probe": "discord-live-probe.json",
                "sidecar_session": "sidecar-session.json",
                "live_turn": "live-turn.json",
            },
        },
    )
    output_dir = tmp_path / "bundle"
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    exit_code = realtime_voice_live_evidence.main(
        [
            "--output-dir",
            str(output_dir),
            "--audit-only",
            "--live-evidence-manifest",
            str(manifest_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["artifact_writes"] is False
    assert payload["discord_probe_run"] is False
    assert payload["report_derivation_run"] is False
    assert payload["live_evidence_manifest"] == str(manifest_path)
    assert payload["reports"] == {}
    assert payload["issues"] == []
    assert payload["strict_validation"]["manifest"] == str(manifest_path)
    assert payload["strict_validation"]["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert payload["strict_validation"]["missing_gates"] == []
    assert not output_dir.exists()


def test_live_evidence_audit_only_reports_missing_manifest_without_noisy_section_failures(
    monkeypatch,
    tmp_path,
    capsys,
):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run in audit-only mode")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run in audit-only mode")

    manifest_path = tmp_path / "missing-manifest.json"
    output_dir = tmp_path / "bundle"
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    exit_code = realtime_voice_live_evidence.main(
        [
            "--output-dir",
            str(output_dir),
            "--audit-only",
            "--live-evidence-manifest",
            str(manifest_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["artifact_writes"] is False
    assert payload["issues"] == ["live_evidence_validation:live_evidence_manifest_not_found"]
    assert payload["strict_validation"] == {
        "schema_version": "voiceops.realtime_voice_live_evidence_validation.v1",
        "manifest": str(manifest_path),
        "loaded": False,
        "overall_status": "partial_live_evidence",
        "issues": ["live_evidence_manifest_not_found"],
        "section_refs": {},
        "missing_gates": ["discord_join", "discord_playback", "live_receiver", "live_turn", "production_sidecar"],
    }
    assert not any("discord_live_probe:" in issue for issue in payload["issues"])
    assert not output_dir.exists()


def test_live_evidence_validate_accepts_manifest_without_running_probes(monkeypatch, tmp_path):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run in validate mode")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run in validate mode")

    evidence_dir = tmp_path / "evidence"
    _write_json(evidence_dir / "discord-live-probe.json", _complete_discord_probe())
    _write_json(evidence_dir / "sidecar-session.json", _complete_sidecar_session())
    _write_json(evidence_dir / "live-turn.json", _complete_live_turn())
    manifest_path = _write_json(
        evidence_dir / "manifest.json",
        {
            "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
            "reports": {
                "discord_live_probe": "discord-live-probe.json",
                "sidecar_session": "sidecar-session.json",
                "live_turn": "live-turn.json",
            },
        },
    )
    output_dir = tmp_path / "bundle"
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    result = asyncio.run(
        realtime_voice_live_evidence.collect_realtime_voice_live_evidence(
            realtime_voice_live_evidence.build_parser().parse_args(
                [
                    "--output-dir",
                    str(output_dir),
                    "--validate-live-evidence",
                    "--live-evidence-manifest",
                    str(manifest_path),
                ]
            )
        )
    )

    validation = json.loads((output_dir / "live-evidence-validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result.ok is True
    assert result.reports == {"live_evidence_manifest": str(manifest_path)}
    assert result.strict_validation["manifest"] == str(manifest_path)
    assert validation["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert validation["missing_gates"] == []
    assert manifest["reports"] == {"live_evidence_manifest": str(manifest_path)}


def test_live_evidence_validate_reports_missing_manifest_without_noisy_section_failures(
    monkeypatch,
    tmp_path,
):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run in validate mode")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run in validate mode")

    manifest_path = tmp_path / "missing-manifest.json"
    output_dir = tmp_path / "bundle"
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    result = asyncio.run(
        realtime_voice_live_evidence.collect_realtime_voice_live_evidence(
            realtime_voice_live_evidence.build_parser().parse_args(
                [
                    "--output-dir",
                    str(output_dir),
                    "--validate-live-evidence",
                    "--live-evidence-manifest",
                    str(manifest_path),
                ]
            )
        )
    )

    validation = json.loads((output_dir / "live-evidence-validation.json").read_text(encoding="utf-8"))
    assert result.ok is False
    assert result.issues == ["live_evidence_validation:live_evidence_manifest_not_found"]
    assert result.strict_validation["issues"] == ["live_evidence_manifest_not_found"]
    assert result.strict_validation["missing_gates"] == [
        "discord_join",
        "discord_playback",
        "live_receiver",
        "live_turn",
        "production_sidecar",
    ]
    assert validation == result.strict_validation
    assert not any("discord_live_probe:" in issue for issue in result.issues)


def test_live_evidence_audit_only_reports_invalid_evidence_without_writing(monkeypatch, tmp_path, capsys):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run in audit-only mode")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run in audit-only mode")

    bad_turn = _complete_live_turn()
    bad_turn["barge_in_stop_ms"] = 999
    discord_path = _write_json(tmp_path / "discord-live-probe.json", _complete_discord_probe())
    sidecar_path = _write_json(tmp_path / "sidecar-session.json", _complete_sidecar_session())
    live_turn_path = _write_json(tmp_path / "live-turn.json", bad_turn)
    output_dir = tmp_path / "bundle"
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    exit_code = realtime_voice_live_evidence.main(
        [
            "--output-dir",
            str(output_dir),
            "--audit-only",
            "--discord-live-probe-evidence",
            str(discord_path),
            "--sidecar-session-evidence",
            str(sidecar_path),
            "--live-turn-evidence",
            str(live_turn_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["artifact_writes"] is False
    assert "live_evidence_validation:live_turn:barge_in_stop_ms_over_target" in payload["issues"]
    assert "live_turn" in payload["strict_validation"]["missing_gates"]
    assert not output_dir.exists()


def test_live_evidence_rejects_transcript_only_turn_without_kame_audio(monkeypatch, tmp_path, capsys):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run in audit-only mode")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run in audit-only mode")

    transcript_only_turn = _complete_live_turn()
    transcript_only_turn["audio_segment_ref_observed"] = False
    transcript_only_turn["interpreter_evidence_observed"] = False
    transcript_only_turn["transcript_hypotheses_labeled"] = True
    discord_path = _write_json(tmp_path / "discord-live-probe.json", _complete_discord_probe())
    sidecar_path = _write_json(tmp_path / "sidecar-session.json", _complete_sidecar_session())
    live_turn_path = _write_json(tmp_path / "live-turn.json", transcript_only_turn)
    output_dir = tmp_path / "bundle"
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    exit_code = realtime_voice_live_evidence.main(
        [
            "--output-dir",
            str(output_dir),
            "--audit-only",
            "--discord-live-probe-evidence",
            str(discord_path),
            "--sidecar-session-evidence",
            str(sidecar_path),
            "--live-turn-evidence",
            str(live_turn_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["artifact_writes"] is False
    assert "live_evidence_validation:live_turn:audio_segment_ref_observed_not_true" in payload["issues"]
    assert "live_evidence_validation:live_turn:interpreter_evidence_observed_not_true" in payload["issues"]
    assert "live_turn" in payload["strict_validation"]["missing_gates"]
    assert not output_dir.exists()


def test_live_evidence_audit_only_rejects_fake_parent_manifest_hash(monkeypatch, tmp_path, capsys):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run in audit-only mode")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run in audit-only mode")

    discord_path = _write_json(tmp_path / "discord-live-probe.json", _complete_discord_probe())
    sidecar_path = _write_json(tmp_path / "sidecar-session.json", _complete_sidecar_session())
    live_turn = _complete_live_turn()
    live_turn_path = _write_json(tmp_path / "live-turn.json", live_turn)
    live_turn["collector_attestation"]["parent_manifest_sha256"] = "d" * 64
    live_turn_path.write_text(json.dumps(live_turn), encoding="utf-8")
    output_dir = tmp_path / "bundle"
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    exit_code = realtime_voice_live_evidence.main(
        [
            "--output-dir",
            str(output_dir),
            "--audit-only",
            "--discord-live-probe-evidence",
            str(discord_path),
            "--sidecar-session-evidence",
            str(sidecar_path),
            "--live-turn-evidence",
            str(live_turn_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert (
        "live_evidence_validation:live_turn:collector_attestation_parent_manifest_sha256_mismatch"
        in payload["issues"]
    )
    assert not output_dir.exists()


def test_live_evidence_audit_only_rejects_report_derivation_without_writing(tmp_path, capsys):
    report_path = _write_json(tmp_path / "realtime-voice-report.json", _alpha_realtime_voice_report(include_discord=True))
    output_dir = tmp_path / "bundle"

    exit_code = realtime_voice_live_evidence.main(
        [
            "--output-dir",
            str(output_dir),
            "--audit-only",
            "--from-realtime-voice-report",
            str(report_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert "audit_only: --from-realtime-voice-report is not supported because derivation writes files" in payload["issues"]
    assert payload["report_derivation_run"] is False
    assert not output_dir.exists()


def test_live_evidence_derives_complete_sections_from_realtime_voice_report(monkeypatch, tmp_path):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run when deriving from an existing report")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run when deriving from an existing report")

    report_path = _write_json(tmp_path / "realtime-voice-report.json", _alpha_realtime_voice_report(include_discord=True))
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--from-realtime-voice-report",
            str(report_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    assert result.ok is True
    assert result.live_probe_status == "not_run"
    assert result.strict_validation["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert result.strict_validation["missing_gates"] == []
    assert result.reports["discord_live_probe"].endswith("discord-live-probe.from-realtime-report.json")
    assert result.reports["sidecar_session"].endswith("sidecar-session.from-realtime-report.json")
    assert result.reports["live_turn"].endswith("live-turn.from-realtime-report.json")
    discord = json.loads(
        (tmp_path / "bundle" / "discord-live-probe.from-realtime-report.json").read_text(encoding="utf-8")
    )
    sidecar = json.loads((tmp_path / "bundle" / "sidecar-session.from-realtime-report.json").read_text(encoding="utf-8"))
    live_turn = json.loads((tmp_path / "bundle" / "live-turn.from-realtime-report.json").read_text(encoding="utf-8"))
    assert discord["collector_attestation"]["collector_name"] == "hermes_cli.realtime_voice_live_evidence"
    assert sidecar["collector_attestation"]["collector_name"] == "hermes_cli.realtime_voice_live_evidence"
    assert live_turn["collector_attestation"]["collector_name"] == "hermes_cli.realtime_voice_live_evidence"
    assert sidecar["sidecar_mode"] == "production"
    assert sidecar["session_closed"] is True
    assert sidecar["provider_transport_observed"] is True
    assert sidecar["latency_metrics_ms"] == {"session_start_ms": 12.0, "shutdown_ms": 80.0}
    assert sidecar["source_artifact"] == str(report_path.resolve())
    assert live_turn["turn_id"] == "voiceops-report-turn-001"
    assert live_turn["audio_segment_ref"] == "artifact://redacted/voiceops-report-turn-001.wav"
    assert live_turn["evidence_bundle_id"] == "kame-evidence-voiceops-report-turn-001"
    assert live_turn["evidence_merge_key"] == "kame-merge-voiceops-report-turn-001"
    assert live_turn["transcript_observed"] is True
    assert live_turn["audio_segment_ref_observed"] is True
    assert live_turn["interpreter_evidence_observed"] is True
    assert live_turn["transcript_hypotheses_labeled"] is True
    assert live_turn["witness_arrival_phases"] == [
        "before_raw_audio",
        "with_raw_audio",
        "after_interpreter_start",
    ]
    assert live_turn["interpreter_input_order"] == [
        "raw_audio",
        "metadata",
        "reflex",
        "transcript_hypotheses",
    ]
    assert live_turn["transcript_hypotheses"][0]["kind"] == "frontend_witness_hypothesis"
    assert live_turn["transcript_hypotheses"][0]["authority"] == "hypothesis"
    assert live_turn["transcript_hypotheses"][0]["tool_authority"] is False
    assert live_turn["transcript_hypotheses"][0]["speaker_guess"]["channel_user_id"] == "jetha-redacted"
    assert live_turn["transcript_hypotheses"][0]["channel_guess"]["channel_id"] == "general-redacted"
    assert live_turn["speaker"]["channel_user_id"] == "jetha-redacted"
    assert live_turn["channel"]["channel_id"] == "general-redacted"
    assert live_turn["interpreter_adjudication_outcomes"] == ["corrected_by_audio"]
    assert live_turn["promoted_evidence_authority"] == {
        "interpreter_corrected_transcript": "interpreter_promoted",
        "interpreter_normalized_intent": "interpreter_promoted",
    }
    assert live_turn["assistant_audio_observed"] is True
    assert live_turn["barge_in_observed"] is True
    assert live_turn["barge_in_stop_ms"] == 45.0
    for forbidden in ("assistant_text", "raw_transcript", "final_text", "assistant_final_text", "text"):
        assert forbidden not in live_turn


def test_live_evidence_derivation_rejects_conflicting_kame_lineage(monkeypatch, tmp_path):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run when deriving from an existing report")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run when deriving from an existing report")

    report = _alpha_realtime_voice_report(include_discord=True)
    for entry in report:
        if entry.get("kind") not in {"audio_session", "session_turn"}:
            continue
        hypotheses = entry.get("transcript_hypotheses")
        if isinstance(hypotheses, list) and hypotheses:
            hypotheses[0]["audio_segment_ref"] = "artifact://redacted/wrong-speaker-or-stale-cut.wav"
    report_path = _write_json(tmp_path / "realtime-voice-report.json", report)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--from-realtime-voice-report",
            str(report_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    validation = json.loads((tmp_path / "bundle" / "realtime-voice-report-validation.json").read_text(encoding="utf-8"))
    assert result.ok is False
    assert "realtime_voice_report:kame_lineage_conflict:audio_segment_ref" in result.issues
    assert "realtime_voice_report: unable to derive live_turn evidence" in result.issues
    assert "live_turn" in result.strict_validation["missing_gates"]
    assert not (tmp_path / "bundle" / "live-turn.from-realtime-report.json").exists()
    assert "kame_lineage_conflict:audio_segment_ref" in validation["issues"]


def test_live_evidence_derivation_collapses_superseded_partial_hypotheses(monkeypatch, tmp_path):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run when deriving from an existing report")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run when deriving from an existing report")

    report = _alpha_realtime_voice_report(include_discord=True)
    for entry in report:
        if entry.get("kind") != "session_turn":
            continue
        hypotheses = entry.get("transcript_hypotheses")
        if isinstance(hypotheses, list) and hypotheses:
            hypotheses.insert(
                0,
                {
                    **hypotheses[0],
                    "text": "what is three to the",
                    "partial": True,
                },
            )
            hypotheses[1]["text"] = "what is three to the power of seventeen"
            hypotheses[1]["partial"] = False
    report_path = _write_json(tmp_path / "realtime-voice-report.json", report)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--from-realtime-voice-report",
            str(report_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    live_turn = json.loads((tmp_path / "bundle" / "live-turn.from-realtime-report.json").read_text(encoding="utf-8"))
    assert result.ok is True
    assert len(live_turn["transcript_hypotheses"]) == 1
    assert live_turn["transcript_hypotheses"][0]["partial"] is False
    assert live_turn["transcript_hypotheses"][0]["text"] == "what is three to the power of seventeen"
    assert live_turn["transcript_hypotheses"][0]["superseded_partial_texts"] == ["what is three to the"]
    assert live_turn["transcript_hypotheses"][0]["superseded_partial_count"] == 1


def test_live_evidence_derivation_does_not_infer_observation_from_kame_ids(monkeypatch, tmp_path):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run when deriving from an existing report")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run when deriving from an existing report")

    report = _alpha_realtime_voice_report(include_discord=True)
    for entry in report:
        if entry.get("kind") in {"session_turn", "audio_session"}:
            entry.pop("audio_segment_ref_observed", None)
            entry.pop("interpreter_evidence_observed", None)
            entry["audio_bytes"] = 0
            entry.pop("interpreter_input_order", None)
            entry.pop("interpreter_adjudication_outcomes", None)
            entry.pop("promoted_evidence_authority", None)
    report_path = _write_json(tmp_path / "realtime-voice-report.json", report)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--from-realtime-voice-report",
            str(report_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    live_turn = json.loads((tmp_path / "bundle" / "live-turn.from-realtime-report.json").read_text(encoding="utf-8"))
    assert result.ok is False
    assert live_turn["turn_id"] == "voiceops-report-turn-001"
    assert live_turn["audio_segment_ref"] == "artifact://redacted/voiceops-report-turn-001.wav"
    assert live_turn["audio_segment_ref_observed"] is False
    assert live_turn["interpreter_evidence_observed"] is False
    assert "live_evidence_validation:live_turn:audio_segment_ref_observed_not_true" in result.issues
    assert "live_evidence_validation:live_turn:interpreter_evidence_observed_not_true" in result.issues
    assert "live_evidence_validation:live_turn:missing_interpreter_input_order" in result.issues
    assert "live_turn" in result.strict_validation["missing_gates"]


def test_live_evidence_collector_attestation_clamps_future_collected_at(monkeypatch):
    monkeypatch.setattr(realtime_voice_live_evidence, "_utc_timestamp", lambda: "2026-06-29T00:00:01Z")

    payload = {
        "kind": "discord_live_probe",
        "collected_at": "2026-06-29T00:00:02Z",
        "ok": True,
    }

    enriched = realtime_voice_live_evidence._with_collector_attestation(
        payload,
        section_name="discord_live_probe",
    )

    assert enriched["collector_attestation"]["started_at"] == "2026-06-29T00:00:01Z"
    assert enriched["collector_attestation"]["finished_at"] == "2026-06-29T00:00:01Z"


def test_live_evidence_derives_partial_report_without_discord_probe(monkeypatch, tmp_path):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run when deriving from an existing report")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run when deriving from an existing report")

    report_path = _write_json(tmp_path / "realtime-voice-report.json", _alpha_realtime_voice_report())
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--from-realtime-voice-report",
            str(report_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    assert result.ok is False
    assert "discord_live_probe" not in result.reports
    assert result.reports["sidecar_session"].endswith("sidecar-session.from-realtime-report.json")
    assert result.reports["live_turn"].endswith("live-turn.from-realtime-report.json")
    assert result.strict_validation["overall_status"] == "partial_live_evidence"
    assert result.strict_validation["missing_gates"] == ["discord_join", "discord_playback", "live_receiver"]
    assert "live_evidence_validation:discord_live_probe:missing_source_artifact" in result.issues


def test_live_evidence_derivation_does_not_overclaim_loopback_sidecar(monkeypatch, tmp_path):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run when deriving from an existing report")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run when deriving from an existing report")

    report_path = _write_json(
        tmp_path / "realtime-voice-report.json",
        _alpha_realtime_voice_report(include_discord=True, sidecar_mode="managed_loopback", bridge_closed=False),
    )
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--from-realtime-voice-report",
            str(report_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    sidecar = json.loads((tmp_path / "bundle" / "sidecar-session.from-realtime-report.json").read_text(encoding="utf-8"))
    assert result.ok is False
    assert sidecar["sidecar_mode"] == "managed_loopback"
    assert sidecar["session_closed"] is False
    assert "sidecar_session" not in result.reports
    assert "sidecar_session: sidecar_mode must be production" in result.issues
    assert "sidecar_session: session_closed must be true" in result.issues


def test_live_evidence_derivation_detects_voice_capability_denial(monkeypatch, tmp_path):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run when deriving from an existing report")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run when deriving from an existing report")

    report_path = _write_json(
        tmp_path / "realtime-voice-report.json",
        _alpha_realtime_voice_report(include_discord=True, denial_text=True),
    )
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--from-realtime-voice-report",
            str(report_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    live_turn = json.loads((tmp_path / "bundle" / "live-turn.from-realtime-report.json").read_text(encoding="utf-8"))
    assert result.ok is False
    assert live_turn["no_voice_denial_observed"] is False
    assert "live_turn" not in result.reports
    assert "live_turn: no_voice_denial_observed must be true" in result.issues
    for forbidden in ("assistant_text", "raw_transcript", "final_text", "assistant_final_text", "text"):
        assert forbidden not in live_turn


def test_live_evidence_derivation_refuses_invalid_alpha_report(monkeypatch, tmp_path):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run when deriving from an existing report")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run when deriving from an existing report")

    report_path = _write_json(tmp_path / "realtime-voice-report.json", _alpha_realtime_voice_report(invalid=True))
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_loopback_smoke", unexpected_loopback)
    monkeypatch.setattr(realtime_voice_live_evidence, "_run_discord_live_probe", unexpected_live)

    args = realtime_voice_live_evidence.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "bundle"),
            "--from-realtime-voice-report",
            str(report_path),
        ]
    )
    result = asyncio.run(realtime_voice_live_evidence.collect_realtime_voice_live_evidence(args))

    sidecar = json.loads((tmp_path / "bundle" / "sidecar-session.from-realtime-report.json").read_text(encoding="utf-8"))
    validation = json.loads((tmp_path / "bundle" / "realtime-voice-report-validation.json").read_text(encoding="utf-8"))
    assert result.ok is False
    assert validation["alpha_valid"] is False
    assert sidecar["sidecar_mode"] == "production"
    assert "sidecar_session" not in result.reports
    assert "realtime_voice_report:barge_in: missing required text (Hello from Hermes.)" in result.issues
    assert "sidecar_session: sidecar_running must be true" in result.issues


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


def test_live_evidence_collection_strict_validates_optional_evidence_without_validate_flag(monkeypatch, tmp_path):
    async def fake_loopback():
        return _FakeProbeResult(ok=True)

    async def fake_live(_args):
        return _FakeDiscordLiveProbeResult()

    bad_turn = _complete_live_turn()
    bad_turn["barge_in_stop_ms"] = 999
    bad_turn["assistant_text"] = "I cannot hear you in Discord voice."
    sidecar_path = _write_json(tmp_path / "sidecar-session.json", _complete_sidecar_session())
    live_turn_path = _write_json(tmp_path / "live-turn.json", bad_turn)
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

    assert result.ok is False
    assert result.validate_live_evidence is False
    assert result.strict_validation["overall_status"] == "partial_live_evidence"
    assert "live_evidence_validation:live_turn:barge_in_stop_ms_over_target" in result.issues
    assert "live_evidence_validation:live_turn.assistant_text:voice_capability_denial_text" in result.issues
    assert (tmp_path / "bundle" / "live-evidence-validation.json").is_file()


def test_live_evidence_manifest_rejects_anonymous_optional_evidence(monkeypatch, tmp_path):
    async def fake_loopback():
        return _FakeProbeResult(ok=True)

    async def fake_live(_args):
        return _FakeDiscordLiveProbeResult()

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


def test_live_evidence_manifest_rejects_schema_only_optional_evidence_without_nested_section(monkeypatch, tmp_path):
    async def fake_loopback():
        return _FakeProbeResult(ok=True)

    async def fake_live(_args):
        return _FakeDiscordLiveProbeResult()

    sidecar_payload = _complete_sidecar_session()
    sidecar_payload.pop("kind")
    sidecar_payload["schema_version"] = "voiceops.milestone1.live_voice_evidence.v1"
    sidecar_path = _write_json(tmp_path / "sidecar-session.json", sidecar_payload)
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


def test_live_evidence_manifest_rejects_incomplete_discord_probe_optional_evidence(monkeypatch, tmp_path):
    async def unexpected_loopback():
        raise AssertionError("loopback probe should not run in validation mode")

    async def unexpected_live(_args):
        raise AssertionError("live Discord probe should not run in validation mode")

    discord_path = _write_json(tmp_path / "discord-live-probe.json", {"kind": "discord_live_probe", "ok": True})
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

    assert result.ok is False
    assert "discord_live_probe" not in result.reports
    assert "discord_live_probe: connect_perm must be true" in result.issues
    assert "discord_live_probe: latency_metrics_ms.connect_ms must be a non-negative number" in result.issues
    assert "discord_live_probe: receiver_frames or receiver_speech_start must be positive" in result.issues


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
    assert f"sidecar_session: evidence file not found at {(tmp_path / 'missing-sidecar.json').resolve()}" in result.issues
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
