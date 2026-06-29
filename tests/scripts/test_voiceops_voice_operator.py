from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.voiceops_voice_operator import (
    DEFAULT_OUTPUT_DIR,
    _load_live_evidence,
    build_live_probe_evidence_example,
    build_live_probe_evidence_template,
    build_voice_operator_report,
    parse_args,
    validate_live_probe_evidence,
    validate_voice_operator_report,
    write_voice_operator_report,
)


def _smoke_payload() -> dict:
    return {
        "ok": True,
        "mode": "discord_loopback",
        "transport": "discord_voice",
        "input_pcm48_bytes": 3840,
        "sidecar_pcm16_bytes": 640,
        "sidecar_pcm16_first_sample": 450,
        "sidecar_pcm16_checksum": 195,
        "mixer_frames": 1,
        "mixer_frame_bytes": 3840,
        "speech_energy_sent": True,
        "barge_in_sent": True,
        "mixer_stop_calls": 1,
        "sidecar_closed": True,
        "shutdown_elapsed_ms": 1,
        "shutdown_bounded": True,
        "shutdown_timed_out": False,
        "events": [
            "transcript.partial",
            "transcript.final",
            "assistant.text.partial",
            "audio.output.chunk",
            "assistant.commit",
            "barge_in",
        ],
        "evidence_context": {"git_commit": "abc", "git_branch": "branch"},
        "latency_metrics_ms": {
            "session_start_ms": 1,
            "input_to_first_mixer_frame_ms": 2,
            "barge_in_ack_ms": 3,
            "shutdown_ms": 1,
        },
        "error": "",
    }


def test_voice_operator_report_maps_loopback_smoke_to_milestone_1_contract():
    report = build_voice_operator_report(_smoke_payload())

    assert report["schema_version"] == "voiceops.milestone1.voice_operator.v1"
    assert report["artifact_only"] is True
    assert report["status"] == "needs_live_probe"
    assert report["missing_live_gates"] == [
        "discord_join",
        "discord_playback",
        "live_receiver",
        "production_sidecar",
        "live_turn",
    ]
    assert report["mode"] == {
        "bounded": True,
        "discord_network": False,
        "env_secret_reads": False,
        "headless": True,
        "outbound_calls": False,
        "outbound_sends": False,
        "provider_sidecar_network": False,
    }
    assert validate_voice_operator_report(report) == []
    assert report["requirements"]["stable_discord_receive_playback_lifecycle"] is True
    assert report["requirements"]["receiver_callback_wiring"] is True
    assert report["requirements"]["pcm_conversion_correctness"] is True
    assert report["requirements"]["mixer_playback_path"] is True
    assert report["requirements"]["barge_in_behavior"] is True
    assert report["requirements"]["latency_metrics"] is True
    assert report["requirements"]["sidecar_session_shutdown"] is True
    assert report["requirements"]["live_discord_join"] is False
    assert report["requirements"]["live_evidence_supplied"] is False
    assert report["proofs"]["lifecycle"]["sidecar_closed"] is True
    assert report["proofs"]["callback_wiring"]["loopback_bypasses_live_discord_receiver"] is True
    assert report["proofs"]["pcm_conversion"]["sidecar_pcm16_first_sample"] == 450
    assert report["proofs"]["barge_in_energy"]["speech_energy_event_forwarded"] is True
    assert report["proofs"]["barge_in_energy"]["energy_gate_proven_by_smoke"] is False
    assert report["proofs"]["barge_in_energy"]["energy_gate_covered_by_tests"] is True
    assert report["proofs"]["shutdown"]["close_timeout_bounded"] is True
    assert report["proofs"]["latency_metrics"]["oracle_metric_status"] == "needs_live_oracle_or_sidecar_probe"
    assert report["live_probe_required_for_completion"]["status"] == "needs_live_probe"
    assert report["live_probe_required_for_completion"]["missing_gates"] == [
        "discord_join",
        "discord_playback",
        "live_receiver",
        "production_sidecar",
        "live_turn",
    ]
    assert report["live_evidence"]["overall_status"] == "needs_live_probe"
    assert "I cannot hear voice." in report["voice_capability_prompt_contract"]["must_not_claim"]
    assert report["barge_in_policy"]["silent_packet_policy"].startswith("silent PCM")


def test_voice_operator_validation_rejects_missing_core_coverage():
    smoke = _smoke_payload()
    smoke["events"] = ["audio.output.chunk"]
    smoke["sidecar_closed"] = False
    smoke["shutdown_bounded"] = False
    report = build_voice_operator_report(smoke)

    issues = validate_voice_operator_report(report)
    assert "missing_coverage:discord_receiver_callback_wiring" in issues
    assert "missing_coverage:lifecycle_start_and_shutdown" in issues
    assert "missing_coverage:sidecar_session_shutdown" in issues


def test_write_voice_operator_report_artifacts(tmp_path):
    report = build_voice_operator_report(_smoke_payload())
    paths = write_voice_operator_report(tmp_path, report)

    assert set(paths) == {
        "events_jsonl",
        "json",
        "live_evidence_example",
        "live_evidence_template",
        "live_probe_closure_json",
        "live_probe_closure_markdown",
        "markdown",
        "smoke_json",
    }
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    smoke = json.loads(Path(paths["smoke_json"]).read_text(encoding="utf-8"))
    live_template = json.loads(Path(paths["live_evidence_template"]).read_text(encoding="utf-8"))
    live_example = json.loads(Path(paths["live_evidence_example"]).read_text(encoding="utf-8"))
    live_closure = json.loads(Path(paths["live_probe_closure_json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    closure_markdown = Path(paths["live_probe_closure_markdown"]).read_text(encoding="utf-8")
    events = Path(paths["events_jsonl"]).read_text(encoding="utf-8").splitlines()
    assert payload["schema_version"] == "voiceops.milestone1.voice_operator.v1"
    assert payload["status"] == "needs_live_probe"
    assert payload["missing_live_gates"] == [
        "discord_join",
        "discord_playback",
        "live_receiver",
        "production_sidecar",
        "live_turn",
    ]
    assert smoke["ok"] is True
    assert live_template["schema_version"] == "voiceops.milestone1.live_voice_evidence.v1"
    assert live_example["example_only"] is True
    assert "example_only_evidence_not_accepted" in validate_live_probe_evidence(live_example)["issues"]
    assert live_closure["schema_version"] == "voiceops.milestone1.live_probe_closure.v1"
    assert "hermes_cli.realtime_voice_live_evidence" in live_closure["recommended_collection"]["live_bundle_manifest"]
    assert "manifest.json" in live_closure["recommended_collection"]["ingest"]
    assert json.loads(events[0])["event_id"] == "voice-m1-001"
    assert "VoiceOps Milestone 1 Voice Operator" in markdown
    assert "Proofs" in markdown
    assert "Live Probe Boundary" in markdown
    assert "Supplied Live Evidence" in markdown
    assert "VoiceOps Milestone 1 Live Probe Closure" in closure_markdown


def test_live_evidence_classifies_partial_discord_probe_without_inbound():
    evidence = {
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
        "receiver_frames": 0,
        "receiver_speech_start": 0,
        "inbound_observed": False,
        "disconnected": True,
        "require_inbound": True,
    }

    result = validate_live_probe_evidence(evidence)

    assert result["overall_status"] == "partial_live_evidence"
    assert result["discord_live_probe"]["join_ok"] is True
    assert result["discord_live_probe"]["playback_ok"] is True
    assert result["discord_live_probe"]["inbound_observed"] is False
    assert "discord_live_probe:inbound_not_observed" in result["issues"]


def test_live_evidence_example_is_not_accepted_as_proof():
    result = validate_live_probe_evidence(build_live_probe_evidence_example())

    assert result["overall_status"] == "partial_live_evidence"
    assert "example_only_evidence_not_accepted" in result["issues"]


def test_voice_operator_accepts_complete_supplied_live_evidence_without_changing_safety_mode():
    evidence = build_live_probe_evidence_template()
    evidence["discord_live_probe"].update(
        {
            "ok": True,
            "connect_perm": True,
            "speak_perm": True,
            "connected": True,
            "opus_loaded": True,
            "accepted_audio_source": True,
            "played": True,
            "playing_during_probe": True,
            "receiver_started": True,
            "receiver_frames": 12,
            "receiver_speech_start": 1,
            "inbound_observed": True,
            "disconnected": True,
            "require_inbound": True,
        }
    )
    evidence["sidecar_session"].update(
        {
            "sidecar_running": True,
            "sidecar_healthy": True,
            "session_started": True,
            "session_closed": True,
            "fallback_mode_visible": True,
        }
    )
    evidence["live_turn"].update(
        {
            "transcript_observed": True,
            "assistant_audio_observed": True,
            "barge_in_observed": True,
            "spoken_reply_short": True,
            "no_voice_denial_observed": True,
            "speech_end_to_first_audio_ms": 900,
            "barge_in_stop_ms": 90,
        }
    )
    live_evidence = validate_live_probe_evidence(evidence)
    report = build_voice_operator_report(_smoke_payload(), live_evidence=live_evidence)

    assert report["mode"]["discord_network"] is False
    assert report["mode"]["env_secret_reads"] is False
    assert report["requirements"]["live_discord_join"] is False
    assert report["requirements"]["live_evidence_supplied"] is True
    assert report["live_evidence"]["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert report["live_probe_required_for_completion"]["missing_gates"] == []
    assert report["proofs"]["live_evidence"]["ok"] is True


def test_voice_operator_ingests_realtime_live_evidence_manifest(tmp_path):
    discord_probe = {
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
    sidecar = {
        "sidecar_running": True,
        "sidecar_healthy": True,
        "session_started": True,
        "session_closed": True,
        "fallback_mode_visible": True,
    }
    live_turn = {
        "transcript_observed": True,
        "assistant_audio_observed": True,
        "barge_in_observed": True,
        "spoken_reply_short": True,
        "no_voice_denial_observed": True,
        "speech_end_to_first_audio_ms": 950,
        "barge_in_stop_ms": 80,
    }
    (tmp_path / "discord-live-probe.json").write_text(json.dumps(discord_probe), encoding="utf-8")
    (tmp_path / "sidecar-session.json").write_text(json.dumps(sidecar), encoding="utf-8")
    (tmp_path / "live-turn.json").write_text(json.dumps(live_turn), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "ok": True,
                "reports": {
                    "discord_live_probe": "discord-live-probe.json",
                    "sidecar_session": "sidecar-session.json",
                    "live_turn": "live-turn.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])
    report = build_voice_operator_report(_smoke_payload(), live_evidence=live_evidence)

    assert live_evidence["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert live_evidence["issues"] == []
    assert live_evidence["discord_live_probe"]["join_ok"] is True
    assert report["live_probe_required_for_completion"]["missing_gates"] == []
    assert report["status"] == "live_evidence_supplied_not_readiness_claim"


def test_voice_operator_cli_smoke(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "voiceops_voice_operator.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["validation_issues"] == []
    assert Path(payload["artifacts"]["json"]).exists()
    assert Path(payload["artifacts"]["markdown"]).exists()
    assert Path(payload["artifacts"]["smoke_json"]).exists()
    assert Path(payload["artifacts"]["events_jsonl"]).exists()


def test_parse_args_defaults_to_requested_artifact_dir():
    args = parse_args([])

    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.live_evidence == []
