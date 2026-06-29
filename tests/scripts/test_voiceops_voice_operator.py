from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.voiceops_voice_operator import (
    DEFAULT_OUTPUT_DIR,
    build_voice_operator_report,
    parse_args,
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
    assert report["proofs"]["lifecycle"]["sidecar_closed"] is True
    assert report["proofs"]["callback_wiring"]["loopback_bypasses_live_discord_receiver"] is True
    assert report["proofs"]["pcm_conversion"]["sidecar_pcm16_first_sample"] == 450
    assert report["proofs"]["barge_in_energy"]["speech_energy_event_forwarded"] is True
    assert report["proofs"]["barge_in_energy"]["energy_gate_proven_by_smoke"] is False
    assert report["proofs"]["barge_in_energy"]["energy_gate_covered_by_tests"] is True
    assert report["proofs"]["shutdown"]["close_timeout_bounded"] is True
    assert report["proofs"]["latency_metrics"]["oracle_metric_status"] == "needs_live_oracle_or_sidecar_probe"
    assert report["live_probe_required_for_completion"]["status"] == "needs_live_probe"
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

    assert set(paths) == {"events_jsonl", "json", "markdown", "smoke_json"}
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    smoke = json.loads(Path(paths["smoke_json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    events = Path(paths["events_jsonl"]).read_text(encoding="utf-8").splitlines()
    assert payload["schema_version"] == "voiceops.milestone1.voice_operator.v1"
    assert smoke["ok"] is True
    assert json.loads(events[0])["event_id"] == "voice-m1-001"
    assert "VoiceOps Milestone 1 Voice Operator" in markdown
    assert "Proofs" in markdown
    assert "Live Probe Boundary" in markdown


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
