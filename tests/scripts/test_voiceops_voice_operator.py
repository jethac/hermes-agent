from __future__ import annotations

import hashlib
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


def _async_oracle_smoke_payload() -> dict:
    return {
        "ok": True,
        "scenario": "async_kame_oracle_jobs_fake",
        "max_running": 4,
        "started_jobs": 5,
        "queued_jobs": 1,
        "completed_jobs": 4,
        "cancelled_jobs": 1,
        "local_turn_committed": True,
        "status_turn_committed": True,
        "status_text": "Oracle jobs: 4 running out of 4, 1 queued. running: Starting smoke task 1.",
        "fifth_job_id": "voice-oracle-005",
        "fifth_job_queued": True,
        "fifth_job_started_after_capacity_freed": True,
        "cancelled_job_id": "voice-oracle-003",
        "late_cancelled_output_attempted": True,
        "cancelled_result_spoken": False,
        "cancelled_result_committed": False,
        "cancelled_result_progress_leaked": False,
        "cancelled_result_durable_completed": False,
        "cancelled_result_durable_text": False,
        "durable_cancelled_record_present": True,
        "durable_completed_jobs": 4,
        "spoken": [
            "Starting smoke task 1.",
            "Starting smoke task 2.",
            "Starting smoke task 3.",
            "Starting smoke task 4.",
            "Yes, I can hear you.",
            "Finished Run smoke task 1.",
            "Finished Run smoke task 2.",
            "Finished Run smoke task 4.",
        ],
        "event_counts": {
            "oracle.job.started": 4,
            "oracle.job.completed": 3,
            "oracle.job.cancelled": 1,
            "assistant.commit": 4,
        },
    }


def _voice_operator_report(live_evidence: dict | None = None, smoke: dict | None = None) -> dict:
    return build_voice_operator_report(
        smoke or _smoke_payload(),
        live_evidence=live_evidence,
        async_oracle_smoke=_async_oracle_smoke_payload(),
    )


def _collector_attestation(section_name: str) -> dict:
    return {
        "collector_name": "pytest.voiceops_live_fixture",
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


def _payload_sha256(payload: dict) -> str:
    attested_payload = dict(payload)
    attested_payload.pop("collector_attestation", None)
    attested_payload.pop("collector_provenance", None)
    raw = json.dumps(attested_payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _attest_section(section: dict, section_name: str) -> dict:
    section["collector_attestation"] = _collector_attestation(section_name)
    payload_sha256 = _payload_sha256(section)
    section["collector_attestation"]["raw_artifact_sha256"] = payload_sha256
    section["collector_attestation"]["redacted_artifact_sha256"] = payload_sha256
    section["collector_attestation"]["parent_manifest_sha256"] = payload_sha256
    return section


def _write_attested_section(path: Path, section: dict, section_name: str) -> None:
    _attest_section(section, section_name)
    path.write_text(json.dumps(section), encoding="utf-8")


def _complete_live_evidence() -> dict:
    evidence = build_live_probe_evidence_template()
    evidence["discord_live_probe"].update(
        {
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
            "receiver_frames": 12,
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
    )
    evidence["sidecar_session"].update(
        {
            "collector_attestation": _collector_attestation("sidecar_session"),
            "sidecar_running": True,
            "sidecar_healthy": True,
            "session_started": True,
            "session_closed": True,
            "shutdown_bounded": True,
            "shutdown_timed_out": False,
            "fallback_mode_visible": True,
            "fallback_reason": "none",
            "sidecar_mode": "production",
            "healthcheck_observed": True,
            "provider_transport_observed": True,
            "session_id_redacted": True,
            "latency_metrics_ms": {"session_start_ms": 110, "shutdown_ms": 80},
        }
    )
    evidence["live_turn"].update(
        {
            "collector_attestation": _collector_attestation("live_turn"),
            "transcript_observed": True,
            "assistant_audio_observed": True,
            "barge_in_observed": True,
            "spoken_reply_short": True,
            "no_voice_denial_observed": True,
            "speech_end_to_first_audio_ms": 900,
            "barge_in_stop_ms": 90,
        }
    )
    return evidence


def _complete_discord_latency_metrics() -> dict:
    return {
        "connect_ms": 420,
        "playback_observed_ms": 180,
        "inbound_observed_ms": 900,
        "disconnect_ms": 120,
    }


def _complete_sidecar_session_fields() -> dict:
    return {
        "sidecar_running": True,
        "sidecar_healthy": True,
        "session_started": True,
        "session_closed": True,
        "shutdown_bounded": True,
        "shutdown_timed_out": False,
        "fallback_mode_visible": True,
        "fallback_reason": "none",
        "sidecar_mode": "production",
        "healthcheck_observed": True,
        "provider_transport_observed": True,
        "session_id_redacted": True,
        "latency_metrics_ms": {"session_start_ms": 110, "shutdown_ms": 80},
    }


def test_voice_operator_report_maps_loopback_smoke_to_milestone_1_contract():
    report = _voice_operator_report()

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
    assert report["requirements"]["async_oracle_four_concurrent_jobs"] is True
    assert report["requirements"]["async_oracle_local_turn_while_running"] is True
    assert report["requirements"]["async_oracle_status_turn_while_running"] is True
    assert report["requirements"]["async_oracle_fifth_job_queued_and_started"] is True
    assert report["requirements"]["async_oracle_cancellation_isolated"] is True
    assert report["requirements"]["async_oracle_late_cancelled_output_attempted"] is True
    assert report["requirements"]["async_oracle_late_cancelled_output_dropped"] is True
    assert report["requirements"]["async_oracle_late_cancelled_output_not_durable"] is True
    assert report["requirements"]["live_discord_join"] is False
    assert report["requirements"]["live_evidence_supplied"] is False
    assert report["proofs"]["lifecycle"]["sidecar_closed"] is True
    assert report["proofs"]["callback_wiring"]["loopback_bypasses_live_discord_receiver"] is True
    assert report["proofs"]["pcm_conversion"]["sidecar_pcm16_first_sample"] == 450
    assert report["proofs"]["barge_in_energy"]["speech_energy_event_forwarded"] is True
    assert report["proofs"]["barge_in_energy"]["energy_gate_proven_by_smoke"] is False
    assert report["proofs"]["barge_in_energy"]["energy_gate_covered_by_tests"] is True
    assert report["proofs"]["shutdown"]["close_timeout_bounded"] is True
    assert report["proofs"]["async_oracle_jobs"]["ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["max_running"] == 4
    assert report["proofs"]["async_oracle_jobs"]["queued_jobs"] == 1
    assert report["proofs"]["async_oracle_jobs"]["status_turn_committed"] is True
    assert report["proofs"]["async_oracle_jobs"]["fifth_job_queued"] is True
    assert report["proofs"]["async_oracle_jobs"]["fifth_job_started_after_capacity_freed"] is True
    assert report["proofs"]["async_oracle_jobs"]["late_cancelled_output_attempted"] is True
    assert report["proofs"]["async_oracle_jobs"]["cancelled_result_spoken"] is False
    assert report["proofs"]["async_oracle_jobs"]["cancelled_result_committed"] is False
    assert report["proofs"]["async_oracle_jobs"]["cancelled_result_progress_leaked"] is False
    assert report["proofs"]["async_oracle_jobs"]["cancelled_result_durable_completed"] is False
    assert report["proofs"]["async_oracle_jobs"]["cancelled_result_durable_text"] is False
    assert report["proofs"]["async_oracle_jobs"]["durable_cancelled_record_present"] is True
    assert report["proofs"]["async_oracle_jobs"]["durable_completed_jobs"] == 4
    assert report["async_oracle_acceptance"]["fifth_job_obeys_overflow_policy"]["ok"] is True
    assert report["async_oracle_acceptance"]["status_reports_running_and_queued_without_oracle_call"]["ok"] is True
    assert report["async_oracle_acceptance"]["result_handling_is_bounded_and_durable"]["ok"] is True
    assert "tests/agent/test_realtime_voice.py::test_kame_engine_local_status_question_uses_oracle_job_state" in (
        report["async_oracle_acceptance"]["status_reports_running_and_queued_without_oracle_call"]["test_refs"]
    )
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
    report = _voice_operator_report(smoke=smoke)

    issues = validate_voice_operator_report(report)
    assert "missing_coverage:discord_receiver_callback_wiring" in issues
    assert "missing_coverage:lifecycle_start_and_shutdown" in issues
    assert "missing_coverage:sidecar_session_shutdown" in issues


def test_voice_operator_validation_rejects_missing_async_oracle_smoke():
    report = build_voice_operator_report(_smoke_payload(), async_oracle_smoke={})

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:async_oracle_smoke_ok" in issues
    assert "missing_async_oracle_coverage:four_jobs_ran_concurrently" in issues
    assert "missing_async_oracle_coverage:local_turn_while_jobs_running" in issues
    assert "missing_async_oracle_coverage:status_turn_while_jobs_running" in issues
    assert "missing_async_oracle_coverage:fifth_job_queued_and_started_after_capacity_freed" in issues
    assert "missing_async_oracle_coverage:one_job_cancelled_while_others_completed" in issues
    assert "missing_async_oracle_coverage:late_cancelled_output_attempted" in issues
    assert "missing_async_oracle_coverage:late_cancelled_output_not_spoken" in issues
    assert "missing_async_oracle_coverage:late_cancelled_output_not_durable" in issues
    assert "missing_async_oracle_acceptance:four_oracle_jobs_reflex_responsive" in issues
    assert "missing_async_oracle_acceptance:fifth_job_obeys_overflow_policy" in issues


def test_write_voice_operator_report_artifacts(tmp_path):
    report = _voice_operator_report()
    paths = write_voice_operator_report(tmp_path, report)

    required_paths = {
        "async_oracle_smoke_json",
        "events_jsonl",
        "json",
        "live_evidence_example",
        "live_evidence_scaffold_manifest",
        "live_evidence_template",
        "live_probe_closure_json",
        "live_probe_closure_markdown",
        "markdown",
        "smoke_json",
    }
    assert required_paths <= set(paths)
    assert {
        "scaffold_discord_live_probe",
        "scaffold_sidecar_session",
        "scaffold_live_turn",
    } <= set(paths)
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    smoke = json.loads(Path(paths["smoke_json"]).read_text(encoding="utf-8"))
    async_oracle_smoke = json.loads(Path(paths["async_oracle_smoke_json"]).read_text(encoding="utf-8"))
    live_template = json.loads(Path(paths["live_evidence_template"]).read_text(encoding="utf-8"))
    live_example = json.loads(Path(paths["live_evidence_example"]).read_text(encoding="utf-8"))
    live_scaffold_manifest_path = Path(paths["live_evidence_scaffold_manifest"])
    live_scaffold_manifest = json.loads(live_scaffold_manifest_path.read_text(encoding="utf-8"))
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
    assert async_oracle_smoke["ok"] is True
    assert async_oracle_smoke["max_running"] == 4
    assert live_template["schema_version"] == "voiceops.milestone1.live_voice_evidence.v1"
    assert live_example["example_only"] is True
    assert "example_only_evidence_not_accepted" in validate_live_probe_evidence(live_example)["issues"]
    assert live_scaffold_manifest["example_only"] is True
    assert live_scaffold_manifest["reports"]["sidecar_session"] == "sections/sidecar-session.json"
    scaffold_evidence = _load_live_evidence([live_scaffold_manifest_path])
    assert scaffold_evidence["overall_status"] == "partial_live_evidence"
    assert "example_only_evidence_not_accepted" in scaffold_evidence["issues"]
    assert "live_evidence_manifest:sidecar_session:example_only_evidence_not_accepted" in scaffold_evidence["issues"]
    assert all("source_artifact_not_found" not in issue for issue in scaffold_evidence["issues"])
    assert json.loads(Path(paths["scaffold_live_turn"]).read_text(encoding="utf-8"))["kind"] == "live_turn"
    assert live_closure["schema_version"] == "voiceops.milestone1.live_probe_closure.v1"
    assert live_closure["live_evidence_scaffold_manifest"] == "live-voice-evidence-scaffold/manifest.json"
    assert live_closure["evidence_contract"]["manifest_schema_version"] == (
        "voiceops.realtime_voice_live_evidence_manifest.v1"
    )
    assert live_closure["evidence_contract"]["required_section_field"] == "source_artifact"
    assert live_closure["evidence_contract"]["source_artifacts_must_exist"] is True
    assert live_closure["evidence_contract"]["source_artifacts_must_be_json"] is True
    assert live_closure["evidence_contract"]["source_artifacts_reject_secret_or_phone_values"] is True
    assert live_closure["evidence_contract"]["source_artifacts_reject_voice_capability_denials"] is True
    assert live_closure["evidence_contract"]["template_source_artifacts_accepted"] is False
    assert live_closure["evidence_contract"]["collector_attestation_required_for_live_readiness"] is True
    assert live_closure["evidence_contract"]["collector_attestation_required_fields"] == [
        "collector_name",
        "collector_version",
        "run_id",
        "command_argv",
        "git_commit",
        "started_at",
        "finished_at",
        "raw_artifact_sha256",
        "redacted_artifact_sha256",
        "parent_manifest_sha256",
    ]
    assert live_closure["evidence_contract"]["placeholder_collector_attestation_accepted"] is False
    assert "kind/evidence_type" in live_closure["evidence_contract"]["manifest_report_identity"]
    assert "standalone non-expanded evidence files" in live_closure["evidence_contract"]["standalone_report_identity"]
    assert live_closure["evidence_shapes"]["discord_live_probe"]["kind"] == "discord_live_probe"
    assert live_closure["evidence_shapes"]["discord_live_probe"]["collector_attestation"]["example_only"] is True
    assert live_closure["evidence_shapes"]["discord_live_probe"]["require_inbound"] is True
    assert live_closure["evidence_shapes"]["sidecar_session"]["kind"] == "sidecar_session"
    assert live_closure["evidence_shapes"]["sidecar_session"]["source_artifact"] == "sidecar-session.json"
    assert live_closure["evidence_shapes"]["sidecar_session"]["collector_attestation"]["example_only"] is True
    assert live_closure["evidence_shapes"]["sidecar_session"]["shutdown_bounded"] is True
    assert live_closure["evidence_shapes"]["sidecar_session"]["shutdown_timed_out"] is False
    assert live_closure["evidence_shapes"]["sidecar_session"]["sidecar_mode"] == "production"
    assert live_closure["evidence_shapes"]["sidecar_session"]["fallback_reason"] == "none"
    assert live_closure["evidence_shapes"]["sidecar_session"]["healthcheck_observed"] is True
    assert live_closure["evidence_shapes"]["sidecar_session"]["provider_transport_observed"] is True
    assert live_closure["evidence_shapes"]["sidecar_session"]["session_id_redacted"] is True
    assert live_closure["evidence_shapes"]["sidecar_session"]["latency_metrics_ms"]["session_start_ms"] == 110
    assert live_closure["evidence_shapes"]["sidecar_session"]["latency_metrics_ms"]["shutdown_ms"] == 80
    assert live_closure["evidence_shapes"]["live_turn"]["kind"] == "live_turn"
    assert live_closure["evidence_shapes"]["live_turn"]["source_artifact"] == "live-turn.json"
    assert live_closure["evidence_shapes"]["live_turn"]["collector_attestation"]["example_only"] is True
    assert "hermes_cli.realtime_voice_live_evidence" in live_closure["recommended_collection"]["live_bundle_manifest"]
    assert "--run-doctor-report" in live_closure["recommended_collection"]["live_bundle_manifest"]
    assert "--require-inbound" in live_closure["recommended_collection"]["live_bundle_manifest"]
    assert "--audit-only" in live_closure["recommended_collection"]["audit_bundle_no_write"]
    assert "--discord-live-probe-evidence" in live_closure["recommended_collection"]["audit_bundle_no_write"]
    assert "--validate-live-evidence" in live_closure["recommended_collection"]["validate_bundle_offline"]
    assert "--discord-live-probe-evidence" in live_closure["recommended_collection"]["validate_bundle_offline"]
    assert "manifest.json" in live_closure["recommended_collection"]["ingest"]
    assert json.loads(events[0])["event_id"] == "voice-m1-001"
    assert "VoiceOps Milestone 1 Voice Operator" in markdown
    assert "Proofs" in markdown
    assert "Live Probe Boundary" in markdown
    assert "Supplied Live Evidence" in markdown
    assert "VoiceOps Milestone 1 Live Probe Closure" in closure_markdown
    assert "live-voice-evidence-scaffold/manifest.json" in closure_markdown
    assert "voiceops.realtime_voice_live_evidence_manifest.v1" in closure_markdown
    assert "source_artifact" in closure_markdown
    assert "collector_attestation" in closure_markdown
    assert "collector_name" in closure_markdown
    assert "collector_version" in closure_markdown
    assert "command_argv" in closure_markdown
    assert "redacted_artifact_sha256" in closure_markdown
    assert "parent_manifest_sha256" in closure_markdown
    assert "kind/evidence_type" in closure_markdown
    assert "--validate-live-evidence" in closure_markdown
    assert "artifacts/realtime-voice-evidence/live-current/sidecar-session.json" in closure_markdown
    assert "sidecar_session" in closure_markdown
    assert "live_turn" in closure_markdown
    assert "sidecar-session.json" in closure_markdown
    assert "raw transcript text" in closure_markdown
    assert "hand-edit manifest.json" in closure_markdown
    assert "example_only" in closure_markdown


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
    assert "missing_schema_version" in result["issues"]
    assert "discord_live_probe:missing_source_artifact" in result["issues"]
    assert "sidecar_session:missing_source_artifact" in result["issues"]
    assert "live_turn:missing_source_artifact" in result["issues"]
    assert "discord_live_probe:inbound_not_observed" in result["issues"]


def test_live_evidence_example_is_not_accepted_as_proof():
    result = validate_live_probe_evidence(build_live_probe_evidence_example())

    assert result["overall_status"] == "partial_live_evidence"
    assert "example_only_evidence_not_accepted" in result["issues"]


def test_voice_operator_accepts_complete_supplied_live_evidence_without_changing_safety_mode(tmp_path):
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["source_artifact"] = str(tmp_path / "discord-live-probe.json")
    evidence["sidecar_session"]["source_artifact"] = str(tmp_path / "sidecar-session.json")
    evidence["live_turn"]["source_artifact"] = str(tmp_path / "live-turn.json")
    _write_attested_section(tmp_path / "discord-live-probe.json", evidence["discord_live_probe"], "discord_live_probe")
    _write_attested_section(tmp_path / "sidecar-session.json", evidence["sidecar_session"], "sidecar_session")
    _write_attested_section(tmp_path / "live-turn.json", evidence["live_turn"], "live_turn")
    live_evidence = validate_live_probe_evidence(evidence)
    report = _voice_operator_report(live_evidence=live_evidence)

    assert report["mode"]["discord_network"] is False
    assert report["mode"]["env_secret_reads"] is False
    assert report["requirements"]["live_discord_join"] is False
    assert report["requirements"]["live_evidence_supplied"] is True
    assert report["live_evidence"]["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert report["live_probe_required_for_completion"]["missing_gates"] == []
    assert report["proofs"]["live_evidence"]["ok"] is True
    assert not any("collector_attestation" in issue for issue in live_evidence["issues"])


def test_live_evidence_rejects_stale_source_artifact_attestation_hash(tmp_path):
    evidence = _complete_live_evidence()
    source_path = tmp_path / "discord-live-probe.json"
    evidence["discord_live_probe"]["source_artifact"] = str(source_path)
    _write_attested_section(source_path, evidence["discord_live_probe"], "discord_live_probe")
    source_path.write_text(
        json.dumps({"kind": "discord_live_probe", "redacted": True, "updated": True}),
        encoding="utf-8",
    )

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:collector_attestation_redacted_sha256_mismatch" in live_evidence["issues"]


def test_live_evidence_rejects_inverted_collector_attestation_window():
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["collector_attestation"]["started_at"] = "2026-06-29T00:00:02Z"
    evidence["discord_live_probe"]["collector_attestation"]["finished_at"] = "2026-06-29T00:00:01Z"

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:collector_attestation_invalid:timestamp_window" in live_evidence["issues"]


def test_live_evidence_rejects_sensitive_collector_attestation_command_argv():
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["collector_attestation"]["command_argv"] = [
        "voiceops-live-collector",
        "--notify=+15551234567",
    ]

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:collector_attestation_secret_or_phone_like_command_argv" in live_evidence["issues"]


def test_live_evidence_allows_spark_matrix_test_path_in_collector_attestation_command_argv():
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["collector_attestation"]["command_argv"] = [
        "pytest",
        "tests/scripts/test_voiceops_spark_matrix.py",
    ]

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:collector_attestation_secret_or_phone_like_command_argv" not in live_evidence["issues"]


def test_live_evidence_rejects_fake_parent_manifest_attestation_hash(tmp_path):
    evidence = _complete_live_evidence()
    source_path = tmp_path / "discord-live-probe.json"
    evidence["discord_live_probe"]["source_artifact"] = str(source_path)
    _write_attested_section(source_path, evidence["discord_live_probe"], "discord_live_probe")
    evidence["discord_live_probe"]["collector_attestation"]["parent_manifest_sha256"] = "d" * 64

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:collector_attestation_parent_manifest_sha256_mismatch" in live_evidence["issues"]


def test_live_evidence_rejects_source_artifact_secret_and_phone_leak(tmp_path):
    evidence = _complete_live_evidence()
    source_path = tmp_path / "discord-live-probe.json"
    evidence["discord_live_probe"]["source_artifact"] = str(source_path)
    source_path.write_text(
        json.dumps(
            {
                "kind": "discord_live_probe",
                "redacted": True,
                "api_key": "sk_live_123456789abcdef",
                "operator_note": "call +15551234567 after the test",
            }
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(
        json.dumps(
            {
                "kind": "discord_live_probe",
                "redacted": True,
                "api_key": "sk_live_123456789abcdef",
                "operator_note": "call +15551234567 after the test",
            },
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    evidence["discord_live_probe"]["collector_attestation"]["redacted_artifact_sha256"] = source_sha256
    evidence["discord_live_probe"]["collector_attestation"]["parent_manifest_sha256"] = source_sha256

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:source_artifact_forbidden_field:api_key" in live_evidence["issues"]
    assert "discord_live_probe:source_artifact_secret_or_phone_like_value:api_key" in live_evidence["issues"]
    assert "discord_live_probe:source_artifact_secret_or_phone_like_value:operator_note" in live_evidence["issues"]


def test_live_evidence_rejects_source_artifact_voice_capability_denial(tmp_path):
    evidence = _complete_live_evidence()
    source_path = tmp_path / "live-turn.json"
    evidence["live_turn"]["source_artifact"] = str(source_path)
    source_path.write_text(
        json.dumps(
            {
                "kind": "live_turn",
                "redacted": True,
                "assistant_text": "I cannot hear you in Discord voice.",
            }
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(
        json.dumps(
            {
                "kind": "live_turn",
                "redacted": True,
                "assistant_text": "I cannot hear you in Discord voice.",
            },
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    evidence["live_turn"]["collector_attestation"]["redacted_artifact_sha256"] = source_sha256
    evidence["live_turn"]["collector_attestation"]["parent_manifest_sha256"] = source_sha256

    live_evidence = validate_live_probe_evidence(evidence)

    assert "live_turn:source_artifact_forbidden_field:assistant_text" in live_evidence["issues"]
    assert "live_turn:source_artifact_voice_capability_denial_text:assistant_text" in live_evidence["issues"]


def test_live_evidence_rejects_complete_hand_authored_sections_without_collector_attestation(tmp_path):
    evidence = _complete_live_evidence()
    for section_name in ("discord_live_probe", "sidecar_session", "live_turn"):
        evidence[section_name]["source_artifact"] = str(tmp_path / f"{section_name}.json")
        evidence[section_name].pop("collector_attestation")
        (tmp_path / f"{section_name}.json").write_text(json.dumps(evidence[section_name]), encoding="utf-8")

    live_evidence = validate_live_probe_evidence(evidence)

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "discord_live_probe:missing_collector_attestation" in live_evidence["issues"]
    assert "sidecar_session:missing_collector_attestation" in live_evidence["issues"]
    assert "live_turn:missing_collector_attestation" in live_evidence["issues"]


def test_voice_operator_rejects_loaded_evidence_with_missing_source_artifact_files(tmp_path):
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["source_artifact"] = "missing-discord-live-probe.json"
    evidence["sidecar_session"]["source_artifact"] = "missing-sidecar-session.json"
    evidence["live_turn"]["source_artifact"] = "missing-live-turn.json"
    evidence_path = tmp_path / "live-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    live_evidence = _load_live_evidence([evidence_path])

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "discord_live_probe:source_artifact_not_found" in live_evidence["issues"]
    assert "sidecar_session:source_artifact_not_found" in live_evidence["issues"]
    assert "live_turn:source_artifact_not_found" in live_evidence["issues"]
    assert (
        f"discord_live_probe:source_artifact_not_found_candidates:{tmp_path / 'missing-discord-live-probe.json'}"
        in live_evidence["issues"]
    )
    assert (
        f"sidecar_session:source_artifact_not_found_candidates:{tmp_path / 'missing-sidecar-session.json'}"
        in live_evidence["issues"]
    )
    assert (
        f"live_turn:source_artifact_not_found_candidates:{tmp_path / 'missing-live-turn.json'}"
        in live_evidence["issues"]
    )
    report = _voice_operator_report(live_evidence=live_evidence)
    assert report["status"] == "needs_live_probe"
    assert report["proofs"]["live_evidence"]["ok"] is False
    assert report["live_probe_required_for_completion"]["missing_gates"] == [
        "discord_join",
        "discord_playback",
        "live_receiver",
        "production_sidecar",
        "live_turn",
    ]


def test_live_evidence_rejects_template_source_artifact_placeholders():
    live_evidence = validate_live_probe_evidence(_complete_live_evidence())

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "discord_live_probe:template_source_artifact_not_accepted" in live_evidence["issues"]
    assert "sidecar_session:template_source_artifact_not_accepted" in live_evidence["issues"]
    assert "live_turn:template_source_artifact_not_accepted" in live_evidence["issues"]


def test_voice_operator_loaded_evidence_does_not_resolve_source_artifacts_from_cwd(monkeypatch, tmp_path):
    evidence_dir = tmp_path / "evidence-dir"
    cwd_dir = tmp_path / "cwd"
    evidence_dir.mkdir()
    cwd_dir.mkdir()
    for name in ("cwd-discord-live-probe.json", "cwd-sidecar-session.json", "cwd-live-turn.json"):
        (cwd_dir / name).write_text("{}", encoding="utf-8")
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["source_artifact"] = "cwd-discord-live-probe.json"
    evidence["sidecar_session"]["source_artifact"] = "cwd-sidecar-session.json"
    evidence["live_turn"]["source_artifact"] = "cwd-live-turn.json"
    evidence_path = evidence_dir / "live-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    monkeypatch.chdir(cwd_dir)

    live_evidence = _load_live_evidence([evidence_path])

    assert "discord_live_probe:source_artifact_not_found" in live_evidence["issues"]
    assert "sidecar_session:source_artifact_not_found" in live_evidence["issues"]
    assert "live_turn:source_artifact_not_found" in live_evidence["issues"]
    assert (
        f"discord_live_probe:source_artifact_not_found_candidates:{evidence_dir / 'cwd-discord-live-probe.json'}"
        in live_evidence["issues"]
    )


def test_voice_operator_manifest_reports_do_not_resolve_from_cwd(monkeypatch, tmp_path):
    evidence_dir = tmp_path / "evidence-dir"
    cwd_dir = tmp_path / "cwd"
    evidence_dir.mkdir()
    cwd_dir.mkdir()
    for name in ("cwd-discord-live-probe.json", "cwd-sidecar-session.json", "cwd-live-turn.json"):
        (cwd_dir / name).write_text(json.dumps({"kind": name.removesuffix(".json")}), encoding="utf-8")
    manifest_path = evidence_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {
                    "discord_live_probe": "cwd-discord-live-probe.json",
                    "sidecar_session": "cwd-sidecar-session.json",
                    "live_turn": "cwd-live-turn.json",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(cwd_dir)

    live_evidence = _load_live_evidence([manifest_path])

    assert "live_evidence_manifest:discord_live_probe:live_evidence_file_not_found" in live_evidence["issues"]
    assert "live_evidence_manifest:sidecar_session:live_evidence_file_not_found" in live_evidence["issues"]
    assert "live_evidence_manifest:live_turn:live_evidence_file_not_found" in live_evidence["issues"]


def test_voice_operator_manifest_reports_do_not_fallback_to_basename(tmp_path):
    discord_probe = _complete_live_evidence()["discord_live_probe"]
    discord_probe["kind"] = "discord_live_probe"
    discord_probe["source_artifact"] = str(tmp_path / "discord-live-probe.json")
    (tmp_path / "discord-live-probe.json").write_text(json.dumps(discord_probe), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {
                    "discord_live_probe": "sections/discord-live-probe.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert "live_evidence_manifest:discord_live_probe:live_evidence_file_not_found" in live_evidence["issues"]


def test_voice_operator_manifest_rejects_absolute_report_path(tmp_path):
    discord_probe = _complete_live_evidence()["discord_live_probe"]
    discord_probe["kind"] = "discord_live_probe"
    report_path = tmp_path / "discord-live-probe.json"
    report_path.write_text(json.dumps(discord_probe), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {
                    "discord_live_probe": str(report_path),
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert "live_evidence_manifest:discord_live_probe:report_path:absolute_path_not_allowed" in live_evidence["issues"]


def test_voice_operator_manifest_rejects_parent_escape_report_path(tmp_path):
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    report_path = tmp_path / "discord-live-probe.json"
    discord_probe = _complete_live_evidence()["discord_live_probe"]
    discord_probe["kind"] = "discord_live_probe"
    report_path.write_text(json.dumps(discord_probe), encoding="utf-8")
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {
                    "discord_live_probe": "../discord-live-probe.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert (
        "live_evidence_manifest:discord_live_probe:report_path:parent_traversal_not_allowed"
        in live_evidence["issues"]
    )


def test_voice_operator_manifest_rejects_user_home_report_path(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {
                    "discord_live_probe": "~/discord-live-probe.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert "live_evidence_manifest:discord_live_probe:report_path:user_home_not_allowed" in live_evidence["issues"]


def test_voice_operator_manifest_rejects_symlink_escape_report_path(tmp_path):
    manifest_dir = tmp_path / "manifest"
    sections_dir = manifest_dir / "sections"
    sections_dir.mkdir(parents=True)
    outside_report = tmp_path / "discord-live-probe.json"
    discord_probe = _complete_live_evidence()["discord_live_probe"]
    discord_probe["kind"] = "discord_live_probe"
    outside_report.write_text(json.dumps(discord_probe), encoding="utf-8")
    (sections_dir / "discord-live-probe.json").symlink_to(outside_report)
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {
                    "discord_live_probe": "sections/discord-live-probe.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert "live_evidence_manifest:discord_live_probe:report_path:path_escape_not_allowed" in live_evidence["issues"]


def test_voice_operator_manifest_cycle_returns_validation_issue(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {
                    "combined": "manifest.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "live_evidence_manifest:combined:cycle_detected" in live_evidence["issues"]


def test_voice_operator_rejects_source_artifact_directory(tmp_path):
    evidence = _complete_live_evidence()
    for section_name in ("discord_live_probe", "sidecar_session", "live_turn"):
        artifact_dir = tmp_path / f"{section_name}.json"
        artifact_dir.mkdir()
        evidence[section_name]["source_artifact"] = str(artifact_dir)

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:source_artifact_not_file" in live_evidence["issues"]
    assert "sidecar_session:source_artifact_not_file" in live_evidence["issues"]
    assert "live_turn:source_artifact_not_file" in live_evidence["issues"]


def test_voice_operator_ingests_realtime_live_evidence_manifest(tmp_path):
    discord_probe = {
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
        "latency_metrics_ms": _complete_discord_latency_metrics(),
    }
    sidecar = {
        "kind": "sidecar_session",
        "collector_attestation": _collector_attestation("sidecar_session"),
        **_complete_sidecar_session_fields(),
    }
    live_turn = {
        "kind": "live_turn",
        "collector_attestation": _collector_attestation("live_turn"),
        "transcript_observed": True,
        "assistant_audio_observed": True,
        "barge_in_observed": True,
        "spoken_reply_short": True,
        "no_voice_denial_observed": True,
        "speech_end_to_first_audio_ms": 950,
        "barge_in_stop_ms": 80,
    }
    _write_attested_section(tmp_path / "discord-live-probe.json", discord_probe, "discord_live_probe")
    _write_attested_section(tmp_path / "sidecar-session.json", sidecar, "sidecar_session")
    _write_attested_section(tmp_path / "live-turn.json", live_turn, "live_turn")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
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
    report = _voice_operator_report(live_evidence=live_evidence)

    assert live_evidence["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert live_evidence["issues"] == []
    assert live_evidence["discord_live_probe"]["join_ok"] is True
    assert live_evidence["section_refs"] == {
        "discord_live_probe": {
            "source_artifact": str(tmp_path / "discord-live-probe.json"),
            "section": "discord_live_probe",
            "wrapper_artifact": str(tmp_path / "discord-live-probe.json"),
        },
        "sidecar_session": {
            "source_artifact": str(tmp_path / "sidecar-session.json"),
            "section": "sidecar_session",
            "wrapper_artifact": str(tmp_path / "sidecar-session.json"),
        },
        "live_turn": {
            "source_artifact": str(tmp_path / "live-turn.json"),
            "section": "live_turn",
            "wrapper_artifact": str(tmp_path / "live-turn.json"),
        },
    }
    assert report["live_probe_required_for_completion"]["missing_gates"] == []
    assert report["status"] == "live_evidence_supplied_not_readiness_claim"


def test_voice_operator_ingests_repeated_standalone_live_evidence_files(tmp_path):
    discord_path = tmp_path / "actual-discord-probe.json"
    sidecar_path = tmp_path / "actual-sidecar-session.json"
    turn_path = tmp_path / "actual-live-turn.json"
    _write_attested_section(
        discord_path,
        {
            "schema_version": "voiceops.milestone1.live_voice_evidence.v1",
            "kind": "discord_live_probe",
            "source_artifact": discord_path.name,
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
            "latency_metrics_ms": _complete_discord_latency_metrics(),
        },
        "discord_live_probe",
    )
    _write_attested_section(
        sidecar_path,
        {
            "schema_version": "voiceops.milestone1.live_voice_evidence.v1",
            "kind": "sidecar_session",
            "source_artifact": sidecar_path.name,
            "sidecar_running": True,
            "sidecar_healthy": True,
            "session_started": True,
            "session_closed": True,
            "shutdown_bounded": True,
            "shutdown_timed_out": False,
            "fallback_mode_visible": True,
            "fallback_reason": "none",
            "sidecar_mode": "production",
            "healthcheck_observed": True,
            "provider_transport_observed": True,
            "session_id_redacted": True,
            "latency_metrics_ms": {"session_start_ms": 110, "shutdown_ms": 80},
        },
        "sidecar_session",
    )
    _write_attested_section(
        turn_path,
        {
            "schema_version": "voiceops.milestone1.live_voice_evidence.v1",
            "kind": "live_turn",
            "source_artifact": turn_path.name,
            "transcript_observed": True,
            "assistant_audio_observed": True,
            "barge_in_observed": True,
            "spoken_reply_short": True,
            "no_voice_denial_observed": True,
            "speech_end_to_first_audio_ms": 950,
            "barge_in_stop_ms": 80,
        },
        "live_turn",
    )

    live_evidence = _load_live_evidence([discord_path, sidecar_path, turn_path])

    assert live_evidence["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert live_evidence["issues"] == []
    assert live_evidence["section_refs"] == {
        "discord_live_probe": {
            "source_artifact": discord_path.name,
            "section": "discord_live_probe",
        },
        "sidecar_session": {
            "source_artifact": sidecar_path.name,
            "section": "sidecar_session",
        },
        "live_turn": {
            "source_artifact": turn_path.name,
            "section": "live_turn",
        },
    }


def test_voice_operator_rejects_standalone_section_without_identity(tmp_path):
    sidecar_path = tmp_path / "actual-sidecar-session.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.milestone1.live_voice_evidence.v1",
                "source_artifact": sidecar_path.name,
                "sidecar_running": True,
                "sidecar_healthy": True,
                "session_started": True,
                "session_closed": True,
                "shutdown_bounded": True,
                "shutdown_timed_out": False,
                "fallback_mode_visible": True,
                "fallback_reason": "none",
                "sidecar_mode": "production",
                "healthcheck_observed": True,
                "provider_transport_observed": True,
                "session_id_redacted": True,
                "latency_metrics_ms": {"session_start_ms": 110, "shutdown_ms": 80},
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([sidecar_path])

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "missing_standalone_report_identity" in live_evidence["issues"]


def test_voice_operator_accepts_standalone_section_evidence_type_identity(tmp_path):
    turn_path = tmp_path / "actual-live-turn.json"
    turn_path.write_text(
        json.dumps(
            {
                "evidence_type": "live_turn",
                "source_artifact": turn_path.name,
                "transcript_observed": True,
                "assistant_audio_observed": True,
                "barge_in_observed": True,
                "spoken_reply_short": True,
                "no_voice_denial_observed": True,
                "speech_end_to_first_audio_ms": 950,
                "barge_in_stop_ms": 80,
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([turn_path])

    assert "missing_standalone_report_identity" not in live_evidence["issues"]
    assert "invalid_standalone_report_identity" not in " ".join(live_evidence["issues"])
    assert "live_turn:missing_source_artifact" not in live_evidence["issues"]


def test_voice_operator_rejects_standalone_section_with_wrong_identity(tmp_path):
    turn_path = tmp_path / "actual-live-turn.json"
    turn_path.write_text(
        json.dumps(
            {
                "kind": "calendar_event",
                "source_artifact": turn_path.name,
                "transcript_observed": True,
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([turn_path])

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "invalid_standalone_report_identity:calendar_event" in live_evidence["issues"]


def test_voice_operator_rejects_combined_manifest_placeholder_source_artifacts(tmp_path):
    evidence = _complete_live_evidence()
    (tmp_path / "all-live-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "ok": True,
                "reports": {
                    "combined": "all-live-evidence.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "discord_live_probe:template_source_artifact_not_accepted" in live_evidence["issues"]
    assert "sidecar_session:template_source_artifact_not_accepted" in live_evidence["issues"]
    assert "live_turn:template_source_artifact_not_accepted" in live_evidence["issues"]


def test_voice_operator_rejects_template_source_artifact_path_variants():
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["source_artifact"] = "./discord-live-probe.json"
    evidence["sidecar_session"]["source_artifact"] = "./sidecar-session.json"
    evidence["live_turn"]["source_artifact"] = "./live-turn.json"

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:template_source_artifact_not_accepted" in live_evidence["issues"]
    assert "sidecar_session:template_source_artifact_not_accepted" in live_evidence["issues"]
    assert "live_turn:template_source_artifact_not_accepted" in live_evidence["issues"]


def test_voice_operator_rejects_anonymous_combined_manifest_report(tmp_path):
    evidence = _complete_live_evidence()
    evidence.pop("schema_version")
    evidence["discord_live_probe"]["source_artifact"] = str(tmp_path / "all-live-evidence.json")
    evidence["sidecar_session"]["source_artifact"] = str(tmp_path / "all-live-evidence.json")
    evidence["live_turn"]["source_artifact"] = str(tmp_path / "all-live-evidence.json")
    (tmp_path / "all-live-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {"combined": "all-live-evidence.json"},
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "live_evidence_manifest:combined:missing_report_identity" in live_evidence["issues"]


def test_voice_operator_rejects_combined_manifest_missing_nested_source_artifact(tmp_path):
    evidence = _complete_live_evidence()
    evidence["live_turn"].pop("source_artifact")
    evidence["discord_live_probe"]["source_artifact"] = str(tmp_path / "all-live-evidence.json")
    evidence["sidecar_session"]["source_artifact"] = str(tmp_path / "all-live-evidence.json")
    (tmp_path / "all-live-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "ok": True,
                "reports": {
                    "combined": "all-live-evidence.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "live_turn:missing_source_artifact" in live_evidence["issues"]


def test_voice_operator_manifest_nested_report_source_artifacts_resolve_relative_to_report(tmp_path):
    report_dir = tmp_path / "reports"
    raw_dir = report_dir / "raw"
    raw_dir.mkdir(parents=True)
    for name in ("discord.json", "sidecar.json", "turn.json"):
        (raw_dir / name).write_text("{}", encoding="utf-8")
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["source_artifact"] = "raw/discord.json"
    evidence["sidecar_session"]["source_artifact"] = "raw/sidecar.json"
    evidence["live_turn"]["source_artifact"] = "raw/turn.json"
    raw_payload_sha256 = hashlib.sha256(b"{}").hexdigest()
    for section_name in ("discord_live_probe", "sidecar_session", "live_turn"):
        evidence[section_name]["collector_attestation"]["raw_artifact_sha256"] = raw_payload_sha256
        evidence[section_name]["collector_attestation"]["redacted_artifact_sha256"] = raw_payload_sha256
        evidence[section_name]["collector_attestation"]["parent_manifest_sha256"] = raw_payload_sha256
    combined_path = report_dir / "combined.json"
    combined_path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "ok": True,
                "reports": {
                    "combined": "reports/combined.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert live_evidence["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert live_evidence["issues"] == []
    assert live_evidence["section_refs"]["sidecar_session"]["source_artifact"] == str(raw_dir / "sidecar.json")
    assert live_evidence["section_refs"]["sidecar_session"]["wrapper_artifact"] == str(report_dir / "combined.json")
    assert live_evidence["section_refs"]["sidecar_session"]["reported_source_artifact"] == "raw/sidecar.json"


def test_voice_operator_manifest_rejects_nested_source_artifact_parent_escape(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    evidence = _complete_live_evidence()
    evidence["sidecar_session"]["source_artifact"] = "../sidecar-raw.json"
    combined_path = report_dir / "combined.json"
    combined_path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "ok": True,
                "reports": {
                    "combined": "reports/combined.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert (
        "live_evidence_manifest:combined:sidecar_session.source_artifact:parent_traversal_not_allowed"
        in live_evidence["issues"]
    )


def test_voice_operator_manifest_rejects_nested_source_artifact_absolute_and_home_refs(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["source_artifact"] = str(report_dir / "raw" / "discord.json")
    evidence["sidecar_session"]["source_artifact"] = "~/sidecar-raw.json"
    combined_path = report_dir / "combined.json"
    combined_path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "ok": True,
                "reports": {
                    "combined": "reports/combined.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert (
        "live_evidence_manifest:combined:discord_live_probe.source_artifact:absolute_path_not_allowed"
        in live_evidence["issues"]
    )
    assert (
        "live_evidence_manifest:combined:sidecar_session.source_artifact:user_home_not_allowed"
        in live_evidence["issues"]
    )


def test_voice_operator_manifest_rejects_nested_source_artifact_symlink_escape(tmp_path):
    report_dir = tmp_path / "reports"
    raw_dir = report_dir / "raw"
    raw_dir.mkdir(parents=True)
    outside_source = tmp_path / "turn-raw.json"
    outside_source.write_text("{}", encoding="utf-8")
    (raw_dir / "turn.json").symlink_to(outside_source)
    evidence = _complete_live_evidence()
    evidence["live_turn"]["source_artifact"] = "raw/turn.json"
    combined_path = report_dir / "combined.json"
    combined_path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "ok": True,
                "reports": {
                    "combined": "reports/combined.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert (
        "live_evidence_manifest:combined:live_turn.source_artifact:path_escape_not_allowed"
        in live_evidence["issues"]
    )


def test_live_evidence_rejects_complete_payload_without_schema_and_source_artifacts():
    evidence = build_live_probe_evidence_template()
    evidence.pop("schema_version")
    evidence["discord_live_probe"].pop("source_artifact")
    evidence["sidecar_session"].pop("source_artifact")
    evidence["live_turn"].pop("source_artifact")
    evidence["discord_live_probe"].update(
        {
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
            "receiver_frames": 12,
            "receiver_speech_start": 1,
            "inbound_observed": True,
            "disconnected": True,
            "require_inbound": True,
            "latency_metrics_ms": _complete_discord_latency_metrics(),
        }
    )
    evidence["sidecar_session"].update(
        {
            "collector_attestation": _collector_attestation("sidecar_session"),
            **_complete_sidecar_session_fields(),
        }
    )
    evidence["live_turn"].update(
        {
            "collector_attestation": _collector_attestation("live_turn"),
            "transcript_observed": True,
            "assistant_audio_observed": True,
            "barge_in_observed": True,
            "spoken_reply_short": True,
            "no_voice_denial_observed": True,
            "speech_end_to_first_audio_ms": 900,
            "barge_in_stop_ms": 90,
        }
    )

    result = validate_live_probe_evidence(evidence)

    assert result["overall_status"] == "partial_live_evidence"
    assert result["issues"] == [
        "discord_live_probe:missing_source_artifact",
        "live_turn:missing_source_artifact",
        "missing_schema_version",
        "sidecar_session:missing_source_artifact",
    ]


def test_voice_operator_rejects_manifest_with_example_only_referenced_section(tmp_path):
    discord_probe = build_live_probe_evidence_example()["discord_live_probe"]
    discord_probe["example_only"] = True
    sidecar = build_live_probe_evidence_template()["sidecar_session"]
    sidecar.update(
        {
            **_complete_sidecar_session_fields(),
        }
    )
    live_turn = build_live_probe_evidence_template()["live_turn"]
    live_turn.update(
        {
            "transcript_observed": True,
            "assistant_audio_observed": True,
            "barge_in_observed": True,
            "spoken_reply_short": True,
            "no_voice_denial_observed": True,
            "speech_end_to_first_audio_ms": 950,
            "barge_in_stop_ms": 80,
        }
    )
    (tmp_path / "discord-live-probe.json").write_text(json.dumps(discord_probe), encoding="utf-8")
    (tmp_path / "sidecar-session.json").write_text(json.dumps(sidecar), encoding="utf-8")
    (tmp_path / "live-turn.json").write_text(json.dumps(live_turn), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
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

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "example_only_evidence_not_accepted" in live_evidence["issues"]
    assert "live_evidence_manifest:discord_live_probe:example_only_evidence_not_accepted" in live_evidence["issues"]


def test_voice_operator_rejects_manifest_with_missing_or_invalid_schema(tmp_path):
    discord_probe = build_live_probe_evidence_example()["discord_live_probe"]
    discord_probe.pop("example_only", None)
    sidecar = build_live_probe_evidence_template()["sidecar_session"]
    sidecar.update(
        {
            **_complete_sidecar_session_fields(),
        }
    )
    live_turn = build_live_probe_evidence_template()["live_turn"]
    live_turn.update(
        {
            "transcript_observed": True,
            "assistant_audio_observed": True,
            "barge_in_observed": True,
            "spoken_reply_short": True,
            "no_voice_denial_observed": True,
            "speech_end_to_first_audio_ms": 950,
            "barge_in_stop_ms": 80,
        }
    )
    (tmp_path / "discord-live-probe.json").write_text(json.dumps(discord_probe), encoding="utf-8")
    (tmp_path / "sidecar-session.json").write_text(json.dumps(sidecar), encoding="utf-8")
    (tmp_path / "live-turn.json").write_text(json.dumps(live_turn), encoding="utf-8")

    base_manifest = {
        "ok": True,
        "reports": {
            "discord_live_probe": "discord-live-probe.json",
            "sidecar_session": "sidecar-session.json",
            "live_turn": "live-turn.json",
        },
    }
    missing_schema_path = tmp_path / "missing-schema-manifest.json"
    missing_schema_path.write_text(json.dumps(base_manifest), encoding="utf-8")
    missing_schema = _load_live_evidence([missing_schema_path])
    assert missing_schema["overall_status"] == "partial_live_evidence"
    assert "live_evidence_manifest:missing_schema_version" in missing_schema["issues"]

    invalid_schema_path = tmp_path / "invalid-schema-manifest.json"
    invalid_schema_path.write_text(json.dumps({**base_manifest, "schema_version": "wrong.schema.v1"}), encoding="utf-8")
    invalid_schema = _load_live_evidence([invalid_schema_path])
    assert invalid_schema["overall_status"] == "partial_live_evidence"
    assert "live_evidence_manifest:invalid_schema_version" in invalid_schema["issues"]


def test_live_evidence_rejects_nested_example_only_sections():
    evidence = build_live_probe_evidence_template()
    evidence["discord_live_probe"].update(
        {
            "example_only": True,
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
            "example_only": True,
            **_complete_sidecar_session_fields(),
        }
    )
    evidence["live_turn"].update(
        {
            "example_only": True,
            "transcript_observed": True,
            "assistant_audio_observed": True,
            "barge_in_observed": True,
            "spoken_reply_short": True,
            "no_voice_denial_observed": True,
            "speech_end_to_first_audio_ms": 950,
            "barge_in_stop_ms": 80,
        }
    )

    result = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:example_only_evidence_not_accepted" in result["issues"]
    assert "sidecar_session:example_only_evidence_not_accepted" in result["issues"]
    assert "live_turn:example_only_evidence_not_accepted" in result["issues"]


def test_live_evidence_rejects_raw_text_secret_and_denial_fields():
    evidence = _complete_live_evidence()
    evidence["live_turn"]["assistant_text"] = "I cannot hear voice. I only process typed text."
    evidence["live_turn"]["raw_transcript"] = "call me at +15551234567"
    evidence["sidecar_session"]["api_key"] = "sk-car-exampletoken123456"

    result = validate_live_probe_evidence(evidence)

    assert "live_turn.assistant_text:forbidden_evidence_field" in result["issues"]
    assert "live_turn.assistant_text:voice_capability_denial_text" in result["issues"]
    assert "live_turn.raw_transcript:forbidden_evidence_field" in result["issues"]
    assert "live_turn.raw_transcript:secret_or_phone_like_value" in result["issues"]
    assert "sidecar_session.api_key:forbidden_evidence_field" in result["issues"]
    assert "sidecar_session.api_key:secret_or_phone_like_value" in result["issues"]


def test_live_evidence_requires_bounded_sidecar_shutdown():
    evidence = _complete_live_evidence()
    evidence["sidecar_session"].pop("shutdown_bounded")
    evidence["sidecar_session"]["shutdown_timed_out"] = True
    evidence["sidecar_session"]["latency_metrics_ms"] = {}

    result = validate_live_probe_evidence(evidence)

    assert "sidecar_session:missing_shutdown_ms" in result["issues"]
    assert "sidecar_session:shutdown_bounded_not_true" in result["issues"]
    assert "sidecar_session:shutdown_timed_out_not_false" in result["issues"]
    assert result["sidecar_session"]["ok"] is False


def test_live_evidence_requires_sidecar_production_provenance_and_fallback_reason():
    evidence = _complete_live_evidence()
    evidence["sidecar_session"].pop("sidecar_mode")
    evidence["sidecar_session"]["healthcheck_observed"] = False
    evidence["sidecar_session"]["provider_transport_observed"] = False
    evidence["sidecar_session"]["session_id_redacted"] = False
    evidence["sidecar_session"]["fallback_reason"] = ""

    result = validate_live_probe_evidence(evidence)

    assert "sidecar_session:sidecar_mode_not_production" in result["issues"]
    assert "sidecar_session:healthcheck_observed_not_true" in result["issues"]
    assert "sidecar_session:provider_transport_observed_not_true" in result["issues"]
    assert "sidecar_session:session_id_redacted_not_true" in result["issues"]
    assert "sidecar_session:missing_fallback_reason" in result["issues"]
    assert result["sidecar_session"]["ok"] is False


def test_live_evidence_requires_discord_and_sidecar_session_start_latencies():
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["latency_metrics_ms"].pop("connect_ms")
    evidence["discord_live_probe"]["latency_metrics_ms"]["disconnect_ms"] = -1
    evidence["sidecar_session"]["latency_metrics_ms"].pop("session_start_ms")

    result = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:missing_connect_ms" in result["issues"]
    assert "discord_live_probe:missing_disconnect_ms" in result["issues"]
    assert "sidecar_session:missing_session_start_ms" in result["issues"]
    assert result["discord_live_probe"]["latency_ok"] is False
    assert result["sidecar_session"]["ok"] is False


def test_live_turn_latency_boundaries_are_exact():
    evidence = _complete_live_evidence()
    evidence["live_turn"]["speech_end_to_first_audio_ms"] = 3000
    evidence["live_turn"]["barge_in_stop_ms"] = 150

    boundary = validate_live_probe_evidence(evidence)

    assert "live_turn:speech_end_to_first_audio_ms_over_target" not in boundary["issues"]
    assert "live_turn:barge_in_stop_ms_over_target" not in boundary["issues"]

    evidence["live_turn"]["speech_end_to_first_audio_ms"] = 3000.1
    evidence["live_turn"]["barge_in_stop_ms"] = 150.1
    over = validate_live_probe_evidence(evidence)

    assert "live_turn:speech_end_to_first_audio_ms_over_target" in over["issues"]
    assert "live_turn:barge_in_stop_ms_over_target" in over["issues"]

    evidence["live_turn"]["speech_end_to_first_audio_ms"] = -1
    evidence["live_turn"]["barge_in_stop_ms"] = "not-a-number"
    invalid = validate_live_probe_evidence(evidence)

    assert "live_turn:missing_speech_end_to_first_audio_ms" in invalid["issues"]
    assert "live_turn:missing_barge_in_stop_ms" in invalid["issues"]


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
    assert Path(payload["artifacts"]["async_oracle_smoke_json"]).exists()
    assert Path(payload["artifacts"]["events_jsonl"]).exists()


def test_parse_args_defaults_to_requested_artifact_dir():
    args = parse_args([])

    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.live_evidence == []
