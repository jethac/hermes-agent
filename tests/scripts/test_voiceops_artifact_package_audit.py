from __future__ import annotations

import json
from pathlib import Path

from scripts.voiceops_artifact_package_audit import audit_package, parse_args, write_audit
from scripts.voiceops_plan_run import build_plan_run, write_plan_run


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generate_package(tmp_path: Path, **plan_kwargs) -> Path:
    artifact_root = tmp_path / "artifacts"
    output_dir = artifact_root / "voiceops-plan" / "current"
    summary = build_plan_run(artifact_root=artifact_root, output_dir=output_dir, env={}, **plan_kwargs)
    write_plan_run(output_dir, summary)
    return artifact_root


def _rewrite_json_strings(value, rewrite):
    if isinstance(value, str):
        return rewrite(value)
    if isinstance(value, list):
        return [_rewrite_json_strings(item, rewrite) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_json_strings(item, rewrite) for key, item in value.items()}
    return value


def test_package_audit_accepts_generated_headless_package(tmp_path):
    artifact_root = _generate_package(tmp_path)

    report = audit_package(artifact_root)
    paths = write_audit(tmp_path / "audit", report)

    assert report["schema_version"] == "voiceops.artifact_package_audit.v1"
    assert report["artifact_id"] == "voiceops-artifact-package-audit"
    assert report["status"] == "pass"
    assert report["ok"] is True
    assert report["issues"] == []
    assert report["readiness_claim"] is False
    assert report["readiness_scope"] == "static_package_consistency_only"
    assert "does not satisfy live Discord" in report["readiness_note"]
    assert report["checked_artifact_count"] == 94
    assert str(artifact_root / "hackathon-voiceops-demo" / "current" / "operator-handoff-preview.json") in report[
        "checked_artifacts"
    ]
    assert str(artifact_root / "hackathon-voiceops-demo" / "current" / "operator-handoff-preview.md") in report[
        "checked_artifacts"
    ]
    assert str(artifact_root / "voiceops-plan" / "current" / "voiceops-plan-run.json") in report[
        "checked_artifacts"
    ]
    assert str(artifact_root / "voiceops-plan" / "current" / "readiness-closure-index.md") in report[
        "checked_artifacts"
    ]
    assert str(artifact_root / "voiceops-plan" / "current" / "operator-handoff.json") in report["checked_artifacts"]
    assert str(artifact_root / "voiceops-plan" / "current" / "operator-handoff.md") in report["checked_artifacts"]
    assert str(artifact_root / "voiceops-channel-policy" / "current" / "channel-policy.json") in report[
        "checked_artifacts"
    ]
    assert str(artifact_root / "voiceops-channel-policy" / "current" / "channel-policy-review.md") in report[
        "checked_artifacts"
    ]
    assert str(artifact_root / "voiceops-voice-operator" / "current" / "live-voice-evidence-scaffold" / "manifest.json") in report[
        "checked_artifacts"
    ]
    assert str(artifact_root / "voiceops-provisioning" / "current" / "provisioning-preflight-scaffold" / "provisioning-preflight-evidence.manifest.json") in report[
        "checked_artifacts"
    ]
    assert str(artifact_root / "voiceops-spark-matrix" / "current" / "spark-operator-runbook.md") in report[
        "checked_artifacts"
    ]
    assert str(artifact_root / "voiceops-operator-state" / "current" / "operator-state.md") in report[
        "checked_artifacts"
    ]
    assert report["safety"] == {
        "discord_io": False,
        "env_files_read": False,
        "live_spend": False,
        "network_io": False,
        "outbound_calls": False,
        "outbound_messages": False,
        "provider_provisioning": False,
        "secret_values_emitted": False,
        "spark_execution": False,
    }
    assert Path(paths["json"]).exists()
    audit_markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert audit_markdown.startswith("# VoiceOps Artifact Package Audit")
    assert "Readiness claim: no" in audit_markdown
    assert "static_package_consistency_only" in audit_markdown


def test_package_audit_accepts_allowlisted_readonly_discovery_safety_summary(tmp_path):
    artifact_root = _generate_package(tmp_path)
    plan_dir = artifact_root / "voiceops-plan" / "current"
    for name in ("voiceops-plan-run.json", "readiness-closure-index.json"):
        payload_path = plan_dir / name
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        payload["safety"]["network_io"] = True
        payload["safety"]["network_io_scope"] = "allowlisted_read_only_discovery"
        payload["safety"]["read_only_discovery_run_requested"] = True
        if isinstance(payload.get("closure_index"), dict):
            payload["closure_index"]["safety"]["network_io"] = True
            payload["closure_index"]["safety"]["network_io_scope"] = "allowlisted_read_only_discovery"
            payload["closure_index"]["safety"]["read_only_discovery_run_requested"] = True
        _write_json(payload_path, payload)
    markdown_path = plan_dir / "readiness-closure-index.md"
    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8").replace(
            "artifact-only; no network I/O",
            "artifact-only; read-only discovery network possible only when explicitly requested",
        ),
        encoding="utf-8",
    )

    report = audit_package(artifact_root)

    assert report["ok"] is True
    assert "closure_markdown:missing_artifact_only_safety" not in report["issues"]


def test_package_audit_rejects_discord_loopback_smoke_artifact_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    smoke_path = artifact_root / "voiceops-voice-operator" / "current" / "discord-loopback-smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["mixer_frames"] = 0
    _write_json(smoke_path, smoke)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "voice_operator_readiness:smoke_standalone_artifact_mismatch" in report["issues"]


def test_package_audit_rejects_async_oracle_smoke_artifact_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    smoke_path = artifact_root / "voiceops-voice-operator" / "current" / "async-oracle-smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["worker_overlap_proved"] = False
    _write_json(smoke_path, smoke)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "voice_operator_readiness:async_oracle_smoke_standalone_artifact_mismatch" in report["issues"]


def test_package_audit_rejects_discord_session_cleanup_smoke_artifact_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    smoke_path = artifact_root / "voiceops-voice-operator" / "current" / "discord-session-cleanup-smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["session_closed_sent"] = False
    _write_json(smoke_path, smoke)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "voice_operator_readiness:discord_session_cleanup_smoke_standalone_artifact_mismatch" in report["issues"]


def test_package_audit_rejects_sidecar_fail_closed_smoke_artifact_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    smoke_path = artifact_root / "voiceops-voice-operator" / "current" / "sidecar-fail-closed-smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["cancelled_observed"] = False
    _write_json(smoke_path, smoke)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "voice_operator_readiness:sidecar_fail_closed_smoke_standalone_artifact_mismatch" in report["issues"]


def test_package_audit_rejects_pcm_conversion_proof_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    readiness_path = artifact_root / "voiceops-voice-operator" / "current" / "voice-operator-readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["proofs"]["pcm_conversion"]["sidecar_pcm16_checksum"] = 999
    _write_json(readiness_path, readiness)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "voice_operator_readiness:proofs.pcm_conversion.sidecar_pcm16_checksum_mismatch" in report["issues"]


def test_package_audit_rejects_async_oracle_proof_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    readiness_path = artifact_root / "voiceops-voice-operator" / "current" / "voice-operator-readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["proofs"]["async_oracle_jobs"]["completed_jobs"] = 0
    readiness["proofs"]["async_oracle_jobs"]["worker_overlap_within_capacity"] = False
    readiness["proofs"]["async_oracle_jobs"]["shutdown_bounded_close_observed"] = False
    readiness["proofs"]["async_oracle_jobs"]["audit_scalar_payload_redacted"] = False
    readiness["proofs"]["async_oracle_jobs"]["external_frontend_terminal_correlation_observed"] = False
    readiness["proofs"]["async_oracle_jobs"]["external_frontend_completion_tool_call_id"] = "wrong-call"
    readiness["proofs"]["async_oracle_jobs"]["external_frontend_evidence_bundle_id_stable"] = False
    readiness["proofs"]["async_oracle_jobs"]["external_frontend_evidence_bundle_single_turn"] = False
    readiness["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_single_bundle_observed"] = False
    readiness["proofs"]["async_oracle_jobs"]["witness_fusion_early_single_bundle"] = False
    readiness["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_degraded_text_only_status"] = (
        "primary_audio"
    )
    readiness["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_promoted_ok"] = False
    _write_json(readiness_path, readiness)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "voice_operator_readiness:proofs.async_oracle_jobs.completed_jobs_mismatch" in report["issues"]
    assert (
        "voice_operator_readiness:proofs.async_oracle_jobs.worker_overlap_within_capacity_mismatch"
        in report["issues"]
    )
    assert (
        "voice_operator_readiness:proofs.async_oracle_jobs.shutdown_bounded_close_observed_mismatch"
        in report["issues"]
    )
    assert (
        "voice_operator_readiness:proofs.async_oracle_jobs.audit_scalar_payload_redacted_mismatch"
        in report["issues"]
    )
    assert (
        "voice_operator_readiness:proofs.async_oracle_jobs.external_frontend_terminal_correlation_observed_mismatch"
        in report["issues"]
    )
    assert (
        "voice_operator_readiness:proofs.async_oracle_jobs.external_frontend_completion_tool_call_id_mismatch"
        in report["issues"]
    )
    assert (
        "voice_operator_readiness:proofs.async_oracle_jobs.external_frontend_evidence_bundle_id_stable_mismatch"
        in report["issues"]
    )
    assert (
        "voice_operator_readiness:proofs.async_oracle_jobs.external_frontend_evidence_bundle_single_turn_mismatch"
        in report["issues"]
    )
    assert (
        "voice_operator_readiness:proofs.async_oracle_jobs.unpromoted_hypothesis_single_bundle_observed_mismatch"
        in report["issues"]
    )
    assert (
        "voice_operator_readiness:proofs.async_oracle_jobs.witness_fusion_early_single_bundle_mismatch"
        in report["issues"]
    )
    assert (
        "voice_operator_readiness:proofs.async_oracle_jobs.runtime_kame_action_gate_degraded_text_only_status_mismatch"
        in report["issues"]
    )
    assert (
        "voice_operator_readiness:proofs.async_oracle_jobs.runtime_kame_action_gate_promoted_ok_mismatch"
        in report["issues"]
    )


def test_package_audit_rejects_async_oracle_proof_identity_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    readiness_path = artifact_root / "voiceops-voice-operator" / "current" / "voice-operator-readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["proofs"]["async_oracle_jobs"]["kind"] = "generic_smoke"
    _write_json(readiness_path, readiness)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "voice_operator_readiness:proofs.async_oracle_jobs.kind_mismatch" in report["issues"]


def test_package_audit_rejects_discord_session_cleanup_proof_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    readiness_path = artifact_root / "voiceops-voice-operator" / "current" / "voice-operator-readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["proofs"]["discord_session_cleanup"]["event_order"] = []
    _write_json(readiness_path, readiness)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "voice_operator_readiness:proofs.discord_session_cleanup.event_order_mismatch" in report["issues"]


def test_package_audit_rejects_sidecar_fail_closed_proof_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    readiness_path = artifact_root / "voiceops-voice-operator" / "current" / "voice-operator-readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["proofs"]["sidecar_fail_closed"]["event_order"] = []
    readiness["proofs"]["sidecar_fail_closed"]["active_capacity_after_failure"] = 1
    readiness["proofs"]["sidecar_fail_closed"]["error_redacted"] = False
    _write_json(readiness_path, readiness)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "voice_operator_readiness:proofs.sidecar_fail_closed.event_order_mismatch" in report["issues"]
    assert (
        "voice_operator_readiness:proofs.sidecar_fail_closed.active_capacity_after_failure_mismatch"
        in report["issues"]
    )
    assert "voice_operator_readiness:proofs.sidecar_fail_closed.error_redacted_mismatch" in report["issues"]


def test_package_audit_rejects_missing_promised_runbook(tmp_path):
    artifact_root = _generate_package(tmp_path)
    missing_path = artifact_root / "voiceops-spark-matrix" / "current" / "spark-operator-runbook.md"
    missing_path.unlink()

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert (
        "package_artifact:voiceops-spark-matrix/current/spark-operator-runbook.md:"
        f"missing:{missing_path}"
    ) in report["issues"]


def test_package_audit_rejects_unexpected_package_artifact(tmp_path):
    artifact_root = _generate_package(tmp_path)
    stale_path = artifact_root / "hackathon-voiceops-demo" / "current" / "stale-live-claim.json"
    stale_path.write_text(json.dumps({"claim": "live_ready"}) + "\n", encoding="utf-8")

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "package_artifact:unexpected:hackathon-voiceops-demo/current/stale-live-claim.json" in report["issues"]


def test_package_audit_rejects_audit_ledger_drift_from_demo(tmp_path):
    artifact_root = _generate_package(tmp_path)
    ledger_path = artifact_root / "hackathon-voiceops-demo" / "current" / "audit-ledger.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[1]["status"] = "executed"
    ledger_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "audit_ledger:rows_mismatch_demo_audit_events" in report["issues"]


def test_package_audit_rejects_nemoclaw_kame_evidence_drift_from_operator_and_audit(tmp_path):
    artifact_root = _generate_package(tmp_path)
    packet_path = artifact_root / "hackathon-voiceops-demo" / "current" / "nemoclaw-action-packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["approval_required_actions"][0]["kame_evidence"]["audio_segment_ref"] = "artifact://tampered.wav"
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = audit_package(artifact_root)

    action_id = packet["approval_required_actions"][0]["action_id"]
    assert report["ok"] is False
    assert f"operator_state:{action_id}:pending_kame_evidence_mismatch" in report["issues"]
    assert f"audit_ledger:{action_id}:kame_evidence_mismatch" in report["issues"]


def test_package_audit_rejects_operator_state_event_drift_from_operator_state(tmp_path):
    artifact_root = _generate_package(tmp_path)
    events_path = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-state-events.jsonl"
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[1]["status"] = "executed"
    events_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "operator_state_events:rows_mismatch_operator_state" in report["issues"]


def test_package_audit_rejects_malformed_promised_scaffold_json(tmp_path):
    artifact_root = _generate_package(tmp_path)
    scaffold_path = (
        artifact_root
        / "voiceops-provisioning"
        / "current"
        / "provisioning-preflight-scaffold"
        / "provisioning-preflight-evidence.manifest.json"
    )
    scaffold_path.write_text("{not-json", encoding="utf-8")

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert (
        "package_artifact:voiceops-provisioning/current/provisioning-preflight-scaffold/"
        "provisioning-preflight-evidence.manifest.json:json_parse_failed:"
        "Expecting property name enclosed in double quotes"
    ) in report["issues"]


def test_package_audit_rejects_secret_like_values_in_artifacts(tmp_path):
    artifact_root = _generate_package(tmp_path)
    phone_context_path = artifact_root / "hackathon-voiceops-demo" / "current" / "phone-context.json"
    phone_context = json.loads(phone_context_path.read_text(encoding="utf-8"))
    phone_context["unsafe_demo_key"] = "sk_test_nonsecretfixture000"
    phone_context["unsafe_phone_number"] = "+15551234567"
    _write_json(phone_context_path, phone_context)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert any(
        issue.startswith("secret_scan:hackathon-voiceops-demo/current/phone-context.json:")
        and issue.endswith(":openai_or_stripe_secret_key")
        for issue in report["issues"]
    )
    assert any(
        issue.startswith("secret_scan:hackathon-voiceops-demo/current/phone-context.json:")
        and issue.endswith(":e164_phone_number")
        for issue in report["issues"]
    )


def test_package_audit_rejects_live_dashboard_claim_with_open_gates(tmp_path):
    artifact_root = _generate_package(tmp_path)
    dashboard = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-dashboard.html"
    dashboard.write_text(
        dashboard.read_text(encoding="utf-8")
        .replace("Static package ready", "Static ready")
        .replace("Live/Spark gaps", "Artifact failures")
        .replace("scripted_static_ack_until_live_voice_evidence", "live_voice_ready")
        .replace("needs_live_probe", "live_probe_complete"),
        encoding="utf-8",
    )

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "dashboard:missing_non_live_token:Static package ready" in report["issues"]
    assert "dashboard:missing_non_live_token:Live/Spark gaps" in report["issues"]
    assert "dashboard:missing_non_live_token:scripted_static_ack_until_live_voice_evidence" in report["issues"]
    assert "dashboard:missing_non_live_token:needs_live_probe" in report["issues"]


def test_package_audit_rejects_dashboard_metric_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    dashboard = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-dashboard.html"
    dashboard.write_text(
        dashboard.read_text(encoding="utf-8").replace(
            '<div class="metric"><small>Live/Spark gaps</small><strong>3</strong></div>',
            '<div class="metric"><small>Live/Spark gaps</small><strong>0</strong></div>',
        ),
        encoding="utf-8",
    )

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "dashboard:metric:Live/Spark gaps:mismatch" in report["issues"]


def test_package_audit_rejects_dashboard_table_row_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    dashboard = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-dashboard.html"
    dashboard.write_text(
        dashboard.read_text(encoding="utf-8").replace(
            '<td>stripe-projects</td><td><span class="pill ok">queued</span></td><td>$25.00</td>',
            '<td>stripe-projects</td><td><span class="pill ok">ready</span></td><td>$25.00</td>',
            1,
        ),
        encoding="utf-8",
    )

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "dashboard:Action Ledger:rows_mismatch" in report["issues"]


def test_package_audit_rejects_dashboard_pending_approval_evidence_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    dashboard = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-dashboard.html"
    dashboard.write_text(
        dashboard.read_text(encoding="utf-8").replace(
            "interpreter_promoted+oracle_promoted; audio=artifact://voiceops-demo/discord-budget-turn.wav; tool=tool_disclosure",
            "reflex_hypothesis; audio=missing_audio_ref; tool=missing_tool_disclosure",
            1,
        ),
        encoding="utf-8",
    )

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "dashboard:Pending Approvals:rows_mismatch" in report["issues"]


def test_package_audit_rejects_spark_readiness_claim_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    demo_path = artifact_root / "hackathon-voiceops-demo" / "current" / "voiceops-demo.json"
    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    active_model = demo["sponsor_stack"]["hermes_active_model"]
    active_model["active_model"] = "Nemotron 3 Super via hosted provider"
    active_model["spark_local"] = True
    active_model["fallback_used"] = True
    demo["recording_readiness"]["spark_local_readiness"] = True
    demo["recording_readiness"]["spark_benchmark_required"] = False
    demo["recording_readiness"]["spark_readiness_source"] = "manual_override"
    _write_json(demo_path, demo)

    readiness_path = artifact_root / "hackathon-voiceops-demo" / "current" / "readiness-report.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["spark_local_readiness"] = True
    readiness["spark_benchmark_required"] = False
    readiness["spark_readiness_source"] = "manual_override"
    readiness["live_demo_missing_evidence"] = [
        item for item in readiness["live_demo_missing_evidence"] if item != "local_spark_stack_matrix"
    ]
    _write_json(readiness_path, readiness)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "spark_model_claim:spark_local_true_for_hosted_model" in report["issues"]
    assert "spark_model_claim:spark_local_true_without_local_marker" in report["issues"]
    assert "spark_model_claim:fallback_used_but_spark_local_true" in report["issues"]
    assert "spark_model_claim:spark_local_readiness_mismatch" in report["issues"]
    assert "spark_model_claim:spark_benchmark_required_mismatch" in report["issues"]
    assert "spark_model_claim:readiness_source_mismatch" in report["issues"]
    assert "spark_model_claim:missing_m4_live_evidence_gap" in report["issues"]


def test_package_audit_rejects_gemma_in_reflex_role(tmp_path):
    artifact_root = _generate_package(
        tmp_path,
        reflex_model="Gemma 4 E2B audio-native reflex",
        interpreter_model="Gemma 4 E2B audio-native interpreter",
    )

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "spark_model_claim:gemma_model_in_reflex_role" in report["issues"]


def test_package_audit_rejects_provider_role_matrix_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    demo_path = artifact_root / "hackathon-voiceops-demo" / "current" / "voiceops-demo.json"
    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    roles = {item["role"]: item for item in demo["provider_role_matrix"]}
    roles["reflex"]["selected_label"] = "wrong reflex model"
    roles["auxiliary_transcript_evidence"]["authority"] = "interpreter_promoted"
    roles["auxiliary_transcript_evidence"]["must_not"] = ["block acknowledgement"]
    roles["oracle"]["must_not"] = ["act on hypothesis-only transcript text for high-risk actions"]
    demo["provider_role_matrix"] = [item for item in roles.values() if item["role"] != "degraded_fallback"]
    _write_json(demo_path, demo)

    dashboard = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-dashboard.html"
    dashboard.write_text(dashboard.read_text(encoding="utf-8").replace("Provider Roles", "Provider Matrix"), encoding="utf-8")

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "provider_role_matrix:roles_mismatch" in report["issues"]
    assert "provider_role_matrix:reflex:selected_label_mismatch" in report["issues"]
    assert "provider_role_matrix:auxiliary_transcript_evidence:authority_mismatch" in report["issues"]
    assert (
        "provider_role_matrix:auxiliary_transcript_evidence:missing_no_second_turn_boundary"
        in report["issues"]
    )
    assert "provider_role_matrix:auxiliary_transcript_evidence:missing_high_risk_boundary" in report["issues"]
    assert "provider_role_matrix:oracle:missing_oracle_model_boundary" in report["issues"]
    assert "dashboard:missing_provider_role_token:Provider Roles" in report["issues"]


def test_package_audit_rejects_hosted_fallback_public_copy_overclaim(tmp_path):
    artifact_root = _generate_package(tmp_path)
    demo_path = artifact_root / "hackathon-voiceops-demo" / "current" / "voiceops-demo.json"
    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    active_model = demo["sponsor_stack"]["hermes_active_model"]
    active_model["active_model"] = "Nemotron 3 Super via hosted provider"
    active_model["path"] = "hosted_nemotron_3_super_fallback"
    active_model["status"] = "hosted_fallback"
    active_model["label"] = "Hosted Nemotron 3 Super /model fallback"
    active_model["spark_local"] = False
    active_model["fallback_used"] = True
    active_model["evidence_status"] = "hosted_fallback_not_spark_local_evidence"
    demo["recording_readiness"]["spark_local_evidence_status"] = "hosted_or_nonlocal_path_not_spark_evidence"
    _write_json(demo_path, demo)

    readiness_path = artifact_root / "hackathon-voiceops-demo" / "current" / "readiness-report.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["spark_local_evidence_status"] = "hosted_or_nonlocal_path_not_spark_evidence"
    _write_json(readiness_path, readiness)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "dashboard:contradicts_active_model_path" in report["issues"]
    assert "dashboard:missing_non_live_token:Hosted fallback selected, Spark-local evidence pending" in report["issues"]
    assert "spark_public_copy:contradicts_active_model_path" in report["issues"]


def test_package_audit_rejects_nemoclaw_operator_contract_mismatch(tmp_path):
    artifact_root = _generate_package(tmp_path)
    packet_path = artifact_root / "hackathon-voiceops-demo" / "current" / "nemoclaw-action-packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["approval_required_actions"][0]["approval_contract"]["command_sha256"] = "0" * 64
    packet["approval_contracts"]["provision-voip-provider"]["command_sha256"] = "0" * 64
    _write_json(packet_path, packet)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "nemoclaw:provision-voip-provider:command_sha256_mismatch" in report["issues"]
    assert "operator_state:provision-voip-provider:approval_contract_mismatch" in report["issues"]
    assert "operator_state:provision-voip-provider:pending_contract_mismatch" in report["issues"]


def test_package_audit_rejects_external_service_execution_claim(tmp_path):
    artifact_root = _generate_package(tmp_path)
    state_path = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["planned_services"][0]["execution_status"] = "executed"
    state["planned_services"][0]["status"] = "provisioned"
    _write_json(state_path, state)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "planned_services:provision-voip-provider:external_execution_claim" in report["issues"]
    assert "planned_services:provision-voip-provider:external_status_invalid" in report["issues"]


def test_package_audit_rejects_operator_state_approval_ref_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    state_path = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["planned_services"][0]["approval_ref"] = None
    _write_json(state_path, state)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert (
        "operator_state:validation:external_service_missing_approval_ref:provision-voip-provider"
        in report["issues"]
    )


def test_package_audit_rejects_closure_gate_mismatch(tmp_path):
    artifact_root = _generate_package(tmp_path)
    closure_path = artifact_root / "voiceops-plan" / "current" / "readiness-closure-index.json"
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    closure["gates"] = closure["gates"][:-1]
    _write_json(closure_path, closure)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "closure:gates_mismatch_between_demo_and_plan" in report["issues"]


def test_package_audit_rejects_plan_run_closure_mismatch(tmp_path):
    artifact_root = _generate_package(tmp_path)
    plan_run_path = artifact_root / "voiceops-plan" / "current" / "voiceops-plan-run.json"
    plan_run = json.loads(plan_run_path.read_text(encoding="utf-8"))
    plan_run["closure_index"]["remaining_gates"] = []
    _write_json(plan_run_path, plan_run)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "plan_run:closure_index_mismatch" in report["issues"]


def test_package_audit_rejects_plan_run_top_level_mirror_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    plan_run_path = artifact_root / "voiceops-plan" / "current" / "voiceops-plan-run.json"
    plan_run = json.loads(plan_run_path.read_text(encoding="utf-8"))
    plan_run["artifact_id"] = "voiceops-plan-run-copy"
    plan_run["artifact_only"] = False
    plan_run["ok"] = False
    plan_run["hard_failures"] = ["milestone_0_hackathon_demo"]
    plan_run["readiness_gaps"] = []
    plan_run["closure_status"] = "complete"
    plan_run["readiness_ok"] = True
    plan_run["current_environment_blockers"] = {}
    plan_run["remaining_gates"] = []
    plan_run["next_actions"] = plan_run["next_actions"][:-1]
    _write_json(plan_run_path, plan_run)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "plan_run:artifact_id_mismatch" in report["issues"]
    assert "plan_run:artifact_only_not_true" in report["issues"]
    assert "plan_run:ok_not_true" in report["issues"]
    assert "plan_run:hard_failures_not_empty" in report["issues"]
    assert "plan_run:readiness_gaps_mismatch" in report["issues"]
    assert "plan_run:closure_status_mismatch" in report["issues"]
    assert "plan_run:readiness_ok_mismatch" in report["issues"]
    assert "plan_run:current_environment_blockers_mismatch" in report["issues"]
    assert "plan_run:remaining_gates_mismatch" in report["issues"]
    assert "plan_run:next_actions_mismatch" in report["issues"]


def test_package_audit_rejects_stale_embedded_package_audit_summary(tmp_path):
    artifact_root = _generate_package(tmp_path)
    plan_run_path = artifact_root / "voiceops-plan" / "current" / "voiceops-plan-run.json"
    plan_run = json.loads(plan_run_path.read_text(encoding="utf-8"))
    plan_run["package_audit"] = {
        "ok": True,
        "status": "pass",
        "issues": [],
        "checked_artifact_count": 94,
    }
    _write_json(plan_run_path, plan_run)
    assert audit_package(artifact_root)["ok"] is True

    plan_run["package_audit"]["status"] = "fail"
    plan_run["package_audit"]["checked_artifact_count"] = 91
    _write_json(plan_run_path, plan_run)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "plan_run:package_audit_summary_mismatch:status" in report["issues"]
    assert "plan_run:package_audit_summary_mismatch:checked_artifact_count" in report["issues"]


def test_package_audit_rejects_unresolvable_handoff_blockers_ref(tmp_path):
    artifact_root = _generate_package(tmp_path)
    handoff_path = artifact_root / "voiceops-plan" / "current" / "operator-handoff.json"
    closure_path = artifact_root / "voiceops-plan" / "current" / "readiness-closure-index.json"
    demo_handoff_path = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-handoff-preview.json"

    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    demo_handoff = json.loads(demo_handoff_path.read_text(encoding="utf-8"))
    handoff["diagnostic_blockers_ref"] = "missing_blockers_field"
    closure["operator_handoff"] = handoff
    demo_handoff["diagnostic_blockers_ref"] = "missing_blockers_field"
    _write_json(handoff_path, handoff)
    _write_json(closure_path, closure)
    _write_json(demo_handoff_path, demo_handoff)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "operator_handoff:diagnostic_blockers_ref_unresolvable" in report["issues"]


def test_package_audit_rejects_missing_channel_policy_review_action(tmp_path):
    artifact_root = _generate_package(tmp_path)
    plan_run_path = artifact_root / "voiceops-plan" / "current" / "voiceops-plan-run.json"
    closure_path = artifact_root / "voiceops-plan" / "current" / "readiness-closure-index.json"
    handoff_path = artifact_root / "voiceops-plan" / "current" / "operator-handoff.json"
    demo_handoff_path = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-handoff-preview.json"

    plan_run = json.loads(plan_run_path.read_text(encoding="utf-8"))
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    demo_handoff = json.loads(demo_handoff_path.read_text(encoding="utf-8"))

    closure["review_actions"] = []
    closure["operator_handoff"]["review_phases"] = []
    plan_run["closure_index"] = closure
    plan_run["review_actions"] = []
    handoff["review_phases"] = []
    demo_handoff["review_phases"] = []

    _write_json(plan_run_path, plan_run)
    _write_json(closure_path, closure)
    _write_json(handoff_path, handoff)
    _write_json(demo_handoff_path, demo_handoff)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "plan_run:missing_channel_policy_review_action" in report["issues"]
    assert "plan_closure:missing_channel_policy_review_action" in report["issues"]
    assert "operator_handoff:missing_channel_policy_review_phase" in report["issues"]
    assert "demo_handoff:missing_channel_policy_review_phase" in report["issues"]


def test_package_audit_rejects_forged_post_approval_receipt_validation(tmp_path):
    artifact_root = _generate_package(tmp_path)
    scaffold_path = (
        artifact_root
        / "voiceops-provisioning"
        / "current"
        / "post-approval-receipts-scaffold"
        / "post-approval-receipts.json"
    )
    validation_path = artifact_root / "voiceops-provisioning" / "current" / "post-approval-receipts.validation.json"
    scaffold = json.loads(scaffold_path.read_text(encoding="utf-8"))
    scaffold.pop("example_only", None)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation.update(
        {
            "loaded": False,
            "status": "valid",
            "receipt_count": 1,
            "ledger_rows": [{"action_id": "provision-voip-provider", "status": "executed"}],
            "validation_issues": [],
        }
    )
    _write_json(scaffold_path, scaffold)
    _write_json(validation_path, validation)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "post_approval_receipts_scaffold:must_remain_example_only" in report["issues"]
    assert "post_approval_receipts_validation:status_not_not_supplied_without_loaded_receipts" in report["issues"]
    assert "post_approval_receipts_validation:receipt_count_nonzero_without_loaded_receipts" in report["issues"]
    assert "post_approval_receipts_validation:ledger_rows_present_without_loaded_receipts" in report["issues"]


def test_package_audit_rejects_execution_plan_command_hash_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    plan_path = artifact_root / "voiceops-provisioning" / "current" / "milestone2-execution-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    action = plan["approval_required_actions"][0]
    forged_hash = "0" * 64
    action["command_sha256"] = forged_hash
    action["approval_contract"]["command_sha256"] = forged_hash
    plan["approval_contracts"][action["action_id"]]["command_sha256"] = forged_hash
    receipt_slot = plan["receipts"][action["expected_receipt_ref"].split(".", 1)[1]]
    receipt_slot["command_sha256"] = forged_hash
    _write_json(plan_path, plan)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "execution_plan:provision-voip-provider:command_sha256_mismatch" in report["issues"]
    assert (
        "execution_plan:provision-voip-provider:approval_contract_command_sha256_mismatch"
        in report["issues"]
    )
    assert (
        "execution_plan:provision-voip-provider:indexed_contract_command_sha256_mismatch"
        in report["issues"]
    )
    assert "execution_plan:provision-voip-provider:receipt_slot_command_sha256_mismatch" in report["issues"]


def test_package_audit_rejects_execution_plan_action_missing_claimed_nemoclaw_surface(tmp_path):
    artifact_root = _generate_package(tmp_path)
    plan_path = artifact_root / "voiceops-provisioning" / "current" / "milestone2-execution-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    action = next(item for item in plan["approval_required_actions"] if item["action_id"] == "publish-status")
    action["approval_artifact"] = "nemoclaw-action-packet.json"
    action["approval_contract"]["approval_artifact"] = "nemoclaw-action-packet.json"
    plan["approval_contracts"]["publish-status"]["approval_artifact"] = "nemoclaw-action-packet.json"
    _write_json(plan_path, plan)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "execution_plan:publish-status:missing_nemoclaw_packet_action" in report["issues"]


def test_package_audit_rejects_execution_plan_unknown_approval_surface(tmp_path):
    artifact_root = _generate_package(tmp_path)
    plan_path = artifact_root / "voiceops-provisioning" / "current" / "milestone2-execution-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    action = next(item for item in plan["approval_required_actions"] if item["action_id"] == "publish-status")
    action["approval_artifact"] = "unreviewed-egress-policy.json"
    action["approval_contract"]["approval_artifact"] = "unreviewed-egress-policy.json"
    plan["approval_contracts"]["publish-status"]["approval_artifact"] = "unreviewed-egress-policy.json"
    _write_json(plan_path, plan)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "execution_plan:publish-status:unknown_approval_artifact:unreviewed-egress-policy.json" in report["issues"]


def test_package_audit_rejects_spark_scaffold_source_hash_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    evidence_path = (
        artifact_root
        / "voiceops-spark-matrix"
        / "current"
        / "spark-benchmark-scaffold"
        / "spark-benchmark-evidence.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["evidence"][0]["source_artifact_sha256"] = "0" * 64
    evidence["evidence"][0]["collector_attestation"]["redacted_artifact_sha256"] = "0" * 64
    _write_json(evidence_path, evidence)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "spark_evidence_scaffold:reflex-moshi-s2s:source_artifact_sha256_mismatch" in report["issues"]
    assert (
        "spark_evidence_scaffold:reflex-moshi-s2s:collector_attestation_redacted_sha256_mismatch"
        in report["issues"]
    )


def test_package_audit_rejects_scaffold_approval_decision_ref_path_escape(tmp_path):
    artifact_root = _generate_package(tmp_path)
    scaffold_path = (
        artifact_root
        / "voiceops-provisioning"
        / "current"
        / "post-approval-receipts-scaffold"
        / "post-approval-receipts.json"
    )
    scaffold = json.loads(scaffold_path.read_text(encoding="utf-8"))
    scaffold["receipts"][0]["approval_decision_ref"] = "/tmp/approval-decision.json"
    scaffold["receipts"][1]["approval_decision_ref"] = "../approval-decision.json"
    _write_json(scaffold_path, scaffold)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert (
        "post_approval_receipts_scaffold:post_approval_receipts:receipt-example-provision-voip-provider:approval_decision_ref:absolute_path_not_allowed"
        in report["issues"]
    )
    assert (
        "post_approval_receipts_scaffold:post_approval_receipts:receipt-example-buy-service-credit:approval_decision_ref:parent_traversal_not_allowed"
        in report["issues"]
    )


def test_package_audit_resolves_loaded_post_approval_receipt_path_and_recomputes(tmp_path):
    artifact_root = _generate_package(tmp_path)
    receipt_path = artifact_root / "voiceops-provisioning" / "current" / "post-approval-receipts.json"
    validation_path = artifact_root / "voiceops-provisioning" / "current" / "post-approval-receipts.validation.json"
    _write_json(
        receipt_path,
        {
            "schema_version": "voiceops.post_approval_receipts.v1",
            "receipts": [],
            "credential_locations": [],
            "rollback_receipts": [],
            "audit_events": [],
        },
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation.update(
        {
            "loaded": True,
            "path": "artifacts/voiceops-provisioning/current/post-approval-receipts.json",
            "status": "valid",
            "validation_issues": [],
        }
    )
    _write_json(validation_path, validation)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "package_artifact:unexpected:voiceops-provisioning/current/post-approval-receipts.json" not in report[
        "issues"
    ]
    assert "post_approval_receipts_validation:status_mismatch" in report["issues"]
    assert "post_approval_receipts_validation:validation_issues_mismatch" in report["issues"]


def test_package_audit_allows_documented_post_approval_executor_sidecars(tmp_path):
    artifact_root = _generate_package(tmp_path)
    provisioning_dir = artifact_root / "voiceops-provisioning" / "current"
    decision_dir = provisioning_dir / "approval-decisions"
    decision_dir.mkdir()
    _write_json(
        provisioning_dir / "approval-decisions.json",
        {
            "schema_version": "voiceops.approval_decisions.v1",
            "decisions": [{"action_id": "provision-voip-provider", "decision": "approve_once"}],
        },
    )
    _write_json(
        decision_dir / "provision-voip-provider.json",
        {
            "schema_version": "voiceops.approval_decision.v1",
            "action_id": "provision-voip-provider",
            "decision": "approve_once",
            "redacted": True,
        },
    )
    _write_json(
        provisioning_dir / "stripe-executor-report.json",
        {
            "schema_version": "voiceops.stripe_executor_report.v1",
            "execute": True,
            "actions": [{"action_id": "provision-voip-provider", "status": "executed"}],
            "redacted": True,
        },
    )

    report = audit_package(artifact_root)

    unexpected = [issue for issue in report["issues"] if issue.startswith("package_artifact:unexpected:")]
    assert "package_artifact:unexpected:voiceops-provisioning/current/approval-decisions.json" not in unexpected
    assert (
        "package_artifact:unexpected:voiceops-provisioning/current/approval-decisions/"
        "provision-voip-provider.json"
        not in unexpected
    )
    assert "package_artifact:unexpected:voiceops-provisioning/current/stripe-executor-report.json" not in unexpected


def test_package_audit_rejects_empty_loaded_post_approval_receipts(tmp_path):
    artifact_root = _generate_package(tmp_path)
    receipt_path = artifact_root / "voiceops-provisioning" / "current" / "post-approval-receipts.json"
    validation_path = artifact_root / "voiceops-provisioning" / "current" / "post-approval-receipts.validation.json"
    _write_json(receipt_path, {})
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation.update(
        {
            "loaded": True,
            "path": "artifacts/voiceops-provisioning/current/post-approval-receipts.json",
            "status": "valid",
            "validation_issues": [],
        }
    )
    _write_json(validation_path, validation)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "post_approval_receipts_validation:loaded_receipts_empty_or_invalid" in report["issues"]


def test_package_audit_checks_post_approval_scaffold_even_when_receipts_loaded(tmp_path):
    artifact_root = _generate_package(tmp_path)
    scaffold_path = (
        artifact_root
        / "voiceops-provisioning"
        / "current"
        / "post-approval-receipts-scaffold"
        / "post-approval-receipts.json"
    )
    receipt_path = artifact_root / "voiceops-provisioning" / "current" / "post-approval-receipts.json"
    validation_path = artifact_root / "voiceops-provisioning" / "current" / "post-approval-receipts.validation.json"
    scaffold = json.loads(scaffold_path.read_text(encoding="utf-8"))
    scaffold.pop("example_only", None)
    _write_json(scaffold_path, scaffold)
    _write_json(
        receipt_path,
        {
            "schema_version": "voiceops.post_approval_receipts.v1",
            "receipts": [],
            "credential_locations": [],
            "rollback_receipts": [],
            "audit_events": [],
        },
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation.update(
        {
            "loaded": True,
            "path": "artifacts/voiceops-provisioning/current/post-approval-receipts.json",
            "status": "valid",
            "validation_issues": [],
        }
    )
    _write_json(validation_path, validation)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "post_approval_receipts_scaffold:must_remain_example_only" in report["issues"]


def test_package_audit_rejects_live_evidence_scaffold_claiming_live_readiness(tmp_path):
    artifact_root = _generate_package(tmp_path)
    scaffold_dir = artifact_root / "voiceops-voice-operator" / "current" / "live-voice-evidence-scaffold"
    manifest_path = scaffold_dir / "manifest.json"
    discord_path = scaffold_dir / "sections" / "discord-live-probe.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    discord = json.loads(discord_path.read_text(encoding="utf-8"))
    manifest.pop("example_only", None)
    manifest["overall_status"] = "live_evidence_supplied_not_readiness_claim"
    discord.pop("example_only", None)
    discord["collector_attestation"].pop("example_only", None)
    _write_json(manifest_path, manifest)
    _write_json(discord_path, discord)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "live_evidence_scaffold:manifest_must_remain_example_only" in report["issues"]
    assert "live_evidence_scaffold:manifest_unexpected_live_claim" in report["issues"]
    assert "live_evidence_scaffold:discord_live_probe:must_remain_example_only" in report["issues"]
    assert "live_evidence_scaffold:discord_live_probe:collector_attestation_must_remain_example_only" in report[
        "issues"
    ]


def test_package_audit_rejects_live_evidence_scaffold_manifest_path_tampering(tmp_path):
    artifact_root = _generate_package(tmp_path)
    manifest_path = (
        artifact_root / "voiceops-voice-operator" / "current" / "live-voice-evidence-scaffold" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reports"]["discord_live_probe"] = "/tmp/discord-live-probe.json"
    _write_json(manifest_path, manifest)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert (
        "live_evidence_scaffold:live_evidence_manifest:discord_live_probe:report_path:absolute_path_not_allowed"
        in report["issues"]
    )


def test_package_audit_rejects_preflight_scaffold_manifest_path_tampering(tmp_path):
    artifact_root = _generate_package(tmp_path)
    manifest_path = (
        artifact_root
        / "voiceops-provisioning"
        / "current"
        / "provisioning-preflight-scaffold"
        / "provisioning-preflight-evidence.manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reports"]["stripe_projects"] = "/tmp/stripe-projects-evidence.json"
    _write_json(manifest_path, manifest)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert (
        "preflight_evidence_scaffold:"
        "preflight_evidence_manifest:stripe_projects:report_path:absolute_path_not_allowed"
        in report["issues"]
    )


def test_package_audit_rejects_plan_run_model_arg_drift_from_demo(tmp_path):
    artifact_root = _generate_package(
        tmp_path,
        active_model="Nemotron 3 Super via hosted provider",
        reflex_model="Moshi fast reflex",
        interpreter_model="Gemma 4 E4B audio-native interpreter",
    )
    plan_run_path = artifact_root / "voiceops-plan" / "current" / "voiceops-plan-run.json"
    plan_run = json.loads(plan_run_path.read_text(encoding="utf-8"))
    plan_run["plan_args"]["active_model"] = "Nemotron 3 Ultra via hosted provider"
    plan_run["plan_args"]["reflex_model"] = "Moshi alternate reflex"
    plan_run["plan_args"]["interpreter_model"] = "Gemma 4 E2B audio-native interpreter"
    _write_json(plan_run_path, plan_run)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "plan_run:active_model_arg_mismatch_demo" in report["issues"]
    assert "plan_run:reflex_model_arg_mismatch_demo" in report["issues"]
    assert "plan_run:interpreter_model_arg_mismatch_demo" in report["issues"]


def test_package_audit_rejects_plan_run_commands_missing_model_args(tmp_path):
    artifact_root = _generate_package(
        tmp_path,
        active_model="Nemotron 3 Super via hosted provider",
        reflex_model="Moshi fast reflex",
        interpreter_model="Gemma 4 E4B audio-native interpreter",
    )
    handoff_path = artifact_root / "voiceops-plan" / "current" / "operator-handoff.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff["phases"][1]["commands"][0] = (
        "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
        "--output-dir artifacts/voiceops-plan/current --dry-audit --package-audit"
    )
    _write_json(handoff_path, handoff)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "operator_handoff:mismatch_with_closure" in report["issues"]
    assert "operator_handoff:plan_run_command_missing_active_model_arg" in report["issues"]
    assert "operator_handoff:plan_run_command_missing_reflex_model_arg" in report["issues"]
    assert "operator_handoff:plan_run_command_missing_interpreter_model_arg" in report["issues"]


def test_package_audit_accepts_equals_form_plan_run_model_args(tmp_path):
    artifact_root = _generate_package(
        tmp_path,
        active_model="Nemotron 3 Super via hosted provider",
        reflex_model="Moshi fast reflex",
        interpreter_model="Gemma 4 E4B audio-native interpreter",
    )

    def rewrite(command: str) -> str:
        return command.replace(
            "--active-model 'Nemotron 3 Super via hosted provider'",
            "--active-model='Nemotron 3 Super via hosted provider'",
        ).replace(
            "--reflex-model 'Moshi fast reflex'",
            "--reflex-model='Moshi fast reflex'",
        ).replace(
            "--interpreter-model 'Gemma 4 E4B audio-native interpreter'",
            "--interpreter-model='Gemma 4 E4B audio-native interpreter'",
        )

    for path in (
        artifact_root / "hackathon-voiceops-demo" / "current" / "readiness-closure-summary.json",
        artifact_root / "hackathon-voiceops-demo" / "current" / "operator-handoff-preview.json",
        artifact_root / "voiceops-plan" / "current" / "operator-handoff.json",
        artifact_root / "voiceops-plan" / "current" / "readiness-closure-index.json",
        artifact_root / "voiceops-plan" / "current" / "voiceops-plan-run.json",
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload = _rewrite_json_strings(payload, rewrite)
        _write_json(path, payload)

    report = audit_package(artifact_root)

    assert report["ok"] is True
    assert report["issues"] == []


def test_package_audit_rejects_non_object_plan_args(tmp_path):
    artifact_root = _generate_package(tmp_path)
    plan_run_path = artifact_root / "voiceops-plan" / "current" / "voiceops-plan-run.json"
    plan_run = json.loads(plan_run_path.read_text(encoding="utf-8"))
    plan_run["plan_args"] = ["not", "an", "object"]
    _write_json(plan_run_path, plan_run)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "plan_run:plan_args_not_object" in report["issues"]


def test_package_audit_rejects_plan_safety_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    plan_run_path = artifact_root / "voiceops-plan" / "current" / "voiceops-plan-run.json"
    closure_path = artifact_root / "voiceops-plan" / "current" / "readiness-closure-index.json"
    plan_run = json.loads(plan_run_path.read_text(encoding="utf-8"))
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    plan_run["safety"]["mutating_network_io"] = True
    plan_run["safety"]["network_io"] = True
    plan_run["safety"]["network_io_scope"] = "provider_mutation"
    plan_run["safety"]["live_spend"] = True
    plan_run["safety"]["provider_provisioning"] = True
    plan_run["safety"]["outbound_calls"] = True
    plan_run["safety"]["outbound_sends"] = True
    plan_run["safety"]["read_only_discovery_grants_approval"] = True
    closure["safety"]["spark_execution"] = True
    _write_json(plan_run_path, plan_run)
    _write_json(closure_path, closure)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "plan_run:closure_index_mismatch" in report["issues"]
    assert "plan_run:safety_mutating_network_io_not_false" in report["issues"]
    assert "plan_run:safety_network_io_scope_invalid" in report["issues"]
    assert "plan_run:safety_live_spend_not_false" in report["issues"]
    assert "plan_run:safety_provider_provisioning_not_false" in report["issues"]
    assert "plan_run:safety_outbound_calls_not_false" in report["issues"]
    assert "plan_run:safety_outbound_sends_not_false" in report["issues"]
    assert "plan_run:safety_read_only_discovery_grants_approval_not_false" in report["issues"]
    assert "plan_closure:safety_spark_execution_not_false" in report["issues"]


def test_package_audit_rejects_unaudited_operator_handoff_reindex(tmp_path):
    artifact_root = _generate_package(tmp_path)
    handoff_path = artifact_root / "voiceops-plan" / "current" / "operator-handoff.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff["final_reindex_command"] = handoff["final_reindex_command"].replace(" --package-audit", "")
    _write_json(handoff_path, handoff)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "operator_handoff:mismatch_with_closure" in report["issues"]
    assert "operator_handoff:plan_run_command_missing_package_audit" in report["issues"]


def test_package_audit_rejects_handoff_phase_order_and_blocker_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    demo_handoff_path = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-handoff-preview.json"
    demo_handoff = json.loads(demo_handoff_path.read_text(encoding="utf-8"))
    demo_handoff["phases"][0]["order"] = None
    demo_handoff["phases"][2].pop("blocked_by_current_environment")
    _write_json(demo_handoff_path, demo_handoff)

    plan_handoff_path = artifact_root / "voiceops-plan" / "current" / "operator-handoff.json"
    plan_handoff = json.loads(plan_handoff_path.read_text(encoding="utf-8"))
    plan_handoff["phases"][1].pop("blocked_by_current_environment")
    _write_json(plan_handoff_path, plan_handoff)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "operator_handoff:mismatch_with_closure" in report["issues"]
    assert "operator_handoff:spend_and_provisioning_preflight:missing_environment_blockers" in report["issues"]
    assert "demo_handoff:phase_order_mismatch" in report["issues"]
    assert "demo_handoff:live_discord_voice:order_mismatch" in report["issues"]
    assert "demo_handoff:local_spark_stack:missing_environment_blockers" in report["issues"]


def test_package_audit_rejects_handoff_safe_command_order_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    plan_run_path = artifact_root / "voiceops-plan" / "current" / "voiceops-plan-run.json"
    closure_path = artifact_root / "voiceops-plan" / "current" / "readiness-closure-index.json"
    handoff_path = artifact_root / "voiceops-plan" / "current" / "operator-handoff.json"
    demo_handoff_path = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-handoff-preview.json"

    plan_run = json.loads(plan_run_path.read_text(encoding="utf-8"))
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    demo_handoff = json.loads(demo_handoff_path.read_text(encoding="utf-8"))

    for payload in (handoff, demo_handoff, closure["operator_handoff"], plan_run["closure_index"]["operator_handoff"]):
        live_phase = payload["phases"][0]
        live_phase["commands"][0], live_phase["commands"][1] = live_phase["commands"][1], live_phase["commands"][0]
        live_phase["first_safe_command"] = live_phase["commands"][0]
        live_phase["first_evidence_command"] = live_phase["commands"][0]
        spark_phase = payload["phases"][2]
        spark_phase["commands"][0], spark_phase["commands"][1] = spark_phase["commands"][1], spark_phase["commands"][0]
        spark_phase["first_safe_command"] = spark_phase["commands"][0]
        spark_phase["first_evidence_command"] = spark_phase["commands"][0]

    for payload in (closure, plan_run["closure_index"]):
        live_action = payload["next_actions"][0]
        live_action["first_safe_command"] = live_action["first_evidence_command"]
        spark_action = payload["next_actions"][2]
        spark_action["first_safe_command"] = spark_action["first_evidence_command"]
    plan_run["next_actions"] = plan_run["closure_index"]["next_actions"]

    _write_json(plan_run_path, plan_run)
    _write_json(closure_path, closure)
    _write_json(handoff_path, handoff)
    _write_json(demo_handoff_path, demo_handoff)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "operator_handoff:live_discord_voice:first_safe_command_not_no_write_audit" in report["issues"]
    assert "operator_handoff:live_discord_voice:no_write_audit_not_before_live_collection" in report["issues"]
    assert "operator_handoff:local_spark_stack:first_safe_command_not_spark_lint" in report["issues"]
    assert "operator_handoff:local_spark_stack:spark_lint_not_before_dgx_eval" in report["issues"]
    assert "demo_handoff:live_discord_voice:first_safe_command_not_no_write_audit" in report["issues"]
    assert "demo_handoff:local_spark_stack:first_safe_command_not_spark_lint" in report["issues"]
    assert "plan_run:live_discord_voice_operator:first_safe_command_not_no_write_audit" in report["issues"]
    assert "plan_run:local_spark_stack_matrix:first_safe_command_not_spark_lint" in report["issues"]


def test_package_audit_rejects_handoff_validation_command_without_safety_label(tmp_path):
    artifact_root = _generate_package(tmp_path)
    handoff_path = artifact_root / "voiceops-plan" / "current" / "operator-handoff.json"
    demo_handoff_path = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-handoff-preview.json"
    closure_path = artifact_root / "voiceops-plan" / "current" / "readiness-closure-index.json"
    plan_run_path = artifact_root / "voiceops-plan" / "current" / "voiceops-plan-run.json"

    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    demo_handoff = json.loads(demo_handoff_path.read_text(encoding="utf-8"))
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    plan_run = json.loads(plan_run_path.read_text(encoding="utf-8"))

    for payload in (handoff, demo_handoff, closure["operator_handoff"], plan_run["closure_index"]["operator_handoff"]):
        payload["phases"][2]["command_safety"].pop("matrix_only", None)

    _write_json(handoff_path, handoff)
    _write_json(demo_handoff_path, demo_handoff)
    _write_json(closure_path, closure)
    _write_json(plan_run_path, plan_run)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert (
        "operator_handoff:local_spark_stack_matrix:validation_command_missing_safety:matrix_only"
        in report["issues"]
    )
    assert (
        "demo_handoff:local_spark_stack_matrix:validation_command_missing_safety:matrix_only"
        in report["issues"]
    )


def test_package_audit_rejects_channel_policy_live_egress_claim(tmp_path):
    artifact_root = _generate_package(tmp_path)
    policy_path = artifact_root / "voiceops-channel-policy" / "current" / "channel-policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["scope"]["real_egress_enabled"] = True
    _write_json(policy_path, policy)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "channel_policy:validation:real_egress_enabled_without_review" in report["issues"]
    assert "channel_policy:real_egress_enabled_not_false" in report["issues"]


def test_package_audit_rejects_channel_policy_review_approval_claim(tmp_path):
    artifact_root = _generate_package(tmp_path)
    review_path = artifact_root / "voiceops-channel-policy" / "current" / "channel-policy-review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["review_status"] = "approved"
    review["real_egress_enabled"] = True
    review["review_commands"] = [
        command.replace(" --package-audit", "")
        if isinstance(command, str) and "scripts/voiceops_plan_run.py" in command
        else command
        for command in review["review_commands"]
    ]
    _write_json(review_path, review)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "channel_policy_review:review_status_not_pending" in report["issues"]
    assert "channel_policy_review:real_egress_enabled_not_false" in report["issues"]
    assert "channel_policy_review:plan_run_command_missing_package_audit" in report["issues"]
    assert "channel_policy_review:missing_package_audit_review_command" in report["issues"]


def test_package_audit_rejects_channel_policy_review_identity_and_artifact_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    review_path = artifact_root / "voiceops-channel-policy" / "current" / "channel-policy-review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["artifact_only"] = False
    review["policy_version"] = "stale-version"
    review["decision_options"] = ["approve_live_egress_after_external_credentials_are_bound"]
    _write_json(review_path, review)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "channel_policy_review:artifact_only_not_true" in report["issues"]
    assert "channel_policy_review:policy_version_mismatch" in report["issues"]
    assert "channel_policy_review:decision_options_missing_safe_choices" in report["issues"]


def test_package_audit_rejects_channel_policy_review_route_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    review_path = artifact_root / "voiceops-channel-policy" / "current" / "channel-policy-review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    phone_review = next(channel for channel in review["per_channel_review"] if channel["channel_id"] == "phone_sms")
    phone_review["approval_routes_to_confirm"].pop("approved_phone_handoff_call")
    phone_review["required_evidence"] = []
    phone_review["blocked_capabilities_to_confirm"] = []
    _write_json(review_path, review)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "channel_policy_review:phone_sms:approval_routes_mismatch" in report["issues"]
    assert "channel_policy_review:phone_sms:required_evidence_mismatch" in report["issues"]
    assert "channel_policy_review:phone_sms:blocked_capabilities_mismatch" in report["issues"]


def test_package_audit_rejects_closure_markdown_safety_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    markdown_path = artifact_root / "voiceops-plan" / "current" / "readiness-closure-index.md"
    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8")
        .replace("needs_external_evidence", "complete")
        .replace("Final package audit command", "Final review command")
        .replace("--package-audit", "--no-package-audit"),
        encoding="utf-8",
    )

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "closure_markdown:missing_needs_external_evidence" in report["issues"]
    assert "closure_markdown:missing_final_package_audit_command" in report["issues"]
    assert "closure_markdown:missing_package_audit_flag" in report["issues"]


def test_package_audit_rejects_operator_handoff_markdown_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    markdown_path = artifact_root / "voiceops-plan" / "current" / "operator-handoff.md"
    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8")
        .replace("Final package audit command", "Final review command")
        .replace("package_audit.status is pass", "package audit optional")
        .replace("never paste secret values into artifacts", "paste secrets into artifacts"),
        encoding="utf-8",
    )

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "operator_handoff_markdown:missing_final_package_audit_command" in report["issues"]
    assert "operator_handoff_markdown:missing_package_audit_status_signal" in report["issues"]
    assert "operator_handoff_markdown:missing_secret_policy" in report["issues"]


def test_package_audit_rejects_demo_handoff_markdown_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    markdown_path = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-handoff-preview.md"
    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8")
        .replace("Package audit:", "Package review:")
        .replace("--package-audit", "--no-package-audit")
        .replace("never paste secret values into artifacts", "paste secrets into artifacts"),
        encoding="utf-8",
    )

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "demo_handoff_markdown:missing_package_audit_section" in report["issues"]
    assert "demo_handoff_markdown:missing_package_audit_flag" in report["issues"]
    assert "demo_handoff_markdown:missing_no_secret_policy" in report["issues"]


def test_package_audit_rejects_public_recording_copy_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    demo_markdown_path = artifact_root / "hackathon-voiceops-demo" / "current" / "voiceops-demo.md"
    demo_script_path = artifact_root / "hackathon-voiceops-demo" / "current" / "demo-script.md"
    dashboard_path = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-dashboard.html"
    runbook_path = artifact_root / "hackathon-voiceops-demo" / "current" / "recording-runbook.md"
    writeup_path = artifact_root / "hackathon-voiceops-demo" / "current" / "submission-writeup.md"
    demo_markdown_path.write_text(
        demo_markdown_path.read_text(encoding="utf-8").replace(
            "static dry-run package", "live demo package"
        ),
        encoding="utf-8",
    )
    demo_script_path.write_text(
        demo_script_path.read_text(encoding="utf-8") + "\nThis turns a DGX Spark into an operator.\n",
        encoding="utf-8",
    )
    dashboard_path.write_text(
        dashboard_path.read_text(encoding="utf-8") + "\nSpark-powered Hermes operator\n",
        encoding="utf-8",
    )
    runbook_path.write_text(
        runbook_path.read_text(encoding="utf-8").replace(
            "Spark target selected, live evidence pending", "Spark evidence complete"
        ),
        encoding="utf-8",
    )
    writeup_path.write_text(
        writeup_path.read_text(encoding="utf-8").replace("Spend gated by approval", "Spend already executed"),
        encoding="utf-8",
    )

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "demo_markdown:missing_static_dry_run_status" in report["issues"]
    assert "recording_runbook_markdown:missing_spark_evidence_boundary" in report["issues"]
    assert "submission_writeup_markdown:missing_spend_gate" in report["issues"]
    assert "spark_public_copy:claims_spark_powered_operator_without_evidence" in report["issues"]
    assert "spark_public_copy:claims_turns_spark_into_operator_without_evidence" in report["issues"]


def test_package_audit_rejects_channel_policy_review_markdown_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    markdown_path = artifact_root / "voiceops-channel-policy" / "current" / "channel-policy-review.md"
    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8")
        .replace("Review status: pending_human_review", "Review status: approved")
        .replace("Real egress enabled: False", "Real egress enabled: True")
        .replace("--package-audit", "--no-package-audit")
        .replace("approved_phone_handoff_call", "unapproved_phone_handoff"),
        encoding="utf-8",
    )

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "channel_policy_review_markdown:missing_pending_review" in report["issues"]
    assert "channel_policy_review_markdown:missing_no_real_egress" in report["issues"]
    assert "channel_policy_review_markdown:missing_package_audit_flag" in report["issues"]
    assert "channel_policy_review_markdown:missing_phone_handoff_route" in report["issues"]


def test_package_audit_rejects_channel_policy_markdown_drift(tmp_path):
    artifact_root = _generate_package(tmp_path)
    markdown_path = artifact_root / "voiceops-channel-policy" / "current" / "channel-policy.md"
    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8")
        .replace("artifact-only; no network, secret reads, sends, SMS, or calls", "live egress allowed")
        .replace("Validation: pass", "Validation: skipped")
        .replace("approved_phone_handoff_call", "unapproved_phone_handoff")
        .replace("phone_number: `<redacted-phone>`", "phone_number: `<raw-phone>`"),
        encoding="utf-8",
    )

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "channel_policy_markdown:missing_artifact_only_safety" in report["issues"]
    assert "channel_policy_markdown:missing_validation_pass" in report["issues"]
    assert "channel_policy_markdown:missing_phone_handoff_route" in report["issues"]
    assert "channel_policy_markdown:missing_phone_redaction" in report["issues"]


def test_package_audit_rejects_contradictory_channel_review_markdown(tmp_path):
    artifact_root = _generate_package(tmp_path)
    markdown_path = artifact_root / "voiceops-channel-policy" / "current" / "channel-policy-review.md"
    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8")
        + "\n- Review status: approved\n- Real egress enabled: True\n- Live egress enabled: True\n",
        encoding="utf-8",
    )

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "channel_policy_review_markdown:contradicts_pending_review" in report["issues"]
    assert "channel_policy_review_markdown:contradicts_no_real_egress" in report["issues"]
    assert "channel_policy_review_markdown:contradicts_no_live_egress" in report["issues"]


def test_package_audit_parse_args_defaults():
    args = parse_args([])

    assert args.artifact_root == Path("artifacts")
    assert args.output_dir == Path("artifacts/voiceops-package-audit/current")
    assert args.audit_only is False
