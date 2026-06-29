from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

from scripts.hackathon_voiceops_demo import (
    _demo_milestone2_report,
    build_demo,
    build_readiness_report,
    parse_args,
    write_demo,
)
from scripts.voiceops_operator_state import validate_operator_state


def _dot_get(payload, ref):
    cursor = payload
    for part in ref.split("."):
        cursor = cursor[part]
    return cursor


def build_milestone2_like_failures(demo, readiness):
    return _demo_milestone2_report(demo, readiness)["required_failures"]


def _discord_live_env() -> dict[str, str]:
    return {
        "DISCORD_BOT_TOKEN": "set",
        "DISCORD_GUILD_ID": "guild-ref",
        "DISCORD_HOME_CHANNEL": "general",
        "DISCORD_VOICE_CHANNEL_ID": "123",
        "DISCORD_VOICE_CHANNEL_NAME": "General",
    }


def test_voiceops_demo_writes_headless_artifacts(tmp_path):
    args = parse_args(["--output-dir", str(tmp_path), "--budget-cents", "7000"])
    demo = build_demo(args)
    paths = write_demo(tmp_path, demo)

    assert set(paths) == {
        "json",
        "markdown",
        "audit_ledger",
        "demo_script",
        "dashboard",
        "milestone2_execution_plan",
        "nemoclaw_packet",
        "nemoclaw_packet_validation",
        "operator_handoff_preview_json",
        "operator_handoff_preview_markdown",
        "operator_state",
        "operator_state_events",
        "phone_context",
        "recording_runbook",
        "readiness_closure_summary_json",
        "readiness_closure_summary_markdown",
        "readiness_json",
        "readiness_markdown",
        "stripe_actions",
        "submission_writeup",
    }
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    readiness = json.loads(Path(paths["readiness_json"]).read_text(encoding="utf-8"))
    closure_summary = json.loads(Path(paths["readiness_closure_summary_json"]).read_text(encoding="utf-8"))
    nemoclaw = json.loads(Path(paths["nemoclaw_packet"]).read_text(encoding="utf-8"))
    nemoclaw_validation = json.loads(Path(paths["nemoclaw_packet_validation"]).read_text(encoding="utf-8"))
    phone_context = json.loads(Path(paths["phone_context"]).read_text(encoding="utf-8"))
    milestone2_plan = json.loads(Path(paths["milestone2_execution_plan"]).read_text(encoding="utf-8"))
    operator_handoff = json.loads(Path(paths["operator_handoff_preview_json"]).read_text(encoding="utf-8"))
    audit_events = [
        json.loads(line)
        for line in Path(paths["audit_ledger"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    operator_state = json.loads(Path(paths["operator_state"]).read_text(encoding="utf-8"))
    operator_events = [
        json.loads(line)
        for line in Path(paths["operator_state_events"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    action_ids = {action["action_id"] for action in payload["ops_actions"]}
    assert payload["schema_version"] == "voiceops.demo_package.v1"
    assert payload["artifact_id"] == "voiceops-demo"
    assert payload["artifact_manifest"]["readiness_json"] == "readiness-report.json"
    assert payload["artifact_manifest"]["readiness_closure_summary_json"] == "readiness-closure-summary.json"
    assert payload["artifact_manifest"]["readiness_closure_summary_markdown"] == "readiness-closure-summary.md"
    assert payload["artifact_manifest"]["operator_handoff_preview_json"] == "operator-handoff-preview.json"
    assert payload["artifact_manifest"]["operator_handoff_preview_markdown"] == "operator-handoff-preview.md"
    assert payload["artifact_manifest"]["operator_state"] == "operator-state.json"
    assert payload["artifact_manifest"]["milestone2_execution_plan"] == "milestone2-execution-plan.json"
    assert readiness["schema_version"] == "voiceops.recording_readiness_report.v1"
    assert readiness["artifact_id"] == "voiceops-recording-readiness-report"
    assert readiness["source_demo_artifact"] == "voiceops-demo.json"
    assert readiness["readiness_closure_summary_ref"] == "readiness-closure-summary.json"
    assert readiness["ready_for_demo"] is False
    assert readiness["ready_for_demo"] == readiness["live_demo_ready"]
    assert readiness["ready_for_static_recording"] is True
    assert readiness["ready_for_static_recording"] == readiness["static_recording_ready"]
    assert readiness["blockers"] == {
        "live_prerequisite_failures": readiness["live_prerequisite_failures"],
        "live_demo_missing_evidence": readiness["live_demo_missing_evidence"],
        "all_required_check_failures": readiness["all_required_check_failures"],
        "artifact_required_failures": readiness["artifact_required_failures"],
    }
    assert payload["recording_readiness"]["artifact_ref"] == "readiness-report.json"
    assert payload["recording_readiness"]["ready_for_recording"] is True
    assert payload["recording_readiness"]["static_recording_ready"] is True
    assert payload["recording_readiness"]["ready_for_recording_scope"] == "static_artifact_recording_only"
    assert payload["recording_readiness"]["live_demo_ready"] is False
    assert payload["recording_readiness"]["live_prerequisite_failures"] == [
        "discord_voice",
        "nemoclaw_boundary",
        "stripe_projects_cli",
        "stripe_link_cli",
        "phone_handoff",
    ]
    assert payload["recording_readiness"]["live_demo_missing_evidence"] == [
        "live_discord_voice_operator",
        "spend_and_provisioning_preflight",
        "local_spark_stack_matrix",
    ]
    assert payload["recording_readiness"]["spark_local_evidence_status"] == "target_selected_needs_benchmark_evidence"
    assert payload["recording_readiness"]["spark_local_readiness"] is False
    assert payload["recording_readiness"]["spark_benchmark_required"] is True
    assert payload["recording_readiness"]["spark_readiness_source"] == (
        "voiceops_spark_matrix.ready_for_one_spark_demo"
    )
    assert readiness["spark_local_readiness"] is False
    assert readiness["spark_benchmark_required"] is True
    assert readiness["spark_readiness_source"] == "voiceops_spark_matrix.ready_for_one_spark_demo"
    assert payload["recording_readiness"]["required_failures"] == []
    assert payload["recording_readiness"]["artifact_required_failures"] == []
    assert payload["recording_readiness"]["all_required_check_failures"] == [
        "discord_voice",
        "stripe_projects_cli",
        "stripe_link_cli",
    ]
    assert payload["readiness_closure_ref"] == "readiness-closure-summary.json"
    assert payload["readiness_closure_summary_ref"] == "readiness-closure-summary.json"
    assert payload["plan_readiness_closure_ref"] == "artifacts/voiceops-plan/current/readiness-closure-index.json"
    assert payload["readiness_closure"]["closure_status"] == "needs_external_evidence"
    assert closure_summary["schema_version"] == "voiceops.demo_closure_summary.v1"
    assert closure_summary["artifact_id"] == "voiceops-demo-readiness-closure-summary"
    assert closure_summary["source_demo_artifact"] == "voiceops-demo.json"
    assert closure_summary["source_readiness_artifact"] == "readiness-report.json"
    assert closure_summary == payload["readiness_closure"]
    assert {gate["gate_id"] for gate in payload["readiness_closure"]["gates"]} == {
        "live_discord_voice_operator",
        "spend_and_provisioning_preflight",
        "local_spark_stack_matrix",
    }
    closure_gates = {gate["gate_id"]: gate for gate in payload["readiness_closure"]["gates"]}
    assert closure_gates["live_discord_voice_operator"]["evidence_contract"]["manifest_schema_version"] == (
        "voiceops.realtime_voice_live_evidence_manifest.v1"
    )
    assert closure_gates["live_discord_voice_operator"]["evidence_contract"]["strict_validation_schema_version"] == (
        "voiceops.realtime_voice_live_evidence_validation.v1"
    )
    assert closure_gates["live_discord_voice_operator"]["evidence_contract"]["required_sidecar_mode"] == "production"
    assert (
        closure_gates["live_discord_voice_operator"]["evidence_contract"][
            "doctor_report_derivation_overclaims_production"
        ]
        is False
    )
    assert "derive_from_realtime_voice_report" in closure_gates["live_discord_voice_operator"]["collection_commands"]
    assert "run_realtime_voice_doctor_report" in closure_gates["live_discord_voice_operator"]["collection_commands"]
    assert "realtime-voice-doctor-report.json" in closure_gates["live_discord_voice_operator"]["collection_commands"][
        "run_realtime_voice_doctor_report"
    ]
    assert "realtime-voice-doctor-report.json" in closure_gates["live_discord_voice_operator"]["collection_commands"][
        "derive_from_realtime_voice_report"
    ]
    assert "path/to/realtime-voice-report.json" not in json.dumps(
        closure_gates["live_discord_voice_operator"]["collection_commands"]
    )
    assert "--validate-live-evidence" in closure_gates["live_discord_voice_operator"]["collection_commands"][
        "validate_live_manifest_offline"
    ]
    assert "--audit-only" in closure_gates["live_discord_voice_operator"]["collection_commands"][
        "audit_live_manifest_no_write"
    ]
    assert any(
        artifact.endswith("live-evidence-validation.json")
        for artifact in closure_gates["live_discord_voice_operator"]["expected_artifacts"]
    )
    assert closure_gates["spend_and_provisioning_preflight"]["evidence_contract"]["required_section_field"] == "source_artifact"
    assert "all_local_stack_smoke:needs_evidence" in closure_gates["local_spark_stack_matrix"]["missing"]
    assert (
        closure_gates["local_spark_stack_matrix"]["evidence_contract"]["hosted_fallback_counts_for_one_spark_readiness"]
        is False
    )
    assert (
        closure_gates["local_spark_stack_matrix"]["evidence_contract"][
            "loopback_smoke_bridge_counts_for_local_speech_readiness"
        ]
        is False
    )
    assert closure_gates["local_spark_stack_matrix"]["evidence_contract"]["local_speech_requires_production_provider"] is True
    assert any(
        artifact.endswith("sources/asr-nemotron-speech-raw.json")
        for artifact in closure_gates["local_spark_stack_matrix"]["expected_artifacts"]
    )
    assert "--run-readonly-discovery" in closure_gates["spend_and_provisioning_preflight"]["collection_commands"][
        "read_only_discovery"
    ]
    assert (
        closure_gates["spend_and_provisioning_preflight"]["evidence_contract"]["read_only_discovery_grants_approval"]
        is False
    )
    assert "artifacts/voiceops-provisioning/current/read-only-discovery.json" in closure_gates[
        "spend_and_provisioning_preflight"
    ]["expected_artifacts"]
    assert "scripts/dgx_spark_gemma4_voice_eval.sh" in closure_gates["local_spark_stack_matrix"]["collection_commands"][
        "dgx_eval"
    ]
    assert payload["operator_state_ref"] == "operator-state.json"
    assert payload["operator_state_events_ref"] == "operator-state-events.jsonl"
    assert payload["operator_handoff_preview_ref"] == "operator-handoff-preview.json"
    assert payload["milestone2_execution_plan_ref"] == "milestone2-execution-plan.json"
    assert operator_handoff["schema_version"] == "voiceops.operator_handoff_preview.v1"
    assert operator_handoff["changes_readiness_by_itself"] is False
    assert operator_handoff["readiness_closure_ref"] == "readiness-closure-summary.json"
    assert [phase["phase_id"] for phase in operator_handoff["phases"]] == [
        "live_discord_voice",
        "spend_and_provisioning_preflight",
        "local_spark_stack",
    ]
    assert [phase["order"] for phase in operator_handoff["phases"]] == [1, 2, 3]
    assert operator_handoff["phases"][0]["blocked_by_current_environment"] == {
        "missing_env_or_config": [
            "DISCORD_BOT_TOKEN",
            "DISCORD_GUILD_ID",
            "DISCORD_HOME_CHANNEL",
            "DISCORD_VOICE_CHANNEL_ID",
            "DISCORD_VOICE_CHANNEL_NAME",
        ],
        "needs_external_live_probe": True,
    }
    assert "--audit-only" in operator_handoff["phases"][0]["first_safe_command"]
    assert operator_handoff["phases"][0]["first_evidence_command"].startswith(
        "uv run python -m hermes_cli.realtime_voice_live_evidence"
    )
    assert "--run-doctor-report" in operator_handoff["phases"][0]["first_evidence_command"]
    assert operator_handoff["phases"][0]["commands"][2].startswith("uv run --extra dev --extra voice hermes doctor")
    assert "realtime-voice-doctor-report.json" in operator_handoff["phases"][0]["commands"][2]
    assert "path/to/realtime-voice-report.json" not in json.dumps(operator_handoff["phases"][0]["commands"])
    assert "run_realtime_voice_doctor_report" in operator_handoff["phases"][0]["command_safety"]
    assert "derive_from_realtime_voice_report" in operator_handoff["phases"][0]["command_safety"]
    assert operator_handoff["phases"][0]["command_safety"]["audit_live_manifest_no_write"] == (
        "no_write_existing_artifact_audit"
    )
    assert "production realtime voice sidecar session evidence" in operator_handoff["phases"][0]["required_inputs"]
    assert operator_handoff["phases"][1]["command_safety"]["read_only_discovery"] == (
        "network_possible_allowlisted_read_only"
    )
    assert operator_handoff["phases"][1]["blocked_by_current_environment"] == {
        "missing_cli_or_config": ["stripe_projects_cli", "stripe_link_cli", "nemoclaw_boundary", "phone_handoff"],
        "needs_read_only_discovery": True,
        "needs_redacted_setup_evidence": True,
    }
    assert "local_spark_stack_matrix" in operator_handoff["phases"][2]["blocked_by_current_package"]
    assert operator_handoff["phases"][2]["blocked_by_current_environment"] == {
        "required_hardware": "1x NVIDIA DGX Spark",
        "current_host_hint": "not_verified_by_demo_package",
        "needs_measured_spark_evidence": True,
    }
    assert "--lint-evidence" in operator_handoff["phases"][2]["commands"][0]
    assert operator_handoff["phases"][2]["commands"][1] == "scripts/dgx_spark_gemma4_voice_eval.sh"
    assert "--refresh-source-hashes" in operator_handoff["phases"][2]["commands"][2]
    assert operator_handoff["phases"][2]["command_safety"]["refresh_source_hashes"] == "local_file_hashing_only"
    assert operator_handoff["phases"][2]["command_safety"]["lint_evidence"] == "no_write_spark_evidence_lint"
    assert any("oracle_model" in item for item in operator_handoff["phases"][2]["must_not"])
    assert "--voice-live-evidence" in operator_handoff["final_reindex_command"]
    handoff_markdown = Path(paths["operator_handoff_preview_markdown"]).read_text(encoding="utf-8")
    assert "VoiceOps Operator Handoff Preview" in handoff_markdown
    assert "### 1. live_discord_voice" in handoff_markdown
    assert "### 2. spend_and_provisioning_preflight" in handoff_markdown
    assert "### 3. local_spark_stack" in handoff_markdown
    assert "Blocked by current environment" in handoff_markdown
    assert "Final Reindex" in handoff_markdown
    assert payload["safety_boundary_refs"] == {
        "nemoclaw_action_packet": "nemoclaw-action-packet.json",
        "nemoclaw_action_packet_validation": "nemoclaw-action-packet.validation.json",
        "phone_context": "phone-context.json",
        "stripe_actions_dry_run": "stripe-actions-dry-run.sh",
    }
    assert payload["spark_stack"]["local_first"] is True
    assert payload["spark_stack"]["compute"] == "1x NVIDIA DGX Spark target; measured evidence pending"
    assert payload["spark_stack"]["local_first_status"] == "strategy_target_not_readiness_claim"
    assert payload["spark_stack"]["current_path_local"] is True
    assert payload["kame_reflex_ack"]["status"] == "scripted_static_ack_until_live_voice_evidence"
    assert payload["kame_reflex_ack"]["ack_text"].startswith("I heard you.")
    assert payload["kame_reflex_ack"]["latency_ms"] is None
    assert payload["kame_reflex_ack"]["live_evidence_required_for_latency_claim"] is True
    assert payload["sponsor_stack"]["hermes_active_model"]["path"] == "spark_local_nemotron_3_super"
    assert payload["sponsor_stack"]["hermes_active_model"]["selected_by"] == "Hermes /model"
    assert payload["sponsor_stack"]["hermes_active_model"]["spark_local"] is True
    assert payload["sponsor_stack"]["hermes_active_model"]["status"] == "preferred_local_target_selected_not_validated"
    assert payload["sponsor_stack"]["hermes_active_model"]["fallback_used"] is False
    assert payload["sponsor_stack"]["hermes_active_model"]["evidence_status"] == "target_selected_needs_benchmark_evidence"
    assert payload["sponsor_stack"]["nemotron_3_super"]["selection"].startswith("Nemotron 3 Super")
    assert payload["sponsor_stack"]["nemotron_3_ultra_hosted_fallback"]["selection"].startswith(
        "Hosted /model fallback"
    )
    assert payload["spark_stack"]["oracle"]["preferred_local_target"] == "Nemotron 3 Super on DGX Spark"
    assert payload["spark_stack"]["oracle"]["hosted_fallback"].startswith("clearly labeled hosted provider")
    assert payload["spark_stack"]["oracle"]["active_model_path"]["path"] == "spark_local_nemotron_3_super"
    assert payload["sponsor_stack"]["stripe_skills"]["skills"] == ["stripe-projects", "stripe-link-cli", "mpp-agent"]
    assert payload["voice_surfaces"][0]["channel"] == "discord"
    assert payload["voice_surfaces"][0]["status"] == "intended-live-front-door-needs-evidence"
    assert {surface["channel"] for surface in payload["voice_surfaces"]} == {"discord", "whatsapp", "phone"}
    assert next(surface for surface in payload["voice_surfaces"] if surface["channel"] == "whatsapp")[
        "status"
    ] == "follow-on-not-configured-in-static-package"
    assert next(surface for surface in payload["voice_surfaces"] if surface["channel"] == "phone")["status"] == "dry-run-queued"
    assert "provision-voip-provider" in action_ids
    assert "call-user-phone" in action_ids
    assert payload["totals"]["held_budget_cents"] > 0
    assert nemoclaw["schema_version"] == "voiceops.nemoclaw_action_packet.v1"
    assert nemoclaw["artifact_id"] == "voiceops-nemoclaw-action-packet"
    assert nemoclaw["runtime"] == "NemoClaw"
    assert nemoclaw["mode"] == "dry_run_until_user_approval"
    assert nemoclaw["dry_run_shell_artifact"] == "stripe-actions-dry-run.sh"
    assert nemoclaw["audit_ledger_artifact"] == "audit-ledger.jsonl"
    assert nemoclaw["safety"] == {
        "live_spend": False,
        "provider_provisioning": False,
        "credential_retrieval": False,
        "outbound_phone_calls": False,
        "network_io": False,
        "requires_operator_approval": True,
        "default_decision": "hold",
    }
    assert nemoclaw_validation["schema_version"] == "voiceops.nemoclaw_action_packet_validation.v1"
    assert nemoclaw_validation["artifact_id"] == "voiceops-nemoclaw-action-packet-validation"
    assert nemoclaw_validation["ok"] is True
    assert nemoclaw_validation["status"] == "valid"
    assert nemoclaw_validation["mode"] == "local_static_validation_only"
    assert nemoclaw_validation["safety"] == {
        "executes_commands": False,
        "network_io": False,
        "live_spend": False,
        "provider_provisioning": False,
        "credential_retrieval": False,
        "outbound_phone_calls": False,
        "secret_values_emitted": False,
    }
    assert nemoclaw_validation["issues"] == []
    assert nemoclaw_validation["validated_contract_count"] == len(nemoclaw["approval_required_actions"])
    assert nemoclaw_validation["dry_run_command_count"] == len(nemoclaw["dry_run_commands"])
    assert nemoclaw["model_selected_by"] == "Hermes /model"
    assert nemoclaw["hermes_active_model"].startswith("Nemotron 3 Super")
    assert "oracle_model" not in nemoclaw
    assert "unapproved_purchase" in nemoclaw["blocked_capabilities"]
    assert "discord_or_whatsapp_send_without_channel_policy_approval" in nemoclaw["blocked_capabilities"]
    assert "status_summary_draft" in nemoclaw["allowed_capabilities"]
    status_action = next(action for action in payload["ops_actions"] if action["action_id"] == "draft-status")
    assert status_action["requires_approval"] is False
    assert status_action["status"] == "ready"
    assert "draft" in status_action["command"]
    assert "post summary" not in status_action["command"]
    assert "stripe projects add twilio/voice" in nemoclaw["dry_run_commands"]
    assert len(audit_events) == len(payload["ops_actions"])
    required_audit_fields = {
        "requested_by",
        "proposed_by",
        "budget_policy_ref",
        "command",
        "approval_required",
        "approval_status",
        "result",
        "receipt_ref",
        "credential_location_ref",
        "rollback_ref",
        "notification_channel",
    }
    assert all(required_audit_fields <= set(event) for event in audit_events)
    assert {event["approval_status"] for event in audit_events if event["approval_required"]} >= {
        "pending_operator_approval",
        "held_budget",
    }
    assert all(event["result"] != "executed" for event in audit_events)
    approval_contracts = nemoclaw["approval_contracts"]
    assert set(approval_contracts) == {action["action_id"] for action in nemoclaw["approval_required_actions"]}
    for action in nemoclaw["approval_required_actions"]:
        contract = action["approval_contract"]
        assert contract == approval_contracts[action["action_id"]]
        assert len(contract["command_sha256"]) == 64
        assert contract["approval_channel"] == "discord_voice_operator_confirmation"
        assert contract["allowed_decisions"] == ["approve_once", "deny", "hold"]
        assert contract["approved_by_ref"] is None
        assert contract["required_preflight_gates"]
    assert phone_context["target_channel"] == "phone"
    assert phone_context["status"] == "queued_requires_approval"
    assert phone_context["pending_approvals"]
    assert all("approval_contract" in approval for approval in phone_context["pending_approvals"])
    assert milestone2_plan["schema_version"] == "voiceops.milestone2.execution_plan.v1"
    assert milestone2_plan["demo_refs"]["phone_context"] == "phone-context.json"
    assert milestone2_plan["spend_policy"] == {
        "currency": "usd",
        "budget_cap_cents": 7000,
        "approval_threshold_cents": 1000,
        "queued_cents": payload["totals"]["approval_required_cents"],
        "held_cents": payload["totals"]["held_budget_cents"],
        "status": "no_live_spend_without_explicit_approval",
    }
    assert milestone2_plan["source_readiness_artifact"] == "provisioning-readiness.json"
    assert {step["step_id"] for step in milestone2_plan["execution_steps"]} >= {
        "provision-voip-provider",
        "buy-service-credit",
        "call-user-phone",
    }
    assert {gate["gate_id"] for gate in milestone2_plan["approval_gates"]} >= {
        "stripe-projects-provisioning",
        "stripe-link-spend",
        "phone-call-handoff",
    }
    assert "deprovision_voip_provider" in milestone2_plan["rollback_plan"]
    assert {step["records_to"] for step in milestone2_plan["read_only_discovery"]} == {
        "audit-ledger.read-only-discovery.jsonl"
    }
    approval_step_ids = {
        step["step_id"]
        for step in milestone2_plan["execution_steps"]
        if step["requires_approval"]
    }
    approval_action_ids = {action["action_id"] for action in milestone2_plan["approval_required_actions"]}
    assert approval_step_ids <= approval_action_ids
    assert "publish-status" in approval_action_ids
    for action in milestone2_plan["approval_required_actions"]:
        assert _dot_get(milestone2_plan, action["expected_receipt_ref"])["schema_ref"] == "receipt_schema"
        assert _dot_get(milestone2_plan, action["rollback_ref"])
        if action["credential_location_required"]:
            assert action["credential_location_schema_ref"] == "credential_location_schema"
            assert _dot_get(milestone2_plan, action["credential_location_ref"])["schema_ref"] == (
                "credential_location_schema"
            )
        else:
            assert action["credential_location_ref"] is None
        evidence = milestone2_plan["expected_post_approval_evidence"][action["action_id"]]
        assert evidence["execution_status"] == "not_executed"
        assert evidence["receipt"] is None
        assert evidence["rollback_receipt"] is None
    assert operator_state["schema_version"] == "voiceops.operator_state.v1"
    assert payload["operator_state"] == operator_state
    assert operator_state["readiness_closure"]["closure_status"] == "needs_external_evidence"
    assert operator_state["current_mode"] == "approval-required"
    assert operator_state["active_voice_surface"]["surface_id"] == "discord_voice"
    assert operator_state["provisioned_services"]
    assert operator_events == operator_state["recent_audit_events"]
    assert operator_events[0]["audit_id"] == "evt-001"
    assert validate_operator_state(operator_state) == []
    assert "DGX Spark" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "static dry-run package" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "Spark target selected, live evidence pending" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "turns a DGX Spark into" not in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "via Hermes /model via Hermes" not in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "nemoclaw-action-packet.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "nemoclaw-action-packet.validation.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "phone-context.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "milestone2-execution-plan.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "readiness-report.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "operator-dashboard.html" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "operator-state.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "operator-state-events.jsonl" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "operator-handoff-preview.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "recording-runbook.md" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "submission-writeup.md" in Path(paths["markdown"]).read_text(encoding="utf-8")
    demo_script = Path(paths["demo_script"]).read_text(encoding="utf-8")
    assert "spoken in Discord" in demo_script
    assert "outbound phone call" in demo_script
    assert "After approval and VoIP provisioning" in demo_script
    assert "Hermes calls the user's phone" not in demo_script
    assert "planned post-approval Stripe-provisioned VoIP path" in demo_script
    runbook = Path(paths["recording_runbook"]).read_text(encoding="utf-8")
    assert "VoiceOps Recording Runbook" in runbook
    assert "static dry-run VoiceOps package" in runbook
    assert "Spark target selected, live evidence pending" in runbook
    assert "Shot List" in runbook
    assert "@NousResearch" in runbook
    assert "Do not show terminal panes or files that contain secrets" in runbook
    assert "Plan Closure Gates" in runbook
    assert "live_discord_voice_operator" in runbook
    assert "production_sidecar" in runbook
    assert "live_turn" in runbook
    assert "spend_and_provisioning_preflight" in runbook
    assert "local_spark_stack_matrix" in runbook
    assert "all_local_stack_smoke:needs_evidence" in runbook
    assert "voiceops.realtime_voice_live_evidence_manifest.v1" in runbook
    assert "voiceops.realtime_voice_live_evidence_validation.v1" in runbook
    assert "--validate-live-evidence" in runbook
    assert "voiceops.milestone2.preflight_evidence.v1" in runbook
    assert "voiceops.milestone2.read_only_discovery.v1" in runbook
    assert "--run-readonly-discovery" in runbook
    assert "read-only-discovery.json" in runbook
    assert "operator-handoff-preview.json" in runbook
    assert "audit-ledger.read-only-discovery.jsonl" in runbook
    assert "spark-matrix-closure-plan.md" in runbook
    assert "Closure artifact: `spark-model-matrix.md`" not in runbook
    writeup = Path(paths["submission_writeup"]).read_text(encoding="utf-8")
    assert "Hermes VoiceOps Submission Writeup" in writeup
    assert "static dry-run package" in writeup
    assert "Spark target selected, live evidence pending" in writeup
    assert "targets a DGX Spark-local household and business operator" in writeup
    assert "turns a DGX Spark into" not in writeup
    assert "NemoClaw" in writeup
    assert "Stripe Skills" in writeup
    assert "@NousResearch" in writeup
    assert "Remaining Closure Gates" in writeup
    assert "live_discord_voice_operator" in writeup
    assert "spend_and_provisioning_preflight" in writeup
    assert "local_spark_stack_matrix" in writeup
    assert "all_local_stack_smoke:needs_evidence" in writeup
    assert "read-only-discovery.json" in Path(paths["dashboard"]).read_text(encoding="utf-8")
    dashboard = Path(paths["dashboard"]).read_text(encoding="utf-8")
    assert "static dry-run package" in dashboard
    assert "Static package ready" in dashboard
    assert "Live/Spark gaps" in dashboard
    assert "Artifact failures" not in dashboard
    assert "Spark target selected, live evidence pending" in dashboard
    assert "Nemotron 3 Super" in dashboard
    assert "KAME Reflex Ack" in dashboard
    assert "scripted_static_ack_until_live_voice_evidence" in dashboard
    assert "requires live voice evidence" in dashboard
    assert "Clearly labeled /model fallback" in dashboard
    assert "Hosted fallback does not count as Spark-local readiness proof" in dashboard
    assert "NemoClaw Blocks" in dashboard
    assert "Plan Closure Gates" in dashboard
    assert "Operator Handoff" in dashboard
    assert "operator-handoff-preview.json" in dashboard
    assert "read_only_discovery" in dashboard
    assert "audit-ledger.read-only-discovery.jsonl" in dashboard
    assert "live_discord_voice_operator" in dashboard
    assert "spend_and_provisioning_preflight" in dashboard
    assert "local_spark_stack_matrix" in dashboard
    assert "Current Mode" in dashboard
    assert "dry_run_until_user_approval" in dashboard
    assert "approval-required" in dashboard
    assert "Active Voice Surface" in dashboard
    assert "discord_voice" in dashboard
    assert "Budget Status" in dashboard
    assert "$70.00" in dashboard
    assert "$10.00" in dashboard
    assert "held-budget" in dashboard
    assert "Pending Approvals" in dashboard
    assert "provision-voip-provider" in dashboard
    assert "call-user-phone" in dashboard
    assert "Action Ledger" in dashboard
    assert "Recent Audit Events" in dashboard
    assert "evt-001" in dashboard
    assert "evt-006" in dashboard
    assert "Planned Services" in dashboard
    assert "Provisioned Services" in dashboard
    assert "repo_local_demo_artifacts" in dashboard
    assert "stripe-projects" in dashboard
    assert "Upcoming Tasks" in dashboard
    assert "operator-state.json" in dashboard
    assert "Phone Handoff" in dashboard
    assert "milestone2-execution-plan.json" in dashboard
    assert Path(paths["readiness_json"]).exists()
    assert "VoiceOps Recording Readiness" in Path(paths["readiness_markdown"]).read_text(encoding="utf-8")
    readiness_markdown = Path(paths["readiness_markdown"]).read_text(encoding="utf-8")
    assert "Static recording ready:" in readiness_markdown
    assert "Live demo ready: no" in readiness_markdown
    assert "Readiness scope: static_artifact_recording_only" in readiness_markdown
    assert "Spark-local evidence: target_selected_needs_benchmark_evidence" in readiness_markdown
    assert "Spark-local=True" not in readiness_markdown
    closure_markdown = Path(paths["readiness_closure_summary_markdown"]).read_text(encoding="utf-8")
    assert "VoiceOps Demo Readiness Closure Summary" in closure_markdown
    assert "voiceops.demo_closure_summary.v1" in closure_markdown
    assert "live_discord_voice_operator" in closure_markdown
    assert "spend_and_provisioning_preflight" in closure_markdown
    assert "local_spark_stack_matrix" in closure_markdown


def test_voiceops_demo_closure_and_handoff_track_plan_run_contracts(tmp_path):
    from scripts.voiceops_plan_run import build_plan_run

    demo_dir = tmp_path / "demo"
    plan_dir = tmp_path / "plan"
    artifact_root = tmp_path / "artifacts"
    args = parse_args(["--output-dir", str(demo_dir)])
    demo = build_demo(args)
    paths = write_demo(demo_dir, demo)

    demo_closure = json.loads(Path(paths["readiness_closure_summary_json"]).read_text(encoding="utf-8"))
    demo_handoff = json.loads(Path(paths["operator_handoff_preview_json"]).read_text(encoding="utf-8"))
    plan = build_plan_run(artifact_root=artifact_root, output_dir=plan_dir, env={})
    plan_closure = plan["closure_index"]

    demo_gates = {gate["gate_id"]: gate for gate in demo_closure["gates"]}
    plan_gates = {gate["gate_id"]: gate for gate in plan_closure["gates"]}
    assert set(demo_gates) == set(plan_gates)
    for gate_id, plan_gate in plan_gates.items():
        demo_gate = demo_gates[gate_id]
        assert demo_gate["collection_commands"] == plan_gate["collection_commands"]
        assert demo_gate["completion_signal"] == plan_gate["completion_signal"]
        assert demo_gate["evidence_contract"] == plan_gate["evidence_contract"]

    demo_phases = {phase["phase_id"]: phase for phase in demo_handoff["phases"]}
    plan_phases = {phase["phase_id"]: phase for phase in plan_closure["operator_handoff"]["phases"]}
    assert set(demo_phases) == set(plan_phases)
    for phase_id, plan_phase in plan_phases.items():
        demo_phase = demo_phases[phase_id]
        assert demo_phase["order"] == plan_phase["order"]
        assert demo_phase["commands"] == plan_phase["commands"]
        assert demo_phase["expected_artifacts"] == plan_phase["expected_artifacts"]
        assert demo_phase["success_check"] == plan_phase["success_check"]
        assert isinstance(demo_phase["blocked_by_current_environment"], dict)


def test_voiceops_demo_classifies_ultra_as_hosted_fallback_and_rejects_unaligned_model(tmp_path):
    hosted_super_args = parse_args(["--active-model", "Nemotron 3 Super via hosted provider"])
    hosted_super_demo = build_demo(hosted_super_args)
    hosted_paths = write_demo(tmp_path / "hosted-super", hosted_super_demo)
    hosted_super_ready = build_readiness_report(
        hosted_super_demo,
        env={
            **_discord_live_env(),
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED": "true",
        },
        which=lambda command: f"/usr/local/bin/{command}" if command in {"stripe", "link-cli"} else None,
    )

    assert hosted_super_demo["sponsor_stack"]["hermes_active_model"]["path"] == "hosted_nemotron_3_super_fallback"
    assert hosted_super_demo["sponsor_stack"]["hermes_active_model"]["spark_local"] is False
    assert hosted_super_demo["sponsor_stack"]["hermes_active_model"]["fallback_used"] is True
    assert hosted_super_demo["spark_stack"]["current_path_local"] is False
    assert hosted_super_demo["spark_stack"]["local_first_status"] == "strategy_target_not_readiness_claim"
    assert hosted_super_demo["sponsor_stack"]["nemotron_3_super"]["selection"] == "not selected"
    assert hosted_super_demo["sponsor_stack"]["nemotron_3_ultra_hosted_fallback"]["selection"].startswith(
        "Nemotron 3 Super"
    )
    assert "nemotron_3_super_spark_or_labeled_hosted_fallback" not in hosted_super_ready["required_failures"]
    assert hosted_super_ready["spark_local_evidence_status"] == "hosted_or_nonlocal_path_not_spark_evidence"
    assert hosted_super_ready["spark_local_readiness"] is False
    assert hosted_super_ready["spark_benchmark_required"] is True
    assert hosted_super_ready["spark_readiness_source"] == "voiceops_spark_matrix.ready_for_one_spark_demo"
    for key in ("markdown", "recording_runbook", "submission_writeup", "dashboard"):
        text = Path(hosted_paths[key]).read_text(encoding="utf-8")
        assert "Hosted fallback selected, Spark-local evidence pending" in text
        assert "Spark target selected, live evidence pending" not in text
        assert "target appliance is one DGX Spark running" not in text
        assert "Spark-powered Hermes operator" not in text
    demo_script = Path(hosted_paths["demo_script"]).read_text(encoding="utf-8")
    assert "clearly labeled hosted fallback today" in demo_script
    assert "Spark-powered Hermes operator" not in demo_script

    ultra_args = parse_args(["--active-model", "Nemotron 3 Ultra via hosted Nous provider"])
    ultra_demo = build_demo(ultra_args)
    ultra_ready = build_readiness_report(
        ultra_demo,
        env={
            **_discord_live_env(),
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED": "true",
        },
        which=lambda command: f"/usr/local/bin/{command}" if command in {"stripe", "link-cli"} else None,
    )

    assert ultra_demo["sponsor_stack"]["hermes_active_model"]["path"] == "hosted_nemotron_3_ultra_fallback"
    assert ultra_demo["sponsor_stack"]["hermes_active_model"]["spark_local"] is False
    assert ultra_demo["sponsor_stack"]["nemotron_3_super"]["selection"] == "not selected"
    assert ultra_demo["sponsor_stack"]["nemotron_3_ultra_hosted_fallback"]["selection"].startswith("Nemotron 3 Ultra")
    assert "nemotron_3_super_spark_or_labeled_hosted_fallback" not in ultra_ready["required_failures"]
    assert ultra_ready["spark_local_evidence_status"] == "hosted_or_nonlocal_path_not_spark_evidence"
    assert ultra_ready["spark_local_readiness"] is False
    assert ultra_ready["spark_benchmark_required"] is True
    assert ultra_ready["live_demo_ready"] is False

    kimi_args = parse_args(["--active-model", "Kimi K2.6"])
    kimi_demo = build_demo(kimi_args)
    kimi_ready = build_readiness_report(
        kimi_demo,
        env={
            **_discord_live_env(),
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED": "true",
        },
        which=lambda command: f"/usr/local/bin/{command}" if command in {"stripe", "link-cli"} else None,
    )

    assert kimi_demo["sponsor_stack"]["hermes_active_model"]["path"] == "non_nvidia_fallback"
    assert kimi_demo["sponsor_stack"]["hermes_active_model"]["spark_local"] is False
    assert "nemotron_3_super_spark_or_labeled_hosted_fallback" in kimi_ready["required_failures"]


def test_voiceops_demo_dry_run_does_not_execute_live_stripe(tmp_path):
    args = parse_args(["--output-dir", str(tmp_path)])
    demo = build_demo(args)
    paths = write_demo(tmp_path, demo)
    text = Path(paths["stripe_actions"]).read_text(encoding="utf-8")
    nemoclaw = json.loads(Path(paths["nemoclaw_packet"]).read_text(encoding="utf-8"))
    metadata = [
        json.loads(line.removeprefix("# voiceops-action-metadata "))
        for line in text.splitlines()
        if line.startswith("# voiceops-action-metadata ")
    ]

    assert "printf '%s\\n'" in text
    assert "voiceops-action-metadata" in text
    assert "stripe projects add twilio/voice" in text
    assert "link-cli spend-request create" in text
    assert "queue outbound call" in text
    assert "draft Discord and WhatsApp status summary" in text
    assert "post summary to Discord and WhatsApp" not in text
    assert "provisioning preflight, channel policy, Link approval, and command review pass" in text
    assert metadata
    assert len(metadata) == len(demo["ops_actions"])
    assert {item["schema_version"] for item in metadata} == {"voiceops.stripe_actions_dry_run.metadata.v1"}
    assert all(item["execution_mode"] == "dry_run_printf_only" for item in metadata)
    assert all(item["provider_command_executes"] is False for item in metadata)
    metadata_by_action = {item["action_id"]: item for item in metadata}
    for action in demo["ops_actions"]:
        item = metadata_by_action[action["action_id"]]
        assert item["provider"] == action["provider"]
        assert item["command"] == action["command"]
        assert item["purpose"] == action["purpose"]
        assert item["estimated_cents"] == action["estimated_cents"]
        assert item["requires_approval"] == action["requires_approval"]
        assert item["status"] == action["status"]
        assert item["command_sha256"] == hashlib.sha256(action["command"].encode("utf-8")).hexdigest()
    for action in nemoclaw["approval_required_actions"]:
        contract = action["approval_contract"]
        item = metadata_by_action[action["action_id"]]
        assert item["approval_id"] == contract["approval_id"]
        assert item["approval_channel"] == contract["approval_channel"]
        assert item["approval_artifact"] == contract["approval_artifact"]
        assert item["approved_by_ref"] == contract["approved_by_ref"]
        assert item["allowed_decisions"] == contract["allowed_decisions"]
        assert item["command_sha256"] == contract["command_sha256"]
        assert item["default_decision"] == contract["default_decision"]
        assert item["required_preflight_gates"] == contract["required_preflight_gates"]
        assert item["ttl_seconds"] == contract["ttl_seconds"]
    executable_lines = [
        line for line in text.splitlines() if line and not line.startswith("#") and not line.startswith("printf ")
    ]
    assert executable_lines == ["set -euo pipefail"]
    for line in text.splitlines():
        if line.startswith("#") or line.startswith("printf ") or line == "set -euo pipefail" or not line:
            continue
        assert not line.startswith(("stripe ", "link-cli ", "queue outbound call"))
    assert "sk_" not in text


def test_voiceops_demo_cli_smoke(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "hackathon_voiceops_demo.py"
    result = subprocess.run(
        ["python", str(script), "--output-dir", str(tmp_path), "--budget-cents", "20000"],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert Path(payload["artifacts"]["json"]).exists()


def test_voiceops_readiness_report_distinguishes_required_failures():
    args = parse_args([])
    demo = build_demo(args)

    def fake_which(command: str) -> str | None:
        commands = {
            "hermes": "/usr/local/bin/hermes",
            "stripe": "/usr/local/bin/stripe",
            "link-cli": "/usr/local/bin/link-cli",
            "nemoclaw": "/usr/local/bin/nemoclaw",
        }
        return commands.get(command)

    ready = build_readiness_report(
        demo,
        env={
            **_discord_live_env(),
            "WHATSAPP_ENABLED": "true",
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "VOICEOPS_PHONE_PROVIDER_ACCOUNT_REF": "twilio:acct:redacted-demo",
            "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED": "true",
        },
        which=fake_which,
    )
    assert ready["ready_for_recording"] is True
    assert ready["static_recording_ready"] is True
    assert ready["ready_for_recording_scope"] == "static_artifact_recording_only"
    assert ready["live_demo_ready"] is False
    assert ready["live_demo_missing_evidence"] == [
        "live_discord_voice_operator",
        "spend_and_provisioning_preflight",
        "local_spark_stack_matrix",
    ]
    assert ready["spark_local_evidence_status"] == "target_selected_needs_benchmark_evidence"
    assert ready["spark_local_readiness"] is False
    assert ready["spark_benchmark_required"] is True
    assert ready["spark_readiness_source"] == "voiceops_spark_matrix.ready_for_one_spark_demo"
    assert ready["required_failures"] == []
    assert ready["artifact_required_failures"] == []
    assert ready["live_prerequisite_failures"] == []
    assert ready["all_required_check_failures"] == []
    checks = {check["check_id"]: check for check in ready["checks"]}
    assert checks["phone_target"]["status"] == "pass"
    assert checks["phone_provider"]["status"] == "pass"
    assert checks["phone_handoff"]["status"] == "pass"

    partial_discord = build_readiness_report(
        demo,
        env={
            "DISCORD_BOT_TOKEN": "set",
            "DISCORD_VOICE_CHANNEL_ID": "123",
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "VOICEOPS_PHONE_PROVIDER_ACCOUNT_REF": "twilio:acct:redacted-demo",
            "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED": "true",
        },
        which=fake_which,
    )
    partial_discord_checks = {check["check_id"]: check for check in partial_discord["checks"]}
    assert partial_discord_checks["discord_voice"]["status"] == "fail"
    assert "DISCORD_GUILD_ID" in partial_discord_checks["discord_voice"]["detail"]
    assert "DISCORD_HOME_CHANNEL" in partial_discord_checks["discord_voice"]["detail"]
    assert "DISCORD_VOICE_CHANNEL_NAME" in partial_discord_checks["discord_voice"]["detail"]
    assert "discord_voice" in partial_discord["live_prerequisite_failures"]

    stripe_without_projects_marker = build_readiness_report(
        demo,
        env={
            **_discord_live_env(),
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
        },
        which=fake_which,
    )
    assert stripe_without_projects_marker["ready_for_recording"] is True
    assert stripe_without_projects_marker["static_recording_ready"] is True
    assert stripe_without_projects_marker["required_failures"] == []
    assert "stripe_projects_cli" in stripe_without_projects_marker["live_prerequisite_failures"]

    def fake_without_nemoclaw(command: str) -> str | None:
        commands = {
            "hermes": "/usr/local/bin/hermes",
            "stripe": "/usr/local/bin/stripe",
            "link-cli": "/usr/local/bin/link-cli",
        }
        return commands.get(command)

    no_boundary = build_readiness_report(
        demo,
        env={
            **_discord_live_env(),
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "VOICEOPS_PHONE_PROVIDER_ACCOUNT_REF": "twilio:acct:redacted-demo",
            "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED": "true",
        },
        which=fake_without_nemoclaw,
    )
    no_boundary_checks = {check["check_id"]: check for check in no_boundary["checks"]}
    assert no_boundary_checks["nemoclaw_boundary"]["status"] == "fail"
    assert "nemoclaw_boundary" in no_boundary["live_prerequisite_failures"]
    assert "mpp_agent" in build_milestone2_like_failures(demo, no_boundary)

    def fake_npx_only(command: str) -> str | None:
        commands = {
            "hermes": "/usr/local/bin/hermes",
            "stripe": "/usr/local/bin/stripe",
            "npx": "/usr/local/bin/npx",
        }
        return commands.get(command)

    npx_not_ready = build_readiness_report(
        demo,
        env={
            **_discord_live_env(),
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED": "true",
        },
        which=fake_npx_only,
    )
    assert npx_not_ready["ready_for_recording"] is True
    assert npx_not_ready["static_recording_ready"] is True
    assert npx_not_ready["required_failures"] == []
    assert "stripe_link_cli" in npx_not_ready["live_prerequisite_failures"]

    target_only = build_readiness_report(
        demo,
        env={
            **_discord_live_env(),
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED": "true",
        },
        which=fake_which,
    )
    target_only_checks = {check["check_id"]: check for check in target_only["checks"]}
    assert target_only_checks["phone_target"]["status"] == "pass"
    assert target_only_checks["phone_provider"]["status"] == "warn"
    assert target_only_checks["phone_handoff"]["status"] == "warn"
    assert "phone_handoff" in target_only["live_prerequisite_failures"]
    assert "phone_provider" in build_milestone2_like_failures(demo, target_only)

    provider_only = build_readiness_report(
        demo,
        env={
            **_discord_live_env(),
            "VOICEOPS_PHONE_PROVIDER_ACCOUNT_REF": "twilio:acct:redacted-demo",
            "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED": "true",
        },
        which=fake_which,
    )
    provider_only_checks = {check["check_id"]: check for check in provider_only["checks"]}
    assert provider_only_checks["phone_target"]["status"] == "warn"
    assert provider_only_checks["phone_provider"]["status"] == "pass"
    assert provider_only_checks["phone_handoff"]["status"] == "warn"
    assert "phone_handoff" in provider_only["live_prerequisite_failures"]
    assert "phone_target" in build_milestone2_like_failures(demo, provider_only)

    not_ready = build_readiness_report(demo, env={}, which=lambda _command: None)
    assert not_ready["ready_for_recording"] is True
    assert not_ready["static_recording_ready"] is True
    assert not_ready["required_failures"] == []
    assert {"discord_voice", "stripe_projects_cli", "stripe_link_cli", "nemoclaw_boundary", "phone_handoff"}.issubset(
        set(not_ready["live_prerequisite_failures"])
    )


def test_voiceops_readiness_report_loads_env_files_without_exposing_values(tmp_path):
    args = parse_args([])
    demo = build_demo(args)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DISCORD_BOT_TOKEN=secret-token",
                "DISCORD_GUILD_ID=guild-ref",
                "DISCORD_HOME_CHANNEL=general",
                "DISCORD_VOICE_CHANNEL_ID=voice-channel-ref",
                "DISCORD_VOICE_CHANNEL_NAME=General",
                "WHATSAPP_ENABLED=true",
                "VOICEOPS_DEMO_PHONE_NUMBER='+15551234567'",
                "VOICEOPS_PHONE_PROVIDER_ACCOUNT_REF=twilio:acct:redacted-demo",
                "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED=true",
            ]
        ),
        encoding="utf-8",
    )

    def fake_which(command: str) -> str | None:
        commands = {
            "stripe": "/usr/local/bin/stripe",
            "link-cli": "/usr/local/bin/link-cli",
            "nemoclaw": "/usr/local/bin/nemoclaw",
        }
        return commands.get(command)

    report = build_readiness_report(demo, env={}, env_files=[env_file], which=fake_which)

    assert report["ready_for_recording"] is True
    assert report["static_recording_ready"] is True
    assert report["live_demo_ready"] is False
    assert report["ready_for_recording_scope"] == "static_artifact_recording_only"
    assert report["required_failures"] == []
    assert report["live_prerequisite_failures"] == []
    assert report["env_sources"][1]["path"] == str(env_file)
    assert report["env_sources"][1]["loaded"] is True
    assert "secret-token" not in json.dumps(report)
