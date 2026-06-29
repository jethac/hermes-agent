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
    phone_context = json.loads(Path(paths["phone_context"]).read_text(encoding="utf-8"))
    milestone2_plan = json.loads(Path(paths["milestone2_execution_plan"]).read_text(encoding="utf-8"))
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
    assert payload["artifact_manifest"]["operator_state"] == "operator-state.json"
    assert payload["artifact_manifest"]["milestone2_execution_plan"] == "milestone2-execution-plan.json"
    assert readiness["schema_version"] == "voiceops.recording_readiness_report.v1"
    assert readiness["artifact_id"] == "voiceops-recording-readiness-report"
    assert readiness["source_demo_artifact"] == "voiceops-demo.json"
    assert readiness["readiness_closure_summary_ref"] == "readiness-closure-summary.json"
    assert payload["recording_readiness"]["artifact_ref"] == "readiness-report.json"
    assert payload["recording_readiness"]["ready_for_recording"] is True
    assert payload["recording_readiness"]["static_recording_ready"] is True
    assert payload["recording_readiness"]["ready_for_recording_scope"] == "static_artifact_recording_only"
    assert payload["recording_readiness"]["live_demo_ready"] is False
    assert payload["recording_readiness"]["live_prerequisite_failures"] == [
        "discord_voice",
        "stripe_projects_cli",
        "stripe_link_cli",
    ]
    assert payload["recording_readiness"]["live_demo_missing_evidence"] == [
        "live_discord_voice_operator",
        "spend_and_provisioning_preflight",
        "local_spark_stack_matrix",
    ]
    assert payload["recording_readiness"]["spark_local_evidence_status"] == "target_selected_needs_benchmark_evidence"
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
    assert "--validate-live-evidence" in closure_gates["live_discord_voice_operator"]["collection_commands"][
        "validate_live_manifest_offline"
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
    assert payload["milestone2_execution_plan_ref"] == "milestone2-execution-plan.json"
    assert payload["safety_boundary_refs"] == {
        "nemoclaw_action_packet": "nemoclaw-action-packet.json",
        "phone_context": "phone-context.json",
        "stripe_actions_dry_run": "stripe-actions-dry-run.sh",
    }
    assert payload["spark_stack"]["local_first"] is True
    assert payload["sponsor_stack"]["hermes_active_model"]["path"] == "spark_local_nemotron_3_super"
    assert payload["sponsor_stack"]["hermes_active_model"]["selected_by"] == "Hermes /model"
    assert payload["sponsor_stack"]["hermes_active_model"]["spark_local"] is True
    assert payload["sponsor_stack"]["nemotron_3_super"]["selection"].startswith("Nemotron 3 Super")
    assert payload["sponsor_stack"]["nemotron_3_ultra_hosted_fallback"]["selection"].startswith(
        "Hosted /model fallback"
    )
    assert payload["spark_stack"]["oracle"]["preferred_local_target"] == "Nemotron 3 Super on DGX Spark"
    assert payload["spark_stack"]["oracle"]["hosted_fallback"].startswith("clearly labeled hosted provider")
    assert payload["spark_stack"]["oracle"]["active_model_path"]["path"] == "spark_local_nemotron_3_super"
    assert payload["sponsor_stack"]["stripe_skills"]["skills"] == ["stripe-projects", "stripe-link-cli", "mpp-agent"]
    assert payload["voice_surfaces"][0]["channel"] == "discord"
    assert {surface["channel"] for surface in payload["voice_surfaces"]} == {"discord", "whatsapp", "phone"}
    assert next(surface for surface in payload["voice_surfaces"] if surface["channel"] == "phone")["status"] == "dry-run-queued"
    assert "provision-voip-provider" in action_ids
    assert "call-user-phone" in action_ids
    assert payload["totals"]["held_budget_cents"] > 0
    assert nemoclaw["runtime"] == "NemoClaw"
    assert nemoclaw["mode"] == "dry_run_until_user_approval"
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
    assert validate_operator_state(operator_state) == []
    assert "DGX Spark" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "via Hermes /model via Hermes" not in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "nemoclaw-action-packet.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "phone-context.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "milestone2-execution-plan.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "readiness-report.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "operator-dashboard.html" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "operator-state.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "operator-state-events.jsonl" in Path(paths["markdown"]).read_text(encoding="utf-8")
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
    assert "audit-ledger.read-only-discovery.jsonl" in runbook
    assert "spark-matrix-closure-plan.md" in runbook
    assert "Closure artifact: `spark-model-matrix.md`" not in runbook
    writeup = Path(paths["submission_writeup"]).read_text(encoding="utf-8")
    assert "Hermes VoiceOps Submission Writeup" in writeup
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
    assert "Nemotron 3 Super" in dashboard
    assert "Clearly labeled /model fallback" in dashboard
    assert "Hosted fallback does not count as Spark-local readiness proof" in dashboard
    assert "NemoClaw Blocks" in dashboard
    assert "Plan Closure Gates" in dashboard
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


def test_voiceops_demo_classifies_ultra_as_hosted_fallback_and_rejects_unaligned_model():
    ultra_args = parse_args(["--active-model", "Nemotron 3 Ultra via hosted Nous provider"])
    ultra_demo = build_demo(ultra_args)
    ultra_ready = build_readiness_report(
        ultra_demo,
        env={
            "DISCORD_BOT_TOKEN": "set",
            "DISCORD_VOICE_CHANNEL_ID": "123",
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
    assert ultra_ready["live_demo_ready"] is False

    kimi_args = parse_args(["--active-model", "Kimi K2.6"])
    kimi_demo = build_demo(kimi_args)
    kimi_ready = build_readiness_report(
        kimi_demo,
        env={
            "DISCORD_BOT_TOKEN": "set",
            "DISCORD_VOICE_CHANNEL_ID": "123",
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
            "DISCORD_BOT_TOKEN": "set",
            "DISCORD_VOICE_CHANNEL_ID": "123",
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
    assert ready["required_failures"] == []
    assert ready["artifact_required_failures"] == []
    assert ready["live_prerequisite_failures"] == []
    assert ready["all_required_check_failures"] == []
    checks = {check["check_id"]: check for check in ready["checks"]}
    assert checks["phone_target"]["status"] == "pass"
    assert checks["phone_provider"]["status"] == "pass"
    assert checks["phone_handoff"]["status"] == "pass"

    stripe_without_projects_marker = build_readiness_report(
        demo,
        env={
            "DISCORD_BOT_TOKEN": "set",
            "DISCORD_VOICE_CHANNEL_ID": "123",
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
        },
        which=fake_which,
    )
    assert stripe_without_projects_marker["ready_for_recording"] is True
    assert stripe_without_projects_marker["static_recording_ready"] is True
    assert stripe_without_projects_marker["required_failures"] == []
    assert "stripe_projects_cli" in stripe_without_projects_marker["live_prerequisite_failures"]

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
            "DISCORD_BOT_TOKEN": "set",
            "DISCORD_VOICE_CHANNEL_ID": "123",
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
            "DISCORD_BOT_TOKEN": "set",
            "DISCORD_VOICE_CHANNEL_ID": "123",
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED": "true",
        },
        which=fake_which,
    )
    target_only_checks = {check["check_id"]: check for check in target_only["checks"]}
    assert target_only_checks["phone_target"]["status"] == "pass"
    assert target_only_checks["phone_provider"]["status"] == "warn"
    assert target_only_checks["phone_handoff"]["status"] == "warn"
    assert "phone_provider" in build_milestone2_like_failures(demo, target_only)

    provider_only = build_readiness_report(
        demo,
        env={
            "DISCORD_BOT_TOKEN": "set",
            "DISCORD_VOICE_CHANNEL_ID": "123",
            "VOICEOPS_PHONE_PROVIDER_ACCOUNT_REF": "twilio:acct:redacted-demo",
            "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED": "true",
        },
        which=fake_which,
    )
    provider_only_checks = {check["check_id"]: check for check in provider_only["checks"]}
    assert provider_only_checks["phone_target"]["status"] == "warn"
    assert provider_only_checks["phone_provider"]["status"] == "pass"
    assert provider_only_checks["phone_handoff"]["status"] == "warn"
    assert "phone_target" in build_milestone2_like_failures(demo, provider_only)

    not_ready = build_readiness_report(demo, env={}, which=lambda _command: None)
    assert not_ready["ready_for_recording"] is True
    assert not_ready["static_recording_ready"] is True
    assert not_ready["required_failures"] == []
    assert {"discord_voice", "stripe_projects_cli", "stripe_link_cli"}.issubset(
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
                "DISCORD_VOICE_CHANNEL_NAME=General",
                "WHATSAPP_ENABLED=true",
                "VOICEOPS_DEMO_PHONE_NUMBER='+15551234567'",
                "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED=true",
            ]
        ),
        encoding="utf-8",
    )

    def fake_which(command: str) -> str | None:
        commands = {
            "stripe": "/usr/local/bin/stripe",
            "link-cli": "/usr/local/bin/link-cli",
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
