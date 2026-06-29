#!/usr/bin/env python3
"""Audit the generated VoiceOps artifact package for cross-file consistency.

This script is intentionally local-only. It reads generated JSON/Markdown/HTML
artifacts and never inspects env files, contacts providers, runs Discord, spends
money, provisions services, retrieves credentials, sends messages, or places
calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.voiceops_channel_policy import CHANNEL_IDS, validate_policy


DEFAULT_ARTIFACT_ROOT = Path("artifacts")
DEFAULT_OUTPUT_DIR = Path("artifacts/voiceops-package-audit/current")
AUDIT_SCHEMA_VERSION = "voiceops.artifact_package_audit.v1"


def _read_json(path: Path, issues: list[str], label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        issues.append(f"{label}:missing:{path}")
        return {}
    except json.JSONDecodeError as exc:
        issues.append(f"{label}:json_parse_failed:{exc.msg}")
        return {}
    if not isinstance(payload, dict):
        issues.append(f"{label}:root_must_be_object")
        return {}
    return payload


def _read_text(path: Path, issues: list[str], label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        issues.append(f"{label}:missing:{path}")
    except UnicodeDecodeError:
        issues.append(f"{label}:not_utf8:{path}")
    return ""


def _read_jsonl(path: Path, issues: list[str], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = _read_text(path, issues, label)
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"{label}:line_{line_number}:json_parse_failed:{exc.msg}")
            continue
        if not isinstance(row, dict):
            issues.append(f"{label}:line_{line_number}:row_must_be_object")
            continue
        rows.append(row)
    return rows


def _dry_run_metadata_rows(script_text: str, issues: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prefix = "# voiceops-action-metadata "
    for line_number, line in enumerate(script_text.splitlines(), start=1):
        if not line.startswith(prefix):
            continue
        try:
            row = json.loads(line.removeprefix(prefix))
        except json.JSONDecodeError as exc:
            issues.append(f"stripe_actions:metadata_line_{line_number}:json_parse_failed:{exc.msg}")
            continue
        if not isinstance(row, dict):
            issues.append(f"stripe_actions:metadata_line_{line_number}:row_must_be_object")
            continue
        rows.append(row)
    executable_lines = [
        line
        for line in script_text.splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("printf ")
    ]
    if executable_lines != ["set -euo pipefail"]:
        issues.append("stripe_actions:unexpected_executable_lines")
    return rows


def _approval_contract_subset(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "approval_id": contract.get("approval_id"),
        "command_sha256": contract.get("command_sha256"),
        "default_decision": contract.get("default_decision"),
        "approved_by_ref": contract.get("approved_by_ref"),
    }


def _audit_static_readiness(
    *,
    demo: Mapping[str, Any],
    readiness: Mapping[str, Any],
    demo_closure: Mapping[str, Any],
    plan_closure: Mapping[str, Any],
    operator_state: Mapping[str, Any],
    dashboard_html: str,
    issues: list[str],
) -> None:
    remaining_gates = plan_closure.get("remaining_gates") or []
    closure_incomplete = bool(remaining_gates) or plan_closure.get("closure_status") != "complete"
    recording_readiness = demo.get("recording_readiness") if isinstance(demo.get("recording_readiness"), Mapping) else {}
    if closure_incomplete:
        if readiness.get("live_demo_ready") is not False:
            issues.append("readiness:live_demo_ready_not_false_with_remaining_gates")
        if recording_readiness.get("live_demo_ready") is not False:
            issues.append("demo:live_demo_ready_not_false_with_remaining_gates")
        if readiness.get("ready_for_recording_scope") != "static_artifact_recording_only":
            issues.append("readiness:scope_not_static_artifact_recording_only")
        if recording_readiness.get("ready_for_recording_scope") != "static_artifact_recording_only":
            issues.append("demo:scope_not_static_artifact_recording_only")
        if not recording_readiness.get("live_demo_missing_evidence"):
            issues.append("demo:missing_live_demo_missing_evidence")
        voice_surface = operator_state.get("active_voice_surface")
        if isinstance(voice_surface, Mapping) and voice_surface.get("status") == "active_for_demo":
            if readiness.get("ready_for_recording_scope") != "static_artifact_recording_only":
                issues.append("operator_state:active_for_demo_without_static_recording_scope")
            if not voice_surface.get("fallback_reason"):
                issues.append("operator_state:active_for_demo_without_visible_fallback_reason")
        required_dashboard_tokens = [
            "scripted_static_ack_until_live_voice_evidence",
            "needs_live_probe",
            "needs_setup",
            "needs_evidence",
        ]
        for token in required_dashboard_tokens:
            if token not in dashboard_html:
                issues.append(f"dashboard:missing_non_live_token:{token}")

    demo_gate_ids = {str(gate.get("gate_id")) for gate in demo_closure.get("gates", []) if isinstance(gate, Mapping)}
    plan_gate_ids = {str(gate.get("gate_id")) for gate in plan_closure.get("gates", []) if isinstance(gate, Mapping)}
    if demo_gate_ids != plan_gate_ids:
        issues.append("closure:gates_mismatch_between_demo_and_plan")


def _audit_action_consistency(
    *,
    packet: Mapping[str, Any],
    packet_validation: Mapping[str, Any],
    operator_state: Mapping[str, Any],
    audit_rows: list[dict[str, Any]],
    dry_run_rows: list[dict[str, Any]],
    issues: list[str],
) -> None:
    if packet_validation.get("status") != "valid" or packet_validation.get("ok") is not True:
        issues.append("nemoclaw:validation_not_valid")

    actions = {
        str(action.get("action_id")): action
        for action in packet.get("approval_required_actions", [])
        if isinstance(action, Mapping)
    }
    contracts = packet.get("approval_contracts") if isinstance(packet.get("approval_contracts"), Mapping) else {}
    pending = {
        str(item.get("action_id")): item
        for item in operator_state.get("pending_approvals", [])
        if isinstance(item, Mapping)
    }
    state_contracts = (
        operator_state.get("approval_contracts")
        if isinstance(operator_state.get("approval_contracts"), Mapping)
        else {}
    )
    audit_by_action = {str(row.get("action")): row for row in audit_rows}
    dry_run_by_action = {str(row.get("action_id")): row for row in dry_run_rows}

    if set(actions) != set(pending):
        issues.append("approvals:pending_action_ids_do_not_match_nemoclaw_actions")
    for action_id, action in actions.items():
        command = str(action.get("command") or "")
        contract = action.get("approval_contract") if isinstance(action.get("approval_contract"), Mapping) else {}
        indexed_contract = contracts.get(action_id)
        pending_item = pending.get(action_id, {})
        pending_contract = (
            pending_item.get("approval_contract") if isinstance(pending_item.get("approval_contract"), Mapping) else {}
        )
        state_contract = state_contracts.get(action_id) if isinstance(state_contracts, Mapping) else None
        expected_hash = hashlib.sha256(command.encode("utf-8")).hexdigest()
        if contract.get("command_sha256") != expected_hash:
            issues.append(f"nemoclaw:{action_id}:command_sha256_mismatch")
        if indexed_contract != contract:
            issues.append(f"nemoclaw:{action_id}:indexed_contract_mismatch")
        if state_contract != contract:
            issues.append(f"operator_state:{action_id}:approval_contract_mismatch")
        if _approval_contract_subset(pending_contract) != _approval_contract_subset(contract):
            issues.append(f"operator_state:{action_id}:pending_contract_mismatch")
        if pending_item.get("execution_status") != "not_executed":
            issues.append(f"operator_state:{action_id}:pending_approval_executed")
        if pending_item.get("status") not in {"pending", "held"}:
            issues.append(f"operator_state:{action_id}:pending_status_invalid")
        if contract.get("default_decision") != "hold":
            issues.append(f"approvals:{action_id}:default_decision_not_hold")
        if contract.get("approved_by_ref") is not None:
            issues.append(f"approvals:{action_id}:approved_by_ref_present")

        audit = audit_by_action.get(action_id)
        if not audit:
            issues.append(f"audit_ledger:{action_id}:missing")
        else:
            if audit.get("approval_required") is not True:
                issues.append(f"audit_ledger:{action_id}:approval_required_not_true")
            if audit.get("approval_status") not in {"pending_operator_approval", "held_budget"}:
                issues.append(f"audit_ledger:{action_id}:approval_status_invalid")
            if audit.get("result") == "executed":
                issues.append(f"audit_ledger:{action_id}:result_executed")
            if audit.get("status") not in {"queued", "held-budget", "blocked"}:
                issues.append(f"audit_ledger:{action_id}:status_invalid")

        dry_run = dry_run_by_action.get(action_id)
        if not dry_run:
            issues.append(f"stripe_actions:{action_id}:metadata_missing")
        else:
            if dry_run.get("command") != command:
                issues.append(f"stripe_actions:{action_id}:command_mismatch")
            if dry_run.get("command_sha256") != expected_hash:
                issues.append(f"stripe_actions:{action_id}:command_sha256_mismatch")
            if dry_run.get("provider_command_executes") is not False:
                issues.append(f"stripe_actions:{action_id}:provider_command_executes_not_false")
            if dry_run.get("execution_mode") != "dry_run_printf_only":
                issues.append(f"stripe_actions:{action_id}:execution_mode_invalid")

    for row in audit_rows:
        if row.get("result") == "executed":
            issues.append(f"audit_ledger:{row.get('action')}:unexpected_executed_result")


def _audit_service_claims(operator_state: Mapping[str, Any], issues: list[str]) -> None:
    for service in operator_state.get("planned_services", []):
        if not isinstance(service, Mapping):
            continue
        if service.get("external") is True:
            if service.get("execution_status") != "not_executed":
                issues.append(f"planned_services:{service.get('service_id')}:external_execution_claim")
            if service.get("status") not in {"approval_required", "planned", "queued"}:
                issues.append(f"planned_services:{service.get('service_id')}:external_status_invalid")
    for service in operator_state.get("provisioned_services", []):
        if not isinstance(service, Mapping):
            continue
        if service.get("external") is True:
            issues.append(f"provisioned_services:{service.get('service_id')}:external_service_claimed_provisioned")
        if service.get("execution_status") not in {"local_artifact_written", "not_executed"}:
            issues.append(f"provisioned_services:{service.get('service_id')}:execution_status_invalid")


def _iter_plan_run_commands(value: Any) -> list[str]:
    commands: list[str] = []
    if isinstance(value, str):
        if "scripts/voiceops_plan_run.py" in value:
            commands.append(value)
    elif isinstance(value, Mapping):
        for nested in value.values():
            commands.extend(_iter_plan_run_commands(nested))
    elif isinstance(value, list):
        for nested in value:
            commands.extend(_iter_plan_run_commands(nested))
    return commands


def _audit_plan_consistency(
    *,
    demo_closure: Mapping[str, Any],
    demo_handoff: Mapping[str, Any],
    plan_run: Mapping[str, Any],
    plan_closure: Mapping[str, Any],
    plan_handoff: Mapping[str, Any],
    issues: list[str],
) -> None:
    if plan_run.get("closure_index") != plan_closure:
        issues.append("plan_run:closure_index_mismatch")
    if plan_handoff != plan_closure.get("operator_handoff"):
        issues.append("operator_handoff:mismatch_with_closure")
    for label, payload in (
        ("demo_closure", demo_closure),
        ("demo_handoff", demo_handoff),
        ("plan_closure", plan_closure),
        ("operator_handoff", plan_handoff),
    ):
        for command in _iter_plan_run_commands(payload):
            if "--package-audit" not in command:
                issues.append(f"{label}:plan_run_command_missing_package_audit")

    plan_final_command = plan_handoff.get("final_reindex_command")
    if demo_handoff.get("final_reindex_command") != plan_final_command:
        issues.append("demo_handoff:final_reindex_command_mismatch")
    package_audit_command = plan_handoff.get("final_package_audit_command")
    if not package_audit_command or "voiceops_artifact_package_audit.py" not in str(package_audit_command):
        issues.append("operator_handoff:missing_final_package_audit_command")
    if demo_handoff.get("final_package_audit_command") != package_audit_command:
        issues.append("demo_handoff:final_package_audit_command_mismatch")
    final_success_signal = str(plan_handoff.get("final_success_signal") or "")
    if "package_audit.status is pass" not in final_success_signal:
        issues.append("operator_handoff:final_success_signal_missing_package_audit")
    if demo_handoff.get("final_success_signal") != plan_handoff.get("final_success_signal"):
        issues.append("demo_handoff:final_success_signal_mismatch")


def _audit_channel_policy(policy: Mapping[str, Any], review: Mapping[str, Any], issues: list[str]) -> None:
    for issue in validate_policy(dict(policy)):
        issues.append(f"channel_policy:validation:{issue}")

    scope = policy.get("scope") if isinstance(policy.get("scope"), Mapping) else {}
    if scope.get("real_egress_enabled") is not False:
        issues.append("channel_policy:real_egress_enabled_not_false")
    if scope.get("review_required_for_real_egress") is not True:
        issues.append("channel_policy:review_required_for_real_egress_not_true")
    if scope.get("review_status") != "pending_human_review":
        issues.append("channel_policy:review_status_not_pending")
    if set(scope.get("channels") or []) != set(CHANNEL_IDS):
        issues.append("channel_policy:scope_channels_mismatch")

    channel_ids = {
        str(channel.get("channel_id"))
        for channel in policy.get("channel_authorization", [])
        if isinstance(channel, Mapping)
    }
    if channel_ids != set(CHANNEL_IDS):
        issues.append("channel_policy:channel_authorization_mismatch")
    policy_channels = {
        str(channel.get("channel_id")): channel
        for channel in policy.get("channel_authorization", [])
        if isinstance(channel, Mapping)
    }
    if review.get("schema_version") != "voiceops.multi_channel_policy_review.v1":
        issues.append("channel_policy_review:schema_version_mismatch")
    if review.get("artifact_id") != "voiceops-m3-channel-policy-review":
        issues.append("channel_policy_review:artifact_id_mismatch")
    for key in ("milestone", "policy_id", "policy_version"):
        if review.get(key) != policy.get(key):
            issues.append(f"channel_policy_review:{key}_mismatch")
    if review.get("artifact_only") is not True:
        issues.append("channel_policy_review:artifact_only_not_true")
    if review.get("policy_ref") != "channel-policy.json":
        issues.append("channel_policy_review:policy_ref_mismatch")
    if review.get("review_status") != "pending_human_review":
        issues.append("channel_policy_review:review_status_not_pending")
    if review.get("real_egress_enabled") is not False:
        issues.append("channel_policy_review:real_egress_enabled_not_false")
    if review.get("changes_policy") is not False:
        issues.append("channel_policy_review:changes_policy_not_false")
    decision_options = set(review.get("decision_options") or [])
    if {"request_changes", "deny", "approve_dry_run_only"} - decision_options:
        issues.append("channel_policy_review:decision_options_missing_safe_choices")
    review_channels = {
        str(channel.get("channel_id")): channel
        for channel in review.get("per_channel_review", [])
        if isinstance(channel, Mapping)
    }
    if set(review_channels) != set(CHANNEL_IDS):
        issues.append("channel_policy_review:per_channel_mismatch")
    for channel_id, channel in review_channels.items():
        if channel.get("live_egress_enabled") is not False:
            issues.append(f"channel_policy_review:{channel_id}:live_egress_enabled_not_false")
        if channel.get("review_status") != "pending_human_review":
            issues.append(f"channel_policy_review:{channel_id}:review_status_not_pending")
        policy_channel = policy_channels.get(channel_id, {})
        if set(channel.get("required_evidence") or []) != set(policy_channel.get("evidence_required") or []):
            issues.append(f"channel_policy_review:{channel_id}:required_evidence_mismatch")
        route_map = policy.get("approval_route_map") if isinstance(policy.get("approval_route_map"), Mapping) else {}
        if dict(channel.get("approval_routes_to_confirm") or {}) != dict(route_map.get(channel_id) or {}):
            issues.append(f"channel_policy_review:{channel_id}:approval_routes_mismatch")
        if set(channel.get("blocked_capabilities_to_confirm") or []) != set(policy_channel.get("prohibited_actions") or []):
            issues.append(f"channel_policy_review:{channel_id}:blocked_capabilities_mismatch")
    phone_review = review_channels.get("phone_sms", {})
    phone_routes = set((phone_review.get("approval_routes_to_confirm") or {}).keys())
    if {"any_sms_send", "approved_phone_handoff_call", "customer_visible_handoff"} - phone_routes:
        issues.append("channel_policy_review:phone_sms:approval_routes_mismatch")
    blocked_capabilities = set(scope.get("blocked_capabilities") or [])
    if {"sms_send_without_approval", "voice_call"} - blocked_capabilities:
        issues.append("channel_policy:phone_sms_blocked_capabilities_missing")

    required_signoff_roles = {"business_owner", "channel_owner", "security_owner", "privacy_reviewer"}
    signoff_roles = {
        str(signoff.get("role"))
        for signoff in review.get("required_signoffs", [])
        if isinstance(signoff, Mapping) and signoff.get("required") is True
    }
    if signoff_roles != required_signoff_roles:
        issues.append("channel_policy_review:required_signoffs_mismatch")
    has_package_audit_review_command = False
    for command in review.get("review_commands", []):
        if isinstance(command, str) and "scripts/voiceops_plan_run.py" in command:
            if "--package-audit" in command:
                has_package_audit_review_command = True
            else:
                issues.append("channel_policy_review:plan_run_command_missing_package_audit")
    if not has_package_audit_review_command:
        issues.append("channel_policy_review:missing_package_audit_review_command")


def _require_markdown_tokens(label: str, markdown: str, tokens: Mapping[str, str], issues: list[str]) -> None:
    for issue, token in tokens.items():
        if token not in markdown:
            issues.append(f"{label}:{issue}")


def _reject_markdown_tokens(label: str, markdown: str, tokens: Mapping[str, str], issues: list[str]) -> None:
    for issue, token in tokens.items():
        if token in markdown:
            issues.append(f"{label}:{issue}")


def _audit_markdown_consistency(
    *,
    closure_markdown: str,
    operator_handoff_markdown: str,
    demo_handoff_markdown: str,
    channel_policy_markdown: str,
    channel_review_markdown: str,
    issues: list[str],
) -> None:
    _require_markdown_tokens(
        "closure_markdown",
        closure_markdown,
        {
            "missing_needs_external_evidence": "needs_external_evidence",
            "missing_artifact_only_safety": "artifact-only; no network I/O",
            "missing_final_package_audit_command": "Final package audit command",
            "missing_package_audit_status_signal": "package_audit.status is pass",
            "missing_package_audit_flag": "--package-audit",
        },
        issues,
    )
    _require_markdown_tokens(
        "operator_handoff_markdown",
        operator_handoff_markdown,
        {
            "missing_final_package_audit_command": "Final package audit command",
            "missing_package_audit_status_signal": "package_audit.status is pass",
            "missing_package_audit_flag": "--package-audit",
            "missing_secret_policy": "never paste secret values into artifacts",
        },
        issues,
    )
    _require_markdown_tokens(
        "demo_handoff_markdown",
        demo_handoff_markdown,
        {
            "missing_package_audit_section": "Package audit:",
            "missing_package_audit_status_signal": "package_audit.status is pass",
            "missing_package_audit_flag": "--package-audit",
            "missing_no_secret_policy": "never paste secret values into artifacts",
        },
        issues,
    )
    _require_markdown_tokens(
        "channel_policy_markdown",
        channel_policy_markdown,
        {
            "missing_artifact_only_safety": "artifact-only; no network, secret reads, sends, SMS, or calls",
            "missing_validation_pass": "Validation: pass",
            "missing_approval_routing": "## Approval Routing",
            "missing_customer_visible_route": "customer_visible_outbound",
            "missing_phone_handoff_route": "approved_phone_handoff_call",
            "missing_audit_id_continuity": "Never overwrite an existing audit_id",
            "missing_phone_redaction": "phone_number: `<redacted-phone>`",
        },
        issues,
    )
    _require_markdown_tokens(
        "channel_policy_review_markdown",
        channel_review_markdown,
        {
            "missing_pending_review": "Review status: pending_human_review",
            "missing_no_real_egress": "Real egress enabled: False",
            "missing_operator_must_not_send": "send Discord, WhatsApp, SMS, or phone traffic from this generated packet",
            "missing_package_audit_flag": "--package-audit",
            "missing_phone_handoff_route": "approved_phone_handoff_call",
        },
        issues,
    )
    _reject_markdown_tokens(
        "channel_policy_review_markdown",
        channel_review_markdown,
        {
            "contradicts_pending_review": "Review status: approved",
            "contradicts_no_real_egress": "Real egress enabled: True",
            "contradicts_no_live_egress": "Live egress enabled: True",
        },
        issues,
    )


def audit_package(artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    demo_dir = artifact_root / "hackathon-voiceops-demo" / "current"
    plan_dir = artifact_root / "voiceops-plan" / "current"
    channel_dir = artifact_root / "voiceops-channel-policy" / "current"

    demo = _read_json(demo_dir / "voiceops-demo.json", issues, "voiceops_demo")
    readiness = _read_json(demo_dir / "readiness-report.json", issues, "readiness_report")
    demo_closure = _read_json(demo_dir / "readiness-closure-summary.json", issues, "demo_closure")
    demo_handoff = _read_json(demo_dir / "operator-handoff-preview.json", issues, "demo_handoff")
    demo_handoff_markdown = _read_text(demo_dir / "operator-handoff-preview.md", issues, "demo_handoff_markdown")
    operator_state = _read_json(demo_dir / "operator-state.json", issues, "operator_state")
    packet = _read_json(demo_dir / "nemoclaw-action-packet.json", issues, "nemoclaw_packet")
    packet_validation = _read_json(
        demo_dir / "nemoclaw-action-packet.validation.json",
        issues,
        "nemoclaw_packet_validation",
    )
    plan_run = _read_json(plan_dir / "voiceops-plan-run.json", issues, "plan_run")
    plan_closure = _read_json(plan_dir / "readiness-closure-index.json", issues, "plan_closure")
    plan_closure_markdown = _read_text(plan_dir / "readiness-closure-index.md", issues, "plan_closure_markdown")
    plan_handoff = _read_json(plan_dir / "operator-handoff.json", issues, "operator_handoff")
    plan_handoff_markdown = _read_text(plan_dir / "operator-handoff.md", issues, "operator_handoff_markdown")
    channel_policy = _read_json(channel_dir / "channel-policy.json", issues, "channel_policy")
    channel_review = _read_json(channel_dir / "channel-policy-review.json", issues, "channel_policy_review")
    channel_policy_markdown = _read_text(channel_dir / "channel-policy.md", issues, "channel_policy_markdown")
    channel_review_markdown = _read_text(
        channel_dir / "channel-policy-review.md",
        issues,
        "channel_policy_review_markdown",
    )
    dashboard_html = _read_text(demo_dir / "operator-dashboard.html", issues, "operator_dashboard")
    audit_rows = _read_jsonl(demo_dir / "audit-ledger.jsonl", issues, "audit_ledger")
    dry_run_rows = _dry_run_metadata_rows(
        _read_text(demo_dir / "stripe-actions-dry-run.sh", issues, "stripe_actions"),
        issues,
    )

    _audit_static_readiness(
        demo=demo,
        readiness=readiness,
        demo_closure=demo_closure,
        plan_closure=plan_closure,
        operator_state=operator_state,
        dashboard_html=dashboard_html,
        issues=issues,
    )
    _audit_action_consistency(
        packet=packet,
        packet_validation=packet_validation,
        operator_state=operator_state,
        audit_rows=audit_rows,
        dry_run_rows=dry_run_rows,
        issues=issues,
    )
    _audit_service_claims(operator_state, issues)
    _audit_plan_consistency(
        demo_closure=demo_closure,
        demo_handoff=demo_handoff,
        plan_run=plan_run,
        plan_closure=plan_closure,
        plan_handoff=plan_handoff,
        issues=issues,
    )
    _audit_channel_policy(channel_policy, channel_review, issues)
    _audit_markdown_consistency(
        closure_markdown=plan_closure_markdown,
        operator_handoff_markdown=plan_handoff_markdown,
        demo_handoff_markdown=demo_handoff_markdown,
        channel_policy_markdown=channel_policy_markdown,
        channel_review_markdown=channel_review_markdown,
        issues=issues,
    )

    checked_artifacts = [
        str(demo_dir / "voiceops-demo.json"),
        str(demo_dir / "readiness-report.json"),
        str(demo_dir / "readiness-closure-summary.json"),
        str(demo_dir / "operator-handoff-preview.json"),
        str(demo_dir / "operator-handoff-preview.md"),
        str(demo_dir / "operator-state.json"),
        str(demo_dir / "operator-dashboard.html"),
        str(demo_dir / "nemoclaw-action-packet.json"),
        str(demo_dir / "nemoclaw-action-packet.validation.json"),
        str(demo_dir / "audit-ledger.jsonl"),
        str(demo_dir / "stripe-actions-dry-run.sh"),
        str(plan_dir / "voiceops-plan-run.json"),
        str(plan_dir / "readiness-closure-index.json"),
        str(plan_dir / "readiness-closure-index.md"),
        str(plan_dir / "operator-handoff.json"),
        str(plan_dir / "operator-handoff.md"),
        str(channel_dir / "channel-policy.json"),
        str(channel_dir / "channel-policy.md"),
        str(channel_dir / "channel-policy-review.json"),
        str(channel_dir / "channel-policy-review.md"),
    ]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "artifact_id": "voiceops-artifact-package-audit",
        "artifact_root": str(artifact_root),
        "mode": "local_static_package_audit_only",
        "safety": {
            "env_files_read": False,
            "secret_values_emitted": False,
            "network_io": False,
            "discord_io": False,
            "provider_provisioning": False,
            "live_spend": False,
            "outbound_messages": False,
            "outbound_calls": False,
            "spark_execution": False,
        },
        "ok": not issues,
        "status": "pass" if not issues else "fail",
        "issues": sorted(set(issues)),
        "warnings": warnings,
        "checked_artifacts": checked_artifacts,
        "checked_artifact_count": len(checked_artifacts),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# VoiceOps Artifact Package Audit",
        "",
        f"- Status: {report['status']}",
        f"- Artifact root: `{report['artifact_root']}`",
        f"- Checked artifacts: {report['checked_artifact_count']}",
        "- Network I/O: no",
        "- Provider provisioning: no",
        "- Live spend: no",
        "- Outbound messages/calls: no",
        "",
        "## Issues",
        "",
    ]
    issues = report.get("issues") or []
    if issues:
        lines.extend(f"- `{issue}`" for issue in issues)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_audit(output_dir: Path, report: Mapping[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "package-audit.json",
        "markdown": output_dir / "package-audit.md",
    }
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Print the audit report without writing package-audit artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_package(args.artifact_root)
    if args.audit_only:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        paths = write_audit(args.output_dir, report)
        print(json.dumps({"ok": report["ok"], "status": report["status"], "artifacts": paths}, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
