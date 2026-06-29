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
from pathlib import Path
from typing import Any, Mapping


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


def audit_package(artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    demo_dir = artifact_root / "hackathon-voiceops-demo" / "current"
    plan_dir = artifact_root / "voiceops-plan" / "current"

    demo = _read_json(demo_dir / "voiceops-demo.json", issues, "voiceops_demo")
    readiness = _read_json(demo_dir / "readiness-report.json", issues, "readiness_report")
    demo_closure = _read_json(demo_dir / "readiness-closure-summary.json", issues, "demo_closure")
    demo_handoff = _read_json(demo_dir / "operator-handoff-preview.json", issues, "demo_handoff")
    operator_state = _read_json(demo_dir / "operator-state.json", issues, "operator_state")
    packet = _read_json(demo_dir / "nemoclaw-action-packet.json", issues, "nemoclaw_packet")
    packet_validation = _read_json(
        demo_dir / "nemoclaw-action-packet.validation.json",
        issues,
        "nemoclaw_packet_validation",
    )
    plan_run = _read_json(plan_dir / "voiceops-plan-run.json", issues, "plan_run")
    plan_closure = _read_json(plan_dir / "readiness-closure-index.json", issues, "plan_closure")
    plan_handoff = _read_json(plan_dir / "operator-handoff.json", issues, "operator_handoff")
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

    checked_artifacts = [
        str(demo_dir / "voiceops-demo.json"),
        str(demo_dir / "readiness-report.json"),
        str(demo_dir / "readiness-closure-summary.json"),
        str(demo_dir / "operator-handoff-preview.json"),
        str(demo_dir / "operator-state.json"),
        str(demo_dir / "operator-dashboard.html"),
        str(demo_dir / "nemoclaw-action-packet.json"),
        str(demo_dir / "nemoclaw-action-packet.validation.json"),
        str(demo_dir / "audit-ledger.jsonl"),
        str(demo_dir / "stripe-actions-dry-run.sh"),
        str(plan_dir / "voiceops-plan-run.json"),
        str(plan_dir / "readiness-closure-index.json"),
        str(plan_dir / "operator-handoff.json"),
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
