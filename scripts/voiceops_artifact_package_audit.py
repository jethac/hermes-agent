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
import re
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.voiceops_channel_policy import CHANNEL_IDS, validate_policy


DEFAULT_ARTIFACT_ROOT = Path("artifacts")
DEFAULT_OUTPUT_DIR = Path("artifacts/voiceops-package-audit/current")
AUDIT_SCHEMA_VERSION = "voiceops.artifact_package_audit.v1"
EXPECTED_HANDOFF_PHASES = (
    (1, "live_discord_voice"),
    (2, "spend_and_provisioning_preflight"),
    (3, "local_spark_stack"),
)
LOCAL_MODEL_MARKERS = ("local", "dgx", "spark", "localhost", "127.0.0.1", "vllm")
HOSTED_MODEL_MARKERS = ("hosted", "cloud", "provider", "remote", "api", "nous")
SECRET_SCAN_PATTERNS = (
    ("openai_or_stripe_secret_key", re.compile(r"\bsk_(?:live|test|car)_[A-Za-z0-9_-]{12,}\b")),
    ("openai_project_key", re.compile(r"\bsk-proj-[A-Za-z0-9_-]{12,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{12,}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b")),
    ("github_fine_grained_token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")),
    ("sendgrid_api_key", re.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b")),
    ("discord_bot_token", re.compile(r"\b[MNO][A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}\b")),
    ("e164_phone_number", re.compile(r"(?<![A-Za-z0-9_])\+[1-9][0-9]{9,14}\b")),
)
EXPECTED_PACKAGE_ARTIFACTS = (
    "hackathon-voiceops-demo/current/audit-ledger.jsonl",
    "hackathon-voiceops-demo/current/demo-script.md",
    "hackathon-voiceops-demo/current/milestone2-execution-plan.json",
    "hackathon-voiceops-demo/current/nemoclaw-action-packet.json",
    "hackathon-voiceops-demo/current/nemoclaw-action-packet.validation.json",
    "hackathon-voiceops-demo/current/operator-dashboard.html",
    "hackathon-voiceops-demo/current/operator-handoff-preview.json",
    "hackathon-voiceops-demo/current/operator-handoff-preview.md",
    "hackathon-voiceops-demo/current/operator-state-events.jsonl",
    "hackathon-voiceops-demo/current/operator-state.json",
    "hackathon-voiceops-demo/current/phone-context.json",
    "hackathon-voiceops-demo/current/readiness-closure-summary.json",
    "hackathon-voiceops-demo/current/readiness-closure-summary.md",
    "hackathon-voiceops-demo/current/readiness-report.json",
    "hackathon-voiceops-demo/current/readiness-report.md",
    "hackathon-voiceops-demo/current/recording-runbook.md",
    "hackathon-voiceops-demo/current/stripe-actions-dry-run.sh",
    "hackathon-voiceops-demo/current/submission-writeup.md",
    "hackathon-voiceops-demo/current/voiceops-demo.json",
    "hackathon-voiceops-demo/current/voiceops-demo.md",
    "voiceops-channel-policy/current/channel-policy-review.json",
    "voiceops-channel-policy/current/channel-policy-review.md",
    "voiceops-channel-policy/current/channel-policy.json",
    "voiceops-channel-policy/current/channel-policy.md",
    "voiceops-operator-state/current/operator-state-events.jsonl",
    "voiceops-operator-state/current/operator-state.json",
    "voiceops-operator-state/current/operator-state.md",
    "voiceops-plan/current/operator-handoff.json",
    "voiceops-plan/current/operator-handoff.md",
    "voiceops-plan/current/readiness-closure-index.json",
    "voiceops-plan/current/readiness-closure-index.md",
    "voiceops-plan/current/voiceops-plan-run.json",
    "voiceops-plan/current/voiceops-plan-run.md",
    "voiceops-provisioning/current/audit-ledger.post-approval.jsonl",
    "voiceops-provisioning/current/audit-ledger.read-only-discovery.jsonl",
    "voiceops-provisioning/current/milestone2-execution-plan.json",
    "voiceops-provisioning/current/milestone2-execution-plan.md",
    "voiceops-provisioning/current/nemoclaw-action-packet.validation.json",
    "voiceops-provisioning/current/post-approval-receipts-scaffold/post-approval-receipts.json",
    "voiceops-provisioning/current/post-approval-receipts.example.json",
    "voiceops-provisioning/current/post-approval-receipts.template.json",
    "voiceops-provisioning/current/post-approval-receipts.validation.json",
    "voiceops-provisioning/current/provisioning-preflight-evidence.example.json",
    "voiceops-provisioning/current/provisioning-preflight-evidence.manifest.example.json",
    "voiceops-provisioning/current/provisioning-preflight-evidence.template.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sections/nemoclaw-boundary-evidence.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sections/phone-handoff-evidence.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sections/rollback-owner-evidence.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sections/stripe-link-evidence.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sections/stripe-projects-evidence.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sources/nemoclaw-boundary-redacted-source.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sources/phone-handoff-redacted-source.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sources/rollback-owners-redacted-source.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sources/stripe-link-redacted-source.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sources/stripe-projects-redacted-source.json",
    "voiceops-provisioning/current/provisioning-readiness.json",
    "voiceops-provisioning/current/provisioning-readiness.md",
    "voiceops-provisioning/current/read-only-discovery.json",
    "voiceops-provisioning/current/read-only-discovery.manifest.json",
    "voiceops-provisioning/current/read-only-discovery.md",
    "voiceops-provisioning/current/safe-command-manifest.json",
    "voiceops-provisioning/current/setup-closure-plan.json",
    "voiceops-provisioning/current/setup-closure-plan.md",
    "voiceops-spark-matrix/current/spark-benchmark-evidence-template.json",
    "voiceops-spark-matrix/current/spark-benchmark-evidence.example.json",
    "voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/all-local-stack-smoke-raw.json",
    "voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/asr-nemotron-speech-raw.json",
    "voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/oracle-nemotron3-super-raw.json",
    "voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/reflex-gemma4-e2b-raw.json",
    "voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/reflex-gemma4-e4b-raw.json",
    "voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/tts-magpie-local-raw.json",
    "voiceops-spark-matrix/current/spark-benchmark-scaffold/spark-benchmark-evidence.json",
    "voiceops-spark-matrix/current/spark-matrix-closure-plan.json",
    "voiceops-spark-matrix/current/spark-matrix-closure-plan.md",
    "voiceops-spark-matrix/current/spark-model-matrix.json",
    "voiceops-spark-matrix/current/spark-model-matrix.md",
    "voiceops-spark-matrix/current/spark-operator-runbook.md",
    "voiceops-voice-operator/current/discord-loopback-smoke.json",
    "voiceops-voice-operator/current/live-probe-closure-plan.json",
    "voiceops-voice-operator/current/live-probe-closure-plan.md",
    "voiceops-voice-operator/current/live-voice-evidence-scaffold/manifest.json",
    "voiceops-voice-operator/current/live-voice-evidence-scaffold/sections/discord-live-probe.json",
    "voiceops-voice-operator/current/live-voice-evidence-scaffold/sections/live-turn.json",
    "voiceops-voice-operator/current/live-voice-evidence-scaffold/sections/sidecar-session.json",
    "voiceops-voice-operator/current/live-voice-evidence-template.json",
    "voiceops-voice-operator/current/live-voice-evidence.example.json",
    "voiceops-voice-operator/current/voice-operator-events.jsonl",
    "voiceops-voice-operator/current/voice-operator-readiness.json",
    "voiceops-voice-operator/current/voice-operator-readiness.md",
)
AUDITED_PACKAGE_DIRS = tuple(
    sorted({"/".join(relative_path.split("/")[:2]) for relative_path in EXPECTED_PACKAGE_ARTIFACTS})
)


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


def _audit_expected_package_artifacts(artifact_root: Path, issues: list[str]) -> list[str]:
    checked_artifacts: list[str] = []
    for relative_path in EXPECTED_PACKAGE_ARTIFACTS:
        path = artifact_root / relative_path
        label = f"package_artifact:{relative_path}"
        if path.suffix == ".json":
            _read_json(path, issues, label)
        elif path.suffix == ".jsonl":
            _read_jsonl(path, issues, label)
        else:
            text = _read_text(path, issues, label)
            if not text.strip():
                issues.append(f"{label}:empty")
        checked_artifacts.append(str(path))
    return checked_artifacts


def _audit_no_unexpected_package_artifacts(
    artifact_root: Path,
    checked_artifacts: list[str],
    issues: list[str],
) -> None:
    expected_paths = {Path(path).resolve(strict=False) for path in checked_artifacts}
    for relative_dir in AUDITED_PACKAGE_DIRS:
        directory = artifact_root / relative_dir
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            if path.resolve(strict=False) not in expected_paths:
                issues.append(f"package_artifact:unexpected:{_artifact_label(artifact_root, path)}")


def _audit_no_secret_like_values(artifact_root: Path, checked_artifacts: list[str], issues: list[str]) -> None:
    for path_text in checked_artifacts:
        path = Path(path_text)
        label = _artifact_label(artifact_root, path)
        try:
            text = path.read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for rule_id, pattern in SECRET_SCAN_PATTERNS:
                if pattern.search(line):
                    issues.append(f"secret_scan:{label}:line_{line_number}:{rule_id}")


def _artifact_label(artifact_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(artifact_root))
    except ValueError:
        return str(path)


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
    spark_matrix: Mapping[str, Any],
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
        sponsor_stack = demo.get("sponsor_stack") if isinstance(demo.get("sponsor_stack"), Mapping) else {}
        active_path = (
            sponsor_stack.get("hermes_active_model")
            if isinstance(sponsor_stack.get("hermes_active_model"), Mapping)
            else {}
        )
        spark_boundary = (
            "Spark target selected, live evidence pending"
            if active_path.get("spark_local") is True
            else "Hosted fallback selected, Spark-local evidence pending"
        )
        rejected_spark_boundary = (
            "Hosted fallback selected, Spark-local evidence pending"
            if active_path.get("spark_local") is True
            else "Spark target selected, live evidence pending"
        )
        required_dashboard_tokens = [
            "static dry-run package",
            "Static package ready",
            "Live/Spark gaps",
            spark_boundary,
            "scripted_static_ack_until_live_voice_evidence",
            "needs_live_probe",
            "needs_setup",
            "needs_evidence",
        ]
        for token in required_dashboard_tokens:
            if token not in dashboard_html:
                issues.append(f"dashboard:missing_non_live_token:{token}")
        if rejected_spark_boundary in dashboard_html:
            issues.append("dashboard:contradicts_active_model_path")

    demo_gate_ids = {str(gate.get("gate_id")) for gate in demo_closure.get("gates", []) if isinstance(gate, Mapping)}
    plan_gate_ids = {str(gate.get("gate_id")) for gate in plan_closure.get("gates", []) if isinstance(gate, Mapping)}
    if demo_gate_ids != plan_gate_ids:
        issues.append("closure:gates_mismatch_between_demo_and_plan")
    _audit_spark_model_claims(demo=demo, readiness=readiness, spark_matrix=spark_matrix, issues=issues)


def _audit_spark_model_claims(
    *,
    demo: Mapping[str, Any],
    readiness: Mapping[str, Any],
    spark_matrix: Mapping[str, Any],
    issues: list[str],
) -> None:
    recording_readiness = demo.get("recording_readiness") if isinstance(demo.get("recording_readiness"), Mapping) else {}
    sponsor_stack = demo.get("sponsor_stack") if isinstance(demo.get("sponsor_stack"), Mapping) else {}
    active_path = (
        sponsor_stack.get("hermes_active_model")
        if isinstance(sponsor_stack.get("hermes_active_model"), Mapping)
        else {}
    )
    active_model = str(active_path.get("active_model") or "").lower()
    spark_local = active_path.get("spark_local")
    hosted_marker_present = any(marker in active_model for marker in HOSTED_MODEL_MARKERS)
    local_marker_present = any(marker in active_model for marker in LOCAL_MODEL_MARKERS)
    if spark_local is True and hosted_marker_present:
        issues.append("spark_model_claim:spark_local_true_for_hosted_model")
    if spark_local is True and not local_marker_present:
        issues.append("spark_model_claim:spark_local_true_without_local_marker")
    if active_path.get("fallback_used") is True and spark_local is True:
        issues.append("spark_model_claim:fallback_used_but_spark_local_true")
    expected_status = (
        "target_selected_needs_benchmark_evidence"
        if spark_local is True
        else "hosted_or_nonlocal_path_not_spark_evidence"
    )
    if readiness.get("spark_local_evidence_status") != expected_status:
        issues.append("spark_model_claim:readiness_status_mismatch")
    for key in ("spark_local_readiness", "spark_benchmark_required", "spark_readiness_source"):
        if recording_readiness.get(key) != readiness.get(key):
            issues.append(f"spark_model_claim:demo_{key}_mismatch")
    if not spark_matrix:
        return
    matrix_ready = spark_matrix.get("ready_for_one_spark_demo") is True
    if readiness.get("spark_local_readiness") is not matrix_ready:
        issues.append("spark_model_claim:spark_local_readiness_mismatch")
    expected_benchmark_required = not matrix_ready
    if readiness.get("spark_benchmark_required") is not expected_benchmark_required:
        issues.append("spark_model_claim:spark_benchmark_required_mismatch")
    if readiness.get("spark_readiness_source") != "voiceops_spark_matrix.ready_for_one_spark_demo":
        issues.append("spark_model_claim:readiness_source_mismatch")
    missing_evidence = readiness.get("live_demo_missing_evidence") or []
    if matrix_ready is False and "local_spark_stack_matrix" not in missing_evidence:
        issues.append("spark_model_claim:missing_m4_live_evidence_gap")


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
    if plan_run.get("artifact_id") != "voiceops-plan-run":
        issues.append("plan_run:artifact_id_mismatch")
    if plan_run.get("artifact_only") is not True:
        issues.append("plan_run:artifact_only_not_true")
    if plan_closure.get("artifact_only") is not True:
        issues.append("plan_closure:artifact_only_not_true")
    if plan_run.get("ok") is not True:
        issues.append("plan_run:ok_not_true")
    if plan_run.get("hard_failures") != []:
        issues.append("plan_run:hard_failures_not_empty")
    if plan_run.get("closure_index") != plan_closure:
        issues.append("plan_run:closure_index_mismatch")
    if plan_run.get("closure_status") != plan_closure.get("closure_status"):
        issues.append("plan_run:closure_status_mismatch")
    if plan_run.get("readiness_gaps") != plan_closure.get("readiness_gaps"):
        issues.append("plan_run:readiness_gaps_mismatch")
    expected_remaining_gate_ids = [
        str(gate.get("gate_id"))
        for gate in plan_closure.get("remaining_gates", [])
        if isinstance(gate, Mapping)
    ]
    if plan_run.get("remaining_gates") != expected_remaining_gate_ids:
        issues.append("plan_run:remaining_gates_mismatch")
    if plan_run.get("next_actions") != plan_closure.get("next_actions"):
        issues.append("plan_run:next_actions_mismatch")
    _audit_plan_safety("plan_run", plan_run.get("safety"), issues)
    _audit_plan_safety("plan_closure", plan_closure.get("safety"), issues)
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
    _audit_handoff_phase_contract("operator_handoff", plan_handoff, issues)
    _audit_handoff_phase_contract("demo_handoff", demo_handoff, issues)
    demo_phases = _handoff_phases_by_id(demo_handoff)
    plan_phases = _handoff_phases_by_id(plan_handoff)
    if set(demo_phases) != set(plan_phases):
        issues.append("demo_handoff:phase_ids_mismatch")
        return
    for phase_id, plan_phase in plan_phases.items():
        demo_phase = demo_phases[phase_id]
        for key in ("order", "commands", "expected_artifacts", "success_check"):
            if demo_phase.get(key) != plan_phase.get(key):
                issues.append(f"demo_handoff:{phase_id}:{key}_mismatch")


def _audit_plan_safety(label: str, safety: Any, issues: list[str]) -> None:
    if not isinstance(safety, Mapping):
        issues.append(f"{label}:safety_missing")
        return
    for key in (
        "env_secret_values_emitted",
        "mutating_network_io",
        "live_spend",
        "provider_provisioning",
        "outbound_calls",
    ):
        if safety.get(key) is not False:
            issues.append(f"{label}:safety_{key}_not_false")
    if safety.get("outbound_sends", safety.get("outbound_messages", False)) is not False:
        issues.append(f"{label}:safety_outbound_sends_not_false")
    if safety.get("read_only_discovery_grants_approval") is not False:
        issues.append(f"{label}:safety_read_only_discovery_grants_approval_not_false")
    network_io = safety.get("network_io")
    network_scope = safety.get("network_io_scope")
    if network_io not in {False, True}:
        issues.append(f"{label}:safety_network_io_not_boolean")
    elif network_io is True and network_scope != "allowlisted_read_only_discovery":
        issues.append(f"{label}:safety_network_io_scope_invalid")
    elif network_io is False and network_scope != "none":
        issues.append(f"{label}:safety_network_io_scope_invalid")
    if label == "plan_closure" and safety.get("spark_execution") is not False:
        issues.append(f"{label}:safety_spark_execution_not_false")


def _handoff_phases_by_id(handoff: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    phases = handoff.get("phases")
    if not isinstance(phases, list):
        return {}
    return {
        str(phase.get("phase_id")): phase
        for phase in phases
        if isinstance(phase, Mapping) and str(phase.get("phase_id") or "").strip()
    }


def _audit_handoff_phase_contract(label: str, handoff: Mapping[str, Any], issues: list[str]) -> None:
    phases = handoff.get("phases")
    if not isinstance(phases, list):
        issues.append(f"{label}:phases_missing")
        return
    observed = [
        (phase.get("order"), phase.get("phase_id"))
        for phase in phases
        if isinstance(phase, Mapping)
    ]
    if observed != list(EXPECTED_HANDOFF_PHASES):
        issues.append(f"{label}:phase_order_mismatch")
    for expected_order, expected_phase_id in EXPECTED_HANDOFF_PHASES:
        phase = next(
            (
                item
                for item in phases
                if isinstance(item, Mapping)
                and item.get("order") == expected_order
                and item.get("phase_id") == expected_phase_id
            ),
            None,
        )
        if not isinstance(phase, Mapping):
            issues.append(f"{label}:{expected_phase_id}:phase_missing")
            continue
        if not isinstance(phase.get("blocked_by_current_environment"), Mapping):
            issues.append(f"{label}:{expected_phase_id}:missing_environment_blockers")


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
    spark_local_target_selected: bool,
    demo_markdown: str,
    recording_runbook_markdown: str,
    submission_writeup_markdown: str,
    closure_markdown: str,
    operator_handoff_markdown: str,
    demo_handoff_markdown: str,
    channel_policy_markdown: str,
    channel_review_markdown: str,
    issues: list[str],
) -> None:
    spark_boundary = (
        "Spark target selected, live evidence pending"
        if spark_local_target_selected
        else "Hosted fallback selected, Spark-local evidence pending"
    )
    rejected_spark_boundary = (
        "Hosted fallback selected, Spark-local evidence pending"
        if spark_local_target_selected
        else "Spark target selected, live evidence pending"
    )
    _require_markdown_tokens(
        "demo_markdown",
        demo_markdown,
        {
            "missing_static_dry_run_status": "static dry-run package",
            "missing_spark_evidence_boundary": spark_boundary,
            "missing_approval_gate": "spend/provisioning gated by approval",
        },
        issues,
    )
    _require_markdown_tokens(
        "recording_runbook_markdown",
        recording_runbook_markdown,
        {
            "missing_static_dry_run_status": "static dry-run VoiceOps package",
            "missing_spark_evidence_boundary": spark_boundary,
            "missing_secret_policy": "Do not show terminal panes or files that contain secrets",
        },
        issues,
    )
    _require_markdown_tokens(
        "submission_writeup_markdown",
        submission_writeup_markdown,
        {
            "missing_static_dry_run_status": "static dry-run package",
            "missing_spark_evidence_boundary": spark_boundary,
            "missing_spend_gate": "Spend gated by approval",
        },
        issues,
    )
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
        "spark_public_copy",
        "\n".join([demo_markdown, recording_runbook_markdown, submission_writeup_markdown, demo_handoff_markdown]),
        {
            "contradicts_active_model_path": rejected_spark_boundary,
            "claims_running_spark_appliance_without_evidence": "target appliance is one DGX Spark running",
            "claims_spark_powered_operator_without_evidence": "Spark-powered Hermes operator",
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
    checked_artifacts = _audit_expected_package_artifacts(artifact_root, issues)
    _audit_no_unexpected_package_artifacts(artifact_root, checked_artifacts, issues)
    _audit_no_secret_like_values(artifact_root, checked_artifacts, issues)
    demo_dir = artifact_root / "hackathon-voiceops-demo" / "current"
    plan_dir = artifact_root / "voiceops-plan" / "current"
    channel_dir = artifact_root / "voiceops-channel-policy" / "current"
    spark_dir = artifact_root / "voiceops-spark-matrix" / "current"

    demo = _read_json(demo_dir / "voiceops-demo.json", issues, "voiceops_demo")
    demo_markdown = _read_text(demo_dir / "voiceops-demo.md", issues, "voiceops_demo_markdown")
    readiness = _read_json(demo_dir / "readiness-report.json", issues, "readiness_report")
    demo_closure = _read_json(demo_dir / "readiness-closure-summary.json", issues, "demo_closure")
    demo_handoff = _read_json(demo_dir / "operator-handoff-preview.json", issues, "demo_handoff")
    demo_handoff_markdown = _read_text(demo_dir / "operator-handoff-preview.md", issues, "demo_handoff_markdown")
    recording_runbook_markdown = _read_text(demo_dir / "recording-runbook.md", issues, "recording_runbook_markdown")
    submission_writeup_markdown = _read_text(demo_dir / "submission-writeup.md", issues, "submission_writeup_markdown")
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
    spark_matrix = _read_json(spark_dir / "spark-model-matrix.json", issues, "spark_matrix")
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
    sponsor_stack = demo.get("sponsor_stack") if isinstance(demo.get("sponsor_stack"), Mapping) else {}
    active_path = (
        sponsor_stack.get("hermes_active_model")
        if isinstance(sponsor_stack.get("hermes_active_model"), Mapping)
        else {}
    )
    spark_local_target_selected = active_path.get("spark_local") is True

    _audit_static_readiness(
        demo=demo,
        readiness=readiness,
        spark_matrix=spark_matrix,
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
        spark_local_target_selected=spark_local_target_selected,
        demo_markdown=demo_markdown,
        recording_runbook_markdown=recording_runbook_markdown,
        submission_writeup_markdown=submission_writeup_markdown,
        closure_markdown=plan_closure_markdown,
        operator_handoff_markdown=plan_handoff_markdown,
        demo_handoff_markdown=demo_handoff_markdown,
        channel_policy_markdown=channel_policy_markdown,
        channel_review_markdown=channel_review_markdown,
        issues=issues,
    )

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
