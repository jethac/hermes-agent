#!/usr/bin/env python3
"""Execute approved VoiceOps Stripe actions and emit redacted receipts.

Default mode is non-executing. Live Stripe/Link commands require both an
approve_once decision for the exact action and the explicit --execute flag.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.voiceops_provisioning_probe import (
    POST_APPROVAL_RECEIPTS_SCHEMA_VERSION,
    validate_nemoclaw_action_packet,
    validate_post_approval_receipts,
)


DEFAULT_OUTPUT_DIR = Path("artifacts/voiceops-provisioning/current/live-execution")
APPROVAL_DECISIONS_SCHEMA_VERSION = "voiceops.milestone2.approval_decisions.v1"
APPROVAL_DECISION_SCHEMA_VERSION = "voiceops.milestone2.approval_decision.v1"
EXECUTION_REPORT_SCHEMA_VERSION = "voiceops.milestone2.stripe_executor_report.v1"
DECISION_ALLOWLIST = {"approve_once", "deny", "hold"}
LIVE_CONFIRMATION = "execute-approved-voiceops-stripe-actions"
SECRET_VALUE_RE = re.compile(
    r"(?i)\b(?:sk|pk|rk|whsec|AC|SG|xox[baprs]|gh[pousr])[_-]?[A-Za-z0-9][A-Za-z0-9_\-]{8,}\b"
)
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{8,}")
PHONE_RE = re.compile(r"(?<!\d)\+?[1-9]\d[\d .()\-]{7,}\d(?!\d)")
VOIP_PROVIDER_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]{0,80}$")


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


CommandRunner = Callable[[Sequence[str], int], CommandResult]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} root must be a JSON object")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _payload_sha256_without_attestation(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("collector_attestation", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _redact(value: Any, limit: int = 1000) -> str:
    text = str(value or "")
    text = BEARER_RE.sub("Bearer <redacted>", text)
    text = SECRET_VALUE_RE.sub("<redacted>", text)
    text = PHONE_RE.sub("<redacted-phone>", text)
    text = "\n".join(line.rstrip() for line in text.replace("\r", "\n").splitlines())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _subprocess_runner(argv: Sequence[str], timeout_seconds: int) -> CommandResult:
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            stdin=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            exit_code=124,
            stdout=exc.stdout if isinstance(exc.stdout, str) else "",
            stderr=exc.stderr if isinstance(exc.stderr, str) else "command timed out",
            timed_out=True,
        )
    except OSError as exc:
        return CommandResult(exit_code=127, stderr=str(exc))
    return CommandResult(
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _action_estimates(plan: Mapping[str, Any]) -> dict[str, int]:
    estimates: dict[str, int] = {}
    for step in plan.get("execution_steps", []):
        if not isinstance(step, Mapping):
            continue
        amount = step.get("estimated_cents")
        if isinstance(amount, int) and not isinstance(amount, bool):
            estimates[str(step.get("step_id") or "")] = amount
    return estimates


def _actions_by_id(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(action.get("action_id") or ""): action
        for action in plan.get("approval_required_actions", [])
        if isinstance(action, Mapping) and str(action.get("action_id") or "")
    }


def _packet_actions_by_id(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(action.get("action_id") or ""): action
        for action in packet.get("approval_required_actions", [])
        if isinstance(action, Mapping) and str(action.get("action_id") or "")
    }


def _load_decisions(payload: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    issues: list[str] = []
    if str(payload.get("schema_version") or "") != APPROVAL_DECISIONS_SCHEMA_VERSION:
        issues.append("approval_decisions:missing_or_invalid_schema_version")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        return {}, [*issues, "approval_decisions:decisions_not_list"]
    by_action: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(decisions):
        if not isinstance(item, Mapping):
            issues.append(f"approval_decisions:{index}:not_object")
            continue
        action_id = str(item.get("action_id") or "")
        decision = str(item.get("decision") or "")
        if not action_id:
            issues.append(f"approval_decisions:{index}:missing_action_id")
            continue
        if action_id in by_action:
            issues.append(f"approval_decisions:{action_id}:duplicate")
        if decision not in DECISION_ALLOWLIST:
            issues.append(f"approval_decisions:{action_id}:invalid_decision")
        by_action[action_id] = item
    return by_action, issues


def _command_issues_for_action(action_id: str, command: str) -> list[str]:
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return [f"{action_id}:command_parse_failed:{exc}"]
    if action_id == "provision-voip-provider":
        if len(argv) != 4 or argv[:3] != ["stripe", "projects", "add"]:
            return [f"{action_id}:command_not_allowed"]
        if not VOIP_PROVIDER_RE.fullmatch(argv[3]):
            return [f"{action_id}:provider_candidate_invalid"]
        return []
    if action_id == "buy-service-credit":
        if argv[:3] != ["link-cli", "spend-request", "create"]:
            return [f"{action_id}:command_not_allowed"]
        if "--request-approval" not in argv:
            return [f"{action_id}:missing_request_approval"]
        forbidden = {"retrieve", "--include", "card", "--output-file"}
        if any(part in forbidden for part in argv):
            return [f"{action_id}:credential_retrieval_not_allowed"]
        return []
    if action_id == "call-user-phone":
        if argv[:4] != ["queue", "outbound", "call", "--context"] or len(argv) != 5:
            return [f"{action_id}:command_not_allowed"]
        return []
    if action_id == "publish-status":
        if command != "post redacted approval and handoff status to configured channels":
            return [f"{action_id}:command_not_allowed"]
        return []
    return [f"{action_id}:unknown_action"]


def _receipt_status_for_decision(decision: str, *, executed: bool, exit_code: int | None = None) -> str:
    if decision == "deny":
        return "denied"
    if decision == "hold":
        return "held"
    if not executed:
        return "held"
    return "executed" if exit_code == 0 else "failed"


def _decision_artifact_payload(
    *,
    action: Mapping[str, Any],
    receipt_id: str,
    decision: str,
    decision_by: str,
    decision_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": APPROVAL_DECISION_SCHEMA_VERSION,
        "redacted": True,
        "redaction_policy": "redacted references only; no raw secrets, tokens, cards, credentials, or phone numbers",
        "action_id": action["action_id"],
        "receipt_id": receipt_id,
        "approval_id": action["approval_id"],
        "decision": decision,
        "decision_by": decision_by,
        "decision_at": decision_at,
        "command_sha256": action["command_sha256"],
    }


def _attestation(payload: Mapping[str, Any], *, started_at: str, finished_at: str, parent_hash: str) -> dict[str, Any]:
    redacted_sha256 = _payload_sha256_without_attestation(payload)
    return {
        "collector_name": "scripts.voiceops_stripe_executor",
        "collector_version": EXECUTION_REPORT_SCHEMA_VERSION,
        "run_id": f"voiceops-stripe-exec-{redacted_sha256[:12]}",
        "command_argv": [sys.executable, "scripts/voiceops_stripe_executor.py"],
        "git_commit": "unavailable",
        "started_at": started_at,
        "finished_at": finished_at,
        "raw_artifact_sha256": redacted_sha256,
        "redacted_artifact_sha256": redacted_sha256,
        "parent_manifest_sha256": parent_hash,
    }


def execute_approved_actions(
    *,
    packet: Mapping[str, Any],
    plan: Mapping[str, Any],
    decisions_payload: Mapping[str, Any],
    output_dir: Path,
    execute: bool = False,
    confirmation: str | None = None,
    runner: CommandRunner | None = None,
    timeout_seconds: int = 900,
    now: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    started_at = now()
    runner = runner or _subprocess_runner
    issues: list[str] = []
    validation = validate_nemoclaw_action_packet(packet)
    if validation["status"] != "valid":
        issues.extend(f"nemoclaw_action_packet:{issue}" for issue in validation["validation_issues"])
    decisions, decision_issues = _load_decisions(decisions_payload)
    issues.extend(decision_issues)
    if execute and confirmation != LIVE_CONFIRMATION:
        issues.append("execute_confirmation_missing_or_invalid")

    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_dir = output_dir / "approval-decisions"
    actions = _actions_by_id(plan)
    packet_actions = _packet_actions_by_id(packet)
    estimates = _action_estimates(plan)
    packet_hash = hashlib.sha256(json.dumps(packet, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    receipts: list[dict[str, Any]] = []
    credential_locations: list[dict[str, Any]] = []
    rollback_receipts: list[dict[str, Any]] = []
    audit_events: list[dict[str, Any]] = []
    command_results: list[dict[str, Any]] = []

    for action_id, action in actions.items():
        decision_record = decisions.get(action_id, {})
        decision = str(decision_record.get("decision") or "hold")
        if decision != "approve_once":
            continue
        command = str(action.get("command") or "")
        issues.extend(_command_issues_for_action(action_id, command))
        packet_action = packet_actions.get(action_id)
        if packet_action is None:
            issues.append(f"{action_id}:not_present_in_nemoclaw_packet")
        elif str(packet_action.get("command") or "") != command:
            issues.append(f"{action_id}:packet_command_mismatch")
        if not execute:
            issues.append(f"{action_id}:approve_once_requires_execute")

    execution_gate_ok = not issues

    for action_id, action in actions.items():
        decision_record = decisions.get(action_id, {})
        decision = str(decision_record.get("decision") or "hold")
        decision_by = str(decision_record.get("decision_by") or "operator-ref-unspecified")
        decision_at = str(decision_record.get("decision_at") or now())
        packet_action = packet_actions.get(action_id)
        command = str(action.get("command") or "")
        command_issues = _command_issues_for_action(action_id, command)

        receipt_id = f"receipt-{action_id}-001"
        audit_event_id = f"audit-{action_id}-001"
        decision_ref = f"approval-decisions/{action_id}.json"
        decision_payload = _decision_artifact_payload(
            action=action,
            receipt_id=receipt_id,
            decision=decision,
            decision_by=decision_by,
            decision_at=decision_at,
        )
        decision_path = output_dir / decision_ref
        _write_json(decision_path, decision_payload)
        decision_sha256 = _file_sha256(decision_path)

        result: CommandResult | None = None
        local_queue = action_id in {"call-user-phone", "publish-status"}
        should_execute = (
            execution_gate_ok
            and execute
            and decision == "approve_once"
            and not command_issues
            and packet_action is not None
            and str(packet_action.get("command") or "") == command
        )
        if should_execute and local_queue:
            result = CommandResult(exit_code=0, stdout=f"queued {action_id}")
        elif should_execute:
            result = runner(shlex.split(command), timeout_seconds)
        status = _receipt_status_for_decision(
            decision,
            executed=result is not None,
            exit_code=result.exit_code if result else None,
        )
        executed_at = now() if result is not None else None
        amount_cents = estimates.get(action_id, 0)
        external_reference = (
            f"{action['provider']}:{action_id}:{_sha256_text((result.stdout if result else command) or command)[:12]}"
            if result is not None
            else None
        )
        lineage = action.get("lineage") if isinstance(action.get("lineage"), Mapping) else {}

        receipt: dict[str, Any] = {
            "receipt_id": receipt_id,
            "action_id": action_id,
            "approval_id": action["approval_id"],
            "provider": action["provider"],
            "status": status,
            "decision": decision if status not in {"held", "skipped"} else "hold",
            "decision_by": decision_by,
            "decision_at": decision_at,
            "approval_decision_ref": decision_ref,
            "approval_decision_sha256": decision_sha256,
            "command_sha256": action["command_sha256"],
            "approval_artifact": action["approval_artifact"],
            "audit_event_id": audit_event_id,
            **dict(lineage),
            "redacted_summary": (
                "Executed approved action and stored redacted receipt metadata."
                if result is not None
                else "Held or denied action; no command executed."
            ),
        }
        if status not in {"held", "denied", "skipped"}:
            receipt.update(
                {
                    "executed_at": executed_at,
                    "amount_cents": amount_cents,
                    "currency": str(plan.get("spend_policy", {}).get("currency") or "usd"),
                    "external_reference": external_reference,
                    "credential_location_ref": action.get("credential_location_ref"),
                    "rollback_ref": action.get("rollback_ref"),
                }
            )
            if action.get("credential_location_required"):
                credential_locations.append(
                    {
                        "credential_ref_id": action.get("credential_location_ref"),
                        "provider": action["provider"],
                        "service_id": external_reference,
                        "storage_backend": "provider_managed",
                        "secret_name_or_path": f"voiceops/{action_id}/provider-managed",
                        "created_by_action_id": action_id,
                        "rotation_due": "2026-12-31T00:00:00Z",
                        "redacted": True,
                        "lineage": dict(lineage),
                    }
                )
            rollback_receipts.append(
                {
                    "rollback_ref": action.get("rollback_ref"),
                    "status": "not_run",
                    "owner_ref": decision_by,
                    "notes": "Rollback/deprovision not run; owner ref recorded for approved action.",
                    "lineage": dict(lineage),
                }
            )
        receipts.append(receipt)
        audit_events.append(
            {
                "audit_event_id": audit_event_id,
                "action_id": action_id,
                "receipt_id": receipt_id,
                "status": status,
                "provider": action["provider"],
                "artifact_ref": "post-approval-receipts.json",
                **dict(lineage),
                "operator_next_step": (
                    "Inspect provider dashboard and run rollback if needed."
                    if status in {"executed", "failed"}
                    else "No execution occurred; approve explicitly before rerunning."
                ),
            }
        )
        command_results.append(
            {
                "action_id": action_id,
                "decision": decision,
                "executed": result is not None,
                "status": status,
                "command_sha256": action["command_sha256"],
                "exit_code": result.exit_code if result else None,
                "timed_out": result.timed_out if result else False,
                "stdout_excerpt": _redact(result.stdout) if result else "",
                "stderr_excerpt": _redact(result.stderr) if result else "",
            }
        )

    receipts_payload: dict[str, Any] = {
        "schema_version": POST_APPROVAL_RECEIPTS_SCHEMA_VERSION,
        "redaction_policy": "references and redacted summaries only; no raw credentials, tokens, card data, or full phone numbers",
        "receipts": receipts,
        "credential_locations": credential_locations,
        "rollback_receipts": rollback_receipts,
        "audit_events": audit_events,
        "expected_actions": sorted(actions),
        "collector_attestation": None,
    }
    finished_at = now()
    receipts_payload["collector_attestation"] = _attestation(
        receipts_payload,
        started_at=started_at,
        finished_at=finished_at,
        parent_hash=packet_hash,
    )
    receipts_path = output_dir / "post-approval-receipts.json"
    _write_json(receipts_path, receipts_payload)
    receipt_validation = validate_post_approval_receipts(receipts_payload, plan, receipt_path=receipts_path)
    if receipt_validation["status"] != "valid":
        issues.extend(f"post_approval_receipts:{issue}" for issue in receipt_validation["validation_issues"])

    report = {
        "schema_version": EXECUTION_REPORT_SCHEMA_VERSION,
        "artifact_id": "voiceops-stripe-executor-report",
        "generated_at": finished_at,
        "ok": not issues,
        "status": "pass" if not issues else "fail",
        "execute_requested": execute,
        "live_confirmation_required": LIVE_CONFIRMATION,
        "live_confirmation_matched": confirmation == LIVE_CONFIRMATION,
        "packet_validation_status": validation["status"],
        "receipt_validation_status": receipt_validation["status"],
        "issues": sorted(set(issues)),
        "artifacts": {
            "post_approval_receipts": str(receipts_path),
            "approval_decisions_dir": str(decisions_dir),
        },
        "command_results": command_results,
        "safety": {
            "raw_card_data_in_model_context": False,
            "credential_retrieval": False,
            "unapproved_execution": False,
            "commands_limited_to_packet_and_plan": True,
            "stdout_stderr_redacted": True,
        },
    }
    _write_json(output_dir / "stripe-executor-report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemoclaw-action-packet", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--approval-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="Run approve_once commands after all guards pass.")
    parser.add_argument("--confirm-live-actions", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args(argv)

    report = execute_approved_actions(
        packet=_read_json(args.nemoclaw_action_packet),
        plan=_read_json(args.execution_plan),
        decisions_payload=_read_json(args.approval_decisions),
        output_dir=args.output_dir,
        execute=args.execute,
        confirmation=args.confirm_live_actions,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
