#!/usr/bin/env python3
"""Generate the Milestone 5 VoiceOps operator-state artifacts.

The generator is intentionally headless and bounded. It does not inspect
environment secrets, perform network I/O, send messages, place calls, spend
money, or provision services. It only emits JSON and Markdown artifacts for
operator review.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("artifacts/voiceops-operator-state/current")
ALLOWED_CURRENT_MODES = {"dry-run", "approval-required", "live"}
MAX_PENDING_APPROVALS = 8
MAX_AUDIT_EVENTS = 12
MAX_SERVICES_PER_SECTION = 8
MAX_UPCOMING_TASKS = 12
REQUIRED_MODE_FALSE_FLAGS = {
    "network_io",
    "env_secret_reads",
    "outbound_sends",
    "outbound_calls",
    "live_spend",
    "provisioning",
}


@dataclass(frozen=True)
class VoiceSurface:
    surface_id: str
    display_name: str
    status: str
    fallback_surface_id: str
    fallback_reason: str


@dataclass(frozen=True)
class BudgetStatus:
    currency: str
    current_mode: str
    approved_budget_cents: int
    reserved_cents: int
    spent_cents: int
    remaining_cents: int
    status: str
    controls: list[str]


@dataclass(frozen=True)
class PendingApproval:
    approval_id: str
    title: str
    category: str
    requester_surface: str
    risk_level: str
    budget_impact_cents: int
    default_decision: str
    status: str
    ttl_minutes: int


@dataclass(frozen=True)
class AuditEvent:
    audit_id: str
    event_type: str
    status: str
    surface: str
    summary: str
    parent_audit_id: str | None = None


@dataclass(frozen=True)
class ServiceState:
    service_id: str
    display_name: str
    provider: str
    status: str
    capability: str
    external: bool
    approval_required: bool
    notes: str


@dataclass(frozen=True)
class UpcomingTask:
    task_id: str
    domain: str
    title: str
    status: str
    due_window: str
    required_surface: str
    approval_required: bool
    budget_impact_cents: int


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_voice_surface() -> VoiceSurface:
    return VoiceSurface(
        surface_id="discord_voice",
        display_name="Discord voice",
        status="active_for_operator_review",
        fallback_surface_id="whatsapp_text",
        fallback_reason="Phone/SMS voice calls remain disabled until a provisioning approval and channel policy review pass.",
    )


def default_budget_status() -> BudgetStatus:
    approved_budget_cents = 20_000
    spent_cents = 0
    reserved_cents = 7_400
    return BudgetStatus(
        currency="USD",
        current_mode="approval-required",
        approved_budget_cents=approved_budget_cents,
        reserved_cents=reserved_cents,
        spent_cents=spent_cents,
        remaining_cents=approved_budget_cents - reserved_cents - spent_cents,
        status="no_live_spend_without_explicit_approval",
        controls=[
            "dry_run_by_default",
            "approval_packet_required_for_any_spend",
            "provisioning_blocked_until_operator_approval",
        ],
    )


def default_pending_approvals() -> list[PendingApproval]:
    return [
        PendingApproval(
            approval_id="vops-m5-approval-001",
            title="Provision VoIP provider account through Stripe Projects",
            category="provisioning",
            requester_surface="discord_voice",
            risk_level="medium",
            budget_impact_cents=2500,
            default_decision="hold_for_operator",
            status="pending",
            ttl_minutes=30,
        ),
        PendingApproval(
            approval_id="vops-m5-approval-002",
            title="Buy prepaid operations service credit through Stripe Link",
            category="spend",
            requester_surface="discord_voice",
            risk_level="medium",
            budget_impact_cents=4900,
            default_decision="hold_for_operator",
            status="pending",
            ttl_minutes=15,
        ),
    ]


def default_audit_events() -> list[AuditEvent]:
    return [
        AuditEvent(
            audit_id="vops-m5-audit-001",
            event_type="operator_state.generated",
            status="recorded",
            surface="artifact",
            summary="Milestone 5 operator state artifact generated headlessly.",
        ),
        AuditEvent(
            audit_id="vops-m5-audit-002",
            event_type="budget.reserve.requested",
            status="held",
            surface="discord_voice",
            summary="Budget reservation packet prepared; no spend executed.",
            parent_audit_id="vops-m5-audit-001",
        ),
        AuditEvent(
            audit_id="vops-m5-audit-003",
            event_type="service.provisioning.planned",
            status="planned",
            surface="artifact",
            summary="Phone/SMS and WhatsApp surfaces listed as planned only.",
            parent_audit_id="vops-m5-audit-001",
        ),
    ]


def default_planned_services() -> list[ServiceState]:
    return [
        ServiceState(
            service_id="stripe_projects_voiceops_budget",
            display_name="Stripe Projects VoiceOps budget envelope",
            provider="stripe_projects",
            status="planned",
            capability="budget packet and spend ledger",
            external=True,
            approval_required=True,
            notes="Plan only; no Stripe command is executed by this generator.",
        ),
        ServiceState(
            service_id="phone_sms_bridge",
            display_name="Phone/SMS bridge",
            provider="twilio_or_vapi",
            status="planned",
            capability="SMS fallback and future approved phone handoff",
            external=True,
            approval_required=True,
            notes="Voice calls are blocked in this artifact-only state.",
        ),
        ServiceState(
            service_id="whatsapp_business_fallback",
            display_name="WhatsApp Business fallback",
            provider="whatsapp_cloud",
            status="planned",
            capability="approved text handoff drafts",
            external=True,
            approval_required=True,
            notes="Customer-visible sends require a separate approval.",
        ),
    ]


def default_provisioned_services() -> list[ServiceState]:
    return [
        ServiceState(
            service_id="repo_local_operator_artifacts",
            display_name="Repo-local operator artifacts",
            provider="filesystem",
            status="provisioned",
            capability="JSON and Markdown review surface",
            external=False,
            approval_required=False,
            notes="Local artifact directory only; no external service was created.",
        )
    ]


def default_upcoming_tasks() -> list[UpcomingTask]:
    return [
        UpcomingTask(
            task_id="household-grocery-restock",
            domain="household",
            title="Draft grocery restock plan from pantry notes",
            status="queued",
            due_window="next_24h",
            required_surface="discord_voice",
            approval_required=False,
            budget_impact_cents=0,
        ),
        UpcomingTask(
            task_id="household-contractor-followup",
            domain="household",
            title="Prepare contractor follow-up message for operator approval",
            status="approval_required",
            due_window="next_48h",
            required_surface="whatsapp_text",
            approval_required=True,
            budget_impact_cents=0,
        ),
        UpcomingTask(
            task_id="business-invoice-triage",
            domain="business",
            title="Summarize unpaid invoice queue and draft next actions",
            status="queued",
            due_window="next_business_day",
            required_surface="artifact",
            approval_required=False,
            budget_impact_cents=0,
        ),
        UpcomingTask(
            task_id="business-phone-handoff-provisioning",
            domain="business",
            title="Prepare phone handoff provisioning approval packet",
            status="approval_required",
            due_window="this_week",
            required_surface="discord_voice",
            approval_required=True,
            budget_impact_cents=2500,
        ),
    ]


def build_operator_state() -> dict[str, Any]:
    budget = default_budget_status()
    return {
        "generated_at": _utc_now(),
        "schema_version": "voiceops.operator_state.v1",
        "artifact_version": "voiceops.operator_state.v1",
        "state_id": "voiceops-m5-operator-state",
        "milestone": "milestone_5_operator_dashboard",
        "current_mode": budget.current_mode,
        "mode": {
            "headless": True,
            "bounded": True,
            "artifact_only": True,
            "network_io": False,
            "env_secret_reads": False,
            "outbound_sends": False,
            "outbound_calls": False,
            "live_spend": False,
            "provisioning": False,
        },
        "bounds": {
            "max_pending_approvals": MAX_PENDING_APPROVALS,
            "max_audit_events": MAX_AUDIT_EVENTS,
            "max_services_per_section": MAX_SERVICES_PER_SECTION,
            "max_upcoming_tasks": MAX_UPCOMING_TASKS,
        },
        "scope": {
            "default_output_dir": str(DEFAULT_OUTPUT_DIR),
            "blocked_capabilities": [
                "network_probe",
                "environment_secret_read",
                "discord_send",
                "whatsapp_send",
                "sms_send",
                "phone_call",
                "spend",
                "service_provisioning",
            ],
        },
        "active_voice_surface": asdict(default_voice_surface()),
        "budget_status": asdict(budget),
        "pending_approvals": [asdict(approval) for approval in default_pending_approvals()],
        "recent_audit_events": [asdict(event) for event in default_audit_events()],
        "planned_services": [asdict(service) for service in default_planned_services()],
        "provisioned_services": [asdict(service) for service in default_provisioned_services()],
        "upcoming_tasks": [asdict(task) for task in default_upcoming_tasks()],
    }


def validate_operator_state(state: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if state.get("current_mode") not in ALLOWED_CURRENT_MODES:
        issues.append("invalid_current_mode")
    mode = state.get("mode", {})
    for key in REQUIRED_MODE_FALSE_FLAGS:
        if mode.get(key) is not False:
            issues.append(f"unsafe_mode:{key}")
    if mode.get("headless") is not True:
        issues.append("unsafe_mode:headless")
    if mode.get("bounded") is not True:
        issues.append("unsafe_mode:bounded")
    if mode.get("artifact_only") is not True:
        issues.append("unsafe_mode:artifact_only")

    voice_surface = state.get("active_voice_surface", {})
    if not voice_surface.get("surface_id"):
        issues.append("missing_active_voice_surface")
    if not voice_surface.get("fallback_reason"):
        issues.append("missing_fallback_reason")

    budget = state.get("budget_status", {})
    approved = budget.get("approved_budget_cents")
    reserved = budget.get("reserved_cents")
    spent = budget.get("spent_cents")
    remaining = budget.get("remaining_cents")
    valid_budget_amounts = all(isinstance(value, int) and value >= 0 for value in (approved, reserved, spent, remaining))
    if not valid_budget_amounts:
        issues.append("invalid_budget_amounts")
    elif approved - reserved - spent != remaining:
        issues.append("budget_remaining_mismatch")
    if budget.get("status") != "no_live_spend_without_explicit_approval":
        issues.append("unsafe_budget_status")
    if budget.get("current_mode") != state.get("current_mode"):
        issues.append("current_mode_mismatch")

    if len(state.get("pending_approvals", [])) > MAX_PENDING_APPROVALS:
        issues.append("bounds_exceeded:pending_approvals")
    if len(state.get("recent_audit_events", [])) > MAX_AUDIT_EVENTS:
        issues.append("bounds_exceeded:recent_audit_events")
    if len(state.get("planned_services", [])) > MAX_SERVICES_PER_SECTION:
        issues.append("bounds_exceeded:planned_services")
    if len(state.get("provisioned_services", [])) > MAX_SERVICES_PER_SECTION:
        issues.append("bounds_exceeded:provisioned_services")
    if len(state.get("upcoming_tasks", [])) > MAX_UPCOMING_TASKS:
        issues.append("bounds_exceeded:upcoming_tasks")

    domains = {task.get("domain") for task in state.get("upcoming_tasks", [])}
    for required_domain in ("household", "business"):
        if required_domain not in domains:
            issues.append(f"missing_task_domain:{required_domain}")

    for section in ("pending_approvals", "recent_audit_events", "planned_services", "provisioned_services"):
        if section not in state:
            issues.append(f"missing_section:{section}")

    return sorted(issues)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _markdown(state: dict[str, Any]) -> str:
    validation_issues = validate_operator_state(state)
    budget = state["budget_status"]
    surface = state["active_voice_surface"]
    lines = [
        "# VoiceOps Milestone 5 Operator State",
        "",
        f"- State ID: {state['state_id']}",
        f"- Artifact version: {state['artifact_version']}",
        f"- Current mode: {state['current_mode']}",
        "- Safety: headless, bounded, artifact-only; no network, env secret reads, sends, calls, spend, or provisioning",
        f"- Validation: {', '.join(validation_issues) if validation_issues else 'pass'}",
        "",
        "## Active Voice Surface",
        "",
        f"- Active: {surface['display_name']} ({surface['surface_id']})",
        f"- Status: {surface['status']}",
        f"- Fallback: {surface['fallback_surface_id']}",
        f"- Fallback reason: {surface['fallback_reason']}",
        "",
        "## Budget Status",
        "",
        f"- Status: {budget['status']}",
        f"- Approved: {budget['currency']} {budget['approved_budget_cents'] / 100:.2f}",
        f"- Reserved: {budget['currency']} {budget['reserved_cents'] / 100:.2f}",
        f"- Spent: {budget['currency']} {budget['spent_cents'] / 100:.2f}",
        f"- Remaining: {budget['currency']} {budget['remaining_cents'] / 100:.2f}",
        "",
        "## Pending Approvals",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Approval", "Category", "Risk", "Decision", "Budget"],
            [
                [
                    approval["title"],
                    approval["category"],
                    approval["risk_level"],
                    approval["default_decision"],
                    f"{budget['currency']} {approval['budget_impact_cents'] / 100:.2f}",
                ]
                for approval in state["pending_approvals"]
            ],
        )
    )
    lines.extend(["", "## Recent Audit Events", ""])
    for event in state["recent_audit_events"]:
        parent = f" parent={event['parent_audit_id']}" if event.get("parent_audit_id") else ""
        lines.append(f"- {event['audit_id']}: {event['event_type']} [{event['status']}]{parent} - {event['summary']}")
    lines.extend(["", "## Planned Services", ""])
    for service in state["planned_services"]:
        lines.append(
            f"- {service['display_name']}: {service['status']} via {service['provider']}; "
            f"approval_required={service['approval_required']}"
        )
    lines.extend(["", "## Provisioned Services", ""])
    for service in state["provisioned_services"]:
        lines.append(f"- {service['display_name']}: {service['status']} ({service['notes']})")
    lines.extend(["", "## Upcoming Tasks", ""])
    lines.extend(
        _markdown_table(
            ["Task", "Domain", "Status", "Due", "Approval"],
            [
                [
                    task["title"],
                    task["domain"],
                    task["status"],
                    task["due_window"],
                    "yes" if task["approval_required"] else "no",
                ]
                for task in state["upcoming_tasks"]
            ],
        )
    )
    lines.append("")
    return "\n".join(lines)


def write_operator_state(output_dir: Path, state: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "operator-state.json",
        "markdown": output_dir / "operator-state.md",
        "events_jsonl": output_dir / "operator-state-events.jsonl",
    }
    _write_json(paths["json"], state)
    paths["markdown"].write_text(_markdown(state), encoding="utf-8")
    _write_jsonl(paths["events_jsonl"], state["recent_audit_events"])
    return {key: str(path) for key, path in paths.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state = build_operator_state()
    issues = validate_operator_state(state)
    paths = write_operator_state(args.output_dir, state)
    print(
        json.dumps(
            {
                "ok": not issues,
                "validation_issues": issues,
                "output_dir": str(args.output_dir),
                "artifacts": paths,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
