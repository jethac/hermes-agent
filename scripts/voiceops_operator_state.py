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
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.voiceops_provisioning_probe import (
    KAME_ACTION_PROMOTED_FIELDS,
    TOOL_DISCLOSURE_BRIDGE_TOOL_NAMES,
    TOOL_DISCLOSURE_TEST_REFS,
    build_kame_action_evidence,
    build_tool_disclosure_proof,
)


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
REQUIRED_BLOCKED_CAPABILITIES = {
    "network_probe",
    "environment_secret_read",
    "discord_send",
    "whatsapp_send",
    "sms_send",
    "phone_call",
    "spend",
    "service_provisioning",
}
APPROVAL_REQUIRED_CATEGORIES = {"spend", "provisioning"}
APPROVAL_REQUIRED_SERVICE_PROVIDERS = {"stripe_projects", "twilio_or_vapi", "whatsapp_cloud"}
ALLOWED_APPROVAL_STATUSES = {"pending", "approved", "denied", "expired"}
ALLOWED_APPROVAL_DECISIONS = {"hold_for_operator", "deny", "approved_after_operator_review"}
REQUIRED_APPROVAL_DECISIONS = {"approve_once", "deny", "hold"}
REQUIRED_KAME_PROMOTIONS = {"interpreter_promoted", "oracle_promoted"}
REJECTED_KAME_APPROVAL_LABELS = {"reflex_hypothesis", "auxiliary_hypothesis", "diagnostic_only", "hypothesis"}
ALLOWED_AUDIT_STATUSES = {"recorded", "held", "planned", "blocked", "approved", "denied"}
ALLOWED_SERVICE_STATUSES = {"planned", "provisioned", "blocked", "approval_required"}
ALLOWED_EXECUTION_STATUSES = {"not_executed", "local_artifact_written"}
ALLOWED_TASK_STATUSES = {"queued", "planned", "approval_required", "blocked_on_approval"}


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
    action_id: str
    provider: str
    title: str
    category: str
    requester_surface: str
    risk_level: str
    budget_impact_cents: int
    default_decision: str
    status: str
    ttl_minutes: int
    approval_artifact: str
    command: str
    approval_contract: dict[str, Any]
    kame_evidence: dict[str, Any]
    tool_disclosure_ref: str
    execution_status: str
    operator_next_step: str


@dataclass(frozen=True)
class AuditEvent:
    audit_id: str
    event_type: str
    status: str
    surface: str
    summary: str
    artifact_ref: str
    operator_next_step: str
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
    artifact_ref: str | None
    approval_ref: str | None
    execution_status: str
    operator_next_step: str


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


def _command_sha256(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def _approval_contract(
    *,
    approval_id: str,
    action_id: str,
    command: str,
    approval_artifact: str,
    required_preflight_gates: list[str],
    ttl_minutes: int,
) -> dict[str, Any]:
    return {
        "approval_id": approval_id,
        "action_id": action_id,
        "approval_channel": "discord_voice_operator_confirmation",
        "approval_artifact": approval_artifact,
        "approved_by_ref": None,
        "command_sha256": _command_sha256(command),
        "required_preflight_gates": required_preflight_gates,
        "allowed_decisions": ["approve_once", "deny", "hold"],
        "default_decision": "hold",
        "ttl_seconds": ttl_minutes * 60,
        "status": "pending",
    }


def default_pending_approvals() -> list[PendingApproval]:
    provision_command = "stripe projects add twilio/voice"
    spend_command = (
        "link-cli spend-request create --merchant-name ExampleOps "
        "--merchant-url https://example.invalid --amount 4900 --request-approval"
    )
    whatsapp_command = (
        "hermes voiceops channel enable --surface whatsapp_business "
        "--dry-run --require-approval"
    )
    return [
        PendingApproval(
            approval_id="vops-m5-approval-001",
            action_id="provision-voip-provider",
            provider="stripe-projects",
            title="Provision VoIP provider account through Stripe Projects",
            category="provisioning",
            requester_surface="discord_voice",
            risk_level="medium",
            budget_impact_cents=2500,
            default_decision="hold_for_operator",
            status="pending",
            ttl_minutes=30,
            approval_artifact="nemoclaw-action-packet.json",
            command=provision_command,
            approval_contract=_approval_contract(
                approval_id="vops-m5-approval-001",
                action_id="provision-voip-provider",
                command=provision_command,
                approval_artifact="nemoclaw-action-packet.json",
                required_preflight_gates=["stripe_cli", "stripe_projects_cli", "mpp_agent"],
                ttl_minutes=30,
            ),
            kame_evidence=build_kame_action_evidence("provision-voip-provider"),
            tool_disclosure_ref="tool_disclosure",
            execution_status="not_executed",
            operator_next_step="Review nemoclaw-action-packet.json, confirm provisioning preflight gates, then approve or hold.",
        ),
        PendingApproval(
            approval_id="vops-m5-approval-002",
            action_id="buy-service-credit",
            provider="stripe-link-cli",
            title="Buy prepaid operations service credit through Stripe Link",
            category="spend",
            requester_surface="discord_voice",
            risk_level="medium",
            budget_impact_cents=4900,
            default_decision="hold_for_operator",
            status="pending",
            ttl_minutes=15,
            approval_artifact="nemoclaw-action-packet.json",
            command=spend_command,
            approval_contract=_approval_contract(
                approval_id="vops-m5-approval-002",
                action_id="buy-service-credit",
                command=spend_command,
                approval_artifact="nemoclaw-action-packet.json",
                required_preflight_gates=["stripe_link_cli", "mpp_agent"],
                ttl_minutes=15,
            ),
            kame_evidence=build_kame_action_evidence("buy-service-credit"),
            tool_disclosure_ref="tool_disclosure",
            execution_status="not_executed",
            operator_next_step="Review the Link spend request details and budget impact before approving any spend.",
        ),
        PendingApproval(
            approval_id="vops-m5-approval-003",
            action_id="enable-whatsapp-egress",
            provider="whatsapp-cloud",
            title="Enable WhatsApp Business fallback egress",
            category="channel_egress",
            requester_surface="discord_voice",
            risk_level="medium",
            budget_impact_cents=0,
            default_decision="hold_for_operator",
            status="pending",
            ttl_minutes=30,
            approval_artifact="channel-policy.json",
            command=whatsapp_command,
            approval_contract=_approval_contract(
                approval_id="vops-m5-approval-003",
                action_id="enable-whatsapp-egress",
                command=whatsapp_command,
                approval_artifact="channel-policy.json",
                required_preflight_gates=["channel_policy_review", "operator_confirmation"],
                ttl_minutes=30,
            ),
            kame_evidence=build_kame_action_evidence("enable-whatsapp-egress"),
            tool_disclosure_ref="tool_disclosure",
            execution_status="not_executed",
            operator_next_step="Review channel-policy.json and confirm recipient policy before enabling WhatsApp egress.",
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
            artifact_ref="operator-state.json",
            operator_next_step="Review pending approvals and planned services before enabling any live operation.",
        ),
        AuditEvent(
            audit_id="vops-m5-audit-002",
            event_type="budget.reserve.requested",
            status="held",
            surface="discord_voice",
            summary="Budget reservation packet prepared; no spend executed.",
            artifact_ref="operator-state.json",
            operator_next_step="Inspect the approval contracts before releasing reserved budget.",
            parent_audit_id="vops-m5-audit-001",
        ),
        AuditEvent(
            audit_id="vops-m5-audit-003",
            event_type="service.provisioning.planned",
            status="planned",
            surface="artifact",
            summary="Phone/SMS and WhatsApp surfaces listed as planned only.",
            artifact_ref="operator-state.json",
            operator_next_step="Confirm channel policy review and provider setup evidence before enabling egress.",
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
            artifact_ref="nemoclaw-action-packet.json",
            approval_ref="vops-m5-approval-001",
            execution_status="not_executed",
            operator_next_step="Complete provisioning preflight evidence, then review the NemoClaw packet before approval.",
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
            artifact_ref="phone-context.json",
            approval_ref="vops-m5-approval-001",
            execution_status="not_executed",
            operator_next_step="Verify phone target, phone provider account, and channel policy before approving handoff.",
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
            artifact_ref="channel-policy.json",
            approval_ref="vops-m5-approval-003",
            execution_status="not_executed",
            operator_next_step="Review channel-policy.json before any WhatsApp egress is enabled.",
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
            artifact_ref="operator-state.json",
            approval_ref=None,
            execution_status="local_artifact_written",
            operator_next_step="Use this artifact as the local review surface.",
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
    pending_approvals = [asdict(approval) for approval in default_pending_approvals()]
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
        "tool_disclosure": build_tool_disclosure_proof(),
        "pending_approvals": pending_approvals,
        "approval_contracts": {
            approval["action_id"]: approval["approval_contract"]
            for approval in pending_approvals
        },
        "recent_audit_events": [asdict(event) for event in default_audit_events()],
        "planned_services": [asdict(service) for service in default_planned_services()],
        "provisioned_services": [asdict(service) for service in default_provisioned_services()],
        "upcoming_tasks": [asdict(task) for task in default_upcoming_tasks()],
    }


def _evidence_labels(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    promoted_fields = payload.get("promoted_fields")
    if not isinstance(promoted_fields, dict):
        return []
    labels: list[str] = []
    for field in promoted_fields.values():
        if isinstance(field, dict):
            label = field.get("evidence_label")
            if isinstance(label, str):
                labels.append(label)
    return labels


def _validate_kame_approval_evidence(
    approval: dict[str, Any],
    approval_id: str,
    action_id: str,
) -> list[str]:
    issues: list[str] = []
    evidence = approval.get("kame_evidence")
    if not isinstance(evidence, dict):
        issues.append(f"missing_kame_evidence:{approval_id}")
        return issues

    if evidence.get("schema_version") != "voiceops.kame_action_evidence.v1":
        issues.append(f"kame_evidence_schema_invalid:{approval_id}")
    if evidence.get("action_id") != action_id:
        issues.append(f"kame_evidence_action_mismatch:{approval_id}")
    if evidence.get("hypotheses_allowed_for_action") is not False:
        issues.append(f"kame_evidence_hypotheses_allowed:{approval_id}")
    if evidence.get("transcript_hypotheses_promoted") is not False:
        issues.append(f"kame_evidence_transcript_hypotheses_promoted:{approval_id}")
    if not evidence.get("turn_id"):
        issues.append(f"kame_evidence_missing_turn_id:{approval_id}")
    if not evidence.get("audio_segment_ref"):
        issues.append(f"kame_evidence_missing_audio_segment_ref:{approval_id}")

    required_promotions = set(evidence.get("required_promotions") or [])
    missing_promotions = REQUIRED_KAME_PROMOTIONS - required_promotions
    if missing_promotions:
        issues.append(f"kame_evidence_missing_required_promotions:{approval_id}:{','.join(sorted(missing_promotions))}")

    promoted_fields = evidence.get("promoted_fields")
    if not isinstance(promoted_fields, dict) or not promoted_fields:
        issues.append(f"kame_evidence_missing_promoted_fields:{approval_id}")
    else:
        required_fields = tuple(
            KAME_ACTION_PROMOTED_FIELDS.get(action_id, ("user_request", "oracle_action_plan", "action_rationale"))
        )
        promotion_required_before = tuple(evidence.get("promotion_required_before") or ())
        if set(promotion_required_before) != set(required_fields):
            issues.append(f"kame_evidence_promotion_required_fields_mismatch:{approval_id}")
        missing_fields = sorted(field for field in required_fields if field not in promoted_fields)
        if missing_fields:
            issues.append(f"kame_evidence_missing_required_promoted_fields:{approval_id}:{','.join(missing_fields)}")
        labels = _evidence_labels(evidence)
        if not labels:
            issues.append(f"kame_evidence_missing_promoted_field_labels:{approval_id}")
        rejected_labels = sorted(set(labels) & REJECTED_KAME_APPROVAL_LABELS)
        if rejected_labels:
            issues.append(f"kame_evidence_rejected_promoted_labels:{approval_id}:{','.join(rejected_labels)}")
        invalid_labels = sorted(set(labels) - REQUIRED_KAME_PROMOTIONS)
        if invalid_labels:
            issues.append(f"kame_evidence_invalid_promoted_labels:{approval_id}:{','.join(invalid_labels)}")
        for field in required_fields:
            promoted = promoted_fields.get(field) if isinstance(promoted_fields.get(field), dict) else {}
            if not promoted:
                continue
            if not promoted.get("source"):
                issues.append(f"kame_evidence_promoted_field_missing_source:{approval_id}:{field}")
            if not promoted.get("ref"):
                issues.append(f"kame_evidence_promoted_field_missing_ref:{approval_id}:{field}")

    if approval.get("tool_disclosure_ref") != "tool_disclosure":
        issues.append(f"tool_disclosure_ref_missing:{approval_id}")
    return issues


def _validate_tool_disclosure(tool_disclosure: Any) -> list[str]:
    issues: list[str] = []
    if not isinstance(tool_disclosure, dict):
        return ["missing_tool_disclosure"]
    if tool_disclosure.get("schema_version") != "voiceops.tool_disclosure_proof.v1":
        issues.append("tool_disclosure_schema_invalid")
    if tool_disclosure.get("ok") is not True:
        issues.append("tool_disclosure_not_ok")
    config = tool_disclosure.get("config") if isinstance(tool_disclosure.get("config"), dict) else {}
    if config.get("enabled") != "on":
        issues.append("tool_disclosure_enabled_not_on")
    if config.get("defer_core") != "all":
        issues.append("tool_disclosure_defer_core_not_all")
    visible_tools = set(tool_disclosure.get("visible_tool_names") or [])
    for tool_name in TOOL_DISCLOSURE_BRIDGE_TOOL_NAMES:
        if tool_name not in visible_tools:
            issues.append(f"tool_disclosure_visible_tool_missing:{tool_name}")
    if tool_disclosure.get("visible_non_bridge_tool_names"):
        issues.append("tool_disclosure_visible_non_bridge_tools_present")
    if tool_disclosure.get("broad_core_tools_visible") is not False:
        issues.append("tool_disclosure_broad_core_tools_visible")
    hidden_tools = set(tool_disclosure.get("hidden_core_tool_names") or [])
    input_core_tools = set(tool_disclosure.get("input_core_tools") or [])
    from toolsets import _HERMES_CORE_TOOLS

    expected_core_tools = set(_HERMES_CORE_TOOLS)
    if input_core_tools != expected_core_tools:
        issues.append("tool_disclosure_stale_input_core_tools")
    if hidden_tools != expected_core_tools:
        issues.append("tool_disclosure_stale_hidden_core_tools")
    for tool_name in input_core_tools or {"read_file", "terminal"}:
        if tool_name not in hidden_tools:
            issues.append(f"tool_disclosure_hidden_core_tool_missing:{tool_name}")
    if input_core_tools and hidden_tools != input_core_tools:
        issues.append("tool_disclosure_hidden_core_tool_set_mismatch")
    if tool_disclosure.get("core_tools_hidden_all") is not True:
        issues.append("tool_disclosure_core_tools_hidden_all_not_true")
    if tool_disclosure.get("hidden_core_tool_count") != len(hidden_tools):
        issues.append("tool_disclosure_hidden_core_tool_count_mismatch")
    if tool_disclosure.get("input_core_tool_count") != len(input_core_tools):
        issues.append("tool_disclosure_input_core_tool_count_mismatch")
    if tool_disclosure.get("deferred_count") != len(hidden_tools):
        issues.append("tool_disclosure_deferred_count_mismatch")
    if int(tool_disclosure.get("token_reduction_estimate") or 0) <= 0:
        issues.append("tool_disclosure_missing_token_reduction")
    external_test_refs = set(tool_disclosure.get("external_test_refs") or [])
    for test_ref in TOOL_DISCLOSURE_TEST_REFS:
        if test_ref not in external_test_refs:
            issues.append(f"tool_disclosure_missing_test_ref:{test_ref}")
    return issues


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
    blocked_capabilities = set(state.get("scope", {}).get("blocked_capabilities") or [])
    missing_blocked = REQUIRED_BLOCKED_CAPABILITIES - blocked_capabilities
    if missing_blocked:
        issues.append(f"missing_blocked_capabilities:{','.join(sorted(missing_blocked))}")

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
    controls = set(budget.get("controls") or [])
    for required_control in (
        "dry_run_by_default",
        "approval_packet_required_for_any_spend",
        "provisioning_blocked_until_operator_approval",
    ):
        if required_control not in controls:
            issues.append(f"missing_budget_control:{required_control}")

    issues.extend(_validate_tool_disclosure(state.get("tool_disclosure")))

    pending_approvals = state.get("pending_approvals", [])
    approval_contracts = state.get("approval_contracts", {})
    if not isinstance(approval_contracts, dict):
        issues.append("invalid_approval_contracts")
        approval_contracts = {}
    approval_ids = [str(approval.get("approval_id") or "") for approval in pending_approvals]
    approval_id_set = set(approval_ids)
    duplicate_approvals = sorted(approval_id for approval_id in set(approval_ids) if approval_ids.count(approval_id) > 1)
    if duplicate_approvals:
        issues.append(f"duplicate_pending_approval_ids:{','.join(duplicate_approvals)}")
    action_ids = [str(approval.get("action_id") or "") for approval in pending_approvals]
    duplicate_actions = sorted(action_id for action_id in set(action_ids) if action_ids.count(action_id) > 1)
    if duplicate_actions:
        issues.append(f"duplicate_pending_approval_action_ids:{','.join(duplicate_actions)}")
    missing_contract_actions = sorted(set(action_ids) - set(approval_contracts))
    if missing_contract_actions:
        issues.append(f"missing_approval_contracts:{','.join(missing_contract_actions)}")
    pending_budget_total = 0
    for approval in pending_approvals:
        approval_id = str(approval.get("approval_id") or "unknown")
        action_id = str(approval.get("action_id") or "")
        category = str(approval.get("category") or "")
        status = str(approval.get("status") or "")
        decision = str(approval.get("default_decision") or "")
        command = str(approval.get("command") or "")
        artifact = str(approval.get("approval_artifact") or "")
        impact = approval.get("budget_impact_cents")
        if not approval_id or approval_id == "unknown":
            issues.append("missing_pending_approval_id")
        if not action_id:
            issues.append(f"missing_pending_approval_action_id:{approval_id}")
        if not approval.get("provider"):
            issues.append(f"missing_pending_approval_provider:{approval_id}")
        if not artifact:
            issues.append(f"missing_pending_approval_artifact:{approval_id}")
        if not command:
            issues.append(f"missing_pending_approval_command:{approval_id}")
        if not approval.get("operator_next_step"):
            issues.append(f"missing_operator_next_step:{approval_id}")
        execution_status = approval.get("execution_status")
        if execution_status != "not_executed":
            issues.append(f"approval_execution_claimed:{approval_id}")
        if category in APPROVAL_REQUIRED_CATEGORIES and decision != "hold_for_operator":
            issues.append(f"unsafe_approval_decision:{approval_id}")
        if status not in ALLOWED_APPROVAL_STATUSES:
            issues.append(f"invalid_approval_status:{approval_id}:{status}")
        if decision not in ALLOWED_APPROVAL_DECISIONS:
            issues.append(f"invalid_approval_decision:{approval_id}:{decision}")
        if not isinstance(impact, int) or impact < 0:
            issues.append(f"invalid_approval_budget_impact:{approval_id}")
        else:
            pending_budget_total += impact
        issues.extend(_validate_kame_approval_evidence(approval, approval_id, action_id))
        embedded_contract = approval.get("approval_contract")
        contract = approval_contracts.get(action_id)
        if not isinstance(embedded_contract, dict):
            issues.append(f"missing_embedded_approval_contract:{approval_id}")
        elif contract != embedded_contract:
            issues.append(f"approval_contract_mismatch:{approval_id}")
        if isinstance(contract, dict):
            if contract.get("approval_id") != approval_id:
                issues.append(f"approval_contract_id_mismatch:{approval_id}")
            if contract.get("action_id") != action_id:
                issues.append(f"approval_contract_action_mismatch:{approval_id}")
            if contract.get("approval_artifact") != artifact:
                issues.append(f"approval_contract_artifact_mismatch:{approval_id}")
            if contract.get("approved_by_ref") is not None:
                issues.append(f"approval_contract_already_approved:{approval_id}")
            allowed_decisions = set(contract.get("allowed_decisions") or [])
            if allowed_decisions != REQUIRED_APPROVAL_DECISIONS:
                issues.append(f"approval_contract_decisions_mismatch:{approval_id}")
            if not contract.get("required_preflight_gates"):
                issues.append(f"approval_contract_missing_preflight_gates:{approval_id}")
            if contract.get("command_sha256") != _command_sha256(command):
                issues.append(f"approval_contract_command_digest_mismatch:{approval_id}")
            ttl_seconds = contract.get("ttl_seconds")
            if not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
                issues.append(f"approval_contract_invalid_ttl:{approval_id}")
    if isinstance(reserved, int) and pending_budget_total != reserved:
        issues.append("pending_approval_budget_mismatch")

    if len(pending_approvals) > MAX_PENDING_APPROVALS:
        issues.append("bounds_exceeded:pending_approvals")
    recent_audit_events = state.get("recent_audit_events", [])
    if len(recent_audit_events) > MAX_AUDIT_EVENTS:
        issues.append("bounds_exceeded:recent_audit_events")
    planned_services = state.get("planned_services", [])
    if len(planned_services) > MAX_SERVICES_PER_SECTION:
        issues.append("bounds_exceeded:planned_services")
    provisioned_services = state.get("provisioned_services", [])
    if len(provisioned_services) > MAX_SERVICES_PER_SECTION:
        issues.append("bounds_exceeded:provisioned_services")
    upcoming_tasks = state.get("upcoming_tasks", [])
    if len(upcoming_tasks) > MAX_UPCOMING_TASKS:
        issues.append("bounds_exceeded:upcoming_tasks")

    audit_ids = [str(event.get("audit_id") or "") for event in recent_audit_events]
    duplicate_audits = sorted(audit_id for audit_id in set(audit_ids) if audit_ids.count(audit_id) > 1)
    if duplicate_audits:
        issues.append(f"duplicate_audit_ids:{','.join(duplicate_audits)}")
    audit_id_set = set(audit_ids)
    for event in recent_audit_events:
        audit_id = str(event.get("audit_id") or "unknown")
        status = str(event.get("status") or "")
        parent = event.get("parent_audit_id")
        if not audit_id or audit_id == "unknown":
            issues.append("missing_audit_id")
        if status not in ALLOWED_AUDIT_STATUSES:
            issues.append(f"invalid_audit_status:{audit_id}:{status}")
        if not event.get("operator_next_step"):
            issues.append(f"missing_operator_next_step:{audit_id}")
        if not event.get("artifact_ref"):
            issues.append(f"missing_audit_artifact_ref:{audit_id}")
        if parent is not None and parent not in audit_id_set:
            issues.append(f"audit_parent_missing:{audit_id}:{parent}")

    service_ids = [
        str(service.get("service_id") or "") for service in [*planned_services, *provisioned_services]
    ]
    duplicate_services = sorted(service_id for service_id in set(service_ids) if service_ids.count(service_id) > 1)
    if duplicate_services:
        issues.append(f"duplicate_service_ids:{','.join(duplicate_services)}")
    for service in planned_services:
        service_id = str(service.get("service_id") or "unknown")
        provider = str(service.get("provider") or "")
        status = str(service.get("status") or "")
        if status not in ALLOWED_SERVICE_STATUSES:
            issues.append(f"invalid_service_status:{service_id}:{status}")
        if provider in APPROVAL_REQUIRED_SERVICE_PROVIDERS and service.get("approval_required") is not True:
            issues.append(f"external_service_missing_approval:{service_id}")
        if service.get("approval_required") is True:
            approval_ref = service.get("approval_ref")
            if not approval_ref:
                issues.append(f"external_service_missing_approval_ref:{service_id}")
            elif str(approval_ref) not in approval_id_set:
                issues.append(f"external_service_unknown_approval_ref:{service_id}:{approval_ref}")
        if service.get("external") is True and status == "provisioned":
            issues.append(f"external_service_claimed_provisioned:{service_id}")
        if service.get("external") is True:
            if service.get("execution_status") != "not_executed":
                issues.append(f"external_service_execution_claimed:{service_id}")
            if not service.get("operator_next_step"):
                issues.append(f"missing_operator_next_step:{service_id}")
            if not service.get("artifact_ref"):
                issues.append(f"external_service_missing_artifact_ref:{service_id}")
        if service.get("execution_status") not in ALLOWED_EXECUTION_STATUSES:
            issues.append(f"invalid_service_execution_status:{service_id}:{service.get('execution_status')}")
    for service in provisioned_services:
        service_id = str(service.get("service_id") or "unknown")
        status = str(service.get("status") or "")
        if status not in ALLOWED_SERVICE_STATUSES:
            issues.append(f"invalid_service_status:{service_id}:{status}")
        if service.get("external") is True:
            issues.append(f"external_service_claimed_provisioned:{service_id}")
        if not service.get("operator_next_step"):
            issues.append(f"missing_operator_next_step:{service_id}")
        if service.get("execution_status") not in ALLOWED_EXECUTION_STATUSES:
            issues.append(f"invalid_service_execution_status:{service_id}:{service.get('execution_status')}")

    domains = {task.get("domain") for task in upcoming_tasks}
    for required_domain in ("household", "business"):
        if required_domain not in domains:
            issues.append(f"missing_task_domain:{required_domain}")
    task_ids = [str(task.get("task_id") or "") for task in upcoming_tasks]
    duplicate_tasks = sorted(task_id for task_id in set(task_ids) if task_ids.count(task_id) > 1)
    if duplicate_tasks:
        issues.append(f"duplicate_task_ids:{','.join(duplicate_tasks)}")
    for task in upcoming_tasks:
        task_id = str(task.get("task_id") or "unknown")
        status = str(task.get("status") or "")
        impact = task.get("budget_impact_cents")
        if status not in ALLOWED_TASK_STATUSES:
            issues.append(f"invalid_task_status:{task_id}:{status}")
        if not isinstance(impact, int) or impact < 0:
            issues.append(f"invalid_task_budget_impact:{task_id}")
        if isinstance(impact, int) and impact > 0 and task.get("approval_required") is not True:
            issues.append(f"task_budget_without_approval:{task_id}")

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
            ["Approval", "Action", "Provider", "Execution", "Artifact", "Next step"],
            [
                [
                    approval["title"],
                    approval["action_id"],
                    approval["provider"],
                    approval["execution_status"],
                    approval["approval_artifact"],
                    approval["operator_next_step"],
                ]
                for approval in state["pending_approvals"]
            ],
        )
    )
    lines.extend(["", "## Recent Audit Events", ""])
    for event in state["recent_audit_events"]:
        parent = f" parent={event['parent_audit_id']}" if event.get("parent_audit_id") else ""
        lines.append(
            f"- {event['audit_id']}: {event['event_type']} [{event['status']}]{parent} - {event['summary']} "
            f"Artifact: `{event['artifact_ref']}`. Next step: {event['operator_next_step']}"
        )
    lines.extend(["", "## Planned Services", ""])
    for service in state["planned_services"]:
        lines.append(
            f"- {service['display_name']}: {service['status']} via {service['provider']}; "
            f"approval_required={service['approval_required']}; execution_status={service['execution_status']}; "
            f"artifact={service['artifact_ref']}; next_step={service['operator_next_step']}"
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
