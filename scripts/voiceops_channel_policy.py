#!/usr/bin/env python3
"""Generate the Milestone 3 VoiceOps multi-channel policy artifacts.

The generator is intentionally headless and bounded. It reads no secrets,
performs no network I/O, and never sends Discord, WhatsApp, SMS, or phone-call
traffic. It only emits static policy artifacts for operator review.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = Path("artifacts/voiceops-channel-policy/current")
CHANNEL_IDS = ("discord", "whatsapp", "phone_sms")
REQUIRED_AUDIT_FIELDS = {
    "audit_id",
    "parent_audit_id",
    "source_audit_id",
    "channel_id",
    "route_id",
    "payload_digest",
}
REQUIRED_REDACTION_RULES = {
    "env_assignment_secret",
    "bearer_token",
    "secret_key_like",
    "phone_number",
    "email_address",
    "payment_card_like",
}
REQUIRED_PROHIBITED_ACTIONS = {
    "discord": {"credential_or_secret_echo", "payment_or_provisioning_action"},
    "whatsapp": {"credential_or_secret_echo", "payment_link_send", "bulk_or_marketing_broadcast"},
    "phone_sms": {"unapproved_voice_call", "payment_link_send", "credential_or_secret_echo"},
}


@dataclass(frozen=True)
class ChannelAuthorization:
    channel_id: str
    display_name: str
    ingress_allowed: bool
    egress_allowed: bool
    authorization_mode: str
    allowed_actions: list[str]
    approval_required_for: list[str]
    prohibited_actions: list[str]
    evidence_required: list[str]
    audit_required: bool


@dataclass(frozen=True)
class ApprovalRoute:
    route_id: str
    applies_to: list[str]
    trigger: str
    default_decision: str
    approver_roles: list[str]
    required_approval_count: int
    ttl_minutes: int
    escalation_level: str
    audit_event: str


@dataclass(frozen=True)
class EscalationStep:
    level: str
    trigger: str
    destination_role: str
    max_response_minutes: int
    permitted_actions: list[str]


@dataclass(frozen=True)
class RedactionRule:
    rule_id: str
    applies_to: list[str]
    pattern: str
    replacement: str
    rationale: str


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_channel_authorizations() -> list[ChannelAuthorization]:
    return [
        ChannelAuthorization(
            channel_id="discord",
            display_name="Discord",
            ingress_allowed=True,
            egress_allowed=True,
            authorization_mode="operator_or_service_webhook_only",
            allowed_actions=[
                "receive_operator_command",
                "post_redacted_status",
                "request_human_approval",
                "acknowledge_policy_decision",
            ],
            approval_required_for=[
                "customer_visible_message",
                "new_webhook_or_bot_scope",
                "role_or_permission_change",
                "sensitive_context_replay",
            ],
            prohibited_actions=[
                "server_admin_mutation",
                "credential_or_secret_echo",
                "payment_or_provisioning_action",
                "unapproved_external_invite",
            ],
            evidence_required=["operator_actor_id", "guild_or_channel_id", "source_audit_id"],
            audit_required=True,
        ),
        ChannelAuthorization(
            channel_id="whatsapp",
            display_name="WhatsApp",
            ingress_allowed=True,
            egress_allowed=True,
            authorization_mode="approved_template_or_operator_reply_only",
            allowed_actions=[
                "receive_customer_or_operator_message",
                "draft_customer_reply",
                "send_approved_customer_reply",
                "deliver_approval_request",
            ],
            approval_required_for=[
                "any_customer_visible_send",
                "template_change",
                "handoff_to_human_agent",
                "attachment_or_media_send",
            ],
            prohibited_actions=[
                "unapproved_template_send",
                "payment_link_send",
                "credential_or_secret_echo",
                "bulk_or_marketing_broadcast",
            ],
            evidence_required=["business_account_id", "conversation_id", "source_audit_id"],
            audit_required=True,
        ),
        ChannelAuthorization(
            channel_id="phone_sms",
            display_name="Phone SMS",
            ingress_allowed=True,
            egress_allowed=True,
            authorization_mode="sms_and_approved_voice_handoff_only",
            allowed_actions=[
                "receive_sms_command",
                "draft_sms_reply",
                "send_approved_sms_reply",
                "deliver_approval_request",
                "queue_approved_phone_handoff_call",
            ],
            approval_required_for=[
                "any_sms_send",
                "approved_phone_handoff_call",
                "phone_number_change",
                "incident_escalation_sms",
                "customer_visible_handoff",
            ],
            prohibited_actions=[
                "unapproved_voice_call",
                "mms_send",
                "payment_link_send",
                "bulk_or_marketing_campaign",
                "credential_or_secret_echo",
            ],
            evidence_required=["normalized_phone_hash", "conversation_id", "source_audit_id", "phone_context_ref"],
            audit_required=True,
        ),
    ]


def default_approval_routes() -> list[ApprovalRoute]:
    return [
        ApprovalRoute(
            route_id="status_only",
            applies_to=list(CHANNEL_IDS),
            trigger="redacted_internal_status_or_acknowledgement",
            default_decision="allow_after_policy_check",
            approver_roles=["channel_policy"],
            required_approval_count=0,
            ttl_minutes=0,
            escalation_level="none",
            audit_event="channel_policy.status_only.allow",
        ),
        ApprovalRoute(
            route_id="customer_visible_outbound",
            applies_to=["whatsapp", "phone_sms", "discord"],
            trigger="message_visible_to_customer_or_external_counterparty",
            default_decision="hold_for_human_approval",
            approver_roles=["operator_on_call", "channel_owner"],
            required_approval_count=1,
            ttl_minutes=15,
            escalation_level="level_1",
            audit_event="channel_policy.customer_visible.request_approval",
        ),
        ApprovalRoute(
            route_id="sensitive_context_replay",
            applies_to=list(CHANNEL_IDS),
            trigger="request_to_replay_transcript_context_or_identifiers",
            default_decision="hold_for_dual_approval",
            approver_roles=["operator_on_call", "privacy_reviewer"],
            required_approval_count=2,
            ttl_minutes=30,
            escalation_level="level_2",
            audit_event="channel_policy.sensitive_context.request_approval",
        ),
        ApprovalRoute(
            route_id="approved_phone_handoff_call",
            applies_to=["phone_sms"],
            trigger="operator_approved_phone_call_with_preserved_context",
            default_decision="hold_for_human_approval",
            approver_roles=["operator_on_call"],
            required_approval_count=1,
            ttl_minutes=10,
            escalation_level="level_1",
            audit_event="channel_policy.phone_handoff.request_approval",
        ),
        ApprovalRoute(
            route_id="spend_provisioning_or_credential",
            applies_to=list(CHANNEL_IDS),
            trigger="payment_provisioning_credential_or_account_mutation_intent",
            default_decision="deny_and_escalate",
            approver_roles=["business_owner", "security_owner"],
            required_approval_count=2,
            ttl_minutes=60,
            escalation_level="level_3",
            audit_event="channel_policy.blocked_capability.escalate",
        ),
    ]


def default_escalation_policy() -> list[EscalationStep]:
    return [
        EscalationStep(
            level="level_1",
            trigger="approval_ttl_expired_or_operator_uncertain",
            destination_role="channel_owner",
            max_response_minutes=15,
            permitted_actions=["keep_draft_unsent", "request_clarifying_approval", "post_internal_status"],
        ),
        EscalationStep(
            level="level_2",
            trigger="privacy_risk_safety_risk_or_repeated_denial",
            destination_role="incident_commander",
            max_response_minutes=10,
            permitted_actions=["freeze_channel_egress", "preserve_audit_chain", "page_privacy_reviewer"],
        ),
        EscalationStep(
            level="level_3",
            trigger="spend_provisioning_credential_legal_or_reputation_risk",
            destination_role="business_owner_and_security_owner",
            max_response_minutes=60,
            permitted_actions=["deny_execution", "record_blocked_intent", "require_written_approval"],
        ),
    ]


def default_redaction_rules() -> list[RedactionRule]:
    return [
        RedactionRule(
            rule_id="env_assignment_secret",
            applies_to=list(CHANNEL_IDS),
            pattern=r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|KEY|PASSWORD|AUTH)[A-Z0-9_]*)\s*=\s*([^\s,;]+)",
            replacement=r"\1=<redacted>",
            rationale="Mask secret-looking env assignments before generic token or phone redaction runs.",
        ),
        RedactionRule(
            rule_id="bearer_token",
            applies_to=list(CHANNEL_IDS),
            pattern=r"(?i)\bBearer\s+[A-Za-z0-9._\-]{8,}",
            replacement="Bearer <redacted>",
            rationale="Do not expose authorization headers in channel logs or replies.",
        ),
        RedactionRule(
            rule_id="secret_key_like",
            applies_to=list(CHANNEL_IDS),
            pattern=r"(?i)\b(?:sk|pk|rk|whsec|AC|SG|xox[baprs]|gh[pousr])[_-]?[A-Za-z0-9][A-Za-z0-9_\-]{8,}\b",
            replacement="<redacted-secret>",
            rationale="Mask provider keys, signing secrets, account identifiers, and bot tokens.",
        ),
        RedactionRule(
            rule_id="payment_card_like",
            applies_to=list(CHANNEL_IDS),
            pattern=r"\b(?:\d[ -]*?){13,19}\b",
            replacement="<redacted-card>",
            rationale="Prevent card-like numbers from being persisted into channel artifacts.",
        ),
        RedactionRule(
            rule_id="phone_number",
            applies_to=["whatsapp", "phone_sms", "discord"],
            pattern=r"(?<!\d)\+?[1-9]\d[\d .()\-]{7,}\d(?!\d)",
            replacement="<redacted-phone>",
            rationale="Keep phone identifiers out of generated artifacts and channel replies.",
        ),
        RedactionRule(
            rule_id="email_address",
            applies_to=list(CHANNEL_IDS),
            pattern=r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
            replacement="<redacted-email>",
            rationale="Mask personal and business email addresses unless separately approved.",
        ),
    ]


def apply_redactions(text: str, rules: Iterable[RedactionRule] | None = None) -> str:
    redacted = text
    for rule in rules if rules is not None else default_redaction_rules():
        redacted = re.sub(rule.pattern, rule.replacement, redacted)
    return redacted


def build_channel_policy() -> dict[str, Any]:
    channels = default_channel_authorizations()
    routes = default_approval_routes()
    escalations = default_escalation_policy()
    redactions = default_redaction_rules()
    return {
        "generated_at": _utc_now(),
        "artifact_version": "voiceops.multi_channel_policy.v1",
        "policy_id": "voiceops-m3-channel-policy",
        "milestone": "milestone_3_multi_channel_policy",
        "policy_version": "2026-06-29.m3",
        "mode": {
            "headless": True,
            "bounded": True,
            "network_io": False,
            "env_secret_reads": False,
            "outbound_sends": False,
            "outbound_calls": False,
            "artifact_only": True,
        },
        "scope": {
            "channels": list(CHANNEL_IDS),
            "default_output_dir": str(DEFAULT_OUTPUT_DIR),
            "blocked_capabilities": [
                "discord_send_without_approval",
                "whatsapp_send_without_approval",
                "sms_send_without_approval",
                "voice_call",
                "payment_or_provisioning_action",
                "credential_or_secret_retrieval",
            ],
        },
        "channel_authorization": [asdict(channel) for channel in channels],
        "approval_routing": [asdict(route) for route in routes],
        "escalation_policy": [asdict(step) for step in escalations],
        "audit_id_continuity": {
            "audit_id_format": "vops-m3-{channel_id}-{utc_yyyymmddThhmmssZ}-{sequence}",
            "required_fields": [
                "audit_id",
                "parent_audit_id",
                "source_audit_id",
                "channel_id",
                "route_id",
                "actor_kind",
                "redaction_profile",
                "payload_digest",
            ],
            "rules": [
                "Never overwrite an existing audit_id; append a child event with parent_audit_id.",
                "Every outbound draft or approval request carries the inbound source_audit_id.",
                "Escalations inherit the blocked or pending event as parent_audit_id.",
                "Redaction events keep the same source_audit_id and record the redaction rule ids applied.",
                "Cross-channel handoff creates a new audit_id and preserves the prior channel event as parent_audit_id.",
            ],
        },
        "redaction_rules": [asdict(rule) for rule in redactions],
    }


def validate_policy(policy: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    channels = {channel.get("channel_id") for channel in policy.get("channel_authorization", [])}
    missing_channels = set(CHANNEL_IDS) - channels
    if missing_channels:
        issues.append(f"missing_channels:{','.join(sorted(missing_channels))}")
    for channel in policy.get("channel_authorization", []):
        channel_id = str(channel.get("channel_id") or "")
        if channel.get("audit_required") is not True:
            issues.append(f"audit_not_required:{channel_id}")
        if not channel.get("approval_required_for"):
            issues.append(f"missing_approval_requirements:{channel_id}")
        if not channel.get("evidence_required"):
            issues.append(f"missing_evidence_requirements:{channel_id}")
        prohibited = set(channel.get("prohibited_actions") or [])
        missing_prohibited = REQUIRED_PROHIBITED_ACTIONS.get(channel_id, set()) - prohibited
        if missing_prohibited:
            issues.append(f"missing_prohibited_actions:{channel_id}:{','.join(sorted(missing_prohibited))}")
    if not policy.get("approval_routing"):
        issues.append("missing_approval_routing")
    else:
        route_ids = {route.get("route_id") for route in policy.get("approval_routing", [])}
        for required_route in {
            "approved_phone_handoff_call",
            "customer_visible_outbound",
            "sensitive_context_replay",
            "spend_provisioning_or_credential",
        }:
            if required_route not in route_ids:
                issues.append(f"missing_approval_route:{required_route}")
    if not policy.get("escalation_policy"):
        issues.append("missing_escalation_policy")
    audit = policy.get("audit_id_continuity", {})
    if not audit.get("rules"):
        issues.append("missing_audit_id_continuity")
    missing_audit_fields = REQUIRED_AUDIT_FIELDS - set(audit.get("required_fields") or [])
    if missing_audit_fields:
        issues.append(f"missing_audit_fields:{','.join(sorted(missing_audit_fields))}")
    redaction_rules = policy.get("redaction_rules") or []
    if not redaction_rules:
        issues.append("missing_redaction_rules")
    else:
        missing_redaction_rules = REQUIRED_REDACTION_RULES - {rule.get("rule_id") for rule in redaction_rules}
        if missing_redaction_rules:
            issues.append(f"missing_redaction_rules:{','.join(sorted(missing_redaction_rules))}")
    mode = policy.get("mode", {})
    for key in ("network_io", "env_secret_reads", "outbound_sends", "outbound_calls"):
        if mode.get(key) is not False:
            issues.append(f"unsafe_mode:{key}")
    return issues


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _markdown(policy: dict[str, Any]) -> str:
    validation_issues = validate_policy(policy)
    lines = [
        "# VoiceOps Milestone 3 Channel Policy",
        "",
        f"- Policy ID: {policy['policy_id']}",
        f"- Artifact version: {policy['artifact_version']}",
        f"- Policy version: {policy['policy_version']}",
        "- Mode: headless, bounded, artifact-only; no network, secret reads, sends, SMS, or calls",
        f"- Validation: {', '.join(validation_issues) if validation_issues else 'pass'}",
        "",
        "## Channel Authorization",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Channel", "Authorization", "Allowed", "Approval Required", "Prohibited"],
            [
                [
                    channel["display_name"],
                    channel["authorization_mode"],
                    ", ".join(channel["allowed_actions"]),
                    ", ".join(channel["approval_required_for"]),
                    ", ".join(channel["prohibited_actions"]),
                ]
                for channel in policy["channel_authorization"]
            ],
        )
    )
    lines.extend(["", "## Approval Routing", ""])
    lines.extend(
        _markdown_table(
            ["Route", "Applies To", "Decision", "Approvers", "Escalation"],
            [
                [
                    route["route_id"],
                    ", ".join(route["applies_to"]),
                    route["default_decision"],
                    ", ".join(route["approver_roles"]) or "none",
                    route["escalation_level"],
                ]
                for route in policy["approval_routing"]
            ],
        )
    )
    lines.extend(["", "## Escalation Policy", ""])
    for step in policy["escalation_policy"]:
        lines.append(
            f"- {step['level']}: {step['trigger']} -> {step['destination_role']} "
            f"within {step['max_response_minutes']} minutes"
        )
    lines.extend(["", "## Audit ID Continuity", ""])
    lines.append(f"- Format: `{policy['audit_id_continuity']['audit_id_format']}`")
    for rule in policy["audit_id_continuity"]["rules"]:
        lines.append(f"- {rule}")
    lines.extend(["", "## Redaction Rules", ""])
    for rule in policy["redaction_rules"]:
        lines.append(f"- {rule['rule_id']}: `{rule['replacement']}` for {', '.join(rule['applies_to'])}")
    lines.append("")
    return "\n".join(lines)


def write_channel_policy(output_dir: Path, policy: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "channel-policy.json",
        "markdown": output_dir / "channel-policy.md",
    }
    _write_json(paths["json"], policy)
    paths["markdown"].write_text(_markdown(policy), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    policy = build_channel_policy()
    issues = validate_policy(policy)
    paths = write_channel_policy(args.output_dir, policy)
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
