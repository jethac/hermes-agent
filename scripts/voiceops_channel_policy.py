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
    "actor_kind",
    "redaction_profile",
    "payload_digest",
}
REQUIRED_SCOPE_BLOCKED_CAPABILITIES = {
    "discord_send_without_approval",
    "whatsapp_send_without_approval",
    "sms_send_without_approval",
    "voice_call",
    "payment_or_provisioning_action",
    "credential_or_secret_retrieval",
}
REQUIRED_APPROVAL_REQUIREMENTS = {
    "discord": {"customer_visible_message", "sensitive_context_replay"},
    "whatsapp": {"any_customer_visible_send", "template_change", "handoff_to_human_agent"},
    "phone_sms": {"any_sms_send", "approved_phone_handoff_call", "customer_visible_handoff"},
}
REQUIRED_EVIDENCE_REQUIREMENTS = {
    "discord": {"operator_actor_id", "guild_or_channel_id", "source_audit_id"},
    "whatsapp": {"business_account_id", "conversation_id", "source_audit_id"},
    "phone_sms": {"normalized_phone_hash", "conversation_id", "source_audit_id", "phone_context_ref"},
}
REQUIRED_REDACTION_RULES = {
    "env_assignment_secret",
    "sensitive_url",
    "bearer_token",
    "discord_bot_token",
    "twilio_auth_token",
    "secret_key_like",
    "phone_number",
    "email_address",
    "payment_card_like",
}
REQUIRED_PROHIBITED_ACTIONS = {
    "discord": {
        "server_admin_mutation",
        "credential_or_secret_echo",
        "payment_or_provisioning_action",
        "unapproved_external_invite",
    },
    "whatsapp": {
        "unapproved_template_send",
        "payment_link_send",
        "credential_or_secret_echo",
        "bulk_or_marketing_broadcast",
    },
    "phone_sms": {
        "unapproved_voice_call",
        "mms_send",
        "payment_link_send",
        "bulk_or_marketing_campaign",
        "credential_or_secret_echo",
    },
}
REQUIRED_APPROVAL_ROUTES = {
    "approved_phone_handoff_call": {
        "applies_to": {"phone_sms"},
        "default_decision": "hold_for_human_approval",
        "min_approval_count": 1,
        "escalation_level": "level_1",
    },
    "customer_visible_outbound": {
        "applies_to": set(CHANNEL_IDS),
        "default_decision": "hold_for_human_approval",
        "min_approval_count": 1,
        "escalation_level": "level_1",
    },
    "sensitive_context_replay": {
        "applies_to": set(CHANNEL_IDS),
        "default_decision": "hold_for_dual_approval",
        "min_approval_count": 2,
        "escalation_level": "level_2",
    },
    "spend_provisioning_or_credential": {
        "applies_to": set(CHANNEL_IDS),
        "default_decision": "deny_and_escalate",
        "min_approval_count": 2,
        "escalation_level": "level_3",
        "approver_roles": {"business_owner", "security_owner"},
    },
}
REQUIRED_ESCALATIONS = {
    "level_1": {
        "destination_role": "channel_owner",
        "permitted_actions": {"keep_draft_unsent", "request_clarifying_approval", "post_internal_status"},
    },
    "level_2": {
        "destination_role": "incident_commander",
        "permitted_actions": {"freeze_channel_egress", "preserve_audit_chain", "page_privacy_reviewer"},
    },
    "level_3": {
        "destination_role": "business_owner_and_security_owner",
        "permitted_actions": {"deny_execution", "record_blocked_intent", "require_written_approval"},
    },
}
LEVEL_3_FORBIDDEN_ACTION_TERMS = (
    "send",
    "call",
    "provision",
    "spend",
    "payment",
    "credential",
    "secret",
    "retrieve",
    "execute",
)
AUDIT_RULE_REQUIRED_TERMS = {
    "outbound": ("outbound", "source_audit_id"),
    "escalation": ("escalation", "parent_audit_id"),
    "redaction": ("redaction", "source_audit_id"),
    "cross_channel": ("cross-channel", "parent_audit_id"),
}
HIGH_RISK_ROUTE_TRIGGER_TERMS = ("payment", "spend", "provisioning", "credential", "account_mutation")
REQUIRED_AUDIT_ID_FORMAT = "vops-m3-{channel_id}-{utc_yyyymmddThhmmssZ}-{sequence}"
REQUIRED_AUDIT_ID_FORMAT_FIELDS = {"channel_id", "utc_yyyymmddThhmmssZ", "sequence"}
AUDIT_EVENT_PATTERN = re.compile(r"^channel_policy\.[a-z0-9_]+(?:\.[a-z0-9_]+)+$")


def default_approval_route_map() -> dict[str, dict[str, str]]:
    return {
        "discord": {
            "customer_visible_message": "customer_visible_outbound",
            "new_webhook_or_bot_scope": "spend_provisioning_or_credential",
            "role_or_permission_change": "spend_provisioning_or_credential",
            "sensitive_context_replay": "sensitive_context_replay",
        },
        "whatsapp": {
            "any_customer_visible_send": "customer_visible_outbound",
            "template_change": "customer_visible_outbound",
            "handoff_to_human_agent": "customer_visible_outbound",
            "attachment_or_media_send": "customer_visible_outbound",
        },
        "phone_sms": {
            "any_sms_send": "customer_visible_outbound",
            "approved_phone_handoff_call": "approved_phone_handoff_call",
            "phone_number_change": "sensitive_context_replay",
            "incident_escalation_sms": "customer_visible_outbound",
            "customer_visible_handoff": "customer_visible_outbound",
        },
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
            rule_id="sensitive_url",
            applies_to=list(CHANNEL_IDS),
            pattern=(
                r"(?i)\bhttps://(?:"
                r"(?:canary\.|ptb\.)?discord(?:app)?\.com/api/webhooks/[^\s,;]+"
                r"|(?:checkout|buy)\.stripe\.com/[^\s,;]+"
                r")"
            ),
            replacement="<redacted-url>",
            rationale="Mask credential-bearing webhook URLs and payment checkout links before channel persistence.",
        ),
        RedactionRule(
            rule_id="discord_bot_token",
            applies_to=list(CHANNEL_IDS),
            pattern=r"\b[A-Za-z0-9_\-]{23,28}\.[A-Za-z0-9_\-]{6,8}\.[A-Za-z0-9_\-]{27,45}\b",
            replacement="<redacted-discord-token>",
            rationale="Mask raw Discord bot token strings even when they are not shown as env assignments.",
        ),
        RedactionRule(
            rule_id="twilio_auth_token",
            applies_to=list(CHANNEL_IDS),
            pattern=r"(?i)\b(twilio(?:[_ -]?auth)?[_ -]?token\s*[:=]?\s*)[a-f0-9]{32}\b",
            replacement=r"\1<redacted>",
            rationale="Mask raw Twilio auth tokens when they appear in prose rather than env assignment form.",
        ),
        RedactionRule(
            rule_id="secret_key_like",
            applies_to=list(CHANNEL_IDS),
            pattern=r"(?i)\b(?:sk|pk|rk|whsec|AC|SG|EAAG|xox[baprs]|gh[pousr])[_-]?[A-Za-z0-9][A-Za-z0-9_\-]{8,}\b",
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
            "review_required_for_real_egress": True,
            "review_status": "pending_human_review",
            "real_egress_enabled": False,
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
        "approval_route_map": default_approval_route_map(),
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


def build_review_packet(policy: dict[str, Any]) -> dict[str, Any]:
    """Build the human signoff packet for enabling future live egress."""

    channel_rows: list[dict[str, Any]] = []
    for channel in policy["channel_authorization"]:
        channel_id = channel["channel_id"]
        route_map = policy["approval_route_map"][channel_id]
        channel_rows.append(
            {
                "channel_id": channel_id,
                "display_name": channel["display_name"],
                "review_status": "pending_human_review",
                "live_egress_enabled": False,
                "required_evidence": channel["evidence_required"],
                "approval_routes_to_confirm": {
                    approval_item: route_map[approval_item] for approval_item in channel["approval_required_for"]
                },
                "blocked_capabilities_to_confirm": channel["prohibited_actions"],
                "checklist": [
                    "Confirm channel owner and operator-on-call identities.",
                    "Confirm inbound source_audit_id is present before drafting outbound content.",
                    "Confirm redaction rules are applied before display, persistence, or handoff.",
                    "Confirm approval route before any customer-visible send or call.",
                    "Confirm no blocked capability is executed through this channel.",
                ],
            }
        )

    return {
        "generated_at": _utc_now(),
        "schema_version": "voiceops.multi_channel_policy_review.v1",
        "artifact_id": "voiceops-m3-channel-policy-review",
        "milestone": policy["milestone"],
        "policy_ref": "channel-policy.json",
        "policy_id": policy["policy_id"],
        "policy_version": policy["policy_version"],
        "review_status": policy["scope"]["review_status"],
        "real_egress_enabled": policy["scope"]["real_egress_enabled"],
        "changes_policy": False,
        "artifact_only": True,
        "decision_options": [
            "approve_artifact_for_demo_recording",
            "approve_dry_run_only",
            "approve_live_egress_after_external_credentials_are_bound",
            "request_changes",
            "deny",
        ],
        "required_signoffs": [
            {
                "role": "business_owner",
                "required": True,
                "reason": "Approves business-visible WhatsApp, SMS, and phone handoff behavior.",
            },
            {
                "role": "channel_owner",
                "required": True,
                "reason": "Approves Discord, WhatsApp, and phone/SMS channel authorization boundaries.",
            },
            {
                "role": "security_owner",
                "required": True,
                "reason": "Confirms no secret retrieval, credential echo, or provisioning bypass is enabled.",
            },
            {
                "role": "privacy_reviewer",
                "required": True,
                "reason": "Confirms transcript, phone, email, payment, and audit redaction handling.",
            },
        ],
        "per_channel_review": channel_rows,
        "egress_enablement_gates": [
            "review_status must be approved in a separate operator decision artifact.",
            "real_egress_enabled must remain false in generated artifacts until credentials and runtime policy are bound.",
            "Every live outbound event must reference source_audit_id, route_id, actor_kind, redaction_profile, and payload_digest.",
            "Customer-visible outbound content must use customer_visible_outbound or approved_phone_handoff_call.",
            "Sensitive context replay must use sensitive_context_replay with dual approval.",
            "Payment, provisioning, credential, and account mutation intents must use spend_provisioning_or_credential and deny/escalate by default.",
            "Discord, WhatsApp, SMS, and phone sends must have a post-action receipt or blocked-action audit event.",
        ],
        "operator_must_not": [
            "send Discord, WhatsApp, SMS, or phone traffic from this generated packet",
            "place a voice call without an approved_phone_handoff_call decision artifact",
            "spend money, provision services, retrieve credentials, or echo secrets from channel context",
            "store raw phone numbers, emails, payment cards, bearer tokens, or provider keys in review artifacts",
            "mark real_egress_enabled true by editing generated artifacts",
        ],
        "review_commands": [
            "uv run python scripts/voiceops_channel_policy.py --output-dir artifacts/voiceops-channel-policy/current",
            "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts --output-dir artifacts/voiceops-plan/current --package-audit",
        ],
    }


def validate_policy(policy: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    known_channels = set(CHANNEL_IDS)
    channel_ids = [str(channel.get("channel_id") or "") for channel in policy.get("channel_authorization", [])]
    channels = set(channel_ids)
    missing_channels = known_channels - channels
    if missing_channels:
        issues.append(f"missing_channels:{','.join(sorted(missing_channels))}")
    unknown_channels = channels - known_channels
    if unknown_channels:
        issues.append(f"unknown_channels:{','.join(sorted(unknown_channels))}")
    duplicate_channels = sorted(channel_id for channel_id in channels if channel_ids.count(channel_id) > 1)
    if duplicate_channels:
        issues.append(f"duplicate_channels:{','.join(duplicate_channels)}")

    blocked_capabilities = set(policy.get("scope", {}).get("blocked_capabilities") or [])
    scope = policy.get("scope", {})
    if scope.get("review_required_for_real_egress") is not True:
        issues.append("missing_real_egress_review_gate")
    if scope.get("review_status") != "pending_human_review":
        issues.append("invalid_review_status")
    if scope.get("real_egress_enabled") is not False:
        issues.append("real_egress_enabled_without_review")
    missing_blocked_capabilities = REQUIRED_SCOPE_BLOCKED_CAPABILITIES - blocked_capabilities
    if missing_blocked_capabilities:
        issues.append(f"missing_blocked_capabilities:{','.join(sorted(missing_blocked_capabilities))}")

    for channel in policy.get("channel_authorization", []):
        channel_id = str(channel.get("channel_id") or "")
        if channel.get("audit_required") is not True:
            issues.append(f"audit_not_required:{channel_id}")
        approval_required = set(channel.get("approval_required_for") or [])
        if not approval_required:
            issues.append(f"missing_approval_requirements:{channel_id}")
        missing_approval = REQUIRED_APPROVAL_REQUIREMENTS.get(channel_id, set()) - approval_required
        if missing_approval:
            issues.append(f"missing_approval_requirements:{channel_id}:{','.join(sorted(missing_approval))}")
        evidence_required = set(channel.get("evidence_required") or [])
        if not evidence_required:
            issues.append(f"missing_evidence_requirements:{channel_id}")
        missing_evidence = REQUIRED_EVIDENCE_REQUIREMENTS.get(channel_id, set()) - evidence_required
        if missing_evidence:
            issues.append(f"missing_evidence_requirements:{channel_id}:{','.join(sorted(missing_evidence))}")
        if channel_id in REQUIRED_EVIDENCE_REQUIREMENTS and "source_audit_id" not in evidence_required:
            issues.append(f"missing_source_audit_id_evidence:{channel_id}")
        prohibited = set(channel.get("prohibited_actions") or [])
        missing_prohibited = REQUIRED_PROHIBITED_ACTIONS.get(channel_id, set()) - prohibited
        if missing_prohibited:
            issues.append(f"missing_prohibited_actions:{channel_id}:{','.join(sorted(missing_prohibited))}")
        allowed_prohibited_overlap = prohibited & set(channel.get("allowed_actions") or [])
        if allowed_prohibited_overlap:
            issues.append(f"allowed_action_is_prohibited:{channel_id}:{','.join(sorted(allowed_prohibited_overlap))}")

    if not policy.get("approval_routing"):
        issues.append("missing_approval_routing")
    else:
        routes = {route.get("route_id"): route for route in policy.get("approval_routing", [])}
        approval_route_map = policy.get("approval_route_map") if isinstance(policy.get("approval_route_map"), dict) else {}
        unknown_route_map_channels = sorted(set(approval_route_map) - known_channels)
        if unknown_route_map_channels:
            issues.append(f"approval_route_map_unknown_channels:{','.join(unknown_route_map_channels)}")
        for channel in policy.get("channel_authorization", []):
            channel_id = str(channel.get("channel_id") or "")
            if channel_id not in known_channels:
                continue
            channel_route_map = approval_route_map.get(channel_id)
            if not isinstance(channel_route_map, dict):
                channel_route_map = {}
            approval_required = {str(item) for item in channel.get("approval_required_for") or []}
            extra_route_items = sorted(set(channel_route_map) - approval_required)
            if extra_route_items:
                issues.append(f"approval_route_map_extra_items:{channel_id}:{','.join(extra_route_items)}")
            for approval_item in channel.get("approval_required_for") or []:
                route_id = str(channel_route_map.get(approval_item) or "")
                if not route_id:
                    issues.append(f"missing_approval_route_map:{channel_id}:{approval_item}")
                    continue
                route = routes.get(route_id)
                if route is None:
                    if route_id not in REQUIRED_APPROVAL_ROUTES:
                        issues.append(f"approval_route_map_unknown_route:{channel_id}:{approval_item}:{route_id}")
                    continue
                if channel_id not in set(route.get("applies_to") or []):
                    issues.append(f"approval_route_map_route_not_applicable:{channel_id}:{approval_item}:{route_id}")
        for required_route, contract in REQUIRED_APPROVAL_ROUTES.items():
            route = routes.get(required_route)
            if route is None:
                issues.append(f"missing_approval_route:{required_route}")
                continue
            actual_channels = set(route.get("applies_to") or [])
            expected_channels = set(contract["applies_to"])
            if actual_channels != expected_channels:
                issues.append(f"unsafe_route_channels:{required_route}:{','.join(sorted(actual_channels))}")
            if route.get("default_decision") != contract["default_decision"]:
                issues.append(f"unsafe_route_decision:{required_route}")
            try:
                approval_count = int(route.get("required_approval_count"))
            except (TypeError, ValueError):
                approval_count = -1
            if approval_count < int(contract["min_approval_count"]):
                issues.append(f"unsafe_route_approval_count:{required_route}")
            if route.get("escalation_level") != contract["escalation_level"]:
                issues.append(f"unsafe_route_escalation:{required_route}")
            missing_approvers = set(contract.get("approver_roles", set())) - set(route.get("approver_roles") or [])
            if missing_approvers:
                issues.append(f"missing_route_approvers:{required_route}:{','.join(sorted(missing_approvers))}")

        escalation_levels = {
            str(step.get("level") or "") for step in policy.get("escalation_policy", []) if step.get("level")
        }
        for route in policy.get("approval_routing", []):
            route_id = str(route.get("route_id") or "unknown")
            audit_event = str(route.get("audit_event") or "").strip()
            if not audit_event:
                issues.append(f"approval_route_missing_audit_event:{route_id}")
            elif not AUDIT_EVENT_PATTERN.match(audit_event):
                issues.append(f"approval_route_invalid_audit_event:{route_id}")
            escalation_level = str(route.get("escalation_level") or "")
            if escalation_level and escalation_level != "none" and escalation_level not in escalation_levels:
                issues.append(f"approval_route_missing_escalation_level:{route_id}:{escalation_level}")
            trigger = str(route.get("trigger") or "").lower()
            if any(term in trigger for term in HIGH_RISK_ROUTE_TRIGGER_TERMS):
                try:
                    approval_count = int(route.get("required_approval_count"))
                except (TypeError, ValueError):
                    approval_count = -1
                decision = str(route.get("default_decision") or "")
                if decision != "deny_and_escalate":
                    issues.append(f"unsafe_high_risk_route_decision:{route_id}")
                if approval_count < 2:
                    issues.append(f"unsafe_high_risk_route_approval_count:{route_id}")

    if not policy.get("escalation_policy"):
        issues.append("missing_escalation_policy")
    else:
        escalations = {step.get("level"): step for step in policy.get("escalation_policy", [])}
        for required_level, contract in REQUIRED_ESCALATIONS.items():
            step = escalations.get(required_level)
            if step is None:
                issues.append(f"missing_escalation_level:{required_level}")
                continue
            if step.get("destination_role") != contract["destination_role"]:
                issues.append(f"escalation_wrong_destination:{required_level}:{step.get('destination_role')}")
            missing_actions = set(contract["permitted_actions"]) - set(step.get("permitted_actions") or [])
            if missing_actions:
                issues.append(f"escalation_missing_actions:{required_level}:{','.join(sorted(missing_actions))}")
        level_3 = escalations.get("level_3")
        if level_3:
            for action in level_3.get("permitted_actions") or []:
                action_text = str(action).lower()
                if any(term in action_text for term in LEVEL_3_FORBIDDEN_ACTION_TERMS):
                    issues.append(f"unsafe_escalation_action:level_3:{action}")

    audit = policy.get("audit_id_continuity", {})
    if not audit.get("rules"):
        issues.append("missing_audit_id_continuity")
    audit_id_format = str(audit.get("audit_id_format") or "")
    if audit_id_format != REQUIRED_AUDIT_ID_FORMAT:
        issues.append("invalid_audit_id_format")
    missing_format_fields = [
        field for field in sorted(REQUIRED_AUDIT_ID_FORMAT_FIELDS) if "{" + field + "}" not in audit_id_format
    ]
    if missing_format_fields:
        issues.append(f"missing_audit_id_format_fields:{','.join(missing_format_fields)}")
    missing_audit_fields = REQUIRED_AUDIT_FIELDS - set(audit.get("required_fields") or [])
    if missing_audit_fields:
        issues.append(f"missing_audit_fields:{','.join(sorted(missing_audit_fields))}")
    audit_rules = [str(rule).lower() for rule in audit.get("rules") or []]
    for requirement_id, required_terms in AUDIT_RULE_REQUIRED_TERMS.items():
        if not any(all(term in rule for term in required_terms) for rule in audit_rules):
            issues.append(f"missing_audit_rule:{requirement_id}")

    redaction_rules = policy.get("redaction_rules") or []
    if not redaction_rules:
        issues.append("missing_redaction_rules")
    else:
        missing_redaction_rules = REQUIRED_REDACTION_RULES - {rule.get("rule_id") for rule in redaction_rules}
        if missing_redaction_rules:
            issues.append(f"missing_redaction_rules:{','.join(sorted(missing_redaction_rules))}")
        rule_ids = [str(rule.get("rule_id") or "") for rule in redaction_rules]
        if "payment_card_like" in rule_ids and "phone_number" in rule_ids:
            if rule_ids.index("payment_card_like") > rule_ids.index("phone_number"):
                issues.append("unsafe_redaction_order:payment_card_like_after_phone_number")
        for rule in redaction_rules:
            rule_id = str(rule.get("rule_id") or "unknown")
            applies_to = set(rule.get("applies_to") or [])
            invalid_channels = applies_to - known_channels
            if invalid_channels:
                issues.append(f"redaction_rule_invalid_channels:{rule_id}:{','.join(sorted(invalid_channels))}")
            if not applies_to:
                issues.append(f"redaction_rule_missing_channels:{rule_id}")
            try:
                re.compile(str(rule.get("pattern") or ""))
            except re.error:
                issues.append(f"redaction_rule_invalid_pattern:{rule_id}")

    mode = policy.get("mode", {})
    for key in ("headless", "bounded", "artifact_only"):
        if mode.get(key) is not True:
            issues.append(f"unsafe_mode:{key}")
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


def _review_markdown(review: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Milestone 3 Channel Policy Review",
        "",
        f"- Artifact ID: {review['artifact_id']}",
        f"- Schema version: {review['schema_version']}",
        f"- Policy ref: `{review['policy_ref']}`",
        f"- Review status: {review['review_status']}",
        f"- Real egress enabled: {review['real_egress_enabled']}",
        f"- Changes policy: {review['changes_policy']}",
        "- Scope: artifact-only signoff packet; it does not enable Discord, WhatsApp, SMS, or phone egress",
        "",
        "## Required Signoffs",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Role", "Required", "Reason"],
            [
                [signoff["role"], str(signoff["required"]).lower(), signoff["reason"]]
                for signoff in review["required_signoffs"]
            ],
        )
    )
    lines.extend(["", "## Per-Channel Review", ""])
    for channel in review["per_channel_review"]:
        lines.extend(
            [
                f"### {channel['display_name']}",
                "",
                f"- Channel ID: `{channel['channel_id']}`",
                f"- Review status: {channel['review_status']}",
                f"- Live egress enabled: {channel['live_egress_enabled']}",
                f"- Required evidence: {', '.join(channel['required_evidence'])}",
                f"- Blocked capabilities: {', '.join(channel['blocked_capabilities_to_confirm'])}",
                "- Approval routes:",
            ]
        )
        for approval_item, route_id in channel["approval_routes_to_confirm"].items():
            lines.append(f"  - {approval_item}: `{route_id}`")
        lines.append("- Checklist:")
        for item in channel["checklist"]:
            lines.append(f"  - {item}")
        lines.append("")
    lines.extend(["## Egress Enablement Gates", ""])
    for gate in review["egress_enablement_gates"]:
        lines.append(f"- {gate}")
    lines.extend(["", "## Operator Must Not", ""])
    for item in review["operator_must_not"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Review Commands", ""])
    for command in review["review_commands"]:
        lines.append(f"- `{command}`")
    lines.append("")
    return "\n".join(lines)


def write_channel_policy(output_dir: Path, policy: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    review = build_review_packet(policy)
    paths = {
        "json": output_dir / "channel-policy.json",
        "markdown": output_dir / "channel-policy.md",
        "review_json": output_dir / "channel-policy-review.json",
        "review_markdown": output_dir / "channel-policy-review.md",
    }
    _write_json(paths["json"], policy)
    paths["markdown"].write_text(_markdown(policy), encoding="utf-8")
    _write_json(paths["review_json"], review)
    paths["review_markdown"].write_text(_review_markdown(review), encoding="utf-8")
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
