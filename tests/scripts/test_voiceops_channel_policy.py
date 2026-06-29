from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.voiceops_channel_policy import (
    DEFAULT_OUTPUT_DIR,
    apply_redactions,
    build_channel_policy,
    parse_args,
    validate_policy,
    write_channel_policy,
)


def test_channel_policy_defines_required_channels_and_boundaries():
    policy = build_channel_policy()
    channels = {channel["channel_id"]: channel for channel in policy["channel_authorization"]}

    assert policy["artifact_version"] == "voiceops.multi_channel_policy.v1"
    assert policy["policy_id"] == "voiceops-m3-channel-policy"
    assert policy["milestone"] == "milestone_3_multi_channel_policy"
    assert policy["mode"] == {
        "artifact_only": True,
        "bounded": True,
        "env_secret_reads": False,
        "headless": True,
        "network_io": False,
        "outbound_calls": False,
        "outbound_sends": False,
    }
    assert set(channels) == {"discord", "whatsapp", "phone_sms"}
    assert channels["discord"]["authorization_mode"] == "operator_or_service_webhook_only"
    assert "server_admin_mutation" in channels["discord"]["prohibited_actions"]
    assert "any_customer_visible_send" in channels["whatsapp"]["approval_required_for"]
    assert channels["phone_sms"]["authorization_mode"] == "sms_and_approved_voice_handoff_only"
    assert "unapproved_voice_call" in channels["phone_sms"]["prohibited_actions"]
    assert "queue_approved_phone_handoff_call" in channels["phone_sms"]["allowed_actions"]
    assert "approved_phone_handoff_call" in channels["phone_sms"]["approval_required_for"]
    assert "phone_context_ref" in channels["phone_sms"]["evidence_required"]
    assert "any_sms_send" in channels["phone_sms"]["approval_required_for"]
    assert all(channel["audit_required"] is True for channel in channels.values())


def test_channel_policy_contains_approval_escalation_audit_and_redaction_rules():
    policy = build_channel_policy()
    routes = {route["route_id"]: route for route in policy["approval_routing"]}
    escalations = {step["level"]: step for step in policy["escalation_policy"]}
    redactions = {rule["rule_id"]: rule for rule in policy["redaction_rules"]}

    assert routes["customer_visible_outbound"]["default_decision"] == "hold_for_human_approval"
    assert routes["customer_visible_outbound"]["required_approval_count"] == 1
    assert routes["approved_phone_handoff_call"]["default_decision"] == "hold_for_human_approval"
    assert routes["approved_phone_handoff_call"]["required_approval_count"] == 1
    assert routes["approved_phone_handoff_call"]["applies_to"] == ["phone_sms"]
    assert routes["spend_provisioning_or_credential"]["default_decision"] == "deny_and_escalate"
    assert routes["spend_provisioning_or_credential"]["escalation_level"] == "level_3"
    assert escalations["level_2"]["destination_role"] == "incident_commander"
    assert "preserve_audit_chain" in escalations["level_2"]["permitted_actions"]
    assert "parent_audit_id" in policy["audit_id_continuity"]["required_fields"]
    assert "source_audit_id" in policy["audit_id_continuity"]["required_fields"]
    assert any("Cross-channel handoff" in rule for rule in policy["audit_id_continuity"]["rules"])
    assert {
        "env_assignment_secret",
        "bearer_token",
        "secret_key_like",
        "phone_number",
        "email_address",
        "payment_card_like",
    } <= set(redactions)


def test_redaction_rules_mask_channel_sensitive_values():
    text = (
        "Bearer token_123456789abcdef, sk_live_123456789abcdef, "
        "+15551234567, user@example.com, 4242 4242 4242 4242, "
        "DISCORD_BOT_TOKEN=discord-secret-token WHATSAPP_TOKEN=EAAG123456789abcdef "
        "TWILIO_AUTH_TOKEN=0123456789abcdef0123456789abcdef"
    )

    redacted = apply_redactions(text)

    assert "token_123456789abcdef" not in redacted
    assert "sk_live_123456789abcdef" not in redacted
    assert "+15551234567" not in redacted
    assert "user@example.com" not in redacted
    assert "4242 4242 4242 4242" not in redacted
    assert "discord-secret-token" not in redacted
    assert "EAAG123456789abcdef" not in redacted
    assert "0123456789abcdef0123456789abcdef" not in redacted
    assert "Bearer <redacted>" in redacted
    assert "DISCORD_BOT_TOKEN=<redacted>" in redacted
    assert "WHATSAPP_TOKEN=<redacted>" in redacted
    assert "TWILIO_AUTH_TOKEN=<redacted>" in redacted
    assert "<redacted-secret>" in redacted
    assert "<redacted-phone>" in redacted
    assert "<redacted-email>" in redacted
    assert "<redacted-card>" in redacted


def test_channel_policy_validates_safety_invariants():
    policy = build_channel_policy()

    assert validate_policy(policy) == []

    unsafe = json.loads(json.dumps(policy))
    unsafe["mode"]["network_io"] = True
    unsafe["channel_authorization"] = [
        channel for channel in unsafe["channel_authorization"] if channel["channel_id"] != "whatsapp"
    ]

    assert validate_policy(unsafe) == ["missing_channels:whatsapp", "unsafe_mode:network_io"]

    missing_blocks = json.loads(json.dumps(policy))
    missing_blocks["channel_authorization"][0]["prohibited_actions"] = []
    assert validate_policy(missing_blocks) == [
        "missing_prohibited_actions:discord:credential_or_secret_echo,payment_or_provisioning_action"
    ]

    missing_phone_route = json.loads(json.dumps(policy))
    missing_phone_route["approval_routing"] = [
        route for route in missing_phone_route["approval_routing"] if route["route_id"] != "approved_phone_handoff_call"
    ]
    assert validate_policy(missing_phone_route) == ["missing_approval_route:approved_phone_handoff_call"]

    missing_audit = json.loads(json.dumps(policy))
    missing_audit["audit_id_continuity"]["required_fields"] = ["audit_id"]
    assert validate_policy(missing_audit) == [
        "missing_audit_fields:channel_id,parent_audit_id,payload_digest,route_id,source_audit_id"
    ]

    missing_redactions = json.loads(json.dumps(policy))
    missing_redactions["redaction_rules"] = [
        rule for rule in missing_redactions["redaction_rules"] if rule["rule_id"] != "phone_number"
    ]
    assert validate_policy(missing_redactions) == ["missing_redaction_rules:phone_number"]


def test_write_channel_policy_artifacts(tmp_path):
    policy = build_channel_policy()
    paths = write_channel_policy(tmp_path, policy)

    assert set(paths) == {"json", "markdown"}
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert payload["scope"]["default_output_dir"] == str(DEFAULT_OUTPUT_DIR)
    assert payload["mode"]["outbound_calls"] is False
    assert payload["policy_id"] == "voiceops-m3-channel-policy"
    assert "VoiceOps Milestone 3 Channel Policy" in markdown
    assert "Policy ID" in markdown
    assert "Channel Authorization" in markdown
    assert "Audit ID Continuity" in markdown
    assert "Redaction Rules" in markdown


def test_channel_policy_cli_smoke(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "voiceops_channel_policy.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["validation_issues"] == []
    assert Path(payload["artifacts"]["json"]).exists()
    assert Path(payload["artifacts"]["markdown"]).exists()


def test_parse_args_defaults_to_requested_artifact_dir():
    args = parse_args([])

    assert args.output_dir == DEFAULT_OUTPUT_DIR