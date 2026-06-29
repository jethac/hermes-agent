from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from scripts.voiceops_channel_policy import (
    DEFAULT_OUTPUT_DIR,
    apply_redactions,
    build_channel_policy,
    build_review_packet,
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
    assert policy["scope"]["review_required_for_real_egress"] is True
    assert policy["scope"]["review_status"] == "pending_human_review"
    assert policy["scope"]["real_egress_enabled"] is False
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
    route_map = policy["approval_route_map"]
    escalations = {step["level"]: step for step in policy["escalation_policy"]}
    redactions = {rule["rule_id"]: rule for rule in policy["redaction_rules"]}

    for channel in policy["channel_authorization"]:
        channel_id = channel["channel_id"]
        assert set(channel["approval_required_for"]) <= set(route_map[channel_id])
        for approval_item in channel["approval_required_for"]:
            route_id = route_map[channel_id][approval_item]
            assert route_id in routes
            assert channel_id in routes[route_id]["applies_to"]
    assert route_map["phone_sms"]["approved_phone_handoff_call"] == "approved_phone_handoff_call"
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
        "sensitive_url",
        "bearer_token",
        "discord_bot_token",
        "twilio_auth_token",
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
        "TWILIO_AUTH_TOKEN=0123456789abcdef0123456789abcdef "
        "https://discord.com/api/webhooks/123456789012345678/abcdefghijklmnopqrstuvwxyzABCDEF "
        "https://checkout.stripe.com/c/pay/cs_live_123456789 https://buy.stripe.com/test_123456789"
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
    assert "discord.com/api/webhooks" not in redacted
    assert "checkout.stripe.com" not in redacted
    assert "buy.stripe.com" not in redacted
    assert "Bearer <redacted>" in redacted
    assert "DISCORD_BOT_TOKEN=<redacted>" in redacted
    assert "WHATSAPP_TOKEN=<redacted>" in redacted
    assert "TWILIO_AUTH_TOKEN=<redacted>" in redacted
    assert "<redacted-url>" in redacted
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
        "missing_prohibited_actions:discord:credential_or_secret_echo,payment_or_provisioning_action,server_admin_mutation,unapproved_external_invite"
    ]

    missing_phone_route = json.loads(json.dumps(policy))
    missing_phone_route["approval_routing"] = [
        route for route in missing_phone_route["approval_routing"] if route["route_id"] != "approved_phone_handoff_call"
    ]
    assert validate_policy(missing_phone_route) == ["missing_approval_route:approved_phone_handoff_call"]

    missing_audit = json.loads(json.dumps(policy))
    missing_audit["audit_id_continuity"]["required_fields"] = ["audit_id"]
    assert validate_policy(missing_audit) == [
        "missing_audit_fields:actor_kind,channel_id,parent_audit_id,payload_digest,redaction_profile,route_id,source_audit_id"
    ]

    missing_redactions = json.loads(json.dumps(policy))
    missing_redactions["redaction_rules"] = [
        rule for rule in missing_redactions["redaction_rules"] if rule["rule_id"] != "phone_number"
    ]
    assert validate_policy(missing_redactions) == ["missing_redaction_rules:phone_number"]


def test_channel_policy_rejects_non_artifact_mode_flags():
    policy = build_channel_policy()

    for key, value in {
        "headless": False,
        "bounded": False,
        "artifact_only": False,
        "outbound_sends": True,
        "outbound_calls": True,
        "env_secret_reads": True,
        "network_io": True,
    }.items():
        unsafe = json.loads(json.dumps(policy))
        unsafe["mode"][key] = value
        assert validate_policy(unsafe) == [f"unsafe_mode:{key}"]


def test_channel_policy_rejects_missing_blocked_live_capabilities():
    policy = build_channel_policy()
    unsafe = json.loads(json.dumps(policy))
    unsafe["scope"]["blocked_capabilities"].remove("payment_or_provisioning_action")

    assert validate_policy(unsafe) == ["missing_blocked_capabilities:payment_or_provisioning_action"]

    missing_review = json.loads(json.dumps(policy))
    missing_review["scope"]["review_required_for_real_egress"] = False
    missing_review["scope"]["review_status"] = "approved"
    missing_review["scope"]["real_egress_enabled"] = True
    assert validate_policy(missing_review) == [
        "missing_real_egress_review_gate",
        "invalid_review_status",
        "real_egress_enabled_without_review",
    ]


def test_channel_policy_rejects_unknown_or_duplicate_channels():
    policy = build_channel_policy()

    duplicate = json.loads(json.dumps(policy))
    duplicate["channel_authorization"].append(duplicate["channel_authorization"][0])
    assert validate_policy(duplicate) == ["duplicate_channels:discord"]

    unknown = json.loads(json.dumps(policy))
    unknown["channel_authorization"].append({**unknown["channel_authorization"][0], "channel_id": "telegram"})
    assert validate_policy(unknown) == ["unknown_channels:telegram"]


def test_channel_policy_rejects_allowed_actions_that_are_prohibited():
    policy = build_channel_policy()
    unsafe = json.loads(json.dumps(policy))
    unsafe["channel_authorization"][1]["allowed_actions"].append("payment_link_send")

    assert validate_policy(unsafe) == ["allowed_action_is_prohibited:whatsapp:payment_link_send"]


def test_channel_policy_validates_channel_specific_required_boundaries():
    policy = build_channel_policy()

    unsafe = json.loads(json.dumps(policy))
    unsafe["channel_authorization"][2]["approval_required_for"].remove("any_sms_send")
    assert validate_policy(unsafe) == ["missing_approval_requirements:phone_sms:any_sms_send"]

    unsafe = json.loads(json.dumps(policy))
    unsafe["channel_authorization"][1]["prohibited_actions"].remove("payment_link_send")
    assert validate_policy(unsafe) == ["missing_prohibited_actions:whatsapp:payment_link_send"]

    unsafe = json.loads(json.dumps(policy))
    unsafe["channel_authorization"][0]["evidence_required"].remove("source_audit_id")
    assert validate_policy(unsafe) == [
        "missing_evidence_requirements:discord:source_audit_id",
        "missing_source_audit_id_evidence:discord",
    ]


def test_channel_policy_validates_spend_provisioning_route_semantics():
    policy = build_channel_policy()
    unsafe = json.loads(json.dumps(policy))
    route = next(route for route in unsafe["approval_routing"] if route["route_id"] == "spend_provisioning_or_credential")
    route["default_decision"] = "hold_for_human_approval"
    route["required_approval_count"] = 1
    route["escalation_level"] = "level_1"
    route["approver_roles"] = ["operator_on_call"]

    assert validate_policy(unsafe) == [
        "unsafe_route_decision:spend_provisioning_or_credential",
        "unsafe_route_approval_count:spend_provisioning_or_credential",
        "unsafe_route_escalation:spend_provisioning_or_credential",
        "missing_route_approvers:spend_provisioning_or_credential:business_owner,security_owner",
        "unsafe_high_risk_route_decision:spend_provisioning_or_credential",
        "unsafe_high_risk_route_approval_count:spend_provisioning_or_credential",
    ]


def test_channel_policy_validates_phone_handoff_route_is_phone_only():
    policy = build_channel_policy()
    unsafe = json.loads(json.dumps(policy))
    route = next(route for route in unsafe["approval_routing"] if route["route_id"] == "approved_phone_handoff_call")
    route["applies_to"] = ["discord", "phone_sms"]

    assert validate_policy(unsafe) == ["unsafe_route_channels:approved_phone_handoff_call:discord,phone_sms"]


def test_channel_policy_validates_approval_route_map_coverage():
    policy = build_channel_policy()

    missing = json.loads(json.dumps(policy))
    missing["approval_route_map"]["phone_sms"].pop("approved_phone_handoff_call")
    assert validate_policy(missing) == ["missing_approval_route_map:phone_sms:approved_phone_handoff_call"]

    unknown = json.loads(json.dumps(policy))
    unknown["approval_route_map"]["discord"]["customer_visible_message"] = "missing-route"
    assert validate_policy(unknown) == [
        "approval_route_map_unknown_route:discord:customer_visible_message:missing-route"
    ]

    wrong_channel = json.loads(json.dumps(policy))
    wrong_channel["approval_route_map"]["discord"]["customer_visible_message"] = "approved_phone_handoff_call"
    assert validate_policy(wrong_channel) == [
        "approval_route_map_route_not_applicable:discord:customer_visible_message:approved_phone_handoff_call"
    ]


def test_channel_policy_rejects_execution_permissions_in_level_3_escalation():
    policy = build_channel_policy()
    unsafe = json.loads(json.dumps(policy))
    level_3 = next(step for step in unsafe["escalation_policy"] if step["level"] == "level_3")
    level_3["permitted_actions"].append("execute_provisioning")

    assert validate_policy(unsafe) == ["unsafe_escalation_action:level_3:execute_provisioning"]


def test_channel_policy_requires_full_audit_continuity_fields_and_rules():
    policy = build_channel_policy()
    unsafe = json.loads(json.dumps(policy))
    unsafe["audit_id_continuity"]["required_fields"].remove("actor_kind")
    unsafe["audit_id_continuity"]["required_fields"].remove("redaction_profile")

    assert validate_policy(unsafe) == ["missing_audit_fields:actor_kind,redaction_profile"]

    unsafe = json.loads(json.dumps(policy))
    unsafe["audit_id_continuity"]["rules"] = ["Never overwrite an existing audit_id."]
    assert validate_policy(unsafe) == [
        "missing_audit_rule:outbound",
        "missing_audit_rule:escalation",
        "missing_audit_rule:redaction",
        "missing_audit_rule:cross_channel",
    ]

    unsafe = json.loads(json.dumps(policy))
    unsafe["audit_id_continuity"]["audit_id_format"] = "vops-m3-{channel_id}-{sequence}"
    assert validate_policy(unsafe) == [
        "invalid_audit_id_format",
        "missing_audit_id_format_fields:utc_yyyymmddThhmmssZ",
    ]

    unsafe = json.loads(json.dumps(policy))
    unsafe["approval_routing"][0]["audit_event"] = ""
    assert validate_policy(unsafe) == ["approval_route_missing_audit_event:status_only"]

    unsafe = json.loads(json.dumps(policy))
    unsafe["approval_routing"][0]["audit_event"] = "bad event"
    assert validate_policy(unsafe) == ["approval_route_invalid_audit_event:status_only"]


def test_redaction_rules_compile_only_apply_to_known_channels_and_keep_safe_order():
    policy = build_channel_policy()

    for rule in policy["redaction_rules"]:
        re.compile(rule["pattern"])
        assert set(rule["applies_to"]) <= {"discord", "whatsapp", "phone_sms"}

    unsafe = json.loads(json.dumps(policy))
    rules = unsafe["redaction_rules"]
    card_index = next(index for index, rule in enumerate(rules) if rule["rule_id"] == "payment_card_like")
    phone_index = next(index for index, rule in enumerate(rules) if rule["rule_id"] == "phone_number")
    rules[card_index], rules[phone_index] = rules[phone_index], rules[card_index]
    assert validate_policy(unsafe) == ["unsafe_redaction_order:payment_card_like_after_phone_number"]

    unsafe = json.loads(json.dumps(policy))
    unsafe["redaction_rules"][0]["applies_to"] = ["telegram"]
    assert validate_policy(unsafe) == ["redaction_rule_invalid_channels:env_assignment_secret:telegram"]

    unsafe = json.loads(json.dumps(policy))
    unsafe["redaction_rules"][0]["pattern"] = "["
    assert validate_policy(unsafe) == ["redaction_rule_invalid_pattern:env_assignment_secret"]


def test_redaction_rules_mask_raw_provider_tokens_and_payment_urls():
    discord_token = "MTAxMjM0NTY3ODkwMTIzNDU2Nw" + ".Gabcdef." + "abcdefghijklmnopqrstuvwxyzABCDE"
    twilio_sid = "AC" + "0123456789abcdef0123456789abcdef"
    twilio_auth_token = "0123456789abcdef0123456789abcdef"
    text = (
        "discord webhook https://discord.com/api/webhooks/123456789012345678/"
        "abcdefghijklmnopqrstuvwxyzABCDEF "
        f"discord token {discord_token} "
        "whatsapp EAAGm0PX4ZCpsBANZCZA1234567890 "
        f"twilio {twilio_sid} "
        f"twilio auth token {twilio_auth_token} "
        "stripe sk_live_51ABCDEF123456789 whsec_123456789abcdef "
        "checkout https://checkout.stripe.com/c/pay/cs_live_123456789 "
        "link https://buy.stripe.com/test_123456789"
    )

    redacted = apply_redactions(text)

    assert "discord.com/api/webhooks" not in redacted
    assert discord_token not in redacted
    assert "EAAGm0PX4ZCpsBANZCZA1234567890" not in redacted
    assert twilio_sid not in redacted
    assert twilio_auth_token not in redacted
    assert "sk_live_51ABCDEF123456789" not in redacted
    assert "whsec_123456789abcdef" not in redacted
    assert "checkout.stripe.com" not in redacted
    assert "buy.stripe.com" not in redacted


def test_redaction_rule_order_masks_cards_before_phone_numbers():
    redacted = apply_redactions("card 4242 4242 4242 4242 phone +1 (555) 123-4567")

    assert "<redacted-card>" in redacted
    assert "<redacted-phone>" in redacted
    assert "4242" not in redacted
    assert "555" not in redacted


def test_write_channel_policy_artifacts(tmp_path):
    policy = build_channel_policy()
    paths = write_channel_policy(tmp_path, policy)

    assert set(paths) == {"json", "markdown", "review_json", "review_markdown"}
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    review_payload = json.loads(Path(paths["review_json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    review_markdown = Path(paths["review_markdown"]).read_text(encoding="utf-8")
    assert payload["scope"]["default_output_dir"] == str(DEFAULT_OUTPUT_DIR)
    assert payload["mode"]["outbound_calls"] is False
    assert payload["policy_id"] == "voiceops-m3-channel-policy"
    assert review_payload["schema_version"] == "voiceops.multi_channel_policy_review.v1"
    assert review_payload["policy_ref"] == "channel-policy.json"
    assert review_payload["review_status"] == "pending_human_review"
    assert review_payload["real_egress_enabled"] is False
    assert review_payload["changes_policy"] is False
    assert "approve_live_egress_after_external_credentials_are_bound" in review_payload["decision_options"]
    assert {signoff["role"] for signoff in review_payload["required_signoffs"]} == {
        "business_owner",
        "channel_owner",
        "privacy_reviewer",
        "security_owner",
    }
    channels = {channel["channel_id"]: channel for channel in review_payload["per_channel_review"]}
    assert set(channels) == {"discord", "whatsapp", "phone_sms"}
    assert channels["phone_sms"]["live_egress_enabled"] is False
    assert "approved_phone_handoff_call" in channels["phone_sms"]["approval_routes_to_confirm"]
    assert "unapproved_voice_call" in channels["phone_sms"]["blocked_capabilities_to_confirm"]
    assert any("source_audit_id" in gate for gate in review_payload["egress_enablement_gates"])
    assert any("mark real_egress_enabled true" in item for item in review_payload["operator_must_not"])
    assert "VoiceOps Milestone 3 Channel Policy" in markdown
    assert "Policy ID" in markdown
    assert "Channel Authorization" in markdown
    assert "Audit ID Continuity" in markdown
    assert "Redaction Rules" in markdown
    assert "VoiceOps Milestone 3 Channel Policy Review" in review_markdown
    assert "Required Signoffs" in review_markdown
    assert "Per-Channel Review" in review_markdown
    assert "Operator Must Not" in review_markdown


def test_channel_policy_review_packet_is_artifact_only_and_per_channel():
    policy = build_channel_policy()
    review = build_review_packet(policy)

    assert review["artifact_only"] is True
    assert review["real_egress_enabled"] is False
    assert review["changes_policy"] is False
    assert review["policy_id"] == policy["policy_id"]
    assert review["review_status"] == policy["scope"]["review_status"]
    assert all(channel["review_status"] == "pending_human_review" for channel in review["per_channel_review"])
    assert all(channel["live_egress_enabled"] is False for channel in review["per_channel_review"])
    assert any("voice call" in item for item in review["operator_must_not"])
    assert any("spend_provisioning_or_credential" in gate for gate in review["egress_enablement_gates"])


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
    assert Path(payload["artifacts"]["review_json"]).exists()
    assert Path(payload["artifacts"]["review_markdown"]).exists()


def test_parse_args_defaults_to_requested_artifact_dir():
    args = parse_args([])

    assert args.output_dir == DEFAULT_OUTPUT_DIR