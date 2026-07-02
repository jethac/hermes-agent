import hashlib
import json
from pathlib import Path

from scripts.voiceops_provisioning_probe import (
    build_milestone2_execution_plan,
    build_probe_report,
    load_post_approval_receipts,
)
from scripts.voiceops_stripe_executor import (
    LIVE_CONFIRMATION,
    CommandResult,
    execute_approved_actions,
)


def _plan():
    report = build_probe_report(env={}, env_files=[], which=lambda _command: None)
    return build_milestone2_execution_plan(report)


def _packet_from_plan(plan):
    actions = []
    for action in plan["approval_required_actions"]:
        command = action["command"]
        contract = {
            "approval_id": action["approval_id"],
            "action_id": action["action_id"],
            "approval_channel": "discord_voice_operator_confirmation",
            "approval_artifact": "nemoclaw-action-packet.json",
            "approved_by_ref": None,
            "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
            "required_preflight_gates": action["approval_contract"]["required_preflight_gates"],
            "allowed_decisions": ["approve_once", "deny", "hold"],
            "default_decision": "hold",
            "ttl_seconds": action["approval_contract"]["ttl_seconds"],
            "status": "pending",
        }
        actions.append(
            {
                "action_id": action["action_id"],
                "provider": action["provider"],
                "command": command,
                "purpose": f"test {action['action_id']}",
                "estimated_cents": 0,
                "requires_approval": True,
                "status": "queued",
                "approval_contract": contract,
                "lineage": dict(action["lineage"]),
            }
        )
    return {
        "schema_version": "voiceops.nemoclaw_action_packet.v1",
        "artifact_id": "voiceops-nemoclaw-action-packet",
        "packet_id": "voiceops-test-packet",
        "runtime": "NemoClaw",
        "mode": "dry_run_until_user_approval",
        "dry_run_shell_artifact": "stripe-actions-dry-run.sh",
        "audit_ledger_artifact": "audit-ledger.jsonl",
        "source_channel": "discord_voice",
        "hermes_active_model": "Nemotron 3 Super",
        "model_selected_by": "Hermes /model",
        "spend_policy": {"limit_cents": 20_000, "currency": "usd"},
        "safety": {
            "live_spend": False,
            "provider_provisioning": False,
            "credential_retrieval": False,
            "outbound_phone_calls": False,
            "network_io": False,
            "requires_operator_approval": True,
            "default_decision": "hold",
        },
        "allowed_capabilities": [
            "stripe_projects_voip_provisioning_after_approval",
            "stripe_link_spend_request_after_approval",
        ],
        "blocked_capabilities": [
            "raw_card_data_in_model_context",
            "unapproved_purchase",
            "unbounded_network_access",
        ],
        "approval_required_actions": actions,
        "approval_contracts": {action["action_id"]: action["approval_contract"] for action in actions},
        "dry_run_commands": [action["command"] for action in actions],
        "audit_event_ids": [],
    }


def _decisions(plan, approved_action_ids):
    decisions = []
    for action in plan["approval_required_actions"]:
        action_id = action["action_id"]
        decisions.append(
            {
                "action_id": action_id,
                "approval_id": action["approval_id"],
                "decision": "approve_once" if action_id in approved_action_ids else "hold",
                "decision_by": "operator-ref-test",
                "decision_at": "2026-06-29T00:00:00Z",
            }
        )
    return {
        "schema_version": "voiceops.milestone2.approval_decisions.v1",
        "redacted": True,
        "decisions": decisions,
    }


def test_stripe_executor_runs_only_approved_packet_commands_and_writes_valid_receipts(tmp_path):
    plan = _plan()
    packet = _packet_from_plan(plan)
    calls = []

    def runner(argv, _timeout_seconds):
        calls.append(list(argv))
        return CommandResult(exit_code=0, stdout=f"ok {' '.join(argv)}")

    report = execute_approved_actions(
        packet=packet,
        plan=plan,
        decisions_payload=_decisions(plan, {"provision-voip-provider", "buy-service-credit"}),
        output_dir=tmp_path,
        execute=True,
        confirmation=LIVE_CONFIRMATION,
        runner=runner,
        now=lambda: "2026-06-29T00:00:30Z",
    )

    assert report["ok"] is True
    assert calls == [
        ["stripe", "projects", "add", "twilio/voice"],
        [
            "link-cli",
            "spend-request",
            "create",
            "--merchant-name",
            "ExampleOps",
            "--merchant-url",
            "https://example.invalid",
            "--amount",
            "4900",
            "--request-approval",
        ],
    ]
    loaded = load_post_approval_receipts(tmp_path / "post-approval-receipts.json", plan)
    assert loaded["status"] == "valid"
    assert loaded["receipt_count"] == 4
    assert loaded["credential_location_count"] == 2
    assert loaded["rollback_receipt_count"] == 2
    receipts = json.loads((tmp_path / "post-approval-receipts.json").read_text(encoding="utf-8"))
    action_lineage = {
        action["action_id"]: action["lineage"]
        for action in plan["approval_required_actions"]
    }
    first_receipt = receipts["receipts"][0]
    first_lineage = action_lineage[first_receipt["action_id"]]
    assert {
        key: first_receipt[key]
        for key in ("source_voice_session_id", "source_oracle_job_id", "parent_audit_event_id")
    } == first_lineage
    assert receipts["audit_events"][0]["source_voice_session_id"] == first_lineage["source_voice_session_id"]
    assert receipts["credential_locations"][0]["lineage"] == first_lineage
    assert receipts["rollback_receipts"][0]["lineage"] == first_lineage


def test_stripe_executor_refuses_approve_once_without_execute(tmp_path):
    plan = _plan()
    packet = _packet_from_plan(plan)

    report = execute_approved_actions(
        packet=packet,
        plan=plan,
        decisions_payload=_decisions(plan, {"provision-voip-provider"}),
        output_dir=tmp_path,
        execute=False,
        confirmation=None,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=0),
        now=lambda: "2026-06-29T00:00:30Z",
    )

    assert report["ok"] is False
    assert "provision-voip-provider:approve_once_requires_execute" in report["issues"]
    assert all(result["executed"] is False for result in report["command_results"])


def test_stripe_executor_refuses_execute_with_invalid_confirmation(tmp_path):
    plan = _plan()
    packet = _packet_from_plan(plan)
    calls = []

    report = execute_approved_actions(
        packet=packet,
        plan=plan,
        decisions_payload=_decisions(plan, {"provision-voip-provider"}),
        output_dir=tmp_path,
        execute=True,
        confirmation="wrong-confirmation",
        runner=lambda argv, _timeout_seconds: calls.append(list(argv)) or CommandResult(exit_code=0),
        now=lambda: "2026-06-29T00:00:30Z",
    )

    assert report["ok"] is False
    assert "execute_confirmation_missing_or_invalid" in report["issues"]
    assert calls == []
    assert all(result["executed"] is False for result in report["command_results"])


def test_stripe_executor_refuses_execute_with_invalid_nemoclaw_packet(tmp_path):
    plan = _plan()
    packet = _packet_from_plan(plan)
    packet["safety"]["live_spend"] = True
    calls = []

    report = execute_approved_actions(
        packet=packet,
        plan=plan,
        decisions_payload=_decisions(plan, {"provision-voip-provider"}),
        output_dir=tmp_path,
        execute=True,
        confirmation=LIVE_CONFIRMATION,
        runner=lambda argv, _timeout_seconds: calls.append(list(argv)) or CommandResult(exit_code=0),
        now=lambda: "2026-06-29T00:00:30Z",
    )

    assert report["ok"] is False
    assert report["packet_validation_status"] == "invalid"
    assert any(issue.startswith("nemoclaw_action_packet:") for issue in report["issues"])
    assert calls == []
    assert all(result["executed"] is False for result in report["command_results"])


def test_stripe_executor_rejects_packet_plan_command_mismatch(tmp_path):
    plan = _plan()
    packet = _packet_from_plan(plan)
    packet["approval_required_actions"][0]["command"] = "stripe projects add neon/postgres"
    packet["approval_contracts"]["provision-voip-provider"]["command_sha256"] = hashlib.sha256(
        b"stripe projects add neon/postgres"
    ).hexdigest()
    packet["approval_required_actions"][0]["approval_contract"]["command_sha256"] = packet["approval_contracts"][
        "provision-voip-provider"
    ]["command_sha256"]
    packet["dry_run_commands"][0] = "stripe projects add neon/postgres"

    report = execute_approved_actions(
        packet=packet,
        plan=plan,
        decisions_payload=_decisions(plan, {"provision-voip-provider"}),
        output_dir=tmp_path,
        execute=True,
        confirmation=LIVE_CONFIRMATION,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=0),
        now=lambda: "2026-06-29T00:00:30Z",
    )

    assert report["ok"] is False
    assert "provision-voip-provider:packet_command_mismatch" in report["issues"]
    assert all(result["executed"] is False for result in report["command_results"])


def test_stripe_executor_rejects_plan_command_sha256_mismatch(tmp_path):
    plan = _plan()
    packet = _packet_from_plan(plan)
    plan["approval_required_actions"][0]["command_sha256"] = "0" * 64
    calls = []

    report = execute_approved_actions(
        packet=packet,
        plan=plan,
        decisions_payload=_decisions(plan, {"provision-voip-provider"}),
        output_dir=tmp_path,
        execute=True,
        confirmation=LIVE_CONFIRMATION,
        runner=lambda argv, _timeout_seconds: calls.append(list(argv)) or CommandResult(exit_code=0),
        now=lambda: "2026-06-29T00:00:30Z",
    )

    assert report["ok"] is False
    assert "provision-voip-provider:command_sha256_mismatch" in report["issues"]
    assert calls == []
    assert all(result["executed"] is False for result in report["command_results"])


def test_stripe_executor_cli_outputs_report(tmp_path, capsys):
    plan = _plan()
    packet = _packet_from_plan(plan)
    packet_path = tmp_path / "packet.json"
    plan_path = tmp_path / "plan.json"
    decisions_path = tmp_path / "decisions.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    decisions_path.write_text(json.dumps(_decisions(plan, set())), encoding="utf-8")

    from scripts.voiceops_stripe_executor import main

    assert main(
        [
            "--nemoclaw-action-packet",
            str(packet_path),
            "--execution-plan",
            str(plan_path),
            "--approval-decisions",
            str(decisions_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    ) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "pass"
    assert Path(printed["artifacts"]["post_approval_receipts"]).exists()
