from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.voiceops_operator_state import (
    DEFAULT_OUTPUT_DIR,
    build_operator_state,
    parse_args,
    validate_operator_state,
    write_operator_state,
)


def _sync_approval_contract(state, approval):
    contract = dict(approval["approval_contract"])
    contract["approval_id"] = approval["approval_id"]
    contract["action_id"] = approval["action_id"]
    contract["approval_artifact"] = approval["approval_artifact"]
    approval["approval_contract"] = contract
    state["approval_contracts"][approval["action_id"]] = contract


def test_operator_state_contains_required_dashboard_sections_and_boundaries():
    state = build_operator_state()

    assert state["schema_version"] == "voiceops.operator_state.v1"
    assert state["artifact_version"] == "voiceops.operator_state.v1"
    assert state["state_id"] == "voiceops-m5-operator-state"
    assert state["milestone"] == "milestone_5_operator_dashboard"
    assert state["current_mode"] == "approval-required"
    assert state["mode"] == {
        "artifact_only": True,
        "bounded": True,
        "env_secret_reads": False,
        "headless": True,
        "live_spend": False,
        "network_io": False,
        "outbound_calls": False,
        "outbound_sends": False,
        "provisioning": False,
    }
    assert state["scope"]["default_output_dir"] == str(DEFAULT_OUTPUT_DIR)
    assert "environment_secret_read" in state["scope"]["blocked_capabilities"]
    assert "spend" in state["scope"]["blocked_capabilities"]
    assert "service_provisioning" in state["scope"]["blocked_capabilities"]

    surface = state["active_voice_surface"]
    assert surface["surface_id"] == "discord_voice"
    assert surface["fallback_surface_id"] == "whatsapp_text"
    assert surface["fallback_reason"]

    budget = state["budget_status"]
    assert budget["status"] == "no_live_spend_without_explicit_approval"
    assert budget["approved_budget_cents"] == 20000
    assert budget["reserved_cents"] == 7400
    assert (
        budget["approved_budget_cents"] - budget["reserved_cents"] - budget["spent_cents"]
        == budget["remaining_cents"]
    )
    assert "approval_packet_required_for_any_spend" in budget["controls"]

    assert state["pending_approvals"]
    assert set(state["approval_contracts"]) == {approval["action_id"] for approval in state["pending_approvals"]}
    approval_ids = {approval["approval_id"] for approval in state["pending_approvals"]}
    for approval in state["pending_approvals"]:
        assert approval["action_id"]
        assert approval["provider"]
        assert approval["approval_artifact"] in {"channel-policy.json", "nemoclaw-action-packet.json"}
        assert approval["command"]
        assert approval["execution_status"] == "not_executed"
        assert approval["operator_next_step"]
        contract = approval["approval_contract"]
        assert contract == state["approval_contracts"][approval["action_id"]]
        assert contract["approval_id"] == approval["approval_id"]
        assert contract["action_id"] == approval["action_id"]
        assert len(contract["command_sha256"]) == 64
        assert contract["allowed_decisions"] == ["approve_once", "deny", "hold"]
        assert contract["approved_by_ref"] is None
        assert contract["required_preflight_gates"]
    assert state["recent_audit_events"]
    assert all(event["operator_next_step"] for event in state["recent_audit_events"])
    assert all(event["artifact_ref"] for event in state["recent_audit_events"])
    assert state["planned_services"]
    for service in state["planned_services"]:
        assert service["operator_next_step"]
        assert service["artifact_ref"]
        if service["external"]:
            assert service["execution_status"] == "not_executed"
        if service["approval_required"]:
            assert service["approval_ref"] in approval_ids
    assert state["provisioned_services"]
    assert {task["domain"] for task in state["upcoming_tasks"]} == {"household", "business"}
    assert len(state["pending_approvals"]) <= state["bounds"]["max_pending_approvals"]
    assert len(state["recent_audit_events"]) <= state["bounds"]["max_audit_events"]
    assert len(state["planned_services"]) <= state["bounds"]["max_services_per_section"]
    assert len(state["provisioned_services"]) <= state["bounds"]["max_services_per_section"]
    assert len(state["upcoming_tasks"]) <= state["bounds"]["max_upcoming_tasks"]


def test_operator_state_validates_safety_and_bounds():
    state = build_operator_state()

    assert validate_operator_state(state) == []

    unsafe = json.loads(json.dumps(state))
    unsafe["mode"]["network_io"] = True
    unsafe["mode"]["live_spend"] = True
    unsafe["active_voice_surface"]["fallback_reason"] = ""
    unsafe["budget_status"]["remaining_cents"] = 1
    unsafe["budget_status"]["current_mode"] = "dry-run"

    assert validate_operator_state(unsafe) == [
        "budget_remaining_mismatch",
        "current_mode_mismatch",
        "missing_fallback_reason",
        "unsafe_mode:live_spend",
        "unsafe_mode:network_io",
    ]

    malformed_budget = json.loads(json.dumps(state))
    malformed_budget["budget_status"]["approved_budget_cents"] = "20000"
    assert validate_operator_state(malformed_budget) == ["invalid_budget_amounts"]

    over_bounds = json.loads(json.dumps(state))
    over_bounds["approval_contracts"] = {}
    over_bounds["pending_approvals"] = [
        {
            **over_bounds["pending_approvals"][0],
            "approval_id": f"vops-m5-extra-{index:02d}",
            "action_id": f"extra-action-{index:02d}",
            "budget_impact_cents": 900 if index < 7 else 550,
            "category": "status",
        }
        for index in range(9)
    ]
    for approval in over_bounds["pending_approvals"]:
        _sync_approval_contract(over_bounds, approval)
    for service in over_bounds["planned_services"]:
        if service["approval_required"]:
            service["approval_ref"] = "vops-m5-extra-00"
    assert validate_operator_state(over_bounds) == ["bounds_exceeded:pending_approvals"]

    missing_business = json.loads(json.dumps(state))
    missing_business["upcoming_tasks"] = [
        task for task in missing_business["upcoming_tasks"] if task["domain"] == "household"
    ]
    assert validate_operator_state(missing_business) == ["missing_task_domain:business"]


def test_operator_state_validates_blocked_capabilities_and_budget_controls():
    state = build_operator_state()

    missing_block = json.loads(json.dumps(state))
    missing_block["scope"]["blocked_capabilities"].remove("spend")
    assert validate_operator_state(missing_block) == ["missing_blocked_capabilities:spend"]

    missing_control = json.loads(json.dumps(state))
    missing_control["budget_status"]["controls"].remove("approval_packet_required_for_any_spend")
    assert validate_operator_state(missing_control) == [
        "missing_budget_control:approval_packet_required_for_any_spend"
    ]

    mismatched_approvals = json.loads(json.dumps(state))
    mismatched_approvals["pending_approvals"][0]["budget_impact_cents"] = 1
    assert validate_operator_state(mismatched_approvals) == ["pending_approval_budget_mismatch"]


def test_operator_state_validates_pending_approval_contracts():
    state = build_operator_state()

    duplicate = json.loads(json.dumps(state))
    duplicate["pending_approvals"][1]["approval_id"] = duplicate["pending_approvals"][0]["approval_id"]
    _sync_approval_contract(duplicate, duplicate["pending_approvals"][1])
    assert validate_operator_state(duplicate) == ["duplicate_pending_approval_ids:vops-m5-approval-001"]

    unsafe_decision = json.loads(json.dumps(state))
    unsafe_decision["pending_approvals"][0]["default_decision"] = "approved_after_operator_review"
    assert validate_operator_state(unsafe_decision) == ["unsafe_approval_decision:vops-m5-approval-001"]

    invalid_status = json.loads(json.dumps(state))
    invalid_status["pending_approvals"][0]["status"] = "executed"
    assert validate_operator_state(invalid_status) == ["invalid_approval_status:vops-m5-approval-001:executed"]

    invalid_budget = json.loads(json.dumps(state))
    invalid_budget["pending_approvals"][0]["budget_impact_cents"] = -1
    assert validate_operator_state(invalid_budget) == [
        "invalid_approval_budget_impact:vops-m5-approval-001",
        "pending_approval_budget_mismatch",
    ]

    missing_contract = json.loads(json.dumps(state))
    missing_contract["approval_contracts"].pop(missing_contract["pending_approvals"][0]["action_id"])
    assert validate_operator_state(missing_contract) == [
        "approval_contract_mismatch:vops-m5-approval-001",
        "missing_approval_contracts:provision-voip-provider",
    ]

    mismatched_digest = json.loads(json.dumps(state))
    mismatched_digest["pending_approvals"][0]["command"] = "stripe projects add different/provider"
    assert validate_operator_state(mismatched_digest) == [
        "approval_contract_command_digest_mismatch:vops-m5-approval-001",
    ]

    unsafe_contract = json.loads(json.dumps(state))
    action_id = unsafe_contract["pending_approvals"][0]["action_id"]
    unsafe_contract["approval_contracts"][action_id]["allowed_decisions"] = ["approve_once"]
    unsafe_contract["pending_approvals"][0]["approval_contract"] = unsafe_contract["approval_contracts"][action_id]
    assert validate_operator_state(unsafe_contract) == [
        "approval_contract_decisions_mismatch:vops-m5-approval-001",
    ]

    missing_next_step = json.loads(json.dumps(state))
    missing_next_step["pending_approvals"][0]["operator_next_step"] = ""
    assert validate_operator_state(missing_next_step) == [
        "missing_operator_next_step:vops-m5-approval-001",
    ]

    claimed_execution = json.loads(json.dumps(state))
    claimed_execution["pending_approvals"][0]["execution_status"] = "executed"
    assert validate_operator_state(claimed_execution) == [
        "approval_execution_claimed:vops-m5-approval-001",
    ]


def test_operator_state_validates_audit_parentage_and_service_claims():
    state = build_operator_state()

    duplicate_audit = json.loads(json.dumps(state))
    duplicate_audit["recent_audit_events"][1]["audit_id"] = duplicate_audit["recent_audit_events"][0]["audit_id"]
    assert validate_operator_state(duplicate_audit) == ["duplicate_audit_ids:vops-m5-audit-001"]

    missing_parent = json.loads(json.dumps(state))
    missing_parent["recent_audit_events"][1]["parent_audit_id"] = "missing-parent"
    assert validate_operator_state(missing_parent) == ["audit_parent_missing:vops-m5-audit-002:missing-parent"]

    invalid_audit = json.loads(json.dumps(state))
    invalid_audit["recent_audit_events"][0]["status"] = "sent"
    assert validate_operator_state(invalid_audit) == ["invalid_audit_status:vops-m5-audit-001:sent"]

    missing_audit_next_step = json.loads(json.dumps(state))
    missing_audit_next_step["recent_audit_events"][0]["operator_next_step"] = ""
    assert validate_operator_state(missing_audit_next_step) == [
        "missing_operator_next_step:vops-m5-audit-001",
    ]

    external_planned_without_approval = json.loads(json.dumps(state))
    external_planned_without_approval["planned_services"][0]["approval_required"] = False
    assert validate_operator_state(external_planned_without_approval) == [
        "external_service_missing_approval:stripe_projects_voiceops_budget"
    ]

    external_planned_without_approval_ref = json.loads(json.dumps(state))
    external_planned_without_approval_ref["planned_services"][0]["approval_ref"] = None
    assert validate_operator_state(external_planned_without_approval_ref) == [
        "external_service_missing_approval_ref:stripe_projects_voiceops_budget"
    ]

    external_planned_with_unknown_approval_ref = json.loads(json.dumps(state))
    external_planned_with_unknown_approval_ref["planned_services"][0]["approval_ref"] = "missing-approval"
    assert validate_operator_state(external_planned_with_unknown_approval_ref) == [
        "external_service_unknown_approval_ref:stripe_projects_voiceops_budget:missing-approval"
    ]

    external_claimed_provisioned = json.loads(json.dumps(state))
    external_claimed_provisioned["planned_services"][0]["status"] = "provisioned"
    assert validate_operator_state(external_claimed_provisioned) == [
        "external_service_claimed_provisioned:stripe_projects_voiceops_budget"
    ]

    external_claimed_execution = json.loads(json.dumps(state))
    external_claimed_execution["planned_services"][0]["execution_status"] = "executed"
    assert validate_operator_state(external_claimed_execution) == [
        "external_service_execution_claimed:stripe_projects_voiceops_budget",
        "invalid_service_execution_status:stripe_projects_voiceops_budget:executed",
    ]

    missing_service_next_step = json.loads(json.dumps(state))
    missing_service_next_step["planned_services"][0]["operator_next_step"] = ""
    assert validate_operator_state(missing_service_next_step) == [
        "missing_operator_next_step:stripe_projects_voiceops_budget",
    ]

    external_provisioned = json.loads(json.dumps(state))
    external_provisioned["provisioned_services"][0]["external"] = True
    assert validate_operator_state(external_provisioned) == [
        "external_service_claimed_provisioned:repo_local_operator_artifacts"
    ]


def test_operator_state_validates_task_contracts():
    state = build_operator_state()

    duplicate_task = json.loads(json.dumps(state))
    duplicate_task["upcoming_tasks"][1]["task_id"] = duplicate_task["upcoming_tasks"][0]["task_id"]
    assert validate_operator_state(duplicate_task) == ["duplicate_task_ids:household-grocery-restock"]

    invalid_status = json.loads(json.dumps(state))
    invalid_status["upcoming_tasks"][0]["status"] = "done"
    assert validate_operator_state(invalid_status) == ["invalid_task_status:household-grocery-restock:done"]

    budget_without_approval = json.loads(json.dumps(state))
    budget_without_approval["upcoming_tasks"][0]["budget_impact_cents"] = 100
    assert validate_operator_state(budget_without_approval) == [
        "task_budget_without_approval:household-grocery-restock"
    ]

    invalid_budget = json.loads(json.dumps(state))
    invalid_budget["upcoming_tasks"][0]["budget_impact_cents"] = "100"
    assert validate_operator_state(invalid_budget) == ["invalid_task_budget_impact:household-grocery-restock"]


def test_write_operator_state_artifacts(tmp_path):
    state = build_operator_state()
    paths = write_operator_state(tmp_path, state)

    assert set(paths) == {"events_jsonl", "json", "markdown"}
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    events = [
        json.loads(line)
        for line in Path(paths["events_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert payload["scope"]["default_output_dir"] == str(DEFAULT_OUTPUT_DIR)
    assert payload["mode"]["outbound_calls"] is False
    assert payload["mode"]["provisioning"] is False
    assert payload["current_mode"] == "approval-required"
    assert events == payload["recent_audit_events"]
    assert "VoiceOps Milestone 5 Operator State" in markdown
    assert "Active Voice Surface" in markdown
    assert "Budget Status" in markdown
    assert "Pending Approvals" in markdown
    assert "provision-voip-provider" in markdown
    assert "nemoclaw-action-packet.json" in markdown
    assert "Next step" in markdown
    assert "not_executed" in markdown
    assert "Recent Audit Events" in markdown
    assert "Planned Services" in markdown
    assert "Provisioned Services" in markdown
    assert "Upcoming Tasks" in markdown


def test_operator_state_cli_smoke(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "voiceops_operator_state.py"
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
