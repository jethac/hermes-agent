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
    assert state["recent_audit_events"]
    assert state["planned_services"]
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
    over_bounds["pending_approvals"] = over_bounds["pending_approvals"] * 5
    assert validate_operator_state(over_bounds) == ["bounds_exceeded:pending_approvals"]

    missing_business = json.loads(json.dumps(state))
    missing_business["upcoming_tasks"] = [
        task for task in missing_business["upcoming_tasks"] if task["domain"] == "household"
    ]
    assert validate_operator_state(missing_business) == ["missing_task_domain:business"]


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
