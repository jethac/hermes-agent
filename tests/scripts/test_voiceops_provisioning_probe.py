from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from scripts.voiceops_provisioning_probe import (
    CommandResult,
    build_milestone2_execution_plan,
    build_probe_report,
    parse_args,
    write_probe_artifacts,
)


def test_probe_passes_with_safe_local_tools_and_redacts_outputs():
    calls: list[list[str]] = []

    def fake_which(command: str) -> str | None:
        paths = {
            "stripe": "/usr/local/bin/stripe",
            "link-cli": "/usr/local/bin/link-cli",
            "mppx": "/usr/local/bin/mppx",
            "twilio": "/usr/local/bin/twilio",
        }
        return paths.get(command)

    def fake_runner(argv: Sequence[str], _timeout_seconds: int) -> CommandResult:
        calls.append(list(argv))
        return CommandResult(
            exit_code=0,
            stdout="ok STRIPE_SECRET_KEY=sk_live_123456789abcdef phone +15551234567",
            stderr="Bearer token_123456789abcdef",
        )

    report = build_probe_report(
        env={
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "TWILIO_ACCOUNT_SID": "AC123456789abcdef",
            "TWILIO_AUTH_TOKEN": "secret-token",
        },
        env_files=[],
        which=fake_which,
        runner=fake_runner,
        run_commands=True,
    )

    assert report["ready"] is True
    assert report["required_failures"] == []
    assert calls == [
        ["stripe", "--version"],
        ["stripe", "projects", "--help"],
        ["link-cli", "--version"],
        ["mppx", "--version"],
        ["twilio", "--version"],
    ]
    assert all(any(arg in {"--version", "--help"} for arg in call[1:]) for call in calls)
    joined_calls = " ".join(" ".join(call) for call in calls)
    for forbidden in ["projects add", "spend-request create", "provision", "call create", "credential"]:
        assert forbidden not in joined_calls

    serialized = json.dumps(report)
    assert "sk_live_123456789abcdef" not in serialized
    assert "+15551234567" not in serialized
    assert "secret-token" not in serialized
    assert "<redacted" in serialized


def test_probe_reports_required_failures_without_running_missing_tools():
    calls: list[list[str]] = []

    report = build_probe_report(
        env={},
        env_files=[],
        which=lambda _command: None,
        runner=lambda argv, _timeout_seconds: calls.append(list(argv)) or CommandResult(exit_code=0),
    )

    assert report["ready"] is False
    assert set(report["required_failures"]) == {
        "stripe_cli",
        "stripe_projects_cli",
        "stripe_link_cli",
        "mpp_agent",
        "phone_target",
        "phone_provider",
    }
    assert calls == []


def test_probe_treats_no_command_probes_as_path_presence_only():
    def fake_which(command: str) -> str | None:
        return f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None

    report = build_probe_report(
        env={"VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567", "VAPI_API_KEY": "secret"},
        env_files=[],
        which=fake_which,
        runner=lambda _argv, _timeout_seconds: (_ for _ in ()).throw(AssertionError("runner should not be called")),
    )

    assert report["ready"] is True
    assert all(probe["executed"] is False for probe in report["command_probes"])
    assert {probe["status"] for probe in report["command_probes"] if probe["found"]} == {"found"}


def test_write_probe_artifacts(tmp_path):
    report = build_probe_report(
        env={"VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567", "TWILIO_ACCOUNT_SID": "AC123"},
        env_files=[],
        which=lambda command: f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=0, stdout="version 1.0"),
    )
    paths = write_probe_artifacts(tmp_path, report)

    assert set(paths) == {
        "command_manifest",
        "execution_plan_json",
        "execution_plan_markdown",
        "json",
        "markdown",
    }
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    execution_plan = json.loads(Path(paths["execution_plan_json"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(paths["command_manifest"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    execution_markdown = Path(paths["execution_plan_markdown"]).read_text(encoding="utf-8")
    assert payload["probe"]["non_mutating"] is True
    assert "VoiceOps Provisioning Readiness Probe" in markdown
    assert execution_plan["schema_version"] == "voiceops.milestone2.execution_plan.v1"
    assert "phone-context.json" in json.dumps(execution_plan)
    assert "VoiceOps Milestone 2 Execution Plan" in execution_markdown
    assert "projects add" in manifest["blocked_patterns"]
    assert "+15551234567" not in json.dumps(payload)


def test_milestone2_execution_plan_defines_safety_gates_receipts_and_rollback():
    report = build_probe_report(
        env={
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "TWILIO_ACCOUNT_SID": "AC123456789abcdef",
            "TWILIO_AUTH_TOKEN": "secret-token",
            "STRIPE_SECRET_KEY": "sk_live_123456789abcdef",
        },
        env_files=[],
        which=lambda command: f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=0),
    )
    plan = build_milestone2_execution_plan(report)

    assert plan["schema_version"] == "voiceops.milestone2.execution_plan.v1"
    assert plan["artifact_id"] == "voiceops-m2-execution-plan"
    assert plan["mode"] == {"artifact_only": True, "bounded": True, "headless": True}
    assert plan["safety"] == {
        "account_mutation": False,
        "credential_retrieval": False,
        "env_secret_reads": False,
        "live_spend": False,
        "network_io": False,
        "outbound_phone_calls": False,
        "provider_provisioning": False,
    }
    assert {
        "live_spend",
        "provider_provisioning",
        "credential_retrieval",
        "outbound_calls",
        "outbound_messages",
        "network_tunnels",
        "raw_card_data",
        "unapproved_recurring_charges",
    } <= set(plan["blocked_capabilities"])
    assert plan["preflight"]["run_command_probes_default"] is False
    assert plan["preflight"]["run_command_probes_does_not_grant_approval"] is True
    assert plan["demo_refs"] == {
        "audit_ledger": "audit-ledger.jsonl",
        "nemoclaw_packet": "nemoclaw-action-packet.json",
        "phone_context": "phone-context.json",
        "stripe_actions_dry_run": "stripe-actions-dry-run.sh",
        "voiceops_demo": "voiceops-demo.json",
    }
    assert "receipt_id" in plan["receipt_schema"]["required_fields"]
    assert "credential_ref_id" in plan["credential_location_schema"]["required_fields"]
    assert "raw_secret" in plan["credential_location_schema"]["forbidden_fields"]
    assert {"deprovision_voip_provider", "refund_or_cancel_service_credit", "cancel_or_end_phone_handoff"} <= set(
        plan["rollback_plan"]
    )

    risky_steps = [
        step
        for step in plan["execution_steps"]
        if step["provider"] in {"stripe-projects", "stripe-link-cli", "voiceops-phone-bridge", "hermes-gateway"}
    ]
    assert risky_steps
    assert all(step["requires_approval"] is True for step in risky_steps)
    assert all(step["status"] == "blocked_until_explicit_approval" for step in risky_steps)
    assert {gate["gate_id"] for gate in plan["approval_gates"]} == {
        "outbound-status-messages",
        "phone-call-handoff",
        "stripe-link-spend",
        "stripe-projects-provisioning",
    }

    serialized = json.dumps(plan)
    assert "sk_live_123456789abcdef" not in serialized
    assert "+15551234567" not in serialized
    assert "secret-token" not in serialized


def test_probe_loads_env_file_key_presence_without_values(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "VOICEOPS_DEMO_PHONE_NUMBER=+15551234567",
                "TWILIO_AUTH_TOKEN=secret-token",
                "TWILIO_ACCOUNT_SID=AC123456789abcdef",
            ]
        ),
        encoding="utf-8",
    )

    report = build_probe_report(
        env={},
        env_files=[env_file],
        which=lambda command: f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=0),
    )

    serialized = json.dumps(report)
    assert report["ready"] is True
    assert report["env_sources"][1]["loaded"] is True
    assert "VOICEOPS_DEMO_PHONE_NUMBER" in serialized
    assert "+15551234567" not in serialized
    assert "secret-token" not in serialized


def test_probe_refuses_forbidden_hermes_agent_env_path():
    forbidden = Path("/Users/jethac/.hermes/hermes-agent/.env")

    with pytest.raises(ValueError, match="forbidden Hermes worktree"):
        build_probe_report(env={}, env_files=[forbidden], which=lambda _command: None)


def test_probe_cli_smoke_no_command_probes(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "voiceops_provisioning_probe.py"
    result = subprocess.run(
        ["python", str(script), "--output-dir", str(tmp_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert Path(payload["artifacts"]["json"]).exists()
    assert Path(payload["artifacts"]["markdown"]).exists()


def test_parse_args_defaults_to_requested_artifact_dir():
    args = parse_args([])

    assert args.output_dir == Path("artifacts/voiceops-provisioning/current")
    assert args.run_command_probes is False
