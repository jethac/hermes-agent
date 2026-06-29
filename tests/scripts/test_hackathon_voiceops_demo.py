from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.hackathon_voiceops_demo import build_demo, build_readiness_report, parse_args, write_demo


def test_voiceops_demo_writes_headless_artifacts(tmp_path):
    args = parse_args(["--output-dir", str(tmp_path), "--budget-cents", "7000"])
    demo = build_demo(args)
    paths = write_demo(tmp_path, demo)

    assert set(paths) == {
        "json",
        "markdown",
        "audit_ledger",
        "demo_script",
        "dashboard",
        "milestone2_execution_plan",
        "nemoclaw_packet",
        "operator_state",
        "operator_state_events",
        "phone_context",
        "recording_runbook",
        "readiness_json",
        "readiness_markdown",
        "stripe_actions",
        "submission_writeup",
    }
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    nemoclaw = json.loads(Path(paths["nemoclaw_packet"]).read_text(encoding="utf-8"))
    phone_context = json.loads(Path(paths["phone_context"]).read_text(encoding="utf-8"))
    milestone2_plan = json.loads(Path(paths["milestone2_execution_plan"]).read_text(encoding="utf-8"))
    operator_state = json.loads(Path(paths["operator_state"]).read_text(encoding="utf-8"))
    operator_events = [
        json.loads(line)
        for line in Path(paths["operator_state_events"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    action_ids = {action["action_id"] for action in payload["ops_actions"]}
    assert payload["artifact_manifest"]["readiness_json"] == "readiness-report.json"
    assert payload["artifact_manifest"]["operator_state"] == "operator-state.json"
    assert payload["artifact_manifest"]["milestone2_execution_plan"] == "milestone2-execution-plan.json"
    assert payload["recording_readiness"]["artifact_ref"] == "readiness-report.json"
    assert payload["recording_readiness"]["ready_for_recording"] is False
    assert payload["recording_readiness"]["required_failures"]
    assert payload["readiness_closure_ref"] == "artifacts/voiceops-plan/current/readiness-closure-index.json"
    assert payload["readiness_closure"]["closure_status"] == "needs_external_evidence"
    assert {gate["gate_id"] for gate in payload["readiness_closure"]["gates"]} == {
        "live_discord_voice_operator",
        "spend_and_provisioning_preflight",
        "local_spark_stack_matrix",
    }
    assert payload["operator_state_ref"] == "operator-state.json"
    assert payload["operator_state_events_ref"] == "operator-state-events.jsonl"
    assert payload["milestone2_execution_plan_ref"] == "milestone2-execution-plan.json"
    assert payload["safety_boundary_refs"] == {
        "nemoclaw_action_packet": "nemoclaw-action-packet.json",
        "phone_context": "phone-context.json",
        "stripe_actions_dry_run": "stripe-actions-dry-run.sh",
    }
    assert payload["spark_stack"]["local_first"] is True
    assert payload["sponsor_stack"]["hermes_active_model"]["path"] == "spark_local_nemotron_3_super"
    assert payload["sponsor_stack"]["hermes_active_model"]["selected_by"] == "Hermes /model"
    assert payload["sponsor_stack"]["hermes_active_model"]["spark_local"] is True
    assert payload["sponsor_stack"]["nemotron_3_super"]["selection"].startswith("Nemotron 3 Super")
    assert payload["sponsor_stack"]["nemotron_3_ultra_hosted_fallback"]["selection"].startswith(
        "Nemotron 3 Ultra hosted fallback"
    )
    assert payload["spark_stack"]["oracle"]["preferred_local_target"] == "Nemotron 3 Super on DGX Spark"
    assert payload["spark_stack"]["oracle"]["hosted_fallback"].startswith("Nemotron 3 Ultra")
    assert payload["spark_stack"]["oracle"]["active_model_path"]["path"] == "spark_local_nemotron_3_super"
    assert payload["sponsor_stack"]["stripe_skills"]["skills"] == ["stripe-projects", "stripe-link-cli", "mpp-agent"]
    assert payload["voice_surfaces"][0]["channel"] == "discord"
    assert {surface["channel"] for surface in payload["voice_surfaces"]} == {"discord", "whatsapp", "phone"}
    assert "provision-voip-provider" in action_ids
    assert "call-user-phone" in action_ids
    assert payload["totals"]["held_budget_cents"] > 0
    assert nemoclaw["runtime"] == "NemoClaw"
    assert nemoclaw["mode"] == "dry_run_until_user_approval"
    assert nemoclaw["model_selected_by"] == "Hermes /model"
    assert nemoclaw["hermes_active_model"].startswith("Nemotron 3 Super")
    assert "oracle_model" not in nemoclaw
    assert "unapproved_purchase" in nemoclaw["blocked_capabilities"]
    assert "stripe projects add twilio/voice" in nemoclaw["dry_run_commands"]
    assert phone_context["target_channel"] == "phone"
    assert phone_context["status"] == "queued_requires_approval"
    assert phone_context["pending_approvals"]
    assert milestone2_plan["schema_version"] == "voiceops.milestone2.execution_plan.v1"
    assert milestone2_plan["demo_refs"]["phone_context"] == "phone-context.json"
    assert milestone2_plan["source_readiness_artifact"] == "provisioning-readiness.json"
    assert {step["step_id"] for step in milestone2_plan["execution_steps"]} >= {
        "provision-voip-provider",
        "buy-service-credit",
        "call-user-phone",
    }
    assert {gate["gate_id"] for gate in milestone2_plan["approval_gates"]} >= {
        "stripe-projects-provisioning",
        "stripe-link-spend",
        "phone-call-handoff",
    }
    assert "deprovision_voip_provider" in milestone2_plan["rollback_plan"]
    assert operator_state["schema_version"] == "voiceops.operator_state.v1"
    assert payload["operator_state"] == operator_state
    assert operator_state["readiness_closure"]["closure_status"] == "needs_external_evidence"
    assert operator_state["current_mode"] == "approval-required"
    assert operator_state["active_voice_surface"]["surface_id"] == "discord_voice"
    assert operator_state["provisioned_services"]
    assert operator_events == operator_state["recent_audit_events"]
    assert "DGX Spark" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "nemoclaw-action-packet.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "phone-context.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "milestone2-execution-plan.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "readiness-report.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "operator-dashboard.html" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "operator-state.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "operator-state-events.jsonl" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "recording-runbook.md" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "submission-writeup.md" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "spoken in Discord" in Path(paths["demo_script"]).read_text(encoding="utf-8")
    assert "outbound phone call" in Path(paths["demo_script"]).read_text(encoding="utf-8")
    runbook = Path(paths["recording_runbook"]).read_text(encoding="utf-8")
    assert "VoiceOps Recording Runbook" in runbook
    assert "Shot List" in runbook
    assert "@NousResearch" in runbook
    assert "Do not show terminal panes or files that contain secrets" in runbook
    assert "Plan Closure Gates" in runbook
    assert "live_discord_voice_operator" in runbook
    assert "production_sidecar" in runbook
    assert "live_turn" in runbook
    assert "spend_and_provisioning_preflight" in runbook
    assert "local_spark_stack_matrix" in runbook
    writeup = Path(paths["submission_writeup"]).read_text(encoding="utf-8")
    assert "Hermes VoiceOps Submission Writeup" in writeup
    assert "NemoClaw" in writeup
    assert "Stripe Skills" in writeup
    assert "@NousResearch" in writeup
    assert "Remaining Closure Gates" in writeup
    assert "live_discord_voice_operator" in writeup
    assert "spend_and_provisioning_preflight" in writeup
    assert "local_spark_stack_matrix" in writeup
    dashboard = Path(paths["dashboard"]).read_text(encoding="utf-8")
    assert "Nemotron 3 Super" in dashboard
    assert "Nemotron 3 Ultra hosted fallback" in dashboard
    assert "Ultra does not count as Spark-local readiness proof" in dashboard
    assert "NemoClaw Blocks" in dashboard
    assert "Plan Closure Gates" in dashboard
    assert "live_discord_voice_operator" in dashboard
    assert "spend_and_provisioning_preflight" in dashboard
    assert "local_spark_stack_matrix" in dashboard
    assert "Current Mode" in dashboard
    assert "dry_run_until_user_approval" in dashboard
    assert "approval-required" in dashboard
    assert "Active Voice Surface" in dashboard
    assert "discord_voice" in dashboard
    assert "Budget Status" in dashboard
    assert "$70.00" in dashboard
    assert "$10.00" in dashboard
    assert "held-budget" in dashboard
    assert "Pending Approvals" in dashboard
    assert "provision-voip-provider" in dashboard
    assert "call-user-phone" in dashboard
    assert "Action Ledger" in dashboard
    assert "Recent Audit Events" in dashboard
    assert "evt-006" in dashboard
    assert "Planned Services" in dashboard
    assert "Provisioned Services" in dashboard
    assert "repo_local_demo_artifacts" in dashboard
    assert "stripe-projects" in dashboard
    assert "Upcoming Tasks" in dashboard
    assert "operator-state.json" in dashboard
    assert "Phone Handoff" in dashboard
    assert "milestone2-execution-plan.json" in dashboard
    assert Path(paths["readiness_json"]).exists()
    assert "VoiceOps Recording Readiness" in Path(paths["readiness_markdown"]).read_text(encoding="utf-8")


def test_voiceops_demo_classifies_ultra_as_hosted_fallback_and_rejects_unaligned_model():
    ultra_args = parse_args(["--active-model", "Nemotron 3 Ultra via hosted Nous provider"])
    ultra_demo = build_demo(ultra_args)
    ultra_ready = build_readiness_report(
        ultra_demo,
        env={
            "DISCORD_BOT_TOKEN": "set",
            "DISCORD_VOICE_CHANNEL_ID": "123",
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED": "true",
        },
        which=lambda command: f"/usr/local/bin/{command}" if command in {"stripe", "link-cli"} else None,
    )

    assert ultra_demo["sponsor_stack"]["hermes_active_model"]["path"] == "hosted_nemotron_3_ultra_fallback"
    assert ultra_demo["sponsor_stack"]["hermes_active_model"]["spark_local"] is False
    assert ultra_demo["sponsor_stack"]["nemotron_3_super"]["selection"] == "not selected"
    assert ultra_demo["sponsor_stack"]["nemotron_3_ultra_hosted_fallback"]["selection"].startswith("Nemotron 3 Ultra")
    assert "nemotron_3_super_spark_or_labeled_ultra_hosted_fallback" not in ultra_ready["required_failures"]

    kimi_args = parse_args(["--active-model", "Kimi K2.6"])
    kimi_demo = build_demo(kimi_args)
    kimi_ready = build_readiness_report(
        kimi_demo,
        env={
            "DISCORD_BOT_TOKEN": "set",
            "DISCORD_VOICE_CHANNEL_ID": "123",
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED": "true",
        },
        which=lambda command: f"/usr/local/bin/{command}" if command in {"stripe", "link-cli"} else None,
    )

    assert kimi_demo["sponsor_stack"]["hermes_active_model"]["path"] == "non_nvidia_fallback"
    assert kimi_demo["sponsor_stack"]["hermes_active_model"]["spark_local"] is False
    assert "nemotron_3_super_spark_or_labeled_ultra_hosted_fallback" in kimi_ready["required_failures"]


def test_voiceops_demo_dry_run_does_not_execute_live_stripe(tmp_path):
    args = parse_args(["--output-dir", str(tmp_path)])
    demo = build_demo(args)
    paths = write_demo(tmp_path, demo)
    text = Path(paths["stripe_actions"]).read_text(encoding="utf-8")

    assert "printf '%s\\n'" in text
    assert "stripe projects add twilio/voice" in text
    assert "link-cli spend-request create" in text
    assert "queue outbound call" in text
    assert "sk_" not in text


def test_voiceops_demo_cli_smoke(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "hackathon_voiceops_demo.py"
    result = subprocess.run(
        ["python", str(script), "--output-dir", str(tmp_path), "--budget-cents", "20000"],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert Path(payload["artifacts"]["json"]).exists()


def test_voiceops_readiness_report_distinguishes_required_failures():
    args = parse_args([])
    demo = build_demo(args)

    def fake_which(command: str) -> str | None:
        commands = {
            "hermes": "/usr/local/bin/hermes",
            "stripe": "/usr/local/bin/stripe",
            "link-cli": "/usr/local/bin/link-cli",
            "nemoclaw": "/usr/local/bin/nemoclaw",
        }
        return commands.get(command)

    ready = build_readiness_report(
        demo,
        env={
            "DISCORD_BOT_TOKEN": "set",
            "DISCORD_VOICE_CHANNEL_ID": "123",
            "WHATSAPP_ENABLED": "true",
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED": "true",
        },
        which=fake_which,
    )
    assert ready["ready_for_recording"] is True
    assert ready["required_failures"] == []

    stripe_without_projects_marker = build_readiness_report(
        demo,
        env={
            "DISCORD_BOT_TOKEN": "set",
            "DISCORD_VOICE_CHANNEL_ID": "123",
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
        },
        which=fake_which,
    )
    assert stripe_without_projects_marker["ready_for_recording"] is False
    assert "stripe_projects_cli" in stripe_without_projects_marker["required_failures"]

    def fake_npx_only(command: str) -> str | None:
        commands = {
            "hermes": "/usr/local/bin/hermes",
            "stripe": "/usr/local/bin/stripe",
            "npx": "/usr/local/bin/npx",
        }
        return commands.get(command)

    npx_not_ready = build_readiness_report(
        demo,
        env={
            "DISCORD_BOT_TOKEN": "set",
            "DISCORD_VOICE_CHANNEL_ID": "123",
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED": "true",
        },
        which=fake_npx_only,
    )
    assert npx_not_ready["ready_for_recording"] is False
    assert "stripe_link_cli" in npx_not_ready["required_failures"]

    not_ready = build_readiness_report(demo, env={}, which=lambda _command: None)
    assert not_ready["ready_for_recording"] is False
    assert {"discord_voice", "stripe_projects_cli", "stripe_link_cli"}.issubset(set(not_ready["required_failures"]))


def test_voiceops_readiness_report_loads_env_files_without_exposing_values(tmp_path):
    args = parse_args([])
    demo = build_demo(args)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DISCORD_BOT_TOKEN=secret-token",
                "DISCORD_VOICE_CHANNEL_NAME=General",
                "WHATSAPP_ENABLED=true",
                "VOICEOPS_DEMO_PHONE_NUMBER='+15551234567'",
                "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED=true",
            ]
        ),
        encoding="utf-8",
    )

    def fake_which(command: str) -> str | None:
        commands = {
            "stripe": "/usr/local/bin/stripe",
            "link-cli": "/usr/local/bin/link-cli",
        }
        return commands.get(command)

    report = build_readiness_report(demo, env={}, env_files=[env_file], which=fake_which)

    assert report["ready_for_recording"] is True
    assert report["required_failures"] == []
    assert report["env_sources"][1]["path"] == str(env_file)
    assert report["env_sources"][1]["loaded"] is True
    assert "secret-token" not in json.dumps(report)
