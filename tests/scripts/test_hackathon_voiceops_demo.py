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
        "nemoclaw_packet",
        "phone_context",
        "readiness_json",
        "readiness_markdown",
        "stripe_actions",
    }
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    nemoclaw = json.loads(Path(paths["nemoclaw_packet"]).read_text(encoding="utf-8"))
    phone_context = json.loads(Path(paths["phone_context"]).read_text(encoding="utf-8"))
    action_ids = {action["action_id"] for action in payload["ops_actions"]}
    assert payload["spark_stack"]["local_first"] is True
    assert payload["sponsor_stack"]["nemotron_3_ultra"]["selection"].startswith("Nemotron 3 Ultra")
    assert payload["sponsor_stack"]["stripe_skills"]["skills"] == ["stripe-projects", "stripe-link-cli", "mpp-agent"]
    assert payload["voice_surfaces"][0]["channel"] == "discord"
    assert {surface["channel"] for surface in payload["voice_surfaces"]} == {"discord", "whatsapp", "phone"}
    assert "provision-voip-provider" in action_ids
    assert "call-user-phone" in action_ids
    assert payload["totals"]["held_budget_cents"] > 0
    assert nemoclaw["runtime"] == "NemoClaw"
    assert nemoclaw["mode"] == "dry_run_until_user_approval"
    assert "unapproved_purchase" in nemoclaw["blocked_capabilities"]
    assert "stripe projects add twilio/voice" in nemoclaw["dry_run_commands"]
    assert phone_context["target_channel"] == "phone"
    assert phone_context["status"] == "queued_requires_approval"
    assert phone_context["pending_approvals"]
    assert "DGX Spark" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "nemoclaw-action-packet.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "phone-context.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "readiness-report.json" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "spoken in Discord" in Path(paths["demo_script"]).read_text(encoding="utf-8")
    assert "outbound phone call" in Path(paths["demo_script"]).read_text(encoding="utf-8")
    assert Path(paths["readiness_json"]).exists()
    assert "VoiceOps Recording Readiness" in Path(paths["readiness_markdown"]).read_text(encoding="utf-8")


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
        },
        which=fake_which,
    )
    assert ready["ready_for_recording"] is True
    assert ready["required_failures"] == []

    not_ready = build_readiness_report(demo, env={}, which=lambda _command: None)
    assert not_ready["ready_for_recording"] is False
    assert {"discord_voice", "stripe_projects_cli", "stripe_link_cli"}.issubset(set(not_ready["required_failures"]))
