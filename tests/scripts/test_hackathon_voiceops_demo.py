from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.hackathon_voiceops_demo import build_demo, parse_args, write_demo


def test_voiceops_demo_writes_headless_artifacts(tmp_path):
    args = parse_args(["--output-dir", str(tmp_path), "--budget-cents", "7000"])
    demo = build_demo(args)
    paths = write_demo(tmp_path, demo)

    assert set(paths) == {"json", "markdown", "audit_ledger", "demo_script", "stripe_actions"}
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    action_ids = {action["action_id"] for action in payload["ops_actions"]}
    assert payload["spark_stack"]["local_first"] is True
    assert payload["sponsor_stack"]["nemotron_3_ultra"]["selection"].startswith("Nemotron 3 Ultra")
    assert payload["sponsor_stack"]["stripe_skills"]["skills"] == ["stripe-projects", "stripe-link-cli", "mpp-agent"]
    assert payload["voice_surfaces"][0]["channel"] == "discord"
    assert {surface["channel"] for surface in payload["voice_surfaces"]} == {"discord", "whatsapp", "phone"}
    assert "provision-voip-provider" in action_ids
    assert "call-user-phone" in action_ids
    assert payload["totals"]["held_budget_cents"] > 0
    assert "DGX Spark" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "spoken in Discord" in Path(paths["demo_script"]).read_text(encoding="utf-8")
    assert "outbound phone call" in Path(paths["demo_script"]).read_text(encoding="utf-8")


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
