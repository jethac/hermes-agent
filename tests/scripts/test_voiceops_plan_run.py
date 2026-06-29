from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.voiceops_plan_run import build_plan_run, parse_args, write_plan_run


def test_plan_run_generates_all_headless_milestone_artifacts(tmp_path):
    artifact_root = tmp_path / "artifacts"
    output_dir = artifact_root / "voiceops-plan" / "current"
    summary = build_plan_run(artifact_root=artifact_root, output_dir=output_dir, env={})
    paths = write_plan_run(output_dir, summary)

    assert summary["schema_version"] == "voiceops.plan_run.v1"
    assert summary["artifact_only"] is True
    assert summary["ok"] is True
    assert summary["hard_failures"] == []
    assert "milestone_1_real_voice_operator" in summary["readiness_gaps"]
    assert "milestone_2_real_spend_and_provisioning_preflight" in summary["readiness_gaps"]
    assert "milestone_4_local_spark_stack_matrix" in summary["readiness_gaps"]
    assert summary["safety"] == {
        "env_presence_inspection": True,
        "env_secret_values_emitted": False,
        "live_spend": False,
        "network_io": False,
        "outbound_calls": False,
        "outbound_sends": False,
        "provider_provisioning": False,
    }
    assert {result["milestone"] for result in summary["results"]} == {
        "milestone_0_hackathon_proof",
        "milestone_1_real_voice_operator",
        "milestone_2_real_spend_and_provisioning_preflight",
        "milestone_3_multi_channel_policy",
        "milestone_4_local_spark_stack_matrix",
        "milestone_5_operator_dashboard_state",
    }

    demo_result = next(result for result in summary["results"] if result["milestone"] == "milestone_0_hackathon_proof")
    assert Path(demo_result["artifacts"]["dashboard"]).exists()
    assert Path(demo_result["artifacts"]["operator_state"]).exists()
    assert Path(demo_result["artifacts"]["operator_state_events"]).exists()

    voice_result = next(result for result in summary["results"] if result["milestone"] == "milestone_1_real_voice_operator")
    assert voice_result["status"] == "needs_live_probe"
    assert voice_result["details"]["live_probe_status"] == "needs_live_probe"
    assert Path(voice_result["artifacts"]["json"]).exists()
    assert Path(voice_result["artifacts"]["markdown"]).exists()
    assert Path(voice_result["artifacts"]["smoke_json"]).exists()
    assert Path(voice_result["artifacts"]["events_jsonl"]).exists()

    provisioning_result = next(
        result for result in summary["results"] if result["milestone"] == "milestone_2_real_spend_and_provisioning_preflight"
    )
    assert provisioning_result["status"] == "needs_setup"
    assert provisioning_result["details"]["required_failures"]
    assert provisioning_result["details"]["run_command_probes"] is False
    assert Path(provisioning_result["artifacts"]["execution_plan_json"]).exists()
    assert Path(provisioning_result["artifacts"]["execution_plan_markdown"]).exists()

    matrix_result = next(result for result in summary["results"] if result["milestone"] == "milestone_4_local_spark_stack_matrix")
    assert matrix_result["status"] == "needs_evidence"
    assert matrix_result["details"]["ready_for_one_spark_demo"] is False

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert payload["ok"] is True
    assert "VoiceOps Plan Run Summary" in markdown
    assert "milestone_0_hackathon_proof" in markdown


def test_plan_run_marks_provisioning_ready_with_presence_only_inputs(tmp_path):
    artifact_root = tmp_path / "artifacts"
    output_dir = artifact_root / "voiceops-plan" / "current"
    summary = build_plan_run(
        artifact_root=artifact_root,
        output_dir=output_dir,
        env={
            "PATH": "",
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "TWILIO_ACCOUNT_SID": "AC123456789abcdef",
            "STRIPE_SECRET_KEY": "sk_live_123456789abcdef",
        },
    )

    provisioning_result = next(
        result for result in summary["results"] if result["milestone"] == "milestone_2_real_spend_and_provisioning_preflight"
    )
    serialized = json.dumps(summary)
    assert "sk_live_123456789abcdef" not in serialized
    assert "+15551234567" not in serialized
    assert provisioning_result["details"]["run_command_probes"] is False


def test_plan_run_cli_smoke(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "voiceops_plan_run.py"
    artifact_root = tmp_path / "artifacts"
    output_dir = artifact_root / "voiceops-plan" / "current"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifact-root",
            str(artifact_root),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert Path(payload["artifacts"]["json"]).exists()
    assert Path(payload["artifacts"]["markdown"]).exists()


def test_parse_args_defaults_to_plan_artifact_paths():
    args = parse_args([])

    assert args.artifact_root == Path("artifacts")
    assert args.output_dir == Path("artifacts/voiceops-plan/current")
    assert args.run_command_probes is False
