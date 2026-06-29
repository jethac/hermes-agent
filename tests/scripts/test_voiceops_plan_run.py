from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.voiceops_plan_run import build_plan_run, parse_args, write_plan_run


GOAL_DOC = Path(__file__).resolve().parents[2] / "docs" / "plans" / "2026-06-29-spark-household-business-voiceops.md"


def test_plan_run_generates_all_headless_milestone_artifacts(tmp_path):
    artifact_root = tmp_path / "artifacts"
    output_dir = artifact_root / "voiceops-plan" / "current"
    summary = build_plan_run(artifact_root=artifact_root, output_dir=output_dir, env={})
    paths = write_plan_run(output_dir, summary)

    assert summary["schema_version"] == "voiceops.plan_run.v1"
    assert summary["artifact_only"] is True
    assert summary["ok"] is True
    assert summary["closure_index"]["schema_version"] == "voiceops.closure_index.v1"
    assert summary["closure_index"]["closure_status"] == "needs_external_evidence"
    assert summary["closure_index"]["source_plan_run_artifact"].endswith("voiceops-plan-run.json")
    assert summary["closure_index"]["remaining_gates"] == summary["closure_index"]["gates"]
    gates = {gate["gate_id"]: gate for gate in summary["closure_index"]["gates"]}
    assert set(gates) == {
        "live_discord_voice_operator",
        "local_spark_stack_matrix",
        "spend_and_provisioning_preflight",
    }
    assert "transcript_observed" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "operator_must_not" in gates["live_discord_voice_operator"]
    assert "manifest.json" in gates["live_discord_voice_operator"]["rerun_command"]
    assert "missing_preflight_fields" in gates["spend_and_provisioning_preflight"]
    assert "required_candidate_fields" in gates["local_spark_stack_matrix"]
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
    assert "live_probe_missing_gates" in voice_result["details"]
    assert Path(voice_result["artifacts"]["json"]).exists()
    assert Path(voice_result["artifacts"]["markdown"]).exists()
    assert Path(voice_result["artifacts"]["smoke_json"]).exists()
    assert Path(voice_result["artifacts"]["events_jsonl"]).exists()
    assert Path(voice_result["artifacts"]["live_evidence_example"]).exists()
    assert Path(voice_result["artifacts"]["live_evidence_template"]).exists()
    assert Path(voice_result["artifacts"]["live_probe_closure_json"]).exists()

    provisioning_result = next(
        result for result in summary["results"] if result["milestone"] == "milestone_2_real_spend_and_provisioning_preflight"
    )
    assert provisioning_result["status"] == "needs_setup"
    assert provisioning_result["details"]["required_failures"]
    assert provisioning_result["details"]["run_command_probes"] is False
    assert Path(provisioning_result["artifacts"]["execution_plan_json"]).exists()
    assert Path(provisioning_result["artifacts"]["execution_plan_markdown"]).exists()
    assert Path(provisioning_result["artifacts"]["preflight_evidence_example"]).exists()
    assert Path(provisioning_result["artifacts"]["preflight_evidence_manifest_example"]).exists()

    matrix_result = next(result for result in summary["results"] if result["milestone"] == "milestone_4_local_spark_stack_matrix")
    assert matrix_result["status"] == "needs_evidence"
    assert matrix_result["details"]["ready_for_one_spark_demo"] is False
    assert matrix_result["details"]["stack_smoke_status"] == "needs_evidence"
    assert Path(matrix_result["artifacts"]["closure_json"]).exists()
    assert Path(matrix_result["artifacts"]["closure_markdown"]).exists()
    assert Path(matrix_result["artifacts"]["evidence_example"]).exists()

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    closure = json.loads(Path(paths["closure_json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    closure_markdown = Path(paths["closure_markdown"]).read_text(encoding="utf-8")
    assert payload["ok"] is True
    assert closure["artifact_id"] == "voiceops-plan-readiness-closure"
    assert closure["schema_version"] == "voiceops.closure_index.v1"
    provisioning_gate = next(gate for gate in closure["gates"] if gate["gate_id"] == "spend_and_provisioning_preflight")
    assert provisioning_gate["evidence_manifest_example"].endswith(
        "provisioning-preflight-evidence.manifest.example.json"
    )
    assert "provisioning-preflight-evidence.manifest.json" in provisioning_gate["rerun_commands"]["plan_index_manifest"]
    spark_gate = next(gate for gate in closure["gates"] if gate["gate_id"] == "local_spark_stack_matrix")
    assert spark_gate["closure_plan"].endswith("spark-matrix-closure-plan.json")
    assert spark_gate["closure_artifact"].endswith("spark-matrix-closure-plan.md")
    assert "VoiceOps Plan Run Summary" in markdown
    assert "Readiness Closure" in markdown
    assert "VoiceOps Readiness Closure Index" in closure_markdown
    assert "live_discord_voice_operator" in closure_markdown
    assert "milestone_0_hackathon_proof" in markdown


def test_goal_doc_lists_voiceops_closure_artifacts():
    text = GOAL_DOC.read_text(encoding="utf-8")

    for artifact in [
        "live-voice-evidence-template.json",
        "live-voice-evidence.example.json",
        "live-probe-closure-plan.json",
        "live-probe-closure-plan.md",
        "provisioning-preflight-evidence.template.json",
        "provisioning-preflight-evidence.example.json",
        "provisioning-preflight-evidence.manifest.example.json",
        "setup-closure-plan.json",
        "setup-closure-plan.md",
        "spark-benchmark-evidence-template.json",
        "spark-benchmark-evidence.example.json",
        "spark-matrix-closure-plan.json",
        "spark-matrix-closure-plan.md",
        "readiness-closure-index.json",
        "readiness-closure-index.md",
    ]:
        assert f"`{artifact}`" in text


def test_goal_doc_keeps_super_local_and_ultra_hosted():
    text = GOAL_DOC.read_text(encoding="utf-8")

    assert "Nemotron 3 Super is the preferred Spark-local NVIDIA oracle/model target" in text
    assert "Nemotron 3 Ultra is the hosted fallback" in text
    assert "Ultra is only an optional hosted/upstream fallback" in text
    assert "must not be used as Spark-local readiness proof" in text
    assert "There should not be a separate `oracle_model` setting for VoiceOps" in text
    assert "`/model` remains authoritative" in text


def test_plan_run_keeps_provisioning_incomplete_without_preflight_evidence(tmp_path):
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
    assert provisioning_result["status"] == "needs_setup"
    assert provisioning_result["details"]["preflight_evidence_loaded"] is False
    assert "stripe_projects_account" in provisioning_result["details"]["required_failures"]
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
    assert Path(payload["artifacts"]["closure_json"]).exists()
    assert Path(payload["artifacts"]["closure_markdown"]).exists()


def test_parse_args_defaults_to_plan_artifact_paths():
    args = parse_args([])

    assert args.artifact_root == Path("artifacts")
    assert args.output_dir == Path("artifacts/voiceops-plan/current")
    assert args.voice_live_evidence == []
    assert args.provisioning_preflight_evidence is None
    assert args.run_command_probes is False
