from __future__ import annotations

import json
from pathlib import Path

from scripts.voiceops_artifact_package_audit import audit_package, parse_args, write_audit
from scripts.voiceops_plan_run import build_plan_run, write_plan_run


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generate_package(tmp_path: Path) -> Path:
    artifact_root = tmp_path / "artifacts"
    output_dir = artifact_root / "voiceops-plan" / "current"
    summary = build_plan_run(artifact_root=artifact_root, output_dir=output_dir, env={})
    write_plan_run(output_dir, summary)
    return artifact_root


def test_package_audit_accepts_generated_headless_package(tmp_path):
    artifact_root = _generate_package(tmp_path)

    report = audit_package(artifact_root)
    paths = write_audit(tmp_path / "audit", report)

    assert report["schema_version"] == "voiceops.artifact_package_audit.v1"
    assert report["artifact_id"] == "voiceops-artifact-package-audit"
    assert report["status"] == "pass"
    assert report["ok"] is True
    assert report["issues"] == []
    assert report["safety"] == {
        "discord_io": False,
        "env_files_read": False,
        "live_spend": False,
        "network_io": False,
        "outbound_calls": False,
        "outbound_messages": False,
        "provider_provisioning": False,
        "secret_values_emitted": False,
        "spark_execution": False,
    }
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith("# VoiceOps Artifact Package Audit")


def test_package_audit_rejects_live_dashboard_claim_with_open_gates(tmp_path):
    artifact_root = _generate_package(tmp_path)
    dashboard = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-dashboard.html"
    dashboard.write_text(
        dashboard.read_text(encoding="utf-8")
        .replace("scripted_static_ack_until_live_voice_evidence", "live_voice_ready")
        .replace("needs_live_probe", "live_probe_complete"),
        encoding="utf-8",
    )

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "dashboard:missing_non_live_token:scripted_static_ack_until_live_voice_evidence" in report["issues"]
    assert "dashboard:missing_non_live_token:needs_live_probe" in report["issues"]


def test_package_audit_rejects_nemoclaw_operator_contract_mismatch(tmp_path):
    artifact_root = _generate_package(tmp_path)
    packet_path = artifact_root / "hackathon-voiceops-demo" / "current" / "nemoclaw-action-packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["approval_required_actions"][0]["approval_contract"]["command_sha256"] = "0" * 64
    packet["approval_contracts"]["provision-voip-provider"]["command_sha256"] = "0" * 64
    _write_json(packet_path, packet)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "nemoclaw:provision-voip-provider:command_sha256_mismatch" in report["issues"]
    assert "operator_state:provision-voip-provider:approval_contract_mismatch" in report["issues"]
    assert "operator_state:provision-voip-provider:pending_contract_mismatch" in report["issues"]


def test_package_audit_rejects_external_service_execution_claim(tmp_path):
    artifact_root = _generate_package(tmp_path)
    state_path = artifact_root / "hackathon-voiceops-demo" / "current" / "operator-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["planned_services"][0]["execution_status"] = "executed"
    state["planned_services"][0]["status"] = "provisioned"
    _write_json(state_path, state)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "planned_services:provision-voip-provider:external_execution_claim" in report["issues"]
    assert "planned_services:provision-voip-provider:external_status_invalid" in report["issues"]


def test_package_audit_rejects_closure_gate_mismatch(tmp_path):
    artifact_root = _generate_package(tmp_path)
    closure_path = artifact_root / "voiceops-plan" / "current" / "readiness-closure-index.json"
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    closure["gates"] = closure["gates"][:-1]
    _write_json(closure_path, closure)

    report = audit_package(artifact_root)

    assert report["ok"] is False
    assert "closure:gates_mismatch_between_demo_and_plan" in report["issues"]


def test_package_audit_parse_args_defaults():
    args = parse_args([])

    assert args.artifact_root == Path("artifacts")
    assert args.output_dir == Path("artifacts/voiceops-package-audit/current")
    assert args.audit_only is False
