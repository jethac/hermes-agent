from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.voiceops_plan_run import build_plan_run, parse_args, write_plan_run
from scripts.voiceops_provisioning_probe import build_milestone2_execution_plan, build_probe_report


GOAL_DOC = Path(__file__).resolve().parents[2] / "docs" / "plans" / "2026-06-29-spark-household-business-voiceops.md"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_fake_bin(bin_dir: Path, name: str) -> None:
    path = bin_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env sh\nprintf '%s\\n' mock\n", encoding="utf-8")
    path.chmod(0o755)


def _write_live_voice_evidence(root: Path) -> Path:
    _write_json(
        root / "discord-live-probe.json",
        {
            "kind": "discord_live_probe",
            "ok": True,
            "connect_perm": True,
            "speak_perm": True,
            "connected": True,
            "opus_loaded": True,
            "accepted_audio_source": True,
            "played": True,
            "playing_during_probe": True,
            "receiver_started": True,
            "receiver_frames": 12,
            "receiver_speech_start": 1,
            "inbound_observed": True,
            "disconnected": True,
            "require_inbound": True,
        },
    )
    _write_json(
        root / "sidecar-session.json",
        {
            "kind": "sidecar_session",
            "sidecar_running": True,
            "sidecar_healthy": True,
            "session_started": True,
            "session_closed": True,
            "shutdown_bounded": True,
            "shutdown_timed_out": False,
            "fallback_mode_visible": True,
            "latency_metrics_ms": {"shutdown_ms": 80},
        },
    )
    _write_json(
        root / "live-turn.json",
        {
            "kind": "live_turn",
            "transcript_observed": True,
            "assistant_audio_observed": True,
            "barge_in_observed": True,
            "spoken_reply_short": True,
            "no_voice_denial_observed": True,
            "speech_end_to_first_audio_ms": 900,
            "barge_in_stop_ms": 90,
        },
    )
    return _write_json(
        root / "manifest.json",
        {
            "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
            "reports": {
                "discord_live_probe": "discord-live-probe.json",
                "sidecar_session": "sidecar-session.json",
                "live_turn": "live-turn.json",
            },
        },
    )


def _write_preflight_evidence(root: Path) -> Path:
    sections_dir = root / "sections"
    sources_dir = root / "sources"
    section_payloads = {
        "stripe_projects": {
            "account_ref": "stripe-projects-account-ref-demo",
            "projects_catalog_checked_at": "2026-06-29T00:00:00Z",
            "voip_provider_candidate": "twilio/voice",
            "can_create_project_after_approval": True,
        },
        "stripe_link": {
            "account_ref": "stripe-link-account-ref-demo",
            "approval_capability_confirmed": True,
            "max_approved_cents": 20_000,
            "currency": "usd",
        },
        "mpp": {
            "boundary_tool": "nemoclaw",
            "policy_ref": "nemoclaw-policy-ref-demo",
            "approval_packet_ref": "nemoclaw-action-packet.json",
        },
        "phone_handoff": {
            "provider": "twilio",
            "provider_account_ref": "twilio-account-ref-demo",
            "phone_target_ref": "operator-phone-ref-demo",
            "credential_location_ref": "1password://VoiceOps/Twilio Demo Credential Ref",
        },
        "rollback": {
            "deprovision_owner": "operator-ref-demo",
            "refund_or_cancel_owner": "operator-ref-demo",
            "call_cancel_owner": "operator-ref-demo",
        },
    }
    report_names = {
        "stripe_projects": "stripe-projects-evidence.json",
        "stripe_link": "stripe-link-evidence.json",
        "mpp": "nemoclaw-boundary-evidence.json",
        "phone_handoff": "phone-handoff-evidence.json",
        "rollback": "rollback-owner-evidence.json",
    }
    reports = {}
    for section_name, payload in section_payloads.items():
        source_path = _write_json(
            sources_dir / f"{section_name}-source.json",
            {
                "schema_version": "voiceops.milestone2.redacted_source_artifact.v1",
                "section": section_name,
                "redacted": True,
                "redaction_policy": "references only; no raw secrets, tokens, cards, or full phone numbers",
                "summary": f"redacted local rehearsal source for {section_name}",
            },
        )
        section = {
            **payload,
            "source_artifact": f"../sources/{source_path.name}",
            "source_artifact_kind": "redacted_setup_evidence",
            "source_artifact_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "source_artifact_redacted_at": "2026-06-29T00:00:00Z",
        }
        report_path = _write_json(sections_dir / report_names[section_name], {section_name: section})
        reports[section_name] = f"sections/{report_path.name}"
    return _write_json(
        root / "provisioning-preflight-evidence.manifest.json",
        {
            "schema_version": "voiceops.milestone2.preflight_evidence_manifest.v1",
            "reports": reports,
        },
    )


def _write_post_approval_receipts(root: Path) -> Path:
    report = build_probe_report(env={}, env_files=[], which=lambda _command: None)
    plan = build_milestone2_execution_plan(report)
    action = next(item for item in plan["approval_required_actions"] if item["action_id"] == "provision-voip-provider")
    return _write_json(
        root / "post-approval-receipts.json",
        {
            "schema_version": "voiceops.milestone2.post_approval_receipts.v1",
            "redaction_policy": "references only",
            "receipts": [
                {
                    "receipt_id": "receipt-provision-voip-provider-001",
                    "action_id": action["action_id"],
                    "approval_id": action["approval_id"],
                    "provider": action["provider"],
                    "status": "executed",
                    "approved_by": "operator-ref-demo",
                    "approved_at": "2026-06-29T00:00:00Z",
                    "executed_at": "2026-06-29T00:00:30Z",
                    "command_sha256": action["command_sha256"],
                    "amount_cents": 0,
                    "currency": "usd",
                    "external_reference": "provider-resource-ref-demo",
                    "credential_location_ref": action["credential_location_ref"],
                    "rollback_ref": action["rollback_ref"],
                    "audit_event_id": "audit-provision-voip-provider-001",
                }
            ],
            "credential_locations": [
                {
                    "credential_ref_id": action["credential_location_ref"],
                    "provider": action["provider"],
                    "service_id": "provider-resource-ref-demo",
                    "storage_backend": "provider_managed",
                    "secret_name_or_path": "credential-location-ref-demo",
                    "created_by_action_id": action["action_id"],
                    "rotation_due": "2026-09-29T00:00:00Z",
                }
            ],
            "rollback_receipts": [
                {
                    "rollback_ref": action["rollback_ref"],
                    "status": "not_run",
                    "owner_ref": "operator-ref-demo",
                    "notes": "No rollback run.",
                }
            ],
            "audit_events": [
                {
                    "audit_event_id": "audit-provision-voip-provider-001",
                    "action_id": action["action_id"],
                    "receipt_id": "receipt-provision-voip-provider-001",
                    "status": "executed",
                    "artifact_ref": "post-approval-receipts.json",
                    "operator_next_step": "Review provider dashboard and rollback window.",
                }
            ],
        },
    )


def _base_spark_evidence(candidate_id: str, *, model: str, source_artifact: str) -> dict:
    return {
        "schema_version": "voiceops.spark_benchmark_evidence.v1",
        "candidate_id": candidate_id,
        "hardware": "1x NVIDIA DGX Spark",
        "locality": "local_spark",
        "model": model,
        "engine": "local rehearsal engine",
        "verified": True,
        "measured_at": "2026-06-29T00:00:00Z",
        "source_artifact": source_artifact,
        "metrics": {},
    }


def _write_spark_evidence(root: Path) -> Path:
    sources = root / "sources"
    for name in ["reflex", "oracle", "asr", "tts", "stack-smoke"]:
        _write_json(
            sources / f"{name}.json",
            {"redacted": True, "source": name, "summary": "local DGX Spark rehearsal source"},
        )
    return _write_json(
        root / "spark-benchmark-evidence.json",
        {
            "evidence": [
                {
                    **_base_spark_evidence(
                        "reflex-gemma4-e2b",
                        model="Gemma 4 E2B audio-native",
                        source_artifact="sources/reflex.json",
                    ),
                    "metrics": {"first_token_ms": 700, "intent_latency_ms": 1100, "steady_state_memory_gb": 20},
                },
                {
                    **_base_spark_evidence(
                        "oracle-nemotron3-super-local",
                        model="Nemotron 3 Super",
                        source_artifact="sources/oracle.json",
                    ),
                    "metrics": {
                        "decode_tok_s": 24,
                        "prefill_tok_s": 3100,
                        "first_token_ms": 2100,
                        "steady_state_memory_gb": 86,
                    },
                },
                {
                    **_base_spark_evidence(
                        "asr-nemotron-speech",
                        model="Nemotron Speech streaming",
                        source_artifact="sources/asr.json",
                    ),
                    "metrics": {"asr_delta_ms": 30, "final_transcript_ms": 600, "word_error_rate": 0.08},
                },
                {
                    **_base_spark_evidence(
                        "tts-magpie-local",
                        model="Magpie local TTS",
                        source_artifact="sources/tts.json",
                    ),
                    "metrics": {"tts_first_audio_ms": 200, "underrun_count": 0},
                },
                {
                    "schema_version": "voiceops.spark_benchmark_evidence.v1",
                    "kind": "voiceops_spark_stack_smoke",
                    "hardware": "1x NVIDIA DGX Spark",
                    "locality": "local_spark",
                    "verified": True,
                    "measured_at": "2026-06-29T00:00:00Z",
                    "source_artifact": "sources/stack-smoke.json",
                    "oracle_selected_by": "Hermes /model",
                    "oracle_authority_routes": ["tools", "files", "memory", "project_context"],
                    "interface_input_sources": ["native_audio"],
                    "reflex_providers": ["vllm"],
                    "components": {"reflex": True, "oracle": True, "asr": True, "tts": True, "sidecar": True},
                    "metrics": {
                        "speech_end_to_first_audio_ms": 900,
                        "barge_in_stop_ms": 90,
                        "local_turns": 2,
                        "local_turn_oracle_calls": 0,
                        "oracle_bound_turns": 4,
                        "oracle_bound_oracle_calls": 4,
                    },
                },
            ]
        },
    )


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
    assert summary["current_environment"]["schema_version"] == "voiceops.current_environment.v1"
    assert summary["current_environment"]["redaction_policy"].startswith("presence booleans only")
    assert summary["current_environment"]["discord"]["env_presence"]["DISCORD_BOT_TOKEN"] is False
    assert summary["current_environment"]["discord"]["live_probe_can_run_here"] is False
    assert summary["current_environment"]["provisioning"]["required_cli_presence"] == {
        "stripe": False,
        "link-cli": False,
        "mppx": False,
    }
    assert summary["current_environment"]["spark"]["hardware_claim"] == "not_verified_by_plan_run"
    assert "dgx_spark_likely" in summary["current_environment"]["spark"]
    assert summary["closure_index"]["current_environment"] == summary["current_environment"]
    blockers = summary["closure_index"]["current_environment_blockers"]
    assert blockers["hard_failure"] is False
    assert blockers["secret_values_emitted"] is False
    assert blockers["diagnostic_only"] is True
    assert "DISCORD_BOT_TOKEN" in blockers["discord_env"]["missing_env_keys"]
    assert "stripe" in blockers["provisioning_cli"]["missing"]
    assert "link-cli" in blockers["provisioning_cli"]["missing"]
    assert "mppx_or_fallback" in blockers["provisioning_cli"]["missing"]
    assert blockers["spark_host"]["required_hardware"] == "1x NVIDIA DGX Spark"
    assert blockers["spark_host"]["blocks_artifact_generation"] is False
    handoff = summary["closure_index"]["operator_handoff"]
    assert handoff["schema_version"] == "voiceops.operator_handoff.v1"
    assert handoff["changes_readiness_by_itself"] is False
    assert handoff["final_success_signal"] == "readiness_gaps is [] and closure_status is complete"
    assert [phase["phase_id"] for phase in handoff["phases"]] == [
        "live_discord_voice",
        "spend_and_provisioning_preflight",
        "local_spark_stack",
    ]
    assert handoff["phases"][0]["can_run_here_now"] is False
    assert "sidecar-session.json" in json.dumps(handoff["phases"][0]["expected_artifacts"])
    assert "live-evidence-validation.json" in json.dumps(handoff["phases"][0]["expected_artifacts"])
    assert "python -m hermes_cli.realtime_voice_live_evidence" in handoff["phases"][0]["commands"][0]
    assert "--validate-live-evidence" in json.dumps(handoff["phases"][0]["commands"])
    assert "provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json" in json.dumps(
        handoff["phases"][1]
    )
    assert "--refresh-preflight-source-hashes" in json.dumps(handoff["phases"][1]["commands"])
    assert "--post-approval-receipts" in json.dumps(handoff["phases"][1]["commands"])
    assert "scripts/dgx_spark_gemma4_voice_eval.sh" in handoff["phases"][2]["commands"]
    assert "path/to/spark-benchmark-evidence.json" in handoff["final_reindex_command"]
    assert "--post-approval-receipts" in handoff["final_reindex_command"]
    gates = {gate["gate_id"]: gate for gate in summary["closure_index"]["gates"]}
    assert set(gates) == {
        "live_discord_voice_operator",
        "local_spark_stack_matrix",
        "spend_and_provisioning_preflight",
    }
    assert "plan_index_manifest_and_post_approval_receipts" in json.dumps(
        gates["spend_and_provisioning_preflight"]["rerun_commands"]
    )
    assert "schema_version" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "transcript_observed" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert gates["live_discord_voice_operator"]["evidence_contract"] == {
        "manifest_schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
        "strict_validation_schema_version": "voiceops.realtime_voice_live_evidence_validation.v1",
        "expanded_evidence_schema_version": "voiceops.milestone1.live_voice_evidence.v1",
        "required_sections": ["discord_live_probe", "sidecar_session", "live_turn"],
        "required_section_refs": ["source_artifact", "section"],
        "source_artifacts_must_exist": True,
        "example_only_accepted": False,
    }
    assert "operator_must_not" in gates["live_discord_voice_operator"]
    assert "manifest.json" in gates["live_discord_voice_operator"]["rerun_command"]
    assert "python -m hermes_cli.realtime_voice_live_evidence" in gates["live_discord_voice_operator"]["collection_commands"]["collect_live_manifest"]
    assert "--validate-live-evidence" in gates["live_discord_voice_operator"]["collection_commands"][
        "validate_live_manifest_offline"
    ]
    assert "--discord-live-probe-evidence" in gates["live_discord_voice_operator"]["collection_commands"][
        "validate_live_manifest_offline"
    ]
    assert "--sidecar-session-evidence" in gates["live_discord_voice_operator"]["collection_commands"]["collect_live_manifest"]
    assert "--live-turn-evidence" in gates["live_discord_voice_operator"]["collection_commands"]["collect_live_manifest"]
    assert gates["live_discord_voice_operator"]["current_environment"]["env_presence"]["DISCORD_BOT_TOKEN"] is False
    assert "missing_preflight_fields" in gates["spend_and_provisioning_preflight"]
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"]["preflight_schema_version"] == (
        "voiceops.milestone2.preflight_evidence.v1"
    )
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"]["required_section_field"] == "source_artifact"
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"]["required_section_provenance_fields"] == [
        "source_artifact_kind",
        "source_artifact_sha256",
        "source_artifact_redacted_at",
    ]
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"]["source_artifacts_must_exist"] is True
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"]["source_artifact_sha256_must_match"] is True
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"]["manifest_report_resolution"].endswith(
        "process cwd is never used"
    )
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"]["example_only_accepted"] is False
    assert "provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json" in gates[
        "spend_and_provisioning_preflight"
    ]["collection_commands"]["ingest_preflight_manifest"]
    assert gates["spend_and_provisioning_preflight"]["current_environment"]["required_cli_presence"]["stripe"] is False
    assert "required_candidate_fields" in gates["local_spark_stack_matrix"]
    assert "schema_version" in gates["local_spark_stack_matrix"]["required_candidate_fields"]
    assert gates["local_spark_stack_matrix"]["evidence_contract"]["hosted_fallback_counts_for_one_spark_readiness"] is False
    assert gates["local_spark_stack_matrix"]["evidence_contract"]["source_artifacts_must_exist"] is True
    assert gates["local_spark_stack_matrix"]["evidence_contract"]["source_artifact_readable"] is True
    assert gates["local_spark_stack_matrix"]["evidence_contract"]["source_artifact_resolution"].endswith(
        "supplied benchmark evidence file"
    )
    assert "all_local_stack_smoke:needs_evidence" in gates["local_spark_stack_matrix"]["missing"]
    assert "all_local_stack_smoke is validated" in gates["local_spark_stack_matrix"]["completion_signal"]
    assert gates["local_spark_stack_matrix"]["collection_commands"]["dgx_eval"] == "scripts/dgx_spark_gemma4_voice_eval.sh"
    assert "host_system" in gates["local_spark_stack_matrix"]["current_environment"]
    assert summary["hard_failures"] == []
    assert "milestone_1_real_voice_operator" in summary["readiness_gaps"]
    assert "milestone_2_real_spend_and_provisioning_preflight" in summary["readiness_gaps"]
    assert "milestone_4_local_spark_stack_matrix" in summary["readiness_gaps"]
    assert summary["safety"] == {
        "env_presence_inspection": True,
        "env_secret_values_emitted": False,
        "live_spend": False,
        "mutating_network_io": False,
        "network_io": False,
        "network_io_scope": "none",
        "outbound_calls": False,
        "outbound_sends": False,
        "provider_provisioning": False,
        "read_only_discovery_grants_approval": False,
        "read_only_discovery_run_requested": False,
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
    assert Path(demo_result["artifacts"]["readiness_closure_summary_json"]).exists()
    assert Path(demo_result["artifacts"]["readiness_closure_summary_markdown"]).exists()

    voice_result = next(result for result in summary["results"] if result["milestone"] == "milestone_1_real_voice_operator")
    assert voice_result["status"] == "needs_live_probe"
    assert voice_result["details"]["live_probe_status"] == "needs_live_probe"
    assert "live_probe_missing_gates" in voice_result["details"]
    assert Path(voice_result["artifacts"]["json"]).exists()
    assert Path(voice_result["artifacts"]["markdown"]).exists()
    assert Path(voice_result["artifacts"]["smoke_json"]).exists()
    assert Path(voice_result["artifacts"]["events_jsonl"]).exists()
    assert Path(voice_result["artifacts"]["live_evidence_example"]).exists()
    assert Path(voice_result["artifacts"]["live_evidence_scaffold_manifest"]).exists()
    assert Path(voice_result["artifacts"]["live_evidence_template"]).exists()
    assert Path(voice_result["artifacts"]["live_probe_closure_json"]).exists()

    provisioning_result = next(
        result for result in summary["results"] if result["milestone"] == "milestone_2_real_spend_and_provisioning_preflight"
    )
    assert provisioning_result["status"] == "needs_setup"
    assert provisioning_result["details"]["required_failures"]
    assert provisioning_result["details"]["run_command_probes"] is False
    assert provisioning_result["details"]["run_readonly_discovery"] is False
    assert provisioning_result["details"]["read_only_discovery_status"] == "not_requested"
    assert provisioning_result["details"]["post_approval_receipts_loaded"] is False
    assert provisioning_result["details"]["post_approval_receipts_status"] == "not_supplied"
    assert provisioning_result["details"]["post_approval_receipt_count"] == 0
    assert Path(provisioning_result["artifacts"]["execution_plan_json"]).exists()
    assert Path(provisioning_result["artifacts"]["execution_plan_markdown"]).exists()
    assert Path(provisioning_result["artifacts"]["post_approval_receipts_template"]).exists()
    assert Path(provisioning_result["artifacts"]["post_approval_receipts_validation"]).exists()
    assert Path(provisioning_result["artifacts"]["post_approval_audit_ledger"]).exists()
    assert Path(provisioning_result["artifacts"]["read_only_discovery_json"]).exists()
    assert Path(provisioning_result["artifacts"]["read_only_discovery_markdown"]).exists()
    assert Path(provisioning_result["artifacts"]["read_only_discovery_manifest"]).exists()
    assert Path(provisioning_result["artifacts"]["read_only_discovery_audit_ledger"]).exists()
    assert Path(provisioning_result["artifacts"]["preflight_evidence_example"]).exists()
    assert Path(provisioning_result["artifacts"]["preflight_evidence_manifest_example"]).exists()
    assert Path(provisioning_result["artifacts"]["preflight_evidence_scaffold_manifest"]).exists()

    channel_result = next(result for result in summary["results"] if result["milestone"] == "milestone_3_multi_channel_policy")
    assert channel_result["status"] == "needs_review"
    assert channel_result["details"]["validation_issues"] == []
    assert channel_result["details"]["review_required_for_real_egress"] is True
    assert channel_result["details"]["review_status"] == "pending_human_review"
    assert channel_result["details"]["real_egress_enabled"] is False
    assert channel_result["details"]["review_packet_schema_version"] == "voiceops.multi_channel_policy_review.v1"
    assert channel_result["details"]["review_packet_status"] == "pending_human_review"
    assert channel_result["details"]["review_packet_artifact_only"] is True
    assert channel_result["details"]["review_packet_changes_policy"] is False
    assert Path(channel_result["artifacts"]["review_json"]).exists()
    assert Path(channel_result["artifacts"]["review_markdown"]).exists()
    assert "milestone_3_multi_channel_policy" not in summary["readiness_gaps"]

    matrix_result = next(result for result in summary["results"] if result["milestone"] == "milestone_4_local_spark_stack_matrix")
    assert matrix_result["status"] == "needs_evidence"
    assert matrix_result["details"]["ready_for_one_spark_demo"] is False
    assert matrix_result["details"]["stack_smoke_status"] == "needs_evidence"
    assert Path(matrix_result["artifacts"]["closure_json"]).exists()
    assert Path(matrix_result["artifacts"]["closure_markdown"]).exists()
    assert Path(matrix_result["artifacts"]["evidence_example"]).exists()
    assert Path(matrix_result["artifacts"]["evidence_scaffold"]).exists()

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    closure = json.loads(Path(paths["closure_json"]).read_text(encoding="utf-8"))
    handoff_payload = json.loads(Path(paths["operator_handoff_json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    closure_markdown = Path(paths["closure_markdown"]).read_text(encoding="utf-8")
    handoff_markdown = Path(paths["operator_handoff_markdown"]).read_text(encoding="utf-8")
    assert payload["ok"] is True
    assert closure["artifact_id"] == "voiceops-plan-readiness-closure"
    assert closure["schema_version"] == "voiceops.closure_index.v1"
    assert handoff_payload == closure["operator_handoff"]
    assert handoff_payload["schema_version"] == "voiceops.operator_handoff.v1"
    voice_gate = next(gate for gate in closure["gates"] if gate["gate_id"] == "live_discord_voice_operator")
    assert voice_gate["evidence_scaffold"].endswith("live-voice-evidence-scaffold/manifest.json")
    provisioning_gate = next(gate for gate in closure["gates"] if gate["gate_id"] == "spend_and_provisioning_preflight")
    assert provisioning_gate["evidence_manifest_example"].endswith(
        "provisioning-preflight-evidence.manifest.example.json"
    )
    assert "provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json" in provisioning_gate[
        "rerun_commands"
    ]["plan_index_manifest"]
    combined_receipt_command = provisioning_gate["rerun_commands"]["plan_index_manifest_and_post_approval_receipts"]
    assert "provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json" in combined_receipt_command
    assert "--post-approval-receipts" in combined_receipt_command
    assert provisioning_gate["evidence_scaffold"].endswith(
        "provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json"
    )
    assert "validate_post_approval_receipts" in provisioning_gate["collection_commands"]
    assert "read_only_discovery" in provisioning_gate["collection_commands"]
    assert "refresh_preflight_source_hashes" in provisioning_gate["collection_commands"]
    assert "--refresh-preflight-source-hashes" in provisioning_gate["collection_commands"]["refresh_preflight_source_hashes"]
    assert "--run-readonly-discovery" in provisioning_gate["collection_commands"]["read_only_discovery"]
    assert "--run-readonly-discovery" in provisioning_gate["rerun_commands"]["plan_index_read_only_discovery"]
    assert "--refresh-preflight-source-hashes" in provisioning_gate["rerun_commands"]["refresh_preflight_source_hashes"]
    assert provisioning_gate["evidence_contract"]["read_only_discovery_grants_approval"] is False
    assert "voiceops.milestone2.post_approval_receipts.v1" == provisioning_gate["evidence_contract"][
        "post_approval_receipts_schema_version"
    ]
    assert "--post-approval-receipts" in provisioning_gate["rerun_commands"]["plan_index_post_approval_receipts"]
    assert "--post-approval-receipts" in handoff_payload["final_reindex_command"]
    spark_gate = next(gate for gate in closure["gates"] if gate["gate_id"] == "local_spark_stack_matrix")
    assert spark_gate["closure_plan"].endswith("spark-matrix-closure-plan.json")
    assert spark_gate["closure_artifact"].endswith("spark-matrix-closure-plan.md")
    assert spark_gate["operator_runbook"].endswith("spark-operator-runbook.md")
    assert spark_gate["evidence_scaffold"].endswith("spark-benchmark-scaffold/spark-benchmark-evidence.json")
    assert "VoiceOps Plan Run Summary" in markdown
    assert "Readiness Closure" in markdown
    assert "Current Environment" in markdown
    assert "Current Environment Blockers" in markdown
    assert "Operator Handoff" in markdown
    assert "VoiceOps Readiness Closure Index" in closure_markdown
    assert "presence booleans only" in closure_markdown
    assert "Diagnostic only" in closure_markdown
    assert "voiceops.operator_handoff.v1" in closure_markdown
    assert "Final reindex command" in closure_markdown
    assert "live_discord_voice_operator" in closure_markdown
    assert "voiceops.realtime_voice_live_evidence_manifest.v1" in closure_markdown
    assert "python -m hermes_cli.realtime_voice_live_evidence" in closure_markdown
    assert "--sidecar-session-evidence" in closure_markdown
    assert "--live-turn-evidence" in closure_markdown
    assert "voiceops.milestone2.preflight_evidence_manifest.v1" in closure_markdown
    assert "provisioning-preflight-evidence.manifest.json" in closure_markdown
    assert "--refresh-preflight-source-hashes" in closure_markdown
    assert "voiceops.spark_benchmark_evidence.v1" in closure_markdown
    assert "spark-benchmark-evidence.example.json" in closure_markdown
    assert "spark-operator-runbook.md" in closure_markdown
    assert "scripts/dgx_spark_gemma4_voice_eval.sh" in closure_markdown
    assert "VoiceOps Operator Handoff" in handoff_markdown
    assert "live_discord_voice" in handoff_markdown
    assert "Final reindex command" in handoff_markdown
    assert "milestone_0_hackathon_proof" in markdown


def test_plan_run_closes_remaining_gates_with_redacted_local_evidence(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    output_dir = artifact_root / "voiceops-plan" / "current"
    fake_bin = tmp_path / "bin"
    for binary in ["stripe", "link-cli", "mppx"]:
        _write_fake_bin(fake_bin, binary)
    monkeypatch.setenv("PATH", str(fake_bin))
    live_evidence = _write_live_voice_evidence(tmp_path / "live-voice")
    preflight_evidence = _write_preflight_evidence(tmp_path / "preflight")
    post_approval_receipts = _write_post_approval_receipts(tmp_path / "post-approval")
    spark_evidence = _write_spark_evidence(tmp_path / "spark")

    summary = build_plan_run(
        artifact_root=artifact_root,
        output_dir=output_dir,
        env={
            "PATH": str(fake_bin),
            "DISCORD_BOT_TOKEN": "present-redacted",
            "DISCORD_GUILD_ID": "guild-ref-demo",
            "DISCORD_HOME_CHANNEL": "channel-ref-demo",
            "DISCORD_VOICE_CHANNEL_ID": "voice-channel-ref-demo",
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "TWILIO_ACCOUNT_SID": "AC123",
        },
        voice_live_evidence_paths=[live_evidence],
        provisioning_preflight_evidence=preflight_evidence,
        post_approval_receipts=post_approval_receipts,
        evidence_paths=[spark_evidence],
    )

    statuses = {result["milestone"]: result["status"] for result in summary["results"]}
    provisioning_result = next(
        result for result in summary["results"] if result["milestone"] == "milestone_2_real_spend_and_provisioning_preflight"
    )
    serialized = json.dumps(summary)

    assert summary["ok"] is True
    assert summary["hard_failures"] == []
    assert summary["readiness_gaps"] == []
    assert summary["closure_index"]["closure_status"] == "complete"
    assert summary["closure_index"]["remaining_gates"] == []
    assert statuses["milestone_1_real_voice_operator"] == "live_evidence_supplied"
    assert statuses["milestone_2_real_spend_and_provisioning_preflight"] == "ready"
    assert statuses["milestone_4_local_spark_stack_matrix"] == "validated"
    assert provisioning_result["details"]["post_approval_receipts_loaded"] is True
    assert provisioning_result["details"]["post_approval_receipts_status"] == "valid"
    assert provisioning_result["details"]["post_approval_receipt_count"] == 1
    assert provisioning_result["details"]["post_approval_receipts_validation_issues"] == []
    assert summary["safety"] == {
        "env_presence_inspection": True,
        "env_secret_values_emitted": False,
        "live_spend": False,
        "mutating_network_io": False,
        "network_io": False,
        "network_io_scope": "none",
        "outbound_calls": False,
        "outbound_sends": False,
        "provider_provisioning": False,
        "read_only_discovery_grants_approval": False,
        "read_only_discovery_run_requested": False,
    }
    assert summary["closure_index"]["operator_handoff"]["changes_readiness_by_itself"] is False
    assert summary["closure_index"]["operator_handoff"]["final_success_signal"] == (
        "readiness_gaps is [] and closure_status is complete"
    )
    assert "present-redacted" not in serialized
    assert "+15551234567" not in serialized
    assert "sk_live" not in serialized
    assert "DISCORD_BOT_TOKEN" in summary["closure_index"]["current_environment_blockers"]["discord_env"][
        "present_env_keys"
    ]


def test_plan_run_readonly_discovery_safety_is_evidence_derived(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    output_dir = artifact_root / "voiceops-plan" / "current"
    fake_bin = tmp_path / "bin"
    for binary in ["stripe", "link-cli", "mppx"]:
        _write_fake_bin(fake_bin, binary)
    fake_path = os.pathsep.join([str(fake_bin), os.environ.get("PATH", "")])
    monkeypatch.setenv("PATH", fake_path)

    summary = build_plan_run(
        artifact_root=artifact_root,
        output_dir=output_dir,
        env={"PATH": fake_path},
        run_readonly_discovery=True,
    )
    paths = write_plan_run(output_dir, summary)
    provisioning_result = next(
        result for result in summary["results"] if result["milestone"] == "milestone_2_real_spend_and_provisioning_preflight"
    )
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    closure_markdown = Path(paths["closure_markdown"]).read_text(encoding="utf-8")

    assert provisioning_result["details"]["run_readonly_discovery"] is True
    assert provisioning_result["details"]["read_only_discovery_status"] == "pass"
    assert summary["safety"]["network_io"] is True
    assert summary["safety"]["network_io_scope"] == "allowlisted_read_only_discovery"
    assert summary["safety"]["mutating_network_io"] is False
    assert summary["safety"]["read_only_discovery_run_requested"] is True
    assert summary["safety"]["read_only_discovery_grants_approval"] is False
    assert summary["safety"]["live_spend"] is False
    assert summary["safety"]["provider_provisioning"] is False
    assert summary["closure_index"]["safety"]["network_io"] is True
    assert summary["closure_index"]["safety"]["network_io_scope"] == "allowlisted_read_only_discovery"
    assert summary["closure_index"]["safety"]["mutating_network_io"] is False
    assert summary["closure_index"]["safety"]["live_spend"] is False
    assert summary["closure_index"]["safety"]["provider_provisioning"] is False
    assert "read-only discovery network possible only when explicitly requested" in markdown
    assert "read-only discovery network possible only when explicitly requested" in closure_markdown


def test_goal_doc_lists_voiceops_closure_artifacts():
    text = GOAL_DOC.read_text(encoding="utf-8")

    for artifact in [
        "live-voice-evidence-template.json",
        "live-voice-evidence.example.json",
        "live-voice-evidence-scaffold/manifest.json",
        "live-probe-closure-plan.json",
        "live-probe-closure-plan.md",
        "provisioning-preflight-evidence.template.json",
        "provisioning-preflight-evidence.example.json",
        "provisioning-preflight-evidence.manifest.example.json",
        "provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json",
        "setup-closure-plan.json",
        "setup-closure-plan.md",
        "read-only-discovery.json",
        "read-only-discovery.md",
        "read-only-discovery.manifest.json",
        "audit-ledger.read-only-discovery.jsonl",
        "post-approval-receipts.template.json",
        "post-approval-receipts.example.json",
        "post-approval-receipts.validation.json",
        "post-approval-receipts-scaffold/post-approval-receipts.json",
        "audit-ledger.post-approval.jsonl",
        "spark-benchmark-evidence-template.json",
        "spark-benchmark-evidence.example.json",
        "spark-benchmark-scaffold/spark-benchmark-evidence.json",
        "spark-matrix-closure-plan.json",
        "spark-matrix-closure-plan.md",
        "spark-operator-runbook.md",
        "readiness-closure-index.json",
        "readiness-closure-index.md",
        "operator-handoff.json",
        "operator-handoff.md",
    ]:
        assert f"`{artifact}`" in text
    assert "voiceops.realtime_voice_live_evidence_manifest.v1" in text
    assert "voiceops.realtime_voice_live_evidence_validation.v1" in text
    assert "`--validate-live-evidence`" in text
    assert "performs no Discord network call" in text
    assert "voiceops.milestone1.live_voice_evidence.v1" in text
    assert "For non-manifest ingestion, pass one `--live-evidence` per section or combined file" in text
    assert "kind` or `evidence_type` values such as `discord_live_probe`, `sidecar_session`, or `live_turn`" in text
    assert "Manifest ingestion is preferred because manifest reports record the actual referenced report path as provenance" in text
    assert "placeholder source paths inside referenced artifacts are not trusted as provenance" in text
    assert "Template source artifact names such as `discord-live-probe.json`, `voice-status-or-sidecar-report.json`, and `voice-turn-evidence.json` are rejected" in text
    assert "source_artifact` for every redacted evidence section" in text
    assert "voiceops.milestone2.post_approval_receipts.v1" in text
    assert "`--post-approval-receipts`" in text
    assert "`--run-readonly-discovery`" in text
    assert "stripe projects list --limit 10" in text
    assert "link-cli auth status" in text
    assert "`source_artifact_kind: redacted_setup_evidence`, `source_artifact_sha256`, and `source_artifact_redacted_at`" in text
    assert "SHA-256 must match the referenced redacted JSON source artifact" in text
    assert "redaction timestamp must be parseable with timezone information" in text
    assert "all_local_stack_smoke" in text
    assert "oracle authority routes include tools/files/memory/project context" in text
    assert "reflex provider includes `vllm`" in text
    assert "`speech_end_to_first_audio_ms <= 1500`" in text
    assert "`barge_in_stop_ms <= 150`" in text
    assert "`local_turn_oracle_calls == 0`" in text
    assert "`oracle_bound_oracle_calls >= oracle_bound_turns`" in text
    assert "local reflex turns must not call the oracle" in text
    assert "scripts/dgx_spark_gemma4_voice_eval.sh" in text
    assert "spark-operator-runbook.md" in text
    assert "The operator handoff is the ordered execution runbook" in text
    assert "does not change readiness by itself" in text
    assert "closure rehearsal" in text.lower()
    assert "`remaining_gates: []`" in text


def test_goal_doc_keeps_super_local_and_ultra_hosted():
    text = GOAL_DOC.read_text(encoding="utf-8")

    assert "Nemotron 3 Super is the preferred Spark-local NVIDIA oracle/model target" in text
    assert "A clearly labeled hosted `/model` fallback is acceptable only when the local Spark path is unavailable" in text
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
            "DISCORD_BOT_TOKEN": "discord-secret-token",
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
    assert "discord-secret-token" not in serialized
    assert summary["closure_index"]["current_environment_blockers"]["discord_env"]["present_env_keys"] == [
        "DISCORD_BOT_TOKEN"
    ]
    assert summary["current_environment"]["provisioning"]["env_presence"]["VOICEOPS_DEMO_PHONE_NUMBER"] is True
    assert "STRIPE_SECRET_KEY" not in summary["current_environment"]["provisioning"]["env_presence"]
    assert provisioning_result["status"] == "needs_setup"
    assert provisioning_result["details"]["preflight_evidence_loaded"] is False
    assert "stripe_projects_account" in provisioning_result["details"]["required_failures"]
    assert provisioning_result["details"]["run_command_probes"] is False


def test_plan_run_env_file_presence_is_redacted_and_reflected_in_handoff(tmp_path):
    artifact_root = tmp_path / "artifacts"
    output_dir = artifact_root / "voiceops-plan" / "current"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DISCORD_BOT_TOKEN=secret-discord-token",
                "DISCORD_GUILD_ID=123",
                "DISCORD_HOME_CHANNEL=general",
                "DISCORD_VOICE_CHANNEL_ID=456",
                "DISCORD_VOICE_CHANNEL_NAME=General",
                "VOICEOPS_DEMO_PHONE_NUMBER=+15551234567",
                "TWILIO_ACCOUNT_SID=AC123456789",
                "TWILIO_AUTH_TOKEN=secret-twilio-token",
            ]
        ),
        encoding="utf-8",
    )

    summary = build_plan_run(artifact_root=artifact_root, output_dir=output_dir, env={}, env_files=[env_file])
    serialized = json.dumps(summary)

    assert summary["current_environment"]["discord"]["env_presence"]["DISCORD_BOT_TOKEN"] is True
    assert summary["current_environment"]["discord"]["live_probe_can_run_here"] is True
    assert summary["closure_index"]["current_environment_blockers"]["discord_env"]["missing_env_keys"] == []
    assert summary["closure_index"]["operator_handoff"]["phases"][0]["can_run_here_now"] is True
    assert summary["current_environment"]["provisioning"]["env_presence"]["VOICEOPS_DEMO_PHONE_NUMBER"] is True
    assert "secret-discord-token" not in serialized
    assert "secret-twilio-token" not in serialized
    assert "+15551234567" not in serialized


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
    assert Path(payload["artifacts"]["operator_handoff_json"]).exists()
    assert Path(payload["artifacts"]["operator_handoff_markdown"]).exists()


def test_parse_args_defaults_to_plan_artifact_paths():
    args = parse_args([])

    assert args.artifact_root == Path("artifacts")
    assert args.output_dir == Path("artifacts/voiceops-plan/current")
    assert args.voice_live_evidence == []
    assert args.provisioning_preflight_evidence is None
    assert args.run_command_probes is False
    assert args.run_readonly_discovery is False
