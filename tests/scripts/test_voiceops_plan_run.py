from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.voiceops_plan_run import build_plan_run, parse_args, write_plan_run
from scripts.voiceops_provisioning_probe import build_milestone2_execution_plan, build_probe_report
from toolsets import _HERMES_CORE_TOOLS


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


def _collector_attestation(section_name: str, *, redacted_sha256: str | None = None) -> dict:
    return {
        "collector_name": "pytest.voiceops_plan_run_fixture",
        "collector_version": "voiceops-plan-run-fixture-v1",
        "run_id": f"pytest-{section_name}",
        "command_argv": ["pytest", section_name],
        "git_commit": "abc123def456",
        "started_at": "2026-06-29T00:00:00Z",
        "finished_at": "2026-06-29T00:00:01Z",
        "raw_artifact_sha256": "a" * 64,
        "redacted_artifact_sha256": redacted_sha256 or ("b" * 64),
        "parent_manifest_sha256": "c" * 64,
    }


def _live_payload_sha256(payload: dict) -> str:
    attested_payload = dict(payload)
    attested_payload.pop("collector_attestation", None)
    attested_payload.pop("collector_provenance", None)
    raw = json.dumps(attested_payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_live_section(path: Path, section_name: str, payload: dict) -> Path:
    payload["collector_attestation"] = _collector_attestation(section_name)
    payload_sha256 = _live_payload_sha256(payload)
    payload["collector_attestation"]["raw_artifact_sha256"] = payload_sha256
    payload["collector_attestation"]["redacted_artifact_sha256"] = payload_sha256
    payload["collector_attestation"]["parent_manifest_sha256"] = payload_sha256
    return _write_json(path, payload)


def _write_live_voice_evidence(root: Path) -> Path:
    _write_live_section(
        root / "discord-live-probe.json",
        "discord_live_probe",
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
            "latency_metrics_ms": {
                "connect_ms": 420,
                "playback_observed_ms": 180,
                "inbound_observed_ms": 900,
                "disconnect_ms": 120,
            },
        },
    )
    _write_live_section(
        root / "sidecar-session.json",
        "sidecar_session",
        {
            "kind": "sidecar_session",
            "sidecar_running": True,
            "sidecar_healthy": True,
            "session_started": True,
            "session_closed": True,
            "shutdown_bounded": True,
            "shutdown_timed_out": False,
            "fallback_mode_visible": True,
            "fallback_reason": "none",
            "sidecar_mode": "production",
            "healthcheck_observed": True,
            "provider_transport_observed": True,
            "session_id_redacted": True,
            "latency_metrics_ms": {"session_start_ms": 110, "shutdown_ms": 80},
        },
    )
    _write_live_section(
        root / "live-turn.json",
        "live_turn",
        {
            "kind": "live_turn",
            "turn_id": "voiceops-live-turn-budget",
            "audio_segment_ref": "artifact://redacted/voiceops-live-turn-budget.wav",
            "evidence_bundle_id": "kame-evidence-live-turn-budget",
            "evidence_merge_key": "kame-merge-live-turn-budget",
            "audio_segment_ref_observed": True,
            "interpreter_evidence_observed": True,
            "transcript_hypotheses_labeled": True,
            "transcript_observed": True,
            "witness_arrival_phases": ["with_raw_audio"],
            "interpreter_input_order": [
                "raw_audio",
                "metadata",
                "reflex",
                "transcript_hypotheses",
            ],
            "interpreter_prompt_policy": {
                "version": "raw_audio_compare_v1",
                "primary_evidence": "raw_audio",
                "transcript_hypotheses_authority": "non_authoritative_context",
            },
            "transcript_hypotheses": [
                {
                    "kind": "frontend_witness_hypothesis",
                    "source": "moshi",
                    "text": "[redacted witness hypothesis]",
                    "arrival_phase": "with_raw_audio",
                    "adjudication": "corrected_by_audio",
                    "authority": "hypothesis",
                    "tool_authority": False,
                }
            ],
            "interpreter_adjudication_outcomes": ["corrected_by_audio"],
            "promoted_evidence_authority": {
                "interpreter_corrected_transcript": "interpreter_promoted",
                "interpreter_normalized_intent": "interpreter_promoted",
            },
            "unpromoted_witness_sink_checks": {
                "spend_clean": True,
                "phone_clean": True,
                "nemoclaw_clean": True,
                "tool_clean": True,
                "memory_clean": True,
                "file_clean": True,
                "message_clean": True,
                "durable_history_clean": True,
            },
            "unpromoted_witness_sink_values": {},
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
            "source_artifact": f"sources/{source_path.name}",
            "source_artifact_kind": "redacted_setup_evidence",
            "source_artifact_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "source_artifact_redacted_at": "2026-06-29T00:00:00Z",
        }
        section["collector_attestation"] = _collector_attestation(
            section_name,
            redacted_sha256=section["source_artifact_sha256"],
        )
        report_path = _write_json(sections_dir / report_names[section_name], {section_name: section})
        reports[section_name] = f"sections/{report_path.name}"
    return _write_json(
        root / "provisioning-preflight-evidence.manifest.json",
        {
            "schema_version": "voiceops.milestone2.preflight_evidence_manifest.v1",
            "reports": reports,
        },
    )


def _write_approval_decision_artifacts(root: Path, payload: dict) -> None:
    for receipt in payload.get("receipts", []):
        if not isinstance(receipt, dict):
            continue
        decision_ref = str(receipt.get("approval_decision_ref") or "")
        if not decision_ref:
            continue
        decision_payload = {
            "schema_version": "voiceops.milestone2.approval_decision.v1",
            "redacted": True,
            "redaction_policy": "redacted references only; no raw secrets, tokens, cards, or phone numbers",
            "action_id": receipt.get("action_id"),
            "receipt_id": receipt.get("receipt_id"),
            "decision": receipt.get("decision"),
            "decision_by": receipt.get("decision_by"),
            "decision_at": receipt.get("decision_at"),
            "approval_id": receipt.get("approval_id"),
        }
        decision_path = root / decision_ref
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(json.dumps(decision_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipt["approval_decision_sha256"] = hashlib.sha256(decision_path.read_bytes()).hexdigest()


def _write_post_approval_receipts(root: Path) -> Path:
    report = build_probe_report(env={}, env_files=[], which=lambda _command: None)
    plan = build_milestone2_execution_plan(report)
    actions = plan["approval_required_actions"]
    estimates = {
        step["step_id"]: step["estimated_cents"]
        for step in plan["execution_steps"]
        if step.get("requires_approval") is True
    }
    payload = {
        "schema_version": "voiceops.milestone2.post_approval_receipts.v1",
        "redaction_policy": "references only",
        "receipts": [
            {
                "receipt_id": f"receipt-{action['action_id']}-001",
                "action_id": action["action_id"],
                "approval_id": action["approval_id"],
                "provider": action["provider"],
                "status": "executed",
                "decision": "approve_once",
                "decision_by": "operator-ref-demo",
                "decision_at": "2026-06-29T00:00:00Z",
                "approval_decision_ref": f"approval-decision-{action['action_id']}.json",
                "approval_decision_sha256": "d" * 64,
                "executed_at": "2026-06-29T00:00:30Z",
                "command_sha256": action["command_sha256"],
                "amount_cents": estimates.get(action["action_id"], 0),
                "currency": "usd",
                "approval_artifact": action["approval_artifact"],
                "external_reference": f"provider-resource-ref-{action['action_id']}",
                "credential_location_ref": action["credential_location_ref"],
                "rollback_ref": action["rollback_ref"],
                "audit_event_id": f"audit-{action['action_id']}-001",
                **dict(action.get("lineage") or {}),
            }
            for action in actions
        ],
        "credential_locations": [
            {
                "credential_ref_id": action["credential_location_ref"],
                "provider": action["provider"],
                "service_id": f"provider-resource-ref-{action['action_id']}",
                "storage_backend": "provider_managed",
                "secret_name_or_path": f"credential-location-ref-{action['action_id']}",
                "created_by_action_id": action["action_id"],
                "rotation_due": "2026-09-29T00:00:00Z",
                "lineage": dict(action.get("lineage") or {}),
            }
            for action in actions
            if action["credential_location_required"]
        ],
        "rollback_receipts": [
            {
                "rollback_ref": action["rollback_ref"],
                "status": "not_run",
                "owner_ref": "operator-ref-demo",
                "notes": "No rollback run.",
                "lineage": dict(action.get("lineage") or {}),
            }
            for action in actions
        ],
        "audit_events": [
            {
                "audit_event_id": f"audit-{action['action_id']}-001",
                "action_id": action["action_id"],
                "receipt_id": f"receipt-{action['action_id']}-001",
                "status": "executed",
                "provider": action["provider"],
                "artifact_ref": "post-approval-receipts.json",
                "operator_next_step": "Review provider dashboard and rollback window.",
                **dict(action.get("lineage") or {}),
            }
            for action in actions
        ],
    }
    _write_approval_decision_artifacts(root, payload)
    attested_payload = dict(payload)
    attested_payload.pop("collector_attestation", None)
    redacted_sha256 = hashlib.sha256(
        json.dumps(attested_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    payload["collector_attestation"] = _collector_attestation(
        "post_approval_receipts",
        redacted_sha256=redacted_sha256,
    )
    return _write_json(root / "post-approval-receipts.json", payload)


def _write_readonly_discovery_evidence(root: Path) -> Path:
    commands = [["link-cli", "auth", "status"], ["stripe", "projects", "list", "--limit", "10"]]
    payload = {
        "schema_version": "voiceops.milestone2.read_only_discovery.v1",
        "generated_at": "2026-06-29T00:00:00Z",
        "run_requested": True,
        "non_mutating": True,
        "does_not_grant_approval": True,
        "redacted_outputs_only": True,
        "required_for_live_provisioning_approval": True,
        "auth_context": "isolated_home",
        "proves_existing_local_auth": False,
        "network_io_possible": True,
        "status": "pass",
        "failed_probe_ids": [],
        "missing_probe_ids": [],
        "allowlisted_commands": commands,
        "blocked_capabilities": [
            "live_spend",
            "provider_provisioning",
            "credential_retrieval",
            "outbound_phone_calls",
            "account_mutation",
            "network_tunnels",
        ],
        "probes": [
            {
                "probe_id": "stripe_projects_catalog_list",
                "area": "stripe_projects",
                "argv": ["stripe", "projects", "list", "--limit", "10"],
                "status": "pass",
                "executed": True,
                "found": True,
                "purpose": "display-only Projects catalog visibility",
            },
            {
                "probe_id": "stripe_link_auth_status",
                "area": "stripe_link",
                "argv": ["link-cli", "auth", "status"],
                "status": "pass",
                "executed": True,
                "found": True,
                "purpose": "display-only Link auth status",
            },
        ],
    }
    attested_payload = dict(payload)
    redacted_sha256 = hashlib.sha256(
        json.dumps(attested_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    payload["collector_attestation"] = _collector_attestation(
        "read_only_discovery",
        redacted_sha256=redacted_sha256,
    )
    report_path = _write_json(root / "read-only-discovery.json", payload)
    return _write_json(
        root / "read-only-discovery.manifest.json",
        {
            "schema_version": "voiceops.milestone2.read_only_discovery_manifest.v1",
            "generated_at": "2026-06-29T00:00:00Z",
            "report": report_path.name,
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "markdown": "read-only-discovery.md",
            "audit_ledger": "audit-ledger.read-only-discovery.jsonl",
            "run_requested": True,
            "status": "pass",
            "failed_probe_ids": [],
            "missing_probe_ids": [],
            "does_not_grant_approval": True,
            "redacted_outputs_only": True,
            "probes": [
                {"probe_id": "stripe_projects_catalog_list", "command": commands[1], "status": "pass", "executed": True},
                {"probe_id": "stripe_link_auth_status", "command": commands[0], "status": "pass", "executed": True},
            ],
        },
    )


def _base_spark_evidence(
    candidate_id: str,
    *,
    model: str,
    source_artifact: str,
    source_artifact_sha256: str,
) -> dict:
    evidence = {
        "schema_version": "voiceops.spark_benchmark_evidence.v1",
        "candidate_id": candidate_id,
        "hardware": "1x NVIDIA DGX Spark",
        "locality": "local_spark",
        "model": model,
        "engine": "local rehearsal engine",
        "verified": True,
        "measured_at": "2026-06-29T00:00:00Z",
        "source_artifact": source_artifact,
        "source_artifact_sha256": source_artifact_sha256,
        "collector_attestation": _collector_attestation(candidate_id, redacted_sha256=source_artifact_sha256),
        "metrics": {},
    }
    if candidate_id == "oracle-nemotron3-super-local":
        evidence["oracle_selected_by"] = "Hermes /model"
    return evidence


def _write_spark_evidence(root: Path) -> Path:
    sources = root / "sources"
    source_sha256: dict[str, str] = {}
    source_keys = {
        "reflex": "reflex-moshi-s2s",
        "interpreter": "interpreter-gemma4-e2b",
        "oracle": "oracle-nemotron3-super-local",
        "asr": "asr-nemotron-speech",
        "tts": "tts-magpie-local",
        "stack-smoke": "voiceops_spark_stack_smoke",
    }
    for name, source_key in source_keys.items():
        payload = {
            "redacted": True,
            "source": name,
            "source_key": source_key,
            "summary": "local DGX Spark rehearsal source",
        }
        if name == "stack-smoke":
            payload["kame_turns"] = [
                {
                    "turn_id": "local-001",
                    "route": "local",
                    "oracle_called": False,
                    "audio_segment_ref": "artifact://redacted/local-001.wav",
                    "audio_time_range_ms": [100, 900],
                    "reflex_transcript_hypothesis": {
                        "authority": "hypothesis",
                        "source": "moshi",
                        "text": "[redacted local hypothesis]",
                    },
                    "auxiliary_transcript_hypotheses": [],
                },
                {
                    "turn_id": "oracle-001",
                    "route": "defer",
                    "oracle_called": True,
                    "oracle_calls": 1,
                    "audio_segment_ref": "artifact://redacted/oracle-001.wav",
                    "audio_time_range_ms": [1200, 3300],
                    "reflex_transcript_hypothesis": {
                        "authority": "hypothesis",
                        "source": "moshi",
                        "text": "[redacted reflex hypothesis]",
                    },
                    "auxiliary_transcript_hypotheses": [
                        {
                            "authority": "hypothesis",
                            "source": "classic_asr_fallback_optional",
                            "text": "[redacted auxiliary hypothesis]",
                        }
                    ],
                    "interpreter_evidence": {
                        "source": "gemma_interpreter",
                        "corrected_transcript": "[redacted interpreter correction]",
                        "confidence": 0.91,
                    },
                    "interpreter_corrected_transcript": "[redacted interpreter correction]",
                    "tool_critical_text_source": "gemma_interpreter",
                },
            ]
        source_path = _write_json(
            sources / f"{name}.json",
            payload,
        )
        source_sha256[name] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return _write_json(
        root / "spark-benchmark-evidence.json",
        {
            "evidence": [
                {
                    **_base_spark_evidence(
                        "reflex-moshi-s2s",
                        model="Moshi/PersonaPlex-class low-latency S2S",
                        source_artifact="sources/reflex.json",
                        source_artifact_sha256=source_sha256["reflex"],
                    ),
                    "metrics": {"ack_latency_ms": 250, "barge_in_stop_ms": 90, "steady_state_memory_gb": 16},
                },
                {
                    **_base_spark_evidence(
                        "interpreter-gemma4-e2b",
                        model="Gemma 4 E2B audio-native interpreter",
                        source_artifact="sources/interpreter.json",
                        source_artifact_sha256=source_sha256["interpreter"],
                    ),
                    "metrics": {"audio_interpretation_ms": 900, "evidence_patch_ms": 1200, "steady_state_memory_gb": 24},
                },
                {
                    **_base_spark_evidence(
                        "oracle-nemotron3-super-local",
                        model="Nemotron 3 Super",
                        source_artifact="sources/oracle.json",
                        source_artifact_sha256=source_sha256["oracle"],
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
                        source_artifact_sha256=source_sha256["asr"],
                    ),
                    "metrics": {"asr_delta_ms": 30, "final_transcript_ms": 600, "word_error_rate": 0.08},
                },
                {
                    **_base_spark_evidence(
                        "tts-magpie-local",
                        model="Magpie local TTS",
                        source_artifact="sources/tts.json",
                        source_artifact_sha256=source_sha256["tts"],
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
                    "source_artifact_sha256": source_sha256["stack-smoke"],
                    "collector_attestation": _collector_attestation(
                        "stack-smoke",
                        redacted_sha256=source_sha256["stack-smoke"],
                    ),
                    "oracle_selected_by": "Hermes /model",
                    "oracle_authority_routes": ["tools", "files", "memory", "project_context"],
                    "interface_input_sources": ["native_audio"],
                    "reflex_providers": ["moshi"],
                    "interpreter_providers": ["vllm", "gemma"],
                    "auxiliary_transcript_sources": ["moshi_hypothesis", "classic_asr_fallback_optional"],
                    "components": {"reflex": True, "interpreter": True, "oracle": True, "tts": True, "sidecar": True},
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
    assert summary["artifact_id"] == "voiceops-plan-run"
    assert summary["artifact_only"] is True
    assert summary["ok"] is True
    assert summary["closure_index"]["schema_version"] == "voiceops.closure_index.v1"
    assert summary["closure_index"]["closure_status"] == "needs_external_evidence"
    assert summary["closure_status"] == summary["closure_index"]["closure_status"]
    assert summary["readiness_ok"] is False
    assert summary["closure_index"]["source_plan_run_artifact"].endswith("voiceops-plan-run.json")
    assert summary["closure_index"]["remaining_gates"] == summary["closure_index"]["gates"]
    assert summary["ready_for_demo"] is False
    assert summary["remaining_gates"] == [
        "live_discord_voice_operator",
        "spend_and_provisioning_preflight",
        "local_spark_stack_matrix",
    ]
    assert summary["current_environment"]["schema_version"] == "voiceops.current_environment.v1"
    assert summary["current_environment"]["redaction_policy"].startswith("presence booleans only")
    assert summary["current_environment"]["discord"]["env_presence"]["DISCORD_BOT_TOKEN"] is False
    assert summary["current_environment"]["discord"]["live_probe_can_run_here"] is False
    assert summary["current_environment"]["provisioning"]["required_cli_presence"] == {
        "stripe": False,
        "link-cli": False,
        "mppx": False,
        "mppx_or_fallback": False,
    }
    assert summary["current_environment"]["spark"]["hardware_claim"] == "not_verified_by_plan_run"
    assert "dgx_spark_likely" in summary["current_environment"]["spark"]
    assert summary["closure_index"]["current_environment"] == summary["current_environment"]
    blockers = summary["closure_index"]["current_environment_blockers"]
    assert summary["current_environment_blockers"] == blockers
    assert blockers["hard_failure"] is False
    assert blockers["secret_values_emitted"] is False
    assert blockers["diagnostic_only"] is True
    assert "DISCORD_BOT_TOKEN" in blockers["discord_env"]["missing_env_keys"]
    assert "stripe" in blockers["provisioning_cli"]["missing"]
    assert "link-cli" in blockers["provisioning_cli"]["missing"]
    assert "mppx_or_fallback" in blockers["provisioning_cli"]["missing"]
    assert blockers["spark_host"]["required_hardware"] == "1x NVIDIA DGX Spark"
    assert blockers["spark_host"]["blocks_artifact_generation"] is False
    assert summary["blockers"] == {
        "readiness_gaps": summary["readiness_gaps"],
        "review_gaps": summary["review_gaps"],
        "remaining_gates": summary["remaining_gates"],
        "review_actions": summary["review_actions"],
        "current_environment": blockers,
    }
    handoff = summary["closure_index"]["operator_handoff"]
    assert handoff["schema_version"] == "voiceops.operator_handoff.v1"
    assert handoff["changes_readiness_by_itself"] is False
    assert handoff["final_success_signal"] == (
        "readiness_gaps is [] and review_gaps is [] and closure_status is complete and package_audit.status is pass"
    )
    assert "--package-audit" in handoff["final_reindex_command"]
    next_actions = summary["closure_index"]["next_actions"]
    assert summary["next_actions"] == next_actions
    review_actions = summary["closure_index"]["review_actions"]
    assert summary["review_actions"] == review_actions
    assert [action["phase_id"] for action in review_actions] == ["multi_channel_policy_review"]
    assert review_actions[0]["milestone"] == "milestone_3_multi_channel_policy"
    assert review_actions[0]["status"] == "pending_human_review"
    assert review_actions[0]["changes_readiness_by_itself"] is False
    assert review_actions[0]["changes_policy_by_itself"] is False
    assert review_actions[0]["real_egress_enabled"] is False
    assert "voiceops_channel_policy.py" in review_actions[0]["review_command"]
    assert any("channel-policy-review.json" in artifact for artifact in review_actions[0]["review_artifacts"])
    assert [action["gate_id"] for action in next_actions] == [
        "live_discord_voice_operator",
        "spend_and_provisioning_preflight",
        "local_spark_stack_matrix",
    ]
    assert [action["phase_id"] for action in next_actions] == [
        "live_discord_voice",
        "spend_and_provisioning_preflight",
        "local_spark_stack",
    ]
    assert next_actions[0]["order"] == 1
    assert next_actions[0]["can_run_here_now"] is False
    assert next_actions[0]["blocked_by_current_environment"]["needs_external_live_probe"] is True
    assert "DISCORD_BOT_TOKEN" in next_actions[0]["blocked_by_current_environment"]["missing_env_keys"]
    assert "--audit-only" in next_actions[0]["first_safe_command"]
    assert "hermes_cli.realtime_voice_live_evidence" in next_actions[0]["first_safe_command"]
    assert "hermes_cli.realtime_voice_live_evidence" in next_actions[0]["first_evidence_command"]
    assert "--run-doctor-report" in next_actions[0]["first_evidence_command"]
    assert "--require-inbound" in next_actions[0]["first_evidence_command"]
    assert next_actions[0]["closure_plan"].endswith("voiceops-voice-operator/current/live-probe-closure-plan.json")
    assert next_actions[0]["closure_artifact"].endswith("voiceops-voice-operator/current/live-probe-closure-plan.md")
    assert next_actions[0]["evidence_template"].endswith(
        "voiceops-voice-operator/current/live-voice-evidence-template.json"
    )
    assert next_actions[0]["evidence_scaffold"].endswith(
        "voiceops-voice-operator/current/live-voice-evidence-scaffold/manifest.json"
    )
    assert "--audit-only" in next_actions[0]["local_audit_command"]
    assert "--validate-live-evidence" in next_actions[0]["local_validation_command"]
    assert sorted(next_actions[0]["validation_commands"]) == [
        "audit_live_manifest_no_write",
        "validate_live_manifest_offline",
    ]
    assert "manifest.json" in next_actions[0]["rerun_command"]
    assert "artifacts/realtime-voice-evidence/live-current/live-turn.json" in next_actions[0]["expected_artifacts"]
    assert next_actions[1]["blocked_by_current_environment"]["missing_cli"] == [
        "stripe",
        "link-cli",
        "mppx_or_fallback",
    ]
    assert next_actions[1]["blocked_by_current_environment"]["needs_read_only_discovery"] is True
    assert "--dry-audit" in next_actions[1]["first_safe_command"]
    assert "--package-audit" in next_actions[1]["first_safe_command"]
    assert "voiceops_provisioning_probe.py" in next_actions[1]["first_evidence_command"]
    assert "--run-readonly-discovery" not in next_actions[1]["first_evidence_command"]
    assert next_actions[1]["closure_plan"].endswith("voiceops-provisioning/current/setup-closure-plan.json")
    assert next_actions[1]["evidence_scaffold"].endswith(
        "voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json"
    )
    assert "--dry-audit" in next_actions[1]["local_audit_command"]
    assert "--preflight-evidence" in next_actions[1]["local_validation_command"]
    assert next_actions[1]["evidence_manifest_example"].endswith(
        "voiceops-provisioning/current/provisioning-preflight-evidence.manifest.example.json"
    )
    assert "execute_approved_stripe_actions" not in next_actions[1]["validation_commands"]
    assert "validate_nemoclaw_action_packet" in next_actions[1]["validation_commands"]
    assert "validate_post_approval_receipts" in next_actions[1]["validation_commands"]
    assert any(
        artifact.endswith("post-approval-receipts.json")
        for artifact in next_actions[1]["expected_artifacts"]
    )
    assert next_actions[2]["blocked_by_current_environment"]["required_hardware"] == "1x NVIDIA DGX Spark"
    assert next_actions[2]["blocked_by_current_environment"]["needs_measured_spark_evidence"] is True
    assert "--lint-evidence" in next_actions[2]["first_safe_command"]
    assert next_actions[2]["first_evidence_command"] == "scripts/dgx_spark_gemma4_voice_eval.sh"
    assert next_actions[2]["closure_plan"].endswith("voiceops-spark-matrix/current/spark-matrix-closure-plan.json")
    assert next_actions[2]["evidence_scaffold"].endswith(
        "voiceops-spark-matrix/current/spark-benchmark-scaffold/spark-benchmark-evidence.json"
    )
    assert "--lint-evidence" in next_actions[2]["local_audit_command"]
    assert "--evidence" in next_actions[2]["local_validation_command"]
    assert next_actions[2]["operator_runbook"].endswith("voiceops-spark-matrix/current/spark-operator-runbook.md")
    assert sorted(next_actions[2]["validation_commands"]) == [
        "lint_evidence",
        "matrix_only",
        "refresh_source_hashes",
        "with_evidence",
    ]
    assert any(
        artifact.endswith("spark-operator-runbook.md")
        for artifact in next_actions[2]["expected_artifacts"]
    )
    assert not any(
        artifact.endswith("asr-nemotron-speech-raw.json")
        for artifact in next_actions[2]["expected_artifacts"]
    )
    assert any(
        artifact.endswith("asr-nemotron-speech-raw.json")
        for artifact in next_actions[2]["optional_artifacts"]
    )
    assert all("never include secret values" in action["secret_policy"] for action in next_actions)
    assert [phase["phase_id"] for phase in handoff["phases"]] == [
        "live_discord_voice",
        "spend_and_provisioning_preflight",
        "local_spark_stack",
    ]
    assert [phase["phase_id"] for phase in handoff["review_phases"]] == ["multi_channel_policy_review"]
    assert handoff["review_phases"][0]["status"] == "pending_human_review"
    assert handoff["review_phases"][0]["changes_readiness_by_itself"] is False
    assert handoff["review_phases"][0]["changes_policy_by_itself"] is False
    assert handoff["review_phases"][0]["real_egress_enabled"] is False
    assert any("channel-policy-review.json" in artifact for artifact in handoff["review_phases"][0]["review_artifacts"])
    assert "voiceops_channel_policy.py" in handoff["review_phases"][0]["first_safe_command"]
    assert [phase["order"] for phase in handoff["phases"]] == [1, 2, 3]
    assert [phase["status"] for phase in handoff["phases"]] == [
        "needs_live_probe",
        "needs_setup",
        "needs_evidence",
    ]
    assert (
        handoff["phases"][2]["command_safety"]["matrix_only"]
        == "local_matrix_generation_no_supplied_evidence"
    )
    assert handoff["phases"][0]["can_run_here_now"] is False
    handoff_phases_by_gate = {phase["gate_id"]: phase for phase in handoff["phases"]}
    for action in next_actions:
        phase = handoff_phases_by_gate[action["gate_id"]]
        for command_key in action["validation_commands"]:
            assert command_key in phase["command_safety"]
    assert handoff["phases"][0]["first_safe_command"] == next_actions[0]["first_safe_command"]
    assert handoff["phases"][0]["first_evidence_command"] == next_actions[0]["first_evidence_command"]
    assert "--audit-only" in handoff["phases"][0]["commands"][0]
    assert handoff["phases"][0]["commands"][1] == next_actions[0]["first_evidence_command"]
    assert "--run-doctor-report" in handoff["phases"][0]["commands"][1]
    assert "hermes_cli.realtime_voice_live_evidence" in handoff["phases"][0]["commands"][1]
    assert handoff["phases"][0]["blocked_by_current_environment"] == {
        "missing_env_keys": next_actions[0]["blocked_by_current_environment"]["missing_env_keys"],
        "present_env_keys": blockers["discord_env"]["present_env_keys"],
        "needs_external_live_probe": True,
    }
    assert "discord-live-probe.json with source_artifact, collector_attestation" in json.dumps(
        handoff["phases"][0]["required_inputs"]
    )
    assert "sidecar_mode=production" in json.dumps(handoff["phases"][0]["required_inputs"])
    assert "healthcheck_observed" in json.dumps(handoff["phases"][0]["required_inputs"])
    assert "provider_transport_observed" in json.dumps(handoff["phases"][0]["required_inputs"])
    assert "session_id_redacted" in json.dumps(handoff["phases"][0]["required_inputs"])
    assert "fallback_reason" in json.dumps(handoff["phases"][0]["required_inputs"])
    assert "sidecar-session.json" in json.dumps(handoff["phases"][0]["expected_artifacts"])
    assert "live-evidence-validation.json" in json.dumps(handoff["phases"][0]["expected_artifacts"])
    assert "hermes doctor" in handoff["phases"][0]["commands"][2]
    assert "--realtime-voice-report artifacts/realtime-voice-evidence/live-current/realtime-voice-doctor-report.json" in handoff["phases"][0]["commands"][2]
    assert "python -m hermes_cli.realtime_voice_live_evidence" in handoff["phases"][0]["commands"][3]
    assert "--from-realtime-voice-report artifacts/realtime-voice-evidence/live-current/realtime-voice-doctor-report.json" in handoff["phases"][0]["commands"][3]
    assert "path/to/realtime-voice-report.json" not in json.dumps(handoff["phases"][0]["commands"])
    assert "--require-live-discord" in handoff["phases"][0]["commands"][4]
    assert "--validate-live-evidence" in handoff["phases"][0]["commands"][5]
    assert "--live-evidence-manifest artifacts/realtime-voice-evidence/live-current/manifest.json" in handoff[
        "phases"
    ][0]["commands"][5]
    assert "--validate-live-evidence" in json.dumps(handoff["phases"][0]["commands"])
    assert "provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json" in json.dumps(
        handoff["phases"][1]
    )
    assert "--refresh-preflight-source-hashes" in json.dumps(handoff["phases"][1]["commands"])
    assert "--run-command-probes" in json.dumps(handoff["phases"][1]["commands"])
    assert "--dry-audit" in handoff["phases"][1]["commands"][0]
    assert "--package-audit" in handoff["phases"][1]["commands"][0]
    assert handoff["phases"][1]["first_safe_command"] == next_actions[1]["first_safe_command"]
    assert handoff["phases"][1]["first_evidence_command"] == next_actions[1]["first_evidence_command"]
    assert handoff["phases"][1]["blocked_by_current_environment"] == {
        "missing_cli": next_actions[1]["blocked_by_current_environment"]["missing_cli"],
        "present_cli": blockers["provisioning_cli"]["present"],
        "needs_read_only_discovery": True,
        "needs_redacted_setup_evidence": True,
    }
    assert handoff["phases"][1]["command_safety"]["plan_index_dry_audit"] == "no_write_no_network_no_probe_audit"
    assert handoff["phases"][1]["command_safety"]["read_only_discovery"] == "network_possible_allowlisted_read_only"
    assert handoff["phases"][1]["command_safety"]["validate_nemoclaw_action_packet"] == (
        "local_static_action_packet_validation_only"
    )
    assert handoff["phases"][1]["command_safety"]["validate_post_approval_receipts"] == "post_approval_local_validation_only"
    assert "scripts/voiceops_plan_run.py" in json.dumps(handoff["phases"][1]["commands"])
    assert "--nemoclaw-action-packet" in json.dumps(handoff["phases"][1]["commands"])
    assert "--run-readonly-discovery" in json.dumps(handoff["phases"][1]["commands"])
    assert "--post-approval-receipts" in json.dumps(handoff["phases"][1]["commands"])
    assert "post-approval-receipts.template.json" in json.dumps(handoff["phases"][1]["expected_artifacts"])
    assert "nemoclaw-action-packet.validation.json" in json.dumps(handoff["phases"][1]["expected_artifacts"])
    assert "post-approval-receipts-scaffold/post-approval-receipts.json" in json.dumps(
        handoff["phases"][1]["expected_artifacts"]
    )
    assert handoff["phases"][2]["first_safe_command"] == next_actions[2]["first_safe_command"]
    assert handoff["phases"][2]["first_evidence_command"] == next_actions[2]["first_evidence_command"]
    assert handoff["phases"][2]["blocked_by_current_environment"] == {
        "required_hardware": next_actions[2]["blocked_by_current_environment"]["required_hardware"],
        "current_host_hint": next_actions[2]["blocked_by_current_environment"]["current_host_hint"],
        "needs_measured_spark_evidence": True,
    }
    assert "--lint-evidence" in handoff["phases"][2]["commands"][0]
    assert handoff["phases"][2]["commands"][1] == "scripts/dgx_spark_gemma4_voice_eval.sh"
    assert "--refresh-source-hashes" in handoff["phases"][2]["commands"][2]
    assert "KAME/reflex/interpreter/oracle/TTS evidence" in next_actions[2]["operator_step"]
    assert "ASR is optional witness/fallback transcript-hypothesis evidence" in next_actions[2]["operator_step"]
    assert "asr-nemotron-speech-raw.json" not in json.dumps(handoff["phases"][2]["expected_artifacts"])
    assert "asr-nemotron-speech-raw.json" in json.dumps(handoff["phases"][2]["optional_artifacts"])
    assert "tts-magpie-local-raw.json" in json.dumps(handoff["phases"][2]["expected_artifacts"])
    assert "all-local-stack-smoke-raw.json" in json.dumps(handoff["phases"][2]["expected_artifacts"])
    assert "loopback_smoke_bridge protocol smoke checks" in json.dumps(handoff["phases"][2]["must_not"])
    assert (
        "artifacts/voiceops-spark-matrix/current/spark-benchmark-scaffold/spark-benchmark-evidence.json"
        in handoff["final_reindex_command"]
    )
    assert "path/to/spark-benchmark-evidence.json" not in handoff["final_reindex_command"]
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
    assert "--dry-audit" in gates["spend_and_provisioning_preflight"]["rerun_commands"]["plan_index_dry_audit"]
    assert "plan_index_command_probes" in gates["spend_and_provisioning_preflight"]["rerun_commands"]
    assert "--run-command-probes" in gates["spend_and_provisioning_preflight"]["rerun_commands"][
        "plan_index_command_probes"
    ]
    assert "schema_version" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "turn_id" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "audio_segment_ref" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "evidence_bundle_id" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "evidence_merge_key" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "transcript_observed" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "audio_segment_ref_observed" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "interpreter_evidence_observed" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "transcript_hypotheses_labeled" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert (
        "raw_audio_interpreter_evidence_observed"
        in gates["live_discord_voice_operator"]["required_evidence_fields"]
    )
    assert (
        "transcript_only_witness_rejected_for_full_kame"
        in gates["live_discord_voice_operator"]["required_evidence_fields"]
    )
    assert "sidecar_mode" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "healthcheck_observed" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "provider_transport_observed" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "session_id_redacted" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "fallback_reason" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "latency_metrics_ms.connect_ms" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "latency_metrics_ms.session_start_ms" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "source_artifact" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert "collector_attestation" in gates["live_discord_voice_operator"]["required_evidence_fields"]
    assert gates["live_discord_voice_operator"]["evidence_contract"] == {
        "manifest_schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
        "strict_validation_schema_version": "voiceops.realtime_voice_live_evidence_validation.v1",
        "expanded_evidence_schema_version": "voiceops.milestone1.live_voice_evidence.v1",
        "required_sections": ["discord_live_probe", "sidecar_session", "live_turn"],
        "required_section_refs": ["source_artifact", "section"],
        "required_collector_attestation_fields": [
            "collector_name",
            "collector_version",
            "run_id",
            "command_argv",
            "git_commit",
            "started_at",
            "finished_at",
            "raw_artifact_sha256",
            "redacted_artifact_sha256",
            "parent_manifest_sha256",
        ],
        "required_discord_latency_metrics_ms": [
            "connect_ms",
            "playback_observed_ms",
            "inbound_observed_ms",
            "disconnect_ms",
        ],
        "required_sidecar_fields": [
            "sidecar_running",
            "sidecar_healthy",
            "session_started",
            "session_closed",
            "fallback_mode_visible",
            "fallback_reason",
            "sidecar_mode",
            "healthcheck_observed",
            "provider_transport_observed",
            "session_id_redacted",
            "shutdown_bounded",
            "shutdown_timed_out",
        ],
        "required_sidecar_mode": "production",
        "required_sidecar_latency_metrics_ms": ["session_start_ms", "shutdown_ms"],
        "template_source_artifacts_accepted": False,
        "unverified_source_artifacts_accepted": False,
        "source_artifacts_must_exist": True,
        "example_only_accepted": False,
        "collector_attestation_required_for_live_readiness": True,
        "collector_attestation_example_only_accepted": False,
        "realtime_voice_report_derivation_schema_version": "voiceops.realtime_voice_report_derivation.v1",
        "doctor_report_derivation_overclaims_production": False,
    }
    assert "operator_must_not" in gates["live_discord_voice_operator"]
    assert "manifest.json" in gates["live_discord_voice_operator"]["rerun_command"]
    assert "hermes doctor" in gates["live_discord_voice_operator"]["collection_commands"]["run_realtime_voice_doctor_report"]
    assert "realtime-voice-doctor-report.json" in gates["live_discord_voice_operator"]["collection_commands"]["run_realtime_voice_doctor_report"]
    assert "python -m hermes_cli.realtime_voice_live_evidence" in gates["live_discord_voice_operator"]["collection_commands"]["collect_live_manifest"]
    assert "--audit-only" in gates["live_discord_voice_operator"]["collection_commands"]["audit_live_manifest_no_write"]
    assert "--validate-live-evidence" in gates["live_discord_voice_operator"]["collection_commands"][
        "validate_live_manifest_offline"
    ]
    assert "--live-evidence-manifest artifacts/realtime-voice-evidence/live-current/manifest.json" in gates[
        "live_discord_voice_operator"
    ]["collection_commands"][
        "validate_live_manifest_offline"
    ]
    assert "--live-evidence-manifest artifacts/realtime-voice-evidence/live-current/manifest.json" in gates[
        "live_discord_voice_operator"
    ]["collection_commands"]["audit_live_manifest_no_write"]
    assert "--sidecar-session-evidence" in gates["live_discord_voice_operator"]["collection_commands"]["collect_live_manifest"]
    assert "--live-turn-evidence" in gates["live_discord_voice_operator"]["collection_commands"]["collect_live_manifest"]
    assert "--from-realtime-voice-report" in gates["live_discord_voice_operator"]["collection_commands"][
        "derive_from_realtime_voice_report"
    ]
    assert "realtime-voice-doctor-report.json" in gates["live_discord_voice_operator"]["collection_commands"][
        "derive_from_realtime_voice_report"
    ]
    assert "path/to/realtime-voice-report.json" not in json.dumps(gates["live_discord_voice_operator"]["collection_commands"])
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
        "collector_attestation",
    ]
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"]["required_collector_attestation_fields"] == [
        "collector_name",
        "collector_version",
        "run_id",
        "command_argv",
        "git_commit",
        "started_at",
        "finished_at",
        "raw_artifact_sha256",
        "redacted_artifact_sha256",
        "parent_manifest_sha256",
    ]
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"]["source_artifacts_must_exist"] is True
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"]["source_artifact_sha256_must_match"] is True
    assert "process cwd fallback are rejected" in gates["spend_and_provisioning_preflight"]["evidence_contract"][
        "manifest_report_resolution"
    ]
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"]["example_only_accepted"] is False
    assert (
        gates["spend_and_provisioning_preflight"]["evidence_contract"][
            "read_only_discovery_required_for_live_provisioning_approval"
        ]
        is True
    )
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"]["read_only_discovery_required_status"] == "pass"
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"]["read_only_discovery_auth_context"] == "isolated_home"
    assert (
        gates["spend_and_provisioning_preflight"]["evidence_contract"][
            "read_only_discovery_proves_existing_local_auth"
        ]
        is False
    )
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"][
        "nemoclaw_action_packet_validation_schema_version"
    ] == "voiceops.nemoclaw_action_packet_validation.v1"
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"][
        "nemoclaw_action_packet_validation_grants_approval"
    ] is False
    assert gates["spend_and_provisioning_preflight"]["evidence_contract"][
        "nemoclaw_action_packet_validation_executes_commands"
    ] is False
    assert "post_approval_receipts_status is valid" in gates["spend_and_provisioning_preflight"]["completion_signal"]
    assert "read_only_discovery_status is pass" in gates["spend_and_provisioning_preflight"]["completion_signal"]
    assert "audit-ledger.post-approval.jsonl is populated" in gates["spend_and_provisioning_preflight"][
        "completion_signal"
    ]
    assert "provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json" in gates[
        "spend_and_provisioning_preflight"
    ]["collection_commands"]["ingest_preflight_manifest"]
    assert "--nemoclaw-action-packet" in gates["spend_and_provisioning_preflight"]["collection_commands"][
        "validate_nemoclaw_action_packet"
    ]
    assert gates["spend_and_provisioning_preflight"]["current_environment"]["required_cli_presence"]["stripe"] is False
    assert (
        gates["spend_and_provisioning_preflight"]["current_environment"]["required_cli_presence"]["mppx_or_fallback"]
        is False
    )
    assert "required_candidate_fields" in gates["local_spark_stack_matrix"]
    assert "schema_version" in gates["local_spark_stack_matrix"]["required_candidate_fields"]
    assert "source_artifact_sha256" in gates["local_spark_stack_matrix"]["required_candidate_fields"]
    assert "source_artifact_sha256" in gates["local_spark_stack_matrix"]["required_stack_smoke_fields"]
    assert gates["local_spark_stack_matrix"]["gate_ids"] == [
        "reflex",
        "interpreter",
        "oracle",
        "tts",
        "all_local_stack_smoke",
    ]
    assert "asr" not in gates["local_spark_stack_matrix"]["gate_ids"]
    assert (
        gates["local_spark_stack_matrix"]["evidence_contract"]["preferred_local_oracle_candidate_id"]
        == "oracle-nemotron3-super-local"
    )
    assert gates["local_spark_stack_matrix"]["evidence_contract"]["required_oracle_authority_routes"] == [
        "tools",
        "files",
        "memory",
        "project_context",
    ]
    assert gates["local_spark_stack_matrix"]["evidence_contract"]["preferred_local_oracle_model"] == "Nemotron 3 Super"
    assert gates["local_spark_stack_matrix"]["evidence_contract"]["non_counting_fallback_oracle_models"] == [
        "Nemotron 3 Ultra"
    ]
    assert gates["local_spark_stack_matrix"]["evidence_contract"]["hosted_fallback_counts_for_one_spark_readiness"] is False
    assert (
        gates["local_spark_stack_matrix"]["evidence_contract"][
            "loopback_smoke_bridge_counts_for_local_speech_readiness"
        ]
        is False
    )
    assert gates["local_spark_stack_matrix"]["evidence_contract"]["local_speech_requires_production_provider"] is True
    assert gates["local_spark_stack_matrix"]["evidence_contract"]["source_artifacts_must_exist"] is True
    assert gates["local_spark_stack_matrix"]["evidence_contract"]["source_artifact_readable"] is True
    assert gates["local_spark_stack_matrix"]["evidence_contract"]["source_artifact_sha256_must_match"] is True
    assert gates["local_spark_stack_matrix"]["evidence_contract"]["source_artifact_resolution"].endswith(
        "supplied benchmark evidence file"
    )
    assert "all_local_stack_smoke:needs_evidence" in gates["local_spark_stack_matrix"]["missing"]
    assert "all_local_stack_smoke is validated" in gates["local_spark_stack_matrix"]["completion_signal"]
    assert gates["local_spark_stack_matrix"]["collection_commands"]["dgx_eval"] == "scripts/dgx_spark_gemma4_voice_eval.sh"
    assert "--refresh-source-hashes" in gates["local_spark_stack_matrix"]["collection_commands"]["refresh_source_hashes"]
    assert "--lint-evidence" in gates["local_spark_stack_matrix"]["collection_commands"]["lint_evidence"]
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
    assert Path(voice_result["artifacts"]["async_oracle_smoke_json"]).exists()
    assert Path(voice_result["artifacts"]["discord_session_cleanup_smoke_json"]).exists()
    assert Path(voice_result["artifacts"]["sidecar_fail_closed_smoke_json"]).exists()
    assert Path(voice_result["artifacts"]["tool_disclosure_smoke_json"]).exists()
    assert Path(voice_result["artifacts"]["ephemeral_tool_router_smoke_json"]).exists()
    assert Path(voice_result["artifacts"]["interpreter_request_packet_json"]).exists()
    assert Path(voice_result["artifacts"]["events_jsonl"]).exists()
    assert Path(voice_result["artifacts"]["live_evidence_example"]).exists()
    assert Path(voice_result["artifacts"]["live_evidence_scaffold_manifest"]).exists()
    assert Path(voice_result["artifacts"]["live_evidence_template"]).exists()
    assert Path(voice_result["artifacts"]["live_probe_closure_json"]).exists()
    tool_details = voice_result["details"]["tool_disclosure"]
    assert tool_details == {
        "schema_source": "registered_core_tool_schemas",
        "representative_schema": False,
        "missing_registered_core_tools": [],
        "config": {
            "defer_core": "all",
            "enabled": "on",
        },
        "broad_core_tools_visible": False,
        "bridge_tool_count": 3,
        "bridge_tool_names": ["tool_call", "tool_describe", "tool_search"],
        "core_tools_hidden_all": True,
        "deferred_count": len(_HERMES_CORE_TOOLS),
        "deferred_tokens": tool_details["deferred_tokens"],
        "hidden_core_tool_count": len(_HERMES_CORE_TOOLS),
        "hidden_core_tool_names": sorted(_HERMES_CORE_TOOLS),
        "input_core_tool_count": len(_HERMES_CORE_TOOLS),
        "input_core_tools": sorted(_HERMES_CORE_TOOLS),
        "input_schema_tokens": tool_details["input_schema_tokens"],
        "ok": True,
        "test_ref_count": 6,
        "token_reduction_estimate": tool_details["token_reduction_estimate"],
        "visible_schema_tokens": tool_details["visible_schema_tokens"],
        "visible_non_bridge_tool_names": [],
        "visible_tool_names": ["tool_call", "tool_describe", "tool_search"],
    }
    assert tool_details["deferred_tokens"] > 0
    assert tool_details["input_schema_tokens"] > tool_details["visible_schema_tokens"]
    assert tool_details["token_reduction_estimate"] > 0
    assert voice_result["details"]["ephemeral_tool_router"] == {
        "ok": True,
        "router_mode": "ephemeral",
        "provider_network": False,
        "model_call": False,
        "router_call_count": 2,
        "selected_voiceops_toolsets": ["voiceops"],
        "selected_no_tools_toolsets": [],
        "router_transcript_persistent": False,
        "router_tool_calls_allowed": False,
        "test_ref_count": 2,
    }
    assert voice_result["details"]["async_oracle_smoke"]["kind"] == "async_oracle_smoke"
    assert voice_result["details"]["async_oracle_smoke"]["scenario"] == "async_kame_oracle_jobs_fake"
    assert voice_result["details"]["async_oracle_smoke"]["late_cancelled_output_attempted"] is True
    assert voice_result["details"]["async_oracle_smoke"]["max_worker_overlap"] == 4
    assert voice_result["details"]["async_oracle_smoke"]["worker_overlap_proved"] is True
    assert voice_result["details"]["async_oracle_smoke"]["worker_overlap_within_capacity"] is True
    assert voice_result["details"]["async_oracle_smoke"]["noncooperative_cancel_overlap_observed"] is False
    assert voice_result["details"]["async_oracle_smoke"]["queued_jobs"] == 1
    assert voice_result["details"]["async_oracle_smoke"]["failed_jobs"] == 2
    assert voice_result["details"]["async_oracle_smoke"]["queued_cancel_smoke_ok"] is True
    assert voice_result["details"]["async_oracle_smoke"]["queued_cancel_observed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["queued_cancelled_before_start"] is True
    assert voice_result["details"]["async_oracle_smoke"]["queued_cancel_not_sent_to_oracle"] is True
    assert voice_result["details"]["async_oracle_smoke"]["queued_cancel_reason"] == "spoken request to cancel oracle job"
    assert voice_result["details"]["async_oracle_smoke"]["queued_cancel_target_job_id"] == "voice-oracle-002"
    assert voice_result["details"]["async_oracle_smoke"]["queued_cancel_running_completed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["approval_capacity_smoke_ok"] is True
    assert voice_result["details"]["async_oracle_smoke"]["approval_capacity_waiting_observed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["approval_capacity_followup_queued"] is True
    assert voice_result["details"]["async_oracle_smoke"]["approval_capacity_active_visible"] is True
    assert voice_result["details"]["async_oracle_smoke"]["approval_capacity_misleading_running_capacity"] is False
    assert "1 active out of 1" in voice_result["details"]["async_oracle_smoke"]["approval_capacity_status_text"]
    assert "0 running out of 1" not in voice_result["details"]["async_oracle_smoke"]["approval_capacity_status_text"]
    assert "1 queued" in voice_result["details"]["async_oracle_smoke"]["approval_capacity_status_text"]
    assert "1 waiting for approval" in voice_result["details"]["async_oracle_smoke"]["approval_capacity_status_text"]
    assert voice_result["details"]["async_oracle_smoke"]["approval_capacity_followup_started_after_approval"] is True
    assert voice_result["details"]["async_oracle_smoke"]["approval_capacity_completed_jobs"] == 1
    assert voice_result["details"]["async_oracle_smoke"]["approval_capacity_failed_gate_suppressed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["approval_capacity_failed_jobs"] == 1
    assert voice_result["details"]["async_oracle_smoke"]["approval_capacity_max_concurrent"] == 1
    assert voice_result["details"]["async_oracle_smoke"]["cancel_drain_capacity_smoke_ok"] is True
    assert voice_result["details"]["async_oracle_smoke"]["cancel_drain_requested_observed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["cancel_drain_cancelled_observed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["cancel_drain_followup_queued"] is True
    assert voice_result["details"]["async_oracle_smoke"]["cancel_drain_active_visible"] is True
    assert voice_result["details"]["async_oracle_smoke"]["cancel_drain_misleading_running_capacity"] is False
    assert "1 active out of 1" in voice_result["details"]["async_oracle_smoke"]["cancel_drain_status_text"]
    assert "0 running out of 1" not in voice_result["details"]["async_oracle_smoke"]["cancel_drain_status_text"]
    assert "1 queued" in voice_result["details"]["async_oracle_smoke"]["cancel_drain_status_text"]
    assert "1 cancelling" in voice_result["details"]["async_oracle_smoke"]["cancel_drain_status_text"]
    assert voice_result["details"]["async_oracle_smoke"]["cancel_drain_followup_started_after_cancel"] is True
    assert voice_result["details"]["async_oracle_smoke"]["cancel_drain_max_concurrent"] == 1
    assert voice_result["details"]["async_oracle_smoke"]["playback_stop_committed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["playback_stop_jobs_still_running"] is True
    assert voice_result["details"]["async_oracle_smoke"]["playback_stop_cancelled_jobs"] is False
    assert voice_result["details"]["async_oracle_smoke"]["playback_stop_does_not_cancel_jobs"] is True
    assert voice_result["details"]["async_oracle_smoke"]["status_turn_committed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["status_ordinal_labels_visible"] is True
    assert voice_result["details"]["async_oracle_smoke"]["status_ordinal_labels"] == [
        "job one",
        "job two",
        "job three",
        "job four",
        "job five",
    ]
    assert voice_result["details"]["async_oracle_smoke"]["status_bounded_overflow_visible"] is True
    assert voice_result["details"]["async_oracle_smoke"]["status_bounded_overflow_visible_job_count"] == 8
    assert voice_result["details"]["async_oracle_smoke"]["status_bounded_overflow_hidden_job_count"] == 2
    assert (
        voice_result["details"]["async_oracle_smoke"]["status_bounded_overflow_more_spoken_status"]
        == "+2 more"
    )
    assert voice_result["details"]["async_oracle_smoke"]["status_bounded_overflow_last_visible_ordinal"] == 8
    assert (
        voice_result["details"]["async_oracle_smoke"]["status_bounded_overflow_last_visible_label"]
        == "job eight"
    )
    assert voice_result["details"]["async_oracle_smoke"]["status_bounded_overflow_hidden_ids_absent"] is True
    assert voice_result["details"]["async_oracle_smoke"]["terminal_status_committed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["completed_result_status_visible"] is True
    assert "completed: First sentence. Second sentence. Third sentence." in voice_result["details"][
        "async_oracle_smoke"
    ]["terminal_status_text"]
    assert voice_result["details"]["async_oracle_smoke"]["fifth_job_queued"] is True
    assert voice_result["details"]["async_oracle_smoke"]["fifth_job_started_after_capacity_freed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["cancelled_result_spoken"] is False
    assert voice_result["details"]["async_oracle_smoke"]["cancelled_result_committed"] is False
    assert voice_result["details"]["async_oracle_smoke"]["cancelled_result_progress_leaked"] is False
    assert voice_result["details"]["async_oracle_smoke"]["cancelled_result_durable_completed"] is False
    assert voice_result["details"]["async_oracle_smoke"]["cancelled_result_durable_text"] is False
    assert voice_result["details"]["async_oracle_smoke"]["durable_cancelled_record_present"] is True
    assert (
        voice_result["details"]["async_oracle_smoke"]["durable_completed_jobs"]
        == voice_result["details"]["async_oracle_smoke"]["completed_jobs"]
    )
    assert voice_result["details"]["async_oracle_smoke"]["approval_wait_observed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["approval_status_committed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["approval_tool_progress_observed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["approval_tool_progress_kame_gate_present"] is True
    assert (
        voice_result["details"]["async_oracle_smoke"]["approval_tool_progress_kame_gate_schema_version"]
        == "voiceops.runtime_kame_action_gate.v1"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["approval_tool_progress_kame_gate_failed_closed"]
        is True
    )
    assert "missing_promoted_evidence" in voice_result["details"]["async_oracle_smoke"][
        "approval_tool_progress_kame_gate_issues"
    ]
    assert voice_result["details"]["async_oracle_smoke"]["approval_payload_redacted"] is True
    assert voice_result["details"]["async_oracle_smoke"]["approval_completed"] is False
    assert voice_result["details"]["async_oracle_smoke"]["approval_gate_failed_closed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["approval_result_suppressed"] is True
    assert "1 waiting for approval" in voice_result["details"]["async_oracle_smoke"]["approval_status_text"]
    assert "waiting_for_approval: Preparing spend approval." in voice_result["details"]["async_oracle_smoke"][
        "approval_status_text"
    ]
    assert voice_result["details"]["async_oracle_smoke"]["failed_job_reported"] is True
    assert voice_result["details"]["async_oracle_smoke"]["failed_job_spoken"] is True
    assert voice_result["details"]["async_oracle_smoke"]["durable_failed_record_present"] is True
    assert voice_result["details"]["async_oracle_smoke"]["session_survived_failed_job"] is True
    assert voice_result["details"]["async_oracle_smoke"]["queued_job_update_observed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["running_job_update_observed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["running_update_latest_update_visible"] is True
    assert (
        voice_result["details"]["async_oracle_smoke"]["running_update_latest_update_text"]
        == "include running update context"
    )
    assert voice_result["details"]["async_oracle_smoke"]["running_update_reached_oracle"] is True
    assert voice_result["details"]["async_oracle_smoke"]["running_update_delivery_metadata_ok"] is True
    assert voice_result["details"]["async_oracle_smoke"]["queued_update_latest_update_visible"] is True
    assert (
        voice_result["details"]["async_oracle_smoke"]["queued_update_latest_update_text"]
        == "include smoke update context"
    )
    assert voice_result["details"]["async_oracle_smoke"]["queued_update_started_with_priority"] is True
    assert voice_result["details"]["async_oracle_smoke"]["queued_update_reached_oracle"] is True
    assert voice_result["details"]["async_oracle_smoke"]["queued_interpreter_fold_in_observed"] is True
    assert (
        voice_result["details"]["async_oracle_smoke"]["queued_interpreter_fold_in_oracle_text"]
        == "run corrected smoke task five"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["queued_interpreter_fold_in_transcript_source"]
        == "gemma_interpreter"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["queued_interpreter_fold_in_evidence_authority"][
            "oracle_text"
        ]
        == "interpreter_promoted"
    )
    assert voice_result["details"]["async_oracle_smoke"]["verbose_result_spoken_bounded"] is True
    assert voice_result["details"]["async_oracle_smoke"]["verbose_result_committed_bounded"] is True
    assert voice_result["details"]["async_oracle_smoke"]["verbose_result_commit_marked_truncated"] is True
    assert voice_result["details"]["async_oracle_smoke"]["verbose_full_result_durable"] is True
    assert voice_result["details"]["async_oracle_smoke"]["verbose_spoken_result"] == "First sentence."
    assert voice_result["details"]["async_oracle_smoke"]["terminal_result_policy_smoke_ok"] is True
    assert voice_result["details"]["async_oracle_smoke"]["terminal_result_auto_summarize_default"] is True
    assert (
        voice_result["details"]["async_oracle_smoke"]["terminal_result_suppression_config"]
        == "oracle_jobs.speak_terminal_results=false"
    )
    assert voice_result["details"]["async_oracle_smoke"]["terminal_result_suppressed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["terminal_result_unsolicited_event_count"] == 0
    assert voice_result["details"]["async_oracle_smoke"]["terminal_result_unsolicited_spoken"] is False
    assert voice_result["details"]["async_oracle_smoke"]["terminal_result_status_available"] is True
    assert "completed: Finished Suppress terminal result." in voice_result["details"]["async_oracle_smoke"][
        "terminal_result_status_text"
    ]
    assert voice_result["details"]["async_oracle_smoke"]["unflagged_high_risk_tool_smoke_ok"] is True
    assert voice_result["details"]["async_oracle_smoke"]["unflagged_high_risk_tool_suppressed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["unflagged_high_risk_tool_failed_closed"] is True
    assert (
        voice_result["details"]["async_oracle_smoke"]["unflagged_high_risk_tool_suppression_reason"]
        == "unapproved_high_risk_tool_event"
    )
    assert voice_result["details"]["async_oracle_smoke"]["unflagged_high_risk_tool_progress_suppressed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["unflagged_high_risk_tool_payload_redacted"] is True
    assert voice_result["details"]["async_oracle_smoke"]["unflagged_high_risk_tool_spoken_payload_clean"] is True
    assert voice_result["details"]["async_oracle_smoke"]["unflagged_high_risk_tool_failure_spoken"] is True
    assert (
        voice_result["details"]["async_oracle_smoke"]["unflagged_high_risk_tool_secret_canary_checked"]
        is True
    )
    assert voice_result["details"]["async_oracle_smoke"]["unflagged_high_risk_tool_name"] == "write_memory"
    assert voice_result["details"]["async_oracle_smoke"]["unflagged_high_risk_tool_spoken"][0] == (
        "Preparing the spend request."
    )
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_bridge_smoke_ok"] is True
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_tool_call_id"] == "voiceclaw-call-1"
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_completion_tool_call_id"]
        == "voiceclaw-call-1"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_status_tool_call_id"]
        == "voiceclaw-call-1"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_terminal_correlation_observed"]
        is True
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_audit_id"]
        == "voiceclaw-audit-001"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_source_audit_id"]
        == "discord-audit-voice-001"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_parent_audit_id"]
        == "discord-audit-root-001"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_audit_id_continuity_observed"]
        is True
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_hypothesis_not_durable_oracle_text"]
        is True
    )
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_provisional_request_summary"] == {
        "text": "Prepare external KAME handoff",
        "source": "reflex_audio",
        "authority": "reflex_hypothesis",
        "tool_authority": False,
    }
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_status_provisional_request_summary"]
        == voice_result["details"]["async_oracle_smoke"]["external_frontend_provisional_request_summary"]
    )
    assert (
        voice_result["details"]["async_oracle_smoke"][
            "external_frontend_provisional_request_summary_non_authoritative"
        ]
        is True
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_witness_tool_authority_false"]
        is True
    )
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_witness_role_context"] is True
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_witness_promotion_required"] is True
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_protocol"] == "kame_session_v1"
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_protocol_contract"]
        == "docs/kame-session-v1.md"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_witness_kind"]
        == "frontend_witness_hypothesis"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"][
            "external_frontend_witness_kind_frontend_hypothesis"
        ]
        is True
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_status_audit_id"]
        == "voiceclaw-audit-001"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_completion_audit_id"]
        == "voiceclaw-audit-001"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_witness_metadata_complete"]
        is True
    )
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_witness_confidence"] == 0.78
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_witness_latency_ms"] == 140
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_witness_partial"] is False
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_witness_audio_time_range_ms"] == [
        120,
        2080,
    ]
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_witness_speaker"][
            "channel_user_id"
        ]
        == "jetha-redacted"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["external_frontend_witness_channel"][
            "channel_id"
        ]
        == "general-redacted"
    )
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_durable_user_messages_empty"] is True
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_durable_oracle_text_absent"] is True
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_durable_record_count"] >= 1
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_direct_tool_authority_exposed"] is False
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_evidence_merge_key"].startswith(
        "kame-merge-"
    )
    assert voice_result["details"]["async_oracle_smoke"]["external_frontend_evidence_merge_key_propagated"] is True
    assert voice_result["details"]["async_oracle_smoke"]["unpromoted_hypothesis_action_sinks_clean"] is True
    assert set(
        voice_result["details"]["async_oracle_smoke"]["unpromoted_hypothesis_action_sink_keys_checked"]
    ) >= {
        "spend_reason",
        "spend_payload",
        "phone_call_payload",
        "call_payload",
        "tool_arguments",
        "arguments",
        "memory_write",
        "file_write",
        "message_payload",
        "external_message",
        "durable_history",
        "durable_user_history",
        "durable_transcript",
    }
    assert voice_result["details"]["async_oracle_smoke"]["unpromoted_hypothesis_action_sink_values"] == {}
    assert voice_result["details"]["async_oracle_smoke"]["unpromoted_hypothesis_not_spend_reason"] is True
    assert voice_result["details"]["async_oracle_smoke"]["unpromoted_hypothesis_not_spend_payload"] is True
    assert voice_result["details"]["async_oracle_smoke"]["unpromoted_hypothesis_not_provider_selection"] is True
    assert (
        voice_result["details"]["async_oracle_smoke"]["unpromoted_hypothesis_not_nemoclaw_action_packet"]
        is True
    )
    assert voice_result["details"]["async_oracle_smoke"]["unpromoted_hypothesis_not_phone_call_payload"] is True
    assert voice_result["details"]["async_oracle_smoke"]["unpromoted_hypothesis_not_call_payload"] is True
    assert voice_result["details"]["async_oracle_smoke"]["unpromoted_hypothesis_not_tool_arguments"] is True
    assert voice_result["details"]["async_oracle_smoke"]["unpromoted_hypothesis_tool_authority_false"] is True
    assert voice_result["details"]["async_oracle_smoke"]["unpromoted_hypothesis_not_memory_write"] is True
    assert voice_result["details"]["async_oracle_smoke"]["unpromoted_hypothesis_not_file_write"] is True
    assert voice_result["details"]["async_oracle_smoke"]["unpromoted_hypothesis_not_message_payload"] is True
    assert voice_result["details"]["async_oracle_smoke"]["unpromoted_hypothesis_not_durable_history"] is True
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_timing_smoke_ok"] is True
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_arrival_phases"] == [
        "before_raw_audio",
        "with_raw_audio",
        "after_interpreter_start",
    ]
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_turn_ids"] == {
        "early": "witness-fusion:early",
        "with": "witness-fusion:with",
        "late": "witness-fusion:late",
    }
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_audio_segment_refs"] == {
        "early": "artifact://voice/witness-early.wav",
        "with": "artifact://voice/witness-with.wav",
        "late": "artifact://voice/witness-late.wav",
    }
    assert all(
        value.startswith("kame-merge-")
        for value in voice_result["details"]["async_oracle_smoke"]["witness_fusion_evidence_merge_keys"].values()
    )
    assert len(set(voice_result["details"]["async_oracle_smoke"]["witness_fusion_evidence_merge_keys"].values())) == 3
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_merge_key_observed"] is True
    assert (
        voice_result["details"]["async_oracle_smoke"]["witness_fusion_same_turn_convergence_ok"]
        is True
    )
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_same_turn_arrival_phases"] == [
        "before_raw_audio",
        "with_raw_audio",
        "after_interpreter_start",
    ]
    same_turn_lineage = voice_result["details"]["async_oracle_smoke"][
        "witness_fusion_same_turn_lineage"
    ]
    assert same_turn_lineage["turn_id"] == "witness-fusion:same-turn"
    assert same_turn_lineage["audio_segment_ref"] == "artifact://voice/witness-same-turn.wav"
    assert (
        same_turn_lineage["evidence_merge_key"]
        == voice_result["details"]["async_oracle_smoke"][
            "witness_fusion_same_turn_expected_merge_key"
        ]
    )
    assert len(
        {
            tuple(sorted(lineage.items()))
            for lineage in voice_result["details"]["async_oracle_smoke"][
                "witness_fusion_same_turn_phase_lineage"
            ].values()
        }
    ) == 1
    assert voice_result["details"]["async_oracle_smoke"][
        "witness_fusion_same_turn_oracle_job_counts"
    ] == {"accepted": 1, "started": 1, "completed": 1}
    assert (
        voice_result["details"]["async_oracle_smoke"][
            "witness_fusion_same_turn_no_duplicate_oracle_job"
        ]
        is True
    )
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_accepted_audio_gate_observed"] is True
    assert (
        voice_result["details"]["async_oracle_smoke"]["witness_fusion_bundle_audio_metadata"]
        == voice_result["details"]["async_oracle_smoke"]["witness_fusion_audio_metadata"]
    )
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_audio_metadata"]["early"][
        "energy_gate"
    ] == {
        "accepted": True,
        "rms": 620,
        "duration_ms": 1300,
        "min_rms": 350,
        "min_speech_ms": 120,
    }
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_early_single_bundle"] is True
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_interpreter_prompt_input_order"] == [
        "raw_audio",
        "metadata",
        "reflex",
        "transcript_hypotheses",
    ]
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_interpreter_prompt_input_order_expected"] == [
        "raw_audio",
        "metadata",
        "reflex",
        "transcript_hypotheses",
    ]
    assert (
        voice_result["details"]["async_oracle_smoke"]["witness_fusion_interpreter_prompt_input_order_visible"]
        is True
    )
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_interpreter_prompt_policy"] == {
        "version": "raw_audio_compare_v1",
        "primary_evidence": "raw_audio",
        "transcript_hypotheses_authority": "non_authoritative_context",
        "promotion_requirement": "compare_transcript_hypotheses_against_raw_audio_before_promotion",
        "forbidden_direct_uses": (
            "oracle_text",
            "durable_transcript",
            "spend_reason",
            "phone_call_payload",
            "tool_arguments",
        ),
    }
    assert (
        voice_result["details"]["async_oracle_smoke"]["witness_fusion_interpreter_prompt_policy_expected"]
        == voice_result["details"]["async_oracle_smoke"]["witness_fusion_interpreter_prompt_policy"]
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["witness_fusion_interpreter_prompt_policy_version"]
        == "raw_audio_compare_v1"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["witness_fusion_interpreter_prompt_policy_visible"]
        is True
    )
    assert voice_result["details"]["async_oracle_smoke"]["energy_gate_smoke_ok"] is True
    assert voice_result["details"]["async_oracle_smoke"]["energy_gate_policy"] == {
        "min_rms": 350,
        "min_speech_ms": 120,
    }
    assert voice_result["details"]["async_oracle_smoke"]["energy_gate_ignored_packet_rms"] == 80
    assert voice_result["details"]["async_oracle_smoke"]["energy_gate_ignored_packet_duration_ms"] == 200
    assert (
        voice_result["details"]["async_oracle_smoke"]["energy_gate_ignored_packet_speech_confirmed"]
        is False
    )
    assert voice_result["details"]["async_oracle_smoke"]["energy_gate_ignored_packet_vad_speech"] is False
    assert voice_result["details"]["async_oracle_smoke"]["energy_gate_ignored_non_speech_packets"] >= 2
    assert (
        voice_result["details"]["async_oracle_smoke"]["energy_gate_low_energy_witness_text"]
        == "spend money from room tone"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["energy_gate_low_energy_witness_source"]
        == "moshi"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["energy_gate_low_energy_witness_promoted"]
        is False
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["energy_gate_low_energy_witness_suppressed"]
        is True
    )
    assert voice_result["details"]["async_oracle_smoke"]["energy_gate_barge_in_events"] == 0
    assert voice_result["details"]["async_oracle_smoke"]["energy_gate_interpreter_requests"] == 0
    assert voice_result["details"]["async_oracle_smoke"]["energy_gate_oracle_work_events"] == 0
    assert voice_result["details"]["async_oracle_smoke"]["energy_gate_oracle_requests"] == 0
    assert (
        voice_result["details"]["async_oracle_smoke"]["energy_gate_raw_packet_buffered_without_turn"]
        is True
    )
    assert "barge_in.detected" not in voice_result["details"]["async_oracle_smoke"]["energy_gate_event_types"]
    assert voice_result["details"]["async_oracle_smoke"]["kame_ack_latency_metrics_smoke_ok"] is True
    assert (
        voice_result["details"]["async_oracle_smoke"]["kame_defer_ack_first_audio_metrics_visible"]
        is True
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["kame_local_first_audio_metrics_visible"]
        is True
    )
    assert (
        "kame_interface_decision_to_defer_first_audio_ms"
        in voice_result["details"]["async_oracle_smoke"]["kame_defer_ack_metric_keys"]
    )
    assert (
        "kame_interface_decision_to_local_first_audio_ms"
        in voice_result["details"]["async_oracle_smoke"]["kame_local_first_audio_metric_keys"]
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["kame_defer_speech_end_to_first_audio_ms"]
        >= 41
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["kame_local_speech_end_to_first_audio_ms"]
        >= 37
    )
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_with_single_bundle"] is True
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_late_single_bundle"] is True
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_no_duplicate_oracle_jobs"] is True
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_partial_superseded_by_final"] is True
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_partial_active_hypothesis"] == {
        "source": "moshi",
        "kind": "frontend_witness_hypothesis",
        "text": "what is three to the power of seventeen",
        "role": "witness_context",
        "authority": "hypothesis",
        "promotion_required": "interpreter_promoted_or_oracle_promoted",
        "tool_authority": False,
        "confidence": 0.88,
        "arrival_phase": "with_raw_audio",
        "partial": False,
        "superseded_partial_texts": ("what is three to the",),
        "superseded_partial_count": 1,
    }
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_adjudications"] == {
        "early": ["corrected_by_audio"],
        "with": ["accepted_as_supporting_evidence"],
        "late": ["rejected_or_diagnostic_only"],
    }
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_rejection_reasons"] == {
        "early": [],
        "with": [],
        "late": ["ambiguous_speaker", "wrong_speaker", "wrong_channel", "stale_witness"],
    }
    assert voice_result["details"]["async_oracle_smoke"]["witness_fusion_adjudication_outcomes_observed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["runtime_kame_action_gate_smoke_ok"] is True
    assert voice_result["details"]["async_oracle_smoke"]["runtime_kame_action_gate_hypothesis_only_ok"] is False
    assert "missing_promoted_evidence" in voice_result["details"]["async_oracle_smoke"][
        "runtime_kame_action_gate_hypothesis_only_issues"
    ]
    assert (
        voice_result["details"]["async_oracle_smoke"]["runtime_kame_action_gate_degraded_text_only_ok"]
        is False
    )
    assert (
        voice_result["details"]["async_oracle_smoke"][
            "runtime_kame_action_gate_degraded_text_only_status"
        ]
        == "degraded_text_only"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"][
            "runtime_kame_action_gate_degraded_text_only_reason"
        ]
        == "degraded_text_only"
    )
    assert (
        voice_result["details"]["async_oracle_smoke"][
            "runtime_kame_action_gate_degraded_text_only_raw_audio_available"
        ]
        is False
    )
    assert (
        voice_result["details"]["async_oracle_smoke"][
            "runtime_kame_action_gate_degraded_text_only_preserves_hypothesis"
        ]
        is True
    )
    assert "missing_promoted_evidence" in voice_result["details"]["async_oracle_smoke"][
        "runtime_kame_action_gate_degraded_text_only_issues"
    ]
    assert voice_result["details"]["async_oracle_smoke"]["runtime_kame_action_gate_promoted_ok"] is True
    assert voice_result["details"]["async_oracle_smoke"]["runtime_kame_action_gate_promoted_authorities"] == [
        "interpreter_promoted"
    ]
    assert (
        voice_result["details"]["async_oracle_smoke"]["runtime_kame_action_gate_promoted_consumed_before_action"]
        is True
    )
    assert voice_result["details"]["async_oracle_smoke"]["runtime_kame_action_gate_self_attested_ok"] is False
    assert voice_result["details"]["async_oracle_smoke"]["runtime_kame_action_gate_self_attested_issues"] == [
        "missing_promoted_evidence"
    ]
    assert voice_result["details"]["async_oracle_smoke"]["runtime_kame_action_gate_self_attested_authorities"] == []
    assert (
        voice_result["details"]["async_oracle_smoke"][
            "runtime_kame_action_gate_self_attested_consumed_before_action"
        ]
        is True
    )
    assert (
        voice_result["details"]["async_oracle_smoke"]["runtime_kame_action_gate_missing_tool_disclosure_ok"]
        is False
    )
    assert voice_result["details"]["async_oracle_smoke"][
        "runtime_kame_action_gate_missing_tool_disclosure_issues"
    ] == ["missing_tool_disclosure_ref"]
    assert voice_result["details"]["async_oracle_smoke"][
        "runtime_kame_action_gate_missing_tool_disclosure_authorities"
    ] == ["interpreter_promoted"]
    assert voice_result["details"]["async_oracle_smoke"]["runtime_kame_action_gate_tool_disclosure_ref_observed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["durable_resume_contract_smoke_ok"] is True
    assert (
        voice_result["details"]["async_oracle_smoke"]["durable_resume_contract_schema_version"]
        == "voiceops.kame_durable_resume_context.v1"
    )
    assert voice_result["details"]["async_oracle_smoke"]["durable_resume_promoted_turn_count"] == 4
    assert voice_result["details"]["async_oracle_smoke"]["durable_resume_recent_promoted_turns_verbatim"] is True
    assert voice_result["details"]["async_oracle_smoke"]["durable_resume_recent_promoted_turns"] == [
        {
            "turn_id": "voice-smoke-durable-resume:3",
            "text": "promoted durable resume request 3",
            "source": "gemma_interpreter",
            "authority": "promoted",
        },
        {
            "turn_id": "voice-smoke-durable-resume:4",
            "text": "promoted durable resume request 4",
            "source": "gemma_interpreter",
            "authority": "promoted",
        },
    ]
    assert voice_result["details"]["async_oracle_smoke"]["durable_resume_older_turns_summarized"] is True
    assert voice_result["details"]["async_oracle_smoke"]["durable_resume_older_promoted_turn_count"] == 2
    assert "voice-smoke-durable-resume:1" in voice_result["details"]["async_oracle_smoke"][
        "durable_resume_older_promoted_turn_summary"
    ]
    assert "voice-smoke-durable-resume:2" in voice_result["details"]["async_oracle_smoke"][
        "durable_resume_older_promoted_turn_summary"
    ]
    assert voice_result["details"]["async_oracle_smoke"]["durable_resume_hypothesis_replay_absent"] is True
    assert voice_result["details"]["async_oracle_smoke"]["durable_resume_ledger_authoritative"] is True
    assert voice_result["details"]["async_oracle_acceptance"]["durable_promoted_turn_resume_contract"][
        "ok"
    ] is True
    assert voice_result["details"]["async_oracle_smoke"]["audit_scalar_smoke_ok"] is True
    assert voice_result["details"]["async_oracle_smoke"]["audit_scalar_payload_redacted"] is True
    assert voice_result["details"]["async_oracle_smoke"]["audit_scalar_secret_canary_checked"] is True
    assert voice_result["details"]["async_oracle_smoke"]["audit_scalar_result_text_omitted"] is True
    assert voice_result["details"]["async_oracle_smoke"]["audit_scalar_completed_event_seen"] is True
    assert voice_result["details"]["async_oracle_smoke"]["audit_scalar_waiting_event_seen"] is True
    assert voice_result["details"]["async_oracle_smoke"]["audit_scalar_row_count"] == 5
    assert voice_result["details"]["async_oracle_smoke"]["shutdown_timeout_configured_ms"] == 10
    assert voice_result["details"]["async_oracle_smoke"]["shutdown_bounded_close_observed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["shutdown_forced_cancel_observed"] is True
    assert voice_result["details"]["async_oracle_smoke"]["shutdown_close_cancel_entered"] is True
    assert voice_result["details"]["async_oracle_smoke"]["shutdown_cancelled_jobs"] == 1
    assert voice_result["details"]["discord_session_cleanup_smoke"]["ok"] is True
    assert voice_result["details"]["discord_session_cleanup_smoke"]["cancel_all_before_session_closed"] is True
    assert voice_result["details"]["discord_session_cleanup_smoke"]["session_closed_sent"] is True
    assert voice_result["details"]["discord_session_cleanup_smoke"]["sidecar_closed"] is True
    assert voice_result["details"]["discord_session_cleanup_smoke"]["degraded_active_job_preserved_failed"] is True
    assert voice_result["details"]["discord_session_cleanup_smoke"]["degraded_session_removed"] is True
    assert voice_result["details"]["discord_session_cleanup_smoke"]["degraded_job_state"] == "failed"
    assert voice_result["details"]["sidecar_fail_closed_smoke"]["ok"] is True
    assert voice_result["details"]["sidecar_fail_closed_smoke"]["request_accepted"] is True
    assert voice_result["details"]["sidecar_fail_closed_smoke"]["cancel_reason"] == "sidecar_send_failed"
    assert voice_result["details"]["sidecar_fail_closed_smoke"]["active_capacity_after_failure"] == 0
    assert voice_result["details"]["sidecar_fail_closed_smoke"]["job_state_after_failure"] == "cancelled"
    assert voice_result["details"]["async_oracle_acceptance"]["four_oracle_jobs_reflex_responsive"]["ok"] is True
    assert voice_result["details"]["async_oracle_acceptance"]["fifth_job_obeys_overflow_policy"]["ok"] is True
    assert voice_result["details"]["async_oracle_acceptance"]["approval_wait_is_visible_and_redacted"]["ok"] is True
    assert (
        voice_result["details"]["async_oracle_acceptance"]["approval_wait_is_visible_and_redacted"][
            "runtime_verified_by_this_report"
        ]
        is True
    )
    assert (
        voice_result["details"]["async_oracle_acceptance"]["failed_job_is_reported_without_crashing_session"][
            "runtime_verified_by_this_report"
        ]
        is True
    )
    assert (
        voice_result["details"]["async_oracle_acceptance"]["job_control_updates_reach_oracle"][
            "runtime_verified_by_this_report"
        ]
        is True
    )
    assert voice_result["details"]["async_oracle_acceptance"]["result_handling_is_bounded_and_durable"][
        "test_ref_count"
    ] >= 1
    assert voice_result["details"]["async_oracle_acceptance"]["result_handling_is_bounded_and_durable"][
        "verification_mode"
    ] == "loopback_smoke_plus_focused_tests"
    assert voice_result["details"]["async_oracle_acceptance"]["result_handling_is_bounded_and_durable"][
        "runtime_verified_by_this_report"
    ] is True
    assert voice_result["details"]["async_oracle_acceptance"]["discord_session_cleanup_preserves_oracle_state"][
        "verification_mode"
    ] == "loopback_smoke_plus_focused_tests"
    assert voice_result["details"]["async_oracle_acceptance"]["discord_session_cleanup_preserves_oracle_state"][
        "runtime_verified_by_this_report"
    ] is True
    assert voice_result["details"]["async_oracle_acceptance"]["sidecar_fail_closed_send_failure_cancels_active_job"][
        "verification_mode"
    ] == "loopback_smoke_plus_focused_tests"
    assert voice_result["details"]["async_oracle_acceptance"]["sidecar_fail_closed_send_failure_cancels_active_job"][
        "runtime_verified_by_this_report"
    ] is True
    assert voice_result["details"]["async_oracle_acceptance"]["shutdown_timeout_is_bounded"][
        "verification_mode"
    ] == "loopback_smoke_plus_focused_tests"
    assert voice_result["details"]["async_oracle_acceptance"]["shutdown_timeout_is_bounded"][
        "runtime_verified_by_this_report"
    ] is True

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
    assert provisioning_result["details"]["nemoclaw_action_packet_status"] == "valid"
    assert provisioning_result["details"]["nemoclaw_action_packet_validation_issues"] == []
    assert "--nemoclaw-action-packet" in provisioning_result["command"]
    assert provisioning_result["details"]["input_paths"]["nemoclaw_action_packet"].endswith(
        "hackathon-voiceops-demo/current/nemoclaw-action-packet.json"
    )
    assert Path(provisioning_result["artifacts"]["execution_plan_json"]).exists()
    assert Path(provisioning_result["artifacts"]["execution_plan_markdown"]).exists()
    assert Path(provisioning_result["artifacts"]["post_approval_receipts_template"]).exists()
    assert Path(provisioning_result["artifacts"]["post_approval_receipts_validation"]).exists()
    assert Path(provisioning_result["artifacts"]["nemoclaw_action_packet_validation"]).exists()
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
    assert summary["review_gaps"] == ["milestone_3_multi_channel_policy"]
    assert summary["closure_index"]["review_gaps"] == summary["review_gaps"]

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
    assert payload["artifact_id"] == "voiceops-plan-run"
    assert payload["ok"] is True
    assert payload["readiness_ok"] is False
    assert payload["ready_for_demo"] is False
    assert payload["artifacts"]["json"] == paths["json"]
    assert payload["artifact_manifest"]["json"] == "voiceops-plan-run.json"
    assert payload["blockers"] == summary["blockers"]
    assert payload["current_environment_blockers"] == closure["current_environment_blockers"]
    assert payload["closure_status"] == closure["closure_status"]
    assert payload["remaining_gates"] == [gate["gate_id"] for gate in closure["remaining_gates"]]
    assert payload["next_actions"] == closure["next_actions"]
    assert payload["review_actions"] == closure["review_actions"]
    assert closure["artifact_id"] == "voiceops-plan-readiness-closure"
    assert closure["schema_version"] == "voiceops.closure_index.v1"
    assert handoff_payload == closure["operator_handoff"]
    assert handoff_payload["schema_version"] == "voiceops.operator_handoff.v1"
    assert [phase["order"] for phase in handoff_payload["phases"]] == [1, 2, 3]
    assert [phase["phase_id"] for phase in handoff_payload["review_phases"]] == ["multi_channel_policy_review"]
    assert handoff_payload["review_phases"][0]["status"] == "pending_human_review"
    assert handoff_payload["review_phases"][0]["real_egress_enabled"] is False
    assert [action["phase_id"] for action in payload["review_actions"]] == ["multi_channel_policy_review"]
    assert payload["review_actions"][0]["status"] == "pending_human_review"
    assert payload["review_actions"][0]["real_egress_enabled"] is False
    assert [phase["first_safe_command"] for phase in handoff_payload["phases"]] == [
        closure["next_actions"][0]["first_safe_command"],
        closure["next_actions"][1]["first_safe_command"],
        closure["next_actions"][2]["first_safe_command"],
    ]
    assert [phase["blocked_by_current_environment"] for phase in handoff_payload["phases"]] == [
        {
            "missing_env_keys": closure["next_actions"][0]["blocked_by_current_environment"]["missing_env_keys"],
            "present_env_keys": closure["current_environment_blockers"]["discord_env"]["present_env_keys"],
            "needs_external_live_probe": True,
        },
        {
            "missing_cli": closure["next_actions"][1]["blocked_by_current_environment"]["missing_cli"],
            "present_cli": closure["current_environment_blockers"]["provisioning_cli"]["present"],
            "needs_read_only_discovery": True,
            "needs_redacted_setup_evidence": True,
        },
        {
            "required_hardware": closure["next_actions"][2]["blocked_by_current_environment"]["required_hardware"],
            "current_host_hint": closure["next_actions"][2]["blocked_by_current_environment"]["current_host_hint"],
            "needs_measured_spark_evidence": True,
        },
    ]
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
    assert "--run-command-probes" in provisioning_gate["rerun_commands"]["plan_index_command_probes"]
    assert provisioning_gate["evidence_scaffold"].endswith(
        "provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json"
    )
    assert "validate_post_approval_receipts" in provisioning_gate["collection_commands"]
    assert "read_only_discovery" in provisioning_gate["collection_commands"]
    assert "refresh_preflight_source_hashes" in provisioning_gate["collection_commands"]
    assert "validate_nemoclaw_action_packet" in provisioning_gate["collection_commands"]
    assert "--dry-audit" in provisioning_gate["rerun_commands"]["plan_index_dry_audit"]
    assert "--refresh-preflight-source-hashes" in provisioning_gate["collection_commands"]["refresh_preflight_source_hashes"]
    assert "--nemoclaw-action-packet" in provisioning_gate["collection_commands"]["validate_nemoclaw_action_packet"]
    assert "--run-readonly-discovery" in provisioning_gate["collection_commands"]["read_only_discovery"]
    assert "--run-readonly-discovery" in provisioning_gate["rerun_commands"]["plan_index_read_only_discovery"]
    assert "--refresh-preflight-source-hashes" in provisioning_gate["rerun_commands"]["refresh_preflight_source_hashes"]
    assert provisioning_gate["evidence_contract"]["read_only_discovery_grants_approval"] is False
    assert provisioning_gate["evidence_contract"]["nemoclaw_action_packet_validation_grants_approval"] is False
    assert provisioning_gate["evidence_contract"]["nemoclaw_action_packet_validation_executes_commands"] is False
    assert "voiceops.milestone2.post_approval_receipts.v1" == provisioning_gate["evidence_contract"][
        "post_approval_receipts_schema_version"
    ]
    assert provisioning_gate["evidence_contract"]["post_approval_collector_attestation_required"] is True
    assert provisioning_gate["evidence_contract"]["post_approval_collector_attestation_redacted_sha256_must_match"] is True
    assert provisioning_gate["evidence_contract"]["post_approval_collector_attestation_required_fields"] == [
        "collector_name",
        "collector_version",
        "run_id",
        "command_argv",
        "git_commit",
        "started_at",
        "finished_at",
        "raw_artifact_sha256",
        "redacted_artifact_sha256",
        "parent_manifest_sha256",
    ]
    assert provisioning_gate["evidence_contract"]["post_approval_linkage_ids_must_be_unique"] == [
        "credential_locations[].credential_ref_id",
        "rollback_receipts[].rollback_ref",
        "audit_events[].audit_event_id",
    ]
    assert "--post-approval-receipts" in provisioning_gate["rerun_commands"]["plan_index_post_approval_receipts"]
    assert "--post-approval-receipts" in handoff_payload["final_reindex_command"]
    spark_gate = next(gate for gate in closure["gates"] if gate["gate_id"] == "local_spark_stack_matrix")
    assert spark_gate["closure_plan"].endswith("spark-matrix-closure-plan.json")
    assert spark_gate["closure_artifact"].endswith("spark-matrix-closure-plan.md")
    assert spark_gate["operator_runbook"].endswith("spark-operator-runbook.md")
    assert spark_gate["evidence_scaffold"].endswith("spark-benchmark-scaffold/spark-benchmark-evidence.json")
    assert (
        "artifacts/voiceops-spark-matrix/current/spark-benchmark-scaffold/spark-benchmark-evidence.json"
        in handoff_payload["final_reindex_command"]
    )
    assert "path/to/spark-benchmark-evidence.json" not in handoff_payload["final_reindex_command"]
    assert spark_gate["evidence_contract"]["example_only_accepted"] is False
    assert spark_gate["evidence_contract"]["source_artifacts_must_exist"] is True
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
    assert "Final package audit command" in closure_markdown
    assert "voiceops_artifact_package_audit.py" in closure_markdown
    assert "--package-audit" in closure_markdown
    assert "Next Actions" in closure_markdown
    assert "Review Actions" in closure_markdown
    assert "multi_channel_policy_review" in closure_markdown
    assert "First safe command" in closure_markdown
    assert "needs_external_live_probe" in closure_markdown
    assert "live_discord_voice_operator" in closure_markdown
    assert "voiceops.realtime_voice_live_evidence_manifest.v1" in closure_markdown
    assert "python -m hermes_cli.realtime_voice_live_evidence" in closure_markdown
    assert "--sidecar-session-evidence" in closure_markdown
    assert "--live-turn-evidence" in closure_markdown
    assert "required_sidecar_mode" in closure_markdown
    assert "required_sidecar_latency_metrics_ms" in closure_markdown
    assert "required_discord_latency_metrics_ms" in closure_markdown
    assert "required_collector_attestation_fields" in closure_markdown
    assert "collector_attestation_required_for_live_readiness" in closure_markdown
    assert "redacted_artifact_sha256" in closure_markdown
    assert "parent_manifest_sha256" in closure_markdown
    assert "realtime_voice_report_derivation_schema_version" in closure_markdown
    assert "unverified_source_artifacts_accepted" in closure_markdown
    assert "collector_attestation" in closure_markdown
    assert "voiceops.milestone2.preflight_evidence_manifest.v1" in closure_markdown
    scaffold_manifest_path = (
        "artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/"
        "provisioning-preflight-evidence.manifest.json"
    )
    stale_manifest_path = "artifacts/voiceops-provisioning/current/provisioning-preflight-evidence.manifest.json"
    assert scaffold_manifest_path in closure_markdown
    assert stale_manifest_path not in closure_markdown
    assert "--refresh-preflight-source-hashes" in closure_markdown
    assert "--dry-audit" in closure_markdown
    assert "network_possible_allowlisted_read_only" in closure_markdown
    assert "voiceops.spark_benchmark_evidence.v1" in closure_markdown
    assert "spark-benchmark-evidence.example.json" in closure_markdown
    assert "spark-operator-runbook.md" in closure_markdown
    assert "scripts/dgx_spark_gemma4_voice_eval.sh" in closure_markdown
    assert "loopback_smoke_bridge protocol smoke checks" in closure_markdown
    assert "required_oracle_authority_routes" in closure_markdown
    assert "VoiceOps Operator Handoff" in handoff_markdown
    assert "### 1. live_discord_voice" in handoff_markdown
    assert "### 2. spend_and_provisioning_preflight" in handoff_markdown
    assert "### 3. local_spark_stack" in handoff_markdown
    assert "Blocked by current environment" in handoff_markdown
    assert "loopback_smoke_bridge protocol smoke checks" in handoff_markdown
    assert "live_discord_voice" in handoff_markdown
    assert "sidecar_mode=production" in handoff_markdown
    assert "provider_transport_observed" in handoff_markdown
    assert "--dry-audit" in handoff_markdown
    assert "no_write_no_network_no_probe_audit" in handoff_markdown
    assert "Final reindex command" in handoff_markdown
    assert "Final package audit command" in handoff_markdown
    assert "package_audit.status is pass" in handoff_markdown
    assert "milestone_0_hackathon_proof" in markdown
    assert "Next Actions" in markdown


def test_plan_run_closes_remaining_gates_with_redacted_local_evidence(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    output_dir = artifact_root / "voiceops-plan" / "current"
    fake_bin = tmp_path / "bin"
    for binary in ["stripe", "link-cli", "mppx"]:
        _write_fake_bin(fake_bin, binary)
    fake_path = os.pathsep.join([str(fake_bin), os.environ.get("PATH", "")])
    monkeypatch.setenv("PATH", fake_path)
    live_evidence = _write_live_voice_evidence(tmp_path / "live-voice")
    preflight_evidence = _write_preflight_evidence(tmp_path / "preflight")
    read_only_discovery_evidence = _write_readonly_discovery_evidence(tmp_path / "read-only-discovery")
    post_approval_receipts = _write_post_approval_receipts(tmp_path / "post-approval")
    spark_evidence = _write_spark_evidence(tmp_path / "spark")

    summary = build_plan_run(
        artifact_root=artifact_root,
        output_dir=output_dir,
        env={
            "PATH": fake_path,
            "DISCORD_BOT_TOKEN": "present-redacted",
            "DISCORD_GUILD_ID": "guild-ref-demo",
            "DISCORD_HOME_CHANNEL": "channel-ref-demo",
            "DISCORD_VOICE_CHANNEL_ID": "voice-channel-ref-demo",
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "TWILIO_ACCOUNT_SID": "AC123",
        },
        voice_live_evidence_paths=[live_evidence],
        provisioning_preflight_evidence=preflight_evidence,
        read_only_discovery_evidence=read_only_discovery_evidence,
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
    assert summary["review_gaps"] == ["milestone_3_multi_channel_policy"]
    assert summary["closure_index"]["closure_status"] == "needs_external_evidence"
    assert summary["closure_index"]["remaining_gates"] == []
    assert statuses["milestone_1_real_voice_operator"] == "live_evidence_supplied"
    assert statuses["milestone_2_real_spend_and_provisioning_preflight"] == "ready"
    assert statuses["milestone_4_local_spark_stack_matrix"] == "validated"
    assert provisioning_result["details"]["post_approval_receipts_loaded"] is True
    assert provisioning_result["details"]["post_approval_receipts_status"] == "valid"
    assert provisioning_result["details"]["post_approval_receipt_count"] == 4
    assert provisioning_result["details"]["post_approval_receipts_validation_issues"] == []
    assert provisioning_result["details"]["read_only_discovery_status"] == "pass"
    assert provisioning_result["details"]["read_only_discovery_failed_probe_ids"] == []
    assert provisioning_result["details"]["read_only_discovery_missing_probe_ids"] == []
    assert provisioning_result["details"]["read_only_discovery_timed_out_probe_ids"] == []
    assert provisioning_result["details"]["nemoclaw_action_packet_status"] == "valid"
    assert provisioning_result["details"]["nemoclaw_action_packet_validation_issues"] == []
    assert provisioning_result["details"]["run_readonly_discovery"] is False
    assert "--run-readonly-discovery" not in provisioning_result["command"]
    assert "--read-only-discovery-evidence" in provisioning_result["command"]
    assert "--preflight-evidence" in provisioning_result["command"]
    assert str(preflight_evidence) in provisioning_result["command"]
    assert str(read_only_discovery_evidence) in provisioning_result["command"]
    assert "--post-approval-receipts" in provisioning_result["command"]
    assert "--nemoclaw-action-packet" in provisioning_result["command"]
    assert str(post_approval_receipts) in provisioning_result["command"]
    assert provisioning_result["details"]["input_paths"]["preflight_evidence"] == str(preflight_evidence)
    assert provisioning_result["details"]["input_paths"]["read_only_discovery_evidence"] == str(read_only_discovery_evidence)
    assert provisioning_result["details"]["input_paths"]["post_approval_receipts"] == str(post_approval_receipts)
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
        "readiness_gaps is [] and review_gaps is [] and closure_status is complete and package_audit.status is pass"
    )
    assert "--package-audit" in summary["closure_index"]["operator_handoff"]["final_reindex_command"]
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
    provisioning_report = json.loads(
        (artifact_root / "voiceops-provisioning" / "current" / "provisioning-readiness.json").read_text(
            encoding="utf-8"
        )
    )
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    closure_markdown = Path(paths["closure_markdown"]).read_text(encoding="utf-8")

    assert provisioning_result["details"]["run_readonly_discovery"] is True
    assert provisioning_report["safety"]["network_io"] is True
    assert provisioning_report["safety"]["network_io_scope"] == "allowlisted_read_only_discovery"
    assert provisioning_report["safety"]["mutating_network_io"] is False
    assert provisioning_report["safety"]["provider_provisioning"] is False
    assert provisioning_report["safety"]["live_spend"] is False
    assert provisioning_report["safety"]["read_only_discovery_run_requested"] is True
    assert provisioning_report["safety"]["read_only_discovery_grants_approval"] is False
    assert provisioning_result["details"]["read_only_discovery_status"] == "pass"
    assert provisioning_result["details"]["read_only_discovery_failed_probe_ids"] == []
    assert provisioning_result["details"]["read_only_discovery_missing_probe_ids"] == []
    assert provisioning_result["details"]["read_only_discovery_timed_out_probe_ids"] == []
    provisioning_gate = next(
        gate
        for gate in summary["closure_index"]["remaining_gates"]
        if gate["gate_id"] == "spend_and_provisioning_preflight"
    )
    provisioning_action = next(
        action
        for action in summary["closure_index"]["next_actions"]
        if action["gate_id"] == "spend_and_provisioning_preflight"
    )
    provisioning_phase = next(
        phase
        for phase in summary["closure_index"]["operator_handoff"]["phases"]
        if phase["gate_id"] == "spend_and_provisioning_preflight"
    )
    demo_handoff = json.loads(
        (artifact_root / "hackathon-voiceops-demo" / "current" / "operator-handoff-preview.json").read_text(
            encoding="utf-8"
        )
    )
    demo_provisioning_phase = next(
        phase
        for phase in demo_handoff["phases"]
        if phase["gate_id"] == "spend_and_provisioning_preflight"
    )
    assert provisioning_gate["read_only_discovery_status"] == "pass"
    assert provisioning_action["blocked_by_current_environment"]["needs_read_only_discovery"] is False
    assert provisioning_phase["blocked_by_current_environment"]["needs_read_only_discovery"] is False
    assert demo_provisioning_phase["blocked_by_current_environment"]["needs_read_only_discovery"] is False
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
        "sidecar-session.from-realtime-report.json",
        "live-turn.from-realtime-report.json",
        "realtime-voice-report-validation.json",
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
        "package-audit.json",
        "package-audit.md",
    ]:
        assert f"`{artifact}`" in text
    assert "voiceops.realtime_voice_live_evidence_manifest.v1" in text
    assert "voiceops.realtime_voice_live_evidence_validation.v1" in text
    assert "`--validate-live-evidence`" in text
    assert "`--live-evidence-manifest`" in text
    assert "`--from-realtime-voice-report`" in text
    assert "`--audit-only`" in text
    assert "`--package-audit`" in text
    assert "package audit is part of final headless verification" in text
    assert "voiceops.realtime_voice_live_evidence_audit.v1" in text
    assert "voiceops.realtime_voice_report_derivation.v1" in text
    assert "must not claim production sidecar evidence from loopback or diagnostic sidecar modes" in text
    assert "performs no Discord network call" in text
    assert "voiceops.milestone1.live_voice_evidence.v1" in text
    assert "For non-manifest ingestion, pass one `--live-evidence` per section or combined file" in text
    assert "kind` or `evidence_type` values such as `discord_live_probe`, `sidecar_session`, or `live_turn`" in text
    assert "Manifest ingestion is preferred because manifest reports record the actual referenced report path as provenance" in text
    assert "placeholder source paths inside referenced artifacts are not trusted as provenance" in text
    assert "Template source artifact names such as `discord-live-probe.json`, `voice-status-or-sidecar-report.json`, `sidecar-session.json`, `voice-turn-evidence.json`, and `live-turn.json` are rejected" in text
    assert "collector_attestation" in text
    assert "collector name/version, run id, command argv, git commit, timestamp window" in text
    assert "raw/redacted SHA-256 hashes, and parent manifest hash" in text
    assert "source_artifact` for every redacted evidence section" in text
    assert "voiceops.milestone2.post_approval_receipts.v1" in text
    assert "`--post-approval-receipts`" in text
    assert "`--run-readonly-discovery`" in text
    assert "stripe projects list --limit 10" in text
    assert "link-cli auth status" in text
    assert "`source_artifact_kind: redacted_setup_evidence`, `source_artifact_sha256`, `source_artifact_redacted_at`, and `collector_attestation`" in text
    assert "SHA-256 must match the referenced redacted JSON source artifact" in text
    assert "attestation redacted hash must match that SHA-256" in text
    assert "refreshes `source_artifact_sha256` and `collector_attestation.redacted_artifact_sha256` together" in text
    assert "redaction and collection timestamps must be parseable with timezone information" in text
    assert "placeholder or `example_only` attestations are rejected" in text
    assert "all_local_stack_smoke" in text
    assert "source_artifact_sha256` and `collector_attestation.redacted_artifact_sha256`" in text
    assert "`--refresh-source-hashes path/to/evidence.json`" in text
    assert "oracle authority routes include tools/files/memory/project context" in text
    assert "reflex provider proves the chosen low-latency S2S or timing path" in text
    assert "interpreter provider proves Gemma raw-audio serving" in text
    assert "`speech_end_to_first_audio_ms <= 1500`" in text
    assert "`barge_in_stop_ms <= 150`" in text
    assert "`local_turn_oracle_calls == 0`" in text
    assert "`oracle_bound_oracle_calls >= oracle_bound_turns`" in text
    assert "local reflex turns must not call the oracle" in text
    assert "`--lint-evidence --evidence path/to/evidence.json`" in text
    assert "voiceops.spark_evidence_lint.v1" in text
    assert "scripts/dgx_spark_gemma4_voice_eval.sh" in text
    assert "spark-operator-runbook.md" in text
    assert "The operator handoff is the ordered execution runbook" in text
    assert "required proof shape for each gate including collector attestation requirements" in text
    assert "does not change readiness by itself" in text
    assert "`--dry-audit` builds the same plan summary in a temporary artifact root" in text
    assert "ordered `next_actions`" in text
    assert "The `next_actions` records are machine-readable" in text
    assert "Its `ok` field means no hard validation failures, not readiness" in text
    assert "closure rehearsal" in text.lower()
    assert "valid collector attestations" in text
    assert "`remaining_gates: []`" in text
    assert "local optional Stripe Skills bundle contracts" in text
    assert "optional-skills/payments/stripe-projects" in text


def test_goal_doc_plan_run_commands_include_package_audit():
    text = GOAL_DOC.read_text(encoding="utf-8")
    commands: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not current and line.startswith("uv run python scripts/voiceops_plan_run.py"):
            current = [line.rstrip(" \\")]
            if not line.endswith("\\"):
                commands.append(" ".join(current))
                current = []
            continue
        if current:
            current.append(line.rstrip(" \\"))
            if not line.endswith("\\"):
                commands.append(" ".join(current))
                current = []

    assert commands
    assert all("--package-audit" in command for command in commands)


def test_goal_doc_keeps_super_local_and_ultra_hosted():
    text = GOAL_DOC.read_text(encoding="utf-8")

    assert "Nemotron 3 Super is the preferred Spark-local NVIDIA oracle/model target" in text
    assert "A clearly labeled hosted `/model` fallback is acceptable only when the local Spark path is unavailable" in text
    assert "Ultra is only an optional hosted/upstream fallback" in text
    assert "must not be used as Spark-local readiness proof" in text
    assert "One-Spark readiness still requires measured local Spark benchmark evidence" in text
    assert "Hosted selections do not count as Spark-local readiness evidence" in text
    assert "There should not be a separate `oracle_model` setting for VoiceOps" in text
    assert "`/model` remains authoritative" in text
    assert "Display-only discovery is a separate opt-in path" in text
    assert "required before Milestone 2 can be considered ready for live provisioning approval" in text
    assert "isolated temporary `HOME`" in text
    assert "does not prove the operator's normal local CLI auth state" in text


def test_plan_run_propagates_active_and_reflex_model_to_demo_package(tmp_path):
    artifact_root = tmp_path / "artifacts"
    output_dir = artifact_root / "voiceops-plan" / "current"
    summary = build_plan_run(
        artifact_root=artifact_root,
        output_dir=output_dir,
        active_model="Nemotron 3 Super via hosted provider",
        reflex_model="Moshi fast reflex",
        interpreter_model="Gemma 4 E4B audio-native interpreter",
        env={},
    )

    demo_result = next(result for result in summary["results"] if result["milestone"] == "milestone_0_hackathon_proof")
    demo_path = artifact_root / "hackathon-voiceops-demo" / "current" / "voiceops-demo.json"
    demo = json.loads(demo_path.read_text(encoding="utf-8"))

    assert demo_result["details"]["active_model"] == "Nemotron 3 Super via hosted provider"
    assert demo_result["details"]["active_model_path"] == "hosted_nemotron_3_super_fallback"
    assert demo_result["details"]["reflex_model"] == "Moshi fast reflex"
    assert demo_result["details"]["interpreter_model"] == "Gemma 4 E4B audio-native interpreter"
    assert "'Nemotron 3 Super via hosted provider'" in demo_result["command"]
    assert "'Moshi fast reflex'" in demo_result["command"]
    assert "'Gemma 4 E4B audio-native interpreter'" in demo_result["command"]
    assert summary["plan_args"] == {
        "active_model": "Nemotron 3 Super via hosted provider",
        "reflex_model": "Moshi fast reflex",
        "interpreter_model": "Gemma 4 E4B audio-native interpreter",
    }
    assert "--active-model 'Nemotron 3 Super via hosted provider'" in summary["closure_index"]["operator_handoff"][
        "final_reindex_command"
    ]
    assert "--reflex-model 'Moshi fast reflex'" in summary["closure_index"]["operator_handoff"][
        "final_reindex_command"
    ]
    assert "--interpreter-model 'Gemma 4 E4B audio-native interpreter'" in summary["closure_index"][
        "operator_handoff"
    ][
        "final_reindex_command"
    ]
    assert "--active-model 'Nemotron 3 Super via hosted provider'" in summary["next_actions"][1]["first_safe_command"]
    assert demo["sponsor_stack"]["hermes_active_model"]["path"] == "hosted_nemotron_3_super_fallback"
    assert demo["spark_stack"]["current_path_local"] is False
    assert demo["spark_stack"]["reflex"]["model"] == "Moshi fast reflex"
    assert demo["spark_stack"]["interpreter"]["model"] == "Gemma 4 E4B audio-native interpreter"


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


def test_plan_run_current_environment_tracks_bland_phone_provider_keys(tmp_path):
    artifact_root = tmp_path / "artifacts"
    output_dir = artifact_root / "voiceops-plan" / "current"
    summary = build_plan_run(
        artifact_root=artifact_root,
        output_dir=output_dir,
        env={
            "PATH": "",
            "BLAND_PHONE_NUMBER": "+15551234567",
            "BLAND_API_KEY": "bland-secret-token",
        },
    )

    env_presence = summary["current_environment"]["provisioning"]["env_presence"]
    serialized = json.dumps(summary)

    assert env_presence["BLAND_PHONE_NUMBER"] is True
    assert env_presence["BLAND_API_KEY"] is True
    assert "+15551234567" not in serialized
    assert "bland-secret-token" not in serialized


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


def test_plan_run_cli_package_audit_writes_consistency_artifacts(tmp_path):
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
            "--package-audit",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["package_audit"]["ok"] is True
    assert payload["package_audit"]["status"] == "pass"
    assert payload["package_audit"]["issues"] == []
    assert payload["package_audit"]["checked_artifact_count"] == 97
    assert Path(payload["package_audit"]["artifacts"]["json"]).exists()
    assert Path(payload["package_audit"]["artifacts"]["markdown"]).exists()
    assert str(artifact_root / "voiceops-package-audit" / "current") in payload["package_audit"]["artifacts"]["json"]
    plan_run = json.loads((output_dir / "voiceops-plan-run.json").read_text(encoding="utf-8"))
    assert plan_run["package_audit"] == payload["package_audit"]
    plan_markdown = (output_dir / "voiceops-plan-run.md").read_text(encoding="utf-8")
    assert "Package audit: pass" in plan_markdown
    assert "Package audit issues: none" in plan_markdown


def test_plan_run_cli_package_audit_accepts_hosted_model_fallback_package(tmp_path):
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
            "--active-model",
            "Nemotron 3 Super via hosted provider",
            "--reflex-model",
            "Moshi fast reflex",
            "--interpreter-model",
            "Gemma 4 E4B audio-native interpreter",
            "--package-audit",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    demo = json.loads(
        (artifact_root / "hackathon-voiceops-demo" / "current" / "voiceops-demo.json").read_text(encoding="utf-8")
    )
    plan_run = json.loads((output_dir / "voiceops-plan-run.json").read_text(encoding="utf-8"))
    dashboard = (artifact_root / "hackathon-voiceops-demo" / "current" / "operator-dashboard.html").read_text(
        encoding="utf-8"
    )

    assert payload["ok"] is True
    assert payload["package_audit"]["ok"] is True
    assert payload["package_audit"]["issues"] == []
    assert plan_run["plan_args"] == {
        "active_model": "Nemotron 3 Super via hosted provider",
        "reflex_model": "Moshi fast reflex",
        "interpreter_model": "Gemma 4 E4B audio-native interpreter",
    }
    assert "--active-model 'Nemotron 3 Super via hosted provider'" in plan_run["closure_index"]["operator_handoff"][
        "final_reindex_command"
    ]
    assert demo["sponsor_stack"]["hermes_active_model"]["path"] == "hosted_nemotron_3_super_fallback"
    assert demo["spark_stack"]["current_path_local"] is False
    assert demo["spark_stack"]["reflex"]["model"] == "Moshi fast reflex"
    assert demo["spark_stack"]["interpreter"]["model"] == "Gemma 4 E4B audio-native interpreter"
    assert "Hosted fallback selected, Spark-local evidence pending" in dashboard
    assert "Spark target selected, live evidence pending" not in dashboard


def test_plan_run_cli_dry_audit_does_not_write_requested_artifacts(tmp_path):
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
            "--dry-audit",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["ok_meaning"] == "no hard validation failures; not a readiness claim"
    assert payload["readiness_ok"] is False
    assert payload["dry_audit"] is True
    assert payload["persistent_writes"] is False
    assert payload["temporary_artifacts_removed_on_exit"] is True
    assert payload["requested_artifact_root"] == str(artifact_root)
    assert payload["requested_output_dir"] == str(output_dir)
    assert payload["remaining_gates"] == [
        "live_discord_voice_operator",
        "spend_and_provisioning_preflight",
        "local_spark_stack_matrix",
    ]
    assert [action["gate_id"] for action in payload["next_actions"]] == payload["remaining_gates"]
    assert [action["phase_id"] for action in payload["next_actions"]] == [
        "live_discord_voice",
        "spend_and_provisioning_preflight",
        "local_spark_stack",
    ]
    assert [action["phase_id"] for action in payload["review_actions"]] == ["multi_channel_policy_review"]
    assert payload["review_actions"][0]["status"] == "pending_human_review"
    assert payload["review_actions"][0]["real_egress_enabled"] is False
    assert "--audit-only" in payload["next_actions"][0]["first_safe_command"]
    assert payload["next_actions"][0]["first_evidence_command"].startswith(
        "uv run python -m hermes_cli.realtime_voice_live_evidence"
    )
    assert "--run-doctor-report" in payload["next_actions"][0]["first_evidence_command"]
    assert payload["next_actions"][0]["closure_plan"].endswith("live-probe-closure-plan.json")
    assert payload["next_actions"][0]["evidence_scaffold"].endswith("live-voice-evidence-scaffold/manifest.json")
    assert "--validate-live-evidence" in payload["next_actions"][0]["local_validation_command"]
    assert "validate_live_manifest_offline" in payload["next_actions"][0]["validation_commands"]
    assert payload["next_actions"][0]["expected_artifacts"]
    assert "--dry-audit" in payload["next_actions"][1]["first_safe_command"]
    assert "voiceops_provisioning_probe.py" in payload["next_actions"][1]["first_evidence_command"]
    assert payload["next_actions"][1]["closure_plan"].endswith("setup-closure-plan.json")
    assert payload["next_actions"][1]["evidence_template"].endswith("provisioning-preflight-evidence.template.json")
    assert "--preflight-evidence" in payload["next_actions"][1]["local_validation_command"]
    assert "execute_approved_stripe_actions" not in payload["next_actions"][1]["validation_commands"]
    assert "validate_post_approval_receipts" in payload["next_actions"][1]["validation_commands"]
    assert "--lint-evidence" in payload["next_actions"][2]["first_safe_command"]
    assert payload["next_actions"][2]["first_evidence_command"] == "scripts/dgx_spark_gemma4_voice_eval.sh"
    assert payload["next_actions"][2]["closure_plan"].endswith("spark-matrix-closure-plan.json")
    assert payload["next_actions"][2]["evidence_scaffold"].endswith("spark-benchmark-scaffold/spark-benchmark-evidence.json")
    assert "--lint-evidence" in payload["next_actions"][2]["local_audit_command"]
    assert "lint_evidence" in payload["next_actions"][2]["validation_commands"]
    assert not output_dir.exists()
    assert not artifact_root.exists()


def test_plan_run_cli_dry_audit_can_run_package_audit_without_persistent_writes(tmp_path):
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
            "--dry-audit",
            "--package-audit",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["dry_audit"] is True
    assert payload["persistent_writes"] is False
    assert payload["package_audit"] == {
            "checked_artifact_count": 97,
        "issues": [],
        "ok": True,
        "persistent_writes": False,
        "status": "pass",
    }
    assert not output_dir.exists()
    assert not artifact_root.exists()


def test_plan_run_cli_dry_audit_refuses_probe_execution(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "voiceops_plan_run.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--dry-audit",
            "--run-readonly-discovery",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["ok"] is False
    assert payload["dry_audit"] is True
    assert payload["persistent_writes"] is False
    assert "refuses" in payload["error"]


def test_parse_args_defaults_to_plan_artifact_paths():
    args = parse_args([])

    assert args.artifact_root == Path("artifacts")
    assert args.output_dir == Path("artifacts/voiceops-plan/current")
    assert args.active_model is None
    assert args.reflex_model is None
    assert args.interpreter_model is None
    assert args.voice_live_evidence == []
    assert args.provisioning_preflight_evidence is None
    assert args.timeout_seconds is None
    assert args.readonly_discovery_timeout_seconds is None
    assert args.run_command_probes is False
    assert args.run_readonly_discovery is False
    assert args.dry_audit is False
    assert args.package_audit is False
    assert args.package_audit_output_dir is None


def test_parse_args_accepts_separate_readonly_discovery_timeout():
    args = parse_args(
        [
            "--timeout-seconds",
            "5",
            "--readonly-discovery-timeout-seconds",
            "13",
        ]
    )

    assert args.timeout_seconds == 5
    assert args.readonly_discovery_timeout_seconds == 13


def test_plan_run_provisioning_command_preserves_timeout_overrides(tmp_path):
    artifact_root = tmp_path / "artifacts"
    output_dir = artifact_root / "voiceops-plan" / "current"

    summary = build_plan_run(
        artifact_root=artifact_root,
        output_dir=output_dir,
        env={"PATH": ""},
        timeout_seconds=5,
        readonly_discovery_timeout_seconds=13,
    )

    provisioning_result = next(
        result
        for result in summary["results"]
        if result["milestone"] == "milestone_2_real_spend_and_provisioning_preflight"
    )
    command = provisioning_result["command"]

    assert "--timeout-seconds 5" in command
    assert "--readonly-discovery-timeout-seconds 13" in command
