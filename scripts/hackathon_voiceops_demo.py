#!/usr/bin/env python3
"""Generate a headless VoiceOps hackathon demo package.

The demo is intentionally credential-free by default. It emits the artifacts
needed to record a 1-3 minute submission while keeping live Stripe/Projects
execution behind explicit operator approval.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import shlex
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.voiceops_provisioning_probe import build_milestone2_execution_plan


DEFAULT_REQUEST = (
    "Hermes, I am giving you 200 dollars to use through Stripe Skills. "
    "Provision yourself a VoIP provider account, then call my phone with "
    "this same context so we can continue outside Discord."
)


@dataclass(frozen=True)
class VoiceSurface:
    channel: str
    role: str
    implementation: str
    status: str


@dataclass(frozen=True)
class SpendPolicy:
    name: str
    limit_cents: int
    approval_required_over_cents: int
    currency: str = "usd"


@dataclass(frozen=True)
class OpsAction:
    action_id: str
    provider: str
    command: str
    purpose: str
    estimated_cents: int
    requires_approval: bool
    status: str


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    actor: str
    action: str
    provider: str
    amount_cents: int
    status: str
    evidence: str
    requested_by: str
    proposed_by: str
    budget_policy_ref: str
    command: str
    approval_required: bool
    approval_status: str
    result: str
    receipt_ref: str | None
    credential_location_ref: str | None
    rollback_ref: str | None
    notification_channel: str


@dataclass(frozen=True)
class ReadinessCheck:
    check_id: str
    status: str
    required_for_video: bool
    detail: str
    next_step: str


STATIC_ARTIFACT_REQUIRED_CHECK_IDS = {"nemotron_3_super_spark_or_labeled_hosted_fallback"}
LIVE_PREREQUISITE_CHECK_IDS = {"discord_voice", "stripe_projects_cli", "stripe_link_cli"}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _dollars(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _demo_closure_summary() -> dict[str, Any]:
    gates = [
        {
            "gate_id": "live_discord_voice_operator",
            "status": "needs_live_probe",
            "missing": ["discord_join", "discord_playback", "live_receiver", "production_sidecar", "live_turn"],
            "template_artifact": "live-voice-evidence-template.json",
            "closure_artifact": "live-probe-closure-plan.md",
            "collection_commands": {
                "derive_from_realtime_voice_report": (
                    "uv run python -m hermes_cli.realtime_voice_live_evidence "
                    "--output-dir artifacts/realtime-voice-evidence/live-current "
                    "--from-realtime-voice-report path/to/realtime-voice-report.json"
                ),
                "collect_live_manifest": (
                    "uv run python -m hermes_cli.realtime_voice_live_evidence "
                    "--output-dir artifacts/realtime-voice-evidence/live-current "
                    "--require-live-discord --require-inbound --wait-seconds 5 "
                    "--sidecar-session-evidence artifacts/realtime-voice-evidence/live-current/sidecar-session.json "
                    "--live-turn-evidence artifacts/realtime-voice-evidence/live-current/live-turn.json"
                ),
                "validate_live_manifest_offline": (
                    "uv run python -m hermes_cli.realtime_voice_live_evidence "
                    "--output-dir artifacts/realtime-voice-evidence/live-current "
                    "--validate-live-evidence "
                    "--discord-live-probe-evidence artifacts/realtime-voice-evidence/live-current/discord-live-probe.json "
                    "--sidecar-session-evidence artifacts/realtime-voice-evidence/live-current/sidecar-session.json "
                    "--live-turn-evidence artifacts/realtime-voice-evidence/live-current/live-turn.json"
                ),
                "ingest_live_manifest": (
                    "uv run python scripts/voiceops_voice_operator.py "
                    "--output-dir artifacts/voiceops-voice-operator/current "
                    "--live-evidence artifacts/realtime-voice-evidence/live-current/manifest.json"
                ),
            },
            "expected_artifacts": [
                "artifacts/realtime-voice-evidence/live-current/manifest.json",
                "artifacts/realtime-voice-evidence/live-current/discord-live-probe.json",
                "artifacts/realtime-voice-evidence/live-current/sidecar-session.json",
                "artifacts/realtime-voice-evidence/live-current/live-turn.json",
                "artifacts/realtime-voice-evidence/live-current/sidecar-session.from-realtime-report.json",
                "artifacts/realtime-voice-evidence/live-current/live-turn.from-realtime-report.json",
                "artifacts/realtime-voice-evidence/live-current/realtime-voice-report-validation.json",
                "artifacts/realtime-voice-evidence/live-current/live-evidence-validation.json",
                "artifacts/voiceops-voice-operator/current/live-voice-evidence-scaffold/manifest.json",
            ],
            "completion_signal": "live_probe_missing_gates becomes [] and live_probe_status is live_evidence_supplied_not_readiness_claim",
            "evidence_contract": {
                "manifest_schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "strict_validation_schema_version": "voiceops.realtime_voice_live_evidence_validation.v1",
                "expanded_evidence_schema_version": "voiceops.milestone1.live_voice_evidence.v1",
                "required_sections": ["discord_live_probe", "sidecar_session", "live_turn"],
                "required_section_refs": ["source_artifact", "section"],
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
                "realtime_voice_report_derivation_schema_version": "voiceops.realtime_voice_report_derivation.v1",
                "doctor_report_derivation_overclaims_production": False,
            },
            "rerun_command": (
                "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                "--output-dir artifacts/voiceops-plan/current "
                "--voice-live-evidence artifacts/realtime-voice-evidence/live-current/manifest.json"
            ),
        },
        {
            "gate_id": "spend_and_provisioning_preflight",
            "status": "needs_setup",
            "missing": [
                "stripe_cli",
                "stripe_projects_cli",
                "stripe_link_cli",
                "mpp_agent",
                "phone_target",
                "phone_provider",
                "stripe_projects_account",
                "stripe_link_approval_capability",
                "mpp_approval_boundary",
                "phone_provider_account",
                "credential_location_reference",
                "rollback_owner_refs",
            ],
            "template_artifact": "provisioning-preflight-evidence.template.json",
            "closure_artifact": "setup-closure-plan.md",
            "collection_commands": {
                "presence_only": (
                    "uv run python scripts/voiceops_provisioning_probe.py "
                    "--output-dir artifacts/voiceops-provisioning/current --env-file .env"
                ),
                "bounded_version_help": (
                    "uv run python scripts/voiceops_provisioning_probe.py "
                    "--output-dir artifacts/voiceops-provisioning/current --env-file .env --run-command-probes"
                ),
                "read_only_discovery": (
                    "uv run python scripts/voiceops_provisioning_probe.py "
                    "--output-dir artifacts/voiceops-provisioning/current --env-file .env --run-readonly-discovery"
                ),
                "ingest_preflight_evidence": (
                    "uv run python scripts/voiceops_provisioning_probe.py "
                    "--output-dir artifacts/voiceops-provisioning/current --env-file .env "
                    "--preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-evidence.json"
                ),
                "ingest_preflight_manifest": (
                    "uv run python scripts/voiceops_provisioning_probe.py "
                    "--output-dir artifacts/voiceops-provisioning/current --env-file .env "
                    "--preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json"
                ),
                "refresh_preflight_source_hashes": (
                    "uv run python scripts/voiceops_provisioning_probe.py "
                    "--refresh-preflight-source-hashes "
                    "artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json"
                ),
                "validate_post_approval_receipts": (
                    "uv run python scripts/voiceops_provisioning_probe.py "
                    "--output-dir artifacts/voiceops-provisioning/current --env-file .env "
                    "--post-approval-receipts artifacts/voiceops-provisioning/current/post-approval-receipts.json"
                ),
            },
            "expected_artifacts": [
                "artifacts/voiceops-provisioning/current/read-only-discovery.json",
                "artifacts/voiceops-provisioning/current/read-only-discovery.md",
                "artifacts/voiceops-provisioning/current/read-only-discovery.manifest.json",
                "artifacts/voiceops-provisioning/current/audit-ledger.read-only-discovery.jsonl",
                "artifacts/voiceops-provisioning/current/provisioning-preflight-evidence.json",
                "artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json",
                "artifacts/voiceops-provisioning/current/post-approval-receipts.template.json",
                "artifacts/voiceops-provisioning/current/post-approval-receipts.example.json",
                "artifacts/voiceops-provisioning/current/post-approval-receipts-scaffold/post-approval-receipts.json",
                "artifacts/voiceops-provisioning/current/post-approval-receipts.json",
                "artifacts/voiceops-provisioning/current/post-approval-receipts.validation.json",
                "artifacts/voiceops-provisioning/current/audit-ledger.post-approval.jsonl",
                "artifacts/voiceops-provisioning/current/provisioning-readiness.json",
            ],
            "completion_signal": (
                "required_failures becomes []; read_only_discovery_status is pass; milestone status becomes ready; if post-approval receipts are "
                "supplied, post_approval_receipts_status is valid, post_approval_receipts_validation_issues is [], "
                "receipt_count covers all expected approval-required actions, and audit-ledger.post-approval.jsonl is populated"
            ),
            "evidence_contract": {
                "preflight_schema_version": "voiceops.milestone2.preflight_evidence.v1",
                "manifest_schema_version": "voiceops.milestone2.preflight_evidence_manifest.v1",
                "required_sections": ["stripe_projects", "stripe_link", "mpp", "phone_handoff", "rollback"],
                "required_section_field": "source_artifact",
                "required_section_provenance_fields": [
                    "source_artifact_kind",
                    "source_artifact_sha256",
                    "source_artifact_redacted_at",
                ],
                "source_artifact_kind": "redacted_setup_evidence",
                "source_artifacts_must_exist": True,
                "source_artifact_sha256_must_match": True,
                "source_artifacts_must_be_redacted_json": True,
                "source_artifact_resolution": "absolute paths or paths relative to the supplied evidence/manifest file",
                "manifest_report_resolution": "absolute paths or paths relative to the supplied manifest file; process cwd is never used",
                "example_only_accepted": False,
                "secret_like_values_accepted": False,
                "full_phone_numbers_accepted": False,
                "read_only_discovery_schema_version": "voiceops.milestone2.read_only_discovery.v1",
                "read_only_discovery_grants_approval": False,
                "read_only_discovery_required_for_live_provisioning_approval": True,
                "read_only_discovery_required_status": "pass",
                "read_only_discovery_auth_context": "isolated_home",
                "read_only_discovery_proves_existing_local_auth": False,
                "post_approval_receipts_schema_version": "voiceops.milestone2.post_approval_receipts.v1",
                "post_approval_linkage_ids_must_be_unique": [
                    "credential_locations[].credential_ref_id",
                    "rollback_receipts[].rollback_ref",
                    "audit_events[].audit_event_id",
                ],
            },
            "rerun_commands": {
                "plan_index_dry_audit": (
                    "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                    "--output-dir artifacts/voiceops-plan/current --dry-audit"
                ),
                "plan_index_command_probes": (
                    "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                    "--output-dir artifacts/voiceops-plan/current --env-file .env --run-command-probes"
                ),
                "plan_index_read_only_discovery": (
                    "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                    "--output-dir artifacts/voiceops-plan/current --env-file .env --run-readonly-discovery"
                ),
                "plan_index_manifest_and_post_approval_receipts": (
                    "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                    "--output-dir artifacts/voiceops-plan/current --env-file .env "
                    "--provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json "
                    "--post-approval-receipts artifacts/voiceops-provisioning/current/post-approval-receipts.json"
                ),
            },
            "rerun_command": (
                "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                "--output-dir artifacts/voiceops-plan/current --env-file .env "
                "--provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json"
            ),
        },
        {
            "gate_id": "local_spark_stack_matrix",
            "status": "needs_evidence",
            "missing": [
                "reflex:needs_evidence",
                "oracle:needs_evidence",
                "asr:needs_evidence",
                "tts:needs_evidence",
                "all_local_stack_smoke:needs_evidence",
            ],
            "template_artifact": "spark-benchmark-evidence-template.json",
            "closure_artifact": "spark-matrix-closure-plan.md",
            "collection_commands": {
                "matrix_only": (
                    "uv run python scripts/voiceops_spark_matrix.py "
                    "--output-dir artifacts/voiceops-spark-matrix/current"
                ),
                "with_evidence": (
                    "uv run python scripts/voiceops_spark_matrix.py "
                    "--output-dir artifacts/voiceops-spark-matrix/current "
                    "--evidence path/to/spark-benchmark-evidence.json"
                ),
                "plan_index": (
                    "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                    "--output-dir artifacts/voiceops-plan/current "
                    "--evidence path/to/spark-benchmark-evidence.json"
                ),
                "dgx_eval": "scripts/dgx_spark_gemma4_voice_eval.sh",
            },
            "expected_artifacts": [
                "artifacts/dgx-spark-gemma4-voice-eval/current/kame-stack",
                "artifacts/voiceops-spark-matrix/current/spark-benchmark-scaffold/spark-benchmark-evidence.json",
                "artifacts/voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/asr-nemotron-speech-raw.json",
                "artifacts/voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/tts-magpie-local-raw.json",
                "artifacts/voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/all-local-stack-smoke-raw.json",
                "artifacts/voiceops-spark-matrix/current/spark-operator-runbook.md",
                "path/to/spark-benchmark-evidence.json",
                "artifacts/voiceops-spark-matrix/current/spark-model-matrix.json",
                "artifacts/voiceops-spark-matrix/current/spark-matrix-closure-plan.md",
            ],
            "completion_signal": "ready_for_one_spark_demo is true, role_status values are validated, and all_local_stack_smoke is validated",
            "evidence_contract": {
                "benchmark_schema_version": "voiceops.spark_benchmark_evidence.v1",
                "required_locality_for_one_spark": "local_spark",
                "required_hardware": "1x NVIDIA DGX Spark",
                "required_oracle_selection": "Hermes /model",
                "required_oracle_authority_routes": ["tools", "files", "memory", "project_context"],
                "preferred_local_oracle_candidate_id": "oracle-nemotron3-super-local",
                "preferred_local_oracle_model": "Nemotron 3 Super",
                "non_counting_fallback_oracle_models": ["Nemotron 3 Ultra"],
                "required_stack_components": ["reflex", "oracle", "asr", "tts", "sidecar"],
                "source_artifacts_must_exist": True,
                "source_artifact_resolution": "absolute paths or paths relative to the supplied benchmark evidence file",
                "source_artifact_readable": True,
                "source_artifact_sha256_must_match": True,
                "hosted_fallback_counts_for_one_spark_readiness": False,
                "example_only_accepted": False,
                "scaffold_is_example_only": True,
                "loopback_smoke_bridge_counts_for_local_speech_readiness": False,
                "local_speech_requires_production_provider": True,
            },
            "rerun_command": (
                "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                "--output-dir artifacts/voiceops-plan/current "
                "--evidence path/to/spark-benchmark-evidence.json"
            ),
        },
    ]
    return {
        "schema_version": "voiceops.demo_closure_summary.v1",
        "artifact_id": "voiceops-demo-readiness-closure-summary",
        "source_demo_artifact": "voiceops-demo.json",
        "source_readiness_artifact": "readiness-report.json",
        "closure_status": "needs_external_evidence",
        "readiness_closure_ref": "artifacts/voiceops-plan/current/readiness-closure-index.json",
        "gates": gates,
    }


def _closure_gate_markdown_lines(closure: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for gate in closure["gates"]:
        lines.extend(
            [
                f"### {gate['gate_id']}",
                "",
                f"- Status: {gate['status']}",
                f"- Missing: {', '.join(gate['missing'])}",
                f"- Template: `{gate['template_artifact']}`",
                f"- Closure artifact: `{gate['closure_artifact']}`",
                f"- Completion signal: {gate['completion_signal']}",
            ]
        )
        contract = gate.get("evidence_contract")
        if isinstance(contract, dict):
            lines.append("- Evidence contract:")
            for key, value in sorted(contract.items()):
                lines.append(f"  - `{key}`: `{value}`")
        commands = gate.get("collection_commands")
        if isinstance(commands, dict):
            lines.append("- Collection commands:")
            for label, command in sorted(commands.items()):
                lines.append(f"  - `{label}`: `{command}`")
        expected = gate.get("expected_artifacts")
        if isinstance(expected, list):
            lines.append("- Expected artifacts:")
            lines.extend(f"  - `{artifact}`" for artifact in expected)
        if gate.get("rerun_command"):
            lines.append(f"- Rerun command: `{gate['rerun_command']}`")
        lines.append("")
    return lines


def _readiness_closure_summary_markdown(closure: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Demo Readiness Closure Summary",
        "",
        f"- Artifact ID: {closure['artifact_id']}",
        f"- Schema version: {closure['schema_version']}",
        f"- Closure status: {closure['closure_status']}",
        f"- Plan closure ref: `{closure['readiness_closure_ref']}`",
        "",
        "## Gates",
        "",
    ]
    lines.extend(_closure_gate_markdown_lines(closure))
    return "\n".join(lines)


def _check_status(readiness: dict[str, Any], check_id: str) -> str:
    for check in readiness.get("checks", []):
        if check.get("check_id") == check_id:
            return str(check.get("status") or "unknown")
    return "unknown"


def _operator_handoff_preview(demo: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    closure = _demo_closure_summary()
    gates = {gate["gate_id"]: gate for gate in closure["gates"]}
    discord_ready = _check_status(readiness, "discord_voice") == "pass"
    provisioning_ready = all(
        _check_status(readiness, check_id) == "pass"
        for check_id in ("stripe_projects_cli", "stripe_link_cli")
    )
    phone_ready = _check_status(readiness, "phone_handoff") == "pass"
    spark_path = demo["sponsor_stack"]["hermes_active_model"]
    live_gate = gates["live_discord_voice_operator"]
    provisioning_gate = gates["spend_and_provisioning_preflight"]
    spark_gate = gates["local_spark_stack_matrix"]
    return {
        "schema_version": "voiceops.operator_handoff_preview.v1",
        "source": "hackathon_voiceops_demo",
        "purpose": "Standalone Milestone 0 operator sequence for closing external evidence without hand-editing artifacts.",
        "changes_readiness_by_itself": False,
        "readiness_closure_ref": "readiness-closure-summary.json",
        "secret_policy": "Use local env/config files and provider CLIs only; never paste secret values into artifacts.",
        "phases": [
            {
                "phase_id": "live_discord_voice",
                "gate_id": live_gate["gate_id"],
                "can_run_here_now": discord_ready,
                "blocked_by_current_package": [] if discord_ready else ["discord_voice"],
                "first_safe_command": live_gate["collection_commands"]["derive_from_realtime_voice_report"],
                "commands": [
                    live_gate["collection_commands"]["derive_from_realtime_voice_report"],
                    live_gate["collection_commands"]["collect_live_manifest"],
                    live_gate["collection_commands"]["validate_live_manifest_offline"],
                    live_gate["collection_commands"]["ingest_live_manifest"],
                    live_gate["rerun_command"],
                ],
                "command_safety": {
                    "derive_from_realtime_voice_report": "local_file_derivation_only_no_discord_network",
                    "collect_live_manifest": "discord_live_probe_requires_config_no_secret_values_in_artifacts",
                    "validate_live_manifest_offline": "local_file_validation_only",
                    "ingest_live_manifest": "local_file_ingest_only",
                    "plan_reindex": "local_reindex_only",
                },
                "required_inputs": [
                    "Discord bot token and channel config for live collection",
                    "production realtime voice sidecar session evidence",
                    "one real live turn with assistant audio, barge-in, and no voice-capability denial",
                    "optional hermes doctor realtime voice report for partial sidecar/live-turn derivation",
                ],
                "expected_artifacts": live_gate["expected_artifacts"],
                "success_check": live_gate["completion_signal"],
                "must_not": [
                    "claim production readiness from managed_loopback or diagnostic sidecar modes",
                    "include Discord/provider tokens or raw private transcripts in evidence artifacts",
                    "treat silent receiver packets as barge-in proof without speech evidence",
                ],
            },
            {
                "phase_id": "spend_and_provisioning_preflight",
                "gate_id": provisioning_gate["gate_id"],
                "can_run_here_now": provisioning_ready and phone_ready,
                "blocked_by_current_package": [
                    check_id
                    for check_id in ("stripe_projects_cli", "stripe_link_cli", "phone_handoff")
                    if _check_status(readiness, check_id) != "pass"
                ],
                "first_safe_command": provisioning_gate["collection_commands"]["presence_only"],
                "commands": [
                    provisioning_gate["rerun_commands"]["plan_index_dry_audit"],
                    provisioning_gate["collection_commands"]["presence_only"],
                    provisioning_gate["collection_commands"]["bounded_version_help"],
                    provisioning_gate["rerun_commands"]["plan_index_command_probes"],
                    provisioning_gate["collection_commands"]["read_only_discovery"],
                    provisioning_gate["rerun_commands"]["plan_index_read_only_discovery"],
                    provisioning_gate["collection_commands"]["refresh_preflight_source_hashes"],
                    provisioning_gate["collection_commands"]["ingest_preflight_manifest"],
                    provisioning_gate["collection_commands"]["validate_post_approval_receipts"],
                    provisioning_gate["rerun_commands"]["plan_index_manifest_and_post_approval_receipts"],
                ],
                "command_safety": {
                    "plan_index_dry_audit": "no_write_no_network_no_probe_audit",
                    "presence_only": "offline_presence_only",
                    "bounded_version_help": "local_subprocess_only_no_network_intent",
                    "plan_index_command_probes": "local_subprocess_only_no_network_intent",
                    "read_only_discovery": "network_possible_allowlisted_read_only",
                    "plan_index_read_only_discovery": "network_possible_allowlisted_read_only",
                    "refresh_preflight_source_hashes": "local_file_hashing_only",
                    "ingest_preflight_manifest": "local_file_validation_only",
                    "validate_post_approval_receipts": "post_approval_local_validation_only",
                    "plan_index_manifest_and_post_approval_receipts": "local_reindex_only",
                },
                "required_inputs": [
                    "Stripe Projects and Link CLI setup",
                    "NemoClaw/MPP execution-boundary setup",
                    "phone provider and target references",
                    "redacted source artifacts with matching SHA-256 fields",
                    "post-approval receipts only after explicit approval",
                ],
                "expected_artifacts": provisioning_gate["expected_artifacts"],
                "success_check": provisioning_gate["completion_signal"],
                "must_not": [
                    "run Stripe Projects provisioning, Link spend, phone calls, messages, or credential retrieval before approval",
                    "store raw card data, provider tokens, or full phone numbers in artifacts",
                    "treat read-only discovery as approval for live actions",
                ],
            },
            {
                "phase_id": "local_spark_stack",
                "gate_id": spark_gate["gate_id"],
                "can_run_here_now": False,
                "blocked_by_current_package": ["local_spark_stack_matrix"],
                "first_safe_command": spark_gate["collection_commands"]["dgx_eval"],
                "commands": [
                    spark_gate["collection_commands"]["dgx_eval"],
                    spark_gate["collection_commands"]["with_evidence"],
                    spark_gate["collection_commands"]["plan_index"],
                ],
                "command_safety": {
                    "dgx_eval": "requires_dgx_spark_local_benchmark_collection",
                    "with_evidence": "local_benchmark_evidence_validation",
                    "plan_index": "local_reindex_only",
                },
                "required_inputs": [
                    "1x NVIDIA DGX Spark",
                    "Gemma 4 E2B/E4B-style native-audio reflex evidence",
                    "Hermes /model-selected oracle evidence, preferably Nemotron 3 Super",
                    "local ASR/TTS evidence",
                    "all-local stack smoke with KAME routing metrics",
                ],
                "expected_artifacts": [
                    artifact
                    for artifact in spark_gate["expected_artifacts"]
                    if not artifact.endswith("spark-matrix-closure-plan.md")
                ],
                "success_check": spark_gate["completion_signal"],
                "must_not": [
                    "count hosted fallback models as Spark-local readiness proof",
                    "count loopback protocol smoke as local ASR/TTS proof",
                    "introduce a separate VoiceOps oracle_model setting",
                ],
                "active_model_path": spark_path,
            },
        ],
        "final_reindex_command": (
            "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
            "--output-dir artifacts/voiceops-plan/current "
            "--voice-live-evidence artifacts/realtime-voice-evidence/live-current/manifest.json "
            "--env-file .env "
            "--provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json "
            "--post-approval-receipts artifacts/voiceops-provisioning/current/post-approval-receipts.json "
            "--evidence path/to/spark-benchmark-evidence.json"
        ),
        "final_success_signal": "readiness_gaps is [] and closure_status is complete",
    }


def _operator_handoff_preview_markdown(handoff: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Operator Handoff Preview",
        "",
        f"- Schema version: {handoff['schema_version']}",
        f"- Changes readiness by itself: {'yes' if handoff['changes_readiness_by_itself'] else 'no'}",
        f"- Readiness closure ref: `{handoff['readiness_closure_ref']}`",
        f"- Secret policy: {handoff['secret_policy']}",
        "",
        "## Phases",
        "",
    ]
    for phase in handoff["phases"]:
        lines.extend(
            [
                f"### {phase['phase_id']}",
                "",
                f"- Gate: `{phase['gate_id']}`",
                f"- Can run here now: {'yes' if phase['can_run_here_now'] else 'no'}",
                f"- Blocked by current package: {', '.join(phase['blocked_by_current_package']) if phase['blocked_by_current_package'] else 'none'}",
                f"- First safe command: `{phase['first_safe_command']}`",
                f"- Success check: {phase['success_check']}",
                "- Command safety:",
            ]
        )
        for label, safety in sorted(phase["command_safety"].items()):
            lines.append(f"  - `{label}`: {safety}")
        lines.append("- Required inputs:")
        lines.extend(f"  - {item}" for item in phase["required_inputs"])
        lines.append("- Must not:")
        lines.extend(f"  - {item}" for item in phase["must_not"])
        lines.append("- Commands:")
        lines.extend(f"  - `{command}`" for command in phase["commands"])
        lines.append("")
    lines.extend(
        [
            "## Final Reindex",
            "",
            "```bash",
            handoff["final_reindex_command"],
            "```",
            "",
            f"Success signal: {handoff['final_success_signal']}",
            "",
        ]
    )
    return "\n".join(lines)


def _slug(value: str) -> str:
    chars = []
    for ch in value.lower():
        if ch.isalnum():
            chars.append(ch)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-") or "voiceops"


def _env_present(env: Mapping[str, str], key: str) -> bool:
    return bool(str(env.get(key) or "").strip())


def _env_truthy(env: Mapping[str, str], key: str) -> bool:
    return str(env.get(key) or "").strip().lower() in {"1", "true", "yes", "on"}


def _which_any(which: Callable[[str], str | None], commands: Iterable[str]) -> str | None:
    for command in commands:
        path = which(command)
        if path:
            return path
    return None


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return values
    except OSError:
        return values
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _merge_env_sources(
    env: Mapping[str, str],
    env_files: Iterable[Path],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    merged = dict(env)
    sources: list[dict[str, Any]] = [{"kind": "process", "loaded": True, "key_count": len(env)}]
    for path in env_files:
        parsed = _parse_env_file(path)
        exists = path.exists()
        # Mirror Hermes readiness semantics: env files can prove configured
        # state even when the current shell did not export those variables.
        merged.update(parsed)
        sources.append(
            {
                "kind": "env_file",
                "path": str(path),
                "exists": exists,
                "loaded": bool(parsed),
                "key_count": len(parsed),
            }
        )
    return merged, sources


def _default_readiness_env_files(hermes_home: Path | None = None) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    resolved_home = hermes_home or Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    return [repo_root / ".env", resolved_home / ".env"]


def _surface_matrix() -> list[VoiceSurface]:
    return [
        VoiceSurface(
            channel="discord",
            role="primary realtime voice room",
            implementation="Hermes Discord gateway /voice join plus KAME realtime voice sidecar",
            status="intended-live-front-door-needs-evidence",
        ),
        VoiceSurface(
            channel="whatsapp",
            role="mobile household command channel",
            implementation="existing WhatsApp bridge and WhatsApp Cloud setup path",
            status="follow-on-not-configured-in-static-package",
        ),
        VoiceSurface(
            channel="phone",
            role="outbound call handoff with the same operational context",
            implementation="Stripe Projects provisions Twilio or another VoIP provider; Hermes queues the call through the phone bridge",
            status="dry-run-queued",
        ),
    ]


def _active_model_path(active_model: str) -> dict[str, Any]:
    normalized = active_model.lower()
    is_super = "nemotron" in normalized and "super" in normalized
    is_ultra = "nemotron" in normalized and "ultra" in normalized
    if is_super:
        return {
            "active_model": active_model,
            "selected_by": "Hermes /model",
            "path": "spark_local_nemotron_3_super",
            "status": "preferred_local_target_selected_not_validated",
            "recording_ready": True,
            "label": "Nemotron 3 Super on DGX Spark target; benchmark evidence still required",
            "spark_local": True,
            "fallback_used": False,
            "evidence_status": "target_selected_needs_benchmark_evidence",
        }
    if is_ultra:
        return {
            "active_model": active_model,
            "selected_by": "Hermes /model",
            "path": "hosted_nemotron_3_ultra_fallback",
            "status": "hosted_fallback",
            "recording_ready": True,
            "label": "Hosted /model fallback",
            "spark_local": False,
            "fallback_used": True,
            "evidence_status": "hosted_fallback_not_spark_local_evidence",
        }
    return {
        "active_model": active_model,
        "selected_by": "Hermes /model",
        "path": "non_nvidia_fallback",
        "status": "needs_nemotron_selection",
        "recording_ready": False,
        "label": "Switch Hermes to Nemotron 3 Super or label the hosted fallback",
        "spark_local": False,
        "fallback_used": True,
        "evidence_status": "non_nvidia_fallback_not_sponsor_aligned",
    }


def _sponsor_stack(active_model: str) -> dict[str, Any]:
    active_path = _active_model_path(active_model)
    return {
        "hermes_active_model": active_path,
        "nemotron_3_super": {
            "role": "preferred Spark-local NVIDIA oracle target for serious planning and reasoning",
            "selection": active_model if active_path["path"] == "spark_local_nemotron_3_super" else "not selected",
            "note": "Configured through Hermes' normal /model flow; VoiceOps does not introduce a separate oracle_model setting.",
        },
        "nemotron_3_ultra_hosted_fallback": {
            "role": "clearly labeled hosted /model fallback when the local Nemotron 3 Super Spark path is unavailable or still under benchmark",
            "selection": (
                active_model
                if active_path["path"] == "hosted_nemotron_3_ultra_fallback"
                else "Hosted /model fallback available if the local Nemotron 3 Super path is unavailable"
            ),
            "note": "Fallback is still selected through Hermes' normal /model flow, not a VoiceOps-specific model setting.",
        },
        "nemoclaw": {
            "role": "safe execution boundary for agent actions that touch tools, credentials, network, and spend",
            "demo_use": "wrap or present the Stripe/VoIP provisioning plan as a sandboxed execution packet before approval",
        },
        "stripe_skills": {
            "role": "controlled economic rail",
            "skills": ["stripe-projects", "stripe-link-cli", "mpp-agent"],
            "demo_use": "provision VoIP service, request approved spend, and preserve receipts/audit events",
        },
    }


def _spark_stack(active_model: str, reflex_model: str) -> dict[str, Any]:
    return {
        "compute": "1x NVIDIA DGX Spark target",
        "local_first": True,
        "reflex": {
            "model": reflex_model,
            "role": "low-latency KAME interface model for turn handling, intent triage, and floor control",
            "input": "native audio when available; explicit local STT fallback state otherwise",
        },
        "oracle": {
            "model": active_model,
            "selected_by": "Hermes /model",
            "active_model_path": _active_model_path(active_model),
            "role": "Hermes active model selected by /model; no separate oracle_model setting",
            "interface_contract": "receives committed intent, transcript evidence, spend policy, and tool plan",
            "preferred_local_target": "Nemotron 3 Super on DGX Spark",
            "hosted_fallback": "clearly labeled hosted provider through Hermes /model",
        },
        "speech": {
            "asr": "Nemotron Speech or equivalent local streaming ASR for durable transcript evidence",
            "tts": "local Magpie/Riva-style TTS target with Cartesia cloud fallback for the demo",
        },
        "guardrails": [
            "budget caps",
            "approval gates",
            "audit ledger",
            "dry-run by default",
            "live spend only through Stripe Link approval",
        ],
    }


def _kame_reflex_ack_trace() -> dict[str, Any]:
    return {
        "trace_id": "kame-reflex-ack-001",
        "source": "discord_voice",
        "input": "voice_segment_after_vad_endpoint",
        "ack_text": "I heard you. I will keep this under 200 dollars and ask before anything billable runs.",
        "scripted": True,
        "latency_ms": None,
        "evidence_ref": "demo-script.md",
        "live_evidence_required_for_latency_claim": True,
        "status": "scripted_static_ack_until_live_voice_evidence",
    }


def _ops_actions(total_budget_cents: int) -> list[OpsAction]:
    base_actions = [
        OpsAction(
            action_id="grant-spend-budget",
            provider="voiceops-policy",
            command="record spend cap usd:200 approval_required:true",
            purpose="bind the spoken Discord budget to the approval and audit policy",
            estimated_cents=0,
            requires_approval=False,
            status="ready",
        ),
        OpsAction(
            action_id="provision-voip-provider",
            provider="stripe-projects",
            command="stripe projects add twilio/voice",
            purpose="provision a VoIP-capable provider account for outbound calls and SMS fallback",
            estimated_cents=2500,
            requires_approval=True,
            status="queued",
        ),
        OpsAction(
            action_id="buy-service-credit",
            provider="stripe-link-cli",
            command=(
                "link-cli spend-request create --merchant-name ExampleOps "
                "--merchant-url https://example.invalid --amount 4900 --request-approval"
            ),
            purpose="buy a prepaid operations API credit only after Link approval",
            estimated_cents=4900,
            requires_approval=True,
            status="queued",
        ),
        OpsAction(
            action_id="persist-call-context",
            provider="hermes-audit-ledger",
            command="write context packet for outbound phone handoff",
            purpose="preserve the Discord conversation, budget, approval state, and VoIP provisioning result for the phone call",
            estimated_cents=0,
            requires_approval=False,
            status="ready",
        ),
        OpsAction(
            action_id="call-user-phone",
            provider="voiceops-phone-bridge",
            command="queue outbound call --context artifacts/hackathon-voiceops-demo/current/voiceops-demo.json",
            purpose="call the user's phone and continue with the same Discord context",
            estimated_cents=0,
            requires_approval=True,
            status="queued",
        ),
        OpsAction(
            action_id="draft-status",
            provider="hermes-gateway",
            command="draft Discord and WhatsApp status summary with phone-call audit ID",
            purpose="prepare a cross-channel approval packet and call handoff summary without sending it",
            estimated_cents=0,
            requires_approval=False,
            status="ready",
        ),
    ]
    running = 0
    selected: list[OpsAction] = []
    for action in base_actions:
        if running + action.estimated_cents <= total_budget_cents:
            selected.append(action)
            running += action.estimated_cents
        else:
            selected.append(
                OpsAction(
                    action_id=action.action_id,
                    provider=action.provider,
                    command=action.command,
                    purpose=action.purpose,
                    estimated_cents=action.estimated_cents,
                    requires_approval=True,
                    status="held-budget",
                )
            )
    return selected


def _action_ref_slug(action_id: str) -> str:
    return action_id.replace("-", "_")


def _approval_status(action: OpsAction | Mapping[str, Any]) -> str:
    requires_approval = bool(action["requires_approval"] if isinstance(action, Mapping) else action.requires_approval)
    status = str(action["status"] if isinstance(action, Mapping) else action.status)
    if not requires_approval:
        return "not_required"
    if status == "queued":
        return "pending_operator_approval"
    if status == "held-budget":
        return "held_budget"
    return "approval_required"


def _action_result(action: OpsAction | Mapping[str, Any]) -> str:
    status = str(action["status"] if isinstance(action, Mapping) else action.status)
    if status == "ready":
        return "planned_not_executed"
    if status == "queued":
        return "blocked_until_explicit_approval"
    if status == "held-budget":
        return "blocked_by_budget_cap"
    return "recorded_not_executed"


def _rollback_ref(action_id: str) -> str | None:
    mapping = {
        "grant-spend-budget": "audit_requirements.superseding_spend_policy_event",
        "provision-voip-provider": "rollback_plan.deprovision_voip_provider",
        "buy-service-credit": "rollback_plan.refund_or_cancel_service_credit",
        "persist-call-context": "audit_requirements.corrected_context_packet",
        "call-user-phone": "rollback_plan.cancel_or_end_phone_handoff",
        "draft-status": "audit_requirements.redacted_status_correction",
    }
    return mapping.get(action_id)


def _credential_location_ref(action_id: str) -> str | None:
    mapping = {
        "provision-voip-provider": "credential_locations.voip_provider",
        "buy-service-credit": "credential_locations.stripe_link",
        "call-user-phone": "credential_locations.phone_bridge",
    }
    return mapping.get(action_id)


def _receipt_ref(action_id: str) -> str | None:
    if action_id in {"provision-voip-provider", "buy-service-credit", "call-user-phone", "draft-status"}:
        return f"receipts.{_action_ref_slug(action_id)}"
    return None


def _audit_events(actions: Iterable[OpsAction]) -> list[AuditEvent]:
    events: list[AuditEvent] = []
    for index, action in enumerate(actions, start=1):
        events.append(
            AuditEvent(
                event_id=f"evt-{index:03d}",
                actor="hermes-voiceops",
                action=action.action_id,
                provider=action.provider,
                amount_cents=action.estimated_cents,
                status=action.status,
                evidence=f"action:{action.provider}:{action.action_id}",
                requested_by="discord_voice:jetha",
                proposed_by=f"{action.provider}:dry_run_planner",
                budget_policy_ref="spend_policy.household-business-daily-ops",
                command=action.command,
                approval_required=action.requires_approval,
                approval_status=_approval_status(action),
                result=_action_result(action),
                receipt_ref=_receipt_ref(action.action_id),
                credential_location_ref=_credential_location_ref(action.action_id),
                rollback_ref=_rollback_ref(action.action_id),
                notification_channel="discord_text_status",
            )
        )
    return events


def _approval_contract(action: Mapping[str, Any]) -> dict[str, Any]:
    preflight_by_action = {
        "provision-voip-provider": ["stripe_cli", "stripe_projects_cli", "mpp_agent"],
        "buy-service-credit": ["stripe_link_cli", "mpp_agent"],
        "call-user-phone": ["phone_target", "phone_provider", "mpp_agent", "channel_policy"],
    }
    ttl_by_action = {
        "buy-service-credit": 900,
        "call-user-phone": 900,
    }
    command = str(action["command"])
    return {
        "approval_id": f"voiceops-demo-{action['action_id']}",
        "action_id": action["action_id"],
        "approval_channel": "discord_voice_operator_confirmation",
        "approval_artifact": "nemoclaw-action-packet.json",
        "approved_by_ref": None,
        "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
        "required_preflight_gates": preflight_by_action.get(str(action["action_id"]), ["mpp_agent"]),
        "allowed_decisions": ["approve_once", "deny", "hold"],
        "default_decision": "hold",
        "ttl_seconds": ttl_by_action.get(str(action["action_id"]), 1800),
        "status": "pending" if action["status"] == "queued" else "blocked",
    }


def _nemoclaw_action_packet(demo: dict[str, Any]) -> dict[str, Any]:
    approval_actions = [
        {**action, "approval_contract": _approval_contract(action)}
        for action in demo["ops_actions"]
        if action["requires_approval"]
    ]
    return {
        "packet_id": "voiceops-nemoclaw-demo-001",
        "runtime": "NemoClaw",
        "mode": "dry_run_until_user_approval",
        "source_channel": "discord_voice",
        "hermes_active_model": demo["sponsor_stack"]["hermes_active_model"]["active_model"],
        "model_selected_by": "Hermes /model",
        "spend_policy": demo["spend_policy"],
        "allowed_capabilities": [
            "stripe_projects_catalog",
            "stripe_projects_voip_provisioning_after_approval",
            "stripe_link_spend_request_after_approval",
            "phone_call_queue_after_approval",
            "status_summary_draft",
        ],
        "blocked_capabilities": [
            "raw_card_data_in_model_context",
            "unapproved_purchase",
            "unapproved_recurring_charge",
            "unapproved_credential_deletion",
            "unbounded_network_access",
            "discord_or_whatsapp_send_without_channel_policy_approval",
        ],
        "approval_required_actions": approval_actions,
        "approval_contracts": {
            action["action_id"]: action["approval_contract"]
            for action in approval_actions
        },
        "dry_run_commands": [action["command"] for action in approval_actions],
        "audit_event_ids": [event["event_id"] for event in demo["audit_events"]],
    }


def _phone_context_packet(demo: dict[str, Any]) -> dict[str, Any]:
    approval_actions = [action for action in demo["ops_actions"] if action["requires_approval"]]
    return {
        "handoff_id": "voiceops-phone-handoff-001",
        "source_channel": "discord_voice",
        "target_channel": "phone",
        "status": "queued_requires_approval",
        "context_summary": (
            "The user gave Hermes a 200 dollar Stripe Skills budget in Discord voice, "
            "asked Hermes to provision a VoIP provider account, and asked Hermes to "
            "call their phone with the same context."
        ),
        "spoken_opening": (
            "I am continuing from Discord. You gave me a 200 dollar budget to "
            "provision VoIP through Stripe Skills, and I am waiting on your approval "
            "before live spend."
        ),
        "budget": demo["spend_policy"],
        "pending_approvals": [
            {
                "action_id": action["action_id"],
                "provider": action["provider"],
                "estimated_cents": action["estimated_cents"],
                "purpose": action["purpose"],
                "approval_contract": _approval_contract(action),
            }
            for action in approval_actions
        ],
        "audit_event_ids": [event["event_id"] for event in demo["audit_events"]],
    }


def _operator_state_packet(demo: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    phone_context = _phone_context_packet(demo)
    discord_check = next((check for check in readiness["checks"] if check["check_id"] == "discord_voice"), None)
    fallback_reason = (
        discord_check["detail"]
        if discord_check and discord_check["status"] != "pass"
        else "Phone/SMS voice calls remain disabled until provisioning approval and channel policy review pass."
    )
    reserved_cents = demo["totals"]["approval_required_cents"]
    spent_cents = 0
    approved_budget_cents = demo["spend_policy"]["limit_cents"]
    pending_approvals = [
        {
            "approval_id": f"voiceops-demo-{action['action_id']}",
            "action_id": action["action_id"],
            "provider": action["provider"],
            "title": action["purpose"],
            "category": (
                "spend"
                if action["action_id"] == "buy-service-credit"
                else "provisioning"
                if action["action_id"] in {"provision-voip-provider", "call-user-phone"}
                else "handoff"
            ),
            "requester_surface": "discord_voice",
            "risk_level": "medium" if action["estimated_cents"] else "low",
            "budget_impact_cents": action["estimated_cents"],
            "currency": demo["spend_policy"]["currency"],
            "default_decision": "hold_for_operator",
            "status": "pending",
            "ttl_minutes": 15 if action["action_id"] == "buy-service-credit" else 30,
            "artifact_ref": "nemoclaw-action-packet.json",
            "approval_artifact": "nemoclaw-action-packet.json",
            "command": action["command"],
            "approval_contract": _approval_contract(action),
            "execution_status": "not_executed",
            "operator_next_step": "Review the approval contract and required preflight gates before approving or holding.",
        }
        for action in demo["ops_actions"]
        if action["requires_approval"] and action["status"] == "queued"
    ]
    status_by_action = {"ready": "planned", "queued": "approval_required", "held-budget": "blocked"}
    planned_services = [
        {
            "service_id": action["action_id"],
            "display_name": action["purpose"],
            "provider": action["provider"],
            "status": status_by_action.get(action["status"], "planned"),
            "capability": action["command"],
            "external": action["provider"] not in {"hermes-gateway", "hermes-audit-ledger"},
            "approval_required": bool(action["requires_approval"]),
            "notes": "Dry-run demo action; no live provider mutation or spend has executed.",
            "artifact_ref": "voiceops-demo.json",
            "approval_ref": f"voiceops-demo-{action['action_id']}" if action["requires_approval"] else None,
            "execution_status": "not_executed",
            "operator_next_step": "Review the linked dry-run artifact and approval contract before any live action.",
        }
        for action in demo["ops_actions"]
        if action["action_id"] in {"provision-voip-provider", "buy-service-credit", "call-user-phone", "draft-status"}
    ]
    source_audit_events = demo["audit_events"][-12:]
    root_recent_audit_id = source_audit_events[0]["event_id"] if source_audit_events else None
    audit_status_by_action = {"ready": "planned", "queued": "held", "held-budget": "blocked"}
    recent_audit_events = [
        {
            "audit_id": event["event_id"],
            "event_type": event["action"],
            "status": audit_status_by_action.get(event["status"], "recorded"),
            "surface": "discord_voice" if event["action"] in {"grant-spend-budget", "call-user-phone"} else "artifact",
            "summary": event["evidence"],
            "parent_audit_id": None if index == 0 else root_recent_audit_id,
            "amount_cents": event["amount_cents"],
            "approval_status": event["approval_status"],
            "receipt_ref": event["receipt_ref"],
            "credential_location_ref": event["credential_location_ref"],
            "rollback_ref": event["rollback_ref"],
            "artifact_ref": "audit-ledger.jsonl",
            "operator_next_step": "Inspect the linked action packet before changing this audit status.",
        }
        for index, event in enumerate(source_audit_events)
    ]
    return {
        "schema_version": "voiceops.operator_state.v1",
        "artifact_version": "voiceops.operator_state.v1",
        "generated_at": _utc_now(),
        "state_id": "voiceops-demo-operator-state",
        "milestone": "milestone_5_operator_dashboard",
        "artifact_only": True,
        "current_mode": "approval-required",
        "mode": {
            "artifact_only": True,
            "bounded": True,
            "env_secret_reads": False,
            "headless": True,
            "live_spend": False,
            "network_io": False,
            "outbound_calls": False,
            "outbound_sends": False,
            "provisioning": False,
        },
        "bounds": {
            "max_pending_approvals": 8,
            "max_audit_events": 12,
            "max_services_per_section": 8,
            "max_upcoming_tasks": 12,
        },
        "scope": {
            "default_output_dir": "artifacts/hackathon-voiceops-demo/current",
            "blocked_capabilities": [
                "network_probe",
                "environment_secret_read",
                "discord_send",
                "whatsapp_send",
                "sms_send",
                "phone_call",
                "spend",
                "service_provisioning",
            ],
        },
        "active_voice_surface": {
            "surface_id": phone_context["source_channel"],
            "display_name": "Discord voice",
            "status": "active_for_demo" if readiness["ready_for_recording"] else "needs_setup",
            "fallback_surface_id": phone_context["target_channel"],
            "fallback_reason": fallback_reason,
        },
        "budget_status": {
            "currency": demo["spend_policy"]["currency"],
            "current_mode": "approval-required",
            "approved_budget_cents": approved_budget_cents,
            "reserved_cents": reserved_cents,
            "spent_cents": spent_cents,
            "remaining_cents": max(0, approved_budget_cents - reserved_cents - spent_cents),
            "status": "no_live_spend_without_explicit_approval",
            "controls": [
                "dry_run_by_default",
                "approval_packet_required_for_any_spend",
                "provisioning_blocked_until_operator_approval",
            ],
            "approval_required_over_cents": demo["spend_policy"]["approval_required_over_cents"],
            "held_budget_cents": demo["totals"]["held_budget_cents"],
        },
        "pending_approvals": pending_approvals,
        "approval_contracts": {
            approval["action_id"]: approval["approval_contract"]
            for approval in pending_approvals
        },
        "readiness_closure": _demo_closure_summary(),
        "recent_audit_events": recent_audit_events,
        "planned_services": planned_services,
        "provisioned_services": [
            {
                "service_id": "repo_local_demo_artifacts",
                "display_name": "Repo-local VoiceOps demo artifacts",
                "provider": "filesystem",
                "status": "provisioned",
                "capability": "static recording artifacts and operator dashboard",
                "external": False,
                "approval_required": False,
                "notes": "Local artifact directory only; no external service was created.",
                "artifact_ref": "operator-dashboard.html",
                "approval_ref": None,
                "execution_status": "local_artifact_written",
                "operator_next_step": "Use operator-dashboard.html as the local recording and review surface.",
            }
        ],
        "upcoming_tasks": [
            {
                "task_id": "household-budget-review",
                "domain": "household",
                "title": "Review household spending requests before approval",
                "status": "planned",
                "required_surface": "discord_voice",
                "approval_required": False,
                "budget_impact_cents": 0,
            },
            {
                "task_id": "business-phone-handoff",
                "domain": "business",
                "title": "Complete VoIP handoff setup after explicit approval",
                "status": "approval_required",
                "required_surface": "discord_voice",
                "approval_required": True,
                "budget_impact_cents": 0,
            },
        ],
    }


def _demo_milestone2_report(demo: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for check in readiness["checks"]:
        if check["check_id"] in {
            "stripe_projects_cli",
            "stripe_link_cli",
            "nemoclaw_boundary",
            "phone_target",
            "phone_provider",
        }:
            area = {
                "stripe_projects_cli": "stripe_projects",
                "stripe_link_cli": "stripe_link",
                "nemoclaw_boundary": "mpp",
                "phone_target": "phone_handoff",
                "phone_provider": "phone_handoff",
            }[check["check_id"]]
            check_id = {
                "stripe_projects_cli": "stripe_projects_cli",
                "stripe_link_cli": "stripe_link_cli",
                "nemoclaw_boundary": "mpp_agent",
                "phone_target": "phone_target",
                "phone_provider": "phone_provider",
            }[check["check_id"]]
            checks.append(
                {
                    "check_id": check_id,
                    "area": area,
                    "status": "pass" if check["status"] == "pass" else "fail",
                    "required": check["required_for_video"] or check_id in {"phone_target", "phone_provider"},
                    "detail": check["detail"],
                    "next_step": check["next_step"],
                    "evidence": {"source": "readiness-report.json"},
                }
            )
    existing = {check["check_id"] for check in checks}
    if "stripe_cli" not in existing:
        stripe_projects = next(check for check in checks if check["check_id"] == "stripe_projects_cli")
        checks.insert(
            0,
            {
                "check_id": "stripe_cli",
                "area": "stripe_projects",
                "status": stripe_projects["status"],
                "required": True,
                "detail": stripe_projects["detail"],
                "next_step": stripe_projects["next_step"],
                "evidence": {"source": "readiness-report.json"},
            },
        )
    required_failures = [check["check_id"] for check in checks if check["required"] and check["status"] != "pass"]
    return {
        "generated_at": readiness["generated_at"],
        "ready": not required_failures,
        "required_failures": required_failures,
        "checks": checks,
        "probe": {
            "name": "voiceops_demo_milestone2_readiness",
            "non_mutating": True,
            "bounded": True,
            "run_commands": False,
            "run_readonly_discovery": False,
            "active_probe_policy": "version_help_only",
            "read_only_discovery_policy": "exact_allowlist_only",
            "blocked_capabilities": [
                "live_spend",
                "provider_provisioning",
                "credential_retrieval",
                "outbound_phone_calls",
                "account_mutation",
            ],
        },
        "read_only_discovery": {
            "schema_version": "voiceops.milestone2.read_only_discovery.v1",
            "run_requested": False,
            "status": "not_requested",
            "non_mutating": True,
            "does_not_grant_approval": True,
            "redacted_outputs_only": True,
            "network_io_possible": False,
            "failed_probe_ids": [],
            "missing_probe_ids": [],
            "allowlisted_commands": [
                ["link-cli", "auth", "status"],
                ["stripe", "projects", "list", "--limit", "10"],
            ],
            "blocked_capabilities": [
                "live_spend",
                "provider_provisioning",
                "credential_retrieval",
                "outbound_phone_calls",
                "account_mutation",
            ],
            "probes": [],
        },
        "demo_refs": {
            "phone_context": "phone-context.json",
            "nemoclaw_packet": "nemoclaw-action-packet.json",
            "budget_cents": demo["spend_policy"]["limit_cents"],
            "approval_threshold_cents": demo["spend_policy"]["approval_required_over_cents"],
            "queued_cents": demo["totals"]["approval_required_cents"],
            "held_cents": demo["totals"]["held_budget_cents"],
        },
    }


def build_readiness_report(
    demo: dict[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    env_files: Iterable[Path] = (),
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, Any]:
    env, env_sources = _merge_env_sources(os.environ if env is None else env, env_files)
    checks: list[ReadinessCheck] = []

    hermes_path = _which_any(which, ["hermes"])
    checks.append(
        ReadinessCheck(
            check_id="hermes_cli",
            status="pass" if hermes_path else "warn",
            required_for_video=False,
            detail=f"hermes command found at {hermes_path}" if hermes_path else "hermes command not found on PATH",
            next_step="Use this repo with uv for artifacts, or install/point the system hermes command at this branch.",
        )
    )

    discord_ok = _env_present(env, "DISCORD_BOT_TOKEN") and (
        _env_present(env, "DISCORD_VOICE_CHANNEL_ID") or _env_present(env, "DISCORD_VOICE_CHANNEL_NAME")
    )
    checks.append(
        ReadinessCheck(
            check_id="discord_voice",
            status="pass" if discord_ok else "fail",
            required_for_video=True,
            detail=(
                "DISCORD_BOT_TOKEN and a voice channel selector are present"
                if discord_ok
                else "missing DISCORD_BOT_TOKEN or DISCORD_VOICE_CHANNEL_ID/DISCORD_VOICE_CHANNEL_NAME"
            ),
            next_step="Set Discord gateway env, restart Hermes gateway, then use /voice join in the recording server.",
        )
    )

    checks.append(
        ReadinessCheck(
            check_id="nemotron_3_super_spark_or_labeled_hosted_fallback",
            status="pass" if demo["sponsor_stack"]["hermes_active_model"]["recording_ready"] else "fail",
            required_for_video=True,
            detail=(
                f"Hermes active model path is {demo['sponsor_stack']['hermes_active_model']['label']}: "
                f"{demo['sponsor_stack']['hermes_active_model']['active_model']}. "
                "Spark-local target selected; benchmark validation is tracked separately."
                if demo["sponsor_stack"]["hermes_active_model"]["spark_local"]
                else "Hosted fallback selected; this does not count as Spark-local readiness proof."
            ),
            next_step="Before recording, prefer /model to Nemotron 3 Super on Spark. Use a hosted model only as a clearly labeled fallback if Super is unavailable.",
        )
    )

    nemoclaw_path = _which_any(which, ["nemoclaw", "openshell"])
    checks.append(
        ReadinessCheck(
            check_id="nemoclaw_boundary",
            status="pass" if nemoclaw_path else "warn",
            required_for_video=False,
            detail=(
                f"NemoClaw/OpenShell command found at {nemoclaw_path}"
                if nemoclaw_path
                else "no nemoclaw or openshell command found; generated packet still demonstrates the policy boundary"
            ),
            next_step="If available, record the action packet inside NemoClaw/OpenShell; otherwise show nemoclaw-action-packet.json as the approval boundary.",
        )
    )

    stripe_path = _which_any(which, ["stripe"])
    stripe_projects_verified = _env_truthy(env, "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED")
    checks.append(
        ReadinessCheck(
            check_id="stripe_projects_cli",
            status="pass" if stripe_path and stripe_projects_verified else "fail",
            required_for_video=True,
            detail=(
                f"stripe CLI found at {stripe_path}; Projects help verification marker is present"
                if stripe_path and stripe_projects_verified
                else "stripe CLI found, but VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED is not set"
                if stripe_path
                else "stripe CLI not found"
            ),
            next_step=(
                "Run `stripe projects --help` or the provisioning preflight command, then set "
                "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED=true before recording this as ready."
            ),
        )
    )

    link_path = _which_any(which, ["link-cli"])
    checks.append(
        ReadinessCheck(
            check_id="stripe_link_cli",
            status="pass" if link_path else "fail",
            required_for_video=True,
            detail=(
                f"link-cli found at {link_path}"
                if link_path
                else "link-cli not found on PATH; npx is not treated as ready because it may fetch packages"
            ),
            next_step="Install a pinned @stripe/link-cli binary, then authenticate Link before any live spend.",
        )
    )

    whatsapp_ready = _env_truthy(env, "WHATSAPP_ENABLED") or (
        _env_present(env, "WHATSAPP_CLOUD_PHONE_NUMBER_ID") and _env_present(env, "WHATSAPP_CLOUD_ACCESS_TOKEN")
    )
    checks.append(
        ReadinessCheck(
            check_id="whatsapp_followup",
            status="pass" if whatsapp_ready else "warn",
            required_for_video=False,
            detail=(
                "WhatsApp env indicates a configured bridge or Cloud API path"
                if whatsapp_ready
                else "WhatsApp is not configured; keep it as a roadmap/follow-on surface in the demo"
            ),
            next_step="Run `hermes whatsapp` or the WhatsApp Cloud setup if mobile follow-up will be shown live.",
        )
    )

    phone_target_ready = _env_present(env, "VOICEOPS_DEMO_PHONE_NUMBER") or _env_present(
        env, "VOICEOPS_PHONE_TARGET_REF"
    )
    checks.append(
        ReadinessCheck(
            check_id="phone_target",
            status="pass" if phone_target_ready else "warn",
            required_for_video=False,
            detail=(
                "phone target reference is present"
                if phone_target_ready
                else "no VOICEOPS_DEMO_PHONE_NUMBER or VOICEOPS_PHONE_TARGET_REF; generated phone-context.json remains dry-run evidence"
            ),
            next_step="Set VOICEOPS_DEMO_PHONE_NUMBER or a redacted target reference before attempting a live call.",
        )
    )

    phone_provider_ready = _env_present(env, "TWILIO_ACCOUNT_SID") or _env_present(
        env, "VOICEOPS_PHONE_PROVIDER_ACCOUNT_REF"
    )
    checks.append(
        ReadinessCheck(
            check_id="phone_provider",
            status="pass" if phone_provider_ready else "warn",
            required_for_video=False,
            detail=(
                "phone provider account reference is present"
                if phone_provider_ready
                else "no TWILIO_ACCOUNT_SID or VOICEOPS_PHONE_PROVIDER_ACCOUNT_REF; provider readiness must come from provisioning evidence"
            ),
            next_step="Complete approved VoIP provider setup and record a redacted provider account reference before live phone handoff.",
        )
    )

    checks.append(
        ReadinessCheck(
            check_id="phone_handoff",
            status="pass" if phone_target_ready and phone_provider_ready else "warn",
            required_for_video=False,
            detail=(
                "phone target and provider account references are both present"
                if phone_target_ready and phone_provider_ready
                else "phone handoff is not live-ready until both target and provider account references are present"
            ),
            next_step="Treat phone-context.json as dry-run until both target and provider evidence are available.",
        )
    )

    check_dicts = [asdict(check) for check in checks]
    all_required_failures = [
        check["check_id"] for check in check_dicts if check["required_for_video"] and check["status"] != "pass"
    ]
    artifact_required_failures = [
        check["check_id"]
        for check in check_dicts
        if check["check_id"] in STATIC_ARTIFACT_REQUIRED_CHECK_IDS and check["status"] != "pass"
    ]
    live_prerequisite_failures = [
        check["check_id"]
        for check in check_dicts
        if check["check_id"] in LIVE_PREREQUISITE_CHECK_IDS and check["status"] != "pass"
    ]
    static_recording_ready = not artifact_required_failures
    active_path = demo["sponsor_stack"]["hermes_active_model"]
    spark_local_evidence_status = (
        "target_selected_needs_benchmark_evidence"
        if active_path["spark_local"]
        else "hosted_or_nonlocal_path_not_spark_evidence"
    )
    return {
        "generated_at": _utc_now(),
        "schema_version": "voiceops.recording_readiness_report.v1",
        "artifact_id": "voiceops-recording-readiness-report",
        "source_demo_artifact": "voiceops-demo.json",
        "readiness_closure_summary_ref": "readiness-closure-summary.json",
        "static_recording_ready": static_recording_ready,
        "ready_for_recording": static_recording_ready,
        "ready_for_recording_scope": "static_artifact_recording_only",
        "live_demo_ready": False,
        "artifact_required_failures": artifact_required_failures,
        "live_prerequisite_failures": live_prerequisite_failures,
        "all_required_check_failures": all_required_failures,
        "live_demo_missing_evidence": [
            "live_discord_voice_operator",
            "spend_and_provisioning_preflight",
            "local_spark_stack_matrix",
        ],
        "spark_local_evidence_status": spark_local_evidence_status,
        "required_failures": artifact_required_failures,
        "env_sources": env_sources,
        "checks": check_dicts,
    }


def build_demo(args: argparse.Namespace) -> dict[str, Any]:
    actions = _ops_actions(args.budget_cents)
    approval_total = sum(action.estimated_cents for action in actions if action.requires_approval and action.status == "queued")
    ready_total = sum(action.estimated_cents for action in actions if action.status in {"queued", "ready"})
    policy = SpendPolicy(
        name="household-business-daily-ops",
        limit_cents=args.budget_cents,
        approval_required_over_cents=args.approval_required_over_cents,
    )
    return {
        "generated_at": _utc_now(),
        "schema_version": "voiceops.demo_package.v1",
        "artifact_id": "voiceops-demo",
        "demo": {
            "name": args.demo_name,
            "request": args.request,
            "operator": "Hermes VoiceOps",
            "submission_theme": "give a Spark-powered Hermes agent spending money over Discord, let it provision VoIP through Stripe Skills, then continue by phone",
        },
        "sponsor_stack": _sponsor_stack(args.active_model),
        "spark_stack": _spark_stack(args.active_model, args.reflex_model),
        "kame_reflex_ack": _kame_reflex_ack_trace(),
        "voice_surfaces": [asdict(surface) for surface in _surface_matrix()],
        "spend_policy": asdict(policy),
        "ops_actions": [asdict(action) for action in actions],
        "audit_events": [asdict(event) for event in _audit_events(actions)],
        "totals": {
            "ready_or_queued_cents": ready_total,
            "approval_required_cents": approval_total,
            "held_budget_cents": sum(action.estimated_cents for action in actions if action.status == "held-budget"),
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, events: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8")


def _markdown(demo: dict[str, Any]) -> str:
    totals = demo["totals"]
    lines = [
        f"# {demo['demo']['name']}",
        "",
        "## One-line pitch",
        "",
        "Hermes VoiceOps turns a DGX Spark into a local-first operator for a household and business, with Discord voice as the intended live front door and WhatsApp/phone as approval-gated follow-on paths.",
        "",
        "## Sponsor stack",
        "",
        f"- Nemotron 3 Super: {demo['sponsor_stack']['nemotron_3_super']['role']}",
        f"- Hosted fallback: {demo['sponsor_stack']['nemotron_3_ultra_hosted_fallback']['role']}",
        f"- NemoClaw: {demo['sponsor_stack']['nemoclaw']['role']}",
        f"- Stripe Skills: {demo['sponsor_stack']['stripe_skills']['demo_use']}",
        "",
        "## Demo request",
        "",
        f"> {demo['demo']['request']}",
        "",
        "## Spark stack",
        "",
        f"- Compute: {demo['spark_stack']['compute']}",
        f"- Reflex: {demo['spark_stack']['reflex']['model']} for low-latency KAME interface behavior",
        f"- Oracle: {demo['spark_stack']['oracle']['model']} selected through Hermes' normal active model flow",
        f"- Speech: {demo['spark_stack']['speech']['asr']} plus {demo['spark_stack']['speech']['tts']}",
        "",
        "## Voice surfaces",
        "",
    ]
    for surface in demo["voice_surfaces"]:
        lines.append(f"- {surface['channel']}: {surface['role']} ({surface['status']})")
    lines.extend([
        "",
        "## Spend controls",
        "",
        f"- Budget: {_dollars(demo['spend_policy']['limit_cents'])}",
        f"- Approval threshold: {_dollars(demo['spend_policy']['approval_required_over_cents'])}",
        f"- Approval-required queued spend: {_dollars(totals['approval_required_cents'])}",
        f"- Held over budget: {_dollars(totals['held_budget_cents'])}",
        "",
        "## Action queue",
        "",
    ])
    for action in demo["ops_actions"]:
        approval = "approval required" if action["requires_approval"] else "no approval needed"
        lines.append(
            f"- {action['action_id']}: {action['provider']} -> {action['status']}, "
            f"{_dollars(action['estimated_cents'])}, {approval}"
        )
    lines.extend([
        "",
        "## Evidence artifacts",
        "",
        "- `nemoclaw-action-packet.json`: sandbox and approval frame for billable/network-capable actions",
        "- `phone-context.json`: outbound phone-call handoff context preserved from Discord",
        "- `readiness-report.json`: local recording prerequisite report",
        "- `milestone2-execution-plan.json`: post-approval execution contract for VoIP provisioning, spend request, call receipt, credential reference, and rollback",
        "- `operator-handoff-preview.json`: ordered safe evidence-collection sequence for closing live Discord, provisioning, and Spark gates",
        "- `operator-dashboard.html`: static recording dashboard for budget, approvals, guardrails, and handoff state",
        "- `operator-state.json`: machine-readable current mode, surface, budget, approval, service, and task state",
        "- `operator-state-events.jsonl`: append-friendly view of recent operator audit events",
        "- `recording-runbook.md`: shot list, fallback plan, and submission checklist",
        "- `submission-writeup.md`: public submission copy for the tweet/thread/form",
        "",
        "## 90-second video beat sheet",
        "",
        "1. Join Discord voice and give Hermes a fixed Stripe Skills budget.",
        "2. Show Hermes producing a KAME reflex acknowledgement immediately, then a Nemotron 3 Super-backed operating plan.",
        "3. Show the NemoClaw/sandboxed action packet before anything billable runs.",
        "4. Show the Stripe/Projects queue for VoIP provisioning and a Link-gated service-credit spend.",
        "5. Show Hermes preserving the Discord context and queuing an outbound phone call.",
        "6. Close by showing the queued phone handoff context; only continue live by phone after approval and call receipt evidence exist.",
        "",
    ])
    return "\n".join(lines)


def _submission_writeup(demo: dict[str, Any]) -> str:
    policy = demo["spend_policy"]
    approval_cents = demo["totals"]["approval_required_cents"]
    closure = _demo_closure_summary()
    lines = [
        "# Hermes VoiceOps Submission Writeup",
        "",
        "## Short Description",
        "",
        "Hermes VoiceOps is a local-first household and business operator targeted at one NVIDIA DGX Spark, with Discord voice as the intended live front door. In this static package, the user gives Hermes a fixed budget through Stripe Skills, Hermes prepares a NemoClaw-format dry-run plan, queues VoIP provisioning through Stripe Projects, and preserves context for a phone handoff.",
        "",
        "## What The Demo Shows",
        "",
        "- Discord voice is the intended live front door; live proof requires the separate Discord evidence gate.",
        f"- {demo['sponsor_stack']['nemotron_3_super']['selection']} is the visible serious planning path.",
        "- Hosted fallback: clearly labeled provider through Hermes /model, only if the local Nemotron 3 Super Spark path is unavailable.",
        "- NemoClaw-format dry-run packet frames billable and network-capable actions before execution.",
        "- Stripe Skills provide the dry-run spend and provisioning queue until preflight and approval receipts exist.",
        f"- The spoken budget becomes a {_dollars(policy['limit_cents'])} spend policy with approval required over {_dollars(policy['approval_required_over_cents'])}.",
        f"- {_dollars(approval_cents)} of queued spend remains approval-gated.",
        "- Phone and WhatsApp are treated as follow-on operational surfaces, not separate assistants; this package does not claim they are configured live.",
        "",
        "## Why It Matters",
        "",
        "The goal is not a generic voice assistant. The goal is an operator that can run real household and small-business workflows while keeping money, credentials, network effects, and external communication behind explicit controls and an audit trail.",
        "",
        "## Spark Strategy",
        "",
        f"- Compute target: {demo['spark_stack']['compute']}",
        f"- Reflex/interface: {demo['spark_stack']['reflex']['model']}",
        f"- Oracle/brain: {demo['spark_stack']['oracle']['model']}",
        f"- Speech path: {demo['spark_stack']['speech']['asr']} plus {demo['spark_stack']['speech']['tts']}",
        "",
        "## Safety Story",
        "",
        "- Dry-run by default.",
        "- Live spend requires user-visible approval.",
        "- Reflex model does not get broad tool or spend authority.",
        "- NemoClaw packet lists blocked capabilities such as raw card data in model context and unapproved purchases.",
        "- Audit ledger records proposed actions, budget impact, approval state, and handoff evidence.",
        "",
        "## Remaining Closure Gates",
        "",
        f"- Closure status: {closure['closure_status']}",
        f"- Closure index: `{closure['readiness_closure_ref']}`",
        *[
            f"- `{gate['gate_id']}`: {gate['status']} (missing: {', '.join(gate['missing'])})"
            for gate in closure["gates"]
        ],
        "",
        "## Demo Prompt",
        "",
        f"> {demo['demo']['request']}",
        "",
        "## Submission Tweet",
        "",
        "Hermes VoiceOps: I give a Spark-powered Hermes agent a budget over Discord voice, it builds a NemoClaw-safe plan, queues Stripe Skills provisioning for VoIP, and preserves context for a phone handoff. Local-first household/business ops, with spend gated by approval. @NousResearch",
        "",
    ]
    return "\n".join(lines)


def _recording_runbook(demo: dict[str, Any], readiness: dict[str, Any]) -> str:
    failures = readiness["required_failures"]
    live_prerequisite_failures = readiness["live_prerequisite_failures"]
    closure = _demo_closure_summary()
    fallback = (
        "Use the static dashboard plus generated dry-run packets; narrate the missing artifact checks directly."
        if failures
        else "Record live Discord voice first, then cut to the dashboard and generated action packets."
        if not live_prerequisite_failures
        else "Record the static dashboard and generated action packets; narrate the missing live prerequisites directly."
    )
    lines = [
        "# VoiceOps Recording Runbook",
        "",
        "## Goal",
        "",
        "Record a 1-3 minute hackathon video showing Hermes as a Spark-powered household and business operator: Discord voice in, Stripe Skills budget and provisioning plan, NemoClaw safety boundary, and phone handoff with preserved context.",
        "",
        "## Regenerate Artifacts",
        "",
        "```bash",
        "uv run python scripts/hackathon_voiceops_demo.py --output-dir artifacts/hackathon-voiceops-demo/current",
        "```",
        "",
        "Open `artifacts/hackathon-voiceops-demo/current/operator-dashboard.html` directly in a browser for the recording surface.",
        "",
        "## Readiness Gate",
        "",
        f"- Static recording ready: {'yes' if readiness['static_recording_ready'] else 'no'}",
        f"- Live demo ready: {'yes' if readiness['live_demo_ready'] else 'no'}",
        f"- Readiness scope: {readiness['ready_for_recording_scope']}",
        f"- Spark-local evidence: {readiness['spark_local_evidence_status']}",
        f"- Missing live evidence: {', '.join(readiness['live_demo_missing_evidence']) if readiness['live_demo_missing_evidence'] else 'none'}",
        (
            "- Live prerequisite failures: "
            f"{', '.join(live_prerequisite_failures) if live_prerequisite_failures else 'none'}"
        ),
        f"- Required failures: {', '.join(failures) if failures else 'none'}",
        f"- Recording fallback: {fallback}",
        f"- Plan closure status: {closure['closure_status']}",
        f"- Closure index: `{closure['readiness_closure_ref']}`",
        "",
        "Do not show terminal panes or files that contain secrets. Do not run live spend or provisioning unless the user explicitly approves it during the recording.",
        "",
        "## Plan Closure Gates",
        "",
        *_closure_gate_markdown_lines(closure),
        "## Spoken Demo Prompt",
        "",
        "Say this in Discord voice:",
        "",
        f"> {demo['demo']['request']}",
        "",
        "## Shot List",
        "",
        "1. Discord voice: join the voice channel and say the prompt above.",
        "2. Reflex response: show Hermes acknowledging the budget immediately and stating that billable actions require approval.",
        "3. Oracle path: show Nemotron 3 Super selected through Hermes' normal `/model` flow or visible in the generated dashboard; label any hosted fallback clearly.",
        "4. NemoClaw boundary: show `nemoclaw-action-packet.json` or the dashboard NemoClaw section before any billable/network-capable action.",
        "5. Stripe Skills: show queued Stripe Projects VoIP provisioning and Link-gated service-credit spend in `stripe-actions-dry-run.sh` or the dashboard approval queue.",
        "6. Phone handoff: show `phone-context.json` and narrate that the same Discord context is preserved for the outbound call.",
        "7. Spark story: show or state that the target appliance is one DGX Spark running the reflex, speech stack, and preferred local model path.",
        "",
        "## If Live Voice Is Not Ready",
        "",
        "Use the generated package as the demo evidence instead of pretending the system is live:",
        "",
        "- `operator-dashboard.html` for the recording surface",
        "- `voiceops-demo.md` for the concise story",
        "- `nemoclaw-action-packet.json` for the approval boundary",
        "- `stripe-actions-dry-run.sh` for non-mutating Stripe/Projects commands",
        "- `phone-context.json` for the handoff context",
        "- `milestone2-execution-plan.json` for receipts, credential references, approval gates, and rollback/deprovision notes",
        "- `operator-handoff-preview.json` for the ordered safe evidence-collection sequence",
        "- `audit-ledger.jsonl` for durable action evidence",
        "",
        "## Submission Checklist",
        "",
        "- Video is between 1 and 3 minutes.",
        "- Video tags `@NousResearch` in the tweet.",
        "- Short writeup mentions DGX Spark, Discord voice, Nemotron 3 Super, labeled hosted fallback if used, NemoClaw, and Stripe Skills.",
        "- Submit the tweet link in the Nous Discord submissions channel.",
        "- Fill out the hackathon Typeform submission.",
        "- Confirm no API keys, phone numbers, tokens, or account secrets are visible.",
        "",
        "## Tweet Draft",
        "",
        "Hermes VoiceOps: I give a Spark-powered Hermes agent a budget over Discord voice, it builds a NemoClaw-safe plan, queues Stripe Skills provisioning for VoIP, and preserves context for a phone handoff. Local-first household/business ops, with spend gated by approval. @NousResearch",
        "",
    ]
    return "\n".join(lines)


def _readiness_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Recording Readiness",
        "",
        f"- Static recording ready: {'yes' if report['static_recording_ready'] else 'no'}",
        f"- Live demo ready: {'yes' if report['live_demo_ready'] else 'no'}",
        f"- Readiness scope: {report['ready_for_recording_scope']}",
        f"- Spark-local evidence: {report['spark_local_evidence_status']}",
        f"- Missing live evidence: {', '.join(report['live_demo_missing_evidence']) if report['live_demo_missing_evidence'] else 'none'}",
        (
            "- Live prerequisite failures: "
            f"{', '.join(report['live_prerequisite_failures']) if report['live_prerequisite_failures'] else 'none'}"
        ),
        f"- Required failures: {', '.join(report['required_failures']) if report['required_failures'] else 'none'}",
        "",
        "## Env Sources",
        "",
    ]
    for source in report.get("env_sources") or []:
        if source.get("kind") == "process":
            lines.append(f"- process env: {source.get('key_count', 0)} keys visible")
        else:
            state = "loaded" if source.get("loaded") else "missing or empty"
            lines.append(f"- {source.get('path')}: {state} ({source.get('key_count', 0)} keys)")
    lines.extend([
        "",
        "## Checks",
        "",
    ])
    for check in report["checks"]:
        required = "required" if check["required_for_video"] else "optional"
        lines.extend(
            [
                f"### {check['check_id']}",
                "",
                f"- Status: {check['status']}",
                f"- Scope: {required}",
                f"- Detail: {check['detail']}",
                f"- Next step: {check['next_step']}",
                "",
            ]
        )
    return "\n".join(lines)


def _status_class(status: Any) -> str:
    normalized = str(status or "").strip().lower().replace("_", "-")
    if normalized in {"pass", "ready", "queued", "demo-call-queued", "implemented-on-branch", "repo-supported"}:
        return "ok"
    if normalized in {
        "warn",
        "held-budget",
        "queued-requires-approval",
        "intended-live-front-door-needs-evidence",
        "follow-on-not-configured-in-static-package",
        "preferred-local-target-selected-not-validated",
        "scripted-static-ack-until-live-voice-evidence",
    }:
        return "warn"
    if normalized in {"fail", "failed"}:
        return "fail"
    return "neutral"


def _dashboard_html(demo: dict[str, Any], readiness: dict[str, Any]) -> str:
    nemoclaw = _nemoclaw_action_packet(demo)
    phone_context = _phone_context_packet(demo)
    operator_state = _operator_state_packet(demo, readiness)
    operator_handoff = _operator_handoff_preview(demo, readiness)
    closure = _demo_closure_summary()
    kame_ack = demo["kame_reflex_ack"]
    budget_status = operator_state["budget_status"]
    voice_surface = operator_state["active_voice_surface"]
    approval_cents = demo["totals"]["approval_required_cents"]
    limit_cents = max(int(demo["spend_policy"]["limit_cents"] or 0), 1)
    approval_percent = min(100, int(round((approval_cents / limit_cents) * 100)))
    readiness_label = "Static ready" if readiness["static_recording_ready"] else "Needs setup"
    held_actions = [action for action in demo["ops_actions"] if action["status"] == "held-budget"]
    pending_rows = []
    for approval in operator_state["pending_approvals"]:
        pending_rows.append(
            "<tr>"
            f"<td>{_h(approval['action_id'])}</td>"
            f"<td>{_h(approval['provider'])}</td>"
            f"<td>{_h(_dollars(approval['budget_impact_cents']))}</td>"
            f"<td>{_h(approval['title'])}</td>"
            "</tr>"
        )
    if not pending_rows:
        pending_rows.append("<tr><td colspan=\"4\">No queued approvals.</td></tr>")
    action_rows = []
    for action in demo["ops_actions"]:
        approval = "approval required" if action["requires_approval"] else "no approval"
        action_rows.append(
            "<tr>"
            f"<td>{_h(action['action_id'])}</td>"
            f"<td>{_h(action['provider'])}</td>"
            f"<td><span class=\"pill {_status_class(action['status'])}\">{_h(action['status'])}</span></td>"
            f"<td>{_h(_dollars(action['estimated_cents']))}</td>"
            f"<td>{_h(approval)}</td>"
            "</tr>"
        )
    audit_rows = []
    for event in operator_state["recent_audit_events"]:
        audit_rows.append(
            "<tr>"
            f"<td>{_h(event['audit_id'])}</td>"
            f"<td>{_h(event['event_type'])}</td>"
            f"<td><span class=\"pill {_status_class(event['status'])}\">{_h(event['status'])}</span></td>"
            f"<td>{_h(_dollars(event['amount_cents']))}</td>"
            f"<td>{_h(event['summary'])}</td>"
            "</tr>"
        )
    planned_service_rows = []
    for service in operator_state["planned_services"]:
        planned_service_rows.append(
            "<tr>"
            f"<td>{_h(service['service_id'])}</td>"
            f"<td>{_h(service['provider'])}</td>"
            f"<td><span class=\"pill {_status_class(service['status'])}\">{_h(service['status'])}</span></td>"
            f"<td>{_h(service['display_name'])}</td>"
            "</tr>"
        )
    provisioned_service_rows = []
    for service in operator_state["provisioned_services"]:
        provisioned_service_rows.append(
            "<tr>"
            f"<td>{_h(service['service_id'])}</td>"
            f"<td>{_h(service['provider'])}</td>"
            f"<td><span class=\"pill {_status_class(service['status'])}\">{_h(service['status'])}</span></td>"
            f"<td>{_h(service['display_name'])}</td>"
            "</tr>"
        )
    readiness_items = []
    for check in readiness["checks"]:
        required = "required" if check["required_for_video"] else "optional"
        readiness_items.append(
            "<li>"
            f"<span class=\"pill {_status_class(check['status'])}\">{_h(check['status'])}</span>"
            f"<strong>{_h(check['check_id'])}</strong>"
            f"<small>{_h(required)} - {_h(check['detail'])}</small>"
            "</li>"
        )
    closure_items = []
    for gate in closure["gates"]:
        command_labels = ", ".join(sorted((gate.get("collection_commands") or {}).keys()))
        artifact_labels = ", ".join(str(artifact) for artifact in gate.get("expected_artifacts", []))
        closure_items.append(
            "<li>"
            f"<span class=\"pill {_status_class(gate['status'])}\">{_h(gate['status'])}</span>"
            f"<strong>{_h(gate['gate_id'])}</strong>"
            f"<small>Missing: {_h(', '.join(gate['missing']))}</small>"
            f"<small>Template: {_h(gate['template_artifact'])}; closure: {_h(gate['closure_artifact'])}</small>"
            f"<small>Commands: {_h(command_labels or 'none')}</small>"
            f"<small>Artifacts: {_h(artifact_labels or 'none')}</small>"
            "</li>"
        )
    handoff_items = []
    for phase in operator_handoff["phases"]:
        handoff_items.append(
            "<li>"
            f"<span class=\"pill {_status_class('pass' if phase['can_run_here_now'] else 'warn')}\">"
            f"{_h('can run' if phase['can_run_here_now'] else 'needs inputs')}</span>"
            f"<strong>{_h(phase['phase_id'])}</strong>"
            f"<small>Gate: {_h(phase['gate_id'])}</small>"
            f"<small>First safe command: {_h(phase['first_safe_command'])}</small>"
            f"<small>Blocked by: {_h(', '.join(phase['blocked_by_current_package']) if phase['blocked_by_current_package'] else 'none')}</small>"
            "</li>"
        )
    guardrail_items = "".join(f"<li>{_h(item)}</li>" for item in nemoclaw["blocked_capabilities"])
    surfaces = "".join(
        "<li>"
        f"<span>{_h(surface['channel'])}</span>"
        f"<strong>{_h(surface['role'])}</strong>"
        f"<small>{_h(surface['status'])}</small>"
        "</li>"
        for surface in demo["voice_surfaces"]
    )
    held_action_text = ", ".join(action["action_id"] for action in held_actions) if held_actions else "none"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_h(demo['demo']['name'])}</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #5f6b7a;
      --line: #d9e0ea;
      --green: #0f7b5f;
      --green-bg: #dff5ec;
      --amber: #9a5b00;
      --amber-bg: #fff1d2;
      --red: #a62f2f;
      --red-bg: #ffe0df;
      --blue: #2458a6;
      --blue-bg: #e6eefc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    main {{
      width: min(1280px, calc(100vw - 48px));
      margin: 0 auto;
      padding: 28px 0 36px;
    }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 20px;
      align-items: end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 18px;
    }}
    h1 {{ margin: 0; font-size: 30px; line-height: 1.1; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.45; }}
    .grid {{
      display: grid;
      grid-template-columns: 1.45fr 0.9fr;
      gap: 18px;
      align-items: start;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .panel, .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .metric small, li small {{
      display: block;
      color: var(--muted);
      margin-top: 5px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
    .metric strong {{ display: block; font-size: 22px; margin-top: 5px; }}
    .stack {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border-radius: 6px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .ok {{ background: var(--green-bg); color: var(--green); }}
    .warn {{ background: var(--amber-bg); color: var(--amber); }}
    .fail {{ background: var(--red-bg); color: var(--red); }}
    .neutral {{ background: var(--blue-bg); color: var(--blue); }}
    ul {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 10px;
    }}
    li {{
      border-top: 1px solid var(--line);
      padding-top: 10px;
      overflow-wrap: anywhere;
    }}
    li:first-child {{ border-top: 0; padding-top: 0; }}
    li strong {{ display: block; margin-top: 5px; }}
    .bar {{
      height: 10px;
      background: #edf1f6;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 10px;
    }}
    .bar span {{
      display: block;
      height: 100%;
      width: {approval_percent}%;
      background: var(--green);
    }}
    .side {{ display: grid; gap: 18px; }}
    .section-gap {{ display: grid; gap: 18px; }}
    @media (max-width: 860px) {{
      main {{ width: min(100vw - 28px, 760px); }}
      header, .grid, .metrics, .stack {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>{_h(demo['demo']['name'])}</h1>
        <p>{_h(demo['demo']['submission_theme'])}</p>
      </div>
      <span class="pill {_status_class('pass' if readiness['ready_for_recording'] else 'warn')}">{_h(readiness_label)}</span>
    </header>

    <section class="metrics">
      <div class="metric"><small>Budget</small><strong>{_h(_dollars(demo['spend_policy']['limit_cents']))}</strong></div>
      <div class="metric"><small>Approval queued</small><strong>{_h(_dollars(approval_cents))}</strong><div class="bar"><span></span></div></div>
      <div class="metric"><small>Artifact failures</small><strong>{_h(len(readiness['required_failures']))}</strong></div>
      <div class="metric"><small>Audit events</small><strong>{_h(len(demo['audit_events']))}</strong></div>
    </section>

    <section class="stack">
      <div class="panel"><h2>Nemotron 3 Super</h2><p>{_h(demo['sponsor_stack']['nemotron_3_super']['role'])}</p></div>
      <div class="panel"><h2>NemoClaw</h2><p>{_h(demo['sponsor_stack']['nemoclaw']['demo_use'])}</p></div>
      <div class="panel"><h2>Stripe Skills</h2><p>{_h(demo['sponsor_stack']['stripe_skills']['demo_use'])}</p></div>
    </section>

    <section class="grid">
      <div class="section-gap">
        <div class="panel">
          <h2>Discord Voice Request</h2>
          <p>{_h(demo['demo']['request'])}</p>
        </div>
        <div class="panel">
          <h2>KAME Reflex Ack</h2>
          <table>
            <tbody>
              <tr><th>Status</th><td><span class="pill {_status_class(kame_ack['status'])}">{_h(kame_ack['status'])}</span></td></tr>
              <tr><th>Source</th><td>{_h(kame_ack['source'])}</td></tr>
              <tr><th>Ack</th><td>{_h(kame_ack['ack_text'])}</td></tr>
              <tr><th>Evidence</th><td>{_h(kame_ack['evidence_ref'])}</td></tr>
              <tr><th>Latency</th><td>{_h('requires live voice evidence' if kame_ack['latency_ms'] is None else str(kame_ack['latency_ms']) + ' ms')}</td></tr>
            </tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Current Mode</h2>
          <table>
            <tbody>
              <tr><th>NemoClaw Mode</th><td>{_h(nemoclaw['mode'])}</td></tr>
              <tr><th>Operator State</th><td>{_h(operator_state['current_mode'])}</td></tr>
              <tr><th>Active Voice Surface</th><td>{_h(voice_surface['surface_id'])}</td></tr>
              <tr><th>Target Surface</th><td>{_h(voice_surface['fallback_surface_id'])}</td></tr>
              <tr><th>Fallback Reason</th><td>{_h(voice_surface['fallback_reason'])}</td></tr>
            </tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Budget Status</h2>
          <table>
            <tbody>
              <tr><th>Limit</th><td>{_h(_dollars(budget_status['approved_budget_cents']))}</td></tr>
              <tr><th>Approval threshold</th><td>{_h(_dollars(budget_status['approval_required_over_cents']))}</td></tr>
              <tr><th>Reserved approval spend</th><td>{_h(_dollars(budget_status['reserved_cents']))}</td></tr>
              <tr><th>Spent</th><td>{_h(_dollars(budget_status['spent_cents']))}</td></tr>
              <tr><th>Remaining before approval</th><td>{_h(_dollars(budget_status['remaining_cents']))}</td></tr>
              <tr><th>Held over budget</th><td>{_h(_dollars(budget_status['held_budget_cents']))} ({_h(held_action_text)})</td></tr>
            </tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Pending Approvals</h2>
          <table>
            <thead><tr><th>Action</th><th>Provider</th><th>Spend</th><th>Purpose</th></tr></thead>
            <tbody>{''.join(pending_rows)}</tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Action Ledger</h2>
          <table>
            <thead><tr><th>Action</th><th>Provider</th><th>Status</th><th>Spend</th><th>Gate</th></tr></thead>
            <tbody>{''.join(action_rows)}</tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Voice Surfaces</h2>
          <ul>{surfaces}</ul>
        </div>
        <div class="panel">
          <h2>Recent Audit Events</h2>
          <table>
            <thead><tr><th>Event</th><th>Action</th><th>Status</th><th>Amount</th><th>Evidence</th></tr></thead>
            <tbody>{''.join(audit_rows)}</tbody>
          </table>
        </div>
      </div>

      <aside class="side">
        <div class="panel">
          <h2>Planned Services</h2>
          <table>
            <thead><tr><th>Action</th><th>Provider</th><th>Status</th><th>Purpose</th></tr></thead>
            <tbody>{''.join(planned_service_rows)}</tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Provisioned Services</h2>
          <table>
            <thead><tr><th>Service</th><th>Provider</th><th>Status</th><th>Capability</th></tr></thead>
            <tbody>{''.join(provisioned_service_rows)}</tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Readiness</h2>
          <ul>{''.join(readiness_items)}</ul>
        </div>
        <div class="panel">
          <h2>Plan Closure Gates</h2>
          <p>Closure index: {_h(closure['readiness_closure_ref'])}</p>
          <ul>{''.join(closure_items)}</ul>
        </div>
        <div class="panel">
          <h2>Operator Handoff</h2>
          <p><a href="operator-handoff-preview.json">operator-handoff-preview.json</a> lists the ordered safe evidence-collection sequence and final reindex command.</p>
          <ul>{''.join(handoff_items)}</ul>
        </div>
        <div class="panel">
          <h2>Model Strategy</h2>
          <ul>
            <li><span>Active</span><strong>{_h(demo['sponsor_stack']['hermes_active_model']['label'])}</strong><small>Selected through Hermes /model.</small></li>
            <li><span>Preferred local</span><strong>Nemotron 3 Super on DGX Spark</strong><small>Spark-local readiness requires measured local evidence.</small></li>
            <li><span>Hosted fallback</span><strong>Clearly labeled /model fallback</strong><small>Hosted fallback does not count as Spark-local readiness proof.</small></li>
          </ul>
        </div>
        <div class="panel">
          <h2>NemoClaw Blocks</h2>
          <ul>{guardrail_items}</ul>
        </div>
        <div class="panel">
          <h2>Phone Handoff</h2>
          <p>{_h(phone_context['spoken_opening'])}</p>
          <p><a href="milestone2-execution-plan.json">Post-approval execution packet</a> records the approval gates, receipt schema, credential-location schema, and rollback notes.</p>
        </div>
        <div class="panel">
          <h2>Upcoming Tasks</h2>
          <p>Persistent household and business task state is bundled in <a href="operator-state.json">operator-state.json</a> and can also be generated independently by the Milestone 5 operator-state command.</p>
        </div>
      </aside>
    </section>
  </main>
</body>
</html>
"""


def _demo_script(demo: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Demo Script",
            "",
            "User, spoken in Discord:",
            "",
            f"  {demo['demo']['request']}",
            "",
            "Hermes reflex reply:",
            "",
            "  I heard you. I will keep this under 200 dollars and ask before anything billable runs.",
            "",
            "Hermes oracle reply, using the hackathon sponsor stack:",
            "",
            "  I prepared a NemoClaw-safe action packet, queued Stripe Projects to provision a VoIP provider, and queued a Link-gated spend request for service credit. I also preserved this Discord context for the outbound phone call.",
            "",
            "Phone handoff:",
            "",
            "  After approval and VoIP provisioning, Hermes would call the user's phone and say: I am continuing from Discord. You gave me a 200 dollar budget to provision VoIP through Stripe Skills, and I am waiting on your approval before live spend.",
            "",
            "Close:",
            "",
            "  This is one Spark-powered Hermes operator carrying context across Discord, a planned post-approval Stripe-provisioned VoIP path, WhatsApp, and phone.",
            "",
        ]
    )


def _stripe_script(actions: Iterable[dict[str, Any]]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Dry-run action queue generated for the Hermes VoiceOps hackathon demo.",
        "# Keep printf in place until provisioning preflight, channel policy, Link approval, and command review pass.",
        "# Lines prefixed with voiceops-action-metadata are comments for audit review; they are not executed.",
        "",
    ]
    for action in actions:
        command = action["command"]
        contract = _approval_contract(action)
        metadata = {
            "schema_version": "voiceops.stripe_actions_dry_run.metadata.v1",
            "action_id": action["action_id"],
            "provider": action["provider"],
            "command": command,
            "purpose": action["purpose"],
            "estimated_cents": action["estimated_cents"],
            "status": action["status"],
            "requires_approval": action["requires_approval"],
            "approval_id": contract["approval_id"],
            "approval_channel": contract["approval_channel"],
            "approval_artifact": contract["approval_artifact"],
            "approved_by_ref": contract["approved_by_ref"],
            "approval_status": contract["status"],
            "allowed_decisions": contract["allowed_decisions"],
            "command_sha256": contract["command_sha256"],
            "default_decision": contract["default_decision"],
            "required_preflight_gates": contract["required_preflight_gates"],
            "ttl_seconds": contract["ttl_seconds"],
            "receipt_ref": _receipt_ref(str(action["action_id"])),
            "credential_location_ref": _credential_location_ref(str(action["action_id"])),
            "rollback_ref": _rollback_ref(str(action["action_id"])),
            "execution_mode": "dry_run_printf_only",
            "provider_command_executes": False,
        }
        lines.append(f"# voiceops-action-metadata {json.dumps(metadata, sort_keys=True)}")
        quoted = shlex.quote(command)
        lines.append(f"printf '%s\\n' {quoted}")
    lines.append("")
    return "\n".join(lines)


def _demo_package(
    demo: dict[str, Any],
    *,
    readiness: dict[str, Any],
    operator_state: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    payload = dict(demo)
    payload["artifact_manifest"] = {key: path.name for key, path in sorted(paths.items())}
    payload["recording_readiness"] = {
        "artifact_ref": paths["readiness_json"].name,
        "ready_for_recording": readiness["ready_for_recording"],
        "static_recording_ready": readiness["static_recording_ready"],
        "ready_for_recording_scope": readiness["ready_for_recording_scope"],
        "live_demo_ready": readiness["live_demo_ready"],
        "live_demo_missing_evidence": readiness["live_demo_missing_evidence"],
        "artifact_required_failures": readiness["artifact_required_failures"],
        "live_prerequisite_failures": readiness["live_prerequisite_failures"],
        "all_required_check_failures": readiness["all_required_check_failures"],
        "spark_local_evidence_status": readiness["spark_local_evidence_status"],
        "required_failures": readiness["required_failures"],
    }
    payload["readiness_closure"] = _demo_closure_summary()
    payload["readiness_closure_ref"] = paths["readiness_closure_summary_json"].name
    payload["readiness_closure_summary_ref"] = paths["readiness_closure_summary_json"].name
    payload["plan_readiness_closure_ref"] = payload["readiness_closure"]["readiness_closure_ref"]
    payload["operator_state"] = operator_state
    payload["operator_state_ref"] = paths["operator_state"].name
    payload["operator_state_events_ref"] = paths["operator_state_events"].name
    payload["operator_handoff_preview_ref"] = paths["operator_handoff_preview_json"].name
    payload["milestone2_execution_plan_ref"] = paths["milestone2_execution_plan"].name
    payload["safety_boundary_refs"] = {
        "nemoclaw_action_packet": paths["nemoclaw_packet"].name,
        "stripe_actions_dry_run": paths["stripe_actions"].name,
        "phone_context": paths["phone_context"].name,
    }
    return payload


def write_demo(
    output_dir: Path,
    demo: dict[str, Any],
    *,
    readiness_env_files: Iterable[Path] = (),
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    readiness = build_readiness_report(demo, env_files=readiness_env_files)
    paths = {
        "json": output_dir / "voiceops-demo.json",
        "markdown": output_dir / "voiceops-demo.md",
        "audit_ledger": output_dir / "audit-ledger.jsonl",
        "demo_script": output_dir / "demo-script.md",
        "nemoclaw_packet": output_dir / "nemoclaw-action-packet.json",
        "phone_context": output_dir / "phone-context.json",
        "readiness_json": output_dir / "readiness-report.json",
        "readiness_markdown": output_dir / "readiness-report.md",
        "readiness_closure_summary_json": output_dir / "readiness-closure-summary.json",
        "readiness_closure_summary_markdown": output_dir / "readiness-closure-summary.md",
        "operator_handoff_preview_json": output_dir / "operator-handoff-preview.json",
        "operator_handoff_preview_markdown": output_dir / "operator-handoff-preview.md",
        "dashboard": output_dir / "operator-dashboard.html",
        "operator_state": output_dir / "operator-state.json",
        "operator_state_events": output_dir / "operator-state-events.jsonl",
        "milestone2_execution_plan": output_dir / "milestone2-execution-plan.json",
        "recording_runbook": output_dir / "recording-runbook.md",
        "submission_writeup": output_dir / "submission-writeup.md",
        "stripe_actions": output_dir / "stripe-actions-dry-run.sh",
    }
    operator_state = _operator_state_packet(demo, readiness)
    readiness_closure = _demo_closure_summary()
    operator_handoff = _operator_handoff_preview(demo, readiness)
    _write_json(paths["json"], _demo_package(demo, readiness=readiness, operator_state=operator_state, paths=paths))
    paths["markdown"].write_text(_markdown(demo), encoding="utf-8")
    _write_jsonl(paths["audit_ledger"], demo["audit_events"])
    paths["demo_script"].write_text(_demo_script(demo), encoding="utf-8")
    _write_json(paths["nemoclaw_packet"], _nemoclaw_action_packet(demo))
    _write_json(paths["phone_context"], _phone_context_packet(demo))
    _write_json(paths["readiness_json"], readiness)
    paths["readiness_markdown"].write_text(_readiness_markdown(readiness), encoding="utf-8")
    _write_json(paths["readiness_closure_summary_json"], readiness_closure)
    paths["readiness_closure_summary_markdown"].write_text(
        _readiness_closure_summary_markdown(readiness_closure),
        encoding="utf-8",
    )
    _write_json(paths["operator_handoff_preview_json"], operator_handoff)
    paths["operator_handoff_preview_markdown"].write_text(
        _operator_handoff_preview_markdown(operator_handoff),
        encoding="utf-8",
    )
    _write_json(paths["milestone2_execution_plan"], build_milestone2_execution_plan(_demo_milestone2_report(demo, readiness)))
    _write_json(paths["operator_state"], operator_state)
    _write_jsonl(paths["operator_state_events"], operator_state["recent_audit_events"])
    paths["dashboard"].write_text(_dashboard_html(demo, readiness), encoding="utf-8")
    paths["recording_runbook"].write_text(_recording_runbook(demo, readiness), encoding="utf-8")
    paths["submission_writeup"].write_text(_submission_writeup(demo), encoding="utf-8")
    paths["stripe_actions"].write_text(_stripe_script(demo["ops_actions"]), encoding="utf-8")
    paths["stripe_actions"].chmod(0o755)
    return {key: str(path) for key, path in paths.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hackathon-voiceops-demo/current"))
    parser.add_argument("--demo-name", default="Hermes VoiceOps on DGX Spark")
    parser.add_argument("--request", default=DEFAULT_REQUEST)
    parser.add_argument("--budget-cents", type=int, default=20_000)
    parser.add_argument("--approval-required-over-cents", type=int, default=1_000)
    parser.add_argument(
        "--active-model",
        dest="active_model",
        default="Nemotron 3 Super local on DGX Spark via Hermes /model",
        help="Hermes active model selected through /model.",
    )
    parser.add_argument("--reflex-model", default="Gemma 4 E2B audio-native reflex on Spark")
    parser.add_argument(
        "--hermes-home",
        type=Path,
        default=Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")),
        help="Hermes home whose .env should be considered for readiness without printing secrets.",
    )
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        type=Path,
        help="Additional .env file to include in the readiness presence check.",
    )
    parser.add_argument(
        "--no-default-env-files",
        action="store_true",
        help="Only use process env and explicit --env-file values for readiness.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.budget_cents < 0:
        raise SystemExit("--budget-cents must be non-negative")
    if args.approval_required_over_cents < 0:
        raise SystemExit("--approval-required-over-cents must be non-negative")
    demo = build_demo(args)
    env_files = [] if args.no_default_env_files else _default_readiness_env_files(args.hermes_home)
    env_files.extend(args.env_file)
    paths = write_demo(args.output_dir, demo, readiness_env_files=env_files)
    print(json.dumps({"ok": True, "output_dir": str(args.output_dir), "artifacts": paths}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
