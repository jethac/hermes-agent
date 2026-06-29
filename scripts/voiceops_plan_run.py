#!/usr/bin/env python3
"""Run the headless VoiceOps plan artifact generators.

This script is intentionally non-mutating. It does not spend money, provision
accounts, send messages, place calls, or read secret values. It runs the
existing bounded generators and writes a single summary that can be used as the
hackathon evidence index. Opt-in read-only discovery may perform allowlisted
provider status/catalog checks, but it never grants approval for live action.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.hackathon_voiceops_demo import (
    _operator_handoff_preview_markdown,
    build_demo,
    parse_args as parse_demo_args,
    write_demo,
)
from scripts.voiceops_artifact_package_audit import audit_package, write_audit as write_package_audit
from scripts.voiceops_channel_policy import build_channel_policy, build_review_packet, validate_policy, write_channel_policy
from scripts.voiceops_operator_state import build_operator_state, validate_operator_state, write_operator_state
from scripts.voiceops_provisioning_probe import (
    PHONE_PROVIDER_ENV_KEYS,
    PHONE_TARGET_ENV_KEYS,
    build_probe_report,
    write_probe_artifacts,
)
from scripts.voiceops_spark_matrix import build_matrix, write_matrix
from scripts.voiceops_voice_operator import (
    build_voice_operator_report_from_smoke,
    validate_voice_operator_report,
    write_voice_operator_report,
)


DEFAULT_ARTIFACT_ROOT = Path("artifacts")
DEFAULT_OUTPUT_DIR = Path("artifacts/voiceops-plan/current")
DEFAULT_PACKAGE_AUDIT_RELATIVE_OUTPUT_DIR = Path("voiceops-package-audit/current")
FORBIDDEN_ENV_ROOT = Path("/Users/jethac/.hermes/hermes-agent").expanduser()
SPARK_BENCHMARK_SCAFFOLD_EVIDENCE = (
    "artifacts/voiceops-spark-matrix/current/spark-benchmark-scaffold/spark-benchmark-evidence.json"
)
REALTIME_VOICE_DOCTOR_REPORT = "artifacts/realtime-voice-evidence/live-current/realtime-voice-doctor-report.json"
REALTIME_VOICE_DOCTOR_REPORT_COMMAND = (
    "uv run --extra dev --extra voice hermes doctor --realtime-voice --realtime-voice-smoke "
    "--discord-voice-live-probe --discord-voice-live-probe-require-inbound "
    "--discord-voice-live-probe-wait-seconds 5 "
    f"--realtime-voice-report {REALTIME_VOICE_DOCTOR_REPORT}"
)

def _build_safety_flags(provisioning: dict[str, Any] | None = None) -> dict[str, Any]:
    discovery = provisioning.get("read_only_discovery", {}) if isinstance(provisioning, dict) else {}
    loaded_from_evidence = bool(discovery.get("loaded_from_evidence"))
    network_io = bool(discovery.get("network_io_possible")) and not loaded_from_evidence
    return {
        "network_io": network_io,
        "network_io_scope": "allowlisted_read_only_discovery" if network_io else "none",
        "mutating_network_io": False,
        "read_only_discovery_run_requested": bool(discovery.get("run_requested")) and not loaded_from_evidence,
        "read_only_discovery_grants_approval": False,
        "env_presence_inspection": True,
        "env_secret_values_emitted": False,
        "outbound_sends": False,
        "outbound_calls": False,
        "live_spend": False,
        "provider_provisioning": False,
    }


SAFETY_FLAGS = _build_safety_flags()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relativize_paths(paths: dict[str, str]) -> dict[str, str]:
    return {key: str(Path(value)) for key, value in paths.items()}


def _milestone_result(
    *,
    milestone: str,
    command: str,
    output_dir: Path,
    artifacts: dict[str, str],
    status: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "milestone": milestone,
        "command": command,
        "output_dir": str(output_dir),
        "artifacts": _relativize_paths(artifacts),
        "status": status,
        "details": details or {},
    }


def _provisioning_probe_command(
    *,
    output_dir: Path,
    env_files: list[Path],
    preflight_evidence: Path | None,
    read_only_discovery_evidence: Path | None,
    post_approval_receipts: Path | None,
    nemoclaw_action_packet: Path | None,
    run_command_probes: bool,
    run_readonly_discovery: bool,
) -> str:
    argv = [
        "uv",
        "run",
        "python",
        "scripts/voiceops_provisioning_probe.py",
        "--output-dir",
        str(output_dir),
    ]
    for env_file in env_files:
        argv.extend(["--env-file", str(env_file)])
    if preflight_evidence is not None:
        argv.extend(["--preflight-evidence", str(preflight_evidence)])
    if read_only_discovery_evidence is not None:
        argv.extend(["--read-only-discovery-evidence", str(read_only_discovery_evidence)])
    if post_approval_receipts is not None:
        argv.extend(["--post-approval-receipts", str(post_approval_receipts)])
    if nemoclaw_action_packet is not None:
        argv.extend(["--nemoclaw-action-packet", str(nemoclaw_action_packet)])
    if run_command_probes:
        argv.append("--run-command-probes")
    if run_readonly_discovery:
        argv.append("--run-readonly-discovery")
    return " ".join(shlex.quote(part) for part in argv)


def _result_by_milestone(results: list[dict[str, Any]], milestone: str) -> dict[str, Any]:
    return next(result for result in results if result["milestone"] == milestone)


def _env_present(env: dict[str, str], keys: list[str]) -> dict[str, bool]:
    return {key: bool(str(env.get(key) or "").strip()) for key in keys}


def _env_file_presence(env_files: list[Path], keys: list[str]) -> dict[str, bool]:
    key_set = set(keys)
    presence = {key: False for key in keys}
    for env_file in env_files:
        resolved = env_file.expanduser().resolve(strict=False)
        if resolved == FORBIDDEN_ENV_ROOT or FORBIDDEN_ENV_ROOT in resolved.parents:
            raise ValueError(f"refusing to inspect forbidden Hermes worktree path: {resolved}")
        try:
            lines = resolved.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            continue
        for line in lines:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("export "):
                text = text[len("export ") :].strip()
            key, separator, value = text.partition("=")
            key = key.strip()
            if separator and key in key_set and value.strip().strip("'\""):
                presence[key] = True
    return presence


def _env_presence(env: dict[str, str], env_files: list[Path], keys: list[str]) -> dict[str, bool]:
    env_presence = _env_present(env, keys)
    file_presence = _env_file_presence(env_files, keys)
    return {key: env_presence[key] or file_presence[key] for key in keys}


def _binary_present(name: str, env: dict[str, str]) -> bool:
    return shutil.which(name, path=env.get("PATH", "")) is not None


def _build_current_environment_snapshot(
    *,
    env: dict[str, str],
    env_files: list[Path],
) -> dict[str, Any]:
    discord_env_keys = [
        "DISCORD_BOT_TOKEN",
        "DISCORD_GUILD_ID",
        "DISCORD_HOME_CHANNEL",
        "DISCORD_VOICE_CHANNEL_ID",
        "DISCORD_VOICE_CHANNEL_NAME",
    ]
    provisioning_env_keys = [*PHONE_TARGET_ENV_KEYS, *PHONE_PROVIDER_ENV_KEYS]
    provisioning_binaries = ["stripe", "link-cli", "mppx", "mpp", "mpp-agent", "nemoclaw", "openshell", "twilio", "vapi", "bland"]
    system = platform.system()
    machine = platform.machine()
    nvidia_smi_present = _binary_present("nvidia-smi", env)
    discord_env_presence = _env_presence(env, env_files, discord_env_keys)
    provisioning_env_presence = _env_presence(env, env_files, provisioning_env_keys)
    return {
        "schema_version": "voiceops.current_environment.v1",
        "redaction_policy": "presence booleans only; no env values, tokens, command output, or phone numbers",
        "env_files": [
            {
                "path": str(path),
                "exists": path.expanduser().is_file(),
            }
            for path in env_files
        ],
        "discord": {
            "env_presence": discord_env_presence,
            "live_probe_can_run_here": all(discord_env_presence[key] for key in discord_env_keys[:4]),
            "sidecar_evidence_files_expected": [
                "artifacts/realtime-voice-evidence/live-current/sidecar-session.json",
                "artifacts/realtime-voice-evidence/live-current/live-turn.json",
            ],
        },
        "provisioning": {
            "env_presence": provisioning_env_presence,
            "binary_presence": {binary: _binary_present(binary, env) for binary in provisioning_binaries},
            "required_cli_presence": {
                "stripe": _binary_present("stripe", env),
                "link-cli": _binary_present("link-cli", env),
                "mppx": _binary_present("mppx", env),
            },
            "optional_phone_cli_presence": {
                "twilio": _binary_present("twilio", env),
                "vapi": _binary_present("vapi", env),
                "bland": _binary_present("bland", env),
            },
        },
        "spark": {
            "host_system": system,
            "host_machine": machine,
            "nvidia_smi_present": nvidia_smi_present,
            "dgx_spark_likely": system == "Linux" and machine in {"aarch64", "arm64"} and nvidia_smi_present,
            "hardware_claim": "not_verified_by_plan_run",
        },
    }


def _build_current_environment_blockers(environment: dict[str, Any]) -> dict[str, Any]:
    discord_presence = environment.get("discord", {}).get("env_presence", {})
    provisioning = environment.get("provisioning", {})
    required_cli_presence = provisioning.get("required_cli_presence", {})
    binary_presence = provisioning.get("binary_presence", {})
    spark = environment.get("spark", {})
    mpp_fallback_present = any(
        bool(binary_presence.get(binary))
        for binary in ("mppx", "mpp", "mpp-agent", "nemoclaw", "openshell")
    )
    provisioning_missing = [
        binary
        for binary in ("stripe", "link-cli")
        if required_cli_presence.get(binary) is not True
    ]
    if not mpp_fallback_present:
        provisioning_missing.append("mppx_or_fallback")
    return {
        "hard_failure": False,
        "secret_values_emitted": False,
        "diagnostic_only": True,
        "discord_env": {
            "missing_env_keys": sorted(key for key, present in discord_presence.items() if not present),
            "present_env_keys": sorted(key for key, present in discord_presence.items() if present),
        },
        "provisioning_cli": {
            "missing": provisioning_missing,
            "present": sorted(key for key, present in binary_presence.items() if present),
        },
        "spark_host": {
            "required_hardware": "1x NVIDIA DGX Spark",
            "current_host_hint": "dgx_spark_candidate" if spark.get("dgx_spark_likely") is True else "not_dgx_collection_host",
            "blocks_local_collection_here": spark.get("dgx_spark_likely") is not True,
            "blocks_artifact_generation": False,
        },
    }


def _build_operator_handoff(gates: list[dict[str, Any]], blockers: dict[str, Any]) -> dict[str, Any]:
    gate_by_id = {str(gate.get("gate_id")): gate for gate in gates}
    live_gate = gate_by_id["live_discord_voice_operator"]
    provisioning_gate = gate_by_id["spend_and_provisioning_preflight"]
    spark_gate = gate_by_id["local_spark_stack_matrix"]
    live_commands = [
        live_gate["collection_commands"]["run_realtime_voice_doctor_report"],
        live_gate["collection_commands"]["derive_from_realtime_voice_report"],
        live_gate["collection_commands"]["collect_live_manifest"],
        live_gate["collection_commands"]["audit_live_manifest_no_write"],
        live_gate["collection_commands"]["validate_live_manifest_offline"],
        live_gate["collection_commands"]["ingest_live_manifest"],
        live_gate["rerun_command"],
    ]
    provisioning_commands = [
        provisioning_gate["rerun_commands"]["plan_index_dry_audit"],
        provisioning_gate["collection_commands"]["presence_only"],
        provisioning_gate["collection_commands"]["bounded_version_help"],
        provisioning_gate["rerun_commands"]["plan_index_command_probes"],
        provisioning_gate["collection_commands"]["read_only_discovery"],
        provisioning_gate["rerun_commands"]["plan_index_read_only_discovery"],
        provisioning_gate["collection_commands"]["ingest_read_only_discovery_evidence"],
        provisioning_gate["rerun_commands"]["plan_index_read_only_discovery_evidence"],
        provisioning_gate["collection_commands"]["validate_nemoclaw_action_packet"],
        provisioning_gate["collection_commands"]["refresh_preflight_source_hashes"],
        provisioning_gate["collection_commands"]["ingest_preflight_manifest"],
        provisioning_gate["collection_commands"]["validate_post_approval_receipts"],
        provisioning_gate["rerun_commands"]["plan_index_manifest_and_post_approval_receipts"],
    ]
    spark_commands = [
        spark_gate["collection_commands"]["dgx_eval"],
        spark_gate["collection_commands"]["refresh_source_hashes"],
        spark_gate["collection_commands"]["lint_evidence"],
        spark_gate["collection_commands"]["with_evidence"],
        spark_gate["collection_commands"]["plan_index"],
    ]
    return {
        "schema_version": "voiceops.operator_handoff.v1",
        "purpose": "Ordered external-evidence collection sequence for closing VoiceOps readiness without hand-editing the index.",
        "diagnostic_blockers_ref": "current_environment_blockers",
        "changes_readiness_by_itself": False,
        "secret_policy": "Operators supply secrets only through local env/config files or provider CLIs; never paste secret values into artifacts.",
        "phases": [
            {
                "order": 1,
                "phase_id": "live_discord_voice",
                "gate_id": live_gate["gate_id"],
                "status": live_gate["status"],
                "can_run_here_now": blockers.get("discord_env", {}).get("missing_env_keys") == [],
                "blocked_by_current_environment": {
                    "missing_env_keys": blockers.get("discord_env", {}).get("missing_env_keys", []),
                    "present_env_keys": blockers.get("discord_env", {}).get("present_env_keys", []),
                    "needs_external_live_probe": live_gate["status"] != "live_evidence_supplied_not_readiness_claim",
                },
                "first_safe_command": live_commands[0],
                "required_inputs": [
                    "Discord bot token and channel env/config presence",
                    "running realtime voice sidecar",
                    f"optional hermes doctor --realtime-voice-report JSON at {REALTIME_VOICE_DOCTOR_REPORT} for offline sidecar/live-turn evidence derivation",
                    "discord-live-probe.json with source_artifact, collector_attestation, and connect/playback/inbound/disconnect latency metrics",
                    "sidecar-session.json with collector_attestation, sidecar_mode=production, healthcheck_observed, provider_transport_observed, session_id_redacted, fallback_reason, and session_start/shutdown latency metrics",
                    "live-turn.json with collector_attestation",
                ],
                "commands": live_commands,
                "expected_artifacts": [
                    "artifacts/realtime-voice-evidence/live-current/manifest.json",
                    "artifacts/realtime-voice-evidence/live-current/discord-live-probe.json",
                    "artifacts/realtime-voice-evidence/live-current/sidecar-session.json",
                    "artifacts/realtime-voice-evidence/live-current/live-turn.json",
                    REALTIME_VOICE_DOCTOR_REPORT,
                    "artifacts/realtime-voice-evidence/live-current/sidecar-session.from-realtime-report.json",
                    "artifacts/realtime-voice-evidence/live-current/live-turn.from-realtime-report.json",
                    "artifacts/realtime-voice-evidence/live-current/realtime-voice-report-validation.json",
                    "artifacts/realtime-voice-evidence/live-current/live-evidence-validation.json",
                    "artifacts/voiceops-voice-operator/current/live-voice-evidence-scaffold/manifest.json",
                ],
                "success_check": live_gate["completion_signal"],
                "must_not": live_gate["operator_must_not"],
            },
            {
                "order": 2,
                "phase_id": "spend_and_provisioning_preflight",
                "gate_id": provisioning_gate["gate_id"],
                "status": provisioning_gate["status"],
                "can_run_here_now": blockers.get("provisioning_cli", {}).get("missing") == [],
                "blocked_by_current_environment": {
                    "missing_cli": blockers.get("provisioning_cli", {}).get("missing", []),
                    "present_cli": blockers.get("provisioning_cli", {}).get("present", []),
                    "needs_read_only_discovery": provisioning_gate["status"] != "ready",
                    "needs_redacted_setup_evidence": provisioning_gate["status"] != "ready",
                },
                "diagnostic_command": provisioning_commands[0],
                "first_safe_command": provisioning_commands[1],
                "required_inputs": [
                    ".env or local CLI auth for Stripe/Link/MPP/phone provider",
                    "redacted preflight evidence JSON or manifest",
                    "optional redacted post-approval receipt bundle",
                    "redacted source artifacts with matching SHA-256 and collector_attestation redacted hash",
                ],
                "commands": provisioning_commands,
                "command_safety": {
                    "plan_index_dry_audit": "no_write_no_network_no_probe_audit",
                    "presence_only": "offline_presence_only",
                    "bounded_version_help": "local_subprocess_only_no_network_intent",
                    "plan_index_command_probes": "local_subprocess_only_no_network_intent",
                    "read_only_discovery": "network_possible_allowlisted_read_only",
                    "plan_index_read_only_discovery": "network_possible_allowlisted_read_only",
                    "ingest_read_only_discovery_evidence": "local_redacted_discovery_validation_only",
                    "plan_index_read_only_discovery_evidence": "local_reindex_only",
                    "validate_nemoclaw_action_packet": "local_static_action_packet_validation_only",
                    "refresh_preflight_source_hashes": "local_file_hashing_only",
                    "ingest_preflight_manifest": "local_file_validation_only",
                    "validate_post_approval_receipts": "post_approval_local_validation_only",
                    "plan_index_manifest_and_post_approval_receipts": "local_reindex_only",
                },
                "expected_artifacts": [
                    "artifacts/voiceops-provisioning/current/read-only-discovery.json",
                    "artifacts/voiceops-provisioning/current/read-only-discovery.md",
                    "artifacts/voiceops-provisioning/current/read-only-discovery.manifest.json",
                    "artifacts/voiceops-provisioning/current/audit-ledger.read-only-discovery.jsonl",
                    "artifacts/voiceops-provisioning/current/provisioning-preflight-evidence.json",
                    "artifacts/hackathon-voiceops-demo/current/nemoclaw-action-packet.json",
                    "artifacts/voiceops-provisioning/current/nemoclaw-action-packet.validation.json",
                    "artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json",
                    "artifacts/voiceops-provisioning/current/post-approval-receipts.template.json",
                    "artifacts/voiceops-provisioning/current/post-approval-receipts.example.json",
                    "artifacts/voiceops-provisioning/current/post-approval-receipts-scaffold/post-approval-receipts.json",
                    "artifacts/voiceops-provisioning/current/post-approval-receipts.json",
                    "artifacts/voiceops-provisioning/current/post-approval-receipts.validation.json",
                    "artifacts/voiceops-provisioning/current/audit-ledger.post-approval.jsonl",
                    "artifacts/voiceops-provisioning/current/provisioning-readiness.json",
                ],
                "success_check": provisioning_gate["completion_signal"],
                "must_not": provisioning_gate["operator_must_not"],
            },
            {
                "order": 3,
                "phase_id": "local_spark_stack",
                "gate_id": spark_gate["gate_id"],
                "status": spark_gate["status"],
                "can_run_here_now": blockers.get("spark_host", {}).get("blocks_local_collection_here") is False,
                "blocked_by_current_environment": {
                    "required_hardware": blockers.get("spark_host", {}).get("required_hardware", "1x NVIDIA DGX Spark"),
                    "current_host_hint": blockers.get("spark_host", {}).get("current_host_hint", "not_dgx_collection_host"),
                    "needs_measured_spark_evidence": spark_gate["status"] != "validated",
                },
                "first_safe_command": spark_commands[0],
                "required_inputs": [
                    "1x NVIDIA DGX Spark host",
                    "local KAME launch pack or equivalent services",
                    "filled voiceops.spark_benchmark_evidence.v1 JSON",
                    "readable benchmark source artifacts",
                ],
                "commands": spark_commands,
                "expected_artifacts": [
                    "artifacts/dgx-spark-gemma4-voice-eval/current/kame-stack",
                    "artifacts/voiceops-spark-matrix/current/spark-benchmark-scaffold/spark-benchmark-evidence.json",
                    "artifacts/voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/asr-nemotron-speech-raw.json",
                    "artifacts/voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/tts-magpie-local-raw.json",
                    "artifacts/voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/all-local-stack-smoke-raw.json",
                    "artifacts/voiceops-spark-matrix/current/spark-operator-runbook.md",
                    SPARK_BENCHMARK_SCAFFOLD_EVIDENCE,
                    "artifacts/voiceops-spark-matrix/current/spark-model-matrix.json",
                ],
                "success_check": spark_gate["completion_signal"],
                "must_not": spark_gate["operator_must_not"],
            },
        ],
        "final_reindex_command": (
            "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
            "--output-dir artifacts/voiceops-plan/current "
            "--package-audit "
            "--voice-live-evidence artifacts/realtime-voice-evidence/live-current/manifest.json "
            "--env-file .env "
            "--read-only-discovery-evidence artifacts/voiceops-provisioning/current/read-only-discovery.manifest.json "
            "--provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json "
            "--post-approval-receipts artifacts/voiceops-provisioning/current/post-approval-receipts.json "
            f"--evidence {SPARK_BENCHMARK_SCAFFOLD_EVIDENCE}"
        ),
        "final_package_audit_command": (
            "uv run python scripts/voiceops_artifact_package_audit.py --artifact-root artifacts "
            "--output-dir artifacts/voiceops-package-audit/current"
        ),
        "final_success_signal": (
            "readiness_gaps is [] and closure_status is complete and package_audit.status is pass"
        ),
    }


def _plan_model_flag_args(plan_args: dict[str, Any] | None) -> list[str]:
    if not isinstance(plan_args, dict):
        return []
    flags: list[str] = []
    active_model = plan_args.get("active_model")
    reflex_model = plan_args.get("reflex_model")
    if active_model:
        flags.extend(["--active-model", str(active_model)])
    if reflex_model:
        flags.extend(["--reflex-model", str(reflex_model)])
    return flags


def _append_plan_model_flags(value: Any, model_flags: list[str]) -> Any:
    if not model_flags:
        return value
    if isinstance(value, str):
        if "scripts/voiceops_plan_run.py" not in value:
            return value
        if "--active-model" in value or "--reflex-model" in value:
            return value
        return f"{value} {shlex.join(model_flags)}"
    if isinstance(value, list):
        return [_append_plan_model_flags(item, model_flags) for item in value]
    if isinstance(value, dict):
        return {key: _append_plan_model_flags(item, model_flags) for key, item in value.items()}
    return value


def _sync_demo_handoff_preview(demo_dir: Path, plan_handoff: dict[str, Any]) -> None:
    preview_path = demo_dir / "operator-handoff-preview.json"
    markdown_path = demo_dir / "operator-handoff-preview.md"
    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    plan_phases = {
        str(phase.get("phase_id")): phase
        for phase in plan_handoff.get("phases", [])
        if isinstance(phase, dict)
    }
    for phase in preview.get("phases", []):
        if not isinstance(phase, dict):
            continue
        plan_phase = plan_phases.get(str(phase.get("phase_id")))
        if not isinstance(plan_phase, dict):
            continue
        for key in ("commands", "expected_artifacts", "success_check"):
            phase[key] = plan_phase.get(key)
        if plan_phase.get("first_safe_command"):
            phase["first_safe_command"] = plan_phase["first_safe_command"]
    for key in ("final_reindex_command", "final_package_audit_command", "final_success_signal"):
        preview[key] = plan_handoff.get(key)
    _write_json(preview_path, preview)
    markdown_path.write_text(_operator_handoff_preview_markdown(preview), encoding="utf-8")


def _build_next_actions(
    *, remaining_gates: list[dict[str, Any]], handoff: dict[str, Any], blockers: dict[str, Any]
) -> list[dict[str, Any]]:
    phases = {
        str(phase.get("gate_id")): phase
        for phase in handoff.get("phases", [])
        if isinstance(phase, dict)
    }
    actions: list[dict[str, Any]] = []
    for index, gate in enumerate(remaining_gates, start=1):
        gate_id = str(gate.get("gate_id"))
        phase = phases.get(gate_id, {})
        commands = phase.get("commands") if isinstance(phase, dict) else None
        first_command = commands[0] if isinstance(commands, list) and commands else gate.get("rerun_command")
        diagnostic_command = None
        if gate_id == "live_discord_voice_operator":
            blocked_by = {
                "missing_env_keys": blockers.get("discord_env", {}).get("missing_env_keys", []),
                "needs_external_live_probe": True,
            }
            operator_step = (
                "Run the realtime voice doctor report into the live evidence artifact directory, derive sidecar/live-turn "
                "evidence from that report, then run the live Discord evidence collector after Discord env/config and "
                "production sidecar are ready."
            )
        elif gate_id == "spend_and_provisioning_preflight":
            if isinstance(commands, list) and len(commands) > 1:
                diagnostic_command = commands[0]
                first_command = commands[1]
            blocked_by = {
                "missing_cli": blockers.get("provisioning_cli", {}).get("missing", []),
                "needs_read_only_discovery": True,
                "needs_redacted_setup_evidence": True,
            }
            operator_step = (
                "Run the no-write dry audit for status if needed, then collect local provisioning presence evidence, "
                "read-only discovery, and redacted setup evidence."
            )
        elif gate_id == "local_spark_stack_matrix":
            blocked_by = {
                "current_host_hint": blockers.get("spark_host", {}).get("current_host_hint", "unknown"),
                "needs_measured_spark_evidence": True,
                "required_hardware": blockers.get("spark_host", {}).get("required_hardware", "1x NVIDIA DGX Spark"),
            }
            operator_step = "Collect measured local DGX Spark KAME/reflex/oracle/ASR/TTS evidence and re-run the matrix with that evidence."
        else:
            blocked_by = {"needs_external_evidence": True}
            operator_step = "Collect the required external evidence for this gate and re-run the plan index."
        actions.append(
            {
                "order": index,
                "gate_id": gate_id,
                "milestone": gate.get("milestone"),
                "status": gate.get("status"),
                "can_run_here_now": bool(phase.get("can_run_here_now")) if isinstance(phase, dict) else False,
                "blocked_by_current_environment": blocked_by,
                "first_safe_command": first_command,
                "first_evidence_command": first_command,
                **({"diagnostic_command": diagnostic_command} if diagnostic_command else {}),
                "success_check": phase.get("success_check") if isinstance(phase, dict) else gate.get("completion_signal"),
                "operator_step": operator_step,
                "secret_policy": "presence booleans and redacted artifact refs only; never include secret values",
            }
        )
    return actions


def build_readiness_closure_index(summary: dict[str, Any]) -> dict[str, Any]:
    results = summary["results"]
    voice = _result_by_milestone(results, "milestone_1_real_voice_operator")
    provisioning = _result_by_milestone(results, "milestone_2_real_spend_and_provisioning_preflight")
    spark = _result_by_milestone(results, "milestone_4_local_spark_stack_matrix")
    current_environment = summary.get("current_environment", {})
    source_plan_run = str(Path(summary["output_dir"]) / "voiceops-plan-run.json")
    voice_missing = voice["details"].get("live_probe_missing_gates", [])
    provisioning_missing = provisioning["details"].get("required_failures", [])
    spark_missing = [
        f"{role}:{status}"
        for role, status in sorted(spark["details"].get("role_status", {}).items())
        if status != "validated"
    ]
    stack_smoke_status = str(spark["details"].get("stack_smoke_status") or "")
    if stack_smoke_status != "validated":
        spark_missing.append(f"all_local_stack_smoke:{stack_smoke_status or 'needs_evidence'}")
    gates = [
        {
            "milestone": voice["milestone"],
            "status": voice["status"],
            "gate_id": "live_discord_voice_operator",
            "gate_ids": voice_missing,
            "missing": voice_missing,
            "blocking_reason": (
                "Headless loopback does not prove real Discord gateway join, live receiver transport, "
                "production sidecar availability, or one real voice turn."
            ),
            "evidence_template": voice["artifacts"].get("live_evidence_template"),
            "template_artifact": voice["artifacts"].get("live_evidence_template"),
            "evidence_example": voice["artifacts"].get("live_evidence_example"),
            "evidence_scaffold": voice["artifacts"].get("live_evidence_scaffold_manifest"),
            "closure_plan": voice["artifacts"].get("live_probe_closure_json"),
            "closure_artifact": voice["artifacts"].get("live_probe_closure_markdown"),
            "collection_commands": {
                "run_realtime_voice_doctor_report": REALTIME_VOICE_DOCTOR_REPORT_COMMAND,
                "derive_from_realtime_voice_report": (
                    "uv run python -m hermes_cli.realtime_voice_live_evidence "
                    "--output-dir artifacts/realtime-voice-evidence/live-current "
                    f"--from-realtime-voice-report {REALTIME_VOICE_DOCTOR_REPORT}"
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
                "audit_live_manifest_no_write": (
                    "uv run python -m hermes_cli.realtime_voice_live_evidence "
                    "--audit-only "
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
            "required_evidence_fields": [
                "schema_version",
                "connect_perm",
                "speak_perm",
                "connected",
                "opus_loaded",
                "accepted_audio_source",
                "played",
                "playing_during_probe",
                "receiver_started",
                "receiver_frames",
                "receiver_speech_start",
                "inbound_observed",
                "disconnected",
                "require_inbound",
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
                "latency_metrics_ms.connect_ms",
                "latency_metrics_ms.playback_observed_ms",
                "latency_metrics_ms.inbound_observed_ms",
                "latency_metrics_ms.disconnect_ms",
                "latency_metrics_ms.session_start_ms",
                "latency_metrics_ms.shutdown_ms",
                "source_artifact",
                "collector_attestation",
                "transcript_observed",
                "assistant_audio_observed",
                "barge_in_observed",
                "spoken_reply_short",
                "no_voice_denial_observed",
                "speech_end_to_first_audio_ms",
                "barge_in_stop_ms",
            ],
            "evidence_contract": {
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
            },
            "rerun_command": (
                "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                "--output-dir artifacts/voiceops-plan/current "
                "--package-audit "
                "--voice-live-evidence artifacts/realtime-voice-evidence/live-current/manifest.json"
            ),
            "operator_must_not": [
                "paste Discord bot tokens or provider tokens into evidence files",
                "include full phone numbers or private transcript content with secrets",
                "claim production readiness from the headless loopback smoke alone",
            ],
            "completion_signal": "live_probe_missing_gates becomes [] and live_probe_status is live_evidence_supplied_not_readiness_claim",
            "current_environment": current_environment.get("discord", {}),
        },
        {
            "milestone": provisioning["milestone"],
            "status": provisioning["status"],
            "gate_id": "spend_and_provisioning_preflight",
            "gate_ids": provisioning_missing,
            "missing": provisioning_missing,
            "evidence_template": provisioning["artifacts"].get("preflight_evidence_template"),
            "template_artifact": provisioning["artifacts"].get("preflight_evidence_template"),
            "evidence_example": provisioning["artifacts"].get("preflight_evidence_example"),
            "evidence_manifest_example": provisioning["artifacts"].get("preflight_evidence_manifest_example"),
            "evidence_scaffold": provisioning["artifacts"].get("preflight_evidence_scaffold_manifest"),
            "closure_plan": provisioning["artifacts"].get("setup_closure_json"),
            "closure_artifact": provisioning["artifacts"].get("setup_closure_markdown"),
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
                "ingest_read_only_discovery_evidence": (
                    "uv run python scripts/voiceops_provisioning_probe.py "
                    "--output-dir artifacts/voiceops-provisioning/current --env-file .env "
                    "--read-only-discovery-evidence artifacts/voiceops-provisioning/current/read-only-discovery.manifest.json"
                ),
                "validate_nemoclaw_action_packet": (
                    "uv run python scripts/voiceops_provisioning_probe.py "
                    "--output-dir artifacts/voiceops-provisioning/current --env-file .env --no-command-probes "
                    "--nemoclaw-action-packet artifacts/hackathon-voiceops-demo/current/nemoclaw-action-packet.json"
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
            "requirement_fields_per_gate": [
                "schema_version",
                "check_id",
                "area",
                "category",
                "status",
                "closure_state",
                "detail",
                "operator_action",
                "next_step",
                "proof",
                "accepted_binaries",
                "accepted_env_keys",
                "safe_probe_commands",
                "evidence_artifacts",
            ],
            "missing_preflight_fields": provisioning["details"].get("preflight_evidence_missing_fields", []),
            "evidence_contract": {
                "preflight_schema_version": "voiceops.milestone2.preflight_evidence.v1",
                "manifest_schema_version": "voiceops.milestone2.preflight_evidence_manifest.v1",
                "required_sections": ["stripe_projects", "stripe_link", "mpp", "phone_handoff", "rollback"],
                "required_section_field": "source_artifact",
                "required_section_provenance_fields": [
                    "source_artifact_kind",
                    "source_artifact_sha256",
                    "source_artifact_redacted_at",
                    "collector_attestation",
                ],
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
                "nemoclaw_action_packet_validation_schema_version": "voiceops.nemoclaw_action_packet_validation.v1",
                "nemoclaw_action_packet_validation_grants_approval": False,
                "nemoclaw_action_packet_validation_executes_commands": False,
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
                    "--output-dir artifacts/voiceops-plan/current --dry-audit --package-audit"
                ),
                "presence_only": (
                    "uv run python scripts/voiceops_provisioning_probe.py "
                    "--output-dir artifacts/voiceops-provisioning/current --env-file .env"
                ),
                "bounded_version_help": (
                    "uv run python scripts/voiceops_provisioning_probe.py "
                    "--output-dir artifacts/voiceops-provisioning/current --env-file .env --run-command-probes"
                ),
                "plan_index_command_probes": (
                    "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                    "--output-dir artifacts/voiceops-plan/current --package-audit --env-file .env --run-command-probes"
                ),
                "read_only_discovery": (
                    "uv run python scripts/voiceops_provisioning_probe.py "
                    "--output-dir artifacts/voiceops-provisioning/current --env-file .env --run-readonly-discovery"
                ),
                "plan_index_read_only_discovery": (
                    "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                    "--output-dir artifacts/voiceops-plan/current --package-audit --env-file .env --run-readonly-discovery"
                ),
                "plan_index_read_only_discovery_evidence": (
                    "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                    "--output-dir artifacts/voiceops-plan/current --package-audit --env-file .env "
                    "--read-only-discovery-evidence artifacts/voiceops-provisioning/current/read-only-discovery.manifest.json"
                ),
                "plan_index": (
                    "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                    "--output-dir artifacts/voiceops-plan/current --package-audit --env-file .env "
                    "--provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-evidence.json"
                ),
                "plan_index_manifest": (
                    "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                    "--output-dir artifacts/voiceops-plan/current --package-audit --env-file .env "
                    "--provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json"
                ),
                "plan_index_manifest_and_post_approval_receipts": (
                    "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                    "--output-dir artifacts/voiceops-plan/current --package-audit --env-file .env "
                    "--read-only-discovery-evidence artifacts/voiceops-provisioning/current/read-only-discovery.manifest.json "
                    "--provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json "
                    "--post-approval-receipts artifacts/voiceops-provisioning/current/post-approval-receipts.json"
                ),
                "refresh_preflight_source_hashes": (
                    "uv run python scripts/voiceops_provisioning_probe.py "
                    "--refresh-preflight-source-hashes "
                    "artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json"
                ),
                "plan_index_post_approval_receipts": (
                    "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                    "--output-dir artifacts/voiceops-plan/current --package-audit --env-file .env "
                    "--post-approval-receipts artifacts/voiceops-provisioning/current/post-approval-receipts.json"
                ),
            },
            "rerun_command": (
                "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                "--output-dir artifacts/voiceops-plan/current "
                "--package-audit "
                "--env-file .env "
                "--provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-evidence.json"
            ),
            "operator_must_not": [
                "paste secret values into chat or artifact files",
                "use /Users/jethac/.hermes/hermes-agent as an env-file source",
                "run mutating Stripe Projects, Link spend, provider provisioning, or phone-call commands before approval",
            ],
            "completion_signal": (
                "required_failures becomes []; read_only_discovery_status is pass; milestone status becomes ready; if post-approval receipts are "
                "supplied, post_approval_receipts_status is valid, post_approval_receipts_validation_issues is [], "
                "receipt_count covers all expected approval-required actions, and audit-ledger.post-approval.jsonl is populated"
            ),
            "current_environment": current_environment.get("provisioning", {}),
        },
        {
            "milestone": spark["milestone"],
            "status": spark["status"],
            "gate_id": "local_spark_stack_matrix",
            "gate_ids": ["reflex", "oracle", "asr", "tts", "all_local_stack_smoke"],
            "missing": spark_missing,
            "evidence_template": spark["artifacts"].get("evidence_template"),
            "template_artifact": spark["artifacts"].get("evidence_template"),
            "evidence_example": spark["artifacts"].get("evidence_example"),
            "evidence_scaffold": spark["artifacts"].get("evidence_scaffold"),
            "matrix_artifact": spark["artifacts"].get("json"),
            "closure_plan": spark["artifacts"].get("closure_json"),
            "closure_artifact": spark["artifacts"].get("closure_markdown"),
            "operator_runbook": spark["artifacts"].get("operator_runbook"),
            "collection_commands": {
                "matrix_only": (
                    "uv run python scripts/voiceops_spark_matrix.py "
                    "--output-dir artifacts/voiceops-spark-matrix/current"
                ),
                "with_evidence": (
                    "uv run python scripts/voiceops_spark_matrix.py "
                    "--output-dir artifacts/voiceops-spark-matrix/current "
                    f"--evidence {SPARK_BENCHMARK_SCAFFOLD_EVIDENCE}"
                ),
                "refresh_source_hashes": (
                    "uv run python scripts/voiceops_spark_matrix.py "
                    f"--refresh-source-hashes {SPARK_BENCHMARK_SCAFFOLD_EVIDENCE}"
                ),
                "lint_evidence": (
                    "uv run python scripts/voiceops_spark_matrix.py "
                    "--lint-evidence "
                    f"--evidence {SPARK_BENCHMARK_SCAFFOLD_EVIDENCE}"
                ),
                "plan_index": (
                    "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                    "--output-dir artifacts/voiceops-plan/current "
                    "--package-audit "
                    f"--evidence {SPARK_BENCHMARK_SCAFFOLD_EVIDENCE}"
                ),
                "dgx_eval": "scripts/dgx_spark_gemma4_voice_eval.sh",
            },
            "required_candidate_fields": [
                "schema_version",
                "candidate_id",
                "model",
                "engine",
                "hardware",
                "locality",
                "measured_at",
                "metrics",
                "source_artifact",
                "source_artifact_sha256",
                "verified",
            ],
            "required_stack_smoke_fields": [
                "schema_version",
                "kind",
                "components",
                "hardware",
                "locality",
                "measured_at",
                "metrics.speech_end_to_first_audio_ms",
                "metrics.barge_in_stop_ms",
                "source_artifact",
                "source_artifact_sha256",
                "oracle_authority_routes",
                "interface_input_sources",
                "reflex_providers",
                "verified",
            ],
            "current_issues": [
                *spark_missing,
                *spark["details"].get("stack_smoke_issues", []),
            ],
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
                "--package-audit "
                f"--evidence {SPARK_BENCHMARK_SCAFFOLD_EVIDENCE}"
            ),
            "operator_must_not": [
                "claim one-Spark readiness from hosted or multi-Spark Nemotron 3 Ultra fallback evidence or cloud TTS fallback evidence",
                "mark benchmark evidence verified without raw source artifacts",
                "treat the matrix template as measured evidence",
                "treat loopback_smoke_bridge protocol smoke checks as verified local ASR/TTS evidence",
            ],
            "completion_signal": "ready_for_one_spark_demo is true, role_status values are validated, and all_local_stack_smoke is validated",
            "current_environment": current_environment.get("spark", {}),
        },
    ]
    model_flags = _plan_model_flag_args(summary.get("plan_args"))
    gates = _append_plan_model_flags(gates, model_flags)
    blockers = _build_current_environment_blockers(current_environment)
    readiness_gap_milestones = set(summary["readiness_gaps"])
    remaining_gates = [gate for gate in gates if gate["milestone"] in readiness_gap_milestones]
    handoff = _build_operator_handoff(gates, blockers)
    handoff = _append_plan_model_flags(handoff, model_flags)
    next_actions = _build_next_actions(remaining_gates=remaining_gates, handoff=handoff, blockers=blockers)
    return {
        "schema_version": "voiceops.closure_index.v1",
        "artifact_id": "voiceops-plan-readiness-closure",
        "source_plan_run_artifact": source_plan_run,
        "artifact_only": True,
        "safety": {
            **summary.get("safety", _build_safety_flags()),
            "spark_execution": False,
            "secret_values_emitted": False,
        },
        "readiness_gaps": summary["readiness_gaps"],
        "current_environment": current_environment,
        "current_environment_blockers": blockers,
        "operator_handoff": handoff,
        "next_actions": next_actions,
        "closure_status": "needs_external_evidence" if summary["readiness_gaps"] else "complete",
        "remaining_gates": remaining_gates,
        "gates": gates,
    }


def build_plan_run(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    budget_cents: int = 20_000,
    active_model: str | None = None,
    reflex_model: str | None = None,
    evidence_paths: list[Path] | None = None,
    env_files: list[Path] | None = None,
    voice_live_evidence_paths: list[Path] | None = None,
    provisioning_preflight_evidence: Path | None = None,
    read_only_discovery_evidence: Path | None = None,
    post_approval_receipts: Path | None = None,
    run_command_probes: bool = False,
    run_readonly_discovery: bool = False,
    timeout_seconds: int = 3,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        build_plan_run_async(
            artifact_root=artifact_root,
            output_dir=output_dir,
            budget_cents=budget_cents,
            active_model=active_model,
            reflex_model=reflex_model,
            evidence_paths=evidence_paths,
            env_files=env_files,
            voice_live_evidence_paths=voice_live_evidence_paths,
            provisioning_preflight_evidence=provisioning_preflight_evidence,
            read_only_discovery_evidence=read_only_discovery_evidence,
            post_approval_receipts=post_approval_receipts,
            run_command_probes=run_command_probes,
            run_readonly_discovery=run_readonly_discovery,
            timeout_seconds=timeout_seconds,
            env=env,
        )
    )


async def build_plan_run_async(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    budget_cents: int = 20_000,
    active_model: str | None = None,
    reflex_model: str | None = None,
    evidence_paths: list[Path] | None = None,
    env_files: list[Path] | None = None,
    voice_live_evidence_paths: list[Path] | None = None,
    provisioning_preflight_evidence: Path | None = None,
    read_only_discovery_evidence: Path | None = None,
    post_approval_receipts: Path | None = None,
    run_command_probes: bool = False,
    run_readonly_discovery: bool = False,
    timeout_seconds: int = 3,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    evidence_paths = evidence_paths or []
    env_files = env_files or []
    effective_env = dict(os.environ if env is None else env)

    demo_dir = artifact_root / "hackathon-voiceops-demo" / "current"
    voice_operator_dir = artifact_root / "voiceops-voice-operator" / "current"
    provisioning_dir = artifact_root / "voiceops-provisioning" / "current"
    channel_policy_dir = artifact_root / "voiceops-channel-policy" / "current"
    spark_matrix_dir = artifact_root / "voiceops-spark-matrix" / "current"
    operator_state_dir = artifact_root / "voiceops-operator-state" / "current"

    results: list[dict[str, Any]] = []

    demo_argv = ["--output-dir", str(demo_dir), "--budget-cents", str(budget_cents)]
    if active_model:
        demo_argv.extend(["--active-model", active_model])
    if reflex_model:
        demo_argv.extend(["--reflex-model", reflex_model])
    demo_args = parse_demo_args(demo_argv)
    demo = build_demo(demo_args)
    demo_paths = write_demo(demo_dir, demo, readiness_env_files=env_files)
    results.append(
        _milestone_result(
            milestone="milestone_0_hackathon_proof",
            command=shlex.join(["uv", "run", "python", "scripts/hackathon_voiceops_demo.py", *demo_argv]),
            output_dir=demo_dir,
            artifacts=demo_paths,
            status="generated",
            details={
                "dry_run": True,
                "active_model": demo["sponsor_stack"]["hermes_active_model"]["active_model"],
                "active_model_path": demo["sponsor_stack"]["hermes_active_model"]["path"],
                "reflex_model": demo["spark_stack"]["reflex"]["model"],
                "budget_cents": demo["spend_policy"]["limit_cents"],
                "held_budget_cents": demo["totals"]["held_budget_cents"],
                "ready_or_queued_cents": demo["totals"]["ready_or_queued_cents"],
            },
        )
    )

    voice_operator = await build_voice_operator_report_from_smoke(voice_live_evidence_paths)
    voice_operator_issues = validate_voice_operator_report(voice_operator)
    voice_operator_paths = write_voice_operator_report(voice_operator_dir, voice_operator)
    results.append(
        _milestone_result(
            milestone="milestone_1_real_voice_operator",
            command=f"uv run python scripts/voiceops_voice_operator.py --output-dir {voice_operator_dir}",
            output_dir=voice_operator_dir,
            artifacts=voice_operator_paths,
            status=(
                "live_evidence_supplied"
                if not voice_operator_issues
                and voice_operator["live_probe_required_for_completion"]["status"]
                == "live_evidence_supplied_not_readiness_claim"
                else "needs_live_probe"
                if not voice_operator_issues
                else "validation_failed"
            ),
            details={
                "validation_issues": voice_operator_issues,
                "live_probe_status": voice_operator["live_probe_required_for_completion"]["status"],
                "live_probe_missing_gates": voice_operator["live_probe_required_for_completion"]["missing_gates"],
                "latency_metrics_ms": voice_operator["latency_metrics_ms"],
            },
        )
    )

    provisioning = build_probe_report(
        env=effective_env,
        env_files=env_files,
        preflight_evidence_path=provisioning_preflight_evidence,
        read_only_discovery_evidence_path=read_only_discovery_evidence,
        post_approval_receipts_path=post_approval_receipts,
        nemoclaw_action_packet_path=Path(demo_paths["nemoclaw_packet"]),
        run_commands=run_command_probes,
        run_readonly_discovery=run_readonly_discovery,
        timeout_seconds=timeout_seconds,
    )
    provisioning_paths = write_probe_artifacts(provisioning_dir, provisioning)
    results.append(
        _milestone_result(
            milestone="milestone_2_real_spend_and_provisioning_preflight",
            command=_provisioning_probe_command(
                output_dir=provisioning_dir,
                env_files=env_files,
                preflight_evidence=provisioning_preflight_evidence,
                read_only_discovery_evidence=read_only_discovery_evidence,
                post_approval_receipts=post_approval_receipts,
                nemoclaw_action_packet=Path(demo_paths["nemoclaw_packet"]),
                run_command_probes=run_command_probes,
                run_readonly_discovery=run_readonly_discovery,
            ),
            output_dir=provisioning_dir,
            artifacts=provisioning_paths,
            status="ready" if provisioning["ready"] else "needs_setup",
            details={
                "ready": provisioning["ready"],
                "input_paths": {
                    "env_files": [str(path) for path in env_files],
                    "preflight_evidence": str(provisioning_preflight_evidence) if provisioning_preflight_evidence else None,
                    "read_only_discovery_evidence": str(read_only_discovery_evidence) if read_only_discovery_evidence else None,
                    "post_approval_receipts": str(post_approval_receipts) if post_approval_receipts else None,
                    "nemoclaw_action_packet": demo_paths["nemoclaw_packet"],
                },
                "required_failures": provisioning["required_failures"],
                "preflight_evidence_loaded": provisioning["preflight_evidence"]["loaded"],
                "preflight_evidence_missing_fields": provisioning["preflight_evidence"]["missing_fields"],
                "post_approval_receipts_loaded": provisioning["post_approval_receipts"]["loaded"],
                "post_approval_receipts_status": provisioning["post_approval_receipts"]["status"],
                "post_approval_receipt_count": provisioning["post_approval_receipts"].get("receipt_count", 0),
                "post_approval_receipts_validation_issues": provisioning["post_approval_receipts"]["validation_issues"],
                "nemoclaw_action_packet_status": provisioning["nemoclaw_action_packet"]["status"],
                "nemoclaw_action_packet_validation_issues": provisioning["nemoclaw_action_packet"][
                    "validation_issues"
                ],
                "run_command_probes": run_command_probes,
                "run_readonly_discovery": run_readonly_discovery,
                "read_only_discovery_status": provisioning["read_only_discovery"]["status"],
            },
        )
    )

    channel_policy = build_channel_policy()
    channel_review = build_review_packet(channel_policy)
    channel_issues = validate_policy(channel_policy)
    channel_paths = write_channel_policy(channel_policy_dir, channel_policy)
    results.append(
        _milestone_result(
            milestone="milestone_3_multi_channel_policy",
            command=f"uv run python scripts/voiceops_channel_policy.py --output-dir {channel_policy_dir}",
            output_dir=channel_policy_dir,
            artifacts=channel_paths,
            status="needs_review" if not channel_issues else "validation_failed",
            details={
                "validation_issues": channel_issues,
                "review_required_for_real_egress": channel_policy["scope"]["review_required_for_real_egress"],
                "review_status": channel_policy["scope"]["review_status"],
                "real_egress_enabled": channel_policy["scope"]["real_egress_enabled"],
                "review_packet_schema_version": channel_review["schema_version"],
                "review_packet_status": channel_review["review_status"],
                "review_packet_artifact_only": channel_review["artifact_only"],
                "review_packet_changes_policy": channel_review["changes_policy"],
            },
        )
    )

    spark_matrix = build_matrix(evidence_paths)
    spark_paths = write_matrix(spark_matrix_dir, spark_matrix)
    results.append(
        _milestone_result(
            milestone="milestone_4_local_spark_stack_matrix",
            command=f"uv run python scripts/voiceops_spark_matrix.py --output-dir {spark_matrix_dir}",
            output_dir=spark_matrix_dir,
            artifacts=spark_paths,
            status="validated" if spark_matrix["ready_for_one_spark_demo"] else "needs_evidence",
            details={
                "ready_for_one_spark_demo": spark_matrix["ready_for_one_spark_demo"],
                "role_status": spark_matrix["role_status"],
                "stack_smoke_status": spark_matrix["stack_smoke"]["status"],
                "stack_smoke_issues": spark_matrix["stack_smoke"]["issues"],
            },
        )
    )

    operator_state = build_operator_state()
    operator_issues = validate_operator_state(operator_state)
    operator_paths = write_operator_state(operator_state_dir, operator_state)
    results.append(
        _milestone_result(
            milestone="milestone_5_operator_dashboard_state",
            command=f"uv run python scripts/voiceops_operator_state.py --output-dir {operator_state_dir}",
            output_dir=operator_state_dir,
            artifacts=operator_paths,
            status="validated" if not operator_issues else "validation_failed",
            details={"validation_issues": operator_issues},
        )
    )

    hard_failures = [
        result["milestone"]
        for result in results
        if result["status"] in {"validation_failed"}
    ]
    readiness_gaps = [
        result["milestone"]
        for result in results
        if result["status"] in {"needs_setup", "needs_evidence", "needs_live_probe"}
    ]
    summary = {
        "schema_version": "voiceops.plan_run.v1",
        "artifact_id": "voiceops-plan-run",
        "artifact_only": True,
        "safety": _build_safety_flags(provisioning),
        "output_dir": str(output_dir),
        "artifact_root": str(artifact_root),
        "plan_args": {
            "active_model": active_model,
            "reflex_model": reflex_model,
        },
        "ok": not hard_failures,
        "hard_failures": hard_failures,
        "readiness_gaps": readiness_gaps,
        "results": results,
        "current_environment": _build_current_environment_snapshot(env=effective_env, env_files=env_files),
    }
    summary["closure_index"] = build_readiness_closure_index(summary)
    _sync_demo_handoff_preview(demo_dir, summary["closure_index"]["operator_handoff"])
    summary["closure_status"] = summary["closure_index"]["closure_status"]
    summary["remaining_gates"] = [
        gate["gate_id"] for gate in summary["closure_index"]["remaining_gates"]
    ]
    summary["next_actions"] = summary["closure_index"]["next_actions"]
    return summary


def _safety_summary_line(safety: dict[str, Any], *, include_spark: bool = False) -> str:
    network = (
        "read-only discovery network possible only when explicitly requested"
        if safety.get("network_io")
        else "no network I/O"
    )
    parts = [
        "artifact-only",
        network,
        "no mutating network I/O",
        "no secret value emission",
        "no sends",
        "no calls",
        "no live spend",
        "no provider provisioning",
    ]
    if include_spark:
        parts.append("no Spark benchmark execution")
    return "; ".join(parts)


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Plan Run Summary",
        "",
        f"- OK: {'yes' if summary['ok'] else 'no'}",
        f"- Artifact root: `{summary['artifact_root']}`",
        f"- Safety: {_safety_summary_line(summary.get('safety', {}))}",
        f"- Hard failures: {', '.join(summary['hard_failures']) if summary['hard_failures'] else 'none'}",
        f"- Readiness gaps: {', '.join(summary['readiness_gaps']) if summary['readiness_gaps'] else 'none'}",
        "",
        "## Current Environment",
        "",
        _environment_markdown(summary.get("current_environment", {})),
        "",
        "## Current Environment Blockers",
        "",
        _environment_blockers_markdown(summary.get("closure_index", {}).get("current_environment_blockers", {})),
        "",
        "## Operator Handoff",
        "",
        _operator_handoff_markdown(summary.get("closure_index", {}).get("operator_handoff", {})),
        "",
        "## Next Actions",
        "",
        _next_actions_markdown(summary.get("closure_index", {}).get("next_actions", [])),
        "",
        "## Milestones",
        "",
    ]
    for result in summary["results"]:
        lines.extend(
            [
                f"### {result['milestone']}",
                "",
                f"- Status: {result['status']}",
                f"- Output: `{result['output_dir']}`",
                f"- Command: `{result['command']}`",
                "- Artifacts:",
            ]
        )
        for name, path in sorted(result["artifacts"].items()):
            lines.append(f"  - `{name}`: `{path}`")
        if result["details"]:
            lines.append("- Details:")
            for key, value in sorted(result["details"].items()):
                lines.append(f"  - `{key}`: `{value}`")
        lines.append("")
    lines.extend(["## Readiness Closure", ""])
    closure = summary.get("closure_index", {})
    lines.append(f"- Status: {closure.get('closure_status', 'unknown')}")
    for gate in closure.get("gates", []):
        lines.extend(
            [
                f"### {gate['gate_id']}",
                "",
                f"- Milestone: `{gate['milestone']}`",
                f"- Status: {gate['status']}",
                f"- Missing: {', '.join(gate['missing']) if gate['missing'] else 'none'}",
                f"- Template: `{gate['template_artifact']}`",
                f"- Example: `{gate.get('evidence_example') or 'none'}`",
                f"- Manifest example: `{gate.get('evidence_manifest_example') or 'none'}`",
                f"- Closure artifact: `{gate['closure_artifact']}`",
                f"- Rerun: `{gate['rerun_command']}`",
                f"- Completion signal: {gate['completion_signal']}",
            ]
        )
        commands = gate.get("collection_commands")
        if isinstance(commands, dict):
            lines.append("- Collection commands:")
            for label, command in sorted(commands.items()):
                lines.append(f"  - `{label}`: `{command}`")
        contract = gate.get("evidence_contract")
        if isinstance(contract, dict):
            lines.append("- Evidence contract:")
            for key, value in sorted(contract.items()):
                lines.append(f"  - `{key}`: `{value}`")
        lines.append("")
    return "\n".join(lines)


def write_plan_run(output_dir: Path, summary: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "voiceops-plan-run.json",
        "markdown": output_dir / "voiceops-plan-run.md",
        "closure_json": output_dir / "readiness-closure-index.json",
        "closure_markdown": output_dir / "readiness-closure-index.md",
        "operator_handoff_json": output_dir / "operator-handoff.json",
        "operator_handoff_markdown": output_dir / "operator-handoff.md",
    }
    _write_json(paths["json"], summary)
    paths["markdown"].write_text(_markdown(summary), encoding="utf-8")
    _write_json(paths["closure_json"], summary["closure_index"])
    paths["closure_markdown"].write_text(_closure_markdown(summary["closure_index"]), encoding="utf-8")
    _write_json(paths["operator_handoff_json"], summary["closure_index"]["operator_handoff"])
    paths["operator_handoff_markdown"].write_text(
        "# VoiceOps Operator Handoff\n\n" + _operator_handoff_markdown(summary["closure_index"]["operator_handoff"]) + "\n",
        encoding="utf-8",
    )
    return {key: str(path) for key, path in paths.items()}


def _closure_markdown(closure: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Readiness Closure Index",
        "",
        f"- Status: {closure['closure_status']}",
        f"- Safety: {_safety_summary_line(closure.get('safety', {}), include_spark=True)}",
        f"- Readiness gaps: {', '.join(closure['readiness_gaps']) if closure['readiness_gaps'] else 'none'}",
        "",
        "## Current Environment",
        "",
        _environment_markdown(closure.get("current_environment", {})),
        "",
        "## Current Environment Blockers",
        "",
        _environment_blockers_markdown(closure.get("current_environment_blockers", {})),
        "",
        "## Operator Handoff",
        "",
        _operator_handoff_markdown(closure.get("operator_handoff", {})),
        "",
        "## Next Actions",
        "",
        _next_actions_markdown(closure.get("next_actions", [])),
        "",
        "## Gates",
        "",
    ]
    for gate in closure["gates"]:
        lines.extend(
            [
                f"### {gate['gate_id']}",
                "",
                f"- Milestone: `{gate['milestone']}`",
                f"- Status: {gate['status']}",
                f"- Missing: {', '.join(gate['missing']) if gate['missing'] else 'none'}",
                f"- Template artifact: `{gate['template_artifact']}`",
                f"- Example artifact: `{gate.get('evidence_example') or 'none'}`",
                f"- Manifest example artifact: `{gate.get('evidence_manifest_example') or 'none'}`",
                f"- Closure artifact: `{gate['closure_artifact']}`",
            ]
        )
        if gate.get("operator_runbook"):
            lines.append(f"- Operator runbook: `{gate['operator_runbook']}`")
        if gate.get("evidence_scaffold"):
            lines.append(f"- Scaffold artifact: `{gate['evidence_scaffold']}`")
        lines.extend(
            [
                f"- Rerun command: `{gate['rerun_command']}`",
                f"- Completion signal: {gate['completion_signal']}",
            ]
        )
        commands = gate.get("collection_commands")
        if isinstance(commands, dict):
            lines.append("- Collection commands:")
            for label, command in sorted(commands.items()):
                lines.append(f"  - `{label}`: `{command}`")
        contract = gate.get("evidence_contract")
        if isinstance(contract, dict):
            lines.append("- Evidence contract:")
            for key, value in sorted(contract.items()):
                lines.append(f"  - `{key}`: `{value}`")
        current = gate.get("current_environment")
        if isinstance(current, dict) and current:
            lines.append("- Current environment:")
            for key, value in sorted(current.items()):
                lines.append(f"  - `{key}`: `{value}`")
        lines.append("")
    return "\n".join(lines)


def _environment_markdown(environment: dict[str, Any]) -> str:
    if not environment:
        return "- Not captured"
    lines = [
        f"- Redaction policy: {environment.get('redaction_policy', 'presence booleans only')}",
    ]
    env_files = environment.get("env_files")
    if isinstance(env_files, list):
        lines.append("- Env files:")
        for item in env_files:
            if isinstance(item, dict):
                lines.append(f"  - `{item.get('path')}`: exists={item.get('exists')}")
    for section_name in ("discord", "provisioning", "spark"):
        section = environment.get(section_name)
        if isinstance(section, dict):
            lines.append(f"- {section_name}:")
            for key, value in sorted(section.items()):
                lines.append(f"  - `{key}`: `{value}`")
    return "\n".join(lines)


def _environment_blockers_markdown(blockers: dict[str, Any]) -> str:
    if not blockers:
        return "- Not captured"
    lines = [
        "- Diagnostic only: does not change OK, hard failures, readiness statuses, or safety policy",
        f"- Hard failure: {blockers.get('hard_failure')}",
        f"- Secret values emitted: {blockers.get('secret_values_emitted')}",
    ]
    for section_name in ("discord_env", "provisioning_cli", "spark_host"):
        section = blockers.get(section_name)
        if isinstance(section, dict):
            lines.append(f"- {section_name}:")
            for key, value in sorted(section.items()):
                lines.append(f"  - `{key}`: `{value}`")
    return "\n".join(lines)


def _next_actions_markdown(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "- No remaining readiness actions"
    lines: list[str] = []
    for action in actions:
        lines.extend(
            [
                f"### {action.get('order')}. {action.get('gate_id')}",
                f"- Can run here now: {action.get('can_run_here_now')}",
                f"- First safe command: `{action.get('first_safe_command')}`",
                f"- First evidence command: `{action.get('first_evidence_command')}`",
                f"- Success check: {action.get('success_check')}",
                f"- Operator step: {action.get('operator_step')}",
                f"- Secret policy: {action.get('secret_policy')}",
            ]
        )
        if action.get("diagnostic_command"):
            lines.append(f"- Diagnostic command: `{action.get('diagnostic_command')}`")
        blocked_by = action.get("blocked_by_current_environment")
        if isinstance(blocked_by, dict):
            lines.append("- Blocked by current environment:")
            for key, value in sorted(blocked_by.items()):
                lines.append(f"  - `{key}`: `{value}`")
    return "\n".join(lines)


def _operator_handoff_markdown(handoff: dict[str, Any]) -> str:
    if not handoff:
        return "- Not captured"
    lines = [
        f"- Schema: `{handoff.get('schema_version')}`",
        f"- Purpose: {handoff.get('purpose')}",
        f"- Changes readiness by itself: {handoff.get('changes_readiness_by_itself')}",
        f"- Secret policy: {handoff.get('secret_policy')}",
    ]
    phases = handoff.get("phases")
    if isinstance(phases, list):
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            lines.extend(
                [
                    f"### {phase.get('order')}. {phase.get('phase_id')}",
                    f"- Gate: `{phase.get('gate_id')}`",
                    f"- Status: `{phase.get('status')}`",
                    f"- Can run here now: {phase.get('can_run_here_now')}",
                    f"- First safe command: `{phase.get('first_safe_command')}`",
                    *(
                        [f"- Diagnostic command: `{phase.get('diagnostic_command')}`"]
                        if phase.get("diagnostic_command")
                        else []
                    ),
                    f"- Success check: {phase.get('success_check')}",
                ]
            )
            blocked_by = phase.get("blocked_by_current_environment")
            if isinstance(blocked_by, dict) and blocked_by:
                lines.append("- Blocked by current environment:")
                for key, value in sorted(blocked_by.items()):
                    lines.append(f"  - `{key}`: `{value}`")
            for label in ("required_inputs", "expected_artifacts", "commands", "must_not"):
                items = phase.get(label)
                if isinstance(items, list):
                    lines.append(f"- {label}:")
                    for item in items:
                        lines.append(f"  - `{item}`" if label in {"commands", "expected_artifacts"} else f"  - {item}")
            command_safety = phase.get("command_safety")
            if isinstance(command_safety, dict):
                lines.append("- command_safety:")
                for label, value in sorted(command_safety.items()):
                    lines.append(f"  - `{label}`: `{value}`")
    lines.append(f"- Final reindex command: `{handoff.get('final_reindex_command')}`")
    lines.append(f"- Final package audit command: `{handoff.get('final_package_audit_command')}`")
    lines.append(f"- Final success signal: {handoff.get('final_success_signal')}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--budget-cents", type=int, default=20_000)
    parser.add_argument(
        "--active-model",
        default=None,
        help="Active Hermes /model value to pass into the generated hackathon demo package.",
    )
    parser.add_argument(
        "--reflex-model",
        default=None,
        help="KAME reflex/interface model label to pass into the generated hackathon demo package.",
    )
    parser.add_argument("--evidence", action="append", default=[], type=Path)
    parser.add_argument("--env-file", action="append", default=[], type=Path)
    parser.add_argument("--voice-live-evidence", action="append", default=[], type=Path)
    parser.add_argument("--provisioning-preflight-evidence", type=Path, default=None)
    parser.add_argument("--read-only-discovery-evidence", "--readonly-discovery-evidence", type=Path, default=None)
    parser.add_argument("--post-approval-receipts", type=Path, default=None)
    parser.add_argument(
        "--package-audit",
        action="store_true",
        help="After generating the artifact tree, run the local static package consistency audit.",
    )
    parser.add_argument(
        "--package-audit-output-dir",
        type=Path,
        default=None,
        help="Output directory for --package-audit. Defaults to ARTIFACT_ROOT/voiceops-package-audit/current.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=3)
    parser.add_argument(
        "--run-command-probes",
        action="store_true",
        help="Opt into isolated version/help subprocess probes for provisioning readiness.",
    )
    parser.add_argument(
        "--run-readonly-discovery",
        "--run-read-only-discovery",
        action="store_true",
        dest="run_readonly_discovery",
        help="Opt into exact allowlisted read-only discovery commands for provisioning readiness artifacts.",
    )
    parser.add_argument(
        "--dry-audit",
        action="store_true",
        help=(
            "Build the plan summary in a temporary artifact root and print the audit without writing "
            "persistent artifacts. Refuses command probes and read-only discovery."
        ),
    )
    return parser.parse_args(argv)


def _package_audit_output_dir(args: argparse.Namespace) -> Path:
    if args.package_audit_output_dir is not None:
        return args.package_audit_output_dir
    return args.artifact_root / DEFAULT_PACKAGE_AUDIT_RELATIVE_OUTPUT_DIR


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_audit:
        if args.run_command_probes or args.run_readonly_discovery:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "dry_audit": True,
                        "error": "--dry-audit refuses --run-command-probes and --run-readonly-discovery",
                        "persistent_writes": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        with tempfile.TemporaryDirectory(prefix="voiceops-plan-dry-audit-") as tmpdir:
            temp_artifact_root = Path(tmpdir) / "artifacts"
            temp_output_dir = temp_artifact_root / "voiceops-plan" / "current"
            summary = build_plan_run(
                artifact_root=temp_artifact_root,
                output_dir=temp_output_dir,
                budget_cents=args.budget_cents,
                active_model=args.active_model,
                reflex_model=args.reflex_model,
                evidence_paths=args.evidence,
                env_files=args.env_file,
                voice_live_evidence_paths=args.voice_live_evidence,
                provisioning_preflight_evidence=args.provisioning_preflight_evidence,
                read_only_discovery_evidence=args.read_only_discovery_evidence,
                post_approval_receipts=args.post_approval_receipts,
                run_command_probes=False,
                run_readonly_discovery=False,
                timeout_seconds=args.timeout_seconds,
            )
            package_audit_report = None
            if args.package_audit:
                write_plan_run(temp_output_dir, summary)
                package_audit_report = audit_package(temp_artifact_root)
            print(
                json.dumps(
                    {
                        "ok": summary["ok"],
                        "ok_meaning": "no hard validation failures; not a readiness claim",
                        "readiness_ok": (
                            summary["closure_index"]["closure_status"] == "complete"
                            and summary["readiness_gaps"] == []
                        ),
                        "dry_audit": True,
                        "persistent_writes": False,
                        "temporary_artifacts_removed_on_exit": True,
                        "requested_artifact_root": str(args.artifact_root),
                        "requested_output_dir": str(args.output_dir),
                        "readiness_gaps": summary["readiness_gaps"],
                        "hard_failures": summary["hard_failures"],
                        "closure_status": summary["closure_index"]["closure_status"],
                        "remaining_gates": [
                            gate["gate_id"] for gate in summary["closure_index"]["remaining_gates"]
                        ],
                        "safety": summary["safety"],
                        "current_environment_blockers": summary["closure_index"]["current_environment_blockers"],
                        "next_actions": summary["closure_index"]["next_actions"],
                        **(
                            {
                                "package_audit": {
                                    "ok": package_audit_report["ok"],
                                    "status": package_audit_report["status"],
                                    "issues": package_audit_report["issues"],
                                    "checked_artifact_count": package_audit_report["checked_artifact_count"],
                                    "persistent_writes": False,
                                }
                            }
                            if package_audit_report is not None
                            else {}
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            package_audit_ok = package_audit_report is None or package_audit_report["ok"]
            return 0 if summary["ok"] and package_audit_ok else 1
    summary = build_plan_run(
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
        budget_cents=args.budget_cents,
        active_model=args.active_model,
        reflex_model=args.reflex_model,
        evidence_paths=args.evidence,
        env_files=args.env_file,
        voice_live_evidence_paths=args.voice_live_evidence,
        provisioning_preflight_evidence=args.provisioning_preflight_evidence,
        read_only_discovery_evidence=args.read_only_discovery_evidence,
        post_approval_receipts=args.post_approval_receipts,
        run_command_probes=args.run_command_probes,
        run_readonly_discovery=args.run_readonly_discovery,
        timeout_seconds=args.timeout_seconds,
    )
    paths = write_plan_run(args.output_dir, summary)
    package_audit_report = None
    package_audit_paths = None
    if args.package_audit:
        package_audit_report = audit_package(args.artifact_root)
        package_audit_paths = write_package_audit(_package_audit_output_dir(args), package_audit_report)
    print(
        json.dumps(
            {
                "ok": summary["ok"],
                "output_dir": str(args.output_dir),
                "artifacts": paths,
                **(
                    {
                        "package_audit": {
                            "ok": package_audit_report["ok"],
                            "status": package_audit_report["status"],
                            "issues": package_audit_report["issues"],
                            "checked_artifact_count": package_audit_report["checked_artifact_count"],
                            "artifacts": package_audit_paths,
                        }
                    }
                    if package_audit_report is not None and package_audit_paths is not None
                    else {}
                ),
                "readiness_gaps": summary["readiness_gaps"],
                "hard_failures": summary["hard_failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    package_audit_ok = package_audit_report is None or package_audit_report["ok"]
    return 0 if summary["ok"] and package_audit_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
