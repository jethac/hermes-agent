#!/usr/bin/env python3
"""Run the headless VoiceOps plan artifact generators.

This script is intentionally non-mutating. It does not perform provider
network I/O, spend money, provision accounts, send messages, place calls, or
read secret values. It runs the existing bounded generators and writes a single
summary that can be used as the hackathon evidence index.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.hackathon_voiceops_demo import build_demo, parse_args as parse_demo_args, write_demo
from scripts.voiceops_channel_policy import build_channel_policy, validate_policy, write_channel_policy
from scripts.voiceops_operator_state import build_operator_state, validate_operator_state, write_operator_state
from scripts.voiceops_provisioning_probe import build_probe_report, write_probe_artifacts
from scripts.voiceops_spark_matrix import build_matrix, write_matrix
from scripts.voiceops_voice_operator import (
    build_voice_operator_report_from_smoke,
    validate_voice_operator_report,
    write_voice_operator_report,
)


DEFAULT_ARTIFACT_ROOT = Path("artifacts")
DEFAULT_OUTPUT_DIR = Path("artifacts/voiceops-plan/current")

SAFETY_FLAGS = {
    "network_io": False,
    "env_presence_inspection": True,
    "env_secret_values_emitted": False,
    "outbound_sends": False,
    "outbound_calls": False,
    "live_spend": False,
    "provider_provisioning": False,
}


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


def _result_by_milestone(results: list[dict[str, Any]], milestone: str) -> dict[str, Any]:
    return next(result for result in results if result["milestone"] == milestone)


def build_readiness_closure_index(summary: dict[str, Any]) -> dict[str, Any]:
    results = summary["results"]
    voice = _result_by_milestone(results, "milestone_1_real_voice_operator")
    provisioning = _result_by_milestone(results, "milestone_2_real_spend_and_provisioning_preflight")
    spark = _result_by_milestone(results, "milestone_4_local_spark_stack_matrix")
    source_plan_run = str(Path(summary["output_dir"]) / "voiceops-plan-run.json")
    voice_missing = voice["details"].get("live_probe_missing_gates", [])
    provisioning_missing = provisioning["details"].get("required_failures", [])
    spark_missing = [
        f"{role}:{status}"
        for role, status in sorted(spark["details"].get("role_status", {}).items())
        if status != "validated"
    ]
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
            "closure_plan": voice["artifacts"].get("live_probe_closure_json"),
            "closure_artifact": voice["artifacts"].get("live_probe_closure_markdown"),
            "required_evidence_fields": [
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
                "transcript_observed",
                "assistant_audio_observed",
                "barge_in_observed",
                "spoken_reply_short",
                "no_voice_denial_observed",
                "speech_end_to_first_audio_ms",
                "barge_in_stop_ms",
            ],
            "rerun_command": (
                "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                "--output-dir artifacts/voiceops-plan/current "
                "--voice-live-evidence artifacts/realtime-voice-evidence/live-current/discord-live-probe.json "
                "--voice-live-evidence path/to/sidecar-session.json "
                "--voice-live-evidence path/to/live-turn.json"
            ),
            "operator_must_not": [
                "paste Discord bot tokens or provider tokens into evidence files",
                "include full phone numbers or private transcript content with secrets",
                "claim production readiness from the headless loopback smoke alone",
            ],
            "completion_signal": "live_probe_missing_gates becomes [] and live_probe_status is live_evidence_supplied_not_readiness_claim",
        },
        {
            "milestone": provisioning["milestone"],
            "status": provisioning["status"],
            "gate_id": "spend_and_provisioning_preflight",
            "gate_ids": provisioning_missing,
            "missing": provisioning_missing,
            "evidence_template": provisioning["artifacts"].get("preflight_evidence_template"),
            "template_artifact": provisioning["artifacts"].get("preflight_evidence_template"),
            "closure_plan": provisioning["artifacts"].get("setup_closure_json"),
            "closure_artifact": provisioning["artifacts"].get("setup_closure_markdown"),
            "requirement_fields_per_gate": [
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
            "rerun_commands": {
                "presence_only": (
                    "uv run python scripts/voiceops_provisioning_probe.py "
                    "--output-dir artifacts/voiceops-provisioning/current --env-file .env"
                ),
                "bounded_version_help": (
                    "uv run python scripts/voiceops_provisioning_probe.py "
                    "--output-dir artifacts/voiceops-provisioning/current --env-file .env --run-command-probes"
                ),
                "plan_index": (
                    "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                    "--output-dir artifacts/voiceops-plan/current --env-file .env "
                    "--provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-evidence.json"
                ),
            },
            "rerun_command": (
                "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                "--output-dir artifacts/voiceops-plan/current "
                "--env-file .env "
                "--provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-evidence.json"
            ),
            "operator_must_not": [
                "paste secret values into chat or artifact files",
                "use /Users/jethac/.hermes/hermes-agent as an env-file source",
                "run mutating Stripe Projects, Link spend, provider provisioning, or phone-call commands before approval",
            ],
            "completion_signal": "required_failures becomes [] and milestone status becomes ready",
        },
        {
            "milestone": spark["milestone"],
            "status": spark["status"],
            "gate_id": "local_spark_stack_matrix",
            "gate_ids": ["reflex", "oracle", "asr", "tts", "all_local_stack_smoke"],
            "missing": spark_missing,
            "evidence_template": spark["artifacts"].get("evidence_template"),
            "template_artifact": spark["artifacts"].get("evidence_template"),
            "matrix_artifact": spark["artifacts"].get("json"),
            "closure_artifact": spark["artifacts"].get("markdown"),
            "required_candidate_fields": [
                "candidate_id",
                "model",
                "engine",
                "hardware",
                "locality",
                "measured_at",
                "metrics",
                "source_artifact",
                "verified",
            ],
            "required_stack_smoke_fields": [
                "kind",
                "components",
                "hardware",
                "locality",
                "measured_at",
                "metrics.speech_end_to_first_audio_ms",
                "metrics.barge_in_stop_ms",
                "source_artifact",
                "verified",
            ],
            "current_issues": [
                *spark_missing,
                *spark["details"].get("stack_smoke_issues", []),
            ],
            "rerun_command": (
                "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                "--output-dir artifacts/voiceops-plan/current "
                "--evidence path/to/spark-benchmark-evidence.json"
            ),
            "operator_must_not": [
                "claim one-Spark readiness from hosted Ultra or cloud TTS fallback evidence",
                "mark benchmark evidence verified without raw source artifacts",
                "treat the matrix template as measured evidence",
            ],
            "completion_signal": "ready_for_one_spark_demo is true and role_status values are validated",
        },
    ]
    return {
        "schema_version": "voiceops.closure_index.v1",
        "artifact_id": "voiceops-plan-readiness-closure",
        "source_plan_run_artifact": source_plan_run,
        "artifact_only": True,
        "safety": {
            "network_io": False,
            "outbound_sends": False,
            "live_spend": False,
            "provider_provisioning": False,
            "outbound_calls": False,
            "spark_execution": False,
            "secret_values_emitted": False,
        },
        "readiness_gaps": summary["readiness_gaps"],
        "closure_status": "needs_external_evidence" if summary["readiness_gaps"] else "complete",
        "remaining_gates": gates,
        "gates": gates,
    }


def build_plan_run(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    budget_cents: int = 20_000,
    evidence_paths: list[Path] | None = None,
    env_files: list[Path] | None = None,
    voice_live_evidence_paths: list[Path] | None = None,
    provisioning_preflight_evidence: Path | None = None,
    run_command_probes: bool = False,
    timeout_seconds: int = 3,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        build_plan_run_async(
            artifact_root=artifact_root,
            output_dir=output_dir,
            budget_cents=budget_cents,
            evidence_paths=evidence_paths,
            env_files=env_files,
            voice_live_evidence_paths=voice_live_evidence_paths,
            provisioning_preflight_evidence=provisioning_preflight_evidence,
            run_command_probes=run_command_probes,
            timeout_seconds=timeout_seconds,
            env=env,
        )
    )


async def build_plan_run_async(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    budget_cents: int = 20_000,
    evidence_paths: list[Path] | None = None,
    env_files: list[Path] | None = None,
    voice_live_evidence_paths: list[Path] | None = None,
    provisioning_preflight_evidence: Path | None = None,
    run_command_probes: bool = False,
    timeout_seconds: int = 3,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    evidence_paths = evidence_paths or []
    env_files = env_files or []

    demo_dir = artifact_root / "hackathon-voiceops-demo" / "current"
    voice_operator_dir = artifact_root / "voiceops-voice-operator" / "current"
    provisioning_dir = artifact_root / "voiceops-provisioning" / "current"
    channel_policy_dir = artifact_root / "voiceops-channel-policy" / "current"
    spark_matrix_dir = artifact_root / "voiceops-spark-matrix" / "current"
    operator_state_dir = artifact_root / "voiceops-operator-state" / "current"

    results: list[dict[str, Any]] = []

    demo_args = parse_demo_args(["--output-dir", str(demo_dir), "--budget-cents", str(budget_cents)])
    demo = build_demo(demo_args)
    demo_paths = write_demo(demo_dir, demo, readiness_env_files=env_files)
    results.append(
        _milestone_result(
            milestone="milestone_0_hackathon_proof",
            command=f"uv run python scripts/hackathon_voiceops_demo.py --output-dir {demo_dir}",
            output_dir=demo_dir,
            artifacts=demo_paths,
            status="generated",
            details={
                "dry_run": True,
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
        env=os.environ if env is None else env,
        env_files=env_files,
        preflight_evidence_path=provisioning_preflight_evidence,
        run_commands=run_command_probes,
        timeout_seconds=timeout_seconds,
    )
    provisioning_paths = write_probe_artifacts(provisioning_dir, provisioning)
    results.append(
        _milestone_result(
            milestone="milestone_2_real_spend_and_provisioning_preflight",
            command=f"uv run python scripts/voiceops_provisioning_probe.py --output-dir {provisioning_dir}",
            output_dir=provisioning_dir,
            artifacts=provisioning_paths,
            status="ready" if provisioning["ready"] else "needs_setup",
            details={
                "ready": provisioning["ready"],
                "required_failures": provisioning["required_failures"],
                "preflight_evidence_loaded": provisioning["preflight_evidence"]["loaded"],
                "preflight_evidence_missing_fields": provisioning["preflight_evidence"]["missing_fields"],
                "run_command_probes": run_command_probes,
            },
        )
    )

    channel_policy = build_channel_policy()
    channel_issues = validate_policy(channel_policy)
    channel_paths = write_channel_policy(channel_policy_dir, channel_policy)
    results.append(
        _milestone_result(
            milestone="milestone_3_multi_channel_policy",
            command=f"uv run python scripts/voiceops_channel_policy.py --output-dir {channel_policy_dir}",
            output_dir=channel_policy_dir,
            artifacts=channel_paths,
            status="validated" if not channel_issues else "validation_failed",
            details={"validation_issues": channel_issues},
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
        "artifact_only": True,
        "safety": SAFETY_FLAGS,
        "output_dir": str(output_dir),
        "artifact_root": str(artifact_root),
        "ok": not hard_failures,
        "hard_failures": hard_failures,
        "readiness_gaps": readiness_gaps,
        "results": results,
    }
    summary["closure_index"] = build_readiness_closure_index(summary)
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Plan Run Summary",
        "",
        f"- OK: {'yes' if summary['ok'] else 'no'}",
        f"- Artifact root: `{summary['artifact_root']}`",
        "- Safety: artifact-only; no network I/O, secret value emission, sends, calls, live spend, or provider provisioning",
        f"- Hard failures: {', '.join(summary['hard_failures']) if summary['hard_failures'] else 'none'}",
        f"- Readiness gaps: {', '.join(summary['readiness_gaps']) if summary['readiness_gaps'] else 'none'}",
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
                f"- Closure artifact: `{gate['closure_artifact']}`",
                f"- Rerun: `{gate['rerun_command']}`",
                f"- Completion signal: {gate['completion_signal']}",
                "",
            ]
        )
    return "\n".join(lines)


def write_plan_run(output_dir: Path, summary: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "voiceops-plan-run.json",
        "markdown": output_dir / "voiceops-plan-run.md",
        "closure_json": output_dir / "readiness-closure-index.json",
        "closure_markdown": output_dir / "readiness-closure-index.md",
    }
    _write_json(paths["json"], summary)
    paths["markdown"].write_text(_markdown(summary), encoding="utf-8")
    _write_json(paths["closure_json"], summary["closure_index"])
    paths["closure_markdown"].write_text(_closure_markdown(summary["closure_index"]), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def _closure_markdown(closure: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Readiness Closure Index",
        "",
        f"- Status: {closure['closure_status']}",
        "- Safety: artifact-only; no network, spend, provisioning, calls, Spark benchmark execution, or secret values",
        f"- Readiness gaps: {', '.join(closure['readiness_gaps']) if closure['readiness_gaps'] else 'none'}",
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
                f"- Closure artifact: `{gate['closure_artifact']}`",
                f"- Rerun command: `{gate['rerun_command']}`",
                f"- Completion signal: {gate['completion_signal']}",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--budget-cents", type=int, default=20_000)
    parser.add_argument("--evidence", action="append", default=[], type=Path)
    parser.add_argument("--env-file", action="append", default=[], type=Path)
    parser.add_argument("--voice-live-evidence", action="append", default=[], type=Path)
    parser.add_argument("--provisioning-preflight-evidence", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=3)
    parser.add_argument(
        "--run-command-probes",
        action="store_true",
        help="Opt into isolated version/help subprocess probes for provisioning readiness.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_plan_run(
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
        budget_cents=args.budget_cents,
        evidence_paths=args.evidence,
        env_files=args.env_file,
        voice_live_evidence_paths=args.voice_live_evidence,
        provisioning_preflight_evidence=args.provisioning_preflight_evidence,
        run_command_probes=args.run_command_probes,
        timeout_seconds=args.timeout_seconds,
    )
    paths = write_plan_run(args.output_dir, summary)
    print(
        json.dumps(
            {
                "ok": summary["ok"],
                "output_dir": str(args.output_dir),
                "artifacts": paths,
                "readiness_gaps": summary["readiness_gaps"],
                "hard_failures": summary["hard_failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
