#!/usr/bin/env python3
"""Run the headless VoiceOps plan artifact generators.

This script is intentionally non-mutating. It does not perform provider
network I/O, spend money, provision accounts, send messages, place calls, or
read secret values. It runs the existing bounded generators and writes a single
summary that can be used as the hackathon evidence index.
"""

from __future__ import annotations

import argparse
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


DEFAULT_ARTIFACT_ROOT = Path("artifacts")
DEFAULT_OUTPUT_DIR = Path("artifacts/voiceops-plan/current")

SAFETY_FLAGS = {
    "network_io": False,
    "env_secret_reads": False,
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


def build_plan_run(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    budget_cents: int = 20_000,
    evidence_paths: list[Path] | None = None,
    env_files: list[Path] | None = None,
    run_command_probes: bool = False,
    timeout_seconds: int = 3,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    evidence_paths = evidence_paths or []
    env_files = env_files or []

    demo_dir = artifact_root / "hackathon-voiceops-demo" / "current"
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

    provisioning = build_probe_report(
        env=os.environ if env is None else env,
        env_files=env_files,
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
        if result["status"] in {"needs_setup", "needs_evidence"}
    ]
    return {
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


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Plan Run Summary",
        "",
        f"- OK: {'yes' if summary['ok'] else 'no'}",
        f"- Artifact root: `{summary['artifact_root']}`",
        "- Safety: artifact-only; no network I/O, env secret reads, sends, calls, live spend, or provider provisioning",
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
    return "\n".join(lines)


def write_plan_run(output_dir: Path, summary: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "voiceops-plan-run.json",
        "markdown": output_dir / "voiceops-plan-run.md",
    }
    _write_json(paths["json"], summary)
    paths["markdown"].write_text(_markdown(summary), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--budget-cents", type=int, default=20_000)
    parser.add_argument("--evidence", action="append", default=[], type=Path)
    parser.add_argument("--env-file", action="append", default=[], type=Path)
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
