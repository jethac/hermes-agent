#!/usr/bin/env python3
"""Generate and validate the VoiceOps DGX Spark model matrix.

The matrix is headless and non-invasive. By default it writes the target
candidate matrix and marks every hardware claim as needing evidence. Optional
benchmark evidence JSON files can promote candidates to validated or failed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Target:
    metric: str
    operator: str
    value: float
    unit: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    role: str
    model: str
    engine: str
    locality: str
    priority: int
    purpose: str
    required_targets: list[Target]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _targets(*targets: tuple[str, str, float, str]) -> list[Target]:
    return [Target(metric=metric, operator=operator, value=value, unit=unit) for metric, operator, value, unit in targets]


def default_candidates() -> list[Candidate]:
    return [
        Candidate(
            candidate_id="reflex-gemma4-e2b",
            role="reflex",
            model="Gemma 4 E2B audio-native",
            engine="vLLM multimodal audio path or equivalent Spark container",
            locality="local_spark",
            priority=1,
            purpose="low-latency KAME interface model for intent triage, turn-taking, and floor control",
            required_targets=_targets(
                ("first_token_ms", "<=", 800, "ms"),
                ("intent_latency_ms", "<=", 1200, "ms"),
                ("steady_state_memory_gb", "<=", 32, "GB"),
            ),
        ),
        Candidate(
            candidate_id="oracle-nemotron3-super-local",
            role="oracle",
            model="Nemotron 3 Super",
            engine="Hermes /model to local NVIDIA Spark endpoint",
            locality="local_spark",
            priority=1,
            purpose="preferred Spark-local NVIDIA oracle target for Hermes planning and tool orchestration",
            required_targets=_targets(
                ("decode_tok_s", ">=", 20, "tok/s"),
                ("prefill_tok_s", ">=", 2500, "tok/s"),
                ("first_token_ms", "<=", 2500, "ms"),
                ("steady_state_memory_gb", "<=", 110, "GB"),
            ),
        ),
        Candidate(
            candidate_id="oracle-nemotron3-ultra-hosted",
            role="oracle",
            model="Nemotron 3 Ultra",
            engine="Hermes /model hosted provider path",
            locality="hosted",
            priority=2,
            purpose="hosted fallback when the local Nemotron 3 Super Spark path is unavailable or still under benchmark",
            required_targets=_targets(
                ("first_token_ms", "<=", 3500, "ms"),
                ("tool_plan_quality", ">=", 4, "score"),
            ),
        ),
        Candidate(
            candidate_id="asr-nemotron-speech",
            role="asr",
            model="Nemotron Speech streaming",
            engine="local NeMo/Riva-style streaming ASR",
            locality="local_spark",
            priority=1,
            purpose="durable transcript evidence lane for the oracle, not the reflex driver",
            required_targets=_targets(
                ("asr_delta_ms", "<=", 120, "ms"),
                ("final_transcript_ms", "<=", 900, "ms"),
                ("word_error_rate", "<=", 0.12, "ratio"),
            ),
        ),
        Candidate(
            candidate_id="tts-magpie-local",
            role="tts",
            model="Magpie or Riva-style local TTS",
            engine="local Spark speech service",
            locality="local_spark",
            priority=1,
            purpose="local voice output target for the household/business appliance",
            required_targets=_targets(
                ("tts_first_audio_ms", "<=", 500, "ms"),
                ("underrun_count", "<=", 0, "count"),
            ),
        ),
        Candidate(
            candidate_id="tts-cartesia-cloud-fallback",
            role="tts",
            model="Cartesia cloud TTS",
            engine="Cartesia API",
            locality="hosted",
            priority=2,
            purpose="demo fallback while local TTS is brought up",
            required_targets=_targets(
                ("tts_first_audio_ms", "<=", 700, "ms"),
                ("cutoff_count", "<=", 0, "count"),
            ),
        ),
    ]


def _load_evidence(paths: Iterable[Path]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            evidence.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict) and isinstance(payload.get("evidence"), list):
            evidence.extend(item for item in payload["evidence"] if isinstance(item, dict))
        elif isinstance(payload, dict):
            evidence.append(payload)
    return evidence


def _metric_passes(actual: float, operator: str, expected: float) -> bool:
    if operator == "<=":
        return actual <= expected
    if operator == ">=":
        return actual >= expected
    if operator == "<":
        return actual < expected
    if operator == ">":
        return actual > expected
    if operator == "==":
        return actual == expected
    raise ValueError(f"unsupported operator: {operator}")


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def evaluate_candidate(candidate: Candidate, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    matching = [item for item in evidence if item.get("candidate_id") == candidate.candidate_id]
    if not matching:
        return {
            "candidate_id": candidate.candidate_id,
            "status": "needs_evidence",
            "issues": ["missing_benchmark_evidence"],
            "evidence_count": 0,
            "target_results": [],
        }

    issues: list[str] = []
    target_results: list[dict[str, Any]] = []
    for target in candidate.required_targets:
        values = [
            _coerce_number(item.get("metrics", {}).get(target.metric) if isinstance(item.get("metrics"), dict) else None)
            for item in matching
        ]
        values = [value for value in values if value is not None]
        if not values:
            target_results.append(
                {
                    "metric": target.metric,
                    "status": "missing",
                    "operator": target.operator,
                    "expected": target.value,
                    "actual": None,
                }
            )
            issues.append(f"missing_metric:{target.metric}")
            continue
        actual = max(values) if target.operator in {">=", ">"} else min(values)
        passed = _metric_passes(actual, target.operator, target.value)
        target_results.append(
            {
                "metric": target.metric,
                "status": "pass" if passed else "fail",
                "operator": target.operator,
                "expected": target.value,
                "actual": actual,
            }
        )
        if not passed:
            issues.append(f"target_failed:{target.metric}")

    unverified = [item for item in matching if item.get("verified") is not True]
    if unverified:
        issues.append("evidence_not_verified")

    return {
        "candidate_id": candidate.candidate_id,
        "status": "validated" if not issues else "fails_target",
        "issues": issues,
        "evidence_count": len(matching),
        "target_results": target_results,
    }


def build_matrix(evidence_paths: Iterable[Path] = ()) -> dict[str, Any]:
    candidates = default_candidates()
    evidence = _load_evidence(evidence_paths)
    evaluations = [evaluate_candidate(candidate, evidence) for candidate in candidates]
    role_status: dict[str, str] = {}
    for role in sorted({candidate.role for candidate in candidates}):
        primary_candidate_ids = {
            candidate.candidate_id for candidate in candidates if candidate.role == role and candidate.priority == 1
        }
        role_evaluations = [
            evaluation
            for evaluation in evaluations
            if evaluation["candidate_id"] in primary_candidate_ids
        ]
        role_status[role] = "validated" if any(item["status"] == "validated" for item in role_evaluations) else "needs_evidence"
    return {
        "generated_at": _utc_now(),
        "hardware_target": "1x NVIDIA DGX Spark",
        "policy": {
            "oracle_selected_by": "Hermes /model",
            "reflex_tool_authority": "none_or_low_risk_only",
            "live_spend_default": "dry_run_until_explicit_approval",
        },
        "candidates": [asdict(candidate) for candidate in candidates],
        "evaluations": evaluations,
        "role_status": role_status,
        "ready_for_one_spark_demo": all(status == "validated" for status in role_status.values()),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps DGX Spark Model Matrix",
        "",
        f"- Hardware target: {matrix['hardware_target']}",
        f"- Ready for one-Spark demo: {'yes' if matrix['ready_for_one_spark_demo'] else 'no'}",
        "",
        "## Role Status",
        "",
    ]
    for role, status in sorted(matrix["role_status"].items()):
        lines.append(f"- {role}: {status}")
    lines.extend(["", "## Candidates", ""])
    evaluations = {item["candidate_id"]: item for item in matrix["evaluations"]}
    for candidate in matrix["candidates"]:
        evaluation = evaluations[candidate["candidate_id"]]
        lines.extend(
            [
                f"### {candidate['candidate_id']}",
                "",
                f"- Role: {candidate['role']}",
                f"- Model: {candidate['model']}",
                f"- Engine: {candidate['engine']}",
                f"- Locality: {candidate['locality']}",
                f"- Status: {evaluation['status']}",
                f"- Issues: {', '.join(evaluation['issues']) if evaluation['issues'] else 'none'}",
                f"- Purpose: {candidate['purpose']}",
                "",
                "Targets:",
            ]
        )
        for target in candidate["required_targets"]:
            lines.append(f"- {target['metric']} {target['operator']} {target['value']:g} {target['unit']}")
        lines.append("")
    return "\n".join(lines)


def _evidence_template(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "evidence": [
            {
                "candidate_id": candidate["candidate_id"],
                "hardware": "1x NVIDIA DGX Spark",
                "engine": candidate["engine"],
                "verified": False,
                "metrics": {target["metric"]: None for target in candidate["required_targets"]},
                "notes": "replace null metrics with measured values from the benchmark run",
            }
            for candidate in candidates
        ]
    }


def write_matrix(output_dir: Path, matrix: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "spark-model-matrix.json",
        "markdown": output_dir / "spark-model-matrix.md",
        "evidence_template": output_dir / "spark-benchmark-evidence-template.json",
    }
    _write_json(paths["json"], matrix)
    paths["markdown"].write_text(_markdown(matrix), encoding="utf-8")
    _write_json(paths["evidence_template"], _evidence_template(matrix["candidates"]))
    return {key: str(path) for key, path in paths.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/voiceops-spark-matrix/current"))
    parser.add_argument(
        "--evidence",
        action="append",
        default=[],
        type=Path,
        help="Benchmark evidence JSON file to validate against the matrix. May be repeated.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = build_matrix(args.evidence)
    paths = write_matrix(args.output_dir, matrix)
    print(json.dumps({"ok": True, "output_dir": str(args.output_dir), "artifacts": paths}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
