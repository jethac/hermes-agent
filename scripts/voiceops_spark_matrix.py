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


SPARK_HARDWARE_TARGET = "1x NVIDIA DGX Spark"
EVIDENCE_SCHEMA_VERSION = "voiceops.spark_benchmark_evidence.v1"
STACK_SMOKE_KIND = "voiceops_spark_stack_smoke"
STACK_SMOKE_REQUIRED_COMPONENTS = ("reflex", "oracle", "asr", "tts", "sidecar")
STACK_SMOKE_REQUIRED_ORACLE_ROUTES = ("tools", "files", "memory", "project_context")


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
    return [*evidence, *_adapt_kame_evidence(evidence)]


def _matches_hardware(value: Any) -> bool:
    return str(value or "").strip().lower() in {
        SPARK_HARDWARE_TARGET.lower(),
        "1x dgx spark",
        "single dgx spark",
    }


def _model_matches_candidate(candidate: Candidate, model: Any) -> bool:
    normalized = str(model or "").strip().lower()
    if not normalized:
        return False
    candidate_id = candidate.candidate_id
    if candidate_id == "reflex-gemma4-e2b":
        return "gemma" in normalized and ("e2b" in normalized or "e-2b" in normalized)
    if candidate_id == "oracle-nemotron3-super-local":
        return "nemotron" in normalized and "super" in normalized
    if candidate_id == "oracle-nemotron3-ultra-hosted":
        return "nemotron" in normalized and "ultra" in normalized
    if candidate_id == "asr-nemotron-speech":
        return "nemotron" in normalized and "speech" in normalized
    if candidate_id == "tts-magpie-local":
        return "magpie" in normalized or "riva" in normalized
    if candidate_id == "tts-cartesia-cloud-fallback":
        return "cartesia" in normalized
    return normalized == candidate.model.lower()


def _source_artifact(item: dict[str, Any]) -> str | None:
    for key in ("source_artifact", "source", "artifact", "run_artifact"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return None


def _measured_at(item: dict[str, Any]) -> str | None:
    for key in ("measured_at", "collected_at", "generated_at", "timestamp"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return None


def _verified(item: dict[str, Any]) -> bool:
    return item.get("verified") is True or item.get("ok") is True


def _base_adapted_evidence(
    source: dict[str, Any],
    *,
    candidate_id: str,
    model: str,
    engine: str,
    locality: str = "local_spark",
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "hardware": source.get("hardware") or SPARK_HARDWARE_TARGET,
        "locality": locality,
        "model": model,
        "engine": source.get("engine") or engine,
        "verified": _verified(source),
        "measured_at": _measured_at(source),
        "source_artifact": _source_artifact(source),
        "metrics": metrics,
        "adapted_from": str(source.get("kind") or ""),
    }


def _oracle_model_from_kame(entries: list[dict[str, Any]]) -> str:
    for entry in entries:
        if (
            entry.get("kind") == "kame_model_assumption_result"
            and entry.get("name") == "oracle_authority"
            and str(entry.get("model") or "").strip()
        ):
            return str(entry["model"])
    return ""


def _adapt_kame_evidence(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    oracle_model = _oracle_model_from_kame(entries)
    adapted: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("kind") == STACK_SMOKE_KIND:
            continue
        if entry.get("kind") == "kame_smoke_result" and entry.get("name") == "all_local_smoke":
            adapted.append(
                {
                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                    "kind": STACK_SMOKE_KIND,
                    "hardware": entry.get("hardware") or SPARK_HARDWARE_TARGET,
                    "locality": "local_spark",
                    "verified": _verified(entry),
                    "measured_at": _measured_at(entry),
                    "source_artifact": _source_artifact(entry),
                    "oracle_selected_by": entry.get("oracle_selected_by") or "Hermes /model",
                    "components": _adapt_kame_stack_components(entry),
                    "metrics": _adapt_kame_stack_metrics(entry),
                    "oracle_authority_routes": _list_values(entry.get("oracle_authority_routes")),
                    "interface_input_sources": _list_values(entry.get("interface_input_sources")),
                    "reflex_providers": _list_values(entry.get("reflex_providers")),
                    "adapted_from": "kame_smoke_result",
                }
            )
            continue
        if entry.get("kind") != "kame_benchmark_result":
            continue
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        category = str(entry.get("category") or "")
        if category == "interface" and str(entry.get("model") or "").lower().replace("-", "").find("e2b") >= 0:
            adapted.append(
                _base_adapted_evidence(
                    entry,
                    candidate_id="reflex-gemma4-e2b",
                    model=str(entry.get("model") or ""),
                    engine=str(entry.get("engine") or "vLLM multimodal audio path"),
                    metrics={
                        "first_token_ms": metrics.get("kame_interface_model_request_ms"),
                        "intent_latency_ms": metrics.get("speech_end_to_interface_decision_p90_ms")
                        or metrics.get("speech_end_to_interface_decision_ms"),
                        "steady_state_memory_gb": metrics.get("steady_state_memory_gb")
                        or metrics.get("memory_gb"),
                    },
                )
            )
        elif category == "oracle":
            adapted.append(
                _base_adapted_evidence(
                    entry,
                    candidate_id="oracle-nemotron3-super-local",
                    model=str(entry.get("model") or oracle_model),
                    engine=str(entry.get("engine") or "Hermes /model to local NVIDIA Spark endpoint"),
                    metrics={
                        "decode_tok_s": metrics.get("decode_tok_s"),
                        "prefill_tok_s": metrics.get("prefill_tok_s"),
                        "first_token_ms": metrics.get("oracle_accepted_to_first_token_ms"),
                        "steady_state_memory_gb": metrics.get("steady_state_memory_gb")
                        or metrics.get("memory_gb"),
                    },
                )
            )
        elif category == "speech" and entry.get("role") == "oracle_verbatim_asr":
            literal_accuracy = _coerce_number(metrics.get("literal_accuracy_names_numbers_code"))
            adapted.append(
                _base_adapted_evidence(
                    entry,
                    candidate_id="asr-nemotron-speech",
                    model=str(entry.get("model") or ""),
                    engine=str(entry.get("engine") or "local NeMo/Riva-style streaming ASR"),
                    metrics={
                        "asr_delta_ms": metrics.get("speech_end_to_asr_final_ms"),
                        "final_transcript_ms": metrics.get("speech_end_to_asr_final_p90_ms")
                        or metrics.get("speech_end_to_asr_final_ms"),
                        "word_error_rate": metrics.get("word_error_rate")
                        if metrics.get("word_error_rate") is not None
                        else (round(1.0 - literal_accuracy, 4) if literal_accuracy is not None else None),
                    },
                )
            )
        elif category == "speech" and entry.get("role") == "tts":
            adapted.append(
                _base_adapted_evidence(
                    entry,
                    candidate_id="tts-magpie-local",
                    model=str(entry.get("model") or ""),
                    engine=str(entry.get("engine") or "local Spark speech service"),
                    metrics={
                        "tts_first_audio_ms": metrics.get("tts_request_to_first_audio_ms"),
                        "underrun_count": metrics.get("underrun_count"),
                    },
                )
            )
    return adapted


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


def _list_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item or "").strip() for item in value if str(item or "").strip()]


def _adapt_kame_stack_components(entry: dict[str, Any]) -> dict[str, bool]:
    components = entry.get("components") if isinstance(entry.get("components"), dict) else {}
    if components:
        return {name: components.get(name) is True for name in STACK_SMOKE_REQUIRED_COMPONENTS}
    return {
        "reflex": "vllm" in set(_list_values(entry.get("reflex_providers"))),
        "oracle": str(entry.get("oracle_selected_by") or "Hermes /model") == "Hermes /model"
        and (_coerce_number(entry.get("oracle_bound_oracle_calls")) or 0) > 0,
        "asr": "native_audio" in set(_list_values(entry.get("interface_input_sources"))),
        "tts": True,
        "sidecar": entry.get("ok") is True,
    }


def _adapt_kame_stack_metrics(entry: dict[str, Any]) -> dict[str, Any]:
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    return {
        "speech_end_to_first_audio_ms": metrics.get("speech_end_to_first_audio_ms"),
        "barge_in_stop_ms": metrics.get("barge_in_stop_ms"),
        "local_turns": entry.get("local_turns"),
        "local_turn_oracle_calls": entry.get("local_turn_oracle_calls"),
        "oracle_bound_turns": entry.get("oracle_bound_turns"),
        "oracle_bound_oracle_calls": entry.get("oracle_bound_oracle_calls"),
    }


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
    for item in matching:
        if item.get("verified") is not True:
            issues.append("evidence_not_verified")
        if str(item.get("schema_version") or "") != EVIDENCE_SCHEMA_VERSION:
            issues.append("missing_schema_version")
        if not _model_matches_candidate(candidate, item.get("model")):
            issues.append("model_mismatch")
        if not str(item.get("source_artifact") or "").strip():
            issues.append("missing_source_artifact")
        if not str(item.get("measured_at") or "").strip():
            issues.append("missing_measured_at")
        locality = str(item.get("locality") or "").strip()
        if locality != candidate.locality:
            issues.append("locality_mismatch")
        if candidate.locality == "local_spark" and not _matches_hardware(item.get("hardware")):
            issues.append("hardware_mismatch")

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

    return {
        "candidate_id": candidate.candidate_id,
        "status": "validated" if not issues else "fails_target",
        "issues": sorted(set(issues)),
        "evidence_count": len(matching),
        "target_results": target_results,
    }


def evaluate_stack_smoke(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    matching = [item for item in evidence if item.get("kind") == STACK_SMOKE_KIND]
    if not matching:
        return {
            "status": "needs_evidence",
            "issues": ["missing_all_local_stack_smoke"],
            "evidence_count": 0,
        }

    issues: list[str] = []
    for item in matching:
        if item.get("verified") is not True:
            issues.append("evidence_not_verified")
        if str(item.get("schema_version") or "") != EVIDENCE_SCHEMA_VERSION:
            issues.append("missing_schema_version")
        if not _matches_hardware(item.get("hardware")):
            issues.append("hardware_mismatch")
        if str(item.get("locality") or "") != "local_spark":
            issues.append("locality_mismatch")
        if str(item.get("oracle_selected_by") or "") != "Hermes /model":
            issues.append("oracle_not_selected_by_model_flow")
        components = item.get("components") if isinstance(item.get("components"), dict) else {}
        missing_components = [name for name in STACK_SMOKE_REQUIRED_COMPONENTS if components.get(name) is not True]
        if missing_components:
            issues.append("missing_components:" + ",".join(missing_components))
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        first_audio_ms = _coerce_number(metrics.get("speech_end_to_first_audio_ms"))
        if first_audio_ms is None:
            issues.append("missing_metric:speech_end_to_first_audio_ms")
        elif first_audio_ms > 3000:
            issues.append("target_failed:speech_end_to_first_audio_ms")
        barge_in_ms = _coerce_number(metrics.get("barge_in_stop_ms"))
        if barge_in_ms is None:
            issues.append("missing_metric:barge_in_stop_ms")
        elif barge_in_ms > 150:
            issues.append("target_failed:barge_in_stop_ms")
        if item.get("adapted_from") == "kame_smoke_result":
            local_turns = _coerce_number(metrics.get("local_turns"))
            local_oracle_calls = _coerce_number(metrics.get("local_turn_oracle_calls"))
            oracle_bound_turns = _coerce_number(metrics.get("oracle_bound_turns"))
            oracle_bound_calls = _coerce_number(metrics.get("oracle_bound_oracle_calls"))
            if local_turns is None or local_turns < 1:
                issues.append("missing_or_failed:local_turns")
            if local_oracle_calls is None or local_oracle_calls != 0:
                issues.append("target_failed:local_turn_oracle_calls")
            if oracle_bound_turns is None or oracle_bound_turns < 1:
                issues.append("missing_or_failed:oracle_bound_turns")
            if oracle_bound_calls is None or oracle_bound_turns is None or oracle_bound_calls < oracle_bound_turns:
                issues.append("target_failed:oracle_bound_oracle_calls")

            routes = set(_list_values(item.get("oracle_authority_routes")))
            missing_routes = sorted(set(STACK_SMOKE_REQUIRED_ORACLE_ROUTES).difference(routes))
            if missing_routes:
                issues.append("missing_oracle_authority_routes:" + ",".join(missing_routes))

            input_sources = set(_list_values(item.get("interface_input_sources")))
            if "native_audio" not in input_sources:
                issues.append("missing_interface_input_source:native_audio")

            reflex_providers = set(_list_values(item.get("reflex_providers")))
            if "vllm" not in reflex_providers:
                issues.append("missing_reflex_provider:vllm")

    return {
        "status": "validated" if not issues else "fails_target",
        "issues": sorted(set(issues)),
        "evidence_count": len(matching),
    }


def build_matrix(evidence_paths: Iterable[Path] = ()) -> dict[str, Any]:
    candidates = default_candidates()
    evidence = _load_evidence(evidence_paths)
    evaluations = [evaluate_candidate(candidate, evidence) for candidate in candidates]
    stack_smoke = evaluate_stack_smoke(evidence)
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
        "hardware_target": SPARK_HARDWARE_TARGET,
        "policy": {
            "oracle_selected_by": "Hermes /model",
            "reflex_tool_authority": "none_or_low_risk_only",
            "live_spend_default": "dry_run_until_explicit_approval",
        },
        "candidates": [asdict(candidate) for candidate in candidates],
        "evaluations": evaluations,
        "stack_smoke": stack_smoke,
        "role_status": role_status,
        "ready_for_one_spark_demo": (
            all(status == "validated" for status in role_status.values())
            and stack_smoke["status"] == "validated"
        ),
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
    lines.extend(
        [
            f"- all_local_stack_smoke: {matrix['stack_smoke']['status']}",
            "",
            "## Stack Smoke",
            "",
            f"- Status: {matrix['stack_smoke']['status']}",
            f"- Issues: {', '.join(matrix['stack_smoke']['issues']) if matrix['stack_smoke']['issues'] else 'none'}",
        ]
    )
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
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "candidate_id": candidate["candidate_id"],
                "hardware": SPARK_HARDWARE_TARGET if candidate["locality"] == "local_spark" else "hosted",
                "locality": candidate["locality"],
                "model": candidate["model"],
                "engine": candidate["engine"],
                "verified": False,
                "measured_at": None,
                "source_artifact": None,
                "metrics": {target["metric"]: None for target in candidate["required_targets"]},
                "notes": "replace null metrics with measured values from the benchmark run; source_artifact must point at the raw run output",
            }
            for candidate in candidates
        ]
        + [
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "kind": STACK_SMOKE_KIND,
                "hardware": SPARK_HARDWARE_TARGET,
                "locality": "local_spark",
                "verified": False,
                "measured_at": None,
                "source_artifact": None,
                "oracle_selected_by": "Hermes /model",
                "components": {name: None for name in STACK_SMOKE_REQUIRED_COMPONENTS},
                "metrics": {
                    "speech_end_to_first_audio_ms": None,
                    "barge_in_stop_ms": None,
                },
                "notes": "Set verified=true only after reflex, oracle, ASR, TTS, and sidecar run together locally on one DGX Spark.",
            }
        ],
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
