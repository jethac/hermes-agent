#!/usr/bin/env python3
"""Generate and validate the VoiceOps DGX Spark model matrix.

The matrix is headless and non-invasive. By default it writes the target
candidate matrix and marks every hardware claim as needing evidence. Optional
benchmark evidence JSON files can promote candidates to validated or failed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SPARK_HARDWARE_TARGET = "1x NVIDIA DGX Spark"
PREFERRED_LOCAL_ORACLE_CANDIDATE_ID = "oracle-nemotron3-super-local"
PREFERRED_LOCAL_ORACLE_MODEL = "Nemotron 3 Super"
NON_COUNTING_FALLBACK_ORACLE_MODELS = ("Nemotron 3 Ultra",)
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
            candidate_id="reflex-gemma4-e4b",
            role="reflex",
            model="Gemma 4 E4B audio-native",
            engine="vLLM multimodal audio path or equivalent Spark container",
            locality="local_spark",
            priority=1,
            purpose="larger KAME interface candidate for audio-native intent triage when E2B quality is insufficient",
            required_targets=_targets(
                ("first_token_ms", "<=", 1000, "ms"),
                ("intent_latency_ms", "<=", 1400, "ms"),
                ("steady_state_memory_gb", "<=", 48, "GB"),
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
            evidence.extend(_with_evidence_source(item, path) for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict) and isinstance(payload.get("evidence"), list):
            wrapper_example_only = payload.get("example_only") is True
            evidence.extend(
                _with_evidence_source(item, path, example_only=wrapper_example_only)
                for item in payload["evidence"]
                if isinstance(item, dict)
            )
        elif isinstance(payload, dict):
            evidence.append(_with_evidence_source(payload, path))
    return [*evidence, *_adapt_kame_evidence(evidence)]


def _with_evidence_source(item: dict[str, Any], path: Path, *, example_only: bool = False) -> dict[str, Any]:
    copy = dict(item)
    copy["_evidence_path"] = str(path)
    if example_only:
        copy["example_only"] = True
    return copy


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
    if candidate_id == "reflex-gemma4-e4b":
        return "gemma" in normalized and ("e4b" in normalized or "e-4b" in normalized)
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


def _has_parseable_timezone_timestamp(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _verified(item: dict[str, Any]) -> bool:
    return item.get("verified") is True or item.get("ok") is True


def _base_adapted_evidence(
    source: dict[str, Any],
    *,
    candidate_id: str,
    model: str,
    engine: str,
    locality: str | None = None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "hardware": source.get("hardware"),
        "locality": source.get("locality") or locality,
        "model": model,
        "engine": source.get("engine") or engine,
        "verified": _verified(source),
        "measured_at": _measured_at(source),
        "source_artifact": _source_artifact(source),
        "source_artifact_sha256": source.get("source_artifact_sha256"),
        "metrics": metrics,
        "adapter": source.get("adapter"),
        "module": source.get("module"),
        "provider": source.get("provider"),
        "protocol_smoke_only": source.get("protocol_smoke_only") is True,
        "example_only": source.get("example_only") is True,
        "adapted_from": str(source.get("kind") or ""),
        "_evidence_path": source.get("_evidence_path"),
    }


def _oracle_authority_from_kame(entries: list[dict[str, Any]]) -> dict[str, Any]:
    for entry in entries:
        if (
            entry.get("kind") == "kame_model_assumption_result"
            and entry.get("name") == "oracle_authority"
            and entry.get("validated_by") == "oracle_models_probe"
            and str(entry.get("model") or "").strip()
            and _verified(entry)
        ):
            return {
                "model": str(entry["model"]),
                "oracle_selected_by": "Hermes /model",
                "validated_by": entry.get("validated_by"),
            }
    return {}


def _adapt_kame_evidence(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    oracle_authority = _oracle_authority_from_kame(entries)
    oracle_model = str(oracle_authority.get("model") or "")
    adapted: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("kind") == STACK_SMOKE_KIND:
            continue
        if entry.get("kind") == "kame_smoke_result" and entry.get("name") == "all_local_smoke":
            adapted.append(
                {
                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                    "kind": STACK_SMOKE_KIND,
                    "hardware": entry.get("hardware"),
                    "locality": entry.get("locality"),
                    "verified": _verified(entry),
                    "measured_at": _measured_at(entry),
                    "source_artifact": _source_artifact(entry),
                    "source_artifact_sha256": entry.get("source_artifact_sha256"),
                    "oracle_selected_by": entry.get("oracle_selected_by"),
                    "components": _adapt_kame_stack_components(entry),
                    "metrics": _adapt_kame_stack_metrics(entry),
                    "oracle_authority_routes": _list_values(entry.get("oracle_authority_routes")),
                    "interface_input_sources": _list_values(entry.get("interface_input_sources")),
                    "reflex_providers": _list_values(entry.get("reflex_providers")),
                    "example_only": entry.get("example_only") is True,
                    "adapted_from": "kame_smoke_result",
                    "_evidence_path": entry.get("_evidence_path"),
                }
            )
            continue
        if entry.get("kind") != "kame_benchmark_result":
            continue
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        category = str(entry.get("category") or "")
        normalized_model = str(entry.get("model") or "").lower().replace("-", "")
        if category == "interface" and ("e2b" in normalized_model or "e4b" in normalized_model):
            reflex_candidate_id = "reflex-gemma4-e4b" if "e4b" in normalized_model else "reflex-gemma4-e2b"
            adapted.append(
                _base_adapted_evidence(
                    entry,
                    candidate_id=reflex_candidate_id,
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
            oracle_evidence = _base_adapted_evidence(
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
            oracle_evidence.update(
                {
                    "oracle_selected_by": oracle_authority.get("oracle_selected_by"),
                    "oracle_authority_validated_by": oracle_authority.get("validated_by"),
                }
            )
            adapted.append(oracle_evidence)
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


def _stack_smoke_metric(item: dict[str, Any], metrics: dict[str, Any], key: str) -> float | None:
    return _coerce_number(metrics.get(key) if key in metrics else item.get(key))


def _adapt_kame_stack_components(entry: dict[str, Any]) -> dict[str, bool]:
    components = entry.get("components") if isinstance(entry.get("components"), dict) else {}
    if components:
        return {name: components.get(name) is True for name in STACK_SMOKE_REQUIRED_COMPONENTS}
    return {
        "reflex": "vllm" in set(_list_values(entry.get("reflex_providers"))),
        "oracle": str(entry.get("oracle_selected_by") or "") == "Hermes /model"
        and (_coerce_number(entry.get("oracle_bound_oracle_calls")) or 0) > 0,
        "asr": "native_audio" in set(_list_values(entry.get("interface_input_sources"))),
        "tts": entry.get("tts") is True or entry.get("tts_participated") is True,
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
        if item.get("example_only") is True:
            issues.append("example_only_evidence_not_accepted")
        if item.get("verified") is not True:
            issues.append("evidence_not_verified")
        if str(item.get("schema_version") or "") != EVIDENCE_SCHEMA_VERSION:
            issues.append("missing_schema_version")
        if not _model_matches_candidate(candidate, item.get("model")):
            issues.append("model_mismatch")
        if not str(item.get("source_artifact") or "").strip():
            issues.append("missing_source_artifact")
        else:
            issues.extend(_source_artifact_issues(item))
        if candidate.candidate_id == "oracle-nemotron3-super-local" and item.get("oracle_selected_by") != "Hermes /model":
            issues.append("missing_oracle_authority_proof")
        if candidate.role in {"asr", "tts"}:
            adapter = str(item.get("adapter") or "").strip()
            module = str(item.get("module") or "").strip()
            if item.get("protocol_smoke_only") is True:
                issues.append("protocol_smoke_only_not_accepted")
            if adapter == "loopback_smoke_bridge" or module == "loopback_smoke_bridge":
                issues.append("loopback_speech_evidence_not_accepted")
        if not str(item.get("measured_at") or "").strip():
            issues.append("missing_measured_at")
        elif not _has_parseable_timezone_timestamp(item.get("measured_at")):
            issues.append("invalid_measured_at")
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
        if item.get("example_only") is True:
            issues.append("example_only_evidence_not_accepted")
        if item.get("verified") is not True:
            issues.append("evidence_not_verified")
        if str(item.get("schema_version") or "") != EVIDENCE_SCHEMA_VERSION:
            issues.append("missing_schema_version")
        if not str(item.get("source_artifact") or "").strip():
            issues.append("missing_source_artifact")
        else:
            issues.extend(_source_artifact_issues(item))
        if not str(item.get("measured_at") or "").strip():
            issues.append("missing_measured_at")
        elif not _has_parseable_timezone_timestamp(item.get("measured_at")):
            issues.append("invalid_measured_at")
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
        elif first_audio_ms > 1500:
            issues.append("target_failed:speech_end_to_first_audio_ms")
        barge_in_ms = _coerce_number(metrics.get("barge_in_stop_ms"))
        if barge_in_ms is None:
            issues.append("missing_metric:barge_in_stop_ms")
        elif barge_in_ms > 150:
            issues.append("target_failed:barge_in_stop_ms")
        local_turns = _stack_smoke_metric(item, metrics, "local_turns")
        local_oracle_calls = _stack_smoke_metric(item, metrics, "local_turn_oracle_calls")
        oracle_bound_turns = _stack_smoke_metric(item, metrics, "oracle_bound_turns")
        oracle_bound_calls = _stack_smoke_metric(item, metrics, "oracle_bound_oracle_calls")
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


def _source_artifact_issues(item: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    source_text = str(item.get("source_artifact") or "").strip()
    evidence_path_text = str(item.get("_evidence_path") or "").strip()
    expected_sha256 = str(item.get("source_artifact_sha256") or "").strip().lower()
    if not expected_sha256:
        issues.append("missing_source_artifact_sha256")
    elif len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        issues.append("invalid_source_artifact_sha256")
    if not evidence_path_text:
        return [*issues, "source_artifact_unverified"]
    source_path = Path(source_text).expanduser()
    if not source_path.is_absolute():
        source_path = Path(evidence_path_text).expanduser().parent / source_text
    if not source_path.exists():
        return [*issues, "source_artifact_not_found"]
    if not source_path.is_file():
        return [*issues, "source_artifact_not_file"]
    try:
        source_bytes = source_path.read_bytes()
    except OSError:
        return [*issues, "source_artifact_unreadable"]
    try:
        source_payload = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        source_payload = None
        issues.append("source_artifact_invalid_json")
    if isinstance(source_payload, dict):
        if source_payload.get("example_only") is True:
            issues.append("source_artifact_example_only_not_accepted")
        redaction_policy = str(source_payload.get("redaction_policy") or "").strip()
        if source_payload.get("redacted") is not True and not redaction_policy:
            issues.append("source_artifact_not_redacted")
    if expected_sha256 and len(expected_sha256) == 64 and all(
        character in "0123456789abcdef" for character in expected_sha256
    ):
        actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if actual_sha256 != expected_sha256:
            issues.append("source_artifact_sha256_mismatch")
    return issues


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
                "source_artifact_sha256": None,
                "metrics": {target["metric"]: None for target in candidate["required_targets"]},
                "notes": (
                    "replace null metrics with measured values from the benchmark run; "
                    "hosted fallback evidence must not validate Spark-local readiness; "
                    "source_artifact must point at the raw run output"
                    if candidate["locality"] == "hosted"
                    else "replace null metrics with measured values from the benchmark run; source_artifact must point at the raw run output"
                ),
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
                "source_artifact_sha256": None,
                "oracle_selected_by": "Hermes /model",
                "oracle_authority_routes": list(STACK_SMOKE_REQUIRED_ORACLE_ROUTES),
                "interface_input_sources": ["native_audio"],
                "reflex_providers": ["vllm"],
                "components": {name: None for name in STACK_SMOKE_REQUIRED_COMPONENTS},
                "metrics": {
                    "speech_end_to_first_audio_ms": None,
                    "barge_in_stop_ms": None,
                    "local_turns": None,
                    "local_turn_oracle_calls": None,
                    "oracle_bound_turns": None,
                    "oracle_bound_oracle_calls": None,
                },
                "notes": "Set verified=true only after reflex, oracle, ASR, TTS, and sidecar run together locally on one DGX Spark.",
            }
        ],
    }


def _evidence_example() -> dict[str, Any]:
    return {
        "example_only": True,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "redaction_policy": "example only; replace metrics/source refs with measured DGX Spark artifacts and remove example_only before ingest",
        "evidence": [
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "candidate_id": "reflex-gemma4-e2b",
                "hardware": SPARK_HARDWARE_TARGET,
                "locality": "local_spark",
                "model": "Gemma 4 E2B audio-native",
                "engine": "vLLM multimodal audio path or equivalent Spark container",
                "verified": True,
                "measured_at": "2026-06-29T00:00:00Z",
                "source_artifact": "artifacts/dgx-spark-gemma4-voice-eval/current/reflex-raw.json",
                "source_artifact_sha256": "replace-with-sha256-of-redacted-raw-artifact",
                "metrics": {
                    "first_token_ms": 700,
                    "intent_latency_ms": 1100,
                    "steady_state_memory_gb": 20,
                },
                "example_only": True,
            },
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "candidate_id": "reflex-gemma4-e4b",
                "hardware": SPARK_HARDWARE_TARGET,
                "locality": "local_spark",
                "model": "Gemma 4 E4B audio-native",
                "engine": "vLLM multimodal audio path or equivalent Spark container",
                "verified": True,
                "measured_at": "2026-06-29T00:00:00Z",
                "source_artifact": "artifacts/dgx-spark-gemma4-voice-eval/current/reflex-e4b-raw.json",
                "source_artifact_sha256": "replace-with-sha256-of-redacted-raw-artifact",
                "metrics": {
                    "first_token_ms": 850,
                    "intent_latency_ms": 1250,
                    "steady_state_memory_gb": 32,
                },
                "example_only": True,
            },
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "candidate_id": "oracle-nemotron3-super-local",
                "hardware": SPARK_HARDWARE_TARGET,
                "locality": "local_spark",
                "model": "Nemotron 3 Super",
                "engine": "Hermes /model to local NVIDIA Spark endpoint",
                "verified": True,
                "measured_at": "2026-06-29T00:00:00Z",
                "source_artifact": "artifacts/dgx-spark-gemma4-voice-eval/current/oracle-raw.json",
                "source_artifact_sha256": "replace-with-sha256-of-redacted-raw-artifact",
                "metrics": {
                    "decode_tok_s": 24,
                    "prefill_tok_s": 3100,
                    "first_token_ms": 2100,
                    "steady_state_memory_gb": 86,
                },
                "example_only": True,
            },
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "candidate_id": "asr-nemotron-speech",
                "hardware": SPARK_HARDWARE_TARGET,
                "locality": "local_spark",
                "model": "Nemotron Speech streaming",
                "engine": "local NeMo/Riva-style streaming ASR",
                "verified": True,
                "measured_at": "2026-06-29T00:00:00Z",
                "source_artifact": "artifacts/dgx-spark-gemma4-voice-eval/current/asr-raw.json",
                "source_artifact_sha256": "replace-with-sha256-of-redacted-raw-artifact",
                "metrics": {
                    "asr_delta_ms": 30,
                    "final_transcript_ms": 600,
                    "word_error_rate": 0.08,
                },
                "example_only": True,
            },
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "candidate_id": "tts-magpie-local",
                "hardware": SPARK_HARDWARE_TARGET,
                "locality": "local_spark",
                "model": "Magpie local TTS",
                "engine": "local Spark speech service",
                "verified": True,
                "measured_at": "2026-06-29T00:00:00Z",
                "source_artifact": "artifacts/dgx-spark-gemma4-voice-eval/current/tts-raw.json",
                "source_artifact_sha256": "replace-with-sha256-of-redacted-raw-artifact",
                "metrics": {
                    "tts_first_audio_ms": 200,
                    "underrun_count": 0,
                },
                "example_only": True,
            },
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "kind": STACK_SMOKE_KIND,
                "hardware": SPARK_HARDWARE_TARGET,
                "locality": "local_spark",
                "verified": True,
                "measured_at": "2026-06-29T00:00:00Z",
                "source_artifact": "artifacts/dgx-spark-gemma4-voice-eval/current/all-local-stack-smoke.json",
                "source_artifact_sha256": "replace-with-sha256-of-redacted-raw-artifact",
                "oracle_selected_by": "Hermes /model",
                "oracle_authority_routes": list(STACK_SMOKE_REQUIRED_ORACLE_ROUTES),
                "interface_input_sources": ["native_audio"],
                "reflex_providers": ["vllm"],
                "components": {name: True for name in STACK_SMOKE_REQUIRED_COMPONENTS},
                "metrics": {
                    "speech_end_to_first_audio_ms": 900,
                    "barge_in_stop_ms": 90,
                    "local_turns": 2,
                    "local_turn_oracle_calls": 0,
                    "oracle_bound_turns": 4,
                    "oracle_bound_oracle_calls": 4,
                },
                "example_only": True,
            },
        ],
    }


def write_evidence_scaffold(output_dir: Path) -> dict[str, Path]:
    scaffold_dir = output_dir / "spark-benchmark-scaffold"
    sources_dir = scaffold_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    scaffold = _evidence_example()
    source_names = {
        "reflex-gemma4-e2b": "reflex-gemma4-e2b-raw.json",
        "reflex-gemma4-e4b": "reflex-gemma4-e4b-raw.json",
        "oracle-nemotron3-super-local": "oracle-nemotron3-super-raw.json",
        "asr-nemotron-speech": "asr-nemotron-speech-raw.json",
        "tts-magpie-local": "tts-magpie-local-raw.json",
        STACK_SMOKE_KIND: "all-local-stack-smoke-raw.json",
    }
    paths: dict[str, Path] = {}
    for item in scaffold["evidence"]:
        source_key = str(item.get("candidate_id") or item.get("kind") or "unknown")
        source_name = source_names[source_key]
        source_path = sources_dir / source_name
        source_payload = {
            "schema_version": "voiceops.spark.raw_benchmark_artifact.v1",
            "example_only": True,
            "source_key": source_key,
            "hardware": SPARK_HARDWARE_TARGET,
            "locality": "local_spark",
            "redacted": True,
            "redaction_policy": "example only; replace with redacted raw benchmark output, no secrets or private transcripts",
            "summary": f"Replace this with measured DGX Spark raw output for {source_key}.",
        }
        _write_json(source_path, source_payload)
        item["source_artifact"] = f"sources/{source_name}"
        item["source_artifact_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        paths[f"scaffold_source_{source_key}"] = source_path

    scaffold_path = scaffold_dir / "spark-benchmark-evidence.json"
    _write_json(scaffold_path, scaffold)
    paths["evidence_scaffold"] = scaffold_path
    return paths


def _closure_plan(matrix: dict[str, Any]) -> dict[str, Any]:
    evaluations = {item["candidate_id"]: item for item in matrix["evaluations"]}
    missing_roles = [
        f"{role}:{status}"
        for role, status in sorted(matrix["role_status"].items())
        if status != "validated"
    ]
    return {
        "schema_version": "voiceops.milestone4.spark_matrix_closure.v1",
        "artifact_id": "voiceops-m4-spark-matrix-closure",
        "milestone": "milestone_4_local_spark_stack_matrix",
        "generated_at": matrix["generated_at"],
        "mode": {
            "artifact_only": True,
            "supplied_artifacts_only": True,
            "spark_execution": False,
            "network_io": False,
            "env_secret_reads": False,
            "live_spend": False,
            "provider_provisioning": False,
            "outbound_calls": False,
        },
        "hardware_target": matrix["hardware_target"],
        "ready": matrix["ready_for_one_spark_demo"],
        "ready_for_one_spark_demo": matrix["ready_for_one_spark_demo"],
        "status": "complete" if matrix["ready_for_one_spark_demo"] else "needs_external_evidence",
        "source_matrix_artifact": "spark-model-matrix.json",
        "missing_gates": [
            *missing_roles,
            *([] if matrix["stack_smoke"]["status"] == "validated" else ["all_local_stack_smoke"]),
        ],
        "missing_roles": missing_roles,
        "stack_smoke_status": matrix["stack_smoke"]["status"],
        "stack_smoke_issues": matrix["stack_smoke"]["issues"],
        "evidence_template": "spark-benchmark-evidence-template.json",
        "evidence_example": "spark-benchmark-evidence.example.json",
        "evidence_scaffold": "spark-benchmark-scaffold/spark-benchmark-evidence.json",
        "matrix_artifact": "spark-model-matrix.json",
        "required_candidate_fields": [
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
        "evidence_contract": {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "preferred_local_oracle_candidate_id": PREFERRED_LOCAL_ORACLE_CANDIDATE_ID,
            "preferred_local_oracle_model": PREFERRED_LOCAL_ORACLE_MODEL,
            "non_counting_fallback_oracle_models": list(NON_COUNTING_FALLBACK_ORACLE_MODELS),
            "source_artifacts_must_exist": True,
            "source_artifact_resolution": "absolute paths or paths relative to the supplied benchmark evidence file",
            "source_artifact_readable": True,
            "source_artifact_sha256_must_match": True,
            "measured_at_timezone_required": True,
            "example_only_accepted": False,
            "hosted_fallback_counts_for_one_spark_readiness": False,
        },
        "required_stack_smoke_fields": [
            "schema_version",
            "kind",
            "components",
            "hardware",
            "locality",
            "measured_at",
            "oracle_selected_by",
            "oracle_authority_routes",
            "interface_input_sources",
            "reflex_providers",
            "metrics.speech_end_to_first_audio_ms",
            "metrics.barge_in_stop_ms",
            "metrics.local_turns",
            "metrics.local_turn_oracle_calls",
            "metrics.oracle_bound_turns",
            "metrics.oracle_bound_oracle_calls",
            "source_artifact",
            "source_artifact_sha256",
            "verified",
        ],
        "candidate_closure": [
            {
                "candidate_id": candidate["candidate_id"],
                "role": candidate["role"],
                "priority": candidate["priority"],
                "locality": candidate["locality"],
                "model": candidate["model"],
                "engine": candidate["engine"],
                "status": evaluations[candidate["candidate_id"]]["status"],
                "issues": evaluations[candidate["candidate_id"]]["issues"],
                "required_targets": candidate["required_targets"],
                "proof": (
                    "Measured local DGX Spark evidence required for role readiness."
                    if candidate["locality"] == "local_spark" and candidate["priority"] == 1
                    else "Fallback evidence is useful for demos but does not satisfy one-Spark local readiness."
                ),
            }
            for candidate in matrix["candidates"]
        ],
        "all_local_stack_smoke": {
            "kind": STACK_SMOKE_KIND,
            "required_components": list(STACK_SMOKE_REQUIRED_COMPONENTS),
            "required_oracle_routes": list(STACK_SMOKE_REQUIRED_ORACLE_ROUTES),
            "required_interface_input_source": "native_audio",
            "required_reflex_provider": "vllm",
            "target_metrics": {
                "speech_end_to_first_audio_ms": {"operator": "<=", "value": 1500, "unit": "ms"},
                "barge_in_stop_ms": {"operator": "<=", "value": 150, "unit": "ms"},
            },
        },
        "benchmark_evidence_shape": {
            "evidence": [
                {
                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                    "candidate_id": "oracle-nemotron3-super-local",
                    "hardware": SPARK_HARDWARE_TARGET,
                    "locality": "local_spark",
                    "model": "Nemotron 3 Super",
                    "engine": "Hermes /model to local NVIDIA Spark endpoint",
                    "verified": True,
                    "measured_at": "2026-06-29T00:00:00Z",
                    "source_artifact": "artifacts/dgx-spark-gemma4-voice-eval/current/oracle-raw.json",
                    "metrics": {
                        "decode_tok_s": 24,
                        "prefill_tok_s": 3100,
                        "first_token_ms": 2100,
                        "steady_state_memory_gb": 86,
                    },
                },
                {
                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                    "kind": STACK_SMOKE_KIND,
                    "hardware": SPARK_HARDWARE_TARGET,
                    "locality": "local_spark",
                    "verified": True,
                    "measured_at": "2026-06-29T00:00:00Z",
                    "source_artifact": "artifacts/dgx-spark-gemma4-voice-eval/current/all-local-stack-smoke.json",
                    "oracle_selected_by": "Hermes /model",
                    "oracle_authority_routes": list(STACK_SMOKE_REQUIRED_ORACLE_ROUTES),
                    "interface_input_sources": ["native_audio"],
                    "reflex_providers": ["vllm"],
                    "components": {name: True for name in STACK_SMOKE_REQUIRED_COMPONENTS},
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
        "rerun_commands": {
            "matrix_only": "uv run python scripts/voiceops_spark_matrix.py --output-dir artifacts/voiceops-spark-matrix/current",
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
        "operator_must_not": [
            "claim one-Spark readiness from hosted or multi-Spark Nemotron 3 Ultra fallback evidence or cloud TTS fallback evidence",
            "mark benchmark evidence verified without raw source artifacts",
            "treat the matrix template or example as measured evidence",
        ],
        "completion_signal": "ready_for_one_spark_demo is true, all primary local role_status values are validated, and all_local_stack_smoke is validated",
    }


def _closure_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Milestone 4 Spark Matrix Closure",
        "",
        f"- Status: {plan['status']}",
        f"- Hardware target: {plan['hardware_target']}",
        f"- Ready for one-Spark demo: {'yes' if plan['ready_for_one_spark_demo'] else 'no'}",
        f"- Missing roles: {', '.join(plan['missing_roles']) if plan['missing_roles'] else 'none'}",
        f"- Stack smoke: {plan['stack_smoke_status']}",
        f"- Stack smoke issues: {', '.join(plan['stack_smoke_issues']) if plan['stack_smoke_issues'] else 'none'}",
        "",
        "## Evidence Artifacts",
        "",
        f"- Template: `{plan['evidence_template']}`",
        f"- Example: `{plan['evidence_example']}`",
        f"- Scaffold: `{plan['evidence_scaffold']}`",
        f"- Matrix: `{plan['matrix_artifact']}`",
        "",
        "## Evidence Contract",
        "",
    ]
    for key, value in sorted(plan["evidence_contract"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
        "## Candidate Closure",
        "",
        ]
    )
    for item in plan["candidate_closure"]:
        target_text = ", ".join(
            f"{target['metric']} {target['operator']} {target['value']:g} {target['unit']}"
            for target in item["required_targets"]
        )
        lines.extend(
            [
                f"### {item['candidate_id']}",
                "",
                f"- Role: {item['role']}",
                f"- Locality: {item['locality']}",
                f"- Status: {item['status']}",
                f"- Issues: {', '.join(item['issues']) if item['issues'] else 'none'}",
                f"- Targets: {target_text}",
                f"- Proof: {item['proof']}",
                "",
            ]
        )
    lines.extend(["## All-Local Stack Smoke", ""])
    smoke = plan["all_local_stack_smoke"]
    lines.append(f"- Kind: `{smoke['kind']}`")
    lines.append(f"- Components: {', '.join(smoke['required_components'])}")
    lines.append(f"- Oracle routes: {', '.join(smoke['required_oracle_routes'])}")
    lines.append(f"- Interface input source: {smoke['required_interface_input_source']}")
    lines.append(f"- Reflex provider: {smoke['required_reflex_provider']}")
    lines.extend(["", "## Benchmark Evidence Shape", "", "```json"])
    lines.append(json.dumps(plan["benchmark_evidence_shape"], indent=2, sort_keys=True))
    lines.extend(["```"])
    lines.extend(["", "## Rerun Commands", ""])
    for label, command in plan["rerun_commands"].items():
        lines.append(f"- {label}: `{command}`")
    lines.extend(["", "## Do Not", ""])
    lines.extend(f"- {item}" for item in plan["operator_must_not"])
    lines.append("")
    return "\n".join(lines)


def _operator_runbook(plan: dict[str, Any]) -> str:
    commands = plan["rerun_commands"]
    lines = [
        "# VoiceOps DGX Spark Operator Runbook",
        "",
        "- Purpose: collect the measured one-Spark evidence required by Milestone 4 without hand-editing readiness.",
        f"- Hardware target: {plan['hardware_target']}",
        f"- Current status: {plan['status']}",
        f"- Ready for one-Spark demo: {'yes' if plan['ready_for_one_spark_demo'] else 'no'}",
        f"- Missing gates: {', '.join(plan['missing_gates']) if plan['missing_gates'] else 'none'}",
        "",
        "## Safety Boundary",
        "",
        "- This runbook does not install models, start servers, read secrets, spend money, provision services, send messages, or place calls.",
        "- It records how to collect and ingest local DGX Spark benchmark evidence.",
        "- Nemotron 3 Super is the preferred one-Spark oracle candidate for this plan.",
        "- Hosted or multi-Spark Nemotron 3 Ultra evidence can be useful fallback context, but it cannot satisfy one-Spark readiness.",
        "",
        "## Collection Sequence",
        "",
        "1. Start the local KAME stack on the DGX Spark: reflex, Hermes oracle endpoint selected through `/model`, ASR, TTS, and realtime voice sidecar.",
        "   If the generated KAME stack still points ASR/TTS at `loopback_smoke_bridge`, treat those services as protocol-only smoke checks.",
        "   They cannot satisfy VoiceOps local ASR/TTS evidence; replace `HERMES_DGX_SPARK_ASR_MODULE`, `HERMES_DGX_SPARK_ASR_ADAPTER`, `HERMES_DGX_SPARK_TTS_MODULE`, `HERMES_DGX_SPARK_TTS_ADAPTER`, and the speech models with production local Nemotron Speech, Magpie, Riva-style, or equivalent Spark providers before collecting verified evidence.",
        "2. Run the repo-side evaluator on the DGX Spark and preserve every raw output artifact it writes.",
        "",
        "```bash",
        commands["dgx_eval"],
        "```",
        "",
        "3. Replace the scaffold source artifacts under `spark-benchmark-scaffold/sources/` with redacted measured raw outputs.",
        "4. Fill `spark-benchmark-scaffold/spark-benchmark-evidence.json` with measured metrics, real `source_artifact` refs, matching `source_artifact_sha256` values, `verified: true`, and no `example_only` markers.",
        "5. Re-run the matrix validator against the measured evidence.",
        "",
        "```bash",
        commands["with_evidence"],
        "```",
        "",
        "6. Re-index the full VoiceOps plan with the same measured evidence file.",
        "",
        "```bash",
        commands["plan_index"],
        "```",
        "",
        "## Required Evidence",
        "",
        f"- Candidate schema: `{plan['evidence_contract']['schema_version']}`",
        f"- Evidence scaffold: `{plan['evidence_scaffold']}`",
        f"- Matrix artifact: `{plan['matrix_artifact']}`",
        "- Source artifacts must exist and resolve relative to the supplied evidence file.",
        "- Source artifacts must be redacted UTF-8 JSON and must not carry `example_only: true`.",
        "- `source_artifact_sha256` must match the referenced source artifact bytes.",
        "- `example_only: true` evidence is rejected.",
        "- `loopback_smoke_bridge` evidence is protocol-only and must remain unverified for local ASR/TTS roles.",
        "",
        "## Role Evidence",
        "",
    ]
    for item in plan["candidate_closure"]:
        if item["priority"] != 1 or item["locality"] != "local_spark":
            continue
        target_text = ", ".join(
            f"{target['metric']} {target['operator']} {target['value']:g} {target['unit']}"
            for target in item["required_targets"]
        )
        lines.extend(
            [
                f"### {item['role']}: {item['candidate_id']}",
                "",
                f"- Model: {item['model']}",
                f"- Engine: {item['engine']}",
                f"- Current status: {item['status']}",
                f"- Targets: {target_text}",
                f"- Current issues: {', '.join(item['issues']) if item['issues'] else 'none'}",
                "",
            ]
        )
    smoke = plan["all_local_stack_smoke"]
    lines.extend(
        [
            "## All-Local Stack Smoke",
            "",
            f"- Kind: `{smoke['kind']}`",
            f"- Required components: {', '.join(smoke['required_components'])}",
            f"- Required oracle routes: {', '.join(smoke['required_oracle_routes'])}",
            f"- Required interface input source: `{smoke['required_interface_input_source']}`",
            f"- Required reflex provider: `{smoke['required_reflex_provider']}`",
            "- Required metrics: `speech_end_to_first_audio_ms <= 1500`, `barge_in_stop_ms <= 150`, `local_turn_oracle_calls == 0`, and `oracle_bound_oracle_calls >= oracle_bound_turns`.",
            "",
            "## Do Not",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in plan["operator_must_not"])
    lines.extend(
        [
            "",
            "## Completion Signal",
            "",
            plan["completion_signal"],
            "",
        ]
    )
    return "\n".join(lines)


def write_matrix(output_dir: Path, matrix: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    closure_plan = _closure_plan(matrix)
    paths = {
        "json": output_dir / "spark-model-matrix.json",
        "markdown": output_dir / "spark-model-matrix.md",
        "evidence_template": output_dir / "spark-benchmark-evidence-template.json",
        "evidence_example": output_dir / "spark-benchmark-evidence.example.json",
        "closure_json": output_dir / "spark-matrix-closure-plan.json",
        "closure_markdown": output_dir / "spark-matrix-closure-plan.md",
        "operator_runbook": output_dir / "spark-operator-runbook.md",
    }
    _write_json(paths["json"], matrix)
    paths["markdown"].write_text(_markdown(matrix), encoding="utf-8")
    _write_json(paths["evidence_template"], _evidence_template(matrix["candidates"]))
    _write_json(paths["evidence_example"], _evidence_example())
    paths.update(write_evidence_scaffold(output_dir))
    _write_json(paths["closure_json"], closure_plan)
    paths["closure_markdown"].write_text(_closure_markdown(closure_plan), encoding="utf-8")
    paths["operator_runbook"].write_text(_operator_runbook(closure_plan), encoding="utf-8")
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
