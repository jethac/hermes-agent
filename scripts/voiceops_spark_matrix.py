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
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, MutableMapping


SPARK_HARDWARE_TARGET = "1x NVIDIA DGX Spark"
PREFERRED_LOCAL_ORACLE_CANDIDATE_ID = "oracle-nemotron3-super-local"
PREFERRED_LOCAL_ORACLE_MODEL = "Nemotron 3 Super"
NON_COUNTING_FALLBACK_ORACLE_MODELS = ("Nemotron 3 Ultra",)
EVIDENCE_SCHEMA_VERSION = "voiceops.spark_benchmark_evidence.v1"
STACK_SMOKE_KIND = "voiceops_spark_stack_smoke"
PRIMARY_LOCAL_ROLES = ("interpreter", "oracle", "reflex", "tts")
STACK_SMOKE_REQUIRED_COMPONENTS = ("reflex", "interpreter", "oracle", "tts", "sidecar")
STACK_SMOKE_REQUIRED_ORACLE_ROUTES = ("tools", "files", "memory", "project_context")
SPARK_BENCHMARK_EVIDENCE_NOT_BEFORE = dt.datetime(2026, 6, 29, tzinfo=dt.timezone.utc)
SPARK_BENCHMARK_SCAFFOLD_EVIDENCE = (
    "artifacts/voiceops-spark-matrix/current/spark-benchmark-scaffold/spark-benchmark-evidence.json"
)
COLLECTOR_ATTESTATION_REQUIRED_FIELDS = (
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
)
SOURCE_SECRET_KEY_RE = re.compile(
    r"(^|[_\-.])("
    r"api[_\-.]?key|access[_\-.]?token|refresh[_\-.]?token|id[_\-.]?token|auth[_\-.]?token|"
    r"authorization|bearer|password|passwd|secret|private[_\-.]?key|client[_\-.]?secret|webhook[_\-.]?secret"
    r")($|[_\-.])",
    re.IGNORECASE,
)
SOURCE_PHONE_KEY_RE = re.compile(
    r"(^|[_\-.])(phone|tel|telephone|mobile|sms|caller|callee|from[_\-.]?number|to[_\-.]?number)($|[_\-.])",
    re.IGNORECASE,
)
SOURCE_SECRET_VALUE_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\bsk-[A-Za-z0-9_-]{20,}\b"
    r"|\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b"
    r"|\bgithub_pat_[A-Za-z0-9_]{20,}\b"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\bAIza[0-9A-Za-z_-]{35}\b"
    r"|\bxox[baprs]-[0-9A-Za-z-]{10,}\b"
    r"|\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b",
)
SOURCE_PHONE_VALUE_RE = re.compile(
    r"(?<![\w])(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?![\w])"
    r"|(?<![\w])\+[1-9]\d{9,14}(?![\w])"
)


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
            candidate_id="reflex-moshi-s2s",
            role="reflex",
            model="Moshi/PersonaPlex-class low-latency S2S",
            engine="local S2S or timing/noise-gated reflex runtime",
            locality="local_spark",
            priority=1,
            purpose="low-latency KAME interface for acknowledgements, barge-in, turn-taking, and rough transcript hypotheses",
            required_targets=_targets(
                ("ack_latency_ms", "<=", 350, "ms"),
                ("barge_in_stop_ms", "<=", 150, "ms"),
                ("steady_state_memory_gb", "<=", 24, "GB"),
            ),
        ),
        Candidate(
            candidate_id="interpreter-gemma4-e2b",
            role="interpreter",
            model="Gemma 4 E2B audio-native interpreter",
            engine="vLLM multimodal audio path or equivalent Spark container",
            locality="local_spark",
            priority=1,
            purpose="raw-audio interpreter that adjudicates clipped audio plus labeled transcript hypotheses",
            required_targets=_targets(
                ("audio_interpretation_ms", "<=", 1200, "ms"),
                ("evidence_patch_ms", "<=", 1500, "ms"),
                ("steady_state_memory_gb", "<=", 32, "GB"),
            ),
        ),
        Candidate(
            candidate_id="interpreter-gemma4-e4b",
            role="interpreter",
            model="Gemma 4 E4B audio-native interpreter",
            engine="vLLM multimodal audio path or equivalent Spark container",
            locality="local_spark",
            priority=1,
            purpose="larger raw-audio interpreter candidate when E2B quality is insufficient",
            required_targets=_targets(
                ("audio_interpretation_ms", "<=", 1500, "ms"),
                ("evidence_patch_ms", "<=", 1800, "ms"),
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


def _load_evidence(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    evidence: list[dict[str, Any]] = []
    issues: list[str] = []
    for path in paths:
        resolved = path.expanduser().resolve(strict=False)
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except FileNotFoundError:
            issues.append(f"evidence_file_not_found:{resolved}")
            continue
        except json.JSONDecodeError as exc:
            issues.append(f"evidence_json_parse_failed:{resolved}:{exc.msg}")
            continue
        except OSError as exc:
            issues.append(f"evidence_file_unreadable:{resolved}:{exc.strerror or exc}")
            continue
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
        else:
            issues.append(f"evidence_root_must_be_object_or_list:{resolved}")
    return [*evidence, *_adapt_kame_evidence(evidence)], issues


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
    if candidate_id == "reflex-moshi-s2s":
        return any(token in normalized for token in ("moshi", "personaplex", "s2s", "voiceclaw", "openclaw"))
    if candidate_id == "interpreter-gemma4-e2b":
        return "gemma" in normalized and ("e2b" in normalized or "e-2b" in normalized)
    if candidate_id == "interpreter-gemma4-e4b":
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


def _parse_timezone_timestamp(value: Any) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _has_parseable_timezone_timestamp(value: Any) -> bool:
    return _parse_timezone_timestamp(value) is not None


def _timestamp_before_evidence_window(value: Any) -> bool:
    parsed = _parse_timezone_timestamp(value)
    return parsed is not None and parsed < SPARK_BENCHMARK_EVIDENCE_NOT_BEFORE


def _verified(item: dict[str, Any]) -> bool:
    return item.get("verified") is True or item.get("ok") is True


def _collector_attestation(item: dict[str, Any]) -> dict[str, Any] | None:
    attestation = item.get("collector_attestation") or item.get("collector_provenance")
    return dict(attestation) if isinstance(attestation, dict) else None


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
        **({"collector_attestation": _collector_attestation(source)} if _collector_attestation(source) else {}),
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
                    "interpreter_providers": _list_values(entry.get("interpreter_providers")),
                    "auxiliary_transcript_sources": _list_values(entry.get("auxiliary_transcript_sources")),
                    "example_only": entry.get("example_only") is True,
                    "adapted_from": "kame_smoke_result",
                    "_evidence_path": entry.get("_evidence_path"),
                    **({"collector_attestation": _collector_attestation(entry)} if _collector_attestation(entry) else {}),
                }
            )
            continue
        if entry.get("kind") != "kame_benchmark_result":
            continue
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        category = str(entry.get("category") or "")
        normalized_model = str(entry.get("model") or "").lower().replace("-", "")
        if category in {"interface", "reflex"} and any(token in normalized_model for token in ("moshi", "personaplex", "s2s")):
            adapted.append(
                _base_adapted_evidence(
                    entry,
                    candidate_id="reflex-moshi-s2s",
                    model=str(entry.get("model") or ""),
                    engine=str(entry.get("engine") or "local S2S or timing/noise-gated reflex runtime"),
                    metrics={
                        "ack_latency_ms": metrics.get("ack_latency_ms")
                        or metrics.get("speech_end_to_ack_ms")
                        or metrics.get("speech_end_to_first_audio_ms"),
                        "barge_in_stop_ms": metrics.get("barge_in_stop_ms"),
                        "steady_state_memory_gb": metrics.get("steady_state_memory_gb")
                        or metrics.get("memory_gb"),
                    },
                )
            )
        elif category in {"interface", "interpreter"} and ("e2b" in normalized_model or "e4b" in normalized_model):
            interpreter_candidate_id = "interpreter-gemma4-e4b" if "e4b" in normalized_model else "interpreter-gemma4-e2b"
            adapted.append(
                _base_adapted_evidence(
                    entry,
                    candidate_id=interpreter_candidate_id,
                    model=str(entry.get("model") or ""),
                    engine=str(entry.get("engine") or "vLLM multimodal audio path"),
                    metrics={
                        "audio_interpretation_ms": metrics.get("audio_interpretation_ms")
                        or metrics.get("kame_interface_model_request_ms"),
                        "evidence_patch_ms": metrics.get("evidence_patch_ms")
                        or metrics.get("speech_end_to_interface_decision_p90_ms")
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
    reflex_providers = {value.lower() for value in _list_values(entry.get("reflex_providers"))}
    interpreter_providers = {value.lower() for value in _list_values(entry.get("interpreter_providers"))}
    input_sources = {value.lower() for value in _list_values(entry.get("interface_input_sources"))}
    return {
        "reflex": bool(reflex_providers.intersection({"moshi", "personaplex", "s2s", "voiceclaw", "openclaw", "timing", "noise_gate"})),
        "interpreter": "native_audio" in input_sources
        and bool(interpreter_providers.intersection({"vllm", "gemma", "gemma4"})),
        "oracle": str(entry.get("oracle_selected_by") or "") == "Hermes /model"
        and (_coerce_number(entry.get("oracle_bound_oracle_calls")) or 0) > 0,
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

    record_results: list[dict[str, Any]] = []
    for index, item in enumerate(matching):
        item_issues: list[str] = []
        if item.get("example_only") is True:
            item_issues.append("example_only_evidence_not_accepted")
        if item.get("verified") is not True:
            item_issues.append("evidence_not_verified")
        if str(item.get("schema_version") or "") != EVIDENCE_SCHEMA_VERSION:
            item_issues.append("missing_schema_version")
        if not _model_matches_candidate(candidate, item.get("model")):
            item_issues.append("model_mismatch")
        if not str(item.get("source_artifact") or "").strip():
            item_issues.append("missing_source_artifact")
        else:
            item_issues.extend(_source_artifact_issues(item))
        if candidate.locality == "local_spark":
            item_issues.extend(_collector_attestation_issues(item))
        if candidate.candidate_id == "oracle-nemotron3-super-local" and item.get("oracle_selected_by") != "Hermes /model":
            item_issues.append("missing_oracle_authority_proof")
        if candidate.role in {"asr", "tts"}:
            adapter = str(item.get("adapter") or "").strip()
            module = str(item.get("module") or "").strip()
            if item.get("protocol_smoke_only") is True:
                item_issues.append("protocol_smoke_only_not_accepted")
            if adapter == "loopback_smoke_bridge" or module == "loopback_smoke_bridge":
                item_issues.append("loopback_speech_evidence_not_accepted")
        if not str(item.get("measured_at") or "").strip():
            item_issues.append("missing_measured_at")
        elif not _has_parseable_timezone_timestamp(item.get("measured_at")):
            item_issues.append("invalid_measured_at")
        elif _timestamp_before_evidence_window(item.get("measured_at")):
            item_issues.append("stale_measured_at")
        locality = str(item.get("locality") or "").strip()
        if locality != candidate.locality:
            item_issues.append("locality_mismatch")
        if candidate.locality == "local_spark" and not _matches_hardware(item.get("hardware")):
            item_issues.append("hardware_mismatch")

        target_results: list[dict[str, Any]] = []
        for target in candidate.required_targets:
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            actual = _coerce_number(metrics.get(target.metric))
            if actual is None:
                target_results.append(
                    {
                        "metric": target.metric,
                        "status": "missing",
                        "operator": target.operator,
                        "expected": target.value,
                        "actual": None,
                    }
                )
                item_issues.append(f"missing_metric:{target.metric}")
                continue
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
                item_issues.append(f"target_failed:{target.metric}")

        record_results.append(
            {
                "record_index": index,
                "source_evidence": item.get("_evidence_path"),
                "status": "validated" if not item_issues else "rejected",
                "issues": sorted(set(item_issues)),
                "target_results": target_results,
            }
        )

    passing_record = next((record for record in record_results if record["status"] == "validated"), None)
    issues = [] if passing_record else sorted(
        {
            "no_single_evidence_record_satisfies_targets",
            *(issue for record in record_results for issue in record["issues"]),
        }
    )
    target_results = passing_record["target_results"] if passing_record else (
        record_results[0]["target_results"] if record_results else []
    )

    return {
        "candidate_id": candidate.candidate_id,
        "status": "validated" if passing_record else "fails_target",
        "issues": sorted(set(issues)),
        "evidence_count": len(matching),
        "target_results": target_results,
        "record_results": record_results,
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
            issues.extend(_stack_smoke_source_artifact_contract_issues(item))
        issues.extend(_collector_attestation_issues(item))
        if not str(item.get("measured_at") or "").strip():
            issues.append("missing_measured_at")
        elif not _has_parseable_timezone_timestamp(item.get("measured_at")):
            issues.append("invalid_measured_at")
        elif _timestamp_before_evidence_window(item.get("measured_at")):
            issues.append("stale_measured_at")
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
        normalized_reflex_providers = {value.lower() for value in reflex_providers}
        if not normalized_reflex_providers.intersection({"moshi", "personaplex", "s2s", "voiceclaw", "openclaw", "timing", "noise_gate"}):
            issues.append("missing_reflex_provider:s2s_or_timing")

        interpreter_providers = {value.lower() for value in _list_values(item.get("interpreter_providers"))}
        if not interpreter_providers.intersection({"vllm", "gemma", "gemma4"}):
            issues.append("missing_interpreter_provider:gemma_audio")

    return {
        "status": "validated" if not issues else "fails_target",
        "issues": sorted(set(issues)),
        "evidence_count": len(matching),
    }


def _stack_smoke_source_artifact_contract_issues(item: dict[str, Any]) -> list[str]:
    source_payload = _read_source_artifact_payload(item)
    if not isinstance(source_payload, dict):
        return []
    turns = source_payload.get("kame_turns")
    if not isinstance(turns, list):
        turns = source_payload.get("turns")
    if not isinstance(turns, list) or not turns:
        return ["source_artifact_missing_kame_turns"]

    issues: set[str] = set()
    local_turn_seen = False
    oracle_bound_turn_seen = False
    for raw_turn in turns:
        if not isinstance(raw_turn, dict):
            issues.add("source_artifact_kame_turn_not_object")
            continue
        if not str(raw_turn.get("audio_segment_ref") or "").strip():
            issues.add("source_artifact_kame_turn_missing_audio_segment_ref")
        if not _valid_audio_time_range_ms(raw_turn.get("audio_time_range_ms")):
            issues.add("source_artifact_kame_turn_missing_audio_time_range_ms")
        if not _transcript_hypothesis_is_labeled(raw_turn.get("reflex_transcript_hypothesis")):
            issues.add("source_artifact_reflex_transcript_not_hypothesis")
        if not _auxiliary_transcript_hypotheses_are_labeled(raw_turn.get("auxiliary_transcript_hypotheses")):
            issues.add("source_artifact_auxiliary_transcript_not_hypothesis")

        route = str(raw_turn.get("route") or raw_turn.get("path") or "").strip().lower()
        oracle_called = _turn_oracle_called(raw_turn)
        if route == "local" and oracle_called is False:
            local_turn_seen = True
        if route in {"defer", "oracle_direct", "oracle_bound"} or oracle_called is True:
            oracle_bound_turn_seen = True
            if not _interpreter_corrected_transcript(raw_turn):
                issues.add("source_artifact_missing_interpreter_corrected_transcript")
            if not _gemma_interpreter_evidence_present(raw_turn):
                issues.add("source_artifact_missing_gemma_interpreter_evidence")
            if not _tool_critical_text_source_is_authoritative(raw_turn):
                issues.add("source_artifact_tool_critical_text_not_interpreter_or_oracle_judgment")
    if not local_turn_seen:
        issues.add("source_artifact_missing_local_kame_turn")
    if not oracle_bound_turn_seen:
        issues.add("source_artifact_missing_oracle_bound_kame_turn")
    return sorted(issues)


def _read_source_artifact_payload(item: dict[str, Any]) -> Any:
    source_text = str(item.get("source_artifact") or "").strip()
    evidence_path_text = str(item.get("_evidence_path") or "").strip()
    if not source_text or not evidence_path_text:
        return None
    source_path = Path(source_text).expanduser()
    if not source_path.is_absolute():
        source_path = Path(evidence_path_text).expanduser().parent / source_text
    try:
        return json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _valid_audio_time_range_ms(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    start = _coerce_number(value[0])
    end = _coerce_number(value[1])
    return start is not None and end is not None and 0 <= start < end


def _transcript_hypothesis_is_labeled(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return str(value.get("authority") or "").strip().lower() == "hypothesis" and bool(
        str(value.get("text") or value.get("summary") or "").strip()
    )


def _auxiliary_transcript_hypotheses_are_labeled(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, list):
        return False
    return all(_transcript_hypothesis_is_labeled(item) for item in value)


def _turn_oracle_called(turn: dict[str, Any]) -> bool | None:
    value = turn.get("oracle_called")
    if isinstance(value, bool):
        return value
    calls = _coerce_number(turn.get("oracle_calls"))
    if calls is not None:
        return calls > 0
    return None


def _interpreter_corrected_transcript(turn: dict[str, Any]) -> str:
    direct = str(turn.get("interpreter_corrected_transcript") or "").strip()
    if direct:
        return direct
    evidence = turn.get("interpreter_evidence")
    if isinstance(evidence, dict):
        return str(evidence.get("corrected_transcript") or evidence.get("transcript") or "").strip()
    return ""


def _gemma_interpreter_evidence_present(turn: dict[str, Any]) -> bool:
    evidence = turn.get("interpreter_evidence")
    if not isinstance(evidence, dict):
        return False
    source = str(evidence.get("source") or evidence.get("provider") or "").strip().lower()
    if "gemma" not in source:
        return False
    return bool(_interpreter_corrected_transcript(turn))


def _tool_critical_text_source_is_authoritative(turn: dict[str, Any]) -> bool:
    source = str(turn.get("tool_critical_text_source") or turn.get("oracle_text_source") or "").strip().lower()
    return source in {"interpreter", "interpreter_corrected_transcript", "gemma_interpreter", "oracle", "oracle_judgment"}


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
        return [*issues, "source_artifact_not_found", f"source_artifact_not_found_path:{source_path.resolve(strict=False)}"]
    if not source_path.is_file():
        return [*issues, "source_artifact_not_file"]
    try:
        source_bytes = source_path.read_bytes()
    except OSError:
        return [*issues, "source_artifact_unreadable"]
    source_payload, safety_issues = _source_artifact_safety_issues(source_bytes)
    issues.extend(safety_issues)
    if isinstance(source_payload, dict):
        issues.extend(_source_artifact_identity_issues(item, source_payload))
    if expected_sha256 and len(expected_sha256) == 64 and all(
        character in "0123456789abcdef" for character in expected_sha256
    ):
        actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if actual_sha256 != expected_sha256:
            issues.append("source_artifact_sha256_mismatch")
        attestation = item.get("collector_attestation") or item.get("collector_provenance")
        if isinstance(attestation, dict):
            redacted_sha256 = str(attestation.get("redacted_artifact_sha256") or "").strip().lower()
            if _valid_sha256(redacted_sha256) and redacted_sha256 != actual_sha256:
                issues.append("collector_attestation_redacted_sha256_mismatch")
    return issues


def _source_artifact_safety_issues(source_bytes: bytes) -> tuple[Any, list[str]]:
    try:
        source_payload = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, ["source_artifact_invalid_json", "source_artifact_not_redacted"]
    if not isinstance(source_payload, dict):
        return source_payload, ["source_artifact_root_not_object", "source_artifact_not_redacted"]

    issues: list[str] = []
    if source_payload.get("example_only") is True:
        issues.append("source_artifact_example_only_not_accepted")
    if source_payload.get("redacted") is not True:
        issues.append("source_artifact_not_redacted")
    issues.extend(_source_artifact_content_safety_issues(source_payload))
    return source_payload, sorted(set(issues))


def _source_artifact_content_safety_issues(value: Any, *, key: str = "") -> list[str]:
    issues: set[str] = set()
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            issues.update(_source_artifact_content_safety_issues(child_value, key=str(child_key or "")))
        return sorted(issues)
    if isinstance(value, list):
        for child_value in value:
            issues.update(_source_artifact_content_safety_issues(child_value, key=key))
        return sorted(issues)

    text = str(value or "").strip()
    if not text:
        return []
    if SOURCE_SECRET_KEY_RE.search(key) and not _source_value_is_redacted_placeholder(text):
        issues.add("source_artifact_contains_likely_secret")
    if SOURCE_PHONE_KEY_RE.search(key) and not _source_value_is_redacted_placeholder(text):
        issues.add("source_artifact_contains_phone_like_value")
    if SOURCE_SECRET_VALUE_RE.search(text):
        issues.add("source_artifact_contains_likely_secret")
    if SOURCE_PHONE_VALUE_RE.search(text):
        issues.add("source_artifact_contains_phone_like_value")
    return sorted(issues)


def _source_value_is_redacted_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"redacted", "[redacted]", "<redacted>", "***", "xxxx", "xxxxx", "null", "none"}


def _source_artifact_identity_issues(item: dict[str, Any], source_payload: dict[str, Any]) -> list[str]:
    expected_candidate = str(item.get("candidate_id") or "").strip()
    expected_kind = str(item.get("kind") or "").strip()
    expected = expected_candidate or expected_kind
    if not expected:
        return []
    actual = set(_source_identity_values(source_payload.get("source_key")))
    actual.update(_source_identity_values(source_payload.get("source_keys")))
    actual.update(_source_identity_values(source_payload.get("candidate_id")))
    actual.update(_source_identity_values(source_payload.get("candidate_ids")))
    actual.update(_source_identity_values(source_payload.get("kind")))
    actual.update(_source_identity_values(source_payload.get("kinds")))
    actual = {str(value).strip() for value in actual if str(value).strip()}
    if not actual:
        return ["source_artifact_identity_missing"]
    if expected not in actual:
        return ["source_artifact_identity_mismatch"]
    return []


def _source_identity_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item or "").strip() for item in value if str(item or "").strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _resolve_source_artifact_path(item: MutableMapping[str, Any], evidence_path: Path) -> Path | None:
    source_text = str(item.get("source_artifact") or "").strip()
    if not source_text:
        return None
    source_path = Path(source_text).expanduser()
    if not source_path.is_absolute():
        source_path = evidence_path.expanduser().parent / source_text
    return source_path


def _spark_refresh_items(payload: Any) -> list[MutableMapping[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, MutableMapping)]
    if isinstance(payload, MutableMapping) and isinstance(payload.get("evidence"), list):
        return [item for item in payload["evidence"] if isinstance(item, MutableMapping)]
    if isinstance(payload, MutableMapping):
        return [payload]
    return []


def refresh_spark_source_hashes(path: Path) -> dict[str, Any]:
    """Refresh benchmark source hashes in a local Spark evidence file."""

    target = path.expanduser()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _spark_refresh_result(target, issues=["target file not found"], updates=[])
    except json.JSONDecodeError as exc:
        return _spark_refresh_result(target, issues=[f"target JSON parse failed: {exc.msg}"], updates=[])
    except OSError as exc:
        return _spark_refresh_result(target, issues=[f"target file unreadable: {exc.strerror or exc}"], updates=[])

    items = _spark_refresh_items(payload)
    if not items:
        return _spark_refresh_result(target, issues=["target root must be an object, list, or object with evidence list"], updates=[])

    issues: list[str] = []
    updates: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        label = str(item.get("candidate_id") or item.get("kind") or f"item-{index}")
        source_path = _resolve_source_artifact_path(item, target)
        if source_path is None:
            issues.append(f"{label}:missing_source_artifact")
            continue
        if not source_path.exists():
            issues.append(f"{label}:source_artifact_not_found:{source_path.resolve(strict=False)}")
            continue
        if not source_path.is_file():
            issues.append(f"{label}:source_artifact_not_file:{source_path.resolve(strict=False)}")
            continue
        try:
            source_bytes = source_path.read_bytes()
        except OSError as exc:
            issues.append(f"{label}:source_artifact_unreadable:{exc.strerror or exc}")
            continue
        source_payload, safety_issues = _source_artifact_safety_issues(source_bytes)
        if isinstance(source_payload, dict):
            safety_issues.extend(_source_artifact_identity_issues(dict(item), source_payload))
        if safety_issues:
            issues.extend(f"{label}:{issue}" for issue in sorted(set(safety_issues)))
            continue
        new_sha256 = hashlib.sha256(source_bytes).hexdigest()
        previous_sha256 = str(item.get("source_artifact_sha256") or "")
        item["source_artifact_sha256"] = new_sha256
        attestation = item.get("collector_attestation")
        previous_attestation_sha256: str | None = None
        attestation_changed = False
        if isinstance(attestation, MutableMapping):
            previous_attestation_sha256 = str(attestation.get("redacted_artifact_sha256") or "")
            attestation["redacted_artifact_sha256"] = new_sha256
            attestation_changed = previous_attestation_sha256 != new_sha256
        updates.append(
            {
                "item": label,
                "source_artifact": str(item.get("source_artifact") or ""),
                "source_artifact_path": str(source_path),
                "previous_sha256": previous_sha256,
                "source_artifact_sha256": new_sha256,
                "previous_collector_attestation_redacted_artifact_sha256": previous_attestation_sha256,
                "collector_attestation_redacted_artifact_sha256": new_sha256 if isinstance(attestation, MutableMapping) else None,
                "collector_attestation_changed": attestation_changed,
                "changed": previous_sha256 != new_sha256,
            }
        )
    if updates and not issues:
        _write_json(target, payload)
    return _spark_refresh_result(target, issues=issues, updates=updates)


def _spark_refresh_result(path: Path, *, issues: list[str], updates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ok": not issues,
        "schema_version": "voiceops.spark_evidence_hash_refresh.v1",
        "artifact_id": "voiceops-m4-spark-evidence-hash-refresh",
        "generated_at": _utc_now(),
        "target_path": str(path),
        "artifact_writes": bool(updates and not issues),
        "network_io": False,
        "spark_execution": False,
        "env_secret_reads": False,
        "updates": updates,
        "issues": issues,
    }


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _collector_attestation_command_arg_is_sensitive(value: str) -> bool:
    return bool(SOURCE_SECRET_VALUE_RE.search(value) or SOURCE_PHONE_VALUE_RE.search(value))


def _collector_attestation_issues(item: dict[str, Any]) -> list[str]:
    attestation = item.get("collector_attestation") or item.get("collector_provenance")
    if not isinstance(attestation, dict):
        return ["missing_collector_attestation"]
    issues: list[str] = []
    if attestation.get("example_only") is True:
        issues.append("collector_attestation_example_only_not_accepted")
    for field in COLLECTOR_ATTESTATION_REQUIRED_FIELDS:
        if field not in attestation:
            issues.append(f"collector_attestation_missing:{field}")
    for field in ("collector_name", "collector_version", "run_id", "git_commit"):
        value = str(attestation.get(field) or "").strip()
        if not value or value.lower() in {"placeholder", "example", "replace-me", "unknown"}:
            issues.append(f"collector_attestation_invalid:{field}")
    command_argv = attestation.get("command_argv")
    if not isinstance(command_argv, list) or not command_argv or not all(isinstance(part, str) and part for part in command_argv):
        issues.append("collector_attestation_invalid:command_argv")
    elif any(_collector_attestation_command_arg_is_sensitive(part) for part in command_argv):
        issues.append("collector_attestation_secret_or_phone_like_command_argv")
    started_at = _parse_timezone_timestamp(attestation.get("started_at"))
    finished_at = _parse_timezone_timestamp(attestation.get("finished_at"))
    if started_at is None:
        issues.append("collector_attestation_invalid:started_at")
    elif started_at < SPARK_BENCHMARK_EVIDENCE_NOT_BEFORE:
        issues.append("collector_attestation_stale:started_at")
    if finished_at is None:
        issues.append("collector_attestation_invalid:finished_at")
    elif finished_at < SPARK_BENCHMARK_EVIDENCE_NOT_BEFORE:
        issues.append("collector_attestation_stale:finished_at")
    if started_at is not None and finished_at is not None and started_at > finished_at:
        issues.append("collector_attestation_invalid:timestamp_window")
    for field in ("raw_artifact_sha256", "redacted_artifact_sha256", "parent_manifest_sha256"):
        if not _valid_sha256(attestation.get(field)):
            issues.append(f"collector_attestation_invalid:{field}")
    source_sha256 = str(item.get("source_artifact_sha256") or "").strip().lower()
    redacted_sha256 = str(attestation.get("redacted_artifact_sha256") or "").strip().lower()
    if _valid_sha256(source_sha256) and _valid_sha256(redacted_sha256) and redacted_sha256 != source_sha256:
        issues.append("collector_attestation_redacted_sha256_mismatch")
    return issues


def build_matrix(evidence_paths: Iterable[Path] = ()) -> dict[str, Any]:
    candidates = default_candidates()
    evidence, evidence_load_issues = _load_evidence(evidence_paths)
    evaluations = [evaluate_candidate(candidate, evidence) for candidate in candidates]
    stack_smoke = evaluate_stack_smoke(evidence)
    role_status: dict[str, str] = {}
    for role in sorted(PRIMARY_LOCAL_ROLES):
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
        "evidence_load_issues": evidence_load_issues,
        "role_status": role_status,
        "ready_for_one_spark_demo": (
            not evidence_load_issues
            and
            all(status == "validated" for status in role_status.values())
            and stack_smoke["status"] == "validated"
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_example_scaffold_source(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("example_only") is True


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
    collector_template = {
        "collector_name": None,
        "collector_version": None,
        "run_id": None,
        "command_argv": [],
        "git_commit": None,
        "started_at": None,
        "finished_at": None,
        "raw_artifact_sha256": None,
        "redacted_artifact_sha256": None,
        "parent_manifest_sha256": None,
    }
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
                "collector_attestation": dict(collector_template),
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
                "collector_attestation": dict(collector_template),
                "oracle_selected_by": "Hermes /model",
                "oracle_authority_routes": list(STACK_SMOKE_REQUIRED_ORACLE_ROUTES),
                "interface_input_sources": ["native_audio"],
                "reflex_providers": ["moshi"],
                "interpreter_providers": ["vllm", "gemma"],
                "auxiliary_transcript_sources": ["moshi_hypothesis", "classic_asr_fallback_optional"],
                "components": {name: None for name in STACK_SMOKE_REQUIRED_COMPONENTS},
                "metrics": {
                    "speech_end_to_first_audio_ms": None,
                    "barge_in_stop_ms": None,
                    "local_turns": None,
                    "local_turn_oracle_calls": None,
                    "oracle_bound_turns": None,
                    "oracle_bound_oracle_calls": None,
                },
                "notes": "Set verified=true only after reflex, interpreter, oracle, TTS, and sidecar run together locally on one DGX Spark. ASR/Moshi transcripts are auxiliary hypothesis evidence, not readiness-critical truth.",
            }
        ],
    }


def _evidence_example() -> dict[str, Any]:
    def example_attestation(name: str, redacted_sha256: str) -> dict[str, Any]:
        return {
            "example_only": True,
            "collector_name": "dgx_spark_gemma4_voice_eval",
            "collector_version": "example",
            "run_id": f"example-{name}-run",
            "command_argv": ["scripts/dgx_spark_gemma4_voice_eval.sh"],
            "git_commit": "0" * 40,
            "started_at": "2026-06-29T00:00:00Z",
            "finished_at": "2026-06-29T00:00:01Z",
            "raw_artifact_sha256": "0" * 64,
            "redacted_artifact_sha256": redacted_sha256,
            "parent_manifest_sha256": "0" * 64,
        }

    return {
        "example_only": True,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "redaction_policy": "example only; replace metrics/source refs with measured DGX Spark artifacts and remove example_only before ingest",
        "evidence": [
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "candidate_id": "reflex-moshi-s2s",
                "hardware": SPARK_HARDWARE_TARGET,
                "locality": "local_spark",
                "model": "Moshi/PersonaPlex-class low-latency S2S",
                "engine": "local S2S or timing/noise-gated reflex runtime",
                "verified": True,
                "measured_at": "2026-06-29T00:00:00Z",
                "source_artifact": "artifacts/dgx-spark-gemma4-voice-eval/current/reflex-moshi-raw.json",
                "source_artifact_sha256": "replace-with-sha256-of-redacted-raw-artifact",
                "collector_attestation": example_attestation(
                    "reflex-moshi-s2s",
                    "replace-with-sha256-of-redacted-raw-artifact",
                ),
                "metrics": {
                    "ack_latency_ms": 250,
                    "barge_in_stop_ms": 90,
                    "steady_state_memory_gb": 16,
                },
                "example_only": True,
            },
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "candidate_id": "interpreter-gemma4-e2b",
                "hardware": SPARK_HARDWARE_TARGET,
                "locality": "local_spark",
                "model": "Gemma 4 E2B audio-native interpreter",
                "engine": "vLLM multimodal audio path or equivalent Spark container",
                "verified": True,
                "measured_at": "2026-06-29T00:00:00Z",
                "source_artifact": "artifacts/dgx-spark-gemma4-voice-eval/current/interpreter-gemma4-e2b-raw.json",
                "source_artifact_sha256": "replace-with-sha256-of-redacted-raw-artifact",
                "collector_attestation": example_attestation(
                    "interpreter-gemma4-e2b",
                    "replace-with-sha256-of-redacted-raw-artifact",
                ),
                "metrics": {
                    "audio_interpretation_ms": 900,
                    "evidence_patch_ms": 1200,
                    "steady_state_memory_gb": 24,
                },
                "example_only": True,
            },
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "candidate_id": "interpreter-gemma4-e4b",
                "hardware": SPARK_HARDWARE_TARGET,
                "locality": "local_spark",
                "model": "Gemma 4 E4B audio-native interpreter",
                "engine": "vLLM multimodal audio path or equivalent Spark container",
                "verified": True,
                "measured_at": "2026-06-29T00:00:00Z",
                "source_artifact": "artifacts/dgx-spark-gemma4-voice-eval/current/interpreter-gemma4-e4b-raw.json",
                "source_artifact_sha256": "replace-with-sha256-of-redacted-raw-artifact",
                "collector_attestation": example_attestation(
                    "interpreter-gemma4-e4b",
                    "replace-with-sha256-of-redacted-raw-artifact",
                ),
                "metrics": {
                    "audio_interpretation_ms": 1100,
                    "evidence_patch_ms": 1500,
                    "steady_state_memory_gb": 36,
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
                "collector_attestation": example_attestation(
                    "oracle-nemotron3-super-local",
                    "replace-with-sha256-of-redacted-raw-artifact",
                ),
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
                "collector_attestation": example_attestation(
                    "asr-nemotron-speech",
                    "replace-with-sha256-of-redacted-raw-artifact",
                ),
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
                "collector_attestation": example_attestation(
                    "tts-magpie-local",
                    "replace-with-sha256-of-redacted-raw-artifact",
                ),
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
                "collector_attestation": example_attestation(
                    STACK_SMOKE_KIND,
                    "replace-with-sha256-of-redacted-raw-artifact",
                ),
                "oracle_selected_by": "Hermes /model",
                "oracle_authority_routes": list(STACK_SMOKE_REQUIRED_ORACLE_ROUTES),
                "interface_input_sources": ["native_audio"],
                "reflex_providers": ["moshi"],
                "interpreter_providers": ["vllm", "gemma"],
                "auxiliary_transcript_sources": ["moshi_hypothesis", "classic_asr_fallback_optional"],
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
        "reflex-moshi-s2s": "reflex-moshi-s2s-raw.json",
        "interpreter-gemma4-e2b": "interpreter-gemma4-e2b-raw.json",
        "interpreter-gemma4-e4b": "interpreter-gemma4-e4b-raw.json",
        "oracle-nemotron3-super-local": "oracle-nemotron3-super-raw.json",
        "asr-nemotron-speech": "asr-nemotron-speech-raw.json",
        "tts-magpie-local": "tts-magpie-local-raw.json",
        STACK_SMOKE_KIND: "all-local-stack-smoke-raw.json",
    }
    expected_source_names = set(source_names.values())
    for stale_source in sources_dir.glob("*.json"):
        if stale_source.name in expected_source_names:
            continue
        if _is_example_scaffold_source(stale_source):
            stale_source.unlink()
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
        if source_key == STACK_SMOKE_KIND:
            source_payload["kame_turns"] = _stack_smoke_source_artifact_shape()
        _write_json(source_path, source_payload)
        item["source_artifact"] = f"sources/{source_name}"
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        item["source_artifact_sha256"] = source_sha256
        if isinstance(item.get("collector_attestation"), dict):
            item["collector_attestation"]["redacted_artifact_sha256"] = source_sha256
        paths[f"scaffold_source_{source_key}"] = source_path

    scaffold_path = scaffold_dir / "spark-benchmark-evidence.json"
    _write_json(scaffold_path, scaffold)
    paths["evidence_scaffold"] = scaffold_path
    return paths


def _stack_smoke_source_artifact_shape() -> list[dict[str, Any]]:
    return [
        {
            "turn_id": "replace-with-redacted-local-turn-id",
            "route": "local",
            "oracle_called": False,
            "audio_segment_ref": "artifact://replace-with-redacted-local-audio-ref",
            "audio_time_range_ms": [0, 800],
            "reflex_transcript_hypothesis": {
                "authority": "hypothesis",
                "source": "moshi_or_reflex",
                "text": "[redacted local transcript hypothesis]",
            },
            "auxiliary_transcript_hypotheses": [],
        },
        {
            "turn_id": "replace-with-redacted-oracle-bound-turn-id",
            "route": "defer",
            "oracle_called": True,
            "oracle_calls": 1,
            "audio_segment_ref": "artifact://replace-with-redacted-oracle-bound-audio-ref",
            "audio_time_range_ms": [1000, 3200],
            "reflex_transcript_hypothesis": {
                "authority": "hypothesis",
                "source": "moshi_or_reflex",
                "text": "[redacted reflex transcript hypothesis]",
            },
            "auxiliary_transcript_hypotheses": [
                {
                    "authority": "hypothesis",
                    "source": "classic_asr_fallback_optional",
                    "text": "[redacted optional auxiliary transcript hypothesis]",
                }
            ],
            "interpreter_evidence": {
                "source": "gemma_interpreter",
                "corrected_transcript": "[redacted interpreter correction]",
                "confidence": 0.9,
                "disagreements": ["redacted disagreement or confirmation note"],
            },
            "interpreter_corrected_transcript": "[redacted interpreter correction]",
            "tool_critical_text_source": "gemma_interpreter",
        },
    ]


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
        "evidence_load_issues": matrix.get("evidence_load_issues", []),
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
            "collector_attestation",
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
            "source_artifact_must_be_explicitly_redacted": True,
            "source_artifact_must_not_contain_likely_secret_or_phone_values": True,
            "source_artifact_sha256_must_match": True,
            "source_artifact_identity_must_match": True,
            "source_artifact_candidate_rows_require_candidate_identity": True,
            "source_artifact_kind_identity_only_for_kind_only_rows": True,
            "stack_smoke_source_artifact_must_include_kame_turn_contract": True,
            "benchmark_evidence_not_before": SPARK_BENCHMARK_EVIDENCE_NOT_BEFORE.isoformat(),
            "measured_at_must_be_in_evidence_window": True,
            "collector_attestation_timestamps_must_be_in_evidence_window": True,
            "accepted_source_artifact_identity_fields": [
                "source_key",
                "source_keys",
                "candidate_id",
                "candidate_ids",
                "kind",
                "kinds",
            ],
            "collector_attestation_required_for_one_spark_readiness": True,
            "collector_attestation_required_fields": list(COLLECTOR_ATTESTATION_REQUIRED_FIELDS),
            "placeholder_collector_attestation_accepted": False,
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
            "interpreter_providers",
            "auxiliary_transcript_sources",
            "metrics.speech_end_to_first_audio_ms",
            "metrics.barge_in_stop_ms",
            "metrics.local_turns",
            "metrics.local_turn_oracle_calls",
            "metrics.oracle_bound_turns",
            "metrics.oracle_bound_oracle_calls",
            "source_artifact",
            "source_artifact_sha256",
            "collector_attestation",
            "verified",
            "source_artifact.kame_turns[].audio_segment_ref",
            "source_artifact.kame_turns[].audio_time_range_ms",
            "source_artifact.kame_turns[].reflex_transcript_hypothesis.authority=hypothesis",
            "source_artifact.kame_turns[].auxiliary_transcript_hypotheses[].authority=hypothesis",
            "source_artifact.kame_turns[].interpreter_evidence.source=gemma_interpreter",
            "source_artifact.kame_turns[].interpreter_corrected_transcript",
            "source_artifact.kame_turns[].tool_critical_text_source=gemma_interpreter|oracle_judgment",
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
            "required_reflex_provider": "s2s_or_timing",
            "required_interpreter_provider": "gemma_audio",
            "auxiliary_transcript_sources_optional": True,
            "required_source_artifact_contract": {
                "turns_field": "kame_turns",
                "requires_local_turn_without_oracle_call": True,
                "requires_oracle_bound_turn_with_oracle_call": True,
                "requires_raw_audio_reference": True,
                "requires_transcript_hypotheses_labeled_authority_hypothesis": True,
                "requires_gemma_interpreter_correction": True,
                "requires_tool_critical_text_source": "gemma_interpreter_or_oracle_judgment",
            },
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
                    "reflex_providers": ["moshi"],
                    "interpreter_providers": ["vllm", "gemma"],
                    "auxiliary_transcript_sources": ["moshi_hypothesis", "classic_asr_fallback_optional"],
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
            "refresh_source_hashes": (
                "uv run python scripts/voiceops_spark_matrix.py "
                f"--refresh-source-hashes {SPARK_BENCHMARK_SCAFFOLD_EVIDENCE}"
            ),
            "with_evidence": (
                "uv run python scripts/voiceops_spark_matrix.py "
                "--output-dir artifacts/voiceops-spark-matrix/current "
                f"--evidence {SPARK_BENCHMARK_SCAFFOLD_EVIDENCE}"
            ),
            "lint_evidence": (
                "uv run python scripts/voiceops_spark_matrix.py "
                "--lint-evidence "
                f"--evidence {SPARK_BENCHMARK_SCAFFOLD_EVIDENCE}"
            ),
            "plan_index": (
                "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts "
                "--output-dir artifacts/voiceops-plan/current "
                f"--evidence {SPARK_BENCHMARK_SCAFFOLD_EVIDENCE}"
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
        f"- Evidence load issues: {', '.join(plan['evidence_load_issues']) if plan['evidence_load_issues'] else 'none'}",
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
    lines.append(f"- Interpreter provider: {smoke['required_interpreter_provider']}")
    lines.append(f"- Auxiliary transcript sources optional: {'yes' if smoke['auxiliary_transcript_sources_optional'] else 'no'}")
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
        "1. Start the local KAME stack on the DGX Spark: reflex, Gemma raw-audio interpreter, Hermes oracle endpoint selected through `/model`, TTS, optional transcript evidence, and realtime voice sidecar.",
        "   If the generated KAME stack still points transcript/TTS services at `loopback_smoke_bridge`, treat those services as protocol-only smoke checks.",
        "   They cannot satisfy VoiceOps local transcript/TTS evidence; replace `HERMES_DGX_SPARK_ASR_MODULE`, `HERMES_DGX_SPARK_ASR_ADAPTER`, `HERMES_DGX_SPARK_TTS_MODULE`, `HERMES_DGX_SPARK_TTS_ADAPTER`, and the speech models with production local Moshi/Nemotron Speech, Magpie, Riva-style, or equivalent Spark providers before collecting verified evidence.",
        "2. Run the repo-side evaluator on the DGX Spark and preserve every raw output artifact it writes.",
        "",
        "```bash",
        commands["dgx_eval"],
        "```",
        "",
        "3. Replace the scaffold source artifacts under `spark-benchmark-scaffold/sources/` with redacted measured raw outputs.",
        "4. Fill `spark-benchmark-scaffold/spark-benchmark-evidence.json` with measured metrics, real `source_artifact` refs, `verified: true`, and no `example_only` markers.",
        "5. Refresh `source_artifact_sha256` and `collector_attestation.redacted_artifact_sha256` values from the redacted source artifacts.",
        "",
        "```bash",
        commands["refresh_source_hashes"],
        "```",
        "",
        "6. Re-run the matrix validator against the measured evidence.",
        "",
        "```bash",
        commands["with_evidence"],
        "```",
        "",
        "7. Re-index the full VoiceOps plan with the same measured evidence file.",
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
        "- Source artifacts must be explicitly redacted UTF-8 JSON, must not carry `example_only: true`, and must not contain likely secrets or phone-like values.",
        "- `source_artifact_sha256` must match the referenced source artifact bytes.",
        "- `collector_attestation` must identify the collector, command argv, git commit, timestamp window, raw/redacted hashes, and parent manifest hash.",
        "- `example_only: true` evidence is rejected.",
        "- `loopback_smoke_bridge` evidence is protocol-only and must remain unverified for local transcript/TTS evidence.",
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
            f"- Required interpreter provider: `{smoke['required_interpreter_provider']}`",
            f"- Auxiliary transcript sources optional: `{smoke['auxiliary_transcript_sources_optional']}`",
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
        "--lint-evidence",
        action="store_true",
        help="Validate supplied benchmark evidence and print a no-write readiness summary.",
    )
    parser.add_argument(
        "--evidence",
        action="append",
        default=[],
        type=Path,
        help="Benchmark evidence JSON file to validate against the matrix. May be repeated.",
    )
    parser.add_argument(
        "--refresh-source-hashes",
        type=Path,
        help=(
            "Refresh source_artifact_sha256 and collector_attestation.redacted_artifact_sha256 "
            "fields in a local Spark benchmark evidence file without running benchmarks."
        ),
    )
    args = parser.parse_args(argv)
    if args.refresh_source_hashes is not None and (args.lint_evidence or args.evidence):
        parser.error("--refresh-source-hashes cannot be combined with --lint-evidence or --evidence")
    if args.lint_evidence and not args.evidence:
        parser.error("--lint-evidence requires at least one --evidence path")
    return args


def _lint_summary(matrix: dict[str, Any], evidence_paths: Iterable[Path]) -> dict[str, Any]:
    return {
        "schema_version": "voiceops.spark_evidence_lint.v1",
        "artifact_writes": False,
        "network_io": False,
        "spark_execution": False,
        "ok": bool(matrix.get("ready_for_one_spark_demo")),
        "ready_for_one_spark_demo": matrix.get("ready_for_one_spark_demo"),
        "evidence_paths": [str(path) for path in evidence_paths],
        "evidence_load_issues": matrix.get("evidence_load_issues", []),
        "role_status": matrix.get("role_status", {}),
        "stack_smoke": {
            "status": matrix.get("stack_smoke", {}).get("status"),
            "issues": matrix.get("stack_smoke", {}).get("issues", []),
        },
        "candidate_results": [
            {
                "candidate_id": evaluation.get("candidate_id"),
                "role": evaluation.get("role"),
                "status": evaluation.get("status"),
                "issues": evaluation.get("issues", []),
            }
            for evaluation in matrix.get("evaluations", [])
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.refresh_source_hashes is not None:
        result = refresh_spark_source_hashes(args.refresh_source_hashes)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    matrix = build_matrix(args.evidence)
    if args.lint_evidence:
        summary = _lint_summary(matrix, args.evidence)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["ok"] else 1
    paths = write_matrix(args.output_dir, matrix)
    ok = not matrix.get("evidence_load_issues")
    print(
        json.dumps(
            {
                "ok": ok,
                "output_dir": str(args.output_dir),
                "artifacts": paths,
                "evidence_load_issues": matrix.get("evidence_load_issues", []),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
