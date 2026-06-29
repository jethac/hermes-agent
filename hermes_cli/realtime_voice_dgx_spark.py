"""Generate and preflight a headless DGX Spark KAME realtime voice stack."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from agent.realtime_voice_kame import kame_reflex_instruction_text, kame_reflex_schema_issues


DEFAULT_OUTPUT_DIR = "./artifacts/realtime-voice-dgx-spark"
DEFAULT_INTERFACE_PROVIDER = "gemma4"
DEFAULT_INTERFACE_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_INTERFACE_MODEL = "gemma-4-E2B-it"
DEFAULT_INTERFACE_CANDIDATE_MODELS = ("gemma-4-E2B-it", "gemma-4-E4B-it")
DEFAULT_INTERFACE_API_KEY_ENV = "HERMES_KAME_INTERFACE_API_KEY"
DEFAULT_INTERFACE_MAX_AUDIO_SECONDS = 30.0
DEFAULT_INTERFACE_LIMIT_MM_PER_PROMPT = {"image": 0, "audio": 1}
DEFAULT_ORACLE_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_ORACLE_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
DEFAULT_ORACLE_LIMIT_MM_PER_PROMPT = {"image": 0, "audio": 0}
DEFAULT_SIDECAR_BASE_URL = "http://127.0.0.1:8765"
DEFAULT_ASR_BASE_URL = "http://127.0.0.1:8767"
DEFAULT_TTS_BASE_URL = "http://127.0.0.1:8768"
DEFAULT_ASR_PROVIDER = "streaming_stt"
DEFAULT_TTS_PROVIDER = "streaming_tts"
DEFAULT_ASR_MODULE = "hermes_cli.realtime_voice_loopback_bridge"
DEFAULT_TTS_MODULE = "hermes_cli.realtime_voice_loopback_bridge"
DEFAULT_ASR_MODEL = "oracle-verbatim-asr"
DEFAULT_TTS_MODEL = "local-streaming-tts"
DEFAULT_ASR_ADAPTER = "loopback_smoke_bridge"
DEFAULT_TTS_ADAPTER = "loopback_smoke_bridge"
DEFAULT_VLLM_IMAGE = "vllm/vllm-openai:latest"
DEFAULT_HERMES_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
DEFAULT_SCRIPT_PYTHON = "python3"
VOICEOPS_SPARK_EVIDENCE_SCHEMA_VERSION = "voiceops.spark_benchmark_evidence.v1"
REQUIRED_DGX_SPARK_SMOKES: tuple[tuple[str, str], ...] = (
    (
        "all_local_smoke",
        (
            "Set ok=true only after Discord or loopback voice proves local interface, oracle, ASR, TTS, "
            "and sidecar are all healthy; local turns bypass the oracle; and authority-sensitive turns "
            "still route through the oracle."
        ),
    ),
    (
        "cloud_fallback_smoke",
        "Set ok=true only after sidecar/local-provider unavailability falls back according to configured policy.",
    ),
    (
        "capability_honesty_smoke",
        "Set ok=true only after the live KAME path answers voice-capability checks without claiming it cannot hear or speak.",
    ),
    (
        "barge_in_interruption_smoke",
        "Set ok=true only after confirmed user speech during playback stops audio within target and avoids committing the interrupted response as complete.",
    ),
)


def _parse_bool(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected one of true, false, 1, 0, yes, no, on, or off")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a headless DGX Spark launch/preflight pack for KAME voice"
    )
    add_dgx_spark_arguments(parser)
    return parser


def add_dgx_spark_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach DGX Spark KAME launch/preflight arguments to an argparse parser."""

    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--hermes-home", default="~/.hermes")
    parser.add_argument(
        "--interface-provider",
        default=DEFAULT_INTERFACE_PROVIDER,
        help="Config/status provider label for the KAME interface/reflex model",
    )
    parser.add_argument("--interface-base-url", default=DEFAULT_INTERFACE_BASE_URL)
    parser.add_argument("--interface-model", default=DEFAULT_INTERFACE_MODEL)
    parser.add_argument(
        "--interface-api-key-env",
        default=DEFAULT_INTERFACE_API_KEY_ENV,
        help="Environment variable containing the KAME interface endpoint bearer token; generated files never store the value",
    )
    parser.add_argument(
        "--interface-candidate-model",
        action="append",
        default=None,
        help=(
            "Interface/reflex model to include in the local benchmark matrix. "
            "Repeat to add candidates; defaults to Gemma 4 E2B plus E4B comparison."
        ),
    )
    parser.add_argument("--interface-context-tokens", type=int, default=8192)
    parser.add_argument("--interface-gpu-memory-utilization", type=float, default=0.18)
    parser.add_argument("--interface-max-audio-seconds", type=float, default=DEFAULT_INTERFACE_MAX_AUDIO_SECONDS)
    parser.add_argument("--interface-temperature", type=float, default=0.2)
    parser.add_argument("--interface-max-output-tokens", type=int, default=160)
    parser.add_argument("--interface-timeout-seconds", type=float, default=0.8)
    parser.add_argument("--oracle-base-url", default=DEFAULT_ORACLE_BASE_URL)
    parser.add_argument("--oracle-model", default=DEFAULT_ORACLE_MODEL)
    parser.add_argument("--oracle-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--oracle-context-tokens", type=int, default=32768)
    parser.add_argument("--oracle-gpu-memory-utilization", type=float, default=0.62)
    parser.add_argument("--max-spoken-sentences", type=int, default=2)
    parser.add_argument(
        "--voice-response-policy",
        default="sentence_cap",
        choices=("sentence_cap", "brief_summary", "full"),
    )
    parser.add_argument(
        "--fallback-policy",
        default="legacy_voice",
        choices=("legacy_voice", "text_only", "fail_closed"),
    )
    parser.add_argument("--allow-local-greetings", type=_parse_bool, default=True)
    parser.add_argument("--allow-local-clarifications", type=_parse_bool, default=True)
    parser.add_argument("--require-oracle-for-tools", type=_parse_bool, default=True)
    parser.add_argument("--require-oracle-for-memory", type=_parse_bool, default=True)
    parser.add_argument("--require-oracle-for-files", type=_parse_bool, default=True)
    parser.add_argument("--local-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--barge-in-min-rms", type=int, default=350)
    parser.add_argument("--barge-in-min-speech-ms", type=int, default=120)
    parser.add_argument("--barge-in-stop-playback-deadline-ms", type=int, default=150)
    parser.add_argument("--metrics-enabled", type=_parse_bool, default=True)
    parser.add_argument("--log-turn-spans", type=_parse_bool, default=True)
    parser.add_argument("--log-provider-spans", type=_parse_bool, default=True)
    parser.add_argument("--sidecar-base-url", default=DEFAULT_SIDECAR_BASE_URL)
    parser.add_argument("--asr-base-url", default=DEFAULT_ASR_BASE_URL)
    parser.add_argument("--tts-base-url", default=DEFAULT_TTS_BASE_URL)
    parser.add_argument(
        "--asr-provider",
        default=DEFAULT_ASR_PROVIDER,
        help="Config/status provider label for the oracle-verbatim ASR lane",
    )
    parser.add_argument(
        "--tts-provider",
        default=DEFAULT_TTS_PROVIDER,
        help="Config/status provider label for the spoken-output TTS lane",
    )
    parser.add_argument("--asr-module", default=DEFAULT_ASR_MODULE)
    parser.add_argument("--tts-module", default=DEFAULT_TTS_MODULE)
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL)
    parser.add_argument("--asr-adapter", default=DEFAULT_ASR_ADAPTER)
    parser.add_argument("--tts-adapter", default=DEFAULT_TTS_ADAPTER)
    parser.add_argument(
        "--asr-mode",
        default="on_escalation",
        choices=("disabled", "on_escalation", "speculative", "debug", "fallback"),
    )
    parser.add_argument("--vllm-image", default=DEFAULT_VLLM_IMAGE)
    parser.add_argument("--hermes-image", default=DEFAULT_HERMES_IMAGE)
    parser.add_argument("--model-cache-dir", default="${HOME}/.cache/huggingface")
    parser.add_argument("--check", action="store_true", help="Probe generated endpoint URLs")
    parser.add_argument("--timeout", type=float, default=2.0, help="Endpoint probe timeout seconds")
    parser.add_argument(
        "--benchmark-evidence",
        help="Validate a JSON array of DGX Spark KAME benchmark/evidence results against the generated matrix",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


def run_from_args(args: argparse.Namespace) -> int:
    """Generate artifacts and optional checks from parsed CLI arguments."""

    manifest = build_dgx_spark_stack_manifest(
        repo_dir=Path(args.repo_dir).expanduser().resolve(),
        hermes_home=Path(args.hermes_home).expanduser(),
        interface_provider=str(args.interface_provider),
        interface_base_url=str(args.interface_base_url),
        interface_model=str(args.interface_model),
        interface_api_key_env=str(args.interface_api_key_env),
        interface_candidate_models=args.interface_candidate_model,
        interface_context_tokens=int(args.interface_context_tokens),
        interface_gpu_memory_utilization=float(args.interface_gpu_memory_utilization),
        interface_max_audio_seconds=float(args.interface_max_audio_seconds),
        interface_temperature=float(args.interface_temperature),
        interface_max_output_tokens=int(args.interface_max_output_tokens),
        interface_timeout_seconds=float(args.interface_timeout_seconds),
        oracle_base_url=str(args.oracle_base_url),
        oracle_model=str(args.oracle_model),
        oracle_timeout_seconds=float(args.oracle_timeout_seconds),
        oracle_context_tokens=int(args.oracle_context_tokens),
        oracle_gpu_memory_utilization=float(args.oracle_gpu_memory_utilization),
        max_spoken_sentences=int(args.max_spoken_sentences),
        voice_response_policy=str(args.voice_response_policy),
        fallback_policy=str(args.fallback_policy),
        allow_local_greetings=bool(args.allow_local_greetings),
        allow_local_clarifications=bool(args.allow_local_clarifications),
        require_oracle_for_tools=bool(args.require_oracle_for_tools),
        require_oracle_for_memory=bool(args.require_oracle_for_memory),
        require_oracle_for_files=bool(args.require_oracle_for_files),
        local_confidence_threshold=float(args.local_confidence_threshold),
        barge_in_min_rms=int(args.barge_in_min_rms),
        barge_in_min_speech_ms=int(args.barge_in_min_speech_ms),
        barge_in_stop_playback_deadline_ms=int(args.barge_in_stop_playback_deadline_ms),
        metrics_enabled=bool(args.metrics_enabled),
        log_turn_spans=bool(args.log_turn_spans),
        log_provider_spans=bool(args.log_provider_spans),
        sidecar_base_url=str(args.sidecar_base_url),
        asr_base_url=str(args.asr_base_url),
        tts_base_url=str(args.tts_base_url),
        asr_provider=str(args.asr_provider),
        tts_provider=str(args.tts_provider),
        asr_module=str(args.asr_module),
        tts_module=str(args.tts_module),
        asr_model=str(args.asr_model),
        tts_model=str(args.tts_model),
        asr_adapter=str(args.asr_adapter),
        tts_adapter=str(args.tts_adapter),
        asr_mode=str(args.asr_mode),
        vllm_image=str(args.vllm_image),
        hermes_image=str(args.hermes_image),
        model_cache_dir=str(args.model_cache_dir),
    )
    output_dir = Path(args.output_dir).expanduser()
    written = write_dgx_spark_stack_artifacts(output_dir, manifest)

    preflight: dict[str, Any] | None = None
    if args.check:
        preflight = preflight_dgx_spark_stack(manifest, timeout_seconds=float(args.timeout))
        preflight_path = output_dir / "preflight.json"
        preflight_path.write_text(_json(preflight), encoding="utf-8")
        written["preflight"] = str(preflight_path)
    evidence_validation: dict[str, Any] | None = None
    if args.benchmark_evidence:
        evidence_entries = load_dgx_spark_benchmark_evidence(args.benchmark_evidence)
        evidence_validation = validate_dgx_spark_benchmark_evidence(
            build_dgx_spark_benchmark_matrix(manifest),
            evidence_entries,
        )

    result = {
        "ok": (preflight is None or bool(preflight.get("ok")))
        and (evidence_validation is None or bool(evidence_validation.get("ok"))),
        "output_dir": str(output_dir),
        "written": written,
    }
    if evidence_validation is not None:
        result["benchmark_evidence"] = evidence_validation
    print(_json(result))
    return 0 if result["ok"] else 1


def build_dgx_spark_stack_manifest(
    *,
    repo_dir: Path,
    hermes_home: Path,
    interface_provider: str = DEFAULT_INTERFACE_PROVIDER,
    interface_base_url: str = DEFAULT_INTERFACE_BASE_URL,
    interface_model: str = DEFAULT_INTERFACE_MODEL,
    interface_api_key_env: str = DEFAULT_INTERFACE_API_KEY_ENV,
    interface_candidate_models: list[str] | tuple[str, ...] | None = None,
    interface_context_tokens: int = 8192,
    interface_gpu_memory_utilization: float = 0.18,
    interface_max_audio_seconds: float = DEFAULT_INTERFACE_MAX_AUDIO_SECONDS,
    interface_temperature: float = 0.2,
    interface_max_output_tokens: int = 160,
    interface_timeout_seconds: float = 0.8,
    oracle_base_url: str = DEFAULT_ORACLE_BASE_URL,
    oracle_model: str = DEFAULT_ORACLE_MODEL,
    oracle_timeout_seconds: float = 60.0,
    oracle_context_tokens: int = 32768,
    oracle_gpu_memory_utilization: float = 0.62,
    max_spoken_sentences: int = 2,
    voice_response_policy: str = "sentence_cap",
    fallback_policy: str = "legacy_voice",
    allow_local_greetings: bool = True,
    allow_local_clarifications: bool = True,
    require_oracle_for_tools: bool = True,
    require_oracle_for_memory: bool = True,
    require_oracle_for_files: bool = True,
    local_confidence_threshold: float = 0.75,
    barge_in_min_rms: int = 350,
    barge_in_min_speech_ms: int = 120,
    barge_in_stop_playback_deadline_ms: int = 150,
    metrics_enabled: bool = True,
    log_turn_spans: bool = True,
    log_provider_spans: bool = True,
    sidecar_base_url: str = DEFAULT_SIDECAR_BASE_URL,
    asr_base_url: str = DEFAULT_ASR_BASE_URL,
    tts_base_url: str = DEFAULT_TTS_BASE_URL,
    asr_provider: str = DEFAULT_ASR_PROVIDER,
    tts_provider: str = DEFAULT_TTS_PROVIDER,
    asr_module: str = DEFAULT_ASR_MODULE,
    tts_module: str = DEFAULT_TTS_MODULE,
    asr_model: str = DEFAULT_ASR_MODEL,
    tts_model: str = DEFAULT_TTS_MODEL,
    asr_adapter: str = DEFAULT_ASR_ADAPTER,
    tts_adapter: str = DEFAULT_TTS_ADAPTER,
    asr_mode: str = "on_escalation",
    vllm_image: str = DEFAULT_VLLM_IMAGE,
    hermes_image: str = DEFAULT_HERMES_IMAGE,
    model_cache_dir: str = "${HOME}/.cache/huggingface",
) -> dict[str, Any]:
    interface_models_url = _openai_models_url(interface_base_url)
    oracle_models_url = _openai_models_url(oracle_base_url)
    sidecar_health_url = _health_url(sidecar_base_url)
    asr_health_url = _health_url(asr_base_url) if asr_base_url else ""
    tts_health_url = _health_url(tts_base_url) if tts_base_url else ""
    interface_candidates = _interface_candidate_models(interface_model, interface_candidate_models)
    asr_module_name = _python_module_name(asr_module, default=DEFAULT_ASR_MODULE)
    tts_module_name = _python_module_name(tts_module, default=DEFAULT_TTS_MODULE)
    asr_model_name = _clean_nonempty(asr_model, default=DEFAULT_ASR_MODEL)
    tts_model_name = _clean_nonempty(tts_model, default=DEFAULT_TTS_MODEL)
    asr_adapter_name = _clean_nonempty(asr_adapter, default=DEFAULT_ASR_ADAPTER)
    tts_adapter_name = _clean_nonempty(tts_adapter, default=DEFAULT_TTS_ADAPTER)
    interface_provider_name = _clean_nonempty(interface_provider, default=DEFAULT_INTERFACE_PROVIDER)
    asr_provider_name = _clean_nonempty(asr_provider, default=DEFAULT_ASR_PROVIDER)
    tts_provider_name = _clean_nonempty(tts_provider, default=DEFAULT_TTS_PROVIDER)
    spoken_sentences = max(1, int(max_spoken_sentences or 2))
    stop_playback_deadline_ms = max(1, int(barge_in_stop_playback_deadline_ms or 150))
    return {
        "kind": "kame_dgx_spark_stack",
        "version": 1,
        "repo_dir": str(repo_dir),
        "hermes_home": str(hermes_home),
        "target": {
            "hardware": "1x DGX Spark",
            "mode": "headless",
            "cloud_fallback": "configurable",
        },
        "engine": {
            "name": "kame_interface_oracle",
            "interface_audio_input": "native_audio",
            "asr_mode": asr_mode,
            "max_spoken_sentences": spoken_sentences,
            "voice_response_policy": str(voice_response_policy or "sentence_cap"),
            "fallback_policy": str(fallback_policy or "legacy_voice"),
            "durability_policy": "commit final intents and oracle results only",
        },
        "model_assumptions": {
            "interface_audio_input_supported": {
                "model": interface_model,
                "required": True,
                "validated_by": "interface_audio_probe",
                "description": "Interface/reflex model accepts a bounded audio prompt segment and returns text JSON.",
            },
            "interface_audio_is_segment_buffered": {
                "required": True,
                "validated_by": "vad_endpoint_then_interface_audio_probe",
                "description": "Realtime endpointer cuts audio before the interface model encodes it; the model is not treated as a streaming VAD.",
            },
            "interface_audio_limit_seconds": {
                "seconds": _bounded_interface_max_audio_seconds(interface_max_audio_seconds),
                "required": True,
                "validated_by": "manifest_and_vllm_limit_mm_per_prompt",
            },
            "vllm_multimodal_audio_prompt_limit": {
                "limit_mm_per_prompt": dict(DEFAULT_INTERFACE_LIMIT_MM_PER_PROMPT),
                "required": True,
                "validated_by": "compose_vllm_args",
            },
            "vllm_oracle_text_only_multimodal_limit": {
                "limit_mm_per_prompt": dict(DEFAULT_ORACLE_LIMIT_MM_PER_PROMPT),
                "required": True,
                "validated_by": "compose_vllm_args",
                "description": "Oracle vLLM disables unused multimodal profiling so memory stays reserved for text-oracle work.",
            },
            "oracle_authority": {
                "model": oracle_model,
                "required": True,
                "validated_by": "oracle_models_probe",
                "description": "Hermes active oracle or local oracle endpoint remains authoritative for tools, memory, files, and project context.",
            },
        },
        "roles": {
            "interface": {
                "provider": interface_provider_name,
                "implementation": "openai_compatible_vllm",
                "model": interface_model,
                "candidate_models": [
                    {
                        "model": candidate_model,
                        "priority": "default" if candidate_model == interface_model else "comparison",
                        "reason": (
                            "preferred_audio_reflex"
                            if candidate_model == DEFAULT_INTERFACE_MODEL
                            else "validate_only_if_default_fails_latency_routing_or_capability_honesty"
                        ),
                    }
                    for candidate_model in interface_candidates
                ],
                "base_url": interface_base_url,
                "api_key_env": _clean_env_name(interface_api_key_env, default=DEFAULT_INTERFACE_API_KEY_ENV),
                "models_url": interface_models_url,
                "max_model_len": interface_context_tokens,
                "gpu_memory_utilization": interface_gpu_memory_utilization,
                "max_audio_seconds": _bounded_interface_max_audio_seconds(interface_max_audio_seconds),
                "temperature": float(interface_temperature),
                "max_output_tokens": int(interface_max_output_tokens),
                "timeout_seconds": float(interface_timeout_seconds),
                "audio_input": "native_audio",
                "limit_mm_per_prompt": dict(DEFAULT_INTERFACE_LIMIT_MM_PER_PROMPT),
                "routing": ["local", "defer", "oracle_direct", "reject_or_clarify"],
            },
            "oracle": {
                "provider": "hermes_active_oracle_or_openai_compatible_vllm",
                "preferred_local_model": oracle_model,
                "base_url": oracle_base_url,
                "models_url": oracle_models_url,
                "max_model_len": oracle_context_tokens,
                "gpu_memory_utilization": oracle_gpu_memory_utilization,
                "timeout_seconds": float(oracle_timeout_seconds),
                "limit_mm_per_prompt": dict(DEFAULT_ORACLE_LIMIT_MM_PER_PROMPT),
                "authority": ["tools", "memory", "files", "project_context"],
            },
            "sidecar": {
                "module": "hermes_cli.realtime_voice_sidecar",
                "base_url": sidecar_base_url,
                "health_url": sidecar_health_url,
                "session_path": "/v1/realtime-text/session",
            },
            "asr": {
                "role": "oracle_verbatim_evidence",
                "provider": asr_provider_name,
                "mode": asr_mode,
                "base_url": asr_base_url,
                "health_url": asr_health_url,
                "module": asr_module_name,
                "model": asr_model_name,
                "adapter": asr_adapter_name,
                "protocol_smoke_only": asr_adapter_name == DEFAULT_ASR_ADAPTER,
                "production_replacement": "local_streaming_asr",
                "feeds_reflex": asr_mode == "fallback",
            },
            "tts": {
                "role": "spoken_output",
                "provider": tts_provider_name,
                "base_url": tts_base_url,
                "health_url": tts_health_url,
                "module": tts_module_name,
                "model": tts_model_name,
                "adapter": tts_adapter_name,
                "protocol_smoke_only": tts_adapter_name == DEFAULT_TTS_ADAPTER,
                "production_replacement": "local_streaming_tts",
            },
        },
        "quality_targets_ms": {
            "local_ack_first_audio": 500,
            "local_reply_first_audio": 1000,
            "oracle_ack": 500,
            "simple_oracle_first_audio": 3000,
            "oracle_first_token_to_first_tts_audio": 1000,
            "first_tts_audio_to_playback_start": 150,
            "tool_or_context_oracle_first_audio": 8000,
            "barge_in_stop": stop_playback_deadline_ms,
        },
        "routing": {
            "allow_local_greetings": bool(allow_local_greetings),
            "allow_local_clarifications": bool(allow_local_clarifications),
            "require_oracle_for_tools": bool(require_oracle_for_tools),
            "require_oracle_for_memory": bool(require_oracle_for_memory),
            "require_oracle_for_files": bool(require_oracle_for_files),
            "local_confidence_threshold": float(local_confidence_threshold),
        },
        "barge_in": {
            "min_rms": max(0, int(barge_in_min_rms or 350)),
            "min_speech_ms": max(1, int(barge_in_min_speech_ms or 120)),
            "stop_playback_deadline_ms": stop_playback_deadline_ms,
        },
        "metrics": {
            "enabled": bool(metrics_enabled),
            "log_turn_spans": bool(log_turn_spans),
            "log_provider_spans": bool(log_provider_spans),
        },
        "artifacts": {
            "compose": "compose.yaml",
            "env_example": ".env.example",
            "launch": "launch-local-stack.sh",
            "preflight_script": "preflight-local-stack.sh",
            "benchmark_validation": "validate-benchmark-evidence.sh",
            "benchmark_matrix": "benchmark-matrix.json",
            "benchmark_evidence_template": "benchmark-evidence-template.json",
            "preflight": "preflight.json",
        },
        "images": {
            "vllm": vllm_image,
            "hermes": hermes_image,
        },
        "volumes": {
            "model_cache_dir": model_cache_dir,
        },
        "evidence_required": [
            "interface_direct_audio_latency",
            "oracle_simple_first_audio_latency",
            "interface_candidate_model_matrix",
            "interface_direct_audio_vs_stt_fallback",
            "oracle_verbatim_asr_latency_and_literal_accuracy",
            "local_asr_tts_benchmark_matrix",
            "all_local_smoke",
            "cloud_fallback_smoke",
            "capability_honesty_smoke",
            "barge_in_interruption_smoke",
        ],
    }


def write_dgx_spark_stack_artifacts(output_dir: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "manifest": output_dir / "manifest.json",
        "compose": output_dir / "compose.yaml",
        "env_example": output_dir / ".env.example",
        "launch": output_dir / "launch-local-stack.sh",
        "preflight_script": output_dir / "preflight-local-stack.sh",
        "benchmark_validation": output_dir / "validate-benchmark-evidence.sh",
        "benchmark_matrix": output_dir / "benchmark-matrix.json",
        "benchmark_evidence_template": output_dir / "benchmark-evidence-template.json",
    }
    files["manifest"].write_text(_json(manifest), encoding="utf-8")
    files["compose"].write_text(render_dgx_spark_compose(manifest), encoding="utf-8")
    files["env_example"].write_text(render_dgx_spark_env_example(manifest), encoding="utf-8")
    files["launch"].write_text(render_dgx_spark_launch_script(manifest), encoding="utf-8")
    files["launch"].chmod(0o755)
    files["preflight_script"].write_text(render_dgx_spark_preflight_script(manifest), encoding="utf-8")
    files["preflight_script"].chmod(0o755)
    files["benchmark_validation"].write_text(render_dgx_spark_benchmark_validation_script(manifest), encoding="utf-8")
    files["benchmark_validation"].chmod(0o755)
    files["benchmark_matrix"].write_text(
        _json(build_dgx_spark_benchmark_matrix(manifest)),
        encoding="utf-8",
    )
    files["benchmark_evidence_template"].write_text(
        _json(build_dgx_spark_benchmark_evidence_template(build_dgx_spark_benchmark_matrix(manifest))),
        encoding="utf-8",
    )
    return {name: str(path) for name, path in files.items()}


def render_dgx_spark_compose(manifest: Mapping[str, Any]) -> str:
    roles = _roles(manifest)
    interface = roles["interface"]
    oracle = roles["oracle"]
    sidecar = roles["sidecar"]
    asr = roles["asr"]
    tts = roles["tts"]
    images = dict(manifest.get("images") or {})
    volumes = dict(manifest.get("volumes") or {})
    interface_internal_url = "http://kame-interface-vllm:8000/v1"
    asr_internal_url = "http://kame-asr-bridge:8767"
    tts_internal_url = "http://kame-tts-bridge:8768"
    asr_bridge_env = _render_dgx_spark_local_speech_bridge_env(asr)
    tts_bridge_env = _render_dgx_spark_local_speech_bridge_env(tts)
    asr_bridge_args = _render_dgx_spark_local_speech_bridge_args(asr, model_env="HERMES_VOICE_STREAMING_STT_MODEL")
    tts_bridge_args = _render_dgx_spark_local_speech_bridge_args(tts, model_env="HERMES_VOICE_STREAMING_TTS_MODEL")
    interface_mm_limit = _compact_json_value(interface.get("limit_mm_per_prompt") or DEFAULT_INTERFACE_LIMIT_MM_PER_PROMPT)
    oracle_mm_limit = _compact_json_value(oracle.get("limit_mm_per_prompt") or DEFAULT_ORACLE_LIMIT_MM_PER_PROMPT)
    return f"""services:
  kame-interface-vllm:
    image: ${{HERMES_DGX_SPARK_VLLM_IMAGE:-{images.get("vllm", DEFAULT_VLLM_IMAGE)}}}
    ipc: host
    gpus: all
    ports:
      - "8000:8000"
    volumes:
      - "${{HERMES_DGX_SPARK_MODEL_CACHE:-{volumes.get("model_cache_dir", "${HOME}/.cache/huggingface")}}}:/root/.cache/huggingface"
    command:
      - --host
      - 0.0.0.0
      - --port
      - "8000"
      - --model
      - ${{HERMES_KAME_INTERFACE_MODEL:-{interface["model"]}}}
      - --served-model-name
      - ${{HERMES_KAME_INTERFACE_MODEL:-{interface["model"]}}}
      - --max-model-len
      - "{interface["max_model_len"]}"
      - --gpu-memory-utilization
      - "{interface["gpu_memory_utilization"]}"
      - --limit-mm-per-prompt
      - '{interface_mm_limit}'

  kame-oracle-vllm:
    image: ${{HERMES_DGX_SPARK_VLLM_IMAGE:-{images.get("vllm", DEFAULT_VLLM_IMAGE)}}}
    ipc: host
    gpus: all
    ports:
      - "8001:8001"
    volumes:
      - "${{HERMES_DGX_SPARK_MODEL_CACHE:-{volumes.get("model_cache_dir", "${HOME}/.cache/huggingface")}}}:/root/.cache/huggingface"
    command:
      - --host
      - 0.0.0.0
      - --port
      - "8001"
      - --model
      - ${{HERMES_KAME_ORACLE_MODEL:-{oracle["preferred_local_model"]}}}
      - --served-model-name
      - ${{HERMES_KAME_ORACLE_MODEL:-{oracle["preferred_local_model"]}}}
      - --max-model-len
      - "{oracle["max_model_len"]}"
      - --gpu-memory-utilization
      - "{oracle["gpu_memory_utilization"]}"
      - --limit-mm-per-prompt
      - '{oracle_mm_limit}'

  hermes-realtime-sidecar:
    image: ${{HERMES_DGX_SPARK_HERMES_IMAGE:-{images.get("hermes", DEFAULT_HERMES_IMAGE)}}}
    working_dir: /workspace/hermes-agent
    depends_on:
      - kame-interface-vllm
      - kame-oracle-vllm
      - kame-asr-bridge
      - kame-tts-bridge
    ports:
      - "8765:8765"
    volumes:
      - ${{HERMES_REPO_DIR:-{manifest["repo_dir"]}}}:/workspace/hermes-agent
      - ${{HERMES_HOME:-{manifest["hermes_home"]}}}:/root/.hermes
    environment:
      HERMES_HOME: /root/.hermes
      HERMES_KAME_INTERFACE_PROVIDER: {interface["provider"]}
      HERMES_KAME_INTERFACE_BASE_URL: {interface_internal_url}
      HERMES_KAME_INTERFACE_API_KEY: ${{{interface["api_key_env"]}:-}}
      HERMES_VOICE_VLLM_BASE_URL: {interface_internal_url}
      HERMES_VOICE_VLLM_TOKEN: ${{{interface["api_key_env"]}:-}}
      HERMES_VOICE_VLLM_MODEL: ${{HERMES_KAME_INTERFACE_MODEL:-{interface["model"]}}}
      HERMES_VOICE_STREAMING_STT_BASE_URL: {asr_internal_url}
      HERMES_DGX_SPARK_ASR_PROVIDER: {asr["provider"]}
      HERMES_VOICE_STREAMING_STT_MODEL: ${{HERMES_VOICE_STREAMING_STT_MODEL:-{asr["model"]}}}
      HERMES_VOICE_STREAMING_TTS_BASE_URL: {tts_internal_url}
      HERMES_DGX_SPARK_TTS_PROVIDER: {tts["provider"]}
      HERMES_VOICE_STREAMING_TTS_MODEL: ${{HERMES_VOICE_STREAMING_TTS_MODEL:-{tts["model"]}}}
    command:
      - uv
      - run
      - --extra
      - voice
      - python
      - -m
      - {sidecar["module"]}
      - --host
      - 0.0.0.0
      - --port
      - "8765"
      - --interface-base-url
      - {interface_internal_url}
      - --vllm-base-url
      - {interface_internal_url}
      - --vllm-model
      - ${{HERMES_KAME_INTERFACE_MODEL:-{interface["model"]}}}
      - --streaming-stt-base-url
      - {asr_internal_url}
      - --streaming-stt-model
      - ${{HERMES_VOICE_STREAMING_STT_MODEL:-{asr["model"]}}}
      - --streaming-tts-base-url
      - {tts_internal_url}
      - --streaming-tts-model
      - ${{HERMES_VOICE_STREAMING_TTS_MODEL:-{tts["model"]}}}

  kame-asr-bridge:
    image: ${{HERMES_DGX_SPARK_HERMES_IMAGE:-{images.get("hermes", DEFAULT_HERMES_IMAGE)}}}
    working_dir: /workspace/hermes-agent
    ports:
      - "8767:8767"
    volumes:
      - ${{HERMES_REPO_DIR:-{manifest["repo_dir"]}}}:/workspace/hermes-agent
      - ${{HERMES_HOME:-{manifest["hermes_home"]}}}:/root/.hermes
    environment:
      HERMES_HOME: /root/.hermes
      HERMES_DGX_SPARK_ASR_PROVIDER: {asr["provider"]}
      HERMES_DGX_SPARK_ASR_ADAPTER: {asr["adapter"]}
      HERMES_VOICE_STREAMING_STT_MODEL: {asr["model"]}
{asr_bridge_env.rstrip()}
    command:
      - uv
      - run
      - --extra
      - voice
      - python
      - -m
      - {asr["module"]}
      - --host
      - 0.0.0.0
      - --port
      - "8767"
{asr_bridge_args.rstrip()}
      - --production-en-ja

  kame-tts-bridge:
    image: ${{HERMES_DGX_SPARK_HERMES_IMAGE:-{images.get("hermes", DEFAULT_HERMES_IMAGE)}}}
    working_dir: /workspace/hermes-agent
    ports:
      - "8768:8768"
    volumes:
      - ${{HERMES_REPO_DIR:-{manifest["repo_dir"]}}}:/workspace/hermes-agent
      - ${{HERMES_HOME:-{manifest["hermes_home"]}}}:/root/.hermes
    environment:
      HERMES_HOME: /root/.hermes
      HERMES_DGX_SPARK_TTS_PROVIDER: {tts["provider"]}
      HERMES_DGX_SPARK_TTS_ADAPTER: {tts["adapter"]}
      HERMES_VOICE_STREAMING_TTS_MODEL: {tts["model"]}
{tts_bridge_env.rstrip()}
    command:
      - uv
      - run
      - --extra
      - voice
      - python
      - -m
      - {tts["module"]}
      - --host
      - 0.0.0.0
      - --port
      - "8768"
{tts_bridge_args.rstrip()}
      - --production-en-ja
"""


def render_dgx_spark_env_example(manifest: Mapping[str, Any]) -> str:
    roles = _roles(manifest)
    images = dict(manifest.get("images") or {})
    volumes = dict(manifest.get("volumes") or {})
    engine = dict(manifest.get("engine") or {})
    routing = dict(manifest.get("routing") or {})
    barge_in = dict(manifest.get("barge_in") or {})
    metrics = dict(manifest.get("metrics") or {})
    speech_env = "\n".join(
        part
        for part in (
            _render_dgx_spark_local_speech_bridge_env_example(roles["asr"]),
            _render_dgx_spark_local_speech_bridge_env_example(roles["tts"]),
        )
        if part
    )
    return f"""# DGX Spark KAME realtime voice stack.
# This file intentionally contains no API keys or bearer tokens.
HERMES_DGX_SPARK_VLLM_IMAGE={images.get("vllm", DEFAULT_VLLM_IMAGE)}
HERMES_DGX_SPARK_HERMES_IMAGE={images.get("hermes", DEFAULT_HERMES_IMAGE)}
HERMES_DGX_SPARK_MODEL_CACHE={volumes.get("model_cache_dir", "${HOME}/.cache/huggingface")}
HERMES_REPO_DIR={manifest["repo_dir"]}
HERMES_HOME={manifest["hermes_home"]}
HERMES_PYTHON={DEFAULT_SCRIPT_PYTHON}

HERMES_KAME_INTERFACE_PROVIDER={roles["interface"]["provider"]}
HERMES_KAME_INTERFACE_MODEL={roles["interface"]["model"]}
HERMES_KAME_INTERFACE_BASE_URL={roles["interface"]["base_url"]}
HERMES_KAME_INTERFACE_API_KEY_ENV={roles["interface"]["api_key_env"]}
{roles["interface"]["api_key_env"]}=
HERMES_KAME_INTERFACE_AUDIO_INPUT=native_audio
HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS={roles["interface"]["max_audio_seconds"]}
HERMES_KAME_INTERFACE_TEMPERATURE={roles["interface"]["temperature"]}
HERMES_KAME_INTERFACE_MAX_OUTPUT_TOKENS={roles["interface"]["max_output_tokens"]}
HERMES_KAME_INTERFACE_TIMEOUT_SECONDS={roles["interface"]["timeout_seconds"]}
HERMES_KAME_ASR_MODE={manifest["engine"]["asr_mode"]}
HERMES_KAME_MAX_SPOKEN_SENTENCES={engine["max_spoken_sentences"]}
HERMES_KAME_VOICE_RESPONSE_POLICY={engine["voice_response_policy"]}
HERMES_KAME_FALLBACK_POLICY={engine["fallback_policy"]}
HERMES_KAME_ALLOW_LOCAL_GREETINGS={_env_bool(routing["allow_local_greetings"])}
HERMES_KAME_ALLOW_LOCAL_CLARIFICATIONS={_env_bool(routing["allow_local_clarifications"])}
HERMES_KAME_REQUIRE_ORACLE_FOR_TOOLS={_env_bool(routing["require_oracle_for_tools"])}
HERMES_KAME_REQUIRE_ORACLE_FOR_MEMORY={_env_bool(routing["require_oracle_for_memory"])}
HERMES_KAME_REQUIRE_ORACLE_FOR_FILES={_env_bool(routing["require_oracle_for_files"])}
HERMES_KAME_LOCAL_CONFIDENCE_THRESHOLD={routing["local_confidence_threshold"]}
HERMES_KAME_BARGE_IN_MIN_RMS={barge_in["min_rms"]}
HERMES_KAME_BARGE_IN_MIN_SPEECH_MS={barge_in["min_speech_ms"]}
HERMES_KAME_BARGE_IN_STOP_PLAYBACK_DEADLINE_MS={barge_in["stop_playback_deadline_ms"]}
HERMES_KAME_METRICS_ENABLED={_env_bool(metrics["enabled"])}
HERMES_KAME_LOG_TURN_SPANS={_env_bool(metrics["log_turn_spans"])}
HERMES_KAME_LOG_PROVIDER_SPANS={_env_bool(metrics["log_provider_spans"])}

HERMES_KAME_ORACLE_MODEL={roles["oracle"]["preferred_local_model"]}
HERMES_KAME_ORACLE_BASE_URL={roles["oracle"]["base_url"]}
HERMES_KAME_ORACLE_TIMEOUT_SECONDS={roles["oracle"]["timeout_seconds"]}
HERMES_VOICE_STREAMING_STT_BASE_URL={roles["asr"]["base_url"]}
HERMES_DGX_SPARK_ASR_PROVIDER={roles["asr"]["provider"]}
HERMES_VOICE_STREAMING_STT_MODEL={roles["asr"]["model"]}
HERMES_DGX_SPARK_ASR_MODULE={roles["asr"]["module"]}
HERMES_DGX_SPARK_ASR_ADAPTER={roles["asr"]["adapter"]}
HERMES_VOICE_STREAMING_TTS_BASE_URL={roles["tts"]["base_url"]}
HERMES_DGX_SPARK_TTS_PROVIDER={roles["tts"]["provider"]}
HERMES_VOICE_STREAMING_TTS_MODEL={roles["tts"]["model"]}
HERMES_DGX_SPARK_TTS_MODULE={roles["tts"]["module"]}
HERMES_DGX_SPARK_TTS_ADAPTER={roles["tts"]["adapter"]}
{speech_env}
"""


def _render_dgx_spark_local_speech_bridge_env(speech_role: Mapping[str, Any]) -> str:
    prefix = _local_speech_bridge_env_prefix(speech_role)
    if not prefix:
        return ""
    return (
        f"      {prefix}_UPSTREAM_BASE_URL: ${{{prefix}_UPSTREAM_BASE_URL:-}}\n"
        f"      {prefix}_UPSTREAM_TOKEN: ${{{prefix}_UPSTREAM_TOKEN:-}}\n"
    )


def _render_dgx_spark_local_speech_bridge_args(
    speech_role: Mapping[str, Any],
    *,
    model_env: str,
) -> str:
    prefix = _local_speech_bridge_env_prefix(speech_role)
    if not prefix:
        return ""
    return (
        "      - --model\n"
        f"      - ${{{model_env}:-{speech_role['model']}}}\n"
        "      - --upstream-base-url\n"
        f"      - ${{{prefix}_UPSTREAM_BASE_URL:-}}\n"
        "      - --upstream-token\n"
        f"      - ${{{prefix}_UPSTREAM_TOKEN:-}}\n"
    )


def _render_dgx_spark_local_speech_bridge_env_example(speech_role: Mapping[str, Any]) -> str:
    prefix = _local_speech_bridge_env_prefix(speech_role)
    if not prefix:
        return ""
    return (
        f"{prefix}_UPSTREAM_BASE_URL=\n"
        f"{prefix}_UPSTREAM_TOKEN=\n"
    )


def _local_speech_bridge_env_prefix(speech_role: Mapping[str, Any]) -> str:
    module = str(speech_role.get("module") or "")
    if module == "hermes_cli.realtime_voice_nemotron_speech_bridge":
        return "HERMES_NEMOTRON_SPEECH"
    if module == "hermes_cli.realtime_voice_magpie_tts_bridge":
        return "HERMES_MAGPIE_TTS"
    return ""


def render_dgx_spark_launch_script(manifest: Mapping[str, Any]) -> str:
    sidecar_host, sidecar_port = _url_host_port(
        str(manifest["roles"]["sidecar"]["base_url"]),
        default_port=8765,
    )
    routing = dict(manifest.get("routing") or {})
    barge_in = dict(manifest.get("barge_in") or {})
    metrics = dict(manifest.get("metrics") or {})
    return f"""#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

: "${{HERMES_REPO_DIR:={manifest["repo_dir"]}}}"
: "${{HERMES_HOME:={manifest["hermes_home"]}}}"
: "${{HERMES_PYTHON:={DEFAULT_SCRIPT_PYTHON}}}"
: "${{HERMES_KAME_INTERFACE_PROVIDER:={manifest["roles"]["interface"]["provider"]}}}"
: "${{HERMES_KAME_INTERFACE_MODEL:={manifest["roles"]["interface"]["model"]}}}"
: "${{HERMES_KAME_INTERFACE_BASE_URL:={manifest["roles"]["interface"]["base_url"]}}}"
: "${{HERMES_KAME_INTERFACE_API_KEY_ENV:={manifest["roles"]["interface"]["api_key_env"]}}}"
: "${{HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS:={manifest["roles"]["interface"]["max_audio_seconds"]}}}"
: "${{HERMES_KAME_INTERFACE_TEMPERATURE:={manifest["roles"]["interface"]["temperature"]}}}"
: "${{HERMES_KAME_INTERFACE_MAX_OUTPUT_TOKENS:={manifest["roles"]["interface"]["max_output_tokens"]}}}"
: "${{HERMES_KAME_INTERFACE_TIMEOUT_SECONDS:={manifest["roles"]["interface"]["timeout_seconds"]}}}"
: "${{HERMES_KAME_ASR_MODE:={manifest["engine"]["asr_mode"]}}}"
: "${{HERMES_KAME_MAX_SPOKEN_SENTENCES:={manifest["engine"]["max_spoken_sentences"]}}}"
: "${{HERMES_KAME_VOICE_RESPONSE_POLICY:={manifest["engine"]["voice_response_policy"]}}}"
: "${{HERMES_KAME_FALLBACK_POLICY:={manifest["engine"]["fallback_policy"]}}}"
: "${{HERMES_KAME_ALLOW_LOCAL_GREETINGS:={_env_bool(routing["allow_local_greetings"])}}}"
: "${{HERMES_KAME_ALLOW_LOCAL_CLARIFICATIONS:={_env_bool(routing["allow_local_clarifications"])}}}"
: "${{HERMES_KAME_REQUIRE_ORACLE_FOR_TOOLS:={_env_bool(routing["require_oracle_for_tools"])}}}"
: "${{HERMES_KAME_REQUIRE_ORACLE_FOR_MEMORY:={_env_bool(routing["require_oracle_for_memory"])}}}"
: "${{HERMES_KAME_REQUIRE_ORACLE_FOR_FILES:={_env_bool(routing["require_oracle_for_files"])}}}"
: "${{HERMES_KAME_LOCAL_CONFIDENCE_THRESHOLD:={routing["local_confidence_threshold"]}}}"
: "${{HERMES_KAME_BARGE_IN_MIN_RMS:={barge_in["min_rms"]}}}"
: "${{HERMES_KAME_BARGE_IN_MIN_SPEECH_MS:={barge_in["min_speech_ms"]}}}"
: "${{HERMES_KAME_BARGE_IN_STOP_PLAYBACK_DEADLINE_MS:={barge_in["stop_playback_deadline_ms"]}}}"
: "${{HERMES_KAME_METRICS_ENABLED:={_env_bool(metrics["enabled"])}}}"
: "${{HERMES_KAME_LOG_TURN_SPANS:={_env_bool(metrics["log_turn_spans"])}}}"
: "${{HERMES_KAME_LOG_PROVIDER_SPANS:={_env_bool(metrics["log_provider_spans"])}}}"
: "${{HERMES_KAME_ORACLE_MODEL:={manifest["roles"]["oracle"]["preferred_local_model"]}}}"
: "${{HERMES_KAME_ORACLE_BASE_URL:={manifest["roles"]["oracle"]["base_url"]}}}"
: "${{HERMES_KAME_ORACLE_TIMEOUT_SECONDS:={manifest["roles"]["oracle"]["timeout_seconds"]}}}"
: "${{HERMES_DGX_SPARK_ASR_PROVIDER:={manifest["roles"]["asr"]["provider"]}}}"
: "${{HERMES_VOICE_STREAMING_STT_BASE_URL:={manifest["roles"]["asr"]["base_url"]}}}"
: "${{HERMES_VOICE_STREAMING_STT_MODEL:={manifest["roles"]["asr"]["model"]}}}"
: "${{HERMES_DGX_SPARK_TTS_PROVIDER:={manifest["roles"]["tts"]["provider"]}}}"
: "${{HERMES_VOICE_STREAMING_TTS_BASE_URL:={manifest["roles"]["tts"]["base_url"]}}}"
: "${{HERMES_VOICE_STREAMING_TTS_MODEL:={manifest["roles"]["tts"]["model"]}}}"
export HERMES_REPO_DIR HERMES_HOME

if [ "${{HERMES_DGX_SPARK_APPLY_PROFILE:-1}}" != "0" ]; then
  (
    cd "$HERMES_REPO_DIR"
    "$HERMES_PYTHON" -m hermes_cli.realtime_voice_profile --preset kame --apply \\
      --kame-interface-provider "$HERMES_KAME_INTERFACE_PROVIDER" \\
      --kame-reflex-model "$HERMES_KAME_INTERFACE_MODEL" \\
      --kame-interface-base-url "$HERMES_KAME_INTERFACE_BASE_URL" \\
      --kame-interface-api-key-env "$HERMES_KAME_INTERFACE_API_KEY_ENV" \\
      --kame-interface-audio-input native_audio \\
      --kame-interface-max-audio-seconds "$HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS" \\
      --kame-interface-temperature "$HERMES_KAME_INTERFACE_TEMPERATURE" \\
      --kame-interface-max-output-tokens "$HERMES_KAME_INTERFACE_MAX_OUTPUT_TOKENS" \\
      --kame-interface-timeout-seconds "$HERMES_KAME_INTERFACE_TIMEOUT_SECONDS" \\
      --kame-asr-mode "$HERMES_KAME_ASR_MODE" \\
      --kame-asr-provider "$HERMES_DGX_SPARK_ASR_PROVIDER" \\
      --kame-preferred-local-oracle-model "$HERMES_KAME_ORACLE_MODEL" \\
      --kame-oracle-base-url "$HERMES_KAME_ORACLE_BASE_URL" \\
      --kame-oracle-provider-name "KAME Local Oracle" \\
      --kame-oracle-timeout-seconds "$HERMES_KAME_ORACLE_TIMEOUT_SECONDS" \\
      --kame-max-spoken-sentences "$HERMES_KAME_MAX_SPOKEN_SENTENCES" \\
      --kame-voice-response-policy "$HERMES_KAME_VOICE_RESPONSE_POLICY" \\
      --kame-tts-provider "$HERMES_DGX_SPARK_TTS_PROVIDER" \\
      --kame-fallback-policy "$HERMES_KAME_FALLBACK_POLICY" \\
      --kame-allow-local-greetings "$HERMES_KAME_ALLOW_LOCAL_GREETINGS" \\
      --kame-allow-local-clarifications "$HERMES_KAME_ALLOW_LOCAL_CLARIFICATIONS" \\
      --kame-require-oracle-for-tools "$HERMES_KAME_REQUIRE_ORACLE_FOR_TOOLS" \\
      --kame-require-oracle-for-memory "$HERMES_KAME_REQUIRE_ORACLE_FOR_MEMORY" \\
      --kame-require-oracle-for-files "$HERMES_KAME_REQUIRE_ORACLE_FOR_FILES" \\
      --kame-local-confidence-threshold "$HERMES_KAME_LOCAL_CONFIDENCE_THRESHOLD" \\
      --kame-barge-in-min-rms "$HERMES_KAME_BARGE_IN_MIN_RMS" \\
      --kame-barge-in-min-speech-ms "$HERMES_KAME_BARGE_IN_MIN_SPEECH_MS" \\
      --kame-barge-in-stop-playback-deadline-ms "$HERMES_KAME_BARGE_IN_STOP_PLAYBACK_DEADLINE_MS" \\
      --kame-metrics-enabled "$HERMES_KAME_METRICS_ENABLED" \\
      --kame-log-turn-spans "$HERMES_KAME_LOG_TURN_SPANS" \\
      --kame-log-provider-spans "$HERMES_KAME_LOG_PROVIDER_SPANS" \\
      --streaming-stt-base-url "$HERMES_VOICE_STREAMING_STT_BASE_URL" \\
      --streaming-stt-model "$HERMES_VOICE_STREAMING_STT_MODEL" \\
      --streaming-tts-base-url "$HERMES_VOICE_STREAMING_TTS_BASE_URL" \\
      --streaming-tts-model "$HERMES_VOICE_STREAMING_TTS_MODEL" \\
      --sidecar-host {sidecar_host} \\
      --sidecar-port {sidecar_port}
  )
fi

cd "$SCRIPT_DIR"
docker compose --env-file .env.example -f compose.yaml up --remove-orphans "$@"

# Readiness check once services are up:
#   ./preflight-local-stack.sh
"""


def render_dgx_spark_preflight_script(manifest: Mapping[str, Any]) -> str:
    roles = _roles(manifest)
    routing = dict(manifest.get("routing") or {})
    barge_in = dict(manifest.get("barge_in") or {})
    metrics = dict(manifest.get("metrics") or {})
    return f"""#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

: "${{HERMES_REPO_DIR:={manifest["repo_dir"]}}}"
: "${{HERMES_HOME:={manifest["hermes_home"]}}}"
: "${{HERMES_PYTHON:={DEFAULT_SCRIPT_PYTHON}}}"
: "${{HERMES_KAME_INTERFACE_PROVIDER:={roles["interface"]["provider"]}}}"
: "${{HERMES_KAME_INTERFACE_MODEL:={roles["interface"]["model"]}}}"
: "${{HERMES_KAME_INTERFACE_BASE_URL:={roles["interface"]["base_url"]}}}"
: "${{HERMES_KAME_INTERFACE_API_KEY_ENV:={roles["interface"]["api_key_env"]}}}"
: "${{HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS:={roles["interface"]["max_audio_seconds"]}}}"
: "${{HERMES_KAME_INTERFACE_TEMPERATURE:={roles["interface"]["temperature"]}}}"
: "${{HERMES_KAME_INTERFACE_MAX_OUTPUT_TOKENS:={roles["interface"]["max_output_tokens"]}}}"
: "${{HERMES_KAME_INTERFACE_TIMEOUT_SECONDS:={roles["interface"]["timeout_seconds"]}}}"
: "${{HERMES_KAME_ASR_MODE:={manifest["engine"]["asr_mode"]}}}"
: "${{HERMES_KAME_MAX_SPOKEN_SENTENCES:={manifest["engine"]["max_spoken_sentences"]}}}"
: "${{HERMES_KAME_VOICE_RESPONSE_POLICY:={manifest["engine"]["voice_response_policy"]}}}"
: "${{HERMES_KAME_FALLBACK_POLICY:={manifest["engine"]["fallback_policy"]}}}"
: "${{HERMES_KAME_ALLOW_LOCAL_GREETINGS:={_env_bool(routing["allow_local_greetings"])}}}"
: "${{HERMES_KAME_ALLOW_LOCAL_CLARIFICATIONS:={_env_bool(routing["allow_local_clarifications"])}}}"
: "${{HERMES_KAME_REQUIRE_ORACLE_FOR_TOOLS:={_env_bool(routing["require_oracle_for_tools"])}}}"
: "${{HERMES_KAME_REQUIRE_ORACLE_FOR_MEMORY:={_env_bool(routing["require_oracle_for_memory"])}}}"
: "${{HERMES_KAME_REQUIRE_ORACLE_FOR_FILES:={_env_bool(routing["require_oracle_for_files"])}}}"
: "${{HERMES_KAME_LOCAL_CONFIDENCE_THRESHOLD:={routing["local_confidence_threshold"]}}}"
: "${{HERMES_KAME_BARGE_IN_MIN_RMS:={barge_in["min_rms"]}}}"
: "${{HERMES_KAME_BARGE_IN_MIN_SPEECH_MS:={barge_in["min_speech_ms"]}}}"
: "${{HERMES_KAME_BARGE_IN_STOP_PLAYBACK_DEADLINE_MS:={barge_in["stop_playback_deadline_ms"]}}}"
: "${{HERMES_KAME_METRICS_ENABLED:={_env_bool(metrics["enabled"])}}}"
: "${{HERMES_KAME_LOG_TURN_SPANS:={_env_bool(metrics["log_turn_spans"])}}}"
: "${{HERMES_KAME_LOG_PROVIDER_SPANS:={_env_bool(metrics["log_provider_spans"])}}}"
: "${{HERMES_KAME_ORACLE_MODEL:={roles["oracle"]["preferred_local_model"]}}}"
: "${{HERMES_KAME_ORACLE_BASE_URL:={roles["oracle"]["base_url"]}}}"
: "${{HERMES_KAME_ORACLE_TIMEOUT_SECONDS:={roles["oracle"]["timeout_seconds"]}}}"
: "${{HERMES_VOICE_STREAMING_STT_BASE_URL:={roles["asr"]["base_url"]}}}"
: "${{HERMES_DGX_SPARK_ASR_PROVIDER:={roles["asr"]["provider"]}}}"
: "${{HERMES_VOICE_STREAMING_STT_MODEL:={roles["asr"]["model"]}}}"
: "${{HERMES_DGX_SPARK_ASR_MODULE:={roles["asr"]["module"]}}}"
: "${{HERMES_DGX_SPARK_ASR_ADAPTER:={roles["asr"]["adapter"]}}}"
: "${{HERMES_VOICE_STREAMING_TTS_BASE_URL:={roles["tts"]["base_url"]}}}"
: "${{HERMES_DGX_SPARK_TTS_PROVIDER:={roles["tts"]["provider"]}}}"
: "${{HERMES_VOICE_STREAMING_TTS_MODEL:={roles["tts"]["model"]}}}"
: "${{HERMES_DGX_SPARK_TTS_MODULE:={roles["tts"]["module"]}}}"
: "${{HERMES_DGX_SPARK_TTS_ADAPTER:={roles["tts"]["adapter"]}}}"

cd "$HERMES_REPO_DIR"
"$HERMES_PYTHON" -m hermes_cli.realtime_voice_dgx_spark \\
  --output-dir "$SCRIPT_DIR" \\
  --repo-dir "$HERMES_REPO_DIR" \\
  --hermes-home "$HERMES_HOME" \\
  --interface-provider "$HERMES_KAME_INTERFACE_PROVIDER" \\
  --interface-base-url "$HERMES_KAME_INTERFACE_BASE_URL" \\
  --interface-model "$HERMES_KAME_INTERFACE_MODEL" \\
  --interface-api-key-env "$HERMES_KAME_INTERFACE_API_KEY_ENV" \\
  --interface-max-audio-seconds "$HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS" \\
  --interface-temperature "$HERMES_KAME_INTERFACE_TEMPERATURE" \\
  --interface-max-output-tokens "$HERMES_KAME_INTERFACE_MAX_OUTPUT_TOKENS" \\
  --interface-timeout-seconds "$HERMES_KAME_INTERFACE_TIMEOUT_SECONDS" \\
  --interface-context-tokens {roles["interface"]["max_model_len"]} \\
  --interface-gpu-memory-utilization {roles["interface"]["gpu_memory_utilization"]} \\
  --oracle-base-url "$HERMES_KAME_ORACLE_BASE_URL" \\
  --oracle-model "$HERMES_KAME_ORACLE_MODEL" \\
  --oracle-timeout-seconds "$HERMES_KAME_ORACLE_TIMEOUT_SECONDS" \\
  --oracle-context-tokens {roles["oracle"]["max_model_len"]} \\
  --oracle-gpu-memory-utilization {roles["oracle"]["gpu_memory_utilization"]} \\
  --max-spoken-sentences "$HERMES_KAME_MAX_SPOKEN_SENTENCES" \\
  --voice-response-policy "$HERMES_KAME_VOICE_RESPONSE_POLICY" \\
  --fallback-policy "$HERMES_KAME_FALLBACK_POLICY" \\
  --allow-local-greetings "$HERMES_KAME_ALLOW_LOCAL_GREETINGS" \\
  --allow-local-clarifications "$HERMES_KAME_ALLOW_LOCAL_CLARIFICATIONS" \\
  --require-oracle-for-tools "$HERMES_KAME_REQUIRE_ORACLE_FOR_TOOLS" \\
  --require-oracle-for-memory "$HERMES_KAME_REQUIRE_ORACLE_FOR_MEMORY" \\
  --require-oracle-for-files "$HERMES_KAME_REQUIRE_ORACLE_FOR_FILES" \\
  --local-confidence-threshold "$HERMES_KAME_LOCAL_CONFIDENCE_THRESHOLD" \\
  --barge-in-min-rms "$HERMES_KAME_BARGE_IN_MIN_RMS" \\
  --barge-in-min-speech-ms "$HERMES_KAME_BARGE_IN_MIN_SPEECH_MS" \\
  --barge-in-stop-playback-deadline-ms "$HERMES_KAME_BARGE_IN_STOP_PLAYBACK_DEADLINE_MS" \\
  --metrics-enabled "$HERMES_KAME_METRICS_ENABLED" \\
  --log-turn-spans "$HERMES_KAME_LOG_TURN_SPANS" \\
  --log-provider-spans "$HERMES_KAME_LOG_PROVIDER_SPANS" \\
  --sidecar-base-url {roles["sidecar"]["base_url"]} \\
  --asr-base-url "$HERMES_VOICE_STREAMING_STT_BASE_URL" \\
  --asr-provider "$HERMES_DGX_SPARK_ASR_PROVIDER" \\
  --asr-model "$HERMES_VOICE_STREAMING_STT_MODEL" \\
  --asr-module "$HERMES_DGX_SPARK_ASR_MODULE" \\
  --asr-adapter "$HERMES_DGX_SPARK_ASR_ADAPTER" \\
  --tts-base-url "$HERMES_VOICE_STREAMING_TTS_BASE_URL" \\
  --tts-provider "$HERMES_DGX_SPARK_TTS_PROVIDER" \\
  --tts-model "$HERMES_VOICE_STREAMING_TTS_MODEL" \\
  --tts-module "$HERMES_DGX_SPARK_TTS_MODULE" \\
  --tts-adapter "$HERMES_DGX_SPARK_TTS_ADAPTER" \\
  --asr-mode "$HERMES_KAME_ASR_MODE" \\
  --vllm-image {manifest["images"]["vllm"]} \\
  --hermes-image {manifest["images"]["hermes"]} \\
  --model-cache-dir {manifest["volumes"]["model_cache_dir"]} \\
  --check "$@"
"""


def render_dgx_spark_benchmark_validation_script(manifest: Mapping[str, Any]) -> str:
    roles = _roles(manifest)
    routing = dict(manifest.get("routing") or {})
    barge_in = dict(manifest.get("barge_in") or {})
    metrics = dict(manifest.get("metrics") or {})
    return f"""#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ "$#" -ne 1 ]; then
  echo "usage: $0 /path/to/benchmark-evidence.json" >&2
  exit 2
fi

: "${{HERMES_REPO_DIR:={manifest["repo_dir"]}}}"
: "${{HERMES_HOME:={manifest["hermes_home"]}}}"
: "${{HERMES_PYTHON:={DEFAULT_SCRIPT_PYTHON}}}"
: "${{HERMES_KAME_INTERFACE_PROVIDER:={roles["interface"]["provider"]}}}"
: "${{HERMES_KAME_INTERFACE_MODEL:={roles["interface"]["model"]}}}"
: "${{HERMES_KAME_INTERFACE_BASE_URL:={roles["interface"]["base_url"]}}}"
: "${{HERMES_KAME_INTERFACE_API_KEY_ENV:={roles["interface"]["api_key_env"]}}}"
: "${{HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS:={roles["interface"]["max_audio_seconds"]}}}"
: "${{HERMES_KAME_INTERFACE_TEMPERATURE:={roles["interface"]["temperature"]}}}"
: "${{HERMES_KAME_INTERFACE_MAX_OUTPUT_TOKENS:={roles["interface"]["max_output_tokens"]}}}"
: "${{HERMES_KAME_INTERFACE_TIMEOUT_SECONDS:={roles["interface"]["timeout_seconds"]}}}"
: "${{HERMES_KAME_ASR_MODE:={manifest["engine"]["asr_mode"]}}}"
: "${{HERMES_KAME_MAX_SPOKEN_SENTENCES:={manifest["engine"]["max_spoken_sentences"]}}}"
: "${{HERMES_KAME_VOICE_RESPONSE_POLICY:={manifest["engine"]["voice_response_policy"]}}}"
: "${{HERMES_KAME_FALLBACK_POLICY:={manifest["engine"]["fallback_policy"]}}}"
: "${{HERMES_KAME_ALLOW_LOCAL_GREETINGS:={_env_bool(routing["allow_local_greetings"])}}}"
: "${{HERMES_KAME_ALLOW_LOCAL_CLARIFICATIONS:={_env_bool(routing["allow_local_clarifications"])}}}"
: "${{HERMES_KAME_REQUIRE_ORACLE_FOR_TOOLS:={_env_bool(routing["require_oracle_for_tools"])}}}"
: "${{HERMES_KAME_REQUIRE_ORACLE_FOR_MEMORY:={_env_bool(routing["require_oracle_for_memory"])}}}"
: "${{HERMES_KAME_REQUIRE_ORACLE_FOR_FILES:={_env_bool(routing["require_oracle_for_files"])}}}"
: "${{HERMES_KAME_LOCAL_CONFIDENCE_THRESHOLD:={routing["local_confidence_threshold"]}}}"
: "${{HERMES_KAME_BARGE_IN_MIN_RMS:={barge_in["min_rms"]}}}"
: "${{HERMES_KAME_BARGE_IN_MIN_SPEECH_MS:={barge_in["min_speech_ms"]}}}"
: "${{HERMES_KAME_BARGE_IN_STOP_PLAYBACK_DEADLINE_MS:={barge_in["stop_playback_deadline_ms"]}}}"
: "${{HERMES_KAME_METRICS_ENABLED:={_env_bool(metrics["enabled"])}}}"
: "${{HERMES_KAME_LOG_TURN_SPANS:={_env_bool(metrics["log_turn_spans"])}}}"
: "${{HERMES_KAME_LOG_PROVIDER_SPANS:={_env_bool(metrics["log_provider_spans"])}}}"
: "${{HERMES_KAME_ORACLE_MODEL:={roles["oracle"]["preferred_local_model"]}}}"
: "${{HERMES_KAME_ORACLE_BASE_URL:={roles["oracle"]["base_url"]}}}"
: "${{HERMES_KAME_ORACLE_TIMEOUT_SECONDS:={roles["oracle"]["timeout_seconds"]}}}"
: "${{HERMES_VOICE_STREAMING_STT_BASE_URL:={roles["asr"]["base_url"]}}}"
: "${{HERMES_DGX_SPARK_ASR_PROVIDER:={roles["asr"]["provider"]}}}"
: "${{HERMES_VOICE_STREAMING_STT_MODEL:={roles["asr"]["model"]}}}"
: "${{HERMES_DGX_SPARK_ASR_MODULE:={roles["asr"]["module"]}}}"
: "${{HERMES_DGX_SPARK_ASR_ADAPTER:={roles["asr"]["adapter"]}}}"
: "${{HERMES_VOICE_STREAMING_TTS_BASE_URL:={roles["tts"]["base_url"]}}}"
: "${{HERMES_DGX_SPARK_TTS_PROVIDER:={roles["tts"]["provider"]}}}"
: "${{HERMES_VOICE_STREAMING_TTS_MODEL:={roles["tts"]["model"]}}}"
: "${{HERMES_DGX_SPARK_TTS_MODULE:={roles["tts"]["module"]}}}"
: "${{HERMES_DGX_SPARK_TTS_ADAPTER:={roles["tts"]["adapter"]}}}"

cd "$HERMES_REPO_DIR"
"$HERMES_PYTHON" -m hermes_cli.realtime_voice_dgx_spark \\
  --output-dir "$SCRIPT_DIR" \\
  --repo-dir "$HERMES_REPO_DIR" \\
  --hermes-home "$HERMES_HOME" \\
  --interface-provider "$HERMES_KAME_INTERFACE_PROVIDER" \\
  --interface-base-url "$HERMES_KAME_INTERFACE_BASE_URL" \\
  --interface-model "$HERMES_KAME_INTERFACE_MODEL" \\
  --interface-api-key-env "$HERMES_KAME_INTERFACE_API_KEY_ENV" \\
  --interface-max-audio-seconds "$HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS" \\
  --interface-temperature "$HERMES_KAME_INTERFACE_TEMPERATURE" \\
  --interface-max-output-tokens "$HERMES_KAME_INTERFACE_MAX_OUTPUT_TOKENS" \\
  --interface-timeout-seconds "$HERMES_KAME_INTERFACE_TIMEOUT_SECONDS" \\
  --interface-context-tokens {roles["interface"]["max_model_len"]} \\
  --interface-gpu-memory-utilization {roles["interface"]["gpu_memory_utilization"]} \\
  --oracle-base-url "$HERMES_KAME_ORACLE_BASE_URL" \\
  --oracle-model "$HERMES_KAME_ORACLE_MODEL" \\
  --oracle-timeout-seconds "$HERMES_KAME_ORACLE_TIMEOUT_SECONDS" \\
  --oracle-context-tokens {roles["oracle"]["max_model_len"]} \\
  --oracle-gpu-memory-utilization {roles["oracle"]["gpu_memory_utilization"]} \\
  --max-spoken-sentences "$HERMES_KAME_MAX_SPOKEN_SENTENCES" \\
  --voice-response-policy "$HERMES_KAME_VOICE_RESPONSE_POLICY" \\
  --fallback-policy "$HERMES_KAME_FALLBACK_POLICY" \\
  --allow-local-greetings "$HERMES_KAME_ALLOW_LOCAL_GREETINGS" \\
  --allow-local-clarifications "$HERMES_KAME_ALLOW_LOCAL_CLARIFICATIONS" \\
  --require-oracle-for-tools "$HERMES_KAME_REQUIRE_ORACLE_FOR_TOOLS" \\
  --require-oracle-for-memory "$HERMES_KAME_REQUIRE_ORACLE_FOR_MEMORY" \\
  --require-oracle-for-files "$HERMES_KAME_REQUIRE_ORACLE_FOR_FILES" \\
  --local-confidence-threshold "$HERMES_KAME_LOCAL_CONFIDENCE_THRESHOLD" \\
  --barge-in-min-rms "$HERMES_KAME_BARGE_IN_MIN_RMS" \\
  --barge-in-min-speech-ms "$HERMES_KAME_BARGE_IN_MIN_SPEECH_MS" \\
  --barge-in-stop-playback-deadline-ms "$HERMES_KAME_BARGE_IN_STOP_PLAYBACK_DEADLINE_MS" \\
  --metrics-enabled "$HERMES_KAME_METRICS_ENABLED" \\
  --log-turn-spans "$HERMES_KAME_LOG_TURN_SPANS" \\
  --log-provider-spans "$HERMES_KAME_LOG_PROVIDER_SPANS" \\
  --sidecar-base-url {roles["sidecar"]["base_url"]} \\
  --asr-base-url "$HERMES_VOICE_STREAMING_STT_BASE_URL" \\
  --asr-provider "$HERMES_DGX_SPARK_ASR_PROVIDER" \\
  --asr-model "$HERMES_VOICE_STREAMING_STT_MODEL" \\
  --asr-module "$HERMES_DGX_SPARK_ASR_MODULE" \\
  --asr-adapter "$HERMES_DGX_SPARK_ASR_ADAPTER" \\
  --tts-base-url "$HERMES_VOICE_STREAMING_TTS_BASE_URL" \\
  --tts-provider "$HERMES_DGX_SPARK_TTS_PROVIDER" \\
  --tts-model "$HERMES_VOICE_STREAMING_TTS_MODEL" \\
  --tts-module "$HERMES_DGX_SPARK_TTS_MODULE" \\
  --tts-adapter "$HERMES_DGX_SPARK_TTS_ADAPTER" \\
  --asr-mode "$HERMES_KAME_ASR_MODE" \\
  --vllm-image {manifest["images"]["vllm"]} \\
  --hermes-image {manifest["images"]["hermes"]} \\
  --model-cache-dir {manifest["volumes"]["model_cache_dir"]} \\
  --benchmark-evidence "$1"
"""


def build_dgx_spark_benchmark_matrix(manifest: Mapping[str, Any]) -> dict[str, Any]:
    roles = _roles(manifest)
    interface_models = _interface_candidate_model_names(roles["interface"])
    interface_candidates = []
    for model in interface_models:
        interface_candidates.extend(
            [
                {
                    "model": model,
                    "input": "direct_audio",
                    "required_metrics": [
                        "sample_count",
                        "speech_end_to_interface_decision_ms",
                        "speech_end_to_interface_decision_p50_ms",
                        "speech_end_to_interface_decision_p90_ms",
                        "kame_interface_model_request_ms",
                        "speech_end_to_local_first_audio_ms",
                        "speech_end_to_local_first_audio_p50_ms",
                        "speech_end_to_local_first_audio_p90_ms",
                        "routing_accuracy",
                        "capability_honesty_rate",
                        "local_route_precision",
                        "oracle_route_recall",
                        "steady_state_memory_gb",
                    ],
                },
                {
                    "model": model,
                    "input": "stt_fallback",
                    "required_metrics": [
                        "sample_count",
                        "speech_end_to_transcript_ms",
                        "speech_end_to_transcript_p50_ms",
                        "speech_end_to_transcript_p90_ms",
                        "transcript_to_interface_decision_ms",
                        "transcript_to_interface_decision_p50_ms",
                        "transcript_to_interface_decision_p90_ms",
                        "routing_accuracy",
                        "capability_honesty_rate",
                        "local_route_precision",
                        "oracle_route_recall",
                        "steady_state_memory_gb",
                    ],
                },
            ]
        )
    return {
        "kind": "kame_dgx_spark_benchmark_matrix",
        "version": 1,
        "model_assumptions": dict(manifest.get("model_assumptions") or {}),
        "candidates": {
            "interface": interface_candidates,
            "oracle": [
                {
                    "model": roles["oracle"]["preferred_local_model"],
                    "required_metrics": [
                        "sample_count",
                        "oracle_request_to_accepted_ms",
                        "oracle_accepted_to_first_token_ms",
                        "oracle_first_token_to_first_tts_audio_ms",
                        "first_tts_audio_to_playback_start_ms",
                        "oracle_request_to_first_audio_p50_ms",
                        "oracle_request_to_first_audio_p90_ms",
                        "decode_tok_s",
                        "prefill_tok_s",
                        "steady_state_memory_gb",
                    ],
                }
            ],
            "oracle_outcome": [
                {
                    "model": roles["oracle"]["preferred_local_model"],
                    "asr_hypothesis": "without_asr_hypothesis",
                    "required_metrics": [
                        "task_success_rate",
                        "literal_argument_accuracy",
                        "tool_argument_error_rate",
                    ],
                },
                {
                    "model": roles["oracle"]["preferred_local_model"],
                    "asr_hypothesis": "with_asr_hypothesis",
                    "required_metrics": [
                        "task_success_rate",
                        "literal_argument_accuracy",
                        "tool_argument_error_rate",
                    ],
                },
            ],
            "speech": [
                {
                    "role": "oracle_verbatim_asr",
                    "mode": roles["asr"]["mode"],
                    "provider": roles["asr"]["provider"],
                    "model": roles["asr"]["model"],
                    "adapter": roles["asr"]["adapter"],
                    "module": roles["asr"]["module"],
                    "protocol_smoke_only": roles["asr"].get("protocol_smoke_only") is True,
                    "required_metrics": [
                        "sample_count",
                        "speech_end_to_asr_final_ms",
                        "speech_end_to_asr_final_p50_ms",
                        "speech_end_to_asr_final_p90_ms",
                        "literal_accuracy_names_numbers_code",
                    ],
                },
                {
                    "role": "tts",
                    "provider": roles["tts"]["provider"],
                    "model": roles["tts"]["model"],
                    "adapter": roles["tts"]["adapter"],
                    "module": roles["tts"]["module"],
                    "protocol_smoke_only": roles["tts"].get("protocol_smoke_only") is True,
                    "required_metrics": [
                        "sample_count",
                        "tts_request_to_first_audio_ms",
                        "tts_request_to_first_audio_p50_ms",
                        "tts_request_to_first_audio_p90_ms",
                        "tts_request_to_audio_end_ms",
                        "tts_request_to_audio_end_p50_ms",
                        "tts_request_to_audio_end_p90_ms",
                        "underrun_count",
                    ],
                },
            ],
            "comparison": [
                {
                    "name": "interface_direct_audio_vs_stt_fallback",
                    "required_metrics": [
                        "paired_turns",
                        "direct_audio_p50_decision_ms",
                        "stt_fallback_p50_decision_ms",
                        "direct_audio_routing_accuracy",
                        "stt_fallback_routing_accuracy",
                        "routing_agreement_rate",
                    ],
                },
                {
                    "name": "oracle_outcome_asr_hypothesis_delta",
                    "required_metrics": [
                        "paired_cases",
                        "with_asr_task_success_rate",
                        "without_asr_task_success_rate",
                        "with_asr_literal_argument_accuracy",
                        "without_asr_literal_argument_accuracy",
                        "with_asr_tool_argument_error_rate",
                        "without_asr_tool_argument_error_rate",
                    ],
                },
            ],
        },
        "acceptance_targets_ms": manifest["quality_targets_ms"],
    }


def build_dgx_spark_benchmark_evidence_template(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return a JSON evidence skeleton matching the generated benchmark matrix."""

    candidates = matrix.get("candidates") if isinstance(matrix.get("candidates"), Mapping) else {}
    template: list[dict[str, Any]] = []

    for candidate in candidates.get("interface", []) if isinstance(candidates.get("interface"), list) else []:
        if not isinstance(candidate, Mapping):
            continue
        template.append(
            _with_voiceops_projection_fields(
                {
                    "kind": "kame_benchmark_result",
                    "category": "interface",
                    "model": candidate.get("model"),
                    "input": candidate.get("input"),
                    "metrics": _null_metric_template(candidate.get("required_metrics")),
                }
            )
        )

    for candidate in candidates.get("oracle", []) if isinstance(candidates.get("oracle"), list) else []:
        if not isinstance(candidate, Mapping):
            continue
        template.append(
            _with_voiceops_projection_fields(
                {
                    "kind": "kame_benchmark_result",
                    "category": "oracle",
                    "model": candidate.get("model"),
                    "metrics": _null_metric_template(candidate.get("required_metrics")),
                }
            )
        )

    for candidate in candidates.get("oracle_outcome", []) if isinstance(candidates.get("oracle_outcome"), list) else []:
        if not isinstance(candidate, Mapping):
            continue
        template.append(
            _with_voiceops_projection_fields(
                {
                    "kind": "kame_benchmark_result",
                    "category": "oracle_outcome",
                    "model": candidate.get("model"),
                    "asr_hypothesis": candidate.get("asr_hypothesis"),
                    "metrics": _null_metric_template(candidate.get("required_metrics")),
                }
            )
        )

    for candidate in candidates.get("speech", []) if isinstance(candidates.get("speech"), list) else []:
        if not isinstance(candidate, Mapping):
            continue
        template.append(
            _with_voiceops_projection_fields(
                {
                    "kind": "kame_benchmark_result",
                    "category": "speech",
                    "role": candidate.get("role"),
                    "mode": candidate.get("mode"),
                    "provider": candidate.get("provider"),
                    "model": candidate.get("model"),
                    "adapter": candidate.get("adapter"),
                    "module": candidate.get("module"),
                    "protocol_smoke_only": candidate.get("protocol_smoke_only") is True,
                    "metrics": _null_metric_template(candidate.get("required_metrics")),
                }
            )
        )

    for candidate in candidates.get("comparison", []) if isinstance(candidates.get("comparison"), list) else []:
        if not isinstance(candidate, Mapping):
            continue
        template.append(
            _with_voiceops_projection_fields(
                {
                    "kind": "kame_comparison_result",
                    "name": candidate.get("name"),
                    "metrics": _null_metric_template(candidate.get("required_metrics")),
                    "notes": "Fill from a paired evaluation over the same utterance/case set.",
                }
            )
        )

    for name, notes in REQUIRED_DGX_SPARK_SMOKES:
        entry = _with_voiceops_projection_fields(
            {"kind": "kame_smoke_result", "name": name, "ok": False, "notes": notes}
        )
        if name == "all_local_smoke":
            entry.update(
                {
                    "local_turns": None,
                    "local_turn_oracle_calls": None,
                    "oracle_bound_turns": None,
                    "oracle_bound_oracle_calls": None,
                    "oracle_authority_routes": [],
                    "interface_input_sources": [],
                    "reflex_providers": [],
                    "oracle_selected_by": "Hermes /model",
                    "components": {
                        "reflex": None,
                        "oracle": None,
                        "asr": None,
                        "tts": None,
                        "sidecar": None,
                    },
                    "metrics": {
                        "speech_end_to_first_audio_ms": None,
                        "barge_in_stop_ms": None,
                    },
                }
            )
        elif name == "cloud_fallback_smoke":
            entry.update(
                {
                    "fallback_trigger": None,
                    "fallback_mode": None,
                    "fallback_reason_visible": False,
                    "configured_policy_applied": False,
                }
            )
        elif name == "capability_honesty_smoke":
            entry.update(
                {
                    "voice_active": False,
                    "voice_capability_checks": None,
                    "voice_denial_count": None,
                    "unsupported_voice_claims": None,
                }
            )
        elif name == "barge_in_interruption_smoke":
            entry.update(
                {
                    "trigger_reason": None,
                    "playback_active": False,
                    "stop_latency_ms": None,
                    "interrupted_response_committed": True,
                }
            )
        template.append(entry)
    assumptions = matrix.get("model_assumptions") if isinstance(matrix.get("model_assumptions"), Mapping) else {}
    for name, assumption in assumptions.items():
        if not isinstance(assumption, Mapping) or assumption.get("required") is not True:
            continue
        template.append(
            _with_voiceops_projection_fields(
                {
                    "kind": "kame_model_assumption_result",
                    "name": str(name),
                    "validated_by": str(assumption.get("validated_by") or ""),
                    "model": str(assumption.get("model") or "") or None,
                    "ok": False,
                    "notes": "Set ok=true only after this model/runtime assumption has been validated on the target DGX Spark.",
                }
            )
        )
    return template


def _with_voiceops_projection_fields(entry: dict[str, Any]) -> dict[str, Any]:
    """Add provenance fields required by the VoiceOps Spark matrix adapter."""

    return {
        "schema_version": VOICEOPS_SPARK_EVIDENCE_SCHEMA_VERSION,
        "hardware": "1x DGX Spark",
        "locality": "local_spark",
        "verified": False,
        "measured_at": None,
        "source_artifact": None,
        "voiceops_projection_notes": (
            "Replace source_artifact with a redacted raw benchmark artifact path that resolves beside "
            "the evidence file; set measured_at and verified=true only after collecting on the DGX Spark."
        ),
        **entry,
    }


def load_dgx_spark_benchmark_evidence(path: str | Path) -> list[dict[str, Any]]:
    evidence_path = Path(path).expanduser()
    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    wrapper_example_only = False
    if isinstance(data, Mapping) and isinstance(data.get("evidence"), list):
        wrapper_example_only = data.get("example_only") is True
        data = data["evidence"]
    if not isinstance(data, list):
        raise ValueError("DGX Spark KAME benchmark evidence must be a JSON array or an object with an evidence array")
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, Mapping):
            raise ValueError(f"DGX Spark KAME benchmark evidence entry {index} must be an object")
        entry_copy = dict(entry)
        entry_copy["_evidence_path"] = str(evidence_path)
        if wrapper_example_only:
            entry_copy["example_only"] = True
        entries.append(entry_copy)
    return entries


def validate_dgx_spark_benchmark_evidence(
    matrix: Mapping[str, Any],
    entries: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate headless DGX Spark KAME benchmark/evidence results.

    Expected evidence entries are intentionally simple JSON objects:
    - ``kind=kame_benchmark_result`` with ``category`` (interface/oracle/speech),
      optional ``input``, ``role``, ``model``, ``adapter``, or ``asr_hypothesis``,
      and a ``metrics`` object.
    - ``kind=kame_comparison_result`` with one of the generated comparison names
      and a ``metrics`` object from a paired evaluation over the same test set.
    - ``kind=kame_model_assumption_result`` with a required model assumption
      ``name``, matching ``validated_by``, and ``ok=true``.
    - ``kind=kame_smoke_result`` with one of ``REQUIRED_DGX_SPARK_SMOKES``
      as ``name`` and ``ok=true``.
    """

    issues: list[str] = []
    candidates = matrix.get("candidates") if isinstance(matrix.get("candidates"), Mapping) else {}
    if not isinstance(candidates, Mapping):
        return {"ok": False, "issues": ["matrix: missing candidates mapping"], "coverage": {}}

    coverage: dict[str, bool] = {}
    projection_issues = _voiceops_matrix_projection_issues(entries)
    coverage["voiceops_matrix_projection_ready"] = not projection_issues
    issues.extend(projection_issues)
    quality_targets = matrix.get("acceptance_targets_ms") if isinstance(matrix.get("acceptance_targets_ms"), Mapping) else {}
    interface_decision_target_ms = _positive_metric_target(quality_targets.get("local_ack_first_audio"), default=500.0)
    interface_first_audio_target_ms = _positive_metric_target(quality_targets.get("local_reply_first_audio"), default=1000.0)
    oracle_first_audio_target_ms = _positive_metric_target(quality_targets.get("simple_oracle_first_audio"), default=3000.0)
    oracle_first_tts_target_ms = _positive_metric_target(
        quality_targets.get("oracle_first_token_to_first_tts_audio"),
        default=1000.0,
    )
    playback_start_target_ms = _positive_metric_target(
        quality_targets.get("first_tts_audio_to_playback_start"),
        default=150.0,
    )
    barge_in_stop_target_ms = _positive_metric_target(
        quality_targets.get("barge_in_stop"),
        default=150.0,
    )
    direct_audio_latency_ok = True
    oracle_latency_ok = True
    comparison_candidates = candidates.get("comparison") if isinstance(candidates.get("comparison"), list) else []
    comparison_required_metrics = {
        str(candidate.get("name") or "").strip(): candidate.get("required_metrics")
        for candidate in comparison_candidates
        if isinstance(candidate, Mapping)
    }
    interface_candidates = candidates.get("interface") if isinstance(candidates.get("interface"), list) else []
    interface_models: set[str] = set()
    for candidate in interface_candidates:
        if not isinstance(candidate, Mapping):
            continue
        model = str(candidate.get("model") or "").strip()
        input_mode = str(candidate.get("input") or "").strip()
        if model:
            interface_models.add(model)
        label = f"interface:{model}:{input_mode}" if model else f"interface:{input_mode}"
        match = _find_benchmark_entry(entries, category="interface", model=model, input_mode=input_mode)
        coverage[label] = match is not None
        if match is None:
            issues.append(f"{label}: missing benchmark result")
            if input_mode == "direct_audio":
                direct_audio_latency_ok = False
            continue
        issues.extend(_missing_metric_issues(label, match, candidate.get("required_metrics")))
        if input_mode == "direct_audio":
            latency_issues = _interface_direct_audio_latency_issues(
                label,
                match,
                decision_target_ms=interface_decision_target_ms,
                first_audio_target_ms=interface_first_audio_target_ms,
            )
            if latency_issues:
                direct_audio_latency_ok = False
                issues.extend(latency_issues)

    coverage["interface_candidate_model_matrix"] = bool(interface_candidates) and all(
        coverage.get(
            f"interface:{str(candidate.get('model') or '').strip()}:{str(candidate.get('input') or '').strip()}"
        )
        is True
        for candidate in interface_candidates
        if isinstance(candidate, Mapping)
    )
    if not coverage["interface_candidate_model_matrix"]:
        issues.append("interface_candidate_model_matrix: requires benchmark results for every interface model/input")

    has_direct = bool(interface_models) and all(
        coverage.get(f"interface:{model}:direct_audio") is True for model in interface_models
    )
    has_fallback = bool(interface_models) and all(
        coverage.get(f"interface:{model}:stt_fallback") is True for model in interface_models
    )
    interface_comparison_issues = _comparison_issues(
        entries,
        "interface_direct_audio_vs_stt_fallback",
        required_metrics=comparison_required_metrics.get("interface_direct_audio_vs_stt_fallback"),
        minimum_paired_count=10,
    )
    coverage["interface_direct_audio_vs_stt_fallback"] = (
        has_direct and has_fallback and not interface_comparison_issues
    )
    if not has_direct or not has_fallback:
        issues.append(
            "interface_direct_audio_vs_stt_fallback: "
            "requires direct_audio and stt_fallback results for every interface model"
        )
    issues.extend(interface_comparison_issues)
    coverage["interface_direct_audio_latency"] = has_direct and direct_audio_latency_ok
    if not coverage["interface_direct_audio_latency"]:
        issues.append(
            "interface_direct_audio_latency: "
            "requires direct_audio speech_end_to_interface_decision_ms and "
            "speech_end_to_local_first_audio_ms within configured targets"
        )

    oracle_candidates = candidates.get("oracle") if isinstance(candidates.get("oracle"), list) else []
    for candidate in oracle_candidates:
        if not isinstance(candidate, Mapping):
            continue
        label = "oracle:local"
        match = _find_benchmark_entry(entries, category="oracle")
        coverage[label] = match is not None
        if match is None:
            issues.append(f"{label}: missing benchmark result")
            oracle_latency_ok = False
            continue
        issues.extend(_missing_metric_issues(label, match, candidate.get("required_metrics")))
        latency_issues = _oracle_first_audio_latency_issues(
            label,
            match,
            first_audio_target_ms=oracle_first_audio_target_ms,
            first_tts_target_ms=oracle_first_tts_target_ms,
            playback_start_target_ms=playback_start_target_ms,
        )
        if latency_issues:
            oracle_latency_ok = False
            issues.extend(latency_issues)
    coverage["oracle_simple_first_audio_latency"] = bool(oracle_candidates) and oracle_latency_ok
    if not coverage["oracle_simple_first_audio_latency"]:
        issues.append(
            "oracle_simple_first_audio_latency: "
            "requires oracle_request_to_accepted_ms, oracle_accepted_to_first_token_ms, "
            "oracle_first_token_to_first_tts_audio_ms, and first_tts_audio_to_playback_start_ms "
            "within configured targets"
        )

    oracle_outcome_candidates = (
        candidates.get("oracle_outcome") if isinstance(candidates.get("oracle_outcome"), list) else []
    )
    for candidate in oracle_outcome_candidates:
        if not isinstance(candidate, Mapping):
            continue
        asr_hypothesis = str(candidate.get("asr_hypothesis") or "").strip()
        label = f"oracle_outcome:{asr_hypothesis}"
        match = _find_benchmark_entry(entries, category="oracle_outcome", asr_hypothesis=asr_hypothesis)
        coverage[label] = match is not None
        if match is None:
            issues.append(f"{label}: missing benchmark result")
            continue
        issues.extend(_missing_metric_issues(label, match, candidate.get("required_metrics")))

    has_oracle_without_asr = coverage.get("oracle_outcome:without_asr_hypothesis") is True
    has_oracle_with_asr = coverage.get("oracle_outcome:with_asr_hypothesis") is True
    oracle_asr_comparison_issues = _comparison_issues(
        entries,
        "oracle_outcome_asr_hypothesis_delta",
        required_metrics=comparison_required_metrics.get("oracle_outcome_asr_hypothesis_delta"),
        minimum_paired_count=10,
    )
    if not oracle_asr_comparison_issues:
        oracle_asr_comparison_issues.extend(_oracle_asr_outcome_delta_issues(entries))
    coverage["oracle_outcomes_with_and_without_asr_hypotheses"] = (
        has_oracle_without_asr and has_oracle_with_asr and not oracle_asr_comparison_issues
    )
    if not has_oracle_without_asr or not has_oracle_with_asr:
        issues.append(
            "oracle_outcomes_with_and_without_asr_hypotheses: "
            "requires with_asr_hypothesis and without_asr_hypothesis results"
        )
    issues.extend(oracle_asr_comparison_issues)

    speech_candidates = candidates.get("speech") if isinstance(candidates.get("speech"), list) else []
    speech_coverage_labels: list[str] = []
    speech_production_labels: list[str] = []
    for candidate in speech_candidates:
        if not isinstance(candidate, Mapping):
            continue
        role = str(candidate.get("role") or "").strip()
        model = str(candidate.get("model") or "").strip()
        adapter = str(candidate.get("adapter") or "").strip()
        label = _speech_candidate_label(role=role, model=model, adapter=adapter)
        match = _find_benchmark_entry(entries, category="speech", role=role, model=model, adapter=adapter)
        coverage[label] = match is not None
        speech_coverage_labels.append(label)
        if candidate.get("protocol_smoke_only") is not True:
            speech_production_labels.append(label)
        if match is None:
            issues.append(f"{label}: missing benchmark result")
            continue
        issues.extend(_missing_metric_issues(label, match, candidate.get("required_metrics")))
        if match.get("protocol_smoke_only") is True:
            issues.append(f"{label}: protocol smoke bridge cannot satisfy local speech benchmark evidence")
            coverage[label] = False

    coverage["oracle_verbatim_asr_latency_and_literal_accuracy"] = any(
        coverage.get(label) is True for label in speech_coverage_labels if label.startswith("speech:oracle_verbatim_asr:")
    )
    coverage["local_asr_tts_benchmark_matrix"] = (
        len(speech_production_labels) == len(speech_candidates)
        and any(
            coverage.get(label) is True for label in speech_production_labels if label.startswith("speech:oracle_verbatim_asr:")
        )
        and any(coverage.get(label) is True for label in speech_production_labels if label.startswith("speech:tts:"))
    )
    if not coverage["local_asr_tts_benchmark_matrix"]:
        issues.append(
            "local_asr_tts_benchmark_matrix: requires benchmark evidence for non-loopback local ASR and TTS adapters"
        )

    required_assumptions = (
        matrix.get("model_assumptions") if isinstance(matrix.get("model_assumptions"), Mapping) else {}
    )
    assumption_labels: list[str] = []
    for name, assumption in required_assumptions.items():
        if not isinstance(assumption, Mapping) or assumption.get("required") is not True:
            continue
        assumption_name = str(name)
        validated_by = str(assumption.get("validated_by") or "").strip()
        label = f"model_assumption:{assumption_name}"
        assumption_labels.append(label)
        ok = _has_passing_model_assumption(entries, assumption_name, validated_by)
        coverage[label] = ok
        if not ok:
            issues.append(
                f"{label}: missing passing model assumption result"
                f"{f' validated_by={validated_by}' if validated_by else ''}"
            )
    coverage["model_assumptions_validated"] = bool(assumption_labels) and all(
        coverage.get(label) is True for label in assumption_labels
    )
    if not coverage["model_assumptions_validated"]:
        issues.append(
            "model_assumptions_validated: requires passing evidence for every required model/runtime assumption"
        )

    for smoke_name, _notes in REQUIRED_DGX_SPARK_SMOKES:
        smoke_issues = _passing_smoke_issues(
            entries,
            smoke_name,
            barge_in_stop_target_ms=barge_in_stop_target_ms,
        )
        coverage[smoke_name] = not smoke_issues
        issues.extend(smoke_issues)

    return {
        "ok": not issues,
        "issues": issues,
        "coverage": coverage,
    }


def _voiceops_matrix_projection_issues(entries: list[Mapping[str, Any]]) -> list[str]:
    issues: list[str] = []
    projected_kinds = {
        "kame_benchmark_result",
        "kame_comparison_result",
        "kame_model_assumption_result",
        "kame_smoke_result",
    }
    for index, entry in enumerate(entries):
        kind = str(entry.get("kind") or "").strip()
        if kind not in projected_kinds:
            continue
        label = f"voiceops_projection:{index}:{kind}"
        if entry.get("example_only") is True:
            issues.append(f"{label}:example_only_evidence_not_accepted")
        if str(entry.get("schema_version") or "") != VOICEOPS_SPARK_EVIDENCE_SCHEMA_VERSION:
            issues.append(f"{label}:missing_or_invalid_schema_version")
        hardware = str(entry.get("hardware") or "").strip().lower()
        if hardware not in {"1x dgx spark", "1x nvidia dgx spark", "single dgx spark"}:
            issues.append(f"{label}:missing_or_invalid_hardware")
        if str(entry.get("locality") or "").strip() != "local_spark":
            issues.append(f"{label}:missing_or_invalid_locality")
        if entry.get("verified") is not True and entry.get("ok") is not True:
            issues.append(f"{label}:verified_not_true")
        if not str(entry.get("measured_at") or "").strip():
            issues.append(f"{label}:missing_measured_at")
        elif not _has_parseable_timezone_timestamp(entry.get("measured_at")):
            issues.append(f"{label}:invalid_measured_at")
        if not str(entry.get("source_artifact") or "").strip():
            issues.append(f"{label}:missing_source_artifact")
        else:
            issues.extend(f"{label}:{issue}" for issue in _projection_source_artifact_issues(entry))
    return issues


def _projection_source_artifact_issues(entry: Mapping[str, Any]) -> list[str]:
    evidence_path_text = str(entry.get("_evidence_path") or "").strip()
    if not evidence_path_text:
        return []
    source_path = Path(str(entry.get("source_artifact") or "").strip()).expanduser()
    if not source_path.is_absolute():
        source_path = Path(evidence_path_text).expanduser().parent / source_path
    if not source_path.exists():
        return ["source_artifact_not_found"]
    if not source_path.is_file():
        return ["source_artifact_not_file"]
    try:
        with source_path.open("rb") as handle:
            handle.read(1)
    except OSError:
        return ["source_artifact_unreadable"]
    return []


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


def _find_benchmark_entry(
    entries: list[Mapping[str, Any]],
    *,
    category: str,
    model: str = "",
    input_mode: str = "",
    role: str = "",
    adapter: str = "",
    asr_hypothesis: str = "",
) -> Mapping[str, Any] | None:
    for entry in entries:
        if str(entry.get("kind") or "") != "kame_benchmark_result":
            continue
        if str(entry.get("category") or "") != category:
            continue
        if model and str(entry.get("model") or "") != model:
            continue
        if input_mode and str(entry.get("input") or "") != input_mode:
            continue
        if role and str(entry.get("role") or "") != role:
            continue
        if adapter and str(entry.get("adapter") or "") != adapter:
            continue
        if asr_hypothesis and str(entry.get("asr_hypothesis") or "") != asr_hypothesis:
            continue
        return entry
    return None


def _find_comparison_entry(entries: list[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    for entry in entries:
        if str(entry.get("kind") or "") != "kame_comparison_result":
            continue
        if str(entry.get("name") or "") == name:
            return entry
    return None


def _interface_candidate_models(
    interface_model: str,
    requested_candidates: list[str] | tuple[str, ...] | None,
) -> list[str]:
    candidates = [interface_model]
    candidates.extend(requested_candidates or DEFAULT_INTERFACE_CANDIDATE_MODELS)
    return _unique_nonempty(candidates)


def _interface_candidate_model_names(interface_role: Mapping[str, Any]) -> list[str]:
    candidates = interface_role.get("candidate_models")
    names: list[str] = []
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                names.append(str(candidate.get("model") or ""))
            else:
                names.append(str(candidate or ""))
    names.append(str(interface_role.get("model") or ""))
    return _unique_nonempty(names)


def _unique_nonempty(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _clean_nonempty(value: Any, *, default: str) -> str:
    normalized = str(value or "").strip()
    return normalized or default


def _clean_env_name(value: Any, *, default: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return default
    if normalized[0].isdigit() or not normalized.replace("_", "").isalnum():
        return default
    return normalized


def _env_bearer_token(env_name: str, *fallback_env_names: str) -> str:
    for candidate in (env_name, *fallback_env_names):
        name = _clean_env_name(candidate, default="")
        if not name:
            continue
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _python_module_name(value: Any, *, default: str) -> str:
    normalized = _clean_nonempty(value, default=default)
    if not all(part.isidentifier() for part in normalized.split(".")):
        return default
    return normalized


def _speech_candidate_label(*, role: str, model: str, adapter: str) -> str:
    parts = ["speech", role or "unknown"]
    if model:
        parts.append(model)
    if adapter:
        parts.append(adapter)
    return ":".join(parts)


def _missing_metric_issues(label: str, entry: Mapping[str, Any], required_metrics: Any) -> list[str]:
    if not isinstance(required_metrics, list):
        return [f"{label}: matrix candidate has no required_metrics list"]
    metrics = entry.get("metrics")
    if not isinstance(metrics, Mapping):
        return [f"{label}: missing metrics object"]
    issues: list[str] = []
    for metric in required_metrics:
        metric_name = str(metric or "").strip()
        if not metric_name:
            continue
        value = metrics.get(metric_name)
        if not _valid_metric_value(metric_name, value):
            issues.append(f"{label}: missing or invalid metric {metric_name}")
    return issues


def _comparison_issues(
    entries: list[Mapping[str, Any]],
    name: str,
    *,
    required_metrics: Any,
    minimum_paired_count: int,
) -> list[str]:
    entry = _find_comparison_entry(entries, name)
    label = f"comparison:{name}"
    if entry is None:
        return [f"{label}: missing paired comparison result"]
    metrics = entry.get("metrics")
    if not isinstance(metrics, Mapping):
        return [f"{label}: missing metrics object"]
    issues = _missing_metric_issues(label, entry, required_metrics)
    count_key = "paired_cases" if "paired_cases" in metrics else "paired_turns"
    paired_count = _metric_float(metrics.get(count_key))
    if paired_count is None or paired_count < minimum_paired_count:
        issues.append(f"{label}: requires {count_key} >= {minimum_paired_count}")
    return issues


def _oracle_asr_outcome_delta_issues(entries: list[Mapping[str, Any]]) -> list[str]:
    entry = _find_comparison_entry(entries, "oracle_outcome_asr_hypothesis_delta")
    if entry is None:
        return []
    metrics = entry.get("metrics")
    if not isinstance(metrics, Mapping):
        return []
    label = "comparison:oracle_outcome_asr_hypothesis_delta"
    issues: list[str] = []
    with_literal = _metric_float(metrics.get("with_asr_literal_argument_accuracy"))
    without_literal = _metric_float(metrics.get("without_asr_literal_argument_accuracy"))
    if with_literal is not None and without_literal is not None and with_literal < without_literal:
        issues.append(
            f"{label}: with_asr_literal_argument_accuracy {with_literal:g} "
            f"is below without_asr_literal_argument_accuracy {without_literal:g}"
        )
    with_errors = _metric_float(metrics.get("with_asr_tool_argument_error_rate"))
    without_errors = _metric_float(metrics.get("without_asr_tool_argument_error_rate"))
    if with_errors is not None and without_errors is not None and with_errors > without_errors:
        issues.append(
            f"{label}: with_asr_tool_argument_error_rate {with_errors:g} "
            f"exceeds without_asr_tool_argument_error_rate {without_errors:g}"
        )
    return issues


def _null_metric_template(required_metrics: Any) -> dict[str, None]:
    if not isinstance(required_metrics, list):
        return {}
    metrics: dict[str, None] = {}
    for metric in required_metrics:
        metric_name = str(metric or "").strip()
        if metric_name:
            metrics[metric_name] = None
    return metrics


def _valid_metric_value(metric_name: str, value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    if not parsed >= 0:
        return False
    if metric_name == "sample_count":
        return parsed >= 1
    if "accuracy" in metric_name or metric_name.endswith("_rate"):
        return parsed <= 1.0
    return True


def _interface_direct_audio_latency_issues(
    label: str,
    entry: Mapping[str, Any],
    *,
    decision_target_ms: float,
    first_audio_target_ms: float,
) -> list[str]:
    metrics = entry.get("metrics")
    if not isinstance(metrics, Mapping):
        return [f"{label}: missing metrics object"]
    issues: list[str] = []
    decision_ms = _metric_float(metrics.get("speech_end_to_interface_decision_ms"))
    if decision_ms is None:
        issues.append(f"{label}: missing valid speech_end_to_interface_decision_ms")
    elif decision_ms > decision_target_ms:
        issues.append(
            f"{label}: speech_end_to_interface_decision_ms {decision_ms:g} exceeds target {decision_target_ms:g}"
        )
    decision_p90_ms = _metric_float(metrics.get("speech_end_to_interface_decision_p90_ms"))
    if decision_p90_ms is None:
        issues.append(f"{label}: missing valid speech_end_to_interface_decision_p90_ms")
    elif decision_p90_ms > decision_target_ms:
        issues.append(
            f"{label}: speech_end_to_interface_decision_p90_ms {decision_p90_ms:g} "
            f"exceeds target {decision_target_ms:g}"
        )
    first_audio_ms = _metric_float(metrics.get("speech_end_to_local_first_audio_ms"))
    if first_audio_ms is None:
        issues.append(f"{label}: missing valid speech_end_to_local_first_audio_ms")
    elif first_audio_ms > first_audio_target_ms:
        issues.append(
            f"{label}: speech_end_to_local_first_audio_ms {first_audio_ms:g} exceeds target {first_audio_target_ms:g}"
        )
    first_audio_p90_ms = _metric_float(metrics.get("speech_end_to_local_first_audio_p90_ms"))
    if first_audio_p90_ms is None:
        issues.append(f"{label}: missing valid speech_end_to_local_first_audio_p90_ms")
    elif first_audio_p90_ms > first_audio_target_ms:
        issues.append(
            f"{label}: speech_end_to_local_first_audio_p90_ms {first_audio_p90_ms:g} "
            f"exceeds target {first_audio_target_ms:g}"
        )
    return issues


def _oracle_first_audio_latency_issues(
    label: str,
    entry: Mapping[str, Any],
    *,
    first_audio_target_ms: float,
    first_tts_target_ms: float,
    playback_start_target_ms: float,
) -> list[str]:
    metrics = entry.get("metrics")
    if not isinstance(metrics, Mapping):
        return [f"{label}: missing metrics object"]
    metric_names = (
        "oracle_request_to_accepted_ms",
        "oracle_accepted_to_first_token_ms",
        "oracle_first_token_to_first_tts_audio_ms",
        "first_tts_audio_to_playback_start_ms",
    )
    total = 0.0
    issues: list[str] = []
    parsed_metrics: dict[str, float] = {}
    for metric_name in metric_names:
        value = _metric_float(metrics.get(metric_name))
        if value is None:
            issues.append(f"{label}: missing valid {metric_name}")
        else:
            parsed_metrics[metric_name] = value
            total += value
    if issues:
        return issues
    first_tts_ms = parsed_metrics["oracle_first_token_to_first_tts_audio_ms"]
    if first_tts_ms > first_tts_target_ms:
        issues.append(
            f"{label}: oracle_first_token_to_first_tts_audio_ms {first_tts_ms:g} "
            f"exceeds target {first_tts_target_ms:g}"
        )
    playback_start_ms = parsed_metrics["first_tts_audio_to_playback_start_ms"]
    if playback_start_ms > playback_start_target_ms:
        issues.append(
            f"{label}: first_tts_audio_to_playback_start_ms {playback_start_ms:g} "
            f"exceeds target {playback_start_target_ms:g}"
        )
    if total > first_audio_target_ms:
        issues.append(f"{label}: oracle first audio total {total:g} exceeds target {first_audio_target_ms:g}")
    first_audio_p90_ms = _metric_float(metrics.get("oracle_request_to_first_audio_p90_ms"))
    if first_audio_p90_ms is None:
        issues.append(f"{label}: missing valid oracle_request_to_first_audio_p90_ms")
    elif first_audio_p90_ms > first_audio_target_ms:
        issues.append(
            f"{label}: oracle_request_to_first_audio_p90_ms {first_audio_p90_ms:g} "
            f"exceeds target {first_audio_target_ms:g}"
        )
    return issues


def _positive_metric_target(value: Any, *, default: float) -> float:
    parsed = _metric_float(value)
    return parsed if parsed is not None and parsed > 0 else default


def _metric_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _passing_smoke_issues(
    entries: list[Mapping[str, Any]],
    name: str,
    *,
    barge_in_stop_target_ms: float,
) -> list[str]:
    entry = _passing_smoke_entry(entries, name)
    if entry is None:
        return [f"{name}: missing passing smoke result"]
    if name == "all_local_smoke":
        return _all_local_smoke_issues(entry, barge_in_stop_target_ms=barge_in_stop_target_ms)
    if name == "cloud_fallback_smoke":
        return _cloud_fallback_smoke_issues(entry)
    if name == "capability_honesty_smoke":
        return _capability_honesty_smoke_issues(entry)
    if name == "barge_in_interruption_smoke":
        return _barge_in_interruption_smoke_issues(
            entry,
            stop_target_ms=barge_in_stop_target_ms,
        )
    return []


def _passing_smoke_entry(entries: list[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    for entry in entries:
        if str(entry.get("kind") or "") != "kame_smoke_result":
            continue
        if str(entry.get("name") or "") == name and entry.get("ok") is True:
            return entry
    return None


def _all_local_smoke_issues(entry: Mapping[str, Any], *, barge_in_stop_target_ms: float) -> list[str]:
    issues: list[str] = []
    local_turns = _metric_float(entry.get("local_turns"))
    local_oracle_calls = _metric_float(entry.get("local_turn_oracle_calls"))
    oracle_bound_turns = _metric_float(entry.get("oracle_bound_turns"))
    oracle_bound_calls = _metric_float(entry.get("oracle_bound_oracle_calls"))
    if local_turns is None or local_turns < 1:
        issues.append("all_local_smoke: requires local_turns >= 1")
    if local_oracle_calls is None or local_oracle_calls != 0:
        issues.append("all_local_smoke: requires local_turn_oracle_calls == 0")
    if oracle_bound_turns is None or oracle_bound_turns < 1:
        issues.append("all_local_smoke: requires oracle_bound_turns >= 1")
    if oracle_bound_calls is None or oracle_bound_calls < oracle_bound_turns:
        issues.append("all_local_smoke: requires oracle_bound_oracle_calls >= oracle_bound_turns")
    if str(entry.get("oracle_selected_by") or "") != "Hermes /model":
        issues.append("all_local_smoke: requires oracle_selected_by == Hermes /model")
    components = entry.get("components")
    if not isinstance(components, Mapping):
        issues.append("all_local_smoke: requires components mapping")
    else:
        missing_components = [
            name
            for name in ("reflex", "oracle", "asr", "tts", "sidecar")
            if components.get(name) is not True
        ]
        if missing_components:
            issues.append("all_local_smoke: components missing " + ",".join(missing_components))
    metrics = entry.get("metrics")
    if not isinstance(metrics, Mapping):
        issues.append("all_local_smoke: requires metrics mapping")
    else:
        first_audio_ms = _metric_float(metrics.get("speech_end_to_first_audio_ms"))
        if first_audio_ms is None:
            issues.append("all_local_smoke: requires metrics.speech_end_to_first_audio_ms")
        elif first_audio_ms > 1500:
            issues.append("all_local_smoke: metrics.speech_end_to_first_audio_ms exceeds 1500")
        barge_in_stop_ms = _metric_float(metrics.get("barge_in_stop_ms"))
        if barge_in_stop_ms is None:
            issues.append("all_local_smoke: requires metrics.barge_in_stop_ms")
        elif barge_in_stop_ms > barge_in_stop_target_ms:
            issues.append("all_local_smoke: metrics.barge_in_stop_ms exceeds target")
    authority_routes = entry.get("oracle_authority_routes")
    if not isinstance(authority_routes, list):
        issues.append("all_local_smoke: requires oracle_authority_routes list")
    else:
        covered = {str(route or "").strip() for route in authority_routes}
        required = {"tools", "files", "memory", "project_context"}
        missing = sorted(required.difference(covered))
        if missing:
            issues.append(
                "all_local_smoke: oracle_authority_routes missing " + ",".join(missing)
            )
    interface_input_sources = entry.get("interface_input_sources")
    if not isinstance(interface_input_sources, list):
        issues.append("all_local_smoke: requires interface_input_sources list")
    else:
        sources = {str(source or "").strip() for source in interface_input_sources}
        if "native_audio" not in sources:
            issues.append("all_local_smoke: interface_input_sources missing native_audio")
    reflex_providers = entry.get("reflex_providers")
    if not isinstance(reflex_providers, list):
        issues.append("all_local_smoke: requires reflex_providers list")
    else:
        providers = {str(provider or "").strip() for provider in reflex_providers}
        if "vllm" not in providers:
            issues.append("all_local_smoke: reflex_providers missing vllm")
    return issues


def _cloud_fallback_smoke_issues(entry: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    trigger = str(entry.get("fallback_trigger") or "").strip()
    allowed_triggers = {
        "sidecar_unavailable",
        "interface_provider_unavailable",
        "local_provider_unavailable",
        "tts_unavailable",
        "audio_native_reflex_unavailable",
    }
    if trigger not in allowed_triggers:
        issues.append("cloud_fallback_smoke: requires recognized fallback_trigger")
    mode = str(entry.get("fallback_mode") or "").strip()
    allowed_modes = {"legacy_voice", "text_only", "stt_fed_reflex", "cloud_provider"}
    if mode not in allowed_modes:
        issues.append("cloud_fallback_smoke: requires recognized fallback_mode")
    if entry.get("fallback_reason_visible") is not True:
        issues.append("cloud_fallback_smoke: requires fallback_reason_visible == true")
    if entry.get("configured_policy_applied") is not True:
        issues.append("cloud_fallback_smoke: requires configured_policy_applied == true")
    return issues


def _capability_honesty_smoke_issues(entry: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    checks = _metric_float(entry.get("voice_capability_checks"))
    denial_count = _metric_float(entry.get("voice_denial_count"))
    unsupported_claims = _metric_float(entry.get("unsupported_voice_claims"))
    if entry.get("voice_active") is not True:
        issues.append("capability_honesty_smoke: requires voice_active == true")
    if checks is None or checks < 1:
        issues.append("capability_honesty_smoke: requires voice_capability_checks >= 1")
    if denial_count is None or denial_count != 0:
        issues.append("capability_honesty_smoke: requires voice_denial_count == 0")
    if unsupported_claims is None or unsupported_claims != 0:
        issues.append("capability_honesty_smoke: requires unsupported_voice_claims == 0")
    return issues


def _barge_in_interruption_smoke_issues(
    entry: Mapping[str, Any],
    *,
    stop_target_ms: float,
) -> list[str]:
    issues: list[str] = []
    trigger = str(entry.get("trigger_reason") or "").strip()
    stop_latency = _metric_float(entry.get("stop_latency_ms"))
    if trigger != "confirmed_user_speech":
        issues.append("barge_in_interruption_smoke: requires trigger_reason == confirmed_user_speech")
    if entry.get("playback_active") is not True:
        issues.append("barge_in_interruption_smoke: requires playback_active == true")
    if stop_latency is None:
        issues.append("barge_in_interruption_smoke: requires stop_latency_ms")
    elif stop_latency > stop_target_ms:
        issues.append(
            f"barge_in_interruption_smoke: stop_latency_ms {stop_latency:g} "
            f"exceeds target {stop_target_ms:g}"
        )
    if entry.get("interrupted_response_committed") is not False:
        issues.append("barge_in_interruption_smoke: requires interrupted_response_committed == false")
    return issues


def _has_passing_model_assumption(
    entries: list[Mapping[str, Any]],
    name: str,
    validated_by: str = "",
) -> bool:
    for entry in entries:
        if str(entry.get("kind") or "") != "kame_model_assumption_result":
            continue
        if str(entry.get("name") or "") != name:
            continue
        if validated_by and str(entry.get("validated_by") or "") != validated_by:
            continue
        if entry.get("ok") is True:
            return True
    return False


def preflight_dgx_spark_stack(
    manifest: Mapping[str, Any],
    *,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    roles = _roles(manifest)
    interface_bearer_token = _env_bearer_token(
        str(roles["interface"].get("api_key_env") or DEFAULT_INTERFACE_API_KEY_ENV),
        "HERMES_VOICE_VLLM_TOKEN",
    )
    checks = {
        "interface_models": probe_json_endpoint(
            roles["interface"]["models_url"],
            timeout_seconds=timeout_seconds,
            expected_model=roles["interface"]["model"],
            bearer_token=interface_bearer_token,
        ),
        "interface_audio_probe": probe_openai_audio_chat_completion(
            roles["interface"]["base_url"],
            model=roles["interface"]["model"],
            timeout_seconds=timeout_seconds,
            bearer_token=interface_bearer_token,
        ),
        "oracle_models": probe_json_endpoint(
            roles["oracle"]["models_url"],
            timeout_seconds=timeout_seconds,
            expected_model=roles["oracle"]["preferred_local_model"],
        ),
        "sidecar_health": probe_json_endpoint(
            roles["sidecar"]["health_url"],
            timeout_seconds=timeout_seconds,
            expected_fields=_sidecar_expected_health_fields(roles),
        ),
    }
    if roles["asr"].get("health_url"):
        checks["asr_health"] = probe_json_endpoint(
            roles["asr"]["health_url"],
            timeout_seconds=timeout_seconds,
        )
    if roles["tts"].get("health_url"):
        checks["tts_health"] = probe_json_endpoint(
            roles["tts"]["health_url"],
            timeout_seconds=timeout_seconds,
        )
    return {
        "ok": all(check["ok"] for check in checks.values()),
        "checks": checks,
    }


def probe_openai_audio_chat_completion(
    base_url: str,
    *,
    model: str,
    timeout_seconds: float,
    bearer_token: str = "",
) -> dict[str, Any]:
    """Probe that the interface endpoint accepts audio prompts and emits KAME JSON."""

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": _silence_wav_data_url()}},
                    {"type": "text", "text": kame_reflex_instruction_text(preflight=True)},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 80,
        "response_format": {"type": "json_object"},
    }
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", 200)
            body = response.read()
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        return {"ok": False, "url": url, "model": model, "error": str(exc)}
    try:
        response_payload = json.loads(body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"ok": False, "url": url, "model": model, "status": status, "error": f"invalid_json: {exc}"}

    content = _chat_completion_message_content(response_payload)
    schema_issues = _kame_preflight_content_schema_issues(content)
    return {
        "ok": 200 <= int(status) < 300 and not schema_issues,
        "url": url,
        "model": model,
        "status": status,
        "audio_prompt": True,
        "schema_issues": schema_issues,
    }


def probe_json_endpoint(
    url: str,
    *,
    timeout_seconds: float,
    expected_model: str | None = None,
    expected_fields: Mapping[str, Any] | None = None,
    bearer_token: str = "",
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", 200)
            body = response.read()
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        return {"ok": False, "url": url, "error": str(exc)}
    try:
        payload = json.loads(body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"ok": False, "url": url, "status": status, "error": f"invalid_json: {exc}"}

    model_ok = True
    if expected_model:
        model_ok = _models_payload_contains(payload, expected_model)
    field_misses = _expected_field_misses(payload, expected_fields or {})
    payload_ok = payload.get("ok")
    payload_ok_found = payload_ok is not False
    return {
        "ok": 200 <= int(status) < 300 and model_ok and payload_ok_found and not field_misses,
        "url": url,
        "status": status,
        "payload_ok": payload_ok if isinstance(payload_ok, bool) else None,
        "expected_model": expected_model,
        "model_found": model_ok if expected_model else None,
        "expected_fields": dict(expected_fields or {}),
        "field_misses": field_misses,
    }


def _openai_models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _health_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/health"


def _url_host_port(base_url: str, *, default_port: int) -> tuple[str, int]:
    try:
        parsed = urllib.parse.urlparse(base_url)
    except Exception:
        return "127.0.0.1", default_port
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or default_port
    return host, port


def _models_payload_contains(payload: Mapping[str, Any], expected_model: str) -> bool:
    data = payload.get("data")
    if not isinstance(data, list):
        return False
    for item in data:
        if isinstance(item, Mapping) and item.get("id") == expected_model:
            return True
    return False


def _chat_completion_message_content(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return ""
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return ""
    return str(message.get("content") or "").strip()


def _kame_preflight_content_schema_issues(content: str) -> list[str]:
    if not content:
        return ["missing message content"]
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return [f"message content is not JSON: {exc}"]
    if not isinstance(payload, Mapping):
        return ["message content JSON is not an object"]
    return kame_reflex_schema_issues(payload)


def _silence_wav_data_url() -> str:
    sample_rate = 16_000
    channels = 1
    bits_per_sample = 16
    sample_count = sample_rate // 10
    block_align = channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    data_size = sample_count * block_align
    header = b"".join(
        [
            b"RIFF",
            (36 + data_size).to_bytes(4, "little"),
            b"WAVE",
            b"fmt ",
            (16).to_bytes(4, "little"),
            (1).to_bytes(2, "little"),
            channels.to_bytes(2, "little"),
            sample_rate.to_bytes(4, "little"),
            byte_rate.to_bytes(4, "little"),
            block_align.to_bytes(2, "little"),
            bits_per_sample.to_bytes(2, "little"),
            b"data",
            data_size.to_bytes(4, "little"),
        ]
    )
    wav = header + (b"\x00" * data_size)
    return "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")


def _sidecar_expected_health_fields(roles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "capabilities.vllm_audio_frontend": True,
        "capabilities.tts": True,
        "capabilities.streaming_tts_bridge": True,
        "frontend.streaming_tts_bridge.healthy": True,
    }
    if str(roles.get("asr", {}).get("mode") or "") != "disabled":
        expected["capabilities.streaming_stt_bridge"] = True
        expected["frontend.streaming_stt_bridge.healthy"] = True
    return expected


def _bounded_interface_max_audio_seconds(value: Any) -> float:
    if isinstance(value, bool):
        return DEFAULT_INTERFACE_MAX_AUDIO_SECONDS
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return DEFAULT_INTERFACE_MAX_AUDIO_SECONDS
    if parsed < 1.0:
        return 1.0
    if parsed > DEFAULT_INTERFACE_MAX_AUDIO_SECONDS:
        return DEFAULT_INTERFACE_MAX_AUDIO_SECONDS
    return parsed


def _expected_field_misses(payload: Mapping[str, Any], expected_fields: Mapping[str, Any]) -> list[dict[str, Any]]:
    misses: list[dict[str, Any]] = []
    for path, expected in expected_fields.items():
        found, actual = _payload_path_value(payload, str(path))
        if not found or actual != expected:
            misses.append({"path": str(path), "expected": expected, "actual": actual if found else None})
    return misses


def _payload_path_value(payload: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _roles(manifest: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("manifest has no roles mapping")
    return roles  # type: ignore[return-value]


def _env_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _compact_json_value(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    raise SystemExit(main())
