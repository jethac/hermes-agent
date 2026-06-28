"""Generate and preflight a headless DGX Spark KAME realtime voice stack."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping


DEFAULT_OUTPUT_DIR = "./artifacts/realtime-voice-dgx-spark"
DEFAULT_INTERFACE_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_INTERFACE_MODEL = "gemma-4-E2B-it"
DEFAULT_ORACLE_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_ORACLE_MODEL = "gemma-4-26B-A4B-it"
DEFAULT_SIDECAR_BASE_URL = "http://127.0.0.1:8765"
DEFAULT_ASR_BASE_URL = "http://127.0.0.1:8767"
DEFAULT_TTS_BASE_URL = "http://127.0.0.1:8768"
DEFAULT_VLLM_IMAGE = "vllm/vllm-openai:latest"
DEFAULT_HERMES_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a headless DGX Spark launch/preflight pack for KAME voice"
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--hermes-home", default="~/.hermes")
    parser.add_argument("--interface-base-url", default=DEFAULT_INTERFACE_BASE_URL)
    parser.add_argument("--interface-model", default=DEFAULT_INTERFACE_MODEL)
    parser.add_argument("--interface-context-tokens", type=int, default=8192)
    parser.add_argument("--interface-gpu-memory-utilization", type=float, default=0.18)
    parser.add_argument("--oracle-base-url", default=DEFAULT_ORACLE_BASE_URL)
    parser.add_argument("--oracle-model", default=DEFAULT_ORACLE_MODEL)
    parser.add_argument("--oracle-context-tokens", type=int, default=32768)
    parser.add_argument("--oracle-gpu-memory-utilization", type=float, default=0.62)
    parser.add_argument("--sidecar-base-url", default=DEFAULT_SIDECAR_BASE_URL)
    parser.add_argument("--asr-base-url", default=DEFAULT_ASR_BASE_URL)
    parser.add_argument("--tts-base-url", default=DEFAULT_TTS_BASE_URL)
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
    args = build_parser().parse_args(argv)
    manifest = build_dgx_spark_stack_manifest(
        repo_dir=Path(args.repo_dir).expanduser().resolve(),
        hermes_home=Path(args.hermes_home).expanduser(),
        interface_base_url=str(args.interface_base_url),
        interface_model=str(args.interface_model),
        interface_context_tokens=int(args.interface_context_tokens),
        interface_gpu_memory_utilization=float(args.interface_gpu_memory_utilization),
        oracle_base_url=str(args.oracle_base_url),
        oracle_model=str(args.oracle_model),
        oracle_context_tokens=int(args.oracle_context_tokens),
        oracle_gpu_memory_utilization=float(args.oracle_gpu_memory_utilization),
        sidecar_base_url=str(args.sidecar_base_url),
        asr_base_url=str(args.asr_base_url),
        tts_base_url=str(args.tts_base_url),
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
    interface_base_url: str,
    interface_model: str,
    interface_context_tokens: int,
    interface_gpu_memory_utilization: float,
    oracle_base_url: str,
    oracle_model: str,
    oracle_context_tokens: int,
    oracle_gpu_memory_utilization: float,
    sidecar_base_url: str,
    asr_base_url: str,
    tts_base_url: str,
    asr_mode: str,
    vllm_image: str,
    hermes_image: str,
    model_cache_dir: str,
) -> dict[str, Any]:
    interface_models_url = _openai_models_url(interface_base_url)
    oracle_models_url = _openai_models_url(oracle_base_url)
    sidecar_health_url = _health_url(sidecar_base_url)
    asr_health_url = _health_url(asr_base_url) if asr_base_url else ""
    tts_health_url = _health_url(tts_base_url) if tts_base_url else ""
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
            "max_spoken_sentences": 2,
            "durability_policy": "commit final intents and oracle results only",
        },
        "roles": {
            "interface": {
                "provider": "openai_compatible_vllm",
                "model": interface_model,
                "base_url": interface_base_url,
                "models_url": interface_models_url,
                "max_model_len": interface_context_tokens,
                "gpu_memory_utilization": interface_gpu_memory_utilization,
                "audio_input": "native_audio",
                "limit_mm_per_prompt": {"audio": 1},
                "routing": ["local", "defer", "oracle_direct", "reject_or_clarify"],
            },
            "oracle": {
                "provider": "hermes_active_oracle_or_openai_compatible_vllm",
                "preferred_local_model": oracle_model,
                "base_url": oracle_base_url,
                "models_url": oracle_models_url,
                "max_model_len": oracle_context_tokens,
                "gpu_memory_utilization": oracle_gpu_memory_utilization,
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
                "mode": asr_mode,
                "base_url": asr_base_url,
                "health_url": asr_health_url,
                "module": "hermes_cli.realtime_voice_loopback_bridge",
                "default_adapter": "loopback_smoke_bridge",
                "production_replacement": "local_streaming_asr",
                "feeds_reflex": asr_mode == "fallback",
            },
            "tts": {
                "role": "spoken_output",
                "base_url": tts_base_url,
                "health_url": tts_health_url,
                "module": "hermes_cli.realtime_voice_loopback_bridge",
                "default_adapter": "loopback_smoke_bridge",
                "production_replacement": "local_streaming_tts",
            },
        },
        "quality_targets_ms": {
            "local_ack_first_audio": 500,
            "local_reply_first_audio": 1000,
            "oracle_ack": 500,
            "simple_oracle_first_audio": 3000,
            "tool_or_context_oracle_first_audio": 8000,
            "barge_in_stop": 150,
        },
        "artifacts": {
            "compose": "compose.yaml",
            "env_example": ".env.example",
            "launch": "launch-local-stack.sh",
            "benchmark_matrix": "benchmark-matrix.json",
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
            "interface_direct_audio_vs_stt_fallback",
            "oracle_verbatim_asr_latency_and_literal_accuracy",
            "local_asr_tts_benchmark_matrix",
            "all_local_smoke",
            "cloud_fallback_smoke",
        ],
    }


def write_dgx_spark_stack_artifacts(output_dir: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "manifest": output_dir / "manifest.json",
        "compose": output_dir / "compose.yaml",
        "env_example": output_dir / ".env.example",
        "launch": output_dir / "launch-local-stack.sh",
        "benchmark_matrix": output_dir / "benchmark-matrix.json",
    }
    files["manifest"].write_text(_json(manifest), encoding="utf-8")
    files["compose"].write_text(render_dgx_spark_compose(manifest), encoding="utf-8")
    files["env_example"].write_text(render_dgx_spark_env_example(manifest), encoding="utf-8")
    files["launch"].write_text(render_dgx_spark_launch_script(manifest), encoding="utf-8")
    files["launch"].chmod(0o755)
    files["benchmark_matrix"].write_text(
        _json(build_dgx_spark_benchmark_matrix(manifest)),
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
      - '{{"audio":1}}'

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

  hermes-realtime-sidecar:
    image: ${{HERMES_DGX_SPARK_HERMES_IMAGE:-{images.get("hermes", DEFAULT_HERMES_IMAGE)}}}
    working_dir: /workspace/hermes-agent
    depends_on:
      - kame-interface-vllm
      - kame-asr-bridge
      - kame-tts-bridge
    ports:
      - "8765:8765"
    volumes:
      - ${{HERMES_REPO_DIR:-{manifest["repo_dir"]}}}:/workspace/hermes-agent
      - ${{HERMES_HOME:-{manifest["hermes_home"]}}}:/root/.hermes
    environment:
      HERMES_HOME: /root/.hermes
      HERMES_VOICE_VLLM_BASE_URL: {interface["base_url"]}
      HERMES_VOICE_VLLM_MODEL: {interface["model"]}
      HERMES_VOICE_STREAMING_STT_BASE_URL: {asr["base_url"]}
      HERMES_VOICE_STREAMING_STT_MODEL: ${{HERMES_VOICE_STREAMING_STT_MODEL:-oracle-verbatim-asr}}
      HERMES_VOICE_STREAMING_TTS_BASE_URL: {tts["base_url"]}
      HERMES_VOICE_STREAMING_TTS_MODEL: ${{HERMES_VOICE_STREAMING_TTS_MODEL:-local-streaming-tts}}
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
      - --vllm-base-url
      - {interface["base_url"]}
      - --vllm-model
      - {interface["model"]}
      - --streaming-stt-base-url
      - {asr["base_url"]}
      - --streaming-stt-model
      - ${{HERMES_VOICE_STREAMING_STT_MODEL:-oracle-verbatim-asr}}
      - --streaming-tts-base-url
      - {tts["base_url"]}
      - --streaming-tts-model
      - ${{HERMES_VOICE_STREAMING_TTS_MODEL:-local-streaming-tts}}

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
      - --production-en-ja
"""


def render_dgx_spark_env_example(manifest: Mapping[str, Any]) -> str:
    roles = _roles(manifest)
    images = dict(manifest.get("images") or {})
    volumes = dict(manifest.get("volumes") or {})
    return f"""# DGX Spark KAME realtime voice stack.
# This file intentionally contains no API keys or bearer tokens.
HERMES_DGX_SPARK_VLLM_IMAGE={images.get("vllm", DEFAULT_VLLM_IMAGE)}
HERMES_DGX_SPARK_HERMES_IMAGE={images.get("hermes", DEFAULT_HERMES_IMAGE)}
HERMES_DGX_SPARK_MODEL_CACHE={volumes.get("model_cache_dir", "${HOME}/.cache/huggingface")}
HERMES_REPO_DIR={manifest["repo_dir"]}
HERMES_HOME={manifest["hermes_home"]}

HERMES_KAME_INTERFACE_MODEL={roles["interface"]["model"]}
HERMES_KAME_INTERFACE_BASE_URL={roles["interface"]["base_url"]}
HERMES_KAME_INTERFACE_AUDIO_INPUT=native_audio
HERMES_KAME_ASR_MODE={manifest["engine"]["asr_mode"]}
HERMES_KAME_MAX_SPOKEN_SENTENCES={manifest["engine"]["max_spoken_sentences"]}

HERMES_KAME_ORACLE_MODEL={roles["oracle"]["preferred_local_model"]}
HERMES_KAME_ORACLE_BASE_URL={roles["oracle"]["base_url"]}
HERMES_VOICE_STREAMING_STT_BASE_URL={roles["asr"]["base_url"]}
HERMES_VOICE_STREAMING_TTS_BASE_URL={roles["tts"]["base_url"]}
"""


def render_dgx_spark_launch_script(manifest: Mapping[str, Any]) -> str:
    return f"""#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"
docker compose --env-file .env.example -f compose.yaml up --remove-orphans "$@"

# Readiness check once services are up:
#   python -m hermes_cli.realtime_voice_dgx_spark --output-dir "$SCRIPT_DIR" --check
#
# Hermes voice profile:
#   python -m hermes_cli.realtime_voice_profile --preset kame \\
#     --kame-reflex-model {manifest["roles"]["interface"]["model"]} \\
#     --kame-interface-audio-input native_audio \\
#     --kame-asr-mode {manifest["engine"]["asr_mode"]} \\
#     --kame-preferred-local-oracle-model {manifest["roles"]["oracle"]["preferred_local_model"]} \\
#     --streaming-stt-base-url {manifest["roles"]["asr"]["base_url"]} \\
#     --streaming-tts-base-url {manifest["roles"]["tts"]["base_url"]}
"""


def build_dgx_spark_benchmark_matrix(manifest: Mapping[str, Any]) -> dict[str, Any]:
    roles = _roles(manifest)
    return {
        "kind": "kame_dgx_spark_benchmark_matrix",
        "version": 1,
        "candidates": {
            "interface": [
                {
                    "model": roles["interface"]["model"],
                    "input": "direct_audio",
                    "required_metrics": [
                        "speech_end_to_interface_decision_ms",
                        "speech_end_to_local_first_audio_ms",
                        "routing_accuracy",
                    ],
                },
                {
                    "model": roles["interface"]["model"],
                    "input": "stt_fallback",
                    "required_metrics": [
                        "speech_end_to_transcript_ms",
                        "transcript_to_interface_decision_ms",
                        "routing_accuracy",
                    ],
                },
            ],
            "oracle": [
                {
                    "model": roles["oracle"]["preferred_local_model"],
                    "required_metrics": [
                        "oracle_request_to_accepted_ms",
                        "oracle_accepted_to_first_token_ms",
                        "oracle_first_token_to_first_audio_ms",
                    ],
                }
            ],
            "speech": [
                {
                    "role": "oracle_verbatim_asr",
                    "mode": roles["asr"]["mode"],
                    "required_metrics": [
                        "speech_end_to_asr_final_ms",
                        "literal_accuracy_names_numbers_code",
                    ],
                },
                {
                    "role": "tts",
                    "required_metrics": [
                        "tts_request_to_first_audio_ms",
                        "tts_request_to_audio_end_ms",
                    ],
                },
            ],
        },
        "acceptance_targets_ms": manifest["quality_targets_ms"],
    }


def load_dgx_spark_benchmark_evidence(path: str | Path) -> list[dict[str, Any]]:
    evidence_path = Path(path).expanduser()
    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("DGX Spark KAME benchmark evidence must be a JSON array")
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, Mapping):
            raise ValueError(f"DGX Spark KAME benchmark evidence entry {index} must be an object")
        entries.append(dict(entry))
    return entries


def validate_dgx_spark_benchmark_evidence(
    matrix: Mapping[str, Any],
    entries: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate headless DGX Spark KAME benchmark/evidence results.

    Expected evidence entries are intentionally simple JSON objects:
    - ``kind=kame_benchmark_result`` with ``category`` (interface/oracle/speech),
      optional ``input`` or ``role``, and a ``metrics`` object.
    - ``kind=kame_smoke_result`` with ``name`` (all_local_smoke/cloud_fallback_smoke)
      and ``ok=true``.
    """

    issues: list[str] = []
    candidates = matrix.get("candidates") if isinstance(matrix.get("candidates"), Mapping) else {}
    if not isinstance(candidates, Mapping):
        return {"ok": False, "issues": ["matrix: missing candidates mapping"], "coverage": {}}

    coverage: dict[str, bool] = {}
    interface_candidates = candidates.get("interface") if isinstance(candidates.get("interface"), list) else []
    for candidate in interface_candidates:
        if not isinstance(candidate, Mapping):
            continue
        input_mode = str(candidate.get("input") or "").strip()
        label = f"interface:{input_mode}"
        match = _find_benchmark_entry(entries, category="interface", input_mode=input_mode)
        coverage[label] = match is not None
        if match is None:
            issues.append(f"{label}: missing benchmark result")
            continue
        issues.extend(_missing_metric_issues(label, match, candidate.get("required_metrics")))

    has_direct = coverage.get("interface:direct_audio") is True
    has_fallback = coverage.get("interface:stt_fallback") is True
    coverage["interface_direct_audio_vs_stt_fallback"] = has_direct and has_fallback
    if not has_direct or not has_fallback:
        issues.append("interface_direct_audio_vs_stt_fallback: requires direct_audio and stt_fallback results")

    oracle_candidates = candidates.get("oracle") if isinstance(candidates.get("oracle"), list) else []
    for candidate in oracle_candidates:
        if not isinstance(candidate, Mapping):
            continue
        label = "oracle:local"
        match = _find_benchmark_entry(entries, category="oracle")
        coverage[label] = match is not None
        if match is None:
            issues.append(f"{label}: missing benchmark result")
            continue
        issues.extend(_missing_metric_issues(label, match, candidate.get("required_metrics")))

    speech_candidates = candidates.get("speech") if isinstance(candidates.get("speech"), list) else []
    for candidate in speech_candidates:
        if not isinstance(candidate, Mapping):
            continue
        role = str(candidate.get("role") or "").strip()
        label = f"speech:{role}"
        match = _find_benchmark_entry(entries, category="speech", role=role)
        coverage[label] = match is not None
        if match is None:
            issues.append(f"{label}: missing benchmark result")
            continue
        issues.extend(_missing_metric_issues(label, match, candidate.get("required_metrics")))

    coverage["oracle_verbatim_asr_latency_and_literal_accuracy"] = coverage.get("speech:oracle_verbatim_asr") is True
    coverage["local_asr_tts_benchmark_matrix"] = (
        coverage.get("speech:oracle_verbatim_asr") is True and coverage.get("speech:tts") is True
    )
    for smoke_name in ("all_local_smoke", "cloud_fallback_smoke"):
        ok = _has_passing_smoke(entries, smoke_name)
        coverage[smoke_name] = ok
        if not ok:
            issues.append(f"{smoke_name}: missing passing smoke result")

    return {
        "ok": not issues,
        "issues": issues,
        "coverage": coverage,
    }


def _find_benchmark_entry(
    entries: list[Mapping[str, Any]],
    *,
    category: str,
    input_mode: str = "",
    role: str = "",
) -> Mapping[str, Any] | None:
    for entry in entries:
        if str(entry.get("kind") or "") != "kame_benchmark_result":
            continue
        if str(entry.get("category") or "") != category:
            continue
        if input_mode and str(entry.get("input") or "") != input_mode:
            continue
        if role and str(entry.get("role") or "") != role:
            continue
        return entry
    return None


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


def _valid_metric_value(metric_name: str, value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    if not parsed >= 0:
        return False
    if "accuracy" in metric_name:
        return parsed <= 1.0
    return True


def _has_passing_smoke(entries: list[Mapping[str, Any]], name: str) -> bool:
    for entry in entries:
        if str(entry.get("kind") or "") != "kame_smoke_result":
            continue
        if str(entry.get("name") or "") == name and entry.get("ok") is True:
            return True
    return False


def preflight_dgx_spark_stack(
    manifest: Mapping[str, Any],
    *,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    roles = _roles(manifest)
    checks = {
        "interface_models": probe_json_endpoint(
            roles["interface"]["models_url"],
            timeout_seconds=timeout_seconds,
            expected_model=roles["interface"]["model"],
        ),
        "oracle_models": probe_json_endpoint(
            roles["oracle"]["models_url"],
            timeout_seconds=timeout_seconds,
            expected_model=roles["oracle"]["preferred_local_model"],
        ),
        "sidecar_health": probe_json_endpoint(
            roles["sidecar"]["health_url"],
            timeout_seconds=timeout_seconds,
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


def probe_json_endpoint(
    url: str,
    *,
    timeout_seconds: float,
    expected_model: str | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
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
    return {
        "ok": 200 <= int(status) < 300 and model_ok,
        "url": url,
        "status": status,
        "expected_model": expected_model,
        "model_found": model_ok if expected_model else None,
    }


def _openai_models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _health_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/health"


def _models_payload_contains(payload: Mapping[str, Any], expected_model: str) -> bool:
    data = payload.get("data")
    if not isinstance(data, list):
        return False
    for item in data:
        if isinstance(item, Mapping) and item.get("id") == expected_model:
            return True
    return False


def _roles(manifest: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("manifest has no roles mapping")
    return roles  # type: ignore[return-value]


def _json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
