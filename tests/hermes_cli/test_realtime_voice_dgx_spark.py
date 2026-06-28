import json
import argparse
from pathlib import Path

from hermes_cli import realtime_voice_dgx_spark
from hermes_cli.subcommands.voice import build_voice_parser


PRODUCTION_ASR_MODULE = "hermes_cli.realtime_voice_nemotron_speech_bridge"
PRODUCTION_TTS_MODULE = "hermes_cli.realtime_voice_magpie_tts_bridge"
PRODUCTION_ASR_MODEL = "nemotron-speech-streaming-0.6b"
PRODUCTION_TTS_MODEL = "magpie-local-streaming-tts"
PRODUCTION_ASR_ADAPTER = "nemotron_speech_streaming"
PRODUCTION_TTS_ADAPTER = "magpie_streaming_tts"


def _manifest(tmp_path: Path, *, production_speech: bool = False) -> dict:
    speech_kwargs = (
        {
            "asr_module": PRODUCTION_ASR_MODULE,
            "tts_module": PRODUCTION_TTS_MODULE,
            "asr_model": PRODUCTION_ASR_MODEL,
            "tts_model": PRODUCTION_TTS_MODEL,
            "asr_adapter": PRODUCTION_ASR_ADAPTER,
            "tts_adapter": PRODUCTION_TTS_ADAPTER,
        }
        if production_speech
        else {}
    )
    return realtime_voice_dgx_spark.build_dgx_spark_stack_manifest(
        repo_dir=tmp_path / "repo",
        hermes_home=tmp_path / "home",
        interface_base_url="http://spark.local:8000/v1",
        interface_model="gemma-4-E2B-it",
        interface_candidate_models=None,
        interface_context_tokens=8192,
        interface_gpu_memory_utilization=0.18,
        interface_max_audio_seconds=30.0,
        oracle_base_url="http://spark.local:8001/v1",
        oracle_model="gemma-4-26B-A4B-it",
        oracle_context_tokens=32768,
        oracle_gpu_memory_utilization=0.62,
        sidecar_base_url="http://spark.local:8765",
        asr_base_url="http://spark.local:8767",
        tts_base_url="http://spark.local:8768",
        asr_mode="on_escalation",
        vllm_image="vllm/vllm-openai:gemma4-cu130",
        hermes_image="ghcr.io/astral-sh/uv:python3.12-bookworm-slim",
        model_cache_dir="/models",
        **speech_kwargs,
    )


def _passing_benchmark_evidence() -> list[dict]:
    interface_entries = []
    for model in ("gemma-4-E2B-it", "gemma-4-E4B-it"):
        interface_entries.extend(
            [
                {
                    "kind": "kame_benchmark_result",
                    "category": "interface",
                    "model": model,
                    "input": "direct_audio",
                    "metrics": {
                        "speech_end_to_interface_decision_ms": 320,
                        "speech_end_to_local_first_audio_ms": 480,
                        "routing_accuracy": 0.94,
                        "capability_honesty_rate": 0.99,
                        "local_route_precision": 0.93,
                        "oracle_route_recall": 0.96,
                    },
                },
                {
                    "kind": "kame_benchmark_result",
                    "category": "interface",
                    "model": model,
                    "input": "stt_fallback",
                    "metrics": {
                        "speech_end_to_transcript_ms": 190,
                        "transcript_to_interface_decision_ms": 280,
                        "routing_accuracy": 0.91,
                        "capability_honesty_rate": 0.98,
                        "local_route_precision": 0.9,
                        "oracle_route_recall": 0.94,
                    },
                },
            ]
        )
    assumption_entries = [
        {
            "kind": "kame_model_assumption_result",
            "name": "interface_audio_input_supported",
            "validated_by": "interface_audio_probe",
            "model": "gemma-4-E2B-it",
            "ok": True,
        },
        {
            "kind": "kame_model_assumption_result",
            "name": "interface_audio_is_segment_buffered",
            "validated_by": "vad_endpoint_then_interface_audio_probe",
            "ok": True,
        },
        {
            "kind": "kame_model_assumption_result",
            "name": "interface_audio_limit_seconds",
            "validated_by": "manifest_and_vllm_limit_mm_per_prompt",
            "ok": True,
        },
        {
            "kind": "kame_model_assumption_result",
            "name": "vllm_multimodal_audio_prompt_limit",
            "validated_by": "compose_vllm_args",
            "ok": True,
        },
        {
            "kind": "kame_model_assumption_result",
            "name": "oracle_authority",
            "validated_by": "oracle_models_probe",
            "model": "gemma-4-26B-A4B-it",
            "ok": True,
        },
    ]
    return [
        *interface_entries,
        *assumption_entries,
        {
            "kind": "kame_benchmark_result",
            "category": "oracle",
            "metrics": {
                "oracle_request_to_accepted_ms": 40,
                "oracle_accepted_to_first_token_ms": 780,
                "oracle_first_token_to_first_tts_audio_ms": 180,
                "first_tts_audio_to_playback_start_ms": 40,
            },
        },
        {
            "kind": "kame_benchmark_result",
            "category": "oracle_outcome",
            "asr_hypothesis": "without_asr_hypothesis",
            "metrics": {
                "task_success_rate": 0.78,
                "literal_argument_accuracy": 0.72,
                "tool_argument_error_rate": 0.21,
            },
        },
        {
            "kind": "kame_benchmark_result",
            "category": "oracle_outcome",
            "asr_hypothesis": "with_asr_hypothesis",
            "metrics": {
                "task_success_rate": 0.84,
                "literal_argument_accuracy": 0.9,
                "tool_argument_error_rate": 0.08,
            },
        },
        {
            "kind": "kame_benchmark_result",
            "category": "speech",
            "role": "oracle_verbatim_asr",
            "model": PRODUCTION_ASR_MODEL,
            "adapter": PRODUCTION_ASR_ADAPTER,
            "protocol_smoke_only": False,
            "metrics": {
                "speech_end_to_asr_final_ms": 110,
                "literal_accuracy_names_numbers_code": 0.88,
            },
        },
        {
            "kind": "kame_benchmark_result",
            "category": "speech",
            "role": "tts",
            "model": PRODUCTION_TTS_MODEL,
            "adapter": PRODUCTION_TTS_ADAPTER,
            "protocol_smoke_only": False,
            "metrics": {
                "tts_request_to_first_audio_ms": 160,
                "tts_request_to_audio_end_ms": 620,
            },
        },
        {
            "kind": "kame_smoke_result",
            "name": "all_local_smoke",
            "ok": True,
            "local_turns": 2,
            "local_turn_oracle_calls": 0,
            "oracle_bound_turns": 4,
            "oracle_bound_oracle_calls": 4,
            "oracle_authority_routes": ["tools", "files", "memory", "project_context"],
            "interface_input_sources": ["native_audio"],
            "reflex_providers": ["vllm"],
        },
        {"kind": "kame_smoke_result", "name": "cloud_fallback_smoke", "ok": True},
        {"kind": "kame_smoke_result", "name": "capability_honesty_smoke", "ok": True},
        {"kind": "kame_smoke_result", "name": "barge_in_interruption_smoke", "ok": True},
    ]


def test_manifest_describes_full_kame_dgx_spark_stack(tmp_path):
    manifest = _manifest(tmp_path)

    assert manifest["kind"] == "kame_dgx_spark_stack"
    assert manifest["target"]["hardware"] == "1x DGX Spark"
    assert manifest["engine"]["name"] == "kame_interface_oracle"
    assert manifest["engine"]["interface_audio_input"] == "native_audio"
    assert manifest["engine"]["asr_mode"] == "on_escalation"
    assert manifest["engine"]["max_spoken_sentences"] == 2
    assert manifest["model_assumptions"]["interface_audio_input_supported"] == {
        "model": "gemma-4-E2B-it",
        "required": True,
        "validated_by": "interface_audio_probe",
        "description": "Interface/reflex model accepts a bounded audio prompt segment and returns text JSON.",
    }
    assert manifest["model_assumptions"]["interface_audio_is_segment_buffered"]["validated_by"] == (
        "vad_endpoint_then_interface_audio_probe"
    )
    assert manifest["model_assumptions"]["interface_audio_limit_seconds"] == {
        "seconds": 30.0,
        "required": True,
        "validated_by": "manifest_and_vllm_limit_mm_per_prompt",
    }
    assert manifest["model_assumptions"]["vllm_multimodal_audio_prompt_limit"]["limit_mm_per_prompt"] == {"audio": 1}
    assert manifest["model_assumptions"]["oracle_authority"]["model"] == "gemma-4-26B-A4B-it"
    assert manifest["roles"]["interface"]["model"] == "gemma-4-E2B-it"
    assert [entry["model"] for entry in manifest["roles"]["interface"]["candidate_models"]] == [
        "gemma-4-E2B-it",
        "gemma-4-E4B-it",
    ]
    assert manifest["roles"]["interface"]["candidate_models"][0]["priority"] == "default"
    assert manifest["roles"]["interface"]["candidate_models"][1]["priority"] == "comparison"
    assert manifest["roles"]["interface"]["limit_mm_per_prompt"] == {"audio": 1}
    assert manifest["roles"]["interface"]["max_audio_seconds"] == 30.0
    assert manifest["roles"]["oracle"]["preferred_local_model"] == "gemma-4-26B-A4B-it"
    assert manifest["roles"]["asr"]["role"] == "oracle_verbatim_evidence"
    assert manifest["roles"]["asr"]["adapter"] == "loopback_smoke_bridge"
    assert manifest["roles"]["asr"]["model"] == "oracle-verbatim-asr"
    assert manifest["roles"]["asr"]["module"] == "hermes_cli.realtime_voice_loopback_bridge"
    assert manifest["roles"]["asr"]["protocol_smoke_only"] is True
    assert manifest["roles"]["asr"]["production_replacement"] == "local_streaming_asr"
    assert manifest["roles"]["asr"]["feeds_reflex"] is False
    assert manifest["roles"]["tts"]["adapter"] == "loopback_smoke_bridge"
    assert manifest["roles"]["tts"]["model"] == "local-streaming-tts"
    assert manifest["roles"]["tts"]["module"] == "hermes_cli.realtime_voice_loopback_bridge"
    assert manifest["roles"]["tts"]["protocol_smoke_only"] is True
    assert manifest["roles"]["tts"]["production_replacement"] == "local_streaming_tts"
    assert "all_local_smoke" in manifest["evidence_required"]
    assert "oracle_simple_first_audio_latency" in manifest["evidence_required"]
    assert "capability_honesty_smoke" in manifest["evidence_required"]
    assert "barge_in_interruption_smoke" in manifest["evidence_required"]


def test_rendered_compose_has_reflex_oracle_and_sidecar_without_secret_material(tmp_path):
    compose = realtime_voice_dgx_spark.render_dgx_spark_compose(_manifest(tmp_path))

    assert "kame-interface-vllm:" in compose
    assert "kame-oracle-vllm:" in compose
    assert "hermes-realtime-sidecar:" in compose
    assert "      - kame-oracle-vllm" in compose
    assert "kame-asr-bridge:" in compose
    assert "kame-tts-bridge:" in compose
    assert "gemma-4-E2B-it" in compose
    assert "gemma-4-26B-A4B-it" in compose
    assert "--limit-mm-per-prompt" in compose
    assert '{"audio":1}' in compose
    assert "HERMES_VOICE_STREAMING_STT_BASE_URL" in compose
    assert "HERMES_KAME_INTERFACE_BASE_URL: http://kame-interface-vllm:8000/v1" in compose
    assert "HERMES_VOICE_VLLM_BASE_URL: http://kame-interface-vllm:8000/v1" in compose
    assert "HERMES_VOICE_STREAMING_STT_BASE_URL: http://kame-asr-bridge:8767" in compose
    assert "HERMES_VOICE_STREAMING_TTS_BASE_URL: http://kame-tts-bridge:8768" in compose
    assert "      - http://kame-interface-vllm:8000/v1" in compose
    assert "      - --interface-base-url" in compose
    assert "      - http://kame-asr-bridge:8767" in compose
    assert "      - http://kame-tts-bridge:8768" in compose
    assert "HERMES_VOICE_VLLM_BASE_URL: http://spark.local:8000/v1" not in compose
    assert "HERMES_KAME_INTERFACE_BASE_URL: http://spark.local:8000/v1" not in compose
    assert "HERMES_VOICE_STREAMING_STT_BASE_URL: http://spark.local:8767" not in compose
    assert "HERMES_VOICE_STREAMING_TTS_BASE_URL: http://spark.local:8768" not in compose
    assert "oracle-verbatim-asr" in compose
    assert "local-streaming-tts" in compose
    assert "HERMES_DGX_SPARK_ASR_ADAPTER: loopback_smoke_bridge" in compose
    assert "HERMES_DGX_SPARK_TTS_ADAPTER: loopback_smoke_bridge" in compose
    assert "hermes_cli.realtime_voice_loopback_bridge" in compose
    assert "API_KEY" not in compose
    assert "sk_" not in compose


def test_rendered_compose_wires_production_speech_bridge_upstreams(tmp_path):
    manifest = _manifest(tmp_path, production_speech=True)
    compose = realtime_voice_dgx_spark.render_dgx_spark_compose(manifest)
    env_example = realtime_voice_dgx_spark.render_dgx_spark_env_example(manifest)

    assert PRODUCTION_ASR_MODULE in compose
    assert PRODUCTION_TTS_MODULE in compose
    assert "HERMES_VOICE_STREAMING_STT_MODEL: nemotron-speech-streaming-0.6b" in compose
    assert "HERMES_VOICE_STREAMING_TTS_MODEL: magpie-local-streaming-tts" in compose
    assert "HERMES_NEMOTRON_SPEECH_UPSTREAM_BASE_URL: ${HERMES_NEMOTRON_SPEECH_UPSTREAM_BASE_URL:-}" in compose
    assert "HERMES_NEMOTRON_SPEECH_UPSTREAM_TOKEN: ${HERMES_NEMOTRON_SPEECH_UPSTREAM_TOKEN:-}" in compose
    assert "HERMES_MAGPIE_TTS_UPSTREAM_BASE_URL: ${HERMES_MAGPIE_TTS_UPSTREAM_BASE_URL:-}" in compose
    assert "HERMES_MAGPIE_TTS_UPSTREAM_TOKEN: ${HERMES_MAGPIE_TTS_UPSTREAM_TOKEN:-}" in compose
    assert "      - ${HERMES_VOICE_STREAMING_STT_MODEL:-nemotron-speech-streaming-0.6b}" in compose
    assert "      - ${HERMES_NEMOTRON_SPEECH_UPSTREAM_BASE_URL:-}" in compose
    assert "      - ${HERMES_VOICE_STREAMING_TTS_MODEL:-magpie-local-streaming-tts}" in compose
    assert "      - ${HERMES_MAGPIE_TTS_UPSTREAM_BASE_URL:-}" in compose
    assert "HERMES_NEMOTRON_SPEECH_UPSTREAM_BASE_URL=" in env_example
    assert "HERMES_MAGPIE_TTS_UPSTREAM_BASE_URL=" in env_example
    assert "sk_" not in compose
    assert "sk_" not in env_example


def test_manifest_clamps_interface_max_audio_seconds(tmp_path):
    high = realtime_voice_dgx_spark.build_dgx_spark_stack_manifest(
        repo_dir=tmp_path / "repo",
        hermes_home=tmp_path / "home",
        interface_base_url="http://spark.local:8000/v1",
        interface_model="gemma-4-E2B-it",
        interface_candidate_models=None,
        interface_context_tokens=8192,
        interface_gpu_memory_utilization=0.18,
        interface_max_audio_seconds=99.0,
        oracle_base_url="http://spark.local:8001/v1",
        oracle_model="gemma-4-26B-A4B-it",
        oracle_context_tokens=32768,
        oracle_gpu_memory_utilization=0.62,
        sidecar_base_url="http://spark.local:8765",
        asr_base_url="http://spark.local:8767",
        tts_base_url="http://spark.local:8768",
        asr_mode="on_escalation",
        vllm_image="vllm/vllm-openai:gemma4-cu130",
        hermes_image="ghcr.io/astral-sh/uv:python3.12-bookworm-slim",
        model_cache_dir="/models",
    )
    low = realtime_voice_dgx_spark.build_dgx_spark_stack_manifest(
        repo_dir=tmp_path / "repo",
        hermes_home=tmp_path / "home",
        interface_base_url="http://spark.local:8000/v1",
        interface_model="gemma-4-E2B-it",
        interface_candidate_models=None,
        interface_context_tokens=8192,
        interface_gpu_memory_utilization=0.18,
        interface_max_audio_seconds=0.25,
        oracle_base_url="http://spark.local:8001/v1",
        oracle_model="gemma-4-26B-A4B-it",
        oracle_context_tokens=32768,
        oracle_gpu_memory_utilization=0.62,
        sidecar_base_url="http://spark.local:8765",
        asr_base_url="http://spark.local:8767",
        tts_base_url="http://spark.local:8768",
        asr_mode="on_escalation",
        vllm_image="vllm/vllm-openai:gemma4-cu130",
        hermes_image="ghcr.io/astral-sh/uv:python3.12-bookworm-slim",
        model_cache_dir="/models",
    )

    assert high["roles"]["interface"]["max_audio_seconds"] == 30.0
    assert low["roles"]["interface"]["max_audio_seconds"] == 1.0


def test_writer_emits_headless_artifact_pack(tmp_path):
    output_dir = tmp_path / "out"
    written = realtime_voice_dgx_spark.write_dgx_spark_stack_artifacts(
        output_dir,
        _manifest(tmp_path),
    )

    assert set(written) == {
        "manifest",
        "compose",
        "env_example",
        "launch",
        "preflight_script",
        "benchmark_validation",
        "benchmark_matrix",
        "benchmark_evidence_template",
    }
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "compose.yaml").is_file()
    assert (output_dir / ".env.example").is_file()
    assert (output_dir / "launch-local-stack.sh").is_file()
    assert (output_dir / "preflight-local-stack.sh").is_file()
    assert (output_dir / "validate-benchmark-evidence.sh").is_file()
    assert (output_dir / "benchmark-matrix.json").is_file()
    assert (output_dir / "benchmark-evidence-template.json").is_file()
    assert (output_dir / "launch-local-stack.sh").stat().st_mode & 0o111
    assert (output_dir / "preflight-local-stack.sh").stat().st_mode & 0o111
    assert (output_dir / "validate-benchmark-evidence.sh").stat().st_mode & 0o111

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    compose = (output_dir / "compose.yaml").read_text(encoding="utf-8")
    env_example = (output_dir / ".env.example").read_text(encoding="utf-8")
    launch = (output_dir / "launch-local-stack.sh").read_text(encoding="utf-8")
    preflight = (output_dir / "preflight-local-stack.sh").read_text(encoding="utf-8")
    validate_benchmark = (output_dir / "validate-benchmark-evidence.sh").read_text(encoding="utf-8")
    matrix = json.loads((output_dir / "benchmark-matrix.json").read_text(encoding="utf-8"))
    evidence_template = json.loads((output_dir / "benchmark-evidence-template.json").read_text(encoding="utf-8"))
    assert manifest["roles"]["interface"]["audio_input"] == "native_audio"
    assert manifest["roles"]["interface"]["max_audio_seconds"] == 30.0
    assert manifest["engine"]["max_spoken_sentences"] == 2
    assert "HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS=30.0" in env_example
    assert "HERMES_KAME_MAX_SPOKEN_SENTENCES=2" in env_example
    assert "HERMES_DGX_SPARK_ASR_ADAPTER=loopback_smoke_bridge" in env_example
    assert "HERMES_DGX_SPARK_TTS_ADAPTER=loopback_smoke_bridge" in env_example
    assert "HERMES_VOICE_VLLM_MODEL: ${HERMES_KAME_INTERFACE_MODEL:-gemma-4-E2B-it}" in compose
    assert "- ${HERMES_KAME_INTERFACE_MODEL:-gemma-4-E2B-it}" in compose
    assert "HERMES_DGX_SPARK_APPLY_PROFILE" in launch
    assert "hermes_cli.realtime_voice_profile --preset kame --apply" in launch
    assert ': "${HERMES_KAME_INTERFACE_MODEL:=gemma-4-E2B-it}"' in launch
    assert ': "${HERMES_KAME_ORACLE_MODEL:=gemma-4-26B-A4B-it}"' in launch
    assert '--kame-reflex-model "$HERMES_KAME_INTERFACE_MODEL"' in launch
    assert "--kame-interface-audio-input native_audio" in launch
    assert '--kame-interface-base-url "$HERMES_KAME_INTERFACE_BASE_URL"' in launch
    assert '--kame-interface-max-audio-seconds "$HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS"' in launch
    assert '--kame-asr-mode "$HERMES_KAME_ASR_MODE"' in launch
    assert '--kame-preferred-local-oracle-model "$HERMES_KAME_ORACLE_MODEL"' in launch
    assert '--kame-oracle-base-url "$HERMES_KAME_ORACLE_BASE_URL"' in launch
    assert '--kame-oracle-provider-name "KAME Local Oracle"' in launch
    assert '--streaming-stt-model "$HERMES_VOICE_STREAMING_STT_MODEL"' in launch
    assert '--streaming-tts-model "$HERMES_VOICE_STREAMING_TTS_MODEL"' in launch
    assert "--sidecar-host spark.local" in launch
    assert "--sidecar-port 8765" in launch
    assert "docker compose --env-file .env.example -f compose.yaml up" in launch
    assert "--check" in preflight
    assert "--output-dir \"$SCRIPT_DIR\"" in preflight
    assert "--interface-model \"$HERMES_KAME_INTERFACE_MODEL\"" in preflight
    assert "--interface-max-audio-seconds \"$HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS\"" in preflight
    assert "--oracle-model \"$HERMES_KAME_ORACLE_MODEL\"" in preflight
    assert "--asr-module \"$HERMES_DGX_SPARK_ASR_MODULE\"" in preflight
    assert "--asr-adapter \"$HERMES_DGX_SPARK_ASR_ADAPTER\"" in preflight
    assert "--tts-module \"$HERMES_DGX_SPARK_TTS_MODULE\"" in preflight
    assert "--tts-adapter \"$HERMES_DGX_SPARK_TTS_ADAPTER\"" in preflight
    assert "--sidecar-base-url http://spark.local:8765" in preflight
    assert "usage: $0 /path/to/benchmark-evidence.json" in validate_benchmark
    assert "--benchmark-evidence \"$1\"" in validate_benchmark
    assert "--interface-model \"$HERMES_KAME_INTERFACE_MODEL\"" in validate_benchmark
    assert "--interface-max-audio-seconds \"$HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS\"" in validate_benchmark
    assert "--oracle-model \"$HERMES_KAME_ORACLE_MODEL\"" in validate_benchmark
    assert "--asr-model \"$HERMES_VOICE_STREAMING_STT_MODEL\"" in validate_benchmark
    assert "--tts-model \"$HERMES_VOICE_STREAMING_TTS_MODEL\"" in validate_benchmark
    assert [
        (candidate["model"], candidate["input"]) for candidate in matrix["candidates"]["interface"]
    ] == [
        ("gemma-4-E2B-it", "direct_audio"),
        ("gemma-4-E2B-it", "stt_fallback"),
        ("gemma-4-E4B-it", "direct_audio"),
        ("gemma-4-E4B-it", "stt_fallback"),
    ]
    assert matrix["model_assumptions"]["interface_audio_input_supported"]["validated_by"] == "interface_audio_probe"
    assert matrix["candidates"]["oracle_outcome"][0]["asr_hypothesis"] == "without_asr_hypothesis"
    assert matrix["candidates"]["oracle_outcome"][1]["asr_hypothesis"] == "with_asr_hypothesis"
    assert evidence_template[0]["kind"] == "kame_benchmark_result"
    assert evidence_template[0]["category"] == "interface"
    assert evidence_template[0]["model"] == "gemma-4-E2B-it"
    assert evidence_template[0]["input"] == "direct_audio"
    assert evidence_template[0]["metrics"]["speech_end_to_interface_decision_ms"] is None
    smoke_entries = [entry for entry in evidence_template if entry.get("kind") == "kame_smoke_result"]
    assert [entry["name"] for entry in smoke_entries] == [
        "all_local_smoke",
        "cloud_fallback_smoke",
        "capability_honesty_smoke",
        "barge_in_interruption_smoke",
    ]
    assert all(entry["ok"] is False for entry in smoke_entries)
    assert smoke_entries[0]["local_turns"] is None
    assert smoke_entries[0]["local_turn_oracle_calls"] is None
    assert smoke_entries[0]["oracle_bound_turns"] is None
    assert smoke_entries[0]["oracle_bound_oracle_calls"] is None
    assert smoke_entries[0]["oracle_authority_routes"] == []
    assert smoke_entries[0]["interface_input_sources"] == []
    assert smoke_entries[0]["reflex_providers"] == []
    assumption_entries = [
        entry for entry in evidence_template if entry.get("kind") == "kame_model_assumption_result"
    ]
    assert [
        (entry["name"], entry["validated_by"], entry["ok"])
        for entry in assumption_entries
    ] == [
        ("interface_audio_input_supported", "interface_audio_probe", False),
        ("interface_audio_is_segment_buffered", "vad_endpoint_then_interface_audio_probe", False),
        ("interface_audio_limit_seconds", "manifest_and_vllm_limit_mm_per_prompt", False),
        ("vllm_multimodal_audio_prompt_limit", "compose_vllm_args", False),
        ("oracle_authority", "oracle_models_probe", False),
    ]


def test_benchmark_evidence_template_matches_matrix_and_does_not_pass_validation(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path))
    template = realtime_voice_dgx_spark.build_dgx_spark_benchmark_evidence_template(matrix)

    assert [
        (entry["model"], entry["input"]) for entry in template if entry.get("category") == "interface"
    ] == [
        ("gemma-4-E2B-it", "direct_audio"),
        ("gemma-4-E2B-it", "stt_fallback"),
        ("gemma-4-E4B-it", "direct_audio"),
        ("gemma-4-E4B-it", "stt_fallback"),
    ]
    assert [entry["asr_hypothesis"] for entry in template if entry.get("category") == "oracle_outcome"] == [
        "without_asr_hypothesis",
        "with_asr_hypothesis",
    ]
    assert {entry["role"] for entry in template if entry.get("category") == "speech"} == {
        "oracle_verbatim_asr",
        "tts",
    }
    assert {
        (entry["role"], entry["model"], entry["adapter"], entry["protocol_smoke_only"])
        for entry in template
        if entry.get("category") == "speech"
    } == {
        ("oracle_verbatim_asr", "oracle-verbatim-asr", "loopback_smoke_bridge", True),
        ("tts", "local-streaming-tts", "loopback_smoke_bridge", True),
    }

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, template)

    assert result["ok"] is False
    assert (
        "interface:gemma-4-E2B-it:direct_audio: "
        "missing or invalid metric speech_end_to_interface_decision_ms"
    ) in result["issues"]
    assert "all_local_smoke: missing passing smoke result" in result["issues"]
    assert (
        "local_asr_tts_benchmark_matrix: "
        "requires benchmark evidence for non-loopback local ASR and TTS adapters"
    ) in result["issues"]


def test_preflight_checks_openai_models_and_health_urls(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)
    seen_urls: list[str] = []

    class _Response:
        status = 200

        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        seen_urls.append(request.full_url)
        if request.full_url.endswith("/models") and ":8000" in request.full_url:
            return _Response({"data": [{"id": "gemma-4-E2B-it"}]})
        if request.full_url.endswith("/chat/completions") and ":8000" in request.full_url:
            body = json.loads(request.data.decode("utf-8"))
            assert body["model"] == "gemma-4-E2B-it"
            assert body["messages"][0]["content"][0]["type"] == "audio_url"
            assert body["messages"][0]["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,")
            return _Response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "route": "reject_or_clarify",
                                        "intent": "preflight audio probe",
                                        "text": "",
                                        "route_confidence": 1.0,
                                        "local_reply": "I did not catch speech.",
                                    }
                                )
                            }
                        }
                    ]
                }
            )
        if request.full_url.endswith("/models") and ":8001" in request.full_url:
            return _Response({"data": [{"id": "gemma-4-26B-A4B-it"}]})
        if request.full_url.endswith("/health") and ":8765" in request.full_url:
            return _Response(
                {
                    "ok": True,
                    "capabilities": {
                        "vllm_audio_frontend": True,
                        "tts": True,
                        "streaming_stt_bridge": True,
                        "streaming_tts_bridge": True,
                    },
                    "frontend": {
                        "streaming_stt_bridge": {"healthy": True},
                        "streaming_tts_bridge": {"healthy": True},
                    },
                }
            )
        return _Response({"ok": True})

    monkeypatch.setattr(realtime_voice_dgx_spark.urllib.request, "urlopen", fake_urlopen)

    preflight = realtime_voice_dgx_spark.preflight_dgx_spark_stack(
        manifest,
        timeout_seconds=0.1,
    )

    assert preflight["ok"] is True
    assert "http://spark.local:8000/v1/models" in seen_urls
    assert "http://spark.local:8000/v1/chat/completions" in seen_urls
    assert "http://spark.local:8001/v1/models" in seen_urls
    assert "http://spark.local:8765/health" in seen_urls
    assert "http://spark.local:8767/health" in seen_urls
    assert "http://spark.local:8768/health" in seen_urls
    assert preflight["checks"]["interface_audio_probe"]["ok"] is True
    assert preflight["checks"]["interface_audio_probe"]["audio_prompt"] is True
    assert preflight["checks"]["sidecar_health"]["field_misses"] == []


def test_preflight_fails_when_sidecar_lacks_kame_reflex_capability(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)

    class _Response:
        status = 200

        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models") and ":8000" in request.full_url:
            return _Response({"data": [{"id": "gemma-4-E2B-it"}]})
        if request.full_url.endswith("/chat/completions") and ":8000" in request.full_url:
            return _Response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "route": "reject_or_clarify",
                                        "intent": "preflight audio probe",
                                        "text": "",
                                        "route_confidence": 1.0,
                                        "local_reply": "I did not catch speech.",
                                    }
                                )
                            }
                        }
                    ]
                }
            )
        if request.full_url.endswith("/models") and ":8001" in request.full_url:
            return _Response({"data": [{"id": "gemma-4-26B-A4B-it"}]})
        if request.full_url.endswith("/health") and ":8765" in request.full_url:
            return _Response(
                {
                    "ok": True,
                    "capabilities": {
                        "vllm_audio_frontend": False,
                        "tts": True,
                        "streaming_stt_bridge": True,
                        "streaming_tts_bridge": True,
                    },
                    "frontend": {
                        "streaming_stt_bridge": {"healthy": True},
                        "streaming_tts_bridge": {"healthy": True},
                    },
                }
            )
        return _Response({"ok": True})

    monkeypatch.setattr(realtime_voice_dgx_spark.urllib.request, "urlopen", fake_urlopen)

    preflight = realtime_voice_dgx_spark.preflight_dgx_spark_stack(
        manifest,
        timeout_seconds=0.1,
    )

    assert preflight["ok"] is False
    assert preflight["checks"]["sidecar_health"]["ok"] is False
    assert {
        "path": "capabilities.vllm_audio_frontend",
        "expected": True,
        "actual": False,
    } in preflight["checks"]["sidecar_health"]["field_misses"]


def test_preflight_fails_when_interface_audio_probe_is_not_kame_json(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)

    class _Response:
        status = 200

        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models") and ":8000" in request.full_url:
            return _Response({"data": [{"id": "gemma-4-E2B-it"}]})
        if request.full_url.endswith("/chat/completions") and ":8000" in request.full_url:
            return _Response({"choices": [{"message": {"content": "not json"}}]})
        if request.full_url.endswith("/models") and ":8001" in request.full_url:
            return _Response({"data": [{"id": "gemma-4-26B-A4B-it"}]})
        if request.full_url.endswith("/health") and ":8765" in request.full_url:
            return _Response(
                {
                    "ok": True,
                    "capabilities": {
                        "vllm_audio_frontend": True,
                        "tts": True,
                        "streaming_stt_bridge": True,
                        "streaming_tts_bridge": True,
                    },
                    "frontend": {
                        "streaming_stt_bridge": {"healthy": True},
                        "streaming_tts_bridge": {"healthy": True},
                    },
                }
            )
        return _Response({"ok": True})

    monkeypatch.setattr(realtime_voice_dgx_spark.urllib.request, "urlopen", fake_urlopen)

    preflight = realtime_voice_dgx_spark.preflight_dgx_spark_stack(
        manifest,
        timeout_seconds=0.1,
    )

    assert preflight["ok"] is False
    assert preflight["checks"]["interface_audio_probe"]["ok"] is False
    assert preflight["checks"]["interface_audio_probe"]["schema_issues"][0].startswith("message content is not JSON")


def test_preflight_fails_when_speech_bridge_health_payload_is_not_ok(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path, production_speech=True)

    class _Response:
        status = 200

        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models") and ":8000" in request.full_url:
            return _Response({"data": [{"id": "gemma-4-E2B-it"}]})
        if request.full_url.endswith("/chat/completions") and ":8000" in request.full_url:
            return _Response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "route": "reject_or_clarify",
                                        "intent": "preflight audio probe",
                                        "text": "",
                                        "route_confidence": 1.0,
                                        "local_reply": "I did not catch speech.",
                                    }
                                )
                            }
                        }
                    ]
                }
            )
        if request.full_url.endswith("/models") and ":8001" in request.full_url:
            return _Response({"data": [{"id": "gemma-4-26B-A4B-it"}]})
        if request.full_url.endswith("/health") and ":8765" in request.full_url:
            return _Response(
                {
                    "ok": True,
                    "capabilities": {
                        "vllm_audio_frontend": True,
                        "tts": True,
                        "streaming_stt_bridge": True,
                        "streaming_tts_bridge": True,
                    },
                    "frontend": {
                        "streaming_stt_bridge": {"healthy": True},
                        "streaming_tts_bridge": {"healthy": True},
                    },
                }
            )
        if request.full_url.endswith("/health") and ":8767" in request.full_url:
            return _Response(
                {
                    "ok": False,
                    "kind": "local_speech_proxy_bridge",
                    "capabilities": {"streaming_stt": False},
                    "frontend": {"provider": "nemotron_speech", "upstream_healthy": False},
                }
            )
        if request.full_url.endswith("/health") and ":8768" in request.full_url:
            return _Response(
                {
                    "ok": True,
                    "kind": "local_speech_proxy_bridge",
                    "capabilities": {"streaming_tts": True},
                    "frontend": {"provider": "magpie_tts", "upstream_healthy": True},
                }
            )
        return _Response({"ok": True})

    monkeypatch.setattr(realtime_voice_dgx_spark.urllib.request, "urlopen", fake_urlopen)

    preflight = realtime_voice_dgx_spark.preflight_dgx_spark_stack(
        manifest,
        timeout_seconds=0.1,
    )

    assert preflight["ok"] is False
    assert preflight["checks"]["asr_health"]["ok"] is False
    assert preflight["checks"]["asr_health"]["payload_ok"] is False
    assert preflight["checks"]["tts_health"]["ok"] is True
    assert preflight["checks"]["tts_health"]["payload_ok"] is True


def test_benchmark_evidence_validator_accepts_complete_comparison_matrix(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(
        matrix,
        _passing_benchmark_evidence(),
    )

    assert result["ok"] is True
    assert result["issues"] == []
    assert result["coverage"]["interface_direct_audio_latency"] is True
    assert result["coverage"]["oracle_simple_first_audio_latency"] is True
    assert result["coverage"]["interface_candidate_model_matrix"] is True
    assert result["coverage"]["interface_direct_audio_vs_stt_fallback"] is True
    assert result["coverage"]["oracle_outcomes_with_and_without_asr_hypotheses"] is True
    assert result["coverage"]["oracle_verbatim_asr_latency_and_literal_accuracy"] is True
    assert result["coverage"]["local_asr_tts_benchmark_matrix"] is True
    assert result["coverage"]["model_assumptions_validated"] is True
    assert result["coverage"]["model_assumption:interface_audio_input_supported"] is True
    assert result["coverage"]["model_assumption:interface_audio_is_segment_buffered"] is True
    assert result["coverage"]["model_assumption:interface_audio_limit_seconds"] is True
    assert result["coverage"]["model_assumption:vllm_multimodal_audio_prompt_limit"] is True
    assert result["coverage"]["model_assumption:oracle_authority"] is True
    assert result["coverage"]["all_local_smoke"] is True
    assert result["coverage"]["cloud_fallback_smoke"] is True
    assert result["coverage"]["capability_honesty_smoke"] is True
    assert result["coverage"]["barge_in_interruption_smoke"] is True


def test_benchmark_evidence_validator_requires_stt_fallback_and_smoke(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = [
        entry
        for entry in _passing_benchmark_evidence()
        if entry.get("input") != "stt_fallback" and entry.get("name") != "cloud_fallback_smoke"
    ]

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert "interface:gemma-4-E2B-it:stt_fallback: missing benchmark result" in result["issues"]
    assert "interface:gemma-4-E4B-it:stt_fallback: missing benchmark result" in result["issues"]
    assert (
        "interface_candidate_model_matrix: requires benchmark results for every interface model/input"
        in result["issues"]
    )
    assert (
        "interface_direct_audio_vs_stt_fallback: "
        "requires direct_audio and stt_fallback results for every interface model"
    ) in result["issues"]
    assert "cloud_fallback_smoke: missing passing smoke result" in result["issues"]


def test_benchmark_evidence_validator_requires_local_bypass_and_oracle_authority_smoke(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = _passing_benchmark_evidence()
    for entry in evidence:
        if entry.get("kind") == "kame_smoke_result" and entry.get("name") == "all_local_smoke":
            entry["local_turn_oracle_calls"] = 1
            entry["oracle_authority_routes"] = ["tools"]
            entry["interface_input_sources"] = ["streaming_stt"]
            entry["reflex_providers"] = ["local_stt"]

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["all_local_smoke"] is False
    assert "all_local_smoke: requires local_turn_oracle_calls == 0" in result["issues"]
    assert (
        "all_local_smoke: oracle_authority_routes missing files,memory,project_context"
        in result["issues"]
    )
    assert "all_local_smoke: interface_input_sources missing native_audio" in result["issues"]
    assert "all_local_smoke: reflex_providers missing vllm" in result["issues"]


def test_benchmark_evidence_validator_requires_model_assumption_results(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = [
        entry
        for entry in _passing_benchmark_evidence()
        if entry.get("kind") != "kame_model_assumption_result"
    ]

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["model_assumptions_validated"] is False
    assert result["coverage"]["model_assumption:interface_audio_input_supported"] is False
    assert (
        "model_assumption:interface_audio_input_supported: "
        "missing passing model assumption result validated_by=interface_audio_probe"
    ) in result["issues"]
    assert (
        "model_assumptions_validated: "
        "requires passing evidence for every required model/runtime assumption"
    ) in result["issues"]


def test_benchmark_evidence_validator_enforces_direct_audio_latency_targets(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = _passing_benchmark_evidence()
    for entry in evidence:
        if entry.get("category") == "interface" and entry.get("input") == "direct_audio":
            entry["metrics"]["speech_end_to_interface_decision_ms"] = 501

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["interface_direct_audio_latency"] is False
    assert (
        "interface:gemma-4-E2B-it:direct_audio: "
        "speech_end_to_interface_decision_ms 501 exceeds target 500"
    ) in result["issues"]
    assert (
        "interface_direct_audio_latency: "
        "requires direct_audio speech_end_to_interface_decision_ms and "
        "speech_end_to_local_first_audio_ms within configured targets"
    ) in result["issues"]


def test_benchmark_evidence_validator_enforces_oracle_latency_target(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = _passing_benchmark_evidence()
    for entry in evidence:
        if entry.get("category") == "oracle":
            entry["metrics"]["oracle_accepted_to_first_token_ms"] = 2800
            entry["metrics"]["oracle_first_token_to_first_tts_audio_ms"] = 250
            entry["metrics"]["first_tts_audio_to_playback_start_ms"] = 40

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["oracle_simple_first_audio_latency"] is False
    assert "oracle:local: oracle first audio total 3130 exceeds target 3000" in result["issues"]
    assert (
        "oracle_simple_first_audio_latency: "
        "requires oracle_request_to_accepted_ms, oracle_accepted_to_first_token_ms, "
        "oracle_first_token_to_first_tts_audio_ms, and first_tts_audio_to_playback_start_ms "
        "within configured targets"
    ) in result["issues"]


def test_benchmark_evidence_validator_requires_capability_and_interruption_smokes(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = [
        entry
        for entry in _passing_benchmark_evidence()
        if entry.get("name") not in {"capability_honesty_smoke", "barge_in_interruption_smoke"}
    ]

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["capability_honesty_smoke"] is False
    assert result["coverage"]["barge_in_interruption_smoke"] is False
    assert "capability_honesty_smoke: missing passing smoke result" in result["issues"]
    assert "barge_in_interruption_smoke: missing passing smoke result" in result["issues"]


def test_benchmark_evidence_validator_requires_every_interface_candidate(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = [
        entry
        for entry in _passing_benchmark_evidence()
        if entry.get("model") != "gemma-4-E4B-it"
    ]

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["interface_candidate_model_matrix"] is False
    assert "interface:gemma-4-E4B-it:direct_audio: missing benchmark result" in result["issues"]
    assert "interface:gemma-4-E4B-it:stt_fallback: missing benchmark result" in result["issues"]


def test_benchmark_evidence_validator_requires_oracle_asr_outcome_comparison(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = [
        entry
        for entry in _passing_benchmark_evidence()
        if entry.get("asr_hypothesis") != "with_asr_hypothesis"
    ]

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert "oracle_outcome:with_asr_hypothesis: missing benchmark result" in result["issues"]
    assert (
        "oracle_outcomes_with_and_without_asr_hypotheses: "
        "requires with_asr_hypothesis and without_asr_hypothesis results"
    ) in result["issues"]


def test_main_validates_benchmark_evidence_file(tmp_path, capsys):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(_passing_benchmark_evidence()), encoding="utf-8")

    exit_code = realtime_voice_dgx_spark.main(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--repo-dir",
            str(tmp_path / "repo"),
            "--hermes-home",
            str(tmp_path / "home"),
            "--asr-module",
            PRODUCTION_ASR_MODULE,
            "--asr-model",
            PRODUCTION_ASR_MODEL,
            "--asr-adapter",
            PRODUCTION_ASR_ADAPTER,
            "--tts-module",
            PRODUCTION_TTS_MODULE,
            "--tts-model",
            PRODUCTION_TTS_MODEL,
            "--tts-adapter",
            PRODUCTION_TTS_ADAPTER,
            "--benchmark-evidence",
            str(evidence_path),
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert result["ok"] is True
    assert result["benchmark_evidence"]["ok"] is True


def test_main_fails_on_incomplete_benchmark_evidence_file(tmp_path, capsys):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps([]), encoding="utf-8")

    exit_code = realtime_voice_dgx_spark.main(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--repo-dir",
            str(tmp_path / "repo"),
            "--hermes-home",
            str(tmp_path / "home"),
            "--benchmark-evidence",
            str(evidence_path),
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert result["ok"] is False
    assert result["benchmark_evidence"]["ok"] is False
    assert result["benchmark_evidence"]["issues"]


def test_main_writes_files_and_reports_json(tmp_path, capsys):
    exit_code = realtime_voice_dgx_spark.main(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--repo-dir",
            str(tmp_path / "repo"),
            "--hermes-home",
            str(tmp_path / "home"),
        ]
    )

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert Path(result["written"]["manifest"]).is_file()
    assert Path(result["written"]["compose"]).is_file()


def test_voice_subcommand_exposes_dgx_spark_launch_profile(tmp_path):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    calls = []

    def cmd_voice(args):
        calls.append(args)

    build_voice_parser(subparsers, cmd_voice=cmd_voice)

    args = parser.parse_args(
        [
            "voice",
            "dgx-spark",
            "--output-dir",
            str(tmp_path / "out"),
            "--interface-model",
            "gemma-4-E2B-it",
            "--interface-candidate-model",
            "gemma-4-E4B-it",
            "--oracle-model",
            "gemma-4-26B-A4B-it",
            "--check",
        ]
    )
    args.func(args)

    assert args.command == "voice"
    assert args.voice_command == "dgx-spark"
    assert args.output_dir == str(tmp_path / "out")
    assert args.interface_model == "gemma-4-E2B-it"
    assert args.interface_candidate_model == ["gemma-4-E4B-it"]
    assert args.oracle_model == "gemma-4-26B-A4B-it"
    assert args.check is True
    assert calls == [args]
