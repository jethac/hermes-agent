import argparse
import hashlib
import json
from pathlib import Path

from hermes_cli import realtime_voice_dgx_spark
from hermes_cli.subcommands.voice import build_voice_parser
from scripts.voiceops_spark_matrix import build_matrix


PRODUCTION_ASR_MODULE = "hermes_cli.realtime_voice_nemotron_speech_bridge"
PRODUCTION_TTS_MODULE = "hermes_cli.realtime_voice_magpie_tts_bridge"
PRODUCTION_ASR_MODEL = "nemotron-speech-streaming-0.6b"
PRODUCTION_TTS_MODEL = "magpie-local-streaming-tts"
PRODUCTION_ASR_ADAPTER = "nemotron_speech_streaming"
PRODUCTION_TTS_ADAPTER = "magpie_streaming_tts"
DEFAULT_SUPER_ORACLE_MODEL = realtime_voice_dgx_spark.DEFAULT_ORACLE_MODEL
VOICEOPS_PROJECTION_PROVENANCE = {
    "schema_version": "voiceops.spark_benchmark_evidence.v1",
    "hardware": "1x DGX Spark",
    "locality": "local_spark",
    "verified": True,
    "measured_at": "2026-06-29T00:00:00Z",
    "source_artifact": "raw-kame-benchmark.json",
}


def _collector_attestation(redacted_sha256: str) -> dict:
    return {
        "collector_name": "pytest_kame_projection_fixture",
        "collector_version": "test-v1",
        "run_id": "test-kame-projection-run",
        "command_argv": ["pytest", "tests/hermes_cli/test_realtime_voice_dgx_spark.py"],
        "git_commit": "a" * 40,
        "started_at": "2026-06-29T00:00:00Z",
        "finished_at": "2026-06-29T00:00:01Z",
        "raw_artifact_sha256": "b" * 64,
        "redacted_artifact_sha256": redacted_sha256,
        "parent_manifest_sha256": "c" * 64,
    }


def _manifest(tmp_path: Path, *, production_speech: bool = False) -> dict:
    speech_kwargs = (
        {
            "asr_module": PRODUCTION_ASR_MODULE,
            "tts_module": PRODUCTION_TTS_MODULE,
            "asr_model": PRODUCTION_ASR_MODEL,
            "tts_model": PRODUCTION_TTS_MODEL,
            "asr_provider": "nvidia_speech",
            "tts_provider": "nvidia_speech",
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
        oracle_model=DEFAULT_SUPER_ORACLE_MODEL,
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
                        "sample_count": 40,
                        "speech_end_to_interface_decision_ms": 320,
                        "speech_end_to_interface_decision_p50_ms": 320,
                        "speech_end_to_interface_decision_p90_ms": 430,
                        "kame_interface_model_request_ms": 240,
                        "speech_end_to_local_first_audio_ms": 480,
                        "speech_end_to_local_first_audio_p50_ms": 480,
                        "speech_end_to_local_first_audio_p90_ms": 780,
                        "routing_accuracy": 0.94,
                        "capability_honesty_rate": 0.99,
                        "local_route_precision": 0.93,
                        "oracle_route_recall": 0.96,
                        "steady_state_memory_gb": 18,
                    },
                },
                {
                    "kind": "kame_benchmark_result",
                    "category": "interface",
                    "model": model,
                    "input": "stt_fallback",
                    "metrics": {
                        "sample_count": 40,
                        "speech_end_to_transcript_ms": 190,
                        "speech_end_to_transcript_p50_ms": 190,
                        "speech_end_to_transcript_p90_ms": 260,
                        "transcript_to_interface_decision_ms": 280,
                        "transcript_to_interface_decision_p50_ms": 280,
                        "transcript_to_interface_decision_p90_ms": 390,
                        "routing_accuracy": 0.91,
                        "capability_honesty_rate": 0.98,
                        "local_route_precision": 0.9,
                        "oracle_route_recall": 0.94,
                        "steady_state_memory_gb": 18,
                    },
                },
            ]
        )
    reflex_entries = [
        {
            "kind": "kame_benchmark_result",
            "category": "reflex",
            "model": "Moshi S2S",
            "metrics": {
                "ack_latency_ms": 240,
                "barge_in_stop_ms": 80,
                "steady_state_memory_gb": 16,
            },
        }
    ]
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
            "name": "vllm_oracle_text_only_multimodal_limit",
            "validated_by": "compose_vllm_args",
            "ok": True,
        },
        {
            "kind": "kame_model_assumption_result",
            "name": "oracle_authority",
            "validated_by": "oracle_models_probe",
            "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
            "ok": True,
        },
    ]
    entries = [
        *reflex_entries,
        *interface_entries,
        *assumption_entries,
        {
            "kind": "kame_benchmark_result",
            "category": "oracle",
            "metrics": {
                "sample_count": 40,
                "oracle_request_to_accepted_ms": 40,
                "oracle_accepted_to_first_token_ms": 780,
                "oracle_first_token_to_first_tts_audio_ms": 180,
                "first_tts_audio_to_playback_start_ms": 40,
                "oracle_request_to_first_audio_p50_ms": 1040,
                "oracle_request_to_first_audio_p90_ms": 1900,
                "decode_tok_s": 24,
                "prefill_tok_s": 3100,
                "steady_state_memory_gb": 92,
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
                "sample_count": 40,
                "speech_end_to_asr_final_ms": 110,
                "speech_end_to_asr_final_p50_ms": 110,
                "speech_end_to_asr_final_p90_ms": 180,
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
                "sample_count": 40,
                "tts_request_to_first_audio_ms": 160,
                "tts_request_to_first_audio_p50_ms": 160,
                "tts_request_to_first_audio_p90_ms": 240,
                "tts_request_to_audio_end_ms": 620,
                "tts_request_to_audio_end_p50_ms": 620,
                "tts_request_to_audio_end_p90_ms": 820,
                "underrun_count": 0,
            },
        },
        {
            "kind": "kame_comparison_result",
            "name": "interface_direct_audio_vs_stt_fallback",
            "metrics": {
                "paired_turns": 40,
                "direct_audio_p50_decision_ms": 320,
                "stt_fallback_p50_decision_ms": 470,
                "direct_audio_routing_accuracy": 0.94,
                "stt_fallback_routing_accuracy": 0.91,
                "routing_agreement_rate": 0.9,
            },
        },
        {
            "kind": "kame_comparison_result",
            "name": "oracle_outcome_asr_hypothesis_delta",
            "metrics": {
                "paired_cases": 40,
                "with_asr_task_success_rate": 0.84,
                "without_asr_task_success_rate": 0.78,
                "with_asr_literal_argument_accuracy": 0.9,
                "without_asr_literal_argument_accuracy": 0.72,
                "with_asr_tool_argument_error_rate": 0.08,
                "without_asr_tool_argument_error_rate": 0.21,
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
            "oracle_selected_by": "Hermes /model",
            "components": {
                "reflex": True,
                "oracle": True,
                "interpreter": True,
                "asr": True,
                "tts": True,
                "sidecar": True,
            },
            "oracle_authority_routes": ["tools", "files", "memory", "project_context"],
            "interface_input_sources": ["native_audio"],
            "reflex_providers": ["vllm", "moshi"],
            "interpreter_providers": ["vllm", "gemma"],
            "auxiliary_transcript_sources": ["moshi_hypothesis", "classic_asr_fallback_optional"],
            "metrics": {
                "speech_end_to_first_audio_ms": 900,
                "barge_in_stop_ms": 90,
            },
        },
        {
            "kind": "kame_smoke_result",
            "name": "cloud_fallback_smoke",
            "ok": True,
            "fallback_trigger": "sidecar_unavailable",
            "fallback_mode": "legacy_voice",
            "fallback_reason_visible": True,
            "configured_policy_applied": True,
        },
        {
            "kind": "kame_smoke_result",
            "name": "capability_honesty_smoke",
            "ok": True,
            "voice_active": True,
            "voice_capability_checks": 3,
            "voice_denial_count": 0,
            "unsupported_voice_claims": 0,
        },
        {
            "kind": "kame_smoke_result",
            "name": "barge_in_interruption_smoke",
            "ok": True,
            "trigger_reason": "confirmed_user_speech",
            "playback_active": True,
            "stop_latency_ms": 95,
            "interrupted_response_committed": False,
        },
    ]
    return [{**VOICEOPS_PROJECTION_PROVENANCE, **entry} for entry in entries]


def _passing_benchmark_evidence_with_source(tmp_path: Path) -> list[dict]:
    evidence_path = tmp_path / "kame-evidence.json"
    raw_path = tmp_path / "raw-kame-benchmark.json"
    raw_path.write_text(
        json.dumps(
            {
                "redacted": True,
                "source": "synthetic KAME benchmark fixture",
                "source_keys": [
                    "kame_benchmark_result",
                    "kame_comparison_result",
                    "kame_model_assumption_result",
                    "kame_smoke_result",
                    "interface",
                    "oracle",
                    "speech",
                    "oracle_verbatim_asr",
                    "tts",
                    "all_local_smoke",
                    "cloud_fallback_smoke",
                    "capability_honesty_smoke",
                    "barge_in_interruption_smoke",
                    "reflex-moshi-s2s",
                    "interpreter-gemma4-e2b",
                    "interpreter-gemma4-e4b",
                    "oracle-nemotron3-super-local",
                    "asr-nemotron-speech",
                    "tts-magpie-local",
                    "voiceops_spark_stack_smoke",
                ],
                "kame_turns": [
                    {
                        "turn_id": "local-001",
                        "route": "local",
                        "oracle_called": False,
                        "audio_segment_ref": "artifact://redacted/local-001.wav",
                        "audio_time_range_ms": [100, 900],
                        "reflex_transcript_hypothesis": {
                            "authority": "hypothesis",
                            "source": "moshi",
                            "text": "[redacted local hypothesis]",
                        },
                        "auxiliary_transcript_hypotheses": [],
                    },
                    {
                        "turn_id": "oracle-001",
                        "route": "defer",
                        "oracle_called": True,
                        "oracle_calls": 1,
                        "audio_segment_ref": "artifact://redacted/oracle-001.wav",
                        "audio_time_range_ms": [1200, 3300],
                        "reflex_transcript_hypothesis": {
                            "authority": "hypothesis",
                            "source": "moshi",
                            "text": "[redacted reflex hypothesis]",
                        },
                        "auxiliary_transcript_hypotheses": [
                            {
                                "authority": "hypothesis",
                                "source": "classic_asr_fallback_optional",
                                "text": "[redacted auxiliary hypothesis]",
                            }
                        ],
                        "interpreter_evidence": {
                            "source": "gemma_interpreter",
                            "corrected_transcript": "[redacted interpreter correction]",
                            "confidence": 0.91,
                        },
                        "interpreter_corrected_transcript": "[redacted interpreter correction]",
                        "tool_critical_text_source": "gemma_interpreter",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    source_artifact_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    return [
        {
            **entry,
            "source_artifact_sha256": source_artifact_sha256,
            "collector_attestation": _collector_attestation(source_artifact_sha256),
            "_evidence_path": str(evidence_path),
        }
        for entry in _passing_benchmark_evidence()
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
    assert manifest["model_assumptions"]["vllm_multimodal_audio_prompt_limit"]["limit_mm_per_prompt"] == {
        "image": 0,
        "audio": 1,
    }
    assert manifest["model_assumptions"]["vllm_oracle_text_only_multimodal_limit"]["limit_mm_per_prompt"] == {
        "image": 0,
        "audio": 0,
    }
    assert manifest["model_assumptions"]["oracle_authority"]["model"] == DEFAULT_SUPER_ORACLE_MODEL
    assert manifest["roles"]["interface"]["provider"] == "gemma4"
    assert manifest["roles"]["interface"]["implementation"] == "openai_compatible_vllm"
    assert manifest["roles"]["interface"]["model"] == "gemma-4-E2B-it"
    assert [entry["model"] for entry in manifest["roles"]["interface"]["candidate_models"]] == [
        "gemma-4-E2B-it",
        "gemma-4-E4B-it",
    ]
    assert manifest["roles"]["interface"]["candidate_models"][0]["priority"] == "default"
    assert manifest["roles"]["interface"]["candidate_models"][1]["priority"] == "comparison"
    assert manifest["roles"]["interface"]["limit_mm_per_prompt"] == {"image": 0, "audio": 1}
    assert manifest["roles"]["interface"]["max_audio_seconds"] == 30.0
    assert manifest["roles"]["interface"]["api_key_env"] == "HERMES_KAME_INTERFACE_API_KEY"
    assert manifest["roles"]["oracle"]["preferred_local_model"] == DEFAULT_SUPER_ORACLE_MODEL
    assert manifest["roles"]["oracle"]["limit_mm_per_prompt"] == {"image": 0, "audio": 0}
    assert manifest["roles"]["asr"]["role"] == "oracle_verbatim_evidence"
    assert manifest["roles"]["asr"]["provider"] == "streaming_stt"
    assert manifest["roles"]["asr"]["adapter"] == "loopback_smoke_bridge"
    assert manifest["roles"]["asr"]["model"] == "oracle-verbatim-asr"
    assert manifest["roles"]["asr"]["module"] == "hermes_cli.realtime_voice_loopback_bridge"
    assert manifest["roles"]["asr"]["protocol_smoke_only"] is True
    assert manifest["roles"]["asr"]["production_replacement"] == "local_streaming_asr"
    assert manifest["roles"]["asr"]["feeds_reflex"] is False
    assert manifest["roles"]["tts"]["provider"] == "streaming_tts"
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
    assert DEFAULT_SUPER_ORACLE_MODEL in compose
    assert "--limit-mm-per-prompt" in compose
    assert '{"audio":1,"image":0}' in compose
    assert '{"audio":0,"image":0}' in compose
    assert "HERMES_VOICE_STREAMING_STT_BASE_URL" in compose
    assert "HERMES_KAME_INTERFACE_BASE_URL: http://kame-interface-vllm:8000/v1" in compose
    assert "HERMES_KAME_INTERFACE_PROVIDER: gemma4" in compose
    assert "HERMES_KAME_INTERFACE_API_KEY: ${HERMES_KAME_INTERFACE_API_KEY:-}" in compose
    assert "HERMES_VOICE_VLLM_BASE_URL: http://kame-interface-vllm:8000/v1" in compose
    assert "HERMES_VOICE_VLLM_TOKEN: ${HERMES_KAME_INTERFACE_API_KEY:-}" in compose
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
    assert "HERMES_DGX_SPARK_ASR_PROVIDER: streaming_stt" in compose
    assert "HERMES_DGX_SPARK_ASR_ADAPTER: loopback_smoke_bridge" in compose
    assert "HERMES_DGX_SPARK_TTS_PROVIDER: streaming_tts" in compose
    assert "HERMES_DGX_SPARK_TTS_ADAPTER: loopback_smoke_bridge" in compose
    assert "hermes_cli.realtime_voice_loopback_bridge" in compose
    assert "interface-secret-token" not in compose
    assert "sk_" not in compose


def test_rendered_compose_wires_production_speech_bridge_upstreams(tmp_path):
    manifest = _manifest(tmp_path, production_speech=True)
    compose = realtime_voice_dgx_spark.render_dgx_spark_compose(manifest)
    env_example = realtime_voice_dgx_spark.render_dgx_spark_env_example(manifest)

    assert manifest["roles"]["asr"]["provider"] == "nvidia_speech"
    assert manifest["roles"]["tts"]["provider"] == "nvidia_speech"
    assert PRODUCTION_ASR_MODULE in compose
    assert PRODUCTION_TTS_MODULE in compose
    assert "HERMES_VOICE_STREAMING_STT_MODEL: nemotron-speech-streaming-0.6b" in compose
    assert "HERMES_VOICE_STREAMING_TTS_MODEL: magpie-local-streaming-tts" in compose
    assert "HERMES_DGX_SPARK_ASR_PROVIDER: nvidia_speech" in compose
    assert "HERMES_DGX_SPARK_TTS_PROVIDER: nvidia_speech" in compose
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
    assert "HERMES_DGX_SPARK_ASR_PROVIDER=nvidia_speech" in env_example
    assert "HERMES_DGX_SPARK_TTS_PROVIDER=nvidia_speech" in env_example
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
        oracle_model=DEFAULT_SUPER_ORACLE_MODEL,
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
        oracle_model=DEFAULT_SUPER_ORACLE_MODEL,
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
    assert manifest["roles"]["interface"]["provider"] == "gemma4"
    assert manifest["roles"]["interface"]["max_audio_seconds"] == 30.0
    assert manifest["roles"]["interface"]["temperature"] == 0.2
    assert manifest["roles"]["interface"]["max_output_tokens"] == 160
    assert manifest["roles"]["interface"]["timeout_seconds"] == 0.8
    assert manifest["engine"]["max_spoken_sentences"] == 2
    assert manifest["engine"]["voice_response_policy"] == "sentence_cap"
    assert manifest["engine"]["fallback_policy"] == "legacy_voice"
    assert manifest["routing"] == {
        "allow_local_greetings": True,
        "allow_local_clarifications": True,
        "require_oracle_for_tools": True,
        "require_oracle_for_memory": True,
        "require_oracle_for_files": True,
        "local_confidence_threshold": 0.75,
    }
    assert manifest["barge_in"] == {
        "min_rms": 350,
        "min_speech_ms": 120,
        "stop_playback_deadline_ms": 150,
    }
    assert manifest["metrics"] == {
        "enabled": True,
        "log_turn_spans": True,
        "log_provider_spans": True,
    }
    assert "HERMES_KAME_INTERFACE_PROVIDER=gemma4" in env_example
    assert "HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS=30.0" in env_example
    assert "HERMES_KAME_INTERFACE_API_KEY_ENV=HERMES_KAME_INTERFACE_API_KEY" in env_example
    assert "HERMES_KAME_INTERFACE_API_KEY=" in env_example
    assert "HERMES_PYTHON=python3" in env_example
    assert "HERMES_KAME_MAX_SPOKEN_SENTENCES=2" in env_example
    assert "HERMES_KAME_VOICE_RESPONSE_POLICY=sentence_cap" in env_example
    assert "HERMES_KAME_FALLBACK_POLICY=legacy_voice" in env_example
    assert "HERMES_KAME_ALLOW_LOCAL_GREETINGS=true" in env_example
    assert "HERMES_KAME_LOCAL_CONFIDENCE_THRESHOLD=0.75" in env_example
    assert "HERMES_KAME_BARGE_IN_MIN_RMS=350" in env_example
    assert "HERMES_KAME_METRICS_ENABLED=true" in env_example
    assert "HERMES_KAME_ORACLE_TIMEOUT_SECONDS=60.0" in env_example
    assert "loopback_smoke_bridge ASR/TTS adapters are protocol-only smoke checks" in env_example
    assert "cannot satisfy VoiceOps local ASR/TTS evidence" in env_example
    assert "HERMES_DGX_SPARK_ASR_PROVIDER=streaming_stt" in env_example
    assert "HERMES_DGX_SPARK_ASR_ADAPTER=loopback_smoke_bridge" in env_example
    assert "HERMES_DGX_SPARK_TTS_PROVIDER=streaming_tts" in env_example
    assert "HERMES_DGX_SPARK_TTS_ADAPTER=loopback_smoke_bridge" in env_example
    assert "HERMES_VOICE_VLLM_MODEL: ${HERMES_KAME_INTERFACE_MODEL:-gemma-4-E2B-it}" in compose
    assert "- ${HERMES_KAME_INTERFACE_MODEL:-gemma-4-E2B-it}" in compose
    assert "HERMES_DGX_SPARK_APPLY_PROFILE" in launch
    assert "hermes_cli.realtime_voice_profile --preset kame --apply" in launch
    assert ': "${HERMES_PYTHON:=python3}"' in launch
    assert ': "${HERMES_KAME_INTERFACE_PROVIDER:=gemma4}"' in launch
    assert ': "${HERMES_KAME_INTERFACE_MODEL:=gemma-4-E2B-it}"' in launch
    assert ': "${HERMES_KAME_INTERFACE_API_KEY_ENV:=HERMES_KAME_INTERFACE_API_KEY}"' in launch
    assert ': "${HERMES_KAME_ORACLE_MODEL:=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"' in launch
    assert ': "${HERMES_VOICE_STREAMING_STT_BASE_URL:=http://spark.local:8767}"' in launch
    assert ': "${HERMES_VOICE_STREAMING_TTS_BASE_URL:=http://spark.local:8768}"' in launch
    assert ': "${HERMES_KAME_VOICE_RESPONSE_POLICY:=sentence_cap}"' in launch
    assert ': "${HERMES_KAME_FALLBACK_POLICY:=legacy_voice}"' in launch
    assert ': "${HERMES_KAME_ALLOW_LOCAL_GREETINGS:=true}"' in launch
    assert ': "${HERMES_KAME_BARGE_IN_MIN_RMS:=350}"' in launch
    assert ': "${HERMES_KAME_METRICS_ENABLED:=true}"' in launch
    assert '--kame-reflex-model "$HERMES_KAME_INTERFACE_MODEL"' in launch
    assert '--kame-interface-provider "$HERMES_KAME_INTERFACE_PROVIDER"' in launch
    assert '--kame-interface-api-key-env "$HERMES_KAME_INTERFACE_API_KEY_ENV"' in launch
    assert "--kame-interface-audio-input native_audio" in launch
    assert '--kame-interface-base-url "$HERMES_KAME_INTERFACE_BASE_URL"' in launch
    assert '--kame-interface-max-audio-seconds "$HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS"' in launch
    assert '--kame-asr-mode "$HERMES_KAME_ASR_MODE"' in launch
    assert '--kame-asr-provider "$HERMES_DGX_SPARK_ASR_PROVIDER"' in launch
    assert '--kame-preferred-local-oracle-model "$HERMES_KAME_ORACLE_MODEL"' in launch
    assert '--kame-oracle-base-url "$HERMES_KAME_ORACLE_BASE_URL"' in launch
    assert '--kame-oracle-provider-name "KAME Local Oracle"' in launch
    assert '--kame-oracle-timeout-seconds "$HERMES_KAME_ORACLE_TIMEOUT_SECONDS"' in launch
    assert '--kame-max-spoken-sentences "$HERMES_KAME_MAX_SPOKEN_SENTENCES"' in launch
    assert '--kame-voice-response-policy "$HERMES_KAME_VOICE_RESPONSE_POLICY"' in launch
    assert '--kame-tts-provider "$HERMES_DGX_SPARK_TTS_PROVIDER"' in launch
    assert '--kame-fallback-policy "$HERMES_KAME_FALLBACK_POLICY"' in launch
    assert '--kame-allow-local-greetings "$HERMES_KAME_ALLOW_LOCAL_GREETINGS"' in launch
    assert '--kame-local-confidence-threshold "$HERMES_KAME_LOCAL_CONFIDENCE_THRESHOLD"' in launch
    assert '--kame-barge-in-min-rms "$HERMES_KAME_BARGE_IN_MIN_RMS"' in launch
    assert '--kame-metrics-enabled "$HERMES_KAME_METRICS_ENABLED"' in launch
    assert '--streaming-stt-base-url "$HERMES_VOICE_STREAMING_STT_BASE_URL"' in launch
    assert '--streaming-stt-model "$HERMES_VOICE_STREAMING_STT_MODEL"' in launch
    assert '--streaming-tts-base-url "$HERMES_VOICE_STREAMING_TTS_BASE_URL"' in launch
    assert '--streaming-tts-model "$HERMES_VOICE_STREAMING_TTS_MODEL"' in launch
    assert "--sidecar-host spark.local" in launch
    assert "--sidecar-port 8765" in launch
    assert "docker compose --env-file .env.example -f compose.yaml up" in launch
    assert "--check" in preflight
    assert "--output-dir \"$SCRIPT_DIR\"" in preflight
    assert ': "${HERMES_PYTHON:=python3}"' in preflight
    assert ': "${HERMES_KAME_INTERFACE_PROVIDER:=gemma4}"' in preflight
    assert "--interface-provider \"$HERMES_KAME_INTERFACE_PROVIDER\"" in preflight
    assert "--interface-model \"$HERMES_KAME_INTERFACE_MODEL\"" in preflight
    assert "--interface-api-key-env \"$HERMES_KAME_INTERFACE_API_KEY_ENV\"" in preflight
    assert "--interface-max-audio-seconds \"$HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS\"" in preflight
    assert "--interface-temperature \"$HERMES_KAME_INTERFACE_TEMPERATURE\"" in preflight
    assert "--oracle-model \"$HERMES_KAME_ORACLE_MODEL\"" in preflight
    assert "--max-spoken-sentences \"$HERMES_KAME_MAX_SPOKEN_SENTENCES\"" in preflight
    assert "--voice-response-policy \"$HERMES_KAME_VOICE_RESPONSE_POLICY\"" in preflight
    assert "--barge-in-min-rms \"$HERMES_KAME_BARGE_IN_MIN_RMS\"" in preflight
    assert "--metrics-enabled \"$HERMES_KAME_METRICS_ENABLED\"" in preflight
    assert ': "${HERMES_KAME_ASR_MODE:=on_escalation}"' in preflight
    assert "--asr-mode \"$HERMES_KAME_ASR_MODE\"" in preflight
    assert "--asr-provider \"$HERMES_DGX_SPARK_ASR_PROVIDER\"" in preflight
    assert "--asr-module \"$HERMES_DGX_SPARK_ASR_MODULE\"" in preflight
    assert "--asr-adapter \"$HERMES_DGX_SPARK_ASR_ADAPTER\"" in preflight
    assert "--tts-module \"$HERMES_DGX_SPARK_TTS_MODULE\"" in preflight
    assert "--tts-provider \"$HERMES_DGX_SPARK_TTS_PROVIDER\"" in preflight
    assert "--tts-adapter \"$HERMES_DGX_SPARK_TTS_ADAPTER\"" in preflight
    assert "--sidecar-base-url http://spark.local:8765" in preflight
    assert "usage: $0 /path/to/benchmark-evidence.json" in validate_benchmark
    assert "--benchmark-evidence \"$1\"" in validate_benchmark
    assert ': "${HERMES_PYTHON:=python3}"' in validate_benchmark
    assert ': "${HERMES_KAME_INTERFACE_PROVIDER:=gemma4}"' in validate_benchmark
    assert "--interface-provider \"$HERMES_KAME_INTERFACE_PROVIDER\"" in validate_benchmark
    assert "--interface-model \"$HERMES_KAME_INTERFACE_MODEL\"" in validate_benchmark
    assert "--interface-api-key-env \"$HERMES_KAME_INTERFACE_API_KEY_ENV\"" in validate_benchmark
    assert "--interface-max-audio-seconds \"$HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS\"" in validate_benchmark
    assert "--interface-temperature \"$HERMES_KAME_INTERFACE_TEMPERATURE\"" in validate_benchmark
    assert "--oracle-model \"$HERMES_KAME_ORACLE_MODEL\"" in validate_benchmark
    assert "--max-spoken-sentences \"$HERMES_KAME_MAX_SPOKEN_SENTENCES\"" in validate_benchmark
    assert "--voice-response-policy \"$HERMES_KAME_VOICE_RESPONSE_POLICY\"" in validate_benchmark
    assert "--barge-in-min-rms \"$HERMES_KAME_BARGE_IN_MIN_RMS\"" in validate_benchmark
    assert "--metrics-enabled \"$HERMES_KAME_METRICS_ENABLED\"" in validate_benchmark
    assert ': "${HERMES_KAME_ASR_MODE:=on_escalation}"' in validate_benchmark
    assert "--asr-mode \"$HERMES_KAME_ASR_MODE\"" in validate_benchmark
    assert "--asr-provider \"$HERMES_DGX_SPARK_ASR_PROVIDER\"" in validate_benchmark
    assert "--asr-model \"$HERMES_VOICE_STREAMING_STT_MODEL\"" in validate_benchmark
    assert "--tts-provider \"$HERMES_DGX_SPARK_TTS_PROVIDER\"" in validate_benchmark
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
    assert [
        (candidate["role"], candidate["provider"], candidate["model"])
        for candidate in matrix["candidates"]["speech"]
    ] == [
        ("oracle_verbatim_asr", "streaming_stt", "oracle-verbatim-asr"),
        ("tts", "streaming_tts", "local-streaming-tts"),
    ]
    assert [entry["name"] for entry in matrix["candidates"]["comparison"]] == [
        "interface_direct_audio_vs_stt_fallback",
        "oracle_outcome_asr_hypothesis_delta",
    ]
    assert evidence_template[0]["kind"] == "kame_benchmark_result"
    assert evidence_template[0]["schema_version"] == "voiceops.spark_benchmark_evidence.v1"
    assert evidence_template[0]["hardware"] == "1x DGX Spark"
    assert evidence_template[0]["locality"] == "local_spark"
    assert evidence_template[0]["verified"] is False
    assert evidence_template[0]["measured_at"] is None
    assert evidence_template[0]["source_artifact"] is None
    assert "voiceops_projection_notes" in evidence_template[0]
    assert evidence_template[0]["category"] == "interface"
    assert evidence_template[0]["model"] == "gemma-4-E2B-it"
    assert evidence_template[0]["input"] == "direct_audio"
    assert evidence_template[0]["metrics"]["sample_count"] is None
    assert evidence_template[0]["metrics"]["speech_end_to_interface_decision_ms"] is None
    assert evidence_template[0]["metrics"]["speech_end_to_interface_decision_p50_ms"] is None
    assert evidence_template[0]["metrics"]["speech_end_to_interface_decision_p90_ms"] is None
    assert evidence_template[0]["metrics"]["kame_interface_model_request_ms"] is None
    assert evidence_template[0]["metrics"]["speech_end_to_local_first_audio_p90_ms"] is None
    comparison_entries = [
        entry for entry in evidence_template if entry.get("kind") == "kame_comparison_result"
    ]
    assert all(entry["schema_version"] == "voiceops.spark_benchmark_evidence.v1" for entry in comparison_entries)
    assert [entry["name"] for entry in comparison_entries] == [
        "interface_direct_audio_vs_stt_fallback",
        "oracle_outcome_asr_hypothesis_delta",
    ]
    assert comparison_entries[0]["metrics"]["paired_turns"] is None
    assert comparison_entries[1]["metrics"]["paired_cases"] is None
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
    assert smoke_entries[1]["fallback_trigger"] is None
    assert smoke_entries[1]["fallback_mode"] is None
    assert smoke_entries[1]["fallback_reason_visible"] is False
    assert smoke_entries[1]["configured_policy_applied"] is False
    assert smoke_entries[2]["voice_active"] is False
    assert smoke_entries[2]["voice_capability_checks"] is None
    assert smoke_entries[2]["voice_denial_count"] is None
    assert smoke_entries[2]["unsupported_voice_claims"] is None
    assert smoke_entries[3]["trigger_reason"] is None
    assert smoke_entries[3]["playback_active"] is False
    assert smoke_entries[3]["stop_latency_ms"] is None
    assert smoke_entries[3]["interrupted_response_committed"] is True
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
        ("vllm_oracle_text_only_multimodal_limit", "compose_vllm_args", False),
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
    assert [entry["name"] for entry in template if entry.get("kind") == "kame_comparison_result"] == [
        "interface_direct_audio_vs_stt_fallback",
        "oracle_outcome_asr_hypothesis_delta",
    ]
    assert all(
        {
            "schema_version",
            "hardware",
            "locality",
            "verified",
            "measured_at",
            "source_artifact",
            "source_artifact_sha256",
        }
        <= set(entry)
        for entry in template
    )
    all_local_smoke = next(
        entry for entry in template if entry.get("kind") == "kame_smoke_result" and entry.get("name") == "all_local_smoke"
    )
    assert all_local_smoke["oracle_selected_by"] == "Hermes /model"
    assert all_local_smoke["components"] == {
        "reflex": None,
        "oracle": None,
        "asr": None,
        "tts": None,
        "sidecar": None,
    }
    assert {
        (entry["role"], entry["provider"], entry["model"], entry["adapter"], entry["protocol_smoke_only"])
        for entry in template
        if entry.get("category") == "speech"
    } == {
        ("oracle_verbatim_asr", "streaming_stt", "oracle-verbatim-asr", "loopback_smoke_bridge", True),
        ("tts", "streaming_tts", "local-streaming-tts", "loopback_smoke_bridge", True),
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
    assert (
        "comparison:interface_direct_audio_vs_stt_fallback: "
        "missing or invalid metric paired_turns"
    ) in result["issues"]


def test_preflight_checks_openai_models_and_health_urls(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)
    seen_urls: list[str] = []
    seen_auth: dict[str, str | None] = {}

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
        seen_auth[request.full_url] = request.get_header("Authorization")
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
            return _Response({"data": [{"id": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"}]})
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

    monkeypatch.setenv("HERMES_KAME_INTERFACE_API_KEY", "interface-secret-token")
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
    assert seen_auth["http://spark.local:8000/v1/models"] == "Bearer interface-secret-token"
    assert seen_auth["http://spark.local:8000/v1/chat/completions"] == "Bearer interface-secret-token"
    assert seen_auth["http://spark.local:8001/v1/models"] is None
    assert seen_auth["http://spark.local:8765/health"] is None
    assert "interface-secret-token" not in json.dumps(preflight)
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
            return _Response({"data": [{"id": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"}]})
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
            return _Response({"data": [{"id": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"}]})
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
            return _Response({"data": [{"id": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"}]})
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
        _passing_benchmark_evidence_with_source(tmp_path),
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
    assert result["coverage"]["model_assumption:vllm_oracle_text_only_multimodal_limit"] is True
    assert result["coverage"]["model_assumption:oracle_authority"] is True
    assert result["coverage"]["all_local_smoke"] is True
    assert result["coverage"]["cloud_fallback_smoke"] is True
    assert result["coverage"]["capability_honesty_smoke"] is True
    assert result["coverage"]["barge_in_interruption_smoke"] is True
    assert result["coverage"]["voiceops_matrix_projection_ready"] is True


def test_benchmark_evidence_validator_accepts_voiceops_closing_kame_evidence(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = _passing_benchmark_evidence_with_source(tmp_path)

    kame_result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert kame_result["ok"] is True

    evidence_path = tmp_path / "kame-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    voiceops_matrix = build_matrix([evidence_path])
    evaluations = {evaluation["candidate_id"]: evaluation for evaluation in voiceops_matrix["evaluations"]}

    assert evaluations["interpreter-gemma4-e2b"]["status"] == "validated"
    assert evaluations["oracle-nemotron3-super-local"]["status"] == "validated"
    assert evaluations["asr-nemotron-speech"]["status"] == "validated"
    assert evaluations["tts-magpie-local"]["status"] == "validated"
    assert voiceops_matrix["stack_smoke"]["status"] == "validated"
    assert voiceops_matrix["ready_for_one_spark_demo"] is True


def test_benchmark_evidence_validator_rejects_missing_voiceops_projection_provenance(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = _passing_benchmark_evidence()
    evidence[0].pop("source_artifact")
    evidence[0].pop("measured_at")
    last_entry_index = len(evidence) - 1
    evidence[-1].pop("hardware")

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["voiceops_matrix_projection_ready"] is False
    assert "voiceops_projection:0:kame_benchmark_result:missing_source_artifact" in result["issues"]
    assert "voiceops_projection:0:kame_benchmark_result:missing_measured_at" in result["issues"]
    assert f"voiceops_projection:{last_entry_index}:kame_smoke_result:missing_or_invalid_hardware" in result["issues"]


def test_benchmark_evidence_validator_rejects_direct_entries_without_evidence_path(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = _passing_benchmark_evidence()

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["voiceops_matrix_projection_ready"] is False
    assert "voiceops_projection:0:kame_benchmark_result:missing_source_artifact_sha256" in result["issues"]
    assert "voiceops_projection:0:kame_benchmark_result:source_artifact_unverified" in result["issues"]


def test_benchmark_evidence_validator_rejects_invalid_voiceops_projection_timestamp(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = _passing_benchmark_evidence()
    evidence[0]["measured_at"] = "2026-06-29T00:00:00"

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["voiceops_matrix_projection_ready"] is False
    assert "voiceops_projection:0:kame_benchmark_result:invalid_measured_at" in result["issues"]


def test_benchmark_evidence_loader_accepts_wrapper_and_preserves_example_marker(tmp_path):
    evidence_path = tmp_path / "wrapped-evidence.json"
    evidence_path.write_text(
        json.dumps({"example_only": True, "evidence": [_passing_benchmark_evidence()[0]]}),
        encoding="utf-8",
    )

    loaded = realtime_voice_dgx_spark.load_dgx_spark_benchmark_evidence(evidence_path)

    assert len(loaded) == 1
    assert loaded[0]["_evidence_path"] == str(evidence_path)
    assert loaded[0]["example_only"] is True


def test_benchmark_evidence_validator_rejects_loaded_example_wrapper(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence_path = tmp_path / "wrapped-evidence.json"
    (tmp_path / "raw-kame-benchmark.json").write_text(json.dumps({"redacted": True}), encoding="utf-8")
    evidence_path.write_text(
        json.dumps({"example_only": True, "evidence": [_passing_benchmark_evidence()[0]]}),
        encoding="utf-8",
    )
    evidence = realtime_voice_dgx_spark.load_dgx_spark_benchmark_evidence(evidence_path)

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["voiceops_matrix_projection_ready"] is False
    assert "voiceops_projection:0:kame_benchmark_result:example_only_evidence_not_accepted" in result["issues"]


def test_benchmark_evidence_validator_rejects_missing_projection_source_file(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence_path = tmp_path / "kame-evidence.json"
    evidence_path.write_text(json.dumps([_passing_benchmark_evidence()[0]]), encoding="utf-8")
    evidence = realtime_voice_dgx_spark.load_dgx_spark_benchmark_evidence(evidence_path)

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["voiceops_matrix_projection_ready"] is False
    assert "voiceops_projection:0:kame_benchmark_result:source_artifact_not_found" in result["issues"]


def test_benchmark_evidence_validator_rejects_stale_projection_source_and_attestation_hashes(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = _passing_benchmark_evidence_with_source(tmp_path)
    raw_path = tmp_path / "raw-kame-benchmark.json"
    raw_path.write_text(
        json.dumps({"redacted": True, "source": "updated synthetic KAME benchmark fixture"}),
        encoding="utf-8",
    )

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["voiceops_matrix_projection_ready"] is False
    assert "voiceops_projection:0:kame_benchmark_result:source_artifact_sha256_mismatch" in result["issues"]
    assert "voiceops_projection:0:kame_benchmark_result:collector_attestation_redacted_sha256_mismatch" in result["issues"]


def test_benchmark_evidence_validator_rejects_projection_source_identity_mismatch(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = _passing_benchmark_evidence_with_source(tmp_path)
    raw_path = tmp_path / "raw-kame-benchmark.json"
    raw_path.write_text(
        json.dumps({"redacted": True, "source": "wrong projection source", "source_key": "unrelated_result"}),
        encoding="utf-8",
    )
    source_artifact_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    for entry in evidence:
        entry["source_artifact_sha256"] = source_artifact_sha256
        entry["collector_attestation"]["redacted_artifact_sha256"] = source_artifact_sha256

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["voiceops_matrix_projection_ready"] is False
    assert "voiceops_projection:0:kame_benchmark_result:source_artifact_identity_mismatch" in result["issues"]


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
            entry["metrics"]["speech_end_to_interface_decision_p90_ms"] = 520

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["interface_direct_audio_latency"] is False
    assert (
        "interface:gemma-4-E2B-it:direct_audio: "
        "speech_end_to_interface_decision_ms 501 exceeds target 500"
    ) in result["issues"]
    assert (
        "interface:gemma-4-E2B-it:direct_audio: "
        "speech_end_to_interface_decision_p90_ms 520 exceeds target 500"
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
            entry["metrics"]["oracle_request_to_first_audio_p90_ms"] = 3200

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["oracle_simple_first_audio_latency"] is False
    assert "oracle:local: oracle first audio total 3130 exceeds target 3000" in result["issues"]
    assert "oracle:local: oracle_request_to_first_audio_p90_ms 3200 exceeds target 3000" in result["issues"]
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


def test_benchmark_evidence_validator_requires_explicit_fallback_honesty_and_barge_in_details(
    tmp_path,
):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = _passing_benchmark_evidence()
    for entry in evidence:
        if entry.get("name") == "cloud_fallback_smoke":
            entry.pop("fallback_trigger")
            entry["fallback_mode"] = "silent_ignore"
            entry["fallback_reason_visible"] = False
            entry["configured_policy_applied"] = False
        elif entry.get("name") == "capability_honesty_smoke":
            entry["voice_active"] = False
            entry["voice_capability_checks"] = 0
            entry["voice_denial_count"] = 1
            entry["unsupported_voice_claims"] = 1
        elif entry.get("name") == "barge_in_interruption_smoke":
            entry["trigger_reason"] = "decoded_packet"
            entry["playback_active"] = False
            entry["stop_latency_ms"] = 151
            entry["interrupted_response_committed"] = True

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["cloud_fallback_smoke"] is False
    assert result["coverage"]["capability_honesty_smoke"] is False
    assert result["coverage"]["barge_in_interruption_smoke"] is False
    assert "cloud_fallback_smoke: requires recognized fallback_trigger" in result["issues"]
    assert "cloud_fallback_smoke: requires recognized fallback_mode" in result["issues"]
    assert "cloud_fallback_smoke: requires fallback_reason_visible == true" in result["issues"]
    assert "cloud_fallback_smoke: requires configured_policy_applied == true" in result["issues"]
    assert "capability_honesty_smoke: requires voice_active == true" in result["issues"]
    assert "capability_honesty_smoke: requires voice_capability_checks >= 1" in result["issues"]
    assert "capability_honesty_smoke: requires voice_denial_count == 0" in result["issues"]
    assert "capability_honesty_smoke: requires unsupported_voice_claims == 0" in result["issues"]
    assert (
        "barge_in_interruption_smoke: requires trigger_reason == confirmed_user_speech"
        in result["issues"]
    )
    assert "barge_in_interruption_smoke: requires playback_active == true" in result["issues"]
    assert "barge_in_interruption_smoke: stop_latency_ms 151 exceeds target 150" in result["issues"]
    assert (
        "barge_in_interruption_smoke: requires interrupted_response_committed == false"
        in result["issues"]
    )


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


def test_benchmark_evidence_validator_requires_paired_comparison_results(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = [
        entry
        for entry in _passing_benchmark_evidence()
        if entry.get("kind") != "kame_comparison_result"
    ]

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["interface_direct_audio_vs_stt_fallback"] is False
    assert result["coverage"]["oracle_outcomes_with_and_without_asr_hypotheses"] is False
    assert (
        "comparison:interface_direct_audio_vs_stt_fallback: missing paired comparison result"
        in result["issues"]
    )
    assert (
        "comparison:oracle_outcome_asr_hypothesis_delta: missing paired comparison result"
        in result["issues"]
    )


def test_benchmark_evidence_validator_rejects_asr_hypothesis_regression(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path, production_speech=True))
    evidence = _passing_benchmark_evidence()
    for entry in evidence:
        if entry.get("kind") == "kame_comparison_result" and entry.get("name") == "oracle_outcome_asr_hypothesis_delta":
            entry["metrics"]["with_asr_literal_argument_accuracy"] = 0.6
            entry["metrics"]["with_asr_tool_argument_error_rate"] = 0.3

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["oracle_outcomes_with_and_without_asr_hypotheses"] is False
    assert (
        "comparison:oracle_outcome_asr_hypothesis_delta: "
        "with_asr_literal_argument_accuracy 0.6 is below without_asr_literal_argument_accuracy 0.72"
    ) in result["issues"]
    assert (
        "comparison:oracle_outcome_asr_hypothesis_delta: "
        "with_asr_tool_argument_error_rate 0.3 exceeds without_asr_tool_argument_error_rate 0.21"
    ) in result["issues"]


def test_main_validates_benchmark_evidence_file(tmp_path, capsys):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(_passing_benchmark_evidence_with_source(tmp_path)), encoding="utf-8")

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
            "--interface-provider",
            "gemma4",
            "--interface-candidate-model",
            "gemma-4-E4B-it",
            "--oracle-model",
            "custom-local-oracle-model",
            "--asr-provider",
            "nvidia_speech",
            "--tts-provider",
            "cartesia",
            "--check",
        ]
    )
    args.func(args)

    assert args.command == "voice"
    assert args.voice_command == "dgx-spark"
    assert args.output_dir == str(tmp_path / "out")
    assert args.interface_model == "gemma-4-E2B-it"
    assert args.interface_provider == "gemma4"
    assert args.interface_candidate_model == ["gemma-4-E4B-it"]
    assert args.oracle_model == "custom-local-oracle-model"
    assert args.asr_provider == "nvidia_speech"
    assert args.tts_provider == "cartesia"
    assert args.check is True
    assert calls == [args]
