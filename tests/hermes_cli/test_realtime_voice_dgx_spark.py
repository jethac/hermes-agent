import json
import argparse
from pathlib import Path

from hermes_cli import realtime_voice_dgx_spark
from hermes_cli.subcommands.voice import build_voice_parser


def _manifest(tmp_path: Path) -> dict:
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
    return [
        *interface_entries,
        {
            "kind": "kame_benchmark_result",
            "category": "oracle",
            "metrics": {
                "oracle_request_to_accepted_ms": 40,
                "oracle_accepted_to_first_token_ms": 780,
                "oracle_first_token_to_first_audio_ms": 220,
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
            "metrics": {
                "speech_end_to_asr_final_ms": 110,
                "literal_accuracy_names_numbers_code": 0.88,
            },
        },
        {
            "kind": "kame_benchmark_result",
            "category": "speech",
            "role": "tts",
            "metrics": {
                "tts_request_to_first_audio_ms": 160,
                "tts_request_to_audio_end_ms": 620,
            },
        },
        {"kind": "kame_smoke_result", "name": "all_local_smoke", "ok": True},
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
    assert manifest["roles"]["asr"]["default_adapter"] == "loopback_smoke_bridge"
    assert manifest["roles"]["asr"]["production_replacement"] == "local_streaming_asr"
    assert manifest["roles"]["asr"]["feeds_reflex"] is False
    assert manifest["roles"]["tts"]["default_adapter"] == "loopback_smoke_bridge"
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
    assert "HERMES_VOICE_VLLM_BASE_URL: http://kame-interface-vllm:8000/v1" in compose
    assert "HERMES_VOICE_STREAMING_STT_BASE_URL: http://kame-asr-bridge:8767" in compose
    assert "HERMES_VOICE_STREAMING_TTS_BASE_URL: http://kame-tts-bridge:8768" in compose
    assert "      - http://kame-interface-vllm:8000/v1" in compose
    assert "      - http://kame-asr-bridge:8767" in compose
    assert "      - http://kame-tts-bridge:8768" in compose
    assert "HERMES_VOICE_VLLM_BASE_URL: http://spark.local:8000/v1" not in compose
    assert "HERMES_VOICE_STREAMING_STT_BASE_URL: http://spark.local:8767" not in compose
    assert "HERMES_VOICE_STREAMING_TTS_BASE_URL: http://spark.local:8768" not in compose
    assert "oracle-verbatim-asr" in compose
    assert "hermes_cli.realtime_voice_loopback_bridge" in compose
    assert "API_KEY" not in compose
    assert "sk_" not in compose


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
    assert "HERMES_DGX_SPARK_APPLY_PROFILE" in launch
    assert "hermes_cli.realtime_voice_profile --preset kame --apply" in launch
    assert "--kame-interface-audio-input native_audio" in launch
    assert "--kame-interface-max-audio-seconds 30.0" in launch
    assert "--kame-asr-mode on_escalation" in launch
    assert "--kame-oracle-base-url http://spark.local:8001/v1" in launch
    assert '--kame-oracle-provider-name "KAME Local Oracle"' in launch
    assert "--sidecar-host spark.local" in launch
    assert "--sidecar-port 8765" in launch
    assert "docker compose --env-file .env.example -f compose.yaml up" in launch
    assert "--check" in preflight
    assert "--output-dir \"$SCRIPT_DIR\"" in preflight
    assert "--interface-model \"$HERMES_KAME_INTERFACE_MODEL\"" in preflight
    assert "--interface-max-audio-seconds \"$HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS\"" in preflight
    assert "--oracle-model \"$HERMES_KAME_ORACLE_MODEL\"" in preflight
    assert "--sidecar-base-url http://spark.local:8765" in preflight
    assert "usage: $0 /path/to/benchmark-evidence.json" in validate_benchmark
    assert "--benchmark-evidence \"$1\"" in validate_benchmark
    assert "--interface-model \"$HERMES_KAME_INTERFACE_MODEL\"" in validate_benchmark
    assert "--interface-max-audio-seconds \"$HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS\"" in validate_benchmark
    assert "--oracle-model \"$HERMES_KAME_ORACLE_MODEL\"" in validate_benchmark
    assert [
        (candidate["model"], candidate["input"]) for candidate in matrix["candidates"]["interface"]
    ] == [
        ("gemma-4-E2B-it", "direct_audio"),
        ("gemma-4-E2B-it", "stt_fallback"),
        ("gemma-4-E4B-it", "direct_audio"),
        ("gemma-4-E4B-it", "stt_fallback"),
    ]
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

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, template)

    assert result["ok"] is False
    assert (
        "interface:gemma-4-E2B-it:direct_audio: "
        "missing or invalid metric speech_end_to_interface_decision_ms"
    ) in result["issues"]
    assert "all_local_smoke: missing passing smoke result" in result["issues"]


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


def test_benchmark_evidence_validator_accepts_complete_comparison_matrix(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path))

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
    assert result["coverage"]["all_local_smoke"] is True
    assert result["coverage"]["cloud_fallback_smoke"] is True
    assert result["coverage"]["capability_honesty_smoke"] is True
    assert result["coverage"]["barge_in_interruption_smoke"] is True


def test_benchmark_evidence_validator_requires_stt_fallback_and_smoke(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path))
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


def test_benchmark_evidence_validator_enforces_direct_audio_latency_targets(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path))
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
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path))
    evidence = _passing_benchmark_evidence()
    for entry in evidence:
        if entry.get("category") == "oracle":
            entry["metrics"]["oracle_accepted_to_first_token_ms"] = 2800
            entry["metrics"]["oracle_first_token_to_first_audio_ms"] = 250

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert result["coverage"]["oracle_simple_first_audio_latency"] is False
    assert "oracle:local: oracle first audio total 3090 exceeds target 3000" in result["issues"]
    assert (
        "oracle_simple_first_audio_latency: "
        "requires oracle_request_to_accepted_ms, oracle_accepted_to_first_token_ms, "
        "and oracle_first_token_to_first_audio_ms total within configured target"
    ) in result["issues"]


def test_benchmark_evidence_validator_requires_capability_and_interruption_smokes(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path))
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
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path))
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
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path))
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
